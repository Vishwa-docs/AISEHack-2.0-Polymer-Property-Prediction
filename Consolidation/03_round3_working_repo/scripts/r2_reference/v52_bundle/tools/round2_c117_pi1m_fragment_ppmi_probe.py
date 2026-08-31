#!/usr/bin/env python3
"""C117: bounded PI1M fragment-context PPMI probe.

The representation is learned from unlabeled official SMILES only.  The
supervised transfer head is a direct fold-local model blended with an already
OOF C050 baseline, so it does not train on global parent residuals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c112_c050_parent_parity_control as parent_control
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("eps", "nc", "ei")
PI1M_SHA256 = "c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8"
SAMPLE_SIZE = 50_000
BUCKETS = 4096
RANK = 64
BLEND = 0.10
ALPHA = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def canonical(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(smiles).replace("[*]", "*"))
    return None if molecule is None else Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def token_matrix(smiles: list[str], generator: Any) -> tuple[sparse.csr_matrix, int, list[str | None]]:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    invalid = 0
    canonicals: list[str | None] = []
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text).replace("[*]", "*"))
        if molecule is None:
            invalid += 1
            canonicals.append(None)
            continue
        canonicals.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
        elements = generator.GetCountFingerprint(molecule).GetNonzeroElements()
        for token, value in elements.items():
            rows.append(row)
            cols.append(int(token) % BUCKETS)
            values.append(float(value))
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(smiles), BUCKETS), dtype=np.float64)
    return matrix, invalid, canonicals


def ppm_embedding(documents: sparse.csr_matrix, seed: int = 2026) -> np.ndarray:
    cooccurrence = (documents.T @ documents).toarray().astype(np.float64, copy=False)
    total = float(np.sum(cooccurrence))
    row_sum = np.sum(cooccurrence, axis=1)
    col_sum = np.sum(cooccurrence, axis=0)
    expected = np.outer(row_sum, col_sum)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(np.maximum(cooccurrence, 1.0) * total / np.maximum(expected, 1.0))
    pmi[cooccurrence <= 0.0] = 0.0
    pmi[pmi < 0.0] = 0.0
    model = TruncatedSVD(n_components=RANK, n_iter=3, random_state=seed)
    model.fit(sparse.csr_matrix(pmi))
    return model.transform(sparse.eye(BUCKETS, format="csr")).astype(np.float64, copy=False)


def molecule_features(documents: sparse.csr_matrix, token_embedding: np.ndarray) -> np.ndarray:
    weighted = documents @ token_embedding
    counts = np.asarray(documents.sum(axis=1)).ravel()
    counts = np.maximum(counts, 1.0)
    mean = np.asarray(weighted) / counts[:, None]
    # The token embedding is fixed per molecule; a second-order context
    # magnitude is a cheap and deterministic companion to the mean.
    second = np.asarray(documents @ (token_embedding * token_embedding)) / counts[:, None]
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return np.hstack([mean, std]).astype(np.float64, copy=False)


def select_pi1m(frame: pd.DataFrame) -> tuple[list[str], list[str], dict[str, Any]]:
    ranked = sorted(
        ((hashlib.sha256(f"2026|{value}".encode("utf-8")).hexdigest(), str(value)) for value in frame["SMILES"].astype(str)),
        key=lambda pair: pair[0],
    )
    selected: list[str] = []
    selected_hashes: list[str] = []
    seen_raw: set[str] = set()
    invalid = 0
    for digest, value in ranked:
        if value in seen_raw:
            continue
        seen_raw.add(value)
        if canonical(value) is None:
            invalid += 1
            continue
        selected.append(value)
        selected_hashes.append(digest)
        if len(selected) == SAMPLE_SIZE:
            break
    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError(f"only {len(selected)} valid distinct PI1M rows available")
    return selected, selected_hashes, {"ranked_rows": len(ranked), "invalid_before_sample": invalid, "selected_rows": len(selected), "distinct_selected": len(set(selected))}


def folds_for(length: int) -> np.ndarray:
    folds = np.full(length, -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(KFold(n_splits=5, shuffle=True, random_state=2026).split(np.arange(length))):
        folds[validation] = fold
    return folds


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {value: np.flatnonzero(groups == value) for value in unique}
    rng = np.random.default_rng(2026)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[value] for value in selected])
        values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray, decontaminate: np.ndarray) -> tuple[float, dict[str, Any]]:
    deltas: list[float] = []
    report: dict[str, Any] = {}

    def add(name: str, mask: np.ndarray, minimum: int = 5) -> None:
        rows = int(np.sum(mask))
        if rows < minimum or np.var(y[mask]) <= 1.0e-15:
            report[name] = {"rows": rows, "status": "inapplicable", "delta_r2": 0.0}
            return
        delta = float(r2_score(y[mask], candidate[mask]) - r2_score(y[mask], parent[mask]))
        report[name] = {"rows": rows, "status": "evaluable", "delta_r2": delta}
        deltas.append(delta)

    add("all_rows", np.ones(len(y), dtype=bool))
    add("decontaminated_overlap", decontaminate)
    add("similarity_lt_0.30", similarity < 0.30)
    add("similarity_0.30_0.50", (similarity >= 0.30) & (similarity < 0.50))
    add("similarity_0.50_0.70", (similarity >= 0.50) & (similarity < 0.70))
    add("similarity_ge_0.70", similarity >= 0.70)
    for value in sorted(set(scaffolds)):
        add(f"scaffold_{value}", scaffolds == value, minimum=10)
    return (min(deltas) if deltas else 0.0), report


def evaluate_target(target: str, parent_oof: pd.DataFrame, keys: list[str], pi_features: np.ndarray, official_features: np.ndarray, fingerprints: list[Any], pi_canonicals: set[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = parent_oof[parent_oof["target_type"] == target].reset_index(drop=True)
    y = rows["target"].to_numpy(float)
    parent = rows["parent_prediction"].to_numpy(float)
    canonical_values = rows["canonical"].astype(str).to_numpy(object)
    key_to_index = {key: index for index, key in enumerate(keys)}
    indices = np.asarray([key_to_index[value] for value in canonical_values], dtype=np.int64)
    folds = folds_for(len(rows))
    candidate = parent.copy()
    control = parent.copy()
    similarity = np.empty(len(rows), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        pi_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(pi_features[indices[training]], y[training])
        official_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(official_features[indices[training]], y[training])
        candidate[validation] = parent[validation] + BLEND * (pi_model.predict(pi_features[indices[validation]]) - parent[validation])
        control[validation] = parent[validation] + BLEND * (official_model.predict(official_features[indices[validation]]) - parent[validation])
        train_fps = [fingerprints[int(index)] for index in indices[training]]
        similarity[validation] = np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps)) for index in indices[validation]], dtype=np.float64)
    groups = np.asarray([plumbing.no_stereo(value) for value in canonical_values], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical_values], dtype=object)
    decontaminate = np.asarray([value not in pi_canonicals for value in canonical_values], dtype=bool)
    parent_r2 = float(r2_score(y, parent))
    candidate_r2 = float(r2_score(y, candidate))
    control_r2 = float(r2_score(y, control))
    fold_rows = []
    for fold in range(5):
        mask = folds == fold
        fold_rows.append({"fold": fold, "rows": int(np.sum(mask)), "delta_r2": float(r2_score(y[mask], candidate[mask]) - r2_score(y[mask], parent[mask]))})
    minimum_panel, panels = panel_delta(y, parent, candidate, similarity, scaffolds, decontaminate)
    delta = candidate_r2 - parent_r2
    control_delta = control_r2 - parent_r2
    gates = {
        "gain_pass": delta >= 0.01,
        "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4,
        "bootstrap_pass": bootstrap_lower(y, parent, candidate, groups) > 0.0,
        "panel_pass": minimum_panel >= 0.0,
        "control_pass": delta - control_delta >= 0.003,
        "strict_no_regression": delta >= -0.003,
    }
    report = {
        "parent_r2": parent_r2,
        "candidate_r2": candidate_r2,
        "control_r2": control_r2,
        "candidate_delta_r2": float(delta),
        "control_delta_r2": float(control_delta),
        "pi1m_minus_control": float(delta - control_delta),
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "group_bootstrap_lower": float(bootstrap_lower(y, parent, candidate, groups)),
        "minimum_panel_delta": float(minimum_panel),
        "folds": fold_rows,
        "panels": panels,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    oof = pd.DataFrame({"canonical": canonical_values, "target_type": target, "target": y, "parent_prediction": parent, "candidate_prediction": candidate, "control_prediction": control, "outer_fold": folds, "nearest_similarity": similarity})
    return report, oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("C117 requires a fresh protocol-only run directory")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    pi1m_path = data_dir / "PI1M.csv"
    pi1m_hash = sha256_file(pi1m_path)
    if pi1m_hash != PI1M_SHA256:
        raise RuntimeError(f"PI1M hash mismatch: {pi1m_hash}")
    raw_pi1m = pd.read_csv(pi1m_path, usecols=["SMILES"])
    selected, selected_hashes, sample_report = select_pi1m(raw_pi1m)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=BUCKETS)
    smoke_matrix, smoke_invalid, _ = token_matrix(selected[:5000], generator)
    if smoke_invalid / 5000.0 > 0.005 or smoke_matrix.nnz == 0:
        raise RuntimeError(f"PI1M fragment smoke failed: invalid={smoke_invalid}, nnz={smoke_matrix.nnz}")
    parent_predictions, parent_oof, context = parent_control.rebuild_parent(root, data_dir, run_dir)
    canonical_dir = root / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7"
    canonical_predictions = pd.read_csv(canonical_dir / "predictions.csv")
    canonical_oof = pd.read_csv(canonical_dir / "oof_predictions.csv")
    test_delta = float(np.max(np.abs(parent_predictions["target"].to_numpy(float) - canonical_predictions["target"].to_numpy(float))))
    replay_left = parent_oof.sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
    replay_right = canonical_oof.sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
    oof_delta = float(np.max(np.abs(replay_left["parent_prediction"].to_numpy(float) - replay_right["candidate_prediction"].to_numpy(float))))
    if test_delta > 1.0e-12 or oof_delta > 1.0e-12:
        raise RuntimeError(f"C050 parity failed: oof={oof_delta}, test={test_delta}")
    train, test, archive, inputs = context["train"], context["test"], context["archive"], context["inputs"]
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    official_union = pd.concat([train["smiles"], test["smiles"], archive["smiles"]], ignore_index=True).astype(str).tolist()
    control_corpus = [official_union[index % len(official_union)] for index in range(SAMPLE_SIZE)]
    pi_matrix, pi_invalid, pi_canonicals_list = token_matrix(selected, generator)
    control_matrix, control_invalid, _ = token_matrix(control_corpus, generator)
    pi_embedding = ppm_embedding(pi_matrix)
    control_embedding = ppm_embedding(control_matrix)
    official_matrix, official_invalid, _ = token_matrix(keys, generator)
    pi_features = molecule_features(official_matrix, pi_embedding)
    control_features = molecule_features(official_matrix, control_embedding)
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    pi_canonicals = {value for value in pi_canonicals_list if value is not None}
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in ACTIVE:
        report, oof = evaluate_target(target, parent_oof, keys, pi_features, control_features, fingerprints, pi_canonicals)
        target_reports[target] = report
        oof_parts.append(oof)
    for target in TARGETS:
        if target not in ACTIVE:
            rows = parent_oof[parent_oof["target_type"] == target].copy()
            rows["candidate_prediction"] = rows["parent_prediction"]
            rows["control_prediction"] = rows["parent_prediction"]
            oof_parts.append(rows[["canonical", "target_type", "target", "parent_prediction", "candidate_prediction", "control_prediction"]])
            target_reports[target] = {"parent_r2": float(r2_score(rows["target"], rows["parent_prediction"])), "candidate_r2": float(r2_score(rows["target"], rows["parent_prediction"])), "control_r2": float(r2_score(rows["target"], rows["parent_prediction"])), "candidate_delta_r2": 0.0, "control_delta_r2": 0.0, "pi1m_minus_control": 0.0, "pass": True, "unchanged_parent": True}
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    clean_pass = bool(mean_candidate - mean_parent >= 0.002 and min(target_reports[target]["candidate_delta_r2"] for target in TARGETS) >= -0.003 and all(target_reports[target]["pass"] for target in ACTIVE))
    audit = {
        "schema_version": "ppp.round2.c117.pi1m-fragment-ppmi-probe.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "local_eval_read": False,
        "external_label_file_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "official_inputs": inputs,
        "pi1m_sha256": pi1m_hash,
        "pi1m_rows_available": int(len(raw_pi1m)),
        "sample_report": sample_report,
        "parse_invalid_selected": int(pi_invalid),
        "parse_invalid_control": int(control_invalid),
        "parse_invalid_official_keys": int(official_invalid),
        "overlap_raw_train": int(sum(value in set(train["smiles"].astype(str)) for value in selected)),
        "overlap_raw_test": int(sum(value in set(test["smiles"].astype(str)) for value in selected)),
        "parent_replay_oof_max_abs": oof_delta,
        "parent_replay_test_max_abs": test_delta,
        "targets": target_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "decision": "clean_gate_pass_pending_test_fit" if clean_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame([{"target": target, "parent_r2": target_reports[target]["parent_r2"], "candidate_r2": target_reports[target]["candidate_r2"], "control_r2": target_reports[target]["control_r2"], "candidate_delta_r2": target_reports[target]["candidate_delta_r2"], "control_delta_r2": target_reports[target]["control_delta_r2"]} for target in TARGETS]).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c117.pi1m-fragment-ppmi-probe.v1", "seed": 2026, "sample_size": SAMPLE_SIZE, "buckets": BUCKETS, "rank": RANK, "ridge_alpha": ALPHA, "blend": BLEND, "pi1m_sha256": pi1m_hash, "official_inputs": inputs})
    (run_dir / "sample_manifest.json").write_text(json.dumps({"sha256": selected_hashes, "rows": len(selected), "raw_smiles_not_tracked": True}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"scikit_learn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{audit['decision']}**. Full-data test fitting and local_eval access were withheld unless all clean gates passed.\n", encoding="utf-8")
    source_paths = [Path(__file__), root / "tools/round2_c112_c050_parent_parity_control.py", root / "tools/initial_reference_pipeline.py"]
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{sha256_file(path)}  SOURCE {path.relative_to(root)}" for path in source_paths)
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
