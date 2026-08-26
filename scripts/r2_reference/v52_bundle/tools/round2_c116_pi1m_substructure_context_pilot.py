#!/usr/bin/env python3
"""C116: bounded official-PI1M substructure-context compression pilot.

PI1M is used only as an unlabeled official corpus. Morgan-count features are
compressed from scratch with TruncatedSVD; no target, external_label file, checkpoint,
or precomputed embedding is read. The supervised part is a fixed residual
Ridge comparison against a same-budget official-structure-only control.
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
ACTIVE = ("eps", "nc", "ei", "eea")
PI1M_SHA256 = "c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8"
PI1M_ROWS = 100_000
BITS = 1024
COMPONENTS = 24
ALPHA = 100.0
RESIDUAL_WEIGHT = 0.15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def morgan_count_matrix(smiles: list[str], generator: Any) -> tuple[sparse.csr_matrix, int]:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    invalid = 0
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text).replace("[*]", "*"))
        if molecule is None:
            invalid += 1
            continue
        elements = generator.GetCountFingerprint(molecule).GetNonzeroElements()
        for column, value in elements.items():
            rows.append(row)
            cols.append(int(column))
            values.append(float(value))
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(smiles), BITS), dtype=np.float64)
    return matrix, invalid


def fit_embedding(matrix: sparse.csr_matrix, seed: int) -> TruncatedSVD:
    model = TruncatedSVD(n_components=COMPONENTS, n_iter=3, random_state=seed)
    model.fit(matrix)
    return model


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else 0.0


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> tuple[dict[str, Any], float]:
    result: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        rows = int(np.sum(selected))
        item: dict[str, Any] = {"rows": rows, "status": "inapplicable", "delta_r2": 0.0}
        if rows >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"status": "evaluable", "delta_r2": delta})
            deltas.append(delta)
        result[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        add(name, selected)
    for value in sorted(set(scaffolds)):
        add(f"scaffold_{value}", scaffolds == value, minimum=10)
    return result, float(min(deltas)) if deltas else 0.0


def fold_map(length: int, seed: int = 2026) -> np.ndarray:
    folds = np.full(length, -1, dtype=np.int64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(length))):
        folds[validation] = fold
    return folds


def run_replica(
    seed: int,
    parent_oof: pd.DataFrame,
    parent_predictions: pd.DataFrame,
    test: pd.DataFrame,
    keys: list[str],
    pi_features: np.ndarray,
    official_features: np.ndarray,
    fingerprints: list[Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    key_to_index = {key: index for index, key in enumerate(keys)}
    reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        rows = parent_oof[parent_oof["target_type"] == target].reset_index(drop=True)
        y = rows["target"].to_numpy(float)
        parent = rows["parent_prediction"].to_numpy(float)
        canonical = rows["canonical"].astype(str).to_numpy(object)
        indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        folds = fold_map(len(rows))
        candidate = parent.copy()
        control = parent.copy()
        similarity = np.full(len(rows), np.nan, dtype=np.float64)
        if target in ACTIVE:
            residual = y - parent
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                candidate_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
                control_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
                candidate_model.fit(pi_features[indices[training]], residual[training])
                control_model.fit(official_features[indices[training]], residual[training])
                candidate[validation] = parent[validation] + RESIDUAL_WEIGHT * candidate_model.predict(pi_features[indices[validation]])
                control[validation] = parent[validation] + RESIDUAL_WEIGHT * control_model.predict(official_features[indices[validation]])
                train_fps = [fingerprints[int(index)] for index in indices[training]]
                similarity[validation] = np.asarray([
                    max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps))
                    for index in indices[validation]
                ], dtype=np.float64)
        parent_r2 = float(r2_score(y, parent))
        candidate_r2 = float(r2_score(y, candidate))
        control_r2 = float(r2_score(y, control))
        folds_report = []
        for fold in range(5):
            selected = folds == fold
            folds_report.append({
                "fold": fold,
                "rows": int(np.sum(selected)),
                "parent_r2": float(r2_score(y[selected], parent[selected])),
                "candidate_r2": float(r2_score(y[selected], candidate[selected])),
                "control_r2": float(r2_score(y[selected], control[selected])),
                "candidate_delta_r2": float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected])),
            })
        groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
        lower = bootstrap_lower(y, parent, candidate, groups) if target in ACTIVE else 0.0
        panels, minimum_panel = panel_report(y, parent, candidate, similarity, scaffolds) if target in ACTIVE else ({"unchanged_parent": {"rows": int(len(y)), "status": "unchanged", "delta_r2": 0.0}}, 0.0)
        candidate_delta = candidate_r2 - parent_r2
        control_delta = control_r2 - parent_r2
        gates = {
            "gain_pass": target not in ACTIVE or candidate_delta >= 0.01,
            "fold_pass": target not in ACTIVE or sum(row["candidate_delta_r2"] > 0 for row in folds_report) >= 4,
            "bootstrap_pass": target not in ACTIVE or lower > 0.0,
            "panel_pass": target not in ACTIVE or minimum_panel >= 0.0,
            "strict_no_regression": candidate_delta >= -0.003,
            "pi1m_beats_official_control": target not in ACTIVE or candidate_delta >= control_delta,
        }
        reports[target] = {
            "parent_r2": parent_r2,
            "candidate_r2": candidate_r2,
            "control_r2": control_r2,
            "candidate_delta_r2": float(candidate_delta),
            "control_delta_r2": float(control_delta),
            "pi1m_minus_control": float(candidate_delta - control_delta),
            "positive_folds": int(sum(row["candidate_delta_r2"] > 0 for row in folds_report)),
            "group_bootstrap_lower": float(lower),
            "minimum_panel_delta": float(minimum_panel),
            "folds": folds_report,
            "panels": panels,
            "gates": gates,
            "pass": bool(all(gates.values())),
            "unchanged_parent": target not in ACTIVE,
        }
        oof_parts.append(pd.DataFrame({
            "canonical": canonical,
            "target_type": target,
            "target": y,
            "parent_prediction": parent,
            "pi1m_candidate": candidate,
            "official_control": control,
            "outer_fold": folds,
            "nearest_similarity": similarity,
        }))
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS]))
    gate_pass = bool(
        mean_candidate - mean_parent >= 0.002
        and min(reports[target]["candidate_delta_r2"] for target in TARGETS) >= -0.003
        and all(reports[target]["pass"] for target in ACTIVE)
    )
    return {
        "seed": seed,
        "target_reports": reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "complete_clean_gate_pass": gate_pass,
        "complete_output_rows": 0,
    }, pd.concat(oof_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("C116 requires a fresh protocol-only run directory")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    pi1m_path = data_dir / "PI1M.csv"
    pi1m_hash = sha256_file(pi1m_path)
    if pi1m_hash != PI1M_SHA256:
        raise RuntimeError(f"PI1M hash mismatch: {pi1m_hash}")
    pi1m_frame = pd.read_csv(pi1m_path, usecols=["SMILES"], nrows=PI1M_ROWS)
    if list(pi1m_frame.columns) != ["SMILES"] or pi1m_frame["SMILES"].isna().any() or pi1m_frame["SMILES"].nunique() != PI1M_ROWS:
        raise RuntimeError("PI1M schema/null/uniqueness preflight failed")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=BITS)
    smoke_start = time.time()
    _, smoke_invalid = morgan_count_matrix(pi1m_frame["SMILES"].astype(str).head(5000).tolist(), generator)
    smoke_seconds = time.time() - smoke_start
    # Invalid unlabeled corpus rows are retained in the audit count and receive
    # an all-zero sparse fingerprint row; they never contribute a target value.
    parent_predictions, parent_oof, context = parent_control.rebuild_parent(root, data_dir, run_dir)
    canonical_dir = root / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7"
    canonical_predictions = pd.read_csv(canonical_dir / "predictions.csv")
    canonical_oof = pd.read_csv(canonical_dir / "oof_predictions.csv")
    parent_test_delta = float(np.max(np.abs(parent_predictions["target"].to_numpy(float) - canonical_predictions["target"].to_numpy(float))))
    left = parent_oof.sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
    right = canonical_oof.sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
    if len(left) != len(right):
        raise RuntimeError("C050 parent OOF row count mismatch")
    parent_oof_delta = float(np.max(np.abs(left["parent_prediction"].to_numpy(float) - right["candidate_prediction"].to_numpy(float))))
    if parent_test_delta > 1.0e-12 or parent_oof_delta > 1.0e-12:
        raise RuntimeError(f"C050 parent parity failed: oof={parent_oof_delta} test={parent_test_delta}")
    train, test, archive, inputs = context["train"], context["test"], context["archive"], context["inputs"]
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    pi_matrix, pi_invalid = morgan_count_matrix(pi1m_frame["SMILES"].astype(str).tolist(), generator)
    official_matrix, official_invalid = morgan_count_matrix(keys, generator)
    pi_svd = fit_embedding(pi_matrix, 2026)
    official_svd = fit_embedding(official_matrix, 2026)
    pi_features = pi_svd.transform(official_matrix).astype(np.float64, copy=False)
    official_features = official_svd.transform(official_matrix).astype(np.float64, copy=False)
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    replica_reports = []
    replica_oof = []
    for seed in (2026, 2027):
        report, oof = run_replica(seed, parent_oof, parent_predictions, test, keys, pi_features, official_features, fingerprints)
        replica_reports.append(report)
        replica_oof.append(oof)
    primary = replica_reports[0]
    audit = {
        "schema_version": "ppp.round2.c116.pi1m-substructure-context-pilot.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "local_eval_read": False,
        "external_label_file_read": False,
        "kaggle_compute": False,
        "official_inputs": inputs,
        "pi1m_sha256": pi1m_hash,
        "pi1m_rows_available": 995799,
        "pi1m_rows_used": PI1M_ROWS,
        "pi1m_unique_rows_used": PI1M_ROWS,
        "pi1m_invalid_rows": int(pi_invalid),
        "official_invalid_rows": int(official_invalid),
        "smoke_seconds": float(smoke_seconds),
        "parent_replay_oof_max_abs": parent_oof_delta,
        "parent_replay_test_max_abs": parent_test_delta,
        "replicas": replica_reports,
        "decision": "clean_gate_pass_pending_test_fit" if all(item["complete_clean_gate_pass"] for item in replica_reports) else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame([{"target": target, "parent_r2": primary["target_reports"][target]["parent_r2"], "candidate_r2": primary["target_reports"][target]["candidate_r2"], "control_r2": primary["target_reports"][target]["control_r2"], "candidate_delta_r2": primary["target_reports"][target]["candidate_delta_r2"], "control_delta_r2": primary["target_reports"][target]["control_delta_r2"]} for target in TARGETS]).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(replica_oof, keys=["replica_2026", "replica_2027"], names=["replica", "row"]).reset_index().to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c116.pi1m-substructure-context-pilot.v1", "seed": 2026, "replica_seeds": [2026, 2027], "pi1m_sha256": pi1m_hash, "pi1m_rows_used": PI1M_ROWS, "morgan_radius": 2, "morgan_bits": BITS, "svd_components": COMPONENTS, "svd_iterations": 3, "ridge_alpha": ALPHA, "residual_weight": RESIDUAL_WEIGHT, "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"scikit_learn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{audit['decision']}**. No full-data test fit or local_eval action was performed before the clean gates.\n", encoding="utf-8")
    source_paths = [Path(__file__), root / "tools/round2_c112_c050_parent_parity_control.py", root / "tools/initial_reference_pipeline.py"]
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{sha256_file(path)}  SOURCE {path.relative_to(root)}" for path in source_paths)
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "mean_parent_r2": primary["mean_parent_r2"], "mean_candidate_r2": primary["mean_candidate_r2"], "mean_gain": primary["mean_gain"], "elapsed_seconds": audit["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
