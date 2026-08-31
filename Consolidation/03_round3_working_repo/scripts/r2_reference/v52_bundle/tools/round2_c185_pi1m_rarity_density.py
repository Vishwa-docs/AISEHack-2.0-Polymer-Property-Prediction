#!/usr/bin/env python3
"""C185: PI1M-derived unlabeled chemical-space rarity/density residuals.

PI1M is used only as an official unlabeled structure pool.  No PI1M target,
pretrained representation, or external cache is used.  The experiment tests
whether small applicability-domain summaries can improve weak-target parent
residuals without becoming a direct string regressor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
ACTIVE = ("ei", "eea", "nc", "eps")
SEED = 20260804
MORGAN_BITS = 2048
PI1M_SAMPLE = 50000
RESIDUAL_WEIGHT = 0.35


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fp_bits(mol: Chem.Mol) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS)
    bits = np.asarray(list(fp), dtype=np.int8)
    return bits


def selected_pi1m(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["SMILES"])
    values = frame["SMILES"].dropna().astype(str).drop_duplicates().tolist()
    values.sort(key=lambda value: hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest())
    return values[:PI1M_SAMPLE]


def build_density_features(parent: dict[str, Any], pi1m_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    pi_smiles = selected_pi1m(pi1m_path)
    df = np.zeros(MORGAN_BITS, dtype=np.int64)
    valid = 0
    for smiles in pi_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        df += fp_bits(mol)
        valid += 1
    idf = np.log((valid + 1.0) / (df + 1.0)) + 1.0
    official_rows: list[np.ndarray] = []
    valid_official = 0
    for mol in parent["molecules"]:
        bits = fp_bits(mol).astype(bool)
        active = idf[bits]
        if len(active) == 0:
            active = np.array([float(idf.mean())])
        valid_official += 1
        official_rows.append(np.array([
            float(bits.sum()),
            float(active.mean()),
            float(active.max()),
            float(active.sum() / max(1, bits.sum())),
            float(np.mean(active >= np.quantile(idf, 0.75))),
            float(np.mean(active <= np.quantile(idf, 0.25))),
            float(np.log1p(mol.GetNumHeavyAtoms())),
            float(mol.GetNumAtoms()),
        ], dtype=np.float64))
    features = np.vstack(official_rows)
    return features, {
        "pi1m_source": str(pi1m_path),
        "pi1m_unique_rows": int(len(values_from_csv(pi1m_path))),
        "pi1m_selected_rows": int(len(pi_smiles)),
        "pi1m_valid_molecules": int(valid),
        "official_feature_rows": int(valid_official),
        "feature_names": ["morgan_bit_count", "mean_bit_idf", "max_bit_idf", "sum_bit_idf_per_bit", "rare_bit_fraction", "common_bit_fraction", "log_heavy_atoms", "atom_count"],
        "feature_shape": [int(x) for x in features.shape],
        "idf_fit_labels": False,
        "pretrained": False,
    }


def values_from_csv(path: Path) -> list[str]:
    return pd.read_csv(path, usecols=["SMILES"])["SMILES"].dropna().astype(str).drop_duplicates().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    pi1m_path = data_dir / "PI1M.csv"
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    X, feature_report = build_density_features(parent, pi1m_path)
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        if target not in ACTIVE:
            target_reports[target] = {"active": False, "parent_r2": float(r2_score(info["y"], info["parent"])), "pass": False}
            oof_parts.append(pd.DataFrame({"target": info["y"], "parent": info["parent"], "candidate": info["parent"], "assembled": info["parent"], "target_type": target}))
            continue
        indices = np.asarray(info["indices"], dtype=np.int64)
        y = np.asarray(info["y"], dtype=float)
        parent_pred = np.asarray(info["parent"], dtype=float)
        groups = np.asarray(info["groups"], dtype=object)
        folds = carrier.grouped_folds(groups)
        residual_oof = np.full(len(y), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(carrier.N_FOLDS):
            va = np.flatnonzero(folds == fold)
            tr = np.flatnonzero(folds != fold)
            # Fixed small model family: the Ridge arm tests smooth applicability
            # signal; ExtraTrees tests bounded nonlinear interactions.
            model = Ridge(alpha=20.0) if target in ("ei", "eea") else ExtraTreesRegressor(n_estimators=300, max_features=0.8, min_samples_leaf=8, random_state=SEED + fold, n_jobs=4)
            model.fit(X[indices[tr]], y[tr] - parent_pred[tr])
            residual_oof[va] = model.predict(X[indices[va]])
            fold_parent = r2_score(y[va], parent_pred[va])
            fold_candidate = parent_pred[va] + RESIDUAL_WEIGHT * residual_oof[va]
            fold_rows.append({"fold": fold, "parent_r2": float(fold_parent), "candidate_r2": float(r2_score(y[va], fold_candidate)), "delta_r2": float(r2_score(y[va], fold_candidate) - fold_parent), "rows": int(len(va))})
        candidate = parent_pred + RESIDUAL_WEIGHT * residual_oof
        delta = float(r2_score(y, candidate) - r2_score(y, parent_pred))
        lower = float(carrier.bootstrap_lower(y, parent_pred, candidate, groups))
        positive = int(sum(row["delta_r2"] > 0 for row in fold_rows))
        passed = bool(delta >= 0.005 and positive >= 4 and lower > 0.0)
        target_reports[target] = {"active": True, "parent_r2": float(r2_score(y, parent_pred)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "pass": passed, "folds": fold_rows}
        oof_parts.append(pd.DataFrame({"target": y, "parent": parent_pred, "candidate": candidate, "assembled": candidate if passed else parent_pred, "target_type": target, "group": groups, "fold": folds}))
        test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
        model = Ridge(alpha=20.0) if target in ("ei", "eea") else ExtraTreesRegressor(n_estimators=300, max_features=0.8, min_samples_leaf=8, random_state=SEED, n_jobs=4)
        model.fit(X[indices], y - parent_pred)
        test_parts.append(pd.DataFrame({"id": test_rows["id"].astype(int), "target_type": target, "candidate": parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target, "target"].to_numpy(float) + RESIDUAL_WEIGHT * model.predict(X[test_indices])}))
    banked = [target for target in ACTIVE if target_reports[target].get("pass")]
    oof = pd.concat(oof_parts, ignore_index=True)
    parent_mean = float(np.mean([r2_score(part["target"], part["parent"]) for part in oof_parts]))
    candidate_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    direct = pd.concat(test_parts, ignore_index=True)
    predictions = parent_test.merge(direct, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["candidate"], predictions["target"])
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C185 complete output contract failed")
    report = {"schema_version": "ppp.round2.c185.pi1m-rarity-density.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "official_inputs": parent["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "kaggle_compute": False, "kaggle_upload": False, "parent_replay_parity": parity, "feature_report": feature_report, "target_reports": target_reports, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": candidate_mean, "mean_gain": candidate_mean - parent_mean, "complete_output_rows": int(len(predictions)), "complete_output_order_pass": True, "full_candidate_gate_pass": bool(banked and candidate_mean - parent_mean >= 0.002), "decision": "candidate_pass_pending_clean_reproduction" if banked and candidate_mean - parent_mean >= 0.002 else "rejected_component_or_full_gate", "elapsed_seconds": float(time.time() - started), "source_hashes": {"runner": sha256_file(Path(__file__)), "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"), "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py")}}
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    direct.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c185.pi1m-rarity-density.v1", "seed": SEED, "morgan_bits": MORGAN_BITS, "pi1m_sample": PI1M_SAMPLE, "residual_weight": RESIDUAL_WEIGHT, "active_targets": list(ACTIVE), "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{parent_mean:.12f}`; assembled `{candidate_mean:.12f}`; gain `{candidate_mean - parent_mean:+.12f}`. Official-only; PI1M used unlabeled for fixed density features; no local_eval read.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.name != "artifact_manifest.sha256" and path.is_file()]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": candidate_mean, "mean_gain": candidate_mean - parent_mean, "decision": report["decision"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
