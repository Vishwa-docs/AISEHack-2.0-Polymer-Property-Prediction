#!/usr/bin/env python3
"""Clean official-only grouped EPS multiview specialist.

This is a bounded component experiment. It never reads test external_labels or local_eval
files. The parent carrier is regenerated inside every outer fold, while the
specialist blend is fitted only from inner grouped folds of that outer train
partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "eps"
PARENT_ID = "R2-C050-20260803-2130-mixed-c001-gap-components-v7"
SEED = 2026
OUTER_FOLDS = 5
INNER_FOLDS = 4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def clean_dense(values: np.ndarray, train: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64).copy()
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(matrix[train]), imputer.transform(matrix[pred])


def normalized_features(descriptor: np.ndarray, descriptor_names: list[str], physical: np.ndarray, physical_names: list[str]) -> tuple[np.ndarray, list[str]]:
    wanted = [
        "MolWt", "HeavyAtomMolWt", "ExactMolWt", "LabuteASA", "TPSA", "FractionCSP3",
        "HeavyAtomCount", "NumAliphaticRings", "NumAromaticRings", "NumRotatableBonds",
        "NumSaturatedRings", "RingCount", "MolLogP", "MolMR", "NumHAcceptors", "NumHDonors",
    ]
    dpos = {name: i for i, name in enumerate(descriptor_names)}
    ppos = {name: i for i, name in enumerate(physical_names)}
    cols = [descriptor[:, dpos[name]] for name in wanted]
    names = [f"rdkit_{name}" for name in wanted]
    physical_wanted = [
        "smiles_length", "atom_count", "heavy_atom_count", "ring_count", "aromatic_atom_count",
        "hetero_atom_count", "halogen_count", "rotatable_bonds_approx", "double_bond_count",
        "branch_count", "n_count", "o_count", "s_count", "si_count",
    ]
    cols.extend(physical[:, ppos[name]] for name in physical_wanted)
    names.extend(f"physical_{name}" for name in physical_wanted)
    molwt = descriptor[:, dpos["MolWt"]]
    mr = descriptor[:, dpos["MolMR"]]
    asa = descriptor[:, dpos["LabuteASA"]]
    tpsa = descriptor[:, dpos["TPSA"]]
    heavy = physical[:, ppos["heavy_atom_count"]]
    atoms = physical[:, ppos["atom_count"]]
    rings = physical[:, ppos["ring_count"]]
    aromatic = physical[:, ppos["aromatic_atom_count"]]
    hetero = physical[:, ppos["hetero_atom_count"]]
    length = physical[:, ppos["smiles_length"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        derived = [
            mr / np.maximum(molwt, 1e-9), mr / np.maximum(asa, 1e-9), tpsa / np.maximum(molwt, 1e-9),
            tpsa / np.maximum(heavy, 1e-9), molwt / np.maximum(heavy, 1e-9),
            rings / np.maximum(atoms, 1e-9), aromatic / np.maximum(atoms, 1e-9),
            hetero / np.maximum(atoms, 1e-9), heavy / np.maximum(length, 1e-9),
        ]
    cols.extend(derived)
    names.extend(["mr_per_molwt", "mr_per_asa", "tpsa_per_molwt", "tpsa_per_heavy", "molwt_per_heavy", "ring_density", "aromatic_fraction", "hetero_fraction", "heavy_per_smiles_length"])
    return np.column_stack(cols).astype(np.float64, copy=False), names


def fit_parent(
    dense_parent: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    fingerprints: list[Any],
    y_global: np.ndarray,
    target_indices: np.ndarray,
    train_local: np.ndarray,
    pred_local: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
    pred_global: np.ndarray | None = None,
) -> np.ndarray:
    train_global = target_indices[train_local]
    prediction_global = target_indices[pred_local] if pred_global is None else pred_global
    base = reference.predict_base_models(
        dense_parent, sparse_parts, fingerprints, y_global,
        train_global, prediction_global, config, TARGET,
    )
    return base @ weights


def fit_arms(
    features: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    y: np.ndarray,
    target_indices: np.ndarray,
    train_local: np.ndarray,
    pred_local: np.ndarray,
    pred_global: np.ndarray | None = None,
) -> np.ndarray:
    train_global = target_indices[train_local]
    prediction_global = target_indices[pred_local] if pred_global is None else pred_global
    x_train, x_pred = clean_dense(features, train_global, prediction_global)
    extra = ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2, max_features=0.60, random_state=SEED, n_jobs=2)
    extra.fit(x_train, y[train_local])
    hgb = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=0.20, random_state=SEED)
    hgb.fit(x_train, y[train_local])
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_pred = scaler.transform(x_pred)
    sparse_train = sparse.hstack([part[train_global] for part in sparse_parts] + [sparse.csr_matrix(scaled_train)], format="csr")
    sparse_pred = sparse.hstack([part[prediction_global] for part in sparse_parts] + [sparse.csr_matrix(scaled_pred)], format="csr")
    ridge = Ridge(alpha=30.0, solver="lsqr", max_iter=5000, tol=1e-4)
    ridge.fit(sparse_train, y[train_local])
    return np.column_stack([extra.predict(x_pred), hgb.predict(x_pred), ridge.predict(sparse_pred)])


def blend_weights(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    centered = predictions - np.mean(predictions, axis=0, keepdims=True)
    target = y - np.mean(y)
    weights, _ = nnls(centered, target)
    if float(weights.sum()) <= 0:
        weights = np.ones(predictions.shape[1], dtype=np.float64)
    weights = weights / float(weights.sum())
    specialist = float(weights[1:].sum())
    if specialist > 0.35:
        weights[1:] *= 0.35 / specialist
        weights[0] = 0.65
    return weights


def score(y: np.ndarray, p: np.ndarray) -> float:
    return float(r2_score(y, p)) if len(y) > 1 and not np.isclose(np.var(y), 0.0) else float("nan")


def grouped_bootstrap(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if np.isclose(np.var(y[idx]), 0.0):
            continue
        values.append(score(y[idx], candidate[idx]) - score(y[idx], parent[idx]))
    return {"lower_2_5": float(np.quantile(values, 0.025)), "median": float(np.quantile(values, 0.5)), "upper_97_5": float(np.quantile(values, 0.975)), "replicates": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if run_dir.exists():
        if {p.name for p in run_dir.iterdir()} - {"protocol.json"}:
            raise RuntimeError(f"refusing to reuse non-empty run directory: {run_dir}")
    else:
        run_dir.mkdir(parents=True)
    started = time.time()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: i for i, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptors, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    enriched, enriched_names = normalized_features(descriptors, descriptor_names, physical, physical_names)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    dense_parent = reference.target_dense_features(np.hstack([descriptors, physical]), cross_values, cross_available, TARGET)
    sparse_parts = [reference.morgan_count_matrix(molecules, radius=2, bits=4096), reference.morgan_count_matrix(molecules, radius=3, bits=4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    target_rows = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    target_indices = np.asarray([key_to_index[value] for value in target_rows["canonical"]], dtype=np.int64)
    y = target_rows["target"].to_numpy(float)
    groups = target_rows["canonical"].to_numpy(object)
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / "R2-C001-20260803-1645-initial-reference-repaired" / "report.json").read_text(encoding="utf-8"))
    model_names = ["sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local"]
    parent_weights = np.asarray([c001_report["validation"]["target_reports"][TARGET]["blend_weights"][name] for name in model_names], dtype=np.float64)
    config = dict(reference.DEFAULT_CONFIG)
    outer = GroupKFold(n_splits=OUTER_FOLDS)
    parent_oof = np.full(len(y), np.nan)
    arm_oof = np.full((len(y), 3), np.nan)
    candidate_oof = np.full(len(y), np.nan)
    fold_rows: list[dict[str, Any]] = []
    fold_weights: list[np.ndarray] = []
    similarity = np.full(len(y), np.nan)
    scaffold_values = [MurckoScaffold.MurckoScaffoldSmiles(smiles=str(value), includeChirality=False) for value in target_rows["canonical"]]
    for fold, (outer_train, outer_valid) in enumerate(outer.split(np.arange(len(y)), y, groups)):
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[target_indices] = y
        parent_valid = fit_parent(dense_parent, sparse_parts, fingerprints, y_global, target_indices, outer_train, outer_valid, parent_weights, config)
        arms_valid = fit_arms(enriched, sparse_parts, y, target_indices, outer_train, outer_valid)
        inner = GroupKFold(n_splits=INNER_FOLDS)
        meta_y: list[float] = []
        meta_predictions: list[np.ndarray] = []
        inner_groups = groups[outer_train]
        for inner_train_rel, inner_valid_rel in inner.split(outer_train, y[outer_train], inner_groups):
            inner_train = outer_train[inner_train_rel]
            inner_valid = outer_train[inner_valid_rel]
            inner_y_global = np.full(len(keys), np.nan, dtype=np.float64)
            inner_y_global[target_indices] = y
            parent_inner = fit_parent(dense_parent, sparse_parts, fingerprints, inner_y_global, target_indices, inner_train, inner_valid, parent_weights, config)
            arms_inner = fit_arms(enriched, sparse_parts, y, target_indices, inner_train, inner_valid)
            meta_y.extend(y[inner_valid].tolist())
            meta_predictions.extend(np.column_stack([parent_inner, arms_inner]).tolist())
        weights = blend_weights(np.asarray(meta_y, dtype=float), np.asarray(meta_predictions, dtype=float))
        parent_oof[outer_valid] = parent_valid
        arm_oof[outer_valid] = arms_valid
        candidate_oof[outer_valid] = np.column_stack([parent_valid, arms_valid]) @ weights
        fold_weights.append(weights)
        train_fps = [fingerprints[target_indices[i]] for i in outer_train]
        for index in outer_valid:
            similarity[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[target_indices[index]], train_fps))
        fold_rows.append({"fold": fold, "rows": int(len(outer_valid)), "parent_r2": score(y[outer_valid], parent_valid), "candidate_r2": score(y[outer_valid], candidate_oof[outer_valid]), "delta_r2": score(y[outer_valid], candidate_oof[outer_valid]) - score(y[outer_valid], parent_valid), "weights": weights.tolist()})
    final_weights = np.mean(np.asarray(fold_weights), axis=0)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[target_indices] = y
    test_rows = test[test["target_type"] == TARGET].reset_index(drop=True)
    test_indices = np.asarray([key_to_index[value] for value in test_rows["canonical"]], dtype=np.int64)
    parent_test = fit_parent(dense_parent, sparse_parts, fingerprints, y_global, target_indices, np.arange(len(y)), np.empty(0, dtype=np.int64), parent_weights, config, pred_global=test_indices)
    arms_test = fit_arms(enriched, sparse_parts, y, target_indices, np.arange(len(y)), np.empty(0, dtype=np.int64), pred_global=test_indices)
    candidate_test = np.column_stack([parent_test, arms_test]) @ final_weights
    bins = [("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)]
    panels: dict[str, Any] = {}
    for name, lower, upper in bins:
        mask = (similarity >= lower) & (similarity < upper)
        panels[name] = {"rows": int(mask.sum()), "parent_r2": score(y[mask], parent_oof[mask]) if mask.sum() > 1 else None, "candidate_r2": score(y[mask], candidate_oof[mask]) if mask.sum() > 1 else None, "delta_r2": (score(y[mask], candidate_oof[mask]) - score(y[mask], parent_oof[mask])) if mask.sum() > 1 else None}
    scaffold_deltas = []
    for scaffold in sorted(set(scaffold_values)):
        mask = np.asarray([value == scaffold for value in scaffold_values])
        if mask.sum() >= 3 and not np.isclose(np.var(y[mask]), 0.0):
            scaffold_deltas.append(score(y[mask], candidate_oof[mask]) - score(y[mask], parent_oof[mask]))
    panels["scaffold_groups_ge_3"] = {"groups": len(scaffold_deltas), "minimum_delta_r2": float(min(scaffold_deltas)) if scaffold_deltas else None, "mean_delta_r2": float(np.mean(scaffold_deltas)) if scaffold_deltas else None}
    bootstrap = grouped_bootstrap(y, candidate_oof, parent_oof)
    positive_folds = int(sum(row["delta_r2"] > 0 for row in fold_rows))
    decision = "component_pass" if (score(y, candidate_oof) - score(y, parent_oof) >= 0.01 and positive_folds >= 4 and bootstrap["lower_2_5"] > 0 and all(item["delta_r2"] is None or item["delta_r2"] >= 0 for item in panels.values() if isinstance(item, dict) and "delta_r2" in item) and (panels["scaffold_groups_ge_3"]["minimum_delta_r2"] is None or panels["scaffold_groups_ge_3"]["minimum_delta_r2"] >= 0)) else "rejected_component_gate"
    report = {
        "schema_version": "ppp.round2.eps-multiview-grouped.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": PARENT_ID, "lane": "CLEAN_OFFICIAL_ONLY", "target": TARGET, "official_inputs": inputs, "official_only": True,
        "features": enriched_names + ["morgan_count_r2", "morgan_count_r3", "char_ngrams"], "models": {"parent": "C001 weights regenerated inside every outer fold", "extra_trees": {"estimators": 800, "min_samples_leaf": 2, "max_features": 0.60}, "hist_gradient_boosting": {"max_iter": 350, "learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 8, "l2_regularization": 0.20}, "ridge": {"alpha": 30.0}, "specialist_cap": 0.35},
        "rows": int(len(y)), "parent_r2": score(y, parent_oof), "candidate_r2": score(y, candidate_oof), "delta_r2": score(y, candidate_oof) - score(y, parent_oof), "positive_folds": positive_folds, "folds": fold_rows, "blend_weights_mean": final_weights.tolist(), "group_bootstrap_delta_r2": bootstrap, "panels": panels, "decision": decision, "negative_control": "not_run; no local_eval or post-hoc source used", "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({"canonical": target_rows["canonical"], "target": y, "parent": parent_oof, "candidate": candidate_oof, "similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({"id": test_rows["id"].astype(int), "target": candidate_test, "parent": parent_test}).to_csv(run_dir / "test_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.eps-multiview-grouped.v1", "parent": PARENT_ID, "seed": SEED, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "official_inputs": inputs, "target": TARGET})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# EPS multiview grouped specialist\n\nDecision: **{decision}**. This is a clean official-only component; no local_eval or test-external_label source was read.\n", encoding="utf-8")
    (run_dir / "run.log").write_text(f"experiment_id={run_dir.name}\nmean_parent_r2={report['parent_r2']:.16f}\nmean_candidate_r2={report['candidate_r2']:.16f}\ndelta_r2={report['delta_r2']:.16f}\nlocal_eval_read=false\nkaggle_action=false\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    files = ["command.txt", "config.json", "decision.md", "environment.txt", "metrics.json", "oof_predictions.csv", "run.log", "test_predictions.csv", "protocol.json"]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(run_dir / name)}  {name}" for name in files) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": decision, "parent_r2": report["parent_r2"], "candidate_r2": report["candidate_r2"], "delta_r2": report["delta_r2"], "positive_folds": positive_folds, "bootstrap_lower": bootstrap["lower_2_5"]}, indent=2))


if __name__ == "__main__":
    main()
