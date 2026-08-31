#!/usr/bin/env python3
"""Clean Round-2 target-specific structural carrier screen.

This experiment uses only the official Round-2 train/archive/test bundle.  It
rebuilds a comparable grouped parent and tests one preregistered structural
carrier per target.  No external_label file, external target, prior prediction, or
fitted artifact is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
SEED = 2026
PARENT_ID = "R2-C053-round1-target-specific-screen"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, pred))


def folds_for(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[valid] = fold
    if np.any(folds < 0):
        raise RuntimeError("incomplete outer group folds")
    return folds


def nested_group_audit(groups: np.ndarray, folds: np.ndarray) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for fold in range(5):
        outer_train = np.flatnonzero(folds != fold)
        outer_valid = np.flatnonzero(folds == fold)
        inner_groups = groups[outer_train]
        inner = GroupKFold(n_splits=4)
        for inner_fold, (train_rel, valid_rel) in enumerate(inner.split(outer_train, groups=inner_groups)):
            train_groups = set(inner_groups[train_rel].tolist())
            valid_groups = set(inner_groups[valid_rel].tolist())
            if train_groups & valid_groups:
                raise RuntimeError("inner group intersection")
            if set(groups[outer_valid].tolist()) & train_groups:
                raise RuntimeError("outer validation group entered inner training")
            if set(groups[outer_valid].tolist()) & valid_groups:
                raise RuntimeError("outer validation group entered inner validation")
            audit.append({"outer_fold": fold, "inner_fold": inner_fold, "train_rows": int(len(train_rel)), "valid_rows": int(len(valid_rel))})
    return audit


def sanitize_dense(matrix: np.ndarray, train_rows: np.ndarray, pred_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(matrix[train_rows], dtype=np.float64).copy()
    pred = np.asarray(matrix[pred_rows], dtype=np.float64).copy()
    train[(~np.isfinite(train)) | (np.abs(train) > 1.0e12)] = np.nan
    pred[(~np.isfinite(pred)) | (np.abs(pred) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(train), imputer.transform(pred)


def top_k(k: int) -> SelectKBest:
    return SelectKBest(score_func=f_regression, k=k)


def model_for(target: str, features: int) -> Pipeline:
    if target == "tg":
        return Pipeline([
            ("select", top_k(min(96, features))),
            ("scale", StandardScaler()),
            ("model", KernelRidge(alpha=2.0, kernel="rbf", gamma=0.02)),
        ])
    if target == "egc":
        return Pipeline([
            ("select", top_k(min(192, features))),
            ("model", lgb.LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=15, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=1, verbosity=-1)),
        ])
    if target == "egb":
        return Pipeline([
            ("select", top_k(min(160, features))),
            ("model", ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.65, random_state=SEED, n_jobs=2)),
        ])
    if target == "nc":
        return Pipeline([
            ("select", top_k(min(128, features))),
            ("model", HistGradientBoostingRegressor(max_iter=450, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=0.3, random_state=SEED)),
        ])
    if target == "eps":
        return Pipeline([
            ("select", top_k(min(96, features))),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.08, max_iter=10000, random_state=SEED)),
        ])
    raise ValueError(target)


def parent_predict(
    dense: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    fingerprints: list[Any],
    y_global: np.ndarray,
    train_indices: np.ndarray,
    pred_indices: np.ndarray,
    target: str,
) -> np.ndarray:
    arms = reference.predict_base_models(
        dense,
        sparse_parts,
        fingerprints,
        y_global,
        train_indices,
        pred_indices,
        reference.DEFAULT_CONFIG,
        target,
    )
    # Equal weights are frozen before this run, avoiding full-data OOF blend fitting.
    return np.mean(arms, axis=1)


def specialist_predict(
    features: np.ndarray,
    y: np.ndarray,
    train_feature_rows: np.ndarray,
    train_y_rows: np.ndarray,
    pred_feature_rows: np.ndarray,
    target: str,
) -> np.ndarray:
    train_x, pred_x = sanitize_dense(features, train_feature_rows, pred_feature_rows)
    train_x = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(train_x)
    # The second imputation is intentionally fitted on the same outer training partition.
    # It keeps the model path explicit and deterministic after sanitization.
    pred_x = SimpleImputer(strategy="median", keep_empty_features=True).fit(train_x).transform(pred_x)
    model = model_for(target, train_x.shape[1])
    model.fit(train_x, y[train_y_rows])
    prediction = np.asarray(model.predict(pred_x), dtype=np.float64)
    q01, q99 = np.quantile(y[train_y_rows], [0.01, 0.99])
    spread = max(float(np.subtract(*np.quantile(y[train_y_rows], [0.75, 0.25]))), float(np.std(y[train_y_rows])), 1.0e-8)
    return np.clip(prediction, q01 - 3.0 * spread, q99 + 3.0 * spread)


def scaffold_values(smiles: list[str]) -> np.ndarray:
    out: list[str] = []
    for value in smiles:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=str(value), includeChirality=False)
        except Exception:
            scaffold = ""
        out.append(scaffold or "ACYCLIC")
    return np.asarray(out, dtype=object)


def similarity_values(fingerprints: list[Any], target_indices: np.ndarray, folds: np.ndarray) -> np.ndarray:
    result = np.full(len(target_indices), np.nan, dtype=np.float64)
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[target_indices[i]] for i in train]
        for local in valid:
            result[local] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[target_indices[local]], train_fps))
    return result


def group_bootstrap_lower(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in chosen])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(r2(y[rows], candidate[rows]) - r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def panel_report(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        count = int(np.sum(selected))
        panels[name] = {"rows": count, "groups": int(np.unique(groups[selected]).size), "eligible": bool(count >= 20 and np.unique(groups[selected]).size >= 5)}
        if panels[name]["eligible"]:
            panels[name]["delta_r2"] = r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected])
        else:
            panels[name]["delta_r2"] = None
    scaffold_deltas: list[float] = []
    for value in sorted(set(scaffolds)):
        selected = scaffolds == value
        count = int(np.sum(selected))
        groups_count = int(np.unique(groups[selected]).size)
        if count >= 10 and groups_count >= 3 and not np.isclose(np.var(y[selected]), 0.0):
            scaffold_deltas.append(r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected]))
    panels["scaffold_groups_ge_3"] = {"evaluated": len(scaffold_deltas), "minimum_delta_r2": float(min(scaffold_deltas)) if scaffold_deltas else None}
    return panels


def chemistry_feature_matrix(base_desc: np.ndarray, descriptor_names: list[str], physical: np.ndarray, physical_names: list[str]) -> np.ndarray:
    """Compact Round-1-inspired descriptor algebra, derived from official SMILES."""
    dpos = {name: i for i, name in enumerate(descriptor_names)}
    ppos = {name: i for i, name in enumerate(physical_names)}
    columns: list[np.ndarray] = [base_desc, physical]
    selected_names = [
        "MolWt", "ExactMolWt", "MolLogP", "MolMR", "TPSA", "HeavyAtomCount",
        "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "RingCount",
        "NumAromaticRings", "FractionCSP3", "NumAliphaticRings",
    ]
    selected = [(f"rdkit_{name}", base_desc[:, dpos[name]]) for name in selected_names if name in dpos]
    selected += [(f"physical_{name}", physical[:, ppos[name]]) for name in physical_names if name in {"atom_count", "heavy_atom_count", "ring_count", "aromatic_atom_count", "hetero_atom_count", "rotatable_bonds_approx", "double_bond_count", "branch_count", "n_count", "o_count", "s_count", "si_count"}]
    engineered: list[np.ndarray] = []
    for _, values in selected:
        values = np.asarray(values, dtype=np.float64)
        engineered.extend([np.sign(values) * np.log1p(np.abs(values)), np.square(np.nan_to_num(values, nan=0.0))])
    for left_index in range(min(len(selected), 18)):
        left = np.nan_to_num(selected[left_index][1], nan=0.0, posinf=0.0, neginf=0.0)
        for right_index in range(left_index + 1, min(len(selected), 18)):
            right = np.nan_to_num(selected[right_index][1], nan=0.0, posinf=0.0, neginf=0.0)
            engineered.append(left / (np.abs(right) + 1.0))
    columns.append(np.column_stack(engineered))
    return np.hstack(columns).astype(np.float64, copy=False)


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
    if not run_dir.is_dir():
        raise RuntimeError(f"pre-created run directory required: {run_dir}")
    if {path.name for path in run_dir.iterdir()} - {"protocol.json"}:
        raise RuntimeError(f"refusing non-empty run directory: {run_dir}")
    started = time.time()
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: i for i, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    base_desc, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([base_desc, physical]).astype(np.float64, copy=False)
    empty_cross = np.full((len(keys), len(TARGETS)), np.nan, dtype=np.float64)
    empty_available = np.zeros_like(empty_cross)
    target_dense = {target: reference.target_dense_features(base_dense, empty_cross, empty_available, target) for target in TARGETS}
    sparse_parts = [reference.morgan_count_matrix(molecules, radius=2, bits=4096), reference.morgan_count_matrix(molecules, radius=3, bits=4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)

    structural_dense = chemistry_feature_matrix(base_desc, descriptor_names, physical, physical_names)
    print(json.dumps({"stage": "features_ready", "keys": len(keys), "features": int(structural_dense.shape[1])}), flush=True)
    changed = {"tg", "egc", "egb", "nc", "eps"}
    targets: dict[str, Any] = {}
    all_oof: list[dict[str, Any]] = []
    test_predictions: list[dict[str, Any]] = []
    inner_audit_rows = 0
    for target in TARGETS:
        print(json.dumps({"stage": "target_start", "target": target}), flush=True)
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
        groups = frame["canonical"].to_numpy(object)
        folds = folds_for(groups)
        inner_audit = nested_group_audit(groups, folds)
        inner_audit_rows += len(inner_audit)
        parent_oof = np.full(len(y), np.nan, dtype=np.float64)
        candidate_oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            valid_rows = np.flatnonzero(folds == fold)
            y_global = np.full(len(keys), np.nan, dtype=np.float64)
            y_global[indices] = y
            parent_oof[valid_rows] = parent_predict(target_dense[target], sparse_parts, fingerprints, y_global, indices[train_rows], indices[valid_rows], target)
            if target in changed:
                candidate_oof[valid_rows] = specialist_predict(structural_dense, y, indices[train_rows], train_rows, indices[valid_rows], target)
            else:
                candidate_oof[valid_rows] = parent_oof[valid_rows]
            fold_rows.append({"fold": fold, "rows": int(len(valid_rows)), "parent_r2": r2(y[valid_rows], parent_oof[valid_rows]), "candidate_r2": r2(y[valid_rows], candidate_oof[valid_rows]), "delta_r2": r2(y[valid_rows], candidate_oof[valid_rows]) - r2(y[valid_rows], parent_oof[valid_rows])})
        similarity = similarity_values(fingerprints, indices, folds)
        scaffolds = scaffold_values(frame["canonical"].tolist())
        panels = panel_report(y, candidate_oof, parent_oof, groups, similarity, scaffolds)
        target_result = {
            "rows": int(len(y)),
            "model": "grouped_parent_equal_four_arm_average" if target not in changed else {"tg": "selectk96_rbf_kernel_ridge", "egc": "selectk192_lightgbm", "egb": "selectk160_extratrees", "nc": "selectk128_histgradientboosting", "eps": "selectk96_elasticnet"}[target],
            "parent_r2": r2(y, parent_oof),
            "candidate_r2": r2(y, candidate_oof),
            "delta_r2": r2(y, candidate_oof) - r2(y, parent_oof),
            "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
            "group_bootstrap_lower": group_bootstrap_lower(y, candidate_oof, parent_oof, groups),
            "folds": fold_rows,
            "panels": panels,
            "outer_folds": folds.tolist(),
            "inner_group_audit_rows": int(len(inner_audit)),
        }
        targets[target] = target_result
        print(json.dumps({"stage": "target_done", "target": target, "parent_r2": target_result["parent_r2"], "candidate_r2": target_result["candidate_r2"], "delta_r2": target_result["delta_r2"]}), flush=True)
        for row_idx in range(len(y)):
            all_oof.append({"target_type": target, "canonical": frame.iloc[row_idx]["canonical"], "target": float(y[row_idx]), "fold": int(folds[row_idx]), "parent": float(parent_oof[row_idx]), "candidate": float(candidate_oof[row_idx]), "group": str(groups[row_idx]), "scaffold": str(scaffolds[row_idx]), "similarity": float(similarity[row_idx])})
        test_frame = test[test["target_type"] == target].reset_index(drop=True)
        test_indices = np.asarray([key_to_index[key] for key in test_frame["canonical"]], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[indices] = y
        parent_test = parent_predict(target_dense[target], sparse_parts, fingerprints, y_global, indices, test_indices, target)
        if target in changed:
            candidate_test = specialist_predict(structural_dense, y, indices, np.arange(len(y), dtype=np.int64), test_indices, target)
        else:
            candidate_test = parent_test
        for row, prediction in zip(test_frame.itertuples(index=False), candidate_test, strict=True):
            test_predictions.append({"id": int(row.id), "target": float(prediction), "target_type": target})

    oof = pd.DataFrame(all_oof)
    predictions = pd.DataFrame(test_predictions).sort_values("id")
    if len(predictions) != len(test) or not predictions["id"].equals(test["id"]):
        raise RuntimeError("test IDs/order mismatch")
    if not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("non-finite candidate predictions")
    submission = predictions[["id", "target"]].copy()
    submission.to_csv(run_dir / "candidate_predictions.csv", index=False)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    mean_parent = float(np.mean([targets[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([targets[target]["candidate_r2"] for target in TARGETS]))
    minimum_panel = []
    for target in TARGETS:
        for value in targets[target]["panels"].values():
            if value.get("delta_r2") is not None:
                minimum_panel.append(float(value["delta_r2"]))
        if targets[target]["panels"]["scaffold_groups_ge_3"]["minimum_delta_r2"] is not None:
            minimum_panel.append(float(targets[target]["panels"]["scaffold_groups_ge_3"]["minimum_delta_r2"]))
    report = {
        "schema_version": "ppp.round2.c053.round1-carrier-screen.v1",
        "experiment_id": run_dir.name,
        "parent": PARENT_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_inputs": inputs,
        "target_results": targets,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "minimum_panel_delta": min(minimum_panel) if minimum_panel else None,
        "target_loss": min(targets[target]["delta_r2"] for target in TARGETS),
        "inner_group_audit_rows": inner_audit_rows,
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": int(len(oof))},
        "source_dependencies": {"reference": sha256_file(root / "tools" / "initial_reference_pipeline.py"), "script": sha256_file(root / "tools" / "round2_c053_round1_carriers.py")},
        "official_only": True,
        "external_label_file_read": False,
        "elapsed_seconds": float(time.time() - started),
        "decision": "component_screen_only" if mean_candidate <= mean_parent else "candidate_requires_full_gate_review",
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "outer_folds": 5, "inner_folds": 4, "changed_targets": sorted(changed), "features": "Round-1 pure structural descriptor blocks", "models": {"tg": "selectk96_rbf_kernel_ridge", "egc": "selectk192_lightgbm", "egb": "selectk160_extratrees", "nc": "selectk128_histgradientboosting", "eps": "selectk96_elasticnet"}, "parent": "equal average of reference sparse ridge, dense ridge, extra trees, and local similarity arms; no cross-property labels"})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nlightgbm={lgb.__version__}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# C053 decision\n\nMean grouped parent R2: {mean_parent:.12f}\nMean grouped candidate R2: {mean_candidate:.12f}\nMean gain: {mean_candidate - mean_parent:+.12f}\n\nThis is a clean official-only screen. It is not a final candidate until all transfer, lifecycle, and notebook gates pass.\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("protocol.json", "config.json", "metrics.json", "candidate_predictions.csv", "oof_predictions.csv", "environment.txt", "command.txt", "decision.md")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "target_loss": report["target_loss"], "minimum_panel_delta": report["minimum_panel_delta"]}, sort_keys=True))


if __name__ == "__main__":
    main()
