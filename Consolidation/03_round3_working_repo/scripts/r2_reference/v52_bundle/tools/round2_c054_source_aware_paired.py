#!/usr/bin/env python3
"""Official-only source-aware paired-property EPS/Nc screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
CHANGED = {"eps", "nc"}
SEED = 2026


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def r2(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, p))


def no_stereo(value: str) -> str:
    mol = Chem.MolFromSmiles(str(value).replace("[*]", "*"))
    if mol is None:
        return str(value)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def fold_ids(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[valid] = fold
    if np.any(folds < 0):
        raise RuntimeError("incomplete group folds")
    return folds


def true_group_bootstrap(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(r2(y[rows], candidate[rows]) - r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def paired_arrays(pooled: pd.DataFrame, keys: list[str], heldout_groups: set[str] | None) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(keys), len(TARGETS)), np.nan, dtype=np.float64)
    available = np.zeros_like(values)
    key_pos = {key: i for i, key in enumerate(keys)}
    for row in pooled.itertuples(index=False):
        group = no_stereo(str(row.canonical))
        if heldout_groups is not None and group in heldout_groups:
            continue
        pos = key_pos[str(row.canonical)]
        target_pos = TARGETS.index(str(row.target_type))
        values[pos, target_pos] = float(row.target)
        available[pos, target_pos] = 1.0
    return values, available


def make_features(base_dense: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray, target: str) -> np.ndarray:
    return reference.target_dense_features(base_dense, cross_values, cross_available, target)


def parent_predict(dense: np.ndarray, sparse_parts: list[sparse.csr_matrix], fingerprints: list[Any], y_global: np.ndarray, train_idx: np.ndarray, pred_idx: np.ndarray, target: str) -> np.ndarray:
    arms = reference.predict_base_models(dense, sparse_parts, fingerprints, y_global, train_idx, pred_idx, reference.DEFAULT_CONFIG, target)
    return np.mean(arms, axis=1)


def lgb_predict(features: np.ndarray, y: np.ndarray, train_feature_rows: np.ndarray, train_y_rows: np.ndarray, pred_feature_rows: np.ndarray) -> np.ndarray:
    train = np.asarray(features[train_feature_rows], dtype=np.float64).copy()
    pred = np.asarray(features[pred_feature_rows], dtype=np.float64).copy()
    train[(~np.isfinite(train)) | (np.abs(train) > 1.0e12)] = np.nan
    pred[(~np.isfinite(pred)) | (np.abs(pred) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(train)
    pred = imputer.transform(pred)
    model = lgb.LGBMRegressor(objective="regression", n_estimators=450, learning_rate=0.03, num_leaves=23, min_child_samples=12, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=6.0, random_state=SEED, n_jobs=1, verbosity=-1)
    model.fit(train, y[train_y_rows])
    prediction = np.asarray(model.predict(pred), dtype=np.float64)
    q01, q99 = np.quantile(y[train_y_rows], [0.01, 0.99])
    spread = max(float(np.subtract(*np.quantile(y[train_y_rows], [0.75, 0.25]))), float(np.std(y[train_y_rows])), 1.0e-8)
    return np.clip(prediction, q01 - 3.0 * spread, q99 + 3.0 * spread)


def panel_deltas(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        count = int(np.sum(selected))
        group_count = int(np.unique(groups[selected]).size)
        report[name] = {"rows": count, "groups": group_count, "eligible": bool(count >= 20 and group_count >= 5), "delta_r2": None}
        if report[name]["eligible"]:
            report[name]["delta_r2"] = r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected])
    scaffold_values: list[float] = []
    for value in sorted(set(scaffolds)):
        selected = scaffolds == value
        if int(np.sum(selected)) >= 10 and int(np.unique(groups[selected]).size) >= 3 and not np.isclose(np.var(y[selected]), 0.0):
            scaffold_values.append(r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected]))
    report["scaffold_groups_ge_3"] = {"evaluated": len(scaffold_values), "minimum_delta_r2": min(scaffold_values) if scaffold_values else None}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} - {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: i for i, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    all_target_results: dict[str, Any] = {}
    all_oof: list[dict[str, Any]] = []
    predictions_by_id: dict[int, float] = {}
    support_audit: dict[str, Any] = {}
    for target in TARGETS:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
        groups = np.asarray([no_stereo(str(value)) for value in frame["canonical"]], dtype=object)
        folds = fold_ids(groups)
        parent_oof = np.full(len(y), np.nan, dtype=np.float64)
        candidate_oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            valid_rows = np.flatnonzero(folds == fold)
            heldout = set(groups[valid_rows].tolist())
            cross_values, cross_available = paired_arrays(pooled, keys, heldout)
            features = make_features(base_dense, cross_values, cross_available, target)
            y_global = np.full(len(keys), np.nan, dtype=np.float64)
            y_global[indices] = y
            parent_oof[valid_rows] = parent_predict(features, sparse_parts, fingerprints, y_global, indices[train_rows], indices[valid_rows], target)
            if target in CHANGED:
                candidate_oof[valid_rows] = lgb_predict(features, y, indices[train_rows], train_rows, indices[valid_rows])
            else:
                candidate_oof[valid_rows] = parent_oof[valid_rows]
            fold_rows.append({"fold": fold, "rows": int(len(valid_rows)), "parent_r2": r2(y[valid_rows], parent_oof[valid_rows]), "candidate_r2": r2(y[valid_rows], candidate_oof[valid_rows]), "delta_r2": r2(y[valid_rows], candidate_oof[valid_rows]) - r2(y[valid_rows], parent_oof[valid_rows])})
        train_fps = [fingerprints[index] for index in indices]
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            valid_rows = np.flatnonzero(folds == fold)
            train_rows = np.flatnonzero(folds != fold)
            fps = [train_fps[row] for row in train_rows]
            for row in valid_rows:
                similarity[row] = max(DataStructs.BulkTanimotoSimilarity(train_fps[row], fps))
        scaffolds = np.asarray([MurckoScaffold.MurckoScaffoldSmiles(smiles=str(value), includeChirality=False) or "ACYCLIC" for value in frame["canonical"]], dtype=object)
        panels = panel_deltas(y, candidate_oof, parent_oof, groups, similarity, scaffolds)
        all_target_results[target] = {"rows": int(len(y)), "parent_r2": r2(y, parent_oof), "candidate_r2": r2(y, candidate_oof), "delta_r2": r2(y, candidate_oof) - r2(y, parent_oof), "positive_folds": int(sum(item["delta_r2"] > 0 for item in fold_rows)), "group_bootstrap_lower": true_group_bootstrap(y, candidate_oof, parent_oof, groups), "folds": fold_rows, "panels": panels, "group_count": int(np.unique(groups).size)}
        for row in range(len(y)):
            all_oof.append({"target_type": target, "canonical": frame.iloc[row]["canonical"], "group": str(groups[row]), "target": float(y[row]), "fold": int(folds[row]), "parent": float(parent_oof[row]), "candidate": float(candidate_oof[row]), "similarity": float(similarity[row])})
        full_cross, full_available = paired_arrays(pooled, keys, None)
        full_features = make_features(base_dense, full_cross, full_available, target)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[indices] = y
        test_frame = test[test["target_type"] == target].reset_index(drop=True)
        test_indices = np.asarray([key_to_index[key] for key in test_frame["canonical"]], dtype=np.int64)
        if target in CHANGED:
            test_pred = lgb_predict(full_features, y, indices, np.arange(len(y), dtype=np.int64), test_indices)
        else:
            test_pred = parent_predict(full_features, sparse_parts, fingerprints, y_global, indices, test_indices, target)
        for row, prediction in zip(test_frame.itertuples(index=False), test_pred, strict=True):
            predictions_by_id[int(row.id)] = float(prediction)
        support_audit[target] = {"test_rows": int(len(test_frame)), "other_property_support_rows": int(np.sum(np.any(full_available[test_indices, :][:, [i for i in range(len(TARGETS)) if TARGETS[i] != target]] > 0, axis=1)))}
    submission = pd.DataFrame({"id": test["id"].to_numpy(dtype=np.int64), "target": [predictions_by_id[int(value)] for value in test["id"]]})
    if len(submission) != 4940 or submission["id"].duplicated().any() or not submission["id"].equals(test["id"].reset_index(drop=True)) or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("final 4940-row output contract failed")
    submission.to_csv(run_dir / "candidate_predictions.csv", index=False)
    pd.DataFrame(all_oof).to_csv(run_dir / "oof_predictions.csv", index=False)
    mean_parent = float(np.mean([all_target_results[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([all_target_results[target]["candidate_r2"] for target in TARGETS]))
    report = {"schema_version": "ppp.round2.c054.source-aware-paired.v1", "experiment_id": run_dir.name, "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7", "created_at": datetime.now().astimezone().isoformat(), "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "changed_targets": sorted(CHANGED), "targets": all_target_results, "support_audit": support_audit, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": len(all_oof)}, "source_sha256": sha256_file(root / "tools" / "round2_c054_source_aware_paired.py"), "elapsed_seconds": time.time() - started, "decision": "component_screen_only"}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "outer_folds": 5, "group_key": "canonical_no_stereo", "changed_targets": sorted(CHANGED), "model": "fixed LightGBM on structural descriptors plus fold-masked paired-property values and availability flags", "parent": "same fold-local source-aware equal four-arm reference", "external_label_file_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nlightgbm={lgb.__version__}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# C054 decision\n\nParent grouped mean: {mean_parent:.12f}\nCandidate grouped mean: {mean_candidate:.12f}\nGain: {mean_candidate - mean_parent:+.12f}\n\nNo external_label file or external target was read.\n", encoding="utf-8")
    manifest = [run_dir / name for name in ("protocol.json", "config.json", "metrics.json", "candidate_predictions.csv", "oof_predictions.csv", "environment.txt", "command.txt", "decision.md")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "targets": {target: result["delta_r2"] for target, result in all_target_results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
