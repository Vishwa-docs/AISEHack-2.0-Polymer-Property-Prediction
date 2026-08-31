#!/usr/bin/env python3
"""Bounded official-only shared seven-target standardized model diagnostic."""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
ARM = "multitask_z_histgb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(y, prediction))


def group_folds(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def group_bootstrap_lower(delta: np.ndarray, groups: np.ndarray, seed: int = 2026) -> float:
    unique = np.unique(groups)
    grouped = {group: delta[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(500, dtype=np.float64)
    for draw in range(len(means)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        means[draw] = float(np.mean(np.concatenate([grouped[group] for group in selected])))
    return float(np.quantile(means, 0.025))


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    result = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in training]
        for index in validation:
            result[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"Pre-created protocol directory is required: {run_dir}")
    existing = {path.name for path in run_dir.iterdir()}
    if existing - {"protocol.json"}:
        raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    pooled_targets = pooled["target_type"].to_numpy(object)
    pooled_keys = pooled["canonical"].to_numpy(object)
    pooled_groups = pooled_keys.copy()
    global_folds = group_folds(pooled_groups)
    pooled_indices = np.asarray([key_to_index[key] for key in pooled_keys], dtype=np.int64)
    y_all = pooled["target"].to_numpy(float)
    target_codes = np.asarray([TARGETS.index(target) for target in pooled_targets], dtype=np.int64)
    feature_cache: dict[str, np.ndarray] = {}
    for target in TARGETS:
        target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        feature_cache[target] = target_dense[pooled_indices]
    onehot = np.zeros((len(pooled), len(TARGETS)), dtype=np.float64)
    onehot[np.arange(len(pooled)), target_codes] = 1.0
    shared_x = np.hstack([np.vstack([feature_cache[target][row] for row, target in enumerate(pooled_targets)]), onehot])
    shared_x[~np.isfinite(shared_x)] = np.nan
    reports: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGETS):
        local_mask = pooled_targets == target
        local_positions = np.flatnonzero(local_mask)
        local_keys = pooled_keys[local_mask]
        local_folds = global_folds[local_mask]
        local_indices = pooled_indices[local_mask]
        y = y_all[local_mask]
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[local_indices] = y
        baseline = np.full(len(y), np.nan, dtype=np.float64)
        candidate = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            local_train = np.flatnonzero(local_folds != fold)
            local_validation = np.flatnonzero(local_folds == fold)
            global_train_mask = global_folds != fold
            global_validation_mask = global_folds == fold
            base = reference.predict_base_models(
                reference.target_dense_features(base_dense, cross_values, cross_available, target),
                sparse_parts,
                fingerprints,
                y_global,
                local_indices[local_train],
                local_indices[local_validation],
                reference.DEFAULT_CONFIG,
                target,
            )
            weights = c001_report["validation"]["target_reports"][target]["blend_weights"]
            blend_weights = np.asarray([weights[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)
            intercept = float(c001_report["validation"]["target_reports"][target]["blend_intercept"])
            baseline[local_validation] = base @ blend_weights + intercept
            train_stats: dict[str, tuple[float, float]] = {}
            y_z = np.empty(int(np.sum(global_train_mask)), dtype=np.float64)
            train_targets = pooled_targets[global_train_mask]
            train_values = y_all[global_train_mask]
            for property_name in TARGETS:
                values = train_values[train_targets == property_name]
                mean = float(np.mean(values))
                std = float(np.std(values))
                train_stats[property_name] = (mean, std if std > 1e-8 else 1.0)
            for property_name in TARGETS:
                mask = train_targets == property_name
                mean, std = train_stats[property_name]
                y_z[mask] = (train_values[mask] - mean) / std
            train_x = shared_x[global_train_mask]
            validation_x = shared_x[global_validation_mask]
            finite_count = np.sum(np.isfinite(train_x), axis=0)
            safe_train_x = np.where(np.isfinite(train_x), train_x, np.nan)
            safe_min = np.nanmin(safe_train_x, axis=0, initial=0.0)
            safe_max = np.nanmax(safe_train_x, axis=0, initial=0.0)
            keep_columns = (finite_count >= 2) & ((safe_max - safe_min) > 1.0e-12)
            if not np.any(keep_columns):
                raise RuntimeError("No nonconstant shared features remained in a training fold")
            model = HistGradientBoostingRegressor(
                max_iter=260,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=12,
                l2_regularization=0.10,
                random_state=2026 + target_index,
            )
            model.fit(train_x[:, keep_columns], y_z)
            prediction_z = model.predict(validation_x[:, keep_columns])
            validation_targets = pooled_targets[global_validation_mask]
            target_prediction = np.asarray(prediction_z, dtype=np.float64)
            mean, std = train_stats[target]
            target_prediction = target_prediction * std + mean
            validation_positions = local_positions[local_folds == fold]
            candidate[validation_positions] = target_prediction[validation_targets == target]
        nearest = nearest_similarity([fingerprints[index] for index in local_indices], local_folds)
        report: dict[str, Any] = {
            "rows": int(len(y)),
            "baseline_r2": r2(y, baseline),
            "candidate_r2": r2(y, candidate),
            "delta_r2": r2(y, candidate) - r2(y, baseline),
            "folds": [],
            "low_similarity": {},
            "target_standardization": "fit-fold-only",
        }
        delta = candidate - baseline
        for fold in range(5):
            selected = local_folds == fold
            report["folds"].append({
                "fold": fold,
                "rows": int(np.sum(selected)),
                "baseline_r2": r2(y[selected], baseline[selected]),
                "candidate_r2": r2(y[selected], candidate[selected]),
                "delta_r2": r2(y[selected], candidate[selected]) - r2(y[selected], baseline[selected]),
            })
        for name_bin, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
            selected = (nearest >= lower) & (nearest < upper)
            if int(np.sum(selected)) < 5:
                continue
            report["low_similarity"][name_bin] = {
                "rows": int(np.sum(selected)),
                "baseline_r2": r2(y[selected], baseline[selected]),
                "candidate_r2": r2(y[selected], candidate[selected]),
                "delta_r2": r2(y[selected], candidate[selected]) - r2(y[selected], baseline[selected]),
            }
        positive_folds = int(sum(fold["delta_r2"] > 0 for fold in report["folds"]))
        low_values = [item["delta_r2"] for item in report["low_similarity"].values()]
        report["positive_folds"] = positive_folds
        report["bootstrap_lower"] = group_bootstrap_lower(delta, local_keys)
        report["min_low_similarity_delta"] = min(low_values) if low_values else None
        report["pass"] = bool(
            report["delta_r2"] >= 0.01
            and positive_folds >= 4
            and report["bootstrap_lower"] > 0.0
            and (not low_values or min(low_values) >= 0.0)
        )
        reports[target] = report
        metric_rows.append({"target": target, **report})
    passing_targets = [target for target, report in reports.items() if report["pass"]]
    audit = {
        "schema_version": "ppp.round2.multitask-z-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C011-20260803-1741-polymer-views-v3",
        "official_inputs": inputs,
        "pooled_rows": int(len(pooled)),
        "shared_features": int(shared_x.shape[1]),
        "target_encoding": {target: index for index, target in enumerate(TARGETS)},
        "global_group_folds": 5,
        "targets": reports,
        "passing_targets": passing_targets,
        "decision": "component_pass" if passing_targets else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame(metric_rows).to_json(run_dir / "metrics_rows.json", orient="records", indent=2)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.multitask-z.v1", "seed": 2026, "folds": 5, "arm": ARM, "shared_features": int(shared_x.shape[1]), "pooled_rows": int(len(pooled)), "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C012 multitask-z decision\n\nDecision: **{audit['decision']}**. No candidate changed in this component diagnostic.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "metrics_rows.json", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "passing_targets": passing_targets, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
