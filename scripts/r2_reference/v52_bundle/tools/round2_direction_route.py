#!/usr/bin/env python3
"""Direction-consistent selective route inside the clean C017 bins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
C013_ID = "R2-C013-20260803-1804-target-tree-zoo-v2"
ROUTES = {
    "eps": {"model": "catboost", "alpha": 0.50, "bins": [(0.30, 0.70)]},
    "egb": {"model": "xgboost", "alpha": 0.50, "bins": [(0.00, 0.50), (0.70, 1.01)]},
    "nc": {"model": "xgboost", "alpha": 0.50, "bins": [(0.50, 0.70)]},
    "ei": {"model": "lightgbm", "alpha": 0.50, "bins": [(0.30, 0.50)]},
}
MODEL_FOR_TARGET = {"tg": "histgb", "egc": "lightgbm", "egb": "xgboost", "ei": "lightgbm", "eea": "catboost", "nc": "xgboost", "eps": "catboost"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(r2_score(y, pred))


def folds_for(groups: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import GroupKFold
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    result = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in training]
        for index in validation:
            result[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return result


def group_bootstrap_lower(delta: np.ndarray, groups: np.ndarray, seed: int = 2026) -> float:
    if len(delta) < 5:
        return 0.0
    unique = np.unique(groups)
    grouped = {group: delta[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(500, dtype=np.float64)
    for draw in range(len(means)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        means[draw] = float(np.mean(np.concatenate([grouped[group] for group in selected])))
    return float(np.quantile(means, 0.025))


def fold_matrix(matrix: np.ndarray, train_index: np.ndarray, validation_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = np.asarray(matrix[train_index], dtype=np.float64).copy()
    validation_x = np.asarray(matrix[validation_index], dtype=np.float64).copy()
    limit = float(reference.DEFAULT_CONFIG["dense_abs_limit"])
    train_x[~np.isfinite(train_x) | (np.abs(train_x) > limit)] = np.nan
    validation_x[~np.isfinite(validation_x) | (np.abs(validation_x) > limit)] = np.nan
    train_x = imputer.fit_transform(train_x)
    validation_x = imputer.transform(validation_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    return train_x[:, keep], validation_x[:, keep]


def score(y: np.ndarray, baseline: np.ndarray, arm: np.ndarray, groups: np.ndarray, folds: np.ndarray, nearest: np.ndarray, spec: dict[str, Any] | None) -> dict[str, Any]:
    bins = spec["bins"] if spec is not None else []
    alpha = float(spec["alpha"]) if spec is not None else 0.0
    eligible = np.zeros(len(y), dtype=bool)
    for lower, upper in bins:
        eligible |= (nearest >= lower) & (nearest < upper)
    correction = arm - baseline
    changed = eligible & (correction > 0.0)
    prediction = baseline.copy()
    prediction[changed] = baseline[changed] + alpha * correction[changed]
    fold_rows = []
    for fold in range(5):
        selected = folds == fold
        fold_rows.append({"fold": fold, "rows": int(np.sum(selected)), "delta_r2": r2(y[selected], prediction[selected]) - r2(y[selected], baseline[selected])})
    low_rows = {}
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        if int(np.sum(selected)) < 5:
            continue
        low_rows[name] = {"rows": int(np.sum(selected)), "changed_rows": int(np.sum(changed & selected)), "delta_r2": r2(y[selected], prediction[selected]) - r2(y[selected], baseline[selected])}
    delta = prediction - baseline
    changed_bootstrap = group_bootstrap_lower(delta[changed], groups[changed])
    return {
        "model": spec["model"] if spec is not None else "c001_incumbent",
        "alpha": alpha,
        "bins": bins,
        "changed_rows": int(np.sum(changed)),
        "eligible_rows": int(np.sum(eligible)),
        "r2": r2(y, prediction),
        "delta_r2": r2(y, prediction) - r2(y, baseline),
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "bootstrap_lower_changed": changed_bootstrap,
        "low_similarity": low_rows,
        "min_low_similarity_delta": min((row["delta_r2"] for row in low_rows.values()), default=0.0),
        "folds": fold_rows,
    }


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
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    tools_dir = root.parent / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(tools_dir))
    import round2_target_tree_zoo_v2 as zoo  # noqa: PLC0415
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    c013_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C013_ID / "metrics.json").read_text(encoding="utf-8"))
    baseline_grouped = {target: float(c013_report["targets"][target]["baseline_r2"]) for target in TARGETS}
    baseline_mean = float(np.mean(list(baseline_grouped.values())))
    reports: dict[str, Any] = {}
    route_rows = []
    for target_index, target in enumerate(TARGETS):
        route_spec = ROUTES.get(target)
        if route_spec is None:
            route_rows.append({"target": target, "model": "c001_incumbent", "alpha": 0.0, "changed_rows": 0, "baseline_r2": baseline_grouped[target], "route_r2": baseline_grouped[target], "delta_r2": 0.0, "bootstrap_lower_changed": 0.0, "min_low_similarity_delta": 0.0, "pass_component": True})
            reports[target] = {"route": route_rows[-1]}
            continue
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        groups = frame["canonical"].to_numpy(object)
        indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[indices] = y
        folds = folds_for(groups)
        target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        baseline = np.full(len(y), np.nan, dtype=np.float64)
        arm_prediction = np.full(len(y), np.nan, dtype=np.float64)
        weights = c001_report["validation"]["target_reports"][target]["blend_weights"]
        blend_weights = np.asarray([weights[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)
        intercept = float(c001_report["validation"]["target_reports"][target]["blend_intercept"])
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            base = reference.predict_base_models(target_dense, sparse_parts, fingerprints, y_global, indices[train_rows], indices[validation_rows], reference.DEFAULT_CONFIG, target)
            baseline[validation_rows] = base @ blend_weights + intercept
            train_x, validation_x = fold_matrix(target_dense, indices[train_rows], indices[validation_rows])
            model = zoo.model_factory(route_spec["model"], 2026 + target_index)
            model.fit(train_x, y[train_rows])
            arm_prediction[validation_rows] = np.asarray(model.predict(validation_x), dtype=np.float64)
        nearest = nearest_similarity([fingerprints[index] for index in indices], folds)
        route_score = score(y, baseline, arm_prediction, groups, folds, nearest, route_spec)
        route_score["baseline_r2"] = r2(y, baseline)
        route_score["pass_component"] = bool(route_score["delta_r2"] > 0.0 and route_score["bootstrap_lower_changed"] > 0.0 and route_score["min_low_similarity_delta"] >= 0.0)
        reports[target] = {"route": route_score}
        route_rows.append({"target": target, "model": route_spec["model"], "alpha": route_score["alpha"], "changed_rows": route_score["changed_rows"], "baseline_r2": route_score["baseline_r2"], "route_r2": route_score["r2"], "delta_r2": route_score["delta_r2"], "bootstrap_lower_changed": route_score["bootstrap_lower_changed"], "min_low_similarity_delta": route_score["min_low_similarity_delta"], "pass_component": route_score["pass_component"]})
    route_mean = float(np.mean([row["route_r2"] for row in route_rows]))
    worst_delta = float(min(row["delta_r2"] for row in route_rows))
    full_pass = bool(route_mean - baseline_mean >= 0.002 and worst_delta >= -0.003 and all(row["pass_component"] for row in route_rows if row["changed_rows"] > 0))
    audit = {"schema_version": "ppp.round2.direction-route-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C017-20260803-1840-similarity-gated-route-v2", "official_inputs": inputs, "direction_rule": "apply positive correction only inside the fixed C017 bins", "baseline_grouped_mean_r2": baseline_mean, "route_grouped_mean_r2": route_mean, "route_mean_gain": route_mean - baseline_mean, "worst_target_delta": worst_delta, "routes": ROUTES, "targets": reports, "route": route_rows, "full_pass": full_pass, "decision": "prospective_route_pass" if full_pass else "rejected_direction_route_gate", "elapsed_seconds": float(time.time() - started)}
    pd.DataFrame(route_rows).to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.direction-route.v1", "seed": 2026, "folds": 5, "routes": ROUTES, "direction_rule": "positive correction only", "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C018 direction-consistent route decision\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "metrics.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "baseline_grouped_mean_r2": baseline_mean, "route_grouped_mean_r2": route_mean, "route_mean_gain": route_mean - baseline_mean, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
