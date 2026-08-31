#!/usr/bin/env python3
"""Bounded official-only Tg mobility/free-volume carrier diagnostic.

This is a Round 2 adaptation of the Round 1 Tg mobility/family evidence.  It
never reads Round 1 predictions or external_labels and only compares a target-specific
carrier with the frozen C001 OOF predictions on the official pooled labels.
"""

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

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "tg"
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
ARMS = ("mobility_histgb", "mobility_lightgbm", "mobility_extra_trees", "mobility_ridge")


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
    unique = np.unique(groups)
    grouped = {group: delta[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(500, dtype=np.float64)
    for draw in range(len(means)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        means[draw] = float(np.mean(np.concatenate([grouped[group] for group in selected])))
    return float(np.quantile(means, 0.025))


def preprocess(matrix: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    train_imputed, validation_imputed, train_scaled, validation_scaled = reference.fit_dense_preprocessor(
        matrix,
        train_indices,
        validation_indices,
        absolute_limit=float(reference.DEFAULT_CONFIG["dense_abs_limit"]),
    )
    keep = np.ptp(train_imputed, axis=0) > 1.0e-12
    if not np.any(keep):
        raise RuntimeError("no nonconstant mobility features remained")
    return train_imputed[:, keep], validation_imputed[:, keep], train_scaled[:, keep], validation_scaled[:, keep], int(np.sum(keep))


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
    base_desc, base_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([base_desc, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)

    tools_dir = root.parent / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(tools_dir))
    import polymer_c70_tg_family_norm_mobility as mobility  # noqa: PLC0415
    import polymer_official_train_eval_loop as rich  # noqa: PLC0415

    family_sparse, family_report = mobility.build_family_norm_matrix(keys, run_dir)
    family_dense = family_sparse.toarray().astype(np.float64, copy=False)
    mobility_dense, mobility_names, mobility_report = rich.mobility_feature_matrix(molecules)
    augmented_dense = np.hstack([base_dense, family_dense, mobility_dense]).astype(np.float64, copy=False)
    augmented_dense[~np.isfinite(augmented_dense)] = np.nan
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)

    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    groups = frame["canonical"].to_numpy(object)
    indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    folds = folds_for(groups)
    target_dense = reference.target_dense_features(augmented_dense, cross_values, cross_available, TARGET)
    baseline_dense = reference.target_dense_features(base_dense, cross_values, cross_available, TARGET)
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    baseline = np.full(len(y), np.nan, dtype=np.float64)
    predictions = {name: np.full(len(y), np.nan, dtype=np.float64) for name in ARMS}
    fold_reports = {name: [] for name in ARMS}
    retained_features: list[int] = []

    for fold in range(5):
        train_rows = np.flatnonzero(folds != fold)
        validation_rows = np.flatnonzero(folds == fold)
        base = reference.predict_base_models(
            baseline_dense,
            sparse_parts,
            fingerprints,
            y_global,
            indices[train_rows],
            indices[validation_rows],
            reference.DEFAULT_CONFIG,
            TARGET,
        )
        weights = c001_report["validation"]["target_reports"][TARGET]["blend_weights"]
        blend_weights = np.asarray([weights[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)
        baseline[validation_rows] = base @ blend_weights + float(c001_report["validation"]["target_reports"][TARGET]["blend_intercept"])

        train_x, validation_x, scaled_train, scaled_validation, kept = preprocess(target_dense, indices[train_rows], indices[validation_rows])
        retained_features.append(kept)
        models = [
            HistGradientBoostingRegressor(max_iter=320, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=12, l2_regularization=0.10, random_state=2026 + fold),
            lgb.LGBMRegressor(objective="regression", n_estimators=350, learning_rate=0.03, num_leaves=25, min_child_samples=16, subsample=0.86, subsample_freq=1, colsample_bytree=0.74, reg_lambda=1.2, random_state=2030 + fold, n_jobs=1, verbosity=-1),
            ExtraTreesRegressor(n_estimators=220, max_features=0.55, min_samples_leaf=3, random_state=2040 + fold, n_jobs=2),
            Ridge(alpha=30.0),
        ]
        matrices = [(train_x, validation_x), (train_x, validation_x), (train_x, validation_x), (scaled_train, scaled_validation)]
        for name, model, (model_train, model_validation) in zip(ARMS, models, matrices, strict=True):
            model.fit(model_train, y[train_rows])
            prediction = np.asarray(model.predict(model_validation), dtype=np.float64)
            predictions[name][validation_rows] = prediction
            fold_reports[name].append({
                "fold": fold,
                "rows": int(len(validation_rows)),
                "baseline_r2": r2(y[validation_rows], baseline[validation_rows]),
                "model_r2": r2(y[validation_rows], prediction),
                "delta_r2": r2(y[validation_rows], prediction) - r2(y[validation_rows], baseline[validation_rows]),
            })

    nearest = nearest_similarity([fingerprints[index] for index in indices], folds)
    baseline_r2 = r2(y, baseline)
    baseline_mean = float(np.mean([item["selected_oof_r2"] for item in c001_report["validation"]["target_reports"].values()]))
    report: dict[str, Any] = {
        "rows": int(len(y)),
        "baseline_r2": baseline_r2,
        "baseline_clean_mean_r2": baseline_mean,
        "feature_count": int(augmented_dense.shape[1]),
        "retained_features_per_fold": retained_features,
        "features": {"base": len(base_names) + len(physical_names), "family_norm": int(family_dense.shape[1]), "mobility": len(mobility_names)},
        "feature_reports": {"family_norm": family_report, "mobility": mobility_report},
        "arms": {},
    }
    for name in ARMS:
        prediction = predictions[name]
        delta = prediction - baseline
        arm: dict[str, Any] = {"r2": r2(y, prediction), "delta_r2": r2(y, prediction) - baseline_r2, "hypothetical_clean_mean_r2": baseline_mean + (r2(y, prediction) - baseline_r2) / 7.0, "folds": fold_reports[name]}
        for name_bin, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
            selected = (nearest >= lower) & (nearest < upper)
            if int(np.sum(selected)) < 5:
                continue
            arm.setdefault("low_similarity", {})[name_bin] = {"rows": int(np.sum(selected)), "baseline_r2": r2(y[selected], baseline[selected]), "model_r2": r2(y[selected], prediction[selected]), "delta_r2": r2(y[selected], prediction[selected]) - r2(y[selected], baseline[selected])}
        low_values = [item["delta_r2"] for item in arm.get("low_similarity", {}).values()]
        arm["positive_folds"] = int(sum(item["delta_r2"] > 0 for item in fold_reports[name]))
        arm["bootstrap_lower"] = group_bootstrap_lower(delta, groups)
        arm["min_low_similarity_delta"] = min(low_values) if low_values else None
        arm["pass"] = bool(arm["delta_r2"] >= 0.01 and arm["positive_folds"] >= 4 and arm["bootstrap_lower"] > 0.0 and (not low_values or arm["min_low_similarity_delta"] >= 0.0) and arm["hypothetical_clean_mean_r2"] > baseline_mean)
        report["arms"][name] = arm
    report["selected_arm"] = max(ARMS, key=lambda name: report["arms"][name]["r2"])
    passing = [name for name in ARMS if report["arms"][name]["pass"]]
    audit = {"schema_version": "ppp.round2.tg-mobility-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C014-20260803-1812-egc-electronic-carrier", "official_inputs": inputs, "target": TARGET, "report": report, "passing_arms": passing, "decision": "component_pass" if passing else "rejected_component_gate", "elapsed_seconds": float(time.time() - started)}
    rows = [{"target": TARGET, "arm": name, "r2": report["arms"][name]["r2"], "delta_r2": report["arms"][name]["delta_r2"], "hypothetical_clean_mean_r2": report["arms"][name]["hypothetical_clean_mean_r2"], "positive_folds": report["arms"][name]["positive_folds"], "bootstrap_lower": report["arms"][name]["bootstrap_lower"], "min_low_similarity_delta": report["arms"][name]["min_low_similarity_delta"], "pass": report["arms"][name]["pass"]} for name in ARMS]
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.tg-mobility.v1", "seed": 2026, "folds": 5, "target": TARGET, "arms": list(ARMS), "feature_count": int(augmented_dense.shape[1]), "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"lightgbm={lgb.__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C015 Tg mobility/free-volume carrier decision\n\nDecision: **{audit['decision']}**. No candidate was routed or local_eval-scored in this component diagnostic.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "metrics.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "selected_arm": report["selected_arm"], "passing_arms": passing, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
