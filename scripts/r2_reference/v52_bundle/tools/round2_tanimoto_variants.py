#!/usr/bin/env python3
"""Clean target-specific Tanimoto kernel variant comparison for sparse targets."""

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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = ("eps", "ei", "nc", "eea")
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
C013_ID = "R2-C013-20260803-1804-target-tree-zoo-v2"
ARM_SETTINGS = {
    target: [(1, 5, 0.05), (2, 5, 0.05), (2, 15, 0.05), (3, 5, 0.05), (3, 15, 0.05), (3, 15, 0.01)]
    for target in TARGETS
}
ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


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


def group_bootstrap_lower(delta: np.ndarray, groups: np.ndarray) -> float:
    if len(delta) < 5:
        return 0.0
    unique = np.unique(groups)
    grouped = {group: delta[groups == group] for group in unique}
    rng = np.random.default_rng(2026)
    means = []
    for _ in range(500):
        selected = rng.choice(unique, size=len(unique), replace=True)
        means.append(float(np.mean(np.concatenate([grouped[group] for group in selected]))))
    return float(np.quantile(means, 0.025))


def score_blend(y: np.ndarray, baseline: np.ndarray, arm: np.ndarray, groups: np.ndarray, folds: np.ndarray, nearest: np.ndarray, alpha: float) -> dict[str, Any]:
    pred = baseline + float(alpha) * (arm - baseline)
    fold_rows = []
    for fold in range(5):
        selected = folds == fold
        fold_rows.append({"fold": fold, "rows": int(np.sum(selected)), "delta_r2": r2(y[selected], pred[selected]) - r2(y[selected], baseline[selected])})
    low_rows = {}
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        if int(np.sum(selected)) < 5:
            continue
        low_rows[name] = {"rows": int(np.sum(selected)), "delta_r2": r2(y[selected], pred[selected]) - r2(y[selected], baseline[selected])}
    delta = pred - baseline
    result = {
        "alpha": float(alpha),
        "r2": r2(y, pred),
        "delta_r2": r2(y, pred) - r2(y, baseline),
        "positive_folds": int(sum(row["delta_r2"] > 0.0 for row in fold_rows)),
        "worst_fold_delta": float(min(row["delta_r2"] for row in fold_rows)),
        "bootstrap_lower": group_bootstrap_lower(delta, groups),
        "low_similarity": low_rows,
        "min_low_similarity_delta": min((row["delta_r2"] for row in low_rows.values()), default=0.0),
        "folds": fold_rows,
    }
    result["pass_component"] = bool(result["delta_r2"] >= 0.01 and result["positive_folds"] >= 4 and result["bootstrap_lower"] > 0.0 and result["min_low_similarity_delta"] >= 0.0)
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
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    baseline_fps = reference.morgan_bits(molecules, 2, 4096)
    fingerprint_by_radius = {radius: reference.morgan_bits(molecules, radius, 4096) for radius in (1, 2, 3)}
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    c013_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C013_ID / "metrics.json").read_text(encoding="utf-8"))
    baseline_grouped = {target: float(c013_report["targets"][target]["baseline_r2"]) for target in reference.TARGETS}
    baseline_mean = float(np.mean(list(baseline_grouped.values())))
    reports = {}
    selected_rows = []
    selected_deltas: dict[str, float] = {}
    for target_index, target in enumerate(TARGETS):
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        groups = frame["canonical"].to_numpy(object)
        indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[indices] = y
        folds = folds_for(groups)
        target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        baseline = np.full(len(y), np.nan, dtype=np.float64)
        weights = c001_report["validation"]["target_reports"][target]["blend_weights"]
        blend_weights = np.asarray([weights[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)
        intercept = float(c001_report["validation"]["target_reports"][target]["blend_intercept"])
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            base = reference.predict_base_models(target_dense, sparse_parts, baseline_fps, y_global, indices[train_rows], indices[validation_rows], reference.DEFAULT_CONFIG, target)
            baseline[validation_rows] = base @ blend_weights + intercept
        nearest = nearest_similarity([baseline_fps[index] for index in indices], folds)
        arms = {}
        safe_candidates = []
        for radius, k, kernel_alpha in ARM_SETTINGS[target]:
            fps = fingerprint_by_radius[radius]
            arm = np.full(len(y), np.nan, dtype=np.float64)
            for fold in range(5):
                train_rows = np.flatnonzero(folds != fold)
                validation_rows = np.flatnonzero(folds == fold)
                arm[validation_rows] = reference.tanimoto_prediction(fps, y_global, indices[train_rows], indices[validation_rows], k=k, krr_alpha=kernel_alpha)
            arm_name = f"radius{radius}_k{k}_alpha{kernel_alpha}"
            scores = [score_blend(y, baseline, arm, groups, folds, nearest, alpha) for alpha in ALPHAS]
            for score_item in scores:
                score_item["arm"] = arm_name
            arms[arm_name] = scores
            safe_candidates.extend([item for item in scores if item["pass_component"]])
        selected = max(safe_candidates, key=lambda item: item["r2"]) if safe_candidates else max((item for score_list in arms.values() for item in score_list if item["alpha"] == 0.0), key=lambda item: item["r2"])
        selected["selected_from_passing_grid"] = bool(safe_candidates)
        selected["baseline_r2"] = r2(y, baseline)
        selected["hypothetical_mean_r2"] = baseline_mean + (selected["r2"] - selected["baseline_r2"]) / 7.0
        reports[target] = {"rows": int(len(y)), "baseline_r2": selected["baseline_r2"], "arms": arms, "selected": selected}
        selected_deltas[target] = float(selected["r2"] - selected["baseline_r2"])
        selected_rows.append({"target": target, "arm": selected["arm"], "alpha": selected["alpha"], "baseline_r2": selected["baseline_r2"], "r2": selected["r2"], "delta_r2": selected["r2"] - selected["baseline_r2"], "positive_folds": selected["positive_folds"], "bootstrap_lower": selected["bootstrap_lower"], "min_low_similarity_delta": selected["min_low_similarity_delta"], "pass_component": selected.get("pass_component", False)})
    route_mean = baseline_mean + float(np.sum(list(selected_deltas.values()))) / 7.0
    passing = [row for row in selected_rows if row["pass_component"]]
    full_pass = bool(route_mean - baseline_mean >= 0.002 and len(passing) > 0)
    audit = {"schema_version": "ppp.round2.tanimoto-variants-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-O001-20260803-1905-score-C019", "official_inputs": inputs, "targets": reports, "selected": selected_rows, "baseline_grouped_mean_r2": baseline_mean, "route_grouped_mean_r2": route_mean, "route_mean_gain": route_mean - baseline_mean, "full_pass": full_pass, "decision": "component_pass" if full_pass else "rejected_tanimoto_gate", "elapsed_seconds": float(time.time() - started)}
    pd.DataFrame(selected_rows).to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.tanimoto-variants.v1", "seed": 2026, "folds": 5, "targets": list(TARGETS), "arm_settings": ARM_SETTINGS, "alphas": list(ALPHAS), "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C020 Tanimoto variant decision\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "metrics.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "route_grouped_mean_r2": route_mean, "route_mean_gain": route_mean - baseline_mean, "selected": selected_rows, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
