#!/usr/bin/env python3
"""Clean, grouped, fold-local pooled multi-target residual diagnostic."""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as panel_tools
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
SEED = 2026
WEIGHT = 0.5
PARENT_ID = "R2-C050-20260803-2130-mixed-c001-gap-components-v7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("incomplete grouped folds")
    return result


def finite_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, prediction))


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(finite_r2(y[rows], candidate[rows]) - finite_r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def build_cross_features(
    target_codes: np.ndarray,
    sample_groups: np.ndarray,
    train_rows: np.ndarray,
    y_all: np.ndarray,
    all_target_codes: np.ndarray,
    all_groups: np.ndarray,
    train_groups: set[str],
) -> np.ndarray:
    """Build label-derived features using only labels from training groups."""
    values = np.full((len(target_codes), len(TARGETS)), np.nan, dtype=np.float64)
    available = np.zeros((len(target_codes), len(TARGETS)), dtype=np.float64)
    lookup: dict[tuple[str, int], float] = {}
    for index in train_rows:
        group = str(all_groups[index])
        if group not in train_groups:
            continue
        key = (group, int(all_target_codes[index]))
        lookup[key] = float(y_all[index])
    for local, _ in enumerate(sample_groups):
        group = str(sample_groups[local])
        own = int(target_codes[local])
        for code in range(len(TARGETS)):
            if code == own:
                continue
            value = lookup.get((group, code))
            if value is not None and np.isfinite(value):
                values[local, code] = value
                available[local, code] = 1.0
    return np.hstack([values, available])


def standardize_cross_features(train_x: np.ndarray, prediction_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_x = train_x.copy()
    prediction_x = prediction_x.copy()
    for column in range(len(TARGETS)):
        mask = np.isfinite(train_x[:, column])
        if np.any(mask):
            mean = float(np.mean(train_x[mask, column]))
            std = float(np.std(train_x[mask, column]))
            if std < 1.0e-12:
                std = 1.0
            train_x[mask, column] = (train_x[mask, column] - mean) / std
            valid = np.isfinite(prediction_x[:, column])
            prediction_x[valid, column] = (prediction_x[valid, column] - mean) / std
    return train_x, prediction_x


def shared_features(
    dense: np.ndarray,
    molecule_rows: np.ndarray,
    target_codes: np.ndarray,
    sample_groups: np.ndarray,
    train_rows: np.ndarray,
    all_target_codes: np.ndarray,
    y_all: np.ndarray,
    all_groups: np.ndarray,
    train_groups: set[str],
) -> np.ndarray:
    cross = build_cross_features(target_codes, sample_groups, train_rows, y_all, all_target_codes, all_groups, train_groups)
    target_onehot = np.zeros((len(molecule_rows), len(TARGETS)), dtype=np.float64)
    target_onehot[np.arange(len(molecule_rows)), target_codes] = 1.0
    output = np.hstack([dense[molecule_rows], cross, target_onehot]).astype(np.float64, copy=False)
    output[~np.isfinite(output)] = np.nan
    return output


def fit_fold_model(x_train: np.ndarray, y_train: np.ndarray, x_prediction: np.ndarray, target_codes: np.ndarray) -> np.ndarray:
    train_x = np.asarray(x_train, dtype=np.float64).copy()
    prediction_x = np.asarray(x_prediction, dtype=np.float64).copy()
    finite = np.isfinite(train_x)
    keep = (np.sum(finite, axis=0) >= 2) & (np.nanmax(np.where(finite, train_x, np.nan), axis=0, initial=0.0) - np.nanmin(np.where(finite, train_x, np.nan), axis=0, initial=0.0) > 1.0e-12)
    if not np.any(keep):
        raise RuntimeError("no variable pooled features remained")
    model = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=0.10,
        random_state=SEED,
    )
    model.fit(train_x[:, keep], y_train)
    return np.asarray(model.predict(prediction_x[:, keep]), dtype=np.float64)


def target_panel(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> tuple[dict[str, Any], float | None]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []
    bins = (("similarity_lt_0.30", 0.0, 0.30), ("similarity_0.30_0.50", 0.30, 0.50), ("similarity_0.50_0.70", 0.50, 0.70), ("similarity_ge_0.70", 0.70, 1.01))
    for name, lower, upper in bins:
        selected = (similarity >= lower) & (similarity < upper)
        row_count = int(np.sum(selected))
        group_count = int(np.unique(groups[selected]).size)
        eligible = row_count >= 20 and group_count >= 5
        delta = finite_r2(y[selected], candidate[selected]) - finite_r2(y[selected], parent[selected]) if eligible else None
        panels[name] = {"rows": row_count, "groups": group_count, "eligible": bool(eligible), "delta_r2": delta}
        if delta is not None:
            deltas.append(float(delta))
    scaffold_deltas: list[float] = []
    for scaffold in np.unique(scaffolds):
        selected = scaffolds == scaffold
        if int(np.sum(selected)) >= 10 and int(np.unique(groups[selected]).size) >= 3 and not np.isclose(np.var(y[selected]), 0.0):
            scaffold_deltas.append(finite_r2(y[selected], candidate[selected]) - finite_r2(y[selected], parent[selected]))
    panels["scaffold_groups_ge_3"] = {"evaluated": len(scaffold_deltas), "minimum_delta_r2": min(scaffold_deltas) if scaffold_deltas else None}
    if scaffold_deltas:
        deltas.append(min(scaffold_deltas))
    return panels, (min(deltas) if deltas else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    pooled = pooled.reset_index(drop=True)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)

    parent_dir = root / "experiments" / "CLEAN_OFFICIAL_ONLY" / PARENT_ID
    parent_oof = pd.read_csv(parent_dir / "oof_predictions.csv")
    parent_key = ["target_type", "canonical", "target"]
    parent_oof = parent_oof[parent_key + ["candidate_prediction"]].rename(columns={"candidate_prediction": "parent_prediction"})
    pooled = pooled.merge(parent_oof, on=parent_key, how="left", validate="one_to_one")
    if pooled["parent_prediction"].isna().any():
        raise RuntimeError("C050-v7 OOF parent alignment failed")
    parent_test = pd.read_csv(parent_dir / "predictions.csv")
    if not np.array_equal(parent_test["id"].to_numpy(dtype=int), test["id"].to_numpy(dtype=int)):
        raise RuntimeError("C050-v7 test-id order mismatch")
    if len(parent_test) != len(test) or not np.isfinite(parent_test["target"].to_numpy(float)).all():
        raise RuntimeError("C050-v7 test parent invalid")

    pooled_targets = pooled["target_type"].astype(str).to_numpy(object)
    target_codes_all = np.asarray([TARGETS.index(value) for value in pooled_targets], dtype=np.int64)
    pooled_groups = np.asarray([plumbing.no_stereo(value) for value in pooled["canonical"]], dtype=object)
    pooled_keys = pooled["canonical"].astype(str).to_numpy(object)
    pooled_rows = np.asarray([key_to_index[value] for value in pooled_keys], dtype=np.int64)
    y_all = pooled["target"].to_numpy(float)
    parent_all = pooled["parent_prediction"].to_numpy(float)
    folds = folds_for(pooled_groups)
    candidate_all = np.full(len(pooled), np.nan, dtype=np.float64)
    fold_meta: list[dict[str, Any]] = []

    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_group_set = set(pooled_groups[training].tolist())
        x_train = shared_features(dense, pooled_rows[training], target_codes_all[training], pooled_groups[training], training, target_codes_all, y_all, pooled_groups, train_group_set)
        x_validation = shared_features(dense, pooled_rows[validation], target_codes_all[validation], pooled_groups[validation], training, target_codes_all, y_all, pooled_groups, train_group_set)
        residual = y_all[training] - parent_all[training]
        residual_z = np.zeros(len(training), dtype=np.float64)
        means: dict[int, tuple[float, float]] = {}
        for code in range(len(TARGETS)):
            selected = target_codes_all[training] == code
            mean = float(np.mean(residual[selected])) if np.any(selected) else 0.0
            std = float(np.std(residual[selected])) if np.any(selected) else 1.0
            std = std if std > 1.0e-8 else 1.0
            residual_z[selected] = (residual[selected] - mean) / std
            means[code] = (mean, std)
        correction_z = fit_fold_model(x_train, residual_z, x_validation, target_codes_all[validation])
        correction = np.empty(len(validation), dtype=np.float64)
        for code in range(len(TARGETS)):
            selected = target_codes_all[validation] == code
            mean, std = means[code]
            correction[selected] = correction_z[selected] * std + mean
        candidate_all[validation] = parent_all[validation] + WEIGHT * correction
        fold_meta.append({"fold": fold, "train_rows": int(len(training)), "validation_rows": int(len(validation)), "shared_features": int(x_train.shape[1]), "train_groups": int(len(train_group_set))})

    if not np.isfinite(candidate_all).all():
        raise RuntimeError("non-finite pooled OOF candidate")
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    similarity_all = np.full(len(pooled), np.nan, dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in pooled_rows[training]]
        for row in validation:
            similarity_all[row] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[pooled_rows[row]], train_fps))
    scaffolds_all = np.asarray([plumbing.scaffold(value) for value in pooled["canonical"]], dtype=object)

    reports: dict[str, Any] = {}
    test_candidates = parent_test["target"].to_numpy(dtype=float).copy()
    test_target = test["target_type"].astype(str).to_numpy(object)
    test_rows = np.asarray([key_to_index[value] for value in test["canonical"]], dtype=np.int64)
    all_train_rows = np.arange(len(pooled), dtype=np.int64)
    all_groups = set(pooled_groups.tolist())
    x_all = shared_features(dense, pooled_rows, target_codes_all, pooled_groups, all_train_rows, target_codes_all, y_all, pooled_groups, all_groups)
    test_codes = np.asarray([TARGETS.index(value) for value in test_target], dtype=np.int64)
    test_groups = np.asarray([plumbing.no_stereo(value) for value in test["canonical"]], dtype=object)
    x_test = shared_features(dense, test_rows, test_codes, test_groups, all_train_rows, target_codes_all, y_all, pooled_groups, all_groups)
    residual_all = y_all - parent_all
    residual_z_all = np.zeros(len(residual_all), dtype=np.float64)
    full_means: dict[int, tuple[float, float]] = {}
    for code in range(len(TARGETS)):
        selected = target_codes_all == code
        mean = float(np.mean(residual_all[selected]))
        std = float(np.std(residual_all[selected])); std = std if std > 1.0e-8 else 1.0
        residual_z_all[selected] = (residual_all[selected] - mean) / std
        full_means[code] = (mean, std)
    test_correction_z = fit_fold_model(x_all, residual_z_all, x_test, test_codes)
    test_correction = np.empty(len(test), dtype=np.float64)
    for code in range(len(TARGETS)):
        selected = test_codes == code
        mean, std = full_means[code]
        test_correction[selected] = test_correction_z[selected] * std + mean
    test_candidates += WEIGHT * test_correction

    for target in TARGETS:
        selected = pooled_targets == target
        y = y_all[selected]; parent = parent_all[selected]; candidate = candidate_all[selected]; groups = pooled_groups[selected]; similarity = similarity_all[selected]; scaffolds = scaffolds_all[selected]
        folds_local = folds[selected]
        parent_r2 = finite_r2(y, parent); candidate_r2 = finite_r2(y, candidate); delta = candidate_r2 - parent_r2
        fold_rows = []
        for fold in range(5):
            rows = folds_local == fold
            fold_rows.append({"fold": fold, "rows": int(np.sum(rows)), "parent_r2": finite_r2(y[rows], parent[rows]), "candidate_r2": finite_r2(y[rows], candidate[rows]), "delta_r2": finite_r2(y[rows], candidate[rows]) - finite_r2(y[rows], parent[rows])})
        panels, minimum_panel = target_panel(y, parent, candidate, groups, similarity, scaffolds)
        lower = bootstrap_lower(y, parent, candidate, groups)
        positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        adjacent = {other: finite_r2(y_all[pooled_targets == other], candidate_all[pooled_targets == other]) - finite_r2(y_all[pooled_targets == other], parent_all[pooled_targets == other]) for other in TARGETS}
        gates = {"gain_pass": delta >= 0.01, "fold_pass": positive_folds >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel is not None and minimum_panel >= 0.0, "adjacent_loss_pass": all(value >= -0.003 for name, value in adjacent.items() if name != target)}
        reports[target] = {"rows": int(len(y)), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": positive_folds, "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panels, "minimum_panel_delta": minimum_panel, "adjacent_target_deltas": adjacent, "gates": gates, "decision": "pass_component_gate" if all(gates.values()) else "rejected_component_gate"}

    test_output = pd.DataFrame({"id": test["id"].astype(int), "target_type": test_target, "parent_prediction": parent_test["target"].to_numpy(float), "candidate_prediction": test_candidates})
    if len(test_output) != 4940 or test_output["id"].duplicated().any() or not np.array_equal(test_output["id"].to_numpy(), test["id"].to_numpy()) or not np.isfinite(test_candidates).all():
        raise RuntimeError("pooled test output contract failed")
    test_output.to_csv(run_dir / "test_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": pooled["canonical"].astype(str), "target_type": pooled_targets, "target": y_all, "parent": parent_all, "candidate": candidate_all, "group": pooled_groups, "scaffold": scaffolds_all, "nearest_similarity": similarity_all, "outer_fold": folds}).to_csv(run_dir / "oof_predictions.csv", index=False)
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS])); mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS])); mean_delta = mean_candidate - mean_parent
    passing_targets = [target for target in TARGETS if all(reports[target]["gates"].values())]
    gates = {"mean_gain_pass": mean_delta >= 0.002, "test_rows_pass": len(test_output) == 4940, "all_finite_pass": bool(np.isfinite(test_candidates).all()), "parent_alignment_pass": True}
    source_names = ("round2_c087_pooled_multitask_v7_residual.py", "initial_reference_pipeline.py", "round2_c063_egb_endpoint_conjugation_residual.py", "round2_eea_cross_target_oof_residual_stack.py")
    source_hashes = {name: sha256_file(root / "tools" / name) for name in source_names}
    report = {"schema_version": "ppp.round2.c087.pooled-multitask-v7-residual.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": PARENT_ID, "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "kaggle_compute": False, "pooled_rows": int(len(pooled)), "shared_features": int(x_all.shape[1]), "folds": fold_meta, "targets": reports, "passing_targets": passing_targets, "parent_mean_r2": mean_parent, "candidate_mean_r2": mean_candidate, "mean_delta_r2": mean_delta, "full_gates": gates, "decision": "pass_component_gate" if passing_targets and all(gates.values()) else "rejected_component_gate", "source_hashes": source_hashes, "elapsed_seconds": time.time() - started}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": report["schema_version"], "seed": SEED, "targets": TARGETS, "weight": WEIGHT, "model": "HistGradientBoostingRegressor", "outer": "canonical no-stereo GroupKFold(5)", "cross_property_mask": "held-out canonical groups excluded", "parent": PARENT_ID, "official_only": True, "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Parent mean `{mean_parent:.12f}`, candidate mean `{mean_candidate:.12f}`, delta `{mean_delta:+.12f}`. Passing components: `{', '.join(passing_targets) or 'none'}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    lines = [f"{sha256_file(path)}  {path.relative_to(run_dir)}" for path in manifest_paths]
    lines.extend(f"{digest}  SOURCE tools/{name}" for name, digest in source_hashes.items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_mean_r2": mean_parent, "candidate_mean_r2": mean_candidate, "mean_delta_r2": mean_delta, "passing_targets": passing_targets, "targets": {target: {"delta_r2": reports[target]["delta_r2"], "positive_folds": reports[target]["positive_folds"], "bootstrap": reports[target]["group_bootstrap_lower"]} for target in TARGETS}, "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
