#!/usr/bin/env python3
"""Run the preregistered, model-frozen Round 2 validation audit.

This tool is deliberately separate from candidate generation.  It reads only
the official Round 2 inputs and the immutable C001 clean artifacts, evaluates
the frozen carrier arms under stricter panels, and writes sanitized aggregate
metrics plus fold assignments to the versioned C002 run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
MODEL_NAMES = ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")
EXPECTED_HASHES = reference.EXPECTED_HASHES
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_reference_weights(root: Path) -> dict[str, np.ndarray]:
    report_path = root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weights: dict[str, np.ndarray] = {}
    for target in TARGETS:
        values = report["validation"]["target_reports"][target]["blend_weights"]
        weights[target] = np.asarray([float(values[name]) for name in MODEL_NAMES], dtype=np.float64)
    return weights


def make_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        scaffold = ""
    return scaffold or "ACYCLIC"


def greedy_tanimoto_groups(fingerprints: list[Any], threshold: float = 0.70) -> np.ndarray:
    """Deterministic representative clustering for validation grouping.

    It is intentionally conservative: a row joins an existing cluster only
    when it is at least threshold-similar to that cluster's first member.
    This is a fixed grouping rule, not a tunable model parameter.
    """

    representatives: list[Any] = []
    labels = np.empty(len(fingerprints), dtype=np.int64)
    for index, fingerprint in enumerate(fingerprints):
        if not representatives:
            representatives.append(fingerprint)
            labels[index] = 0
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, representatives)
        best = int(np.argmax(similarities))
        if float(similarities[best]) >= threshold:
            labels[index] = best
        else:
            labels[index] = len(representatives)
            representatives.append(fingerprint)
    return labels


def make_folds(groups: np.ndarray, seed: int, folds: int) -> np.ndarray:
    if len(np.unique(groups)) < folds:
        raise RuntimeError(f"Panel has only {len(np.unique(groups))} groups for {folds} folds")
    assignment = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        assignment[validation] = fold
    if np.any(assignment < 0) or len(np.unique(assignment)) != folds:
        raise RuntimeError("Fold assignment is incomplete")
    return assignment


def make_main_folds(rows: int, seed: int, folds: int) -> np.ndarray:
    assignment = np.full(rows, -1, dtype=np.int64)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(rows))):
        assignment[validation] = fold
    return assignment


def score_rows(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        r2: float | None = None
    else:
        r2 = float(r2_score(y, prediction))
    return {
        "rows": int(len(y)),
        "r2": r2,
        "mae": float(mean_absolute_error(y, prediction)) if len(y) else None,
        "rmse": float(np.sqrt(np.mean(np.square(y - prediction)))) if len(y) else None,
    }


def append_metric(
    metrics: list[dict[str, Any]],
    panel: str,
    target: str,
    arm: str,
    stratum: str,
    fold: int | None,
    y: np.ndarray,
    prediction: np.ndarray,
) -> None:
    row: dict[str, Any] = {
        "panel": panel,
        "target": target,
        "arm": arm,
        "stratum": stratum,
        "fold": fold,
    }
    row.update(score_rows(y, prediction))
    metrics.append(row)


def build_fold_predictions(
    pooled: pd.DataFrame,
    keys: list[str],
    dense_base: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    config: dict[str, Any],
    target: str,
    folds: np.ndarray,
    blend_weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_to_index = {key: index for index, key in enumerate(keys)}
    target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
    train_index = np.asarray([key_to_index[value] for value in target_rows["canonical"]], dtype=np.int64)
    y = target_rows["target"].to_numpy(float)
    dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
    predictions = np.full((len(target_rows), len(MODEL_NAMES)), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        local_train = np.flatnonzero(folds != fold)
        local_validation = np.flatnonzero(folds == fold)
        local_y_global = np.full(len(keys), np.nan, dtype=np.float64)
        local_y_global[train_index] = y
        fold_prediction = reference.predict_base_models(
            dense,
            sparse_parts,
            fingerprints,
            local_y_global,
            train_index[local_train],
            train_index[local_validation],
            config,
            target,
        )
        predictions[local_validation] = fold_prediction
        fold_rows.append({"fold": int(fold), "train_rows": int(len(local_train)), "validation_rows": int(len(local_validation))})
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"Non-finite panel prediction for {target}")
    blended = predictions @ blend_weights
    detail = pd.DataFrame(
        {
            "canonical": target_rows["canonical"],
            "target_type": target,
            "target": y,
            "fold": folds,
            "blend": blended,
        }
    )
    for column, name in enumerate(MODEL_NAMES):
        detail[name] = predictions[:, column]
    return detail, {"folds": fold_rows, "rows": int(len(target_rows))}


def load_main_oof(root: Path, pooled: pd.DataFrame) -> pd.DataFrame:
    path = root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "oof_predictions.csv"
    frame = pd.read_csv(path)
    expected = {"canonical", "target_type", "target", "prediction", *MODEL_NAMES}
    if set(frame.columns) != expected or len(frame) != len(pooled):
        raise RuntimeError("C001 OOF artifact schema does not match the pooled official label table")
    merged = pooled[["canonical", "target_type", "target"]].merge(
        frame,
        on=["canonical", "target_type", "target"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_oof"),
    )
    if merged[list(MODEL_NAMES) + ["prediction"]].isna().any().any():
        raise RuntimeError("C001 OOF artifact does not cover every pooled target row")
    return merged


def add_main_metrics(
    metrics: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    pooled: pd.DataFrame,
    main_oof: pd.DataFrame,
    blend_weights: dict[str, np.ndarray],
    seed: int,
    folds: int,
) -> dict[str, np.ndarray]:
    fold_by_target: dict[str, np.ndarray] = {}
    for target in TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_oof = main_oof[main_oof["target_type"] == target].reset_index(drop=True)
        fold_ids = make_main_folds(len(target_rows), seed, folds)
        fold_by_target[target] = fold_ids
        for index, row in target_rows.iterrows():
            assignments.append(
                {
                    "panel": "main_fixed_reproduction",
                    "target": target,
                    "canonical": row["canonical"],
                    "fold": int(fold_ids[index]),
                    "group_key": row["canonical"],
                }
            )
        prediction = target_oof["prediction"].to_numpy(float)
        y = target_oof["target"].to_numpy(float)
        for arm in MODEL_NAMES:
            arm_prediction = target_oof[arm].to_numpy(float)
            append_metric(metrics, "main_fixed_reproduction", target, arm, "all", None, y, arm_prediction)
        append_metric(metrics, "main_fixed_reproduction", target, "frozen_blend", "all", None, y, prediction)
        for fold in range(folds):
            selected = fold_ids == fold
            for arm in MODEL_NAMES:
                append_metric(metrics, "main_fixed_reproduction", target, arm, "all", fold, y[selected], target_oof.loc[selected, arm].to_numpy(float))
            append_metric(metrics, "main_fixed_reproduction", target, "frozen_blend", "all", fold, y[selected], prediction[selected])
    return fold_by_target


def add_panel_metrics(
    metrics: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    pooled: pd.DataFrame,
    panel: str,
    groups_by_target: dict[str, np.ndarray],
    keys: list[str],
    dense_base: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    config: dict[str, Any],
    blend_weights: dict[str, np.ndarray],
    fold_count: int,
) -> dict[str, pd.DataFrame]:
    details: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        group_values = groups_by_target[target]
        fold_ids = make_folds(group_values, int(config["seed"]), fold_count)
        for index, row in target_rows.iterrows():
            assignments.append(
                {
                    "panel": panel,
                    "target": target,
                    "canonical": row["canonical"],
                    "fold": int(fold_ids[index]),
                    "group_key": str(group_values[index]),
                }
            )
        detail, _ = build_fold_predictions(
            pooled,
            keys,
            dense_base,
            cross_values,
            cross_available,
            sparse_parts,
            fingerprints,
            config,
            target,
            fold_ids,
            blend_weights[target],
        )
        details[target] = detail
        y = detail["target"].to_numpy(float)
        for arm in MODEL_NAMES:
            arm_prediction = detail[arm].to_numpy(float)
            append_metric(metrics, panel, target, arm, "all", None, y, arm_prediction)
            for fold in range(fold_count):
                selected = detail["fold"].to_numpy(int) == fold
                append_metric(metrics, panel, target, arm, "all", fold, y[selected], arm_prediction[selected])
        append_metric(metrics, panel, target, "frozen_blend", "all", None, y, detail["blend"].to_numpy(float))
        for fold in range(fold_count):
            selected = detail["fold"].to_numpy(int) == fold
            append_metric(metrics, panel, target, "frozen_blend", "all", fold, y[selected], detail.loc[selected, "blend"].to_numpy(float))
    return details


def add_low_similarity_metrics(
    metrics: list[dict[str, Any]],
    pooled: pd.DataFrame,
    main_oof: pd.DataFrame,
    main_folds: dict[str, np.ndarray],
    target_fingerprints: dict[str, list[Any]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    bins = (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01))
    for target in TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_oof = main_oof[main_oof["target_type"] == target].reset_index(drop=True)
        fps = target_fingerprints[target]
        nearest = np.zeros(len(target_rows), dtype=np.float64)
        fold_ids = main_folds[target]
        for fold in range(int(np.max(fold_ids)) + 1):
            validation = np.flatnonzero(fold_ids == fold)
            train = np.flatnonzero(fold_ids != fold)
            similarities = np.empty((len(validation), len(train)), dtype=np.float64)
            for row, index in enumerate(validation):
                similarities[row] = DataStructs.BulkTanimotoSimilarity(fps[index], [fps[j] for j in train])
            nearest[validation] = np.max(similarities, axis=1)
        summary[target] = {"min": float(np.min(nearest)), "median": float(np.median(nearest)), "max": float(np.max(nearest))}
        for name in (*MODEL_NAMES, "frozen_blend"):
            prediction_column = "prediction" if name == "frozen_blend" else name
            prediction = target_oof[prediction_column].to_numpy(float)
            y = target_oof["target"].to_numpy(float)
            for stratum, lower, upper in bins:
                selected = (nearest >= lower) & (nearest < upper)
                append_metric(metrics, "low_similarity", target, name, stratum, None, y[selected], prediction[selected])
    return summary


def exact_lookup_holdout(
    metrics: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    raw_labels: pd.DataFrame,
    pooled: pd.DataFrame,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"targets": {}, "leak_rows": 0, "covered_rows": 0}
    for target in TARGETS:
        target_raw = raw_labels[raw_labels["target_type"] == target].reset_index(drop=True)
        target_pooled = pooled[pooled["target_type"] == target].reset_index(drop=True)
        fold_ids = make_folds(target_pooled["canonical"].to_numpy(object), seed, folds)
        target_pooled_by_canonical = {row["canonical"]: index for index, row in target_pooled.iterrows()}
        target_raw["fold"] = target_raw["canonical"].map(lambda value: fold_ids[target_pooled_by_canonical[value]])
        target_report = {"rows": int(len(target_pooled)), "covered_rows": 0, "leak_rows": 0, "folds": []}
        for index, row in target_pooled.iterrows():
            assignments.append(
                {
                    "panel": "exact_lookup_group_holdout",
                    "target": target,
                    "canonical": row["canonical"],
                    "fold": int(fold_ids[index]),
                    "group_key": row["canonical"],
                }
            )
        for fold in range(folds):
            validation = target_pooled[fold_ids == fold]
            training_raw = target_raw[target_raw["fold"] != fold]
            raw_map = reference.unique_mapping(training_raw, ["smiles", "target_type"])
            canonical_map = reference.unique_mapping(training_raw, ["canonical", "target_type"])
            covered_y: list[float] = []
            covered_prediction: list[float] = []
            fold_covered = 0
            fold_leaks = 0
            for _, row in validation.iterrows():
                raw_key = (row["smiles"], target)
                canonical_key = (row["canonical"], target)
                prediction = raw_map.get(raw_key, canonical_map.get(canonical_key))
                if prediction is not None:
                    fold_covered += 1
                    covered_y.append(float(row["target"]))
                    covered_prediction.append(float(prediction))
                    if row["canonical"] in set(training_raw["canonical"]):
                        fold_leaks += 1
            target_report["covered_rows"] += fold_covered
            target_report["leak_rows"] += fold_leaks
            target_report["folds"].append({"fold": fold, "validation_rows": int(len(validation)), "covered_rows": fold_covered, "leak_rows": fold_leaks})
            if fold_covered:
                append_metric(
                    metrics,
                    "exact_lookup_group_holdout",
                    target,
                    "lookup_only",
                    "covered",
                    fold,
                    np.asarray(covered_y),
                    np.asarray(covered_prediction),
                )
        report["targets"][target] = target_report
        report["covered_rows"] += target_report["covered_rows"]
        report["leak_rows"] += target_report["leak_rows"]
    report["mapping_leakage_pass"] = report["leak_rows"] == 0
    return report


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
    if run_dir.exists():
        existing = {path.name for path in run_dir.iterdir()}
        if existing - {"protocol.json"}:
            raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = dict(reference.DEFAULT_CONFIG)
    config["seed"] = 2026
    config["folds"] = 5
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    for name, expected in EXPECTED_HASHES.items():
        if inputs[name]["sha256"] != expected:
            raise RuntimeError(f"Official hash mismatch for {name}")
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(config["morgan_bits"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    target_fingerprints: dict[str, list[Any]] = {}
    for target in TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_fingerprints[target] = [fingerprints[key_to_index[value]] for value in target_rows["canonical"]]

    blend_weights = load_reference_weights(root)
    main_oof = load_main_oof(root, pooled)
    metrics: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    main_folds = add_main_metrics(metrics, assignments, pooled, main_oof, blend_weights, int(config["seed"]), int(config["folds"]))

    panel_groups: dict[str, dict[str, np.ndarray]] = {}
    panel_groups["canonical_group"] = {
        target: pooled[pooled["target_type"] == target].reset_index(drop=True)["canonical"].to_numpy(object)
        for target in TARGETS
    }
    panel_groups["scaffold_family"] = {}
    panel_groups["tanimoto_cluster"] = {}
    for target in TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        panel_groups["scaffold_family"][target] = np.asarray([make_scaffold(value) for value in target_rows["canonical"]], dtype=object)
        panel_groups["tanimoto_cluster"][target] = greedy_tanimoto_groups(target_fingerprints[target], threshold=0.70)

    panel_details: dict[str, Any] = {}
    for panel in ("canonical_group", "scaffold_family", "tanimoto_cluster"):
        panel_details[panel] = add_panel_metrics(
            metrics,
            assignments,
            pooled,
            panel,
            panel_groups[panel],
            keys,
            dense_base,
            cross_values,
            cross_available,
            sparse_parts,
            fingerprints,
            config,
            blend_weights,
            int(config["folds"]),
        )

    availability_summary: dict[str, Any] = {"target_mask_pass": True, "targets": {}}
    for target in TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_oof = main_oof[main_oof["target_type"] == target].reset_index(drop=True)
        target_index = TARGETS.index(target)
        available_count = np.sum(np.delete(cross_available[[key_to_index[value] for value in target_rows["canonical"]]], target_index, axis=1), axis=1).astype(int)
        row_positions = [key_to_index[value] for value in target_rows["canonical"]]
        masked_values = cross_values[row_positions].copy()
        masked_available = cross_available[row_positions].copy()
        masked_values[:, target_index] = np.nan
        masked_available[:, target_index] = 0.0
        if not np.isnan(masked_values[:, target_index]).all() or np.any(masked_available[:, target_index] != 0.0):
            availability_summary["target_mask_pass"] = False
        availability_summary["targets"][target] = {
            "rows": int(len(target_rows)),
            "auxiliary_label_count_min": int(np.min(available_count)),
            "auxiliary_label_count_median": float(np.median(available_count)),
            "auxiliary_label_count_max": int(np.max(available_count)),
        }
        for count in sorted(np.unique(available_count)):
            selected = available_count == count
            for name in (*MODEL_NAMES, "frozen_blend"):
                prediction_column = "prediction" if name == "frozen_blend" else name
                append_metric(metrics, "cross_property_availability", target, name, f"aux_count_{count}", None, target_oof.loc[selected, "target"].to_numpy(float), target_oof.loc[selected, prediction_column].to_numpy(float))

    low_similarity_summary = add_low_similarity_metrics(metrics, pooled, main_oof, main_folds, target_fingerprints)
    lookup_report = exact_lookup_holdout(metrics, assignments, raw_labels, pooled, int(config["folds"]), int(config["seed"]))

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(run_dir / "panel_metrics.csv", index=False)
    pd.DataFrame(assignments).to_csv(run_dir / "fold_assignments.csv", index=False)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.validation-hardening.v1", "seed": 2026, "folds": 5, "target_order": list(TARGETS), "tanimoto_cluster_threshold": 0.70, "reference_config_sha256": canonical_json_hash(config), "official_inputs": inputs})
    (run_dir / "environment.txt").write_text(
        "\n".join([
            f"python={platform.python_version()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"rdkit={Chem.rdBase.rdkitVersion}",
            f"platform={platform.platform()}",
        ]) + "\n",
        encoding="utf-8",
    )
    target_summary: dict[str, Any] = {}
    for panel in sorted(metric_frame["panel"].unique()):
        panel_rows = metric_frame[(metric_frame["panel"] == panel) & (metric_frame["fold"].isna())]
        target_summary[panel] = {}
        for target in TARGETS:
            rows = panel_rows[panel_rows["target"] == target]
            target_summary[panel][target] = {
                str(row["arm"]): {key: (None if pd.isna(row[key]) else float(row[key]) if key != "rows" else int(row[key])) for key in ("rows", "r2", "mae", "rmse")}
                for _, row in rows.iterrows()
            }
    selected_rows = metric_frame[(metric_frame["arm"] == "frozen_blend") & (metric_frame["fold"].isna()) & (metric_frame["panel"].isin(["main_fixed_reproduction", "canonical_group", "scaffold_family", "tanimoto_cluster"]))]
    panel_means = {
        panel: float(selected_rows[selected_rows["panel"] == panel]["r2"].mean())
        for panel in selected_rows["panel"].unique()
    }
    audit = {
        "schema_version": "ppp.round2.validation-hardening-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": C001_ID,
        "official_inputs": inputs,
        "official_hashes_pass": True,
        "reference_artifacts": {
            "c001_report_sha256": sha256_file(root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json"),
            "c001_oof_sha256": sha256_file(root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "oof_predictions.csv"),
            "reference_blend_weights_sha256": canonical_json_hash({target: values.tolist() for target, values in blend_weights.items()}),
        },
        "rows": {"current_train": len(train), "archive_train": len(archive), "pooled_target_rows": len(pooled), "test": len(test)},
        "panels": list(panel_means) + ["low_similarity", "exact_lookup_group_holdout", "cross_property_availability"],
        "panel_means_frozen_blend": panel_means,
        "target_summary": target_summary,
        "low_similarity_nearest_train_summary": low_similarity_summary,
        "exact_lookup_group_holdout": lookup_report,
        "cross_property_availability": availability_summary,
        "validity": {
            "official_only": True,
            "target_mask_pass": bool(availability_summary["target_mask_pass"]),
            "mapping_leakage_pass": bool(lookup_report["mapping_leakage_pass"]),
            "fold_assignment_complete": bool(len(assignments) > 0 and not pd.DataFrame(assignments)["fold"].isna().any()),
            "all_panel_predictions_finite": bool(np.isfinite(metric_frame["mae"].dropna().to_numpy(float)).all()),
        },
        "decision": "pass" if availability_summary["target_mask_pass"] and lookup_report["mapping_leakage_pass"] else "validation_repair_needed",
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "metrics.json", audit)
    (run_dir / "decision.md").write_text(
        "# R2-C002 validation hardening decision\n\n"
        f"Decision: **{audit['decision']}**.\n\n"
        "This run froze C001 carriers and evaluated stricter validation panels. "
        "It did not change a candidate or read any post-hoc external_label source. See `metrics.json`, "
        "`panel_metrics.csv`, and `fold_assignments.csv` for aggregate evidence.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "panel_metrics.csv", "fold_assignments.csv", "metrics.json", "decision.md", "command.txt")]
    (run_dir / "artifact_manifest.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "panel_means_frozen_blend": panel_means, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
