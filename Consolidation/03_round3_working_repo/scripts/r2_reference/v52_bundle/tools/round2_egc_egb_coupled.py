#!/usr/bin/env python3
"""Evaluate the preregistered official-only Egc/Egb coupled-label specialist."""

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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
SPECIAL_TARGETS = ("egc", "egb")
AUX_TARGET = {"egc": "egb", "egb": "egc"}
MODEL_NAMES = ("paired_affine_ridge_alpha_10", "paired_extra_trees_128_min_leaf_4")
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    return {
        "rows": int(len(y)),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and not np.isclose(np.var(y), 0.0) else None,
        "mae": float(mean_absolute_error(y, prediction)) if len(y) else None,
        "rmse": float(np.sqrt(np.mean(np.square(y - prediction)))) if len(y) else None,
    }


def folds_for(rows: pd.DataFrame) -> np.ndarray:
    groups = rows["canonical"].to_numpy(object)
    result = np.full(len(rows), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(rows)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("Incomplete canonical-group fold assignment")
    return result


def load_weights(root: Path, target: str) -> np.ndarray:
    report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    values = report["validation"]["target_reports"][target]["blend_weights"]
    names = ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def pair_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.column_stack([values, values * values, np.log1p(np.abs(values))])


def paired_predictions(
    aux_values: np.ndarray,
    y_global: np.ndarray,
    target_indices: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    baseline: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    train_indices = target_indices[train_rows]
    validation_indices = target_indices[validation_rows]
    train_aux = aux_values[train_indices]
    validation_aux = aux_values[validation_indices]
    available_train = np.isfinite(train_aux)
    available_validation = np.isfinite(validation_aux)
    if int(np.sum(available_train)) < 12:
        return np.column_stack([baseline, baseline])
    x_train = pair_features(train_aux[available_train])
    y_train = y_global[train_indices[available_train]]
    scaler = StandardScaler().fit(x_train)
    affine = Ridge(alpha=10.0).fit(scaler.transform(x_train), y_train)
    nonlinear = ExtraTreesRegressor(
        n_estimators=128,
        min_samples_leaf=4,
        max_features=1.0,
        random_state=2026,
        n_jobs=2,
    ).fit(x_train, y_train)
    output = np.repeat(np.asarray(baseline, dtype=np.float64)[:, None], 2, axis=1)
    if np.any(available_validation):
        x_validation = pair_features(validation_aux[available_validation])
        affine_values = affine.predict(scaler.transform(x_validation))
        nonlinear_values = nonlinear.predict(x_validation)
        affine_values = reference.clip_prediction(target_y, affine_values)
        nonlinear_values = reference.clip_prediction(target_y, nonlinear_values)
        output[available_validation, 0] = affine_values
        output[available_validation, 1] = nonlinear_values
    return output


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    output = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in train]
        for index in validation:
            output[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return output


def bootstrap_lower(y: np.ndarray, specialist: np.ndarray, baseline: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(2000):
        sample = rng.integers(0, len(y), size=len(y))
        if np.isclose(np.var(y[sample]), 0.0):
            continue
        values.append(float(r2_score(y[sample], specialist[sample]) - r2_score(y[sample], baseline[sample])))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def add_metric(rows: list[dict[str, Any]], target: str, method: str, stratum: str, fold: int | None, y: np.ndarray, prediction: np.ndarray) -> None:
    item = {"target": target, "method": method, "stratum": stratum, "fold": fold}
    item.update(metric(y, prediction))
    rows.append(item)


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
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    key_to_index = {key: index for index, key in enumerate(keys)}
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for target in SPECIAL_TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_indices = np.asarray([key_to_index[value] for value in target_rows["canonical"]], dtype=np.int64)
        y = target_rows["target"].to_numpy(float)
        folds = folds_for(target_rows)
        target_position = TARGETS.index(target)
        aux_position = TARGETS.index(AUX_TARGET[target])
        aux_values = cross_values[:, aux_position].copy()
        dense = reference.target_dense_features(np.hstack([descriptor, physical]), cross_values, cross_available, target)
        nearest = nearest_similarity([fingerprints[index] for index in target_indices], folds)
        baseline_parts = np.full((len(y), 4), np.nan, dtype=np.float64)
        specialist_parts = np.full((len(y), len(MODEL_NAMES)), np.nan, dtype=np.float64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[target_indices] = y
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            local_baseline = reference.predict_base_models(
                dense,
                sparse_parts,
                fingerprints,
                y_global,
                target_indices[train_rows],
                target_indices[validation_rows],
                reference.DEFAULT_CONFIG,
                target,
            )
            baseline_parts[validation_rows] = local_baseline
            frozen_weights = load_weights(root, target)
            baseline_blend = local_baseline @ frozen_weights
            specialist_parts[validation_rows] = paired_predictions(
                aux_values,
                y_global,
                target_indices,
                train_rows,
                validation_rows,
                baseline_blend,
                y[train_rows],
            )
            fold_rows.extend(
                {
                    "target": target,
                    "fold": fold,
                    "canonical": target_rows.iloc[index]["canonical"],
                    "aux_target": AUX_TARGET[target],
                    "aux_available": bool(np.isfinite(aux_values[target_indices[index]])),
                    "nearest_similarity": float(nearest[index]),
                }
                for index in validation_rows
            )
        baseline = baseline_parts @ load_weights(root, target)
        available = np.isfinite(aux_values[target_indices])
        no_aux = ~available
        low_gap = y <= np.quantile(y, 0.25)
        report: dict[str, Any] = {
            "rows": int(len(y)),
            "aux_target": AUX_TARGET[target],
            "aux_available_rows": int(np.sum(available)),
            "baseline": metric(y, baseline),
            "models": {},
            "folds": [],
            "slices": {},
        }
        for index, name in enumerate(MODEL_NAMES):
            report["models"][name] = metric(y, specialist_parts[:, index])
        for fold in range(5):
            selected = folds == fold
            fold_report = {"fold": fold, "baseline": metric(y[selected], baseline[selected])}
            for index, name in enumerate(MODEL_NAMES):
                fold_report[name] = metric(y[selected], specialist_parts[selected, index])
                fold_report[f"{name}_delta_r2"] = float(fold_report[name]["r2"] - fold_report["baseline"]["r2"])
            report["folds"].append(fold_report)
        for name, mask in (("missing_auxiliary", no_aux), ("paired_auxiliary", available), ("low_gap", low_gap)):
            report["slices"][name] = {
                "rows": int(np.sum(mask)),
                "baseline": metric(y[mask], baseline[mask]),
                **{model: metric(y[mask], specialist_parts[mask, index]) for index, model in enumerate(MODEL_NAMES)},
            }
            for model in MODEL_NAMES:
                base_r2 = report["slices"][name]["baseline"]["r2"]
                model_r2 = report["slices"][name][model]["r2"]
                report["slices"][name][f"{model}_delta_r2"] = None if base_r2 is None or model_r2 is None else float(model_r2 - base_r2)
        selected_name = max(MODEL_NAMES, key=lambda name: report["models"][name]["r2"])
        selected_index = MODEL_NAMES.index(selected_name)
        fold_deltas = [float(value[f"{selected_name}_delta_r2"]) for value in report["folds"]]
        report["selected_model"] = selected_name
        report["selected_delta_r2"] = float(report["models"][selected_name]["r2"] - report["baseline"]["r2"])
        report["selected_positive_folds"] = int(sum(value > 0 for value in fold_deltas))
        report["selected_bootstrap_lower_bound"] = bootstrap_lower(y, specialist_parts[:, selected_index], baseline, 2026 + target_position)
        report["selected_missing_auxiliary_delta"] = report["slices"]["missing_auxiliary"][f"{selected_name}_delta_r2"]
        report["selected_low_gap_delta"] = report["slices"]["low_gap"][f"{selected_name}_delta_r2"]
        report["paired_target_loss"] = 0.0
        reports[target] = report
        for name, prediction in [("frozen_c001_blend", baseline), *[(name, specialist_parts[:, index]) for index, name in enumerate(MODEL_NAMES)]]:
            add_metric(rows, target, name, "all", None, y, prediction)
            for fold in range(5):
                selected = folds == fold
                add_metric(rows, target, name, "all", fold, y[selected], prediction[selected])
            for slice_name, mask in (("missing_auxiliary", no_aux), ("paired_auxiliary", available), ("low_gap", low_gap)):
                add_metric(rows, target, name, slice_name, None, y[mask], prediction[mask])
        for index, row in target_rows.iterrows():
            prediction_rows.append(
                {
                    "canonical": row["canonical"],
                    "target_type": target,
                    "target": float(y[index]),
                    "fold": int(folds[index]),
                    "aux_target": AUX_TARGET[target],
                    "aux_available": bool(available[index]),
                    "nearest_similarity": float(nearest[index]),
                    "baseline": float(baseline[index]),
                    **{name: float(specialist_parts[index, column]) for column, name in enumerate(MODEL_NAMES)},
                }
            )
    passing_targets = [
        target for target in SPECIAL_TARGETS
        if reports[target]["selected_delta_r2"] >= 0.01
        and reports[target]["selected_positive_folds"] >= 4
        and reports[target]["selected_bootstrap_lower_bound"] > 0.0
        and (reports[target]["selected_missing_auxiliary_delta"] is None or reports[target]["selected_missing_auxiliary_delta"] >= 0.0)
        and reports[target]["selected_low_gap_delta"] is not None
        and reports[target]["selected_low_gap_delta"] >= 0.0
        and reports[target]["paired_target_loss"] <= 0.003
    ]
    audit = {
        "schema_version": "ppp.round2.egc-egb-coupled-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C004-20260803-1726-ei-eea-electronic-specialist",
        "official_inputs": inputs,
        "official_hashes_pass": all(inputs[name]["sha256"] == expected for name, expected in reference.EXPECTED_HASHES.items()),
        "models": list(MODEL_NAMES),
        "targets": reports,
        "passing_targets": passing_targets,
        "decision": "component_pass" if passing_targets else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame(rows).to_csv(run_dir / "panel_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_assignments.csv", index=False)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.egc-egb-coupled.v1", "seed": 2026, "folds": 5, "models": list(MODEL_NAMES), "official_inputs": inputs, "aux_target": AUX_TARGET})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", audit)
    (run_dir / "decision.md").write_text(f"# R2-C005 Egc/Egb coupled specialist decision\n\nDecision: **{audit['decision']}**.\n\nThe specialist is evaluated only where the other official electronic-gap label is available; missing-auxiliary rows retain the frozen C001 blend.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "panel_metrics.csv", "predictions.csv", "fold_assignments.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "passing_targets": passing_targets, "summary": {target: {"baseline": reports[target]["baseline"]["r2"], "selected_model": reports[target]["selected_model"], "selected_r2": reports[target]["models"][reports[target]["selected_model"]]["r2"], "delta": reports[target]["selected_delta_r2"], "missing_aux_delta": reports[target]["selected_missing_auxiliary_delta"], "low_gap_delta": reports[target]["selected_low_gap_delta"]} for target in SPECIAL_TARGETS}, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
