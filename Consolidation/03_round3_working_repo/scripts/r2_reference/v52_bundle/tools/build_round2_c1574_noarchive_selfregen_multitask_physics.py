#!/usr/bin/env python3
"""C1574 noarchive self-regenerating multitask physics residual.

This builder reads only official current Round 2 train/test files.  It
regenerates a C282-style per-target base from scratch, then trains a fold-safe
shared residual layer using structure features, target identity, base component
predictions, co-property base predictions, and simple physics residual features.

No archive labels, prior candidate CSVs, local_eval/external_label/nonofficial files, external
data, pretrained assets, Kaggle state, or cached features are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_round2_c385_archive_weak_target_model_zoo as c385
import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
MODEL_COLUMNS = ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local", "prediction")
SHRINK_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50)
SCHEMA = "ppp.round2.c1574.noarchive-selfregen-multitask-physics.v1"
SEED = 20260808


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def guard_path(path: Path, *, role: str, output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial", "/archive/", "with_archive"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if output and "polymer prediction challenge round 2" not in low:
        raise RuntimeError(f"{role} outside Round 2 boundary: {path}")


def sanitize(values: np.ndarray, limit: float = 1.0e6) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[(~np.isfinite(out)) | (np.abs(out) > limit)] = np.nan
    return out


def finite_r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(float(np.var(y)), 0.0):
        return float("nan")
    return float(r2_score(y, pred))


def pivot_prediction(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return frame.pivot_table(index="canonical", columns="target_type", values=value_col, aggfunc="mean")


def lookup_pivot(pivot: pd.DataFrame, canon: str, target: str) -> float:
    try:
        value = pivot.at[canon, target]
    except Exception:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_row_features(
    *,
    frame: pd.DataFrame,
    key_to_index: dict[str, int],
    structure_features: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    component_values: np.ndarray,
    co_pivot: pd.DataFrame,
) -> np.ndarray:
    rows = []
    target_index_map = {target: index for index, target in enumerate(TARGETS)}
    for row_pos, row in enumerate(frame.itertuples(index=False)):
        canon = str(row.canonical)
        target = str(row.target_type)
        key_index = key_to_index[canon]
        target_index = target_index_map[target]
        target_onehot = np.zeros(len(TARGETS), dtype=np.float64)
        target_onehot[target_index] = 1.0

        cross_v = cross_values[key_index].astype(np.float64, copy=True)
        cross_a = cross_available[key_index].astype(np.float64, copy=True)
        cross_v[target_index] = np.nan
        cross_a[target_index] = 0.0

        co_values = np.asarray([lookup_pivot(co_pivot, canon, other) for other in TARGETS], dtype=np.float64)
        co_available = np.isfinite(co_values).astype(np.float64)
        co_values[target_index] = np.nan
        co_available[target_index] = 0.0

        current_base = float(component_values[row_pos, MODEL_COLUMNS.index("prediction")])
        physics_values = np.asarray(
            [
                (co_values[TARGETS.index("egb")] - current_base) if target == "egc" else np.nan,
                (current_base - co_values[TARGETS.index("egc")]) if target == "egb" else np.nan,
                (co_values[TARGETS.index("ei")] - co_values[TARGETS.index("eea")] - current_base)
                if target == "egc"
                else np.nan,
                (current_base - co_values[TARGETS.index("eea")] - co_values[TARGETS.index("egc")])
                if target == "ei"
                else np.nan,
                (co_values[TARGETS.index("ei")] - current_base - co_values[TARGETS.index("egc")])
                if target == "eea"
                else np.nan,
                (current_base - np.square(co_values[TARGETS.index("nc")])) if target == "eps" else np.nan,
                (co_values[TARGETS.index("eps")] - np.square(current_base)) if target == "nc" else np.nan,
            ],
            dtype=np.float64,
        )
        physics_available = np.isfinite(physics_values).astype(np.float64)
        rows.append(
            np.concatenate(
                [
                    structure_features[key_index],
                    cross_v,
                    cross_a,
                    component_values[row_pos],
                    co_values,
                    co_available,
                    physics_values,
                    physics_available,
                    target_onehot,
                ]
            )
        )
    return sanitize(np.vstack(rows))


def fit_predict_residual(kind: str, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, seed: int) -> np.ndarray:
    if kind == "ridge":
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=45.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        )
    elif kind == "hgb":
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=12,
                l2_regularization=0.20,
                random_state=seed,
            ),
        )
    else:
        raise RuntimeError(f"Unknown residual kind: {kind}")
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_pred), dtype=np.float64)


def parse_residual_kinds(value: str) -> tuple[str, ...]:
    kinds = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not kinds:
        raise RuntimeError("No residual kinds supplied")
    invalid = [kind for kind in kinds if kind not in {"ridge", "hgb"}]
    if invalid:
        raise RuntimeError(f"Invalid residual kinds: {invalid}")
    return kinds


def grouped_residual_oof(
    kind: str,
    features: np.ndarray,
    residual_z: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    oof = np.full(len(residual_z), np.nan, dtype=np.float64)
    fold_reports = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (tr, va) in enumerate(splitter.split(features, residual_z, groups=groups)):
        pred = fit_predict_residual(kind, features[tr], residual_z[tr], features[va], seed + fold)
        oof[va] = pred
        fold_reports.append({"fold": int(fold), "train_rows": int(len(tr)), "validation_rows": int(len(va))})
    if not np.isfinite(oof).all():
        raise RuntimeError(f"Non-finite residual OOF for {kind}")
    return oof, fold_reports


def target_stds(pooled: pd.DataFrame) -> dict[str, float]:
    result = {}
    for target in TARGETS:
        values = pooled.loc[pooled["target_type"].eq(target), "target"].to_numpy(float)
        std = float(np.std(values))
        result[target] = std if std > 1.0e-8 else 1.0
    return result


def choose_target_residuals(
    *,
    pooled: pd.DataFrame,
    base_oof: np.ndarray,
    residual_oof_by_kind: dict[str, np.ndarray],
    std_map: dict[str, float],
    groups: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    selected: dict[str, Any] = {}
    candidate_oof_by_target: dict[str, np.ndarray] = {}
    y_all = pooled["target"].to_numpy(float)
    target_types = pooled["target_type"].astype(str).to_numpy(object)
    for target in TARGETS:
        mask = target_types == target
        y = y_all[mask]
        base = base_oof[mask]
        base_r2 = finite_r2(y, base)
        local_groups = groups[mask]
        best = {
            "target": target,
            "kind": "none",
            "shrink": 0.0,
            "oof_r2": base_r2,
            "delta_r2": 0.0,
            "positive_folds": 0,
            "minimum_fold_delta": 0.0,
            "accepted": False,
        }
        best_values = base.copy()
        for kind, residual_z in residual_oof_by_kind.items():
            local_residual = residual_z[mask] * std_map[target]
            for shrink in SHRINK_GRID:
                candidate = base + float(shrink) * local_residual
                score = finite_r2(y, candidate)
                fold_deltas = []
                for group_fold, (_, va) in enumerate(GroupKFold(n_splits=min(5, len(np.unique(local_groups)))).split(np.arange(len(y)), y, groups=local_groups)):
                    fold_deltas.append(finite_r2(y[va], candidate[va]) - finite_r2(y[va], base[va]))
                positive_folds = int(sum(delta > 0.0 for delta in fold_deltas if np.isfinite(delta)))
                min_fold_delta = float(np.nanmin(fold_deltas)) if fold_deltas else float("nan")
                accepted = bool(
                    shrink > 0.0
                    and score > best["oof_r2"]
                    and score > base_r2
                    and positive_folds >= 3
                    and min_fold_delta >= -0.03
                )
                if accepted:
                    best = {
                        "target": target,
                        "kind": kind,
                        "shrink": float(shrink),
                        "oof_r2": score,
                        "delta_r2": float(score - base_r2),
                        "positive_folds": positive_folds,
                        "minimum_fold_delta": min_fold_delta,
                        "accepted": True,
                    }
                    best_values = candidate.copy()
        selected[target] = {
            "base_oof_r2": base_r2,
            **best,
            "rows": int(np.sum(mask)),
        }
        candidate_oof_by_target[target] = best_values
    return selected, candidate_oof_by_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--residual-kinds", default="ridge")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (root / output).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    for path, role, is_output in ((data_dir, "data dir", False), (output, "output", True), (run_dir, "run dir", True)):
        guard_path(path, role=role, output=is_output)
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    configuration = dict(reference.DEFAULT_CONFIG)
    configuration["seed"] = int(args.seed)
    train, test, inputs = c282.load_current_only_inputs(data_dir)
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    structure_features = sanitize(dense_base)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(configuration["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(configuration["morgan_bits"])),
        reference.text_matrix(keys, int(configuration["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(configuration["morgan_bits"]))

    detail, oof, model_report = reference.fit_targets(
        pooled,
        test,
        keys,
        dense_base,
        cross_values,
        cross_available,
        sparse_parts,
        fingerprints,
        configuration,
    )
    final_detail, override_report = reference.apply_official_overrides(detail, test, raw_labels)

    oof = oof.copy()
    oof["target_type"] = oof["target_type"].astype(str).str.lower()
    test_features_frame = test[["id", "canonical", "target_type"]].merge(detail, on=["id", "target_type"], how="left", validate="one_to_one")
    test_features_frame["prediction"] = test_features_frame["model_prediction"].astype(float)
    if test_features_frame[list(MODEL_COLUMNS)].isna().any().any():
        raise RuntimeError("Missing base test component predictions")

    oof_components = oof[list(MODEL_COLUMNS)].to_numpy(float)
    test_components = test_features_frame[list(MODEL_COLUMNS)].to_numpy(float)
    oof_feature_frame = oof[["canonical", "target_type"]].copy()
    test_feature_frame = test_features_frame[["canonical", "target_type"]].copy()
    oof_co_pivot = pivot_prediction(oof, "prediction")
    test_co_pivot = pivot_prediction(test_features_frame, "prediction")

    x_train = build_row_features(
        frame=oof_feature_frame,
        key_to_index=key_to_index,
        structure_features=structure_features,
        cross_values=cross_values,
        cross_available=cross_available,
        component_values=oof_components,
        co_pivot=oof_co_pivot,
    )
    x_test = build_row_features(
        frame=test_feature_frame,
        key_to_index=key_to_index,
        structure_features=structure_features,
        cross_values=cross_values,
        cross_available=cross_available,
        component_values=test_components,
        co_pivot=test_co_pivot,
    )

    base_oof = oof["prediction"].to_numpy(float)
    y_all = oof["target"].to_numpy(float)
    std_map = target_stds(oof)
    target_types = oof["target_type"].to_numpy(object)
    residual_z = np.empty(len(oof), dtype=np.float64)
    for target in TARGETS:
        mask = target_types == target
        residual_z[mask] = (y_all[mask] - base_oof[mask]) / std_map[target]
    groups = np.asarray([c385.no_stereo(value) for value in oof["canonical"].astype(str)], dtype=object)

    residual_oof_by_kind: dict[str, np.ndarray] = {}
    residual_test_by_kind: dict[str, np.ndarray] = {}
    residual_reports: dict[str, Any] = {}
    residual_kinds = parse_residual_kinds(args.residual_kinds)
    for kind in residual_kinds:
        residual_oof, folds = grouped_residual_oof(kind, x_train, residual_z, groups, int(args.seed))
        residual_test = fit_predict_residual(kind, x_train, residual_z, x_test, int(args.seed) + 1009)
        residual_oof_by_kind[kind] = residual_oof
        residual_test_by_kind[kind] = residual_test
        residual_reports[kind] = {
            "folds": folds,
            "residual_z_r2": finite_r2(residual_z, residual_oof),
        }

    selected, candidate_oof_by_target = choose_target_residuals(
        pooled=oof,
        base_oof=base_oof,
        residual_oof_by_kind=residual_oof_by_kind,
        std_map=std_map,
        groups=groups,
    )
    accepted_targets = [target for target, row in selected.items() if row["accepted"]]

    test_values = final_detail["target"].to_numpy(float).copy()
    test_target_types = test["target_type"].astype(str).to_numpy(object)
    base_model_prediction = test_features_frame["prediction"].to_numpy(float)
    override_mask = final_detail["override"].astype(str).ne("model").to_numpy()
    for target, row in selected.items():
        if not row["accepted"]:
            continue
        mask = test_target_types == target
        kind = str(row["kind"])
        shrink = float(row["shrink"])
        update = base_model_prediction[mask] + shrink * residual_test_by_kind[kind][mask] * std_map[target]
        test_values[mask] = reference.clip_prediction(
            oof.loc[oof["target_type"].eq(target), "target"].to_numpy(float),
            update,
        )
    test_values[override_mask] = final_detail.loc[override_mask, "target"].to_numpy(float)
    if len(test_values) != 4940 or not np.isfinite(test_values).all():
        raise RuntimeError("Invalid final output")
    pd.DataFrame({"id": test["id"].astype(int), "target": test_values}).to_csv(output, index=False)

    # OOF report for the selected composite.
    selected_oof = base_oof.copy()
    for target, values in candidate_oof_by_target.items():
        if selected[target]["accepted"]:
            mask = target_types == target
            selected_oof[mask] = values
    target_reports = {}
    for target in TARGETS:
        mask = target_types == target
        target_reports[target] = {
            "rows": int(np.sum(mask)),
            "base_oof_r2": finite_r2(y_all[mask], base_oof[mask]),
            "selected_oof_r2": finite_r2(y_all[mask], selected_oof[mask]),
            "delta_r2": finite_r2(y_all[mask], selected_oof[mask]) - finite_r2(y_all[mask], base_oof[mask]),
            "selection": selected[target],
        }
    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": run_dir.name,
        "method": "current-train/test-only regenerated per-target ensemble with fold-safe shared residual layer",
        "official_current_train_used": True,
        "official_current_test_used": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read_by_builder": False,
        "external_label_file_read_by_builder": False,
        "nonofficial_file_read_by_builder": False,
        "prior_candidate_csv_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "inputs": inputs,
        "config": {"seed": int(args.seed), "base_config": configuration, "shrink_grid": SHRINK_GRID, "residual_kinds": residual_kinds},
        "features": {
            "unique_structures": int(len(keys)),
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "row_feature_count": int(x_train.shape[1]),
        },
        "base_validation": model_report,
        "official_overrides": override_report,
        "residual_models": residual_reports,
        "accepted_targets": accepted_targets,
        "targets": target_reports,
        "base_mean_oof_r2": float(np.mean([target_reports[target]["base_oof_r2"] for target in TARGETS])),
        "selected_mean_oof_r2": float(np.mean([target_reports[target]["selected_oof_r2"] for target in TARGETS])),
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(test_values)), "bytes": output.stat().st_size},
        "elapsed_seconds": float(time.time() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", report["config"])
    final_detail.drop(columns=["smiles", "canonical"]).to_csv(run_dir / "base_test_predictions_detail.csv", index=False)
    oof.assign(selected_prediction=selected_oof).to_csv(run_dir / "oof_predictions.csv", index=False)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text("\n".join(f"{k}={v}" for k, v in report["environment"].items()) + "\n", encoding="utf-8")
    manifest_paths = [
        output,
        run_dir / "metrics.json",
        run_dir / "config.json",
        run_dir / "base_test_predictions_detail.csv",
        run_dir / "oof_predictions.csv",
        run_dir / "command.txt",
        run_dir / "environment.txt",
    ]
    source_paths = [
        Path(__file__).resolve(),
        root / "tools" / "initial_reference_pipeline.py",
        root / "tools" / "round2_c282_current_only_reference.py",
    ]
    lines = [f"{sha256_file(path)}  {path}" for path in manifest_paths]
    lines.extend(f"{sha256_file(path)}  SOURCE {path}" for path in source_paths)
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": report["output"],
                "accepted_targets": accepted_targets,
                "base_mean_oof_r2": report["base_mean_oof_r2"],
                "selected_mean_oof_r2": report["selected_mean_oof_r2"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
