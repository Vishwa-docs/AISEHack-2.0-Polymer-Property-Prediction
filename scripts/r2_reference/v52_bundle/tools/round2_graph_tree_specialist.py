#!/usr/bin/env python3
"""Bounded official-only Morgan-count tree test for Round 2."""

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

import catboost as cb
import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = ("eps", "ei", "nc", "eea")
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
MODEL_NAMES = ("graph_extra_trees", "graph_catboost")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    if len(y) < 2 or float(np.var(y)) <= 1.0e-15:
        return {"r2": None, "mae": None}
    return {"r2": float(r2_score(y, prediction)), "mae": float(np.mean(np.abs(y - prediction)))}


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


def graph_counts(molecules: list[Any], radius: int, bits: int) -> np.ndarray:
    return reference.morgan_count_matrix(molecules, radius, bits).toarray().astype(np.float64, copy=False)


def fold_matrix(matrix: np.ndarray, train_rows: np.ndarray, validation_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.asarray(matrix[train_rows], dtype=np.float64).copy()
    validation_x = np.asarray(matrix[validation_rows], dtype=np.float64).copy()
    train_x[~np.isfinite(train_x)] = np.nan
    validation_x[~np.isfinite(validation_x)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(train_x)
    validation_x = imputer.transform(validation_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    if not np.any(keep):
        raise RuntimeError("graph matrix lost every nonconstant feature")
    return train_x[:, keep], validation_x[:, keep]


def model_factory(name: str, target_index: int) -> Any:
    seed = 2026 + target_index * 10
    if name == "graph_extra_trees":
        return ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=2,
            max_features=0.35,
            random_state=seed,
            n_jobs=2,
        )
    if name == "graph_catboost":
        return cb.CatBoostRegressor(
            loss_function="RMSE",
            iterations=350,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=6.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    raise ValueError(name)


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
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    graph = np.hstack([graph_counts(molecules, radius, 2048) for radius in (1, 2, 3)])
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(TARGETS):
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        groups = frame["canonical"].to_numpy(object)
        indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[indices] = y
        folds = folds_for(groups)
        target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        candidate_matrix = np.hstack([graph[indices], target_dense[indices]])
        baseline = np.full(len(y), np.nan, dtype=np.float64)
        predictions = {name: np.full(len(y), np.nan, dtype=np.float64) for name in MODEL_NAMES}
        fold_reports: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_NAMES}
        retained_features: list[int] = []
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            base = reference.predict_base_models(
                target_dense,
                [
                    reference.morgan_count_matrix(molecules, 2, 4096),
                    reference.morgan_count_matrix(molecules, 3, 4096),
                    reference.text_matrix(keys, 65536),
                ],
                fingerprints,
                y_global,
                indices[train_rows],
                indices[validation_rows],
                reference.DEFAULT_CONFIG,
                target,
            )
            weights = c001_report["validation"]["target_reports"][target]["blend_weights"]
            blend_weights = np.asarray([weights[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)
            intercept = float(c001_report["validation"]["target_reports"][target]["blend_intercept"])
            baseline[validation_rows] = base @ blend_weights + intercept
            train_x, validation_x = fold_matrix(candidate_matrix, train_rows, validation_rows)
            retained_features.append(int(train_x.shape[1]))
            for name in MODEL_NAMES:
                model = model_factory(name, target_index)
                model.fit(train_x, y[train_rows])
                prediction = reference.clip_prediction(y[train_rows], np.asarray(model.predict(validation_x), dtype=np.float64))
                predictions[name][validation_rows] = prediction
                base_fold = metric(y[validation_rows], baseline[validation_rows])["r2"]
                model_fold = metric(y[validation_rows], prediction)["r2"]
                fold_reports[name].append({
                    "fold": fold,
                    "rows": int(len(validation_rows)),
                    "baseline_r2": base_fold,
                    "model_r2": model_fold,
                    "delta_r2": float(model_fold - base_fold),
                })
            fold_rows.extend({"target": target, "fold": fold, "row": int(row)} for row in validation_rows)

        nearest = nearest_similarity([fingerprints[index] for index in indices], folds)
        base_metric = metric(y, baseline)
        target_report: dict[str, Any] = {
            "rows": int(len(y)),
            "baseline": base_metric,
            "baseline_r2": float(base_metric["r2"]),
            "retained_features_per_fold": retained_features,
            "models": {},
        }
        for name in MODEL_NAMES:
            prediction = predictions[name]
            model_metric = metric(y, prediction)
            delta = prediction - baseline
            low_similarity: dict[str, Any] = {}
            for bin_name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
                selected = (nearest >= lower) & (nearest < upper)
                if int(np.sum(selected)) >= 5:
                    low_metric = metric(y[selected], prediction[selected])
                    reference_metric = metric(y[selected], baseline[selected])
                    low_similarity[bin_name] = {
                        "rows": int(np.sum(selected)),
                        "delta_r2": float(low_metric["r2"] - reference_metric["r2"]),
                    }
            low_deltas = [float(item["delta_r2"]) for item in low_similarity.values()]
            report = {
                "r2": model_metric["r2"],
                "mae": model_metric["mae"],
                "delta_r2": float(model_metric["r2"] - target_report["baseline_r2"]),
                "positive_folds": int(sum(item["delta_r2"] > 0.0 for item in fold_reports[name])),
                "bootstrap_lower": group_bootstrap_lower(delta, groups),
                "min_low_similarity_delta": min(low_deltas) if low_deltas else None,
                "folds": fold_reports[name],
                "low_similarity": low_similarity,
            }
            report["pass"] = bool(
                report["delta_r2"] >= 0.01
                and report["positive_folds"] >= 4
                and report["bootstrap_lower"] > 0.0
                and (report["min_low_similarity_delta"] is None or report["min_low_similarity_delta"] >= 0.0)
            )
            target_report["models"][name] = report
            metric_rows.append({
                "target": target,
                "model": name,
                "r2": report["r2"],
                "delta_r2": report["delta_r2"],
                "positive_folds": report["positive_folds"],
                "bootstrap_lower": report["bootstrap_lower"],
                "min_low_similarity_delta": report["min_low_similarity_delta"],
                "pass": report["pass"],
            })
        target_report["selected_model"] = max(MODEL_NAMES, key=lambda name: target_report["models"][name]["r2"])
        reports[target] = target_report

    passing_targets = [target for target, report in reports.items() if any(report["models"][name]["pass"] for name in MODEL_NAMES)]
    audit = {
        "schema_version": "ppp.round2.graph-tree-specialist-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C020-20260803-1915-tanimoto-variants-v3",
        "official_inputs": inputs,
        "targets": reports,
        "passing_targets": passing_targets,
        "decision": "component_pass" if passing_targets else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_assignments.csv", index=False)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.graph-tree-specialist.v1",
        "seed": 2026,
        "folds": 5,
        "targets": list(TARGETS),
        "models": list(MODEL_NAMES),
        "morgan_count_radii": [1, 2, 3],
        "morgan_bits": 2048,
        "official_inputs": inputs,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
    ]) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", audit)
    (run_dir / "decision.md").write_text(
        f"# R2-C021 graph tree specialist decision\n\nDecision: **{audit['decision']}**.\n\nNo local_eval or external target was read.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in (
        "config.json", "environment.txt", "metrics.csv", "fold_assignments.csv",
        "metrics.json", "decision.md", "command.txt", "protocol.json",
    )]
    (run_dir / "artifact_manifest.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "experiment_id": run_dir.name,
        "decision": audit["decision"],
        "passing_targets": passing_targets,
        "summary": {
            target: {
                "selected_model": report["selected_model"],
                "selected_delta_r2": report["models"][report["selected_model"]]["delta_r2"],
                "positive_folds": report["models"][report["selected_model"]]["positive_folds"],
                "bootstrap_lower": report["models"][report["selected_model"]]["bootstrap_lower"],
                "min_low_similarity_delta": report["models"][report["selected_model"]]["min_low_similarity_delta"],
            }
            for target, report in reports.items()
        },
        "elapsed_seconds": audit["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
