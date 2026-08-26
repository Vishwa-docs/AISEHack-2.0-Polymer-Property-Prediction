#!/usr/bin/env python3
"""Nested, threshold-selection-safe EPS graph route diagnostic."""

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
from round2_graph_tree_specialist import fold_matrix, graph_counts, model_factory


RDLogger.DisableLog("rdApp.*")
TARGET = "eps"
THRESHOLDS = (0.0, 0.30, 0.50, 0.70, 1.01)
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fold_ids(groups: np.ndarray, n_splits: int) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def nearest_to_train(left: list[Any], right: list[Any]) -> np.ndarray:
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, right)) for fp in left], dtype=np.float64)


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, route: np.ndarray, groups: np.ndarray, seed: int = 2026) -> float:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(500):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in selected])
        if len(rows) < 2 or np.var(y[rows]) <= 1.0e-15:
            continue
        values.append(float(r2_score(y[rows], route[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def route_for_threshold(candidate: np.ndarray, baseline: np.ndarray, nearest: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(nearest < threshold, candidate, baseline)


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
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, TARGET)
    graph = np.hstack([graph_counts(molecules, radius, 2048) for radius in (1, 2, 3)])
    matrix = np.hstack([graph, target_dense])
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    groups = frame["canonical"].to_numpy(object)
    indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    outer = fold_ids(groups, 5)
    oof_baseline = np.full(len(y), np.nan)
    oof_candidate = np.full(len(y), np.nan)
    oof_nearest = np.full(len(y), np.nan)
    chosen_thresholds: list[dict[str, Any]] = []

    def baseline_predict(train_rows: np.ndarray, validation_rows: np.ndarray) -> np.ndarray:
        base = reference.predict_base_models(
            target_dense,
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
        intercept = float(c001_report["validation"]["target_reports"][TARGET]["blend_intercept"])
        return base @ blend_weights + intercept

    for outer_fold in range(5):
        outer_train = np.flatnonzero(outer != outer_fold)
        outer_validation = np.flatnonzero(outer == outer_fold)
        inner = fold_ids(groups[outer_train], 4)
        inner_scores = {threshold: [] for threshold in THRESHOLDS}
        for inner_fold in range(4):
            local_train = outer_train[inner != inner_fold]
            local_validation = outer_train[inner == inner_fold]
            base_inner = baseline_predict(local_train, local_validation)
            train_x, validation_x = fold_matrix(matrix[indices], local_train, local_validation)
            model = model_factory("graph_catboost", 0)
            model.fit(train_x, y[local_train])
            candidate_inner = reference.clip_prediction(y[local_train], model.predict(validation_x))
            nearest_inner = nearest_to_train(
                [fingerprints[indices[row]] for row in local_validation],
                [fingerprints[indices[row]] for row in local_train],
            )
            for threshold in THRESHOLDS:
                route = route_for_threshold(candidate_inner, base_inner, nearest_inner, threshold)
                inner_scores[threshold].append(float(r2_score(y[local_validation], route)))
        selected_threshold = max(THRESHOLDS, key=lambda threshold: float(np.mean(inner_scores[threshold])))
        outer_base = baseline_predict(outer_train, outer_validation)
        train_x, validation_x = fold_matrix(matrix[indices], outer_train, outer_validation)
        model = model_factory("graph_catboost", 0)
        model.fit(train_x, y[outer_train])
        outer_candidate = reference.clip_prediction(y[outer_train], model.predict(validation_x))
        outer_nearest = nearest_to_train(
            [fingerprints[indices[row]] for row in outer_validation],
            [fingerprints[indices[row]] for row in outer_train],
        )
        oof_baseline[outer_validation] = outer_base
        oof_candidate[outer_validation] = outer_candidate
        oof_nearest[outer_validation] = outer_nearest
        selected_route = route_for_threshold(outer_candidate, outer_base, outer_nearest, selected_threshold)
        chosen_thresholds.append({
            "outer_fold": outer_fold,
            "selected_threshold": selected_threshold,
            "inner_mean_r2": {str(threshold): float(np.mean(inner_scores[threshold])) for threshold in THRESHOLDS},
            "outer_baseline_r2": float(r2_score(y[outer_validation], outer_base)),
            "outer_route_r2": float(r2_score(y[outer_validation], selected_route)),
            "outer_delta_r2": float(r2_score(y[outer_validation], selected_route) - r2_score(y[outer_validation], outer_base)),
            "outer_changed_rows": int(np.sum(outer_nearest < selected_threshold)),
        })

    route = np.full(len(y), np.nan)
    for row in range(len(y)):
        threshold = chosen_thresholds[int(outer[row])]["selected_threshold"]
        route[row] = route_for_threshold(
            np.asarray([oof_candidate[row]]),
            np.asarray([oof_baseline[row]]),
            np.asarray([oof_nearest[row]]),
            threshold,
        )[0]
    route_r2 = float(r2_score(y, route))
    baseline_r2 = float(r2_score(y, oof_baseline))
    route_delta = route_r2 - baseline_r2
    outer_positive = int(sum(row["outer_delta_r2"] > 0.0 for row in chosen_thresholds))
    bootstrap = bootstrap_r2_lower(y, oof_baseline, route, groups)
    low_bins = {}
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (oof_nearest >= lower) & (oof_nearest < upper)
        if int(np.sum(selected)) >= 5:
            low_bins[name] = {
                "rows": int(np.sum(selected)),
                "delta_r2": float(r2_score(y[selected], route[selected]) - r2_score(y[selected], oof_baseline[selected])),
            }
    min_low = min((value["delta_r2"] for value in low_bins.values()), default=None)
    passed = bool(route_delta >= 0.01 and outer_positive >= 4 and bootstrap > 0.0 and (min_low is None or min_low >= 0.0))
    audit = {
        "schema_version": "ppp.round2.nested-eps-graph-route-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C022-20260803-1945-eps-graph-similarity-route",
        "official_inputs": inputs,
        "target": TARGET,
        "threshold_grid": list(THRESHOLDS),
        "baseline_r2": baseline_r2,
        "route_r2": route_r2,
        "route_delta_r2": route_delta,
        "outer_positive_folds": outer_positive,
        "outer_bootstrap_lower": bootstrap,
        "low_similarity": low_bins,
        "selected_thresholds": chosen_thresholds,
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_nested_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({
        "canonical": groups,
        "outer_fold": outer,
        "nearest_tanimoto": oof_nearest,
        "y": y,
        "baseline": oof_baseline,
        "candidate": oof_candidate,
        "route": route,
    }).to_csv(run_dir / "nested_oof.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.nested-eps-graph-route.v1",
        "seed": 2026,
        "outer_folds": 5,
        "inner_folds": 4,
        "threshold_grid": list(THRESHOLDS),
        "official_inputs": inputs,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C024 nested EPS graph route\n\nDecision: **{audit['decision']}**. Thresholds were selected inside inner folds only.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(
        "stage=official_input_hashes_pass\nstage=nested_inner_threshold_selection\nstage=untouched_outer_scoring\nstage=corrected_r2_bootstrap\n"
        f"decision={audit['decision']}\n",
        encoding="utf-8",
    )
    manifest_paths = [run_dir / name for name in (
        "config.json", "environment.txt", "nested_oof.csv", "metrics.json",
        "decision.md", "command.txt", "run.log", "protocol.json",
    )]
    (run_dir / "artifact_manifest.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "experiment_id": run_dir.name,
        "decision": audit["decision"],
        "baseline_r2": baseline_r2,
        "route_r2": route_r2,
        "route_delta_r2": route_delta,
        "outer_positive_folds": outer_positive,
        "outer_bootstrap_lower": bootstrap,
        "selected_thresholds": [row["selected_threshold"] for row in chosen_thresholds],
        "low_similarity": low_bins,
        "elapsed_seconds": audit["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
