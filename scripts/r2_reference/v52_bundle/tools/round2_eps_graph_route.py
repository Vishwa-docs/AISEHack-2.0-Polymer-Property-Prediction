#!/usr/bin/env python3
"""One fixed, train-only similarity route for the C021 EPS graph arm."""

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
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
from round2_graph_tree_specialist import (
    fold_matrix,
    graph_counts,
    metric,
    model_factory,
    nearest_similarity,
)


TARGET = "eps"
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
THRESHOLD = 0.70


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def group_bootstrap_r2_lower(
    y: np.ndarray,
    baseline: np.ndarray,
    route: np.ndarray,
    groups: np.ndarray,
    seed: int = 2026,
) -> float:
    unique = np.unique(groups)
    row_groups = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(500):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([row_groups[group] for group in selected_groups])
        if len(rows) < 2 or np.var(y[rows]) <= 1.0e-15:
            continue
        values.append(float(r2_score(y[rows], route[rows]) - r2_score(y[rows], baseline[rows])))
    if not values:
        return float("-inf")
    return float(np.quantile(np.asarray(values), 0.025))


def bin_delta(y: np.ndarray, baseline: np.ndarray, route: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], route[selected]) - r2_score(y[selected], baseline[selected]))


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
    graph = np.hstack([graph_counts(molecules, radius, 2048) for radius in (1, 2, 3)])
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
    folds = folds_for(groups)
    target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, TARGET)
    candidate_matrix = np.hstack([graph[indices], target_dense[indices]])
    baseline = np.full(len(y), np.nan, dtype=np.float64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(5):
        train_rows = np.flatnonzero(folds != fold)
        validation_rows = np.flatnonzero(folds == fold)
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
        baseline[validation_rows] = base @ blend_weights + intercept
        train_x, validation_x = fold_matrix(candidate_matrix, train_rows, validation_rows)
        model = model_factory("graph_catboost", 0)
        model.fit(train_x, y[train_rows])
        candidate[validation_rows] = reference.clip_prediction(y[train_rows], model.predict(validation_x))

    nearest = nearest_similarity([fingerprints[index] for index in indices], folds)
    changed = nearest < THRESHOLD
    route = np.where(changed, candidate, baseline)
    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    route_r2 = float(r2_score(y, route))
    fold_rows = []
    for fold in range(5):
        selected = folds == fold
        fold_rows.append({
            "fold": fold,
            "rows": int(np.sum(selected)),
            "baseline_r2": float(r2_score(y[selected], baseline[selected])),
            "candidate_r2": float(r2_score(y[selected], candidate[selected])),
            "route_r2": float(r2_score(y[selected], route[selected])),
            "route_delta_r2": float(r2_score(y[selected], route[selected]) - r2_score(y[selected], baseline[selected])),
        })
    retained_bins = {}
    for name, lower, upper in (
        ("lt_0.30", 0.0, 0.30),
        ("0.30_0.50", 0.30, 0.50),
        ("0.50_0.70", 0.50, 0.70),
    ):
        selected = (nearest >= lower) & (nearest < upper)
        value = bin_delta(y, baseline, route, selected)
        if value is not None:
            retained_bins[name] = {"rows": int(np.sum(selected)), "delta_r2": value}
    changed_groups = groups[changed]
    changed_bootstrap = group_bootstrap_r2_lower(y[changed], baseline[changed], route[changed], changed_groups) if int(np.sum(changed)) >= 2 else float("-inf")
    route_delta = route_r2 - baseline_r2
    positive_folds = int(sum(row["route_delta_r2"] > 0.0 for row in fold_rows))
    min_retained = min((row["delta_r2"] for row in retained_bins.values()), default=None)
    full_pass = bool(
        route_delta >= 0.01
        and positive_folds >= 4
        and changed_bootstrap > 0.0
        and (min_retained is None or min_retained >= 0.0)
    )
    audit = {
        "schema_version": "ppp.round2.eps-graph-similarity-route-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C021-20260803-1935-graph-tree-specialist",
        "official_inputs": inputs,
        "target": TARGET,
        "threshold": THRESHOLD,
        "rows": int(len(y)),
        "changed_rows": int(np.sum(changed)),
        "baseline_r2": baseline_r2,
        "candidate_r2": candidate_r2,
        "candidate_delta_r2": candidate_r2 - baseline_r2,
        "route_r2": route_r2,
        "route_delta_r2": route_delta,
        "positive_folds": positive_folds,
        "folds": fold_rows,
        "retained_similarity_bins": retained_bins,
        "changed_group_bootstrap_lower": changed_bootstrap,
        "pass": full_pass,
        "decision": "component_pass" if full_pass else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({
        "target": TARGET,
        "canonical": groups,
        "fold": folds,
        "nearest_tanimoto": nearest,
        "y": y,
        "baseline": baseline,
        "candidate": candidate,
        "route": route,
        "changed": changed,
    }).to_csv(run_dir / "oof_route.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.eps-graph-similarity-route.v1",
        "seed": 2026,
        "folds": 5,
        "target": TARGET,
        "threshold": THRESHOLD,
        "official_inputs": inputs,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C022 EPS graph similarity route\n\nDecision: **{audit['decision']}**.\n\nThreshold is fixed at nearest Tanimoto < 0.70; no local_eval was read.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in (
        "config.json", "environment.txt", "oof_route.csv", "folds.csv",
        "metrics.json", "decision.md", "command.txt", "protocol.json",
    )]
    (run_dir / "artifact_manifest.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "experiment_id": run_dir.name,
        "decision": audit["decision"],
        "baseline_r2": baseline_r2,
        "candidate_r2": candidate_r2,
        "route_r2": route_r2,
        "route_delta_r2": route_delta,
        "positive_folds": positive_folds,
        "changed_rows": int(np.sum(changed)),
        "changed_group_bootstrap_lower": changed_bootstrap,
        "retained_similarity_bins": retained_bins,
        "elapsed_seconds": audit["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
