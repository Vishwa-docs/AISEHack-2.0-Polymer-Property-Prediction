#!/usr/bin/env python3
"""Bounded official-only polymer-specific feature-view diagnostic for Round 2."""

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
from rdkit import Chem, DataStructs, RDLogger
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
ARMS = ("rich_dense_histgb", "rich_sparse_ridge")
EXPECTED_INPUTS = reference.EXPECTED_HASHES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(y, prediction))


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


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


def feature_views(root: Path, keys: list[str], molecules: list[Any]) -> tuple[np.ndarray, dict[str, Any]]:
    tools_dir = root.parent / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(tools_dir))
    import polymer_official_train_eval_loop as rich  # noqa: PLC0415

    original_desc, original_names = reference.descriptor_matrix(molecules)
    original_physical, physical_names = reference.physical_matrix(molecules, keys)
    capped_molecules, cap_status = rich.capped_descriptor_mols(keys, molecules)
    capped_desc, capped_names = reference.descriptor_matrix(capped_molecules)
    periodic_molecules = [rich.periodic_closure_mol(key, molecule) for key, molecule in zip(keys, molecules, strict=True)]
    periodic_desc, periodic_names = reference.descriptor_matrix(periodic_molecules)
    backbone, backbone_names, backbone_report = rich.backbone_sidechain_matrix(molecules)
    arrays = [original_desc, original_physical, capped_desc, periodic_desc, backbone]
    dense = np.hstack(arrays).astype(np.float64, copy=False)
    dense[~np.isfinite(dense)] = np.nan
    report = {
        "source": "official Round 2 SMILES only",
        "view_counts": {
            "original_rdkit": len(original_names),
            "original_physical": len(physical_names),
            "capped_rdkit": len(capped_names),
            "periodic_rdkit": len(periodic_names),
            "backbone_sidechain": len(backbone_names),
            "total": int(dense.shape[1]),
        },
        "capped_status": cap_status,
        "backbone_report": backbone_report,
        "nonfinite_values": int(np.count_nonzero(~np.isfinite(dense))),
    }
    return dense, report


def rich_predict(
    dense: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    y: np.ndarray,
    train_index: np.ndarray,
    prediction_index: np.ndarray,
    target: str,
    seed: int,
) -> np.ndarray:
    _, _, train_scaled, prediction_scaled = reference.fit_dense_preprocessor(
        dense,
        train_index,
        prediction_index,
        absolute_limit=float(reference.DEFAULT_CONFIG["dense_abs_limit"]),
    )
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=0.10,
        random_state=seed + TARGETS.index(target),
    )
    model.fit(train_scaled, y[train_index])
    return np.asarray(model.predict(prediction_scaled), dtype=np.float64)


def rich_sparse_predict(
    dense: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    y: np.ndarray,
    train_index: np.ndarray,
    prediction_index: np.ndarray,
    target: str,
) -> np.ndarray:
    _, _, train_scaled, prediction_scaled = reference.fit_dense_preprocessor(
        dense,
        train_index,
        prediction_index,
        absolute_limit=float(reference.DEFAULT_CONFIG["dense_abs_limit"]),
    )
    train_x = sparse.hstack([part[train_index] for part in sparse_parts] + [sparse.csr_matrix(train_scaled)], format="csr")
    prediction_x = sparse.hstack([part[prediction_index] for part in sparse_parts] + [sparse.csr_matrix(prediction_scaled)], format="csr")
    model = Ridge(alpha=30.0, solver="lsqr", max_iter=5000, tol=1e-4)
    model.fit(train_x, y[train_index])
    return np.asarray(model.predict(prediction_x), dtype=np.float64)


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
    pi1m_path = data_dir / "PI1M.csv"
    keys = sorted(set(train["canonical"]) | set(archive["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    rich_dense_base, view_report = feature_views(root, keys, molecules)
    base_dense, _ = reference.descriptor_matrix(molecules)
    base_physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([base_dense, base_physical]).astype(np.float64, copy=False)
    raw_labels, pooled = reference.build_label_pool(train, archive)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    key_to_index = {key: index for index, key in enumerate(keys)}
    reference_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        groups = frame["canonical"].to_numpy(object)
        indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[indices] = y
        folds = folds_for(groups)
        target_base_dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        target_rich_dense = reference.target_dense_features(rich_dense_base, cross_values, cross_available, target)
        baseline = np.full(len(y), np.nan, dtype=np.float64)
        candidates = np.full((len(y), len(ARMS)), np.nan, dtype=np.float64)
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            base = reference.predict_base_models(
                target_base_dense,
                sparse_parts,
                fingerprints,
                y_global,
                indices[train_rows],
                indices[validation_rows],
                reference.DEFAULT_CONFIG,
                target,
            )
            baseline[validation_rows] = base @ np.asarray(
                [reference_report["validation"]["target_reports"][target]["blend_weights"][name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")],
                dtype=np.float64,
            ) + float(reference_report["validation"]["target_reports"][target]["blend_intercept"])
            candidates[validation_rows, 0] = rich_predict(target_rich_dense, sparse_parts, y, indices[train_rows], indices[validation_rows], target, 2026)
            candidates[validation_rows, 1] = rich_sparse_predict(target_rich_dense, sparse_parts, y, indices[train_rows], indices[validation_rows], target)
        nearest = nearest_similarity(fingerprints, folds)
        report: dict[str, Any] = {
            "rows": int(len(y)),
            "baseline_r2": r2(y, baseline),
            "arms": {},
            "folds": [],
            "low_similarity": {},
        }
        for arm, name in enumerate(ARMS):
            report["arms"][name] = {
                "r2": r2(y, candidates[:, arm]),
                "delta_r2": r2(y, candidates[:, arm]) - report["baseline_r2"],
            }
        for fold in range(5):
            selected = folds == fold
            fold_report = {"fold": fold, "baseline_r2": r2(y[selected], baseline[selected])}
            for arm, name in enumerate(ARMS):
                fold_report[f"{name}_r2"] = r2(y[selected], candidates[selected, arm])
                fold_report[f"{name}_delta_r2"] = fold_report[f"{name}_r2"] - fold_report["baseline_r2"]
            report["folds"].append(fold_report)
        for name_bin, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
            selected = (nearest >= lower) & (nearest < upper)
            if int(np.sum(selected)) < 5:
                continue
            baseline_r2 = r2(y[selected], baseline[selected])
            report["low_similarity"][name_bin] = {"rows": int(np.sum(selected)), "baseline_r2": baseline_r2, "arms": {}}
            for arm, name in enumerate(ARMS):
                arm_r2 = r2(y[selected], candidates[selected, arm])
                report["low_similarity"][name_bin]["arms"][name] = {f"{name}_r2": arm_r2, f"{name}_delta_r2": arm_r2 - baseline_r2}
        for arm, name in enumerate(ARMS):
            delta = candidates[:, arm] - baseline
            positive_folds = int(sum(fold[f"{name}_delta_r2"] > 0 for fold in report["folds"]))
            low_values = [value["arms"][name][f"{name}_delta_r2"] for value in report["low_similarity"].values()]
            report["arms"][name]["positive_folds"] = positive_folds
            report["arms"][name]["bootstrap_lower"] = group_bootstrap_lower(delta, groups)
            report["arms"][name]["min_low_similarity_delta"] = min(low_values) if low_values else None
            report["arms"][name]["pass"] = bool(
                report["arms"][name]["delta_r2"] >= 0.01
                and positive_folds >= 4
                and report["arms"][name]["bootstrap_lower"] > 0.0
                and (not low_values or min(low_values) >= 0.0)
            )
            metric_rows.append({"target": target, "arm": name, **report["arms"][name]})
        report["selected_arm"] = max(ARMS, key=lambda name: report["arms"][name]["r2"])
        reports[target] = report
    passing_targets = [target for target, report in reports.items() if any(report["arms"][name]["pass"] for name in ARMS)]
    audit = {
        "schema_version": "ppp.round2.polymer-views-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C010-20260803-1742-pi1m-scratch-control-v2",
        "official_inputs": inputs,
        "pi1m_read": False,
        "feature_views": view_report,
        "targets": reports,
        "passing_targets": passing_targets,
        "decision": "component_pass" if passing_targets else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.polymer-views.v1", "seed": 2026, "folds": 5, "arms": list(ARMS), "views": view_report["view_counts"], "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C011 polymer views decision\n\nDecision: **{audit['decision']}**. No candidate changed in this component diagnostic.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "metrics.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "passing_targets": passing_targets, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
