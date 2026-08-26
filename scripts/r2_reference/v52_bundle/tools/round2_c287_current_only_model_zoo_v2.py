#!/usr/bin/env python3
"""C287-v2 bounded current-only model-zoo candidates.

This replaces the overlong C287-v1. It uses only official current train/test,
emits progress after every target, avoids archive labels/local_eval/prior
predictions, and writes a small fixed set of complete candidate CSVs.
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
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from rdkit import Chem

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
VARIANTS = ("sparse_ridge_a1", "sparse_ridge_a10", "sparse_ridge_a50", "dense_extra_trees", "dense_hgb", "mean5", "median5")
CONFIG = {
    "seed": 20260807,
    "morgan_bits": 4096,
    "text_features": 65536,
    "dense_abs_limit": 1.0e12,
    "extra_trees_estimators": 240,
    "extra_trees_min_leaf_large": 2,
    "extra_trees_min_leaf_sparse": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def progress(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "time": datetime.now().astimezone().isoformat(), **payload}, sort_keys=True, allow_nan=False) + "\n")


def clip(y: np.ndarray, values: np.ndarray) -> np.ndarray:
    return reference.clip_prediction(y, np.asarray(values, dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.time()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    progress(progress_path, "started", experiment_id=run_dir.name)
    data_dir = Path(args.data_dir).resolve()
    train, test, inputs = c282.load_current_only_inputs(data_dir)
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(CONFIG["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(CONFIG["morgan_bits"])),
        reference.text_matrix(keys, int(CONFIG["text_features"])),
    ]
    predictions = {variant: pd.Series(index=test["id"].to_numpy(int), dtype=float) for variant in VARIANTS}
    target_reports: dict[str, Any] = {}
    progress(progress_path, "features_ready", keys=len(keys), dense_features=int(dense_base.shape[1]))
    for target in TARGETS:
        target_train = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_test = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        train_index = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
        test_index = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
        y = target_train["target"].to_numpy(float)
        dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
        _, _, train_scaled, test_scaled = reference.fit_dense_preprocessor(
            dense,
            train_index,
            test_index,
            absolute_limit=float(CONFIG["dense_abs_limit"]),
        )
        x_train = sparse.hstack([part[train_index] for part in sparse_parts] + [sparse.csr_matrix(train_scaled)], format="csr")
        x_test = sparse.hstack([part[test_index] for part in sparse_parts] + [sparse.csr_matrix(test_scaled)], format="csr")
        ids = target_test["id"].to_numpy(int)
        arms: dict[str, np.ndarray] = {}
        for alpha, name in ((1.0, "sparse_ridge_a1"), (10.0, "sparse_ridge_a10"), (50.0, "sparse_ridge_a50")):
            model = Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1e-4)
            model.fit(x_train, y)
            arms[name] = clip(y, model.predict(x_test))
            predictions[name].loc[ids] = arms[name]
        large = target in {"tg", "egc"}
        leaf = int(CONFIG["extra_trees_min_leaf_large"] if large else CONFIG["extra_trees_min_leaf_sparse"])
        et = ExtraTreesRegressor(n_estimators=int(CONFIG["extra_trees_estimators"]), min_samples_leaf=leaf, max_features=0.75, random_state=int(CONFIG["seed"]) + TARGETS.index(target), n_jobs=4)
        et.fit(train_scaled, y)
        arms["dense_extra_trees"] = clip(y, et.predict(test_scaled))
        predictions["dense_extra_trees"].loc[ids] = arms["dense_extra_trees"]
        hgb = HistGradientBoostingRegressor(max_iter=220 if large else 160, learning_rate=0.04, max_leaf_nodes=31 if large else 15, min_samples_leaf=16 if large else 5, l2_regularization=0.1, random_state=int(CONFIG["seed"]) + 31 + TARGETS.index(target))
        hgb.fit(train_scaled, y)
        arms["dense_hgb"] = clip(y, hgb.predict(test_scaled))
        predictions["dense_hgb"].loc[ids] = arms["dense_hgb"]
        stack = np.vstack([arms[name] for name in ("sparse_ridge_a1", "sparse_ridge_a10", "sparse_ridge_a50", "dense_extra_trees", "dense_hgb")])
        predictions["mean5"].loc[ids] = np.mean(stack, axis=0)
        predictions["median5"].loc[ids] = np.median(stack, axis=0)
        target_reports[target] = {"train_rows": int(len(target_train)), "test_rows": int(len(target_test))}
        progress(progress_path, "target_complete", target=target, rows=int(len(target_test)))

    raw_map = reference.unique_mapping(raw_labels, ["smiles", "target_type"])
    canonical_map = reference.unique_mapping(raw_labels, ["canonical", "target_type"])
    output_records: dict[str, Any] = {}
    for variant, series in predictions.items():
        if series.isna().any():
            raise RuntimeError(f"{variant} missing predictions")
        ordered = test[["id", "smiles", "canonical", "target_type"]].copy()
        ordered["target"] = series.loc[ordered["id"].to_numpy(int)].to_numpy(float)
        override_count = 0
        for index, row in ordered.iterrows():
            raw_key = (row["smiles"], row["target_type"])
            canonical_key = (row["canonical"], row["target_type"])
            if raw_key in raw_map:
                ordered.at[index, "target"] = raw_map[raw_key]
                override_count += 1
            elif canonical_key in canonical_map:
                ordered.at[index, "target"] = canonical_map[canonical_key]
                override_count += 1
        submission = ordered[["id", "target"]].copy()
        path = out_dir / f"R2-C287v2-ZOO-{variant}-without_archive-20260807.csv"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        submission.to_csv(path, index=False)
        output_records[variant] = {"path": str(path), "sha256": sha256_file(path), "rows": int(len(submission)), "official_current_overrides": int(override_count)}
    report = {
        "schema_version": "ppp.round2.c287v2.current-only-model-zoo.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "inputs": inputs,
        "config": CONFIG,
        "features": {"keys": int(len(keys)), "rdkit_descriptors": int(len(descriptor_names)), "physical_features": int(len(physical_names))},
        "target_reports": target_reports,
        "outputs": output_records,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", CONFIG)
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nGenerated {len(output_records)} bounded current-only model-zoo candidates. No archive labels, local_eval, prior predictions, Kaggle action, or pretrained weights.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    manifest.extend([f"{record['sha256']}  OUTPUT {name}" for name, record in output_records.items()])
    manifest.append(f"{sha256_file(Path(__file__))}  SOURCE tools/round2_c287_current_only_model_zoo_v2.py")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    progress(progress_path, "completed", outputs=len(output_records), elapsed_seconds=float(time.time() - started))
    print(json.dumps({"experiment_id": run_dir.name, "outputs": output_records, "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
