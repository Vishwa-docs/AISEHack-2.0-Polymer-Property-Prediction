#!/usr/bin/env python3
"""C287 current-only fixed model-zoo candidates.

This runner creates a small, predeclared set of no-archive full-test candidates
from official current train/test only. It does not read archive labels, local_eval
files, prior prediction CSVs, pretrained weights, or Kaggle services.

The purpose is to generate method-diverse frozen candidates for post-freeze
local_eval scoring and per-target portfolio selection without local_eval-tuned weights.
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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
VARIANTS = ("ridge_a1", "ridge_a10", "ridge_a50", "extra_trees", "hgb", "tanimoto_k7", "tanimoto_k25", "mean5", "median5")
CONFIG: dict[str, Any] = {
    "seed": 20260807,
    "morgan_bits": 4096,
    "text_features": 65536,
    "svd_components": 160,
    "dense_abs_limit": 1.0e12,
    "extra_trees_estimators": 800,
    "extra_trees_min_leaf_large": 2,
    "extra_trees_min_leaf_sparse": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def clip(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return reference.clip_prediction(y, np.asarray(pred, dtype=np.float64))


def variant_models(target: str, seed: int) -> dict[str, Any]:
    large = target in {"tg", "egc"}
    leaf = CONFIG["extra_trees_min_leaf_large"] if large else CONFIG["extra_trees_min_leaf_sparse"]
    return {
        "ridge_a1": Ridge(alpha=1.0),
        "ridge_a10": Ridge(alpha=10.0),
        "ridge_a50": Ridge(alpha=50.0),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=int(CONFIG["extra_trees_estimators"]),
            min_samples_leaf=int(leaf),
            max_features=0.65,
            random_state=seed,
            n_jobs=4,
        ),
        "hgb": HistGradientBoostingRegressor(
            max_iter=360 if large else 240,
            learning_rate=0.035,
            max_leaf_nodes=31 if large else 15,
            min_samples_leaf=18 if large else 6,
            l2_regularization=0.05,
            random_state=seed,
        ),
    }


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
    sparse_full = sparse.hstack(sparse_parts, format="csr")
    n_components = min(int(CONFIG["svd_components"]), sparse_full.shape[0] - 1, sparse_full.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=int(CONFIG["seed"]))
    sparse_svd = svd.fit_transform(sparse_full).astype(np.float64, copy=False)
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(CONFIG["morgan_bits"]))
    predictions: dict[str, pd.Series] = {
        variant: pd.Series(index=test["id"].to_numpy(int), dtype=float)
        for variant in VARIANTS
    }
    target_reports: dict[str, Any] = {}
    for target in TARGETS:
        target_train = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_test = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        train_index = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
        test_index = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
        y = target_train["target"].to_numpy(float)
        dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
        _, _, train_dense_scaled, test_dense_scaled = reference.fit_dense_preprocessor(
            dense,
            train_index,
            test_index,
            absolute_limit=float(CONFIG["dense_abs_limit"]),
        )
        dense_scaler = StandardScaler()
        train_svd = dense_scaler.fit_transform(sparse_svd[train_index])
        test_svd = dense_scaler.transform(sparse_svd[test_index])
        x_train = np.hstack([train_dense_scaled, train_svd])
        x_test = np.hstack([test_dense_scaled, test_svd])
        ids = target_test["id"].to_numpy(int)
        arms: dict[str, np.ndarray] = {}
        for name, model in variant_models(target, int(CONFIG["seed"]) + 19 * TARGETS.index(target)).items():
            model.fit(x_train, y)
            arms[name] = clip(y, model.predict(x_test))
            predictions[name].loc[ids] = arms[name]
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[train_index] = y
        arms["tanimoto_k7"] = clip(
            y,
            reference.tanimoto_prediction(fingerprints, y_global, train_index, test_index, k=7, krr_alpha=0.05),
        )
        arms["tanimoto_k25"] = clip(
            y,
            reference.tanimoto_prediction(fingerprints, y_global, train_index, test_index, k=25, krr_alpha=0.05),
        )
        predictions["tanimoto_k7"].loc[ids] = arms["tanimoto_k7"]
        predictions["tanimoto_k25"].loc[ids] = arms["tanimoto_k25"]
        stack5 = np.vstack([arms[name] for name in ("ridge_a10", "ridge_a50", "extra_trees", "hgb", "tanimoto_k7")])
        predictions["mean5"].loc[ids] = np.mean(stack5, axis=0)
        predictions["median5"].loc[ids] = np.median(stack5, axis=0)
        target_reports[target] = {
            "train_rows": int(len(target_train)),
            "test_rows": int(len(target_test)),
            "variants": list(VARIANTS),
        }

    # Current-train exact duplicate overrides are official and branch-safe.
    raw_map = reference.unique_mapping(raw_labels, ["smiles", "target_type"])
    canonical_map = reference.unique_mapping(raw_labels, ["canonical", "target_type"])
    output_records: dict[str, Any] = {}
    for variant, series in predictions.items():
        if series.isna().any():
            raise RuntimeError(f"{variant} produced missing predictions")
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
        if len(submission) != 4940 or not np.array_equal(submission["id"].to_numpy(int), np.arange(1, 4941)) or not np.isfinite(submission["target"].to_numpy(float)).all():
            raise RuntimeError(f"{variant} output validation failed")
        path = out_dir / f"R2-C287-ZOO-{variant}-without_archive-20260807.csv"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        submission.to_csv(path, index=False)
        output_records[variant] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": int(len(submission)),
            "official_current_overrides": int(override_count),
        }
    report = {
        "schema_version": "ppp.round2.c287.current-only-model-zoo.v1",
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
        "config": CONFIG,
        "inputs": inputs,
        "feature_report": {
            "keys": int(len(keys)),
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "sparse_svd_components": int(n_components),
            "sparse_svd_explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        },
        "target_reports": target_reports,
        "outputs": output_records,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", CONFIG)
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nGenerated {len(output_records)} frozen current-only model-zoo candidates. No archive labels, local_eval, pretrained weights, Kaggle action, or prior prediction input.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    manifest.extend([f"{record['sha256']}  OUTPUT {name}" for name, record in output_records.items()])
    manifest.append(f"{sha256_file(Path(__file__))}  SOURCE tools/round2_c287_current_only_model_zoo.py")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "outputs": output_records, "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
