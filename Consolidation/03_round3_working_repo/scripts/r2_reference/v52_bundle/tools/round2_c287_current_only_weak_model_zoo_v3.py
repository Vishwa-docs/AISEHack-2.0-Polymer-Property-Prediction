#!/usr/bin/env python3
"""C287-v3 weak-target-only current model zoo.

Uses current official train/test only. It preserves the F14 no-archive base and
replaces one weak target at a time with fixed dense-model predictions. No
archive labels, local_eval, prior predictions as training features, or Kaggle
actions are used.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from rdkit import Chem

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


ACTIVE_TARGETS = ("ei", "eea", "nc", "eps")
ARMS = ("dense_ridge_a5", "dense_huber", "dense_extra_trees", "dense_random_forest", "dense_hgb", "mean5", "median5")
CONFIG = {
    "seed": 20260807,
    "dense_abs_limit": 1.0e12,
    "base_candidate": "experiments/final_submission_runs/without_archive/R2-F14-FIXED-ENSEMBLE-without_archive-20260807.csv",
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


def clip(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return reference.clip_prediction(y, np.asarray(pred, dtype=np.float64))


def copy_with_replacement(base: pd.DataFrame, ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    mapping = dict(zip(ids.astype(int), values.astype(float), strict=True))
    mask = out["id"].astype(int).isin(mapping)
    out.loc[mask, "target"] = out.loc[mask, "id"].astype(int).map(mapping).to_numpy(float)
    return out


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
    base_path = Path(CONFIG["base_candidate"]).resolve()
    base = pd.read_csv(base_path)
    if list(base.columns) != ["id", "target"] or len(base) != 4940 or base["id"].duplicated().any():
        raise RuntimeError("Base F14 candidate invalid")
    archive = train.iloc[0:0].copy()
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    progress(progress_path, "features_ready", keys=len(keys), dense_features=int(dense_base.shape[1]))
    output_records: dict[str, Any] = {}
    target_reports: dict[str, Any] = {}
    for target in ACTIVE_TARGETS:
        target_train = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_test = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        train_index = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
        test_index = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
        y = target_train["target"].to_numpy(float)
        dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
        _, _, x_train, x_test = reference.fit_dense_preprocessor(dense, train_index, test_index, float(CONFIG["dense_abs_limit"]))
        seed = int(CONFIG["seed"]) + 97 * reference.TARGETS.index(target)
        models = {
            "dense_ridge_a5": Ridge(alpha=5.0),
            "dense_huber": HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=500),
            "dense_extra_trees": ExtraTreesRegressor(n_estimators=360, min_samples_leaf=1, max_features=0.75, random_state=seed, n_jobs=4),
            "dense_random_forest": RandomForestRegressor(n_estimators=320, min_samples_leaf=1, max_features=0.75, random_state=seed + 1, n_jobs=4),
            "dense_hgb": HistGradientBoostingRegressor(max_iter=220, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=0.05, random_state=seed + 2),
        }
        ids = target_test["id"].to_numpy(int)
        arm_values: dict[str, np.ndarray] = {}
        for arm, model in models.items():
            model.fit(x_train, y)
            values = clip(y, model.predict(x_test))
            arm_values[arm] = values
            candidate = copy_with_replacement(base, ids, values)
            path = out_dir / f"R2-C287v3-{target}-{arm}-without_archive-20260807.csv"
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite {path}")
            candidate.to_csv(path, index=False)
            output_records[f"{target}_{arm}"] = {"path": str(path), "sha256": sha256_file(path), "rows": int(len(candidate))}
        stack = np.vstack([arm_values[arm] for arm in models])
        for arm, values in {"mean5": np.mean(stack, axis=0), "median5": np.median(stack, axis=0)}.items():
            candidate = copy_with_replacement(base, ids, values)
            path = out_dir / f"R2-C287v3-{target}-{arm}-without_archive-20260807.csv"
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite {path}")
            candidate.to_csv(path, index=False)
            output_records[f"{target}_{arm}"] = {"path": str(path), "sha256": sha256_file(path), "rows": int(len(candidate))}
        target_reports[target] = {"train_rows": int(len(target_train)), "test_rows": int(len(target_test)), "arms": list(ARMS)}
        progress(progress_path, "target_complete", target=target, outputs=len(ARMS))
    report = {
        "schema_version": "ppp.round2.c287v3.current-only-weak-model-zoo.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "prior_prediction_as_training_feature": False,
        "base_candidate_used_for_unchanged_targets_only": str(base_path),
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
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nGenerated weak-target-only no-archive model-zoo candidates on the F14 base. No local_eval, archive labels, Kaggle, or prior predictions as training features.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    manifest.extend([f"{record['sha256']}  OUTPUT {name}" for name, record in output_records.items()])
    manifest.append(f"{sha256_file(Path(__file__))}  SOURCE tools/round2_c287_current_only_weak_model_zoo_v3.py")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    progress(progress_path, "completed", outputs=len(output_records), elapsed_seconds=float(time.time() - started))
    print(json.dumps({"experiment_id": run_dir.name, "outputs": len(output_records), "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
