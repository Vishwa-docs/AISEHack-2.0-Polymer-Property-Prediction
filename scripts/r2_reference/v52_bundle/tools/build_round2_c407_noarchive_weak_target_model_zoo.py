#!/usr/bin/env python3
"""C407 current-only weak-target direct model-zoo source over C404.

This is an local_eval-free source generator for the without-archive branch. It
reads only official Round 2 current train/test files, trains target-local
models from scratch, replaces only requested targets over a branch-local base
CSV, and writes one complete frozen candidate. Target-wise acceptance is
decided later by the separate post-freeze local_eval scorer and splice builder.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import r2_score

import build_round2_c385_archive_weak_target_model_zoo as c385
import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C404-NOARCHIVE-TARGET-SPLICE-C403-EGB-NC-OVER-C401-20260808.csv"
)
SEED = 20260808


def guard_path(path: Path, *, role: str, require_without_archive: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "with_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for no-archive run: {path}")
    if require_without_archive and "/without_archive/" not in low:
        raise RuntimeError(f"{role} path must stay in without_archive namespace: {path}")


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path, role="base candidate", require_without_archive=True)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--targets", default="ei,eps,nc,tg,egc")
    parser.add_argument("--models", default="ridge_200,extra_trees,hist_gbdt,lightgbm")
    parser.add_argument("--morgan-bits", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    base_path = Path(args.base_csv).resolve()
    output = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    guard_path(data_dir, role="data dir")
    guard_path(base_path, role="base candidate", require_without_archive=True)
    guard_path(output, role="output", require_without_archive=True)
    guard_path(run_dir, role="run dir")
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    append_progress(progress_path, {"stage": "started", "created_at": datetime.now().astimezone().isoformat()})

    train, test, inputs = c282.load_current_only_inputs(data_dir)
    archive = train.iloc[0:0].copy()
    _, pooled = reference.build_label_pool(train, archive)
    for frame in (pooled, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    base = load_base(base_path, ids)
    predictions = base["target"].to_numpy(float).copy()

    active_targets = c385.parse_targets(args.targets)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: i for i, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    maccs = c385.maccs_matrix(molecules)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    base_dense = c385.sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs]).astype(np.float32))
    append_progress(progress_path, {"stage": "features_ready", "keys": len(keys), "dense_shape": list(base_dense.shape)})

    target_reports: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    for target in active_targets:
        target_train = pooled[pooled["target_type"].eq(target)].reset_index(drop=True)
        target_test = test[test["target_type"].eq(target)].reset_index(drop=False)
        train_idx = np.asarray([key_to_index[x] for x in target_train["canonical"]], dtype=int)
        test_idx = np.asarray([key_to_index[x] for x in target_test["canonical"]], dtype=int)
        y = target_train["target"].to_numpy(float)
        groups = np.asarray([c385.no_stereo(value) for value in target_train["canonical"].astype(str)], dtype=object)
        dense = c385.sanitize_dense(reference.target_dense_features(base_dense, cross_values, cross_available, target))
        x_train = dense[train_idx]
        x_test = dense[test_idx]
        candidates: dict[str, np.ndarray] = {}
        reports: dict[str, Any] = {}
        selected_factories = c385.parse_models(args.models, c385.model_factories(target))
        for model_name, factory in selected_factories.items():
            model_started = time.time()
            try:
                oof_raw, folds = c385.grouped_oof(factory, x_train, y, groups, int(args.seed) + 100 * TARGETS.index(target))
                oof = c385.clipped_by_train(y, oof_raw)
                reports[model_name] = {
                    "oof_r2": float(r2_score(y, oof)),
                    "folds": folds,
                    "elapsed_seconds": float(time.time() - model_started),
                }
                candidates[model_name] = oof
                append_progress(progress_path, {"stage": "model_oof", "target": target, "model": model_name, "oof_r2": reports[model_name]["oof_r2"]})
            except Exception as exc:
                reports[model_name] = {"error": repr(exc), "elapsed_seconds": float(time.time() - model_started)}
                append_progress(progress_path, {"stage": "model_failed", "target": target, "model": model_name, "error": repr(exc)})
        if not candidates:
            raise RuntimeError(f"No successful models for {target}")
        best_name = max(candidates, key=lambda name: reports[name]["oof_r2"])
        final_model = selected_factories[best_name](int(args.seed) + 999 + TARGETS.index(target))
        final_model.fit(x_train, y)
        final_pred = c385.clipped_by_train(y, np.asarray(final_model.predict(x_test), dtype=np.float64))
        lookup = target_train.groupby("canonical")["target"].mean().to_dict()
        overrides = 0
        for local_pos, row in enumerate(target_test.itertuples(index=False)):
            if row.canonical in lookup:
                final_pred[local_pos] = float(lookup[row.canonical])
                overrides += 1
        positions = target_test["index"].to_numpy(int)
        predictions[positions] = final_pred
        target_reports[target] = {
            "rows": int(len(y)),
            "test_rows": int(len(target_test)),
            "models": reports,
            "selected_model": best_name,
            "selected_oof_r2": float(reports[best_name]["oof_r2"]),
            "exact_current_overrides": int(overrides),
        }
        for rid, pred in zip(target_test["id"].to_numpy(int), final_pred, strict=True):
            detail_rows.append({"id": int(rid), "target_type": target, "target": float(pred), "selected_model": best_name})

    if not np.isfinite(predictions).all():
        raise RuntimeError("Missing/non-finite final predictions")
    submission = pd.DataFrame({"id": ids, "target": predictions})
    if len(submission) != 4940 or not np.array_equal(submission["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Output row contract failed")
    submission.to_csv(output, index=False)
    pd.DataFrame(detail_rows).sort_values("id").to_csv(run_dir / "prediction_detail.csv", index=False)
    report = {
        "schema_version": "ppp.round2.c407.noarchive-weak-target-model-zoo.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": vars(args),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
            "lightgbm": getattr(c385.lgb, "__version__", None) if c385.lgb is not None else None,
        },
        "inputs": inputs,
        "features": {
            "keys": int(len(keys)),
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "morgan_count_radii": [2, 3],
            "morgan_bits_each": int(args.morgan_bits),
            "maccs_bits": 167,
            "dense_shape": list(base_dense.shape),
        },
        "target_reports": target_reports,
        "mean_selected_oof_r2": float(np.mean([target_reports[target]["selected_oof_r2"] for target in active_targets])),
        "output": {"path": str(output), "sha256": c385.sha256_file(output), "rows": int(len(submission)), "bytes": output.stat().st_size},
        "elapsed_seconds": float(time.time() - started),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest_lines.append(f"{c385.sha256_file(path)}  {path.name}")
    manifest_lines.append(f"{c385.sha256_file(output)}  {output}")
    manifest_lines.append(f"{c385.sha256_file(Path(__file__))}  SOURCE tools/build_round2_c407_noarchive_weak_target_model_zoo.py")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    append_progress(progress_path, {"stage": "finished", "output_sha256": report["output"]["sha256"], "elapsed_seconds": report["elapsed_seconds"]})
    print(json.dumps({"output": report["output"], "mean_selected_oof_r2": report["mean_selected_oof_r2"], "targets": active_targets, "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
