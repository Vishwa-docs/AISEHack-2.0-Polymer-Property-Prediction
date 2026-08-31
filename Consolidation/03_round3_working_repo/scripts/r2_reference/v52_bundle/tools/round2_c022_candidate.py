#!/usr/bin/env python3
"""Freeze a full candidate by combining C019 with the passing C022 EPS route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from sklearn.impute import SimpleImputer

import initial_reference_pipeline as reference
from round2_graph_tree_specialist import graph_counts, model_factory


RDLogger.DisableLog("rdApp.*")
TARGET = "eps"
THRESHOLD = 0.70
C019_CANDIDATE = "submissions/Sandman_ppp_round2_C019_three_target_route_20260803.csv"
C019_DETAIL = "experiments/CLEAN_OFFICIAL_ONLY/R2-C019-20260803-1900-three-target-route/c001_reproduction/test_predictions_detail.csv"
OUTPUT_NAME = "Sandman_ppp_round2_C022_eps_graph_route_20260803.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def nearest_to_train(test_fps: list[Any], train_fps: list[Any]) -> np.ndarray:
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in test_fps], dtype=np.float64)


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
        raise RuntimeError(f"Pre-created run directory is required: {run_dir}")
    output_path = root / "submissions" / OUTPUT_NAME
    existing_candidate_sha256 = None
    if output_path.exists():
        existing_candidate_sha256 = sha256_file(output_path)

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
    candidate_matrix = np.hstack([graph, target_dense])
    target_train = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    target_test_mask = test["target_type"].to_numpy() == TARGET
    target_test = test[target_test_mask].reset_index(drop=True)
    train_indices = np.asarray([key_to_index[key] for key in target_train["canonical"]], dtype=np.int64)
    test_indices = np.asarray([key_to_index[key] for key in target_test["canonical"]], dtype=np.int64)
    train_x = np.asarray(candidate_matrix[train_indices], dtype=np.float64).copy()
    test_x = np.asarray(candidate_matrix[test_indices], dtype=np.float64).copy()
    train_x[~np.isfinite(train_x)] = np.nan
    test_x[~np.isfinite(test_x)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(train_x)
    test_x = imputer.transform(test_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    model = model_factory("graph_catboost", 0)
    model.fit(train_x[:, keep], target_train["target"].to_numpy(float))
    graph_prediction = reference.clip_prediction(
        target_train["target"].to_numpy(float),
        np.asarray(model.predict(test_x[:, keep]), dtype=np.float64),
    )
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    nearest = nearest_to_train(
        [fingerprints[index] for index in test_indices],
        [fingerprints[index] for index in train_indices],
    )
    base_candidate = pd.read_csv(root / C019_CANDIDATE)
    detail = pd.read_csv(root / C019_DETAIL)
    if not np.array_equal(base_candidate["id"].to_numpy(), test["id"].to_numpy()):
        raise RuntimeError("C019 candidate IDs do not match official test order")
    if not np.array_equal(detail["id"].to_numpy(), test["id"].to_numpy()):
        raise RuntimeError("C019 detail IDs do not match official test order")
    candidate = base_candidate.copy()
    positions = np.flatnonzero(target_test_mask)
    model_rows = detail.loc[target_test_mask, "override"].to_numpy() == "model"
    changed = (nearest < THRESHOLD) & model_rows
    candidate.loc[positions[changed], "target"] = graph_prediction[changed]
    if len(candidate) != len(test) or not np.array_equal(candidate["id"].to_numpy(), test["id"].to_numpy()):
        raise RuntimeError("Candidate row integrity failed")
    if candidate["id"].duplicated().any() or not np.isfinite(candidate["target"].to_numpy(float)).all():
        raise RuntimeError("Candidate contains duplicate IDs or non-finite targets")
    candidate_sha256_before_write = sha256_file(output_path) if output_path.exists() else None
    if candidate_sha256_before_write is None:
        candidate.to_csv(output_path, index=False)
    elif candidate_sha256_before_write != sha256_file(output_path):
        raise RuntimeError("Existing candidate hash changed during audit")
    c022_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    c019_metrics = json.loads((root / "experiments/CLEAN_OFFICIAL_ONLY/R2-C019-20260803-1900-three-target-route/metrics.json").read_text(encoding="utf-8"))
    c019_grouped_mean = 0.873325726796651
    prospective_mean = float(c019_grouped_mean + (c022_metrics["route_delta_r2"] - 0.013554191897311885) / 7.0)
    audit = {
        "schema_version": "ppp.round2.c022-candidate.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent_candidate": C019_CANDIDATE,
        "source_component": "C022 fixed EPS graph route",
        "official_inputs": inputs,
        "target": TARGET,
        "threshold": THRESHOLD,
        "test_target_rows": int(len(target_test)),
        "changed_rows": int(np.sum(changed)),
        "official_override_rows_preserved": int(np.sum(~model_rows)),
        "candidate": {
            "path": str(output_path.relative_to(root)),
            "rows": int(len(candidate)),
            "sha256": sha256_file(output_path),
            "preexisting_on_audit_retry": existing_candidate_sha256 is not None,
        },
        "c019_grouped_prospective_mean_r2": c019_grouped_mean,
        "c022_eps_route_delta_r2": float(c022_metrics["route_delta_r2"]),
        "prospective_seven_target_mean_r2": prospective_mean,
        "local_eval_status": "not_read; candidate frozen before local_eval evaluation",
        "decision": "candidate_generated_pending_notebook_parity",
    }
    write_json(run_dir / "candidate_generation.json", audit)
    (run_dir / "candidate_environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "candidate_command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
