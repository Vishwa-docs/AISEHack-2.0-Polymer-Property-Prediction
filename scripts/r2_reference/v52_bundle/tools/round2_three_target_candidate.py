#!/usr/bin/env python3
"""Generate the frozen clean C019 three-target route candidate."""

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
from rdkit import DataStructs, RDLogger

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
ROUTES = {
    "eps": {"model": "catboost", "alpha": 0.50, "bins": [(0.30, 0.70)]},
    "nc": {"model": "xgboost", "alpha": 0.50, "bins": [(0.50, 0.70)]},
    "ei": {"model": "lightgbm", "alpha": 0.50, "bins": [(0.30, 0.50)]},
}
C018_METRICS = "experiments/CLEAN_OFFICIAL_ONLY/R2-C018-20260803-1850-direction-consistent-route/metrics.json"
SUBMISSION_NAME = "Sandman_ppp_round2_C019_three_target_route_20260803.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fold_matrix(matrix: np.ndarray, train_index: np.ndarray, prediction_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = np.asarray(matrix[train_index], dtype=np.float64).copy()
    prediction_x = np.asarray(matrix[prediction_index], dtype=np.float64).copy()
    limit = float(reference.DEFAULT_CONFIG["dense_abs_limit"])
    train_x[~np.isfinite(train_x) | (np.abs(train_x) > limit)] = np.nan
    prediction_x[~np.isfinite(prediction_x) | (np.abs(prediction_x) > limit)] = np.nan
    train_x = imputer.fit_transform(train_x)
    prediction_x = imputer.transform(prediction_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    return train_x[:, keep], prediction_x[:, keep]


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
        raise RuntimeError(f"Pre-created protocol directory is required: {run_dir}")
    existing = {path.name for path in run_dir.iterdir()}
    if existing - {"protocol.json"}:
        raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    fingerprints = reference.morgan_bits(molecules, 2, 4096)

    tools_dir = root.parent / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(tools_dir))
    import round2_target_tree_zoo_v2 as zoo  # noqa: PLC0415

    c001_output = run_dir / "c001_base_submission.csv"
    c001_runtime = run_dir / "c001_reproduction"
    c001_report = reference.run_pipeline(data_dir, c001_output, c001_runtime)
    base_frame = pd.read_csv(c001_output)
    detail = pd.read_csv(c001_runtime / "test_predictions_detail.csv")
    if not np.array_equal(base_frame["id"].to_numpy(), test["id"].to_numpy()):
        raise RuntimeError("C001 reproduction IDs do not match official test order")
    candidate = base_frame.copy()
    route_reports: dict[str, Any] = {}
    for target in TARGETS:
        if target not in ROUTES:
            route_reports[target] = {"route": "c001", "changed_rows": 0}
            continue
        spec = ROUTES[target]
        target_train = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_test_mask = test["target_type"].to_numpy() == target
        target_test = test[target_test_mask].reset_index(drop=True)
        train_indices = np.asarray([key_to_index[key] for key in target_train["canonical"]], dtype=np.int64)
        test_indices = np.asarray([key_to_index[key] for key in target_test["canonical"]], dtype=np.int64)
        target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        train_x, test_x = fold_matrix(target_dense, train_indices, test_indices)
        model = zoo.model_factory(spec["model"], 2026 + TARGETS.index(target))
        model.fit(train_x, target_train["target"].to_numpy(float))
        arm_prediction = np.asarray(model.predict(test_x), dtype=np.float64)
        target_detail = detail[target_test_mask].reset_index(drop=True)
        baseline_model = target_detail["model_prediction"].to_numpy(float)
        nearest = nearest_to_train([fingerprints[index] for index in test_indices], [fingerprints[index] for index in train_indices])
        eligible = np.zeros(len(target_test), dtype=bool)
        for lower, upper in spec["bins"]:
            eligible |= (nearest >= lower) & (nearest < upper)
        correction = arm_prediction - baseline_model
        model_rows = target_detail["override"].to_numpy() == "model"
        changed = eligible & (correction > 0.0) & model_rows
        positions = np.flatnonzero(target_test_mask)
        candidate.loc[positions[changed], "target"] = baseline_model[changed] + float(spec["alpha"]) * correction[changed]
        route_reports[target] = {
            "model": spec["model"],
            "alpha": spec["alpha"],
            "bins": spec["bins"],
            "test_rows": int(len(target_test)),
            "eligible_rows": int(np.sum(eligible)),
            "changed_rows": int(np.sum(changed)),
            "exact_override_rows_preserved": int(np.sum(eligible & (correction > 0.0) & ~model_rows)),
        }

    candidate_path = root / "submissions" / SUBMISSION_NAME
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_path, index=False)
    if len(candidate) != len(test) or not np.array_equal(candidate["id"].to_numpy(), test["id"].to_numpy()) or candidate["id"].duplicated().any() or not np.isfinite(candidate["target"].to_numpy(float)).all():
        raise RuntimeError("C019 candidate failed full official ID/finite-value integrity")
    override_preserved = int(np.sum(detail["override"].to_numpy() != "model"))
    c018 = json.loads((root / C018_METRICS).read_text(encoding="utf-8"))
    audit = {
        "schema_version": "ppp.round2.three-target-candidate.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C018-20260803-1850-direction-consistent-route",
        "official_inputs": inputs,
        "c001_reproduction": c001_report["submission"],
        "frozen_clean_route_source": {"path": C018_METRICS, "sha256": sha256_file(root / C018_METRICS), "grouped_mean_r2": c018["route_grouped_mean_r2"], "grouped_mean_gain": c018["route_mean_gain"]},
        "routes": ROUTES,
        "route_reports": route_reports,
        "official_override_rows_preserved": override_preserved,
        "candidate": {"path": str(candidate_path.relative_to(root)), "rows": int(len(candidate)), "sha256": sha256_file(candidate_path)},
        "local_eval_status": "not_read; candidate frozen before local_eval evaluation",
        "decision": "candidate_generated_pending_notebook_parity",
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.three-target-candidate.v1", "seed": 2026, "routes": ROUTES, "candidate_name": SUBMISSION_NAME, "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text("# R2-C019 three-target route decision\n\nThe candidate was generated from official Round 2 inputs only. Its bytes are frozen in the audit record. LocalEval evaluation remains a separate post-freeze diagnostic, and notebook parity is still pending.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + f"\n{sha256_file(candidate_path)}  {candidate_path.relative_to(root)}\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "candidate": audit["candidate"], "route_reports": route_reports, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
