#!/usr/bin/env python3
"""Audit a completed Round 2 clean experiment directory.

This is a lightweight metadata/schema/hash checker.  It does not train, infer,
read local_eval external_labels, contact Kaggle, or inspect hidden labels.  Its purpose is
to make post-watchdog terminal review repeatable before any component is used by
the compound assembler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def audit_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "artifact_manifest.sha256"
    if not path.exists():
        return {"present": False, "pass": False, "errors": ["missing_artifact_manifest"]}
    errors: list[str] = []
    checked = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            expected, name = line.split("  ", 1)
        except ValueError:
            errors.append(f"line_{line_no}_malformed")
            continue
        if name.startswith("SOURCE "):
            if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                errors.append(f"line_{line_no}_bad_source_hash")
            continue
        artifact = run_dir / name
        if not artifact.exists():
            errors.append(f"line_{line_no}_missing_{name}")
            continue
        actual = sha256_file(artifact)
        if actual != expected:
            errors.append(f"line_{line_no}_hash_mismatch_{name}")
        checked += 1
    return {"present": True, "pass": not errors, "checked_files": checked, "errors": errors}


def audit_run(root: Path, data_dir: Path, run_dir: Path, allow_incomplete: bool) -> dict[str, Any]:
    protocol = load_json(run_dir / "protocol.json")
    metrics = load_json(run_dir / "metrics.json")
    report: dict[str, Any] = {
        "schema_version": "ppp.round2.terminal-artifact-audit.v1",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "protocol_present": protocol is not None,
        "metrics_present": metrics is not None,
        "allow_incomplete": allow_incomplete,
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    if protocol is None:
        errors.append("missing_or_invalid_protocol_json")
    if metrics is None:
        if allow_incomplete:
            report["state"] = "incomplete_no_metrics"
            report["pass"] = False
            return report
        errors.append("missing_or_invalid_metrics_json")
        report["state"] = "invalid_no_metrics"
        report["pass"] = False
        return report

    report["state"] = "terminal_metrics_present"
    if metrics.get("official_only") is not True:
        errors.append("official_only_not_true")
    for key in ("external_label_file_read", "local_eval_read", "kaggle_compute", "kaggle_upload"):
        if key not in metrics:
            warnings.append(f"missing_{key}_flag")
        elif metrics.get(key) is not False:
            errors.append(f"{key}_not_false")
    if metrics.get("kaggle_submission", False) is not False:
        errors.append("kaggle_submission_not_false")
    elif "kaggle_submission" not in metrics:
        warnings.append("missing_kaggle_submission_flag_legacy_neutral")
    if metrics.get("pretrained_weights", False) is not False:
        errors.append("pretrained_weights_not_false")

    test = pd.read_csv(data_dir / "test.csv", usecols=["id", "target_type"])
    expected_ids = test["id"].astype(int).to_numpy()
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        errors.append("missing_predictions_csv")
    else:
        predictions = pd.read_csv(predictions_path)
        report["prediction_rows"] = int(len(predictions))
        if not {"id", "target"}.issubset(predictions.columns):
            errors.append("predictions_missing_id_target_columns")
        else:
            pred_ids = predictions["id"].astype(int).to_numpy()
            pred_values = predictions["target"].to_numpy(float)
            report["prediction_unique_ids"] = bool(pd.Series(pred_ids).is_unique)
            report["prediction_finite_targets"] = bool(np.isfinite(pred_values).all())
            report["prediction_exact_order"] = bool(np.array_equal(pred_ids, expected_ids))
            report["prediction_id_set_match"] = bool(set(pred_ids.tolist()) == set(expected_ids.tolist()))
            if len(predictions) != len(test):
                errors.append("prediction_row_count_mismatch")
            if not report["prediction_unique_ids"]:
                errors.append("prediction_duplicate_ids")
            if not report["prediction_finite_targets"]:
                errors.append("prediction_nonfinite_targets")
            if not report["prediction_id_set_match"]:
                errors.append("prediction_id_set_mismatch")
            if not report["prediction_exact_order"]:
                errors.append("prediction_order_mismatch")

    if int(metrics.get("complete_output_rows", -1)) != len(test):
        errors.append("metrics_complete_output_rows_mismatch")
    if metrics.get("complete_output_order_pass", True) is not True:
        errors.append("metrics_complete_output_order_not_pass")
    parity = metrics.get("parent_replay_parity")
    if isinstance(parity, dict) and parity.get("pass") is not True:
        errors.append("parent_replay_parity_not_pass")

    manifest = audit_manifest(run_dir)
    report["manifest"] = manifest
    if not manifest["pass"]:
        errors.append("artifact_manifest_not_pass")

    report["mean_parent_r2"] = metrics.get("mean_parent_r2")
    report["mean_candidate_r2"] = metrics.get("mean_candidate_r2")
    report["mean_gain"] = metrics.get("mean_gain")
    report["banked_targets"] = metrics.get("banked_targets", [])
    report["full_candidate_gate_pass"] = metrics.get("full_candidate_gate_pass")
    report["goal_0_95_met"] = bool(metrics.get("goal_0_95_met", False))
    report["pass"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()

    report = audit_run(root, data_dir, run_dir, args.allow_incomplete)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report.get("pass") is True:
        return 0
    if args.allow_incomplete and report.get("state") == "incomplete_no_metrics":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
