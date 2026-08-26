#!/usr/bin/env python3
"""C1400 no-archive regenerated-C282 PGFP target-gated candidate.

This is a final-notebook-parity repair of the C340/C279 PGFP path.  It does
not read saved parent predictions.  Instead it regenerates the current-only
C282 parent from official Round 2 train/test files inside this run directory,
runs the C279 Polymer Genome / morphology residual module against that parent,
and emits a complete no-archive CSV where only OOF-gated targets are replaced.

No archive, local_eval, external_label, Kaggle, pretrained, or external target inputs are
read by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import round2_c279_polymer_genome_hierarchical_portfolio as c279
import round2_c282_current_only_reference as c282


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, role: str, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "with_archive", "/archive/"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden/cross-branch {role} path: {path}")
    if allow_output and "without_archive" not in low:
        raise RuntimeError(f"{role} path must stay in without_archive namespace: {path}")
    if "Polymer Prediction Challenge Round 2" not in str(path.resolve()):
        raise RuntimeError(f"{role} path is outside Round 2 boundary: {path}")


def parse_targets(raw: str) -> set[str]:
    selected = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = selected.difference(TARGETS)
    if unknown:
        raise RuntimeError(f"Unknown target(s): {sorted(unknown)}")
    if not selected:
        raise RuntimeError("At least one target must be requested")
    return selected


def parent_oof_for_c279(c282_oof_path: Path, output_path: Path) -> dict[str, Any]:
    oof = pd.read_csv(c282_oof_path)
    required = {"canonical", "target_type", "target", "prediction"}
    if not required.issubset(oof.columns):
        raise RuntimeError(f"Unexpected C282 OOF schema: {c282_oof_path}")
    parent = oof[["canonical", "target_type", "target", "prediction"]].rename(
        columns={"prediction": "candidate_prediction"}
    )
    parent["target_type"] = parent["target_type"].astype(str).str.lower()
    parent["group"] = parent["canonical"].astype(str)
    if set(parent["target_type"]) != set(TARGETS):
        raise RuntimeError("C282 OOF target set mismatch")
    if not np.isfinite(parent[["target", "candidate_prediction"]].to_numpy(float)).all():
        raise RuntimeError("C282 OOF contains non-finite numeric values")
    parent.to_csv(output_path, index=False)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "rows": int(len(parent)),
    }


def parent_test_for_c279(c282_detail_path: Path, output_path: Path) -> dict[str, Any]:
    detail = pd.read_csv(c282_detail_path)
    required = {"id", "target"}
    if not required.issubset(detail.columns):
        raise RuntimeError(f"Unexpected C282 test detail schema: {c282_detail_path}")
    parent = detail[["id", "target"]].copy()
    if len(parent) != 4940:
        raise RuntimeError("Unexpected C282 test row count")
    if parent["id"].duplicated().any() or not np.array_equal(parent["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C282 test detail ID contract failed")
    if not np.isfinite(parent["target"].to_numpy(float)).all():
        raise RuntimeError("C282 test detail contains non-finite predictions")
    parent.to_csv(output_path, index=False)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "rows": int(len(parent)),
    }


def target_passes(report: dict[str, Any], *, min_delta: float, min_positive_folds: int, min_fold_delta: float) -> bool:
    folds = report.get("folds", [])
    fold_deltas = [float(row["delta_r2"]) for row in folds]
    return bool(
        float(report["delta_r2"]) >= float(min_delta)
        and int(report["positive_folds"]) >= int(min_positive_folds)
        and fold_deltas
        and min(fold_deltas) >= float(min_fold_delta)
        and float(report.get("full_weight", 0.0)) > 0.0
    )


def write_clean_protocol(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--targets", default="eea,egb,tg")
    parser.add_argument("--min-target-delta", type=float, default=0.002)
    parser.add_argument("--min-positive-folds", type=int, default=4)
    parser.add_argument("--min-fold-delta", type=float, default=-0.003)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path, role in ((data_dir, "data-dir"), (run_dir, "run-dir")):
        guard_path(path, role=role)
    for path, role in ((output, "output"), (manifest, "manifest")):
        guard_path(path, role=role, allow_output=True)
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse run directory: {run_dir}")
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")

    run_dir.mkdir(parents=True, exist_ok=False)
    c282_dir = run_dir / "c282_current_only_parent"
    c279_dir = run_dir / "pgfp_c279_target_source"
    c282_submission = run_dir / "c282_current_only_submission.csv"

    c282_report = c282.run_current_only_reference(
        data_dir=data_dir,
        output_path=c282_submission,
        run_dir=c282_dir,
    )
    c282_oof_path = c282_dir / "oof_predictions.csv"
    c282_detail_path = c282_dir / "test_predictions_detail.csv"
    parent_oof_path = c279_dir / "parent_c282_oof_for_c279.csv"
    parent_test_path = c279_dir / "parent_c282_test_for_c279.csv"
    c279_dir.mkdir(parents=True, exist_ok=False)
    parent_oof_record = parent_oof_for_c279(c282_oof_path, parent_oof_path)
    parent_test_record = parent_test_for_c279(c282_detail_path, parent_test_path)

    c279.RUN = c279_dir
    c279.PARENT_OOF = parent_oof_path
    c279.PARENT_TEST = parent_test_path
    c279.main()

    c279_metrics_path = c279_dir / "metrics.json"
    c279_metrics = json.loads(c279_metrics_path.read_text(encoding="utf-8"))
    target_reports = c279_metrics["target_reports"]
    selected_targets = []
    rejected_targets = []
    for target in sorted(targets):
        report = target_reports[target]
        passes = target_passes(
            report,
            min_delta=float(args.min_target_delta),
            min_positive_folds=int(args.min_positive_folds),
            min_fold_delta=float(args.min_fold_delta),
        )
        entry = {
            "target": target,
            "passes": passes,
            "parent_r2": float(report["parent_r2"]),
            "candidate_r2": float(report["candidate_r2"]),
            "delta_r2": float(report["delta_r2"]),
            "positive_folds": int(report["positive_folds"]),
            "full_alpha": float(report["full_alpha"]),
            "full_weight": float(report["full_weight"]),
            "min_fold_delta": float(min(float(row["delta_r2"]) for row in report["folds"])),
        }
        if passes:
            selected_targets.append(entry)
        else:
            rejected_targets.append(entry)

    test = pd.read_csv(data_dir / "test.csv")
    test["target_type"] = test["target_type"].astype(str).str.lower()
    parent_submission = pd.read_csv(c282_submission)
    pgfp_submission = pd.read_csv(c279_dir / "predictions.csv")
    for name, frame in (("parent", parent_submission), ("pgfp", pgfp_submission)):
        if list(frame.columns) != ["id", "target"] or len(frame) != 4940:
            raise RuntimeError(f"{name} submission schema failed")
        if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), np.arange(1, 4941)):
            raise RuntimeError(f"{name} submission ID contract failed")
        if not np.isfinite(frame["target"].to_numpy(float)).all():
            raise RuntimeError(f"{name} submission contains non-finite predictions")
    selected_names = {entry["target"] for entry in selected_targets}
    result = parent_submission.copy()
    pgfp_values = pgfp_submission["target"].to_numpy(float)
    mask = test["target_type"].isin(selected_names).to_numpy()
    result.loc[mask, "target"] = pgfp_values[mask]
    if not np.isfinite(result["target"].to_numpy(float)).all():
        raise RuntimeError("Final C1400 output contains non-finite predictions")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    clean_protocol = {
        "schema_version": "ppp.round2.c1400.noarchive-regen-c282-pgfp-target-gated.protocol.v1",
        "experiment_id": run_dir.name,
        "branch": "without_archive",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "Regenerate C282 current-only parent, run PGFP/morphology residual source, apply only nested OOF-gated targets",
        "targets_requested": sorted(targets),
        "gate": {
            "min_target_delta": float(args.min_target_delta),
            "min_positive_folds": int(args.min_positive_folds),
            "min_fold_delta": float(args.min_fold_delta),
            "requires_full_weight_gt": 0.0,
        },
        "selected_targets": selected_targets,
        "rejected_targets": rejected_targets,
        "note": "The imported C279 module writes an inherited protocol with stale archive wording; this C1400 protocol is the controlling sanitized protocol for this run.",
    }
    write_clean_protocol(run_dir / "protocol.json", clean_protocol)

    record = {
        "schema_version": "ppp.round2.c1400.noarchive-regen-c282-pgfp-target-gated.manifest.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "params": {
            "targets": sorted(targets),
            "min_target_delta": float(args.min_target_delta),
            "min_positive_folds": int(args.min_positive_folds),
            "min_fold_delta": float(args.min_fold_delta),
        },
        "c282_report": {
            "submission": c282_report["submission"],
            "mean_oof_r2": c282_report["validation"]["mean_selected_oof_r2"],
            "official_overrides": c282_report["official_overrides"]["total_overrides"],
            "elapsed_seconds": c282_report["elapsed_seconds"],
        },
        "pgfp_metrics": {
            "mean_parent_r2": c279_metrics["mean_parent_r2"],
            "mean_candidate_r2": c279_metrics["mean_candidate_r2"],
            "mean_gain": c279_metrics["mean_gain"],
            "target_reports": target_reports,
        },
        "parent_oof_for_pgfp": parent_oof_record,
        "parent_test_for_pgfp": parent_test_record,
        "selected_targets": selected_targets,
        "rejected_targets": rejected_targets,
        "inputs": {
            "train.csv": {
                "path": str(data_dir / "train.csv"),
                "sha256": sha256_file(data_dir / "train.csv"),
                "bytes": (data_dir / "train.csv").stat().st_size,
            },
            "test.csv": {
                "path": str(data_dir / "test.csv"),
                "sha256": sha256_file(data_dir / "test.csv"),
                "bytes": (data_dir / "test.csv").stat().st_size,
            },
            "c279_source": {"path": str(Path(c279.__file__).resolve()), "sha256": sha256_file(Path(c279.__file__).resolve())},
            "c282_source": {"path": str(Path(c282.__file__).resolve()), "sha256": sha256_file(Path(c282.__file__).resolve())},
        },
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(result)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": record["output"],
                "selected_targets": selected_targets,
                "rejected_targets": rejected_targets,
                "mean_pgfp_oof_gain": c279_metrics["mean_gain"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
