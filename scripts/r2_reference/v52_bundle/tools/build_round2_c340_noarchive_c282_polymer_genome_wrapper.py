#!/usr/bin/env python3
"""C340 no-archive C282 Polymer Genome hierarchical residual wrapper.

This reuses the C279 hierarchical residual implementation but swaps its parent
from the archive-enabled C050 artifacts to current-only C282 artifacts.  The
wrapper writes C279-compatible parent OOF/test files under a fresh run
directory, runs the nested residual portfolio, then copies the generated
submission to the requested noarchive output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import round2_c279_polymer_genome_hierarchical_portfolio as c279


DEFAULT_C282_DIR = "experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "with_archive", "/archive/"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden/cross-branch path: {path}")
    if allow_output and "without_archive" not in low:
        raise RuntimeError(f"C340 output must be in without_archive namespace: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--c282-dir", default=DEFAULT_C282_DIR)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    c282_dir = Path(args.c282_dir).resolve()
    guard_path(run_dir)
    for path in (output, manifest):
        guard_path(path, allow_output=True)
    guard_path(c282_dir)
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse run directory: {run_dir}")
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    run_dir.mkdir(parents=True, exist_ok=False)

    c282_oof_path = c282_dir / "oof_predictions.csv"
    c282_test_path = c282_dir / "test_predictions_detail.csv"
    if not c282_oof_path.is_file() or not c282_test_path.is_file():
        raise FileNotFoundError("C282 parent OOF/test artifacts are required")
    oof = pd.read_csv(c282_oof_path)
    test_detail = pd.read_csv(c282_test_path)
    required_oof = ["canonical", "target_type", "target", "prediction"]
    if not all(column in oof.columns for column in required_oof):
        raise RuntimeError("Unexpected C282 OOF schema")
    if not {"id", "target"}.issubset(test_detail.columns):
        raise RuntimeError("Unexpected C282 test detail schema")
    parent_oof = oof[["canonical", "target_type", "target", "prediction"]].rename(
        columns={"prediction": "candidate_prediction"}
    )
    parent_oof["group"] = parent_oof["canonical"].astype(str)
    parent_test = test_detail[["id", "target"]].copy()
    parent_oof_path = run_dir / "parent_c282_oof_for_c279.csv"
    parent_test_path = run_dir / "parent_c282_test_for_c279.csv"
    parent_oof.to_csv(parent_oof_path, index=False)
    parent_test.to_csv(parent_test_path, index=False)

    c279.RUN = run_dir
    c279.PARENT_OOF = parent_oof_path
    c279.PARENT_TEST = parent_test_path
    c279.main()

    generated = run_dir / "predictions.csv"
    candidate = pd.read_csv(generated)
    if list(candidate.columns) != ["id", "target"] or len(candidate) != 4940:
        raise RuntimeError("Generated C340 candidate schema failed")
    if candidate["id"].duplicated().any() or not np.array_equal(candidate["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Generated C340 candidate ID order failed")
    if not np.isfinite(candidate["target"].to_numpy(float)).all():
        raise RuntimeError("Generated C340 candidate has non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output, index=False)

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["schema_version"] = "ppp.round2.c340.noarchive-c282-polymer-genome-wrapper.v1"
    metrics["branch"] = "without_archive"
    metrics["archive_labels_used"] = False
    metrics["archive_file_read"] = False
    metrics["parent"] = "C282 current-only reference OOF/test transformed to C279-compatible schema"
    metrics["output"] = {"path": str(output), "sha256": sha256_file(output), "rows": int(len(candidate))}
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    record = {
        "schema_version": "ppp.round2.c340.noarchive-c282-polymer-genome-wrapper.manifest.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "local_eval_read_by_builder": False,
        "archive_labels_used": False,
        "archive_file_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "inputs": {
            "c282_oof": {"path": str(c282_oof_path), "sha256": sha256_file(c282_oof_path), "bytes": c282_oof_path.stat().st_size},
            "c282_test_detail": {"path": str(c282_test_path), "sha256": sha256_file(c282_test_path), "bytes": c282_test_path.stat().st_size},
            "c279_source": {"path": str(Path(c279.__file__).resolve()), "sha256": sha256_file(Path(c279.__file__).resolve())},
        },
        "run_dir": str(run_dir),
        "metrics": {
            "mean_parent_r2": metrics.get("mean_parent_r2"),
            "mean_candidate_r2": metrics.get("mean_candidate_r2"),
            "mean_gain": metrics.get("mean_gain"),
        },
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(candidate)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "metrics": record["metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
