#!/usr/bin/env python3
"""Build F22 eligible archive broad equal-combo candidate.

This is the C270-free archive source map frozen from:
experiments/LOCAL_DIAGNOSTIC_ONLY/
  R2-RESCORE-20260807-eligible-no-c270-no-f20-f21-equal-combo-inventory.json

The assembler is local_eval-free: it only reads official test.csv for row/target
alignment and previously frozen complete prediction CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_MAP = {
    "tg": ["experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv"],
    "egc": ["experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv"],
    "egb": [
        "experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv",
        "experiments/final_submission_runs/with_archive/R2-C288-NC-PROJECTION-over-F19-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F01-COMPOUND-with_archive-20260807.csv",
    ],
    "ei": [
        "experiments/final_submission_runs/with_archive/R2-F02-B3-COMPOUND-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F10-PORTFOLIO-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F06-PI1M-with_archive-candidate.csv",
    ],
    "eea": [
        "experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv",
        "experiments/final_submission_runs/with_archive/R2-C288-NC-PROJECTION-over-F19-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F02-B3-COMPOUND-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F02-COMPOUND-with_archive-candidate.csv",
        "experiments/final_submission_runs/with_archive/R2-F10-PORTFOLIO-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F01-COMPOUND-with_archive-20260807.csv",
    ],
    "nc": ["experiments/final_submission_runs/with_archive/R2-F12-WEAK-EQUAL-ENSEMBLE-with_archive-20260807.csv"],
    "eps": [
        "experiments/final_submission_runs/with_archive/R2-C288-NC-PROJECTION-over-F19-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F19-FIXED-EQUAL-BLENDS-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F12-WEAK-EQUAL-ENSEMBLE-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-F13-WEAK-TRIPLE-ENSEMBLE-with_archive-20260807.csv",
        "experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv",
        "experiments/final_submission_runs/with_archive/R2-F02-COMPOUND-with_archive-candidate.csv",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(path: Path) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden input path: {path}")


def load_prediction(path: Path, ids: np.ndarray) -> np.ndarray:
    guard(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid candidate schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid candidate IDs: {path}")
    values = frame["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite candidate values: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    test_path = Path(args.test_csv).resolve()
    guard(test_path)
    test = pd.read_csv(test_path)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected test.csv schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")

    target_type = test["target_type"].astype(str).str.lower().to_numpy()
    loaded: dict[str, np.ndarray] = {}
    result = np.empty(len(ids), dtype=np.float64)
    for target, rels in SOURCE_MAP.items():
        arrays = []
        for rel in rels:
            if rel not in loaded:
                loaded[rel] = load_prediction(Path(rel).resolve(), ids)
            arrays.append(loaded[rel])
        mask = target_type == target
        result[mask] = np.mean(np.vstack([arr[mask] for arr in arrays]), axis=0)

    if not np.isfinite(result).all():
        raise RuntimeError("Output has non-finite predictions")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    paths = sorted({Path(rel).resolve() for rels in SOURCE_MAP.values() for rel in rels})
    record = {
        "schema_version": "ppp.round2.f22-eligible-broad-equal-combo.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "source_inventory": "experiments/LOCAL_DIAGNOSTIC_ONLY/R2-RESCORE-20260807-eligible-no-c270-no-f20-f21-equal-combo-inventory.json",
        "excluded_components": ["R2-C270-EI-DIAGNOSTIC", "R2-F20", "R2-F21"],
        "local_eval_read_by_assembler": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "target_source_map": SOURCE_MAP,
        "official_test": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "inputs": {str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"branch": "with_archive", "output": record["output"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
