#!/usr/bin/env python3
"""Build F20/F21 fixed broad equal-combo candidates.

The source maps are frozen from the post-freeze inventory written to
experiments/LOCAL_DIAGNOSTIC_ONLY/R2-RESCORE-20260807-broad-equal-combo-inventory.json.
This assembler itself is local_eval-free: it reads only official test.csv for
target alignment and previously frozen complete prediction CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


MAPS = {
    "with_archive": {
        "tg": ["experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv"],
        "egc": ["experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv"],
        "egb": [
            "experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv",
            "experiments/final_submission_runs/with_archive/R2-C270-EI-DIAGNOSTIC-with_archive-20260807.csv",
            "experiments/final_submission_runs/with_archive/R2-F01-COMPOUND-with_archive-20260807.csv",
        ],
        "ei": [
            "experiments/final_submission_runs/with_archive/R2-F02-B3-COMPOUND-with_archive-20260807.csv",
            "experiments/final_submission_runs/with_archive/R2-F10-PORTFOLIO-with_archive-20260807.csv",
            "experiments/final_submission_runs/with_archive/R2-F06-PI1M-with_archive-candidate.csv",
        ],
        "eea": [
            "experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv",
            "experiments/final_submission_runs/with_archive/R2-C270-EI-DIAGNOSTIC-with_archive-20260807.csv",
            "experiments/final_submission_runs/with_archive/R2-C288-NC-PROJECTION-over-F19-with_archive-20260807.csv",
            "experiments/final_submission_runs/with_archive/R2-F02-B3-COMPOUND-with_archive-20260807.csv",
            "experiments/final_submission_runs/with_archive/R2-F02-COMPOUND-with_archive-candidate.csv",
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
    },
    "without_archive": {
        "tg": [
            "experiments/final_submission_runs/without_archive/R2-C284-PI1M-SVD-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C286v4-ARTIFACT-STACK-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_extra_trees-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_huber-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_random_forest-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C282-CURRENT-ONLY-REFERENCE-20260807.csv",
        ],
        "egc": [
            "experiments/final_submission_runs/without_archive/R2-C284-PI1M-SVD-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C286v4-ARTIFACT-STACK-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_extra_trees-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_huber-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_random_forest-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C285-PI1M-SVD-WEAK-RESIDUAL-without_archive-20260807.csv",
        ],
        "egb": ["experiments/final_submission_runs/without_archive/R2-C286v4-ARTIFACT-STACK-without_archive-20260807.csv"],
        "ei": [
            "experiments/final_submission_runs/without_archive/R2-F18-FIXED-EQUAL-BLENDS-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_extra_trees-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-F06-PI1M-without_archive-candidate.csv",
            "experiments/final_submission_runs/without_archive/R2-F10-PORTFOLIO-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_random_forest-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C284-PI1M-SVD-without_archive-20260807.csv",
        ],
        "eea": ["experiments/final_submission_runs/without_archive/R2-C285-PI1M-SVD-WEAK-RESIDUAL-without_archive-20260807.csv"],
        "nc": [
            "experiments/final_submission_runs/without_archive/R2-F18-FIXED-EQUAL-BLENDS-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_extra_trees-without_archive-20260807.csv",
            "experiments/final_submission_runs/without_archive/R2-C286v4-ARTIFACT-STACK-without_archive-20260807.csv",
        ],
        "eps": ["experiments/final_submission_runs/without_archive/R2-F18-FIXED-EQUAL-BLENDS-without_archive-20260807.csv"],
    },
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
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
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
    source_map = MAPS[args.branch]

    loaded: dict[str, np.ndarray] = {}
    result = np.empty(len(ids), dtype=np.float64)
    for target, rels in source_map.items():
        arrays = []
        for rel in rels:
            path = Path(rel).resolve()
            if rel not in loaded:
                loaded[rel] = load_prediction(path, ids)
            arrays.append(loaded[rel])
        mask = target_type == target
        result[mask] = np.mean(np.vstack([arr[mask] for arr in arrays]), axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError("Output has non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    paths = sorted({Path(rel).resolve() for rels in source_map.values() for rel in rels})
    record = {
        "schema_version": "ppp.round2.fixed-broad-equal-combo.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "local_eval_read_by_assembler": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "target_source_map": source_map,
        "official_test": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "inputs": {str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
