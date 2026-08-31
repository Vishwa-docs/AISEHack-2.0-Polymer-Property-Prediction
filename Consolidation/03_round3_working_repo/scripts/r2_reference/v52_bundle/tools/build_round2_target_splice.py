#!/usr/bin/env python3
"""Assemble a target-level splice from frozen local_eval-free candidate CSVs.

This is a mechanical branch-local assembler: start from a base CSV, then for
each requested target replace only the official test rows whose target_type
matches that target with values from a complete source CSV.  It reads no local_eval
or external_label files and does not choose rows dynamically.
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


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


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


def require_branch_path(path: Path, branch: str, *, role: str) -> None:
    """Require branch-local candidate/output paths for compound assembly."""
    low = str(path).lower()
    branch_token = f"/{branch.lower()}/"
    opposite = "without_archive" if branch == "with_archive" else "with_archive"
    opposite_token = f"/{opposite}/"
    if opposite_token in low:
        raise RuntimeError(f"{role} path crosses branch boundary for {branch}: {path}")
    if branch_token not in low:
        raise RuntimeError(f"{role} path must be explicitly under {branch}: {path}")


def load_candidate(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"]:
        raise RuntimeError(f"Invalid candidate schema: {path}")
    if len(frame) != len(ids):
        raise RuntimeError(f"Invalid candidate row count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Candidate IDs do not match official test order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Candidate contains non-finite predictions: {path}")
    return frame


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise RuntimeError(f"Expected target=path source, got: {value}")
    target, raw_path = value.split("=", 1)
    target = target.strip().lower()
    if target not in TARGETS:
        raise RuntimeError(f"Invalid splice target: {target}")
    return target, Path(raw_path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--source", action="append", default=[], help="target=csv_path; repeatable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path in (test_path, base_path, output, manifest):
        guard(path)
    require_branch_path(base_path, args.branch, role="base candidate")
    require_branch_path(output, args.branch, role="output")
    require_branch_path(manifest, args.branch, role="manifest")
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    test = pd.read_csv(test_path)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    target_type = test["target_type"].astype(str).str.lower().to_numpy(object)

    base = load_candidate(base_path, ids)
    result = base["target"].to_numpy(float).copy()
    source_records: dict[str, Any] = {}
    seen: set[str] = set()
    for item in args.source:
        target, path = parse_source(item)
        if target in seen:
            raise RuntimeError(f"Duplicate source target: {target}")
        seen.add(target)
        require_branch_path(path, args.branch, role=f"{target} source candidate")
        source = load_candidate(path, ids)
        mask = target_type == target
        result[mask] = source["target"].to_numpy(float)[mask]
        source_records[target] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "changed_rows": int(mask.sum()),
        }

    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite assembled output")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.target-splice.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "official_test_path": str(test_path),
        "official_test_sha256": sha256_file(test_path),
        "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "sources": source_records,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"], "sources": source_records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
