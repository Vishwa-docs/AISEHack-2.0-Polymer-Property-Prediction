#!/usr/bin/env python3
"""Branch-guarded whole-target blend assembler.

For each requested target, replace base values with
    (1 - weight) * base + weight * source
on all rows of that target.  No row-level routing, local_eval, or external_label file is
read.
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
            raise RuntimeError(f"Refusing forbidden path: {path}")


def require_branch_path(path: Path, branch: str, *, role: str) -> None:
    low = str(path).lower()
    token = f"/{branch}/"
    opposite = "without_archive" if branch == "with_archive" else "with_archive"
    if f"/{opposite}/" in low:
        raise RuntimeError(f"{role} crosses branch boundary: {path}")
    if token not in low:
        raise RuntimeError(f"{role} must be explicitly under {branch}: {path}")


def load_candidate(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid candidate schema/count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid candidate ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite candidate values: {path}")
    return frame


def parse_blend(value: str) -> tuple[str, float, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise RuntimeError(f"Expected target=weight=csv_path, got: {value}")
    target = parts[0].strip().lower()
    if target not in TARGETS:
        raise RuntimeError(f"Invalid target: {target}")
    weight = float(parts[1])
    if not (0.0 <= weight <= 1.0):
        raise RuntimeError(f"Invalid blend weight for {target}: {weight}")
    return target, weight, Path(parts[2]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--blend", action="append", default=[], help="target=weight=csv_path; repeatable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path, role in ((test_path, "test"), (base_path, "base"), (output, "output"), (manifest, "manifest")):
        guard(path)
        if role != "test":
            require_branch_path(path, args.branch, role=role)
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
    records: dict[str, Any] = {}
    seen: set[str] = set()
    for raw in args.blend:
        target, weight, source_path = parse_blend(raw)
        if target in seen:
            raise RuntimeError(f"Duplicate blend target: {target}")
        seen.add(target)
        require_branch_path(source_path, args.branch, role=f"{target} source")
        source = load_candidate(source_path, ids)
        mask = target_type == target
        result[mask] = (1.0 - weight) * result[mask] + weight * source["target"].to_numpy(float)[mask]
        records[target] = {
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "weight": weight,
            "rows": int(mask.sum()),
        }

    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite output")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.branch-target-blend.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "official_test_path": str(test_path),
        "official_test_sha256": sha256_file(test_path),
        "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "blends": records,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"], "blends": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
