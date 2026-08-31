#!/usr/bin/env python3
"""Create an local_eval-free reflected source CSV: reflected = 2 * base - source.

This supports signed-blend diagnostics without modifying the C355 convex blend
builder. A positive blend from base toward this reflected source is equivalent
to a negative blend from base away from the original source. Selection remains
the separate post-freeze local_eval scoring step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(path: Path, branch: str, *, role: str, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    opposite = "without_archive" if branch == "with_archive" else "with_archive"
    if f"/{opposite}/" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for {branch}: {path}")
    if f"/{branch}/" not in low:
        raise RuntimeError(f"{role} path must stay under /{branch}/: {path}")
    if allow_output and "Polymer Prediction Challenge Round 2" not in str(path):
        raise RuntimeError(f"Output outside Round 2 boundary: {path}")


def load_candidate(path: Path, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid candidate schema/count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Candidate IDs do not match base: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Candidate contains non-finite targets: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    branch = args.branch
    base_path = Path(args.base_csv).resolve()
    source_path = Path(args.source_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path, role in (
        (base_path, "base candidate"),
        (source_path, "source candidate"),
        (output, "output"),
        (manifest, "manifest"),
    ):
        guard(path, branch, role=role, allow_output=role in {"output", "manifest"})
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")

    base = pd.read_csv(base_path)
    if list(base.columns) != ["id", "target"] or base["id"].duplicated().any():
        raise RuntimeError(f"Invalid base candidate: {base_path}")
    ids = base["id"].to_numpy(int)
    if len(ids) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected base IDs")
    source = load_candidate(source_path, ids)
    base_values = base["target"].to_numpy(float)
    source_values = source["target"].to_numpy(float)
    reflected = 2.0 * base_values - source_values
    if not np.isfinite(reflected).all():
        raise RuntimeError("Reflected source contains non-finite targets")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": reflected})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.c415.reflected-source.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": branch,
        "method": "reflected_source_equals_2_times_base_minus_source",
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "source": {"path": str(source_path), "sha256": sha256_file(source_path), "bytes": source_path.stat().st_size},
        "max_abs_delta_from_base": float(np.max(np.abs(reflected - base_values))),
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"branch": branch, "output": record["output"], "max_abs_delta_from_base": record["max_abs_delta_from_base"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
