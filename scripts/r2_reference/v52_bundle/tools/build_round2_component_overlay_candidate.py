#!/usr/bin/env python3
"""Overlay a one-target component prediction file onto a branch base CSV.

This is a guarded mechanical assembler for local diagnostics.  It reads no
local_eval/external_label files, makes no row-level decisions, and replaces only the
requested official test target rows whose IDs appear in the component file.
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


def guard(path: Path, *, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden path: {path}")
    if not allow_output and "LOCAL_EVAL_ASSISTED" in str(path):
        raise RuntimeError(f"Refusing local_eval-assisted component path: {path}")


def require_branch(path: Path, branch: str, *, role: str) -> None:
    low = str(path).lower()
    opposite = "without_archive" if branch == "with_archive" else "with_archive"
    if f"/{opposite}/" in low:
        raise RuntimeError(f"{role} crosses branch boundary for {branch}: {path}")
    if role in {"base", "output", "manifest"} and f"/{branch}/" not in low:
        raise RuntimeError(f"{role} path must be explicitly under {branch}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--component-csv", required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--value-column", default="candidate_prediction")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    component_path = Path(args.component_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path in (test_path, base_path, component_path):
        guard(path)
    for path in (output, manifest):
        guard(path, allow_output=True)
    for role, path in (("base", base_path), ("output", output), ("manifest", manifest)):
        require_branch(path, args.branch, role=role)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    test = pd.read_csv(test_path)
    base = pd.read_csv(base_path)
    component = pd.read_csv(component_path)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    if list(base.columns) != ["id", "target"] or len(base) != len(test):
        raise RuntimeError("Unexpected base schema")
    if base["id"].duplicated().any() or not np.array_equal(base["id"].to_numpy(int), ids):
        raise RuntimeError("Base ID mismatch")
    if args.value_column not in component.columns or "id" not in component.columns:
        raise RuntimeError(f"Component missing id/{args.value_column}: {component_path}")
    if "target_type" in component.columns:
        bad = component["target_type"].astype(str).str.lower() != args.target
        if bool(bad.any()):
            raise RuntimeError("Component contains non-requested target_type rows")
    target_ids = set(test.loc[test["target_type"].astype(str).str.lower() == args.target, "id"].astype(int))
    component_ids = component["id"].astype(int).tolist()
    if len(component_ids) != len(set(component_ids)):
        raise RuntimeError("Duplicate component IDs")
    if set(component_ids) != target_ids:
        raise RuntimeError(f"Component IDs do not exactly match official {args.target} test IDs")
    component_values = component.set_index(component["id"].astype(int))[args.value_column].astype(float)
    if not np.isfinite(component_values.to_numpy(float)).all():
        raise RuntimeError("Component contains non-finite predictions")

    result = base["target"].to_numpy(float).copy()
    if not np.isfinite(result).all():
        raise RuntimeError("Base contains non-finite predictions")
    changed = 0
    for row_index, row in test.iterrows():
        if str(row["target_type"]).lower() != args.target:
            continue
        value = float(component_values.loc[int(row["id"])])
        if abs(result[int(row_index)] - value) > 1.0e-12:
            changed += 1
        result[int(row_index)] = value
    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite predictions")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record: dict[str, Any] = {
        "schema_version": "ppp.round2.component-overlay-candidate.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "target": args.target,
        "value_column": args.value_column,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "component": {
            "path": str(component_path),
            "sha256": sha256_file(component_path),
            "bytes": component_path.stat().st_size,
            "rows": int(len(component)),
        },
        "official_test": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "changed_rows": int(changed),
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "changed_rows": changed}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
