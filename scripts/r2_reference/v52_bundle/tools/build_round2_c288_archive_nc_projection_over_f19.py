#!/usr/bin/env python3
"""Build C288 archive-branch NC projection over F19.

This assembler is local_eval-free. It reads:
- official test.csv for ID/target alignment;
- frozen F19 archive candidate as the base carrier;
- C252 clean NC component predictions, using the C252 candidate only on rows
  where the selected EPS counterpart was available.

Unsupported NC rows fall back to F19, not to C252's older parent.
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


def guard(path: Path) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden input path: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base", default="experiments/final_submission_runs/with_archive/R2-F19-FIXED-EQUAL-BLENDS-with_archive-20260807.csv")
    parser.add_argument("--c252-component", default="experiments/CLEAN_OFFICIAL_ONLY/R2-C252-20260805-0856-nc-eps-ionic-projection-v1/nc_component_predictions.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    paths = {name: Path(value).resolve() for name, value in vars(args).items() if name in {"test_csv", "base", "c252_component"}}
    for path in paths.values():
        guard(path)
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    test = pd.read_csv(paths["test_csv"])
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected test schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    target_type = test["target_type"].astype(str).str.lower().to_numpy()

    base = pd.read_csv(paths["base"])
    if list(base.columns) != ["id", "target"] or len(base) != len(ids):
        raise RuntimeError("Base schema invalid")
    if base["id"].duplicated().any() or not np.array_equal(base["id"].to_numpy(int), ids):
        raise RuntimeError("Base ID validation failed")
    if not np.isfinite(base["target"].to_numpy(float)).all():
        raise RuntimeError("Base has non-finite predictions")

    component = pd.read_csv(paths["c252_component"])
    required = ["id", "target_type", "selected_eps_available", "parent", "candidate"]
    if list(component.columns) != required:
        raise RuntimeError("C252 component schema invalid")
    component["target_type"] = component["target_type"].astype(str).str.lower()
    if set(component["target_type"]) != {"nc"} or component["id"].duplicated().any():
        raise RuntimeError("C252 component ID/target validation failed")

    out = base.copy()
    nc_ids = set(test.loc[target_type == "nc", "id"].astype(int))
    usable = component[component["selected_eps_available"].astype(bool)].copy()
    if not set(usable["id"].astype(int)).issubset(nc_ids):
        raise RuntimeError("C252 usable IDs are not NC test rows")
    replacement = dict(zip(usable["id"].astype(int), usable["candidate"].astype(float), strict=True))
    mask = out["id"].astype(int).isin(replacement)
    out.loc[mask, "target"] = out.loc[mask, "id"].astype(int).map(replacement).to_numpy(float)
    if not np.isfinite(out["target"].to_numpy(float)).all():
        raise RuntimeError("Output has non-finite predictions")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.c288.archive-nc-projection-over-f19.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "local_eval_read_by_assembler": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "base": str(paths["base"]),
        "component": str(paths["c252_component"]),
        "replacement_target": "nc",
        "replacement_rows": int(len(replacement)),
        "fallback_rows": int((target_type == "nc").sum() - len(replacement)),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "replacement_rows": record["replacement_rows"], "fallback_rows": record["fallback_rows"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
