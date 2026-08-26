#!/usr/bin/env python3
"""Build a fixed archive weak-target equal ensemble.

This is a single predeclared no-sweep assembly:
- base: archive F10 portfolio;
- ei: equal average of F10 and V2/R2-BEST archive predictions;
- eps: equal average of F10 and C214 EPS-ionic full-amplitude predictions;
- nc: equal average of F10 and V2/R2-BEST archive predictions.

The assembler reads no local_eval files. It is a local research candidate; target
weights are fixed constants, not tuned after scoring.
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, official_ids: np.ndarray, name: str) -> pd.Series:
    if "local_eval" in str(path).lower() or "external_label" in str(path).lower():
        raise RuntimeError(f"Refusing local_eval-like input {name}: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "target"] or len(df) != len(official_ids):
        raise RuntimeError(f"{name} schema/row count invalid")
    if df["id"].duplicated().any() or not np.array_equal(df["id"].to_numpy(int), official_ids):
        raise RuntimeError(f"{name} ID validation failed")
    values = df["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} has non-finite targets")
    return pd.Series(values, index=official_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--f10", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--c214", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    test = pd.read_csv(args.test_csv)
    official_ids = test["id"].to_numpy(int)
    target_type = test["target_type"].astype(str).str.lower().to_numpy()
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test schema")
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError("Official test IDs are not 1..4940")
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing to overwrite output/manifest")
    paths = {"f10": Path(args.f10).resolve(), "v2": Path(args.v2).resolve(), "c214": Path(args.c214).resolve()}
    pred = {name: load(path, official_ids, name) for name, path in paths.items()}
    result = pred["f10"].copy()
    for target, sources in {
        "ei": ("f10", "v2"),
        "eps": ("f10", "c214"),
        "nc": ("f10", "v2"),
    }.items():
        ids = official_ids[target_type == target]
        result.loc[ids] = np.mean(np.vstack([pred[source].loc[ids].to_numpy(float) for source in sources]), axis=0)
    out = pd.DataFrame({"id": official_ids, "target": result.loc[official_ids].to_numpy(float)})
    if not np.isfinite(out["target"].to_numpy(float)).all():
        raise RuntimeError("Output contains non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.f12.archive-weak-equal-ensemble.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "local_eval_read_by_assembler": False,
        "weights": {"ei": {"f10": 0.5, "v2": 0.5}, "eps": {"f10": 0.5, "c214": 0.5}, "nc": {"f10": 0.5, "v2": 0.5}},
        "unchanged_targets": ["tg", "egc", "egb", "eea"],
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "weights": record["weights"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
