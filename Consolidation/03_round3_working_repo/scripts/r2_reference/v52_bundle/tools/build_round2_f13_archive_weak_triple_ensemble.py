#!/usr/bin/env python3
"""Build a fixed archive weak-target triple ensemble.

This is one no-sweep follow-up to F12. It preserves archive F10 for strong
targets and applies equal means on weak targets using three frozen,
method-diverse official-only candidate sources:
- ei: F10, V2, C199;
- eps: F10, C214, C187;
- nc: F10, V2, F01.

The assembler reads no local_eval files and writes a local research candidate.
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
        raise RuntimeError(f"{name} contains non-finite predictions")
    return pd.Series(values, index=official_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--f10", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--c199", required=True)
    parser.add_argument("--c214", required=True)
    parser.add_argument("--c187", required=True)
    parser.add_argument("--f01", required=True)
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
        raise RuntimeError("Refusing overwrite")
    paths = {
        "f10": Path(args.f10).resolve(),
        "v2": Path(args.v2).resolve(),
        "c199": Path(args.c199).resolve(),
        "c214": Path(args.c214).resolve(),
        "c187": Path(args.c187).resolve(),
        "f01": Path(args.f01).resolve(),
    }
    pred = {name: load(path, official_ids, name) for name, path in paths.items()}
    result = pred["f10"].copy()
    weights = {"ei": ("f10", "v2", "c199"), "eps": ("f10", "c214", "c187"), "nc": ("f10", "v2", "f01")}
    for target, sources in weights.items():
        ids = official_ids[target_type == target]
        result.loc[ids] = np.mean(np.vstack([pred[source].loc[ids].to_numpy(float) for source in sources]), axis=0)
    out = pd.DataFrame({"id": official_ids, "target": result.loc[official_ids].to_numpy(float)})
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.f13.archive-weak-triple-ensemble.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "local_eval_read_by_assembler": False,
        "weights": {target: {source: 1.0 / len(sources) for source in sources} for target, sources in weights.items()},
        "unchanged_targets": ["tg", "egc", "egb", "eea"],
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "weights": record["weights"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
