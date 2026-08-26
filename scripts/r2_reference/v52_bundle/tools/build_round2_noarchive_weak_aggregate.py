#!/usr/bin/env python3
"""Build fixed no-archive weak-target aggregate candidates.

Supported variants are intentionally limited and predeclared:
- mean3: targetwise mean of F11, C284, C285 for ei/nc/eps;
- median3: targetwise median of F11, C284, C285 for ei/nc/eps.

For both variants, Tg/Egc remain C284, Egb remains F11, and Eea remains the
F14-style mean of C282 and C285. The assembler reads no local_eval/archive files.
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
    lowered = str(path).lower()
    if "local_eval" in lowered or "external_label" in lowered:
        raise RuntimeError(f"Refusing local_eval-like input {name}: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(official_ids):
        raise RuntimeError(f"{name} schema/row count invalid")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), official_ids):
        raise RuntimeError(f"{name} ID validation failed")
    values = frame["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} non-finite predictions")
    return pd.Series(values, index=official_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mean3", "median3"], required=True)
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--f11", required=True)
    parser.add_argument("--c282", required=True)
    parser.add_argument("--c284", required=True)
    parser.add_argument("--c285", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    test = pd.read_csv(args.test_csv)
    official_ids = test["id"].to_numpy(int)
    target_type = test["target_type"].astype(str).str.lower().to_numpy()
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test.csv")
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError("Official test IDs are not 1..4940")
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    paths = {
        "f11": Path(args.f11).resolve(),
        "c282": Path(args.c282).resolve(),
        "c284": Path(args.c284).resolve(),
        "c285": Path(args.c285).resolve(),
    }
    pred = {name: load(path, official_ids, name) for name, path in paths.items()}
    result = pred["f11"].copy()
    for target, source in {"tg": "c284", "egc": "c284", "egb": "f11"}.items():
        ids = official_ids[target_type == target]
        result.loc[ids] = pred[source].loc[ids].to_numpy(float)
    ids = official_ids[target_type == "eea"]
    result.loc[ids] = np.mean(np.vstack([pred["c282"].loc[ids].to_numpy(float), pred["c285"].loc[ids].to_numpy(float)]), axis=0)
    for target in ("ei", "nc", "eps"):
        ids = official_ids[target_type == target]
        stacked = np.vstack([pred["f11"].loc[ids].to_numpy(float), pred["c284"].loc[ids].to_numpy(float), pred["c285"].loc[ids].to_numpy(float)])
        if args.variant == "mean3":
            values = np.mean(stacked, axis=0)
        else:
            values = np.median(stacked, axis=0)
        result.loc[ids] = values
    out = pd.DataFrame({"id": official_ids, "target": result.loc[official_ids].to_numpy(float)})
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.noarchive-weak-aggregate.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "variant": args.variant,
        "local_eval_read_by_assembler": False,
        "archive_labels_used": False,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"variant": args.variant, "output": record["output"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
