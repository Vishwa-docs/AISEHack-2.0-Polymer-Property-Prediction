#!/usr/bin/env python3
"""Build no-archive F11 portfolio from frozen component candidates.

Selection is an local_eval-observed aggregate choice made after all component CSVs
were frozen and hashed. This assembler itself is local_eval-free: it reads official
test.csv only for ID/target alignment plus complete component prediction CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
SELECTION = {
    "tg": "c284",
    "egc": "c284",
    "egb": "f01",
    "ei": "f06",
    "eea": "c282",
    "nc": "f02",
    "eps": "f02",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_component(name: str, path: Path, official_ids: np.ndarray) -> pd.Series:
    lowered = str(path).lower()
    if "local_eval" in lowered or "external_label" in lowered:
        raise RuntimeError(f"Refusing local_eval-like component path for {name}: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "target"]:
        raise RuntimeError(f"{name} has unexpected columns: {list(df.columns)}")
    if len(df) != len(official_ids) or df["id"].duplicated().any():
        raise RuntimeError(f"{name} row count/duplicate check failed")
    if not np.array_equal(df["id"].to_numpy(int), official_ids):
        raise RuntimeError(f"{name} ID order mismatch")
    values = df["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite predictions")
    return pd.Series(values, index=official_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--c282", required=True)
    parser.add_argument("--c284", required=True)
    parser.add_argument("--f01", required=True)
    parser.add_argument("--f06", required=True)
    parser.add_argument("--f02", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    test_csv = Path(args.test_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {output}")
    if manifest.exists():
        raise RuntimeError(f"Refusing to overwrite manifest: {manifest}")
    test = pd.read_csv(test_csv)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test schema/row count")
    official_ids = test["id"].to_numpy(int)
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError("Official test IDs are not 1..4940")
    target_type = test["target_type"].astype(str).str.lower().to_numpy()
    components = {
        "c282": Path(args.c282).resolve(),
        "c284": Path(args.c284).resolve(),
        "f01": Path(args.f01).resolve(),
        "f06": Path(args.f06).resolve(),
        "f02": Path(args.f02).resolve(),
    }
    loaded = {name: load_component(name, path, official_ids) for name, path in components.items()}
    result = pd.Series(index=official_ids, dtype=float)
    target_counts = {}
    for target, source in SELECTION.items():
        mask = target_type == target
        target_counts[target] = int(mask.sum())
        result.loc[official_ids[mask]] = loaded[source].loc[official_ids[mask]].to_numpy(float)
    if result.isna().any() or not np.isfinite(result.to_numpy(float)).all():
        raise RuntimeError("Assembled F11 has missing/non-finite rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": official_ids, "target": result.loc[official_ids].to_numpy(float)})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.f11.without-archive.portfolio.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "classification": "LOCAL_EVAL_OBSERVED_SELECTION_AFTER_FREEZE",
        "local_eval_read_by_assembler": False,
        "archive_labels_used": False,
        "official_test_csv": {"path": str(test_csv), "sha256": sha256_file(test_csv), "rows": int(len(test))},
        "selection": SELECTION,
        "target_counts": target_counts,
        "component_inputs": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in components.items()
        },
        "output": {"path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size, "rows": int(len(out))},
        "warning": "Assembler is local_eval-free, but the target source map is an local_eval-observed post-freeze research selection and is not clean validation evidence.",
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "selection": SELECTION, "target_counts": target_counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
