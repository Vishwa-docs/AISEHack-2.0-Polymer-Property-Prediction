#!/usr/bin/env python3
"""Build the no-archive F10 portfolio from frozen clean component candidates.

This assembler is local_eval-free: it reads official `test.csv` for ID/target_type
alignment and reads only complete component prediction CSVs. It does not read
local_eval files, test external_labels, or archive labels. It is a local reconstruction
helper, not the final single-notebook generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
DEFAULT_SELECTION = {
    "tg": "c282",
    "egc": "c282",
    "egb": "f01",
    "ei": "f06",
    "eea": "c282",
    "nc": "f02",
    "eps": "f02",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reject_local_eval_path(path: Path) -> None:
    lowered = str(path).lower()
    if "local_eval" in lowered or "external_label" in lowered or "test_external_label" in lowered:
        raise RuntimeError(f"Refusing local_eval/external_label-like component path: {path}")


def load_component(path: Path, official_ids: np.ndarray, name: str) -> pd.Series:
    reject_local_eval_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "target"]:
        raise RuntimeError(f"{name} has unexpected columns: {list(df.columns)}")
    if len(df) != len(official_ids):
        raise RuntimeError(f"{name} row count {len(df)} != official test rows {len(official_ids)}")
    ids = df["id"].to_numpy(int)
    if not np.array_equal(ids, official_ids):
        raise RuntimeError(f"{name} ID/order mismatch versus official test.csv")
    values = df["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite predictions")
    if df["id"].duplicated().any():
        raise RuntimeError(f"{name} contains duplicate IDs")
    return pd.Series(values, index=ids, name=name)


def build_portfolio(
    *,
    test_csv: Path,
    components: Mapping[str, Path],
    output: Path,
    manifest: Path,
) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {output}")
    if manifest.exists():
        raise RuntimeError(f"Refusing to overwrite manifest: {manifest}")
    test = pd.read_csv(test_csv)
    if list(test.columns) != ["id", "smiles", "target_type"]:
        raise RuntimeError(f"Unexpected official test schema: {list(test.columns)}")
    if len(test) != 4940:
        raise RuntimeError(f"Unexpected official test row count: {len(test)}")
    official_ids = test["id"].to_numpy(int)
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError("Official test IDs are not sequential 1..4940")
    target_type = test["target_type"].astype(str).str.lower().to_numpy()
    if sorted(set(target_type)) != sorted(TARGETS):
        raise RuntimeError(f"Unexpected target set: {sorted(set(target_type))}")

    loaded = {
        name: load_component(path.resolve(), official_ids, name)
        for name, path in components.items()
    }
    result = pd.Series(index=official_ids, dtype=float)
    counts: dict[str, int] = {}
    for target, component_name in DEFAULT_SELECTION.items():
        mask = target_type == target
        result.loc[official_ids[mask]] = loaded[component_name].loc[official_ids[mask]].to_numpy(float)
        counts[target] = int(mask.sum())
    if result.isna().any() or not np.isfinite(result.to_numpy(float)).all():
        raise RuntimeError("Assembled portfolio has missing or non-finite predictions")
    out = pd.DataFrame({"id": official_ids, "target": result.loc[official_ids].to_numpy(float)})
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    record: dict[str, object] = {
        "schema_version": "ppp.round2.f10.without-archive.portfolio.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "official_test_csv": {
            "path": str(test_csv),
            "sha256": sha256_file(test_csv),
            "rows": int(len(test)),
        },
        "local_eval_read": False,
        "archive_labels_used": False,
        "component_inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in components.items()
        },
        "selection": DEFAULT_SELECTION,
        "target_counts": counts,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": int(len(out)),
        },
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--c282", required=True)
    parser.add_argument("--f01", required=True)
    parser.add_argument("--f06", required=True)
    parser.add_argument("--f02", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    record = build_portfolio(
        test_csv=Path(args.test_csv).resolve(),
        components={
            "c282": Path(args.c282).resolve(),
            "f01": Path(args.f01).resolve(),
            "f06": Path(args.f06).resolve(),
            "f02": Path(args.f02).resolve(),
        },
        output=Path(args.output).resolve(),
        manifest=Path(args.manifest).resolve(),
    )
    print(
        json.dumps(
            {
                "output": record["output"],
                "selection": record["selection"],
                "target_counts": record["target_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
