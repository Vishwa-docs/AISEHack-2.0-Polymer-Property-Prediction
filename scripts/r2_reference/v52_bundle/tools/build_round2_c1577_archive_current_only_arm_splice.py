#!/usr/bin/env python3
"""C1577 explicit archive compound with current-only target arms.

This local arithmetic assembler starts from an archive-branch base candidate
and replaces declared target columns with a current-only/noarchive candidate.
It is deliberately explicit about the cross-branch source because an archive
pipeline may contain a submodel that ignores archive labels, but this must not
be silent branch mixing.

The script reads no local_eval/external_label/nonofficial files and performs no selection.
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


def guard(path: Path, *, role: str) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")


def load_candidate(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard(path, role="candidate")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid candidate schema/count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid candidate ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite candidate values: {path}")
    return frame


def parse_targets(raw: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets supplied")
    invalid = [target for target in targets if target not in TARGETS]
    if invalid:
        raise RuntimeError(f"Invalid targets: {invalid}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--archive-base", required=True)
    parser.add_argument("--current-only-source", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.archive_base).resolve()
    source_path = Path(args.current_only_source).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    targets = parse_targets(args.targets)
    for path, role in ((test_path, "test"), (base_path, "archive base"), (source_path, "current-only source"), (output, "output"), (manifest, "manifest")):
        guard(path, role=role)
    if "/with_archive/" not in str(base_path).lower():
        raise RuntimeError(f"Archive base must be in with_archive namespace: {base_path}")
    if "/without_archive/" not in str(source_path).lower():
        raise RuntimeError(f"Current-only source must be in without_archive namespace: {source_path}")
    if "/with_archive/" not in str(output).lower() or "/with_archive/" not in str(manifest).lower():
        raise RuntimeError("Output and manifest must stay in with_archive namespace")
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "polymer prediction challenge round 2" not in str(output).lower():
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    test = pd.read_csv(test_path)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    target_type = test["target_type"].astype(str).str.lower().to_numpy(object)
    base = load_candidate(base_path, ids)
    source = load_candidate(source_path, ids)
    values = base["target"].to_numpy(float).copy()
    source_values = source["target"].to_numpy(float)
    source_records: dict[str, Any] = {}
    for target in targets:
        mask = target_type == target
        values[mask] = source_values[mask]
        source_records[target] = {"changed_rows": int(np.sum(mask))}
    pd.DataFrame({"id": ids, "target": values}).to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.c1577.archive-current-only-arm-splice.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "ARCHIVE_COMPOUND_WITH_EXPLICIT_CURRENT_ONLY_TARGET_ARMS_LOCAL",
        "branch": "with_archive",
        "note": "Explicit local compound: archive branch base plus current-only target arms. Not a standalone notebook artifact.",
        "targets": source_records,
        "local_eval_read_by_builder": False,
        "external_label_file_read_by_builder": False,
        "nonofficial_file_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "archive_base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "current_only_source": {"path": str(source_path), "sha256": sha256_file(source_path), "bytes": source_path.stat().st_size},
        "official_test": {"path": str(test_path), "sha256": sha256_file(test_path)},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(values)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "targets": source_records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
