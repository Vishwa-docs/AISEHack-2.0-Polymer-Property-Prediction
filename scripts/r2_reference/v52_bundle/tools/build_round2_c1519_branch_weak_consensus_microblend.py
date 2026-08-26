#!/usr/bin/env python3
"""Branch-local weak-target consensus microblend.

This builder is local_eval-free.  It reads:

* official current test.csv, only for id order and target_type masks;
* one branch-local base candidate CSV;
* two or more branch-local source candidate CSVs.

For each requested weak target, it computes source deltas from the base and
applies a small fixed shrink toward the trimmed median delta only on rows where
at least `min_agree` source families agree on sign.  It does not train a model,
read local_eval/external_label files, choose rows from local_eval scores, or write a submission.
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
DEFAULT_SHRINK = {"ei": 0.06, "eps": 0.08, "nc": 0.06}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, branch: str, role: str) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    opposite = "without_archive" if branch == "with_archive" else "with_archive"
    if role != "test" and f"/{opposite}/" in low:
        raise RuntimeError(f"Refusing opposite-branch {role} path: {path}")
    if role != "test" and f"/{branch}/" not in low:
        raise RuntimeError(f"{role} path must be branch-local for {branch}: {path}")
    if role in {"output", "manifest"} and "Polymer Prediction Challenge Round 2" not in str(path):
        raise RuntimeError(f"{role} outside Round 2 boundary: {path}")


def load_candidate(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid candidate schema/count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Candidate ID order mismatch: {path}")
    values = frame["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Candidate contains non-finite values: {path}")
    return values


def parse_targets(raw: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets supplied")
    invalid = [target for target in targets if target not in TARGETS]
    if invalid:
        raise RuntimeError(f"Invalid targets: {invalid}")
    return targets


def parse_shrink(raw: str | None, targets: tuple[str, ...]) -> dict[str, float]:
    shrink = {target: DEFAULT_SHRINK.get(target, 0.0) for target in targets}
    if raw:
        for item in raw.split(","):
            if not item.strip():
                continue
            target, value = item.split("=", 1)
            target = target.strip().lower()
            if target not in targets:
                raise RuntimeError(f"Shrink supplied for inactive target: {target}")
            shrink[target] = float(value)
    for target, value in shrink.items():
        if not (0.0 <= value <= 0.5):
            raise RuntimeError(f"Invalid shrink for {target}: {value}")
    return shrink


def clipped_median_delta(deltas: np.ndarray, max_abs_delta: float) -> np.ndarray:
    """Median over source axis after removing one extreme when possible."""
    if deltas.shape[0] >= 5:
        ordered = np.sort(deltas, axis=0)
        med = np.median(ordered[1:-1], axis=0)
    else:
        med = np.median(deltas, axis=0)
    return np.clip(med, -max_abs_delta, max_abs_delta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--source-csv", action="append", required=True)
    parser.add_argument("--targets", default="ei,eps,nc")
    parser.add_argument("--min-agree", type=int, default=3)
    parser.add_argument("--min-abs-delta", type=float, default=1.0e-9)
    parser.add_argument("--max-abs-delta", type=float, default=0.25)
    parser.add_argument("--shrink", default=None, help="comma-separated target=value overrides")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    branch = args.branch
    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    targets = parse_targets(args.targets)
    shrink = parse_shrink(args.shrink, targets)
    if args.min_agree < 2:
        raise RuntimeError("min_agree must be at least 2")
    if args.min_abs_delta < 0.0 or args.max_abs_delta <= 0.0:
        raise RuntimeError("Invalid delta thresholds")

    for path, role in (
        (test_path, "test"),
        (base_path, "base"),
        (output_path, "output"),
        (manifest_path, "manifest"),
    ):
        guard_path(path, branch, role)
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError("Refusing overwrite")

    source_paths: list[Path] = []
    seen: set[Path] = set()
    for raw in args.source_csv:
        path = Path(raw).resolve()
        guard_path(path, branch, "source")
        if path == base_path or path in seen:
            continue
        seen.add(path)
        source_paths.append(path)
    if len(source_paths) < args.min_agree:
        raise RuntimeError("Not enough unique sources for min_agree")

    test = pd.read_csv(test_path)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    target_type = test["target_type"].astype(str).str.lower().to_numpy(object)
    base_values = load_candidate(base_path, ids)
    source_values = [load_candidate(path, ids) for path in source_paths]
    source_array = np.stack(source_values, axis=0)
    deltas = source_array - base_values.reshape(1, -1)

    result = base_values.copy()
    target_reports: dict[str, Any] = {}
    for target in targets:
        mask = target_type == target
        target_deltas = deltas[:, mask]
        active = np.abs(target_deltas) > float(args.min_abs_delta)
        pos_agree = np.sum(target_deltas > float(args.min_abs_delta), axis=0)
        neg_agree = np.sum(target_deltas < -float(args.min_abs_delta), axis=0)
        row_agrees = np.maximum(pos_agree, neg_agree)
        chosen = row_agrees >= int(args.min_agree)
        direction = np.where(pos_agree >= neg_agree, 1.0, -1.0)
        same_sign = np.sign(target_deltas) == direction.reshape(1, -1)
        usable = active & same_sign
        consensus_delta = np.zeros(int(mask.sum()), dtype=np.float64)
        for col in np.where(chosen)[0]:
            values = target_deltas[usable[:, col], col]
            if len(values) < args.min_agree:
                continue
            consensus_delta[col] = clipped_median_delta(values.reshape(-1, 1), float(args.max_abs_delta))[0]
        update = float(shrink[target]) * consensus_delta
        target_positions = np.flatnonzero(mask)
        changed_positions = target_positions[np.abs(update) > 0.0]
        result[target_positions] = result[target_positions] + update
        target_reports[target] = {
            "rows": int(mask.sum()),
            "changed_rows": int(len(changed_positions)),
            "positive_consensus_rows": int(np.sum((update > 0.0))),
            "negative_consensus_rows": int(np.sum((update < 0.0))),
            "min_agree": int(args.min_agree),
            "shrink": float(shrink[target]),
            "max_abs_update": float(np.max(np.abs(update))) if len(update) else 0.0,
            "mean_abs_update_on_changed": float(np.mean(np.abs(update[np.abs(update) > 0.0]))) if np.any(np.abs(update) > 0.0) else 0.0,
        }

    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output_path, index=False)
    record = {
        "schema_version": "ppp.round2.c1519.branch-weak-consensus-microblend.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": branch,
        "classification": "CLEAN_OFFICIAL_ONLY_BRANCH_LOCAL_SOURCE_CONSENSUS",
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": {
            "targets": list(targets),
            "min_agree": int(args.min_agree),
            "min_abs_delta": float(args.min_abs_delta),
            "max_abs_delta": float(args.max_abs_delta),
            "shrink": shrink,
        },
        "test": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "sources": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "target_reports": target_reports,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": int(len(out)),
            "bytes": output_path.stat().st_size,
        },
    }
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "target_reports": target_reports}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
