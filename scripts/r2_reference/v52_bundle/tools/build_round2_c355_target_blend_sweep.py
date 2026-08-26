#!/usr/bin/env python3
"""C355 branch-local target blend sweep builder.

This builder is deliberately local_eval-free.  It starts from one branch incumbent
CSV and writes a fixed grid of complete candidate CSVs where exactly one target
is replaced by a convex blend of the incumbent target values and one other
branch-local clean candidate's values.

The companion scoring step must run separately after these CSVs are frozen.
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
BAD_NAME_TOKENS = (
    "local_eval-portfolio",
    "local_eval_target",
    "local_eval-target",
    "target-winners",
    "f09-local_eval",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, branch: str, *, role: str, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    branch_token = f"/{branch}/"
    opposite = "without_archive" if branch == "with_archive" else "with_archive"
    if f"/{opposite}/" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for {branch}: {path}")
    if branch_token not in low and not low.endswith(f"/{branch}"):
        raise RuntimeError(f"{role} path must be inside /{branch}/: {path}")
    if allow_output and "Polymer Prediction Challenge Round 2" not in str(path):
        raise RuntimeError(f"Output outside Round 2 boundary: {path}")


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    invalid = [target for target in targets if target not in TARGETS]
    if invalid:
        raise RuntimeError(f"Invalid targets: {invalid}")
    return targets


def parse_weights(value: str) -> tuple[float, ...]:
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not weights:
        raise RuntimeError("No weights requested")
    bad = [weight for weight in weights if not (0.0 <= weight <= 1.0)]
    if bad:
        raise RuntimeError(f"Weights must be in [0, 1]: {bad}")
    return weights


def load_candidate(path: Path, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"]:
        raise RuntimeError(f"Invalid candidate schema: {path}")
    if len(frame) != len(ids):
        raise RuntimeError(f"Invalid row count for {path}: {len(frame)}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Candidate IDs do not match official test order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Candidate contains non-finite values: {path}")
    return frame


def safe_slug(path: Path) -> str:
    slug = path.stem
    for prefix in ("R2-", "Sandman_"):
        if slug.startswith(prefix):
            slug = slug[len(prefix):]
    keep = []
    for char in slug:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        else:
            keep.append("-")
    return "".join(keep)[:96]


def discover_sources(branch_dir: Path, branch: str, base_path: Path) -> list[Path]:
    sources: list[Path] = []
    for path in sorted(branch_dir.glob("*.csv")):
        low = path.name.lower()
        if path.resolve() == base_path.resolve():
            continue
        if any(token in low for token in BAD_NAME_TOKENS):
            continue
        if path.name.startswith("."):
            continue
        guard_path(path.resolve(), branch, role="source candidate")
        sources.append(path.resolve())
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument(
        "--source-csv",
        action="append",
        default=[],
        help="Explicit branch-local source CSV; repeatable. When set, branch-dir discovery is skipped.",
    )
    parser.add_argument("--targets", default="tg,egc,egb,ei,eea,nc,eps")
    parser.add_argument("--weights", default="0.125,0.25,0.375,0.5,0.625,0.75,0.875")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    branch = args.branch
    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    branch_dir = Path(args.branch_dir).resolve() if args.branch_dir else Path("experiments/final_submission_runs", branch).resolve()
    targets = parse_targets(args.targets)
    weights = parse_weights(args.weights)

    for path, role in ((base_path, "base candidate"), (branch_dir, "branch dir"), (output_dir, "output dir")):
        guard_path(path, branch, role=role, allow_output=role == "output dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(test_path)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    target_type = test["target_type"].astype(str).str.lower().to_numpy(object)
    base = load_candidate(base_path, ids)
    base_values = base["target"].to_numpy(float)

    if args.source_csv:
        sources = []
        seen_sources: set[Path] = set()
        for raw_source in args.source_csv:
            source_path = Path(raw_source).resolve()
            guard_path(source_path, branch, role="source candidate")
            if source_path == base_path:
                continue
            if source_path in seen_sources:
                continue
            seen_sources.add(source_path)
            sources.append(source_path)
    else:
        sources = discover_sources(branch_dir, branch, base_path)
    if not sources:
        raise RuntimeError(f"No source candidates found in {branch_dir}")

    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    sequence = 0
    for source_path in sources:
        source = load_candidate(source_path, ids)
        source_values = source["target"].to_numpy(float)
        source_delta_max = float(np.max(np.abs(source_values - base_values)))
        if source_delta_max <= 1.0e-12:
            continue
        for target in targets:
            mask = target_type == target
            if not np.any(mask):
                raise RuntimeError(f"No test rows for target {target}")
            target_delta_max = float(np.max(np.abs(source_values[mask] - base_values[mask])))
            if target_delta_max <= 1.0e-12:
                continue
            for weight in weights:
                sequence += 1
                result = base_values.copy()
                result[mask] = (1.0 - weight) * base_values[mask] + weight * source_values[mask]
                if not np.isfinite(result).all():
                    raise RuntimeError(f"Non-finite result for {source_path}/{target}/{weight}")
                out_path = output_dir / f"R2-C355-{branch}-blend-{target}-w{weight:.3f}-{safe_slug(source_path)}.csv"
                if out_path.exists():
                    raise RuntimeError(f"Refusing overwrite: {out_path}")
                pd.DataFrame({"id": ids, "target": result}).to_csv(out_path, index=False)
                record = {
                    "schema_version": "ppp.round2.c355.target-blend-sweep.v1",
                    "sequence": sequence,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "branch": branch,
                    "target": target,
                    "weight_on_source": float(weight),
                    "source": {
                        "path": str(source_path),
                        "sha256": sha256_file(source_path),
                        "bytes": source_path.stat().st_size,
                    },
                    "base": {
                        "path": str(base_path),
                        "sha256": sha256_file(base_path),
                        "bytes": base_path.stat().st_size,
                    },
                    "changed_rows": int(np.sum(mask)),
                    "max_abs_target_delta_before_weight": target_delta_max,
                    "local_eval_read_by_builder": False,
                    "kaggle_compute": False,
                    "kaggle_upload": False,
                    "kaggle_submission": False,
                    "output": {
                        "path": str(out_path),
                        "sha256": sha256_file(out_path),
                        "rows": int(len(result)),
                        "bytes": out_path.stat().st_size,
                    },
                }
                records.append(record)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    summary = {
        "schema_version": "ppp.round2.c355.target-blend-sweep-summary.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": branch,
        "targets": list(targets),
        "weights": list(weights),
        "source_count": len(sources),
        "candidate_count": len(records),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
