#!/usr/bin/env python3
"""Build fixed equal-blend Round 2 branch candidates from frozen CSV components.

The assembler itself is local_eval-free. It reads only:
- official Round 2 test.csv for ID order and target_type alignment;
- previously frozen complete prediction CSVs generated from official inputs.

The target maps are fixed in code from the prior post-freeze inventory:
- F18 without_archive: weak-target equal blends plus current target winners.
- F19 with_archive: weak-target equal blends plus archive target winners.

Any post-hoc local_eval scoring must be performed by the separate
score_round2_LOCAL_DIAGNOSTIC_ONLY.py scorer after this script has
written and hashed the candidate.
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


NOARCHIVE_SOURCES = {
    "c284": "experiments/final_submission_runs/without_archive/R2-C284-PI1M-SVD-without_archive-20260807.csv",
    "c285": "experiments/final_submission_runs/without_archive/R2-C285-PI1M-SVD-WEAK-RESIDUAL-without_archive-20260807.csv",
    "f02": "experiments/final_submission_runs/without_archive/R2-F02-COMPOUND-without_archive-candidate.csv",
    "f06": "experiments/final_submission_runs/without_archive/R2-F06-PI1M-without_archive-candidate.csv",
    "f14": "experiments/final_submission_runs/without_archive/R2-F14-FIXED-ENSEMBLE-without_archive-20260807.csv",
    "f15": "experiments/final_submission_runs/without_archive/R2-F15-WEAK-MEAN3-without_archive-20260807.csv",
    "c287_et": "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_extra_trees-without_archive-20260807.csv",
    "c287_huber": "experiments/final_submission_runs/without_archive/R2-C287v3-ei-dense_huber-without_archive-20260807.csv",
}


NOARCHIVE_TARGET_MAP = {
    "tg": ("c284",),
    "egc": ("c284",),
    "egb": ("f14",),
    "ei": ("f14", "c287_et", "f06"),
    "eea": ("c285",),
    "nc": ("c287_et", "c287_huber", "f15", "f02"),
    "eps": ("c287_et", "f02"),
}


ARCHIVE_SOURCES = {
    "v2": "final_submission/with_archive/R2-BEST-COMPOUND-with_archive-V2.csv",
    "f02b3": "experiments/final_submission_runs/with_archive/R2-F02-B3-COMPOUND-with_archive-20260807.csv",
    "f10": "experiments/final_submission_runs/with_archive/R2-F10-PORTFOLIO-with_archive-20260807.csv",
    "f12": "experiments/final_submission_runs/with_archive/R2-F12-WEAK-EQUAL-ENSEMBLE-with_archive-20260807.csv",
    "f13": "experiments/final_submission_runs/with_archive/R2-F13-WEAK-TRIPLE-ENSEMBLE-with_archive-20260807.csv",
}


ARCHIVE_TARGET_MAP = {
    "tg": ("v2",),
    "egc": ("v2",),
    "egb": ("v2",),
    "ei": ("f12", "f02b3", "f10", "v2"),
    "eea": ("v2",),
    "nc": ("f12",),
    "eps": ("f12", "f13", "v2"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction(path: Path, ids: np.ndarray, label: str) -> pd.Series:
    lowered = str(path).lower()
    if "local_eval" in lowered or "external_label" in lowered:
        raise RuntimeError(f"Refusing local_eval-like input for {label}: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"{label} has invalid schema or row count")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"{label} ID validation failed")
    values = frame["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains non-finite predictions")
    return pd.Series(values, index=ids)


def resolve_map(branch: str) -> tuple[dict[str, str], dict[str, tuple[str, ...]], str]:
    if branch == "without_archive":
        return NOARCHIVE_SOURCES, NOARCHIVE_TARGET_MAP, "ppp.round2.f18.without-archive-fixed-equal-blends.v1"
    if branch == "with_archive":
        return ARCHIVE_SOURCES, ARCHIVE_TARGET_MAP, "ppp.round2.f19.with-archive-fixed-equal-blends.v1"
    raise RuntimeError(f"Unsupported branch: {branch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    test_path = (base / args.test_csv).resolve()
    output = (base / args.output).resolve()
    manifest = (base / args.manifest).resolve()
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite of output or manifest")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output is outside the Round 2 boundary: {output}")

    test = pd.read_csv(test_path)
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected official test.csv schema or row count")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Official test IDs are not 1..4940")
    target_type = test["target_type"].astype(str).str.lower().to_numpy()
    if set(target_type) != set(TARGETS):
        raise RuntimeError("Unexpected target types in official test.csv")

    sources, target_map, schema = resolve_map(args.branch)
    resolved = {name: (base / rel).resolve() for name, rel in sources.items()}
    pred = {name: load_prediction(path, ids, name) for name, path in resolved.items()}

    result = pd.Series(np.nan, index=ids, dtype=float)
    for target, source_names in target_map.items():
        mask_ids = ids[target_type == target]
        matrix = np.vstack([pred[name].loc[mask_ids].to_numpy(float) for name in source_names])
        result.loc[mask_ids] = np.mean(matrix, axis=0)
    if not np.isfinite(result.loc[ids].to_numpy(float)).all():
        raise RuntimeError("Assembled prediction contains non-finite values")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result.loc[ids].to_numpy(float)})
    out.to_csv(output, index=False)

    record = {
        "schema_version": schema,
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "official_test_path": str(test_path),
        "official_test_sha256": sha256_file(test_path),
        "local_eval_read_by_assembler": False,
        "external_label_file_read_by_assembler": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "target_map": {target: list(names) for target, names in target_map.items()},
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in resolved.items()
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": int(len(out)),
            "bytes": output.stat().st_size,
        },
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"], "target_map": record["target_map"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
