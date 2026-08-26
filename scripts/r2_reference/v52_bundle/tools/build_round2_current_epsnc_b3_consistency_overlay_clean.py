#!/usr/bin/env python3
"""Clean branch-local current EPS/NC B3 consistency overlay.

This is the clean-output wrapper for the previously diagnostic B3 overlay.
It reads only:

- official current `ppp-round-2/train.csv`;
- official current `ppp-round-2/test.csv`;
- a frozen branch-local base prediction CSV.

It does not read local_eval, external_label, archive labels, opposite-branch predictions,
or Kaggle artifacts. For `with_archive`, the base CSV may itself be produced by
an archive-enabled pipeline; this overlay still only derives its EPS/NC
adjustments from official current train/test rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROUND2_DIR = Path(__file__).resolve().parents[1]
SOURCE_MODULE = ROUND2_DIR / "tools" / "LOCAL_DIAGNOSTIC_ONLY" / "build_current_epsnc_b3_consistency_overlay.py"
spec = importlib.util.spec_from_file_location("round2_b3_overlay_source", SOURCE_MODULE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load source module: {SOURCE_MODULE}")
b3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b3
spec.loader.exec_module(b3)


def forbid_path(path: Path, role: str, branch: str) -> None:
    low = str(path).lower()
    forbidden = ("local_eval", "external_label", "test_external_labels")
    if any(token in low for token in forbidden):
        raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if branch == "without_archive":
        if "/with_archive/" in low or "/archive/" in low:
            raise RuntimeError(f"Refusing archive/cross-branch {role} path: {path}")
        if "/without_archive/" not in low:
            raise RuntimeError(f"{role} path must be branch-local without_archive: {path}")
    elif branch == "with_archive":
        if "/without_archive/" in low:
            raise RuntimeError(f"Refusing opposite-branch {role} path: {path}")
        if "/with_archive/" not in low:
            raise RuntimeError(f"{role} path must be branch-local with_archive: {path}")
    else:
        raise RuntimeError(f"Invalid branch: {branch}")


def default_output(root: Path, cid: int, base_path: Path, branch: str) -> Path:
    prefix = "ARCHIVE" if branch == "with_archive" else "NOARCHIVE"
    return (
        root
        / "experiments"
        / "final_submission_runs"
        / branch
        / f"R2-C{cid}-{prefix}-CLEAN-CURRENT-EPSNC-B3-OVER-{base_path.stem}-20260808.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), default="without_archive")
    parser.add_argument("--base", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--eps-weight", type=float, default=0.10)
    parser.add_argument("--nc-weight", type=float, default=0.25)
    parser.add_argument("--consistency-pull", type=float, default=0.81)
    parser.add_argument("--output", default="")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_path = root / "ppp-round-2" / "train.csv"
    test_path = root / "ppp-round-2" / "test.csv"
    base_path = (root / args.base).resolve()
    output_path = Path(args.output).resolve() if args.output else default_output(root, args.cid, base_path, args.branch)
    manifest_path = Path(args.manifest).resolve() if args.manifest else output_path.with_suffix(".manifest.json")

    if "Polymer Prediction Challenge Round 2" not in str(output_path):
        raise RuntimeError(f"Output outside Round 2 boundary: {output_path}")
    forbid_path(base_path, "base", args.branch)
    forbid_path(output_path, "output", args.branch)
    forbid_path(manifest_path, "manifest", args.branch)
    for path, role in ((train_path, "train"), (test_path, "test")):
        b3.guard_read(path, role)
    if b3.sha256_file(train_path) != b3.EXPECTED_TRAIN_SHA:
        raise RuntimeError("train.csv hash mismatch")
    if b3.sha256_file(test_path) != b3.EXPECTED_TEST_SHA:
        raise RuntimeError("test.csv hash mismatch")
    for name, value in (
        ("eps_weight", args.eps_weight),
        ("nc_weight", args.nc_weight),
        ("consistency_pull", args.consistency_pull),
    ):
        if not (0.0 <= float(value) <= 1.0):
            raise RuntimeError(f"{name} outside [0, 1]: {value}")
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError(f"Refusing overwrite for C{args.cid}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["target_type"] = train["target_type"].astype(str).str.lower()
    test["target_type"] = test["target_type"].astype(str).str.lower()
    train["canonical"] = [b3.canonical(value) for value in train["smiles"]]
    test["canonical"] = [b3.canonical(value) for value in test["smiles"]]
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")

    base = b3.load_base(base_path, ids)
    base_values = base["target"].to_numpy(float)
    values = base_values.copy()

    official_wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    pair_frame = official_wide[["eps", "nc"]].dropna().copy()
    ionic = pair_frame["eps"].to_numpy(float) - pair_frame["nc"].to_numpy(float) ** 2
    if len(pair_frame) < 50 or np.any(ionic <= 0):
        raise RuntimeError("Insufficient or invalid current EPS/NC pairs")
    ionic_predict, ionic_report = b3.fit_ionic_predictor(pair_frame.index.astype(str).tolist(), ionic)

    test_pred = test[["id", "canonical", "target_type"]].copy()
    test_pred["base_target"] = base_values
    base_wide = test_pred.pivot_table(index="canonical", columns="target_type", values="base_target", aggfunc="mean")

    eps_train = train.loc[train["target_type"] == "eps", "target"].to_numpy(float)
    nc_train = train.loc[train["target_type"] == "nc", "target"].to_numpy(float)
    eps_low, eps_high = float(np.quantile(eps_train, 0.002)), float(np.quantile(eps_train, 0.998))
    nc_low, nc_high = float(np.quantile(nc_train, 0.002)), float(np.quantile(nc_train, 0.998))

    applied = {"eps": 0, "nc": 0, "b3_pairs": 0}
    support: dict[str, dict[str, int]] = {"eps": {}, "nc": {}}
    ionic_cache: dict[str, float] = {}
    row_by_canon_target: dict[tuple[str, str], int] = {}
    for row_index, row in test.iterrows():
        target = str(row["target_type"])
        if target in {"eps", "nc"}:
            row_by_canon_target[(str(row["canonical"]), target)] = int(row_index)

    for row_index, row in test.iterrows():
        target = str(row["target_type"])
        if target not in {"eps", "nc"}:
            continue
        canon = str(row["canonical"])
        if canon not in ionic_cache:
            ionic_cache[canon] = float(ionic_predict([canon])[0])
        ion = max(float(ionic_cache[canon]), b3.MIN_IONIC)
        old = float(values[row_index])
        if target == "eps":
            partner, src = b3.partner_value(canon, "nc", official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(partner**2 + ion, max(0.0, eps_low - 0.05), eps_high + 0.05))
            values[row_index] = (1.0 - args.eps_weight) * old + args.eps_weight * raw
        else:
            partner, src = b3.partner_value(canon, "eps", official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(np.sqrt(max(partner - ion, 1.0)), max(1.0, nc_low - 0.05), min(2.8, nc_high + 0.05)))
            values[row_index] = (1.0 - args.nc_weight) * old + args.nc_weight * raw
        applied[target] += 1
        support[target][src] = support[target].get(src, 0) + 1

    for canon in sorted({canon for canon, target in row_by_canon_target if target == "eps"}):
        eps_idx = row_by_canon_target.get((canon, "eps"))
        nc_idx = row_by_canon_target.get((canon, "nc"))
        if eps_idx is None or nc_idx is None:
            continue
        ion = max(float(ionic_cache.get(canon, ionic_predict([canon])[0])), b3.MIN_IONIC)
        eps_cons, nc_cons = b3.project_pair(
            float(values[eps_idx]),
            float(values[nc_idx]),
            ion,
            max(args.eps_weight, 1.0e-6),
            max(args.nc_weight, 1.0e-6),
        )
        values[eps_idx] = (1.0 - args.consistency_pull) * values[eps_idx] + args.consistency_pull * eps_cons
        values[nc_idx] = (1.0 - args.consistency_pull) * values[nc_idx] + args.consistency_pull * nc_cons
        applied["b3_pairs"] += 1

    if not np.isfinite(values).all():
        raise RuntimeError("Output contains non-finite predictions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "target": values}).to_csv(output_path, index=False)
    manifest = {
        "schema_version": "ppp.round2.clean-current-epsnc-b3-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "CLEAN_OFFICIAL_ONLY",
        "branch": args.branch,
        "local_eval_read_by_builder": False,
        "archive_labels_used_by_overlay": False,
        "base_may_use_archive_labels": args.branch == "with_archive",
        "with_archive_inputs_used": args.branch == "with_archive",
        "opposite_branch_inputs_used": False,
        "base": {"path": str(base_path.relative_to(root)), "sha256": b3.sha256_file(base_path)},
        "train": {"path": str(train_path.relative_to(root)), "sha256": b3.sha256_file(train_path)},
        "test": {"path": str(test_path.relative_to(root)), "sha256": b3.sha256_file(test_path)},
        "weights": {"eps": args.eps_weight, "nc": args.nc_weight, "consistency_pull": args.consistency_pull},
        "ionic_predictor": ionic_report,
        "pair_rows": int(len(pair_frame)),
        "applied_rows": applied,
        "support": support,
        "output": {
            "path": str(output_path.relative_to(root)),
            "sha256": b3.sha256_file(output_path),
            "rows": int(len(values)),
            "bytes": output_path.stat().st_size,
        },
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": manifest["output"], "applied_rows": applied}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
