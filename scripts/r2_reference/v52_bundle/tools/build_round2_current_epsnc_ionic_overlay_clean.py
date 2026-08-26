#!/usr/bin/env python3
"""Clean no-archive current EPS/NC ionic overlay.

This wraps the current-label ionic consistency method with clean output paths.
It reads only official current train/test files plus a frozen no-archive base
prediction CSV. LocalEval/external_label/archive/with-archive paths are refused.
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
SOURCE_MODULE = ROUND2_DIR / "tools" / "LOCAL_DIAGNOSTIC_ONLY" / "build_current_epsnc_ionic_overlay.py"
spec = importlib.util.spec_from_file_location("round2_epsnc_ionic_source", SOURCE_MODULE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load source module: {SOURCE_MODULE}")
ionic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ionic
spec.loader.exec_module(ionic)


def forbid_path(path: Path, role: str, *, allow_base: bool = False) -> None:
    low = str(path).lower()
    if any(token in low for token in ("local_eval", "external_label", "test_external_labels")):
        raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "/with_archive/" in low or "/archive/" in low:
        raise RuntimeError(f"Refusing archive/cross-branch {role} path: {path}")
    if role in {"output", "manifest"} and "/final_submission_runs/without_archive/" not in low:
        raise RuntimeError(f"{role} must be in final_submission_runs/without_archive: {path}")
    if not allow_base and role == "base" and "/final_submission_runs/without_archive/" not in low:
        raise RuntimeError(f"base must be branch-local without_archive: {path}")


def default_output(root: Path, cid: int, base_path: Path) -> Path:
    return (
        root
        / "experiments"
        / "final_submission_runs"
        / "without_archive"
        / f"R2-C{cid}-NOARCHIVE-CLEAN-CURRENT-EPSNC-IONIC-OVER-{base_path.stem}-20260808.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--base", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--ionic-mode", choices=("median", "extra_trees_raw", "extra_trees_log"), default="extra_trees_raw")
    parser.add_argument("--eps-weight", type=float, default=0.10)
    parser.add_argument("--nc-weight", type=float, default=0.10)
    parser.add_argument("--output", default="")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_path = root / "ppp-round-2" / "train.csv"
    test_path = root / "ppp-round-2" / "test.csv"
    base_path = (root / args.base).resolve()
    output_path = Path(args.output).resolve() if args.output else default_output(root, args.cid, base_path)
    manifest_path = Path(args.manifest).resolve() if args.manifest else output_path.with_suffix(".manifest.json")

    if "Polymer Prediction Challenge Round 2" not in str(root):
        raise RuntimeError(f"Root outside Round 2 boundary: {root}")
    if "Polymer Prediction Challenge Round 2" not in str(output_path):
        raise RuntimeError(f"Output outside Round 2 boundary: {output_path}")
    forbid_path(base_path, "base", allow_base=True)
    forbid_path(output_path, "output")
    forbid_path(manifest_path, "manifest")
    for path, role in ((train_path, "train"), (test_path, "test")):
        ionic.guard_read(path, role)
    if ionic.sha256_file(train_path) != ionic.EXPECTED_TRAIN_SHA:
        raise RuntimeError("train.csv hash mismatch")
    if ionic.sha256_file(test_path) != ionic.EXPECTED_TEST_SHA:
        raise RuntimeError("test.csv hash mismatch")
    for name, value in (("eps_weight", args.eps_weight), ("nc_weight", args.nc_weight)):
        if not (0.0 <= float(value) <= 1.0):
            raise RuntimeError(f"{name} outside [0, 1]: {value}")
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError(f"Refusing overwrite for C{args.cid}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["target_type"] = train["target_type"].astype(str).str.lower()
    test["target_type"] = test["target_type"].astype(str).str.lower()
    train["canonical"] = [ionic.canonical(value) for value in train["smiles"]]
    test["canonical"] = [ionic.canonical(value) for value in test["smiles"]]
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = ionic.load_base(base_path, ids)

    official_wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    pair_frame = official_wide[["eps", "nc"]].dropna().copy()
    if len(pair_frame) < 50:
        raise RuntimeError("Insufficient current EPS/NC pairs")
    ionic_coord = pair_frame["eps"].to_numpy(float) - pair_frame["nc"].to_numpy(float) ** 2
    if np.any(ionic_coord <= 0):
        raise RuntimeError("Non-positive ionic coordinate in official current pairs")
    ionic_predict, ionic_report = ionic.fit_ionic_predictor(
        pair_frame.index.astype(str).tolist(),
        ionic_coord,
        str(args.ionic_mode),
    )

    test_pred = test[["id", "canonical", "target_type"]].copy()
    test_pred["base_target"] = base["target"].to_numpy(float)
    base_wide = test_pred.pivot_table(index="canonical", columns="target_type", values="base_target", aggfunc="mean")

    values = base["target"].to_numpy(float).copy()
    eps_train = train.loc[train["target_type"] == "eps", "target"].to_numpy(float)
    nc_train = train.loc[train["target_type"] == "nc", "target"].to_numpy(float)
    eps_low, eps_high = float(np.quantile(eps_train, 0.002)), float(np.quantile(eps_train, 0.998))
    nc_low, nc_high = float(np.quantile(nc_train, 0.002)), float(np.quantile(nc_train, 0.998))

    applied: dict[str, int] = {target: 0 for target in ionic.TARGETS}
    support: dict[str, dict[str, int]] = {target: {} for target in ("eps", "nc")}
    examples: list[dict[str, object]] = []
    ionic_cache: dict[str, float] = {}
    for row_index, row in test.iterrows():
        target = str(row["target_type"])
        if target not in {"eps", "nc"}:
            continue
        canon = str(row["canonical"])
        if canon not in ionic_cache:
            ionic_cache[canon] = float(ionic_predict([canon])[0])
        ion = max(float(ionic_cache[canon]), ionic.MIN_IONIC)
        old = float(values[row_index])
        if target == "eps":
            partner, src = ionic.partner_value(canon, "nc", official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(partner**2 + ion, max(0.0, eps_low - 0.05), eps_high + 0.05))
            values[row_index] = (1.0 - args.eps_weight) * old + args.eps_weight * raw
        else:
            partner, src = ionic.partner_value(canon, "eps", official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(np.sqrt(max(partner - ion, 1.0)), max(1.0, nc_low - 0.05), min(2.8, nc_high + 0.05)))
            values[row_index] = (1.0 - args.nc_weight) * old + args.nc_weight * raw
        applied[target] += 1
        support[target][src] = support[target].get(src, 0) + 1
        if len(examples) < 10 and float(values[row_index]) != old:
            examples.append({"id": int(row["id"]), "target": target, "old": old, "new": float(values[row_index]), "ionic": ion})

    if not np.isfinite(values).all():
        raise RuntimeError("Output contains non-finite predictions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "target": values}).to_csv(output_path, index=False)
    manifest = {
        "schema_version": "ppp.round2.clean-current-epsnc-ionic-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "CLEAN_OFFICIAL_ONLY",
        "branch": "without_archive",
        "local_eval_read_by_builder": False,
        "archive_labels_used": False,
        "with_archive_inputs_used": False,
        "base": {"path": str(base_path.relative_to(root)), "sha256": ionic.sha256_file(base_path)},
        "train": {"path": str(train_path.relative_to(root)), "sha256": ionic.sha256_file(train_path)},
        "test": {"path": str(test_path.relative_to(root)), "sha256": ionic.sha256_file(test_path)},
        "weights": {"eps": args.eps_weight, "nc": args.nc_weight},
        "ionic_predictor": ionic_report,
        "pair_rows": int(len(pair_frame)),
        "applied_rows": applied,
        "support": support,
        "examples": examples,
        "output": {
            "path": str(output_path.relative_to(root)),
            "sha256": ionic.sha256_file(output_path),
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
