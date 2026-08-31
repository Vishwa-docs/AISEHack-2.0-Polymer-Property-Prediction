#!/usr/bin/env python3
"""Clean no-archive current-label identity overlay.

This is the clean-output wrapper for the current-identity overlay. It reads only:

- official current `ppp-round-2/train.csv`;
- official current `ppp-round-2/test.csv`;
- a frozen no-archive base prediction CSV.

It does not read local_eval, external_label, archive, with-archive, or Kaggle artifacts.
The output is a complete no-archive `final_submission_runs` CSV suitable for
separate post-freeze local_eval scoring.
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
SOURCE_MODULE = ROUND2_DIR / "tools" / "LOCAL_DIAGNOSTIC_ONLY" / "build_current_identity_overlay.py"
spec = importlib.util.spec_from_file_location("round2_identity_overlay_source", SOURCE_MODULE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load source module: {SOURCE_MODULE}")
identity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = identity
spec.loader.exec_module(identity)


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
        / f"R2-C{cid}-NOARCHIVE-CLEAN-CURRENT-IDENTITY-OVER-{base_path.stem}-20260808.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--base", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--eea-weight", type=float, default=0.10)
    parser.add_argument("--ei-weight", type=float, default=0.10)
    parser.add_argument("--egb-weight", type=float, default=0.0)
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
        identity.guard_read(path, role)
    if identity.sha256_file(train_path) != identity.EXPECTED_TRAIN_SHA:
        raise RuntimeError("train.csv hash mismatch")
    if identity.sha256_file(test_path) != identity.EXPECTED_TEST_SHA:
        raise RuntimeError("test.csv hash mismatch")
    for name, value in (
        ("eea_weight", args.eea_weight),
        ("ei_weight", args.ei_weight),
        ("egb_weight", args.egb_weight),
    ):
        if not (0.0 <= float(value) <= 1.0):
            raise RuntimeError(f"{name} outside [0, 1]: {value}")
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError(f"Refusing overwrite for C{args.cid}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["target_type"] = train["target_type"].astype(str).str.lower()
    test["target_type"] = test["target_type"].astype(str).str.lower()
    train["canonical"] = [identity.canonical(value) for value in train["smiles"]]
    test["canonical"] = [identity.canonical(value) for value in test["smiles"]]
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")

    base = identity.load_base(base_path, ids)
    official_wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    test_pred = test[["id", "canonical", "target_type"]].copy()
    test_pred["base_target"] = base["target"].to_numpy(float)
    base_wide = test_pred.pivot_table(index="canonical", columns="target_type", values="base_target", aggfunc="mean")

    values = base["target"].to_numpy(float).copy()
    applied: dict[str, int] = {target: 0 for target in identity.TARGETS}
    support: dict[str, dict[str, int]] = {target: {} for target in ("eea", "ei", "egb")}
    examples: list[dict[str, object]] = []
    for row_index, row in test.iterrows():
        target = str(row["target_type"])
        canon = str(row["canonical"])
        old = float(values[row_index])
        if target == "eea":
            ei, ei_src = identity.partner_value(canon, "ei", official_wide, base_wide)
            egc, egc_src = identity.partner_value(canon, "egc", official_wide, base_wide)
            if ei is not None and egc is not None and ei_src == "official_current_train":
                raw = ei - egc
                values[row_index] = (1.0 - args.eea_weight) * old + args.eea_weight * raw
                applied[target] += 1
                key = f"ei:{ei_src}|egc:{egc_src}"
                support[target][key] = support[target].get(key, 0) + 1
        elif target == "ei":
            eea, eea_src = identity.partner_value(canon, "eea", official_wide, base_wide)
            egc, egc_src = identity.partner_value(canon, "egc", official_wide, base_wide)
            if eea is not None and egc is not None and eea_src == "official_current_train":
                raw = eea + egc
                values[row_index] = (1.0 - args.ei_weight) * old + args.ei_weight * raw
                applied[target] += 1
                key = f"eea:{eea_src}|egc:{egc_src}"
                support[target][key] = support[target].get(key, 0) + 1
        elif target == "egb":
            egc, egc_src = identity.partner_value(canon, "egc", official_wide, base_wide)
            if egc is not None and egc_src == "official_current_train":
                raw = 1.1178 * egc - 0.9221
                values[row_index] = (1.0 - args.egb_weight) * old + args.egb_weight * raw
                applied[target] += 1
                key = f"egc:{egc_src}"
                support[target][key] = support[target].get(key, 0) + 1
        if len(examples) < 10 and float(values[row_index]) != old:
            examples.append({"id": int(row["id"]), "target": target, "old": old, "new": float(values[row_index])})

    if not np.isfinite(values).all():
        raise RuntimeError("Output contains non-finite predictions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "target": values}).to_csv(output_path, index=False)
    manifest = {
        "schema_version": "ppp.round2.clean-current-identity-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "CLEAN_OFFICIAL_ONLY",
        "branch": "without_archive",
        "local_eval_read_by_builder": False,
        "archive_labels_used": False,
        "with_archive_inputs_used": False,
        "base": {"path": str(base_path.relative_to(root)), "sha256": identity.sha256_file(base_path)},
        "train": {"path": str(train_path.relative_to(root)), "sha256": identity.sha256_file(train_path)},
        "test": {"path": str(test_path.relative_to(root)), "sha256": identity.sha256_file(test_path)},
        "weights": {"eea": args.eea_weight, "ei": args.ei_weight, "egb": args.egb_weight},
        "applied_rows": applied,
        "support": support,
        "examples": examples,
        "output": {
            "path": str(output_path.relative_to(root)),
            "sha256": identity.sha256_file(output_path),
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
