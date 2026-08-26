#!/usr/bin/env python3
"""No-archive current-label physical identity overlay over a frozen base.

This builder does not read local_eval/external_label files. It uses only official current
train/test rows plus an already-frozen base prediction CSV. If the base is an
local_eval-assisted diagnostic, the output remains local_eval-assisted diagnostic too.

The tested identities are the same physical hypotheses audited in the earlier
C160 branch, but rewritten without archive labels or scratchpad arrays:

* Eea ~= Ei - Egc
* Ei  ~= Eea + Egc
* Egb ~= 1.1178 * Egc - 0.9221

For no-archive deployment support, an identity is applied only when at least
one required partner is observed in current train.csv for the same canonical
structure. Missing partner values are filled from the base prediction for the
same canonical/target when present in test.csv. This is diagnostic over the
base, not clean promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROUND2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROUND2_DIR / "tools"))
import initial_reference_pipeline as reference


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
EXPECTED_TRAIN_SHA = "609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2"
EXPECTED_TEST_SHA = "d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_read(path: Path, role: str) -> None:
    low = str(path).lower()
    forbidden = ("external_label", "test_external_labels") if role == "base" else ("local_eval", "external_label", "test_external_labels")
    for token in forbidden:
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "/archive/" in low or "/with_archive/" in low:
        raise RuntimeError(f"Refusing archive/cross-branch {role} path: {path}")


def canonical(smiles: str) -> str:
    return reference.canonicalize(str(smiles))


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base has non-finite predictions: {path}")
    return frame


def partner_value(canon: str, target: str, official_wide: pd.DataFrame, base_wide: pd.DataFrame) -> tuple[float | None, str]:
    if target in official_wide.columns and canon in official_wide.index:
        value = official_wide.at[canon, target]
        if pd.notna(value):
            return float(value), "official_current_train"
    if target in base_wide.columns and canon in base_wide.index:
        value = base_wide.at[canon, target]
        if pd.notna(value):
            return float(value), "base_same_canonical_test_prediction"
    return None, "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--base", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--eea-weight", type=float, default=0.50)
    parser.add_argument("--ei-weight", type=float, default=0.25)
    parser.add_argument("--egb-weight", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if "Polymer Prediction Challenge Round 2" not in str(root):
        raise RuntimeError(f"Root outside Round 2 boundary: {root}")
    train_path = root / "ppp-round-2" / "train.csv"
    test_path = root / "ppp-round-2" / "test.csv"
    base_path = (root / args.base).resolve()
    for path, role in ((train_path, "train"), (test_path, "test"), (base_path, "base")):
        guard_read(path, role)
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("train.csv hash mismatch")
    if sha256_file(test_path) != EXPECTED_TEST_SHA:
        raise RuntimeError("test.csv hash mismatch")
    for name, value in (("eea_weight", args.eea_weight), ("ei_weight", args.ei_weight), ("egb_weight", args.egb_weight)):
        if not (0.0 <= float(value) <= 1.0):
            raise RuntimeError(f"{name} outside [0, 1]: {value}")

    out_dir = root / "experiments" / "LOCAL_DIAGNOSTIC_ONLY"
    stem = f"R2-C{args.cid}-NOARCHIVE-LOCAL_DIAGNOSTIC_ONLY-CURRENT-IDENTITY-OVER-{base_path.stem}-20260808"
    output_path = out_dir / f"{stem}.csv"
    manifest_path = out_dir / f"{stem}.manifest.json"
    for path in (output_path, manifest_path):
        if path.exists():
            raise RuntimeError(f"Refusing overwrite: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["target_type"] = train["target_type"].astype(str).str.lower()
    test["target_type"] = test["target_type"].astype(str).str.lower()
    train["canonical"] = [canonical(value) for value in train["smiles"]]
    test["canonical"] = [canonical(value) for value in test["smiles"]]
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = load_base(base_path, ids)

    official_wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    test_pred = test[["id", "canonical", "target_type"]].copy()
    test_pred["base_target"] = base["target"].to_numpy(float)
    base_wide = test_pred.pivot_table(index="canonical", columns="target_type", values="base_target", aggfunc="mean")

    values = base["target"].to_numpy(float).copy()
    applied: dict[str, int] = {target: 0 for target in TARGETS}
    support: dict[str, dict[str, int]] = {target: {} for target in ("eea", "ei", "egb")}
    examples: list[dict[str, object]] = []
    for row_index, row in test.iterrows():
        target = str(row["target_type"])
        canon = str(row["canonical"])
        old = float(values[row_index])
        if target == "eea":
            ei, ei_src = partner_value(canon, "ei", official_wide, base_wide)
            egc, egc_src = partner_value(canon, "egc", official_wide, base_wide)
            if ei is not None and egc is not None and ei_src == "official_current_train":
                raw = ei - egc
                values[row_index] = (1.0 - args.eea_weight) * old + args.eea_weight * raw
                applied[target] += 1
                key = f"ei:{ei_src}|egc:{egc_src}"
                support[target][key] = support[target].get(key, 0) + 1
        elif target == "ei":
            eea, eea_src = partner_value(canon, "eea", official_wide, base_wide)
            egc, egc_src = partner_value(canon, "egc", official_wide, base_wide)
            if eea is not None and egc is not None and eea_src == "official_current_train":
                raw = eea + egc
                values[row_index] = (1.0 - args.ei_weight) * old + args.ei_weight * raw
                applied[target] += 1
                key = f"eea:{eea_src}|egc:{egc_src}"
                support[target][key] = support[target].get(key, 0) + 1
        elif target == "egb":
            egc, egc_src = partner_value(canon, "egc", official_wide, base_wide)
            if egc is not None and egc_src == "official_current_train":
                raw = 1.1178 * egc - 0.9221
                values[row_index] = (1.0 - args.egb_weight) * old + args.egb_weight * raw
                applied[target] += 1
                key = f"egc:{egc_src}"
                support[target][key] = support[target].get(key, 0) + 1
        if len(examples) < 10 and float(values[row_index]) != old:
            examples.append({"id": int(row["id"]), "target": target, "old": old, "new": float(values[row_index])})

    pd.DataFrame({"id": ids, "target": values}).to_csv(output_path, index=False)
    manifest = {
        "schema_version": "ppp.round2.local_eval-assisted-current-identity-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "LOCAL_DIAGNOSTIC_ONLY",
        "warning": "Base-dependent diagnostic. Builder reads no local_eval/external_label/archive files, but output inherits base evidence class.",
        "branch": "without_archive",
        "base": {"path": str(base_path.relative_to(root)), "sha256": sha256_file(base_path)},
        "train": {"path": str(train_path.relative_to(root)), "sha256": sha256_file(train_path)},
        "test": {"path": str(test_path.relative_to(root)), "sha256": sha256_file(test_path)},
        "weights": {"eea": args.eea_weight, "ei": args.ei_weight, "egb": args.egb_weight},
        "applied_rows": applied,
        "support": support,
        "examples": examples,
        "output": {"path": str(output_path.relative_to(root)), "sha256": sha256_file(output_path), "rows": int(len(values)), "bytes": output_path.stat().st_size},
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path.relative_to(root)), "applied_rows": applied, "sha256": manifest["output"]["sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
