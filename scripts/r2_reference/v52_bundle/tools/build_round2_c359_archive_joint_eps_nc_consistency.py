#!/usr/bin/env python3
"""C359 with-archive joint EPS/NC ionic consistency over an archive base.

This is the archive-branch counterpart of C350.  It uses the same current-only
official EPS/NC ionic fit and test co-row reconciliation, but permits a
with_archive base candidate as the prediction carrier.  No archive labels are
used by the ionic model itself.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import build_round2_c350_noarchive_joint_eps_nc_consistency as c350
import initial_reference_pipeline as reference


DEFAULT_BASE = (
    "experiments/final_submission_runs/with_archive/"
    "R2-C334-ARCHIVE-TARGET-SPLICE-C333-EEA-EPS-C332-NC-20260808.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, role: str, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "without_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for with_archive run: {path}")
    if allow_output and "/with_archive/" not in low:
        raise RuntimeError(f"{role} path must stay in with_archive namespace: {path}")


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path, role="base candidate")
    if "/with_archive/" not in str(path):
        raise RuntimeError(f"C359 base must be with_archive: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="ppp-round-2/train.csv")
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--pull", type=float, default=0.50)
    parser.add_argument("--ionic-leaf", type=int, default=2)
    parser.add_argument("--weight-eps", type=float, default=1.0)
    parser.add_argument("--weight-nc", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    train_path = Path(args.train_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path, role in ((train_path, "train"), (test_path, "test"), (base_path, "base")):
        guard_path(path, role=role)
    for path, role in ((output, "output"), (manifest, "manifest")):
        guard_path(path, role=role, allow_output=True)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")
    if not (0.0 <= args.pull <= 1.0):
        raise RuntimeError("--pull must be in [0, 1]")
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)
    if train_sha != reference.EXPECTED_HASHES["train.csv"]:
        raise RuntimeError("train.csv hash mismatch")
    if test_sha != reference.EXPECTED_HASHES["test.csv"]:
        raise RuntimeError("test.csv hash mismatch")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for frame in (train, test):
        frame["tt"] = frame["target_type"].astype(str).str.lower()
        frame["canon"] = [c350.canonical_smiles(value) for value in frame["smiles"]]
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = load_base(base_path, ids)
    test["base_prediction"] = base["target"].to_numpy(float)

    wide = train.pivot_table(index="canon", columns="tt", values="target", aggfunc="mean")
    pairs = wide[["eps", "nc"]].dropna()
    if len(pairs) < 50:
        raise RuntimeError("Insufficient current official EPS/NC pairs")
    ionic = pairs["eps"].to_numpy(float) - pairs["nc"].to_numpy(float) ** 2
    if float(np.min(ionic)) < 0.0:
        raise RuntimeError("Unexpected negative ionic residual in official train pairs")

    groups = np.asarray([c350.no_stereo(value) for value in pairs.index], dtype=object)
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    ionic_oof = np.full(len(pairs), np.nan, dtype=np.float64)
    for tr, va in folds.split(pairs.index.to_numpy(), ionic, groups=groups):
        model = c350.fit_ionic(pairs.index[tr].tolist(), ionic[tr], int(args.ionic_leaf))
        ionic_oof[va] = model.predict(c350.polar_block(pairs.index[va]))
    ionic_oof = np.maximum(ionic_oof, c350.MIN_IONIC)
    eps_phys_oof = pairs["nc"].to_numpy(float) ** 2 + ionic_oof
    nc_phys_oof = np.sqrt(np.maximum(pairs["eps"].to_numpy(float) - ionic_oof, 1.0))
    oof_report = {
        "pair_rows": int(len(pairs)),
        "ionic_oof_r2": float(r2_score(ionic, ionic_oof)),
        "eps_from_true_nc_oof_r2": float(r2_score(pairs["eps"].to_numpy(float), eps_phys_oof)),
        "nc_from_true_eps_oof_r2": float(r2_score(pairs["nc"].to_numpy(float), nc_phys_oof)),
    }

    model = c350.fit_ionic(pairs.index.tolist(), ionic, int(args.ionic_leaf))
    test_pivot = test.pivot_table(index="canon", columns="tt", values="base_prediction", aggfunc="mean")
    result = test["base_prediction"].to_numpy(float).copy()
    changed = {"eps": [], "nc": []}
    pair_canons = [
        c
        for c in test_pivot.index
        if "eps" in test_pivot.columns
        and "nc" in test_pivot.columns
        and pd.notna(test_pivot.loc[c].get("eps", np.nan))
        and pd.notna(test_pivot.loc[c].get("nc", np.nan))
    ]
    ionic_pred = pd.Series(np.maximum(model.predict(c350.polar_block(pair_canons)), c350.MIN_IONIC), index=pair_canons) if pair_canons else pd.Series(dtype=float)
    solved: dict[str, tuple[float, float]] = {}
    for canon in pair_canons:
        eps_base = float(test_pivot.loc[canon, "eps"])
        nc_base = float(test_pivot.loc[canon, "nc"])
        eps_star, nc_star = c350.solve_pair(
            eps_base,
            nc_base,
            float(ionic_pred.loc[canon]),
            weight_eps=float(args.weight_eps),
            weight_nc=float(args.weight_nc),
        )
        solved[str(canon)] = (
            (1.0 - float(args.pull)) * eps_base + float(args.pull) * eps_star,
            (1.0 - float(args.pull)) * nc_base + float(args.pull) * nc_star,
        )
    for row_idx, row in test.iterrows():
        canon = str(row["canon"])
        if canon not in solved:
            continue
        target = str(row["tt"])
        if target == "eps":
            result[int(row_idx)] = solved[canon][0]
            changed["eps"].append(int(row["id"]))
        elif target == "nc":
            result[int(row_idx)] = solved[canon][1]
            changed["nc"].append(int(row["id"]))
    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record: dict[str, Any] = {
        "schema_version": "ppp.round2.c359.archive-joint-eps-nc-consistency.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "official_current_train_used": True,
        "archive_labels_used_by_ionic_model": False,
        "archive_base_predictions_used": True,
        "archive_file_read": False,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "with_archive base joint EPS/NC test co-row reconciliation under eps=nc^2+ionic with current-train ionic ExtraTrees",
        "config": {"pull": float(args.pull), "ionic_leaf": int(args.ionic_leaf), "weight_eps": float(args.weight_eps), "weight_nc": float(args.weight_nc), "min_ionic": c350.MIN_IONIC},
        "oof_ionic_model_audit": oof_report,
        "inputs": {
            "train.csv": {"path": str(train_path), "sha256": train_sha, "bytes": train_path.stat().st_size},
            "test.csv": {"path": str(test_path), "sha256": test_sha, "bytes": test_path.stat().st_size},
            "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        },
        "changed_rows": {
            "eps": len(changed["eps"]),
            "nc": len(changed["nc"]),
            "eps_ids_sha256": hashlib.sha256(json.dumps(changed["eps"], separators=(",", ":")).encode("utf-8")).hexdigest(),
            "nc_ids_sha256": hashlib.sha256(json.dumps(changed["nc"], separators=(",", ":")).encode("utf-8")).hexdigest(),
            "paired_canonical_rows": len(pair_canons),
        },
        "rows": {"train": int(len(train)), "test": int(len(test)), "official_eps_nc_train_pairs": int(len(pairs))},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "changed_rows": record["changed_rows"], "oof_ionic_model_audit": oof_report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
