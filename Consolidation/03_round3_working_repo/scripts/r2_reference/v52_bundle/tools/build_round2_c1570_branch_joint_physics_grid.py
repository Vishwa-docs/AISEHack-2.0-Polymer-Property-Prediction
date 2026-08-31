#!/usr/bin/env python3
"""C1570 branch-local joint physics projection grid.

This is an local_eval-free fixed-grid generator over a branch-local base candidate.
It reads official current train/test files and one branch-local base CSV, then
writes complete candidates for a declared grid of Egc/Egb, Ei/Eea/Egc, and
EPS/Nc consistency pulls.

The companion scoring step must run after outputs are frozen.  This script does
not read local_eval, external_label, nonofficial, Kaggle, external, pretrained, or cache files.
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
from rdkit import RDLogger

import build_round2_c1446_branch_joint_physics_projection as phys
import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, role: str, branch: str | None = None, require_branch: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if branch is not None:
        opposite = "without_archive" if branch == "with_archive" else "with_archive"
        if f"/{opposite}/" in low:
            raise RuntimeError(f"Refusing cross-branch {role} path for {branch}: {path}")
        if require_branch and f"/{branch}/" not in low:
            raise RuntimeError(f"{role} must be inside /{branch}/: {path}")
    if role in {"output dir"} and "polymer prediction challenge round 2" not in low:
        raise RuntimeError(f"{role} outside Round 2 boundary: {path}")


def parse_float_list(value: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise RuntimeError(f"No {name} values supplied")
    bad = [item for item in values if item < 0.0 or item > 0.5]
    if bad:
        raise RuntimeError(f"{name} values outside [0, 0.5]: {bad}")
    return values


def safe_param(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def safe_slug(path: Path) -> str:
    keep: list[str] = []
    for char in path.stem:
        keep.append(char if char.isalnum() or char in ("-", "_") else "-")
    return "".join(keep)[:90]


def apply_projection(
    *,
    base_values: np.ndarray,
    test: pd.DataFrame,
    train: pd.DataFrame,
    wide: pd.DataFrame,
    row_index: dict[tuple[str, str], int],
    egb_relation: dict[str, float],
    gap_relation: dict[str, float],
    median_ionic: float,
    egb_pull: float,
    gap_pull: float,
    epsnc_pull: float,
) -> tuple[np.ndarray, dict[str, int]]:
    values = base_values.copy()
    applied = {"egc_egb_groups": 0, "ei_eea_egc_groups": 0, "eps_nc_groups": 0}
    for canon_raw, row in wide.iterrows():
        canon = str(canon_raw)
        if {"egc", "egb"}.issubset(row.index) and pd.notna(row.get("egc")) and pd.notna(row.get("egb")):
            egc_new, egb_new = phys.project_egc_egb(float(row["egc"]), float(row["egb"]), egb_relation, egb_pull)
            for target, new_value in (("egc", egc_new), ("egb", egb_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
            applied["egc_egb_groups"] += 1
        if {"ei", "eea", "egc"}.issubset(row.index) and all(pd.notna(row.get(t)) for t in ("ei", "eea", "egc")):
            ei_new, eea_new, egc_new = phys.project_gap(
                float(row["ei"]),
                float(row["eea"]),
                float(row["egc"]),
                gap_relation,
                gap_pull,
            )
            for target, new_value in (("ei", ei_new), ("eea", eea_new), ("egc", egc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
            applied["ei_eea_egc_groups"] += 1
        if {"eps", "nc"}.issubset(row.index) and pd.notna(row.get("eps")) and pd.notna(row.get("nc")):
            eps_new, nc_new = phys.project_eps_nc(float(row["eps"]), float(row["nc"]), median_ionic, epsnc_pull)
            for target, new_value in (("eps", eps_new), ("nc", nc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
            applied["eps_nc_groups"] += 1
    target_type = test["target_type"].to_numpy(object)
    for target in TARGETS:
        mask = target_type == target
        values[mask] = phys.clip_target(values[mask], target, train)
    if not np.isfinite(values).all():
        raise RuntimeError("Projection produced non-finite values")
    return values, applied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--egb-pulls", default="0,0.02,0.05")
    parser.add_argument("--gap-pulls", default="0,0.01,0.02,0.035,0.05")
    parser.add_argument("--epsnc-pulls", default="0,0.01,0.02,0.035,0.05,0.075")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_path = root / "ppp-round-2" / "train.csv"
    test_path = root / "ppp-round-2" / "test.csv"
    base_path = (root / args.base_csv).resolve()
    output_dir = (root / args.output_dir).resolve()
    for path, role, require_branch in (
        (train_path, "train", False),
        (test_path, "test", False),
        (base_path, "base", True),
        (output_dir, "output dir", True),
    ):
        guard_path(path, role=role, branch=args.branch if role not in {"train", "test"} else None, require_branch=require_branch)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if sha256_file(train_path) != reference.EXPECTED_HASHES["train.csv"]:
        raise RuntimeError("train.csv hash mismatch")
    if sha256_file(test_path) != reference.EXPECTED_HASHES["test.csv"]:
        raise RuntimeError("test.csv hash mismatch")
    egb_pulls = parse_float_list(args.egb_pulls, name="egb-pulls")
    gap_pulls = parse_float_list(args.gap_pulls, name="gap-pulls")
    epsnc_pulls = parse_float_list(args.epsnc_pulls, name="epsnc-pulls")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for frame in (train, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
        frame["canonical"] = [reference.canonicalize(value) for value in frame["smiles"]]
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    base = phys.load_base(base_path, ids)
    base_values = base["target"].to_numpy(float)

    train_wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    eg_rows = train_wide[["egc", "egb"]].dropna()
    egb_relation = phys.fit_affine(eg_rows["egc"].to_numpy(float), eg_rows["egb"].to_numpy(float))
    gap_relation = phys.fit_gap_relation(train_wide)
    epsnc_rows = train_wide[["eps", "nc"]].dropna()
    ionic = epsnc_rows["eps"].to_numpy(float) - np.square(epsnc_rows["nc"].to_numpy(float))
    median_ionic = float(np.median(ionic)) if len(ionic) else 0.20
    median_ionic = max(median_ionic, 0.02)

    pred = test[["id", "canonical", "target_type"]].copy()
    pred["value"] = base_values
    wide = pred.pivot_table(index="canonical", columns="target_type", values="value", aggfunc="mean")
    row_index = {(str(row.canonical), str(row.target_type)): int(idx) for idx, row in test.iterrows()}

    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    sequence = 0
    for egb_pull in egb_pulls:
        for gap_pull in gap_pulls:
            for epsnc_pull in epsnc_pulls:
                values, applied = apply_projection(
                    base_values=base_values,
                    test=test,
                    train=train,
                    wide=wide,
                    row_index=row_index,
                    egb_relation=egb_relation,
                    gap_relation=gap_relation,
                    median_ionic=median_ionic,
                    egb_pull=egb_pull,
                    gap_pull=gap_pull,
                    epsnc_pull=epsnc_pull,
                )
                if np.max(np.abs(values - base_values)) <= 1.0e-12:
                    continue
                sequence += 1
                output = output_dir / (
                    f"R2-C1570-{args.branch}-joint-phys-e{safe_param(egb_pull)}-"
                    f"g{safe_param(gap_pull)}-p{safe_param(epsnc_pull)}-OVER-{safe_slug(base_path)}.csv"
                )
                if output.exists():
                    raise RuntimeError(f"Refusing overwrite: {output}")
                pd.DataFrame({"id": ids, "target": values}).to_csv(output, index=False)
                record = {
                    "schema_version": "ppp.round2.c1570.branch-joint-physics-grid.v1",
                    "sequence": sequence,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "branch": args.branch,
                    "target": "all",
                    "weight_on_source": None,
                    "classification": "OFFICIAL_INPUT_BRANCH_BASE_OVERLAY",
                    "local_eval_read_by_builder": False,
                    "external_label_file_read_by_builder": False,
                    "nonofficial_file_read_by_builder": False,
                    "kaggle_compute": False,
                    "kaggle_upload": False,
                    "kaggle_submission": False,
                    "official_current_train_used": True,
                    "official_current_test_used": True,
                    "base_may_use_archive_labels": bool(args.branch == "with_archive"),
                    "params": {"egb_pull": egb_pull, "gap_pull": gap_pull, "epsnc_pull": epsnc_pull},
                    "relations": {
                        "egb_from_egc": egb_relation,
                        "egc_from_ei_minus_eea": gap_relation,
                        "eps_minus_nc_squared_median": {"rows": int(len(epsnc_rows)), "median_ionic": median_ionic},
                    },
                    "applied": applied,
                    "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
                    "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(values)), "bytes": output.stat().st_size},
                }
                records.append(record)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "schema_version": "ppp.round2.c1570.branch-joint-physics-grid-summary.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "base": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "candidate_count": len(records),
        "grid": {"egb_pulls": egb_pulls, "gap_pulls": gap_pulls, "epsnc_pulls": epsnc_pulls},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "bytes": manifest_path.stat().st_size},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
