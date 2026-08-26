#!/usr/bin/env python3
"""Fixed branch-local joint physics projection for Round 2 candidates.

This is an official-input-only co-test consistency adjustment.  It reads:

* official current train.csv and test.csv;
* one branch-local base candidate CSV.

It does not read local_eval files, external_label files, archive labels, opposite-branch
predictions, external data, or Kaggle state.  The base candidate may be a
with-archive or noarchive branch parent; this solver itself uses current train
labels only and writes to the same branch namespace.

The projection is intentionally fixed, small, and preregistered by CLI values:

* Egc/Egb affine consistency learned from current train paired structures;
* Ei/Eea/Egc gap identity consistency learned from current train triples;
* EPS/Nc median-ionic consistency learned from current train paired structures.
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
from rdkit import Chem, RDLogger
from sklearn.linear_model import HuberRegressor, Ridge

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
EXPECTED_TRAIN_SHA = reference.EXPECTED_HASHES["train.csv"]
EXPECTED_TEST_SHA = reference.EXPECTED_HASHES["test.csv"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, role: str, branch: str) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if role in {"train", "test"} and ("/archive/" in low or "with_archive" in low or "without_archive" in low):
        # Official current files live directly under ppp-round-2, not branch namespaces.
        if "/ppp-round-2/train.csv" not in low and "/ppp-round-2/test.csv" not in low:
            raise RuntimeError(f"Unexpected current official {role} path: {path}")
    if role in {"base", "output", "manifest"}:
        opposite = "without_archive" if branch == "with_archive" else "with_archive"
        if opposite in low:
            raise RuntimeError(f"Refusing opposite branch {role} path: {path}")
        if branch not in low and ("noarchive" not in low or branch != "without_archive"):
            raise RuntimeError(f"{role} path must be branch-local for {branch}: {path}")


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Base ID order mismatch: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite values: {path}")
    return frame


def fit_affine(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 10:
        return {"intercept": 0.0, "slope": 1.0, "rows": int(len(x)), "r2": float("nan"), "fallback": True}
    model = Ridge(alpha=0.01)
    model.fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return {
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "rows": int(len(x)),
        "r2": float(1.0 - ss_res / max(ss_tot, 1.0e-12)),
        "fallback": False,
    }


def fit_gap_relation(wide: pd.DataFrame) -> dict[str, float]:
    rows = wide[["egc", "ei", "eea"]].dropna()
    if len(rows) < 10:
        return {"intercept": 0.0, "slope": 1.0, "rows": int(len(rows)), "r2": float("nan"), "fallback": True}
    x = (rows["ei"].to_numpy(float) - rows["eea"].to_numpy(float)).reshape(-1, 1)
    y = rows["egc"].to_numpy(float)
    model = HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=1000)
    model.fit(x, y)
    pred = model.predict(x)
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return {
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "rows": int(len(rows)),
        "r2": float(1.0 - ss_res / max(ss_tot, 1.0e-12)),
        "fallback": False,
    }


def project_egc_egb(egc0: float, egb0: float, relation: dict[str, float], pull: float) -> tuple[float, float]:
    a = float(relation["intercept"])
    b = float(relation["slope"])
    x_star = (egc0 + b * (egb0 - a)) / max(1.0 + b * b, 1.0e-12)
    y_star = a + b * x_star
    return (1.0 - pull) * egc0 + pull * x_star, (1.0 - pull) * egb0 + pull * y_star


def project_gap(ei0: float, eea0: float, egc0: float, relation: dict[str, float], pull: float) -> tuple[float, float, float]:
    a = float(relation["intercept"])
    b = float(relation["slope"])
    gap_from_ei_eea = a + b * (ei0 - eea0)
    if abs(b) < 1.0e-8:
        ei_target = ei0
        eea_target = eea0
    else:
        correction = (egc0 - gap_from_ei_eea) / (2.0 * b)
        ei_target = ei0 + correction
        eea_target = eea0 - correction
    egc_target = gap_from_ei_eea
    return (
        (1.0 - pull) * ei0 + pull * ei_target,
        (1.0 - pull) * eea0 + pull * eea_target,
        (1.0 - pull) * egc0 + pull * egc_target,
    )


def project_eps_nc(eps0: float, nc0: float, ionic: float, pull: float) -> tuple[float, float]:
    grid = np.linspace(1.0, 2.8, 901)
    eps_grid = grid * grid + ionic
    loss = np.square(eps_grid - eps0) + np.square(grid - nc0)
    idx = int(np.argmin(loss))
    nc_star = float(grid[idx])
    eps_star = float(eps_grid[idx])
    return (1.0 - pull) * eps0 + pull * eps_star, (1.0 - pull) * nc0 + pull * nc_star


def clip_target(values: np.ndarray, target: str, train: pd.DataFrame) -> np.ndarray:
    y = train.loc[train["target_type"] == target, "target"].to_numpy(float)
    q001, q999 = np.quantile(y, [0.002, 0.998])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1.0e-8)
    lower = max(0.0, q001 - 2.0 * margin) if target in {"eps", "nc"} else q001 - 2.0 * margin
    upper = q999 + 2.0 * margin
    return np.clip(values, lower, upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--egb-pull", type=float, default=0.08)
    parser.add_argument("--gap-pull", type=float, default=0.05)
    parser.add_argument("--epsnc-pull", type=float, default=0.02)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_path = root / "ppp-round-2" / "train.csv"
    test_path = root / "ppp-round-2" / "test.csv"
    base_path = (root / args.base).resolve()
    output_path = (root / args.output).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    for path, role in (
        (train_path, "train"),
        (test_path, "test"),
        (base_path, "base"),
        (output_path, "output"),
        (manifest_path, "manifest"),
    ):
        guard_path(path, role, args.branch)
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError(f"Refusing overwrite for C{args.cid}")
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("train.csv hash mismatch")
    if sha256_file(test_path) != EXPECTED_TEST_SHA:
        raise RuntimeError("test.csv hash mismatch")
    for name, value in (("egb_pull", args.egb_pull), ("gap_pull", args.gap_pull), ("epsnc_pull", args.epsnc_pull)):
        if not (0.0 <= float(value) <= 0.5):
            raise RuntimeError(f"{name} outside [0,0.5]: {value}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for frame in (train, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
        frame["canonical"] = [reference.canonicalize(value) for value in frame["smiles"]]
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = load_base(base_path, ids)
    values = base["target"].to_numpy(float).copy()

    train_wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    eg_rows = train_wide[["egc", "egb"]].dropna()
    egb_relation = fit_affine(eg_rows["egc"].to_numpy(float), eg_rows["egb"].to_numpy(float))
    gap_relation = fit_gap_relation(train_wide)
    epsnc_rows = train_wide[["eps", "nc"]].dropna()
    ionic = epsnc_rows["eps"].to_numpy(float) - np.square(epsnc_rows["nc"].to_numpy(float))
    median_ionic = float(np.median(ionic)) if len(ionic) else 0.20
    median_ionic = max(median_ionic, 0.02)

    pred = test[["id", "canonical", "target_type"]].copy()
    pred["value"] = values
    wide = pred.pivot_table(index="canonical", columns="target_type", values="value", aggfunc="mean")
    row_index = {(str(row.canonical), str(row.target_type)): int(idx) for idx, row in test.iterrows()}
    applied = {"egc_egb_groups": 0, "ei_eea_egc_groups": 0, "eps_nc_groups": 0}
    for canon, row in wide.iterrows():
        canon = str(canon)
        if {"egc", "egb"}.issubset(row.index) and pd.notna(row.get("egc")) and pd.notna(row.get("egb")):
            egc_new, egb_new = project_egc_egb(float(row["egc"]), float(row["egb"]), egb_relation, float(args.egb_pull))
            for target, new_value in (("egc", egc_new), ("egb", egb_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
            applied["egc_egb_groups"] += 1
        if {"ei", "eea", "egc"}.issubset(row.index) and all(pd.notna(row.get(t)) for t in ("ei", "eea", "egc")):
            ei_new, eea_new, egc_new = project_gap(float(row["ei"]), float(row["eea"]), float(row["egc"]), gap_relation, float(args.gap_pull))
            for target, new_value in (("ei", ei_new), ("eea", eea_new), ("egc", egc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
            applied["ei_eea_egc_groups"] += 1
        if {"eps", "nc"}.issubset(row.index) and pd.notna(row.get("eps")) and pd.notna(row.get("nc")):
            eps_new, nc_new = project_eps_nc(float(row["eps"]), float(row["nc"]), median_ionic, float(args.epsnc_pull))
            for target, new_value in (("eps", eps_new), ("nc", nc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
            applied["eps_nc_groups"] += 1

    for target in TARGETS:
        mask = test["target_type"].to_numpy(object) == target
        values[mask] = clip_target(values[mask], target, train)
    if not np.isfinite(values).all():
        raise RuntimeError("Output contains non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "target": values}).to_csv(output_path, index=False)
    manifest = {
        "schema_version": "ppp.round2.c1446.branch-joint-physics-projection.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "CLEAN_OFFICIAL_ONLY_BRANCH_LOCAL_OVERLAY",
        "branch": args.branch,
        "cid": int(args.cid),
        "official_only": True,
        "archive_labels_used_by_solver": False,
        "archive_file_read_by_solver": False,
        "base_may_use_archive_labels": bool(args.branch == "with_archive"),
        "local_eval_read": False,
        "external_label_file_read": False,
        "pretrained_weights": False,
        "external_data_used": False,
        "base": {"path": str(base_path.relative_to(root)), "sha256": sha256_file(base_path)},
        "train": {"path": str(train_path.relative_to(root)), "sha256": sha256_file(train_path)},
        "test": {"path": str(test_path.relative_to(root)), "sha256": sha256_file(test_path)},
        "pulls": {"egb_pull": float(args.egb_pull), "gap_pull": float(args.gap_pull), "epsnc_pull": float(args.epsnc_pull)},
        "relations": {
            "egb_from_egc": egb_relation,
            "egc_from_ei_minus_eea": gap_relation,
            "eps_minus_nc_squared_median": {"rows": int(len(epsnc_rows)), "median_ionic": median_ionic},
        },
        "applied": applied,
        "output": {"path": str(output_path.relative_to(root)), "sha256": sha256_file(output_path), "rows": int(len(values)), "bytes": output_path.stat().st_size},
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": manifest["output"], "applied": applied, "relations": manifest["relations"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
