#!/usr/bin/env python3
"""No-archive current EPS/NC B3 consistency overlay.

Construction reads only official current train/test rows plus a frozen base
prediction CSV.  It never reads local_eval/external_label/archive files.  If the base is an
local_eval-assisted diagnostic, the output inherits that evidence class.

For EPS/NC co-test canonical structures, first apply the current ionic overlay
then solve a one-dimensional weighted projection onto

    eps = nc**2 + ionic(smiles)

and blend that consistent pair back into the target rows.
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
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor

ROUND2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROUND2_DIR / "tools"))
import initial_reference_pipeline as reference


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
EXPECTED_TRAIN_SHA = "609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2"
EXPECTED_TEST_SHA = "d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d"
SEED = 20260808
MIN_IONIC = 0.02

POLAR_SMARTS = {
    "CF": "[#6][F]",
    "CCl": "[#6][Cl]",
    "ester": "C(=O)O",
    "carbonyl": "[CX3]=[OX1]",
    "ether": "[OD2]([#6])[#6]",
    "OH": "[OX2H]",
    "nitrile": "C#N",
    "amide": "C(=O)N",
    "NH": "[NX3;H1,H2]",
    "sulfone": "S(=O)(=O)",
    "thioether": "[#16X2]",
    "aromatic_N": "n",
    "aromatic_O": "o",
    "aromatic_S": "s",
    "imide": "C(=O)NC(=O)",
    "siloxane": "[Si][O]",
    "phosphate": "P=O",
    "urethane": "NC(=O)O",
}
POLAR_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in POLAR_SMARTS.items()}


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
    if role != "base" and ("/archive/" in low or "/with_archive/" in low):
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


def polar_block(canons: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for canon in canons:
        mol = Chem.MolFromSmiles(str(canon))
        if mol is None:
            rows.append([0.0] * (len(POLAR_PATTERNS) + 8))
            continue
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in POLAR_PATTERNS.values()]
        row += [
            Descriptors.TPSA(mol) / heavy,
            Descriptors.NumHDonors(mol) / heavy,
            Descriptors.NumHAcceptors(mol) / heavy,
            Descriptors.FractionCSP3(mol),
            Descriptors.NumRotatableBonds(mol) / heavy,
            Crippen.MolMR(mol) / heavy,
            Crippen.MolLogP(mol) / heavy,
            rdMolDescriptors.CalcNumAromaticRings(mol) / heavy,
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def fit_ionic_predictor(pair_canons: list[str], ionic: np.ndarray):
    x_train = polar_block(pair_canons)
    model = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=SEED,
        n_jobs=2,
    )
    model.fit(x_train, np.asarray(ionic, dtype=np.float64))

    def predict(canons: list[str]) -> np.ndarray:
        raw = model.predict(polar_block(canons))
        return np.maximum(np.asarray(raw, dtype=np.float64), MIN_IONIC)

    return predict, {
        "mode": "extra_trees_raw_b3",
        "train_rows": int(len(pair_canons)),
        "min_ionic": float(np.min(ionic)),
        "median_ionic": float(np.median(ionic)),
        "max_ionic": float(np.max(ionic)),
    }


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


def project_pair(eps_ref: float, nc_ref: float, ionic: float, eps_weight: float, nc_weight: float) -> tuple[float, float]:
    grid = np.linspace(1.0, 2.8, 901)
    eps_grid = grid * grid + ionic
    loss = eps_weight * np.square(eps_grid - eps_ref) + nc_weight * np.square(grid - nc_ref)
    idx = int(np.argmin(loss))
    nc = float(grid[idx])
    eps = float(eps_grid[idx])
    return eps, nc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--base", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--eps-weight", type=float, default=0.10)
    parser.add_argument("--nc-weight", type=float, default=0.25)
    parser.add_argument("--consistency-pull", type=float, default=0.50)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_path = root / "ppp-round-2" / "train.csv"
    test_path = root / "ppp-round-2" / "test.csv"
    base_path = (root / args.base).resolve()
    for path, role in ((train_path, "train"), (test_path, "test"), (base_path, "base")):
        guard_read(path, role)
    if "/with_archive/" in str(base_path).lower() or "/archive/" in str(base_path).lower():
        raise RuntimeError(f"Refusing archive/cross-branch base path: {base_path}")
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("train.csv hash mismatch")
    if sha256_file(test_path) != EXPECTED_TEST_SHA:
        raise RuntimeError("test.csv hash mismatch")
    for name, value in (("eps_weight", args.eps_weight), ("nc_weight", args.nc_weight), ("consistency_pull", args.consistency_pull)):
        if not (0.0 <= float(value) <= 1.0):
            raise RuntimeError(f"{name} outside [0, 1]: {value}")

    out_dir = root / "experiments" / "LOCAL_DIAGNOSTIC_ONLY"
    stem = f"R2-C{args.cid}-NOARCHIVE-LOCAL_DIAGNOSTIC_ONLY-CURRENT-EPSNC-B3-OVER-{base_path.stem}-20260808"
    output_path = out_dir / f"{stem}.csv"
    manifest_path = out_dir / f"{stem}.manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError(f"Refusing overwrite for C{args.cid}")

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
    base_values = base["target"].to_numpy(float)
    values = base_values.copy()

    official_wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    pair_frame = official_wide[["eps", "nc"]].dropna().copy()
    ionic = pair_frame["eps"].to_numpy(float) - pair_frame["nc"].to_numpy(float) ** 2
    if len(pair_frame) < 50 or np.any(ionic <= 0):
        raise RuntimeError("Insufficient or invalid current EPS/NC pairs")
    ionic_predict, ionic_report = fit_ionic_predictor(pair_frame.index.astype(str).tolist(), ionic)

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
        ion = max(float(ionic_cache[canon]), MIN_IONIC)
        old = float(values[row_index])
        if target == "eps":
            partner, src = partner_value(canon, "nc", official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(partner**2 + ion, max(0.0, eps_low - 0.05), eps_high + 0.05))
            values[row_index] = (1.0 - args.eps_weight) * old + args.eps_weight * raw
        else:
            partner, src = partner_value(canon, "eps", official_wide, base_wide)
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
        ion = max(float(ionic_cache.get(canon, ionic_predict([canon])[0])), MIN_IONIC)
        eps_cons, nc_cons = project_pair(
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
    pd.DataFrame({"id": ids, "target": values}).to_csv(output_path, index=False)
    manifest = {
        "schema_version": "ppp.round2.local_eval-assisted-current-epsnc-b3-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "LOCAL_DIAGNOSTIC_ONLY",
        "warning": "Base-dependent diagnostic. Builder reads no local_eval/external_label/archive files, but output inherits base evidence class.",
        "branch": "without_archive",
        "base": {"path": str(base_path.relative_to(root)), "sha256": sha256_file(base_path)},
        "train": {"path": str(train_path.relative_to(root)), "sha256": sha256_file(train_path)},
        "test": {"path": str(test_path.relative_to(root)), "sha256": sha256_file(test_path)},
        "weights": {"eps": args.eps_weight, "nc": args.nc_weight, "consistency_pull": args.consistency_pull},
        "ionic_predictor": ionic_report,
        "pair_rows": int(len(pair_frame)),
        "applied_rows": applied,
        "support": support,
        "output": {"path": str(output_path.relative_to(root)), "sha256": sha256_file(output_path), "rows": int(len(values)), "bytes": output_path.stat().st_size},
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path.relative_to(root)), "applied_rows": applied, "sha256": manifest["output"]["sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
