#!/usr/bin/env python3
"""C350 no-archive joint EPS/NC ionic consistency over C348.

This is the current-only counterpart to the F02 B3 idea: for structures that
appear as both EPS and NC rows in the official test set, predict an ionic
residual from official current train EPS/NC pairs and jointly reconcile the two
base predictions under eps = nc^2 + ionic.

It uses no archive labels, no PI1M, no local_eval/external_label files, and no Kaggle
action.  It writes one complete branch-local candidate for separate post-freeze
scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C348-NOARCHIVE-TARGET-SPLICE-C346-EGC-EPS-OVER-C343-20260808.csv"
)
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
PATS = {name: Chem.MolFromSmarts(smarts) for name, smarts in POLAR_SMARTS.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden path: {path}")
    if "/archive/" in low or low.endswith("/archive") or "with_archive" in low:
        raise RuntimeError(f"Refusing archive/cross-branch path for no-archive run: {path}")
    if allow_output and "without_archive" not in low:
        raise RuntimeError(f"Output must stay in without_archive namespace: {path}")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def no_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def polar_block(cans: list[str] | pd.Index | np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for can in cans:
        mol = Chem.MolFromSmiles(str(can))
        if mol is None:
            raise RuntimeError(f"Invalid canonical SMILES: {can}")
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in PATS.values()]
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


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path)
    if "without_archive" not in str(path):
        raise RuntimeError(f"C350 base must be without_archive: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def fit_ionic(cans: list[str], ionic: np.ndarray, leaf: int) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(n_estimators=800, min_samples_leaf=leaf, random_state=20260808, n_jobs=-1)
    model.fit(polar_block(cans), ionic)
    return model


def solve_pair(eps_base: float, nc_base: float, ionic: float, *, weight_eps: float, weight_nc: float) -> tuple[float, float]:
    ionic = max(float(ionic), MIN_IONIC)
    n = float(np.clip(nc_base, 1.0, 2.5))
    # Newton solve for min w_e*(n^2+i-e0)^2 + w_n*(n-n0)^2.
    for _ in range(40):
        diff = n * n + ionic - eps_base
        grad = 4.0 * weight_eps * n * diff + 2.0 * weight_nc * (n - nc_base)
        hess = 4.0 * weight_eps * (3.0 * n * n + ionic - eps_base) + 2.0 * weight_nc
        if abs(hess) < 1.0e-12:
            break
        step = grad / hess
        n_next = float(np.clip(n - step, 1.0, 2.5))
        if abs(n_next - n) < 1.0e-10:
            n = n_next
            break
        n = n_next
    # Small grid fallback/check because the objective is one-dimensional.
    grid = np.linspace(max(1.0, n - 0.12), min(2.5, n + 0.12), 241)
    obj = weight_eps * (grid * grid + ionic - eps_base) ** 2 + weight_nc * (grid - nc_base) ** 2
    n = float(grid[int(np.argmin(obj))])
    return float(n * n + ionic), n


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
    for path in (train_path, test_path, base_path):
        guard_path(path)
    for path in (output, manifest):
        guard_path(path, allow_output=True)
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
        frame["canon"] = [canonical_smiles(value) for value in frame["smiles"]]
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

    # Train-side ionic model audit only; this is not used as local_eval evidence.
    groups = np.asarray([no_stereo(value) for value in pairs.index], dtype=object)
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    ionic_oof = np.full(len(pairs), np.nan, dtype=np.float64)
    for tr, va in folds.split(pairs.index.to_numpy(), ionic, groups=groups):
        model = fit_ionic(pairs.index[tr].tolist(), ionic[tr], int(args.ionic_leaf))
        ionic_oof[va] = model.predict(polar_block(pairs.index[va]))
    ionic_oof = np.maximum(ionic_oof, MIN_IONIC)
    eps_phys_oof = pairs["nc"].to_numpy(float) ** 2 + ionic_oof
    nc_phys_oof = np.sqrt(np.maximum(pairs["eps"].to_numpy(float) - ionic_oof, 1.0))
    oof_report = {
        "pair_rows": int(len(pairs)),
        "ionic_oof_r2": float(r2_score(ionic, ionic_oof)),
        "eps_from_true_nc_oof_r2": float(r2_score(pairs["eps"].to_numpy(float), eps_phys_oof)),
        "nc_from_true_eps_oof_r2": float(r2_score(pairs["nc"].to_numpy(float), nc_phys_oof)),
    }

    model = fit_ionic(pairs.index.tolist(), ionic, int(args.ionic_leaf))
    test_pivot = test.pivot_table(index="canon", columns="tt", values="base_prediction", aggfunc="mean")
    result = test["base_prediction"].to_numpy(float).copy()
    changed = {"eps": [], "nc": []}
    pair_canons = [c for c in test_pivot.index if "eps" in test_pivot.columns and "nc" in test_pivot.columns and pd.notna(test_pivot.loc[c].get("eps", np.nan)) and pd.notna(test_pivot.loc[c].get("nc", np.nan))]
    if pair_canons:
        ionic_pred = pd.Series(np.maximum(model.predict(polar_block(pair_canons)), MIN_IONIC), index=pair_canons)
    else:
        ionic_pred = pd.Series(dtype=float)
    solved: dict[str, tuple[float, float]] = {}
    for canon in pair_canons:
        eps_base = float(test_pivot.loc[canon, "eps"])
        nc_base = float(test_pivot.loc[canon, "nc"])
        eps_star, nc_star = solve_pair(eps_base, nc_base, float(ionic_pred.loc[canon]), weight_eps=float(args.weight_eps), weight_nc=float(args.weight_nc))
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
    record = {
        "schema_version": "ppp.round2.c350.noarchive-joint-eps-nc-consistency.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "joint EPS/NC test co-row reconciliation under eps=nc^2+ionic with current-train ionic ExtraTrees",
        "config": {"pull": float(args.pull), "ionic_leaf": int(args.ionic_leaf), "weight_eps": float(args.weight_eps), "weight_nc": float(args.weight_nc), "min_ionic": MIN_IONIC},
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
