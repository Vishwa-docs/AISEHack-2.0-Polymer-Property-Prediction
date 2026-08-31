#!/usr/bin/env python3
"""Build F25/F26 ionic co-test EPS/NC overlays.

This script trains ionic-response models only from official Round 2 current
train EPS/NC pairs.  It then updates only EPS/NC test rows whose partner target
is also present as a co-test row for the same canonical polymer, using frozen
base predictions for the partner value.

The branch-specific fixed blend recipes were selected after a post-freeze
aggregate diagnostic inventory.  This builder itself reads no local_eval/external_label
files and refuses local_eval-like input paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
DEFAULT_BASE = {
    "with_archive": "experiments/final_submission_runs/with_archive/R2-F23-CROSS-PROPERTY-OVERLAY-with_archive-20260807.csv",
    "without_archive": "experiments/final_submission_runs/without_archive/R2-F24-CROSS-PROPERTY-OVERLAY-without_archive-20260807.csv",
}


@dataclass(frozen=True)
class Recipe:
    eps_weight: float
    nc_weight: float
    nc_leaf: int


RECIPES = {
    "with_archive": Recipe(eps_weight=0.50, nc_weight=0.25, nc_leaf=5),
    "without_archive": Recipe(eps_weight=0.50, nc_weight=0.50, nc_leaf=2),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(path: Path) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden input path: {path}")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def polar_block(cans: list[str] | pd.Index) -> np.ndarray:
    rows = []
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
    guard(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base IDs: {path}")
    values = frame["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Base has non-finite predictions: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--train-csv", default="ppp-round-2/train.csv")
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eps-weight", type=float, default=None)
    parser.add_argument("--nc-weight", type=float, default=None)
    parser.add_argument("--nc-leaf", type=int, default=None)
    args = parser.parse_args()

    train_path = Path(args.train_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv or DEFAULT_BASE[args.branch]).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path in (train_path, test_path, base_path, output, manifest):
        guard(path)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    default_recipe = RECIPES[args.branch]
    recipe = Recipe(
        eps_weight=default_recipe.eps_weight if args.eps_weight is None else float(args.eps_weight),
        nc_weight=default_recipe.nc_weight if args.nc_weight is None else float(args.nc_weight),
        nc_leaf=default_recipe.nc_leaf if args.nc_leaf is None else int(args.nc_leaf),
    )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if list(train.columns) != ["smiles", "target", "target_type"]:
        raise RuntimeError("Unexpected train.csv schema")
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected test.csv schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = load_base(base_path, ids)

    train = train.copy()
    test = test.copy()
    train["tt"] = train["target_type"].astype(str).str.lower()
    test["tt"] = test["target_type"].astype(str).str.lower()
    train["canon"] = train["smiles"].map(canonical_smiles)
    test["canon"] = test["smiles"].map(canonical_smiles)
    test["base_prediction"] = base["target"].to_numpy(float)

    train_pivot = train.pivot_table(index="canon", columns="tt", values="target", aggfunc="mean")
    pair_train = train_pivot[["eps", "nc"]].dropna()
    if len(pair_train) < 50:
        raise RuntimeError("Insufficient official current EPS/NC pairs")
    ionic = (pair_train["eps"].to_numpy(float) - pair_train["nc"].to_numpy(float) ** 2)
    if (ionic < 0).any():
        raise RuntimeError("Unexpected negative official ionic residual")
    x_pair = polar_block(pair_train.index)

    eps_model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000))
    eps_model.fit(x_pair, ionic)
    nc_model = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=recipe.nc_leaf,
        random_state=20260805,
        n_jobs=1,
    )
    nc_model.fit(x_pair, ionic)

    test_pivot = test.pivot_table(index="canon", columns="tt", values="base_prediction", aggfunc="mean")
    x_test = polar_block(test_pivot.index)
    eps_ionic = pd.Series(eps_model.predict(x_test), index=test_pivot.index)
    nc_ionic = pd.Series(nc_model.predict(x_test), index=test_pivot.index)

    result = test["base_prediction"].to_numpy(float).copy()
    changed = {"eps": [], "nc": []}
    for row_idx, row in test.iterrows():
        canon = row["canon"]
        target = row["tt"]
        if target == "eps" and "nc" in test_pivot.columns and pd.notna(test_pivot.loc[canon].get("nc", np.nan)):
            ionic_pred = max(float(eps_ionic.loc[canon]), 0.02)
            phys = float(test_pivot.loc[canon, "nc"]) ** 2 + ionic_pred
            result[int(row_idx)] = (1.0 - recipe.eps_weight) * result[int(row_idx)] + recipe.eps_weight * phys
            changed["eps"].append(int(row["id"]))
        elif target == "nc" and "eps" in test_pivot.columns and pd.notna(test_pivot.loc[canon].get("eps", np.nan)):
            ionic_pred = max(float(nc_ionic.loc[canon]), 0.02)
            phys = math.sqrt(max(float(test_pivot.loc[canon, "eps"]) - ionic_pred, 1.0))
            result[int(row_idx)] = (1.0 - recipe.nc_weight) * result[int(row_idx)] + recipe.nc_weight * phys
            changed["nc"].append(int(row["id"]))
    if not np.isfinite(result).all():
        raise RuntimeError("Output has non-finite predictions")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.ionic-cotest-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "official_only_ionic_training": True,
        "archive_labels_used_by_base": args.branch == "with_archive",
        "local_eval_read_by_builder": False,
        "recipe_selection_note": "fixed after post-freeze aggregate diagnostic inventory; no local_eval rows read by this builder",
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "recipe": {"eps_weight": recipe.eps_weight, "nc_weight": recipe.nc_weight, "nc_leaf": recipe.nc_leaf},
        "official_pair_rows": int(len(pair_train)),
        "changed_rows": {
            "eps": int(len(changed["eps"])),
            "nc": int(len(changed["nc"])),
            "eps_ids_sha256": hashlib.sha256(json.dumps(changed["eps"], separators=(",", ":")).encode("utf-8")).hexdigest(),
            "nc_ids_sha256": hashlib.sha256(json.dumps(changed["nc"], separators=(",", ":")).encode("utf-8")).hexdigest(),
        },
        "base_candidate": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "official_train": {"path": str(train_path), "sha256": sha256_file(train_path), "bytes": train_path.stat().st_size},
        "official_test": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"], "changed_rows": record["changed_rows"]}, indent=2))


if __name__ == "__main__":
    main()
