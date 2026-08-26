#!/usr/bin/env python3
"""Diagnostic-only current EPS/NC ionic consistency overlay.

Construction reads only official current train/test rows plus a frozen base
prediction CSV.  If the base is an local_eval-assisted diagnostic, this output
inherits that evidence class.  LocalEval/external_label files are never read here; scoring
must happen only after the candidate CSV is written and hashed.

The bounded hypothesis is target-specific compound replacement:

* eps ~= nc^2 + ionic(smiles)
* nc  ~= sqrt(max(eps - ionic(smiles), 1.0))

Partner values come from current official train labels for the same canonical
structure when available, otherwise from the frozen base prediction for the
same canonical structure and partner target in test.csv.  Archive labels are
forbidden for this no-archive branch.
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


def fit_ionic_predictor(pair_canons: list[str], ionic: np.ndarray, mode: str):
    if mode == "median":
        median = float(np.median(ionic))

        def predict(canons: list[str]) -> np.ndarray:
            return np.full(len(canons), median, dtype=np.float64)

        return predict, {"mode": mode, "median": median}
    if mode not in {"extra_trees_raw", "extra_trees_log"}:
        raise RuntimeError(f"Unknown ionic mode: {mode}")
    x_train = polar_block(pair_canons)
    y = np.asarray(ionic, dtype=np.float64)
    if mode == "extra_trees_log":
        y = np.log(np.maximum(y, MIN_IONIC))
    model = ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=SEED,
        n_jobs=2,
    )
    model.fit(x_train, y)

    def predict(canons: list[str]) -> np.ndarray:
        raw = model.predict(polar_block(canons))
        if mode == "extra_trees_log":
            raw = np.exp(np.clip(raw, -8, 4))
        return np.maximum(np.asarray(raw, dtype=np.float64), MIN_IONIC)

    return predict, {"mode": mode, "train_rows": int(len(pair_canons)), "min_ionic": float(np.min(ionic)), "median_ionic": float(np.median(ionic)), "max_ionic": float(np.max(ionic))}


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
    parser.add_argument("--ionic-mode", choices=("median", "extra_trees_raw", "extra_trees_log"), default="extra_trees_raw")
    parser.add_argument("--eps-weight", type=float, default=0.10)
    parser.add_argument("--nc-weight", type=float, default=0.10)
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
    for name, value in (("eps_weight", args.eps_weight), ("nc_weight", args.nc_weight)):
        if not (0.0 <= float(value) <= 1.0):
            raise RuntimeError(f"{name} outside [0, 1]: {value}")

    out_dir = root / "experiments" / "LOCAL_DIAGNOSTIC_ONLY"
    stem = f"R2-C{args.cid}-NOARCHIVE-LOCAL_DIAGNOSTIC_ONLY-CURRENT-EPSNC-IONIC-OVER-{base_path.stem}-20260808"
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
    pair_frame = official_wide[["eps", "nc"]].dropna().copy()
    if len(pair_frame) < 50:
        raise RuntimeError("Insufficient current EPS/NC pairs")
    ionic = pair_frame["eps"].to_numpy(float) - pair_frame["nc"].to_numpy(float) ** 2
    if np.any(ionic <= 0):
        raise RuntimeError("Non-positive ionic coordinate in official current pairs")
    ionic_predict, ionic_report = fit_ionic_predictor(pair_frame.index.astype(str).tolist(), ionic, str(args.ionic_mode))

    test_pred = test[["id", "canonical", "target_type"]].copy()
    test_pred["base_target"] = base["target"].to_numpy(float)
    base_wide = test_pred.pivot_table(index="canonical", columns="target_type", values="base_target", aggfunc="mean")

    values = base["target"].to_numpy(float).copy()
    eps_train = train.loc[train["target_type"] == "eps", "target"].to_numpy(float)
    nc_train = train.loc[train["target_type"] == "nc", "target"].to_numpy(float)
    eps_low, eps_high = float(np.quantile(eps_train, 0.002)), float(np.quantile(eps_train, 0.998))
    nc_low, nc_high = float(np.quantile(nc_train, 0.002)), float(np.quantile(nc_train, 0.998))

    applied: dict[str, int] = {target: 0 for target in TARGETS}
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
        ion = max(float(ionic_cache[canon]), MIN_IONIC)
        old = float(values[row_index])
        if target == "eps":
            partner, src = partner_value(canon, "nc", official_wide, base_wide)
            if partner is None:
                continue
            raw = partner**2 + ion
            raw = float(np.clip(raw, max(0.0, eps_low - 0.05), eps_high + 0.05))
            values[row_index] = (1.0 - args.eps_weight) * old + args.eps_weight * raw
            applied[target] += 1
            support[target][src] = support[target].get(src, 0) + 1
        elif target == "nc":
            partner, src = partner_value(canon, "eps", official_wide, base_wide)
            if partner is None:
                continue
            raw = np.sqrt(max(partner - ion, 1.0))
            raw = float(np.clip(raw, max(1.0, nc_low - 0.05), min(2.8, nc_high + 0.05)))
            values[row_index] = (1.0 - args.nc_weight) * old + args.nc_weight * raw
            applied[target] += 1
            support[target][src] = support[target].get(src, 0) + 1
        if len(examples) < 10 and float(values[row_index]) != old:
            examples.append({"id": int(row["id"]), "target": target, "old": old, "new": float(values[row_index]), "ionic": ion})

    if not np.isfinite(values).all():
        raise RuntimeError("Output contains non-finite predictions")
    pd.DataFrame({"id": ids, "target": values}).to_csv(output_path, index=False)
    manifest = {
        "schema_version": "ppp.round2.local_eval-assisted-current-epsnc-ionic-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "LOCAL_DIAGNOSTIC_ONLY",
        "warning": "Base-dependent diagnostic. Builder reads no local_eval/external_label/archive files, but output inherits base evidence class.",
        "branch": "without_archive",
        "base": {"path": str(base_path.relative_to(root)), "sha256": sha256_file(base_path)},
        "train": {"path": str(train_path.relative_to(root)), "sha256": sha256_file(train_path)},
        "test": {"path": str(test_path.relative_to(root)), "sha256": sha256_file(test_path)},
        "weights": {"eps": args.eps_weight, "nc": args.nc_weight},
        "ionic_predictor": ionic_report,
        "pair_rows": int(len(pair_frame)),
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
