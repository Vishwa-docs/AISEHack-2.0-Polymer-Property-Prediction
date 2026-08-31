#!/usr/bin/env python3
"""Official-only Polymer train/evaluate/predict loop.

Training, fitting, feature preprocessing, blending, and test prediction use only
`Polymer Prediction Challenge/aisehack-2-0/train.csv` and `test.csv`.
External/nonofficial external_label files are loaded only after a prediction CSV exists, and
only for validation/reporting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, BRICS, Descriptors, Descriptors3D, Lipinski, MACCSkeys, RDKFingerprint, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter
from rdkit.DataStructs import ConvertToNumpyArray
from scipy import sparse
from scipy.optimize import nnls
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.svm import SVR

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except Exception:
    cb = None
    CATBOOST_AVAILABLE = False

try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors

    MORDRED_AVAILABLE = True
except Exception:
    MordredCalculator = None
    mordred_descriptors = None
    MORDRED_AVAILABLE = False

try:
    from polymer_property_prediction import polymer_properties_from_smiles as bicerano_ppf

    BICERANO_AVAILABLE = True
except Exception:
    bicerano_ppf = None
    BICERANO_AVAILABLE = False


rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Polymer Prediction Challenge" / "aisehack-2-0"
SUBMISSION_DIR = REPO_ROOT / "Polymer Prediction Challenge" / "submissions"
RUN_ROOT = REPO_ROOT / "experiments" / "polymer" / "official_loops"
DEFAULT_EXTERNAL_LABELS = REPO_ROOT / "Polymer Prediction Challenge" / "nonofficial" / "nonofficial" / "test_external_labels_expanded_nostereo.csv"
FALLBACK_EXTERNAL_LABELS = REPO_ROOT / "Polymer Prediction Challenge" / "nonofficial" / "nonofficial" / "test_external_labels.csv"
TARGETS = ("Tg", "Egc")

SMARTS_MOTIFS: tuple[tuple[str, str], ...] = (
    ("ester", "[CX3](=O)[OX2H0][#6]"),
    ("carbonate", "[OX2][CX3](=[OX1])[OX2]"),
    ("amide", "[NX3][CX3](=[OX1])[#6]"),
    ("imide", "[NX3]([CX3]=[OX1])[CX3]=[OX1]"),
    ("cyclic_imide", "[NX3]1[CX3](=[OX1])[#6][CX3](=[OX1])1"),
    ("phthalimide_like", "O=C1N([#6])C(=O)c2ccccc12"),
    ("naphthalimide_like", "O=C1N([#6])C(=O)c2ccc3ccccc3c12"),
    ("maleimide_like", "O=C1NC(=O)C=C1"),
    ("urethane", "[NX3][CX3](=[OX1])[OX2]"),
    ("urea", "[NX3][CX3](=[OX1])[NX3]"),
    ("aromatic_urea", "c[NX3][CX3](=[OX1])[NX3]c"),
    ("ether", "[OD2]([#6])[#6]"),
    ("phenoxy", "c[OX2][#6]"),
    ("sulfone", "[SX4](=[OX1])(=[OX1])([#6])[#6]"),
    ("sulfoxide", "[SX3](=[OX1])([#6])[#6]"),
    ("sulfonamide", "[SX4](=[OX1])(=[OX1])([NX3])[#6]"),
    ("nitrile", "[CX2]#N"),
    ("dicyano_methine", "[CX3]([CX2]#N)([CX2]#N)"),
    ("alkene", "[CX3]=[CX3]"),
    ("alkyne", "[CX2]#[CX2]"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("benzophenone", "c[CX3](=O)c"),
    ("anhydride", "[CX3](=[OX1])[OX2][CX3](=[OX1])"),
    ("aromatic_n", "[n,N;R]"),
    ("aromatic_o", "[o,O;R]"),
    ("aromatic_s", "[s,S;R]"),
    ("benzothiadiazole_like", "c1nc2scnc2c1"),
    ("benzoxazole_like", "c1nc2occc2c1"),
    ("carbazole_like", "c1ccc2[nH,nX3]c3ccccc3c2c1"),
    ("fluoroaryl", "c[F]"),
    ("chloroaryl", "c[Cl]"),
    ("bromoaryl", "c[Br]"),
    ("aryl_trifluoromethyl", "cC(F)(F)F"),
    ("perfluoroalkyl", "[CX4](F)(F)[CX4](F)(F)"),
    ("siloxane", "[Si][OX2][Si]"),
    ("organosilicon", "[Si]([#6])([#6])"),
    ("organotin", "[Sn]"),
    ("phosphonate", "[PX4](=[OX1])([OX2])[OX2]"),
    ("phosphoric_acid", "[PX4](=[OX1])([OX2H])[OX2H]"),
    ("phosphine_oxide", "[PX4](=[OX1])([#6])([#6])"),
    ("quaternary_ammonium", "[NX4+]"),
    ("azo", "[NX2]=[NX2]"),
)
COMPILED_SMARTS: tuple[tuple[str, Chem.Mol], ...] = tuple(
    (name, pattern) for name, smarts in SMARTS_MOTIFS if (pattern := Chem.MolFromSmarts(smarts)) is not None
)

REGION_DESCRIPTOR_NAMES = [
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "HeavyAtomCount",
    "FractionCSP3",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "BertzCT",
    "BalabanJ",
    "Kappa1",
    "Kappa2",
    "Kappa3",
]
REGION_ATOM_NUMBERS = (0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: dict[str, Any]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return value


def append_progress(run_dir: Path, stage: str, **payload: Any) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        **payload,
    }
    line = json.dumps(json_safe(record), sort_keys=True, allow_nan=False)
    with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def canonical_target(value: Any) -> str:
    token = str(value).strip().lower()
    if token == "tg":
        return "Tg"
    if token == "egc":
        return "Egc"
    raise ValueError(f"unknown target_type {value!r}")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    if train.columns.tolist() != ["smiles", "target", "target_type"]:
        raise RuntimeError(f"unexpected train columns: {train.columns.tolist()}")
    if test.columns.tolist() != ["id", "smiles", "target_type"]:
        raise RuntimeError(f"unexpected test columns: {test.columns.tolist()}")
    train = train.copy()
    test = test.copy()
    train["row_index"] = np.arange(len(train), dtype=np.int64)
    test["test_index"] = np.arange(len(test), dtype=np.int64)
    train["target_type"] = train["target_type"].map(canonical_target)
    test["target_type"] = test["target_type"].map(canonical_target)
    train["target"] = pd.to_numeric(train["target"], errors="raise").astype(np.float64)
    if len(train) != 6171 or len(test) != 4115:
        raise RuntimeError(f"unexpected row counts train={len(train)} test={len(test)}")
    if train["target_type"].value_counts().to_dict() != {"Tg": 4143, "Egc": 2028}:
        raise RuntimeError(f"unexpected train target counts {train['target_type'].value_counts().to_dict()}")
    if test["target_type"].value_counts().to_dict() != {"Tg": 2763, "Egc": 1352}:
        raise RuntimeError(f"unexpected test target counts {test['target_type'].value_counts().to_dict()}")
    if test["id"].duplicated().any():
        raise RuntimeError("test IDs are not unique")
    return train, test


def mol_from_smiles(smiles: str, label: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        raise RuntimeError(f"RDKit parse failed for {label}")
    return mol


def build_mols(smiles: list[str]) -> list[Chem.Mol]:
    return [mol_from_smiles(value, f"row={index}") for index, value in enumerate(smiles)]


def descriptor_matrix(mols: list[Chem.Mol], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    descriptor_items = list(Descriptors._descList)
    names = [name for name, _ in descriptor_items]
    values = np.empty((len(mols), len(names)), dtype=np.float64)
    values.fill(np.nan)
    for row, mol in enumerate(mols):
        for col, (_, func) in enumerate(descriptor_items):
            try:
                value = float(func(mol))
            except Exception:
                value = math.nan
            values[row, col] = value if math.isfinite(value) else math.nan

    extra_names = [
        "smiles_len",
        "star_count",
        "atom_count",
        "heavy_atom_count",
        "dummy_atom_count",
        "ring_count",
        "aromatic_atom_count",
        "hetero_atom_count",
        "halogen_count",
        "n_count",
        "o_count",
        "s_count",
        "si_count",
        "f_count",
        "cl_count",
        "br_count",
        "double_bond_count",
        "triple_bond_count",
        "branch_count",
        "bracket_count",
    ]
    extra = np.zeros((len(mols), len(extra_names)), dtype=np.float64)
    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        extra[row, 0] = len(str(smi))
        extra[row, 1] = str(smi).count("*")
        extra[row, 2] = len(atoms)
        extra[row, 3] = sum(1 for atom in atoms if atom.GetAtomicNum() > 1)
        extra[row, 4] = sum(1 for atom in atoms if atom.GetAtomicNum() == 0)
        extra[row, 5] = mol.GetRingInfo().NumRings()
        extra[row, 6] = sum(1 for atom in atoms if atom.GetIsAromatic())
        extra[row, 7] = sum(1 for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6))
        extra[row, 8] = sum(1 for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53))
        for col, atomic_num in ((9, 7), (10, 8), (11, 16), (12, 14), (13, 9), (14, 17), (15, 35)):
            extra[row, col] = sum(1 for atom in atoms if atom.GetAtomicNum() == atomic_num)
        extra[row, 16] = sum(1 for bond in bonds if str(bond.GetBondType()) == "DOUBLE")
        extra[row, 17] = sum(1 for bond in bonds if str(bond.GetBondType()) == "TRIPLE")
        extra[row, 18] = str(smi).count("(")
        extra[row, 19] = str(smi).count("[")
    estate_names = [f"estate_min_{index}" for index in range(79)] + [f"estate_max_{index}" for index in range(79)]
    estate = np.empty((len(mols), len(estate_names)), dtype=np.float64)
    estate.fill(np.nan)
    for row, mol in enumerate(mols):
        try:
            mins, maxs = EStateFingerprinter.FingerprintMol(mol)
            parsed = np.asarray(list(mins) + list(maxs), dtype=np.float64)
            if parsed.shape[0] == len(estate_names):
                estate[row] = parsed
        except Exception:
            pass
    return np.hstack([values, extra, estate]), names + extra_names + estate_names


def mordred_descriptor_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    if not MORDRED_AVAILABLE or MordredCalculator is None or mordred_descriptors is None:
        raise RuntimeError("mordred is not available in this environment")
    calculator = MordredCalculator(mordred_descriptors, ignore_3D=True)
    frame = calculator.pandas(mols, nproc=1, quiet=True)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    names = [f"mordred_{str(name)}" for name in numeric.columns]
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values, names


def rdkit_3d_descriptor_matrix(smiles: list[str], *, seed: int, optimize_steps: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    descriptor_functions = [
        ("rdkit3d_Asphericity", Descriptors3D.Asphericity),
        ("rdkit3d_Eccentricity", Descriptors3D.Eccentricity),
        ("rdkit3d_InertialShapeFactor", Descriptors3D.InertialShapeFactor),
        ("rdkit3d_NPR1", Descriptors3D.NPR1),
        ("rdkit3d_NPR2", Descriptors3D.NPR2),
        ("rdkit3d_PBF", Descriptors3D.PBF),
        ("rdkit3d_PMI1", Descriptors3D.PMI1),
        ("rdkit3d_PMI2", Descriptors3D.PMI2),
        ("rdkit3d_PMI3", Descriptors3D.PMI3),
        ("rdkit3d_RadiusOfGyration", Descriptors3D.RadiusOfGyration),
        ("rdkit3d_SpherocityIndex", Descriptors3D.SpherocityIndex),
    ]
    names = [name for name, _ in descriptor_functions]
    values = np.empty((len(smiles), len(names)), dtype=np.float64)
    values.fill(np.nan)
    status_counts: dict[str, int] = {}
    for row, smi in enumerate(smiles):
        status = "ok"
        try:
            capped = Chem.MolFromSmiles(cap_polymer_smiles(str(smi)), sanitize=True)
            if capped is None:
                raise RuntimeError("capped parse failed")
            mol = Chem.AddHs(capped)
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed + row)
            params.useRandomCoords = True
            embed_code = int(AllChem.EmbedMolecule(mol, params))
            if embed_code != 0:
                status = f"embed_code_{embed_code}"
                raise RuntimeError(status)
            if optimize_steps > 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=int(optimize_steps))
                except Exception:
                    status = "uff_failed_descriptor_attempted"
            for col, (_, function) in enumerate(descriptor_functions):
                try:
                    value = float(function(mol))
                except Exception:
                    value = math.nan
                values[row, col] = value if math.isfinite(value) else math.nan
        except Exception:
            pass
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "source": "official train/test SMILES only; dummy endpoints capped with carbon before deterministic RDKit ETKDG",
        "seed": int(seed),
        "optimize_steps": int(optimize_steps),
        "descriptor_count": len(names),
        "status_counts": status_counts,
        "nonfinite_by_column": {
            name: int(np.count_nonzero(~np.isfinite(values[:, index]))) for index, name in enumerate(names)
        },
    }
    return values, names, report


def parse_int_csv(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        parsed = value
    elif isinstance(value, list):
        parsed = tuple(int(item) for item in value)
    else:
        parsed = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def parse_token_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        parsed = value
    elif isinstance(value, list):
        parsed = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        parsed = tuple(part.strip().lower() for part in str(value).split(",") if part.strip())
    if not parsed:
        raise ValueError("expected at least one token")
    return parsed


def rdkit_3d_descriptor_names(include_extended: bool) -> list[str]:
    names = [
        "rdkit3d_Asphericity",
        "rdkit3d_Eccentricity",
        "rdkit3d_InertialShapeFactor",
        "rdkit3d_NPR1",
        "rdkit3d_NPR2",
        "rdkit3d_PBF",
        "rdkit3d_PMI1",
        "rdkit3d_PMI2",
        "rdkit3d_PMI3",
        "rdkit3d_RadiusOfGyration",
        "rdkit3d_SpherocityIndex",
    ]
    if include_extended:
        names.extend([f"rdkit3d_WHIM_{index:03d}" for index in range(114)])
        names.extend([f"rdkit3d_GETAWAY_{index:03d}" for index in range(273)])
        names.extend([f"rdkit3d_MORSE_{index:03d}" for index in range(224)])
        names.extend([f"rdkit3d_RDF_{index:03d}" for index in range(210)])
        names.extend([f"rdkit3d_AUTOCORR3D_{index:03d}" for index in range(80)])
    return names


def rdkit_3d_values_for_conformer(mol: Chem.Mol, conf_id: int, *, include_extended: bool) -> np.ndarray:
    values: list[float] = []
    scalar_functions = [
        Descriptors3D.Asphericity,
        Descriptors3D.Eccentricity,
        Descriptors3D.InertialShapeFactor,
        Descriptors3D.NPR1,
        Descriptors3D.NPR2,
        Descriptors3D.PBF,
        Descriptors3D.PMI1,
        Descriptors3D.PMI2,
        Descriptors3D.PMI3,
        Descriptors3D.RadiusOfGyration,
        Descriptors3D.SpherocityIndex,
    ]
    for function in scalar_functions:
        try:
            value = float(function(mol, confId=int(conf_id)))
        except Exception:
            value = math.nan
        values.append(value if math.isfinite(value) else math.nan)
    if include_extended:
        vector_functions = [
            rdMolDescriptors.CalcWHIM,
            rdMolDescriptors.CalcGETAWAY,
            rdMolDescriptors.CalcMORSE,
            rdMolDescriptors.CalcRDF,
            rdMolDescriptors.CalcAUTOCORR3D,
        ]
        for function in vector_functions:
            try:
                vector = [float(item) for item in function(mol, confId=int(conf_id))]
            except Exception:
                vector = []
            values.extend([item if math.isfinite(item) else math.nan for item in vector])
    expected = len(rdkit_3d_descriptor_names(include_extended))
    if len(values) < expected:
        values.extend([math.nan] * (expected - len(values)))
    elif len(values) > expected:
        values = values[:expected]
    return np.asarray(values, dtype=np.float64)


def pool_conformer_descriptors(matrix: np.ndarray, poolings: tuple[str, ...]) -> np.ndarray:
    finite = np.isfinite(matrix)
    count = finite.sum(axis=0).astype(np.float64)
    pieces: list[np.ndarray] = []
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.divide(
            np.where(finite, matrix, 0.0).sum(axis=0),
            count,
            out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
            where=count > 0,
        )
    for pooling in poolings:
        if pooling == "mean":
            pooled = mean
        elif pooling == "std":
            diff = np.where(finite, matrix - mean[None, :], 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                pooled = np.sqrt(
                    np.divide(
                        np.square(diff).sum(axis=0),
                        count,
                        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
                        where=count > 0,
                    )
                )
        elif pooling == "min":
            pooled = np.where(count > 0, np.min(np.where(finite, matrix, np.inf), axis=0), np.nan)
        elif pooling == "max":
            pooled = np.where(count > 0, np.max(np.where(finite, matrix, -np.inf), axis=0), np.nan)
        else:
            raise ValueError(f"unknown conformer pooling {pooling!r}")
        pieces.append(np.asarray(pooled, dtype=np.float64))
    return np.concatenate(pieces)


def pooled_3d_descriptor_for_mol(
    mol: Chem.Mol,
    *,
    seed: int,
    conformers: int,
    optimize_steps: int,
    poolings: tuple[str, ...],
    include_extended: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    descriptor_names = rdkit_3d_descriptor_names(include_extended)
    out = np.full(len(descriptor_names) * len(poolings), np.nan, dtype=np.float64)
    report: dict[str, Any] = {
        "status": "ok",
        "requested_conformers": int(conformers),
        "embedded_conformers": 0,
        "descriptor_conformers": 0,
        "uff_failures": 0,
    }
    try:
        working = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(working, numConfs=int(conformers), params=params))
        report["embedded_conformers"] = int(len(conf_ids))
        if not conf_ids:
            report["status"] = "embed_failed"
            return out, report
        rows: list[np.ndarray] = []
        for conf_id in conf_ids:
            if optimize_steps > 0:
                try:
                    AllChem.UFFOptimizeMolecule(working, confId=int(conf_id), maxIters=int(optimize_steps))
                except Exception:
                    report["uff_failures"] = int(report["uff_failures"]) + 1
            row = rdkit_3d_values_for_conformer(working, int(conf_id), include_extended=include_extended)
            if np.isfinite(row).any():
                rows.append(row)
        report["descriptor_conformers"] = int(len(rows))
        if not rows:
            report["status"] = "no_descriptor_conformer"
            return out, report
        out = pool_conformer_descriptors(np.vstack(rows), poolings)
        if not np.isfinite(out).any():
            report["status"] = "all_nonfinite"
        return out, report
    except Exception as exc:
        report["status"] = f"failed_{type(exc).__name__}"
        return out, report


def physics_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    names = [
        "gasteiger_min",
        "gasteiger_max",
        "gasteiger_mean",
        "gasteiger_std",
        "gasteiger_abs_mean",
        "gasteiger_abs_max",
        "formal_charge_sum",
        "formal_charge_abs_sum",
        "radical_electron_sum",
        "avg_total_valence",
        "avg_total_degree",
        "sp_atom_fraction",
        "sp2_atom_fraction",
        "sp3_atom_fraction",
        "conjugated_bond_fraction",
        "aromatic_bond_fraction",
        "single_bond_fraction",
        "double_bond_fraction",
        "triple_bond_fraction",
        "endpoint_neighbor_atomic_min",
        "endpoint_neighbor_atomic_max",
        "endpoint_neighbor_aromatic_sum",
        "endpoint_neighbor_ring_sum",
        "endpoint_neighbor_degree_sum",
        "endpoint_path_length",
        "endpoint_direct_bond_present",
    ]
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    for row, mol in enumerate(mols):
        local = Chem.Mol(mol)
        atoms = list(local.GetAtoms())
        bonds = list(local.GetBonds())
        n_atoms = max(len(atoms), 1)
        n_bonds = max(len(bonds), 1)
        try:
            AllChem.ComputeGasteigerCharges(local)
            charges = []
            for atom in local.GetAtoms():
                charge = float(atom.GetProp("_GasteigerCharge"))
                charges.append(charge if math.isfinite(charge) else 0.0)
            charge_arr = np.asarray(charges, dtype=np.float64)
        except Exception:
            charge_arr = np.zeros(len(atoms), dtype=np.float64)
        if charge_arr.size:
            values[row, 0] = float(np.min(charge_arr))
            values[row, 1] = float(np.max(charge_arr))
            values[row, 2] = float(np.mean(charge_arr))
            values[row, 3] = float(np.std(charge_arr))
            values[row, 4] = float(np.mean(np.abs(charge_arr)))
            values[row, 5] = float(np.max(np.abs(charge_arr)))
        values[row, 6] = float(sum(atom.GetFormalCharge() for atom in atoms))
        values[row, 7] = float(sum(abs(atom.GetFormalCharge()) for atom in atoms))
        values[row, 8] = float(sum(atom.GetNumRadicalElectrons() for atom in atoms))
        values[row, 9] = float(sum(atom.GetTotalValence() for atom in atoms)) / n_atoms
        values[row, 10] = float(sum(atom.GetTotalDegree() for atom in atoms)) / n_atoms
        values[row, 11] = float(sum(atom.GetHybridization() == Chem.HybridizationType.SP for atom in atoms)) / n_atoms
        values[row, 12] = float(sum(atom.GetHybridization() == Chem.HybridizationType.SP2 for atom in atoms)) / n_atoms
        values[row, 13] = float(sum(atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in atoms)) / n_atoms
        values[row, 14] = float(sum(bond.GetIsConjugated() for bond in bonds)) / n_bonds
        values[row, 15] = float(sum(bond.GetBondType() == Chem.BondType.AROMATIC for bond in bonds)) / n_bonds
        values[row, 16] = float(sum(bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds)) / n_bonds
        values[row, 17] = float(sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds)) / n_bonds
        values[row, 18] = float(sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds)) / n_bonds

        endpoint_neighbors = []
        endpoint_bond_types = []
        for atom in atoms:
            if atom.GetAtomicNum() != 0:
                continue
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                continue
            neighbor = neighbors[0]
            bond = local.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            endpoint_neighbors.append(neighbor)
            if bond is not None:
                endpoint_bond_types.append(bond.GetBondType())
        if len(endpoint_neighbors) == 2:
            atomic_nums = sorted(neighbor.GetAtomicNum() for neighbor in endpoint_neighbors)
            values[row, 19] = float(atomic_nums[0])
            values[row, 20] = float(atomic_nums[1])
            values[row, 21] = float(sum(neighbor.GetIsAromatic() for neighbor in endpoint_neighbors))
            values[row, 22] = float(sum(neighbor.IsInRing() for neighbor in endpoint_neighbors))
            values[row, 23] = float(sum(neighbor.GetTotalDegree() for neighbor in endpoint_neighbors))
            try:
                path = Chem.rdmolops.GetShortestPath(local, endpoint_neighbors[0].GetIdx(), endpoint_neighbors[1].GetIdx())
                values[row, 24] = float(max(0, len(path) - 1))
            except Exception:
                values[row, 24] = 0.0
            values[row, 25] = float(local.GetBondBetweenAtoms(endpoint_neighbors[0].GetIdx(), endpoint_neighbors[1].GetIdx()) is not None)
    return values, names


LOWGAP_SMARTS = (
    ("cyano", "[CX2]#N"),
    ("imide", "[NX3]([CX3](=O))[CX3](=O)"),
    ("sulfone", "[SX4](=O)(=O)"),
    ("sulfoxide", "[SX3](=O)"),
    ("quinone_like", "[#6]1(=O)[#6]=[#6][#6](=O)[#6]=[#6]1"),
    ("thiophene", "[s]1[c][c][c][c]1"),
    ("thiazole", "[s]1[c][n][c][c]1"),
    ("oxadiazole", "[o]1[n][c][n][c]1"),
    ("triazine", "[n]1[c][n][c][n][c]1"),
    ("acceptor_carbonyl_aromatic", "[c][CX3](=O)[#6,#7,#8]"),
    ("vinylene", "[CX3]=[CX3]"),
    ("ethynylene", "[CX2]#[CX2]"),
)


def electronic_tail_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    compiled = [(name, Chem.MolFromSmarts(pattern)) for name, pattern in LOWGAP_SMARTS]
    compiled = [(name, pattern) for name, pattern in compiled if pattern is not None]
    base_names = [
        "electronic_lowgap_total_count",
        "electronic_acceptor_total_count",
        "electronic_acceptor_per_heavy",
        "electronic_path_acceptor_count",
        "electronic_path_acceptor_fraction",
        "electronic_path_da_alternations",
        "electronic_path_longest_acceptor_run",
        "electronic_path_longest_donor_run",
        "electronic_path_conj_acceptor_fraction",
        "electronic_endpoint_acceptor_asymmetry",
    ]
    smarts_names: list[str] = []
    for name, _ in compiled:
        smarts_names.extend([f"electronic_smarts_{name}_count", f"electronic_smarts_{name}_per_heavy"])
    names = base_names + smarts_names
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}

    for row, mol in enumerate(mols):
        atoms = list(mol.GetAtoms())
        heavy = max(sum(atom.GetAtomicNum() > 1 for atom in atoms), 1)
        atom_acceptor_score = np.zeros(len(atoms), dtype=np.float64)
        atom_donor_score = np.zeros(len(atoms), dtype=np.float64)
        atom_conj_score = np.zeros(len(atoms), dtype=np.float64)
        for atom in atoms:
            idx = atom.GetIdx()
            anum = atom.GetAtomicNum()
            is_acceptor = anum in (7, 8, 16) and atom.GetFormalCharge() <= 0
            is_acceptor = is_acceptor or anum in (9, 17, 35, 53)
            is_acceptor = is_acceptor or (anum == 6 and any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in atom.GetBonds()))
            atom_acceptor_score[idx] = float(is_acceptor)
            atom_donor_score[idx] = float(anum in (7, 8, 16) and atom.GetTotalNumHs() > 0)
            atom_conj_score[idx] = float(atom.GetIsAromatic() or any(bond.GetIsConjugated() for bond in atom.GetBonds()))

        offset = len(base_names)
        total_count = 0.0
        for name, pattern in compiled:
            try:
                matches = mol.GetSubstructMatches(pattern, uniquify=True)
                count = float(len(matches))
            except Exception:
                matches = ()
                count = 0.0
            total_count += count
            if matches:
                for match in matches:
                    for atom_idx in match:
                        if 0 <= int(atom_idx) < len(atom_acceptor_score):
                            atom_acceptor_score[int(atom_idx)] = 1.0
            values[row, offset] = count
            values[row, offset + 1] = count / heavy
            offset += 2

        values[row, 0] = total_count
        values[row, 1] = float(np.sum(atom_acceptor_score))
        values[row, 2] = float(np.sum(atom_acceptor_score)) / heavy

        _, endpoint_path = endpoint_neighbors_and_path(mol)
        if endpoint_path:
            path = [int(idx) for idx in endpoint_path]
            path_acceptor = atom_acceptor_score[path]
            path_donor = atom_donor_score[path]
            path_conj = atom_conj_score[path]
            path_len = max(len(path), 1)
            values[row, 3] = float(np.sum(path_acceptor))
            values[row, 4] = float(np.mean(path_acceptor))
            tokens = np.where(path_acceptor > 0, 1, np.where(path_donor > 0, -1, 0))
            alternations = 0
            last = 0
            for token in tokens:
                current = int(token)
                if current == 0:
                    continue
                if last != 0 and current != last:
                    alternations += 1
                last = current
            values[row, 5] = float(alternations)
            for token_value, col in ((1, 6), (-1, 7)):
                best_run = 0
                current_run = 0
                for token in tokens:
                    if int(token) == token_value:
                        current_run += 1
                        best_run = max(best_run, current_run)
                    else:
                        current_run = 0
                values[row, col] = float(best_run)
            values[row, 8] = float(np.sum((path_acceptor > 0) & (path_conj > 0))) / path_len
            left = float(np.sum(path_acceptor[: min(3, len(path_acceptor))]))
            right = float(np.sum(path_acceptor[max(0, len(path_acceptor) - 3) :]))
            values[row, 9] = abs(left - right)
        status_counts["ok"] = status_counts.get("ok", 0) + 1

    report = {
        "source": "official train/test SMILES only; explicit low-bandgap acceptor motifs and ordered donor/acceptor endpoint-path signatures",
        "smarts_patterns": [name for name, _ in compiled],
        "dense_feature_count": len(names),
        "status_counts": status_counts,
    }
    return values, names, report


def huckel_spectrum_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Topology-only pi-system spectral descriptors for optical/electronic gap signal."""

    names = [
        "huckel_pi_atom_count",
        "huckel_pi_atom_fraction",
        "huckel_pi_edge_count",
        "huckel_pi_edge_fraction",
        "huckel_component_count",
        "huckel_largest_component_fraction",
        "huckel_gap_mid",
        "huckel_gap_mid_norm",
        "huckel_periodic_gap_mid",
        "huckel_periodic_gap_mid_norm",
        "huckel_gap_delta_periodic",
        "huckel_spectral_radius",
        "huckel_periodic_spectral_radius",
        "huckel_bandwidth",
        "huckel_periodic_bandwidth",
        "huckel_center_level_density",
        "huckel_periodic_center_level_density",
        "huckel_lowest_abs_eigen",
        "huckel_periodic_lowest_abs_eigen",
        "huckel_hetero_pi_fraction",
        "huckel_halogen_neighbor_pi_fraction",
        "huckel_carbonyl_neighbor_pi_fraction",
        "huckel_endpoint_pi_fraction",
        "huckel_endpoint_periodic_edge",
        "huckel_endpoint_path_pi_fraction",
        "huckel_endpoint_path_pi_run",
        "huckel_endpoint_path_bond_order_mean",
        "huckel_endpoint_path_bond_order_std",
        "huckel_pi_avg_degree",
        "huckel_pi_max_degree",
        "huckel_pi_branch_fraction",
        "huckel_electronegativity_mean",
        "huckel_electronegativity_std",
        "huckel_diag_perturb_mean",
        "huckel_diag_perturb_std",
    ]
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    electronegativity = {5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
    carbonyl = Chem.MolFromSmarts("[CX3]=[OX1]")

    def is_pi_atom(atom: Chem.Atom) -> bool:
        if atom.GetAtomicNum() <= 1:
            return False
        if atom.GetIsAromatic():
            return True
        if atom.GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2):
            return True
        return any(
            bond.GetIsConjugated()
            or bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
            for bond in atom.GetBonds()
        )

    def bond_weight(bond: Chem.Bond) -> float:
        btype = bond.GetBondType()
        if btype == Chem.BondType.AROMATIC:
            return 1.35
        if btype == Chem.BondType.DOUBLE:
            return 1.60
        if btype == Chem.BondType.TRIPLE:
            return 1.90
        if bond.GetIsConjugated():
            return 1.00
        return 0.80

    def spectrum_stats(matrix: np.ndarray) -> dict[str, float]:
        if matrix.shape[0] < 2:
            return {
                "gap": 0.0,
                "gap_norm": 0.0,
                "radius": 0.0,
                "bandwidth": 0.0,
                "center_density": 0.0,
                "lowest_abs": 0.0,
            }
        try:
            eig = np.linalg.eigvalsh(matrix)
        except np.linalg.LinAlgError:
            eig = np.linalg.eigvals(matrix).real
            eig.sort()
        n = len(eig)
        mid = n // 2
        if n % 2 == 0:
            gap = float(eig[mid] - eig[mid - 1])
        else:
            gap = float(min(eig[mid] - eig[mid - 1], eig[mid + 1] - eig[mid]) if 0 < mid < n - 1 else 0.0)
        bandwidth = float(eig[-1] - eig[0]) if n else 0.0
        window = max(0.5, 0.1 * bandwidth)
        return {
            "gap": abs(gap),
            "gap_norm": safe_ratio(abs(gap), bandwidth),
            "radius": float(np.max(np.abs(eig))) if n else 0.0,
            "bandwidth": bandwidth,
            "center_density": safe_ratio(float(np.count_nonzero(np.abs(eig) <= window)), float(n)),
            "lowest_abs": float(np.min(np.abs(eig))) if n else 0.0,
        }

    for row, mol in enumerate(mols):
        status = "ok"
        try:
            atoms = list(mol.GetAtoms())
            heavy = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() > 1}
            heavy_count = max(len(heavy), 1)
            pi_atoms = sorted(int(atom.GetIdx()) for atom in atoms if is_pi_atom(atom))
            pi_set = set(pi_atoms)
            local_index = {atom_idx: pos for pos, atom_idx in enumerate(pi_atoms)}
            n_pi = len(pi_atoms)
            adjacency = np.zeros((n_pi, n_pi), dtype=np.float64)
            pi_edge_count = 0
            pi_degrees = np.zeros(n_pi, dtype=np.float64)
            pi_bond_weights: list[float] = []
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtomIdx())
                b = int(bond.GetEndAtomIdx())
                if a not in pi_set or b not in pi_set:
                    continue
                weight = bond_weight(bond)
                ia = local_index[a]
                ib = local_index[b]
                adjacency[ia, ib] = weight
                adjacency[ib, ia] = weight
                pi_degrees[ia] += 1.0
                pi_degrees[ib] += 1.0
                pi_edge_count += 1
                pi_bond_weights.append(weight)

            diag = np.zeros(n_pi, dtype=np.float64)
            en_values = []
            for atom_idx, pos in local_index.items():
                atom = mol.GetAtomWithIdx(atom_idx)
                en = electronegativity.get(atom.GetAtomicNum(), 2.55)
                en_values.append(en)
                diag[pos] = 0.35 * (en - 2.55)
            huckel = adjacency + np.diag(diag)

            endpoint_neighbors, endpoint_path = endpoint_neighbors_and_path(mol)
            periodic = np.array(huckel, copy=True)
            endpoint_periodic_edge = 0.0
            if len(endpoint_neighbors) == 2 and endpoint_neighbors[0] in pi_set and endpoint_neighbors[1] in pi_set:
                left = local_index[int(endpoint_neighbors[0])]
                right = local_index[int(endpoint_neighbors[1])]
                if left != right:
                    periodic[left, right] = max(periodic[left, right], 1.0)
                    periodic[right, left] = max(periodic[right, left], 1.0)
                    endpoint_periodic_edge = 1.0

            component_sizes: list[int] = []
            seen: set[int] = set()
            graph_adj: dict[int, set[int]] = {idx: set() for idx in pi_atoms}
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtomIdx())
                b = int(bond.GetEndAtomIdx())
                if a in pi_set and b in pi_set:
                    graph_adj[a].add(b)
                    graph_adj[b].add(a)
            for start in pi_atoms:
                if start in seen:
                    continue
                stack = [start]
                seen.add(start)
                size = 0
                while stack:
                    cur = stack.pop()
                    size += 1
                    for nxt in graph_adj.get(cur, ()):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                component_sizes.append(size)

            stats = spectrum_stats(huckel)
            periodic_stats = spectrum_stats(periodic)
            hetero_pi = [idx for idx in pi_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() not in (6,)]
            halogen_neighbor_pi = [
                idx
                for idx in pi_atoms
                if any(neighbor.GetAtomicNum() in (9, 17, 35, 53) for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors())
            ]
            carbonyl_atoms = {
                int(idx)
                for match in (mol.GetSubstructMatches(carbonyl) if carbonyl is not None else ())
                for idx in match
            }
            carbonyl_neighbor_pi = [
                idx
                for idx in pi_atoms
                if idx in carbonyl_atoms or any(int(neighbor.GetIdx()) in carbonyl_atoms for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors())
            ]
            endpoint_pi = [idx for idx in endpoint_neighbors if idx in pi_set]
            path_atoms = [int(idx) for idx in endpoint_path if int(idx) in heavy] if endpoint_path else []
            path_pi_flags = [int(idx in pi_set) for idx in path_atoms]
            best_run = 0
            current_run = 0
            for flag in path_pi_flags:
                if flag:
                    current_run += 1
                    best_run = max(best_run, current_run)
                else:
                    current_run = 0
            path_weights = []
            if endpoint_path:
                for a, b in zip(endpoint_path[:-1], endpoint_path[1:], strict=True):
                    bond = mol.GetBondBetweenAtoms(int(a), int(b))
                    if bond is not None:
                        path_weights.append(bond_weight(bond))
            en_arr = np.asarray(en_values, dtype=np.float64)
            values[row] = np.asarray(
                [
                    n_pi,
                    safe_ratio(n_pi, heavy_count),
                    pi_edge_count,
                    safe_ratio(pi_edge_count, max(len(mol.GetBonds()), 1)),
                    len(component_sizes),
                    safe_ratio(max(component_sizes) if component_sizes else 0, max(n_pi, 1)),
                    stats["gap"],
                    stats["gap_norm"],
                    periodic_stats["gap"],
                    periodic_stats["gap_norm"],
                    periodic_stats["gap"] - stats["gap"],
                    stats["radius"],
                    periodic_stats["radius"],
                    stats["bandwidth"],
                    periodic_stats["bandwidth"],
                    stats["center_density"],
                    periodic_stats["center_density"],
                    stats["lowest_abs"],
                    periodic_stats["lowest_abs"],
                    safe_ratio(len(hetero_pi), n_pi),
                    safe_ratio(len(halogen_neighbor_pi), n_pi),
                    safe_ratio(len(carbonyl_neighbor_pi), n_pi),
                    safe_ratio(len(endpoint_pi), max(len(endpoint_neighbors), 1)),
                    endpoint_periodic_edge,
                    safe_ratio(sum(path_pi_flags), max(len(path_pi_flags), 1)),
                    best_run,
                    float(np.mean(path_weights)) if path_weights else 0.0,
                    float(np.std(path_weights)) if path_weights else 0.0,
                    float(np.mean(pi_degrees)) if len(pi_degrees) else 0.0,
                    float(np.max(pi_degrees)) if len(pi_degrees) else 0.0,
                    safe_ratio(float(np.count_nonzero(pi_degrees > 2.0)), n_pi),
                    float(np.mean(en_arr)) if len(en_arr) else 0.0,
                    float(np.std(en_arr)) if len(en_arr) else 0.0,
                    float(np.mean(diag)) if len(diag) else 0.0,
                    float(np.std(diag)) if len(diag) else 0.0,
                ],
                dtype=np.float64,
            )
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
            values[row] = np.nan
        status_counts[status] = status_counts.get(status, 0) + 1

    return values, names, {
        "source": "official train/test SMILES only; Huckel-style weighted pi-graph spectrum and periodic endpoint closure proxies",
        "descriptor_count": len(names),
        "status_counts": status_counts,
        "nonfinite_values": int(np.count_nonzero(~np.isfinite(values))),
        "external_data_training_use": False,
        "pretrained_model_use": False,
    }


def topological_autocorr_feature_matrix(mols: list[Chem.Mol], max_distance: int = 8) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    properties = ("charge", "atomic_num", "hetero", "aromatic", "acceptor", "donor")
    names: list[str] = []
    for distance in range(1, int(max_distance) + 1):
        for prop in properties:
            names.extend(
                [
                    f"topo_autocorr_d{distance}_{prop}_prod_mean",
                    f"topo_autocorr_d{distance}_{prop}_absdiff_mean",
                ]
            )
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        local = Chem.Mol(mol)
        atoms = list(local.GetAtoms())
        if not atoms:
            status_counts["empty"] = status_counts.get("empty", 0) + 1
            continue
        try:
            AllChem.ComputeGasteigerCharges(local)
        except Exception:
            pass
        prop_values: dict[str, np.ndarray] = {}
        charges = []
        for atom in local.GetAtoms():
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                charge = 0.0
            charges.append(charge if math.isfinite(charge) else 0.0)
        prop_values["charge"] = np.asarray(charges, dtype=np.float64)
        prop_values["atomic_num"] = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=np.float64)
        prop_values["hetero"] = np.asarray([atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms], dtype=np.float64)
        prop_values["aromatic"] = np.asarray([atom.GetIsAromatic() for atom in atoms], dtype=np.float64)
        prop_values["acceptor"] = np.asarray(
            [
                (atom.GetAtomicNum() in (7, 8, 16) and atom.GetFormalCharge() <= 0)
                or atom.GetAtomicNum() in (9, 17, 35, 53)
                for atom in atoms
            ],
            dtype=np.float64,
        )
        prop_values["donor"] = np.asarray(
            [atom.GetAtomicNum() in (7, 8, 16) and atom.GetTotalNumHs() > 0 for atom in atoms],
            dtype=np.float64,
        )
        try:
            distances = Chem.rdmolops.GetDistanceMatrix(local).astype(np.int64)
        except Exception:
            status_counts["distance_failed"] = status_counts.get("distance_failed", 0) + 1
            continue
        col = 0
        for distance in range(1, int(max_distance) + 1):
            pairs = np.argwhere(np.triu(distances == distance, k=1))
            for prop in properties:
                arr = prop_values[prop]
                if pairs.size:
                    left = arr[pairs[:, 0]]
                    right = arr[pairs[:, 1]]
                    values[row, col] = float(np.mean(left * right))
                    values[row, col + 1] = float(np.mean(np.abs(left - right)))
                col += 2
        status_counts["ok"] = status_counts.get("ok", 0) + 1
    report = {
        "source": "official train/test SMILES only; graph-distance autocorrelations over charges and electronic atom flags",
        "max_distance": int(max_distance),
        "dense_feature_count": len(names),
        "status_counts": status_counts,
    }
    return values, names, report


def infinite_chain_proxy_feature_matrix(smiles: list[str], mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = [
        "icd_heavy_atoms",
        "icd_exact_mass",
        "icd_bond_count",
        "icd_endpoint_path_bonds",
        "icd_endpoint_path_heavy_atoms",
        "icd_side_heavy_atoms",
        "icd_side_heavy_fraction",
        "icd_mass_per_path_bond",
        "icd_heavy_atoms_per_path_bond",
        "icd_side_heavy_per_path_bond",
        "icd_side_mass_per_path_bond",
        "icd_side_to_backbone_mass_ratio",
        "icd_backbone_aromatic_fraction",
        "icd_backbone_conjugated_bond_fraction",
        "icd_backbone_rotatable_fraction",
        "icd_backbone_sp2sp_fraction",
        "icd_backbone_hetero_fraction",
        "icd_backbone_halogen_fraction",
        "icd_side_aromatic_fraction",
        "icd_side_hetero_fraction",
        "icd_side_halogen_fraction",
        "icd_aromatic_atoms_per_path_bond",
        "icd_conjugated_bonds_per_path_bond",
        "icd_ring_count_per_path_bond",
        "icd_fused_ring_pairs_per_path_bond",
        "icd_acceptor_atoms_per_path_bond",
        "icd_donor_atoms_per_path_bond",
        "icd_lowgap_motifs_per_path_bond",
        "icd_polar_atoms_per_exact_mass",
        "icd_halogen_atoms_per_exact_mass",
        "icd_aromatic_atoms_per_exact_mass",
        "icd_conjugated_bonds_per_exact_mass",
        "icd_periodic_heavy_atoms",
        "icd_periodic_ring_count",
        "icd_periodic_ring_count_delta",
        "icd_periodic_aromatic_fraction",
        "icd_periodic_conjugated_bond_fraction",
        "icd_periodic_rotatable_fraction",
        "icd_periodic_mean_topological_distance",
        "icd_periodic_max_topological_distance",
    ]
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    lowgap_patterns = [pattern for _, pattern in ((name, Chem.MolFromSmarts(smarts)) for name, smarts in LOWGAP_SMARTS) if pattern is not None]
    status_counts: dict[str, int] = {}
    for row, (smi, mol) in enumerate(zip(smiles, mols, strict=True)):
        status = "ok"
        try:
            atoms = list(mol.GetAtoms())
            bonds = list(mol.GetBonds())
            heavy_indices = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() > 1}
            heavy_count = max(len(heavy_indices), 1)
            exact_mass = float(
                sum(atom.GetMass() for atom in atoms if atom.GetAtomicNum() > 1)
            )
            mass_denom = max(exact_mass, 1e-9)
            bond_count = max(len(bonds), 1)
            _, endpoint_path = endpoint_neighbors_and_path(mol)
            path_set = set(int(idx) for idx in endpoint_path)
            path_heavy = {idx for idx in path_set if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() > 1}
            side_heavy = heavy_indices.difference(path_set)
            path_bonds: list[Chem.Bond] = []
            for left, right in zip(endpoint_path[:-1], endpoint_path[1:], strict=True):
                bond = mol.GetBondBetweenAtoms(int(left), int(right))
                if bond is not None:
                    path_bonds.append(bond)
            path_bond_denom = max(len(path_bonds), 1)
            backbone_mass = float(sum(mol.GetAtomWithIdx(int(idx)).GetMass() for idx in path_heavy))
            side_mass = float(sum(mol.GetAtomWithIdx(int(idx)).GetMass() for idx in side_heavy))
            aromatic_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetIsAromatic()}
            hetero_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6)}
            halogen_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53)}
            acceptor_atoms = {
                int(atom.GetIdx())
                for atom in atoms
                if (atom.GetAtomicNum() in (7, 8, 16) and atom.GetFormalCharge() <= 0)
                or atom.GetAtomicNum() in (9, 17, 35, 53)
            }
            donor_atoms = {
                int(atom.GetIdx())
                for atom in atoms
                if atom.GetAtomicNum() in (7, 8, 16) and atom.GetTotalNumHs() > 0
            }
            lowgap_count = 0
            for pattern in lowgap_patterns:
                try:
                    lowgap_count += len(mol.GetSubstructMatches(pattern, uniquify=True))
                except Exception:
                    pass
            rotatable_indices = region_rotatable_bond_indices(mol)
            sp2sp_bonds = [
                bond
                for bond in path_bonds
                if bond.GetBeginAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)
                and bond.GetEndAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)
            ]
            ring_info = mol.GetRingInfo()
            atom_rings = [set(ring) for ring in ring_info.AtomRings()]
            fused_pairs = 0
            for i, ring_i in enumerate(atom_rings):
                for ring_j in atom_rings[i + 1 :]:
                    fused_pairs += int(bool(ring_i.intersection(ring_j)))
            periodic = periodic_closure_mol(str(smi), mol)
            periodic_atoms = [atom for atom in periodic.GetAtoms() if atom.GetAtomicNum() > 1]
            periodic_bonds = list(periodic.GetBonds())
            periodic_bond_denom = max(len(periodic_bonds), 1)
            periodic_heavy_denom = max(len(periodic_atoms), 1)
            try:
                distances = Chem.GetDistanceMatrix(periodic)
                tri = distances[np.triu_indices_from(distances, k=1)]
                tri = tri[np.isfinite(tri) & (tri > 0)]
                mean_distance = float(np.mean(tri)) if tri.size else 0.0
                max_distance = float(np.max(tri)) if tri.size else 0.0
            except Exception:
                mean_distance = 0.0
                max_distance = 0.0

            values[row] = np.asarray(
                [
                    float(heavy_count),
                    exact_mass,
                    float(bond_count),
                    float(len(path_bonds)),
                    float(len(path_heavy)),
                    float(len(side_heavy)),
                    safe_ratio(len(side_heavy), heavy_count),
                    exact_mass / path_bond_denom,
                    heavy_count / path_bond_denom,
                    len(side_heavy) / path_bond_denom,
                    side_mass / path_bond_denom,
                    safe_ratio(side_mass, backbone_mass),
                    safe_ratio(sum(idx in aromatic_atoms for idx in path_heavy), len(path_heavy)),
                    safe_ratio(sum(bond.GetIsConjugated() for bond in path_bonds), len(path_bonds)),
                    safe_ratio(sum(int(bond.GetIdx()) in rotatable_indices for bond in path_bonds), len(path_bonds)),
                    safe_ratio(len(sp2sp_bonds), len(path_bonds)),
                    safe_ratio(sum(idx in hetero_atoms for idx in path_heavy), len(path_heavy)),
                    safe_ratio(sum(idx in halogen_atoms for idx in path_heavy), len(path_heavy)),
                    safe_ratio(sum(idx in aromatic_atoms for idx in side_heavy), len(side_heavy)),
                    safe_ratio(sum(idx in hetero_atoms for idx in side_heavy), len(side_heavy)),
                    safe_ratio(sum(idx in halogen_atoms for idx in side_heavy), len(side_heavy)),
                    len(aromatic_atoms) / path_bond_denom,
                    sum(bond.GetIsConjugated() for bond in bonds) / path_bond_denom,
                    ring_info.NumRings() / path_bond_denom,
                    fused_pairs / path_bond_denom,
                    sum(idx in acceptor_atoms for idx in heavy_indices) / path_bond_denom,
                    sum(idx in donor_atoms for idx in heavy_indices) / path_bond_denom,
                    lowgap_count / path_bond_denom,
                    len(hetero_atoms) / mass_denom,
                    len(halogen_atoms) / mass_denom,
                    len(aromatic_atoms) / mass_denom,
                    sum(bond.GetIsConjugated() for bond in bonds) / mass_denom,
                    float(len(periodic_atoms)),
                    float(periodic.GetRingInfo().NumRings()),
                    float(periodic.GetRingInfo().NumRings() - ring_info.NumRings()),
                    safe_ratio(sum(atom.GetIsAromatic() for atom in periodic_atoms), periodic_heavy_denom),
                    safe_ratio(sum(bond.GetIsConjugated() for bond in periodic_bonds), periodic_bond_denom),
                    safe_ratio(
                        sum(int(bond.GetIdx()) in region_rotatable_bond_indices(periodic) for bond in periodic_bonds),
                        len(periodic_bonds),
                    ),
                    mean_distance,
                    max_distance,
                ],
                dtype=np.float64,
            )
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
            values[row] = np.nan
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "source": "official train/test SMILES only; compact infinite-chain proxy ratios over repeat-core mass, endpoint backbone path, sidechain bulk, electronic density, and periodic closure graph",
        "dense_feature_count": len(names),
        "status_counts": status_counts,
        "nonfinite_values": int(np.count_nonzero(~np.isfinite(values))),
    }
    return values, names, report


BICERANO_COLUMNS = (
    "density_at_298_k_gcm3",
    "density_at_t_gcm3",
    "molar_volume_cm3mol",
    "ecoh1_at_298_k_jmol",
    "solub_ratio",
    "fh_parameter",
    "tg_k",
    "temperature_of_half_decomposition_k",
    "bulk_modulus_mpa",
    "youngs_modulus_mpa",
    "shear_modulus_mpa",
    "brittle_fracture_stress_mpa",
    "tensile_yield_stress_mpa",
    "number_hydrogen_bonding",
    "charge_of_counterion",
    "permeability_co2_barrer",
    "permeability_n2_barrer",
    "permeability_o2_barrer",
    "selectivity_co2_n2",
    "selectivity_o2_n2",
)


def _raise_bicerano_timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("bicerano row calculation timed out")


def opsin_endpoint_markers(smiles: str) -> str:
    parts: list[str] = []
    marker_index = 0
    for char in str(smiles):
        if char == "*":
            marker_index += 1
            parts.append(f"[*:{marker_index}]")
        else:
            parts.append(char)
    if marker_index != 2:
        raise RuntimeError(f"expected exactly two polymer endpoints, found {marker_index}")
    return "".join(parts)


def bicerano_feature_matrix(smiles: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not BICERANO_AVAILABLE or bicerano_ppf is None:
        raise RuntimeError(
            "polymer_property_prediction is required for --bicerano-features. "
            "Install public code only, for example: pip install --no-deps polymer_property_prediction"
        )
    names = [f"bicerano_{column}" for column in BICERANO_COLUMNS]
    values = np.empty((len(smiles), len(names)), dtype=np.float64)
    values.fill(np.nan)
    status_counts: dict[str, int] = {}
    for row, smi in enumerate(smiles):
        status = "ok"
        try:
            obj = SimpleNamespace(
                name=f"row_{row}",
                smiles=opsin_endpoint_markers(str(smi)),
                temperature=298.0,
                pressure=101325.0,
                polymer_concentration_wt=1.0,
                Mn=10000.0,
            )
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_bicerano_timeout)
            signal.setitimer(signal.ITIMER_REAL, 3.0)
            blocker = rdBase.BlockLogs()
            try:
                result = bicerano_ppf.calculateMol(obj)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
                del blocker
            record = result.iloc[0].to_dict()
            for col, column in enumerate(BICERANO_COLUMNS):
                parsed = read_float(record.get(column))
                values[row, col] = parsed if parsed is not None else math.nan
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "source": (
            "public IBM polymer_property_prediction Bicerano-style group-contribution formulae; "
            "computed from official train/test SMILES only after adapting '*' endpoints to [*:1]/[*:2]"
        ),
        "descriptor_count": len(names),
        "status_counts": status_counts,
        "nonfinite_by_column": {
            name: int(np.count_nonzero(~np.isfinite(values[:, index]))) for index, name in enumerate(names)
        },
        "external_data_training_use": False,
        "pretrained_model_use": False,
    }
    return values, names, report


def conjugation_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = [
        "conj_component_count",
        "largest_conj_atom_count",
        "largest_conj_atom_fraction",
        "largest_conj_bond_count",
        "largest_conj_bond_fraction",
        "largest_conj_hetero_fraction",
        "largest_conj_aromatic_atom_fraction",
        "largest_conj_mean_atomic_num",
        "largest_conj_path_len",
        "conj_atom_count",
        "conj_atom_fraction",
        "conj_bond_count",
        "conj_bond_fraction",
        "aromatic_bond_count",
        "aromatic_bond_fraction",
        "nonaromatic_conj_bond_count",
        "nonaromatic_conj_bond_fraction",
        "sp2sp_bond_count",
        "sp2sp_bond_fraction",
        "hetero_conj_atom_count",
        "hetero_conj_atom_fraction",
        "donor_atom_count",
        "acceptor_atom_count",
        "donor_acceptor_pair_count",
        "donor_acceptor_min_path",
        "donor_acceptor_mean_path",
        "donor_acceptor_conj_path_fraction",
        "carbonyl_count",
        "carbonyl_aromatic_neighbor_count",
        "nitrile_count_electronic",
        "imine_count",
        "azo_count_electronic",
        "halogen_on_conj_count",
        "dummy_neighbor_conj_count",
        "dummy_neighbor_aromatic_count",
        "dummy_neighbor_unsat_count",
        "ring_count_conj",
        "aromatic_ring_count_conj",
        "fused_ring_pair_count",
        "largest_fused_ring_component",
        "aromatic_atom_count_conj",
        "aromatic_atom_fraction_conj",
        "aliphatic_unsat_bond_count",
        "hetero_to_carbon_conj_ratio",
    ]
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    carbonyl = Chem.MolFromSmarts("[CX3]=[OX1]")
    nitrile = Chem.MolFromSmarts("[CX2]#N")
    imine = Chem.MolFromSmarts("[CX3]=[NX2,NX3]")
    azo = Chem.MolFromSmarts("[NX2]=[NX2]")
    status_counts: dict[str, int] = {}

    for row, mol in enumerate(mols):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        heavy_atoms = [atom for atom in atoms if atom.GetAtomicNum() > 1]
        heavy_count = max(len(heavy_atoms), 1)
        bond_count = max(len(bonds), 1)
        conj_bonds = [bond for bond in bonds if bond.GetIsConjugated() or bond.GetBondType() == Chem.BondType.AROMATIC]
        conj_bond_indices = {bond.GetIdx() for bond in conj_bonds}
        conj_atoms = sorted({idx for bond in conj_bonds for idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())})
        conj_atom_set = set(conj_atoms)
        aromatic_atoms = {atom.GetIdx() for atom in atoms if atom.GetIsAromatic()}
        hetero_atoms = {atom.GetIdx() for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6)}

        adjacency: dict[int, set[int]] = {idx: set() for idx in conj_atoms}
        for bond in conj_bonds:
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        components: list[set[int]] = []
        seen: set[int] = set()
        for start in conj_atoms:
            if start in seen:
                continue
            stack = [start]
            comp: set[int] = set()
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nxt in adjacency.get(cur, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            components.append(comp)
        largest = max(components, key=len, default=set())
        largest_bonds = [
            bond
            for bond in conj_bonds
            if bond.GetBeginAtomIdx() in largest and bond.GetEndAtomIdx() in largest
        ]

        def farthest_from(start: int, component: set[int]) -> tuple[int, int]:
            queue = [(start, 0)]
            local_seen = {start}
            farthest = (start, 0)
            for cur, dist in queue:
                if dist > farthest[1]:
                    farthest = (cur, dist)
                for nxt in adjacency.get(cur, ()):
                    if nxt in component and nxt not in local_seen:
                        local_seen.add(nxt)
                        queue.append((nxt, dist + 1))
            return farthest

        def approximate_longest_path_length(component: set[int]) -> int:
            if len(component) < 2:
                return 0
            first = min(component)
            edge, _ = farthest_from(first, component)
            _, diameter = farthest_from(edge, component)
            return int(diameter)

        donors = []
        acceptors = []
        for atom in atoms:
            anum = atom.GetAtomicNum()
            if anum in (7, 8, 16) and atom.GetTotalNumHs() > 0:
                donors.append(atom.GetIdx())
            if anum in (7, 8, 16) and atom.GetFormalCharge() <= 0:
                acceptors.append(atom.GetIdx())
        da_paths = []
        da_conj_paths = 0
        da_pairs = [(donor, acceptor) for donor in donors for acceptor in acceptors if donor != acceptor]
        if len(da_pairs) > 64:
            step = max(1, len(da_pairs) // 64)
            da_pairs = da_pairs[::step][:64]
        for donor, acceptor in da_pairs:
            try:
                path = list(Chem.rdmolops.GetShortestPath(mol, donor, acceptor))
            except Exception:
                continue
            if len(path) < 2:
                continue
            length = len(path) - 1
            da_paths.append(length)
            all_conj = True
            for a, b in zip(path[:-1], path[1:], strict=True):
                bond = mol.GetBondBetweenAtoms(int(a), int(b))
                if bond is None or bond.GetIdx() not in conj_bond_indices:
                    all_conj = False
                    break
            da_conj_paths += int(all_conj)

        ring_info = mol.GetRingInfo()
        atom_rings = [set(ring) for ring in ring_info.AtomRings()]
        aromatic_ring_count = 0
        for ring in atom_rings:
            if ring and all(mol.GetAtomWithIdx(int(idx)).GetIsAromatic() for idx in ring):
                aromatic_ring_count += 1
        fused_edges: dict[int, set[int]] = {idx: set() for idx in range(len(atom_rings))}
        fused_pairs = 0
        for i, ring_i in enumerate(atom_rings):
            for j in range(i + 1, len(atom_rings)):
                if ring_i.intersection(atom_rings[j]):
                    fused_pairs += 1
                    fused_edges[i].add(j)
                    fused_edges[j].add(i)
        largest_fused = 0
        ring_seen: set[int] = set()
        for start in fused_edges:
            if start in ring_seen:
                continue
            stack = [start]
            ring_seen.add(start)
            size = 0
            while stack:
                cur = stack.pop()
                size += 1
                for nxt in fused_edges[cur]:
                    if nxt not in ring_seen:
                        ring_seen.add(nxt)
                        stack.append(nxt)
            largest_fused = max(largest_fused, size)

        carbonyl_matches = mol.GetSubstructMatches(carbonyl) if carbonyl is not None else ()
        carbonyl_aromatic = 0
        for match in carbonyl_matches:
            c_idx = int(match[0])
            atom = mol.GetAtomWithIdx(c_idx)
            if any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()):
                carbonyl_aromatic += 1

        dummy_neighbor_conj = 0
        dummy_neighbor_aromatic = 0
        dummy_neighbor_unsat = 0
        for atom in atoms:
            if atom.GetAtomicNum() != 0:
                continue
            for neighbor in atom.GetNeighbors():
                nidx = neighbor.GetIdx()
                dummy_neighbor_conj += int(nidx in conj_atom_set)
                dummy_neighbor_aromatic += int(neighbor.GetIsAromatic())
                connecting = mol.GetBondBetweenAtoms(atom.GetIdx(), nidx)
                if connecting is not None:
                    dummy_neighbor_unsat += int(connecting.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC))

        conj_heavy = [idx for idx in conj_atoms if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() > 1]
        conj_carbon_count = sum(1 for idx in conj_heavy if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() == 6)
        hetero_conj_count = sum(1 for idx in conj_heavy if idx in hetero_atoms)
        sp2sp_bonds = [
            bond
            for bond in bonds
            if bond.GetBeginAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)
            and bond.GetEndAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)
        ]
        aliphatic_unsat = [
            bond
            for bond in bonds
            if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
            and not bond.GetBeginAtom().GetIsAromatic()
            and not bond.GetEndAtom().GetIsAromatic()
        ]

        values[row] = np.asarray(
            [
                len(components),
                len(largest),
                len(largest) / heavy_count,
                len(largest_bonds),
                len(largest_bonds) / bond_count,
                safe_ratio(sum(1 for idx in largest if idx in hetero_atoms), len(largest)),
                safe_ratio(sum(1 for idx in largest if idx in aromatic_atoms), len(largest)),
                safe_ratio(sum(mol.GetAtomWithIdx(int(idx)).GetAtomicNum() for idx in largest), len(largest)),
                approximate_longest_path_length(largest),
                len(conj_heavy),
                len(conj_heavy) / heavy_count,
                len(conj_bonds),
                len(conj_bonds) / bond_count,
                sum(1 for bond in bonds if bond.GetBondType() == Chem.BondType.AROMATIC),
                sum(1 for bond in bonds if bond.GetBondType() == Chem.BondType.AROMATIC) / bond_count,
                sum(1 for bond in conj_bonds if bond.GetBondType() != Chem.BondType.AROMATIC),
                sum(1 for bond in conj_bonds if bond.GetBondType() != Chem.BondType.AROMATIC) / bond_count,
                len(sp2sp_bonds),
                len(sp2sp_bonds) / bond_count,
                hetero_conj_count,
                hetero_conj_count / heavy_count,
                len(donors),
                len(acceptors),
                len(da_paths),
                min(da_paths) if da_paths else 0,
                float(np.mean(da_paths)) if da_paths else 0.0,
                safe_ratio(da_conj_paths, len(da_paths)),
                len(carbonyl_matches),
                carbonyl_aromatic,
                len(mol.GetSubstructMatches(nitrile)) if nitrile is not None else 0,
                len(mol.GetSubstructMatches(imine)) if imine is not None else 0,
                len(mol.GetSubstructMatches(azo)) if azo is not None else 0,
                sum(1 for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53) and any(n.GetIdx() in conj_atom_set for n in atom.GetNeighbors())),
                dummy_neighbor_conj,
                dummy_neighbor_aromatic,
                dummy_neighbor_unsat,
                len(atom_rings),
                aromatic_ring_count,
                fused_pairs,
                largest_fused,
                len(aromatic_atoms),
                len(aromatic_atoms) / heavy_count,
                len(aliphatic_unsat),
                safe_ratio(hetero_conj_count, conj_carbon_count),
            ],
            dtype=np.float64,
        )
        status_counts["ok"] = status_counts.get("ok", 0) + 1

    return values, names, {
        "source": "official train/test SMILES only; conjugation/electronic topology descriptors from RDKit graph, no external data",
        "descriptor_count": len(names),
        "status_counts": status_counts,
    }


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def mobility_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Effective-mobility descriptors inspired by conjugated-polymer Tg work."""

    names = [
        "mobility_zeta_mean_a06",
        "mobility_zeta_mass_weighted_a06",
        "mobility_side_atom_fraction",
        "mobility_side_mass_fraction",
        "mobility_backbone_atom_fraction",
        "mobility_backbone_mass_fraction",
        "mobility_flexible_side_atom_fraction",
        "mobility_rigid_atom_fraction",
        "mobility_aromatic_rigid_atom_fraction",
        "mobility_thiophene_atom_fraction",
        "mobility_phenyl_atom_fraction",
        "mobility_carbonyl_atom_fraction",
        "mobility_alkenyl_atom_fraction",
        "mobility_side_alkyl_atom_fraction",
        "mobility_side_hetero_atom_fraction",
        "mobility_side_rotatable_per_heavy",
        "mobility_bb_rotatable_per_heavy",
        "mobility_side_to_bb_mass_ratio",
        "mobility_flexible_to_rigid_ratio",
        "mobility_fox_inv_proxy_a06",
        "mobility_backbone_stiffness_proxy_a06",
        "mobility_internal_plasticization_proxy",
        "mobility_rigid_ring_count",
        "mobility_side_component_count",
        "mobility_longest_side_component_fraction",
        "mobility_linear_alkyl_tail_max",
        "mobility_linear_alkyl_tail_mean",
        "mobility_endpoint_path_aromatic_fraction",
        "mobility_endpoint_path_sp2_fraction",
        "mobility_endpoint_path_hetero_fraction",
    ]
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    carbonyl = Chem.MolFromSmarts("[CX3]=[OX1]")
    thiophene = Chem.MolFromSmarts("[s]1[c][c][c][c]1")
    phenyl = Chem.MolFromSmarts("c1ccccc1")

    def side_components(mol: Chem.Mol, side_heavy: set[int]) -> list[set[int]]:
        components: list[set[int]] = []
        seen: set[int] = set()
        for start in sorted(side_heavy):
            if start in seen:
                continue
            comp = {start}
            seen.add(start)
            queue = [start]
            cursor = 0
            while cursor < len(queue):
                current = queue[cursor]
                cursor += 1
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    idx = int(neighbor.GetIdx())
                    if idx not in side_heavy or idx in seen:
                        continue
                    seen.add(idx)
                    comp.add(idx)
                    queue.append(idx)
            components.append(comp)
        return components

    def longest_alkyl_tail(mol: Chem.Mol, comp: set[int]) -> int:
        alkyl = {
            idx
            for idx in comp
            if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
            and not mol.GetAtomWithIdx(idx).GetIsAromatic()
            and mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3
        }
        if not alkyl:
            return 0
        best = 1
        for start in alkyl:
            stack = [(start, 1, {start})]
            while stack:
                current, length, path_seen = stack.pop()
                best = max(best, length)
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    idx = int(neighbor.GetIdx())
                    if idx not in alkyl or idx in path_seen:
                        continue
                    stack.append((idx, length + 1, path_seen | {idx}))
        return best

    for row, mol in enumerate(mols):
        status = "ok"
        try:
            dummy_atoms = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError(f"expected exactly two dummy atoms, observed {len(dummy_atoms)}")
            path = tuple(int(idx) for idx in Chem.rdmolops.GetShortestPath(mol, dummy_atoms[0], dummy_atoms[1]))
            if not path:
                raise RuntimeError("dummy endpoints are disconnected")
            path_set = set(path)
            atoms = list(mol.GetAtoms())
            heavy_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() > 1}
            heavy_count = max(len(heavy_atoms), 1)
            backbone_heavy = {idx for idx in path_set if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
            side_heavy = heavy_atoms.difference(path_set)
            masses = {idx: float(mol.GetAtomWithIdx(idx).GetMass()) for idx in heavy_atoms}
            total_mass = sum(masses.values())
            backbone_mass = sum(masses[idx] for idx in backbone_heavy)
            side_mass = sum(masses[idx] for idx in side_heavy)
            aromatic_atoms = {idx for idx in heavy_atoms if mol.GetAtomWithIdx(idx).GetIsAromatic()}
            hetero_side = {idx for idx in side_heavy if mol.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6)}
            carbonyl_atoms = {
                int(idx)
                for match in (mol.GetSubstructMatches(carbonyl) if carbonyl is not None else ())
                for idx in match
                if int(idx) in heavy_atoms
            }
            thiophene_atoms = {
                int(idx)
                for match in (mol.GetSubstructMatches(thiophene) if thiophene is not None else ())
                for idx in match
                if int(idx) in heavy_atoms
            }
            phenyl_atoms = {
                int(idx)
                for match in (mol.GetSubstructMatches(phenyl) if phenyl is not None else ())
                for idx in match
                if int(idx) in heavy_atoms
            }
            alkenyl_atoms = {
                int(bond.GetBeginAtomIdx())
                for bond in mol.GetBonds()
                if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
            } | {
                int(bond.GetEndAtomIdx())
                for bond in mol.GetBonds()
                if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
            }
            side_alkyl_atoms = {
                idx
                for idx in side_heavy
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
                and not mol.GetAtomWithIdx(idx).GetIsAromatic()
                and mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3
            }
            flexible_atoms = set(side_alkyl_atoms)
            rigid_atoms = heavy_atoms.difference(flexible_atoms)
            mobility = {}
            for idx in heavy_atoms:
                if idx in flexible_atoms:
                    mobility[idx] = 1.0
                elif idx in thiophene_atoms:
                    mobility[idx] = 0.72
                else:
                    mobility[idx] = 0.60
            zeta_mean = sum(mobility.values()) / heavy_count
            zeta_mass = safe_ratio(sum(mobility[idx] * masses[idx] for idx in heavy_atoms), total_mass)
            rotatable = region_rotatable_bond_indices(mol)
            bb_bonds: set[int] = set()
            side_bonds: set[int] = set()
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtomIdx())
                b = int(bond.GetEndAtomIdx())
                if a in path_set and b in path_set:
                    bb_bonds.add(int(bond.GetIdx()))
                elif a in side_heavy and b in side_heavy:
                    side_bonds.add(int(bond.GetIdx()))
            components = side_components(mol, side_heavy)
            comp_sizes = [len(comp) for comp in components]
            alkyl_lengths = [longest_alkyl_tail(mol, comp) for comp in components]
            path_heavy = [idx for idx in path if idx in heavy_atoms]
            path_denom = max(len(path_heavy), 1)
            fox_inv_proxy = safe_ratio(side_mass, total_mass) * 1.0 + safe_ratio(backbone_mass, total_mass) * safe_ratio(1.0, max(zeta_mean, 1e-9))
            values[row] = np.asarray(
                [
                    zeta_mean,
                    zeta_mass,
                    safe_ratio(len(side_heavy), heavy_count),
                    safe_ratio(side_mass, total_mass),
                    safe_ratio(len(backbone_heavy), heavy_count),
                    safe_ratio(backbone_mass, total_mass),
                    safe_ratio(len(flexible_atoms), heavy_count),
                    safe_ratio(len(rigid_atoms), heavy_count),
                    safe_ratio(len(aromatic_atoms & rigid_atoms), heavy_count),
                    safe_ratio(len(thiophene_atoms), heavy_count),
                    safe_ratio(len(phenyl_atoms), heavy_count),
                    safe_ratio(len(carbonyl_atoms), heavy_count),
                    safe_ratio(len(alkenyl_atoms & heavy_atoms), heavy_count),
                    safe_ratio(len(side_alkyl_atoms), heavy_count),
                    safe_ratio(len(hetero_side), heavy_count),
                    safe_ratio(sum(idx in rotatable for idx in side_bonds), heavy_count),
                    safe_ratio(sum(idx in rotatable for idx in bb_bonds), heavy_count),
                    safe_ratio(side_mass, backbone_mass),
                    safe_ratio(len(flexible_atoms), len(rigid_atoms)),
                    fox_inv_proxy,
                    safe_ratio(1.0 - zeta_mean, max(safe_ratio(backbone_mass, total_mass), 1e-9)),
                    safe_ratio(side_mass, total_mass) * safe_ratio(len(flexible_atoms), len(side_heavy)),
                    float(sum(1 for ring in mol.GetRingInfo().AtomRings() if any(int(idx) in rigid_atoms for idx in ring))),
                    float(len(components)),
                    safe_ratio(max(comp_sizes) if comp_sizes else 0, len(side_heavy)),
                    float(max(alkyl_lengths) if alkyl_lengths else 0),
                    float(np.mean(alkyl_lengths)) if alkyl_lengths else 0.0,
                    safe_ratio(sum(idx in aromatic_atoms for idx in path_heavy), path_denom),
                    safe_ratio(
                        sum(
                            mol.GetAtomWithIdx(idx).GetHybridization()
                            in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)
                            for idx in path_heavy
                        ),
                        path_denom,
                    ),
                    safe_ratio(sum(mol.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6) for idx in path_heavy), path_denom),
                ],
                dtype=np.float64,
            )
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
            values[row] = np.nan
        status_counts[status] = status_counts.get(status, 0) + 1

    return values, names, {
        "source": (
            "official train/test SMILES only; effective atomic mobility, side-chain mass fraction, "
            "and rigid/flexible backbone proxies inspired by conjugated-polymer Tg literature"
        ),
        "descriptor_count": len(names),
        "status_counts": status_counts,
        "nonfinite_values": int(np.count_nonzero(~np.isfinite(values))),
        "external_data_training_use": False,
        "pretrained_model_use": False,
    }


def region_rotatable_bond_indices(mol: Chem.Mol) -> set[int]:
    out: set[int] = set()
    for match in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts):
        if len(match) >= 2:
            bond = mol.GetBondBetweenAtoms(int(match[0]), int(match[1]))
            if bond is not None:
                out.add(int(bond.GetIdx()))
    return out


def region_atom_counts(mol: Chem.Mol, atom_indices: set[int]) -> list[float]:
    atoms = [mol.GetAtomWithIdx(int(idx)) for idx in atom_indices]
    denom = sum(1 for atom in atoms if atom.GetAtomicNum() > 1)
    values: list[float] = []
    for atomic_num in REGION_ATOM_NUMBERS:
        count = sum(1 for atom in atoms if atom.GetAtomicNum() == atomic_num)
        values.append(float(count))
        values.append(safe_ratio(count, denom))
    return values


def region_charge_stats(mol: Chem.Mol, atom_indices: set[int]) -> list[float]:
    charges: list[float] = []
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(idx))
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            charge = math.nan
        if math.isfinite(charge):
            charges.append(charge)
    if not charges:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    arr = np.asarray(charges, dtype=np.float64)
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.sum(np.maximum(arr, 0.0))),
        float(np.sum(np.minimum(arr, 0.0))),
    ]


def region_fragment_descriptor_values(fragment: Chem.Mol | None) -> list[float]:
    if fragment is None or fragment.GetNumAtoms() == 0:
        return [0.0] * len(REGION_DESCRIPTOR_NAMES)
    values: list[float] = []
    for name in REGION_DESCRIPTOR_NAMES:
        func = getattr(Descriptors, name)
        try:
            value = float(func(fragment))
        except Exception:
            value = math.nan
        values.append(value if math.isfinite(value) else math.nan)
    return values


def region_fragment_from_atoms(mol: Chem.Mol, atom_indices: set[int]) -> Chem.Mol | None:
    if not atom_indices:
        return None
    try:
        fragment_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(int(index) for index in atom_indices), canonical=True)
        return Chem.MolFromSmiles(fragment_smiles, sanitize=True) if fragment_smiles else None
    except Exception:
        return None


def side_distance_stats(mol: Chem.Mol, side_indices: set[int], backbone_indices: set[int]) -> list[float]:
    if not side_indices or not backbone_indices:
        return [0.0, 0.0, 0.0, 0.0]
    distances = {int(idx): 0 for idx in backbone_indices}
    queue = list(distances)
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = int(neighbor.GetIdx())
            if idx in distances:
                continue
            distances[idx] = distances[current] + 1
            queue.append(idx)
    vals = np.asarray([distances.get(int(idx), 0) for idx in side_indices], dtype=np.float64)
    if vals.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(vals.max()), float(vals.mean()), float(np.median(vals)), float(vals.std())]


def side_component_stats(mol: Chem.Mol, side_indices: set[int], path: tuple[int, ...]) -> list[float]:
    if not side_indices:
        return [0.0] * 26
    side_set = {int(idx) for idx in side_indices}
    path_set = {int(idx) for idx in path}
    path_pos = {int(idx): pos for pos, idx in enumerate(path)}
    components: list[set[int]] = []
    seen: set[int] = set()
    for start in sorted(side_set):
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        queue = [start]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                idx = int(neighbor.GetIdx())
                if idx not in side_set or idx in seen:
                    continue
                seen.add(idx)
                comp.add(idx)
                queue.append(idx)
        components.append(comp)

    comp_sizes = np.asarray([len(comp) for comp in components], dtype=np.float64)
    comp_masses = np.asarray(
        [sum(float(mol.GetAtomWithIdx(idx).GetMass()) for idx in comp) for comp in components],
        dtype=np.float64,
    )
    comp_attachment_counts: list[float] = []
    comp_attachment_positions: list[float] = []
    for comp in components:
        positions: list[int] = []
        attachment_count = 0
        for idx in comp:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = int(neighbor.GetIdx())
                if n_idx in path_set:
                    attachment_count += 1
                    if n_idx in path_pos:
                        positions.append(path_pos[n_idx])
        comp_attachment_counts.append(float(attachment_count))
        comp_attachment_positions.append(float(np.mean(positions)) if positions else 0.0)

    attachment_counts = np.asarray(comp_attachment_counts, dtype=np.float64)
    attachment_positions = np.asarray(comp_attachment_positions, dtype=np.float64)
    terminal_count = sum(
        1
        for idx in side_set
        if sum(int(neighbor.GetIdx()) in side_set for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()) <= 1
    )
    side_total = max(float(len(side_set)), 1.0)
    size_fraction = comp_sizes / side_total
    diversity = 1.0 - float(np.sum(size_fraction * size_fraction))
    max_component_fraction = float(np.max(size_fraction)) if size_fraction.size else 0.0

    endpoint_distances: list[float] = []
    distance_ratios: list[float] = []
    dummy_atoms = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) == 2:
        try:
            distances = Chem.GetDistanceMatrix(mol)
            for idx in side_set:
                nearest = float(min(distances[idx, dummy_atoms[0]], distances[idx, dummy_atoms[1]]))
                endpoint_distances.append(nearest)
                distance_ratios.append(safe_ratio(1.0, nearest))
        except Exception:
            endpoint_distances = []
            distance_ratios = []
    endpoint_arr = np.asarray(endpoint_distances, dtype=np.float64)
    ratio_arr = np.asarray(distance_ratios, dtype=np.float64)

    def stats(vals: np.ndarray) -> list[float]:
        if vals.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), float(np.std(vals))]

    return [
        float(len(components)),
        *stats(comp_sizes),
        *stats(comp_masses),
        *stats(attachment_counts),
        float(terminal_count),
        safe_ratio(float(terminal_count), len(side_set)),
        diversity,
        max_component_fraction,
        *stats(attachment_positions),
        *stats(endpoint_arr),
        float(np.mean(ratio_arr)) if ratio_arr.size else 0.0,
    ]


def endpoint_environment_values(mol: Chem.Mol, dummy_atoms: list[int]) -> list[float]:
    values: list[float] = []
    for idx in sorted(dummy_atoms):
        atom = mol.GetAtomWithIdx(int(idx))
        neighbors = list(atom.GetNeighbors())
        neighbor = neighbors[0] if neighbors else None
        if neighbor is None:
            values.extend([0.0, 0.0, 0.0, 0.0])
            continue
        bond = mol.GetBondBetweenAtoms(int(idx), int(neighbor.GetIdx()))
        values.extend(
            [
                float(neighbor.GetAtomicNum()),
                float(neighbor.GetDegree()),
                float(neighbor.GetIsAromatic()),
                float(bond.GetBondTypeAsDouble()) if bond is not None else 0.0,
            ]
        )
    return values


def backbone_sidechain_feature_names() -> list[str]:
    names = [
        "bb_attachment_shortest_path_bonds",
        "bb_heavy_atom_count",
        "side_heavy_atom_count",
        "side_heavy_atom_ratio",
        "bb_bond_count",
        "side_bond_count",
        "bb_side_crossing_bond_count",
        "bb_rotatable_bond_count",
        "side_rotatable_bond_count",
        "bb_rotatable_bond_ratio",
        "side_rotatable_bond_ratio",
        "bb_aromatic_atom_count",
        "side_aromatic_atom_count",
        "bb_aromatic_atom_fraction",
        "side_aromatic_atom_fraction",
        "bb_hetero_atom_count",
        "side_hetero_atom_count",
        "bb_hetero_atom_fraction",
        "side_hetero_atom_fraction",
        "bb_ring_touch_count",
        "side_ring_touch_count",
        "side_max_distance_to_backbone",
        "side_mean_distance_to_backbone",
        "side_median_distance_to_backbone",
        "side_std_distance_to_backbone",
        "side_component_count",
        "side_component_heavy_min",
        "side_component_heavy_mean",
        "side_component_heavy_max",
        "side_component_heavy_std",
        "side_component_mass_min",
        "side_component_mass_mean",
        "side_component_mass_max",
        "side_component_mass_std",
        "side_component_attachment_min",
        "side_component_attachment_mean",
        "side_component_attachment_max",
        "side_component_attachment_std",
        "side_terminal_atom_count",
        "side_terminal_atom_fraction",
        "side_component_diversity",
        "side_max_component_fraction",
        "side_attachment_path_position_min",
        "side_attachment_path_position_mean",
        "side_attachment_path_position_max",
        "side_attachment_path_position_std",
        "side_distance_to_endpoint_min",
        "side_distance_to_endpoint_mean",
        "side_distance_to_endpoint_max",
        "side_distance_to_endpoint_std",
        "side_inverse_endpoint_distance_mean",
    ]
    for endpoint in ("a", "b"):
        names.extend(
            [
                f"endpoint_{endpoint}_neighbor_atomic_num",
                f"endpoint_{endpoint}_neighbor_degree",
                f"endpoint_{endpoint}_neighbor_is_aromatic",
                f"endpoint_{endpoint}_bond_order",
            ]
        )
    for region in ("bb", "side"):
        for atomic_num in REGION_ATOM_NUMBERS:
            names.append(f"{region}_atomic_num_{atomic_num}_count")
            names.append(f"{region}_atomic_num_{atomic_num}_fraction")
    for region in ("whole", "bb", "side"):
        names.extend(
            [
                f"{region}_gasteiger_mean",
                f"{region}_gasteiger_std",
                f"{region}_gasteiger_min",
                f"{region}_gasteiger_max",
                f"{region}_gasteiger_positive_sum",
                f"{region}_gasteiger_negative_sum",
            ]
        )
    for prefix in ("whole", "bb", "side"):
        names.extend([f"{prefix}_rdkit_{name}" for name in REGION_DESCRIPTOR_NAMES])
    for name in REGION_DESCRIPTOR_NAMES:
        names.append(f"side_over_whole_rdkit_{name}")
        names.append(f"bb_over_whole_rdkit_{name}")
    return names


def backbone_sidechain_values(mol: Chem.Mol) -> list[float]:
    work = Chem.Mol(mol)
    try:
        AllChem.ComputeGasteigerCharges(work, nIter=12, throwOnParamFailure=False)
    except Exception:
        pass
    dummy_atoms = [int(atom.GetIdx()) for atom in work.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 2:
        raise RuntimeError(f"expected exactly two dummy atoms, observed {len(dummy_atoms)}")
    path = list(Chem.rdmolops.GetShortestPath(work, int(dummy_atoms[0]), int(dummy_atoms[1])))
    if not path:
        raise RuntimeError("dummy endpoints are disconnected")
    path_set = set(int(idx) for idx in path)
    all_atoms = set(range(work.GetNumAtoms()))
    heavy_atoms = {int(atom.GetIdx()) for atom in work.GetAtoms() if atom.GetAtomicNum() > 1}
    backbone_heavy = {idx for idx in path_set if work.GetAtomWithIdx(idx).GetAtomicNum() > 1}
    side_heavy = heavy_atoms.difference(path_set)
    rotatable = region_rotatable_bond_indices(work)
    backbone_bonds: set[int] = set()
    side_bonds: set[int] = set()
    crossing_bonds = 0
    for bond in work.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        if a in path_set and b in path_set:
            backbone_bonds.add(int(bond.GetIdx()))
        elif a not in path_set and b not in path_set:
            side_bonds.add(int(bond.GetIdx()))
        else:
            crossing_bonds += 1
    aromatic_backbone = sum(1 for idx in backbone_heavy if work.GetAtomWithIdx(idx).GetIsAromatic())
    aromatic_side = sum(1 for idx in side_heavy if work.GetAtomWithIdx(idx).GetIsAromatic())
    hetero_backbone = sum(1 for idx in backbone_heavy if work.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6))
    hetero_side = sum(1 for idx in side_heavy if work.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6))
    rings = work.GetRingInfo().AtomRings()
    backbone_ring_touches = sum(1 for ring in rings if any(int(idx) in backbone_heavy for idx in ring))
    side_ring_touches = sum(1 for ring in rings if any(int(idx) in side_heavy for idx in ring))
    backbone_fragment = region_fragment_from_atoms(work, backbone_heavy)
    side_fragment = region_fragment_from_atoms(work, side_heavy)
    values = [
        float(len(path) - 1),
        float(len(backbone_heavy)),
        float(len(side_heavy)),
        safe_ratio(len(side_heavy), len(heavy_atoms)),
        float(len(backbone_bonds)),
        float(len(side_bonds)),
        float(crossing_bonds),
        float(sum(1 for idx in backbone_bonds if idx in rotatable)),
        float(sum(1 for idx in side_bonds if idx in rotatable)),
        safe_ratio(sum(1 for idx in backbone_bonds if idx in rotatable), len(backbone_bonds)),
        safe_ratio(sum(1 for idx in side_bonds if idx in rotatable), len(side_bonds)),
        float(aromatic_backbone),
        float(aromatic_side),
        safe_ratio(aromatic_backbone, len(backbone_heavy)),
        safe_ratio(aromatic_side, len(side_heavy)),
        float(hetero_backbone),
        float(hetero_side),
        safe_ratio(hetero_backbone, len(backbone_heavy)),
        safe_ratio(hetero_side, len(side_heavy)),
        float(backbone_ring_touches),
        float(side_ring_touches),
    ]
    values.extend(side_distance_stats(work, side_heavy, backbone_heavy))
    values.extend(side_component_stats(work, side_heavy, tuple(path)))
    values.extend(endpoint_environment_values(work, dummy_atoms))
    values.extend(region_atom_counts(work, backbone_heavy))
    values.extend(region_atom_counts(work, side_heavy))
    values.extend(region_charge_stats(work, heavy_atoms))
    values.extend(region_charge_stats(work, backbone_heavy))
    values.extend(region_charge_stats(work, side_heavy))
    whole_desc = region_fragment_descriptor_values(work)
    backbone_desc = region_fragment_descriptor_values(backbone_fragment)
    side_desc = region_fragment_descriptor_values(side_fragment)
    values.extend(whole_desc)
    values.extend(backbone_desc)
    values.extend(side_desc)
    for whole, back, side in zip(whole_desc, backbone_desc, side_desc, strict=True):
        values.append(safe_ratio(side, whole) if math.isfinite(side) and math.isfinite(whole) else math.nan)
        values.append(safe_ratio(back, whole) if math.isfinite(back) and math.isfinite(whole) else math.nan)
    return values


def backbone_sidechain_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = backbone_sidechain_feature_names()
    values = np.empty((len(mols), len(names)), dtype=np.float64)
    values.fill(np.nan)
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        status = "ok"
        try:
            row_values = backbone_sidechain_values(mol)
            if len(row_values) != len(names):
                raise RuntimeError(f"name/value mismatch {len(row_values)} != {len(names)}")
            values[row] = np.asarray(row_values, dtype=np.float64)
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
        status_counts[status] = status_counts.get(status, 0) + 1
    values[~np.isfinite(values)] = np.nan
    return values, [f"bb_side_{name}" for name in names], {
        "source": "official train/test SMILES only; shortest dummy-endpoint path treated as backbone and off-path heavy atoms as side-chain region",
        "descriptor_count": len(names),
        "status_counts": status_counts,
        "nonfinite_values": int(np.count_nonzero(~np.isfinite(values))),
    }


def atom_category(atom: Chem.Atom) -> str:
    if atom.GetAtomicNum() == 0:
        return "*"
    symbol = atom.GetSymbol()
    aromatic = "a" if atom.GetIsAromatic() else ""
    ring = "r" if atom.IsInRing() else ""
    hyb = str(atom.GetHybridization()).rsplit(".", 1)[-1]
    return f"{aromatic}{symbol}{ring}:{hyb}"


def bond_category(bond: Chem.Bond) -> str:
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        base = "arom"
    elif bond.GetBondType() == Chem.BondType.SINGLE:
        base = "single"
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
        base = "double"
    elif bond.GetBondType() == Chem.BondType.TRIPLE:
        base = "triple"
    else:
        base = str(bond.GetBondType()).lower()
    if bond.GetIsConjugated():
        base += ":conj"
    if bond.IsInRing():
        base += ":ring"
    return base


def stable_hash_index(token: str, n_features: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % int(n_features)


def endpoint_neighbors_and_path(mol: Chem.Mol) -> tuple[list[Chem.Atom], tuple[int, ...]]:
    endpoints: list[Chem.Atom] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) == 1:
            endpoints.append(neighbors[0])
    if len(endpoints) != 2:
        return endpoints, tuple()
    try:
        path = tuple(Chem.rdmolops.GetShortestPath(mol, endpoints[0].GetIdx(), endpoints[1].GetIdx()))
    except Exception:
        path = tuple()
    return endpoints, path


def motif_dense_matrix(mols: list[Chem.Mol], smiles: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    base_names = [
        "motif_heavy_atoms",
        "motif_total_atoms",
        "motif_total_bonds",
        "motif_ring_count",
        "motif_aromatic_ring_count",
        "motif_aliphatic_ring_count",
        "motif_saturated_ring_count",
        "motif_rotatable_bonds",
        "motif_hba",
        "motif_hbd",
        "motif_tpsa",
        "motif_labute_asa",
        "motif_fraction_csp3",
        "motif_aromatic_atom_fraction",
        "motif_ring_atom_fraction",
        "motif_hetero_atom_fraction",
        "motif_halogen_atom_fraction",
        "motif_o_atom_fraction",
        "motif_n_atom_fraction",
        "motif_s_atom_fraction",
        "motif_si_atom_fraction",
        "motif_single_bond_fraction",
        "motif_double_bond_fraction",
        "motif_triple_bond_fraction",
        "motif_aromatic_bond_fraction",
        "motif_conjugated_bond_fraction",
        "motif_endpoint_neighbor_same_atomic",
        "motif_endpoint_neighbor_mean_atomic",
        "motif_endpoint_neighbor_aromatic_fraction",
        "motif_endpoint_neighbor_ring_fraction",
        "motif_endpoint_path_length",
        "motif_endpoint_path_heavy_atoms",
        "motif_endpoint_path_aromatic_fraction",
        "motif_endpoint_path_ring_fraction",
        "motif_endpoint_path_hetero_fraction",
        "motif_endpoint_path_branch_off_count",
        "motif_endpoint_path_branch_off_fraction",
        "motif_endpoint_path_single_bond_fraction",
        "motif_endpoint_path_double_bond_fraction",
        "motif_endpoint_path_triple_bond_fraction",
        "motif_endpoint_path_aromatic_bond_fraction",
        "motif_endpoint_path_conjugated_bond_fraction",
        "motif_smiles_star_count",
        "motif_smiles_ring_digit_count",
        "motif_smiles_branch_count",
        "motif_smiles_aromatic_char_fraction",
    ]
    smarts_names: list[str] = []
    for name, _ in COMPILED_SMARTS:
        smarts_names.extend([f"motif_smarts_{name}_count", f"motif_smarts_{name}_per_heavy"])
    names = base_names + smarts_names
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}

    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        heavy = sum(1 for atom in atoms if atom.GetAtomicNum() > 1)
        total_atoms = len(atoms)
        total_bonds = len(bonds)
        heavy_denom = max(heavy, 1)
        bond_denom = max(total_bonds, 1)

        values[row, 0] = heavy
        values[row, 1] = total_atoms
        values[row, 2] = total_bonds
        values[row, 3] = mol.GetRingInfo().NumRings()
        values[row, 4] = rdMolDescriptors.CalcNumAromaticRings(mol)
        values[row, 5] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        values[row, 6] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        values[row, 7] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        values[row, 8] = rdMolDescriptors.CalcNumHBA(mol)
        values[row, 9] = rdMolDescriptors.CalcNumHBD(mol)
        try:
            values[row, 10] = rdMolDescriptors.CalcTPSA(mol)
        except Exception:
            values[row, 10] = 0.0
        try:
            values[row, 11] = rdMolDescriptors.CalcLabuteASA(mol)
        except Exception:
            values[row, 11] = 0.0
        try:
            values[row, 12] = rdMolDescriptors.CalcFractionCSP3(mol)
        except Exception:
            values[row, 12] = 0.0
        values[row, 13] = sum(atom.GetIsAromatic() for atom in atoms) / heavy_denom
        values[row, 14] = sum(atom.IsInRing() for atom in atoms if atom.GetAtomicNum() > 1) / heavy_denom
        values[row, 15] = sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms) / heavy_denom
        values[row, 16] = sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms) / heavy_denom
        values[row, 17] = sum(atom.GetAtomicNum() == 8 for atom in atoms) / heavy_denom
        values[row, 18] = sum(atom.GetAtomicNum() == 7 for atom in atoms) / heavy_denom
        values[row, 19] = sum(atom.GetAtomicNum() == 16 for atom in atoms) / heavy_denom
        values[row, 20] = sum(atom.GetAtomicNum() == 14 for atom in atoms) / heavy_denom
        values[row, 21] = sum(bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds) / bond_denom
        values[row, 22] = sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds) / bond_denom
        values[row, 23] = sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds) / bond_denom
        values[row, 24] = sum(bond.GetBondType() == Chem.BondType.AROMATIC for bond in bonds) / bond_denom
        values[row, 25] = sum(bond.GetIsConjugated() for bond in bonds) / bond_denom

        endpoint_neighbors, endpoint_path = endpoint_neighbors_and_path(mol)
        if len(endpoint_neighbors) == 2:
            atomic_values = [atom.GetAtomicNum() for atom in endpoint_neighbors]
            values[row, 26] = float(atomic_values[0] == atomic_values[1])
            values[row, 27] = float(np.mean(atomic_values))
            values[row, 28] = float(np.mean([atom.GetIsAromatic() for atom in endpoint_neighbors]))
            values[row, 29] = float(np.mean([atom.IsInRing() for atom in endpoint_neighbors]))
        if endpoint_path:
            path_atoms = [mol.GetAtomWithIdx(index) for index in endpoint_path]
            path_bonds = [
                mol.GetBondBetweenAtoms(endpoint_path[i], endpoint_path[i + 1])
                for i in range(len(endpoint_path) - 1)
            ]
            path_bonds = [bond for bond in path_bonds if bond is not None]
            path_atom_denom = max(len(path_atoms), 1)
            path_bond_denom = max(len(path_bonds), 1)
            branch_off = 0
            path_set = set(endpoint_path)
            for atom in path_atoms:
                branch_off += sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetIdx() not in path_set and neighbor.GetAtomicNum() != 0)
            values[row, 30] = max(0, len(endpoint_path) - 1)
            values[row, 31] = sum(atom.GetAtomicNum() > 1 for atom in path_atoms)
            values[row, 32] = sum(atom.GetIsAromatic() for atom in path_atoms) / path_atom_denom
            values[row, 33] = sum(atom.IsInRing() for atom in path_atoms) / path_atom_denom
            values[row, 34] = sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in path_atoms) / path_atom_denom
            values[row, 35] = branch_off
            values[row, 36] = branch_off / heavy_denom
            values[row, 37] = sum(bond.GetBondType() == Chem.BondType.SINGLE for bond in path_bonds) / path_bond_denom
            values[row, 38] = sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in path_bonds) / path_bond_denom
            values[row, 39] = sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in path_bonds) / path_bond_denom
            values[row, 40] = sum(bond.GetBondType() == Chem.BondType.AROMATIC for bond in path_bonds) / path_bond_denom
            values[row, 41] = sum(bond.GetIsConjugated() for bond in path_bonds) / path_bond_denom

        smi_text = str(smi)
        values[row, 42] = smi_text.count("*")
        values[row, 43] = sum(ch.isdigit() for ch in smi_text)
        values[row, 44] = smi_text.count("(")
        values[row, 45] = sum(ch in "bcnops" for ch in smi_text) / max(len(smi_text), 1)

        offset = len(base_names)
        for name, pattern in COMPILED_SMARTS:
            try:
                count = float(len(mol.GetSubstructMatches(pattern, uniquify=True)))
            except Exception:
                count = 0.0
            values[row, offset] = count
            values[row, offset + 1] = count / heavy_denom
            offset += 2
        status_counts["ok"] = status_counts.get("ok", 0) + 1

    report = {
        "source": "official train/test SMILES only; QSPR/GAP-inspired explicit motif counts, normalized chain descriptors, and endpoint-path descriptors",
        "dense_feature_count": len(names),
        "smarts_motif_count": len(COMPILED_SMARTS),
        "status_counts": status_counts,
    }
    return values, names, report


def motif_hash_matrix(mols: list[Chem.Mol], smiles: list[str], n_features: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    token_counts: dict[str, int] = {"smarts": 0, "brics": 0, "path": 0}

    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        for name, pattern in COMPILED_SMARTS:
            try:
                count = len(mol.GetSubstructMatches(pattern, uniquify=True))
            except Exception:
                count = 0
            if count:
                rows.append(row)
                cols.append(stable_hash_index(f"smarts:{name}", n_features))
                data.append(float(math.log1p(count)))
                token_counts["smarts"] += 1

        capped = Chem.MolFromSmiles(cap_polymer_smiles(str(smi)), sanitize=True)
        if capped is not None:
            try:
                fragments = BRICS.BRICSDecompose(capped, keepNonLeafNodes=True, minFragmentSize=2)
            except Exception:
                fragments = []
            for fragment in fragments:
                token = f"brics:{fragment}"
                rows.append(row)
                cols.append(stable_hash_index(token, n_features))
                data.append(1.0)
                token_counts["brics"] += 1

        seen_path_tokens: set[str] = set()
        for distance in range(1, 7):
            try:
                atom_paths = Chem.rdmolops.FindAllPathsOfLengthN(
                    mol,
                    distance,
                    useBonds=False,
                    useHs=False,
                    onlyShortestPaths=True,
                )
            except Exception:
                atom_paths = []
            for raw_path in atom_paths:
                path = tuple(int(index) for index in raw_path)
                if len(path) != distance + 1:
                    continue
                atom_tokens = [atom_category(mol.GetAtomWithIdx(index)) for index in path]
                bond_tokens: list[str] = []
                valid = True
                for index in range(distance):
                    bond = mol.GetBondBetweenAtoms(path[index], path[index + 1])
                    if bond is None:
                        valid = False
                        break
                    bond_tokens.append(bond_category(bond))
                if not valid:
                    continue
                forward = "|".join(sum(zip(atom_tokens, bond_tokens + [""]), ()))
                reverse_atom_tokens = list(reversed(atom_tokens))
                reverse_bond_tokens = list(reversed(bond_tokens))
                reverse = "|".join(sum(zip(reverse_atom_tokens, reverse_bond_tokens + [""]), ()))
                token = f"path{distance}:{min(forward, reverse)}"
                if token in seen_path_tokens:
                    continue
                seen_path_tokens.add(token)
                rows.append(row)
                cols.append(stable_hash_index(token, n_features))
                data.append(1.0 / distance)
                token_counts["path"] += 1

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), int(n_features)), dtype=np.float32)
    if matrix.nnz:
        matrix.sum_duplicates()
    report = {
        "source": "official train/test SMILES only; deterministic hashed SMARTS/BRICS/topological-path motif counts",
        "n_features": int(n_features),
        "nnz": int(matrix.nnz),
        "token_counts": token_counts,
    }
    return matrix, report


def atom_environment_token(mol: Chem.Mol, atom_index: int, radius: int) -> str:
    atom = mol.GetAtomWithIdx(int(atom_index))
    if radius <= 0:
        return atom_category(atom)
    try:
        bond_ids = list(Chem.rdmolops.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(atom_index)))
    except Exception:
        bond_ids = []
    if not bond_ids:
        return atom_category(atom)
    atom_ids: set[int] = {int(atom_index)}
    for bond_id in bond_ids:
        bond = mol.GetBondWithIdx(int(bond_id))
        atom_ids.add(int(bond.GetBeginAtomIdx()))
        atom_ids.add(int(bond.GetEndAtomIdx()))
    try:
        return Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=sorted(atom_ids),
            bondsToUse=sorted(int(bond_id) for bond_id in bond_ids),
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:
        return atom_category(atom)


def map4_like_matrix(
    mols: list[Chem.Mol],
    n_features: int,
    max_distance: int,
    env_radius: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    token_count = 0
    skipped_pair_count = 0
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        status = "ok"
        try:
            atom_indices = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
            env_tokens = {idx: atom_environment_token(mol, idx, env_radius) for idx in atom_indices}
            distances = Chem.GetDistanceMatrix(mol)
            local_counts: dict[int, float] = {}
            for pos_i, atom_i in enumerate(atom_indices):
                for atom_j in atom_indices[pos_i:]:
                    distance = int(distances[atom_i, atom_j])
                    if distance < 0 or distance > int(max_distance):
                        skipped_pair_count += 1
                        continue
                    left = env_tokens[atom_i]
                    right = env_tokens[atom_j]
                    if right < left:
                        left, right = right, left
                    token = f"map4r{env_radius}d{distance}:{left}::{right}"
                    col = stable_hash_index(token, n_features)
                    local_counts[col] = local_counts.get(col, 0.0) + 1.0
                    token_count += 1
            for col, count in local_counts.items():
                rows.append(row)
                cols.append(col)
                data.append(float(math.log1p(count)))
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
        status_counts[status] = status_counts.get(status, 0) + 1
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), int(n_features)), dtype=np.float32)
    if matrix.nnz:
        matrix.sum_duplicates()
    return matrix, {
        "source": "official train/test SMILES only; dependency-free MAP4-like hashed atom-environment pair fingerprint",
        "n_features": int(n_features),
        "max_distance": int(max_distance),
        "env_radius": int(env_radius),
        "nnz": int(matrix.nnz),
        "token_count": int(token_count),
        "skipped_pair_count": int(skipped_pair_count),
        "status_counts": status_counts,
    }


def endpoint_path_ngram_matrix(
    mols: list[Chem.Mol],
    *,
    n_features: int,
    max_bonds: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    token_count = 0
    status_counts: dict[str, int] = {}
    max_window = max(1, int(max_bonds))
    for row, mol in enumerate(mols):
        status = "ok"
        try:
            _, endpoint_path = endpoint_neighbors_and_path(mol)
            if len(endpoint_path) < 2:
                raise RuntimeError("endpoint_path_missing")
            atom_tokens = [atom_category(mol.GetAtomWithIdx(int(index))) for index in endpoint_path]
            bond_tokens: list[str] = []
            for left, right in zip(endpoint_path[:-1], endpoint_path[1:], strict=True):
                bond = mol.GetBondBetweenAtoms(int(left), int(right))
                if bond is None:
                    raise RuntimeError("endpoint_path_bond_missing")
                bond_tokens.append(bond_category(bond))
            local_counts: dict[int, float] = {}
            path_bonds = len(bond_tokens)
            for width in range(1, min(max_window, path_bonds) + 1):
                for start in range(0, path_bonds - width + 1):
                    local_atoms = atom_tokens[start : start + width + 1]
                    local_bonds = bond_tokens[start : start + width]
                    forward_parts: list[str] = []
                    reverse_parts: list[str] = []
                    for pos, atom_token in enumerate(local_atoms):
                        forward_parts.append(atom_token)
                        if pos < len(local_bonds):
                            forward_parts.append(local_bonds[pos])
                    rev_atoms = list(reversed(local_atoms))
                    rev_bonds = list(reversed(local_bonds))
                    for pos, atom_token in enumerate(rev_atoms):
                        reverse_parts.append(atom_token)
                        if pos < len(rev_bonds):
                            reverse_parts.append(rev_bonds[pos])
                    token = min("|".join(forward_parts), "|".join(reverse_parts))
                    col = stable_hash_index(f"endpoint_path_w{width}:{token}", n_features)
                    local_counts[col] = local_counts.get(col, 0.0) + 1.0
                    token_count += 1
            whole_token = "|".join(
                part
                for pair in zip(atom_tokens, bond_tokens + [""], strict=True)
                for part in pair
                if part
            )
            reverse_whole = "|".join(
                part
                for pair in zip(list(reversed(atom_tokens)), list(reversed(bond_tokens)) + [""], strict=True)
                for part in pair
                if part
            )
            col = stable_hash_index(f"endpoint_path_full:{min(whole_token, reverse_whole)}", n_features)
            local_counts[col] = local_counts.get(col, 0.0) + 1.0
            token_count += 1
            for col, count in local_counts.items():
                rows.append(row)
                cols.append(col)
                data.append(float(math.log1p(count)))
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
        status_counts[status] = status_counts.get(status, 0) + 1
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), int(n_features)), dtype=np.float32)
    if matrix.nnz:
        matrix.sum_duplicates()
    return matrix, {
        "source": "official train/test SMILES only; orientation-invariant sparse atom/bond n-grams along the polymer endpoint path",
        "n_features": int(n_features),
        "max_bonds": int(max_bonds),
        "nnz": int(matrix.nnz),
        "token_count": int(token_count),
        "status_counts": status_counts,
    }


def backbone_sidechain_fragment_mols(mols: list[Chem.Mol]) -> tuple[list[Chem.Mol], list[Chem.Mol], dict[str, Any]]:
    """Return endpoint-backbone and off-backbone fragment molecules per row."""

    empty = Chem.Mol()
    backbone_mols: list[Chem.Mol] = []
    side_mols: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    empty_backbone_rows = 0
    empty_side_rows = 0
    for mol in mols:
        status = "ok"
        backbone_fragment: Chem.Mol | None = None
        side_fragment: Chem.Mol | None = None
        try:
            dummy_atoms = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError(f"endpoint_count_{len(dummy_atoms)}")
            path = tuple(Chem.rdmolops.GetShortestPath(mol, int(dummy_atoms[0]), int(dummy_atoms[1])))
            if not path:
                raise RuntimeError("endpoint_path_missing")
            path_set = {int(index) for index in path}
            heavy_atoms = {int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
            backbone_heavy = {idx for idx in path_set if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
            side_heavy = heavy_atoms.difference(path_set)
            backbone_fragment = region_fragment_from_atoms(mol, backbone_heavy)
            side_fragment = region_fragment_from_atoms(mol, side_heavy)
        except Exception as exc:
            status = f"failed_{type(exc).__name__}"
        if backbone_fragment is None or backbone_fragment.GetNumAtoms() == 0:
            backbone_fragment = empty
            empty_backbone_rows += 1
        if side_fragment is None or side_fragment.GetNumAtoms() == 0:
            side_fragment = empty
            empty_side_rows += 1
        backbone_mols.append(backbone_fragment)
        side_mols.append(side_fragment)
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "source": "official train/test SMILES only; endpoint shortest-path backbone and off-path side-chain fragments for region-specific sparse kernels",
        "status_counts": status_counts,
        "empty_backbone_rows": int(empty_backbone_rows),
        "empty_side_rows": int(empty_side_rows),
    }
    return backbone_mols, side_mols, report


def cap_polymer_smiles(smiles: str) -> str:
    capped = re.sub(r"\[\*[^\]]*\]", "C", str(smiles))
    return capped.replace("*", "C")


def cap_polymer_smiles_hydrogen(smiles: str) -> str:
    capped = re.sub(r"\[\*[^\]]*\]", "[H]", str(smiles))
    return capped.replace("*", "[H]")


def capped_descriptor_mols(smiles: list[str], fallback_mols: list[Chem.Mol]) -> tuple[list[Chem.Mol], dict[str, int]]:
    mols: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = "hydrogen_cap"
        mol = Chem.MolFromSmiles(cap_polymer_smiles_hydrogen(smi), sanitize=True)
        if mol is None:
            mol = Chem.MolFromSmiles(cap_polymer_smiles(smi), sanitize=True)
            status = "carbon_cap_fallback" if mol is not None else "original_fallback"
        if mol is None:
            mol = fallback
        mols.append(mol)
        status_counts[status] = status_counts.get(status, 0) + 1
    return mols, status_counts


def periodic_closure_mol(smiles: str, fallback_mol: Chem.Mol) -> Chem.Mol:
    """Close the two polymer attachment endpoints into one repeat-cycle graph.

    This uses only the official repeat-unit SMILES. If closure is chemically
    invalid for a row, the original molecule is retained so the feature block
    remains total and deterministic.
    """
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return fallback_mol
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 2:
        return fallback_mol
    endpoints: list[tuple[int, int, Chem.BondType]] = []
    for atom in dummy_atoms:
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            return fallback_mol
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
        if bond is None:
            return fallback_mol
        endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
    first_neighbor = endpoints[0][1]
    second_neighbor = endpoints[1][1]
    if first_neighbor == second_neighbor:
        return fallback_mol

    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(first_neighbor, second_neighbor) is None:
        closure_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE
        try:
            rw.AddBond(first_neighbor, second_neighbor, closure_type)
        except Exception:
            return fallback_mol
    for dummy_idx in sorted((endpoints[0][0], endpoints[1][0]), reverse=True):
        rw.RemoveAtom(dummy_idx)
    try:
        closed = rw.GetMol()
        Chem.SanitizeMol(closed)
        return closed
    except Exception:
        return fallback_mol


def oligomer_mols(smiles: list[str], fallback_mols: list[Chem.Mol], repeats: int) -> tuple[list[Chem.Mol], dict[str, Any]]:
    if repeats < 2:
        raise ValueError("oligomer repeats must be at least 2")
    out: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = f"{repeats}mer"
        try:
            mol = Chem.MolFromSmiles(str(smi), sanitize=True)
            if mol is None:
                raise RuntimeError("parse_failed")
            dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError("endpoint_count")
            endpoints: list[tuple[int, int, Chem.BondType]] = []
            for atom in dummy_atoms:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1:
                    raise RuntimeError("endpoint_degree")
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                if bond is None:
                    raise RuntimeError("endpoint_bond")
                endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
            endpoints.sort(key=lambda item: item[0])
            remove_indices = sorted((endpoints[0][0], endpoints[1][0]))

            rw_core = Chem.RWMol(mol)
            for dummy_idx in reversed(remove_indices):
                rw_core.RemoveAtom(dummy_idx)
            core = rw_core.GetMol()
            Chem.SanitizeMol(core)
            if core.GetNumAtoms() <= 1:
                raise RuntimeError("empty_core")

            def remap(old_idx: int) -> int:
                return old_idx - sum(1 for dummy_idx in remove_indices if dummy_idx < old_idx)

            left_idx = remap(endpoints[0][1])
            right_idx = remap(endpoints[1][1])
            if left_idx == right_idx:
                raise RuntimeError("same_endpoint_neighbor")
            bond_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE

            combined = core
            for _ in range(repeats - 1):
                combined = Chem.CombineMols(combined, core)
            rw = Chem.RWMol(combined)
            n_core = core.GetNumAtoms()
            for repeat_idx in range(repeats - 1):
                a_idx = repeat_idx * n_core + right_idx
                b_idx = (repeat_idx + 1) * n_core + left_idx
                if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
                    rw.AddBond(a_idx, b_idx, bond_type)
            candidate = rw.GetMol()
            Chem.SanitizeMol(candidate)
            out.append(candidate)
        except Exception as exc:
            status = f"fallback_{type(exc).__name__}"
            out.append(fallback)
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "source": "official train/test SMILES only; deterministic linear oligomer built by removing two dummy endpoints and joining repeat cores",
        "repeats": int(repeats),
        "status_counts": status_counts,
    }
    return out, report


def oligomer_mols_with_row_status(
    smiles: list[str],
    fallback_mols: list[Chem.Mol],
    repeats: int,
) -> tuple[list[Chem.Mol], list[str], dict[str, Any]]:
    if repeats < 2:
        raise ValueError("oligomer repeats must be at least 2")
    out: list[Chem.Mol] = []
    row_statuses: list[str] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = f"{repeats}mer"
        try:
            mol = Chem.MolFromSmiles(str(smi), sanitize=True)
            if mol is None:
                raise RuntimeError("parse_failed")
            dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError("endpoint_count")
            endpoints: list[tuple[int, int, Chem.BondType]] = []
            for atom in dummy_atoms:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1:
                    raise RuntimeError("endpoint_degree")
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                if bond is None:
                    raise RuntimeError("endpoint_bond")
                endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
            endpoints.sort(key=lambda item: item[0])
            remove_indices = sorted((endpoints[0][0], endpoints[1][0]))

            rw_core = Chem.RWMol(mol)
            for dummy_idx in reversed(remove_indices):
                rw_core.RemoveAtom(dummy_idx)
            core = rw_core.GetMol()
            Chem.SanitizeMol(core)
            if core.GetNumAtoms() <= 1:
                raise RuntimeError("empty_core")

            def remap(old_idx: int) -> int:
                return old_idx - sum(1 for dummy_idx in remove_indices if dummy_idx < old_idx)

            left_idx = remap(endpoints[0][1])
            right_idx = remap(endpoints[1][1])
            if left_idx == right_idx:
                raise RuntimeError("same_endpoint_neighbor")
            bond_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE

            combined = core
            for _ in range(repeats - 1):
                combined = Chem.CombineMols(combined, core)
            rw = Chem.RWMol(combined)
            n_core = core.GetNumAtoms()
            for repeat_idx in range(repeats - 1):
                a_idx = repeat_idx * n_core + right_idx
                b_idx = (repeat_idx + 1) * n_core + left_idx
                if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
                    rw.AddBond(a_idx, b_idx, bond_type)
            candidate = rw.GetMol()
            Chem.SanitizeMol(candidate)
            out.append(candidate)
        except Exception as exc:
            status = f"fallback_{type(exc).__name__}"
            out.append(fallback)
        row_statuses.append(status)
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "source": "official train/test SMILES only; deterministic linear oligomer with row-level construction status",
        "repeats": int(repeats),
        "status_counts": status_counts,
    }
    return out, row_statuses, report


def oligomer_3d_descriptor_matrix(
    smiles: list[str],
    mols: list[Chem.Mol],
    *,
    repeats_values: tuple[int, ...],
    conformers: int,
    seed: int,
    optimize_steps: int,
    poolings: tuple[str, ...],
    include_extended: bool,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    descriptor_names = rdkit_3d_descriptor_names(include_extended)
    names: list[str] = []
    for repeats in repeats_values:
        for pooling in poolings:
            names.extend([f"oligomer_{repeats}mer_3d_{pooling}_{name}" for name in descriptor_names])
    values = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    repeat_reports: dict[str, Any] = {}
    offset = 0
    width = len(descriptor_names) * len(poolings)
    for repeat_index, repeats in enumerate(repeats_values):
        if repeats < 2:
            raise ValueError("oligomer 3D repeats must be at least 2")
        olig_mols, row_statuses, olig_report = oligomer_mols_with_row_status(smiles, mols, repeats)
        descriptor_status_counts: dict[str, int] = {}
        total_embedded = 0
        total_descriptor_conformers = 0
        total_uff_failures = 0
        skipped_construction = 0
        for row, (mol, row_status) in enumerate(zip(olig_mols, row_statuses, strict=True)):
            if row_status != f"{repeats}mer":
                skipped_construction += 1
                continue
            pooled, pooled_report = pooled_3d_descriptor_for_mol(
                mol,
                seed=int(seed + 100000 * repeat_index + row),
                conformers=conformers,
                optimize_steps=optimize_steps,
                poolings=poolings,
                include_extended=include_extended,
            )
            status = str(pooled_report["status"])
            descriptor_status_counts[status] = descriptor_status_counts.get(status, 0) + 1
            total_embedded += int(pooled_report.get("embedded_conformers", 0))
            total_descriptor_conformers += int(pooled_report.get("descriptor_conformers", 0))
            total_uff_failures += int(pooled_report.get("uff_failures", 0))
            values[row, offset : offset + width] = pooled
        repeat_reports[f"{repeats}mer"] = olig_report | {
            "skipped_rows_due_to_oligomer_construction_status": int(skipped_construction),
            "descriptor_status_counts": descriptor_status_counts,
            "total_embedded_conformers": int(total_embedded),
            "total_descriptor_conformers": int(total_descriptor_conformers),
            "total_uff_failures": int(total_uff_failures),
        }
        offset += width
    report = {
        "source": "official train/test SMILES only; deterministic oligomer ETKDG 3D conformer descriptors with fold-local downstream imputation",
        "repeats": [int(item) for item in repeats_values],
        "conformers_per_mol": int(conformers),
        "optimize_steps": int(optimize_steps),
        "poolings": list(poolings),
        "include_extended_descriptors": bool(include_extended),
        "base_descriptor_count": int(len(descriptor_names)),
        "output_feature_count": int(values.shape[1]),
        "nonfinite_output_values": int(np.size(values) - np.isfinite(values).sum()),
        "repeat_reports": repeat_reports,
    }
    return values, names, report


def oligomer_slope_descriptor_matrix(
    smiles: list[str],
    mols: list[Chem.Mol],
    *,
    max_repeats: int,
    include_physics: bool,
    transform: str = "raw",
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if max_repeats < 2:
        raise ValueError("oligomer slope max repeats must be at least 2")
    if transform not in {"raw", "signed_log", "both"}:
        raise ValueError("oligomer slope transform must be raw, signed_log, or both")

    repeat_values = np.arange(1, max_repeats + 1, dtype=np.float64)
    matrices: list[np.ndarray] = []
    repeat_reports: dict[str, Any] = {}

    base_dense, base_names = descriptor_matrix(mols, smiles)
    dense_names = list(base_names)
    if include_physics:
        base_physics, physics_names = physics_feature_matrix(mols)
        base_dense = np.hstack([base_dense, base_physics])
        dense_names = dense_names + physics_names
    matrices.append(base_dense)
    repeat_reports["1mer"] = {"status_counts": {"original_with_dummy_endpoints": len(mols)}}

    for repeats in range(2, max_repeats + 1):
        repeat_mols, repeat_report = oligomer_mols(smiles, mols, repeats=repeats)
        repeat_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in repeat_mols]
        repeat_dense, repeat_names = descriptor_matrix(repeat_mols, repeat_smiles)
        current_names = list(repeat_names)
        if include_physics:
            repeat_physics, physics_names = physics_feature_matrix(repeat_mols)
            repeat_dense = np.hstack([repeat_dense, repeat_physics])
            current_names = current_names + physics_names
        if current_names != dense_names:
            raise RuntimeError(f"oligomer descriptor schema changed at repeats={repeats}")
        matrices.append(repeat_dense)
        repeat_reports[f"{repeats}mer"] = repeat_report

    stack = np.stack(matrices, axis=0).astype(np.float64, copy=False)
    finite = np.isfinite(stack)
    y = np.where(finite, stack, 0.0)
    x = repeat_values[:, None, None]
    count = finite.sum(axis=0).astype(np.float64)
    sx = (finite * x).sum(axis=0)
    sxx = (finite * x * x).sum(axis=0)
    sy = y.sum(axis=0)
    sxy = (y * x).sum(axis=0)
    denom = count * sxx - sx * sx
    valid = (count >= 2.0) & (np.abs(denom) > 1e-12)

    slope = np.full_like(sy, np.nan, dtype=np.float64)
    intercept = np.full_like(sy, np.nan, dtype=np.float64)
    slope[valid] = (count[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / count[valid]

    raw_out = np.hstack([intercept, slope])
    raw_names = [f"oligomer_linear_intercept_{name}" for name in dense_names] + [f"oligomer_linear_slope_{name}" for name in dense_names]
    signed_log_out = np.sign(raw_out) * np.log1p(np.abs(raw_out))
    signed_log_names = [f"oligomer_signedlog_{name}" for name in raw_names]
    if transform == "raw":
        out = raw_out
        names = raw_names
    elif transform == "signed_log":
        out = signed_log_out
        names = signed_log_names
    else:
        out = np.hstack([raw_out, signed_log_out])
        names = raw_names + signed_log_names
    report = {
        "source": "official train/test SMILES only; deterministic 1..N oligomer descriptor linear fit over repeat count",
        "max_repeats": int(max_repeats),
        "include_physics": bool(include_physics),
        "transform": transform,
        "base_descriptor_count": int(len(dense_names)),
        "raw_output_feature_count": int(raw_out.shape[1]),
        "output_feature_count": int(out.shape[1]),
        "nonfinite_output_values": int(np.size(out) - np.isfinite(out).sum()),
        "raw_nonfinite_output_values": int(np.size(raw_out) - np.isfinite(raw_out).sum()),
        "repeat_reports": repeat_reports,
    }
    return out, names, report


def oligomer_repeat_mols(
    smiles: list[str],
    fallback_mols: list[Chem.Mol],
    repeats: int,
) -> tuple[list[Chem.Mol], dict[str, Any]]:
    if repeats < 1:
        raise ValueError("oligomer repeat count must be at least 1")
    out: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = f"{repeats}mer"
        try:
            mol = Chem.MolFromSmiles(str(smi), sanitize=True)
            if mol is None:
                raise RuntimeError("parse_failed")
            dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError("endpoint_count")
            endpoints: list[tuple[int, int, Chem.BondType]] = []
            for atom in dummy_atoms:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1:
                    raise RuntimeError("endpoint_degree")
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                if bond is None:
                    raise RuntimeError("endpoint_bond")
                endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
            endpoints.sort(key=lambda item: item[0])
            remove_indices = sorted((endpoints[0][0], endpoints[1][0]))

            rw_core = Chem.RWMol(mol)
            for dummy_idx in reversed(remove_indices):
                rw_core.RemoveAtom(dummy_idx)
            core = rw_core.GetMol()
            Chem.SanitizeMol(core)
            if core.GetNumAtoms() <= 1:
                raise RuntimeError("empty_core")

            def remap(old_idx: int) -> int:
                return old_idx - sum(1 for dummy_idx in remove_indices if dummy_idx < old_idx)

            left_idx = remap(endpoints[0][1])
            right_idx = remap(endpoints[1][1])
            if left_idx == right_idx:
                raise RuntimeError("same_endpoint_neighbor")
            bond_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE

            combined = core
            for _ in range(repeats - 1):
                combined = Chem.CombineMols(combined, core)
            rw = Chem.RWMol(combined)
            n_core = core.GetNumAtoms()
            for repeat_idx in range(repeats - 1):
                a_idx = repeat_idx * n_core + right_idx
                b_idx = (repeat_idx + 1) * n_core + left_idx
                if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
                    rw.AddBond(a_idx, b_idx, bond_type)
            candidate = rw.GetMol()
            Chem.SanitizeMol(candidate)
            out.append(candidate)
        except Exception as exc:
            status = f"fallback_{type(exc).__name__}"
            out.append(fallback)
        status_counts[status] = status_counts.get(status, 0) + 1
    return out, {
        "source": "official train/test SMILES only; deterministic endpoint-stripped n-mer repeat core for Flory-Fox-style asymptotic descriptors",
        "repeats": int(repeats),
        "status_counts": status_counts,
    }


def heavy_atom_counts(mols: list[Chem.Mol]) -> np.ndarray:
    counts = np.asarray(
        [max(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1), 1) for mol in mols],
        dtype=np.float64,
    )
    counts[~np.isfinite(counts)] = 1.0
    counts[counts <= 0] = 1.0
    return counts


FFOX_DESCRIPTOR_NAMES = [
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "FractionCSP3",
    "BertzCT",
    "BalabanJ",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Kappa1",
    "Kappa2",
    "Kappa3",
]


def oligomer_ffox_base_descriptor_matrix(mols: list[Chem.Mol], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    values = np.full((len(mols), len(FFOX_DESCRIPTOR_NAMES)), np.nan, dtype=np.float64)
    for row, mol in enumerate(mols):
        for col, name in enumerate(FFOX_DESCRIPTOR_NAMES):
            try:
                val = float(getattr(Descriptors, name)(mol))
            except Exception:
                val = math.nan
            values[row, col] = val if math.isfinite(val) else math.nan

    extra_names = [
        "smiles_len",
        "atom_count",
        "heavy_atom_count",
        "ring_count",
        "aromatic_atom_count",
        "hetero_atom_count",
        "halogen_count",
        "n_count",
        "o_count",
        "s_count",
        "si_count",
        "f_count",
        "cl_count",
        "br_count",
        "double_bond_count",
        "triple_bond_count",
        "conjugated_bond_count",
        "aromatic_bond_count",
        "branch_count",
        "bracket_count",
    ]
    extra = np.zeros((len(mols), len(extra_names)), dtype=np.float64)
    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        extra[row, 0] = len(str(smi))
        extra[row, 1] = len(atoms)
        extra[row, 2] = sum(1 for atom in atoms if atom.GetAtomicNum() > 1)
        extra[row, 3] = mol.GetRingInfo().NumRings()
        extra[row, 4] = sum(1 for atom in atoms if atom.GetIsAromatic())
        extra[row, 5] = sum(1 for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6))
        extra[row, 6] = sum(1 for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53))
        for col, atomic_num in ((7, 7), (8, 8), (9, 16), (10, 14), (11, 9), (12, 17), (13, 35)):
            extra[row, col] = sum(1 for atom in atoms if atom.GetAtomicNum() == atomic_num)
        extra[row, 14] = sum(1 for bond in bonds if str(bond.GetBondType()) == "DOUBLE")
        extra[row, 15] = sum(1 for bond in bonds if str(bond.GetBondType()) == "TRIPLE")
        extra[row, 16] = sum(1 for bond in bonds if bond.GetIsConjugated())
        extra[row, 17] = sum(1 for bond in bonds if bond.GetIsAromatic())
        extra[row, 18] = str(smi).count("(")
        extra[row, 19] = str(smi).count("[")
    return np.hstack([values, extra]), FFOX_DESCRIPTOR_NAMES + extra_names


def oligomer_ffox_descriptor_matrix(
    smiles: list[str],
    mols: list[Chem.Mol],
    *,
    max_repeats: int,
    include_physics: bool,
    transform: str = "raw",
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if max_repeats < 3:
        raise ValueError("Flory-Fox asymptotic descriptors require monomer/dimer/trimer, so max_repeats must be >= 3")
    if transform not in {"raw", "signed_log", "both"}:
        raise ValueError("oligomer Flory-Fox transform must be raw, signed_log, or both")

    repeat_values = np.arange(1, max_repeats + 1, dtype=np.float64)
    x_values = 1.0 / repeat_values
    matrices: list[np.ndarray] = []
    repeat_reports: dict[str, Any] = {}
    dense_names: list[str] | None = None

    for repeats in range(1, max_repeats + 1):
        repeat_mols, repeat_report = oligomer_repeat_mols(smiles, mols, repeats=repeats)
        repeat_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in repeat_mols]
        repeat_dense, repeat_names = oligomer_ffox_base_descriptor_matrix(repeat_mols, repeat_smiles)
        current_names = list(repeat_names)
        if include_physics:
            repeat_physics, physics_names = physics_feature_matrix(repeat_mols)
            repeat_dense = np.hstack([repeat_dense, repeat_physics])
            current_names = current_names + physics_names
        if dense_names is None:
            dense_names = current_names
        elif current_names != dense_names:
            raise RuntimeError(f"oligomer Flory-Fox descriptor schema changed at repeats={repeats}")
        scale = heavy_atom_counts(repeat_mols)[:, None]
        matrices.append(repeat_dense.astype(np.float64, copy=False) / scale)
        repeat_reports[f"{repeats}mer"] = repeat_report | {
            "heavy_atom_count_min": float(np.min(scale)),
            "heavy_atom_count_median": float(np.median(scale)),
            "heavy_atom_count_max": float(np.max(scale)),
        }

    if dense_names is None:
        raise RuntimeError("no Flory-Fox descriptor schema was built")
    stack = np.stack(matrices, axis=0).astype(np.float64, copy=False)
    finite = np.isfinite(stack)
    y = np.where(finite, stack, 0.0)
    x = x_values[:, None, None]
    count = finite.sum(axis=0).astype(np.float64)
    sx = (finite * x).sum(axis=0)
    sxx = (finite * x * x).sum(axis=0)
    sy = y.sum(axis=0)
    sxy = (y * x).sum(axis=0)
    denom = count * sxx - sx * sx
    valid = (count >= 2.0) & (np.abs(denom) > 1e-12)

    slope = np.full_like(sy, np.nan, dtype=np.float64)
    intercept = np.full_like(sy, np.nan, dtype=np.float64)
    slope[valid] = (count[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / count[valid]

    inf3 = np.full_like(intercept, np.nan, dtype=np.float64)
    m = stack[0]
    d = stack[1]
    t = stack[2]
    inf3_raw = (6.0 * t - 2.0 * d - m) / 3.0
    inf3[np.isfinite(inf3_raw)] = inf3_raw[np.isfinite(inf3_raw)]

    raw_out = np.hstack([intercept, slope, inf3])
    raw_names = (
        [f"oligomer_ffox_inf_per_heavy_{name}" for name in dense_names]
        + [f"oligomer_ffox_k_per_heavy_{name}" for name in dense_names]
        + [f"oligomer_ffox_inf3_formula_per_heavy_{name}" for name in dense_names]
    )
    raw_out[~np.isfinite(raw_out)] = np.nan
    signed_log_out = np.sign(raw_out) * np.log1p(np.abs(raw_out))
    signed_log_names = [f"oligomer_ffox_signedlog_{name}" for name in raw_names]
    if transform == "raw":
        out = raw_out
        names = raw_names
    elif transform == "signed_log":
        out = signed_log_out
        names = signed_log_names
    else:
        out = np.hstack([raw_out, signed_log_out])
        names = raw_names + signed_log_names
    report = {
        "source": "official train/test SMILES only; Flory-Fox-style n-mer descriptors normalized per heavy atom and linearly extrapolated against 1/n",
        "max_repeats": int(max_repeats),
        "repeat_values": [int(item) for item in repeat_values],
        "fit_x": "1/n",
        "normalization": "descriptor_value / heavy_atom_count before extrapolation",
        "include_physics": bool(include_physics),
        "transform": transform,
        "base_descriptor_count": int(len(dense_names)),
        "raw_output_feature_count": int(raw_out.shape[1]),
        "output_feature_count": int(out.shape[1]),
        "nonfinite_output_values": int(np.size(out) - np.isfinite(out).sum()),
        "raw_nonfinite_output_values": int(np.size(raw_out) - np.isfinite(raw_out).sum()),
        "repeat_reports": repeat_reports,
    }
    return out, names, report


def sparse_fingerprint(
    mols: list[Chem.Mol],
    *,
    fp_type: str,
    radius: int,
    n_bits: int,
    kind: str,
    log_counts: bool,
) -> sparse.csr_matrix:
    if fp_type == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            countSimulation=False,
            includeChirality=False,
            useBondTypes=True,
            onlyNonzeroInvariants=False,
            includeRingMembership=True,
            includeRedundantEnvironments=False,
        )
    elif fp_type == "atom_pair":
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits, includeChirality=False)
    elif fp_type == "topological_torsion":
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits, includeChirality=False)
    else:
        raise ValueError(fp_type)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    if kind == "bit":
        for row, mol in enumerate(mols):
            arr = np.zeros((n_bits,), dtype=np.int8)
            ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
            idx = np.flatnonzero(arr)
            rows.extend([row] * len(idx))
            cols.extend(idx.tolist())
            data.extend([1.0] * len(idx))
    elif kind == "count":
        for row, mol in enumerate(mols):
            for col, value in generator.GetCountFingerprint(mol).GetNonzeroElements().items():
                parsed = float(value)
                rows.append(row)
                cols.append(int(col))
                data.append(math.log1p(parsed) if log_counts else parsed)
    else:
        raise ValueError(kind)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), n_bits), dtype=np.float32)


def exact_morgan_count_dicts(
    mols: list[Chem.Mol],
    *,
    radii: tuple[int, ...],
    prefix: str,
    log_counts: bool = True,
) -> list[dict[str, float]]:
    """Return unfolded Morgan count tokens.

    The returned dictionaries are not vectorized here. DictVectorizer fitting is
    intentionally deferred to the target/fold fit rows so exact-feature
    vocabulary remains train-only.
    """

    generators = [
        (
            int(radius),
            rdFingerprintGenerator.GetMorganGenerator(
                radius=int(radius),
                countSimulation=False,
                includeChirality=False,
                useBondTypes=True,
                onlyNonzeroInvariants=False,
                includeRingMembership=True,
                includeRedundantEnvironments=False,
            ),
        )
        for radius in radii
    ]
    rows: list[dict[str, float]] = []
    for mol in mols:
        row: dict[str, float] = {}
        for radius, generator in generators:
            fp = generator.GetSparseCountFingerprint(mol)
            for key, value in fp.GetNonzeroElements().items():
                parsed = float(value)
                row[f"{prefix}_morgan_r{radius}:{int(key)}"] = math.log1p(parsed) if log_counts else parsed
        rows.append(row)
    return rows


def wl_subtree_count_dicts(
    mols: list[Chem.Mol],
    *,
    iterations: int,
    prefix: str,
    log_counts: bool = True,
) -> list[dict[str, float]]:
    """Return Weisfeiler-Lehman subtree count tokens.

    Vectorization is intentionally deferred to the target/fold fit rows. The
    compact labels are stable digests of deterministic atom/bond neighborhood
    signatures, not learned embeddings or pretrained chemistry features.
    """

    depth = max(0, int(iterations))
    rows: list[dict[str, float]] = []
    for mol in mols:
        labels = [atom_category(atom) for atom in mol.GetAtoms()]
        row_counts: dict[str, float] = {}
        for step in range(depth + 1):
            counts: dict[str, float] = {}
            for label in labels:
                token = f"{prefix}_wl{step}:{label}"
                counts[token] = counts.get(token, 0.0) + 1.0
            for token, count in counts.items():
                row_counts[token] = math.log1p(count) if log_counts else count
            if step == depth:
                break
            next_labels: list[str] = []
            for atom in mol.GetAtoms():
                pieces: list[str] = []
                atom_idx = int(atom.GetIdx())
                for bond in atom.GetBonds():
                    begin = int(bond.GetBeginAtomIdx())
                    end = int(bond.GetEndAtomIdx())
                    other = end if begin == atom_idx else begin
                    pieces.append(f"{bond_category(bond)}>{labels[other]}")
                signature = labels[atom_idx] + "|" + "|".join(sorted(pieces))
                digest = hashlib.blake2b(signature.encode("utf-8"), digest_size=12).hexdigest()
                next_labels.append(digest)
            labels = next_labels
        rows.append(row_counts)
    return rows


def morgan_feature_fingerprint(
    mols: list[Chem.Mol],
    *,
    radius: int,
    n_bits: int,
    kind: str,
    log_counts: bool,
) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    if kind == "bit":
        for row, mol in enumerate(mols):
            arr = np.zeros((n_bits,), dtype=np.int8)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=True)
            ConvertToNumpyArray(fp, arr)
            idx = np.flatnonzero(arr)
            rows.extend([row] * len(idx))
            cols.extend(idx.tolist())
            data.extend([1.0] * len(idx))
    elif kind == "count":
        for row, mol in enumerate(mols):
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits, useFeatures=True)
            for col, value in fp.GetNonzeroElements().items():
                parsed = float(value)
                rows.append(row)
                cols.append(int(col))
                data.append(math.log1p(parsed) if log_counts else parsed)
    else:
        raise ValueError(kind)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), n_bits), dtype=np.float32)


def rdk_fingerprint_matrix(mols: list[Chem.Mol], n_bits: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for row, mol in enumerate(mols):
        arr = np.zeros((n_bits,), dtype=np.int8)
        ConvertToNumpyArray(RDKFingerprint(mol, fpSize=n_bits), arr)
        idx = np.flatnonzero(arr)
        rows.extend([row] * len(idx))
        cols.extend(idx.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(mols), n_bits), dtype=np.float32)


def maccs_matrix(mols: list[Chem.Mol]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for row, mol in enumerate(mols):
        arr = np.zeros((167,), dtype=np.int8)
        ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
        idx = np.flatnonzero(arr)
        rows.extend([row] * len(idx))
        cols.extend(idx.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(mols), 167), dtype=np.float32)


def text_matrix(smiles: list[str], n_features: int) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 8),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    return vectorizer.transform(smiles).tocsr()


def rooted_smiles_text_matrix(mols: list[Chem.Mol], n_features: int, max_roots: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 9),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    flat_smiles: list[str] = []
    owners: list[int] = []
    status_counts: dict[str, int] = {}
    root_counts: list[int] = []
    for row, mol in enumerate(mols):
        row_smiles: list[str] = []
        try:
            root_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0]
            if max_roots > 0 and len(root_atoms) > max_roots:
                positions = np.linspace(0, len(root_atoms) - 1, int(max_roots), dtype=int)
                root_atoms = [root_atoms[int(pos)] for pos in sorted(set(positions.tolist()))]
            for root in root_atoms:
                try:
                    row_smiles.append(Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=int(root), isomericSmiles=True))
                except Exception:
                    continue
            if not row_smiles:
                row_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)]
            status = "ok"
        except Exception:
            row_smiles = [""]
            status = "fallback_blank"
        row_smiles = [value for value in row_smiles if value]
        if not row_smiles:
            row_smiles = [""]
            status = "fallback_blank"
        status_counts[status] = status_counts.get(status, 0) + 1
        root_counts.append(len(row_smiles))
        flat_smiles.extend(row_smiles)
        owners.extend([row] * len(row_smiles))
    hashed = vectorizer.transform(flat_smiles).tocsr()
    weights = np.asarray([1.0 / max(root_counts[owner], 1) for owner in owners], dtype=np.float32)
    aggregator = sparse.csr_matrix((weights, (owners, np.arange(len(owners)))), shape=(len(mols), len(owners)), dtype=np.float32)
    matrix = (aggregator @ hashed).tocsr()
    report = {
        "source": "official train/test SMILES only; deterministic rooted noncanonical SMILES hashed char n-grams averaged per molecule",
        "n_features": int(n_features),
        "max_roots": int(max_roots),
        "flat_smiles": int(len(flat_smiles)),
        "root_count_min": int(min(root_counts)) if root_counts else 0,
        "root_count_median": float(np.median(root_counts)) if root_counts else 0.0,
        "root_count_max": int(max(root_counts)) if root_counts else 0,
        "status_counts": status_counts,
        "nnz": int(matrix.nnz),
    }
    return matrix, report


def random_smiles_text_matrix(
    mols: list[Chem.Mol],
    n_features: int,
    augmentations: int,
    seed: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 10),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    flat_smiles: list[str] = []
    owners: list[int] = []
    counts: list[int] = []
    status_counts: dict[str, int] = {}
    per_mol = max(1, int(augmentations))
    base_seed = int(seed)
    for row, mol in enumerate(mols):
        row_smiles: list[str] = []
        status = "ok"
        try:
            row_smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
            random_values = Chem.MolToRandomSmilesVect(mol, max(0, per_mol - 1), randomSeed=base_seed + row, isomericSmiles=True)
            row_smiles.extend(str(value) for value in random_values if value)
        except Exception:
            row_smiles = [""]
            status = "fallback_blank"
        deduped = []
        seen: set[str] = set()
        for value in row_smiles:
            if value not in seen:
                deduped.append(value)
                seen.add(value)
        if not deduped:
            deduped = [""]
            status = "fallback_blank"
        status_counts[status] = status_counts.get(status, 0) + 1
        counts.append(len(deduped))
        flat_smiles.extend(deduped)
        owners.extend([row] * len(deduped))
    hashed = vectorizer.transform(flat_smiles).tocsr()
    weights = np.asarray([1.0 / max(counts[owner], 1) for owner in owners], dtype=np.float32)
    aggregator = sparse.csr_matrix((weights, (owners, np.arange(len(owners)))), shape=(len(mols), len(owners)), dtype=np.float32)
    matrix = (aggregator @ hashed).tocsr()
    report = {
        "source": "official train/test SMILES only; deterministic random noncanonical SMILES hashed char n-grams averaged per molecule",
        "n_features": int(n_features),
        "augmentations_requested": int(augmentations),
        "seed": int(seed),
        "flat_smiles": int(len(flat_smiles)),
        "variant_count_min": int(min(counts)) if counts else 0,
        "variant_count_median": float(np.median(counts)) if counts else 0.0,
        "variant_count_max": int(max(counts)) if counts else 0,
        "status_counts": status_counts,
        "nnz": int(matrix.nnz),
    }
    return matrix, report


def kekule_smiles_text_matrix(mols: list[Chem.Mol], n_features: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 9),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    values: list[str] = []
    status_counts: dict[str, int] = {}
    for mol in mols:
        try:
            working = Chem.Mol(mol)
            Chem.Kekulize(working, clearAromaticFlags=True)
            value = Chem.MolToSmiles(working, canonical=True, isomericSmiles=True, kekuleSmiles=True)
            status = "ok"
        except Exception:
            try:
                value = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                status = "fallback_canonical"
            except Exception:
                value = ""
                status = "fallback_blank"
        values.append(value)
        status_counts[status] = status_counts.get(status, 0) + 1
    matrix = vectorizer.transform(values).tocsr()
    report = {
        "source": "official train/test SMILES only; deterministic canonical kekulized SMILES hashed char n-grams",
        "n_features": int(n_features),
        "status_counts": status_counts,
        "nnz": int(matrix.nnz),
    }
    return matrix, report


def build_features(
    smiles: list[str],
    n_bits: int,
    text_features: int,
    motif_hash_features: int,
    rich_features: bool,
    periodic_features: bool,
    periodic_dense_features: bool,
    capped_dense_features: bool,
    motif_features: bool,
    physics_features: bool,
    mordred_features: bool,
    oligomer_features: bool,
    oligomer_repeats: int,
    oligomer_slope_features: bool = False,
    oligomer_slope_max_repeats: int = 4,
    oligomer_slope_transform: str = "raw",
    oligomer_ffox_features: bool = False,
    oligomer_ffox_max_repeats: int = 3,
    oligomer_ffox_transform: str = "raw",
    oligomer_3d_features: bool = False,
    oligomer_3d_repeats: str | tuple[int, ...] = (2, 3),
    conformers_per_mol: int = 1,
    conformer_pooling: str | tuple[str, ...] = ("mean", "std"),
    oligomer_3d_extended: bool = True,
    oligomer_mordred_features: bool = False,
    rdkit_3d_features: bool = False,
    conformer_seed: int = 20260721,
    conformer_opt_steps: int = 0,
    backbone_sidechain_features: bool = False,
    conjugation_features: bool = False,
    mobility_features: bool = False,
    huckel_features: bool = False,
    electronic_tail_features: bool = False,
    topological_autocorr_features: bool = False,
    topological_autocorr_max_distance: int = 8,
    infinite_chain_features: bool = False,
    bicerano_features: bool = False,
    map4_features: bool = False,
    map4_hash_features: int = 131072,
    map4_max_distance: int = 12,
    map4_env_radius: int = 1,
    region_sparse_features: bool = False,
    region_sparse_hash_features: int = 32768,
    endpoint_path_sparse_features: bool = False,
    endpoint_path_hash_features: int = 32768,
    endpoint_path_max_bonds: int = 8,
    rooted_smiles_features: bool = False,
    rooted_smiles_max_roots: int = 16,
    rooted_smiles_text_features: int | None = None,
    random_smiles_features: bool = False,
    random_smiles_augmentations: int = 16,
    random_smiles_seed: int = 20260722,
    random_smiles_text_features: int | None = None,
    kekule_smiles_features: bool = False,
    kekule_smiles_text_features: int | None = None,
    exact_sparse_features: bool = False,
    exact_sparse_radii: str | tuple[int, ...] = (1, 2, 3),
    wl_sparse_features: bool = False,
    wl_iterations: int = 3,
) -> dict[str, Any]:
    mols = build_mols(smiles)
    dense, dense_names = descriptor_matrix(mols, smiles)
    feature_reports: dict[str, Any] = {}
    extra_blocks: dict[str, sparse.csr_matrix] = {}
    exact_blocks: dict[str, list[dict[str, float]]] = {}
    exact_radii = tuple(parse_int_csv(exact_sparse_radii)) if isinstance(exact_sparse_radii, str) else tuple(int(item) for item in exact_sparse_radii)
    if not exact_radii:
        exact_radii = (1, 2, 3)
    if any(radius < 0 for radius in exact_radii):
        raise ValueError("exact sparse Morgan radii must be nonnegative")
    if capped_dense_features:
        capped_mols, capped_status = capped_descriptor_mols(smiles, mols)
        capped_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in capped_mols]
        capped_dense, capped_names = descriptor_matrix(capped_mols, capped_smiles)
        if physics_features:
            capped_physics_dense, capped_physics_names = physics_feature_matrix(capped_mols)
            capped_dense = np.hstack([capped_dense, capped_physics_dense])
            capped_names = capped_names + capped_physics_names
        dense = np.hstack([dense, capped_dense])
        dense_names = dense_names + [f"capped_h_{name}" for name in capped_names]
        feature_reports["capped_dense"] = {
            "source": "official train/test SMILES only; polymer dummy endpoints capped with explicit hydrogens before descriptor calculation",
            "status_counts": capped_status,
            "descriptor_count": len(capped_names),
        }
    if motif_features:
        motif_dense, motif_names, motif_dense_report = motif_dense_matrix(mols, smiles)
        dense = np.hstack([dense, motif_dense])
        dense_names = dense_names + motif_names
        feature_reports["motif_dense"] = motif_dense_report
        if int(motif_hash_features) > 0:
            motif_hash, motif_hash_report = motif_hash_matrix(mols, smiles, motif_hash_features)
            extra_blocks["motif_hash_count"] = motif_hash
            feature_reports["motif_hash"] = motif_hash_report
        else:
            feature_reports["motif_hash"] = {
                "source": "disabled by motif_hash_features=0; dense motif descriptors still enabled",
                "n_features": 0,
                "nnz": 0,
            }
    if rooted_smiles_features:
        rooted_n_features = int(rooted_smiles_text_features or text_features)
        rooted_block, rooted_report = rooted_smiles_text_matrix(mols, rooted_n_features, rooted_smiles_max_roots)
        extra_blocks["rooted_smiles_text"] = rooted_block
        feature_reports["rooted_smiles_text"] = rooted_report
    if random_smiles_features:
        random_n_features = int(random_smiles_text_features or text_features)
        random_block, random_report = random_smiles_text_matrix(
            mols,
            random_n_features,
            random_smiles_augmentations,
            random_smiles_seed,
        )
        extra_blocks["random_smiles_text"] = random_block
        feature_reports["random_smiles_text"] = random_report
    if kekule_smiles_features:
        kekule_n_features = int(kekule_smiles_text_features or text_features)
        kekule_block, kekule_report = kekule_smiles_text_matrix(mols, kekule_n_features)
        extra_blocks["kekule_smiles_text"] = kekule_block
        feature_reports["kekule_smiles_text"] = kekule_report
    if map4_features:
        map4_block, map4_report = map4_like_matrix(
            mols,
            n_features=int(map4_hash_features),
            max_distance=int(map4_max_distance),
            env_radius=int(map4_env_radius),
        )
        extra_blocks["map4_like_count"] = map4_block
        feature_reports["map4_like"] = map4_report
    if region_sparse_features:
        region_hash_width = int(region_sparse_hash_features)
        if region_hash_width <= 0:
            raise ValueError("region sparse hash feature width must be positive")
        backbone_mols, side_mols, region_report = backbone_sidechain_fragment_mols(mols)
        extra_blocks.update(
            {
                "region_bb_morgan_count_r2": sparse_fingerprint(backbone_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "region_side_morgan_count_r2": sparse_fingerprint(side_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "region_bb_fcfp_count_r2": morgan_feature_fingerprint(backbone_mols, radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "region_side_fcfp_count_r2": morgan_feature_fingerprint(side_mols, radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "region_bb_morgan_bit_r2": sparse_fingerprint(backbone_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="bit", log_counts=False),
                "region_side_morgan_bit_r2": sparse_fingerprint(side_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="bit", log_counts=False),
                "region_bb_rdk_bit": rdk_fingerprint_matrix(backbone_mols, n_bits=n_bits),
                "region_side_rdk_bit": rdk_fingerprint_matrix(side_mols, n_bits=n_bits),
            }
        )
        bb_map4, bb_map4_report = map4_like_matrix(
            backbone_mols,
            n_features=region_hash_width,
            max_distance=int(map4_max_distance),
            env_radius=int(map4_env_radius),
        )
        side_map4, side_map4_report = map4_like_matrix(
            side_mols,
            n_features=region_hash_width,
            max_distance=int(map4_max_distance),
            env_radius=int(map4_env_radius),
        )
        extra_blocks["region_bb_map4_like_count"] = bb_map4
        extra_blocks["region_side_map4_like_count"] = side_map4
        if exact_sparse_features:
            exact_blocks["exact_region_bb_morgan_count"] = exact_morgan_count_dicts(backbone_mols, radii=exact_radii, prefix="exact_region_bb")
            exact_blocks["exact_region_side_morgan_count"] = exact_morgan_count_dicts(side_mols, radii=exact_radii, prefix="exact_region_side")
        if wl_sparse_features:
            exact_blocks["wl_region_bb_subtree"] = wl_subtree_count_dicts(backbone_mols, iterations=wl_iterations, prefix="region_bb")
            exact_blocks["wl_region_side_subtree"] = wl_subtree_count_dicts(side_mols, iterations=wl_iterations, prefix="region_side")
        feature_reports["region_sparse"] = region_report | {
            "hash_width": region_hash_width,
            "blocks": sorted(name for name in extra_blocks if name.startswith("region_")),
            "exact_blocks_enabled": bool(exact_sparse_features),
            "wl_blocks_enabled": bool(wl_sparse_features),
            "bb_map4": bb_map4_report,
            "side_map4": side_map4_report,
        }
    if endpoint_path_sparse_features:
        endpoint_path_hash_width = int(endpoint_path_hash_features)
        if endpoint_path_hash_width <= 0:
            raise ValueError("endpoint path hash feature width must be positive")
        endpoint_path_block, endpoint_path_report = endpoint_path_ngram_matrix(
            mols,
            n_features=endpoint_path_hash_width,
            max_bonds=int(endpoint_path_max_bonds),
        )
        extra_blocks["endpoint_path_ngram_count"] = endpoint_path_block
        feature_reports["endpoint_path_sparse"] = endpoint_path_report
    if backbone_sidechain_features:
        bb_side_dense, bb_side_names, bb_side_report = backbone_sidechain_matrix(mols)
        dense = np.hstack([dense, bb_side_dense])
        dense_names = dense_names + bb_side_names
        feature_reports["backbone_sidechain"] = bb_side_report
    if conjugation_features:
        conjugation_dense, conjugation_names, conjugation_report = conjugation_feature_matrix(mols)
        dense = np.hstack([dense, conjugation_dense])
        dense_names = dense_names + conjugation_names
        feature_reports["conjugation"] = conjugation_report
    if mobility_features:
        mobility_dense, mobility_names, mobility_report = mobility_feature_matrix(mols)
        dense = np.hstack([dense, mobility_dense])
        dense_names = dense_names + mobility_names
        feature_reports["mobility"] = mobility_report
    if huckel_features:
        huckel_dense, huckel_names, huckel_report = huckel_spectrum_feature_matrix(mols)
        dense = np.hstack([dense, huckel_dense])
        dense_names = dense_names + huckel_names
        feature_reports["huckel"] = huckel_report
    if electronic_tail_features:
        electronic_dense, electronic_names, electronic_report = electronic_tail_feature_matrix(mols)
        dense = np.hstack([dense, electronic_dense])
        dense_names = dense_names + electronic_names
        feature_reports["electronic_tail"] = electronic_report
    if topological_autocorr_features:
        autocorr_dense, autocorr_names, autocorr_report = topological_autocorr_feature_matrix(
            mols,
            max_distance=topological_autocorr_max_distance,
        )
        dense = np.hstack([dense, autocorr_dense])
        dense_names = dense_names + autocorr_names
        feature_reports["topological_autocorr"] = autocorr_report
    if infinite_chain_features:
        infinite_dense, infinite_names, infinite_report = infinite_chain_proxy_feature_matrix(smiles, mols)
        dense = np.hstack([dense, infinite_dense])
        dense_names = dense_names + infinite_names
        feature_reports["infinite_chain_proxy"] = infinite_report
    if bicerano_features:
        bicerano_dense, bicerano_names, bicerano_report = bicerano_feature_matrix(smiles)
        dense = np.hstack([dense, bicerano_dense])
        dense_names = dense_names + bicerano_names
        feature_reports["bicerano"] = bicerano_report
    if exact_sparse_features:
        exact_blocks["exact_morgan_count"] = exact_morgan_count_dicts(mols, radii=exact_radii, prefix="exact_raw")
        feature_reports["exact_sparse"] = {
            "source": "official SMILES only; unfolded Morgan count dictionaries are vectorized only on target/fold official train rows",
            "radii": [int(radius) for radius in exact_radii],
            "blocks": sorted(exact_blocks),
        }
    if wl_sparse_features:
        exact_blocks["wl_subtree"] = wl_subtree_count_dicts(mols, iterations=wl_iterations, prefix="raw")
        feature_reports["wl_sparse"] = {
            "source": "official SMILES only; WL subtree count dictionaries are vectorized only on target/fold official train rows",
            "iterations": int(wl_iterations),
            "blocks": sorted([name for name in exact_blocks if name.startswith("wl_")]),
            "label_digest": "blake2b-96bit deterministic signature compaction; no learned or pretrained labels",
        }
    if physics_features:
        physics_dense, physics_names = physics_feature_matrix(mols)
        dense = np.hstack([dense, physics_dense])
        dense_names = dense_names + physics_names
    if mordred_features:
        mordred_dense, mordred_names = mordred_descriptor_matrix(mols)
        dense = np.hstack([dense, mordred_dense])
        dense_names = dense_names + mordred_names
    if oligomer_features:
        olig_mols, olig_report = oligomer_mols(smiles, mols, repeats=oligomer_repeats)
        olig_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in olig_mols]
        olig_dense, olig_names = descriptor_matrix(olig_mols, olig_smiles)
        if physics_features:
            olig_physics_dense, olig_physics_names = physics_feature_matrix(olig_mols)
            olig_dense = np.hstack([olig_dense, olig_physics_dense])
            olig_names = olig_names + olig_physics_names
        if oligomer_mordred_features:
            olig_mordred_dense, olig_mordred_names = mordred_descriptor_matrix(olig_mols)
            olig_dense = np.hstack([olig_dense, olig_mordred_dense])
            olig_names = olig_names + olig_mordred_names
        prefix = f"oligomer_{oligomer_repeats}mer"
        dense = np.hstack([dense, olig_dense])
        dense_names = dense_names + [f"{prefix}_{name}" for name in olig_names]
        extra_blocks.update(
            {
                f"{prefix}_maccs_bit": maccs_matrix(olig_mols),
                f"{prefix}_morgan_count_r2": sparse_fingerprint(olig_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="count", log_counts=True),
                f"{prefix}_morgan_count_r3": sparse_fingerprint(olig_mols, fp_type="morgan", radius=3, n_bits=n_bits, kind="count", log_counts=True),
                f"{prefix}_morgan_bit_r2": sparse_fingerprint(olig_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="bit", log_counts=False),
                f"{prefix}_morgan_bit_r3": sparse_fingerprint(olig_mols, fp_type="morgan", radius=3, n_bits=n_bits, kind="bit", log_counts=False),
                f"{prefix}_fcfp_count_r2": morgan_feature_fingerprint(olig_mols, radius=2, n_bits=n_bits, kind="count", log_counts=True),
                f"{prefix}_fcfp_bit_r2": morgan_feature_fingerprint(olig_mols, radius=2, n_bits=n_bits, kind="bit", log_counts=False),
                f"{prefix}_rdk_bit": rdk_fingerprint_matrix(olig_mols, n_bits=n_bits),
            }
        )
        olig_report["dense_descriptor_count"] = len(olig_names)
        olig_report["mordred_enabled"] = bool(oligomer_mordred_features)
        feature_reports["oligomer"] = olig_report
    if oligomer_slope_features:
        olig_slope_dense, olig_slope_names, olig_slope_report = oligomer_slope_descriptor_matrix(
            smiles,
            mols,
            max_repeats=oligomer_slope_max_repeats,
            include_physics=physics_features,
            transform=oligomer_slope_transform,
        )
        dense = np.hstack([dense, olig_slope_dense])
        dense_names = dense_names + olig_slope_names
        feature_reports["oligomer_slope"] = olig_slope_report
    if oligomer_ffox_features:
        olig_ffox_dense, olig_ffox_names, olig_ffox_report = oligomer_ffox_descriptor_matrix(
            smiles,
            mols,
            max_repeats=oligomer_ffox_max_repeats,
            include_physics=physics_features,
            transform=oligomer_ffox_transform,
        )
        dense = np.hstack([dense, olig_ffox_dense])
        dense_names = dense_names + olig_ffox_names
        feature_reports["oligomer_ffox"] = olig_ffox_report
    if oligomer_3d_features:
        parsed_repeats = parse_int_csv(oligomer_3d_repeats)
        parsed_pooling = parse_token_csv(conformer_pooling)
        olig_3d_dense, olig_3d_names, olig_3d_report = oligomer_3d_descriptor_matrix(
            smiles,
            mols,
            repeats_values=parsed_repeats,
            conformers=conformers_per_mol,
            seed=conformer_seed,
            optimize_steps=conformer_opt_steps,
            poolings=parsed_pooling,
            include_extended=oligomer_3d_extended,
        )
        dense = np.hstack([dense, olig_3d_dense])
        dense_names = dense_names + olig_3d_names
        feature_reports["oligomer_3d"] = olig_3d_report
    if rdkit_3d_features:
        rdkit3d_dense, rdkit3d_names, rdkit3d_report = rdkit_3d_descriptor_matrix(
            smiles,
            seed=conformer_seed,
            optimize_steps=conformer_opt_steps,
        )
        dense = np.hstack([dense, rdkit3d_dense])
        dense_names = dense_names + rdkit3d_names
        feature_reports["rdkit_3d"] = rdkit3d_report
    blocks = {
        "maccs_bit": maccs_matrix(mols),
        "morgan_count_r1": sparse_fingerprint(mols, fp_type="morgan", radius=1, n_bits=n_bits, kind="count", log_counts=True),
        "morgan_count_r2": sparse_fingerprint(mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="count", log_counts=True),
        "morgan_count_r3": sparse_fingerprint(mols, fp_type="morgan", radius=3, n_bits=n_bits, kind="count", log_counts=True),
        "morgan_bit_r1": sparse_fingerprint(mols, fp_type="morgan", radius=1, n_bits=n_bits, kind="bit", log_counts=False),
        "morgan_bit_r2": sparse_fingerprint(mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="bit", log_counts=False),
        "morgan_bit_r3": sparse_fingerprint(mols, fp_type="morgan", radius=3, n_bits=n_bits, kind="bit", log_counts=False),
        "atom_pair_count": sparse_fingerprint(mols, fp_type="atom_pair", radius=0, n_bits=n_bits, kind="count", log_counts=True),
        "topological_torsion_count": sparse_fingerprint(mols, fp_type="topological_torsion", radius=0, n_bits=n_bits, kind="count", log_counts=True),
        "char_text": text_matrix(smiles, text_features),
    }
    blocks.update(extra_blocks)
    if rich_features:
        capped_mols = []
        for smi, original_mol in zip(smiles, mols, strict=True):
            capped = Chem.MolFromSmiles(cap_polymer_smiles(smi), sanitize=True)
            capped_mols.append(capped if capped is not None else original_mol)
        if exact_sparse_features:
            exact_blocks["exact_capped_morgan_count"] = exact_morgan_count_dicts(capped_mols, radii=exact_radii, prefix="exact_capped")
        if wl_sparse_features:
            exact_blocks["wl_capped_subtree"] = wl_subtree_count_dicts(capped_mols, iterations=wl_iterations, prefix="capped")
        blocks.update(
            {
                "morgan_count_r4": sparse_fingerprint(mols, fp_type="morgan", radius=4, n_bits=n_bits, kind="count", log_counts=True),
                "morgan_count_r5": sparse_fingerprint(mols, fp_type="morgan", radius=5, n_bits=n_bits, kind="count", log_counts=True),
                "morgan_bit_r4": sparse_fingerprint(mols, fp_type="morgan", radius=4, n_bits=n_bits, kind="bit", log_counts=False),
                "morgan_bit_r5": sparse_fingerprint(mols, fp_type="morgan", radius=5, n_bits=n_bits, kind="bit", log_counts=False),
                "fcfp_count_r2": morgan_feature_fingerprint(mols, radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "fcfp_bit_r2": morgan_feature_fingerprint(mols, radius=2, n_bits=n_bits, kind="bit", log_counts=False),
                "rdk_bit": rdk_fingerprint_matrix(mols, n_bits=n_bits),
                "capped_morgan_count_r2": sparse_fingerprint(capped_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "capped_morgan_bit_r2": sparse_fingerprint(capped_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="bit", log_counts=False),
            }
        )
    if periodic_features:
        periodic_mols = [periodic_closure_mol(smi, mol) for smi, mol in zip(smiles, mols, strict=True)]
        if exact_sparse_features:
            exact_blocks["exact_periodic_morgan_count"] = exact_morgan_count_dicts(periodic_mols, radii=exact_radii, prefix="exact_periodic")
        if wl_sparse_features:
            exact_blocks["wl_periodic_subtree"] = wl_subtree_count_dicts(periodic_mols, iterations=wl_iterations, prefix="periodic")
        if periodic_dense_features:
            periodic_dense, periodic_dense_names = descriptor_matrix(periodic_mols, smiles)
            if physics_features:
                periodic_physics_dense, periodic_physics_names = physics_feature_matrix(periodic_mols)
                periodic_dense = np.hstack([periodic_dense, periodic_physics_dense])
                periodic_dense_names = periodic_dense_names + periodic_physics_names
            dense = np.hstack([dense, periodic_dense])
            dense_names = dense_names + [f"periodic_{name}" for name in periodic_dense_names]
        blocks.update(
            {
                "periodic_maccs_bit": maccs_matrix(periodic_mols),
                "periodic_morgan_count_r2": sparse_fingerprint(periodic_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "periodic_morgan_count_r3": sparse_fingerprint(periodic_mols, fp_type="morgan", radius=3, n_bits=n_bits, kind="count", log_counts=True),
                "periodic_morgan_bit_r2": sparse_fingerprint(periodic_mols, fp_type="morgan", radius=2, n_bits=n_bits, kind="bit", log_counts=False),
                "periodic_morgan_bit_r3": sparse_fingerprint(periodic_mols, fp_type="morgan", radius=3, n_bits=n_bits, kind="bit", log_counts=False),
                "periodic_fcfp_count_r2": morgan_feature_fingerprint(periodic_mols, radius=2, n_bits=n_bits, kind="count", log_counts=True),
                "periodic_fcfp_bit_r2": morgan_feature_fingerprint(periodic_mols, radius=2, n_bits=n_bits, kind="bit", log_counts=False),
                "periodic_rdk_bit": rdk_fingerprint_matrix(periodic_mols, n_bits=n_bits),
            }
        )
    if exact_sparse_features and "exact_sparse" in feature_reports:
        feature_reports["exact_sparse"]["blocks"] = sorted(exact_blocks)
        feature_reports["exact_sparse"]["nonempty_row_counts"] = {
            name: int(sum(1 for row in rows if row)) for name, rows in exact_blocks.items()
        }
    if wl_sparse_features and "wl_sparse" in feature_reports:
        wl_names = sorted(name for name in exact_blocks if name.startswith("wl_"))
        feature_reports["wl_sparse"]["blocks"] = wl_names
        feature_reports["wl_sparse"]["nonempty_row_counts"] = {
            name: int(sum(1 for row in exact_blocks[name] if row)) for name in wl_names
        }
    return {"dense": dense, "dense_names": dense_names, "blocks": blocks, "exact_blocks": exact_blocks, "feature_reports": feature_reports}


def canonical_no_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        raise RuntimeError(f"RDKit parse failed while canonicalizing {smiles!r}")
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def lookup_by_canonical(train: pd.DataFrame, target: str, agg: str = "median") -> pd.DataFrame:
    if agg not in {"mean", "median"}:
        raise ValueError(agg)
    sub = train[train["target_type"] == target].copy()
    grouped = sub.groupby("canon_no_stereo", sort=False)["target"]
    values = grouped.median() if agg == "median" else grouped.mean()
    return pd.DataFrame({"lookup_value": values, "lookup_count": grouped.size()})


def append_opposite_target_lookup(
    base_dense: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Add official-train opposite-property lookup features for one target model.

    For a Tg model, this exposes only official Egc labels sharing the same
    no-stereo canonical SMILES; for an Egc model, it exposes only official Tg
    labels. Same-target labels are intentionally excluded here so a train row
    cannot receive its own target as a feature.
    """
    opposite = "Egc" if target == "Tg" else "Tg"
    lookup = lookup_by_canonical(train, opposite, agg="median")
    all_keys = pd.concat([train["canon_no_stereo"], test["canon_no_stereo"]], ignore_index=True)
    aligned = all_keys.to_frame("canon_no_stereo").join(lookup, on="canon_no_stereo")
    present = aligned["lookup_value"].notna().to_numpy(dtype=np.float64)
    value = aligned["lookup_value"].to_numpy(dtype=np.float64)
    count = aligned["lookup_count"].fillna(0).to_numpy(dtype=np.float64)
    extra = np.column_stack([value, present, count]).astype(np.float64)
    report = {
        "target": target,
        "opposite_target": opposite,
        "lookup_train_rows_with_value": int(present[: len(train)].sum()),
        "lookup_test_rows_with_value": int(present[len(train) :].sum()),
        "unique_lookup_keys": int(len(lookup)),
    }
    return np.hstack([base_dense, extra]), [
        f"{opposite}_official_train_lookup_median_no_stereo",
        f"{opposite}_official_train_lookup_present_no_stereo",
        f"{opposite}_official_train_lookup_count_no_stereo",
    ], report


def same_target_test_overrides(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
    lookup = lookup_by_canonical(train, target, agg="median").rename(
        columns={"lookup_value": "override_value", "lookup_count": "override_count"}
    )
    local = test[test["target_type"] == target][["id", "canon_no_stereo"]].copy()
    return local.join(lookup, on="canon_no_stereo").dropna(subset=["override_value"])


def prepared_sparse(
    dense: np.ndarray,
    blocks: list[sparse.csr_matrix],
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    exact_dict_blocks: list[list[dict[str, float]]] | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_dense = dense[train_idx].astype(np.float64, copy=True)
    pred_dense = dense[pred_idx].astype(np.float64, copy=True)
    train_dense[~np.isfinite(train_dense)] = np.nan
    pred_dense[~np.isfinite(pred_dense)] = np.nan
    train_dense = scaler.fit_transform(imputer.fit_transform(train_dense)).astype(np.float32)
    pred_dense = scaler.transform(imputer.transform(pred_dense)).astype(np.float32)
    train_dense = np.nan_to_num(train_dense, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    pred_dense = np.nan_to_num(pred_dense, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    train_parts = [sparse.csr_matrix(train_dense), *[block[train_idx] for block in blocks]]
    pred_parts = [sparse.csr_matrix(pred_dense), *[block[pred_idx] for block in blocks]]
    for dict_rows in exact_dict_blocks or []:
        vectorizer = DictVectorizer(sparse=True, sort=True)
        train_dicts = [dict_rows[int(row_index)] for row_index in train_idx]
        pred_dicts = [dict_rows[int(row_index)] for row_index in pred_idx]
        train_exact = vectorizer.fit_transform(train_dicts).astype(np.float32)
        pred_exact = vectorizer.transform(pred_dicts).astype(np.float32)
        train_parts.append(train_exact)
        pred_parts.append(pred_exact)
    return sparse.hstack(train_parts, format="csr"), sparse.hstack(pred_parts, format="csr")


def tanimoto_kernel(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    inter = a @ b.T
    if sparse.issparse(inter):
        inter = inter.toarray()
    a_sum = np.asarray(a.sum(axis=1)).ravel()[:, None]
    b_sum = np.asarray(b.sum(axis=1)).ravel()[None, :]
    denom = a_sum + b_sum - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float64), where=denom > 0)


def generalized_tanimoto_kernel(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    inter = a @ b.T
    if sparse.issparse(inter):
        inter = inter.toarray()
    a_norm = np.asarray(a.multiply(a).sum(axis=1)).ravel()[:, None]
    b_norm = np.asarray(b.multiply(b).sum(axis=1)).ravel()[None, :]
    denom = a_norm + b_norm - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float64), where=denom > 0)


def clip_predictions(y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y_train, [0.01, 0.99])
    iqr = float(np.subtract(*np.quantile(y_train, [0.75, 0.25])))
    if not math.isfinite(iqr) or iqr <= 0.0:
        iqr = float(np.std(y_train))
    if not math.isfinite(iqr) or iqr <= 0.0:
        return pred.astype(np.float64)
    return np.clip(pred.astype(np.float64), q01 - 3.0 * iqr, q99 + 3.0 * iqr)


def ordinal_edges(y_train: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.float64)
    if strategy == "uniform":
        lo = float(np.min(y_train))
        hi = float(np.max(y_train))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise RuntimeError("cannot build uniform ordinal bins")
        edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
        edges = np.quantile(y_train, quantiles)
    else:
        raise ValueError(strategy)
    edges = np.unique(edges.astype(np.float64))
    if len(edges) < 4:
        raise RuntimeError("too few unique ordinal bin edges")
    edges[0] = min(edges[0], float(np.min(y_train))) - 1e-9
    edges[-1] = max(edges[-1], float(np.max(y_train))) + 1e-9
    return edges


def ordinal_labels_and_centers(y_train: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.searchsorted(edges[1:-1], y_train, side="right").astype(np.int64)
    n_classes = len(edges) - 1
    labels = np.clip(labels, 0, n_classes - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for cls in range(n_classes):
        mask = labels == cls
        if np.any(mask):
            centers[cls] = float(np.mean(y_train[mask]))
    return labels, centers.astype(np.float64)


def density_sample_weight(
    y_train: np.ndarray,
    *,
    n_bins: int,
    power: float,
    max_weight: float,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.ndim != 1 or len(y_train) == 0:
        raise RuntimeError("density weights require a nonempty 1D target")
    if not np.isfinite(y_train).all():
        raise RuntimeError("density weights require finite targets")
    if n_bins < 2 or power <= 0.0:
        return np.ones(len(y_train), dtype=np.float64)
    lo = float(np.min(y_train))
    hi = float(np.max(y_train))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.ones(len(y_train), dtype=np.float64)
    edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)
    labels = np.searchsorted(edges[1:-1], y_train, side="right")
    labels = np.clip(labels, 0, int(n_bins) - 1).astype(np.int64)
    counts = np.bincount(labels, minlength=int(n_bins)).astype(np.float64)
    weights = np.power(np.maximum(counts[labels], 1.0), -float(power))
    weights = weights / float(np.mean(weights))
    if math.isfinite(max_weight) and max_weight > 1.0:
        weights = np.clip(weights, 1.0 / float(max_weight), float(max_weight))
        weights = weights / float(np.mean(weights))
    return weights.astype(np.float64)


def fit_target_transform(y_train: np.ndarray, transform: str, seed: int) -> tuple[np.ndarray, Any]:
    y_train = np.asarray(y_train, dtype=np.float64)
    if transform == "identity":
        return y_train.copy(), None
    if transform == "fahrenheit_affine":
        return (y_train * 1.8 + 32.0).astype(np.float64), {"kind": "affine", "scale": 1.8, "offset": 32.0}
    if transform == "kelvin_affine":
        return (y_train + 273.15).astype(np.float64), {"kind": "affine", "scale": 1.0, "offset": 273.15}
    if transform == "yeo_johnson":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        transformed = transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
        return transformed.astype(np.float64), transformer
    if transform == "quantile_normal":
        n_quantiles = int(min(1000, max(10, len(y_train))))
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=None,
            random_state=int(seed),
        )
        transformed = transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
        return transformed.astype(np.float64), transformer
    if transform == "rank_uniform":
        n_quantiles = int(min(1000, max(10, len(y_train))))
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=None,
            random_state=int(seed),
        )
        transformed = transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
        return transformed.astype(np.float64), transformer
    raise ValueError(f"unknown target transform {transform!r}")


def inverse_target_transform(pred: np.ndarray, transformer: Any) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    if transformer is None:
        return pred
    if isinstance(transformer, dict) and transformer.get("kind") == "affine":
        scale = float(transformer["scale"])
        offset = float(transformer["offset"])
        if scale == 0.0:
            raise RuntimeError("invalid zero-scale affine target transform")
        return ((pred - offset) / scale).astype(np.float64)
    inverted = transformer.inverse_transform(pred.reshape(-1, 1)).ravel()
    return np.asarray(inverted, dtype=np.float64)


def duplicate_robust_fit_targets(
    y: np.ndarray,
    train_idx: np.ndarray,
    group_keys: np.ndarray | None,
    *,
    median_shrink: float,
    count_weight_power: float,
    mad_weight_power: float,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if group_keys is None:
        return y[train_idx].astype(np.float64, copy=True), None
    y_train = y[train_idx].astype(np.float64, copy=True)
    if len(y_train) == 0:
        raise RuntimeError("duplicate robust targets require nonempty train_idx")
    local_keys = np.asarray(group_keys, dtype=object)[train_idx]
    frame = pd.DataFrame({"key": local_keys, "target": y_train})
    grouped = frame.groupby("key", sort=False)["target"]
    medians = grouped.transform("median").to_numpy(dtype=np.float64)
    counts = grouped.transform("size").to_numpy(dtype=np.float64)
    abs_dev = np.abs(y_train - medians)
    frame_dev = pd.DataFrame({"key": local_keys, "abs_dev": abs_dev})
    mads = frame_dev.groupby("key", sort=False)["abs_dev"].transform("median").to_numpy(dtype=np.float64)

    shrink = float(np.clip(median_shrink, 0.0, 1.0))
    fit_y = (1.0 - shrink) * y_train + shrink * medians

    weights = np.ones(len(y_train), dtype=np.float64)
    if count_weight_power > 0.0:
        weights *= np.power(np.maximum(counts, 1.0), -float(count_weight_power))
    if mad_weight_power > 0.0:
        scale = float(np.subtract(*np.quantile(y_train, [0.75, 0.25])))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(y_train))
        if math.isfinite(scale) and scale > 0.0:
            weights *= np.power(1.0 + np.maximum(mads, 0.0) / scale, -float(mad_weight_power))
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    weights = weights / float(np.mean(weights))
    if math.isfinite(max_weight) and max_weight > 1.0:
        weights = np.clip(weights, 1.0 / float(max_weight), float(max_weight))
        weights = weights / float(np.mean(weights))
    return fit_y.astype(np.float64), weights.astype(np.float64)


def combine_sample_weights(*pieces: np.ndarray | None) -> np.ndarray | None:
    active = [np.asarray(piece, dtype=np.float64) for piece in pieces if piece is not None]
    if not active:
        return None
    out = np.ones(len(active[0]), dtype=np.float64)
    for piece in active:
        if len(piece) != len(out) or not np.isfinite(piece).all():
            raise RuntimeError("invalid sample weight piece")
        out *= np.maximum(piece, 0.0)
    if not np.any(out > 0):
        return np.ones(len(out), dtype=np.float64)
    out = np.where(np.isfinite(out) & (out > 0), out, 0.0)
    out = out / float(np.mean(out))
    return out.astype(np.float64)


def duplicate_robust_summary(
    y: np.ndarray,
    train_idx: np.ndarray,
    group_keys: np.ndarray,
    *,
    median_shrink: float,
    count_weight_power: float,
    mad_weight_power: float,
    max_weight: float,
) -> dict[str, Any]:
    fit_y, weights = duplicate_robust_fit_targets(
        y,
        train_idx,
        group_keys,
        median_shrink=median_shrink,
        count_weight_power=count_weight_power,
        mad_weight_power=mad_weight_power,
        max_weight=max_weight,
    )
    y_train = y[train_idx].astype(np.float64)
    local_keys = np.asarray(group_keys, dtype=object)[train_idx]
    frame = pd.DataFrame({"key": local_keys, "target": y_train, "fit_target": fit_y, "weight": weights})
    grouped = frame.groupby("key", sort=False)
    sizes = grouped.size()
    medians = grouped["target"].median()
    max_abs_dev = grouped.apply(lambda sub: float(np.max(np.abs(sub["target"].to_numpy(dtype=np.float64) - float(np.median(sub["target"].to_numpy(dtype=np.float64)))))))
    duplicate_mask = sizes > 1
    changed = np.abs(fit_y - y_train) > 1e-12
    return {
        "rows": int(len(train_idx)),
        "unique_groups": int(len(sizes)),
        "duplicate_groups": int(duplicate_mask.sum()),
        "duplicate_rows": int(sizes[duplicate_mask].sum()) if bool(duplicate_mask.any()) else 0,
        "conflicting_duplicate_groups": int((max_abs_dev[duplicate_mask] > 1e-12).sum()) if bool(duplicate_mask.any()) else 0,
        "fit_targets_changed_rows": int(np.count_nonzero(changed)),
        "fit_target_delta_mae": float(np.mean(np.abs(fit_y - y_train))) if len(y_train) else 0.0,
        "median_shrink": float(median_shrink),
        "count_weight_power": float(count_weight_power),
        "mad_weight_power": float(mad_weight_power),
        "max_weight": float(max_weight),
        "weight_min": float(np.min(weights)) if weights is not None else 1.0,
        "weight_max": float(np.max(weights)) if weights is not None else 1.0,
        "weight_mean": float(np.mean(weights)) if weights is not None else 1.0,
        "median_target_min": float(medians.min()) if len(medians) else None,
        "median_target_max": float(medians.max()) if len(medians) else None,
    }


def fit_predict_sparse(
    spec: ModelSpec,
    dense: np.ndarray,
    sparse_blocks: list[sparse.csr_matrix],
    exact_dict_blocks: list[list[dict[str, float]]] | None,
    y: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    select_k: int | None = None,
    density_weighted: bool = False,
    density_weight_bins: int = 24,
    density_weight_power: float = 0.5,
    density_weight_max: float = 6.0,
    duplicate_group_keys: np.ndarray | None = None,
    duplicate_median_shrink: float = 1.0,
    duplicate_count_weight_power: float = 0.5,
    duplicate_mad_weight_power: float = 1.0,
    duplicate_weight_max: float = 4.0,
) -> np.ndarray:
    x_train, x_pred = prepared_sparse(dense, sparse_blocks, train_idx, pred_idx, exact_dict_blocks)
    fit_y, duplicate_weight = duplicate_robust_fit_targets(
        y,
        train_idx,
        duplicate_group_keys,
        median_shrink=duplicate_median_shrink,
        count_weight_power=duplicate_count_weight_power,
        mad_weight_power=duplicate_mad_weight_power,
        max_weight=duplicate_weight_max,
    )
    target_transform = str(spec.params.get("target_transform", "identity"))
    model_y, target_transformer = fit_target_transform(
        fit_y,
        target_transform,
        seed=int(spec.params.get("seed", 17)),
    )
    selection_y = model_y if target_transform != "identity" else fit_y
    if select_k is not None and select_k > 0 and select_k < x_train.shape[1]:
        selector = SelectKBest(score_func=f_regression, k=int(select_k))
        selector.fit(x_train, selection_y)
        x_train = selector.transform(x_train)
        x_pred = selector.transform(x_pred)
    density_weight = None
    if density_weighted:
        density_weight = density_sample_weight(
            fit_y,
            n_bins=int(density_weight_bins),
            power=float(density_weight_power),
            max_weight=float(density_weight_max),
        )
    fit_weight = combine_sample_weights(duplicate_weight, density_weight)
    if spec.family == "ridge":
        y_mean = float(np.mean(model_y))
        y_std = float(np.std(model_y))
        if y_std <= 0:
            raise RuntimeError("zero target variance")
        model = Ridge(alpha=float(spec.params["alpha"]), solver="lsqr", fit_intercept=True, max_iter=10000, tol=1e-5)
        model.fit(x_train, (model_y - y_mean) / y_std, sample_weight=fit_weight)
        pred = inverse_target_transform(y_mean + y_std * model.predict(x_pred), target_transformer)
    elif spec.family == "lgbm":
        objective = str(spec.params.get("objective", "regression"))
        extra_params: dict[str, Any] = {}
        if objective == "quantile":
            extra_params["alpha"] = float(spec.params["alpha"])
        model = lgb.LGBMRegressor(
            objective=objective,
            boosting_type="gbdt",
            n_estimators=int(spec.params["n_estimators"]),
            learning_rate=float(spec.params["learning_rate"]),
            num_leaves=int(spec.params["num_leaves"]),
            min_child_samples=int(spec.params["min_child_samples"]),
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=float(spec.params.get("colsample_bytree", 0.8)),
            reg_alpha=float(spec.params.get("reg_alpha", 0.0)),
            reg_lambda=float(spec.params.get("reg_lambda", 5.0)),
            random_state=int(spec.params.get("seed", 17)),
            n_jobs=1,
            verbosity=-1,
            **extra_params,
        )
        model.fit(x_train, model_y, sample_weight=fit_weight)
        pred = inverse_target_transform(model.predict(x_pred), target_transformer)
    elif spec.family == "lgbm_ordinal":
        y_train = fit_y.astype(np.float64)
        edges = ordinal_edges(y_train, int(spec.params["n_bins"]), str(spec.params.get("bin_strategy", "uniform")))
        labels, centers = ordinal_labels_and_centers(y_train, edges)
        counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)
        weight_power = float(spec.params.get("weight_power", 0.5))
        sample_weight = np.power(np.maximum(counts[labels], 1.0), -weight_power)
        sample_weight = sample_weight / float(np.mean(sample_weight))
        if fit_weight is not None:
            sample_weight = sample_weight * fit_weight
            sample_weight = sample_weight / float(np.mean(sample_weight))
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=int(len(centers)),
            boosting_type="gbdt",
            n_estimators=int(spec.params["n_estimators"]),
            learning_rate=float(spec.params["learning_rate"]),
            num_leaves=int(spec.params["num_leaves"]),
            min_child_samples=int(spec.params["min_child_samples"]),
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=float(spec.params.get("colsample_bytree", 0.8)),
            reg_alpha=float(spec.params.get("reg_alpha", 0.0)),
            reg_lambda=float(spec.params.get("reg_lambda", 5.0)),
            random_state=int(spec.params.get("seed", 17)),
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(x_train, labels, sample_weight=sample_weight)
        raw_proba = np.asarray(model.predict_proba(x_pred), dtype=np.float64)
        if raw_proba.ndim != 2:
            raise RuntimeError("unexpected ordinal classifier probability rank")
        proba = np.zeros((raw_proba.shape[0], len(centers)), dtype=np.float64)
        classes = np.asarray(getattr(model, "classes_", np.arange(raw_proba.shape[1])), dtype=np.int64)
        if len(classes) != raw_proba.shape[1]:
            raise RuntimeError("ordinal classifier class/probability length mismatch")
        valid_class_mask = (classes >= 0) & (classes < len(centers))
        if not bool(np.any(valid_class_mask)):
            raise RuntimeError("ordinal classifier produced no usable class probabilities")
        proba[:, classes[valid_class_mask]] = raw_proba[:, valid_class_mask]
        row_sums = proba.sum(axis=1)
        missing_rows = row_sums <= 0
        if bool(np.any(missing_rows)):
            prior = np.bincount(labels, minlength=len(centers)).astype(np.float64)
            prior = prior / float(prior.sum())
            proba[missing_rows] = prior
            row_sums = proba.sum(axis=1)
        proba = proba / row_sums[:, None]
        pred = proba @ centers
    elif spec.family == "xgb":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(spec.params["n_estimators"]),
            learning_rate=float(spec.params["learning_rate"]),
            max_depth=int(spec.params["max_depth"]),
            min_child_weight=float(spec.params["min_child_weight"]),
            subsample=0.9,
            colsample_bytree=float(spec.params.get("colsample_bytree", 0.8)),
            reg_alpha=float(spec.params.get("reg_alpha", 0.0)),
            reg_lambda=float(spec.params.get("reg_lambda", 5.0)),
            random_state=int(spec.params.get("seed", 17)),
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(x_train, model_y, sample_weight=fit_weight)
        pred = inverse_target_transform(model.predict(x_pred), target_transformer)
    elif spec.family == "catboost":
        if not CATBOOST_AVAILABLE or cb is None:
            raise RuntimeError("catboost is not installed")
        model = cb.CatBoostRegressor(
            loss_function="RMSE",
            iterations=int(spec.params.get("iterations", 600)),
            learning_rate=float(spec.params.get("learning_rate", 0.035)),
            depth=int(spec.params.get("depth", 6)),
            l2_leaf_reg=float(spec.params.get("l2_leaf_reg", 5.0)),
            random_seed=int(spec.params.get("random_seed", spec.params.get("seed", 17))),
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        model.fit(x_train, model_y, sample_weight=fit_weight)
        pred = inverse_target_transform(model.predict(x_pred), target_transformer)
    elif spec.family == "sgd":
        y_mean = float(np.mean(model_y))
        y_std = float(np.std(model_y))
        if y_std <= 0:
            raise RuntimeError("zero target variance")
        model = SGDRegressor(
            loss=str(spec.params.get("loss", "huber")),
            penalty=str(spec.params.get("penalty", "elasticnet")),
            alpha=float(spec.params.get("alpha", 1e-5)),
            l1_ratio=float(spec.params.get("l1_ratio", 0.05)),
            epsilon=float(spec.params.get("epsilon", 0.1)),
            learning_rate=str(spec.params.get("learning_rate", "adaptive")),
            eta0=float(spec.params.get("eta0", 0.01)),
            max_iter=int(spec.params.get("max_iter", 3000)),
            tol=float(spec.params.get("tol", 1e-4)),
            average=bool(spec.params.get("average", True)),
            fit_intercept=True,
            random_state=int(spec.params.get("seed", 17)),
        )
        model.fit(x_train, (model_y - y_mean) / y_std, sample_weight=fit_weight)
        pred = inverse_target_transform(y_mean + y_std * model.predict(x_pred), target_transformer)
    elif spec.family == "elasticnet":
        y_mean = float(np.mean(model_y))
        y_std = float(np.std(model_y))
        if y_std <= 0:
            raise RuntimeError("zero target variance")
        model = ElasticNet(
            alpha=float(spec.params.get("alpha", 1e-4)),
            l1_ratio=float(spec.params.get("l1_ratio", 0.05)),
            fit_intercept=True,
            max_iter=int(spec.params.get("max_iter", 3000)),
            tol=float(spec.params.get("tol", 1e-4)),
            selection="random",
            random_state=int(spec.params.get("seed", 17)),
        )
        model.fit(x_train, (model_y - y_mean) / y_std, sample_weight=fit_weight)
        pred = inverse_target_transform(y_mean + y_std * model.predict(x_pred), target_transformer)
    elif spec.family == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=int(spec.params["n_estimators"]),
            max_features=float(spec.params["max_features"]),
            min_samples_leaf=int(spec.params["min_samples_leaf"]),
            max_depth=spec.params.get("max_depth"),
            bootstrap=False,
            random_state=int(spec.params.get("seed", 17)),
            n_jobs=1,
        )
        model.fit(x_train, model_y, sample_weight=fit_weight)
        pred = inverse_target_transform(model.predict(x_pred), target_transformer)
    else:
        raise ValueError(spec.family)
    return clip_predictions(fit_y, np.asarray(pred, dtype=np.float64))


def fit_predict_krr(
    bit_blocks: list[sparse.csr_matrix],
    y: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    alpha: float,
    duplicate_group_keys: np.ndarray | None = None,
    duplicate_median_shrink: float = 1.0,
    duplicate_count_weight_power: float = 0.5,
    duplicate_mad_weight_power: float = 1.0,
    duplicate_weight_max: float = 4.0,
) -> np.ndarray:
    k_train = None
    k_pred = None
    weight = 1.0 / len(bit_blocks)
    for block in bit_blocks:
        train_piece = tanimoto_kernel(block[train_idx], block[train_idx])
        pred_piece = tanimoto_kernel(block[pred_idx], block[train_idx])
        k_train = train_piece * weight if k_train is None else k_train + train_piece * weight
        k_pred = pred_piece * weight if k_pred is None else k_pred + pred_piece * weight
    assert k_train is not None and k_pred is not None
    fit_y, _ = duplicate_robust_fit_targets(
        y,
        train_idx,
        duplicate_group_keys,
        median_shrink=duplicate_median_shrink,
        count_weight_power=duplicate_count_weight_power,
        mad_weight_power=duplicate_mad_weight_power,
        max_weight=duplicate_weight_max,
    )
    y_mean = float(np.mean(fit_y))
    centered = fit_y - y_mean
    k_train.flat[:: k_train.shape[0] + 1] += alpha
    coef = np.linalg.solve(k_train, centered)
    pred = y_mean + k_pred @ coef
    return clip_predictions(fit_y, pred)


def fit_predict_count_tanimoto_krr(
    count_blocks: list[sparse.csr_matrix],
    y: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    alpha: float,
    duplicate_group_keys: np.ndarray | None = None,
    duplicate_median_shrink: float = 1.0,
    duplicate_count_weight_power: float = 0.5,
    duplicate_mad_weight_power: float = 1.0,
    duplicate_weight_max: float = 4.0,
) -> np.ndarray:
    if not count_blocks:
        raise RuntimeError("count Tanimoto KRR requires at least one count block")
    k_train = None
    k_pred = None
    weight = 1.0 / len(count_blocks)
    for block in count_blocks:
        train_piece = generalized_tanimoto_kernel(block[train_idx], block[train_idx])
        pred_piece = generalized_tanimoto_kernel(block[pred_idx], block[train_idx])
        k_train = train_piece * weight if k_train is None else k_train + train_piece * weight
        k_pred = pred_piece * weight if k_pred is None else k_pred + pred_piece * weight
    assert k_train is not None and k_pred is not None
    fit_y, _ = duplicate_robust_fit_targets(
        y,
        train_idx,
        duplicate_group_keys,
        median_shrink=duplicate_median_shrink,
        count_weight_power=duplicate_count_weight_power,
        mad_weight_power=duplicate_mad_weight_power,
        max_weight=duplicate_weight_max,
    )
    y_mean = float(np.mean(fit_y))
    centered = fit_y - y_mean
    k_train.flat[:: k_train.shape[0] + 1] += alpha
    coef = np.linalg.solve(k_train, centered)
    pred = y_mean + k_pred @ coef
    return clip_predictions(fit_y, pred)


def averaged_tanimoto_kernel(bit_blocks: list[sparse.csr_matrix], train_idx: np.ndarray, pred_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_train = None
    k_pred = None
    weight = 1.0 / len(bit_blocks)
    for block in bit_blocks:
        train_piece = tanimoto_kernel(block[train_idx], block[train_idx])
        pred_piece = tanimoto_kernel(block[pred_idx], block[train_idx])
        k_train = train_piece * weight if k_train is None else k_train + train_piece * weight
        k_pred = pred_piece * weight if k_pred is None else k_pred + pred_piece * weight
    assert k_train is not None and k_pred is not None
    return k_train.astype(np.float64, copy=False), k_pred.astype(np.float64, copy=False)


def fit_predict_tanimoto_svr(
    bit_blocks: list[sparse.csr_matrix],
    y: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    *,
    c_value: float,
    epsilon: float,
    duplicate_group_keys: np.ndarray | None = None,
    duplicate_median_shrink: float = 1.0,
    duplicate_count_weight_power: float = 0.5,
    duplicate_mad_weight_power: float = 1.0,
    duplicate_weight_max: float = 4.0,
) -> np.ndarray:
    k_train, k_pred = averaged_tanimoto_kernel(bit_blocks, train_idx, pred_idx)
    fit_y, _ = duplicate_robust_fit_targets(
        y,
        train_idx,
        duplicate_group_keys,
        median_shrink=duplicate_median_shrink,
        count_weight_power=duplicate_count_weight_power,
        mad_weight_power=duplicate_mad_weight_power,
        max_weight=duplicate_weight_max,
    )
    y_mean = float(np.mean(fit_y))
    y_std = float(np.std(fit_y))
    if y_std <= 0:
        raise RuntimeError("zero target variance")
    model = SVR(kernel="precomputed", C=float(c_value), epsilon=float(epsilon), cache_size=1024)
    model.fit(k_train, (fit_y - y_mean) / y_std)
    pred = y_mean + y_std * model.predict(k_pred)
    return clip_predictions(fit_y, np.asarray(pred, dtype=np.float64))


def fit_predict_svd_kernel_krr(
    dense: np.ndarray,
    sparse_blocks: list[sparse.csr_matrix],
    exact_dict_blocks: list[list[dict[str, float]]] | None,
    y: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    *,
    select_k: int | None,
    components: int,
    alpha: float,
    kernel: str,
    seed: int,
    duplicate_group_keys: np.ndarray | None = None,
    duplicate_median_shrink: float = 1.0,
    duplicate_count_weight_power: float = 0.5,
    duplicate_mad_weight_power: float = 1.0,
    duplicate_weight_max: float = 4.0,
) -> np.ndarray:
    if kernel not in {"laplacian", "rbf"}:
        raise ValueError(kernel)
    x_train, x_pred = prepared_sparse(dense, sparse_blocks, train_idx, pred_idx, exact_dict_blocks)
    fit_y, _ = duplicate_robust_fit_targets(
        y,
        train_idx,
        duplicate_group_keys,
        median_shrink=duplicate_median_shrink,
        count_weight_power=duplicate_count_weight_power,
        mad_weight_power=duplicate_mad_weight_power,
        max_weight=duplicate_weight_max,
    )
    if select_k is not None and select_k > 0 and select_k < x_train.shape[1]:
        selector = SelectKBest(score_func=f_regression, k=int(select_k))
        selector.fit(x_train, fit_y)
        x_train = selector.transform(x_train)
        x_pred = selector.transform(x_pred)
    n_components = int(min(max(4, components), max(2, x_train.shape[0] - 1), max(2, x_train.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
    z_train = svd.fit_transform(x_train)
    z_pred = svd.transform(x_pred)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train).astype(np.float64, copy=False)
    z_pred = scaler.transform(z_pred).astype(np.float64, copy=False)

    sample_size = min(800, z_train.shape[0])
    rng = np.random.default_rng(int(seed) + 7919)
    sample_idx = rng.choice(z_train.shape[0], size=sample_size, replace=False)
    sample_dist = euclidean_distances(z_train[sample_idx], z_train[sample_idx], squared=False)
    nonzero = sample_dist[sample_dist > 1e-12]
    bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    bandwidth = max(bandwidth, 1e-6)

    train_dist = euclidean_distances(z_train, z_train, squared=False)
    pred_dist = euclidean_distances(z_pred, z_train, squared=False)
    if kernel == "laplacian":
        k_train = np.exp(-train_dist / bandwidth)
        k_pred = np.exp(-pred_dist / bandwidth)
    else:
        denom = 2.0 * bandwidth * bandwidth
        k_train = np.exp(-(train_dist * train_dist) / denom)
        k_pred = np.exp(-(pred_dist * pred_dist) / denom)
    y_mean = float(np.mean(fit_y))
    centered = fit_y - y_mean
    k_train.flat[:: k_train.shape[0] + 1] += float(alpha)
    coef = np.linalg.solve(k_train, centered)
    pred = y_mean + k_pred @ coef
    return clip_predictions(fit_y, np.asarray(pred, dtype=np.float64))


def parse_float_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 < value < 1.0:
            raise ValueError(f"quantile alpha must be between 0 and 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("expected at least one quantile alpha")
    return tuple(values)


def model_specs(
    quick: bool,
    extra_trees: bool,
    lgbm_quantile: bool = False,
    ordinal_classifier: bool = False,
    target_transform_models: bool = False,
    robust_linear_models: bool = False,
    catboost_models: bool = False,
    quantile_alphas: tuple[float, ...] = (0.35, 0.5, 0.65),
    seed: int = 17,
) -> list[ModelSpec]:
    base_seed = int(seed)
    specs = [
        ModelSpec("ridge_a10", "ridge", {"alpha": 10.0}),
        ModelSpec("ridge_a100", "ridge", {"alpha": 100.0}),
        ModelSpec("lgbm_deep_lr03", "lgbm", {"n_estimators": 650, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 8, "reg_lambda": 3.0, "seed": base_seed + 101}),
        ModelSpec("lgbm_smooth_lr03", "lgbm", {"n_estimators": 650, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 8.0, "seed": base_seed + 131}),
        ModelSpec("xgb_depth4_lr03", "xgb", {"n_estimators": 650, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 2.0, "reg_lambda": 5.0, "seed": base_seed + 151}),
    ]
    if quick:
        specs = specs[:3]
    if extra_trees:
        trees = 300 if quick else 800
        specs.extend(
            [
                ModelSpec("extratrees_leaf1_mf55", "extratrees", {"n_estimators": trees, "max_features": 0.55, "min_samples_leaf": 1, "max_depth": None}),
                ModelSpec("extratrees_leaf2_mf70", "extratrees", {"n_estimators": trees, "max_features": 0.70, "min_samples_leaf": 2, "max_depth": None}),
            ]
        )
        specs[-2].params["seed"] = base_seed + 181
        specs[-1].params["seed"] = base_seed + 191
    if catboost_models:
        cat_iters = 450 if quick else 900
        specs.extend(
            [
                ModelSpec(
                    "catboost_d6_lr035_l2_5",
                    "catboost",
                    {
                        "iterations": cat_iters,
                        "learning_rate": 0.035,
                        "depth": 6,
                        "l2_leaf_reg": 5.0,
                        "random_seed": base_seed + 201,
                    },
                ),
                ModelSpec(
                    "catboost_d8_lr025_l2_8",
                    "catboost",
                    {
                        "iterations": cat_iters,
                        "learning_rate": 0.025,
                        "depth": 8,
                        "l2_leaf_reg": 8.0,
                        "random_seed": base_seed + 207,
                    },
                ),
            ]
        )
    if robust_linear_models:
        specs.extend(
            [
                ModelSpec(
                    "sgd_huber_enet_a1e5_l1p05",
                    "sgd",
                    {
                        "loss": "huber",
                        "penalty": "elasticnet",
                        "alpha": 1e-5,
                        "l1_ratio": 0.05,
                        "epsilon": 0.08,
                        "eta0": 0.01,
                        "max_iter": 2000 if quick else 4000,
                        "seed": base_seed + 211,
                    },
                ),
                ModelSpec(
                    "sgd_huber_enet_a3e5_l1p15",
                    "sgd",
                    {
                        "loss": "huber",
                        "penalty": "elasticnet",
                        "alpha": 3e-5,
                        "l1_ratio": 0.15,
                        "epsilon": 0.12,
                        "eta0": 0.01,
                        "max_iter": 2000 if quick else 4000,
                        "seed": base_seed + 221,
                    },
                ),
                ModelSpec(
                    "sgd_sqeps_l2_a1e5",
                    "sgd",
                    {
                        "loss": "squared_epsilon_insensitive",
                        "penalty": "l2",
                        "alpha": 1e-5,
                        "l1_ratio": 0.0,
                        "epsilon": 0.04,
                        "eta0": 0.006,
                        "max_iter": 2000 if quick else 4000,
                        "seed": base_seed + 231,
                    },
                ),
                ModelSpec(
                    "elasticnet_a1e4_l1p05",
                    "elasticnet",
                    {
                        "alpha": 1e-4,
                        "l1_ratio": 0.05,
                        "max_iter": 2500 if quick else 5000,
                        "seed": base_seed + 241,
                    },
                ),
            ]
        )
    if target_transform_models:
        specs.extend(
            [
                ModelSpec(
                    "ridge_yj_a30",
                    "ridge",
                    {"alpha": 30.0, "target_transform": "yeo_johnson", "seed": base_seed + 271},
                ),
                ModelSpec(
                    "lgbm_yj_smooth_lr03",
                    "lgbm",
                    {
                        "target_transform": "yeo_johnson",
                        "n_estimators": 650,
                        "learning_rate": 0.03,
                        "num_leaves": 31,
                        "min_child_samples": 16,
                        "reg_lambda": 8.0,
                        "seed": base_seed + 281,
                    },
                ),
                ModelSpec(
                    "lgbm_ranknorm_smooth_lr03",
                    "lgbm",
                    {
                        "target_transform": "quantile_normal",
                        "n_estimators": 650,
                        "learning_rate": 0.03,
                        "num_leaves": 31,
                        "min_child_samples": 16,
                        "reg_lambda": 8.0,
                        "seed": base_seed + 291,
                    },
                ),
                ModelSpec(
                    "xgb_yj_depth3_lr03",
                    "xgb",
                    {
                        "target_transform": "yeo_johnson",
                        "n_estimators": 650,
                        "learning_rate": 0.03,
                        "max_depth": 3,
                        "min_child_weight": 2.0,
                        "reg_lambda": 8.0,
                        "seed": base_seed + 301,
                    },
                ),
                ModelSpec(
                    "lgbm_fahrenheit_smooth_lr03",
                    "lgbm",
                    {
                        "target_transform": "fahrenheit_affine",
                        "n_estimators": 650,
                        "learning_rate": 0.03,
                        "num_leaves": 31,
                        "min_child_samples": 16,
                        "reg_lambda": 8.0,
                        "seed": base_seed + 311,
                    },
                ),
                ModelSpec(
                    "xgb_fahrenheit_depth3_lr03",
                    "xgb",
                    {
                        "target_transform": "fahrenheit_affine",
                        "n_estimators": 650,
                        "learning_rate": 0.03,
                        "max_depth": 3,
                        "min_child_weight": 2.0,
                        "reg_lambda": 8.0,
                        "seed": base_seed + 321,
                    },
                ),
                ModelSpec(
                    "extratrees_fahrenheit_leaf1_mf55",
                    "extratrees",
                    {
                        "target_transform": "fahrenheit_affine",
                        "n_estimators": 300 if quick else 800,
                        "max_features": 0.55,
                        "min_samples_leaf": 1,
                        "max_depth": None,
                        "seed": base_seed + 331,
                    },
                ),
                ModelSpec(
                    "extratrees_fahrenheit_leaf2_mf70",
                    "extratrees",
                    {
                        "target_transform": "fahrenheit_affine",
                        "n_estimators": 300 if quick else 800,
                        "max_features": 0.70,
                        "min_samples_leaf": 2,
                        "max_depth": None,
                        "seed": base_seed + 341,
                    },
                ),
                ModelSpec(
                    "extratrees_kelvin_leaf1_mf55",
                    "extratrees",
                    {
                        "target_transform": "kelvin_affine",
                        "n_estimators": 300 if quick else 800,
                        "max_features": 0.55,
                        "min_samples_leaf": 1,
                        "max_depth": None,
                        "seed": base_seed + 351,
                    },
                ),
            ]
        )
    if lgbm_quantile:
        for alpha in quantile_alphas:
            token = str(alpha).replace(".", "p")
            specs.append(
                ModelSpec(
                    f"lgbm_quantile_q{token}",
                    "lgbm",
                    {
                        "objective": "quantile",
                        "alpha": alpha,
                        "n_estimators": 650,
                        "learning_rate": 0.03,
                        "num_leaves": 31,
                        "min_child_samples": 12,
                        "reg_lambda": 5.0,
                        "seed": base_seed + int(round(alpha * 1000)) + 211,
                    },
                )
            )
    if ordinal_classifier:
        bins = (32, 48) if quick else (32, 48, 64)
        for n_bins in bins:
            for strategy in ("uniform", "quantile"):
                specs.append(
                    ModelSpec(
                        f"lgbm_ordinal_{strategy}_b{n_bins}",
                        "lgbm_ordinal",
                        {
                            "n_bins": n_bins,
                            "bin_strategy": strategy,
                            "weight_power": 0.5 if strategy == "uniform" else 0.25,
                            "n_estimators": 500 if quick else 800,
                            "learning_rate": 0.03,
                            "num_leaves": 31,
                            "min_child_samples": 10,
                            "reg_lambda": 5.0,
                            "seed": base_seed + n_bins * (3 if strategy == "uniform" else 5) + 251,
                        },
                    )
                )
    return specs


def score_frame(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(frame)), "by_target": {}}
    scores = []
    for target in TARGETS:
        sub = frame[frame["target_type"] == target]
        y = sub["target"].to_numpy(dtype=np.float64)
        pred = sub["prediction"].to_numpy(dtype=np.float64)
        r2 = float(r2_score(y, pred))
        scores.append(r2)
        out["by_target"][target] = {
            "rows": int(len(sub)),
            "r2": r2,
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
            "bias": float(np.mean(pred - y)),
        }
    out["combined_r2"] = float(np.mean(scores))
    return out


def fit_blend(y: np.ndarray, pred_matrix: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    design = np.column_stack([pred_matrix, np.ones(len(pred_matrix), dtype=np.float64)])
    fit_y = np.asarray(y, dtype=np.float64)
    fit_design = design
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if len(weight) != len(fit_y) or not np.isfinite(weight).all():
            raise RuntimeError("invalid blend sample weights")
        root_weight = np.sqrt(np.maximum(weight, 0.0))
        fit_design = design * root_weight[:, None]
        fit_y = fit_y * root_weight
    coef, _ = nnls(fit_design, fit_y)
    weights = coef[:-1]
    intercept = float(coef[-1])
    if float(weights.sum()) <= 0:
        return np.full(pred_matrix.shape[1], 1.0 / pred_matrix.shape[1]), 0.0
    return weights / float(weights.sum()), intercept


def target_indices(frame: pd.DataFrame, target: str) -> np.ndarray:
    return frame.index[frame["target_type"] == target].to_numpy(dtype=np.int64)


def make_holdout(train: pd.DataFrame, target: str, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = target_indices(train, target)
    train_local, val_local = train_test_split(idx, test_size=holdout_fraction, random_state=seed, shuffle=True)
    return np.sort(train_local.astype(np.int64)), np.sort(val_local.astype(np.int64))


def make_full_fit_indices(train: pd.DataFrame, target: str, fold_count: int, seed: int) -> list[np.ndarray]:
    idx = target_indices(train, target)
    if int(fold_count) <= 1:
        return [idx]
    n_splits = min(int(fold_count), len(idx))
    if n_splits < 2:
        return [idx]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    out: list[np.ndarray] = []
    for fit_local, _ in splitter.split(idx):
        out.append(np.sort(idx[fit_local].astype(np.int64)))
    return out


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = args.run_name or f"official_loop_{stamp}"
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    append_progress(run_dir, "run_dir_created", run_name=run_name)

    train, test = read_inputs()
    append_progress(run_dir, "inputs_loaded", train_rows=len(train), test_rows=len(test))
    train["canon_no_stereo"] = train["smiles"].map(canonical_no_stereo)
    test["canon_no_stereo"] = test["smiles"].map(canonical_no_stereo)
    all_smiles = train["smiles"].tolist() + test["smiles"].tolist()
    append_progress(
        run_dir,
        "build_features_start",
        total_smiles=len(all_smiles),
        rich_features=args.rich_features,
        periodic_features=args.periodic_features,
        capped_dense_features=args.capped_dense_features,
        motif_features=args.motif_features,
        backbone_sidechain_features=args.backbone_sidechain_features,
        conjugation_features=args.conjugation_features,
        mobility_features=args.mobility_features,
        huckel_features=args.huckel_features,
        electronic_tail_features=args.electronic_tail_features,
        topological_autocorr_features=args.topological_autocorr_features,
        topological_autocorr_max_distance=args.topological_autocorr_max_distance,
        infinite_chain_features=args.infinite_chain_features,
        bicerano_features=args.bicerano_features,
        map4_features=args.map4_features,
        map4_hash_features=args.map4_hash_features,
        map4_max_distance=args.map4_max_distance,
        map4_env_radius=args.map4_env_radius,
        region_sparse_features=args.region_sparse_features,
        region_sparse_hash_features=args.region_sparse_hash_features,
        endpoint_path_sparse_features=args.endpoint_path_sparse_features,
        endpoint_path_hash_features=args.endpoint_path_hash_features,
        endpoint_path_max_bonds=args.endpoint_path_max_bonds,
        rooted_smiles_features=args.rooted_smiles_features,
        rooted_smiles_max_roots=args.rooted_smiles_max_roots,
        rooted_smiles_text_features=args.rooted_smiles_text_features,
        kekule_smiles_features=args.kekule_smiles_features,
        kekule_smiles_text_features=args.kekule_smiles_text_features,
        exact_sparse_features=args.exact_sparse_features,
        exact_sparse_radii=args.exact_sparse_radii,
        oligomer_slope_features=args.oligomer_slope_features,
        oligomer_slope_max_repeats=args.oligomer_slope_max_repeats,
        oligomer_slope_transform=args.oligomer_slope_transform,
        oligomer_ffox_features=args.oligomer_ffox_features,
        oligomer_ffox_max_repeats=args.oligomer_ffox_max_repeats,
        oligomer_ffox_transform=args.oligomer_ffox_transform,
        oligomer_3d_features=args.oligomer_3d_features,
        oligomer_3d_repeats=args.oligomer_3d_repeats,
        conformers_per_mol=args.conformers_per_mol,
        conformer_pooling=args.conformer_pooling,
        oligomer_3d_extended=not args.no_oligomer_3d_extended,
    )
    features = build_features(
        all_smiles,
        n_bits=args.n_bits,
        text_features=args.text_features,
        motif_hash_features=args.motif_hash_features,
        rich_features=args.rich_features,
        periodic_features=args.periodic_features,
        periodic_dense_features=args.periodic_features and not args.periodic_sparse_only,
        capped_dense_features=args.capped_dense_features,
        motif_features=args.motif_features,
        physics_features=args.physics_features,
        mordred_features=args.mordred_features,
        oligomer_features=args.oligomer_features,
        oligomer_repeats=args.oligomer_repeats,
        oligomer_slope_features=args.oligomer_slope_features,
        oligomer_slope_max_repeats=args.oligomer_slope_max_repeats,
        oligomer_slope_transform=args.oligomer_slope_transform,
        oligomer_ffox_features=args.oligomer_ffox_features,
        oligomer_ffox_max_repeats=args.oligomer_ffox_max_repeats,
        oligomer_ffox_transform=args.oligomer_ffox_transform,
        oligomer_3d_features=args.oligomer_3d_features,
        oligomer_3d_repeats=args.oligomer_3d_repeats,
        conformers_per_mol=args.conformers_per_mol,
        conformer_pooling=args.conformer_pooling,
        oligomer_3d_extended=not args.no_oligomer_3d_extended,
        oligomer_mordred_features=args.oligomer_mordred_features,
        rdkit_3d_features=args.rdkit_3d_features,
        conformer_seed=args.conformer_seed,
        conformer_opt_steps=args.conformer_opt_steps,
        backbone_sidechain_features=args.backbone_sidechain_features,
        conjugation_features=args.conjugation_features,
        mobility_features=args.mobility_features,
        huckel_features=args.huckel_features,
        electronic_tail_features=args.electronic_tail_features,
        topological_autocorr_features=args.topological_autocorr_features,
        topological_autocorr_max_distance=args.topological_autocorr_max_distance,
        infinite_chain_features=args.infinite_chain_features,
        bicerano_features=args.bicerano_features,
        map4_features=args.map4_features,
        map4_hash_features=args.map4_hash_features,
        map4_max_distance=args.map4_max_distance,
        map4_env_radius=args.map4_env_radius,
        region_sparse_features=args.region_sparse_features,
        region_sparse_hash_features=args.region_sparse_hash_features,
        endpoint_path_sparse_features=args.endpoint_path_sparse_features,
        endpoint_path_hash_features=args.endpoint_path_hash_features,
        endpoint_path_max_bonds=args.endpoint_path_max_bonds,
        rooted_smiles_features=args.rooted_smiles_features,
        rooted_smiles_max_roots=args.rooted_smiles_max_roots,
        rooted_smiles_text_features=args.rooted_smiles_text_features,
        random_smiles_features=args.random_smiles_features,
        random_smiles_augmentations=args.random_smiles_augmentations,
        random_smiles_seed=args.random_smiles_seed,
        random_smiles_text_features=args.random_smiles_text_features,
        kekule_smiles_features=args.kekule_smiles_features,
        kekule_smiles_text_features=args.kekule_smiles_text_features,
        exact_sparse_features=args.exact_sparse_features,
        exact_sparse_radii=args.exact_sparse_radii,
        wl_sparse_features=args.wl_sparse_features,
        wl_iterations=args.wl_iterations,
    )
    append_progress(
        run_dir,
        "build_features_done",
        dense_features=features["dense"].shape[1],
        sparse_blocks=len(features["blocks"]),
        exact_blocks=len(features.get("exact_blocks", {})),
    )
    dense = features["dense"]
    blocks = features["blocks"]
    train_rows = len(train)
    y_all = train["target"].to_numpy(dtype=np.float64)
    duplicate_group_keys = None
    if args.duplicate_robust_training:
        duplicate_group_keys = (
            train["target_type"].astype(str).to_numpy(dtype=object)
            + "\t"
            + train["canon_no_stereo"].astype(str).to_numpy(dtype=object)
        )
        append_progress(
            run_dir,
            "duplicate_robust_training_enabled",
            median_shrink=args.duplicate_median_shrink,
            count_weight_power=args.duplicate_count_weight_power,
            mad_weight_power=args.duplicate_mad_weight_power,
            max_weight=args.duplicate_weight_max,
            unique_groups=int(len(pd.unique(duplicate_group_keys))),
        )
    sparse_block_names = [
        "maccs_bit",
        "morgan_count_r1",
        "morgan_count_r2",
        "morgan_count_r3",
        "atom_pair_count",
        "topological_torsion_count",
        "char_text",
    ]
    bit_block_names = ["morgan_bit_r1", "morgan_bit_r2", "morgan_bit_r3"]
    if args.rich_features:
        sparse_block_names.extend(
            [
                "morgan_count_r4",
                "morgan_count_r5",
                "fcfp_count_r2",
                "capped_morgan_count_r2",
                "rdk_bit",
            ]
        )
        bit_block_names.extend(["morgan_bit_r4", "morgan_bit_r5", "fcfp_bit_r2", "capped_morgan_bit_r2", "rdk_bit"])
    if args.periodic_features:
        sparse_block_names.extend(
            [
                "periodic_maccs_bit",
                "periodic_morgan_count_r2",
                "periodic_morgan_count_r3",
                "periodic_fcfp_count_r2",
                "periodic_rdk_bit",
            ]
        )
        bit_block_names.extend(["periodic_morgan_bit_r2", "periodic_morgan_bit_r3", "periodic_fcfp_bit_r2", "periodic_rdk_bit"])
    if args.motif_features and args.motif_hash_features > 0:
        sparse_block_names.append("motif_hash_count")
    if args.rooted_smiles_features:
        sparse_block_names.append("rooted_smiles_text")
    if args.random_smiles_features:
        sparse_block_names.append("random_smiles_text")
    if args.kekule_smiles_features:
        sparse_block_names.append("kekule_smiles_text")
    if args.map4_features:
        sparse_block_names.append("map4_like_count")
    if args.region_sparse_features:
        sparse_block_names.extend(
            [
                "region_bb_morgan_count_r2",
                "region_side_morgan_count_r2",
                "region_bb_fcfp_count_r2",
                "region_side_fcfp_count_r2",
                "region_bb_map4_like_count",
                "region_side_map4_like_count",
                "region_bb_rdk_bit",
                "region_side_rdk_bit",
            ]
        )
        bit_block_names.extend(
            [
                "region_bb_morgan_bit_r2",
                "region_side_morgan_bit_r2",
                "region_bb_rdk_bit",
                "region_side_rdk_bit",
            ]
        )
    if args.endpoint_path_sparse_features:
        sparse_block_names.append("endpoint_path_ngram_count")
    if args.oligomer_features:
        olig_prefix = f"oligomer_{args.oligomer_repeats}mer"
        sparse_block_names.extend(
            [
                f"{olig_prefix}_maccs_bit",
                f"{olig_prefix}_morgan_count_r2",
                f"{olig_prefix}_morgan_count_r3",
                f"{olig_prefix}_fcfp_count_r2",
                f"{olig_prefix}_rdk_bit",
            ]
        )
        bit_block_names.extend([f"{olig_prefix}_morgan_bit_r2", f"{olig_prefix}_morgan_bit_r3", f"{olig_prefix}_fcfp_bit_r2", f"{olig_prefix}_rdk_bit"])
    count_kernel_block_names = [
        name
        for name in sparse_block_names
        if (
            name.endswith("_count")
            or "_count_" in name
            or name
            in {
                "atom_pair_count",
                "topological_torsion_count",
                "motif_hash_count",
                "map4_like_count",
                "region_bb_map4_like_count",
                "region_side_map4_like_count",
                "endpoint_path_ngram_count",
            }
        )
    ]
    sparse_blocks = [blocks[name] for name in sparse_block_names]
    exact_block_names = sorted(features.get("exact_blocks", {}))
    exact_dict_blocks = [features["exact_blocks"][name] for name in exact_block_names]
    bit_blocks = [blocks[name] for name in bit_block_names]
    count_kernel_blocks = [blocks[name] for name in count_kernel_block_names]
    quantile_alphas = parse_float_list(args.lgbm_quantile_alphas)
    specs = model_specs(
        args.quick,
        args.extra_trees,
        args.lgbm_quantile,
        args.ordinal_classifier,
        args.target_transform_models,
        args.robust_linear_models,
        args.catboost_models,
        quantile_alphas,
        seed=args.seed,
    )
    spec_names = [spec.name for spec in specs] + [f"krr_a{args.krr_alpha:g}"]
    if args.count_tanimoto_krr:
        spec_names.append(f"count_tanimoto_krr_a{args.count_krr_alpha:g}")
    if args.tanimoto_svr:
        spec_names.append(f"tanimoto_svr_c{args.svr_c:g}_e{args.svr_epsilon:g}")
    svd_kernel_names: list[str] = []
    if args.svd_kernel_krr:
        svd_kernels = tuple(token.strip().lower() for token in args.svd_krr_kernels.split(",") if token.strip())
        if not svd_kernels:
            raise ValueError("--svd-krr-kernels must list at least one kernel")
        bad_kernels = sorted(set(svd_kernels) - {"laplacian", "rbf"})
        if bad_kernels:
            raise ValueError(f"unsupported --svd-krr-kernels: {bad_kernels}")
        for kernel in svd_kernels:
            name = f"svd_{kernel}_krr_c{args.svd_krr_components}_a{args.svd_krr_alpha:g}"
            svd_kernel_names.append(name)
            spec_names.append(name)
    else:
        svd_kernels = ()

    holdout_rows: list[dict[str, Any]] = []
    final_pred_by_target: dict[str, dict[str, np.ndarray]] = {}
    blend_weights: dict[str, Any] = {}
    lookup_feature_reports: dict[str, Any] = {}
    override_reports: dict[str, Any] = {}
    duplicate_robust_reports: dict[str, Any] = {}

    for target in TARGETS:
        target_dense, lookup_feature_names, lookup_report = append_opposite_target_lookup(dense, train, test, target)
        lookup_feature_reports[target] = lookup_report | {"feature_names": lookup_feature_names}
        fit_idx, val_idx = make_holdout(train, target, args.holdout_fraction, args.seed)
        test_local = test.index[test["target_type"] == target].to_numpy(dtype=np.int64)
        test_idx = train_rows + test_local
        val_pred_columns = []
        final_pred_by_target[target] = {}
        target_local_idx = target_indices(train, target)
        target_y = y_all[target_local_idx]
        target_group_keys = duplicate_group_keys[target_local_idx] if duplicate_group_keys is not None else None
        full_fit_indices = make_full_fit_indices(train, target, args.full_fold_ensemble, args.seed + (17 if target == "Tg" else 43))
        full_local_fit_indices = [np.searchsorted(target_local_idx, idx) for idx in full_fit_indices]
        full_pred_idx_local = np.arange(len(target_local_idx), len(target_local_idx) + len(test_idx), dtype=np.int64)
        if duplicate_group_keys is not None:
            duplicate_robust_reports[target] = {
                "holdout_fit": duplicate_robust_summary(
                    y_all,
                    fit_idx,
                    duplicate_group_keys,
                    median_shrink=args.duplicate_median_shrink,
                    count_weight_power=args.duplicate_count_weight_power,
                    mad_weight_power=args.duplicate_mad_weight_power,
                    max_weight=args.duplicate_weight_max,
                ),
                "full_fit": duplicate_robust_summary(
                    y_all,
                    target_local_idx,
                    duplicate_group_keys,
                    median_shrink=args.duplicate_median_shrink,
                    count_weight_power=args.duplicate_count_weight_power,
                    mad_weight_power=args.duplicate_mad_weight_power,
                    max_weight=args.duplicate_weight_max,
                ),
            }

        for spec in specs:
            val_pred = fit_predict_sparse(
                spec,
                target_dense,
                sparse_blocks,
                exact_dict_blocks,
                y_all,
                fit_idx,
                val_idx,
                args.select_k,
                args.density_weighted,
                args.density_weight_bins,
                args.density_weight_power,
                args.density_weight_max,
                duplicate_group_keys=duplicate_group_keys,
                duplicate_median_shrink=args.duplicate_median_shrink,
                duplicate_count_weight_power=args.duplicate_count_weight_power,
                duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                duplicate_weight_max=args.duplicate_weight_max,
            )
            full_pred = np.mean(
                [
                    fit_predict_sparse(
                        spec,
                        target_dense,
                        sparse_blocks,
                        exact_dict_blocks,
                        y_all,
                        fold_fit_idx,
                        test_idx,
                        args.select_k,
                        args.density_weighted,
                        args.density_weight_bins,
                        args.density_weight_power,
                        args.density_weight_max,
                        duplicate_group_keys=duplicate_group_keys,
                        duplicate_median_shrink=args.duplicate_median_shrink,
                        duplicate_count_weight_power=args.duplicate_count_weight_power,
                        duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                        duplicate_weight_max=args.duplicate_weight_max,
                    )
                    for fold_fit_idx in full_fit_indices
                ],
                axis=0,
            )
            val_pred_columns.append(val_pred)
            final_pred_by_target[target][spec.name] = full_pred
            for row_index, pred in zip(val_idx, val_pred, strict=True):
                holdout_rows.append(
                    {
                        "row_index": int(row_index),
                        "target_type": target,
                        "target": float(train.loc[row_index, "target"]),
                        "model": spec.name,
                        "prediction": float(pred),
                    }
                )

        local_fit = np.searchsorted(target_local_idx, fit_idx)
        local_val = np.searchsorted(target_local_idx, val_idx)
        target_bit_blocks = [block[target_local_idx] for block in bit_blocks]
        val_krr = fit_predict_krr(
            target_bit_blocks,
            target_y,
            local_fit,
            local_val,
            args.krr_alpha,
            duplicate_group_keys=target_group_keys,
            duplicate_median_shrink=args.duplicate_median_shrink,
            duplicate_count_weight_power=args.duplicate_count_weight_power,
            duplicate_mad_weight_power=args.duplicate_mad_weight_power,
            duplicate_weight_max=args.duplicate_weight_max,
        )
        final_bit_blocks = [sparse.vstack([block[target_local_idx], block[test_idx]], format="csr") for block in bit_blocks]
        full_krr = fit_predict_krr(
            final_bit_blocks,
            target_y,
            full_local_fit_indices[0],
            full_pred_idx_local,
            args.krr_alpha,
            duplicate_group_keys=target_group_keys,
            duplicate_median_shrink=args.duplicate_median_shrink,
            duplicate_count_weight_power=args.duplicate_count_weight_power,
            duplicate_mad_weight_power=args.duplicate_mad_weight_power,
            duplicate_weight_max=args.duplicate_weight_max,
        )
        if len(full_local_fit_indices) > 1:
            full_krr = np.mean(
                [
                    fit_predict_krr(
                        final_bit_blocks,
                        target_y,
                        fold_local_fit,
                        full_pred_idx_local,
                        args.krr_alpha,
                        duplicate_group_keys=target_group_keys,
                        duplicate_median_shrink=args.duplicate_median_shrink,
                        duplicate_count_weight_power=args.duplicate_count_weight_power,
                        duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                        duplicate_weight_max=args.duplicate_weight_max,
                    )
                    for fold_local_fit in full_local_fit_indices
                ],
                axis=0,
            )
        val_pred_columns.append(val_krr)
        final_pred_by_target[target][f"krr_a{args.krr_alpha:g}"] = full_krr
        for row_index, pred in zip(val_idx, val_krr, strict=True):
            holdout_rows.append(
                {
                    "row_index": int(row_index),
                    "target_type": target,
                    "target": float(train.loc[row_index, "target"]),
                    "model": f"krr_a{args.krr_alpha:g}",
                    "prediction": float(pred),
                }
            )

        if args.count_tanimoto_krr:
            count_krr_name = f"count_tanimoto_krr_a{args.count_krr_alpha:g}"
            target_count_blocks = [block[target_local_idx] for block in count_kernel_blocks]
            val_count_krr = fit_predict_count_tanimoto_krr(
                target_count_blocks,
                target_y,
                local_fit,
                local_val,
                args.count_krr_alpha,
                duplicate_group_keys=target_group_keys,
                duplicate_median_shrink=args.duplicate_median_shrink,
                duplicate_count_weight_power=args.duplicate_count_weight_power,
                duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                duplicate_weight_max=args.duplicate_weight_max,
            )
            final_count_blocks = [sparse.vstack([block[target_local_idx], block[test_idx]], format="csr") for block in count_kernel_blocks]
            full_count_krr = fit_predict_count_tanimoto_krr(
                final_count_blocks,
                target_y,
                full_local_fit_indices[0],
                full_pred_idx_local,
                args.count_krr_alpha,
                duplicate_group_keys=target_group_keys,
                duplicate_median_shrink=args.duplicate_median_shrink,
                duplicate_count_weight_power=args.duplicate_count_weight_power,
                duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                duplicate_weight_max=args.duplicate_weight_max,
            )
            if len(full_local_fit_indices) > 1:
                full_count_krr = np.mean(
                    [
                        fit_predict_count_tanimoto_krr(
                            final_count_blocks,
                            target_y,
                            fold_local_fit,
                            full_pred_idx_local,
                            args.count_krr_alpha,
                            duplicate_group_keys=target_group_keys,
                            duplicate_median_shrink=args.duplicate_median_shrink,
                            duplicate_count_weight_power=args.duplicate_count_weight_power,
                            duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                            duplicate_weight_max=args.duplicate_weight_max,
                        )
                        for fold_local_fit in full_local_fit_indices
                    ],
                    axis=0,
                )
            val_pred_columns.append(val_count_krr)
            final_pred_by_target[target][count_krr_name] = full_count_krr
            for row_index, pred in zip(val_idx, val_count_krr, strict=True):
                holdout_rows.append(
                    {
                        "row_index": int(row_index),
                        "target_type": target,
                        "target": float(train.loc[row_index, "target"]),
                        "model": count_krr_name,
                        "prediction": float(pred),
                    }
                )

        if args.tanimoto_svr:
            svr_name = f"tanimoto_svr_c{args.svr_c:g}_e{args.svr_epsilon:g}"
            val_svr = fit_predict_tanimoto_svr(
                target_bit_blocks,
                target_y,
                local_fit,
                local_val,
                c_value=args.svr_c,
                epsilon=args.svr_epsilon,
                duplicate_group_keys=target_group_keys,
                duplicate_median_shrink=args.duplicate_median_shrink,
                duplicate_count_weight_power=args.duplicate_count_weight_power,
                duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                duplicate_weight_max=args.duplicate_weight_max,
            )
            full_svr = fit_predict_tanimoto_svr(
                final_bit_blocks,
                target_y,
                full_local_fit_indices[0],
                full_pred_idx_local,
                c_value=args.svr_c,
                epsilon=args.svr_epsilon,
                duplicate_group_keys=target_group_keys,
                duplicate_median_shrink=args.duplicate_median_shrink,
                duplicate_count_weight_power=args.duplicate_count_weight_power,
                duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                duplicate_weight_max=args.duplicate_weight_max,
            )
            if len(full_local_fit_indices) > 1:
                full_svr = np.mean(
                    [
                        fit_predict_tanimoto_svr(
                            final_bit_blocks,
                            target_y,
                            fold_local_fit,
                            full_pred_idx_local,
                            c_value=args.svr_c,
                            epsilon=args.svr_epsilon,
                            duplicate_group_keys=target_group_keys,
                            duplicate_median_shrink=args.duplicate_median_shrink,
                            duplicate_count_weight_power=args.duplicate_count_weight_power,
                            duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                            duplicate_weight_max=args.duplicate_weight_max,
                        )
                        for fold_local_fit in full_local_fit_indices
                    ],
                    axis=0,
                )
            val_pred_columns.append(val_svr)
            final_pred_by_target[target][svr_name] = full_svr
            for row_index, pred in zip(val_idx, val_svr, strict=True):
                holdout_rows.append(
                    {
                        "row_index": int(row_index),
                        "target_type": target,
                        "target": float(train.loc[row_index, "target"]),
                        "model": svr_name,
                    "prediction": float(pred),
                }
            )

        if args.svd_kernel_krr:
            for kernel, svd_kernel_name in zip(svd_kernels, svd_kernel_names, strict=True):
                val_svd_krr = fit_predict_svd_kernel_krr(
                    target_dense,
                    sparse_blocks,
                    exact_dict_blocks,
                    y_all,
                    fit_idx,
                    val_idx,
                    select_k=args.select_k,
                    components=args.svd_krr_components,
                    alpha=args.svd_krr_alpha,
                    kernel=kernel,
                    seed=args.seed + (101 if target == "Tg" else 211),
                    duplicate_group_keys=duplicate_group_keys,
                    duplicate_median_shrink=args.duplicate_median_shrink,
                    duplicate_count_weight_power=args.duplicate_count_weight_power,
                    duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                    duplicate_weight_max=args.duplicate_weight_max,
                )
                full_svd_krr = np.mean(
                    [
                        fit_predict_svd_kernel_krr(
                            target_dense,
                            sparse_blocks,
                            exact_dict_blocks,
                            y_all,
                            fold_fit_idx,
                            test_idx,
                            select_k=args.select_k,
                            components=args.svd_krr_components,
                            alpha=args.svd_krr_alpha,
                            kernel=kernel,
                            seed=args.seed + fold_id * 1009 + (101 if target == "Tg" else 211),
                            duplicate_group_keys=duplicate_group_keys,
                            duplicate_median_shrink=args.duplicate_median_shrink,
                            duplicate_count_weight_power=args.duplicate_count_weight_power,
                            duplicate_mad_weight_power=args.duplicate_mad_weight_power,
                            duplicate_weight_max=args.duplicate_weight_max,
                        )
                        for fold_id, fold_fit_idx in enumerate(full_fit_indices)
                    ],
                    axis=0,
                )
                val_pred_columns.append(val_svd_krr)
                final_pred_by_target[target][svd_kernel_name] = full_svd_krr
                for row_index, pred in zip(val_idx, val_svd_krr, strict=True):
                    holdout_rows.append(
                        {
                            "row_index": int(row_index),
                            "target_type": target,
                            "target": float(train.loc[row_index, "target"]),
                            "model": svd_kernel_name,
                            "prediction": float(pred),
                        }
                    )

        val_matrix = np.column_stack(val_pred_columns)
        blend_sample_weight = None
        if args.density_weighted:
            blend_sample_weight = density_sample_weight(
                y_all[val_idx],
                n_bins=args.density_weight_bins,
                power=args.density_weight_power,
                max_weight=args.density_weight_max,
            )
        weights, intercept = fit_blend(y_all[val_idx], val_matrix, sample_weight=blend_sample_weight)
        blend_weights[target] = {
            "model_order": spec_names,
            "weights": {name: float(weight) for name, weight in zip(spec_names, weights, strict=True)},
            "intercept": float(intercept),
        }
        blend_val = val_matrix @ weights + intercept
        for row_index, pred in zip(val_idx, blend_val, strict=True):
            holdout_rows.append(
                {
                    "row_index": int(row_index),
                    "target_type": target,
                    "target": float(train.loc[row_index, "target"]),
                    "model": "nnls_blend",
                    "prediction": float(pred),
                }
            )

    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(run_dir / "holdout_predictions_long.csv", index=False)
    metrics_by_model = {}
    for model_name, sub in holdout.groupby("model"):
        metrics_by_model[model_name] = score_frame(sub[["target_type", "target", "prediction"]].copy())
    pd.DataFrame(
        [
            {
                "model": name,
                "combined_r2": item["combined_r2"],
                "Tg_r2": item["by_target"]["Tg"]["r2"],
                "Egc_r2": item["by_target"]["Egc"]["r2"],
                "Tg_mae": item["by_target"]["Tg"]["mae"],
                "Egc_mae": item["by_target"]["Egc"]["mae"],
            }
            for name, item in metrics_by_model.items()
        ]
    ).sort_values("combined_r2", ascending=False).to_csv(run_dir / "holdout_metrics.csv", index=False)

    final = test[["id", "target_type"]].copy()
    final["target"] = np.nan
    final_details = []
    for target in TARGETS:
        local = test.index[test["target_type"] == target].to_numpy(dtype=np.int64)
        weights = np.array([blend_weights[target]["weights"][name] for name in spec_names], dtype=np.float64)
        intercept = float(blend_weights[target]["intercept"])
        matrix = np.column_stack([final_pred_by_target[target][name] for name in spec_names])
        pred = matrix @ weights + intercept
        overrides = same_target_test_overrides(train, test, target)
        override_map = {int(row.id): float(row.override_value) for row in overrides.itertuples(index=False)}
        override_count_map = {int(row.id): int(row.override_count) for row in overrides.itertuples(index=False)}
        if override_map:
            pred = pred.copy()
            local_id_to_pos = {int(test_id): pos for pos, test_id in enumerate(test.loc[local, "id"])}
            for test_id, value in override_map.items():
                if test_id in local_id_to_pos:
                    pred[local_id_to_pos[test_id]] = value
        override_reports[target] = {
            "same_target_no_stereo_test_rows_overridden": int(len(override_map)),
            "ids": sorted(override_map),
        }
        final.loc[local, "target"] = pred
        for row_index, test_id, value in zip(local, test.loc[local, "id"], pred, strict=True):
            detail = {
                "test_index": int(row_index),
                "id": int(test_id),
                "target_type": target,
                "prediction": float(value),
                "same_target_official_override": bool(int(test_id) in override_map),
                "same_target_official_override_count": int(override_count_map.get(int(test_id), 0)),
            }
            for name, column in zip(spec_names, matrix.T, strict=True):
                detail[name] = float(column[list(local).index(row_index)])
            final_details.append(detail)
    if final["target"].isna().any() or not np.isfinite(final["target"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("final predictions are incomplete or non-finite")

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"Sandman_polymer_LOCAL_OFFICIAL_LOOP_{run_name}.csv"
    final[["id", "target"]].to_csv(submission_path, index=False)
    pd.DataFrame(final_details).sort_values("id").to_csv(run_dir / "test_predictions_detail.csv", index=False)

    external_label_report = None
    if args.external_labels is not None:
        external_label_report = validate_submission(submission_path, args.external_labels, run_dir)
    report = {
        "schema_version": "local.polymer.official_train_eval_loop.v1",
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "rule_boundary": {
            "training_inputs": ["Polymer Prediction Challenge/aisehack-2-0/train.csv"],
            "inference_inputs": ["Polymer Prediction Challenge/aisehack-2-0/test.csv"],
            "external_external_labels_loaded_after_submission_path": repo_relative(submission_path) if args.external_labels is not None else None,
            "external_external_labels_training_use": False,
        },
        "inputs": {
            "train": {"path": repo_relative(DATA_DIR / "train.csv"), "sha256": sha256_file(DATA_DIR / "train.csv"), "rows": int(len(train))},
            "test": {"path": repo_relative(DATA_DIR / "test.csv"), "sha256": sha256_file(DATA_DIR / "test.csv"), "rows": int(len(test))},
            "external_labels": None
            if args.external_labels is None
            else {"path": repo_relative(args.external_labels), "sha256": sha256_file(args.external_labels), "training_use": False},
        },
        "configuration": {
            "seed": int(args.seed),
            "holdout_fraction": float(args.holdout_fraction),
            "full_fold_ensemble": int(args.full_fold_ensemble),
            "n_bits": int(args.n_bits),
            "text_features": int(args.text_features),
            "motif_hash_features": int(args.motif_hash_features),
            "quick": bool(args.quick),
            "rich_features": bool(args.rich_features),
            "periodic_features": bool(args.periodic_features),
            "periodic_sparse_only": bool(args.periodic_sparse_only),
            "capped_dense_features": bool(args.capped_dense_features),
            "motif_features": bool(args.motif_features),
            "backbone_sidechain_features": bool(args.backbone_sidechain_features),
            "conjugation_features": bool(args.conjugation_features),
            "mobility_features": bool(args.mobility_features),
            "huckel_features": bool(args.huckel_features),
            "electronic_tail_features": bool(args.electronic_tail_features),
            "topological_autocorr_features": bool(args.topological_autocorr_features),
            "topological_autocorr_max_distance": int(args.topological_autocorr_max_distance),
            "infinite_chain_features": bool(args.infinite_chain_features),
            "bicerano_features": bool(args.bicerano_features),
            "map4_features": bool(args.map4_features),
            "map4_hash_features": int(args.map4_hash_features),
            "map4_max_distance": int(args.map4_max_distance),
            "map4_env_radius": int(args.map4_env_radius),
            "region_sparse_features": bool(args.region_sparse_features),
            "region_sparse_hash_features": int(args.region_sparse_hash_features),
            "endpoint_path_sparse_features": bool(args.endpoint_path_sparse_features),
            "endpoint_path_hash_features": int(args.endpoint_path_hash_features),
            "endpoint_path_max_bonds": int(args.endpoint_path_max_bonds),
            "rooted_smiles_features": bool(args.rooted_smiles_features),
            "rooted_smiles_max_roots": int(args.rooted_smiles_max_roots),
            "rooted_smiles_text_features": None if args.rooted_smiles_text_features is None else int(args.rooted_smiles_text_features),
            "random_smiles_features": bool(args.random_smiles_features),
            "random_smiles_augmentations": int(args.random_smiles_augmentations),
            "random_smiles_seed": int(args.random_smiles_seed),
            "random_smiles_text_features": None if args.random_smiles_text_features is None else int(args.random_smiles_text_features),
            "kekule_smiles_features": bool(args.kekule_smiles_features),
            "kekule_smiles_text_features": None if args.kekule_smiles_text_features is None else int(args.kekule_smiles_text_features),
            "exact_sparse_features": bool(args.exact_sparse_features),
            "exact_sparse_radii": args.exact_sparse_radii,
            "exact_sparse_block_names": exact_block_names,
            "wl_sparse_features": bool(args.wl_sparse_features),
            "wl_iterations": int(args.wl_iterations),
            "physics_features": bool(args.physics_features),
            "mordred_features": bool(args.mordred_features),
            "oligomer_features": bool(args.oligomer_features),
            "oligomer_repeats": int(args.oligomer_repeats),
            "oligomer_slope_features": bool(args.oligomer_slope_features),
            "oligomer_slope_max_repeats": int(args.oligomer_slope_max_repeats),
            "oligomer_slope_transform": args.oligomer_slope_transform,
            "oligomer_ffox_features": bool(args.oligomer_ffox_features),
            "oligomer_ffox_max_repeats": int(args.oligomer_ffox_max_repeats),
            "oligomer_ffox_transform": args.oligomer_ffox_transform,
            "oligomer_3d_features": bool(args.oligomer_3d_features),
            "oligomer_3d_repeats": args.oligomer_3d_repeats,
            "conformers_per_mol": int(args.conformers_per_mol),
            "conformer_pooling": args.conformer_pooling,
            "oligomer_3d_extended": not args.no_oligomer_3d_extended,
            "oligomer_mordred_features": bool(args.oligomer_mordred_features),
            "rdkit_3d_features": bool(args.rdkit_3d_features),
            "conformer_seed": int(args.conformer_seed),
            "conformer_opt_steps": int(args.conformer_opt_steps),
            "select_k": None if args.select_k is None else int(args.select_k),
            "extra_trees": bool(args.extra_trees),
            "lgbm_quantile": bool(args.lgbm_quantile),
            "lgbm_quantile_alphas": [float(alpha) for alpha in quantile_alphas],
            "ordinal_classifier": bool(args.ordinal_classifier),
            "target_transform_models": bool(args.target_transform_models),
            "robust_linear_models": bool(args.robust_linear_models),
            "catboost_models": bool(args.catboost_models),
            "density_weighted": bool(args.density_weighted),
            "density_weight_bins": int(args.density_weight_bins),
            "density_weight_power": float(args.density_weight_power),
            "density_weight_max": float(args.density_weight_max),
            "duplicate_robust_training": bool(args.duplicate_robust_training),
            "duplicate_median_shrink": float(args.duplicate_median_shrink),
            "duplicate_count_weight_power": float(args.duplicate_count_weight_power),
            "duplicate_mad_weight_power": float(args.duplicate_mad_weight_power),
            "duplicate_weight_max": float(args.duplicate_weight_max),
            "count_tanimoto_krr": bool(args.count_tanimoto_krr),
            "count_krr_alpha": float(args.count_krr_alpha),
            "tanimoto_svr": bool(args.tanimoto_svr),
            "svr_c": float(args.svr_c),
            "svr_epsilon": float(args.svr_epsilon),
            "svd_kernel_krr": bool(args.svd_kernel_krr),
            "svd_krr_kernels": list(svd_kernels),
            "svd_krr_components": int(args.svd_krr_components),
            "svd_krr_alpha": float(args.svd_krr_alpha),
            "svd_kernel_members": svd_kernel_names,
            "sparse_block_names": sparse_block_names,
            "krr_bit_block_names": bit_block_names,
            "count_krr_block_names": count_kernel_block_names,
            "model_specs": [{"name": spec.name, "family": spec.family, "params": spec.params} for spec in specs],
            "krr_alpha": float(args.krr_alpha),
            "blend": "target-wise NNLS with official train holdout labels only",
        },
        "features": {
            "base_dense_feature_count": len(features["dense_names"]),
            "feature_reports": features.get("feature_reports", {}),
            "target_specific_lookup_features": lookup_feature_reports,
            "duplicate_robust_training_reports": duplicate_robust_reports,
            "same_target_final_overrides": override_reports,
            "sparse_blocks": {name: {"shape": list(block.shape), "nnz": int(block.nnz)} for name, block in blocks.items()},
        },
        "holdout_metrics_by_model": metrics_by_model,
        "blend_weights": blend_weights,
        "artifacts": {
            "run_dir": repo_relative(run_dir),
            "holdout_predictions_long": repo_relative(run_dir / "holdout_predictions_long.csv"),
            "holdout_metrics": repo_relative(run_dir / "holdout_metrics.csv"),
            "test_predictions_detail": repo_relative(run_dir / "test_predictions_detail.csv"),
            "submission": {"path": repo_relative(submission_path), "sha256": sha256_file(submission_path), "rows": int(len(final))},
            "external_label_validation_report": None if external_label_report is None else repo_relative(run_dir / "external_label_validation_report.json"),
        },
        "external_label_validation": external_label_report,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "catboost": None if cb is None else cb.__version__,
        },
        "method_notes": [
            "Web search found recent polymer solutions leaning on multi-view ensembles, GNNs, fingerprints, SMILES augmentation, and pretrained language models.",
            "This run keeps only the rule-compatible pieces: RDKit descriptors, Morgan/MACCS/atom-pair/topological-torsion fingerprints, optional official-only periodic-closure features, character n-grams, boosters, ridge, and Tanimoto KRR trained from official labels.",
            "Pretrained encoders, external-label augmentation, and externally fitted assets are intentionally excluded.",
        ]
        + (
            [
                "Duplicate/noise-robust training, when enabled, computes within-fold official-train canonical-SMILES target medians and sample weights from the training slice only; validation external_labels are loaded only after the submission CSV is written."
            ]
            if args.duplicate_robust_training
            else []
        )
        + (
            [
                "Full-test fold ensembling, when enabled, averages target-wise predictions from multiple official-train KFold fits; no validation external_labels or test labels enter fold selection or fitting."
            ]
            if int(args.full_fold_ensemble) > 1
            else []
        ),
    }
    write_json(run_dir / "run_report.json", report)
    print(
        json.dumps(
            {
                "run_dir": repo_relative(run_dir),
                "submission": repo_relative(submission_path),
                "holdout_best": max(metrics_by_model.items(), key=lambda item: item[1]["combined_r2"]),
                "external_label_validation": None if external_label_report is None else external_label_report["score"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return report


def read_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def validate_submission(submission_path: Path, external_labels_path: Path, run_dir: Path) -> dict[str, Any]:
    external_labels = pd.read_csv(external_labels_path)
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(DATA_DIR / "test.csv")
    external_labels = external_labels.rename(columns={"target": "external_label"})
    submission = submission.rename(columns={"target": "prediction"})
    merged = test.merge(external_labels[["id", "external_label"]], on="id", how="left", validate="one_to_one").merge(
        submission[["id", "prediction"]], on="id", how="left", validate="one_to_one"
    )
    merged["target_type"] = merged["target_type"].map(canonical_target)
    merged["external_label"] = merged["external_label"].map(read_float)
    merged["prediction"] = merged["prediction"].map(read_float)
    scored = merged[merged["external_label"].notna() & merged["prediction"].notna()].copy()
    score: dict[str, Any] = {"by_target": {}}
    target_scores = []
    for target in TARGETS:
        sub = scored[scored["target_type"] == target]
        y = sub["external_label"].to_numpy(dtype=np.float64)
        pred = sub["prediction"].to_numpy(dtype=np.float64)
        r2 = float(r2_score(y, pred)) if len(sub) >= 2 else None
        if r2 is not None:
            target_scores.append(r2)
        score["by_target"][target] = {
            "rows": int(len(sub)),
            "r2": r2,
            "mae": float(mean_absolute_error(y, pred)) if len(sub) else None,
            "rmse": float(np.sqrt(np.mean(np.square(y - pred)))) if len(sub) else None,
            "bias": float(np.mean(pred - y)) if len(sub) else None,
        }
    score["combined_r2"] = float(np.mean(target_scores)) if len(target_scores) == len(TARGETS) else None

    scored["abs_error"] = (scored["prediction"] - scored["external_label"]).abs()
    scored["signed_error"] = scored["prediction"] - scored["external_label"]
    scored[["id", "target_type", "external_label", "prediction", "signed_error", "abs_error"]].to_csv(run_dir / "external_label_validation_rows.csv", index=False)
    slices = []
    for target in TARGETS:
        sub = scored[scored["target_type"] == target].copy()
        if sub.empty:
            continue
        sub["smiles_len_bin"] = pd.qcut(sub["smiles"].astype(str).str.len(), q=min(4, len(sub)), duplicates="drop")
        sub["external_label_bin"] = pd.qcut(sub["external_label"], q=min(4, len(sub)), duplicates="drop")
        for field in ("smiles_len_bin", "external_label_bin"):
            for key, group in sub.groupby(field, observed=True):
                if len(group) < 2:
                    continue
                slices.append(
                    {
                        "target_type": target,
                        "slice_field": field,
                        "slice": str(key),
                        "rows": int(len(group)),
                        "r2": float(r2_score(group["external_label"], group["prediction"])),
                        "mae": float(mean_absolute_error(group["external_label"], group["prediction"])),
                    }
                )
    pd.DataFrame(slices).to_csv(run_dir / "external_label_validation_slices.csv", index=False)
    report = {
        "external_labels": {
            "path": repo_relative(external_labels_path),
            "sha256": sha256_file(external_labels_path),
            "rows": int(len(external_labels)),
            "external_labeled_rows": int(external_labels["external_label"].map(read_float).notna().sum()),
            "training_use": False,
        },
        "submission": {"path": repo_relative(submission_path), "sha256": sha256_file(submission_path), "rows": int(len(submission))},
        "score": score,
        "artifacts": {
            "rows": repo_relative(run_dir / "external_label_validation_rows.csv"),
            "slices": repo_relative(run_dir / "external_label_validation_slices.csv"),
        },
        "warning": "Validation-only external external_labels. Do not use for training, fitted state, calibration, copied predictions, or submission construction.",
    }
    write_json(run_dir / "external_label_validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--full-fold-ensemble", type=int, default=1, help="Average full-test predictions from N train-only KFold models per target; 1 uses the full target train rows once.")
    parser.add_argument("--n-bits", type=int, default=8192)
    parser.add_argument("--text-features", type=int, default=2**18)
    parser.add_argument("--motif-hash-features", type=int, default=16384)
    parser.add_argument("--krr-alpha", type=float, default=0.01)
    parser.add_argument("--external_labels", type=Path, default=None, help="Optional external_labels CSV for post-write validation only. Omit for clean train/test-only generation.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rich-features", action="store_true")
    parser.add_argument("--periodic-features", action="store_true")
    parser.add_argument("--periodic-sparse-only", action="store_true")
    parser.add_argument("--capped-dense-features", action="store_true", help="Append RDKit/physics descriptors on explicit-hydrogen-capped official polymer SMILES.")
    parser.add_argument("--motif-features", action="store_true", help="Append QSPR/GAP-inspired explicit motif descriptors plus hashed BRICS/path motifs from official SMILES only.")
    parser.add_argument("--backbone-sidechain-features", action="store_true", help="Append PolyMetriX-style shortest-backbone and off-backbone side-chain descriptors from official SMILES only.")
    parser.add_argument("--conjugation-features", action="store_true", help="Append Egc-oriented conjugation, fused-ring, and donor/acceptor path descriptors from official SMILES only.")
    parser.add_argument("--mobility-features", action="store_true", help="Append effective atomic-mobility, side-chain mass, and rigid/flexible Tg proxy descriptors from official SMILES only.")
    parser.add_argument("--huckel-features", action="store_true", help="Append Huckel-style pi-graph spectral and periodic endpoint-closure descriptors from official SMILES only.")
    parser.add_argument("--electronic-tail-features", action="store_true", help="Append explicit low-gap acceptor SMARTS and ordered donor/acceptor endpoint-path descriptors from official SMILES only.")
    parser.add_argument("--topological-autocorr-features", action="store_true", help="Append graph-distance autocorrelation descriptors over charge, atom, donor, acceptor, hetero, and aromatic flags.")
    parser.add_argument("--topological-autocorr-max-distance", type=int, default=8, help="Maximum graph distance for --topological-autocorr-features.")
    parser.add_argument("--infinite-chain-features", action="store_true", help="Append compact infinite-chain proxy ratios over repeat-core mass, backbone path, sidechain bulk, electronic density, and periodic closure graph.")
    parser.add_argument("--bicerano-features", action="store_true", help="Append public-code Bicerano-style group-contribution descriptors computed from official SMILES only.")
    parser.add_argument("--map4-features", action="store_true", help="Append dependency-free MAP4-like hashed atom-environment pair fingerprints from official SMILES only.")
    parser.add_argument("--map4-hash-features", type=int, default=131072, help="Hash width for --map4-features.")
    parser.add_argument("--map4-max-distance", type=int, default=12, help="Maximum topological atom-pair distance for --map4-features.")
    parser.add_argument("--map4-env-radius", type=int, default=1, help="Atom-environment radius for --map4-features.")
    parser.add_argument("--region-sparse-features", action="store_true", help="Append region-specific sparse kernels for endpoint-backbone and side-chain fragments from official SMILES only.")
    parser.add_argument("--region-sparse-hash-features", type=int, default=32768, help="Hash width for region MAP4-like sparse kernels.")
    parser.add_argument("--endpoint-path-sparse-features", action="store_true", help="Append sparse atom/bond n-grams along the polymer endpoint path from official SMILES only.")
    parser.add_argument("--endpoint-path-hash-features", type=int, default=32768, help="Hash width for --endpoint-path-sparse-features.")
    parser.add_argument("--endpoint-path-max-bonds", type=int, default=8, help="Maximum endpoint-path n-gram width in bonds.")
    parser.add_argument("--rooted-smiles-features", action="store_true", help="Append deterministic rooted noncanonical SMILES hashed char n-grams averaged per official molecule.")
    parser.add_argument("--rooted-smiles-max-roots", type=int, default=16, help="Maximum rooted SMILES enumerations per molecule for --rooted-smiles-features.")
    parser.add_argument("--rooted-smiles-text-features", type=int, help="Hash width for --rooted-smiles-features; defaults to --text-features.")
    parser.add_argument("--random-smiles-features", action="store_true", help="Append deterministic random noncanonical SMILES hashed char n-grams averaged per official molecule.")
    parser.add_argument("--random-smiles-augmentations", type=int, default=16, help="Requested canonical-plus-random SMILES variants per molecule for --random-smiles-features.")
    parser.add_argument("--random-smiles-seed", type=int, default=20260722, help="Base random seed for --random-smiles-features.")
    parser.add_argument("--random-smiles-text-features", type=int, help="Hash width for --random-smiles-features; defaults to --text-features.")
    parser.add_argument("--kekule-smiles-features", action="store_true", help="Append deterministic canonical kekulized SMILES hashed char n-grams from official SMILES only.")
    parser.add_argument("--kekule-smiles-text-features", type=int, help="Hash width for --kekule-smiles-features; defaults to --text-features.")
    parser.add_argument("--exact-sparse-features", action="store_true", help="Append unfolded Morgan count fingerprint dictionaries vectorized only on target/fold official train rows.")
    parser.add_argument("--exact-sparse-radii", default="1,2,3", help="Comma-separated Morgan radii for --exact-sparse-features.")
    parser.add_argument("--wl-sparse-features", action="store_true", help="Append Weisfeiler-Lehman subtree count dictionaries vectorized only on target/fold official train rows.")
    parser.add_argument("--wl-iterations", type=int, default=3, help="WL refinement iterations for --wl-sparse-features.")
    parser.add_argument("--physics-features", action="store_true")
    parser.add_argument("--mordred-features", action="store_true", help="Append Mordred 2D descriptors computed from official SMILES only.")
    parser.add_argument("--oligomer-features", action="store_true", help="Append deterministic linear oligomer RDKit descriptors/fingerprints from official polymer SMILES only.")
    parser.add_argument("--oligomer-repeats", type=int, default=2, help="Repeat count for --oligomer-features; 2 is the PolyMon-style dimer default.")
    parser.add_argument("--oligomer-slope-features", action="store_true", help="Append linear intercept/slope descriptors over deterministic 1..N official-SMILES oligomers.")
    parser.add_argument("--oligomer-slope-max-repeats", type=int, default=4, help="Maximum repeat count for --oligomer-slope-features.")
    parser.add_argument(
        "--oligomer-slope-transform",
        choices=("raw", "signed_log", "both"),
        default="raw",
        help="Transform for N-mer slope descriptors. signed_log reduces finite-value overflow from very large oligomer descriptors.",
    )
    parser.add_argument("--oligomer-ffox-features", action="store_true", help="Append Flory-Fox-style intensive n-mer descriptors: per-heavy-atom monomer/dimer/trimer descriptors extrapolated against 1/n.")
    parser.add_argument("--oligomer-ffox-max-repeats", type=int, default=3, help="Maximum repeat count for --oligomer-ffox-features; 3 gives monomer/dimer/trimer infinite-chain estimates.")
    parser.add_argument(
        "--oligomer-ffox-transform",
        choices=("raw", "signed_log", "both"),
        default="raw",
        help="Transform for Flory-Fox asymptotic descriptors. raw keeps intercept/slope/inf3 on per-heavy-atom descriptors.",
    )
    parser.add_argument("--oligomer-3d-features", action="store_true", help="Append pooled optimized RDKit 3D descriptors over deterministic official-SMILES oligomers.")
    parser.add_argument("--oligomer-3d-repeats", default="2,3", help="Comma-separated repeat counts for --oligomer-3d-features.")
    parser.add_argument("--conformers-per-mol", type=int, default=1, help="Number of ETKDG conformers per molecule for pooled oligomer 3D descriptors.")
    parser.add_argument("--conformer-pooling", default="mean,std", help="Comma-separated pooling modes for oligomer 3D descriptors: mean,std,min,max.")
    parser.add_argument("--no-oligomer-3d-extended", action="store_true", help="Use only the 11 scalar RDKit 3D shape descriptors for oligomer 3D instead of WHIM/GETAWAY/MORSE/RDF/AUTOCORR3D too.")
    parser.add_argument("--oligomer-mordred-features", action="store_true", help="Append Mordred 2D descriptors on the deterministic oligomer; slower and official-SMILES-only.")
    parser.add_argument("--rdkit-3d-features", action="store_true", help="Append deterministic RDKit ETKDG 3D shape descriptors computed from official SMILES only.")
    parser.add_argument("--conformer-seed", type=int, default=20260721, help="Base random seed for deterministic RDKit ETKDG conformer generation.")
    parser.add_argument("--conformer-opt-steps", type=int, default=50, help="UFF optimization iterations for RDKit 3D descriptor conformers; use 0 to skip optimization.")
    parser.add_argument("--select-k", type=int)
    parser.add_argument("--extra-trees", action="store_true", help="Append train-only ExtraTreesRegressor members to the sparse model roster.")
    parser.add_argument("--lgbm-quantile", action="store_true", help="Append train-only LightGBM quantile-objective members; final NNLS weights are fitted only on official train holdout labels.")
    parser.add_argument("--lgbm-quantile-alphas", default="0.35,0.5,0.65", help="Comma-separated quantile alphas for --lgbm-quantile, e.g. 0.5,0.85,0.9.")
    parser.add_argument("--ordinal-classifier", action="store_true", help="Append train-only LightGBM ordinal expected-value classifier members for tail/extreme robustness.")
    parser.add_argument("--target-transform-models", action="store_true", help="Append train-only Yeo-Johnson/rank-normal transformed-target regressors with inverse-transformed predictions.")
    parser.add_argument("--robust-linear-models", action="store_true", help="Append sparse SGD/ElasticNet additive heads for QSPR/GAP-style fragment signal.")
    parser.add_argument("--catboost-models", action="store_true", help="Append train-only CatBoostRegressor members when catboost is installed.")
    parser.add_argument("--density-weighted", action="store_true", help="Use train-only inverse target-density sample weights for eligible regressors and the holdout NNLS blend.")
    parser.add_argument("--density-weight-bins", type=int, default=24)
    parser.add_argument("--density-weight-power", type=float, default=0.5)
    parser.add_argument("--density-weight-max", type=float, default=6.0)
    parser.add_argument(
        "--duplicate-robust-training",
        action="store_true",
        help="Use train-slice-only canonical duplicate medians and duplicate-noise weights when fitting eligible members.",
    )
    parser.add_argument("--duplicate-median-shrink", type=float, default=1.0, help="0 keeps raw labels; 1 fully replaces duplicate-group labels with the group median.")
    parser.add_argument("--duplicate-count-weight-power", type=float, default=0.5, help="Downweight repeated official-train canonical groups by count**(-power).")
    parser.add_argument("--duplicate-mad-weight-power", type=float, default=1.0, help="Downweight noisy duplicate canonical groups by robust within-group MAD.")
    parser.add_argument("--duplicate-weight-max", type=float, default=4.0, help="Clip duplicate robust sample weights to [1/max, max] before renormalizing.")
    parser.add_argument("--count-tanimoto-krr", action="store_true", help="Append train-only generalized Tanimoto KRR over count fingerprint blocks.")
    parser.add_argument("--count-krr-alpha", type=float, default=0.03)
    parser.add_argument("--tanimoto-svr", action="store_true", help="Append a train-only precomputed Tanimoto-kernel SVR member over bit fingerprints.")
    parser.add_argument("--svr-c", type=float, default=10.0)
    parser.add_argument("--svr-epsilon", type=float, default=0.03)
    parser.add_argument("--svd-kernel-krr", action="store_true", help="Append train-only KRR members on fold-local TruncatedSVD latent descriptor features.")
    parser.add_argument("--svd-krr-kernels", default="laplacian,rbf", help="Comma-separated kernels for --svd-kernel-krr: laplacian,rbf.")
    parser.add_argument("--svd-krr-components", type=int, default=256)
    parser.add_argument("--svd-krr-alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
