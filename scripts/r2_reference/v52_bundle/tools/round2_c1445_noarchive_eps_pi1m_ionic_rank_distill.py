#!/usr/bin/env python3
"""C1445 noarchive EPS ionic-residual PI1M rank-distillation pilot.

Official-input-only local experiment:

* current Round 2 train/test labels only;
* optional PI1M unlabeled SMILES, pseudo-labeled inside this run by a
  fold-local teacher;
* no local_eval, external_label file, archive labels, pretrained weights, Kaggle action,
  or external data;
* replaces only EPS in a branch-local base CSV after the candidate is frozen.

The experiment tests a distinct mechanism from the cooled PI1M representation
branches: supervised teacher-generated ranking/soft-label distillation of the
ionic dielectric residual eps - nc_hat**2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as current_only


RDLogger.DisableLog("rdApp.*")

SEED = 20260808
TARGET = "eps"
PARTNER = "nc"
MIN_IONIC = 0.02
EXPECTED_PI1M_SHA = "c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8"

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
    "carbonate": "OC(=O)O",
    "sulfide_aromatic": "a[#16]a",
    "fluoro_aromatic": "aF",
}
POLAR_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in POLAR_SMARTS.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def guard_path(path: Path, role: str) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if role in {"data_dir", "train", "test", "pi1m"} and ("/archive/" in low or "with_archive" in low):
        raise RuntimeError(f"Refusing archive/cross-branch {role} path for noarchive experiment: {path}")
    if role == "base" and ("with_archive" in low or "/archive/" in low):
        raise RuntimeError(f"Refusing archive/cross-branch base for noarchive experiment: {path}")


def canonical(smiles: object) -> str | None:
    try:
        return reference.canonicalize(str(smiles))
    except Exception:
        return None


def no_stereo_group(canon: str) -> str:
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return canon
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def stable_rank_key(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def read_pi1m_subset(pi1m_path: Path, limit: int) -> tuple[list[str], dict[str, Any]]:
    if sha256_file(pi1m_path) != EXPECTED_PI1M_SHA:
        raise RuntimeError("PI1M hash mismatch")
    frame = pd.read_csv(pi1m_path, usecols=["SMILES"])
    frame["rank"] = [stable_rank_key(value) for value in frame["SMILES"].astype(str)]
    selected = frame.sort_values("rank", kind="mergesort").head(int(limit))
    canons: list[str] = []
    invalid = 0
    seen: set[str] = set()
    for value in selected["SMILES"].astype(str):
        canon = canonical(value)
        if canon is None:
            invalid += 1
            continue
        if canon in seen:
            continue
        seen.add(canon)
        canons.append(canon)
    return canons, {
        "pi1m_rows_read": int(len(frame)),
        "hash_rank_limit": int(limit),
        "selected_raw_rows": int(len(selected)),
        "valid_unique_canonical": int(len(canons)),
        "invalid_selected_rows": int(invalid),
        "sha256": sha256_file(pi1m_path),
    }


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base CSV schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Base CSV ID order mismatch: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base CSV contains non-finite predictions: {path}")
    return frame


def morgan_count_dense(molecules: list[Any], radius: int, bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    out = np.zeros((len(molecules), bits), dtype=np.float32)
    for row, mol in enumerate(molecules):
        fp = generator.GetCountFingerprint(mol)
        for col, count in fp.GetNonzeroElements().items():
            out[row, int(col)] = math.log1p(float(count))
    return out


def feature_matrix(canons: list[str], morgan_bits: int) -> tuple[np.ndarray, list[str], dict[str, int]]:
    names = list(POLAR_PATTERNS.keys()) + [
        "heavy_atoms",
        "dummy_atoms",
        "rings",
        "aromatic_rings",
        "aromatic_atom_fraction",
        "hetero_fraction",
        "halogen_fraction",
        "n_fraction",
        "o_fraction",
        "s_fraction",
        "si_fraction",
        "rotatable_per_heavy",
        "double_bond_per_heavy",
        "triple_bond_per_heavy",
        "branch_per_heavy",
        "tpsa_per_heavy",
        "h_donors_per_heavy",
        "h_acceptors_per_heavy",
        "fraction_csp3",
        "mol_mr_per_heavy",
        "mol_logp_per_heavy",
    ]
    rows: list[list[float]] = []
    molecules: list[Any] = []
    invalid = 0
    for canon in canons:
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            invalid += 1
            mol = Chem.MolFromSmiles("*")
        assert mol is not None
        molecules.append(mol)
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        polar = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in POLAR_PATTERNS.values()]
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in atoms)
        hetero = sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)
        halogen = sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)
        row = polar + [
            float(mol.GetNumHeavyAtoms()),
            float(sum(atom.GetAtomicNum() == 0 for atom in atoms)),
            float(mol.GetRingInfo().NumRings()),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(aromatic_atoms / heavy),
            float(hetero / heavy),
            float(halogen / heavy),
            float(sum(atom.GetAtomicNum() == 7 for atom in atoms) / heavy),
            float(sum(atom.GetAtomicNum() == 8 for atom in atoms) / heavy),
            float(sum(atom.GetAtomicNum() == 16 for atom in atoms) / heavy),
            float(sum(atom.GetAtomicNum() == 14 for atom in atoms) / heavy),
            float(Descriptors.NumRotatableBonds(mol) / heavy),
            float(sum(bond.GetBondTypeAsDouble() == 2.0 for bond in bonds) / heavy),
            float(sum(bond.GetBondTypeAsDouble() == 3.0 for bond in bonds) / heavy),
            float(str(canon).count("(") / heavy),
            float(Descriptors.TPSA(mol) / heavy),
            float(Descriptors.NumHDonors(mol) / heavy),
            float(Descriptors.NumHAcceptors(mol) / heavy),
            float(Descriptors.FractionCSP3(mol)),
            float(Crippen.MolMR(mol) / heavy),
            float(Crippen.MolLogP(mol) / heavy),
        ]
        rows.append(row)
    dense = np.asarray(rows, dtype=np.float32)
    morgan2 = morgan_count_dense(molecules, radius=2, bits=morgan_bits)
    morgan3 = morgan_count_dense(molecules, radius=3, bits=morgan_bits)
    matrix = np.hstack([dense, morgan2, morgan3]).astype(np.float32)
    all_names = names + [f"morgan2_{i}" for i in range(morgan_bits)] + [f"morgan3_{i}" for i in range(morgan_bits)]
    bad = ~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)
    if bad.any():
        matrix = matrix.copy()
        matrix[bad] = np.nan
    return matrix, all_names, {"invalid_molecules": int(invalid)}


def train_eps_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    seed: int,
    mode: str,
) -> np.ndarray:
    if mode == "teacher_extra_trees":
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=700,
                min_samples_leaf=2,
                max_features=0.70,
                random_state=seed,
                n_jobs=4,
            ),
        )
    elif mode == "student_hgb":
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                learning_rate=0.035,
                max_iter=260,
                l2_regularization=0.08,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                random_state=seed,
            ),
        )
    elif mode == "student_ridge_rank":
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            QuantileTransformer(n_quantiles=min(1000, max(10, len(y_train))), output_distribution="normal", random_state=seed),
            StandardScaler(with_mean=False),
            Ridge(alpha=30.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        )
    else:
        raise RuntimeError(f"Unknown mode: {mode}")
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_pred), dtype=np.float64)


def fit_nc_signal(
    x_all: np.ndarray,
    train_frame: pd.DataFrame,
    key_to_index: dict[str, int],
    predict_indices: np.ndarray,
    excluded_groups: set[str],
    seed: int,
) -> np.ndarray:
    nc_rows = train_frame[train_frame["target_type"] == PARTNER].copy()
    nc_rows = nc_rows[~nc_rows["group"].isin(excluded_groups)].reset_index(drop=True)
    if len(nc_rows) < 50:
        raise RuntimeError("Insufficient NC training rows after fold exclusion")
    nc_indices = np.asarray([key_to_index[value] for value in nc_rows["canonical"]], dtype=np.int64)
    y_nc = nc_rows["target"].to_numpy(float)
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            max_features=0.70,
            random_state=seed,
            n_jobs=4,
        ),
    )
    model.fit(x_all[nc_indices], y_nc)
    return np.asarray(model.predict(x_all[predict_indices]), dtype=np.float64)


def clip_eps(y_train: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    q001, q999 = np.quantile(y_train, [0.002, 0.998])
    q25, q75 = np.quantile(y_train, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y_train)), 1.0e-8)
    return np.clip(np.asarray(prediction, dtype=np.float64), max(0.0, q001 - 2.0 * margin), q999 + 2.0 * margin)


def add_nc_features(x: np.ndarray, nc_signal: np.ndarray) -> np.ndarray:
    nc = np.asarray(nc_signal, dtype=np.float64)
    ionic_basis = np.column_stack([nc, nc * nc, np.sqrt(np.maximum(nc, 0.0))])
    return np.hstack([x, ionic_basis]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pi1m-limit", type=int, default=50000)
    parser.add_argument("--morgan-bits", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    base_path = Path(args.base).resolve()
    output_path = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path, role in ((data_dir, "data_dir"), (base_path, "base"), (output_path, "output"), (run_dir, "run_dir")):
        guard_path(path, role)
    if "without_archive" not in str(output_path).lower() and "noarchive" not in output_path.name.lower():
        raise RuntimeError("Output path must remain noarchive/without_archive-scoped")
    if output_path.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    progress_path.write_text(json.dumps({"stage": "started", "created_at": datetime.now().astimezone().isoformat()}) + "\n", encoding="utf-8")

    train, test, inputs = current_only.load_current_only_inputs(data_dir)
    pi1m_path = data_dir / "PI1M.csv"
    guard_path(data_dir / "train.csv", "train")
    guard_path(data_dir / "test.csv", "test")
    guard_path(pi1m_path, "pi1m")
    if sha256_file(pi1m_path) != EXPECTED_PI1M_SHA:
        raise RuntimeError("PI1M hash mismatch")
    ids = test["id"].to_numpy(int)
    base = load_base(base_path, ids)
    base_values = base["target"].to_numpy(float)

    train["group"] = [no_stereo_group(value) for value in train["canonical"]]
    test["group"] = [no_stereo_group(value) for value in test["canonical"]]
    eps_train = (
        train[train["target_type"] == TARGET]
        .groupby(["canonical", "group"], as_index=False)
        .agg(target=("target", "median"), measurements=("target", "size"))
        .reset_index(drop=True)
    )
    if len(eps_train) < 100:
        raise RuntimeError("Insufficient EPS rows")
    test_eps = test[test["target_type"] == TARGET].copy().reset_index()

    pi1m_canons, pi1m_report = read_pi1m_subset(pi1m_path, int(args.pi1m_limit))
    official_canons = sorted(set(train["canonical"]) | set(test["canonical"]))
    all_canons = sorted(set(official_canons) | set(pi1m_canons))
    key_to_index = {key: idx for idx, key in enumerate(all_canons)}
    x_all, feature_names, feature_report = feature_matrix(all_canons, int(args.morgan_bits))
    progress_path.open("a", encoding="utf-8").write(
        json.dumps({"stage": "features_ready", "keys": len(all_canons), "features": len(feature_names), **feature_report}) + "\n"
    )

    eps_indices = np.asarray([key_to_index[value] for value in eps_train["canonical"]], dtype=np.int64)
    y_eps = eps_train["target"].to_numpy(float)
    groups = eps_train["group"].to_numpy(object)
    test_indices = np.asarray([key_to_index[value] for value in test_eps["canonical"]], dtype=np.int64)
    pi1m_indices = np.asarray([key_to_index[value] for value in pi1m_canons], dtype=np.int64)

    splitter = GroupKFold(n_splits=5)
    oof_teacher = np.full(len(y_eps), np.nan, dtype=np.float64)
    oof_student = np.full(len(y_eps), np.nan, dtype=np.float64)
    oof_rank = np.full(len(y_eps), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (tr, va) in enumerate(splitter.split(eps_indices, y_eps, groups=groups)):
        excluded = set(str(value) for value in groups[va])
        nc_train_signal = fit_nc_signal(
            x_all,
            train,
            key_to_index,
            eps_indices[tr],
            excluded_groups=excluded,
            seed=int(args.seed) + 100 * fold,
        )
        nc_val_signal = fit_nc_signal(
            x_all,
            train,
            key_to_index,
            eps_indices[va],
            excluded_groups=excluded,
            seed=int(args.seed) + 100 * fold + 1,
        )
        nc_pi1m_signal = fit_nc_signal(
            x_all,
            train,
            key_to_index,
            pi1m_indices,
            excluded_groups=excluded,
            seed=int(args.seed) + 100 * fold + 2,
        )
        x_tr = add_nc_features(x_all[eps_indices[tr]], nc_train_signal)
        x_va = add_nc_features(x_all[eps_indices[va]], nc_val_signal)
        x_pi = add_nc_features(x_all[pi1m_indices], nc_pi1m_signal)
        teacher_val = train_eps_model(x_tr, y_eps[tr], x_va, seed=int(args.seed) + fold, mode="teacher_extra_trees")
        teacher_pi = train_eps_model(x_tr, y_eps[tr], x_pi, seed=int(args.seed) + fold, mode="teacher_extra_trees")
        teacher_pi = clip_eps(y_eps[tr], teacher_pi)
        pseudo_x = np.vstack([x_tr, x_pi])
        pseudo_y = np.concatenate([y_eps[tr], teacher_pi])
        student_val = train_eps_model(pseudo_x, pseudo_y, x_va, seed=int(args.seed) + 1000 + fold, mode="student_hgb")
        rank_y = pd.Series(teacher_pi).rank(method="average").to_numpy(float) / max(float(len(teacher_pi)), 1.0)
        real_rank = pd.Series(y_eps[tr]).rank(method="average").to_numpy(float) / max(float(len(tr)), 1.0)
        rank_x = np.vstack([x_tr, x_pi])
        rank_target = np.concatenate([real_rank, rank_y])
        rank_score_val = train_eps_model(rank_x, rank_target, x_va, seed=int(args.seed) + 2000 + fold, mode="student_ridge_rank")
        order = np.argsort(rank_score_val)
        quantiles = np.linspace(0.0, 1.0, len(tr), endpoint=True)
        sorted_train_y = np.sort(y_eps[tr])
        rank_pred = np.empty(len(va), dtype=np.float64)
        rank_pred[order] = np.interp(
            np.linspace(0.0, 1.0, len(va), endpoint=True),
            quantiles,
            sorted_train_y,
        )
        oof_teacher[va] = clip_eps(y_eps[tr], teacher_val)
        oof_student[va] = clip_eps(y_eps[tr], student_val)
        oof_rank[va] = clip_eps(y_eps[tr], rank_pred)
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(va)),
                "teacher_r2": float(r2_score(y_eps[va], oof_teacher[va])),
                "student_hgb_r2": float(r2_score(y_eps[va], oof_student[va])),
                "student_rank_r2": float(r2_score(y_eps[va], oof_rank[va])),
                "student_minus_teacher": float(r2_score(y_eps[va], oof_student[va]) - r2_score(y_eps[va], oof_teacher[va])),
            }
        )
        progress_path.open("a", encoding="utf-8").write(json.dumps({"stage": "fold_done", **fold_rows[-1]}) + "\n")

    scores = {
        "teacher_extra_trees": float(r2_score(y_eps, oof_teacher)),
        "student_hgb_pi1m": float(r2_score(y_eps, oof_student)),
        "student_rank_pi1m": float(r2_score(y_eps, oof_rank)),
    }
    best_name = max(scores, key=scores.get)
    best_oof = {"teacher_extra_trees": oof_teacher, "student_hgb_pi1m": oof_student, "student_rank_pi1m": oof_rank}[best_name]
    student_delta = scores["student_hgb_pi1m"] - scores["teacher_extra_trees"]
    rank_delta = scores["student_rank_pi1m"] - scores["teacher_extra_trees"]
    positive_student_folds = int(sum(row["student_minus_teacher"] > 0.0 for row in fold_rows))
    gate = {
        "best_model": best_name,
        "best_oof_r2": float(scores[best_name]),
        "teacher_oof_r2": float(scores["teacher_extra_trees"]),
        "student_hgb_minus_teacher": float(student_delta),
        "student_rank_minus_teacher": float(rank_delta),
        "positive_student_hgb_folds": positive_student_folds,
        "pi1m_beats_teacher_by_0p01": bool(max(student_delta, rank_delta) >= 0.01),
        "student_positive_4_of_5": bool(positive_student_folds >= 4),
        "kill_gate_clean_eps_gain_lt_0p01": bool(max(student_delta, rank_delta) < 0.01),
    }

    # Full-data fit for a frozen post-freeze diagnostic candidate.  If the PI1M
    # student missed the gate, we still materialize the best official-only EPS
    # arm for one post-freeze transfer measurement, but the manifest marks it
    # research-only and non-promoted.
    nc_eps_signal = fit_nc_signal(
        x_all,
        train,
        key_to_index,
        eps_indices,
        excluded_groups=set(),
        seed=int(args.seed) + 9000,
    )
    nc_test_signal = fit_nc_signal(
        x_all,
        train,
        key_to_index,
        test_indices,
        excluded_groups=set(),
        seed=int(args.seed) + 9001,
    )
    nc_pi1m_signal = fit_nc_signal(
        x_all,
        train,
        key_to_index,
        pi1m_indices,
        excluded_groups=set(),
        seed=int(args.seed) + 9002,
    )
    full_x = add_nc_features(x_all[eps_indices], nc_eps_signal)
    test_x = add_nc_features(x_all[test_indices], nc_test_signal)
    pi_x = add_nc_features(x_all[pi1m_indices], nc_pi1m_signal)
    teacher_test = train_eps_model(full_x, y_eps, test_x, seed=int(args.seed) + 9100, mode="teacher_extra_trees")
    teacher_pi = train_eps_model(full_x, y_eps, pi_x, seed=int(args.seed) + 9101, mode="teacher_extra_trees")
    teacher_pi = clip_eps(y_eps, teacher_pi)
    student_test = train_eps_model(np.vstack([full_x, pi_x]), np.concatenate([y_eps, teacher_pi]), test_x, seed=int(args.seed) + 9200, mode="student_hgb")
    rank_y = pd.Series(teacher_pi).rank(method="average").to_numpy(float) / max(float(len(teacher_pi)), 1.0)
    real_rank = pd.Series(y_eps).rank(method="average").to_numpy(float) / max(float(len(y_eps)), 1.0)
    rank_score_test = train_eps_model(
        np.vstack([full_x, pi_x]),
        np.concatenate([real_rank, rank_y]),
        test_x,
        seed=int(args.seed) + 9300,
        mode="student_ridge_rank",
    )
    rank_order = np.argsort(rank_score_test)
    rank_prediction = np.empty(len(test_x), dtype=np.float64)
    rank_prediction[rank_order] = np.interp(
        np.linspace(0.0, 1.0, len(test_x), endpoint=True),
        np.linspace(0.0, 1.0, len(y_eps), endpoint=True),
        np.sort(y_eps),
    )
    full_predictions = {
        "teacher_extra_trees": clip_eps(y_eps, teacher_test),
        "student_hgb_pi1m": clip_eps(y_eps, student_test),
        "student_rank_pi1m": clip_eps(y_eps, rank_prediction),
    }
    selected_values = full_predictions[best_name]
    output_values = base_values.copy()
    output_values[test_eps["index"].to_numpy(int)] = selected_values
    if not np.isfinite(output_values).all():
        raise RuntimeError("Output contains non-finite predictions")
    pd.DataFrame({"id": ids, "target": output_values}).to_csv(output_path, index=False)

    oof_path = run_dir / "eps_oof_predictions.csv"
    pd.DataFrame(
        {
            "canonical": eps_train["canonical"],
            "group": eps_train["group"],
            "target": y_eps,
            "teacher_extra_trees": oof_teacher,
            "student_hgb_pi1m": oof_student,
            "student_rank_pi1m": oof_rank,
            "selected": best_oof,
        }
    ).to_csv(oof_path, index=False)
    test_pred_path = run_dir / "eps_test_predictions.csv"
    pd.DataFrame(
        {
            "id": test_eps["id"].to_numpy(int),
            "canonical": test_eps["canonical"],
            "teacher_extra_trees": full_predictions["teacher_extra_trees"],
            "student_hgb_pi1m": full_predictions["student_hgb_pi1m"],
            "student_rank_pi1m": full_predictions["student_rank_pi1m"],
            "selected": selected_values,
        }
    ).to_csv(test_pred_path, index=False)

    report = {
        "schema_version": "ppp.round2.c1445.noarchive-eps-pi1m-ionic-rank-distill.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "CLEAN_OFFICIAL_ONLY_RESEARCH_DIAGNOSTIC",
        "branch": "without_archive",
        "hypothesis": "PI1M teacher-generated ranking/soft-label distillation improves a current-only EPS ionic-residual predictor beyond the matched no-PI1M teacher.",
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "external_label_file_read": False,
        "pretrained_weights": False,
        "external_data_used": False,
        "pi1m_unlabeled_used": True,
        "prior_prediction_input": True,
        "prior_prediction_role": "base carrier for non-EPS targets only; EPS is regenerated from official current train/test/PI1M inside this run",
        "base": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "inputs": {
            **inputs,
            "PI1M.csv": {"path": str(pi1m_path), "sha256": sha256_file(pi1m_path), "bytes": pi1m_path.stat().st_size},
        },
        "rows": {
            "current_train": int(len(train)),
            "test": int(len(test)),
            "eps_train_groups": int(len(eps_train)),
            "eps_test_rows": int(len(test_eps)),
        },
        "pi1m": pi1m_report,
        "features": {
            "morgan_bits_per_radius": int(args.morgan_bits),
            "feature_count": int(len(feature_names)),
            "feature_report": feature_report,
        },
        "folds": fold_rows,
        "oof_scores": scores,
        "gate": gate,
        "decision": "pi1m_student_passed_clean_pilot_gate" if gate["pi1m_beats_teacher_by_0p01"] and gate["student_positive_4_of_5"] else "research_only_transfer_diagnostic_gate_failed",
        "output": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": int(len(output_values)), "bytes": output_path.stat().st_size},
        "artifacts": {
            "oof_predictions": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
            "eps_test_predictions": {"path": str(test_pred_path), "sha256": sha256_file(test_pred_path)},
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "elapsed_seconds": float(time.time() - started),
    }
    report_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    command_path = run_dir / "command.txt"
    write_json(report_path, report)
    write_json(
        config_path,
        {
            "seed": int(args.seed),
            "target": TARGET,
            "pi1m_limit": int(args.pi1m_limit),
            "morgan_bits": int(args.morgan_bits),
            "base": str(base_path),
            "output": str(output_path),
        },
    )
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [progress_path, oof_path, test_pred_path, report_path, config_path, command_path, output_path]
    (run_dir / "artifact_manifest.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment": run_dir.name,
                "decision": report["decision"],
                "oof_scores": scores,
                "selected_model": best_name,
                "output": report["output"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
