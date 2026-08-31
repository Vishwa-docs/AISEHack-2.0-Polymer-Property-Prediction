"""
helpers.py — shared library for the Phase 4 explainability & robustness suite.

Standalone by construction: every path resolves relative to the Phase_4 folder
itself (Dataset/, final_submissions/, outputs/). Works on the GPU standalone
layout (Phase_4_Explainability/Dataset) and on the Mac repo layout
(Phase_4_Round3_Explainability/../Dataset).

This module never touches ground-truth answer files. The two scripts that are
allowed to do so (16_khazana_verification.py, F3_oracle_sweep.py) resolve that
path themselves.
"""
from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

RDLogger.DisableLog("rdApp.*")
import warnings
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="The argument .eval_set. is deprecated")

SEED = 42
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

# ── paths (standalone: resolve from this file's location) ───────────────────
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "Dataset"
FINAL_SUB_DIR = ROOT / "final_submissions"
MLP_DIR = OUTPUT_DIR / "mlp_checkpoints"
OUT_AND_LOG = ROOT / "outputs_and_logs"

for _d in (OUTPUT_DIR, MLP_DIR, OUT_AND_LOG / "logs", OUT_AND_LOG / "output"):
    _d.mkdir(parents=True, exist_ok=True)

SMOKE = os.environ.get("PHASE4_SMOKE", "0") == "1"


def project_root() -> Path:
    return ROOT


def data_file(name: str) -> Path:
    """Locate an official data file: standalone layout first, then Mac-repo layout."""
    for base in (DATA_DIR, ROOT.parent / "Dataset"):
        p = base / name
        if p.exists():
            return p
    return DATA_DIR / name


def final_submission_file(name: str = "submission.csv") -> Path:
    for base in (FINAL_SUB_DIR, ROOT.parent / "final_submissions"):
        p = base / name
        if p.exists():
            return p
    return FINAL_SUB_DIR / name


def load_train() -> pd.DataFrame:
    return pd.read_csv(data_file("train.csv"))


def load_test() -> pd.DataFrame:
    return pd.read_csv(data_file("test.csv"))


def load_submission() -> pd.DataFrame:
    return pd.read_csv(final_submission_file("submission.csv"))


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def canonical_smiles(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return smi


def morgan_fp(smi: str, radius: int = 2, nBits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
    arr = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_descriptors(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(len(Descriptors.descList))
    d = Descriptors.CalcMolDescriptors(mol)
    return np.array(list(d.values()), dtype=float)


# ── feature pipeline (V57 Stage A stack) ─────────────────────────────────────
DEFAULT_NGRAM_RANGE = (2, 6)
DEFAULT_NGRAM_FEATURES = 8192


def featurize(smiles_list, pipe=None, canonicalize: bool = True, seed: int = SEED):
    """Build morgan + rdkit-desc + char-ngram stack.

    pipe is a dict carrying the fitted n-gram vectorizer, descriptor imputer and
    descriptor variance mask (saved inside each proxy pkl). None -> fit new.
    canonicalize=False keeps raw strings (needed for SMILES-invariance tests).
    Returns (X, pipe).
    """
    seed_all(seed)
    if pipe is None:
        pipe = {"ngram_range": DEFAULT_NGRAM_RANGE,
                "ngram_max_features": DEFAULT_NGRAM_FEATURES}
    work = [canonical_smiles(s) if canonicalize else str(s) for s in smiles_list]
    X_morgan = np.array([morgan_fp(s) for s in work])
    X_desc_raw = np.array([rdkit_descriptors(s) for s in work])
    X_desc_raw = np.where(np.isfinite(X_desc_raw), X_desc_raw, np.nan)  # RDKit can emit inf
    if "imp" not in pipe:
        pipe["imp"] = SimpleImputer(strategy="median").fit(X_desc_raw)
    X_desc = pipe["imp"].transform(X_desc_raw)
    if "desc_mask" not in pipe:
        var = np.nan_to_num(X_desc, nan=0.0).var(axis=0)
        pipe["desc_mask"] = var > 1e-8
        if int(pipe["desc_mask"].sum()) == 0:
            pipe["desc_mask"] = np.ones(X_desc.shape[1], dtype=bool)
    X_desc = X_desc[:, pipe["desc_mask"]]
    if "cv" not in pipe:
        pipe["cv"] = CountVectorizer(
            ngram_range=tuple(pipe["ngram_range"]),
            max_features=int(pipe["ngram_max_features"]),
            analyzer="char", lowercase=False)
        X_ngram = pipe["cv"].fit_transform(work).toarray()
    else:
        X_ngram = pipe["cv"].transform(work).toarray()
    X = np.hstack([X_morgan, X_desc, X_ngram])
    # Guard: RDKit can emit astronomically large values (e.g. Chi1v ~ 1e46 on
    # star-containing polymers) that overflow sklearn's internal float32 cast.
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6)   # all legitimate descriptors are < 1e4
    if "feat_names" not in pipe:
        morgan_names = [f"morgan_{i}" for i in range(X_morgan.shape[1])]
        desc_names = [n for n, keep in zip([d[0] for d in Descriptors.descList],
                                           pipe["desc_mask"]) if keep]
        ngram_names = [f"ngram_{t}" for t in pipe["cv"].get_feature_names_out()]
        pipe["feat_names"] = morgan_names + desc_names + ngram_names
    return X, pipe


def rebuild_features(smiles_list, pipe) -> np.ndarray:
    return featurize(smiles_list, pipe=pipe)[0]


def predict_ensemble(X, pkl) -> np.ndarray:
    """NNLS-weighted prediction from the last-fold Ridge/ET/LGBM models."""
    w = np.array(pkl["nnls_weights"])
    scaler, m_ridge = pkl["models"]["ridge"][-1]
    m_et = pkl["models"]["et"][-1]
    m_lgbm = pkl["models"]["lgbm"][-1]
    X2 = X.reshape(1, -1) if X.ndim == 1 else X
    p_r = m_ridge.predict(scaler.transform(X2))
    p_e = m_et.predict(X2)
    p_l = m_lgbm.predict(X2)
    return w[0] * p_r + w[1] * p_e + w[2] * p_l


def load_proxy(target: str):
    with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", "rb") as f:
        return pickle.load(f)


def oof_df(target: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / f"proxy_oof_{target}.csv")


def train_std(target: str) -> float:
    train = load_train()
    return float(train[train["target_type"] == target]["target"].std())


def save_plot(fig, name: str) -> None:
    fig.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def smoke_n(full: int, smoke_val: int) -> int:
    return smoke_val if SMOKE else full


def random_smiles(smi: str, k: int = 30):
    """Generate up to k randomized (non-canonical) SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    variants = set()
    attempts = 0
    while len(variants) < k and attempts < k * 10:
        rsmi = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
        if rsmi:
            variants.add(rsmi)
        attempts += 1
    return list(variants)


def morgan_bit_fp(smi: str, radius: int = 2, nBits: int = 1024):
    """Morgan bit vector + bitInfo for per-atom attribution."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, {}
    info = {}
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=info)
    return bv, info
