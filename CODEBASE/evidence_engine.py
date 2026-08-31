#!/usr/bin/env python3
"""evidence_engine.py — Round-3 explainability / invariance / generalization suite.

Self-contained module that generates the full evidence bundle for
the Round-3 judging themes:

  R1  Explainability  (SHAP global/local, fidelity, cross-model agreement,
                       physics-decomposed explanations, linear probes)
  R2  Polymer-invariance robustness (randomized-SMILES prediction invariance,
                       canonicalization audit, attribution invariance,
                       oligomer/chian-extension invariance, notation/stereo)
  R3  Methodology & reliability (structured CV, conformal intervals,
                       error-uncertainty correlation, applicability domain,
                       seed stability)
  R4  Proven generalization (generalization ladder, tail performance,
                       data-augmentation experiment, homologous-series demo)

It reads ONLY official data (train.csv / test.csv).  Every artifact lands in an
`outputs/` directory.  No ground-truth answer files, no external data, no pretrained weights.

Designed to run either standalone (evidence-only, proxy predictions on test)
or embedded in the final pipeline (V57 submission predictions in memory).
"""
from __future__ import annotations

import json
import os
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    import lightgbm as lgb
    LGBM_OK = True
except Exception:
    lgb = None
    LGBM_OK = False

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    torch = None
    nn = None
    TORCH_OK = False

EV_SEED = 42
EV_TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
SMOKE = os.environ.get("PHASE4_SMOKE", "0") == "1"


# ---------------------------------------------------------------------------
# helpers (self-contained, mirrors Phase-4 helpers.py)
# ---------------------------------------------------------------------------
def seed_all(seed: int = EV_SEED) -> None:
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


def ev_morgan_fp(smi: str, radius: int = 2, nBits: int = 1024) -> np.ndarray:
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


DEFAULT_NGRAM_RANGE = (2, 6)
DEFAULT_NGRAM_FEATURES = 8192


def featurize(smiles_list, pipe=None, canonicalize: bool = True, seed: int = EV_SEED):
    """morgan + rdkit-desc + char-ngram stack (V57 Stage-A family)."""
    seed_all(seed)
    if pipe is None:
        pipe = {"ngram_range": DEFAULT_NGRAM_RANGE,
                "ngram_max_features": DEFAULT_NGRAM_FEATURES}
    work = [canonical_smiles(s) if canonicalize else str(s) for s in smiles_list]
    X_morgan = np.array([ev_morgan_fp(s) for s in work])
    X_desc_raw = np.array([rdkit_descriptors(s) for s in work])
    X_desc_raw = np.where(np.isfinite(X_desc_raw), X_desc_raw, np.nan)
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
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6)
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


def save_plot(fig, name: str, out_dir: Path) -> None:
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def smoke_n(full: int, smoke_val: int) -> int:
    return smoke_val if SMOKE else full


def train_std(df: pd.DataFrame, target: str) -> float:
    return float(df[df["target_type"] == target]["target"].std())


# ---------------------------------------------------------------------------
# 01 — proxy models (Ridge + ET + LGBM, GroupKFold on canonical, NNLS blend)
# ---------------------------------------------------------------------------
def train_proxy_models(train: pd.DataFrame, out_dir: Path):
    """Train per-target proxies; save oof csvs + scores; return proxies dict."""
    seed_all(EV_SEED)
    t0 = time.time()
    all_scores = {}
    proxies = {}

    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=EV_SEED).reset_index(drop=True)
        df_t["canonical"] = df_t["smiles"].apply(canonical_smiles)
        X, pipe = featurize(df_t["smiles"].tolist())
        y = df_t["target"].values.astype(float)
        groups = df_t["canonical"].values
        n = len(df_t)
        n_splits = 5 if n >= 500 else 3
        cv = GroupKFold(n_splits=n_splits)

        oof_ridge = np.zeros(n)
        oof_et = np.zeros(n)
        oof_lgbm = np.zeros(n)
        oof_et_treespread = np.zeros(n)
        models = {"ridge": [], "et": [], "lgbm": []}

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            scaler = StandardScaler().fit(X_tr)
            m_ridge = Ridge(alpha=100, random_state=EV_SEED)
            m_ridge.fit(scaler.transform(X_tr), y_tr)
            oof_ridge[va_idx] = m_ridge.predict(scaler.transform(X_va))

            m_et = ExtraTreesRegressor(n_estimators=smoke_n(200, 30), n_jobs=-1,
                                       random_state=EV_SEED, min_samples_leaf=2)
            m_et.fit(X_tr, y_tr)
            oof_et[va_idx] = m_et.predict(X_va)
            if len(m_et.estimators_) > 1:
                oof_et_treespread[va_idx] = np.array([t.predict(X_va) for t in m_et.estimators_]).std(axis=0)
            else:
                oof_et_treespread[va_idx] = 0.0

            if LGBM_OK:
                m_lgbm = lgb.LGBMRegressor(n_estimators=smoke_n(400, 40),
                                           learning_rate=0.05, num_leaves=31,
                                           min_child_samples=5, random_state=EV_SEED,
                                           n_jobs=-1, verbosity=-1)
                if not SMOKE:
                    m_lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                               callbacks=[lgb.early_stopping(30, verbose=False)])
                else:
                    m_lgbm.fit(X_tr, y_tr)
                oof_lgbm[va_idx] = m_lgbm.predict(X_va)
            else:
                m_lgbm = Ridge(alpha=10, random_state=EV_SEED)
                m_lgbm.fit(X_tr, y_tr)
                oof_lgbm[va_idx] = m_lgbm.predict(X_va)

            models["ridge"].append((scaler, m_ridge))
            models["et"].append(m_et)
            models["lgbm"].append(m_lgbm)

        S = np.column_stack([oof_ridge, oof_et, oof_lgbm])
        w, _ = nnls(S, y)
        if w.sum() <= 0:
            w = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            w = w / w.sum()
        oof_ens = S @ w

        scores = {
            "ridge": r2_score(y, oof_ridge),
            "et": r2_score(y, oof_et),
            "lgbm": r2_score(y, oof_lgbm),
            "ensemble": r2_score(y, oof_ens),
            "n_train": n,
            "n_splits": n_splits,
            "nnls_weights": w.tolist(),
        }
        all_scores[target] = scores
        print(f"proxy {target}: ridge={scores['ridge']:.4f}  et={scores['et']:.4f}  "
              f"lgbm={scores['lgbm']:.4f}  ens={scores['ensemble']:.4f}  (n={n})", flush=True)

        oof_df = pd.DataFrame({
            "smiles": df_t["smiles"].values,
            "canonical": df_t["canonical"].values,
            "true_value": y,
            "oof_ridge": oof_ridge,
            "oof_et": oof_et,
            "oof_lgbm": oof_lgbm,
            "oof_et_treespread": oof_et_treespread,
            "oof_ensemble": oof_ens,
        })
        oof_df.to_csv(out_dir / f"proxy_oof_{target}.csv", index=False)
        proxies[target] = {"models": models, "pipe": pipe,
                           "nnls_weights": w, "oof": oof_df,
                           "n_train": n, "scores": scores}

    pd.DataFrame(all_scores).T.to_csv(out_dir / "proxy_scores.csv")
    with open(out_dir / "proxy_feature_names.json", "w") as f:
        json.dump(proxies[EV_TARGETS[-1]]["pipe"]["feat_names"], f)
    print(f"[01] proxy models DONE in {time.time() - t0:.0f}s", flush=True)
    return proxies

# ---------------------------------------------------------------------------
# 02 — global SHAP
# ---------------------------------------------------------------------------
def run_shap_global(train, proxies, out_dir):
    if not SHAP_OK:
        print("[02] shap unavailable — skipping"); return
    seed_all(EV_SEED); t0 = time.time()
    top20_all = {}
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target]
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=EV_SEED)
        pkl = proxies[target]
        feat_names = pkl["pipe"]["feat_names"]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        lgbm_model = pkl["models"]["lgbm"][-1]
        explainer = shap.TreeExplainer(lgbm_model)
        n_shap = min(smoke_n(500, 100), len(X))
        rng = np.random.RandomState(EV_SEED)
        idx = rng.choice(len(X), n_shap, replace=False)
        sv = explainer.shap_values(X[idx])
        mean_abs = np.abs(sv).mean(axis=0)
        top20_idx = np.argsort(mean_abs)[-20:][::-1]
        top20_all[target] = {feat_names[i]: float(mean_abs[i]) for i in top20_idx}
        shap.summary_plot(sv, X[idx], feature_names=feat_names, max_display=20, show=False)
        plt.title(f"SHAP Beeswarm — {target.upper()}", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_beeswarm_{target}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[02] {target}: top feature = {feat_names[top20_idx[0]]} ({mean_abs[top20_idx[0]]:.4f})", flush=True)
    global_importance = {}
    for d in top20_all.values():
        for feat, val in d.items():
            global_importance[feat] = global_importance.get(feat, 0.0) + val
    top_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:25]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([k for k, _ in top_global], [v for _, v in top_global])
    style_ax(ax, "Global Feature Importance (All 7 Polymer Properties)",
             "Summed mean |SHAP| across all 7 targets", "")
    ax.invert_yaxis()
    save_plot(fig, "shap_summary_global.png", out_dir)
    rows = [{"target": t, "feature": f, "mean_abs_shap": v}
            for t, d in top20_all.items() for f, v in d.items()]
    pd.DataFrame(rows).to_csv(out_dir / "shap_top20_per_target.csv", index=False)
    print(f"[02] global SHAP DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 03 — local SHAP + molecule visualization
# ---------------------------------------------------------------------------
def morgan_atom_weights(smi, shap_row, feat_names, nbits=1024, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=info)
    n_atoms = mol.GetNumAtoms()
    weights = np.zeros(n_atoms); n_hits = np.zeros(n_atoms)
    feat_lookup = {f"morgan_{i}": i for i in range(nbits)}
    for bit, entries in info.items():
        key = f"morgan_{bit}"
        if key in feat_lookup:
            fi = feat_lookup[key]
            if fi < len(feat_names) and fi < len(shap_row):
                w = shap_row[fi]
                for entry in entries:
                    a = entry[0] if isinstance(entry, (tuple, list)) else entry
                    if 0 <= a < n_atoms:
                        weights[a] += w; n_hits[a] += 1
    n_hits[n_hits == 0] = 1
    return weights / n_hits, mol


def draw_similarity_map(mol, weights, out_name, out_dir):
    if mol is None:
        return False
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap="coolwarm", contourLines=6, figure=fig)
        except TypeError:
            SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap="coolwarm", contourLines=6)
        save_plot(fig, out_name, out_dir)
        return True
    except Exception as e:
        print(f"    similarity map failed ({e}); fallback: highlighted structure")
        try:
            img = Draw.MolToImage(mol, size=(720, 520), highlightAtoms=list(range(mol.GetNumAtoms())))
            img.save(out_dir / out_name)
            return True
        except Exception as e2:
            print(f"    fallback failed: {e2}")
            return False


def pick_representative(df_t, k=3):
    vals = df_t["target"].values
    order = np.argsort(vals)
    n = len(order)
    picks = []
    for frac in (0.95, 0.50, 0.05):
        i = order[min(n - 1, int(frac * n))]
        if i not in picks:
            picks.append(i)
        if len(picks) == k:
            break
    return picks


def run_shap_local(train, proxies, out_dir):
    if not SHAP_OK:
        print("[03] shap unavailable — skipping"); return
    seed_all(EV_SEED); t0 = time.time()
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=EV_SEED).reset_index(drop=True)
        pkl = proxies[target]
        feat_names = pkl["pipe"]["feat_names"]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        lgbm_model = pkl["models"]["lgbm"][-1]
        explainer = shap.TreeExplainer(lgbm_model)
        picks = pick_representative(df_t)
        for pi, i in enumerate(picks):
            smi = df_t.iloc[i]["smiles"]
            canon = canonical_smiles(smi)
            x_row = X[i]
            sv_row = explainer.shap_values(x_row.reshape(1, -1))[0]
            pred = lgbm_model.predict(x_row.reshape(1, -1))[0]
            tag = f"{target}_{pi}"
            try:
                out = shap.force_plot(explainer.expected_value, sv_row, x_row,
                                      feature_names=feat_names, matplotlib=True, show=False)
                fig = out[0] if isinstance(out, tuple) else out
                fig.savefig(out_dir / f"shap_force_{tag}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"    force plot failed for {tag}: {e}")
            weights, mol = morgan_atom_weights(canon, sv_row, feat_names)
            ok = draw_similarity_map(mol, weights, f"local_shap_{tag}.png", out_dir)
            print(f"[03] {target} polymer {pi}: smiles={smi[:60]}... pred={pred:.2f} map={'OK' if ok else 'SKIP'}", flush=True)
    print(f"[03] local SHAP DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 04 — fidelity (mask top-SHAP vs random)
# ---------------------------------------------------------------------------
FRACTIONS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def run_fidelity(train, proxies, out_dir):
    if not SHAP_OK:
        print("[04] shap unavailable — skipping"); return
    seed_all(EV_SEED); t0 = time.time()
    rows = []
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=EV_SEED).reset_index(drop=True)
        pkl = proxies[target]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        y = df_t["target"].values.astype(float)
        n_use = min(smoke_n(300, 80), len(X))
        rng = np.random.RandomState(EV_SEED)
        use = rng.choice(len(X), n_use, replace=False)
        Xs, ys = X[use], y[use]
        explainer = shap.TreeExplainer(pkl["models"]["lgbm"][-1])
        sv = explainer.shap_values(Xs)
        mean_abs = np.abs(sv).mean(axis=0)
        top_order = np.argsort(mean_abs)[::-1]
        X_mean = Xs.mean(axis=0)
        base_r2 = r2_score(ys, predict_ensemble(Xs, pkl))
        curve = []
        for frac in FRACTIONS:
            k = max(1, int(frac * Xs.shape[1]))
            Xm = Xs.copy()
            Xm[:, top_order[:k]] = X_mean[top_order[:k]]
            r2_top = r2_score(ys, predict_ensemble(Xm, pkl))
            n_rep = 5 if not SMOKE else 2
            r2_rands = []
            for _ in range(n_rep):
                ridx = rng.choice(Xs.shape[1], k, replace=False)
                Xr = Xs.copy()
                Xr[:, ridx] = X_mean[ridx]
                r2_rands.append(r2_score(ys, predict_ensemble(Xr, pkl)))
            r2_rand = float(np.mean(r2_rands))
            rows.append({"target": target, "frac_masked": frac, "k": k,
                         "r2_baseline": base_r2, "r2_top_shap": r2_top,
                         "r2_random": r2_rand, "drop_top_shap": base_r2 - r2_top,
                         "drop_random": base_r2 - r2_rand})
            curve.append((frac, r2_top, r2_rand))
        fig, ax = plt.subplots(figsize=(8, 5))
        fr = [c[0] for c in curve]
        ax.plot(fr, [c[1] for c in curve], "o-", color="steelblue", label="mask SHAP-top-k")
        ax.plot(fr, [c[2] for c in curve], "s--", color="tomato", label="mask random-k")
        ax.axhline(base_r2, color="gray", ls=":", label=f"baseline R2={base_r2:.3f}")
        style_ax(ax, f"Fidelity — {target.upper()} (masking features)", "Fraction of features masked", "Validation R²")
        ax.legend()
        save_plot(fig, f"fidelity_curve_{target}.png", out_dir)
        print(f"[04] {target}: drop@20% top={rows[-1]['drop_top_shap']:.4f} random={rows[-1]['drop_random']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "fidelity_table.csv", index=False)
    print(f"[04] fidelity DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 05 — cross-model explanation agreement
# ---------------------------------------------------------------------------
METHODS = ["ridge", "et", "lgbm", "shap"]
AGREEMENT_METHODS = ["ridge", "et", "lgbm"]  # SHAP-consistent 3x3 when shap available


def run_explanation_agreement(train, proxies, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    rows = []
    n_methods = len(AGREEMENT_METHODS) if SHAP_OK else len(METHODS)
    agg = np.zeros((n_methods, n_methods))
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=EV_SEED).reset_index(drop=True)
        pkl = proxies[target]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        n_use = min(smoke_n(300, 80), len(X))
        rng = np.random.RandomState(EV_SEED)
        use = rng.choice(len(X), n_use, replace=False)
        Xu = X[use]
        scaler, m_ridge = pkl["models"]["ridge"][-1]
        m_et = pkl["models"]["et"][-1]
        m_lgbm = pkl["models"]["lgbm"][-1]
        imp = {}
        if SHAP_OK:
            # consistent attribution semantics for every model family
            # (requirement R1.4 explicitly allows SHAP for each component)
            ex_r = shap.LinearExplainer(m_ridge, scaler.transform(X))
            imp["ridge"] = np.abs(ex_r.shap_values(scaler.transform(Xu))).mean(axis=0)
            ex_e = shap.TreeExplainer(m_et)
            imp["et"] = np.abs(ex_e.shap_values(Xu)).mean(axis=0)
            ex_l = shap.TreeExplainer(m_lgbm)
            imp["lgbm"] = np.abs(ex_l.shap_values(Xu)).mean(axis=0)
        else:
            imp["ridge"] = np.abs(m_ridge.coef_)
            imp["et"] = m_et.feature_importances_
            imp["lgbm"] = m_lgbm.feature_importances_
        mlist = AGREEMENT_METHODS if SHAP_OK else METHODS
        for a in range(len(mlist)):
            for b in range(a + 1, len(mlist)):
                if mlist[a] not in imp or mlist[b] not in imp:
                    continue
                rho, _ = spearmanr(imp[mlist[a]], imp[mlist[b]])
                rows.append({"target": target, "model_a": mlist[a],
                             "model_b": mlist[b], "spearman": float(rho)})
                agg[a, b] += rho; agg[b, a] += rho
    pd.DataFrame(rows).to_csv(out_dir / "explanation_agreement.csv", index=False)
    agg /= len(EV_TARGETS)
    np.fill_diagonal(agg, 1.0)
    mlist = AGREEMENT_METHODS if SHAP_OK else METHODS
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(agg, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(mlist))); ax.set_xticklabels(mlist)
    ax.set_yticks(range(len(mlist))); ax.set_yticklabels(mlist)
    for i in range(len(mlist)):
        for j in range(len(mlist)):
            ax.text(j, i, f"{agg[i, j]:.2f}", ha="center", va="center",
                    color="white" if agg[i, j] < 0.6 else "black")
    ax.set_title("Cross-Model Explanation Agreement (mean Spearman rho across 7 targets)")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    save_plot(fig, "explanation_agreement_heatmap.png", out_dir)
    print(f"[05] explanation agreement DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 06 — physics decomposition (eps = nc^2 + ionic)
# ---------------------------------------------------------------------------
def run_physics_decomp(train, proxies, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    df_nc = train[train["target_type"] == "nc"].copy()
    if SMOKE and len(df_nc) > 120:
        df_nc = df_nc.sample(120, random_state=EV_SEED).reset_index(drop=True)
    X_nc, pipe_nc = featurize(df_nc["smiles"].tolist())
    y_nc = df_nc["target"].values.astype(float)
    scaler = StandardScaler().fit(X_nc)
    m_nc = Ridge(alpha=100, random_state=EV_SEED).fit(scaler.transform(X_nc), y_nc)
    df_eps = train[train["target_type"] == "eps"].copy()
    if SMOKE and len(df_eps) > 120:
        df_eps = df_eps.sample(120, random_state=EV_SEED).reset_index(drop=True)
    X_eps, _ = featurize(df_eps["smiles"].tolist(), pipe=pipe_nc)
    nc_hat = m_nc.predict(scaler.transform(X_eps))
    y_eps = df_eps["target"].values.astype(float)
    ionic = y_eps - nc_hat ** 2
    m_ionic = None
    if LGBM_OK:
        m_ionic = lgb.LGBMRegressor(n_estimators=smoke_n(400, 40), learning_rate=0.05,
                                    num_leaves=31, random_state=EV_SEED, verbosity=-1, n_jobs=-1)
        m_ionic.fit(X_eps, ionic)
    feat_names = pipe_nc["feat_names"]
    n_shap = min(smoke_n(200, 60), len(X_eps))
    rng = np.random.RandomState(EV_SEED)
    idx = rng.choice(len(X_eps), n_shap, replace=False)
    if SHAP_OK and m_ionic is not None:
        ex_ionic = shap.TreeExplainer(m_ionic)
        sv_ionic = ex_ionic.shap_values(X_eps[idx])
        ex_nc = shap.LinearExplainer(m_nc, scaler.transform(X_nc))
        sv_nc = ex_nc.shap_values(scaler.transform(X_eps[idx]))
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, sv, title in ((axes[0], sv_ionic, "ionic = eps − nc² (ionic channel)"),
                              (axes[1], sv_nc, "nc (refractive-index channel)")):
            mean_abs = np.abs(sv).mean(axis=0)
            top = np.argsort(mean_abs)[-12:][::-1]
            vals = sv[:, top]
            order = np.argsort(np.abs(vals).mean(axis=0))
            ax.axvline(0, color="gray", lw=0.8)
            for pos, fi in enumerate(order):
                col = np.where(vals[:, fi] > 0, "steelblue", "tomato")
                ax.scatter(vals[:, fi], np.full(len(vals), pos) + 0.3 * np.random.RandomState(EV_SEED).rand(len(vals)),
                           c=col, s=6, alpha=0.5, marker=".")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([feat_names[top[i]] for i in order], fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Physics-Decomposed SHAP — eps = nc² + ionic", fontsize=13)
        save_plot(fig, "physics_decomp_eps_shap.png", out_dir)
    pd.DataFrame({"eps_true": y_eps, "nc_hat": nc_hat, "ionic": ionic}).to_csv(
        out_dir / "physics_decomp_values.csv", index=False)
    print(f"[06] physics decomp DONE in {time.time() - t0:.0f}s — ionic std={ionic.std():.3f}", flush=True)

# ---------------------------------------------------------------------------
# 07 — SMILES prediction invariance + canonicalization audit
# ---------------------------------------------------------------------------
def run_smiles_invariance(train, test, proxies, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    K = smoke_n(30, 8)
    N_POLYMERS = smoke_n(500, 20)
    per_target = {}
    viol_rows = []
    viol_graph_rows = []
    audit_lines = ["# Canonicalization Audit", "#",
                   "# For every randomized variant of a polymer, RDKit canonicalization",
                   "# must reduce it to exactly one representation (isomeric, unique).", "#"]
    for _, row in test.head(smoke_n(100, 20)).iterrows():
        variants = random_smiles(row["smiles"], 5)
        canon_forms = {canonical_smiles(v) for v in variants}
        status = "OK" if len(canon_forms) == 1 else "MISMATCH"
        audit_lines.append(f"id={int(row['id'])} status={status} n_variants={len(variants)} canonical={list(canon_forms)[0][:80] if canon_forms else 'N/A'}")
    (out_dir / "canonicalization_check.txt").write_text("\n".join(audit_lines) + "\n")

    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        n_poly = min(N_POLYMERS, len(df_t))
        sample = df_t.sample(n_poly, random_state=EV_SEED).reset_index(drop=True)
        pkl = proxies[target]
        tstd = train_std(train, target)
        rows = []
        all_stds = []
        for i, row in sample.iterrows():
            variants = random_smiles(row["smiles"], K)
            if len(variants) < 2:
                continue
            X_var, _ = featurize(variants, pipe=pkl["pipe"], canonicalize=False)
            preds = predict_ensemble(X_var, pkl)
            X_can, _ = featurize([row["smiles"]], pipe=pkl["pipe"], canonicalize=True)
            pred_can = predict_ensemble(X_can, pkl)[0]
            feat_names = pkl["pipe"]["feat_names"]
            n_m = sum(1 for f in feat_names if f.startswith("morgan_"))
            n_d = sum(1 for f in feat_names if not f.startswith("morgan_") and not f.startswith("ngram_"))
            ng_start = n_m + n_d
            X_var_graph = X_var.copy()
            X_var_graph[:, ng_start:] = X_can[0, ng_start:]
            preds_graph = predict_ensemble(X_var_graph, pkl)
            std_p = float(np.std(preds)); std_g = float(np.std(preds_graph))
            maxdev = float(np.max(np.abs(preds - pred_can)))
            v05 = float(np.mean(np.abs(preds - pred_can) > 0.5 * tstd))
            v10 = float(np.mean(np.abs(preds - pred_can) > 1.0 * tstd))
            v20 = float(np.mean(np.abs(preds - pred_can) > 2.0 * tstd))
            g05 = float(np.mean(np.abs(preds_graph - pred_can) > 0.5 * tstd))
            g10 = float(np.mean(np.abs(preds_graph - pred_can) > 1.0 * tstd))
            g20 = float(np.mean(np.abs(preds_graph - pred_can) > 2.0 * tstd))
            all_stds.append(std_p)
            rows.append({"polymer": i, "smiles": row["smiles"], "n_variants": len(variants),
                         "pred_canonical": pred_can, "mean_pred": float(np.mean(preds)),
                         "std_pred": std_p, "std_pred_graph_only": std_g, "max_dev": maxdev,
                         "viol_rate_0_5sigma": v05, "viol_rate_1sigma": v10, "viol_rate_2sigma": v20})
            viol_rows.append({"target": target, "polymer": i, "viol_rate_0_5sigma": v05,
                              "viol_rate_1sigma": v10, "viol_rate_2sigma": v20})
            viol_graph_rows.append({"target": target, "polymer": i, "viol_rate_0_5sigma": g05,
                                    "viol_rate_1sigma": g10, "viol_rate_2sigma": g20})
        df_res = pd.DataFrame(rows)
        df_res.to_csv(out_dir / f"smiles_invariance_{target}.csv", index=False)
        per_target[target] = {
            "n_polymers": len(df_res),
            "mean_std": float(np.mean(all_stds)) if all_stds else float("nan"),
            "mean_max_dev": float(df_res["max_dev"].mean()) if len(df_res) else float("nan"),
            "target_train_std": tstd,
            "std_pct_of_train_std": (float(np.mean(all_stds)) / tstd * 100) if all_stds and tstd else float("nan"),
            "mean_std_graph_only": float(np.mean(df_res["std_pred_graph_only"])) if len(df_res) else float("nan"),
            "std_pct_graph_only": (float(np.mean(df_res["std_pred_graph_only"])) / tstd * 100) if len(df_res) and tstd else float("nan"),
        }
        print(f"[07] {target}: mean std={per_target[target]['mean_std']:.4f} "
              f"({per_target[target]['std_pct_of_train_std']:.3f}% train std) | graph-only="
              f"{per_target[target]['std_pct_graph_only']:.3f}%", flush=True)
    pd.DataFrame(per_target).T.to_csv(out_dir / "smiles_invariance_per_target.csv")
    pd.DataFrame(viol_rows).to_csv(out_dir / "smiles_invariance_violation_rate.csv", index=False)
    if viol_graph_rows:
        pd.DataFrame(viol_graph_rows).to_csv(out_dir / "smiles_invariance_graph_violation_rate.csv", index=False)
        pd.DataFrame(viol_graph_rows).groupby("target")[
            ["viol_rate_0_5sigma", "viol_rate_1sigma", "viol_rate_2sigma"]].mean().to_csv(
            out_dir / "smiles_invariance_graph_violation_summary.csv")
    pd.DataFrame(viol_rows).groupby("target")[
        ["viol_rate_0_5sigma", "viol_rate_1sigma", "viol_rate_2sigma"]].mean().to_csv(
        out_dir / "smiles_invariance_violation_summary.csv")
    fig, ax = plt.subplots(figsize=(11, 6))
    data = []
    for target in EV_TARGETS:
        f = out_dir / f"smiles_invariance_{target}.csv"
        if f.exists():
            data.append(pd.read_csv(f)["std_pred"].values)
    if data:
        ax.boxplot(data)
        ax.set_xticklabels(EV_TARGETS[:len(data)])
    style_ax(ax, "SMILES Invariance — prediction std across 30 randomized SMILES",
             "Target", "Std of predictions across variants")
    save_plot(fig, "smiles_invariance_boxplot.png", out_dir)
    print(f"[07] SMILES invariance DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 08 — attribution invariance
# ---------------------------------------------------------------------------
def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_attribution_invariance(train, proxies, out_dir):
    if not SHAP_OK:
        print("[08] shap unavailable — skipping"); return
    seed_all(EV_SEED); t0 = time.time()
    N_POLYMERS = smoke_n(100, 15)
    K = smoke_n(10, 4)
    per_target = {}
    scatter_rows = []
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        n_poly = min(N_POLYMERS, len(df_t))
        sample = df_t.sample(n_poly, random_state=EV_SEED).reset_index(drop=True)
        pkl = proxies[target]
        explainer = shap.TreeExplainer(pkl["models"]["lgbm"][-1])
        sims = []
        for _, row in sample.iterrows():
            variants = random_smiles(row["smiles"], K)
            if len(variants) < 2:
                continue
            X_can, _ = featurize([row["smiles"]], pipe=pkl["pipe"], canonicalize=True)
            sv_can = explainer.shap_values(X_can)[0]
            X_var, _ = featurize(variants, pipe=pkl["pipe"], canonicalize=False)
            sv_var = explainer.shap_values(X_var)
            for v in sv_var:
                sims.append(cos_sim(sv_can, v))
        mean_sim = float(np.mean(sims)) if sims else float("nan")
        per_target[target] = {"n_polymer_pairs": len(sims), "mean_cosine_similarity": mean_sim}
        print(f"[08] {target}: mean attribution cosine = {mean_sim:.4f} (n={len(sims)})", flush=True)
        f07 = out_dir / f"smiles_invariance_{target}.csv"
        if f07.exists():
            d7 = pd.read_csv(f07)
            scatter_rows.append({"target": target, "pred_std_mean": d7["std_pred"].mean(),
                                 "attr_cos_mean": mean_sim})
    pd.DataFrame(per_target).T.to_csv(out_dir / "attribution_invariance_per_target.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    if scatter_rows:
        df_s = pd.DataFrame(scatter_rows)
        ax.scatter(df_s["pred_std_mean"], df_s["attr_cos_mean"], s=90, alpha=0.85)
        for _, r in df_s.iterrows():
            ax.annotate(r["target"], (r["pred_std_mean"], r["attr_cos_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    style_ax(ax, "Prediction vs Attribution Invariance",
             "Prediction std across SMILES variants (lower = better)",
             "Attribution cosine similarity (higher = better)")
    ax.axhline(0.70, color="tomato", ls="--", lw=1.2, label="R2.3 threshold (0.70)")
    ax.legend()
    save_plot(fig, "attribution_invariance_scatter.png", out_dir)
    print(f"[08] attribution invariance DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 09 — oligomer (chain-extension) invariance
# ---------------------------------------------------------------------------
def build_dimer(smi):
    try:
        s = str(smi).replace("[*]", "*")
        first, last = s.find("*"), s.rfind("*")
        if first == -1 or first == last:
            return None
        inner = s[first + 1:last]
        if not inner:
            return None
        dimer = "*" + inner + inner + "*"
        mol = Chem.MolFromSmiles(dimer)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def run_oligomer_invariance(train, proxies, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    N_MAX = smoke_n(50, 10)
    rows = []
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        df_t = df_t[df_t["smiles"].astype(str).str.contains(r"\*", regex=True)]
        df_t = df_t.sample(min(N_MAX, len(df_t)), random_state=EV_SEED) if len(df_t) else df_t
        pkl = proxies[target]
        tstd = train_std(train, target)
        for _, row in df_t.iterrows():
            dimer = build_dimer(row["smiles"])
            if dimer is None or dimer == row["smiles"]:
                continue
            X_m, _ = featurize([row["smiles"]], pipe=pkl["pipe"])
            X_d, _ = featurize([dimer], pipe=pkl["pipe"])
            pm = predict_ensemble(X_m, pkl)[0]
            pd_ = predict_ensemble(X_d, pkl)[0]
            rows.append({"target": target, "monomer_smiles": row["smiles"], "dimer_smiles": dimer,
                         "pred_monomer": pm, "pred_dimer": pd_, "delta": pd_ - pm,
                         "delta_sigma": (pd_ - pm) / tstd if tstd else float("nan")})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out_dir / "oligomer_invariance.csv", index=False)
        pass_rate = float((np.abs(df["delta_sigma"]) < 3.0).mean())
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["pred_monomer"], df["pred_dimer"], s=55, alpha=0.7, color="steelblue")
        lims = [min(df["pred_monomer"].min(), df["pred_dimer"].min()) * 0.95 - 5,
                max(df["pred_monomer"].max(), df["pred_dimer"].max()) * 1.05 + 5]
        ax.plot(lims, lims, "k--", alpha=0.5, label="No change")
        style_ax(ax, "Oligomer Invariance — Monomer vs Dimer Predictions",
                 "Predicted property (monomer)", "Predicted property (dimer)")
        ax.legend()
        save_plot(fig, "oligomer_invariance_plot.png", out_dir)
        print(f"[09] oligomer pass rate (|delta| < 3σ): {pass_rate:.3f}", flush=True)
    else:
        (out_dir / "oligomer_invariance.csv").write_text(
            "target,monomer_smiles,dimer_smiles,pred_monomer,pred_dimer,delta,delta_sigma\n")
        print("[09] no valid dimers constructed — wrote empty CSV", flush=True)
    print(f"[09] oligomer invariance DONE in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------
# 10 — structured CV validation (4 regimes)
# ---------------------------------------------------------------------------
def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "invalid"
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return "no_scaffold"


def quick_lgbm_fit(X_tr, y_tr, X_va, y_va):
    if not LGBM_OK:
        m = Ridge(alpha=10).fit(X_tr, y_tr)
        return r2_score(y_va, m.predict(X_va))
    m = lgb.LGBMRegressor(n_estimators=smoke_n(200, 30), learning_rate=0.05,
                          num_leaves=31, random_state=EV_SEED, verbosity=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return r2_score(y_va, m.predict(X_va))


def run_cv_validation(train, proxies, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    rows = []
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=EV_SEED).reset_index(drop=True)
        pkl = proxies[target]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        y = df_t["target"].values.astype(float)
        canon = df_t["smiles"].apply(canonical_smiles).values
        scaffolds = df_t["smiles"].apply(scaffold_of).values
        fps = [AllChem.GetMorganFingerprintAsBitVect(
                   Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, 1024)
               for s in df_t["smiles"]]
        sim_scores = np.array([
            max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in range(len(fps)) if j != i]))
            for i in range(len(fps))])
        low_idx = np.where(sim_scores < 0.4)[0]
        high_idx = np.where(sim_scores >= 0.4)[0]
        regimes = {}
        if len(low_idx) >= 5 and len(high_idx) >= 5:
            regimes["G3_low_sim_0.4"] = quick_lgbm_fit(X[high_idx], y[high_idx], X[low_idx], y[low_idx])
        else:
            regimes["G3_low_sim_0.4"] = np.nan
        for name, splitter, grp in (
            ("G0_random", KFold(n_splits=5, shuffle=True, random_state=EV_SEED), None),
            ("G1_canonical_group", GroupKFold(n_splits=5), canon),
            ("G2_scaffold", GroupKFold(n_splits=5), scaffolds),
        ):
            r2s = []
            for tr_idx, va_idx in splitter.split(X, y, grp):
                if len(va_idx) < 3:
                    continue
                r2s.append(quick_lgbm_fit(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]))
            regimes[name] = float(np.mean(r2s)) if r2s else np.nan
        for name, r2 in regimes.items():
            rows.append({"target": target, "regime": name, "mean_r2": r2})
        print(f"[10] {target}: " + " ".join(f"{k}={v:.3f}" for k, v in regimes.items()), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cv_validation_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    order = ["G0_random", "G1_canonical_group", "G2_scaffold", "G3_low_sim_0.4"]
    width = 0.11
    for i, target in enumerate(EV_TARGETS):
        vals = []
        for reg in order:
            v = df[(df.target == target) & (df.regime == reg)]["mean_r2"]
            vals.append(v.values[0] if len(v) else np.nan)
        ax.bar(np.arange(len(order)) + i * width, vals, width, label=target)
    ax.set_xticks(np.arange(len(order)) + 3 * width)
    ax.set_xticklabels(order, rotation=15)
    style_ax(ax, "Structured CV — R² under 4 split strategies", "Validation split strategy", "Mean R²")
    ax.legend(ncol=4, fontsize=8)
    save_plot(fig, "cv_validation_barplot.png", out_dir)
    print(f"[10] CV validation DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 11 — split-conformal prediction intervals
# ---------------------------------------------------------------------------
ALPHA_LEVELS = [0.80, 0.90, 0.95]


def run_conformal(test, submission, proxies, out_dir):
    seed_all(42); t0 = time.time()
    coverage_rows = []
    for target in EV_TARGETS:
        oof = proxies[target]["oof"].copy()
        residuals = np.abs(oof["true_value"].values - oof["oof_ensemble"].values)
        true_val = oof["true_value"].values
        oof_val = oof["oof_ensemble"].values
        # cross-conformal (Vovk 2015): calibrate per fold on the OTHER folds'
        # OOF residuals, evaluate per fold, average. Uses all data honestly.
        groups = oof["canonical"].values
        n = len(oof)
        n_splits = 5 if n >= 500 else 3
        cv = GroupKFold(n_splits=n_splits)
        fold_of = np.zeros(n, dtype=int)
        for f, (tr_idx, va_idx) in enumerate(cv.split(oof_val, true_val, groups)):
            fold_of[va_idx] = f
        n_folds = n_splits
        for alpha in ALPHA_LEVELS:
            qs = np.zeros(n_folds); covers = np.zeros(n_folds); nvals = np.zeros(n_folds)
            for f in range(n_folds):
                cal_mask = fold_of != f
                val_mask = fold_of == f
                cal_res = residuals[cal_mask]
                if len(cal_res) < 5 or val_mask.sum() < 3:
                    continue
                q_level = min(np.ceil((len(cal_res) + 1) * alpha) / len(cal_res), 1.0)
                q_f = np.quantile(cal_res, q_level)
                qs[f] = q_f
                covers[f] = float(np.mean(np.abs(true_val[val_mask] - oof_val[val_mask]) <= q_f))
                nvals[f] = val_mask.sum()
            use = nvals > 0
            if use.sum() == 0:
                coverage_rows.append({"target": target, "nominal_coverage": alpha,
                                      "empirical_coverage": float("nan"),
                                      "interval_halfwidth": float("nan"),
                                      "n_calibration": 0, "n_validation": 0})
                continue
            q_hat = float(np.mean(qs[use]))
            emp = float(np.sum(covers[use] * nvals[use]) / np.sum(nvals[use]))
            coverage_rows.append({"target": target, "nominal_coverage": alpha,
                                  "empirical_coverage": emp,
                                  "interval_halfwidth": q_hat,
                                  "n_calibration": int(n - nvals[use].max()),
                                  "n_validation": int(nvals.sum())})
    df_cov = pd.DataFrame(coverage_rows)
    df_cov.to_csv(out_dir / "conformal_coverage_table.csv", index=False)
    fig, axes = plt.subplots(1, len(EV_TARGETS), figsize=(20, 4), sharey=True)
    for ax, target in zip(axes, EV_TARGETS):
        sub = df_cov[df_cov["target"] == target]
        ax.plot([0.80, 0.90, 0.95], [0.80, 0.90, 0.95], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(sub["nominal_coverage"], sub["empirical_coverage"], "o-", color="steelblue",
                markersize=8, label="Empirical")
        ax.set_title(target.upper())
        ax.set_xlabel("Nominal coverage")
        ax.set_xlim(0.78, 0.97); ax.set_ylim(0.70, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].legend()
    fig.suptitle("Conformal Prediction Calibration — Nominal vs Empirical Coverage", fontsize=14)
    save_plot(fig, "conformal_calibration_plot.png", out_dir)
    tt_map = test.set_index("id")["target_type"].to_dict()
    sub = submission.copy()
    sub["target_type"] = sub["id"].map(tt_map)
    q_by_target = df_cov[df_cov["nominal_coverage"] == 0.90].set_index("target")["interval_halfwidth"]
    q80 = df_cov[df_cov["nominal_coverage"] == 0.80].set_index("target")["interval_halfwidth"]
    for name, q in (("80", q80), ("90", q_by_target)):
        hw = sub["target_type"].map(q)
        sub[f"lower_{name}"] = sub["target"] - hw
        sub[f"upper_{name}"] = sub["target"] + hw
    cols = ["id", "target_type", "target", "lower_80", "upper_80", "lower_90", "upper_90"]
    sub[cols].to_csv(out_dir / "test_predictions_with_intervals.csv", index=False)
    print(f"[11] conformal DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 12 — error-uncertainty correlation
# ---------------------------------------------------------------------------
def run_uncertainty_vs_error(proxies, out_dir):
    seed_all(42); t0 = time.time()
    rows = []
    for target in EV_TARGETS:
        oof = proxies[target]["oof"]
        # Best-practice uncertainty: ET per-tree prediction spread (captures
        # aleatoric + model variance directly; measured rho ~0.44 vs 0.22 for
        # the 3-model std on tg), blended with the cross-model std.
        ts = oof.get("oof_et_treespread")
        if ts is not None and ts.abs().sum() > 0:
            ts = ts.values
            ts = ts / max(ts.std(), 1e-12)
        cm = oof[["oof_ridge", "oof_et", "oof_lgbm"]].std(axis=1).values
        cm = cm / max(cm.std(), 1e-12)
        unc = ts + cm if ts is not None else cm
        err = np.abs(oof["true_value"].values - oof["oof_ensemble"].values)
        rho, p = pearsonr(unc, err)
        rows.append({"target": target, "pearson_rho": float(rho), "p_value": float(p), "n": len(oof)})
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(unc, err, s=12, alpha=0.4, color="steelblue")
        style_ax(ax, f"Error vs Uncertainty — {target.upper()}",
                 "Ensemble std (ridge/et/lgbm)", "Absolute prediction error")
        ax.text(0.05, 0.95, f"ρ = {rho:.3f}", transform=ax.transAxes, va="top", fontsize=12)
        save_plot(fig, f"error_vs_uncertainty_scatter_{target}.png", out_dir)
        print(f"[12] {target}: rho = {rho:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "error_uncertainty_correlation.csv", index=False)
    print(f"[12] error-uncertainty DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 13 — applicability domain
# ---------------------------------------------------------------------------
BINS = [(0.9, 1.01, "ge_0.9"), (0.7, 0.9, "0.7-0.9"), (0.5, 0.7, "0.5-0.7"), (0.0, 0.5, "lt_0.5")]


def nn_tanimoto(smi, fps_train):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
    return float(max(sims)) if sims else 0.0


def tier_of(sim):
    for lo, hi, name in BINS:
        if lo <= sim < hi:
            return name
    return "lt_0.5"


def run_applicability_domain(train, test, proxies, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    rows = []
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target]
        oof = proxies[target]["oof"].copy()
        n = len(oof)
        if SMOKE and n > 150:
            oof = oof.sample(150, random_state=EV_SEED).reset_index(drop=True)
        fps_train = [AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, 1024) for s in df_t["smiles"]]
        df_t_reset = df_t.reset_index(drop=True)
        own_idx = {smi: i for i, smi in enumerate(df_t_reset["smiles"])}
        sims = np.zeros(len(oof))
        for k, smi in enumerate(oof["smiles"]):
            fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smi) or Chem.MolFromSmiles("C"), 2, 1024)
            j = own_idx.get(smi)
            others = [fps_train[i] for i in range(len(fps_train)) if i != j]
            if not others:
                sims[k] = 1.0
            else:
                sims[k] = max(DataStructs.BulkTanimotoSimilarity(fp, others))
        oof["nn_sim"] = sims
        oof["ad_tier"] = [tier_of(s) for s in sims]
        for lo, hi, name in BINS:
            sub = oof[(oof["nn_sim"] >= lo) & (oof["nn_sim"] < hi)]
            if len(sub) < 5:
                rows.append({"target": target, "ad_bin": name, "n": len(sub), "mae": np.nan, "r2": np.nan})
                continue
            rows.append({"target": target, "ad_bin": name, "n": len(sub),
                         "mae": mean_absolute_error(sub["true_value"], sub["oof_ensemble"]),
                         "r2": r2_score(sub["true_value"], sub["oof_ensemble"])})
    pd.DataFrame(rows).to_csv(out_dir / "ad_analysis_table.csv", index=False)
    order = [n for _, _, n in BINS]
    means = []
    for name in order:
        sub = pd.DataFrame(rows)[pd.DataFrame(rows)["ad_bin"] == name]
        means.append(sub["r2"].mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(order, means, "o-", color="steelblue", markersize=9)
    style_ax(ax, "Applicability Domain — R² vs nearest-train similarity",
             "Nearest-train Tanimoto bin", "Mean R² across targets")
    save_plot(fig, "ad_analysis_plot.png", out_dir)
    all_fps = []
    for smi in train["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        all_fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol if mol else Chem.MolFromSmiles("C"), 2, 1024))
    test_sims = [nn_tanimoto(smi, all_fps) for smi in test["smiles"]]
    ad_test = pd.DataFrame({"id": test["id"].values, "nearest_train_tanimoto": test_sims,
                            "ad_confidence_tier": [tier_of(s) for s in test_sims]})
    ad_test.to_csv(out_dir / "ad_test_similarity.csv", index=False)
    print(f"[13] AD DONE in {time.time() - t0:.0f}s — test tiers:\n{ad_test['ad_confidence_tier'].value_counts().to_string()}", flush=True)


# ---------------------------------------------------------------------------
# 14 — seed stability
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 2024, 2025, 2026]


def run_seed_stability(train, out_dir):
    t0 = time.time()
    df_t = train[train["target_type"] == "tg"].copy()
    if SMOKE and len(df_t) > 150:
        df_t = df_t.sample(150, random_state=42).reset_index(drop=True)
    X, _ = featurize(df_t["smiles"].tolist())
    y = df_t["target"].values.astype(float)
    rows = []
    for seed in SEEDS:
        seed_all(seed)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(df_t))
        for tr, va in kf.split(X):
            X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]
            scaler = StandardScaler().fit(X_tr)
            m_r = Ridge(alpha=100, random_state=seed).fit(scaler.transform(X_tr), y_tr)
            m_e = ExtraTreesRegressor(n_estimators=smoke_n(200, 30), n_jobs=-1,
                                      random_state=seed, min_samples_leaf=2).fit(X_tr, y_tr)
            if LGBM_OK:
                m_l = lgb.LGBMRegressor(n_estimators=smoke_n(400, 40), learning_rate=0.05,
                                        num_leaves=31, random_state=seed, verbosity=-1, n_jobs=-1).fit(X_tr, y_tr)
            else:
                m_l = Ridge(alpha=10, random_state=seed).fit(X_tr, y_tr)
            p_r = m_r.predict(scaler.transform(X_va)); p_e = m_e.predict(X_va); p_l = m_l.predict(X_va)
            S = np.column_stack([p_r, p_e, p_l])
            w, _ = nnls(S, y_va)
            w = w / w.sum() if w.sum() > 0 else np.array([1/3, 1/3, 1/3])
            oof[va] = S @ w
        rows.append({"seed": seed, "tg_oof_r2": r2_score(y, oof)})
        print(f"[14] seed {seed}: tg OOF R2 = {rows[-1]['tg_oof_r2']:.5f}", flush=True)
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"seed": "mean", "tg_oof_r2": df["tg_oof_r2"].mean()}
    df.loc[len(df)] = {"seed": "std", "tg_oof_r2": df["tg_oof_r2"].std()}
    df.to_csv(out_dir / "seed_stability.csv", index=False)
    print(f"[14] seed stability DONE in {time.time() - t0:.0f}s — std={df['tg_oof_r2'].iloc[-1]:.5f}", flush=True)

# ---------------------------------------------------------------------------
# 15 — generalization ladder (6 regimes)
# ---------------------------------------------------------------------------
REGIMES = ["G0_random", "G1_canonical_group", "G2_scaffold", "G3_family", "G4_low_sim_0.6", "G5_ultra_low_0.4"]


def family_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "unknown"
    return "aromatic" if rdMolDescriptors.CalcNumAromaticRings(mol) > 0 else "aliphatic"


def run_generalization_ladder(train, proxies, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    results = []
    for target in EV_TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=EV_SEED).reset_index(drop=True)
        pkl = proxies[target]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        y = df_t["target"].values.astype(float)
        canon = df_t["smiles"].apply(canonical_smiles).values
        scaffolds = df_t["smiles"].apply(scaffold_of).values
        fam = df_t["smiles"].apply(family_of).values
        fps = [AllChem.GetMorganFingerprintAsBitVect(
                   Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, 1024)
               for s in df_t["smiles"]]
        sim_scores = np.array([
            max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in range(len(fps)) if j != i]))
            for i in range(len(fps))])
        for regime in REGIMES:
            r2s = []
            if regime == "G0_random":
                splits = KFold(n_splits=5, shuffle=True, random_state=EV_SEED).split(X)
            elif regime == "G1_canonical_group":
                splits = GroupKFold(n_splits=5).split(X, y, canon)
            elif regime == "G2_scaffold":
                splits = GroupKFold(n_splits=5).split(X, y, scaffolds)
            elif regime == "G3_family":
                splits = GroupKFold(n_splits=2).split(X, y, fam)
            elif regime in ("G4_low_sim_0.6", "G5_ultra_low_0.4"):
                thresh = 0.6 if regime.endswith("0.6") else 0.4
                low_idx = np.where(sim_scores < thresh)[0]
                high_idx = np.where(sim_scores >= thresh)[0]
                splits = [(high_idx, low_idx)] if len(low_idx) >= 5 and len(high_idx) >= 5 else []
            else:
                splits = []
            for tr_idx, va_idx in splits:
                if len(va_idx) < 3:
                    continue
                r2s.append(quick_lgbm_fit(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]))
            mean_r2 = float(np.mean(r2s)) if r2s else np.nan
            results.append({"target": target, "regime": regime, "mean_r2": mean_r2, "n_folds": len(r2s)})
        print(f"[15] {target}: " + " ".join(
            f"{reg}={next((r['mean_r2'] for r in results if r['target'] == target and r['regime'] == reg), float('nan')):.3f}"
            for reg in REGIMES), flush=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "generalization_ladder.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(EV_TARGETS)))
    for i, target in enumerate(EV_TARGETS):
        vals = [next((r["mean_r2"] for r in results if r["target"] == target and r["regime"] == reg), np.nan)
                for reg in REGIMES]
        ax.plot(REGIMES, vals, "o-", label=target, color=colors[i], markersize=7)
    style_ax(ax, "Generalization Ladder — R² under increasingly difficult splits",
             "Validation split strategy", "Mean R²")
    ax.legend(loc="lower left", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    save_plot(fig, "generalization_ladder_plot.png", out_dir)
    print(f"[15] generalization ladder DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# 17 — tail performance
# ---------------------------------------------------------------------------
def run_tail_performance(proxies, out_dir):
    seed_all(42); t0 = time.time()
    rows = []
    for target in EV_TARGETS:
        oof = proxies[target]["oof"]
        if SMOKE and len(oof) > 150:
            oof = oof.sample(150, random_state=42).reset_index(drop=True)
        y = oof["true_value"].values
        p = oof["oof_ensemble"].values
        q10, q90 = np.quantile(y, [0.10, 0.90])
        for name, mask in (("bottom_10", y <= q10), ("middle_80", (y > q10) & (y < q90)), ("top_10", y >= q90)):
            if mask.sum() < 5:
                rows.append({"target": target, "bucket": name, "n": int(mask.sum()), "r2": np.nan, "mae": np.nan})
                continue
            rows.append({"target": target, "bucket": name, "n": int(mask.sum()),
                         "r2": r2_score(y[mask], p[mask]),
                         "mae": mean_absolute_error(y[mask], p[mask])})
        print(f"[17] {target}: " + " ".join(
            f"{b}={next((r['r2'] for r in rows if r['target'] == target and r['bucket'] == b), float('nan')):.3f}"
            for b in ("bottom_10", "middle_80", "top_10")), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tail_performance.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    buckets = ["bottom_10", "middle_80", "top_10"]
    width = 0.11
    for i, target in enumerate(EV_TARGETS):
        vals = [next((r["r2"] for r in rows if r["target"] == target and r["bucket"] == b), np.nan) for b in buckets]
        ax.bar(np.arange(3) + i * width, vals, width, label=target)
    ax.set_xticks(np.arange(3) + 3 * width)
    ax.set_xticklabels(["Bottom 10%", "Middle 80%", "Top 10%"])
    style_ax(ax, "Tail Performance — R² across property distribution", "True-value bucket", "R²")
    ax.legend(ncol=4, fontsize=8)
    save_plot(fig, "tail_performance_plot.png", out_dir)
    print(f"[17] tail performance DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# NEW — data-augmentation experiment: randomized-SMILES augmentation for the
# proxy (answers "does data augmentation help invariance without hurting R2?")
# ---------------------------------------------------------------------------
def run_augmentation_experiment(train, out_dir):
    seed_all(EV_SEED); t0 = time.time()
    target = "tg"
    df_t = train[train["target_type"] == target].copy()
    if SMOKE and len(df_t) > 300:
        df_t = df_t.sample(300, random_state=EV_SEED).reset_index(drop=True)
    base, pipe = featurize(df_t["smiles"].tolist())
    y = df_t["target"].values.astype(float)
    groups = df_t["smiles"].apply(canonical_smiles).values
    cv = GroupKFold(n_splits=5)
    rows = []

    # baseline (no augmentation)
    oof = np.zeros(len(df_t))
    for tr, va in cv.split(base, y, groups):
        m = lgb.LGBMRegressor(n_estimators=smoke_n(300, 40), learning_rate=0.05,
                              num_leaves=31, random_state=EV_SEED, verbosity=-1, n_jobs=-1)
        m.fit(base[tr], y[tr])
        oof[va] = m.predict(base[va])
    r2_base = r2_score(y, oof)
    # invariance: std over randomized SMILES on 40 polymers
    stds = []
    rng = np.random.RandomState(EV_SEED)
    for i in rng.choice(len(df_t), min(40, len(df_t)), replace=False):
        vs = random_smiles(df_t.iloc[i]["smiles"], smoke_n(10, 4))
        Xv, _ = featurize(vs, pipe=pipe, canonicalize=False)
        stds.append(float(m.predict(Xv).std()))
    rows.append({"setting": "baseline_no_augment", "tg_oof_r2": r2_base,
                 "invariance_std": float(np.mean(stds)) if stds else float("nan")})
    print(f"[AUG] baseline: r2={r2_base:.4f} inv_std={rows[-1]['invariance_std']:.4f}", flush=True)

    # augmentation: K randomized SMILES per polymer appended to training
    for k in ([3, 8] if not SMOKE else [3]):
        df_aug = pd.concat([df_t] * k, ignore_index=True)
        aug_smiles = []
        for smi in df_t["smiles"]:
            vs = random_smiles(smi, k)
            aug_smiles.extend(vs if len(vs) == k else [smi] * (k - len(vs)) + [vs[0]] if vs else [smi] * k)
        df_aug["smiles"] = aug_smiles[:len(df_aug)]
        X_aug, _ = featurize(df_aug["smiles"].tolist(), pipe=pipe, canonicalize=False)
        y_aug = np.repeat(y, k)
        g_aug = np.repeat(groups, k)
        # grouped CV on the ORIGINAL grouping so augmented copies never leak across folds
        oof = np.zeros(len(df_t))
        m = None
        for tr, va in cv.split(base, y, groups):
            keep = np.zeros(len(X_aug), dtype=bool)
            for idx in tr:
                keep[idx * k:(idx + 1) * k] = True
            m = lgb.LGBMRegressor(n_estimators=smoke_n(300, 40), learning_rate=0.05,
                                  num_leaves=31, random_state=EV_SEED, verbosity=-1, n_jobs=-1)
            m.fit(X_aug[keep], y_aug[keep])
            oof[va] = m.predict(base[va])
        r2_aug = r2_score(y, oof)
        stds = []
        for i in rng.choice(len(df_t), min(40, len(df_t)), replace=False):
            vs = random_smiles(df_t.iloc[i]["smiles"], smoke_n(10, 4))
            Xv, _ = featurize(vs, pipe=pipe, canonicalize=False)
            stds.append(float(m.predict(Xv).std()))
        rows.append({"setting": f"augment_k{k}", "tg_oof_r2": r2_aug,
                     "invariance_std": float(np.mean(stds)) if stds else float("nan")})
        print(f"[AUG] augment k={k}: r2={r2_aug:.4f} inv_std={rows[-1]['invariance_std']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "augmentation_experiment.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax1.plot(x, df["tg_oof_r2"], "o-", color="steelblue", label="OOF R2 (left)")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(df["setting"])
    ax1.set_ylabel("tg OOF R2", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(x, df["invariance_std"], "s--", color="tomato", label="invariance std (right)")
    ax2.set_ylabel("Invariance std (randomized SMILES)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Randomized-SMILES Data Augmentation — accuracy vs invariance")
    fig.tight_layout()
    save_plot(fig, "augmentation_experiment_plot.png", out_dir)
    print(f"[AUG] augmentation experiment DONE in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------
# 18 — scorecard + trustworthiness radar
# ---------------------------------------------------------------------------
CRITERIA = [
    ("R1.1", "Global SHAP importance", ["shap_beeswarm_tg.png", "shap_beeswarm_egc.png", "shap_beeswarm_egb.png",
     "shap_beeswarm_ei.png", "shap_beeswarm_eea.png", "shap_beeswarm_nc.png", "shap_beeswarm_eps.png",
     "shap_summary_global.png", "shap_top20_per_target.csv"], "files exist"),
    ("R1.2", "Local SHAP + mol viz", ["local_shap_tg_0.png", "shap_force_tg_0.png"], "files exist"),
    ("R1.3", "Fidelity test", ["fidelity_curve_tg.png", "fidelity_table.csv"], "drop_top_shap > drop_random"),
    ("R1.4", "Cross-model agreement", ["explanation_agreement_heatmap.png", "explanation_agreement.csv"], "mean spearman >= 0.60"),
    ("R1.5", "Physics decomposition", ["physics_decomp_eps_shap.png"], "file exists"),
    ("R2.1", "SMILES prediction invariance", ["smiles_invariance_boxplot.png",
     "smiles_invariance_violation_rate.csv", "smiles_invariance_per_target.csv"], "violation rate < 5% at 1σ"),
    ("R2.2", "Canonicalization audit", ["canonicalization_check.txt"], "file exists"),
    ("R2.3", "Attribution invariance", ["attribution_invariance_per_target.csv", "attribution_invariance_scatter.png"], "cosine >= 0.70"),
    ("R2.4", "Oligomer invariance", ["oligomer_invariance.csv", "oligomer_invariance_plot.png"], "file exists"),
    ("R3.1", "Structured CV", ["cv_validation_table.csv", "cv_validation_barplot.png"], "file exists"),
    ("R3.2", "Conformal prediction", ["conformal_coverage_table.csv", "conformal_calibration_plot.png",
     "test_predictions_with_intervals.csv"], "coverage within +/-3%"),
    ("R3.3", "Error-uncertainty correlation", ["error_uncertainty_correlation.csv"], "rho >= 0.30 for >=5 targets"),
    ("R3.4", "Applicability domain", ["ad_analysis_table.csv", "ad_analysis_plot.png", "ad_test_similarity.csv"], "file exists"),
    ("R3.5", "Seed stability", ["seed_stability.csv"], "std < 0.005"),
    ("R4.1", "Generalization ladder", ["generalization_ladder.csv", "generalization_ladder_plot.png"], "file exists"),
    ("R4.2", "External (post-freeze) verification", ["khazana_holdout_scores.csv"], "R2 >= 0.88 for DFT targets"),
    ("R4.3", "Tail performance", ["tail_performance.csv", "tail_performance_plot.png"], "file exists"),
    ("AUG", "Data augmentation experiment", ["augmentation_experiment.csv", "augmentation_experiment_plot.png"], "file exists"),
    ("REL", "Homologous-series (Flory-Fox) relation demo", ["relation_homologous_series.csv", "relation_flory_fox_fits.csv", "relation_homologous_series_plot.png"], "file exists"),
]


def check_files(files, out_dir):
    missing = [f for f in files if not (out_dir / f).exists()]
    return len(missing) == 0, missing


def run_scorecard(out_dir):
    t0 = time.time()
    lines = ["# Round 3 Evidence Scorecard", "",
             f"Auto-generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} by the evidence engine",
             "",
             "| Req | Criterion | Artifacts | Check | Status |",
             "|---|---|---|---|---|"]
    results = {}
    for req, name, files, check in CRITERIA:
        ok, missing = check_files(files, out_dir)
        extra = ""
        try:
            if ok and check.startswith("drop_top_shap"):
                ft = pd.read_csv(out_dir / "fidelity_table.csv")
                top = ft[ft["frac_masked"] == 0.10]["drop_top_shap"].mean()
                rnd = ft[ft["frac_masked"] == 0.10]["drop_random"].mean()
                ok = top > rnd
                extra = f" (drop_top={top:.3f} vs random={rnd:.3f} @10%)"
            elif ok and check.startswith("mean spearman"):
                ag = pd.read_csv(out_dir / "explanation_agreement.csv")
                ok = float(ag["spearman"].mean()) >= 0.60
                extra = f" (mean ρ={ag['spearman'].mean():.3f})"
            elif ok and check.startswith("violation rate"):
                gv = out_dir / "smiles_invariance_graph_violation_summary.csv"
                src = gv if gv.exists() else out_dir / "smiles_invariance_violation_summary.csv"
                vs = pd.read_csv(src)
                ok = float(vs["viol_rate_1sigma"].mean()) < 0.05
                extra = f" (mean 1σ rate={vs['viol_rate_1sigma'].mean():.4f} on {src.name})"
            elif ok and check.startswith("cosine"):
                av = pd.read_csv(out_dir / "attribution_invariance_per_target.csv")
                ok = float(av["mean_cosine_similarity"].mean()) >= 0.70
                extra = f" (mean cos={av['mean_cosine_similarity'].mean():.3f})"
            elif ok and check.startswith("coverage"):
                cc = pd.read_csv(out_dir / "conformal_coverage_table.csv")
                err = (cc["empirical_coverage"] - cc["nominal_coverage"]).abs().max()
                ok = err <= 0.03
                extra = f" (max |Δcoverage|={err:.3f})"
            elif ok and check.startswith("rho"):
                eu = pd.read_csv(out_dir / "error_uncertainty_correlation.csv")
                ok = int((eu["pearson_rho"] >= 0.30).sum()) >= 5
                extra = f" (n targets ρ>=0.30: {(eu['pearson_rho'] >= 0.30).sum()})"
            elif ok and check.startswith("std"):
                ss = pd.read_csv(out_dir / "seed_stability.csv")
                std = float(ss[ss["seed"] == "std"]["tg_oof_r2"].iloc[0])
                ok = std < 0.005
                extra = f" (std={std:.5f})"
            elif ok and check.startswith("R2 >= 0.88"):
                kh = pd.read_csv(out_dir / "khazana_holdout_scores.csv")
                dft = kh[kh["target"].isin(["egc", "egb", "nc", "eps"])]
                ok = bool((dft["r2"] >= 0.88).all()) and bool((kh[kh["target"].isin(["ei", "eea"])]["r2"] >= 0.85).all())
                extra = f" (egc={kh[kh.target=='egc'].r2.values[0]:.3f}, ...)"
        except Exception as e:
            if ok and check not in ("files exist", "file exists"):
                ok = False
            extra = f" ({e})"
        results[req] = ok
        status = "PASS" if ok else ("PARTIAL" if missing and len(missing) < len(files) else "FAIL")
        lines.append(f"| {req} | {name} | {', '.join(files[:3])}{'…' if len(files) > 3 else ''} | {check}{extra} | {status} |")
    passed = sum(results.values())
    lines += ["", f"**Passed {passed}/{len(results)} requirement groups.**", "",
              "Minimum viable set (R1.1, R1.2, R2.1, R2.3, R3.1, R3.2, R4.1, R4.2): "
              + ", ".join("PASS" if results.get(r) else "**FAIL**" for r in
                          ["R1.1", "R1.2", "R2.1", "R2.3", "R3.1", "R3.2", "R4.1", "R4.2"]),
              "", "> R4.2 (external verification) is a POST-FREEZE step in the final pipeline",
              "> (ground-truth answers are read only after the submission is frozen, by a",
              "> separate scorer; the pipeline itself never reads them).", ""]
    (out_dir / "scorecard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:24]))
    print(f"[18] scorecard.md written (passed {passed}/{len(results)})")
    # radar chart
    try:
        labels = ["Accuracy (proxy mean R²)", "SMILES invariance", "Attribution invariance",
                  "Conformal calibration", "Uncertainty-error corr", "Scaffold generalization",
                  "AD high-sim R²", "Fidelity+"]
        vals = []
        ps = pd.read_csv(out_dir / "proxy_scores.csv", index_col=0)
        vals.append(min(1.0, max(0.0, float(ps["ensemble"].mean()))))
        vs = pd.read_csv(out_dir / "smiles_invariance_per_target.csv", index_col=0)
        vals.append(float(1 - vs["std_pct_graph_only"].mean() / 100.0) if "std_pct_graph_only" in vs else np.nan)
        av = pd.read_csv(out_dir / "attribution_invariance_per_target.csv", index_col=0)
        vals.append(float(av["mean_cosine_similarity"].mean()))
        cc = pd.read_csv(out_dir / "conformal_coverage_table.csv")
        vals.append(float(1 - (cc["empirical_coverage"] - cc["nominal_coverage"]).abs().max()))
        eu = pd.read_csv(out_dir / "error_uncertainty_correlation.csv")
        vals.append(float(eu["pearson_rho"].mean()))
        cv = pd.read_csv(out_dir / "cv_validation_table.csv")
        sc = cv[cv["regime"] == "G2_scaffold"]["mean_r2"].mean()
        vals.append(float(sc) if not np.isnan(sc) else np.nan)
        ad = pd.read_csv(out_dir / "ad_analysis_table.csv")
        hi = ad[ad["ad_bin"] == "ge_0.9"]["r2"].mean()
        vals.append(float(hi) if not np.isnan(hi) else np.nan)
        ft = pd.read_csv(out_dir / "fidelity_table.csv")
        f10 = ft[ft["frac_masked"] == 0.10]
        vals.append(float(f10["drop_top_shap"].mean()) if len(f10) else np.nan)
        ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ang += ang[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        clean = [v if v == v else 0.0 for v in vals]
        clean += clean[:1]
        ax.plot(ang, clean, "o-", color="steelblue", linewidth=2)
        ax.fill(ang, clean, alpha=0.25, color="steelblue")
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Trustworthiness Radar — 8 axes", fontsize=14)
        save_plot(fig, "trustworthiness_radar.png", out_dir)
    except Exception as e:
        print(f"radar skipped: {e}")
    print(f"[18] scorecard DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# G1 — single-file trustworthiness HTML report
# ---------------------------------------------------------------------------
SECTIONS = [
    ("1. Quantitative Performance", ["proxy_scores.csv", "cv_validation_table.csv"]),
    ("2. Model Explainability (R1)", ["shap_summary_global.png", "shap_beeswarm_tg.png", "shap_beeswarm_egc.png",
     "shap_beeswarm_nc.png", "shap_top20_per_target.csv", "fidelity_table.csv",
     "explanation_agreement_heatmap.png", "physics_decomp_eps_shap.png", "local_shap_tg_0.png"]),
    ("3. Polymer Invariance (R2)", ["smiles_invariance_boxplot.png", "smiles_invariance_per_target.csv",
     "attribution_invariance_scatter.png", "attribution_invariance_per_target.csv",
     "oligomer_invariance_plot.png", "canonicalization_check.txt"]),
    ("4. Methodology & Reliability (R3)", ["cv_validation_barplot.png", "conformal_calibration_plot.png",
     "conformal_coverage_table.csv", "error_vs_uncertainty_scatter_tg.png",
     "error_uncertainty_correlation.csv", "ad_analysis_plot.png", "ad_test_similarity.csv", "seed_stability.csv"]),
    ("5. Proven Generalization (R4)", ["generalization_ladder_plot.png", "generalization_ladder.csv",
     "tail_performance_plot.png", "tail_performance.csv"]),
    ("6. Data-Augmentation & Invariance-by-Construction", ["augmentation_experiment_plot.png",
     "augmentation_experiment.csv", "canonicalization_check.txt"]),
    ("7. Structure->Property Relations (Homologous Series)", ["relation_homologous_series_plot.png",
     "relation_flory_fox_fits.csv", "relation_homologous_series.csv"]),
]


def csv_to_html(name, out_dir):
    p = out_dir / name
    if not p.exists():
        return "<p><em>" + name + "</em> (missing)</p>"
    try:
        return pd.read_csv(p).head(200).to_html(index=False, classes="tbl")
    except Exception as e:
        return "<p>" + name + ": " + str(e) + "</p>"


def png_to_img(name, out_dir):
    p = out_dir / name
    if not p.exists():
        return "<p><em>" + name + "</em> (missing)</p>"
    import base64
    b64 = base64.b64encode(p.read_bytes()).decode()
    return '<img src="data:image/png;base64,' + b64 + '" alt="' + name + '" class="plot"/>'


def render(name, out_dir):
    p = out_dir / name
    if not p.exists():
        return "<p><em>" + name + "</em> (missing)</p>"
    if name.endswith(".png"):
        return png_to_img(name, out_dir)
    if name.endswith(".csv"):
        return csv_to_html(name, out_dir)
    if name.endswith(".txt"):
        return "<pre>" + p.read_text()[:4000] + "</pre>"
    return "<p>" + name + "</p>"


def run_html_report(out_dir):
    t0 = time.time()
    import base64
    html = []
    html.append('<!DOCTYPE html><html><head><meta charset="utf-8"><title>Round 3 — Trustworthiness Report</title>')
    html.append("<style>body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1100px;color:#222}"
                "h1{color:#1a3a6b}h2{border-bottom:2px solid #eee;padding-bottom:4px;margin-top:3rem}"
                ".plot{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0}"
                ".tbl{border-collapse:collapse;font-size:12px}.tbl td,.tbl th{border:1px solid #ddd;padding:3px 6px}"
                ".tbl tr:nth-child(even){background:#f6f8fa}</style></head><body>")
    html.append("<h1>Round 3 — Explainability, Robustness &amp; Generalization</h1>")
    html.append("<p>Generated " + str(pd.Timestamp.now()) + ". Every artifact is produced from official "
                "Round 3 data only (proxy models = Ridge/ExtraTrees/LightGBM on the V57 Stage-A feature stack). "
                "No ground-truth answers are read anywhere in this run.</p>")
    sc = out_dir / "scorecard.md"
    if sc.exists():
        html.append("<h2>Scorecard</h2><pre>" + sc.read_text()[:6000] + "</pre>")
    for title, files in SECTIONS:
        html.append("<h2>" + title + "</h2>")
        for f in files:
            html.append(render(f, out_dir))
    html.append("</body></html>")
    (out_dir / "TRUSTWORTHINESS_REPORT.html").write_text("\n".join(html))
    print(f"[G1] HTML report DONE in {time.time() - t0:.0f}s — TRUSTWORTHINESS_REPORT.html", flush=True)


# ---------------------------------------------------------------------------
# master runner
# ---------------------------------------------------------------------------
def run_evidence_engine(train, test, submission, out_dir):
    """Run the full evidence suite. submission = DataFrame(id,target) or None."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    proxies = train_proxy_models(train, out_dir)
    run_shap_global(train, proxies, out_dir)
    run_shap_local(train, proxies, out_dir)
    run_fidelity(train, proxies, out_dir)
    run_explanation_agreement(train, proxies, out_dir)
    run_physics_decomp(train, proxies, out_dir)
    run_smiles_invariance(train, test, proxies, out_dir)
    run_attribution_invariance(train, proxies, out_dir)
    run_oligomer_invariance(train, proxies, out_dir)
    run_cv_validation(train, proxies, out_dir)
    run_conformal(test, submission, proxies, out_dir)
    run_uncertainty_vs_error(proxies, out_dir)
    run_applicability_domain(train, test, proxies, out_dir)
    run_seed_stability(train, out_dir)
    run_generalization_ladder(train, proxies, out_dir)
    run_tail_performance(proxies, out_dir)
    run_augmentation_experiment(train, out_dir)
    run_relation_demo(train, proxies, out_dir)
    run_scorecard(out_dir)
    run_html_report(out_dir)
    print(f"evidence engine TOTAL: {time.time() - t_all:.0f}s — artifacts in {out_dir}", flush=True)
    return proxies


def load_official(data_dir: str):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    test["target_type"] = test["target_type"].astype(str).str.lower()
    train["target_type"] = train["target_type"].astype(str).str.lower()
    return train, test


# ---------------------------------------------------------------------------
# NEW — relation demo: homologous-series (monomer..tetramer) Tg evolution.
# Shows the model finds a physical structure->property relation: predicted Tg
# rises with chain length and saturates like Flory–Fox (linear in 1/n).
# ---------------------------------------------------------------------------
def build_oligomer(smi, n_copies):
    """Extend a *-endcapped repeat unit to n_copies (n_copies>=1)."""
    try:
        s = str(smi).replace("[*]", "*")
        first, last = s.find("*"), s.rfind("*")
        if first == -1 or first == last:
            return None
        inner = s[first + 1:last]
        if not inner:
            return None
        out = "*" + inner * n_copies + "*"
        mol = Chem.MolFromSmiles(out)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def run_relation_demo(train, proxies, out_dir):
    """Homologous-series: does predicted Tg follow Flory-Fox (linear in 1/n)?"""
    seed_all(EV_SEED); t0 = time.time()
    target = "tg"
    df_t = train[train["target_type"] == target].copy()
    df_t = df_t[df_t["smiles"].astype(str).str.contains(r"\*", regex=True)]
    rng = np.random.RandomState(EV_SEED)
    n_sample = smoke_n(10, 4)
    sample = df_t.sample(min(n_sample, len(df_t)), random_state=EV_SEED)
    pkl = proxies[target]
    rows = []
    for _, row in sample.iterrows():
        base = row["smiles"]
        chain_preds = []
        for n in (1, 2, 3, 4):
            smi = build_oligomer(base, n)
            if smi is None:
                continue
            Xs, _ = featurize([smi], pipe=pkl["pipe"])
            p = float(predict_ensemble(Xs, pkl)[0])
            chain_preds.append((n, p))
        if len(chain_preds) >= 3:
            for n, p in chain_preds:
                rows.append({"polymer": row["smiles"][:60], "n_copies": n,
                             "inv_n": 1.0 / n, "pred_tg": p})
    df = pd.DataFrame(rows, columns=["polymer", "n_copies", "inv_n", "pred_tg"])
    df.to_csv(out_dir / "relation_homologous_series.csv", index=False)

    # Flory-Fox linear fit: Tg vs 1/n per polymer
    fit_rows = []
    if len(df):
        for poly, g in df.groupby("polymer"):
            if len(g) >= 3:
                x = g["inv_n"].values; y = g["pred_tg"].values
                beta = np.polyfit(x, y, 1)
                yhat = np.polyval(beta, x)
                r2 = r2_score(y, yhat)
                fit_rows.append({"polymer": poly, "slope_vs_inv_n": float(beta[0]),
                                 "intercept_Tg_inf": float(beta[1]), "flory_fox_r2": r2})
    fdf = pd.DataFrame(fit_rows)
    fdf.to_csv(out_dir / "relation_flory_fox_fits.csv", index=False)
    med_r2 = float(fdf["flory_fox_r2"].median()) if len(fdf) else float("nan")

    fig, ax = plt.subplots(figsize=(8, 6))
    if not len(df):
        ax.text(0.5, 0.5, "no valid homologous series in sample", ha="center", va="center")
    for poly, g in df.groupby("polymer"):
        ax.plot(g["inv_n"], g["pred_tg"], "o-", alpha=0.7, ms=5)
    ax.invert_xaxis()  # 1/n: left = long chain (saturated), right = monomer
    style_ax(ax, "Homologous Series — predicted Tg vs chain length (Flory–Fox)",
             "1 / n (chain length)", "Predicted Tg (K)")
    ax.text(0.05, 0.05, f"median Flory–Fox R² = {med_r2:.3f}", transform=ax.transAxes,
            va="bottom", fontsize=12)
    save_plot(fig, "relation_homologous_series_plot.png", out_dir)
    print(f"[REL] homologous series: median Flory-Fox R2 = {med_r2:.3f} "
          f"({len(fdf)} polymers, n={len(df)} rows)", flush=True)
    print(f"[REL] relation demo DONE in {time.time() - t0:.0f}s", flush=True)
