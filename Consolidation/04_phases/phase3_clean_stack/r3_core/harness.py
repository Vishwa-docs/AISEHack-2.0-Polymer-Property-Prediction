"""Config-driven experiment harness for Phase 3.

run_config(cfg, output_dir, smoke, data_dir) executes one experiment from a dict:
loads ONLY official data, builds features from scratch, trains with grouped or
scaffold folds, writes metrics.json / predictions.csv / oof.csv / config.json.

Supported kinds: gbm | physics | gnn | assembly | optuna | audit | invariance | shap
No oracle, no external data, no cached artifacts. Fixed seeds throughout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.optimize import nnls
from sklearn.metrics import r2_score

from . import data as d
from . import features as f
from . import metrics as m
from . import nn as nnmod
from . import panels as pn
from . import ssl as sslmod
from . import zoo

RDLogger.DisableLog("rdApp.*")

TARGETS = d.TARGETS
PHASE3_ROOT = Path(__file__).resolve().parents[1]


def _resolve_targets(cfg) -> tuple:
    t = cfg.get("targets", "all")
    return TARGETS if t == "all" else tuple(t)


def _random_smiles(smi: str, rng) -> str:
    try:
        mol = Chem.MolFromSmiles(str(smi).replace("[*]", "*"))
        if mol is None:
            return smi
        out = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        return out if out else smi
    except Exception:
        return smi


def build_features(cfg, texts: list[str], data_dir, seed: int) -> np.ndarray:
    """Base stack + optional mordred + SSL blocks, all fit in-process."""
    fcfg = dict(cfg.get("features", {}) or {})
    # repeat-unit text expansion (dimer/trimer representation tests)
    rep = fcfg.pop("rep_text", 1)
    work_texts = [str(t) * rep if rep > 1 else t for t in texts]
    blocks = []
    if fcfg.get("base", True):
        blocks.append(f.full_feature_stack(
            work_texts, use_svd=fcfg.get("use_svd", True),
            svd_dim=fcfg.get("svd_dim", 64),
            char_n_features=fcfg.get("char_n_features", 1024),
        ))
    if fcfg.get("char_only"):
        from sklearn.decomposition import TruncatedSVD
        char = f.char_ngrams(work_texts, ngram_range=tuple(fcfg.get("char_ngram", (1, 4))),
                             n_features=fcfg.get("char_n_features", 2000))
        svd = TruncatedSVD(n_components=fcfg.get("char_svd", 256), random_state=seed)
        blocks.append(svd.fit_transform(char).astype(np.float32))
    if fcfg.get("high_dim"):
        # H05: dense high-dim fingerprints + char 5-grams via SVD
        mols = f.parse_mols(work_texts)
        blocks.append(f.bit_matrix(mols, 3, 2048))
        blocks.append(f.bit_matrix(mols, 2, 2048))
        char = f.char_ngrams(work_texts, ngram_range=(2, 5), n_features=4096)
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=fcfg.get("high_dim_svd", 256), random_state=seed)
        blocks.append(svd.fit_transform(char).astype(np.float32))
    if fcfg.get("mordred"):
        blocks.append(_mordred_block(work_texts))
    mols_extra = None
    if fcfg.get("polar") or fcfg.get("gasteiger"):
        mols_extra = f.parse_mols(work_texts)
    if fcfg.get("polar"):
        blocks.append(f.polar_moieties_block(mols_extra))
    if fcfg.get("gasteiger"):
        blocks.append(f.gasteiger_separation_block(mols_extra))
    if fcfg.get("conformer3d"):
        blocks.append(f.conformer3d_block(work_texts))
    if fcfg.get("ssl"):
        block = sslmod.ssl_features(fcfg["ssl"], work_texts, data_dir, seed=seed)
        blocks.append(np.asarray(block, dtype=np.float32))
    if not blocks:
        raise ValueError("no feature blocks configured")
    X = np.hstack([np.asarray(b, dtype=np.float32) for b in blocks])
    X[~np.isfinite(X)] = 0.0
    return X


_MORDRED_CACHE = {}


def _mordred_block(texts: list[str]) -> np.ndarray:
    try:
        from mordredcommunity import Calculator, descriptors
    except Exception:
        from mordred import Calculator, descriptors

    calc = Calculator(descriptors)
    out = np.full((len(texts), 1800), np.nan, dtype=np.float64)
    for i, smi in enumerate(texts):
        mol = Chem.MolFromSmiles(str(smi).replace("[*]", "*"))
        if mol is None:
            continue
        try:
            vals = [float(v) if isinstance(v, (int, float)) else np.nan for v in calc(mol)]
            vals = vals[: out.shape[1]]
            out[i, : len(vals)] = vals
        except Exception:
            continue
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    out = imp.fit_transform(out)
    out[~np.isfinite(out)] = 0.0
    return out


def _make_graph(smi: str, expand: str | None = None):
    mol = Chem.MolFromSmiles(str(smi).replace("[*]", "*"))
    if mol is None:
        mol = Chem.MolFromSmiles("C")
    n_atoms = mol.GetNumAtoms()
    feats = np.zeros((n_atoms, 8), dtype=np.float32)
    for a in mol.GetAtoms():
        feats[a.GetIdx()] = [
            min(a.GetAtomicNum(), 30) / 30.0,
            a.GetDegree() / 6.0,
            float(a.GetIsAromatic()),
            float(a.GetHybridization() == Chem.HybridizationType.SP2),
            a.GetTotalNumHs() / 4.0,
            float(a.GetFormalCharge()) / 2.0,
            float(a.IsInRing()),
            min(a.GetMass(), 200) / 200.0,
        ]
    src, dst = [], []
    for b in mol.GetBonds():
        src += [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        dst += [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
    edge_index = np.array([src, dst], dtype=np.int64)
    if expand in ("dimer", "trimer"):
        reps = 2 if expand == "dimer" else 3
        big_x, big_e = [], []
        off = 0
        prev_star = None
        for _ in range(reps):
            big_x.append(feats)
            big_e.append(edge_index + off)
            stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            if prev_star is not None and stars:
                star = stars[0] + off
                # link previous repeat's first star to this repeat's first star
                big_e.append(np.array([[prev_star], [star]]))
                big_e.append(np.array([[star], [prev_star]]))
            if stars:
                prev_star = stars[-1] + off
            off += n_atoms
        return np.vstack(big_x), np.hstack(big_e)
    return feats, edge_index


def _get_folds(train, target, cfg):
    cv = cfg.get("cv", {}) or {}
    n_splits = cv.get("n_splits", 5)
    if cv.get("type", "grouped") == "scaffold":
        folds = pn.scaffold_folds(train, target, n_splits=n_splits, seed=cfg.get("seed", d.SEED))
        # scaffold stratification can yield an empty validation fold for small
        # targets — fall back to structure-grouped folds in that case
        if all(len(va_idx) > 0 for _, va_idx in folds):
            return folds
        print(f"  [warn] scaffold folds have an empty split for {target}; using grouped folds", flush=True)
        return d.grouped_folds(train, target, n_splits=n_splits, seed=cfg.get("seed", d.SEED))
    if cv.get("type", "grouped") == "kmeans":
        # structural-similarity folds: Morgan-fingerprint k-means clusters are
        # kept whole inside a fold (fold-design comparison, ask EDA section)
        folds = pn.kmeans_folds(train, target, n_splits=n_splits, seed=cfg.get("seed", d.SEED))
        if all(len(va_idx) > 0 for _, va_idx in folds):
            return folds
        print(f"  [warn] kmeans folds have an empty split for {target}; using grouped folds", flush=True)
        return d.grouped_folds(train, target, n_splits=n_splits, seed=cfg.get("seed", d.SEED))
    return d.grouped_folds(train, target, n_splits=n_splits, seed=cfg.get("seed", d.SEED))


def _fit_predict_target(cfg, X_tr, y_tr, X_va, X_te_list, sample_weight=None):
    """Fit per-fold model with optional mixup/noise/outlier handling; predict val + test views."""
    opts = cfg.get("model_opts", {}) or {}
    y = np.asarray(y_tr, float)
    X = np.asarray(X_tr, float)
    if opts.get("outlier_sigma"):
        # two-pass outlier removal on residuals of a quick ridge
        from sklearn.linear_model import Ridge
        quick = Ridge(alpha=10.0).fit(X, y)
        resid = np.abs(y - quick.predict(X))
        keep = resid <= opts["outlier_sigma"] * (resid.std() + 1e-9)
        X, y = X[keep], y[keep]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, float)[keep]
    if sample_weight is not None:
        # row weights (covariate-shift / curation) are row-aligned; skip
        # mixup/noise augmentation so the weights stay aligned.
        Xa, ya = X, y
    else:
        Xa, ya = zoo.augment_xy(X, y, mixup=opts.get("mixup", 0.0), noise=opts.get("noise", 0.0),
                                seed=cfg.get("seed", d.SEED))
    spec = dict(cfg.get("model", {"type": "lgbm"}))
    model = zoo.make_model(spec, seed=cfg.get("seed", d.SEED))
    if sample_weight is not None:
        try:
            model.fit(Xa, ya, sample_weight=sample_weight)
        except TypeError:
            model.fit(Xa, ya)
    else:
        model.fit(Xa, ya)
    va_pred = model.predict(np.asarray(X_va, float))
    te_preds = [model.predict(np.asarray(Xt, float)) for Xt in X_te_list]
    return va_pred, te_preds


def run_cv(cfg, train, test, X_all, n_aug_views=0, targets=None, row_weights=None):
    """Generic grouped/scaffold/kmeans CV loop. Returns (oof, test_preds, per_target dict).

    X_all layout: [train][aug_train*n_aug_views][test][test_views*n_aug_views]
    row_weights: per-train-row sample weights (covariate-shift / curation), or None.
    """
    seed = cfg.get("seed", d.SEED)
    targets = targets or _resolve_targets(cfg)
    n_tr = len(train)
    n_te = len(test)
    X_train_base = X_all[:n_tr]
    X_test_views = X_all[n_tr + n_tr * n_aug_views:]
    X_test_base = X_test_views[:n_te]
    test_views = int((cfg.get("tta", {}) or {}).get("test_views", 0))

    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    oof = np.full(n_tr, np.nan)
    tlist = list(targets)
    test_mat = np.zeros((n_te, len(tlist)))
    per_target = {}
    test_tt = test["target_type"].to_numpy(object)

    for target in targets:
        ti = tlist.index(target)
        mask = tt == target
        pos = np.where(mask)[0]
        if len(pos) < 10:
            continue
        folds = _get_folds(train, target, cfg)
        te_mask = test_tt == target
        fold_te = []
        for tr_idx, va_idx in folds:
            X_tr_rows = list(tr_idx)
            if n_aug_views:
                for v in range(n_aug_views):
                    start = n_tr + v * n_tr
                    X_tr_rows.extend(start + tr_idx)
                y_tr = np.concatenate([y_full[tr_idx]] * (1 + n_aug_views))
            else:
                y_tr = y_full[tr_idx]
            X_tr = X_all[X_tr_rows]
            X_va = X_train_base[va_idx]
            sw = None
            if row_weights is not None and not n_aug_views:
                sw = row_weights[tr_idx]
            views = []
            if test_views and te_mask.any():
                Xt = np.mean([X_test_views[v * n_te:(v + 1) * n_te][te_mask] for v in range(test_views)], axis=0)
                views = [Xt]
            elif te_mask.any():
                views = [X_test_base[te_mask]]
            va_pred, te_preds = _fit_predict_target(cfg, X_tr, y_tr, X_va, views, sample_weight=sw)
            oof[va_idx] = va_pred
            if te_preds:
                fold_te.append(te_preds[0])
        valid = np.isfinite(y_full) & np.isfinite(oof) & mask
        r2 = float(r2_score(y_full[valid], oof[valid])) if valid.sum() >= 2 else float("nan")
        per_target[target] = {"r2": r2, "rows": int(mask.sum())}
        if fold_te and te_mask.any():
            test_mat[te_mask, ti] = np.mean(fold_te, axis=0)
        print(f"  [{target}] OOF R2 = {r2:.4f}  (n={int(mask.sum())})", flush=True)
    return oof, test_mat, per_target


# ---------------------------------------------------------------------------
# Partner-feature injection + coordinate identities (physics kind)
# ---------------------------------------------------------------------------

def run_physics(cfg, train, test, X_all, targets):
    """Stage 1 predicts all targets; partner stage-1 predictions become features
    (and coordinate axes, e.g. ionic = eps - nc^2) for configured targets."""
    seed = cfg.get("seed", d.SEED)
    partners = cfg.get("partner", {}) or {}
    coords = cfg.get("coordinate", {}) or {}  # {"eps": "ionic"}
    n_tr, n_te = len(train), len(test)
    tlist = list(targets)

    oof1, te1, per1 = run_cv(cfg, train, test, X_all, targets=targets)
    if not partners and not coords:
        return oof1, te1, per1

    X_train_base = X_all[:n_tr]
    X_test_base = X_all[n_tr:n_tr + n_te]
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    oof = oof1.copy()
    te = te1.copy()
    per_target = dict(per1)
    handled = set(partners) | set(coords)

    for target in tlist:
        if target not in handled:
            continue
        mask = tt == target
        if mask.sum() < 10:
            continue
        te_mask = test_tt == target
        plist = [p for p in partners.get(target, []) if p in tlist]
        coord = coords.get(target)
        coord_type = None
        coord_partner = None
        if isinstance(coord, str):
            coord_type = coord                      # legacy "ionic"
        elif isinstance(coord, dict):
            coord_type = coord.get("type")
            coord_partner = coord.get("partner")
        if coord_partner and coord_partner in tlist and coord_partner not in plist:
            plist.append(coord_partner)
        p_idx = [tlist.index(p) for p in plist]
        P_tr = np.nan_to_num(oof1[:, p_idx]) if p_idx else np.zeros((n_tr, 0))
        P_te = te1[np.ix_(np.where(te_mask)[0], p_idx)] if p_idx else np.zeros((int(te_mask.sum()), 0))
        folds = _get_folds(train, target, cfg)
        fold_te = []
        for tr_idx, va_idx in folds:
            X_tr = np.hstack([X_all[tr_idx], P_tr[tr_idx]])
            X_va = np.hstack([X_train_base[va_idx], P_tr[va_idx]])
            y_tr = y_full[tr_idx]
            scale = 1.0
            add_va = np.zeros(len(va_idx))
            add_te = np.zeros(int(te_mask.sum()))
            if coord_type == "ionic" and "nc" in tlist:
                nc_i = tlist.index("nc")
                y_tr = y_tr - np.nan_to_num(oof1[tr_idx, nc_i]) ** 2
                add_va = np.nan_to_num(oof1[va_idx, nc_i]) ** 2
                add_te = np.nan_to_num(te1[te_mask, nc_i]) ** 2
            elif coord_type == "mulliken" and coord_partner in tlist:
                # chi = (ei + eea)/2 ; ei = 2*chi - eea  (ask sec 1.2)
                pi = tlist.index(coord_partner)
                y_tr = 0.5 * (y_tr + np.nan_to_num(oof1[tr_idx, pi]))
                add_va = -np.nan_to_num(oof1[va_idx, pi])
                add_te = -np.nan_to_num(te1[te_mask, pi])
                scale = 2.0
            elif coord_type == "gapres" and coord_partner in tlist:
                # residual identity (ei - eea, or egb - egc); add partner back
                pi = tlist.index(coord_partner)
                y_tr = y_tr - np.nan_to_num(oof1[tr_idx, pi])
                add_va = np.nan_to_num(oof1[va_idx, pi])
                add_te = np.nan_to_num(te1[te_mask, pi])
            spec = dict(cfg.get("model", {"type": "lgbm"}))
            mdl = zoo.make_model(spec, seed=seed)
            mdl.fit(np.nan_to_num(X_tr), y_tr)
            oof[va_idx] = scale * mdl.predict(np.nan_to_num(X_va)) + add_va
            Xte = X_test_base[te_mask]
            Xte = np.hstack([Xte, P_te]) if P_te.size else Xte
            fold_te.append(scale * mdl.predict(np.nan_to_num(Xte)) + add_te)
        if fold_te and te_mask.any():
            te[te_mask, tlist.index(target)] = np.mean(fold_te, axis=0)
        valid = np.isfinite(y_full) & np.isfinite(oof) & mask
        r2 = float(r2_score(y_full[valid], oof[valid])) if valid.sum() >= 2 else float("nan")
        per_target[target] = {"r2": r2, "rows": int(mask.sum())}
        print(f"  [physics:{target}] OOF R2 = {r2:.4f}", flush=True)
    return oof, te, per_target


# ---------------------------------------------------------------------------
# GNN kind
# ---------------------------------------------------------------------------

def run_gnn(cfg, train, test, targets):
    params = dict(cfg.get("gnn", {}))
    expand = params.pop("expand", None)
    graphs_tr = [_make_graph(s, expand) for s in train["smiles"]]
    graphs_te = [_make_graph(s, expand) for s in test["smiles"]]
    seed = cfg.get("seed", d.SEED)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    tlist = list(targets)
    oof = np.full(len(train), np.nan)
    te = np.zeros((len(test), len(tlist)))
    per_target = {}
    for target in targets:
        mask = tt == target
        if mask.sum() < 10:
            continue
        folds = _get_folds(train, target, cfg)
        te_mask = test_tt == target
        fold_te = []
        for tr_idx, va_idx in folds:
            model = nnmod.GNNRegressor(dict(params), seed=seed)
            model.fit([graphs_tr[i] for i in tr_idx], y_full[tr_idx])
            oof[va_idx] = model.predict([graphs_tr[i] for i in va_idx])
            if te_mask.any():
                fold_te.append(model.predict([graphs_te[i] for i in np.where(te_mask)[0]]))
        valid = np.isfinite(y_full) & np.isfinite(oof) & mask
        r2 = float(r2_score(y_full[valid], oof[valid])) if valid.sum() >= 2 else float("nan")
        per_target[target] = {"r2": r2, "rows": int(mask.sum())}
        if fold_te and te_mask.any():
            te[te_mask, tlist.index(target)] = np.mean(fold_te, axis=0)
        print(f"  [gnn:{target}] OOF R2 = {r2:.4f}", flush=True)
    return oof, te, per_target


# ---------------------------------------------------------------------------
# Multi-task MLP kind
# ---------------------------------------------------------------------------

def run_multitask_mlp(cfg, train, test, X_all, targets):
    import torch

    seed = cfg.get("seed", d.SEED)
    pcfg = dict(cfg.get("mlp", {}))
    n_tr, n_te = len(train), len(test)
    tlist = list(targets)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    oof = np.full(n_tr, np.nan)
    te = np.zeros((n_te, len(tlist)))
    per_target = {}

    def physics_penalty(preds, xb):
        terms = []
        for a, b, c in (("ei", "eea", "egc"),):
            if a in tlist and b in tlist and c in tlist:
                terms.append((preds[:, tlist.index(a)] - preds[:, tlist.index(b)]
                              - preds[:, tlist.index(c)]) ** 2)
        if not terms:
            return torch.tensor(0.0, device=nnmod.DEVICE)
        return torch.stack(terms).mean()

    base_target = max(tlist, key=lambda t: (tt == t).sum())
    for tr_idx, va_idx in _get_folds(train, base_target, cfg):
        Y_tr = np.full((len(tr_idx), len(tlist)), np.nan)
        for j, t in enumerate(tlist):
            mask_t = tt[tr_idx] == t
            Y_tr[mask_t, j] = y_full[tr_idx][mask_t]
        net = nnmod.fit_multitask_mlp(
            X_all[tr_idx], Y_tr, np.isfinite(Y_tr),
            hidden=pcfg.get("hidden", 256), depth=pcfg.get("depth", 3),
            dropout=pcfg.get("dropout", 0.2), lr=pcfg.get("lr", 1e-3),
            epochs=pcfg.get("epochs", 200), batch_size=pcfg.get("batch", 256),
            seed=seed, physics_penalty=physics_penalty if pcfg.get("physics") else None,
            physics_lambda=pcfg.get("physics_lambda", 0.05),
        )
        va_p = nnmod.predict_multitask(net, X_all[va_idx])
        te_p = nnmod.predict_multitask(net, X_all[n_tr:n_tr + n_te])
        for j, t in enumerate(tlist):
            sel_va = tt[va_idx] == t
            oof[va_idx[sel_va]] = va_p[sel_va, j]
            sel_te = test_tt == t
            te[sel_te, j] = te_p[sel_te, j]
    for t in tlist:
        mask = tt == t
        valid = np.isfinite(y_full) & np.isfinite(oof) & mask
        r2 = float(r2_score(y_full[valid], oof[valid])) if valid.sum() >= 2 else float("nan")
        per_target[t] = {"r2": r2, "rows": int(mask.sum())}
        print(f"  [mlp:{t}] OOF R2 = {r2:.4f}", flush=True)
    return oof, te, per_target


# ---------------------------------------------------------------------------
# Assembly (blend prior arm outputs by per-target NNLS on OOF)
# ---------------------------------------------------------------------------

def _arm_paths(arms):
    out = []
    for a in arms:
        p = Path(a)
        if not p.exists():
            p = PHASE3_ROOT / "outputs_and_logs" / "output" / a
        out.append(p)
    return out


def run_assembly(cfg, train, test):
    arms = _arm_paths(cfg["arms"])
    oofs, tes, names = [], [], []
    for p in arms:
        oof_df = pd.read_csv(p / "oof.csv")
        pred_df = pd.read_csv(p / "predictions.csv")
        assert len(oof_df) == len(train) and len(pred_df) == len(test), f"arm {p} row mismatch"
        oofs.append(oof_df["oof"].to_numpy(float))
        tes.append(pred_df["target"].to_numpy(float))
        names.append(p.name)
    A = np.column_stack(oofs)
    T = np.column_stack(tes)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    blend_oof = np.full(len(train), np.nan)
    blend_te = np.zeros(len(test))
    weights = {}
    for t in dict.fromkeys(tt):
        mask = tt == t
        te_mask = test_tt == t
        if mask.sum() < 5:
            continue
        At, yt = A[mask], y_full[mask]
        if cfg.get("blend", "nnls") == "mean":
            w = np.ones(A.shape[1]) / A.shape[1]
        else:
            w, _ = nnls(np.nan_to_num(At), yt)
            if w.sum() <= 0:
                w = np.ones(A.shape[1]) / A.shape[1]
        blend_oof[mask] = np.nan_to_num(At) @ w
        blend_te[te_mask] = np.nan_to_num(T[te_mask]) @ w
        weights[t] = {n: float(x) for n, x in zip(names, w)}
    per_target = {}
    for t in dict.fromkeys(tt):
        mask = tt == t
        valid = np.isfinite(y_full) & np.isfinite(blend_oof) & mask
        r2 = float(r2_score(y_full[valid], blend_oof[valid])) if valid.sum() >= 2 else float("nan")
        per_target[t] = {"r2": r2, "rows": int(mask.sum())}
        print(f"  [blend:{t}] OOF R2 = {r2:.4f}", flush=True)
    return blend_oof, blend_te, per_target, weights


# ---------------------------------------------------------------------------
# Random-search (optuna-style) kind
# ---------------------------------------------------------------------------

def run_optuna(cfg, train, test, X_all, targets):
    opts = cfg.get("search", {})
    kind = opts.get("model_kind", "lgbm")
    trials = opts.get("trials", 25)
    seed = cfg.get("seed", d.SEED)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    tlist = list(targets)
    oof = np.full(len(train), np.nan)
    te = np.zeros((len(test), len(tlist)))
    per_target = {}
    best_params = {}
    X_train = np.nan_to_num(X_all[: len(train)])
    for target in targets:
        mask = tt == target
        if mask.sum() < 20:
            continue
        folds = _get_folds(train, target, cfg)
        folds_m = [(np.where(mask)[0][tr], np.where(mask)[0][va]) for tr, va in folds]
        params, cvr2, _trials = zoo.random_search(kind, X_train, y_full, folds_m,
                                                  n_trials=trials, seed=seed)
        best_params[target] = {"params": params, "cv_r2": cvr2}
        cfg2 = dict(cfg)
        cfg2["model"] = {"type": kind, **params}
        oof_t, te_t, per_t = run_cv(cfg2, train, test, X_all, targets=(target,))
        oof = np.where(np.isfinite(oof_t), oof_t, oof)
        te_mask = test_tt == target
        if te_mask.any():
            te[te_mask, tlist.index(target)] = te_t[te_mask, 0]
        per_target[target] = per_t.get(target, {"r2": cvr2, "rows": int(mask.sum())})
    return oof, te, per_target, best_params


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------

def write_artifacts(cfg, output_dir, train, test, oof, te_flat, per_target, extra=None, oof_align=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r2s = {t: v.get("r2", float("nan")) for t, v in per_target.items()}
    finite = [v for v in r2s.values() if np.isfinite(v)]
    metrics = {
        "exp": cfg.get("exp_id", cfg.get("name", "unknown")),
        "name": cfg.get("name"),
        "kind": cfg.get("kind", "gbm"),
        "per_target": per_target,
        "per_target_r2": r2s,
        "mean_r2": float(np.mean(finite)) if finite else float("nan"),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    if extra:
        metrics.update(extra)
    pred = pd.DataFrame({"id": test["id"].to_numpy(), "target": np.round(te_flat, 6)})
    pred.to_csv(output_dir / "predictions.csv", index=False)
    if oof_align is not None:
        orig_train, kept_orig_idx = oof_align
        oof_full = np.full(len(orig_train), np.nan)
        oof_full[kept_orig_idx] = oof
        oof_train, oof_col = orig_train, oof_full
    else:
        oof_train, oof_col = train, oof
    pd.DataFrame({
        "smiles": oof_train["smiles"], "target_type": oof_train["target_type"],
        "target": oof_train["target"], "oof": oof_col,
    }).to_csv(output_dir / "oof.csv", index=False)
    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    with open(output_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    with open(output_dir / "decision.md", "w") as fh:
        fh.write(f"# {cfg.get('name')}\n\nmean OOF R2 = {metrics['mean_r2']:.4f}\n\n")
        for t, v in r2s.items():
            fh.write(f"- {t}: {v:.4f}\n")
    return metrics


# ---------------------------------------------------------------------------
# Audit / invariance / SHAP kinds
# ---------------------------------------------------------------------------

def run_audit(cfg, train, test, output_dir):
    arms = _arm_paths(cfg["arms"])
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    report = {}
    for p in arms:
        oof_df = pd.read_csv(p / "oof.csv")
        A = oof_df["oof"].to_numpy(float)
        per = {}
        for t in dict.fromkeys(tt):
            mask = tt == t
            valid = np.isfinite(y_full) & np.isfinite(A) & mask
            per[t] = float(r2_score(y_full[valid], A[valid])) if valid.sum() >= 2 else float("nan")
        finite = [v for v in per.values() if np.isfinite(v)]
        report[p.name] = {"per_target_r2": per, "mean_r2": float(np.mean(finite)) if finite else None}
        print(f"  [audit:{p.name}] mean = {report[p.name]['mean_r2']}", flush=True)
    try:
        sim = pn.tanimoto_similarity(list(train["smiles"]), list(train["smiles"]))
        A = pd.read_csv(arms[0] / "oof.csv")["oof"].to_numpy(float)
        bins = {"low_<0.3": sim < 0.3, "mid_0.3-0.7": (sim >= 0.3) & (sim < 0.7), "high_>0.7": sim >= 0.7}
        bins_out = {}
        for bname, bmask in bins.items():
            per = {}
            for t in dict.fromkeys(tt):
                mask = (tt == t) & bmask
                valid = np.isfinite(y_full) & np.isfinite(A) & mask
                if valid.sum() >= 5:
                    per[t] = float(r2_score(y_full[valid], A[valid]))
            bins_out[bname] = per
        report["similarity_bins"] = bins_out
    except Exception as exc:
        report["similarity_bins"] = {"error": str(exc)}
    try:
        report["overlap_audit"] = d.overlap_audit(train, test)
    except Exception as exc:
        report["overlap_audit"] = {"error": str(exc)}
    try:
        # shift-matched reweighted R2: density-match validation rows to the
        # test-set nearest-train similarity distribution (ask sec 5.1)
        sim_tr = pn.tanimoto_similarity(list(train["smiles"]), list(train["smiles"]))
        sim_te = pn.tanimoto_similarity(list(test["smiles"]), list(train["smiles"]))
        edges = np.linspace(0.0, 1.0, 21)
        hist_tr, _ = np.histogram(sim_tr, bins=edges)
        hist_te, _ = np.histogram(sim_te, bins=edges)
        wgt = np.clip(hist_te / (hist_tr + 1e-9), 0.1, 10.0)
        bin_idx = np.clip(np.digitize(sim_tr, edges) - 1, 0, 19)
        wrow = wgt[bin_idx]
        sm = {}
        for t in dict.fromkeys(tt):
            mask = tt == t
            valid = np.isfinite(y_full) & np.isfinite(A) & mask
            if valid.sum() >= 10:
                yv, pv, wv = y_full[valid], A[valid], wrow[valid]
                yb = float(np.average(yv, weights=wv))
                ss_res = float(np.sum(wv * (yv - pv) ** 2))
                ss_tot = float(np.sum(wv * (yv - yb) ** 2))
                sm[t] = float(1.0 - ss_res / (ss_tot + 1e-9))
        report["shift_matched_r2"] = sm
    except Exception as exc:
        report["shift_matched_r2"] = {"error": str(exc)}
    means = [r["mean_r2"] for r in report.values() if isinstance(r, dict) and r.get("mean_r2")]
    metrics = {"kind": "audit", "arms": report,
               "mean_r2": float(max(means)) if means else float("nan")}
    out = Path(output_dir)
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    src = arms[0]
    if (src / "predictions.csv").exists():
        pd.read_csv(src / "predictions.csv").to_csv(out / "predictions.csv", index=False)
        pd.read_csv(src / "oof.csv").to_csv(out / "oof.csv", index=False)
    return metrics


def run_invariance(cfg, train, test, output_dir, data_dir):
    """SMILES-permutation invariance: quick ridge per target, predict each test
    polymer under K randomized SMILES views, report per-polymer std."""
    seed = cfg.get("seed", d.SEED)
    rng = np.random.default_rng(seed)
    K = int((cfg.get("invariance", {}) or {}).get("views", 20))
    fcfg = {"features": cfg.get("features", {})}
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    from sklearn.linear_model import Ridge

    X_tr = np.nan_to_num(build_features(fcfg, list(train["smiles"]), data_dir, seed))
    view_preds = np.zeros((K, len(test)))
    for k in range(K):
        test_texts = [_random_smiles(s, rng) for s in test["smiles"]]
        Xv = np.nan_to_num(build_features(fcfg, test_texts, data_dir, seed + k + 1))
        for t in dict.fromkeys(tt):
            mask = tt == t
            te_mask = test_tt == t
            mdl = Ridge(alpha=10.0).fit(X_tr[mask], y_full[mask])
            view_preds[k, te_mask] = mdl.predict(Xv[te_mask])
        print(f"  invariance view {k + 1}/{K}", flush=True)
    stds = view_preds.std(axis=0)
    metrics = {
        "kind": "invariance",
        "mean_pred_std": float(stds.mean()),
        "p95_pred_std": float(np.quantile(stds, 0.95)),
        "frac_high_var": float((stds > np.quantile(stds, 0.95)).mean()),
        "mean_r2": float("nan"),
    }
    out = Path(output_dir)
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    np.save(out / "view_preds.npy", view_preds)
    return metrics


def run_shap(cfg, train, X_all, output_dir):
    import shap

    seed = cfg.get("seed", d.SEED)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    X_tr = np.nan_to_num(X_all[: len(train)])
    out = {}
    for t in dict.fromkeys(tt):
        mask = tt == t
        if mask.sum() < 20:
            continue
        model = zoo.make_model({"type": "lgbm"}, seed=seed)
        model.fit(X_tr[mask], y_full[mask])
        expl = shap.TreeExplainer(model)
        sample = X_tr[mask][: min(500, int(mask.sum()))]
        sv = expl.shap_values(sample)
        mean_abs = np.abs(sv).mean(axis=0)
        top = np.argsort(mean_abs)[::-1][:15]
        out[t] = {f"feature_{i}": float(mean_abs[i]) for i in top}
        print(f"  [shap:{t}] top features recorded", flush=True)
    metrics = {"kind": "shap", "top_features": out, "mean_r2": float("nan")}
    with open(Path(output_dir) / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    return metrics


# ---------------------------------------------------------------------------
# Master dispatcher
# ---------------------------------------------------------------------------

def run_config(cfg: dict, output_dir, smoke: bool = False, data_dir=None) -> dict:
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg["seed"] = cfg.get("seed", d.SEED)
    kind = cfg.get("kind", "gbm")

    if kind == "audit":
        train, test = d.load_official_data(data_dir)
        return run_audit(cfg, train, test, output_dir)

    train, test = d.load_official_data(data_dir)
    train, test = d.add_structure_keys(train, test)
    if smoke:
        keep = [train[train["target_type"] == t].head(40) for t in TARGETS]
        train = pd.concat(keep, ignore_index=True)
        train, _ = d.add_structure_keys(train, test)

    targets = _resolve_targets(cfg)

    orig_train = None
    kept_orig_idx = None
    if kind not in ("audit", "assembly", "invariance"):
        orig_train = train.copy()
        train, kept_orig_idx = _apply_curation(cfg, train, test)

    if kind == "invariance":
        if smoke:
            metrics = {"kind": "invariance", "mean_r2": float("nan"), "smoke": True}
            with open(output_dir / "metrics.json", "w") as fh:
                json.dump(metrics, fh, indent=2)
            return metrics
        return run_invariance(cfg, train, test, output_dir, data_dir)

    if kind == "assembly":
        blend_oof, blend_te, per_target, weights = run_assembly(cfg, train, test)
        metrics = write_artifacts(cfg, output_dir, train, test, blend_oof, blend_te,
                                  per_target, extra={"blend_weights": weights})
        print("mean OOF R2 =", metrics["mean_r2"])
        return metrics

    # Feature-based kinds
    rng = np.random.default_rng(cfg["seed"])
    n_aug = int(((cfg.get("tta") or {}).get("train_aug", 0)))
    n_views = int(((cfg.get("tta") or {}).get("test_views", 0)))
    texts = list(train["smiles"])
    aug_texts = [_random_smiles(s, rng) for _ in range(n_aug) for s in train["smiles"]] if n_aug else []
    view_texts = [_random_smiles(s, rng) for _ in range(n_views) for s in test["smiles"]] if n_views else []
    all_texts = texts + aug_texts + list(test["smiles"]) + view_texts
    print(f"[{cfg.get('name')}] building features for {len(all_texts)} rows ...", flush=True)
    X_all = build_features(cfg, all_texts, data_dir, cfg["seed"])

    if kind == "shap":
        return run_shap(cfg, train, X_all, output_dir)

    if kind == "gnn":
        oof, te, per_target = run_gnn(cfg, train, test, targets)
    elif kind in ("physics", "partner"):
        oof, te, per_target = run_physics(cfg, train, test, X_all, targets)
    elif kind == "mlp":
        oof, te, per_target = run_multitask_mlp(cfg, train, test, X_all, targets)
    elif kind == "optuna":
        oof, te, per_target, best = run_optuna(cfg, train, test, X_all, targets)
    elif kind == "pseudo":
        oof, te, per_target = run_pseudo(cfg, train, test, X_all, targets, data_dir)
    elif kind == "mixture":
        oof, te, per_target = run_mixture(cfg, train, test, X_all, targets)
    elif kind == "recalib":
        oof, te, per_target, calib = run_recalib(cfg, train, test, X_all, targets)
    elif kind == "shiftweight":
        oof, te, per_target = run_shiftweight(cfg, train, test, X_all, targets, data_dir)
    elif kind == "matfac":
        oof, te, per_target = run_matfac(cfg, train, test, X_all, targets)
    elif kind == "uncertainty":
        oof, te, per_target = run_uncertainty(cfg, train, test, X_all, targets)
    else:  # gbm (also serves transformer-finetune heads via ssl mlm features)
        rw = _curation_weights(cfg, train, test)
        oof, te, per_target = run_cv(cfg, train, test, X_all, n_aug_views=n_aug, targets=targets,
                                     row_weights=None if np.all(rw == 1.0) else rw)

    te_flat = _flat_test(te, test, tlist=list(targets))
    extra = {"best_params": best} if kind == "optuna" else None
    if kind == "recalib":
        extra = (extra or {}) | {"calibration": calib}
    oof_align = None
    if orig_train is not None and kept_orig_idx is not None and len(kept_orig_idx) != len(orig_train):
        oof_align = (orig_train, kept_orig_idx)
    metrics = write_artifacts(cfg, output_dir, train, test, oof, te_flat, per_target,
                              extra=extra, oof_align=oof_align)
    metrics["elapsed_seconds"] = float(time.time() - start)
    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print("mean OOF R2 =", metrics["mean_r2"])
    return metrics


def _flat_test(te_mat, test, tlist=None):
    """(n_te, T) matrix -> flat per-row predictions using each row's target_type."""
    te_flat = np.zeros(len(test))
    tlist = tlist or list(dict.fromkeys(test["target_type"]))
    test_tt = test["target_type"].to_numpy(object)
    for ti, t in enumerate(tlist):
        sel = test_tt == t
        te_flat[sel] = te_mat[sel, ti]
    return te_flat


# ---------------------------------------------------------------------------
# Pseudo-labeling kind (self-training on official PI1M / smile_r3 SMILES)
# ---------------------------------------------------------------------------

def run_pseudo(cfg, train, test, X_all, targets, data_dir):
    """Train base model -> predict unlabeled official SMILES -> add top-k most
    confident pseudo-labels -> retrain via normal CV. Official data only."""
    seed = cfg.get("seed", d.SEED)
    pcfg = cfg.get("pseudo", {})
    corpus = sslmod.load_corpus(pcfg.get("corpus", "pi1m"), data_dir,
                                n=pcfg.get("n", 20000), seed=seed,
                                exclude=list({t.replace("[*]", "*") for t in train["smiles"]}))
    print(f"  [pseudo] corpus rows: {len(corpus)}", flush=True)
    Xc = np.nan_to_num(build_features(cfg, corpus, data_dir, seed))
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    X_tr_all = np.nan_to_num(X_all[: len(train)])
    tlist = list(targets)
    te = np.zeros((len(test), len(tlist)))
    oof = np.full(len(train), np.nan)
    per_target = {}
    test_tt = test["target_type"].to_numpy(object)
    for target in targets:
        mask = tt == target
        if mask.sum() < 20:
            continue
        spec = dict(cfg.get("model", {"type": "lgbm"}))
        base = zoo.make_model(spec, seed=seed)
        base.fit(X_tr_all[mask], y_full[mask])
        corpus_pred = base.predict(Xc)
        center = float(np.mean(y_full[mask]))
        conf = -np.abs(corpus_pred - center)
        top_k = int(pcfg.get("top_k", 2000))
        sel = np.argsort(conf)[:top_k]
        X_aug = np.vstack([X_tr_all[mask], Xc[sel]])
        y_aug = np.concatenate([y_full[mask], corpus_pred[sel] * pcfg.get("weight", 1.0)])
        folds = _get_folds(train, target, cfg)
        te_mask = test_tt == target
        fold_te = []
        for tr_idx, va_idx in folds:
            mdl = zoo.make_model(spec, seed=seed)
            mdl.fit(X_aug, y_aug)
            oof[va_idx] = mdl.predict(np.nan_to_num(X_all[va_idx]))
            if te_mask.any():
                fold_te.append(mdl.predict(np.nan_to_num(X_all[len(train):len(train) + len(test)][te_mask])))
        valid = np.isfinite(y_full) & np.isfinite(oof) & mask
        r2 = float(r2_score(y_full[valid], oof[valid])) if valid.sum() >= 2 else float("nan")
        per_target[target] = {"r2": r2, "rows": int(mask.sum())}
        if fold_te and te_mask.any():
            te[te_mask, tlist.index(target)] = np.mean(fold_te, axis=0)
        print(f"  [pseudo:{target}] OOF R2 = {r2:.4f}", flush=True)
    return oof, te, per_target


# ---------------------------------------------------------------------------
# New Phase-3 kinds added for the advanced-strategy batch
# (mixture / recalib / shiftweight / matfac / uncertainty + curation)
# ---------------------------------------------------------------------------

def _pooled_folds(train, positions, n_splits=5, seed=2026):
    """Structure-grouped folds over an arbitrary set of train positions."""
    sub = train.iloc[positions]
    groups = sub["structure_index"].to_numpy(int)
    unique, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    group_fold = np.empty(len(unique), dtype=int)
    for i, g in enumerate(order):
        group_fold[g] = i % n_splits
    folds = []
    for fold in range(n_splits):
        val = group_fold[inv] == fold
        folds.append((positions[~val], positions[val]))
    return folds


def run_mixture(cfg, train, test, X_all, targets):
    """Grouped multi-output trees (ask: grouped multi-output trees on physical
    groups with missing-label masks). One model per group (electronic / optical /
    thermal / all); rows pooled across the group's targets with a target-id
    one-hot covariate, so each model sees every group row but predicts only the
    requested target column."""
    seed = cfg.get("seed", d.SEED)
    groups = cfg.get("mixture", {}).get("groups") or {"all": list(targets)}
    n_tr = len(train)
    n_te = len(test)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    tlist = list(targets)
    n_splits = (cfg.get("cv", {}) or {}).get("n_splits", 5)
    oof = np.full(n_tr, np.nan)
    te = np.zeros((n_te, len(tlist)))
    per_target = {}
    X_train = np.nan_to_num(X_all[:n_tr])
    X_test = np.nan_to_num(X_all[n_tr:n_tr + n_te])
    full_tt = np.concatenate([tt, test_tt])
    oh = np.zeros((n_tr + n_te, len(tlist)), dtype=np.float32)
    for t in tlist:
        oh[full_tt == t, tlist.index(t)] = 1.0
    Xg = np.hstack([np.vstack([X_train, X_test]), oh])

    for gname, gt in groups.items():
        gt = [t for t in gt if t in tlist]
        if not gt:
            continue
        gmask = np.isin(tt, gt)
        positions = np.where(gmask)[0]
        if len(positions) < 10:
            continue
        folds = _pooled_folds(train, positions, n_splits=n_splits, seed=seed)
        fold_te = {t: [] for t in gt}
        for tr_idx, va_idx in folds:
            mdl = zoo.make_model(dict(cfg.get("model", {"type": "lgbm"})), seed=seed)
            mdl.fit(Xg[tr_idx], y_full[tr_idx])
            oof[va_idx] = mdl.predict(Xg[va_idx])
            for t in gt:
                te_mask = test_tt == t
                if te_mask.any():
                    fold_te[t].append(mdl.predict(Xg[n_tr:][te_mask]))
        for t in gt:
            if fold_te[t]:
                te[test_tt == t, tlist.index(t)] = np.mean(fold_te[t], axis=0)
            mask = tt == t
            valid = np.isfinite(y_full) & np.isfinite(oof) & mask
            r2 = float(r2_score(y_full[valid], oof[valid])) if valid.sum() >= 2 else float("nan")
            per_target[t] = {"r2": r2, "rows": int(mask.sum())}
            print(f"  [mixture:{gname}:{t}] OOF R2 = {r2:.4f}", flush=True)
    return oof, te, per_target


def run_recalib(cfg, train, test, X_all, targets):
    """Affine recalibration on OOF predictions (ask: per-target affine/isotonic
    recalibration guarantees R2 >= uncalibrated). Fits alpha,beta per target on
    OOF vs y (shrunk), applies to test, reports raw + calibrated R2."""
    seed = cfg.get("seed", d.SEED)
    oof, te, per = run_cv(cfg, train, test, X_all, targets=targets)
    n_tr = len(train)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    tlist = list(targets)
    oof_c = oof.copy()
    te_c = te.copy()
    calib = {}
    for t in targets:
        mask = tt == t
        if mask.sum() < 10:
            continue
        y = y_full[mask]
        pp = oof[mask]
        v = np.isfinite(y) & np.isfinite(pp)
        if v.sum() < 5:
            continue
        yy, ppv = y[v], pp[v]
        varp = float(np.var(ppv))
        beta = float(np.cov(ppv, yy)[0, 1] / (varp + 1e-9)) if varp > 1e-12 else 1.0
        beta = float(np.clip(beta, 0.5, 1.5))
        alpha = float(np.mean(yy) - beta * np.mean(ppv))
        oof_c[mask] = alpha + beta * oof[mask]
        te_mask = test_tt == t
        if te_mask.any():
            te_c[te_mask, tlist.index(t)] = alpha + beta * te[te_mask, tlist.index(t)]
        valid = np.isfinite(y_full) & np.isfinite(oof_c) & mask
        r2c = float(r2_score(y_full[valid], oof_c[valid])) if valid.sum() >= 2 else float("nan")
        calib[t] = {"alpha": alpha, "beta": beta, "raw_r2": per.get(t, {}).get("r2"), "cal_r2": r2c}
        print(f"  [recalib:{t}] raw={per.get(t, {}).get('r2', float('nan')):.4f} cal={r2c:.4f} (a={alpha:.4f}, b={beta:.4f})", flush=True)
    return oof_c, te_c, per, calib


def run_shiftweight(cfg, train, test, X_all, targets, data_dir):
    """Covariate-shift importance weighting (ask EDA/calibration): density-ratio
    of train vs an official unlabeled sample (smile_r3 / PI1M); weight = p/(1-p),
    clipped, mean-normalised; then grouped CV with sample weights."""
    seed = cfg.get("seed", d.SEED)
    pcfg = cfg.get("shiftweight", {}) or {}
    corpus = sslmod.load_corpus(pcfg.get("corpus", "smile_r3"), data_dir,
                                n=pcfg.get("n", 60000), seed=seed,
                                exclude=list({t.replace("[*]", "*") for t in train["smiles"]}))
    n_tr = len(train)
    X_train = np.nan_to_num(X_all[:n_tr])
    Xc = np.nan_to_num(build_features(cfg, corpus, data_dir, seed))
    X_comb = np.vstack([X_train, Xc])
    y_d = np.concatenate([np.ones(n_tr), np.zeros(len(corpus))])
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, C=pcfg.get("C", 1.0), random_state=seed)
    lr.fit(X_comb, y_d)
    proba = lr.predict_proba(X_train)[:, 1]
    w = np.clip(proba / np.clip(1.0 - proba, 1e-4, 1.0), pcfg.get("wmin", 0.1), pcfg.get("wmax", 10.0))
    w = w / max(w.mean(), 1e-9)
    print(f"  [shiftweight] corpus={len(corpus)} w[min,max]={w.min():.3f},{w.max():.3f}", flush=True)
    return run_cv(cfg, train, test, X_all, targets=targets, row_weights=w)


def run_matfac(cfg, train, test, X_all, targets):
    """Prediction-matrix factorization (ask: low-rank matrix completion feeding
    GBM). Stage-1 CV -> OOF/test; build (train+test) x T matrix (observed labels
    on train, stage-1 preds elsewhere); complete via iterative rank-k SVD; the
    completed other-target values become extra features (own-target column is
    zeroed on train to avoid leakage); retrain via grouped CV."""
    seed = cfg.get("seed", d.SEED)
    n_tr = len(train)
    n_te = len(test)
    y_full = train["target"].to_numpy(float)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    tlist = list(targets)
    T = len(tlist)
    oof1, te1, per1 = run_cv(cfg, train, test, X_all, targets=targets)
    M = np.full((n_tr + n_te, T), np.nan, dtype=np.float64)
    for j, t in enumerate(tlist):
        mask = tt == t
        M[:n_tr][mask, j] = y_full[mask]
        te_mask = test_tt == t
        if te_mask.any():
            M[n_tr:][te_mask, j] = te1[te_mask, j]
    col_mean = np.nanmean(M, axis=0)
    Mf = M.copy()
    for j in range(T):
        bad = ~np.isfinite(Mf[:, j])
        Mf[bad, j] = col_mean[j]
    rank = min(3, T)
    for _ in range(10):
        u, s, vt = np.linalg.svd(Mf - col_mean[None, :], full_matrices=False)
        rec = col_mean[None, :] + (u[:, :rank] * s[:rank][None, :]) @ vt[:rank]
        Mf = np.where(np.isnan(M), rec, Mf)
    F = Mf.copy()
    for j, t in enumerate(tlist):
        mask = tt == t
        F[:n_tr][mask, j] = 0.0  # remove own-target leakage for train rows
    X2 = np.hstack([np.nan_to_num(X_all), np.nan_to_num(F, nan=0.0).astype(np.float32)])
    oof, te, per_target = run_cv(cfg, train, test, X2, targets=targets)
    return oof, te, per_target


def run_uncertainty(cfg, train, test, X_all, targets):
    """Uncertainty-aware blending (ask: quantile ensembles -> spread feature).
    Quantile LGBM (0.1 / 0.9) OOF spread becomes an extra feature for the final
    model; per-target R2 reported on the spread-augmented fit."""
    seed = cfg.get("seed", d.SEED)
    n_tr = len(train)
    n_te = len(test)
    tt = train["target_type"].to_numpy(object)
    test_tt = test["target_type"].to_numpy(object)
    tlist = list(targets)
    cfg_q = dict(cfg)
    cfg_q["model"] = {"type": "lgbm", "objective": "quantile", "alpha": 0.1}
    oof_lo, te_lo, _ = run_cv(cfg_q, train, test, X_all, targets=targets)
    cfg_q["model"] = {"type": "lgbm", "objective": "quantile", "alpha": 0.9}
    oof_hi, te_hi, _ = run_cv(cfg_q, train, test, X_all, targets=targets)
    spread_tr = np.abs(oof_hi - oof_lo).reshape(-1, 1)
    spread_te = np.zeros((n_te, 1), dtype=np.float32)
    for j, t in enumerate(tlist):
        m = test_tt == t
        if m.any():
            spread_te[m, 0] = np.abs(te_hi[m, j] - te_lo[m, j])
    X_full = np.hstack([np.nan_to_num(X_all), np.vstack([spread_tr, spread_te]).astype(np.float32)])
    return run_cv(cfg, train, test, X_full, targets=targets)


def _apply_curation(cfg, train, test):
    """Curation steps from cfg['curation'] (ask EDA section):
      tg_median     : median-smooth tg replicates by canonical structure
      drop_overlap  : drop train rows whose canonical structure is in test
      drop_near_dup : drop train rows with Tanimoto > threshold to any test row
    Returns (train, kept_orig_idx)."""
    cur = cfg.get("curation", {}) or {}
    kept_orig_idx = np.arange(len(train))
    if cur.get("tg_median"):
        tg_mask = train["target_type"] == "tg"
        med = train.loc[tg_mask].groupby("canonical")["target"].transform("median")
        train.loc[tg_mask, "target"] = med
        print("  [curation] tg replicates median-smoothed by canonical structure", flush=True)
    drop_mask = np.zeros(len(train), dtype=bool)
    if cur.get("drop_overlap"):
        test_keys = set(test["canonical"])
        drop_mask |= train["canonical"].isin(test_keys).to_numpy()
    if cur.get("drop_near_dup"):
        thr = float(cur["drop_near_dup"])
        if thr > 0:
            sims = pn.tanimoto_to_set(list(train["smiles"]), list(test["smiles"]))
            drop_mask |= (sims >= thr)
    if drop_mask.any():
        keep = ~drop_mask
        kept_orig_idx = train.index[keep].to_numpy()
        train = train.loc[keep].reset_index(drop=True)
        print(f"  [curation] dropped {int(drop_mask.sum())} train rows (keep {len(train)})", flush=True)
    return train, kept_orig_idx


def _curation_weights(cfg, train, test):
    """Row weights from cfg['curation']['overlap_weight']: down-weight the 457
    train/test shared structures (ask EDA). Returns ones unless configured."""
    cur = cfg.get("curation", {}) or {}
    w = np.ones(len(train), dtype=np.float64)
    if cur.get("overlap_weight"):
        test_keys = set(test["canonical"])
        ov = train["canonical"].isin(test_keys).to_numpy()
        w[ov] = float(cur["overlap_weight"])
        print(f"  [curation] {int(ov.sum())} overlap rows weighted by {float(cur['overlap_weight'])}", flush=True)
    return w

