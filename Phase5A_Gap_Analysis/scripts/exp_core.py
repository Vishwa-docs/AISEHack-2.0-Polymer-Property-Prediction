#!/usr/bin/env python3
"""P5A shared experiment core (clean lane).
Reads ONLY official train.csv / test.csv from --data-dir. Fixed seeds. Grouped folds
(no molecule ever straddles folds). No reference answer panels, no precomputed artifacts.
Predictions are frozen by run.sh and scored afterwards."""
import os, json, time
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.optimize import nnls
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb

TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
SEED = 2026
N_FOLDS = 5

def load(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test

# ------------------------------------------------------------------ features
def morgan(smiles_list, radius=2, nbits=1024):
    out = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    for i, s in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = gen.GetFingerprint(m)
            arr = np.zeros((1, nbits), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr[0])
            out[i] = arr[0]
        except Exception:
            pass
    return out

def descriptors(smiles_list):
    feats = np.zeros((len(smiles_list), 15), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            rb = Descriptors.NumRotatableBonds(m)
            ar = Descriptors.NumAromaticRings(m)
            hvy = Descriptors.HeavyAtomCount(m)
            feats[i] = [Descriptors.MolWt(m), rb, ar, Descriptors.NumHDonors(m),
                        Descriptors.NumHAcceptors(m), hvy, Descriptors.RingCount(m),
                        Descriptors.FractionCSP3(m), Descriptors.TPSA(m),
                        Descriptors.Chi0v(m), Descriptors.Chi1v(m), Descriptors.MolLogP(m),
                        rb / max(hvy, 1.0), ar / max(hvy, 1.0), ar / (rb + 1.0)]
        except Exception:
            pass
    return feats

def char_ngram(smiles_all, max_features=4096, ngram=(1, 4)):
    cv = CountVectorizer(analyzer="char", ngram_range=ngram, max_features=max_features,
                         binary=True, lowercase=True)
    cv.fit(smiles_all)
    return cv.transform(smiles_all).astype(np.float32)

def build_features(train_smiles, test_smiles, config):
    t0 = time.time()
    mor_tr = morgan(train_smiles, nbits=config.get("morgan_nbits", 1024))
    mor_te = morgan(test_smiles, nbits=config.get("morgan_nbits", 1024))
    des_tr = descriptors(train_smiles)
    des_te = descriptors(test_smiles)
    cg_all = char_ngram(list(train_smiles) + list(test_smiles),
                        max_features=config.get("char_max_features", 4096))
    cg_tr = cg_all[: len(train_smiles)]
    cg_te = cg_all[len(train_smiles):]
    Xtr = sp.hstack([sp.csr_matrix(mor_tr), sp.csr_matrix(des_tr), cg_tr]).tocsr().astype(np.float32)
    Xte = sp.hstack([sp.csr_matrix(mor_te), sp.csr_matrix(des_te), cg_te]).tocsr().astype(np.float32)
    print("[core] features built in %.1fs  Xtr=%s Xte=%s" % (time.time() - t0, Xtr.shape, Xte.shape))
    return Xtr, Xte

def smoke_sample(train, config):
    if not config.get("smoke"):
        return train
    keep = []
    for t in TARGETS:
        sub = train[train.target_type == t]
        n = min(len(sub), 1500)
        keep.append(sub.sample(n=n, random_state=SEED))
    return pd.concat(keep, ignore_index=True)

# ------------------------------------------------------------------ models
def make_models(config, target):
    n = config.get("n_estimators", 300)
    lgb_kw = dict(n_estimators=n, random_state=SEED, n_jobs=-1, verbose=-1)
    if target == "tg" and config.get("tg_huber"):
        lgb_kw["objective"] = "huber"
    return {
        "lgb": lgb.LGBMRegressor(**lgb_kw),
        "xgb": xgb.XGBRegressor(n_estimators=n, random_state=SEED, n_jobs=-1,
                                objective="reg:squarederror", verbosity=0, tree_method="hist"),
        "et": ExtraTreesRegressor(n_estimators=max(150, n // 2), random_state=SEED, n_jobs=-1),
    }

def grouped_oof_predict(model, X, y, groups, w=None, n_splits=N_FOLDS):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=np.float64)
    models = []
    fold_r2 = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        if w is None:
            model.fit(X[tr_idx], y[tr_idx])
        else:
            model.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        oof[va_idx] = model.predict(X[va_idx])
        models.append(model)
        fold_r2.append(r2_score(y[va_idx], oof[va_idx]))
    return oof, models, np.array(fold_r2)

def predict_mean(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)

# ------------------------------------------------------------------ stage 1 (per target; Xall = vstack of train+test rows)
def stage1(train, Xall, config):
    res = {}
    for t in TARGETS:
        idx = np.where(train.target_type.values == t)[0]
        y = train.target.values[idx].astype(float)
        grp = train.smiles.values[idx]
        Xt = Xall[idx]
        if len(y) < 20:
            med = float(np.median(y))
            res[t] = {"idx": idx, "y": y, "oof": np.full(len(y), med),
                      "pred_all": np.full(Xall.shape[0], med),
                      "r2": 0.0, "fold_std": 0.0, "key": "median"}
            print("[core] stage1 %s: median fallback" % t)
            continue
        w = None
        if t == "tg" and config.get("tg_reweight"):
            m0 = lgb.LGBMRegressor(n_estimators=min(150, config.get("n_estimators", 300)),
                                   random_state=SEED, n_jobs=-1, verbose=-1)
            oof0, _, _ = grouped_oof_predict(m0, Xt, y, grp)
            r = np.abs(y - oof0) / max(np.std(y - oof0), 1e-9)
            w = np.clip(1.0 + r, 1.0, 4.0)
        oofs, preds, keys, fold_stds = {}, {}, {}, {}
        for key, model in make_models(config, t).items():
            oof, models, fold_r2 = grouped_oof_predict(model, Xt, y, grp, w=w)
            oofs[key] = oof
            preds[key] = predict_mean(models, Xall)
            keys[key] = models
            fold_stds[key] = float(np.std(fold_r2))
        if t == "tg" and config.get("tg_median"):
            oofs["median"] = np.median([oofs[k] for k in oofs], axis=0)
            preds["median"] = np.median([preds[k] for k in preds], axis=0)
            keys["median"] = None
            fold_stds["median"] = 0.0
        if t == "egc" and config.get("egc_zoo") and len(oofs) > 1:
            klist = list(oofs.keys())
            M = np.column_stack([oofs[k] for k in klist])
            wts, _ = nnls(M, y)
            oofs["nnls"] = M @ wts
            preds["nnls"] = np.column_stack([preds[k] for k in klist]) @ wts
            keys["nnls"] = None
            fold_stds["nnls"] = 0.0
        best_key = max(oofs, key=lambda k: r2_score(y, oofs[k]))
        res[t] = {"idx": idx, "y": y, "oof": oofs[best_key], "pred_all": preds[best_key],
                  "r2": r2_score(y, oofs[best_key]), "fold_std": fold_stds[best_key],
                  "key": best_key}
        print("[core] stage1 %s: OOF R2=%.4f (%s)" % (t, res[t]["r2"], best_key))
    return res

# ------------------------------------------------------------------ stage 2 (cross-property partner features)
def partner_block(t, train, test, pa, lab):
    n_all = len(train) + len(test)
    partners = [p for p in TARGETS if p != t]
    blk = np.column_stack([pa[p] for p in partners] +
                          [pa["egc"] + pa["eea"], pa["nc"] ** 2, pa["egc"]]).astype(np.float32)
    for m in range(n_all):
        sm = train.smiles.values[m] if m < len(train) else test.smiles.values[m - len(train)]
        d = lab.get(sm)
        if d:
            for c, p in enumerate(partners):
                if p in d:
                    blk[m, c] = d[p]
    return blk

def stage2(train, test, Xtr, Xte, res, pa, config):
    lab = {}
    for t in TARGETS:
        for i in res[t]["idx"]:
            lab.setdefault(train.smiles.values[i], {})[t] = train.target.values[i]
    out = {}
    for t in TARGETS:
        idx = res[t]["idx"]
        y = res[t]["y"]
        grp = train.smiles.values[idx]
        if len(y) < 20:
            out[t] = {"oof": res[t]["oof"], "test": np.full(len(test), float(np.median(y))),
                      "r2": 0.0, "fold_std": 0.0, "key": "median"}
            continue
        blk = partner_block(t, train, test, pa, lab)
        X2tr = sp.hstack([Xtr, sp.csr_matrix(blk[: len(train)])]).tocsr().astype(np.float32)
        X2te = sp.hstack([Xte, sp.csr_matrix(blk[len(train):])]).tocsr().astype(np.float32)
        Xt = X2tr[idx]
        model = lgb.LGBMRegressor(n_estimators=config.get("n_estimators", 300),
                                  random_state=SEED, n_jobs=-1, verbose=-1)
        y_use = y
        add_tr = np.zeros(len(y))
        add_te = np.zeros(len(test))
        if t == "eps" and config.get("eps_residual"):
            nc2_tr = pa["nc"][idx] ** 2
            nc2_te = pa["nc"][len(train):] ** 2
            y_use = y - nc2_tr
            add_tr = nc2_tr
            add_te = nc2_te
        oof, models, fold_r2 = grouped_oof_predict(model, Xt, y_use, grp)
        oof = oof + add_tr
        test_pred = predict_mean(models, X2te) + add_te
        out[t] = {"oof": oof, "test": test_pred, "r2": r2_score(y, oof),
                  "fold_std": float(np.std(fold_r2)), "key": "lgb2"}
        print("[core] stage2 %s: OOF R2=%.4f" % (t, out[t]["r2"]))
    return out

# ------------------------------------------------------------------ overlays (alpha fitted on OOF only)
def line_search_alpha(y, base, cand, max_a=0.6):
    best = (0.0, r2_score(y, base))
    for a in np.linspace(0.0, max_a, 25):
        r2 = r2_score(y, base + a * (cand - base))
        if r2 > best[1]:
            best = (float(a), float(r2))
    return best

def apply_overlays(train, res, pa, finals_oof, finals_test, config):
    alphas = {}
    for t in ["eps", "egb", "ei"]:
        idx = res[t]["idx"]
        y = res[t]["y"]
        base_oof = finals_oof[t]
        if t == "eps":
            eps_d = train[train.target_type == "eps"][["smiles", "target"]]
            nc_d = train[train.target_type == "nc"][["smiles", "target"]]
            j = eps_d.merge(nc_d, on="smiles", suffixes=("_e", "_n"))
            ionic_med = float(np.median(j.target_e - j.target_n ** 2)) if len(j) else 0.767
            cand_oof = pa["nc"][idx] ** 2 + ionic_med
            cand_test = finals_test["nc"] ** 2 + ionic_med
            max_a = 0.5
        elif t == "egb":
            a, b = np.polyfit(pa["egc"][idx], y, 1)
            cand_oof = a * pa["egc"][idx] + b
            cand_test = a * finals_test["egc"] + b
            max_a = config.get("egb_egc_max_alpha", 0.6)
        else:
            cand_oof = pa["eea"][idx] + pa["egc"][idx]
            cand_test = finals_test["eea"] + finals_test["egc"]
            max_a = config.get("ei_identity_max_alpha", 0.5)
        alpha, r2_after = line_search_alpha(y, base_oof, cand_oof, max_a=max_a)
        if alpha > 0:
            finals_oof[t] = base_oof + alpha * (cand_oof - base_oof)
            finals_test[t] = finals_test[t] + alpha * (cand_test - finals_test[t])
        alphas[t] = alpha
        print("[core] overlay %s: alpha=%.3f -> OOF R2=%.4f" % (t, alpha, r2_after))
    return finals_oof, finals_test, alphas

# ------------------------------------------------------------------ assembly
def run_pipeline(train, test, Xtr, Xte, config, post_hook=None):
    Xall = sp.vstack([Xtr, Xte]).tocsr().astype(np.float32)
    res = stage1(train, Xall, config)
    pa = {t: res[t]["pred_all"] for t in TARGETS}
    s2 = stage2(train, test, Xtr, Xte, res, pa, config)
    finals_oof, finals_test, chosen = {}, {}, {}
    for t in TARGETS:
        if s2[t]["r2"] > res[t]["r2"]:
            finals_oof[t], finals_test[t], chosen[t] = s2[t]["oof"], s2[t]["test"], "stage2"
        else:
            finals_oof[t], finals_test[t], chosen[t] = res[t]["oof"], pa[t][len(train):], "stage1"
    if post_hook is not None:
        finals_oof, finals_test = post_hook(train, test, res, pa, finals_oof, finals_test, config)
    finals_oof, finals_test, alphas = apply_overlays(train, res, pa, finals_oof, finals_test, config)
    if config.get("clip", True):
        for t in TARGETS:
            tr = train[train.target_type == t]["target"]
            lo, hi = float(tr.min()), float(tr.max())
            finals_test[t] = np.clip(finals_test[t], lo, hi)
    sub = pd.DataFrame({"id": test["id"].values, "target": np.zeros(len(test))})
    for t in TARGETS:
        mask = test.target_type.values == t
        sub.loc[mask, "target"] = finals_test[t][mask]
    sub = sub.sort_values("id").reset_index(drop=True)
    metrics = {
        "mean_oof_r2": float(np.mean([r2_score(res[t]["y"], finals_oof[t]) for t in TARGETS])),
        "per_target_oof_r2": {t: float(r2_score(res[t]["y"], finals_oof[t])) for t in TARGETS},
        "per_target_fold_std": {t: float(s2[t]["fold_std"] if chosen[t] == "stage2" else res[t]["fold_std"]) for t in TARGETS},
        "chosen_stage": chosen,
        "overlay_alphas": alphas,
    }
    out_dir = config.get("output_dir", ".")
    sub.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return sub, metrics
