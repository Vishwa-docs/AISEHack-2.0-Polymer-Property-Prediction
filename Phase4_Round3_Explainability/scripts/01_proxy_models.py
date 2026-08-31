"""
01_proxy_models.py
==================
Trains per-target proxy ensembles (Ridge + ExtraTrees + LightGBM) on the same
features used by V57 Stage A:
  - Morgan count fingerprint (radius 2, 1024 bits)
  - RDKit 2D descriptors (~200, median-imputed, zero-variance dropped)
  - character n-gram TF-IDF on canonical SMILES (2..6, max 8192)
GroupKFold on canonical SMILES (no duplicate leakage). NNLS blend.
Saves: proxy_oof_{t}.csv, proxy_scores.csv, proxy_feature_names.json,
       proxy_models_{t}.pkl (models + fitted pipeline for downstream reuse).

Reads ONLY Dataset/train.csv. All seeds = 42.
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all,
                     canonical_smiles, featurize, load_train, smoke_n)


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    all_scores = {}

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
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
        models = {"ridge": [], "et": [], "lgbm": []}

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            scaler = StandardScaler().fit(X_tr)
            m_ridge = Ridge(alpha=100, random_state=SEED)
            m_ridge.fit(scaler.transform(X_tr), y_tr)
            oof_ridge[va_idx] = m_ridge.predict(scaler.transform(X_va))

            m_et = ExtraTreesRegressor(n_estimators=smoke_n(200, 30), n_jobs=-1,
                                       random_state=SEED, min_samples_leaf=2)
            m_et.fit(X_tr, y_tr)
            oof_et[va_idx] = m_et.predict(X_va)

            m_lgbm = lgb.LGBMRegressor(n_estimators=smoke_n(400, 40),
                                       learning_rate=0.05, num_leaves=31,
                                       min_child_samples=5, random_state=SEED,
                                       n_jobs=-1, verbosity=-1)
            if not SMOKE:
                m_lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                           callbacks=[lgb.early_stopping(30, verbose=False)])
            else:
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
        print(f"{target}: ridge={scores['ridge']:.4f}  et={scores['et']:.4f}  "
              f"lgbm={scores['lgbm']:.4f}  ens={scores['ensemble']:.4f}  (n={n})")

        oof_df = pd.DataFrame({
            "smiles": df_t["smiles"].values,
            "canonical": df_t["canonical"].values,
            "true_value": y,
            "oof_ridge": oof_ridge,
            "oof_et": oof_et,
            "oof_lgbm": oof_lgbm,
            "oof_ensemble": oof_ens,
        })
        oof_df.to_csv(OUTPUT_DIR / f"proxy_oof_{target}.csv", index=False)

        with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", "wb") as f:
            pickle.dump({"models": models, "pipe": pipe, "nnls_weights": w}, f)

    pd.DataFrame(all_scores).T.to_csv(OUTPUT_DIR / "proxy_scores.csv")
    with open(OUTPUT_DIR / "proxy_feature_names.json", "w") as f:
        json.dump(pipe["feat_names"], f)
    print(f"01_proxy_models.py DONE in {time.time() - t0:.0f}s — outputs in outputs/")


if __name__ == "__main__":
    main()
