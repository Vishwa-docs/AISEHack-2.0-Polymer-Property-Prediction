"""
14_seed_stability.py
====================
R3.5 — OOF R2 across 5 different seeds for the tg proxy (random 5-fold).
Outputs: seed_stability.csv
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from helpers import (OUTPUT_DIR, SMOKE, seed_all, load_train, featurize,
                     canonical_smiles, smoke_n)

SEEDS = [42, 137, 2024, 2025, 2026]


def main():
    t0 = time.time()
    train = load_train()
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
            m_l = lgb.LGBMRegressor(n_estimators=smoke_n(400, 40), learning_rate=0.05,
                                    num_leaves=31, random_state=seed, verbosity=-1,
                                    n_jobs=-1).fit(X_tr, y_tr)
            p_r = m_r.predict(scaler.transform(X_va))
            p_e = m_e.predict(X_va)
            p_l = m_l.predict(X_va)
            S = np.column_stack([p_r, p_e, p_l])
            w, _ = nnls(S, y_va)
            w = w / w.sum() if w.sum() > 0 else np.array([1/3, 1/3, 1/3])
            oof[va] = S @ w
        rows.append({"seed": seed, "tg_oof_r2": r2_score(y, oof)})
        print(f"  seed {seed}: tg OOF R2 = {rows[-1]['tg_oof_r2']:.5f}")

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"seed": "mean", "tg_oof_r2": df["tg_oof_r2"].mean()}
    df.loc[len(df)] = {"seed": "std", "tg_oof_r2": df["tg_oof_r2"].std()}
    df.to_csv(OUTPUT_DIR / "seed_stability.csv", index=False)
    print(f"14_seed_stability.py DONE in {time.time() - t0:.0f}s — "
          f"std of mean R2 = {df['tg_oof_r2'].iloc[-1]:.5f}")


if __name__ == "__main__":
    main()
