"""
06_physics_decomp.py
====================
R1.5 — physics-decomposed explanation for eps = nc^2 + ionic.
1) train an nc proxy (Ridge) on the nc training rows
2) predict nc-hat for eps rows, define ionic = eps - nc-hat^2
3) SHAP on the ionic model (LightGBM) and on the nc model
4) side-by-side beeswarm -> physics_decomp_eps_shap.png
"""
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge

from helpers import (SEED, OUTPUT_DIR, SMOKE, seed_all, load_train, load_proxy,
                     featurize, smoke_n, save_plot, style_ax)


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()

    df_nc = train[train["target_type"] == "nc"].copy()
    if SMOKE and len(df_nc) > 120:
        df_nc = df_nc.sample(120, random_state=SEED).reset_index(drop=True)
    X_nc, pipe_nc = featurize(df_nc["smiles"].tolist())
    y_nc = df_nc["target"].values.astype(float)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_nc)
    m_nc = Ridge(alpha=100, random_state=SEED).fit(scaler.transform(X_nc), y_nc)

    df_eps = train[train["target_type"] == "eps"].copy()
    if SMOKE and len(df_eps) > 120:
        df_eps = df_eps.sample(120, random_state=SEED).reset_index(drop=True)
    X_eps, _ = featurize(df_eps["smiles"].tolist(), pipe=pipe_nc)
    nc_hat = m_nc.predict(scaler.transform(X_eps))
    y_eps = df_eps["target"].values.astype(float)
    ionic = y_eps - nc_hat ** 2

    m_ionic = lgb.LGBMRegressor(n_estimators=smoke_n(400, 40),
                                learning_rate=0.05, num_leaves=31,
                                random_state=SEED, verbosity=-1, n_jobs=-1)
    m_ionic.fit(X_eps, ionic)
    feat_names = pipe_nc["feat_names"]

    n_shap = min(smoke_n(200, 60), len(X_eps))
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_eps), n_shap, replace=False)

    ex_ionic = shap.TreeExplainer(m_ionic)
    sv_ionic = ex_ionic.shap_values(X_eps[idx])
    ex_nc = shap.LinearExplainer(m_nc, scaler.transform(X_nc))
    sv_nc = ex_nc.shap_values(scaler.transform(X_eps[idx]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, sv, title in ((axes[0], sv_ionic, f"ionic = eps − nc² (target: ionic channel)"),
                          (axes[1], sv_nc, "nc (refractive-index channel)")):
        mean_abs = np.abs(sv).mean(axis=0)
        top = np.argsort(mean_abs)[-12:][::-1]
        vals = sv[:, top]
        order = np.argsort(np.abs(vals).mean(axis=0))
        ax.axvline(0, color="gray", lw=0.8)
        for pos, fi in enumerate(order):
            col = np.where(vals[:, fi] > 0, "steelblue", "tomato")
            ax.scatter(vals[:, fi], np.full(len(vals), pos) + 0.3 * np.random.RandomState(SEED).rand(len(vals)),
                       c=col, s=6, alpha=0.5, marker=".")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feat_names[top[i]] for i in order], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Physics-Decomposed SHAP — eps = nc² + ionic", fontsize=13)
    save_plot(fig, "physics_decomp_eps_shap.png")

    # also save the decomposition numbers
    pd.DataFrame({"eps_true": y_eps, "nc_hat": nc_hat,
                  "ionic": ionic}).to_csv(
        OUTPUT_DIR / "physics_decomp_values.csv", index=False)
    print(f"06_physics_decomp.py DONE in {time.time() - t0:.0f}s — "
          f"ionic std={ionic.std():.3f}, nc-hat mean={nc_hat.mean():.3f}")


if __name__ == "__main__":
    main()
