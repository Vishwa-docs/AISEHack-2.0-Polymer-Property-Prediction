"""
04_fidelity.py
==============
R1.3 — explanation faithfulness. Mask top-SHAP features (set to train mean,
in-distribution) and compare the R2 drop against masking random features.
Outputs: fidelity_curve_{target}.png, fidelity_table.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, predict_ensemble,
                     save_plot, smoke_n, style_ax)
from sklearn.metrics import r2_score

FRACTIONS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        y = df_t["target"].values.astype(float)
        feat_names = pkl["pipe"]["feat_names"]

        n_use = min(smoke_n(300, 80), len(X))
        rng = np.random.RandomState(SEED)
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
                         "r2_random": r2_rand,
                         "drop_top_shap": base_r2 - r2_top,
                         "drop_random": base_r2 - r2_rand})
            curve.append((frac, r2_top, r2_rand))

        fig, ax = plt.subplots(figsize=(8, 5))
        fr = [c[0] for c in curve]
        ax.plot(fr, [c[1] for c in curve], "o-", color="steelblue",
                label="mask SHAP-top-k")
        ax.plot(fr, [c[2] for c in curve], "s--", color="tomato",
                label="mask random-k")
        ax.axhline(base_r2, color="gray", ls=":", label=f"baseline R2={base_r2:.3f}")
        style_ax(ax, f"Fidelity — {target.upper()} (masking features)",
                 "Fraction of features masked", "Validation R²")
        ax.legend()
        save_plot(fig, f"fidelity_curve_{target}.png")
        print(f"  {target}: base={base_r2:.4f} "
              f"drop@20% top={rows[-1]['drop_top_shap']:.4f} "
              f"random={rows[-1]['drop_random']:.4f}")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "fidelity_table.csv", index=False)
    print(f"04_fidelity.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
