"""
05_explanation_agreement.py
===========================
R1.4 — cross-model explanation agreement. Rank features by Ridge |coef|,
ExtraTrees importance, LightGBM importance and SHAP mean |value|; report
pairwise Spearman rho per target (mean across targets in the heatmap).
Outputs: explanation_agreement_heatmap.png, explanation_agreement.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, save_plot, smoke_n)

METHODS = ["ridge", "et", "lgbm", "shap"]


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []
    agg = np.zeros((4, 4))

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        n_feat = X.shape[1]
        feat_names = pkl["pipe"]["feat_names"]

        imp = {}
        scaler, m_ridge = pkl["models"]["ridge"][-1]
        imp["ridge"] = np.abs(m_ridge.coef_)
        imp["et"] = pkl["models"]["et"][-1].feature_importances_
        imp["lgbm"] = pkl["models"]["lgbm"][-1].feature_importances_

        n_use = min(smoke_n(300, 80), len(X))
        rng = np.random.RandomState(SEED)
        use = rng.choice(len(X), n_use, replace=False)
        explainer = shap.TreeExplainer(pkl["models"]["lgbm"][-1])
        sv = explainer.shap_values(X[use])
        imp["shap"] = np.abs(sv).mean(axis=0)

        for a in range(4):
            for b in range(a + 1, 4):
                rho, _ = spearmanr(imp[METHODS[a]], imp[METHODS[b]])
                rows.append({"target": target,
                             "model_a": METHODS[a], "model_b": METHODS[b],
                             "spearman": float(rho)})
                agg[a, b] += rho
                agg[b, a] += rho

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "explanation_agreement.csv", index=False)
    agg /= len(TARGETS)
    np.fill_diagonal(agg, 1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(agg, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(METHODS)
    ax.set_yticks(range(4)); ax.set_yticklabels(METHODS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{agg[i, j]:.2f}", ha="center", va="center",
                    color="white" if agg[i, j] < 0.6 else "black")
    ax.set_title("Cross-Model Explanation Agreement (mean Spearman rho across 7 targets)")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    save_plot(fig, "explanation_agreement_heatmap.png")
    print(f"05_explanation_agreement.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
