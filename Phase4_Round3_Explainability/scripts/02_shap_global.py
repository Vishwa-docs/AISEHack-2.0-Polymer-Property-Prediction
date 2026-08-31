"""
02_shap_global.py
=================
R1.1 — global SHAP feature importance per target.
TreeExplainer on the last-fold LightGBM proxy; beeswarm plots (top 20),
global summary bar chart, and shap_top20_per_target.csv.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, save_plot, smoke_n, style_ax)


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    top20_all = {}

    for target in TARGETS:
        df_t = train[train["target_type"] == target]
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED)
        pkl = load_proxy(target)
        feat_names = pkl["pipe"]["feat_names"]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        lgbm_model = pkl["models"]["lgbm"][-1]

        explainer = shap.TreeExplainer(lgbm_model)
        n_shap = min(smoke_n(500, 100), len(X))
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), n_shap, replace=False)
        sv = explainer.shap_values(X[idx])

        mean_abs = np.abs(sv).mean(axis=0)
        top20_idx = np.argsort(mean_abs)[-20:][::-1]
        top20_all[target] = {feat_names[i]: float(mean_abs[i]) for i in top20_idx}

        shap.summary_plot(sv, X[idx], feature_names=feat_names,
                          max_display=20, show=False)
        plt.title(f"SHAP Beeswarm — {target.upper()}", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_beeswarm_{target}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {target}: top feature = {feat_names[top20_idx[0]]} "
              f"({mean_abs[top20_idx[0]]:.4f})")

    # global summary bar chart
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
    save_plot(fig, "shap_summary_global.png")

    rows = [{"target": t, "feature": f, "mean_abs_shap": v}
            for t, d in top20_all.items() for f, v in d.items()]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "shap_top20_per_target.csv", index=False)
    print(f"02_shap_global.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
