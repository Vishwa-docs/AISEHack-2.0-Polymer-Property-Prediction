"""
08_attribution_invariance.py
============================
R2.3 — attribution stability across equivalent SMILES. For N polymers x K
randomized variants, compute the LightGBM SHAP vector per variant and the
cosine similarity of each variant's attribution to the canonical form.
Outputs: attribution_invariance_per_target.csv, attribution_invariance_scatter.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, featurize, random_smiles, canonical_smiles,
                     save_plot, smoke_n, style_ax)


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    N_POLYMERS = smoke_n(100, 15)
    K = smoke_n(10, 4)

    per_target = {}
    scatter_rows = []
    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        n_poly = min(N_POLYMERS, len(df_t))
        sample = df_t.sample(n_poly, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
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
        per_target[target] = {"n_polymer_pairs": len(sims),
                              "mean_cosine_similarity": mean_sim}
        print(f"  {target}: mean attribution cosine = {mean_sim:.4f} (n={len(sims)})")

        # scatter: prediction invariance (from 07) vs attribution invariance
        f07 = OUTPUT_DIR / f"smiles_invariance_{target}.csv"
        if f07.exists():
            d7 = pd.read_csv(f07)
            scatter_rows.append({"target": target,
                                 "pred_std_mean": d7["std_pred"].mean(),
                                 "attr_cos_mean": mean_sim})

    pd.DataFrame(per_target).T.to_csv(OUTPUT_DIR / "attribution_invariance_per_target.csv")

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
    save_plot(fig, "attribution_invariance_scatter.png")
    print(f"08_attribution_invariance.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
