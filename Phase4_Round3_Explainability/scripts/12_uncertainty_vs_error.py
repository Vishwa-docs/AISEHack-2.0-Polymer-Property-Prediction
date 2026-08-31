"""
12_uncertainty_vs_error.py
==========================
R3.3 — ensemble uncertainty (std of ridge/et/lgbm OOF) must track prediction
error. Pearson rho per target + scatter plots.
Outputs: error_vs_uncertainty_scatter_{target}.png, error_uncertainty_correlation.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from helpers import (TARGETS, OUTPUT_DIR, SMOKE, seed_all, oof_df, save_plot,
                     style_ax)


def main():
    seed_all(42)
    t0 = time.time()
    rows = []
    for target in TARGETS:
        oof = oof_df(target)
        unc = oof[["oof_ridge", "oof_et", "oof_lgbm"]].std(axis=1).values
        err = np.abs(oof["true_value"].values - oof["oof_ensemble"].values)
        rho, p = pearsonr(unc, err)
        rows.append({"target": target, "pearson_rho": float(rho),
                     "p_value": float(p), "n": len(oof)})
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(unc, err, s=12, alpha=0.4, color="steelblue")
        style_ax(ax, f"Error vs Uncertainty — {target.upper()}",
                 "Ensemble std (ridge/et/lgbm)", "Absolute prediction error")
        ax.text(0.05, 0.95, f"ρ = {rho:.3f}", transform=ax.transAxes,
                va="top", fontsize=12)
        save_plot(fig, f"error_vs_uncertainty_scatter_{target}.png")
        print(f"  {target}: rho = {rho:.3f}")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "error_uncertainty_correlation.csv", index=False)
    print(f"12_uncertainty_vs_error.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
