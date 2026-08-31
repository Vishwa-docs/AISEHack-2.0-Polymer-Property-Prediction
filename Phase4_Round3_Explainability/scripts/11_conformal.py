"""
11_conformal.py
===============
R3.2 — split-conformal prediction intervals (pure numpy). Calibrates on the
OOF residuals; reports empirical coverage at 80/90/95%; writes intervals for
all 4,940 test predictions from the frozen submission.
Outputs: conformal_coverage_table.csv, conformal_calibration_plot.png,
         test_predictions_with_intervals.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_test,
                     load_submission, oof_df, save_plot, style_ax)

ALPHA_LEVELS = [0.80, 0.90, 0.95]


def main():
    seed_all(42)
    t0 = time.time()
    test_df = load_test()
    submission = load_submission()
    coverage_rows = []

    for target in TARGETS:
        oof = oof_df(target)
        residuals = np.abs(oof["true_value"].values - oof["oof_ensemble"].values)
        n = len(residuals)
        split_idx = int(0.8 * n)
        cal_res = residuals[:split_idx]
        val_res = residuals[split_idx:]
        oof_val = oof["oof_ensemble"].values[split_idx:]
        true_val = oof["true_value"].values[split_idx:]

        for alpha in ALPHA_LEVELS:
            q_level = min(np.ceil((len(cal_res) + 1) * alpha) / len(cal_res), 1.0)
            q_hat_cal = np.quantile(cal_res, q_level)
            in_int = np.abs(true_val - oof_val) <= q_hat_cal
            coverage_rows.append({
                "target": target,
                "nominal_coverage": alpha,
                "empirical_coverage": float(in_int.mean()),
                "interval_halfwidth": float(q_hat_cal),
                "n_calibration": len(cal_res),
                "n_validation": len(val_res),
            })

    df_cov = pd.DataFrame(coverage_rows)
    df_cov.to_csv(OUTPUT_DIR / "conformal_coverage_table.csv", index=False)

    # reliability diagram: one subplot per target
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(20, 4), sharey=True)
    for ax, target in zip(axes, TARGETS):
        sub = df_cov[df_cov["target"] == target]
        ax.plot([0.80, 0.90, 0.95], [0.80, 0.90, 0.95], "k--", alpha=0.5,
                label="Perfect calibration")
        ax.plot(sub["nominal_coverage"], sub["empirical_coverage"], "o-",
                color="steelblue", markersize=8, label="Empirical")
        ax.set_title(target.upper())
        ax.set_xlabel("Nominal coverage")
        ax.set_xlim(0.78, 0.97); ax.set_ylim(0.70, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].legend()
    fig.suptitle("Conformal Prediction Calibration — Nominal vs Empirical Coverage",
                 fontsize=14)
    save_plot(fig, "conformal_calibration_plot.png")

    # test intervals
    tt_map = test_df.set_index("id")["target_type"].to_dict()
    sub = submission.copy()
    sub["target_type"] = sub["id"].map(tt_map)
    q_by_target = df_cov[df_cov["nominal_coverage"] == 0.90].set_index("target")["interval_halfwidth"]
    q80 = df_cov[df_cov["nominal_coverage"] == 0.80].set_index("target")["interval_halfwidth"]
    for name, q in (("80", q80), ("90", q_by_target)):
        hw = sub["target_type"].map(q)
        sub[f"lower_{name}"] = sub["target"] - hw
        sub[f"upper_{name}"] = sub["target"] + hw
    cols = ["id", "target_type", "target", "lower_80", "upper_80", "lower_90", "upper_90"]
    sub[cols].to_csv(OUTPUT_DIR / "test_predictions_with_intervals.csv", index=False)
    print(f"11_conformal.py DONE in {time.time() - t0:.0f}s — "
          f"coverage errors: " +
          ", ".join(f"{r['target']}:{abs(r['empirical_coverage']-0.9)*100:.1f}%"
                    for r in coverage_rows if r["nominal_coverage"] == 0.90))


if __name__ == "__main__":
    main()
