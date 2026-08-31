"""
16_khazana_verification.py
==========================
R4.2 — post-freeze external verification of the submitted predictions against
ground-truth answer values (Khazana/online polymer panels).  THIS IS THE ONLY
CORE SCRIPT ALLOWED TO READ THE GROUND-TRUTH FILE.  It is evaluation-only:
values never enter any model, feature, or training step.

Ground truth file: <folder>/Oracle/final_oracle.csv  (id, target, target_type,
oracle_status, ...).  Panels: verified / external_verified / proxy.
DFT targets use the 'verified' panel; tg uses verified + external_verified.

Outputs: khazana_holdout_scores.csv, khazana_scatter_{target}.png (6 DFT)
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from helpers import (TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_test,
                     load_submission, project_root, save_plot, style_ax)

DFT_TARGETS = ["egc", "egb", "ei", "eea", "nc", "eps"]
GT_FILE = project_root() / "Oracle" / "final_oracle.csv"


def main():
    seed_all(42)
    t0 = time.time()
    if not GT_FILE.exists():
        print(f"16_khazana_verification.py SKIPPED — ground-truth file missing: {GT_FILE}")
        (OUTPUT_DIR / "khazana_holdout_scores.csv").write_text(
            "target,n,r2,mae,note\ntg,0,,,ground_truth_file_missing\n")
        return

    gt = pd.read_csv(GT_FILE)
    sub = load_submission()
    test = load_test()

    merged = (sub.rename(columns={"target": "pred"})
                 .merge(test[["id", "target_type"]], on="id")
                 .merge(gt[["id", "target", "oracle_status"]], on="id", how="inner"))
    score_rows = []
    for target in TARGETS:
        allowed = (["verified", "external_verified"] if target == "tg"
                   else ["verified"])
        df_t = merged[(merged["target_type"] == target) &
                      (merged["oracle_status"].isin(allowed))].copy()
        df_t = df_t[df_t["target"].notna()]
        if len(df_t) < 10:
            score_rows.append({"target": target, "n": len(df_t),
                               "r2": np.nan, "mae": np.nan})
            continue
        y_true = df_t["target"].values
        y_pred = df_t["pred"].values
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        score_rows.append({"target": target, "n": len(df_t), "r2": r2, "mae": mae})
        print(f"  {target}: R2={r2:.4f}  MAE={mae:.4f}  n={len(df_t)}")

        if target in DFT_TARGETS:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="steelblue")
            lims = [min(y_true.min(), y_pred.min()) * 0.95,
                    max(y_true.max(), y_pred.max()) * 1.05]
            ax.plot(lims, lims, "r--", alpha=0.8, label=f"R²={r2:.4f}")
            style_ax(ax, f"External Verification: {target.upper()}\n(n={len(df_t)}, post-freeze)",
                     "Ground-truth value", "Model prediction")
            ax.legend()
            save_plot(fig, f"khazana_scatter_{target}.png")

    pd.DataFrame(score_rows).to_csv(OUTPUT_DIR / "khazana_holdout_scores.csv", index=False)
    print(f"16_khazana_verification.py DONE in {time.time() - t0:.0f}s — "
          f"ground truth used for post-freeze evaluation only")


if __name__ == "__main__":
    main()
