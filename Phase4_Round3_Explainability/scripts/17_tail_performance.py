"""
17_tail_performance.py
======================
R4.3 — distribution-tail generalization: R2/MAE on bottom-10%, middle-80%,
top-10% of true values per target (on the OOF predictions).
Outputs: tail_performance.csv, tail_performance_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from helpers import (TARGETS, OUTPUT_DIR, SMOKE, seed_all, oof_df, save_plot,
                     style_ax)


def main():
    seed_all(42)
    t0 = time.time()
    rows = []
    for target in TARGETS:
        oof = oof_df(target)
        if SMOKE and len(oof) > 150:
            oof = oof.sample(150, random_state=42).reset_index(drop=True)
        y = oof["true_value"].values
        p = oof["oof_ensemble"].values
        q10, q90 = np.quantile(y, [0.10, 0.90])
        for name, mask in (("bottom_10", y <= q10),
                           ("middle_80", (y > q10) & (y < q90)),
                           ("top_10", y >= q90)):
            if mask.sum() < 5:
                rows.append({"target": target, "bucket": name, "n": int(mask.sum()),
                             "r2": np.nan, "mae": np.nan})
                continue
            rows.append({"target": target, "bucket": name, "n": int(mask.sum()),
                         "r2": r2_score(y[mask], p[mask]),
                         "mae": mean_absolute_error(y[mask], p[mask])})
        print(f"  {target}: " + " ".join(
            f"{b}={next((r['r2'] for r in rows if r['target'] == target and r['bucket'] == b), float('nan')):.3f}"
            for b in ("bottom_10", "middle_80", "top_10")))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tail_performance.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    buckets = ["bottom_10", "middle_80", "top_10"]
    width = 0.11
    for i, target in enumerate(TARGETS):
        vals = [next((r["r2"] for r in rows if r["target"] == target and r["bucket"] == b), np.nan)
                for b in buckets]
        ax.bar(np.arange(3) + i * width, vals, width, label=target)
    ax.set_xticks(np.arange(3) + 3 * width)
    ax.set_xticklabels(["Bottom 10%", "Middle 80%", "Top 10%"])
    style_ax(ax, "Tail Performance — R² across property distribution",
             "True-value bucket", "R²")
    ax.legend(ncol=4, fontsize=8)
    save_plot(fig, "tail_performance_plot.png")
    print(f"17_tail_performance.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
