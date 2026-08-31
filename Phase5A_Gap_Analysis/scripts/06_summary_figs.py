#!/usr/bin/env python3
"""Phase5A Gap Analysis - 06: summary figures for the report."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", OUT)
plt.style.use("seaborn-v0_8-whitegrid")

df  = pd.read_csv(os.path.join(OUT, "04_headroom.csv"))
var = pd.read_csv(os.path.join(OUT, "03_variants.csv"))
tg  = pd.read_csv(os.path.join(OUT, "03_tg_categories.csv"))

# fig 02: current R2 vs practical ceiling with SE bars
fig, ax = plt.subplots(figsize=(9.5, 4.6))
x = np.arange(7)
ax.errorbar(x, df.r2, yerr=1.96 * df.se_r2, fmt="o", ms=7, capsize=4, color="#4C72B0",
            label="V57 R2 (final_oracle panel) +/- 95% CI")
ax.scatter(x, df.practical_ceiling, marker="^", s=90, color="#C44E52", label="practical ceiling (assumed)")
for i, (r, c) in enumerate(zip(df.r2, df.practical_ceiling)):
    ax.plot([i, i], [r, c], color="gray", lw=1, ls=":")
ax.set_xticks(x); ax.set_xticklabels(df.target)
ax.set_ylim(0.80, 0.97)
ax.set_title("Per-target R2: where we are, uncertainty, and assumed headroom")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_02_headroom.png"), dpi=120); plt.close()

# fig 03: metric variants vs LB anchors
fig, ax = plt.subplots(figsize=(9.5, 4.4))
labels = ["A: mean of per-target R2\n(OFFICIAL metric)", "B: pooled r2_score\n(all rows as one array)",
          "C: TSS-weighted mean\n(within-target pooling)", "A on verified panel\n(3,818 rows)"]
vals = var.score.to_numpy()
bars = ax.bar(range(4), vals, color=["#55A868", "#C44E52", "#DD8452", "#4C72B0"], width=0.6)
ax.axhline(0.891, color="#2C2C2C", lw=2, ls="--", label="actual private LB = 0.891")
ax.axhline(0.917, color="#2C2C2C", lw=1, ls=":", label="actual public LB = 0.917")
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.85, 0.97)
ax.set_title("Metric variants for the same frozen submission - only variant A calibrates to the LB")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_03_metric_variants.png"), dpi=120); plt.close()

# fig 04: Tg difficulty by oracle category
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(tg.category, tg.r2, color=["#4C72B0", "#DD8452", "#C44E52"])
for i, (r2, n) in enumerate(zip(tg.r2, tg.n)):
    ax.text(i, r2 + 0.004, f"R2={r2:.4f}\nn={n}", ha="center", fontsize=9)
ax.axhline(0.8842, color="gray", ls="--", lw=1)
ax.text(2.02, 0.886, "true Tg R2 est. 0.884", fontsize=8, color="gray")
ax.set_ylim(0.75, 0.95)
ax.set_title("Tg R2 by oracle row difficulty (V57) - oracle covers easy rows best")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_04_tg_categories.png"), dpi=120); plt.close()
print("figures written:", os.listdir(OUT))
