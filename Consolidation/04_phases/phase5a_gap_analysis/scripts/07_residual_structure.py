#!/usr/bin/env python3
"""Phase5A Gap Analysis - 07: residual structure per target (bias, slope, outlier concentration).
ORACLE-ASSISTED, POST-FREEZE diagnostic only."""
import os
import numpy as np, pandas as pd
from scipy import stats as sps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, "output")
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

sub    = pd.read_csv(os.path.join(ROOT, "final_submissions", "submission.csv"))
oracle = pd.read_csv(os.path.join(ROOT, "Oracle", "final_oracle.csv"))
m = oracle[["id", "target_type", "target"]].merge(sub.rename(columns={"target": "pred"}), on="id", how="left")
cov = m[m["target"].notna() & m["pred"].notna()].copy()

rows = []
for t in TARGETS:
    r = cov[cov.target_type == t]
    y = r["target"].to_numpy(float); p = r["pred"].to_numpy(float)
    res = y - p
    slope = float(np.cov(y, p)[0, 1] / np.var(p))       # regress y on pred
    bias = float(res.mean())
    het = sps.spearmanr(np.abs(res), y).statistic        # |res| vs y correlation
    sse = float((res ** 2).sum()); tss = float(((y - y.mean()) ** 2).sum())
    top5 = np.sort(res ** 2)[::-1][: max(1, len(res) // 20)]
    rows.append(dict(target=t, n=len(r), mean_res=bias, resid_sd=float(res.std(ddof=1)),
                     calib_slope=slope, het_spearman=het,
                     sse_top5pct=float(top5.sum()) / sse,
                     frac_over_3sig=float((np.abs(res) > 3 * res.std(ddof=1)).mean())))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "07_residual_structure.csv"), index=False)
print("Per-target residual structure (V57 vs final_oracle):")
print(df.round(4).to_string(index=False))
print()
print("  mean_res      = mean(y - pred); a CONSTANT bias does NOT hurt per-target R2 (R2 is shift-invariant)")
print("  calib_slope   = cov(y,p)/var(p); slope<1 means shrinkage toward the mean (under-shooting extremes)")
print("  het_spearman  = corr(|residual|, y); >0 => larger absolute errors on high-value rows")
print("  sse_top5pct   = share of target SSE coming from the worst 5% of rows (outlier concentration)")
print("  frac_over_3sig= fraction of rows beyond 3 sigma of the residual distribution")
