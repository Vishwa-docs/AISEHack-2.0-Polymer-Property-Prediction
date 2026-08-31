#!/usr/bin/env python3
"""Phase5A - 09: error concentration, prediction clipping, affine invariance, thought experiments."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, "output")
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

sub    = pd.read_csv(os.path.join(ROOT, "final_submissions", "submission.csv"))
oracle = pd.read_csv(os.path.join(ROOT, "Oracle", "final_oracle.csv"))
train  = pd.read_csv(os.path.join(ROOT, "Dataset", "train.csv"))
train["target"] = pd.to_numeric(train["target"], errors="coerce")
m = oracle[["id", "target_type", "target"]].merge(sub.rename(columns={"target": "pred"}), on="id", how="left")
cov = m[m["target"].notna() & m["pred"].notna()].copy()

lines = []
lines.append("=== 09: ERROR CONCENTRATION, CLIPPING, AFFINE INVARIANCE ===")
lines.append("")
rows = []
for t in TARGETS:
    r = cov[cov.target_type == t]
    y = r["target"].to_numpy(float); p = r["pred"].to_numpy(float)
    r2 = r2_score(y, p)
    sse_all = float(((y - p) ** 2).sum())
    res2 = np.sort((y - p) ** 2)[::-1]
    n = len(y)
    def share(frac):
        k = max(1, int(round(n * frac)))
        return float(res2[:k].sum()) / sse_all
    tr = train[train.target_type == t]["target"]
    lo, hi = float(tr.min()), float(tr.max())
    p_clip = np.clip(p, lo, hi)
    qlo, qhi = float(tr.quantile(0.005)), float(tr.quantile(0.995))
    p_qclip = np.clip(p, qlo, qhi)
    r2_clip = r2_score(y, p_clip); r2_qclip = r2_score(y, p_qclip)
    k1 = max(1, int(round(n * 0.01)))
    idx = np.argsort((y - p) ** 2)[::-1][:k1]
    p_half = p.copy(); p_half[idx] = 0.5 * (p[idx] + y[idx])
    r2_half = r2_score(y, p_half)
    rows.append(dict(target=t, n=n, r2=r2, top1pct_share=share(0.01), top5pct_share=share(0.05),
                     top10pct_share=share(0.10), clip_bounds_gain=r2_clip - r2,
                     clip_q995_gain=r2_qclip - r2, halve_top1pct_gain=r2_half - r2))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "09_error_concentration.csv"), index=False)
lines.append(df.round(4).to_string(index=False))
lines.append("")
lines.append("clip_bounds_gain  = R2 gain from clipping predictions to [train min, train max] (clean, principled)")
lines.append("clip_q995_gain    = R2 gain from clipping to train 0.5%/99.5% quantiles")
lines.append("halve_top1pct_gain= hypothetical R2 gain if the worst 1% rows errors were halved (upper bound of fixing)")
lines.append("")
tg = cov[cov.target_type == "tg"]
y = tg["target"].to_numpy(float); p = tg["pred"].to_numpy(float)
r2a = r2_score(y, p); r2b = r2_score(y * 0.01, p * 0.01); r2c = r2_score(y * 0.01 + 100, p * 0.01 + 100)
lines.append("AFFINE INVARIANCE: R2(y,p)=%.6f | R2(0.01*y,0.01*p)=%.6f | R2(0.01*y+100,0.01*p+100)=%.6f" % (r2a, r2b, r2c))
lines.append("=> normalizing/scaling Tg (or any single target) changes NOTHING for per-target R2.")
lines.append("   Normalization only matters when targets share ONE loss (joint/multi-task training).")
txt = chr(10).join(lines)
with open(os.path.join(OUT, "09_error_concentration.txt"), "w") as f:
    f.write(txt)
print(txt)
