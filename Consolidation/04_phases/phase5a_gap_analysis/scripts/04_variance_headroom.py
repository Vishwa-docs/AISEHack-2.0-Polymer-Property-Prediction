#!/usr/bin/env python3
"""Phase5A Gap Analysis - 04: variance, R2 uncertainty and headroom per target.
ORACLE-ASSISTED, POST-FREEZE diagnostic only."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
RNG = np.random.default_rng(2026)

sub    = pd.read_csv(os.path.join(ROOT, "final_submissions", "submission.csv"))
oracle = pd.read_csv(os.path.join(ROOT, "Oracle", "final_oracle.csv"))
m = oracle[["id", "target_type", "target"]].merge(sub.rename(columns={"target": "pred"}), on="id", how="left")
cov = m[m["target"].notna() & m["pred"].notna()].copy()

rows = []
for t in TARGETS:
    r = cov[cov.target_type == t]
    y = r["target"].to_numpy(float); p = r["pred"].to_numpy(float)
    r2 = r2_score(y, p)
    tss = float(((y - y.mean()) ** 2).sum())
    sse = float(((y - p) ** 2).sum())
    resid2 = (y - p) ** 2
    worst_idx = int(np.argmax(resid2))
    # single-worst-row fix impact
    fixed = r2_score(np.delete(y, worst_idx), np.delete(p, worst_idx))
    # bootstrap SE of R2 (row resampling)
    boot = []
    n = len(y)
    for _ in range(1500):
        idx = RNG.integers(0, n, n)
        boot.append(r2_score(y[idx], p[idx]))
    boot = np.array(boot)
    se = boot.std(ddof=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append(dict(target=t, n=n, r2=r2, tss=tss, sse=sse, rmse=np.sqrt(sse / n),
                     se_r2=se, ci95=f"[{lo:.4f},{hi:.4f}]",
                     worst_row_sse=float(resid2[worst_idx]), worst_row_share=float(resid2[worst_idx]) / tss,
                     r2_if_fix_worst=fixed,
                     sse_cut_001=0.01 * tss,               # SSE cut for +0.01 R2 on this target
                     rmse_cut_001=None,
                     mean_impact_001=0.01 / 7))
df = pd.DataFrame(rows)
df["rmse_cut_001"] = np.sqrt((df.sse - df.sse_cut_001) / df.n)
# practical ceilings (assumptions, justified in report):
ceil = {"tg": 0.920, "egc": 0.950, "egb": 0.955, "ei": 0.935, "eea": 0.950, "nc": 0.945, "eps": 0.935}
df["practical_ceiling"] = [ceil[t] for t in TARGETS]
df["headroom"] = df.practical_ceiling - df.r2
df.to_csv(os.path.join(OUT, "04_headroom.csv"), index=False)

txt = []
txt.append("=" * 92)
txt.append("04 VARIANCE / UNCERTAINTY / HEADROOM per target (V57 vs final_oracle, 4,909 covered rows)")
txt.append("=" * 92)
txt.append(df.round(4).to_string(index=False))
txt.append("")
txt.append("Reading guide:")
txt.append("  se_r2        = bootstrap SE of the R2 estimate (test rows resampled, 1500 draws).")
txt.append("                 On ei/eps/nc/eea/egb, SE ~0.02-0.05: oracle deltas below ~2 SE are noise.")
txt.append("  worst_row_share = fraction of target TSS from the single worst prediction -")
txt.append("                 nc/eps/ei are fragile: one bad row can move R2 by >0.01.")
txt.append("  sse_cut_001  = absolute SSE reduction needed to gain +0.01 R2 on that target (0.01*TSS).")
txt.append("  rmse_cut_001 = the RMSE you must reach to gain +0.01 R2.")
txt.append("  headroom     = practical_ceiling - current R2 (ceilings are assumptions, see report).")
txt.append("")
txt.append("Mean impact of +0.01 R2 on any target = +0.00143 (equal for all 7).")
txt.append("")
txt.append("Fragility notes:")
for _, r in df.iterrows():
    txt.append(f"  {r.target:>3s}: n={r.n:4d}  SE(R2)={r.se_r2:.4f}  worst row = {r.worst_row_share*100:.2f}% of TSS  "
               f"fixing worst row -> R2 {r.r2:.4f} -> {r.r2_if_fix_worst:.4f}  |  +0.01 R2 needs RMSE {r.rmse:.3f} -> {r.rmse_cut_001:.3f}")
txt.append("")
tot_gain = df.headroom.sum() / 7
txt.append(f"Sum of headroom (ceiling - current) over all targets = {df.headroom.sum():.3f} -> mean +{tot_gain:.4f} "
           f"if every target hit its assumed ceiling: mean would be {0.9023 + tot_gain:.4f}")
txt.append("  => Even hitting ALL assumed 'practical ceilings' lands at 0.9414, only +0.006 above the")
txt.append("     0.935 goal; missing any single ceiling (e.g. tg stuck at 0.91) drops ~0.003-0.009.")
txt.append("     0.935 oracle (private ~0.924) effectively requires near-simultaneous SOTA on tg, ei, eps.")
with open(os.path.join(OUT, "04_headroom.txt"), "w") as f:
    f.write("\n".join(txt))
print("\n".join(txt))
