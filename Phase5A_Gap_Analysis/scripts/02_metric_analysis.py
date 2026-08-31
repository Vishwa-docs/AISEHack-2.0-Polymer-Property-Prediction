#!/usr/bin/env python3
"""Phase5A Gap Analysis - 02: Metric math. Unweighted mean of per-target R2 vs pooled vs TSS-weighted;
leverage tables; scenarios to 0.935; 'make target perfect' ceilings. ORACLE-ASSISTED analysis only."""
import os, json
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

train = pd.read_csv(os.path.join(ROOT, "Dataset", "train.csv"))
train["target"] = pd.to_numeric(train["target"], errors="coerce")
test = pd.read_csv(os.path.join(ROOT, "Dataset", "test.csv"))

st = {}
for t in TARGETS:
    sub = train[train.target_type == t]["target"].dropna()
    st[t] = dict(n=len(sub), mean=sub.mean(), sd=sub.std(ddof=1), tss=(len(sub)-1)*sub.var(ddof=1),
                 tcount=int((test.target_type == t).sum()))

# current V57 per-target R2 (final_oracle panel, from final_submissions/README.md)
cur = {"tg": 0.895346, "egc": 0.911096, "egb": 0.926818, "ei": 0.871121, "eea": 0.918330, "nc": 0.908647, "eps": 0.884667}
mean_now = float(np.mean(list(cur.values())))

L = []
for t in TARGETS:
    TSS_train = st[t]["tss"]
    n_test = st[t]["tcount"]
    # test-proxy TSS = (n_test-1)*sd_train^2 ; and with mean-centering per target
    TSS_test_proxy = (n_test - 1) * st[t]["sd"] ** 2
    w_tsw = TSS_test_proxy  # relative weight under TSS-weighted metric
    L.append(dict(target=t, cur_r2=cur[t],
                  train_n=st[t]["n"], test_n=n_test,
                  tss_train=TSS_train, tss_test_proxy=TSS_test_proxy,
                  dmean_per_001=0.01/7,                                  # unweighted: +0.01 R2 -> +0.00143 mean
                  sse_cut_for_001_mean=0.07*TSS_test_proxy,              # need to cut SSE by this for +0.01 mean R2
                  rmse_now=np.sqrt((1-cur[t])*TSS_test_proxy/n_test),    # current RMSE proxy
                  rmse_for_001 = None,
                  perfect_ceiling_mean = None))
Ldf = pd.DataFrame(L)
Ldf["rmse_for_001"] = np.sqrt((Ldf.cur_r2 - 0.01) * Ldf.tss_test_proxy / Ldf.test_n)
for _, r in Ldf.iterrows():
    others = sum(cur[k] for k in TARGETS if k != r.target)
    Ldf.loc[Ldf.target == r.target, "perfect_ceiling_mean"] = (others + 1.0) / 7
Ldf.to_csv(os.path.join(OUT, "02_leverage.csv"), index=False)

txt = []
txt.append("=" * 88)
txt.append("02 METRIC MATH - official metric = UNWEIGHTED MEAN of the seven per-target R2 values")
txt.append("   Score = (R2_tg + R2_egc + R2_egb + R2_ei + R2_eea + R2_nc + R2_eps) / 7")
txt.append("=" * 88)
txt.append("")
txt.append("KEY RESULT 1 - every target is worth EXACTLY 1/7 of the score, regardless of rows.")
txt.append(f"  +0.01 R2 on ANY single target = +{0.01/7:.5f} on the mean.  (+0.05 -> +{0.05/7:.5f})")
txt.append("")
txt.append("KEY RESULT 2 - the marginal value of an absolute SSE reduction is (1/7)/TSS_j:")
txt.append("  cutting SSE by 1 unit on nc is worth ~4 MILLION times more than on tg,")
txt.append("  but the SSE you can realistically cut scales with TSS_j. R2 already normalises this.")
txt.append("")
txt.append("KEY RESULT 3 - 'make one target perfect' ceilings (current mean 0.9024):")
for _, r in Ldf.iterrows():
    txt.append(f"  perfect {r.target:>3s} -> mean {r.perfect_ceiling_mean:.4f}   (gain {r.perfect_ceiling_mean-mean_now:+.4f})")
txt.append("")
txt.append("  => Even a PERFECT Tg model gives mean 0.9175 only; perfect ei gives 0.9210.")
txt.append("     'Throw everything at Tg' alone CANNOT reach 0.935. (Refutes analysis #1's strategy.)")
txt.append("")
txt.append("KEY RESULT 4 - TSS-weighted (pooled-within-target) metric would give tg ~98-99% weight:")
tot = Ldf.tss_test_proxy.sum()
for _, r in Ldf.iterrows():
    txt.append(f"  {r.target:>3s}: test-proxy TSS {r.tss_test_proxy:>12,.0f}  = {r.tss_test_proxy/tot*100:5.2f}% of total")
txt.append(f"  Under such a metric, LB ~ R2_tg = {cur['tg']:.4f} and the 0.891 private LB coincidence would be explained.")
txt.append("  The official page says 'mean ... across the seven targets' = unweighted. Empirical check in script 03.")
txt.append("")
txt.append("KEY RESULT 5 - what it takes to reach 0.935 (need +0.0326 mean = +0.228 R2 spread over 7 targets):")
txt.append("  Per-target R2 needed if every target improves by the same amount: +0.0326 each.")
txt.append("  That means ~0.928 avg (tg 0.928, ei 0.904, eps 0.917, ...) - i.e. state-of-the-art on EVERY target.")

# ---------------- scenarios ----------------
scen = []
def add(name, profile):
    m = float(np.mean([profile[t] for t in TARGETS]))
    scen.append(dict(scenario=name, mean=m, gain=m - mean_now, **profile))
add("V57 current baseline", cur)
add("S1 all targets +0.033 (uniform)", {t: cur[t] + 0.033 for t in TARGETS})
add("S2 realistic 7-target push (tg .920, egc .922, egb .940, ei .905, eea .928, nc .920, eps .910)",
    {"tg": .920, "egc": .922, "egb": .940, "ei": .905, "eea": .928, "nc": .920, "eps": .910})
add("S3 'Tg-only hack' - tg to 0.95, nothing else", {"tg": .950, **{t: cur[t] for t in TARGETS if t != "tg"}})
add("S4 perfect Tg only (R2=1.0)", {"tg": 1.0, **{t: cur[t] for t in TARGETS if t != "tg"}})
add("S5 weak-target push only (ei .905, eps .915, others unchanged)",
    {"tg": cur["tg"], "egc": cur["egc"], "egb": cur["egb"], "ei": .905, "eea": cur["eea"], "nc": cur["nc"], "eps": .915})
add("S6 big+weak (tg .915, ei .905, eps .910, nc .918, eea .925, egc .918, egb .935)",
    {"tg": .915, "egc": .918, "egb": .935, "ei": .905, "eea": .925, "nc": .918, "eps": .910})
add("S7 aggressive (tg .925, egc .925, egb .945, ei .915, eea .935, nc .925, eps .920)",
    {"tg": .925, "egc": .925, "egb": .945, "ei": .915, "eea": .935, "nc": .925, "eps": .920})
add("S8 ceiling-flirt (tg .93, egc .93, egb .95, ei .92, eea .94, nc .93, eps .925)",
    {"tg": .930, "egc": .930, "egb": .950, "ei": .920, "eea": .940, "nc": .930, "eps": .925})
sdf = pd.DataFrame(scen)
sdf.to_csv(os.path.join(OUT, "02_scenarios.csv"), index=False)
txt.append("")
txt.append("Scenario table (mean of 7 per-target R2, oracle panel):")
txt.append(sdf.round(4).to_string(index=False))
txt.append("")
txt.append("Takeaways: S3/S4 (Tg-only) cap ~0.910-0.917. 0.935 needs near-uniform ~0.03 gains,")
txt.append("with the biggest absolute lifts coming from tg (+~0.025), ei (+~0.035), eps (+~0.02).")

with open(os.path.join(OUT, "02_metric_math.txt"), "w") as f:
    f.write("\n".join(txt))
print("\n".join(txt))
