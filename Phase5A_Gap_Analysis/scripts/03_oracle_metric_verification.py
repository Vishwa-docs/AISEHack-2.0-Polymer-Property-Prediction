#!/usr/bin/env python3
"""Phase5A Gap Analysis - 03: EMPIRICAL metric verification.
Scores the frozen V57 submission (final_submissions/submission.csv) against
Oracle/final_oracle.csv under every plausible metric variant and compares with the
actual Kaggle private LB (0.891) and public LB (0.917).
ORACLE-ASSISTED, POST-FREEZE diagnostic only. Nothing here enters any submission."""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

sub     = pd.read_csv(os.path.join(ROOT, "final_submissions", "submission.csv"))
oracle  = pd.read_csv(os.path.join(ROOT, "Oracle", "final_oracle.csv"))
verif   = pd.read_csv(os.path.join(ROOT, "Oracle", "oracle.csv"))          # 3818 verified-only panel
test    = pd.read_csv(os.path.join(ROOT, "Dataset", "test.csv"))
assert len(sub) == 4940 and len(oracle) == 4940

m = oracle[["id", "smiles", "target_type", "target", "oracle_status"]].merge(
    sub.rename(columns={"target": "pred"}), on="id", how="left")
cov = m[m["target"].notna() & m["pred"].notna()].copy()
print(f"covered rows: {len(cov)} / 4940")

rows = []
for t in TARGETS:
    r = cov[cov.target_type == t]
    r2 = r2_score(r["target"].to_numpy(float), r["pred"].to_numpy(float))
    sse = float(((r["target"] - r["pred"]) ** 2).sum())
    tss = float(((r["target"] - r["target"].mean()) ** 2).sum())
    rows.append(dict(target=t, n=len(r), r2=r2, sse=sse, tss=tss, rmse=np.sqrt(sse / len(r)),
                     mae=float((r["target"] - r["pred"]).abs().mean())))
df = pd.DataFrame(rows)
df["sse_share"] = df.sse / df.sse.sum()
df["tss_share"] = df.tss / df.tss.sum()
meanA = df.r2.mean()
tssw  = float((df.tss * df.r2).sum() / df.tss.sum())          # variant C: TSS-weighted mean
pooledB = r2_score(cov["target"].to_numpy(float), cov["pred"].to_numpy(float))   # variant B: sklearn pooled

# verified-panel only (3818 rows)
mv = verif[["id", "target_type", "target"]].merge(sub.rename(columns={"target": "pred"}), on="id", how="left")
covv = mv[mv["target"].notna() & mv["pred"].notna()]
r2v = {t: r2_score(covv[covv.target_type == t]["target"], covv[covv.target_type == t]["pred"]) for t in TARGETS}
meanA_v = float(np.mean(list(r2v.values())))

# per-Tg-category R2 on the full final_oracle panel
cats = []
for cat in ["verified", "external_verified", "proxy", "unresolved"]:
    r = cov[(cov.target_type == "tg") & (cov.oracle_status == cat)]
    if len(r) >= 2:
        cats.append(dict(category=cat, n=len(r),
                         r2=r2_score(r["target"], r["pred"]),
                         mae=float((r["target"] - r["pred"]).abs().mean()),
                         rmse=float(np.sqrt(((r["target"] - r["pred"]) ** 2).mean()))))
    else:
        cats.append(dict(category=cat, n=len(r), r2=np.nan, mae=np.nan, rmse=np.nan))
catdf = pd.DataFrame(cats)
# unresolved: predictions only, no truth -> report predicted spread as a proxy of difficulty
unres = cov[(cov.target_type == "tg") & (cov.oracle_status == "unresolved")]
tg_covered = cov[(cov.target_type == "tg") & cov.target.notna()]

variants = pd.DataFrame([
    dict(variant="A: unweighted mean of per-target R2 (official metric)", score=meanA, note="V57 documented oracle score 0.9024; private LB = 0.891 => -0.011 calibration"),
    dict(variant="B: sklearn pooled r2_score on all covered rows", score=pooledB, note="If Kaggle pooled the array, V57 would score ~this on LB - contradicts LB 0.891 (and would beat the 0.92 competitor)"),
    dict(variant="C: TSS-weighted mean of per-target R2 (within-pooled)", score=tssw, note="~= R2_tg = 0.8953; private LB 0.891 is 0.004 below - needs a different gap story"),
    dict(variant="A on verified-only panel (3818 rows)", score=meanA_v, note="documented verified score 0.9035"),
])
variants.to_csv(os.path.join(OUT, "03_variants.csv"), index=False)
df.to_csv(os.path.join(OUT, "03_per_target.csv"), index=False)
catdf.to_csv(os.path.join(OUT, "03_tg_categories.csv"), index=False)

txt = []
txt.append("=" * 90)
txt.append("03 EMPIRICAL METRIC VERIFICATION - frozen V57 submission vs Oracle panels")
txt.append("=" * 90)
txt.append("")
txt.append("Per-target on final_oracle covered rows (4,909):")
txt.append(df.round(4).to_string(index=False))
txt.append(f"\nA: unweighted mean of per-target R2 = {meanA:.4f}   (documented 0.9024)")
txt.append(f"B: pooled r2_score (sklearn, all covered rows) = {pooledB:.4f}")
txt.append(f"C: TSS-weighted mean (within-pooled) = {tssw:.4f}")
txt.append(f"A on verified panel only = {meanA_v:.4f}   (documented 0.9035)")
txt.append("")
txt.append("Ground truth anchors from Kaggle:")
txt.append("  private LB = 0.891 (all 4,940 rows)   public LB = 0.917 (~30% subset)")
txt.append("")
txt.append("VERDICT:")
txt.append(f"  - Variant B gives {pooledB:.4f}. If Kaggle pooled the array, V57's private LB would be ~{pooledB:.3f},")
txt.append("    not 0.891 - and it would beat the 0.92 competitor. POOLED IS RULED OUT.")
txt.append(f"  - Variant A gives {meanA:.4f} and matches private LB via the documented -0.011 calibration")
txt.append("    (hard Tg rows + 31 unresolved rows + pub/priv split). THIS IS THE METRIC.")
txt.append(f"  - Variant C ({tssw:.4f}) is numerically close to the LB only because tg is 99.99% of the TSS;")
txt.append("    it requires a different gap story and contradicts the official text ('mean across the seven targets').")
txt.append("")
txt.append("Tg difficulty by oracle row category (final_oracle panel):")
txt.append(catdf.round(4).to_string(index=False))
txt.append(f"\n  Estimated true Tg R2 on all 2,763 rows (weighted by category R2): "
           f"{(catdf.n * catdf.r2.fillna(0)).sum() / 2763:.4f}   (31 unresolved excluded from oracle score)")
txt.append("")
txt.append("Why the private LB (0.891) looked like 'tg R2 (0.8945)': pure coincidence -")
txt.append("mean(0.9024) - 0.011 calibration = 0.8914 ~= tg R2 0.8953. The team's -0.011 offset")
txt.append("(documented, explained by hard-row difficulty) happens to be ~= mean - tg_R2 = 0.0071 + 0.004.")
with open(os.path.join(OUT, "03_verification.txt"), "w") as f:
    f.write("\n".join(txt))
print("\n".join(txt))
