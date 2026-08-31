#!/usr/bin/env python3
"""Phase5A Gap Analysis - 01: Dataset EDA. Verifies counts/variance/TSS claims from the user's analysis.
All outputs written under Phase5A_Gap_Analysis/output/. ORACLE-ASSISTED analysis only - nothing here enters a submission."""
import os, json
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Phase5A_Gap_Analysis
ROOT = os.path.dirname(BASE)                                          # repo root
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", OUT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ORDER   = {"tg": 0, "egc": 1, "egb": 2, "ei": 3, "eea": 4, "nc": 5, "eps": 6}
NAMES = {"tg": "Glass Transition Tg (degC)", "egc": "Chain Bandgap (eV)", "egb": "Bulk Bandgap (eV)",
         "ei": "Ionisation Energy (eV)", "eea": "Electron Affinity (eV)", "nc": "Refractive Index",
         "eps": "Dielectric Constant"}

train = pd.read_csv(os.path.join(ROOT, "Dataset", "train.csv"))
test  = pd.read_csv(os.path.join(ROOT, "Dataset", "test.csv"))
train["target"] = pd.to_numeric(train["target"], errors="coerce")
print(f"train rows: {len(train)}  test rows: {len(test)}")

# ---------------- per-target stats ----------------
rows = []
for t in TARGETS:
    sub = train[train["target_type"] == t]["target"].dropna()
    n = len(sub); mu = sub.mean(); sd = sub.std(ddof=1); var = sub.var(ddof=1)
    tss = (n - 1) * var
    rows.append({
        "target": t, "name": NAMES[t],
        "train_n": n, "train_mean": mu, "train_std": sd, "train_min": sub.min(),
        "train_q25": sub.quantile(0.25), "train_median": sub.median(), "train_q75": sub.quantile(0.75),
        "train_max": sub.max(), "train_skew": sub.skew(),
        "train_var": var, "train_TSS": tss,
        "test_n": int((test["target_type"] == t).sum()),
    })
df = pd.DataFrame(rows)
df["train_row_share"] = df["train_n"] / len(train)
df["test_row_share"]  = df["test_n"] / len(test)
df["train_TSS_share"] = df["train_TSS"] / df["train_TSS"].sum()
df.to_csv(os.path.join(OUT, "01_eda_summary.csv"), index=False)

txt = []
txt.append("=" * 78)
txt.append("01 DATASET EDA - per-target statistics (train.csv, 7,409 rows; test.csv, 4,940 rows)")
txt.append("=" * 78)
txt.append(df.round(4).to_string(index=False))
txt.append("")
txt.append(f"Total train TSS (sum over targets, within-target only): {df['train_TSS'].sum():,.0f}")
txt.append(f"Total test  rows by target sum: {df['test_n'].sum()}  (must be 4940)")

# ---------------- verify the user's specific claims ----------------
claims = []
def check(name, ok, detail):
    claims.append({"claim": name, "verified": "YES" if ok else "NO", "detail": detail})

tg = df[df.target == "tg"].iloc[0]; ei = df[df.target == "ei"].iloc[0]
nc = df[df.target == "nc"].iloc[0]; egc = df[df.target == "egc"].iloc[0]
eps = df[df.target == "eps"].iloc[0]; egb = df[df.target == "egb"].iloc[0]; eea = df[df.target == "eea"].iloc[0]

check("tg has 4,143 train rows", tg.train_n == 4143, f"actual {tg.train_n}")
check("ei has 222 train rows", ei.train_n == 222, f"actual {ei.train_n}")
check("nc has 229 train rows", nc.train_n == 229, f"actual {nc.train_n}")
check("tg std ~ 109.1", abs(tg.train_std - 109.1) < 0.3, f"actual {tg.train_std:.4f}")
check("ei std ~ 1.04", abs(ei.train_std - 1.04) < 0.05, f"actual {ei.train_std:.4f}")
check("nc std ~ 0.23", abs(nc.train_std - 0.23) < 0.02, f"actual {nc.train_std:.4f}")
v_tg_claim = 4143 * 109.1 ** 2
v_tg_real  = tg.train_n * tg.train_var
check("tg internal variance ~ 49,298,827 (4143*109.1^2)", abs(v_tg_claim - 49_298_827) < 50_000,
      f"claim {v_tg_claim:,.0f} vs exact n*var {v_tg_real:,.0f}")
check("ei internal variance ~ 240 (222*1.04^2)", abs(222 * 1.04 ** 2 - 240.1) < 5,
      f"claim {222*1.04**2:.1f} vs exact {ei.train_n*ei.train_var:.1f}")
check("nc internal variance ~ 12 (229*0.23^2)", abs(229 * 0.23 ** 2 - 12.1) < 2,
      f"claim {229*0.23**2:.1f} vs exact {nc.train_n*nc.train_var:.1f}")
check("Tg is 55.9% of TEST rows (2763/4940)", abs(tg.test_n / 4940 - 0.559) < 0.001,
      f"actual {tg.test_n}/4940 = {tg.test_n/4940:.4f}")
check("Tg is 55.9% of TRAIN rows", abs(tg.train_n / 7409 - 0.559) < 0.001,
      f"actual {tg.train_n}/7409 = {tg.train_n/7409:.4f}")

# pooled TSS decomposition (train, real; test, proxy using train mean/std)
gmean = train["target"].mean()
within_train = float(df["train_TSS"].sum())
between_train = float(((df["train_n"] * (df["train_mean"] - gmean) ** 2)).sum())
pooled_train = within_train + between_train
tcounts = df["test_n"].to_numpy(); tmeans = df["train_mean"].to_numpy(); tstds = df["train_std"].to_numpy()
within_test = float(((tcounts - 1) * tstds ** 2).sum())
tglobal = float((tcounts * tmeans).sum() / tcounts.sum())
between_test = float((tcounts * (tmeans - tglobal) ** 2).sum())
pooled_test = within_test + between_test

txt.append("")
txt.append("-" * 78)
txt.append("POOLED (all-rows) TSS decomposition  [what a 'pool the array' metric would see]")
txt.append("-" * 78)
txt.append(f"TRAIN: within-target TSS  = {within_train:,.0f}  ({within_train/pooled_train*100:.1f}% of pooled)")
txt.append(f"TRAIN: between-target TSS = {between_train:,.0f}  ({between_train/pooled_train*100:.1f}% of pooled)")
txt.append(f"TRAIN: pooled TSS         = {pooled_train:,.0f}   (global mean = {gmean:.2f})")
txt.append(f"TEST(proxy): within  = {within_test:,.0f}   between = {between_test:,.0f}   pooled = {pooled_test:,.0f}")
txt.append(f"TEST(proxy): between-target variance is {between_test/within_test:.1f}x the within-target variance")
txt.append("")
txt.append("Claim check: 'saving 7,000 SSE on a denominator of ~84,000,000 improves score by 0.00008'")
sum6 = df[df.target != "tg"]["train_TSS"].sum()
txt.append(f"  - Actual TEST-proxy pooled TSS = {pooled_test:,.0f}  (claim said ~84M; actual ~{pooled_test/1e6:.0f}M)")
txt.append(f"  - Actual TRAIN TSS of the 6 non-tg targets = {sum6:,.0f} (claim implied ~7,000)")
txt.append(f"  - So even under a fully pooled metric, perfecting all 6 small targets saves at most {sum6:,.0f} SSE,")
txt.append(f"    i.e. {sum6/pooled_test:.6f} in pooled R^2 terms (claim said 0.00008; actual {sum6/pooled_test:.8f})")

# ---------------- label conflicts / duplicates ----------------
dup = train.groupby(["smiles", "target_type"])["target"].agg(["count", "nunique", "min", "max"])
conflict = dup[dup["nunique"] > 1]
txt.append("")
txt.append("-" * 78)
txt.append("LABEL QUALITY")
txt.append("-" * 78)
txt.append(f"exact (smiles,target_type) duplicated rows: {(dup['count']>1).sum()}  |  with CONFLICTING values (nunique>1): {len(conflict)}")
if len(conflict):
    txt.append(conflict.head(10).round(3).to_string())
multi = train.groupby("smiles")["target_type"].nunique()
txt.append(f"train SMILES appearing under >=2 target types: {(multi>1).sum()} (of {multi.shape[0]} unique) -> cross-property learning possible for these")
tmulti = test.groupby("smiles")["target_type"].nunique()
txt.append(f"test  SMILES appearing under >=2 target types: {(tmulti>1).sum()} (of {tmulti.shape[0]} unique)")
overlap_tt = len(set(train['smiles']) & set(test['smiles']))
txt.append(f"SMILES in BOTH train and test: {overlap_tt} (AGENTS says 457)")

with open(os.path.join(OUT, "01_eda_summary.txt"), "w") as f:
    f.write("\n".join(txt))
with open(os.path.join(OUT, "01_claims.json"), "w") as f:
    json.dump(claims, f, indent=1)

# ---------------- plots ----------------
plt.style.use("seaborn-v0_8-whitegrid")
c = df["target"].tolist()
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.38
ax.bar(np.arange(7) - w/2, df["train_n"], w, label="train", color="#4C72B0")
ax.bar(np.arange(7) + w/2, df["test_n"], w, label="test", color="#DD8452")
ax.set_xticks(np.arange(7)); ax.set_xticklabels(c)
ax.set_title("Rows per target (train 7,409 / test 4,940) - Tg = 56% of every row")
ax.legend(); ax.set_ylabel("rows")
for i, (a, b) in enumerate(zip(df["train_n"], df["test_n"])):
    ax.text(i - w/2, a + 60, str(a), ha="center", fontsize=8)
    ax.text(i + w/2, b + 60, str(b), ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_01_counts.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(np.arange(7), df["train_std"], color="#55A868")
ax.set_xticks(np.arange(7)); ax.set_xticklabels(c)
ax.set_title("Per-target std (train) - scale mixing is extreme; Tg std is 470x nc std")
for i, v in enumerate(df["train_std"]):
    ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_01_std.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(np.arange(7), df["train_TSS_share"] * 100, color="#C44E52")
ax.set_xticks(np.arange(7)); ax.set_xticklabels(c)
ax.set_yscale("log")
ax.set_title("Share of within-target TSS (train, %) - log scale; Tg = 99.5% of all TSS")
for i, v in enumerate(df["train_TSS_share"] * 100):
    ax.text(i, v * 1.2, f"{v:.2f}%", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_01_tss.png"), dpi=120); plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
for ax, t in zip(axes.ravel(), TARGETS + ["ALL"]):
    if t == "ALL":
        data = train["target"]
        ax.hist(data, bins=80, color="gray", alpha=0.8)
        ax.set_title("ALL targets pooled")
    else:
        data = train[train["target_type"] == t]["target"].dropna()
        ax.hist(data, bins=60, color="#4C72B0", alpha=0.85)
        ax.set_title(f"{t}  (n={len(data)}, sd={data.std():.3g})")
    ax.tick_params(labelsize=8)
plt.suptitle("Train distributions per target - pooled histogram is a meaningless scale soup", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_01_hist.png"), dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
zs = [ (train[train.target_type == t]["target"] - train[train.target_type == t]["target"].mean()) /
       train[train.target_type == t]["target"].std(ddof=1) for t in TARGETS ]
ax.boxplot(zs, labels=TARGETS, showfliers=False)
ax.set_title("Per-target distributions z-scored - shapes are similar once standardized")
ax.set_ylabel("z-score")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_01_box.png"), dpi=120); plt.close()

print("\n".join(txt))
print("\nCLAIMS:")
for cl in claims:
    print(f"  [{cl['verified']}] {cl['claim']}  ->  {cl['detail']}")
