#!/usr/bin/env python3
"""Phase5A - 10: physics identity validation on train (egb~egc, ei=egc+eea, eps=nc^2+ionic)."""
import os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, "output")

train = pd.read_csv(os.path.join(ROOT, "Dataset", "train.csv"))
train["target"] = pd.to_numeric(train["target"], errors="coerce")

def grab(t):
    d = train[train.target_type == t][["smiles", "target"]].dropna()
    return d.rename(columns={"target": t})

egc = grab("egc"); egb = grab("egb"); ei = grab("ei"); eea = grab("eea"); eps = grab("eps"); nc = grab("nc")

lines = []
lines.append("=== 10: PHYSICS IDENTITIES ON TRAIN (same-polymer paired labels) ===")
lines.append("")

j = egc.merge(egb, on="smiles")
lines.append("egb vs egc: %d same-polymer pairs (of egb %d rows)" % (len(j), len(egb)))
if len(j) > 10:
    x = j.egc.to_numpy(float); y = j.egb.to_numpy(float)
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    lines.append("  linear fit egb = a*egc + b:  a=%.4f  b=%.3f  R2=%.4f  resid_std=%.4f eV" % (coef[0], coef[1], r2, np.std(y - pred, ddof=1)))

jei = ei.merge(eea, on="smiles").merge(egc, on="smiles")
lines.append("ei vs (egc+eea): %d same-polymer triples (ei %d rows)" % (len(jei), len(ei)))
if len(jei) > 10:
    d = (jei.ei - (jei.egc + jei.eea)).to_numpy(float)
    r2_id = 1 - float(((jei.ei - (jei.egc + jei.eea)) ** 2).sum()) / float(((jei.ei - jei.ei.mean()) ** 2).sum())
    lines.append("  mean(ei - egc - eea) = %.4f eV   std = %.4f eV" % (d.mean(), d.std(ddof=1)))
    lines.append("  R2 of predicting ei = egc + eea on these pairs: %.4f" % r2_id)

jnc = eps.merge(nc, on="smiles")
lines.append("eps vs nc: %d same-polymer pairs" % len(jnc))
if len(jnc) > 10:
    ionic = (jnc.eps - jnc.nc ** 2).to_numpy(float)
    r2_eps_nc2 = 1 - float(((jnc.eps - jnc.nc ** 2) ** 2).sum()) / float(((jnc.eps - jnc.eps.mean()) ** 2).sum())
    lines.append("  ionic = eps - nc^2: mean %.4f  std %.4f" % (ionic.mean(), ionic.std(ddof=1)))
    lines.append("  eps std %.4f vs ionic std %.4f -> %.1f%% of eps variance removed by the nc^2 baseline"
                 % (jnc.eps.std(ddof=1), ionic.std(ddof=1), (1 - ionic.std(ddof=1) / jnc.eps.std(ddof=1)) * 100))
    lines.append("  R2 of predicting eps = nc^2 alone: %.4f  (the residual = ionic is what must be modeled)" % r2_eps_nc2)

dup = train[train.target_type == "tg"].groupby("smiles")["target"].agg(["count", "nunique", "std"])
dup = dup[dup["count"] > 1]
lines.append("")
if len(dup):
    lines.append("Tg duplicated-polymer labels: %d polymers with 2+ measurements; avg within-polymer scatter sd %.2f C"
                 % (len(dup), dup["std"].mean()))
else:
    lines.append("Tg duplicated-polymer labels: none")

txt = chr(10).join(lines)
with open(os.path.join(OUT, "10_physics_identities.txt"), "w") as f:
    f.write(txt)
print(txt)
