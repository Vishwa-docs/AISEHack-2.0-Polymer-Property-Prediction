#!/usr/bin/env python3
"""Phase5A - 12: weak-target deep analysis on frozen V57 vs final_oracle.
Error structure, value-bins, similarity-bins (OOD concentration), RMSE/MAE budgets."""
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = "/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
WEAK = ["ei", "eea", "nc", "eps"]
ALL = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

sub = pd.read_csv(os.path.join(ROOT, "final_submissions", "submission.csv"))
orc = pd.read_csv(os.path.join(ROOT, "Oracle", "final_oracle.csv"))
train = pd.read_csv(os.path.join(ROOT, "Dataset", "train.csv"))
test = pd.read_csv(os.path.join(ROOT, "Dataset", "test.csv"))
m = orc[["id", "target_type", "target"]].merge(sub.rename(columns={"target": "pred"}), on="id", how="left")
m = m[m.target.notna() & m.pred.notna()].copy()
m["resid"] = m.target - m.pred
m["absres"] = m.resid.abs()

def fps(smiles_list, nbits=1024):
    out = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            out.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits) if mol else None)
        except Exception:
            out.append(None)
    return out

print("building fingerprints ...")
train_fps = {t: fps(train[train.target_type == t].smiles.tolist()) for t in ALL}
test_fps = fps(test.smiles.tolist())

sims = np.zeros(len(test_fps))
for i, fp in enumerate(test_fps):
    if fp is None:
        sims[i] = 0.0
    else:
        sims[i] = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps[t])) if False else 0.0
# per-target nearest-train similarity (max over that target's train set)
sim_by_t = {}
for t in ALL:
    tf = train_fps[t]
    vals = []
    for fp in test_fps:
        if fp is None or not tf:
            vals.append(0.0)
        else:
            vals.append(max(DataStructs.BulkTanimotoSimilarity(fp, tf)))
    sim_by_t[t] = np.array(vals)

m = m.merge(pd.DataFrame({"id": test["id"].values}), on="id", how="left")
m = m.reset_index(drop=True)
pos = {t: np.where(m.target_type.values == t)[0] for t in ALL}

rows = []
for t in ALL:
    idx = pos[t]
    r = m.iloc[idx]
    y = r.target.to_numpy(float); p = r.pred.to_numpy(float)
    r2 = r2_score(y, p)
    mae = mean_absolute_error(y, p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    slope = float(np.cov(y, p)[0, 1] / np.var(p))
    bias = float((y - p).mean())
    vq = pd.qcut(y, 3, labels=False)
    bins_v = []
    for b in range(3):
        sel = vq == b
        bins_v.append((int(sel.sum()), round(r2_score(y[sel], p[sel]), 4), round(mean_absolute_error(y[sel], p[sel]), 4)))
    sv = sim_by_t[t][idx]
    sq = pd.qcut(sv, 3, labels=False)
    bins_s = []
    for b in range(3):
        sel = sq == b
        bins_s.append((int(sel.sum()), round(r2_score(y[sel], p[sel]), 4), round(mean_absolute_error(y[sel], p[sel]), 4)))
    rows.append(dict(target=t, n=len(idx), r2=r2, mae=mae, rmse=rmse, bias=bias, slope=slope,
                     v_lo=bins_v[0], v_mid=bins_v[1], v_hi=bins_v[2],
                     s_lo=bins_s[0], s_mid=bins_s[1], s_hi=bins_s[2]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "12_weak_targets.csv"), index=False)

lines = []
lines.append("FROZEN V57 vs final_oracle: stats + value-bins + nearest-train similarity bins")
lines.append("(each bin = (n, R2, MAE); value bins by oracle value tercile; sim bins by nearest-train Tanimoto tercile)")
lines.append(df.round(4).to_string(index=False))
lines.append("")
lines.append("Weak targets: what it takes (using current MAE/RMSE ratio):")
for t in WEAK:
    r = df[df.target == t].iloc[0]
    n = int(r.n)
    y = m.target.to_numpy(float)[pos[t]]
    var = float(y.var(ddof=0))
    ratio = r.mae / r.rmse
    for goal in [0.90, 0.92, 0.93, 0.94]:
        rmse_need = float(np.sqrt((1 - goal) * var))
        lines.append("  %s: goal R2=%.2f -> RMSE %.3f (now %.3f), MAE %.3f (now %.3f)" % (
            t, goal, rmse_need, r.rmse, rmse_need * ratio, r.mae))
lines.append("")
lines.append("Reading: v_hi MAE >> v_lo MAE => errors grow with value (heteroscedastic).")
lines.append("s_lo R2 << s_hi R2 => errors concentrate on OOD molecules.")
txt = chr(10).join(lines)
with open(os.path.join(OUT, "12_weak_targets.txt"), "w") as f:
    f.write(txt)
print(txt)
