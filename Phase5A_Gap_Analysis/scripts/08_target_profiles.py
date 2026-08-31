#!/usr/bin/env python3
"""Phase5A - 08: exact RMSE budgets for target profiles."""
import os
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "output")
df = pd.read_csv(os.path.join(OUT, "03_per_target.csv")).set_index("target")
profiles = {
  "S2 realistic":  {"tg":.920,"egc":.922,"egb":.940,"ei":.905,"eea":.928,"nc":.920,"eps":.910},
  "S7 aggressive": {"tg":.925,"egc":.925,"egb":.945,"ei":.915,"eea":.935,"nc":.925,"eps":.920},
  "S8 ceiling":    {"tg":.930,"egc":.930,"egb":.950,"ei":.920,"eea":.940,"nc":.930,"eps":.925},
  "0.935 profile": {"tg":.935,"egc":.935,"egb":.955,"ei":.920,"eea":.945,"nc":.935,"eps":.920},
}
lines = []
lines.append(f"{'target':>6} {'curR2':>7} {'n':>5} {'var':>10} {'curRMSE':>8} {'R2@prof':>8} {'RMSE@prof':>9} {'dRMSE':>7}")
for t in df.index:
    r = df.loc[t]
    var = r.tss / r.n
    prof = profiles["0.935 profile"][t]
    rmse_prof = np.sqrt((1 - prof) * var)
    lines.append(f"{t:>6} {r.r2:7.4f} {int(r.n):5d} {var:10.4f} {r.rmse:8.4f} {prof:8.3f} {rmse_prof:9.4f} {rmse_prof - r.rmse:7.3f}")
lines.append("")
for name, pr in profiles.items():
    m = float(np.mean(list(pr.values())))
    gains = {t: round(pr[t] - df.loc[t, "r2"], 4) for t in pr}
    lines.append(f"{name}: mean {m:.4f} | est.private {m - 0.011:.4f} | per-target gains {gains}")
txt = chr(10).join(lines)
with open(os.path.join(OUT, "08_target_profiles.txt"), "w") as f:
    f.write(txt)
print(txt)
