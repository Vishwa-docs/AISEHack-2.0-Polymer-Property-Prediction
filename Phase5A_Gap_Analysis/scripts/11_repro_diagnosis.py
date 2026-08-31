#!/usr/bin/env python3
"""Diagnose V57 reproduction: frozen submission vs P5A-100 fresh run per target."""
import os, sys
import pandas as pd, numpy as np
ROOT = "/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
T = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
frozen = pd.read_csv(os.path.join(ROOT, "final_submissions", "submission.csv"))
fresh = pd.read_csv(os.path.join(ROOT, "Phase5A_Gap_Analysis", "experiments", "P5A-100-v57-baseline", "predictions.csv"))
test = pd.read_csv(os.path.join(ROOT, "Dataset", "test.csv"))
m = test[["id", "target_type"]].merge(frozen.rename(columns={"target": "frozen"}), on="id").merge(fresh.rename(columns={"target": "fresh"}), on="id")
print("FROZEN V57 vs P5A-100 FRESH RUN (both are 'V57' - fresh run diverged on this machine):")
for t in T:
    r = m[m.target_type == t]
    d = (r.frozen - r.fresh).abs()
    corr = np.corrcoef(r.frozen, r.fresh)[0, 1]
    print("  %-4s n=%4d  mean|diff|=%.4f  max=%.2f  corr=%.4f" % (t, len(r), d.mean(), d.max(), corr))
print()
print("Version-sensitive models in the standalone (sklearn/RDKit dependent):")
import re
src = open(os.path.join(ROOT, "final_submissions", "v57_reproduction_standalone.py")).read()
for pat in ["SVR(", "MLPRegressor(", "rdEHT", "GaussianProcessRegressor(", "Descriptors3D", "EmbeddedETKDG"]:
    n = src.count(pat)
    if n:
        print("  %-28s x%d" % (pat, n))
