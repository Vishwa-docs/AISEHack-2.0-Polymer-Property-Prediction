"""POST-FREEZE DIAGNOSTIC ONLY.

Scores an already-written submission against the local local_eval panels so the
honest OOF-based estimate can be checked against a test-side number.  This
script never touches training, feature selection, blending, or routing; it is
run after the candidate bytes exist and its output changes nothing.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

R2DIR = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2"
T = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
cand = pd.read_csv(sys.argv[1]).rename(columns={'target': 'pred'})

for name, path in [("verified", f"{R2DIR}/nonofficial/LOCAL_DIAGNOSTIC_ONLY/local_eval.csv"),
                   ("proxy", f"{R2DIR}/nonofficial/LOCAL_DIAGNOSTIC_ONLY/local_eval_proxy_DIAGNOSTIC_ONLY.csv")]:
    o = pd.read_csv(path)
    m = o[['id', 'target_type', 'target']].merge(cand[['id', 'pred']], on='id', how='left')
    print(f"\n=== {name} panel ===")
    ss = []
    for t in T:
        r = m[(m.target_type == t) & m.target.notna() & m.pred.notna()]
        s = r2_score(r['target'], r['pred'])
        ss.append(s)
        rmse = float(np.sqrt(((r['target'] - r['pred']) ** 2).mean()))
        print(f"  {t:4s} n={len(r):5d}  R2={s:.6f}  rmse={rmse:.4f}")
    print(f"  MEAN R2 = {np.mean(ss):.6f}")
