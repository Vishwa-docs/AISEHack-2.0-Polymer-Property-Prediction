#!/usr/bin/env python3
"""Reusable per-target scorer (DIAGNOSTIC — reads final_oracle, NOT part of the pipeline).
Usage: python score.py <submission.csv> [oracle.csv]
Prints per-target R2, mean, and estimated private LB (mean - 0.011)."""
import sys, numpy as np, pandas as pd
ROOT = "/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']

def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    t = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - p) ** 2) / t if t > 0 else float('nan')

def main():
    sub_path = sys.argv[1]
    orc_path = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/Oracle/final_oracle.csv"
    sub = pd.read_csv(sub_path)                       # id, target
    orc = pd.read_csv(orc_path)                       # id, smiles, target_type, target, ...
    m = orc[['id', 'target_type', 'target']].rename(columns={'target': 'oracle'}).merge(
        sub.rename(columns={'target': 'pred'}), on='id', how='left')
    m = m[m['oracle'].notna() & m['pred'].notna()]
    rs = {}
    print(f"{'target':6s} {'n':>5s} {'R2':>9s}")
    for t in TARGETS:
        s = m[m['target_type'] == t]
        rs[t] = r2(s['oracle'], s['pred'])
        print(f"{t:6s} {len(s):5d} {rs[t]:9.5f}")
    mean = float(np.mean([rs[t] for t in TARGETS]))
    print("-" * 22)
    print(f"{'MEAN':6s} {sum(len(m[m['target_type']==t]) for t in TARGETS):5d} {mean:9.5f}")
    print(f"est. private LB (mean - 0.011): {mean - 0.011:.5f}")

if __name__ == '__main__':
    main()
