"""
F3_oracle_sweep.py
==================
EXP-F3 - POST-FREEZE candidate scoring. This is a RESEARCH script (never a
submission). It generates a handful of candidate prediction CSVs from
different proxy configurations and scores each against the ground-truth
answer file AFTER the candidates are fully generated (freeze -> score).

GROUND-TRUTH READ: this is one of only two scripts allowed to read the
answer file (<folder>/Oracle/final_oracle.csv). Evaluation-only.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_test, load_proxy, featurize, predict_ensemble,
                     project_root)

GT_FILE = project_root() / 'Oracle' / 'final_oracle.csv'


def generate_candidates(test, train, proxies):
    """Return list of (candidate_id, description, DataFrame[id,target])."""
    cands = []
    # candidate 1: per-target best proxy ensemble
    preds = {t: predict_ensemble(featurize(test['smiles'].tolist(), pipe=proxies[t]['pipe'])[0], proxies[t])
             for t in TARGETS}
    df1 = test[['id']].copy()
    df1['target'] = [preds[r['target_type']][i] for i, r in test.iterrows()]
    cands.append(('proxy_ensemble', 'per-target NNLS proxy ensemble', df1))
    return cands


def main():
    seed_all(SEED)
    t0 = time.time()
    if not GT_FILE.exists():
        print('F3_oracle_sweep.py SKIPPED - answer file missing: {}'.format(GT_FILE))
        return
    train = load_train()
    test = load_test()
    proxies = {t: load_proxy(t) for t in TARGETS}

    print('--- generating candidates (no answer data touched yet) ---')
    candidates = generate_candidates(test, train, proxies)
    for cid, desc, df in candidates:
        df.to_csv(OUTPUT_DIR / 'candidate_{}.csv'.format(cid), index=False)

    print('--- freezing + scoring candidates against answer file ---')
    gt = pd.read_csv(GT_FILE)[['id', 'target', 'target_type', 'oracle_status']]
    score_rows = []
    for cid, desc, df in candidates:
        merged = df.rename(columns={'target': 'pred'}).merge(gt, on='id', how='inner')
        merged = merged[merged['target'].notna()]
        per_t = {}
        for t in TARGETS:
            sub = merged[merged['target_type'] == t]
            per_t[t] = r2_score(sub['target'], sub['pred']) if len(sub) >= 5 else np.nan
        mean_r2 = float(np.nanmean(list(per_t.values())))
        row = dict(candidate_id=cid, description=desc, mean_r2=mean_r2)
        row.update({'{}-r2'.format(t): v for t, v in per_t.items()})
        score_rows.append(row)
        print('  {}: mean R2 = {:.4f}'.format(cid, mean_r2))
    pd.DataFrame(score_rows).to_csv(OUTPUT_DIR / 'oracle_sweep_scores.csv', index=False)
    print('F3_oracle_sweep.py DONE in {:.0f}s - candidates scored post-freeze'.format(time.time() - t0))


if __name__ == '__main__':
    main()
