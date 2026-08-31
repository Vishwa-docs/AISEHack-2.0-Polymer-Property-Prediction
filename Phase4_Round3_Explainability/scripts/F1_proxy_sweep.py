"""
F1_proxy_sweep.py
=================
EXP-F1 - hyperparameter sweep over proxy models (10 configs x 7 targets = 70
experiments), GroupKFold on canonical SMILES, logged to
proxy_sweep_results.csv; best config per target saved to proxy_sweep_best.json.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     featurize, canonical_smiles, smoke_n)

CONFIGS = [
    dict(model='lgbm', n_estimators=200, lr=0.05, leaves=31),
    dict(model='lgbm', n_estimators=400, lr=0.05, leaves=63),
    dict(model='lgbm', n_estimators=400, lr=0.03, leaves=63),
    dict(model='lgbm', n_estimators=600, lr=0.02, leaves=127),
    dict(model='ridge', alpha=10),
    dict(model='ridge', alpha=100),
    dict(model='ridge', alpha=1000),
    dict(model='et', n_estimators=100),
    dict(model='et', n_estimators=300),
    dict(model='et', n_estimators=500),
]


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []
    best = {}

    for target in TARGETS:
        df_t = train[train['target_type'] == target].copy()
        if SMOKE and len(df_t) > 100:
            df_t = df_t.sample(100, random_state=SEED).reset_index(drop=True)
        df_t['canonical'] = df_t['smiles'].apply(canonical_smiles)
        X, _ = featurize(df_t['smiles'].tolist())
        y = df_t['target'].values.astype(float)
        gkf = GroupKFold(n_splits=5 if len(df_t) >= 500 else 3)
        target_best = None
        for cfg in CONFIGS:
            oof = np.zeros(len(df_t))
            for tr, va in gkf.split(X, y, df_t['canonical'].values):
                X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]
                if cfg['model'] == 'ridge':
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler().fit(X_tr)
                    m = Ridge(alpha=cfg['alpha'], random_state=SEED)
                    m.fit(sc.transform(X_tr), y_tr)
                    oof[va] = m.predict(sc.transform(X_va))
                elif cfg['model'] == 'et':
                    m = ExtraTreesRegressor(n_estimators=smoke_n(cfg['n_estimators'], 30),
                                            n_jobs=-1, random_state=SEED, min_samples_leaf=2)
                    m.fit(X_tr, y_tr)
                    oof[va] = m.predict(X_va)
                else:
                    m = lgb.LGBMRegressor(n_estimators=smoke_n(cfg['n_estimators'], 30),
                                          learning_rate=cfg['lr'], num_leaves=cfg['leaves'],
                                          random_state=SEED, verbosity=-1, n_jobs=-1)
                    m.fit(X_tr, y_tr)
                    oof[va] = m.predict(X_va)
            r2 = r2_score(y, oof)
            rows.append(dict(target=target, model=cfg['model'], config=json.dumps(cfg),
                             oof_r2=r2, n_train=len(df_t)))
            if target_best is None or r2 > target_best[1]:
                target_best = (cfg, r2)
        best[target] = target_best[0]
        print('  {}: best = {} ({:.4f})'.format(target, target_best[0]['model'], target_best[1]))

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'proxy_sweep_results.csv', index=False)
    with open(OUTPUT_DIR / 'proxy_sweep_best.json', 'w') as f:
        json.dump(best, f, indent=2)
    print('F1_proxy_sweep.py DONE in {:.0f}s - {} runs logged'.format(time.time() - t0, len(rows)))


if __name__ == '__main__':
    main()
