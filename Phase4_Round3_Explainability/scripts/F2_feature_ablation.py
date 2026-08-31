"""
F2_feature_ablation.py
======================
EXP-F2 - systematic feature-group ablation (10 configs x 7 targets = 70
experiments) with the quick LightGBM proxy. Answers: which feature groups
are essential, which are redundant.
Outputs: feature_ablation_results.csv, feature_ablation_heatmap.png
"""
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, canonical_smiles, save_plot,
                     smoke_n, style_ax)

ABLATIONS = [
    'all_features', 'no_morgan', 'no_rdkit_desc', 'no_char_ngram',
    'morgan_only', 'rdkit_only', 'char_only', 'morgan_rdkit', 'morgan_char', 'rdkit_char',
]


def select_blocks(X, feat_names, ablation):
    n_m = sum(1 for f in feat_names if f.startswith('morgan_'))
    n_d = sum(1 for f in feat_names if not f.startswith('morgan_') and not f.startswith('ngram_'))
    n_g = len(feat_names) - n_m - n_d
    m = np.arange(0, n_m)
    d = np.arange(n_m, n_m + n_d)
    g = np.arange(n_m + n_d, n_m + n_d + n_g)
    keep = {'all_features': [m, d, g], 'no_morgan': [d, g], 'no_rdkit_desc': [m, g],
            'no_char_ngram': [m, d], 'morgan_only': [m], 'rdkit_only': [d],
            'char_only': [g], 'morgan_rdkit': [m, d], 'morgan_char': [m, g],
            'rdkit_char': [d, g]}[ablation]
    idx = np.concatenate(keep)
    return X[:, idx]


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []
    for target in TARGETS:
        df_t = train[train['target_type'] == target].copy()
        if SMOKE and len(df_t) > 100:
            df_t = df_t.sample(100, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        X = rebuild_features(df_t['smiles'].tolist(), pkl['pipe'])
        y = df_t['target'].values.astype(float)
        canon = df_t['smiles'].apply(canonical_smiles).values
        feat_names = pkl['pipe']['feat_names']
        for ab in ABLATIONS:
            Xs = select_blocks(X, feat_names, ab) if ab != 'all_features' else X
            gkf = GroupKFold(n_splits=3)
            oof = np.zeros(len(df_t))
            for tr, va in gkf.split(Xs, y, canon):
                m = lgb.LGBMRegressor(n_estimators=smoke_n(200, 30), learning_rate=0.05,
                                      num_leaves=31, random_state=SEED, verbosity=-1, n_jobs=-1)
                m.fit(Xs[tr], y[tr])
                oof[va] = m.predict(Xs[va])
            rows.append(dict(target=target, ablation=ab, oof_r2=r2_score(y, oof),
                             n_features=Xs.shape[1]))
        core = [r for r in rows if r['target'] == target and r['ablation'] in
                ('all_features', 'no_morgan', 'no_rdkit_desc', 'no_char_ngram')]
        print('  {}: {}'.format(target, ' '.join('{}={:.3f}'.format(r['ablation'], r['oof_r2']) for r in core)))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'feature_ablation_results.csv', index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot(index='ablation', columns='target', values='oof_r2').reindex(ABLATIONS)
    im = ax.imshow(pivot.values, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS)
    ax.set_yticks(range(len(ABLATIONS))); ax.set_yticklabels(ABLATIONS)
    for i in range(len(ABLATIONS)):
        for j in range(len(TARGETS)):
            v = pivot.values[i, j]
            if v == v:
                ax.text(j, i, '{:.2f}'.format(v), ha='center', va='center', fontsize=8,
                        color='white' if v < 0.6 else 'black')
    ax.set_title('Feature ablation - OOF R2 (quick LGBM)')
    fig.colorbar(im, ax=ax, label='R2')
    save_plot(fig, 'feature_ablation_heatmap.png')
    print('F2_feature_ablation.py DONE in {:.0f}s - {} runs logged'.format(time.time() - t0, len(rows)))


if __name__ == '__main__':
    main()
