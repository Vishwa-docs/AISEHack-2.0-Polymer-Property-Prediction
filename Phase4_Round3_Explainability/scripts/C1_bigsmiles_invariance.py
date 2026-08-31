"""
C1_bigsmiles_invariance.py
==========================
EXP-C1 - polymer notation invariance: '*CC*', '[*]CC[*]', 'CC(*)*' etc. must
produce identical predictions after canonicalization.
Outputs: notation_invariance.csv
"""
import time

import numpy as np
import pandas as pd

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, featurize, predict_ensemble, smoke_n)


def variants(smi):
    """Alternative valid notations for the same polymer."""
    s = str(smi).replace('[', '').replace(']', '')
    out = [smi]
    if '*' in s:
        out.append(s.replace('*', '[*]'))          # bracket stars
        out.append('(' + s + ')')                  # extra parens
        # reorder terminal star positions: *A* vs A(*)
        if s.startswith('*') and s.endswith('*') and len(s) > 2:
            inner = s[1:-1]
            out.append(inner + '(*)')
            out.append('(*)' + inner)
    return list(dict.fromkeys(out))[:4]


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []
    for target in TARGETS:
        pkl = load_proxy(target)
        df_t = train[train['target_type'] == target].copy()
        df_t = df_t[df_t['smiles'].astype(str).str.contains(chr(42), regex=False)]
        sample = df_t.sample(min(smoke_n(50, 10), len(df_t)), random_state=SEED) if len(df_t) else df_t
        for _, row in sample.iterrows():
            vs = variants(row['smiles'])
            if len(vs) < 2:
                continue
            X, _ = featurize(vs, pipe=pkl['pipe'], canonicalize=True)
            preds = predict_ensemble(X, pkl)
            rows.append({'target': target, 'smiles': row['smiles'],
                         'n_notations': len(vs), 'pred_mean': float(preds.mean()),
                         'pred_std': float(preds.std()),
                         'max_abs_dev': float(np.abs(preds - preds[0]).max())})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'notation_invariance.csv', index=False)
    if len(df):
        print(f'  mean prediction std across notations: {df[chr(112)+chr(114)+chr(101)+chr(100)+chr(95)+chr(115)+chr(116)+chr(100)].mean():.5f}')
    print(f'C1_bigsmiles_invariance.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
