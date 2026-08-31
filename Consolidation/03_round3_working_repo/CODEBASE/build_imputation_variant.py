#!/usr/bin/env python3
"""
build_imputation_variant.py  —  ROUTE A (compound / physics-imputation variant)

PURE, ORACLE-FREE post-processing overlay on the V57 base submission.
Reads ONLY:  Dataset/train.csv, Dataset/test.csv, and a base submission CSV
(produced by pipeline_v57_final.py). Writes a new submission CSV.

WHAT IT DOES
------------
Polymer electronic properties obey exact single-chain identities:
    egc (chain gap) = ei (ionisation) - eea (electron affinity)
For a TEST row asking for `egc`, when the SAME polymer (RDKit isomeric-canonical
SMILES) carries BOTH an `ei` and an `eea` label in train.csv, we can reconstruct
egc directly from those exact partner labels instead of relying on the learned
model. This is the ONE identity that is train-validated to help (see FEASIBILITY.md):
on the covered rows the reconstruction tracks the truth far better than the base.

Every OTHER identity (ei=egc+eea, eps=nc^2+ionic, nc=sqrt(eps-ionic), egb=a*egc+b)
was tested and REJECTED — the train partner labels are too noisy (identity RMSE
0.18-0.41) and the V57 base already consumes partner labels as learned features,
so a hard override degrades those targets. We therefore GUARD: apply egc only,
fall back to the V57 base everywhere else. Result is >= V57 by construction.

USAGE
-----
    python build_imputation_variant.py \
        --train Dataset/train.csv --test Dataset/test.csv \
        --base submission_v57.csv --out submission_imputation.csv
"""
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def canonical(smiles):
    """RDKit isomeric-canonical SMILES — the join key used throughout V57."""
    try:
        mol = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def build_partner_lookup(train_df):
    """canonical SMILES -> {target_type: median label} over train duplicates."""
    tr = train_df.copy()
    tr['cs'] = tr['smiles'].map(canonical)
    tr = tr.dropna(subset=['cs'])
    lut = {}
    for (cs, tt), g in tr.groupby(['cs', 'target_type']):
        lut.setdefault(cs, {})[tt] = float(np.median(g['target'].to_numpy(float)))
    return lut


def main(argv=None):
    ap = argparse.ArgumentParser(description='Route A: physics-imputation overlay on V57 base')
    ap.add_argument('--train', default='Dataset/train.csv')
    ap.add_argument('--test', default='Dataset/test.csv')
    ap.add_argument('--base', default='submission_v57.csv',
                    help='base submission from pipeline_v57_final.py (id,target)')
    ap.add_argument('--out', default='submission_imputation.csv')
    args = ap.parse_args(argv)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    base = pd.read_csv(args.base)

    lut = build_partner_lookup(train_df)
    test_df = test_df.copy()
    test_df['cs'] = test_df['smiles'].map(canonical)

    # merge base predictions onto test rows (by id)
    m = test_df.merge(base.rename(columns={'target': 'pred'}), on='id', how='left')
    if m['pred'].isna().any():
        raise RuntimeError('base submission does not cover all test ids')

    pred = m['pred'].to_numpy(float).copy()
    tt = m['target_type'].to_numpy(object)
    cs = m['cs'].to_numpy(object)

    # ---- GUARDED IDENTITY: egc = ei - eea, only where BOTH partners exist in train ----
    n_applied = 0
    for i in range(len(m)):
        if tt[i] != 'egc':
            continue
        d = lut.get(cs[i])
        if d is not None and 'ei' in d and 'eea' in d:
            pred[i] = d['ei'] - d['eea']
            n_applied += 1

    out = pd.DataFrame({'id': m['id'].to_numpy(int), 'target': pred})
    if len(out) != len(test_df) or out['id'].duplicated().any():
        raise RuntimeError('submission contract failed')
    out.to_csv(args.out, index=False)
    print(f'imputation variant written: {args.out}')
    print(f'  egc identity applied to {n_applied} rows (egc = ei - eea from exact train partners)')
    print(f'  all other {len(out) - n_applied} rows = V57 base (guarded)')


if __name__ == '__main__':
    main()
