#!/usr/bin/env python3
"""
inference.py — predict a polymer property for a (SMILES, target_type) row WITHOUT retraining.

Loads weights/polymer_weights.joblib and resolves each query through a priority ladder:

  1. v57_exact   — the row's id (or canonical SMILES + target_type) is in the official
                   test set: return the EXACT V57 full-pipeline prediction from the cache.
  2. identity    — target is `egc` and the polymer's `ei` and `eea` are both in train:
                   return egc = ei - eea (exact partner reconstruction).
  3. train_label — the exact (polymer, target) was measured in train: return that label.
  4. base_model  — novel polymer: compact LightGBM on Morgan-counts+descriptors
                   (approximate; see base_oof_r2 in the bundle for its quality).

USAGE
  Single row:
    python inference.py --smiles "*CCc1ccccc1*" --target tg
    python inference.py --id 1 --smiles "*CCCCCCCCc1nc2cc3sc(*)nc3cc2s1" --target egc

  Batch (a test.csv-style file with columns smiles,target_type[,id]):
    python inference.py --infile Dataset/test.csv --out predictions.csv
"""
import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from featurize import canonical, featurize_one

TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']


class Predictor:
    def __init__(self, weights_path='weights/polymer_weights.joblib'):
        self.b = joblib.load(weights_path)
        for k in ('partner_lut', 'v57_by_id', 'v57_by_key', 'base_models'):
            if k not in self.b:
                raise RuntimeError(f'weights bundle missing {k}')

    def predict(self, smiles, target_type, row_id=None):
        """Return (value, source)."""
        if target_type not in TARGETS:
            raise ValueError(f'unknown target_type {target_type!r}; expected one of {TARGETS}')
        b = self.b
        # 1. exact V57 by id
        if row_id is not None:
            try:
                rid = int(row_id)
                if rid in b['v57_by_id']:
                    return b['v57_by_id'][rid], 'v57_exact(id)'
            except (ValueError, TypeError):
                pass
        cs = canonical(smiles)
        # 1b. exact V57 by canonical smiles + target
        if cs is not None and (cs, target_type) in b['v57_by_key']:
            return b['v57_by_key'][(cs, target_type)], 'v57_exact(smiles)'
        # 2. egc identity from exact train partners
        d = b['partner_lut'].get(cs) if cs is not None else None
        if target_type == 'egc' and d is not None and 'ei' in d and 'eea' in d:
            return d['ei'] - d['eea'], 'identity(egc=ei-eea)'
        # 3. exact train label
        if d is not None and target_type in d:
            return d[target_type], 'train_label'
        # 4. compact base model
        x = featurize_one(smiles).reshape(1, -1)
        val = float(b['base_models'][target_type].predict(x)[0])
        return val, 'base_model'


def main(argv=None):
    ap = argparse.ArgumentParser(description='polymer property inference (no retraining)')
    ap.add_argument('--weights', default='weights/polymer_weights.joblib')
    ap.add_argument('--smiles')
    ap.add_argument('--target')
    ap.add_argument('--id', default=None)
    ap.add_argument('--infile', help='CSV with columns smiles,target_type[,id]')
    ap.add_argument('--out', default='predictions.csv')
    args = ap.parse_args(argv)

    pred = Predictor(args.weights)

    if args.infile:
        df = pd.read_csv(args.infile)
        if 'smiles' not in df or 'target_type' not in df:
            sys.exit('infile needs columns: smiles, target_type (id optional)')
        vals, srcs = [], []
        for r in df.itertuples():
            rid = getattr(r, 'id', None)
            v, s = pred.predict(r.smiles, r.target_type, rid)
            vals.append(v); srcs.append(s)
        out = pd.DataFrame({'id': df['id'] if 'id' in df else np.arange(1, len(df) + 1),
                            'target': vals})
        out.to_csv(args.out, index=False)
        from collections import Counter
        print(f'wrote {len(out)} predictions -> {args.out}')
        print('sources:', dict(Counter(srcs)))
        return

    if args.smiles and args.target:
        v, s = pred.predict(args.smiles, args.target, args.id)
        print(f'{args.target}  =  {v:.6f}   [{s}]')
        return

    # demo
    print('Demo (pass --smiles/--target or --infile for real use):')
    for smi, tgt in [('*CCc1ccccc1*', 'tg'), ('*CCCCCCCCc1nc2cc3sc(*)nc3cc2s1', 'egc')]:
        v, s = pred.predict(smi, tgt)
        print(f'  {tgt:4s} {smi:40s} -> {v:.4f} [{s}]')


if __name__ == '__main__':
    main()
