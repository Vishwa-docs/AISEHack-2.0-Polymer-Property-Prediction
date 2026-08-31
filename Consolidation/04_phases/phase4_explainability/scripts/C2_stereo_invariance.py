"""
C2_stereo_invariance.py
=======================
EXP-C2 - stereo annotation invariance: predictions must not depend on
stereochemistry markers that bulk polymer properties do not care about.
Outputs: stereo_invariance.csv
"""
import time

import numpy as np
import pandas as pd
from rdkit import Chem

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, featurize, predict_ensemble, smoke_n)


def add_stereo(smi):
    """Try to add a tetrahedral stereo marker at one atom (if chiral possible)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # pick first atom with >= 4 heavy neighbors
    for atom in mol.GetAtoms():
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_neighbors) >= 4:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            try:
                return Chem.MolToSmiles(mol, isomericSmiles=True)
            except Exception:
                return None
    return None


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []
    for target in TARGETS:
        pkl = load_proxy(target)
        df_t = train[train['target_type'] == target].copy()
        sample = df_t.sample(min(smoke_n(50, 10), len(df_t)), random_state=SEED)
        for _, row in sample.iterrows():
            smi_stereo = add_stereo(row['smiles'])
            if smi_stereo is None or smi_stereo == row['smiles']:
                continue
            X, _ = featurize([row['smiles'], smi_stereo], pipe=pkl['pipe'], canonicalize=True)
            preds = predict_ensemble(X, pkl)
            rows.append({'target': target, 'smiles': row['smiles'],
                         'smiles_stereo': smi_stereo,
                         'pred_plain': preds[0], 'pred_stereo': preds[1],
                         'delta': preds[1] - preds[0]})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'stereo_invariance.csv', index=False)
    if len(df):
        print(f'  n pairs with stereo: {len(df)}, mean |delta| = {df[chr(100)+chr(101)+chr(108)+chr(116)+chr(97)].abs().mean():.5f}')
    else:
        print('  no chiral polymers found in sample - report n/a')
    print(f'C2_stereo_invariance.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
