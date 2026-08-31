"""
B2_structural_counterfactuals.py
================================
EXP-B2 - the most falsifiable explainability test: apply known polymer-
chemistry structural modifications and verify the model predicts the
expected directional change.
  add phenyl ring        -> Tg up (rigidity)
  add ether linkage      -> Tg down (flexibility)
  add C=C unsaturation   -> egc/egb down (conjugation)
  add fluorine           -> ei up (electron withdrawal)
Outputs: structural_counterfactuals.csv, structural_counterfactuals_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, RWMol

from helpers import (SEED, OUTPUT_DIR, SMOKE, seed_all, load_train, load_proxy,
                     featurize, predict_ensemble, save_plot, smoke_n, style_ax)


def replace_segment(smi, smarts_from, smarts_to, max_replace=1):
    """Replace up to max_replace occurrences of smarts_from with smarts_to."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        patt = Chem.MolFromSmarts(smarts_from)
        if patt is None:
            return None
        n = len(mol.GetSubstructMatches(patt))
        if n == 0:
            return None
        n = min(n, max_replace)
        rw = RWMol(mol)
        new_mol = AllChem.ReplaceSubstructs(rw, patt, Chem.MolFromSmiles(smarts_to), n)[0]
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception:
        return None


MODS = [
    {'name': 'add_phenyl_rigidity', 'targets': ['tg'], 'expected': 'up',
     'smarts_from': '[CH2][CH2]', 'smarts_to': '[CH2]c1ccccc1[CH2]',
     'desc': 'aliphatic C-C -> benzyl (rigid backbone)'},
    {'name': 'add_ether_flexibility', 'targets': ['tg'], 'expected': 'down',
     'smarts_from': '[CH2][CH2]', 'smarts_to': '[CH2]O[CH2]',
     'desc': 'aliphatic C-C -> ether linkage (flexible)'},
    {'name': 'add_unsaturation', 'targets': ['egc', 'egb'], 'expected': 'down',
     'smarts_from': '[CH2][CH2]', 'smarts_to': '[CH]=[CH]',
     'desc': 'single bond -> double bond (conjugation)'},
    {'name': 'add_fluorine', 'targets': ['ei', 'eea'], 'expected': 'up',
     'smarts_from': '[CH3]', 'smarts_to': 'CF',
     'desc': 'methyl -> fluoromethyl (electron withdrawal)'},
]


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []

    for mod in MODS:
        for target in mod['targets']:
            pkl = load_proxy(target)
            df_t = train[train['target_type'] == target].copy()
            sample = df_t.sample(min(smoke_n(25, 6), len(df_t)), random_state=SEED)
            for _, row in sample.iterrows():
                new_smi = replace_segment(row['smiles'], mod['smarts_from'], mod['smarts_to'])
                if new_smi is None or new_smi == row['smiles']:
                    continue
                X0, _ = featurize([row['smiles']], pipe=pkl['pipe'])
                X1, _ = featurize([new_smi], pipe=pkl['pipe'])
                p0 = predict_ensemble(X0, pkl)[0]
                p1 = predict_ensemble(X1, pkl)[0]
                delta = p1 - p0
                agree = (delta > 0) if mod['expected'] == 'up' else (delta < 0)
                rows.append({'modification': mod['name'], 'desc': mod['desc'],
                             'target': target, 'expected': mod['expected'],
                             'smiles_original': row['smiles'], 'smiles_modified': new_smi,
                             'pred_original': p0, 'pred_modified': p1, 'delta': delta,
                             'direction_agrees': bool(agree)})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'structural_counterfactuals.csv', index=False)
    if len(df) == 0:
        print('no valid structural modifications constructed');
        return

    rate = float(df['direction_agrees'].mean())
    print(f'  direction agreement: {rate:.3f} ({int(df[chr(100)+chr(105)+chr(114)+chr(101)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(97)+chr(103)+chr(114)+chr(101)+chr(101)+chr(115)].sum())}/{len(df)})')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axes, ['pred_original', 'delta']):
        for mod_name in df['modification'].unique():
            sub = df[df['modification'] == mod_name]
            ax.hist(sub[col], bins=15, alpha=0.5, label=mod_name)
        ax.set_title(col.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Structural Counterfactuals - predicted response to known chemistry', fontsize=13)
    save_plot(fig, 'structural_counterfactuals_plot.png')
    print(f'B2_structural_counterfactuals.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
