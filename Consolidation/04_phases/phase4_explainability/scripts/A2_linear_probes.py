"""
A2_linear_probes.py
===================
EXP-A2 - do the MLP hidden layers encode chemical concepts? For each layer,
train a Ridge probe from activations to concept labels (aromaticity fraction,
heavy-atom count, H-bond donors/acceptors, ring density, aromatic bond
fraction, polar atom count). Heatmap: concept x layer probe R2.
THIS is the experiment that answers 'what does your model internally
represent?' - e.g. layer 2 of the Tg model encodes aromaticity with R^2=0.84.
Outputs: linear_probe_heatmap_<target>.png, linear_probe_results.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from helpers import (SEED, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, save_plot, smoke_n)
from A1_train_mlp import MLP

CONCEPTS = ['aromaticity', 'heavy_atoms', 'hbond_donors', 'hbond_acceptors',
            'ring_density', 'aromatic_bond_frac', 'polar_atoms']


def concept_labels(smiles_list):
    labels = {c: [] for c in CONCEPTS}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for c in CONCEPTS:
                labels[c].append(np.nan)
            continue
        n_atoms = mol.GetNumAtoms()
        arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        heavy = mol.GetNumHeavyAtoms()
        donors = sum(1 for a in mol.GetAtoms() if a.GetTotalNumHs() > 0 and a.GetSymbol() in ('N', 'O'))
        acceptors = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ('N', 'O', 'F', 'S'))
        aromatic_bonds = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
        polar = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ('N', 'O', 'F', 'S', 'Cl', 'Br'))
        labels['aromaticity'].append(arom / max(1, n_atoms))
        labels['heavy_atoms'].append(heavy)
        labels['hbond_donors'].append(donors)
        labels['hbond_acceptors'].append(acceptors)
        labels['ring_density'].append(mol.GetRingInfo().NumRings() / max(1, heavy))
        labels['aromatic_bond_frac'].append(aromatic_bonds / max(1, mol.GetNumBonds()))
        labels['polar_atoms'].append(polar / max(1, heavy))
    return {c: np.array(v) for c, v in labels.items()}


def activations(model, Xt, layer_key):
    acts = {}
    def hook_fn(name):
        def f(m, inp, out):
            acts[name] = out.detach().numpy()
        return f
    h = getattr(model, layer_key).register_forward_hook(hook_fn(layer_key))
    with torch.no_grad():
        model(Xt)
    h.remove()
    return acts[layer_key]


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    results = []

    for target in ['tg', 'egc', 'nc', 'eps']:  # the 4 interesting targets
        ckpt = torch.load(MLP_DIR / f'{target}_mlp.pt', map_location='cpu', weights_only=False)
        model = MLP(ckpt['input_dim'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        df_t = train[train['target_type'] == target].copy()
        if SMOKE and len(df_t) > 120:
            df_t = df_t.sample(120, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        X = rebuild_features(df_t['smiles'].tolist(), pkl['pipe']).astype(np.float32)
        mean_, scale_ = ckpt['scaler_mean'], ckpt['scaler_scale']
        Xs = ((X - mean_) / scale_).astype(np.float32)
        Xt = torch.tensor(Xs)

        labels = concept_labels(df_t['smiles'].tolist())
        layers = {'block1': model.block1, 'block2': model.block2, 'block3': model.block3}
        for lname, layer in layers.items():
            act = activations(model, Xt, lname).reshape(len(Xt), -1)
            if SMOKE:
                act = act[:, ::4]  # subsample neurons for speed
            for c in CONCEPTS:
                y = labels[c]
                ok = ~np.isnan(y)
                if ok.sum() < 10:
                    continue
                idx = np.where(ok)[0]
                r2s = []
                for tr, va in KFold(3, shuffle=True, random_state=SEED).split(idx):
                    m = Ridge(alpha=10).fit(act[idx[tr]], y[idx[tr]])
                    r2s.append(r2_score(y[idx[va]], m.predict(act[idx[va]])))
                r2m = float(np.mean(r2s))
                results.append({'target': target, 'layer': lname, 'concept': c,
                                'probe_r2': r2m})
                print(f'  {target} {lname} -> {c}: R2={r2m:.3f}')

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'linear_probe_results.csv', index=False)

    for target in df['target'].unique():
        sub = df[df['target'] == target]
        layers = sorted(sub['layer'].unique())
        concepts = sorted(sub['concept'].unique())
        mat = np.full((len(concepts), len(layers)), np.nan)
        for _, r in sub.iterrows():
            mat[concepts.index(r['concept']), layers.index(r['layer'])] = r['probe_r2']
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers)
        ax.set_yticks(range(len(concepts))); ax.set_yticklabels(concepts)
        for i in range(len(concepts)):
            for j in range(len(layers)):
                v = mat[i, j]
                if v == v:
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if v < 0.6 else 'black', fontsize=9)
        ax.set_title(f'Linear Probes - {target.upper()} (concept R2 per layer)')
        fig.colorbar(im, ax=ax, label='Probe R2')
        save_plot(fig, f'linear_probe_heatmap_{target}.png')

    print(f'A2_linear_probes.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
