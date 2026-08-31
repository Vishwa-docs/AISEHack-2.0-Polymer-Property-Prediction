"""
A3_activation_patching.py
=========================
EXP-A3 - representation invariance: for a polymer and a randomized SMILES
variant, patch the variant's layer-2 activations with the canonical form's
activations and measure how much the prediction moves. If the internal
representation is already invariant, patching changes nothing.
Outputs: activation_patch_invariance.csv, activation_patch_invariance_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from helpers import (SEED, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, random_smiles, save_plot,
                     smoke_n)
from A1_train_mlp import MLP


def main():
    seed_all(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    train = load_train()
    N_POLY = smoke_n(200, 15)
    results = []

    for target in ['tg', 'egc', 'nc', 'eps']:
        ckpt = torch.load(MLP_DIR / f'{target}_mlp.pt', map_location='cpu', weights_only=False)
        model = MLP(ckpt['input_dim'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        mean_, scale_ = ckpt['scaler_mean'], ckpt['scaler_scale']
        y_mean = float(ckpt['y_mean']); y_std = float(ckpt['y_std'])

        df_t = train[train['target_type'] == target].copy()
        sample = df_t.sample(min(N_POLY, len(df_t)), random_state=SEED)
        pkl = load_proxy(target)
        for _, row in sample.iterrows():
            canon_smi = row['smiles']
            variants = random_smiles(canon_smi, smoke_n(5, 3))
            for var_smi in variants[:smoke_n(5, 3)]:
                X_all = rebuild_features([canon_smi, var_smi], pkl['pipe']).astype(np.float32)
                Xs = ((X_all - mean_) / scale_).astype(np.float32)
                Xt = torch.tensor(Xs)
                with torch.no_grad():
                    pred_canon = model(Xt[0:1]).item() * y_std + y_mean
                    pred_var = model(Xt[1:2]).item() * y_std + y_mean
                act_canon = None
                def grab(m, inp, out):
                    nonlocal act_canon
                    act_canon = out.detach()
                h = model.block2.register_forward_hook(grab)
                with torch.no_grad():
                    model(Xt[0:1])
                h.remove()
                # patched forward: replace block2 output for the variant
                orig = None
                def set_hook(m, inp, out):
                    return act_canon
                h2 = model.block2.register_forward_hook(set_hook)
                with torch.no_grad():
                    pred_patch = model(Xt[1:2]).item() * y_std + y_mean
                h2.remove()
                results.append({
                    'target': target, 'smiles': canon_smi, 'variant': var_smi,
                    'pred_canon': pred_canon, 'pred_var': pred_var,
                    'pred_patch_l2': pred_patch,
                    'delta_pred': abs(pred_var - pred_canon),
                    'delta_after_l2_patch': abs(pred_patch - pred_canon),
                })
        print(f'  {target}: {sum(1 for r in results if r[chr(116)+chr(97)+chr(114)+chr(103)+chr(101)+chr(116)] == target)} pairs')

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'activation_patch_invariance.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].hist(df['delta_pred'], bins=30, color='tomato', alpha=0.8)
    axes[0].set_title('Delta prediction (no patching)')
    axes[0].set_xlabel('Absolute prediction difference')
    axes[1].hist(df['delta_after_l2_patch'], bins=30, color='steelblue', alpha=0.8)
    axes[1].set_title('Delta after patching layer-2 activations')
    axes[1].set_xlabel('Absolute prediction difference')
    for ax in axes:
        ax.set_ylabel('Count'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Activation Patching - Representation Invariance (n={len(df)} pairs)',
                 fontsize=13)
    save_plot(fig, 'activation_patch_invariance_plot.png')
    print(f'A3_activation_patching.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
