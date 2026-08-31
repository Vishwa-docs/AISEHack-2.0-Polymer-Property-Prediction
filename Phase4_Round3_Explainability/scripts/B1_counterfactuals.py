"""
B1_counterfactuals.py
=====================
EXP-B1 - feature-space counterfactuals via gradient descent on the MLP:
find a sparse minimal feature delta that shifts the predicted property to a
target value; report the chemical interpretation of the top descriptors.
Outputs: counterfactual_directions_<target>.csv, counterfactual_plot_<target>.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from helpers import (SEED, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, save_plot, smoke_n, style_ax)
from A1_train_mlp import MLP


def main():
    seed_all(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    train = load_train()
    N_POLY = smoke_n(20, 5)

    for target in ['tg']:  # Tg is the flagship counterfactual target
        ckpt = torch.load(MLP_DIR / f'{target}_mlp.pt', map_location='cpu', weights_only=False)
        model = MLP(ckpt['input_dim'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        mean_, scale_ = ckpt['scaler_mean'], ckpt['scaler_scale']
        y_mean = float(ckpt['y_mean']); y_std = float(ckpt['y_std'])

        df_t = train[train['target_type'] == target].copy()
        sample = df_t.sample(min(N_POLY, len(df_t)), random_state=SEED)
        pkl = load_proxy(target)
        feat_names = pkl['pipe']['feat_names']

        rows = []
        for _, row in sample.iterrows():
            smi = row['smiles']
            X = rebuild_features([smi], pkl['pipe']).astype(np.float32)
            Xs = torch.tensor(((X - mean_) / scale_).astype(np.float32), requires_grad=True)
            with torch.no_grad():
                pred0_z = model(Xs).item()
            pred0 = pred0_z * y_std + y_mean
            y_target_z = pred0_z + 20.0 / y_std  # '+20 K' in z-space
            # gradient descent in feature space
            x = Xs.clone().detach().requires_grad_(True)   # fresh leaf for SGD
            opt = torch.optim.SGD([x], lr=0.05)
            for it in range(smoke_n(200, 20)):
                opt.zero_grad()
                loss = (model(x) - y_target_z) ** 2 + 1e-4 * (x - Xs).pow(2).sum()
                loss.backward()
                opt.step()
            delta = (x.detach().numpy() - Xs.detach().numpy())[0] * scale_  # unscale to feature units
            order = np.argsort(np.abs(delta))[::-1][:10]
            rows.append({'smiles': smi, 'pred_original': pred0, 'y_target': pred0 + 20.0,
                         'pred_after_cf': model(x).item() * y_std + y_mean,
                         'top_feature': feat_names[order[0]],
                         'top_delta': float(delta[order[0]]),
                         'feature_deltas': ';'.join(
                             f'{feat_names[i]}:{delta[i]:.3f}' for i in order[:5])})
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / f'counterfactual_directions_{target}.csv', index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['pred_original'], df['pred_after_cf'], s=60, alpha=0.8, color='steelblue')
        ax.plot([df['pred_original'].min(), df['pred_original'].max()],
                [df['pred_original'].min(), df['pred_original'].max()], 'k--', alpha=0.5)
        style_ax(ax, f'Counterfactuals (Tg, target +20 K)',
                 'Original prediction (K)', 'Counterfactual prediction (K)')
        save_plot(fig, f'counterfactual_plot_{target}.png')
        print(f'  {target}: {len(df)} counterfactuals; top deltas:');
        for _, r in df.iterrows():
            print(f'    {r[chr(115)+chr(109)+chr(105)+chr(108)+chr(101)+chr(115)][:50]} -> {r[chr(116)+chr(111)+chr(112)+chr(95)+chr(102)+chr(101)+chr(97)+chr(116)+chr(117)+chr(114)+chr(101)]} {r[chr(116)+chr(111)+chr(112)+chr(95)+chr(100)+chr(101)+chr(108)+chr(116)+chr(97)]:+.2f}')
    print(f'B1_counterfactuals.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
