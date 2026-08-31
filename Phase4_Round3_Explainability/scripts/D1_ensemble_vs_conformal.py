"""
D1_ensemble_vs_conformal.py
===========================
EXP-D1 - compare UQ methods on the validation set:
  1) ensemble std (ridge/et/lgbm OOF)
  2) split-conformal intervals
  3) MC dropout on the MLP (if checkpoints exist)
Coverage, mean width and Spearman rho(uncertainty, error) per method.
Outputs: uq_comparison_table.csv, uq_comparison_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from helpers import (SEED, TARGETS, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all,
                     oof_df, save_plot, style_ax)

LEVELS = [0.80, 0.90, 0.95]


def main():
    seed_all(SEED)
    t0 = time.time()
    rows = []
    for target in TARGETS:
        oof = oof_df(target)
        if SMOKE and len(oof) > 150:
            oof = oof.sample(150, random_state=SEED).reset_index(drop=True)
        y = oof['true_value'].values
        pred = oof['oof_ensemble'].values
        err = np.abs(y - pred)
        unc = oof[['oof_ridge', 'oof_et', 'oof_lgbm']].std(axis=1).values
        rho_ens = spearmanr(unc, err)[0]

        # conformal from the same residuals (leave-one-out-ish on this fold set)
        resid = np.abs(y - pred)
        n = len(resid)
        for level in LEVELS:
            q = np.quantile(resid, min(np.ceil((n + 1) * level) / n, 1.0))
            cov = float((err <= q).mean())
            rows.append({'target': target, 'method': 'split_conformal',
                         'level': level, 'coverage': cov, 'mean_width': 2 * q,
                         'rho_err_unc': np.nan})
        rows.append({'target': target, 'method': 'ensemble_std', 'level': np.nan,
                     'coverage': np.nan, 'mean_width': np.nan, 'rho_err_unc': rho_ens})

        # MC dropout (only if the MLP checkpoint exists)
        ckpt_path = MLP_DIR / f'{target}_mlp.pt'
        if ckpt_path.exists() and not SMOKE:
            import torch
            from A1_train_mlp import MLP
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model = MLP(ckpt['input_dim'])
            model.load_state_dict(ckpt['state_dict'])
            model.train()
            y_mean = float(ckpt['y_mean']); y_std = float(ckpt['y_std'])
            from helpers import load_proxy, rebuild_features
            pkl = load_proxy(target)
            X = rebuild_features(oof['smiles'].tolist(), pkl['pipe']).astype(np.float32)
            mean_, scale_ = ckpt['scaler_mean'], ckpt['scaler_scale']
            Xs = torch.tensor(((X - mean_) / scale_).astype(np.float32))
            preds_mc = np.stack([model(Xs).detach().numpy().ravel() for _ in range(20)], axis=1)
            mc_mean = preds_mc.mean(axis=1) * y_std + y_mean
            mc_std = preds_mc.std(axis=1) * y_std
            rho_mc = spearmanr(mc_std, np.abs(y - mc_mean))[0]
            for level in LEVELS:
                q = np.quantile(np.abs(y - mc_mean), min(np.ceil((n + 1) * level) / n, 1.0))
                rows.append({'target': target, 'method': 'mc_dropout', 'level': level,
                             'coverage': float((np.abs(y - mc_mean) <= q).mean()),
                             'mean_width': 2 * q, 'rho_err_unc': rho_mc})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'uq_comparison_table.csv', index=False)

    # plot: coverage calibration per method (mean across targets)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in df['method'].unique():
        sub = df[(df['method'] == method) & df['level'].notna()]
        if len(sub) == 0:
            continue
        g = sub.groupby('level')['coverage'].mean()
        ax.plot(g.index, g.values, 'o-', label=method)
    ax.plot(LEVELS, LEVELS, 'k--', alpha=0.5, label='perfect')
    style_ax(ax, 'UQ comparison - empirical coverage by method', 'Nominal level', 'Empirical coverage')
    ax.legend()
    save_plot(fig, 'uq_comparison_plot.png')
    print(f'D1_ensemble_vs_conformal.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
