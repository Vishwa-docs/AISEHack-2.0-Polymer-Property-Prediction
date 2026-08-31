"""
C3_consistency_reg.py
=====================
EXP-C3 - does training with a consistency regularizer
L = L_pred + lambda * ||f(x) - f(T(x))||^2  (T = randomized SMILES)
produce more invariant predictions AND more invariant explanations?
Outputs: consistency_reg_comparison.csv, consistency_reg_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from helpers import (SEED, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all, load_train, load_proxy,
                     rebuild_features, random_smiles, save_plot, smoke_n, style_ax)
from A1_train_mlp import MLP


def run_variant(train_df_t, pipe, scale, lam, n_epochs, device='cpu'):
    X = rebuild_features(train_df_t['smiles'].tolist(), pipe).astype(np.float32)
    y = train_df_t['target'].values.astype(np.float32)
    y_mean = float(np.mean(y)); y_std = float(np.std(y))
    yz = (y - y_mean) / max(y_std, 1e-9)
    mean_, scale_ = scale
    Xs = torch.tensor(((X - mean_) / scale_).astype(np.float32))
    ys = torch.tensor(yz).view(-1, 1)
    # build randomized-SMILES pairs for the consistency term
    Xt_pairs = []
    for smi in train_df_t['smiles']:
        v = random_smiles(smi, 1)[0]
        Xt_pairs.append(v)
    Xt = rebuild_features(Xt_pairs, pipe).astype(np.float32)
    Xts = torch.tensor(((Xt - mean_) / scale_).astype(np.float32))

    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        p = model(Xs)
        loss = lossf(p, ys)
        if lam > 0:
            pt = model(Xts)
            loss = loss + lam * ((p - pt) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        oof = model(Xs).numpy().ravel() * y_std + y_mean
    # invariance: prediction std over 5 randomizations on a small subset
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(train_df_t), min(40, len(train_df_t)), replace=False)
    stds = []
    for i in idx:
        vs = random_smiles(train_df_t.iloc[i]['smiles'], smoke_n(10, 4))
        Xv = rebuild_features(vs, pipe).astype(np.float32)
        Xvs = torch.tensor(((Xv - mean_) / scale_).astype(np.float32))
        with torch.no_grad():
            stds.append(float(model(Xvs).numpy().std()))
    return r2_score(y, oof), float(np.mean(stds))


def main():
    seed_all(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    train = load_train()
    target = 'tg'
    df_t = train[train['target_type'] == target].copy()
    if SMOKE and len(df_t) > 150:
        df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
    pkl = load_proxy(target)
    ckpt = torch.load(MLP_DIR / f'{target}_mlp.pt', map_location='cpu', weights_only=False)
    scale = (ckpt['scaler_mean'], ckpt['scaler_scale'])

    rows = []
    for lam in ([0.0, 0.1, 0.5] if not SMOKE else [0.0, 0.5]):
        r2, std = run_variant(df_t, pkl['pipe'], scale, lam, smoke_n(150, 6))
        rows.append({'lambda': lam, 'tg_oof_r2': r2, 'invariance_std': std})
        print(f'  lambda={lam}: OOF R2={r2:.4f}, invariance std={std:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'consistency_reg_comparison.csv', index=False)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df['lambda'], df['tg_oof_r2'], 'o-', color='steelblue', label='OOF R2')
    ax1.set_xlabel('Consistency weight lambda')
    ax1.set_ylabel('OOF R2', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(df['lambda'], df['invariance_std'], 's--', color='tomato', label='invariance std')
    ax2.set_ylabel('Invariance std', color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Consistency regularization - accuracy vs invariance trade-off')
    fig.tight_layout()
    save_plot(fig, 'consistency_reg_plot.png')
    print(f'C3_consistency_reg.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
