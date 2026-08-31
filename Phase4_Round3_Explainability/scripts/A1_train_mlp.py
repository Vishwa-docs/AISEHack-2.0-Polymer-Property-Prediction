"""
A1_train_mlp.py
===============
EXP-A1 - train small per-target MLPs (torch, CPU) from scratch on the proxy
feature stack. These are the vessels for mechanistic interpretability
(linear probes, activation patching, causal tracing).
Saves mlp_checkpoints/{target}_mlp.pt + mlp_proxy_scores.csv
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from helpers import (SEED, TARGETS, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all,
                     load_train, load_proxy, rebuild_features, smoke_n,
                     canonical_smiles)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=512):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden),
                                    nn.ReLU(), nn.Dropout(0.2))
        self.block2 = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2),
                                    nn.ReLU(), nn.Dropout(0.2))
        self.block3 = nn.Sequential(nn.Linear(d_hidden // 2, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.block3(self.block2(self.block1(x))))


def main():
    seed_all(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    t0 = time.time()
    train = load_train()
    rows = []

    for target in TARGETS:
        df_t = train[train['target_type'] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        X = rebuild_features(df_t['smiles'].tolist(), pkl['pipe']).astype(np.float32)
        y = df_t['target'].values.astype(np.float32)
        df_t['canonical'] = df_t['smiles'].apply(canonical_smiles)
        # z-score the target so the MLP head (default init ~O(1)) can reach it
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        yz = (y - y_mean) / max(y_std, 1e-9)

        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        gkf = GroupKFold(n_splits=3)
        oof = np.zeros(len(df_t))
        scaler_all = StandardScaler().fit(X)
        Xs = scaler_all.transform(X).astype(np.float32)

        for tr, va in gkf.split(Xs, yz, df_t['canonical'].values):
            Xtr = torch.tensor(Xs[tr]); ytr = torch.tensor(yz[tr]).view(-1, 1)
            Xva = torch.tensor(Xs[va])
            model = MLP(X.shape[1])
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = nn.MSELoss()
            n_epochs = smoke_n(200, 8)
            for ep in range(n_epochs):
                model.train()
                opt.zero_grad()
                loss = lossf(model(Xtr), ytr)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                oof[va] = model(Xva).numpy().ravel()
        oof = oof * y_std + y_mean
        r2 = r2_score(y, oof)
        rows.append({'target': target, 'n': len(df_t), 'oof_r2': r2,
                     'input_dim': X.shape[1]})
        print(f'  {target}: MLP OOF R2 = {r2:.4f} (n={len(df_t)}, dim={X.shape[1]})')

        # save final model trained on all data (for probes/patching)
        Xall = torch.tensor(Xs)
        yall = torch.tensor(yz).view(-1, 1)
        final = MLP(X.shape[1])
        opt = torch.optim.Adam(final.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        n_epochs = smoke_n(150, 6)
        for ep in range(n_epochs):
            final.train()
            opt.zero_grad()
            loss = lossf(final(Xall), yall)
            loss.backward()
            opt.step()
        torch.save({'state_dict': final.state_dict(),
                    'input_dim': X.shape[1],
                    'scaler_mean': scaler_all.mean_, 'scaler_scale': scaler_all.scale_,
                    'y_mean': y_mean, 'y_std': y_std,
                    'target': target}, MLP_DIR / f'{target}_mlp.pt')

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'mlp_proxy_scores.csv', index=False)
    print(f'A1_train_mlp.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
