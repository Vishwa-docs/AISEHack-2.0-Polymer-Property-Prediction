"""
E2_physics_decomp_comparison.py
================================
EXP-E2 - three routes to eps:
  A: direct proxy (eps target)
  B: reconstructed eps = nc^2 + ionic via proxies
  C: average of A and B
Compare OOF R2/MAE, and on the low-similarity validation subset (does the
physics route extrapolate better?).
Outputs: physics_route_comparison.csv, physics_route_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from helpers import (SEED, OUTPUT_DIR, SMOKE, seed_all, load_train, load_proxy,
                     featurize, canonical_smiles, save_plot, smoke_n, style_ax)


def nn_sim(smi, fps):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol if mol else Chem.MolFromSmiles('C'), 2, 1024)
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    return float(max(sims)) if sims else 0.0


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()

    df_eps = train[train['target_type'] == 'eps'].copy()
    df_nc = train[train['target_type'] == 'nc'].copy()
    if SMOKE and len(df_eps) > 100:
        df_eps = df_eps.sample(100, random_state=SEED).reset_index(drop=True)
    if SMOKE and len(df_nc) > 100:
        df_nc = df_nc.sample(100, random_state=SEED).reset_index(drop=True)

    # nc proxy on nc rows
    X_nc, pipe = featurize(df_nc['smiles'].tolist())
    y_nc = df_nc['target'].values.astype(float)
    sc = StandardScaler().fit(X_nc)
    m_nc = Ridge(alpha=100, random_state=SEED).fit(sc.transform(X_nc), y_nc)

    # OOF for route A (direct eps) and route B reconstruction on eps rows
    X_eps, _ = featurize(df_eps['smiles'].tolist(), pipe=pipe)
    y_eps = df_eps['target'].values.astype(float)
    df_eps['canonical'] = df_eps['smiles'].apply(canonical_smiles)
    gkf = GroupKFold(n_splits=3)
    oof_direct = np.zeros(len(df_eps))
    oof_recon = np.zeros(len(df_eps))
    for tr, va in gkf.split(X_eps, y_eps, df_eps['canonical'].values):
        m = Ridge(alpha=100).fit(sc.transform(X_eps[tr]), y_eps[tr])
        oof_direct[va] = m.predict(sc.transform(X_eps[va]))
        nc_va = m_nc.predict(sc.transform(X_eps[va]))
        # ionic model fit on train part
        nc_tr = m_nc.predict(sc.transform(X_eps[tr]))
        ionic_tr = y_eps[tr] - nc_tr ** 2
        m_ion = Ridge(alpha=100).fit(sc.transform(X_eps[tr]), ionic_tr)
        ionic_va = m_ion.predict(sc.transform(X_eps[va]))
        oof_recon[va] = nc_va ** 2 + ionic_va
    oof_avg = 0.5 * (oof_direct + oof_recon)

    fps_nc = [AllChem.GetMorganFingerprintAsBitVect(
               Chem.MolFromSmiles(s) or Chem.MolFromSmiles('C'), 2, 1024) for s in df_nc['smiles']]
    sims = df_eps['smiles'].apply(lambda s: nn_sim(s, fps_nc)).values
    low = sims < 0.5

    rows = []
    for name, preds in (('A_direct', oof_direct), ('B_reconstructed', oof_recon), ('C_ensemble', oof_avg)):
        rows.append({'route': name, 'r2_all': r2_score(y_eps, preds),
                     'mae_all': mean_absolute_error(y_eps, preds),
                     'r2_low_sim': r2_score(y_eps[low], preds[low]) if low.sum() >= 5 else np.nan,
                     'mae_low_sim': mean_absolute_error(y_eps[low], preds[low]) if low.sum() >= 5 else np.nan,
                     'n_low_sim': int(low.sum())})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'physics_route_comparison.csv', index=False)
    print(pd.DataFrame(rows).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r['route'] for r in rows]
    r2s = [r['r2_all'] for r in rows]
    r2low = [r['r2_low_sim'] for r in rows]
    x = np.arange(len(names))
    ax.bar(x - 0.18, r2s, 0.36, label='all validation', color='steelblue')
    ax.bar(x + 0.18, r2low, 0.36, label='low-sim subset', color='tomato')
    ax.set_xticks(x); ax.set_xticklabels(names)
    style_ax(ax, 'Physics-decomposed eps - route comparison', 'Route', 'R2')
    ax.legend()
    save_plot(fig, 'physics_route_plot.png')
    print(f'E2_physics_decomp_comparison.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
