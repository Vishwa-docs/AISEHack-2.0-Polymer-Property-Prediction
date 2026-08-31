"""
E1_physics_violations.py
========================
EXP-E1 - physics identity analysis on the frozen test predictions:
  eps vs nc^2 + ionic   (ionic estimated from the eps training rows)
  ei  vs egc + eea      (identity uses the model's own per-target outputs)
Report mean absolute violation and the fraction of rows exceeding a
chemistry-plausible threshold.
Outputs: physics_violation_analysis.csv, physics_violation_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (SEED, OUTPUT_DIR, SMOKE, seed_all, load_train, load_test,
                     load_submission, load_proxy, featurize, predict_ensemble,
                     save_plot, smoke_n, style_ax)


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    test = load_test()
    sub = load_submission()

    # predict nc for every test row using the nc proxy
    pkl_nc = load_proxy('nc')
    test_rows = test.merge(sub, on='id')
    X_test, _ = featurize(test_rows['smiles'].tolist(), pipe=pkl_nc['pipe'])
    nc_hat = predict_ensemble(X_test, pkl_nc)
    test_rows['nc_hat'] = nc_hat

    # ionic reference from eps training rows
    df_eps = train[train['target_type'] == 'eps']
    eps_train_mean = float(df_eps['target'].mean())
    nc_train_mean = float(train[train['target_type'] == 'nc']['target'].mean())
    ionic_ref = eps_train_mean - nc_train_mean ** 2

    eps_rows = test_rows[test_rows['target_type'] == 'eps'].copy()
    eps_rows['eps_pred'] = eps_rows['target']
    eps_rows['reconstructed'] = eps_rows['nc_hat'] ** 2 + ionic_ref
    eps_rows['identity_violation'] = np.abs(eps_rows['eps_pred'] - eps_rows['reconstructed'])

    # ei vs egc + eea identity (per-target means as reference since per-row triplets don't exist in test)
    pkl_ei = load_proxy('ei'); pkl_egc = load_proxy('egc'); pkl_eea = load_proxy('eea')
    X_egc, _ = featurize(test_rows['smiles'].tolist(), pipe=pkl_egc['pipe'])
    egc_hat = predict_ensemble(X_egc, pkl_egc)
    X_eea, _ = featurize(test_rows['smiles'].tolist(), pipe=pkl_eea['pipe'])
    eea_hat = predict_ensemble(X_eea, pkl_eea)
    test_rows['egc_hat'] = egc_hat
    test_rows['eea_hat'] = eea_hat
    ei_rows = test_rows[test_rows['target_type'] == 'ei'].copy()
    ei_rows['identity_violation'] = np.abs(ei_rows['target'] - (ei_rows['egc_hat'] + ei_rows['eea_hat']))

    out_rows = [
        {'identity': 'eps ~ nc^2 + ionic', 'n': len(eps_rows),
         'mean_abs_violation': float(eps_rows['identity_violation'].mean()) if len(eps_rows) else np.nan,
         'frac_gt_0_1': float((eps_rows['identity_violation'] > 0.1).mean()) if len(eps_rows) else np.nan},
        {'identity': 'ei ~ egc + eea', 'n': len(ei_rows),
         'mean_abs_violation': float(ei_rows['identity_violation'].mean()) if len(ei_rows) else np.nan,
         'frac_gt_0_1': float((ei_rows['identity_violation'] > 0.1).mean()) if len(ei_rows) else np.nan},
    ]
    pd.DataFrame(out_rows).to_csv(OUTPUT_DIR / 'physics_violation_analysis.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(eps_rows):
        ax.hist(eps_rows['identity_violation'], bins=30, alpha=0.6, label='eps ~ nc^2 + ionic')
    if len(ei_rows):
        ax.hist(ei_rows['identity_violation'], bins=30, alpha=0.6, label='ei ~ egc + eea')
    style_ax(ax, 'Physics identity violations on test predictions', '|violation|', 'Count')
    ax.legend()
    save_plot(fig, 'physics_violation_plot.png')
    print(pd.DataFrame(out_rows).to_string())
    print(f'E1_physics_violations.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
