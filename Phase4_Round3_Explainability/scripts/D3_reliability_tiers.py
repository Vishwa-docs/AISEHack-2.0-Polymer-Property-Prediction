"""
D3_reliability_tiers.py
=======================
EXP-D3 - reliability tiers from AD similarity + ensemble uncertainty.
Tier1 (high): sim > 0.7 and unc < 0.5 sigma; Tier2: sim > 0.5 and unc < 1.0 sigma;
else Tier3. Verify tier MAE ordering on validation; assign tiers to 4,940 test rows.
Outputs: reliability_tiers_validation.csv, reliability_tiers_test.csv,
         reliability_2d_map.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import mean_absolute_error

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_test, oof_df, save_plot, smoke_n, style_ax)


def fp_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles('C')
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)


def nn_sim(smi, fps):
    fp = fp_of(smi)
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    return float(max(sims)) if sims else 0.0


def tier_of(sim, unc, tstd):
    if sim > 0.7 and unc < 0.5 * tstd:
        return 1
    if sim > 0.5 and unc < 1.0 * tstd:
        return 2
    return 3


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    test = load_test()
    v_rows = []
    test_rows = []

    for target in TARGETS:
        df_t = train[train['target_type'] == target]
        fps_train = [fp_of(s) for s in df_t['smiles']]
        oof = oof_df(target)
        if SMOKE and len(oof) > 150:
            oof = oof.sample(150, random_state=SEED).reset_index(drop=True)
        unc = oof[['oof_ridge', 'oof_et', 'oof_lgbm']].std(axis=1).values
        err = np.abs(oof['true_value'] - oof['oof_ensemble']).values
        tstd = float(df_t['target'].std())
        sims = oof['smiles'].apply(lambda s: nn_sim(s, fps_train)).values
        tiers = [tier_of(s, u, tstd) for s, u in zip(sims, unc)]
        oof['ad_sim'] = sims; oof['unc'] = unc; oof['tier'] = tiers
        for tier in (1, 2, 3):
            sub = oof[oof['tier'] == tier]
            v_rows.append({'target': target, 'tier': tier, 'n': len(sub),
                           'mae': mean_absolute_error(sub['true_value'], sub['oof_ensemble']) if len(sub) else np.nan})
        # test rows
        test_t = test[test['target_type'] == target]
        from helpers import load_proxy, rebuild_features, predict_ensemble
        pkl = load_proxy(target)
        for _, r in test_t.iterrows():
            sim = nn_sim(r['smiles'], fps_train)
            X = rebuild_features([r['smiles']], pkl['pipe'])
            # uncertainty from proxy ensemble members
            scaler, mr = pkl['models']['ridge'][-1]
            me = pkl['models']['et'][-1]
            ml = pkl['models']['lgbm'][-1]
            Xs = scaler.transform(X.reshape(1, -1))
            u = float(np.std([mr.predict(Xs)[0], me.predict(X.reshape(1, -1))[0], ml.predict(X.reshape(1, -1))[0]]))
            test_rows.append({'id': int(r['id']), 'target_type': target,
                              'ad_sim': sim, 'unc': u, 'tier': tier_of(sim, u, tstd)})
        print(f'  {target}: tiers {pd.Series(tiers).value_counts().to_dict()}')

    pd.DataFrame(v_rows).to_csv(OUTPUT_DIR / 'reliability_tiers_validation.csv', index=False)
    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / 'reliability_tiers_test.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    dfv = pd.DataFrame(v_rows).dropna(subset=['mae'])
    for tier in (1, 2, 3):
        sub = dfv[dfv['tier'] == tier]
        if len(sub):
            ax.bar(tier, sub['mae'].mean(), width=0.6, label=f'Tier {tier}', alpha=0.8)
    style_ax(ax, 'Reliability tiers - validation MAE', 'Tier (1 = highest confidence)', 'Mean MAE')
    ax.legend()
    save_plot(fig, 'reliability_2d_map.png')
    print(f'D3_reliability_tiers.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
