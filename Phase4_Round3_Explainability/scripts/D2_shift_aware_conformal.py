"""
D2_shift_aware_conformal.py
===========================
EXP-D2 - covariate-shift correction for conformal prediction: reweight
calibration residuals by a k-NN density ratio on Morgan fingerprints
(calibration vs test). Compare standard vs shift-aware coverage.
Outputs: shift_aware_conformal.csv
"""
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_test,
                     oof_df, smoke_n)


def fp_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles('C')
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)


def nn_density_ratio(query_fps, ref_a, ref_b, k=20):
    """log density ratio of query under ref_a vs ref_b (k-NN distances)."""
    out = []
    for q in query_fps:
        da = sorted(DataStructs.BulkTanimotoSimilarity(q, ref_a), reverse=True)[:k]
        db = sorted(DataStructs.BulkTanimotoSimilarity(q, ref_b), reverse=True)[:k]
        # use mean top-k similarity as density proxy
        out.append(float(np.mean(da)) / max(1e-9, float(np.mean(db))))
    return np.array(out)


def main():
    seed_all(SEED)
    t0 = time.time()
    test = load_test()
    test_fps = [fp_of(s) for s in test['smiles']]
    rows = []
    for target in TARGETS:
        oof = oof_df(target)
        if SMOKE and len(oof) > 150:
            oof = oof.sample(150, random_state=SEED).reset_index(drop=True)
        cal = oof.iloc[:int(0.8 * len(oof))]
        val = oof.iloc[int(0.8 * len(oof)):]
        err = np.abs(cal['true_value'] - cal['oof_ensemble']).values
        cal_fps = [fp_of(s) for s in cal['smiles']]
        val_fps = [fp_of(s) for s in val['smiles']]
        # density ratio of calibration vs test (weights >1 where cal is rarer than test)
        w = nn_density_ratio(cal_fps, cal_fps, test_fps, k=10)
        w = w / w.mean()
        val_err = np.abs(val['true_value'] - val['oof_ensemble']).values
        for level in [0.80, 0.90, 0.95]:
            # standard
            q_std = np.quantile(err, min(np.ceil((len(err) + 1) * level) / len(err), 1.0))
            cov_std = float((val_err <= q_std).mean())
            # weighted (shift-aware) quantile
            order = np.argsort(err)
            cw = np.cumsum(w[order]) / w.sum()
            idx = np.searchsorted(cw, level)
            q_w = err[order[min(idx, len(err) - 1)]]
            cov_w = float((val_err <= q_w).mean())
            rows.append({'target': target, 'level': level, 'coverage_standard': cov_std,
                         'coverage_shift_aware': cov_w, 'n_cal': len(cal), 'n_val': len(val)})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'shift_aware_conformal.csv', index=False)
    print(df.groupby('level')[['coverage_standard', 'coverage_shift_aware']].mean().to_string())
    print(f'D2_shift_aware_conformal.py DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
