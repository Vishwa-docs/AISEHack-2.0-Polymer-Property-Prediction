"""
10_cv_validation.py
===================
R3.1 — structured validation beyond random CV. Four regimes per target:
  G0 random 5-fold, G1 canonical-group 5-fold, G2 Murcko-scaffold 5-fold,
  G3 low-similarity holdout (Tanimoto < 0.4).
Quick LightGBM fit per regime. Outputs: cv_validation_table.csv,
cv_validation_barplot.png
"""
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, canonical_smiles, save_plot,
                     smoke_n, style_ax)


def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "invalid"
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return "no_scaffold"


def quick_lgbm_fit(X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMRegressor(n_estimators=smoke_n(200, 30), learning_rate=0.05,
                          num_leaves=31, random_state=SEED, verbosity=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return r2_score(y_va, m.predict(X_va))


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    rows = []

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        y = df_t["target"].values.astype(float)
        canon = df_t["smiles"].apply(canonical_smiles).values
        scaffolds = df_t["smiles"].apply(scaffold_of).values

        # G3 similarity holdout
        fps = [AllChem.GetMorganFingerprintAsBitVect(
                   Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, 1024)
               for s in df_t["smiles"]]
        sim_scores = np.array([
            max(DataStructs.BulkTanimotoSimilarity(
                fps[i], [fps[j] for j in range(len(fps)) if j != i]))
            for i in range(len(fps))])
        low_idx = np.where(sim_scores < 0.4)[0]
        high_idx = np.where(sim_scores >= 0.4)[0]

        regimes = {}
        if len(low_idx) >= 5 and len(high_idx) >= 5:
            r2s = quick_lgbm_fit(X[high_idx], y[high_idx], X[low_idx], y[low_idx])
            regimes["G3_low_sim_0.4"] = r2s
        else:
            regimes["G3_low_sim_0.4"] = np.nan

        for name, splitter, grp in (
            ("G0_random", KFold(n_splits=5, shuffle=True, random_state=SEED), None),
            ("G1_canonical_group", GroupKFold(n_splits=5), canon),
            ("G2_scaffold", GroupKFold(n_splits=5), scaffolds),
        ):
            r2s = []
            for tr_idx, va_idx in splitter.split(X, y, grp):
                if len(va_idx) < 3:
                    continue
                r2s.append(quick_lgbm_fit(X[tr_idx], y[tr_idx],
                                          X[va_idx], y[va_idx]))
            regimes[name] = float(np.mean(r2s)) if r2s else np.nan

        for name, r2 in regimes.items():
            rows.append({"target": target, "regime": name, "mean_r2": r2})
        print(f"  {target}: " + " ".join(f"{k}={v:.3f}" for k, v in regimes.items()))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "cv_validation_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    order = ["G0_random", "G1_canonical_group", "G2_scaffold", "G3_low_sim_0.4"]
    width = 0.11
    for i, target in enumerate(TARGETS):
        vals = []
        for reg in order:
            v = df[(df.target == target) & (df.regime == reg)]["mean_r2"]
            vals.append(v.values[0] if len(v) else np.nan)
        ax.bar(np.arange(len(order)) + i * width, vals, width, label=target)
    ax.set_xticks(np.arange(len(order)) + 3 * width)
    ax.set_xticklabels(order, rotation=15)
    style_ax(ax, "Structured CV — R² under 4 split strategies",
             "Validation split strategy", "Mean R²")
    ax.legend(ncol=4, fontsize=8)
    save_plot(fig, "cv_validation_barplot.png")
    print(f"10_cv_validation.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
