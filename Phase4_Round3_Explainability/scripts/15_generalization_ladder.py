"""
15_generalization_ladder.py
===========================
R4.1 — the "staircase": R2 under 6 increasingly difficult split regimes.
Outputs: generalization_ladder.csv, generalization_ladder_plot.png
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

REGIMES = ["G0_random", "G1_canonical_group", "G2_scaffold",
           "G3_family", "G4_low_sim_0.6", "G5_ultra_low_0.4"]


def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "invalid"
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return "no_scaffold"


def family_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "unknown"
    from rdkit.Chem import rdMolDescriptors
    return "aromatic" if rdMolDescriptors.CalcNumAromaticRings(mol) > 0 else "aliphatic"


def quick_lgbm(X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMRegressor(n_estimators=smoke_n(200, 30), learning_rate=0.05,
                          num_leaves=31, random_state=SEED, verbosity=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return r2_score(y_va, m.predict(X_va))


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    results = []

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        y = df_t["target"].values.astype(float)
        canon = df_t["smiles"].apply(canonical_smiles).values
        scaffolds = df_t["smiles"].apply(scaffold_of).values
        fam = df_t["smiles"].apply(family_of).values

        fps = [AllChem.GetMorganFingerprintAsBitVect(
                   Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, 1024)
               for s in df_t["smiles"]]
        sim_scores = np.array([
            max(DataStructs.BulkTanimotoSimilarity(
                fps[i], [fps[j] for j in range(len(fps)) if j != i]))
            for i in range(len(fps))])

        for regime in REGIMES:
            r2s = []
            if regime == "G0_random":
                splits = KFold(n_splits=5, shuffle=True, random_state=SEED).split(X)
            elif regime == "G1_canonical_group":
                splits = GroupKFold(n_splits=5).split(X, y, canon)
            elif regime == "G2_scaffold":
                splits = GroupKFold(n_splits=5).split(X, y, scaffolds)
            elif regime == "G3_family":
                splits = GroupKFold(n_splits=2).split(X, y, fam)
            elif regime in ("G4_low_sim_0.6", "G5_ultra_low_0.4"):
                thresh = 0.6 if regime.endswith("0.6") else 0.4
                low_idx = np.where(sim_scores < thresh)[0]
                high_idx = np.where(sim_scores >= thresh)[0]
                splits = [(high_idx, low_idx)] if len(low_idx) >= 5 and len(high_idx) >= 5 else []
            else:
                splits = []
            for tr_idx, va_idx in splits:
                if len(va_idx) < 3:
                    continue
                r2s.append(quick_lgbm(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]))
            mean_r2 = float(np.mean(r2s)) if r2s else np.nan
            results.append({"target": target, "regime": regime,
                            "mean_r2": mean_r2, "n_folds": len(r2s)})
        print(f"  {target}: " + " ".join(
            f"{reg}={next((r['mean_r2'] for r in results if r['target'] == target and r['regime'] == reg), float('nan')):.3f}"
            for reg in REGIMES))

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "generalization_ladder.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGETS)))
    for i, target in enumerate(TARGETS):
        vals = [next((r["mean_r2"] for r in results
                      if r["target"] == target and r["regime"] == reg), np.nan)
                for reg in REGIMES]
        ax.plot(REGIMES, vals, "o-", label=target, color=colors[i], markersize=7)
    style_ax(ax, "Generalization Ladder — R² under increasingly difficult splits",
             "Validation split strategy", "Mean R²")
    ax.legend(loc="lower left", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    save_plot(fig, "generalization_ladder_plot.png")
    print(f"15_generalization_ladder.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
