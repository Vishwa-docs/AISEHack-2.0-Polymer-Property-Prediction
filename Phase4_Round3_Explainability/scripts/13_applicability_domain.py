"""
13_applicability_domain.py
==========================
R3.4 — reliability vs structural novelty. Nearest-train Tanimoto (Morgan FP)
per test polymer; bin the validation set by similarity and report MAE/R2.
Outputs: ad_analysis_table.csv, ad_analysis_plot.png, ad_test_similarity.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_test, load_proxy, rebuild_features, canonical_smiles,
                     oof_df, save_plot, style_ax, smoke_n)
from sklearn.metrics import r2_score, mean_absolute_error

BINS = [(0.9, 1.01, "ge_0.9"), (0.7, 0.9, "0.7-0.9"),
        (0.5, 0.7, "0.5-0.7"), (0.0, 0.5, "lt_0.5")]


def nn_tanimoto(smi, fps_train):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
    return float(max(sims)) if sims else 0.0


def tier_of(sim):
    for lo, hi, name in BINS:
        if lo <= sim < hi:
            return name
    return "lt_0.5"


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    test = load_test()

    # ---- per-target AD analysis on validation OOF ----
    rows = []
    for target in TARGETS:
        df_t = train[train["target_type"] == target]
        oof = oof_df(target)
        n = len(oof)
        if SMOKE and n > 150:
            oof = oof.sample(150, random_state=SEED).reset_index(drop=True)
        # nearest-train similarity for each OOF row (exclude the row's own fingerprint)
        fps_train = [AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, 1024)
            for s in df_t["smiles"]]
        df_t_reset = df_t.reset_index(drop=True)
        own_idx = {smi: i for i, smi in enumerate(df_t_reset["smiles"])}
        sims = np.zeros(len(oof))
        for k, smi in enumerate(oof["smiles"]):
            fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smi) or Chem.MolFromSmiles("C"), 2, 1024)
            j = own_idx.get(smi)
            others = [fps_train[i] for i in range(len(fps_train)) if i != j]
            if not others:
                sims[k] = 1.0
            else:
                sims[k] = max(DataStructs.BulkTanimotoSimilarity(fp, others))
        oof["nn_sim"] = sims
        oof["ad_tier"] = [tier_of(s) for s in sims]

        for lo, hi, name in BINS:
            sub = oof[(oof["nn_sim"] >= lo) & (oof["nn_sim"] < hi)]
            if len(sub) < 5:
                rows.append({"target": target, "ad_bin": name, "n": len(sub),
                             "mae": np.nan, "r2": np.nan})
                continue
            mae = mean_absolute_error(sub["true_value"], sub["oof_ensemble"])
            r2 = r2_score(sub["true_value"], sub["oof_ensemble"])
            rows.append({"target": target, "ad_bin": name, "n": len(sub),
                         "mae": mae, "r2": r2})
        print(f"  {target}: " + " ".join(
            f"{name}={next((r['r2'] for r in rows if r['target'] == target and r['ad_bin'] == name), float('nan')):.3f}"
            for _, _, name in BINS))

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ad_analysis_table.csv", index=False)

    # plot: R2 vs similarity bin (mean across targets)
    order = [n for _, _, n in BINS]
    means = []
    for name in order:
        sub = pd.DataFrame(rows)[pd.DataFrame(rows)["ad_bin"] == name]
        means.append(sub["r2"].mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(order, means, "o-", color="steelblue", markersize=9)
    style_ax(ax, "Applicability Domain — R² vs nearest-train similarity",
             "Nearest-train Tanimoto bin", "Mean R² across targets")
    save_plot(fig, "ad_analysis_plot.png")

    # ---- per-test-row nearest-train similarity (all targets' train) ----
    all_fps = []
    for smi in train["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        all_fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol if mol else Chem.MolFromSmiles("C"), 2, 1024))
    test_sims = []
    for smi in test["smiles"]:
        test_sims.append(nn_tanimoto(smi, all_fps))
    ad_test = pd.DataFrame({
        "id": test["id"].values,
        "nearest_train_tanimoto": test_sims,
        "ad_confidence_tier": [tier_of(s) for s in test_sims],
    })
    ad_test.to_csv(OUTPUT_DIR / "ad_test_similarity.csv", index=False)
    print(f"13_applicability_domain.py DONE in {time.time() - t0:.0f}s — "
          f"test tier distribution:\n{ad_test['ad_confidence_tier'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
