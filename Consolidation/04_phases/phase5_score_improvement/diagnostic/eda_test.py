#!/usr/bin/env python3
"""
Diagnostic 2: eda_test.py — Test Data Analysis & Overlap / Similarity
Analyzes test.csv (4,940 rows, 4,497 unique SMILES), train/test overlap, and Tanimoto similarity distribution.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

def main():
    parser = argparse.ArgumentParser(description="EDA on test.csv")
    parser.add_argument("--data-dir", default="../data", help="Path to data directory")
    parser.add_argument("--output-dir", default="outputs", help="Path to output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train.csv and test.csv...")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    smiles_col_train = 'smiles' if 'smiles' in train.columns else 'SMILES'
    smiles_col_test = 'smiles' if 'smiles' in test.columns else 'SMILES'

    print("Computing canonical SMILES...")
    train['canon_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) if Chem.MolFromSmiles(s) is not None else s for s in train[smiles_col_train]]
    test['canon_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) if Chem.MolFromSmiles(s) is not None else s for s in test[smiles_col_test]]

    # 1. Basic counts
    n_test = len(test)
    n_test_unique_smiles = test[smiles_col_test].nunique()
    n_test_unique_canon = test['canon_smiles'].nunique()

    # Overlap with train
    train_canon_set = set(train['canon_smiles'])
    test_canon_set = set(test['canon_smiles'])
    overlap_canon = test_canon_set.intersection(train_canon_set)
    overlap_count = len(overlap_canon)

    test_overlap_rows = test[test['canon_smiles'].isin(overlap_canon)]
    print(f"Test total rows: {n_test}, Unique canon: {n_test_unique_canon}, Overlap canon: {overlap_count} ({len(test_overlap_rows)} test rows)")

    # 2. Compute Morgan Fingerprints (radius 2, 2048 bits)
    print("Computing Morgan fingerprints for Tanimoto similarity...")
    train_fps = []
    train_valid_idx = []
    for idx, s in enumerate(train['canon_smiles'].unique()):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048))
            train_valid_idx.append(idx)

    # Unique test molecules
    test_unique_canon = test['canon_smiles'].drop_duplicates().reset_index(drop=True)
    test_fps = []
    test_valid_idx = []
    for idx, s in enumerate(test_unique_canon):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            test_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048))
            test_valid_idx.append(idx)

    print("Computing nearest-neighbor train similarity for each test molecule...")
    max_sims = []
    mean_top3_sims = []
    for tfp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps)
        sims_sorted = sorted(sims, reverse=True)
        max_sims.append(sims_sorted[0])
        mean_top3_sims.append(np.mean(sims_sorted[:3]))

    sim_df = pd.DataFrame({
        'canon_smiles': [test_unique_canon[i] for i in test_valid_idx],
        'max_tanimoto_train': max_sims,
        'mean_top3_tanimoto_train': mean_top3_sims
    })

    # Merge back to full test
    test_merged = test.merge(sim_df, on='canon_smiles', how='left')

    # Binning similarity
    bins = [0.0, 0.3, 0.5, 0.7, 1.01]
    labels = ['< 0.3 (Very Low / OOD)', '0.3 - 0.5 (Low / Novel)', '0.5 - 0.7 (Medium)', '0.7 - 1.0 (High / Scaffold Match)']
    test_merged['sim_bin'] = pd.cut(test_merged['max_tanimoto_train'], bins=bins, labels=labels, right=False)

    bin_counts = test_merged['sim_bin'].value_counts().sort_index()
    bin_summary = pd.DataFrame({
        'Similarity Bin': bin_counts.index,
        'Row Count': bin_counts.values,
        'Percentage': (bin_counts.values / len(test_merged) * 100).round(2)
    })
    bin_summary.to_csv(tables_dir / "test_similarity_bins.csv", index=False)
    test_merged[['id', 'smiles', 'canon_smiles', 'max_tanimoto_train', 'sim_bin']].to_csv(tables_dir / "test_with_similarity.csv", index=False)

    # 3. Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(test_merged['max_tanimoto_train'], bins=40, kde=True, ax=axes[0], color='darkorange')
    axes[0].set_title("Distribution of Max Tanimoto Similarity (Test → Train)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Max Tanimoto Similarity (Morgan r=2, 2048 bits)")
    axes[0].grid(True, alpha=0.3)

    sns.barplot(data=bin_summary, x='Similarity Bin', y='Row Count', ax=axes[1], palette='Oranges')
    axes[1].set_title("Test Rows by Similarity to Training Set", fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha='right')
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height())} ({p.get_height()/len(test)*100:.1f}%)",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                         textcoords='offset points')
    plt.tight_layout()
    plt.savefig(plots_dir / "test_similarity_distribution.png", dpi=200)
    plt.close()

    # 4. Report
    report = f"""# Test Data & Similarity Analysis Report (eda_test.py)

**Execution Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Test Rows:** {n_test:,}  
**Unique SMILES Strings:** {n_test_unique_smiles:,}  
**Unique Canonical Structures:** {n_test_unique_canon:,}  
**Direct Train-Test Structure Overlap:** {overlap_count:,} unique structures ({len(test_overlap_rows):,} test rows = {len(test_overlap_rows)/n_test*100:.2f}%)

## 1. Train-Test Tanimoto Similarity Breakdown

| Similarity Bin | Test Rows | Percentage | Category Characterization |
|---|---|---|---|
"""
    for _, row in bin_summary.iterrows():
        report += f"| **{row['Similarity Bin']}** | {int(row['Row Count']):,} | {row['Percentage']:.1f}% | {'High private-LB risk (OOD)' if '< 0.3' in str(row['Similarity Bin']) else 'Requires robust generalization'} |\n"

    report += f"""
## 2. Key Findings for Modeling & Validation

1. **The 457 Overlap SMILES:** 
   - There are exactly {overlap_count} unique structures appearing in both train and test.
   - **Crucial Rule:** Folds must be grouped by canonical SMILES so that no identical structure is ever in both train and val folds during CV.
2. **Distribution Shift / Low-Similarity Bins:**
   - **{bin_summary[bin_summary['Similarity Bin'].str.contains('< 0.3')]['Row Count'].values[0]:,} rows ({bin_summary[bin_summary['Similarity Bin'].str.contains('< 0.3')]['Percentage'].values[0]}%)** have max Tanimoto similarity < 0.3 to any training molecule.
   - Standard tree models memorizing local Morgan bits collapse on these rows.
   - Self-supervised representation learning from `smile_r3.csv` (5.97M molecules) and latent property regularization directly target this OOD segment.
"""

    with open(reports_dir / "test_analysis.md", "w") as f:
        f.write(report)

    print(f"eda_test.py complete! Report written to {reports_dir / 'test_analysis.md'}")

if __name__ == "__main__":
    main()
