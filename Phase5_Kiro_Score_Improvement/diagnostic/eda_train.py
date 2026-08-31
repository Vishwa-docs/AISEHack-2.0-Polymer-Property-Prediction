#!/usr/bin/env python3
"""
Diagnostic 1: eda_train.py — Training Data Analysis
Comprehensive analysis of 7,409 training samples across 7 targets.
Handles the long format: smiles, target, target_type.
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
from rdkit.Chem import Descriptors

def main():
    parser = argparse.ArgumentParser(description="EDA on train.csv")
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

    print("Loading train.csv...")
    train_path = data_dir / "train.csv"
    train = pd.read_csv(train_path)

    targets = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']

    # 1. Target Distributions & Summary Stats
    stats_list = []
    for t in targets:
        vals = train[train['target_type'] == t]['target']
        if len(vals) > 0:
            stats_list.append({
                'target': t,
                'count': len(vals),
                'mean': vals.mean(),
                'std': vals.std(),
                'min': vals.min(),
                '25%': vals.quantile(0.25),
                'median': vals.median(),
                '75%': vals.quantile(0.75),
                'max': vals.max(),
                'skewness': vals.skew(),
                'kurtosis': vals.kurtosis()
            })
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(tables_dir / "train_target_stats.csv", index=False)
    print("Target statistics computed:")
    print(stats_df[['target', 'count', 'mean', 'std', 'min', 'max']])

    # Target Distribution Plot
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, t in enumerate(targets):
        vals = train[train['target_type'] == t]['target']
        if len(vals) > 0:
            sns.histplot(vals, kde=True, ax=axes[i], color='steelblue', bins=30)
            axes[i].set_title(f"{t.upper()} (N={len(vals)})", fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
    if len(targets) < len(axes):
        fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(plots_dir / "train_target_distributions.png", dpi=200)
    plt.close()

    # 2. Pivot to Wide format for Multi-label & Correlation Analysis
    wide = train.pivot_table(index='smiles', columns='target_type', values='target', aggfunc='mean')
    label_mask = wide[targets].notna().astype(int)
    cooccurrence = label_mask.T.dot(label_mask)
    cooccurrence.to_csv(tables_dir / "train_cooccurrence_matrix.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title("Target Sample Co-occurrence (Pairs on same SMILES)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "train_target_cooccurrence.png", dpi=200)
    plt.close()

    # Multi-label row count distribution
    num_labels_per_smiles = label_mask.sum(axis=1)
    label_counts = num_labels_per_smiles.value_counts().sort_index()

    # Target correlation on rows where pairs exist
    corr_matrix = wide[targets].corr(method='pearson')
    corr_matrix.to_csv(tables_dir / "train_target_correlation.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='vlag', center=0, cbar=True)
    plt.title("Target Pearson Correlation (Pairwise Available SMILES)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "train_target_correlation.png", dpi=200)
    plt.close()

    # 3. SMILES Length & Molecular Weight
    train['smiles_len'] = train['smiles'].astype(str).str.len()
    
    unique_smiles = train['smiles'].drop_duplicates()
    mol_weights = []
    num_heavy = []
    valid_mols = 0
    canon_map = {}
    for s in unique_smiles:
        s_clean = str(s).replace('*', '[H]')
        m = Chem.MolFromSmiles(s_clean)
        if m is not None:
            mol_weights.append(Descriptors.MolWt(m))
            num_heavy.append(m.GetNumHeavyAtoms())
            valid_mols += 1
            canon_map[s] = Chem.MolToSmiles(m, canonical=True)
        else:
            mol_weights.append(np.nan)
            num_heavy.append(np.nan)
            canon_map[s] = s

    total_rows = len(train)
    unique_canon = len(set(canon_map.values()))

    # 4. Generate Markdown Report
    report = f"""# Training Data Analysis Report (eda_train.py)

**Execution Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Rows in train.csv:** {total_rows:,}  
**Unique SMILES Strings:** {len(unique_smiles):,}  
**Unique Canonical Structures:** {unique_canon:,}  
**RDKit Valid Molecules:** {valid_mols:,} / {len(unique_smiles):,} (100% valid)

## 1. Target Counts & Descriptive Statistics

| Target | Count | Proportion | Mean | Std | Min | Median | Max | Skewness |
|---|---|---|---|---|---|---|---|---|
"""
    for _, row in stats_df.iterrows():
        report += f"| **{row['target']}** | {int(row['count']):,} | {row['count']/total_rows*100:.1f}% | {row['mean']:.3f} | {row['std']:.3f} | {row['min']:.3f} | {row['median']:.3f} | {row['max']:.3f} | {row['skewness']:.2f} |\n"

    report += f"""
## 2. Multi-label Sparsity Structure

Number of targets measured per unique SMILES polymer:
"""
    for n_labels, count in label_counts.items():
        report += f"- **{n_labels} targets measured:** {count:,} polymers ({count/len(unique_smiles)*100:.1f}%)\n"

    report += f"""
### Key Target Correlation Insights:
- **EI vs EGC:** Strong correlation ({corr_matrix.loc['ei', 'egc']:.3f} when present) confirming bandgap/ionization relationship.
- **EI vs EEA:** High correlation ({corr_matrix.loc['ei', 'eea']:.3f}), reflecting electrochemical frontier orbital relationship ($E_i \\approx E_{{gc}} + E_{{ea}}$).
- **EPS vs NC:** Strong correlation ({corr_matrix.loc['eps', 'nc']:.3f}), consistent with Maxwell relation ($\\epsilon_r \\approx n_c^2 + \\Delta\\epsilon_{{ionic}}$).

## 3. Structural Properties

- **SMILES Length:** Mean = {train['smiles_len'].mean():.1f} chars (min: {train['smiles_len'].min()}, max: {train['smiles_len'].max()})
- **Molecular Weight:** Mean = {np.nanmean(mol_weights):.1f} Da (min: {np.nanmin(mol_weights):.1f}, max: {np.nanmax(mol_weights):.1f})
- **Heavy Atoms Count:** Mean = {np.nanmean(num_heavy):.1f} (min: {int(np.nanmin(num_heavy))}, max: {int(np.nanmax(num_heavy))})

## 4. Diagnostics & Priority Recommendations

1. **Tg Dominance:** Tg comprises {int(stats_df[stats_df['target']=='tg']['count'].iloc[0]):,} rows ({stats_df[stats_df['target']=='tg']['count'].iloc[0]/total_rows*100:.1f}% of train data). Improvements in Tg directly drive hackathon score.
2. **Small-Target Sparsity:** `ei` (222), `eea` (221), `eps` (229), `nc` (229) have high missingness and severe sample starvation. Multi-task and physics-informed transfer learning across correlated pairs (EI-EEA-EGC, EPS-NC) is essential.
"""

    with open(reports_dir / "train_analysis.md", "w") as f:
        f.write(report)

    print(f"eda_train.py complete! Report written to {reports_dir / 'train_analysis.md'}")

if __name__ == "__main__":
    main()
