#!/usr/bin/env python3
"""
Diagnostic 7: eda_chemical_families.py — Polymer Chemical Family Classification
Classifies train and test polymers into functional classes (polyolefins, polyesters, polyamides,
polyimides, polyethers, vinyl, halogenated, conjugated/aromatic) using SMARTS substructure definitions.
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

FAMILIES = {
    'Polyester': '[#6](=O)[#8][#6]',
    'Polyamide/Imide': '[#6](=O)[#7]',
    'Polyether': '[#6][#8][#6]',
    'Polyurethane/Urea': '[#7][#6](=O)[#8,#7]',
    'Halogenated (F, Cl, Br)': '[F,Cl,Br]',
    'Aromatic/Conjugated': 'a',
    'Sulfur-containing': '[#16]',
    'Silicon-containing': '[#14]',
    'Pure Hydrocarbon/Polyolefin': '[#6;!$([#6]~[!#6])]'
}

def classify_smiles(s, patterns):
    s_clean = str(s).replace('*', '[H]')
    m = Chem.MolFromSmiles(s_clean)
    if m is None:
        return 'Invalid'
    
    matched = []
    for name, pat in patterns.items():
        if m.HasSubstructMatch(pat):
            matched.append(name)
    if not matched:
        return 'Other/Unclassified'
    return '; '.join(matched)

def main():
    parser = argparse.ArgumentParser(description="Chemical family classification for train/test")
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

    print("Compiling SMARTS patterns...")
    patterns = {k: Chem.MolFromSmarts(v) for k, v in FAMILIES.items()}

    print("Loading train.csv and test.csv...")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    print("Classifying train molecules...")
    train['families'] = [classify_smiles(s, patterns) for s in train['smiles']]
    print("Classifying test molecules...")
    test['families'] = [classify_smiles(s, patterns) for s in test['smiles']]

    # Breakdown of individual family flags
    train_family_matrix = pd.DataFrame(index=train.index)
    test_family_matrix = pd.DataFrame(index=test.index)

    for fam in FAMILIES.keys():
        train_family_matrix[fam] = train['families'].str.contains(fam, regex=False).astype(int)
        test_family_matrix[fam] = test['families'].str.contains(fam, regex=False).astype(int)

    train_counts = train_family_matrix.sum()
    test_counts = test_family_matrix.sum()

    comparison_df = pd.DataFrame({
        'Chemical Family': FAMILIES.keys(),
        'Train Count': [train_counts[f] for f in FAMILIES.keys()],
        'Train Pct': [(train_counts[f]/len(train)*100).round(1) for f in FAMILIES.keys()],
        'Test Count': [test_counts[f] for f in FAMILIES.keys()],
        'Test Pct': [(test_counts[f]/len(test)*100).round(1) for f in FAMILIES.keys()]
    })
    comparison_df.to_csv(tables_dir / "chemical_family_comparison.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width/2, comparison_df['Train Pct'], width, label='Train %', color='steelblue')
    ax.bar(x + width/2, comparison_df['Test Pct'], width, label='Test %', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Chemical Family'], rotation=25, ha='right')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Polymer Chemical Family Distribution (Train vs Test)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "chemical_family_distribution.png", dpi=200)
    plt.close()

    # Report
    report = f"""# Polymer Chemical Family Classification Report (eda_chemical_families.py)

**Execution Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  

## 1. Family Distribution (Train vs Test)

| Chemical Family | Train Count | Train % | Test Count | Test % | Shift / Hazard Ratio |
|---|---|---|---|---|---|
"""
    for _, row in comparison_df.iterrows():
        ratio = row['Test Pct'] / max(row['Train Pct'], 0.1)
        hazard = "⚠️ High Overrepresentation in Test" if ratio > 1.3 else ("Balanced" if 0.7 <= ratio <= 1.3 else "Underrepresented in Test")
        report += f"| **{row['Chemical Family']}** | {int(row['Train Count']):,} | {row['Train Pct']}% | {int(row['Test Count']):,} | {row['Test Pct']}% | {hazard} |\n"

    report += f"""
## 2. Key Observations

1. **Aromatic / Conjugated Backbone Dominance:** Both train ({comparison_df.loc[comparison_df['Chemical Family']=='Aromatic/Conjugated', 'Train Pct'].values[0]}%) and test ({comparison_df.loc[comparison_df['Chemical Family']=='Aromatic/Conjugated', 'Test Pct'].values[0]}%) are heavily aromatic polymers (polyphenylene, polyimide, polyetherketone cores).
2. **Heteroatom Substructures:** Polyesters and polyamides represent key functional classes with hydrogen-bonding and polar contributions critical to Tg and dielectric constant ($\\epsilon$).
"""

    with open(reports_dir / "chemical_families_analysis.md", "w") as f:
        f.write(report)

    print(f"eda_chemical_families.py complete! Report written to {reports_dir / 'chemical_families_analysis.md'}")

if __name__ == "__main__":
    main()
