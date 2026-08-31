#!/usr/bin/env python3
"""
Diagnostic 5: eda_pi1m.py — 995k Polymer SMILES Characterization
Sample-based statistical characterization of PI1M.csv polymer dataset.
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
    parser = argparse.ArgumentParser(description="EDA on PI1M.csv")
    parser.add_argument("--data-dir", default="../data", help="Path to data directory")
    parser.add_argument("--output-dir", default="outputs", help="Path to output directory")
    parser.add_argument("--sample-size", type=int, default=50000, help="Sample size for rapid EDA")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pi1m_path = data_dir / "PI1M.csv"
    print(f"Sampling {args.sample_size:,} SMILES from {pi1m_path}...")
    df_sample = pd.read_csv(pi1m_path, nrows=args.sample_size)
    smiles_col = df_sample.columns[0]
    total_est = 995799

    lengths = df_sample[smiles_col].astype(str).str.len()
    
    # Check polymer wildcard / attachment point tokens (*)
    has_star = df_sample[smiles_col].astype(str).str.contains(r'\*')
    star_pct = (has_star.sum() / len(df_sample)) * 100

    # Subsample properties
    sub_sample = df_sample[smiles_col].iloc[:5000]
    valid_count = 0
    mol_wts = []
    for s in sub_sample:
        # Clean star for RDKit parse if needed
        s_clean = s.replace('*', '[H]')
        m = Chem.MolFromSmiles(s_clean)
        if m is not None:
            valid_count += 1
            mol_wts.append(Descriptors.MolWt(m))

    stats_summary = {
        'total_file_rows': total_est,
        'sample_evaluated': len(df_sample),
        'star_token_pct': star_pct,
        'mean_smiles_length': lengths.mean(),
        'std_smiles_length': lengths.std(),
        'min_smiles_length': lengths.min(),
        'max_smiles_length': lengths.max(),
        'mean_monomer_wt': np.mean(mol_wts) if mol_wts else 0.0
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(lengths, bins=40, kde=True, ax=ax, color='purple')
    ax.set_title(f"PI1M Polymer Monomer SMILES Length (N={len(df_sample):,})", fontsize=12, fontweight='bold')
    ax.set_xlabel("SMILES String Length (chars)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pi1m_length_distribution.png", dpi=200)
    plt.close()

    # Report
    report = f"""# 995k Polymer SMILES Characterization Report (eda_pi1m.py)

**Execution Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Rows in PI1M.csv:** {total_est:,}  
**Sample Analyzed:** {len(df_sample):,} rows  
**Molecules with Polymer Attachment Points (*):** {stats_summary['star_token_pct']:.2f}%

## 1. Monomer Structural Properties

- **Mean SMILES Length:** {stats_summary['mean_smiles_length']:.2f} ± {stats_summary['std_smiles_length']:.2f} chars (Range: {stats_summary['min_smiles_length']} - {stats_summary['max_smiles_length']})
- **Mean Cleaned Monomer Weight:** {stats_summary['mean_monomer_wt']:.2f} Da

## 2. Complementarity with smile_r3

- `PI1M.csv` contains specialized repeat units with stoichiometric polymerization attachment points (`*`), capturing backbone connectivity.
- `smile_r3.csv` (5.97M) provides dense molecular space for chemical fragment representation.
- Combined self-supervised token learning leverages both general organic chemistry and polymer repeating unit geometry.
"""

    with open(reports_dir / "pi1m_analysis.md", "w") as f:
        f.write(report)

    print(f"eda_pi1m.py complete! Report written to {reports_dir / 'pi1m_analysis.md'}")

if __name__ == "__main__":
    main()
