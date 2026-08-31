#!/usr/bin/env python3
"""
Phase 5 Real-Time Experiment Monitor & Composite Leaderboard
Parses logs/phase5_summary.tsv and individual experiment metrics.json / oracle_scores.json
Displays:
1. Per-experiment history with OOF R² and Oracle R²
2. Best-performing model per individual target
3. Theoretical composite score (unweighted mean of best individual components)
4. Distance to 0.9350 breakthrough target
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']

def main():
    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / "logs"
    exp_dir = base_dir / "experiments"
    summary_file = logs_dir / "phase5_summary.tsv"

    print("=" * 80)
    print("🔬 PHASE 5 EXPERIMENT MONITOR & COMPONENT LEADERBOARD")
    print("=" * 80)

    if not summary_file.exists():
        print("No experiments logged yet.")
        return

    # Read summary TSV
    df_summary = pd.read_csv(summary_file, sep='\t')
    print(f"\n📊 TOTAL COMPLETED EXPERIMENTS: {len(df_summary)}\n")
    print(df_summary.to_string(index=False))

    # Inspect individual experiment scores to find per-target bests
    best_per_target = {t: {'exp_id': 'None', 'r2': -1.0, 'mae': 999.0} for t in TARGETS}
    
    # Iterate through all experiment folders
    for exp_folder in sorted(exp_dir.glob("P5-*")):
        score_file = exp_folder / "oracle_scores.json"
        if score_file.exists():
            try:
                with open(score_file, 'r') as f:
                    scores = json.load(f)
                exp_name = exp_folder.name
                exp_id = exp_name.split('-')[0]
                
                per_tgt = scores.get('per_target', {})
                for t in TARGETS:
                    if t in per_tgt and per_tgt[t].get('r2') is not None:
                        r2_val = per_tgt[t]['r2']
                        mae_val = per_tgt[t].get('mae', 0.0)
                        if r2_val > best_per_target[t]['r2']:
                            best_per_target[t] = {
                                'exp_id': exp_name,
                                'r2': r2_val,
                                'mae': mae_val
                            }
            except Exception as e:
                pass

    print("\n" + "=" * 80)
    print("🏆 PER-TARGET BEST COMPONENT REGISTRY")
    print("=" * 80)
    
    composite_r2_list = []
    print(f"{'Target':<8} | {'Best Exp ID':<35} | {'Oracle R²':<12} | {'MAE':<10}")
    print("-" * 72)
    for t in TARGETS:
        b = best_per_target[t]
        if b['r2'] > -1.0:
            composite_r2_list.append(b['r2'])
            print(f"{t.upper():<8} | {b['exp_id'][:35]:<35} | {b['r2']:<12.4f} | {b['mae']:<10.4f}")
        else:
            print(f"{t.upper():<8} | {'None':<35} | {'N/A':<12} | {'N/A':<10}")

    if composite_r2_list:
        comp_mean = np.mean(composite_r2_list)
        est_priv = comp_mean - 0.011
        print("-" * 72)
        print(f"🔥 THEORETICAL COMPOSITE ORACLE MEAN R²: {comp_mean:.5f}")
        print(f"🎯 ESTIMATED PRIVATE LB:                 {est_priv:.5f} (formula: oracle - 0.011)")
        print(f"🏁 TARGET TO BEAT:                       0.9240 (0.9350 Oracle)")
        gap = 0.9350 - comp_mean
        if gap <= 0:
            print(f"✅ BREAKTHROUGH ACHIEVED! Beats 0.935 by +{(-gap):.4f}")
        else:
            print(f"⏳ Remaining Gap to 0.935 Target:         {gap:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
