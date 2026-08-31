#!/bin/bash
# Scaffold a new experiment directory

set -e

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <exp_id> <slug>"
  echo "Example: $0 P5-016 svd-100k-tfidf"
  exit 1
fi

EXP_ID="$1"
SLUG="$2"
EXP_DATE=$(date +%Y%m%d-%H%M)
EXP_DIR="experiments/${EXP_ID}-${EXP_DATE}-${SLUG}"

echo "Creating experiment directory: ${EXP_DIR}"
mkdir -p "${EXP_DIR}"

# Create placeholder run_experiment.py
cat > "${EXP_DIR}/run_experiment.py" <<'EOF'
#!/usr/bin/env python3
"""
Experiment: [FILL IN EXPERIMENT NAME]
Goal: [FILL IN GOAL]
Expected: [FILL IN EXPECTED GAIN]

See PLAN.md for detailed specification.
"""

import argparse
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

def main(args):
    print(f"Running experiment: [EXPERIMENT NAME]")
    
    # 1. Load data
    DATA_DIR = Path(args.data_dir)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    if not args.smoke:
        verify_data_hashes(DATA_DIR)
    
    # 2. TODO: Implement experiment logic
    # - Feature engineering
    # - Model training
    # - Cross-validation
    # - Test predictions
    
    # 3. Placeholder: Generate dummy predictions
    targets = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']
    predictions = pd.DataFrame({'id': range(1, len(test) + 1)})
    
    for target in targets:
        # TODO: Replace with actual model predictions
        predictions[target] = 0.0
    
    # 4. Save predictions
    output_path = Path(args.output_dir) / "predictions.csv"
    predictions.to_csv(output_path, index=False)
    
    # 5. Compute hash
    with open(output_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    
    # 6. Save metrics
    metrics = {
        'experiment_id': '[EXPERIMENT ID]',
        'mean_oof_r2': 0.0,  # TODO: Compute from CV
        'prediction_hash': pred_hash,
        'smoke_test': args.smoke
    }
    
    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Experiment complete")
    print(f"Hash: {pred_hash}")

def verify_data_hashes(data_dir):
    """Verify data integrity"""
    expected = {
        'train.csv': '609b0f48',
        'test.csv': 'd8a0da26',
        'PI1M.csv': 'c5e1017b',
        'smile_r3.csv': 'c64f96ee'
    }
    
    for fname, expected_hash in expected.items():
        path = data_dir / fname
        if not path.exists():
            print(f"WARNING: {fname} not found, skipping hash check")
            continue
            
        with open(path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()[:8]
        
        if actual_hash != expected_hash:
            raise ValueError(f"{fname} hash mismatch: {actual_hash} != {expected_hash}")
    
    print("✅ Data hashes verified")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../Dataset', help='Path to dataset')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--smoke', action='store_true', help='Quick smoke test')
    args = parser.parse_args()
    
    main(args)
EOF

chmod +x "${EXP_DIR}/run_experiment.py"

echo "✅ Created: ${EXP_DIR}"
echo ""
echo "Next steps:"
echo "  1. Edit ${EXP_DIR}/run_experiment.py"
echo "  2. Implement experiment logic (see PLAN.md for spec)"
echo "  3. Test: ./run.sh --exp ${EXP_ID} --smoke"
echo "  4. Run: ./run.sh --exp ${EXP_ID}"
echo ""
echo "Templates available in PROMPT.md"
