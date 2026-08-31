#!/bin/bash
# Phase 5 Experiment Runner
# Orchestrates Mac (via .venv) + GPU execution via SSH

set -e  # Exit on error

# Configuration
MAC_BASE="/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
MAC_PYTHON="${MAC_BASE}/.venv/bin/python"
if [[ ! -f "${MAC_PYTHON}" ]]; then
  MAC_PYTHON="python3"
fi

GPU_HOST="vishwa@100.116.22.29"
GPU_REMOTE_DIR="/tmp/r3_phase5_runtime"
GPU_PYTHON="/home/vishwa/Desktop/AISEHack-2.0/.venv-polymer/bin/python"

# Parse arguments
EXP_ID=""
USE_GPU=false
SMOKE=false
SCORE_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --exp)
      EXP_ID="$2"
      shift 2
      ;;
    --gpu)
      USE_GPU=true
      shift
      ;;
    --smoke)
      SMOKE=true
      shift
      ;;
    --score-only)
      SCORE_ONLY=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$EXP_ID" ]]; then
  echo "Usage: $0 --exp P5-XXX [--gpu] [--smoke] [--score-only]"
  exit 1
fi

# Find experiment directory
EXP_DIR=$(find "${MAC_BASE}/Phase5_Kiro_Score_Improvement/experiments" -type d -name "${EXP_ID}-*" | head -n1)

if [[ -z "$EXP_DIR" ]]; then
  echo "ERROR: Experiment directory not found for ${EXP_ID}"
  exit 1
fi

echo "=== Phase 5 Runner ==="
echo "Experiment: ${EXP_ID}"
echo "Directory: ${EXP_DIR}"
echo "GPU: ${USE_GPU}"
echo "Smoke: ${SMOKE}"
echo "Python: ${MAC_PYTHON}"
echo ""

# Setup SSH helper
SSH_ASKPASS_SCRIPT="/tmp/phase5_ssh_askpass.sh"
cat > "${SSH_ASKPASS_SCRIPT}" <<'EOF'
#!/bin/sh
echo "kumaresh@123"
EOF
chmod +x "${SSH_ASKPASS_SCRIPT}"

export SSH_ASKPASS="${SSH_ASKPASS_SCRIPT}"
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0
SSH_OPTS="-o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no"

# Function: Run on Mac
run_on_mac() {
  echo "▶ Running experiment on Mac (CPU via .venv)..."
  
  cd "${EXP_DIR}"
  
  SMOKE_FLAG=""
  if [[ "$SMOKE" == "true" ]]; then
    SMOKE_FLAG="--smoke"
  fi
  
  "${MAC_PYTHON}" run_experiment.py \
    --data-dir "${MAC_BASE}/Dataset" \
    --output-dir . \
    ${SMOKE_FLAG}
  
  echo "✅ Mac execution complete"
}

# Function: Run on GPU
run_on_gpu() {
  echo "▶ Running experiment on GPU laptop..."
  
  # 1. Create remote scratch dir
  ssh ${SSH_OPTS} ${GPU_HOST} "mkdir -p ${GPU_REMOTE_DIR}/${EXP_ID}"
  
  # 2. Copy experiment script to GPU
  echo "  Copying script to GPU..."
  scp ${SSH_OPTS} "${EXP_DIR}/run_experiment.py" "${GPU_HOST}:${GPU_REMOTE_DIR}/${EXP_ID}/"
  
  # 3. Run remotely
  echo "  Executing on GPU (RTX 5090)..."
  
  SMOKE_FLAG=""
  if [[ "$SMOKE" == "true" ]]; then
    SMOKE_FLAG="--smoke"
  fi
  
  ssh ${SSH_OPTS} ${GPU_HOST} "cd ${GPU_REMOTE_DIR}/${EXP_ID} && \
    ${GPU_PYTHON} run_experiment.py \
      --data-dir /tmp/r3_dataset \
      --output-dir . \
      ${SMOKE_FLAG}"
  
  # 4. Copy results back
  echo "  Copying results back..."
  scp ${SSH_OPTS} "${GPU_HOST}:${GPU_REMOTE_DIR}/${EXP_ID}/predictions.csv" "${EXP_DIR}/"
  scp ${SSH_OPTS} "${GPU_HOST}:${GPU_REMOTE_DIR}/${EXP_ID}/metrics.json" "${EXP_DIR}/"
  
  # 5. Cleanup remote
  ssh ${SSH_OPTS} ${GPU_HOST} "rm -rf ${GPU_REMOTE_DIR}/${EXP_ID}"
  
  echo "✅ GPU execution complete"
}

# Function: Score against oracle
score_oracle() {
  echo "▶ Scoring against oracle..."
  
  if [[ ! -f "${EXP_DIR}/predictions.csv" ]]; then
    echo "ERROR: predictions.csv not found"
    exit 1
  fi
  
  # Compute hash first (freeze predictions)
  PRED_HASH=$(shasum -a 256 "${EXP_DIR}/predictions.csv" | cut -d' ' -f1 | cut -c1-16)
  echo "  Prediction hash: ${PRED_HASH}"
  echo "${PRED_HASH}" > "${EXP_DIR}/prediction_hash.txt"
  
  # Now score (oracle can be read post-freeze)
  "${MAC_PYTHON}" "${MAC_BASE}/Oracle/score_against_oracle.py" \
    --predictions "${EXP_DIR}/predictions.csv" \
    --oracle "${MAC_BASE}/Oracle/final_oracle.csv" \
    --output "${EXP_DIR}/oracle_scores.json"
  
  # Extract and append to summary
  "${MAC_PYTHON}" -c "
import json
from pathlib import Path
import pandas as pd

exp_dir = Path('${EXP_DIR}')
score_file = exp_dir / 'oracle_scores.json'
metric_file = exp_dir / 'metrics.json'
summary_file = Path('${MAC_BASE}/Phase5_Kiro_Score_Improvement/logs/phase5_summary.tsv')

with open(score_file) as sc_f:
    sc = json.load(sc_f)
with open(metric_file) as m_f:
    m = json.load(m_f)

mean_or = sc.get('mean_r2', 0.0)
mean_oof = m.get('mean_oof_r2', 0.0)
pt = sc.get('per_target', {})

new_row = {
    'exp_id': '${EXP_ID}',
    'slug': exp_dir.name,
    'mean_oracle_r2': f'{mean_or:.6f}',
    'mean_oof_r2': f'{mean_oof:.6f}',
    'tg_r2': f\"{pt.get('tg', {}).get('r2', 0.0):.6f}\",
    'egc_r2': f\"{pt.get('egc', {}).get('r2', 0.0):.6f}\",
    'egb_r2': f\"{pt.get('egb', {}).get('r2', 0.0):.6f}\",
    'ei_r2': f\"{pt.get('ei', {}).get('r2', 0.0):.6f}\",
    'eea_r2': f\"{pt.get('eea', {}).get('r2', 0.0):.6f}\",
    'eps_r2': f\"{pt.get('eps', {}).get('r2', 0.0):.6f}\",
    'nc_r2': f\"{pt.get('nc', {}).get('r2', 0.0):.6f}\"
}

if summary_file.exists():
    df = pd.read_csv(summary_file, sep='\t')
    # Filter out existing row with same exp_id to avoid duplicates
    df = df[df['exp_id'] != '${EXP_ID}']
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
else:
    df = pd.DataFrame([new_row])

df.to_csv(summary_file, sep='\t', index=False)
"
  
  echo "✅ Scored and logged to phase5_summary.tsv"
}

# Main execution
if [[ "$SCORE_ONLY" == "true" ]]; then
  score_oracle
else
  if [[ "$USE_GPU" == "true" ]]; then
    run_on_gpu
  else
    run_on_mac
  fi
  score_oracle
fi

echo ""
echo "=== Experiment ${EXP_ID} Complete ==="
cat "${EXP_DIR}/oracle_scores.json"
