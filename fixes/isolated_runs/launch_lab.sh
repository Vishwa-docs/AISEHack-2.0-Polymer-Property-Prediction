#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$run_dir"
export PPP_DATA_DIR="$run_dir/data"
export PPP_RUN_DIR="$run_dir"
export PPP_OUTPUT_DIR="$run_dir/outputs"
export PPP_CHECKPOINT_DIR="$run_dir/checkpoints"
export JUPYTER_DATA_DIR="$run_dir/.jupyter_data"
export IPYTHONDIR="$run_dir/.ipython"
export MPLCONFIGDIR="$run_dir/.matplotlib"
mkdir -p "$PPP_OUTPUT_DIR" "$PPP_CHECKPOINT_DIR" "$JUPYTER_DATA_DIR"
exec "$run_dir/.venv/bin/jupyter" lab "$run_dir/Sandman_Polymer_Property_Prediction_2_906.ipynb"
