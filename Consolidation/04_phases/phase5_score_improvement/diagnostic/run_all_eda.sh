#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"
if [[ ! -f "${VENV_PYTHON}" ]]; then
    VENV_PYTHON="python3"
fi

mkdir -p outputs/plots outputs/tables outputs/reports

echo "=== Phase 5 Diagnostic / EDA Suite ==="
echo "Using Python: ${VENV_PYTHON}"

echo ""
echo "▶ [1/6] Training data analysis (eda_train.py)..."
"${VENV_PYTHON}" eda_train.py --data-dir ../data --output-dir outputs

echo ""
echo "▶ [2/6] Test data & similarity analysis (eda_test.py)..."
"${VENV_PYTHON}" eda_test.py --data-dir ../data --output-dir outputs

echo ""
echo "▶ [3/6] Oracle gap analysis (eda_oracle.py)..."
"${VENV_PYTHON}" eda_oracle.py --data-dir ../data --output-dir outputs

echo ""
echo "▶ [4/6] 5.97M smile_r3 characterization (eda_smile_r3.py)..."
"${VENV_PYTHON}" eda_smile_r3.py --data-dir ../data --output-dir outputs --sample-size 100000

echo ""
echo "▶ [5/6] 995k PI1M characterization (eda_pi1m.py)..."
"${VENV_PYTHON}" eda_pi1m.py --data-dir ../data --output-dir outputs --sample-size 50000

echo ""
echo "▶ [6/6] Chemical families & multi-metric similarity..."
"${VENV_PYTHON}" eda_cross_dataset.py --data-dir ../data --output-dir outputs
"${VENV_PYTHON}" eda_chemical_families.py --data-dir ../data --output-dir outputs
"${VENV_PYTHON}" eda_similarity_analysis.py --data-dir ../data --output-dir outputs

echo ""
echo "=============================================="
echo "✅ All Diagnostic & EDA tasks completed successfully!"
echo "Reports: outputs/reports/"
echo "Plots:   outputs/plots/"
echo "Tables:  outputs/tables/"
echo "=============================================="
