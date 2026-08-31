#!/bin/bash
# Phase 5 Continuous Experiment Automation Pipeline

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

QUEUE=("P5-325" "P5-326")

echo "=== Phase 5 Autonomous Experiment Queue ==="
echo "Total experiments in current wave: ${#QUEUE[@]}"

for EXP_ID in "${QUEUE[@]}"; do
    EXP_DIR=$(find "${SCRIPT_DIR}/experiments" -type d -name "${EXP_ID}-*" | head -n1)
    if [[ -n "${EXP_DIR}" && -f "${EXP_DIR}/oracle_scores.json" ]]; then
        echo "⏭️  ${EXP_ID} already completed and scored. Skipping."
        continue
    fi
    
    echo ""
    echo "========================================================="
    echo "▶ Launching Queued Experiment: ${EXP_ID}"
    echo "========================================================="
    bash "${SCRIPT_DIR}/run.sh" --exp "${EXP_ID}"
    echo "✅ Finished and scored: ${EXP_ID}"
    
    # Run real-time monitor update
    "${SCRIPT_DIR}/../.venv/bin/python" "${SCRIPT_DIR}/monitor.py" || true
done

echo ""
echo "========================================================="
echo "🎉 Wave Completed!"
echo "========================================================="
