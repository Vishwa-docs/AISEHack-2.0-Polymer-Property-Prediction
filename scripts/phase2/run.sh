#!/usr/bin/env bash
# =============================================================================
# Phase_2 Master Sequential Runner - 150 real experiments, one by one.
#
#   ./run.sh                run experiments 001..150 in order
#   ./run.sh 5 12           run only experiments 005..012
#   ./run.sh 1 1 --smoke    smoke mode (fast, small data) for experiment 001
#
# Every run writes:
#   outputs_and_logs/output/<exp_name>/   metrics.json, predictions.csv, oof.csv  (VALUES ONLY)
#   outputs_and_logs/logs/<exp_name>.log  full run log
#   outputs_and_logs/logs/summary.tsv     one row per experiment (status + mean R2)
# Status is printed to the terminal as each experiment starts and finishes.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_DIR="${SCRIPT_DIR}/experiments"
OUT_ROOT="${SCRIPT_DIR}/outputs_and_logs/output"
LOG_ROOT="${SCRIPT_DIR}/outputs_and_logs/logs"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

SUMMARY="${LOG_ROOT}/summary.tsv"
touch "${SUMMARY}"

START="${1:-1}"
END="${2:-150}"
EXTRA_ARGS="${3:-}"

PYTHON="${PYTHON:-python3}"
if [ -x "/home/vishwa/Desktop/AISEHack-2.0/.venv-polymer/bin/python" ]; then
    PYTHON="/home/vishwa/Desktop/AISEHack-2.0/.venv-polymer/bin/python"
fi

echo "======================================================================"
echo "  Phase_2 runner: experiments ${START}..${END} of 150"
echo "  Started: $(date)"
echo "  Python : ${PYTHON}"
echo "======================================================================"

FAILED=0
for i in $(seq -f "%03g" "${START}" "${END}"); do
    EXP_FILE=$(ls "${EXP_DIR}"/exp${i}_*.py 2>/dev/null | head -1)
    if [ -z "${EXP_FILE}" ]; then
        echo "[${i}/${END}] MISSING experiment file for index ${i}"
        continue
    fi
    EXP_NAME=$(basename "${EXP_FILE}" .py)
    OUT_DIR="${OUT_ROOT}/${EXP_NAME}"
    LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "[${i}/${END}] RUNNING  ${EXP_NAME}"
    echo "           started $(date +%H:%M:%S)"
    echo "----------------------------------------------------------------------"

    START_TS=$(date +%s)
    if [ -n "${EXTRA_ARGS}" ]; then
        ${PYTHON} "${EXP_FILE}" ${EXTRA_ARGS} --output "${OUT_DIR}" 2>&1 | tee "${LOG_FILE}"
    else
        ${PYTHON} "${EXP_FILE}" --output "${OUT_DIR}" 2>&1 | tee "${LOG_FILE}"
    fi
    RC=${PIPESTATUS[0]}
    END_TS=$(date +%s)
    DURATION=$((END_TS - START_TS))

    if [ ${RC} -eq 0 ]; then
        MEAN_R2=$(${PYTHON} -c "
import json
try:
    m = json.load(open('${OUT_DIR}/metrics.json'))
    print(m.get('mean_r2', 'NA'))
except Exception:
    print('NA')
" 2>/dev/null)
        echo "----------------------------------------------------------------------"
        echo "[${i}/${END}] COMPLETED ${EXP_NAME}  (${DURATION}s)  mean R2 = ${MEAN_R2}"
        printf "%s\t%s\t%s\t%s\t%s\n" "${i}" "${EXP_NAME}" "completed" "${MEAN_R2}" "${DURATION}" >> "${SUMMARY}"
    else
        FAILED=$((FAILED + 1))
        echo "----------------------------------------------------------------------"
        echo "[${i}/${END}] FAILED    ${EXP_NAME}  (${DURATION}s) - see ${LOG_FILE}"
        printf "%s\t%s\t%s\t%s\t%s\n" "${i}" "${EXP_NAME}" "failed" "NA" "${DURATION}" >> "${SUMMARY}"
    fi
done

echo ""
echo "======================================================================"
echo "  Phase_2 run finished: $(date)"
echo "  Failed: ${FAILED}   Summary: ${SUMMARY}"
echo "  Outputs : ${OUT_ROOT}"
echo "  Logs    : ${LOG_ROOT}"
echo "======================================================================"
exit ${FAILED}
