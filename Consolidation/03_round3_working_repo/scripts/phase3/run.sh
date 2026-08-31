#!/usr/bin/env bash
# =============================================================================
# Phase_3 Master PARALLEL Runner - 282 experiments, resource-aware, resumable.
#
#   ./run.sh                     run all pending experiments; smile_r3 / PI1M
#                                representation-learning exps (blocks C/F/L and
#                                any name with r3|pi1m|mlm|ssl|w2v|ppmi|svd|bert)
#                                are scheduled FIRST, then everything else.
#   ./run.sh 5 12                only experiments 005..012
#   ./run.sh 1 1 --smoke         smoke mode (fast, tiny data) for experiment 001
#   ./run.sh --smoke             smoke mode for the whole pending range
#   ./run.sh --long              ALSO run LONG RUN experiments (multi-hour SSL,
#                                full-corpus SVD/PPMI, 3D conformers, big MLMs)
#   ./run.sh --force             rerun experiments even if already completed
#   ./run.sh --jobs 3            force a fixed global parallel job count
#
# Resumability: any experiment already marked "completed" in summary.tsv (and
# whose metrics.json exists) is SKIPPED.  The 16 experiments that already ran
# (001-009, 015/016/018-021/024) are picked up automatically - never re-done
# unless --force.
#
# Parallelism - two independent token pools, both driven by a live monitor:
#   * GLOBAL slots  : CPU/RAM/load headroom.   Each GBM job uses ~8 threads, so
#                     slots = ncpu/8, capped at MAX_JOBS_CAP (default 4), and
#                     reduced when RAM or load1 headroom is tight.  This pool is
#                     what parallelizes the CPU-heavy experiments.
#   * GPU slots     : free VRAM headroom.  GPU jobs (torch: gnn / mlp / mlm /
#                     bert - detected by peeking the experiment config) must
#                     also hold a GPU token, so at most N torch jobs run at
#                     once.  With the user's llama-server holding ~21 GB VRAM,
#                     the monitor allows exactly 1 GPU job; if it is stopped
#                     the pool grows (cap 2).  Never OOMs the GPU.
#   A worker that pops a GPU job while the GPU pool is full re-queues it after
#   ~2 minutes instead of blocking CPU work behind it.
#
# Ordering: priority names first (numeric order within the priority list), then
# the remaining indices in order.  A shared flock-guarded queue file hands out
# one index at a time, so slow/heavy experiments never block fast ones.
#
# Logs: outputs_and_logs/logs/<expname>.log      (per-experiment stdout)
#       outputs_and_logs/logs/summary.tsv        (idx name status mean_r2 secs)
#       outputs_and_logs/logs/runner.log         (monitor + scheduler timeline)
#
# Safety: per-worker OMP/OPENBLAS thread caps; load1 and swap guards; nothing
# here ever writes outside outputs_and_logs/ (except the experiment outputs).
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_DIR="${SCRIPT_DIR}/experiments"
OUT_ROOT="${SCRIPT_DIR}/outputs_and_logs/output"
LOG_ROOT="${SCRIPT_DIR}/outputs_and_logs/logs"
QUEUE_DIR="${SCRIPT_DIR}/outputs_and_logs/.queue"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${QUEUE_DIR}"

SUMMARY="${LOG_ROOT}/summary.tsv"
RUNNER_LOG="${LOG_ROOT}/runner.log"
touch "${SUMMARY}"

DATA_DIR="${DATA_DIR:-/home/vishwa/Desktop/r3_runtime/data}"

START=1
END=""
EXTRA_ARGS=()
INCLUDE_LONG=0
FORCE=0
JOBS_OVERRIDE=""

POSITIONALS=()
PREV=""
for arg in "$@"; do
    if [[ "${PREV}" == "--jobs" ]]; then
        JOBS_OVERRIDE="${arg}"
        PREV=""
        continue
    fi
    if [[ "${arg}" == --* ]]; then
        case "${arg}" in
            --long) INCLUDE_LONG=1 ;;
            --force) FORCE=1 ;;
            --jobs) PREV="--jobs" ;;
            *) EXTRA_ARGS+=("${arg}") ;;
        esac
    else
        POSITIONALS+=("${arg}")
    fi
    [[ "${arg}" != "--jobs" ]] && PREV=""
done

if [ ${#POSITIONALS[@]} -ge 1 ]; then START="${POSITIONALS[0]}"; fi

# Auto-detect the experiment count so newly generated indices (251+) are
# included without editing this script.
MAX_INDEX=$(ls "${EXP_DIR}"/exp[0-9]*_*.py 2>/dev/null \
    | sed -E 's|.*/exp([0-9]+)_.*|\1|' | sort -n | tail -1)
MAX_INDEX=${MAX_INDEX:-250}
if [ ${#POSITIONALS[@]} -ge 2 ]; then
    END="${POSITIONALS[1]}"
else
    END="${MAX_INDEX}"
fi

PYTHON="${PYTHON:-python3}"
if [ -x "/home/vishwa/Desktop/AISEHack-2.0/.venv-polymer/bin/python" ]; then
    PYTHON="/home/vishwa/Desktop/AISEHack-2.0/.venv-polymer/bin/python"
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RUNNER_LOG}"; }

echo "======================================================================"
echo "  Phase_3 PARALLEL runner: experiments ${START}..${END} of ${MAX_INDEX}"
echo "  Started: $(date)"
echo "  Python : ${PYTHON}"
echo "  Data   : ${DATA_DIR}"
echo "  Long   : $([ ${INCLUDE_LONG} -eq 1 ] && echo included || echo skipped)"
echo "  Force  : $([ ${FORCE} -eq 1 ] && echo yes || echo no)"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then echo "  Flags  : ${EXTRA_ARGS[*]}"; fi
echo "======================================================================"

# ---------------------------------------------------------------------------
# Build the ordered work list (priority first, numeric order within each list).
# ---------------------------------------------------------------------------
declare -A ALREADY_DONE
if [ -f "${SUMMARY}" ]; then
    while IFS=$'\t' read -r idx name status meanr2 dur; do
        [[ "${status}" == "completed" ]] && ALREADY_DONE["${idx}"]=1
    done < "${SUMMARY}"
fi

is_r3_priority() {
    case "$1" in
        *-r3-*|*pi1m*|*mlm*|*ssl*|*w2v*|*ppmi*|*svd*|*bert*|*smile_r3*) return 0 ;;
        *) return 1 ;;
    esac
}

is_gpu_exp() {
    # Torch-backed experiments (gnn / mlp kinds, mlm/bert SSL feature builders)
    grep -qiE "'gnn'|'mlm'|'mlp'|'bert'" "$1" 2>/dev/null
}

PRIORITY_LIST=()
NORMAL_LIST=()
for i in $(seq -f "%03g" "${START}" "${END}"); do
    EXP_FILE=$(ls "${EXP_DIR}"/exp${i}_*.py 2>/dev/null | head -1)
    [ -z "${EXP_FILE}" ] && continue
    EXP_NAME=$(basename "${EXP_FILE}" .py)
    METRICS="${OUT_ROOT}/${EXP_NAME}/metrics.json"
    if [ ${FORCE} -eq 0 ] && [ -n "${ALREADY_DONE[${i}]:-}" ] && [ -f "${METRICS}" ]; then
        echo "[${i}] SKIP (already completed) ${EXP_NAME}"
        continue
    fi
    if [ ${INCLUDE_LONG} -eq 0 ] && grep -q "LONG RUN" "${EXP_FILE}"; then
        echo "[${i}] SKIPPED (long) ${EXP_NAME} - rerun with --long"
        printf "%s\t%s\t%s\t%s\t%s\n" "${i}" "${EXP_NAME}" "skipped_long" "NA" "0" >> "${SUMMARY}"
        continue
    fi
    if is_r3_priority "${EXP_NAME}"; then
        PRIORITY_LIST+=("${i}")
    else
        NORMAL_LIST+=("${i}")
    fi
done

QUEUE_FILE="${QUEUE_DIR}/pending.txt"
: > "${QUEUE_FILE}"
for i in "${PRIORITY_LIST[@]}" "${NORMAL_LIST[@]}"; do echo "${i}" >> "${QUEUE_FILE}"; done
TOTAL_PENDING=$(wc -l < "${QUEUE_FILE}" | tr -d ' ')
log "Queue built: ${#PRIORITY_LIST[@]} smile_r3/PI1M-priority + ${#NORMAL_LIST[@]} normal = ${TOTAL_PENDING} pending experiments"

if [ "${TOTAL_PENDING}" -eq 0 ]; then
    log "Nothing to run - all experiments in range already completed. Use --force to rerun."
    exit 0
fi

# ---------------------------------------------------------------------------
# Resource-aware concurrency.  The monitor re-computes both budgets every 20s
# and writes them to files the workers read before taking a token, so the fleet
# throttles up/down without being restarted.  Initial values are computed
# synchronously so workers never stampede on a placeholder.
# ---------------------------------------------------------------------------
NCPU=$(nproc 2>/dev/null || echo 8)
MAX_JOBS_CAP=4
GPU_JOBS_CAP=2
SLOTS_FILE="${QUEUE_DIR}/slots"
GPU_SLOTS_FILE="${QUEUE_DIR}/gpu_slots"
TOKENS_DIR="${QUEUE_DIR}/tokens"
GPU_TOKENS_DIR="${QUEUE_DIR}/gpu_tokens"
mkdir -p "${TOKENS_DIR}" "${GPU_TOKENS_DIR}"
rm -f "${TOKENS_DIR}"/* "${GPU_TOKENS_DIR}"/* 2>/dev/null
echo "1" > "${SLOTS_FILE}"
echo "1" > "${GPU_SLOTS_FILE}"

compute_slots() {
    if [ -n "${JOBS_OVERRIDE}" ]; then echo "${JOBS_OVERRIDE}"; return; fi
    local load1 mem_avail_gi cpu_slots mem_slots slots swap_used_pct load_int
    load1=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
    mem_avail_gi=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')
    [ -z "${mem_avail_gi}" ] && mem_avail_gi=8
    slots=1
    cpu_slots=$(( NCPU / 8 ))
    [ "${cpu_slots}" -lt 1 ] && cpu_slots=1
    mem_slots=$(( mem_avail_gi / 8 ))
    [ "${mem_slots}" -lt 1 ] && mem_slots=1
    slots=${cpu_slots}
    [ "${slots}" -gt "${mem_slots}" ] && slots=${mem_slots}
    load_int=${load1%.*}
    [ "${load_int}" -ge $(( NCPU - 2 )) ] && slots=1
    # hard RAM guard: below 10 GiB available -> serialise to avoid swap-death
    [ "${mem_avail_gi}" -lt 10 ] && slots=1
    swap_used_pct=$(free 2>/dev/null | awk '/^Swap:/{ if ($2>0) printf "%d", $3*100/$2; else print 0 }')
    [ -n "${swap_used_pct}" ] && [ "${swap_used_pct}" -ge 60 ] && slots=1
    [ "${slots}" -gt "${MAX_JOBS_CAP}" ] && slots=${MAX_JOBS_CAP}
    [ "${slots}" -lt 1 ] && slots=1
    echo "${slots}"
}

compute_gpu_slots() {
    # GPU jobs are tiny torch models (<= ~1.5 GB each); budget from free VRAM,
    # always >= 1, capped at GPU_JOBS_CAP so a second torch job can overlap a
    # fast first one only when there is real VRAM headroom.
    local free_mib slots
    if ! command -v nvidia-smi >/dev/null 2>&1; then echo "1"; return; fi
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -z "${free_mib}" ] && free_mib=0
    slots=$(( free_mib / 2500 ))
    [ "${slots}" -lt 1 ] && slots=1
    [ "${slots}" -gt "${GPU_JOBS_CAP}" ] && slots=${GPU_JOBS_CAP}
    echo "${slots}"
}

MONITOR_PID=""
monitor_loop() {
    while true; do
        local s g
        s=$(compute_slots)
        g=$(compute_gpu_slots)
        echo "${s}" > "${SLOTS_FILE}.tmp" && mv "${SLOTS_FILE}.tmp" "${SLOTS_FILE}"
        echo "${g}" > "${GPU_SLOTS_FILE}.tmp" && mv "${GPU_SLOTS_FILE}.tmp" "${GPU_SLOTS_FILE}"
        log "[monitor] global_slots=${s} gpu_slots=${g} (cpu=${NCPU} load=$(awk '{print $1}' /proc/loadavg 2>/dev/null) mem_avail=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')GiB)"
        sleep 20
    done
}

INIT_SLOTS=$(compute_slots)
INIT_GPU_SLOTS=$(compute_gpu_slots)
echo "${INIT_SLOTS}" > "${SLOTS_FILE}"
echo "${INIT_GPU_SLOTS}" > "${GPU_SLOTS_FILE}"
log "[init] global_slots=${INIT_SLOTS} gpu_slots=${INIT_GPU_SLOTS} (MAX_JOBS_CAP=${MAX_JOBS_CAP}, GPU_JOBS_CAP=${GPU_JOBS_CAP})"
monitor_loop &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null; jobs -p | xargs -r kill 2>/dev/null' EXIT INT TERM

# Per-worker BLAS/OpenMP thread cap so N workers don't oversubscribe the cores.
OMP_THREADS=$(( NCPU / MAX_JOBS_CAP ))
[ "${OMP_THREADS}" -lt 1 ] && OMP_THREADS=1
export OMP_NUM_THREADS="${OMP_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_THREADS}"
export MKL_NUM_THREADS="${OMP_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_THREADS}"

# ---------------------------------------------------------------------------
# Queue helpers (flock-guarded) + worker loop with dual token pools.
# ---------------------------------------------------------------------------
pop_next() {
    (
        flock -x 200
        local line
        line=$(head -1 "${QUEUE_FILE}" 2>/dev/null)
        [ -z "${line}" ] && exit 1
        tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
        echo "${line}"
    ) 200>"${QUEUE_DIR}/queue.lock"
}

requeue() {
    # put an index back at the FRONT of the queue (GPU-job retry path)
    (
        flock -x 200
        local tmp
        tmp=$(mktemp "${QUEUE_DIR}/req.XXXXXX")
        echo "$1" > "${tmp}"
        cat "${QUEUE_FILE}" >> "${tmp}"
        mv "${tmp}" "${QUEUE_FILE}"
    ) 200>"${QUEUE_DIR}/queue.lock"
}

run_one() {
    local i="$1" EXP_FILE EXP_NAME OUT_DIR LOG_FILE START_TS END_TS DURATION RC MEAN_R2
    EXP_FILE=$(ls "${EXP_DIR}"/exp${i}_*.py 2>/dev/null | head -1)
    [ -z "${EXP_FILE}" ] && return
    EXP_NAME=$(basename "${EXP_FILE}" .py)
    OUT_DIR="${OUT_ROOT}/${EXP_NAME}"
    LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"
    mkdir -p "${OUT_DIR}"
    log "[${i}] RUNNING ${EXP_NAME}"
    START_TS=$(date +%s)
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        "${PYTHON}" -u "${EXP_FILE}" "${EXTRA_ARGS[@]}" --output "${OUT_DIR}" --data-dir "${DATA_DIR}" > "${LOG_FILE}" 2>&1
    else
        "${PYTHON}" -u "${EXP_FILE}" --output "${OUT_DIR}" --data-dir "${DATA_DIR}" > "${LOG_FILE}" 2>&1
    fi
    RC=$?
    END_TS=$(date +%s); DURATION=$((END_TS - START_TS))
    (
        flock -x 201
        if [ ${RC} -eq 0 ]; then
            MEAN_R2=$("${PYTHON}" -c "import json,sys; m=json.load(open(sys.argv[1])); v=m.get('mean_r2'); print('NA' if v is None or v!=v else f'{v:.5f}')" "${OUT_DIR}/metrics.json" 2>/dev/null)
            log "[${i}] COMPLETED ${EXP_NAME} (${DURATION}s) mean R2 = ${MEAN_R2}"
            printf "%s\t%s\t%s\t%s\t%s\n" "${i}" "${EXP_NAME}" "completed" "${MEAN_R2}" "${DURATION}" >> "${SUMMARY}"
        else
            log "[${i}] FAILED ${EXP_NAME} (${DURATION}s) - see ${LOG_FILE}"
            printf "%s\t%s\t%s\t%s\t%s\n" "${i}" "${EXP_NAME}" "failed" "NA" "${DURATION}" >> "${SUMMARY}"
        fi
    ) 201>"${QUEUE_DIR}/summary.lock"
}

worker() {
    local wid="$1"
    # stagger so workers don't all poll the slots/tokens in the same instant
    sleep $(( (RANDOM % 4) ))
    while true; do
        local slots token_count i EXP_FILE
        slots=$(cat "${SLOTS_FILE}" 2>/dev/null || echo 1)
        while true; do
            token_count=$(find "${TOKENS_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')
            if [ "${token_count}" -lt "${slots}" ]; then
                touch "${TOKENS_DIR}/w${wid}"
                break
            fi
            sleep 5
        done
        i=$(pop_next) || { rm -f "${TOKENS_DIR}/w${wid}"; break; }
        EXP_FILE=$(ls "${EXP_DIR}"/exp${i}_*.py 2>/dev/null | head -1)
        if [ -n "${EXP_FILE}" ] && is_gpu_exp "${EXP_FILE}"; then
            local gpu_slots gtc gwait
            gwait=0
            while true; do
                gpu_slots=$(cat "${GPU_SLOTS_FILE}" 2>/dev/null || echo 1)
                gtc=$(find "${GPU_TOKENS_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')
                if [ "${gtc}" -lt "${gpu_slots}" ]; then
                    touch "${GPU_TOKENS_DIR}/w${wid}"
                    break
                fi
                gwait=$((gwait + 1))
                if [ "${gwait}" -ge 12 ]; then
                    # GPU pool saturated: re-queue and free the global slot so
                    # CPU work is not blocked behind a waiting torch job.
                    rm -f "${TOKENS_DIR}/w${wid}"
                    requeue "${i}"
                    log "[worker-${wid}] [${i}] GPU pool full - re-queued ${i}, retrying later"
                    sleep 30
                    continue 2
                fi
                sleep 10
            done
            run_one "${i}"
            rm -f "${GPU_TOKENS_DIR}/w${wid}"
        else
            run_one "${i}"
        fi
        rm -f "${TOKENS_DIR}/w${wid}"
    done
}

NWORKERS=${MAX_JOBS_CAP}
WORKER_PIDS=()
for w in $(seq 1 "${NWORKERS}"); do
    worker "${w}" &
    WORKER_PIDS+=($!)
done

log "Launched ${NWORKERS} worker loops (slots file: ${SLOTS_FILE}, gpu slots: ${GPU_SLOTS_FILE}, queue: ${QUEUE_FILE})"
wait "${WORKER_PIDS[@]}" 2>/dev/null
kill "${MONITOR_PID}" 2>/dev/null

FAILED=$(awk -F'\t' '$3=="failed"{c++} END{print c+0}' "${SUMMARY}")
echo ""
echo "======================================================================"
echo "  Phase_3 parallel run finished: $(date)"
echo "  Failed (cumulative): ${FAILED}   Summary: ${SUMMARY}"
echo "  Outputs : ${OUT_ROOT}"
echo "  Logs    : ${LOG_ROOT} (runner.log has the scheduler timeline)"
echo "======================================================================"
exit 0
