#!/bin/bash
# Phase5A FINAL series runner: V57-standalone + arm experiments (P5A-100..124).
# Auto-routes to the GPU laptop when SSH_PASS is set and the host is reachable;
# otherwise runs on CPU. Scores every run vs Oracle/final_oracle.csv, stores all results.
# Usage: bash run_final.sh                       (run all unfinished P5A-100+ exps)
#        bash run_final.sh P5A-124               (run one)
#        FORCE=1 bash run_final.sh               (re-run even if scored)
#        P5A_PYTHON=/path/python bash run_final.sh   (interpreter override; default python3)
set -e
set -o pipefail
BASE="/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
P5A="$BASE/Phase5A_Gap_Analysis"
PY="$P5A_PYTHON"
if [[ -z "$PY" ]]; then PY="python3"; fi
cd "$P5A"

if [[ -n "$1" ]]; then QUEUE="$1"; else QUEUE="P5A-100 P5A-101 P5A-102 P5A-103 P5A-104 P5A-105 P5A-106 P5A-107 P5A-108 P5A-109 P5A-110 P5A-111 P5A-112 P5A-113 P5A-114 P5A-115 P5A-116 P5A-117 P5A-118 P5A-119 P5A-120 P5A-121 P5A-122 P5A-123 P5A-124 P5A-125 P5A-126 P5A-127"; fi

# ---- GPU auto-detection (only when SSH_PASS is provided) ----
GPU_OK=false
if [[ -n "$SSH_PASS" ]]; then
  cat > /tmp/r3_final_askpass.sh <<ASKEOF
#!/bin/sh
echo "$SSH_PASS"
ASKEOF
  chmod +x /tmp/r3_final_askpass.sh
  export SSH_ASKPASS=/tmp/r3_final_askpass.sh
  export SSH_ASKPASS_REQUIRE=force
  export DISPLAY=:0
  SSH_OPTS="-o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=6"
  if ssh $SSH_OPTS vishwa@100.116.22.29 "echo reachable" >/dev/null 2>&1; then
    GPU_OK=true
  fi
fi
GPU_HOST="vishwa@100.116.22.29"
GPU_PY="/home/vishwa/Desktop/AISEHack-2.0/.venv-polymer/bin/python"
GPU_DIR="/tmp/r3_final_runtime"
if [[ "$GPU_OK" == "true" ]]; then
  echo "GPU laptop reachable - P5A-1xx experiments will run there"
  ssh $SSH_OPTS $GPU_HOST "mkdir -p /tmp/r3_dataset $GPU_DIR"
  for DF in train.csv test.csv PI1M.csv smile_r3.csv; do
    if ! ssh $SSH_OPTS $GPU_HOST "test -f /tmp/r3_dataset/$DF" >/dev/null 2>&1; then
      echo "  syncing $DF to laptop (one-time) ..."
      scp $SSH_OPTS "$BASE/Dataset/$DF" "$GPU_HOST:/tmp/r3_dataset/"
    fi
  done
else
  echo "GPU laptop not available (set SSH_PASS to enable) - running on CPU"
fi

for EXP_ID in $QUEUE; do
  EXP_DIR=$(find "$P5A/experiments" -type d -name "$EXP_ID-*" | head -n1)
  if [[ -z "$EXP_DIR" ]]; then echo "ERROR: experiment dir not found for $EXP_ID"; exit 1; fi
  if [[ -z "$FORCE" && -f "$EXP_DIR/oracle_scores.json" ]]; then echo "skip $EXP_ID (already scored)"; continue; fi
  echo "========================================================="
  echo ">>> Running $EXP_ID  ($EXP_DIR)"
  echo "========================================================="
  if grep -rniE "oracle" "$EXP_DIR/run_experiment.py" "$EXP_DIR/v57_standalone.py" >/dev/null 2>&1; then
    echo "ERROR: clean-lane scan failed for $EXP_ID. Aborting."; exit 1
  fi
  if [[ "$GPU_OK" == "true" ]]; then
    ssh $SSH_OPTS $GPU_HOST "mkdir -p $GPU_DIR/$EXP_ID"
    scp $SSH_OPTS "$EXP_DIR/v57_standalone.py" "$EXP_DIR/run_experiment.py" "$GPU_HOST:$GPU_DIR/$EXP_ID/"
    ssh $SSH_OPTS $GPU_HOST "cd $GPU_DIR/$EXP_ID && $GPU_PY run_experiment.py --data-dir /tmp/r3_dataset --output-dir . > run.log 2>&1"
    scp $SSH_OPTS "$GPU_HOST:$GPU_DIR/$EXP_ID/predictions.csv" "$EXP_DIR/"
    scp $SSH_OPTS "$GPU_HOST:$GPU_DIR/$EXP_ID/run.log" "$EXP_DIR/"
    ssh $SSH_OPTS $GPU_HOST "rm -rf $GPU_DIR/$EXP_ID"
  else
    cd "$EXP_DIR"
    "$PY" run_experiment.py --data-dir "$BASE/Dataset" --output-dir . 2>&1 | tee run.log
    cd "$P5A"
  fi
  PRED_HASH=$(shasum -a 256 "$EXP_DIR/predictions.csv" | cut -d' ' -f1 | cut -c1-16)
  echo "$PRED_HASH" > "$EXP_DIR/prediction_hash.txt"
  "$PY" "$P5A/scripts/score_oracle.py" \
    --predictions "$EXP_DIR/predictions.csv" \
    --oracle "$BASE/Oracle/final_oracle.csv" \
    --output "$EXP_DIR/oracle_scores.json"
  "$PY" - "$EXP_DIR" "$EXP_ID" <<'PYEOF'
import json, sys
from pathlib import Path
import pandas as pd
exp_dir = Path(sys.argv[1]); exp_id = sys.argv[2]
sc = json.load(open(exp_dir / "oracle_scores.json"))
try:
    mt = json.load(open(exp_dir / "metrics.json"))
except Exception:
    mt = {}
pt = sc.get("per_target", {})
row = {"exp_id": exp_id, "slug": exp_dir.name,
       "mean_oracle_r2": round(sc["mean_r2"], 6),
       "mean_oof_r2": round(mt.get("mean_oof_r2", 0.0), 6)}
for t in ["tg", "egc", "egb", "ei", "eea", "eps", "nc"]:
    row[t + "_r2"] = round(pt.get(t, {}).get("r2", 0.0), 6)
summary = Path("/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/Phase5A_Gap_Analysis/logs/phase5a_final_summary.tsv")
if summary.exists():
    df = pd.read_csv(summary, sep="\t")
    df = df[df["exp_id"] != exp_id]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
else:
    df = pd.DataFrame([row])
df.to_csv(summary, sep="\t", index=False)
print("logged", exp_id, "mean_oracle_r2 =", round(sc["mean_r2"], 5))
PYEOF
done

echo ""
echo "========== Phase5A FINAL series summary (Oracle/final_oracle.csv, post-freeze) =========="
cat "$P5A/logs/phase5a_final_summary.tsv"
echo ""
echo "Incumbent V57 = 0.9023 | target 0.9350 | est private = mean - 0.011"
