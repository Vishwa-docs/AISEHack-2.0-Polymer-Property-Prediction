#!/bin/bash
# Phase5A runner: runs ALL scaffolded experiments, freezes + scores each against
# Oracle/final_oracle.csv, appends logs/phase5a_summary.tsv, prints results.
# Usage: bash run.sh            (run all unfinished experiments)
#        bash run.sh P5A-003    (run one experiment)
#        FORCE=1 bash run.sh    (re-run even if already scored)
set -e
set -o pipefail
BASE="/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
P5A="$BASE/Phase5A_Gap_Analysis"
PY="$BASE/.venv/bin/python"
if [[ ! -f "$PY" ]]; then PY="python3"; fi
cd "$P5A"

if [[ -n "$1" ]]; then
  QUEUE="$1"
else
  QUEUE="P5A-000 P5A-001 P5A-002 P5A-003 P5A-004 P5A-005 P5A-006 P5A-007 P5A-008"
fi

for EXP_ID in $QUEUE; do
  EXP_DIR=$(find "$P5A/experiments" -type d -name "$EXP_ID-*" | head -n1)
  if [[ -z "$EXP_DIR" ]]; then
    echo "ERROR: experiment dir not found for $EXP_ID"; exit 1
  fi
  if [[ -z "$FORCE" && -f "$EXP_DIR/oracle_scores.json" ]]; then
    echo "skip $EXP_ID (already scored)"
    continue
  fi
  echo "========================================================="
  echo ">>> Running $EXP_ID  ($EXP_DIR)"
  echo "========================================================="
  # clean-lane scan: experiment code must never reference the answer panel
  if grep -rniE "oracle" "$EXP_DIR/run_experiment.py" "$EXP_DIR/exp_core.py" >/dev/null 2>&1; then
    echo "ERROR: clean-lane scan failed for $EXP_ID. Aborting."
    exit 1
  fi
  cd "$EXP_DIR"
  "$PY" run_experiment.py --data-dir "$BASE/Dataset" --output-dir . 2>&1 | tee run.log
  cd "$P5A"
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
mt = json.load(open(exp_dir / "metrics.json"))
pt = sc.get("per_target", {})
row = {"exp_id": exp_id, "slug": exp_dir.name,
       "mean_oracle_r2": round(sc["mean_r2"], 6),
       "mean_oof_r2": round(mt.get("mean_oof_r2", 0.0), 6)}
for t in ["tg", "egc", "egb", "ei", "eea", "eps", "nc"]:
    row[t + "_r2"] = round(pt.get(t, {}).get("r2", 0.0), 6)
summary = Path("/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/Phase5A_Gap_Analysis/logs/phase5a_summary.tsv")
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
echo "========== Phase5A summary (Oracle/final_oracle.csv, post-freeze) =========="
cat "$P5A/logs/phase5a_summary.tsv"
echo ""
echo "Incumbent V57 mean R2 = 0.9024 | target 0.9350 | est private = mean - 0.011"
