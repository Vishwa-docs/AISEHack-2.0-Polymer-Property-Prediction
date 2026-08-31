#!/usr/bin/env python3
"""P5A post-freeze scorer. Reads a frozen predictions.csv + the reference answer panel
(Oracle/final_oracle.csv), writes per-target and mean R2 to --output. Scoring only."""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    pred = pd.read_csv(a.predictions)
    orc = pd.read_csv(a.oracle)
    m = orc[["id", "target_type", "target"]].merge(
        pred.rename(columns={"target": "pred"}), on="id", how="left")
    m = m[m["target"].notna() & m["pred"].notna()]
    scores = {}
    for t in TARGETS:
        r = m[m.target_type == t]
        if len(r) < 2:
            scores[t] = {"r2": None, "mae": None, "rmse": None, "n": 0}
            continue
        y = r["target"].to_numpy(float)
        p_ = r["pred"].to_numpy(float)
        scores[t] = {"r2": float(r2_score(y, p_)), "mae": float(mean_absolute_error(y, p_)),
                     "rmse": float(np.sqrt(np.mean((y - p_) ** 2))), "n": int(len(r))}
    r2s = [scores[t]["r2"] for t in TARGETS if scores[t]["r2"] is not None]
    mean_r2 = float(np.mean(r2s))
    out = {"mean_r2": mean_r2, "est_private_lb": mean_r2 - 0.011,
           "per_target": scores, "covered_rows": int(len(m))}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print("P5A score: mean R2 = %.5f  est.private = %.5f" % (mean_r2, mean_r2 - 0.011))
    for t in TARGETS:
        s = scores[t]
        print("  %-4s R2=%.5f  MAE=%.4f  n=%d" % (t, s["r2"] or -1, s["mae"] or -1, s["n"]))

if __name__ == "__main__":
    main()
