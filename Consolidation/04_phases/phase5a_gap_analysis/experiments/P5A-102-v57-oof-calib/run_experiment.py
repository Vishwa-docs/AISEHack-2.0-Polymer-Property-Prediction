#!/usr/bin/env python3
"""P5A-102 wrapper: runs the patched V57 standalone in this dir."""
import argparse, json, os, subprocess, sys

EXP_ID = "P5A-102"
ARMS = "calib"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default=".")
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(a.output_dir, "predictions.csv")
    cmd = [sys.executable, os.path.join(here, "v57_standalone.py"),
           "--data-dir", a.data_dir, "--out", out]
    r = subprocess.run(cmd, cwd=a.output_dir)
    if r.returncode != 0:
        sys.exit(r.returncode)
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump({"exp_id": EXP_ID, "arms": ARMS}, f, indent=1)
    print("DONE", EXP_ID)

if __name__ == "__main__":
    main()
