#!/usr/bin/env python3
"""P5A-004 - eps modeled as ionic residual: predict (eps - nc^2) with nc^2 partner feature, reconstruct. Clean lane: reads ONLY official train.csv / test.csv
(plus smile_r3.csv where stated). Fixed seeds. Grouped folds."""
import argparse, json, os
import numpy as np
from scipy import sparse as sp
import exp_core as core


CONFIG = {
    "exp_id": "P5A-004",
    "focus": "eps modeled as ionic residual: predict (eps - nc^2) with nc^2 partner feature, reconstruct",
    "eps_residual": True,
    "n_estimators": 300,
    "char_max_features": 4096,
    "morgan_nbits": 1024,
    "clip": True,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg["smoke"] = a.smoke
    cfg["output_dir"] = a.output_dir
    if a.smoke:
        cfg["n_estimators"] = 20
        cfg["char_max_features"] = 512
        cfg["morgan_nbits"] = 256
    train, test = core.load(a.data_dir)
    train = core.smoke_sample(train, cfg)
    Xtr, Xte = core.build_features(train.smiles.values, test.smiles.values, cfg)

    sub, metrics = core.run_pipeline(train, test, Xtr, Xte, cfg)
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=1)
    print("DONE P5A-004 mean_oof_r2=%.4f" % metrics["mean_oof_r2"])

if __name__ == "__main__":
    main()
