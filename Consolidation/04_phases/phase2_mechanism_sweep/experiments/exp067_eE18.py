#!/usr/bin/env python3
"""R3-EE18 [eE18] - Phase E: SSL at scale (long runs). Experiment 67/150. Real grouped-CV pipeline; reads ONLY official Dataset/ inputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3_core import data as d
from r3_core import engine as eng
from r3_core import features as f
from r3_core import metrics as m
from r3_core import models as mo
from r3_core import physics as ph
from r3_core import panels as pn
from rdkit.Chem import Crippen, rdMolDescriptors

EXP_ID = "R3-EE18-20260827-eE18"
EXP_NAME = "eE18"
TARGETS = d.TARGETS
SEED = 2093


def run_experiment(output_dir: Path, smoke: bool = False, data_dir: Path | None = None) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] starting - SSL at scale (long runs)")
    model_fn = lambda X, y: mo.PerTargetEnsemble(seed=SEED, ridge_alpha=18, et_estimators=140, gbm_rounds=300).fit(X, y)
    feature_fn = lambda texts: f.full_feature_stack(list(texts), use_svd=True, svd_dim=80)
    metrics = eng.run_protocol(
        name=EXP_NAME, exp_id=EXP_ID, output_dir=output_dir,
        feature_fn=feature_fn, model_fn=model_fn,
        n_splits=5, seed=SEED, targets=TARGETS, data_dir=data_dir, smoke=smoke,
    )
    print("mean OOF R2 =", metrics.get("mean_r2"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs_and_logs/output/" + EXP_NAME, help="output directory")
    parser.add_argument("--smoke", action="store_true", help="fast smoke mode")
    parser.add_argument("--data-dir", default=None, help="official Dataset dir")
    args = parser.parse_args()
    metrics = run_experiment(Path(args.output), smoke=args.smoke,
                             data_dir=Path(args.data_dir) if args.data_dir else None)
    print(json.dumps({"exp_id": EXP_ID, "mean_r2": metrics.get("mean_r2"),
                      "per_target": metrics.get("per_target", {})}, indent=2))


if __name__ == "__main__":
    main()
