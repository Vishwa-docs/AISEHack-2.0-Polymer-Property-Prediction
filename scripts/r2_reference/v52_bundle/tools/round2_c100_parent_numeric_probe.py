#!/usr/bin/env python3
"""Temporary-style audit helper: write local parent arrays for tolerance probing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import round2_c098_target_routed_qspr_full as c098


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = c098.parent_bundle(root, (root / args.data_dir).resolve())
    for target in c098.TARGETS:
        info = bundle["target_info"][target]
        ids = bundle["test_detail"].loc[bundle["test_detail"]["target_type"] == target, "id"].to_numpy()
        test_values = bundle["test_detail"].set_index("id").loc[ids, "model_prediction"].to_numpy(float)
        np.save(output / f"{target}_oof.npy", np.asarray(info["parent"], dtype=np.float64))
        np.save(output / f"{target}_test.npy", test_values)


if __name__ == "__main__":
    main()
