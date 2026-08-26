#!/usr/bin/env python3
"""Generate a sanitized parent fingerprint in an independent process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import round2_c098_target_routed_qspr_full as c098


def array_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve()
    bundle = c098.parent_bundle(root, (root / args.data_dir).resolve())
    replay_dir = run_dir / "replay_parent"
    replay_dir.mkdir(parents=True, exist_ok=True)
    targets = {}
    for target in c098.TARGETS:
        info = bundle["target_info"][target]
        ids = bundle["test_detail"].loc[bundle["test_detail"]["target_type"] == target, "id"].to_numpy()
        test_values = bundle["test_detail"].set_index("id").loc[ids, "model_prediction"].to_numpy(float)
        np.save(replay_dir / f"{target}_oof.npy", np.asarray(info["parent"], dtype=np.float64))
        np.save(replay_dir / f"{target}_test.npy", test_values)
        targets[target] = {
            "oof_rows": int(len(info["parent"])),
            "test_rows": int(len(test_values)),
            "oof_sha256": array_hash(info["parent"]),
            "test_sha256": array_hash(test_values),
        }
    payload = {
        "schema_version": "ppp.round2.c100.parent-replay.v1",
        "experiment_id": run_dir.name,
        "source": "independent_process_official_source_rebuild",
        "official_inputs": bundle["inputs"],
        "targets": targets,
        "raw_prediction_rows": False,
        "local_eval_read": False,
        "kaggle_compute": False,
    }
    (run_dir / "pre_run_parent_replay.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "targets": {k: v["oof_sha256"][:12] for k, v in targets.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
