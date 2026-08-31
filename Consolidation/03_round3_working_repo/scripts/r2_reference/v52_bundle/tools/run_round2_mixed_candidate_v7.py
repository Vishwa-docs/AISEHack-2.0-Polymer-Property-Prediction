#!/usr/bin/env python3
"""Local clean runner for the v7 candidate library; not embedded in the notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from round2_mixed_candidate_v7 import run_candidate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("candidate run directory must begin with protocol.json only")
    tool_root = root / "tools"
    source_hashes = {
        "candidate_script": sha256_file(tool_root / "round2_mixed_candidate_v7.py"),
        "reference_module": sha256_file(tool_root / "initial_reference_pipeline.py"),
        "ei_route_module": sha256_file(tool_root / "round2_ei_scaffold_abstaining_gap_identity_v4_portable.py"),
        "eea_route_module": sha256_file(tool_root / "round2_eea_scaffold_abstaining_gap_identity_v7_portable.py"),
        "metric_plumbing": sha256_file(tool_root / "round2_eea_cross_target_oof_residual_stack.py"),
    }
    result = run_candidate(
        args.data_dir,
        run_dir,
        output,
        root_override=root,
        source_hashes_override=source_hashes,
    )
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "rows": result["submission"]["rows"],
        "mean_candidate_r2": result["mean_candidate_r2"],
        "mean_gain": result["mean_gain"],
    }, indent=2))


if __name__ == "__main__":
    main()
