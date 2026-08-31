#!/usr/bin/env python3
"""Egb specialization of the nested cross-target residual diagnostic.

The implementation is shared only as generic metric plumbing; this wrapper
sets a new target/auxiliary set and masks every raw cross-property value in the
parent reference before execution.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import round2_eea_cross_target_oof_residual_stack as implementation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_target_dense(base_dense, cross_values, cross_available, target):
    return implementation.np.hstack([
        base_dense,
        implementation.np.full_like(cross_values, implementation.np.nan),
        implementation.np.zeros_like(cross_available),
    ]).astype(implementation.np.float64, copy=False)


def main() -> None:
    implementation.TARGET = "egb"
    implementation.AUXILIARY = ("egc", "eea", "nc", "eps", "ei")
    implementation.reference.target_dense_features = masked_target_dense
    implementation.main()
    arguments = sys.argv[1:]
    run_dir = None
    for index, argument in enumerate(arguments):
        if argument == "--run-dir" and index + 1 < len(arguments):
            run_dir = Path(arguments[index + 1]).resolve()
            break
    if run_dir is None:
        raise RuntimeError("--run-dir is required")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    wrapper_path = Path(__file__).resolve()
    implementation_path = Path(implementation.__file__).resolve()
    metrics["schema_version"] = "ppp.round2.egb-cross-target-oof-residual-stack-run.v1"
    metrics["target"] = "egb"
    metrics["auxiliary_targets"] = list(implementation.AUXILIARY)
    metrics["parent"] = "R2-C032-20260803-2330-eps-structure-key-oof-target-encoding"
    metrics["source_hashes"]["wrapper"] = sha256_file(wrapper_path)
    metrics["source_hashes"]["shared_implementation"] = sha256_file(implementation_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = "ppp.round2.egb-cross-target-oof-residual-stack.v1"
    config["target"] = "egb"
    config["auxiliary_targets"] = list(implementation.AUXILIARY)
    config["source_hashes"] = metrics["source_hashes"]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C033 Egb cross-target OOF residual stack\n\nDecision: **{metrics['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n",
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        f"experiment_id={run_dir.name}\n"
        "target=egb\n"
        f"nested_parent_r2={metrics['baseline_r2_nested_parent']:.12f}\n"
        f"candidate_r2={metrics['candidate_r2_cross_target_residual']:.12f}\n"
        f"delta_r2={metrics['delta_r2']:.12f}\n"
        f"pass={metrics['pass']}\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([
        f"{metrics['source_hashes']['wrapper']}  SOURCE tools/round2_egb_cross_target_oof_residual_stack.py",
        f"{metrics['source_hashes']['shared_implementation']}  SOURCE tools/round2_eea_cross_target_oof_residual_stack.py",
        f"{metrics['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
