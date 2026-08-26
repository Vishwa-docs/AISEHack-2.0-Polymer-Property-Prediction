#!/usr/bin/env python3
"""Eea availability/similarity-gated specialization of the C031 diagnostic.

The shared implementation is reused only for deterministic metric plumbing.
This wrapper masks raw cross-property values from the parent and applies the
pre-registered correction gate to each held-out row.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import round2_eea_cross_target_oof_residual_stack as implementation


ROUTE_LOG: list[dict[str, int]] = []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_target_dense(base_dense, cross_values, cross_available, target):
    return np.hstack([
        base_dense,
        np.full_like(cross_values, np.nan),
        np.zeros_like(cross_available),
    ]).astype(np.float64, copy=False)


def gated_nested_split(
    y, groups, outer_train, outer_validation, dense, sparse_parts, fingerprints,
    y_global, global_indices, cross_available, aux_info, deterministic,
):
    result = implementation._ORIGINAL_NESTED_SPLIT(
        y, groups, outer_train, outer_validation, dense, sparse_parts,
        fingerprints, y_global, global_indices, cross_available, aux_info,
        deterministic,
    )
    aux_columns = [implementation.reference.TARGETS.index(target) for target in implementation.AUXILIARY]
    validation_global = global_indices[outer_validation]
    training_global = global_indices[outer_train]
    available_count = np.sum(cross_available[validation_global][:, aux_columns], axis=1)
    nearest = implementation.nearest_to_train(
        [fingerprints[index] for index in validation_global],
        [fingerprints[index] for index in training_global],
    )
    eligible = (available_count >= 2) & (nearest >= 0.30)
    candidate = np.asarray(result["candidate"], dtype=np.float64).copy()
    candidate[~eligible] = np.asarray(result["parent"], dtype=np.float64)[~eligible]
    result["candidate"] = candidate
    ROUTE_LOG.append({
        "rows": int(len(eligible)),
        "eligible_rows": int(np.sum(eligible)),
        "low_similarity_rows": int(np.sum(nearest < 0.30)),
        "low_auxiliary_rows": int(np.sum(available_count < 2)),
        "low_similarity_changed_rows": int(np.sum((nearest < 0.30) & (np.abs(candidate - result["parent"]) > 1.0e-12))),
        "low_auxiliary_changed_rows": int(np.sum((available_count < 2) & (np.abs(candidate - result["parent"]) > 1.0e-12))),
    })
    return result


def main() -> None:
    implementation.TARGET = "eea"
    implementation.AUXILIARY = ("egb", "egc", "nc", "eps", "ei")
    implementation.reference.target_dense_features = masked_target_dense
    implementation._ORIGINAL_NESTED_SPLIT = implementation.nested_split
    implementation.nested_split = gated_nested_split
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
    metrics["schema_version"] = "ppp.round2.eea-availability-gated-residual-stack-run.v1"
    metrics["parent"] = "R2-C033-20260803-2330-egb-cross-target-oof-residual-stack"
    metrics["target"] = "eea"
    metrics["auxiliary_targets"] = list(implementation.AUXILIARY)
    metrics["route_definition"] = {
        "minimum_available_auxiliary_count": 2,
        "minimum_nearest_tanimoto": 0.30,
        "correction_strength": 0.5,
        "route_log": ROUTE_LOG,
        "low_similarity_changed_rows": int(sum(item["low_similarity_changed_rows"] for item in ROUTE_LOG)),
        "low_auxiliary_changed_rows": int(sum(item["low_auxiliary_changed_rows"] for item in ROUTE_LOG)),
    }
    metrics["source_hashes"]["wrapper"] = sha256_file(wrapper_path)
    metrics["source_hashes"]["shared_implementation"] = sha256_file(implementation_path)
    panel_values = []
    for value in metrics.get("panels", {}).values():
        items = value.values() if isinstance(value, dict) and "delta_r2" not in value else [value]
        for item in items:
            delta = item.get("delta_r2")
            if delta is not None:
                panel_values.append(float(delta))
    panel_values.extend(float(item["delta_r2"]) for item in metrics.get("scaffold_holdout", {}).values())
    metrics["min_panel_delta"] = min(panel_values) if panel_values else None
    metrics["panel_incomplete"] = False
    metrics["pass"] = bool(
        metrics["delta_r2"] >= 0.01
        and metrics["positive_outer_folds"] >= 4
        and metrics["group_r2_bootstrap_lower"] > 0.0
        and (metrics["min_panel_delta"] is None or metrics["min_panel_delta"] >= -0.003)
        and metrics["route_definition"]["low_similarity_changed_rows"] == 0
        and metrics["route_definition"]["low_auxiliary_changed_rows"] == 0
    )
    metrics["decision"] = "component_pass" if metrics["pass"] else "rejected_component_gate"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = "ppp.round2.eea-availability-gated-residual-stack.v1"
    config["target"] = "eea"
    config["auxiliary_targets"] = list(implementation.AUXILIARY)
    config["route"] = {"minimum_available_auxiliary_count": 2, "minimum_nearest_tanimoto": 0.30}
    config["source_hashes"] = metrics["source_hashes"]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C034 Eea availability-gated residual stack\n\nDecision: **{metrics['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n",
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        f"experiment_id={run_dir.name}\n"
        "target=eea\n"
        f"nested_parent_r2={metrics['baseline_r2_nested_parent']:.12f}\n"
        f"candidate_r2={metrics['candidate_r2_cross_target_residual']:.12f}\n"
        f"delta_r2={metrics['delta_r2']:.12f}\n"
        f"low_similarity_changed_rows={metrics['route_definition']['low_similarity_changed_rows']}\n"
        f"low_auxiliary_changed_rows={metrics['route_definition']['low_auxiliary_changed_rows']}\n"
        f"pass={metrics['pass']}\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([
        f"{metrics['source_hashes']['wrapper']}  SOURCE tools/round2_eea_availability_gated_residual_stack.py",
        f"{metrics['source_hashes']['shared_implementation']}  SOURCE tools/round2_eea_cross_target_oof_residual_stack.py",
        f"{metrics['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
