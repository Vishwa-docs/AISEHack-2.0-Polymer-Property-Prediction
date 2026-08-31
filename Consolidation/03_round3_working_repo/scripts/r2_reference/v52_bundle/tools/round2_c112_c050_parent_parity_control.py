#!/usr/bin/env python3
"""C112 audit-only replay of the historical C050 parent.

This does not train a new model or read local_eval material.  It rebuilds the
portable C050 source path from official inputs and compares the regenerated
parent OOF/test values with the canonical C050-v7 audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_ei_scaffold_abstaining_gap_identity_v4_portable as ei_route
import round2_eea_scaffold_abstaining_gap_identity_v7_portable as eea_route
import round2_mixed_candidate_v7 as mixed


TARGETS = tuple(reference.TARGETS)
SPECIAL = {"ei": ei_route, "eea": eea_route}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rebuild_parent(root: Path, data_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, test, archive, inputs = reference.load_inputs(data_dir)
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "mixed_candidate": True, "special_targets": list(mixed.SPECIAL_TARGETS)})
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(config["morgan_bits"]))
    parent_detail, parent_oof, parent_report = reference.fit_targets(
        pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
    )
    detail = parent_detail[["id", "target_type", "model_prediction"]].copy()
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        if target in SPECIAL:
            route = mixed.specialized_target(
                target, pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
            )
            special_oof, special_test, _ = route
            replacement = special_test.set_index("id")["target"]
            mask = detail["target_type"].to_numpy(object) == target
            detail.loc[mask, "model_prediction"] = detail.loc[mask, "id"].map(replacement).astype(float).to_numpy()
            oof_parts.append(pd.DataFrame({
                "canonical": special_oof["canonical"].astype(str),
                "target_type": target,
                "target": special_oof["target"].astype(float),
                "parent_prediction": special_oof["candidate"].astype(float),
            }))
        else:
            rows = parent_oof[parent_oof["target_type"] == target].copy()
            oof_parts.append(pd.DataFrame({
                "canonical": rows["canonical"].astype(str),
                "target_type": target,
                "target": rows["target"].astype(float),
                "parent_prediction": rows["prediction"].astype(float),
            }))
    final_detail, override_report = reference.apply_official_overrides(detail, test, raw_labels)
    predictions = final_detail[["id", "target"]].copy()
    oof = pd.concat(oof_parts, ignore_index=True)
    predictions.to_csv(run_dir / "parent_replay_predictions.csv", index=False)
    oof.to_csv(run_dir / "parent_replay_oof.csv", index=False)
    return predictions, oof, {"train": train, "test": test, "archive": archive, "inputs": inputs, "parent_report": parent_report, "override_report": override_report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", default="experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    canonical_dir = (root / args.canonical_run).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only C112 directory required")
    started = time.time()
    predictions, oof, context = rebuild_parent(root, (root / args.data_dir).resolve(), run_dir)
    canonical_predictions = pd.read_csv(canonical_dir / "predictions.csv")
    canonical_oof = pd.read_csv(canonical_dir / "oof_predictions.csv")
    if not predictions["id"].equals(canonical_predictions["id"]):
        raise RuntimeError("canonical and replay test IDs/order differ")
    if len(oof) != len(canonical_oof):
        raise RuntimeError("canonical and replay OOF row counts differ")
    replay_oof = oof.sort_values(["target_type", "canonical"]).reset_index(drop=True)
    reference_oof = canonical_oof[["canonical", "target_type", "target", "candidate_prediction"]].rename(columns={"candidate_prediction": "canonical_prediction"}).sort_values(["target_type", "canonical"]).reset_index(drop=True)
    if not replay_oof[["canonical", "target_type", "target"]].astype(str).equals(reference_oof[["canonical", "target_type", "target"]].astype(str)):
        raise RuntimeError("canonical and replay OOF row identity differs")
    oof_delta = np.abs(replay_oof["parent_prediction"].to_numpy(float) - reference_oof["canonical_prediction"].to_numpy(float))
    test_delta = np.abs(predictions["target"].to_numpy(float) - canonical_predictions["target"].to_numpy(float))
    parity = {
        "oof_rows": int(len(oof_delta)),
        "test_rows": int(len(test_delta)),
        "oof_max_abs": float(np.max(oof_delta)),
        "test_max_abs": float(np.max(test_delta)),
        "oof_tolerance": 1.0e-12,
        "test_tolerance": 1.0e-12,
        "oof_pass": bool(np.max(oof_delta) <= 1.0e-12),
        "test_pass": bool(np.max(test_delta) <= 1.0e-12),
        "pass": bool(np.max(oof_delta) <= 1.0e-12 and np.max(test_delta) <= 1.0e-12),
    }
    source_paths = {
        "runner": root / "tools" / "round2_c112_c050_parent_parity_control.py",
        "reference": root / "tools" / "initial_reference_pipeline.py",
        "mixed_parent": root / "tools" / "round2_mixed_candidate_v7.py",
        "metric_plumbing": root / "tools" / "round2_eea_cross_target_oof_residual_stack.py",
        "ei_route": root / "tools" / "round2_ei_scaffold_abstaining_gap_identity_v4_portable.py",
        "eea_route": root / "tools" / "round2_eea_scaffold_abstaining_gap_identity_v7_portable.py",
    }
    report = {
        "schema_version": "ppp.round2.c112.c050-parent-parity-control.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "local_eval_read": False,
        "external_label_file_read": False,
        "kaggle_compute": False,
        "canonical_parent": str(canonical_dir.relative_to(root)),
        "official_inputs": context["inputs"],
        "parity": parity,
        "decision": "parity_pass" if parity["pass"] else "blocked_parent_nonreproducible",
        "parent_replay_oof_max_abs": parity["oof_max_abs"],
        "parent_replay_test_max_abs": parity["test_max_abs"],
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "rdkit": reference.Chem.rdBase.rdkitVersion, "platform": platform.platform()},
    }
    write_json(run_dir / "parity_report.json", report)
    (run_dir / "environment.txt").write_text("\n".join(f"{key}={value}" for key, value in report["environment"].items()) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. OOF max absolute difference `{parity['oof_max_abs']:.16g}`; test max absolute difference `{parity['test_max_abs']:.16g}`. Audit-only; no candidate or local_eval action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{digest}  SOURCE tools/{name}.py" for name, digest in report["source_hashes"].items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "oof_max_abs": parity["oof_max_abs"], "test_max_abs": parity["test_max_abs"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
