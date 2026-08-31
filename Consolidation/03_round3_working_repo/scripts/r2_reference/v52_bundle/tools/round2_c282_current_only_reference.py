#!/usr/bin/env python3
"""C282 current-only reference candidate.

This is the Round 2 initial-reference model family rerun with archive labels
removed from the label pool. It reads only official current train/test files for
model fitting and inference. The archive file is not read.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

import initial_reference_pipeline as reference


def load_current_only_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = {
        "train.csv": data_dir / "train.csv",
        "test.csv": data_dir / "test.csv",
    }
    hashes = {name: reference.sha256_file(path) for name, path in paths.items()}
    for name, expected in {
        "train.csv": reference.EXPECTED_HASHES["train.csv"],
        "test.csv": reference.EXPECTED_HASHES["test.csv"],
    }.items():
        if hashes[name] != expected:
            raise RuntimeError(f"Official input hash mismatch for {name}: {hashes[name]}")
    train = pd.read_csv(paths["train.csv"])
    test = pd.read_csv(paths["test.csv"])
    if list(train.columns) != ["smiles", "target", "target_type"]:
        raise RuntimeError("Unexpected current train schema")
    if list(test.columns) != ["id", "smiles", "target_type"]:
        raise RuntimeError("Unexpected current test schema")
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError("Unexpected official current row count")
    for frame in (train, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
        frame["canonical"] = [reference.canonicalize(value) for value in frame["smiles"]]
    if set(train["target_type"]) != set(reference.TARGETS) or set(test["target_type"]) != set(reference.TARGETS):
        raise RuntimeError("Unexpected target set")
    if test["id"].duplicated().any() or not np.array_equal(test["id"].to_numpy(), np.arange(1, 4941)):
        raise RuntimeError("Test IDs are not unique sequential IDs 1..4940")
    if not np.isfinite(train["target"].to_numpy(float)).all():
        raise RuntimeError("Current train contains a non-finite target")
    manifest = {
        name: {"path": str(path), "sha256": hashes[name], "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    return train, test, manifest


def package_manifest(run_dir: Path, paths: list[Path]) -> None:
    lines = [f"{reference.sha256_file(path)}  {path.relative_to(run_dir)}" for path in paths]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_current_only_reference(
    data_dir: str | Path,
    output_path: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    configuration = dict(reference.DEFAULT_CONFIG)
    if config:
        configuration.update(config)
    np.random.seed(int(configuration["seed"]))
    data_path = Path(data_dir).resolve()
    runtime = Path(run_dir).resolve()
    output = Path(output_path).resolve()
    if runtime.exists():
        raise RuntimeError(f"Refusing to reuse existing run directory: {runtime}")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {output}")
    runtime.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)

    train, test, inputs = load_current_only_inputs(data_path)
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(configuration["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(configuration["morgan_bits"])),
        reference.text_matrix(keys, int(configuration["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(configuration["morgan_bits"]))
    detail, oof, model_report = reference.fit_targets(
        pooled,
        test,
        keys,
        dense_base,
        cross_values,
        cross_available,
        sparse_parts,
        fingerprints,
        configuration,
    )
    final_detail, override_report = reference.apply_official_overrides(detail, test, raw_labels)
    submission = final_detail[["id", "target"]].copy()
    if len(submission) != len(test) or not submission["id"].equals(test["id"]):
        raise RuntimeError("Submission row order differs from official test")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("Submission contains duplicate IDs or non-finite targets")
    submission.to_csv(output, index=False)

    detail_path = runtime / "test_predictions_detail.csv"
    oof_path = runtime / "oof_predictions.csv"
    config_path = runtime / "config.json"
    environment_path = runtime / "environment.txt"
    report_path = runtime / "report.json"
    command_path = runtime / "command.txt"

    final_detail.drop(columns=["smiles", "canonical"]).to_csv(detail_path, index=False)
    oof.to_csv(oof_path, index=False)
    reference.write_json(config_path, configuration)
    environment_path.write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"rdkit={Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema_version": "ppp.round2.c282.current-only-reference.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "official current-only target-specific classical ensemble with cross-property covariates",
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": configuration,
        "config_sha256": reference.canonical_json_hash(configuration),
        "inputs": inputs,
        "rows": {
            "current_train": int(len(train)),
            "archive_train_used": 0,
            "raw_label_pool": int(len(raw_labels)),
            "canonical_model_rows": int(len(pooled)),
            "test": int(len(test)),
            "unique_feature_structures": int(len(keys)),
        },
        "features": {
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "cross_property_values": len(reference.TARGETS) - 1,
            "cross_property_availability": len(reference.TARGETS) - 1,
            "morgan_count_radii": [2, 3],
            "morgan_bits": int(configuration["morgan_bits"]),
            "character_ngrams": [2, 7],
            "character_hash_features": int(configuration["text_features"]),
            "dense_abs_limit": float(configuration["dense_abs_limit"]),
        },
        "validation": model_report,
        "official_overrides": override_report,
        "submission": {
            "path": str(output),
            "rows": int(len(submission)),
            "sha256": reference.sha256_file(output),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    reference.write_json(report_path, report)
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    package_manifest(runtime, [config_path, environment_path, detail_path, oof_path, report_path, command_path])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = run_current_only_reference(args.data_dir, args.output, args.run_dir)
    print(
        json.dumps(
            {
                "submission": result["submission"],
                "mean_oof_r2": result["validation"]["mean_selected_oof_r2"],
                "official_overrides": result["official_overrides"]["total_overrides"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
