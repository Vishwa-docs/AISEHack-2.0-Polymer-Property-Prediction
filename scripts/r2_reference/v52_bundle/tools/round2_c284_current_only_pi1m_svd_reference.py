#!/usr/bin/env python3
"""C284 current-only PI1M-SVD reference candidate.

Official unlabeled PI1M SMILES are used only to learn a from-scratch,
label-free character n-gram SVD representation. The supervised target models
use current Round 2 train labels only. No archive labels, local_eval files,
pretrained weights, cached embeddings, or prior predictions are read.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


DEFAULT_CONFIG: dict[str, Any] = {
    **reference.DEFAULT_CONFIG,
    "pi1m_limit": 100_000,
    "pi1m_hash_features": 32_768,
    "pi1m_svd_components": 96,
    "pi1m_ngram_min": 2,
    "pi1m_ngram_max": 7,
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_pi1m_smiles(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["SMILES"])
    values = frame["SMILES"].dropna().astype(str).tolist()
    if len(values) < 1000:
        raise RuntimeError("PI1M file unexpectedly small")
    return values


def hash_ranked_unique(values: list[str], limit: int) -> list[str]:
    unique = sorted(
        set(values),
        key=lambda value: hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
    )
    return unique[:limit]


def pi1m_svd_features(
    *,
    keys: list[str],
    pi1m_path: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    pi1m = read_pi1m_smiles(pi1m_path)
    corpus = hash_ranked_unique(pi1m, int(config["pi1m_limit"]))
    corpus_plus_keys = corpus + list(keys)
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(int(config["pi1m_ngram_min"]), int(config["pi1m_ngram_max"])),
        n_features=int(config["pi1m_hash_features"]),
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float64,
    )
    hashed = vectorizer.transform(corpus_plus_keys).tocsr()
    max_components = min(hashed.shape[0] - 1, hashed.shape[1] - 1)
    n_components = min(int(config["pi1m_svd_components"]), max_components)
    if n_components < 8:
        raise RuntimeError(f"PI1M SVD component count too small: {n_components}")
    svd = TruncatedSVD(n_components=n_components, random_state=int(config["seed"]))
    embedding = svd.fit_transform(hashed).astype(np.float64, copy=False)
    key_embedding = embedding[-len(keys) :]
    report = {
        "source": "official ppp-round-2/PI1M.csv plus official current train/test SMILES",
        "pi1m_sha256": reference.sha256_file(pi1m_path),
        "pi1m_rows_read": int(len(pi1m)),
        "pi1m_hash_ranked_unique_rows_used": int(len(corpus)),
        "official_key_rows": int(len(keys)),
        "hash_features": int(config["pi1m_hash_features"]),
        "ngram_range": [int(config["pi1m_ngram_min"]), int(config["pi1m_ngram_max"])],
        "svd_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "labels_used_for_representation": False,
        "pretrained_weights": False,
    }
    return key_embedding, report


def package_manifest(run_dir: Path, paths: list[Path]) -> None:
    lines = [f"{reference.sha256_file(path)}  {path.relative_to(run_dir)}" for path in paths]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pi1m_svd_reference(
    *,
    data_dir: str | Path,
    output_path: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    configuration = dict(DEFAULT_CONFIG)
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

    protocol = {
        "schema_version": "ppp.round2.c284.current-only-pi1m-svd.protocol.v1",
        "experiment_id": runtime.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "hypothesis": "A from-scratch PI1M character n-gram SVD representation supplies current-only transfer signal beyond C282 descriptors/fingerprints for at least one no-archive target.",
        "changed_factor": "append label-free PI1M SVD dense features to the C282 current-only reference feature set",
        "branch": "without_archive",
        "official_inputs_only": True,
        "archive_labels_used": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "stop_gate": "reject as a component family if no target improves post-freeze transfer against the current no-archive portfolio component for that target",
        "config": configuration,
    }
    write_json(runtime / "protocol.json", protocol)

    train, test, inputs = c282.load_current_only_inputs(data_path)
    pi1m_path = data_path / "PI1M.csv"
    if reference.sha256_file(pi1m_path) != "c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8":
        raise RuntimeError("Official PI1M hash mismatch")
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    pi1m_features, pi1m_report = pi1m_svd_features(keys=keys, pi1m_path=pi1m_path, config=configuration)
    dense_base = np.hstack([descriptor, physical, pi1m_features]).astype(np.float64, copy=False)
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
    write_json(config_path, configuration)
    environment_path.write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"sklearn={__import__('sklearn').__version__}",
                f"rdkit={Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema_version": "ppp.round2.c284.current-only-pi1m-svd.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "current-only C282 reference plus label-free PI1M char-ngram SVD dense features",
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": configuration,
        "config_sha256": reference.canonical_json_hash(configuration),
        "inputs": inputs | {"PI1M.csv": {"path": str(pi1m_path), "sha256": reference.sha256_file(pi1m_path), "bytes": pi1m_path.stat().st_size}},
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
            "pi1m_svd_features": int(pi1m_features.shape[1]),
            "cross_property_values": len(reference.TARGETS) - 1,
            "cross_property_availability": len(reference.TARGETS) - 1,
            "morgan_count_radii": [2, 3],
            "morgan_bits": int(configuration["morgan_bits"]),
            "character_ngrams": [2, 7],
            "character_hash_features": int(configuration["text_features"]),
            "dense_abs_limit": float(configuration["dense_abs_limit"]),
        },
        "pi1m_representation": pi1m_report,
        "validation": model_report,
        "official_overrides": override_report,
        "submission": {
            "path": str(output),
            "rows": int(len(submission)),
            "sha256": reference.sha256_file(output),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(report_path, report)
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    package_manifest(runtime, [runtime / "protocol.json", config_path, environment_path, detail_path, oof_path, report_path, command_path])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pi1m-limit", type=int, default=DEFAULT_CONFIG["pi1m_limit"])
    parser.add_argument("--pi1m-components", type=int, default=DEFAULT_CONFIG["pi1m_svd_components"])
    args = parser.parse_args()
    result = run_pi1m_svd_reference(
        data_dir=args.data_dir,
        output_path=args.output,
        run_dir=args.run_dir,
        config={"pi1m_limit": args.pi1m_limit, "pi1m_svd_components": args.pi1m_components},
    )
    print(
        json.dumps(
            {
                "submission": result["submission"],
                "mean_oof_r2": result["validation"]["mean_selected_oof_r2"],
                "official_overrides": result["official_overrides"]["total_overrides"],
                "pi1m": result["pi1m_representation"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
