#!/usr/bin/env python3
"""Bounded fold-local Eea character n-gram Ridge diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "eea"
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def nearest_to_train(left: list[Any], right: list[Any]) -> np.ndarray:
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, right)) for fp in left], dtype=np.float64)


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray, seed: int = 2026) -> float:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(500):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in selected])
        if len(rows) < 2 or np.var(y[rows]) <= 1.0e-15:
            continue
        values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"Pre-created protocol directory is required: {run_dir}")
    existing = {path.name for path in run_dir.iterdir()}
    if existing - {"protocol.json"}:
        raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()

    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, TARGET)
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    groups = frame["canonical"].to_numpy(object)
    smiles = frame["canonical"].astype(str).to_numpy(object)
    indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    folds = folds_for(groups)
    baseline = np.full(len(y), np.nan)
    candidate = np.full(len(y), np.nan)
    fold_rows: list[dict[str, Any]] = []
    vectorizer_features: list[int] = []

    weights = c001_report["validation"]["target_reports"][TARGET]["blend_weights"]
    blend_weights = np.asarray([weights[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)
    intercept = float(c001_report["validation"]["target_reports"][TARGET]["blend_intercept"])
    target_index = reference.TARGETS.index(TARGET)
    for fold in range(5):
        train_rows = np.flatnonzero(folds != fold)
        validation_rows = np.flatnonzero(folds == fold)
        base = reference.predict_base_models(
            target_dense,
            sparse_parts,
            fingerprints,
            y_global,
            indices[train_rows],
            indices[validation_rows],
            reference.DEFAULT_CONFIG,
            TARGET,
        )
        baseline[validation_rows] = base @ blend_weights + intercept
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 7),
            lowercase=False,
            sublinear_tf=True,
            max_features=65536,
            dtype=np.float64,
        )
        text_train = vectorizer.fit_transform(smiles[train_rows])
        text_validation = vectorizer.transform(smiles[validation_rows])
        dense_train, dense_validation, dense_train_scaled, dense_validation_scaled = reference.fit_dense_preprocessor(
            target_dense,
            indices[train_rows],
            indices[validation_rows],
            absolute_limit=float(reference.DEFAULT_CONFIG["dense_abs_limit"]),
        )
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(dense_train)
        scaled_validation = scaler.transform(dense_validation)
        matrix_train = sparse.hstack([text_train, sparse.csr_matrix(scaled_train)], format="csr")
        matrix_validation = sparse.hstack([text_validation, sparse.csr_matrix(scaled_validation)], format="csr")
        vectorizer_features.append(int(matrix_train.shape[1]))
        model = Ridge(alpha=10.0, solver="lsqr", max_iter=5000, tol=1.0e-4)
        model.fit(matrix_train, y[train_rows])
        prediction = reference.clip_prediction(y[train_rows], model.predict(matrix_validation))
        candidate[validation_rows] = prediction
        base_r2 = float(r2_score(y[validation_rows], baseline[validation_rows]))
        candidate_r2 = float(r2_score(y[validation_rows], prediction))
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation_rows)),
            "baseline_r2": base_r2,
            "candidate_r2": candidate_r2,
            "delta_r2": candidate_r2 - base_r2,
        })

    nearest = np.empty(len(y), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        nearest[validation] = nearest_to_train(
            [fingerprints[indices[row]] for row in validation],
            [fingerprints[indices[row]] for row in training],
        )
    auxiliary = np.sum(cross_available[indices], axis=1) - cross_available[indices, target_index] > 0
    low_similarity = {}
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            low_similarity[name] = {"rows": int(np.sum(selected)), "delta_r2": value}
    availability = {}
    for name, selected in (("available_other_property", auxiliary), ("missing_other_property", ~auxiliary)):
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            availability[name] = {"rows": int(np.sum(selected)), "delta_r2": value}
    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    delta_r2 = candidate_r2 - baseline_r2
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    panel_values = [value["delta_r2"] for value in (*low_similarity.values(), *availability.values())]
    min_panel = min(panel_values) if panel_values else None
    passed = bool(delta_r2 >= 0.01 and positive_folds >= 4 and bootstrap > 0.0 and (min_panel is None or min_panel >= 0.0))
    audit = {
        "schema_version": "ppp.round2.eea-char-ngram-ridge-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C024-20260803-1940-nested-eps-graph-route",
        "official_inputs": inputs,
        "target": TARGET,
        "rows": int(len(y)),
        "baseline_r2": baseline_r2,
        "candidate_r2": candidate_r2,
        "delta_r2": delta_r2,
        "positive_folds": positive_folds,
        "group_r2_bootstrap_lower": bootstrap,
        "folds": fold_rows,
        "vectorizer_features_per_fold": vectorizer_features,
        "low_similarity": low_similarity,
        "availability": availability,
        "min_panel_delta": min_panel,
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({
        "canonical": groups,
        "fold": folds,
        "nearest_tanimoto": nearest,
        "has_other_property": auxiliary,
        "y": y,
        "baseline": baseline,
        "candidate": candidate,
    }).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.eea-char-ngram-ridge.v1",
        "seed": 2026,
        "folds": 5,
        "target": TARGET,
        "ngram_range": [2, 7],
        "max_features": 65536,
        "ridge_alpha": 10.0,
        "official_inputs": inputs,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C025 Eea character n-gram Ridge\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(
        "stage=official_input_hashes_pass\nstage=fold_local_tfidf\nstage=corrected_r2_bootstrap\nstage=availability_and_similarity_panels\n"
        f"decision={audit['decision']}\n",
        encoding="utf-8",
    )
    manifest_paths = [run_dir / name for name in (
        "config.json", "environment.txt", "oof.csv", "folds.csv", "metrics.json",
        "decision.md", "command.txt", "run.log", "protocol.json",
    )]
    (run_dir / "artifact_manifest.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "experiment_id": run_dir.name,
        "decision": audit["decision"],
        "baseline_r2": baseline_r2,
        "candidate_r2": candidate_r2,
        "delta_r2": delta_r2,
        "positive_folds": positive_folds,
        "group_r2_bootstrap_lower": bootstrap,
        "low_similarity": low_similarity,
        "availability": availability,
        "elapsed_seconds": audit["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
