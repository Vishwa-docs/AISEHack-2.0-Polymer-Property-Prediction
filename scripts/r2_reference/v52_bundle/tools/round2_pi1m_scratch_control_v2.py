#!/usr/bin/env python3
"""Bounded PI1M character-vocabulary control for Round 2."""

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
ARMS = ("pi1m_tfidf", "official_tfidf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    result = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in training]
        for index in validation:
            result[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return result


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(r2_score(y, pred))


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
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    pi1m_path = data_dir / "PI1M.csv"
    pi1m_sha256 = sha256_file(pi1m_path)
    pi1m_smiles = pd.read_csv(pi1m_path, usecols=["SMILES"], nrows=200000)["SMILES"].astype(str).tolist()
    official_smiles = pd.concat([train["smiles"], test["smiles"], archive["smiles"]], ignore_index=True).astype(str).tolist()
    pi_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=8192, min_df=3, lowercase=False, sublinear_tf=True, dtype=np.float64)
    official_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=8192, min_df=2, lowercase=False, sublinear_tf=True, dtype=np.float64)
    pi_vectorizer.fit(pi1m_smiles)
    official_vectorizer.fit(official_smiles)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical])
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    c001_sparse = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    c001_fingerprints = reference.morgan_bits(molecules, 2, 4096)
    reference_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        indices = np.asarray([key_to_index[key] for key in frame["canonical"]], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[indices] = y
        folds = folds_for(frame["canonical"].to_numpy(object))
        pi_x = pi_vectorizer.transform(frame["canonical"].astype(str))
        official_x = official_vectorizer.transform(frame["canonical"].astype(str))
        dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
        weights_map = reference_report["validation"]["target_reports"][target]["blend_weights"]
        weights = np.asarray([weights_map[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=float)
        baseline = np.full(len(y), np.nan)
        candidates = np.full((len(y), 2), np.nan)
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            base = reference.predict_base_models(dense, c001_sparse, c001_fingerprints, y_global, indices[train_rows], indices[validation_rows], reference.DEFAULT_CONFIG, target)
            baseline[validation_rows] = base @ weights
            for arm, matrix in enumerate((pi_x, official_x)):
                model = Ridge(alpha=10.0).fit(matrix[train_rows], y[train_rows])
                candidates[validation_rows, arm] = model.predict(matrix[validation_rows])
        nearest = nearest_similarity([c001_fingerprints[index] for index in indices], folds)
        report = {"rows": int(len(y)), "baseline_r2": r2(y, baseline), "arms": {}, "folds": [], "low_similarity": {}}
        for arm, name in enumerate(ARMS):
            report["arms"][name] = {"r2": r2(y, candidates[:, arm]), "delta_r2": r2(y, candidates[:, arm]) - report["baseline_r2"]}
        for fold in range(5):
            selected = folds == fold
            fold_report = {"fold": fold, "baseline_r2": r2(y[selected], baseline[selected])}
            for arm, name in enumerate(ARMS):
                fold_report[f"{name}_r2"] = r2(y[selected], candidates[selected, arm])
                fold_report[f"{name}_delta_r2"] = fold_report[f"{name}_r2"] - fold_report["baseline_r2"]
            report["folds"].append(fold_report)
        for name_bin, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
            selected = (nearest >= lower) & (nearest < upper)
            if int(np.sum(selected)) < 5:
                continue
            baseline_r2 = r2(y[selected], baseline[selected])
            report["low_similarity"][name_bin] = {"rows": int(np.sum(selected)), "baseline_r2": baseline_r2, "arms": {}}
            for arm, name in enumerate(ARMS):
                report["low_similarity"][name_bin]["arms"][name] = {
                    f"{name}_r2": r2(y[selected], candidates[selected, arm]),
                    f"{name}_delta_r2": r2(y[selected], candidates[selected, arm]) - baseline_r2,
                }
        report["pi1m_minus_official_control"] = report["arms"]["pi1m_tfidf"]["delta_r2"] - report["arms"]["official_tfidf"]["delta_r2"]
        reports[target] = report
        for index, name in enumerate(("c001", *ARMS)):
            prediction = baseline if index == 0 else candidates[:, index - 1]
            rows.append({"target": target, "arm": name, "r2": r2(y, prediction), "rows": int(len(y))})
    passing_targets = []
    for target, report in reports.items():
        best = max(ARMS, key=lambda name: report["arms"][name]["r2"])
        best_folds = [fold[f"{best}_delta_r2"] for fold in report["folds"]]
        low = [value["arms"][best][f"{best}_delta_r2"] for value in report["low_similarity"].values()]
        report["selected_arm"] = best
        report["positive_folds"] = int(sum(delta > 0 for delta in best_folds))
        report["min_low_similarity_delta"] = min(low) if low else None
        if report["arms"][best]["delta_r2"] >= 0.01 and report["positive_folds"] >= 4 and report["min_low_similarity_delta"] is not None and report["min_low_similarity_delta"] >= 0 and report["pi1m_minus_official_control"] >= 0:
            passing_targets.append(target)
    audit = {"schema_version": "ppp.round2.pi1m-scratch-control-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C009-20260803-1734-nc-size-specialist", "official_inputs": inputs, "pi1m_sha256": pi1m_sha256, "pi1m_rows_used": 200000, "targets": reports, "passing_targets": passing_targets, "decision": "component_pass" if passing_targets else "rejected_component_gate", "elapsed_seconds": float(time.time() - started)}
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.pi1m-scratch-control.v1", "seed": 2026, "folds": 5, "arms": list(ARMS), "pi1m_rows_used": 200000, "pi1m_sha256": pi1m_sha256, "max_features": 8192, "ngram_range": [2, 6], "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C010 PI1M scratch control decision\n\nDecision: **{audit['decision']}**. No candidate changed in this diagnostic.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "metrics.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "passing_targets": passing_targets, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
