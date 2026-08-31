#!/usr/bin/env python3
"""Evaluate a bounded PI1M from-scratch character representation control."""

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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
MODEL_NAMES = ("pi1m_tfidf_ridge_alpha_10", "official_covariate_tfidf_ridge_alpha_10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    return {
        "rows": int(len(y)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and not np.isclose(np.var(y), 0.0) else None,
        "mae": float(mean_absolute_error(y, pred)) if len(y) else None,
        "rmse": float(np.sqrt(np.mean(np.square(y - pred)))) if len(y) else None,
    }


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("Incomplete fold assignment")
    return result


def load_weights(root: Path, target: str) -> np.ndarray:
    report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    names = ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")
    values = report["validation"]["target_reports"][target]["blend_weights"]
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    output = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in training]
        for index in validation:
            output[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return output


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
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    pi1m_path = data_dir / "PI1M.csv"
    pi1m_hash = sha256_file(pi1m_path)
    pi1m = pd.read_csv(pi1m_path, usecols=["SMILES"], nrows=200000)["SMILES"].astype(str).tolist()
    official_corpus = pd.concat([train["smiles"], test["smiles"], archive["smiles"]], ignore_index=True).astype(str).tolist()
    pi_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=8192, min_df=3, lowercase=False, sublinear_tf=True, dtype=np.float64)
    official_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=8192, min_df=2, lowercase=False, sublinear_tf=True, dtype=np.float64)
    pi_vectorizer.fit(pi1m)
    official_vectorizer.fit(official_corpus)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    dense_descriptor, _ = reference.descriptor_matrix(molecules)
    dense_physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([dense_descriptor, dense_physical])
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    c001_sparse = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    c001_fingerprints = reference.morgan_bits(molecules, 2, 4096)
    c001_predictions = np.full((len(pooled), len(MODEL_NAMES) + 1), np.nan)\n+    metric_rows: list[dict[str, Any]] = []\n+    target_reports: dict[str, Any] = {}\n+    prediction_rows: list[dict[str, Any]] = []\n+    fold_rows: list[dict[str, Any]] = []\n+    for target in TARGETS:\n+        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)\n+        y = frame["target"].to_numpy(float)\n+        target_indices = np.asarray([key_to_index[value] for value in frame["canonical"]], dtype=np.int64)\n+        y_global = np.full(len(keys), np.nan, dtype=np.float64)\n+        y_global[target_indices] = y\n+        folds = folds_for(frame["canonical"].to_numpy(object))\n+        pi_x = pi_vectorizer.transform(frame["canonical"].astype(str))\n+        official_x = official_vectorizer.transform(frame["canonical"].astype(str))\n+        dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)\n+        baseline = np.full(len(y), np.nan)\n+        candidates = np.full((len(y), len(MODEL_NAMES)), np.nan)\n+        for fold in range(5):\n+            train_rows = np.flatnonzero(folds != fold)\n+            validation_rows = np.flatnonzero(folds == fold)\n+            base_parts = reference.predict_base_models(dense, c001_sparse, c001_fingerprints, y_global, target_indices[train_rows], target_indices[validation_rows], reference.DEFAULT_CONFIG, target)\n+            baseline[validation_rows] = base_parts @ load_weights(root, target)\n+            for column, matrix in enumerate((pi_x, official_x)):\n+                model = Ridge(alpha=10.0).fit(matrix[train_rows], y[train_rows])\n+                candidates[validation_rows, column] = reference.clip_prediction(y[train_rows], model.predict(matrix[validation_rows]))\n+            fold_rows.extend({\"target\": target, \"fold\": fold, \"index\": int(index)} for index in validation_rows)\n+        fingerprints = [c001_fingerprints[index] for index in target_indices]\n+        nearest = nearest_similarity(fingerprints, folds)\n+        report: dict[str, Any] = {\"rows\": int(len(y)), \"baseline\": metric(y, baseline), \"models\": {name: metric(y, candidates[:, index]) for index, name in enumerate(MODEL_NAMES)}, \"folds\": [], \"low_similarity\": {}}\n+        for fold in range(5):\n+            selected = folds == fold\n+            row = {\"fold\": fold, \"baseline\": metric(y[selected], baseline[selected])}\n+            for index, name in enumerate(MODEL_NAMES):\n+                row[name] = metric(y[selected], candidates[selected, index])\n+                row[f\"{name}_delta_r2\"] = float(row[name][\"r2\"] - row[\"baseline\"][\"r2\"])\n+            report[\"folds\"].append(row)\n+        for name_bin, lower, upper in ((\"lt_0.30\", 0.0, 0.30), (\"0.30_0.50\", 0.30, 0.50), (\"0.50_0.70\", 0.50, 0.70), (\"ge_0.70\", 0.70, 1.01)):\n+            selected = (nearest >= lower) & (nearest < upper)\n+            if int(np.sum(selected)) < 5:\n+                continue\n+            report[\"low_similarity\"][name_bin] = {\"rows\": int(np.sum(selected)), \"baseline\": metric(y[selected], baseline[selected]), **{name: metric(y[selected], candidates[selected, index]) for index, name in enumerate(MODEL_NAMES)}}\n+            for name in MODEL_NAMES:\n+                base_r2 = report[\"low_similarity\"][name_bin][\"baseline\"][\"r2\"]\n+                arm_r2 = report[\"low_similarity\"][name_bin][name][\"r2\"]\n+                report[\"low_similarity\"][name_bin][f\"{name}_delta_r2\"] = None if base_r2 is None or arm_r2 is None else float(arm_r2 - base_r2)\n+        target_reports[target] = report\n+        for index, name in enumerate((\"frozen_c001_blend\", *MODEL_NAMES)):\n+            prediction = baseline if index == 0 else candidates[:, index - 1]\n+            metric_rows.append({\"target\": target, \"method\": name, **metric(y, prediction)})\n+        for index, row in frame.iterrows():\n+            prediction_rows.append({\"target\": target, \"index\": int(index), \"fold\": int(folds[index]), \"baseline\": float(baseline[index]), **{name: float(candidates[index, column]) for column, name in enumerate(MODEL_NAMES)}})\n+    passing_targets = []\n+    for target, report in target_reports.items():\n+        best_name = max(MODEL_NAMES, key=lambda name: report[\"models\"][name][\"r2\"])\n+        best_index = MODEL_NAMES.index(best_name)\n+        fold_deltas = [float(value[f\"{best_name}_delta_r2\"]) for value in report[\"folds\"]]\n+        low_deltas = [value[f\"{best_name}_delta_r2\"] for value in report[\"low_similarity\"].values() if value[f\"{best_name}_delta_r2\"] is not None]\n+        report[\"selected_model\"] = best_name\n+        report[\"selected_delta_r2\"] = float(report[\"models\"][best_name][\"r2\"] - report[\"baseline\"][\"r2\"])\n+        report[\"selected_positive_folds\"] = int(sum(value > 0 for value in fold_deltas))\n+        report[\"selected_min_low_similarity_delta\"] = min(low_deltas) if low_deltas else None\n+        pi_gain = float(report[\"models\"][MODEL_NAMES[0]][\"r2\"] - report[\"baseline\"][\"r2\"])\n+        control_gain = float(report[\"models\"][MODEL_NAMES[1]][\"r2\"] - report[\"baseline\"][\"r2\"])\n+        report[\"pi1m_minus_official_control\"] = float(pi_gain - control_gain)\n+        if report[\"selected_delta_r2\"] >= 0.01 and report[\"selected_positive_folds\"] >= 4 and report[\"selected_min_low_similarity_delta\"] is not None and report[\"selected_min_low_similarity_delta\"] >= 0.0 and pi_gain >= control_gain:\n+            passing_targets.append(target)\n+    audit = {\n+        \"schema_version\": \"ppp.round2.pi1m-scratch-control-run.v1\",\n+        \"experiment_id\": run_dir.name,\n+        \"created_at\": datetime.now().astimezone().isoformat(),\n+        \"parent\": \"R2-C009-20260803-1734-nc-size-specialist\",\n+        \"official_inputs\": inputs,\n+        \"pi1m_sha256\": pi1m_hash,\n+        \"pi1m_rows_used\": 200000,\n+        \"official_hashes_pass\": all(inputs[name][\"sha256\"] == expected for name, expected in reference.EXPECTED_HASHES.items()),\n+        \"targets\": target_reports,\n+        \"passing_targets\": passing_targets,\n+        \"decision\": \"component_pass\" if passing_targets else \"rejected_component_gate\",\n+        \"elapsed_seconds\": float(time.time() - started),\n+    }\n+    pd.DataFrame(metric_rows).to_csv(run_dir / \"metrics.csv\", index=False)\n+    pd.DataFrame(prediction_rows).to_csv(run_dir / \"predictions.csv\", index=False)\n+    pd.DataFrame(fold_rows).to_csv(run_dir / \"fold_assignments.csv\", index=False)\n+    write_json(run_dir / \"config.json\", {\"schema_version\": \"ppp.round2.pi1m-scratch-control.v1\", \"seed\": 2026, \"folds\": 5, \"models\": list(MODEL_NAMES), \"pi1m_rows_used\": 200000, \"max_features\": 8192, \"ngram_range\": [2, 6], \"pi1m_sha256\": pi1m_hash, \"official_inputs\": inputs})\n+    (run_dir / \"environment.txt\").write_text(\"\\n\".join([f\"python={platform.python_version()}\", f\"numpy={np.__version__}\", f\"pandas={pd.__version__}\", f\"sklearn={__import__('sklearn').__version__}\", f\"platform={platform.platform()}"]) + \"\\n\", encoding=\"utf-8\")\n+    write_json(run_dir / \"metrics.json\", audit)\n+    (run_dir / \"decision.md\").write_text(f\"# R2-C010 PI1M scratch control decision\\n\\nDecision: **{audit['decision']}**.\\n\\nThe PI1M vocabulary and official-covariate control were fit without labels; no candidate changed in this comparison.\\n\", encoding=\"utf-8\")\n+    (run_dir / \"command.txt\").write_text(\" \".join(os.sys.argv) + \"\\n\", encoding=\"utf-8\")\n+    manifest_paths = [run_dir / name for name in (\"config.json\", \"environment.txt\", \"metrics.csv\", \"predictions.csv\", \"fold_assignments.csv\", \"metrics.json\", \"decision.md\", \"command.txt\", \"protocol.json\")]\n+    (run_dir / \"artifact_manifest.sha256\").write_text(\"\\n\".join(f\"{sha256_file(path)}  {path.name}\" for path in manifest_paths) + \"\\n\", encoding=\"utf-8\")\n+    print(json.dumps({\"experiment_id\": run_dir.name, \"decision\": audit[\"decision\"], \"passing_targets\": passing_targets, \"summary\": {target: {\"selected_model\": report[\"selected_model\"], \"selected_delta_r2\": report[\"selected_delta_r2\"], \"pi1m_minus_control\": report[\"pi1m_minus_official_control\"], \"min_low_similarity_delta\": report[\"selected_min_low_similarity_delta\"]} for target, report in target_reports.items()}, \"elapsed_seconds\": audit[\"elapsed_seconds\"]}, indent=2))\n+\n+\n+if __name__ == \"__main__\":\n+    main()
