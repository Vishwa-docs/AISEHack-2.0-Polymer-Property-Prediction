"""C279: from-scratch Polymer Genome-style hierarchical feature portfolio.

The feature family is rebuilt from official canonical SMILES: atom singles,
bonds, coordinated atomic triples, backbone-restricted triples, and compact
ring/side-chain morphology.  A target-specific Ridge residual is nested inside
grouped folds and blended with the clean C050 parent using a nonnegative weight
selected only on inner grouped folds.  The full-test writer uses the same
parent-plus-weighted-residual formula as OOF inference.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ppp-round-2"
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C279-polymer-genome-hierarchical-portfolio-v1"
PARENT_DIR = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-notebook-runtime-v8"
PARENT_OOF = PARENT_DIR / "oof_predictions.csv"
PARENT_TEST = PARENT_DIR / "notebook_predictions.csv"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "claude_r2_01"))
import initial_reference_pipeline as reference  # noqa: E402
import build_pgfp  # noqa: E402


TARGETS = tuple(reference.TARGETS)
ALPHAS = (30.0, 100.0, 300.0)
WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, **payload}, sort_keys=True) + "\n")


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        return np.zeros(len(groups), dtype=np.int64)
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[valid] = fold
    return folds


def fit_predict(X_train: np.ndarray, residual: np.ndarray, X_valid: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1.0e-4),
    )
    model.fit(X_train, residual)
    return np.asarray(model.predict(X_valid), dtype=float)


def select_inner(X: np.ndarray, y: np.ndarray, base: np.ndarray, groups: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        return 100.0, 0.0, {"inner_folds": 1, "scores": {}}
    predictions: dict[tuple[float, float], np.ndarray] = {
        (alpha, weight): np.full(len(y), np.nan, dtype=float)
        for alpha in ALPHAS for weight in WEIGHTS
    }
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, valid_idx in splitter.split(X, y, groups=groups):
        for alpha in ALPHAS:
            residual = fit_predict(X[train_idx], y[train_idx] - base[train_idx], X[valid_idx], alpha)
            for weight in WEIGHTS:
                predictions[(alpha, weight)][valid_idx] = base[valid_idx] + weight * residual
    scores = {key: float(r2_score(y, pred)) for key, pred in predictions.items()}
    best = max(scores, key=lambda key: (scores[key], -key[1], -key[0]))
    return best[0], best[1], {"inner_folds": n_splits, "scores": {f"{a}:{w}": value for (a, w), value in scores.items()}}


def build_features(keys: list[str], progress_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    records = []
    morphology = []
    document_frequency: Counter[str] = Counter()
    for i, key in enumerate(keys, start=1):
        record = build_pgfp.keys_for(key)
        records.append(record)
        document_frequency.update(set(record[0]) | set(record[1]))
        if i % 500 == 0 or i == len(keys):
            progress(progress_path, "hierarchical_features", processed=i, total=len(keys), phase="tokenize")
    vocabulary = sorted(value for value, count in document_frequency.items() if count >= 5)
    vocabulary_index = {value: index for index, value in enumerate(vocabulary)}
    counts = np.zeros((len(keys), len(vocabulary)), dtype=np.float32)
    for row, (overall, backbone, _) in enumerate(records):
        for token, value in overall.items():
            index = vocabulary_index.get(token)
            if index is not None:
                counts[row, index] += float(value)
        for token, value in backbone.items():
            index = vocabulary_index.get(token)
            if index is not None:
                counts[row, index] += float(value)
    for i, key in enumerate(keys, start=1):
        morphology.append(build_pgfp.morphological(key))
        if i % 500 == 0 or i == len(keys):
            progress(progress_path, "hierarchical_features", processed=i, total=len(keys), phase="morphology")
    morph = np.asarray(morphology, dtype=np.float32)
    counts_sum = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.maximum(1.0, counts_sum)
    features = np.hstack([np.log1p(counts), normalized, morph]).astype(np.float32)
    features[~np.isfinite(features)] = np.nan
    return features, {"vocabulary_size": len(vocabulary), "feature_shape": list(features.shape)}


def main() -> None:
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    progress_path = RUN / "progress.jsonl"
    progress_path.write_text(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n", encoding="utf-8")
    oof = pd.read_csv(PARENT_OOF)
    parent_test = pd.read_csv(PARENT_TEST)
    test = pd.read_csv(DATA_DIR / "test.csv")
    test["canonical"] = test["smiles"].map(reference.canonicalize)
    parent_test = parent_test.merge(test[["id", "target_type", "canonical"]], on="id", how="left", validate="one_to_one")
    if len(parent_test) != 4940 or not np.isfinite(parent_test["target"].to_numpy(float)).all():
        raise RuntimeError("C050 test parent contract failed")
    keys = sorted(set(oof["canonical"].astype(str)) | set(test["canonical"].astype(str)))
    progress(progress_path, "parent_ready", oof_rows=len(oof), test_rows=len(parent_test), feature_rows=len(keys))
    X, feature_report = build_features(keys, progress_path)
    key_index = {key: index for index, key in enumerate(keys)}
    progress(progress_path, "features_ready", **feature_report)

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = oof[oof["target_type"].astype(str).eq(target)].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        base = frame["candidate_prediction"].to_numpy(float)
        groups = frame["group"].astype(str).to_numpy(object)
        folds = grouped_folds(groups)
        indices = np.asarray([key_index[value] for value in frame["canonical"].astype(str)], dtype=int)
        candidate = np.full(len(frame), np.nan, dtype=float)
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            valid = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            alpha, weight, inner = select_inner(X[indices[train_idx]], y[train_idx], base[train_idx], groups[train_idx])
            residual = fit_predict(X[indices[train_idx]], y[train_idx] - base[train_idx], X[indices[valid]], alpha)
            candidate[valid] = base[valid] + weight * residual
            parent_score = float(r2_score(y[valid], base[valid]))
            candidate_score = float(r2_score(y[valid], candidate[valid]))
            fold_rows.append({"fold": int(fold), "rows": int(len(valid)), "alpha": alpha, "weight": weight, "parent_r2": parent_score, "candidate_r2": candidate_score, "delta_r2": candidate_score - parent_score, "inner": inner})
        alpha_full, weight_full, full_inner = select_inner(X[indices], y, base, groups)
        target_test = parent_test[parent_test["target_type"].astype(str).eq(target)].copy()
        test_indices = np.asarray([key_index[value] for value in target_test["canonical"].astype(str)], dtype=int)
        residual_test = fit_predict(X[indices], y - base, X[test_indices], alpha_full)
        target_test["candidate"] = target_test["target"].to_numpy(float) + weight_full * residual_test
        parent_r2 = float(r2_score(y, base))
        candidate_r2 = float(r2_score(y, candidate))
        target_reports[target] = {"rows": int(len(frame)), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "full_alpha": alpha_full, "full_weight": weight_full, "folds": fold_rows, "full_inner": full_inner}
        oof_parts.append(pd.DataFrame({"canonical": frame["canonical"].astype(str), "target_type": target, "target": y, "parent": base, "candidate": candidate, "group": groups, "outer_fold": folds}))
        test_parts.append(target_test[["id", "target_type", "candidate"]])
        print(json.dumps({"target": target, **target_reports[target]}, sort_keys=True), flush=True)

    candidates = pd.concat(test_parts, ignore_index=True).rename(columns={"candidate": "target"})
    submission = test[["id"]].merge(candidates[["id", "target"]], on="id", how="left", validate="one_to_one").sort_values("id").reset_index(drop=True)
    if len(submission) != 4940 or not np.array_equal(submission["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C279 test output contract failed")
    pd.concat(oof_parts, ignore_index=True).to_csv(RUN / "oof_predictions.csv", index=False)
    submission.to_csv(RUN / "predictions.csv", index=False)
    mean_parent = float(np.mean([value["parent_r2"] for value in target_reports.values()]))
    mean_candidate = float(np.mean([value["candidate_r2"] for value in target_reports.values()]))
    report = {"schema_version": "ppp.round2.c279.polymer-genome-hierarchical-portfolio.v1", "experiment_id": RUN.name, "status": "completed_research_candidate", "official_only": True, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "pretrained_weights": False, "external_targets": False, "target_reports": target_reports, "feature_report": feature_report, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "complete_output_rows": int(len(submission)), "complete_output_order_pass": True, "elapsed_seconds": time.time() - started, "source_hashes": {"runner": digest(Path(__file__)), "parent_oof": digest(PARENT_OOF), "parent_test": digest(PARENT_TEST), "pgfp_source": digest(ROOT / "tools" / "claude_r2_01" / "build_pgfp.py"), "reference": digest(ROOT / "tools" / "initial_reference_pipeline.py")}}
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (RUN / "protocol.json").write_text(json.dumps({"schema_version": "ppp.round2.c279.polymer-genome-hierarchical-portfolio.v1", "experiment_id": RUN.name, "hypothesis": "Polymer Genome-style coordinated atomic triples, backbone-restricted triples, and morphology contain complementary target-specific residual signal not captured by generic repeat-view fingerprints.", "changed_factor": "from-scratch hierarchical atomic-triple/backbone/morphology features with nested parent portfolio", "baseline": "clean C050 candidate_prediction parent", "source_inputs": ["ppp-round-2/train.csv", "ppp-round-2/test.csv", "ppp-round-2/archive/train.csv"], "promotion_gate": "candidate must beat the frozen incumbent on post-freeze transfer; standalone target replacement retains existing target gates; portfolio arm requires positive common-fold mean and no target collapse", "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "pretrained_weights": False, "external_targets": False}, indent=2) + "\n", encoding="utf-8")
    (RUN / "decision.md").write_text(f"# {RUN.name}\n\nMean parent: `{mean_parent:.12f}`. Mean candidate: `{mean_candidate:.12f}`. Mean gain: `{mean_candidate - mean_parent:+.12f}`. Full inference uses the same nested parent-plus-weighted-residual formula as OOF. Promotion still requires post-freeze transfer and all submission gates.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    progress(progress_path, "finished", mean_parent_r2=mean_parent, mean_candidate_r2=mean_candidate, mean_gain=mean_candidate - mean_parent)
    print(json.dumps({"experiment_id": RUN.name, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
