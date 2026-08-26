"""C278: deployment-faithful repeat-view residual portfolio.

This child tests the repeat-view representation proposed after the C276 audit.
It uses the clean C050 out-of-fold prediction as an immutable parent arm and
fits one structure-only residual arm per target.  The residual weight is
selected only inside inner grouped folds and the selected weight is applied to
the untouched outer fold and to the full-data test inference with the exact
same formula.  No external_label file, leaderboard result, or local_eval value is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ppp-round-2"
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C278-repeat-view-nested-portfolio-v1"
PARENT_DIR = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-notebook-runtime-v8"
PARENT_OOF = PARENT_DIR / "oof_predictions.csv"
PARENT_TEST = PARENT_DIR / "notebook_predictions.csv"
sys.path.insert(0, str(ROOT / "tools"))
import initial_reference_pipeline as reference  # noqa: E402
import round2_c258_ei_eht_orbital_residual as view_helpers  # noqa: E402


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
BITS = 256
RIDGE_ALPHA = 100.0
WEIGHT_GRID = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_progress(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, **payload}, sort_keys=True) + "\n")


def fp_bits(molecule: Chem.Mol | None) -> np.ndarray:
    out = np.zeros(BITS, dtype=np.float32)
    if molecule is None:
        return out
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=BITS, useChirality=False)
        DataStructs.ConvertToNumpyArray(fp, out)
    except Exception:
        pass
    return out


def compact_descriptors(molecule: Chem.Mol | None) -> np.ndarray:
    if molecule is None:
        return np.full(10, np.nan, dtype=np.float32)
    try:
        return np.asarray([
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumRotatableBonds(molecule),
            rdMolDescriptors.CalcNumRings(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            molecule.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcNumHBA(molecule),
            rdMolDescriptors.CalcNumHBD(molecule),
            rdMolDescriptors.CalcLabuteASA(molecule),
        ], dtype=np.float32)
    except Exception:
        return np.full(10, np.nan, dtype=np.float32)


def build_view_features(keys: list[str], molecules: list[Chem.Mol | None], progress: Path) -> tuple[np.ndarray, dict[str, int]]:
    rows: list[np.ndarray] = []
    supported_hcap = 0
    supported_ring = 0
    for i, (key, original) in enumerate(zip(keys, molecules), start=1):
        capped = view_helpers.remove_dummy_caps(key)
        periodic = view_helpers.ring_close_dummy_caps(key)
        if capped is not None:
            supported_hcap += 1
        if periodic is not None:
            supported_ring += 1
        views = np.vstack([fp_bits(original), fp_bits(capped), fp_bits(periodic)])
        mean_fp = views.mean(axis=0)
        descriptors = np.concatenate([
            compact_descriptors(original),
            compact_descriptors(capped),
            compact_descriptors(periodic),
        ])
        rows.append(np.concatenate([views.reshape(-1), mean_fp, descriptors]))
        if i % 500 == 0 or i == len(keys):
            append_progress(progress, "features", processed=i, total=len(keys))
    return np.asarray(rows, dtype=np.float32), {"hcap_supported": supported_hcap, "periodic_supported": supported_ring}


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


def fit_residual(X_train: np.ndarray, residual_train: np.ndarray, X_valid: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4),
    )
    model.fit(X_train, residual_train)
    return np.asarray(model.predict(X_valid), dtype=float)


def choose_weight(X: np.ndarray, y: np.ndarray, base: np.ndarray, groups: np.ndarray, seed_tag: str) -> tuple[float, dict[str, Any]]:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        return 0.0, {"seed_tag": seed_tag, "inner_folds": 1, "scores": {str(w): None for w in WEIGHT_GRID}}
    splitter = GroupKFold(n_splits=n_splits)
    predictions: dict[float, np.ndarray] = {weight: np.full(len(y), np.nan, dtype=float) for weight in WEIGHT_GRID}
    for train_idx, valid_idx in splitter.split(X, y, groups=groups):
        residual = fit_residual(X[train_idx], y[train_idx] - base[train_idx], X[valid_idx])
        for weight in WEIGHT_GRID:
            predictions[weight][valid_idx] = base[valid_idx] + weight * residual
    scores = {weight: float(r2_score(y, predictions[weight])) for weight in WEIGHT_GRID}
    best = max(WEIGHT_GRID, key=lambda weight: (scores[weight], -weight))
    return float(best), {"seed_tag": seed_tag, "inner_folds": n_splits, "scores": {str(k): v for k, v in scores.items()}}


def main() -> None:
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    progress = RUN / "progress.jsonl"
    progress.write_text(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n", encoding="utf-8")
    if not PARENT_OOF.is_file() or not PARENT_TEST.is_file():
        raise FileNotFoundError("C050 parent artifacts are required")
    oof = pd.read_csv(PARENT_OOF)
    parent_test = pd.read_csv(PARENT_TEST)
    test = pd.read_csv(DATA_DIR / "test.csv")
    required = {"canonical", "target_type", "target", "candidate_prediction", "group"}
    missing = required.difference(oof.columns)
    if missing:
        raise RuntimeError(f"missing C050 OOF columns: {sorted(missing)}")
    test["canonical"] = test["smiles"].map(reference.canonicalize)
    parent_test = parent_test.merge(test[["id", "target_type", "canonical"]], on="id", how="left", validate="one_to_one")
    if parent_test["target"].isna().any() or len(parent_test) != 4940:
        raise RuntimeError("C050 parent test contract failed")
    keys = sorted(set(oof["canonical"].astype(str)) | set(test["canonical"].astype(str)))
    molecules = reference.build_molecules(keys)
    key_index = {key: i for i, key in enumerate(keys)}
    append_progress(progress, "parent_ready", oof_rows=len(oof), test_rows=len(parent_test), feature_rows=len(keys))
    X, support = build_view_features(keys, molecules, progress)
    append_progress(progress, "features_ready", feature_shape=list(X.shape), **support)

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
        fold_rows: list[dict[str, Any]] = []
        for fold in sorted(np.unique(folds)):
            valid = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            chosen, inner = choose_weight(X[indices[train_idx]], y[train_idx], base[train_idx], groups[train_idx], f"{target}-outer-{fold}")
            residual = fit_residual(X[indices[train_idx]], y[train_idx] - base[train_idx], X[indices[valid]])
            candidate[valid] = base[valid] + chosen * residual
            parent_score = float(r2_score(y[valid], base[valid]))
            candidate_score = float(r2_score(y[valid], candidate[valid]))
            fold_rows.append({"fold": int(fold), "rows": int(len(valid)), "chosen_weight": chosen, "parent_r2": parent_score, "candidate_r2": candidate_score, "delta_r2": candidate_score - parent_score, "inner": inner})
        chosen_full, full_inner = choose_weight(X[indices], y, base, groups, f"{target}-full")
        full_residual = fit_residual(X[indices], y - base, X[np.asarray([key_index[value] for value in parent_test[parent_test["target_type"].astype(str).eq(target)]["canonical"].astype(str)], dtype=int)])
        target_test = parent_test[parent_test["target_type"].astype(str).eq(target)].copy()
        target_test["candidate"] = target_test["target"].to_numpy(float) + chosen_full * full_residual
        parent_r2 = float(r2_score(y, base))
        candidate_r2 = float(r2_score(y, candidate))
        target_reports[target] = {"rows": int(len(frame)), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "chosen_full_weight": chosen_full, "folds": fold_rows, "full_inner": full_inner}
        oof_parts.append(pd.DataFrame({"canonical": frame["canonical"].astype(str), "target_type": target, "target": y, "parent": base, "candidate": candidate, "group": groups, "outer_fold": folds}))
        test_parts.append(target_test[["id", "target_type", "candidate"]])
        print(json.dumps({"target": target, **target_reports[target]}, sort_keys=True), flush=True)

    candidates = pd.concat(test_parts, ignore_index=True).rename(columns={"candidate": "target"})
    submission = test[["id"]].merge(candidates[["id", "target"]], on="id", how="left", validate="one_to_one").sort_values("id").reset_index(drop=True)
    if len(submission) != 4940 or not np.array_equal(submission["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C278 full-test output contract failed")
    oof_out = pd.concat(oof_parts, ignore_index=True)
    oof_out.to_csv(RUN / "oof_predictions.csv", index=False)
    submission.to_csv(RUN / "predictions.csv", index=False)
    mean_parent = float(np.mean([report["parent_r2"] for report in target_reports.values()]))
    mean_candidate = float(np.mean([report["candidate_r2"] for report in target_reports.values()]))
    report = {"schema_version": "ppp.round2.c278.repeat-view-nested-portfolio.v1", "experiment_id": RUN.name, "status": "completed_research_candidate", "official_only": True, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "pretrained_weights": False, "external_targets": False, "feature_shape": list(X.shape), "weights": list(WEIGHT_GRID), "targets": target_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "complete_output_rows": int(len(submission)), "complete_output_order_pass": True, "elapsed_seconds": time.time() - started, "source_hashes": {"runner": digest(Path(__file__)), "parent_oof": digest(PARENT_OOF), "parent_test": digest(PARENT_TEST), "reference": digest(ROOT / "tools/initial_reference_pipeline.py")}}
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (RUN / "protocol.json").write_text(json.dumps({"schema_version": "ppp.round2.c278.repeat-view-nested-portfolio.v1", "experiment_id": RUN.name, "hypothesis": "Equivalent open, capped, and periodicized repeat views add deployment-transferable residual information when the residual weight is selected inside grouped inner folds.", "changed_factor": "repeat-view feature residual plus nested nonnegative parent portfolio weight", "baseline": "clean C050 candidate_prediction parent", "promotion_gate": "candidate must beat the frozen incumbent on post-freeze transfer; standalone target replacement requires the existing target gates; portfolio arm requires positive common-fold mean and no target collapse", "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False}, indent=2) + "\n", encoding="utf-8")
    (RUN / "decision.md").write_text(f"# {RUN.name}\n\nMean parent: `{mean_parent:.12f}`. Mean candidate: `{mean_candidate:.12f}`. Mean gain: `{mean_candidate - mean_parent:+.12f}`. The test formula is exactly `parent + selected_weight * residual`; promotion still requires post-freeze transfer and all submission gates.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    append_progress(progress, "finished", mean_parent_r2=mean_parent, mean_candidate_r2=mean_candidate, mean_gain=mean_candidate - mean_parent)
    print(json.dumps({"experiment_id": RUN.name, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
