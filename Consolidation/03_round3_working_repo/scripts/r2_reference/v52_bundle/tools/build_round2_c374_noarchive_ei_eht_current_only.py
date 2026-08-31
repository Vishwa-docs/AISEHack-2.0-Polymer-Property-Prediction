#!/usr/bin/env python3
"""C374 no-archive EI EHT residual over a noarchive base.

This is a current-only, target-specific port of the EHT idea.  It computes
deterministic RDKit/YAeHMOP extended-Hueckel features from official current
train/test EI SMILES, fits a grouped residual model against C282 current-only
OOF EI predictions, and deploys a whole-EI replacement over a branch-local
noarchive base CSV.

No archive, local_eval, external_label, Kaggle, pretrained, or external-data inputs are
read by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime
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

import initial_reference_pipeline as reference


DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C371-NOARCHIVE-TARGET-SPLICE-C370-EEA-EI-EGC-EPS-NC-BLENDS-20260808.csv"
)
DEFAULT_C282_OOF = "experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1/oof_predictions.csv"
SEED = 20260808


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, role: str, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "/archive/" in low or low.endswith("/archive") or "with_archive" in low:
        raise RuntimeError(f"Refusing archive/cross-branch {role} path for no-archive run: {path}")
    if allow_output and "/without_archive/" not in low:
        raise RuntimeError(f"{role} path must stay in without_archive namespace: {path}")


def load_eht_module():
    source = Path(__file__).resolve().parent / "round2_c258_ei_eht_orbital_residual.py"
    spec = importlib.util.spec_from_file_location("c374_eht_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load C258 EHT source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_seed(smiles: str, variant: str) -> int:
    raw = hashlib.sha256(f"C374|{variant}|{smiles}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) % 2_000_000_000 + 1


def stable_eht_features(eht, smiles: str) -> tuple[np.ndarray, dict[str, bool]]:
    hcap, hcap_ok = eht.eht_variant_features(eht.remove_dummy_caps(smiles), stable_seed(smiles, "hcap"))
    ring, ring_ok = eht.eht_variant_features(eht.ring_close_dummy_caps(smiles), stable_seed(smiles, "ring"))
    diffs = [hcap[i] - ring[i] if hcap_ok and ring_ok else np.nan for i in (0, 1, 2)]
    row = np.asarray(hcap + [float(hcap_ok)] + ring + [float(ring_ok)] + diffs, dtype=float)
    return row, {"hcap_supported": bool(hcap_ok), "ring_supported": bool(ring_ok)}


def no_stereo(smiles: str) -> str:
    mol = reference.Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return reference.Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (_, va) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError("Fold assignment failed")
    return folds


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(1000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path, role="base candidate")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="ppp-round-2/train.csv")
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--c282-oof-csv", default=DEFAULT_C282_OOF)
    parser.add_argument("--residual-weight", type=float, default=0.35)
    parser.add_argument("--ridge-alpha", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    train_path = Path(args.train_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    oof_path = Path(args.c282_oof_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path, role in ((train_path, "train"), (test_path, "test"), (base_path, "base"), (oof_path, "C282 OOF")):
        guard_path(path, role=role)
    for path, role in ((output, "output"), (manifest, "manifest")):
        guard_path(path, role=role, allow_output=True)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")
    residual_weight = float(args.residual_weight)
    if not (0.0 <= residual_weight <= 1.0):
        raise RuntimeError("--residual-weight must be in [0, 1]")

    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)
    if train_sha != reference.EXPECTED_HASHES["train.csv"]:
        raise RuntimeError("train.csv hash mismatch")
    if test_sha != reference.EXPECTED_HASHES["test.csv"]:
        raise RuntimeError("test.csv hash mismatch")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for frame in (train, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
        frame["canonical"] = [reference.canonicalize(value) for value in frame["smiles"]]
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = load_base(base_path, ids)
    oof = pd.read_csv(oof_path)
    guard_path(oof_path, role="C282 OOF")
    required = {"canonical", "target_type", "target", "prediction"}
    if not required.issubset(oof.columns):
        raise RuntimeError("Unexpected C282 OOF schema")
    ei_oof = oof[oof["target_type"].astype(str).str.lower() == "ei"].reset_index(drop=True)
    if len(ei_oof) < 50:
        raise RuntimeError("Insufficient EI OOF rows")
    if not np.isfinite(ei_oof[["target", "prediction"]].to_numpy(float)).all():
        raise RuntimeError("Non-finite EI OOF values")

    test_ei = test[test["target_type"] == "ei"].sort_values("id").reset_index(drop=True)
    feature_keys = sorted(set(ei_oof["canonical"].astype(str)) | set(test_ei["canonical"].astype(str)))
    eht = load_eht_module()
    feature_rows: list[np.ndarray] = []
    support: list[dict[str, bool]] = []
    for smiles in feature_keys:
        row, report = stable_eht_features(eht, smiles)
        feature_rows.append(row)
        support.append(report)
    feature_matrix = np.asarray(feature_rows, dtype=np.float64)
    key_to_index = {key: i for i, key in enumerate(feature_keys)}
    train_indices = np.asarray([key_to_index[value] for value in ei_oof["canonical"].astype(str)], dtype=np.int64)
    test_indices = np.asarray([key_to_index[value] for value in test_ei["canonical"].astype(str)], dtype=np.int64)
    y = ei_oof["target"].to_numpy(float)
    parent = ei_oof["prediction"].to_numpy(float)
    groups = np.asarray([no_stereo(value) for value in ei_oof["canonical"].astype(str)], dtype=object)
    folds = grouped_folds(groups)
    residual_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        va = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=float(args.ridge_alpha), solver="lsqr", max_iter=5000, tol=1.0e-4),
        )
        model.fit(feature_matrix[train_indices[tr]], y[tr] - parent[tr])
        residual_oof[va] = model.predict(feature_matrix[train_indices[va]])
        pred = parent[va] + residual_weight * residual_oof[va]
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(va)),
                "parent_r2": float(r2_score(y[va], parent[va])),
                "candidate_r2": float(r2_score(y[va], pred)),
                "delta_r2": float(r2_score(y[va], pred) - r2_score(y[va], parent[va])),
            }
        )
    if not np.isfinite(residual_oof).all():
        raise RuntimeError("Non-finite EI residual OOF")
    candidate_oof = parent + residual_weight * residual_oof
    report = {
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, candidate_oof)),
        "delta_r2": float(r2_score(y, candidate_oof) - r2_score(y, parent)),
        "positive_folds": int(sum(row["delta_r2"] > 0.0 for row in fold_rows)),
        "group_bootstrap_lower": bootstrap_lower(y, parent, candidate_oof, groups),
        "folds": fold_rows,
        "train_rows": int(len(y)),
        "test_rows": int(len(test_ei)),
    }
    report["clean_gate_pass"] = bool(report["delta_r2"] >= 0.001 and report["positive_folds"] >= 3 and report["group_bootstrap_lower"] > -0.004)

    full_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=float(args.ridge_alpha), solver="lsqr", max_iter=5000, tol=1.0e-4),
    )
    full_model.fit(feature_matrix[train_indices], y - parent)
    residual_test = full_model.predict(feature_matrix[test_indices])
    result = base["target"].to_numpy(float).copy()
    ei_positions = np.flatnonzero(test["target_type"].to_numpy(str) == "ei")
    if len(ei_positions) != len(test_ei):
        raise RuntimeError("EI test alignment failed")
    result[ei_positions] = base["target"].to_numpy(float)[ei_positions] + residual_weight * residual_test
    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.c374.noarchive-ei-eht-current-only.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "current-only EI EHT residual over C282 clean OOF and noarchive base deployment",
        "config": {"residual_weight": residual_weight, "ridge_alpha": float(args.ridge_alpha), "seed": SEED},
        "target_reports": {"ei": report},
        "feature_report": {
            "feature_rows": int(len(feature_keys)),
            "feature_width": int(feature_matrix.shape[1]),
            "hcap_supported": int(sum(item["hcap_supported"] for item in support)),
            "ring_supported": int(sum(item["ring_supported"] for item in support)),
        },
        "inputs": {
            "train.csv": {"path": str(train_path), "sha256": train_sha, "bytes": train_path.stat().st_size},
            "test.csv": {"path": str(test_path), "sha256": test_sha, "bytes": test_path.stat().st_size},
            "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
            "c282_oof_csv": {"path": str(oof_path), "sha256": sha256_file(oof_path), "bytes": oof_path.stat().st_size, "role": "local current-only OOF artifact; final notebook must regenerate"},
        },
        "changed_rows": {"ei": int(len(test_ei))},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "feature_report": record["feature_report"], "ei_report": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
