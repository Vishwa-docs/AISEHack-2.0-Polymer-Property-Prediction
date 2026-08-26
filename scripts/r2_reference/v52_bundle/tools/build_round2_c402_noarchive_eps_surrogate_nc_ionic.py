#!/usr/bin/env python3
"""C402 no-archive EPS ionic route with surrogate-NC deployment.

This is a bounded fork of C366.  C366 had strong current-only train-side EPS
ionic-coordinate evidence but changed zero EPS test rows because deployment
required a current-train NC label for the same canonical structure.  C402 keeps
the same official-only/noarchive constraints and changes only the deployment
support route:

* train an EPS ionic residual model on current official EPS/NC train pairs;
* train a current-only NC surrogate on current official NC train labels;
* for EPS test rows whose SMILES are sufficiently similar to current support,
  predict surrogate NC and apply eps = nc^2 + ionic as a whole-target update;
* blend the update into a frozen noarchive base CSV.

No archive labels, local_eval/external_label files, Kaggle compute, pretrained weights, or
external labels are read.  Scoring is intentionally separate and post-freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference
import round2_c180_flory_fox_oligomer_carriers as c180
import round2_c187_ionic_eps_only as c187


DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C401-NOARCHIVE-C396-SECONDORDER-MICROBLEND-20260808.csv"
)
SEED = 20260808
MIN_IONIC = 0.02
TARGETS = tuple(reference.TARGETS)
IONIC_MODELS = tuple(c187.MODEL_KINDS)


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
        raise RuntimeError(f"Refusing archive/cross-branch {role} path for noarchive run: {path}")
    if allow_output and "/without_archive/" not in low:
        raise RuntimeError(f"{role} path must stay in without_archive namespace: {path}")


def canonical_smiles(smiles: str) -> str:
    return reference.canonicalize(smiles)


def no_stereo(smiles: str) -> str:
    mol = reference.Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return reference.Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def load_candidate(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path, role="base candidate")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema/count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base has non-finite predictions: {path}")
    return frame


def grouped_folds(groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for fold, (_, va) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError("Fold assignment failed")
    return folds


def make_nc_model(kind: str, fold: int):
    if kind == "ridge":
        return Ridge(alpha=80.0)
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=SEED + fold,
        )
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.60,
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=4,
        )
    raise RuntimeError(f"Unknown surrogate NC model: {kind}")


def morgan_fp(smiles: str):
    mol = reference.Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES for Morgan support: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)


def max_tanimoto_to_support(test_smiles: list[str], support_smiles: list[str]) -> np.ndarray:
    support_fps = [morgan_fp(value) for value in support_smiles]
    values: list[float] = []
    for value in test_smiles:
        fp = morgan_fp(value)
        sims = DataStructs.BulkTanimotoSimilarity(fp, support_fps)
        values.append(float(max(sims)) if sims else 0.0)
    return np.asarray(values, dtype=np.float64)


def fit_predict_ensemble(
    dense: np.ndarray,
    sparse_matrix,
    train_indices: np.ndarray,
    y: np.ndarray,
    pred_indices: np.ndarray,
    *,
    model_kind: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_tr, x_te = c187.fold_matrix(dense, sparse_matrix, train_indices, pred_indices)
    model = make_nc_model(model_kind, 0)
    model.fit(x_tr, y)
    pred = model.predict(x_te)
    return np.asarray(pred, dtype=np.float64), {
        "model_kind": model_kind,
        "train_rows": int(len(train_indices)),
        "pred_rows": int(len(pred_indices)),
    }


def nc_surrogate_oof(
    dense: np.ndarray,
    sparse_matrix,
    nc_indices: np.ndarray,
    nc_y: np.ndarray,
    groups: np.ndarray,
    *,
    model_kind: str,
) -> dict[str, Any]:
    folds = grouped_folds(groups)
    pred = np.full(len(nc_indices), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        x_tr, x_va = c187.fold_matrix(dense, sparse_matrix, nc_indices[tr], nc_indices[va])
        model = make_nc_model(model_kind, int(fold))
        model.fit(x_tr, nc_y[tr])
        pred[va] = model.predict(x_va)
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(va)),
                "r2": float(r2_score(nc_y[va], pred[va])) if len(va) > 1 and np.var(nc_y[va]) > 1.0e-15 else None,
            }
        )
    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite NC surrogate OOF")
    return {
        "model_kind": model_kind,
        "rows": int(len(nc_y)),
        "oof_r2": float(r2_score(nc_y, pred)),
        "folds": fold_rows,
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
    }


def ionic_oof_audit(
    dense: np.ndarray,
    sparse_matrix,
    pair_indices: np.ndarray,
    eps_y: np.ndarray,
    nc_y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    folds = grouped_folds(groups)
    log_ionic = np.log(np.maximum(eps_y - nc_y**2, MIN_IONIC))
    pred_by_kind = {kind: np.full(len(pair_indices), np.nan, dtype=np.float64) for kind in IONIC_MODELS}
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        x_tr, x_va = c187.fold_matrix(dense, sparse_matrix, pair_indices[tr], pair_indices[va])
        for kind in IONIC_MODELS:
            model = c187.make_model(kind, int(fold))
            model.fit(x_tr, log_ionic[tr])
            pred_by_kind[kind][va] = np.exp(np.clip(model.predict(x_va), -8, 4))
        raw_eps = nc_y[va] ** 2 + np.mean([pred_by_kind[kind][va] for kind in IONIC_MODELS], axis=0)
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(va)),
                "eps_from_true_nc_r2": float(r2_score(eps_y[va], raw_eps)),
            }
        )
    if any(not np.isfinite(value).all() for value in pred_by_kind.values()):
        raise RuntimeError("Non-finite ionic OOF")
    ionic_pred = np.mean(np.column_stack([pred_by_kind[kind] for kind in IONIC_MODELS]), axis=1)
    raw_eps = nc_y**2 + ionic_pred
    return {
        "pair_rows": int(len(pair_indices)),
        "ionic_log_model_kinds": list(IONIC_MODELS),
        "ionic_oof_r2": float(r2_score(np.maximum(eps_y - nc_y**2, MIN_IONIC), ionic_pred)),
        "eps_from_true_nc_oof_r2": float(r2_score(eps_y, raw_eps)),
        "folds": fold_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="ppp-round-2/train.csv")
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--surrogate-nc-model", choices=("extra_trees", "hgb", "ridge"), default="extra_trees")
    parser.add_argument("--support-min-similarity", type=float, default=0.35)
    parser.add_argument("--pull", type=float, default=0.50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    started = time.time()
    train_path = Path(args.train_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path, role in ((train_path, "train"), (test_path, "test"), (base_path, "base")):
        guard_path(path, role=role)
    for path, role in ((output, "output"), (manifest, "manifest")):
        guard_path(path, role=role, allow_output=True)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if not (0.0 <= float(args.pull) <= 1.0):
        raise RuntimeError("--pull must be in [0, 1]")
    if not (0.0 <= float(args.support_min_similarity) <= 1.0):
        raise RuntimeError("--support-min-similarity must be in [0, 1]")

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
        frame["canonical"] = [canonical_smiles(value) for value in frame["smiles"]]
        frame["nostereo"] = [no_stereo(value) for value in frame["canonical"]]
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError("Unexpected current official row counts")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")

    base = load_candidate(base_path, ids)
    result = base["target"].to_numpy(float).copy()
    wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    pair_frame = wide[["eps", "nc"]].dropna().copy()
    nc_frame = wide[["nc"]].dropna().copy()
    if len(pair_frame) < 50 or len(nc_frame) < 50:
        raise RuntimeError("Insufficient current official EPS/NC or NC train rows")
    ionic = pair_frame["eps"].to_numpy(float) - pair_frame["nc"].to_numpy(float) ** 2
    if np.any(ionic <= 0):
        raise RuntimeError("Non-positive ionic coordinate in current official pair rows")

    keys = sorted(set(train["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    round2_root = Path(".").resolve()
    repo_root = round2_root.parent if round2_root.name == "Polymer Prediction Challenge Round 2" else round2_root
    dense, sparse_matrix, feature_report = c180.build_features(repo_root, keys)
    dense = np.asarray(dense, dtype=np.float64)
    sparse_matrix = sparse_matrix.astype(np.float64)

    pair_canons = pair_frame.index.astype(str).tolist()
    nc_canons = nc_frame.index.astype(str).tolist()
    pair_indices = np.asarray([key_to_index[value] for value in pair_canons], dtype=np.int64)
    nc_indices = np.asarray([key_to_index[value] for value in nc_canons], dtype=np.int64)
    pair_groups = np.asarray([no_stereo(value) for value in pair_canons], dtype=object)
    nc_groups = np.asarray([no_stereo(value) for value in nc_canons], dtype=object)
    eps_y = pair_frame["eps"].to_numpy(float)
    pair_nc_y = pair_frame["nc"].to_numpy(float)
    nc_y = nc_frame["nc"].to_numpy(float)

    ionic_report = ionic_oof_audit(dense, sparse_matrix, pair_indices, eps_y, pair_nc_y, pair_groups)
    nc_report = nc_surrogate_oof(
        dense,
        sparse_matrix,
        nc_indices,
        nc_y,
        nc_groups,
        model_kind=str(args.surrogate_nc_model),
    )

    test_eps = test[test["target_type"] == "eps"].copy()
    test_eps_indices = test_eps.index.to_numpy(int)
    test_eps_canons = test_eps["canonical"].astype(str).tolist()
    test_eps_key_indices = np.asarray([key_to_index[value] for value in test_eps_canons], dtype=np.int64)
    support_similarity = max_tanimoto_to_support(test_eps_canons, sorted(set(pair_canons) | set(nc_canons)))
    supported = support_similarity >= float(args.support_min_similarity)

    nc_pred, nc_full_report = fit_predict_ensemble(
        dense,
        sparse_matrix,
        nc_indices,
        nc_y,
        test_eps_key_indices,
        model_kind=str(args.surrogate_nc_model),
    )
    nc_low = float(np.quantile(nc_y, 0.005))
    nc_high = float(np.quantile(nc_y, 0.995))
    nc_pred = np.clip(nc_pred, max(1.0, nc_low - 0.05), min(2.8, nc_high + 0.05))

    x_tr, x_te = c187.fold_matrix(dense, sparse_matrix, pair_indices, test_eps_key_indices)
    log_ionic = np.log(np.maximum(ionic, MIN_IONIC))
    ionic_preds = []
    for kind in IONIC_MODELS:
        model = c187.make_model(kind, SEED)
        model.fit(x_tr, log_ionic)
        ionic_preds.append(np.exp(np.clip(model.predict(x_te), -8, 4)))
    ionic_pred = np.maximum(np.mean(np.column_stack(ionic_preds), axis=1), MIN_IONIC)

    raw_eps = nc_pred**2 + ionic_pred
    eps_train = train[train["target_type"] == "eps"]["target"].to_numpy(float)
    eps_low = float(np.quantile(eps_train, 0.002))
    eps_high = float(np.quantile(eps_train, 0.998))
    raw_eps = np.clip(raw_eps, max(0.0, eps_low - 0.05), eps_high + 0.05)
    base_eps = result[test_eps_indices]
    replacement = (1.0 - float(args.pull)) * base_eps + float(args.pull) * raw_eps
    result[test_eps_indices[supported]] = replacement[supported]

    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)

    record: dict[str, Any] = {
        "schema_version": "ppp.round2.c402.noarchive-eps-surrogate-nc-ionic.v1",
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
        "method": "current-only EPS ionic-coordinate deployment using surrogate NC model and fixed Morgan support gate",
        "config": {
            "surrogate_nc_model": str(args.surrogate_nc_model),
            "support_min_similarity": float(args.support_min_similarity),
            "pull": float(args.pull),
            "seed": SEED,
            "ionic_models": list(IONIC_MODELS),
        },
        "inputs": {
            "train.csv": {"path": str(train_path), "sha256": train_sha, "bytes": train_path.stat().st_size},
            "test.csv": {"path": str(test_path), "sha256": test_sha, "bytes": test_path.stat().st_size},
            "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        },
        "feature_report": feature_report,
        "target_reports": {"eps_ionic": ionic_report, "nc_surrogate": nc_report},
        "deployment": {
            "eps_test_rows": int(len(test_eps_indices)),
            "supported_eps_rows": int(np.sum(supported)),
            "support_similarity_min": float(np.min(support_similarity)) if len(support_similarity) else None,
            "support_similarity_median": float(np.median(support_similarity)) if len(support_similarity) else None,
            "support_similarity_max": float(np.max(support_similarity)) if len(support_similarity) else None,
            "nc_full_model": nc_full_report,
            "raw_eps_min": float(np.min(raw_eps)) if len(raw_eps) else None,
            "raw_eps_max": float(np.max(raw_eps)) if len(raw_eps) else None,
            "max_abs_delta_on_eps_rows": float(np.max(np.abs(replacement[supported] - base_eps[supported]))) if np.any(supported) else 0.0,
        },
        "elapsed_seconds": float(time.time() - started),
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": record["output"],
                "supported_eps_rows": record["deployment"]["supported_eps_rows"],
                "ionic_oof_r2": ionic_report["eps_from_true_nc_oof_r2"],
                "nc_surrogate_oof_r2": nc_report["oof_r2"],
                "elapsed_seconds": record["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
