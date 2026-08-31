#!/usr/bin/env python3
"""C366 no-archive current-only EPS ionic-coordinate route.

This ports the C187/C214 EPS mechanism into the no-archive branch:

* train/development rows come only from official current train.csv;
* clean parent OOF is the current-only C282 OOF artifact;
* deployment fallback/base is a no-archive candidate CSV;
* archive files, local_eval files, Kaggle compute, and external weights are not
  read by this builder.

The builder writes one complete candidate CSV.  LocalEval scoring happens only in
the separate post-freeze scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c187_ionic_eps_only as c187
import round2_c180_flory_fox_oligomer_carriers as c180


DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C361-NOARCHIVE-TARGET-SPLICE-C356-EEA-EI-EGC-NC-BLENDS-20260808.csv"
)
DEFAULT_C282_OOF = "experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1/oof_predictions.csv"
TARGETS = tuple(reference.TARGETS)
SEED = 20260808
MODEL_KINDS = c187.MODEL_KINDS


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


def canonical_smiles(smiles: str) -> str:
    return reference.canonicalize(smiles)


def no_stereo(smiles: str) -> str:
    mol = reference.Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return reference.Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def grouped_folds(groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for fold, (_, va) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError("Fold assignment failed")
    return folds


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(1000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in selected])
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


def load_c282_oof(path: Path) -> pd.DataFrame:
    guard_path(path, role="C282 OOF")
    frame = pd.read_csv(path)
    required = {"canonical", "target_type", "target", "prediction"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Unexpected C282 OOF schema: {path}")
    frame = frame[["canonical", "target_type", "target", "prediction"]].copy()
    frame["target_type"] = frame["target_type"].astype(str).str.lower()
    if set(frame["target_type"]) != set(TARGETS):
        raise RuntimeError("C282 OOF has unexpected targets")
    if not np.isfinite(frame[["target", "prediction"]].to_numpy(float)).all():
        raise RuntimeError("C282 OOF contains non-finite numeric values")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="ppp-round-2/train.csv")
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--c282-oof-csv", default=DEFAULT_C282_OOF)
    parser.add_argument("--half-parent", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    started = time.time()
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
    half_parent = float(args.half_parent)
    if not (0.0 <= half_parent <= 1.25):
        raise RuntimeError("--half-parent outside bounded range [0, 1.25]")

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
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError("Unexpected current official row counts")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = load_base(base_path, ids)
    c282_oof = load_c282_oof(oof_path)

    wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    pair_frame = wide[["eps", "nc"]].dropna().copy()
    pair_canons = pair_frame.index.astype(str).tolist()
    if len(pair_canons) < 50:
        raise RuntimeError("Insufficient current official EPS/NC pair rows")
    eps_y = pair_frame["eps"].to_numpy(float)
    nc_y = pair_frame["nc"].to_numpy(float)
    ionic_y = eps_y - nc_y ** 2
    if np.any(ionic_y <= 0):
        raise RuntimeError("Non-positive ionic coordinate in current official pair rows")
    log_ionic = np.log(ionic_y)

    # Feature matrix over every train/test canonical structure; no labels are
    # consumed by feature generation.
    keys = sorted(set(train["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    round2_root = Path(".").resolve()
    repo_root = round2_root.parent if round2_root.name == "Polymer Prediction Challenge Round 2" else round2_root
    dense, sparse_matrix, feature_report = c180.build_features(repo_root, keys)
    dense = np.asarray(dense, dtype=np.float64)
    sparse_matrix = sparse_matrix.astype(np.float64)
    pair_indices = np.asarray([key_to_index[value] for value in pair_canons], dtype=np.int64)
    groups = np.asarray([no_stereo(value) for value in pair_canons], dtype=object)
    folds = grouped_folds(groups)

    pair_oof = {kind: np.full(len(pair_canons), np.nan, dtype=float) for kind in MODEL_KINDS}
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        va = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        x_tr, x_va = c187.fold_matrix(dense, sparse_matrix, pair_indices[tr], pair_indices[va])
        for kind in MODEL_KINDS:
            model = c187.make_model(kind, int(fold))
            model.fit(x_tr, log_ionic[tr])
            pair_oof[kind][va] = np.exp(np.clip(model.predict(x_va), -8, 4))
        raw_eps = nc_y[va] ** 2 + np.mean([pair_oof[kind][va] for kind in MODEL_KINDS], axis=0)
        # For the pair panel, parent is the C282 OOF EPS prediction.
        parent_map = (
            c282_oof[c282_oof["target_type"] == "eps"]
            .drop_duplicates("canonical")
            .set_index("canonical")["prediction"]
            .to_dict()
        )
        parent_pair = np.asarray([parent_map[value] for value in np.asarray(pair_canons)[va]], dtype=float)
        candidate_pair = (1.0 - half_parent) * parent_pair + half_parent * raw_eps
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(va)),
                "parent_r2": float(r2_score(eps_y[va], parent_pair)),
                "candidate_r2": float(r2_score(eps_y[va], candidate_pair)),
                "delta_r2": float(r2_score(eps_y[va], candidate_pair) - r2_score(eps_y[va], parent_pair)),
            }
        )
    if any(not np.isfinite(values).all() for values in pair_oof.values()):
        raise RuntimeError("Non-finite pair OOF ionic predictions")

    eps_oof = c282_oof[c282_oof["target_type"] == "eps"].reset_index(drop=True)
    eps_parent = eps_oof["prediction"].to_numpy(float)
    eps_candidate = eps_parent.copy()
    pair_raw_oof = nc_y ** 2 + np.mean(np.column_stack([pair_oof[kind] for kind in MODEL_KINDS]), axis=1)
    pair_candidate = (1.0 - half_parent) * np.asarray(
        [eps_oof.drop_duplicates("canonical").set_index("canonical").loc[value, "prediction"] for value in pair_canons],
        dtype=float,
    ) + half_parent * pair_raw_oof
    pair_candidate_by_canon = dict(zip(pair_canons, pair_candidate, strict=True))
    changed_oof = 0
    for idx, canon in enumerate(eps_oof["canonical"].astype(str)):
        if canon in pair_candidate_by_canon:
            eps_candidate[idx] = pair_candidate_by_canon[canon]
            changed_oof += 1
    eps_y_all = eps_oof["target"].to_numpy(float)
    eps_groups_all = np.asarray([no_stereo(value) for value in eps_oof["canonical"].astype(str)], dtype=object)
    eps_report = {
        "parent_r2": float(r2_score(eps_y_all, eps_parent)),
        "candidate_r2": float(r2_score(eps_y_all, eps_candidate)),
        "delta_r2": float(r2_score(eps_y_all, eps_candidate) - r2_score(eps_y_all, eps_parent)),
        "pair_parent_r2": float(r2_score(eps_y, [eps_oof.drop_duplicates("canonical").set_index("canonical").loc[value, "prediction"] for value in pair_canons])),
        "pair_candidate_r2": float(r2_score(eps_y, pair_candidate)),
        "pair_delta_r2": float(r2_score(eps_y, pair_candidate) - r2_score(eps_y, [eps_oof.drop_duplicates("canonical").set_index("canonical").loc[value, "prediction"] for value in pair_canons])),
        "positive_folds": int(sum(row["delta_r2"] > 0.0 for row in fold_rows)),
        "group_bootstrap_lower": bootstrap_lower(eps_y_all, eps_parent, eps_candidate, eps_groups_all),
        "folds": fold_rows,
        "pair_rows": int(len(pair_canons)),
        "changed_oof_rows": int(changed_oof),
    }
    eps_report["clean_gate_pass"] = bool(
        eps_report["delta_r2"] >= 0.002
        and eps_report["positive_folds"] >= 3
        and eps_report["group_bootstrap_lower"] > -0.002
    )

    # Deploy over noarchive base: replace EPS test rows whose canonical has a
    # current official NC train label.  This mirrors C187's labelled-NC support
    # route and does not use test local_eval labels or co-test target values.
    result = base["target"].to_numpy(float).copy()
    test_eps = test[test["target_type"] == "eps"].sort_values("id").reset_index(drop=True)
    base_eps = base.loc[test_eps["id"].to_numpy(int) - 1, "target"].to_numpy(float)
    supported = np.asarray([value in set(pair_canons) for value in test_eps["canonical"]], dtype=bool)
    deployed = 0
    if np.any(supported):
        x_tr, _ = c187.fold_matrix(dense, sparse_matrix, pair_indices, pair_indices)
        pred_indices = np.asarray([key_to_index[value] for value in test_eps.loc[supported, "canonical"]], dtype=np.int64)
        _, x_te = c187.fold_matrix(dense, sparse_matrix, pair_indices, pred_indices)
        full_preds = []
        for kind in MODEL_KINDS:
            model = c187.make_model(kind, SEED)
            model.fit(x_tr, log_ionic)
            full_preds.append(np.exp(np.clip(model.predict(x_te), -8, 4)))
        nc_label = np.asarray([wide.loc[value, "nc"] for value in test_eps.loc[supported, "canonical"]], dtype=float)
        raw = nc_label ** 2 + np.mean(np.column_stack(full_preds), axis=1)
        replacement = (1.0 - half_parent) * base_eps[supported] + half_parent * raw
        eps_positions = np.flatnonzero(test["target_type"].to_numpy(str) == "eps")
        result[eps_positions[supported]] = replacement
        deployed = int(np.sum(supported))
    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)

    record: dict[str, Any] = {
        "schema_version": "ppp.round2.c366.noarchive-eps-ionic-current-only.v1",
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
        "method": "current-only EPS ionic-coordinate route using current train EPS/NC labels and C282 clean OOF gate; noarchive base deployment",
        "config": {"half_parent": half_parent, "model_kinds": list(MODEL_KINDS), "seed": SEED},
        "target_reports": {"eps": eps_report},
        "inputs": {
            "train.csv": {"path": str(train_path), "sha256": train_sha, "bytes": train_path.stat().st_size},
            "test.csv": {"path": str(test_path), "sha256": test_sha, "bytes": test_path.stat().st_size},
            "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
            "c282_oof_csv": {"path": str(oof_path), "sha256": sha256_file(oof_path), "bytes": oof_path.stat().st_size, "role": "local current-only OOF artifact; final notebook must regenerate"},
        },
        "feature_report": feature_report,
        "changed_rows": {"eps_test_rows": deployed, "eps_oof_rows": int(changed_oof)},
        "rows": {"train": int(len(train)), "test": int(len(test)), "official_eps_nc_train_pairs": int(len(pair_canons))},
        "elapsed_seconds": float(time.time() - started),
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "changed_rows": record["changed_rows"], "eps_report": eps_report, "elapsed_seconds": record["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
