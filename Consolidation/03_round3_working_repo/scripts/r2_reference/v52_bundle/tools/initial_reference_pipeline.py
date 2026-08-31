#!/usr/bin/env python3
"""Official-only Round 2 initial reference pipeline.

The same source body is embedded into the submission notebook. It reads only the
official Round 2 bundle and regenerates every prediction from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy import sparse
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")
TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
EXPECTED_HASHES = {
    "train.csv": "609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2",
    "test.csv": "d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d",
    "archive/train.csv": "b12cadb31c747b26e3616474ce3a2839a22e4b2603e392b6b528e714b6864f68",
}
DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 2026,
    "folds": 5,
    "morgan_bits": 4096,
    "text_features": 65536,
    "ridge_alpha_large": 10.0,
    "ridge_alpha_sparse": 30.0,
    "tanimoto_krr_alpha": 0.05,
    "tanimoto_knn_k": 15,
    "extra_trees_estimators": 160,
    "extra_trees_min_leaf_large": 2,
    "extra_trees_min_leaf_sparse": 3,
    "dense_abs_limit": 1.0e12,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    current = Path.cwd().resolve()
    candidates.extend(parent / "ppp-round-2" for parent in (current, *current.parents))
    candidates.append(Path("/kaggle/input/ppp-round-2"))
    for candidate in candidates:
        if (candidate / "train.csv").is_file() and (candidate / "test.csv").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the official ppp-round-2 input directory")


def canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace("[*]", "*"))
    if molecule is None:
        raise ValueError("RDKit could not parse an official SMILES value")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = {
        "train.csv": data_dir / "train.csv",
        "test.csv": data_dir / "test.csv",
        "archive/train.csv": data_dir / "archive" / "train.csv",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(f"Official input hash mismatch for {name}: {hashes[name]}")
    train = pd.read_csv(paths["train.csv"])
    test = pd.read_csv(paths["test.csv"])
    archive = pd.read_csv(paths["archive/train.csv"])
    if list(train.columns) != ["smiles", "target", "target_type"]:
        raise RuntimeError("Unexpected current train schema")
    if list(test.columns) != ["id", "smiles", "target_type"]:
        raise RuntimeError("Unexpected current test schema")
    if list(archive.columns) != ["smiles", "target", "target_type"]:
        raise RuntimeError("Unexpected archive train schema")
    if len(train) != 7409 or len(test) != 4940 or len(archive) != 6171:
        raise RuntimeError("Unexpected official row count")
    for frame in (train, test, archive):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
        frame["canonical"] = [canonicalize(value) for value in frame["smiles"]]
    if set(train["target_type"]) != set(TARGETS) or set(test["target_type"]) != set(TARGETS):
        raise RuntimeError("Unexpected target set")
    if test["id"].duplicated().any() or not np.array_equal(test["id"].to_numpy(), np.arange(1, 4941)):
        raise RuntimeError("Test IDs are not unique sequential IDs 1..4940")
    if not np.isfinite(train["target"].to_numpy(float)).all():
        raise RuntimeError("Current train contains a non-finite target")
    manifest = {
        name: {"path": str(path), "sha256": hashes[name], "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    return train, test, archive, manifest


def build_label_pool(train: pd.DataFrame, archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = train[["smiles", "canonical", "target_type", "target"]].copy()
    current["source"] = "current_train"
    old = archive[["smiles", "canonical", "target_type", "target"]].copy()
    old["source"] = "archive_train"
    raw = pd.concat([current, old], ignore_index=True)
    raw = raw.drop_duplicates(["smiles", "target_type", "target"]).reset_index(drop=True)
    pooled = (
        raw.groupby(["canonical", "target_type"], as_index=False)
        .agg(target=("target", "median"), smiles=("smiles", "first"), measurements=("target", "size"))
    )
    return raw, pooled


def unique_mapping(frame: pd.DataFrame, keys: list[str]) -> dict[tuple[Any, ...], float]:
    grouped = frame.groupby(keys, dropna=False)["target"].agg(["nunique", "first"])
    eligible = grouped[grouped["nunique"] == 1]
    return {tuple(index) if isinstance(index, tuple) else (index,): float(row["first"]) for index, row in eligible.iterrows()}


def build_molecules(keys: list[str]) -> list[Any]:
    molecules = [Chem.MolFromSmiles(value) for value in keys]
    if any(molecule is None for molecule in molecules):
        raise RuntimeError("Canonical official structure failed RDKit parsing")
    return molecules


def descriptor_matrix(molecules: list[Any]) -> tuple[np.ndarray, list[str]]:
    items = list(Descriptors._descList)
    matrix = np.full((len(molecules), len(items)), np.nan, dtype=np.float64)
    for row, molecule in enumerate(molecules):
        for column, (_, function) in enumerate(items):
            try:
                value = float(function(molecule))
            except Exception:
                value = math.nan
            matrix[row, column] = value if math.isfinite(value) else math.nan
    return matrix, [name for name, _ in items]


def physical_matrix(molecules: list[Any], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [
        "smiles_length",
        "atom_count",
        "heavy_atom_count",
        "dummy_atom_count",
        "ring_count",
        "aromatic_atom_count",
        "hetero_atom_count",
        "halogen_count",
        "rotatable_bonds_approx",
        "double_bond_count",
        "triple_bond_count",
        "branch_count",
        "n_count",
        "o_count",
        "s_count",
        "si_count",
    ]
    matrix = np.zeros((len(molecules), len(names)), dtype=np.float64)
    for row, (molecule, value) in enumerate(zip(molecules, smiles, strict=True)):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        matrix[row] = [
            len(value),
            molecule.GetNumAtoms(),
            molecule.GetNumHeavyAtoms(),
            sum(atom.GetAtomicNum() == 0 for atom in atoms),
            molecule.GetRingInfo().NumRings(),
            sum(atom.GetIsAromatic() for atom in atoms),
            sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms),
            sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms),
            sum(bond.GetBondTypeAsDouble() == 1.0 and not bond.IsInRing() for bond in bonds),
            sum(bond.GetBondTypeAsDouble() == 2.0 for bond in bonds),
            sum(bond.GetBondTypeAsDouble() == 3.0 for bond in bonds),
            value.count("("),
            sum(atom.GetAtomicNum() == 7 for atom in atoms),
            sum(atom.GetAtomicNum() == 8 for atom in atoms),
            sum(atom.GetAtomicNum() == 16 for atom in atoms),
            sum(atom.GetAtomicNum() == 14 for atom in atoms),
        ]
    return matrix, names


def morgan_count_matrix(molecules: list[Any], radius: int, bits: int) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, molecule in enumerate(molecules):
        fingerprint = generator.GetCountFingerprint(molecule)
        for column, count in fingerprint.GetNonzeroElements().items():
            rows.append(row)
            columns.append(int(column))
            values.append(math.log1p(float(count)))
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(molecules), bits), dtype=np.float64)


def morgan_bits(molecules: list[Any], radius: int, bits: int) -> list[Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    return [generator.GetFingerprint(molecule) for molecule in molecules]


def text_matrix(smiles: list[str], features: int) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 7),
        n_features=features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float64,
    )
    return vectorizer.transform(smiles).tocsr()


def cross_property_arrays(pooled: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pivot = pooled.pivot(index="canonical", columns="target_type", values="target")
    values = np.full((len(keys), len(TARGETS)), np.nan, dtype=np.float64)
    available = np.zeros((len(keys), len(TARGETS)), dtype=np.float64)
    key_position = {key: index for index, key in enumerate(keys)}
    for target_index, target in enumerate(TARGETS):
        if target not in pivot:
            continue
        series = pivot[target].dropna()
        for key, value in series.items():
            position = key_position.get(key)
            if position is not None:
                values[position, target_index] = float(value)
                available[position, target_index] = 1.0
    return values, available


def target_dense_features(
    base_dense: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    target: str,
) -> np.ndarray:
    values = cross_values.copy()
    available = cross_available.copy()
    target_index = TARGETS.index(target)
    values[:, target_index] = np.nan
    available[:, target_index] = 0.0
    return np.hstack([base_dense, values, available]).astype(np.float64, copy=False)


def fit_dense_preprocessor(
    dense: np.ndarray,
    train_index: np.ndarray,
    prediction_index: np.ndarray,
    absolute_limit: float,
):
    sanitized = np.asarray(dense, dtype=np.float64).copy()
    invalid = ~np.isfinite(sanitized) | (np.abs(sanitized) > absolute_limit)
    sanitized[invalid] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(sanitized[train_index])
    prediction_imputed = imputer.transform(sanitized[prediction_index])
    train_scaled = scaler.fit_transform(train_imputed)
    prediction_scaled = scaler.transform(prediction_imputed)
    return train_imputed, prediction_imputed, train_scaled, prediction_scaled


def clip_prediction(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-8)
    return np.clip(prediction, q01 - 2.0 * margin, q99 + 2.0 * margin)


def predict_base_models(
    dense: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    fingerprints: list[Any],
    y: np.ndarray,
    train_index: np.ndarray,
    prediction_index: np.ndarray,
    config: dict[str, Any],
    target: str,
) -> np.ndarray:
    large = target in {"tg", "egc"}
    alpha = float(config["ridge_alpha_large"] if large else config["ridge_alpha_sparse"])
    leaf = int(config["extra_trees_min_leaf_large"] if large else config["extra_trees_min_leaf_sparse"])
    train_imputed, prediction_imputed, train_scaled, prediction_scaled = fit_dense_preprocessor(
        dense,
        train_index,
        prediction_index,
        absolute_limit=float(config["dense_abs_limit"]),
    )
    sparse_train = sparse.hstack(
        [part[train_index] for part in sparse_parts] + [sparse.csr_matrix(train_scaled)],
        format="csr",
    )
    sparse_prediction = sparse.hstack(
        [part[prediction_index] for part in sparse_parts] + [sparse.csr_matrix(prediction_scaled)],
        format="csr",
    )
    sparse_model = Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1e-4)
    sparse_model.fit(sparse_train, y[train_index])
    sparse_prediction_values = sparse_model.predict(sparse_prediction)

    dense_model = Ridge(alpha=alpha)
    dense_model.fit(train_scaled, y[train_index])
    dense_prediction_values = dense_model.predict(prediction_scaled)

    tree_model = ExtraTreesRegressor(
        n_estimators=int(config["extra_trees_estimators"]),
        min_samples_leaf=leaf,
        max_features=0.75,
        random_state=int(config["seed"]),
        n_jobs=2,
    )
    tree_model.fit(train_imputed, y[train_index])
    tree_prediction_values = tree_model.predict(prediction_imputed)

    local_prediction_values = tanimoto_prediction(
        fingerprints,
        y,
        train_index,
        prediction_index,
        k=int(config["tanimoto_knn_k"]),
        krr_alpha=float(config["tanimoto_krr_alpha"]),
    )
    prediction = np.column_stack(
        [
            sparse_prediction_values,
            dense_prediction_values,
            tree_prediction_values,
            local_prediction_values,
        ]
    )
    for column in range(prediction.shape[1]):
        prediction[:, column] = clip_prediction(y[train_index], prediction[:, column])
    return prediction


def tanimoto_matrix(left: list[Any], right: list[Any]) -> np.ndarray:
    matrix = np.empty((len(left), len(right)), dtype=np.float64)
    for row, fingerprint in enumerate(left):
        matrix[row] = DataStructs.BulkTanimotoSimilarity(fingerprint, right)
    return matrix


def tanimoto_prediction(
    fingerprints: list[Any],
    y: np.ndarray,
    train_index: np.ndarray,
    prediction_index: np.ndarray,
    k: int,
    krr_alpha: float,
) -> np.ndarray:
    train_fingerprints = [fingerprints[index] for index in train_index]
    prediction_fingerprints = [fingerprints[index] for index in prediction_index]
    train_y = y[train_index]
    if len(train_index) <= 600:
        kernel_train = tanimoto_matrix(train_fingerprints, train_fingerprints)
        kernel_prediction = tanimoto_matrix(prediction_fingerprints, train_fingerprints)
        center = float(np.mean(train_y))
        kernel_train.flat[:: len(kernel_train) + 1] += krr_alpha
        coefficient = np.linalg.solve(kernel_train, train_y - center)
        return center + kernel_prediction @ coefficient
    output = np.empty(len(prediction_index), dtype=np.float64)
    batch_size = 256
    take = min(k, len(train_index))
    for start in range(0, len(prediction_index), batch_size):
        stop = min(start + batch_size, len(prediction_index))
        similarity = tanimoto_matrix(prediction_fingerprints[start:stop], train_fingerprints)
        nearest = np.argpartition(similarity, -take, axis=1)[:, -take:]
        for local_row in range(stop - start):
            selected = nearest[local_row]
            weights = np.maximum(similarity[local_row, selected], 1e-6) ** 4
            output[start + local_row] = float(np.dot(weights, train_y[selected]) / np.sum(weights))
    return output


def blend_from_oof(y: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, float, str, float]:
    centered_base = base - np.mean(base, axis=0, keepdims=True)
    centered_y = y - np.mean(y)
    weights, _ = nnls(centered_base, centered_y)
    if float(np.sum(weights)) <= 0:
        weights = np.full(base.shape[1], 1.0 / base.shape[1])
    else:
        weights = weights / np.sum(weights)
    intercept = float(np.mean(y - base @ weights))
    blend = base @ weights + intercept
    blend_score = float(r2_score(y, blend))
    base_scores = [float(r2_score(y, base[:, column])) for column in range(base.shape[1])]
    best_index = int(np.argmax(base_scores))
    if base_scores[best_index] > blend_score:
        weights = np.zeros(base.shape[1], dtype=np.float64)
        weights[best_index] = 1.0
        intercept = 0.0
        return weights, intercept, f"base_{best_index}", base_scores[best_index]
    return weights, intercept, "nonnegative_blend", blend_score


def fit_targets(
    pooled: pd.DataFrame,
    test: pd.DataFrame,
    keys: list[str],
    dense_base: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    fingerprints: list[Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    key_to_index = {key: index for index, key in enumerate(keys)}
    detail_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    target_reports: dict[str, Any] = {}
    model_names = ["sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local"]
    for target in TARGETS:
        target_train = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_test = test[test["target_type"] == target].reset_index(drop=False)
        train_index = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
        test_index = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
        y = target_train["target"].to_numpy(float)
        dense = target_dense_features(dense_base, cross_values, cross_available, target)
        folds = KFold(
            n_splits=int(config["folds"]),
            shuffle=True,
            random_state=int(config["seed"]),
        )
        oof = np.zeros((len(target_train), len(model_names)), dtype=np.float64)
        fold_reports: list[dict[str, Any]] = []
        for fold_id, (local_train, local_validation) in enumerate(folds.split(np.arange(len(target_train)))):
            local_y_global = np.full(len(keys), np.nan, dtype=np.float64)
            local_y_global[train_index] = y
            fold_prediction = predict_base_models(
                dense,
                sparse_parts,
                fingerprints,
                local_y_global,
                train_index[local_train],
                train_index[local_validation],
                config,
                target,
            )
            oof[local_validation] = fold_prediction
            fold_reports.append(
                {
                    "fold": fold_id,
                    "rows": int(len(local_validation)),
                    "base_r2": {
                        name: float(r2_score(y[local_validation], fold_prediction[:, column]))
                        for column, name in enumerate(model_names)
                    },
                }
            )
        weights, intercept, selected, selected_r2 = blend_from_oof(y, oof)
        oof_blend = oof @ weights + intercept
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[train_index] = y
        final_base = predict_base_models(
            dense,
            sparse_parts,
            fingerprints,
            y_global,
            train_index,
            test_index,
            config,
            target,
        )
        final_prediction = final_base @ weights + intercept
        target_reports[target] = {
            "model_rows": int(len(target_train)),
            "measurements": int(target_train["measurements"].sum()),
            "oof_base_r2": {
                name: float(r2_score(y, oof[:, column])) for column, name in enumerate(model_names)
            },
            "oof_base_mae": {
                name: float(mean_absolute_error(y, oof[:, column])) for column, name in enumerate(model_names)
            },
            "selected": selected,
            "selected_oof_r2": selected_r2,
            "blend_weights": {name: float(weights[column]) for column, name in enumerate(model_names)},
            "blend_intercept": intercept,
            "folds": fold_reports,
        }
        for row, prediction_set, prediction in zip(
            target_test.itertuples(index=False), final_base, final_prediction, strict=True
        ):
            record = {
                "id": int(row.id),
                "target_type": target,
                "model_prediction": float(prediction),
            }
            record.update({name: float(prediction_set[column]) for column, name in enumerate(model_names)})
            detail_rows.append(record)
        for row, prediction_set, prediction in zip(
            target_train.itertuples(index=False), oof, oof_blend, strict=True
        ):
            record = {
                "canonical": row.canonical,
                "target_type": target,
                "target": float(row.target),
                "prediction": float(prediction),
            }
            record.update({name: float(prediction_set[column]) for column, name in enumerate(model_names)})
            oof_rows.append(record)
    report = {
        "target_reports": target_reports,
        "mean_selected_oof_r2": float(np.mean([target_reports[target]["selected_oof_r2"] for target in TARGETS])),
        "model_names": model_names,
    }
    return pd.DataFrame(detail_rows), pd.DataFrame(oof_rows), report


def apply_official_overrides(
    detail: pd.DataFrame,
    test: pd.DataFrame,
    raw_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_map = unique_mapping(raw_labels, ["smiles", "target_type"])
    canonical_map = unique_mapping(raw_labels, ["canonical", "target_type"])
    ordered = test[["id", "smiles", "canonical", "target_type"]].merge(
        detail,
        on=["id", "target_type"],
        how="left",
        validate="one_to_one",
    )
    ordered["target"] = ordered["model_prediction"].astype(float)
    ordered["override"] = "model"
    for index, row in ordered.iterrows():
        raw_key = (row["smiles"], row["target_type"])
        canonical_key = (row["canonical"], row["target_type"])
        if raw_key in raw_map:
            ordered.at[index, "target"] = raw_map[raw_key]
            ordered.at[index, "override"] = "official_raw_unique"
        elif canonical_key in canonical_map:
            ordered.at[index, "target"] = canonical_map[canonical_key]
            ordered.at[index, "override"] = "official_canonical_unique"
    counts = ordered.groupby(["target_type", "override"]).size().unstack(fill_value=0)
    report = {
        "total_overrides": int((ordered["override"] != "model").sum()),
        "by_target_and_route": {
            target: {route: int(value) for route, value in row.items()}
            for target, row in counts.iterrows()
        },
        "raw_unique_map_keys": int(len(raw_map)),
        "canonical_unique_map_keys": int(len(canonical_map)),
    }
    return ordered, report


def package_manifest(run_dir: Path, paths: list[Path]) -> None:
    lines = []
    for path in paths:
        lines.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    data_dir: str | Path | None,
    output_path: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    configuration = dict(DEFAULT_CONFIG)
    if config:
        configuration.update(config)
    np.random.seed(int(configuration["seed"]))
    resolved_data = resolve_data_dir(data_dir)
    output = Path(output_path)
    runtime = Path(run_dir)
    runtime.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    train, test, archive, inputs = load_inputs(resolved_data)
    raw_labels, pooled = build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = build_molecules(keys)
    descriptor, descriptor_names = descriptor_matrix(molecules)
    physical, physical_names = physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = cross_property_arrays(pooled, keys)
    sparse_parts = [
        morgan_count_matrix(molecules, radius=2, bits=int(configuration["morgan_bits"])),
        morgan_count_matrix(molecules, radius=3, bits=int(configuration["morgan_bits"])),
        text_matrix(keys, int(configuration["text_features"])),
    ]
    fingerprints = morgan_bits(molecules, radius=2, bits=int(configuration["morgan_bits"]))
    detail, oof, model_report = fit_targets(
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
    final_detail, override_report = apply_official_overrides(detail, test, raw_labels)
    submission = final_detail[["id", "target"]].copy()
    if len(submission) != len(test) or not submission["id"].equals(test["id"]):
        raise RuntimeError("Submission row order differs from official test")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("Submission contains duplicate IDs or non-finite targets")
    submission.to_csv(output, index=False)
    detail_path = runtime / "test_predictions_detail.csv"
    oof_path = runtime / "oof_predictions.csv"
    detail_for_storage = final_detail.drop(columns=["smiles", "canonical"])
    detail_for_storage.to_csv(detail_path, index=False)
    oof.to_csv(oof_path, index=False)
    config_path = runtime / "config.json"
    write_json(config_path, configuration)
    environment_path = runtime / "environment.txt"
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
        "schema_version": "ppp.round2.initial-reference-run.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "official current+archive target-specific classical ensemble with cross-property covariates",
        "config": configuration,
        "config_sha256": canonical_json_hash(configuration),
        "inputs": inputs,
        "rows": {
            "current_train": int(len(train)),
            "archive_train": int(len(archive)),
            "raw_label_pool": int(len(raw_labels)),
            "canonical_model_rows": int(len(pooled)),
            "test": int(len(test)),
            "unique_feature_structures": int(len(keys)),
        },
        "features": {
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "cross_property_values": len(TARGETS) - 1,
            "cross_property_availability": len(TARGETS) - 1,
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
            "sha256": sha256_file(output),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    report_path = runtime / "report.json"
    write_json(report_path, report)
    command_path = runtime / "command.txt"
    command_path.write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    package_manifest(
        runtime,
        [config_path, environment_path, detail_path, oof_path, report_path, command_path],
    )
    return report


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = run_pipeline(args.data_dir, args.output, args.run_dir)
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


# NOTEBOOK_ENTRYPOINT
if __name__ == "__main__":
    cli_main()
