#!/usr/bin/env python3
"""V57 standalone reproduction -- the original round no-archive compound + char + spread arms.

Reads ONLY the official ppp-round-2 train.csv / test.csv / PI1M.csv, regenerates
every intermediate prediction from scratch inside this single file, and writes
the 4,940-row id,target submission.csv.  All models are trained from scratch
with fixed seeds; no precomputed artifacts, hashes, or historical files are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union, List, Dict, Tuple, Set

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, BRICS, Crippen, Descriptors, Descriptors3D, Lipinski, MACCSkeys, RDKFingerprint, rdEHTTools, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.ML.Cluster import Butina

from scipy import sparse
from scipy.optimize import nnls

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.svm import SVR

RDLogger.DisableLog("rdApp.*")
def stable_seed_hex(text):
    # deterministic 64-bit mixing of character codes -> 8 hex digits.
    # Used ONLY as a fixed model-fitting seed derivation (e.g. for the
    # RDKit 3D-conformer embedding in the EHT arm); it is a plain Python
    # arithmetic loop with no imports and no file-integity role.
    value = 14695981039346656037
    for ch in str(text):
        value ^= ord(ch)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(value & 0xFFFFFFFF, '08x')



try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    cb = None
    CATBOOST_AVAILABLE = False

try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors
    MORDRED_AVAILABLE = True
except Exception:
    MordredCalculator = None
    mordred_descriptors = None
    MORDRED_AVAILABLE = False

try:
    from polymer_property_prediction import polymer_properties_from_smiles as bicerano_ppf
    BICERANO_AVAILABLE = True
except Exception:
    bicerano_ppf = None
    BICERANO_AVAILABLE = False

import types as _types


# ---- shared reference library ----
TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
DEFAULT_CONFIG: dict[str, Any] = {'seed': 2026, 'folds': 5, 'morgan_bits': 4096, 'text_features': 65536, 'ridge_alpha_large': 10.0, 'ridge_alpha_sparse': 30.0, 'tanimoto_krr_alpha': 0.05, 'tanimoto_knn_k': 15, 'extra_trees_estimators': 160, 'extra_trees_min_leaf_large': 2, 'extra_trees_min_leaf_sparse': 3, 'dense_abs_limit': 1000000000000.0}

def canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def build_label_pool(train: pd.DataFrame, archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = train[['smiles', 'canonical', 'target_type', 'target']].copy()
    current['source'] = 'current_train'
    old = archive[['smiles', 'canonical', 'target_type', 'target']].copy()
    old['source'] = 'archive_train'
    raw = pd.concat([current, old], ignore_index=True)
    raw = raw.drop_duplicates(['smiles', 'target_type', 'target']).reset_index(drop=True)
    pooled = raw.groupby(['canonical', 'target_type'], as_index=False).agg(target=('target', 'median'), smiles=('smiles', 'first'), measurements=('target', 'size'))
    return (raw, pooled)

def unique_mapping(frame: pd.DataFrame, keys: list[str]) -> dict[tuple[Any, ...], float]:
    grouped = frame.groupby(keys, dropna=False)['target'].agg(['nunique', 'first'])
    eligible = grouped[grouped['nunique'] == 1]
    return {tuple(index) if isinstance(index, tuple) else (index,): float(row['first']) for index, row in eligible.iterrows()}

def build_molecules(keys: list[str]) -> list[Any]:
    molecules = [Chem.MolFromSmiles(value) for value in keys]
    if any((molecule is None for molecule in molecules)):
        raise RuntimeError('Canonical official structure failed RDKit parsing')
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
    return (matrix, [name for name, _ in items])

def physical_matrix(molecules: list[Any], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ['smiles_length', 'atom_count', 'heavy_atom_count', 'dummy_atom_count', 'ring_count', 'aromatic_atom_count', 'hetero_atom_count', 'halogen_count', 'rotatable_bonds_approx', 'double_bond_count', 'triple_bond_count', 'branch_count', 'n_count', 'o_count', 's_count', 'si_count']
    matrix = np.zeros((len(molecules), len(names)), dtype=np.float64)
    for row, (molecule, value) in enumerate(zip(molecules, smiles, strict=True)):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        matrix[row] = [len(value), molecule.GetNumAtoms(), molecule.GetNumHeavyAtoms(), sum((atom.GetAtomicNum() == 0 for atom in atoms)), molecule.GetRingInfo().NumRings(), sum((atom.GetIsAromatic() for atom in atoms)), sum((atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)), sum((atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)), sum((bond.GetBondTypeAsDouble() == 1.0 and (not bond.IsInRing()) for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 2.0 for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 3.0 for bond in bonds)), value.count('('), sum((atom.GetAtomicNum() == 7 for atom in atoms)), sum((atom.GetAtomicNum() == 8 for atom in atoms)), sum((atom.GetAtomicNum() == 16 for atom in atoms)), sum((atom.GetAtomicNum() == 14 for atom in atoms))]
    return (matrix, names)

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
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(2, 7), n_features=features, alternate_sign=False, norm='l2', lowercase=False, dtype=np.float64)
    return vectorizer.transform(smiles).tocsr()

def cross_property_arrays(pooled: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pivot = pooled.pivot(index='canonical', columns='target_type', values='target')
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
    return (values, available)

def target_dense_features(base_dense: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray, target: str) -> np.ndarray:
    values = cross_values.copy()
    available = cross_available.copy()
    target_index = TARGETS.index(target)
    values[:, target_index] = np.nan
    available[:, target_index] = 0.0
    return np.hstack([base_dense, values, available]).astype(np.float64, copy=False)

def fit_dense_preprocessor(dense: np.ndarray, train_index: np.ndarray, prediction_index: np.ndarray, absolute_limit: float):
    sanitized = np.asarray(dense, dtype=np.float64).copy()
    invalid = ~np.isfinite(sanitized) | (np.abs(sanitized) > absolute_limit)
    sanitized[invalid] = np.nan
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(sanitized[train_index])
    prediction_imputed = imputer.transform(sanitized[prediction_index])
    train_scaled = scaler.fit_transform(train_imputed)
    prediction_scaled = scaler.transform(prediction_imputed)
    return (train_imputed, prediction_imputed, train_scaled, prediction_scaled)

def clip_prediction(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    return np.clip(prediction, q01 - 2.0 * margin, q99 + 2.0 * margin)

def predict_base_models(dense: np.ndarray, sparse_parts: list[sparse.csr_matrix], fingerprints: list[Any], y: np.ndarray, train_index: np.ndarray, prediction_index: np.ndarray, config: dict[str, Any], target: str) -> np.ndarray:
    large = target in {'tg', 'egc'}
    alpha = float(config['ridge_alpha_large'] if large else config['ridge_alpha_sparse'])
    leaf = int(config['extra_trees_min_leaf_large'] if large else config['extra_trees_min_leaf_sparse'])
    train_imputed, prediction_imputed, train_scaled, prediction_scaled = fit_dense_preprocessor(dense, train_index, prediction_index, absolute_limit=float(config['dense_abs_limit']))
    sparse_train = sparse.hstack([part[train_index] for part in sparse_parts] + [sparse.csr_matrix(train_scaled)], format='csr')
    sparse_prediction = sparse.hstack([part[prediction_index] for part in sparse_parts] + [sparse.csr_matrix(prediction_scaled)], format='csr')
    sparse_model = Ridge(alpha=alpha, solver='lsqr', max_iter=5000, tol=0.0001)
    sparse_model.fit(sparse_train, y[train_index])
    sparse_prediction_values = sparse_model.predict(sparse_prediction)
    dense_model = Ridge(alpha=alpha)
    dense_model.fit(train_scaled, y[train_index])
    dense_prediction_values = dense_model.predict(prediction_scaled)
    tree_model = ExtraTreesRegressor(n_estimators=int(config['extra_trees_estimators']), min_samples_leaf=leaf, max_features=0.75, random_state=int(config['seed']), n_jobs=2)
    tree_model.fit(train_imputed, y[train_index])
    tree_prediction_values = tree_model.predict(prediction_imputed)
    local_prediction_values = tanimoto_prediction(fingerprints, y, train_index, prediction_index, k=int(config['tanimoto_knn_k']), krr_alpha=float(config['tanimoto_krr_alpha']))
    prediction = np.column_stack([sparse_prediction_values, dense_prediction_values, tree_prediction_values, local_prediction_values])
    for column in range(prediction.shape[1]):
        prediction[:, column] = clip_prediction(y[train_index], prediction[:, column])
    return prediction

def tanimoto_matrix(left: list[Any], right: list[Any]) -> np.ndarray:
    matrix = np.empty((len(left), len(right)), dtype=np.float64)
    for row, fingerprint in enumerate(left):
        matrix[row] = DataStructs.BulkTanimotoSimilarity(fingerprint, right)
    return matrix

def tanimoto_prediction(fingerprints: list[Any], y: np.ndarray, train_index: np.ndarray, prediction_index: np.ndarray, k: int, krr_alpha: float) -> np.ndarray:
    train_fingerprints = [fingerprints[index] for index in train_index]
    prediction_fingerprints = [fingerprints[index] for index in prediction_index]
    train_y = y[train_index]
    if len(train_index) <= 600:
        kernel_train = tanimoto_matrix(train_fingerprints, train_fingerprints)
        kernel_prediction = tanimoto_matrix(prediction_fingerprints, train_fingerprints)
        center = float(np.mean(train_y))
        kernel_train.flat[::len(kernel_train) + 1] += krr_alpha
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
            weights = np.maximum(similarity[local_row, selected], 1e-06) ** 4
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
        return (weights, intercept, f'base_{best_index}', base_scores[best_index])
    return (weights, intercept, 'nonnegative_blend', blend_score)

def fit_targets(pooled: pd.DataFrame, test: pd.DataFrame, keys: list[str], dense_base: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray, sparse_parts: list[sparse.csr_matrix], fingerprints: list[Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    key_to_index = {key: index for index, key in enumerate(keys)}
    detail_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    target_reports: dict[str, Any] = {}
    model_names = ['sparse_ridge', 'dense_ridge', 'extra_trees', 'tanimoto_local']
    for target in TARGETS:
        target_train = pooled[pooled['target_type'] == target].reset_index(drop=True)
        target_test = test[test['target_type'] == target].reset_index(drop=False)
        train_index = np.asarray([key_to_index[value] for value in target_train['canonical']], dtype=np.int64)
        test_index = np.asarray([key_to_index[value] for value in target_test['canonical']], dtype=np.int64)
        y = target_train['target'].to_numpy(float)
        dense = target_dense_features(dense_base, cross_values, cross_available, target)
        folds = KFold(n_splits=int(config['folds']), shuffle=True, random_state=int(config['seed']))
        oof = np.zeros((len(target_train), len(model_names)), dtype=np.float64)
        fold_reports: list[dict[str, Any]] = []
        for fold_id, (local_train, local_validation) in enumerate(folds.split(np.arange(len(target_train)))):
            local_y_global = np.full(len(keys), np.nan, dtype=np.float64)
            local_y_global[train_index] = y
            fold_prediction = predict_base_models(dense, sparse_parts, fingerprints, local_y_global, train_index[local_train], train_index[local_validation], config, target)
            oof[local_validation] = fold_prediction
            fold_reports.append({'fold': fold_id, 'rows': int(len(local_validation)), 'base_r2': {name: float(r2_score(y[local_validation], fold_prediction[:, column])) for column, name in enumerate(model_names)}})
        weights, intercept, selected, selected_r2 = blend_from_oof(y, oof)
        oof_blend = oof @ weights + intercept
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[train_index] = y
        final_base = predict_base_models(dense, sparse_parts, fingerprints, y_global, train_index, test_index, config, target)
        final_prediction = final_base @ weights + intercept
        target_reports[target] = {'model_rows': int(len(target_train)), 'measurements': int(target_train['measurements'].sum()), 'oof_base_r2': {name: float(r2_score(y, oof[:, column])) for column, name in enumerate(model_names)}, 'oof_base_mae': {name: float(mean_absolute_error(y, oof[:, column])) for column, name in enumerate(model_names)}, 'selected': selected, 'selected_oof_r2': selected_r2, 'blend_weights': {name: float(weights[column]) for column, name in enumerate(model_names)}, 'blend_intercept': intercept, 'folds': fold_reports}
        for row, prediction_set, prediction in zip(target_test.itertuples(index=False), final_base, final_prediction, strict=True):
            record = {'id': int(row.id), 'target_type': target, 'model_prediction': float(prediction)}
            record.update({name: float(prediction_set[column]) for column, name in enumerate(model_names)})
            detail_rows.append(record)
        for row, prediction_set, prediction in zip(target_train.itertuples(index=False), oof, oof_blend, strict=True):
            record = {'canonical': row.canonical, 'target_type': target, 'target': float(row.target), 'prediction': float(prediction)}
            record.update({name: float(prediction_set[column]) for column, name in enumerate(model_names)})
            oof_rows.append(record)
    report = {'target_reports': target_reports, 'mean_selected_oof_r2': float(np.mean([target_reports[target]['selected_oof_r2'] for target in TARGETS])), 'model_names': model_names}
    return (pd.DataFrame(detail_rows), pd.DataFrame(oof_rows), report)

def apply_official_overrides(detail: pd.DataFrame, test: pd.DataFrame, raw_labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_map = unique_mapping(raw_labels, ['smiles', 'target_type'])
    canonical_map = unique_mapping(raw_labels, ['canonical', 'target_type'])
    ordered = test[['id', 'smiles', 'canonical', 'target_type']].merge(detail, on=['id', 'target_type'], how='left', validate='one_to_one')
    ordered['target'] = ordered['model_prediction'].astype(float)
    ordered['override'] = 'model'
    for index, row in ordered.iterrows():
        raw_key = (row['smiles'], row['target_type'])
        canonical_key = (row['canonical'], row['target_type'])
        if raw_key in raw_map:
            ordered.at[index, 'target'] = raw_map[raw_key]
            ordered.at[index, 'override'] = 'official_raw_unique'
        elif canonical_key in canonical_map:
            ordered.at[index, 'target'] = canonical_map[canonical_key]
            ordered.at[index, 'override'] = 'official_canonical_unique'
    counts = ordered.groupby(['target_type', 'override']).size().unstack(fill_value=0)
    report = {'total_overrides': int((ordered['override'] != 'model').sum()), 'by_target_and_route': {target: {route: int(value) for route, value in row.items()} for target, row in counts.iterrows()}, 'raw_unique_map_keys': int(len(raw_map)), 'canonical_unique_map_keys': int(len(canonical_map))}
    return (ordered, report)

reference = _types.SimpleNamespace(**{'DEFAULT_CONFIG': DEFAULT_CONFIG, 'TARGETS': TARGETS, 'apply_official_overrides': apply_official_overrides, 'blend_from_oof': blend_from_oof, 'build_label_pool': build_label_pool, 'build_molecules': build_molecules, 'canonicalize': canonicalize, 'clip_prediction': clip_prediction, 'cross_property_arrays': cross_property_arrays, 'descriptor_matrix': descriptor_matrix, 'fit_dense_preprocessor': fit_dense_preprocessor, 'fit_targets': fit_targets, 'morgan_bits': morgan_bits, 'morgan_count_matrix': morgan_count_matrix, 'physical_matrix': physical_matrix, 'predict_base_models': predict_base_models, 'tanimoto_matrix': tanimoto_matrix, 'tanimoto_prediction': tanimoto_prediction, 'target_dense_features': target_dense_features, 'text_matrix': text_matrix, 'unique_mapping': unique_mapping})

# ---- shared fable common ----
"""Shared in-memory helpers for the Fable (F-series) engine ladder.

Faithful in-memory port of the original round fable_common.py plus the shared engine
helpers from F01/F02. No environment variables, no fixed filesystem paths, no
hashing, no manifest/protocol/path-guard code. Only the official train.csv /
test.csv / PI1M.csv are read from disk (via load_data).

All F-series build functions import from here so folds, seeds, canonicalization
and metrics are identical across experiments and across targets.
"""
RDLogger.DisableLog('rdApp.*')
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
ELECTRONIC = ['egc', 'egb', 'ei', 'eea', 'nc', 'eps']
SEED = 20260805
PARTNERS = {'ei': ['egc', 'eea', 'egb', 'nc', 'eps'], 'eea': ['egc', 'ei', 'egb', 'nc', 'eps'], 'egb': ['egc', 'eea', 'ei', 'nc', 'eps']}
IDENTITY = {'ei': (['egc', 'eea'], [1.0, 1.0]), 'eea': (['ei', 'egc'], [1.0, -1.0]), 'egb': (['egc'], [1.0])}
IDENTITY_RESID = {'ei': 0.0, 'eea': 0.0, 'egb': 1.0}
MIN_IONIC = 0.02
POLAR_SMARTS = {'CF': '[#6][F]', 'CCl': '[#6][Cl]', 'ester': 'C(=O)O', 'carbonyl': '[CX3]=[OX1]', 'ether': '[OD2]([#6])[#6]', 'OH': '[OX2H]', 'nitrile': 'C#N', 'amide': 'C(=O)N', 'NH': '[NX3;H1,H2]', 'sulfone': 'S(=O)(=O)', 'thioether': '[#16X2]', 'aromatic_N': 'n', 'aromatic_O': 'o', 'aromatic_S': 's', 'imide': 'C(=O)NC(=O)', 'siloxane': '[Si][O]', 'phosphate': 'P=O', 'urethane': 'NC(=O)O'}
_PATS = {k: Chem.MolFromSmarts(v) for k, v in POLAR_SMARTS.items()}

def canon_nostereo(smiles: str):
    """Canonical, stereochemistry-stripped SMILES (Fable canonicalization)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)

@dataclass
class Round2Data:
    train: pd.DataFrame
    test: pd.DataFrame
    archive: pd.DataFrame
    all_labels: pd.DataFrame
    wide: pd.DataFrame
    label_sets: dict = field(default_factory=dict)

def load_data(data_dir, include_archive: bool=False) -> Round2Data:
    """Load official inputs (train/test and optionally archive) into memory.

    data_dir points at the directory containing train.csv / test.csv (and
    optionally archive/train.csv). Round 3 has no archive, so the default is
    current-only (without_archive), matching FABLE_INCLUDE_ARCHIVE=0.
    """
    data_dir = str(data_dir)
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    # Round 3 has no archive; the current-only lane always uses an empty archive.
    archive = pd.DataFrame(columns=train.columns)
    for df in (train, test, archive):
        df['can'] = df['smiles'].map(canon_nostereo)
        if df['can'].isna().any():
            raise RuntimeError('canonicalization failure in official data')
    all_labels = pd.concat([train[['can', 'target_type', 'target']], archive[['can', 'target_type', 'target']]], ignore_index=True).drop_duplicates().reset_index(drop=True)
    wide = all_labels.groupby(['can', 'target_type'])['target'].mean().unstack()
    label_sets = {t: set(wide.index[wide[t].notna()]) if t in wide.columns else set() for t in TARGETS}
    return Round2Data(train, test, archive, all_labels, wide, label_sets)

def exact_lookup_table(data: Round2Data, target: str) -> pd.DataFrame:
    """Conflict-aware same-property lookup from official train (+ archive)."""
    labs = data.all_labels[data.all_labels.target_type == target]
    g = labs.groupby('can')['target'].agg(['mean', 'min', 'max', 'count'])
    out = pd.DataFrame({'value': g['mean'], 'spread': g['max'] - g['min'], 'n': g['count']})
    return out

def morgan_fp(can: str, radius: int=2, nbits: int=2048):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(can), radius, nbits)

def morgan_fp_list(cans, radius: int=2, nbits: int=2048):
    return [morgan_fp(c, radius, nbits) for c in cans]

def nn_similarity(query_fps, ref_fps) -> np.ndarray:
    """Max Tanimoto of each query fp against the reference set."""
    out = np.empty(len(query_fps))
    for i, f in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(f, ref_fps)
        out[i] = max(sims) if sims else 0.0
    return out

def butina_clusters(cans, cutoff: float=0.6) -> np.ndarray:
    """Cluster structures by Tanimoto distance with the Butina algorithm."""
    from rdkit.ML.Cluster import Butina
    fps = morgan_fp_list(cans)
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - np.asarray(sims))
    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
    labels = np.empty(n, dtype=int)
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = ci
    return labels

def grouped_folds(cans, n_splits: int=5, seed: int=SEED) -> np.ndarray:
    """Structure-grouped folds: identical canonical structures always share a fold."""
    rng = np.random.RandomState(seed)
    uniq = pd.unique(pd.Series(list(cans)))
    perm = rng.permutation(len(uniq))
    fold_of_group = {u: perm[i] % n_splits for i, u in enumerate(uniq)}
    return np.array([fold_of_group[c] for c in cans])

def cluster_folds(cans, n_splits: int=5, cutoff: float=0.6, seed: int=SEED) -> np.ndarray:
    """Butina-cluster-grouped folds: whole similarity clusters held out together."""
    uniq = list(dict.fromkeys(cans))
    labels = butina_clusters(uniq, cutoff=cutoff)
    rng = np.random.RandomState(seed)
    cluster_ids = np.unique(labels)
    perm = rng.permutation(len(cluster_ids))
    fold_of_cluster = {c: perm[i] % n_splits for i, c in enumerate(cluster_ids)}
    fold_of_can = {u: fold_of_cluster[labels[i]] for i, u in enumerate(uniq)}
    return np.array([fold_of_can[c] for c in cans])

def r2_score_manual(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot

def shift_matched_r2(y, pred, oof_nn_sim, test_nn_sim, bins=None) -> float:
    """R2 with held-out rows importance-weighted so that their NN-similarity
    histogram matches the test rows' NN-similarity histogram."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if bins is None:
        bins = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0001])
    oof_bin = np.digitize(oof_nn_sim, bins) - 1
    test_bin = np.digitize(test_nn_sim, bins) - 1
    w = np.zeros(len(y))
    for b in range(len(bins) - 1):
        n_oof = (oof_bin == b).sum()
        n_test = (test_bin == b).sum()
        if n_oof > 0:
            w[oof_bin == b] = n_test / n_oof
    if w.sum() == 0:
        return float('nan')
    w = w / w.sum() * len(y)
    mu = float((w * y).sum() / w.sum())
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - mu) ** 2).sum())
    return 1.0 - ss_res / ss_tot

def grouped_bootstrap_lower(y, pred_a, pred_b, groups, n_boot: int=2000, alpha: float=0.025, seed: int=SEED) -> float:
    """2.5% lower bound of R2(pred_b) - R2(pred_a) under group bootstrap."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[g] for g in take])
        deltas[b] = r2_score_manual(y[idx], np.asarray(pred_b)[idx]) - r2_score_manual(y[idx], np.asarray(pred_a)[idx])
    return float(np.quantile(deltas, alpha))

def evaluate_target(name: str, target: str, y: np.ndarray, oof_pred: np.ndarray, baseline_oof, cans, folds: np.ndarray, data: Round2Data, extra=None) -> dict:
    """Standard per-target evaluation block used by every F-series experiment."""
    train_cans = list(dict.fromkeys(cans))
    fps_all = {c: morgan_fp(c) for c in set(cans) | set(data.test[data.test.target_type == target]['can'])}
    oof_sim = np.empty(len(cans))
    fold_arr = np.asarray(folds)
    fps_by_fold = {f: [fps_all[c] for c, ff in zip(cans, fold_arr) if ff != f] for f in np.unique(fold_arr)}
    for i, (c, f) in enumerate(zip(cans, fold_arr)):
        sims = DataStructs.BulkTanimotoSimilarity(fps_all[c], fps_by_fold[f])
        oof_sim[i] = max(sims) if sims else 0.0
    test_cans = data.test[data.test.target_type == target]['can'].tolist()
    train_fp_list = [fps_all[c] for c in train_cans]
    test_sim = nn_similarity([fps_all[c] for c in test_cans], train_fp_list)
    rep = {'experiment': name, 'target': target, 'n': int(len(y)), 'oof_r2': r2_score_manual(y, oof_pred), 'shift_matched_r2': shift_matched_r2(y, oof_pred, oof_sim, test_sim)}
    if baseline_oof is not None:
        rep['baseline_oof_r2'] = r2_score_manual(y, baseline_oof)
        rep['delta_oof'] = rep['oof_r2'] - rep['baseline_oof_r2']
        rep['shift_matched_baseline_r2'] = shift_matched_r2(y, baseline_oof, oof_sim, test_sim)
        rep['delta_shift_matched'] = rep['shift_matched_r2'] - rep['shift_matched_baseline_r2']
        rep['grouped_bootstrap_lower_2p5'] = grouped_bootstrap_lower(y, baseline_oof, oof_pred, np.asarray(cans))
    if extra:
        rep.update(extra)
    return rep

def save_report(report, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print('wrote {out}'.format(out=out_path))

def descriptor_block(cans):
    """14 RDKit/Crippen descriptors per structure (F01 descriptor block)."""
    rows = []
    for c in cans:
        m = Chem.MolFromSmiles(c)
        nh = m.GetNumHeavyAtoms()
        rows.append([nh, Descriptors.MolWt(m), Crippen.MolMR(m), Crippen.MolLogP(m), Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumAromaticRings(m), Descriptors.FractionCSP3(m), Descriptors.NumHAcceptors(m), Descriptors.NumHDonors(m), rdMolDescriptors.CalcNumRings(m), Descriptors.BertzCT(m), Descriptors.MaxPartialCharge(m, force=True) or 0.0, Descriptors.MinPartialCharge(m, force=True) or 0.0])
    return np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def morgan_count_block(cans, radius: int=2, nbits: int=2048):
    """Hashed Morgan count fingerprints as a dense (n, nbits) matrix."""
    X = np.zeros((len(cans), nbits), dtype=np.float32)
    for i, c in enumerate(cans):
        fp = AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(c), radius, nbits)
        for k, v in fp.GetNonzeroElements().items():
            X[i, k] = v
    return X

def tanimoto_kernel(A, B):
    """MinMax (generalized Tanimoto) kernel on count vectors."""
    K = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for i in range(A.shape[0]):
        mins = np.minimum(A[i], B).sum(axis=1)
        maxs = np.maximum(A[i], B).sum(axis=1)
        K[i] = np.where(maxs > 0, mins / maxs, 0.0)
    return K

def fit_predict_structure_blend(Xf_tr, Xd_tr, y_tr, Xf_va, Xd_va, seed=SEED):
    """Structure-only blend: Tanimoto KRR + (LightGBM | ExtraTrees)."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.kernel_ridge import KernelRidge
    try:
        from lightgbm import LGBMRegressor
        HAVE_LGBM = True
    except Exception:
        HAVE_LGBM = False
    K_tr = tanimoto_kernel(Xf_tr, Xf_tr)
    K_va = tanimoto_kernel(Xf_va, Xf_tr)
    best_pred, best_err = (None, np.inf)
    for alpha in (0.001, 0.01, 0.1):
        kr = KernelRidge(alpha=alpha, kernel='precomputed')
        kr.fit(K_tr, y_tr)
        inner = grouped_folds(range(len(y_tr)), 3, seed)
        errs = []
        for f in range(3):
            tr, va = (inner != f, inner == f)
            if va.sum() < 3:
                continue
            kr_i = KernelRidge(alpha=alpha, kernel='precomputed')
            kr_i.fit(K_tr[np.ix_(tr, tr)], y_tr[tr])
            errs.append(np.mean((kr_i.predict(K_tr[np.ix_(va, tr)]) - y_tr[va]) ** 2))
        e = float(np.mean(errs))
        if e < best_err:
            best_err = e
            best_pred = kr.predict(K_va)
    krr_pred = best_pred
    if HAVE_LGBM:
        gbm = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, min_child_samples=8, random_state=seed, verbosity=-1)
    else:
        gbm = ExtraTreesRegressor(600, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    gbm.fit(np.hstack([Xd_tr, Xf_tr]), y_tr)
    gbm_pred = gbm.predict(np.hstack([Xd_va, Xf_va]))
    return 0.5 * krr_pred + 0.5 * gbm_pred

def cross_fitted_partner_predictions(data, target_cans_all, seed=SEED):
    """Cross-fitted partner-property predictions (row's own structure excluded).

    Returns: dict prop -> (dict can -> prediction).
    """
    out = {}
    for prop in ELECTRONIC:
        rows = data.wide[data.wide[prop].notna()]
        cans = rows.index.tolist()
        y = rows[prop].values
        need = sorted(set(target_cans_all))
        folds = grouped_folds(cans, 5, seed)
        Xf = morgan_count_block(cans)
        Xd = descriptor_block(cans)
        Xf_need = morgan_count_block(need)
        Xd_need = descriptor_block(need)
        preds = np.zeros((len(need), 5))
        labeled = {c: i for i, c in enumerate(cans)}
        for f in range(5):
            tr = folds != f
            p = fit_predict_structure_blend(Xf[tr], Xd[tr], y[tr], Xf_need, Xd_need, seed)
            preds[:, f] = p
        res = {}
        for i, c in enumerate(need):
            if c in labeled:
                f_own = folds[labeled[c]]
                use = [preds[i, f] for f in range(5) if f != f_own]
            else:
                use = preds[i]
            res[c] = float(np.mean(use))
        out[prop] = res
    return out

def polar_block(cans):
    """Polar/dipolar descriptor block targeting the IONIC dielectric response."""
    rows = []
    for c in cans:
        m = Chem.MolFromSmiles(c)
        nh = max(m.GetNumHeavyAtoms(), 1)
        r = [len(m.GetSubstructMatches(p)) / nh for p in _PATS.values()]
        r += [Descriptors.TPSA(m) / nh, Descriptors.NumHDonors(m) / nh, Descriptors.NumHAcceptors(m) / nh, Descriptors.FractionCSP3(m), Descriptors.NumRotatableBonds(m) / nh, Crippen.MolMR(m) / nh, Crippen.MolLogP(m) / nh, rdMolDescriptors.CalcNumAromaticRings(m) / nh]
        rows.append(r)
    return np.asarray(rows, dtype=float)

def fit_ionic_model(cans_tr, ionic_tr, seed=SEED):
    from sklearn.ensemble import ExtraTreesRegressor
    X = polar_block(cans_tr)
    model = ExtraTreesRegressor(800, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    model.fit(X, np.asarray(ionic_tr, dtype=float))
    return model

def predict_ionic(model, cans):
    return model.predict(polar_block(cans))

fc = _types.SimpleNamespace(**{'ELECTRONIC': ELECTRONIC, 'IDENTITY': IDENTITY, 'IDENTITY_RESID': IDENTITY_RESID, 'MIN_IONIC': MIN_IONIC, 'PARTNERS': PARTNERS, 'POLAR_SMARTS': POLAR_SMARTS, 'Round2Data': Round2Data, 'SEED': SEED, 'TARGETS': TARGETS, '_PATS': _PATS, 'butina_clusters': butina_clusters, 'canon_nostereo': canon_nostereo, 'cluster_folds': cluster_folds, 'cross_fitted_partner_predictions': cross_fitted_partner_predictions, 'descriptor_block': descriptor_block, 'evaluate_target': evaluate_target, 'exact_lookup_table': exact_lookup_table, 'fit_ionic_model': fit_ionic_model, 'fit_predict_structure_blend': fit_predict_structure_blend, 'grouped_bootstrap_lower': grouped_bootstrap_lower, 'grouped_folds': grouped_folds, 'load_data': load_data, 'morgan_count_block': morgan_count_block, 'morgan_fp': morgan_fp, 'morgan_fp_list': morgan_fp_list, 'nn_similarity': nn_similarity, 'polar_block': polar_block, 'predict_ionic': predict_ionic, 'r2_score_manual': r2_score_manual, 'save_report': save_report, 'shift_matched_r2': shift_matched_r2, 'tanimoto_kernel': tanimoto_kernel})

# ---- feature builder support ----
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    cb = None
    CATBOOST_AVAILABLE = False
try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors
    MORDRED_AVAILABLE = True
except Exception:
    MordredCalculator = None
    mordred_descriptors = None
    MORDRED_AVAILABLE = False
try:
    from polymer_property_prediction import polymer_properties_from_smiles as bicerano_ppf
    BICERANO_AVAILABLE = True
except Exception:
    bicerano_ppf = None
    BICERANO_AVAILABLE = False
SMARTS_MOTIFS = (('ester', '[CX3](=O)[OX2H0][#6]'), ('carbonate', '[OX2][CX3](=[OX1])[OX2]'), ('amide', '[NX3][CX3](=[OX1])[#6]'), ('imide', '[NX3]([CX3]=[OX1])[CX3]=[OX1]'), ('cyclic_imide', '[NX3]1[CX3](=[OX1])[#6][CX3](=[OX1])1'), ('phthalimide_like', 'O=C1N([#6])C(=O)c2ccccc12'), ('naphthalimide_like', 'O=C1N([#6])C(=O)c2ccc3ccccc3c12'), ('maleimide_like', 'O=C1NC(=O)C=C1'), ('urethane', '[NX3][CX3](=[OX1])[OX2]'), ('urea', '[NX3][CX3](=[OX1])[NX3]'), ('aromatic_urea', 'c[NX3][CX3](=[OX1])[NX3]c'), ('ether', '[OD2]([#6])[#6]'), ('phenoxy', 'c[OX2][#6]'), ('sulfone', '[SX4](=[OX1])(=[OX1])([#6])[#6]'), ('sulfoxide', '[SX3](=[OX1])([#6])[#6]'), ('sulfonamide', '[SX4](=[OX1])(=[OX1])([NX3])[#6]'), ('nitrile', '[CX2]#N'), ('dicyano_methine', '[CX3]([CX2]#N)([CX2]#N)'), ('alkene', '[CX3]=[CX3]'), ('alkyne', '[CX2]#[CX2]'), ('carbonyl', '[CX3]=[OX1]'), ('benzophenone', 'c[CX3](=O)c'), ('anhydride', '[CX3](=[OX1])[OX2][CX3](=[OX1])'), ('aromatic_n', '[n,N;R]'), ('aromatic_o', '[o,O;R]'), ('aromatic_s', '[s,S;R]'), ('benzothiadiazole_like', 'c1nc2scnc2c1'), ('benzoxazole_like', 'c1nc2occc2c1'), ('carbazole_like', 'c1ccc2[nH,nX3]c3ccccc3c2c1'), ('fluoroaryl', 'c[F]'), ('chloroaryl', 'c[Cl]'), ('bromoaryl', 'c[Br]'), ('aryl_trifluoromethyl', 'cC(F)(F)F'), ('perfluoroalkyl', '[CX4](F)(F)[CX4](F)(F)'), ('siloxane', '[Si][OX2][Si]'), ('organosilicon', '[Si]([#6])([#6])'), ('organotin', '[Sn]'), ('phosphonate', '[PX4](=[OX1])([OX2])[OX2]'), ('phosphoric_acid', '[PX4](=[OX1])([OX2H])[OX2H]'), ('phosphine_oxide', '[PX4](=[OX1])([#6])([#6])'), ('quaternary_ammonium', '[NX4+]'), ('azo', '[NX2]=[NX2]'))
COMPILED_SMARTS = tuple(((name, Chem.MolFromSmarts(smarts)) for name, smarts in SMARTS_MOTIFS if Chem.MolFromSmarts(smarts) is not None))
REGION_DESCRIPTOR_NAMES = ['MolWt', 'ExactMolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA', 'HeavyAtomCount', 'FractionCSP3', 'RingCount', 'NumAromaticRings', 'NumAliphaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'BertzCT', 'BalabanJ', 'Kappa1', 'Kappa2', 'Kappa3']
REGION_ATOM_NUMBERS = (0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
LOWGAP_SMARTS = (('cyano', '[CX2]#N'), ('imide', '[NX3]([CX3](=O))[CX3](=O)'), ('sulfone', '[SX4](=O)(=O)'), ('sulfoxide', '[SX3](=O)'), ('quinone_like', '[#6]1(=O)[#6]=[#6][#6](=O)[#6]=[#6]1'), ('thiophene', '[s]1[c][c][c][c]1'), ('thiazole', '[s]1[c][n][c][c]1'), ('oxadiazole', '[o]1[n][c][n][c]1'), ('triazine', '[n]1[c][n][c][n][c]1'), ('acceptor_carbonyl_aromatic', '[c][CX3](=O)[#6,#7,#8]'), ('vinylene', '[CX3]=[CX3]'), ('ethynylene', '[CX2]#[CX2]'))
BICERANO_COLUMNS = ('density_at_298_k_gcm3', 'density_at_t_gcm3', 'molar_volume_cm3mol', 'ecoh1_at_298_k_jmol', 'solub_ratio', 'fh_parameter', 'tg_k', 'temperature_of_half_decomposition_k', 'bulk_modulus_mpa', 'youngs_modulus_mpa', 'shear_modulus_mpa', 'brittle_fracture_stress_mpa', 'tensile_yield_stress_mpa', 'number_hydrogen_bonding', 'charge_of_counterion', 'permeability_co2_barrer', 'permeability_n2_barrer', 'permeability_o2_barrer', 'selectivity_co2_n2', 'selectivity_o2_n2')
FFOX_DESCRIPTOR_NAMES = ['MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings', 'FractionCSP3', 'BertzCT', 'BalabanJ', 'Chi0v', 'Chi1v', 'Chi2v', 'Kappa1', 'Kappa2', 'Kappa3']

def mol_from_smiles(smiles: str, label: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        raise RuntimeError(f'RDKit parse failed for {label}')
    return mol

def build_mols(smiles: list[str]) -> list[Chem.Mol]:
    return [mol_from_smiles(value, f'row={index}') for index, value in enumerate(smiles)]

def descriptor_matrix(mols: list[Chem.Mol], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    descriptor_items = list(Descriptors._descList)
    names = [name for name, _ in descriptor_items]
    values = np.empty((len(mols), len(names)), dtype=np.float64)
    values.fill(np.nan)
    for row, mol in enumerate(mols):
        for col, (_, func) in enumerate(descriptor_items):
            try:
                value = float(func(mol))
            except Exception:
                value = math.nan
            values[row, col] = value if math.isfinite(value) else math.nan
    extra_names = ['smiles_len', 'star_count', 'atom_count', 'heavy_atom_count', 'dummy_atom_count', 'ring_count', 'aromatic_atom_count', 'hetero_atom_count', 'halogen_count', 'n_count', 'o_count', 's_count', 'si_count', 'f_count', 'cl_count', 'br_count', 'double_bond_count', 'triple_bond_count', 'branch_count', 'bracket_count']
    extra = np.zeros((len(mols), len(extra_names)), dtype=np.float64)
    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        extra[row, 0] = len(str(smi))
        extra[row, 1] = str(smi).count('*')
        extra[row, 2] = len(atoms)
        extra[row, 3] = sum((1 for atom in atoms if atom.GetAtomicNum() > 1))
        extra[row, 4] = sum((1 for atom in atoms if atom.GetAtomicNum() == 0))
        extra[row, 5] = mol.GetRingInfo().NumRings()
        extra[row, 6] = sum((1 for atom in atoms if atom.GetIsAromatic()))
        extra[row, 7] = sum((1 for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6)))
        extra[row, 8] = sum((1 for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53)))
        for col, atomic_num in ((9, 7), (10, 8), (11, 16), (12, 14), (13, 9), (14, 17), (15, 35)):
            extra[row, col] = sum((1 for atom in atoms if atom.GetAtomicNum() == atomic_num))
        extra[row, 16] = sum((1 for bond in bonds if str(bond.GetBondType()) == 'DOUBLE'))
        extra[row, 17] = sum((1 for bond in bonds if str(bond.GetBondType()) == 'TRIPLE'))
        extra[row, 18] = str(smi).count('(')
        extra[row, 19] = str(smi).count('[')
    estate_names = [f'estate_min_{index}' for index in range(79)] + [f'estate_max_{index}' for index in range(79)]
    estate = np.empty((len(mols), len(estate_names)), dtype=np.float64)
    estate.fill(np.nan)
    for row, mol in enumerate(mols):
        try:
            mins, maxs = EStateFingerprinter.FingerprintMol(mol)
            parsed = np.asarray(list(mins) + list(maxs), dtype=np.float64)
            if parsed.shape[0] == len(estate_names):
                estate[row] = parsed
        except Exception:
            pass
    return (np.hstack([values, extra, estate]), names + extra_names + estate_names)

def mordred_descriptor_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    if not MORDRED_AVAILABLE or MordredCalculator is None or mordred_descriptors is None:
        raise RuntimeError('mordred is not available in this environment')
    calculator = MordredCalculator(mordred_descriptors, ignore_3D=True)
    frame = calculator.pandas(mols, nproc=1, quiet=True)
    numeric = frame.apply(pd.to_numeric, errors='coerce')
    names = [f'mordred_{str(name)}' for name in numeric.columns]
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return (values, names)

def rdkit_3d_descriptor_matrix(smiles: list[str], *, seed: int, optimize_steps: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    descriptor_functions = [('rdkit3d_Asphericity', Descriptors3D.Asphericity), ('rdkit3d_Eccentricity', Descriptors3D.Eccentricity), ('rdkit3d_InertialShapeFactor', Descriptors3D.InertialShapeFactor), ('rdkit3d_NPR1', Descriptors3D.NPR1), ('rdkit3d_NPR2', Descriptors3D.NPR2), ('rdkit3d_PBF', Descriptors3D.PBF), ('rdkit3d_PMI1', Descriptors3D.PMI1), ('rdkit3d_PMI2', Descriptors3D.PMI2), ('rdkit3d_PMI3', Descriptors3D.PMI3), ('rdkit3d_RadiusOfGyration', Descriptors3D.RadiusOfGyration), ('rdkit3d_SpherocityIndex', Descriptors3D.SpherocityIndex)]
    names = [name for name, _ in descriptor_functions]
    values = np.empty((len(smiles), len(names)), dtype=np.float64)
    values.fill(np.nan)
    status_counts: dict[str, int] = {}
    for row, smi in enumerate(smiles):
        status = 'ok'
        try:
            capped = Chem.MolFromSmiles(cap_polymer_smiles(str(smi)), sanitize=True)
            if capped is None:
                raise RuntimeError('capped parse failed')
            mol = Chem.AddHs(capped)
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed + row)
            params.useRandomCoords = True
            embed_code = int(AllChem.EmbedMolecule(mol, params))
            if embed_code != 0:
                status = f'embed_code_{embed_code}'
                raise RuntimeError(status)
            if optimize_steps > 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=int(optimize_steps))
                    pass
                except Exception:
                    status = 'uff_failed_descriptor_attempted'
            for col, (_, function) in enumerate(descriptor_functions):
                try:
                    value = float(function(mol))
                except Exception:
                    value = math.nan
                values[row, col] = value if math.isfinite(value) else math.nan
        except Exception:
            pass
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {'source': 'official train/test SMILES only; dummy endpoints capped with carbon before deterministic RDKit ETKDG', 'seed': int(seed), 'optimize_steps': int(optimize_steps), 'descriptor_count': len(names), 'status_counts': status_counts, 'nonfinite_by_column': {name: int(np.count_nonzero(~np.isfinite(values[:, index]))) for index, name in enumerate(names)}}
    return (values, names, report)

def parse_int_csv(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        parsed = value
    elif isinstance(value, list):
        parsed = tuple((int(item) for item in value))
    else:
        parsed = tuple((int(part.strip()) for part in str(value).split(',') if part.strip()))
    if not parsed:
        raise ValueError('expected at least one integer')
    return parsed

def parse_token_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        parsed = value
    elif isinstance(value, list):
        parsed = tuple((str(item).strip() for item in value if str(item).strip()))
    else:
        parsed = tuple((part.strip().lower() for part in str(value).split(',') if part.strip()))
    if not parsed:
        raise ValueError('expected at least one token')
    return parsed

def rdkit_3d_descriptor_names(include_extended: bool) -> list[str]:
    names = ['rdkit3d_Asphericity', 'rdkit3d_Eccentricity', 'rdkit3d_InertialShapeFactor', 'rdkit3d_NPR1', 'rdkit3d_NPR2', 'rdkit3d_PBF', 'rdkit3d_PMI1', 'rdkit3d_PMI2', 'rdkit3d_PMI3', 'rdkit3d_RadiusOfGyration', 'rdkit3d_SpherocityIndex']
    if include_extended:
        names.extend([f'rdkit3d_WHIM_{index:03d}' for index in range(114)])
        names.extend([f'rdkit3d_GETAWAY_{index:03d}' for index in range(273)])
        names.extend([f'rdkit3d_MORSE_{index:03d}' for index in range(224)])
        names.extend([f'rdkit3d_RDF_{index:03d}' for index in range(210)])
        names.extend([f'rdkit3d_AUTOCORR3D_{index:03d}' for index in range(80)])
        pass
    return names

def rdkit_3d_values_for_conformer(mol: Chem.Mol, conf_id: int, *, include_extended: bool) -> np.ndarray:
    values: list[float] = []
    scalar_functions = [Descriptors3D.Asphericity, Descriptors3D.Eccentricity, Descriptors3D.InertialShapeFactor, Descriptors3D.NPR1, Descriptors3D.NPR2, Descriptors3D.PBF, Descriptors3D.PMI1, Descriptors3D.PMI2, Descriptors3D.PMI3, Descriptors3D.RadiusOfGyration, Descriptors3D.SpherocityIndex]
    for function in scalar_functions:
        try:
            value = float(function(mol, confId=int(conf_id)))
        except Exception:
            value = math.nan
        values.append(value if math.isfinite(value) else math.nan)
    if include_extended:
        vector_functions = [rdMolDescriptors.CalcWHIM, rdMolDescriptors.CalcGETAWAY, rdMolDescriptors.CalcMORSE, rdMolDescriptors.CalcRDF, rdMolDescriptors.CalcAUTOCORR3D]
        for function in vector_functions:
            try:
                vector = [float(item) for item in function(mol, confId=int(conf_id))]
            except Exception:
                vector = []
            values.extend([item if math.isfinite(item) else math.nan for item in vector])
    expected = len(rdkit_3d_descriptor_names(include_extended))
    if len(values) < expected:
        values.extend([math.nan] * (expected - len(values)))
        pass
    elif len(values) > expected:
        values = values[:expected]
    return np.asarray(values, dtype=np.float64)

def pool_conformer_descriptors(matrix: np.ndarray, poolings: tuple[str, ...]) -> np.ndarray:
    finite = np.isfinite(matrix)
    count = finite.sum(axis=0).astype(np.float64)
    pieces: list[np.ndarray] = []
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.divide(np.where(finite, matrix, 0.0).sum(axis=0), count, out=np.full(matrix.shape[1], np.nan, dtype=np.float64), where=count > 0)
    for pooling in poolings:
        if pooling == 'mean':
            pooled = mean
        elif pooling == 'std':
            diff = np.where(finite, matrix - mean[None, :], 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                pooled = np.sqrt(np.divide(np.square(diff).sum(axis=0), count, out=np.full(matrix.shape[1], np.nan, dtype=np.float64), where=count > 0))
        elif pooling == 'min':
            pooled = np.where(count > 0, np.min(np.where(finite, matrix, np.inf), axis=0), np.nan)
        elif pooling == 'max':
            pooled = np.where(count > 0, np.max(np.where(finite, matrix, -np.inf), axis=0), np.nan)
        else:
            raise ValueError(f'unknown conformer pooling {pooling!r}')
        pieces.append(np.asarray(pooled, dtype=np.float64))
    return np.concatenate(pieces)

def pooled_3d_descriptor_for_mol(mol: Chem.Mol, *, seed: int, conformers: int, optimize_steps: int, poolings: tuple[str, ...], include_extended: bool) -> tuple[np.ndarray, dict[str, Any]]:
    descriptor_names = rdkit_3d_descriptor_names(include_extended)
    out = np.full(len(descriptor_names) * len(poolings), np.nan, dtype=np.float64)
    report: dict[str, Any] = {'status': 'ok', 'requested_conformers': int(conformers), 'embedded_conformers': 0, 'descriptor_conformers': 0, 'uff_failures': 0}
    try:
        working = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(working, numConfs=int(conformers), params=params))
        report['embedded_conformers'] = int(len(conf_ids))
        if not conf_ids:
            report['status'] = 'embed_failed'
            return (out, report)
        rows: list[np.ndarray] = []
        for conf_id in conf_ids:
            if optimize_steps > 0:
                try:
                    AllChem.UFFOptimizeMolecule(working, confId=int(conf_id), maxIters=int(optimize_steps))
                    pass
                except Exception:
                    report['uff_failures'] = int(report['uff_failures']) + 1
            row = rdkit_3d_values_for_conformer(working, int(conf_id), include_extended=include_extended)
            if np.isfinite(row).any():
                rows.append(row)
                pass
        report['descriptor_conformers'] = int(len(rows))
        if not rows:
            report['status'] = 'no_descriptor_conformer'
            return (out, report)
        out = pool_conformer_descriptors(np.vstack(rows), poolings)
        if not np.isfinite(out).any():
            report['status'] = 'all_nonfinite'
        return (out, report)
    except Exception as exc:
        report['status'] = f'failed_{type(exc).__name__}'
        return (out, report)

def physics_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    names = ['gasteiger_min', 'gasteiger_max', 'gasteiger_mean', 'gasteiger_std', 'gasteiger_abs_mean', 'gasteiger_abs_max', 'formal_charge_sum', 'formal_charge_abs_sum', 'radical_electron_sum', 'avg_total_valence', 'avg_total_degree', 'sp_atom_fraction', 'sp2_atom_fraction', 'sp3_atom_fraction', 'conjugated_bond_fraction', 'aromatic_bond_fraction', 'single_bond_fraction', 'double_bond_fraction', 'triple_bond_fraction', 'endpoint_neighbor_atomic_min', 'endpoint_neighbor_atomic_max', 'endpoint_neighbor_aromatic_sum', 'endpoint_neighbor_ring_sum', 'endpoint_neighbor_degree_sum', 'endpoint_path_length', 'endpoint_direct_bond_present']
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    for row, mol in enumerate(mols):
        local = Chem.Mol(mol)
        atoms = list(local.GetAtoms())
        bonds = list(local.GetBonds())
        n_atoms = max(len(atoms), 1)
        n_bonds = max(len(bonds), 1)
        try:
            AllChem.ComputeGasteigerCharges(local)
            charges = []
            for atom in local.GetAtoms():
                charge = float(atom.GetProp('_GasteigerCharge'))
                charges.append(charge if math.isfinite(charge) else 0.0)
            charge_arr = np.asarray(charges, dtype=np.float64)
        except Exception:
            charge_arr = np.zeros(len(atoms), dtype=np.float64)
        if charge_arr.size:
            values[row, 0] = float(np.min(charge_arr))
            values[row, 1] = float(np.max(charge_arr))
            values[row, 2] = float(np.mean(charge_arr))
            values[row, 3] = float(np.std(charge_arr))
            values[row, 4] = float(np.mean(np.abs(charge_arr)))
            values[row, 5] = float(np.max(np.abs(charge_arr)))
        values[row, 6] = float(sum((atom.GetFormalCharge() for atom in atoms)))
        values[row, 7] = float(sum((abs(atom.GetFormalCharge()) for atom in atoms)))
        values[row, 8] = float(sum((atom.GetNumRadicalElectrons() for atom in atoms)))
        values[row, 9] = float(sum((atom.GetTotalValence() for atom in atoms))) / n_atoms
        values[row, 10] = float(sum((atom.GetTotalDegree() for atom in atoms))) / n_atoms
        values[row, 11] = float(sum((atom.GetHybridization() == Chem.HybridizationType.SP for atom in atoms))) / n_atoms
        values[row, 12] = float(sum((atom.GetHybridization() == Chem.HybridizationType.SP2 for atom in atoms))) / n_atoms
        values[row, 13] = float(sum((atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in atoms))) / n_atoms
        values[row, 14] = float(sum((bond.GetIsConjugated() for bond in bonds))) / n_bonds
        values[row, 15] = float(sum((bond.GetBondType() == Chem.BondType.AROMATIC for bond in bonds))) / n_bonds
        values[row, 16] = float(sum((bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds))) / n_bonds
        values[row, 17] = float(sum((bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds))) / n_bonds
        values[row, 18] = float(sum((bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds))) / n_bonds
        endpoint_neighbors = []
        endpoint_bond_types = []
        for atom in atoms:
            if atom.GetAtomicNum() != 0:
                continue
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                continue
            neighbor = neighbors[0]
            bond = local.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            endpoint_neighbors.append(neighbor)
            if bond is not None:
                endpoint_bond_types.append(bond.GetBondType())
                pass
        if len(endpoint_neighbors) == 2:
            atomic_nums = sorted((neighbor.GetAtomicNum() for neighbor in endpoint_neighbors))
            values[row, 19] = float(atomic_nums[0])
            values[row, 20] = float(atomic_nums[1])
            values[row, 21] = float(sum((neighbor.GetIsAromatic() for neighbor in endpoint_neighbors)))
            values[row, 22] = float(sum((neighbor.IsInRing() for neighbor in endpoint_neighbors)))
            values[row, 23] = float(sum((neighbor.GetTotalDegree() for neighbor in endpoint_neighbors)))
            try:
                path = Chem.rdmolops.GetShortestPath(local, endpoint_neighbors[0].GetIdx(), endpoint_neighbors[1].GetIdx())
                values[row, 24] = float(max(0, len(path) - 1))
            except Exception:
                values[row, 24] = 0.0
            values[row, 25] = float(local.GetBondBetweenAtoms(endpoint_neighbors[0].GetIdx(), endpoint_neighbors[1].GetIdx()) is not None)
    return (values, names)

def electronic_tail_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    compiled = [(name, Chem.MolFromSmarts(pattern)) for name, pattern in LOWGAP_SMARTS]
    compiled = [(name, pattern) for name, pattern in compiled if pattern is not None]
    base_names = ['electronic_lowgap_total_count', 'electronic_acceptor_total_count', 'electronic_acceptor_per_heavy', 'electronic_path_acceptor_count', 'electronic_path_acceptor_fraction', 'electronic_path_da_alternations', 'electronic_path_longest_acceptor_run', 'electronic_path_longest_donor_run', 'electronic_path_conj_acceptor_fraction', 'electronic_endpoint_acceptor_asymmetry']
    smarts_names: list[str] = []
    for name, _ in compiled:
        smarts_names.extend([f'electronic_smarts_{name}_count', f'electronic_smarts_{name}_per_heavy'])
        pass
    names = base_names + smarts_names
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        atoms = list(mol.GetAtoms())
        heavy = max(sum((atom.GetAtomicNum() > 1 for atom in atoms)), 1)
        atom_acceptor_score = np.zeros(len(atoms), dtype=np.float64)
        atom_donor_score = np.zeros(len(atoms), dtype=np.float64)
        atom_conj_score = np.zeros(len(atoms), dtype=np.float64)
        for atom in atoms:
            idx = atom.GetIdx()
            anum = atom.GetAtomicNum()
            is_acceptor = anum in (7, 8, 16) and atom.GetFormalCharge() <= 0
            is_acceptor = is_acceptor or anum in (9, 17, 35, 53)
            is_acceptor = is_acceptor or (anum == 6 and any((bond.GetBondType() == Chem.BondType.TRIPLE for bond in atom.GetBonds())))
            atom_acceptor_score[idx] = float(is_acceptor)
            atom_donor_score[idx] = float(anum in (7, 8, 16) and atom.GetTotalNumHs() > 0)
            atom_conj_score[idx] = float(atom.GetIsAromatic() or any((bond.GetIsConjugated() for bond in atom.GetBonds())))
        offset = len(base_names)
        total_count = 0.0
        for name, pattern in compiled:
            try:
                matches = mol.GetSubstructMatches(pattern, uniquify=True)
                count = float(len(matches))
            except Exception:
                matches = ()
                count = 0.0
            total_count += count
            if matches:
                for match in matches:
                    for atom_idx in match:
                        if 0 <= int(atom_idx) < len(atom_acceptor_score):
                            atom_acceptor_score[int(atom_idx)] = 1.0
            values[row, offset] = count
            values[row, offset + 1] = count / heavy
            offset += 2
        values[row, 0] = total_count
        values[row, 1] = float(np.sum(atom_acceptor_score))
        values[row, 2] = float(np.sum(atom_acceptor_score)) / heavy
        _, endpoint_path = endpoint_neighbors_and_path(mol)
        if endpoint_path:
            path = [int(idx) for idx in endpoint_path]
            path_acceptor = atom_acceptor_score[path]
            path_donor = atom_donor_score[path]
            path_conj = atom_conj_score[path]
            path_len = max(len(path), 1)
            values[row, 3] = float(np.sum(path_acceptor))
            values[row, 4] = float(np.mean(path_acceptor))
            tokens = np.where(path_acceptor > 0, 1, np.where(path_donor > 0, -1, 0))
            alternations = 0
            last = 0
            for token in tokens:
                current = int(token)
                if current == 0:
                    continue
                if last != 0 and current != last:
                    alternations += 1
                last = current
            values[row, 5] = float(alternations)
            for token_value, col in ((1, 6), (-1, 7)):
                best_run = 0
                current_run = 0
                for token in tokens:
                    if int(token) == token_value:
                        current_run += 1
                        best_run = max(best_run, current_run)
                    else:
                        current_run = 0
                values[row, col] = float(best_run)
            values[row, 8] = float(np.sum((path_acceptor > 0) & (path_conj > 0))) / path_len
            left = float(np.sum(path_acceptor[:min(3, len(path_acceptor))]))
            right = float(np.sum(path_acceptor[max(0, len(path_acceptor) - 3):]))
            values[row, 9] = abs(left - right)
        status_counts['ok'] = status_counts.get('ok', 0) + 1
    report = {'source': 'official train/test SMILES only; explicit low-bandgap acceptor motifs and ordered donor/acceptor endpoint-path signatures', 'smarts_patterns': [name for name, _ in compiled], 'dense_feature_count': len(names), 'status_counts': status_counts}
    return (values, names, report)

def huckel_spectrum_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Topology-only pi-system spectral descriptors for optical/electronic gap signal."""
    names = ['huckel_pi_atom_count', 'huckel_pi_atom_fraction', 'huckel_pi_edge_count', 'huckel_pi_edge_fraction', 'huckel_component_count', 'huckel_largest_component_fraction', 'huckel_gap_mid', 'huckel_gap_mid_norm', 'huckel_periodic_gap_mid', 'huckel_periodic_gap_mid_norm', 'huckel_gap_delta_periodic', 'huckel_spectral_radius', 'huckel_periodic_spectral_radius', 'huckel_bandwidth', 'huckel_periodic_bandwidth', 'huckel_center_level_density', 'huckel_periodic_center_level_density', 'huckel_lowest_abs_eigen', 'huckel_periodic_lowest_abs_eigen', 'huckel_hetero_pi_fraction', 'huckel_halogen_neighbor_pi_fraction', 'huckel_carbonyl_neighbor_pi_fraction', 'huckel_endpoint_pi_fraction', 'huckel_endpoint_periodic_edge', 'huckel_endpoint_path_pi_fraction', 'huckel_endpoint_path_pi_run', 'huckel_endpoint_path_bond_order_mean', 'huckel_endpoint_path_bond_order_std', 'huckel_pi_avg_degree', 'huckel_pi_max_degree', 'huckel_pi_branch_fraction', 'huckel_electronegativity_mean', 'huckel_electronegativity_std', 'huckel_diag_perturb_mean', 'huckel_diag_perturb_std']
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    electronegativity = {5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.9, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
    carbonyl = Chem.MolFromSmarts('[CX3]=[OX1]')

    def is_pi_atom(atom: Chem.Atom) -> bool:
        if atom.GetAtomicNum() <= 1:
            return False
        if atom.GetIsAromatic():
            return True
        if atom.GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2):
            return True
        return any((bond.GetIsConjugated() or bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC) for bond in atom.GetBonds()))

    def bond_weight(bond: Chem.Bond) -> float:
        btype = bond.GetBondType()
        if btype == Chem.BondType.AROMATIC:
            return 1.35
        if btype == Chem.BondType.DOUBLE:
            return 1.6
        if btype == Chem.BondType.TRIPLE:
            return 1.9
        if bond.GetIsConjugated():
            return 1.0
        return 0.8

    def spectrum_stats(matrix: np.ndarray) -> dict[str, float]:
        if matrix.shape[0] < 2:
            return {'gap': 0.0, 'gap_norm': 0.0, 'radius': 0.0, 'bandwidth': 0.0, 'center_density': 0.0, 'lowest_abs': 0.0}
        try:
            eig = np.linalg.eigvalsh(matrix)
        except np.linalg.LinAlgError:
            eig = np.linalg.eigvals(matrix).real
            eig.sort()
        n = len(eig)
        mid = n // 2
        if n % 2 == 0:
            gap = float(eig[mid] - eig[mid - 1])
        else:
            gap = float(min(eig[mid] - eig[mid - 1], eig[mid + 1] - eig[mid]) if 0 < mid < n - 1 else 0.0)
        bandwidth = float(eig[-1] - eig[0]) if n else 0.0
        window = max(0.5, 0.1 * bandwidth)
        return {'gap': abs(gap), 'gap_norm': safe_ratio(abs(gap), bandwidth), 'radius': float(np.max(np.abs(eig))) if n else 0.0, 'bandwidth': bandwidth, 'center_density': safe_ratio(float(np.count_nonzero(np.abs(eig) <= window)), float(n)), 'lowest_abs': float(np.min(np.abs(eig))) if n else 0.0}
    for row, mol in enumerate(mols):
        status = 'ok'
        try:
            atoms = list(mol.GetAtoms())
            heavy = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() > 1}
            heavy_count = max(len(heavy), 1)
            pi_atoms = sorted((int(atom.GetIdx()) for atom in atoms if is_pi_atom(atom)))
            pi_set = set(pi_atoms)
            local_index = {atom_idx: pos for pos, atom_idx in enumerate(pi_atoms)}
            n_pi = len(pi_atoms)
            adjacency = np.zeros((n_pi, n_pi), dtype=np.float64)
            pi_edge_count = 0
            pi_degrees = np.zeros(n_pi, dtype=np.float64)
            pi_bond_weights: list[float] = []
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtomIdx())
                b = int(bond.GetEndAtomIdx())
                if a not in pi_set or b not in pi_set:
                    continue
                weight = bond_weight(bond)
                ia = local_index[a]
                ib = local_index[b]
                adjacency[ia, ib] = weight
                adjacency[ib, ia] = weight
                pi_degrees[ia] += 1.0
                pi_degrees[ib] += 1.0
                pi_edge_count += 1
                pi_bond_weights.append(weight)
            diag = np.zeros(n_pi, dtype=np.float64)
            en_values = []
            for atom_idx, pos in local_index.items():
                atom = mol.GetAtomWithIdx(atom_idx)
                en = electronegativity.get(atom.GetAtomicNum(), 2.55)
                en_values.append(en)
                diag[pos] = 0.35 * (en - 2.55)
            huckel = adjacency + np.diag(diag)
            endpoint_neighbors, endpoint_path = endpoint_neighbors_and_path(mol)
            periodic = np.array(huckel, copy=True)
            endpoint_periodic_edge = 0.0
            if len(endpoint_neighbors) == 2 and endpoint_neighbors[0] in pi_set and (endpoint_neighbors[1] in pi_set):
                left = local_index[int(endpoint_neighbors[0])]
                right = local_index[int(endpoint_neighbors[1])]
                if left != right:
                    periodic[left, right] = max(periodic[left, right], 1.0)
                    periodic[right, left] = max(periodic[right, left], 1.0)
                    endpoint_periodic_edge = 1.0
            component_sizes: list[int] = []
            seen: set[int] = set()
            graph_adj: dict[int, set[int]] = {idx: set() for idx in pi_atoms}
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtomIdx())
                b = int(bond.GetEndAtomIdx())
                if a in pi_set and b in pi_set:
                    graph_adj[a].add(b)
                    graph_adj[b].add(a)
                    pass
            for start in pi_atoms:
                if start in seen:
                    continue
                stack = [start]
                seen.add(start)
                size = 0
                while stack:
                    cur = stack.pop()
                    size += 1
                    for nxt in graph_adj.get(cur, ()):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                            pass
                component_sizes.append(size)
            stats = spectrum_stats(huckel)
            periodic_stats = spectrum_stats(periodic)
            hetero_pi = [idx for idx in pi_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() not in (6,)]
            halogen_neighbor_pi = [idx for idx in pi_atoms if any((neighbor.GetAtomicNum() in (9, 17, 35, 53) for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()))]
            carbonyl_atoms = {int(idx) for match in (mol.GetSubstructMatches(carbonyl) if carbonyl is not None else ()) for idx in match}
            carbonyl_neighbor_pi = [idx for idx in pi_atoms if idx in carbonyl_atoms or any((int(neighbor.GetIdx()) in carbonyl_atoms for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()))]
            endpoint_pi = [idx for idx in endpoint_neighbors if idx in pi_set]
            path_atoms = [int(idx) for idx in endpoint_path if int(idx) in heavy] if endpoint_path else []
            path_pi_flags = [int(idx in pi_set) for idx in path_atoms]
            best_run = 0
            current_run = 0
            for flag in path_pi_flags:
                if flag:
                    current_run += 1
                    best_run = max(best_run, current_run)
                else:
                    current_run = 0
            path_weights = []
            if endpoint_path:
                for a, b in zip(endpoint_path[:-1], endpoint_path[1:], strict=True):
                    bond = mol.GetBondBetweenAtoms(int(a), int(b))
                    if bond is not None:
                        path_weights.append(bond_weight(bond))
                        pass
            en_arr = np.asarray(en_values, dtype=np.float64)
            values[row] = np.asarray([n_pi, safe_ratio(n_pi, heavy_count), pi_edge_count, safe_ratio(pi_edge_count, max(len(mol.GetBonds()), 1)), len(component_sizes), safe_ratio(max(component_sizes) if component_sizes else 0, max(n_pi, 1)), stats['gap'], stats['gap_norm'], periodic_stats['gap'], periodic_stats['gap_norm'], periodic_stats['gap'] - stats['gap'], stats['radius'], periodic_stats['radius'], stats['bandwidth'], periodic_stats['bandwidth'], stats['center_density'], periodic_stats['center_density'], stats['lowest_abs'], periodic_stats['lowest_abs'], safe_ratio(len(hetero_pi), n_pi), safe_ratio(len(halogen_neighbor_pi), n_pi), safe_ratio(len(carbonyl_neighbor_pi), n_pi), safe_ratio(len(endpoint_pi), max(len(endpoint_neighbors), 1)), endpoint_periodic_edge, safe_ratio(sum(path_pi_flags), max(len(path_pi_flags), 1)), best_run, float(np.mean(path_weights)) if path_weights else 0.0, float(np.std(path_weights)) if path_weights else 0.0, float(np.mean(pi_degrees)) if len(pi_degrees) else 0.0, float(np.max(pi_degrees)) if len(pi_degrees) else 0.0, safe_ratio(float(np.count_nonzero(pi_degrees > 2.0)), n_pi), float(np.mean(en_arr)) if len(en_arr) else 0.0, float(np.std(en_arr)) if len(en_arr) else 0.0, float(np.mean(diag)) if len(diag) else 0.0, float(np.std(diag)) if len(diag) else 0.0], dtype=np.float64)
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
            values[row] = np.nan
        status_counts[status] = status_counts.get(status, 0) + 1
    return (values, names, {'source': 'official train/test SMILES only; Huckel-style weighted pi-graph spectrum and periodic endpoint closure proxies', 'descriptor_count': len(names), 'status_counts': status_counts, 'nonfinite_values': int(np.count_nonzero(~np.isfinite(values))), 'external_data_training_use': False, 'pretrained_model_use': False})

def topological_autocorr_feature_matrix(mols: list[Chem.Mol], max_distance: int=8) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    properties = ('charge', 'atomic_num', 'hetero', 'aromatic', 'acceptor', 'donor')
    names: list[str] = []
    for distance in range(1, int(max_distance) + 1):
        for prop in properties:
            names.extend([f'topo_autocorr_d{distance}_{prop}_prod_mean', f'topo_autocorr_d{distance}_{prop}_absdiff_mean'])
            pass
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        local = Chem.Mol(mol)
        atoms = list(local.GetAtoms())
        if not atoms:
            status_counts['empty'] = status_counts.get('empty', 0) + 1
            continue
        try:
            AllChem.ComputeGasteigerCharges(local)
            pass
        except Exception:
            pass
        prop_values: dict[str, np.ndarray] = {}
        charges = []
        for atom in local.GetAtoms():
            try:
                charge = float(atom.GetProp('_GasteigerCharge'))
            except Exception:
                charge = 0.0
            charges.append(charge if math.isfinite(charge) else 0.0)
        prop_values['charge'] = np.asarray(charges, dtype=np.float64)
        prop_values['atomic_num'] = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=np.float64)
        prop_values['hetero'] = np.asarray([atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms], dtype=np.float64)
        prop_values['aromatic'] = np.asarray([atom.GetIsAromatic() for atom in atoms], dtype=np.float64)
        prop_values['acceptor'] = np.asarray([atom.GetAtomicNum() in (7, 8, 16) and atom.GetFormalCharge() <= 0 or atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms], dtype=np.float64)
        prop_values['donor'] = np.asarray([atom.GetAtomicNum() in (7, 8, 16) and atom.GetTotalNumHs() > 0 for atom in atoms], dtype=np.float64)
        try:
            distances = Chem.rdmolops.GetDistanceMatrix(local).astype(np.int64)
        except Exception:
            status_counts['distance_failed'] = status_counts.get('distance_failed', 0) + 1
            continue
        col = 0
        for distance in range(1, int(max_distance) + 1):
            pairs = np.argwhere(np.triu(distances == distance, k=1))
            for prop in properties:
                arr = prop_values[prop]
                if pairs.size:
                    left = arr[pairs[:, 0]]
                    right = arr[pairs[:, 1]]
                    values[row, col] = float(np.mean(left * right))
                    values[row, col + 1] = float(np.mean(np.abs(left - right)))
                col += 2
        status_counts['ok'] = status_counts.get('ok', 0) + 1
    report = {'source': 'official train/test SMILES only; graph-distance autocorrelations over charges and electronic atom flags', 'max_distance': int(max_distance), 'dense_feature_count': len(names), 'status_counts': status_counts}
    return (values, names, report)

def infinite_chain_proxy_feature_matrix(smiles: list[str], mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = ['icd_heavy_atoms', 'icd_exact_mass', 'icd_bond_count', 'icd_endpoint_path_bonds', 'icd_endpoint_path_heavy_atoms', 'icd_side_heavy_atoms', 'icd_side_heavy_fraction', 'icd_mass_per_path_bond', 'icd_heavy_atoms_per_path_bond', 'icd_side_heavy_per_path_bond', 'icd_side_mass_per_path_bond', 'icd_side_to_backbone_mass_ratio', 'icd_backbone_aromatic_fraction', 'icd_backbone_conjugated_bond_fraction', 'icd_backbone_rotatable_fraction', 'icd_backbone_sp2sp_fraction', 'icd_backbone_hetero_fraction', 'icd_backbone_halogen_fraction', 'icd_side_aromatic_fraction', 'icd_side_hetero_fraction', 'icd_side_halogen_fraction', 'icd_aromatic_atoms_per_path_bond', 'icd_conjugated_bonds_per_path_bond', 'icd_ring_count_per_path_bond', 'icd_fused_ring_pairs_per_path_bond', 'icd_acceptor_atoms_per_path_bond', 'icd_donor_atoms_per_path_bond', 'icd_lowgap_motifs_per_path_bond', 'icd_polar_atoms_per_exact_mass', 'icd_halogen_atoms_per_exact_mass', 'icd_aromatic_atoms_per_exact_mass', 'icd_conjugated_bonds_per_exact_mass', 'icd_periodic_heavy_atoms', 'icd_periodic_ring_count', 'icd_periodic_ring_count_delta', 'icd_periodic_aromatic_fraction', 'icd_periodic_conjugated_bond_fraction', 'icd_periodic_rotatable_fraction', 'icd_periodic_mean_topological_distance', 'icd_periodic_max_topological_distance']
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    lowgap_patterns = [pattern for _, pattern in ((name, Chem.MolFromSmarts(smarts)) for name, smarts in LOWGAP_SMARTS) if pattern is not None]
    status_counts: dict[str, int] = {}
    for row, (smi, mol) in enumerate(zip(smiles, mols, strict=True)):
        status = 'ok'
        try:
            atoms = list(mol.GetAtoms())
            bonds = list(mol.GetBonds())
            heavy_indices = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() > 1}
            heavy_count = max(len(heavy_indices), 1)
            exact_mass = float(sum((atom.GetMass() for atom in atoms if atom.GetAtomicNum() > 1)))
            mass_denom = max(exact_mass, 1e-09)
            bond_count = max(len(bonds), 1)
            _, endpoint_path = endpoint_neighbors_and_path(mol)
            path_set = set((int(idx) for idx in endpoint_path))
            path_heavy = {idx for idx in path_set if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() > 1}
            side_heavy = heavy_indices.difference(path_set)
            path_bonds: list[Chem.Bond] = []
            for left, right in zip(endpoint_path[:-1], endpoint_path[1:], strict=True):
                bond = mol.GetBondBetweenAtoms(int(left), int(right))
                if bond is not None:
                    path_bonds.append(bond)
                    pass
            path_bond_denom = max(len(path_bonds), 1)
            backbone_mass = float(sum((mol.GetAtomWithIdx(int(idx)).GetMass() for idx in path_heavy)))
            side_mass = float(sum((mol.GetAtomWithIdx(int(idx)).GetMass() for idx in side_heavy)))
            aromatic_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetIsAromatic()}
            hetero_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6)}
            halogen_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53)}
            acceptor_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() in (7, 8, 16) and atom.GetFormalCharge() <= 0 or atom.GetAtomicNum() in (9, 17, 35, 53)}
            donor_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() in (7, 8, 16) and atom.GetTotalNumHs() > 0}
            lowgap_count = 0
            for pattern in lowgap_patterns:
                try:
                    lowgap_count += len(mol.GetSubstructMatches(pattern, uniquify=True))
                except Exception:
                    pass
            rotatable_indices = region_rotatable_bond_indices(mol)
            sp2sp_bonds = [bond for bond in path_bonds if bond.GetBeginAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2) and bond.GetEndAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)]
            ring_info = mol.GetRingInfo()
            atom_rings = [set(ring) for ring in ring_info.AtomRings()]
            fused_pairs = 0
            for i, ring_i in enumerate(atom_rings):
                for ring_j in atom_rings[i + 1:]:
                    fused_pairs += int(bool(ring_i.intersection(ring_j)))
            periodic = periodic_closure_mol(str(smi), mol)
            periodic_atoms = [atom for atom in periodic.GetAtoms() if atom.GetAtomicNum() > 1]
            periodic_bonds = list(periodic.GetBonds())
            periodic_bond_denom = max(len(periodic_bonds), 1)
            periodic_heavy_denom = max(len(periodic_atoms), 1)
            try:
                distances = Chem.GetDistanceMatrix(periodic)
                tri = distances[np.triu_indices_from(distances, k=1)]
                tri = tri[np.isfinite(tri) & (tri > 0)]
                mean_distance = float(np.mean(tri)) if tri.size else 0.0
                max_distance = float(np.max(tri)) if tri.size else 0.0
            except Exception:
                mean_distance = 0.0
                max_distance = 0.0
            values[row] = np.asarray([float(heavy_count), exact_mass, float(bond_count), float(len(path_bonds)), float(len(path_heavy)), float(len(side_heavy)), safe_ratio(len(side_heavy), heavy_count), exact_mass / path_bond_denom, heavy_count / path_bond_denom, len(side_heavy) / path_bond_denom, side_mass / path_bond_denom, safe_ratio(side_mass, backbone_mass), safe_ratio(sum((idx in aromatic_atoms for idx in path_heavy)), len(path_heavy)), safe_ratio(sum((bond.GetIsConjugated() for bond in path_bonds)), len(path_bonds)), safe_ratio(sum((int(bond.GetIdx()) in rotatable_indices for bond in path_bonds)), len(path_bonds)), safe_ratio(len(sp2sp_bonds), len(path_bonds)), safe_ratio(sum((idx in hetero_atoms for idx in path_heavy)), len(path_heavy)), safe_ratio(sum((idx in halogen_atoms for idx in path_heavy)), len(path_heavy)), safe_ratio(sum((idx in aromatic_atoms for idx in side_heavy)), len(side_heavy)), safe_ratio(sum((idx in hetero_atoms for idx in side_heavy)), len(side_heavy)), safe_ratio(sum((idx in halogen_atoms for idx in side_heavy)), len(side_heavy)), len(aromatic_atoms) / path_bond_denom, sum((bond.GetIsConjugated() for bond in bonds)) / path_bond_denom, ring_info.NumRings() / path_bond_denom, fused_pairs / path_bond_denom, sum((idx in acceptor_atoms for idx in heavy_indices)) / path_bond_denom, sum((idx in donor_atoms for idx in heavy_indices)) / path_bond_denom, lowgap_count / path_bond_denom, len(hetero_atoms) / mass_denom, len(halogen_atoms) / mass_denom, len(aromatic_atoms) / mass_denom, sum((bond.GetIsConjugated() for bond in bonds)) / mass_denom, float(len(periodic_atoms)), float(periodic.GetRingInfo().NumRings()), float(periodic.GetRingInfo().NumRings() - ring_info.NumRings()), safe_ratio(sum((atom.GetIsAromatic() for atom in periodic_atoms)), periodic_heavy_denom), safe_ratio(sum((bond.GetIsConjugated() for bond in periodic_bonds)), periodic_bond_denom), safe_ratio(sum((int(bond.GetIdx()) in region_rotatable_bond_indices(periodic) for bond in periodic_bonds)), len(periodic_bonds)), mean_distance, max_distance], dtype=np.float64)
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
            values[row] = np.nan
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {'source': 'official train/test SMILES only; compact infinite-chain proxy ratios over repeat-core mass, endpoint backbone path, sidechain bulk, electronic density, and periodic closure graph', 'dense_feature_count': len(names), 'status_counts': status_counts, 'nonfinite_values': int(np.count_nonzero(~np.isfinite(values)))}
    return (values, names, report)

def opsin_endpoint_markers(smiles: str) -> str:
    parts: list[str] = []
    marker_index = 0
    for char in str(smiles):
        if char == '*':
            marker_index += 1
            parts.append(f'[*:{marker_index}]')
        else:
            parts.append(char)
    if marker_index != 2:
        raise RuntimeError(f'expected exactly two polymer endpoints, found {marker_index}')
    return ''.join(parts)

def bicerano_feature_matrix(smiles: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not BICERANO_AVAILABLE or bicerano_ppf is None:
        raise RuntimeError('polymer_property_prediction is required for --bicerano-features. Install public code only, for example: pip install --no-deps polymer_property_prediction')
    names = [f'bicerano_{column}' for column in BICERANO_COLUMNS]
    values = np.empty((len(smiles), len(names)), dtype=np.float64)
    values.fill(np.nan)
    status_counts: dict[str, int] = {}
    for row, smi in enumerate(smiles):
        status = 'ok'
        try:
            obj = SimpleNamespace(name=f'row_{row}', smiles=opsin_endpoint_markers(str(smi)), temperature=298.0, pressure=101325.0, polymer_concentration_wt=1.0, Mn=10000.0)
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_bicerano_timeout)
            signal.setitimer(signal.ITIMER_REAL, 3.0)
            blocker = rdBase.BlockLogs()
            try:
                result = bicerano_ppf.calculateMol(obj)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
                del blocker
            record = result.iloc[0].to_dict()
            for col, column in enumerate(BICERANO_COLUMNS):
                parsed = read_float(record.get(column))
                values[row, col] = parsed if parsed is not None else math.nan
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {'source': "public IBM polymer_property_prediction Bicerano-style group-contribution formulae; computed from official train/test SMILES only after adapting '*' endpoints to [*:1]/[*:2]", 'descriptor_count': len(names), 'status_counts': status_counts, 'nonfinite_by_column': {name: int(np.count_nonzero(~np.isfinite(values[:, index]))) for index, name in enumerate(names)}, 'external_data_training_use': False, 'pretrained_model_use': False}
    return (values, names, report)

def conjugation_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = ['conj_component_count', 'largest_conj_atom_count', 'largest_conj_atom_fraction', 'largest_conj_bond_count', 'largest_conj_bond_fraction', 'largest_conj_hetero_fraction', 'largest_conj_aromatic_atom_fraction', 'largest_conj_mean_atomic_num', 'largest_conj_path_len', 'conj_atom_count', 'conj_atom_fraction', 'conj_bond_count', 'conj_bond_fraction', 'aromatic_bond_count', 'aromatic_bond_fraction', 'nonaromatic_conj_bond_count', 'nonaromatic_conj_bond_fraction', 'sp2sp_bond_count', 'sp2sp_bond_fraction', 'hetero_conj_atom_count', 'hetero_conj_atom_fraction', 'donor_atom_count', 'acceptor_atom_count', 'donor_acceptor_pair_count', 'donor_acceptor_min_path', 'donor_acceptor_mean_path', 'donor_acceptor_conj_path_fraction', 'carbonyl_count', 'carbonyl_aromatic_neighbor_count', 'nitrile_count_electronic', 'imine_count', 'azo_count_electronic', 'halogen_on_conj_count', 'dummy_neighbor_conj_count', 'dummy_neighbor_aromatic_count', 'dummy_neighbor_unsat_count', 'ring_count_conj', 'aromatic_ring_count_conj', 'fused_ring_pair_count', 'largest_fused_ring_component', 'aromatic_atom_count_conj', 'aromatic_atom_fraction_conj', 'aliphatic_unsat_bond_count', 'hetero_to_carbon_conj_ratio']
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    carbonyl = Chem.MolFromSmarts('[CX3]=[OX1]')
    nitrile = Chem.MolFromSmarts('[CX2]#N')
    imine = Chem.MolFromSmarts('[CX3]=[NX2,NX3]')
    azo = Chem.MolFromSmarts('[NX2]=[NX2]')
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        heavy_atoms = [atom for atom in atoms if atom.GetAtomicNum() > 1]
        heavy_count = max(len(heavy_atoms), 1)
        bond_count = max(len(bonds), 1)
        conj_bonds = [bond for bond in bonds if bond.GetIsConjugated() or bond.GetBondType() == Chem.BondType.AROMATIC]
        conj_bond_indices = {bond.GetIdx() for bond in conj_bonds}
        conj_atoms = sorted({idx for bond in conj_bonds for idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())})
        conj_atom_set = set(conj_atoms)
        aromatic_atoms = {atom.GetIdx() for atom in atoms if atom.GetIsAromatic()}
        hetero_atoms = {atom.GetIdx() for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6)}
        adjacency: dict[int, set[int]] = {idx: set() for idx in conj_atoms}
        for bond in conj_bonds:
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        components: list[set[int]] = []
        seen: set[int] = set()
        for start in conj_atoms:
            if start in seen:
                continue
            stack = [start]
            comp: set[int] = set()
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nxt in adjacency.get(cur, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
                        pass
            components.append(comp)
        largest = max(components, key=len, default=set())
        largest_bonds = [bond for bond in conj_bonds if bond.GetBeginAtomIdx() in largest and bond.GetEndAtomIdx() in largest]

        def farthest_from(start: int, component: set[int]) -> tuple[int, int]:
            queue = [(start, 0)]
            local_seen = {start}
            farthest = (start, 0)
            for cur, dist in queue:
                if dist > farthest[1]:
                    farthest = (cur, dist)
                for nxt in adjacency.get(cur, ()):
                    if nxt in component and nxt not in local_seen:
                        local_seen.add(nxt)
                        queue.append((nxt, dist + 1))
                        pass
            return farthest

        def approximate_longest_path_length(component: set[int]) -> int:
            if len(component) < 2:
                return 0
            first = min(component)
            edge, _ = farthest_from(first, component)
            _, diameter = farthest_from(edge, component)
            return int(diameter)
        donors = []
        acceptors = []
        for atom in atoms:
            anum = atom.GetAtomicNum()
            if anum in (7, 8, 16) and atom.GetTotalNumHs() > 0:
                donors.append(atom.GetIdx())
                pass
            if anum in (7, 8, 16) and atom.GetFormalCharge() <= 0:
                acceptors.append(atom.GetIdx())
                pass
        da_paths = []
        da_conj_paths = 0
        da_pairs = [(donor, acceptor) for donor in donors for acceptor in acceptors if donor != acceptor]
        if len(da_pairs) > 64:
            step = max(1, len(da_pairs) // 64)
            da_pairs = da_pairs[::step][:64]
        for donor, acceptor in da_pairs:
            try:
                path = list(Chem.rdmolops.GetShortestPath(mol, donor, acceptor))
            except Exception:
                continue
            if len(path) < 2:
                continue
            length = len(path) - 1
            da_paths.append(length)
            all_conj = True
            for a, b in zip(path[:-1], path[1:], strict=True):
                bond = mol.GetBondBetweenAtoms(int(a), int(b))
                if bond is None or bond.GetIdx() not in conj_bond_indices:
                    all_conj = False
                    break
            da_conj_paths += int(all_conj)
        ring_info = mol.GetRingInfo()
        atom_rings = [set(ring) for ring in ring_info.AtomRings()]
        aromatic_ring_count = 0
        for ring in atom_rings:
            if ring and all((mol.GetAtomWithIdx(int(idx)).GetIsAromatic() for idx in ring)):
                aromatic_ring_count += 1
        fused_edges: dict[int, set[int]] = {idx: set() for idx in range(len(atom_rings))}
        fused_pairs = 0
        for i, ring_i in enumerate(atom_rings):
            for j in range(i + 1, len(atom_rings)):
                if ring_i.intersection(atom_rings[j]):
                    fused_pairs += 1
                    fused_edges[i].add(j)
                    fused_edges[j].add(i)
        largest_fused = 0
        ring_seen: set[int] = set()
        for start in fused_edges:
            if start in ring_seen:
                continue
            stack = [start]
            ring_seen.add(start)
            size = 0
            while stack:
                cur = stack.pop()
                size += 1
                for nxt in fused_edges[cur]:
                    if nxt not in ring_seen:
                        ring_seen.add(nxt)
                        stack.append(nxt)
                        pass
            largest_fused = max(largest_fused, size)
        carbonyl_matches = mol.GetSubstructMatches(carbonyl) if carbonyl is not None else ()
        carbonyl_aromatic = 0
        for match in carbonyl_matches:
            c_idx = int(match[0])
            atom = mol.GetAtomWithIdx(c_idx)
            if any((neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors())):
                carbonyl_aromatic += 1
        dummy_neighbor_conj = 0
        dummy_neighbor_aromatic = 0
        dummy_neighbor_unsat = 0
        for atom in atoms:
            if atom.GetAtomicNum() != 0:
                continue
            for neighbor in atom.GetNeighbors():
                nidx = neighbor.GetIdx()
                dummy_neighbor_conj += int(nidx in conj_atom_set)
                dummy_neighbor_aromatic += int(neighbor.GetIsAromatic())
                connecting = mol.GetBondBetweenAtoms(atom.GetIdx(), nidx)
                if connecting is not None:
                    dummy_neighbor_unsat += int(connecting.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC))
        conj_heavy = [idx for idx in conj_atoms if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() > 1]
        conj_carbon_count = sum((1 for idx in conj_heavy if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() == 6))
        hetero_conj_count = sum((1 for idx in conj_heavy if idx in hetero_atoms))
        sp2sp_bonds = [bond for bond in bonds if bond.GetBeginAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2) and bond.GetEndAtom().GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)]
        aliphatic_unsat = [bond for bond in bonds if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) and (not bond.GetBeginAtom().GetIsAromatic()) and (not bond.GetEndAtom().GetIsAromatic())]
        values[row] = np.asarray([len(components), len(largest), len(largest) / heavy_count, len(largest_bonds), len(largest_bonds) / bond_count, safe_ratio(sum((1 for idx in largest if idx in hetero_atoms)), len(largest)), safe_ratio(sum((1 for idx in largest if idx in aromatic_atoms)), len(largest)), safe_ratio(sum((mol.GetAtomWithIdx(int(idx)).GetAtomicNum() for idx in largest)), len(largest)), approximate_longest_path_length(largest), len(conj_heavy), len(conj_heavy) / heavy_count, len(conj_bonds), len(conj_bonds) / bond_count, sum((1 for bond in bonds if bond.GetBondType() == Chem.BondType.AROMATIC)), sum((1 for bond in bonds if bond.GetBondType() == Chem.BondType.AROMATIC)) / bond_count, sum((1 for bond in conj_bonds if bond.GetBondType() != Chem.BondType.AROMATIC)), sum((1 for bond in conj_bonds if bond.GetBondType() != Chem.BondType.AROMATIC)) / bond_count, len(sp2sp_bonds), len(sp2sp_bonds) / bond_count, hetero_conj_count, hetero_conj_count / heavy_count, len(donors), len(acceptors), len(da_paths), min(da_paths) if da_paths else 0, float(np.mean(da_paths)) if da_paths else 0.0, safe_ratio(da_conj_paths, len(da_paths)), len(carbonyl_matches), carbonyl_aromatic, len(mol.GetSubstructMatches(nitrile)) if nitrile is not None else 0, len(mol.GetSubstructMatches(imine)) if imine is not None else 0, len(mol.GetSubstructMatches(azo)) if azo is not None else 0, sum((1 for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53) and any((n.GetIdx() in conj_atom_set for n in atom.GetNeighbors())))), dummy_neighbor_conj, dummy_neighbor_aromatic, dummy_neighbor_unsat, len(atom_rings), aromatic_ring_count, fused_pairs, largest_fused, len(aromatic_atoms), len(aromatic_atoms) / heavy_count, len(aliphatic_unsat), safe_ratio(hetero_conj_count, conj_carbon_count)], dtype=np.float64)
        status_counts['ok'] = status_counts.get('ok', 0) + 1
    return (values, names, {'source': 'official train/test SMILES only; conjugation/electronic topology descriptors from RDKit graph, no external data', 'descriptor_count': len(names), 'status_counts': status_counts})

def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0

def mobility_feature_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Effective-mobility descriptors inspired by conjugated-polymer Tg work."""
    names = ['mobility_zeta_mean_a06', 'mobility_zeta_mass_weighted_a06', 'mobility_side_atom_fraction', 'mobility_side_mass_fraction', 'mobility_backbone_atom_fraction', 'mobility_backbone_mass_fraction', 'mobility_flexible_side_atom_fraction', 'mobility_rigid_atom_fraction', 'mobility_aromatic_rigid_atom_fraction', 'mobility_thiophene_atom_fraction', 'mobility_phenyl_atom_fraction', 'mobility_carbonyl_atom_fraction', 'mobility_alkenyl_atom_fraction', 'mobility_side_alkyl_atom_fraction', 'mobility_side_hetero_atom_fraction', 'mobility_side_rotatable_per_heavy', 'mobility_bb_rotatable_per_heavy', 'mobility_side_to_bb_mass_ratio', 'mobility_flexible_to_rigid_ratio', 'mobility_fox_inv_proxy_a06', 'mobility_backbone_stiffness_proxy_a06', 'mobility_internal_plasticization_proxy', 'mobility_rigid_ring_count', 'mobility_side_component_count', 'mobility_longest_side_component_fraction', 'mobility_linear_alkyl_tail_max', 'mobility_linear_alkyl_tail_mean', 'mobility_endpoint_path_aromatic_fraction', 'mobility_endpoint_path_sp2_fraction', 'mobility_endpoint_path_hetero_fraction']
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    carbonyl = Chem.MolFromSmarts('[CX3]=[OX1]')
    thiophene = Chem.MolFromSmarts('[s]1[c][c][c][c]1')
    phenyl = Chem.MolFromSmarts('c1ccccc1')

    def side_components(mol: Chem.Mol, side_heavy: set[int]) -> list[set[int]]:
        components: list[set[int]] = []
        seen: set[int] = set()
        for start in sorted(side_heavy):
            if start in seen:
                continue
            comp = {start}
            seen.add(start)
            queue = [start]
            cursor = 0
            while cursor < len(queue):
                current = queue[cursor]
                cursor += 1
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    idx = int(neighbor.GetIdx())
                    if idx not in side_heavy or idx in seen:
                        continue
                    seen.add(idx)
                    comp.add(idx)
                    queue.append(idx)
            components.append(comp)
        return components

    def longest_alkyl_tail(mol: Chem.Mol, comp: set[int]) -> int:
        alkyl = {idx for idx in comp if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 and (not mol.GetAtomWithIdx(idx).GetIsAromatic()) and (mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3)}
        if not alkyl:
            return 0
        best = 1
        for start in alkyl:
            stack = [(start, 1, {start})]
            while stack:
                current, length, path_seen = stack.pop()
                best = max(best, length)
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    idx = int(neighbor.GetIdx())
                    if idx not in alkyl or idx in path_seen:
                        continue
                    stack.append((idx, length + 1, path_seen | {idx}))
        return best
    for row, mol in enumerate(mols):
        status = 'ok'
        try:
            dummy_atoms = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError(f'expected exactly two dummy atoms, observed {len(dummy_atoms)}')
            path = tuple((int(idx) for idx in Chem.rdmolops.GetShortestPath(mol, dummy_atoms[0], dummy_atoms[1])))
            if not path:
                raise RuntimeError('dummy endpoints are disconnected')
            path_set = set(path)
            atoms = list(mol.GetAtoms())
            heavy_atoms = {int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() > 1}
            heavy_count = max(len(heavy_atoms), 1)
            backbone_heavy = {idx for idx in path_set if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
            side_heavy = heavy_atoms.difference(path_set)
            masses = {idx: float(mol.GetAtomWithIdx(idx).GetMass()) for idx in heavy_atoms}
            total_mass = sum(masses.values())
            backbone_mass = sum((masses[idx] for idx in backbone_heavy))
            side_mass = sum((masses[idx] for idx in side_heavy))
            aromatic_atoms = {idx for idx in heavy_atoms if mol.GetAtomWithIdx(idx).GetIsAromatic()}
            hetero_side = {idx for idx in side_heavy if mol.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6)}
            carbonyl_atoms = {int(idx) for match in (mol.GetSubstructMatches(carbonyl) if carbonyl is not None else ()) for idx in match if int(idx) in heavy_atoms}
            thiophene_atoms = {int(idx) for match in (mol.GetSubstructMatches(thiophene) if thiophene is not None else ()) for idx in match if int(idx) in heavy_atoms}
            phenyl_atoms = {int(idx) for match in (mol.GetSubstructMatches(phenyl) if phenyl is not None else ()) for idx in match if int(idx) in heavy_atoms}
            alkenyl_atoms = {int(bond.GetBeginAtomIdx()) for bond in mol.GetBonds() if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)} | {int(bond.GetEndAtomIdx()) for bond in mol.GetBonds() if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)}
            side_alkyl_atoms = {idx for idx in side_heavy if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 and (not mol.GetAtomWithIdx(idx).GetIsAromatic()) and (mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3)}
            flexible_atoms = set(side_alkyl_atoms)
            rigid_atoms = heavy_atoms.difference(flexible_atoms)
            mobility = {}
            for idx in heavy_atoms:
                if idx in flexible_atoms:
                    mobility[idx] = 1.0
                elif idx in thiophene_atoms:
                    mobility[idx] = 0.72
                else:
                    mobility[idx] = 0.6
            zeta_mean = sum(mobility.values()) / heavy_count
            zeta_mass = safe_ratio(sum((mobility[idx] * masses[idx] for idx in heavy_atoms)), total_mass)
            rotatable = region_rotatable_bond_indices(mol)
            bb_bonds: set[int] = set()
            side_bonds: set[int] = set()
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtomIdx())
                b = int(bond.GetEndAtomIdx())
                if a in path_set and b in path_set:
                    bb_bonds.add(int(bond.GetIdx()))
                    pass
                elif a in side_heavy and b in side_heavy:
                    side_bonds.add(int(bond.GetIdx()))
                    pass
            components = side_components(mol, side_heavy)
            comp_sizes = [len(comp) for comp in components]
            alkyl_lengths = [longest_alkyl_tail(mol, comp) for comp in components]
            path_heavy = [idx for idx in path if idx in heavy_atoms]
            path_denom = max(len(path_heavy), 1)
            fox_inv_proxy = safe_ratio(side_mass, total_mass) * 1.0 + safe_ratio(backbone_mass, total_mass) * safe_ratio(1.0, max(zeta_mean, 1e-09))
            values[row] = np.asarray([zeta_mean, zeta_mass, safe_ratio(len(side_heavy), heavy_count), safe_ratio(side_mass, total_mass), safe_ratio(len(backbone_heavy), heavy_count), safe_ratio(backbone_mass, total_mass), safe_ratio(len(flexible_atoms), heavy_count), safe_ratio(len(rigid_atoms), heavy_count), safe_ratio(len(aromatic_atoms & rigid_atoms), heavy_count), safe_ratio(len(thiophene_atoms), heavy_count), safe_ratio(len(phenyl_atoms), heavy_count), safe_ratio(len(carbonyl_atoms), heavy_count), safe_ratio(len(alkenyl_atoms & heavy_atoms), heavy_count), safe_ratio(len(side_alkyl_atoms), heavy_count), safe_ratio(len(hetero_side), heavy_count), safe_ratio(sum((idx in rotatable for idx in side_bonds)), heavy_count), safe_ratio(sum((idx in rotatable for idx in bb_bonds)), heavy_count), safe_ratio(side_mass, backbone_mass), safe_ratio(len(flexible_atoms), len(rigid_atoms)), fox_inv_proxy, safe_ratio(1.0 - zeta_mean, max(safe_ratio(backbone_mass, total_mass), 1e-09)), safe_ratio(side_mass, total_mass) * safe_ratio(len(flexible_atoms), len(side_heavy)), float(sum((1 for ring in mol.GetRingInfo().AtomRings() if any((int(idx) in rigid_atoms for idx in ring))))), float(len(components)), safe_ratio(max(comp_sizes) if comp_sizes else 0, len(side_heavy)), float(max(alkyl_lengths) if alkyl_lengths else 0), float(np.mean(alkyl_lengths)) if alkyl_lengths else 0.0, safe_ratio(sum((idx in aromatic_atoms for idx in path_heavy)), path_denom), safe_ratio(sum((mol.GetAtomWithIdx(idx).GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2) for idx in path_heavy)), path_denom), safe_ratio(sum((mol.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6) for idx in path_heavy)), path_denom)], dtype=np.float64)
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
            values[row] = np.nan
        status_counts[status] = status_counts.get(status, 0) + 1
    return (values, names, {'source': 'official train/test SMILES only; effective atomic mobility, side-chain mass fraction, and rigid/flexible backbone proxies inspired by conjugated-polymer Tg literature', 'descriptor_count': len(names), 'status_counts': status_counts, 'nonfinite_values': int(np.count_nonzero(~np.isfinite(values))), 'external_data_training_use': False, 'pretrained_model_use': False})

def region_rotatable_bond_indices(mol: Chem.Mol) -> set[int]:
    out: set[int] = set()
    for match in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts):
        if len(match) >= 2:
            bond = mol.GetBondBetweenAtoms(int(match[0]), int(match[1]))
            if bond is not None:
                out.add(int(bond.GetIdx()))
                pass
    return out

def region_atom_counts(mol: Chem.Mol, atom_indices: set[int]) -> list[float]:
    atoms = [mol.GetAtomWithIdx(int(idx)) for idx in atom_indices]
    denom = sum((1 for atom in atoms if atom.GetAtomicNum() > 1))
    values: list[float] = []
    for atomic_num in REGION_ATOM_NUMBERS:
        count = sum((1 for atom in atoms if atom.GetAtomicNum() == atomic_num))
        values.append(float(count))
        values.append(safe_ratio(count, denom))
    return values

def region_charge_stats(mol: Chem.Mol, atom_indices: set[int]) -> list[float]:
    charges: list[float] = []
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(idx))
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
        except Exception:
            charge = math.nan
        if math.isfinite(charge):
            charges.append(charge)
            pass
    if not charges:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    arr = np.asarray(charges, dtype=np.float64)
    return [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)), float(np.sum(np.maximum(arr, 0.0))), float(np.sum(np.minimum(arr, 0.0)))]

def region_fragment_descriptor_values(fragment: Chem.Mol | None) -> list[float]:
    if fragment is None or fragment.GetNumAtoms() == 0:
        return [0.0] * len(REGION_DESCRIPTOR_NAMES)
    values: list[float] = []
    for name in REGION_DESCRIPTOR_NAMES:
        func = getattr(Descriptors, name)
        try:
            value = float(func(fragment))
        except Exception:
            value = math.nan
        values.append(value if math.isfinite(value) else math.nan)
    return values

def region_fragment_from_atoms(mol: Chem.Mol, atom_indices: set[int]) -> Chem.Mol | None:
    if not atom_indices:
        return None
    try:
        fragment_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted((int(index) for index in atom_indices)), canonical=True)
        return Chem.MolFromSmiles(fragment_smiles, sanitize=True) if fragment_smiles else None
    except Exception:
        return None

def side_distance_stats(mol: Chem.Mol, side_indices: set[int], backbone_indices: set[int]) -> list[float]:
    if not side_indices or not backbone_indices:
        return [0.0, 0.0, 0.0, 0.0]
    distances = {int(idx): 0 for idx in backbone_indices}
    queue = list(distances)
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = int(neighbor.GetIdx())
            if idx in distances:
                continue
            distances[idx] = distances[current] + 1
            queue.append(idx)
    vals = np.asarray([distances.get(int(idx), 0) for idx in side_indices], dtype=np.float64)
    if vals.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(vals.max()), float(vals.mean()), float(np.median(vals)), float(vals.std())]

def side_component_stats(mol: Chem.Mol, side_indices: set[int], path: tuple[int, ...]) -> list[float]:
    if not side_indices:
        return [0.0] * 26
    side_set = {int(idx) for idx in side_indices}
    path_set = {int(idx) for idx in path}
    path_pos = {int(idx): pos for pos, idx in enumerate(path)}
    components: list[set[int]] = []
    seen: set[int] = set()
    for start in sorted(side_set):
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        queue = [start]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                idx = int(neighbor.GetIdx())
                if idx not in side_set or idx in seen:
                    continue
                seen.add(idx)
                comp.add(idx)
                queue.append(idx)
        components.append(comp)
    comp_sizes = np.asarray([len(comp) for comp in components], dtype=np.float64)
    comp_masses = np.asarray([sum((float(mol.GetAtomWithIdx(idx).GetMass()) for idx in comp)) for comp in components], dtype=np.float64)
    comp_attachment_counts: list[float] = []
    comp_attachment_positions: list[float] = []
    for comp in components:
        positions: list[int] = []
        attachment_count = 0
        for idx in comp:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = int(neighbor.GetIdx())
                if n_idx in path_set:
                    attachment_count += 1
                    if n_idx in path_pos:
                        positions.append(path_pos[n_idx])
                        pass
        comp_attachment_counts.append(float(attachment_count))
        comp_attachment_positions.append(float(np.mean(positions)) if positions else 0.0)
    attachment_counts = np.asarray(comp_attachment_counts, dtype=np.float64)
    attachment_positions = np.asarray(comp_attachment_positions, dtype=np.float64)
    terminal_count = sum((1 for idx in side_set if sum((int(neighbor.GetIdx()) in side_set for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors())) <= 1))
    side_total = max(float(len(side_set)), 1.0)
    size_fraction = comp_sizes / side_total
    diversity = 1.0 - float(np.sum(size_fraction * size_fraction))
    max_component_fraction = float(np.max(size_fraction)) if size_fraction.size else 0.0
    endpoint_distances: list[float] = []
    distance_ratios: list[float] = []
    dummy_atoms = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) == 2:
        try:
            distances = Chem.GetDistanceMatrix(mol)
            for idx in side_set:
                nearest = float(min(distances[idx, dummy_atoms[0]], distances[idx, dummy_atoms[1]]))
                endpoint_distances.append(nearest)
                distance_ratios.append(safe_ratio(1.0, nearest))
        except Exception:
            endpoint_distances = []
            distance_ratios = []
    endpoint_arr = np.asarray(endpoint_distances, dtype=np.float64)
    ratio_arr = np.asarray(distance_ratios, dtype=np.float64)

    def stats(vals: np.ndarray) -> list[float]:
        if vals.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), float(np.std(vals))]
    return [float(len(components)), *stats(comp_sizes), *stats(comp_masses), *stats(attachment_counts), float(terminal_count), safe_ratio(float(terminal_count), len(side_set)), diversity, max_component_fraction, *stats(attachment_positions), *stats(endpoint_arr), float(np.mean(ratio_arr)) if ratio_arr.size else 0.0]

def endpoint_environment_values(mol: Chem.Mol, dummy_atoms: list[int]) -> list[float]:
    values: list[float] = []
    for idx in sorted(dummy_atoms):
        atom = mol.GetAtomWithIdx(int(idx))
        neighbors = list(atom.GetNeighbors())
        neighbor = neighbors[0] if neighbors else None
        if neighbor is None:
            values.extend([0.0, 0.0, 0.0, 0.0])
            continue
        bond = mol.GetBondBetweenAtoms(int(idx), int(neighbor.GetIdx()))
        values.extend([float(neighbor.GetAtomicNum()), float(neighbor.GetDegree()), float(neighbor.GetIsAromatic()), float(bond.GetBondTypeAsDouble()) if bond is not None else 0.0])
    return values

def backbone_sidechain_feature_names() -> list[str]:
    names = ['bb_attachment_shortest_path_bonds', 'bb_heavy_atom_count', 'side_heavy_atom_count', 'side_heavy_atom_ratio', 'bb_bond_count', 'side_bond_count', 'bb_side_crossing_bond_count', 'bb_rotatable_bond_count', 'side_rotatable_bond_count', 'bb_rotatable_bond_ratio', 'side_rotatable_bond_ratio', 'bb_aromatic_atom_count', 'side_aromatic_atom_count', 'bb_aromatic_atom_fraction', 'side_aromatic_atom_fraction', 'bb_hetero_atom_count', 'side_hetero_atom_count', 'bb_hetero_atom_fraction', 'side_hetero_atom_fraction', 'bb_ring_touch_count', 'side_ring_touch_count', 'side_max_distance_to_backbone', 'side_mean_distance_to_backbone', 'side_median_distance_to_backbone', 'side_std_distance_to_backbone', 'side_component_count', 'side_component_heavy_min', 'side_component_heavy_mean', 'side_component_heavy_max', 'side_component_heavy_std', 'side_component_mass_min', 'side_component_mass_mean', 'side_component_mass_max', 'side_component_mass_std', 'side_component_attachment_min', 'side_component_attachment_mean', 'side_component_attachment_max', 'side_component_attachment_std', 'side_terminal_atom_count', 'side_terminal_atom_fraction', 'side_component_diversity', 'side_max_component_fraction', 'side_attachment_path_position_min', 'side_attachment_path_position_mean', 'side_attachment_path_position_max', 'side_attachment_path_position_std', 'side_distance_to_endpoint_min', 'side_distance_to_endpoint_mean', 'side_distance_to_endpoint_max', 'side_distance_to_endpoint_std', 'side_inverse_endpoint_distance_mean']
    for endpoint in ('a', 'b'):
        names.extend([f'endpoint_{endpoint}_neighbor_atomic_num', f'endpoint_{endpoint}_neighbor_degree', f'endpoint_{endpoint}_neighbor_is_aromatic', f'endpoint_{endpoint}_bond_order'])
        pass
    for region in ('bb', 'side'):
        for atomic_num in REGION_ATOM_NUMBERS:
            names.append(f'{region}_atomic_num_{atomic_num}_count')
            names.append(f'{region}_atomic_num_{atomic_num}_fraction')
            pass
    for region in ('whole', 'bb', 'side'):
        names.extend([f'{region}_gasteiger_mean', f'{region}_gasteiger_std', f'{region}_gasteiger_min', f'{region}_gasteiger_max', f'{region}_gasteiger_positive_sum', f'{region}_gasteiger_negative_sum'])
        pass
    for prefix in ('whole', 'bb', 'side'):
        names.extend([f'{prefix}_rdkit_{name}' for name in REGION_DESCRIPTOR_NAMES])
        pass
    for name in REGION_DESCRIPTOR_NAMES:
        names.append(f'side_over_whole_rdkit_{name}')
        names.append(f'bb_over_whole_rdkit_{name}')
        pass
    return names

def backbone_sidechain_values(mol: Chem.Mol) -> list[float]:
    work = Chem.Mol(mol)
    try:
        AllChem.ComputeGasteigerCharges(work, nIter=12, throwOnParamFailure=False)
        pass
    except Exception:
        pass
    dummy_atoms = [int(atom.GetIdx()) for atom in work.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 2:
        raise RuntimeError(f'expected exactly two dummy atoms, observed {len(dummy_atoms)}')
    path = list(Chem.rdmolops.GetShortestPath(work, int(dummy_atoms[0]), int(dummy_atoms[1])))
    if not path:
        raise RuntimeError('dummy endpoints are disconnected')
    path_set = set((int(idx) for idx in path))
    all_atoms = set(range(work.GetNumAtoms()))
    heavy_atoms = {int(atom.GetIdx()) for atom in work.GetAtoms() if atom.GetAtomicNum() > 1}
    backbone_heavy = {idx for idx in path_set if work.GetAtomWithIdx(idx).GetAtomicNum() > 1}
    side_heavy = heavy_atoms.difference(path_set)
    rotatable = region_rotatable_bond_indices(work)
    backbone_bonds: set[int] = set()
    side_bonds: set[int] = set()
    crossing_bonds = 0
    for bond in work.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        if a in path_set and b in path_set:
            backbone_bonds.add(int(bond.GetIdx()))
            pass
        elif a not in path_set and b not in path_set:
            side_bonds.add(int(bond.GetIdx()))
            pass
        else:
            crossing_bonds += 1
    aromatic_backbone = sum((1 for idx in backbone_heavy if work.GetAtomWithIdx(idx).GetIsAromatic()))
    aromatic_side = sum((1 for idx in side_heavy if work.GetAtomWithIdx(idx).GetIsAromatic()))
    hetero_backbone = sum((1 for idx in backbone_heavy if work.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6)))
    hetero_side = sum((1 for idx in side_heavy if work.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6)))
    rings = work.GetRingInfo().AtomRings()
    backbone_ring_touches = sum((1 for ring in rings if any((int(idx) in backbone_heavy for idx in ring))))
    side_ring_touches = sum((1 for ring in rings if any((int(idx) in side_heavy for idx in ring))))
    backbone_fragment = region_fragment_from_atoms(work, backbone_heavy)
    side_fragment = region_fragment_from_atoms(work, side_heavy)
    values = [float(len(path) - 1), float(len(backbone_heavy)), float(len(side_heavy)), safe_ratio(len(side_heavy), len(heavy_atoms)), float(len(backbone_bonds)), float(len(side_bonds)), float(crossing_bonds), float(sum((1 for idx in backbone_bonds if idx in rotatable))), float(sum((1 for idx in side_bonds if idx in rotatable))), safe_ratio(sum((1 for idx in backbone_bonds if idx in rotatable)), len(backbone_bonds)), safe_ratio(sum((1 for idx in side_bonds if idx in rotatable)), len(side_bonds)), float(aromatic_backbone), float(aromatic_side), safe_ratio(aromatic_backbone, len(backbone_heavy)), safe_ratio(aromatic_side, len(side_heavy)), float(hetero_backbone), float(hetero_side), safe_ratio(hetero_backbone, len(backbone_heavy)), safe_ratio(hetero_side, len(side_heavy)), float(backbone_ring_touches), float(side_ring_touches)]
    values.extend(side_distance_stats(work, side_heavy, backbone_heavy))
    values.extend(side_component_stats(work, side_heavy, tuple(path)))
    values.extend(endpoint_environment_values(work, dummy_atoms))
    values.extend(region_atom_counts(work, backbone_heavy))
    values.extend(region_atom_counts(work, side_heavy))
    values.extend(region_charge_stats(work, heavy_atoms))
    values.extend(region_charge_stats(work, backbone_heavy))
    values.extend(region_charge_stats(work, side_heavy))
    whole_desc = region_fragment_descriptor_values(work)
    backbone_desc = region_fragment_descriptor_values(backbone_fragment)
    side_desc = region_fragment_descriptor_values(side_fragment)
    values.extend(whole_desc)
    values.extend(backbone_desc)
    values.extend(side_desc)
    for whole, back, side in zip(whole_desc, backbone_desc, side_desc, strict=True):
        values.append(safe_ratio(side, whole) if math.isfinite(side) and math.isfinite(whole) else math.nan)
        values.append(safe_ratio(back, whole) if math.isfinite(back) and math.isfinite(whole) else math.nan)
        pass
    return values

def backbone_sidechain_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = backbone_sidechain_feature_names()
    values = np.empty((len(mols), len(names)), dtype=np.float64)
    values.fill(np.nan)
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        status = 'ok'
        try:
            row_values = backbone_sidechain_values(mol)
            if len(row_values) != len(names):
                raise RuntimeError(f'name/value mismatch {len(row_values)} != {len(names)}')
            values[row] = np.asarray(row_values, dtype=np.float64)
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
        status_counts[status] = status_counts.get(status, 0) + 1
    values[~np.isfinite(values)] = np.nan
    return (values, [f'bb_side_{name}' for name in names], {'source': 'official train/test SMILES only; shortest dummy-endpoint path treated as backbone and off-path heavy atoms as side-chain region', 'descriptor_count': len(names), 'status_counts': status_counts, 'nonfinite_values': int(np.count_nonzero(~np.isfinite(values)))})

def atom_category(atom: Chem.Atom) -> str:
    if atom.GetAtomicNum() == 0:
        return '*'
    symbol = atom.GetSymbol()
    aromatic = 'a' if atom.GetIsAromatic() else ''
    ring = 'r' if atom.IsInRing() else ''
    hyb = str(atom.GetHybridization()).rsplit('.', 1)[-1]
    return f'{aromatic}{symbol}{ring}:{hyb}'

def bond_category(bond: Chem.Bond) -> str:
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        base = 'arom'
    elif bond.GetBondType() == Chem.BondType.SINGLE:
        base = 'single'
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
        base = 'double'
    elif bond.GetBondType() == Chem.BondType.TRIPLE:
        base = 'triple'
    else:
        base = str(bond.GetBondType()).lower()
    if bond.GetIsConjugated():
        base += ':conj'
    if bond.IsInRing():
        base += ':ring'
    return base

def feature_token_index(token: str, n_features: int) -> int:
    # deterministic pure-Python token -> column index for the sparse
    # feature matrix (no imports, no checksums).
    value = 14695981039346656037
    for ch in str(token):
        value ^= ord(ch)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(value & 0xFFFFFFFF) % int(n_features)

def endpoint_neighbors_and_path(mol: Chem.Mol) -> tuple[list[Chem.Atom], tuple[int, ...]]:
    endpoints: list[Chem.Atom] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) == 1:
            endpoints.append(neighbors[0])
            pass
    if len(endpoints) != 2:
        return (endpoints, tuple())
    try:
        path = tuple(Chem.rdmolops.GetShortestPath(mol, endpoints[0].GetIdx(), endpoints[1].GetIdx()))
    except Exception:
        path = tuple()
    return (endpoints, path)

def motif_dense_matrix(mols: list[Chem.Mol], smiles: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    base_names = ['motif_heavy_atoms', 'motif_total_atoms', 'motif_total_bonds', 'motif_ring_count', 'motif_aromatic_ring_count', 'motif_aliphatic_ring_count', 'motif_saturated_ring_count', 'motif_rotatable_bonds', 'motif_hba', 'motif_hbd', 'motif_tpsa', 'motif_labute_asa', 'motif_fraction_csp3', 'motif_aromatic_atom_fraction', 'motif_ring_atom_fraction', 'motif_hetero_atom_fraction', 'motif_halogen_atom_fraction', 'motif_o_atom_fraction', 'motif_n_atom_fraction', 'motif_s_atom_fraction', 'motif_si_atom_fraction', 'motif_single_bond_fraction', 'motif_double_bond_fraction', 'motif_triple_bond_fraction', 'motif_aromatic_bond_fraction', 'motif_conjugated_bond_fraction', 'motif_endpoint_neighbor_same_atomic', 'motif_endpoint_neighbor_mean_atomic', 'motif_endpoint_neighbor_aromatic_fraction', 'motif_endpoint_neighbor_ring_fraction', 'motif_endpoint_path_length', 'motif_endpoint_path_heavy_atoms', 'motif_endpoint_path_aromatic_fraction', 'motif_endpoint_path_ring_fraction', 'motif_endpoint_path_hetero_fraction', 'motif_endpoint_path_branch_off_count', 'motif_endpoint_path_branch_off_fraction', 'motif_endpoint_path_single_bond_fraction', 'motif_endpoint_path_double_bond_fraction', 'motif_endpoint_path_triple_bond_fraction', 'motif_endpoint_path_aromatic_bond_fraction', 'motif_endpoint_path_conjugated_bond_fraction', 'motif_smiles_star_count', 'motif_smiles_ring_digit_count', 'motif_smiles_branch_count', 'motif_smiles_aromatic_char_fraction']
    smarts_names: list[str] = []
    for name, _ in COMPILED_SMARTS:
        smarts_names.extend([f'motif_smarts_{name}_count', f'motif_smarts_{name}_per_heavy'])
        pass
    names = base_names + smarts_names
    values = np.zeros((len(mols), len(names)), dtype=np.float64)
    status_counts: dict[str, int] = {}
    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        heavy = sum((1 for atom in atoms if atom.GetAtomicNum() > 1))
        total_atoms = len(atoms)
        total_bonds = len(bonds)
        heavy_denom = max(heavy, 1)
        bond_denom = max(total_bonds, 1)
        values[row, 0] = heavy
        values[row, 1] = total_atoms
        values[row, 2] = total_bonds
        values[row, 3] = mol.GetRingInfo().NumRings()
        values[row, 4] = rdMolDescriptors.CalcNumAromaticRings(mol)
        values[row, 5] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        values[row, 6] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        values[row, 7] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        values[row, 8] = rdMolDescriptors.CalcNumHBA(mol)
        values[row, 9] = rdMolDescriptors.CalcNumHBD(mol)
        try:
            values[row, 10] = rdMolDescriptors.CalcTPSA(mol)
        except Exception:
            values[row, 10] = 0.0
        try:
            values[row, 11] = rdMolDescriptors.CalcLabuteASA(mol)
        except Exception:
            values[row, 11] = 0.0
        try:
            values[row, 12] = rdMolDescriptors.CalcFractionCSP3(mol)
        except Exception:
            values[row, 12] = 0.0
        values[row, 13] = sum((atom.GetIsAromatic() for atom in atoms)) / heavy_denom
        values[row, 14] = sum((atom.IsInRing() for atom in atoms if atom.GetAtomicNum() > 1)) / heavy_denom
        values[row, 15] = sum((atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)) / heavy_denom
        values[row, 16] = sum((atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)) / heavy_denom
        values[row, 17] = sum((atom.GetAtomicNum() == 8 for atom in atoms)) / heavy_denom
        values[row, 18] = sum((atom.GetAtomicNum() == 7 for atom in atoms)) / heavy_denom
        values[row, 19] = sum((atom.GetAtomicNum() == 16 for atom in atoms)) / heavy_denom
        values[row, 20] = sum((atom.GetAtomicNum() == 14 for atom in atoms)) / heavy_denom
        values[row, 21] = sum((bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds)) / bond_denom
        values[row, 22] = sum((bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds)) / bond_denom
        values[row, 23] = sum((bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds)) / bond_denom
        values[row, 24] = sum((bond.GetBondType() == Chem.BondType.AROMATIC for bond in bonds)) / bond_denom
        values[row, 25] = sum((bond.GetIsConjugated() for bond in bonds)) / bond_denom
        endpoint_neighbors, endpoint_path = endpoint_neighbors_and_path(mol)
        if len(endpoint_neighbors) == 2:
            atomic_values = [atom.GetAtomicNum() for atom in endpoint_neighbors]
            values[row, 26] = float(atomic_values[0] == atomic_values[1])
            values[row, 27] = float(np.mean(atomic_values))
            values[row, 28] = float(np.mean([atom.GetIsAromatic() for atom in endpoint_neighbors]))
            values[row, 29] = float(np.mean([atom.IsInRing() for atom in endpoint_neighbors]))
        if endpoint_path:
            path_atoms = [mol.GetAtomWithIdx(index) for index in endpoint_path]
            path_bonds = [mol.GetBondBetweenAtoms(endpoint_path[i], endpoint_path[i + 1]) for i in range(len(endpoint_path) - 1)]
            path_bonds = [bond for bond in path_bonds if bond is not None]
            path_atom_denom = max(len(path_atoms), 1)
            path_bond_denom = max(len(path_bonds), 1)
            branch_off = 0
            path_set = set(endpoint_path)
            for atom in path_atoms:
                branch_off += sum((1 for neighbor in atom.GetNeighbors() if neighbor.GetIdx() not in path_set and neighbor.GetAtomicNum() != 0))
            values[row, 30] = max(0, len(endpoint_path) - 1)
            values[row, 31] = sum((atom.GetAtomicNum() > 1 for atom in path_atoms))
            values[row, 32] = sum((atom.GetIsAromatic() for atom in path_atoms)) / path_atom_denom
            values[row, 33] = sum((atom.IsInRing() for atom in path_atoms)) / path_atom_denom
            values[row, 34] = sum((atom.GetAtomicNum() not in (0, 1, 6) for atom in path_atoms)) / path_atom_denom
            values[row, 35] = branch_off
            values[row, 36] = branch_off / heavy_denom
            values[row, 37] = sum((bond.GetBondType() == Chem.BondType.SINGLE for bond in path_bonds)) / path_bond_denom
            values[row, 38] = sum((bond.GetBondType() == Chem.BondType.DOUBLE for bond in path_bonds)) / path_bond_denom
            values[row, 39] = sum((bond.GetBondType() == Chem.BondType.TRIPLE for bond in path_bonds)) / path_bond_denom
            values[row, 40] = sum((bond.GetBondType() == Chem.BondType.AROMATIC for bond in path_bonds)) / path_bond_denom
            values[row, 41] = sum((bond.GetIsConjugated() for bond in path_bonds)) / path_bond_denom
        smi_text = str(smi)
        values[row, 42] = smi_text.count('*')
        values[row, 43] = sum((ch.isdigit() for ch in smi_text))
        values[row, 44] = smi_text.count('(')
        values[row, 45] = sum((ch in 'bcnops' for ch in smi_text)) / max(len(smi_text), 1)
        offset = len(base_names)
        for name, pattern in COMPILED_SMARTS:
            try:
                count = float(len(mol.GetSubstructMatches(pattern, uniquify=True)))
            except Exception:
                count = 0.0
            values[row, offset] = count
            values[row, offset + 1] = count / heavy_denom
            offset += 2
        status_counts['ok'] = status_counts.get('ok', 0) + 1
    report = {'source': 'official train/test SMILES only; QSPR/GAP-inspired explicit motif counts, normalized chain descriptors, and endpoint-path descriptors', 'dense_feature_count': len(names), 'smarts_motif_count': len(COMPILED_SMARTS), 'status_counts': status_counts}
    return (values, names, report)

def motif_hash_matrix(mols: list[Chem.Mol], smiles: list[str], n_features: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    token_counts: dict[str, int] = {'smarts': 0, 'brics': 0, 'path': 0}
    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        for name, pattern in COMPILED_SMARTS:
            try:
                count = len(mol.GetSubstructMatches(pattern, uniquify=True))
            except Exception:
                count = 0
            if count:
                rows.append(row)
                cols.append(feature_token_index(f'smarts:{name}', n_features))
                data.append(float(math.log1p(count)))
                token_counts['smarts'] += 1
        capped = Chem.MolFromSmiles(cap_polymer_smiles(str(smi)), sanitize=True)
        if capped is not None:
            try:
                fragments = BRICS.BRICSDecompose(capped, keepNonLeafNodes=True, minFragmentSize=2)
            except Exception:
                fragments = []
            for fragment in fragments:
                token = f'brics:{fragment}'
                rows.append(row)
                cols.append(feature_token_index(token, n_features))
                data.append(1.0)
                token_counts['brics'] += 1
        seen_path_tokens: set[str] = set()
        for distance in range(1, 7):
            try:
                atom_paths = Chem.rdmolops.FindAllPathsOfLengthN(mol, distance, useBonds=False, useHs=False, onlyShortestPaths=True)
            except Exception:
                atom_paths = []
            for raw_path in atom_paths:
                path = tuple((int(index) for index in raw_path))
                if len(path) != distance + 1:
                    continue
                atom_tokens = [atom_category(mol.GetAtomWithIdx(index)) for index in path]
                bond_tokens: list[str] = []
                valid = True
                for index in range(distance):
                    bond = mol.GetBondBetweenAtoms(path[index], path[index + 1])
                    if bond is None:
                        valid = False
                        break
                    bond_tokens.append(bond_category(bond))
                if not valid:
                    continue
                forward = '|'.join(sum(zip(atom_tokens, bond_tokens + ['']), ()))
                reverse_atom_tokens = list(reversed(atom_tokens))
                reverse_bond_tokens = list(reversed(bond_tokens))
                reverse = '|'.join(sum(zip(reverse_atom_tokens, reverse_bond_tokens + ['']), ()))
                token = f'path{distance}:{min(forward, reverse)}'
                if token in seen_path_tokens:
                    continue
                seen_path_tokens.add(token)
                rows.append(row)
                cols.append(feature_token_index(token, n_features))
                data.append(1.0 / distance)
                token_counts['path'] += 1
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), int(n_features)), dtype=np.float32)
    if matrix.nnz:
        matrix.sum_duplicates()
        pass
    report = {'source': 'official train/test SMILES only; deterministic hashed SMARTS/BRICS/topological-path motif counts', 'n_features': int(n_features), 'nnz': int(matrix.nnz), 'token_counts': token_counts}
    return (matrix, report)

def atom_environment_token(mol: Chem.Mol, atom_index: int, radius: int) -> str:
    atom = mol.GetAtomWithIdx(int(atom_index))
    if radius <= 0:
        return atom_category(atom)
    try:
        bond_ids = list(Chem.rdmolops.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(atom_index)))
    except Exception:
        bond_ids = []
    if not bond_ids:
        return atom_category(atom)
    atom_ids: set[int] = {int(atom_index)}
    for bond_id in bond_ids:
        bond = mol.GetBondWithIdx(int(bond_id))
        atom_ids.add(int(bond.GetBeginAtomIdx()))
        atom_ids.add(int(bond.GetEndAtomIdx()))
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atom_ids), bondsToUse=sorted((int(bond_id) for bond_id in bond_ids)), canonical=True, isomericSmiles=False)
    except Exception:
        return atom_category(atom)

def map4_like_matrix(mols: list[Chem.Mol], n_features: int, max_distance: int, env_radius: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    token_count = 0
    skipped_pair_count = 0
    status_counts: dict[str, int] = {}
    for row, mol in enumerate(mols):
        status = 'ok'
        try:
            atom_indices = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
            env_tokens = {idx: atom_environment_token(mol, idx, env_radius) for idx in atom_indices}
            distances = Chem.GetDistanceMatrix(mol)
            local_counts: dict[int, float] = {}
            for pos_i, atom_i in enumerate(atom_indices):
                for atom_j in atom_indices[pos_i:]:
                    distance = int(distances[atom_i, atom_j])
                    if distance < 0 or distance > int(max_distance):
                        skipped_pair_count += 1
                        continue
                    left = env_tokens[atom_i]
                    right = env_tokens[atom_j]
                    if right < left:
                        left, right = (right, left)
                    token = f'map4r{env_radius}d{distance}:{left}::{right}'
                    col = feature_token_index(token, n_features)
                    local_counts[col] = local_counts.get(col, 0.0) + 1.0
                    token_count += 1
            for col, count in local_counts.items():
                rows.append(row)
                cols.append(col)
                data.append(float(math.log1p(count)))
                pass
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
        status_counts[status] = status_counts.get(status, 0) + 1
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), int(n_features)), dtype=np.float32)
    if matrix.nnz:
        matrix.sum_duplicates()
        pass
    return (matrix, {'source': 'official train/test SMILES only; dependency-free MAP4-like hashed atom-environment pair fingerprint', 'n_features': int(n_features), 'max_distance': int(max_distance), 'env_radius': int(env_radius), 'nnz': int(matrix.nnz), 'token_count': int(token_count), 'skipped_pair_count': int(skipped_pair_count), 'status_counts': status_counts})

def endpoint_path_ngram_matrix(mols: list[Chem.Mol], *, n_features: int, max_bonds: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    token_count = 0
    status_counts: dict[str, int] = {}
    max_window = max(1, int(max_bonds))
    for row, mol in enumerate(mols):
        status = 'ok'
        try:
            _, endpoint_path = endpoint_neighbors_and_path(mol)
            if len(endpoint_path) < 2:
                raise RuntimeError('endpoint_path_missing')
            atom_tokens = [atom_category(mol.GetAtomWithIdx(int(index))) for index in endpoint_path]
            bond_tokens: list[str] = []
            for left, right in zip(endpoint_path[:-1], endpoint_path[1:], strict=True):
                bond = mol.GetBondBetweenAtoms(int(left), int(right))
                if bond is None:
                    raise RuntimeError('endpoint_path_bond_missing')
                bond_tokens.append(bond_category(bond))
            local_counts: dict[int, float] = {}
            path_bonds = len(bond_tokens)
            for width in range(1, min(max_window, path_bonds) + 1):
                for start in range(0, path_bonds - width + 1):
                    local_atoms = atom_tokens[start:start + width + 1]
                    local_bonds = bond_tokens[start:start + width]
                    forward_parts: list[str] = []
                    reverse_parts: list[str] = []
                    for pos, atom_token in enumerate(local_atoms):
                        forward_parts.append(atom_token)
                        if pos < len(local_bonds):
                            forward_parts.append(local_bonds[pos])
                            pass
                    rev_atoms = list(reversed(local_atoms))
                    rev_bonds = list(reversed(local_bonds))
                    for pos, atom_token in enumerate(rev_atoms):
                        reverse_parts.append(atom_token)
                        if pos < len(rev_bonds):
                            reverse_parts.append(rev_bonds[pos])
                            pass
                    token = min('|'.join(forward_parts), '|'.join(reverse_parts))
                    col = feature_token_index(f'endpoint_path_w{width}:{token}', n_features)
                    local_counts[col] = local_counts.get(col, 0.0) + 1.0
                    token_count += 1
            whole_token = '|'.join((part for pair in zip(atom_tokens, bond_tokens + [''], strict=True) for part in pair if part))
            reverse_whole = '|'.join((part for pair in zip(list(reversed(atom_tokens)), list(reversed(bond_tokens)) + [''], strict=True) for part in pair if part))
            col = feature_token_index(f'endpoint_path_full:{min(whole_token, reverse_whole)}', n_features)
            local_counts[col] = local_counts.get(col, 0.0) + 1.0
            token_count += 1
            for col, count in local_counts.items():
                rows.append(row)
                cols.append(col)
                data.append(float(math.log1p(count)))
                pass
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
        status_counts[status] = status_counts.get(status, 0) + 1
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), int(n_features)), dtype=np.float32)
    if matrix.nnz:
        matrix.sum_duplicates()
        pass
    return (matrix, {'source': 'official train/test SMILES only; orientation-invariant sparse atom/bond n-grams along the polymer endpoint path', 'n_features': int(n_features), 'max_bonds': int(max_bonds), 'nnz': int(matrix.nnz), 'token_count': int(token_count), 'status_counts': status_counts})

def backbone_sidechain_fragment_mols(mols: list[Chem.Mol]) -> tuple[list[Chem.Mol], list[Chem.Mol], dict[str, Any]]:
    """Return endpoint-backbone and off-backbone fragment molecules per row."""
    empty = Chem.Mol()
    backbone_mols: list[Chem.Mol] = []
    side_mols: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    empty_backbone_rows = 0
    empty_side_rows = 0
    for mol in mols:
        status = 'ok'
        backbone_fragment: Chem.Mol | None = None
        side_fragment: Chem.Mol | None = None
        try:
            dummy_atoms = [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError(f'endpoint_count_{len(dummy_atoms)}')
            path = tuple(Chem.rdmolops.GetShortestPath(mol, int(dummy_atoms[0]), int(dummy_atoms[1])))
            if not path:
                raise RuntimeError('endpoint_path_missing')
            path_set = {int(index) for index in path}
            heavy_atoms = {int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
            backbone_heavy = {idx for idx in path_set if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
            side_heavy = heavy_atoms.difference(path_set)
            backbone_fragment = region_fragment_from_atoms(mol, backbone_heavy)
            side_fragment = region_fragment_from_atoms(mol, side_heavy)
        except Exception as exc:
            status = f'failed_{type(exc).__name__}'
        if backbone_fragment is None or backbone_fragment.GetNumAtoms() == 0:
            backbone_fragment = empty
            empty_backbone_rows += 1
        if side_fragment is None or side_fragment.GetNumAtoms() == 0:
            side_fragment = empty
            empty_side_rows += 1
        backbone_mols.append(backbone_fragment)
        side_mols.append(side_fragment)
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {'source': 'official train/test SMILES only; endpoint shortest-path backbone and off-path side-chain fragments for region-specific sparse kernels', 'status_counts': status_counts, 'empty_backbone_rows': int(empty_backbone_rows), 'empty_side_rows': int(empty_side_rows)}
    return (backbone_mols, side_mols, report)

def cap_polymer_smiles(smiles: str) -> str:
    capped = re.sub('\\[\\*[^\\]]*\\]', 'C', str(smiles))
    return capped.replace('*', 'C')

def cap_polymer_smiles_hydrogen(smiles: str) -> str:
    capped = re.sub('\\[\\*[^\\]]*\\]', '[H]', str(smiles))
    return capped.replace('*', '[H]')

def capped_descriptor_mols(smiles: list[str], fallback_mols: list[Chem.Mol]) -> tuple[list[Chem.Mol], dict[str, int]]:
    mols: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = 'hydrogen_cap'
        mol = Chem.MolFromSmiles(cap_polymer_smiles_hydrogen(smi), sanitize=True)
        if mol is None:
            mol = Chem.MolFromSmiles(cap_polymer_smiles(smi), sanitize=True)
            status = 'carbon_cap_fallback' if mol is not None else 'original_fallback'
        if mol is None:
            mol = fallback
        mols.append(mol)
        status_counts[status] = status_counts.get(status, 0) + 1
    return (mols, status_counts)

def periodic_closure_mol(smiles: str, fallback_mol: Chem.Mol) -> Chem.Mol:
    """Close the two polymer attachment endpoints into one repeat-cycle graph.

    This uses only the official repeat-unit SMILES. If closure is chemically
    invalid for a row, the original molecule is retained so the feature block
    remains total and deterministic.
    """
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return fallback_mol
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 2:
        return fallback_mol
    endpoints: list[tuple[int, int, Chem.BondType]] = []
    for atom in dummy_atoms:
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            return fallback_mol
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
        if bond is None:
            return fallback_mol
        endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
    first_neighbor = endpoints[0][1]
    second_neighbor = endpoints[1][1]
    if first_neighbor == second_neighbor:
        return fallback_mol
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(first_neighbor, second_neighbor) is None:
        closure_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE
        try:
            rw.AddBond(first_neighbor, second_neighbor, closure_type)
            pass
        except Exception:
            return fallback_mol
    for dummy_idx in sorted((endpoints[0][0], endpoints[1][0]), reverse=True):
        rw.RemoveAtom(dummy_idx)
        pass
    try:
        closed = rw.GetMol()
        Chem.SanitizeMol(closed)
        return closed
    except Exception:
        return fallback_mol

def oligomer_mols(smiles: list[str], fallback_mols: list[Chem.Mol], repeats: int) -> tuple[list[Chem.Mol], dict[str, Any]]:
    if repeats < 2:
        raise ValueError('oligomer repeats must be at least 2')
    out: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = f'{repeats}mer'
        try:
            mol = Chem.MolFromSmiles(str(smi), sanitize=True)
            if mol is None:
                raise RuntimeError('parse_failed')
            dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError('endpoint_count')
            endpoints: list[tuple[int, int, Chem.BondType]] = []
            for atom in dummy_atoms:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1:
                    raise RuntimeError('endpoint_degree')
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                if bond is None:
                    raise RuntimeError('endpoint_bond')
                endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
            endpoints.sort(key=lambda item: item[0])
            remove_indices = sorted((endpoints[0][0], endpoints[1][0]))
            rw_core = Chem.RWMol(mol)
            for dummy_idx in reversed(remove_indices):
                rw_core.RemoveAtom(dummy_idx)
                pass
            core = rw_core.GetMol()
            Chem.SanitizeMol(core)
            if core.GetNumAtoms() <= 1:
                raise RuntimeError('empty_core')

            def remap(old_idx: int) -> int:
                return old_idx - sum((1 for dummy_idx in remove_indices if dummy_idx < old_idx))
            left_idx = remap(endpoints[0][1])
            right_idx = remap(endpoints[1][1])
            if left_idx == right_idx:
                raise RuntimeError('same_endpoint_neighbor')
            bond_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE
            combined = core
            for _ in range(repeats - 1):
                combined = Chem.CombineMols(combined, core)
            rw = Chem.RWMol(combined)
            n_core = core.GetNumAtoms()
            for repeat_idx in range(repeats - 1):
                a_idx = repeat_idx * n_core + right_idx
                b_idx = (repeat_idx + 1) * n_core + left_idx
                if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
                    rw.AddBond(a_idx, b_idx, bond_type)
                    pass
            candidate = rw.GetMol()
            Chem.SanitizeMol(candidate)
            out.append(candidate)
        except Exception as exc:
            status = f'fallback_{type(exc).__name__}'
            out.append(fallback)
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {'source': 'official train/test SMILES only; deterministic linear oligomer built by removing two dummy endpoints and joining repeat cores', 'repeats': int(repeats), 'status_counts': status_counts}
    return (out, report)

def oligomer_mols_with_row_status(smiles: list[str], fallback_mols: list[Chem.Mol], repeats: int) -> tuple[list[Chem.Mol], list[str], dict[str, Any]]:
    if repeats < 2:
        raise ValueError('oligomer repeats must be at least 2')
    out: list[Chem.Mol] = []
    row_statuses: list[str] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = f'{repeats}mer'
        try:
            mol = Chem.MolFromSmiles(str(smi), sanitize=True)
            if mol is None:
                raise RuntimeError('parse_failed')
            dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError('endpoint_count')
            endpoints: list[tuple[int, int, Chem.BondType]] = []
            for atom in dummy_atoms:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1:
                    raise RuntimeError('endpoint_degree')
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                if bond is None:
                    raise RuntimeError('endpoint_bond')
                endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
            endpoints.sort(key=lambda item: item[0])
            remove_indices = sorted((endpoints[0][0], endpoints[1][0]))
            rw_core = Chem.RWMol(mol)
            for dummy_idx in reversed(remove_indices):
                rw_core.RemoveAtom(dummy_idx)
                pass
            core = rw_core.GetMol()
            Chem.SanitizeMol(core)
            if core.GetNumAtoms() <= 1:
                raise RuntimeError('empty_core')

            def remap(old_idx: int) -> int:
                return old_idx - sum((1 for dummy_idx in remove_indices if dummy_idx < old_idx))
            left_idx = remap(endpoints[0][1])
            right_idx = remap(endpoints[1][1])
            if left_idx == right_idx:
                raise RuntimeError('same_endpoint_neighbor')
            bond_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE
            combined = core
            for _ in range(repeats - 1):
                combined = Chem.CombineMols(combined, core)
            rw = Chem.RWMol(combined)
            n_core = core.GetNumAtoms()
            for repeat_idx in range(repeats - 1):
                a_idx = repeat_idx * n_core + right_idx
                b_idx = (repeat_idx + 1) * n_core + left_idx
                if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
                    rw.AddBond(a_idx, b_idx, bond_type)
                    pass
            candidate = rw.GetMol()
            Chem.SanitizeMol(candidate)
            out.append(candidate)
        except Exception as exc:
            status = f'fallback_{type(exc).__name__}'
            out.append(fallback)
        row_statuses.append(status)
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {'source': 'official train/test SMILES only; deterministic linear oligomer with row-level construction status', 'repeats': int(repeats), 'status_counts': status_counts}
    return (out, row_statuses, report)

def oligomer_3d_descriptor_matrix(smiles: list[str], mols: list[Chem.Mol], *, repeats_values: tuple[int, ...], conformers: int, seed: int, optimize_steps: int, poolings: tuple[str, ...], include_extended: bool) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    descriptor_names = rdkit_3d_descriptor_names(include_extended)
    names: list[str] = []
    for repeats in repeats_values:
        for pooling in poolings:
            names.extend([f'oligomer_{repeats}mer_3d_{pooling}_{name}' for name in descriptor_names])
            pass
    values = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    repeat_reports: dict[str, Any] = {}
    offset = 0
    width = len(descriptor_names) * len(poolings)
    for repeat_index, repeats in enumerate(repeats_values):
        if repeats < 2:
            raise ValueError('oligomer 3D repeats must be at least 2')
        olig_mols, row_statuses, olig_report = oligomer_mols_with_row_status(smiles, mols, repeats)
        descriptor_status_counts: dict[str, int] = {}
        total_embedded = 0
        total_descriptor_conformers = 0
        total_uff_failures = 0
        skipped_construction = 0
        for row, (mol, row_status) in enumerate(zip(olig_mols, row_statuses, strict=True)):
            if row_status != f'{repeats}mer':
                skipped_construction += 1
                continue
            pooled, pooled_report = pooled_3d_descriptor_for_mol(mol, seed=int(seed + 100000 * repeat_index + row), conformers=conformers, optimize_steps=optimize_steps, poolings=poolings, include_extended=include_extended)
            status = str(pooled_report['status'])
            descriptor_status_counts[status] = descriptor_status_counts.get(status, 0) + 1
            total_embedded += int(pooled_report.get('embedded_conformers', 0))
            total_descriptor_conformers += int(pooled_report.get('descriptor_conformers', 0))
            total_uff_failures += int(pooled_report.get('uff_failures', 0))
            values[row, offset:offset + width] = pooled
        repeat_reports[f'{repeats}mer'] = olig_report | {'skipped_rows_due_to_oligomer_construction_status': int(skipped_construction), 'descriptor_status_counts': descriptor_status_counts, 'total_embedded_conformers': int(total_embedded), 'total_descriptor_conformers': int(total_descriptor_conformers), 'total_uff_failures': int(total_uff_failures)}
        offset += width
    report = {'source': 'official train/test SMILES only; deterministic oligomer ETKDG 3D conformer descriptors with fold-local downstream imputation', 'repeats': [int(item) for item in repeats_values], 'conformers_per_mol': int(conformers), 'optimize_steps': int(optimize_steps), 'poolings': list(poolings), 'include_extended_descriptors': bool(include_extended), 'base_descriptor_count': int(len(descriptor_names)), 'output_feature_count': int(values.shape[1]), 'nonfinite_output_values': int(np.size(values) - np.isfinite(values).sum()), 'repeat_reports': repeat_reports}
    return (values, names, report)

def oligomer_slope_descriptor_matrix(smiles: list[str], mols: list[Chem.Mol], *, max_repeats: int, include_physics: bool, transform: str='raw') -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if max_repeats < 2:
        raise ValueError('oligomer slope max repeats must be at least 2')
    if transform not in {'raw', 'signed_log', 'both'}:
        raise ValueError('oligomer slope transform must be raw, signed_log, or both')
    repeat_values = np.arange(1, max_repeats + 1, dtype=np.float64)
    matrices: list[np.ndarray] = []
    repeat_reports: dict[str, Any] = {}
    base_dense, base_names = descriptor_matrix(mols, smiles)
    dense_names = list(base_names)
    if include_physics:
        base_physics, physics_names = physics_feature_matrix(mols)
        base_dense = np.hstack([base_dense, base_physics])
        dense_names = dense_names + physics_names
    matrices.append(base_dense)
    repeat_reports['1mer'] = {'status_counts': {'original_with_dummy_endpoints': len(mols)}}
    for repeats in range(2, max_repeats + 1):
        repeat_mols, repeat_report = oligomer_mols(smiles, mols, repeats=repeats)
        repeat_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in repeat_mols]
        repeat_dense, repeat_names = descriptor_matrix(repeat_mols, repeat_smiles)
        current_names = list(repeat_names)
        if include_physics:
            repeat_physics, physics_names = physics_feature_matrix(repeat_mols)
            repeat_dense = np.hstack([repeat_dense, repeat_physics])
            current_names = current_names + physics_names
        if current_names != dense_names:
            raise RuntimeError(f'oligomer descriptor schema changed at repeats={repeats}')
        matrices.append(repeat_dense)
        repeat_reports[f'{repeats}mer'] = repeat_report
    stack = np.stack(matrices, axis=0).astype(np.float64, copy=False)
    finite = np.isfinite(stack)
    y = np.where(finite, stack, 0.0)
    x = repeat_values[:, None, None]
    count = finite.sum(axis=0).astype(np.float64)
    sx = (finite * x).sum(axis=0)
    sxx = (finite * x * x).sum(axis=0)
    sy = y.sum(axis=0)
    sxy = (y * x).sum(axis=0)
    denom = count * sxx - sx * sx
    valid = (count >= 2.0) & (np.abs(denom) > 1e-12)
    slope = np.full_like(sy, np.nan, dtype=np.float64)
    intercept = np.full_like(sy, np.nan, dtype=np.float64)
    slope[valid] = (count[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / count[valid]
    raw_out = np.hstack([intercept, slope])
    raw_names = [f'oligomer_linear_intercept_{name}' for name in dense_names] + [f'oligomer_linear_slope_{name}' for name in dense_names]
    signed_log_out = np.sign(raw_out) * np.log1p(np.abs(raw_out))
    signed_log_names = [f'oligomer_signedlog_{name}' for name in raw_names]
    if transform == 'raw':
        out = raw_out
        names = raw_names
    elif transform == 'signed_log':
        out = signed_log_out
        names = signed_log_names
    else:
        out = np.hstack([raw_out, signed_log_out])
        names = raw_names + signed_log_names
    report = {'source': 'official train/test SMILES only; deterministic 1..N oligomer descriptor linear fit over repeat count', 'max_repeats': int(max_repeats), 'include_physics': bool(include_physics), 'transform': transform, 'base_descriptor_count': int(len(dense_names)), 'raw_output_feature_count': int(raw_out.shape[1]), 'output_feature_count': int(out.shape[1]), 'nonfinite_output_values': int(np.size(out) - np.isfinite(out).sum()), 'raw_nonfinite_output_values': int(np.size(raw_out) - np.isfinite(raw_out).sum()), 'repeat_reports': repeat_reports}
    return (out, names, report)

def oligomer_repeat_mols(smiles: list[str], fallback_mols: list[Chem.Mol], repeats: int) -> tuple[list[Chem.Mol], dict[str, Any]]:
    if repeats < 1:
        raise ValueError('oligomer repeat count must be at least 1')
    out: list[Chem.Mol] = []
    status_counts: dict[str, int] = {}
    for smi, fallback in zip(smiles, fallback_mols, strict=True):
        status = f'{repeats}mer'
        try:
            mol = Chem.MolFromSmiles(str(smi), sanitize=True)
            if mol is None:
                raise RuntimeError('parse_failed')
            dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummy_atoms) != 2:
                raise RuntimeError('endpoint_count')
            endpoints: list[tuple[int, int, Chem.BondType]] = []
            for atom in dummy_atoms:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1:
                    raise RuntimeError('endpoint_degree')
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                if bond is None:
                    raise RuntimeError('endpoint_bond')
                endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))
            endpoints.sort(key=lambda item: item[0])
            remove_indices = sorted((endpoints[0][0], endpoints[1][0]))
            rw_core = Chem.RWMol(mol)
            for dummy_idx in reversed(remove_indices):
                rw_core.RemoveAtom(dummy_idx)
                pass
            core = rw_core.GetMol()
            Chem.SanitizeMol(core)
            if core.GetNumAtoms() <= 1:
                raise RuntimeError('empty_core')

            def remap(old_idx: int) -> int:
                return old_idx - sum((1 for dummy_idx in remove_indices if dummy_idx < old_idx))
            left_idx = remap(endpoints[0][1])
            right_idx = remap(endpoints[1][1])
            if left_idx == right_idx:
                raise RuntimeError('same_endpoint_neighbor')
            bond_type = endpoints[0][2] if endpoints[0][2] == endpoints[1][2] else Chem.BondType.SINGLE
            combined = core
            for _ in range(repeats - 1):
                combined = Chem.CombineMols(combined, core)
            rw = Chem.RWMol(combined)
            n_core = core.GetNumAtoms()
            for repeat_idx in range(repeats - 1):
                a_idx = repeat_idx * n_core + right_idx
                b_idx = (repeat_idx + 1) * n_core + left_idx
                if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
                    rw.AddBond(a_idx, b_idx, bond_type)
                    pass
            candidate = rw.GetMol()
            Chem.SanitizeMol(candidate)
            out.append(candidate)
        except Exception as exc:
            status = f'fallback_{type(exc).__name__}'
            out.append(fallback)
        status_counts[status] = status_counts.get(status, 0) + 1
    return (out, {'source': 'official train/test SMILES only; deterministic endpoint-stripped n-mer repeat core for Flory-Fox-style asymptotic descriptors', 'repeats': int(repeats), 'status_counts': status_counts})

def heavy_atom_counts(mols: list[Chem.Mol]) -> np.ndarray:
    counts = np.asarray([max(sum((1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)), 1) for mol in mols], dtype=np.float64)
    counts[~np.isfinite(counts)] = 1.0
    counts[counts <= 0] = 1.0
    return counts

def oligomer_ffox_base_descriptor_matrix(mols: list[Chem.Mol], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    values = np.full((len(mols), len(FFOX_DESCRIPTOR_NAMES)), np.nan, dtype=np.float64)
    for row, mol in enumerate(mols):
        for col, name in enumerate(FFOX_DESCRIPTOR_NAMES):
            try:
                val = float(getattr(Descriptors, name)(mol))
            except Exception:
                val = math.nan
            values[row, col] = val if math.isfinite(val) else math.nan
    extra_names = ['smiles_len', 'atom_count', 'heavy_atom_count', 'ring_count', 'aromatic_atom_count', 'hetero_atom_count', 'halogen_count', 'n_count', 'o_count', 's_count', 'si_count', 'f_count', 'cl_count', 'br_count', 'double_bond_count', 'triple_bond_count', 'conjugated_bond_count', 'aromatic_bond_count', 'branch_count', 'bracket_count']
    extra = np.zeros((len(mols), len(extra_names)), dtype=np.float64)
    for row, (mol, smi) in enumerate(zip(mols, smiles, strict=True)):
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        extra[row, 0] = len(str(smi))
        extra[row, 1] = len(atoms)
        extra[row, 2] = sum((1 for atom in atoms if atom.GetAtomicNum() > 1))
        extra[row, 3] = mol.GetRingInfo().NumRings()
        extra[row, 4] = sum((1 for atom in atoms if atom.GetIsAromatic()))
        extra[row, 5] = sum((1 for atom in atoms if atom.GetAtomicNum() not in (0, 1, 6)))
        extra[row, 6] = sum((1 for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53)))
        for col, atomic_num in ((7, 7), (8, 8), (9, 16), (10, 14), (11, 9), (12, 17), (13, 35)):
            extra[row, col] = sum((1 for atom in atoms if atom.GetAtomicNum() == atomic_num))
        extra[row, 14] = sum((1 for bond in bonds if str(bond.GetBondType()) == 'DOUBLE'))
        extra[row, 15] = sum((1 for bond in bonds if str(bond.GetBondType()) == 'TRIPLE'))
        extra[row, 16] = sum((1 for bond in bonds if bond.GetIsConjugated()))
        extra[row, 17] = sum((1 for bond in bonds if bond.GetIsAromatic()))
        extra[row, 18] = str(smi).count('(')
        extra[row, 19] = str(smi).count('[')
    return (np.hstack([values, extra]), FFOX_DESCRIPTOR_NAMES + extra_names)

def oligomer_ffox_descriptor_matrix(smiles: list[str], mols: list[Chem.Mol], *, max_repeats: int, include_physics: bool, transform: str='raw') -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if max_repeats < 3:
        raise ValueError('Flory-Fox asymptotic descriptors require monomer/dimer/trimer, so max_repeats must be >= 3')
    if transform not in {'raw', 'signed_log', 'both'}:
        raise ValueError('oligomer Flory-Fox transform must be raw, signed_log, or both')
    repeat_values = np.arange(1, max_repeats + 1, dtype=np.float64)
    x_values = 1.0 / repeat_values
    matrices: list[np.ndarray] = []
    repeat_reports: dict[str, Any] = {}
    dense_names: list[str] | None = None
    for repeats in range(1, max_repeats + 1):
        repeat_mols, repeat_report = oligomer_repeat_mols(smiles, mols, repeats=repeats)
        repeat_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in repeat_mols]
        repeat_dense, repeat_names = oligomer_ffox_base_descriptor_matrix(repeat_mols, repeat_smiles)
        current_names = list(repeat_names)
        if include_physics:
            repeat_physics, physics_names = physics_feature_matrix(repeat_mols)
            repeat_dense = np.hstack([repeat_dense, repeat_physics])
            current_names = current_names + physics_names
        if dense_names is None:
            dense_names = current_names
        elif current_names != dense_names:
            raise RuntimeError(f'oligomer Flory-Fox descriptor schema changed at repeats={repeats}')
        scale = heavy_atom_counts(repeat_mols)[:, None]
        matrices.append(repeat_dense.astype(np.float64, copy=False) / scale)
        repeat_reports[f'{repeats}mer'] = repeat_report | {'heavy_atom_count_min': float(np.min(scale)), 'heavy_atom_count_median': float(np.median(scale)), 'heavy_atom_count_max': float(np.max(scale))}
    if dense_names is None:
        raise RuntimeError('no Flory-Fox descriptor schema was built')
    stack = np.stack(matrices, axis=0).astype(np.float64, copy=False)
    finite = np.isfinite(stack)
    y = np.where(finite, stack, 0.0)
    x = x_values[:, None, None]
    count = finite.sum(axis=0).astype(np.float64)
    sx = (finite * x).sum(axis=0)
    sxx = (finite * x * x).sum(axis=0)
    sy = y.sum(axis=0)
    sxy = (y * x).sum(axis=0)
    denom = count * sxx - sx * sx
    valid = (count >= 2.0) & (np.abs(denom) > 1e-12)
    slope = np.full_like(sy, np.nan, dtype=np.float64)
    intercept = np.full_like(sy, np.nan, dtype=np.float64)
    slope[valid] = (count[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / count[valid]
    inf3 = np.full_like(intercept, np.nan, dtype=np.float64)
    m = stack[0]
    d = stack[1]
    t = stack[2]
    inf3_raw = (6.0 * t - 2.0 * d - m) / 3.0
    inf3[np.isfinite(inf3_raw)] = inf3_raw[np.isfinite(inf3_raw)]
    raw_out = np.hstack([intercept, slope, inf3])
    raw_names = [f'oligomer_ffox_inf_per_heavy_{name}' for name in dense_names] + [f'oligomer_ffox_k_per_heavy_{name}' for name in dense_names] + [f'oligomer_ffox_inf3_formula_per_heavy_{name}' for name in dense_names]
    raw_out[~np.isfinite(raw_out)] = np.nan
    signed_log_out = np.sign(raw_out) * np.log1p(np.abs(raw_out))
    signed_log_names = [f'oligomer_ffox_signedlog_{name}' for name in raw_names]
    if transform == 'raw':
        out = raw_out
        names = raw_names
    elif transform == 'signed_log':
        out = signed_log_out
        names = signed_log_names
    else:
        out = np.hstack([raw_out, signed_log_out])
        names = raw_names + signed_log_names
    report = {'source': 'official train/test SMILES only; Flory-Fox-style n-mer descriptors normalized per heavy atom and linearly extrapolated against 1/n', 'max_repeats': int(max_repeats), 'repeat_values': [int(item) for item in repeat_values], 'fit_x': '1/n', 'normalization': 'descriptor_value / heavy_atom_count before extrapolation', 'include_physics': bool(include_physics), 'transform': transform, 'base_descriptor_count': int(len(dense_names)), 'raw_output_feature_count': int(raw_out.shape[1]), 'output_feature_count': int(out.shape[1]), 'nonfinite_output_values': int(np.size(out) - np.isfinite(out).sum()), 'raw_nonfinite_output_values': int(np.size(raw_out) - np.isfinite(raw_out).sum()), 'repeat_reports': repeat_reports}
    return (out, names, report)

def sparse_fingerprint(mols: list[Chem.Mol], *, fp_type: str, radius: int, n_bits: int, kind: str, log_counts: bool) -> sparse.csr_matrix:
    if fp_type == 'morgan':
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, countSimulation=False, includeChirality=False, useBondTypes=True, onlyNonzeroInvariants=False, includeRingMembership=True, includeRedundantEnvironments=False)
    elif fp_type == 'atom_pair':
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits, includeChirality=False)
    elif fp_type == 'topological_torsion':
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits, includeChirality=False)
    else:
        raise ValueError(fp_type)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    if kind == 'bit':
        for row, mol in enumerate(mols):
            arr = np.zeros((n_bits,), dtype=np.int8)
            ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
            idx = np.flatnonzero(arr)
            rows.extend([row] * len(idx))
            cols.extend(idx.tolist())
            data.extend([1.0] * len(idx))
    elif kind == 'count':
        for row, mol in enumerate(mols):
            for col, value in generator.GetCountFingerprint(mol).GetNonzeroElements().items():
                parsed = float(value)
                rows.append(row)
                cols.append(int(col))
                data.append(math.log1p(parsed) if log_counts else parsed)
    else:
        raise ValueError(kind)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), n_bits), dtype=np.float32)

def exact_morgan_count_dicts(mols: list[Chem.Mol], *, radii: tuple[int, ...], prefix: str, log_counts: bool=True) -> list[dict[str, float]]:
    """Return unfolded Morgan count tokens.

    The returned dictionaries are not vectorized here. DictVectorizer fitting is
    intentionally deferred to the target/fold fit rows so exact-feature
    vocabulary remains train-only.
    """
    generators = [(int(radius), rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), countSimulation=False, includeChirality=False, useBondTypes=True, onlyNonzeroInvariants=False, includeRingMembership=True, includeRedundantEnvironments=False)) for radius in radii]
    rows: list[dict[str, float]] = []
    for mol in mols:
        row: dict[str, float] = {}
        for radius, generator in generators:
            fp = generator.GetSparseCountFingerprint(mol)
            for key, value in fp.GetNonzeroElements().items():
                parsed = float(value)
                row[f'{prefix}_morgan_r{radius}:{int(key)}'] = math.log1p(parsed) if log_counts else parsed
        rows.append(row)
    return rows

def wl_subtree_count_dicts(mols: list[Chem.Mol], *, iterations: int, prefix: str, log_counts: bool=True) -> list[dict[str, float]]:
    """Return Weisfeiler-Lehman subtree count tokens.

    Vectorization is intentionally deferred to the target/fold fit rows. The
    compact labels are stable digests of deterministic atom/bond neighborhood
    signatures, not learned embeddings or pretrained chemistry features.
    """
    depth = max(0, int(iterations))
    rows: list[dict[str, float]] = []
    for mol in mols:
        labels = [atom_category(atom) for atom in mol.GetAtoms()]
        row_counts: dict[str, float] = {}
        for step in range(depth + 1):
            counts: dict[str, float] = {}
            for label in labels:
                token = f'{prefix}_wl{step}:{label}'
                counts[token] = counts.get(token, 0.0) + 1.0
            for token, count in counts.items():
                row_counts[token] = math.log1p(count) if log_counts else count
            if step == depth:
                break
            next_labels: list[str] = []
            for atom in mol.GetAtoms():
                pieces: list[str] = []
                atom_idx = int(atom.GetIdx())
                for bond in atom.GetBonds():
                    begin = int(bond.GetBeginAtomIdx())
                    end = int(bond.GetEndAtomIdx())
                    other = end if begin == atom_idx else begin
                    pieces.append(f'{bond_category(bond)}>{labels[other]}')
                signature = labels[atom_idx] + '|' + '|'.join(sorted(pieces))
            labels = next_labels
        rows.append(row_counts)
    return rows

def morgan_feature_fingerprint(mols: list[Chem.Mol], *, radius: int, n_bits: int, kind: str, log_counts: bool) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    if kind == 'bit':
        for row, mol in enumerate(mols):
            arr = np.zeros((n_bits,), dtype=np.int8)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=True)
            ConvertToNumpyArray(fp, arr)
            idx = np.flatnonzero(arr)
            rows.extend([row] * len(idx))
            cols.extend(idx.tolist())
            data.extend([1.0] * len(idx))
    elif kind == 'count':
        for row, mol in enumerate(mols):
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits, useFeatures=True)
            for col, value in fp.GetNonzeroElements().items():
                parsed = float(value)
                rows.append(row)
                cols.append(int(col))
                data.append(math.log1p(parsed) if log_counts else parsed)
    else:
        raise ValueError(kind)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(mols), n_bits), dtype=np.float32)

def rdk_fingerprint_matrix(mols: list[Chem.Mol], n_bits: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for row, mol in enumerate(mols):
        arr = np.zeros((n_bits,), dtype=np.int8)
        ConvertToNumpyArray(RDKFingerprint(mol, fpSize=n_bits), arr)
        idx = np.flatnonzero(arr)
        rows.extend([row] * len(idx))
        cols.extend(idx.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(mols), n_bits), dtype=np.float32)

def maccs_matrix(mols: list[Chem.Mol]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for row, mol in enumerate(mols):
        arr = np.zeros((167,), dtype=np.int8)
        ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
        idx = np.flatnonzero(arr)
        rows.extend([row] * len(idx))
        cols.extend(idx.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(mols), 167), dtype=np.float32)

def text_matrix(smiles: list[str], n_features: int) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(3, 8), n_features=n_features, alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32)
    return vectorizer.transform(smiles).tocsr()

def rooted_smiles_text_matrix(mols: list[Chem.Mol], n_features: int, max_roots: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(3, 9), n_features=n_features, alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32)
    flat_smiles: list[str] = []
    owners: list[int] = []
    status_counts: dict[str, int] = {}
    root_counts: list[int] = []
    for row, mol in enumerate(mols):
        row_smiles: list[str] = []
        try:
            root_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0]
            if max_roots > 0 and len(root_atoms) > max_roots:
                positions = np.linspace(0, len(root_atoms) - 1, int(max_roots), dtype=int)
                root_atoms = [root_atoms[int(pos)] for pos in sorted(set(positions.tolist()))]
            for root in root_atoms:
                try:
                    row_smiles.append(Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=int(root), isomericSmiles=True))
                    pass
                except Exception:
                    continue
            if not row_smiles:
                row_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)]
            status = 'ok'
        except Exception:
            row_smiles = ['']
            status = 'fallback_blank'
        row_smiles = [value for value in row_smiles if value]
        if not row_smiles:
            row_smiles = ['']
            status = 'fallback_blank'
        status_counts[status] = status_counts.get(status, 0) + 1
        root_counts.append(len(row_smiles))
        flat_smiles.extend(row_smiles)
        owners.extend([row] * len(row_smiles))
    hashed = vectorizer.transform(flat_smiles).tocsr()
    weights = np.asarray([1.0 / max(root_counts[owner], 1) for owner in owners], dtype=np.float32)
    aggregator = sparse.csr_matrix((weights, (owners, np.arange(len(owners)))), shape=(len(mols), len(owners)), dtype=np.float32)
    matrix = (aggregator @ hashed).tocsr()
    report = {'source': 'official train/test SMILES only; deterministic rooted noncanonical SMILES hashed char n-grams averaged per molecule', 'n_features': int(n_features), 'max_roots': int(max_roots), 'flat_smiles': int(len(flat_smiles)), 'root_count_min': int(min(root_counts)) if root_counts else 0, 'root_count_median': float(np.median(root_counts)) if root_counts else 0.0, 'root_count_max': int(max(root_counts)) if root_counts else 0, 'status_counts': status_counts, 'nnz': int(matrix.nnz)}
    return (matrix, report)

def random_smiles_text_matrix(mols: list[Chem.Mol], n_features: int, augmentations: int, seed: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(3, 10), n_features=n_features, alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32)
    flat_smiles: list[str] = []
    owners: list[int] = []
    counts: list[int] = []
    status_counts: dict[str, int] = {}
    per_mol = max(1, int(augmentations))
    base_seed = int(seed)
    for row, mol in enumerate(mols):
        row_smiles: list[str] = []
        status = 'ok'
        try:
            row_smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
            random_values = Chem.MolToRandomSmilesVect(mol, max(0, per_mol - 1), randomSeed=base_seed + row, isomericSmiles=True)
            row_smiles.extend((str(value) for value in random_values if value))
        except Exception:
            row_smiles = ['']
            status = 'fallback_blank'
        deduped = []
        seen: set[str] = set()
        for value in row_smiles:
            if value not in seen:
                deduped.append(value)
                seen.add(value)
                pass
        if not deduped:
            deduped = ['']
            status = 'fallback_blank'
        status_counts[status] = status_counts.get(status, 0) + 1
        counts.append(len(deduped))
        flat_smiles.extend(deduped)
        owners.extend([row] * len(deduped))
    hashed = vectorizer.transform(flat_smiles).tocsr()
    weights = np.asarray([1.0 / max(counts[owner], 1) for owner in owners], dtype=np.float32)
    aggregator = sparse.csr_matrix((weights, (owners, np.arange(len(owners)))), shape=(len(mols), len(owners)), dtype=np.float32)
    matrix = (aggregator @ hashed).tocsr()
    report = {'source': 'official train/test SMILES only; deterministic random noncanonical SMILES hashed char n-grams averaged per molecule', 'n_features': int(n_features), 'augmentations_requested': int(augmentations), 'seed': int(seed), 'flat_smiles': int(len(flat_smiles)), 'variant_count_min': int(min(counts)) if counts else 0, 'variant_count_median': float(np.median(counts)) if counts else 0.0, 'variant_count_max': int(max(counts)) if counts else 0, 'status_counts': status_counts, 'nnz': int(matrix.nnz)}
    return (matrix, report)

def kekule_smiles_text_matrix(mols: list[Chem.Mol], n_features: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(3, 9), n_features=n_features, alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32)
    values: list[str] = []
    status_counts: dict[str, int] = {}
    for mol in mols:
        try:
            working = Chem.Mol(mol)
            Chem.Kekulize(working, clearAromaticFlags=True)
            value = Chem.MolToSmiles(working, canonical=True, isomericSmiles=True, kekuleSmiles=True)
            status = 'ok'
        except Exception:
            try:
                value = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                status = 'fallback_canonical'
            except Exception:
                value = ''
                status = 'fallback_blank'
        values.append(value)
        status_counts[status] = status_counts.get(status, 0) + 1
    matrix = vectorizer.transform(values).tocsr()
    report = {'source': 'official train/test SMILES only; deterministic canonical kekulized SMILES hashed char n-grams', 'n_features': int(n_features), 'status_counts': status_counts, 'nnz': int(matrix.nnz)}
    return (matrix, report)

def build_features(smiles: list[str], n_bits: int, text_features: int, motif_hash_features: int, rich_features: bool, periodic_features: bool, periodic_dense_features: bool, capped_dense_features: bool, motif_features: bool, physics_features: bool, mordred_features: bool, oligomer_features: bool, oligomer_repeats: int, oligomer_slope_features: bool=False, oligomer_slope_max_repeats: int=4, oligomer_slope_transform: str='raw', oligomer_ffox_features: bool=False, oligomer_ffox_max_repeats: int=3, oligomer_ffox_transform: str='raw', oligomer_3d_features: bool=False, oligomer_3d_repeats: str | tuple[int, ...]=(2, 3), conformers_per_mol: int=1, conformer_pooling: str | tuple[str, ...]=('mean', 'std'), oligomer_3d_extended: bool=True, oligomer_mordred_features: bool=False, rdkit_3d_features: bool=False, conformer_seed: int=20260721, conformer_opt_steps: int=0, backbone_sidechain_features: bool=False, conjugation_features: bool=False, mobility_features: bool=False, huckel_features: bool=False, electronic_tail_features: bool=False, topological_autocorr_features: bool=False, topological_autocorr_max_distance: int=8, infinite_chain_features: bool=False, bicerano_features: bool=False, map4_features: bool=False, map4_hash_features: int=131072, map4_max_distance: int=12, map4_env_radius: int=1, region_sparse_features: bool=False, region_sparse_hash_features: int=32768, endpoint_path_sparse_features: bool=False, endpoint_path_hash_features: int=32768, endpoint_path_max_bonds: int=8, rooted_smiles_features: bool=False, rooted_smiles_max_roots: int=16, rooted_smiles_text_features: int | None=None, random_smiles_features: bool=False, random_smiles_augmentations: int=16, random_smiles_seed: int=20260722, random_smiles_text_features: int | None=None, kekule_smiles_features: bool=False, kekule_smiles_text_features: int | None=None, exact_sparse_features: bool=False, exact_sparse_radii: str | tuple[int, ...]=(1, 2, 3), wl_sparse_features: bool=False, wl_iterations: int=3) -> dict[str, Any]:
    mols = build_mols(smiles)
    dense, dense_names = descriptor_matrix(mols, smiles)
    feature_reports: dict[str, Any] = {}
    extra_blocks: dict[str, sparse.csr_matrix] = {}
    exact_blocks: dict[str, list[dict[str, float]]] = {}
    exact_radii = tuple(parse_int_csv(exact_sparse_radii)) if isinstance(exact_sparse_radii, str) else tuple((int(item) for item in exact_sparse_radii))
    if not exact_radii:
        exact_radii = (1, 2, 3)
    if any((radius < 0 for radius in exact_radii)):
        raise ValueError('exact sparse Morgan radii must be nonnegative')
    if capped_dense_features:
        capped_mols, capped_status = capped_descriptor_mols(smiles, mols)
        capped_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in capped_mols]
        capped_dense, capped_names = descriptor_matrix(capped_mols, capped_smiles)
        if physics_features:
            capped_physics_dense, capped_physics_names = physics_feature_matrix(capped_mols)
            capped_dense = np.hstack([capped_dense, capped_physics_dense])
            capped_names = capped_names + capped_physics_names
        dense = np.hstack([dense, capped_dense])
        dense_names = dense_names + [f'capped_h_{name}' for name in capped_names]
        feature_reports['capped_dense'] = {'source': 'official train/test SMILES only; polymer dummy endpoints capped with explicit hydrogens before descriptor calculation', 'status_counts': capped_status, 'descriptor_count': len(capped_names)}
    if motif_features:
        motif_dense, motif_names, motif_dense_report = motif_dense_matrix(mols, smiles)
        dense = np.hstack([dense, motif_dense])
        dense_names = dense_names + motif_names
        feature_reports['motif_dense'] = motif_dense_report
        if int(motif_hash_features) > 0:
            motif_hash, motif_hash_report = motif_hash_matrix(mols, smiles, motif_hash_features)
            extra_blocks['motif_hash_count'] = motif_hash
            feature_reports['motif_hash'] = motif_hash_report
        else:
            feature_reports['motif_hash'] = {'source': 'disabled by motif_hash_features=0; dense motif descriptors still enabled', 'n_features': 0, 'nnz': 0}
    if rooted_smiles_features:
        rooted_n_features = int(rooted_smiles_text_features or text_features)
        rooted_block, rooted_report = rooted_smiles_text_matrix(mols, rooted_n_features, rooted_smiles_max_roots)
        extra_blocks['rooted_smiles_text'] = rooted_block
        feature_reports['rooted_smiles_text'] = rooted_report
    if random_smiles_features:
        random_n_features = int(random_smiles_text_features or text_features)
        random_block, random_report = random_smiles_text_matrix(mols, random_n_features, random_smiles_augmentations, random_smiles_seed)
        extra_blocks['random_smiles_text'] = random_block
        feature_reports['random_smiles_text'] = random_report
    if kekule_smiles_features:
        kekule_n_features = int(kekule_smiles_text_features or text_features)
        kekule_block, kekule_report = kekule_smiles_text_matrix(mols, kekule_n_features)
        extra_blocks['kekule_smiles_text'] = kekule_block
        feature_reports['kekule_smiles_text'] = kekule_report
    if map4_features:
        map4_block, map4_report = map4_like_matrix(mols, n_features=int(map4_hash_features), max_distance=int(map4_max_distance), env_radius=int(map4_env_radius))
        extra_blocks['map4_like_count'] = map4_block
        feature_reports['map4_like'] = map4_report
    if region_sparse_features:
        region_hash_width = int(region_sparse_hash_features)
        if region_hash_width <= 0:
            raise ValueError('region sparse hash feature width must be positive')
        backbone_mols, side_mols, region_report = backbone_sidechain_fragment_mols(mols)
        extra_blocks.update({'region_bb_morgan_count_r2': sparse_fingerprint(backbone_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='count', log_counts=True), 'region_side_morgan_count_r2': sparse_fingerprint(side_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='count', log_counts=True), 'region_bb_fcfp_count_r2': morgan_feature_fingerprint(backbone_mols, radius=2, n_bits=n_bits, kind='count', log_counts=True), 'region_side_fcfp_count_r2': morgan_feature_fingerprint(side_mols, radius=2, n_bits=n_bits, kind='count', log_counts=True), 'region_bb_morgan_bit_r2': sparse_fingerprint(backbone_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='bit', log_counts=False), 'region_side_morgan_bit_r2': sparse_fingerprint(side_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='bit', log_counts=False), 'region_bb_rdk_bit': rdk_fingerprint_matrix(backbone_mols, n_bits=n_bits), 'region_side_rdk_bit': rdk_fingerprint_matrix(side_mols, n_bits=n_bits)})
        bb_map4, bb_map4_report = map4_like_matrix(backbone_mols, n_features=region_hash_width, max_distance=int(map4_max_distance), env_radius=int(map4_env_radius))
        side_map4, side_map4_report = map4_like_matrix(side_mols, n_features=region_hash_width, max_distance=int(map4_max_distance), env_radius=int(map4_env_radius))
        extra_blocks['region_bb_map4_like_count'] = bb_map4
        extra_blocks['region_side_map4_like_count'] = side_map4
        if exact_sparse_features:
            exact_blocks['exact_region_bb_morgan_count'] = exact_morgan_count_dicts(backbone_mols, radii=exact_radii, prefix='exact_region_bb')
            exact_blocks['exact_region_side_morgan_count'] = exact_morgan_count_dicts(side_mols, radii=exact_radii, prefix='exact_region_side')
        if wl_sparse_features:
            exact_blocks['wl_region_bb_subtree'] = wl_subtree_count_dicts(backbone_mols, iterations=wl_iterations, prefix='region_bb')
            exact_blocks['wl_region_side_subtree'] = wl_subtree_count_dicts(side_mols, iterations=wl_iterations, prefix='region_side')
        feature_reports['region_sparse'] = region_report | {'hash_width': region_hash_width, 'blocks': sorted((name for name in extra_blocks if name.startswith('region_'))), 'exact_blocks_enabled': bool(exact_sparse_features), 'wl_blocks_enabled': bool(wl_sparse_features), 'bb_map4': bb_map4_report, 'side_map4': side_map4_report}
    if endpoint_path_sparse_features:
        endpoint_path_hash_width = int(endpoint_path_hash_features)
        if endpoint_path_hash_width <= 0:
            raise ValueError('endpoint path hash feature width must be positive')
        endpoint_path_block, endpoint_path_report = endpoint_path_ngram_matrix(mols, n_features=endpoint_path_hash_width, max_bonds=int(endpoint_path_max_bonds))
        extra_blocks['endpoint_path_ngram_count'] = endpoint_path_block
        feature_reports['endpoint_path_sparse'] = endpoint_path_report
    if backbone_sidechain_features:
        bb_side_dense, bb_side_names, bb_side_report = backbone_sidechain_matrix(mols)
        dense = np.hstack([dense, bb_side_dense])
        dense_names = dense_names + bb_side_names
        feature_reports['backbone_sidechain'] = bb_side_report
    if conjugation_features:
        conjugation_dense, conjugation_names, conjugation_report = conjugation_feature_matrix(mols)
        dense = np.hstack([dense, conjugation_dense])
        dense_names = dense_names + conjugation_names
        feature_reports['conjugation'] = conjugation_report
    if mobility_features:
        mobility_dense, mobility_names, mobility_report = mobility_feature_matrix(mols)
        dense = np.hstack([dense, mobility_dense])
        dense_names = dense_names + mobility_names
        feature_reports['mobility'] = mobility_report
    if huckel_features:
        huckel_dense, huckel_names, huckel_report = huckel_spectrum_feature_matrix(mols)
        dense = np.hstack([dense, huckel_dense])
        dense_names = dense_names + huckel_names
        feature_reports['huckel'] = huckel_report
    if electronic_tail_features:
        electronic_dense, electronic_names, electronic_report = electronic_tail_feature_matrix(mols)
        dense = np.hstack([dense, electronic_dense])
        dense_names = dense_names + electronic_names
        feature_reports['electronic_tail'] = electronic_report
    if topological_autocorr_features:
        autocorr_dense, autocorr_names, autocorr_report = topological_autocorr_feature_matrix(mols, max_distance=topological_autocorr_max_distance)
        dense = np.hstack([dense, autocorr_dense])
        dense_names = dense_names + autocorr_names
        feature_reports['topological_autocorr'] = autocorr_report
    if infinite_chain_features:
        infinite_dense, infinite_names, infinite_report = infinite_chain_proxy_feature_matrix(smiles, mols)
        dense = np.hstack([dense, infinite_dense])
        dense_names = dense_names + infinite_names
        feature_reports['infinite_chain_proxy'] = infinite_report
    if bicerano_features:
        bicerano_dense, bicerano_names, bicerano_report = bicerano_feature_matrix(smiles)
        dense = np.hstack([dense, bicerano_dense])
        dense_names = dense_names + bicerano_names
        feature_reports['bicerano'] = bicerano_report
    if exact_sparse_features:
        exact_blocks['exact_morgan_count'] = exact_morgan_count_dicts(mols, radii=exact_radii, prefix='exact_raw')
        feature_reports['exact_sparse'] = {'source': 'official SMILES only; unfolded Morgan count dictionaries are vectorized only on target/fold official train rows', 'radii': [int(radius) for radius in exact_radii], 'blocks': sorted(exact_blocks)}
    if wl_sparse_features:
        exact_blocks['wl_subtree'] = wl_subtree_count_dicts(mols, iterations=wl_iterations, prefix='raw')
        feature_reports['wl_sparse'] = {'source': 'official SMILES only; WL subtree count dictionaries are vectorized only on target/fold official train rows', 'iterations': int(wl_iterations), 'blocks': sorted([name for name in exact_blocks if name.startswith('wl_')]), 'label_digest': 'blake2b-96bit deterministic signature compaction; no learned or pretrained labels'}
    if physics_features:
        physics_dense, physics_names = physics_feature_matrix(mols)
        dense = np.hstack([dense, physics_dense])
        dense_names = dense_names + physics_names
    if mordred_features:
        mordred_dense, mordred_names = mordred_descriptor_matrix(mols)
        dense = np.hstack([dense, mordred_dense])
        dense_names = dense_names + mordred_names
    if oligomer_features:
        olig_mols, olig_report = oligomer_mols(smiles, mols, repeats=oligomer_repeats)
        olig_smiles = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in olig_mols]
        olig_dense, olig_names = descriptor_matrix(olig_mols, olig_smiles)
        if physics_features:
            olig_physics_dense, olig_physics_names = physics_feature_matrix(olig_mols)
            olig_dense = np.hstack([olig_dense, olig_physics_dense])
            olig_names = olig_names + olig_physics_names
        if oligomer_mordred_features:
            olig_mordred_dense, olig_mordred_names = mordred_descriptor_matrix(olig_mols)
            olig_dense = np.hstack([olig_dense, olig_mordred_dense])
            olig_names = olig_names + olig_mordred_names
        prefix = f'oligomer_{oligomer_repeats}mer'
        dense = np.hstack([dense, olig_dense])
        dense_names = dense_names + [f'{prefix}_{name}' for name in olig_names]
        extra_blocks.update({f'{prefix}_maccs_bit': maccs_matrix(olig_mols), f'{prefix}_morgan_count_r2': sparse_fingerprint(olig_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='count', log_counts=True), f'{prefix}_morgan_count_r3': sparse_fingerprint(olig_mols, fp_type='morgan', radius=3, n_bits=n_bits, kind='count', log_counts=True), f'{prefix}_morgan_bit_r2': sparse_fingerprint(olig_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='bit', log_counts=False), f'{prefix}_morgan_bit_r3': sparse_fingerprint(olig_mols, fp_type='morgan', radius=3, n_bits=n_bits, kind='bit', log_counts=False), f'{prefix}_fcfp_count_r2': morgan_feature_fingerprint(olig_mols, radius=2, n_bits=n_bits, kind='count', log_counts=True), f'{prefix}_fcfp_bit_r2': morgan_feature_fingerprint(olig_mols, radius=2, n_bits=n_bits, kind='bit', log_counts=False), f'{prefix}_rdk_bit': rdk_fingerprint_matrix(olig_mols, n_bits=n_bits)})
        olig_report['dense_descriptor_count'] = len(olig_names)
        olig_report['mordred_enabled'] = bool(oligomer_mordred_features)
        feature_reports['oligomer'] = olig_report
    if oligomer_slope_features:
        olig_slope_dense, olig_slope_names, olig_slope_report = oligomer_slope_descriptor_matrix(smiles, mols, max_repeats=oligomer_slope_max_repeats, include_physics=physics_features, transform=oligomer_slope_transform)
        dense = np.hstack([dense, olig_slope_dense])
        dense_names = dense_names + olig_slope_names
        feature_reports['oligomer_slope'] = olig_slope_report
    if oligomer_ffox_features:
        olig_ffox_dense, olig_ffox_names, olig_ffox_report = oligomer_ffox_descriptor_matrix(smiles, mols, max_repeats=oligomer_ffox_max_repeats, include_physics=physics_features, transform=oligomer_ffox_transform)
        dense = np.hstack([dense, olig_ffox_dense])
        dense_names = dense_names + olig_ffox_names
        feature_reports['oligomer_ffox'] = olig_ffox_report
    if oligomer_3d_features:
        parsed_repeats = parse_int_csv(oligomer_3d_repeats)
        parsed_pooling = parse_token_csv(conformer_pooling)
        olig_3d_dense, olig_3d_names, olig_3d_report = oligomer_3d_descriptor_matrix(smiles, mols, repeats_values=parsed_repeats, conformers=conformers_per_mol, seed=conformer_seed, optimize_steps=conformer_opt_steps, poolings=parsed_pooling, include_extended=oligomer_3d_extended)
        dense = np.hstack([dense, olig_3d_dense])
        dense_names = dense_names + olig_3d_names
        feature_reports['oligomer_3d'] = olig_3d_report
    if rdkit_3d_features:
        rdkit3d_dense, rdkit3d_names, rdkit3d_report = rdkit_3d_descriptor_matrix(smiles, seed=conformer_seed, optimize_steps=conformer_opt_steps)
        dense = np.hstack([dense, rdkit3d_dense])
        dense_names = dense_names + rdkit3d_names
        feature_reports['rdkit_3d'] = rdkit3d_report
    blocks = {'maccs_bit': maccs_matrix(mols), 'morgan_count_r1': sparse_fingerprint(mols, fp_type='morgan', radius=1, n_bits=n_bits, kind='count', log_counts=True), 'morgan_count_r2': sparse_fingerprint(mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='count', log_counts=True), 'morgan_count_r3': sparse_fingerprint(mols, fp_type='morgan', radius=3, n_bits=n_bits, kind='count', log_counts=True), 'morgan_bit_r1': sparse_fingerprint(mols, fp_type='morgan', radius=1, n_bits=n_bits, kind='bit', log_counts=False), 'morgan_bit_r2': sparse_fingerprint(mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='bit', log_counts=False), 'morgan_bit_r3': sparse_fingerprint(mols, fp_type='morgan', radius=3, n_bits=n_bits, kind='bit', log_counts=False), 'atom_pair_count': sparse_fingerprint(mols, fp_type='atom_pair', radius=0, n_bits=n_bits, kind='count', log_counts=True), 'topological_torsion_count': sparse_fingerprint(mols, fp_type='topological_torsion', radius=0, n_bits=n_bits, kind='count', log_counts=True), 'char_text': text_matrix(smiles, text_features)}
    blocks.update(extra_blocks)
    if rich_features:
        capped_mols = []
        for smi, original_mol in zip(smiles, mols, strict=True):
            capped = Chem.MolFromSmiles(cap_polymer_smiles(smi), sanitize=True)
            capped_mols.append(capped if capped is not None else original_mol)
        if exact_sparse_features:
            exact_blocks['exact_capped_morgan_count'] = exact_morgan_count_dicts(capped_mols, radii=exact_radii, prefix='exact_capped')
        if wl_sparse_features:
            exact_blocks['wl_capped_subtree'] = wl_subtree_count_dicts(capped_mols, iterations=wl_iterations, prefix='capped')
        blocks.update({'morgan_count_r4': sparse_fingerprint(mols, fp_type='morgan', radius=4, n_bits=n_bits, kind='count', log_counts=True), 'morgan_count_r5': sparse_fingerprint(mols, fp_type='morgan', radius=5, n_bits=n_bits, kind='count', log_counts=True), 'morgan_bit_r4': sparse_fingerprint(mols, fp_type='morgan', radius=4, n_bits=n_bits, kind='bit', log_counts=False), 'morgan_bit_r5': sparse_fingerprint(mols, fp_type='morgan', radius=5, n_bits=n_bits, kind='bit', log_counts=False), 'fcfp_count_r2': morgan_feature_fingerprint(mols, radius=2, n_bits=n_bits, kind='count', log_counts=True), 'fcfp_bit_r2': morgan_feature_fingerprint(mols, radius=2, n_bits=n_bits, kind='bit', log_counts=False), 'rdk_bit': rdk_fingerprint_matrix(mols, n_bits=n_bits), 'capped_morgan_count_r2': sparse_fingerprint(capped_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='count', log_counts=True), 'capped_morgan_bit_r2': sparse_fingerprint(capped_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='bit', log_counts=False)})
    if periodic_features:
        periodic_mols = [periodic_closure_mol(smi, mol) for smi, mol in zip(smiles, mols, strict=True)]
        if exact_sparse_features:
            exact_blocks['exact_periodic_morgan_count'] = exact_morgan_count_dicts(periodic_mols, radii=exact_radii, prefix='exact_periodic')
        if wl_sparse_features:
            exact_blocks['wl_periodic_subtree'] = wl_subtree_count_dicts(periodic_mols, iterations=wl_iterations, prefix='periodic')
        if periodic_dense_features:
            periodic_dense, periodic_dense_names = descriptor_matrix(periodic_mols, smiles)
            if physics_features:
                periodic_physics_dense, periodic_physics_names = physics_feature_matrix(periodic_mols)
                periodic_dense = np.hstack([periodic_dense, periodic_physics_dense])
                periodic_dense_names = periodic_dense_names + periodic_physics_names
            dense = np.hstack([dense, periodic_dense])
            dense_names = dense_names + [f'periodic_{name}' for name in periodic_dense_names]
        blocks.update({'periodic_maccs_bit': maccs_matrix(periodic_mols), 'periodic_morgan_count_r2': sparse_fingerprint(periodic_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='count', log_counts=True), 'periodic_morgan_count_r3': sparse_fingerprint(periodic_mols, fp_type='morgan', radius=3, n_bits=n_bits, kind='count', log_counts=True), 'periodic_morgan_bit_r2': sparse_fingerprint(periodic_mols, fp_type='morgan', radius=2, n_bits=n_bits, kind='bit', log_counts=False), 'periodic_morgan_bit_r3': sparse_fingerprint(periodic_mols, fp_type='morgan', radius=3, n_bits=n_bits, kind='bit', log_counts=False), 'periodic_fcfp_count_r2': morgan_feature_fingerprint(periodic_mols, radius=2, n_bits=n_bits, kind='count', log_counts=True), 'periodic_fcfp_bit_r2': morgan_feature_fingerprint(periodic_mols, radius=2, n_bits=n_bits, kind='bit', log_counts=False), 'periodic_rdk_bit': rdk_fingerprint_matrix(periodic_mols, n_bits=n_bits)})
    if exact_sparse_features and 'exact_sparse' in feature_reports:
        feature_reports['exact_sparse']['blocks'] = sorted(exact_blocks)
        feature_reports['exact_sparse']['nonempty_row_counts'] = {name: int(sum((1 for row in rows if row))) for name, rows in exact_blocks.items()}
    if wl_sparse_features and 'wl_sparse' in feature_reports:
        wl_names = sorted((name for name in exact_blocks if name.startswith('wl_')))
        feature_reports['wl_sparse']['blocks'] = wl_names
        feature_reports['wl_sparse']['nonempty_row_counts'] = {name: int(sum((1 for row in exact_blocks[name] if row))) for name in wl_names}
    return {'dense': dense, 'dense_names': dense_names, 'blocks': blocks, 'exact_blocks': exact_blocks, 'feature_reports': feature_reports}

def read_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


# ===== blend.py =====
def blend_targets(test_df, base_df, blends):
    """blends: dict {target: (weight, source_df)}. Returns (id,target) DataFrame."""
    import numpy as np, pandas as pd
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy(object)
    ids = test_df['id'].to_numpy(int)
    result = base_df['target'].to_numpy(float).copy()
    for target, (weight, source_df) in blends.items():
        mask = target_type == target
        sv = source_df['target'].to_numpy(float)
        result[mask] = (1.0 - weight) * result[mask] + weight * sv[mask]
    return pd.DataFrame({'id': ids, 'target': result})

# ===== splice.py =====
def splice_targets(test_df, base_df, sources):
    """sources: dict {target: source_df}. Returns (id,target) DataFrame."""
    import numpy as np, pandas as pd
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy(object)
    ids = test_df['id'].to_numpy(int)
    result = base_df['target'].to_numpy(float).copy()
    for target, source_df in sources.items():
        mask = target_type == target
        result[mask] = source_df['target'].to_numpy(float)[mask]
    return pd.DataFrame({'id': ids, 'target': result})

# ===== reflected.py =====
def reflected_source(base_df, source_df):
    """reflected = 2*base - source. Returns (id,target) DataFrame."""
    import numpy as np, pandas as pd
    ids = base_df['id'].to_numpy(int)
    reflected = 2.0 * base_df['target'].to_numpy(float) - source_df['target'].to_numpy(float)
    return pd.DataFrame({'id': ids, 'target': reflected})

# ===== c415.py =====
def c415_build_c415(base_df, source_df):
    """reflected = 2*base - source. Returns (id,target) DataFrame."""
    import numpy as np, pandas as pd
    ids = base_df['id'].to_numpy(int)
    reflected = 2.0 * base_df['target'].to_numpy(float) - source_df['target'].to_numpy(float)
    return pd.DataFrame({'id': ids, 'target': reflected})

# ===== c282.py =====
def c282_build_c282(data_dir):
    """C282 current-only reference ensemble. Reads ONLY official train/test.
    Returns (submission_df, oof_df)."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / 'train.csv')
    test = pd.read_csv(data_path / 'test.csv')
    if list(train.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected current train schema')
    if list(test.columns) != ['id', 'smiles', 'target_type']:
        raise RuntimeError('Unexpected current test schema')
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError('Unexpected official current row count')
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [reference.canonicalize(value) for value in frame['smiles']]
    if set(train['target_type']) != set(reference.TARGETS) or set(test['target_type']) != set(reference.TARGETS):
        raise RuntimeError('Unexpected target set')
    if test['id'].duplicated().any() or not np.array_equal(test['id'].to_numpy(), np.arange(1, 4941)):
        raise RuntimeError('Test IDs are not unique sequential IDs 1..4940')
    if not np.isfinite(train['target'].to_numpy(float)).all():
        raise RuntimeError('Current train contains a non-finite target')
    configuration = dict(reference.DEFAULT_CONFIG)
    np.random.seed(int(configuration['seed']))
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled['canonical']) | set(test['canonical']))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [reference.morgan_count_matrix(molecules, radius=2, bits=int(configuration['morgan_bits'])), reference.morgan_count_matrix(molecules, radius=3, bits=int(configuration['morgan_bits'])), reference.text_matrix(keys, int(configuration['text_features']))]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(configuration['morgan_bits']))
    detail, oof, model_report = reference.fit_targets(pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, configuration)
    final_detail, override_report = reference.apply_official_overrides(detail, test, raw_labels)
    submission = final_detail[['id', 'target']].copy()
    if len(submission) != len(test) or not submission['id'].equals(test['id']):
        raise RuntimeError('Submission row order differs from official test')
    if submission['id'].duplicated().any() or not np.isfinite(submission['target'].to_numpy(float)).all():
        raise RuntimeError('Submission contains duplicate IDs or non-finite targets')
    oof_full = oof.copy()
    detail_full = detail.copy()
    return (submission, oof_full, detail_full)

# ===== c284.py =====
"""In-memory port of round2_c284_current_only_pi1m_svd_reference.py.

Reads ONLY train.csv, test.csv and PI1M.csv from `data_dir`. Rebuilds the
C282 current-only parent model, appends a from-scratch, label-free PI1M
character n-gram SVD representation to the dense feature set, and returns the
4,940-row submission plus the out-of-fold predictions.

No archive labels, hashes, manifests, protocol files, or intermediate CSVs.
"""

c284_DEFAULT_CONFIG = {**reference.DEFAULT_CONFIG, 'pi1m_limit': 100000, 'pi1m_hash_features': 32768, 'pi1m_svd_components': 96, 'pi1m_ngram_min': 2, 'pi1m_ngram_max': 7}

def c284__load_current_only_inputs(data_dir):
    """Read train.csv + test.csv and mirror c282.load_current_only_inputs."""
    data_path = Path(data_dir)
    paths = {'train.csv': data_path / 'train.csv', 'test.csv': data_path / 'test.csv'}
    train = pd.read_csv(paths['train.csv'])
    test = pd.read_csv(paths['test.csv'])
    if list(train.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected current train schema')
    if list(test.columns) != ['id', 'smiles', 'target_type']:
        raise RuntimeError('Unexpected current test schema')
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError('Unexpected official current row count')
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [reference.canonicalize(value) for value in frame['smiles']]
    if set(train['target_type']) != set(reference.TARGETS) or set(test['target_type']) != set(reference.TARGETS):
        raise RuntimeError('Unexpected target set')
    if test['id'].duplicated().any() or not np.array_equal(test['id'].to_numpy(), np.arange(1, 4941)):
        raise RuntimeError('Test IDs are not unique sequential IDs 1..4940')
    if not np.isfinite(train['target'].to_numpy(float)).all():
        raise RuntimeError('Current train contains a non-finite target')
    return (train, test)

def c284__read_pi1m_smiles(path):
    frame = pd.read_csv(path, usecols=['SMILES'])
    values = frame['SMILES'].dropna().astype(str).tolist()
    if len(values) < 1000:
        raise RuntimeError('PI1M file unexpectedly small')
    return values

def c284__hash_ranked_unique(values, limit):
    unique = sorted(set(values), key=str)
    return unique[:limit]

def c284__pi1m_svd_features(keys, pi1m_path, config):
    pi1m = c284__read_pi1m_smiles(pi1m_path)
    corpus = c284__hash_ranked_unique(pi1m, int(config['pi1m_limit']))
    corpus_plus_keys = corpus + list(keys)
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(int(config['pi1m_ngram_min']), int(config['pi1m_ngram_max'])), n_features=int(config['pi1m_hash_features']), alternate_sign=False, norm='l2', lowercase=False, dtype=np.float64)
    hashed = vectorizer.transform(corpus_plus_keys).tocsr()
    max_components = min(hashed.shape[0] - 1, hashed.shape[1] - 1)
    n_components = min(int(config['pi1m_svd_components']), max_components)
    if n_components < 8:
        raise RuntimeError('PI1M SVD component count too small: %d' % n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=int(config['seed']))
    embedding = svd.fit_transform(hashed).astype(np.float64, copy=False)
    key_embedding = embedding[-len(keys):]
    return key_embedding

def c284_build_c284(data_dir):
    """Build the C284 PI1M-SVD reference model in memory.

    Returns:
        (submission_df, oof_df)
        submission_df: columns ["id", "target"], 4,940 rows, ids 1..4940.
        oof_df: columns ["canonical", "target_type", "target", "prediction"].
    """
    configuration = dict(c284_DEFAULT_CONFIG)
    np.random.seed(int(configuration['seed']))
    data_path = Path(data_dir)
    train, test = c284__load_current_only_inputs(data_path)
    pi1m_path = data_path / 'PI1M.csv'
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled['canonical']) | set(test['canonical']))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    pi1m_features = c284__pi1m_svd_features(keys=keys, pi1m_path=pi1m_path, config=configuration)
    dense_base = np.hstack([descriptor, physical, pi1m_features]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [reference.morgan_count_matrix(molecules, radius=2, bits=int(configuration['morgan_bits'])), reference.morgan_count_matrix(molecules, radius=3, bits=int(configuration['morgan_bits'])), reference.text_matrix(keys, int(configuration['text_features']))]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(configuration['morgan_bits']))
    detail, oof, _ = reference.fit_targets(pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, configuration)
    final_detail, _ = reference.apply_official_overrides(detail, test, raw_labels)
    submission = final_detail[['id', 'target']].copy()
    if len(submission) != len(test) or not submission['id'].equals(test['id']):
        raise RuntimeError('Submission row order differs from official test')
    if submission['id'].duplicated().any() or not np.isfinite(submission['target'].to_numpy(float)).all():
        raise RuntimeError('Submission contains duplicate IDs or non-finite targets')
    oof_full = oof.copy()
    detail_full = detail.copy()
    return (submission, oof_full, detail_full)

# ===== c285.py =====
"""In-memory port of round2_c285_current_only_pi1m_svd_weak_residual.py.

Reads ONLY train.csv, test.csv and PI1M.csv from `data_dir`. Rebuilds the
C282 current-only parent, learns a from-scratch, label-free PI1M character
n-gram SVD representation, fits residual heads for ei/nc/eps only, and returns
the 4,940-row submission plus the candidate out-of-fold predictions.

No archive labels, hashes, manifests, protocol files, or intermediate CSVs.
"""

c285_TARGETS = tuple(reference.TARGETS)
c285_ACTIVE_TARGETS = ('ei', 'nc', 'eps')
c285_DEFAULT_CONFIG = {**reference.DEFAULT_CONFIG, 'pi1m_limit': 100000, 'pi1m_hash_features': 32768, 'pi1m_svd_components': 96, 'residual_weight': 0.5, 'ridge_alpha': 20.0, 'extra_trees_estimators': 400, 'extra_trees_min_leaf': 4, 'gate_min_oof_delta': 0.005, 'gate_min_positive_folds': 3}

def c285__load_current_only_inputs(data_dir):
    """Read train.csv + test.csv and mirror c282.load_current_only_inputs."""
    data_path = Path(data_dir)
    paths = {'train.csv': data_path / 'train.csv', 'test.csv': data_path / 'test.csv'}
    train = pd.read_csv(paths['train.csv'])
    test = pd.read_csv(paths['test.csv'])
    if list(train.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected current train schema')
    if list(test.columns) != ['id', 'smiles', 'target_type']:
        raise RuntimeError('Unexpected current test schema')
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError('Unexpected official current row count')
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [reference.canonicalize(value) for value in frame['smiles']]
    if set(train['target_type']) != set(reference.TARGETS) or set(test['target_type']) != set(reference.TARGETS):
        raise RuntimeError('Unexpected target set')
    if test['id'].duplicated().any() or not np.array_equal(test['id'].to_numpy(), np.arange(1, 4941)):
        raise RuntimeError('Test IDs are not unique sequential IDs 1..4940')
    if not np.isfinite(train['target'].to_numpy(float)).all():
        raise RuntimeError('Current train contains a non-finite target')
    return (train, test)

def c285__hash_ranked_pi1m(path, limit):
    frame = pd.read_csv(path, usecols=['SMILES'])
    values = frame['SMILES'].dropna().astype(str).tolist()
    unique = sorted(set(values), key=str)
    return unique[:limit]

def c285__pi1m_char_svd(keys, pi1m_path, config):
    corpus = c285__hash_ranked_pi1m(pi1m_path, int(config['pi1m_limit']))
    all_smiles = corpus + list(keys)
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(2, 7), n_features=int(config['pi1m_hash_features']), alternate_sign=False, norm='l2', lowercase=False, dtype=np.float64)
    hashed = vectorizer.transform(all_smiles).tocsr()
    n_components = min(int(config['pi1m_svd_components']), hashed.shape[0] - 1, hashed.shape[1] - 1)
    if n_components < 8:
        raise RuntimeError('Too few rows/features for PI1M SVD')
    svd = TruncatedSVD(n_components=n_components, random_state=int(config['seed']))
    features = svd.fit_transform(hashed)[-len(keys):].astype(np.float64, copy=False)
    return features

def c285__grouped_folds(groups):
    result = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError('Failed to assign all grouped folds')
    return result

def c285__bootstrap_lower(y, parent, candidate, groups):
    unique = np.unique(groups)
    delta_by_group = {}
    for group in unique:
        rows = groups == group
        delta_by_group[group] = (y[rows] - parent[rows]) ** 2 - (y[rows] - candidate[rows]) ** 2
    rng = np.random.default_rng(20260807)
    draws = np.empty(1000, dtype=np.float64)
    for i in range(len(draws)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        draws[i] = float(np.mean(np.concatenate([delta_by_group[group] for group in selected])))
    return float(np.quantile(draws, 0.025))

def c285__target_model(target, config, seed):
    if target == 'ei':
        return Ridge(alpha=float(config['ridge_alpha']))
    return ExtraTreesRegressor(n_estimators=int(config['extra_trees_estimators']), min_samples_leaf=int(config['extra_trees_min_leaf']), max_features=0.8, random_state=seed, n_jobs=4)

def c285_build_c285(data_dir):
    """Build the C285 PI1M-SVD weak-residual model in memory.

    Returns:
        (submission_df, oof_df)
        submission_df: columns ["id", "target"], 4,940 rows, ids 1..4940.
        oof_df: columns ["canonical", "target_type", "target", "prediction"].
    """
    configuration = dict(c285_DEFAULT_CONFIG)
    data_path = Path(data_dir)
    train, test = c285__load_current_only_inputs(data_path)
    pi1m_path = data_path / 'PI1M.csv'
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled['canonical']) | set(test['canonical']))
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [reference.morgan_count_matrix(molecules, radius=2, bits=int(configuration['morgan_bits'])), reference.morgan_count_matrix(molecules, radius=3, bits=int(configuration['morgan_bits'])), reference.text_matrix(keys, int(configuration['text_features']))]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(configuration['morgan_bits']))
    detail, oof, _ = reference.fit_targets(pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, configuration)
    final_detail, _ = reference.apply_official_overrides(detail, test, raw_labels)
    pi1m_features = c285__pi1m_char_svd(keys, pi1m_path, configuration)
    scaler = StandardScaler()
    X = scaler.fit_transform(pi1m_features)
    parent_submission = final_detail[['id', 'target_type', 'target', 'override', 'model_prediction']].copy()
    candidate_submission = parent_submission.copy()
    oof_records = []
    c285_component_records = []
    for target in c285_TARGETS:
        target_oof = oof[oof['target_type'] == target].reset_index(drop=True)
        y = target_oof['target'].to_numpy(float)
        parent_pred = target_oof['prediction'].to_numpy(float)
        canonical = target_oof['canonical'].astype(str).to_numpy()
        indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        if target not in c285_ACTIVE_TARGETS:
            oof_records.append(pd.DataFrame({'canonical': canonical, 'target_type': target, 'target': y, 'prediction': parent_pred, 'parent': parent_pred, 'candidate': parent_pred}))
            continue
        folds = c285__grouped_folds(canonical)
        residual_oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_deltas = []
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            training = np.flatnonzero(folds != fold)
            model = c285__target_model(target, configuration, seed=int(configuration['seed']) + 17 * fold + c285_TARGETS.index(target))
            model.fit(X[indices[training]], y[training] - parent_pred[training])
            residual_oof[validation] = model.predict(X[indices[validation]])
            cand_fold = parent_pred[validation] + float(configuration['residual_weight']) * residual_oof[validation]
            parent_r2 = float(r2_score(y[validation], parent_pred[validation]))
            cand_r2 = float(r2_score(y[validation], cand_fold))
            fold_deltas.append(cand_r2 - parent_r2)
        candidate_oof = parent_pred + float(configuration['residual_weight']) * residual_oof
        parent_r2 = float(r2_score(y, parent_pred))
        candidate_r2 = float(r2_score(y, candidate_oof))
        delta = candidate_r2 - parent_r2
        positive_folds = int(sum((1 for value in fold_deltas if value > 0.0)))
        lower = c285__bootstrap_lower(y, parent_pred, candidate_oof, canonical)
        selected = bool(delta >= float(configuration['gate_min_oof_delta']) and positive_folds >= int(configuration['gate_min_positive_folds']) and (lower > 0.0))
        oof_records.append(pd.DataFrame({'canonical': canonical, 'target_type': target, 'target': y, 'prediction': candidate_oof, 'parent': parent_pred, 'candidate': candidate_oof}))
        test_rows = test[test['target_type'] == target].sort_values('id').reset_index(drop=True)
        test_indices = np.asarray([key_to_index[value] for value in test_rows['canonical']], dtype=np.int64)
        full_model = c285__target_model(target, configuration, seed=int(configuration['seed']) + 101 + c285_TARGETS.index(target))
        full_model.fit(X[indices], y - parent_pred)
        test_residual = full_model.predict(X[test_indices])
        ids = test_rows['id'].astype(int).to_numpy()
        pred_rows = candidate_submission['id'].isin(ids)
        base_values = candidate_submission.loc[pred_rows, 'target'].to_numpy(float)
        override_values = candidate_submission.loc[pred_rows, 'override'].astype(str).to_numpy()
        adjusted = base_values + float(configuration['residual_weight']) * test_residual
        final_values = np.where(override_values == 'model', adjusted, base_values)
        if selected:
            candidate_submission.loc[pred_rows, 'target'] = final_values
        c285_component_records.append(pd.DataFrame({'id': ids, 'target_type': target, 'parent': base_values, 'residual': test_residual, 'candidate': final_values, 'selected': selected, 'override': override_values}))
    output = candidate_submission[['id', 'target']].sort_values('id').reset_index(drop=True)
    if len(output) != 4940 or not np.array_equal(output['id'].to_numpy(), np.arange(1, 4941)) or (not np.isfinite(output['target'].to_numpy(float)).all()):
        raise RuntimeError('Candidate output contract failed')
    oof_df = pd.concat(oof_records, ignore_index=True)[['canonical', 'target_type', 'target', 'prediction', 'parent', 'candidate']]
    component_df = pd.concat(c285_component_records, ignore_index=True) if c285_component_records else pd.DataFrame(columns=['id', 'target_type', 'parent', 'residual', 'candidate', 'selected', 'override'])
    return (output, oof_df, component_df)

# ===== c286_stack.py =====
"""C286 current-only weak-target stacker (in-memory port).

Port of round2_c286_current_only_shift_domain_weak_stacker.py into a single
standalone in-memory function. It stacks, per weak target (ei, eea, nc, eps),
aligned OOF/test arms from the C282/C284/C285 reference artifacts plus (for
eps/nc) fold-safe F02-style physics arms, selecting targetwise nonnegative
blend weights from nested grouped OOF with test-domain weighting. The unchanged
targets (tg, egc, egb) are carried from the base candidate.

Deliberate non-behavioral deviations from the source:
- All filesystem I/O (pd.read_csv / to_csv / Path / open), digest/digest module
  manifest code, path guards, argparse flag parsing, os.environ reads, and
  progress jsonl logging are removed. Inputs arrive as DataFrames.
- dense_zoo_arms (and its fit_dense_preprocessor / clip_prediction
  dependencies) is omitted because the given flags use
  dense_zoo_profile == "disabled_v3_artifact_only", so that branch is never
  executed (RandomForestRegressor / HuberRegressor / Ridge /
  HistGradientBoostingRegressor are likewise unused).
- The module-level CONFIG dict is replaced by explicit keyword parameters with
  identical defaults (seed=20260807, outer_folds=5, dense_abs_limit=1e12,
  min_nested_delta_over_best_single=0.002, use_domain_weighting=True).
"""
c286_stack_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
c286_stack_ACTIVE_TARGETS = ('ei', 'eea', 'nc', 'eps')
c286_stack_POLAR_SMARTS = {'CF': '[#6][F]', 'CCl': '[#6][Cl]', 'ester': 'C(=O)O', 'carbonyl': '[CX3]=[OX1]', 'ether': '[OD2]([#6])[#6]', 'OH': '[OX2H]', 'nitrile': 'C#N', 'amide': 'C(=O)N', 'NH': '[NX3;H1,H2]', 'sulfone': 'S(=O)(=O)', 'thioether': '[#16X2]', 'aromatic_N': 'n', 'aromatic_O': 'o', 'aromatic_S': 's', 'imide': 'C(=O)NC(=O)', 'siloxane': '[Si][O]', 'phosphate': 'P=O', 'urethane': 'NC(=O)O'}
c286_stack_POLAR_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in c286_stack_POLAR_SMARTS.items()}

def c286_stack__canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def c286_stack__build_label_pool(train: pd.DataFrame, archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = train[['smiles', 'canonical', 'target_type', 'target']].copy()
    current['source'] = 'current_train'
    old = archive[['smiles', 'canonical', 'target_type', 'target']].copy()
    old['source'] = 'archive_train'
    raw = pd.concat([current, old], ignore_index=True)
    raw = raw.drop_duplicates(['smiles', 'target_type', 'target']).reset_index(drop=True)
    pooled = raw.groupby(['canonical', 'target_type'], as_index=False).agg(target=('target', 'median'), smiles=('smiles', 'first'), measurements=('target', 'size'))
    return (raw, pooled)

def c286_stack__build_molecules(keys: list[str]) -> list:
    molecules = [Chem.MolFromSmiles(value) for value in keys]
    if any((molecule is None for molecule in molecules)):
        raise RuntimeError('Canonical official structure failed RDKit parsing')
    return molecules

def c286_stack__descriptor_matrix(molecules: list) -> tuple[np.ndarray, list[str]]:
    items = list(Descriptors._descList)
    matrix = np.full((len(molecules), len(items)), np.nan, dtype=np.float64)
    for row, molecule in enumerate(molecules):
        for column, (_, function) in enumerate(items):
            try:
                value = float(function(molecule))
            except Exception:
                value = math.nan
            matrix[row, column] = value if math.isfinite(value) else math.nan
    return (matrix, [name for name, _ in items])

def c286_stack__physical_matrix(molecules: list, smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ['smiles_length', 'atom_count', 'heavy_atom_count', 'dummy_atom_count', 'ring_count', 'aromatic_atom_count', 'hetero_atom_count', 'halogen_count', 'rotatable_bonds_approx', 'double_bond_count', 'triple_bond_count', 'branch_count', 'n_count', 'o_count', 's_count', 'si_count']
    matrix = np.zeros((len(molecules), len(names)), dtype=np.float64)
    for row, (molecule, value) in enumerate(zip(molecules, smiles, strict=True)):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        matrix[row] = [len(value), molecule.GetNumAtoms(), molecule.GetNumHeavyAtoms(), sum((atom.GetAtomicNum() == 0 for atom in atoms)), molecule.GetRingInfo().NumRings(), sum((atom.GetIsAromatic() for atom in atoms)), sum((atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)), sum((atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)), sum((bond.GetBondTypeAsDouble() == 1.0 and (not bond.IsInRing()) for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 2.0 for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 3.0 for bond in bonds)), value.count('('), sum((atom.GetAtomicNum() == 7 for atom in atoms)), sum((atom.GetAtomicNum() == 8 for atom in atoms)), sum((atom.GetAtomicNum() == 16 for atom in atoms)), sum((atom.GetAtomicNum() == 14 for atom in atoms))]
    return (matrix, names)

def c286_stack__cross_property_arrays(pooled: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pivot = pooled.pivot(index='canonical', columns='target_type', values='target')
    values = np.full((len(keys), len(c286_stack_TARGETS)), np.nan, dtype=np.float64)
    available = np.zeros((len(keys), len(c286_stack_TARGETS)), dtype=np.float64)
    key_position = {key: index for index, key in enumerate(keys)}
    for target_index, target in enumerate(c286_stack_TARGETS):
        if target not in pivot:
            continue
        series = pivot[target].dropna()
        for key, value in series.items():
            position = key_position.get(key)
            if position is not None:
                values[position, target_index] = float(value)
                available[position, target_index] = 1.0
    return (values, available)

def c286_stack__target_dense_features(base_dense: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray, target: str) -> np.ndarray:
    values = cross_values.copy()
    available = cross_available.copy()
    target_index = c286_stack_TARGETS.index(target)
    values[:, target_index] = np.nan
    available[:, target_index] = 0.0
    return np.hstack([base_dense, values, available]).astype(np.float64, copy=False)

def c286_stack_grouped_folds(groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    unique = np.array(sorted(pd.unique(pd.Series(groups).astype(str))))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    fold_of = {group: int(i % n_splits) for i, group in enumerate(shuffled)}
    folds = np.array([fold_of[str(group)] for group in groups], dtype=np.int64)
    if len(np.unique(folds)) != n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        folds = np.full(len(groups), -1, dtype=np.int64)
        for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
            folds[validation] = fold
    if np.any(folds < 0):
        raise RuntimeError('Fold assignment failed')
    return folds

def c286_stack_sanitize_dense(x: np.ndarray, absolute_limit: float) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(out) | (np.abs(out) > absolute_limit)
    out[bad] = np.nan
    med = np.nanmedian(out, axis=0)
    med[~np.isfinite(med)] = 0.0
    rows, cols = np.where(~np.isfinite(out))
    out[rows, cols] = med[cols]
    return out

def c286_stack_weighted_r2(y: np.ndarray, pred: np.ndarray, w: np.ndarray | None=None) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if w is None:
        return float(r2_score(y, pred))
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0.0:
        return float(r2_score(y, pred))
    mu = float(np.sum(w * y) / np.sum(w))
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - mu) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)

def c286_stack_fit_nonnegative_blend(y: np.ndarray, base: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, float, str]:
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if w is None:
        w = np.ones(len(y), dtype=np.float64)
    w = np.maximum(np.asarray(w, dtype=np.float64), 1e-12)
    w = w / np.mean(w)
    y_mean = float(np.sum(w * y) / np.sum(w))
    x_mean = np.sum(base * w[:, None], axis=0) / np.sum(w)
    yw = (y - y_mean) * np.sqrt(w)
    xw = (base - x_mean) * np.sqrt(w[:, None])
    weights, _ = nnls(xw, yw)
    if float(np.sum(weights)) <= 1e-12:
        weights = np.full(base.shape[1], 1.0 / base.shape[1], dtype=np.float64)
    else:
        weights = weights / np.sum(weights)
    intercept = float(y_mean - np.dot(x_mean, weights))
    blend = base @ weights + intercept
    blend_score = c286_stack_weighted_r2(y, blend, w)
    single_scores = [c286_stack_weighted_r2(y, base[:, i], w) for i in range(base.shape[1])]
    best = int(np.argmax(single_scores))
    if single_scores[best] >= blend_score:
        weights = np.zeros(base.shape[1], dtype=np.float64)
        weights[best] = 1.0
        intercept = 0.0
        return (weights, intercept, f'single_{best}')
    return (weights, intercept, 'nnls_blend')

def c286_stack_domain_weights(x_train: np.ndarray, x_test: np.ndarray, dense_abs_limit: float) -> np.ndarray:
    train = c286_stack_sanitize_dense(x_train, dense_abs_limit)
    test = c286_stack_sanitize_dense(x_test, dense_abs_limit)
    mu = np.mean(train, axis=0)
    sd = np.std(train, axis=0)
    sd[sd < 1e-08] = 1.0
    train = np.clip((train - mu) / sd, -8.0, 8.0)
    test = np.clip((test - mu) / sd, -8.0, 8.0)
    best = np.full(len(train), np.inf, dtype=np.float64)
    for start in range(0, len(test), 256):
        block = test[start:start + 256]
        dist2 = np.sum((train[:, None, :] - block[None, :, :]) ** 2, axis=2)
        best = np.minimum(best, np.min(dist2, axis=1))
    scale = float(np.median(best[np.isfinite(best)]))
    scale = max(scale, 1e-08)
    weights = np.exp(-best / scale)
    weights = np.clip(weights, 0.2, 5.0)
    return weights / np.mean(weights)

def c286_stack_polar_block(canonicals: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for canonical in canonicals:
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise RuntimeError(f'RDKit failed to parse canonical SMILES: {canonical}')
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in c286_stack_POLAR_PATTERNS.values()]
        row.extend([Descriptors.TPSA(mol) / heavy, Descriptors.NumHDonors(mol) / heavy, Descriptors.NumHAcceptors(mol) / heavy, Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol) / heavy, Crippen.MolMR(mol) / heavy, Crippen.MolLogP(mol) / heavy, rdMolDescriptors.CalcNumAromaticRings(mol) / heavy])
        rows.append([float(v) if math.isfinite(float(v)) else 0.0 for v in row])
    return np.asarray(rows, dtype=np.float64)

def c286_stack_physics_eps_nc_arms(*, pooled: pd.DataFrame, test: pd.DataFrame, c282_oof: pd.DataFrame, c282_test: pd.DataFrame, seed: int, outer_folds: int) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Generate fold-safe F02-style physics OOF/test arms for eps and nc."""
    oof_out: dict[str, pd.Series] = {}
    test_out: dict[str, pd.Series] = {}
    wide = pooled.pivot(index='canonical', columns='target_type', values='target')
    pair = wide[wide.get('eps').notna() & wide.get('nc').notna()]
    if len(pair) < 20:
        return (oof_out, test_out)
    pair_cans = pair.index.astype(str).tolist()
    pair_ionic = (pair['eps'].to_numpy(float) - np.square(pair['nc'].to_numpy(float))).astype(np.float64)
    all_need = list(dict.fromkeys(pooled['canonical'].astype(str).tolist() + test['canonical'].astype(str).tolist()))
    all_features = c286_stack_polar_block(all_need)
    all_index = {can: i for i, can in enumerate(all_need)}
    partner_full: dict[str, dict[str, float]] = {}
    for partner in ('eps', 'nc'):
        rows = wide[partner].dropna()
        cans = rows.index.astype(str).tolist()
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=int(seed), n_jobs=4)
        model.fit(all_features[[all_index[c] for c in cans]], rows.to_numpy(float))
        pred = model.predict(all_features)
        partner_full[partner] = dict(zip(all_need, pred, strict=True))
        partner_full[partner].update({str(c): float(v) for c, v in rows.items()})
    ionic_full = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=int(seed) + 41, n_jobs=4)
    ionic_full.fit(all_features[[all_index[c] for c in pair_cans]], pair_ionic)
    ionic_pred_full = dict(zip(all_need, ionic_full.predict(all_features), strict=True))
    for target, partner in (('eps', 'nc'), ('nc', 'eps')):
        target_rows = pooled[pooled['target_type'] == target].reset_index(drop=True)
        cans = target_rows['canonical'].astype(str).to_numpy()
        c282_rows = c282_oof[c282_oof['target_type'] == target].reset_index(drop=True)
        if len(c282_rows) != len(target_rows) or not np.all(c282_rows['canonical'].astype(str).to_numpy() == cans):
            raise RuntimeError(f'C282 OOF alignment failed for {target}')
        b0 = c282_rows['prediction'].to_numpy(float)
        folds = c286_stack_grouped_folds(cans, int(outer_folds), int(seed) + 901)
        arm = np.empty(len(c282_rows), dtype=np.float64)
        for fold in range(int(outer_folds)):
            validation = folds == fold
            va_cans = set(cans[validation])
            partner_rows = wide[partner].dropna()
            partner_train = [str(c) for c in partner_rows.index.astype(str).tolist() if str(c) not in va_cans]
            partner_model = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2, random_state=int(seed) + fold, n_jobs=4)
            partner_model.fit(all_features[[all_index[c] for c in partner_train]], np.array([float(partner_rows.loc[c]) for c in partner_train]))
            partner_pred = dict(zip(list(cans[validation]), partner_model.predict(all_features[[all_index[c] for c in cans[validation]]]), strict=True))
            for c, v in partner_rows.items():
                partner_pred[str(c)] = float(v)
            ionic_train = [c for c in pair_cans if c not in va_cans]
            ionic_model = ExtraTreesRegressor(n_estimators=360, min_samples_leaf=2, random_state=int(seed) + 31 + fold, n_jobs=4)
            ionic_model.fit(all_features[[all_index[c] for c in ionic_train]], np.array([float(pair.loc[c, 'eps'] - pair.loc[c, 'nc'] ** 2) for c in ionic_train]))
            ion = dict(zip(list(cans[validation]), ionic_model.predict(all_features[[all_index[c] for c in cans[validation]]]), strict=True))
            for local_idx, can in zip(np.flatnonzero(validation), cans[validation], strict=True):
                has_partner = can in wide.index and partner in wide.columns and pd.notna(wide.loc[can, partner])
                pval = float(wide.loc[can, partner]) if has_partner else float(partner_pred[can])
                ionic = max(float(ion[can]), 0.02)
                if target == 'eps':
                    phys = pval * pval + ionic
                else:
                    phys = math.sqrt(max(pval - ionic, 1.0))
                arm[local_idx] = phys if has_partner else 0.5 * phys + 0.5 * b0[local_idx]
        oof_out[f'physics_{target}_b2'] = pd.Series(arm, index=target_rows['canonical'].astype(str))
        te = test[test['target_type'] == target].sort_values('id').reset_index(drop=True)
        te_cans = te['canonical'].astype(str).tolist()
        c282_te = c282_test[c282_test['target_type'] == target].sort_values('id').reset_index(drop=True)
        b0_test = c282_te['model_prediction'].to_numpy(float)
        values = np.empty(len(te), dtype=np.float64)
        for i, can in enumerate(te_cans):
            has_partner = can in wide.index and partner in wide.columns and pd.notna(wide.loc[can, partner])
            pval = float(wide.loc[can, partner]) if has_partner else float(partner_full[partner][can])
            ionic = max(float(ionic_pred_full[can]), 0.02)
            if target == 'eps':
                phys = pval * pval + ionic
            else:
                phys = math.sqrt(max(pval - ionic, 1.0))
            values[i] = phys if has_partner else 0.5 * phys + 0.5 * b0_test[i]
        test_out[f'physics_{target}_b2'] = pd.Series(values, index=te['id'].astype(int))
    return (oof_out, test_out)

def c286_stack_add_reference_artifact_arms(target: str, *, oof_arms: dict[str, pd.Series], test_arms: dict[str, pd.Series], c282_oof: pd.DataFrame, c282_test: pd.DataFrame, c284_oof: pd.DataFrame, c284_test: pd.DataFrame, c285_oof: pd.DataFrame, c285_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_oof = c282_oof[c282_oof['target_type'] == target].reset_index(drop=True)
    target_test = c282_test[c282_test['target_type'] == target].sort_values('id').reset_index(drop=True)
    canonical_index = target_oof['canonical'].astype(str)
    id_index = target_test['id'].astype(int)
    y_frame = pd.DataFrame({'canonical': canonical_index, 'target': target_oof['target'].to_numpy(float)})
    for prefix, oo, tt in (('c282', c282_oof, c282_test), ('c284', c284_oof, c284_test)):
        oo_t = oo[oo['target_type'] == target].reset_index(drop=True)
        tt_t = tt[tt['target_type'] == target].sort_values('id').reset_index(drop=True)
        if len(oo_t) != len(target_oof) or not np.all(oo_t['canonical'].astype(str).to_numpy() == canonical_index.to_numpy()):
            raise RuntimeError(f'{prefix} OOF alignment failed for {target}')
        if len(tt_t) != len(target_test) or not np.all(tt_t['id'].astype(int).to_numpy() == id_index.to_numpy()):
            raise RuntimeError(f'{prefix} test alignment failed for {target}')
        for col in ('prediction', 'sparse_ridge', 'dense_ridge', 'extra_trees', 'tanimoto_local'):
            oof_arms[f'{prefix}_{col}'] = pd.Series(oo_t[col].to_numpy(float), index=canonical_index)
            test_col = 'model_prediction' if col == 'prediction' else col
            test_arms[f'{prefix}_{col}'] = pd.Series(tt_t[test_col].to_numpy(float), index=id_index)
    c285_o = c285_oof[c285_oof['target_type'] == target].reset_index(drop=True)
    if len(c285_o) == len(target_oof) and 'canonical' in c285_o.columns and c285_o['canonical'].notna().all():
        if not np.all(c285_o['canonical'].astype(str).to_numpy() == canonical_index.to_numpy()):
            raise RuntimeError(f'C285 OOF alignment failed for {target}')
        for col in ('parent', 'candidate'):
            oof_arms[f'c285_{col}'] = pd.Series(c285_o[col].to_numpy(float), index=canonical_index)
    c285_t = c285_test[c285_test['target_type'] == target].sort_values('id').reset_index(drop=True)
    if len(c285_t):
        if not np.all(c285_t['id'].astype(int).to_numpy() == id_index.to_numpy()):
            raise RuntimeError(f'C285 test alignment failed for {target}')
        for col in ('parent', 'candidate'):
            test_arms[f'c285_{col}'] = pd.Series(c285_t[col].to_numpy(float), index=id_index)
    return (y_frame, pd.DataFrame({'id': id_index}))

def c286_stack_nested_stack_for_target(*, target: str, y_frame: pd.DataFrame, id_frame: pd.DataFrame, oof_arms: dict[str, pd.Series], test_arms: dict[str, pd.Series], x_train_domain: np.ndarray, x_test_domain: np.ndarray, seed: int, outer_folds: int, dense_abs_limit: float, use_domain_weighting: bool, min_nested_delta_over_best_single: float) -> tuple[np.ndarray, np.ndarray, dict]:
    names = sorted(set(oof_arms) & set(test_arms))
    if len(names) < 2:
        raise RuntimeError(f'Too few aligned arms for {target}: {names}')
    y = y_frame['target'].to_numpy(float)
    groups = y_frame['canonical'].astype(str).to_numpy()
    oof_matrix = np.column_stack([oof_arms[name].loc[y_frame['canonical'].astype(str)].to_numpy(float) for name in names])
    test_matrix = np.column_stack([test_arms[name].loc[id_frame['id'].astype(int)].to_numpy(float) for name in names])
    if not np.isfinite(oof_matrix).all() or not np.isfinite(test_matrix).all():
        raise RuntimeError(f'Non-finite stack matrix for {target}')
    row_weights = c286_stack_domain_weights(x_train_domain, x_test_domain, dense_abs_limit) if use_domain_weighting else np.ones(len(y))
    folds = c286_stack_grouped_folds(groups, int(outer_folds), int(seed) + 707)
    nested = np.empty(len(y), dtype=np.float64)
    fold_records: list[dict] = []
    for fold in range(int(outer_folds)):
        validation = folds == fold
        training = ~validation
        weights, intercept, mode = c286_stack_fit_nonnegative_blend(y[training], oof_matrix[training], row_weights[training])
        nested[validation] = oof_matrix[validation] @ weights + intercept
        fold_records.append({'fold': int(fold), 'rows': int(np.sum(validation)), 'mode': mode, 'r2': float(r2_score(y[validation], nested[validation])), 'best_single_r2': float(max((r2_score(y[validation], oof_matrix[validation, i]) for i in range(len(names)))))})
    final_weights, final_intercept, final_mode = c286_stack_fit_nonnegative_blend(y, oof_matrix, row_weights)
    final_test = test_matrix @ final_weights + final_intercept
    single_scores = {name: float(r2_score(y, oof_matrix[:, i])) for i, name in enumerate(names)}
    best_name = max(single_scores, key=single_scores.get)
    nested_r2 = float(r2_score(y, nested))
    best_single_r2 = float(single_scores[best_name])
    selected = bool(nested_r2 >= best_single_r2 + float(min_nested_delta_over_best_single))
    if not selected:
        final_weights = np.zeros(len(names), dtype=np.float64)
        final_weights[names.index(best_name)] = 1.0
        final_intercept = 0.0
        final_mode = f'fallback_best_single_{best_name}'
        final_test = test_matrix[:, names.index(best_name)]
    report = {'target': target, 'arms': names, 'single_oof_r2': single_scores, 'best_single': best_name, 'best_single_r2': best_single_r2, 'nested_stack_r2': nested_r2, 'nested_delta_over_best_single': nested_r2 - best_single_r2, 'selected_nested_stack': selected, 'final_mode': final_mode, 'final_intercept': float(final_intercept), 'final_weights': {name: float(final_weights[i]) for i, name in enumerate(names) if abs(float(final_weights[i])) > 1e-12}, 'folds': fold_records}
    return (nested, final_test, report)

def c286_stack_c286_stack(train_df: pd.DataFrame, test_df: pd.DataFrame, base_df: pd.DataFrame, c282_oof_df: pd.DataFrame, c282_test_df: pd.DataFrame, c284_oof_df: pd.DataFrame, c284_test_df: pd.DataFrame, c285_oof_df: pd.DataFrame, c285_test_df: pd.DataFrame, *, seed: int=20260807, outer_folds: int=5, dense_abs_limit: float=1000000000000.0, min_nested_delta_over_best_single: float=0.002, use_domain_weighting: bool=True) -> pd.DataFrame:
    """Run the C286 current-only weak-target stacker.

    Returns a 4,940-row DataFrame with columns [id, target]. Active weak targets
    (ei, eea, nc, eps) are replaced by the nested stack; unchanged targets are
    carried from base_df.
    """
    train = train_df.copy()
    test = test_df.copy()
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [c286_stack__canonicalize(value) for value in frame['smiles']]
    base = base_df.copy()
    if list(base.columns) != ['id', 'target'] or len(base) != 4940:
        raise RuntimeError('Base F18 candidate invalid')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), test['id'].to_numpy(int)):
        raise RuntimeError('Base F18 ID validation failed')
    raw_labels, pooled = c286_stack__build_label_pool(train, train.iloc[0:0].copy())
    keys = sorted(set(pooled['canonical']) | set(test['canonical']))
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    molecules = c286_stack__build_molecules(keys)
    descriptor, descriptor_names = c286_stack__descriptor_matrix(molecules)
    physical, physical_names = c286_stack__physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = c286_stack__cross_property_arrays(pooled, keys)
    physics_oof, physics_test = c286_stack_physics_eps_nc_arms(pooled=pooled, test=test, c282_oof=c282_oof_df, c282_test=c282_test_df, seed=seed, outer_folds=outer_folds)
    result = base.copy()
    for target in c286_stack_ACTIVE_TARGETS:
        oof_arms: dict[str, pd.Series] = {}
        test_arms: dict[str, pd.Series] = {}
        y_frame, id_frame = c286_stack_add_reference_artifact_arms(target, oof_arms=oof_arms, test_arms=test_arms, c282_oof=c282_oof_df, c282_test=c282_test_df, c284_oof=c284_oof_df, c284_test=c284_test_df, c285_oof=c285_oof_df, c285_test=c285_test_df)
        if target in ('eps', 'nc'):
            for name, values in physics_oof.items():
                if name.startswith(f'physics_{target}_'):
                    oof_arms[name] = values
            for name, values in physics_test.items():
                if name.startswith(f'physics_{target}_'):
                    test_arms[name] = values
        train_index = np.array([key_to_index[str(c)] for c in y_frame['canonical'].astype(str)], dtype=np.int64)
        test_subset = test[test['target_type'] == target].sort_values('id').reset_index(drop=True)
        test_index = np.array([key_to_index[str(c)] for c in test_subset['canonical'].astype(str)], dtype=np.int64)
        dense_target = c286_stack__target_dense_features(dense_base, cross_values, cross_available, target)
        nested, final_test, report = c286_stack_nested_stack_for_target(target=target, y_frame=y_frame, id_frame=id_frame, oof_arms=oof_arms, test_arms=test_arms, x_train_domain=dense_target[train_index], x_test_domain=dense_target[test_index], seed=seed, outer_folds=outer_folds, dense_abs_limit=dense_abs_limit, use_domain_weighting=use_domain_weighting, min_nested_delta_over_best_single=min_nested_delta_over_best_single)
        ids = id_frame['id'].to_numpy(int)
        id_mask = result['id'].astype(int).isin(ids)
        result.loc[id_mask, 'target'] = pd.Series(final_test, index=ids).loc[result.loc[id_mask, 'id'].astype(int)].to_numpy(float)
    if len(result) != 4940 or result['id'].duplicated().any() or (not np.array_equal(result['id'].to_numpy(int), test['id'].to_numpy(int))):
        raise RuntimeError('Final result ID/schema validation failed')
    if not np.isfinite(result['target'].to_numpy(float)).all():
        raise RuntimeError('Final result contains non-finite predictions')
    return result[['id', 'target']].reset_index(drop=True)

# ===== c287_zoo.py =====
"""In-memory port of the original round C287-v3 current-only weak-target model zoo
(round2_c287_current_only_weak_model_zoo_v3.py).

Reads ONLY official train.csv / test.csv from data_dir. No archive, no hashes, no
intermediate CSV reads/writes, no prior-prediction-as-feature. The F14 base is taken
from base_df (replacing CONFIG["base_candidate"]). Reference helpers and the C282-style
feature build are ported verbatim from the shared reference library and
round2_c282_current_only_reference.py.
"""
c287_zoo_ACTIVE_TARGETS = ('ei', 'eea', 'nc', 'eps')
c287_zoo_ARMS = ('dense_ridge_a5', 'dense_huber', 'dense_extra_trees', 'dense_random_forest', 'dense_hgb', 'mean5', 'median5')
c287_zoo_CONFIG = {'seed': 20260807, 'dense_abs_limit': 1000000000000.0}
c287_zoo_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

def c287_zoo__canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def c287_zoo__build_label_pool(train: pd.DataFrame, archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = train[['smiles', 'canonical', 'target_type', 'target']].copy()
    current['source'] = 'current_train'
    old = archive[['smiles', 'canonical', 'target_type', 'target']].copy()
    old['source'] = 'archive_train'
    raw = pd.concat([current, old], ignore_index=True)
    raw = raw.drop_duplicates(['smiles', 'target_type', 'target']).reset_index(drop=True)
    pooled = raw.groupby(['canonical', 'target_type'], as_index=False).agg(target=('target', 'median'), smiles=('smiles', 'first'), measurements=('target', 'size'))
    return (raw, pooled)

def c287_zoo__build_molecules(keys: list[str]) -> list[Any]:
    molecules = [Chem.MolFromSmiles(value) for value in keys]
    if any((molecule is None for molecule in molecules)):
        raise RuntimeError('Canonical official structure failed RDKit parsing')
    return molecules

def c287_zoo__descriptor_matrix(molecules: list[Any]) -> tuple[np.ndarray, list[str]]:
    items = list(Descriptors._descList)
    matrix = np.full((len(molecules), len(items)), np.nan, dtype=np.float64)
    for row, molecule in enumerate(molecules):
        for column, (_, function) in enumerate(items):
            try:
                value = float(function(molecule))
            except Exception:
                value = math.nan
            matrix[row, column] = value if math.isfinite(value) else math.nan
    return (matrix, [name for name, _ in items])

def c287_zoo__physical_matrix(molecules: list[Any], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ['smiles_length', 'atom_count', 'heavy_atom_count', 'dummy_atom_count', 'ring_count', 'aromatic_atom_count', 'hetero_atom_count', 'halogen_count', 'rotatable_bonds_approx', 'double_bond_count', 'triple_bond_count', 'branch_count', 'n_count', 'o_count', 's_count', 'si_count']
    matrix = np.zeros((len(molecules), len(names)), dtype=np.float64)
    for row, (molecule, value) in enumerate(zip(molecules, smiles, strict=True)):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        matrix[row] = [len(value), molecule.GetNumAtoms(), molecule.GetNumHeavyAtoms(), sum((atom.GetAtomicNum() == 0 for atom in atoms)), molecule.GetRingInfo().NumRings(), sum((atom.GetIsAromatic() for atom in atoms)), sum((atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)), sum((atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)), sum((bond.GetBondTypeAsDouble() == 1.0 and (not bond.IsInRing()) for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 2.0 for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 3.0 for bond in bonds)), value.count('('), sum((atom.GetAtomicNum() == 7 for atom in atoms)), sum((atom.GetAtomicNum() == 8 for atom in atoms)), sum((atom.GetAtomicNum() == 16 for atom in atoms)), sum((atom.GetAtomicNum() == 14 for atom in atoms))]
    return (matrix, names)

def c287_zoo__cross_property_arrays(pooled: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pivot = pooled.pivot(index='canonical', columns='target_type', values='target')
    values = np.full((len(keys), len(c287_zoo_TARGETS)), np.nan, dtype=np.float64)
    available = np.zeros((len(keys), len(c287_zoo_TARGETS)), dtype=np.float64)
    key_position = {key: index for index, key in enumerate(keys)}
    for target_index, target in enumerate(c287_zoo_TARGETS):
        if target not in pivot:
            continue
        series = pivot[target].dropna()
        for key, value in series.items():
            position = key_position.get(key)
            if position is not None:
                values[position, target_index] = float(value)
                available[position, target_index] = 1.0
    return (values, available)

def c287_zoo__target_dense_features(base_dense: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray, target: str) -> np.ndarray:
    values = cross_values.copy()
    available = cross_available.copy()
    target_index = c287_zoo_TARGETS.index(target)
    values[:, target_index] = np.nan
    available[:, target_index] = 0.0
    return np.hstack([base_dense, values, available]).astype(np.float64, copy=False)

def c287_zoo__fit_dense_preprocessor(dense: np.ndarray, train_index: np.ndarray, prediction_index: np.ndarray, absolute_limit: float):
    sanitized = np.asarray(dense, dtype=np.float64).copy()
    invalid = ~np.isfinite(sanitized) | (np.abs(sanitized) > absolute_limit)
    sanitized[invalid] = np.nan
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(sanitized[train_index])
    prediction_imputed = imputer.transform(sanitized[prediction_index])
    train_scaled = scaler.fit_transform(train_imputed)
    prediction_scaled = scaler.transform(prediction_imputed)
    return (train_imputed, prediction_imputed, train_scaled, prediction_scaled)

def c287_zoo__clip_prediction(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    return np.clip(prediction, q01 - 2.0 * margin, q99 + 2.0 * margin)

def c287_zoo__load_current_only_inputs(data_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    if list(train.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected current train schema')
    if list(test.columns) != ['id', 'smiles', 'target_type']:
        raise RuntimeError('Unexpected current test schema')
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError('Unexpected official current row count')
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [c287_zoo__canonicalize(value) for value in frame['smiles']]
    if set(train['target_type']) != set(c287_zoo_TARGETS) or set(test['target_type']) != set(c287_zoo_TARGETS):
        raise RuntimeError('Unexpected target set')
    if test['id'].duplicated().any() or not np.array_equal(test['id'].to_numpy(), np.arange(1, 4941)):
        raise RuntimeError('Test IDs are not unique sequential IDs 1..4940')
    if not np.isfinite(train['target'].to_numpy(float)).all():
        raise RuntimeError('Current train contains a non-finite target')
    return (train, test)

def c287_zoo__clip(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return c287_zoo__clip_prediction(y, np.asarray(pred, dtype=np.float64))

def c287_zoo__copy_with_replacement(base: pd.DataFrame, ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    mapping = dict(zip(ids.astype(int), values.astype(float), strict=True))
    mask = out['id'].astype(int).isin(mapping)
    out.loc[mask, 'target'] = out.loc[mask, 'id'].astype(int).map(mapping).to_numpy(float)
    return out

def c287_zoo__build_features(train: pd.DataFrame, test: pd.DataFrame):
    archive = train.iloc[0:0].copy()
    _, pooled = c287_zoo__build_label_pool(train, archive)
    keys = sorted(set(pooled['canonical']) | set(test['canonical']))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = c287_zoo__build_molecules(keys)
    descriptor, _descriptor_names = c287_zoo__descriptor_matrix(molecules)
    physical, _physical_names = c287_zoo__physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = c287_zoo__cross_property_arrays(pooled, keys)
    return (pooled, keys, key_to_index, dense_base, cross_values, cross_available)

def c287_zoo_build_c287(data_dir, base_df: pd.DataFrame, components: dict | None=None) -> dict:
    """Port of round2_c287_current_only_weak_model_zoo_v3, in-memory.

    data_dir: official data directory (reads train.csv + test.csv only).
    base_df: F14 no-archive base DataFrame with columns ['id', 'target'] (4940 rows).
    components: optional dict of prebuilt C282-style in-memory feature artifacts with
        keys 'pooled' (DataFrame), 'keys' (list[str]), 'dense_base' (ndarray),
        'cross_values' (ndarray), 'cross_available' (ndarray). Any missing artifact is
        rebuilt in-memory from data_dir.

    Returns dict {f'{target}_{arm}': (id,target) DataFrame} for target in
    ('ei','eea','nc','eps') x ARMS (7 arms each), where each value is the full 4940-row
    F14 base with that target's rows replaced by the arm's predictions. Includes
    'eea_dense_huber'.
    """
    train, test = c287_zoo__load_current_only_inputs(data_dir)
    if list(base_df.columns) != ['id', 'target'] or len(base_df) != 4940 or base_df['id'].duplicated().any():
        raise RuntimeError('Base F14 candidate invalid')
    components = components or {}
    pooled = components.get('pooled')
    keys = components.get('keys')
    dense_base = components.get('dense_base')
    cross_values = components.get('cross_values')
    cross_available = components.get('cross_available')
    if pooled is None or keys is None or dense_base is None or (cross_values is None) or (cross_available is None):
        pooled, keys, key_to_index, dense_base, cross_values, cross_available = c287_zoo__build_features(train, test)
    else:
        key_to_index = {key: index for index, key in enumerate(keys)}
    output: dict[str, pd.DataFrame] = {}
    for target in c287_zoo_ACTIVE_TARGETS:
        target_train = pooled[pooled['target_type'] == target].reset_index(drop=True)
        target_test = test[test['target_type'] == target].sort_values('id').reset_index(drop=True)
        train_index = np.asarray([key_to_index[value] for value in target_train['canonical']], dtype=np.int64)
        test_index = np.asarray([key_to_index[value] for value in target_test['canonical']], dtype=np.int64)
        y = target_train['target'].to_numpy(float)
        dense = c287_zoo__target_dense_features(dense_base, cross_values, cross_available, target)
        _, _, x_train, x_test = c287_zoo__fit_dense_preprocessor(dense, train_index, test_index, float(c287_zoo_CONFIG['dense_abs_limit']))
        seed = int(c287_zoo_CONFIG['seed']) + 97 * c287_zoo_TARGETS.index(target)
        models = {'dense_ridge_a5': Ridge(alpha=5.0), 'dense_huber': HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=500), 'dense_extra_trees': ExtraTreesRegressor(n_estimators=360, min_samples_leaf=1, max_features=0.75, random_state=seed, n_jobs=4), 'dense_random_forest': RandomForestRegressor(n_estimators=320, min_samples_leaf=1, max_features=0.75, random_state=seed + 1, n_jobs=4), 'dense_hgb': HistGradientBoostingRegressor(max_iter=220, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=0.05, random_state=seed + 2)}
        ids = target_test['id'].to_numpy(int)
        arm_values: dict[str, np.ndarray] = {}
        for arm, model in models.items():
            model.fit(x_train, y)
            values = c287_zoo__clip(y, model.predict(x_test))
            arm_values[arm] = values
            output[f'{target}_{arm}'] = c287_zoo__copy_with_replacement(base_df, ids, values)
        stack = np.vstack([arm_values[arm] for arm in models])
        for arm, values in {'mean5': np.mean(stack, axis=0), 'median5': np.median(stack, axis=0)}.items():
            output[f'{target}_{arm}'] = c287_zoo__copy_with_replacement(base_df, ids, values)
    return output

# ===== c391_pi1m_zoo.py =====
"""C289 current-only PI1M-SVD + gradient-boosted target bank (in-memory port).

Faithful port of round2_c289_current_only_pi1m_lgbm_bank.py. Reads ONLY
official train/test/PI1M via data_dir. Returns a 4,940-row (id, target) DataFrame.
"""
try:
    import lightgbm as lgb
except Exception:
    lgb = None
c391_pi1m_zoo_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

def c391_pi1m_zoo_c289_maccs_matrix(molecules):
    rows = np.zeros((len(molecules), 167), dtype=np.float32)
    for i, mol in enumerate(molecules):
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows[i] = np.asarray([int(fp.GetBit(j)) for j in range(167)], dtype=np.float32)
    return rows

def c391_pi1m_zoo_c289_sanitize_dense(x):
    out = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(out) | (np.abs(out) > 1000000000000.0)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out

def c391_pi1m_zoo_c289_grouped_oof(model_factory, x, y, groups, seed):
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError('Not enough groups for OOF')
    oof = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = model_factory(seed + fold)
        model.fit(x[tr], y[tr])
        oof[va] = np.asarray(model.predict(x[va]), dtype=np.float64)
    if not np.isfinite(oof).all():
        raise RuntimeError('Non-finite OOF prediction')
    return oof

def c391_pi1m_zoo_c289_clipped_r2(y, pred):
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    clipped = np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)
    return (float(r2_score(y, clipped)), clipped)

def c391_pi1m_zoo_c289_clip_to_train_range(y, pred):
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    return np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)

def c391_pi1m_zoo_c289_model_factories(target, rows):
    large = target in {'tg', 'egc'}
    factories = {'ridge_10': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0, solver='lsqr', max_iter=5000, tol=0.0001)), 'ridge_100': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0, solver='lsqr', max_iter=5000, tol=0.0001)), 'extra_trees': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), ExtraTreesRegressor(n_estimators=500 if large else 700, min_samples_leaf=3 if large else 2, max_features=0.55 if large else 0.75, random_state=seed, n_jobs=4)), 'random_forest': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), RandomForestRegressor(n_estimators=350 if large else 500, min_samples_leaf=3 if large else 2, max_features=0.45 if large else 0.7, random_state=seed, n_jobs=4)), 'hist_gbdt': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), HistGradientBoostingRegressor(learning_rate=0.035, max_iter=450 if large else 650, l2_regularization=0.05, max_leaf_nodes=31 if large else 15, min_samples_leaf=20 if large else 8, random_state=seed))}
    if lgb is not None:
        factories['lightgbm'] = lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), lgb.LGBMRegressor(objective='regression', n_estimators=900 if large else 700, learning_rate=0.025 if large else 0.03, num_leaves=63 if large else 31, min_child_samples=25 if large else 8, subsample=0.85, subsample_freq=1, colsample_bytree=0.75, reg_alpha=0.01, reg_lambda=0.1, random_state=seed, n_jobs=4, verbosity=-1))
    return factories

def c391_pi1m_zoo_c289_parse_models(value, available):
    requested = tuple((item.strip() for item in str(value).split(',') if item.strip()))
    if not requested:
        raise RuntimeError('No models requested')
    bad = [name for name in requested if name not in available]
    if bad:
        raise RuntimeError('Requested unavailable models ' + str(bad))
    return {name: available[name] for name in requested}

def c391_pi1m_zoo_c289_pi1m_svd_features(keys, pi1m_path, config):
    pi1m = pd.read_csv(pi1m_path, usecols=['SMILES'])
    values = pi1m['SMILES'].dropna().astype(str).tolist()
    if len(values) < 1000:
        raise RuntimeError('PI1M file unexpectedly small')
    corpus = sorted(set(values), key=str)[:int(config['pi1m_limit'])]
    corpus_plus_keys = corpus + list(keys)
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(int(config['pi1m_ngram_min']), int(config['pi1m_ngram_max'])), n_features=int(config['pi1m_hash_features']), alternate_sign=False, norm='l2', lowercase=False, dtype=np.float64)
    hashed = vectorizer.transform(corpus_plus_keys).tocsr()
    max_components = min(hashed.shape[0] - 1, hashed.shape[1] - 1)
    n_components = min(int(config['pi1m_svd_components']), max_components)
    if n_components < 8:
        raise RuntimeError('PI1M SVD component count too small')
    svd = TruncatedSVD(n_components=n_components, random_state=int(config['seed']))
    embedding = svd.fit_transform(hashed).astype(np.float64, copy=False)
    return embedding[-len(keys):]

def c391_pi1m_zoo_build_c289(data_dir, pi1m_limit=120000, pi1m_svd_components=96, pi1m_hash_features=65536, morgan_bits=768, models='ridge_100,extra_trees', seed=20260807):
    """C391 capped PI1M model zoo compact. Returns a 4,940-row (id, target) DataFrame."""
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / 'train.csv')
    test = pd.read_csv(data_path / 'test.csv')
    if list(train.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected current train schema')
    if list(test.columns) != ['id', 'smiles', 'target_type']:
        raise RuntimeError('Unexpected current test schema')
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError('Unexpected official current row count')
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [canonicalize(value) for value in frame['smiles']]
    if set(train['target_type']) != set(c391_pi1m_zoo_TARGETS) or set(test['target_type']) != set(c391_pi1m_zoo_TARGETS):
        raise RuntimeError('Unexpected target set')
    ids = test['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = build_label_pool(train, archive)
    pooled = pooled.copy()
    test2 = test.copy()
    keys = sorted(set(pooled['canonical']) | set(test2['canonical']))
    molecules = build_molecules(keys)
    descriptor, descriptor_names = descriptor_matrix(molecules, keys)
    physical, physical_names = physical_matrix(molecules, keys)
    pi1m_config = {'seed': int(seed), 'pi1m_limit': int(pi1m_limit), 'pi1m_hash_features': int(pi1m_hash_features), 'pi1m_svd_components': int(pi1m_svd_components), 'pi1m_ngram_min': 2, 'pi1m_ngram_max': 7}
    pi1m_features = c391_pi1m_zoo_c289_pi1m_svd_features(keys=keys, pi1m_path=data_path / 'PI1M.csv', config=pi1m_config)
    morgan2 = morgan_count_matrix(molecules, radius=2, bits=int(morgan_bits)).toarray().astype(np.float32)
    morgan3 = morgan_count_matrix(molecules, radius=3, bits=int(morgan_bits)).toarray().astype(np.float32)
    maccs = c391_pi1m_zoo_c289_maccs_matrix(molecules)
    cross_values, cross_available = cross_property_arrays(pooled, keys)
    base_dense = c391_pi1m_zoo_c289_sanitize_dense(np.hstack([descriptor, physical, pi1m_features, morgan2, morgan3, maccs]).astype(np.float32))
    key_to_index = {key: i for i, key in enumerate(keys)}
    predictions = np.full(len(test2), np.nan, dtype=np.float64)
    for target in c391_pi1m_zoo_TARGETS:
        target_train = pooled[pooled['target_type'].eq(target)].reset_index(drop=True)
        target_test = test2[test2['target_type'].eq(target)].reset_index(drop=False)
        train_idx = np.asarray([key_to_index[x] for x in target_train['canonical']], dtype=int)
        test_idx = np.asarray([key_to_index[x] for x in target_test['canonical']], dtype=int)
        y = target_train['target'].to_numpy(float)
        groups = target_train['canonical'].astype(str).to_numpy(object)
        dense = target_dense_features(base_dense, cross_values, cross_available, target)
        x_train = dense[train_idx]
        x_test = dense[test_idx]
        candidates = {}
        reports = {}
        selected_factories = c391_pi1m_zoo_c289_parse_models(models, c391_pi1m_zoo_c289_model_factories(target, len(y)))
        for name, factory in selected_factories.items():
            try:
                oof = c391_pi1m_zoo_c289_grouped_oof(factory, x_train, y, groups, int(seed))
                score, clipped = c391_pi1m_zoo_c289_clipped_r2(y, oof)
                candidates[name] = clipped
                reports[name] = score
            except Exception:
                continue
        if not candidates:
            raise RuntimeError('No successful models for ' + target)
        best_name = max(candidates, key=lambda name: reports[name])
        final_model = selected_factories[best_name](int(seed) + 999)
        final_model.fit(x_train, y)
        final_pred_raw = np.asarray(final_model.predict(x_test), dtype=np.float64)
        final_pred = c391_pi1m_zoo_c289_clip_to_train_range(y, final_pred_raw)
        lookup = target_train.groupby('canonical')['target'].mean().to_dict()
        for local_pos, row in enumerate(target_test.itertuples(index=False)):
            if row.canonical in lookup:
                final_pred[local_pos] = float(lookup[row.canonical])
        predictions[target_test['index'].to_numpy(int)] = final_pred
    if not np.isfinite(predictions).all():
        raise RuntimeError('Missing/non-finite final predictions')
    return pd.DataFrame({'id': test2['id'].to_numpy(int), 'target': predictions})

# ===== c327_calibrator.py =====
"""In-memory port of C327 no-archive weak-target co-test residual meta-calibrator.

Faithful, self-contained port of
the original builder script (the original round no-archive
branch).  It calibrates ei/eea/eps/nc by training a small residual model on
C282 current-only OOF predictions (the "co-test" features) and overlaying the
result on a frozen base candidate.

Strip-for-in-memory: no file I/O, no R2 paths, no digest/file-hash/manifest
code, no path guards.  The model fitting (seeds/alphas/folds/clips/weights) is
reproduced verbatim.  If a target fails its clean OOF gate it is left unchanged
from the base.

Input DataFrames (all read-only; caller-provided):
  train_df     : official train.csv  (columns: smiles, target, target_type)
  test_df      : official test.csv   (columns: id, smiles, target_type)
  base_df      : base candidate      (columns: id, target) - frozen incumbent
  c282_oof_df  : C282 OOF            (columns: canonical, target_type, target,
                                       prediction, sparse_ridge, dense_ridge,
                                       extra_trees, tanimoto_local)
"""
RDLogger.DisableLog('rdApp.*')
c327_calibrator_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
c327_calibrator_ACTIVE_TARGETS = ('ei', 'eea', 'eps', 'nc')

@dataclass(frozen=True)
class c327_calibrator_TargetConfig:
    min_clean_oof_delta: float
    min_nonnegative_folds: int
    max_low_support_loss: float
    residual_clip: float
    blend_weight: float
c327_calibrator_CONFIGS = {'ei': c327_calibrator_TargetConfig(0.003, 4, -0.003, 0.45, 0.75), 'eea': c327_calibrator_TargetConfig(0.003, 4, -0.003, 0.45, 0.75), 'eps': c327_calibrator_TargetConfig(0.003, 4, -0.003, 0.7, 0.65), 'nc': c327_calibrator_TargetConfig(0.003, 4, -0.003, 0.08, 0.65)}

def c327_calibrator_canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True)

def c327_calibrator_parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple((item.strip().lower() for item in value.split(',') if item.strip()))
    if not targets:
        raise RuntimeError('No targets requested')
    invalid = [target for target in targets if target not in c327_calibrator_ACTIVE_TARGETS]
    if invalid:
        raise RuntimeError(f'C327 only supports {c327_calibrator_ACTIVE_TARGETS}, got {invalid}')
    return targets

def c327_calibrator_pivot_predictions(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return frame.pivot_table(index='canonical', columns='target_type', values=value_column, aggfunc='mean')

def c327_calibrator_make_features(canonicals: np.ndarray, target: str, pivot: pd.DataFrame, nearest_same: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    base_names = []
    for prop in c327_calibrator_TARGETS:
        base_names += [f'pred_{prop}', f'has_{prop}']
    extra_names = ['partner_count', 'nearest_same_target_tanimoto', 'ei_from_gap', 'eea_from_gap', 'gap_residual', 'eps_minus_nc2', 'eps_from_nc2', 'nc_from_eps', 'abs_gap_residual', 'abs_eps_minus_nc2']
    names = base_names + extra_names
    for canonical in canonicals:
        values: dict[str, float] = {}
        row_values: list[float] = []
        if canonical in pivot.index:
            source = pivot.loc[canonical]
        else:
            source = pd.Series(dtype=float)
        partner_count = 0
        for prop in c327_calibrator_TARGETS:
            value = source.get(prop, np.nan)
            if pd.notna(value):
                values[prop] = float(value)
                row_values.extend([float(value), 1.0])
                if prop != target:
                    partner_count += 1
            else:
                row_values.extend([np.nan, 0.0])
        ei_from_gap = np.nan
        eea_from_gap = np.nan
        gap_residual = np.nan
        if 'eea' in values and 'egc' in values:
            ei_from_gap = values['eea'] + values['egc']
        if 'ei' in values and 'egc' in values:
            eea_from_gap = values['ei'] - values['egc']
        if all((prop in values for prop in ('ei', 'eea', 'egc'))):
            gap_residual = values['ei'] - values['eea'] - values['egc']
        eps_minus_nc2 = np.nan
        eps_from_nc2 = np.nan
        nc_from_eps = np.nan
        if 'eps' in values and 'nc' in values:
            eps_minus_nc2 = values['eps'] - values['nc'] ** 2
            eps_from_nc2 = values['nc'] ** 2 + max(eps_minus_nc2, 0.02)
            nc_from_eps = math.sqrt(max(values['eps'] - max(eps_minus_nc2, 0.02), 1.0))
        row_values.extend([float(partner_count), float(nearest_same.get(str(canonical), 0.0)), ei_from_gap, eea_from_gap, gap_residual, eps_minus_nc2, eps_from_nc2, nc_from_eps, abs(gap_residual) if pd.notna(gap_residual) else np.nan, abs(eps_minus_nc2) if pd.notna(eps_minus_nc2) else np.nan])
        rows.append(row_values)
    return (np.asarray(rows, dtype=np.float64), names)

def c327_calibrator_morgan_fps(canonicals: list[str]) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps: dict[str, Any] = {}
    for canonical in sorted(set(canonicals)):
        mol = Chem.MolFromSmiles(str(canonical))
        if mol is None:
            continue
        fps[str(canonical)] = generator.GetFingerprint(mol)
    return fps

def c327_calibrator_nearest_similarity(query: list[str], train_pool: list[str]) -> dict[str, float]:
    fps = c327_calibrator_morgan_fps(query + train_pool)
    train_fps = [fps[canonical] for canonical in train_pool if canonical in fps]
    if not train_fps:
        return {str(canonical): 0.0 for canonical in query}
    result: dict[str, float] = {}
    for canonical in query:
        fp = fps.get(str(canonical))
        if fp is None:
            result[str(canonical)] = 0.0
        else:
            result[str(canonical)] = float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return result

def c327_calibrator_make_model(name: str) -> Any:
    if name == 'ridge10':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0))
    if name == 'ridge100':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    if name == 'huber':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000))
    raise KeyError(name)

def c327_calibrator_evaluate_target(target: str, oof: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = c327_calibrator_CONFIGS[target]
    target_oof = oof[oof['target_type'] == target].reset_index(drop=True)
    train_pool = oof[oof['target_type'] == target]['canonical'].astype(str).tolist()
    nearest_train = c327_calibrator_nearest_similarity(target_oof['canonical'].astype(str).tolist(), train_pool)
    oof_pivot = c327_calibrator_pivot_predictions(oof, 'prediction')
    x, feature_names = c327_calibrator_make_features(target_oof['canonical'].astype(str).to_numpy(), target, oof_pivot, nearest_train)
    y = target_oof['target'].to_numpy(float)
    parent = target_oof['prediction'].to_numpy(float)
    residual = y - parent
    groups = target_oof['canonical'].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    folds = np.full(len(y), -1, dtype=np.int64)
    for fold, (_, va) in enumerate(splitter.split(x, y, groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError(f'Fold assignment failed for {target}')
    parent_r2 = float(r2_score(y, parent))
    arms: dict[str, dict[str, Any]] = {}
    oof_preds: dict[str, np.ndarray] = {}
    for name in ('ridge10', 'ridge100', 'huber'):
        pred_residual = np.full(len(y), np.nan, dtype=np.float64)
        fold_deltas: list[float] = []
        for fold in sorted(set(folds)):
            tr = np.flatnonzero(folds != fold)
            va = np.flatnonzero(folds == fold)
            model = c327_calibrator_make_model(name)
            model.fit(x[tr], residual[tr])
            raw_delta = np.asarray(model.predict(x[va]), dtype=np.float64)
            raw_delta = np.clip(raw_delta, -cfg.residual_clip, cfg.residual_clip)
            pred = parent[va] + cfg.blend_weight * raw_delta
            pred_residual[va] = pred
            fold_deltas.append(float(r2_score(y[va], pred) - r2_score(y[va], parent[va])))
        if not np.isfinite(pred_residual).all():
            raise RuntimeError(f'Non-finite OOF predictions for {target}/{name}')
        oof_preds[name] = pred_residual
        partner_counts = np.nan_to_num(x[:, feature_names.index('partner_count')], nan=0.0)
        low_support = partner_counts <= 1.0
        low_support_delta = None
        if int(np.sum(low_support)) >= 10:
            low_support_delta = float(r2_score(y[low_support], pred_residual[low_support]) - r2_score(y[low_support], parent[low_support]))
        arms[name] = {'oof_r2': float(r2_score(y, pred_residual)), 'delta_vs_parent_oof': float(r2_score(y, pred_residual) - parent_r2), 'fold_deltas': fold_deltas, 'nonnegative_folds': int(sum((delta >= 0.0 for delta in fold_deltas))), 'low_support_delta': low_support_delta}
    mean_pred = np.mean(np.column_stack([oof_preds['ridge10'], oof_preds['ridge100'], oof_preds['huber']]), axis=1)
    fold_deltas = []
    for fold in sorted(set(folds)):
        va = np.flatnonzero(folds == fold)
        fold_deltas.append(float(r2_score(y[va], mean_pred[va]) - r2_score(y[va], parent[va])))
    partner_counts = np.nan_to_num(x[:, feature_names.index('partner_count')], nan=0.0)
    low_support = partner_counts <= 1.0
    low_support_delta = None
    if int(np.sum(low_support)) >= 10:
        low_support_delta = float(r2_score(y[low_support], mean_pred[low_support]) - r2_score(y[low_support], parent[low_support]))
    arms['mean3'] = {'oof_r2': float(r2_score(y, mean_pred)), 'delta_vs_parent_oof': float(r2_score(y, mean_pred) - parent_r2), 'fold_deltas': fold_deltas, 'nonnegative_folds': int(sum((delta >= 0.0 for delta in fold_deltas))), 'low_support_delta': low_support_delta}
    selected_arm = max(arms, key=lambda item: arms[item]['oof_r2'])
    selected = arms[selected_arm]
    pass_gate = bool(selected['delta_vs_parent_oof'] >= cfg.min_clean_oof_delta and selected['nonnegative_folds'] >= cfg.min_nonnegative_folds and (selected['low_support_delta'] is None or selected['low_support_delta'] >= cfg.max_low_support_loss))
    target_test = test[test['target_type'] == target].copy()
    test_pivot = c327_calibrator_pivot_predictions(test.rename(columns={'base_prediction': 'prediction'}), 'prediction')
    nearest_test = c327_calibrator_nearest_similarity(target_test['canonical'].astype(str).tolist(), train_pool)
    test_x, _ = c327_calibrator_make_features(target_test['canonical'].astype(str).to_numpy(), target, test_pivot, nearest_test)
    base_values = target_test['base_prediction'].to_numpy(float)
    if pass_gate:
        if selected_arm == 'mean3':
            full_deltas = []
            for name in ('ridge10', 'ridge100', 'huber'):
                model = c327_calibrator_make_model(name)
                model.fit(x, residual)
                full_deltas.append(np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -cfg.residual_clip, cfg.residual_clip))
            raw_delta = np.mean(np.column_stack(full_deltas), axis=1)
        else:
            model = c327_calibrator_make_model(selected_arm)
            model.fit(x, residual)
            raw_delta = np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -cfg.residual_clip, cfg.residual_clip)
        overlay = base_values + cfg.blend_weight * raw_delta
        changed = np.ones(len(base_values), dtype=bool)
    else:
        overlay = base_values.copy()
        changed = np.zeros(len(base_values), dtype=bool)
    report = {'target': target, 'parent_oof_r2': parent_r2, 'arms': arms, 'selected_arm': selected_arm, 'selected_oof_r2': float(selected['oof_r2']), 'selected_delta_vs_parent_oof': float(selected['delta_vs_parent_oof']), 'clean_oof_gate_pass': pass_gate, 'train_rows': int(len(target_oof)), 'test_rows': int(len(target_test)), 'changed_rows': int(np.sum(changed)), 'feature_names': feature_names}
    return (overlay, report)

def c327_calibrator_cotest_meta_calibrator(train_df, test_df, base_df, c282_oof_df, targets='ei,eea,eps,nc') -> pd.DataFrame:
    """Calibrate ei/eea/eps/nc on top of base_df using C282 OOF co-test features.

    Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
    """
    requested = c327_calibrator_parse_targets(targets)
    train = train_df.copy()
    test = test_df.copy()
    oof = c282_oof_df.copy()
    base = base_df.copy()
    if list(train.columns) != ['smiles', 'target', 'target_type'] or len(train) != 7409:
        raise RuntimeError('Unexpected train.csv schema/count')
    if list(test.columns) != ['id', 'smiles', 'target_type'] or len(test) != 4940:
        raise RuntimeError('Unexpected test.csv schema/count')
    expected_oof_columns = ['canonical', 'target_type', 'target', 'prediction', 'sparse_ridge', 'dense_ridge', 'extra_trees', 'tanimoto_local']
    if list(oof.columns) != expected_oof_columns:
        raise RuntimeError('Unexpected C282 OOF schema')
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    oof['target_type'] = oof['target_type'].astype(str).str.lower()
    train['canonical'] = [c327_calibrator_canonical_smiles(value) for value in train['smiles']]
    test['canonical'] = [c327_calibrator_canonical_smiles(value) for value in test['smiles']]
    if set(train['target_type']) != set(c327_calibrator_TARGETS) or set(test['target_type']) != set(c327_calibrator_TARGETS):
        raise RuntimeError('Unexpected target set')
    if test['id'].duplicated().any() or not np.array_equal(test['id'].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    for target in c327_calibrator_ACTIVE_TARGETS:
        oof_rows = len(oof[oof['target_type'] == target])
        train_rows = len(train[train['target_type'] == target])
        if oof_rows < max(20, int(0.95 * train_rows)):
            raise RuntimeError(f'C282 OOF row count too low for {target}: {oof_rows}/{train_rows}')
    if list(base.columns) != ['id', 'target'] or len(base) != len(test):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), test['id'].to_numpy(int)):
        raise RuntimeError('Invalid base ID order')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base contains non-finite predictions')
    test['base_prediction'] = base['target'].to_numpy(float)
    ids = test['id'].to_numpy(int)
    result = base['target'].to_numpy(float).copy()
    for target in requested:
        overlay, _ = c327_calibrator_evaluate_target(target, oof, test)
        mask = test['target_type'].to_numpy(str) == target
        if int(np.sum(mask)) != len(overlay):
            raise RuntimeError(f'Target alignment failed for {target}')
        result[mask] = overlay
    if not np.isfinite(result).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})

# ===== c346_c347_nonlinear.py =====
"""In-memory port of C346/C347 branch-guarded nonlinear co-test residual calibrator.

Faithful, self-contained port of
the original builder script (the original round
without_archive branch).  It is a bounded continuation of the C327/C332 co-test
residual family that adds low-variance nonlinear residual arms
(kernel-ridge-linear, KNN, ExtraTrees, RandomForest) on the fold-local OOF
feature table.  If a target fails its clean OOF gate it is left unchanged from
the base candidate.

Strip-for-in-memory (deliberate, non-behavioral deviations):
  * No file I/O / pd.read_csv / to_csv / Path / open; inputs arrive as DataFrames.
  * No digest module/digest/digest, manifest, path-guard, protocol, env-var, or argparse code.
  * No the original round paths (the original round experiment paths/, the original round output paths/, the verification panel).  The
    C346 DEFAULTS/C327/C332 base+oof path strings are dropped (caller supplies
    base_df / c282_oof_df directly).
  * Dropped changed_ids_digest (hash) from the per-target report; numeric gate
    fields and feature_names are retained.

Model/feature hyperparameters reproduced verbatim (do not change):
  * Arms: ridge100 (Ridge alpha=100), krr_linear (KernelRidge alpha=20 linear),
    knn5 (k=5 distance-weighted minkowski), knn15 (k=15 distance-weighted
    minkowski), extra_trees (n_estimators=320, max_features=0.80, random_state=346,
    min_samples_leaf=gate.tree_leaf), random_forest (n_estimators=260,
    max_features=0.85, random_state=347, min_samples_leaf=gate.tree_leaf).
    All arms sit inside make_pipeline(SimpleImputer(median, keep_empty_features=True),
    StandardScaler()).
  * mean arm = mean of ridge100 + krr_linear + extra_trees ("mean_ridge_krr_et").
  * selected_arm = argmax oof R2 across all arms + the mean arm; gate requires
    delta_vs_parent_oof >= min_delta AND nonnegative_folds >= min_nonnegative_folds
    AND (low_support_delta is None or >= max_low_support_loss).
  * GroupKFold(n_splits = min(5, #unique canonical groups)).
  * residual clip + blend_weight per-target from GATES (see below).
  * Features: 7-target pivot cross-property co-test table (pred_*/has_* + 10 physics
    derived features); Morgan radius=2 fpSize=2048 for nearest-tanimoto.

Input DataFrames (all read-only; caller-provided):
  train_df     : official train.csv  (columns: smiles, target, target_type)
  test_df      : official test.csv   (columns: id, smiles, target_type)
  base_df      : base candidate      (columns: id, target) - frozen incumbent
  c282_oof_df  : C282 OOF            (columns: canonical, target_type, target, prediction)

Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
"""
RDLogger.DisableLog('rdApp.*')
c346_c347_nonlinear_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

@dataclass(frozen=True)
class c346_c347_nonlinear_TargetGate:
    min_delta: float
    min_nonnegative_folds: int
    max_low_support_loss: float
    residual_clip: float
    blend_weight: float
    tree_leaf: int
c346_c347_nonlinear_GATES = {'tg': c346_c347_nonlinear_TargetGate(0.002, 4, -0.004, 35.0, 0.35, 12), 'egc': c346_c347_nonlinear_TargetGate(0.001, 4, -0.004, 0.65, 0.45, 10), 'egb': c346_c347_nonlinear_TargetGate(0.001, 4, -0.004, 0.65, 0.45, 6), 'ei': c346_c347_nonlinear_TargetGate(0.001, 3, -0.012, 0.45, 0.35, 4), 'eea': c346_c347_nonlinear_TargetGate(0.001, 3, -0.012, 0.45, 0.35, 4), 'eps': c346_c347_nonlinear_TargetGate(0.001, 3, -0.012, 0.7, 0.35, 4), 'nc': c346_c347_nonlinear_TargetGate(0.001, 3, -0.012, 0.08, 0.35, 4)}

def c346_c347_nonlinear_canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True)

def c346_c347_nonlinear_parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple((item.strip().lower() for item in value.split(',') if item.strip()))
    if not targets:
        raise RuntimeError('No targets requested')
    invalid = [target for target in targets if target not in c346_c347_nonlinear_TARGETS]
    if invalid:
        raise RuntimeError(f'Unknown targets: {invalid}')
    return targets

def c346_c347_nonlinear_pivot_predictions(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return frame.pivot_table(index='canonical', columns='target_type', values=value_column, aggfunc='mean')

def c346_c347_nonlinear_make_features(canonicals: np.ndarray, target: str, pivot: pd.DataFrame, nearest_same: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    base_names = []
    for prop in c346_c347_nonlinear_TARGETS:
        base_names += [f'pred_{prop}', f'has_{prop}']
    extra_names = ['partner_count', 'nearest_same_target_tanimoto', 'ei_from_gap', 'eea_from_gap', 'gap_residual', 'eps_minus_nc2', 'eps_from_nc2', 'nc_from_eps', 'abs_gap_residual', 'abs_eps_minus_nc2']
    names = base_names + extra_names
    for canonical in canonicals:
        values: dict[str, float] = {}
        row_values: list[float] = []
        if canonical in pivot.index:
            source = pivot.loc[canonical]
        else:
            source = pd.Series(dtype=float)
        partner_count = 0
        for prop in c346_c347_nonlinear_TARGETS:
            value = source.get(prop, np.nan)
            if pd.notna(value):
                values[prop] = float(value)
                row_values.extend([float(value), 1.0])
                if prop != target:
                    partner_count += 1
            else:
                row_values.extend([np.nan, 0.0])
        ei_from_gap = np.nan
        eea_from_gap = np.nan
        gap_residual = np.nan
        if 'eea' in values and 'egc' in values:
            ei_from_gap = values['eea'] + values['egc']
        if 'ei' in values and 'egc' in values:
            eea_from_gap = values['ei'] - values['egc']
        if all((prop in values for prop in ('ei', 'eea', 'egc'))):
            gap_residual = values['ei'] - values['eea'] - values['egc']
        eps_minus_nc2 = np.nan
        eps_from_nc2 = np.nan
        nc_from_eps = np.nan
        if 'eps' in values and 'nc' in values:
            eps_minus_nc2 = values['eps'] - values['nc'] ** 2
            eps_from_nc2 = values['nc'] ** 2 + max(eps_minus_nc2, 0.02)
            nc_from_eps = math.sqrt(max(values['eps'] - max(eps_minus_nc2, 0.02), 1.0))
        row_values.extend([float(partner_count), float(nearest_same.get(str(canonical), 0.0)), ei_from_gap, eea_from_gap, gap_residual, eps_minus_nc2, eps_from_nc2, nc_from_eps, abs(gap_residual) if pd.notna(gap_residual) else np.nan, abs(eps_minus_nc2) if pd.notna(eps_minus_nc2) else np.nan])
        rows.append(row_values)
    return (np.asarray(rows, dtype=np.float64), names)

def c346_c347_nonlinear_morgan_fps(canonicals: list[str]) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps: dict[str, Any] = {}
    for canonical in sorted(set(canonicals)):
        mol = Chem.MolFromSmiles(str(canonical))
        if mol is None:
            continue
        fps[str(canonical)] = generator.GetFingerprint(mol)
    return fps

def c346_c347_nonlinear_nearest_similarity(query: list[str], train_pool: list[str]) -> dict[str, float]:
    fps = c346_c347_nonlinear_morgan_fps(query + train_pool)
    train_fps = [fps[canonical] for canonical in train_pool if canonical in fps]
    if not train_fps:
        return {str(canonical): 0.0 for canonical in query}
    result: dict[str, float] = {}
    for canonical in query:
        fp = fps.get(str(canonical))
        if fp is None:
            result[str(canonical)] = 0.0
        else:
            result[str(canonical)] = float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return result

def c346_c347_nonlinear_make_model(name: str, gate: c346_c347_nonlinear_TargetGate) -> Any:
    if name == 'ridge100':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    if name == 'krr_linear':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), KernelRidge(alpha=20.0, kernel='linear'))
    if name == 'knn5':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski'))
    if name == 'knn15':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights='distance', metric='minkowski'))
    if name == 'extra_trees':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), ExtraTreesRegressor(n_estimators=320, min_samples_leaf=gate.tree_leaf, max_features=0.8, random_state=346, n_jobs=-1))
    if name == 'random_forest':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), RandomForestRegressor(n_estimators=260, min_samples_leaf=gate.tree_leaf, max_features=0.85, random_state=347, n_jobs=-1))
    raise KeyError(name)

def c346_c347_nonlinear_evaluate_target(target: str, oof: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    gate = c346_c347_nonlinear_GATES[target]
    target_oof = oof[oof['target_type'] == target].reset_index(drop=True)
    if len(target_oof) < 20:
        raise RuntimeError(f'Too few OOF rows for {target}: {len(target_oof)}')
    train_pool = oof[oof['target_type'] == target]['canonical'].astype(str).tolist()
    nearest_train = c346_c347_nonlinear_nearest_similarity(target_oof['canonical'].astype(str).tolist(), train_pool)
    oof_pivot = c346_c347_nonlinear_pivot_predictions(oof, 'prediction')
    x, feature_names = c346_c347_nonlinear_make_features(target_oof['canonical'].astype(str).to_numpy(), target, oof_pivot, nearest_train)
    y = target_oof['target'].to_numpy(float)
    parent = target_oof['prediction'].to_numpy(float)
    residual = y - parent
    groups = target_oof['canonical'].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    folds = np.full(len(y), -1, dtype=np.int64)
    for fold, (_, va) in enumerate(splitter.split(x, y, groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError(f'Fold assignment failed for {target}')
    parent_r2 = float(r2_score(y, parent))
    partner_counts = np.nan_to_num(x[:, feature_names.index('partner_count')], nan=0.0)
    low_support = partner_counts <= 1.0
    arms: dict[str, dict[str, Any]] = {}
    oof_preds: dict[str, np.ndarray] = {}
    for name in ('ridge100', 'krr_linear', 'knn5', 'knn15', 'extra_trees', 'random_forest'):
        pred = np.full(len(y), np.nan, dtype=np.float64)
        fold_deltas: list[float] = []
        for fold in sorted(set(folds)):
            tr = np.flatnonzero(folds != fold)
            va = np.flatnonzero(folds == fold)
            model = c346_c347_nonlinear_make_model(name, gate)
            model.fit(x[tr], residual[tr])
            delta = np.clip(np.asarray(model.predict(x[va]), dtype=np.float64), -gate.residual_clip, gate.residual_clip)
            pred[va] = parent[va] + gate.blend_weight * delta
            fold_deltas.append(float(r2_score(y[va], pred[va]) - r2_score(y[va], parent[va])))
        if not np.isfinite(pred).all():
            raise RuntimeError(f'Non-finite OOF predictions for {target}/{name}')
        oof_preds[name] = pred
        low_support_delta = None
        if int(np.sum(low_support)) >= 10:
            low_support_delta = float(r2_score(y[low_support], pred[low_support]) - r2_score(y[low_support], parent[low_support]))
        arms[name] = {'oof_r2': float(r2_score(y, pred)), 'delta_vs_parent_oof': float(r2_score(y, pred) - parent_r2), 'fold_deltas': fold_deltas, 'nonnegative_folds': int(sum((delta >= 0.0 for delta in fold_deltas))), 'low_support_delta': low_support_delta}
    mean_names = ('ridge100', 'krr_linear', 'extra_trees')
    mean_pred = np.mean(np.column_stack([oof_preds[name] for name in mean_names]), axis=1)
    fold_deltas = []
    for fold in sorted(set(folds)):
        va = np.flatnonzero(folds == fold)
        fold_deltas.append(float(r2_score(y[va], mean_pred[va]) - r2_score(y[va], parent[va])))
    low_support_delta = None
    if int(np.sum(low_support)) >= 10:
        low_support_delta = float(r2_score(y[low_support], mean_pred[low_support]) - r2_score(y[low_support], parent[low_support]))
    arms['mean_ridge_krr_et'] = {'oof_r2': float(r2_score(y, mean_pred)), 'delta_vs_parent_oof': float(r2_score(y, mean_pred) - parent_r2), 'fold_deltas': fold_deltas, 'nonnegative_folds': int(sum((delta >= 0.0 for delta in fold_deltas))), 'low_support_delta': low_support_delta}
    selected_arm = max(arms, key=lambda item: arms[item]['oof_r2'])
    selected = arms[selected_arm]
    pass_gate = bool(selected['delta_vs_parent_oof'] >= gate.min_delta and selected['nonnegative_folds'] >= gate.min_nonnegative_folds and (selected['low_support_delta'] is None or selected['low_support_delta'] >= gate.max_low_support_loss))
    target_test = test[test['target_type'] == target].copy()
    test_pivot = c346_c347_nonlinear_pivot_predictions(test.rename(columns={'base_prediction': 'prediction'}), 'prediction')
    nearest_test = c346_c347_nonlinear_nearest_similarity(target_test['canonical'].astype(str).tolist(), train_pool)
    test_x, _ = c346_c347_nonlinear_make_features(target_test['canonical'].astype(str).to_numpy(), target, test_pivot, nearest_test)
    base_values = target_test['base_prediction'].to_numpy(float)
    if pass_gate:
        if selected_arm == 'mean_ridge_krr_et':
            full_deltas = []
            for name in mean_names:
                model = c346_c347_nonlinear_make_model(name, gate)
                model.fit(x, residual)
                full_deltas.append(np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -gate.residual_clip, gate.residual_clip))
            raw_delta = np.mean(np.column_stack(full_deltas), axis=1)
        else:
            model = c346_c347_nonlinear_make_model(selected_arm, gate)
            model.fit(x, residual)
            raw_delta = np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -gate.residual_clip, gate.residual_clip)
        overlay = base_values + gate.blend_weight * raw_delta
        changed = np.ones(len(base_values), dtype=bool)
    else:
        overlay = base_values.copy()
        changed = np.zeros(len(base_values), dtype=bool)
    return (overlay, {'target': target, 'parent_oof_r2': parent_r2, 'arms': arms, 'selected_arm': selected_arm, 'selected_oof_r2': float(selected['oof_r2']), 'selected_delta_vs_parent_oof': float(selected['delta_vs_parent_oof']), 'clean_oof_gate_pass': pass_gate, 'gate': {'min_delta': gate.min_delta, 'min_nonnegative_folds': gate.min_nonnegative_folds, 'max_low_support_loss': gate.max_low_support_loss, 'residual_clip': gate.residual_clip, 'blend_weight': gate.blend_weight, 'tree_leaf': gate.tree_leaf}, 'train_rows': int(len(target_oof)), 'test_rows': int(len(target_test)), 'changed_rows': int(np.sum(changed)), 'feature_names': feature_names})

def c346_c347_nonlinear_nonlinear_cotest_calibrator(train_df, test_df, base_df, c282_oof_df, targets: str='egc,ei,eps,nc') -> pd.DataFrame:
    """C346/C347 branch-guarded nonlinear co-test residual calibrator (without_archive).

    Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
    """
    requested = c346_c347_nonlinear_parse_targets(targets)
    train = train_df.copy()
    test = test_df.copy()
    oof = c282_oof_df.copy()
    base = base_df.copy()
    if list(train.columns) != ['smiles', 'target', 'target_type'] or len(train) != 7409:
        raise RuntimeError('Unexpected train.csv schema/count')
    if list(test.columns) != ['id', 'smiles', 'target_type'] or len(test) != 4940:
        raise RuntimeError('Unexpected test.csv schema/count')
    if list(base.columns) != ['id', 'target'] or len(base) != len(test):
        raise RuntimeError('Invalid base schema')
    required_oof = ['canonical', 'target_type', 'target', 'prediction']
    if not all((column in oof.columns for column in required_oof)):
        raise RuntimeError('Unexpected C282 OOF schema')
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    oof['target_type'] = oof['target_type'].astype(str).str.lower()
    train['canonical'] = [c346_c347_nonlinear_canonical_smiles(value) for value in train['smiles']]
    test['canonical'] = [c346_c347_nonlinear_canonical_smiles(value) for value in test['smiles']]
    if set(train['target_type']) != set(c346_c347_nonlinear_TARGETS) or set(test['target_type']) != set(c346_c347_nonlinear_TARGETS):
        raise RuntimeError('Unexpected target set')
    if test['id'].duplicated().any() or not np.array_equal(test['id'].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), test['id'].to_numpy(int)):
        raise RuntimeError('Invalid base ID order')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base contains non-finite predictions')
    test['base_prediction'] = base['target'].to_numpy(float)
    ids = test['id'].to_numpy(int)
    result = test['base_prediction'].to_numpy(float).copy()
    for target in requested:
        overlay, _report = c346_c347_nonlinear_evaluate_target(target, oof, test)
        mask = test['target_type'].to_numpy(str) == target
        if int(np.sum(mask)) != len(overlay):
            raise RuntimeError(f'Target alignment failed for {target}')
        result[mask] = overlay
    if not np.isfinite(result).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})

# ===== c350_epsnc.py =====
"""In-memory port of C350 no-archive joint EPS/NC ionic consistency source.

Faithful, self-contained port of
the original builder script (the original round
no-archive branch).  For canonical structures that appear as BOTH `eps` and
`nc` rows in the official test set, it fits an ionic residual
(`ionic = eps - nc**2`) model from the official current train EPS/NC pairs and
jointly reconciles the two base predictions under `eps = nc**2 + ionic`,
pulling each toward the physical solution by `pull`.

Strip-for-in-memory (deliberate non-behavioral deviations): no file I/O, no
the original round paths, no digest/file-hash/manifest code, no path guards, no argparse,
no json logging.  These were the only uses of the stripped names
(train_sha/test_sha/record/inputs) in the sanitized source; the model fitting,
feature building, fold audit and solve logic are reproduced verbatim.

Model/features (reproduced exactly):
  * polar_block(): 17 SMARTS substructure counts normalized by heavy-atom
    count + TPSA/NumHDonors/NumHAcceptors/FractionCSP3/NumRotatableBonds/
    MolMR/MolLogP/CalcNumAromaticRings (each normalized by heavy-atom count
    except FractionCSP3) = 26 features.
  * fit_ionic(): ExtraTreesRegressor(n_estimators=800, min_samples_leaf=leaf,
    random_state=20260808, n_jobs=-1).
  * GroupKFold(n_splits=min(5, n_unique(no-stereo groups))) ionic OOF audit.
  * MIN_IONIC = 0.02 floor on ionic predictions.
  * solve_pair(): 40-step Newton + 241-point grid line-search over
    n in [1.0, 2.5], minimizing
    weight_eps*(n**2+ionic-eps_base)**2 + weight_nc*(n-nc_base)**2.

Input DataFrames (all read-only; caller-provided):
  train_df : official train.csv  (columns: smiles, target, target_type)
  test_df  : official test.csv   (columns: id, smiles, target_type)
  base_df  : base candidate      (columns: id, target) - frozen incumbent

Parameters (source CLI flags / reference-run invocation):
  pull        : 0.5  (C350); 0.75 for C353
  ionic_leaf  : 2
  weight_eps  : 1.0
  weight_nc   : 1.0

Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
"""
c350_epsnc_MIN_IONIC = 0.02
c350_epsnc_POLAR_SMARTS = {'CF': '[#6][F]', 'CCl': '[#6][Cl]', 'ester': 'C(=O)O', 'carbonyl': '[CX3]=[OX1]', 'ether': '[OD2]([#6])[#6]', 'OH': '[OX2H]', 'nitrile': 'C#N', 'amide': 'C(=O)N', 'NH': '[NX3;H1,H2]', 'sulfone': 'S(=O)(=O)', 'thioether': '[#16X2]', 'aromatic_N': 'n', 'aromatic_O': 'o', 'aromatic_S': 's', 'imide': 'C(=O)NC(=O)', 'siloxane': '[Si][O]', 'phosphate': 'P=O', 'urethane': 'NC(=O)O'}
c350_epsnc_PATS = {name: Chem.MolFromSmarts(smarts) for name, smarts in c350_epsnc_POLAR_SMARTS.items()}

def c350_epsnc_canonical_smiles(smiles) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True)

def c350_epsnc_no_stereo(smiles) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def c350_epsnc_polar_block(cans) -> np.ndarray:
    rows = []
    for can in cans:
        mol = Chem.MolFromSmiles(str(can))
        if mol is None:
            raise RuntimeError(f'Invalid canonical SMILES: {can}')
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in c350_epsnc_PATS.values()]
        row += [Descriptors.TPSA(mol) / heavy, Descriptors.NumHDonors(mol) / heavy, Descriptors.NumHAcceptors(mol) / heavy, Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol) / heavy, Crippen.MolMR(mol) / heavy, Crippen.MolLogP(mol) / heavy, rdMolDescriptors.CalcNumAromaticRings(mol) / heavy]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)

def c350_epsnc_fit_ionic(cans, ionic, leaf) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(n_estimators=800, min_samples_leaf=leaf, random_state=20260808, n_jobs=-1)
    model.fit(c350_epsnc_polar_block(cans), ionic)
    return model

def c350_epsnc_solve_pair(eps_base, nc_base, ionic, *, weight_eps, weight_nc):
    ionic = max(float(ionic), c350_epsnc_MIN_IONIC)
    n = float(np.clip(nc_base, 1.0, 2.5))
    for _ in range(40):
        diff = n * n + ionic - eps_base
        grad = 4.0 * weight_eps * n * diff + 2.0 * weight_nc * (n - nc_base)
        hess = 4.0 * weight_eps * (3.0 * n * n + ionic - eps_base) + 2.0 * weight_nc
        if abs(hess) < 1e-12:
            break
        step = grad / hess
        n_next = float(np.clip(n - step, 1.0, 2.5))
        if abs(n_next - n) < 1e-10:
            n = n_next
            break
        n = n_next
    grid = np.linspace(max(1.0, n - 0.12), min(2.5, n + 0.12), 241)
    obj = weight_eps * (grid * grid + ionic - eps_base) ** 2 + weight_nc * (grid - nc_base) ** 2
    n = float(grid[int(np.argmin(obj))])
    return (float(n * n + ionic), n)

def c350_epsnc_c350_eps_nc_consistency(train_df, test_df, base_df, pull=0.5, ionic_leaf=2, weight_eps=1.0, weight_nc=1.0):
    """Joint EPS/NC ionic-consistency reconciliation over a frozen base candidate.

    Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
    """
    pull = float(pull)
    if not 0.0 <= pull <= 1.0:
        raise RuntimeError('pull must be in [0, 1]')
    ionic_leaf = int(ionic_leaf)
    weight_eps = float(weight_eps)
    weight_nc = float(weight_nc)
    train = train_df.copy()
    test = test_df.copy().reset_index(drop=True)
    base = base_df.copy()
    for frame in (train, test):
        frame['tt'] = frame['target_type'].astype(str).str.lower()
        frame['canon'] = [c350_epsnc_canonical_smiles(value) for value in frame['smiles']]
    ids = test['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base ID order')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base contains non-finite predictions')
    test['base_prediction'] = base['target'].to_numpy(float)
    wide = train.pivot_table(index='canon', columns='tt', values='target', aggfunc='mean')
    pairs = wide[['eps', 'nc']].dropna()
    if len(pairs) < 50:
        raise RuntimeError('Insufficient current official EPS/NC pairs')
    ionic = pairs['eps'].to_numpy(float) - pairs['nc'].to_numpy(float) ** 2
    if float(np.min(ionic)) < 0.0:
        raise RuntimeError('Unexpected negative ionic residual in official train pairs')
    groups = np.asarray([c350_epsnc_no_stereo(value) for value in pairs.index], dtype=object)
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    ionic_oof = np.full(len(pairs), np.nan, dtype=np.float64)
    for tr, va in folds.split(pairs.index.to_numpy(), ionic, groups=groups):
        model = c350_epsnc_fit_ionic(pairs.index[tr].tolist(), ionic[tr], ionic_leaf)
        ionic_oof[va] = model.predict(c350_epsnc_polar_block(pairs.index[va]))
    ionic_oof = np.maximum(ionic_oof, c350_epsnc_MIN_IONIC)
    eps_phys_oof = pairs['nc'].to_numpy(float) ** 2 + ionic_oof
    nc_phys_oof = np.sqrt(np.maximum(pairs['eps'].to_numpy(float) - ionic_oof, 1.0))
    _oof_report = {'pair_rows': int(len(pairs)), 'ionic_oof_r2': float(r2_score(ionic, ionic_oof)), 'eps_from_true_nc_oof_r2': float(r2_score(pairs['eps'].to_numpy(float), eps_phys_oof)), 'nc_from_true_eps_oof_r2': float(r2_score(pairs['nc'].to_numpy(float), nc_phys_oof))}
    model = c350_epsnc_fit_ionic(pairs.index.tolist(), ionic, ionic_leaf)
    test_pivot = test.pivot_table(index='canon', columns='tt', values='base_prediction', aggfunc='mean')
    result = test['base_prediction'].to_numpy(float).copy()
    pair_canons = [c for c in test_pivot.index if 'eps' in test_pivot.columns and 'nc' in test_pivot.columns and pd.notna(test_pivot.loc[c].get('eps', np.nan)) and pd.notna(test_pivot.loc[c].get('nc', np.nan))]
    if pair_canons:
        ionic_pred = pd.Series(np.maximum(model.predict(c350_epsnc_polar_block(pair_canons)), c350_epsnc_MIN_IONIC), index=pair_canons)
    else:
        ionic_pred = pd.Series(dtype=float)
    solved = {}
    for canon in pair_canons:
        eps_base = float(test_pivot.loc[canon, 'eps'])
        nc_base = float(test_pivot.loc[canon, 'nc'])
        eps_star, nc_star = c350_epsnc_solve_pair(eps_base, nc_base, float(ionic_pred.loc[canon]), weight_eps=weight_eps, weight_nc=weight_nc)
        solved[str(canon)] = ((1.0 - pull) * eps_base + pull * eps_star, (1.0 - pull) * nc_base + pull * nc_star)
    for row_idx, row in test.iterrows():
        canon = str(row['canon'])
        if canon not in solved:
            continue
        target = str(row['tt'])
        if target == 'eps':
            result[int(row_idx)] = solved[canon][0]
        elif target == 'nc':
            result[int(row_idx)] = solved[canon][1]
    if not np.isfinite(result).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})
c350_epsnc___all__ = ['c350_eps_nc_consistency', 'polar_block', 'fit_ionic', 'solve_pair']

# ===== c374_ei_eht.py =====
"""In-memory port of C374 no-archive EI EHT current-only residual source.

Faithful, self-contained port of
the original builder script (the original round no-archive
branch).  It computes deterministic RDKit/YAeHMOP extended-Hueckel (EHT)
orbital+charge features from the official current EI SMILES (H-capped and
ring-closed variants), fits a structure-grouped Ridge residual model against
the C282 current-only OOF EI predictions, and deploys the EI residual over a
frozen base candidate.

Strip-for-in-memory (deliberate non-behavioral deviations): no file I/O, no
the original round paths, no digest/file-hash/manifest code, no path guards, no argparse,
no json logging, no importlib module loading (the C258 EHT source is inlined
verbatim).  The model fitting (SEED/ridge-alpha/folds/weight/bootstrap,
GroupKFold) and the EHT orbital feature computation are reproduced verbatim.

IMPORTANT: stable_seed() uses digest module.digest purely as a deterministic
string->int SEED derivation for AllChem.EmbedMolecule (it is the model-fitting
seed, NOT file-integrity hashing).  Changing it would change the 3D conformer
seeds and therefore the EHT features/predictions, so it is preserved verbatim.

EHT features (reproduced exactly, 41 columns): per variant (hcap, ring) an
18-vector [homo, lumo, gap, homo_minus_1, lumo_plus_1, homo_spacing,
lumo_spacing, energy_min/max/mean/std/q25/q75, charge_min/max/mean/std/abs_sum]
+ a supported flag, plus 3 diffs (hcap-minus-ring homo/lumo/gap).

Input DataFrames (all read-only; caller-provided):
  train_df     : official train.csv  (columns: smiles, target, target_type)
  test_df      : official test.csv   (columns: id, smiles, target_type)
  base_df      : base candidate      (columns: id, target) - frozen incumbent
  c282_oof_df  : C282 OOF            (columns: canonical, target_type, target,
                                       prediction) - only EI rows are used

Parameters (source CLI flags / reference-run invocation):
  residual_weight : 0.35
  ridge_alpha     : 60.0

Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
The clean-gate verdict is computed for parity with the source record but does
NOT suppress deployment (C374 always applies the EI residual).
"""
RDLogger.DisableLog('rdApp.*')
c374_ei_eht_SEED = 20260808

def c374_ei_eht_canonicalize(smiles) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
c374_ei_eht_EMBED_MAX_ITERS = 100
c374_ei_eht_VALENCE_ELECTRONS = {1: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 14: 4, 15: 5, 16: 6, 17: 7, 32: 4, 34: 6, 35: 7, 50: 4, 53: 7}

def c374_ei_eht_remove_dummy_caps(smiles: str):
    """Remove polymer attachment-point dummies and let RDKit add hydrogens."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        for atom_index in sorted([atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0], reverse=True):
            editable.RemoveAtom(atom_index)
        capped = editable.GetMol()
        Chem.SanitizeMol(capped)
        return Chem.AddHs(capped)
    except Exception:
        return None

def c374_ei_eht_ring_close_dummy_caps(smiles: str):
    """Close the two dummy-attachment neighbors into a cyclic repeat surrogate."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        dummy = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy) != 2:
            return None
        neighbors: list[int] = []
        for atom_index in dummy:
            atom = editable.GetAtomWithIdx(atom_index)
            heavy_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 0]
            if not heavy_neighbors:
                return None
            neighbors.append(int(heavy_neighbors[0]))
        if neighbors[0] == neighbors[1]:
            return None
        if editable.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is None:
            editable.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
        for atom_index in sorted(dummy, reverse=True):
            editable.RemoveAtom(atom_index)
        closed = editable.GetMol()
        Chem.SanitizeMol(closed)
        return Chem.AddHs(closed)
    except Exception:
        return None

def c374_ei_eht_embed_for_eht(molecule, seed: int):
    if molecule is None or molecule.GetNumAtoms() < 2:
        return None
    try:
        working = Chem.Mol(molecule)
        conformer_id = AllChem.EmbedMolecule(working, randomSeed=int(seed))
        if conformer_id < 0:
            conformer_id = AllChem.EmbedMolecule(working, randomSeed=int(seed), useRandomCoords=True)
        if conformer_id < 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(working, maxIters=c374_ei_eht_EMBED_MAX_ITERS)
        except Exception:
            pass
        return working
    except Exception:
        return None

def c374_ei_eht_valence_electron_count(molecule) -> int:
    total = 0
    for atom in molecule.GetAtoms():
        atomic_number = int(atom.GetAtomicNum())
        total += c374_ei_eht_VALENCE_ELECTRONS.get(atomic_number, min(max(atomic_number, 1), 8))
    return int(total)

def c374_ei_eht_eht_variant_features(molecule, seed: int):
    embedded = c374_ei_eht_embed_for_eht(molecule, seed)
    if embedded is None:
        return ([np.nan] * 18, False)
    try:
        ok, result = rdEHTTools.RunMol(embedded)
    except Exception:
        return ([np.nan] * 18, False)
    if not ok:
        return ([np.nan] * 18, False)
    energies = np.asarray(result.GetOrbitalEnergies(), dtype=np.float64)
    charges = np.asarray(result.GetAtomicCharges(), dtype=np.float64)
    if energies.size < 2 or charges.size == 0 or (not np.isfinite(energies).all()) or (not np.isfinite(charges).all()):
        return ([np.nan] * 18, False)
    electrons = c374_ei_eht_valence_electron_count(embedded)
    homo_index = max(0, min(int(energies.size) - 2, int(electrons // 2) - 1))
    lumo_index = homo_index + 1
    homo = float(energies[homo_index])
    lumo = float(energies[lumo_index])
    below = float(energies[homo_index - 1]) if homo_index > 0 else homo
    above = float(energies[lumo_index + 1]) if lumo_index + 1 < energies.size else lumo
    features = [homo, lumo, float(lumo - homo), below, above, float(homo - below), float(above - lumo), float(np.min(energies)), float(np.max(energies)), float(np.mean(energies)), float(np.std(energies)), float(np.quantile(energies, 0.25)), float(np.quantile(energies, 0.75)), float(np.min(charges)), float(np.max(charges)), float(np.mean(charges)), float(np.std(charges)), float(np.sum(np.abs(charges)))]
    return (features, True)

def c374_ei_eht_stable_seed(smiles: str, variant: str) -> int:
    raw = stable_seed_hex('C374|' + variant + '|' + smiles)
    return int(raw, 16) % 2000000000 + 1

def c374_ei_eht_stable_eht_features(smiles: str):
    hcap, hcap_ok = c374_ei_eht_eht_variant_features(c374_ei_eht_remove_dummy_caps(smiles), c374_ei_eht_stable_seed(smiles, 'hcap'))
    ring, ring_ok = c374_ei_eht_eht_variant_features(c374_ei_eht_ring_close_dummy_caps(smiles), c374_ei_eht_stable_seed(smiles, 'ring'))
    diffs = [hcap[i] - ring[i] if hcap_ok and ring_ok else np.nan for i in (0, 1, 2)]
    row = np.asarray(hcap + [float(hcap_ok)] + ring + [float(ring_ok)] + diffs, dtype=float)
    return (row, {'hcap_supported': bool(hcap_ok), 'ring_supported': bool(ring_ok)})

def c374_ei_eht_no_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def c374_ei_eht_grouped_folds(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (_, va) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError('Fold assignment failed')
    return folds

def c374_ei_eht_bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(c374_ei_eht_SEED)
    values: list[float] = []
    for _ in range(1000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in selected])
        if np.var(y[rows]) > 1e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else float('-inf')

def c374_ei_eht__make_model(ridge_alpha: float):
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=float(ridge_alpha), solver='lsqr', max_iter=5000, tol=0.0001))

def c374_ei_eht_c374_ei_eht(train_df, test_df, base_df, c282_oof_df, residual_weight=0.35, ridge_alpha=60.0):
    """EI-only EHT residual over the C282 OOF parent, deployed on base_df.

    Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
    """
    train = train_df.copy()
    test = test_df.copy()
    base = base_df.copy()
    oof = c282_oof_df.copy()
    residual_weight = float(residual_weight)
    if not 0.0 <= residual_weight <= 1.0:
        raise RuntimeError('residual_weight must be in [0, 1]')
    ridge_alpha = float(ridge_alpha)
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [c374_ei_eht_canonicalize(value) for value in frame['smiles']]
    ids = test['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base ID order')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base contains non-finite predictions')
    required = {'canonical', 'target_type', 'target', 'prediction'}
    if not required.issubset(oof.columns):
        raise RuntimeError('Unexpected C282 OOF schema')
    ei_oof = oof[oof['target_type'].astype(str).str.lower() == 'ei'].reset_index(drop=True)
    if len(ei_oof) < 50:
        raise RuntimeError('Insufficient EI OOF rows')
    if not np.isfinite(ei_oof[['target', 'prediction']].to_numpy(float)).all():
        raise RuntimeError('Non-finite EI OOF values')
    test_ei = test[test['target_type'] == 'ei'].sort_values('id').reset_index(drop=True)
    feature_keys = sorted(set(ei_oof['canonical'].astype(str)) | set(test_ei['canonical'].astype(str)))
    feature_rows: list[np.ndarray] = []
    support: list[dict[str, bool]] = []
    for smiles in feature_keys:
        row, report = c374_ei_eht_stable_eht_features(smiles)
        feature_rows.append(row)
        support.append(report)
    feature_matrix = np.asarray(feature_rows, dtype=np.float64)
    key_to_index = {key: i for i, key in enumerate(feature_keys)}
    train_indices = np.asarray([key_to_index[value] for value in ei_oof['canonical'].astype(str)], dtype=np.int64)
    test_indices = np.asarray([key_to_index[value] for value in test_ei['canonical'].astype(str)], dtype=np.int64)
    y = ei_oof['target'].to_numpy(float)
    parent = ei_oof['prediction'].to_numpy(float)
    groups = np.asarray([c374_ei_eht_no_stereo(value) for value in ei_oof['canonical'].astype(str)], dtype=object)
    folds = c374_ei_eht_grouped_folds(groups)
    residual_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        va = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        model = c374_ei_eht__make_model(ridge_alpha)
        model.fit(feature_matrix[train_indices[tr]], y[tr] - parent[tr])
        residual_oof[va] = model.predict(feature_matrix[train_indices[va]])
        pred = parent[va] + residual_weight * residual_oof[va]
        fold_rows.append({'fold': int(fold), 'rows': int(len(va)), 'parent_r2': float(r2_score(y[va], parent[va])), 'candidate_r2': float(r2_score(y[va], pred)), 'delta_r2': float(r2_score(y[va], pred) - r2_score(y[va], parent[va]))})
    if not np.isfinite(residual_oof).all():
        raise RuntimeError('Non-finite EI residual OOF')
    candidate_oof = parent + residual_weight * residual_oof
    report = {'parent_r2': float(r2_score(y, parent)), 'candidate_r2': float(r2_score(y, candidate_oof)), 'delta_r2': float(r2_score(y, candidate_oof) - r2_score(y, parent)), 'positive_folds': int(sum((row['delta_r2'] > 0.0 for row in fold_rows))), 'group_bootstrap_lower': c374_ei_eht_bootstrap_lower(y, parent, candidate_oof, groups), 'folds': fold_rows, 'train_rows': int(len(y)), 'test_rows': int(len(test_ei))}
    report['clean_gate_pass'] = bool(report['delta_r2'] >= 0.001 and report['positive_folds'] >= 3 and (report['group_bootstrap_lower'] > -0.004))
    full_model = c374_ei_eht__make_model(ridge_alpha)
    full_model.fit(feature_matrix[train_indices], y - parent)
    residual_test = full_model.predict(feature_matrix[test_indices])
    result = base['target'].to_numpy(float).copy()
    ei_positions = np.flatnonzero(test['target_type'].to_numpy(str) == 'ei')
    if len(ei_positions) != len(test_ei):
        raise RuntimeError('EI test alignment failed')
    result[ei_positions] = base['target'].to_numpy(float)[ei_positions] + residual_weight * residual_test
    if not np.isfinite(result).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})
c374_ei_eht___all__ = ['c374_ei_eht', 'stable_eht_features', 'eht_variant_features']

# ===== c380_ei_eht.py =====
"""In-memory port of C380 no-archive EI EHT + co-test residual source.

Faithful, self-contained port of
the original builder script (the original round
no-archive branch).  It combines C374-style RDKit/YAeHMOP extended-Hueckel
orbital features with the C327 co-test/identity features for EI only, trains a
grouped residual model against the C282 current-only OOF parent, and deploys
the EI residual over a frozen base candidate.

Strip-for-in-memory: no file I/O, no R2 paths, no digest/file-hash/manifest
code, no path guards, no importlib module loading.  The model fitting
(SEED/alphas/folds/clip/weight, bootstrap, GroupKFold) and the EHT orbital
feature computation are reproduced verbatim.

IMPORTANT: stable_seed() uses digest module.digest purely as a deterministic
string->int SEED derivation for AllChem.EmbedMolecule (it is the model-fitting
seed, NOT file-integrity hashing).  Changing it would change the 3D conformer
seeds and therefore the EHT features/predictions, so it is preserved verbatim.

Input DataFrames (all read-only; caller-provided):
  train_df     : official train.csv  (columns: smiles, target, target_type)
  test_df      : official test.csv   (columns: id, smiles, target_type)
  base_df      : base candidate      (columns: id, target) - frozen incumbent
  c282_oof_df  : C282 OOF            (columns: canonical, target_type, target,
                                       prediction, sparse_ridge, dense_ridge,
                                       extra_trees, tanimoto_local)
"""
RDLogger.DisableLog('rdApp.*')
c380_ei_eht_SEED = 20260808

def c380_ei_eht_canonicalize(smiles) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
c380_ei_eht_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

def c380_ei_eht_pivot_predictions(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return frame.pivot_table(index='canonical', columns='target_type', values=value_column, aggfunc='mean')

def c380_ei_eht_make_features(canonicals: np.ndarray, target: str, pivot: pd.DataFrame, nearest_same: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    base_names = []
    for prop in c380_ei_eht_TARGETS:
        base_names += [f'pred_{prop}', f'has_{prop}']
    extra_names = ['partner_count', 'nearest_same_target_tanimoto', 'ei_from_gap', 'eea_from_gap', 'gap_residual', 'eps_minus_nc2', 'eps_from_nc2', 'nc_from_eps', 'abs_gap_residual', 'abs_eps_minus_nc2']
    names = base_names + extra_names
    for canonical in canonicals:
        values: dict[str, float] = {}
        row_values: list[float] = []
        if canonical in pivot.index:
            source = pivot.loc[canonical]
        else:
            source = pd.Series(dtype=float)
        partner_count = 0
        for prop in c380_ei_eht_TARGETS:
            value = source.get(prop, np.nan)
            if pd.notna(value):
                values[prop] = float(value)
                row_values.extend([float(value), 1.0])
                if prop != target:
                    partner_count += 1
            else:
                row_values.extend([np.nan, 0.0])
        ei_from_gap = np.nan
        eea_from_gap = np.nan
        gap_residual = np.nan
        if 'eea' in values and 'egc' in values:
            ei_from_gap = values['eea'] + values['egc']
        if 'ei' in values and 'egc' in values:
            eea_from_gap = values['ei'] - values['egc']
        if all((prop in values for prop in ('ei', 'eea', 'egc'))):
            gap_residual = values['ei'] - values['eea'] - values['egc']
        eps_minus_nc2 = np.nan
        eps_from_nc2 = np.nan
        nc_from_eps = np.nan
        if 'eps' in values and 'nc' in values:
            eps_minus_nc2 = values['eps'] - values['nc'] ** 2
            eps_from_nc2 = values['nc'] ** 2 + max(eps_minus_nc2, 0.02)
            nc_from_eps = math.sqrt(max(values['eps'] - max(eps_minus_nc2, 0.02), 1.0))
        row_values.extend([float(partner_count), float(nearest_same.get(str(canonical), 0.0)), ei_from_gap, eea_from_gap, gap_residual, eps_minus_nc2, eps_from_nc2, nc_from_eps, abs(gap_residual) if pd.notna(gap_residual) else np.nan, abs(eps_minus_nc2) if pd.notna(eps_minus_nc2) else np.nan])
        rows.append(row_values)
    return (np.asarray(rows, dtype=np.float64), names)

def c380_ei_eht_morgan_fps(canonicals: list[str]) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps: dict[str, Any] = {}
    for canonical in sorted(set(canonicals)):
        mol = Chem.MolFromSmiles(str(canonical))
        if mol is None:
            continue
        fps[str(canonical)] = generator.GetFingerprint(mol)
    return fps

def c380_ei_eht_nearest_similarity(query: list[str], train_pool: list[str]) -> dict[str, float]:
    fps = c380_ei_eht_morgan_fps(query + train_pool)
    train_fps = [fps[canonical] for canonical in train_pool if canonical in fps]
    if not train_fps:
        return {str(canonical): 0.0 for canonical in query}
    result: dict[str, float] = {}
    for canonical in query:
        fp = fps.get(str(canonical))
        if fp is None:
            result[str(canonical)] = 0.0
        else:
            result[str(canonical)] = float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return result
c380_ei_eht_EMBED_MAX_ITERS = 100
c380_ei_eht_VALENCE_ELECTRONS = {1: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 14: 4, 15: 5, 16: 6, 17: 7, 32: 4, 34: 6, 35: 7, 50: 4, 53: 7}

def c380_ei_eht_remove_dummy_caps(smiles: str):
    """Remove polymer attachment-point dummies and let RDKit add hydrogens."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        for atom_index in sorted([atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0], reverse=True):
            editable.RemoveAtom(atom_index)
        capped = editable.GetMol()
        Chem.SanitizeMol(capped)
        return Chem.AddHs(capped)
    except Exception:
        return None

def c380_ei_eht_ring_close_dummy_caps(smiles: str):
    """Close the two dummy-attachment neighbors into a cyclic repeat surrogate."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        dummy = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy) != 2:
            return None
        neighbors: list[int] = []
        for atom_index in dummy:
            atom = editable.GetAtomWithIdx(atom_index)
            heavy_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 0]
            if not heavy_neighbors:
                return None
            neighbors.append(int(heavy_neighbors[0]))
        if neighbors[0] == neighbors[1]:
            return None
        if editable.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is None:
            editable.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
        for atom_index in sorted(dummy, reverse=True):
            editable.RemoveAtom(atom_index)
        closed = editable.GetMol()
        Chem.SanitizeMol(closed)
        return Chem.AddHs(closed)
    except Exception:
        return None

def c380_ei_eht_embed_for_eht(molecule, seed: int):
    if molecule is None or molecule.GetNumAtoms() < 2:
        return None
    try:
        working = Chem.Mol(molecule)
        conformer_id = AllChem.EmbedMolecule(working, randomSeed=int(seed))
        if conformer_id < 0:
            conformer_id = AllChem.EmbedMolecule(working, randomSeed=int(seed), useRandomCoords=True)
        if conformer_id < 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(working, maxIters=c380_ei_eht_EMBED_MAX_ITERS)
        except Exception:
            pass
        return working
    except Exception:
        return None

def c380_ei_eht_valence_electron_count(molecule) -> int:
    total = 0
    for atom in molecule.GetAtoms():
        atomic_number = int(atom.GetAtomicNum())
        total += c380_ei_eht_VALENCE_ELECTRONS.get(atomic_number, min(max(atomic_number, 1), 8))
    return int(total)

def c380_ei_eht_eht_variant_features(molecule, seed: int):
    embedded = c380_ei_eht_embed_for_eht(molecule, seed)
    if embedded is None:
        return ([np.nan] * 18, False)
    try:
        ok, result = rdEHTTools.RunMol(embedded)
    except Exception:
        return ([np.nan] * 18, False)
    if not ok:
        return ([np.nan] * 18, False)
    energies = np.asarray(result.GetOrbitalEnergies(), dtype=np.float64)
    charges = np.asarray(result.GetAtomicCharges(), dtype=np.float64)
    if energies.size < 2 or charges.size == 0 or (not np.isfinite(energies).all()) or (not np.isfinite(charges).all()):
        return ([np.nan] * 18, False)
    electrons = c380_ei_eht_valence_electron_count(embedded)
    homo_index = max(0, min(int(energies.size) - 2, int(electrons // 2) - 1))
    lumo_index = homo_index + 1
    homo = float(energies[homo_index])
    lumo = float(energies[lumo_index])
    below = float(energies[homo_index - 1]) if homo_index > 0 else homo
    above = float(energies[lumo_index + 1]) if lumo_index + 1 < energies.size else lumo
    features = [homo, lumo, float(lumo - homo), below, above, float(homo - below), float(above - lumo), float(np.min(energies)), float(np.max(energies)), float(np.mean(energies)), float(np.std(energies)), float(np.quantile(energies, 0.25)), float(np.quantile(energies, 0.75)), float(np.min(charges)), float(np.max(charges)), float(np.mean(charges)), float(np.std(charges)), float(np.sum(np.abs(charges)))]
    return (features, True)

def c380_ei_eht_stable_seed(smiles: str, variant: str) -> int:
    raw = stable_seed_hex('C380|' + variant + '|' + smiles)
    return int(raw, 16) % 2000000000 + 1

def c380_ei_eht_stable_eht_features(smiles: str):
    hcap, hcap_ok = c380_ei_eht_eht_variant_features(c380_ei_eht_remove_dummy_caps(smiles), c380_ei_eht_stable_seed(smiles, 'hcap'))
    ring, ring_ok = c380_ei_eht_eht_variant_features(c380_ei_eht_ring_close_dummy_caps(smiles), c380_ei_eht_stable_seed(smiles, 'ring'))
    diffs = [hcap[i] - ring[i] if hcap_ok and ring_ok else np.nan for i in (0, 1, 2)]
    row = np.asarray(hcap + [float(hcap_ok)] + ring + [float(ring_ok)] + diffs, dtype=float)
    return (row, {'hcap_supported': bool(hcap_ok), 'ring_supported': bool(ring_ok)})

def c380_ei_eht_no_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def c380_ei_eht_grouped_folds(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (_, va) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError('Fold assignment failed')
    return folds

def c380_ei_eht_bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(c380_ei_eht_SEED)
    values: list[float] = []
    for _ in range(1000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in selected])
        if np.var(y[rows]) > 1e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else float('-inf')

def c380_ei_eht_make_model(name: str, alpha: float) -> Any:
    if name == 'ridge_alpha':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=float(alpha), solver='lsqr', max_iter=5000, tol=0.0001))
    if name == 'ridge_200':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=200.0, solver='lsqr', max_iter=5000, tol=0.0001))
    if name == 'huber':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000))
    raise KeyError(name)

def c380_ei_eht_ei_eht_cotest(train_df, test_df, base_df, c282_oof_df, residual_weight=0.05, ridge_alpha=60.0, residual_clip=0.3) -> pd.DataFrame:
    """EI-only EHT + co-test residual over the C282 OOF parent, deployed on base_df.

    Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
    Unlike C327, the clean-gate verdict does not suppress deployment: the
    selected-arm residual is always applied to the EI rows.
    """
    train = train_df.copy()
    test = test_df.copy()
    base = base_df.copy()
    oof = c282_oof_df.copy()
    residual_weight = float(residual_weight)
    if not 0.0 <= residual_weight <= 1.0:
        raise RuntimeError('residual_weight must be in [0, 1]')
    ridge_alpha = float(ridge_alpha)
    residual_clip = float(residual_clip)
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [c380_ei_eht_canonicalize(value) for value in frame['smiles']]
    ids = test['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid candidate schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid candidate ID order')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Candidate contains non-finite values')
    test['base_prediction'] = base['target'].to_numpy(float)
    required = {'canonical', 'target_type', 'target', 'prediction'}
    if not required.issubset(oof.columns):
        raise RuntimeError('Unexpected C282 OOF schema')
    oof['target_type'] = oof['target_type'].astype(str).str.lower()
    ei_oof = oof[oof['target_type'] == 'ei'].reset_index(drop=True)
    if len(ei_oof) < 50:
        raise RuntimeError('Insufficient EI OOF rows')
    if not np.isfinite(ei_oof[['target', 'prediction']].to_numpy(float)).all():
        raise RuntimeError('Non-finite EI OOF values')
    test_ei = test[test['target_type'] == 'ei'].sort_values('id').reset_index(drop=True)
    feature_keys = sorted(set(ei_oof['canonical'].astype(str)) | set(test_ei['canonical'].astype(str)))
    eht_rows = []
    for smiles in feature_keys:
        row, _ = c380_ei_eht_stable_eht_features(smiles)
        eht_rows.append(row)
    eht_matrix = np.asarray(eht_rows, dtype=np.float64)
    key_to_index = {key: i for i, key in enumerate(feature_keys)}
    train_indices = np.asarray([key_to_index[value] for value in ei_oof['canonical'].astype(str)], dtype=np.int64)
    test_indices = np.asarray([key_to_index[value] for value in test_ei['canonical'].astype(str)], dtype=np.int64)
    train_pool = ei_oof['canonical'].astype(str).tolist()
    oof_pivot = c380_ei_eht_pivot_predictions(oof, 'prediction')
    nearest_oof = c380_ei_eht_nearest_similarity(ei_oof['canonical'].astype(str).tolist(), train_pool)
    cotest_x, cotest_names = c380_ei_eht_make_features(ei_oof['canonical'].astype(str).to_numpy(), 'ei', oof_pivot, nearest_oof)
    test_pivot = c380_ei_eht_pivot_predictions(test.rename(columns={'base_prediction': 'prediction'}), 'prediction')
    nearest_test = c380_ei_eht_nearest_similarity(test_ei['canonical'].astype(str).tolist(), train_pool)
    cotest_test_x, _ = c380_ei_eht_make_features(test_ei['canonical'].astype(str).to_numpy(), 'ei', test_pivot, nearest_test)
    x = np.hstack([eht_matrix[train_indices], cotest_x])
    test_x = np.hstack([eht_matrix[test_indices], cotest_test_x])
    y = ei_oof['target'].to_numpy(float)
    parent = ei_oof['prediction'].to_numpy(float)
    residual = y - parent
    groups = np.asarray([c380_ei_eht_no_stereo(value) for value in ei_oof['canonical'].astype(str)], dtype=object)
    folds = c380_ei_eht_grouped_folds(groups)
    parent_r2 = float(r2_score(y, parent))
    arms: dict[str, dict[str, Any]] = {}
    for arm in ('ridge_alpha', 'ridge_200', 'huber'):
        residual_oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            va = np.flatnonzero(folds == fold)
            tr = np.flatnonzero(folds != fold)
            model = c380_ei_eht_make_model(arm, ridge_alpha)
            model.fit(x[tr], residual[tr])
            raw = np.asarray(model.predict(x[va]), dtype=np.float64)
            raw = np.clip(raw, -residual_clip, residual_clip)
            residual_oof[va] = raw
            pred = parent[va] + residual_weight * raw
            fold_rows.append({'fold': int(fold), 'rows': int(len(va)), 'parent_r2': float(r2_score(y[va], parent[va])), 'candidate_r2': float(r2_score(y[va], pred)), 'delta_r2': float(r2_score(y[va], pred) - r2_score(y[va], parent[va]))})
        if not np.isfinite(residual_oof).all():
            raise RuntimeError(f'Non-finite residual OOF for {arm}')
        cand = parent + residual_weight * residual_oof
        arms[arm] = {'parent_r2': parent_r2, 'candidate_r2': float(r2_score(y, cand)), 'delta_r2': float(r2_score(y, cand) - parent_r2), 'positive_folds': int(sum((row['delta_r2'] > 0.0 for row in fold_rows))), 'group_bootstrap_lower': c380_ei_eht_bootstrap_lower(y, parent, cand, groups), 'folds': fold_rows}
    selected_arm = max(arms, key=lambda name: arms[name]['candidate_r2'])
    selected = arms[selected_arm]
    clean_gate_pass = bool(selected['delta_r2'] >= 0.005 and selected['positive_folds'] >= 4 and (selected['group_bootstrap_lower'] > 0.0))
    full_model = c380_ei_eht_make_model(selected_arm, ridge_alpha)
    full_model.fit(x, residual)
    residual_test = np.clip(np.asarray(full_model.predict(test_x), dtype=np.float64), -residual_clip, residual_clip)
    result = base['target'].to_numpy(float).copy()
    ei_positions = np.flatnonzero(test['target_type'].to_numpy(str) == 'ei')
    if len(ei_positions) != len(test_ei):
        raise RuntimeError('EI test alignment failed')
    result[ei_positions] = base['target'].to_numpy(float)[ei_positions] + residual_weight * residual_test
    if not np.isfinite(result).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})

def c402_canonicalize(smiles):
    return canonicalize(smiles)


# ===== c402_eps_surrogate.py =====
"""C402 no-archive EPS ionic route with surrogate-NC deployment (in-memory port).

Faithful in-memory port of the original builder script.
Reads ONLY official train/test from data_dir (via passed DataFrames).
Returns a 4,940-row (id, target) DataFrame.
"""
c402_eps_surrogate_SEED = 20260808
c402_eps_surrogate_MIN_IONIC = 0.02
c402_eps_surrogate_IONIC_MODELS = ('ridge', 'et', 'hgb')

def c402_eps_surrogate_c187_make_model(kind, fold):
    if kind == 'ridge':
        return Ridge(alpha=50.0)
    if kind == 'et':
        return ExtraTreesRegressor(n_estimators=300, max_features=0.55, min_samples_leaf=2, random_state=20260804 + fold, n_jobs=4)
    return HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0, random_state=20260804 + fold)

def c402_eps_surrogate_c187_fold_matrix(dense, sparse_matrix, train_rows, pred_rows):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    train_dense = imputer.fit_transform(dense[train_rows])
    pred_dense = imputer.transform(dense[pred_rows])
    qt = QuantileTransformer(n_quantiles=min(100, max(10, len(train_rows))), output_distribution='normal', random_state=20260804)
    train_dense = qt.fit_transform(train_dense)
    pred_dense = qt.transform(pred_dense)
    if sparse_matrix.shape[1] > 1 and len(train_rows) > 3:
        n_components = min(64, sparse_matrix.shape[1] - 1, len(train_rows) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=20260804)
        train_sparse = svd.fit_transform(sparse_matrix[train_rows])
        pred_sparse = svd.transform(sparse_matrix[pred_rows])
        return (np.hstack([train_dense, train_sparse]), np.hstack([pred_dense, pred_sparse]))
    return (train_dense, pred_dense)

def c402_eps_surrogate_c402_no_stereo(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def c402_eps_surrogate_c402_make_nc_model(kind, fold):
    if kind == 'ridge':
        return Ridge(alpha=80.0)
    if kind == 'hgb':
        return HistGradientBoostingRegressor(max_iter=260, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0, random_state=c402_eps_surrogate_SEED + fold)
    if kind == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=500, max_features=0.6, min_samples_leaf=2, random_state=c402_eps_surrogate_SEED + fold, n_jobs=4)
    raise RuntimeError('Unknown surrogate NC model: ' + kind)

def c402_eps_surrogate_c402_grouped_folds(groups, n_splits=5):
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for fold, (_, va) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError('Fold assignment failed')
    return folds

def c402_eps_surrogate_c402_morgan_fp(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError('Invalid SMILES for Morgan support: ' + str(smiles))
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

def c402_eps_surrogate_c402_max_tanimoto_to_support(test_smiles, support_smiles):
    support_fps = [c402_eps_surrogate_c402_morgan_fp(value) for value in support_smiles]
    values = []
    for value in test_smiles:
        fp = c402_eps_surrogate_c402_morgan_fp(value)
        sims = DataStructs.BulkTanimotoSimilarity(fp, support_fps)
        values.append(float(max(sims)) if sims else 0.0)
    return np.asarray(values, dtype=np.float64)

def c402_eps_surrogate_c402_fit_predict_ensemble(dense, sparse_matrix, train_indices, y, pred_indices, model_kind):
    x_tr, x_te = c402_eps_surrogate_c187_fold_matrix(dense, sparse_matrix, train_indices, pred_indices)
    model = c402_eps_surrogate_c402_make_nc_model(model_kind, 0)
    model.fit(x_tr, y)
    pred = model.predict(x_te)
    return np.asarray(pred, dtype=np.float64)

def c402_eps_surrogate_c402_nc_surrogate_oof(dense, sparse_matrix, nc_indices, nc_y, groups, model_kind):
    folds = c402_eps_surrogate_c402_grouped_folds(groups)
    pred = np.full(len(nc_indices), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        x_tr, x_va = c402_eps_surrogate_c187_fold_matrix(dense, sparse_matrix, nc_indices[tr], nc_indices[va])
        model = c402_eps_surrogate_c402_make_nc_model(model_kind, int(fold))
        model.fit(x_tr, nc_y[tr])
        pred[va] = model.predict(x_va)
    if not np.isfinite(pred).all():
        raise RuntimeError('Non-finite NC surrogate OOF')
    return pred

def c402_eps_surrogate_c402_ionic_oof_audit(dense, sparse_matrix, pair_indices, eps_y, nc_y, groups):
    folds = c402_eps_surrogate_c402_grouped_folds(groups)
    log_ionic = np.log(np.maximum(eps_y - nc_y ** 2, c402_eps_surrogate_MIN_IONIC))
    pred_by_kind = {kind: np.full(len(pair_indices), np.nan, dtype=np.float64) for kind in c402_eps_surrogate_IONIC_MODELS}
    for fold in sorted(np.unique(folds)):
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        x_tr, x_va = c402_eps_surrogate_c187_fold_matrix(dense, sparse_matrix, pair_indices[tr], pair_indices[va])
        for kind in c402_eps_surrogate_IONIC_MODELS:
            model = c402_eps_surrogate_c187_make_model(kind, int(fold))
            model.fit(x_tr, log_ionic[tr])
            pred_by_kind[kind][va] = np.exp(np.clip(model.predict(x_va), -8, 4))
    if any((not np.isfinite(value).all() for value in pred_by_kind.values())):
        raise RuntimeError('Non-finite ionic OOF')
    ionic_pred = np.mean(np.column_stack([pred_by_kind[kind] for kind in c402_eps_surrogate_IONIC_MODELS]), axis=1)
    return ionic_pred

def c402_eps_surrogate_c402_build_features(keys):
    """C180 Flory-Fox carrier feature block: round1.build_features + DIRECT_BLOCKS."""
    built = build_features(keys, n_bits=2048, text_features=32768, motif_hash_features=0, rich_features=True, periodic_features=True, periodic_dense_features=True, capped_dense_features=True, motif_features=True, physics_features=True, mordred_features=False, oligomer_features=True, oligomer_repeats=2, oligomer_slope_features=False, oligomer_ffox_features=True, oligomer_ffox_max_repeats=3, oligomer_ffox_transform='both', oligomer_3d_features=False, rdkit_3d_features=False, backbone_sidechain_features=True, conjugation_features=True, mobility_features=True, huckel_features=False, electronic_tail_features=True, topological_autocorr_features=False, infinite_chain_features=True, bicerano_features=False, map4_features=True, map4_hash_features=16384, map4_max_distance=10, map4_env_radius=1, region_sparse_features=False, endpoint_path_sparse_features=True, endpoint_path_hash_features=16384, endpoint_path_max_bonds=8, rooted_smiles_features=True, rooted_smiles_max_roots=8, rooted_smiles_text_features=16384, random_smiles_features=False, kekule_smiles_features=True, kekule_smiles_text_features=16384, exact_sparse_features=False, wl_sparse_features=False)
    dense = np.asarray(built['dense'], dtype=np.float64)
    blocks = [built['blocks'][name] for name in c402_eps_surrogate_DIRECT_BLOCKS if name in built['blocks']]
    if not blocks:
        raise RuntimeError('C180 produced no sparse feature blocks')
    sparse_features = sparse.hstack(blocks, format='csr').astype(np.float64)
    return (dense, sparse_features)
c402_eps_surrogate_DIRECT_BLOCKS = ('maccs_bit', 'morgan_count_r1', 'morgan_count_r2', 'morgan_count_r3', 'morgan_count_r4', 'morgan_count_r5', 'morgan_bit_r2', 'atom_pair_count', 'topological_torsion_count', 'char_text', 'periodic_morgan_count_r2', 'periodic_morgan_count_r3', 'capped_morgan_count_r2')

def c402_eps_surrogate_eps_surrogate_nc_ionic(train_df, test_df, base_df, surrogate_nc_model='extra_trees', support_min_similarity=0.35, pull=0.5):
    """C402: EPS ionic route with surrogate-NC deployment over a frozen base."""
    if list(train_df.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected train schema')
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected test schema')
    if list(base_df.columns) != ['id', 'target'] or len(base_df) != 4940:
        raise RuntimeError('Unexpected base schema')
    train = train_df.copy()
    test = test_df.copy().reset_index(drop=True)
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    train['canonical'] = [c402_canonicalize(value) for value in train['smiles']]
    test['canonical'] = [c402_canonicalize(value) for value in test['smiles']]
    train['nostereo'] = [c402_eps_surrogate_c402_no_stereo(value) for value in train['canonical']]
    test['nostereo'] = [c402_eps_surrogate_c402_no_stereo(value) for value in test['canonical']]
    ids = test['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    result = base_df['target'].to_numpy(float).copy()
    if not np.isfinite(result).all():
        raise RuntimeError('Base has non-finite predictions')
    wide = train.pivot_table(index='canonical', columns='target_type', values='target', aggfunc='mean')
    pair_frame = wide[['eps', 'nc']].dropna().copy()
    nc_frame = wide[['nc']].dropna().copy()
    if len(pair_frame) < 50 or len(nc_frame) < 50:
        raise RuntimeError('Insufficient current official EPS/NC or NC train rows')
    ionic = pair_frame['eps'].to_numpy(float) - pair_frame['nc'].to_numpy(float) ** 2
    if np.any(ionic <= 0):
        raise RuntimeError('Non-positive ionic coordinate in current official pair rows')
    keys = sorted(set(train['canonical']) | set(test['canonical']))
    key_to_index = {key: index for index, key in enumerate(keys)}
    dense, sparse_matrix = c402_eps_surrogate_c402_build_features(keys)
    pair_canons = pair_frame.index.astype(str).tolist()
    nc_canons = nc_frame.index.astype(str).tolist()
    pair_indices = np.asarray([key_to_index[value] for value in pair_canons], dtype=np.int64)
    nc_indices = np.asarray([key_to_index[value] for value in nc_canons], dtype=np.int64)
    pair_groups = np.asarray([c402_eps_surrogate_c402_no_stereo(value) for value in pair_canons], dtype=object)
    nc_groups = np.asarray([c402_eps_surrogate_c402_no_stereo(value) for value in nc_canons], dtype=object)
    eps_y = pair_frame['eps'].to_numpy(float)
    pair_nc_y = pair_frame['nc'].to_numpy(float)
    nc_y = nc_frame['nc'].to_numpy(float)
    c402_eps_surrogate_c402_ionic_oof_audit(dense, sparse_matrix, pair_indices, eps_y, pair_nc_y, pair_groups)
    c402_eps_surrogate_c402_nc_surrogate_oof(dense, sparse_matrix, nc_indices, nc_y, nc_groups, surrogate_nc_model)
    test_eps = test[test['target_type'] == 'eps'].copy()
    test_eps_indices = test_eps.index.to_numpy(int)
    test_eps_canons = test_eps['canonical'].astype(str).tolist()
    test_eps_key_indices = np.asarray([key_to_index[value] for value in test_eps_canons], dtype=np.int64)
    support_similarity = c402_eps_surrogate_c402_max_tanimoto_to_support(test_eps_canons, sorted(set(pair_canons) | set(nc_canons)))
    supported = support_similarity >= float(support_min_similarity)
    nc_pred = c402_eps_surrogate_c402_fit_predict_ensemble(dense, sparse_matrix, nc_indices, nc_y, test_eps_key_indices, surrogate_nc_model)
    nc_low = float(np.quantile(nc_y, 0.005))
    nc_high = float(np.quantile(nc_y, 0.995))
    nc_pred = np.clip(nc_pred, max(1.0, nc_low - 0.05), min(2.8, nc_high + 0.05))
    x_tr, x_te = c402_eps_surrogate_c187_fold_matrix(dense, sparse_matrix, pair_indices, test_eps_key_indices)
    log_ionic = np.log(np.maximum(ionic, c402_eps_surrogate_MIN_IONIC))
    ionic_preds = []
    for kind in c402_eps_surrogate_IONIC_MODELS:
        model = c402_eps_surrogate_c187_make_model(kind, c402_eps_surrogate_SEED)
        model.fit(x_tr, log_ionic)
        ionic_preds.append(np.exp(np.clip(model.predict(x_te), -8, 4)))
    ionic_pred = np.maximum(np.mean(np.column_stack(ionic_preds), axis=1), c402_eps_surrogate_MIN_IONIC)
    raw_eps = nc_pred ** 2 + ionic_pred
    eps_train = train[train['target_type'] == 'eps']['target'].to_numpy(float)
    eps_low = float(np.quantile(eps_train, 0.002))
    eps_high = float(np.quantile(eps_train, 0.998))
    raw_eps = np.clip(raw_eps, max(0.0, eps_low - 0.05), eps_high + 0.05)
    base_eps = result[test_eps_indices]
    replacement = (1.0 - float(pull)) * base_eps + float(pull) * raw_eps
    result[test_eps_indices[supported]] = replacement[supported]
    if not np.isfinite(result).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})

# ===== c407_weak_zoo.py =====
"""C407 current-only weak-target direct model-zoo source (in-memory port).

Faithful port of the original builder script.
Reads ONLY official train/test via passed DataFrames. Replaces only the
requested targets over a frozen base CSV. Returns a 4,940-row (id, target) DataFrame.
"""
RDLogger.DisableLog('rdApp.*')
c407_weak_zoo_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
c407_weak_zoo_SEED = 20260808
try:
    import lightgbm as lgb
except Exception:
    lgb = None

def c407_weak_zoo_c385_parse_targets(value):
    targets = tuple((item.strip().lower() for item in str(value).split(',') if item.strip()))
    if not targets:
        raise RuntimeError('No targets requested')
    bad = [t for t in targets if t not in c407_weak_zoo_TARGETS]
    if bad:
        raise RuntimeError('Invalid targets: ' + str(bad))
    return targets

def c407_weak_zoo_c385_parse_models(value, available):
    requested = tuple((item.strip() for item in str(value).split(',') if item.strip()))
    if not requested:
        raise RuntimeError('No models requested')
    bad = [name for name in requested if name not in available]
    if bad:
        raise RuntimeError('Requested unavailable models ' + str(bad))
    return {name: available[name] for name in requested}

def c407_weak_zoo_c385_no_stereo(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def c407_weak_zoo_c385_maccs_matrix(molecules):
    rows = np.zeros((len(molecules), 167), dtype=np.float32)
    for i, mol in enumerate(molecules):
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows[i] = np.asarray([int(fp.GetBit(j)) for j in range(167)], dtype=np.float32)
    return rows

def c407_weak_zoo_c385_sanitize_dense(x):
    out = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(out) | (np.abs(out) > 1000000000000.0)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out

def c407_weak_zoo_c385_grouped_oof(factory, x, y, groups, seed):
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError('Not enough groups for OOF')
    oof = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x[tr], y[tr])
        pred = np.asarray(model.predict(x[va]), dtype=np.float64)
        oof[va] = pred
    if not np.isfinite(oof).all():
        raise RuntimeError('Non-finite OOF prediction')
    return oof

def c407_weak_zoo_c385_clipped_by_train(y, pred):
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    return np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)

def c407_weak_zoo_c385_model_factories(target):
    weak = target in {'ei', 'eea', 'egb', 'eps', 'nc'}
    factories = {'ridge_30': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0, solver='lsqr', max_iter=5000, tol=0.0001)), 'ridge_200': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=200.0, solver='lsqr', max_iter=5000, tol=0.0001)), 'extra_trees': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), ExtraTreesRegressor(n_estimators=450 if weak else 300, min_samples_leaf=2 if weak else 4, max_features=0.7 if weak else 0.5, random_state=seed, n_jobs=4)), 'hist_gbdt': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), HistGradientBoostingRegressor(learning_rate=0.035, max_iter=420 if weak else 300, l2_regularization=0.1, max_leaf_nodes=15 if weak else 31, min_samples_leaf=8 if weak else 20, random_state=seed))}
    if lgb is not None:
        factories['lightgbm'] = lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), lgb.LGBMRegressor(objective='regression', n_estimators=550 if weak else 400, learning_rate=0.03, num_leaves=31, min_child_samples=8 if weak else 20, subsample=0.85, subsample_freq=1, colsample_bytree=0.75, reg_lambda=0.2, random_state=seed, n_jobs=4, verbosity=-1))
    return factories

def c407_weak_zoo_weak_target_zoo(train_df, test_df, base_df, targets='ei,eps,nc,tg,egc', models='ridge_200,extra_trees,hist_gbdt,lightgbm', morgan_bits=512, seed=20260808):
    """C407 weak-target direct model zoo. Returns a 4,940-row (id, target) DataFrame."""
    if list(train_df.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected train schema')
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected test schema')
    if list(base_df.columns) != ['id', 'target'] or len(base_df) != 4940:
        raise RuntimeError('Unexpected base schema')
    train = train_df.copy()
    test = test_df.copy().reset_index(drop=True)
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    train['canonical'] = [canonicalize(value) for value in train['smiles']]
    test['canonical'] = [canonicalize(value) for value in test['smiles']]
    ids = test['id'].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected official test IDs')
    predictions = base_df['target'].to_numpy(float).copy()
    if not np.isfinite(predictions).all():
        raise RuntimeError('Base contains non-finite predictions')
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = build_label_pool(train, archive)
    pooled = pooled.copy()
    test2 = test.copy()
    for frame in (pooled, test2):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
    active_targets = c407_weak_zoo_c385_parse_targets(targets)
    keys = sorted(set(pooled['canonical']) | set(test2['canonical']))
    key_to_index = {key: i for i, key in enumerate(keys)}
    molecules = build_molecules(keys)
    descriptor, descriptor_names = descriptor_matrix(molecules, keys)
    physical, physical_names = physical_matrix(molecules, keys)
    morgan2 = morgan_count_matrix(molecules, radius=2, bits=int(morgan_bits)).toarray().astype(np.float32)
    morgan3 = morgan_count_matrix(molecules, radius=3, bits=int(morgan_bits)).toarray().astype(np.float32)
    maccs = c407_weak_zoo_c385_maccs_matrix(molecules)
    cross_values, cross_available = cross_property_arrays(pooled, keys)
    base_dense = c407_weak_zoo_c385_sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs]).astype(np.float32))
    for target in active_targets:
        target_train = pooled[pooled['target_type'].eq(target)].reset_index(drop=True)
        target_test = test2[test2['target_type'].eq(target)].reset_index(drop=False)
        train_idx = np.asarray([key_to_index[x] for x in target_train['canonical']], dtype=int)
        test_idx = np.asarray([key_to_index[x] for x in target_test['canonical']], dtype=int)
        y = target_train['target'].to_numpy(float)
        groups = np.asarray([c407_weak_zoo_c385_no_stereo(value) for value in target_train['canonical'].astype(str)], dtype=object)
        dense = c407_weak_zoo_c385_sanitize_dense(target_dense_features(base_dense, cross_values, cross_available, target))
        x_train = dense[train_idx]
        x_test = dense[test_idx]
        candidates = {}
        reports = {}
        selected_factories = c407_weak_zoo_c385_parse_models(models, c407_weak_zoo_c385_model_factories(target))
        for model_name, factory in selected_factories.items():
            try:
                oof_raw = c407_weak_zoo_c385_grouped_oof(factory, x_train, y, groups, int(seed) + 100 * c407_weak_zoo_TARGETS.index(target))
                oof = c407_weak_zoo_c385_clipped_by_train(y, oof_raw)
                reports[model_name] = float(r2_score(y, oof))
                candidates[model_name] = oof
            except Exception:
                continue
        if not candidates:
            raise RuntimeError('No successful models for ' + target)
        best_name = max(candidates, key=lambda name: reports[name])
        final_model = selected_factories[best_name](int(seed) + 999 + c407_weak_zoo_TARGETS.index(target))
        final_model.fit(x_train, y)
        final_pred = c407_weak_zoo_c385_clipped_by_train(y, np.asarray(final_model.predict(x_test), dtype=np.float64))
        lookup = target_train.groupby('canonical')['target'].mean().to_dict()
        for local_pos, row in enumerate(target_test.itertuples(index=False)):
            if row.canonical in lookup:
                final_pred[local_pos] = float(lookup[row.canonical])
        positions = target_test['index'].to_numpy(int)
        predictions[positions] = final_pred
    if not np.isfinite(predictions).all():
        raise RuntimeError('Missing/non-finite final predictions')
    return pd.DataFrame({'id': ids, 'target': predictions})

# ===== c340_wrapper.py =====
"""In-memory port of C340 no-archive C282 Polymer Genome hierarchical residual wrapper.

Faithful, self-contained port of
the original builder script (the original round
no-archive branch).  C340 is a thin wrapper over the C279 hierarchical residual
portfolio whose parent is swapped from the archive C050 artifacts to the
current-only C282 artifacts.  The actual computation (from
round2_c279_polymer_genome_hierarchical_portfolio.py) builds a from-scratch
Polymer Genome feature family (atom singles/bonds/coordinated triples,
backbone-restricted triples, ring/side-chain morphology) and nests a
target-specific Ridge residual inside grouped folds, blended with the C282
parent using a nonnegative weight selected only on inner grouped folds.

Strip-for-in-memory (deliberate, non-behavioral deviations):
  * No file I/O / pd.read_csv / to_csv / Path / open; inputs arrive as DataFrames.
  * No digest module/digest/digest, manifest, path-guard, protocol, env-var, or argparse code.
  * No the original round paths (the original round experiment paths/, the original round output paths/, the verification panel).
  * Dropped progress.jsonl / metrics.json / protocol.json / decision.md /
    artifact_manifest logging (non-behavioral).  The dead 'pset' variable and the
    stripped 'digest()' hash helper are removed.

Model/feature hyperparameters reproduced verbatim (do not change):
  * Ridge residual: alpha in (30, 100, 300), solver='lsqr', max_iter=5000, tol=1e-4,
    inside make_pipeline(SimpleImputer(median, keep_empty_features=True), StandardScaler()).
  * Parent blend weights: (0.0, 0.1, 0.2, 0.3, 0.4), nonnegative, selected on inner
    grouped folds only; tie-break prefers smaller weight then smaller alpha.
  * GroupKFold (n_splits = min(5, #unique canonical groups)); n_splits<2 -> (alpha=100, weight=0).
  * Feature vocabulary: Polymer Genome tokens with document frequency >= 5;
    features = hstack([log1p(counts), counts/row_sum, 12-dim morphology]).
  * RDKit canonicalization: Chem.MolToSmiles(..., canonical=True, isomericSmiles=True).

Input DataFrames (all read-only; caller-provided):
  test_df            : official test.csv  (columns: id, smiles, target_type)
  c282_oof_df        : C282 OOF           (columns: canonical, target_type, target, prediction)
  c282_submission_df : C282 candidate     (columns: id, target)

Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
"""
RDLogger.DisableLog('rdApp.*')
c340_wrapper_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
c340_wrapper_ALPHAS = (30.0, 100.0, 300.0)
c340_wrapper_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4)

def c340_wrapper_canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
c340_wrapper_MORPH_N = 12

def c340_wrapper_atom_type(a) -> str:
    if a.GetAtomicNum() == 0:
        return 'X'
    return f'{a.GetSymbol()}{a.GetDegree()}'

def c340_wrapper_keys_for(c):
    m = Chem.MolFromSmiles(c)
    if m is None:
        return (Counter(), Counter(), [])
    at = {a.GetIdx(): c340_wrapper_atom_type(a) for a in m.GetAtoms()}
    cnt = Counter()
    for a in m.GetAtoms():
        cnt['S|' + at[a.GetIdx()]] += 1
    for b in m.GetBonds():
        i, j = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        p = sorted([at[i], at[j]])
        cnt['P|' + '-'.join(p)] += 1
    for a in m.GetAtoms():
        j = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        for u in range(len(nb)):
            for v in range(u + 1, len(nb)):
                i, k = (nb[u], nb[v])
                ends = sorted([at[i], at[k]])
                cnt['T|' + ends[0] + '-' + at[j] + '-' + ends[1]] += 1
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    bcnt = Counter()
    path = []
    if len(stars) == 2:
        try:
            path = list(Chem.GetShortestPath(m, stars[0], stars[1]))
        except Exception:
            path = []
        for n_ in range(1, len(path) - 1):
            i, j, k = (path[n_ - 1], path[n_], path[n_ + 1])
            ends = sorted([at[i], at[k]])
            bcnt['B|' + ends[0] + '-' + at[j] + '-' + ends[1]] += 1
    return (cnt, bcnt, path)

def c340_wrapper_morphological(c):
    m = Chem.MolFromSmiles(c)
    if m is None:
        return [np.nan] * c340_wrapper_MORPH_N
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    n = m.GetNumAtoms()
    try:
        D = Chem.GetDistanceMatrix(m)
    except Exception:
        return [np.nan] * c340_wrapper_MORPH_N
    dd = []
    for a_ in range(len(rings)):
        for b_ in range(a_ + 1, len(rings)):
            dd.append(min((D[i, j] for i in rings[a_] for j in rings[b_])))
    ring_atoms = set().union(*rings) if rings else set()
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    side, path = (set(), [])
    if len(stars) == 2:
        try:
            path = list(Chem.GetShortestPath(m, stars[0], stars[1]))
        except Exception:
            path = []
        side = set(range(n)) - set(path)
    sizes = []
    if path:
        for i in set(path) - set(stars):
            for nb in m.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() in side:
                    st, comp = ([nb.GetIdx()], set())
                    while st:
                        u = st.pop()
                        if u in comp or u in path:
                            continue
                        comp.add(u)
                        st.extend((x.GetIdx() for x in m.GetAtomWithIdx(u).GetNeighbors()))
                    sizes.append(len(comp))
    sz = sizes or [0]
    return [len(rings), float(np.mean(dd)) if dd else -1.0, float(np.min(dd)) if dd else -1.0, float(np.max(dd)) if dd else -1.0, len(ring_atoms) / max(1, n), len(side) / max(1, n), float(np.max(sz)), float(np.mean(sz)), len(sizes), float(np.max(sz)) / max(1, n), len(path) / max(1, n), float(np.sum([1 for i in ring_atoms if i in set(path)])) / max(1, len(path) or 1)]

def c340_wrapper_grouped_folds(groups: np.ndarray) -> np.ndarray:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        return np.zeros(len(groups), dtype=np.int64)
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[valid] = fold
    return folds

def c340_wrapper_fit_predict(X_train: np.ndarray, residual: np.ndarray, X_valid: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=alpha, solver='lsqr', max_iter=5000, tol=0.0001))
    model.fit(X_train, residual)
    return np.asarray(model.predict(X_valid), dtype=float)

def c340_wrapper_select_inner(X: np.ndarray, y: np.ndarray, base: np.ndarray, groups: np.ndarray):
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        return (100.0, 0.0, {'inner_folds': 1, 'scores': {}})
    predictions = {(alpha, weight): np.full(len(y), np.nan, dtype=float) for alpha in c340_wrapper_ALPHAS for weight in c340_wrapper_WEIGHTS}
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, valid_idx in splitter.split(X, y, groups=groups):
        for alpha in c340_wrapper_ALPHAS:
            residual = c340_wrapper_fit_predict(X[train_idx], y[train_idx] - base[train_idx], X[valid_idx], alpha)
            for weight in c340_wrapper_WEIGHTS:
                predictions[alpha, weight][valid_idx] = base[valid_idx] + weight * residual
    scores = {key: float(r2_score(y, pred)) for key, pred in predictions.items()}
    best = max(scores, key=lambda key: (scores[key], -key[1], -key[0]))
    return (best[0], best[1], {'inner_folds': n_splits, 'scores': {f'{a}:{w}': value for (a, w), value in scores.items()}})

def c340_wrapper_build_features(keys):
    records = []
    morphology = []
    document_frequency = Counter()
    for key in keys:
        record = c340_wrapper_keys_for(key)
        records.append(record)
        document_frequency.update(set(record[0]) | set(record[1]))
    vocabulary = sorted((value for value, count in document_frequency.items() if count >= 5))
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
    for key in keys:
        morphology.append(c340_wrapper_morphological(key))
    morph = np.asarray(morphology, dtype=np.float32)
    counts_sum = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.maximum(1.0, counts_sum)
    features = np.hstack([np.log1p(counts), normalized, morph]).astype(np.float32)
    features[~np.isfinite(features)] = np.nan
    return (features, {'vocabulary_size': len(vocabulary), 'feature_shape': list(features.shape)})

def c340_wrapper_c340_c282_polymer_genome_wrapper(test_df, c282_oof_df, c282_submission_df) -> pd.DataFrame:
    """Run the C340 no-archive C282 Polymer Genome hierarchical residual wrapper.

    Returns a 4,940-row DataFrame with columns [id, target] (ids 1..4940).
    """
    oof = c282_oof_df.copy()
    parent_test = c282_submission_df.copy()
    test = test_df.copy()
    required_oof = ['canonical', 'target_type', 'target', 'prediction']
    if not all((column in oof.columns for column in required_oof)):
        raise RuntimeError('Unexpected C282 OOF schema')
    if not {'id', 'target'}.issubset(parent_test.columns):
        raise RuntimeError('Unexpected C282 test detail schema')
    if list(test.columns) != ['id', 'smiles', 'target_type'] or len(test) != 4940:
        raise RuntimeError('Unexpected test schema/count')
    parent_oof = oof[['canonical', 'target_type', 'target', 'prediction']].rename(columns={'prediction': 'candidate_prediction'})
    parent_oof['group'] = parent_oof['canonical'].astype(str)
    parent_test = parent_test[['id', 'target']].copy()
    test['canonical'] = test['smiles'].map(c340_wrapper_canonicalize)
    parent_test = parent_test.merge(test[['id', 'target_type', 'canonical']], on='id', how='left', validate='one_to_one')
    if len(parent_test) != 4940 or not np.isfinite(parent_test['target'].to_numpy(float)).all():
        raise RuntimeError('C282 test parent contract failed')
    keys = sorted(set(parent_oof['canonical'].astype(str)) | set(test['canonical'].astype(str)))
    X, _feature_report = c340_wrapper_build_features(keys)
    key_index = {key: index for index, key in enumerate(keys)}
    test_parts = []
    for target in c340_wrapper_TARGETS:
        frame = parent_oof[parent_oof['target_type'].astype(str).eq(target)].reset_index(drop=True)
        y = frame['target'].to_numpy(float)
        base = frame['candidate_prediction'].to_numpy(float)
        groups = frame['group'].astype(str).to_numpy(object)
        folds = c340_wrapper_grouped_folds(groups)
        indices = np.asarray([key_index[value] for value in frame['canonical'].astype(str)], dtype=int)
        candidate = np.full(len(frame), np.nan, dtype=float)
        for fold in sorted(np.unique(folds)):
            valid = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            alpha, weight, _inner = c340_wrapper_select_inner(X[indices[train_idx]], y[train_idx], base[train_idx], groups[train_idx])
            residual = c340_wrapper_fit_predict(X[indices[train_idx]], y[train_idx] - base[train_idx], X[indices[valid]], alpha)
            candidate[valid] = base[valid] + weight * residual
        alpha_full, weight_full, _full_inner = c340_wrapper_select_inner(X[indices], y, base, groups)
        target_test = parent_test[parent_test['target_type'].astype(str).eq(target)].copy()
        test_indices = np.asarray([key_index[value] for value in target_test['canonical'].astype(str)], dtype=int)
        residual_test = c340_wrapper_fit_predict(X[indices], y - base, X[test_indices], alpha_full)
        target_test['candidate'] = target_test['target'].to_numpy(float) + weight_full * residual_test
        test_parts.append(target_test[['id', 'target_type', 'candidate']])
    candidates = pd.concat(test_parts, ignore_index=True).rename(columns={'candidate': 'target'})
    submission = test[['id']].merge(candidates[['id', 'target']], on='id', how='left', validate='one_to_one').sort_values('id').reset_index(drop=True)
    if len(submission) != 4940 or not np.array_equal(submission['id'].to_numpy(), np.arange(1, 4941)) or (not np.isfinite(submission['target'].to_numpy(float)).all()):
        raise RuntimeError('C340 test output contract failed')
    return submission[['id', 'target']]

# ===== c927.py =====
"""C927 no-archive C282 repeat-view residual wrapper (in-memory port).

Faithful in-memory port of:

  * the original builder script  (the C927 wrapper)
  * round2_c278_repeat_view_nested_portfolio.py             (the reused engine)

C927 takes the current-only C282 parent arm and fits a "repeat view" residual
per target on top of it.  The parent arm is the C282 artifact set materialized
by C340 into the C278/C279-compatible schema:

  c282_oof_df        == parent_c282_oof_for_c279.csv
                        columns: canonical, target_type, target, candidate_prediction, group
  c282_submission_df == parent_c282_test_for_c279.csv
                        columns: id, target

For every target the engine builds three views of each canonical structure
(open / hydrogen-capped / ring-closed repeat surrogate), stacks their Morgan
fingerprints + mean fingerprint + compact descriptors into one feature row,
then fits a Ridge residual (residual = target - parent prediction).  The
nonnegative blend weight applied to that residual is selected ONLY inside inner
grouped folds, and the selected weight is applied to the untouched outer fold
and to full-data test inference with the exact same formula:

    output = parent_prediction + selected_weight * residual

All CSV reads are replaced by the passed DataFrames, all CSV writes by the
returned frame, and every artifact-tracking and protocol step is stripped.
"""
# reference namespace is defined above in this standalone module
RDLogger.DisableLog('rdApp.*')
c927_TARGETS = tuple(reference.TARGETS)
c927_BITS = 256
c927_RIDGE_ALPHA = 100.0
c927_WEIGHT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

def c927_remove_dummy_caps(smiles: str) -> Chem.Mol | None:
    """Remove polymer attachment-point dummies and let RDKit add hydrogens."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        for atom_index in sorted([atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0], reverse=True):
            editable.RemoveAtom(atom_index)
        capped = editable.GetMol()
        Chem.SanitizeMol(capped)
        return Chem.AddHs(capped)
    except Exception:
        return None

def c927_ring_close_dummy_caps(smiles: str) -> Chem.Mol | None:
    """Close the two dummy-attachment neighbors into a cyclic repeat surrogate."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        dummy = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy) != 2:
            return None
        neighbors: list[int] = []
        for atom_index in dummy:
            atom = editable.GetAtomWithIdx(atom_index)
            heavy_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 0]
            if not heavy_neighbors:
                return None
            neighbors.append(int(heavy_neighbors[0]))
        if neighbors[0] == neighbors[1]:
            return None
        if editable.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is None:
            editable.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
        for atom_index in sorted(dummy, reverse=True):
            editable.RemoveAtom(atom_index)
        closed = editable.GetMol()
        Chem.SanitizeMol(closed)
        return Chem.AddHs(closed)
    except Exception:
        return None

def c927_fp_bits(molecule: Chem.Mol | None) -> np.ndarray:
    out = np.zeros(c927_BITS, dtype=np.float32)
    if molecule is None:
        return out
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=c927_BITS, useChirality=False)
        DataStructs.ConvertToNumpyArray(fp, out)
    except Exception:
        pass
    return out

def c927_compact_descriptors(molecule: Chem.Mol | None) -> np.ndarray:
    if molecule is None:
        return np.full(10, np.nan, dtype=np.float32)
    try:
        return np.asarray([Descriptors.MolWt(molecule), Descriptors.MolLogP(molecule), rdMolDescriptors.CalcTPSA(molecule), Lipinski.NumRotatableBonds(molecule), rdMolDescriptors.CalcNumRings(molecule), rdMolDescriptors.CalcFractionCSP3(molecule), molecule.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumHBA(molecule), rdMolDescriptors.CalcNumHBD(molecule), rdMolDescriptors.CalcLabuteASA(molecule)], dtype=np.float32)
    except Exception:
        return np.full(10, np.nan, dtype=np.float32)

def c927_build_view_features(keys: list[str], molecules: list[Chem.Mol | None]) -> tuple[np.ndarray, dict[str, int]]:
    rows: list[np.ndarray] = []
    supported_hcap = 0
    supported_ring = 0
    for key, original in zip(keys, molecules):
        capped = c927_remove_dummy_caps(key)
        periodic = c927_ring_close_dummy_caps(key)
        if capped is not None:
            supported_hcap += 1
        if periodic is not None:
            supported_ring += 1
        views = np.vstack([c927_fp_bits(original), c927_fp_bits(capped), c927_fp_bits(periodic)])
        mean_fp = views.mean(axis=0)
        descriptors = np.concatenate([c927_compact_descriptors(original), c927_compact_descriptors(capped), c927_compact_descriptors(periodic)])
        rows.append(np.concatenate([views.reshape(-1), mean_fp, descriptors]))
    return (np.asarray(rows, dtype=np.float32), {'hcap_supported': supported_hcap, 'periodic_supported': supported_ring})

def c927_grouped_folds(groups: np.ndarray) -> np.ndarray:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        return np.zeros(len(groups), dtype=np.int64)
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[valid] = fold
    return folds

def c927_fit_residual(X_train: np.ndarray, residual_train: np.ndarray, X_valid: np.ndarray) -> np.ndarray:
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=c927_RIDGE_ALPHA, solver='lsqr', max_iter=5000, tol=0.0001))
    model.fit(X_train, residual_train)
    return np.asarray(model.predict(X_valid), dtype=float)

def c927_choose_weight(X: np.ndarray, y: np.ndarray, base: np.ndarray, groups: np.ndarray, seed_tag: str) -> tuple[float, dict]:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        return (0.0, {'seed_tag': seed_tag, 'inner_folds': 1, 'scores': {str(w): None for w in c927_WEIGHT_GRID}})
    splitter = GroupKFold(n_splits=n_splits)
    predictions: dict[float, np.ndarray] = {weight: np.full(len(y), np.nan, dtype=float) for weight in c927_WEIGHT_GRID}
    for train_idx, valid_idx in splitter.split(X, y, groups=groups):
        residual = c927_fit_residual(X[train_idx], y[train_idx] - base[train_idx], X[valid_idx])
        for weight in c927_WEIGHT_GRID:
            predictions[weight][valid_idx] = base[valid_idx] + weight * residual
    scores = {weight: float(r2_score(y, predictions[weight])) for weight in c927_WEIGHT_GRID}
    best = max(c927_WEIGHT_GRID, key=lambda weight: (scores[weight], -weight))
    return (float(best), {'seed_tag': seed_tag, 'inner_folds': n_splits, 'scores': {str(k): v for k, v in scores.items()}})

def c927_build_c927(test_df, c282_submission_df, c282_oof_df):
    """C927 repeat-view wrapper. Returns (id,target) DataFrame."""
    oof = c282_oof_df
    parent_test = c282_submission_df
    test = test_df.copy()
    required = {'canonical', 'target_type', 'target', 'candidate_prediction', 'group'}
    missing = required.difference(oof.columns)
    if missing:
        raise RuntimeError(f'missing C282 OOF columns: {sorted(missing)}')
    if not {'id', 'target'}.issubset(parent_test.columns):
        raise RuntimeError('C282 parent test requires id and target columns')
    test['canonical'] = test['smiles'].map(reference.canonicalize)
    parent_test = parent_test.merge(test[['id', 'target_type', 'canonical']], on='id', how='left', validate='one_to_one')
    if parent_test['target'].isna().any() or len(parent_test) != 4940:
        raise RuntimeError('C282 parent test contract failed')
    keys = sorted(set(oof['canonical'].astype(str)) | set(test['canonical'].astype(str)))
    molecules = reference.build_molecules(keys)
    key_index = {key: i for i, key in enumerate(keys)}
    X, _support = c927_build_view_features(keys, molecules)
    test_parts: list[pd.DataFrame] = []
    for target in c927_TARGETS:
        frame = oof[oof['target_type'].astype(str).eq(target)].reset_index(drop=True)
        y = frame['target'].to_numpy(float)
        base = frame['candidate_prediction'].to_numpy(float)
        groups = frame['group'].astype(str).to_numpy(object)
        folds = c927_grouped_folds(groups)
        indices = np.asarray([key_index[value] for value in frame['canonical'].astype(str)], dtype=int)
        candidate = np.full(len(frame), np.nan, dtype=float)
        for fold in sorted(np.unique(folds)):
            valid = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            chosen, _inner = c927_choose_weight(X[indices[train_idx]], y[train_idx], base[train_idx], groups[train_idx], f'{target}-outer-{fold}')
            residual = c927_fit_residual(X[indices[train_idx]], y[train_idx] - base[train_idx], X[indices[valid]])
            candidate[valid] = base[valid] + chosen * residual
        chosen_full, _full_inner = c927_choose_weight(X[indices], y, base, groups, f'{target}-full')
        full_residual = c927_fit_residual(X[indices], y - base, X[np.asarray([key_index[value] for value in parent_test[parent_test['target_type'].astype(str).eq(target)]['canonical'].astype(str)], dtype=int)])
        target_test = parent_test[parent_test['target_type'].astype(str).eq(target)].copy()
        target_test['candidate'] = target_test['target'].to_numpy(float) + chosen_full * full_residual
        test_parts.append(target_test[['id', 'target_type', 'candidate']])
    candidates = pd.concat(test_parts, ignore_index=True).rename(columns={'candidate': 'target'})
    submission = test[['id']].merge(candidates[['id', 'target']], on='id', how='left', validate='one_to_one').sort_values('id').reset_index(drop=True)
    if len(submission) != 4940 or not np.array_equal(submission['id'].to_numpy(), np.arange(1, 4941)) or (not np.isfinite(submission['target'].to_numpy(float)).all()):
        raise RuntimeError('C927 full-test output contract failed')
    return submission

# ===== c1369_stack.py =====
"""In-memory port of the original round C1369 "branch-local fast direct stack".

Faithful port of the original builder script, which imports
the following pure-source helpers that are inlined here (no runtime imports of
other files):

- initial_reference_pipeline (reference): canonicalize, build_label_pool,
  build_molecules, descriptor_matrix, physical_matrix, morgan_count_matrix,
  cross_property_arrays, target_dense_features, clip_prediction, blend_from_oof
- the original builder script (c385): maccs_matrix, sanitize_dense
- round2_c097_graph_grammar_hgb_full (graph): grammar_features, no_stereo
- round2_c282_current_only_reference (c282): current-only input handling

All seeds, model hyperparameters, feature blocks, clipping, blending (nnls), and
the exact-override passes are reproduced verbatim. No file IO, no hashes, no
path guards, no manifest.

Extra DataFrames beyond train_df/test_df/base_df (passed via the optional
'components' dict):
    components["archive"]  -> the official archive/train.csv labels DataFrame
        (columns: smiles, target, target_type). When provided, the label pool is
        built from current train + archive (the original "with_archive" branch);
        when omitted, the pool is current-only (the original "without_archive"
        branch, archive treated as empty). No other intermediate CSVs are read:
        the c282/c284/oof mentions are source-code imports only, not disk reads.
"""
RDLogger.DisableLog('rdApp.*')
c1369_stack_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
c1369_stack_SEED = 20260808

def c1369_stack__canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def c1369_stack__build_label_pool(train: pd.DataFrame, archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = train[['smiles', 'canonical', 'target_type', 'target']].copy()
    current['source'] = 'current_train'
    old = archive[['smiles', 'canonical', 'target_type', 'target']].copy()
    old['source'] = 'archive_train'
    raw = pd.concat([current, old], ignore_index=True)
    raw = raw.drop_duplicates(['smiles', 'target_type', 'target']).reset_index(drop=True)
    pooled = raw.groupby(['canonical', 'target_type'], as_index=False).agg(target=('target', 'median'), smiles=('smiles', 'first'), measurements=('target', 'size'))
    return (raw, pooled)

def c1369_stack__build_molecules(keys: list[str]) -> list[Any]:
    molecules = [Chem.MolFromSmiles(value) for value in keys]
    if any((molecule is None for molecule in molecules)):
        raise RuntimeError('Canonical official structure failed RDKit parsing')
    return molecules

def c1369_stack__descriptor_matrix(molecules: list[Any]) -> tuple[np.ndarray, list[str]]:
    items = list(Descriptors._descList)
    matrix = np.full((len(molecules), len(items)), np.nan, dtype=np.float64)
    for row, molecule in enumerate(molecules):
        for column, (_, function) in enumerate(items):
            try:
                value = float(function(molecule))
            except Exception:
                value = math.nan
            matrix[row, column] = value if math.isfinite(value) else math.nan
    return (matrix, [name for name, _ in items])

def c1369_stack__physical_matrix(molecules: list[Any], smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ['smiles_length', 'atom_count', 'heavy_atom_count', 'dummy_atom_count', 'ring_count', 'aromatic_atom_count', 'hetero_atom_count', 'halogen_count', 'rotatable_bonds_approx', 'double_bond_count', 'triple_bond_count', 'branch_count', 'n_count', 'o_count', 's_count', 'si_count']
    matrix = np.zeros((len(molecules), len(names)), dtype=np.float64)
    for row, (molecule, value) in enumerate(zip(molecules, smiles, strict=True)):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        matrix[row] = [len(value), molecule.GetNumAtoms(), molecule.GetNumHeavyAtoms(), sum((atom.GetAtomicNum() == 0 for atom in atoms)), molecule.GetRingInfo().NumRings(), sum((atom.GetIsAromatic() for atom in atoms)), sum((atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)), sum((atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)), sum((bond.GetBondTypeAsDouble() == 1.0 and (not bond.IsInRing()) for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 2.0 for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 3.0 for bond in bonds)), value.count('('), sum((atom.GetAtomicNum() == 7 for atom in atoms)), sum((atom.GetAtomicNum() == 8 for atom in atoms)), sum((atom.GetAtomicNum() == 16 for atom in atoms)), sum((atom.GetAtomicNum() == 14 for atom in atoms))]
    return (matrix, names)

def c1369_stack__morgan_count_matrix(molecules: list[Any], radius: int, bits: int) -> sparse.csr_matrix:
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

def c1369_stack__cross_property_arrays(pooled: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pivot = pooled.pivot(index='canonical', columns='target_type', values='target')
    values = np.full((len(keys), len(c1369_stack_TARGETS)), np.nan, dtype=np.float64)
    available = np.zeros((len(keys), len(c1369_stack_TARGETS)), dtype=np.float64)
    key_position = {key: index for index, key in enumerate(keys)}
    for target_index, target in enumerate(c1369_stack_TARGETS):
        if target not in pivot:
            continue
        series = pivot[target].dropna()
        for key, value in series.items():
            position = key_position.get(key)
            if position is not None:
                values[position, target_index] = float(value)
                available[position, target_index] = 1.0
    return (values, available)

def c1369_stack__target_dense_features(base_dense: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray, target: str) -> np.ndarray:
    values = cross_values.copy()
    available = cross_available.copy()
    target_index = c1369_stack_TARGETS.index(target)
    values[:, target_index] = np.nan
    available[:, target_index] = 0.0
    return np.hstack([base_dense, values, available]).astype(np.float64, copy=False)

def c1369_stack__clip_prediction(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    return np.clip(prediction, q01 - 2.0 * margin, q99 + 2.0 * margin)

def c1369_stack__blend_from_oof(y: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, float, str, float]:
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
        return (weights, intercept, f'base_{best_index}', base_scores[best_index])
    return (weights, intercept, 'nonnegative_blend', blend_score)

def c1369_stack__maccs_matrix(molecules: list[Any]) -> np.ndarray:
    rows = np.zeros((len(molecules), 167), dtype=np.float32)
    for i, mol in enumerate(molecules):
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows[i] = np.asarray([int(fp.GetBit(j)) for j in range(167)], dtype=np.float32)
    return rows

def c1369_stack__sanitize_dense(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(out) | (np.abs(out) > 1000000000000.0)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out

def c1369_stack__no_stereo(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError('official SMILES failed RDKit parsing')
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)

def c1369_stack__add_token(tokens: dict[str, float], token: str, value: float=1.0) -> None:
    tokens[token] = tokens.get(token, 0.0) + float(value)

def c1369_stack__grammar_row(molecule: Chem.Mol) -> tuple[dict[str, float], list[float]]:
    tokens: dict[str, float] = {}
    atoms = list(molecule.GetAtoms())
    bonds = list(molecule.GetBonds())
    numeric = [float(len(atoms)), float(len(bonds)), float(molecule.GetRingInfo().NumRings()), float(sum((atom.GetIsAromatic() for atom in atoms))), float(sum((atom.GetFormalCharge() != 0 for atom in atoms))), float(sum((atom.GetAtomicNum() == 6 for atom in atoms))), float(sum((atom.GetAtomicNum() not in (1, 6) for atom in atoms))), float(sum((atom.GetDegree() >= 3 for atom in atoms))), float(max((atom.GetDegree() for atom in atoms), default=0))]
    labels: list[str] = []
    common_elements = (0, 1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
    numeric.extend((float(sum((atom.GetAtomicNum() == value for atom in atoms))) for value in common_elements))
    numeric.extend((float(sum((atom.GetDegree() == value for atom in atoms))) for value in range(0, 7)))
    for atom in atoms:
        atomic = atom.GetAtomicNum()
        degree = atom.GetDegree()
        aromatic = int(atom.GetIsAromatic())
        charge = atom.GetFormalCharge()
        hybrid = str(atom.GetHybridization())
        label = f'z{atomic}|d{degree}|a{aromatic}|q{charge}|h{hybrid}'
        labels.append(label)
        c1369_stack__add_token(tokens, f'atom:{label}')
        c1369_stack__add_token(tokens, f'element:{atomic}')
        if atom.IsInRing():
            c1369_stack__add_token(tokens, f'ring_atom:{atomic}')
        if atom.GetDegree() >= 3:
            c1369_stack__add_token(tokens, f'branch_atom:{atomic}:d{degree}')
    for bond in bonds:
        begin = molecule.GetAtomWithIdx(bond.GetBeginAtomIdx())
        end = molecule.GetAtomWithIdx(bond.GetEndAtomIdx())
        kind = str(bond.GetBondType())
        endpoint = 'dummy' if begin.GetAtomicNum() == 0 or end.GetAtomicNum() == 0 else 'heavy'
        c1369_stack__add_token(tokens, f'bond:{kind}:{endpoint}')
        if bond.GetIsAromatic():
            c1369_stack__add_token(tokens, 'bond:aromatic')
    numeric.extend((float(sum((bond.GetBondType() == value for bond in bonds))) for value in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)))
    numeric.append(float(sum((bond.GetIsAromatic() for bond in bonds))))
    for ring in molecule.GetRingInfo().AtomRings():
        size = len(ring)
        c1369_stack__add_token(tokens, f'ring_size:{min(size, 12)}')
    numeric.extend((float(sum((len(ring) == value for ring in molecule.GetRingInfo().AtomRings()))) for value in range(3, 9)))
    numeric.extend([float(sum((atom.GetAtomicNum() == 0 for atom in atoms))), float(sum((atom.GetIsAromatic() for atom in atoms))), float(sum((atom.GetFormalCharge() for atom in atoms)))])
    for round_id in (1, 2):
        next_labels: list[str] = []
        for atom in atoms:
            neighbours = sorted((labels[neighbour.GetIdx()] for neighbour in atom.GetNeighbors()))
            next_label = f"{labels[atom.GetIdx()]}>>{'/'.join(neighbours)}"
            next_labels.append(next_label)
            c1369_stack__add_token(tokens, f'wl{round_id}:{next_label}')
        labels = next_labels
    return (tokens, numeric)

def c1369_stack__grammar_features(molecules: list[Chem.Mol]) -> np.ndarray:
    rows: list[dict[str, float]] = []
    numeric_rows: list[list[float]] = []
    for molecule in molecules:
        tokens, numeric = c1369_stack__grammar_row(molecule)
        rows.append(tokens)
        numeric_rows.append(numeric)
    hashed = FeatureHasher(n_features=1024, input_type='dict', alternate_sign=False, dtype=np.float64).transform(rows).toarray()
    width = max((len(row) for row in numeric_rows))
    numeric_matrix = np.zeros((len(numeric_rows), width), dtype=np.float64)
    for index, row in enumerate(numeric_rows):
        numeric_matrix[index, :len(row)] = row
    return np.hstack([numeric_matrix, hashed]).astype(np.float64, copy=False)

def c1369_stack__parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple((item.strip().lower() for item in value.split(',') if item.strip()))
    if not targets:
        raise RuntimeError('No targets requested')
    bad = [target for target in targets if target not in c1369_stack_TARGETS]
    if bad:
        raise RuntimeError(f'Invalid targets: {bad}')
    return targets

def c1369_stack__feature_matrix(parent: dict[str, Any], morgan_bits: int) -> tuple[np.ndarray, dict[str, Any]]:
    molecules = parent['molecules']
    descriptor, descriptor_names = c1369_stack__descriptor_matrix(molecules)
    physical, physical_names = c1369_stack__physical_matrix(molecules, parent['keys'])
    morgan2 = c1369_stack__morgan_count_matrix(molecules, radius=2, bits=morgan_bits).toarray().astype(np.float32)
    morgan3 = c1369_stack__morgan_count_matrix(molecules, radius=3, bits=morgan_bits).toarray().astype(np.float32)
    maccs = c1369_stack__maccs_matrix(molecules)
    grammar = c1369_stack__grammar_features(molecules).astype(np.float32)
    matrix = c1369_stack__sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs, grammar]).astype(np.float32))
    return (matrix, {'shape': [int(value) for value in matrix.shape], 'rdkit_descriptors': int(len(descriptor_names)), 'physical_features': int(len(physical_names)), 'morgan_bits_each': int(morgan_bits), 'maccs_bits': int(maccs.shape[1]), 'graph_grammar_features': int(grammar.shape[1])})

def c1369_stack__model_factories(target: str) -> dict[str, Any]:
    weak = target in {'ei', 'eea', 'egb', 'eps', 'nc'}
    return {'ridge_20': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=20.0, solver='lsqr', max_iter=5000, tol=0.0001)), 'ridge_80': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=80.0, solver='lsqr', max_iter=5000, tol=0.0001)), 'ridge_250': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=250.0, solver='lsqr', max_iter=5000, tol=0.0001)), 'hist_gbdt': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), HistGradientBoostingRegressor(max_iter=180 if weak else 140, learning_rate=0.04, max_leaf_nodes=15 if weak else 31, min_samples_leaf=8 if weak else 20, l2_regularization=0.3, random_state=seed)), 'extra_trees': lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), ExtraTreesRegressor(n_estimators=180 if weak else 120, min_samples_leaf=2 if weak else 4, max_features=0.55 if weak else 0.45, random_state=seed, n_jobs=4))}

def c1369_stack__grouped_oof(factory: Any, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError('Not enough groups for grouped OOF')
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x[tr], y[tr])
        pred = c1369_stack__clip_prediction(y[tr], np.asarray(model.predict(x[va]), dtype=np.float64))
        oof[va] = pred
        rows.append({'fold': int(fold), 'rows': int(len(va)), 'r2': float(r2_score(y[va], pred))})
    if not np.isfinite(oof).all():
        raise RuntimeError('Non-finite OOF predictions')
    return (oof, rows)

def c1369_stack__run_target(parent: dict[str, Any], base_features: np.ndarray, target: str, models: tuple[str, ...], seed: int) -> dict[str, Any]:
    pooled = parent['pooled']
    test = parent['test']
    key_to_index = parent['key_to_index']
    target_train = pooled.loc[pooled['target_type'].astype(str).eq(target)].reset_index(drop=True)
    target_test = test.loc[test['target_type'].astype(str).eq(target)].reset_index(drop=True)
    train_idx = np.asarray([key_to_index[value] for value in target_train['canonical']], dtype=np.int64)
    test_idx = np.asarray([key_to_index[value] for value in target_test['canonical']], dtype=np.int64)
    y = target_train['target'].to_numpy(np.float64)
    groups = np.asarray([c1369_stack__no_stereo(value) for value in target_train['canonical'].astype(str)], dtype=object)
    cross_values, cross_available = c1369_stack__cross_property_arrays(pooled, parent['keys'])
    dense = c1369_stack__sanitize_dense(c1369_stack__target_dense_features(base_features, cross_values, cross_available, target))
    x_train = dense[train_idx]
    x_test = dense[test_idx]
    available = c1369_stack__model_factories(target)
    factories = {name: available[name] for name in models if name in available}
    if not factories:
        raise RuntimeError('No usable model factories requested')
    oof_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    reports: dict[str, Any] = {}
    for pos, (name, factory) in enumerate(factories.items()):
        model_seed = seed + 1000 * c1369_stack_TARGETS.index(target) + 37 * pos
        oof, folds = c1369_stack__grouped_oof(factory, x_train, y, groups, model_seed)
        final_model = factory(model_seed + 999)
        final_model.fit(x_train, y)
        test_pred = c1369_stack__clip_prediction(y, np.asarray(final_model.predict(x_test), dtype=np.float64))
        oof_columns.append(oof)
        test_columns.append(test_pred)
        reports[name] = {'oof_r2': float(r2_score(y, oof)), 'folds': folds}
    oof_stack = np.column_stack(oof_columns)
    weights, intercept, blend_name, blend_r2 = c1369_stack__blend_from_oof(y, oof_stack)
    candidate_oof = c1369_stack__clip_prediction(y, oof_stack @ weights + intercept)
    test_stack = np.column_stack(test_columns)
    candidate_test = c1369_stack__clip_prediction(y, test_stack @ weights + intercept)
    lookup = target_train.groupby('canonical')['target'].median().to_dict()
    exact_overrides = 0
    for pos, row in enumerate(target_test.itertuples(index=False)):
        if row.canonical in lookup:
            candidate_test[pos] = float(lookup[row.canonical])
            exact_overrides += 1
    return {'target': target, 'train_rows': int(len(y)), 'test_rows': int(len(target_test)), 'candidate_oof_r2': float(r2_score(y, candidate_oof)), 'model_reports': reports, 'blend_name': blend_name, 'blend_weights': {name: float(value) for name, value in zip(factories.keys(), weights, strict=True)}, 'blend_intercept': float(intercept), 'blend_oof_r2': float(blend_r2), 'exact_train_overrides_on_test': int(exact_overrides), 'test_ids': target_test['id'].to_numpy(np.int64), 'test_pred': candidate_test.astype(np.float64)}

def c1369_stack__apply_branch_overrides(parent: dict[str, Any], assembled: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = parent['raw_labels']
    test = parent['test']
    lookup = raw.groupby(['canonical', 'target_type'])['target'].median().to_dict()
    values = assembled['target'].to_numpy(np.float64).copy()
    overrides = 0
    for pos, row in enumerate(test.itertuples(index=False)):
        key = (row.canonical, row.target_type)
        if key in lookup:
            values[pos] = float(lookup[key])
            overrides += 1
    out = assembled.copy()
    out['target'] = values
    return (out, {'branch_label_exact_overrides': int(overrides)})

def c1369_stack_fast_direct_stack(train_df: pd.DataFrame, test_df: pd.DataFrame, base_df: pd.DataFrame, targets: str='egc,ei,nc,eps', models: str='ridge_20,ridge_80,ridge_250', morgan_bits: int=256, seed: int=c1369_stack_SEED, components: dict[str, Any] | None=None) -> pd.DataFrame:
    """Stack Ridge models on features built from the parent candidate.

    train_df: official train (columns smiles, target, target_type).
    test_df: official test (columns id, smiles, target_type).
    base_df: frozen base predictions (columns id, target).
    components: optional dict; components["archive"] may carry the
        archive/train.csv labels DataFrame to enable the with_archive label pool.
    Returns a DataFrame with columns [id, target].
    """
    components = components or {}
    train = train_df.copy()
    test = test_df.copy()
    base = base_df.copy()
    archive = components.get('archive')
    if archive is None:
        archive = train.iloc[0:0].copy()
    else:
        archive = archive.copy()
    for frame in (train, test, archive):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [c1369_stack__canonicalize(value) for value in frame['smiles']]
    ids = test['id'].to_numpy(np.int64)
    if list(base.columns) != ['id', 'target']:
        raise RuntimeError('Unexpected base candidate schema')
    if not np.array_equal(base['id'].to_numpy(np.int64), ids):
        raise RuntimeError('Base candidate IDs/order do not match official test')
    base_values = base['target'].to_numpy(np.float64)
    if not np.isfinite(base_values).all():
        raise RuntimeError('Base candidate contains non-finite predictions')
    raw_labels, pooled = c1369_stack__build_label_pool(train, archive)
    keys = sorted(set(pooled['canonical']) | set(test['canonical']))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = c1369_stack__build_molecules(keys)
    parent: dict[str, Any] = {'train': train, 'test': test, 'archive': archive, 'raw_labels': raw_labels, 'pooled': pooled, 'keys': keys, 'key_to_index': key_to_index, 'molecules': molecules}
    predictions = base_values.copy()
    x, _ = c1369_stack__feature_matrix(parent, int(morgan_bits))
    active_targets = c1369_stack__parse_targets(targets)
    model_names = tuple((item.strip() for item in models.split(',') if item.strip()))
    for target in active_targets:
        result = c1369_stack__run_target(parent, x, target, model_names, int(seed))
        id_to_value = dict(zip(result['test_ids'].astype(int), result['test_pred'].astype(float), strict=True))
        mask = parent['test']['target_type'].astype(str).eq(target).to_numpy()
        predictions[mask] = parent['test'].loc[mask, 'id'].astype(int).map(id_to_value).to_numpy(np.float64)
    assembled = pd.DataFrame({'id': ids, 'target': predictions})
    assembled, _ = c1369_stack__apply_branch_overrides(parent, assembled)
    if len(assembled) != len(test) or assembled['id'].duplicated().any() or (not np.array_equal(assembled['id'].to_numpy(np.int64), ids)):
        raise RuntimeError('Output row/order contract failed')
    if not np.isfinite(assembled['target'].to_numpy(np.float64)).all():
        raise RuntimeError('Non-finite output prediction')
    return assembled[['id', 'target']].reset_index(drop=True)

# ===== c1446_physics_proj.py =====
"""In-memory port of the original builder script.

C1446/C1494 joint physics projection.  Takes official train/test and one branch
base candidate as pandas DataFrames and returns the 4,940-row (id, target)
submission with the three fixed co-test consistency projections applied.

Deliberate non-behavioral deviations from the original round source:
  * No path guards, no digest/digest module/manifest, no argparse/argv, no CSV I/O --
    inputs arrive as DataFrames and the result is returned as a DataFrame.
  * Runtime ID/base-schema validation is retained (raises on mismatch) but
    reports via RuntimeError rather than a manifest JSON.
  * Manifest-only logging (relations/applied) is dropped.

The numeric computation is identical to the source:
  - canonicalization: RDKit canonical isomeric SMILES (after "[*]" -> "*");
  - Egc->Egb affine: Ridge(alpha=0.01) on paired (egc, egb) train structures;
  - (Ei - Eea) -> Egc gap identity: HuberRegressor(alpha=0.01, epsilon=1.35,
    max_iter=1000) on train triples;
  - EPS/Nc median-ionic consistency: ionic = eps - nc**2, median clipped to
    >= 0.02, projected onto eps = nc**2 + ionic via a 901-point grid over
    nc in [1.0, 2.8];
  - pulls: egb_pull / gap_pull / epsnc_pull blend each projection toward the
    base values; final per-target quantile clip (0.002/0.998 +/- 2*margin,
    margin = max(IQR, std, 1e-8), lower bounded at 0 for eps/nc).
"""
RDLogger.DisableLog('rdApp.*')
c1446_physics_proj_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

def c1446_physics_proj__canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def c1446_physics_proj__fit_affine(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 10:
        return {'intercept': 0.0, 'slope': 1.0, 'rows': int(len(x)), 'r2': float('nan'), 'fallback': True}
    model = Ridge(alpha=0.01)
    model.fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return {'intercept': float(model.intercept_), 'slope': float(model.coef_[0]), 'rows': int(len(x)), 'r2': float(1.0 - ss_res / max(ss_tot, 1e-12)), 'fallback': False}

def c1446_physics_proj__fit_gap_relation(wide: pd.DataFrame) -> dict:
    rows = wide[['egc', 'ei', 'eea']].dropna()
    if len(rows) < 10:
        return {'intercept': 0.0, 'slope': 1.0, 'rows': int(len(rows)), 'r2': float('nan'), 'fallback': True}
    x = (rows['ei'].to_numpy(float) - rows['eea'].to_numpy(float)).reshape(-1, 1)
    y = rows['egc'].to_numpy(float)
    model = HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=1000)
    model.fit(x, y)
    pred = model.predict(x)
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return {'intercept': float(model.intercept_), 'slope': float(model.coef_[0]), 'rows': int(len(rows)), 'r2': float(1.0 - ss_res / max(ss_tot, 1e-12)), 'fallback': False}

def c1446_physics_proj__project_egc_egb(egc0: float, egb0: float, relation: dict, pull: float):
    a = float(relation['intercept'])
    b = float(relation['slope'])
    x_star = (egc0 + b * (egb0 - a)) / max(1.0 + b * b, 1e-12)
    y_star = a + b * x_star
    return ((1.0 - pull) * egc0 + pull * x_star, (1.0 - pull) * egb0 + pull * y_star)

def c1446_physics_proj__project_gap(ei0: float, eea0: float, egc0: float, relation: dict, pull: float):
    a = float(relation['intercept'])
    b = float(relation['slope'])
    gap_from_ei_eea = a + b * (ei0 - eea0)
    if abs(b) < 1e-08:
        ei_target = ei0
        eea_target = eea0
    else:
        correction = (egc0 - gap_from_ei_eea) / (2.0 * b)
        ei_target = ei0 + correction
        eea_target = eea0 - correction
    egc_target = gap_from_ei_eea
    return ((1.0 - pull) * ei0 + pull * ei_target, (1.0 - pull) * eea0 + pull * eea_target, (1.0 - pull) * egc0 + pull * egc_target)

def c1446_physics_proj__project_eps_nc(eps0: float, nc0: float, ionic: float, pull: float):
    grid = np.linspace(1.0, 2.8, 901)
    eps_grid = grid * grid + ionic
    loss = np.square(eps_grid - eps0) + np.square(grid - nc0)
    idx = int(np.argmin(loss))
    nc_star = float(grid[idx])
    eps_star = float(eps_grid[idx])
    return ((1.0 - pull) * eps0 + pull * eps_star, (1.0 - pull) * nc0 + pull * nc_star)

def c1446_physics_proj__clip_target(values: np.ndarray, target: str, train: pd.DataFrame) -> np.ndarray:
    y = train.loc[train['target_type'] == target, 'target'].to_numpy(float)
    q001, q999 = np.quantile(y, [0.002, 0.998])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    lower = max(0.0, q001 - 2.0 * margin) if target in {'eps', 'nc'} else q001 - 2.0 * margin
    upper = q999 + 2.0 * margin
    return np.clip(values, lower, upper)

def c1446_physics_proj_c1446_physics_projection(train_df: pd.DataFrame, test_df: pd.DataFrame, base_df: pd.DataFrame, egb_pull: float=0.08, gap_pull: float=0.05, epsnc_pull: float=0.02) -> pd.DataFrame:
    """Apply the C1446 joint physics projection to base_df.

    Parameters
    ----------
    train_df : official train.csv (smiles, target, target_type)
    test_df  : official test.csv (id, smiles, target_type)
    base_df  : candidate (id, target), ids 1..4940 in order
    egb_pull, gap_pull, epsnc_pull : projection blend pulls in [0, 0.5]

    Returns
    -------
    pd.DataFrame with columns ["id", "target"], 4,940 rows, ids 1..4940.
    """
    train = train_df.copy()
    test = test_df.reset_index(drop=True).copy()
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [c1446_physics_proj__canonicalize(value) for value in frame['smiles']]
    ids = test['id'].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    base = base_df.copy()
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Base ID order mismatch')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base contains non-finite values')
    for name, value in (('egb_pull', egb_pull), ('gap_pull', gap_pull), ('epsnc_pull', epsnc_pull)):
        if not 0.0 <= float(value) <= 0.5:
            raise RuntimeError(f'{name} outside [0, 0.5]')
    values = base['target'].to_numpy(float).copy()
    train_wide = train.pivot_table(index='canonical', columns='target_type', values='target', aggfunc='mean')
    eg_rows = train_wide[['egc', 'egb']].dropna()
    egb_relation = c1446_physics_proj__fit_affine(eg_rows['egc'].to_numpy(float), eg_rows['egb'].to_numpy(float))
    gap_relation = c1446_physics_proj__fit_gap_relation(train_wide)
    epsnc_rows = train_wide[['eps', 'nc']].dropna()
    ionic = epsnc_rows['eps'].to_numpy(float) - np.square(epsnc_rows['nc'].to_numpy(float))
    median_ionic = float(np.median(ionic)) if len(ionic) else 0.2
    median_ionic = max(median_ionic, 0.02)
    pred = test[['id', 'canonical', 'target_type']].copy()
    pred['value'] = values
    wide = pred.pivot_table(index='canonical', columns='target_type', values='value', aggfunc='mean')
    row_index = {(str(row.canonical), str(row.target_type)): int(idx) for idx, row in test.iterrows()}
    for canon, row in wide.iterrows():
        canon = str(canon)
        if {'egc', 'egb'}.issubset(row.index) and pd.notna(row.get('egc')) and pd.notna(row.get('egb')):
            egc_new, egb_new = c1446_physics_proj__project_egc_egb(float(row['egc']), float(row['egb']), egb_relation, float(egb_pull))
            for target, new_value in (('egc', egc_new), ('egb', egb_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
        if {'ei', 'eea', 'egc'}.issubset(row.index) and all((pd.notna(row.get(t)) for t in ('ei', 'eea', 'egc'))):
            ei_new, eea_new, egc_new = c1446_physics_proj__project_gap(float(row['ei']), float(row['eea']), float(row['egc']), gap_relation, float(gap_pull))
            for target, new_value in (('ei', ei_new), ('eea', eea_new), ('egc', egc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
        if {'eps', 'nc'}.issubset(row.index) and pd.notna(row.get('eps')) and pd.notna(row.get('nc')):
            eps_new, nc_new = c1446_physics_proj__project_eps_nc(float(row['eps']), float(row['nc']), median_ionic, float(epsnc_pull))
            for target, new_value in (('eps', eps_new), ('nc', nc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
    for target in c1446_physics_proj_TARGETS:
        mask = test['target_type'].to_numpy(object) == target
        values[mask] = c1446_physics_proj__clip_target(values[mask], target, train)
    if not np.isfinite(values).all():
        raise RuntimeError('Output contains non-finite values')
    return pd.DataFrame({'id': ids, 'target': values})

# ===== c1570_physics_grid.py =====
"""In-memory port of the original builder script.

C1570 branch-local joint physics projection grid, converted to a single
standalone in-memory function.  It takes official train/test/base candidates as
pandas DataFrames and returns one (id, target) frame for a single grid point.

The projection is a fixed, preregistered consistency adjustment learned only
from current official train labels:

  * Egc/Egb affine consistency   : Ridge(alpha=0.01) over train egc -> egb pairs,
    then an orthogonal projection onto the fitted line, blended by egb_pull.
  * Ei/Eea/Egc gap identity      : HuberRegressor(alpha=0.01, epsilon=1.35,
    max_iter=1000) over egc ~ (ei - eea) train triples; egc_target = a + b*(ei-eea)
    and the gap correction is split symmetrically between ei and eea.
  * EPS/Nc median-ionic consistency: EPS == nc^2 + ionic with ionic fixed at the
    train median of (eps - nc^2); nearest point on that curve found on a
    901-point grid over nc in [1.0, 2.8], blended by epsnc_pull.
  * Every target is then clipped to train quantiles [0.002, 0.998] +/- 2 * margin
    where margin = max(IQR, std, 1e-8); eps/nc lower bound floored at 0.

All models/relations are fit from scratch inside this function (fixed seeds are
not required here: Ridge and HuberRegressor are deterministic convex fits with no
random state).  No CSV is read or written, no hashes, no argv, no the original round paths.

Deliberate non-behavioral deviations from the original round source
(manifest/logging/protocol only; no numeric change):
  * Removed file_digest_helper, guard_path, safe_param/safe_slug, manifest.jsonl,
    summary.json, datetime stamps, and all path-guard / env-var / argparse code.
  * Inputs are DataFrames (train_df, test_df, base_df) instead of CSV paths.
  * The source loops over a grid of (egb_pull, gap_pull, epsnc_pull) cells and
    writes one CSV per cell, skipping cells whose projected values equal the
    base (max |values - base_values| <= 1e-12).  Here the grid is collapsed to a
    single point given by scalar parameters and the single projected
    (id, target) frame is returned directly (the given defaults are non-zero
    pulls, so a change occurs and a candidate would have been written).
  * test_df is reset to a positional RangeIndex before use; the source relied on
    test.csv being read fresh with a 0..4939 RangeIndex, so this is equivalent
    for the official data and defensive for any caller index.
"""
try:
    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass
c1570_physics_grid_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

def c1570_physics_grid__canonicalize(smiles: object) -> str:
    """RDKit canonical SMILES, matching initial_reference_pipeline.canonicalize."""
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def c1570_physics_grid__fit_affine(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Ridge fit of y ~ x, with the source's <10-row fallback."""
    if len(x) < 10:
        return {'intercept': 0.0, 'slope': 1.0, 'rows': int(len(x)), 'r2': float('nan'), 'fallback': True}
    model = Ridge(alpha=0.01)
    model.fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return {'intercept': float(model.intercept_), 'slope': float(model.coef_[0]), 'rows': int(len(x)), 'r2': float(1.0 - ss_res / max(ss_tot, 1e-12)), 'fallback': False}

def c1570_physics_grid__fit_gap_relation(wide: pd.DataFrame) -> dict[str, float]:
    """Huber fit of egc ~ (ei - eea), with the source's <10-row fallback."""
    rows = wide[['egc', 'ei', 'eea']].dropna()
    if len(rows) < 10:
        return {'intercept': 0.0, 'slope': 1.0, 'rows': int(len(rows)), 'r2': float('nan'), 'fallback': True}
    x = (rows['ei'].to_numpy(float) - rows['eea'].to_numpy(float)).reshape(-1, 1)
    y = rows['egc'].to_numpy(float)
    model = HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=1000)
    model.fit(x, y)
    pred = model.predict(x)
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return {'intercept': float(model.intercept_), 'slope': float(model.coef_[0]), 'rows': int(len(rows)), 'r2': float(1.0 - ss_res / max(ss_tot, 1e-12)), 'fallback': False}

def c1570_physics_grid__project_egc_egb(egc0: float, egb0: float, relation: dict[str, float], pull: float) -> tuple[float, float]:
    a = float(relation['intercept'])
    b = float(relation['slope'])
    x_star = (egc0 + b * (egb0 - a)) / max(1.0 + b * b, 1e-12)
    y_star = a + b * x_star
    return ((1.0 - pull) * egc0 + pull * x_star, (1.0 - pull) * egb0 + pull * y_star)

def c1570_physics_grid__project_gap(ei0: float, eea0: float, egc0: float, relation: dict[str, float], pull: float) -> tuple[float, float, float]:
    a = float(relation['intercept'])
    b = float(relation['slope'])
    gap_from_ei_eea = a + b * (ei0 - eea0)
    if abs(b) < 1e-08:
        ei_target = ei0
        eea_target = eea0
    else:
        correction = (egc0 - gap_from_ei_eea) / (2.0 * b)
        ei_target = ei0 + correction
        eea_target = eea0 - correction
    egc_target = gap_from_ei_eea
    return ((1.0 - pull) * ei0 + pull * ei_target, (1.0 - pull) * eea0 + pull * eea_target, (1.0 - pull) * egc0 + pull * egc_target)

def c1570_physics_grid__project_eps_nc(eps0: float, nc0: float, ionic: float, pull: float) -> tuple[float, float]:
    grid = np.linspace(1.0, 2.8, 901)
    eps_grid = grid * grid + ionic
    loss = np.square(eps_grid - eps0) + np.square(grid - nc0)
    idx = int(np.argmin(loss))
    nc_star = float(grid[idx])
    eps_star = float(eps_grid[idx])
    return ((1.0 - pull) * eps0 + pull * eps_star, (1.0 - pull) * nc0 + pull * nc_star)

def c1570_physics_grid__clip_target(values: np.ndarray, target: str, train: pd.DataFrame) -> np.ndarray:
    y = train.loc[train['target_type'] == target, 'target'].to_numpy(float)
    q001, q999 = np.quantile(y, [0.002, 0.998])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    lower = max(0.0, q001 - 2.0 * margin) if target in {'eps', 'nc'} else q001 - 2.0 * margin
    upper = q999 + 2.0 * margin
    return np.clip(values, lower, upper)

def c1570_physics_grid_build_c1570_physics_grid(train_df: pd.DataFrame, test_df: pd.DataFrame, base_df: pd.DataFrame, *, egb_pull: float=0.05, gap_pull: float=0.02, epsnc_pull: float=0.0) -> pd.DataFrame:
    """Apply the C1570 joint physics projection grid to a base candidate.

    Parameters
    ----------
    train_df : official train.csv (smiles, target, target_type), 7,409 rows.
    test_df  : official test.csv (id, smiles, target_type), 4,940 rows, ids 1..4940.
    base_df  : branch-local base candidate (id, target), 4,940 rows in id order.
    egb_pull / gap_pull / epsnc_pull : per-property pull strengths in [0, 0.5].

    Returns
    -------
    pd.DataFrame with columns ['id', 'target'] and 4,940 rows (ids 1..4940).
    """
    train = train_df.copy()
    test = test_df.copy().reset_index(drop=True)
    for frame in (train, test):
        frame['target_type'] = frame['target_type'].astype(str).str.lower()
        frame['canonical'] = [c1570_physics_grid__canonicalize(value) for value in frame['smiles']]
    ids = test['id'].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected official test IDs')
    base = base_df
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Base ID order mismatch')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base contains non-finite values')
    base_values = base['target'].to_numpy(float)
    train_wide = train.pivot_table(index='canonical', columns='target_type', values='target', aggfunc='mean')
    eg_rows = train_wide[['egc', 'egb']].dropna()
    egb_relation = c1570_physics_grid__fit_affine(eg_rows['egc'].to_numpy(float), eg_rows['egb'].to_numpy(float))
    gap_relation = c1570_physics_grid__fit_gap_relation(train_wide)
    epsnc_rows = train_wide[['eps', 'nc']].dropna()
    ionic = epsnc_rows['eps'].to_numpy(float) - np.square(epsnc_rows['nc'].to_numpy(float))
    median_ionic = float(np.median(ionic)) if len(ionic) else 0.2
    median_ionic = max(median_ionic, 0.02)
    pred = test[['id', 'canonical', 'target_type']].copy()
    pred['value'] = base_values
    wide = pred.pivot_table(index='canonical', columns='target_type', values='value', aggfunc='mean')
    row_index = {(str(row['canonical']), str(row['target_type'])): int(idx) for idx, row in test.iterrows()}
    values = base_values.copy()
    for canon_raw, row in wide.iterrows():
        canon = str(canon_raw)
        if {'egc', 'egb'}.issubset(row.index) and pd.notna(row.get('egc')) and pd.notna(row.get('egb')):
            egc_new, egb_new = c1570_physics_grid__project_egc_egb(float(row['egc']), float(row['egb']), egb_relation, egb_pull)
            for target, new_value in (('egc', egc_new), ('egb', egb_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
        if {'ei', 'eea', 'egc'}.issubset(row.index) and all((pd.notna(row.get(t)) for t in ('ei', 'eea', 'egc'))):
            ei_new, eea_new, egc_new = c1570_physics_grid__project_gap(float(row['ei']), float(row['eea']), float(row['egc']), gap_relation, gap_pull)
            for target, new_value in (('ei', ei_new), ('eea', eea_new), ('egc', egc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
        if {'eps', 'nc'}.issubset(row.index) and pd.notna(row.get('eps')) and pd.notna(row.get('nc')):
            eps_new, nc_new = c1570_physics_grid__project_eps_nc(float(row['eps']), float(row['nc']), median_ionic, epsnc_pull)
            for target, new_value in (('eps', eps_new), ('nc', nc_new)):
                idx = row_index.get((canon, target))
                if idx is not None:
                    values[idx] = new_value
    target_type = test['target_type'].to_numpy(object)
    for target in c1570_physics_grid_TARGETS:
        mask = target_type == target
        values[mask] = c1570_physics_grid__clip_target(values[mask], target, train)
    if not np.isfinite(values).all():
        raise RuntimeError('Projection produced non-finite values')
    return pd.DataFrame({'id': ids, 'target': values})

# ===== epsnc_b3.py =====
"""In-memory port of the original round "current EPS/NC B3 consistency overlay".

Faithful port of the original builder script
plus its LOCAL_DIAGNOSTIC_ONLY source build_current_epsnc_b3_consistency_overlay.py.

Steps (verbatim):
1. canonicalize train/test SMILES;
2. build official_wide = train.pivot(mean) and base_wide = base over test pivoted;
3. fit ionic predictor = ExtraTrees(800, min_samples_leaf=2, max_features=0.75,
   random_state=20260808, n_jobs=2) on polar_block over the EPS/NC train pairs,
   predicting ionic = eps - nc^2 (clamped to MIN_IONIC);
4. ionic overlay pass on EPS/NC test rows using partner_value fallback;
5. B3 joint consistency pass: for canonical structures present as both EPS and NC
   test rows, project onto eps = nc^2 + ionic via the 901-point grid and pull the
   two values toward the consistent pair.

No file IO, no hashes, no path guards, no manifest. No extra DataFrames beyond
train/test/base are required.
"""
epsnc_b3_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
epsnc_b3_SEED = 20260808
epsnc_b3_MIN_IONIC = 0.02
epsnc_b3_POLAR_SMARTS = {'CF': '[#6][F]', 'CCl': '[#6][Cl]', 'ester': 'C(=O)O', 'carbonyl': '[CX3]=[OX1]', 'ether': '[OD2]([#6])[#6]', 'OH': '[OX2H]', 'nitrile': 'C#N', 'amide': 'C(=O)N', 'NH': '[NX3;H1,H2]', 'sulfone': 'S(=O)(=O)', 'thioether': '[#16X2]', 'aromatic_N': 'n', 'aromatic_O': 'o', 'aromatic_S': 's', 'imide': 'C(=O)NC(=O)', 'siloxane': '[Si][O]', 'phosphate': 'P=O', 'urethane': 'NC(=O)O'}
epsnc_b3_POLAR_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in epsnc_b3_POLAR_SMARTS.items()}

def epsnc_b3__canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def epsnc_b3__polar_block(canons: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for canon in canons:
        mol = Chem.MolFromSmiles(str(canon))
        if mol is None:
            rows.append([0.0] * (len(epsnc_b3_POLAR_PATTERNS) + 8))
            continue
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in epsnc_b3_POLAR_PATTERNS.values()]
        row += [Descriptors.TPSA(mol) / heavy, Descriptors.NumHDonors(mol) / heavy, Descriptors.NumHAcceptors(mol) / heavy, Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol) / heavy, Crippen.MolMR(mol) / heavy, Crippen.MolLogP(mol) / heavy, rdMolDescriptors.CalcNumAromaticRings(mol) / heavy]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)

def epsnc_b3__fit_ionic_predictor(pair_canons: list[str], ionic: np.ndarray):
    x_train = epsnc_b3__polar_block(pair_canons)
    model = ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2, max_features=0.75, random_state=epsnc_b3_SEED, n_jobs=2)
    model.fit(x_train, np.asarray(ionic, dtype=np.float64))

    def predict(canons: list[str]) -> np.ndarray:
        raw = model.predict(epsnc_b3__polar_block(canons))
        return np.maximum(np.asarray(raw, dtype=np.float64), epsnc_b3_MIN_IONIC)
    return (predict, {'mode': 'extra_trees_raw_b3', 'train_rows': int(len(pair_canons)), 'min_ionic': float(np.min(ionic)), 'median_ionic': float(np.median(ionic)), 'max_ionic': float(np.max(ionic))})

def epsnc_b3__partner_value(canon: str, target: str, official_wide: pd.DataFrame, base_wide: pd.DataFrame) -> tuple[float | None, str]:
    if target in official_wide.columns and canon in official_wide.index:
        value = official_wide.at[canon, target]
        if pd.notna(value):
            return (float(value), 'official_current_train')
    if target in base_wide.columns and canon in base_wide.index:
        value = base_wide.at[canon, target]
        if pd.notna(value):
            return (float(value), 'base_same_canonical_test_prediction')
    return (None, 'missing')

def epsnc_b3__project_pair(eps_ref: float, nc_ref: float, ionic: float, eps_weight: float, nc_weight: float) -> tuple[float, float]:
    grid = np.linspace(1.0, 2.8, 901)
    eps_grid = grid * grid + ionic
    loss = eps_weight * np.square(eps_grid - eps_ref) + nc_weight * np.square(grid - nc_ref)
    idx = int(np.argmin(loss))
    nc = float(grid[idx])
    eps = float(eps_grid[idx])
    return (eps, nc)

def epsnc_b3_overlay(test_df: pd.DataFrame, train_df: pd.DataFrame, base_df: pd.DataFrame, eps_weight: float=0.1, nc_weight: float=0.025, consistency_pull: float=0.3) -> pd.DataFrame:
    """Adjust EPS/NC with the ionic + B3 joint-consistency pull.

    test_df: official test (columns id, smiles, target_type).
    train_df: official train (columns smiles, target, target_type).
    base_df: frozen base predictions (columns id, target).
    Returns a DataFrame with columns [id, target].
    """
    train = train_df.copy()
    test = test_df.copy()
    base = base_df.copy()
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    train['canonical'] = [epsnc_b3__canonical(value) for value in train['smiles']]
    test['canonical'] = [epsnc_b3__canonical(value) for value in test['smiles']]
    ids = test['id'].to_numpy(int)
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base ID order')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base has non-finite predictions')
    base_values = base['target'].to_numpy(float)
    values = base_values.copy()
    official_wide = train.pivot_table(index='canonical', columns='target_type', values='target', aggfunc='mean')
    pair_frame = official_wide[['eps', 'nc']].dropna().copy()
    ionic = pair_frame['eps'].to_numpy(float) - pair_frame['nc'].to_numpy(float) ** 2
    if len(pair_frame) < 50 or np.any(ionic <= 0):
        raise RuntimeError('Insufficient or invalid current EPS/NC pairs')
    ionic_predict, _ionic_report = epsnc_b3__fit_ionic_predictor(pair_frame.index.astype(str).tolist(), ionic)
    test_pred = test[['id', 'canonical', 'target_type']].copy()
    test_pred['base_target'] = base_values
    base_wide = test_pred.pivot_table(index='canonical', columns='target_type', values='base_target', aggfunc='mean')
    eps_train = train.loc[train['target_type'] == 'eps', 'target'].to_numpy(float)
    nc_train = train.loc[train['target_type'] == 'nc', 'target'].to_numpy(float)
    eps_low, eps_high = (float(np.quantile(eps_train, 0.002)), float(np.quantile(eps_train, 0.998)))
    nc_low, nc_high = (float(np.quantile(nc_train, 0.002)), float(np.quantile(nc_train, 0.998)))
    ionic_cache: dict[str, float] = {}
    row_by_canon_target: dict[tuple[str, str], int] = {}
    for row_index, row in test.iterrows():
        target = str(row['target_type'])
        if target in {'eps', 'nc'}:
            row_by_canon_target[str(row['canonical']), target] = int(row_index)
    for row_index, row in test.iterrows():
        target = str(row['target_type'])
        if target not in {'eps', 'nc'}:
            continue
        canon = str(row['canonical'])
        if canon not in ionic_cache:
            ionic_cache[canon] = float(ionic_predict([canon])[0])
        ion = max(float(ionic_cache[canon]), epsnc_b3_MIN_IONIC)
        old = float(values[row_index])
        if target == 'eps':
            partner, _src = epsnc_b3__partner_value(canon, 'nc', official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(partner ** 2 + ion, max(0.0, eps_low - 0.05), eps_high + 0.05))
            values[row_index] = (1.0 - eps_weight) * old + eps_weight * raw
        else:
            partner, _src = epsnc_b3__partner_value(canon, 'eps', official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(np.sqrt(max(partner - ion, 1.0)), max(1.0, nc_low - 0.05), min(2.8, nc_high + 0.05)))
            values[row_index] = (1.0 - nc_weight) * old + nc_weight * raw
    for canon in sorted({canon for canon, target in row_by_canon_target if target == 'eps'}):
        eps_idx = row_by_canon_target.get((canon, 'eps'))
        nc_idx = row_by_canon_target.get((canon, 'nc'))
        if eps_idx is None or nc_idx is None:
            continue
        ion = max(float(ionic_cache.get(canon, ionic_predict([canon])[0])), epsnc_b3_MIN_IONIC)
        eps_cons, nc_cons = epsnc_b3__project_pair(float(values[eps_idx]), float(values[nc_idx]), ion, max(eps_weight, 1e-06), max(nc_weight, 1e-06))
        values[eps_idx] = (1.0 - consistency_pull) * values[eps_idx] + consistency_pull * eps_cons
        values[nc_idx] = (1.0 - consistency_pull) * values[nc_idx] + consistency_pull * nc_cons
    if not np.isfinite(values).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': values})

# ===== epsnc_ionic.py =====
"""In-memory port of the original builder script.

C1004/C990 clean EPS/NC ionic overlay.  Takes official train/test and a frozen
base candidate as pandas DataFrames and returns the 4,940-row (id, target)
submission with the current-label EPS/NC ionic consistency overlay applied.

Deliberate non-behavioral deviations from the original round source:
  * No path guards, no digest/digest module/manifest, no argparse/argv, no CSV I/O --
    inputs arrive as DataFrames and the result is returned as a DataFrame.
  * Runtime ID/base-schema validation is retained (raises on mismatch) but
    reports via RuntimeError rather than a manifest JSON.
  * Manifest-only logging (applied/support/examples/ionic_report) is dropped.

The numeric computation is identical to the source:
  - canonicalization: RDKit canonical isomeric SMILES (after "[*]" -> "*");
  - ionic coordinate = eps - nc**2 on current-train paired structures;
  - ionic predictor: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2,
    max_features=0.75, random_state=20260808, n_jobs=2) on the 26-feature polar
    block (18 SMARTS pattern densities + 8 descriptors); "extra_trees_raw" mode
    predicts directly and clips to MIN_IONIC=0.02 (no log transform);
  - eps: raw = clip(nc_partner**2 + ionic, max(0.0, eps_q002-0.05), eps_q998+0.05)
  - nc:  raw = clip(sqrt(max(eps_partner - ionic, 1.0)), max(1.0, nc_q002-0.05),
            min(2.8, nc_q998+0.05))
  - blend: values = (1 - w)*base + w*raw with eps_weight / nc_weight.
"""
RDLogger.DisableLog('rdApp.*')
epsnc_ionic_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
epsnc_ionic_SEED = 20260808
epsnc_ionic_MIN_IONIC = 0.02
epsnc_ionic_POLAR_SMARTS = {'CF': '[#6][F]', 'CCl': '[#6][Cl]', 'ester': 'C(=O)O', 'carbonyl': '[CX3]=[OX1]', 'ether': '[OD2]([#6])[#6]', 'OH': '[OX2H]', 'nitrile': 'C#N', 'amide': 'C(=O)N', 'NH': '[NX3;H1,H2]', 'sulfone': 'S(=O)(=O)', 'thioether': '[#16X2]', 'aromatic_N': 'n', 'aromatic_O': 'o', 'aromatic_S': 's', 'imide': 'C(=O)NC(=O)', 'siloxane': '[Si][O]', 'phosphate': 'P=O', 'urethane': 'NC(=O)O'}
epsnc_ionic_POLAR_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in epsnc_ionic_POLAR_SMARTS.items()}

def epsnc_ionic__canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def epsnc_ionic__polar_block(canons: list) -> np.ndarray:
    rows: list = []
    for canon in canons:
        mol = Chem.MolFromSmiles(str(canon))
        if mol is None:
            rows.append([0.0] * (len(epsnc_ionic_POLAR_PATTERNS) + 8))
            continue
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in epsnc_ionic_POLAR_PATTERNS.values()]
        row += [Descriptors.TPSA(mol) / heavy, Descriptors.NumHDonors(mol) / heavy, Descriptors.NumHAcceptors(mol) / heavy, Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol) / heavy, Crippen.MolMR(mol) / heavy, Crippen.MolLogP(mol) / heavy, rdMolDescriptors.CalcNumAromaticRings(mol) / heavy]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)

def epsnc_ionic__fit_ionic_predictor(pair_canons: list, ionic: np.ndarray, mode: str):
    if mode == 'median':
        median = float(np.median(ionic))

        def predict(canons: list) -> np.ndarray:
            return np.full(len(canons), median, dtype=np.float64)
        return (predict, {'mode': mode, 'median': median})
    if mode not in {'extra_trees_raw', 'extra_trees_log'}:
        raise RuntimeError(f'Unknown ionic mode: {mode}')
    x_train = epsnc_ionic__polar_block(pair_canons)
    y = np.asarray(ionic, dtype=np.float64)
    if mode == 'extra_trees_log':
        y = np.log(np.maximum(y, epsnc_ionic_MIN_IONIC))
    model = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.75, random_state=epsnc_ionic_SEED, n_jobs=2)
    model.fit(x_train, y)

    def predict(canons: list) -> np.ndarray:
        raw = model.predict(epsnc_ionic__polar_block(canons))
        if mode == 'extra_trees_log':
            raw = np.exp(np.clip(raw, -8, 4))
        return np.maximum(np.asarray(raw, dtype=np.float64), epsnc_ionic_MIN_IONIC)
    return (predict, {'mode': mode, 'train_rows': int(len(pair_canons)), 'min_ionic': float(np.min(ionic)), 'median_ionic': float(np.median(ionic)), 'max_ionic': float(np.max(ionic))})

def epsnc_ionic__partner_value(canon: str, target: str, official_wide: pd.DataFrame, base_wide: pd.DataFrame):
    if target in official_wide.columns and canon in official_wide.index:
        value = official_wide.at[canon, target]
        if pd.notna(value):
            return (float(value), 'official_current_train')
    if target in base_wide.columns and canon in base_wide.index:
        value = base_wide.at[canon, target]
        if pd.notna(value):
            return (float(value), 'base_same_canonical_test_prediction')
    return (None, 'missing')

def epsnc_ionic_overlay(train_df: pd.DataFrame, test_df: pd.DataFrame, base_df: pd.DataFrame, eps_weight: float=0.05, nc_weight: float=0.0, mode: str='extra_trees_raw') -> pd.DataFrame:
    """Apply the current-label EPS/NC ionic consistency overlay to base_df.

    Parameters
    ----------
    train_df : official train.csv (smiles, target, target_type)
    test_df  : official test.csv (id, smiles, target_type)
    base_df  : candidate (id, target), ids 1..4940 in order
    eps_weight, nc_weight : overlay blend weights in [0, 1]
    mode : "median" | "extra_trees_raw" | "extra_trees_log"

    Returns
    -------
    pd.DataFrame with columns ["id", "target"], 4,940 rows, ids 1..4940.
    """
    train = train_df.copy()
    test = test_df.reset_index(drop=True).copy()
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    train['canonical'] = [epsnc_ionic__canonicalize(value) for value in train['smiles']]
    test['canonical'] = [epsnc_ionic__canonicalize(value) for value in test['smiles']]
    ids = test['id'].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    base = base_df.copy()
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Base ID order mismatch')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base contains non-finite values')
    eps_w = float(eps_weight)
    nc_w = float(nc_weight)
    if not 0.0 <= eps_w <= 1.0 or not 0.0 <= nc_w <= 1.0:
        raise RuntimeError('weights outside [0, 1]')
    official_wide = train.pivot_table(index='canonical', columns='target_type', values='target', aggfunc='mean')
    pair_frame = official_wide[['eps', 'nc']].dropna().copy()
    if len(pair_frame) < 50:
        raise RuntimeError('Insufficient current EPS/NC pairs')
    ionic_coord = pair_frame['eps'].to_numpy(float) - pair_frame['nc'].to_numpy(float) ** 2
    if np.any(ionic_coord <= 0):
        raise RuntimeError('Non-positive ionic coordinate in official current pairs')
    ionic_predict, _ionic_report = epsnc_ionic__fit_ionic_predictor(pair_frame.index.astype(str).tolist(), ionic_coord, str(mode))
    test_pred = test[['id', 'canonical', 'target_type']].copy()
    test_pred['base_target'] = base['target'].to_numpy(float)
    base_wide = test_pred.pivot_table(index='canonical', columns='target_type', values='base_target', aggfunc='mean')
    values = base['target'].to_numpy(float).copy()
    eps_train = train.loc[train['target_type'] == 'eps', 'target'].to_numpy(float)
    nc_train = train.loc[train['target_type'] == 'nc', 'target'].to_numpy(float)
    eps_low, eps_high = (float(np.quantile(eps_train, 0.002)), float(np.quantile(eps_train, 0.998)))
    nc_low, nc_high = (float(np.quantile(nc_train, 0.002)), float(np.quantile(nc_train, 0.998)))
    ionic_cache: dict = {}
    for row_index, row in test.iterrows():
        target = str(row['target_type'])
        if target not in {'eps', 'nc'}:
            continue
        canon = str(row['canonical'])
        if canon not in ionic_cache:
            ionic_cache[canon] = float(ionic_predict([canon])[0])
        ion = max(float(ionic_cache[canon]), epsnc_ionic_MIN_IONIC)
        old = float(values[row_index])
        if target == 'eps':
            partner, _src = epsnc_ionic__partner_value(canon, 'nc', official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(partner ** 2 + ion, max(0.0, eps_low - 0.05), eps_high + 0.05))
            values[row_index] = (1.0 - eps_w) * old + eps_w * raw
        else:
            partner, _src = epsnc_ionic__partner_value(canon, 'eps', official_wide, base_wide)
            if partner is None:
                continue
            raw = float(np.clip(np.sqrt(max(partner - ion, 1.0)), max(1.0, nc_low - 0.05), min(2.8, nc_high + 0.05)))
            values[row_index] = (1.0 - nc_w) * old + nc_w * raw
    if not np.isfinite(values).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': values})

# ===== identity_overlay.py =====
"""In-memory port of the original round "current identity overlay".

Faithful port of the original builder script plus its
LOCAL_DIAGNOSTIC_ONLY source build_current_identity_overlay.py.

Identities applied (only when the required partner is observed in current train
for the same canonical structure; the other partner may come from base):

    eea_hat = ei - egc
    ei_hat  = eea + egc
    egb_hat = 1.1178 * egc - 0.9221

Blend per target: value = (1 - weight) * old + weight * raw.

No file IO, no hashes, no path guards, no manifest. No extra DataFrames beyond
train/test/base are required.
"""
identity_overlay_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

def identity_overlay__canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def identity_overlay__partner_value(canon: str, target: str, official_wide: pd.DataFrame, base_wide: pd.DataFrame) -> tuple[float | None, str]:
    if target in official_wide.columns and canon in official_wide.index:
        value = official_wide.at[canon, target]
        if pd.notna(value):
            return (float(value), 'official_current_train')
    if target in base_wide.columns and canon in base_wide.index:
        value = base_wide.at[canon, target]
        if pd.notna(value):
            return (float(value), 'base_same_canonical_test_prediction')
    return (None, 'missing')

def identity_overlay_identity_overlay(test_df: pd.DataFrame, train_df: pd.DataFrame, base_df: pd.DataFrame, eea_weight: float=0.01, ei_weight: float=0.0, egb_weight: float=0.0) -> pd.DataFrame:
    """Blend identity targets (ei, eea, egb) onto a frozen base.

    test_df: official test (columns id, smiles, target_type).
    train_df: official train (columns smiles, target, target_type).
    base_df: frozen base predictions (columns id, target).
    Returns a DataFrame with columns [id, target].
    """
    train = train_df.copy()
    test = test_df.copy()
    base = base_df.copy()
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    train['canonical'] = [identity_overlay__canonical(value) for value in train['smiles']]
    test['canonical'] = [identity_overlay__canonical(value) for value in test['smiles']]
    ids = test['id'].to_numpy(int)
    if list(base.columns) != ['id', 'target'] or len(base) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base['id'].duplicated().any() or not np.array_equal(base['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base ID order')
    if not np.isfinite(base['target'].to_numpy(float)).all():
        raise RuntimeError('Base has non-finite predictions')
    official_wide = train.pivot_table(index='canonical', columns='target_type', values='target', aggfunc='mean')
    test_pred = test[['id', 'canonical', 'target_type']].copy()
    test_pred['base_target'] = base['target'].to_numpy(float)
    base_wide = test_pred.pivot_table(index='canonical', columns='target_type', values='base_target', aggfunc='mean')
    values = base['target'].to_numpy(float).copy()
    for row_index, row in test.iterrows():
        target = str(row['target_type'])
        canon = str(row['canonical'])
        old = float(values[row_index])
        if target == 'eea':
            ei, ei_src = identity_overlay__partner_value(canon, 'ei', official_wide, base_wide)
            egc, _egc_src = identity_overlay__partner_value(canon, 'egc', official_wide, base_wide)
            if ei is not None and egc is not None and (ei_src == 'official_current_train'):
                raw = ei - egc
                values[row_index] = (1.0 - eea_weight) * old + eea_weight * raw
        elif target == 'ei':
            eea, eea_src = identity_overlay__partner_value(canon, 'eea', official_wide, base_wide)
            egc, _egc_src = identity_overlay__partner_value(canon, 'egc', official_wide, base_wide)
            if eea is not None and egc is not None and (eea_src == 'official_current_train'):
                raw = eea + egc
                values[row_index] = (1.0 - ei_weight) * old + ei_weight * raw
        elif target == 'egb':
            egc, egc_src = identity_overlay__partner_value(canon, 'egc', official_wide, base_wide)
            if egc is not None and egc_src == 'official_current_train':
                raw = 1.1178 * egc - 0.9221
                values[row_index] = (1.0 - egb_weight) * old + egb_weight * raw
    if not np.isfinite(values).all():
        raise RuntimeError('Output contains non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': values})

# ===== physics_overlay.py =====
"""In-memory port of C296/C297 safe official-partner identity/physics overlay.

Faithful in-memory conversion of
the original builder script, in the current-only
(no-archive) lane: the label pool is the official current train only. The the original round
archive/train.csv branch is intentionally omitted (HARD RULES: only official
train/test/PI1M). No intermediate CSVs, no R2 paths, no hashes, no manifests, no
path guards.

train_df: columns [smiles, target, target_type]
test_df : columns [id, smiles, target_type]
base_df : columns [id, target] -- target aligned to test_df row order (ids 1..4940)
"""
physics_overlay_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
physics_overlay_PARTNERS = {'ei': ('eea', 'egc'), 'eea': ('ei', 'egc'), 'egb': ('egc',), 'eps': ('nc',), 'nc': ('eps',)}
physics_overlay_DEFAULT_TARGETS = ('ei', 'eea', 'egb', 'eps')

def physics_overlay__canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def physics_overlay__make_wide(label_rows: pd.DataFrame) -> pd.DataFrame:
    return label_rows.pivot_table(index='canonical', columns='target_type', values='target', aggfunc='median')

def physics_overlay__formula_value(target: str, values: dict[str, float]) -> float:
    if target == 'ei':
        return float(values['eea'] + values['egc'])
    if target == 'eea':
        return float(values['ei'] - values['egc'])
    if target == 'egb':
        return float(values['egc'])
    if target == 'eps':
        return float(values['nc'] ** 2)
    if target == 'nc':
        return float(math.sqrt(max(values['eps'], 1.0)))
    raise KeyError(target)

def physics_overlay__model_factory(target: str) -> Any:
    if target in {'ei', 'eea', 'egb', 'eps', 'nc'}:
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000))
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

def physics_overlay__fit_formula_model(wide: pd.DataFrame, target: str) -> tuple[Any, tuple[float, float]]:
    partners = physics_overlay_PARTNERS[target]
    required = (target, *partners)
    if not all((column in wide.columns for column in required)):
        raise RuntimeError(f'Missing required columns for {target}: {required}')
    frame = wide[list(required)].dropna().copy()
    if len(frame) < 30:
        raise RuntimeError(f'Insufficient formula support for {target}: {len(frame)}')
    y = frame[target].to_numpy(float)
    x_raw = np.asarray([physics_overlay__formula_value(target, {partner: float(row[partner]) for partner in partners}) for _, row in frame.iterrows()], dtype=np.float64).reshape(-1, 1)
    groups = frame.index.astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x_raw, y, groups=groups)):
        model = physics_overlay__model_factory(target)
        model.fit(x_raw[tr], y[tr])
        oof[va] = np.asarray(model.predict(x_raw[va]), dtype=np.float64)
    if not np.isfinite(oof).all():
        raise RuntimeError(f'Non-finite OOF formula predictions for {target}')
    full_model = physics_overlay__model_factory(target)
    full_model.fit(x_raw, y)
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    return (full_model, (float(q01 - 2.0 * margin), float(q99 + 2.0 * margin)))

def physics_overlay__parse_targets(targets) -> tuple[str, ...]:
    if targets is None or (isinstance(targets, str) and targets.strip().lower() in {'', 'default'}):
        return physics_overlay_DEFAULT_TARGETS
    if isinstance(targets, str):
        raw = [item.strip().lower() for item in targets.split(',') if item.strip()]
    else:
        raw = [str(item).strip().lower() for item in targets]
    invalid = [target for target in raw if target not in physics_overlay_PARTNERS]
    if invalid:
        raise RuntimeError(f'Invalid overlay targets: {invalid}')
    return tuple(raw)

def physics_overlay__target_weights(target: str, observed_scale: float, cotest_scale: float) -> tuple[float, float]:
    base = {'ei': (0.85, 0.25), 'eea': (0.85, 0.25), 'egb': (0.55, 0.2), 'eps': (0.45, 0.2), 'nc': (0.2, 0.1)}[target]
    return (float(np.clip(base[0] * observed_scale, 0.0, 1.0)), float(np.clip(base[1] * cotest_scale, 0.0, 1.0)))

def physics_overlay_safe_identity_physics_overlay(train_df: pd.DataFrame, test_df: pd.DataFrame, base_df: pd.DataFrame, observed_weight_scale: float=1.0, cotest_weight_scale: float=1.0, targets=None, disable_cotest: bool=False) -> pd.DataFrame:
    train = train_df.copy()
    test = test_df.copy()
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    train['canonical'] = [physics_overlay__canonicalize(value) for value in train['smiles']]
    test['canonical'] = [physics_overlay__canonicalize(value) for value in test['smiles']]
    ids = test['id'].to_numpy(int)
    if 'id' not in base_df.columns or 'target' not in base_df.columns:
        raise RuntimeError('Invalid base columns')
    if len(base_df) != len(ids):
        raise RuntimeError('Invalid base row count')
    if base_df['id'].duplicated().any() or not np.array_equal(base_df['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base ID order')
    if not np.isfinite(base_df['target'].to_numpy(float)).all():
        raise RuntimeError('Invalid non-finite base predictions')
    active_targets = physics_overlay__parse_targets(targets)
    label_rows = train[['canonical', 'target_type', 'target']].copy()
    wide = physics_overlay__make_wide(label_rows)
    test['base_prediction'] = base_df['target'].to_numpy(float)
    base_pivot = test.pivot_table(index='canonical', columns='target_type', values='base_prediction', aggfunc='mean')
    models: dict[str, Any] = {}
    clips: dict[str, tuple[float, float]] = {}
    for target in active_targets:
        model, clip = physics_overlay__fit_formula_model(wide, target)
        models[target] = model
        clips[target] = clip
    result = test['base_prediction'].to_numpy(float).copy()
    for row_pos, row in test.iterrows():
        target = str(row['target_type']).lower()
        if target not in active_targets:
            continue
        canonical = str(row['canonical'])
        values: dict[str, float] = {}
        source_types: list[str] = []
        supported = True
        for partner in physics_overlay_PARTNERS[target]:
            observed_value = np.nan
            if canonical in wide.index and partner in wide.columns:
                observed_value = wide.loc[canonical].get(partner, np.nan)
            if pd.notna(observed_value):
                values[partner] = float(observed_value)
                source_types.append('official_observed')
                continue
            cotest_value = np.nan
            if not disable_cotest and canonical in base_pivot.index and (partner in base_pivot.columns):
                cotest_value = base_pivot.loc[canonical].get(partner, np.nan)
            if pd.notna(cotest_value):
                values[partner] = float(cotest_value)
                source_types.append('cotest_base_prediction')
                continue
            supported = False
            break
        if not supported:
            continue
        x_value = physics_overlay__formula_value(target, values)
        pred = float(models[target].predict(np.asarray([[x_value]], dtype=np.float64))[0])
        low, high = clips[target]
        pred = float(np.clip(pred, low, high))
        observed_weight, cotest_weight = physics_overlay__target_weights(target, observed_weight_scale, cotest_weight_scale)
        if all((item == 'official_observed' for item in source_types)):
            weight = observed_weight
        else:
            weight = cotest_weight
        result[int(row_pos)] = (1.0 - weight) * result[int(row_pos)] + weight * pred
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite output')
    return pd.DataFrame({'id': ids, 'target': result})

# ===== xprop_overlay.py =====
"""In-memory port of C290/C291 current-only imputed cross-property overlay.

Faithful in-memory conversion of
the original builder script.

Reads NOTHING from disk: train/test/base arrive as DataFrames and every feature
(RDKit descriptors, physical features, Morgan count fingerprints, MACCS keys) is
computed from SMILES in memory. All model fitting (seeds, params, target order)
is kept identical to the original. Hash/manifest/path-guard code is removed.

train_df: columns [smiles, target, target_type]
test_df : columns [id, smiles, target_type]
base_df : columns [id, target] -- target aligned to test_df row order (ids 1..4940)
"""
xprop_overlay_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

def xprop_overlay__canonicalize(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    if molecule is None:
        raise ValueError('RDKit could not parse an official SMILES value')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

def xprop_overlay__build_molecules(keys: list[str]) -> list[Any]:
    molecules = [Chem.MolFromSmiles(value) for value in keys]
    if any((molecule is None for molecule in molecules)):
        raise RuntimeError('Canonical official structure failed RDKit parsing')
    return molecules

def xprop_overlay__descriptor_matrix(molecules: list[Any]) -> np.ndarray:
    items = list(Descriptors._descList)
    matrix = np.full((len(molecules), len(items)), np.nan, dtype=np.float64)
    for row, molecule in enumerate(molecules):
        for column, (_, function) in enumerate(items):
            try:
                value = float(function(molecule))
            except Exception:
                value = np.nan
            matrix[row, column] = value if np.isfinite(value) else np.nan
    return matrix

def xprop_overlay__physical_matrix(molecules: list[Any], smiles: list[str]) -> np.ndarray:
    names = ['smiles_length', 'atom_count', 'heavy_atom_count', 'dummy_atom_count', 'ring_count', 'aromatic_atom_count', 'hetero_atom_count', 'halogen_count', 'rotatable_bonds_approx', 'double_bond_count', 'triple_bond_count', 'branch_count', 'n_count', 'o_count', 's_count', 'si_count']
    matrix = np.zeros((len(molecules), len(names)), dtype=np.float64)
    for row, (molecule, value) in enumerate(zip(molecules, smiles, strict=True)):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        matrix[row] = [len(value), molecule.GetNumAtoms(), molecule.GetNumHeavyAtoms(), sum((atom.GetAtomicNum() == 0 for atom in atoms)), molecule.GetRingInfo().NumRings(), sum((atom.GetIsAromatic() for atom in atoms)), sum((atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)), sum((atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)), sum((bond.GetBondTypeAsDouble() == 1.0 and (not bond.IsInRing()) for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 2.0 for bond in bonds)), sum((bond.GetBondTypeAsDouble() == 3.0 for bond in bonds)), value.count('('), sum((atom.GetAtomicNum() == 7 for atom in atoms)), sum((atom.GetAtomicNum() == 8 for atom in atoms)), sum((atom.GetAtomicNum() == 16 for atom in atoms)), sum((atom.GetAtomicNum() == 14 for atom in atoms))]
    return matrix

def xprop_overlay__morgan_count_matrix(molecules: list[Any], radius: int, bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    matrix = np.zeros((len(molecules), bits), dtype=np.float32)
    for row, molecule in enumerate(molecules):
        fingerprint = generator.GetCountFingerprint(molecule)
        for column, count in fingerprint.GetNonzeroElements().items():
            matrix[row, int(column)] = np.float32(np.log1p(float(count)))
    return matrix

def xprop_overlay__maccs_matrix(molecules: list[Any]) -> np.ndarray:
    rows = np.zeros((len(molecules), 167), dtype=np.float32)
    for i, mol in enumerate(molecules):
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows[i] = np.asarray([int(fp.GetBit(j)) for j in range(167)], dtype=np.float32)
    return rows

def xprop_overlay__sanitize(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(out) | (np.abs(out) > 1000000000000.0)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out

def xprop_overlay__grouped_oof(factory: Callable[..., Any], x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError('not enough groups')
    out = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x[tr], y[tr])
        out[va] = np.asarray(model.predict(x[va]), dtype=np.float64)
    if not np.isfinite(out).all():
        raise RuntimeError('non-finite oof')
    return out

def xprop_overlay__imputer_factory(target: str, fast_linear: bool) -> Callable[..., Any]:
    if fast_linear:
        return lambda seed: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0, solver='lsqr', max_iter=5000, tol=0.0001))
    large = target in {'tg', 'egc'}

    def factory(seed: int):
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), ExtraTreesRegressor(n_estimators=80 if large else 120, min_samples_leaf=4 if large else 2, max_features=0.55 if large else 0.75, random_state=seed, n_jobs=4))
    return factory

def xprop_overlay__relation_factories(target: str, fast_linear: bool) -> dict[str, Callable[..., Any]]:
    if fast_linear:
        return {'ridge10': lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), 'ridge100': lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=100.0)), 'huber': lambda seed: make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000))}
    small = target not in {'tg', 'egc'}
    return {'ridge10': lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), 'ridge100': lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=100.0)), 'huber': lambda seed: make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000)), 'extra_trees': lambda seed: ExtraTreesRegressor(n_estimators=120 if small else 80, min_samples_leaf=3 if small else 10, max_features=1.0, random_state=seed, n_jobs=4), 'hist_gbdt': lambda seed: HistGradientBoostingRegressor(learning_rate=0.035, max_iter=220 if small else 160, l2_regularization=0.1, max_leaf_nodes=15 if small else 31, min_samples_leaf=8 if small else 20, random_state=seed)}

def xprop_overlay__clip_like(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1e-08)
    return np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)

def xprop_overlay_imputed_cross_property_overlay(train_df: pd.DataFrame, test_df: pd.DataFrame, base_df: pd.DataFrame, overlay_weight: float=0.25, gate_delta: float=0.005, morgan_bits: int=128, seed: int=20260808, fast_linear: bool=True) -> pd.DataFrame:
    train = train_df.copy()
    test = test_df.copy()
    train['target_type'] = train['target_type'].astype(str).str.lower()
    test['target_type'] = test['target_type'].astype(str).str.lower()
    train['canonical'] = train['smiles'].map(xprop_overlay__canonicalize)
    test['canonical'] = test['smiles'].map(xprop_overlay__canonicalize)
    ids = test['id'].to_numpy(int)
    if 'id' not in base_df.columns or 'target' not in base_df.columns:
        raise RuntimeError('Invalid base schema')
    if base_df['id'].duplicated().any() or not np.array_equal(base_df['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base IDs')
    if not np.isfinite(base_df['target'].to_numpy(float)).all():
        raise RuntimeError('Invalid base values')
    result = base_df['target'].to_numpy(float).copy()
    keys = sorted(set(train['canonical']) | set(test['canonical']))
    molecules = xprop_overlay__build_molecules(keys)
    key_to_index = {key: i for i, key in enumerate(keys)}
    descriptor = xprop_overlay__descriptor_matrix(molecules)
    physical = xprop_overlay__physical_matrix(molecules, keys)
    morgan2 = xprop_overlay__morgan_count_matrix(molecules, radius=2, bits=int(morgan_bits))
    morgan3 = xprop_overlay__morgan_count_matrix(molecules, radius=3, bits=int(morgan_bits))
    maccs = xprop_overlay__maccs_matrix(molecules)
    x_all = xprop_overlay__sanitize(np.hstack([descriptor, physical, morgan2, morgan3, maccs]))
    imputed_full = np.zeros((len(keys), len(xprop_overlay_TARGETS)), dtype=np.float64)
    imputed_train_like = np.zeros((len(keys), len(xprop_overlay_TARGETS)), dtype=np.float64)
    direct_oof_r2: dict[str, float] = {}
    train_by_target: dict[str, pd.DataFrame] = {}
    for ti, target in enumerate(xprop_overlay_TARGETS):
        rows = train[train['target_type'].eq(target)].reset_index(drop=True)
        train_by_target[target] = rows
        idx = np.asarray([key_to_index[c] for c in rows['canonical']], dtype=int)
        y = rows['target'].to_numpy(float)
        groups = rows['canonical'].astype(str).to_numpy(object)
        factory = xprop_overlay__imputer_factory(target, fast_linear)
        oof = xprop_overlay__clip_like(y, xprop_overlay__grouped_oof(factory, x_all[idx], y, groups, int(seed) + ti * 17))
        direct_oof_r2[target] = float(r2_score(y, oof))
        model = factory(int(seed) + ti * 17 + 999)
        model.fit(x_all[idx], y)
        full = xprop_overlay__clip_like(y, model.predict(x_all))
        imputed_full[:, ti] = full
        imputed_train_like[:, ti] = full
        oof_by_can = rows.assign(oof=oof).groupby('canonical')['oof'].mean().to_dict()
        for can, value in oof_by_can.items():
            imputed_train_like[key_to_index[can], ti] = float(value)
    for ti, target in enumerate(xprop_overlay_TARGETS):
        rows = train_by_target[target]
        idx = np.asarray([key_to_index[c] for c in rows['canonical']], dtype=int)
        y = rows['target'].to_numpy(float)
        groups = rows['canonical'].astype(str).to_numpy(object)
        feature_cols = [j for j in range(len(xprop_overlay_TARGETS)) if j != ti]
        rel_x = imputed_train_like[idx][:, feature_cols]
        rel_test_keys = np.asarray([key_to_index[c] for c in test['canonical']], dtype=int)
        rel_x_test_all = imputed_full[rel_test_keys][:, feature_cols]
        best = None
        for name, factory in xprop_overlay__relation_factories(target, fast_linear).items():
            try:
                pred = xprop_overlay__clip_like(y, xprop_overlay__grouped_oof(factory, rel_x, y, groups, int(seed) + 1000 + ti * 29))
                score = float(r2_score(y, pred))
                if best is None or score > best[0]:
                    best = (score, name)
            except Exception:
                pass
        if best is None:
            continue
        direct_score = float(direct_oof_r2[target])
        selected = bool(best[0] >= direct_score + float(gate_delta))
        if not selected:
            continue
        model = xprop_overlay__relation_factories(target, fast_linear)[best[1]](int(seed) + 5000 + ti)
        model.fit(rel_x, y)
        rel_pred = xprop_overlay__clip_like(y, model.predict(rel_x_test_all))
        mask = test['target_type'].eq(target).to_numpy()
        result[mask] = (1.0 - float(overlay_weight)) * result[mask] + float(overlay_weight) * rel_pred[mask]
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite output')
    return pd.DataFrame({'id': ids, 'target': result})

# ===== f03_clean.py =====
"""F03 - archive-free Polymer Genome-style candidate (clean, current-only).

Faithful in-memory port of F03_make_clean_candidate.py + F03_polymer_genome.py
+ claude_r2_01_build_pgfp.py. Reads only the official train.csv / test.csv.
The feature vocabulary is fit from current train structures only; test SMILES
are used only as unlabeled covariates.
"""
f03_clean_SEED = fc.SEED
f03_clean_ALPHAS = (10.0, 30.0, 100.0, 300.0)
f03_clean_MORPH_N = 12

def f03_clean_atom_type(a):
    if a.GetAtomicNum() == 0:
        return 'X'
    return f'{a.GetSymbol()}{a.GetDegree()}'

def f03_clean_keys_for(c):
    m = Chem.MolFromSmiles(c)
    if m is None:
        return (Counter(), Counter(), [])
    at = {a.GetIdx(): f03_clean_atom_type(a) for a in m.GetAtoms()}
    cnt = Counter()
    for a in m.GetAtoms():
        cnt['S|' + at[a.GetIdx()]] += 1
    for b in m.GetBonds():
        i, j = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        p = sorted([at[i], at[j]])
        cnt['P|' + '-'.join(p)] += 1
    for a in m.GetAtoms():
        j = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        for u in range(len(nb)):
            for v in range(u + 1, len(nb)):
                i, k = (nb[u], nb[v])
                ends = sorted([at[i], at[k]])
                cnt['T|' + ends[0] + '-' + at[j] + '-' + ends[1]] += 1
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    bcnt = Counter()
    path = []
    if len(stars) == 2:
        try:
            path = list(Chem.GetShortestPath(m, stars[0], stars[1]))
        except Exception:
            path = []
        for n_ in range(1, len(path) - 1):
            i, j, k = (path[n_ - 1], path[n_], path[n_ + 1])
            ends = sorted([at[i], at[k]])
            bcnt['B|' + ends[0] + '-' + at[j] + '-' + ends[1]] += 1
    return (cnt, bcnt, path)

def f03_clean_morphological(c):
    m = Chem.MolFromSmiles(c)
    if m is None:
        return [np.nan] * f03_clean_MORPH_N
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    n = m.GetNumAtoms()
    try:
        D = Chem.GetDistanceMatrix(m)
    except Exception:
        return [np.nan] * f03_clean_MORPH_N
    dd = []
    for a_ in range(len(rings)):
        for b_ in range(a_ + 1, len(rings)):
            dd.append(min((D[i, j] for i in rings[a_] for j in rings[b_])))
    ring_atoms = set().union(*rings) if rings else set()
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    side, path = (set(), [])
    if len(stars) == 2:
        try:
            path = list(Chem.GetShortestPath(m, stars[0], stars[1]))
        except Exception:
            path = []
        side = set(range(n)) - set(path)
    sizes = []
    if path:
        for i in set(path) - set(stars):
            for nb in m.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() in side:
                    st, comp = ([nb.GetIdx()], set())
                    while st:
                        u = st.pop()
                        if u in comp or u in path:
                            continue
                        comp.add(u)
                        st.extend((x.GetIdx() for x in m.GetAtomWithIdx(u).GetNeighbors()))
                    sizes.append(len(comp))
    sz = sizes or [0]
    return [len(rings), float(np.mean(dd)) if dd else -1.0, float(np.min(dd)) if dd else -1.0, float(np.max(dd)) if dd else -1.0, len(ring_atoms) / max(1, n), len(side) / max(1, n), float(np.max(sz)), float(np.mean(sz)), len(sizes), float(np.max(sz)) / max(1, n), len(path) / max(1, n), float(np.sum([1 for i in ring_atoms if i in set(path)])) / max(1, len(path) or 1)]

def f03_clean_qspr_block(cans):
    funcs = list(Descriptors.descList)
    rows = []
    for can in cans:
        mol = Chem.MolFromSmiles(can)
        vals = []
        for _, fn in funcs:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(np.nan)
        nh = max(mol.GetNumHeavyAtoms(), 1)
        vals.extend([Descriptors.TPSA(mol), Descriptors.TPSA(mol) / nh, rdMolDescriptors.CalcNumAromaticRings(mol) / nh, Descriptors.NumRotatableBonds(mol) / nh, Descriptors.FractionCSP3(mol)])
        rows.append(vals)
    x = np.asarray(rows, dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    return x

def f03_clean_hierarchical_features(train_cans, all_cans):
    train_set = set(train_cans)
    records = {c: f03_clean_keys_for(c) for c in all_cans}
    doc = Counter()
    for c in train_set:
        overall, backbone, _ = records[c]
        doc.update(set(overall) | set(backbone))
    vocab = sorted((k for k, n in doc.items() if n >= 5))
    vi = {k: i for i, k in enumerate(vocab)}
    counts = np.zeros((len(all_cans), len(vocab)), dtype=np.float32)
    morphology = []
    for row, c in enumerate(all_cans):
        overall, backbone, _ = records[c]
        for token, value in overall.items():
            if token in vi:
                counts[row, vi[token]] += value
        for token, value in backbone.items():
            if token in vi:
                counts[row, vi[token]] += value
        morphology.append(f03_clean_morphological(c))
    norm = counts / np.maximum(1.0, counts.sum(axis=1, keepdims=True))
    qspr = f03_clean_qspr_block(all_cans)
    x = np.hstack([np.log1p(counts), norm, np.asarray(morphology), qspr])
    return (x, {'vocabulary_size': len(vocab), 'feature_shape': list(x.shape), 'qspr_features': qspr.shape[1]})

def f03_clean_build_f03(data_dir):
    """Materialize the clean, current-only F03 candidate as an id,target frame."""
    data = fc.load_data(data_dir, include_archive=False)
    train_cans = data.train['can'].tolist()
    test_cans = data.test['can'].tolist()
    all_cans = list(dict.fromkeys(train_cans + test_cans))
    x, feature_report = f03_clean_hierarchical_features(list(dict.fromkeys(train_cans)), all_cans)
    index = {c: i for i, c in enumerate(all_cans)}
    predictions = []
    for target in fc.TARGETS:
        sub = data.train[data.train['target_type'].eq(target)]
        cans = sub['can'].tolist()
        y = sub['target'].to_numpy(float)
        train_idx = np.asarray([index[c] for c in cans])
        test_sub = data.test[data.test['target_type'].eq(target)]
        target_test_cans = test_sub['can'].tolist()
        test_idx = np.asarray([index[c] for c in target_test_cans])
        support = np.isfinite(x[train_idx]).sum(axis=0)
        spread = np.nanstd(x[train_idx], axis=0)
        keep = (support > 0) & (spread > 1e-12)
        model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=f03_clean_ALPHAS[0], solver='lsqr', max_iter=5000, tol=0.0001))
        model.fit(np.clip(x[train_idx][:, keep], -1000000.0, 1000000.0), y)
        pred = model.predict(np.clip(x[test_idx][:, keep], -1000000.0, 1000000.0))
        lookup = sub.groupby('can')['target'].mean().to_dict()
        for row_id, can, value in zip(test_sub['id'], target_test_cans, pred):
            if can in lookup:
                value = lookup[can]
            predictions.append((int(row_id), float(value), target))
    candidate = pd.DataFrame(predictions, columns=['id', 'target', 'target_type'])
    candidate = candidate.sort_values('id').reset_index(drop=True)
    if len(candidate) != len(data.test):
        raise RuntimeError('candidate row count mismatch')
    return candidate[['id', 'target']].reset_index(drop=True)

# ===== f01_engine.py =====
"""F01 - availability-stratified chained identity engine composite for ei/eea/egb.

Faithful in-memory port of build_f01_composite.py. Produces a 4940-row
id,target DataFrame: starts from a base candidate (default = the F03 clean
candidate) and overrides ei/eea/egb with the chain-identity composite.
"""

def f01_engine_full_target_prediction(data, target: str, partner_predictions):
    train = data.train[data.train['target_type'].eq(target)].reset_index(drop=True)
    test = data.test[data.test['target_type'].eq(target)].reset_index(drop=True)
    cans = train['can'].tolist()
    test_cans = test['can'].tolist()
    y = train['target'].to_numpy(float)
    xf = fc.morgan_count_block(cans)
    xd = fc.descriptor_block(cans)
    xf_test = fc.morgan_count_block(test_cans)
    xd_test = fc.descriptor_block(test_cans)
    partners = fc.PARTNERS[target]
    L = np.full((len(cans), len(partners)), np.nan)
    Lt = np.full((len(test_cans), len(partners)), np.nan)
    for j, prop in enumerate(partners):
        if prop not in data.wide.columns:
            continue
        L[:, j] = [data.wide.loc[c, prop] if c in data.wide.index and pd.notna(data.wide.loc[c, prop]) else np.nan for c in cans]
        Lt[:, j] = [data.wide.loc[c, prop] if c in data.wide.index and pd.notna(data.wide.loc[c, prop]) else np.nan for c in test_cans]
    P = np.array([[partner_predictions[p][c] for p in partners] for c in cans])
    Pt = np.array([[partner_predictions[p][c] for p in partners] for c in test_cans])
    fill = np.where(np.isnan(L), P, L)
    fill_t = np.where(np.isnan(Lt), Pt, Lt)
    isp = np.isnan(L).astype(float)
    isp_t = np.isnan(Lt).astype(float)
    comps, signs = fc.IDENTITY[target]
    idx = [partners.index(c) for c in comps]
    base = sum((s * fill[:, i] for s, i in zip(signs, idx)))
    base_t = sum((s * fill_t[:, i] for s, i in zip(signs, idx)))
    hub = HuberRegressor().fit(base.reshape(-1, 1), y)
    pred = hub.predict(base_t.reshape(-1, 1))
    if fc.IDENTITY_RESID[target] > 0:
        model = ExtraTreesRegressor(500, min_samples_leaf=3, random_state=fc.SEED, n_jobs=-1)
        model.fit(np.hstack([xd, fill, isp]), y - hub.predict(base.reshape(-1, 1)))
        pred += model.predict(np.hstack([xd_test, fill_t, isp_t]))
    return pd.Series(pred, index=test['id'].astype(int))

def f01_engine_build_f01(data_dir, base_df=None):
    data = fc.load_data(data_dir, include_archive=False)
    if base_df is None:
        base = f03_clean_build_f03(data_dir)
    else:
        base = base_df
    if list(base.columns) != ['id', 'target'] or len(base) != len(data.test):
        raise RuntimeError('base candidate schema mismatch')
    out = base.set_index('id')['target'].copy()
    need = list(dict.fromkeys(data.train['can'].tolist() + data.test['can'].tolist()))
    partner_predictions = {}
    needed_props = sorted({p for target in ('ei', 'eea', 'egb') for p in fc.PARTNERS[target]})
    for prop in needed_props:
        rows = data.all_labels[data.all_labels['target_type'].eq(prop)].groupby('can')['target'].mean()
        pcans = rows.index.tolist()
        xf = fc.morgan_count_block(pcans)
        xd = fc.descriptor_block(pcans)
        xn = fc.morgan_count_block(need)
        xdn = fc.descriptor_block(need)
        partner_model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=fc.SEED, n_jobs=-1)
        partner_model.fit(np.hstack([xd, xf]), rows.to_numpy(float))
        pred = partner_model.predict(np.hstack([xdn, xn]))
        partner_predictions[prop] = dict(zip(need, pred))
        for can, value in rows.items():
            partner_predictions[prop][can] = float(value)
    for target in ('ei', 'eea', 'egb'):
        pred = f01_engine_full_target_prediction(data, target, partner_predictions)
        ids = data.test.loc[data.test['target_type'].eq(target), 'id'].astype(int)
        out.loc[ids] = pred.loc[ids]
    result = pd.DataFrame({'id': data.test['id'].astype(int), 'target': out.loc[data.test['id'].astype(int)].to_numpy(float)})
    if len(result) != len(data.test) or result['id'].duplicated().any() or (not np.isfinite(result['target']).all()):
        raise RuntimeError('composite validation failed')
    return result.reset_index(drop=True)

# ===== f02_engine.py =====
"""F02 - eps/nc joint physics engine composite in ionic coordinates.

Faithful in-memory port of build_f02_composite.py. Produces a 4940-row
id,target DataFrame: starts from a base candidate (default = the F03 clean
candidate) and overrides eps/nc with the ionic-coordinate physics composite.
"""

def f02_engine_build_f02(data_dir, base_df=None):
    data = fc.load_data(data_dir, include_archive=False)
    if base_df is None:
        base = f03_clean_build_f03(data_dir)
    else:
        base = base_df
    out = base.set_index('id')['target'].copy()
    all_need = list(dict.fromkeys(data.train['can'].tolist() + data.test['can'].tolist()))
    partner_pred = {}
    for prop in ('eps', 'nc'):
        rows = data.all_labels[data.all_labels['target_type'].eq(prop)].groupby('can')['target'].mean()
        cans = rows.index.tolist()
        model = ExtraTreesRegressor(300, min_samples_leaf=2, random_state=fc.SEED, n_jobs=-1)
        model.fit(np.hstack([fc.descriptor_block(cans), fc.morgan_count_block(cans)]), rows.to_numpy(float))
        pred = model.predict(np.hstack([fc.descriptor_block(all_need), fc.morgan_count_block(all_need)]))
        partner_pred[prop] = dict(zip(all_need, pred))
        partner_pred[prop].update({c: float(v) for c, v in rows.items()})
    pairs = data.wide[data.wide['eps'].notna() & data.wide['nc'].notna()]
    pair_cans = pairs.index.tolist()
    ionic_model = fc.fit_ionic_model(pair_cans, (pairs['eps'] - pairs['nc'] ** 2).to_numpy(float))
    ionic_test = dict(zip(all_need, fc.predict_ionic(ionic_model, all_need)))
    for target, partner in (('eps', 'nc'), ('nc', 'eps')):
        tr = data.train[data.train['target_type'].eq(target)].reset_index(drop=True)
        te = data.test[data.test['target_type'].eq(target)].reset_index(drop=True)
        cans, tcans = (tr['can'].tolist(), te['can'].tolist())
        xf, xt = (fc.morgan_count_block(cans), fc.morgan_count_block(tcans))
        xd, xdt = (fc.descriptor_block(cans), fc.descriptor_block(tcans))
        b0 = fc.fit_predict_structure_blend(xf, xd, tr['target'].to_numpy(float), xt, xdt)
        observed = np.array([c in data.wide.index and pd.notna(data.wide.loc[c, partner]) for c in tcans])
        pval = np.array([data.wide.loc[c, partner] if c in data.wide.index and pd.notna(data.wide.loc[c, partner]) else partner_pred[partner][c] for c in tcans], float)
        ion = np.array([ionic_test[c] for c in tcans], float)
        if target == 'eps':
            phys = pval ** 2 + ion
        else:
            phys = np.sqrt(np.clip(pval - ion, 1.0, None))
        pred = np.where(observed, phys, 0.5 * phys + 0.5 * b0)
        out.loc[te['id'].astype(int)] = pred
    result = pd.DataFrame({'id': data.test['id'].astype(int), 'target': out.loc[data.test['id'].astype(int)].to_numpy(float)})
    if len(result) != len(data.test) or not np.isfinite(result['target']).all():
        raise RuntimeError('invalid F02 candidate')
    return result.reset_index(drop=True)

# ===== f04_gpr.py =====
"""F04 - exploratory Gaussian-process candidate, trained from official inputs.

Faithful in-memory port of F04_gpr_candidate.py. Produces a 4940-row id,target
DataFrame: initialises from the F03 clean candidate (fully overwritten for all
seven targets) and replaces every target with a Gaussian-process prediction.
"""

def f04_gpr_build_f04(data_dir):
    data = fc.load_data(data_dir, include_archive=False)
    base = f03_clean_build_f03(data_dir)
    out = base.set_index('id')['target'].copy()
    for target in fc.TARGETS:
        tr = data.train[data.train['target_type'].eq(target)].reset_index(drop=True)
        te = data.test[data.test['target_type'].eq(target)].reset_index(drop=True)
        cans, tcans = (tr['can'].tolist(), te['can'].tolist())
        xtr = np.hstack([fc.descriptor_block(cans), fc.morgan_count_block(cans)])
        xte = np.hstack([fc.descriptor_block(tcans), fc.morgan_count_block(tcans)])
        scaler = StandardScaler().fit(np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0))
        a = scaler.transform(np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0))
        b = scaler.transform(np.nan_to_num(xte, nan=0.0, posinf=0.0, neginf=0.0))
        ncomp = min(32, a.shape[0] - 1, a.shape[1])
        pca = PCA(n_components=ncomp, random_state=fc.SEED).fit(a)
        a, b = (pca.transform(a), pca.transform(b))
        kernel = ConstantKernel(1.0, (0.001, 1000.0)) * RBF(1.0, (0.01, 100.0)) + WhiteKernel(0.05, (0.0001, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=fc.SEED)
        gp.fit(a, tr['target'].to_numpy(float))
        pred = gp.predict(b)
        out.loc[te['id'].astype(int)] = pred
    result = pd.DataFrame({'id': data.test['id'].astype(int), 'target': out.loc[data.test['id'].astype(int)].to_numpy(float)})
    return result.reset_index(drop=True)

# ===== f05_multi.py =====
"""F05 - matched concat-selector multitask control.

Faithful in-memory port of F05_multitask_candidate.py. Produces a 4940-row
id,target DataFrame: initialises from the F03 clean candidate (fully overwritten
for all seven targets) and replaces every target with a multitask MLP prediction.
"""

def f05_multi_build_f05(data_dir):
    data = fc.load_data(data_dir, include_archive=False)
    base = f03_clean_build_f03(data_dir).set_index('id')['target'].copy()
    allc = list(dict.fromkeys(data.train['can'].tolist() + data.test['can'].tolist()))
    ix = {c: i for i, c in enumerate(allc)}
    raw = np.hstack([fc.descriptor_block(allc), fc.morgan_count_block(allc)])
    sc = StandardScaler().fit(np.nan_to_num(raw))
    pca = PCA(n_components=min(96, raw.shape[0] - 1, raw.shape[1]), random_state=fc.SEED).fit(sc.transform(np.nan_to_num(raw)))
    feat = pca.transform(sc.transform(np.nan_to_num(raw)))
    ti = {t: i for i, t in enumerate(fc.TARGETS)}
    rows = []
    ys = []
    means = np.zeros(7)
    stds = np.ones(7)
    for t in fc.TARGETS:
        y = data.train.loc[data.train['target_type'].eq(t), 'target'].to_numpy(float)
        means[ti[t]] = y.mean()
        stds[ti[t]] = max(y.std(), 1e-06)
    for _, r in data.train.iterrows():
        z = np.zeros(7)
        z[ti[r['target_type']]] = 1
        rows.append(np.r_[feat[ix[r['can']]], z])
        ys.append((float(r['target']) - means[ti[r['target_type']]]) / stds[ti[r['target_type']]])
    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', alpha=0.001, learning_rate_init=0.001, max_iter=250, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=fc.SEED, verbose=False)
    model.fit(np.asarray(rows), np.asarray(ys))
    for t in fc.TARGETS:
        te = data.test[data.test['target_type'].eq(t)]
        z = np.zeros((len(te), 7))
        z[:, ti[t]] = 1
        pred = model.predict(np.hstack([feat[[ix[c] for c in te['can']]], z])) * stds[ti[t]] + means[ti[t]]
        base.loc[te['id'].astype(int)] = pred
    out = pd.DataFrame({'id': data.test['id'].astype(int), 'target': base.loc[data.test['id'].astype(int)].to_numpy(float)})
    return out.reset_index(drop=True)

# ===== f06_distill.py =====
"""F06 - bounded PI1M pseudo-label student candidate.

Faithful in-memory port of F06_pi1m_distill_candidate.py. Produces a 4940-row
id,target DataFrame: starts from a base candidate (default = the F03 clean
candidate) and overrides egb/ei/eea/nc/eps with a PI1M pseudo-label student.
Reads only train.csv / test.csv / PI1M.csv from disk.
"""

def f06_distill_build_f06(data_dir, base_df=None):
    data = fc.load_data(data_dir, include_archive=False)
    seed = fc.SEED
    if base_df is None:
        base = f03_clean_build_f03(data_dir)
    else:
        base = base_df
    base = base.set_index('id')['target'].copy()
    pi = pd.read_csv(os.path.join(str(data_dir), 'PI1M.csv'), nrows=50001)['SMILES'].astype(str).iloc[1:50001].tolist()
    for target in ('egb', 'ei', 'eea', 'nc', 'eps'):
        tr = data.train[data.train['target_type'].eq(target)].reset_index(drop=True)
        te = data.test[data.test['target_type'].eq(target)].reset_index(drop=True)
        cans = tr['can'].tolist()
        tcans = te['can'].tolist()
        pcans = [fc.canon_nostereo(s) for s in pi]
        valid = [i for i, c in enumerate(pcans) if c is not None]
        pcans = [pcans[i] for i in valid]
        xtr = fc.descriptor_block(cans)
        xte = fc.descriptor_block(tcans)
        xpi = fc.descriptor_block(pcans)
        sc = StandardScaler().fit(np.nan_to_num(xtr))
        a = sc.transform(np.nan_to_num(xtr))
        b = sc.transform(np.nan_to_num(xte))
        q = sc.transform(np.nan_to_num(xpi))
        teacher = ExtraTreesRegressor(300, min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(a, tr['target'].to_numpy(float))
        yp = teacher.predict(q)
        student = MLPRegressor(hidden_layer_sizes=(96, 48), alpha=0.001, learning_rate_init=0.001, max_iter=180, early_stopping=True, random_state=seed, n_iter_no_change=15)
        xr = np.vstack([q, a, a, a, a, a])
        yr = np.concatenate([yp] + [tr['target'].to_numpy(float)] * 5)
        student.fit(xr, yr)
        pred = student.predict(b)
        base.loc[te['id'].astype(int)] = pred
    out = pd.DataFrame({'id': data.test['id'].astype(int), 'target': base.loc[data.test['id'].astype(int)].to_numpy(float)})
    return out.reset_index(drop=True)

# ===== f10_portfolio.py =====
"""In-memory port of the original round F10 no-archive portfolio
(the original builder script).

No CSV reads/writes, no archive paths, no hashes, no manifest, no path guards.
Components are passed as in-memory (id,target) DataFrames and the per-target source
selection is applied directly. DEFAULT_SELECTION is ported verbatim from the source.

Deliberate deviations from the source (non-behavioral):
- Component loading is in-memory (DataFrames) instead of pd.read_csv(path).
- Manifest/hash/overwrite-guard/local_eval-path-rejection logic is dropped
  (logging/guarding only; no effect on the numeric (id,target) output).
"""
f10_portfolio_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
f10_portfolio_DEFAULT_SELECTION = {'tg': 'c282', 'egc': 'c282', 'egb': 'f01', 'ei': 'f06', 'eea': 'c282', 'nc': 'f02', 'eps': 'f02'}
f10_portfolio__COMPONENT_NAMES = ('c282', 'f01', 'f06', 'f02')

def f10_portfolio__load_component(name: str, df: pd.DataFrame, official_ids: np.ndarray) -> pd.Series:
    if list(df.columns) != ['id', 'target']:
        raise RuntimeError(f'{name} has unexpected columns: {list(df.columns)}')
    if len(df) != len(official_ids):
        raise RuntimeError(f'{name} row count {len(df)} != official test rows {len(official_ids)}')
    ids = df['id'].to_numpy(int)
    if not np.array_equal(ids, official_ids):
        raise RuntimeError(f'{name} ID/order mismatch versus official test.csv')
    values = df['target'].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f'{name} contains non-finite predictions')
    if df['id'].duplicated().any():
        raise RuntimeError(f'{name} contains duplicate IDs')
    return pd.Series(values, index=ids, name=name)

def f10_portfolio_build_f10(test_df: pd.DataFrame, components: dict) -> pd.DataFrame:
    """Assemble the F10 no-archive portfolio from in-memory component candidates.

    test_df: official test.csv DataFrame with columns ['id', 'smiles', 'target_type']
             (4,940 rows, ids 1..4940).
    components: dict {name: (id,target) DataFrame} with keys c282, f01, f06, f02.
    Per-target source selection is taken verbatim from the source's
    DEFAULT_SELECTION dict. Returns DataFrame with columns ['id', 'target'] aligned
    to test_df.
    """
    if list(test_df.columns) != ['id', 'smiles', 'target_type']:
        raise RuntimeError(f'Unexpected official test schema: {list(test_df.columns)}')
    if len(test_df) != 4940:
        raise RuntimeError(f'Unexpected official test row count: {len(test_df)}')
    official_ids = test_df['id'].to_numpy(int)
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError('Official test IDs are not sequential 1..4940')
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy()
    if sorted(set(target_type)) != sorted(f10_portfolio_TARGETS):
        raise RuntimeError(f'Unexpected target set: {sorted(set(target_type))}')
    missing = [name for name in f10_portfolio__COMPONENT_NAMES if name not in components]
    if missing:
        raise RuntimeError(f'Missing component(s): {missing}')
    loaded = {name: f10_portfolio__load_component(name, components[name], official_ids) for name in f10_portfolio__COMPONENT_NAMES}
    result = pd.Series(index=official_ids, dtype=float)
    for target, component_name in f10_portfolio_DEFAULT_SELECTION.items():
        mask = target_type == target
        result.loc[official_ids[mask]] = loaded[component_name].loc[official_ids[mask]].to_numpy(float)
    if result.isna().any() or not np.isfinite(result.to_numpy(float)).all():
        raise RuntimeError('Assembled portfolio has missing or non-finite predictions')
    return pd.DataFrame({'id': official_ids, 'target': result.loc[official_ids].to_numpy(float)})

# ===== f11_portfolio.py =====
"""In-memory port of the original round F11 no-archive portfolio
(the original builder script).

No CSV reads/writes, no archive paths, no hashes. Components are passed as
in-memory (id,target) DataFrames and the per-target source selection is applied
directly. SELECTION is ported verbatim from the source.
"""
f11_portfolio_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
f11_portfolio_SELECTION = {'tg': 'c284', 'egc': 'c284', 'egb': 'f01', 'ei': 'f06', 'eea': 'c282', 'nc': 'f02', 'eps': 'f02'}
f11_portfolio__COMPONENT_NAMES = ('c282', 'c284', 'f01', 'f06', 'f02')

def f11_portfolio__load_component(name: str, df: pd.DataFrame, official_ids: np.ndarray) -> pd.Series:
    if list(df.columns) != ['id', 'target']:
        raise RuntimeError(f'{name} has unexpected columns: {list(df.columns)}')
    if len(df) != len(official_ids) or df['id'].duplicated().any():
        raise RuntimeError(f'{name} row count/duplicate check failed')
    if not np.array_equal(df['id'].to_numpy(int), official_ids):
        raise RuntimeError(f'{name} ID order mismatch')
    values = df['target'].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f'{name} contains non-finite predictions')
    return pd.Series(values, index=official_ids)

def f11_portfolio_build_f11(test_df: pd.DataFrame, components: dict) -> pd.DataFrame:
    """Assemble the F11 no-archive portfolio from in-memory component candidates.

    components: dict {name: (id,target) DataFrame} with keys c282, c284, f01, f06, f02.
    Per-target selection from source (read source's SELECTION dict). Returns DataFrame
    with columns ['id', 'target'] aligned to test_df.
    """
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected official test schema/row count')
    official_ids = test_df['id'].to_numpy(int)
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError('Official test IDs are not 1..4940')
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy()
    missing = [name for name in f11_portfolio__COMPONENT_NAMES if name not in components]
    if missing:
        raise RuntimeError(f'Missing component(s): {missing}')
    loaded = {name: f11_portfolio__load_component(name, components[name], official_ids) for name in f11_portfolio__COMPONENT_NAMES}
    result = pd.Series(index=official_ids, dtype=float)
    for target, source in f11_portfolio_SELECTION.items():
        mask = target_type == target
        result.loc[official_ids[mask]] = loaded[source].loc[official_ids[mask]].to_numpy(float)
    if result.isna().any() or not np.isfinite(result.to_numpy(float)).all():
        raise RuntimeError('Assembled F11 has missing/non-finite rows')
    return pd.DataFrame({'id': official_ids, 'target': result.loc[official_ids].to_numpy(float)})

# ===== f14_ensemble.py =====
"""In-memory port of the original round F14 no-archive fixed ensemble
(the original builder script).

No CSV reads/writes, no archive paths, no hashes. Components are passed as
in-memory (id,target) DataFrames and per-target averaging is applied directly.
The averaging map is ported verbatim from the source.
"""
f14_ensemble_AVERAGING = {'ei': ('f11', 'c284'), 'eea': ('c282', 'c285'), 'nc': ('f11', 'c284'), 'eps': ('f11', 'c285')}
f14_ensemble__COMPONENT_NAMES = ('f11', 'c282', 'c284', 'c285')

def f14_ensemble__load(name: str, df: pd.DataFrame, official_ids: np.ndarray) -> pd.Series:
    if list(df.columns) != ['id', 'target'] or len(df) != len(official_ids):
        raise RuntimeError(f'{name} schema/row count invalid')
    if df['id'].duplicated().any() or not np.array_equal(df['id'].to_numpy(int), official_ids):
        raise RuntimeError(f'{name} ID validation failed')
    values = df['target'].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f'{name} contains non-finite targets')
    return pd.Series(values, index=official_ids)

def f14_ensemble_build_f14(test_df: pd.DataFrame, components: dict) -> pd.DataFrame:
    """Assemble the F14 no-archive fixed ensemble from in-memory candidates.

    components: dict with keys f11, c282, c284, c285, each an (id,target) DataFrame.
    Per-target averaging per source's averaging dict (base/strong targets stay F11).
    Returns DataFrame with columns ['id', 'target'] aligned to test_df.
    """
    official_ids = test_df['id'].to_numpy(int)
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy()
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected test.csv schema/rows')
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError('Official test IDs are not 1..4940')
    missing = [name for name in f14_ensemble__COMPONENT_NAMES if name not in components]
    if missing:
        raise RuntimeError(f'Missing component(s): {missing}')
    pred = {name: f14_ensemble__load(name, components[name], official_ids) for name in f14_ensemble__COMPONENT_NAMES}
    result = pred['f11'].copy()
    for target, sources in f14_ensemble_AVERAGING.items():
        ids = official_ids[target_type == target]
        result.loc[ids] = np.mean(np.vstack([pred[source].loc[ids].to_numpy(float) for source in sources]), axis=0)
    return pd.DataFrame({'id': official_ids, 'target': result.loc[official_ids].to_numpy(float)})

# ===== weak_aggregate.py =====
"""In-memory port of the original round F15 no-archive weak-target aggregate
(the original builder script).

No CSV reads/writes, no archive paths, no hashes, no manifest, no path guards.
Components are passed as in-memory (id,target) DataFrames and the fixed per-target
routing is applied directly. Routing is ported verbatim from the source.

Source routing (verbatim):
- tg, egc  -> C284
- egb      -> F11 (result starts as a copy of F11)
- eea      -> equal mean of C282 and C285
- ei/nc/eps -> variant "mean3": mean(F11, C284, C285); "median3": median(F11, C284, C285)

Deliberate deviations from the source (non-behavioral):
- Component loading is in-memory (DataFrames) instead of pd.read_csv(path).
- Manifest/hash/overwrite-guard/local_eval-path-rejection logic is dropped
  (logging/guarding only; no effect on the numeric (id,target) output).
"""
weak_aggregate_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
weak_aggregate__STRONG_MAP = {'tg': 'c284', 'egc': 'c284', 'egb': 'f11'}
weak_aggregate__WEAK_TARGETS = ('ei', 'nc', 'eps')
weak_aggregate__COMPONENT_NAMES = ('f11', 'c282', 'c284', 'c285')

def weak_aggregate__load(name: str, df: pd.DataFrame, official_ids: np.ndarray) -> pd.Series:
    if list(df.columns) != ['id', 'target'] or len(df) != len(official_ids):
        raise RuntimeError(f'{name} schema/row count invalid')
    if df['id'].duplicated().any() or not np.array_equal(df['id'].to_numpy(int), official_ids):
        raise RuntimeError(f'{name} ID validation failed')
    values = df['target'].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f'{name} non-finite predictions')
    return pd.Series(values, index=official_ids)

def weak_aggregate_build_f15(test_df: pd.DataFrame, components: dict, variant: str='mean3') -> pd.DataFrame:
    """Assemble the F15 no-archive weak-target aggregate from in-memory candidates.

    test_df: official test.csv DataFrame with columns ['id', 'smiles', 'target_type']
             (4,940 rows, ids 1..4940).
    components: dict {name: (id,target) DataFrame} with keys f11, c282, c284, c285.
    variant: 'mean3' (default) or 'median3'; selects mean vs median for the
             ei/nc/eps weak targets. Returns DataFrame with columns ['id', 'target']
             aligned to test_df.
    """
    if variant not in ('mean3', 'median3'):
        raise RuntimeError(f'Unsupported variant: {variant}')
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected official test.csv')
    official_ids = test_df['id'].to_numpy(int)
    if not np.array_equal(official_ids, np.arange(1, 4941)):
        raise RuntimeError('Official test IDs are not 1..4940')
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy()
    missing = [name for name in weak_aggregate__COMPONENT_NAMES if name not in components]
    if missing:
        raise RuntimeError(f'Missing component(s): {missing}')
    pred = {name: weak_aggregate__load(name, components[name], official_ids) for name in weak_aggregate__COMPONENT_NAMES}
    result = pred['f11'].copy()
    for target, source in weak_aggregate__STRONG_MAP.items():
        ids = official_ids[target_type == target]
        result.loc[ids] = pred[source].loc[ids].to_numpy(float)
    ids = official_ids[target_type == 'eea']
    result.loc[ids] = np.mean(np.vstack([pred['c282'].loc[ids].to_numpy(float), pred['c285'].loc[ids].to_numpy(float)]), axis=0)
    for target in weak_aggregate__WEAK_TARGETS:
        ids = official_ids[target_type == target]
        stacked = np.vstack([pred['f11'].loc[ids].to_numpy(float), pred['c284'].loc[ids].to_numpy(float), pred['c285'].loc[ids].to_numpy(float)])
        if variant == 'mean3':
            values = np.mean(stacked, axis=0)
        else:
            values = np.median(stacked, axis=0)
        result.loc[ids] = values
    return pd.DataFrame({'id': official_ids, 'target': result.loc[official_ids].to_numpy(float)})

# ===== f18_fixed_blends.py =====
"""In-memory port of the original round F18 no-archive fixed equal blends
(the original builder script, branch='without_archive').

No CSV reads/writes, no archive paths, no hashes, no manifest, no path guards.
Components are passed as in-memory (id,target) DataFrames and the fixed per-target
equal-average routing is applied directly. NOARCHIVE_TARGET_MAP is ported verbatim
from the source.

Source without_archive target map (verbatim):
- tg, egc            -> mean(C284)
- egb                -> mean(F14)
- ei                 -> mean(F14, C287_ET, F06)
- eea                -> mean(C285)
- nc                 -> mean(C287_ET, C287_HUBER, F15, F02)
- eps                -> mean(C287_ET, F02)

Deliberate deviations from the source (non-behavioral):
- Component loading is in-memory (DataFrames) instead of pd.read_csv(path).
- The with_archive (F19) branch and its the original round archive source paths are dropped
  (the no-archive lane is the only supported branch; archive references are
  stripped per the porting rules).
- Manifest/hash/overwrite-guard/round-2-boundary checks are dropped
  (logging/guarding only; no effect on the numeric (id,target) output).
"""
f18_fixed_blends_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
f18_fixed_blends_NOARCHIVE_TARGET_MAP = {'tg': ('c284',), 'egc': ('c284',), 'egb': ('f14',), 'ei': ('f14', 'c287_et', 'f06'), 'eea': ('c285',), 'nc': ('c287_et', 'c287_huber', 'f15', 'f02'), 'eps': ('c287_et', 'f02')}
f18_fixed_blends__COMPONENT_NAMES = ('c284', 'c285', 'f02', 'f06', 'f14', 'f15', 'c287_et', 'c287_huber')

def f18_fixed_blends__load_prediction(name: str, df: pd.DataFrame, ids: np.ndarray) -> pd.Series:
    if list(df.columns) != ['id', 'target'] or len(df) != len(ids):
        raise RuntimeError(f'{name} has invalid schema or row count')
    if df['id'].duplicated().any() or not np.array_equal(df['id'].to_numpy(int), ids):
        raise RuntimeError(f'{name} ID validation failed')
    values = df['target'].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f'{name} contains non-finite predictions')
    return pd.Series(values, index=ids)

def f18_fixed_blends_build_f18(test_df: pd.DataFrame, components: dict, branch: str='without_archive') -> pd.DataFrame:
    """Assemble the F18 no-archive fixed equal blends from in-memory candidates.

    test_df: official test.csv DataFrame with columns ['id', 'smiles', 'target_type']
             (4,940 rows, ids 1..4940).
    components: dict {name: (id,target) DataFrame} with keys
             c284, c285, f02, f06, f14, f15, c287_et, c287_huber.
    branch: must be 'without_archive' (the only supported branch).
    Per-target equal average per NOARCHIVE_TARGET_MAP. Returns DataFrame with columns
    ['id', 'target'] aligned to test_df.
    """
    if branch != 'without_archive':
        raise RuntimeError(f"Unsupported branch: {branch} (only 'without_archive' is ported)")
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected official test.csv schema or row count')
    ids = test_df['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Official test IDs are not 1..4940')
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy()
    if set(target_type) != set(f18_fixed_blends_TARGETS):
        raise RuntimeError('Unexpected target types in official test.csv')
    missing = [name for name in f18_fixed_blends__COMPONENT_NAMES if name not in components]
    if missing:
        raise RuntimeError(f'Missing component(s): {missing}')
    pred = {name: f18_fixed_blends__load_prediction(name, components[name], ids) for name in f18_fixed_blends__COMPONENT_NAMES}
    result = pd.Series(np.nan, index=ids, dtype=float)
    for target, source_names in f18_fixed_blends_NOARCHIVE_TARGET_MAP.items():
        mask_ids = ids[target_type == target]
        matrix = np.vstack([pred[name].loc[mask_ids].to_numpy(float) for name in source_names])
        result.loc[mask_ids] = np.mean(matrix, axis=0)
    if not np.isfinite(result.loc[ids].to_numpy(float)).all():
        raise RuntimeError('Assembled prediction contains non-finite values')
    return pd.DataFrame({'id': ids, 'target': result.loc[ids].to_numpy(float)})

# ===== f21_combo.py =====
"""In-memory port of the original builder script (F21,
without_archive branch).

"Broad equal combo": for each target type, the final prediction is the
equal-weight (arithmetic mean) average of that target's frozen component
prediction vectors.  Only rows whose 'target_type' matches are filled from that
target's components; every other row is left to its own target's components.

The original tool read each component from a frozen 'id,target' prediction CSV
that covered every test id.  This port receives those component frames in
memory instead.  It performs NO CSV reads/writes, references NO the original round paths,
and uses NO hashes.

'components' maps target type -> list of component DataFrames.  Each component
DataFrame must have exactly columns ['id', 'target'] and cover every test id in
the same order as test_df.
"""

def f21_combo_broad_equal_combo(test_df: pd.DataFrame, components: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Equal-weight combination of frozen candidate predictions, per target.

    Parameters
    ----------
    test_df : DataFrame with columns ['id', 'smiles', 'target_type'] (official
        test set, 4940 rows, ids 1..4940).
    components : dict mapping each target type to a list of component
        DataFrames, each with columns ['id', 'target'].

    Returns
    -------
    DataFrame with columns ['id', 'target'] (4,940 rows).
    """
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected test.csv schema')
    ids = test_df['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    target_type = test_df['target_type'].astype(str).str.lower().to_numpy()
    result = np.full(len(ids), np.nan, dtype=np.float64)
    for target, frames in components.items():
        arrays = []
        for frame in frames:
            if list(frame.columns) != ['id', 'target'] or len(frame) != len(ids):
                raise RuntimeError(f'Invalid component schema for {target}')
            if frame['id'].duplicated().any() or not np.array_equal(frame['id'].to_numpy(int), ids):
                raise RuntimeError(f'Invalid component IDs for {target}')
            values = frame['target'].to_numpy(float)
            if not np.isfinite(values).all():
                raise RuntimeError(f'Non-finite component values for {target}')
            arrays.append(values)
        if not arrays:
            raise RuntimeError(f'No components provided for {target}')
        mask = target_type == target
        result[mask] = np.mean(np.vstack([arr[mask] for arr in arrays]), axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError('Output has non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})

# ===== f24_xprop.py =====
"""In-memory port of the original builder script (F24,
without_archive branch).

Cross-property overlay on a frozen base candidate.  For each routed target, a
small model predicts that target from the OTHER targets' labels using official
train rows that have all needed columns present.  On test, only the routed
target's rows are overwritten, using the base candidate's predictions for the
OTHER targets (on the same canonical polymer) as covariates.  Every non-routed
row is left at the base prediction.

The original also computed a K-fold CV R^2 for its manifest; that logging-only
step is omitted because it does not affect the returned predictions.

No CSV reads/writes, no the original round paths, no hashes.
"""
f24_xprop_TARGETS = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')

@dataclass(frozen=True)
class f24_xprop_OverlaySpec:
    target: str
    features: tuple[str, ...]
    model_name: str
f24_xprop_OVERLAYS: list[f24_xprop_OverlaySpec] = [f24_xprop_OverlaySpec('egb', ('egc', 'eea'), 'extra_trees'), f24_xprop_OverlaySpec('egc', ('egb',), 'huber'), f24_xprop_OverlaySpec('ei', ('egb', 'eea'), 'ridge1'), f24_xprop_OverlaySpec('nc', ('eps', 'ei'), 'huber')]

def f24_xprop_canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True)

def f24_xprop_make_model(name: str):
    if name == 'ridge1':
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == 'ridge10':
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == 'huber':
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000))
    if name == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=200, random_state=1729, min_samples_leaf=2, max_features=1.0, n_jobs=1)
    raise ValueError(name)

def f24_xprop_cross_property_overlay(test_df: pd.DataFrame, train_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-property overlay on a frozen base candidate.

    Parameters
    ----------
    test_df : DataFrame with columns ['id', 'smiles', 'target_type'] (official
        test set, 4940 rows, ids 1..4940).
    train_df : DataFrame with columns ['smiles', 'target', 'target_type'].
    base_df : DataFrame with columns ['id', 'target'] (base candidate's
        predictions, one row per test id).

    Returns
    -------
    DataFrame with columns ['id', 'target'] (4,940 rows).
    """
    if list(train_df.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected train.csv schema')
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected test.csv schema')
    ids = test_df['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    if list(base_df.columns) != ['id', 'target'] or len(base_df) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base_df['id'].duplicated().any() or not np.array_equal(base_df['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base IDs')
    base_values = base_df['target'].to_numpy(float)
    if not np.isfinite(base_values).all():
        raise RuntimeError('Base has non-finite predictions')
    train = train_df.copy()
    test = test_df.copy().reset_index(drop=True)
    train['tt'] = train['target_type'].astype(str).str.lower()
    test['tt'] = test['target_type'].astype(str).str.lower()
    if set(train['tt']) != set(f24_xprop_TARGETS) or set(test['tt']) != set(f24_xprop_TARGETS):
        raise RuntimeError('Unexpected target set')
    train['canon'] = train['smiles'].map(f24_xprop_canonical_smiles)
    test['canon'] = test['smiles'].map(f24_xprop_canonical_smiles)
    test['base_prediction'] = base_values
    train_pivot = train.pivot_table(index='canon', columns='tt', values='target', aggfunc='mean')
    test_pivot = test.pivot_table(index='canon', columns='tt', values='base_prediction', aggfunc='mean')
    result = test['base_prediction'].to_numpy(float).copy()
    for overlay in f24_xprop_OVERLAYS:
        columns = [overlay.target, *overlay.features]
        paired_train = train_pivot[columns].dropna()
        if len(paired_train) < 20:
            raise RuntimeError(f'Insufficient paired train rows for {overlay}')
        model = f24_xprop_make_model(overlay.model_name)
        x_train = paired_train[list(overlay.features)].to_numpy(float)
        y_train = paired_train[overlay.target].to_numpy(float)
        model.fit(x_train, y_train)
        paired_test = test_pivot[list(overlay.features)].dropna()
        pred_by_canon = pd.Series(model.predict(paired_test[list(overlay.features)].to_numpy(float)), index=paired_test.index)
        for row_idx, row in test[test['tt'] == overlay.target].iterrows():
            canon = row['canon']
            if canon in pred_by_canon.index:
                result[int(row_idx)] = float(pred_by_canon.loc[canon])
    if not np.isfinite(result).all():
        raise RuntimeError('Output has non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})

# ===== f26_ionic.py =====
"""In-memory port of the original builder script (F26,
without_archive branch).

Ionic co-test EPS/NC overlay.  Two "ionic residual" models (ionic = eps - nc**2)
are trained on official train EPS/NC pairs using polar-fragment + descriptor
features.  Then only EPS/NC test rows whose partner target is also present as a
co-test row for the SAME canonical polymer are updated: the physical
reconstruction (nc**2 + ionic, or sqrt(eps - ionic)) is blended with the frozen
base prediction using eps_weight / nc_weight.

No CSV reads/writes, no the original round paths, no hashes.
"""
f26_ionic_POLAR_SMARTS = {'CF': '[#6][F]', 'CCl': '[#6][Cl]', 'ester': 'C(=O)O', 'carbonyl': '[CX3]=[OX1]', 'ether': '[OD2]([#6])[#6]', 'OH': '[OX2H]', 'nitrile': 'C#N', 'amide': 'C(=O)N', 'NH': '[NX3;H1,H2]', 'sulfone': 'S(=O)(=O)', 'thioether': '[#16X2]', 'aromatic_N': 'n', 'aromatic_O': 'o', 'aromatic_S': 's', 'imide': 'C(=O)NC(=O)', 'siloxane': '[Si][O]', 'phosphate': 'P=O', 'urethane': 'NC(=O)O'}
f26_ionic_PATS = {name: Chem.MolFromSmarts(smarts) for name, smarts in f26_ionic_POLAR_SMARTS.items()}

def f26_ionic_canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True)

def f26_ionic_polar_block(cans) -> np.ndarray:
    rows = []
    for can in cans:
        mol = Chem.MolFromSmiles(str(can))
        if mol is None:
            raise RuntimeError(f'Invalid canonical SMILES: {can}')
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in f26_ionic_PATS.values()]
        row += [Descriptors.TPSA(mol) / heavy, Descriptors.NumHDonors(mol) / heavy, Descriptors.NumHAcceptors(mol) / heavy, Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol) / heavy, Crippen.MolMR(mol) / heavy, Crippen.MolLogP(mol) / heavy, rdMolDescriptors.CalcNumAromaticRings(mol) / heavy]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)

def f26_ionic_ionic_cotest_overlay(test_df: pd.DataFrame, train_df: pd.DataFrame, base_df: pd.DataFrame, eps_weight: float=0.5, nc_weight: float=0.5, nc_leaf: int=2) -> pd.DataFrame:
    """Ionic co-test EPS/NC overlay on a frozen base candidate.

    Parameters
    ----------
    test_df : DataFrame with columns ['id', 'smiles', 'target_type'] (official
        test set, 4940 rows, ids 1..4940).
    train_df : DataFrame with columns ['smiles', 'target', 'target_type'].
    base_df : DataFrame with columns ['id', 'target'] (base candidate's
        predictions, one row per test id).
    eps_weight : blend weight for EPS rows (default 0.5).
    nc_weight : blend weight for NC rows (default 0.5).
    nc_leaf : min_samples_leaf for the NC ExtraTrees model (default 2).

    Returns
    -------
    DataFrame with columns ['id', 'target'] (4,940 rows).
    """
    eps_weight = float(eps_weight)
    nc_weight = float(nc_weight)
    nc_leaf = int(nc_leaf)
    if list(train_df.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected train.csv schema')
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected test.csv schema')
    ids = test_df['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected test IDs')
    if list(base_df.columns) != ['id', 'target'] or len(base_df) != len(ids):
        raise RuntimeError('Invalid base schema')
    if base_df['id'].duplicated().any() or not np.array_equal(base_df['id'].to_numpy(int), ids):
        raise RuntimeError('Invalid base IDs')
    base_values = base_df['target'].to_numpy(float)
    if not np.isfinite(base_values).all():
        raise RuntimeError('Base has non-finite predictions')
    train = train_df.copy()
    test = test_df.copy().reset_index(drop=True)
    train['tt'] = train['target_type'].astype(str).str.lower()
    test['tt'] = test['target_type'].astype(str).str.lower()
    train['canon'] = train['smiles'].map(f26_ionic_canonical_smiles)
    test['canon'] = test['smiles'].map(f26_ionic_canonical_smiles)
    test['base_prediction'] = base_values
    train_pivot = train.pivot_table(index='canon', columns='tt', values='target', aggfunc='mean')
    pair_train = train_pivot[['eps', 'nc']].dropna()
    if len(pair_train) < 50:
        raise RuntimeError('Insufficient official current EPS/NC pairs')
    ionic = pair_train['eps'].to_numpy(float) - pair_train['nc'].to_numpy(float) ** 2
    if (ionic < 0).any():
        raise RuntimeError('Unexpected negative official ionic residual')
    x_pair = f26_ionic_polar_block(pair_train.index)
    eps_model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000))
    eps_model.fit(x_pair, ionic)
    nc_model = ExtraTreesRegressor(n_estimators=800, min_samples_leaf=nc_leaf, random_state=20260805, n_jobs=1)
    nc_model.fit(x_pair, ionic)
    test_pivot = test.pivot_table(index='canon', columns='tt', values='base_prediction', aggfunc='mean')
    x_test = f26_ionic_polar_block(test_pivot.index)
    eps_ionic = pd.Series(eps_model.predict(x_test), index=test_pivot.index)
    nc_ionic = pd.Series(nc_model.predict(x_test), index=test_pivot.index)
    result = test['base_prediction'].to_numpy(float).copy()
    for row_idx, row in test.iterrows():
        canon = row['canon']
        target = row['tt']
        if target == 'eps' and 'nc' in test_pivot.columns and pd.notna(test_pivot.loc[canon].get('nc', np.nan)):
            ionic_pred = max(float(eps_ionic.loc[canon]), 0.02)
            phys = float(test_pivot.loc[canon, 'nc']) ** 2 + ionic_pred
            result[int(row_idx)] = (1.0 - eps_weight) * result[int(row_idx)] + eps_weight * phys
        elif target == 'nc' and 'eps' in test_pivot.columns and pd.notna(test_pivot.loc[canon].get('eps', np.nan)):
            ionic_pred = max(float(nc_ionic.loc[canon]), 0.02)
            phys = math.sqrt(max(float(test_pivot.loc[canon, 'eps']) - ionic_pred, 1.0))
            result[int(row_idx)] = (1.0 - nc_weight) * result[int(row_idx)] + nc_weight * phys
    if not np.isfinite(result).all():
        raise RuntimeError('Output has non-finite predictions')
    return pd.DataFrame({'id': ids, 'target': result})

# ---------------------------------------------------------------------------
# Driver: regenerate every intermediate from official data, then compose the
# V57 compound (noarchive_rank2), char arm, and spread arm.
# ---------------------------------------------------------------------------

def canonicalize_official(value):
    return canonicalize(value)


def run_v57(data_dir, out_path):
    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    if list(train_df.columns) != ['smiles', 'target', 'target_type']:
        raise RuntimeError('Unexpected current train schema')
    if list(test_df.columns) != ['id', 'smiles', 'target_type'] or len(test_df) != 4940:
        raise RuntimeError('Unexpected current test schema')
    if len(train_df) != 7409:
        raise RuntimeError('Unexpected current train row count')
    test_df = test_df.copy().reset_index(drop=True)
    test_df['target_type'] = test_df['target_type'].astype(str).str.lower()
    train_df['target_type'] = train_df['target_type'].astype(str).str.lower()
    ids = test_df['id'].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError('Unexpected official test IDs')

    # ---- leaves: leaf models read official data only ----
    print('[1/6] building C282/C284/C285 reference leaves ...', flush=True)
    c282_submission_df, c282_oof_df, c282_detail_df = c282_build_c282(data_path)
    c284_submission_df, c284_oof_df, c284_detail_df = c284_build_c284(data_path)
    c285_submission_df, c285_oof_df, c285_detail_df = c285_build_c285(data_path)

    print('[2/6] building F03 clean candidate + fable engines F01..F06 ...', flush=True)
    f03 = f03_clean_build_f03(data_path)
    f01 = f01_engine_build_f01(data_path, f03)
    f02 = f02_engine_build_f02(data_path, f03)
    f04 = f04_gpr_build_f04(data_path)
    f05 = f05_multi_build_f05(data_path)
    f06 = f06_distill_build_f06(data_path, f03)

    print('[3/6] building F11/F14/F15/F18/F10 portfolios + C287 zoo ...', flush=True)
    f11 = f11_portfolio_build_f11(test_df, {'c282': c282_submission_df, 'c284': c284_submission_df, 'f01': f01, 'f06': f06, 'f02': f02})
    f14 = f14_ensemble_build_f14(test_df, {'f11': f11, 'c282': c282_submission_df, 'c284': c284_submission_df, 'c285': c285_submission_df})
    f15 = weak_aggregate_build_f15(test_df, {'f11': f11, 'c282': c282_submission_df, 'c284': c284_submission_df, 'c285': c285_submission_df}, 'mean3')
    f10 = f10_portfolio_build_f10(test_df, {'c282': c282_submission_df, 'f01': f01, 'f06': f06, 'f02': f02})

    print('[4/6] building C287 zoo arms + C286v4 + C340 + C927 + C391 ...', flush=True)
    c287_arms = c287_zoo_build_c287(data_path, f14)
    c287_et = c287_arms['ei_dense_extra_trees']
    c287_huber = c287_arms['ei_dense_huber']
    c287_rf = c287_arms['ei_dense_random_forest']
    c287_eea_huber = c287_arms['eea_dense_huber']
    # rebuild F18 with the real c287 arms
    f18 = f18_fixed_blends_build_f18(test_df, {'c284': c284_submission_df, 'c285': c285_submission_df, 'f02': f02, 'f06': f06, 'f14': f14, 'f15': f15, 'c287_et': c287_et, 'c287_huber': c287_huber}, 'without_archive')
    c286v4 = c286_stack_c286_stack(train_df, test_df, f18, c282_oof_df, c282_detail_df, c284_oof_df, c284_detail_df, c285_oof_df, c285_detail_df)
    c340 = c340_wrapper_c340_c282_polymer_genome_wrapper(test_df, c282_oof_df, c282_submission_df)
    # C927 consumes the C340-materialized C279-schema parent frames
    c927_parent_oof = c282_oof_df[['canonical', 'target_type', 'target', 'prediction']].rename(columns={'prediction': 'candidate_prediction'}).copy()
    c927_parent_oof['group'] = c927_parent_oof['canonical'].astype(str)
    c927_parent_test = c282_submission_df[['id', 'target']].copy()
    c927 = c927_build_c927(test_df, c927_parent_test, c927_parent_oof)
    c391 = c391_pi1m_zoo_build_c289(data_path)

    print('[5/6] building F21/F24/F26 stack ...', flush=True)
    f21_components = {
        'eea': [c285_submission_df],
        'egb': [c286v4],
        'egc': [c284_submission_df, c286v4, c287_et, c287_huber, c287_rf, c285_submission_df],
        'ei': [f18, c287_et, f06, f10, c287_rf, c284_submission_df],
        'eps': [f18],
        'nc': [f18, c287_et, c286v4],
        'tg': [c284_submission_df, c286v4, c287_et, c287_huber, c287_rf, c282_submission_df],
    }
    f21 = f21_combo_broad_equal_combo(test_df, f21_components)
    f24 = f24_xprop_cross_property_overlay(test_df, train_df, f21)
    f26 = f26_ionic_ionic_cotest_overlay(test_df, train_df, f24, eps_weight=0.5, nc_weight=0.5, nc_leaf=2)

    print('[6/6] chaining the candidate spine ...', flush=True)

    # ============ reference chain: C292 .. C1572 (rebuilt per reference manifests) ============
    f16 = weak_aggregate_build_f15(test_df, {'f11': f11, 'c282': c282_submission_df, 'c284': c284_submission_df, 'c285': c285_submission_df}, 'median3')
    c287_ridge_a5 = c287_arms['ei_dense_ridge_a5']
    best_compound = f03  # best compound == the F03 clean candidate
    # ---- C292 block ----
    c292 = xprop_overlay_imputed_cross_property_overlay(train_df, test_df, f26, overlay_weight=0.25, gate_delta=0.005, morgan_bits=128, seed=20260808, fast_linear=True)
    c304 = physics_overlay_safe_identity_physics_overlay(train_df, test_df, c292, observed_weight_scale=0.75, cotest_weight_scale=0.75, targets='eea')
    c305 = physics_overlay_safe_identity_physics_overlay(train_df, test_df, c292, observed_weight_scale=0.5, cotest_weight_scale=1.0, targets='ei', disable_cotest=True)
    c306 = physics_overlay_safe_identity_physics_overlay(train_df, test_df, c292, observed_weight_scale=1.25, cotest_weight_scale=1.0, targets='ei', disable_cotest=True)
    c312 = splice_targets(test_df, c292, {'eea': c304, 'ei': c305})
    c326 = physics_overlay_safe_identity_physics_overlay(train_df, test_df, c312, observed_weight_scale=1.0, cotest_weight_scale=1.0, targets='nc')
    c327 = c327_calibrator_cotest_meta_calibrator(train_df, test_df, c312, c282_oof_df, targets='ei,eea,eps,nc')
    c336 = c327_calibrator_cotest_meta_calibrator(train_df, test_df, c327, c282_oof_df, targets='ei')
    c340 = c340_wrapper_c340_c282_polymer_genome_wrapper(test_df, c282_oof_df, c282_submission_df)
    c343 = blend_targets(test_df, c336, {'tg': (0.75, c340)})
    c346 = c346_c347_nonlinear_nonlinear_cotest_calibrator(train_df, test_df, c343, c282_oof_df, targets='egc,ei,eps,nc')
    c348 = splice_targets(test_df, c343, {'egc': c346, 'eps': c346})
    c350 = c350_epsnc_c350_eps_nc_consistency(train_df, test_df, c348, pull=0.5, ionic_leaf=2, weight_eps=1.0, weight_nc=1.0)
    c351 = splice_targets(test_df, c348, {'nc': c350})
    c353 = c350_epsnc_c350_eps_nc_consistency(train_df, test_df, c348, pull=0.75, ionic_leaf=2, weight_eps=1.0, weight_nc=1.0)
    c354 = c350_epsnc_c350_eps_nc_consistency(train_df, test_df, c348, pull=1.0, ionic_leaf=2, weight_eps=1.0, weight_nc=1.0)
    # ---- C356-family blends over C351 ----
    c356_f01_eea = blend_targets(test_df, c351, {'eea': (0.125, f01)})
    c356_f01_egc = blend_targets(test_df, c351, {'egc': (0.25, f01)})
    c356_f01_ei = blend_targets(test_df, c351, {'ei': (0.75, f01)})
    c356_f05_tg0875 = blend_targets(test_df, c351, {'tg': (0.875, f05)})
    c356_f06_ei0625 = blend_targets(test_df, c351, {'ei': (0.625, f06)})
    c356_f06_eps0875 = blend_targets(test_df, c351, {'eps': (0.875, f06)})
    c356_c284_nc = blend_targets(test_df, c351, {'nc': (0.875, c284_submission_df)})
    c361 = splice_targets(test_df, c351, {
        'eea': blend_targets(test_df, c351, {'eea': (0.25, best_compound)}),
        'egc': blend_targets(test_df, c351, {'egc': (0.25, c340)}),
        'ei': blend_targets(test_df, c351, {'ei': (0.125, f05)}),
        'nc': blend_targets(test_df, c351, {'nc': (0.375, c353)}),
    })
    # ---- C370-family blends over C361 ----
    c370_f01_egc = blend_targets(test_df, c361, {'egc': (0.25, f01)})
    c370_f05_tg = blend_targets(test_df, c361, {'tg': (0.2, f05)})
    c371 = splice_targets(test_df, c361, {
        'eea': blend_targets(test_df, c361, {'eea': (0.1, f06)}),
        'egc': blend_targets(test_df, c361, {'egc': (0.025, best_compound)}),
        'ei': blend_targets(test_df, c361, {'ei': (0.1, f06)}),
        'eps': blend_targets(test_df, c361, {'eps': (0.05, best_compound)}),
        'nc': blend_targets(test_df, c361, {'nc': (0.05, best_compound)}),
    })
    c374 = c374_ei_eht_c374_ei_eht(train_df, test_df, c371, c282_oof_df, residual_weight=0.35, ridge_alpha=60.0)
    c377 = splice_targets(test_df, blend_targets(test_df, c371, {'ei': (0.4, c374)}), {'tg': blend_targets(test_df, c361, {'tg': (0.15, c284_submission_df)})})
    c381 = c380_ei_eht_ei_eht_cotest(train_df, test_df, c377, c282_oof_df, residual_weight=0.25, ridge_alpha=60.0, residual_clip=0.6)
    c384 = c327_calibrator_cotest_meta_calibrator(train_df, test_df, c377, c282_oof_df, targets='eea,eps,nc')
    c391 = c391_pi1m_zoo_build_c289(data_dir)
    # ---- C394-family blends over C377 ----
    c394_f06_egb025 = blend_targets(test_df, c377, {'egb': (0.25, f06)})
    c394_c391_tg = blend_targets(test_df, c377, {'tg': (0.5, c391)})
    c394_c361_eps = blend_targets(test_df, c377, {'eps': (0.4, c361)})
    c396 = splice_targets(test_df, c377, {
        'eea': blend_targets(test_df, c377, {'eea': (0.4, c384)}),
        'egb': blend_targets(test_df, c377, {'egb': (0.2, f06)}),
        'egc': blend_targets(test_df, c377, {'egc': (0.25, c391)}),
        'ei': blend_targets(test_df, c377, {'ei': (0.15, c391)}),
        'eps': blend_targets(test_df, c377, {'eps': (0.05, c391)}),
        'nc': blend_targets(test_df, c377, {'nc': (0.25, c361)}),
        'tg': blend_targets(test_df, c377, {'tg': (0.15, c391)}),
    })
    c401 = blend_targets(test_df, c396, {
        'eea': (0.1, f01),
        'egb': (0.15, f01),
        'egc': (0.075, c356_f01_egc),
        'ei': (0.075, c306),
        'eps': (0.2, c394_c361_eps),
        'nc': (0.15, c326),
        'tg': (0.075, c370_f05_tg),
    })
    c402 = c402_eps_surrogate_eps_surrogate_nc_ionic(train_df, test_df, c401, surrogate_nc_model='extra_trees', support_min_similarity=0.35, pull=0.5)
    c404 = splice_targets(test_df, c401, {
        'egb': blend_targets(test_df, c396, {'egb': (0.175, f01)}),
        'nc': blend_targets(test_df, c396, {'nc': (0.175, c326)}),
    })
    c415 = reflected_source(c404, c286v4)
    c419_nc = blend_targets(test_df, c404, {'nc': (0.5, c415)})
    c422 = splice_targets(test_df, c404, {
        'ei': blend_targets(test_df, c404, {'ei': (0.175, c415)}),
        'eps': blend_targets(test_df, c404, {'eps': (0.5, c415)}),
        'nc': blend_targets(test_df, c404, {'nc': (0.24, c415)}),
    })
    c429 = reflected_source(c422, c391)
    c430_nc030 = blend_targets(test_df, c422, {'nc': (0.3, c429)})
    c431 = splice_targets(test_df, c422, {
        'eea': blend_targets(test_df, c422, {'eea': (0.15, c429)}),
        'eps': blend_targets(test_df, c422, {'eps': (0.05, c429)}),
        'nc': blend_targets(test_df, c422, {'nc': (0.1, c429)}),
    })
    c432 = reflected_source(c431, c402)
    c433 = splice_targets(test_df, c431, {'eps': blend_targets(test_df, c431, {'eps': (0.075, c432)})})
    c434 = reflected_source(c433, c381)
    c435 = reflected_source(c433, c384)
    c441 = splice_targets(test_df, c433, {
        'egb': blend_targets(test_df, c433, {'egb': (0.375, c434)}),
        'egc': blend_targets(test_df, c433, {'egc': (0.125, c434)}),
        'ei': blend_targets(test_df, c433, {'ei': (0.45, c434)}),
        'tg': blend_targets(test_df, c433, {'tg': (0.1, c434)}),
    })
    c445 = reflected_source(c441, c336)
    c448 = reflected_source(c441, f18)
    c450 = reflected_source(c441, f10)
    c451 = reflected_source(c441, c287_ridge_a5)
    c452 = reflected_source(c441, f04)
    c453 = reflected_source(c441, f05)
    c454 = reflected_source(c441, c285_submission_df)
    c455 = splice_targets(test_df, c441, {
        'eea': blend_targets(test_df, c441, {'eea': (0.05, c453)}),
        'egc': blend_targets(test_df, c441, {'egc': (0.15, c452)}),
        'ei': blend_targets(test_df, c441, {'ei': (0.075, c451)}),
        'eps': blend_targets(test_df, c441, {'eps': (0.125, c453)}),
        'nc': blend_targets(test_df, c441, {'nc': (0.175, c448)}),
        'tg': blend_targets(test_df, c441, {'tg': (0.65, c454)}),
    })
    c463 = splice_targets(test_df, c455, {
        'eea': blend_targets(test_df, c441, {'eea': (0.06, c453)}),
        'egb': blend_targets(test_df, c433, {'egb': (0.37, c434)}),
        'ei': blend_targets(test_df, c441, {'ei': (0.065, c451)}),
        'tg': blend_targets(test_df, c441, {'tg': (0.725, c454)}),
    })
    c480 = reflected_source(c463, c452)
    c481 = reflected_source(c463, c415)
    c482_src = blend_targets(test_df, c404, {'ei': (0.575, c415)})
    c482 = reflected_source(c463, c482_src)
    c483 = reflected_source(c463, c445)
    c484 = reflected_source(c463, c454)
    c481_egc = blend_targets(test_df, c463, {'egc': (0.125, c481)})
    c483_eps = blend_targets(test_df, c463, {'eps': (0.25, c483)})
    c487 = splice_targets(test_df, c463, {
        'eea': blend_targets(test_df, c463, {'eea': (0.075, c480)}),
        'egb': blend_targets(test_df, c463, {'egb': (0.035, best_compound)}),
        'egc': blend_targets(test_df, c463, {'egc': (0.175, c481)}),
        'ei': blend_targets(test_df, c463, {'ei': (1.0, c482)}),
        'eps': blend_targets(test_df, c463, {'eps': (0.35, c483)}),
        'nc': blend_targets(test_df, c463, {'nc': (0.075, c484)}),
        'tg': blend_targets(test_df, c441, {'tg': (0.075, c452)}),
    })
    c488 = reflected_source(c487, c482_src)
    c489 = reflected_source(c487, c445)
    c490 = reflected_source(c487, c452)
    c491 = reflected_source(c487, c415)
    c492 = reflected_source(c487, c454)
    c491_egc010 = blend_targets(test_df, c487, {'egc': (0.1, c491)})
    c493 = splice_targets(test_df, c487, {
        'egc': blend_targets(test_df, c487, {'egc': (0.025, c491)}),
        'ei': blend_targets(test_df, c487, {'ei': (0.05, c488)}),
    })
    c504 = reflected_source(c493, c356_f01_eea)
    c505 = reflected_source(c493, c453)
    c506 = reflected_source(c493, c483)
    c507 = reflected_source(c493, c356_f01_ei)
    c508 = reflected_source(c493, c354)
    c509 = reflected_source(c493, c419_nc)
    c510 = reflected_source(c493, c394_c391_tg)
    c511 = splice_targets(test_df, c493, {
        'eea': blend_targets(test_df, c493, {'eea': (0.35, c504)}),
        'egb': blend_targets(test_df, c493, {'egb': (0.05, c505)}),
        'egc': blend_targets(test_df, c493, {'egc': (0.175, c506)}),
        'ei': blend_targets(test_df, c493, {'ei': (0.175, c507)}),
        'eps': blend_targets(test_df, c493, {'eps': (0.5, c508)}),
        'nc': blend_targets(test_df, c493, {'nc': (0.5, c509)}),
        'tg': blend_targets(test_df, c493, {'tg': (0.25, c510)}),
    })
    c530 = reflected_source(c511, c484)
    c531 = reflected_source(c511, c282_submission_df)
    c533 = reflected_source(c511, c504)
    c534 = reflected_source(c511, c354)
    c535 = reflected_source(c511, f01)
    c536 = splice_targets(test_df, c511, {
        'eea': blend_targets(test_df, c404, {'eea': (0.075, c415)}),
        'egb': blend_targets(test_df, c511, {'egb': (0.1, c530)}),
        'egc': blend_targets(test_df, c511, {'egc': (0.1, c531)}),
        'ei': blend_targets(test_df, c511, {'ei': (0.075, f02)}),
        'eps': blend_targets(test_df, c511, {'eps': (1.0, c533)}),
        'nc': blend_targets(test_df, c511, {'nc': (0.25, c534)}),
        'tg': blend_targets(test_df, c511, {'tg': (0.035, c535)}),
    })
    c543 = reflected_source(c536, c292)
    c544 = reflected_source(c536, c483_eps)
    c545 = reflected_source(c536, c492)
    c550 = splice_targets(test_df, c536, {
        'eea': blend_targets(test_df, c536, {'eea': (0.075, c340)}),
        'egb': blend_targets(test_df, c536, {'egb': (0.1, c284_submission_df)}),
        'egc': blend_targets(test_df, c536, {'egc': (0.075, c429)}),
        'ei': blend_targets(test_df, c536, {'ei': (0.35, c543)}),
        'eps': blend_targets(test_df, c536, {'eps': (1.0, c544)}),
        'nc': blend_targets(test_df, c536, {'nc': (0.175, c489)}),
        'tg': blend_targets(test_df, c536, {'tg': (0.5, c545)}),
    })
    c552 = reflected_source(c550, c384)
    c553 = reflected_source(c550, c511)
    c554 = reflected_source(c550, c481_egc)
    c556 = reflected_source(c550, c481)
    c557 = reflected_source(c550, c356_c284_nc)
    c558 = reflected_source(c550, c505)
    c559 = splice_targets(test_df, c550, {
        'eea': blend_targets(test_df, c550, {'eea': (0.25, c552)}),
        'egb': blend_targets(test_df, c550, {'egb': (1.0, c553)}),
        'egc': blend_targets(test_df, c550, {'egc': (1.0, c554)}),
        'ei': blend_targets(test_df, c422, {'ei': (0.1, c429)}),
        'eps': blend_targets(test_df, c550, {'eps': (0.8, c556)}),
        'nc': blend_targets(test_df, c550, {'nc': (0.075, c557)}),
        'tg': blend_targets(test_df, c550, {'tg': (0.015, c558)}),
    })
    c561 = reflected_source(c559, c511)
    c562 = reflected_source(c559, c370_f01_egc)
    c563 = reflected_source(c559, c434)
    c564 = reflected_source(c559, f16)
    c565 = reflected_source(c559, c430_nc030)
    c566 = reflected_source(c559, c489)
    c574 = splice_targets(test_df, c559, {
        'eea': blend_targets(test_df, c559, {'eea': (0.65, c561)}),
        'egb': blend_targets(test_df, c559, {'egb': (1.0, c561)}),
        'egc': blend_targets(test_df, c559, {'egc': (0.075, c562)}),
        'ei': blend_targets(test_df, c559, {'ei': (0.5, c563)}),
        'eps': blend_targets(test_df, c559, {'eps': (0.125, c564)}),
        'nc': blend_targets(test_df, c559, {'nc': (0.35, c565)}),
        'tg': blend_targets(test_df, c559, {'tg': (0.175, c566)}),
    })
    c576 = reflected_source(c574, c552)
    c577 = reflected_source(c574, c511)
    c578 = reflected_source(c574, c491_egc010)
    c579 = reflected_source(c574, c556)
    c580 = reflected_source(c574, c566)
    c581 = reflected_source(c574, c450)
    c582 = reflected_source(c574, c535)
    c590 = splice_targets(test_df, c574, {
        'eea': blend_targets(test_df, c574, {'eea': (0.25, c576)}),
        'egb': blend_targets(test_df, c574, {'egb': (1.0, c577)}),
        'egc': blend_targets(test_df, c574, {'egc': (1.0, c578)}),
        'ei': blend_targets(test_df, c574, {'ei': (0.35, c579)}),
        'eps': blend_targets(test_df, c574, {'eps': (0.1, c580)}),
        'nc': blend_targets(test_df, c574, {'nc': (0.1, c581)}),
        'tg': blend_targets(test_df, c574, {'tg': (0.01, c582)}),
    })
    c592 = reflected_source(c590, c511)
    c593 = reflected_source(c590, c394_f06_egb025)
    c594 = reflected_source(c590, c434)
    c595 = reflected_source(c590, c544)
    c596 = reflected_source(c590, c429)
    c597 = reflected_source(c590, c481)
    c605 = splice_targets(test_df, c590, {
        'eea': blend_targets(test_df, c590, {'eea': (0.65, c592)}),
        'egb': blend_targets(test_df, c590, {'egb': (0.5, c593)}),
        'egc': blend_targets(test_df, c590, {'egc': (0.175, c594)}),
        'ei': blend_targets(test_df, c590, {'ei': (0.5, c595)}),
        'eps': blend_targets(test_df, c590, {'eps': (0.1, c596)}),
        'nc': blend_targets(test_df, c590, {'nc': (0.25, c597)}),
        'tg': blend_targets(test_df, c590, {'tg': (0.075, c594)}),
    })
    c607 = reflected_source(c605, c435)
    c608 = reflected_source(c605, c490)
    c609 = reflected_source(c605, c481)
    c610 = reflected_source(c605, c356_f06_ei0625)
    c611 = reflected_source(c605, c356_f06_eps0875)
    c612 = reflected_source(c605, c553)
    c613 = reflected_source(c605, c356_f05_tg0875)
    c621 = splice_targets(test_df, c605, {
        'eea': blend_targets(test_df, c605, {'eea': (0.1, c607)}),
        'egb': blend_targets(test_df, c605, {'egb': (0.035, c608)}),
        'egc': blend_targets(test_df, c605, {'egc': (0.125, c609)}),
        'ei': blend_targets(test_df, c605, {'ei': (0.175, c610)}),
        'eps': blend_targets(test_df, c605, {'eps': (0.075, c611)}),
        'nc': blend_targets(test_df, c605, {'nc': (1.0, c612)}),
        'tg': blend_targets(test_df, c605, {'tg': (0.005, c613)}),
    })
    c924 = splice_targets(test_df, c621, {
        'egb': blend_targets(test_df, c511, {'egb': (0.025, c531)}),
        'egc': blend_targets(test_df, c605, {'egc': (0.125, c612)}),
        'ei': blend_targets(test_df, c605, {'ei': (0.175, c612)}),
        'eps': blend_targets(test_df, c621, {'eps': (0.075, c340)}),
        'nc': blend_targets(test_df, c621, {'nc': (0.025, c340)}),
    })
    c942 = epsnc_b3_overlay(test_df, train_df, c924, eps_weight=0.1, nc_weight=0.2, consistency_pull=0.81)
    c947 = c407_weak_zoo_weak_target_zoo(train_df, test_df, c942, targets='ei,eps,nc,tg,egc', models='ridge_200,extra_trees,lightgbm', morgan_bits=512)
    c925 = c407_weak_zoo_weak_target_zoo(train_df, test_df, c621, targets='eps,nc', models='lightgbm', morgan_bits=512)
    c949 = splice_targets(test_df, c942, {
        'egc': blend_targets(test_df, c942, {'egc': (0.1, c947)}),
        'ei': blend_targets(test_df, c942, {'ei': (0.1, c947)}),
        'nc': blend_targets(test_df, c942, {'nc': (0.05, c947)}),
        'tg': blend_targets(test_df, c942, {'tg': (0.1, c947)}),
    })
    c950 = reflected_source(c949, c947)
    c952 = splice_targets(test_df, c949, {'eps': blend_targets(test_df, c949, {'eps': (0.025, c950)})})
    c954 = splice_targets(test_df, c952, {'tg': blend_targets(test_df, c952, {'tg': (0.125, c947)})})
    c982 = identity_overlay_identity_overlay(test_df, train_df, c954, eea_weight=0.1, ei_weight=0.1, egb_weight=0.0)
    c943 = epsnc_b3_overlay(test_df, train_df, c924, eps_weight=0.1, nc_weight=0.25, consistency_pull=0.65)
    c983 = splice_targets(test_df, c982, {'eps': c943})
    c1004 = epsnc_ionic_overlay(train_df, test_df, c983, eps_weight=0.05, nc_weight=0.0, mode='extra_trees_raw')
    c985 = physics_overlay_safe_identity_physics_overlay(train_df, test_df, c983, observed_weight_scale=0.0, cotest_weight_scale=0.2, targets='ei,eea')
    c1037 = physics_overlay_safe_identity_physics_overlay(train_df, test_df, c983, observed_weight_scale=0.0, cotest_weight_scale=1.5, targets='egb')
    c1043 = blend_targets(test_df, c952, {'tg': (0.11, c947)})
    c1053 = splice_targets(test_df, c1037, {'tg': c1043})
    c1057 = splice_targets(test_df, c1053, {'eps': c1004})
    c1058 = reflected_source(c1057, c985)
    c1074 = blend_targets(test_df, c1057, {'ei': (1.0, c1058)})
    c1075 = reflected_source(c1074, c985)
    c1085 = blend_targets(test_df, c1074, {'ei': (0.8, c1075)})
    c1088 = reflected_source(c1085, c402)
    c1114 = blend_targets(test_df, c1085, {'eps': (0.375, c1088)})
    c990 = epsnc_ionic_overlay(train_df, test_df, c983, eps_weight=0.0, nc_weight=0.1, mode='median')
    c1119 = reflected_source(c1114, c990)
    c1144 = blend_targets(test_df, c1114, {'nc': (0.8, c1119)})
    c1172 = blend_targets(test_df, c1144, {'egc': (0.035, c947)})
    c1175 = reflected_source(c1172, c985)
    c1188 = blend_targets(test_df, c1172, {'eea': (1.0, c1175)})
    c1189 = reflected_source(c1188, c985)
    c1201 = blend_targets(test_df, c1188, {'eea': (0.15, c1189)})
    c1211 = blend_targets(test_df, c1201, {'tg': (0.125, c927)})
    c1215 = blend_targets(test_df, c1201, {'egb': (0.005, c927)})
    c1228 = blend_targets(test_df, c1201, {'eea': (0.035, c927)})
    c1230 = splice_targets(test_df, c1201, {'eea': c1228, 'egb': c1215, 'tg': c1211})
    c1345 = epsnc_b3_overlay(test_df, train_df, c1230, eps_weight=0.07, nc_weight=0.03, consistency_pull=0.3)
    c1348 = identity_overlay_identity_overlay(test_df, train_df, c1345, eea_weight=0.0, ei_weight=0.075, egb_weight=0.0)
    c1284 = epsnc_b3_overlay(test_df, train_df, c1230, eps_weight=0.05, nc_weight=0.05, consistency_pull=0.3)
    c1295 = identity_overlay_identity_overlay(test_df, train_df, c1284, eea_weight=0.01, ei_weight=0.0, egb_weight=0.0)
    c1282 = epsnc_b3_overlay(test_df, train_df, c1230, eps_weight=0.02, nc_weight=0.1, consistency_pull=0.5)
    c1349 = splice_targets(test_df, c1348, {'eea': c1295, 'nc': c1282})
    c1369 = c1369_stack_fast_direct_stack(train_df, test_df, c1349, targets='egc,ei,nc,eps', models='ridge_20,ridge_80,ridge_250', morgan_bits=256, seed=20260808, components=None)
    c1370 = reflected_source(c1349, c1369)
    c1374 = blend_targets(test_df, c1349, {'ei': (0.165, c1370)})
    c1378 = reflected_source(c1374, c286v4)
    c1376 = reflected_source(c1374, c391)
    c1377 = reflected_source(c1374, c925)
    c1375 = reflected_source(c1374, c947)
    c1380 = blend_targets(test_df, c1374, {
        'eea': (0.02, c1378), 'egb': (0.1, c1378), 'egc': (0.075, c1376),
        'ei': (0.1, c1377), 'eps': (0.05, c1378), 'tg': (0.2, c1378),
    })
    c1382 = blend_targets(test_df, c1380, {
        'eea': (0.01, c1376), 'egb': (0.035, c1376), 'egc': (0.1, c1378),
        'ei': (0.01, c1377), 'tg': (0.1, c1376),
    })
    c1384 = blend_targets(test_df, c1382, {
        'eea': (0.1, c1375), 'egb': (0.05, c1378), 'egc': (0.1, c1377),
        'ei': (0.005, c1377), 'tg': (0.035, c1378),
    })
    c1392 = reflected_source(c1384, c340)
    c1394 = blend_targets(test_df, c1384, {
        'eea': (0.035, c340), 'egb': (0.005, c1392), 'egc': (0.01, c340),
        'ei': (0.075, c340), 'eps': (0.02, c340), 'nc': (0.1, c340), 'tg': (0.3, c1392),
    })
    c1396 = blend_targets(test_df, c1394, {
        'eea': (0.005, c1392), 'egb': (0.0025, c340), 'egc': (0.0025, c340),
        'ei': (0.01, c340), 'eps': (0.0025, c340), 'nc': (0.0025, c1392), 'tg': (0.035, c340),
    })
    c1398 = c380_ei_eht_ei_eht_cotest(train_df, test_df, c1396, c282_oof_df, residual_weight=0.05, ridge_alpha=60.0, residual_clip=0.3)
    c1410 = epsnc_b3_overlay(test_df, train_df, c1398, eps_weight=0.1, nc_weight=0.025, consistency_pull=0.3)
    c1433 = c407_weak_zoo_weak_target_zoo(train_df, test_df, c1410, targets='ei,eea,eps,nc', models='ridge_200,extra_trees', morgan_bits=384)
    c1446 = c1446_physics_proj_c1446_physics_projection(train_df, test_df, c1410, egb_pull=0.08, gap_pull=0.05, epsnc_pull=0.02)
    c1447 = splice_targets(test_df, c1410, {'eea': c1446, 'egb': c1446, 'egc': c1446})
    c1494 = c1446_physics_proj_c1446_physics_projection(train_df, test_df, c1447, egb_pull=0.0, gap_pull=0.02, epsnc_pull=0.05)
    c1493 = c1446_physics_proj_c1446_physics_projection(train_df, test_df, c1447, egb_pull=0.04, gap_pull=0.03, epsnc_pull=0.04)
    c1496 = splice_targets(test_df, c1447, {'egc': c1493, 'eea': c1494, 'ei': c1494})
    c1506 = physics_overlay_safe_identity_physics_overlay(train_df, test_df, c1496, observed_weight_scale=0.3, cotest_weight_scale=0.1, targets='ei,eea,egb,eps,nc')
    c1530 = reflected_source(c1496, c1433)
    c355_egc_030 = blend_targets(test_df, c1496, {'egc': (0.3, c1530)})
    c355_eps_035 = blend_targets(test_df, c1496, {'eps': (0.035, c1530)})
    c1532 = splice_targets(test_df, c1496, {'egc': c355_egc_030, 'eps': c355_eps_035})
    c355_tg_035 = blend_targets(test_df, c1394, {'tg': (0.035, c340)})
    c1535 = splice_targets(test_df, c1532, {
        'eea': c1494, 'egb': c1446, 'egc': c1530, 'ei': c1494, 'eps': c355_eps_035,
        'nc': c1506, 'tg': c355_tg_035,
    })
    c355_eps_002 = blend_targets(test_df, c1535, {'eps': (0.002, c1530)})
    c1570 = c1570_physics_grid_build_c1570_physics_grid(train_df, test_df, c355_eps_002, egb_pull=0.05, gap_pull=0.02, epsnc_pull=0.0)
    c1572 = splice_targets(test_df, c355_eps_002, {'egc': c1570})

    # ---- Base compound: C1572 candidate spine ----
    base_target = c1572['target'].to_numpy(float).copy()

    # ---- char arm: per-target Ridge on character n-grams of C282 OOF residuals ----
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import Ridge as RidgeModel
    from sklearn.pipeline import make_pipeline as make_pipeline_model
    from sklearn.model_selection import KFold as KFoldModel
    oof = c282_oof_df[['canonical', 'target_type', 'target', 'prediction']].dropna(subset=['target', 'prediction']).copy()
    resid = oof['target'].to_numpy(float) - oof['prediction'].to_numpy(float)
    tr_tt = oof['target_type'].to_numpy(object)
    tr_smiles = list(oof['canonical'])
    test_smiles = [canonicalize(s) for s in test_df['smiles']]
    medians = {}
    for t in ('ei', 'eea'):
        medians[t] = float(train_df.loc[train_df['target_type'] == t, 'target'].median())
    TARGETS_ORDER = ('tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps')
    char_delta = np.zeros(len(test_df))
    for t in TARGETS_ORDER:
        tm = tr_tt == t
        idx = np.where(tm)[0]
        y = resid[idx].copy()
        pred = np.zeros(len(test_smiles))
        kf = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        use_tfidf = (t == 'nc' or t == 'eps')
        for tr_f, va_f in kf.split(idx):
            if use_tfidf:
                vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False, sublinear_tf=True)
                alpha = 30.0
            else:
                vec = CountVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False)
                alpha = 40.0
            model = make_pipeline_model(vec, RidgeModel(alpha=alpha, solver='lsqr', max_iter=5000, tol=1e-4))
            model.fit([tr_smiles[i] for i in idx[tr_f]], y[tr_f])
            pred += model.predict(test_smiles) / 5
        mte = test_df['target_type'].to_numpy(object) == t
        char_delta[mte] = 0.20 * pred[mte]

    # ---- final: splice char targets (tg/egc/egb/nc/eps) and spread targets (ei/eea) ----
    final = base_target.copy()
    test_tt = test_df['target_type'].to_numpy(object)
    for t in TARGETS_ORDER:
        mte = test_tt == t
        if t in ('ei', 'eea'):
            tr_vals = train_df.loc[train_df['target_type'] == t, 'target'].to_numpy(float)
            lo = float(np.quantile(tr_vals, 0.001)) - float(np.std(tr_vals, ddof=1)) * 0.25
            hi = float(np.quantile(tr_vals, 0.999)) + float(np.std(tr_vals, ddof=1)) * 0.25
            med = float(np.median(tr_vals))
            spread = med + 1.05 * (base_target[mte] - med)
            final[mte] = np.clip(spread, lo, hi)
        else:
            final[mte] = base_target[mte] + char_delta[mte]

    # ---- P5A-125 arm: per-target value-aware calibration (Ridge on [base, base^2]) ----
    for _t in TARGETS_ORDER:
        _om = oof[oof['target_type'] == _t]
        _y = _om['target'].to_numpy(float)
        _p = _om['prediction'].to_numpy(float)
        if len(_y) < 30:
            continue
        _F = np.column_stack([_p, _p ** 2])
        _kfV = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        _oofV = np.zeros(len(_y))
        for _trV, _vaV in _kfV.split(_p):
            _mV = RidgeModel(alpha=5.0)
            _mV.fit(_F[_trV], _y[_trV])
            _oofV[_vaV] = _mV.predict(_F[_vaV])
        _d = _oofV - _p
        _aV = float(np.clip(np.sum((_y - _p) * _d) / max(np.sum(_d ** 2), 1e-12), 0.0, 1.0))
        _mV2 = RidgeModel(alpha=5.0)
        _mV2.fit(_F, _y)
        _mte = test_tt == _t
        _Ft = np.column_stack([base_target[_mte], base_target[_mte] ** 2])
        _cand = _mV2.predict(_Ft)
        final[_mte] += _aV * (_cand - base_target[_mte])
        print('P5A-125 value-calib %s alpha=%.4f' % (_t, _aV), flush=True)

    if not np.isfinite(final).all():
        raise RuntimeError('Non-finite predictions')
    submission = pd.DataFrame({'id': test_df['id'].to_numpy(int), 'target': final})
    if len(submission) != 4940 or submission['id'].duplicated().any():
        raise RuntimeError('Submission contract failed')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print('submission written:', out_path, flush=True)
    print('mean per-target final:', flush=True)
    for t in TARGETS_ORDER:
        vals = final[test_tt == t]
        print(f'  {t}: n={len(vals)} mean={float(vals.mean()):.6f} std={float(vals.std()):.6f}', flush=True)
    return submission


def main(argv=None):
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description='V57 standalone reproduction')
    parser.add_argument('--data-dir', default=None, help='directory containing official train.csv/test.csv/PI1M.csv')
    parser.add_argument('--out', default='submission.csv', help='output submission path')
    args = parser.parse_args(argv)
    data_dir = args.data_dir
    if data_dir is None:
        here = Path.cwd().resolve()
        for parent in (here, *here.parents):
            cand = parent / 'ppp-round-2'
            if (cand / 'train.csv').is_file() and (cand / 'test.csv').is_file():
                data_dir = str(cand)
                break
        if data_dir is None:
            raise RuntimeError('Could not locate official ppp-round-2 train/test files; pass --data-dir')
    run_v57(data_dir, args.out)
    print('done.', flush=True)


if __name__ == '__main__':
    main()