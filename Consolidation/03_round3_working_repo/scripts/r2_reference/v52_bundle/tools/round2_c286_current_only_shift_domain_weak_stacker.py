#!/usr/bin/env python3
"""C286 current-only weak-target stacker.

Branch: without_archive.

This is an local_eval-blind construction run. It reads:
- official current train/test and PI1M-derived official-input artifacts;
- existing current-only OOF/test prediction artifacts from C282/C284/C285;
- the F18 no-archive CSV only as the unchanged-target carrier.

It does not read archive labels, local_eval files, external_label files, public scores, or
Kaggle state. It learns targetwise nonnegative weights from nested grouped OOF
predictions for weak targets, then writes one frozen CSV. Post-hoc local_eval
scoring must happen in the separate local_eval-only scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("ei", "eea", "nc", "eps")

CONFIG: dict[str, Any] = {
    "seed": 20260807,
    "branch": "without_archive",
    "outer_folds": 5,
    "dense_abs_limit": 1.0e12,
    "min_nested_delta_over_best_single": 0.002,
    "use_domain_weighting": True,
    "base_candidate": "experiments/final_submission_runs/without_archive/R2-F18-FIXED-EQUAL-BLENDS-without_archive-20260807.csv",
    "c282_run": "experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1",
    "c284_run": "experiments/CLEAN_OFFICIAL_ONLY/R2-C284-20260807-current-only-pi1m-svd-reference-v1",
    "c285_run": "experiments/CLEAN_OFFICIAL_ONLY/R2-C285-20260807-current-only-pi1m-svd-weak-residual-v1",
    "dense_zoo_profile": "disabled_v3_artifact_only",
}


POLAR_SMARTS = {
    "CF": "[#6][F]",
    "CCl": "[#6][Cl]",
    "ester": "C(=O)O",
    "carbonyl": "[CX3]=[OX1]",
    "ether": "[OD2]([#6])[#6]",
    "OH": "[OX2H]",
    "nitrile": "C#N",
    "amide": "C(=O)N",
    "NH": "[NX3;H1,H2]",
    "sulfone": "S(=O)(=O)",
    "thioether": "[#16X2]",
    "aromatic_N": "n",
    "aromatic_O": "o",
    "aromatic_S": "s",
    "imide": "C(=O)NC(=O)",
    "siloxane": "[Si][O]",
    "phosphate": "P=O",
    "urethane": "NC(=O)O",
}
POLAR_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in POLAR_SMARTS.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def guard_input_path(path: Path) -> None:
    lowered = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in lowered:
            raise RuntimeError(f"Refusing forbidden input path containing {token!r}: {path}")


def progress(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "time": datetime.now().astimezone().isoformat(), **payload},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def load_csv(path: Path) -> pd.DataFrame:
    guard_input_path(path)
    return pd.read_csv(path)


def grouped_folds(groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    # Deterministic group order with randomized assignment, no label dependence.
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
        raise RuntimeError("Fold assignment failed")
    return folds


def sanitize_dense(x: np.ndarray, absolute_limit: float) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(out) | (np.abs(out) > absolute_limit)
    out[bad] = np.nan
    med = np.nanmedian(out, axis=0)
    med[~np.isfinite(med)] = 0.0
    rows, cols = np.where(~np.isfinite(out))
    out[rows, cols] = med[cols]
    return out


def weighted_r2(y: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> float:
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
    return 1.0 - ss_res / max(ss_tot, 1.0e-30)


def fit_nonnegative_blend(y: np.ndarray, base: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, float, str]:
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if w is None:
        w = np.ones(len(y), dtype=np.float64)
    w = np.maximum(np.asarray(w, dtype=np.float64), 1.0e-12)
    w = w / np.mean(w)
    y_mean = float(np.sum(w * y) / np.sum(w))
    x_mean = np.sum(base * w[:, None], axis=0) / np.sum(w)
    yw = (y - y_mean) * np.sqrt(w)
    xw = (base - x_mean) * np.sqrt(w[:, None])
    weights, _ = nnls(xw, yw)
    if float(np.sum(weights)) <= 1.0e-12:
        weights = np.full(base.shape[1], 1.0 / base.shape[1], dtype=np.float64)
    else:
        weights = weights / np.sum(weights)
    intercept = float(y_mean - np.dot(x_mean, weights))
    blend = base @ weights + intercept
    blend_score = weighted_r2(y, blend, w)
    single_scores = [weighted_r2(y, base[:, i], w) for i in range(base.shape[1])]
    best = int(np.argmax(single_scores))
    if single_scores[best] >= blend_score:
        weights = np.zeros(base.shape[1], dtype=np.float64)
        weights[best] = 1.0
        intercept = 0.0
        return weights, intercept, f"single_{best}"
    return weights, intercept, "nnls_blend"


def domain_weights(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    # Label-free test-domain weighting: each training row is weighted by its
    # inverse distance to the nearest test row in sanitized/scaled dense space.
    train = sanitize_dense(x_train, float(CONFIG["dense_abs_limit"]))
    test = sanitize_dense(x_test, float(CONFIG["dense_abs_limit"]))
    mu = np.mean(train, axis=0)
    sd = np.std(train, axis=0)
    sd[sd < 1.0e-8] = 1.0
    train = np.clip((train - mu) / sd, -8.0, 8.0)
    test = np.clip((test - mu) / sd, -8.0, 8.0)
    # Use chunks to avoid a large dense distance matrix.
    best = np.full(len(train), np.inf, dtype=np.float64)
    for start in range(0, len(test), 256):
        block = test[start : start + 256]
        dist2 = np.sum((train[:, None, :] - block[None, :, :]) ** 2, axis=2)
        best = np.minimum(best, np.min(dist2, axis=1))
    scale = float(np.median(best[np.isfinite(best)]))
    scale = max(scale, 1.0e-8)
    weights = np.exp(-best / scale)
    weights = np.clip(weights, 0.2, 5.0)
    return weights / np.mean(weights)


def polar_block(canonicals: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for canonical in canonicals:
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse canonical SMILES: {canonical}")
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        row = [len(mol.GetSubstructMatches(pattern)) / heavy for pattern in POLAR_PATTERNS.values()]
        row.extend(
            [
                Descriptors.TPSA(mol) / heavy,
                Descriptors.NumHDonors(mol) / heavy,
                Descriptors.NumHAcceptors(mol) / heavy,
                Descriptors.FractionCSP3(mol),
                Descriptors.NumRotatableBonds(mol) / heavy,
                Crippen.MolMR(mol) / heavy,
                Crippen.MolLogP(mol) / heavy,
                rdMolDescriptors.CalcNumAromaticRings(mol) / heavy,
            ]
        )
        rows.append([float(v) if math.isfinite(float(v)) else 0.0 for v in row])
    return np.asarray(rows, dtype=np.float64)


def physics_eps_nc_arms(
    *,
    pooled: pd.DataFrame,
    test: pd.DataFrame,
    c282_oof: pd.DataFrame,
    c282_test: pd.DataFrame,
    progress_path: Path,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Generate fold-safe F02-style physics OOF/test arms for eps and nc."""
    oof_out: dict[str, pd.Series] = {}
    test_out: dict[str, pd.Series] = {}
    wide = pooled.pivot(index="canonical", columns="target_type", values="target")
    pair = wide[wide.get("eps").notna() & wide.get("nc").notna()]
    if len(pair) < 20:
        return oof_out, test_out
    pair_cans = pair.index.astype(str).tolist()
    pair_ionic = (pair["eps"].to_numpy(float) - np.square(pair["nc"].to_numpy(float))).astype(np.float64)

    # Partner full-fit models for test missing-partner fills.
    all_need = list(dict.fromkeys(pooled["canonical"].astype(str).tolist() + test["canonical"].astype(str).tolist()))
    all_features = polar_block(all_need)
    all_index = {can: i for i, can in enumerate(all_need)}
    partner_full: dict[str, dict[str, float]] = {}
    for partner in ("eps", "nc"):
        rows = wide[partner].dropna()
        cans = rows.index.astype(str).tolist()
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=int(CONFIG["seed"]), n_jobs=4)
        model.fit(all_features[[all_index[c] for c in cans]], rows.to_numpy(float))
        pred = model.predict(all_features)
        partner_full[partner] = dict(zip(all_need, pred, strict=True))
        partner_full[partner].update({str(c): float(v) for c, v in rows.items()})

    ionic_full = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=int(CONFIG["seed"]) + 41, n_jobs=4)
    ionic_full.fit(all_features[[all_index[c] for c in pair_cans]], pair_ionic)
    ionic_pred_full = dict(zip(all_need, ionic_full.predict(all_features), strict=True))

    for target, partner in (("eps", "nc"), ("nc", "eps")):
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        cans = target_rows["canonical"].astype(str).to_numpy()
        y = target_rows["target"].to_numpy(float)
        c282_rows = c282_oof[c282_oof["target_type"] == target].reset_index(drop=True)
        if len(c282_rows) != len(target_rows) or not np.all(c282_rows["canonical"].astype(str).to_numpy() == cans):
            raise RuntimeError(f"C282 OOF alignment failed for {target}")
        b0 = c282_rows["prediction"].to_numpy(float)
        folds = grouped_folds(cans, int(CONFIG["outer_folds"]), int(CONFIG["seed"]) + 901)
        arm = np.empty(len(y), dtype=np.float64)
        for fold in range(int(CONFIG["outer_folds"])):
            validation = folds == fold
            va_cans = set(cans[validation])
            partner_rows = wide[partner].dropna()
            partner_train = [str(c) for c in partner_rows.index.astype(str).tolist() if str(c) not in va_cans]
            partner_model = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2, random_state=int(CONFIG["seed"]) + fold, n_jobs=4)
            partner_model.fit(all_features[[all_index[c] for c in partner_train]], np.array([float(partner_rows.loc[c]) for c in partner_train]))
            partner_pred = dict(
                zip(
                    list(cans[validation]),
                    partner_model.predict(all_features[[all_index[c] for c in cans[validation]]]),
                    strict=True,
                )
            )
            for c, v in partner_rows.items():
                partner_pred[str(c)] = float(v)

            ionic_train = [c for c in pair_cans if c not in va_cans]
            ionic_model = ExtraTreesRegressor(n_estimators=360, min_samples_leaf=2, random_state=int(CONFIG["seed"]) + 31 + fold, n_jobs=4)
            ionic_model.fit(
                all_features[[all_index[c] for c in ionic_train]],
                np.array([float(pair.loc[c, "eps"] - pair.loc[c, "nc"] ** 2) for c in ionic_train]),
            )
            ion = dict(
                zip(
                    list(cans[validation]),
                    ionic_model.predict(all_features[[all_index[c] for c in cans[validation]]]),
                    strict=True,
                )
            )
            for local_idx, can in zip(np.flatnonzero(validation), cans[validation], strict=True):
                has_partner = can in wide.index and partner in wide.columns and pd.notna(wide.loc[can, partner])
                pval = float(wide.loc[can, partner]) if has_partner else float(partner_pred[can])
                ionic = max(float(ion[can]), 0.02)
                if target == "eps":
                    phys = pval * pval + ionic
                else:
                    phys = math.sqrt(max(pval - ionic, 1.0))
                arm[local_idx] = phys if has_partner else 0.5 * phys + 0.5 * b0[local_idx]
        oof_out[f"physics_{target}_b2"] = pd.Series(arm, index=target_rows["canonical"].astype(str))

        te = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        te_cans = te["canonical"].astype(str).tolist()
        c282_te = c282_test[c282_test["target_type"] == target].sort_values("id").reset_index(drop=True)
        b0_test = c282_te["model_prediction"].to_numpy(float)
        values = np.empty(len(te), dtype=np.float64)
        for i, can in enumerate(te_cans):
            has_partner = can in wide.index and partner in wide.columns and pd.notna(wide.loc[can, partner])
            pval = float(wide.loc[can, partner]) if has_partner else float(partner_full[partner][can])
            ionic = max(float(ionic_pred_full[can]), 0.02)
            if target == "eps":
                phys = pval * pval + ionic
            else:
                phys = math.sqrt(max(pval - ionic, 1.0))
            values[i] = phys if has_partner else 0.5 * phys + 0.5 * b0_test[i]
        test_out[f"physics_{target}_b2"] = pd.Series(values, index=te["id"].astype(int))
        progress(progress_path, "physics_arm_ready", target=target, rows=int(len(target_rows)), test_rows=int(len(te)))
    return oof_out, test_out


def dense_zoo_arms(
    *,
    target: str,
    pooled: pd.DataFrame,
    test: pd.DataFrame,
    keys: list[str],
    key_to_index: dict[str, int],
    dense_base: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
    target_test = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
    train_index = np.array([key_to_index[str(c)] for c in target_rows["canonical"].astype(str)], dtype=np.int64)
    test_index = np.array([key_to_index[str(c)] for c in target_test["canonical"].astype(str)], dtype=np.int64)
    y = target_rows["target"].to_numpy(float)
    dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
    _, _, x_train, x_test = reference.fit_dense_preprocessor(dense, train_index, test_index, float(CONFIG["dense_abs_limit"]))
    groups = target_rows["canonical"].astype(str).to_numpy()
    folds = grouped_folds(groups, int(CONFIG["outer_folds"]), int(CONFIG["seed"]) + 113)
    seed = int(CONFIG["seed"]) + 97 * TARGETS.index(target)
    model_factories = {
        "dense_ridge_a5": lambda s: Ridge(alpha=5.0),
        "dense_extra_trees_reduced": lambda s: ExtraTreesRegressor(n_estimators=80, min_samples_leaf=1, max_features=0.75, random_state=s, n_jobs=2),
        "dense_hgb_reduced": lambda s: HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=0.05, random_state=s),
    }
    oof: dict[str, np.ndarray] = {name: np.empty(len(y), dtype=np.float64) for name in model_factories}
    test_values: dict[str, np.ndarray] = {}
    for arm, factory in model_factories.items():
        for fold in range(int(CONFIG["outer_folds"])):
            validation = folds == fold
            training = ~validation
            model = factory(seed + fold)
            model.fit(x_train[training], y[training])
            oof[arm][validation] = reference.clip_prediction(y[training], model.predict(x_train[validation]))
        model = factory(seed + 99)
        model.fit(x_train, y)
        test_values[arm] = reference.clip_prediction(y, model.predict(x_test))
    stack_oof = np.vstack([oof[name] for name in model_factories])
    stack_test = np.vstack([test_values[name] for name in model_factories])
    oof["dense_mean3"] = np.mean(stack_oof, axis=0)
    oof["dense_median3"] = np.median(stack_oof, axis=0)
    test_values["dense_mean3"] = np.mean(stack_test, axis=0)
    test_values["dense_median3"] = np.median(stack_test, axis=0)
    oof_series = {name: pd.Series(values, index=target_rows["canonical"].astype(str)) for name, values in oof.items()}
    test_series = {name: pd.Series(values, index=target_test["id"].astype(int)) for name, values in test_values.items()}
    return oof_series, test_series


def add_reference_artifact_arms(
    target: str,
    *,
    oof_arms: dict[str, pd.Series],
    test_arms: dict[str, pd.Series],
    c282_oof: pd.DataFrame,
    c282_test: pd.DataFrame,
    c284_oof: pd.DataFrame,
    c284_test: pd.DataFrame,
    c285_oof: pd.DataFrame,
    c285_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_oof = c282_oof[c282_oof["target_type"] == target].reset_index(drop=True)
    target_test = c282_test[c282_test["target_type"] == target].sort_values("id").reset_index(drop=True)
    canonical_index = target_oof["canonical"].astype(str)
    id_index = target_test["id"].astype(int)
    y_frame = pd.DataFrame({"canonical": canonical_index, "target": target_oof["target"].to_numpy(float)})

    for prefix, oo, tt in (("c282", c282_oof, c282_test), ("c284", c284_oof, c284_test)):
        oo_t = oo[oo["target_type"] == target].reset_index(drop=True)
        tt_t = tt[tt["target_type"] == target].sort_values("id").reset_index(drop=True)
        if len(oo_t) != len(target_oof) or not np.all(oo_t["canonical"].astype(str).to_numpy() == canonical_index.to_numpy()):
            raise RuntimeError(f"{prefix} OOF alignment failed for {target}")
        if len(tt_t) != len(target_test) or not np.all(tt_t["id"].astype(int).to_numpy() == id_index.to_numpy()):
            raise RuntimeError(f"{prefix} test alignment failed for {target}")
        for col in ("prediction", "sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local"):
            oof_arms[f"{prefix}_{col}"] = pd.Series(oo_t[col].to_numpy(float), index=canonical_index)
            test_col = "model_prediction" if col == "prediction" else col
            test_arms[f"{prefix}_{col}"] = pd.Series(tt_t[test_col].to_numpy(float), index=id_index)

    c285_o = c285_oof[c285_oof["target_type"] == target].reset_index(drop=True)
    if len(c285_o) == len(target_oof) and "canonical" in c285_o.columns and c285_o["canonical"].notna().all():
        # C285 is current-only and has the same pooled canonical ordering.
        if not np.all(c285_o["canonical"].astype(str).to_numpy() == canonical_index.to_numpy()):
            raise RuntimeError(f"C285 OOF alignment failed for {target}")
        for col in ("parent", "candidate"):
            oof_arms[f"c285_{col}"] = pd.Series(c285_o[col].to_numpy(float), index=canonical_index)
    c285_t = c285_test[c285_test["target_type"] == target].sort_values("id").reset_index(drop=True)
    if len(c285_t):
        if not np.all(c285_t["id"].astype(int).to_numpy() == id_index.to_numpy()):
            raise RuntimeError(f"C285 test alignment failed for {target}")
        for col in ("parent", "candidate"):
            test_arms[f"c285_{col}"] = pd.Series(c285_t[col].to_numpy(float), index=id_index)
    return y_frame, pd.DataFrame({"id": id_index})


def nested_stack_for_target(
    *,
    target: str,
    y_frame: pd.DataFrame,
    id_frame: pd.DataFrame,
    oof_arms: dict[str, pd.Series],
    test_arms: dict[str, pd.Series],
    x_train_domain: np.ndarray,
    x_test_domain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    names = sorted(set(oof_arms) & set(test_arms))
    if len(names) < 2:
        raise RuntimeError(f"Too few aligned arms for {target}: {names}")
    y = y_frame["target"].to_numpy(float)
    groups = y_frame["canonical"].astype(str).to_numpy()
    oof_matrix = np.column_stack([oof_arms[name].loc[y_frame["canonical"].astype(str)].to_numpy(float) for name in names])
    test_matrix = np.column_stack([test_arms[name].loc[id_frame["id"].astype(int)].to_numpy(float) for name in names])
    if not np.isfinite(oof_matrix).all() or not np.isfinite(test_matrix).all():
        raise RuntimeError(f"Non-finite stack matrix for {target}")
    row_weights = domain_weights(x_train_domain, x_test_domain) if CONFIG["use_domain_weighting"] else np.ones(len(y))
    folds = grouped_folds(groups, int(CONFIG["outer_folds"]), int(CONFIG["seed"]) + 707)
    nested = np.empty(len(y), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold in range(int(CONFIG["outer_folds"])):
        validation = folds == fold
        training = ~validation
        weights, intercept, mode = fit_nonnegative_blend(y[training], oof_matrix[training], row_weights[training])
        nested[validation] = oof_matrix[validation] @ weights + intercept
        fold_records.append(
            {
                "fold": int(fold),
                "rows": int(np.sum(validation)),
                "mode": mode,
                "r2": float(r2_score(y[validation], nested[validation])),
                "best_single_r2": float(max(r2_score(y[validation], oof_matrix[validation, i]) for i in range(len(names)))),
            }
        )
    final_weights, final_intercept, final_mode = fit_nonnegative_blend(y, oof_matrix, row_weights)
    final_test = test_matrix @ final_weights + final_intercept
    single_scores = {name: float(r2_score(y, oof_matrix[:, i])) for i, name in enumerate(names)}
    best_name = max(single_scores, key=single_scores.get)
    nested_r2 = float(r2_score(y, nested))
    best_single_r2 = float(single_scores[best_name])
    full_weighted_r2 = weighted_r2(y, oof_matrix @ final_weights + final_intercept, row_weights)
    selected = bool(nested_r2 >= best_single_r2 + float(CONFIG["min_nested_delta_over_best_single"]))
    if not selected:
        final_weights = np.zeros(len(names), dtype=np.float64)
        final_weights[names.index(best_name)] = 1.0
        final_intercept = 0.0
        final_mode = f"fallback_best_single_{best_name}"
        final_test = test_matrix[:, names.index(best_name)]
    report = {
        "target": target,
        "arms": names,
        "single_oof_r2": single_scores,
        "best_single": best_name,
        "best_single_r2": best_single_r2,
        "nested_stack_r2": nested_r2,
        "nested_delta_over_best_single": nested_r2 - best_single_r2,
        "full_weighted_fit_r2": full_weighted_r2,
        "selected_nested_stack": selected,
        "final_mode": final_mode,
        "final_intercept": float(final_intercept),
        "final_weights": {name: float(final_weights[i]) for i, name in enumerate(names) if abs(float(final_weights[i])) > 1.0e-12},
        "folds": fold_records,
    }
    return nested, final_test, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse run directory: {run_dir}")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {output}")
    run_dir.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    progress(progress_path, "started", experiment_id=run_dir.name)

    train, test, inputs = c282.load_current_only_inputs(data_dir)
    archive_path = data_dir / "archive" / "train.csv"
    if archive_path.exists():
        # Do not read it; just record that no archive path was opened.
        pass
    base_path = Path(CONFIG["base_candidate"]).resolve()
    base = load_csv(base_path)
    if list(base.columns) != ["id", "target"] or len(base) != 4940:
        raise RuntimeError("Base F18 candidate invalid")
    if base["id"].duplicated().any() or not np.array_equal(base["id"].to_numpy(int), test["id"].to_numpy(int)):
        raise RuntimeError("Base F18 ID validation failed")

    c282_dir = Path(CONFIG["c282_run"]).resolve()
    c284_dir = Path(CONFIG["c284_run"]).resolve()
    c285_dir = Path(CONFIG["c285_run"]).resolve()
    c282_oof = load_csv(c282_dir / "oof_predictions.csv")
    c282_test = load_csv(c282_dir / "test_predictions_detail.csv")
    c284_oof = load_csv(c284_dir / "oof_predictions.csv")
    c284_test = load_csv(c284_dir / "test_predictions_detail.csv")
    c285_oof = load_csv(c285_dir / "oof_predictions.csv")
    c285_test = load_csv(c285_dir / "component_predictions.csv")

    raw_labels, pooled = reference.build_label_pool(train, train.iloc[0:0].copy())
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    progress(progress_path, "base_features_ready", keys=len(keys), dense_features=int(dense_base.shape[1]))

    physics_oof, physics_test = physics_eps_nc_arms(
        pooled=pooled,
        test=test,
        c282_oof=c282_oof,
        c282_test=c282_test,
        progress_path=progress_path,
    )

    result = base.copy()
    target_reports: dict[str, Any] = {}
    oof_output_records: list[pd.DataFrame] = []
    for target in ACTIVE_TARGETS:
        progress(progress_path, "target_started", target=target)
        oof_arms: dict[str, pd.Series] = {}
        test_arms: dict[str, pd.Series] = {}
        y_frame, id_frame = add_reference_artifact_arms(
            target,
            oof_arms=oof_arms,
            test_arms=test_arms,
            c282_oof=c282_oof,
            c282_test=c282_test,
            c284_oof=c284_oof,
            c284_test=c284_test,
            c285_oof=c285_oof,
            c285_test=c285_test,
        )
        if CONFIG["dense_zoo_profile"] != "disabled_v3_artifact_only":
            dense_oof, dense_test = dense_zoo_arms(
                target=target,
                pooled=pooled,
                test=test,
                keys=keys,
                key_to_index=key_to_index,
                dense_base=dense_base,
                cross_values=cross_values,
                cross_available=cross_available,
            )
            oof_arms.update(dense_oof)
            test_arms.update(dense_test)
        if target in ("eps", "nc"):
            for name, values in physics_oof.items():
                if name.startswith(f"physics_{target}_"):
                    oof_arms[name] = values
            for name, values in physics_test.items():
                if name.startswith(f"physics_{target}_"):
                    test_arms[name] = values
        train_index = np.array([key_to_index[str(c)] for c in y_frame["canonical"].astype(str)], dtype=np.int64)
        test_subset = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_index = np.array([key_to_index[str(c)] for c in test_subset["canonical"].astype(str)], dtype=np.int64)
        dense_target = reference.target_dense_features(dense_base, cross_values, cross_available, target)
        nested, final_test, report = nested_stack_for_target(
            target=target,
            y_frame=y_frame,
            id_frame=id_frame,
            oof_arms=oof_arms,
            test_arms=test_arms,
            x_train_domain=dense_target[train_index],
            x_test_domain=dense_target[test_index],
        )
        ids = id_frame["id"].to_numpy(int)
        result.loc[result["id"].astype(int).isin(ids), "target"] = pd.Series(final_test, index=ids).loc[
            result.loc[result["id"].astype(int).isin(ids), "id"].astype(int)
        ].to_numpy(float)
        target_reports[target] = report
        oof_output_records.append(
            pd.DataFrame(
                {
                    "canonical": y_frame["canonical"].astype(str),
                    "target_type": target,
                    "target": y_frame["target"].to_numpy(float),
                    "nested_prediction": nested,
                }
            )
        )
        progress(
            progress_path,
            "target_complete",
            target=target,
            nested_r2=report["nested_stack_r2"],
            best_single_r2=report["best_single_r2"],
            final_mode=report["final_mode"],
        )

    if len(result) != 4940 or result["id"].duplicated().any() or not np.array_equal(result["id"].to_numpy(int), test["id"].to_numpy(int)):
        raise RuntimeError("Final result ID/schema validation failed")
    if not np.isfinite(result["target"].to_numpy(float)).all():
        raise RuntimeError("Final result contains non-finite predictions")
    result.to_csv(output, index=False)

    oof_path = run_dir / "nested_oof_predictions.csv"
    pd.concat(oof_output_records, ignore_index=True).to_csv(oof_path, index=False)
    report = {
        "schema_version": "ppp.round2.c286.current-only-shift-domain-weak-stacker.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "external_label_file_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "pretrained_weights": False,
        "prior_predictions_used_as_training_features": False,
        "branch": "without_archive",
        "config": CONFIG,
        "inputs": inputs
        | {
            "base_candidate": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
            "c282_oof": {"path": str(c282_dir / "oof_predictions.csv"), "sha256": sha256_file(c282_dir / "oof_predictions.csv")},
            "c282_test_detail": {"path": str(c282_dir / "test_predictions_detail.csv"), "sha256": sha256_file(c282_dir / "test_predictions_detail.csv")},
            "c284_oof": {"path": str(c284_dir / "oof_predictions.csv"), "sha256": sha256_file(c284_dir / "oof_predictions.csv")},
            "c284_test_detail": {"path": str(c284_dir / "test_predictions_detail.csv"), "sha256": sha256_file(c284_dir / "test_predictions_detail.csv")},
            "c285_oof": {"path": str(c285_dir / "oof_predictions.csv"), "sha256": sha256_file(c285_dir / "oof_predictions.csv")},
            "c285_component_predictions": {"path": str(c285_dir / "component_predictions.csv"), "sha256": sha256_file(c285_dir / "component_predictions.csv")},
        },
        "features": {
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "physics_eps_nc_arm": True,
            "dense_zoo_arms": True,
            "domain_weighting": bool(CONFIG["use_domain_weighting"]),
        },
        "target_reports": target_reports,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(result)), "bytes": output.stat().st_size},
        "oof_output": {"path": str(oof_path), "sha256": sha256_file(oof_path), "rows": int(sum(len(x) for x in oof_output_records))},
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", CONFIG)
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"sklearn={__import__('sklearn').__version__}",
                f"rdkit={Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    manifest_lines.append(f"{sha256_file(output)}  OUTPUT {output}")
    manifest_lines.append(f"{sha256_file(Path(__file__))}  SOURCE tools/round2_c286_current_only_shift_domain_weak_stacker.py")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nC286 completed as a current-only, local_eval-blind weak-target stacker. It writes one frozen CSV for post-freeze local_eval scoring. No archive labels, local_eval files, Kaggle compute, upload, or submission were used.\n",
        encoding="utf-8",
    )
    progress(progress_path, "completed", output=str(output), elapsed_seconds=report["elapsed_seconds"])
    print(json.dumps({"experiment_id": run_dir.name, "output": report["output"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
