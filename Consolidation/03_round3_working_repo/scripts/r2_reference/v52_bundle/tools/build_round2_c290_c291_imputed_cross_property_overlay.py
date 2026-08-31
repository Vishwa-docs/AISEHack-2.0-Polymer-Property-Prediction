#!/usr/bin/env python3
"""C290/C291 current-only imputed cross-property overlays.

This experiment uses no archive labels and no local_eval values.  It first trains
current-only property imputers so every canonical train/test polymer has an
estimated vector of all seven properties.  It then trains target relation
models on official current train labels using imputed partner-property vectors.
Targets whose relation model beats the direct imputer OOF by a fixed gate are
conservatively overlaid onto a frozen branch base CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
FAST_LINEAR = False
DEFAULT_BASE = {
    "with_archive": "experiments/final_submission_runs/with_archive/R2-F25-IONIC-COTEST-OVERLAY-with_archive-20260807.csv",
    "without_archive": "experiments/final_submission_runs/without_archive/R2-F26-IONIC-COTEST-OVERLAY-without_archive-20260807.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(path: Path) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden input path: {path}")


def canonicalize(smiles: str) -> str:
    return reference.canonicalize(smiles)


def maccs_matrix(molecules: list[Any]) -> np.ndarray:
    rows = np.zeros((len(molecules), 167), dtype=np.float32)
    for i, mol in enumerate(molecules):
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows[i] = np.asarray([int(fp.GetBit(j)) for j in range(167)], dtype=np.float32)
    return rows


def sanitize(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(out) | (np.abs(out) > 1.0e12)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out


def grouped_oof(factory, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError("not enough groups")
    out = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x[tr], y[tr])
        out[va] = np.asarray(model.predict(x[va]), dtype=np.float64)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite oof")
    return out


def imputer_factory(target: str):
    if FAST_LINEAR:
        return lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=100.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        )
    large = target in {"tg", "egc"}

    def factory(seed: int):
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=80 if large else 120,
                min_samples_leaf=4 if large else 2,
                max_features=0.55 if large else 0.75,
                random_state=seed,
                n_jobs=4,
            ),
        )

    return factory


def relation_factories(target: str) -> dict[str, Any]:
    if FAST_LINEAR:
        return {
            "ridge10": lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "ridge100": lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
            "huber": lambda seed: make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000)),
        }
    small = target not in {"tg", "egc"}
    return {
        "ridge10": lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "ridge100": lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "huber": lambda seed: make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000)),
        "extra_trees": lambda seed: ExtraTreesRegressor(
            n_estimators=120 if small else 80,
            min_samples_leaf=3 if small else 10,
            max_features=1.0,
            random_state=seed,
            n_jobs=4,
        ),
        "hist_gbdt": lambda seed: HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=220 if small else 160,
            l2_regularization=0.10,
            max_leaf_nodes=15 if small else 31,
            min_samples_leaf=8 if small else 20,
            random_state=seed,
        ),
    }


def clip_like(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1.0e-8)
    return np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base IDs: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Invalid base values: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--overlay-weight", type=float, default=0.25)
    parser.add_argument("--gate-delta", type=float, default=0.005)
    parser.add_argument("--morgan-bits", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--fast-linear", action="store_true")
    args = parser.parse_args()
    global FAST_LINEAR
    FAST_LINEAR = bool(args.fast_linear)

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    base_path = Path(args.base_csv or DEFAULT_BASE[args.branch]).resolve()
    for path in (data_dir, output, manifest, base_path):
        guard(path)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")

    train, test, inputs = c282.load_current_only_inputs(data_dir)
    ids = test["id"].to_numpy(int)
    base = load_base(base_path, ids)
    train = train.copy()
    test = test.copy()
    train["canonical"] = train["smiles"].map(canonicalize)
    test["canonical"] = test["smiles"].map(canonicalize)
    train["target_type"] = train["target_type"].astype(str).str.lower()
    test["target_type"] = test["target_type"].astype(str).str.lower()

    keys = sorted(set(train["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: i for i, key in enumerate(keys)}
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    maccs = maccs_matrix(molecules)
    x_all = sanitize(np.hstack([descriptor, physical, morgan2, morgan3, maccs]))

    imputed_full = np.zeros((len(keys), len(TARGETS)), dtype=np.float64)
    imputed_train_like = np.zeros((len(keys), len(TARGETS)), dtype=np.float64)
    imputer_reports: dict[str, Any] = {}
    train_by_target = {}
    for ti, target in enumerate(TARGETS):
        rows = train[train["target_type"].eq(target)].reset_index(drop=True)
        train_by_target[target] = rows
        idx = np.asarray([key_to_index[c] for c in rows["canonical"]], dtype=int)
        y = rows["target"].to_numpy(float)
        groups = rows["canonical"].astype(str).to_numpy(object)
        factory = imputer_factory(target)
        oof = clip_like(y, grouped_oof(factory, x_all[idx], y, groups, int(args.seed) + ti * 17))
        model = factory(int(args.seed) + ti * 17 + 999)
        model.fit(x_all[idx], y)
        full = clip_like(y, model.predict(x_all))
        imputed_full[:, ti] = full
        imputed_train_like[:, ti] = full
        # For canonicals observed for this target, use OOF estimates in the
        # relation training matrix to avoid same-target leakage.
        oof_by_can = rows.assign(oof=oof).groupby("canonical")["oof"].mean().to_dict()
        for can, value in oof_by_can.items():
            imputed_train_like[key_to_index[can], ti] = float(value)
        imputer_reports[target] = {"rows": int(len(rows)), "direct_oof_r2": float(r2_score(y, oof))}

    result = base["target"].to_numpy(float).copy()
    overlay_reports: dict[str, Any] = {}
    for ti, target in enumerate(TARGETS):
        rows = train_by_target[target]
        idx = np.asarray([key_to_index[c] for c in rows["canonical"]], dtype=int)
        y = rows["target"].to_numpy(float)
        groups = rows["canonical"].astype(str).to_numpy(object)
        feature_cols = [j for j in range(len(TARGETS)) if j != ti]
        rel_x = imputed_train_like[idx][:, feature_cols]
        rel_test_keys = np.asarray([key_to_index[c] for c in test["canonical"]], dtype=int)
        rel_x_test_all = imputed_full[rel_test_keys][:, feature_cols]

        best = None
        model_reports = {}
        for name, factory in relation_factories(target).items():
            try:
                pred = clip_like(y, grouped_oof(factory, rel_x, y, groups, int(args.seed) + 1000 + ti * 29))
                score = float(r2_score(y, pred))
                model_reports[name] = {"oof_r2": score}
                if best is None or score > best[0]:
                    best = (score, name)
            except Exception as exc:
                model_reports[name] = {"error": repr(exc)}
        if best is None:
            overlay_reports[target] = {"selected": false, "reason": "all_relation_models_failed", "models": model_reports}
            continue
        direct_score = float(imputer_reports[target]["direct_oof_r2"])
        selected = bool(best[0] >= direct_score + float(args.gate_delta))
        overlay_reports[target] = {
            "selected": selected,
            "best_model": best[1],
            "best_relation_oof_r2": float(best[0]),
            "direct_imputer_oof_r2": direct_score,
            "gate_delta": float(args.gate_delta),
            "overlay_weight": float(args.overlay_weight),
            "models": model_reports,
        }
        if not selected:
            continue
        model = relation_factories(target)[best[1]](int(args.seed) + 5000 + ti)
        model.fit(rel_x, y)
        rel_pred = clip_like(y, model.predict(rel_x_test_all))
        mask = test["target_type"].eq(target).to_numpy()
        result[mask] = (1.0 - float(args.overlay_weight)) * result[mask] + float(args.overlay_weight) * rel_pred[mask]
        overlay_reports[target]["changed_rows"] = int(mask.sum())

    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite output")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.imputed-cross-property-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "official_only_imputers": True,
        "archive_labels_used_by_builder": False,
        "archive_labels_used_by_base": args.branch == "with_archive",
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": vars(args),
        "inputs": {**inputs, "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size}},
        "features": {
            "keys": int(len(keys)),
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "morgan_bits_each_radius": int(args.morgan_bits),
            "maccs_bits": 167,
            "dense_shape": list(x_all.shape),
        },
        "imputer_reports": imputer_reports,
        "overlay_reports": overlay_reports,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"], "selected": {k: v for k, v in overlay_reports.items() if v.get("selected")}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
