#!/usr/bin/env python3
"""C1385 no-archive EI OOF-signed residual guard.

Purpose: test whether the anti-C1369 EI direction can be justified without
local_eval weight selection.  The script uses official current train/test only,
builds a C1369-style direct EI model with grouped OOF predictions, selects a
signed blend weight through nested OOF against a current-reference identity
parent, and only writes a full candidate when the EI OOF gate passes.

No local_eval, external_label file, archive labels, Kaggle action, prior model weights, or
nonofficial data are read by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
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

import build_round2_c385_archive_weak_target_model_zoo as c385
import build_round2_c1369_branch_fast_direct_stack as c1369
import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as graph
import round2_c282_current_only_reference as c282


SCHEMA = "ppp.round2.c1385.noarchive-ei-oof-signed-residual-guard.v1"
SEED = 20260808
TARGET = "ei"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def guard_path(path: Path, *, role: str, branch_required: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "/with_archive/" in low or "/archive/" in low:
        raise RuntimeError(f"Refusing archive/cross-branch {role} path: {path}")
    if branch_required and "/without_archive/" not in low:
        raise RuntimeError(f"{role} must be under /without_archive/: {path}")


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path, role="base candidate", branch_required=True)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"]:
        raise RuntimeError(f"Unexpected base schema: {path}")
    if len(frame) != len(ids) or frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(np.int64), ids):
        raise RuntimeError("Base candidate IDs/order do not match official test")
    if not np.isfinite(frame["target"].to_numpy(np.float64)).all():
        raise RuntimeError("Base candidate contains non-finite values")
    return frame


def ridge_factory(alpha: float):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1.0e-4),
    )


def grouped_oof_direct(x: np.ndarray, y: np.ndarray, groups: np.ndarray, progress_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    factories = {
        "ridge_20": lambda: ridge_factory(20.0),
        "ridge_80": lambda: ridge_factory(80.0),
        "ridge_250": lambda: ridge_factory(250.0),
    }
    columns: list[np.ndarray] = []
    reports: dict[str, Any] = {}
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for name, factory in factories.items():
        append_jsonl(progress_path, {"stage": "direct_model_started", "model": name})
        oof = np.full(len(y), np.nan, dtype=np.float64)
        folds: list[dict[str, Any]] = []
        for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
            model = factory()
            model.fit(x[tr], y[tr])
            pred = reference.clip_prediction(y[tr], np.asarray(model.predict(x[va]), dtype=np.float64))
            oof[va] = pred
            folds.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Non-finite OOF for {name}")
        columns.append(oof)
        reports[name] = {"oof_r2": float(r2_score(y, oof)), "folds": folds}
        append_jsonl(progress_path, {"stage": "direct_model_finished", "model": name, "oof_r2": reports[name]["oof_r2"]})
    stack = np.column_stack(columns)
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, stack)
    direct_oof = reference.clip_prediction(y, stack @ weights + intercept)
    reports["blend"] = {
        "name": blend_name,
        "weights": {name: float(value) for name, value in zip(factories.keys(), weights, strict=True)},
        "intercept": float(intercept),
        "oof_r2": float(r2_score(y, direct_oof)),
        "blend_oof_r2": float(blend_r2),
    }
    return direct_oof, reports


def direct_test_predictions(x_train: np.ndarray, y: np.ndarray, x_test: np.ndarray, blend_report: dict[str, Any]) -> np.ndarray:
    preds: list[np.ndarray] = []
    for name in ("ridge_20", "ridge_80", "ridge_250"):
        alpha = float(name.split("_", 1)[1])
        model = ridge_factory(alpha)
        model.fit(x_train, y)
        preds.append(reference.clip_prediction(y, np.asarray(model.predict(x_test), dtype=np.float64)))
    stack = np.column_stack(preds)
    weights = np.asarray([blend_report["weights"][name] for name in ("ridge_20", "ridge_80", "ridge_250")], dtype=np.float64)
    pred = stack @ weights + float(blend_report["intercept"])
    return reference.clip_prediction(y, pred)


def identity_parent_oof(train: pd.DataFrame, c282_oof: pd.DataFrame, weight: float) -> pd.DataFrame:
    if list(c282_oof.columns)[:4] != ["canonical", "target_type", "target", "prediction"]:
        raise RuntimeError("Unexpected C282 OOF schema")
    ei = c282_oof.loc[c282_oof["target_type"].astype(str).eq(TARGET)].copy().reset_index(drop=True)
    wide = train.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="mean")
    parent = ei["prediction"].to_numpy(np.float64).copy()
    applied = 0
    support: dict[str, int] = {}
    for pos, canon in enumerate(ei["canonical"].astype(str)):
        if canon in wide.index and "eea" in wide.columns and "egc" in wide.columns:
            eea = wide.at[canon, "eea"]
            egc = wide.at[canon, "egc"]
            if pd.notna(eea) and pd.notna(egc):
                raw = float(eea) + float(egc)
                parent[pos] = (1.0 - weight) * parent[pos] + weight * raw
                applied += 1
                support["eea_current|egc_current"] = support.get("eea_current|egc_current", 0) + 1
    ei["parent_prediction"] = parent
    ei.attrs["identity_applied"] = applied
    ei.attrs["identity_support"] = support
    return ei


def nested_signed_weight(y: np.ndarray, parent: np.ndarray, source: np.ndarray, groups: np.ndarray, weights: tuple[float, ...]) -> dict[str, Any]:
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    nested = np.full(len(y), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for fold, (tr, va) in enumerate(splitter.split(np.arange(len(y)), y, groups=groups)):
        train_scores = []
        for weight in weights:
            pred_tr = (1.0 - weight) * parent[tr] + weight * source[tr]
            train_scores.append((float(r2_score(y[tr], pred_tr)), float(weight)))
        _, chosen = max(train_scores, key=lambda item: item[0])
        pred_va = (1.0 - chosen) * parent[va] + chosen * source[va]
        nested[va] = pred_va
        parent_r2 = float(r2_score(y[va], parent[va]))
        candidate_r2 = float(r2_score(y[va], pred_va))
        folds.append({"fold": int(fold), "rows": int(len(va)), "chosen_weight": float(chosen), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2})
    if not np.isfinite(nested).all():
        raise RuntimeError("Nested signed predictions are non-finite")
    full_scores = []
    for weight in weights:
        pred = (1.0 - weight) * parent + weight * source
        full_scores.append((float(r2_score(y, pred)), float(weight)))
    best_full_r2, best_full_weight = max(full_scores, key=lambda item: item[0])
    parent_r2 = float(r2_score(y, parent))
    nested_r2 = float(r2_score(y, nested))
    deltas = [row["delta_r2"] for row in folds]
    return {
        "parent_oof_r2": parent_r2,
        "direct_source_oof_r2": float(r2_score(y, source)),
        "nested_signed_oof_r2": nested_r2,
        "nested_delta_r2": nested_r2 - parent_r2,
        "best_full_weight": float(best_full_weight),
        "best_full_weight_oof_r2": float(best_full_r2),
        "best_full_weight_delta_r2": float(best_full_r2 - parent_r2),
        "positive_folds": int(sum(value > 0 for value in deltas)),
        "minimum_fold_delta": float(min(deltas)),
        "folds": folds,
        "weight_grid": list(weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--c282-oof", default="experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1/oof_predictions.csv")
    parser.add_argument("--identity-weight", type=float, default=0.075)
    parser.add_argument("--morgan-bits", type=int, default=256)
    parser.add_argument("--weights", default="-0.5,-0.4,-0.3,-0.25,-0.2,-0.175,-0.165,-0.15,-0.125,-0.1,-0.075,-0.05,-0.025,0,0.025,0.05")
    parser.add_argument("--min-nested-delta", type=float, default=0.003)
    parser.add_argument("--min-positive-folds", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    started = time.time()
    round2_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (round2_root / data_dir).resolve()
    base_path = Path(args.base_csv)
    if not base_path.is_absolute():
        base_path = (round2_root / base_path).resolve()
    c282_oof_path = Path(args.c282_oof)
    if not c282_oof_path.is_absolute():
        c282_oof_path = (round2_root / c282_oof_path).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (round2_root / output).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (round2_root / run_dir).resolve()
    for path, role, branch_required in (
        (base_path, "base candidate", True),
        (c282_oof_path, "C282 OOF", False),
        (output, "output", True),
        (run_dir, "run dir", False),
    ):
        guard_path(path, role=role, branch_required=branch_required)
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    append_jsonl(progress_path, {"stage": "started", "created_at": datetime.now().astimezone().isoformat()})

    train, test, inputs = c282.load_current_only_inputs(data_dir)
    archive = train.iloc[0:0].copy()
    _, pooled = reference.build_label_pool(train, archive)
    parent = {
        "train": train,
        "test": test,
        "archive": archive,
        "pooled": pooled,
        "keys": sorted(set(pooled["canonical"]) | set(test["canonical"])),
    }
    parent["key_to_index"] = {key: index for index, key in enumerate(parent["keys"])}
    parent["molecules"] = reference.build_molecules(parent["keys"])
    ids = test["id"].to_numpy(np.int64)
    base = load_base(base_path, ids)
    append_jsonl(progress_path, {"stage": "inputs_ready", "train_rows": int(len(train)), "keys": int(len(parent["keys"]))})

    features, feature_report = c1369.feature_matrix(parent, int(args.morgan_bits))
    cross_values, cross_available = reference.cross_property_arrays(pooled, parent["keys"])
    dense = c385.sanitize_dense(reference.target_dense_features(features, cross_values, cross_available, TARGET))
    target_train = pooled.loc[pooled["target_type"].astype(str).eq(TARGET)].reset_index(drop=True)
    target_test = test.loc[test["target_type"].astype(str).eq(TARGET)].reset_index(drop=True)
    train_idx = np.asarray([parent["key_to_index"][value] for value in target_train["canonical"]], dtype=np.int64)
    test_idx = np.asarray([parent["key_to_index"][value] for value in target_test["canonical"]], dtype=np.int64)
    x_train = dense[train_idx]
    x_test = dense[test_idx]
    y = target_train["target"].to_numpy(np.float64)
    groups = np.asarray([graph.no_stereo(value) for value in target_train["canonical"].astype(str)], dtype=object)
    append_jsonl(progress_path, {"stage": "features_ready", "shape": feature_report["shape"], "ei_rows": int(len(y))})

    c282_oof = pd.read_csv(c282_oof_path)
    parent_oof_frame = identity_parent_oof(train, c282_oof, float(args.identity_weight))
    if not np.array_equal(parent_oof_frame["canonical"].astype(str).to_numpy(object), target_train["canonical"].astype(str).to_numpy(object)):
        raise RuntimeError("EI OOF canonical alignment failed")
    if not np.allclose(parent_oof_frame["target"].to_numpy(np.float64), y):
        raise RuntimeError("EI OOF target alignment failed")
    parent_oof = parent_oof_frame["parent_prediction"].to_numpy(np.float64)

    direct_oof, direct_report = grouped_oof_direct(x_train, y, groups, progress_path)
    weights = tuple(float(item.strip()) for item in args.weights.split(",") if item.strip())
    signed = nested_signed_weight(y, parent_oof, direct_oof, groups, weights)
    gate = bool(signed["nested_delta_r2"] >= float(args.min_nested_delta) and signed["positive_folds"] >= int(args.min_positive_folds) and signed["minimum_fold_delta"] > -0.005 and signed["best_full_weight"] < 0.0)
    append_jsonl(progress_path, {"stage": "signed_gate", "pass": gate, **{key: signed[key] for key in ("nested_delta_r2", "positive_folds", "minimum_fold_delta", "best_full_weight")}})

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "CLEAN_OFFICIAL_ONLY",
        "branch": "without_archive",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "local_eval_read_by_builder": False,
        "external_label_file_read_by_builder": False,
        "nonofficial_file_read_by_builder": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": vars(args),
        "inputs": inputs,
        "base_candidate": {"path": str(base_path), "sha256": sha256_file(base_path), "rows": int(len(base))},
        "c282_oof": {"path": str(c282_oof_path), "sha256": sha256_file(c282_oof_path)},
        "feature_report": feature_report,
        "identity_parent": {
            "identity_weight": float(args.identity_weight),
            "applied_rows": int(parent_oof_frame.attrs.get("identity_applied", 0)),
            "support": parent_oof_frame.attrs.get("identity_support", {}),
            "oof_r2": float(r2_score(y, parent_oof)),
        },
        "direct_report": direct_report,
        "signed_gate": signed,
        "gate_pass": gate,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "rdkit": reference.Chem.rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "c1369": sha256_file(round2_root / "tools/build_round2_c1369_branch_fast_direct_stack.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
        },
    }

    if gate:
        direct_test = direct_test_predictions(x_train, y, x_test, direct_report["blend"])
        output_values = base["target"].to_numpy(np.float64).copy()
        source_map = dict(zip(target_test["id"].astype(int), direct_test.astype(float), strict=True))
        mask = test["target_type"].astype(str).eq(TARGET).to_numpy()
        base_ei = base.loc[mask, "target"].to_numpy(np.float64)
        source_ei = test.loc[mask, "id"].astype(int).map(source_map).to_numpy(np.float64)
        output_values[mask] = (1.0 - float(signed["best_full_weight"])) * base_ei + float(signed["best_full_weight"]) * source_ei
        if not np.isfinite(output_values).all():
            raise RuntimeError("Non-finite output")
        pd.DataFrame({"id": ids, "target": output_values}).to_csv(output, index=False)
        report["output"] = {"path": str(output), "sha256": sha256_file(output), "rows": int(len(ids)), "bytes": output.stat().st_size}
    else:
        report["output"] = None

    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", vars(args))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{sha256_file(path)}  {path.name}")
    if gate:
        manifest.append(f"{sha256_file(output)}  OUTPUT {output}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    append_jsonl(progress_path, {"stage": "finished", "gate_pass": gate, "elapsed_seconds": report["elapsed_seconds"], "output": report["output"]})
    print(json.dumps({"gate_pass": gate, "signed_gate": signed, "output": report["output"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
