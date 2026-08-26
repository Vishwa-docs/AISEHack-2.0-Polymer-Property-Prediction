#!/usr/bin/env python3
"""C1369 branch-local fast direct stack.

This is the fast fallback after C1365-C1368 showed that rebuilding the full
reference parent is too expensive for quick branch iteration.  It reads only
official branch inputs, trains target-local models directly on branch labels,
overlays requested target predictions onto a frozen branch base CSV, and writes
a candidate for separate local_eval scoring.

The builder does not read local_eval, external_label, nonofficial, cached predictions, stored
weights, PI1M, Kaggle compute, uploads, or submissions.
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
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_round2_c385_archive_weak_target_model_zoo as c385
import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as graph
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
SCHEMA = "ppp.round2.c1369.branch-fast-direct-stack.v1"
SEED = 20260808


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


def guard_path(path: Path, *, role: str, branch: str | None = None, require_output_branch: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if branch == "without_archive" and "with_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for without_archive: {path}")
    if branch == "with_archive" and "without_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for with_archive: {path}")
    if require_output_branch and branch is not None and f"/{branch}/" not in low:
        raise RuntimeError(f"{role} path must stay in /{branch}/ namespace: {path}")


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    bad = [target for target in targets if target not in TARGETS]
    if bad:
        raise RuntimeError(f"Invalid targets: {bad}")
    return targets


def load_branch_inputs(data_dir: Path, branch: str) -> dict[str, Any]:
    if branch == "with_archive":
        train, test, archive, inputs = reference.load_inputs(data_dir)
    elif branch == "without_archive":
        train, test, inputs = c282.load_current_only_inputs(data_dir)
        archive = train.iloc[0:0].copy()
    else:
        raise RuntimeError(f"Unknown branch: {branch}")
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    return {
        "train": train,
        "test": test,
        "archive": archive,
        "inputs": inputs,
        "raw_labels": raw_labels,
        "pooled": pooled,
        "keys": keys,
        "key_to_index": key_to_index,
        "molecules": molecules,
    }


def load_base(path: Path, ids: np.ndarray, branch: str) -> pd.DataFrame:
    guard_path(path, role="base candidate", branch=branch, require_output_branch=True)
    base = pd.read_csv(path)
    if list(base.columns) != ["id", "target"]:
        raise RuntimeError(f"Unexpected base candidate schema: {path}")
    if not np.array_equal(base["id"].to_numpy(np.int64), ids):
        raise RuntimeError("Base candidate IDs/order do not match official test")
    values = base["target"].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Base candidate contains non-finite predictions")
    return base


def feature_matrix(parent: dict[str, Any], morgan_bits: int) -> tuple[np.ndarray, dict[str, Any]]:
    molecules = parent["molecules"]
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, parent["keys"])
    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=morgan_bits).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=morgan_bits).toarray().astype(np.float32)
    maccs = c385.maccs_matrix(molecules)
    grammar = graph.grammar_features(molecules).astype(np.float32)
    matrix = c385.sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs, grammar]).astype(np.float32))
    return matrix, {
        "shape": [int(value) for value in matrix.shape],
        "rdkit_descriptors": int(len(descriptor_names)),
        "physical_features": int(len(physical_names)),
        "morgan_bits_each": int(morgan_bits),
        "maccs_bits": int(maccs.shape[1]),
        "graph_grammar_features": int(grammar.shape[1]),
    }


def model_factories(target: str) -> dict[str, Callable[[int], Any]]:
    weak = target in {"ei", "eea", "egb", "eps", "nc"}
    return {
        "ridge_20": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=20.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "ridge_80": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=80.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "ridge_250": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=250.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "hist_gbdt": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=180 if weak else 140,
                learning_rate=0.04,
                max_leaf_nodes=15 if weak else 31,
                min_samples_leaf=8 if weak else 20,
                l2_regularization=0.30,
                random_state=seed,
            ),
        ),
        "extra_trees": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=180 if weak else 120,
                min_samples_leaf=2 if weak else 4,
                max_features=0.55 if weak else 0.45,
                random_state=seed,
                n_jobs=4,
            ),
        ),
    }


def clip_by_train(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return reference.clip_prediction(np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64))


def grouped_oof(factory: Callable[[int], Any], x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for grouped OOF")
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x[tr], y[tr])
        pred = clip_by_train(y[tr], np.asarray(model.predict(x[va]), dtype=np.float64))
        oof[va] = pred
        rows.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
    if not np.isfinite(oof).all():
        raise RuntimeError("Non-finite OOF predictions")
    return oof, rows


def run_target(parent: dict[str, Any], base_features: np.ndarray, target: str, models: tuple[str, ...], seed: int, progress_path: Path) -> dict[str, Any]:
    pooled = parent["pooled"]
    test = parent["test"]
    key_to_index = parent["key_to_index"]
    target_train = pooled.loc[pooled["target_type"].astype(str).eq(target)].reset_index(drop=True)
    target_test = test.loc[test["target_type"].astype(str).eq(target)].reset_index(drop=True)
    train_idx = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
    test_idx = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
    y = target_train["target"].to_numpy(np.float64)
    groups = np.asarray([graph.no_stereo(value) for value in target_train["canonical"].astype(str)], dtype=object)
    cross_values, cross_available = reference.cross_property_arrays(pooled, parent["keys"])
    dense = c385.sanitize_dense(reference.target_dense_features(base_features, cross_values, cross_available, target))
    x_train = dense[train_idx]
    x_test = dense[test_idx]
    available = model_factories(target)
    factories = {name: available[name] for name in models if name in available}
    if not factories:
        raise RuntimeError("No usable model factories requested")

    oof_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    reports: dict[str, Any] = {}
    for pos, (name, factory) in enumerate(factories.items()):
        append_jsonl(progress_path, {"stage": "model_started", "target": target, "model": name})
        model_seed = seed + 1000 * TARGETS.index(target) + 37 * pos
        oof, folds = grouped_oof(factory, x_train, y, groups, model_seed)
        final_model = factory(model_seed + 999)
        final_model.fit(x_train, y)
        test_pred = clip_by_train(y, np.asarray(final_model.predict(x_test), dtype=np.float64))
        oof_columns.append(oof)
        test_columns.append(test_pred)
        reports[name] = {"oof_r2": float(r2_score(y, oof)), "folds": folds}
        append_jsonl(progress_path, {"stage": "model_finished", "target": target, "model": name, "oof_r2": reports[name]["oof_r2"]})

    oof_stack = np.column_stack(oof_columns)
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, oof_stack)
    candidate_oof = clip_by_train(y, oof_stack @ weights + intercept)
    test_stack = np.column_stack(test_columns)
    candidate_test = clip_by_train(y, test_stack @ weights + intercept)
    lookup = target_train.groupby("canonical")["target"].median().to_dict()
    exact_overrides = 0
    for pos, row in enumerate(target_test.itertuples(index=False)):
        if row.canonical in lookup:
            candidate_test[pos] = float(lookup[row.canonical])
            exact_overrides += 1
    return {
        "target": target,
        "train_rows": int(len(y)),
        "test_rows": int(len(target_test)),
        "candidate_oof_r2": float(r2_score(y, candidate_oof)),
        "model_reports": reports,
        "blend_name": blend_name,
        "blend_weights": {name: float(value) for name, value in zip(factories.keys(), weights, strict=True)},
        "blend_intercept": float(intercept),
        "blend_oof_r2": float(blend_r2),
        "exact_train_overrides_on_test": int(exact_overrides),
        "test_ids": target_test["id"].to_numpy(np.int64),
        "test_pred": candidate_test.astype(np.float64),
    }


def apply_branch_overrides(parent: dict[str, Any], assembled: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = parent["raw_labels"]
    test = parent["test"]
    lookup = raw.groupby(["canonical", "target_type"])["target"].median().to_dict()
    values = assembled["target"].to_numpy(np.float64).copy()
    overrides = 0
    for pos, row in enumerate(test.itertuples(index=False)):
        key = (row.canonical, row.target_type)
        if key in lookup:
            values[pos] = float(lookup[key])
            overrides += 1
    out = assembled.copy()
    out["target"] = values
    return out, {"branch_label_exact_overrides": int(overrides)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--targets", default="egc,ei,nc,eps")
    parser.add_argument("--models", default="ridge_20,ridge_80,ridge_250")
    parser.add_argument("--morgan-bits", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
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
    output = Path(args.output)
    if not output.is_absolute():
        output = (round2_root / output).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (round2_root / run_dir).resolve()
    guard_path(base_path, role="base candidate", branch=args.branch, require_output_branch=True)
    guard_path(output, role="output", branch=args.branch, require_output_branch=True)
    guard_path(run_dir, role="run dir")
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    append_jsonl(progress_path, {"stage": "started", "created_at": datetime.now().astimezone().isoformat(), "branch": args.branch})

    parent = load_branch_inputs(data_dir, args.branch)
    ids = parent["test"]["id"].to_numpy(np.int64)
    base = load_base(base_path, ids, args.branch)
    predictions = base["target"].to_numpy(np.float64).copy()
    append_jsonl(progress_path, {"stage": "inputs_ready", "keys": len(parent["keys"]), "archive_rows_used": int(len(parent["archive"]))})

    x, feature_report = feature_matrix(parent, int(args.morgan_bits))
    append_jsonl(progress_path, {"stage": "features_ready", "shape": feature_report["shape"]})

    active_targets = parse_targets(args.targets)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    target_reports: dict[str, Any] = {}
    for target in active_targets:
        append_jsonl(progress_path, {"stage": "target_started", "target": target})
        result = run_target(parent, x, target, models, int(args.seed), progress_path)
        target_reports[target] = {key: value for key, value in result.items() if key not in {"test_ids", "test_pred"}}
        id_to_value = dict(zip(result["test_ids"].astype(int), result["test_pred"].astype(float), strict=True))
        mask = parent["test"]["target_type"].astype(str).eq(target).to_numpy()
        predictions[mask] = parent["test"].loc[mask, "id"].astype(int).map(id_to_value).to_numpy(np.float64)
        append_jsonl(progress_path, {"stage": "target_finished", "target": target, "candidate_oof_r2": float(result["candidate_oof_r2"])})

    assembled = pd.DataFrame({"id": ids, "target": predictions})
    assembled, override_report = apply_branch_overrides(parent, assembled)
    if len(assembled) != 4940 or assembled["id"].duplicated().any() or not np.array_equal(assembled["id"].to_numpy(np.int64), np.arange(1, 4941)):
        raise RuntimeError("Output row/order contract failed")
    if not np.isfinite(assembled["target"].to_numpy(np.float64)).all():
        raise RuntimeError("Non-finite output prediction")
    assembled.to_csv(output, index=False)
    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "classification": "CLEAN_OFFICIAL_ONLY",
        "official_current_train_used": True,
        "archive_labels_used": bool(args.branch == "with_archive"),
        "archive_rows_used": int(len(parent["archive"])),
        "local_eval_read_by_builder": False,
        "external_label_file_read_by_builder": False,
        "nonofficial_file_read_by_builder": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": vars(args),
        "inputs": parent["inputs"],
        "base_candidate": {"path": str(base_path), "sha256": sha256_file(base_path), "rows": int(len(base))},
        "feature_report": feature_report,
        "target_reports": target_reports,
        "accepted_targets": list(active_targets),
        "override_report": override_report,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(assembled)), "bytes": output.stat().st_size},
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
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "current_only_loader": sha256_file(round2_root / "tools/round2_c282_current_only_reference.py"),
            "graph": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "c385_features": sha256_file(round2_root / "tools/build_round2_c385_archive_weak_target_model_zoo.py"),
        },
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", vars(args))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "target": target,
                "candidate_oof_r2": value["candidate_oof_r2"],
                "blend_name": value["blend_name"],
                "blend_oof_r2": value["blend_oof_r2"],
            }
            for target, value in target_reports.items()
        ]
    ).to_csv(run_dir / "component_summary.csv", index=False)
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{sha256_file(path)}  {path.name}")
    manifest.append(f"{sha256_file(output)}  OUTPUT {output}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    append_jsonl(progress_path, {"stage": "finished", "accepted_targets": report["accepted_targets"], "output_sha256": report["output"]["sha256"], "elapsed_seconds": report["elapsed_seconds"]})
    print(json.dumps({"output": report["output"], "accepted_targets": report["accepted_targets"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
