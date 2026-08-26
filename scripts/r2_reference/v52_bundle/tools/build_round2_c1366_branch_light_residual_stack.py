#!/usr/bin/env python3
"""C1366 branch-local light residual stack.

Fast clean fallback after C1365's old C180/Round1 feature builder proved too
heavy.  This script builds deterministic official-input features only
(RDKit descriptors, physical counts, MACCS, small Morgan count blocks, and
graph grammar), trains target-local residual/blend models under grouped OOF,
and overlays only OOF-accepted targets onto a branch-local base CSV.

No local_eval, external_label, nonofficial, stored model, PI1M, Kaggle compute, upload, or
submission action is used by this builder.
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
import build_round2_c1365_branch_oof_stability_overlay as c1365
import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as graph


TARGETS = tuple(reference.TARGETS)
SCHEMA = "ppp.round2.c1366.branch-light-residual-stack.v1"
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


def model_factories(target: str) -> dict[str, Callable[[int], Any]]:
    weak = target in {"ei", "eea", "egb", "eps", "nc"}
    return {
        "ridge_50": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=50.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "ridge_250": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=250.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "extra_trees": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=260 if weak else 180,
                min_samples_leaf=2 if weak else 4,
                max_features=0.60 if weak else 0.50,
                random_state=seed,
                n_jobs=4,
            ),
        ),
        "hist_gbdt": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=260 if weak else 200,
                learning_rate=0.035,
                max_leaf_nodes=15 if weak else 31,
                min_samples_leaf=8 if weak else 20,
                l2_regularization=0.30,
                random_state=seed,
            ),
        ),
    }


def grouped_oof(factory: Callable[[int], Any], x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for grouped OOF")
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x[tr], y[tr])
        pred = np.asarray(model.predict(x[va]), dtype=np.float64)
        oof[va] = pred
        rows.append(
            {
                "fold": int(fold),
                "rows": int(len(va)),
                "r2": float(r2_score(y[va], pred)),
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("Non-finite OOF predictions")
    return oof, rows


def fold_delta_rows(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> list[dict[str, Any]]:
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rows: list[dict[str, Any]] = []
    for fold, (_, va) in enumerate(splitter.split(np.arange(len(y)), y, groups=groups)):
        parent_r2 = float(r2_score(y[va], parent[va]))
        candidate_r2 = float(r2_score(y[va], candidate[va]))
        rows.append({"fold": int(fold), "rows": int(len(va)), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2})
    return rows


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, seed: int) -> float:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(1200):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in chosen])
        if len(rows) > 1 and float(np.var(y[rows])) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def clip_by_train(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return reference.clip_prediction(np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64))


def feature_matrix(parent: dict[str, Any], morgan_bits: int) -> tuple[np.ndarray, dict[str, Any]]:
    molecules = parent["molecules"]
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, parent["keys"])
    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=morgan_bits).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=morgan_bits).toarray().astype(np.float32)
    maccs = c385.maccs_matrix(molecules)
    grammar = graph.grammar_features(molecules).astype(np.float32)
    x = c385.sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs, grammar]).astype(np.float32))
    return x, {
        "shape": [int(value) for value in x.shape],
        "rdkit_descriptors": int(len(descriptor_names)),
        "physical_features": int(len(physical_names)),
        "morgan_bits_each": int(morgan_bits),
        "maccs_bits": int(maccs.shape[1]),
        "graph_grammar_features": int(grammar.shape[1]),
    }


def run_target(parent: dict[str, Any], base_features: np.ndarray, target: str, models: tuple[str, ...], seed: int) -> dict[str, Any]:
    target_train = parent["pooled"].loc[parent["pooled"]["target_type"].astype(str).eq(target)].reset_index(drop=True)
    target_test = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(target)].reset_index(drop=False)
    key_to_index = parent["key_to_index"]
    train_idx = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
    test_idx = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
    info = parent["target_info"][target]
    y = target_train["target"].to_numpy(np.float64)
    if not np.array_equal(y, np.asarray(info["y"], dtype=np.float64)):
        raise RuntimeError(f"Training target alignment failed for {target}")
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    groups = np.asarray(info["groups"], dtype=object)
    cross_values, cross_available = reference.cross_property_arrays(parent["pooled"], parent["keys"])
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
        model_seed = seed + 1000 * TARGETS.index(target) + 37 * pos
        oof_raw, folds = grouped_oof(factory, x_train, y, groups, model_seed)
        oof = clip_by_train(y, oof_raw)
        final_model = factory(model_seed + 999)
        final_model.fit(x_train, y)
        test_pred = clip_by_train(y, np.asarray(final_model.predict(x_test), dtype=np.float64))
        oof_columns.append(oof)
        test_columns.append(test_pred)
        reports[name] = {"oof_r2": float(r2_score(y, oof)), "folds": folds}

    oof_stack = np.column_stack([parent_oof] + oof_columns)
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, oof_stack)
    candidate_oof = oof_stack @ weights + intercept
    parent_test = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(target)]
        .sort_values("id")["target"]
        .to_numpy(np.float64)
    )
    test_stack = np.column_stack([parent_test] + test_columns)
    candidate_test = test_stack @ weights + intercept
    lookup = target_train.groupby("canonical")["target"].mean().to_dict()
    exact_overrides = 0
    for local_pos, row in enumerate(target_test.itertuples(index=False)):
        if row.canonical in lookup:
            candidate_test[local_pos] = float(lookup[row.canonical])
            exact_overrides += 1
    candidate_oof = clip_by_train(y, candidate_oof)
    candidate_test = clip_by_train(y, candidate_test)
    fold_rows = fold_delta_rows(y, parent_oof, candidate_oof, groups)
    delta = float(r2_score(y, candidate_oof) - r2_score(y, parent_oof))
    positive = int(sum(row["delta_r2"] > 0 for row in fold_rows))
    lower = bootstrap_lower(y, parent_oof, candidate_oof, groups, seed + TARGETS.index(target))
    gate = bool(delta >= 0.004 and positive >= 4 and lower > -0.001)
    return {
        "target": target,
        "train_rows": int(len(y)),
        "test_rows": int(len(target_test)),
        "parent_r2": float(r2_score(y, parent_oof)),
        "candidate_r2": float(r2_score(y, candidate_oof)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "pass": gate,
        "folds": fold_rows,
        "model_reports": reports,
        "blend_name": blend_name,
        "blend_weights": {name: float(value) for name, value in zip(("parent", *factories.keys()), weights, strict=True)},
        "blend_intercept": float(intercept),
        "blend_oof_r2": float(blend_r2),
        "exact_train_overrides_on_test": int(exact_overrides),
        "test_ids": target_test["id"].to_numpy(np.int64),
        "test_pred": candidate_test.astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--targets", default="tg,egc,ei,nc,eps")
    parser.add_argument("--models", default="ridge_50,ridge_250,extra_trees,hist_gbdt")
    parser.add_argument("--morgan-bits", type=int, default=384)
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
    c1365.guard_path(base_path, role="base candidate", branch=args.branch, require_output_branch=True)
    c1365.guard_path(output, role="output", branch=args.branch, require_output_branch=True)
    c1365.guard_path(run_dir, role="run dir")
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    append_jsonl(progress_path, {"stage": "started", "created_at": datetime.now().astimezone().isoformat(), "branch": args.branch})

    parent = c1365.build_branch_parent(data_dir, args.branch)
    ids = parent["test"]["id"].to_numpy(np.int64)
    base = c1365.load_base(base_path, ids, args.branch)
    predictions = base["target"].to_numpy(np.float64).copy()
    append_jsonl(progress_path, {"stage": "parent_ready", "keys": len(parent["keys"]), "archive_rows_used": int(len(parent["archive"]))})

    x, feature_report = feature_matrix(parent, int(args.morgan_bits))
    append_jsonl(progress_path, {"stage": "features_ready", "shape": feature_report["shape"]})

    active_targets = c1365.parse_targets(args.targets)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    target_reports: dict[str, Any] = {}
    for target in active_targets:
        append_jsonl(progress_path, {"stage": "target_started", "target": target})
        result = run_target(parent, x, target, models, int(args.seed))
        target_reports[target] = {key: value for key, value in result.items() if key not in {"test_ids", "test_pred"}}
        if result["pass"]:
            id_to_value = dict(zip(result["test_ids"].astype(int), result["test_pred"].astype(float), strict=True))
            mask = parent["test"]["target_type"].astype(str).eq(target).to_numpy()
            predictions[mask] = parent["test"].loc[mask, "id"].astype(int).map(id_to_value).to_numpy(np.float64)
        append_jsonl(
            progress_path,
            {
                "stage": "target_finished",
                "target": target,
                "accepted": bool(result["pass"]),
                "delta_r2": float(result["delta_r2"]),
                "positive_folds": int(result["positive_folds"]),
                "bootstrap_lower": float(result["group_bootstrap_lower"]),
            },
        )

    assembled = pd.DataFrame({"id": ids, "target": predictions})
    assembled, override_report = c1365.apply_branch_overrides(parent, assembled)
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
        "accepted_targets": [target for target, value in target_reports.items() if value["pass"]],
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
            "c1365_helpers": sha256_file(round2_root / "tools/build_round2_c1365_branch_oof_stability_overlay.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "graph": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
        },
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", vars(args))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "target": target,
                "accepted": value["pass"],
                "delta_r2": value["delta_r2"],
                "positive_folds": value["positive_folds"],
                "group_bootstrap_lower": value["group_bootstrap_lower"],
                "parent_r2": value["parent_r2"],
                "candidate_r2": value["candidate_r2"],
                "blend_name": value["blend_name"],
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
