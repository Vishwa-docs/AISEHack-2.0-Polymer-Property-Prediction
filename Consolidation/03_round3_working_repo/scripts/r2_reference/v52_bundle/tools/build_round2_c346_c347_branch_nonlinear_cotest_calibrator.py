#!/usr/bin/env python3
"""C346/C347 branch-guarded nonlinear co-test residual calibrator.

This is a bounded continuation of the C327/C332 co-test residual family.  It
keeps the same official-only provenance and branch boundaries, but tests a
small set of additional low-variance nonlinear residual arms on the fold-local
OOF feature table before writing a complete candidate CSV.

No local_eval, external_label file, Kaggle compute, upload, or submission action is used by
this builder.  Post-freeze scoring must be performed by the separate local_eval
diagnostic scorer after the CSV hash is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_round2_c327_noarchive_cotest_meta_calibrator as c327
import build_round2_c332_archive_c050_cotest_meta_calibrator as c332
import initial_reference_pipeline as reference


DEFAULTS = {
    "without_archive": {
        "base": "experiments/final_submission_runs/without_archive/R2-C343-NOARCHIVE-TG-C340-BLEND075-OVER-C336-20260808.csv",
        "oof": "experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1/oof_predictions.csv",
        "targets": "egc,ei,eps,nc",
    },
    "with_archive": {
        "base": "experiments/final_submission_runs/with_archive/R2-C334-ARCHIVE-TARGET-SPLICE-C333-EEA-EPS-C332-NC-20260808.csv",
        "oof": "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7/oof_predictions.csv",
        "targets": "ei,eps,nc",
    },
}


@dataclass(frozen=True)
class TargetGate:
    min_delta: float
    min_nonnegative_folds: int
    max_low_support_loss: float
    residual_clip: float
    blend_weight: float
    tree_leaf: int


GATES = {
    "tg": TargetGate(0.0020, 4, -0.004, 35.0, 0.35, 12),
    "egc": TargetGate(0.0010, 4, -0.004, 0.65, 0.45, 10),
    "egb": TargetGate(0.0010, 4, -0.004, 0.65, 0.45, 6),
    "ei": TargetGate(0.0010, 3, -0.012, 0.45, 0.35, 4),
    "eea": TargetGate(0.0010, 3, -0.012, 0.45, 0.35, 4),
    "eps": TargetGate(0.0010, 3, -0.012, 0.70, 0.35, 4),
    "nc": TargetGate(0.0010, 3, -0.012, 0.080, 0.35, 4),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, branch: str, *, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden path: {path}")
    if branch == "without_archive":
        if not allow_output and ("/archive/" in low or low.endswith("/archive") or "with_archive" in low):
            raise RuntimeError(f"Refusing archive/cross-branch input for no-archive run: {path}")
        if allow_output and "with_archive" in low:
            raise RuntimeError(f"Refusing cross-branch output for no-archive run: {path}")
    elif branch == "with_archive":
        if not allow_output and "without_archive" in low:
            raise RuntimeError(f"Refusing cross-branch input for archive run: {path}")
        if allow_output and "without_archive" in low:
            raise RuntimeError(f"Refusing cross-branch output for archive run: {path}")
    else:
        raise RuntimeError(f"Unknown branch: {branch}")


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    invalid = [target for target in targets if target not in reference.TARGETS]
    if invalid:
        raise RuntimeError(f"Unknown targets: {invalid}")
    return targets


def load_branch_inputs(
    branch: str,
    data_dir: Path,
    base_path: Path,
    oof_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    guard_path(base_path, branch)
    guard_path(oof_path, branch)
    if branch == "without_archive":
        train, test, oof, base, inputs = c327.load_inputs(data_dir, base_path, oof_path)
        archive = None
    else:
        train, test, archive, oof, base, inputs = c332.load_inputs(data_dir, base_path, oof_path)
    return train, test, oof, archive, {**inputs, "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size}}


def make_model(name: str, gate: TargetGate) -> Any:
    if name == "ridge100":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), c327.Ridge(alpha=100.0))
    if name == "krr_linear":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), KernelRidge(alpha=20.0, kernel="linear"))
    if name == "knn5":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5, weights="distance", metric="minkowski"),
        )
    if name == "knn15":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=15, weights="distance", metric="minkowski"),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=320,
                min_samples_leaf=gate.tree_leaf,
                max_features=0.80,
                random_state=346,
                n_jobs=-1,
            ),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=260,
                min_samples_leaf=gate.tree_leaf,
                max_features=0.85,
                random_state=347,
                n_jobs=-1,
            ),
        )
    raise KeyError(name)


def evaluate_target(target: str, oof: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    gate = GATES[target]
    target_oof = oof[oof["target_type"] == target].reset_index(drop=True)
    if len(target_oof) < 20:
        raise RuntimeError(f"Too few OOF rows for {target}: {len(target_oof)}")
    train_pool = oof[oof["target_type"] == target]["canonical"].astype(str).tolist()
    nearest_train = c327.nearest_similarity(target_oof["canonical"].astype(str).tolist(), train_pool)
    oof_pivot = c327.pivot_predictions(oof, "prediction")
    x, feature_names = c327.make_features(target_oof["canonical"].astype(str).to_numpy(), target, oof_pivot, nearest_train)
    y = target_oof["target"].to_numpy(float)
    parent = target_oof["prediction"].to_numpy(float)
    residual = y - parent
    groups = target_oof["canonical"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    folds = np.full(len(y), -1, dtype=np.int64)
    for fold, (_, va) in enumerate(splitter.split(x, y, groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError(f"Fold assignment failed for {target}")
    parent_r2 = float(r2_score(y, parent))
    partner_counts = np.nan_to_num(x[:, feature_names.index("partner_count")], nan=0.0)
    low_support = partner_counts <= 1.0
    arms: dict[str, dict[str, Any]] = {}
    oof_preds: dict[str, np.ndarray] = {}
    for name in ("ridge100", "krr_linear", "knn5", "knn15", "extra_trees", "random_forest"):
        pred = np.full(len(y), np.nan, dtype=np.float64)
        fold_deltas: list[float] = []
        for fold in sorted(set(folds)):
            tr = np.flatnonzero(folds != fold)
            va = np.flatnonzero(folds == fold)
            model = make_model(name, gate)
            model.fit(x[tr], residual[tr])
            delta = np.clip(np.asarray(model.predict(x[va]), dtype=np.float64), -gate.residual_clip, gate.residual_clip)
            pred[va] = parent[va] + gate.blend_weight * delta
            fold_deltas.append(float(r2_score(y[va], pred[va]) - r2_score(y[va], parent[va])))
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Non-finite OOF predictions for {target}/{name}")
        oof_preds[name] = pred
        low_support_delta = None
        if int(np.sum(low_support)) >= 10:
            low_support_delta = float(r2_score(y[low_support], pred[low_support]) - r2_score(y[low_support], parent[low_support]))
        arms[name] = {
            "oof_r2": float(r2_score(y, pred)),
            "delta_vs_parent_oof": float(r2_score(y, pred) - parent_r2),
            "fold_deltas": fold_deltas,
            "nonnegative_folds": int(sum(delta >= 0.0 for delta in fold_deltas)),
            "low_support_delta": low_support_delta,
        }
    mean_names = ("ridge100", "krr_linear", "extra_trees")
    mean_pred = np.mean(np.column_stack([oof_preds[name] for name in mean_names]), axis=1)
    fold_deltas = []
    for fold in sorted(set(folds)):
        va = np.flatnonzero(folds == fold)
        fold_deltas.append(float(r2_score(y[va], mean_pred[va]) - r2_score(y[va], parent[va])))
    low_support_delta = None
    if int(np.sum(low_support)) >= 10:
        low_support_delta = float(r2_score(y[low_support], mean_pred[low_support]) - r2_score(y[low_support], parent[low_support]))
    arms["mean_ridge_krr_et"] = {
        "oof_r2": float(r2_score(y, mean_pred)),
        "delta_vs_parent_oof": float(r2_score(y, mean_pred) - parent_r2),
        "fold_deltas": fold_deltas,
        "nonnegative_folds": int(sum(delta >= 0.0 for delta in fold_deltas)),
        "low_support_delta": low_support_delta,
    }
    selected_arm = max(arms, key=lambda item: arms[item]["oof_r2"])
    selected = arms[selected_arm]
    pass_gate = bool(
        selected["delta_vs_parent_oof"] >= gate.min_delta
        and selected["nonnegative_folds"] >= gate.min_nonnegative_folds
        and (selected["low_support_delta"] is None or selected["low_support_delta"] >= gate.max_low_support_loss)
    )

    target_test = test[test["target_type"] == target].copy()
    test_pivot = c327.pivot_predictions(test.rename(columns={"base_prediction": "prediction"}), "prediction")
    nearest_test = c327.nearest_similarity(target_test["canonical"].astype(str).tolist(), train_pool)
    test_x, _ = c327.make_features(target_test["canonical"].astype(str).to_numpy(), target, test_pivot, nearest_test)
    base_values = target_test["base_prediction"].to_numpy(float)
    if pass_gate:
        if selected_arm == "mean_ridge_krr_et":
            full_deltas = []
            for name in mean_names:
                model = make_model(name, gate)
                model.fit(x, residual)
                full_deltas.append(np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -gate.residual_clip, gate.residual_clip))
            raw_delta = np.mean(np.column_stack(full_deltas), axis=1)
        else:
            model = make_model(selected_arm, gate)
            model.fit(x, residual)
            raw_delta = np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -gate.residual_clip, gate.residual_clip)
        overlay = base_values + gate.blend_weight * raw_delta
        changed = np.ones(len(base_values), dtype=bool)
    else:
        overlay = base_values.copy()
        changed = np.zeros(len(base_values), dtype=bool)
    return overlay, {
        "target": target,
        "parent_oof_r2": parent_r2,
        "arms": arms,
        "selected_arm": selected_arm,
        "selected_oof_r2": float(selected["oof_r2"]),
        "selected_delta_vs_parent_oof": float(selected["delta_vs_parent_oof"]),
        "clean_oof_gate_pass": pass_gate,
        "gate": {
            "min_delta": gate.min_delta,
            "min_nonnegative_folds": gate.min_nonnegative_folds,
            "max_low_support_loss": gate.max_low_support_loss,
            "residual_clip": gate.residual_clip,
            "blend_weight": gate.blend_weight,
            "tree_leaf": gate.tree_leaf,
        },
        "train_rows": int(len(target_oof)),
        "test_rows": int(len(target_test)),
        "changed_rows": int(np.sum(changed)),
        "changed_ids_sha256": hashlib.sha256(
            json.dumps(target_test.loc[changed, "id"].astype(int).tolist(), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "feature_names": feature_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True, choices=("with_archive", "without_archive"))
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv")
    parser.add_argument("--oof-csv")
    parser.add_argument("--targets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    branch_defaults = DEFAULTS[args.branch]
    data_dir = Path(args.data_dir).resolve()
    base_path = Path(args.base_csv or branch_defaults["base"]).resolve()
    oof_path = Path(args.oof_csv or branch_defaults["oof"]).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path in (base_path, oof_path):
        guard_path(path, args.branch)
    for path in (output, manifest):
        guard_path(path, args.branch, allow_output=True)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if args.branch not in str(output):
        raise RuntimeError(f"Output must live in branch namespace {args.branch}: {output}")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")
    targets = parse_targets(args.targets or branch_defaults["targets"])

    train, test, oof, archive, inputs = load_branch_inputs(args.branch, data_dir, base_path, oof_path)
    ids = test["id"].to_numpy(int)
    result = test["base_prediction"].to_numpy(float).copy()
    target_reports: dict[str, Any] = {}
    for target in targets:
        overlay, report = evaluate_target(target, oof, test)
        mask = test["target_type"].to_numpy(str) == target
        if int(np.sum(mask)) != len(overlay):
            raise RuntimeError(f"Target alignment failed for {target}")
        result[mask] = overlay
        target_reports[target] = report
    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.c346-c347.branch-nonlinear-cotest-calibrator.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "official_current_train_used": True,
        "archive_labels_used": bool(args.branch == "with_archive"),
        "archive_file_read": bool(args.branch == "with_archive"),
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "branch-guarded nonlinear co-test residual calibrator with grouped OOF gates",
        "targets_requested": list(targets),
        "target_reports": target_reports,
        "inputs": inputs,
        "rows": {
            "train": int(len(train)),
            "test": int(len(test)),
            "archive": int(len(archive)) if archive is not None else 0,
            "oof": int(len(oof)),
        },
        "final_notebook_note": "development script reads stored OOF/base artifacts; final notebook must regenerate branch parent and OOF components from official inputs in one run",
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": record["output"],
                "target_reports": {
                    target: {
                        "selected_arm": report["selected_arm"],
                        "parent_oof_r2": report["parent_oof_r2"],
                        "selected_oof_r2": report["selected_oof_r2"],
                        "selected_delta_vs_parent_oof": report["selected_delta_vs_parent_oof"],
                        "clean_oof_gate_pass": report["clean_oof_gate_pass"],
                        "changed_rows": report["changed_rows"],
                    }
                    for target, report in target_reports.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
