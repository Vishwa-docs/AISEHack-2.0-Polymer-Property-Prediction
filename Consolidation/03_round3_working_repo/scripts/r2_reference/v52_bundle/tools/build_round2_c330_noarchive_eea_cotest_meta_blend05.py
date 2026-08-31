#!/usr/bin/env python3
"""C330 no-archive EEA co-test residual meta-calibrator, blend 0.5.

This is the clean-OOF-selected EEA-only child after C327. C327 showed only EEA
passed the clean gate, and a clean OOF sensitivity check selected blend 0.5 over
the original 0.75. This builder changes only EEA rows and leaves all other
targets unchanged from C327.
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")

TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("ei", "eea", "eps", "nc")
DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C327-NOARCHIVE-COTEST-META-CALIBRATOR-20260808.csv"
)
DEFAULT_OOF = "experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1/oof_predictions.csv"


@dataclass(frozen=True)
class TargetConfig:
    min_clean_oof_delta: float
    min_nonnegative_folds: int
    max_low_support_loss: float
    residual_clip: float
    blend_weight: float


CONFIGS = {
    "ei": TargetConfig(0.003, 4, -0.003, 0.45, 0.75),
    "eea": TargetConfig(0.003, 4, -0.003, 0.45, 0.50),
    "eps": TargetConfig(0.003, 4, -0.003, 0.70, 0.65),
    "nc": TargetConfig(0.003, 4, -0.003, 0.080, 0.65),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden path: {path}")
    if not allow_output:
        if "/archive/" in low or low.endswith("/archive") or "with_archive" in low:
            raise RuntimeError(f"Refusing archive/cross-branch input path for C330: {path}")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    invalid = [target for target in targets if target not in ACTIVE_TARGETS]
    if invalid:
        raise RuntimeError(f"C330 only supports {ACTIVE_TARGETS}, got {invalid}")
    return targets


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path)
    if "without_archive" not in str(path):
        raise RuntimeError(f"C330 base must be branch-local without_archive: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def load_inputs(data_dir: Path, base_path: Path, oof_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    for path in (train_path, test_path, base_path, oof_path):
        guard_path(path)
    inputs = {
        "train.csv": {"path": str(train_path), "sha256": sha256_file(train_path), "bytes": train_path.stat().st_size},
        "test.csv": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "c282_oof_predictions.csv": {
            "path": str(oof_path),
            "sha256": sha256_file(oof_path),
            "bytes": oof_path.stat().st_size,
            "role": "local development OOF artifact generated from official current train; final notebook must regenerate it",
        },
    }
    if inputs["train.csv"]["sha256"] != reference.EXPECTED_HASHES["train.csv"]:
        raise RuntimeError("train.csv hash mismatch")
    if inputs["test.csv"]["sha256"] != reference.EXPECTED_HASHES["test.csv"]:
        raise RuntimeError("test.csv hash mismatch")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    oof = pd.read_csv(oof_path)
    if list(train.columns) != ["smiles", "target", "target_type"] or len(train) != 7409:
        raise RuntimeError("Unexpected train.csv schema/count")
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected test.csv schema/count")
    expected_oof_columns = ["canonical", "target_type", "target", "prediction", "sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local"]
    if list(oof.columns) != expected_oof_columns:
        raise RuntimeError("Unexpected C282 OOF schema")
    train = train.copy()
    test = test.copy()
    oof = oof.copy()
    train["target_type"] = train["target_type"].astype(str).str.lower()
    test["target_type"] = test["target_type"].astype(str).str.lower()
    oof["target_type"] = oof["target_type"].astype(str).str.lower()
    train["canonical"] = [canonical_smiles(value) for value in train["smiles"]]
    test["canonical"] = [canonical_smiles(value) for value in test["smiles"]]
    if set(train["target_type"]) != set(TARGETS) or set(test["target_type"]) != set(TARGETS):
        raise RuntimeError("Unexpected target set")
    if test["id"].duplicated().any() or not np.array_equal(test["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    for target in ACTIVE_TARGETS:
        if len(oof[oof["target_type"] == target]) != len(train[train["target_type"] == target]):
            raise RuntimeError(f"C282 OOF row count mismatch for {target}")
    base = load_base(base_path, test["id"].to_numpy(int))
    test["base_prediction"] = base["target"].to_numpy(float)
    return train, test, oof, base, inputs


def pivot_predictions(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return frame.pivot_table(index="canonical", columns="target_type", values=value_column, aggfunc="mean")


def make_features(canonicals: np.ndarray, target: str, pivot: pd.DataFrame, nearest_same: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    base_names = []
    for prop in TARGETS:
        base_names += [f"pred_{prop}", f"has_{prop}"]
    extra_names = [
        "partner_count",
        "nearest_same_target_tanimoto",
        "ei_from_gap",
        "eea_from_gap",
        "gap_residual",
        "eps_minus_nc2",
        "eps_from_nc2",
        "nc_from_eps",
        "abs_gap_residual",
        "abs_eps_minus_nc2",
    ]
    names = base_names + extra_names
    for canonical in canonicals:
        values: dict[str, float] = {}
        row_values: list[float] = []
        if canonical in pivot.index:
            source = pivot.loc[canonical]
        else:
            source = pd.Series(dtype=float)
        partner_count = 0
        for prop in TARGETS:
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
        if "eea" in values and "egc" in values:
            ei_from_gap = values["eea"] + values["egc"]
        if "ei" in values and "egc" in values:
            eea_from_gap = values["ei"] - values["egc"]
        if all(prop in values for prop in ("ei", "eea", "egc")):
            gap_residual = values["ei"] - values["eea"] - values["egc"]
        eps_minus_nc2 = np.nan
        eps_from_nc2 = np.nan
        nc_from_eps = np.nan
        if "eps" in values and "nc" in values:
            eps_minus_nc2 = values["eps"] - values["nc"] ** 2
            eps_from_nc2 = values["nc"] ** 2 + max(eps_minus_nc2, 0.02)
            nc_from_eps = math.sqrt(max(values["eps"] - max(eps_minus_nc2, 0.02), 1.0))
        row_values.extend(
            [
                float(partner_count),
                float(nearest_same.get(str(canonical), 0.0)),
                ei_from_gap,
                eea_from_gap,
                gap_residual,
                eps_minus_nc2,
                eps_from_nc2,
                nc_from_eps,
                abs(gap_residual) if pd.notna(gap_residual) else np.nan,
                abs(eps_minus_nc2) if pd.notna(eps_minus_nc2) else np.nan,
            ]
        )
        rows.append(row_values)
    return np.asarray(rows, dtype=np.float64), names


def morgan_fps(canonicals: list[str]) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps: dict[str, Any] = {}
    for canonical in sorted(set(canonicals)):
        mol = Chem.MolFromSmiles(str(canonical))
        if mol is None:
            continue
        fps[str(canonical)] = generator.GetFingerprint(mol)
    return fps


def nearest_similarity(query: list[str], train_pool: list[str]) -> dict[str, float]:
    fps = morgan_fps(query + train_pool)
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


def make_model(name: str) -> Any:
    if name == "ridge10":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0))
    if name == "ridge100":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    if name == "huber":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000),
        )
    raise KeyError(name)


def evaluate_target(target: str, oof: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = CONFIGS[target]
    target_oof = oof[oof["target_type"] == target].reset_index(drop=True)
    train_pool = oof[oof["target_type"] == target]["canonical"].astype(str).tolist()
    nearest_train = nearest_similarity(target_oof["canonical"].astype(str).tolist(), train_pool)
    oof_pivot = pivot_predictions(oof, "prediction")
    x, feature_names = make_features(target_oof["canonical"].astype(str).to_numpy(), target, oof_pivot, nearest_train)
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
    arms: dict[str, dict[str, Any]] = {}
    oof_preds: dict[str, np.ndarray] = {}
    for name in ("ridge10", "ridge100", "huber"):
        pred_residual = np.full(len(y), np.nan, dtype=np.float64)
        fold_deltas: list[float] = []
        for fold in sorted(set(folds)):
            tr = np.flatnonzero(folds != fold)
            va = np.flatnonzero(folds == fold)
            model = make_model(name)
            model.fit(x[tr], residual[tr])
            raw_delta = np.asarray(model.predict(x[va]), dtype=np.float64)
            raw_delta = np.clip(raw_delta, -cfg.residual_clip, cfg.residual_clip)
            pred = parent[va] + cfg.blend_weight * raw_delta
            pred_residual[va] = pred
            fold_deltas.append(float(r2_score(y[va], pred) - r2_score(y[va], parent[va])))
        if not np.isfinite(pred_residual).all():
            raise RuntimeError(f"Non-finite OOF predictions for {target}/{name}")
        oof_preds[name] = pred_residual
        partner_counts = np.nan_to_num(x[:, feature_names.index("partner_count")], nan=0.0)
        low_support = partner_counts <= 1.0
        low_support_delta = None
        if int(np.sum(low_support)) >= 10:
            low_support_delta = float(r2_score(y[low_support], pred_residual[low_support]) - r2_score(y[low_support], parent[low_support]))
        arms[name] = {
            "oof_r2": float(r2_score(y, pred_residual)),
            "delta_vs_parent_oof": float(r2_score(y, pred_residual) - parent_r2),
            "fold_deltas": fold_deltas,
            "nonnegative_folds": int(sum(delta >= 0.0 for delta in fold_deltas)),
            "low_support_delta": low_support_delta,
        }
    mean_pred = np.mean(np.column_stack([oof_preds["ridge10"], oof_preds["ridge100"], oof_preds["huber"]]), axis=1)
    fold_deltas = []
    for fold in sorted(set(folds)):
        va = np.flatnonzero(folds == fold)
        fold_deltas.append(float(r2_score(y[va], mean_pred[va]) - r2_score(y[va], parent[va])))
    partner_counts = np.nan_to_num(x[:, feature_names.index("partner_count")], nan=0.0)
    low_support = partner_counts <= 1.0
    low_support_delta = None
    if int(np.sum(low_support)) >= 10:
        low_support_delta = float(r2_score(y[low_support], mean_pred[low_support]) - r2_score(y[low_support], parent[low_support]))
    arms["mean3"] = {
        "oof_r2": float(r2_score(y, mean_pred)),
        "delta_vs_parent_oof": float(r2_score(y, mean_pred) - parent_r2),
        "fold_deltas": fold_deltas,
        "nonnegative_folds": int(sum(delta >= 0.0 for delta in fold_deltas)),
        "low_support_delta": low_support_delta,
    }
    selected_arm = max(arms, key=lambda item: arms[item]["oof_r2"])
    selected = arms[selected_arm]
    pass_gate = bool(
        selected["delta_vs_parent_oof"] >= cfg.min_clean_oof_delta
        and selected["nonnegative_folds"] >= cfg.min_nonnegative_folds
        and (selected["low_support_delta"] is None or selected["low_support_delta"] >= cfg.max_low_support_loss)
    )

    target_test = test[test["target_type"] == target].copy()
    test_pivot = pivot_predictions(test.rename(columns={"base_prediction": "prediction"}), "prediction")
    nearest_test = nearest_similarity(target_test["canonical"].astype(str).tolist(), train_pool)
    test_x, _ = make_features(target_test["canonical"].astype(str).to_numpy(), target, test_pivot, nearest_test)
    base_values = target_test["base_prediction"].to_numpy(float)
    if pass_gate:
        if selected_arm == "mean3":
            full_deltas = []
            for name in ("ridge10", "ridge100", "huber"):
                model = make_model(name)
                model.fit(x, residual)
                full_deltas.append(np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -cfg.residual_clip, cfg.residual_clip))
            raw_delta = np.mean(np.column_stack(full_deltas), axis=1)
        else:
            model = make_model(selected_arm)
            model.fit(x, residual)
            raw_delta = np.clip(np.asarray(model.predict(test_x), dtype=np.float64), -cfg.residual_clip, cfg.residual_clip)
        overlay = base_values + cfg.blend_weight * raw_delta
        changed = np.ones(len(base_values), dtype=bool)
    else:
        overlay = base_values.copy()
        changed = np.zeros(len(base_values), dtype=bool)
    report = {
        "target": target,
        "parent_oof_r2": parent_r2,
        "arms": arms,
        "selected_arm": selected_arm,
        "selected_oof_r2": float(selected["oof_r2"]),
        "selected_delta_vs_parent_oof": float(selected["delta_vs_parent_oof"]),
        "clean_oof_gate_pass": pass_gate,
        "gate": {
            "min_clean_oof_delta": cfg.min_clean_oof_delta,
            "min_nonnegative_folds": cfg.min_nonnegative_folds,
            "max_low_support_loss": cfg.max_low_support_loss,
            "residual_clip": cfg.residual_clip,
            "blend_weight": cfg.blend_weight,
        },
        "train_rows": int(len(target_oof)),
        "test_rows": int(len(target_test)),
        "changed_rows": int(np.sum(changed)),
        "changed_ids_sha256": hashlib.sha256(
            json.dumps(target_test.loc[changed, "id"].astype(int).tolist(), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "feature_names": feature_names,
    }
    return overlay, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--oof-csv", default=DEFAULT_OOF)
    parser.add_argument("--targets", default="eea")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    base_path = Path(args.base_csv).resolve()
    oof_path = Path(args.oof_csv).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path in (base_path, oof_path):
        guard_path(path)
    for path in (output, manifest):
        guard_path(path, allow_output=True)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "without_archive" not in str(output):
        raise RuntimeError(f"C330 output must live in without_archive namespace: {output}")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    train, test, oof, base, inputs = load_inputs(data_dir, base_path, oof_path)
    ids = test["id"].to_numpy(int)
    result = base["target"].to_numpy(float).copy()
    target_reports: dict[str, Any] = {}
    for target in parse_targets(args.targets):
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
        "schema_version": "ppp.round2.c330.noarchive-eea-cotest-meta-blend05.v1",
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
        "method": "EEA-only co-test residual meta-calibrator over C327 with clean-OOF-selected blend 0.5",
        "targets_requested": list(parse_targets(args.targets)),
        "target_reports": target_reports,
        "inputs": {
            **inputs,
            "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        },
        "rows": {"train": int(len(train)), "test": int(len(test)), "oof": int(len(oof))},
        "final_notebook_note": "development script reads a stored C282 OOF artifact; a final notebook must regenerate the OOF and base components from official inputs rather than reading this artifact",
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
