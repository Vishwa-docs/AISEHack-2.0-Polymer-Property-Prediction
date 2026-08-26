#!/usr/bin/env python3
"""C349 archive EI identity-residual route over C334.

Hypothesis: on the archive-enabled branch, EI can be improved for structures
with official, conflict-free EEA and Egc labels by using the physical identity
EI ~= EEA + Egc plus a small fold-nested residual.  Unsupported rows are exact
C334 fallback.

This builder reads only official Round 2 files plus the local C050 OOF artifact
used for clean gating.  It reads no local_eval/external_label files and performs no Kaggle
action.  Final-notebook eligibility would require regenerating the C050 parent
OOF/base path from official inputs inside the notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

DEFAULT_BASE = (
    "experiments/final_submission_runs/with_archive/"
    "R2-C334-ARCHIVE-TARGET-SPLICE-C333-EEA-EPS-C332-NC-20260808.csv"
)
DEFAULT_OOF = (
    "experiments/CLEAN_OFFICIAL_ONLY/"
    "R2-C050-20260803-2130-mixed-c001-gap-components-v7/oof_predictions.csv"
)
TARGET = "ei"
PARTNERS = ("eea", "egc")
CONFLICT_TOL = 1.0e-8


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    blend_weight: float
    alpha: float = 10.0


ARMS = (
    Arm("identity_w100", "identity", 1.00),
    Arm("identity_w075", "identity", 0.75),
    Arm("identity_w050", "identity", 0.50),
    Arm("median_resid_w075", "median", 0.75),
    Arm("median_resid_w050", "median", 0.50),
    Arm("ridge10_w075", "ridge", 0.75, 10.0),
    Arm("ridge10_w050", "ridge", 0.50, 10.0),
    Arm("ridge100_w075", "ridge", 0.75, 100.0),
    Arm("ridge100_w050", "ridge", 0.50, 100.0),
    Arm("huber_w050", "huber", 0.50),
)


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
    if not allow_output and "without_archive" in low:
        raise RuntimeError(f"Refusing no-archive path in archive EI route: {path}")
    if allow_output and "without_archive" in low:
        raise RuntimeError(f"Refusing cross-branch output: {path}")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def no_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path)
    if "with_archive" not in str(path):
        raise RuntimeError(f"C349 base must be with_archive: {path}")
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
    archive_path = data_dir / "archive" / "train.csv"
    for path in (train_path, test_path, archive_path, base_path, oof_path):
        guard_path(path)
    inputs = {
        "train.csv": {"path": str(train_path), "sha256": sha256_file(train_path), "bytes": train_path.stat().st_size},
        "test.csv": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "archive/train.csv": {"path": str(archive_path), "sha256": sha256_file(archive_path), "bytes": archive_path.stat().st_size},
        "c050_oof_predictions.csv": {
            "path": str(oof_path),
            "sha256": sha256_file(oof_path),
            "bytes": oof_path.stat().st_size,
            "role": "local development OOF artifact generated from official current+archive inputs; final notebook must regenerate it",
        },
        "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
    }
    for name, expected in (
        ("train.csv", reference.EXPECTED_HASHES["train.csv"]),
        ("test.csv", reference.EXPECTED_HASHES["test.csv"]),
        ("archive/train.csv", reference.EXPECTED_HASHES["archive/train.csv"]),
    ):
        if inputs[name]["sha256"] != expected:
            raise RuntimeError(f"{name} hash mismatch")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    archive = pd.read_csv(archive_path)
    oof_raw = pd.read_csv(oof_path)
    required = {"canonical", "target_type", "target", "candidate_prediction"}
    if not required.issubset(set(oof_raw.columns)):
        raise RuntimeError("Unexpected C050 OOF schema")
    for frame in (train, test, archive, oof_raw):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
    train = train.copy()
    test = test.copy()
    archive = archive.copy()
    oof = oof_raw[["canonical", "target_type", "target", "candidate_prediction"]].rename(
        columns={"candidate_prediction": "prediction"}
    )
    train["canonical"] = [canonical_smiles(value) for value in train["smiles"]]
    test["canonical"] = [canonical_smiles(value) for value in test["smiles"]]
    archive["canonical"] = [canonical_smiles(value) for value in archive["smiles"]]
    if set(train["target_type"]) != set(reference.TARGETS) or set(test["target_type"]) != set(reference.TARGETS):
        raise RuntimeError("Unexpected target set")
    if test["id"].duplicated().any() or not np.array_equal(test["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    target_oof_rows = len(oof[oof["target_type"] == TARGET])
    target_train_rows = len(train[train["target_type"] == TARGET])
    if target_oof_rows < max(20, int(0.95 * target_train_rows)):
        raise RuntimeError(f"C050 OOF EI coverage too low: {target_oof_rows}/{target_train_rows}")
    base = load_base(base_path, test["id"].to_numpy(int))
    test["base_prediction"] = base["target"].to_numpy(float)
    return train, test, archive, oof, inputs


def conflict_free_maps(train: pd.DataFrame, archive: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    pool = pd.concat([train.assign(source="current"), archive.assign(source="archive")], ignore_index=True)
    grouped = (
        pool.groupby(["canonical", "target_type"], sort=True)["target"]
        .agg(["count", "median", "min", "max"])
        .reset_index()
    )
    grouped["range"] = grouped["max"] - grouped["min"]
    maps: dict[str, dict[str, float]] = {}
    report: dict[str, Any] = {}
    for target in ("egc", "egb", "eea", "ei"):
        rows = grouped[(grouped["target_type"] == target) & (grouped["range"].abs() <= CONFLICT_TOL)].copy()
        maps[target] = dict(zip(rows["canonical"], rows["median"].astype(float), strict=True))
        report[target] = {
            "conflict_free_canonical_count": int(len(rows)),
            "conflicted_canonical_count": int(np.sum((grouped["target_type"] == target) & (grouped["range"].abs() > CONFLICT_TOL))),
        }
    return maps, report


def make_fingerprints(canonicals: list[str]) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps: dict[str, Any] = {}
    for canonical in sorted(set(canonicals)):
        mol = Chem.MolFromSmiles(str(canonical))
        if mol is None:
            continue
        fps[str(canonical)] = generator.GetFingerprint(mol)
    return fps


def nearest_similarity(query: list[str], train_pool: list[str]) -> dict[str, float]:
    fps = make_fingerprints(query + train_pool)
    train_fps = [fps[value] for value in train_pool if value in fps]
    if not train_fps:
        return {str(value): 0.0 for value in query}
    out: dict[str, float] = {}
    for value in query:
        fp = fps.get(str(value))
        if fp is None:
            out[str(value)] = 0.0
        else:
            out[str(value)] = float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return out


def feature_frame(canonicals: np.ndarray, maps: dict[str, dict[str, float]], nearest: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for canonical in canonicals:
        c = str(canonical)
        egc = maps["egc"].get(c, np.nan)
        eea = maps["eea"].get(c, np.nan)
        egb = maps["egb"].get(c, np.nan)
        identity = eea + egc if np.isfinite(eea) and np.isfinite(egc) else np.nan
        rows.append(
            {
                "egc": egc,
                "eea": eea,
                "egb": egb,
                "has_egb": float(np.isfinite(egb)),
                "identity": identity,
                "egc_minus_eea": egc - eea if np.isfinite(egc) and np.isfinite(eea) else np.nan,
                "egc_minus_egb": egc - egb if np.isfinite(egc) and np.isfinite(egb) else np.nan,
                "identity_minus_egb": identity - egb if np.isfinite(identity) and np.isfinite(egb) else np.nan,
                "egc_sq": egc * egc if np.isfinite(egc) else np.nan,
                "eea_sq": eea * eea if np.isfinite(eea) else np.nan,
                "nearest_same_ei": nearest.get(c, 0.0),
            }
        )
    return pd.DataFrame(rows)


def model_for(arm: Arm) -> Any:
    if arm.kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=arm.alpha))
    if arm.kind == "huber":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000),
        )
    raise RuntimeError(f"No sklearn model for arm {arm}")


def fit_predict_arm(
    arm: Arm,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    identity_train: np.ndarray,
    x_pred: pd.DataFrame,
    identity_pred: np.ndarray,
) -> np.ndarray:
    residual_train = y_train - identity_train
    if arm.kind == "identity":
        return identity_pred.copy()
    if arm.kind == "median":
        return identity_pred + float(np.median(residual_train))
    model = model_for(arm)
    model.fit(x_train, residual_train)
    return identity_pred + np.asarray(model.predict(x_pred), dtype=np.float64)


def evaluate_oof(oof: pd.DataFrame, maps: dict[str, dict[str, float]]) -> tuple[dict[str, Any], pd.DataFrame]:
    target_oof = oof[oof["target_type"] == TARGET].reset_index(drop=True)
    y = target_oof["target"].to_numpy(float)
    parent = target_oof["prediction"].to_numpy(float)
    canonicals = target_oof["canonical"].astype(str).to_numpy()
    groups = np.asarray([no_stereo(value) for value in canonicals], dtype=object)
    train_pool = target_oof["canonical"].astype(str).tolist()
    nearest = nearest_similarity(train_pool, train_pool)
    x = feature_frame(canonicals, maps, nearest)
    support = x["egc"].notna().to_numpy() & x["eea"].notna().to_numpy()
    identity = x["identity"].to_numpy(float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    folds = np.full(len(y), -1, dtype=np.int64)
    for fold, (_, va) in enumerate(splitter.split(x, y, groups=groups)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError("EI fold assignment failed")
    y_min = float(np.min(y) - 0.25)
    y_max = float(np.max(y) + 0.25)
    parent_r2 = float(r2_score(y, parent))
    arms: dict[str, Any] = {}
    oof_predictions: dict[str, np.ndarray] = {}
    for arm in ARMS:
        pred = parent.copy()
        fold_deltas: list[float] = []
        route_counts: list[int] = []
        for fold in sorted(set(folds)):
            tr = np.flatnonzero((folds != fold) & support)
            va = np.flatnonzero(folds == fold)
            va_support = va[support[va]]
            if len(tr) < 8 or len(va_support) == 0:
                fold_pred = parent[va].copy()
            else:
                raw = fit_predict_arm(arm, x.iloc[tr], y[tr], identity[tr], x.iloc[va_support], identity[va_support])
                raw = np.clip(raw, y_min, y_max)
                fold_pred = parent[va].copy()
                local_parent = parent[va_support]
                fold_pred[np.isin(va, va_support)] = local_parent + arm.blend_weight * (raw - local_parent)
            pred[va] = fold_pred
            fold_deltas.append(float(r2_score(y[va], fold_pred) - r2_score(y[va], parent[va])))
            route_counts.append(int(len(va_support)))
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Non-finite OOF predictions for {arm.name}")
        support_delta = None
        if int(np.sum(support)) >= 10:
            support_delta = float(r2_score(y[support], pred[support]) - r2_score(y[support], parent[support]))
        unsupported_delta = None
        if int(np.sum(~support)) >= 10:
            unsupported_delta = float(r2_score(y[~support], pred[~support]) - r2_score(y[~support], parent[~support]))
        arms[arm.name] = {
            "kind": arm.kind,
            "blend_weight": arm.blend_weight,
            "alpha": arm.alpha,
            "oof_r2": float(r2_score(y, pred)),
            "delta_vs_parent_oof": float(r2_score(y, pred) - parent_r2),
            "fold_deltas": fold_deltas,
            "nonnegative_folds": int(sum(delta >= 0.0 for delta in fold_deltas)),
            "support_delta": support_delta,
            "unsupported_delta": unsupported_delta,
            "route_rows_by_fold": route_counts,
        }
        oof_predictions[arm.name] = pred
    selected_arm = max(arms, key=lambda name: arms[name]["oof_r2"])
    selected = arms[selected_arm]
    pass_gate = bool(
        selected["delta_vs_parent_oof"] >= 0.005
        and selected["nonnegative_folds"] >= 4
        and (selected["support_delta"] is None or selected["support_delta"] >= 0.0)
        and (selected["unsupported_delta"] is None or abs(selected["unsupported_delta"]) <= 1.0e-12)
    )
    oof_out = pd.DataFrame(
        {
            "canonical": canonicals,
            "target_type": TARGET,
            "target": y,
            "parent": parent,
            "candidate": oof_predictions[selected_arm],
            "support": support,
            "outer_fold": folds,
            "identity": identity,
        }
    )
    return {
        "target": TARGET,
        "parent_oof_r2": parent_r2,
        "arms": arms,
        "selected_arm": selected_arm,
        "selected_oof_r2": float(selected["oof_r2"]),
        "selected_delta_vs_parent_oof": float(selected["delta_vs_parent_oof"]),
        "clean_oof_gate_pass": pass_gate,
        "gate": {
            "min_delta": 0.005,
            "min_nonnegative_folds": 4,
            "min_support_delta": 0.0,
            "unsupported_rows_must_be_unchanged": True,
        },
        "train_rows": int(len(target_oof)),
        "supported_oof_rows": int(np.sum(support)),
    }, oof_out


def apply_to_test(test: pd.DataFrame, oof: pd.DataFrame, maps: dict[str, dict[str, float]], report: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    target_test = test[test["target_type"] == TARGET].copy()
    base = target_test["base_prediction"].to_numpy(float)
    canonicals = target_test["canonical"].astype(str).to_numpy()
    train_pool = oof[oof["target_type"] == TARGET]["canonical"].astype(str).tolist()
    nearest = nearest_similarity(canonicals.astype(str).tolist(), train_pool)
    x_test = feature_frame(canonicals, maps, nearest)
    support = x_test["egc"].notna().to_numpy() & x_test["eea"].notna().to_numpy()
    route = support.copy()
    result = base.copy()
    if report["clean_oof_gate_pass"]:
        target_oof = oof[oof["target_type"] == TARGET].reset_index(drop=True)
        y = target_oof["target"].to_numpy(float)
        parent_canonicals = target_oof["canonical"].astype(str).to_numpy()
        nearest_train = nearest_similarity(parent_canonicals.astype(str).tolist(), parent_canonicals.astype(str).tolist())
        x_train = feature_frame(parent_canonicals, maps, nearest_train)
        support_train = x_train["egc"].notna().to_numpy() & x_train["eea"].notna().to_numpy()
        identity_train = x_train["identity"].to_numpy(float)
        identity_test = x_test["identity"].to_numpy(float)
        arm_by_name = {arm.name: arm for arm in ARMS}
        arm = arm_by_name[str(report["selected_arm"])]
        raw = fit_predict_arm(arm, x_train.iloc[support_train], y[support_train], identity_train[support_train], x_test.iloc[route], identity_test[route])
        raw = np.clip(raw, float(np.min(y) - 0.25), float(np.max(y) + 0.25))
        result[route] = base[route] + arm.blend_weight * (raw - base[route])
    else:
        route[:] = False
    return result, {
        "test_rows": int(len(target_test)),
        "supported_test_rows": int(np.sum(support)),
        "changed_rows": int(np.sum(route)),
        "changed_ids_sha256": hashlib.sha256(
            json.dumps(target_test.loc[route, "id"].astype(int).tolist(), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "unsupported_max_abs_change": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--oof-csv", default=DEFAULT_OOF)
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
    if "with_archive" not in str(output):
        raise RuntimeError(f"C349 output must live in with_archive namespace: {output}")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    train, test, archive, oof, inputs = load_inputs(data_dir, base_path, oof_path)
    maps, map_report = conflict_free_maps(train, archive)
    report, oof_out = evaluate_oof(oof, maps)
    overlay, test_report = apply_to_test(test, oof, maps, report)
    ids = test["id"].to_numpy(int)
    result = test["base_prediction"].to_numpy(float).copy()
    mask = test["target_type"].to_numpy(str) == TARGET
    if int(np.sum(mask)) != len(overlay):
        raise RuntimeError("EI target alignment failed")
    result[mask] = overlay
    if not np.isfinite(result).all():
        raise RuntimeError("Output contains non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.c349.archive-ei-identity-residual.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "official_current_train_used": True,
        "archive_labels_used": True,
        "archive_file_read": True,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "archive EI equals official EEA plus Egc with nested residual, unsupported rows fallback to C334",
        "target_reports": {TARGET: {**report, **test_report}},
        "conflict_free_label_maps": map_report,
        "inputs": inputs,
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": int(len(oof))},
        "final_notebook_note": "development script reads stored C050 OOF/base artifacts; final notebook must regenerate these from official inputs rather than reading local artifacts",
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    record["oof_selected_preview"] = {
        "sha256": hashlib.sha256(oof_out.to_csv(index=False).encode("utf-8")).hexdigest(),
        "rows": int(len(oof_out)),
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": record["output"],
                "target_report": {
                    "selected_arm": report["selected_arm"],
                    "parent_oof_r2": report["parent_oof_r2"],
                    "selected_oof_r2": report["selected_oof_r2"],
                    "selected_delta_vs_parent_oof": report["selected_delta_vs_parent_oof"],
                    "clean_oof_gate_pass": report["clean_oof_gate_pass"],
                    "supported_oof_rows": report["supported_oof_rows"],
                    "supported_test_rows": test_report["supported_test_rows"],
                    "changed_rows": test_report["changed_rows"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
