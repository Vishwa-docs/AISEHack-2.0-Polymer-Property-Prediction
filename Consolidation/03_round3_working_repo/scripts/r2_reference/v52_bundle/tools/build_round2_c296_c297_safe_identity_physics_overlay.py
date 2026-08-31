#!/usr/bin/env python3
"""C296/C297 safe official-partner identity/physics overlay.

The generator is local_eval-free.  It reads official Round 2 train/test files and,
for the archive branch only, the official bundled archive/train.csv label pool.
It starts from a frozen branch base CSV, then applies fixed target-level
calibrated identities only where partner properties are available from official
labels or, optionally, as co-test base predictions for the same canonical
polymer.

This intentionally avoids the unsafe iterative fallback used by older scratch
cross-property experiments: missing partner labels fall back to the
structure-only/base prediction lane, not to a freshly propagated cross-property
prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


TARGETS = tuple(reference.TARGETS)
DEFAULT_BASE = {
    "with_archive": "experiments/final_submission_runs/with_archive/R2-F25-IONIC-COTEST-OVERLAY-with_archive-20260807.csv",
    "without_archive": "experiments/final_submission_runs/without_archive/R2-C292-FAST-LINEAR-XPROP-OVERLAY-w025-without_archive-20260808.csv",
}
DEFAULT_TARGETS = {
    "with_archive": ("ei", "eea", "eps"),
    "without_archive": ("ei", "eea", "egb", "eps"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden input path: {path}")


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"]:
        raise RuntimeError(f"Invalid base columns: {path}")
    if len(frame) != len(ids):
        raise RuntimeError(f"Invalid base row count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Invalid non-finite base predictions: {path}")
    return frame


def load_branch_inputs(data_dir: Path, branch: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    for path in (train_path, test_path):
        guard_path(path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if list(train.columns) != ["smiles", "target", "target_type"]:
        raise RuntimeError("Unexpected train.csv schema")
    if list(test.columns) != ["id", "smiles", "target_type"]:
        raise RuntimeError("Unexpected test.csv schema")
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError("Unexpected current train/test row count")

    inputs: dict[str, Any] = {
        "train.csv": {"path": str(train_path), "sha256": sha256_file(train_path), "bytes": train_path.stat().st_size},
        "test.csv": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
    }
    if inputs["train.csv"]["sha256"] != reference.EXPECTED_HASHES["train.csv"]:
        raise RuntimeError("train.csv hash mismatch")
    if inputs["test.csv"]["sha256"] != reference.EXPECTED_HASHES["test.csv"]:
        raise RuntimeError("test.csv hash mismatch")

    frames = [train.copy()]
    archive = train.iloc[0:0].copy()
    if branch == "with_archive":
        archive_path = data_dir / "archive" / "train.csv"
        guard_path(archive_path)
        archive = pd.read_csv(archive_path)
        if list(archive.columns) != ["smiles", "target", "target_type"] or len(archive) != 6171:
            raise RuntimeError("Unexpected archive/train.csv schema or row count")
        archive_hash = sha256_file(archive_path)
        if archive_hash != reference.EXPECTED_HASHES["archive/train.csv"]:
            raise RuntimeError("archive/train.csv hash mismatch")
        inputs["archive/train.csv"] = {
            "path": str(archive_path),
            "sha256": archive_hash,
            "bytes": archive_path.stat().st_size,
        }
        frames.append(archive.copy())

    for frame in frames + [test]:
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
        frame["canonical"] = [reference.canonicalize(value) for value in frame["smiles"]]
    if set(train["target_type"].astype(str).str.lower()) != set(TARGETS):
        raise RuntimeError("Unexpected train target set")
    if set(test["target_type"]) != set(TARGETS):
        raise RuntimeError("Unexpected test target set")
    if test["id"].duplicated().any() or not np.array_equal(test["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    label_rows = pd.concat(
        [frame[["canonical", "target_type", "target"]].copy() for frame in frames],
        ignore_index=True,
    )
    return label_rows, test, archive, inputs


def make_wide(label_rows: pd.DataFrame) -> pd.DataFrame:
    return label_rows.pivot_table(index="canonical", columns="target_type", values="target", aggfunc="median")


def formula_value(target: str, values: dict[str, float]) -> float:
    if target == "ei":
        return float(values["eea"] + values["egc"])
    if target == "eea":
        return float(values["ei"] - values["egc"])
    if target == "egb":
        return float(values["egc"])
    if target == "eps":
        return float(values["nc"] ** 2)
    if target == "nc":
        return float(math.sqrt(max(values["eps"], 1.0)))
    raise KeyError(target)


PARTNERS = {
    "ei": ("eea", "egc"),
    "eea": ("ei", "egc"),
    "egb": ("egc",),
    "eps": ("nc",),
    "nc": ("eps",),
}


def model_factory(target: str) -> Any:
    if target in {"ei", "eea", "egb", "eps", "nc"}:
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000))
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def fit_formula_model(
    wide: pd.DataFrame,
    target: str,
) -> tuple[Any, dict[str, Any], tuple[float, float]]:
    partners = PARTNERS[target]
    required = (target, *partners)
    if not all(column in wide.columns for column in required):
        raise RuntimeError(f"Missing required columns for {target}: {required}")
    frame = wide[list(required)].dropna().copy()
    if len(frame) < 30:
        raise RuntimeError(f"Insufficient formula support for {target}: {len(frame)}")
    y = frame[target].to_numpy(float)
    x_raw = np.asarray(
        [formula_value(target, {partner: float(row[partner]) for partner in partners}) for _, row in frame.iterrows()],
        dtype=np.float64,
    ).reshape(-1, 1)
    groups = frame.index.astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x_raw, y, groups=groups)):
        model = model_factory(target)
        model.fit(x_raw[tr], y[tr])
        oof[va] = np.asarray(model.predict(x_raw[va]), dtype=np.float64)
    if not np.isfinite(oof).all():
        raise RuntimeError(f"Non-finite OOF formula predictions for {target}")
    full_model = model_factory(target)
    full_model.fit(x_raw, y)
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1.0e-8)
    report = {
        "rows": int(len(frame)),
        "raw_formula_r2": float(r2_score(y, x_raw[:, 0])),
        "calibrated_group_oof_r2": float(r2_score(y, oof)),
        "mae": float(np.mean(np.abs(y - oof))),
        "partners": list(partners),
        "clip_low": float(q01 - 2.0 * margin),
        "clip_high": float(q99 + 2.0 * margin),
    }
    return full_model, report, (float(q01 - 2.0 * margin), float(q99 + 2.0 * margin))


def parse_targets(value: str | None, branch: str) -> tuple[str, ...]:
    if value is None or value.strip().lower() in {"", "default"}:
        return DEFAULT_TARGETS[branch]
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = [target for target in targets if target not in PARTNERS]
    if invalid:
        raise RuntimeError(f"Invalid overlay targets: {invalid}")
    return targets


def target_weights(target: str, observed_scale: float, cotest_scale: float) -> tuple[float, float]:
    # Fixed before post-freeze scoring.  Larger weights are reserved for the
    # high-identity electronic targets; EPS/NC are deliberately more cautious.
    base = {
        "ei": (0.85, 0.25),
        "eea": (0.85, 0.25),
        "egb": (0.55, 0.20),
        "eps": (0.45, 0.20),
        "nc": (0.20, 0.10),
    }[target]
    return (float(np.clip(base[0] * observed_scale, 0.0, 1.0)), float(np.clip(base[1] * cotest_scale, 0.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--targets", default=None, help="comma-separated overlay targets, or default")
    parser.add_argument("--disable-cotest", action="store_true")
    parser.add_argument("--observed-weight-scale", type=float, default=1.0)
    parser.add_argument("--cotest-weight-scale", type=float, default=1.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    base_path = Path(args.base_csv or DEFAULT_BASE[args.branch]).resolve()
    for path in (data_dir, output, manifest, base_path):
        guard_path(path)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    active_targets = parse_targets(args.targets, args.branch)
    label_rows, test, archive, inputs = load_branch_inputs(data_dir, args.branch)
    ids = test["id"].to_numpy(int)
    base = load_base(base_path, ids)
    wide = make_wide(label_rows)

    test = test.copy()
    test["base_prediction"] = base["target"].to_numpy(float)
    base_pivot = test.pivot_table(index="canonical", columns="target_type", values="base_prediction", aggfunc="mean")

    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    clips: dict[str, tuple[float, float]] = {}
    for target in active_targets:
        model, report, clip = fit_formula_model(wide, target)
        models[target] = model
        reports[target] = report
        clips[target] = clip

    result = test["base_prediction"].to_numpy(float).copy()
    overlay_reports: dict[str, Any] = {
        target: {
            "changed_rows": 0,
            "observed_label_rows": 0,
            "cotest_or_mixed_rows": 0,
            "observed_weight": target_weights(target, args.observed_weight_scale, args.cotest_weight_scale)[0],
            "cotest_weight": target_weights(target, args.observed_weight_scale, args.cotest_weight_scale)[1],
        }
        for target in active_targets
    }

    for row_pos, row in test.iterrows():
        target = str(row["target_type"]).lower()
        if target not in active_targets:
            continue
        canonical = str(row["canonical"])
        values: dict[str, float] = {}
        source_types: list[str] = []
        supported = True
        for partner in PARTNERS[target]:
            observed_value = np.nan
            if canonical in wide.index and partner in wide.columns:
                observed_value = wide.loc[canonical].get(partner, np.nan)
            if pd.notna(observed_value):
                values[partner] = float(observed_value)
                source_types.append("official_observed")
                continue
            cotest_value = np.nan
            if (not args.disable_cotest) and canonical in base_pivot.index and partner in base_pivot.columns:
                cotest_value = base_pivot.loc[canonical].get(partner, np.nan)
            if pd.notna(cotest_value):
                values[partner] = float(cotest_value)
                source_types.append("cotest_base_prediction")
                continue
            supported = False
            break
        if not supported:
            continue
        x_value = formula_value(target, values)
        pred = float(models[target].predict(np.asarray([[x_value]], dtype=np.float64))[0])
        low, high = clips[target]
        pred = float(np.clip(pred, low, high))
        observed_weight, cotest_weight = target_weights(target, args.observed_weight_scale, args.cotest_weight_scale)
        if all(item == "official_observed" for item in source_types):
            weight = observed_weight
            overlay_reports[target]["observed_label_rows"] += 1
        else:
            weight = cotest_weight
            overlay_reports[target]["cotest_or_mixed_rows"] += 1
        result[int(row_pos)] = (1.0 - weight) * result[int(row_pos)] + weight * pred
        overlay_reports[target]["changed_rows"] += 1

    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite output")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.safe-identity-physics-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "active_targets": list(active_targets),
        "official_only_generation": True,
        "archive_labels_used_by_builder": args.branch == "with_archive",
        "archive_labels_used_by_base": args.branch == "with_archive",
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "unsafe_iterative_prediction_fallback_used": False,
        "missing_partner_fallback": "leave_base_prediction_unchanged",
        "cotest_base_predictions_enabled": not args.disable_cotest,
        "config": vars(args),
        "inputs": {
            **inputs,
            "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        },
        "formula_reports": reports,
        "overlay_reports": overlay_reports,
        "rows": {
            "test": int(len(test)),
            "official_label_rows": int(len(label_rows)),
            "archive_rows_read": int(len(archive)) if args.branch == "with_archive" else 0,
            "wide_structures": int(len(wide)),
        },
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"], "overlays": overlay_reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
