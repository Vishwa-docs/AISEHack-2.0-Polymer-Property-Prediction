#!/usr/bin/env python3
"""C332 archive weak-target C050 co-test residual meta-calibrator.

Archive-branch analogue of C327.  It starts from the frozen F25 archive
compound, uses the C050 archive-enabled OOF table as development evidence, and
tries whole-target residual overlays for EI/EEA/EPS/NC only.  Targets that fail
the clean OOF gate remain unchanged from F25.

This builder reads no local_eval/external_label files and performs no Kaggle action.  It
does read a stored local C050 OOF artifact; a final notebook would need to
regenerate that artifact from official inputs in the same run.
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

import build_round2_c327_noarchive_cotest_meta_calibrator as c327
import initial_reference_pipeline as reference


DEFAULT_BASE = (
    "experiments/final_submission_runs/with_archive/"
    "R2-F25-IONIC-COTEST-OVERLAY-with_archive-20260807.csv"
)
DEFAULT_OOF = (
    "experiments/CLEAN_OFFICIAL_ONLY/"
    "R2-C050-20260803-2130-mixed-c001-gap-components-v7/oof_predictions.csv"
)
ACTIVE_TARGETS = ("ei", "eea", "eps", "nc")


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
        raise RuntimeError(f"Refusing cross-branch input for C332: {path}")


def canonical_smiles(smiles: str) -> str:
    return reference.canonicalize(smiles)


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    invalid = [target for target in targets if target not in ACTIVE_TARGETS]
    if invalid:
        raise RuntimeError(f"C332 only supports {ACTIVE_TARGETS}, got {invalid}")
    return targets


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path)
    if "with_archive" not in str(path):
        raise RuntimeError(f"C332 base must be branch-local with_archive: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def load_inputs(
    data_dir: Path,
    base_path: Path,
    oof_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    archive_path = data_dir / "archive" / "train.csv"
    for path in (train_path, test_path, archive_path, base_path, oof_path):
        guard_path(path)
    inputs = {
        "train.csv": {"path": str(train_path), "sha256": sha256_file(train_path), "bytes": train_path.stat().st_size},
        "test.csv": {"path": str(test_path), "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size},
        "archive/train.csv": {
            "path": str(archive_path),
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "c050_oof_predictions.csv": {
            "path": str(oof_path),
            "sha256": sha256_file(oof_path),
            "bytes": oof_path.stat().st_size,
            "role": "local development OOF artifact generated from official current+archive inputs; final notebook must regenerate it",
        },
    }
    if inputs["train.csv"]["sha256"] != reference.EXPECTED_HASHES["train.csv"]:
        raise RuntimeError("train.csv hash mismatch")
    if inputs["test.csv"]["sha256"] != reference.EXPECTED_HASHES["test.csv"]:
        raise RuntimeError("test.csv hash mismatch")
    if inputs["archive/train.csv"]["sha256"] != reference.EXPECTED_HASHES["archive/train.csv"]:
        raise RuntimeError("archive/train.csv hash mismatch")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    archive = pd.read_csv(archive_path)
    oof_raw = pd.read_csv(oof_path)
    expected_oof_columns = {
        "canonical",
        "target_type",
        "target",
        "baseline_prediction",
        "candidate_prediction",
    }
    if not expected_oof_columns.issubset(set(oof_raw.columns)):
        raise RuntimeError("Unexpected C050 OOF schema")
    oof = oof_raw[["canonical", "target_type", "target", "candidate_prediction"]].rename(
        columns={"candidate_prediction": "prediction"}
    )
    for frame in (train, test, archive, oof):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
    train = train.copy()
    test = test.copy()
    train["canonical"] = [canonical_smiles(value) for value in train["smiles"]]
    test["canonical"] = [canonical_smiles(value) for value in test["smiles"]]
    if set(train["target_type"]) != set(reference.TARGETS) or set(test["target_type"]) != set(reference.TARGETS):
        raise RuntimeError("Unexpected target set")
    if test["id"].duplicated().any() or not np.array_equal(test["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    for target in ACTIVE_TARGETS:
        oof_rows = len(oof[oof["target_type"] == target])
        train_rows = len(train[train["target_type"] == target])
        if oof_rows < max(20, int(0.95 * train_rows)):
            raise RuntimeError(f"C050 OOF row count too low for {target}: {oof_rows}/{train_rows}")
    base = load_base(base_path, test["id"].to_numpy(int))
    test["base_prediction"] = base["target"].to_numpy(float)
    return train, test, archive, oof, base, inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--oof-csv", default=DEFAULT_OOF)
    parser.add_argument("--targets", default="ei,eea,eps,nc")
    parser.add_argument("--min-clean-oof-delta", type=float, default=0.002)
    parser.add_argument("--min-nonnegative-folds", type=int, default=4)
    parser.add_argument("--max-low-support-loss", type=float, default=-0.003)
    parser.add_argument("--blend-scale", type=float, default=1.0)
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
        raise RuntimeError(f"C332 output must live in with_archive namespace: {output}")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    # Archive-target gate settings.  These are intentionally stricter than the
    # C327 EEA-only winner because F25 is already strong on the weak targets.
    c327.ACTIVE_TARGETS = ACTIVE_TARGETS
    min_delta = float(args.min_clean_oof_delta)
    min_folds = int(args.min_nonnegative_folds)
    low_support = float(args.max_low_support_loss)
    scale = float(args.blend_scale)
    c327.CONFIGS.update(
        {
            "ei": c327.TargetConfig(min_delta, min_folds, low_support, 0.45, 0.55 * scale),
            "eea": c327.TargetConfig(min_delta, min_folds, low_support, 0.45, 0.55 * scale),
            "eps": c327.TargetConfig(min_delta, min_folds, low_support, 0.70, 0.55 * scale),
            "nc": c327.TargetConfig(min_delta, min_folds, low_support, 0.080, 0.55 * scale),
        }
    )

    train, test, archive, oof, base, inputs = load_inputs(data_dir, base_path, oof_path)
    ids = test["id"].to_numpy(int)
    result = base["target"].to_numpy(float).copy()
    target_reports: dict[str, Any] = {}
    for target in parse_targets(args.targets):
        overlay, report = c327.evaluate_target(target, oof, test)
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
        "schema_version": "ppp.round2.c332.archive-c050-cotest-meta-calibrator.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "official_current_train_used": True,
        "archive_labels_used": True,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "archive weak-target co-test residual meta-calibrator over F25 with C050 OOF clean gates",
        "config": {
            "min_clean_oof_delta": min_delta,
            "min_nonnegative_folds": min_folds,
            "max_low_support_loss": low_support,
            "blend_scale": scale,
        },
        "targets_requested": list(parse_targets(args.targets)),
        "target_reports": target_reports,
        "inputs": {
            **inputs,
            "base_csv": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        },
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": int(len(oof))},
        "final_notebook_note": "development script reads stored C050 OOF/base artifacts; a final notebook must regenerate these from official inputs rather than reading local artifacts",
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
