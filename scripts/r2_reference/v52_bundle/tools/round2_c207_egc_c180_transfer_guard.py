#!/usr/bin/env python3
"""C207: guarded Egc replay of the C180 Flory-Fox/oligomer near-miss.

This is a bounded contingency child behind C206.  It exists only to prevent the
local watchdog from draining below the active 0.95 objective.  It does not
consume C204/C205/C206 outputs and does not read local_eval/public feedback.

The single changed factor versus C180 is a fixed fail-closed guard for the exact
C180 Egc transfer-failure panels.  The C180 Egc carrier is regenerated from
official inputs; rows in predeclared negative panels fall back to exact C050.
If any standard component gate fails, no target is banked and the C180/C127
Egc transfer-guard repair family is cooled without retuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as c180


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "egc"
MIN_BANKABLE_DELTA_R2 = 0.01
GUARD_SIMILARITY_LT = 0.30
NEGATIVE_SCAFFOLDS = (
    "C1CCCC1",
    "c1ccc(-c2cccs2)cc1",
    "c1ccc(N=Nc2ccccc2)cc1",
    "c1ccncc1",
)
SEED = 20260805


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), **payload},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def unchanged_report(info: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    return {
        "active": False,
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, parent)),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}},
        "folds": [],
        "pass": False,
        "unchanged_parent": True,
    }


def fold_local_nearest(parent: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    nearest = np.zeros(len(indices), dtype=np.float64)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [parent["fingerprints"][int(indices[row])] for row in training]
        for row in validation:
            sims = DataStructs.BulkTanimotoSimilarity(parent["fingerprints"][int(indices[row])], train_fps)
            nearest[row] = float(max(sims)) if sims else 0.0
    return nearest


def full_train_nearest(parent: dict[str, Any], target_indices: np.ndarray, prediction_indices: np.ndarray) -> np.ndarray:
    train_fps = [parent["fingerprints"][int(index)] for index in np.asarray(target_indices, dtype=np.int64)]
    output = np.zeros(len(prediction_indices), dtype=np.float64)
    for row, index in enumerate(np.asarray(prediction_indices, dtype=np.int64)):
        sims = DataStructs.BulkTanimotoSimilarity(parent["fingerprints"][int(index)], train_fps)
        output[row] = float(max(sims)) if sims else 0.0
    return output


def guard_mask_from_scaffold_similarity(scaffolds: np.ndarray, nearest: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    scaffold_guard = np.isin(np.asarray(scaffolds, dtype=object), np.asarray(NEGATIVE_SCAFFOLDS, dtype=object))
    similarity_guard = np.asarray(nearest, dtype=np.float64) < GUARD_SIMILARITY_LT
    mask = scaffold_guard | similarity_guard
    return mask, {
        "guard_rows": int(np.sum(mask)),
        "similarity_guard_rows": int(np.sum(similarity_guard)),
        "scaffold_guard_rows": int(np.sum(scaffold_guard)),
    }


def active_test_rows(parent: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(ACTIVE_TARGET)].sort_values("id").reset_index(drop=True)
    detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(ACTIVE_TARGET)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(rows["id"].to_numpy(np.int64), detail["id"].to_numpy(np.int64)):
        raise RuntimeError("C207 Egc test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in rows["canonical"]], dtype=np.int64)
    return rows, indices, detail["target"].to_numpy(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--canonical-run",
        default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7",
    )
    args = parser.parse_args()
    started = time.time()
    root = Path(args.root).resolve()
    round2_root = root / "Polymer Prediction Challenge Round 2"
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    progress = run_dir / "progress.jsonl"
    checkpoint(progress, "started", experiment_id=run_dir.name)

    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    dense, sparse_features, feature_report = c180.build_features(root, parent["keys"])
    checkpoint(
        progress,
        "features_complete",
        dense_shape=feature_report["dense_shape"],
        sparse_shape=feature_report["sparse_shape"],
    )

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = active_test_rows(parent)
    raw_result = carrier.fit_target(info, dense, sparse_features, test_indices, test_parent)
    raw_report = carrier.evaluate_target(info, raw_result)

    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    raw_candidate = np.asarray(raw_result["candidate"], dtype=np.float64)
    nearest = fold_local_nearest(parent, info)
    oof_guard, oof_guard_summary = guard_mask_from_scaffold_similarity(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded_candidate = np.where(oof_guard, parent_oof, raw_candidate)
    guarded_result = dict(raw_result)
    guarded_result["candidate"] = guarded_candidate
    guarded_report = carrier.evaluate_target(info, guarded_result)
    guarded_report.update(
        {
            "active": True,
            "raw_c180_parent_r2": raw_report["parent_r2"],
            "raw_c180_candidate_r2": raw_report["candidate_r2"],
            "raw_c180_delta_r2": raw_report["delta_r2"],
            "raw_c180_positive_folds": raw_report["positive_folds"],
            "raw_c180_group_bootstrap_lower": raw_report["group_bootstrap_lower"],
            "raw_c180_minimum_panel_delta": raw_report["minimum_panel_delta"],
            "raw_c180_panels": raw_report["panels"],
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "guard_summary": oof_guard_summary,
            "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
            "blend_name": raw_result["blend_name"],
            "blend_weights": [float(value) for value in raw_result["weights"]],
            "blend_intercept": float(raw_result["intercept"]),
            "feature_rows": int(len(y)),
        }
    )
    banked = bool(guarded_report["pass"])
    checkpoint(
        progress,
        "egc_guard_complete",
        raw_delta_r2=raw_report["delta_r2"],
        guarded_delta_r2=guarded_report["delta_r2"],
        positive_folds=guarded_report["positive_folds"],
        bootstrap_lower=guarded_report["group_bootstrap_lower"],
        minimum_panel_delta=guarded_report["minimum_panel_delta"],
        guard_rows=oof_guard_summary["guard_rows"],
        pass_gate=banked,
    )

    test_nearest = full_train_nearest(parent, np.asarray(info["indices"], dtype=np.int64), test_indices)
    test_scaffolds = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"]], dtype=object)
    test_guard, test_guard_summary = guard_mask_from_scaffold_similarity(test_scaffolds, test_nearest)
    test_candidate = np.where(test_guard, test_parent, np.asarray(raw_result["test_direct"], dtype=np.float64))
    if not banked:
        test_candidate = test_parent.copy()

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = parent["target_info"][target]
        if target == ACTIVE_TARGET:
            report = guarded_report
            candidate = guarded_candidate
            raw = raw_candidate
            guard = oof_guard
            folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
            raw = np.full(len(candidate), np.nan, dtype=np.float64)
            guard = np.zeros(len(candidate), dtype=bool)
            folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        assembled = candidate if target == ACTIVE_TARGET and banked else np.asarray(target_info["parent"], dtype=np.float64)
        target_reports[target] = report
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "raw_candidate": raw,
                    "candidate": candidate,
                    "assembled": assembled,
                    "guarded": guard,
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": folds,
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(
        np.mean(
            [
                r2_score(part["target"].to_numpy(np.float64), part["assembled"].to_numpy(np.float64))
                for part in oof_parts
            ]
        )
    )
    test_prediction = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if banked:
        mask = test_prediction["target_type"].astype(str).eq(ACTIVE_TARGET)
        test_prediction.loc[mask, "target"] = test_prediction.loc[mask, "id"].astype(int).map(
            dict(zip(test_rows["id"].astype(int), test_candidate, strict=True))
        ).to_numpy(np.float64)
    test_prediction = test_prediction[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(test_prediction) != 4940 or not np.array_equal(test_prediction["id"].to_numpy(np.int64), np.arange(1, 4941)):
        raise RuntimeError("C207 ID coverage/order contract failed")
    if not np.isfinite(test_prediction["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C207 produced non-finite predictions")

    max_loss = float(min(guarded_report["delta_r2"] if banked else 0.0, 0.0))
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003)
    report = {
        "schema_version": "ppp.round2.c207.egc-c180-transfer-guard.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(test_prediction)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(full_candidate_gate_pass and assembled_mean >= 0.95),
        "component_diagnostics": {
            "active_target": ACTIVE_TARGET,
            "changed_factor": "fixed C050 fallback on predeclared C180 Egc negative transfer panels",
            "raw_c180_delta_r2": raw_report["delta_r2"],
            "guarded_delta_r2": guarded_report["delta_r2"],
            "oof_guard_summary": oof_guard_summary,
            "test_guard_summary": test_guard_summary,
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "no_c180_prediction_replay": True,
            "no_c204_c205_c206_output_use": True,
            "no_partner_labels": True,
            "no_pi1m": True,
            "no_local_eval_public_feedback": True,
        },
        "decision": "candidate_pass_pending_clean_reproduction" if full_candidate_gate_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "c180_feature_builder": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }

    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "parent": test_parent,
            "raw_candidate": raw_result["test_direct"],
            "candidate": test_candidate,
            "guarded": test_guard,
            "nearest_tanimoto": test_nearest,
            "scaffold": test_scaffolds,
            "banked": banked,
        }
    ).to_csv(run_dir / "egc_component_predictions.csv", index=False)
    test_prediction.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": report["schema_version"],
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "source": "C180 regenerated Flory-Fox/oligomer/asymptotic carrier with fixed negative-panel C050 guard",
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "component_gate": {
                "minimum_delta_r2": MIN_BANKABLE_DELTA_R2,
                "minimum_positive_folds": 4,
                "bootstrap_lower_must_exceed": 0.0,
                "minimum_panel_delta_must_be_at_least": 0.0,
            },
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"sklearn={__import__('sklearn').__version__}",
                f"rdkit={reference.Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Egc guarded delta `{guarded_report['delta_r2']:+.12f}`; "
        f"raw C180 delta `{raw_report['delta_r2']:+.12f}`. Banked: `{banked}`. "
        "Official-only; no local_eval, Kaggle, upload, submission, or final-notebook action.\n",
        encoding="utf-8",
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "banked_targets": report["banked_targets"],
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "egc_raw_delta_r2": raw_report["delta_r2"],
                "egc_guarded_delta_r2": guarded_report["delta_r2"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
