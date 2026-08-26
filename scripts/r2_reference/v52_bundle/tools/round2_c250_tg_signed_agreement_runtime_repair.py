#!/usr/bin/env python3
"""C244: fixed Tg signed-agreement median residual stack.

This is a bounded clean continuation after C243.  It regenerates Tg near-miss
mechanisms from official Round 2 inputs, but uses only the two
measurement-noise arms in the deployable consensus rule:

  C050 parent + median(C228 residual, C232 residual)

only where the regenerated C228-style guarded residual and regenerated
C232-style reliability residual share a nonzero sign.  Everywhere else the
prediction falls back exactly to C050.  The C127 direct carrier is regenerated
as a diagnostic arm but is not part of the candidate rule.

There is no weight grid, learned meta-model, stored prediction replay, PI1M,
local_eval/public feedback, cross-target label use, or Kaggle action.
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
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c208_tg_robust_group_measurement as c208
import round2_c228_tg_c208_transfer_guard as c228
import round2_c232_tg_replicate_reliability_feature as c232


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "tg"
SCHEMA = "ppp.round2.c244.tg-median-residual-stack.v1"
SEED = 20260805
MIN_FULL_MEAN_GAIN = 0.002
STACK_ARMS = (
    "c228_style_guarded_c208",
    "c232_style_replicate_reliability",
)


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


def guarded_c208_candidate(
    parent: dict[str, Any],
    info: dict[str, Any],
    raw_result: dict[str, Any],
    test_rows: pd.DataFrame,
    test_indices: np.ndarray,
    test_parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Regenerate the C228-style fixed guard without reading C228 artifacts."""

    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    raw_candidate = np.asarray(raw_result["candidate"], dtype=np.float64)
    nearest = c228.fold_local_nearest(parent, info)
    oof_guard = c228.guard_mask(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded = raw_candidate.copy()
    guarded[oof_guard] = parent_oof[oof_guard]
    guarded_result = dict(raw_result)
    guarded_result["candidate"] = guarded
    report = c208.evaluate_tg(info, guarded_result)
    report.update(
        {
            "active": True,
            "source": "C228-style regenerated fixed C208 transfer guard",
            "guard_similarity_lt": c228.GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(c228.NEGATIVE_SCAFFOLDS),
            "guarded_oof_rows": int(np.sum(oof_guard)),
            "guarded_similarity_oof_rows": int(np.sum(nearest < c228.GUARD_SIMILARITY_LT)),
            "guarded_scaffold_oof_rows": int(
                np.sum(np.isin(np.asarray(info["scaffolds"], dtype=object), np.asarray(c228.NEGATIVE_SCAFFOLDS, dtype=object)))
            ),
        }
    )

    test_nearest = c228.full_train_nearest(parent, info, test_indices)
    test_scaffolds = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"].astype(str)], dtype=object)
    test_guard = c228.guard_mask(test_scaffolds, test_nearest)
    guarded_test = np.asarray(raw_result["test_direct"], dtype=np.float64).copy()
    guarded_test[test_guard] = test_parent[test_guard]
    return guarded, guarded_test, report, oof_guard, test_guard


def agreement_median_residual_stack(parent_values: np.ndarray, arms: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    parent_values = np.asarray(parent_values, dtype=np.float64)
    residuals = np.column_stack([np.asarray(arms[name], dtype=np.float64) - parent_values for name in STACK_ARMS])
    signs = np.sign(residuals)
    agreement = np.all(signs > 0.0, axis=1) | np.all(signs < 0.0, axis=1)
    candidate = parent_values.copy()
    candidate[agreement] = parent_values[agreement] + np.median(residuals[agreement], axis=1)
    if not np.isfinite(candidate).all():
        raise RuntimeError("C244 signed-agreement median residual stack produced non-finite values")
    return candidate, agreement


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

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    canonical_run = Path(args.canonical_run)
    if not canonical_run.is_absolute():
        canonical_run = (root / canonical_run).resolve()

    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, canonical_run)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    dense, sparse_features, feature_report = carrier.build_round1_features(root, parent["keys"])
    feature_report = dict(feature_report)
    feature_report.update(
        {
            "active_feature_route": "C127 official-SMILES/RDKit/Morgan carrier plus regenerated Tg near-miss arms",
            "active_target": ACTIVE_TARGET,
            "changed_factor": "fixed signed-agreement row-wise median of regenerated C228/C232 Tg residuals",
            "stack_arms": list(STACK_ARMS),
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
        }
    )
    checkpoint(progress, "features_complete", dense_shape=feature_report["dense_shape"], sparse_shape=feature_report["sparse_shape"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c208.target_test_rows(parent, ACTIVE_TARGET)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)

    c127_result = carrier.fit_target(info, dense, sparse_features, test_indices, test_parent)
    c127_report = carrier.evaluate_target(info, c127_result)
    checkpoint(
        progress,
        "c127_direct_diagnostic_complete",
        delta_r2=c127_report["delta_r2"],
        positive_folds=c127_report["positive_folds"],
        bootstrap_lower=c127_report["group_bootstrap_lower"],
        minimum_panel_delta=c127_report["minimum_panel_delta"],
    )

    c208_result = c208.fit_tg_robust(info, dense, sparse_features, test_indices, test_parent)
    c208_raw_report = c208.evaluate_tg(info, c208_result)
    guarded_oof, guarded_test, guarded_report, oof_guard, test_guard = guarded_c208_candidate(
        parent, info, c208_result, test_rows, test_indices, test_parent
    )
    checkpoint(
        progress,
        "guarded_c208_complete",
        raw_delta_r2=c208_raw_report["delta_r2"],
        guarded_delta_r2=guarded_report["delta_r2"],
        positive_folds=guarded_report["positive_folds"],
        bootstrap_lower=guarded_report["group_bootstrap_lower"],
        minimum_panel_delta=guarded_report["minimum_panel_delta"],
        guarded_oof_rows=guarded_report["guarded_oof_rows"],
    )

    reliability_result = c232.fit_tg_reliability(info, dense, sparse_features, test_indices, test_parent)
    reliability_report = c208.evaluate_tg(info, reliability_result)
    checkpoint(
        progress,
        "replicate_reliability_complete",
        delta_r2=reliability_report["delta_r2"],
        positive_folds=reliability_report["positive_folds"],
        bootstrap_lower=reliability_report["group_bootstrap_lower"],
        minimum_panel_delta=reliability_report["minimum_panel_delta"],
    )

    arms = {
        "c228_style_guarded_c208": np.asarray(guarded_oof, dtype=np.float64),
        "c232_style_replicate_reliability": np.asarray(reliability_result["candidate"], dtype=np.float64),
    }
    test_arms = {
        "c228_style_guarded_c208": np.asarray(guarded_test, dtype=np.float64),
        "c232_style_replicate_reliability": np.asarray(reliability_result["test_direct"], dtype=np.float64),
    }
    candidate, agreement_mask = agreement_median_residual_stack(parent_oof, arms)
    test_candidate, test_agreement_mask = agreement_median_residual_stack(np.asarray(test_parent, dtype=np.float64), test_arms)
    active_result = {
        "candidate": candidate,
        "weights": [0.0, 0.0],
        "intercept": 0.0,
        "blend_name": "fixed_signed_agreement_median_residual_not_linear_blend",
        "blend_r2": float(r2_score(np.asarray(info["y"], dtype=np.float64), candidate)),
        "fold_robust_reports": [{"source": "not_applicable_fixed_signed_agreement_median_residual_stack"}],
        "full_robust_report": {"source": "not_applicable_fixed_signed_agreement_median_residual_stack"},
    }
    active_report = c208.evaluate_tg(info, active_result)
    active_report.update(
        {
            "active": True,
            "changed_factor": "fixed signed-agreement median residual stack over regenerated C228/C232-style Tg arms",
            "stack_rule": "parent + median(C228 residual, C232 residual) only where both residuals share nonzero sign; C050 fallback elsewhere",
            "stack_arms": list(STACK_ARMS),
            "agreement_oof_rows": int(np.sum(agreement_mask)),
            "agreement_oof_fraction": float(np.mean(agreement_mask)),
            "agreement_test_rows": int(np.sum(test_agreement_mask)),
            "agreement_test_fraction": float(np.mean(test_agreement_mask)),
            "no_weight_grid": True,
            "no_learned_meta_model": True,
            "no_stored_prediction_replay": True,
            "no_cross_target_labels": True,
            "no_pi1m": True,
            "arm_reports": {
                "c127_direct_round1_carrier": c127_report,
                "c208_raw_robust_measurement": c208_raw_report,
                "c228_style_guarded_c208": guarded_report,
                "c232_style_replicate_reliability": reliability_report,
            },
        }
    )
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "median_residual_stack_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        positive_folds=active_report["positive_folds"],
        bootstrap_lower=active_report["group_bootstrap_lower"],
        minimum_panel_delta=active_report["minimum_panel_delta"],
        agreement_oof_rows=int(np.sum(agreement_mask)),
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
            target_candidate = candidate
            arm_columns = arms
            guarded = oof_guard
            agreement_gate = agreement_mask
        else:
            report = unchanged_report(target_info)
            target_candidate = np.asarray(target_info["parent"], dtype=np.float64)
            arm_columns = {name: np.full(len(target_candidate), np.nan, dtype=np.float64) for name in STACK_ARMS}
            guarded = np.zeros(len(target_candidate), dtype=bool)
            agreement_gate = np.zeros(len(target_candidate), dtype=bool)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        assembled = target_candidate if target == ACTIVE_TARGET and banked else np.asarray(target_info["parent"], dtype=np.float64)
        frame = pd.DataFrame(
            {
                "canonical": target_info["canonical"],
                "target_type": target,
                "target": target_info["y"],
                "parent": target_info["parent"],
                "candidate": target_candidate,
                "assembled": assembled,
                "group": target_info["groups"],
                "scaffold": target_info["scaffolds"],
                "fold": folds,
                "guarded_c208": guarded,
                "agreement_gate": agreement_gate,
                "banked": bool(target == ACTIVE_TARGET and banked),
            }
        )
        for name, values in arm_columns.items():
            frame[name] = values
        oof_parts.append(frame)

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    maximum_target_loss = float(min(active_report["delta_r2"] if banked else 0.0, 0.0))
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= MIN_FULL_MEAN_GAIN and maximum_target_loss >= -0.003)

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component_test = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "parent": test_parent,
            "candidate": test_candidate,
            "guarded_c208": test_guard,
            "agreement_gate": test_agreement_mask,
            **{name: values for name, values in test_arms.items()},
        }
    )
    predictions = parent_test.merge(component_test[["id", "target_type", "candidate"]], on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(
        (predictions["target_type"] == ACTIVE_TARGET) & banked,
        predictions["candidate"],
        predictions["target"],
    )
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940:
        raise RuntimeError("C244 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C244 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C244 prediction finite check failed")

    report = {
        "schema_version": SCHEMA,
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "stored_prediction_replay": False,
        "cross_target_labels": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "component_sources": {
            "c127_direct_round1_carrier": "regenerated C127 direct official-SMILES/RDKit/Morgan carrier for diagnostics only",
            "c228_style_guarded_c208": "regenerated C208 robust-measurement Tg arm plus fixed C228 low-similarity/scaffold fallback",
            "c232_style_replicate_reliability": "regenerated fold-local predicted replicate-reliability feature Tg arm",
        },
        "selection_rule": "fixed signed-agreement median of C228/C232 Tg residual arms; C050 fallback on disagreement; no grid search and no learned meta-model",
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": maximum_target_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(full_candidate_gate_pass and assembled_mean >= 0.95),
        "decision": "banked_component_pending_compound_audit" if banked else "rejected_component_or_full_gate",
        "component_diagnostics": {
            "active_target": ACTIVE_TARGET,
            "tg_delta_r2": active_report["delta_r2"],
            "tg_positive_folds": active_report["positive_folds"],
            "tg_group_bootstrap_lower": active_report["group_bootstrap_lower"],
            "tg_minimum_panel_delta": active_report["minimum_panel_delta"],
            "stack_rule": "signed_agreement_median_residual",
            "stack_arms": list(STACK_ARMS),
            "agreement_oof_rows": int(np.sum(agreement_mask)),
            "agreement_test_rows": int(np.sum(test_agreement_mask)),
            "no_local_eval_public_feedback": True,
            "no_stored_prediction_replay": True,
            "no_learned_meta_model": True,
        },
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c208": sha256_file(round2_root / "tools/round2_c208_tg_robust_group_measurement.py"),
            "c228_guard": sha256_file(round2_root / "tools/round2_c228_tg_c208_transfer_guard.py"),
            "c232_reliability": sha256_file(round2_root / "tools/round2_c232_tg_replicate_reliability_feature.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    oof.loc[oof["target_type"].astype(str).eq(ACTIVE_TARGET)].to_csv(run_dir / "tg_oof_predictions.csv", index=False)
    component_test.to_csv(run_dir / "tg_component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "stack_arms": list(STACK_ARMS),
            "stack_rule": report["selection_rule"],
            "agreement_gate": "both C228-style and C232-style residuals must share nonzero sign; otherwise C050 fallback",
            "guard_similarity_lt": c228.GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(c228.NEGATIVE_SCAFFOLDS),
            "official_only": True,
            "local_eval_read": False,
            "pi1m_used": False,
            "stored_prediction_replay": False,
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
        f"# {run_dir.name}\n\n"
        f"Decision: **{report['decision']}**. Tg parent `{active_report['parent_r2']:.12f}`; "
        f"candidate `{active_report['candidate_r2']:.12f}`; delta `{active_report['delta_r2']:+.12f}`. "
        f"Positive folds `{active_report['positive_folds']}/5`; bootstrap lower `{active_report['group_bootstrap_lower']:.12f}`; "
        f"minimum panel delta `{active_report['minimum_panel_delta']:.12f}`. "
        "Official-only; no local_eval/Kaggle/submission/final-notebook action.\n",
        encoding="utf-8",
    )
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "decision": report["decision"],
                "banked_targets": report["banked_targets"],
                "tg_delta_r2": active_report["delta_r2"],
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
