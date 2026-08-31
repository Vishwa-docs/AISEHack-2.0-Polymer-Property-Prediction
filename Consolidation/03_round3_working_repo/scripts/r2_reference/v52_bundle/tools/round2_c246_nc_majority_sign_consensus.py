#!/usr/bin/env python3
"""C242: fixed Nc near-miss stability ensemble.

This is a bounded clean continuation after C241.  It regenerates four prior
near-miss Nc mechanisms from official Round 2 inputs and tests one fixed
convex ensemble:

  0.40 * C195 fixed near-miss diversity candidate
  0.25 * C226-style guarded C180 candidate
  0.10 * C234-style replicate-reliability candidate
  0.25 * C240-style electro-polar autocorrelation candidate

The weights are frozen before execution.  There is no grid search, no learned
meta-model, no stored prediction replay, no cross-target labels, no PI1M, no
local_eval/public feedback, and no Kaggle action.
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
import round2_c180_flory_fox_oligomer_carriers as c180
import round2_c195_nc_nearmiss_residual_diversity as c195
import round2_c207_egc_c180_transfer_guard as c207
import round2_c208_tg_robust_group_measurement as c208
import round2_c220_ei_electro_polar_autocorr as c220
import round2_c232_tg_replicate_reliability_feature as c232


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "nc"
SCHEMA = "ppp.round2.c242.nc-nearmiss-stability-ensemble.v1"
SEED = 20260805
MIN_FULL_MEAN_GAIN = 0.002
WEIGHTS = {
    "c195_fixed_nearmiss_diversity": 0.40,
    "c226_style_guarded_c180": 0.25,
    "c234_style_replicate_reliability": 0.10,
    "c240_style_electro_polar_autocorr": 0.25,
}
NEGATIVE_SCAFFOLDS = ("c1ccc(-c2cccs2)cc1",)
GUARD_SIMILARITY_LT = 0.30


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


def guarded_c180_candidate(
    parent: dict[str, Any],
    info: dict[str, Any],
    raw_result: dict[str, Any],
    test_rows: pd.DataFrame,
    test_indices: np.ndarray,
    test_parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    c207.ACTIVE_TARGET = ACTIVE_TARGET
    c207.NEGATIVE_SCAFFOLDS = NEGATIVE_SCAFFOLDS
    c207.GUARD_SIMILARITY_LT = GUARD_SIMILARITY_LT
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    raw_candidate = np.asarray(raw_result["candidate"], dtype=np.float64)
    nearest = c207.fold_local_nearest(parent, info)
    oof_guard, oof_guard_summary = c207.guard_mask_from_scaffold_similarity(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded = np.where(oof_guard, parent_oof, raw_candidate)

    test_nearest = c207.full_train_nearest(parent, np.asarray(info["indices"], dtype=np.int64), test_indices)
    test_scaffolds = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"]], dtype=object)
    test_guard, test_guard_summary = c207.guard_mask_from_scaffold_similarity(test_scaffolds, test_nearest)
    guarded_test = np.where(test_guard, test_parent, np.asarray(raw_result["test_direct"], dtype=np.float64))
    report = carrier.evaluate_target(info, {"candidate": guarded})
    report.update(
        {
            "active": True,
            "source": "C226-style regenerated C180 fixed guard",
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "oof_guard_summary": oof_guard_summary,
            "test_guard_summary": test_guard_summary,
        }
    )
    return guarded, guarded_test, report


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

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c195.nc_test_indices(parent)

    c180_dense, c180_sparse, c180_feature_report = c180.build_features(root, parent["keys"])
    c180_result = carrier.fit_target(info, c180_dense, c180_sparse, test_indices, test_parent)
    c180_report = carrier.evaluate_target(info, c180_result)
    guarded_oof, guarded_test, guarded_report = guarded_c180_candidate(parent, info, c180_result, test_rows, test_indices, test_parent)
    checkpoint(
        progress,
        "c180_and_guarded_c180_complete",
        raw_delta_r2=c180_report["delta_r2"],
        guarded_delta_r2=guarded_report["delta_r2"],
        guarded_positive_folds=guarded_report["positive_folds"],
        guarded_bootstrap_lower=guarded_report["group_bootstrap_lower"],
        guarded_minimum_panel_delta=guarded_report["minimum_panel_delta"],
    )

    physical_matrix, physical_feature_report = c195.physical_feature_matrix(parent)
    physical_report, physical_oof, physical_test = c195.physical_nc_run(parent, physical_matrix, progress)
    c195_fixed_oof = 0.5 * np.asarray(c180_result["candidate"], dtype=np.float64) + 0.5 * np.asarray(physical_oof, dtype=np.float64)
    c195_fixed_test = 0.5 * np.asarray(c180_result["test_direct"], dtype=np.float64) + 0.5 * np.asarray(physical_test, dtype=np.float64)
    c195_fixed_report = carrier.evaluate_target(info, {"candidate": c195_fixed_oof})
    checkpoint(
        progress,
        "c195_fixed_nearmiss_complete",
        delta_r2=c195_fixed_report["delta_r2"],
        positive_folds=c195_fixed_report["positive_folds"],
        bootstrap_lower=c195_fixed_report["group_bootstrap_lower"],
        minimum_panel_delta=c195_fixed_report["minimum_panel_delta"],
    )

    round1_dense, round1_sparse, round1_feature_report = carrier.build_round1_features(root, parent["keys"])
    reliability_result = c232.fit_tg_reliability(info, round1_dense, round1_sparse, test_indices, test_parent)
    reliability_report = c208.evaluate_tg(info, reliability_result)
    checkpoint(
        progress,
        "replicate_reliability_complete",
        delta_r2=reliability_report["delta_r2"],
        positive_folds=reliability_report["positive_folds"],
        bootstrap_lower=reliability_report["group_bootstrap_lower"],
        minimum_panel_delta=reliability_report["minimum_panel_delta"],
    )

    c220.ACTIVE_TARGET = ACTIVE_TARGET
    c220.SEED = SEED
    autocorr_features, autocorr_feature_report = c220.build_autocorr_features(list(parent["keys"]))
    electro_result = c220.fit_ei_residual(info, autocorr_features, test_indices, test_parent)
    electro_report = carrier.evaluate_target(info, {"candidate": electro_result["candidate"]})
    checkpoint(
        progress,
        "electro_polar_autocorr_complete",
        delta_r2=electro_report["delta_r2"],
        positive_folds=electro_report["positive_folds"],
        bootstrap_lower=electro_report["group_bootstrap_lower"],
        minimum_panel_delta=electro_report["minimum_panel_delta"],
    )

    arms = {
        "c195_fixed_nearmiss_diversity": np.asarray(c195_fixed_oof, dtype=np.float64),
        "c226_style_guarded_c180": np.asarray(guarded_oof, dtype=np.float64),
        "c234_style_replicate_reliability": np.asarray(reliability_result["candidate"], dtype=np.float64),
        "c240_style_electro_polar_autocorr": np.asarray(electro_result["candidate"], dtype=np.float64),
    }
    test_arms = {
        "c195_fixed_nearmiss_diversity": np.asarray(c195_fixed_test, dtype=np.float64),
        "c226_style_guarded_c180": np.asarray(guarded_test, dtype=np.float64),
        "c234_style_replicate_reliability": np.asarray(reliability_result["test_direct"], dtype=np.float64),
        "c240_style_electro_polar_autocorr": np.asarray(electro_result["test_candidate"], dtype=np.float64),
    }
    weight_sum = float(sum(WEIGHTS.values()))
    if abs(weight_sum - 1.0) > 1e-12 or any(value < 0 for value in WEIGHTS.values()):
        raise RuntimeError(f"C242 invalid convex weights: {WEIGHTS}")
    candidate = sum(WEIGHTS[name] * values for name, values in arms.items())
    test_candidate = sum(WEIGHTS[name] * values for name, values in test_arms.items())
    if not np.isfinite(candidate).all() or not np.isfinite(test_candidate).all():
        raise RuntimeError("C242 produced non-finite Nc candidate")
    active_report = carrier.evaluate_target(info, {"candidate": candidate})
    active_report.update(
        {
            "active": True,
            "changed_factor": "fixed convex stability ensemble of regenerated Nc near-miss arms",
            "fixed_weights": WEIGHTS,
            "no_weight_grid": True,
            "no_learned_meta_model": True,
            "no_stored_prediction_replay": True,
            "no_cross_target_labels": True,
            "no_pi1m": True,
            "arm_reports": {
                "c180_raw": c180_report,
                "c195_fixed_nearmiss_diversity": c195_fixed_report,
                "c226_style_guarded_c180": guarded_report,
                "c234_style_replicate_reliability": reliability_report,
                "c240_style_electro_polar_autocorr": electro_report,
                "physical_electronic": physical_report,
            },
        }
    )
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "fixed_stability_ensemble_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        positive_folds=active_report["positive_folds"],
        bootstrap_lower=active_report["group_bootstrap_lower"],
        minimum_panel_delta=active_report["minimum_panel_delta"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
            target_candidate = candidate
            arm_columns = arms
        else:
            report = unchanged_report(target_info)
            target_candidate = np.asarray(target_info["parent"], dtype=np.float64)
            arm_columns = {name: np.full(len(target_candidate), np.nan, dtype=np.float64) for name in WEIGHTS}
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
            "candidate": test_candidate,
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
        raise RuntimeError("C242 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C242 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C242 prediction finite check failed")

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
            "c195_fixed_nearmiss_diversity": "regenerated C180 + regenerated C195 physical/electronic arm, fixed 0.5/0.5",
            "c226_style_guarded_c180": "regenerated C180 with fixed low-similarity and scaffold C050 fallback",
            "c234_style_replicate_reliability": "regenerated fold-local predicted replicate-reliability features",
            "c240_style_electro_polar_autocorr": "regenerated fixed electro-polar graph-distance autocorrelation Ridge residual",
        },
        "fixed_weights": WEIGHTS,
        "selection_rule": "one fixed convex ensemble; no weight/model/threshold search and no learned meta-model",
        "feature_report": {
            "c180": c180_feature_report,
            "physical_electronic": physical_feature_report,
            "round1_reliability": round1_feature_report,
            "electro_polar_autocorr": autocorr_feature_report,
        },
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
            "nc_delta_r2": active_report["delta_r2"],
            "nc_positive_folds": active_report["positive_folds"],
            "nc_group_bootstrap_lower": active_report["group_bootstrap_lower"],
            "nc_minimum_panel_delta": active_report["minimum_panel_delta"],
            "fixed_weights": WEIGHTS,
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
            "c180": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "c195": sha256_file(round2_root / "tools/round2_c195_nc_nearmiss_residual_diversity.py"),
            "c207_guard": sha256_file(round2_root / "tools/round2_c207_egc_c180_transfer_guard.py"),
            "c220_electro": sha256_file(round2_root / "tools/round2_c220_ei_electro_polar_autocorr.py"),
            "c232_reliability": sha256_file(round2_root / "tools/round2_c232_tg_replicate_reliability_feature.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    oof.loc[oof["target_type"].astype(str).eq(ACTIVE_TARGET)].to_csv(run_dir / "nc_oof_predictions.csv", index=False)
    component_test.to_csv(run_dir / "nc_component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "fixed_weights": WEIGHTS,
            "selection_rule": report["selection_rule"],
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
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
        f"Decision: **{report['decision']}**. Nc parent `{active_report['parent_r2']:.12f}`; "
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
                "nc_delta_r2": active_report["delta_r2"],
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
