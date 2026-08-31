#!/usr/bin/env python3
"""C197: fixed Nc consensus-gated replay of the C195 near-miss.

This is a bounded official-only Nc child queued after C193.  It does not replay
stored predictions.  It rebuilds the exact C050 parent, regenerates the C180
clean near-miss Nc carrier from source, regenerates a CatBoost-free
physical/electronic Nc carrier from the C129 feature family, and applies one
fixed label-free disagreement gate to the C195 equal-weight ensemble:

  Nc_candidate = 0.5 * C180_FloryFox_Nc + 0.5 * PhysElec_HGB_ET_Nc

The ensemble is used only when the absolute disagreement between the two arms
is at or below the fixed 75th percentile of the current fold's unlabeled arm
disagreement distribution.  High-disagreement rows fall back to exact C050.

The scientific question is whether C195's near-bankable Nc signal is
concentrated in low-disagreement rows rather than being globally unstable.  If
the fixed consensus gate misses any standard component gate, no target is
banked.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as endpoint_features
import round2_c076_eps_paired_charge_polarizability_residual as physical_features
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as c180


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "nc"
SEED = 20260805
C180_WEIGHT = 0.5
PHYS_ELEC_WEIGHT = 0.5
CONSENSUS_PERCENTILE = 75.0


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
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    return {
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, parent)),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}},
        "folds": [],
        "pass": True,
        "unchanged_parent": True,
    }


def nc_test_indices(parent: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_rows = (
        parent["test"]
        .loc[parent["test"]["target_type"].astype(str).eq(ACTIVE_TARGET)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(ACTIVE_TARGET)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
        raise RuntimeError("C197 Nc test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    parent_values = test_detail["target"].to_numpy(float)
    return test_rows, indices, parent_values


def physical_feature_matrix(parent: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    molecules = parent["molecules"]
    indices = list(range(len(molecules)))
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, parent["keys"])
    endpoint, endpoint_names = endpoint_features.fixed_features(molecules, indices)
    normalized, normalized_names = physical_features.physics_features(molecules, indices)
    charges, charge_names = physical_features.charge_features(molecules, indices)
    grammar = parent_builder.grammar_features(molecules)
    matrix = np.hstack([descriptor, physical, endpoint, normalized, charges, grammar]).astype(np.float64)
    return matrix, {
        "shape": [int(value) for value in matrix.shape],
        "blocks": {
            "rdkit": len(descriptor_names),
            "physical": len(physical_names),
            "endpoint": len(endpoint_names),
            "normalized_physical": len(normalized_names),
            "charge": len(charge_names),
            "fragment_grammar": int(grammar.shape[1]),
        },
        "paired_labels_used": False,
        "source_family": "C129 physical/electronic feature family, CatBoost-free local arm set",
    }


def fit_physical_arms(
    matrix: np.ndarray,
    y: np.ndarray,
    train_matrix_rows: np.ndarray,
    prediction_matrix_rows: np.ndarray,
    train_y_rows: np.ndarray,
    seed: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(matrix[train_matrix_rows])
    prediction_x = imputer.transform(matrix[prediction_matrix_rows])
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    train_z = transformer.fit_transform(y[train_y_rows].reshape(-1, 1)).ravel()
    hist = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=3.0,
        random_state=seed,
    )
    hist.fit(train_x, train_z)
    hist_prediction = transformer.inverse_transform(hist.predict(prediction_x).reshape(-1, 1)).ravel()
    trees = ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.55,
        min_samples_leaf=2,
        n_jobs=2,
        random_state=seed,
    )
    trees.fit(train_x, train_z)
    tree_prediction = transformer.inverse_transform(trees.predict(prediction_x).reshape(-1, 1)).ravel()
    return np.column_stack([hist_prediction, tree_prediction])


def physical_nc_run(
    parent: dict[str, Any],
    matrix: np.ndarray,
    progress: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    y = np.asarray(info["y"], dtype=float)
    parent_oof = np.asarray(info["parent"], dtype=float)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    graph_indices = np.asarray(info["indices"], dtype=np.int64)
    direct_oof = np.full((len(y), 2), np.nan, dtype=float)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        direct_oof[validation] = fit_physical_arms(
            matrix,
            y,
            graph_indices[training],
            graph_indices[validation],
            training,
            SEED + 195 + fold,
        )
    arms = np.column_stack([parent_oof, direct_oof])
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, arms)
    candidate = arms @ weights + intercept
    test_rows, test_indices, test_parent = nc_test_indices(parent)
    test_arms = fit_physical_arms(
        matrix,
        y,
        graph_indices,
        test_indices,
        np.arange(len(y), dtype=np.int64),
        SEED + 1195,
    )
    test_candidate = np.column_stack([test_parent, test_arms]) @ weights + intercept
    report = carrier.evaluate_target(info, {"candidate": candidate})
    report.update(
        {
            "blend_name": blend_name,
            "blend_weights": [float(value) for value in weights],
            "blend_intercept": float(intercept),
            "blend_r2": float(blend_r2),
            "hist_r2": float(r2_score(y, direct_oof[:, 0])),
            "extratrees_r2": float(r2_score(y, direct_oof[:, 1])),
            "test_rows": int(len(test_rows)),
        }
    )
    checkpoint(progress, "physical_electronic_nc_complete", delta_r2=report["delta_r2"], pass_gate=report["pass"])
    return report, candidate, test_candidate


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

    data_dir = (root / args.data_dir).resolve()
    canonical_run = (root / args.canonical_run).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, canonical_run)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = nc_test_indices(parent)

    ffox_dense, ffox_sparse, ffox_feature_report = c180.build_features(root, parent["keys"])
    ffox_result = carrier.fit_target(info, ffox_dense, ffox_sparse, test_indices, test_parent)
    ffox_report = carrier.evaluate_target(info, ffox_result)
    checkpoint(progress, "c180_nc_regenerated", delta_r2=ffox_report["delta_r2"], pass_gate=ffox_report["pass"])

    physical_matrix, physical_feature_report = physical_feature_matrix(parent)
    physical_report, physical_oof, physical_test = physical_nc_run(parent, physical_matrix, progress)
    checkpoint(progress, "physical_electronic_nc_regenerated", delta_r2=physical_report["delta_r2"], pass_gate=physical_report["pass"])

    parent_oof = np.asarray(info["parent"], dtype=float)
    ffox_oof = np.asarray(ffox_result["candidate"], dtype=float)
    physical_oof = np.asarray(physical_oof, dtype=float)
    fixed_candidate = C180_WEIGHT * ffox_oof + PHYS_ELEC_WEIGHT * physical_oof
    fixed_test = C180_WEIGHT * np.asarray(ffox_result["test_direct"], dtype=float) + PHYS_ELEC_WEIGHT * np.asarray(physical_test, dtype=float)
    if len(fixed_candidate) != len(parent_oof) or len(fixed_test) != len(test_rows):
        raise RuntimeError("C197 fixed ensemble length mismatch")

    fixed_report = carrier.evaluate_target(info, {"candidate": fixed_candidate})
    folds_for_gate = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
    disagreement = np.abs(ffox_oof - physical_oof)
    consensus_gate = np.zeros(len(disagreement), dtype=bool)
    fold_thresholds: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = folds_for_gate == fold
        threshold = float(np.percentile(disagreement[validation], CONSENSUS_PERCENTILE))
        consensus_gate[validation] = disagreement[validation] <= threshold
        fold_thresholds.append(
            {
                "fold": int(fold),
                "rows": int(validation.sum()),
                "threshold": threshold,
                "changed_rows": int(consensus_gate[validation].sum()),
                "fallback_rows": int(validation.sum() - consensus_gate[validation].sum()),
            }
        )
    gated_candidate = np.where(consensus_gate, fixed_candidate, parent_oof)
    test_disagreement = np.abs(np.asarray(ffox_result["test_direct"], dtype=float) - np.asarray(physical_test, dtype=float))
    test_threshold = float(np.percentile(test_disagreement, CONSENSUS_PERCENTILE))
    test_gate = test_disagreement <= test_threshold
    gated_test = np.where(test_gate, fixed_test, test_parent)

    active_report = carrier.evaluate_target(info, {"candidate": gated_candidate})
    active_report.update(
        {
            "ensemble_rule": "use 0.5*C180_FloryFox_Nc + 0.5*PhysElec_HGB_ET_Nc only when absolute arm disagreement is within the fixed 75th percentile; otherwise exact C050",
            "changed_factor": "fixed label-free consensus disagreement fallback added to the regenerated C195 Nc ensemble",
            "consensus_percentile": CONSENSUS_PERCENTILE,
            "consensus_changed_rows": int(consensus_gate.sum()),
            "consensus_fallback_rows": int((~consensus_gate).sum()),
            "consensus_fraction_changed": float(consensus_gate.mean()),
            "fold_disagreement_thresholds": fold_thresholds,
            "test_disagreement_threshold": test_threshold,
            "test_changed_rows": int(test_gate.sum()),
            "test_fallback_rows": int((~test_gate).sum()),
            "fixed_c195_delta_r2": fixed_report["delta_r2"],
            "fixed_c195_candidate_r2": fixed_report["candidate_r2"],
            "fixed_c195_positive_folds": fixed_report["positive_folds"],
            "fixed_c195_group_bootstrap_lower": fixed_report["group_bootstrap_lower"],
            "fixed_c195_minimum_panel_delta": fixed_report["minimum_panel_delta"],
            "c180_delta_r2": ffox_report["delta_r2"],
            "c180_positive_folds": ffox_report["positive_folds"],
            "c180_group_bootstrap_lower": ffox_report["group_bootstrap_lower"],
            "c180_minimum_panel_delta": ffox_report["minimum_panel_delta"],
            "physical_electronic_delta_r2": physical_report["delta_r2"],
            "physical_electronic_positive_folds": physical_report["positive_folds"],
            "physical_electronic_group_bootstrap_lower": physical_report["group_bootstrap_lower"],
            "physical_electronic_minimum_panel_delta": physical_report["minimum_panel_delta"],
        }
    )
    banked = [ACTIVE_TARGET] if active_report["pass"] else []

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = parent["target_info"][target]
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = gated_candidate
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=float)
        target_reports[target] = report
        assembled = candidate if target in banked else np.asarray(target_info["parent"], dtype=float)
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "assembled": assembled,
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "outer_fold": carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object)),
                }
            )
        )

    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    maximum_target_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_candidate_gate_pass = bool(mean_candidate >= mean_parent + 0.002 and maximum_target_loss >= -0.003 and bool(banked))

    parent_detail = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    raw_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(target)].sort_values("id").reset_index(drop=True)
        if target == ACTIVE_TARGET and ACTIVE_TARGET in banked:
            values = gated_test
        else:
            values = (
                parent_detail.loc[parent_detail["target_type"].astype(str).eq(target)]
                .sort_values("id")["target"]
                .to_numpy(float)
            )
        raw_parts.append(pd.DataFrame({"id": frame["id"].astype(int), "target_type": target, "model_prediction": values}))
    raw = pd.concat(raw_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    raw_labels, _ = reference.build_label_pool(parent["train"], parent["archive"])
    detail, override_report = reference.apply_official_overrides(raw, parent["test"], raw_labels)
    predictions = detail[["id", "target"]].copy()
    expected_ids = parent["test"]["id"].astype(int).to_numpy()
    if (
        len(predictions) != 4940
        or not np.array_equal(predictions["id"].astype(int).to_numpy(), expected_ids)
        or not np.isfinite(predictions["target"].to_numpy(float)).all()
    ):
        raise RuntimeError("C197 complete output contract failed")

    report = {
        "schema_version": "ppp.round2.c197.nc-consensus-gated-c195.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "parent": "C050 source rebuild; no stored C180/C129/C195 prediction input",
        "hypothesis": "C195's two regenerated Nc mechanisms contain real but unstable Nc signal; a fixed label-free consensus/disagreement fallback can remove high-disagreement corrections while preserving the near-bankable residual-diversity gain.",
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "active_targets": [ACTIVE_TARGET],
        "component_sources": {
            "c180": "regenerated from round2_c180_flory_fox_oligomer_carriers.build_features + C127 carrier.fit_target",
            "physical_electronic": "regenerated from C129 feature family with local scikit-learn HGB/ExtraTrees arms",
        },
        "ensemble_weights": {"c180_flory_fox_nc": C180_WEIGHT, "physical_electronic_hgb_et_nc": PHYS_ELEC_WEIGHT},
        "consensus_gate": {
            "percentile": CONSENSUS_PERCENTILE,
            "criterion": "abs(c180_candidate - physical_electronic_candidate) <= percentile threshold computed from unlabeled arm disagreements for each OOF fold; test uses the same fixed percentile on test arm disagreements",
            "uses_target_labels": False,
            "uses_local_eval_or_public_feedback": False,
            "threshold_tuned": False,
        },
        "feature_report": {
            "c180_flory_fox": ffox_feature_report,
            "physical_electronic_hgb_et": physical_feature_report,
        },
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": maximum_target_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(mean_candidate >= 0.95 and full_candidate_gate_pass),
        "official_override_report": override_report,
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c180": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "endpoint_features": sha256_file(round2_root / "tools/round2_c063_egb_endpoint_conjugation_residual.py"),
            "physical_features": sha256_file(round2_root / "tools/round2_c076_eps_paired_charge_polarizability_residual.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
        "decision": "candidate_pass_pending_clean_reproduction" if full_candidate_gate_pass else "rejected_component_or_full_gate",
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    oof.loc[oof["target_type"].astype(str).eq(ACTIVE_TARGET), ["canonical", "target", "parent", "candidate"]].to_csv(
        run_dir / "nc_oof_predictions.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "canonical": info["canonical"],
            "target": info["y"],
            "parent": info["parent"],
            "c180_candidate": ffox_oof,
            "physical_electronic_candidate": physical_oof,
            "fixed_candidate": fixed_candidate,
            "arm_disagreement": disagreement,
            "consensus_gate": consensus_gate,
            "gated_candidate": gated_candidate,
        }
    ).to_csv(run_dir / "component_oof_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "c180_candidate": ffox_result["test_direct"],
            "physical_electronic_candidate": physical_test,
            "fixed_candidate": fixed_test,
            "arm_disagreement": test_disagreement,
            "consensus_gate": test_gate,
            "gated_candidate": gated_test,
        }
    ).to_csv(run_dir / "component_test_diagnostics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": "ppp.round2.c197.nc-consensus-gated-c195.v1",
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "ensemble_rule": "fixed_equal_weight_when_consensus_else_parent",
            "weights": {"c180": C180_WEIGHT, "physical_electronic": PHYS_ELEC_WEIGHT},
            "consensus_percentile": CONSENSUS_PERCENTILE,
            "selection": "no tuning; fixed percentile disagreement gate; bank only if standard component gate passes",
            "official_only": True,
            "local_eval_read": False,
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
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. "
        f"Nc consensus-gated delta `{active_report['delta_r2']:+.12f}`; "
        f"C195 fixed-ensemble diagnostic delta `{fixed_report['delta_r2']:+.12f}`; "
        f"banked targets: `{','.join(banked) or 'none'}`. "
        f"Mean parent `{mean_parent:.12f}`; assembled `{mean_candidate:.12f}`; "
        f"gain `{mean_candidate - mean_parent:+.12f}`. No local_eval/Kaggle/upload/submission action.\n",
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
                "banked_targets": banked,
                "nc_delta_r2": active_report["delta_r2"],
                "fixed_c195_delta_r2": fixed_report["delta_r2"],
                "consensus_changed_rows": int(consensus_gate.sum()),
                "mean_parent_r2": mean_parent,
                "mean_candidate_r2": mean_candidate,
                "mean_gain": mean_candidate - mean_parent,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
