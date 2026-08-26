#!/usr/bin/env python3
"""C205: independent Ei dense oligomer/asymptotic confirmation.

This child de-risks the C199 guarded Ei component without consuming C196/C199
prediction artifacts, post-hoc failure-slice guards, identity/Huber routes,
cross-property Stage-2 labels, PI1M, graph/WL/path kernels, local_eval values,
public-score feedback, Kaggle compute, upload, submission, or final-notebook
artifacts.

The single changed factor is a dense-only official-SMILES Flory--Fox/oligomer
asymptotic feature basis from C180, fitted fresh against exact C050 under the
standard grouped component gates.  A normal component pass is recorded as
independent confirmation, but it is banked as an Ei replacement only if it also
matches or beats the prior C199 Ei R2.
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
from scipy import sparse
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as c180


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "ei"
C199_REFERENCE_EI_R2 = 0.8566558157138717
MIN_BANKABLE_DELTA_R2 = 0.01


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


def target_test_rows(parent: dict[str, Any], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    test_detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"] == target]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(test_rows["id"].to_numpy(np.int64), test_detail["id"].to_numpy(np.int64)):
        raise RuntimeError(f"test ID alignment failed for {target}")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    return test_rows, indices, test_detail["target"].to_numpy(np.float64)


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

    dense, ignored_sparse, feature_report = c180.build_features(root, parent["keys"])
    dense = np.asarray(dense, dtype=np.float64)
    dense_only_sparse = sparse.csr_matrix((dense.shape[0], 0), dtype=np.float64)
    feature_report = dict(feature_report)
    feature_report.update(
        {
            "active_feature_route": "dense_only",
            "ignored_sparse_shape": [int(value) for value in ignored_sparse.shape],
            "ignored_sparse_nnz": int(ignored_sparse.nnz),
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
            "uses_c196_or_c199_predictions": False,
            "uses_posthoc_failure_guard": False,
        }
    )
    checkpoint(progress, "features_complete", dense_shape=feature_report["dense_shape"], ignored_sparse_shape=feature_report["ignored_sparse_shape"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = target_test_rows(parent, ACTIVE_TARGET)
    result = carrier.fit_target(info, dense, dense_only_sparse, test_indices, test_parent)
    active_report = carrier.evaluate_target(info, result)
    normal_component_gate_pass = bool(active_report["pass"])
    replacement_bank_gate_pass = bool(active_report["candidate_r2"] >= C199_REFERENCE_EI_R2 - 1.0e-12)
    banked = normal_component_gate_pass and replacement_bank_gate_pass
    active_report.update(
        {
            "active": True,
            "normal_component_gate_pass": normal_component_gate_pass,
            "replacement_bank_gate_pass": replacement_bank_gate_pass,
            "confirmation_only": bool(normal_component_gate_pass and not replacement_bank_gate_pass),
            "pass": bool(banked),
            "c199_reference_ei_r2": C199_REFERENCE_EI_R2,
            "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
            "exclusions": {
                "c196_predictions": False,
                "c199_predictions": False,
                "c196_failure_panel_guard": False,
                "identity_huber_route": False,
                "partner_labels": False,
                "stage2_cross_property_block": False,
                "pi1m": False,
                "graph_wl_path_kernel": False,
                "local_eval_public_feedback": False,
            },
            "blend_name": result["blend_name"],
            "blend_weights": [float(value) for value in result["weights"]],
            "blend_intercept": float(result["intercept"]),
            "feature_rows": int(len(info["y"])),
            "test_rows": int(len(test_rows)),
        }
    )
    checkpoint(
        progress,
        "ei_dense_oligomer_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        normal_component_gate_pass=normal_component_gate_pass,
        replacement_bank_gate_pass=replacement_bank_gate_pass,
        pass_gate=banked,
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = np.asarray(result["candidate"], dtype=np.float64)
            direct_oof = np.asarray(result["direct_oof"], dtype=np.float64)
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
            direct_oof = np.full((len(candidate), 2), np.nan, dtype=np.float64)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        assembled = candidate if target == ACTIVE_TARGET and banked else np.asarray(target_info["parent"], dtype=np.float64)
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "assembled": assembled,
                    "direct_ridge": direct_oof[:, 0],
                    "direct_tree": direct_oof[:, 1],
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": folds,
                    "banked": bool(target == ACTIVE_TARGET and banked),
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target == ACTIVE_TARGET and banked else 0.0 for target in TARGETS))
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003)

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component_test = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "direct_candidate": result["test_direct"],
        }
    )
    predictions = parent_test.merge(component_test, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(
        (predictions["target_type"] == ACTIVE_TARGET) & banked,
        predictions["direct_candidate"],
        predictions["target"],
    )
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940:
        raise RuntimeError("C205 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C205 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C205 prediction finite check failed")

    report = {
        "schema_version": "ppp.round2.c205.ei-dense-oligomer-confirmation.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "confirmation_targets": [ACTIVE_TARGET] if normal_component_gate_pass else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(full_candidate_gate_pass and assembled_mean >= 0.95),
        "decision": "banked_replacement_component" if banked else ("confirmation_only_not_banked" if normal_component_gate_pass else "rejected_component_or_full_gate"),
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "rich_builder": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component_test.to_csv(run_dir / "ei_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": report["schema_version"],
            "seed": carrier.SEED,
            "target": ACTIVE_TARGET,
            "feature_basis": "C180 dense-only Flory-Fox/oligomer/asymptotic official-SMILES features",
            "normal_component_gate": "delta >= 0.01, positive folds >= 4/5, grouped-bootstrap lower > 0, all explicit panel minima >= 0",
            "replacement_bank_gate": f"candidate Ei R2 >= C199 Ei R2 {C199_REFERENCE_EI_R2}",
            "confirmation_only_if_below_c199": True,
            "no_hyperparameter_sweep": True,
            "local_eval_read": False,
            "pi1m_used": False,
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
        f"Decision: **{report['decision']}**. Ei parent `{active_report['parent_r2']:.12f}`; "
        f"candidate `{active_report['candidate_r2']:.12f}`; delta `{active_report['delta_r2']:+.12f}`. "
        f"Normal confirmation gate `{normal_component_gate_pass}`; replacement bank gate `{replacement_bank_gate_pass}`. "
        "No local_eval/Kaggle/submission/final-notebook action.\n",
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
                "banked_targets": report["banked_targets"],
                "confirmation_targets": report["confirmation_targets"],
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "ei_delta_r2": active_report["delta_r2"],
                "ei_candidate_r2": active_report["candidate_r2"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
