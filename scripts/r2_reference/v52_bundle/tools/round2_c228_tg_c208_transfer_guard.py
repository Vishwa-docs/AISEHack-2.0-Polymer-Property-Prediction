#!/usr/bin/env python3
"""C228: Tg C208 transfer-guard repair.

C208 was a clean Tg near-miss: +0.009638 R2, 5/5 positive folds, and positive
group bootstrap, but it failed explicit scaffold/low-similarity transfer panels.
This child tests one bounded factor: regenerate the same C208 official-only Tg
robust-group carrier, then fall back to exact C050 on the predeclared C208
negative transfer panels.

No model hyperparameters, folds, feature blocks, blend search space, source
data, PI1M, stored predictions, local_eval values, public feedback, Kaggle compute,
upload, submission, or final-notebook action are used.
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


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "tg"
SCHEMA = "ppp.round2.c228.tg-c208-transfer-guard.v1"
GUARD_SIMILARITY_LT = 0.30
NEGATIVE_SCAFFOLDS = (
    "c1ccc(-n2on2-c2ccccc2)cc1",
    "O=C(Oc1ccc(Cc2ccccc2)cc1)c1ccccc1",
    "O=C(Nc1ccccc1)c1ccc(C(=O)Nc2ccccc2)cc1",
    "O=C(Nc1ccccc1)c1cccc(C(=O)Nc2ccccc2)c1",
    "O=C1CCC(=O)N1",
    "O=S(=O)(c1ccccc1)c1ccccc1",
    "c1ccc(CCc2ccccc2)cc1",
    "C1CCCC1",
    "O=S(=O)(c1ccccc1)c1ccc(Oc2ccc(Cc3ccccc3)cc2)cc1",
    "O=C1c2ccccc2C(=O)N1c1ccc(Oc2ccc(N3C(=O)c4ccccc4C3=O)cc2)cc1",
    "c1ccc(Oc2ccccc2)cc1",
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


def fold_local_nearest(parent: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    indices = np.asarray(info["indices"], dtype=np.int64)
    result = np.full(len(indices), np.nan, dtype=np.float64)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [info["fingerprints"][int(indices[row])] for row in training]
        for row in validation:
            result[row] = max(reference.DataStructs.BulkTanimotoSimilarity(info["fingerprints"][int(indices[row])], train_fps))
    return result


def full_train_nearest(parent: dict[str, Any], info: dict[str, Any], test_indices: np.ndarray) -> np.ndarray:
    train_fps = [parent["fingerprints"][int(index)] for index in np.asarray(info["indices"], dtype=np.int64)]
    result = np.full(len(test_indices), np.nan, dtype=np.float64)
    for row, index in enumerate(np.asarray(test_indices, dtype=np.int64)):
        result[row] = max(reference.DataStructs.BulkTanimotoSimilarity(parent["fingerprints"][int(index)], train_fps))
    return result


def guard_mask(scaffolds: np.ndarray, nearest: np.ndarray) -> np.ndarray:
    scaffold_guard = np.isin(np.asarray(scaffolds, dtype=object), np.asarray(NEGATIVE_SCAFFOLDS, dtype=object))
    similarity_guard = np.asarray(nearest, dtype=np.float64) < GUARD_SIMILARITY_LT
    return scaffold_guard | similarity_guard


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

    dense, sparse_features, feature_report = carrier.build_round1_features(root, parent["keys"])
    feature_report = dict(feature_report)
    feature_report.update(
        {
            "active_feature_route": "C127 official-SMILES/RDKit/Morgan carrier",
            "active_target": ACTIVE_TARGET,
            "changed_factor": "C208 robust Tg carrier plus fixed C050 transfer-panel fallback",
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
        }
    )
    checkpoint(progress, "features_complete", dense_shape=feature_report["dense_shape"], sparse_shape=feature_report["sparse_shape"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c208.target_test_rows(parent, ACTIVE_TARGET)
    raw_result = c208.fit_tg_robust(info, dense, sparse_features, test_indices, test_parent)
    raw_report = c208.evaluate_tg(info, raw_result)

    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    raw_candidate = np.asarray(raw_result["candidate"], dtype=np.float64)
    nearest = fold_local_nearest(parent, info)
    oof_guard = guard_mask(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded_candidate = raw_candidate.copy()
    guarded_candidate[oof_guard] = parent_oof[oof_guard]
    guarded_result = dict(raw_result)
    guarded_result["candidate"] = guarded_candidate
    active_report = c208.evaluate_tg(info, guarded_result)
    active_report.update(
        {
            "posthoc_repair_from_c208_failure_panels": True,
            "independent_confirmation_required_before_final_notebook": True,
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "guarded_oof_rows": int(np.sum(oof_guard)),
            "guarded_scaffold_oof_rows": int(np.sum(np.isin(np.asarray(info["scaffolds"], dtype=object), np.asarray(NEGATIVE_SCAFFOLDS, dtype=object)))),
            "guarded_similarity_oof_rows": int(np.sum(nearest < GUARD_SIMILARITY_LT)),
            "raw_c208_candidate_r2": raw_report["candidate_r2"],
            "raw_c208_delta_r2": raw_report["delta_r2"],
            "raw_c208_positive_folds": raw_report["positive_folds"],
            "raw_c208_group_bootstrap_lower": raw_report["group_bootstrap_lower"],
            "raw_c208_minimum_panel_delta": raw_report["minimum_panel_delta"],
            "c208_known_failure_panels": [f"scaffold_{value}" for value in NEGATIVE_SCAFFOLDS] + ["similarity_lt_0.30"],
            "changed_factor": "C208 regenerated robust Tg carrier plus label-free fallback on predeclared C208 negative transfer panels",
        }
    )
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "tg_transfer_guard_complete",
        raw_delta_r2=raw_report["delta_r2"],
        guarded_delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        minimum_panel_delta=active_report["minimum_panel_delta"],
        group_bootstrap_lower=active_report["group_bootstrap_lower"],
        guarded_oof_rows=active_report["guarded_oof_rows"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
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
        target_reports[target] = report
        assembled = candidate if target == ACTIVE_TARGET and banked else np.asarray(target_info["parent"], dtype=np.float64)
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
                    "banked": bool(target == ACTIVE_TARGET and banked),
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target == ACTIVE_TARGET and banked else 0.0 for target in TARGETS))
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003)

    test_nearest = full_train_nearest(parent, info, test_indices)
    test_scaffolds = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"].astype(str)], dtype=object)
    test_guard = guard_mask(test_scaffolds, test_nearest)
    guarded_test = np.asarray(raw_result["test_direct"], dtype=np.float64).copy()
    guarded_test[test_guard] = test_parent[test_guard]

    predictions = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if banked:
        mask = predictions["target_type"].astype(str).eq(ACTIVE_TARGET)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(
            dict(zip(test_rows["id"].astype(int), guarded_test, strict=True))
        ).to_numpy(np.float64)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941)):
        raise RuntimeError("C228 prediction ID coverage/order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C228 prediction finite check failed")

    component_test = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "parent": test_parent,
            "raw_candidate": raw_result["test_direct"],
            "candidate": guarded_test,
            "nearest_train_tanimoto": test_nearest,
            "scaffold": test_scaffolds,
            "guarded": test_guard,
            "banked": banked,
        }
    )
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
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(full_candidate_gate_pass and assembled_mean >= 0.95),
        "decision": "banked_component_pending_compound_audit" if banked else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "c208_runner": sha256_file(round2_root / "tools/round2_c208_tg_robust_group_measurement.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).loc[lambda frame: frame["target_type"].eq(ACTIVE_TARGET)].to_csv(
        run_dir / "tg_oof_predictions.csv", index=False
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component_test.to_csv(run_dir / "tg_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": carrier.SEED,
            "target": ACTIVE_TARGET,
            "feature_basis": "C127 official-SMILES/RDKit/Morgan carrier",
            "changed_factor": "C208 robust Tg carrier plus fixed fallback on predeclared negative transfer panels",
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "normal_component_gate": "delta >= 0.01, positive folds >= 4/5, grouped-bootstrap lower > 0, all explicit panel minima >= 0",
            "no_hyperparameter_sweep": True,
            "local_eval_read": False,
            "pi1m_used": False,
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
        f"# {run_dir.name}\n\n"
        f"Decision: **{report['decision']}**. Raw Tg delta `{raw_report['delta_r2']:+.12f}`; "
        f"guarded Tg delta `{active_report['delta_r2']:+.12f}`. Positive folds "
        f"`{active_report['positive_folds']}/5`; bootstrap lower `{active_report['group_bootstrap_lower']:.12f}`; "
        f"minimum panel delta `{active_report['minimum_panel_delta']:.12f}`. Banked: `{banked}`. "
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
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "tg_raw_delta_r2": raw_report["delta_r2"],
                "tg_guarded_delta_r2": active_report["delta_r2"],
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
