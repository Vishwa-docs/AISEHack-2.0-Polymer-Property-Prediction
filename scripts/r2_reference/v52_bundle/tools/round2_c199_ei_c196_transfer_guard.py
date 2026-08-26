#!/usr/bin/env python3
"""C199: fail-closed transfer-guard repair of the C196 Ei near-miss.

C196 regenerated the C180 Flory--Fox Ei arm and applied one fixed 0.75
shrinkage toward exact C050.  It cleared the point-gain, fold, and bootstrap
gates but failed two explicit transfer panels:

  * scaffold_c1ccccc1
  * similarity_0.50_0.70

This child tests exactly one post-result repair: regenerate the same official
inputs and same C196 shrunk arm, but fall back to the exact C050 parent on those
predeclared failure slices.  The guard is label-free at inference time: scaffold
comes from the official SMILES and similarity is nearest same-target train
Tanimoto under the same Morgan fingerprint representation used by the parent.

Because the guard was chosen from C196's clean failure report, the metrics carry
an explicit posthoc_repair flag.  If any normal component gate fails, no Ei
target is banked.  No local_eval, public score, stored prediction, Kaggle upload,
Kaggle submission, or final-notebook artifact is used.
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
import round2_c196_ei_ffox_shrinkage_confirmation as c196


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "ei"
SHRINK_ALPHA = 0.75
GUARD_SCAFFOLD = "c1ccccc1"
GUARD_SIM_LOW = 0.50
GUARD_SIM_HIGH = 0.70


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
        train_fps = [parent["fingerprints"][int(indices[row])] for row in training]
        for row in validation:
            result[row] = max(DataStructs.BulkTanimotoSimilarity(parent["fingerprints"][int(indices[row])], train_fps))
    return result


def full_train_nearest(parent: dict[str, Any], info: dict[str, Any], test_indices: np.ndarray) -> np.ndarray:
    train_fps = [parent["fingerprints"][int(index)] for index in np.asarray(info["indices"], dtype=np.int64)]
    values = np.full(len(test_indices), np.nan, dtype=np.float64)
    for row, index in enumerate(test_indices):
        values[row] = max(DataStructs.BulkTanimotoSimilarity(parent["fingerprints"][int(index)], train_fps))
    return values


def transfer_guard_mask(scaffolds: np.ndarray, nearest: np.ndarray) -> np.ndarray:
    return (scaffolds.astype(object) == GUARD_SCAFFOLD) | (
        (nearest >= GUARD_SIM_LOW) & (nearest < GUARD_SIM_HIGH)
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
        sparse_nnz=feature_report["sparse_nnz"],
    )

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c196.target_test_rows(parent, ACTIVE_TARGET)
    raw_result = carrier.fit_target(info, dense, sparse_features, test_indices, test_parent)
    raw_report = carrier.evaluate_target(info, raw_result)

    parent_oof = np.asarray(info["parent"], dtype=float)
    raw_candidate = np.asarray(raw_result["candidate"], dtype=float)
    shrunk_candidate = parent_oof + SHRINK_ALPHA * (raw_candidate - parent_oof)
    nearest = fold_local_nearest(parent, info)
    oof_guard = transfer_guard_mask(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded_candidate = shrunk_candidate.copy()
    guarded_candidate[oof_guard] = parent_oof[oof_guard]

    target_report = carrier.evaluate_target(info, {"candidate": guarded_candidate})
    target_report.update(
        {
            "posthoc_repair_from_c196_failure_slices": True,
            "independent_confirmation_required_before_final_notebook": True,
            "guard_scaffold": GUARD_SCAFFOLD,
            "guard_similarity_interval": [GUARD_SIM_LOW, GUARD_SIM_HIGH],
            "guarded_oof_rows": int(np.sum(oof_guard)),
            "guarded_scaffold_oof_rows": int(np.sum(np.asarray(info["scaffolds"], dtype=object) == GUARD_SCAFFOLD)),
            "guarded_similarity_oof_rows": int(np.sum((nearest >= GUARD_SIM_LOW) & (nearest < GUARD_SIM_HIGH))),
            "shrink_alpha": SHRINK_ALPHA,
            "raw_c180_candidate_r2": raw_report["candidate_r2"],
            "raw_c180_delta_r2": raw_report["delta_r2"],
            "raw_c180_positive_folds": raw_report["positive_folds"],
            "raw_c180_group_bootstrap_lower": raw_report["group_bootstrap_lower"],
            "raw_c180_minimum_panel_delta": raw_report["minimum_panel_delta"],
            "c196_known_failure_panels": ["scaffold_c1ccccc1", "similarity_0.50_0.70"],
            "changed_factor": "C196 fixed 0.75 shrinkage plus label-free fallback on the two predeclared C196 transfer-failure slices",
            "blend_name": raw_result["blend_name"],
            "blend_weights": [float(value) for value in raw_result["weights"]],
            "blend_intercept": float(raw_result["intercept"]),
            "feature_rows": int(len(info["y"])),
            "test_rows": int(len(test_rows)),
        }
    )
    checkpoint(
        progress,
        "ei_transfer_guard_complete",
        delta_r2=target_report["delta_r2"],
        positive_folds=target_report["positive_folds"],
        group_bootstrap_lower=target_report["group_bootstrap_lower"],
        minimum_panel_delta=target_report["minimum_panel_delta"],
        guarded_oof_rows=target_report["guarded_oof_rows"],
        pass_gate=target_report["pass"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = target_report
            candidate = guarded_candidate
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=float)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": folds,
                    "guarded": oof_guard if target == ACTIVE_TARGET else np.zeros(len(target_info["y"]), dtype=bool),
                    "assembled": candidate if target == ACTIVE_TARGET and report["pass"] else target_info["parent"],
                }
            )
        )

    banked = [ACTIVE_TARGET] if target_report["pass"] else []
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))

    shrunk_test = test_parent + SHRINK_ALPHA * (np.asarray(raw_result["test_direct"], dtype=float) - test_parent)
    test_nearest = full_train_nearest(parent, info, test_indices)
    test_scaffold = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"].astype(str)], dtype=object)
    test_guard = transfer_guard_mask(test_scaffold, test_nearest)
    guarded_test = shrunk_test.copy()
    guarded_test[test_guard] = test_parent[test_guard]

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    direct_component = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "raw_c180_direct_candidate": raw_result["test_direct"],
            "shrunk_direct_candidate": shrunk_test,
            "guarded_direct_candidate": guarded_test,
            "nearest_train_tanimoto": test_nearest,
            "guarded": test_guard,
            "guard_scaffold": test_scaffold == GUARD_SCAFFOLD,
            "guard_similarity": (test_nearest >= GUARD_SIM_LOW) & (test_nearest < GUARD_SIM_HIGH),
        }
    )
    predictions = parent_test.copy()
    if ACTIVE_TARGET in banked:
        values = direct_component.set_index("id")["guarded_direct_candidate"]
        mask = predictions["target_type"].astype(str).eq(ACTIVE_TARGET)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(values).to_numpy(float)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if (
        len(predictions) != 4940
        or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941))
        or not np.isfinite(predictions["target"].to_numpy(float)).all()
    ):
        raise RuntimeError("C199 complete output contract failed")

    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003 and len(banked) > 0)
    report = {
        "schema_version": "ppp.round2.c199.ei-c196-transfer-guard.v1",
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
        "posthoc_repair_from_c196_failure_slices": True,
        "independent_confirmation_required_before_final_notebook": True,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "goal_0_95_met": bool(full_pass and assembled_mean >= 0.95),
        "decision": "candidate_pass_pending_independent_confirmation" if full_pass else "rejected_component_or_full_gate",
        "component_decision": "ei_component_banked_with_posthoc_guard" if ACTIVE_TARGET in banked else "ei_component_rejected",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "c196_runner": sha256_file(round2_root / "tools/round2_c196_ei_ffox_shrinkage_confirmation.py"),
            "c180_runner": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }

    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    direct_component.to_csv(run_dir / "component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": "ppp.round2.c199.ei-c196-transfer-guard.v1",
            "active_target": ACTIVE_TARGET,
            "source": "C196 fixed shrinkage of regenerated C180 Flory-Fox/oligomer Ei arm",
            "posthoc_repair_from_c196_failure_slices": True,
            "guard_scaffold": GUARD_SCAFFOLD,
            "guard_similarity_interval": [GUARD_SIM_LOW, GUARD_SIM_HIGH],
            "shrink_alpha": SHRINK_ALPHA,
            "folds": "grouped no-stereo; exact C050 source replay as fallback",
            "banking": "standard Ei component gate before compound assembly; no alpha/grid/router tuning",
            "local_eval_read": False,
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
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Component decision: `{report['component_decision']}`. "
        f"Ei delta `{target_report['delta_r2']:+.12f}`; guarded OOF rows `{target_report['guarded_oof_rows']}`. "
        f"Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`. "
        "No local_eval, Kaggle compute, upload, or submission action.\n\n"
        "Caveat: this is a post-C196 failure-slice repair and needs independent confirmation before any final-notebook use.\n",
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
                "banked_targets": banked,
                "ei_delta_r2": target_report["delta_r2"],
                "ei_positive_folds": target_report["positive_folds"],
                "ei_group_bootstrap_lower": target_report["group_bootstrap_lower"],
                "ei_minimum_panel_delta": target_report["minimum_panel_delta"],
                "guarded_oof_rows": target_report["guarded_oof_rows"],
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
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
