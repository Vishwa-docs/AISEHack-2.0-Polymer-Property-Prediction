#!/usr/bin/env python3
"""C196: fixed Ei shrinkage audit of the C180 Flory--Fox near-miss.

This is a bounded official-only child queued before the deterministic C193
assembler.  It exists because C180 produced the only recent Ei signal that was
both target-positive and non-identity/non-PI1M/non-graph: Ei gained +0.01085
with 5/5 positive folds, but failed grouped-bootstrap and panel gates.

C196 changes exactly one scientific factor relative to the C180 Ei arm:

    Ei_candidate = C050_parent + 0.75 * (C180_FloryFox_Ei - C050_parent)

The fixed shrinkage is intended to test whether the near-miss was an
over-amplitude correction rather than a non-transferable signal.  There is no
alpha grid, no route tuning, no stored-prediction replay, and no local_eval/public
feedback.  If any standard component gate fails, no Ei target is banked and C193
must keep the exact C050 Ei fallback.
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


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "ei"
SHRINK_ALPHA = 0.75


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


def target_test_rows(parent: dict[str, Any], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_rows = (
        parent["test"]
        .loc[parent["test"]["target_type"].astype(str).eq(target)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(target)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
        raise RuntimeError(f"C196 {target} test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    parent_values = test_detail["target"].to_numpy(float)
    return test_rows, indices, parent_values


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
    test_rows, test_indices, test_parent = target_test_rows(parent, ACTIVE_TARGET)
    raw_result = carrier.fit_target(info, dense, sparse_features, test_indices, test_parent)
    raw_report = carrier.evaluate_target(info, raw_result)

    parent_oof = np.asarray(info["parent"], dtype=float)
    shrunk_candidate = parent_oof + SHRINK_ALPHA * (np.asarray(raw_result["candidate"], dtype=float) - parent_oof)
    shrunk_test = test_parent + SHRINK_ALPHA * (np.asarray(raw_result["test_direct"], dtype=float) - test_parent)
    target_report = carrier.evaluate_target(info, {"candidate": shrunk_candidate})
    target_report.update(
        {
            "shrink_alpha": SHRINK_ALPHA,
            "raw_c180_candidate_r2": raw_report["candidate_r2"],
            "raw_c180_delta_r2": raw_report["delta_r2"],
            "raw_c180_positive_folds": raw_report["positive_folds"],
            "raw_c180_group_bootstrap_lower": raw_report["group_bootstrap_lower"],
            "raw_c180_minimum_panel_delta": raw_report["minimum_panel_delta"],
            "blend_name": raw_result["blend_name"],
            "blend_weights": [float(value) for value in raw_result["weights"]],
            "blend_intercept": float(raw_result["intercept"]),
            "feature_rows": int(len(info["y"])),
            "test_rows": int(len(test_rows)),
            "changed_factor": "fixed 0.75 shrinkage of regenerated C180 Ei direct arm toward exact C050 parent",
        }
    )
    checkpoint(
        progress,
        "ei_shrinkage_complete",
        delta_r2=target_report["delta_r2"],
        positive_folds=target_report["positive_folds"],
        group_bootstrap_lower=target_report["group_bootstrap_lower"],
        minimum_panel_delta=target_report["minimum_panel_delta"],
        pass_gate=target_report["pass"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = target_report
            candidate = shrunk_candidate
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
                    "assembled": candidate if target == ACTIVE_TARGET and report["pass"] else target_info["parent"],
                }
            )
        )

    banked = [ACTIVE_TARGET] if target_report["pass"] else []
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    direct_component = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "raw_c180_direct_candidate": raw_result["test_direct"],
            "shrunk_direct_candidate": shrunk_test,
        }
    )
    predictions = parent_test.copy()
    if ACTIVE_TARGET in banked:
        values = direct_component.set_index("id")["shrunk_direct_candidate"]
        mask = predictions["target_type"].astype(str).eq(ACTIVE_TARGET)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(values).to_numpy(float)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if (
        len(predictions) != 4940
        or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941))
        or not np.isfinite(predictions["target"].to_numpy(float)).all()
    ):
        raise RuntimeError("C196 complete output contract failed")

    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003 and len(banked) > 0)
    report = {
        "schema_version": "ppp.round2.c196.ei-ffox-shrinkage-confirmation.v1",
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
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "goal_0_95_met": bool(full_pass and assembled_mean >= 0.95),
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
        "component_decision": "ei_component_banked" if ACTIVE_TARGET in banked else "ei_component_rejected",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
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
            "schema_version": "ppp.round2.c196.ei-ffox-shrinkage-confirmation.v1",
            "active_target": ACTIVE_TARGET,
            "source": "C180 Flory-Fox/oligomer feature family with one fixed parent-shrinkage factor",
            "shrink_alpha": SHRINK_ALPHA,
            "folds": "grouped no-stereo; exact C050 source replay as fallback",
            "banking": "Ei component gate before compound assembly; no alpha/grid/router tuning",
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
        f"Ei delta `{target_report['delta_r2']:+.12f}`; mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`. "
        "No local_eval, Kaggle compute, upload, or submission action.\n",
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
