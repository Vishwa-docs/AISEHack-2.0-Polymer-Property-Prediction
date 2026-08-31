#!/usr/bin/env python3
"""C189: Eea-only confirmation wrapper for the C180 Flory--Fox carrier.

The C189 protocol is intentionally narrower than C180: it may evaluate all
targets for exact C050 parent accounting, but only Eea can be changed or banked.
This prevents a protocol/script mismatch where the generic C180 runner could
promote a non-Eea target in an Eea-confirmation child.
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
ACTIVE_TARGET = "eea"
SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")

    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    dense, sparse_features, feature_report = c180.build_features(root, parent["keys"])
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    direct_test_parts: list[pd.DataFrame] = []

    for target in TARGETS:
        info = dict(parent["target_info"][target])
        info["fingerprints"] = parent["fingerprints"]
        folds = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
        if target == ACTIVE_TARGET:
            test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
                raise RuntimeError(f"C189 test ID alignment failed for {target}")
            test_indices = np.asarray(
                [
                    parent["key_to_index"][value]
                    for value in test_rows["canonical"]
                ],
                dtype=np.int64,
            )
            result = carrier.fit_target(info, dense, sparse_features, test_indices, test_detail["target"].to_numpy(float))
            report = carrier.evaluate_target(info, result)
            report.update({
                "active": True,
                "blend_name": result["blend_name"],
                "blend_weights": [float(value) for value in result["weights"]],
                "blend_intercept": result["intercept"],
                "feature_rows": int(len(info["y"])),
            })
            candidate = np.asarray(result["candidate"], dtype=float)
            direct_test_parts.append(pd.DataFrame({
                "id": test_rows["id"].astype(int),
                "target_type": target,
                "direct_candidate": result["test_direct"],
            }))
        else:
            candidate = np.asarray(info["parent"], dtype=float)
            report = {
                "active": False,
                "parent_r2": float(r2_score(info["y"], info["parent"])),
                "candidate_r2": float(r2_score(info["y"], info["parent"])),
                "delta_r2": 0.0,
                "positive_folds": 0,
                "group_bootstrap_lower": None,
                "minimum_panel_delta": 0.0,
                "pass": False,
            }
        target_reports[target] = report
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": info["y"],
            "parent": info["parent"],
            "candidate": candidate,
            "assembled": candidate if target == ACTIVE_TARGET and report["pass"] else info["parent"],
            "group": info["groups"],
            "scaffold": info["scaffolds"],
            "fold": folds,
        }))

    banked = [ACTIVE_TARGET] if target_reports[ACTIVE_TARGET]["pass"] else []
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if direct_test_parts:
        direct_test = pd.concat(direct_test_parts, ignore_index=True)
        predictions = parent_test.merge(direct_test, on=["id", "target_type"], how="left", validate="one_to_one")
        predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["direct_candidate"], predictions["target"])
    else:
        direct_test = pd.DataFrame(columns=["id", "target_type", "direct_candidate"])
        predictions = parent_test.copy()
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C189 complete output contract failed")

    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003 and banked)
    report = {
        "schema_version": "ppp.round2.c189.ffox-eea-confirmation.v1",
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
        "active_targets": [ACTIVE_TARGET],
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "test_feature_order": "active_target_indices_derived_from_sorted_id_target_slice",
        "test_parent_blend": "C189 passes sorted C050 parent predictions into carrier.fit_target so test predictions use the same parent/ridge/tree blend as OOF",
        "full_candidate_gate_pass": full_pass,
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "c180_feature_runner": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    direct_test.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.c189.ffox-eea-confirmation.v1",
        "seed": SEED,
        "source": "Eea-only confirmation of C180 Flory-Fox-style asymptotic descriptors",
        "active_targets": [ACTIVE_TARGET],
        "flory_fox_max_repeats": c180.FFOX_MAX_REPEATS,
        "flory_fox_transform": c180.FFOX_TRANSFORM,
        "folds": "grouped no-stereo; exact C050 source replay as fallback",
        "banking": "Eea-only component gate before compound assembly",
        "local_eval_read": False,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
        f"rdkit={reference.Chem.rdBase.rdkitVersion}",
        f"platform={platform.platform()}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Eea banked: `{bool(banked)}`. Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; gain `{assembled_mean - parent_mean:+.12f}`. No local_eval read.\n",
        encoding="utf-8",
    )
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": run_dir.name,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "decision": report["decision"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
