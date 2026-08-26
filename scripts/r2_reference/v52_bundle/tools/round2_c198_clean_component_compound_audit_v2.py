#!/usr/bin/env python3
"""C198: deterministic clean component compound audit v2.

This child does not introduce a new model family.  It runs after the queued
C197 child and assembles only targets whose own completed clean run passed its
preregistered component gate.  The priority order is frozen here to avoid
selecting the highest same-OOF number post hoc.

The output is a local audit candidate only.  It is not a final notebook, does
not read local_eval external_labels, and does not authorize Kaggle upload/submission.
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


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
MIN_COMPONENT_DELTA_R2 = 0.01
MIN_COMPONENT_POSITIVE_FOLDS = 4
MAX_ADJACENT_TARGET_LOSS_R2 = 0.003

# Fixed priority, not a same-OOF max selector.
COMPONENT_PRIORITY: dict[str, list[str]] = {
    "ei": [
        "R2-C196-20260805-0225-ei-ffox-shrinkage-confirmation-v1",
        "R2-C194-20260805-0152-safe-ei-cross-property-stage2-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
    "eea": [
        "R2-C189-20260804-ffox-eea-confirmation-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
    "nc": [
        "R2-C197-20260805-0237-nc-c195-consensus-gated-v1",
        "R2-C195-20260805-0215-nc-nearmiss-residual-diversity-v1",
        "R2-C191-20260805-0027-nested-predicted-eps-to-nc-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
    "eps": [
        "R2-C190-20260805-0023-ionic-eps-reproduction-v3",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_dir(root: Path, run_id: str) -> Path:
    return root / "Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY" / run_id


def metric_passes(metrics: dict[str, Any], target: str) -> tuple[bool, str]:
    if metrics.get("official_only") is not True:
        return False, "official_only_not_true"
    if metrics.get("external_label_file_read") is not False or metrics.get("local_eval_read") is not False:
        return False, "local_eval_or_external_label_read_flag"
    if (
        metrics.get("kaggle_compute") is not False
        or metrics.get("kaggle_upload") is not False
        or metrics.get("kaggle_submission", False) is not False
    ):
        return False, "kaggle_flag"
    if int(metrics.get("complete_output_rows", 0)) != 4940:
        return False, "incomplete_output_rows"
    if target not in set(metrics.get("banked_targets", [])):
        return False, "target_not_banked"
    target_report = metrics.get("target_reports", {}).get(target, {})
    if target_report.get("pass") is not True:
        return False, "target_report_not_pass"
    try:
        delta_r2 = float(target_report["delta_r2"])
        positive_folds = int(target_report["positive_folds"])
        bootstrap_lower = float(target_report["group_bootstrap_lower"])
    except (KeyError, TypeError, ValueError):
        return False, "missing_common_component_gate_fields"
    if delta_r2 < MIN_COMPONENT_DELTA_R2:
        return False, "target_delta_below_component_gate"
    if positive_folds < MIN_COMPONENT_POSITIVE_FOLDS:
        return False, "insufficient_positive_folds"
    if bootstrap_lower <= 0.0:
        return False, "group_bootstrap_lower_not_positive"
    for field in ("minimum_transfer_panel_delta", "minimum_panel_delta", "minimum_stratum_delta"):
        if field in target_report and target_report[field] is not None:
            try:
                if float(target_report[field]) < 0.0:
                    return False, f"{field}_negative"
            except (TypeError, ValueError):
                return False, f"{field}_not_numeric"
    if target_report.get("pair_delta_r2") is not None:
        try:
            if float(target_report["pair_delta_r2"]) < -MAX_ADJACENT_TARGET_LOSS_R2:
                return False, "paired_or_adjacent_target_loss"
        except (TypeError, ValueError):
            return False, "pair_delta_r2_not_numeric"
    parity = metrics.get("parent_replay_parity", {})
    if isinstance(parity, dict) and parity.get("pass") is not True:
        return False, "parent_parity_not_pass"
    return True, "eligible"


def target_oof_frame(component_dir: Path, target: str) -> pd.DataFrame | None:
    specific = component_dir / f"{target}_oof_predictions.csv"
    if specific.exists():
        return pd.read_csv(specific)
    if target == "eps" and (component_dir / "eps_oof_predictions.csv").exists():
        return pd.read_csv(component_dir / "eps_oof_predictions.csv")
    oof = component_dir / "oof_predictions.csv"
    if not oof.exists():
        return None
    frame = pd.read_csv(oof)
    if "target_type" not in frame.columns:
        return None
    return frame.loc[frame["target_type"].astype(str).eq(target)].reset_index(drop=True)


def aligned_candidate(
    parent: dict[str, Any],
    target: str,
    component_dir: Path,
) -> np.ndarray:
    frame = target_oof_frame(component_dir, target)
    if frame is None:
        raise RuntimeError(f"missing OOF predictions for {component_dir.name} target {target}")
    info = parent["target_info"][target]
    y = np.asarray(info["y"], dtype=float)
    parent_oof = np.asarray(info["parent"], dtype=float)
    if len(frame) != len(y):
        raise RuntimeError(f"OOF row count mismatch for {component_dir.name} {target}: {len(frame)} != {len(y)}")
    if (
        "canonical" in frame.columns
        and pd.Series(info["canonical"]).is_unique
        and frame["canonical"].is_unique
    ):
        expected = pd.DataFrame({
            "canonical": np.asarray(info["canonical"], dtype=object),
            "target": y,
            "parent": parent_oof,
        })
        observed = frame[["canonical", "target", "parent", "candidate"]].copy()
        merged = expected.merge(observed, on="canonical", how="left", suffixes=("_expected", "_observed"), validate="one_to_one")
        if merged["candidate"].isna().any():
            raise RuntimeError(f"canonical alignment failed for {component_dir.name} {target}")
        if np.max(np.abs(merged["target_expected"].to_numpy(float) - merged["target_observed"].to_numpy(float))) > 1.0e-10:
            raise RuntimeError(f"target mismatch for {component_dir.name} {target}")
        if np.max(np.abs(merged["parent_expected"].to_numpy(float) - merged["parent_observed"].to_numpy(float))) > 1.0e-10:
            raise RuntimeError(f"parent mismatch for {component_dir.name} {target}")
        return merged["candidate"].to_numpy(float)
    if np.max(np.abs(frame["target"].to_numpy(float) - y)) > 1.0e-10:
        raise RuntimeError(f"target sequence mismatch for {component_dir.name} {target}")
    if np.max(np.abs(frame["parent"].to_numpy(float) - parent_oof)) > 1.0e-10:
        raise RuntimeError(f"parent sequence mismatch for {component_dir.name} {target}")
    return frame["candidate"].to_numpy(float)


def full_prediction_values(parent: dict[str, Any], target: str, component_dir: Path) -> pd.Series:
    predictions = pd.read_csv(component_dir / "predictions.csv")
    test = parent["test"][["id", "target_type"]].copy().sort_values("id").reset_index(drop=True)
    if not {"id", "target"}.issubset(predictions.columns):
        raise RuntimeError(f"missing prediction columns for {component_dir.name}")
    if len(predictions) != len(test):
        raise RuntimeError(f"prediction row count mismatch for {component_dir.name}: {len(predictions)} != {len(test)}")
    prediction_ids = predictions["id"].astype(int)
    expected_ids = test["id"].astype(int)
    if not prediction_ids.is_unique:
        raise RuntimeError(f"duplicate prediction IDs for {component_dir.name}")
    if set(prediction_ids) != set(expected_ids):
        raise RuntimeError(f"prediction ID set mismatch for {component_dir.name}")
    merged = test.merge(predictions, on="id", how="left", validate="one_to_one")
    selected = merged.loc[merged["target_type"].astype(str).eq(target), ["id", "target"]].copy()
    if selected["target"].isna().any() or not np.isfinite(selected["target"].to_numpy(float)).all():
        raise RuntimeError(f"non-finite test values for {component_dir.name} {target}")
    return selected.set_index("id")["target"]


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
    output_dir = Path(args.run_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")

    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    selected: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    test_prediction = parent["test_parent_detail"][["id", "target_type", "target"]].copy()

    for target in TARGETS:
        info = parent["target_info"][target]
        y = np.asarray(info["y"], dtype=float)
        base = np.asarray(info["parent"], dtype=float)
        candidate = base.copy()
        source = "C050_parent"
        source_run: str | None = None
        if target in COMPONENT_PRIORITY:
            for run_id in COMPONENT_PRIORITY[target]:
                directory = run_dir(root, run_id)
                metrics = load_json(directory / "metrics.json")
                if metrics is None:
                    skipped.append({"target": target, "run_id": run_id, "reason": "missing_metrics"})
                    continue
                ok, reason = metric_passes(metrics, target)
                if not ok:
                    skipped.append({"target": target, "run_id": run_id, "reason": reason})
                    continue
                candidate = aligned_candidate(parent, target, directory)
                values = full_prediction_values(parent, target, directory)
                mask = test_prediction["target_type"].astype(str).eq(target)
                test_prediction.loc[mask, "target"] = test_prediction.loc[mask, "id"].astype(int).map(values).to_numpy(float)
                source = f"component:{run_id}"
                source_run = run_id
                break
        selected[target] = {
            "source": source,
            "run_id": source_run,
            "parent_r2": float(r2_score(y, base)),
            "candidate_r2": float(r2_score(y, candidate)),
            "delta_r2": float(r2_score(y, candidate) - r2_score(y, base)),
            "rows": int(len(y)),
        }
        oof_parts.append(pd.DataFrame({
            "target_type": target,
            "target": y,
            "parent": base,
            "candidate": candidate,
            "selected_source": source,
        }))

    oof = pd.concat(oof_parts, ignore_index=True)
    parent_mean = float(np.mean([selected[target]["parent_r2"] for target in TARGETS]))
    candidate_mean = float(np.mean([selected[target]["candidate_r2"] for target in TARGETS]))
    test_prediction = test_prediction[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(test_prediction) != 4940 or not np.array_equal(test_prediction["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C193 ID coverage/order contract failed")
    if not np.isfinite(test_prediction["target"].to_numpy(float)).all():
        raise RuntimeError("C193 produced non-finite predictions")
    max_loss = float(min(item["delta_r2"] for item in selected.values()))
    full_candidate_gate_pass = bool(candidate_mean - parent_mean >= 0.002 and max_loss >= -0.003)
    goal_0_95_met = bool(full_candidate_gate_pass and candidate_mean >= 0.95)
    report = {
        "schema_version": "ppp.round2.c198.clean-component-compound-audit.v2",
        "experiment_id": output_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "audit_only_not_final_notebook": True,
        "component_priority": COMPONENT_PRIORITY,
        "parent_replay_parity": parity,
        "selected_components": selected,
        "skipped_components": skipped,
        "banked_targets": [target for target, item in selected.items() if item["run_id"] is not None],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "gap_to_0_93": 0.93 - candidate_mean,
        "gap_to_0_95": 0.95 - candidate_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(test_prediction)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": goal_0_95_met,
        "decision": "compound_audit_goal_met_pending_notebook" if goal_0_95_met else "compound_audit_goal_unmet_continue_loop",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
            "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py"),
        },
    }
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_prediction.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "metrics.json", report)
    write_json(output_dir / "config.json", {
        "schema_version": "ppp.round2.c198.clean-component-compound-audit.v2",
        "component_priority": COMPONENT_PRIORITY,
        "selection_rule": "first completed clean-passing target component in frozen priority order; no local_eval/public feedback; no same-OOF max search",
        "local_eval_read": False,
    })
    (output_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
        f"rdkit={reference.Chem.rdBase.rdkitVersion}",
        f"platform={platform.platform()}",
    ]) + "\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "decision.md").write_text(
        f"# {output_dir.name}\n\nDecision: `{report['decision']}`. Mean parent `{parent_mean:.12f}`; compound `{candidate_mean:.12f}`; gain `{candidate_mean - parent_mean:+.12f}`. Gap to 0.95: `{0.95 - candidate_mean:+.12f}`. Audit-only; no local_eval read; no Kaggle action.\n",
        encoding="utf-8",
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (output_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": output_dir.name,
        "banked_targets": report["banked_targets"],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "goal_0_95_met": report["goal_0_95_met"],
        "decision": report["decision"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
