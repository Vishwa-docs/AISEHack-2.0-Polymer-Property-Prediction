#!/usr/bin/env python3
"""C224: source-priority label aggregation for weak targets.

This child changes one factor: when current train and archive labels conflict
for a canonical structure/target, the candidate rebuild prefers current-train
labels instead of the C050 all-source median.  Model features, model classes,
folding mechanics, overrides, and hyperparameters remain the C050 parent route.

The output is a clean component candidate only.  It does not read local_eval
external_labels, public feedback, PI1M, pretrained weights, or stored predictions.
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
import round2_c222_structure_semantics_weaktarget as c222


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("ei", "nc", "eps")
SCHEMA = "ppp.round2.c224.source-priority-label-aggregation.v1"
MIN_SELECTED_REFERENCE_DELTA = 0.010


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


def source_priority_label_pool(train: pd.DataFrame, archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = train[["smiles", "canonical", "target_type", "target"]].copy()
    current["source"] = "current_train"
    old = archive[["smiles", "canonical", "target_type", "target"]].copy()
    old["source"] = "archive_train"
    raw = pd.concat([current, old], ignore_index=True)
    raw = raw.drop_duplicates(["smiles", "target_type", "target"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for (canonical, target_type), group in raw.groupby(["canonical", "target_type"], sort=True):
        current_group = group.loc[group["source"].eq("current_train")]
        selected = current_group if len(current_group) else group
        rows.append(
            {
                "canonical": canonical,
                "target_type": target_type,
                "target": float(selected["target"].median()),
                "smiles": str(selected["smiles"].iloc[0]),
                "measurements": int(len(group)),
            }
        )
    pooled = pd.DataFrame(rows, columns=["canonical", "target_type", "target", "smiles", "measurements"])
    return raw, pooled


def build_source_priority_parent(root: Path, data_dir: Path) -> dict[str, Any]:
    original = parent_builder.reference.build_label_pool
    parent_builder.reference.build_label_pool = source_priority_label_pool
    try:
        return parent_builder.build_parent(root, data_dir)
    finally:
        parent_builder.reference.build_label_pool = original


def source_priority_summary(data_dir: Path) -> dict[str, Any]:
    train, _, archive, _ = reference.load_inputs(data_dir)
    raw_default, pooled_default = reference.build_label_pool(train, archive)
    _, pooled_priority = source_priority_label_pool(train, archive)
    merged = pooled_default.merge(
        pooled_priority,
        on=["canonical", "target_type"],
        how="inner",
        suffixes=("_default", "_priority"),
        validate="one_to_one",
    )
    changed = np.abs(merged["target_default"].to_numpy(float) - merged["target_priority"].to_numpy(float)) > 1.0e-12
    by_target: dict[str, Any] = {}
    for target in TARGETS:
        selected = merged["target_type"].astype(str).eq(target).to_numpy()
        by_target[target] = {
            "rows": int(np.sum(selected)),
            "changed_rows": int(np.sum(selected & changed)),
            "max_abs_target_shift": float(
                np.max(np.abs(merged.loc[selected, "target_default"].to_numpy(float) - merged.loc[selected, "target_priority"].to_numpy(float)))
            )
            if int(np.sum(selected))
            else 0.0,
        }
    return {
        "raw_label_rows": int(len(raw_default)),
        "pooled_rows": int(len(pooled_default)),
        "changed_rows_total": int(np.sum(changed)),
        "changed_rows_by_target": by_target,
    }


def align_candidate(original_info: dict[str, Any], candidate_info: dict[str, Any]) -> np.ndarray:
    expected = pd.DataFrame(
        {
            "canonical": np.asarray(original_info["canonical"], dtype=object),
            "target": np.asarray(original_info["y"], dtype=float),
            "parent": np.asarray(original_info["parent"], dtype=float),
        }
    )
    observed = pd.DataFrame(
        {
            "canonical": np.asarray(candidate_info["canonical"], dtype=object),
            "candidate": np.asarray(candidate_info["parent"], dtype=float),
        }
    )
    merged = expected.merge(observed, on="canonical", how="left", validate="one_to_one")
    if merged["candidate"].isna().any():
        raise RuntimeError("C224 canonical alignment failed")
    return merged["candidate"].to_numpy(float)


def candidate_test_values(candidate_parent: dict[str, Any], target: str) -> pd.Series:
    frame = candidate_parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    selected = frame.loc[frame["target_type"].astype(str).eq(target), ["id", "target"]]
    if selected["id"].duplicated().any() or not np.isfinite(selected["target"].to_numpy(float)).all():
        raise RuntimeError(f"C224 invalid candidate test values for {target}")
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

    original_parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, original_parent, canonical_run)
    checkpoint(progress, "parent_parity", **parity)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    summary = source_priority_summary(data_dir)
    checkpoint(progress, "source_priority_summary", changed_rows_total=summary["changed_rows_total"])
    candidate_parent = build_source_priority_parent(root, data_dir)
    checkpoint(progress, "candidate_parent_complete")

    target_reports: dict[str, Any] = {}
    candidates_by_target: dict[str, np.ndarray] = {}
    banked: list[str] = []
    for target in ACTIVE_TARGETS:
        info = dict(original_parent["target_info"][target])
        info["fingerprints"] = original_parent["fingerprints"]
        candidate = align_candidate(info, candidate_parent["target_info"][target])
        report = carrier.evaluate_target(info, {"candidate": candidate})
        ref = c222.selected_reference(root, target)
        selected_r2 = float(ref["candidate_r2"]) if ref.get("candidate_r2") is not None else float(report["parent_r2"])
        delta_vs_selected = float(report["candidate_r2"] - selected_r2)
        replacement_gate = bool(report["pass"] and delta_vs_selected >= MIN_SELECTED_REFERENCE_DELTA)
        report.update(
            {
                "active": True,
                "changed_factor": "current-train-priority label aggregation in C050 parent rebuild",
                "model_family": "C050 parent route with fixed source-priority target aggregation",
                "selected_current_reference": ref,
                "delta_vs_selected_current_reference": delta_vs_selected,
                "minimum_selected_reference_delta": MIN_SELECTED_REFERENCE_DELTA,
                "replacement_gate_pass": replacement_gate,
                "no_feature_change": True,
                "no_model_hyperparameter_change": True,
                "no_cross_target_label_change": False,
            }
        )
        target_reports[target] = report
        candidates_by_target[target] = candidate
        if replacement_gate:
            banked.append(target)

    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(original_parent["target_info"][target])
        if target in ACTIVE_TARGETS:
            report = target_reports[target]
            candidate = candidates_by_target[target]
        else:
            report = c222.unchanged_report(info)
            candidate = np.asarray(info["parent"], dtype=float)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
        assembled = candidate if target in banked else np.asarray(info["parent"], dtype=float)
        frame = pd.DataFrame(
            {
                "canonical": info["canonical"],
                "target_type": target,
                "target": info["y"],
                "parent": info["parent"],
                "candidate": candidate,
                "group": info["groups"],
                "scaffold": info["scaffolds"],
                "fold": folds,
                "assembled": assembled,
            }
        )
        oof_parts.append(frame)
        if target in ACTIVE_TARGETS:
            frame[["canonical", "target", "parent", "candidate"]].to_csv(run_dir / f"{target}_oof_predictions.csv", index=False)

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    predictions = original_parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component_parts: list[pd.DataFrame] = []
    for target in ACTIVE_TARGETS:
        values = candidate_test_values(candidate_parent, target)
        component_parts.append(
            pd.DataFrame(
                {
                    "id": values.index.astype(int),
                    "target_type": target,
                    "candidate": values.to_numpy(float),
                }
            )
        )
        if target in banked:
            mask = predictions["target_type"].astype(str).eq(target)
            predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(values).to_numpy(float)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C224 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C224 produced non-finite predictions")

    report = {
        "schema_version": SCHEMA,
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": original_parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "pi1m_used": False,
        "stored_prediction_replay": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "active_targets": list(ACTIVE_TARGETS),
        "source_priority_summary": summary,
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": bool(assembled_mean - parent_mean >= 0.002 and bool(banked)),
        "goal_0_95_met": bool(assembled_mean >= 0.95 and bool(banked)),
        "decision": "candidate_pass_pending_clean_reproduction" if banked else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c222_helpers": sha256_file(round2_root / "tools/round2_c222_structure_semantics_weaktarget.py"),
        },
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.concat(component_parts, ignore_index=True).to_csv(run_dir / "component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "active_targets": list(ACTIVE_TARGETS),
            "changed_factor": "current-train-priority label aggregation for candidate parent rebuild",
            "selection_rule": "bank only active targets clearing normal and selected-reference gates",
            "official_only": True,
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
        f"# {run_dir.name}\n\n"
        f"Decision: `{report['decision']}`. Banked targets: `{','.join(banked) or 'none'}`. "
        f"Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; "
        f"gain `{assembled_mean - parent_mean:+.12f}`. Official-only; no local_eval read; no Kaggle action.\n",
        encoding="utf-8",
    )
    checkpoint(progress, "metrics_written", decision=report["decision"], banked_targets=banked, mean_candidate_r2=assembled_mean)
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
                "banked_targets": banked,
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
