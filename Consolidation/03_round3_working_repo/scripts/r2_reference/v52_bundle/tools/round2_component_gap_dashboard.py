#!/usr/bin/env python3
"""Summarize clean Round 2 component evidence and remaining target gaps.

This is an audit/dashboard helper, not a selector.  It reads existing
`metrics.json` files in `experiments/CLEAN_OFFICIAL_ONLY`, reports clean
component-pass evidence, and quantifies the arithmetic gap to 0.93/0.95 from the
current C050-style parent.  It does not read local_eval external_labels, train models, infer
predictions, contact Kaggle, or authorize same-OOF max selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
BASELINE_PREFERRED_RUNS = (
    "R2-C187-20260804-ionic-eps-only-reproduction-v2",
    "R2-C180-20260804-ffox-oligomer-carriers-v2",
)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def clean_flags(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if metrics.get("official_only") is not True:
        errors.append("official_only_not_true")
    for key in ("external_label_file_read", "local_eval_read", "kaggle_compute", "kaggle_upload"):
        if metrics.get(key) is not False:
            errors.append(f"{key}_not_false")
    if metrics.get("kaggle_submission", False) is not False:
        errors.append("kaggle_submission_not_false")
    if int(metrics.get("complete_output_rows", 0)) != 4940:
        errors.append("complete_output_rows_not_4940")
    return not errors, errors


def baseline_scores(records: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, float]]:
    for run_id in BASELINE_PREFERRED_RUNS:
        metrics = records.get(run_id)
        reports = (metrics or {}).get("target_reports", {})
        if all(target in reports and "parent_r2" in reports[target] for target in TARGETS):
            return run_id, {target: float(reports[target]["parent_r2"]) for target in TARGETS}
    for run_id, metrics in sorted(records.items()):
        reports = metrics.get("target_reports", {})
        if all(target in reports and "parent_r2" in reports[target] for target in TARGETS):
            return run_id, {target: float(reports[target]["parent_r2"]) for target in TARGETS}
    return None, {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="experiments/CLEAN_OFFICIAL_ONLY")
    parser.add_argument("--queue", default="research/watchdog-queue.json")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(experiments_dir.glob("R2-C*/metrics.json")):
        metrics = load_json(path)
        if metrics is not None:
            records[path.parent.name] = metrics

    baseline_run, baseline = baseline_scores(records)
    component_passes: list[dict[str, Any]] = []
    top_positive: list[dict[str, Any]] = []
    for run_id, metrics in sorted(records.items()):
        clean_ok, clean_errors = clean_flags(metrics)
        banked_targets = set(metrics.get("banked_targets", []) or [])
        reports = metrics.get("target_reports", {}) or {}
        for target, report in reports.items():
            if not isinstance(report, dict) or "delta_r2" not in report:
                continue
            item = {
                "run_id": run_id,
                "target": target,
                "parent_r2": report.get("parent_r2"),
                "candidate_r2": report.get("candidate_r2"),
                "delta_r2": report.get("delta_r2"),
                "target_report_pass": report.get("pass"),
                "target_in_banked_targets": target in banked_targets,
                "clean_flags_pass": clean_ok,
                "clean_flag_errors": clean_errors,
                "decision": metrics.get("decision"),
            }
            try:
                if float(report.get("delta_r2", 0.0)) > 0:
                    top_positive.append(item)
            except (TypeError, ValueError):
                pass
            if clean_ok and target in banked_targets and report.get("pass") is True:
                component_passes.append(item)

    provisional_by_target: dict[str, dict[str, Any]] = {}
    for item in component_passes:
        target = str(item["target"])
        current = provisional_by_target.get(target)
        if current is None or float(item["candidate_r2"]) > float(current["candidate_r2"]):
            provisional_by_target[target] = item

    provisional_scores = dict(baseline)
    for target, item in provisional_by_target.items():
        if target in provisional_scores:
            provisional_scores[target] = float(item["candidate_r2"])

    queue = load_json(Path(args.queue)) or {}
    queued = [entry.get("run_id") for entry in queue.get("entries", []) if isinstance(entry, dict)]

    def mean_score(scores: dict[str, float]) -> float | None:
        if not all(target in scores for target in TARGETS):
            return None
        return sum(scores[target] for target in TARGETS) / len(TARGETS)

    baseline_mean = mean_score(baseline)
    provisional_mean = mean_score(provisional_scores)
    result = {
        "schema_version": "ppp.round2.component-gap-dashboard.v1",
        "audit_only_do_not_select_by_max": True,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "baseline_source_run": baseline_run,
        "baseline_scores": baseline,
        "baseline_mean": baseline_mean,
        "baseline_gap_to_0_93": None if baseline_mean is None else 0.93 - baseline_mean,
        "baseline_gap_to_0_95": None if baseline_mean is None else 0.95 - baseline_mean,
        "component_passes": component_passes,
        "provisional_by_target": provisional_by_target,
        "provisional_scores_if_all_passed_components_confirmed": provisional_scores,
        "provisional_mean_if_all_passed_components_confirmed": provisional_mean,
        "provisional_gap_to_0_93": None if provisional_mean is None else 0.93 - provisional_mean,
        "provisional_gap_to_0_95": None if provisional_mean is None else 0.95 - provisional_mean,
        "top_positive_deltas_for_review_only": sorted(
            top_positive,
            key=lambda item: float(item.get("delta_r2") or 0.0),
            reverse=True,
        )[:25],
        "queued_runs": queued,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
