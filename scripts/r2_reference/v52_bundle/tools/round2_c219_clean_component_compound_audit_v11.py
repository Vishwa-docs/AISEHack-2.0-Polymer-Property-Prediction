#!/usr/bin/env python3
"""C219: deterministic compound audit v11 with guarded C218 Nc priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c217_clean_component_compound_audit_v10 as c217


SCHEMA = "ppp.round2.c219.clean-component-compound-audit.v11"
C216_ID = c217.C216_ID
C218_ID = "R2-C218-20260805-0500-nc-high-tail-ordinal-residual-v1"
CHANGED_FACTOR = (
    "Insert C218 as first Nc priority under the normal component gate, while preserving C217's guarded C216 EPS priority."
)
ORIGINAL_METRIC_PASSES = c217.ORIGINAL_METRIC_PASSES


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def resolve_run_dir() -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir")
    args, _ = parser.parse_known_args()
    if args.run_dir is None:
        return None
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    return run_dir.resolve() if run_dir.is_absolute() else (root / run_dir).resolve()


def guarded_metric_passes(metrics: dict[str, Any], target: str) -> tuple[bool, str]:
    ok, reason = ORIGINAL_METRIC_PASSES(metrics, target)
    if not ok:
        return ok, reason
    if target == "eps" and metrics.get("experiment_id") == C216_ID:
        report = metrics.get("target_reports", {}).get("eps", {})
        if report.get("replacement_gate_pass") is not True:
            return False, "c216_replacement_gate_not_pass"
        try:
            delta = float(report["delta_vs_selected_eps_reference"])
        except (KeyError, TypeError, ValueError):
            return False, "c216_missing_selected_reference_delta"
        if delta < 0.010:
            return False, "c216_selected_reference_delta_below_gate"
    return True, "eligible"


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    manifest = [
        f"{c217.base.c200.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in source_hashes.items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def post_patch(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = metrics.get("schema_version")
    metrics["schema_version"] = SCHEMA
    metrics["changed_factor"] = CHANGED_FACTOR
    metrics["component_priority"] = c217.base.COMPONENT_PRIORITY
    metrics["selection_rule"] = (
        "first completed clean-passing target component in frozen priority order; C216 first for EPS under "
        "a selected-reference replacement gate and C218 first for Nc under the normal component gate; no local_eval/public feedback; "
        "no same-OOF max search"
    )
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c213_runner"] = c217.base.c200.sha256_file(Path(c217.base.__file__).resolve())
    source_hashes["c217_wrapper_runner"] = c217.base.c200.sha256_file(Path(c217.__file__).resolve())
    source_hashes["wrapper_runner"] = c217.base.c200.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "component_priority": c217.base.COMPONENT_PRIORITY,
            "changed_factor": CHANGED_FACTOR,
            "selection_rule": metrics["selection_rule"],
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: `{metrics.get('decision')}`. {CHANGED_FACTOR}\n\n"
        f"Mean parent `{float(metrics.get('mean_parent_r2', float('nan'))):.12f}`; "
        f"compound `{float(metrics.get('mean_candidate_r2', float('nan'))):.12f}`; "
        f"gain `{float(metrics.get('mean_gain', float('nan'))):+.12f}`. "
        f"Gap to 0.95: `{float(metrics.get('gap_to_0_95', float('nan'))):+.12f}`. "
        "Audit-only; no local_eval read; no Kaggle action; no final-notebook action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def main() -> None:
    run_dir = resolve_run_dir()
    c217.base.COMPONENT_PRIORITY["eps"] = [
        C216_ID,
        "R2-C214-20260805-0440-eps-ionic-full-amplitude-v1",
        "R2-C190-20260805-0023-ionic-eps-reproduction-v3",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    c217.base.COMPONENT_PRIORITY["nc"] = [
        C218_ID,
        "R2-C212-20260805-0422-nc-robust-rank-loss-v1",
        "R2-C210-20260805-0415-nc-optical-dispersion-gap-v1",
        "R2-C202-20260805-0315-nc-support-uncertainty-refractivity-v1",
        "R2-C197-20260805-0237-nc-c195-consensus-gated-v1",
        "R2-C195-20260805-0215-nc-nearmiss-residual-diversity-v1",
        "R2-C191-20260805-0027-nested-predicted-eps-to-nc-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    c217.base.c203.metric_passes = guarded_metric_passes
    c217.base.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
