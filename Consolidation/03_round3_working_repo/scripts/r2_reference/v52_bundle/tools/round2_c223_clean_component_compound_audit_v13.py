#!/usr/bin/env python3
"""C223: deterministic compound audit v13 with guarded C222 priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c221_clean_component_compound_audit_v12 as c221


SCHEMA = "ppp.round2.c223.clean-component-compound-audit.v13"
C216_ID = c221.C216_ID
C218_ID = c221.C218_ID
C220_ID = c221.C220_ID
C222_ID = "R2-C222-20260805-0540-structure-semantics-weaktarget-v1"
CHANGED_FACTOR = (
    "Insert C222 as first priority for Ei, Nc, and EPS under its selected-reference replacement gate, "
    "while preserving C221's guarded C220 Ei, C216 EPS, and C218 Nc priorities."
)


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
    ok, reason = c221.guarded_metric_passes(metrics, target)
    if not ok:
        return ok, reason
    if metrics.get("experiment_id") == C222_ID and target in {"ei", "nc", "eps"}:
        report = metrics.get("target_reports", {}).get(target, {})
        if report.get("replacement_gate_pass") is not True:
            return False, "c222_replacement_gate_not_pass"
        try:
            delta = float(report["delta_vs_selected_current_reference"])
        except (KeyError, TypeError, ValueError):
            return False, "c222_missing_selected_reference_delta"
        if delta < 0.010:
            return False, "c222_selected_reference_delta_below_gate"
    return True, "eligible"


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    manifest = [
        f"{c221.c219.c217.base.c200.sha256_file(path)}  {path.name}"
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
    metrics["component_priority"] = c221.c219.c217.base.COMPONENT_PRIORITY
    metrics["selection_rule"] = (
        "first completed clean-passing target component in frozen priority order; C222 first for Ei/Nc/EPS "
        "under its selected-reference replacement gate, then C220 for Ei, C216 for EPS, and C218 for Nc "
        "under their prior guards; no local_eval/public feedback; no same-OOF max search"
    )
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c213_runner"] = c221.c219.c217.base.c200.sha256_file(Path(c221.c219.c217.base.__file__).resolve())
    source_hashes["c217_wrapper_runner"] = c221.c219.c217.base.c200.sha256_file(Path(c221.c219.c217.__file__).resolve())
    source_hashes["c219_wrapper_runner"] = c221.c219.c217.base.c200.sha256_file(Path(c221.c219.__file__).resolve())
    source_hashes["c221_wrapper_runner"] = c221.c219.c217.base.c200.sha256_file(Path(c221.__file__).resolve())
    source_hashes["wrapper_runner"] = c221.c219.c217.base.c200.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "component_priority": c221.c219.c217.base.COMPONENT_PRIORITY,
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
    c221.c219.c217.base.COMPONENT_PRIORITY["eps"] = [
        C222_ID,
        C216_ID,
        "R2-C214-20260805-0440-eps-ionic-full-amplitude-v1",
        "R2-C190-20260805-0023-ionic-eps-reproduction-v3",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    c221.c219.c217.base.COMPONENT_PRIORITY["nc"] = [
        C222_ID,
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
    c221.c219.c217.base.COMPONENT_PRIORITY["ei"] = [
        C222_ID,
        C220_ID,
        "R2-C199-20260805-0254-ei-c196-transfer-guard-v1",
        "R2-C196-20260805-0225-ei-ffox-shrinkage-confirmation-v1",
        "R2-C194-20260805-0152-safe-ei-cross-property-stage2-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    c221.c219.c217.base.c203.metric_passes = guarded_metric_passes
    c221.c219.c217.base.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
