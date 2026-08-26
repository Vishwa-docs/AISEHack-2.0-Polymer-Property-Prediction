#!/usr/bin/env python3
"""C215: deterministic compound audit v9 with C214 as first EPS priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c213_clean_component_compound_audit_v8 as base


SCHEMA = "ppp.round2.c215.clean-component-compound-audit.v9"
BASE_SCHEMA = "ppp.round2.c213.clean-component-compound-audit.v8"
CHANGED_FACTOR = "Insert C214 full-amplitude EPS ionic-coordinate component as first EPS priority before C190; preserve strict component eligibility."
C214_ID = "R2-C214-20260805-0440-eps-ionic-full-amplitude-v1"


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


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    manifest = [
        f"{base.c200.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in source_hashes.items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def post_patch(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = metrics.get("schema_version", BASE_SCHEMA)
    metrics["schema_version"] = SCHEMA
    metrics["changed_factor"] = CHANGED_FACTOR
    metrics["audit_only_not_final_notebook"] = True
    metrics["component_priority"] = base.COMPONENT_PRIORITY
    metrics["selection_rule"] = "first completed clean-passing target component in frozen priority order; C214 first for EPS; no local_eval/public feedback; no same-OOF max search"
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c213_runner"] = base.c200.sha256_file(Path(base.__file__).resolve())
    source_hashes["wrapper_runner"] = base.c200.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)

    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "base_schema_version": BASE_SCHEMA,
            "component_priority": base.COMPONENT_PRIORITY,
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
    base.COMPONENT_PRIORITY["eps"] = [
        C214_ID,
        "R2-C190-20260805-0023-ionic-eps-reproduction-v3",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    base.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
