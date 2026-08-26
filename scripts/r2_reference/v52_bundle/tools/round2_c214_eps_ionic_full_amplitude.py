#!/usr/bin/env python3
"""C214: one-shot full-amplitude EPS ionic-coordinate test.

This wrapper intentionally reuses the formal C187/C190 ionic EPS runner and
changes one scientific factor: the parent/raw ionic blend amplitude is set from
0.50 to 1.00. It does not tune alpha, thresholds, folds, features, or fallback
slices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import round2_c187_ionic_eps_only as base


SCHEMA = "ppp.round2.c214.eps-ionic-full-amplitude.v1"
BASE_SCHEMA = "ppp.round2.c187.ionic-eps-only.v1"
CHANGED_FACTOR = "Set C187/C190 EPS ionic-coordinate parent/raw blend amplitude from 0.50 to 1.00; no alpha grid, no fallback retuning."


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
        f"{base.sha256_file(path)}  {path.name}"
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
    metrics["full_amplitude_ionic_coordinate"] = True
    metrics["no_alpha_grid"] = True
    metrics["no_fallback_slice_tuning"] = True
    metrics["half_parent_blend"] = 1.0
    metrics.setdefault("component_family", "eps_ionic_coordinate")
    metrics.setdefault("notes", [])
    if isinstance(metrics["notes"], list):
        metrics["notes"].append("C214 is a wrapper around C187 with only HALF_PARENT changed to 1.0 before execution.")
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c187_runner"] = base.sha256_file(Path(base.__file__).resolve())
    source_hashes["wrapper_runner"] = base.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)

    config = {
        "schema_version": SCHEMA,
        "base_schema_version": BASE_SCHEMA,
        "seed": base.SEED,
        "pair_rows": metrics.get("pair_rows"),
        "model_kinds": list(base.MODEL_KINDS),
        "half_parent_blend": 1.0,
        "changed_factor": CHANGED_FACTOR,
        "selection_rule": "one-shot full-amplitude EPS ionic-coordinate test; no alpha grid, no fallback/slice retuning, no local_eval/public feedback",
        "official_only": True,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
    }
    write_json(run_dir / "config.json", config)

    decision = metrics.get("decision", "unknown")
    eps_report = metrics.get("target_reports", {}).get("eps", {})
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: `{decision}`. C214 changed exactly one factor: `{CHANGED_FACTOR}`\n\n"
        f"EPS parent `{float(eps_report.get('parent_r2', float('nan'))):.12f}`; "
        f"candidate `{float(eps_report.get('candidate_r2', float('nan'))):.12f}`; "
        f"delta `{float(eps_report.get('delta_r2', float('nan'))):+.12f}`. "
        f"Mean parent `{float(metrics.get('mean_parent_r2', float('nan'))):.12f}`; "
        f"assembled `{float(metrics.get('mean_candidate_r2', float('nan'))):.12f}`; "
        f"gain `{float(metrics.get('mean_gain', float('nan'))):+.12f}`. "
        "Official-only; no local_eval read; no Kaggle action; no final-notebook action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def main() -> None:
    run_dir = resolve_run_dir()
    base.HALF_PARENT = 1.0
    base.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
