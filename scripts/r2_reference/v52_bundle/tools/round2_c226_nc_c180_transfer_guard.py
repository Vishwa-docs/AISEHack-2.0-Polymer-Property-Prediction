#!/usr/bin/env python3
"""C226: Nc C180 direct-carrier transfer guard.

This child reuses the already audited C207 C180-transfer-guard mechanics, but
changes the active target to Nc and freezes the guard from the prior C180
near-miss evidence.  C180's raw Nc carrier improved Nc by about +0.008 R2 but
failed the bank gate on bootstrap/panel fragility; C226 tests one bounded
factor: fixed C050 fallback on the predeclared negative Nc scaffold panel.

No local_eval external_labels, public feedback, PI1M, cross-target labels, stored
predictions, pretrained weights, Kaggle compute, upload, submission, or final
notebook action is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c207_egc_c180_transfer_guard as c207


SCHEMA = "ppp.round2.c226.nc-c180-transfer-guard.v1"
ACTIVE_TARGET = "nc"
GUARD_SIMILARITY_LT = 0.30
NEGATIVE_SCAFFOLDS = ("c1ccc(-c2cccs2)cc1",)
CHANGED_FACTOR = (
    "Regenerate the C180 direct Nc structure carrier from official inputs and apply fixed C050 fallback "
    "on the predeclared C180-negative Nc scaffold panel plus the existing low-similarity safety guard."
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


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    manifest = [
        f"{c207.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in source_hashes.items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def post_patch(run_dir: Path) -> None:
    old_component = run_dir / "egc_component_predictions.csv"
    new_component = run_dir / "nc_component_predictions.csv"
    if old_component.exists() and not new_component.exists():
        old_component.rename(new_component)

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = metrics.get("schema_version")
    metrics["schema_version"] = SCHEMA
    metrics["changed_factor"] = CHANGED_FACTOR
    metrics["active_target"] = ACTIVE_TARGET
    diagnostics = dict(metrics.get("component_diagnostics", {}))
    diagnostics.update(
        {
            "active_target": ACTIVE_TARGET,
            "changed_factor": CHANGED_FACTOR,
            "raw_c180_nc_delta_r2": diagnostics.pop("raw_c180_delta_r2", None),
            "guarded_nc_delta_r2": diagnostics.pop("guarded_delta_r2", None),
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "no_cross_target_labels": True,
            "no_pi1m": True,
            "no_local_eval_public_feedback": True,
            "sidecar_egb_proposal_rejected_reason": (
                "C180 direct Egb was only +0.0009247040838141762 while C180 direct Nc was the stronger "
                "unbanked bottleneck near-miss at +0.008054405518458374."
            ),
        }
    )
    metrics["component_diagnostics"] = diagnostics
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c207_runner"] = c207.sha256_file(Path(c207.__file__).resolve())
    source_hashes["wrapper_runner"] = c207.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)

    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "active_target": ACTIVE_TARGET,
            "source": "C180 regenerated Flory-Fox/oligomer/asymptotic carrier with fixed negative-panel C050 guard",
            "changed_factor": CHANGED_FACTOR,
            "guard_similarity_lt": GUARD_SIMILARITY_LT,
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "component_gate": {
                "minimum_delta_r2": 0.01,
                "minimum_positive_folds": 4,
                "bootstrap_lower_must_exceed": 0.0,
                "minimum_panel_delta_must_be_at_least": 0.0,
            },
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    nc_report = metrics.get("target_reports", {}).get("nc", {})
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: **{metrics.get('decision')}**. Nc guarded delta "
        f"`{float(nc_report.get('delta_r2', float('nan'))):+.12f}`. "
        f"Positive folds `{nc_report.get('positive_folds')}/5`; bootstrap lower "
        f"`{nc_report.get('group_bootstrap_lower')}`; minimum panel delta "
        f"`{nc_report.get('minimum_panel_delta')}`. Banked targets: "
        f"`{metrics.get('banked_targets', [])}`.\n\n"
        "Official-only; no local_eval, Kaggle compute, upload, submission, or final-notebook action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def main() -> None:
    run_dir = resolve_run_dir()
    c207.ACTIVE_TARGET = ACTIVE_TARGET
    c207.NEGATIVE_SCAFFOLDS = NEGATIVE_SCAFFOLDS
    c207.GUARD_SIMILARITY_LT = GUARD_SIMILARITY_LT
    c207.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
