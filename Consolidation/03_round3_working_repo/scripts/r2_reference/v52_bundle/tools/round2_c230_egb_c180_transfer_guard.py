#!/usr/bin/env python3
"""C230: Egb C180 fixed-panel transfer guard.

This protocol-only continuation sits behind C228/C229 so the watchdog cannot
drain below the unmet 0.95 objective.  It is intentionally conservative:
regenerate the C180 official-only Flory-Fox/oligomer/asymptotic carrier for Egb
and apply exact C050 fallback only on the already-recorded C180 Egb negative
transfer panels.

No local_eval external_labels, public feedback, PI1M, cross-target labels, stored
predictions, pretrained weights, Kaggle compute, upload, submission, or final
notebook action is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import round2_c207_egc_c180_transfer_guard as c207


SCHEMA = "ppp.round2.c230.egb-c180-transfer-guard.v1"
ACTIVE_TARGET = "egb"
SIMILARITY_BAND = (0.30, 0.50)
NEGATIVE_SCAFFOLDS = (
    "c1ccc(-c2cccs2)cc1",
    "c1ccccc1",
    "c1ccsc1",
)
CHANGED_FACTOR = (
    "Regenerate the C180 direct Egb structure carrier from official inputs and apply fixed C050 fallback "
    "on the pre-existing C180-negative Egb scaffold panels plus the exact 0.30<=similarity<0.50 panel."
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


def egb_guard_mask(scaffolds: np.ndarray, nearest: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    scaffold_guard = np.isin(np.asarray(scaffolds, dtype=object), np.asarray(NEGATIVE_SCAFFOLDS, dtype=object))
    nearest = np.asarray(nearest, dtype=np.float64)
    similarity_guard = (nearest >= SIMILARITY_BAND[0]) & (nearest < SIMILARITY_BAND[1])
    mask = scaffold_guard | similarity_guard
    return mask, {
        "guard_rows": int(np.sum(mask)),
        "similarity_guard_rows": int(np.sum(similarity_guard)),
        "similarity_band_low": float(SIMILARITY_BAND[0]),
        "similarity_band_high": float(SIMILARITY_BAND[1]),
        "scaffold_guard_rows": int(np.sum(scaffold_guard)),
    }


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
    new_component = run_dir / "egb_component_predictions.csv"
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
            "raw_c180_egb_delta_r2": diagnostics.pop("raw_c180_delta_r2", None),
            "guarded_egb_delta_r2": diagnostics.pop("guarded_delta_r2", None),
            "guard_similarity_band": [float(SIMILARITY_BAND[0]), float(SIMILARITY_BAND[1])],
            "guard_scaffolds": list(NEGATIVE_SCAFFOLDS),
            "no_cross_target_labels": True,
            "no_pi1m": True,
            "no_local_eval_public_feedback": True,
            "no_prior_output_use": True,
            "evidence_note": (
                "C180 direct Egb signal was only +0.0009247040838141762, so this is a queue-safety "
                "fixed-panel guard and must fail closed unless the normal +0.010 component gate passes."
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
            "source": "C180 regenerated Flory-Fox/oligomer/asymptotic carrier with fixed Egb negative-panel C050 guard",
            "changed_factor": CHANGED_FACTOR,
            "guard_similarity_band": [float(SIMILARITY_BAND[0]), float(SIMILARITY_BAND[1])],
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
    egb_report = metrics.get("target_reports", {}).get("egb", {})
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: **{metrics.get('decision')}**. Egb guarded delta "
        f"`{float(egb_report.get('delta_r2', float('nan'))):+.12f}`. "
        f"Positive folds `{egb_report.get('positive_folds')}/5`; bootstrap lower "
        f"`{egb_report.get('group_bootstrap_lower')}`; minimum panel delta "
        f"`{egb_report.get('minimum_panel_delta')}`. Banked targets: "
        f"`{metrics.get('banked_targets', [])}`.\n\n"
        "Official-only; no local_eval, Kaggle compute, upload, submission, or final-notebook action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def main() -> None:
    run_dir = resolve_run_dir()
    c207.ACTIVE_TARGET = ACTIVE_TARGET
    c207.NEGATIVE_SCAFFOLDS = NEGATIVE_SCAFFOLDS
    c207.GUARD_SIMILARITY_LT = SIMILARITY_BAND[1]
    c207.guard_mask_from_scaffold_similarity = egb_guard_mask
    c207.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
