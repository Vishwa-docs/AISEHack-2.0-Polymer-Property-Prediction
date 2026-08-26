#!/usr/bin/env python3
"""C240: Nc electro-polar topological autocorrelation residual.

This is a queue-safety child after C239.  It reuses the fixed, already audited
C220 atom-channel graph-distance autocorrelation feature generator, but changes
the active target to unbanked Nc.  It is materially distinct from the recent Nc
branches: no replicate-reliability targets, no C180 transfer guard, no
backbone/pendant partition, no optical-dispersion coordinates, no robust-rank
loss, no EPS counterpart labels, no PI1M, and no stored predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c220_ei_electro_polar_autocorr as base


SCHEMA = "ppp.round2.c240.nc-electro-polar-autocorr.v1"
BASE_SCHEMA = "ppp.round2.c220.ei-electro-polar-autocorr.v1"
ACTIVE_TARGET = "nc"
CHANGED_FACTOR = (
    "Retarget the fixed C220 electro-polar topological autocorrelation residual "
    "from Ei to unbanked Nc, preserving max_lag, Ridge alpha, residual weight, "
    "folds, model class, and gates."
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
        f"{base.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in sorted(source_hashes.items()):
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def post_patch(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = BASE_SCHEMA
    metrics["schema_version"] = SCHEMA
    metrics["active_target"] = ACTIVE_TARGET
    metrics["changed_factor"] = CHANGED_FACTOR
    metrics["retargeted_from_c220_ei"] = True
    metrics["uses_cross_target_labels"] = False
    metrics["uses_pi1m"] = False
    metrics["uses_stored_prediction_replay"] = False
    nc_report = metrics.get("target_reports", {}).get(ACTIVE_TARGET, {})
    if isinstance(nc_report, dict):
        if "selected_ei_reference" in nc_report:
            nc_report["selected_nc_reference"] = nc_report.pop("selected_ei_reference")
        if "delta_vs_selected_ei_reference" in nc_report:
            nc_report["delta_vs_selected_nc_reference"] = nc_report.pop("delta_vs_selected_ei_reference")
        if "beats_selected_ei_reference_gate" in nc_report:
            nc_report["beats_selected_nc_reference_gate"] = nc_report.pop("beats_selected_ei_reference_gate")
        nc_report["changed_factor"] = CHANGED_FACTOR
        nc_report["model_family"] = "Ridge residual over fixed electro-polar graph-distance autocorrelation features"
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c220_runner"] = base.sha256_file(Path(base.__file__).resolve())
    source_hashes["wrapper_runner"] = base.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)

    old_oof = run_dir / "ei_oof_predictions.csv"
    new_oof = run_dir / "nc_oof_predictions.csv"
    if old_oof.exists() and not new_oof.exists():
        old_oof.rename(new_oof)
    old_component = run_dir / "ei_component_predictions.csv"
    new_component = run_dir / "nc_component_predictions.csv"
    if old_component.exists() and not new_component.exists():
        old_component.rename(new_component)

    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "base_schema_version": BASE_SCHEMA,
            "seed": base.SEED,
            "active_target": ACTIVE_TARGET,
            "max_lag": base.MAX_LAG,
            "ridge_alpha": base.RIDGE_ALPHA,
            "residual_weight": base.RESIDUAL_WEIGHT,
            "minimum_c050_delta": base.MIN_C050_DELTA,
            "minimum_selected_reference_delta": base.MIN_SELECTED_REFERENCE_DELTA,
            "changed_factor": CHANGED_FACTOR,
            "selection_rule": "one fixed Nc electro-polar autocorrelation residual; no lag/alpha/weight/model grid; no local_eval/public feedback",
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    selected_delta = float(nc_report.get("delta_vs_selected_nc_reference", float("nan"))) if isinstance(nc_report, dict) else float("nan")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: `{metrics.get('decision')}`. {CHANGED_FACTOR}\n\n"
        f"Nc parent `{float(nc_report.get('parent_r2', float('nan'))):.12f}`; "
        f"candidate `{float(nc_report.get('candidate_r2', float('nan'))):.12f}`; "
        f"C050/selected-reference delta `{selected_delta:+.12f}`. "
        f"Mean parent `{float(metrics.get('mean_parent_r2', float('nan'))):.12f}`; "
        f"assembled `{float(metrics.get('mean_candidate_r2', float('nan'))):.12f}`; "
        f"gain `{float(metrics.get('mean_gain', float('nan'))):+.12f}`. "
        "Official-only; no local_eval read; no Kaggle action; no final-notebook action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def main() -> None:
    run_dir = resolve_run_dir()
    base.ACTIVE_TARGET = ACTIVE_TARGET
    base.SCHEMA = SCHEMA
    base.SEED = 20260805
    base.C199_REFERENCE_EI_R2 = 0.8397322432486007
    base.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
