#!/usr/bin/env python3
"""C218: Nc canonical-group robust-response component.

This bounded official-only Nc child is queued after C217. It reuses the audited
C208 robust canonical-group measurement-noise machinery with the active target
switched to Nc. It changes one factor only: fold-local duplicate canonical-group
median targets and fixed MAD downweighting for Nc training rows. It does not
retune C212 Huber/rank features, use EPS partner labels, use PI1M, replay stored
predictions, or read local_eval/public feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import round2_c216_eps_high_tail_ordinal_residual as base
import round2_c200_clean_component_compound_audit_v3 as c200
import round2_c208_tg_robust_group_measurement as robust_base


SCHEMA = "ppp.round2.c218.nc-canonical-group-robust-response.v1"
ACTIVE_TARGET = "nc"
C218_ID = "R2-C218-20260805-0500-nc-high-tail-ordinal-residual-v1"
SEED = 20260821
HIGH_QUANTILE = 0.75
RESIDUAL_WEIGHT = 0.50
MIN_SELECTED_REFERENCE_DELTA = 0.010


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


def target_test_indices(parent: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_rows = (
        parent["test"]
        .loc[parent["test"]["target_type"].astype(str).eq(ACTIVE_TARGET)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(ACTIVE_TARGET)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
        raise RuntimeError("C218 Nc test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    return test_rows, indices, test_detail["target"].to_numpy(float)


def selected_nc_reference(root: Path) -> dict[str, Any]:
    """Read prior clean audit metrics only to set a stricter C218 replacement gate."""
    for run_id in (
        "R2-C217-20260805-0450-clean-component-compound-audit-v10",
        "R2-C215-20260805-0440-clean-component-compound-audit-v9",
        "R2-C213-20260805-0422-clean-component-compound-audit-v8",
        "R2-C211-20260805-0419-clean-component-compound-audit-v7",
    ):
        metrics = c200.load_json(c200.run_dir(root, run_id) / "metrics.json")
        if not isinstance(metrics, dict):
            continue
        nc = metrics.get("selected_components", {}).get(ACTIVE_TARGET, {})
        if isinstance(nc, dict) and nc.get("candidate_r2") is not None:
            return {
                "reference_source": f"{run_id}_selected_component",
                "run_id": nc.get("run_id"),
                "candidate_r2": float(nc.get("candidate_r2")),
            }
    return {
        "reference_source": "missing_reference_fail_closed",
        "run_id": None,
        "candidate_r2": None,
    }


def patch_base_module() -> None:
    base.ACTIVE_TARGET = ACTIVE_TARGET
    base.SEED = SEED
    base.HIGH_QUANTILE = HIGH_QUANTILE
    base.RESIDUAL_WEIGHT = RESIDUAL_WEIGHT
    base.MIN_SELECTED_PARENT_DELTA = MIN_SELECTED_REFERENCE_DELTA
    base.selected_eps_reference = selected_nc_reference
    base.eps_test_indices = target_test_indices


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    manifest = [
        f"{base.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in source_hashes.items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def patch_outputs(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = metrics.get("schema_version")
    metrics["schema_version"] = SCHEMA
    metrics["active_targets"] = [ACTIVE_TARGET]
    metrics["hypothesis"] = (
        "A fixed fold-local high-Nc ordinal classifier plus two residual heads can improve the high-refractive-index "
        "tail without reusing C212 Huber/rank stacking, EPS partner labels, stored predictions, or local_eval/public feedback."
    )
    metrics["selection_rule"] = (
        "one fixed Nc high-tail ordinal residual; no threshold/blend/model grid; no C212 stack retuning; "
        "no local_eval/public feedback"
    )
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c216_runner"] = base.sha256_file(Path(base.__file__).resolve())
    source_hashes["wrapper_runner"] = base.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes

    target_report = metrics.get("target_reports", {}).get(ACTIVE_TARGET, {})
    if isinstance(target_report, dict):
        target_report["changed_factor"] = "fixed Nc high-tail ordinal classifier plus two residual heads"
        target_report["selected_nc_reference"] = target_report.get("selected_eps_reference")
        target_report["delta_vs_selected_nc_reference"] = target_report.get("delta_vs_selected_eps_reference")
        target_report["beats_selected_nc_reference_gate"] = target_report.get("beats_selected_eps_reference_gate")
        if isinstance(target_report.get("regime_deltas"), dict):
            target_report["regime_deltas"] = {
                "nc_high_q75_delta": target_report["regime_deltas"].get("eps_high_q75_delta"),
                "nc_low_mid_delta": target_report["regime_deltas"].get("eps_low_mid_delta"),
            }
        target_report["no_c212_huber_rank_stack"] = True
        target_report["no_eps_partner_labels"] = True
        target_report["no_threshold_grid"] = True
        target_report["no_blend_grid"] = True

    # Preserve the base-generated files, but add target-named aliases so later
    # deterministic assemblers can align by the usual target-specific filename.
    generated_oof = run_dir / "eps_oof_predictions.csv"
    if generated_oof.exists():
        (run_dir / "nc_oof_predictions.csv").write_bytes(generated_oof.read_bytes())
    generated_component = run_dir / "eps_component_predictions.csv"
    if generated_component.exists():
        (run_dir / "nc_component_predictions.csv").write_bytes(generated_component.read_bytes())

    write_json(metrics_path, metrics)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "high_quantile": HIGH_QUANTILE,
            "residual_weight": RESIDUAL_WEIGHT,
            "n_estimators": base.N_ESTIMATORS,
            "minimum_selected_reference_delta": MIN_SELECTED_REFERENCE_DELTA,
            "selection_rule": metrics["selection_rule"],
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    report = metrics["target_reports"][ACTIVE_TARGET]
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: `{metrics.get('decision')}`. Nc parent `{float(report.get('parent_r2')):.12f}`; "
        f"candidate `{float(report.get('candidate_r2')):.12f}`; C050-relative delta "
        f"`{float(report.get('delta_r2')):+.12f}`; selected-reference delta "
        f"`{report.get('delta_vs_selected_nc_reference')}`. Mean parent "
        f"`{float(metrics.get('mean_parent_r2')):.12f}`; assembled "
        f"`{float(metrics.get('mean_candidate_r2')):.12f}`; gain "
        f"`{float(metrics.get('mean_gain')):+.12f}`. Official-only; no local_eval read; "
        "no Kaggle action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def patch_robust_base_module() -> None:
    robust_base.ACTIVE_TARGET = ACTIVE_TARGET


def patch_robust_outputs(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = metrics.get("schema_version")
    metrics["schema_version"] = SCHEMA
    metrics["hypothesis"] = (
        "Fold-local duplicate canonical-group median targets and fixed MAD downweighting can improve Nc label robustness "
        "without changing features, folds, model classes, fallback, or any non-Nc target."
    )
    metrics["selection_rule"] = (
        "one fixed Nc robust-response treatment; only Nc training targets/sample weights change inside each outer fold; "
        "no duplicate-support threshold grid; no C212 Huber/rank retuning; no cross-target labels; no local_eval/public feedback"
    )
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c208_runner"] = robust_base.sha256_file(Path(robust_base.__file__).resolve())
    source_hashes["wrapper_runner"] = robust_base.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    feature_report = metrics.get("feature_report", {})
    if isinstance(feature_report, dict):
        feature_report["changed_factor"] = "fold-local duplicate canonical-group Nc median targets and fixed dispersion downweighting"
        feature_report["active_target"] = ACTIVE_TARGET
        feature_report["uses_cross_property_labels"] = False
        feature_report["uses_pi1m"] = False
        feature_report["no_c212_huber_rank_stack"] = True
    target_report = metrics.get("target_reports", {}).get(ACTIVE_TARGET, {})
    if isinstance(target_report, dict):
        target_report["changed_factor"] = "fold-local duplicate canonical-group Nc median targets and fixed MAD downweighting"
        target_report["no_c212_huber_rank_stack"] = True
        target_report["no_eps_partner_labels"] = True
        target_report["no_threshold_grid"] = True
        target_report["no_model_class_change"] = True
    generated_component = run_dir / "tg_component_predictions.csv"
    if generated_component.exists():
        (run_dir / "nc_component_predictions.csv").write_bytes(generated_component.read_bytes())
    write_json(metrics_path, metrics)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": robust_base.carrier.SEED,
            "target": ACTIVE_TARGET,
            "feature_basis": "C127 official-SMILES/RDKit/Morgan carrier",
            "changed_factor": "fold-local canonical-no-stereo duplicate Nc group median labels plus fixed MAD downweighting",
            "normal_component_gate": "delta >= 0.01, positive folds >= 4/5, grouped-bootstrap lower > 0, all explicit panel minima >= 0",
            "no_hyperparameter_sweep": True,
            "local_eval_read": False,
            "pi1m_used": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    report = metrics["target_reports"][ACTIVE_TARGET]
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: **{metrics['decision']}**. Nc parent `{float(report['parent_r2']):.12f}`; "
        f"candidate `{float(report['candidate_r2']):.12f}`; delta `{float(report['delta_r2']):+.12f}`. "
        f"Positive folds `{int(report['positive_folds'])}/5`; bootstrap lower "
        f"`{float(report['group_bootstrap_lower']):.12f}`; minimum panel delta "
        f"`{float(report['minimum_panel_delta']):.12f}`. No local_eval/Kaggle/submission/final-notebook action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def main() -> None:
    run_dir = resolve_run_dir()
    patch_robust_base_module()
    robust_base.main()
    if run_dir is not None:
        patch_robust_outputs(run_dir)


if __name__ == "__main__":
    main()
