#!/usr/bin/env python3
"""C216: EPS high-tail ordinal residual.

This is a bounded official-only EPS child queued after C215.  It does not
change the C190/C214 ionic-coordinate family.  It tests one distinct factor:
whether a fixed fold-local high-EPS regime classifier plus two residual heads
can improve the EPS tail while retaining exact C050 fallback compatibility.
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as rich_builder
import round2_c187_ionic_eps_only as c187
import round2_c200_clean_component_compound_audit_v3 as c200


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "eps"
SEED = 20260805
HIGH_QUANTILE = 0.75
RESIDUAL_WEIGHT = 0.50
N_ESTIMATORS = 300
MIN_SELECTED_PARENT_DELTA = 0.010


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


def unchanged_report(info: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    return {
        "active": False,
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, parent)),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}},
        "folds": [],
        "pass": False,
        "unchanged_parent": True,
    }


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    selected = np.asarray(selected, dtype=bool)
    if int(np.sum(selected)) < 3:
        return None
    if float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))


def selected_eps_reference(root: Path) -> dict[str, Any]:
    """Read prior clean metrics only to set a stricter C216 replacement gate."""
    c215_metrics = c200.load_json(c200.run_dir(root, "R2-C215-20260805-0440-clean-component-compound-audit-v9") / "metrics.json")
    if isinstance(c215_metrics, dict):
        eps = c215_metrics.get("selected_components", {}).get(ACTIVE_TARGET, {})
        if eps.get("run_id") is not None:
            return {
                "reference_source": "c215_selected_component",
                "run_id": eps.get("run_id"),
                "candidate_r2": float(eps.get("candidate_r2")),
            }
    c211_metrics = c200.load_json(c200.run_dir(root, "R2-C211-20260805-0419-clean-component-compound-audit-v7") / "metrics.json")
    if isinstance(c211_metrics, dict):
        eps = c211_metrics.get("selected_components", {}).get(ACTIVE_TARGET, {})
        if eps.get("run_id") is not None:
            return {
                "reference_source": "c211_selected_component_fallback",
                "run_id": eps.get("run_id"),
                "candidate_r2": float(eps.get("candidate_r2")),
            }
    return {
        "reference_source": "c050_parent_fallback",
        "run_id": None,
        "candidate_r2": None,
    }


def eps_test_indices(parent: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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
        raise RuntimeError("C216 EPS test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    return test_rows, indices, test_detail["target"].to_numpy(float)


def fit_tail_residual(
    X_train: np.ndarray,
    y_train: np.ndarray,
    parent_train: np.ndarray,
    X_pred: np.ndarray,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    threshold = float(np.quantile(y_train, HIGH_QUANTILE))
    high = y_train >= threshold
    residual = y_train - parent_train
    if int(np.sum(high)) == 0 or int(np.sum(~high)) == 0:
        probability = np.full(len(X_pred), float(np.mean(high)), dtype=float)
    else:
        classifier = ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            max_features=0.65,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=fold_seed,
            n_jobs=2,
        )
        classifier.fit(X_train, high.astype(int))
        classes = list(classifier.classes_)
        if 1 in classes:
            probability = classifier.predict_proba(X_pred)[:, classes.index(1)]
        else:
            probability = np.zeros(len(X_pred), dtype=float)

    def fit_head(mask: np.ndarray, offset: int) -> ExtraTreesRegressor:
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_features=0.65,
            min_samples_leaf=2,
            random_state=fold_seed + offset,
            n_jobs=2,
        )
        model.fit(X_train[mask], residual[mask])
        return model

    low_model = fit_head(~high, 101)
    high_model = fit_head(high, 211)
    low_prediction = low_model.predict(X_pred)
    high_prediction = high_model.predict(X_pred)
    blended = probability * high_prediction + (1.0 - probability) * low_prediction
    return blended, probability, threshold, int(np.sum(high)), int(np.sum(~high))


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

    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    dense, sparse_matrix, feature_report = rich_builder.build_features(root, parent["keys"])
    dense = np.asarray(dense, dtype=np.float64)
    sparse_matrix = sparse_matrix.astype(np.float64)
    checkpoint(
        progress,
        "features_complete",
        dense_shape=feature_report["dense_shape"],
        sparse_shape=feature_report["sparse_shape"],
        sparse_nnz=feature_report["sparse_nnz"],
    )

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    y = np.asarray(info["y"], dtype=float)
    parent_oof = np.asarray(info["parent"], dtype=float)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = parent_oof.copy()
    probabilities = np.full(len(y), np.nan, dtype=float)
    fold_tail_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        Xtr, Xva = c187.fold_matrix(dense, sparse_matrix, indices[training], indices[validation])
        residual_pred, probability, threshold, high_rows, low_rows = fit_tail_residual(
            Xtr,
            y[training],
            parent_oof[training],
            Xva,
            SEED + fold,
        )
        lower_clip = float(np.quantile(y[training], 0.005) - 0.10)
        upper_clip = float(np.quantile(y[training], 0.995) + 0.10)
        candidate[validation] = np.clip(parent_oof[validation] + RESIDUAL_WEIGHT * residual_pred, lower_clip, upper_clip)
        probabilities[validation] = probability
        fold_tail_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(training)),
                "validation_rows": int(len(validation)),
                "threshold_q75": threshold,
                "high_train_rows": high_rows,
                "low_train_rows": low_rows,
                "mean_high_probability": float(np.mean(probability)),
            }
        )

    target_report = carrier.evaluate_target(info, {"candidate": candidate})
    high_panel = y >= np.quantile(y, HIGH_QUANTILE)
    low_panel = ~high_panel
    regime_deltas = {
        "eps_high_q75_delta": panel_delta(y, parent_oof, candidate, high_panel),
        "eps_low_mid_delta": panel_delta(y, parent_oof, candidate, low_panel),
    }
    evaluable_regime = [value for value in regime_deltas.values() if value is not None]
    minimum_regime_delta = min(evaluable_regime) if evaluable_regime else 0.0
    reference = selected_eps_reference(root)
    selected_reference_r2 = reference.get("candidate_r2")
    delta_vs_selected = None if selected_reference_r2 is None else float(target_report["candidate_r2"] - selected_reference_r2)
    beats_selected = bool(delta_vs_selected is not None and delta_vs_selected >= MIN_SELECTED_PARENT_DELTA)
    target_report.update(
        {
            "changed_factor": "fixed EPS high-tail ordinal classifier plus two residual heads",
            "high_quantile": HIGH_QUANTILE,
            "residual_weight": RESIDUAL_WEIGHT,
            "model_family": "ExtraTreesClassifier high-vs-rest plus separate ExtraTreesRegressor residual heads",
            "fold_tail_rows": fold_tail_rows,
            "probability_mean": float(np.nanmean(probabilities)),
            "probability_min": float(np.nanmin(probabilities)),
            "probability_max": float(np.nanmax(probabilities)),
            "regime_deltas": regime_deltas,
            "minimum_regime_delta": float(minimum_regime_delta),
            "selected_eps_reference": reference,
            "delta_vs_selected_eps_reference": delta_vs_selected,
            "beats_selected_eps_reference_gate": beats_selected,
            "normal_component_gate_pass": bool(target_report["pass"]),
            "replacement_gate_pass": bool(target_report["pass"] and beats_selected and minimum_regime_delta >= 0.0),
            "no_ionic_amplitude_change": True,
            "no_threshold_grid": True,
            "no_blend_grid": True,
        }
    )
    banked = [ACTIVE_TARGET] if target_report["replacement_gate_pass"] else []

    test_rows, test_indices, test_parent = eps_test_indices(parent)
    Xtr, Xt = c187.fold_matrix(dense, sparse_matrix, indices, test_indices)
    residual_pred, probability, threshold, high_rows, low_rows = fit_tail_residual(
        Xtr,
        y,
        parent_oof,
        Xt,
        SEED + 991,
    )
    lower_clip = float(np.quantile(y, 0.005) - 0.10)
    upper_clip = float(np.quantile(y, 0.995) + 0.10)
    test_candidate = np.clip(test_parent + RESIDUAL_WEIGHT * residual_pred, lower_clip, upper_clip)

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = target_report
            target_candidate = candidate
        else:
            report = unchanged_report(target_info)
            target_candidate = np.asarray(target_info["parent"], dtype=float)
        target_reports[target] = report
        target_folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": target_candidate,
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": target_folds,
                    "assembled": target_candidate if target in banked else target_info["parent"],
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    predictions = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if banked:
        mask = predictions["target_type"].astype(str).eq(ACTIVE_TARGET)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(
            pd.Series(test_candidate, index=test_rows["id"].astype(int))
        ).to_numpy(float)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C216 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C216 produced non-finite predictions")

    report = {
        "schema_version": "ppp.round2.c216.eps-high-tail-ordinal-residual.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
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
        "feature_report": feature_report,
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
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "rich_builder": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "c187_helpers": sha256_file(round2_root / "tools/round2_c187_ionic_eps_only.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
        },
    }
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "canonical": info["canonical"],
            "target": y,
            "parent": parent_oof,
            "candidate": candidate,
            "high_tail_probability": probabilities,
        }
    ).to_csv(run_dir / "eps_oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "parent": test_parent,
            "candidate": test_candidate,
            "high_tail_probability": probability,
            "threshold_q75": threshold,
            "high_train_rows": high_rows,
            "low_train_rows": low_rows,
        }
    ).to_csv(run_dir / "eps_component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": report["schema_version"],
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "high_quantile": HIGH_QUANTILE,
            "residual_weight": RESIDUAL_WEIGHT,
            "n_estimators": N_ESTIMATORS,
            "minimum_selected_parent_delta": MIN_SELECTED_PARENT_DELTA,
            "selection_rule": "one fixed EPS high-tail ordinal residual; no threshold/blend/model grid; no local_eval/public feedback",
            "local_eval_read": False,
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
        f"# {run_dir.name}\n\nDecision: `{report['decision']}`. "
        f"EPS parent `{target_report['parent_r2']:.12f}`; candidate `{target_report['candidate_r2']:.12f}`; "
        f"C050-relative delta `{target_report['delta_r2']:+.12f}`; "
        f"selected-reference delta `{target_report['delta_vs_selected_eps_reference']}`. "
        f"Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; gain `{assembled_mean - parent_mean:+.12f}`. "
        "Official-only; no local_eval read; no Kaggle action.\n",
        encoding="utf-8",
    )
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
                "eps_delta": target_report["delta_r2"],
                "delta_vs_selected_eps_reference": target_report["delta_vs_selected_eps_reference"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
