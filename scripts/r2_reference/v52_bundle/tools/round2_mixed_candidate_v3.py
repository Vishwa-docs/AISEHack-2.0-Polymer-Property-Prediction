#!/usr/bin/env python3
"""Clean mixed seven-target candidate regenerated from official Round 2 inputs.

This script deliberately does not load component predictions or component runtime
artifacts. It rebuilds the C001 reference pipelines and the fixed Ei/Eea gap
routes from source data in one process; a notebook can embed this same source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_ei_scaffold_abstaining_gap_identity_v4 as ei_route
import round2_eea_scaffold_abstaining_gap_identity_v7 as eea_route


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
SPECIAL_TARGETS = ("ei", "eea")
SIMILARITY_BARRIER = 0.70
BLOCKED_SCAFFOLDS = {"c1ccsc1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parent_test_prediction(
    target: str,
    y: np.ndarray,
    groups: np.ndarray,
    target_dense: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    test_global_indices: np.ndarray,
    y_global: np.ndarray,
    config: dict[str, object],
    route_module: object,
) -> np.ndarray:
    inner_folds = route_module.folds_for(groups, 4)
    inner_arms = np.full((len(y), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = np.flatnonzero(inner_folds != fold)
        local_validation = np.flatnonzero(inner_folds == fold)
        inner_arms[local_validation] = reference.predict_base_models(
            target_dense,
            sparse_parts,
            fingerprints,
            y_global,
            global_indices[local_train],
            global_indices[local_validation],
            config,
            target,
        )
    weights, intercept, _, _ = reference.blend_from_oof(y, inner_arms)
    test_arms = reference.predict_base_models(
        target_dense,
        sparse_parts,
        fingerprints,
        y_global,
        global_indices,
        test_global_indices,
        config,
        target,
    )
    return reference.clip_prediction(y, test_arms @ weights + intercept)


def specialized_target(
    target: str,
    pooled: pd.DataFrame,
    test: pd.DataFrame,
    keys: list[str],
    dense_base: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    route_module = ei_route if target == "ei" else eea_route
    key_to_index = {key: index for index, key in enumerate(keys)}
    frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].to_numpy(object)
    groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    target_dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
    gap_table = route_module.make_gap_table(pooled)
    maps = {
        name: dict(zip(part["canonical"], part["target"].astype(float), strict=True))
        for name in ("ei", "eea", "egc")
        for part in [pooled[pooled["target_type"] == name].reset_index(drop=True)]
    }
    folds = route_module.folds_for(groups, 5)
    baseline = np.full(len(y), np.nan, dtype=np.float64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    routed = np.zeros(len(y), dtype=bool)
    raw_support_all = np.zeros(len(y), dtype=bool)
    supported_abstained_all = np.zeros(len(y), dtype=bool)
    nearest = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for fold_id in range(5):
        validation = np.flatnonzero(folds == fold_id)
        training = np.flatnonzero(folds != fold_id)
        parent, parent_meta, _ = route_module.nested_parent(
            target,
            y,
            groups,
            training,
            validation,
            target_dense,
            sparse_parts,
            fingerprints,
            global_indices,
            y_global,
            config,
            canonical,
            scaffolds,
            f"candidate_outer_fold_{fold_id}",
        )
        model, training_rows = route_module.fit_gap_model(gap_table, set(groups[validation]))
        raw_candidate, route, partner, _ = route_module.route_predictions(
            target,
            canonical[validation],
            parent,
            y[training],
            model,
            maps,
        )
        nearest_validation = route_module.nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
        egc_validation = np.asarray([maps["egc"].get(value, np.nan) for value in canonical[validation]], dtype=np.float64)
        raw_support = np.isfinite(partner) & np.isfinite(egc_validation)
        allowed = (nearest_validation < SIMILARITY_BARRIER) & ~np.isin(scaffolds[validation], list(BLOCKED_SCAFFOLDS))
        raw_candidate[~allowed] = parent[~allowed]
        route[~allowed] = False
        baseline[validation] = parent
        candidate[validation] = raw_candidate
        routed[validation] = route
        raw_support_all[validation] = raw_support
        supported_abstained_all[validation] = raw_support & ~route
        nearest[validation] = nearest_validation
        fold_rows.append({
            "fold": fold_id,
            "rows": int(len(validation)),
            "baseline_r2": float(r2_score(y[validation], parent)),
            "candidate_r2": float(r2_score(y[validation], raw_candidate)),
            "delta_r2": float(r2_score(y[validation], raw_candidate) - r2_score(y[validation], parent)),
            "routed_rows": int(np.sum(route)),
            "gap_training_rows": training_rows,
            "parent_blend": parent_meta,
        })
    oof = pd.DataFrame({
        "canonical": canonical,
        "target_type": target,
        "target": y,
        "baseline": baseline,
        "candidate": candidate,
        "route": routed,
        "raw_support": raw_support_all,
        "supported_but_abstained": supported_abstained_all,
        "nearest_similarity": nearest,
        "scaffold": scaffolds,
        "group": groups,
        "outer_fold": folds,
    })

    test_frame = test[test["target_type"] == target].reset_index(drop=True)
    test_canonical = test_frame["canonical"].to_numpy(object)
    test_global_indices = np.asarray([key_to_index[value] for value in test_canonical], dtype=np.int64)
    parent_test = parent_test_prediction(
        target,
        y,
        groups,
        target_dense,
        sparse_parts,
        fingerprints,
        global_indices,
        test_global_indices,
        y_global,
        config,
        route_module,
    )
    full_model, full_training_rows = route_module.fit_gap_model(gap_table, set())
    test_nearest = route_module.nearest_similarity(fingerprints, test_global_indices, global_indices)
    test_scaffolds = np.asarray([plumbing.scaffold(value) for value in test_canonical], dtype=object)
    raw_candidate, test_route, _, _ = route_module.route_predictions(
        target,
        test_canonical,
        parent_test,
        y,
        full_model,
        maps,
    )
    allowed_test = (test_nearest < SIMILARITY_BARRIER) & ~np.isin(test_scaffolds, list(BLOCKED_SCAFFOLDS))
    raw_candidate[~allowed_test] = parent_test[~allowed_test]
    test_route[~allowed_test] = False
    test_predictions = pd.DataFrame({
        "id": test_frame["id"].astype(int),
        "target": raw_candidate.astype(float),
        "target_type": target,
        "route": test_route,
        "nearest_similarity": test_nearest,
        "scaffold": test_scaffolds,
        "parent_prediction": parent_test,
        "gap_training_rows": full_training_rows,
    })
    report = {
        "target": target,
        "baseline_r2": float(r2_score(y, baseline)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": float(r2_score(y, candidate) - r2_score(y, baseline)),
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "group_bootstrap_lower": float(plumbing.bootstrap_r2_lower(y, baseline, candidate, groups)),
        "routed_oof_rows": int(np.sum(routed)),
        "folds": fold_rows,
        "test_rows": int(len(test_predictions)),
        "test_routed_rows": int(np.sum(test_route)),
        "paired_target_policy": "official covariate only; no cross-target predictions consumed",
    }
    return oof, test_predictions, report


def unchanged_panel_report(
    rows: pd.DataFrame,
    target: str,
    pooled: pd.DataFrame,
    keys: list[str],
    fingerprints: list[object],
) -> dict[str, object]:
    """Report direct, auditable transfer panels for a C001-preserved target."""
    y = rows["target"].to_numpy(float)
    baseline = rows["baseline_prediction"].to_numpy(float)
    candidate = rows["candidate_prediction"].to_numpy(float)
    canonical = rows["canonical"].astype(str).to_numpy(object)
    groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
    measurements_by_canonical = (
        pooled[pooled["target_type"] == target]
        .drop_duplicates("canonical")
        .set_index("canonical")["measurements"]
        .to_dict()
    )
    measurements = np.asarray([int(measurements_by_canonical.get(value, 0)) for value in canonical], dtype=np.int64)
    key_to_index = {key: index for index, key in enumerate(keys)}
    row_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    target_keys = set(canonical)
    other_indices = np.asarray([index for index, key in enumerate(keys) if key not in target_keys], dtype=np.int64)
    similarity = ei_route.nearest_similarity(fingerprints, row_indices, other_indices)

    panels: dict[str, dict[str, object]] = {}

    def add_panel(name: str, selected: np.ndarray, source: str) -> None:
        selected = np.asarray(selected, dtype=bool)
        count = int(np.sum(selected))
        detail: dict[str, object] = {"rows": count, "source": source, "delta_r2": 0.0}
        if count >= 5 and float(np.var(y[selected])) > 1.0e-15:
            detail.update({
                "eligible_for_r2": True,
                "baseline_r2": float(r2_score(y[selected], baseline[selected])),
                "candidate_r2": float(r2_score(y[selected], candidate[selected])),
                "delta_r2": float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected])),
            })
        else:
            detail["eligible_for_r2"] = False
        panels[name] = detail

    add_panel("all_rows", np.ones(len(rows), dtype=bool), "official_oof_rows")
    group_counts = pd.Series(groups).value_counts()
    add_panel("canonical_group_singletons", np.isin(groups, group_counts[group_counts == 1].index), "canonical_no_stereo_group")
    add_panel("canonical_group_duplicates", np.isin(groups, group_counts[group_counts > 1].index), "canonical_no_stereo_group")
    for scaffold_name in sorted(set(scaffolds)):
        add_panel(f"scaffold_{scaffold_name}", scaffolds == scaffold_name, "murcko_scaffold")
    similarity_bins = {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
        "similarity_unavailable": ~np.isfinite(similarity),
    }
    for name, selected in similarity_bins.items():
        add_panel(name, selected, "nearest_off_target_fingerprint_similarity")
    add_panel("official_single_measurement", measurements == 1, "official_current_archive_measurements")
    add_panel("official_multiple_measurements", measurements > 1, "official_current_archive_measurements")
    add_panel("official_measurement_available", measurements > 0, "official_current_archive_measurements")
    return {
        "target": target,
        "mode": "unchanged_c001_reference",
        "panels": panels,
        "minimum_panel_delta": float(min(float(value["delta_r2"]) for value in panels.values())),
        "availability": {
            "rows": int(len(rows)),
            "measurement_rows": int(np.sum(measurements > 0)),
            "single_measurement_rows": int(np.sum(measurements == 1)),
            "multiple_measurement_rows": int(np.sum(measurements > 1)),
        },
    }


def run_candidate(data_dir: str | Path, run_dir: Path, output: Path) -> dict[str, object]:
    started = time.time()
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    data_path = (root / data_dir).resolve() if not Path(data_dir).is_absolute() else Path(data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_path)
    inputs = {name: {**record, "path": str(Path(record["path"]).resolve().relative_to(root))} for name, record in inputs.items()}
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "mixed_candidate": True, "special_targets": list(SPECIAL_TARGETS)})
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(config["morgan_bits"]))

    parent_detail, parent_oof, parent_report = reference.fit_targets(
        pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
    )
    detail = parent_detail[["id", "target_type", "model_prediction"]].copy()
    oof_parts: list[pd.DataFrame] = []
    target_reports: dict[str, object] = {}
    special_test_reports: dict[str, object] = {}
    for target in TARGETS:
        if target in SPECIAL_TARGETS:
            special_oof, special_test, special_report = specialized_target(
                target, pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
            )
            special_test_reports[target] = special_test
            replacement = special_test.set_index("id")["target"]
            mask = detail["target_type"] == target
            detail.loc[mask, "model_prediction"] = detail.loc[mask, "id"].map(replacement).astype(float).to_numpy()
            special_oof = special_oof.rename(columns={"baseline": "baseline_prediction", "candidate": "candidate_prediction"})
            oof_parts.append(special_oof)
            target_reports[target] = special_report
        else:
            rows = parent_oof[parent_oof["target_type"] == target].copy()
            rows["baseline_prediction"] = rows["prediction"].astype(float)
            rows["candidate_prediction"] = rows["prediction"].astype(float)
            rows["baseline"] = rows["prediction"].astype(float)
            rows["candidate"] = rows["prediction"].astype(float)
            rows["group"] = [plumbing.no_stereo(value) for value in rows["canonical"]]
            rows["scaffold"] = [plumbing.scaffold(value) for value in rows["canonical"]]
            oof_parts.append(rows[["canonical", "target_type", "target", "baseline_prediction", "candidate_prediction", "baseline", "candidate", "group", "scaffold"]])
            target_reports[target] = {
                "target": target,
                "baseline_r2": float(r2_score(rows["target"], rows["baseline_prediction"])),
                "candidate_r2": float(r2_score(rows["target"], rows["candidate_prediction"])),
                "delta_r2": 0.0,
                "positive_folds": 0,
                "group_bootstrap_lower": 0.0,
                "unchanged_c001_reference": True,
            }
            target_reports[target].update(unchanged_panel_report(rows, target, pooled, keys, fingerprints))
    final_detail, override_report = reference.apply_official_overrides(detail, test, raw_labels)
    submission = final_detail[["id", "target"]].copy()
    if len(submission) != len(test) or not submission["id"].equals(test["id"]):
        raise RuntimeError("mixed candidate IDs/order differ from official test")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("mixed candidate contains duplicate IDs or non-finite predictions")
    submission.to_csv(output, index=False)

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    for target in TARGETS:
        rows = oof[oof["target_type"] == target]
        groups = np.asarray([plumbing.no_stereo(value) for value in rows["canonical"]], dtype=object)
        target_reports[target]["baseline_r2"] = float(r2_score(rows["target"], rows["baseline_prediction"]))
        target_reports[target]["candidate_r2"] = float(r2_score(rows["target"], rows["candidate_prediction"]))
        target_reports[target]["delta_r2"] = float(target_reports[target]["candidate_r2"] - target_reports[target]["baseline_r2"])
        target_reports[target]["group_bootstrap_lower"] = float(
            plumbing.bootstrap_r2_lower(rows["target"].to_numpy(float), rows["baseline_prediction"].to_numpy(float), rows["candidate_prediction"].to_numpy(float), groups)
        )
        if target in SPECIAL_TARGETS:
            panels: dict[str, float] = {}
            for name, selected in {
                "raw_support": rows["raw_support"].to_numpy(bool),
                "missing_support": ~rows["raw_support"].to_numpy(bool),
                "supported_but_abstained": rows["supported_but_abstained"].to_numpy(bool),
                "similarity_lt_0.30": rows["nearest_similarity"].to_numpy(float) < 0.30,
                "similarity_0.30_0.50": (rows["nearest_similarity"].to_numpy(float) >= 0.30) & (rows["nearest_similarity"].to_numpy(float) < 0.50),
                "similarity_0.50_0.70": (rows["nearest_similarity"].to_numpy(float) >= 0.50) & (rows["nearest_similarity"].to_numpy(float) < 0.70),
                "similarity_ge_0.70": rows["nearest_similarity"].to_numpy(float) >= 0.70,
            }.items():
                if int(np.sum(selected)) >= 5 and float(np.var(rows["target"].to_numpy(float)[selected])) > 1.0e-15:
                    panels[name] = float(
                        r2_score(rows["target"].to_numpy(float)[selected], rows["candidate_prediction"].to_numpy(float)[selected])
                        - r2_score(rows["target"].to_numpy(float)[selected], rows["baseline_prediction"].to_numpy(float)[selected])
                    )
            for scaffold_name in sorted(rows["scaffold"].unique()):
                selected = rows["scaffold"].to_numpy(object) == scaffold_name
                if int(np.sum(selected)) >= 10 and float(np.var(rows["target"].to_numpy(float)[selected])) > 1.0e-15:
                    panels[f"scaffold_{scaffold_name}"] = float(
                        r2_score(rows["target"].to_numpy(float)[selected], rows["candidate_prediction"].to_numpy(float)[selected])
                        - r2_score(rows["target"].to_numpy(float)[selected], rows["baseline_prediction"].to_numpy(float)[selected])
                    )
            target_reports[target]["panels"] = panels
            target_reports[target]["minimum_panel_delta"] = float(min(panels.values())) if panels else 0.0
        else:
            # The direct panel report was built from official rows above; retain
            # it here while refreshing the aggregate OOF metrics.
            target_reports[target]["minimum_panel_delta"] = float(target_reports[target]["minimum_panel_delta"])
    mean_baseline = float(np.mean([target_reports[target]["baseline_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    report = {
        "schema_version": "ppp.round2.mixed-seven-target-candidate-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": protocol["parent"],
        "official_inputs": inputs,
        "config": config,
        "config_sha256": reference.canonical_json_hash(config),
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": int(len(oof))},
        "target_counts": {target: int(np.sum(test["target_type"] == target)) for target in TARGETS},
        "targets": target_reports,
        "mean_baseline_r2": mean_baseline,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_baseline,
        "minimum_required_mean_gain": 0.002,
        "mean_gate_pass": bool(mean_candidate - mean_baseline >= 0.002),
        "maximum_target_loss": float(min(target_reports[target]["delta_r2"] for target in TARGETS)),
        "target_loss_gate_pass": bool(min(target_reports[target]["delta_r2"] for target in TARGETS) >= -0.003),
        "all_transfer_panels_pass": bool(all(target_reports[target]["minimum_panel_delta"] >= 0.0 for target in TARGETS)),
        "special_target_policy": "Ei uses official Eea/Egc; Eea uses official Ei/Egc; no predicted cross-target values are consumed.",
        "official_overrides": override_report,
        "parent_reference_report": parent_report,
        "submission": {"path": str(output.relative_to(root)), "rows": int(len(submission)), "sha256": sha256_file(output)},
        "source_hashes": {
            "candidate_script": sha256_file(Path(__file__).resolve()),
            "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py"),
            "ei_route_module": sha256_file(root / "tools" / "round2_ei_scaffold_abstaining_gap_identity_v4.py"),
            "eea_route_module": sha256_file(root / "tools" / "round2_eea_scaffold_abstaining_gap_identity_v7.py"),
            "metric_plumbing": sha256_file(root / "tools" / "round2_eea_cross_target_oof_residual_stack.py"),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", config)
    (run_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "test_route_diagnostics.csv").write_text(
        pd.concat([value.assign(target_type=target) for target, value in special_test_reports.items()], ignore_index=True).to_csv(index=False),
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([
        f"{report['source_hashes']['candidate_script']}  SOURCE tools/round2_mixed_candidate_v3.py",
        f"{report['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
        f"{report['source_hashes']['ei_route_module']}  SOURCE tools/round2_ei_scaffold_abstaining_gap_identity_v4.py",
        f"{report['source_hashes']['eea_route_module']}  SOURCE tools/round2_eea_scaffold_abstaining_gap_identity_v7.py",
        f"{report['source_hashes']['metric_plumbing']}  SOURCE tools/round2_eea_cross_target_oof_residual_stack.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("candidate run directory must begin with protocol.json only")
    report = run_candidate(args.data_dir, run_dir, output)
    print(json.dumps({"experiment_id": report["experiment_id"], "rows": report["submission"]["rows"], "mean_candidate_r2": report["mean_candidate_r2"], "mean_gain": report["mean_gain"]}, indent=2))


if __name__ == "__main__":
    main()
