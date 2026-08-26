"""C269: memory-safe regenerated C199 versus EHT Ei comparison."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C269-20260805-ei-eht-c199-comparison-v1"
sys.path.insert(0, str(TOOLS))
import round2_c097_graph_grammar_hgb_full as parent_builder  # noqa: E402
import round2_c127_round1_carrier_factory as carrier  # noqa: E402
import round2_c180_flory_fox_oligomer_carriers as c180  # noqa: E402
import round2_c196_ei_ffox_shrinkage_confirmation as c196  # noqa: E402
import round2_c199_ei_c196_transfer_guard as c199  # noqa: E402


def load_eht_module():
    spec = importlib.util.spec_from_file_location("c269_eht_features", TOOLS / "round2_c258_ei_eht_orbital_residual.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load EHT module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path: Path, stage: str, **payload) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, **payload}, sort_keys=True) + "\n")


def main() -> None:
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    progress = RUN / "progress.jsonl"
    progress.write_text(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n", encoding="utf-8")
    root = ROOT.parent.parent
    parent = parent_builder.build_parent(root, ROOT / "ppp-round-2")
    info = dict(parent["target_info"]["ei"])
    info["fingerprints"] = parent["fingerprints"]
    c050 = np.asarray(info["parent"], dtype=float)
    train_indices_global = np.asarray(info["indices"], dtype=np.int64)
    test_rows, test_indices_global, test_parent = c196.target_test_rows(parent, "ei")
    feature_indices_global = np.unique(np.concatenate([train_indices_global, test_indices_global]))
    feature_keys = [parent["keys"][int(index)] for index in feature_indices_global]
    checkpoint(progress, "parent_ready", train_rows=int(len(train_indices_global)), test_rows=int(len(test_indices_global)), feature_rows=int(len(feature_keys)))

    dense, sparse_features, feature_report = c180.build_features(root, feature_keys)
    local_index = {int(global_index): local for local, global_index in enumerate(feature_indices_global)}
    local_info = dict(info)
    local_info["indices"] = np.asarray([local_index[int(index)] for index in train_indices_global], dtype=np.int64)
    local_info["fingerprints"] = [parent["fingerprints"][int(index)] for index in feature_indices_global]
    local_test_indices = np.asarray([local_index[int(index)] for index in test_indices_global], dtype=np.int64)
    checkpoint(progress, "c199_feature_subset_ready", dense_shape=list(dense.shape), sparse_shape=list(sparse_features.shape), sparse_nnz=int(sparse_features.nnz))

    raw_result = carrier.fit_target(local_info, dense, sparse_features, local_test_indices, test_parent)
    raw_candidate = np.asarray(raw_result["candidate"], dtype=float)
    c199_shrunk = c050 + 0.75 * (raw_candidate - c050)
    nearest = c199.fold_local_nearest(parent, info)
    guarded = c199_shrunk.copy()
    guard = c199.transfer_guard_mask(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded[guard] = c050[guard]
    selected_r2 = float(r2_score(info["y"], guarded))
    replay_error = abs(selected_r2 - 0.8566558157138717)
    checkpoint(progress, "c199_replayed", selected_r2=selected_r2, replay_error=replay_error, guarded_rows=int(np.sum(guard)))
    if replay_error > 1e-8:
        raise RuntimeError(f"C199 subset replay failed: {selected_r2} vs 0.8566558157138717")

    eht = load_eht_module()
    eht_rows = []
    test_eht_rows = []
    supports = []
    test_supports = []
    for row_number, index in enumerate(feature_indices_global):
        row, support = eht.eht_features_for_smiles(str(parent["keys"][int(index)]), row_number)
        eht_rows.append(row)
        supports.append(support)
    X_all = np.asarray(eht_rows, dtype=float)
    X_train = X_all[[local_index[int(index)] for index in train_indices_global]]
    X_test = X_all[local_test_indices]
    checkpoint(progress, "eht_features_ready", feature_count=int(X_all.shape[1]), hcap_supported=int(sum(x["hcap_supported"] for x in supports)), ring_supported=int(sum(x["ring_supported"] for x in supports)))

    y = np.asarray(info["y"], dtype=float)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    residual = np.full(len(y), np.nan)
    fold_rows = []
    for fold in range(carrier.N_FOLDS):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=60.0, solver="lsqr", max_iter=5000, tol=1e-4))
        model.fit(X_train[train], y[train] - guarded[train])
        residual[valid] = model.predict(X_train[valid])
        pred = guarded[valid] + 0.35 * residual[valid]
        fold_rows.append({"fold": fold, "rows": int(len(valid)), "reference_r2": float(r2_score(y[valid], guarded[valid])), "candidate_r2": float(r2_score(y[valid], pred)), "delta_r2": float(r2_score(y[valid], pred) - r2_score(y[valid], guarded[valid]))})
    candidate = guarded + 0.35 * residual
    candidate_r2 = float(r2_score(y, candidate))
    delta = candidate_r2 - selected_r2
    passed = bool(delta >= 0.010 and sum(row["delta_r2"] > 0 for row in fold_rows) >= 4)
    full_model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=60.0, solver="lsqr", max_iter=5000, tol=1e-4))
    full_model.fit(X_train, y - guarded)
    candidate_test = np.asarray(c199_shrunk if False else test_parent, dtype=float) + 0.35 * full_model.predict(X_test)
    test_nearest = c199.full_train_nearest(parent, info, test_indices_global)
    test_scaffold = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"].astype(str)], dtype=object)
    test_guard = c199.transfer_guard_mask(test_scaffold, test_nearest)
    candidate_test[test_guard] = test_parent[test_guard]
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment_id": RUN.name,
        "status": "bankable_ei_candidate" if passed else "rejected_against_c199",
        "official_only": True,
        "target": "ei",
        "c199_reference_r2": selected_r2,
        "expected_c199_r2": 0.8566558157138717,
        "c199_replay_error": replay_error,
        "candidate_r2": candidate_r2,
        "delta_r2": delta,
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "bootstrap_lower_not_computed": True,
        "transfer_panels_not_computed": True,
        "full_gate_pass": passed,
        "promotion_eligible": False,
        "feature_report": feature_report,
        "feature_count": int(X_all.shape[1]),
        "test_rows": int(len(test_rows)),
        "test_guarded_rows": int(np.sum(test_guard)),
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "elapsed_seconds": time.time() - started,
        "source_hashes": {"runner": digest(Path(__file__)), "c180": digest(TOOLS / "round2_c180_flory_fox_oligomer_carriers.py"), "c199": digest(TOOLS / "round2_c199_ei_c196_transfer_guard.py")},
    }
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"target": y, "reference": guarded, "candidate": candidate, "fold": folds}).to_csv(RUN / "oof_predictions.csv", index=False)
    pd.DataFrame({"id": test_rows["id"].astype(int), "reference": test_parent, "candidate": candidate_test, "guarded": test_guard}).to_csv(RUN / "component_predictions.csv", index=False)
    (RUN / "decision.md").write_text(f"# C269 decision\n\nStatus: **{report['status']}**. Regenerated C199 R2 `{selected_r2:.9f}`; EHT candidate `{candidate_r2:.9f}`; delta `{delta:+.9f}`; positive folds `{report['positive_folds']}/5`. Bootstrap and transfer panels remain required before promotion.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    checkpoint(progress, "finished", status=report["status"], candidate_r2=candidate_r2, delta_r2=delta)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
