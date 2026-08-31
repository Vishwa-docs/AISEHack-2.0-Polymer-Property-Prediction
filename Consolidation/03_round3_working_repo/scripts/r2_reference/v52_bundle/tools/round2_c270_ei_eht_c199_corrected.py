"""C270: corrected, memory-safe EHT comparison against selected C199."""

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
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C270-20260805-ei-eht-c199-corrected-v1"
sys.path.insert(0, str(TOOLS))
import round2_c097_graph_grammar_hgb_full as parent_builder  # noqa: E402
import round2_c127_round1_carrier_factory as carrier  # noqa: E402
import round2_c180_flory_fox_oligomer_carriers as c180  # noqa: E402
import round2_c196_ei_ffox_shrinkage_confirmation as c196  # noqa: E402
import round2_c199_ei_c196_transfer_guard as c199  # noqa: E402


def load_eht_module():
    spec = importlib.util.spec_from_file_location("c270_eht_source", TOOLS / "round2_c258_ei_eht_orbital_residual.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load EHT source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(smiles: str, variant: str) -> int:
    raw = hashlib.sha256(f"C270|{variant}|{smiles}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) % 2_000_000_000 + 1


def stable_eht_features(eht, smiles: str) -> tuple[np.ndarray, dict[str, bool]]:
    hcap, hcap_ok = eht.eht_variant_features(eht.remove_dummy_caps(smiles), stable_seed(smiles, "hcap"))
    ring, ring_ok = eht.eht_variant_features(eht.ring_close_dummy_caps(smiles), stable_seed(smiles, "ring"))
    diffs = [hcap[i] - ring[i] if hcap_ok and ring_ok else np.nan for i in (0, 1, 2)]
    row = np.asarray(hcap + [float(hcap_ok)] + ring + [float(ring_ok)] + diffs, dtype=float)
    return row, {"hcap_supported": bool(hcap_ok), "ring_supported": bool(ring_ok)}


def checkpoint(path: Path, stage: str, **payload) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, **payload}, sort_keys=True) + "\n")


def main() -> None:
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    progress = RUN / "progress.jsonl"
    progress.write_text(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n", encoding="utf-8")
    root = ROOT.parent
    parent = parent_builder.build_parent(root, ROOT / "ppp-round-2")
    info = dict(parent["target_info"]["ei"])
    info["fingerprints"] = parent["fingerprints"]
    y = np.asarray(info["y"], dtype=float)
    c050 = np.asarray(info["parent"], dtype=float)
    train_global = np.asarray(info["indices"], dtype=np.int64)
    test_rows, test_global, test_parent = c196.target_test_rows(parent, "ei")
    feature_global = np.unique(np.concatenate([train_global, test_global]))
    feature_keys = [parent["keys"][int(index)] for index in feature_global]
    local_index = {int(index): i for i, index in enumerate(feature_global)}
    checkpoint(progress, "parent_ready", train_rows=int(len(train_global)), test_rows=int(len(test_global)), feature_rows=int(len(feature_keys)))

    dense, sparse_features, feature_report = c180.build_features(root, feature_keys)
    local_info = dict(info)
    local_info["indices"] = np.asarray([local_index[int(index)] for index in train_global], dtype=np.int64)
    local_info["fingerprints"] = [parent["fingerprints"][int(index)] for index in feature_global]
    local_test = np.asarray([local_index[int(index)] for index in test_global], dtype=np.int64)
    raw = carrier.fit_target(local_info, dense, sparse_features, local_test, test_parent)
    c199_oof = c050 + 0.75 * (np.asarray(raw["candidate"], dtype=float) - c050)
    nearest = c199.fold_local_nearest(parent, info)
    c199_guard = c199.transfer_guard_mask(np.asarray(info["scaffolds"], dtype=object), nearest)
    selected_oof = c199_oof.copy()
    selected_oof[c199_guard] = c050[c199_guard]
    selected_r2 = float(r2_score(y, selected_oof))
    replay_error = abs(selected_r2 - 0.8566558157138717)
    if replay_error > 1e-8:
        raise RuntimeError(f"selected C199 replay failed: {selected_r2}")
    test_nearest = c199.full_train_nearest(parent, info, test_global)
    test_scaffold = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"].astype(str)], dtype=object)
    test_guard = c199.transfer_guard_mask(test_scaffold, test_nearest)
    selected_test = test_parent + 0.75 * (np.asarray(raw["test_direct"], dtype=float) - test_parent)
    selected_test[test_guard] = test_parent[test_guard]
    checkpoint(progress, "c199_replayed", selected_r2=selected_r2, replay_error=replay_error, guarded_oof_rows=int(np.sum(c199_guard)), guarded_test_rows=int(np.sum(test_guard)))

    eht = load_eht_module()
    features = []
    support = []
    for i, smiles in enumerate(feature_keys):
        row, report = stable_eht_features(eht, str(smiles))
        features.append(row)
        support.append(report)
        if (i + 1) % 25 == 0:
            checkpoint(progress, "eht_features", processed=i + 1)
    all_eht = np.asarray(features, dtype=float)
    X_train = all_eht[[local_index[int(index)] for index in train_global]]
    X_test = all_eht[local_test]
    residual = np.full(len(y), np.nan)
    folds = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
    fold_rows = []
    for fold in range(carrier.N_FOLDS):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=60.0, solver="lsqr", max_iter=5000, tol=1e-4))
        model.fit(X_train[train], y[train] - selected_oof[train])
        residual[valid] = model.predict(X_train[valid])
        pred = selected_oof[valid] + 0.35 * residual[valid]
        fold_rows.append({"fold": fold, "rows": int(len(valid)), "reference_r2": float(r2_score(y[valid], selected_oof[valid])), "candidate_r2": float(r2_score(y[valid], pred)), "delta_r2": float(r2_score(y[valid], pred) - r2_score(y[valid], selected_oof[valid]))})
    candidate_oof = selected_oof + 0.35 * residual
    # Fill a full-index matrix only for the structures used by this target so
    # the transfer helper sees correct support without a full feature build.
    full_eht = np.full((len(parent["keys"]), all_eht.shape[1]), np.nan)
    full_eht[feature_global] = all_eht
    transfer = eht.transfer_report(parent, info, selected_oof, candidate_oof, full_eht)
    candidate_r2 = float(r2_score(y, candidate_oof))
    delta = candidate_r2 - selected_r2
    passed = bool(replay_error <= 1e-8 and delta >= 0.010 and transfer["positive_folds"] >= 4 and transfer["group_bootstrap_lower"] > 0.0 and transfer["minimum_panel_delta"] >= 0.0)
    full_model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=60.0, solver="lsqr", max_iter=5000, tol=1e-4))
    full_model.fit(X_train, y - selected_oof)
    candidate_test = selected_test + 0.35 * full_model.predict(X_test)
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment_id": RUN.name,
        "status": "bankable_ei_candidate" if passed else "rejected_full_gate",
        "official_only": True,
        "target": "ei",
        "selected_c199_r2": selected_r2,
        "expected_c199_r2": 0.8566558157138717,
        "c199_replay_error": replay_error,
        "candidate_r2": candidate_r2,
        "delta_r2": delta,
        "positive_folds": transfer["positive_folds"],
        "group_bootstrap_lower": transfer["group_bootstrap_lower"],
        "minimum_panel_delta": transfer["minimum_panel_delta"],
        "folds": fold_rows,
        "transfer_panels": transfer["panels"],
        "feature_report": feature_report,
        "stable_seed": True,
        "hcap_supported": int(sum(x["hcap_supported"] for x in support)),
        "ring_supported": int(sum(x["ring_supported"] for x in support)),
        "test_guarded_rows": int(np.sum(test_guard)),
        "promotion_eligible": passed,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "elapsed_seconds": time.time() - started,
        "source_hashes": {"runner": digest(Path(__file__)), "c180": digest(TOOLS / "round2_c180_flory_fox_oligomer_carriers.py"), "c199": digest(TOOLS / "round2_c199_ei_c196_transfer_guard.py")},
    }
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"target": y, "reference": selected_oof, "candidate": candidate_oof, "fold": folds}).to_csv(RUN / "oof_predictions.csv", index=False)
    pd.DataFrame({"id": test_rows["id"].astype(int), "selected_c199": selected_test, "candidate": candidate_test, "guarded_c199": test_guard}).to_csv(RUN / "component_predictions.csv", index=False)
    (RUN / "decision.md").write_text(f"# C270 decision\n\nStatus: **{report['status']}**. Selected C199 R2 `{selected_r2:.9f}`; EHT candidate `{candidate_r2:.9f}`; delta `{delta:+.9f}`; bootstrap lower `{transfer['group_bootstrap_lower']:+.9f}`; minimum panel `{transfer['minimum_panel_delta']:+.9f}`.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    checkpoint(progress, "finished", status=report["status"], candidate_r2=candidate_r2, delta_r2=delta)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
