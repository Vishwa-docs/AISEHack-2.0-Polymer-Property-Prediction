"""C277: repeat-view invariant structure residual against the C050 parent.

The representation concatenates and aggregates open, H-capped, and periodicized
repeat-unit Morgan views plus compact descriptors. It is structure-only and
rebuilds the C050 parent from official inputs; no stored predictions or external_label
files are read.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C277-repeat-view-invariance-v1"
DATA_DIR = ROOT / "ppp-round-2"
sys.path.insert(0, str(ROOT / "tools"))
import round2_c097_graph_grammar_hgb_full as parent_builder  # noqa: E402
import round2_c127_round1_carrier_factory as carrier  # noqa: E402
import round2_c258_ei_eht_orbital_residual as eht  # noqa: E402

RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(parent_builder.TARGETS)
BITS = 256
RESIDUAL_WEIGHT = 0.25
RIDGE_ALPHA = 100.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fp_bits(mol: Chem.Mol | None) -> np.ndarray:
    out = np.zeros(BITS, dtype=np.float32)
    if mol is None:
        return out
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=BITS, useChirality=False)
        DataStructs.ConvertToNumpyArray(fp, out)
    except Exception:
        pass
    return out


def desc(mol: Chem.Mol | None) -> np.ndarray:
    if mol is None:
        return np.full(10, np.nan, dtype=np.float32)
    try:
        return np.asarray([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcLabuteASA(mol),
        ], dtype=np.float32)
    except Exception:
        return np.full(10, np.nan, dtype=np.float32)


def view_features(keys: list[str], molecules: list[Chem.Mol]) -> tuple[np.ndarray, dict[str, int]]:
    features = np.full((len(keys), BITS * 4 + 30), np.nan, dtype=np.float32)
    hcap_count = 0
    ring_count = 0
    for i, (key, original) in enumerate(zip(keys, molecules)):
        capped = eht.remove_dummy_caps(key)
        periodic = eht.ring_close_dummy_caps(key)
        views = [fp_bits(original), fp_bits(capped), fp_bits(periodic)]
        supports = [original is not None, capped is not None, periodic is not None]
        if supports[1]:
            hcap_count += 1
        if supports[2]:
            ring_count += 1
        block = np.concatenate(views + [np.nanmean(np.vstack(views), axis=0)])
        dblock = np.concatenate([desc(original), desc(capped), desc(periodic)])
        features[i, : len(block)] = block
        features[i, len(block):] = dblock
    return features, {"hcap_supported": hcap_count, "periodic_supported": ring_count}


def main() -> None:
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    progress = RUN / "progress.jsonl"
    progress.write_text(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n", encoding="utf-8")
    root = ROOT.parent
    parent = parent_builder.build_parent(root, DATA_DIR)
    keys = parent["keys"]
    molecules = parent["molecules"]
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "parent_ready", "feature_rows": len(keys)}) + "\n")
    X, support = view_features(keys, molecules)
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "features_ready", "feature_shape": list(X.shape), **support}) + "\n")

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    test_detail = parent["test_parent_detail"]
    for target in TARGETS:
        info = parent["target_info"][target]
        indices = np.asarray(info["indices"], dtype=int)
        y = np.asarray(info["y"], dtype=float)
        base = np.asarray(info["parent"], dtype=float)
        groups = np.asarray(info["groups"], dtype=object)
        folds = carrier.grouped_folds(groups)
        residual_oof = np.full(len(y), np.nan, dtype=float)
        fold_rows = []
        for fold in range(carrier.N_FOLDS):
            valid = np.flatnonzero(folds == fold)
            train_rows = np.flatnonzero(folds != fold)
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000))
            model.fit(X[indices[train_rows]], y[train_rows] - base[train_rows])
            residual_oof[valid] = model.predict(X[indices[valid]])
            pred = base[valid] + RESIDUAL_WEIGHT * residual_oof[valid]
            fold_rows.append({"fold": fold, "rows": int(len(valid)), "parent_r2": float(r2_score(y[valid], base[valid])), "candidate_r2": float(r2_score(y[valid], pred)), "delta_r2": float(r2_score(y[valid], pred) - r2_score(y[valid], base[valid]))})
        candidate = base + RESIDUAL_WEIGHT * residual_oof
        full_model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000))
        full_model.fit(X[indices], y - base)
        target_test = test_detail.loc[test_detail["target_type"].astype(str).eq(target)].sort_values("id")
        test_indices = np.asarray([parent["key_to_index"][value] for value in target_test["canonical"]], dtype=int)
        test_candidate = full_model.predict(X[test_indices])
        parent_r2 = float(r2_score(y, base))
        candidate_r2 = float(r2_score(y, candidate))
        target_reports[target] = {"parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "rows": int(len(y))}
        oof_parts.append(pd.DataFrame({"target_type": target, "target": y, "parent": base, "candidate": candidate, "fold": folds}))
        test_parts.append(pd.DataFrame({"id": target_test["id"].astype(int), "target_type": target, "candidate": test_candidate}))
        print(json.dumps({"target": target, **target_reports[target]}), flush=True)

    oof = pd.concat(oof_parts, ignore_index=True)
    test_candidates = pd.concat(test_parts, ignore_index=True)
    submission = test_detail[["id", "target_type", "target"]].merge(test_candidates, on=["id", "target_type"], how="left", validate="one_to_one")
    submission["target"] = submission["candidate"]
    submission = submission[["id", "target"]].sort_values("id").reset_index(drop=True)
    assert len(submission) == 4940 and np.array_equal(submission["id"].to_numpy(), np.arange(1, 4941)) and np.isfinite(submission["target"].to_numpy(float)).all()
    oof.to_csv(RUN / "oof_predictions.csv", index=False)
    submission.to_csv(RUN / "predictions.csv", index=False)
    mean_parent = float(np.mean([report["parent_r2"] for report in target_reports.values()]))
    mean_candidate = float(np.mean([report["candidate_r2"] for report in target_reports.values()]))
    report = {"schema_version": "ppp.round2.c277.repeat-view-invariance.v1", "experiment_id": RUN.name, "status": "completed_research_candidate", "official_only": True, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "pretrained_weights": False, "external_targets": False, "view_definition": ["open_repeat", "h_capped", "periodic_ring_closed", "view_mean"], "feature_shape": list(X.shape), "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": RIDGE_ALPHA, "targets": target_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "complete_output_rows": int(len(submission)), "complete_output_order_pass": True, "elapsed_seconds": time.time() - started, "source_hashes": {"runner": sha256(Path(__file__)), "parent_builder": sha256(ROOT / "tools/round2_c097_graph_grammar_hgb_full.py"), "reference": sha256(ROOT / "tools/initial_reference_pipeline.py")}}
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (RUN / "protocol.json").write_text(json.dumps({"schema_version": "ppp.round2.c277.repeat-view-invariance.v1", "experiment_id": RUN.name, "hypothesis": "Open, capped, and periodicized views provide complementary repeat-unit signal when all equivalent views remain within the same grouped fold.", "changed_factor": "structure-only repeat-view feature concatenation and aggregation", "baseline": "rebuilt C050 parent", "promotion_gate": "candidate must beat incumbent on clean grouped folds and full-test transfer; standalone component requires >=0.010 target gain, 4/5 positive folds, positive bootstrap, support panels, and parent parity", "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False}, indent=2) + "\n", encoding="utf-8")
    (RUN / "decision.md").write_text(f"# {RUN.name}\n\nMean parent: `{mean_parent:.12f}`. Mean candidate: `{mean_candidate:.12f}`. Mean gain: `{mean_candidate - mean_parent:+.12f}`. This candidate is not promoted until post-freeze transfer scoring and the full target gates pass.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{sha256(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": RUN.name, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
