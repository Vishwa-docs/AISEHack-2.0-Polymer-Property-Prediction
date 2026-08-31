"""C268: bounded EHT feasibility test over the C050 Ei parent.

The original C258 route terminated while regenerating the heavier selected C199
reference.  C268 isolates the new EHT feature family from that failure mode:
it builds the official C050 parent, computes EHT features only for the 222 Ei
structures, and evaluates one fixed fold-local residual.  It cannot promote a
component by itself because the current selected Ei incumbent is C199; a
positive result only authorizes a later memory-safe C199 comparison.
"""

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
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C268-20260805-ei-eht-c050-feasibility-v1"
sys.path.insert(0, str(TOOLS))
import round2_c097_graph_grammar_hgb_full as parent_builder  # noqa: E402
import round2_c127_round1_carrier_factory as carrier  # noqa: E402


def load_eht_module():
    path = TOOLS / "round2_c258_ei_eht_orbital_residual.py"
    spec = importlib.util.spec_from_file_location("c258_eht_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load EHT feature module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    progress = RUN / "progress.jsonl"
    with progress.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n")
    root = ROOT.parent.parent
    data_dir = ROOT / "ppp-round-2"
    parent = parent_builder.build_parent(root, data_dir)
    info = dict(parent["target_info"]["ei"])
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    eht = load_eht_module()
    features = []
    support = []
    for row_number, parent_index in enumerate(indices):
        row, report = eht.eht_features_for_smiles(str(parent["keys"][int(parent_index)]), row_number)
        features.append(row)
        support.append(report)
        if (row_number + 1) % 25 == 0:
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": "eht_features", "processed": row_number + 1}) + "\n")
    X = np.asarray(features, dtype=np.float64)
    oof_residual = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    for fold in sorted(set(int(value) for value in folds)):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=60.0, solver="lsqr", max_iter=5000, tol=1e-4),
        )
        model.fit(X[train], y[train] - parent_oof[train])
        oof_residual[valid] = model.predict(X[valid])
        candidate = parent_oof[valid] + 0.35 * oof_residual[valid]
        fold_rows.append({"fold": fold, "rows": int(len(valid)), "parent_r2": float(r2_score(y[valid], parent_oof[valid])), "candidate_r2": float(r2_score(y[valid], candidate)), "delta_r2": float(r2_score(y[valid], candidate) - r2_score(y[valid], parent_oof[valid]))})
    candidate_oof = parent_oof + 0.35 * oof_residual
    parent_r2 = float(r2_score(y, parent_oof))
    candidate_r2 = float(r2_score(y, candidate_oof))
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment_id": RUN.name,
        "status": "feasibility_pass" if candidate_r2 - parent_r2 >= 0.005 and sum(row["delta_r2"] > 0 for row in fold_rows) >= 4 else "rejected_feasibility",
        "official_only": True,
        "target": "ei",
        "parent": "regenerated C050 official-only Ei component",
        "parent_r2": parent_r2,
        "candidate_r2": candidate_r2,
        "delta_r2": candidate_r2 - parent_r2,
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "rows": int(len(y)),
        "eht_feature_count": int(X.shape[1]),
        "hcap_supported": int(sum(item["hcap_supported"] for item in support)),
        "ring_supported": int(sum(item["ring_supported"] for item in support)),
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "promotion_eligible": False,
        "note": "A positive result requires a separate memory-safe comparison against regenerated C199 before it can affect the clean compound.",
        "elapsed_seconds": time.time() - started,
        "source_hashes": {"runner": digest(Path(__file__)), "eht_source": digest(TOOLS / "round2_c258_ei_eht_orbital_residual.py")},
    }
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"target": y, "parent": parent_oof, "candidate": candidate_oof, "fold": folds}).to_csv(RUN / "oof_predictions.csv", index=False)
    (RUN / "decision.md").write_text(f"# C268 decision\n\nStatus: **{report['status']}**. C050 Ei R2 `{parent_r2:.9f}`; EHT candidate `{candidate_r2:.9f}`; delta `{candidate_r2 - parent_r2:+.9f}`. This feasibility run is not promotable without C199 parent comparison.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "finished", "status": report["status"], "delta_r2": report["delta_r2"]}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
