"""F03: archive-free Polymer Genome-style feature experiment.

The feature vocabulary is fit from current train structures only. Test SMILES
are used only as unlabeled covariates. No prior predictions or cached features
are inputs to this runner.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fable_common as fc
import F01_ei_eea_egb_chain_engine as f01

sys.path.insert(0, os.path.join(fc.ROUND2_DIR, "tools", "claude_r2_01"))
import build_pgfp

SEED = fc.SEED
ALPHAS = (10.0, 30.0, 100.0, 300.0)


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qspr_block(cans: list[str]) -> np.ndarray:
    funcs = list(Descriptors.descList)
    rows = []
    for can in cans:
        mol = Chem.MolFromSmiles(can)
        vals = []
        for _, fn in funcs:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(np.nan)
        nh = max(mol.GetNumHeavyAtoms(), 1)
        vals.extend([
            Descriptors.TPSA(mol),
            Descriptors.TPSA(mol) / nh,
            rdMolDescriptors.CalcNumAromaticRings(mol) / nh,
            Descriptors.NumRotatableBonds(mol) / nh,
            Descriptors.FractionCSP3(mol),
        ])
        rows.append(vals)
    x = np.asarray(rows, dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    return x


def hierarchical_features(train_cans: list[str], all_cans: list[str]) -> tuple[np.ndarray, dict]:
    train_set = set(train_cans)
    records = {c: build_pgfp.keys_for(c) for c in all_cans}
    doc = Counter()
    for c in train_set:
        overall, backbone, _ = records[c]
        doc.update(set(overall) | set(backbone))
    vocab = sorted(k for k, n in doc.items() if n >= 5)
    vi = {k: i for i, k in enumerate(vocab)}
    counts = np.zeros((len(all_cans), len(vocab)), dtype=np.float32)
    morphology = []
    for row, c in enumerate(all_cans):
        overall, backbone, _ = records[c]
        for token, value in overall.items():
            if token in vi:
                counts[row, vi[token]] += value
        for token, value in backbone.items():
            if token in vi:
                counts[row, vi[token]] += value
        morphology.append(build_pgfp.morphological(c))
    norm = counts / np.maximum(1.0, counts.sum(axis=1, keepdims=True))
    qspr = qspr_block(all_cans)
    x = np.hstack([np.log1p(counts), norm, np.asarray(morphology), qspr])
    return x, {"vocabulary_size": len(vocab), "feature_shape": list(x.shape), "qspr_features": qspr.shape[1]}


def fit_predict(xtr, ytr, xva, alpha):
    # A descriptor that is entirely undefined in the training fold must not
    # acquire an arbitrary validation-only coefficient. This selection is fit
    # locally and is part of the preprocessing state for that fold.
    support = np.isfinite(xtr).sum(axis=0)
    spread = np.nanstd(xtr, axis=0)
    keep = (support > 0) & (spread > 1.0e-12)
    xtr = np.clip(xtr[:, keep], -1.0e6, 1.0e6)
    xva = np.clip(xva[:, keep], -1.0e6, 1.0e6)
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1.0e-4),
    )
    model.fit(xtr, ytr)
    return model.predict(xva)


def main() -> None:
    started = time.time()
    data = fc.load_data()
    run_id = time.strftime("R2-F03-%Y%m%d-%H%M-polymer-genome")
    output_root = os.environ.get("FABLE_OUTPUT_ROOT", os.path.join(fc.ROUND2_DIR, "experiments", "CLEAN_OFFICIAL_ONLY"))
    mode = "with_archive" if os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1" else "without_archive"
    out_dir = os.path.join(output_root, f"{run_id}-{mode}")
    os.makedirs(out_dir, exist_ok=False)
    train_cans = data.train["can"].tolist()
    test_cans = data.test["can"].tolist()
    all_cans = list(dict.fromkeys(train_cans + test_cans))
    x, feature_report = hierarchical_features(list(dict.fromkeys(train_cans)), all_cans)
    index = {c: i for i, c in enumerate(all_cans)}
    reports = []
    for target in fc.TARGETS:
        sub = data.train[data.train.target_type == target].reset_index(drop=True)
        cans = sub["can"].tolist()
        y = sub["target"].to_numpy(float)
        folds = fc.grouped_folds(cans, 5, SEED)
        indices = np.asarray([index[c] for c in cans])
        oof = np.full(len(y), np.nan)
        alpha_rows = []
        for fold in range(5):
            tr, va = folds != fold, folds == fold
            inner = []
            inner_folds = fc.grouped_folds(np.asarray(cans)[tr], 3, SEED + fold + 1)
            for alpha in ALPHAS:
                pred = np.full(tr.sum(), np.nan)
                tri = np.flatnonzero(tr)
                for inner_fold in range(3):
                    it, iv = inner_folds != inner_fold, inner_folds == inner_fold
                    pred[iv] = fit_predict(x[indices[tri[it]]], y[tri[it]], x[indices[tri[iv]]], alpha)
                inner.append((float(np.mean((y[tri] - pred) ** 2)), alpha))
            alpha = min(inner)[1]
            oof[va] = fit_predict(x[indices[tr]], y[tr], x[indices[va]], alpha)
            alpha_rows.append({"fold": fold, "alpha": alpha})
        report = fc.evaluate_target("F03", target, y, oof, None, cans, folds, data,
                                    extra={"alpha_by_fold": alpha_rows})
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    result = {"experiment_id": run_id, "official_only": True, "archive_read": False,
              "local_eval_read": False, "pretrained_weights": False,
              "feature_report": feature_report, "reports": reports,
              "runtime_s": time.time() - started,
              "input_hashes": {name: digest(os.path.join(fc.DATA_DIR, name))
                               for name in ("train.csv", "test.csv", "PI1M.csv")}}
    with open(os.path.join(out_dir, "report.json"), "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    print(json.dumps({"run_id": run_id, "out_dir": out_dir,
                      "mean_shift_matched": float(np.mean([r["shift_matched_r2"] for r in reports]))}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
