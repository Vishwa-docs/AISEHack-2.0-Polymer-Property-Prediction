"""C167: clean Ei specialist using observed non-Ei partner covariates.

This is a target-specific screen against the corrected C143-style parent.  Ei
is never included as a feature.  The other official properties are used only
when their value is observed for the same structure, with a missingness mask;
this is a legitimate test-time covariate and is fit fold-locally.  Rows with
both Eea and Egc observed retain the existing physical parent, while the
specialist is tested only on the harder missing-partner rows.

The run is deliberately a clean screen: it stops after grouped OOF metrics and
does not read test_external_labels.csv, fit a full-data submission, or create an local_eval
artifact.
"""
from pathlib import Path
import importlib.util
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
EI = TARGETS.index("ei")
started = time.time()


def r2(y, p):
    return float(r2_score(np.asarray(y, float), np.asarray(p, float)))


def load_features():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    P = pickle.loads((SCR / "physics.pkl").read_bytes())
    G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    spec = importlib.util.spec_from_file_location("c162_feature_builder", ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return F, mod.features(F, P, G).astype(np.float64)


def labels_and_features(F):
    cmap, idx = F["canon_map"], F["idx"]
    ns = len(F["canon_list"])
    labels = np.full((ns, len(TARGETS)), np.nan, dtype=np.float64)
    train = pd.read_csv(BASE / "train.csv")
    archive = pd.read_csv(BASE / "archive/train.csv")
    for frame in (train, archive):
        frame["canon"] = frame["smiles"].map(cmap)
    # Round-2 labels take precedence over archive labels, matching the
    # corrected parent and the official bundle's current-row policy.
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            values = frame.loc[frame["target_type"].eq(target)].groupby("canon")["target"].mean()
            for canon, value in values.items():
                labels[idx[canon], j] = float(value)
    return labels, train, archive


def make_partner_matrix(labels, rows, train_rows):
    """Structure-independent partner block: values plus availability masks.

    The Ei column is excluded completely.  Imputation is performed inside each
    fold on the training rows only.
    """
    other = [j for j in range(len(TARGETS)) if j != EI]
    values = labels[:, other]
    masks = np.isfinite(values).astype(np.float64)
    # Include a small number of fixed physically motivated products only when
    # both operands are observed; otherwise use zero and the mask carries the
    # information.  These do not contain Ei.
    tg, egc, egb, eea, nc, eps = [labels[:, j] for j in (0, 1, 2, 4, 5, 6)]
    derived = np.column_stack([
        np.nan_to_num(eea + egc, nan=0.0),
        np.nan_to_num(egb - egc, nan=0.0),
        np.nan_to_num(eps - nc * nc, nan=0.0),
        np.nan_to_num(tg, nan=0.0),
    ])
    derived_masks = np.column_stack([
        np.isfinite(eea) & np.isfinite(egc),
        np.isfinite(egb) & np.isfinite(egc),
        np.isfinite(eps) & np.isfinite(nc),
        np.isfinite(tg),
    ]).astype(np.float64)
    block = np.hstack([values, masks, derived, derived_masks])
    return block[np.asarray(rows)]


def fit_model(kind, x, y, seed):
    if kind == "ridge_abs":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0)).fit(x, y)
    if kind == "ridge_res":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0)).fit(x, y)
    if kind == "et_abs":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), ExtraTreesRegressor(n_estimators=500, max_features=0.45, min_samples_leaf=2, n_jobs=10, random_state=seed)).fit(x, y)
    if kind == "et_res":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), ExtraTreesRegressor(n_estimators=500, max_features=0.45, min_samples_leaf=2, n_jobs=10, random_state=seed)).fit(x, y)
    if kind == "hgb_abs":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), HistGradientBoostingRegressor(max_iter=350, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=2.0, random_state=seed)).fit(x, y)
    if kind == "hgb_res":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), HistGradientBoostingRegressor(max_iter=350, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=2.0, random_state=seed)).fit(x, y)
    raise ValueError(kind)


def main():
    F, X = load_features()
    labels, train, archive = labels_and_features(F)
    ns = len(labels)
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").astype(np.float64).copy()
    observed = np.isfinite(labels)
    ei_rows = np.where(observed[:, EI])[0]
    y = labels[ei_rows, EI]
    # Corrected physical parent: independently observed Eea and Egc are
    # legitimate covariates, but Ei itself is not used in any fit.
    both = observed[ei_rows, 4] & observed[ei_rows, 1]
    parent_ei = parent[ei_rows, EI].copy()
    parent_ei[both] = 0.5 * parent_ei[both] + 0.5 * (labels[ei_rows[both], 4] + labels[ei_rows[both], 1])
    partner = make_partner_matrix(labels, ei_rows, ei_rows)
    # Fit on a reproducible shuffled fold schedule.  The route is frozen before
    # looking at metrics: only missing-Eea/Egc rows are eligible for a new arm.
    folds = np.full(len(ei_rows), -1, dtype=int)
    for fold, (_, validation) in enumerate(KFold(5, shuffle=True, random_state=SEED + 167).split(ei_rows)):
        folds[validation] = fold
    kinds = ["ridge_abs", "ridge_res", "et_abs", "et_res", "hgb_abs", "hgb_res"]
    predictions = {kind: parent_ei.copy() for kind in kinds}
    fold_reports = {kind: [] for kind in kinds}
    eligible = ~both
    for fold in range(5):
        va = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        # Train on all Ei rows in the outer training split.  The target row is
        # never an input; other property values are covariates only.
        xtr = np.hstack([X[ei_rows[tr]], partner[tr]])
        xva = np.hstack([X[ei_rows[va]], partner[va]])
        for kind in kinds:
            residual = kind.endswith("_res")
            target = y[tr] - parent_ei[tr] if residual else y[tr]
            model = fit_model(kind, xtr, target, SEED + 167 + fold)
            raw = np.asarray(model.predict(xva), dtype=np.float64)
            if residual:
                raw = parent_ei[va] + 0.5 * raw
            else:
                raw = 0.5 * parent_ei[va] + 0.5 * raw
            use = eligible[va]
            predictions[kind][va[use]] = raw[use]
            fold_reports[kind].append({
                "fold": fold,
                "eligible_rows": int(np.sum(use)),
                "parent_r2_all": r2(y[va], parent_ei[va]),
                "candidate_r2_all": r2(y[va], predictions[kind][va]),
                "delta_r2_all": r2(y[va], predictions[kind][va]) - r2(y[va], parent_ei[va]),
            })
    metrics = {}
    for kind in kinds:
        metrics[kind] = {
            "parent_r2": r2(y, parent_ei),
            "candidate_r2": r2(y, predictions[kind]),
            "delta_r2": r2(y, predictions[kind]) - r2(y, parent_ei),
            "eligible_rows": int(np.sum(eligible)),
            "eligible_parent_r2": r2(y[eligible], parent_ei[eligible]),
            "eligible_candidate_r2": r2(y[eligible], predictions[kind][eligible]),
            "eligible_delta_r2": r2(y[eligible], predictions[kind][eligible]) - r2(y[eligible], parent_ei[eligible]),
            "folds": fold_reports[kind],
        }
    # A fixed equal blend is included as a stability check, not selected from
    # local_eval/test-external_label feedback.  The strongest clean arm is reported but no
    # full-data artifact is materialized by this screen.
    blend = np.mean(np.column_stack([predictions[k] for k in kinds]), axis=1)
    metrics["fixed_equal_blend"] = {
        "parent_r2": r2(y, parent_ei),
        "candidate_r2": r2(y, blend),
        "delta_r2": r2(y, blend) - r2(y, parent_ei),
        "eligible_rows": int(np.sum(eligible)),
        "eligible_parent_r2": r2(y[eligible], parent_ei[eligible]),
        "eligible_candidate_r2": r2(y[eligible], blend[eligible]),
        "eligible_delta_r2": r2(y[eligible], blend[eligible]) - r2(y[eligible], parent_ei[eligible]),
    }
    best = max(metrics, key=lambda k: metrics[k]["candidate_r2"])
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C167-ei-partner-conditioned-cached",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "same_row_ei_label_as_feature": False,
        "mechanism": "Ei-only absolute/residual Ridge, ExtraTrees, and HistGradientBoosting arms using cached official structure features plus observed non-Ei partner values and masks; both Eea+Egc rows retain C143-style identity parent",
        "rows": {"ei": int(len(ei_rows)), "eligible_missing_e ea_or_egc": int(np.sum(eligible)), "both_e ea_and_egc": int(np.sum(both))},
        "feature_count": int(X.shape[1] + partner.shape[1]),
        "metrics": metrics,
        "best_arm_by_clean_oof": best,
        "gate": {
            "ei_gain_at_least_0.005": bool(metrics[best]["delta_r2"] >= 0.005),
            "eligible_gain_at_least_0.010": bool(metrics[best]["eligible_delta_r2"] >= 0.010),
            "passed_screen": bool(metrics[best]["delta_r2"] >= 0.005 and metrics[best]["eligible_delta_r2"] >= 0.010),
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C167-ei-partner-conditioned-cached-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["gate"]["passed_screen"]:
        print("C167 STOP: Ei screen failed; no full-data fit, local_eval read, or submission artifact.", flush=True)


if __name__ == "__main__":
    main()
