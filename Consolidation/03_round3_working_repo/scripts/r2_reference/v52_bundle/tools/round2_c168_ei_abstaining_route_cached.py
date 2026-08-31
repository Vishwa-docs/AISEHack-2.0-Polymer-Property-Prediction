"""C168: clean Ei correction portfolio with fold-local abstention.

The parent is retained by default.  Three fixed correction arms are proposed:
an observed-partner Ei=Eea+Egc identity, Tanimoto read-across from official Ei
labels, and a structure residual Ridge.  For every outer fold, a second inner
OOF pass trains a LogisticRegression gate on official training labels only.
The gate applies an arm only when its inner probability of beating the parent
is at least 0.75.  No test external_labels, local_eval values, or post-freeze scores are
read, and this screen does not fit a full-data submission.
"""
from pathlib import Path
import importlib.util
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
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
GATE_THRESHOLD = 0.75
started = time.time()


def r2(y, p):
    return float(r2_score(np.asarray(y, float), np.asarray(p, float)))


def load_all():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    P = pickle.loads((SCR / "physics.pkl").read_bytes())
    G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    spec = importlib.util.spec_from_file_location("c162_feature_builder", ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    X = mod.features(F, P, G).astype(np.float64)
    cmap, idx = F["canon_map"], F["idx"]
    ns = len(F["canon_list"])
    labels = np.full((ns, len(TARGETS)), np.nan, dtype=np.float64)
    for name in ("archive/train.csv", "train.csv"):
        frame = pd.read_csv(BASE / name); frame["canon"] = frame["smiles"].map(cmap)
        for j, target in enumerate(TARGETS):
            vals = frame.loc[frame["target_type"].eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items(): labels[idx[canon], j] = float(value)
    return F, X, labels


def similarity_features(bits, query, pool):
    """Return max, top-5 mean, and top-5 weighted read-across prediction."""
    q = bits[int(query)].astype(np.float64)
    p = bits[np.asarray(pool, dtype=int)].astype(np.float64)
    inter = p @ q
    denom = p.sum(1) + q.sum() - inter + 1e-12
    sim = inter / denom
    order = np.argsort(sim)[::-1]
    top = order[: min(5, len(order))]
    ss = sim[top]
    weights = np.maximum(ss, 1e-6) ** 4
    return float(ss[0]), float(np.mean(ss)), float(np.sum(weights)), top, ss


def readacross(bits, query_rows, train_rows, y_train):
    pred = np.zeros(len(query_rows), dtype=np.float64)
    meta = np.zeros((len(query_rows), 3), dtype=np.float64)
    for pos, query in enumerate(np.asarray(query_rows, dtype=int)):
        mx, mean5, _, top, ss = similarity_features(bits, query, train_rows)
        weights = np.maximum(ss, 1e-6) ** 4
        pred[pos] = float(np.sum(weights * y_train[top]) / np.sum(weights))
        meta[pos] = [mx, mean5, float(len(train_rows))]
    return pred, meta


def structural_residual(X, train_rows, query_rows, y_global, parent_global, alpha=50.0):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=alpha))
    model.fit(X[train_rows], y_global[train_rows] - parent_global[train_rows])
    return parent_global[query_rows] + 0.30 * np.asarray(model.predict(X[query_rows]), dtype=np.float64)


def correction_features(parent, correction, partner_mask, sim_meta):
    return np.column_stack([
        parent, correction, correction - parent, np.abs(correction - parent),
        partner_mask.astype(float), sim_meta,
    ]).astype(np.float64)


def arm_predictions(kind, fit_rows, query_rows, y_global, parent_global, labels, bits, X):
    """Create a correction without using query Ei labels."""
    if kind == "identity":
        correction = parent_global[query_rows].copy()
        available = np.isfinite(labels[query_rows, 1]) & np.isfinite(labels[query_rows, 4])
        correction[available] = labels[query_rows[available], 1] + labels[query_rows[available], 4]
        sim_meta = np.zeros((len(query_rows), 3), dtype=np.float64)
        return correction, available, sim_meta
    if kind == "readacross":
        correction, sim_meta = readacross(bits, query_rows, fit_rows, y_global[fit_rows])
        available = np.ones(len(query_rows), dtype=bool)
        return correction, available, sim_meta
    correction = structural_residual(X, fit_rows, query_rows, y_global, parent_global)
    sim_meta = np.zeros((len(query_rows), 3), dtype=np.float64)
    available = np.ones(len(query_rows), dtype=bool)
    return correction, available, sim_meta


def fit_gate(train_features, train_good):
    good = np.asarray(train_good, dtype=int)
    if len(np.unique(good)) < 2 or len(good) < 20:
        return None
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED),
    ).fit(train_features, good)


def main():
    F, X, labels = load_all()
    bits = np.unpackbits(F["morgan_bin"], axis=1).astype(np.float64)
    parent_all = np.load(SCR / "out_clean_corrected/PFINAL.npy").astype(np.float64)
    observed = np.isfinite(labels)
    ns = len(labels)
    ei_rows = np.where(observed[:, EI])[0]
    y = labels[ei_rows, EI]
    parent = parent_all[ei_rows, EI].copy()
    y_global = np.full(ns, np.nan, dtype=np.float64)
    y_global[ei_rows] = y
    parent_global = parent_all[:, EI].copy()
    both = observed[ei_rows, 1] & observed[ei_rows, 4]
    parent[both] = 0.5 * parent[both] + 0.5 * (labels[ei_rows[both], 1] + labels[ei_rows[both], 4])
    partner_mask = (observed[ei_rows, 1].astype(int) + observed[ei_rows, 4].astype(int))
    kinds = ["identity", "readacross", "structural"]
    outer = list(KFold(5, shuffle=True, random_state=SEED + 168).split(ei_rows))
    candidate = parent.copy()
    arm_routes = {kind: 0 for kind in kinds}
    folds = []
    for fold, (outer_tr, outer_va) in enumerate(outer):
        inner_splits = list(KFold(4, shuffle=True, random_state=SEED + 1680 + fold).split(outer_tr))
        gates = {}
        for kind in kinds:
            inner_pred = np.full(len(outer_tr), np.nan)
            inner_sim = np.zeros((len(outer_tr), 3), dtype=np.float64)
            inner_available = np.zeros(len(outer_tr), dtype=bool)
            for inner_fit_pos, inner_va_pos in inner_splits:
                fit_local = outer_tr[inner_fit_pos]
                va_local = outer_tr[inner_va_pos]
                fit_global = ei_rows[fit_local]
                va_global = ei_rows[va_local]
                raw, available, sim_meta = arm_predictions(kind, fit_global, va_global, y_global, parent_global, labels, bits, X)
                # Structural and read-across arms need predictions in the local
                # target index; parent is globally indexed above.
                inner_pred[inner_va_pos] = raw
                inner_sim[inner_va_pos] = sim_meta
                inner_available[inner_va_pos] = available
            valid = np.isfinite(inner_pred) & inner_available
            if np.sum(valid) < 20:
                gates[kind] = None
                continue
            parent_inner = parent[outer_tr]
            feat = correction_features(parent_inner[valid], inner_pred[valid], partner_mask[outer_tr][valid], inner_sim[valid])
            good = (np.square(y[outer_tr][valid] - inner_pred[valid]) < np.square(y[outer_tr][valid] - parent_inner[valid])).astype(int)
            gates[kind] = fit_gate(feat, good)
        va_global = ei_rows[outer_va]
        chosen = parent[outer_va].copy()
        route_counts = {kind: 0 for kind in kinds}
        for kind in kinds:
            fit_global = ei_rows[outer_tr]
            raw, available, sim_meta = arm_predictions(kind, fit_global, va_global, y_global, parent_global, labels, bits, X)
            feat = correction_features(parent[outer_va], raw, partner_mask[outer_va], sim_meta)
            gate = gates[kind]
            if gate is None:
                use = np.zeros(len(outer_va), dtype=bool)
            else:
                use = available & (gate.predict_proba(feat)[:, 1] >= GATE_THRESHOLD)
            # Keep the first accepted arm in fixed priority order.  This avoids
            # stacking several data-dependent corrections on one row.
            use &= chosen == parent[outer_va]
            chosen[use] = raw[use]
            route_counts[kind] = int(np.sum(use))
            arm_routes[kind] += route_counts[kind]
        candidate[outer_va] = chosen
        folds.append({
            "fold": fold,
            "rows": int(len(outer_va)),
            "parent_r2": r2(y[outer_va], parent[outer_va]),
            "candidate_r2": r2(y[outer_va], chosen),
            "delta_r2": r2(y[outer_va], chosen) - r2(y[outer_va], parent[outer_va]),
            "routes": route_counts,
        })
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C168-ei-abstaining-route-cached",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "same_row_ei_label_as_feature": False,
        "mechanism": "fixed identity/read-across/structure-residual portfolio with 4-fold inner official-only LogisticRegression abstention gate, threshold 0.75, unchanged parent fallback",
        "rows": {"ei": int(len(ei_rows)), "both_e ea_and_egc": int(np.sum(both)), "missing_one_or_both": int(np.sum(~both))},
        "gate_threshold": GATE_THRESHOLD,
        "routes_total": arm_routes,
        "parent_r2": r2(y, parent),
        "candidate_r2": r2(y, candidate),
        "delta_r2": r2(y, candidate) - r2(y, parent),
        "folds": folds,
        "gate": {
            "ei_gain_at_least_0.005": bool(r2(y, candidate) - r2(y, parent) >= 0.005),
            "positive_folds_at_least_4": bool(sum(row["delta_r2"] > 0 for row in folds) >= 4),
            "routes_nonzero": bool(sum(arm_routes.values()) > 0),
            "passed_screen": bool(r2(y, candidate) - r2(y, parent) >= 0.005 and sum(row["delta_r2"] > 0 for row in folds) >= 4),
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "R2-C168-ei-abstaining-route-cached-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["gate"]["passed_screen"]:
        print("C168 STOP: abstaining Ei screen failed; no full-data fit, local_eval read, or submission artifact.", flush=True)


if __name__ == "__main__":
    main()
