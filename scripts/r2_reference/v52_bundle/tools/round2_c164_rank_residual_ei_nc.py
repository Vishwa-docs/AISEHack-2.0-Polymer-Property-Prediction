"""C164: fixed rank/sign residual correction for Ei and Nc.

Unlike coordinate reconstruction, this arm learns only a coarse high/low
residual direction from structure and independently observed non-target
partners. The target column is removed from every feature row.
"""
from pathlib import Path
import importlib.util
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
CLEAN_OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
started = time.time()


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def load_c162():
    path = ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py"
    spec = importlib.util.spec_from_file_location("c162_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def c162_oof(mod, X, labels, parent, pair_rows):
    eps_y = labels[pair_rows, ti["eps"]]; nc_y = labels[pair_rows, ti["nc"]]
    log_ionic = np.log(eps_y - nc_y ** 2)
    eo = {k: parent[pair_rows, ti["eps"]].copy() for k in ("ridge", "et", "hgb")}
    no = {k: parent[pair_rows, ti["nc"]].copy() for k in ("ridge", "et", "hgb")}
    for fold, (tr, va) in enumerate(__import__("sklearn.model_selection").model_selection.KFold(5, shuffle=True, random_state=SEED + 162).split(pair_rows), 1):
        for kind in ("ridge", "et", "hgb"):
            m = mod.model(kind, SEED + fold)
            if kind == "ridge":
                mu = X[pair_rows[tr]].mean(0); sd = X[pair_rows[tr]].std(0); sd[sd < 1e-12] = 1.0
                m.fit((X[pair_rows[tr]] - mu) / sd, log_ionic[tr]); pred = m.predict((X[pair_rows[va]] - mu) / sd)
            else:
                m.fit(X[pair_rows[tr]], log_ionic[tr]); pred = m.predict(X[pair_rows[va]])
            ip = np.exp(np.clip(pred, -8, 4)); eo[kind][va] = nc_y[va] ** 2 + ip; no[kind][va] = np.sqrt(np.maximum(eps_y[va] - ip, .05 ** 2))
    return .5 * parent[pair_rows, ti["eps"]] + .5 * np.mean(np.column_stack([eo[k] for k in ("ridge", "et", "hgb")]), axis=1), .5 * parent[pair_rows, ti["nc"]] + .5 * np.mean(np.column_stack([no[k] for k in ("ridge", "et", "hgb")]), axis=1)


def main():
    mod = load_c162()
    F = pickle.loads((SCR / "features.pkl").read_bytes()); P = pickle.loads((SCR / "physics.pkl").read_bytes()); G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    idx, cmap = F["idx"], F["canon_map"]; ns = len(F["canon_list"])
    train = pd.read_csv(BASE / "train.csv"); archive = pd.read_csv(BASE / "archive/train.csv")
    for frame in (train, archive): frame["canon"] = frame["smiles"].map(cmap)
    labels = np.full((ns, 7), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items(): labels[idx[canon], j] = value
    obs = np.isfinite(labels); X = mod.features(F, P, G); P1 = np.load(SCR / "out_clean_corrected/P1.npy"); base = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both = obs[:, ti["ei"]] & obs[:, ti["eea"]] & obs[:, ti["egc"]]
    base[both, ti["ei"]] = .5 * base[both, ti["ei"]] + .5 * (labels[both, ti["eea"]] + labels[both, ti["egc"]])
    pair = np.where(obs[:, ti["eps"]] & obs[:, ti["nc"]])[0]
    c162_eps, c162_nc = c162_oof(mod, X, labels, base, pair)
    candidate = base.copy(); candidate[pair, ti["eps"]] = c162_eps; candidate[pair, ti["nc"]] = c162_nc
    fold_records = {}; route_records = {}
    # Fixed sign correction: train only on the extreme residual thirds and
    # apply 15% of training residual std when classifier confidence is >.55.
    for target in ("ei", "nc"):
        j = ti[target]; rows = np.where(obs[:, j])[0]
        other = labels[rows].copy(); other[:, j] = np.nan
        other_values = np.nan_to_num(other, nan=0.0); other_flags = np.isfinite(other).astype(float)
        optical = np.column_stack([X[rows], other_values, other_flags, base[rows, j], c162_eps[np.searchsorted(pair, rows)] if target == "nc" and np.isin(rows, pair).all() else np.zeros(len(rows))])
        y = labels[rows, j]; parent = candidate[rows, j].copy(); residual = y - parent
        out = parent.copy(); groups = np.asarray([str(frame) for frame in rows], dtype=object)
        # Canonical rows are unique in the target-local label table; GroupKFold
        # still enforces a deterministic row-disjoint outer boundary.
        gkf = GroupKFold(n_splits=5)
        fold_rows = []
        for fold, (tr, va) in enumerate(gkf.split(optical, y, groups=groups), 1):
            q1, q2 = np.quantile(residual[tr], [1/3, 2/3]); extreme = (residual[tr] <= q1) | (residual[tr] >= q2)
            labels_rank = (residual[tr][extreme] >= q2).astype(int)
            clf = ExtraTreesClassifier(n_estimators=400, max_features=.6, min_samples_leaf=3, class_weight="balanced", n_jobs=10, random_state=SEED + fold)
            clf.fit(optical[tr][extreme], labels_rank)
            proba = clf.predict_proba(optical[va]); high_col = int(np.where(clf.classes_ == 1)[0][0]); high = proba[:, high_col]
            sign = np.where(high >= .55, 1.0, np.where(high <= .45, -1.0, 0.0))
            magnitude = .15 * float(np.std(residual[tr]))
            out[va] = parent[va] + sign * magnitude
            fold_rows.append({"fold": fold, "fit_rows": int(len(tr)), "extreme_fit_rows": int(np.sum(extreme)), "validation_rows": int(len(va)), "routed_rows": int(np.sum(sign != 0)), "magnitude": magnitude})
        candidate[rows, j] = out
        fold_records[target] = fold_rows; route_records[target] = {"rows": int(len(rows)), "routed_rows": int(np.sum(out != parent)), "residual_std": float(np.std(residual))}

    metrics = {}
    for j, target in enumerate(TARGETS):
        rows = np.where(obs[:, j])[0]; pr = r2(labels[rows, j], base[rows, j]); cr = r2(labels[rows, j], candidate[rows, j]); metrics[target] = {"n": int(len(rows)), "parent_r2": pr, "candidate_r2": cr, "delta_r2": cr - pr}
    # Preserve C162 EPS explicitly in the gate; Nc/Ei must be a material new
    # contribution and no other target may regress materially.
    mean_parent = float(np.mean([x["parent_r2"] for x in metrics.values()])); mean_candidate = float(np.mean([x["candidate_r2"] for x in metrics.values()])); deltas = np.array([x["delta_r2"] for x in metrics.values()])
    gate = {"mean_gain_at_least_0.002": bool(mean_candidate - mean_parent >= .002), "no_target_loss_worse_than_0.003": bool(np.min(deltas) >= -.003), "eps_at_least_c162_minus_0.003": bool(metrics["eps"]["candidate_r2"] >= .8927568748456813 - .003), "ei_or_nc_gain_at_least_0.010": bool(max(metrics["ei"]["delta_r2"], metrics["nc"]["delta_r2"]) >= .010), "passed": bool(mean_candidate - mean_parent >= .002 and np.min(deltas) >= -.003 and metrics["eps"]["candidate_r2"] >= .8927568748456813 - .003 and max(metrics["ei"]["delta_r2"], metrics["nc"]["delta_r2"]) >= .010)}
    report = {"schema_version": "ppp.round2.clean-oof.v1", "experiment": "R2-C164-rank-residual-ei-nc", "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "C162 EPS coordinate carrier plus fixed extreme-residual ExtraTrees sign classifier for Ei/Nc, 15-percent residual standard deviation correction", "folds": fold_records, "routes": route_records, "metrics": metrics, "parent_mean_r2": mean_parent, "candidate_mean_r2": mean_candidate, "gate": gate, "elapsed_seconds": time.time() - started}
    CLEAN_OUT.mkdir(parents=True, exist_ok=True); (CLEAN_OUT / "R2-C164-rank-residual-ei-nc-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2), flush=True)
    if not gate["passed"]: print("C164 STOP: clean gate failed; no full-data fit and no score verification.", flush=True)


if __name__ == "__main__": main()
