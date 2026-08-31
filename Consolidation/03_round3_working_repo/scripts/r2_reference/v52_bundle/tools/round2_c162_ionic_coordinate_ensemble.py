"""C162: fixed clean EPS/Nc ionic-coordinate ensemble.

This is independent of the interrupted C160/C161 arms.  The current target
is excluded from every feature row; an independently observed counterpart is
allowed, and rows without it retain the corrected structure-only parent.
"""
from pathlib import Path
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
CLEAN_OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
LOCAL_EVAL_OUT = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY"
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
started = time.time()


def r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def clean(x):
    x = np.asarray(x, np.float64)
    x[~np.isfinite(x)] = np.nan
    return np.clip(x, -1e10, 1e10)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def features(F, P, G):
    blocks = F["blocks"]
    PG = G["M"]
    PGN = PG / np.maximum(1.0, PG.sum(1, keepdims=True))
    dense = np.hstack([
        clean(blocks["desc"]), clean(blocks["extra"]), clean(blocks["oligo"]),
        clean(blocks["ipc"]), clean(P["M"]), clean(PGN), clean(G["morph"]),
    ])
    keep = np.array([np.nanstd(dense[:, j]) > 1e-12 and np.isfinite(dense[:, j]).mean() > .3 for j in range(dense.shape[1])])
    dense = dense[:, keep]
    med = np.nanmedian(dense, axis=0)
    bad = np.where(~np.isfinite(dense))
    dense[bad] = np.take(med, bad[1])
    dense_qt = QuantileTransformer(n_quantiles=min(1000, len(dense)), output_distribution="normal", random_state=SEED).fit_transform(dense)
    sparse = np.hstack([
        np.log1p(blocks["morgan"]), blocks["maccs"], np.log1p(blocks["ap"]),
        np.log1p(blocks["tt"]), np.log1p(blocks["rk"]), np.log1p(PG),
    ]).astype(np.float32)
    sparse = sparse[:, (sparse != 0).sum(0) >= 4]
    svd = TruncatedSVD(192, random_state=SEED).fit_transform(sparse)
    return np.hstack([dense_qt, svd, blocks["maccs"]]).astype(np.float32)


def model(kind, seed):
    if kind == "ridge":
        return Ridge(alpha=50.0)
    if kind == "et":
        return ExtraTreesRegressor(n_estimators=500, max_features=.55, min_samples_leaf=2, n_jobs=10, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=260, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1.0, random_state=seed)


def main():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    P = pickle.loads((SCR / "physics.pkl").read_bytes())
    G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    idx, cmap = F["idx"], F["canon_map"]
    ns = len(F["canon_list"])
    train = pd.read_csv(BASE / "train.csv")
    archive = pd.read_csv(BASE / "archive/train.csv")
    test = pd.read_csv(BASE / "test.csv")
    for frame in (train, archive, test):
        frame["canon"] = frame["smiles"].map(cmap)
        frame["fi"] = frame["canon"].map(idx).astype(int)
    labels = np.full((ns, 7), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items():
                labels[idx[canon], j] = value
    obs = np.isfinite(labels)
    P1 = np.load(SCR / "out_clean_corrected/P1.npy")
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both_ei = obs[:, ti["ei"]] & obs[:, ti["eea"]] & obs[:, ti["egc"]]
    parent[both_ei, ti["ei"]] = .5 * parent[both_ei, ti["ei"]] + .5 * (labels[both_ei, ti["eea"]] + labels[both_ei, ti["egc"]])
    X = features(F, P, G)
    pair_rows = np.where(obs[:, ti["eps"]] & obs[:, ti["nc"]])[0]
    eps_y = labels[pair_rows, ti["eps"]]
    nc_y = labels[pair_rows, ti["nc"]]
    ionic_y = eps_y - nc_y ** 2
    if np.any(ionic_y <= 0):
        raise RuntimeError("non-positive ionic coordinate")
    log_ionic = np.log(ionic_y)
    eps_oof = {k: parent[pair_rows, ti["eps"]].copy() for k in ("ridge", "et", "hgb")}
    nc_oof = {k: parent[pair_rows, ti["nc"]].copy() for k in ("ridge", "et", "hgb")}
    kf = KFold(5, shuffle=True, random_state=SEED + 162)
    folds = []
    for fold, (tr, va) in enumerate(kf.split(pair_rows), 1):
        arms = {}
        for kind in ("ridge", "et", "hgb"):
            # Structure-only coordinate prediction. The counterpart target is
            # used only algebraically after prediction, never as a feature.
            m = model(kind, SEED + fold)
            if kind == "ridge":
                mu = X[pair_rows[tr]].mean(0)
                sd = X[pair_rows[tr]].std(0)
                sd[sd < 1e-12] = 1.0
                m.fit((X[pair_rows[tr]] - mu) / sd, log_ionic[tr])
                pred = m.predict((X[pair_rows[va]] - mu) / sd)
            else:
                m.fit(X[pair_rows[tr]], log_ionic[tr])
                pred = m.predict(X[pair_rows[va]])
            arms[kind] = np.exp(np.clip(pred, -8, 4))
        for kind, ip in arms.items():
            rr = pair_rows[va]
            eps_oof[kind][va] = nc_y[va] ** 2 + ip
            nc_oof[kind][va] = np.sqrt(np.maximum(eps_y[va] - ip, .05 ** 2))
        folds.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})

    # Fixed ensemble: retain half of the corrected parent and average the
    # three independently trained coordinate arms in the other half.
    eps_ens = .5 * parent[pair_rows, ti["eps"]] + .5 * np.mean(np.column_stack([eps_oof[k] for k in ("ridge", "et", "hgb")]), axis=1)
    nc_ens = .5 * parent[pair_rows, ti["nc"]] + .5 * np.mean(np.column_stack([nc_oof[k] for k in ("ridge", "et", "hgb")]), axis=1)
    arm_metrics = {}
    for kind in ("ridge", "et", "hgb"):
        arm_metrics[kind] = {
            "eps_r2": r2(eps_y, eps_oof[kind]),
            "nc_r2": r2(nc_y, nc_oof[kind]),
        }
    arm_metrics["ensemble"] = {"eps_r2": r2(eps_y, eps_ens), "nc_r2": r2(nc_y, nc_ens)}
    parent_pair = {"eps_r2": r2(eps_y, parent[pair_rows, ti["eps"]]), "nc_r2": r2(nc_y, parent[pair_rows, ti["nc"]])}
    candidate = parent.copy()
    candidate[pair_rows, ti["eps"]] = eps_ens
    candidate[pair_rows, ti["nc"]] = nc_ens
    metrics = {}
    for j, target in enumerate(TARGETS):
        rows = np.where(obs[:, j])[0]
        pr = r2(labels[rows, j], parent[rows, j])
        cr = r2(labels[rows, j], candidate[rows, j])
        metrics[target] = {"n": int(len(rows)), "parent_r2": pr, "candidate_r2": cr, "delta_r2": cr - pr}
    parent_mean = float(np.mean([v["parent_r2"] for v in metrics.values()]))
    candidate_mean = float(np.mean([v["candidate_r2"] for v in metrics.values()]))
    deltas = np.array([v["delta_r2"] for v in metrics.values()])
    gate = {
        "mean_gain_at_least_0.002": bool(candidate_mean - parent_mean >= .002),
        "no_target_loss_worse_than_0.003": bool(np.min(deltas) >= -.003),
        "eps_gain_at_least_0.010": bool(metrics["eps"]["delta_r2"] >= .010),
        "nc_gain_at_least_0.010": bool(metrics["nc"]["delta_r2"] >= .010),
        "passed": bool(candidate_mean - parent_mean >= .002 and np.min(deltas) >= -.003 and metrics["eps"]["delta_r2"] >= .010 and metrics["nc"]["delta_r2"] >= .010),
    }
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C162-ionic-coordinate-ensemble",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "mechanism": "structure-only log(EPS-Nc^2) Ridge/ExtraTrees/HistGradientBoosting fixed equal ensemble, half-parent blend, paired counterpart algebra",
        "pair_rows": int(len(pair_rows)),
        "folds": folds,
        "parent_pair_metrics": parent_pair,
        "arm_metrics": arm_metrics,
        "metrics": metrics,
        "parent_mean_r2": parent_mean,
        "candidate_mean_r2": candidate_mean,
        "gate": gate,
        "elapsed_seconds": time.time() - started,
    }
    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    (CLEAN_OUT / "R2-C162-ionic-coordinate-ensemble-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not gate["passed"]:
        print("C162 STOP: clean gate failed; no full-data fit and no score verification.", flush=True)
        return

    # Full-data fit is kept in the isolated post-freeze lane and is not read by
    # the clean pipeline. It starts from the strongest already-frozen clean
    # EPS-only carrier, then replaces only paired EPS/Nc rows.
    source = LOCAL_EVAL_OUT / "R2-C148-et-eps-only-LOCAL_DIAGNOSTIC_ONLY.csv"
    full = pd.read_csv(source)
    full_models = {}
    for kind in ("ridge", "et", "hgb"):
        m = model(kind, SEED)
        if kind == "ridge":
            mu = X[pair_rows].mean(0); sd = X[pair_rows].std(0); sd[sd < 1e-12] = 1.0
            m.fit((X[pair_rows] - mu) / sd, log_ionic)
            full_models[kind] = (m, mu, sd)
        else:
            m.fit(X[pair_rows], log_ionic)
            full_models[kind] = (m, None, None)
    test_fi = test["fi"].to_numpy()
    for i, row in test.iterrows():
        fi = int(row.fi)
        preds = []
        for kind in ("ridge", "et", "hgb"):
            m, mu, sd = full_models[kind]
            xp = X[fi:fi + 1]
            if kind == "ridge":
                pred = m.predict((xp - mu) / sd)[0]
            else:
                pred = m.predict(xp)[0]
            preds.append(float(np.exp(np.clip(pred, -8, 4))))
        ionic_pred = float(np.mean(preds))
        if row.target_type == "eps" and obs[fi, ti["nc"]]:
            raw = labels[fi, ti["nc"]] ** 2 + ionic_pred
            full.loc[i, "target"] = .5 * full.loc[i, "target"] + .5 * raw
        elif row.target_type == "nc" and obs[fi, ti["eps"]]:
            raw = np.sqrt(max(labels[fi, ti["eps"]] - ionic_pred, .05 ** 2))
            full.loc[i, "target"] = .5 * full.loc[i, "target"] + .5 * raw
    out = LOCAL_EVAL_OUT / "R2-C162-ionic-coordinate-ensemble-LOCAL_DIAGNOSTIC_ONLY.csv"
    full.to_csv(out, index=False)
    post = {"schema_version": "ppp.round2.postfreeze-candidate.v1", "experiment": report["experiment"], "classification": "LOCAL_DIAGNOSTIC_ONLY", "local_eval_read": False, "clean_gate": gate, "candidate_path": str(out), "candidate_sha256": sha256(out), "rows": int(len(full))}
    (LOCAL_EVAL_OUT / "R2-C162-ionic-coordinate-ensemble-LOCAL_DIAGNOSTIC_ONLY.json").write_text(json.dumps(post, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(post, indent=2), flush=True)


if __name__ == "__main__":
    main()
