"""C163: dedicated Nc residual on top of the clean C162 optical route."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold

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
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_c162_module():
    path = ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py"
    spec = importlib.util.spec_from_file_location("c162_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fit_c162_oof(mod, X, labels, obs, parent, pair_rows):
    eps_y = labels[pair_rows, ti["eps"]]
    nc_y = labels[pair_rows, ti["nc"]]
    log_ionic = np.log(eps_y - nc_y ** 2)
    eps_oof = {k: parent[pair_rows, ti["eps"]].copy() for k in ("ridge", "et", "hgb")}
    nc_oof = {k: parent[pair_rows, ti["nc"]].copy() for k in ("ridge", "et", "hgb")}
    kf = KFold(5, shuffle=True, random_state=SEED + 162)
    for fold, (tr, va) in enumerate(kf.split(pair_rows), 1):
        arms = {}
        for kind in ("ridge", "et", "hgb"):
            m = mod.model(kind, SEED + fold)
            if kind == "ridge":
                mu = X[pair_rows[tr]].mean(0); sd = X[pair_rows[tr]].std(0); sd[sd < 1e-12] = 1.0
                m.fit((X[pair_rows[tr]] - mu) / sd, log_ionic[tr])
                pred = m.predict((X[pair_rows[va]] - mu) / sd)
            else:
                m.fit(X[pair_rows[tr]], log_ionic[tr]); pred = m.predict(X[pair_rows[va]])
            arms[kind] = np.exp(np.clip(pred, -8, 4))
        rr = pair_rows[va]
        for kind, ip in arms.items():
            eps_oof[kind][va] = nc_y[va] ** 2 + ip
            nc_oof[kind][va] = np.sqrt(np.maximum(eps_y[va] - ip, .05 ** 2))
    eps = .5 * parent[pair_rows, ti["eps"]] + .5 * np.mean(np.column_stack([eps_oof[k] for k in ("ridge", "et", "hgb")]), axis=1)
    nc = .5 * parent[pair_rows, ti["nc"]] + .5 * np.mean(np.column_stack([nc_oof[k] for k in ("ridge", "et", "hgb")]), axis=1)
    return eps, nc


def main():
    mod = load_c162_module()
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    P = pickle.loads((SCR / "physics.pkl").read_bytes())
    G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    idx, cmap = F["idx"], F["canon_map"]
    ns = len(F["canon_list"])
    train = pd.read_csv(BASE / "train.csv"); archive = pd.read_csv(BASE / "archive/train.csv"); test = pd.read_csv(BASE / "test.csv")
    for frame in (train, archive, test):
        frame["canon"] = frame["smiles"].map(cmap); frame["fi"] = frame["canon"].map(idx).astype(int)
    labels = np.full((ns, 7), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items(): labels[idx[canon], j] = value
    obs = np.isfinite(labels)
    P1 = np.load(SCR / "out_clean_corrected/P1.npy")
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both_ei = obs[:, ti["ei"]] & obs[:, ti["eea"]] & obs[:, ti["egc"]]
    parent[both_ei, ti["ei"]] = .5 * parent[both_ei, ti["ei"]] + .5 * (labels[both_ei, ti["eea"]] + labels[both_ei, ti["egc"]])
    X = mod.features(F, P, G)
    pair_rows = np.where(obs[:, ti["eps"]] & obs[:, ti["nc"]])[0]
    c162_eps, c162_nc = fit_c162_oof(mod, X, labels, obs, parent, pair_rows)
    eps_y = labels[pair_rows, ti["eps"]]; nc_y = labels[pair_rows, ti["nc"]]

    # Refractivity-aware features for the Nc residual. EPS is a counterpart
    # label, not the current target, and is therefore permitted on supported
    # rows. The remaining blocks are structure-only.
    eps_counterpart = eps_y
    n_atoms = np.asarray(F["blocks"]["desc"][:, 0], float) if F["blocks"]["desc"].shape[1] else np.zeros(ns)
    extra = np.asarray(F["blocks"]["extra"], float)
    extra[~np.isfinite(extra)] = 0.0
    base_opt = np.column_stack([
        X[pair_rows],
        eps_counterpart,
        np.log(np.maximum(eps_counterpart, .05)),
        parent[pair_rows, ti["nc"]],
        parent[pair_rows, ti["eps"]],
        c162_nc,
        c162_eps,
        np.log(np.maximum(c162_eps, .05)),
        extra[pair_rows, :min(extra.shape[1], 32)],
    ]).astype(np.float64)
    residual = nc_y - c162_nc
    folds = []
    nc_candidate = c162_nc.copy()
    fold_preds = np.full(len(pair_rows), np.nan)
    kf = KFold(5, shuffle=True, random_state=SEED + 163)
    for fold, (tr, va) in enumerate(kf.split(pair_rows), 1):
        preds = []
        for kind in ("ridge", "huber", "et"):
            if kind == "ridge":
                mu = base_opt[tr].mean(0); sd = base_opt[tr].std(0); sd[sd < 1e-12] = 1.0
                m = Ridge(alpha=25.0).fit((base_opt[tr] - mu) / sd, residual[tr])
                preds.append(m.predict((base_opt[va] - mu) / sd))
            elif kind == "huber":
                mu = base_opt[tr].mean(0); sd = base_opt[tr].std(0); sd[sd < 1e-12] = 1.0
                m = HuberRegressor(epsilon=1.35, alpha=0.002, max_iter=400).fit((base_opt[tr] - mu) / sd, residual[tr])
                preds.append(m.predict((base_opt[va] - mu) / sd))
            else:
                m = ExtraTreesRegressor(n_estimators=450, max_features=.55, min_samples_leaf=3, n_jobs=10, random_state=SEED + fold).fit(base_opt[tr], residual[tr])
                preds.append(m.predict(base_opt[va]))
        fold_preds[va] = np.mean(np.column_stack(preds), axis=1)
        nc_candidate[va] = c162_nc[va] + .5 * fold_preds[va]
        folds.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})

    candidate = parent.copy()
    candidate[pair_rows, ti["eps"]] = c162_eps
    candidate[pair_rows, ti["nc"]] = nc_candidate
    metrics = {}
    for j, target in enumerate(TARGETS):
        rows = np.where(obs[:, j])[0]
        pr = r2(labels[rows, j], parent[rows, j]); cr = r2(labels[rows, j], candidate[rows, j])
        metrics[target] = {"n": int(len(rows)), "parent_r2": pr, "candidate_r2": cr, "delta_r2": cr - pr}
    parent_mean = float(np.mean([v["parent_r2"] for v in metrics.values()])); candidate_mean = float(np.mean([v["candidate_r2"] for v in metrics.values()]))
    deltas = np.array([v["delta_r2"] for v in metrics.values()])
    gate = {
        "mean_gain_at_least_0.002": bool(candidate_mean - parent_mean >= .002),
        "no_target_loss_worse_than_0.003": bool(np.min(deltas) >= -.003),
        "eps_not_below_c162_by_0.003": bool(metrics["eps"]["candidate_r2"] >= .8927568748456813 - .003),
        "nc_gain_at_least_0.010": bool(metrics["nc"]["delta_r2"] >= .010),
        "passed": bool(candidate_mean - parent_mean >= .002 and np.min(deltas) >= -.003 and metrics["eps"]["candidate_r2"] >= .8927568748456813 - .003 and metrics["nc"]["delta_r2"] >= .010),
    }
    report = {
        "schema_version": "ppp.round2.clean-oof.v1", "experiment": "R2-C163-nc-specialist", "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False,
        "mechanism": "C162 EPS route preserved; Nc half-parent residual ensemble of standardized Ridge, Huber, and ExtraTrees using structure/refractivity/official EPS counterpart features",
        "pair_rows": int(len(pair_rows)), "folds": folds, "metrics": metrics, "parent_mean_r2": parent_mean, "candidate_mean_r2": candidate_mean, "gate": gate,
        "elapsed_seconds": time.time() - started,
    }
    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    (CLEAN_OUT / "R2-C163-nc-specialist-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not gate["passed"]:
        print("C163 STOP: clean gate failed; no full-data fit and no score verification.", flush=True)
        return
    print("C163 clean gate passed; full-data materialization is intentionally deferred to the next audited packaging step.", flush=True)


if __name__ == "__main__":
    main()
