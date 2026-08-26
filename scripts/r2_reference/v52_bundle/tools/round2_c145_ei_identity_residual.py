"""C145: target-excluded structural residual on Ei = Egc + Eea."""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
OUT = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY"
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
started = time.time()

def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def clean(x):
    x = np.asarray(x, np.float64); x[~np.isfinite(x)] = np.nan
    return np.clip(x, -1e10, 1e10)

def fit_ridge(x, y, alpha=50.0):
    mu = x.mean(0, dtype=np.float64); sd = x.std(0, dtype=np.float64); sd[sd < 1e-12] = 1.0
    return Ridge(alpha=alpha).fit((x - mu) / sd, y), mu, sd

def pred(fit, x):
    model, mu, sd = fit
    return model.predict((x - mu) / sd)

F = pickle.loads((SCR / "features.pkl").read_bytes())
P = pickle.loads((SCR / "physics.pkl").read_bytes())
G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
idx, canon_map, blocks = F["idx"], F["canon_map"], F["blocks"]
ns = len(F["canon_list"])
train = pd.read_csv(BASE / "train.csv"); test = pd.read_csv(BASE / "test.csv"); archive = pd.read_csv(BASE / "archive/train.csv")
for frame in (train, test, archive):
    frame["canon"] = frame["smiles"].map(canon_map); frame["fi"] = frame["canon"].map(idx).astype(int)
labels = np.full((ns, len(TARGETS)), np.nan)
for j, target in enumerate(TARGETS):
    for frame in (archive, train):
        vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
        for canon, value in vals.items(): labels[idx[canon], j] = value
obs = np.isfinite(labels)

PG = G["M"]; PGN = PG / np.maximum(1.0, PG.sum(1, keepdims=True))
dense = np.hstack([clean(blocks["desc"]), clean(blocks["extra"]), clean(blocks["oligo"]), clean(blocks["ipc"]), clean(P["M"]), clean(PGN), clean(G["morph"])])
keep = np.array([np.nanstd(dense[:, j]) > 1e-12 and np.isfinite(dense[:, j]).mean() > .3 for j in range(dense.shape[1])])
dense = dense[:, keep]; med = np.nanmedian(dense, 0); miss = np.where(~np.isfinite(dense)); dense[miss] = np.take(med, miss[1])
dense_qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED).fit_transform(dense)
sparse = np.hstack([np.log1p(blocks["morgan"]), blocks["maccs"], np.log1p(blocks["ap"]), np.log1p(blocks["tt"]), np.log1p(blocks["rk"]), np.log1p(PG)]).astype(np.float32)
sparse = sparse[:, (sparse != 0).sum(0) >= 4]
svd = TruncatedSVD(192, random_state=SEED).fit_transform(sparse)
X = np.hstack([dense_qt, svd, blocks["maccs"]]).astype(np.float64)

ei_rows = np.where(obs[:, ti["ei"]])[0]
both = obs[:, ti["egc"]] & obs[:, ti["eea"]] & obs[:, ti["ei"]]
fit_rows = np.where(both)[0]
residual = labels[fit_rows, ti["ei"]] - labels[fit_rows, ti["egc"]] - labels[fit_rows, ti["eea"]]
base = np.load(SCR / "out_clean_corrected/PFINAL.npy")
base_ei = base[ei_rows, ti["ei"]].copy()
ei_pos = {r: i for i, r in enumerate(ei_rows)}
both_pos = {r: i for i, r in enumerate(fit_rows)}
identity = labels[:, ti["egc"]] + labels[:, ti["eea"]]
for r in ei_rows:
    if both[r]: base_ei[ei_pos[r]] = .5 * base[r, ti["ei"]] + .5 * identity[r]

candidate_ei = base_ei.copy(); folds = []
for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + 145).split(fit_rows), 1):
    train_rows = fit_rows[tr]; val_rows = fit_rows[va]
    fit = fit_ridge(X[train_rows], residual[tr], alpha=50.0)
    residual_hat = pred(fit, X[val_rows])
    for r, rh in zip(val_rows, residual_hat):
        i = ei_pos[r]
        physics = identity[r] + rh
        candidate_ei[i] = .5 * base_ei[i] + .5 * physics
    folds.append({"fold": fold, "fit_rows": int(len(train_rows)), "validation_rows": int(len(val_rows))})

y = labels[ei_rows, ti["ei"]]
metrics = {"base_c143_style_r2": r2(y, base_ei), "candidate_r2": r2(y, candidate_ei)}
metrics["delta_r2"] = metrics["candidate_r2"] - metrics["base_c143_style_r2"]
print("C145 Ei OOF", json.dumps(metrics, sort_keys=True), flush=True)

full_fit = fit_ridge(X[fit_rows], residual, alpha=50.0)
base_path = OUT / "R2-C144-log-ionic-paired-reconstruction-LOCAL_DIAGNOSTIC_ONLY.csv"
candidate = pd.read_csv(base_path)
test_fi = test.fi.to_numpy()
resid_test = pred(full_fit, X[test_fi])
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "ei" and obs[fi, ti["egc"]] and obs[fi, ti["eea"]]:
        physics = labels[fi, ti["egc"]] + labels[fi, ti["eea"]] + resid_test[i]
        candidate.loc[i, "target"] = .5 * candidate.loc[i, "target"] + .5 * physics

name = "R2-C145-ei-identity-structural-residual-LOCAL_DIAGNOSTIC_ONLY"
path = OUT / f"{name}.csv"; candidate.to_csv(path, index=False)
report = {"experiment": name, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "Ei identity plus structure-only Ridge residual, 5-fold target-excluded fit, fixed 0.5 blend, both partners observed only", "fit_rows": int(len(fit_rows)), "folds": folds, "metrics": metrics, "candidate_path": str(path), "elapsed_seconds": time.time() - started}
(OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("C145 candidate", path, len(candidate), flush=True)
