"""C147: partner-conditioned, target-excluded log ionic-coordinate models."""
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
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def clean(x):
    x = np.asarray(x, np.float64); x[~np.isfinite(x)] = np.nan
    return np.clip(x, -1e10, 1e10)

def fit_ridge(x, y, alpha=50.0):
    mu = x.mean(0, dtype=np.float64); sd = x.std(0, dtype=np.float64); sd[sd < 1e-12] = 1.0
    return Ridge(alpha=alpha).fit((x - mu) / sd, y), mu, sd

def pred(fit, x):
    m, mu, sd = fit
    return m.predict((x - mu) / sd)

F = pickle.loads((SCR / "features.pkl").read_bytes()); P = pickle.loads((SCR / "physics.pkl").read_bytes()); G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
idx, cmap, blocks = F["idx"], F["canon_map"], F["blocks"]
ns = len(F["canon_list"])
train = pd.read_csv(BASE / "train.csv"); test = pd.read_csv(BASE / "test.csv"); archive = pd.read_csv(BASE / "archive/train.csv")
for frame in (train, test, archive):
    frame["canon"] = frame["smiles"].map(cmap); frame["fi"] = frame["canon"].map(idx).astype(int)
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

pair_mask = obs[:, ti["eps"]] & obs[:, ti["nc"]]
pair_rows = np.where(pair_mask)[0]
ionic = labels[pair_rows, ti["eps"]] - labels[pair_rows, ti["nc"]] ** 2
if np.nanmin(ionic) <= 0: raise RuntimeError("non-positive ionic coordinate")
base = np.load(SCR / "out_clean_corrected/PFINAL.npy")
eps_rows = np.where(obs[:, ti["eps"]])[0]; nc_rows = np.where(obs[:, ti["nc"]])[0]
eps_pos = {r: i for i, r in enumerate(eps_rows)}; nc_pos = {r: i for i, r in enumerate(nc_rows)}
eps_oof = base[eps_rows, ti["eps"]].copy(); nc_oof = base[nc_rows, ti["nc"]].copy()
kf = KFold(5, shuffle=True, random_state=SEED + 147); folds = []
for fold, (tr, va) in enumerate(kf.split(pair_rows), 1):
    tr_rows = pair_rows[tr]; va_rows = pair_rows[va]
    ylog = np.log(np.clip(ionic[tr], 1e-6, None))
    x_eps = np.hstack([X[tr_rows], labels[tr_rows, ti["nc"]][:, None]])
    x_nc = np.hstack([X[tr_rows], labels[tr_rows, ti["eps"]][:, None]])
    fit_eps = fit_ridge(x_eps, ylog, alpha=50.0); fit_nc = fit_ridge(x_nc, ylog, alpha=50.0)
    ion_eps = np.exp(np.clip(pred(fit_eps, np.hstack([X[va_rows], labels[va_rows, ti["nc"]][:, None]])), -8, 4))
    ion_nc = np.exp(np.clip(pred(fit_nc, np.hstack([X[va_rows], labels[va_rows, ti["eps"]][:, None]])), -8, 4))
    for r, ip in zip(va_rows, ion_eps):
        raw = labels[r, ti["nc"]] ** 2 + ip
        eps_oof[eps_pos[r]] = .5 * base[r, ti["eps"]] + .5 * raw
    for r, ip in zip(va_rows, ion_nc):
        raw = np.sqrt(max(labels[r, ti["eps"]] - ip, .05 ** 2))
        nc_oof[nc_pos[r]] = .5 * base[r, ti["nc"]] + .5 * raw
    folds.append({"fold": fold, "fit_rows": int(len(tr_rows)), "validation_rows": int(len(va_rows))})

eps_y = labels[eps_rows, ti["eps"]]; nc_y = labels[nc_rows, ti["nc"]]
metrics = {
    "eps": {"base_r2": r2(eps_y, base[eps_rows, ti["eps"]]), "candidate_r2": r2(eps_y, eps_oof)},
    "nc": {"base_r2": r2(nc_y, base[nc_rows, ti["nc"]]), "candidate_r2": r2(nc_y, nc_oof)},
}
for v in metrics.values(): v["delta_r2"] = v["candidate_r2"] - v["base_r2"]
print("C147 OOF", json.dumps(metrics, sort_keys=True), flush=True)

full_y = np.log(np.clip(ionic, 1e-6, None))
fit_eps = fit_ridge(np.hstack([X[pair_rows], labels[pair_rows, ti["nc"]][:, None]]), full_y, alpha=50.0)
fit_nc = fit_ridge(np.hstack([X[pair_rows], labels[pair_rows, ti["eps"]][:, None]]), full_y, alpha=50.0)
test_fi = test.fi.to_numpy()
base_path = OUT / "R2-C144-log-ionic-paired-reconstruction-LOCAL_DIAGNOSTIC_ONLY.csv"
candidate = pd.read_csv(base_path)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "eps" and obs[fi, ti["nc"]]:
        ip = np.exp(np.clip(pred(fit_eps, np.r_[X[fi], labels[fi, ti["nc"]]][None, :])[0], -8, 4))
        candidate.loc[i, "target"] = .5 * candidate.loc[i, "target"] + .5 * (labels[fi, ti["nc"]] ** 2 + ip)
    elif row.target_type == "nc" and obs[fi, ti["eps"]]:
        ip = np.exp(np.clip(pred(fit_nc, np.r_[X[fi], labels[fi, ti["eps"]]][None, :])[0], -8, 4))
        candidate.loc[i, "target"] = .5 * candidate.loc[i, "target"] + .5 * np.sqrt(max(labels[fi, ti["eps"]] - ip, .05 ** 2))

name = "R2-C147-conditional-log-ionic-LOCAL_DIAGNOSTIC_ONLY"
path = OUT / f"{name}.csv"; candidate.to_csv(path, index=False)
report = {"experiment": name, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "separate EPS|Nc partner-conditioned log(EPS-Nc^2) Ridge(alpha=50), 5-fold target-excluded paired fit, fixed 0.5 blend, counterpart-observed only", "paired_rows": int(len(pair_rows)), "folds": folds, "metrics": metrics, "candidate_path": str(path), "elapsed_seconds": time.time() - started}
(OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("C147 candidate", path, len(candidate), flush=True)
