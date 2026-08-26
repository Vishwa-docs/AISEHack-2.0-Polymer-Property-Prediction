"""C144: target-excluded log ionic-coordinate reconstruction for EPS/Nc."""
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
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
started = time.time()

def r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def clean(x):
    x = np.asarray(x, np.float64)
    x[~np.isfinite(x)] = np.nan
    return np.clip(x, -1e10, 1e10)

def scaled_ridge_fit(x, y, alpha=50.0):
    mu = x.mean(0, dtype=np.float64)
    sd = x.std(0, dtype=np.float64)
    sd[sd < 1e-12] = 1.0
    model = Ridge(alpha=alpha).fit((x - mu) / sd, y)
    return model, mu, sd

def predict(fit, x):
    model, mu, sd = fit
    return model.predict((x - mu) / sd)

print("C144 loading official structures/features", flush=True)
F = pickle.loads((SCR / "features.pkl").read_bytes())
P = pickle.loads((SCR / "physics.pkl").read_bytes())
G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
idx, canon_map, blocks = F["idx"], F["canon_map"], F["blocks"]
ns = len(F["canon_list"])
train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")
archive = pd.read_csv(BASE / "archive/train.csv")
for frame in (train, test, archive):
    frame["canon"] = frame["smiles"].map(canon_map)
    frame["fi"] = frame["canon"].map(idx).astype(int)

labels = np.full((ns, len(TARGETS)), np.nan)
for j, target in enumerate(TARGETS):
    for frame in (archive, train):
        vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
        for canon, value in vals.items():
            labels[idx[canon], j] = value
observed = np.isfinite(labels)

ionic = np.full(ns, np.nan)
pair = observed[:, ti["eps"]] & observed[:, ti["nc"]]
ionic[pair] = labels[pair, ti["eps"]] - labels[pair, ti["nc"]] ** 2
if np.nanmin(ionic) <= 0:
    raise RuntimeError("ionic coordinate is not strictly positive; log transform invalid")
pair_rows = np.where(np.isfinite(ionic))[0]
print("C144 paired ionic rows", len(pair_rows), "range", float(np.nanmin(ionic)), float(np.nanmax(ionic)), flush=True)

# Match the Stage-1 structure representation without using target labels.
PG = G["M"]
PGN = PG / np.maximum(1.0, PG.sum(1, keepdims=True))
dense = np.hstack([
    clean(blocks["desc"]), clean(blocks["extra"]), clean(blocks["oligo"]),
    clean(blocks["ipc"]), clean(P["M"]), clean(PGN), clean(G["morph"]),
])
keep = np.array([
    np.nanstd(dense[:, j]) > 1e-12 and np.isfinite(dense[:, j]).mean() > 0.3
    for j in range(dense.shape[1])
])
dense = dense[:, keep]
med = np.nanmedian(dense, 0)
missing = np.where(~np.isfinite(dense))
dense[missing] = np.take(med, missing[1])
dense_qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED).fit_transform(dense)
sparse = np.hstack([
    np.log1p(blocks["morgan"]), blocks["maccs"], np.log1p(blocks["ap"]),
    np.log1p(blocks["tt"]), np.log1p(blocks["rk"]), np.log1p(PG),
]).astype(np.float32)
sparse = sparse[:, (sparse != 0).sum(0) >= 4]
svd = TruncatedSVD(192, random_state=SEED).fit_transform(sparse)
X = np.hstack([dense_qt, svd, blocks["maccs"]]).astype(np.float64)
print("C144 structure feature matrix", X.shape, flush=True)

base = np.load(SCR / "out_clean_corrected/PFINAL.npy")
eps_rows = np.where(observed[:, ti["eps"]])[0]
nc_rows = np.where(observed[:, ti["nc"]])[0]
oof = {"eps": base[eps_rows, ti["eps"]].copy(), "nc": base[nc_rows, ti["nc"]].copy()}
fold_meta = []
eps_pos = {r: i for i, r in enumerate(eps_rows)}
nc_pos = {r: i for i, r in enumerate(nc_rows)}
kf = KFold(5, shuffle=True, random_state=SEED + 144)

for fold, (tr_pair_pos, va_pair_pos) in enumerate(kf.split(pair_rows), 1):
    # Exclude every validation structure from the ionic-coordinate fit.
    excluded = set(pair_rows[va_pair_pos].tolist())
    tr_pair = np.array([r for r in pair_rows if r not in excluded], dtype=int)
    fit = scaled_ridge_fit(X[tr_pair], np.log(np.clip(ionic[tr_pair], 1e-6, None)))
    va_eps = eps_rows[np.isin(eps_rows, pair_rows[va_pair_pos])]
    va_nc = nc_rows[np.isin(nc_rows, pair_rows[va_pair_pos])]
    pred_eps = np.exp(np.clip(predict(fit, X[va_eps]), -8, 4)) if len(va_eps) else np.array([])
    pred_nc = np.exp(np.clip(predict(fit, X[va_nc]), -8, 4)) if len(va_nc) else np.array([])
    for r, ip in zip(va_eps, pred_eps):
        if observed[r, ti["nc"]]:
            raw = labels[r, ti["nc"]] ** 2 + ip
            oof["eps"][eps_pos[r]] = 0.5 * base[r, ti["eps"]] + 0.5 * raw
    for r, ip in zip(va_nc, pred_nc):
        if observed[r, ti["eps"]]:
            raw = np.sqrt(max(labels[r, ti["eps"]] - ip, 0.05 ** 2))
            oof["nc"][nc_pos[r]] = 0.5 * base[r, ti["nc"]] + 0.5 * raw
    fold_meta.append({"fold": fold, "fit_rows": int(len(tr_pair)), "excluded_rows": int(len(excluded)), "eps_routed": int(len(va_eps)), "nc_routed": int(len(va_nc))})

eps_y = labels[eps_rows, ti["eps"]]
nc_y = labels[nc_rows, ti["nc"]]
eps_base = base[eps_rows, ti["eps"]]
nc_base = base[nc_rows, ti["nc"]]
metrics = {
    "eps": {"base_r2": r2(eps_y, eps_base), "candidate_r2": r2(eps_y, oof["eps"])},
    "nc": {"base_r2": r2(nc_y, nc_base), "candidate_r2": r2(nc_y, oof["nc"])},
}
for target in metrics:
    metrics[target]["delta_r2"] = metrics[target]["candidate_r2"] - metrics[target]["base_r2"]
print("C144 OOF", json.dumps(metrics, sort_keys=True), flush=True)

# Full-data fit for a post-freeze diagnostic only.  Missing counterparts retain
# the corrected C143 carrier; no external_label/local_eval value is read by this script.
full_fit = scaled_ridge_fit(X[pair_rows], np.log(np.clip(ionic[pair_rows], 1e-6, None)))
test_fi = test.fi.to_numpy()
test_ionic = np.exp(np.clip(predict(full_fit, X[test_fi]), -8, 4))
base_candidate_path = OUT / "R2-C143-ei-both-partners-identity-half-LOCAL_DIAGNOSTIC_ONLY.csv"
candidate = pd.read_csv(base_candidate_path)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "eps" and observed[fi, ti["nc"]]:
        raw = labels[fi, ti["nc"]] ** 2 + test_ionic[i]
        candidate.loc[i, "target"] = 0.5 * candidate.loc[i, "target"] + 0.5 * raw
    elif row.target_type == "nc" and observed[fi, ti["eps"]]:
        raw = np.sqrt(max(labels[fi, ti["eps"]] - test_ionic[i], 0.05 ** 2))
        candidate.loc[i, "target"] = 0.5 * candidate.loc[i, "target"] + 0.5 * raw

name = "R2-C144-log-ionic-paired-reconstruction-LOCAL_DIAGNOSTIC_ONLY"
candidate_path = OUT / f"{name}.csv"
candidate.to_csv(candidate_path, index=False)
report = {
    "experiment": name,
    "official_only_fitting": True,
    "local_eval_read": False,
    "pretrained_weights": False,
    "mechanism": "log(EPS-Nc^2) Ridge(alpha=50), 5-fold target-excluded paired-label audit, fixed 0.5 reconstruction blend, counterpart-observed routing only",
    "paired_rows": int(len(pair_rows)),
    "folds": fold_meta,
    "metrics": metrics,
    "candidate_path": str(candidate_path),
    "elapsed_seconds": time.time() - started,
}
(OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("C144 candidate", candidate_path, "rows", len(candidate), flush=True)
print(json.dumps(report, sort_keys=True), flush=True)
