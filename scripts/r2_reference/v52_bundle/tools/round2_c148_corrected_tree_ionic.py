"""C148: corrected-parent tree model for the ionic EPS/Nc coordinate."""
from pathlib import Path
import json
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
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

def make_features():
    F = pickle.loads((SCR / "features.pkl").read_bytes()); P = pickle.loads((SCR / "physics.pkl").read_bytes()); G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    idx, cmap, blocks = F["idx"], F["canon_map"], F["blocks"]
    ns = len(F["canon_list"])
    PG = G["M"]; PGN = PG / np.maximum(1.0, PG.sum(1, keepdims=True))
    dense = np.hstack([clean(blocks["desc"]), clean(blocks["extra"]), clean(blocks["oligo"]), clean(blocks["ipc"]), clean(P["M"]), clean(PGN), clean(G["morph"])])
    keep = np.array([np.nanstd(dense[:, j]) > 1e-12 and np.isfinite(dense[:, j]).mean() > .3 for j in range(dense.shape[1])])
    dense = dense[:, keep]; med = np.nanmedian(dense, 0); miss = np.where(~np.isfinite(dense)); dense[miss] = np.take(med, miss[1])
    dense_qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED).fit_transform(dense)
    sparse = np.hstack([np.log1p(blocks["morgan"]), blocks["maccs"], np.log1p(blocks["ap"]), np.log1p(blocks["tt"]), np.log1p(blocks["rk"]), np.log1p(PG)]).astype(np.float32)
    sparse = sparse[:, (sparse != 0).sum(0) >= 4]
    svd = TruncatedSVD(192, random_state=SEED).fit_transform(sparse)
    treex = np.hstack([dense, svd, blocks["maccs"]]).astype(np.float32)
    # Structure-only representation used by corrected C141 fallbacks.
    p1 = np.load(SCR / "out_clean_corrected/P1.npy"); p1d = np.load(SCR / "out_clean_corrected/P1D.npy")
    cur = np.load(SCR / "out_clean_corrected/PFINAL.npy"); curd = np.load(SCR / "out_clean_corrected/PFINALD.npy")
    return F, idx, cmap, ns, treex, p1, p1d, cur, curd

F, idx, cmap, ns, TREEX, P1, P1D, CUR, CURD = make_features()
train = pd.read_csv(BASE / "train.csv"); test = pd.read_csv(BASE / "test.csv"); archive = pd.read_csv(BASE / "archive/train.csv")
for frame in (train, test, archive):
    frame["canon"] = frame["smiles"].map(cmap); frame["fi"] = frame["canon"].map(idx).astype(int)
L = np.full((ns, len(TARGETS)), np.nan)
for j, target in enumerate(TARGETS):
    for frame in (archive, train):
        vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
        for canon, value in vals.items(): L[idx[canon], j] = value
OBS = np.isfinite(L)

def cp_block(exclude):
    ex = set(exclude)
    ba = np.where(OBS, L, P1).astype(float)
    ob = OBS.astype(float).copy()
    for j in ex: ba[:, j] = CUR[:, j]; ob[:, j] = 0
    cols = []
    for j in range(len(TARGETS)):
        if j in ex: continue
        cols += [ba[:, j], ob[:, j], np.where(OBS[:, j], L[:, j], np.nan)]
    g = {t: ba[:, ti[t]] for t in TARGETS}
    chi, ion, dg = P1D[:, 0], P1D[:, 1], P1D[:, 2]
    cols += [g["eea"] + g["egc"], chi + g["egc"] / 2, g["ei"] - g["egc"], chi - g["egc"] / 2, g["ei"] - g["eea"], (g["egb"] + .9221) / 1.1178, g["egb"] - dg, 1.1178 * g["egc"] - .9221, g["egc"] + dg, g["nc"] ** 2 + .737, g["nc"] ** 2 + ion, np.sqrt(np.clip(g["eps"] - .652, 1, None)), np.sqrt(np.clip(g["eps"] - ion, 1, None)), chi, ion, dg, (g["ei"] + g["eea"]) / 2, (g["ei"] - g["eea"]) / 2, np.clip(g["egc"], .05, None) ** -.25, 1 / np.clip(g["egb"], .05, None), g["nc"] ** 2, np.sqrt(np.clip(g["eps"], 0, None)), ob.sum(1)]
    B = np.column_stack(cols)
    med = np.nanmedian(B, 0); med[~np.isfinite(med)] = 0
    bad = np.where(~np.isfinite(B)); B[bad] = np.take(med, bad[1])
    return B.astype(np.float32)

B = cp_block([ti["eps"], ti["nc"]])
X = np.hstack([TREEX, B, CURD[:, 1:2], P1D[:, 1:2]]).astype(np.float32)
pair = OBS[:, ti["eps"]] & OBS[:, ti["nc"]]
pair_rows = np.where(pair)[0]
ionic = L[pair_rows, ti["eps"]] - L[pair_rows, ti["nc"]] ** 2
if np.nanmin(ionic) <= 0: raise RuntimeError("non-positive ionic coordinate")
base = np.load(SCR / "out_clean_corrected/PFINAL.npy")
eps_rows = np.where(OBS[:, ti["eps"]])[0]; nc_rows = np.where(OBS[:, ti["nc"]])[0]
eps_pos = {r: i for i, r in enumerate(eps_rows)}; nc_pos = {r: i for i, r in enumerate(nc_rows)}

def model(kind):
    if kind == "et": return ExtraTreesRegressor(n_estimators=700, max_features=.5, min_samples_leaf=2, n_jobs=10, random_state=SEED)
    return lgb.LGBMRegressor(n_estimators=900, learning_rate=.035, num_leaves=31, min_child_samples=6, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=1., n_jobs=10, random_state=SEED, verbose=-1)

results = {}
for kind in ("et", "lgbm"):
    eps_oof = base[eps_rows, ti["eps"]].copy(); nc_oof = base[nc_rows, ti["nc"]].copy()
    folds = []
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + 148).split(pair_rows), 1):
        z = model(kind); z.fit(X[pair_rows[tr]], ionic[tr]); pred_ion = z.predict(X[pair_rows[va]])
        for r, ip in zip(pair_rows[va], pred_ion):
            eps_oof[eps_pos[r]] = L[r, ti["nc"]] ** 2 + ip
            nc_oof[nc_pos[r]] = np.sqrt(max(L[r, ti["eps"]] - ip, .05 ** 2))
        folds.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})
    results[kind] = {"eps": {"base_r2": r2(L[eps_rows, ti["eps"]], base[eps_rows, ti["eps"]]), "candidate_r2": r2(L[eps_rows, ti["eps"]], eps_oof)}, "nc": {"base_r2": r2(L[nc_rows, ti["nc"]], base[nc_rows, ti["nc"]]), "candidate_r2": r2(L[nc_rows, ti["nc"]], nc_oof)}, "folds": folds}
    for v in (results[kind]["eps"], results[kind]["nc"]): v["delta_r2"] = v["candidate_r2"] - v["base_r2"]
    print("C148", kind, json.dumps(results[kind], sort_keys=True), flush=True)

# Use the fixed LightGBM arm for the post-freeze diagnostic; this choice is
# preregistered from the old mechanism's model family, not chosen by local_eval.
z = model("lgbm"); z.fit(X[pair_rows], ionic); ion_all = z.predict(X)
base_path = OUT / "R2-C144-log-ionic-paired-reconstruction-LOCAL_DIAGNOSTIC_ONLY.csv"
candidate = pd.read_csv(base_path)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "eps" and OBS[fi, ti["nc"]]: candidate.loc[i, "target"] = L[fi, ti["nc"]] ** 2 + ion_all[fi]
    elif row.target_type == "nc" and OBS[fi, ti["eps"]]: candidate.loc[i, "target"] = np.sqrt(max(L[fi, ti["eps"]] - ion_all[fi], .05 ** 2))
name = "R2-C148-corrected-tree-ionic-LOCAL_DIAGNOSTIC_ONLY"
path = OUT / f"{name}.csv"; candidate.to_csv(path, index=False)
report = {"experiment": name, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "corrected C141 parent, LightGBM/ExtraTrees ionic=EPS-Nc^2 trees, target-excluded 5-fold paired audit, fixed LightGBM full fit, observed-counterpart routing only", "paired_rows": int(len(pair_rows)), "results": results, "candidate_path": str(path), "elapsed_seconds": time.time() - started}
(OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("C148 candidate", path, len(candidate), flush=True)

# Target-specific arm: the ExtraTrees EPS result is positive while both tree
# models are negative for Nc.  Preserve the C144 Nc carrier and change EPS only.
z_et = model("et"); z_et.fit(X[pair_rows], ionic); ion_et = z_et.predict(X)
eps_only = pd.read_csv(base_path)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "eps" and OBS[fi, ti["nc"]]:
        eps_only.loc[i, "target"] = L[fi, ti["nc"]] ** 2 + ion_et[fi]
name2 = "R2-C148-et-eps-only-LOCAL_DIAGNOSTIC_ONLY"
path2 = OUT / f"{name2}.csv"; eps_only.to_csv(path2, index=False)
report2 = {"experiment": name2, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "C148 corrected tree ionic ExtraTrees EPS carrier only; C144 retained for all other targets including Nc", "eps_oof": results["et"]["eps"], "candidate_path": str(path2), "elapsed_seconds": time.time() - started}
(OUT / f"{name2}-oof.json").write_text(json.dumps(report2, indent=2) + "\n", encoding="utf-8")
print("C148 EPS-only candidate", path2, len(eps_only), flush=True)

# A separate corrected-parent Ei direct tree arm, using the same target-excluded
# cross-property feature contract but keeping the observed-partner identity
# carrier for the both-partner slice.
B_ei = cp_block([ti["ei"]])
X_ei = np.hstack([TREEX, B_ei, CUR[:, ti["ei"]:ti["ei"] + 1], P1[:, ti["ei"]:ti["ei"] + 1]]).astype(np.float32)
ei_rows = np.where(OBS[:, ti["ei"]])[0]
both_ei = OBS[:, ti["egc"]] & OBS[:, ti["eea"]] & OBS[:, ti["ei"]]
identity_ei = L[:, ti["egc"]] + L[:, ti["eea"]]
ei_pos = {r: i for i, r in enumerate(ei_rows)}
ei_base = CUR[ei_rows, ti["ei"]].copy()
for r in ei_rows:
    if both_ei[r]: ei_base[ei_pos[r]] = .5 * CUR[r, ti["ei"]] + .5 * identity_ei[r]
ei_oof = ei_base.copy(); ei_folds = []
for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + 149).split(ei_rows), 1):
    z = ExtraTreesRegressor(n_estimators=700, max_features=1.0, min_samples_leaf=1, n_jobs=10, random_state=SEED + fold)
    z.fit(X_ei[ei_rows[tr]], L[ei_rows[tr], ti["ei"]]); p = z.predict(X_ei[ei_rows[va]])
    for r, pp in zip(ei_rows[va], p):
        if not both_ei[r]: ei_oof[ei_pos[r]] = .5 * ei_base[ei_pos[r]] + .5 * pp
    ei_folds.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})
ei_metric = {"base_r2": r2(L[ei_rows, ti["ei"]], ei_base), "candidate_r2": r2(L[ei_rows, ti["ei"]], ei_oof)}
ei_metric["delta_r2"] = ei_metric["candidate_r2"] - ei_metric["base_r2"]
print("C149 corrected Ei ET OOF", json.dumps(ei_metric, sort_keys=True), flush=True)
z = ExtraTreesRegressor(n_estimators=900, max_features=1.0, min_samples_leaf=1, n_jobs=10, random_state=SEED)
z.fit(X_ei[ei_rows], L[ei_rows, ti["ei"]]); ei_all = z.predict(X_ei)
ei_candidate = pd.read_csv(path2)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "ei" and not (OBS[fi, ti["egc"]] and OBS[fi, ti["eea"]]):
        ei_candidate.loc[i, "target"] = .5 * ei_candidate.loc[i, "target"] + .5 * ei_all[fi]
name3 = "R2-C149-corrected-ei-et-missing-partner-LOCAL_DIAGNOSTIC_ONLY"
path3 = OUT / f"{name3}.csv"; ei_candidate.to_csv(path3, index=False)
report3 = {"experiment": name3, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "C148 EPS-only carrier plus corrected-parent ExtraTrees Ei arm on missing-partner rows; both-partner observed identity retained", "ei_oof": ei_metric, "folds": ei_folds, "candidate_path": str(path3), "elapsed_seconds": time.time() - started}
(OUT / f"{name3}-oof.json").write_text(json.dumps(report3, indent=2) + "\n", encoding="utf-8")
print("C149 candidate", path3, len(ei_candidate), flush=True)

from xgboost import XGBRegressor
xgb_ei = XGBRegressor(n_estimators=650, max_depth=3, learning_rate=.035, min_child_weight=3, subsample=.82, colsample_bytree=.55, reg_alpha=.05, reg_lambda=4.0, objective="reg:squarederror", tree_method="hist", n_jobs=10, random_state=SEED, verbosity=0)
xgb_oof = ei_base.copy(); xgb_folds = []
for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + 151).split(ei_rows), 1):
    m = XGBRegressor(n_estimators=650, max_depth=3, learning_rate=.035, min_child_weight=3, subsample=.82, colsample_bytree=.55, reg_alpha=.05, reg_lambda=4.0, objective="reg:squarederror", tree_method="hist", n_jobs=10, random_state=SEED + fold, verbosity=0)
    m.fit(X_ei[ei_rows[tr]], L[ei_rows[tr], ti["ei"]]); p = m.predict(X_ei[ei_rows[va]])
    for r, pp in zip(ei_rows[va], p):
        if not both_ei[r]: xgb_oof[ei_pos[r]] = .5 * ei_base[ei_pos[r]] + .5 * pp
    xgb_folds.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})
xgb_metric = {"base_r2": r2(L[ei_rows, ti["ei"]], ei_base), "candidate_r2": r2(L[ei_rows, ti["ei"]], xgb_oof)}; xgb_metric["delta_r2"] = xgb_metric["candidate_r2"] - xgb_metric["base_r2"]
print("C151 corrected Ei XGB OOF", json.dumps(xgb_metric, sort_keys=True), flush=True)
xgb_ei.fit(X_ei[ei_rows], L[ei_rows, ti["ei"]]); xgb_all = xgb_ei.predict(X_ei)
xgb_candidate = pd.read_csv(path2)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "ei" and not (OBS[fi, ti["egc"]] and OBS[fi, ti["eea"]]): xgb_candidate.loc[i, "target"] = .5 * xgb_candidate.loc[i, "target"] + .5 * xgb_all[fi]
name4 = "R2-C151-corrected-ei-xgb-missing-partner-LOCAL_DIAGNOSTIC_ONLY"
path4 = OUT / f"{name4}.csv"; xgb_candidate.to_csv(path4, index=False)
report4 = {"experiment": name4, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "C148 EPS-only carrier plus corrected target-masked XGB Ei arm on missing-partner rows; both-partner identity retained", "ei_oof": xgb_metric, "folds": xgb_folds, "candidate_path": str(path4), "elapsed_seconds": time.time() - started}
(OUT / f"{name4}-oof.json").write_text(json.dumps(report4, indent=2) + "\n", encoding="utf-8")
print("C151 candidate", path4, len(xgb_candidate), flush=True)
