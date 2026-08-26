"""C150: corrected-parent dummy-capped charge residual for Ei."""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdPartialCharges
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
OUT = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY"
SEED = 20260804
T = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(T)}
started = time.time()

def charge_features(smiles):
    out = []
    for text in smiles:
        m = Chem.MolFromSmiles(text)
        row = np.full(31, np.nan)
        try:
            atoms = list(m.GetAtoms()); bonds = list(m.GetBonds()); heavy = max(m.GetNumHeavyAtoms(), 1); nb = max(len(bonds), 1)
            hetero = sum(a.GetAtomicNum() not in (0, 1, 6) for a in atoms)
            hba = float(rdMolDescriptors.CalcNumHBA(m)); hbd = float(rdMolDescriptors.CalcNumHBD(m)); tpsa = float(rdMolDescriptors.CalcTPSA(m)); mr = float(Crippen.MolMR(m)); logp = float(Crippen.MolLogP(m))
            arom = sum(a.GetIsAromatic() for a in atoms); rings = max(m.GetRingInfo().NumRings(), 1); rot = float(Descriptors.NumRotatableBonds(m)); db = sum(b.GetBondTypeAsDouble() == 2 for b in bonds); conj = sum(b.GetIsAromatic() or b.GetBondTypeAsDouble() > 1 for b in bonds)
            hal = sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in atoms); n = sum(a.GetAtomicNum() == 7 for a in atoms); o = sum(a.GetAtomicNum() == 8 for a in atoms); s = sum(a.GetAtomicNum() == 16 for a in atoms); q = float(sum(a.GetFormalCharge() for a in atoms))
            row[:21] = [hetero/heavy, (hetero+hba+hbd)/heavy, tpsa/heavy, mr/heavy, logp/heavy, hba/heavy, hbd/heavy, arom/heavy, float(rdMolDescriptors.CalcNumAromaticRings(m))/rings, rot/heavy, m.GetRingInfo().NumRings()/heavy, db/nb, conj/nb, hal/heavy, n/heavy, o/heavy, s/heavy, q, abs(q), abs(q)/heavy, np.log1p(heavy)]
            rw = Chem.RWMol(Chem.Mol(m))
            for a in rw.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetAtomicNum(6); a.SetFormalCharge(0); a.SetNoImplicit(False)
            cm = rw.GetMol(); Chem.SanitizeMol(cm); rdPartialCharges.ComputeGasteigerCharges(cm)
            charges = np.asarray([float(a.GetProp("_GasteigerCharge")) for a in cm.GetAtoms()])
            if np.isfinite(charges).all():
                dist = np.asarray(Chem.GetDistanceMatrix(cm, useBO=False), float); aa = list(cm.GetAtoms()); het = np.asarray([a.GetAtomicNum() not in (0,1,6) for a in aa], bool); w = np.abs(charges[:,None] * charges[None,:]); wt = max(float(w.sum()), 1e-12); wh = w * (het[:,None] | het[None,:]); wht = max(float(wh.sum()), 1e-12)
                row[21:] = [charges.mean(), charges.std(), charges.min(), charges.max(), np.ptp(charges), np.abs(charges).mean(), np.abs(charges).sum()/heavy, np.abs(charges[het]).mean() if het.any() else 0., float((w*dist).sum()/wt), float((wh*dist).sum()/wht)]
        except Exception:
            pass
        out.append(row)
    return np.asarray(out, float)

F = pickle.loads((SCR / "features.pkl").read_bytes())
canon = np.asarray(F["canon_list"], dtype=object); cmap = F["canon_map"]; idx = F["idx"]; ns = len(canon)
train = pd.read_csv(BASE / "train.csv"); test = pd.read_csv(BASE / "test.csv"); archive = pd.read_csv(BASE / "archive/train.csv")
for frame in (train, test, archive): frame["canon"] = frame["smiles"].map(cmap); frame["fi"] = frame["canon"].map(idx).astype(int)
L = np.full((ns, len(T)), np.nan)
for j, t in enumerate(T):
    for frame in (archive, train):
        vals = frame.loc[frame.target_type.eq(t)].groupby("canon")["target"].mean()
        for c, v in vals.items(): L[idx[c], j] = v
obs = np.isfinite(L)
ei_rows = np.where(obs[:, ti["ei"]])[0]
X = charge_features(canon)
base = np.load(SCR / "out_clean_corrected/PFINAL.npy")
both = obs[:, ti["egc"]] & obs[:, ti["eea"]] & obs[:, ti["ei"]]
identity = L[:, ti["egc"]] + L[:, ti["eea"]]
base_ei = base[ei_rows, ti["ei"]].copy()
for pos, r in enumerate(ei_rows):
    if both[r]: base_ei[pos] = .5 * base[r, ti["ei"]] + .5 * identity[r]
y = L[ei_rows, ti["ei"]]
ei_pos = {r: i for i, r in enumerate(ei_rows)}
groups = np.asarray([str(F["scaffolds"][r]) for r in ei_rows], dtype=object)
oof = np.zeros(len(ei_rows)); folds = []
for fold, (tr, va) in enumerate(GroupKFold(5).split(ei_rows, y, groups), 1):
    residual = y[tr] - base_ei[tr]
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0))
    m.fit(X[ei_rows[tr]], residual)
    oof[va] = base_ei[va] + .20 * m.predict(X[ei_rows[va]])
    folds.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})
metrics = {"base_r2": float(r2_score(y, base_ei)), "candidate_r2": float(r2_score(y, oof))}; metrics["delta_r2"] = metrics["candidate_r2"] - metrics["base_r2"]
print("C150 Ei OOF", json.dumps(metrics, sort_keys=True), flush=True)

m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0)); m.fit(X[ei_rows], y - base_ei)
base_path = OUT / "R2-C148-et-eps-only-LOCAL_DIAGNOSTIC_ONLY.csv"
candidate = pd.read_csv(base_path)
test_fi = test.fi.to_numpy(); test_res = m.predict(X[test_fi])
for i, row in test.iterrows():
    if row.target_type == "ei": candidate.loc[i, "target"] = candidate.loc[i, "target"] + .20 * test_res[i]
name = "R2-C150-ei-charge-residual-corrected-LOCAL_DIAGNOSTIC_ONLY"
path = OUT / f"{name}.csv"; candidate.to_csv(path, index=False)
report = {"experiment": name, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "dummy-capped Gasteiger/charge-distance plus normalized RDKit physics Ridge residual, GroupKFold(5) by scaffold, fixed residual weight 0.20, C143-style both-partner Ei carrier and C148 EPS carrier", "metrics": metrics, "folds": folds, "candidate_path": str(path), "elapsed_seconds": time.time() - started}
(OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("C150 candidate", path, len(candidate), flush=True)

from sklearn.ensemble import HistGradientBoostingRegressor
hgb_oof = base_ei.copy(); hgb_folds = []
for fold, (tr, va) in enumerate(GroupKFold(5).split(ei_rows, y, groups), 1):
    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=SEED + fold)
    h.fit(X[ei_rows[tr]], y[tr]); p = h.predict(X[ei_rows[va]])
    for r, pp in zip(ei_rows[va], p):
        if not both[r]: hgb_oof[ei_pos[r]] = .5 * base_ei[ei_pos[r]] + .5 * pp
    hgb_folds.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})
hgb_metric = {"base_r2": float(r2_score(y, base_ei)), "candidate_r2": float(r2_score(y, hgb_oof))}; hgb_metric["delta_r2"] = hgb_metric["candidate_r2"] - hgb_metric["base_r2"]
print("C152 Ei HGB OOF", json.dumps(hgb_metric, sort_keys=True), flush=True)
h = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=SEED); h.fit(X[ei_rows], y); h_all = h.predict(X)
hgb_candidate = pd.read_csv(base_path)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "ei" and not (obs[fi, ti["egc"]] and obs[fi, ti["eea"]]): hgb_candidate.loc[i, "target"] = .5 * hgb_candidate.loc[i, "target"] + .5 * h_all[fi]
name2 = "R2-C152-ei-hgb-missing-partner-LOCAL_DIAGNOSTIC_ONLY"
path2 = OUT / f"{name2}.csv"; hgb_candidate.to_csv(path2, index=False)
report2 = {"experiment": name2, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "C148 EPS-only carrier plus fixed HGB Ei physical/charge arm on missing-partner rows; both-partner identity retained", "ei_oof": hgb_metric, "folds": hgb_folds, "candidate_path": str(path2), "elapsed_seconds": time.time() - started}
(OUT / f"{name2}-oof.json").write_text(json.dumps(report2, indent=2) + "\n", encoding="utf-8")
print("C152 candidate", path2, len(hgb_candidate), flush=True)

# C153: fixed transductive support subset. Test structures enter only as
# unlabeled covariates for the domain classifier; target values remain hidden.
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
test_ei_rows = test.loc[test.target_type.eq("ei"), "fi"].to_numpy(dtype=int)
dom_rows = np.concatenate([ei_rows, test_ei_rows])
dom_y = np.concatenate([np.zeros(len(ei_rows)), np.ones(len(test_ei_rows))])
dom = LGBMClassifier(n_estimators=250, num_leaves=15, learning_rate=.04, min_child_samples=10, colsample_bytree=.6, reg_lambda=2., n_jobs=10, random_state=SEED, verbosity=-1)
dom.fit(X[dom_rows], dom_y)
ptest = np.clip(dom.predict_proba(X[ei_rows])[:, 1], .03, .97)
threshold = float(np.quantile(ptest, .30))
selected = ptest >= threshold
support_oof = base_ei.copy(); support_folds = []
for fold, (tr, va) in enumerate(GroupKFold(5).split(ei_rows, y, groups), 1):
    keep_tr = tr[selected[tr]]
    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=SEED + 300 + fold)
    h.fit(X[ei_rows[keep_tr]], y[keep_tr]); p = h.predict(X[ei_rows[va]])
    for local, r in enumerate(ei_rows[va]):
        if not both[r]: support_oof[ei_pos[r]] = .5 * base_ei[ei_pos[r]] + .5 * p[local]
    support_folds.append({"fold": fold, "fit_rows": int(len(keep_tr)), "validation_rows": int(len(va))})
support_metric = {"base_r2": float(r2_score(y, base_ei)), "candidate_r2": float(r2_score(y, support_oof)), "delta_r2": float(r2_score(y, support_oof) - r2_score(y, base_ei)), "domain_auc": float(roc_auc_score(dom_y, dom.predict_proba(X[dom_rows])[:, 1])), "support_fraction": float(selected.mean()), "support_threshold": threshold}
print("C153 support-subset Ei OOF", json.dumps(support_metric, sort_keys=True), flush=True)
keep_all = np.flatnonzero(selected)
h = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=SEED + 300)
h.fit(X[ei_rows[keep_all]], y[keep_all]); support_all = h.predict(X)
support_candidate = pd.read_csv(base_path)
for i, row in test.iterrows():
    fi = int(row.fi)
    if row.target_type == "ei" and not (obs[fi, ti["egc"]] and obs[fi, ti["eea"]]): support_candidate.loc[i, "target"] = .5 * support_candidate.loc[i, "target"] + .5 * support_all[fi]
name3 = "R2-C153-ei-test-support-subset-hgb-LOCAL_DIAGNOSTIC_ONLY"
path3 = OUT / f"{name3}.csv"; support_candidate.to_csv(path3, index=False)
report3 = {"experiment": name3, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "unlabeled-test domain classifier, fixed top-70-percent Ei support subset, HGB absolute head on missing-partner rows, GroupKFold(5), both-partner identity retained", "metrics": support_metric, "folds": support_folds, "candidate_path": str(path3), "elapsed_seconds": time.time() - started}
(OUT / f"{name3}-oof.json").write_text(json.dumps(report3, indent=2) + "\n", encoding="utf-8")
print("C153 candidate", path3, len(support_candidate), flush=True)
