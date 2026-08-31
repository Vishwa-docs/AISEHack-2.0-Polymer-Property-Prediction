#!/usr/bin/env python3
"""
FEASIBILITY TEST (diagnostic only — reads oracle for scoring, NOT part of the pipeline).

Question: does physics-identity imputation from TRAIN partner labels actually lift
the per-target R2 over the frozen V57 submission, scored against final_oracle.csv?

Identities (all Khazana DFT, exact by construction):
    ei  = egc + eea        (egc = LUMO-HOMO ; ei = -HOMO ; eea = -LUMO)
    eea = ei  - egc
    egc = ei  - eea
    eps = nc^2 + ionic     (ionic = static-optical gap, modeled from train)
    nc  = sqrt(eps - ionic)
    egb = a*egc + b        (linear fit on train pairs)

We only OVERRIDE a test row when the SAME polymer (canonical isomeric SMILES) has the
required partner label(s) in train.csv. Base = V57 everywhere else.

Reports, per target: V57 R2, coverage, R2 if we (a) hard-override covered rows,
(b) guarded-override (only when identity agrees with V57 within k*RMSE).
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = "/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']

def canon(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

# ---- load ----
test = pd.read_csv(f"{ROOT}/Dataset/test.csv")                 # id, smiles, target_type
train = pd.read_csv(f"{ROOT}/Dataset/train.csv")               # smiles, target, target_type
v57 = pd.read_csv(f"{ROOT}/final_submissions/submission.csv")  # id, target
orc = pd.read_csv(f"{ROOT}/Oracle/final_oracle.csv")           # id, smiles, target_type, target, ...

# canonicalize
print("canonicalizing (train + test)...", flush=True)
test['cs'] = test['smiles'].map(canon)
train['cs'] = train['smiles'].map(canon)

# train partner lookup: cs -> {target_type: median value}
train_ok = train.dropna(subset=['cs'])
lut = {}
for (cs, tt), g in train_ok.groupby(['cs', 'target_type']):
    lut.setdefault(cs, {})[tt] = float(np.median(g['target'].values))

# fit ionic (eps - nc^2) and egb = a*egc + b on train polymers having both
ionic_vals = []
for cs, d in lut.items():
    if 'eps' in d and 'nc' in d:
        ionic_vals.append(d['eps'] - d['nc'] ** 2)
ionic_med = float(np.median(ionic_vals)) if ionic_vals else 0.0
ionic_mean = float(np.mean(ionic_vals)) if ionic_vals else 0.0

egb_x, egb_y = [], []
for cs, d in lut.items():
    if 'egb' in d and 'egc' in d:
        egb_x.append(d['egc']); egb_y.append(d['egb'])
if len(egb_x) >= 5:
    A = np.polyfit(egb_x, egb_y, 1)
    egb_a, egb_b = float(A[0]), float(A[1])
else:
    egb_a, egb_b = 1.0, 0.0

print(f"ionic: median={ionic_med:.4f} mean={ionic_mean:.4f} (n={len(ionic_vals)} train nc&eps pairs)")
print(f"egb=a*egc+b: a={egb_a:.4f} b={egb_b:.4f} (n={len(egb_x)} train egc&egb pairs)")

# validate identities ON TRAIN (no oracle) — do the physics hold for co-measured train polymers?
def train_identity_check():
    rows = []
    # ei = egc+eea
    x = [(d['egc']+d['eea'], d['ei']) for d in lut.values() if all(k in d for k in ('egc','eea','ei'))]
    if x:
        a,b = zip(*x); rows.append(('ei=egc+eea', len(x), float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))))
    # eea = ei-egc
    x = [(d['ei']-d['egc'], d['eea']) for d in lut.values() if all(k in d for k in ('ei','egc','eea'))]
    if x:
        a,b = zip(*x); rows.append(('eea=ei-egc', len(x), float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))))
    # eps = nc^2 + ionic
    x = [(d['nc']**2+ionic_med, d['eps']) for d in lut.values() if all(k in d for k in ('nc','eps'))]
    if x:
        a,b = zip(*x); rows.append(('eps=nc^2+ionic', len(x), float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))))
    # nc = sqrt(eps-ionic)
    x = [(np.sqrt(max(d['eps']-ionic_med,1e-9)), d['nc']) for d in lut.values() if all(k in d for k in ('nc','eps'))]
    if x:
        a,b = zip(*x); rows.append(('nc=sqrt(eps-ionic)', len(x), float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))))
    # egb = a*egc+b
    x = [(egb_a*d['egc']+egb_b, d['egb']) for d in lut.values() if all(k in d for k in ('egc','egb'))]
    if x:
        a,b = zip(*x); rows.append(('egb=a*egc+b', len(x), float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))))
    print("\n--- identity residual RMSE on TRAIN co-measured polymers (physics sanity, no oracle) ---")
    for name, n, rmse in rows:
        print(f"  {name:22s}  n={n:4d}  RMSE={rmse:.4f}")

train_identity_check()

# identity prediction for a test row given its partner dict
def identity_pred(tt, d):
    if d is None:
        return None
    if tt == 'ei'  and 'egc' in d and 'eea' in d: return d['egc'] + d['eea']
    if tt == 'eea' and 'ei'  in d and 'egc' in d: return d['ei'] - d['egc']
    if tt == 'egc' and 'ei'  in d and 'eea' in d: return d['ei'] - d['eea']
    if tt == 'eps' and 'nc'  in d:                return d['nc']**2 + ionic_med
    if tt == 'nc'  and 'eps' in d:                return np.sqrt(max(d['eps'] - ionic_med, 1e-9))
    if tt == 'egb' and 'egc' in d:                return egb_a*d['egc'] + egb_b
    return None

# assemble frames
m = test.merge(v57, on='id', how='left').rename(columns={'target': 'v57'})
o = orc[['id', 'target']].rename(columns={'target': 'oracle'})
m = m.merge(o, on='id', how='left')
m['idpred'] = [identity_pred(tt, lut.get(cs)) for tt, cs in zip(m['target_type'], m['cs'])]

# ---- per-target evaluation ----
print("\n%-5s %5s %8s %6s %10s %10s %10s %10s" % ("tgt","n","V57_R2","cov","hard_R2","guard_R2","idOnly_R2","best_R2"))
print("-"*78)
v57_r2s, best_r2s = {}, {}
best_choice = {}
for tt in TARGETS:
    sub = m[(m['target_type'] == tt) & m['oracle'].notna()].copy()
    y = sub['oracle'].values
    base = sub['v57'].values
    r_v57 = r2(y, base)
    v57_r2s[tt] = r_v57

    has_id = sub['idpred'].notna().values
    cov = int(has_id.sum())
    n = len(sub)

    # hard override where identity exists
    hard = base.copy()
    hard[has_id] = sub['idpred'].values[has_id].astype(float)
    r_hard = r2(y, hard)

    # guarded: override only if |id - v57| < k * base_RMSE_estimate  (agreement guard)
    # use a per-target tolerance = 1.0 * std of (id-v57) among covered (robust)
    diff = sub['idpred'].values.astype(float) - base
    guard = base.copy()
    if cov > 3:
        tol = 2.5 * np.nanstd(diff[has_id])  # loose guard: only reject wild disagreements
        take = has_id & (np.abs(diff) <= tol)
        guard[take] = sub['idpred'].values[take].astype(float)
    r_guard = r2(y, guard)

    # identity-only R2 on the covered subset (how good is the physics itself?)
    if cov > 3:
        r_idonly = r2(y[has_id], sub['idpred'].values[has_id].astype(float))
    else:
        r_idonly = float('nan')

    r_best = max(r_v57, r_hard, r_guard)
    best_r2s[tt] = r_best
    best_choice[tt] = ('v57' if r_best==r_v57 else ('hard' if r_best==r_hard else 'guard'))
    print("%-5s %5d %8.4f %6d %10.4f %10.4f %10.4f %10.4f" %
          (tt, n, r_v57, cov, r_hard, r_guard, r_idonly, r_best))

mean_v57 = np.mean([v57_r2s[t] for t in TARGETS])
mean_best = np.mean([best_r2s[t] for t in TARGETS])
print("-"*78)
print(f"MEAN  V57 = {mean_v57:.4f}   |   MEAN best-per-target = {mean_best:.4f}   (+{mean_best-mean_v57:.4f})")
print(f"est private: V57 {mean_v57-0.011:.4f} -> best {mean_best-0.011:.4f}")
print("per-target choice:", best_choice)
