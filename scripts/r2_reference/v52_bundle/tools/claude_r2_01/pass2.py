"""Second pass: availability-conditioned physics routing for the five weak targets.

Stage 2/3 feed the physics reconstructions in as ordinary features, so one global
model has to serve both the ~60% of rows whose physics partner label is observed
(where the reconstruction is nearly exact) and the ~40% where the partner has to be
predicted (where it is noise).  Those two regimes want different weightings.

This pass fits, per target and SEPARATELY per availability regime, a tiny
non-negative blend between
    (a) the stage-3 prediction, and
    (b) the physics reconstruction from the observed partner label,
with at most two free parameters per regime.  Everything is fitted out of fold on
the training pool and applied to the test rows in the matching regime.

Reads the completed run's saved arrays; refits no heavy model.
"""
import os, sys, pickle, json
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{SCR}/out_full"
BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
T = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
ti = {t: j for j, t in enumerate(T)}
SEED = 20260804

F = pickle.load(open(f"{SCR}/features.pkl", "rb"))
idx, canon_map = F['idx'], F['canon_map']
NS = len(F['canon_list'])
train = pd.read_csv(f"{BASE}/train.csv"); test = pd.read_csv(f"{BASE}/test.csv")
arch = pd.read_csv(f"{BASE}/archive/train.csv")
for df in (train, test, arch):
    df['canon'] = df['smiles'].map(canon_map); df['fi'] = df['canon'].map(idx).astype(int)

def pivot(df):
    p = df.pivot_table(index='canon', columns='target_type', values='target', aggfunc='mean')
    for c in T:
        if c not in p.columns: p[c] = np.nan
    return p[T]
lab_r2, lab_ar = pivot(train), pivot(arch)
L = np.full((NS, len(T)), np.nan); A = np.full((NS, len(T)), np.nan)
for j, t in enumerate(T):
    for src in (lab_ar, lab_r2):
        v = src[t].dropna(); L[[idx[c] for c in v.index], j] = v.values
    v = lab_ar[t].dropna(); A[[idx[c] for c in v.index], j] = v.values
LOBS = np.isfinite(L)
pool = {t: np.where(LOBS[:, j])[0] for j, t in enumerate(T)}

cur = np.load(f"{OUT}/PFINAL.npy"); curD = np.load(f"{OUT}/PFINALD.npy")
BA = np.where(LOBS, L, cur)           # best available value per structure/property
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

# ---- physics reconstruction and its availability condition, per target ----
def recon(t):
    """(value array over all structures, boolean 'partner observed' mask)."""
    g = {c: BA[:, ti[c]] for c in T}
    chi, ionic, dgap = curD[:, 0], curD[:, 1], curD[:, 2]
    if t == 'ei':
        return g['eea'] + g['egc'], LOBS[:, ti['eea']]
    if t == 'eea':
        return g['ei'] - g['egc'], LOBS[:, ti['ei']]
    if t == 'egb':
        return g['egc'] + dgap, LOBS[:, ti['egc']]
    if t == 'egc':
        return g['egb'] - dgap, LOBS[:, ti['egb']]
    if t == 'eps':
        return g['nc'] ** 2 + ionic, LOBS[:, ti['nc']]
    if t == 'nc':
        return np.sqrt(np.clip(g['eps'] - ionic, 1.0, None)), LOBS[:, ti['eps']]
    return np.full(NS, np.nan), np.zeros(NS, bool)

new = cur.copy()
report = {}
print(f"{'target':6s} {'regime':10s} {'n_tr':>5s} {'n_te':>5s} "
      f"{'stack':>9s} {'physics':>9s} {'routed':>9s} {'gain':>9s}")
print("-" * 78)
for t in ['egb', 'ei', 'eea', 'nc', 'eps', 'egc']:
    j = ti[t]
    rows = pool[t]; y = L[rows, j]
    rec, obs = recon(t)
    base_all = r2(y, cur[rows, j])
    fitted = np.zeros(len(rows)); used = np.zeros(len(rows), bool)
    coefs = {}
    for regime, mask_all in [('partner-obs', obs), ('partner-miss', ~obs)]:
        sel = np.where(mask_all[rows])[0]
        te_fi = test.loc[test.target_type == t, 'fi'].values
        n_te = int(mask_all[te_fi].sum())
        if len(sel) < 40 or n_te == 0:
            print(f"{t:6s} {regime:10s} {len(sel):5d} {n_te:5d}   (skipped, too few)")
            continue
        ys = y[sel]
        Xs = np.column_stack([cur[rows[sel], j], rec[rows[sel]]])
        ok = np.isfinite(Xs).all(1)
        sel, ys, Xs = sel[ok], ys[ok], Xs[ok]
        # out-of-fold evaluation of the 2-parameter routed blend
        oof = np.zeros(len(sel))
        kf = KFold(n_splits=min(10, max(3, len(sel) // 20)), shuffle=True, random_state=SEED)
        for tr_, va_ in kf.split(Xs):
            m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(Xs[tr_], ys[tr_])
            oof[va_] = m.predict(Xs[va_])
        s_stack = r2(ys, Xs[:, 0]); s_phys = r2(ys, Xs[:, 1]); s_rout = r2(ys, oof)
        take = s_rout > s_stack + 1e-6
        print(f"{t:6s} {regime:10s} {len(sel):5d} {n_te:5d} "
              f"{s_stack:9.4f} {s_phys:9.4f} {s_rout:9.4f} {s_rout-s_stack:+9.4f}"
              f"{'' if take else '   (rejected)'}")
        if take:
            fitted[sel] = oof; used[sel] = True
            coefs[regime] = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(Xs, ys)
    if not used.any():
        report[t] = dict(before=base_all, after=base_all, applied=0)
        continue
    comb = cur[rows, j].copy(); comb[used] = fitted[used]
    after = r2(y, comb)
    if after <= base_all + 1e-6:
        print(f"  -> {t}: combined {after:.6f} <= base {base_all:.6f}, not applied")
        report[t] = dict(before=base_all, after=base_all, applied=0)
        continue
    print(f"  -> {t}: OOF {base_all:.6f} -> {after:.6f}  ({after-base_all:+.6f})")
    report[t] = dict(before=base_all, after=after, applied=int(used.sum()))
    # apply to every structure in the matching regime
    for regime, mask_all in [('partner-obs', obs), ('partner-miss', ~obs)]:
        if regime not in coefs: continue
        allsel = np.where(mask_all)[0]
        Xa = np.column_stack([cur[allsel, j], rec[allsel]])
        good = np.isfinite(Xa).all(1)
        new[allsel[good], j] = coefs[regime].predict(Xa[good])

np.save(f"{OUT}/PFINAL_pass2.npy", new)
json.dump(report, open(f"{OUT}/pass2_report.json", 'w'), indent=2)
print("\nper-target OOF after pass 2:")
tot = []
for j, t in enumerate(T):
    rows = pool[t]
    s = r2(L[rows, j], new[rows, j]); tot.append(s)
    print(f"  {t:4s} {s:.6f}")
print(f"  MEAN OOF {np.mean(tot):.6f}")
print(f"\nsaved {OUT}/PFINAL_pass2.npy")
