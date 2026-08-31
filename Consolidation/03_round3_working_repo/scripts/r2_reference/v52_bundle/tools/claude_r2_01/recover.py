"""Rebuild the submission and the estimate from saved stage predictions.

Insurance against a failure in the cheap tail of the pipeline (constraints,
override, submission write) after the expensive model fitting has already run.
Reads PFINAL.npy / P1.npy from the run's output directory.
"""
import os, sys, pickle, json
import numpy as np, pandas as pd

SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{SCR}/out_full"
BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
T = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
ti = {t: j for j, t in enumerate(T)}

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

cur = np.load(f"{OUT}/PFINAL.npy")
print("loaded", f"{OUT}/PFINAL.npy", cur.shape)

def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

rows_out = []
for j, t in enumerate(T):
    te = test[test.target_type == t]
    fi = te['fi'].values
    p = cur[fi, j].copy()
    bad = ~np.isfinite(p)
    if bad.any():
        p[bad] = np.nanmean(L[:, j]); print(f"  {t}: filled {bad.sum()} non-finite")
    if t in ('tg', 'egc'):
        av = A[fi, j]; hit = np.isfinite(av); p[hit] = av[hit]
        print(f"  {t}: archive override {hit.sum()}/{len(fi)} ({100*hit.mean():.1f}%)")
    rows_out.append(pd.DataFrame({'id': te['id'].values, 'target': p,
                                  'target_type': t, 'fi': fi}))
sub = pd.concat(rows_out, ignore_index=True)

nfi = {f: i for f, i in zip(sub.loc[sub.target_type == 'nc', 'fi'],
                            sub.index[sub.target_type == 'nc'])}
nviol = 0
for f_, ei_ in zip(sub.loc[sub.target_type == 'eps', 'fi'],
                   sub.index[sub.target_type == 'eps']):
    nc_v = (L[f_, ti['nc']] if LOBS[f_, ti['nc']]
            else (sub.at[nfi[f_], 'target'] if f_ in nfi else cur[f_, ti['nc']]))
    if sub.at[ei_, 'target'] < nc_v ** 2 + 0.024:
        sub.at[ei_, 'target'] = nc_v ** 2 + 0.024; nviol += 1
print(f"  EPS>=Nc^2 projection: {nviol} rows")

for j, t in enumerate(T):
    lo, hi = np.nanmin(L[:, j]), np.nanmax(L[:, j]); span = hi - lo
    m = sub.target_type == t
    sub.loc[m, 'target'] = sub.loc[m, 'target'].clip(lo - 0.02 * span, hi + 0.02 * span)

sub = sub.sort_values('id')
assert len(sub) == 4940 and sub['id'].is_unique and np.isfinite(sub['target']).all()
sub[['id', 'target']].to_csv(f"{OUT}/submission.csv", index=False)
print("wrote", f"{OUT}/submission.csv")

C050_OOF = dict(tg=.9088768072, egc=.9115043879, egb=.9221467344, ei=.8454440895,
                eea=.9008357940, nc=.8397322432, eps=.7835054390)
C050_TEST = dict(tg=.9539044922, egc=.9572538004, egb=.8990616896, ei=.7568828960,
                 eea=.8681479551, nc=.8295703105, eps=.7698645559)
est, cal = {}, {}
print("\n=== estimate ===")
for j, t in enumerate(T):
    rows = pool[t]; y = L[rows, j]; p = cur[rows, j]
    oof = r2(y, p)
    if t in ('tg', 'egc'):
        fi = test.loc[test.target_type == t, 'fi'].values
        cov = np.isfinite(A[fi, j]).mean()
        unc = ~np.isfinite(A[rows, j])
        mse = ((y[unc] - p[unc]) ** 2).mean() if unc.sum() > 30 else ((y - p) ** 2).mean()
        est[t] = 1 - (1 - cov) * mse / np.var(y); cal[t] = est[t]
    else:
        est[t] = oof; cal[t] = oof - (C050_OOF[t] - C050_TEST[t])
    print(f"  {t:4s} OOF={oof:.6f}  est={est[t]:.6f}  cal={cal[t]:.6f}  "
          f"vs C050 test {C050_TEST[t]:.4f}  delta {est[t]-C050_TEST[t]:+.4f}")
print(f"  MEAN est={np.mean(list(est.values())):.6f}  cal={np.mean(list(cal.values())):.6f}"
      f"   (C050 test 0.8621)")
json.dump(dict(per_target_est=est, mean=float(np.mean(list(est.values()))),
               per_target_gap_calibrated=cal,
               mean_gap_calibrated=float(np.mean(list(cal.values())))),
          open(f"{OUT}/estimate.json", 'w'), indent=2)
