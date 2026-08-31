"""Weak-targets-only run: egb, ei, eea, nc, eps.

Tg and Egc are already at/above the best numbers ever measured in this project and
their test scores are dominated by the official archive exact override, so here they
are fitted cheaply and only used as cross-property covariates.  All the compute goes
into the five targets that actually decide whether the mean reaches 0.93.

Pipeline per weak target:
  1. structure-only zoo
  2. cross-property stack (observed partner labels + physics reconstructions)
  3. one more round, propagating between test rows that partner each other
  4. availability-conditioned routing: separate blends for rows whose physics
     partner label is observed vs missing
"""
import os, sys, pickle, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.optimize import nnls
import lightgbm as lgb

SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
OUT = f"{SCR}/out_weak"; os.makedirs(OUT, exist_ok=True)
T = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
ti = {t: j for j, t in enumerate(T)}
WEAK = ['egb', 'ei', 'eea', 'nc', 'eps']
DERIVED = ['chi', 'ionic', 'dgap']
SEED = 20260804; NJOBS = int(os.environ.get("NJOBS", "10"))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

F = pickle.load(open(f"{SCR}/features.pkl", "rb"))
P = pickle.load(open(f"{SCR}/physics.pkl", "rb"))
G = pickle.load(open(f"{SCR}/pgfp.pkl", "rb"))
idx, canon_map, blocks = F['idx'], F['canon_map'], F['blocks']
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
L = np.full((NS, len(T)), np.nan)
for j, t in enumerate(T):
    for src in (lab_ar, lab_r2):
        v = src[t].dropna(); L[[idx[c] for c in v.index], j] = v.values
LOBS = np.isfinite(L)
pool = {t: np.where(LOBS[:, j])[0] for j, t in enumerate(T)}
log("pools:", {t: len(pool[t]) for t in T})

D = np.full((NS, 3), np.nan)
m = LOBS[:, ti['ei']] & LOBS[:, ti['eea']]; D[m, 0] = (L[m, ti['ei']] + L[m, ti['eea']]) / 2
m = LOBS[:, ti['eps']] & LOBS[:, ti['nc']]; D[m, 1] = L[m, ti['eps']] - L[m, ti['nc']] ** 2
m = LOBS[:, ti['egb']] & LOBS[:, ti['egc']]; D[m, 2] = L[m, ti['egb']] - L[m, ti['egc']]
DOBS = np.isfinite(D)
dpool = {d: np.where(DOBS[:, j])[0] for j, d in enumerate(DERIVED)}

def clean(M):
    M = np.asarray(M, np.float64); M[~np.isfinite(M)] = np.nan
    return np.clip(M, -1e10, 1e10)
PG = G['M']; PGN = PG / np.maximum(1.0, PG.sum(1, keepdims=True))
DENSE = np.hstack([clean(blocks['desc']), clean(blocks['extra']), clean(blocks['oligo']),
                   clean(blocks['ipc']), clean(P['M']), clean(PGN), clean(G['morph'])])
keep = np.array([np.nanstd(DENSE[:, j]) > 1e-12 and np.isfinite(DENSE[:, j]).mean() > .3
                 for j in range(DENSE.shape[1])])
DENSE = DENSE[:, keep]
md = np.nanmedian(DENSE, 0); ii = np.where(np.isnan(DENSE)); DENSE[ii] = np.take(md, ii[1])
DENSE_QT = QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                               random_state=SEED).fit_transform(DENSE).astype(np.float32)
DENSE = DENSE.astype(np.float32)
SPARSE = np.hstack([np.log1p(blocks['morgan']), blocks['maccs'], np.log1p(blocks['ap']),
                    np.log1p(blocks['tt']), np.log1p(blocks['rk']), np.log1p(PG)]).astype(np.float32)
SPARSE = SPARSE[:, (SPARSE != 0).sum(0) >= 4]
SVD = TruncatedSVD(192, random_state=SEED).fit_transform(SPARSE).astype(np.float32)
TREEX = np.hstack([DENSE, SVD, blocks['maccs']]).astype(np.float32)
MTX = np.hstack([DENSE, SVD[:, :64]]).astype(np.float32)
log("DENSE", DENSE.shape, "SPARSE", SPARSE.shape, "TREEX", TREEX.shape)

MB = np.unpackbits(F['morgan_bin'], axis=1).astype(np.float32); POP = MB.sum(1)
def tanimoto(a, b):
    inter = MB[a] @ MB[b].T
    return inter / (POP[a][:, None] + POP[b][None, :] - inter + 1e-9)
PCX = PCA(120, random_state=SEED).fit_transform(StandardScaler().fit_transform(DENSE_QT)).astype(np.float32)
PCX /= (PCX.std(0) + 1e-9)
_s = np.random.RandomState(SEED).choice(NS, 1200, replace=False)
_X = PCX[_s].astype(np.float64)
_d2 = (_X**2).sum(1)[:, None] + (_X**2).sum(1)[None, :] - 2*_X@_X.T
GAMMA = 1.0 / (2.0 * np.median(_d2[_d2 > 1e-9])); del _d2, _X
def rbf(a, b, ea=None, eb=None):
    Xa, Xb = PCX[a], PCX[b]
    if ea is not None: Xa = np.hstack([Xa, ea]); Xb = np.hstack([Xb, eb])
    d2 = (Xa**2).sum(1)[:, None] + (Xb**2).sum(1)[None, :] - 2*Xa@Xb.T
    return np.exp(-GAMMA*np.clip(d2, 0, None))

def r2(y, p): return 1 - ((y-p)**2).sum() / ((y-y.mean())**2).sum()
def ridge_fit(X, y, al):
    mu = X.mean(0, dtype=np.float64); sd = X.std(0, dtype=np.float64); sd[sd < 1e-12] = 1
    m = Ridge(alpha=al).fit((X-mu)/sd, y)
    w = m.coef_/sd
    return w.astype(np.float32), float(m.intercept_ - mu@w)
def ridge_pred(mdl, X):
    w, b = mdl; return (X @ w).astype(np.float64) + b

MT_MU = np.nanmean(L, 0); MT_SD = np.nanstd(L, 0) + 1e-9
LZ = (L - MT_MU) / MT_SD
def mt_fit_predict(tj, hold, pred):
    Xs, ys, ps, ws = [], [], [], []
    hs = set(np.asarray(hold).tolist()); n_t = max(1, len(pool[T[tj]]))
    for j, t in enumerate(T):
        rr = pool[t]
        if j == tj and hs: rr = rr[~np.isin(rr, list(hs))]
        if not len(rr): continue
        Xs.append(rr); ys.append(LZ[rr, j]); ps.append(np.full(len(rr), j))
        ws.append(np.full(len(rr), 1.0 if j == tj else min(1.0, n_t/len(rr))))
    R = np.concatenate(Xs); yz = np.concatenate(ys)
    pj = np.concatenate(ps); wt = np.concatenate(ws)
    oh = np.zeros((len(R), len(T)), np.float32); oh[np.arange(len(R)), pj] = 1
    def part(rows_, pr):
        PL = LZ[rows_].copy(); PM = LOBS[rows_].astype(np.float32).copy()
        PL[np.arange(len(rows_)), pr] = np.nan; PM[np.arange(len(rows_)), pr] = 0
        PL[:, tj] = np.nan; PM[:, tj] = 0
        return np.hstack([np.nan_to_num(PL), PM]).astype(np.float32)
    Xtr = np.hstack([MTX[R], oh, part(R, pj)])
    Xpr = np.hstack([MTX[pred], np.eye(len(T), dtype=np.float32)[np.full(len(pred), tj)],
                     part(pred, np.full(len(pred), tj))])
    m = lgb.LGBMRegressor(n_estimators=450, learning_rate=.05, num_leaves=31,
                          min_child_samples=10, subsample=.8, subsample_freq=1,
                          colsample_bytree=.5, reg_lambda=1., n_jobs=NJOBS,
                          random_state=SEED, verbose=-1)
    m.fit(Xtr, yz, sample_weight=wt)
    return m.predict(Xpr)*MT_SD[tj] + MT_MU[tj]

def zoo(rows, y, extra_tr=None, extra_te=None, te=None, tag="", mt_col=None, cheap=False):
    n = len(rows); te = np.asarray([] if te is None else te)
    reps = ([KFold(10, shuffle=True, random_state=SEED+1000*r) for r in range(2)]
            if n < 1200 and not cheap else [KFold(5, shuffle=True, random_state=SEED)])
    def st(base, sel, ex): return base[sel] if ex is None else np.hstack([base[sel], ex]).astype(np.float32)
    Xd, Xt, Xs = st(DENSE_QT, rows, extra_tr), st(TREEX, rows, extra_tr), SPARSE[rows]
    Xdte = st(DENSE_QT, te, extra_te) if len(te) else None
    Xtte = st(TREEX, te, extra_te) if len(te) else None
    Xste = SPARSE[te] if len(te) else None
    if extra_tr is not None:
        EM, ES = np.nanmean(extra_tr, 0), np.nanstd(extra_tr, 0)+1e-9
        sc_ = 1/np.sqrt(max(1, extra_tr.shape[1])/8.0)
        KX = ((extra_tr-EM)/ES*sc_).astype(np.float32)
        KXte = ((extra_te-EM)/ES*sc_).astype(np.float32) if len(te) else None
    else:
        KX = KXte = None
    names = ['ridge_d', 'ridge_s', 'et', 'lgbm', 'krr_tan', 'krr_rbf'] + (['mt'] if mt_col is not None else [])
    if cheap: names = ['lgbm', 'et', 'ridge_d']
    oof = {k: np.zeros(n) for k in names}; tep = {k: np.zeros(len(te)) for k in names}
    nrep = {k: 0 for k in names}
    ymu, ysd = y.mean(), y.std()+1e-12
    def mk(k):
        if k == 'et': return ExtraTreesRegressor(n_estimators=400, max_features=.25 if n > 2000 else .30,
                                                 n_jobs=NJOBS, random_state=SEED)
        return lgb.LGBMRegressor(n_estimators=700 if n > 2000 else 900, learning_rate=.035,
                                 num_leaves=31, min_child_samples=max(3, min(20, n//40)),
                                 subsample=.8, subsample_freq=1, colsample_bytree=.35 if n > 2000 else .5,
                                 reg_lambda=1., n_jobs=NJOBS, random_state=SEED, verbose=-1)
    for ri, kf in enumerate(reps):
        for k in names:
            if k != 'mt' or ri == 0: nrep[k] += 1
        for tr, va in kf.split(np.arange(n)):
            for k in names:
                if k == 'mt':
                    if ri == 0: oof[k][va] += mt_fit_predict(mt_col, rows[va], rows[va])
                elif k == 'krr_tan':
                    K = tanimoto(rows[tr], rows[tr]); Kv = tanimoto(rows[va], rows[tr])
                    w = np.linalg.solve(K+.12*np.eye(len(tr)), (y[tr]-ymu)/ysd)
                    oof[k][va] += Kv@w*ysd+ymu
                elif k == 'krr_rbf':
                    e1 = None if KX is None else KX[tr]; e2 = None if KX is None else KX[va]
                    K = rbf(rows[tr], rows[tr], e1, e1); Kv = rbf(rows[va], rows[tr], e2, e1)
                    w = np.linalg.solve(K+.05*np.eye(len(tr)), (y[tr]-ymu)/ysd)
                    oof[k][va] += Kv@w*ysd+ymu
                elif k.startswith('ridge'):
                    X = Xd if k == 'ridge_d' else Xs
                    oof[k][va] += ridge_pred(ridge_fit(X[tr], y[tr], 20. if k == 'ridge_d' else 6.), X[va])
                else:
                    mdl = mk(k); mdl.fit(Xt[tr], y[tr]); oof[k][va] += mdl.predict(Xt[va])
    for k in names: oof[k] /= max(1, nrep[k])
    if len(te):
        for k in names:
            if k == 'mt': tep[k] = mt_fit_predict(mt_col, np.array([], int), te)
            elif k == 'krr_tan':
                K = tanimoto(rows, rows); Kt = tanimoto(te, rows)
                w = np.linalg.solve(K+.12*np.eye(n), (y-ymu)/ysd); tep[k] = Kt@w*ysd+ymu
            elif k == 'krr_rbf':
                K = rbf(rows, rows, KX, KX); Kt = rbf(te, rows, KXte, KX)
                w = np.linalg.solve(K+.05*np.eye(n), (y-ymu)/ysd); tep[k] = Kt@w*ysd+ymu
            elif k.startswith('ridge'):
                X, Xt_ = (Xd, Xdte) if k == 'ridge_d' else (Xs, Xste)
                tep[k] = ridge_pred(ridge_fit(X, y, 20. if k == 'ridge_d' else 6.), Xt_)
            else:
                mdl = mk(k); mdl.fit(Xt, y); tep[k] = mdl.predict(Xtte)
    Mo = np.column_stack([oof[k] for k in names])
    w, _ = nnls(Mo, y)
    if w.sum() <= 1e-9: w = np.ones(len(names))
    w /= w.sum()
    best = max(((r2(y, Mo@((1-l)*w+l/len(names))), (1-l)*w+l/len(names))
                for l in (0, .1, .2, .35, .5)), key=lambda z: z[0])
    bs, w = best
    solo = max(names, key=lambda k: r2(y, oof[k]))
    if r2(y, oof[solo]) > bs:
        w = np.array([1.0 if k == solo else 0.0 for k in names])
    bo = Mo@w
    bt = np.column_stack([tep[k] for k in names])@w if len(te) else np.zeros(0)
    log(f"   {tag:18s} " + " ".join(f"{k}={r2(y,oof[k]):.4f}" for k in names)
        + f" | BLEND={r2(y,bo):.4f}")
    return bo, bt

# ---------------- stage 1 ----------------
if os.environ.get("RESUME_FROM_P1") == "1" and os.path.exists(f"{OUT}/P1.npy") and os.path.exists(f"{OUT}/P1D.npy"):
    P1 = np.load(f"{OUT}/P1.npy")
    P1D = np.load(f"{OUT}/P1D.npy")
    if P1.shape != (NS, len(T)) or P1D.shape != (NS, len(DERIVED)):
        raise RuntimeError("cached Stage 1 arrays have incompatible shapes")
    log("RESUME: loaded completed Stage 1 arrays; skipping Stage 1")
else:
    log("=" * 70); log("STAGE 1"); log("=" * 70)
    P1 = np.zeros((NS, len(T)))
    for j, t in enumerate(T):
        rows = pool[t]; y = L[rows, j]; other = np.setdiff1d(np.arange(NS), rows)
        cheap = t in ('tg', 'egc')             # partner covariates only
        bo, bt = zoo(rows, y, te=other, tag=f"S1:{t}", mt_col=None if cheap else j, cheap=cheap)
        P1[rows, j] = bo; P1[other, j] = bt
        log(f"  STAGE1 {t:4s} OOF R2 = {r2(y, bo):.6f}")
    P1D = np.zeros((NS, 3))
    for j, d in enumerate(DERIVED):
        rows = dpool[d]; y = D[rows, j]; other = np.setdiff1d(np.arange(NS), rows)
        bo, bt = zoo(rows, y, te=other, tag=f"S1:{d}")
        P1D[rows, j] = bo; P1D[other, j] = bt
        log(f"  STAGE1 {d:6s} OOF R2 = {r2(y, bo):.6f}")
    np.save(f"{OUT}/P1.npy", P1); np.save(f"{OUT}/P1D.npy", P1D)

def cp_block(cur, curD, exclude):
    ex = set(exclude or [])
    BA = np.where(LOBS, L, cur); OB = LOBS.astype(np.float64).copy()
    for j in ex: BA[:, j] = cur[:, j]; OB[:, j] = 0
    cols = []
    for j, t in enumerate(T):
        if j in ex: continue
        cols += [BA[:, j], OB[:, j], np.where(LOBS[:, j], L[:, j], np.nan)]
    g = {t: BA[:, ti[t]] for t in T}
    chi, ion, dg = curD[:, 0], curD[:, 1], curD[:, 2]
    cols += [g['eea']+g['egc'], chi+g['egc']/2, g['ei']-g['egc'], chi-g['egc']/2,
             g['ei']-g['eea'], (g['egb']+.9221)/1.1178, g['egb']-dg,
             1.1178*g['egc']-.9221, g['egc']+dg,
             g['nc']**2+.737, g['nc']**2+ion,
             np.sqrt(np.clip(g['eps']-.652, 1, None)), np.sqrt(np.clip(g['eps']-ion, 1, None)),
             chi, ion, dg, (g['ei']+g['eea'])/2, (g['ei']-g['eea'])/2,
             np.clip(g['egc'], .05, None)**-.25, 1/np.clip(g['egb'], .05, None),
             g['nc']**2, np.sqrt(np.clip(g['eps'], 0, None)), OB.sum(1)]
    B = np.column_stack(cols).astype(np.float64)
    bm = np.nanmedian(B, 0); bm[~np.isfinite(bm)] = 0
    kk = np.where(~np.isfinite(B)); B[kk] = np.take(bm, kk[1])
    return B.astype(np.float32)

# ---------------- stages 2 and 3, weak targets only ----------------
if os.environ.get("RESUME_FROM_P2") == "1" and os.path.exists(f"{OUT}/P2.npy") and os.path.exists(f"{OUT}/P2D.npy"):
    cur = np.load(f"{OUT}/P2.npy")
    curD = np.load(f"{OUT}/P2D.npy")
    if cur.shape != P1.shape or curD.shape != P1D.shape:
        raise RuntimeError("cached Stage 2 arrays have incompatible shapes")
    stage_sequence = (3,)
    log("RESUME: loaded completed Stage 2 arrays; skipping Stage 2")
else:
    cur, curD = P1.copy(), P1D.copy()
    stage_sequence = (2, 3)
for rnd in stage_sequence:
    log("=" * 70); log(f"STAGE {rnd} (weak targets)"); log("=" * 70)
    nxt, nxtD = cur.copy(), curD.copy()
    for t in WEAK:
        j = ti[t]; rows = pool[t]; y = L[rows, j]
        other = np.setdiff1d(np.arange(NS), rows)
        Ball = cp_block(cur, curD, [j])
        Ball = np.hstack([Ball, cur[:, j:j+1].astype(np.float32), P1[:, j:j+1].astype(np.float32)])
        try:
            bo, bt = zoo(rows, y, Ball[rows], Ball[other], other, f"S{rnd}:{t}", mt_col=j)
        except Exception as e:
            log(f"   !! {t}: {type(e).__name__}: {e}"); continue
        prev = r2(y, cur[rows, j])
        if r2(y, bo) >= prev: nxt[rows, j] = bo; nxt[other, j] = bt
        log(f"  STAGE{rnd} {t:4s} OOF R2 = {max(r2(y,bo),prev):.6f}   (was {prev:.6f})")
    for j, d in enumerate(DERIVED):
        rows = dpool[d]; y = D[rows, j]; other = np.setdiff1d(np.arange(NS), rows)
        drop = [ti[c] for c in {'chi': ['ei','eea'], 'ionic': ['eps','nc'], 'dgap': ['egb','egc']}[d]]
        Ball = cp_block(cur, curD, drop)
        try:
            bo, bt = zoo(rows, y, Ball[rows], Ball[other], other, f"S{rnd}:{d}")
        except Exception as e:
            log(f"   !! {d}: {type(e).__name__}: {e}"); continue
        if r2(y, bo) >= r2(y, curD[rows, j]): nxtD[rows, j] = bo; nxtD[other, j] = bt
        log(f"  STAGE{rnd} {d:6s} OOF R2 = {max(r2(y,bo), r2(y,curD[rows,j])):.6f}")
    cur, curD = nxt, nxtD
    if rnd == 2:
        np.save(f"{OUT}/P2.npy", cur); np.save(f"{OUT}/P2D.npy", curD)
        log("CHECKPOINT: saved Stage 2 arrays")
np.save(f"{OUT}/PFINAL.npy", cur); np.save(f"{OUT}/PFINALD.npy", curD)

# ---------------- stage 4: availability-conditioned routing ----------------
log("=" * 70); log("STAGE 4 — availability-conditioned physics routing"); log("=" * 70)
BA = np.where(LOBS, L, cur)
def recon(t):
    g = {c: BA[:, ti[c]] for c in T}; chi, ion, dg = curD[:, 0], curD[:, 1], curD[:, 2]
    return {'ei': (g['eea']+g['egc'], LOBS[:, ti['eea']]),
            'eea': (g['ei']-g['egc'], LOBS[:, ti['ei']]),
            'egb': (g['egc']+dg, LOBS[:, ti['egc']]),
            'eps': (g['nc']**2+ion, LOBS[:, ti['nc']]),
            'nc': (np.sqrt(np.clip(g['eps']-ion, 1, None)), LOBS[:, ti['eps']])}[t]
final = cur.copy()
for t in WEAK:
    j = ti[t]; rows = pool[t]; y = L[rows, j]
    rec, obs = recon(t); base = r2(y, cur[rows, j])
    fitted = np.zeros(len(rows)); used = np.zeros(len(rows), bool); models = {}
    for rg, mask in [('obs', obs), ('miss', ~obs)]:
        sel = np.where(mask[rows])[0]
        if len(sel) < 40: continue
        Xs = np.column_stack([cur[rows[sel], j], rec[rows[sel]]]); ys = y[sel]
        ok = np.isfinite(Xs).all(1); sel, Xs, ys = sel[ok], Xs[ok], ys[ok]
        oo = np.zeros(len(sel))
        for tr_, va_ in KFold(min(10, max(3, len(sel)//20)), shuffle=True, random_state=SEED).split(Xs):
            oo[va_] = RidgeCV(alphas=[.01, .1, 1., 10.]).fit(Xs[tr_], ys[tr_]).predict(Xs[va_])
        s0, sp, sr = r2(ys, Xs[:, 0]), r2(ys, Xs[:, 1]), r2(ys, oo)
        log(f"  {t:4s} {rg:5s} n={len(sel):4d}  stack={s0:.4f} physics={sp:.4f} routed={sr:.4f}"
            f"  {'take' if sr > s0+1e-6 else 'reject'}")
        if sr > s0 + 1e-6:
            fitted[sel] = oo; used[sel] = True
            models[rg] = (RidgeCV(alphas=[.01, .1, 1., 10.]).fit(Xs, ys), mask)
    if used.any():
        comb = cur[rows, j].copy(); comb[used] = fitted[used]
        if r2(y, comb) > base:
            log(f"  -> {t}: {base:.6f} -> {r2(y, comb):.6f}  ({r2(y,comb)-base:+.6f})")
            for rg, (mdl, mask) in models.items():
                a = np.where(mask)[0]
                Xa = np.column_stack([cur[a, j], rec[a]]); gd = np.isfinite(Xa).all(1)
                final[a[gd], j] = mdl.predict(Xa[gd])
np.save(f"{OUT}/PFINAL_routed.npy", final)

log("=" * 70)
C050_TEST = dict(egb=.8990616896, ei=.7568828960, eea=.8681479551, nc=.8295703105, eps=.7698645559)
C050_OOF = dict(egb=.9221467344, ei=.8454440895, eea=.9008357940, nc=.8397322432, eps=.7835054390)
res = {}
for t in WEAK:
    j = ti[t]; rows = pool[t]
    s = r2(L[rows, j], final[rows, j]); res[t] = s
    log(f"  {t:4s} OOF {s:.6f}   cal {s-(C050_OOF[t]-C050_TEST[t]):.6f}   "
        f"C050 test {C050_TEST[t]:.4f}   delta {s-C050_TEST[t]:+.4f}")
log(f"  weak-5 mean OOF {np.mean(list(res.values())):.6f}")
json.dump(res, open(f"{OUT}/weak_oof.json", 'w'), indent=2)
log("DONE")
