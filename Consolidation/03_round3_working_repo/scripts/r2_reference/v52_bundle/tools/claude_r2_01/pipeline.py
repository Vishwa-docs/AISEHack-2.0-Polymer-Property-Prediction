"""
Seven-target polymer property pipeline — official data only, no pretrained weights.

Architecture
------------
 Stage 0  Assemble the official label table over canonical structures from
          ppp-round-2/train.csv UNION ppp-round-2/archive/train.csv (both are in
          the official competition bundle).  Assemble structure features.

 Stage 1  Per-target structure-only model zoo (Ridge-dense, Ridge-sparse,
          ExtraTrees, LightGBM, Tanimoto-kernel ridge), NNLS-blended on
          out-of-fold predictions.  Also fit the same zoo on three derived
          physics coordinates that are far lower-variance than the raw targets:
            chi   = (Ei + Eea)/2      gap centre / Mulliken electronegativity
            ionic = EPS - Nc^2        ionic part of the dielectric constant
            dgap  = Egb - Egc         chain->bulk band-gap offset

 Stage 2  Cross-property stack.  Every test row's structure carries labels for
          the OTHER properties in the official train file at almost exactly the
          rate seen in train (measured: eps<-nc 62% test vs 59% train, ei<-eea
          66% vs 55%, egb<-egc 55% vs 52%), so those labels are legitimate
          test-time covariates.  Features = structure block + observed other
          labels + missing masks + best-available values + physics
          reconstructions built from them.

 Stage 3  One more round of Stage 2 using Stage-2 outputs as best-available,
          which propagates information between test rows that partner each
          other (Nc and EPS cover the same 382 structures, Eea subset of Ei).

 Stage 4  Hard physical constraint EPS >= Nc^2 projected onto the joint output.

 Stage 5  Exact override for the Tg/Egc test rows whose structure carries an
          official archive label.
"""
import os, sys, pickle, time, warnings, json, hashlib
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import nnls
import lightgbm as lgb

BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
OUT = f"{SCR}/out" + (os.environ.get('OUT_TAG', '') or '')
os.makedirs(OUT, exist_ok=True)
T = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
DERIVED = ['chi', 'ionic', 'dgap']
SMOKE = os.environ.get('SMOKE', '') == '1'   # big pools subsampled, weak targets full size
TINY = os.environ.get('TINY', '') == '1'     # every pool tiny, code-path validation only
OUT_TAG = os.environ.get('OUT_TAG', '')
SEED = 20260804
NJOBS = 22
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# =====================================================================
# Stage 0 — data
# =====================================================================
F = pickle.load(open(f"{SCR}/features.pkl", "rb"))
P = pickle.load(open(f"{SCR}/physics.pkl", "rb"))
G = pickle.load(open(f"{SCR}/pgfp.pkl", "rb"))     # Polymer Genome atomic triples
idx, canon_map, blocks = F['idx'], F['canon_map'], F['blocks']
NS = len(F['canon_list'])

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
arch = pd.read_csv(f"{BASE}/archive/train.csv")
for df in (train, test, arch):
    df['canon'] = df['smiles'].map(canon_map)
    df['fi'] = df['canon'].map(idx).astype(int)

lab_r2 = train.pivot_table(index='canon', columns='target_type', values='target', aggfunc='mean')
lab_ar = arch.pivot_table(index='canon', columns='target_type', values='target', aggfunc='mean')
for c in T:
    if c not in lab_r2.columns: lab_r2[c] = np.nan
    if c not in lab_ar.columns: lab_ar[c] = np.nan
lab_r2, lab_ar = lab_r2[T], lab_ar[T]

# L[s, t] = official label of property t for structure s (round-2 preferred, archive fills)
L = np.full((NS, len(T)), np.nan)
for j, t in enumerate(T):
    for src in (lab_ar, lab_r2):                       # round-2 written last -> wins
        v = src[t].dropna()
        L[[idx[c] for c in v.index], j] = v.values
LOBS = np.isfinite(L)
log("label table:", {t: int(LOBS[:, j].sum()) for j, t in enumerate(T)})

# archive-only lookup table used for the final exact override
A = np.full((NS, len(T)), np.nan)
for j, t in enumerate(T):
    v = lab_ar[t].dropna()
    A[[idx[c] for c in v.index], j] = v.values

# training pool per target = every structure carrying that official label
pool = {t: np.where(LOBS[:, j])[0] for j, t in enumerate(T)}
if SMOKE:   # subsample the two big pools so a full structural pass runs in minutes
    rs = np.random.RandomState(0)
    for t in ('tg', 'egc'):
        pool[t] = np.sort(rs.choice(pool[t], 500, replace=False))
    log("SMOKE mode: tg/egc pools subsampled to 500")
if TINY:    # every pool tiny: exercises all three stages end to end in a few minutes
    rs = np.random.RandomState(0)
    for t in T:
        if len(pool[t]) > 160:
            pool[t] = np.sort(rs.choice(pool[t], 160, replace=False))
    log("TINY mode: every pool subsampled to <=160 (code-path validation only)")
test_fi = {t: test.loc[test.target_type == t, 'fi'].values for t in T}
test_id = {t: test.loc[test.target_type == t, 'id'].values for t in T}
for t in T:
    log(f"  {t:4s} pool={len(pool[t]):5d}  test={len(test_fi[t]):5d}")

# ---- derived physics coordinates -------------------------------------------------
ti = {t: j for j, t in enumerate(T)}
D = np.full((NS, 3), np.nan)
m = LOBS[:, ti['ei']] & LOBS[:, ti['eea']]
D[m, 0] = (L[m, ti['ei']] + L[m, ti['eea']]) / 2.0                 # chi
m = LOBS[:, ti['eps']] & LOBS[:, ti['nc']]
D[m, 1] = L[m, ti['eps']] - L[m, ti['nc']] ** 2                    # ionic
m = LOBS[:, ti['egb']] & LOBS[:, ti['egc']]
D[m, 2] = L[m, ti['egb']] - L[m, ti['egc']]                        # dgap
DOBS = np.isfinite(D)
for j, d in enumerate(DERIVED):
    log(f"  derived {d:6s} n={int(DOBS[:,j].sum()):4d} "
        f"mean={np.nanmean(D[:,j]):.4f} std={np.nanstd(D[:,j]):.4f}")
dpool = {d: np.where(DOBS[:, j])[0] for j, d in enumerate(DERIVED)}

# ---- structure feature blocks ----------------------------------------------------
def clean(M):
    M = np.asarray(M, dtype=np.float64)
    M[~np.isfinite(M)] = np.nan
    return np.clip(M, -1e10, 1e10)

PG = G['M']
PGN = PG / np.maximum(1.0, PG.sum(1, keepdims=True))       # composition fractions
DENSE = np.hstack([clean(blocks['desc']), clean(blocks['extra']),
                   clean(blocks['oligo']), clean(blocks['ipc']), clean(P['M']),
                   clean(PGN), clean(G['morph'])])
keep = np.array([np.nanstd(DENSE[:, j]) > 1e-12 and np.isfinite(DENSE[:, j]).mean() > 0.30
                 for j in range(DENSE.shape[1])])
DENSE = DENSE[:, keep]
dmed = np.nanmedian(DENSE, axis=0)
ii = np.where(np.isnan(DENSE)); DENSE[ii] = np.take(dmed, ii[1])
# rank-gauss so linear models are not dominated by heavy-tailed descriptors
DENSE_QT = QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                               random_state=SEED).fit_transform(DENSE)
DENSE = DENSE.astype(np.float32); DENSE_QT = DENSE_QT.astype(np.float32)

SPARSE = np.hstack([np.log1p(blocks['morgan']), blocks['maccs'], np.log1p(blocks['ap']),
                    np.log1p(blocks['tt']), np.log1p(blocks['rk']),
                    np.log1p(PG)]).astype(np.float32)
SPARSE = SPARSE[:, (SPARSE != 0).sum(0) >= 4]
# unsupervised compression of the fingerprint block (no labels touched)
SVD = TruncatedSVD(n_components=192, random_state=SEED).fit_transform(SPARSE).astype(np.float32)
TREEX = np.hstack([DENSE, SVD, blocks['maccs']]).astype(np.float32)
MTX = np.hstack([DENSE, SVD[:, :64]]).astype(np.float32)   # lighter block for the pooled arm
log("DENSE", DENSE.shape, "SPARSE", SPARSE.shape, "TREEX", TREEX.shape)

# Tanimoto kernel over Morgan-r2 bits
MB = np.unpackbits(F['morgan_bin'], axis=1).astype(np.float32)
POP = MB.sum(1)
def tanimoto(rows, cols):
    inter = MB[rows] @ MB[cols].T
    return inter / (POP[rows][:, None] + POP[cols][None, :] - inter + 1e-9)

# compact isotropic space for the RBF kernel model (GP on handcrafted descriptors is
# the strongest classical baseline reported for these exact six DFT properties)
from sklearn.decomposition import PCA
PCX = PCA(n_components=120, random_state=SEED).fit_transform(
    StandardScaler().fit_transform(DENSE_QT)).astype(np.float32)
PCX /= (PCX.std(0) + 1e-9)
_sub = np.random.RandomState(SEED).choice(NS, 1500, replace=False)
_X = PCX[_sub].astype(np.float64)
_d2 = (_X ** 2).sum(1)[:, None] + (_X ** 2).sum(1)[None, :] - 2 * _X @ _X.T
GAMMA = 1.0 / (2.0 * np.median(_d2[_d2 > 1e-9]))
del _d2, _X
log(f"RBF gamma={GAMMA:.6g}")

def rbf(rows, cols, extra_r=None, extra_c=None):
    Xr, Xc = PCX[rows], PCX[cols]
    if extra_r is not None:
        Xr = np.hstack([Xr, extra_r]); Xc = np.hstack([Xc, extra_c])
    d2 = ((Xr ** 2).sum(1)[:, None] + (Xc ** 2).sum(1)[None, :] - 2 * Xr @ Xc.T)
    return np.exp(-GAMMA * np.clip(d2, 0, None))

def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def ridge_fit(Xtr, y, alpha):
    """Standardise + ridge, returned as a plain (w, b) pair.

    Predicting through StandardScaler.transform would materialise a full scaled
    copy of the 8,990 x 7,458 sparse block on every call; folding the scaling into
    the coefficients turns prediction into a single matmul.
    """
    mu = Xtr.mean(0, dtype=np.float64)
    sd = Xtr.std(0, dtype=np.float64); sd[sd < 1e-12] = 1.0
    Z = (Xtr - mu) / sd
    m = Ridge(alpha=alpha).fit(Z, y)
    w = (m.coef_ / sd).astype(np.float64)
    return w, float(m.intercept_ - mu @ w)

def ridge_pred(model, X):
    w, b = model
    # keep the matmul in float32 so an 8,990 x 7,458 float64 temporary is never made
    return (X @ w.astype(np.float32, copy=False)).astype(np.float64) + b

def folds_for(n):
    """The five weak targets have only 220-340 rows, so a single 5-fold OOF is a
    very noisy estimate and a noisy blend-fitting signal.  Average over several
    fold seeds there; the two large pools stay at a single 5-fold pass."""
    if n < 1200:
        return [KFold(n_splits=10, shuffle=True, random_state=SEED + 1000 * r)
                for r in range(2)]
    return [KFold(n_splits=5, shuffle=True, random_state=SEED)]

# =====================================================================
# model zoo
# =====================================================================
# ---------------------------------------------------------------------
# multi-task arm
# ---------------------------------------------------------------------
# Kuenneth et al. (Patterns 2021) report that a single multi-task model over all
# properties is still the best published result on Egb / EPS / Nc (R2 0.95 / 0.86 /
# 0.91) versus 0.91 / 0.68 / 0.79 for per-property single-task models, because the
# sparse optical/dielectric targets borrow strength from the dense ones.  This arm
# reproduces that: one model over every (structure, property) pair, each property
# z-scored, with a property one-hot appended to the structural features.
MT_MU = np.array([np.nanmean(L[:, j]) for j in range(len(T))])
MT_SD = np.array([np.nanstd(L[:, j]) + 1e-9 for j in range(len(T))])

LZ = (L - MT_MU) / MT_SD            # per-property z-scored official labels

def mt_fit_predict(target_j, hold_rows, pred_rows, extra_all=None):
    """Train the pooled multi-task model excluding (s, target_j) for s in hold_rows,
    then predict property target_j for pred_rows.

    Two details matter:
      * rows of the other properties are down-weighted so their total weight equals
        the target property's — otherwise Tg's 5,781 rows drown Eps's 229;
      * every row also carries the OTHER properties' observed labels, which makes
        this a joint multi-task + cross-property model.  Column `target_j` is
        blanked for every row in the pool, so the quantity being predicted can
        never appear as an input anywhere in this model.
    """
    Xs, ys, ps, ws = [], [], [], []
    hold = set(hold_rows.tolist())
    n_t = max(1, len(pool[T[target_j]]))
    for j, t in enumerate(T):
        rr = pool[t]
        if j == target_j and hold:
            rr = rr[~np.isin(rr, list(hold))]
        if len(rr) == 0:
            continue
        Xs.append(rr); ys.append(LZ[rr, j]); ps.append(np.full(len(rr), j))
        w = 1.0 if j == target_j else min(1.0, n_t / len(rr))
        ws.append(np.full(len(rr), w))
    R = np.concatenate(Xs); yz = np.concatenate(ys)
    pj = np.concatenate(ps); wt = np.concatenate(ws)
    onehot = np.zeros((len(R), len(T)), dtype=np.float32); onehot[np.arange(len(R)), pj] = 1

    def partner(rows_, prop_of_row):
        PL = LZ[rows_].copy(); PM = LOBS[rows_].astype(np.float32).copy()
        if prop_of_row is not None:                    # blank each row's own property
            PL[np.arange(len(rows_)), prop_of_row] = np.nan
            PM[np.arange(len(rows_)), prop_of_row] = 0.0
        PL[:, target_j] = np.nan                       # and the predicted property, always
        PM[:, target_j] = 0.0
        return np.hstack([np.nan_to_num(PL, nan=0.0), PM]).astype(np.float32)

    Xtr = np.hstack([MTX[R], onehot, partner(R, pj)])
    Xpr = np.hstack([MTX[pred_rows],
                     np.eye(len(T), dtype=np.float32)[np.full(len(pred_rows), target_j)],
                     partner(pred_rows, np.full(len(pred_rows), target_j))])
    m = lgb.LGBMRegressor(n_estimators=450, learning_rate=0.05, num_leaves=31,
                          min_child_samples=10, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, reg_lambda=1.0, n_jobs=NJOBS,
                          random_state=SEED, verbose=-1)
    m.fit(Xtr, yz, sample_weight=wt)
    return m.predict(Xpr) * MT_SD[target_j] + MT_MU[target_j]


def fit_predict_zoo(rows, y, extra_tr=None, extra_te=None, te_rows=None, tag="",
                    mt_col=None, extra_all=None):
    """Cross-fitted OOF over `rows` plus full-fit predictions for `te_rows`.

    extra_tr / extra_te are optional extra feature columns (cross-property block)
    aligned with rows / te_rows.
    """
    n = len(rows)
    reps = folds_for(n)
    te_rows = np.asarray([]) if te_rows is None else np.asarray(te_rows)
    def stack(base, sel, ex):
        return base[sel] if ex is None else np.hstack([base[sel], ex]).astype(np.float32)

    Xd_all = stack(DENSE_QT, rows, extra_tr)
    Xt_all = stack(TREEX, rows, extra_tr)
    Xs_all = SPARSE[rows]
    Xd_te = stack(DENSE_QT, te_rows, extra_te) if len(te_rows) else None
    Xt_te = stack(TREEX, te_rows, extra_te) if len(te_rows) else None
    Xs_te = SPARSE[te_rows] if len(te_rows) else None

    # the extra (cross-property) block also enters the kernel models, scaled so it
    # does not dominate the 120 structural PCA directions
    def kx(ex):
        if ex is None: return None
        Z = np.asarray(ex, dtype=np.float64)
        return ((Z - EXMU) / EXSD * (1.0 / np.sqrt(max(1, Z.shape[1]) / 8.0))).astype(np.float32)
    if extra_tr is not None:
        EXMU = np.nanmean(extra_tr, axis=0); EXSD = np.nanstd(extra_tr, axis=0) + 1e-9
        KEX_tr, KEX_te = kx(extra_tr), (kx(extra_te) if len(te_rows) else None)
    else:
        KEX_tr = KEX_te = None

    names = ['ridge_d', 'ridge_s', 'et', 'lgbm', 'krr_tan', 'krr_rbf']
    if mt_col is not None:
        names.append('mt')
    oof = {k: np.zeros(n) for k in names}
    tep = {k: np.zeros(len(te_rows)) for k in names}

    def make(k):
        if k == 'ridge_d': return Ridge(alpha=20.0, random_state=SEED)
        if k == 'ridge_s': return Ridge(alpha=6.0, random_state=SEED)
        big = n > 2000
        if k == 'et':
            return ExtraTreesRegressor(n_estimators=300 if big else 500,
                                       max_features=0.15 if big else 0.30,
                                       min_samples_leaf=1, n_jobs=NJOBS, random_state=SEED)
        if k == 'lgbm':
            return lgb.LGBMRegressor(n_estimators=700 if big else 900,
                                     learning_rate=0.035 if big else 0.03, num_leaves=31,
                                     min_child_samples=max(3, min(20, n // 40)),
                                     subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.30 if big else 0.5,
                                     reg_lambda=1.0, n_jobs=NJOBS, random_state=SEED,
                                     verbose=-1)
        return None

    ymu, ysd = y.mean(), y.std() + 1e-12
    KERN = {'krr_tan': (lambda a, b: tanimoto(a, b), 0.12),
            'krr_rbf': (None, 0.05)}
    nrep = {k: 0 for k in names}
    for ri, kf in enumerate(reps):
        for k in names:
            # the pooled arm refits ~9.6k rows per fold, so it runs one pass only
            if k != 'mt' or ri == 0:
                nrep[k] += 1
        for tr, va in kf.split(np.arange(n)):
            for k in names:
                if k == 'mt':
                    if ri == 0:
                        oof[k][va] += mt_fit_predict(mt_col, rows[va], rows[va], extra_all)
                    continue
                if k in KERN:
                    al = KERN[k][1]
                    if k == 'krr_tan':
                        Ktr, Kva = tanimoto(rows[tr], rows[tr]), tanimoto(rows[va], rows[tr])
                    else:
                        et_ = None if KEX_tr is None else KEX_tr[tr]
                        ev_ = None if KEX_tr is None else KEX_tr[va]
                        Ktr = rbf(rows[tr], rows[tr], et_, et_)
                        Kva = rbf(rows[va], rows[tr], ev_, et_)
                    w = np.linalg.solve(Ktr + al * np.eye(len(tr)), (y[tr] - ymu) / ysd)
                    oof[k][va] += Kva @ w * ysd + ymu
                    continue
                X = {'ridge_d': Xd_all, 'ridge_s': Xs_all}.get(k, Xt_all)
                if k.startswith('ridge'):
                    al = 20.0 if k == 'ridge_d' else 6.0
                    oof[k][va] += ridge_pred(ridge_fit(X[tr], y[tr], al), X[va])
                else:
                    mdl = make(k)
                    mdl.fit(X[tr], y[tr]); oof[k][va] += mdl.predict(X[va])
    for k in names:
        oof[k] /= max(1, nrep[k])

    if len(te_rows):
        for k in names:
            if k == 'mt':
                tep[k] = mt_fit_predict(mt_col, np.array([], dtype=int), te_rows, extra_all)
                continue
            if k in KERN:
                al = KERN[k][1]
                if k == 'krr_tan':
                    Ktr, Kte = tanimoto(rows, rows), tanimoto(te_rows, rows)
                else:
                    Ktr = rbf(rows, rows, KEX_tr, KEX_tr)
                    Kte = rbf(te_rows, rows, KEX_te, KEX_tr)
                w = np.linalg.solve(Ktr + al * np.eye(n), (y - ymu) / ysd)
                tep[k] = Kte @ w * ysd + ymu
                continue
            X, Xte = ({'ridge_d': (Xd_all, Xd_te), 'ridge_s': (Xs_all, Xs_te)}
                      .get(k, (Xt_all, Xt_te)))
            if k.startswith('ridge'):
                al = 20.0 if k == 'ridge_d' else 6.0
                tep[k] = ridge_pred(ridge_fit(X, y, al), Xte)
            else:
                mdl = make(k)
                mdl.fit(X, y); tep[k] = mdl.predict(Xte)

    # NNLS blend on OOF, shrunk toward the uniform average, and never worse on OOF
    # than the single best member.
    Mo = np.column_stack([oof[k] for k in names])
    Mt = np.column_stack([tep[k] for k in names]) if len(te_rows) else None
    w, _ = nnls(Mo, y)
    if w.sum() <= 1e-9:
        w = np.ones(len(names))
    w = w / w.sum()
    best = None
    for lam in (0.0, 0.10, 0.20, 0.35, 0.50):
        wl = (1 - lam) * w + lam / len(names)
        s = r2(y, Mo @ wl)
        if best is None or s > best[0]:
            best = (s, wl)
    bscore, w = best
    solo = max(range(len(names)), key=lambda i: r2(y, oof[names[i]]))
    if r2(y, oof[names[solo]]) > bscore:
        w = np.zeros(len(names)); w[solo] = 1.0
    bo = Mo @ w
    bt = Mt @ w if len(te_rows) else np.zeros(0)
    parts = "  ".join(f"{k}={r2(y, oof[k]):.4f}" for k in names)
    log(f"    {tag:22s} {parts}  | BLEND={r2(y, bo):.4f}  w={np.round(w,3)}")
    return bo, bt, {k: oof[k] for k in names}, {k: tep[k] for k in names}, w


# =====================================================================
# Stage 1 — structure-only base models
# =====================================================================
if os.environ.get("RESUME_FROM_P1") == "1" and os.path.exists(f"{OUT}/P1.npy") and os.path.exists(f"{OUT}/P1D.npy"):
    P1 = np.load(f"{OUT}/P1.npy")
    P1D = np.load(f"{OUT}/P1D.npy")
    if P1.shape != (NS, len(T)) or P1D.shape != (NS, len(DERIVED)):
        raise RuntimeError("cached Stage 1 arrays have incompatible shapes")
    S1_OOF = {}
    log("RESUME: loaded completed Stage 1 arrays; skipping Stage 1")
else:
    log("=" * 78); log("STAGE 1 — structure-only base models"); log("=" * 78)
    # P1[s, t] = best structure-only estimate of property t for structure s.
    # For labeled structures it is the out-of-fold estimate; for the rest, a full fit.
    P1 = np.zeros((NS, len(T)))
    S1_OOF = {}
    for j, t in enumerate(T):
        rows = pool[t]; y = L[rows, j]
        other = np.setdiff1d(np.arange(NS), rows)
        bo, bt, _, _, _ = fit_predict_zoo(rows, y, te_rows=other, tag=f"S1:{t}", mt_col=j)
        P1[rows, j] = bo; P1[other, j] = bt
        S1_OOF[t] = (rows, bo, y)
        log(f"  STAGE1 {t:4s} OOF R2 = {r2(y, bo):.6f}")

    P1D = np.zeros((NS, 3))
    for j, d in enumerate(DERIVED):
        rows = dpool[d]; y = D[rows, j]
        other = np.setdiff1d(np.arange(NS), rows)
        bo, bt, _, _, _ = fit_predict_zoo(rows, y, te_rows=other, tag=f"S1:{d}")
        P1D[rows, j] = bo; P1D[other, j] = bt
        log(f"  STAGE1 {d:6s} OOF R2 = {r2(y, bo):.6f}")

    np.save(f"{OUT}/P1.npy", P1); np.save(f"{OUT}/P1D.npy", P1D)

# =====================================================================
# cross-property covariate construction
# =====================================================================
def cp_block(cur, curD, exclude):
    """Cross-property feature block for EVERY structure (callers slice it).

    Building it once over all 8,990 structures keeps the median imputation and the
    column order identical between the fitted rows and the predicted rows.

    `exclude` is the list of property columns that must NOT contribute their
    observed label — for a raw target that is the target itself; for a derived
    coordinate it is the properties the coordinate is built from.
    """
    exclude = set(exclude or [])
    rows = np.arange(NS)
    BA = np.where(LOBS, L, cur)
    OB = LOBS.astype(np.float64).copy()
    for j in exclude:
        BA[:, j] = cur[:, j]
        OB[:, j] = 0.0
    ba = BA; ob = OB
    cols, names = [], []
    for j, t in enumerate(T):
        if j in exclude: continue
        cols += [ba[:, j], ob[:, j], np.where(LOBS[:, j], L[:, j], np.nan)]
        names += [f'ba_{t}', f'obs_{t}', f'raw_{t}']
    g = {t: ba[:, j] for j, t in enumerate(T)}
    chi, ionic, dgap = curD[:, 0], curD[:, 1], curD[:, 2]
    # physics reconstructions, all built from best-available values
    rec = {
        'ei_from_gap':   g['eea'] + g['egc'],
        'ei_from_chi':   chi + g['egc'] / 2.0,
        'eea_from_gap':  g['ei'] - g['egc'],
        'eea_from_chi':  chi - g['egc'] / 2.0,
        'egc_from_ie':   g['ei'] - g['eea'],
        'egc_from_egb':  (g['egb'] + 0.9221) / 1.1178,
        'egc_from_dgap': g['egb'] - dgap,
        'egb_from_egc':  1.1178 * g['egc'] - 0.9221,
        'egb_from_dgap': g['egc'] + dgap,
        'eps_from_nc':   g['nc'] ** 2 + 0.737,
        'eps_from_ion':  g['nc'] ** 2 + ionic,
        'nc_from_eps':   np.sqrt(np.clip(g['eps'] - 0.652, 1.0, None)),
        'nc_from_ion':   np.sqrt(np.clip(g['eps'] - ionic, 1.0, None)),
        'chi_hat':       chi, 'ionic_hat': ionic, 'dgap_hat': dgap,
        'mulliken':      (g['ei'] + g['eea']) / 2.0,
        'hardness':      (g['ei'] - g['eea']) / 2.0,
        'moss':          np.clip(g['egc'], 0.05, None) ** -0.25,
        'inv_egb':       1.0 / np.clip(g['egb'], 0.05, None),
        'n_from_moss':   (54.5 / np.clip(g['egc'], 0.05, None)) ** 0.25,
        'nsq':           g['nc'] ** 2,
        'sqrt_eps':      np.sqrt(np.clip(g['eps'], 0.0, None)),
    }
    for k, v in rec.items():
        cols.append(v); names.append(k)
    # how many official labels this structure actually carries
    cols.append(ob.sum(1)); names.append('n_obs')
    B = np.column_stack(cols).astype(np.float64)
    bmed = np.nanmedian(B, axis=0)
    bmed[~np.isfinite(bmed)] = 0.0
    kk = np.where(~np.isfinite(B)); B[kk] = np.take(bmed, kk[1])
    return B.astype(np.float32), names

# =====================================================================
# Stages 2 and 3 — cross-property stack, iterated
# =====================================================================
cur, curD = P1.copy(), P1D.copy()
history = []
for rnd in (2, 3):
    log("=" * 78); log(f"STAGE {rnd} — cross-property stack (round {rnd-1})"); log("=" * 78)
    nxt = cur.copy(); nxtD = curD.copy()
    scores = {}
    for j, t in enumerate(T):
        rows = pool[t]; y = L[rows, j]
        other = np.setdiff1d(np.arange(NS), rows)
        Ball, names = cp_block(cur, curD, exclude=[j])
        # also give the stack the running estimate for this very target
        Ball = np.hstack([Ball, cur[:, j:j+1].astype(np.float32),
                          P1[:, j:j+1].astype(np.float32)])
        Btr, Bte = Ball[rows], Ball[other]
        try:
            bo, bt, _, _, _ = fit_predict_zoo(rows, y, extra_tr=Btr, extra_te=Bte,
                                              te_rows=other, tag=f"S{rnd}:{t}", mt_col=j)
        except Exception as exc:                 # never lose the run over one target
            log(f"      !! {t} stage {rnd} failed ({type(exc).__name__}: {exc}); "
                f"keeping previous stage")
            scores[t] = r2(y, cur[rows, j])
            continue
        # keep the better of stage-1 and the stack, judged on OOF
        prev = r2(y, cur[rows, j]); new = r2(y, bo)
        if new >= prev:
            nxt[rows, j] = bo; nxt[other, j] = bt
        else:
            log(f"      {t}: stack {new:.6f} < prev {prev:.6f} -> keeping previous")
        scores[t] = max(new, prev)
        log(f"  STAGE{rnd} {t:4s} OOF R2 = {scores[t]:.6f}   (was {prev:.6f})")
    for j, d in enumerate(DERIVED):
        rows = dpool[d]; y = D[rows, j]
        other = np.setdiff1d(np.arange(NS), rows)
        # a derived coordinate must not see the labels it is built from
        drop = [ti[c] for c in {'chi': ['ei', 'eea'], 'ionic': ['eps', 'nc'],
                                'dgap': ['egb', 'egc']}[d]]
        Ball, _ = cp_block(cur, curD, exclude=drop)
        Btr, Bte = Ball[rows], Ball[other]
        try:
            bo, bt, _, _, _ = fit_predict_zoo(rows, y, extra_tr=Btr, extra_te=Bte,
                                              te_rows=other, tag=f"S{rnd}:{d}")
        except Exception as exc:
            log(f"      !! {d} stage {rnd} failed ({type(exc).__name__}: {exc})")
            continue
        prev = r2(y, curD[rows, j])
        if r2(y, bo) >= prev:
            nxtD[rows, j] = bo; nxtD[other, j] = bt
        log(f"  STAGE{rnd} {d:6s} OOF R2 = {max(r2(y, bo), prev):.6f}   (was {prev:.6f})")
    cur, curD = nxt, nxtD
    history.append(scores)
    log(f"  ---- round mean OOF R2 = {np.mean([scores[t] for t in T]):.6f}")

np.save(f"{OUT}/PFINAL.npy", cur); np.save(f"{OUT}/PFINALD.npy", curD)

# =====================================================================
# Stage 4/5 — constraints and exact archive override, then write submission
# =====================================================================
log("=" * 78); log("STAGE 4/5 — constraints, archive override, submission"); log("=" * 78)
pred_rows = []
per_target_est = {}
for j, t in enumerate(T):
    fi = test_fi[t]; ids = test_id[t]
    p = cur[fi, j].copy()
    if t in ('tg', 'egc'):
        av = A[fi, j]
        hit = np.isfinite(av)
        p[hit] = av[hit]
        log(f"  {t}: archive exact override on {hit.sum()}/{len(fi)} rows ({100*hit.mean():.1f}%)")
    pred_rows.append(pd.DataFrame({'id': ids, 'target': p, 'target_type': t, 'fi': fi}))
sub = pd.concat(pred_rows, ignore_index=True)

# EPS >= Nc^2 joint projection on the test predictions
e_mask = sub.target_type == 'eps'; n_mask = sub.target_type == 'nc'
efi = dict(zip(sub.loc[e_mask, 'fi'], sub.loc[e_mask].index))
nfi = dict(zip(sub.loc[n_mask, 'fi'], sub.loc[n_mask].index))
nviol = 0
for f_, ei_ in efi.items():
    eps_v = sub.at[ei_, 'target']
    nc_v = L[f_, ti['nc']] if LOBS[f_, ti['nc']] else (sub.at[nfi[f_], 'target'] if f_ in nfi else cur[f_, ti['nc']])
    if eps_v < nc_v ** 2 + 0.024:
        nviol += 1
        sub.at[ei_, 'target'] = nc_v ** 2 + 0.024
log(f"  EPS >= Nc^2 projection applied to {nviol} test rows")
# clip every target into its official observed range
for j, t in enumerate(T):
    lo, hi = np.nanmin(L[:, j]), np.nanmax(L[:, j])
    m_ = sub.target_type == t
    span = hi - lo
    sub.loc[m_, 'target'] = sub.loc[m_, 'target'].clip(lo - 0.02 * span, hi + 0.02 * span)

sub = sub.sort_values('id')
assert len(sub) == 4940 and sub['id'].is_unique and np.isfinite(sub['target']).all()
sub[['id', 'target']].to_csv(f"{OUT}/submission.csv", index=False)
log(f"  wrote {OUT}/submission.csv  rows={len(sub)}")

# =====================================================================
# honest test-score estimate
# =====================================================================
log("=" * 78); log("ESTIMATED TEST SCORE"); log("=" * 78)
est = {}
for j, t in enumerate(T):
    rows = pool[t]; y = L[rows, j]; p = cur[rows, j]
    oof = r2(y, p)
    if t in ('tg', 'egc'):
        # test rows split into archive-covered (exact, zero error) and modelled
        fi = test_fi[t]
        cov = np.isfinite(A[fi, j])
        # model quality measured only on pool rows that carry no archive label,
        # i.e. the same population as the uncovered test rows
        unc = ~np.isfinite(A[rows, j])
        if unc.sum() > 30:
            sse_per_row = ((y[unc] - p[unc]) ** 2).mean()
        else:
            sse_per_row = ((y - p) ** 2).mean()
        var_test = np.var(y)          # best available proxy for test variance
        e = 1.0 - (1 - cov.mean()) * sse_per_row / var_test
        log(f"  {t:4s} OOF={oof:.6f}  archive-covered={100*cov.mean():.1f}%  "
            f"uncovered-only R2={r2(y[unc], p[unc]):.6f}  ->  est test R2={e:.6f}")
    else:
        e = oof
        log(f"  {t:4s} OOF={oof:.6f}  ->  est test R2={e:.6f}")
    est[t] = e
log(f"  ESTIMATED MEAN TEST R2 = {np.mean([est[t] for t in T]):.6f}")

# The same OOF-vs-test comparison is available for the existing incumbent C050, and
# its OOF was optimistic on every weak target.  Subtracting that measured per-target
# gap gives a conservative second estimate.  Tg/Egc are excluded because their test
# numbers are dominated by the exact archive override and are already handled above.
C050_OOF = {'tg': 0.9088768072, 'egc': 0.9115043879, 'egb': 0.9221467344,
            'ei': 0.8454440895, 'eea': 0.9008357940, 'nc': 0.8397322432,
            'eps': 0.7835054390}
C050_TEST = {'tg': 0.9539044922, 'egc': 0.9572538004, 'egb': 0.8990616896,
             'ei': 0.7568828960, 'eea': 0.8681479551, 'nc': 0.8295703105,
             'eps': 0.7698645559}
log("-" * 78)
log("  gap-calibrated estimate (subtracting C050's measured OOF-minus-test gap)")
cal = {}
for t in T:
    if t in ('tg', 'egc'):
        cal[t] = est[t]
        log(f"  {t:4s} {cal[t]:.6f}   (archive-override arithmetic, no gap applied)")
    else:
        gap = C050_OOF[t] - C050_TEST[t]
        cal[t] = est[t] - gap
        log(f"  {t:4s} OOF-based {est[t]:.6f}  - C050 gap {gap:.6f}  =  {cal[t]:.6f}")
log(f"  GAP-CALIBRATED MEAN = {np.mean([cal[t] for t in T]):.6f}")
log("-" * 78)
log("  per-target comparison vs the incumbent")
for t in T:
    log(f"  {t:4s} est {est[t]:.4f} (cal {cal[t]:.4f})  vs C050 test {C050_TEST[t]:.4f}"
        f"   delta {est[t]-C050_TEST[t]:+.4f} (cal {cal[t]-C050_TEST[t]:+.4f})")
log(f"  MEAN  est {np.mean([est[t] for t in T]):.4f} "
    f"(cal {np.mean([cal[t] for t in T]):.4f})  vs C050 test 0.8621")

json.dump({'per_target_est': est, 'mean': float(np.mean([est[t] for t in T])),
           'per_target_gap_calibrated': cal,
           'mean_gap_calibrated': float(np.mean([cal[t] for t in T])),
           'oof': {t: float(r2(L[pool[t], j], cur[pool[t], j])) for j, t in enumerate(T)}},
          open(f"{OUT}/estimate.json", 'w'), indent=2)
log("DONE")
