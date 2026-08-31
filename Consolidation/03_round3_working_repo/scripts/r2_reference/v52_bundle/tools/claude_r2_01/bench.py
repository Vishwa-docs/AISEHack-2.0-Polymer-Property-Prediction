"""Time each zoo member's fit and predict at the sizes the pipeline actually uses."""
import pickle, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
import lightgbm as lgb

SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
t0 = time.time()
F = pickle.load(open(f"{SCR}/features.pkl", "rb"))
P = pickle.load(open(f"{SCR}/physics.pkl", "rb"))
G = pickle.load(open(f"{SCR}/pgfp.pkl", "rb"))
b = F['blocks']
def clean(M):
    M = np.asarray(M, np.float64); M[~np.isfinite(M)] = np.nan
    return np.clip(M, -1e10, 1e10)
PG = G['M']; PGN = PG / np.maximum(1.0, PG.sum(1, keepdims=True))
DENSE = np.hstack([clean(b['desc']), clean(b['extra']), clean(b['oligo']),
                   clean(b['ipc']), clean(P['M']), clean(PGN), clean(G['morph'])])
keep = np.array([np.nanstd(DENSE[:, j]) > 1e-12 and np.isfinite(DENSE[:, j]).mean() > .3
                 for j in range(DENSE.shape[1])])
DENSE = DENSE[:, keep]
md = np.nanmedian(DENSE, 0); ii = np.where(np.isnan(DENSE)); DENSE[ii] = np.take(md, ii[1])
DENSE = DENSE.astype(np.float32)
SPARSE = np.hstack([np.log1p(b['morgan']), b['maccs'], np.log1p(b['ap']),
                    np.log1p(b['tt']), np.log1p(b['rk']), np.log1p(PG)]).astype(np.float32)
SPARSE = SPARSE[:, (SPARSE != 0).sum(0) >= 4]
print("load+build", round(time.time()-t0, 1), DENSE.shape, SPARSE.shape)

t = time.time(); SVD = TruncatedSVD(192, random_state=0).fit_transform(SPARSE); print("SVD", round(time.time()-t,1))
TREEX = np.hstack([DENSE, SVD.astype(np.float32), b['maccs']]).astype(np.float32)
print("TREEX", TREEX.shape)

NS = len(DENSE); N = 160
rng = np.random.RandomState(0)
rows = rng.choice(NS, N, replace=False)
other = np.setdiff1d(np.arange(NS), rows)
y = rng.randn(N)

def timeit(name, fn):
    t = time.time(); fn(); print(f"  {name:34s} {time.time()-t:7.2f}s")

print(f"\n=== single fit+predict, n_train={N}, n_predict={len(other)} ===")
timeit("ExtraTrees(500,mf=.30) fit+pred", lambda: ExtraTreesRegressor(
    n_estimators=500, max_features=0.30, n_jobs=22, random_state=0
).fit(TREEX[rows], y).predict(TREEX[other]))
timeit("ExtraTrees fit only", lambda: ExtraTreesRegressor(
    n_estimators=500, max_features=0.30, n_jobs=22, random_state=0).fit(TREEX[rows], y))
et = ExtraTreesRegressor(n_estimators=500, max_features=0.30, n_jobs=22,
                         random_state=0).fit(TREEX[rows], y)
timeit("ExtraTrees predict 8830", lambda: et.predict(TREEX[other]))
timeit("ExtraTrees predict 160", lambda: et.predict(TREEX[rows]))

timeit("LGBM(900) fit+pred", lambda: lgb.LGBMRegressor(
    n_estimators=900, learning_rate=.03, num_leaves=31, colsample_bytree=.5,
    subsample=.8, subsample_freq=1, n_jobs=22, verbose=-1, random_state=0
).fit(TREEX[rows], y).predict(TREEX[other]))
lg = lgb.LGBMRegressor(n_estimators=900, learning_rate=.03, num_leaves=31,
                       colsample_bytree=.5, subsample=.8, subsample_freq=1,
                       n_jobs=22, verbose=-1, random_state=0).fit(TREEX[rows], y)
timeit("LGBM predict 8830", lambda: lg.predict(TREEX[other]))

def ridge_old():
    sc = StandardScaler().fit(SPARSE[rows])
    m = Ridge(alpha=6).fit(sc.transform(SPARSE[rows]), y)
    return m.predict(sc.transform(SPARSE[other]))
def ridge_new():
    X = SPARSE[rows]
    mu = X.mean(0, dtype=np.float64); sd = X.std(0, dtype=np.float64); sd[sd < 1e-12] = 1
    m = Ridge(alpha=6).fit((X - mu) / sd, y)
    w = m.coef_ / sd; bb = m.intercept_ - mu @ w
    out = np.empty(len(other))
    for s in range(0, len(other), 2048):
        out[s:s+2048] = SPARSE[other][s:s+2048] @ w + bb
    return out
timeit("ridge_sparse OLD (scaler.transform)", ridge_old)
timeit("ridge_sparse NEW (folded coef)", ridge_new)

MB = np.unpackbits(F['morgan_bin'], axis=1).astype(np.float32); POP = MB.sum(1)
def tan():
    inter = MB[other] @ MB[rows].T
    return inter / (POP[other][:, None] + POP[rows][None, :] - inter + 1e-9)
timeit("tanimoto 8830x160", tan)
print("\ntotal", round(time.time()-t0, 1))
