"""Follow-up EDA: per-test-row label availability, archive lookup potential, label consistency."""
import pandas as pd, numpy as np, pickle
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
arch = pd.read_csv(f"{BASE}/archive/train.csv")

def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None

cache = {}
def canon_c(s):
    if s not in cache: cache[s] = canon(s)
    return cache[s]

train['canon'] = [canon_c(s) for s in train['smiles']]
test['canon'] = [canon_c(s) for s in test['smiles']]
arch_smiles_col = [c for c in arch.columns if 'smile' in c.lower()][0]
arch['canon'] = [canon_c(s) for s in arch[arch_smiles_col]]

piv = train.pivot_table(index='canon', columns='target_type', values='target', aggfunc='mean')
targets = ['tg','egc','egb','ei','eea','nc','eps']

print("=== per test-target: availability pattern of other-property train labels ===")
for t in targets:
    te = test[test.target_type==t]
    n = len(te)
    avail = piv.reindex(te['canon'].values)
    counts = {o: int(avail[o].notna().sum()) for o in targets if o in avail.columns}
    print(f"\ntest {t} (n={n}): rows with train label per property: {counts}")
    # key combos
    if t=='ei':
        both = (avail['eea'].notna() & avail['egc'].notna()).sum()
        print(f"  ei rows with BOTH eea+egc: {both} ({100*both/n:.1f}%)")
        both2 = (avail['eea'].notna() & avail['egb'].notna()).sum()
        print(f"  ei rows with BOTH eea+egb: {both2}")
    if t=='eps':
        print(f"  eps rows with nc: {int(avail['nc'].notna().sum())} ({100*avail['nc'].notna().mean():.1f}%)")
    if t=='nc':
        print(f"  nc rows with eps: {int(avail['eps'].notna().sum())} ({100*avail['eps'].notna().mean():.1f}%)")
    if t=='egb':
        print(f"  egb rows with egc: {int(avail['egc'].notna().sum())} ({100*avail['egc'].notna().mean():.1f}%)")
    if t=='eea':
        both = (avail['ei'].notna() & avail['egc'].notna()).sum()
        print(f"  eea rows with BOTH ei+egc: {both} ({100*both/n:.1f}%)")

print("\n=== archive lookup potential per test target ===")
arch_piv = arch.pivot_table(index='canon', columns='target_type', values='target', aggfunc='mean')
for t in targets:
    te = test[test.target_type==t]
    if t in arch_piv.columns:
        hits = arch_piv[t].reindex(te['canon'].values).notna().sum()
        print(f"test {t}: {hits}/{len(te)} rows have SAME-target label in archive")
# cross: archive egc for other targets
for t in targets:
    te = test[test.target_type==t]
    av = arch_piv.reindex(te['canon'].values)
    c = {o:int(av[o].notna().sum()) for o in arch_piv.columns}
    print(f"test {t}: archive labels available: {c}")

print("\n=== label consistency: round2 train vs archive on shared (structure,target) ===")
for t in ['tg','egc']:
    a = arch_piv[t].dropna()
    b = piv[t].dropna() if t in piv.columns else pd.Series(dtype=float)
    common = a.index.intersection(b.index)
    if len(common):
        diff = (a.loc[common]-b.loc[common])
        r = np.corrcoef(a.loc[common], b.loc[common])[0,1]
        print(f"{t}: {len(common)} shared structures, corr={r:.6f}, |diff| mean={diff.abs().mean():.4f}, max={diff.abs().max():.4f}, exact-equal={(diff.abs()<1e-9).mean():.3f}")

print("\n=== do archive-only structures show up in test? (train-removal hypothesis) ===")
r2_structs = set(train['canon'].dropna())
arch_only = set(arch_piv.index) - r2_structs
for t in targets:
    te = test[test.target_type==t]
    in_arch_only = te['canon'].isin(arch_only).sum()
    print(f"test {t}: {in_arch_only}/{len(te)} rows in archive-only structures")

# for test tg rows in archive-only with tg label: these were R1 train rows removed in R2!
te_tg = test[test.target_type=='tg']
m = te_tg['canon'].isin(arch_only) & arch_piv['tg'].reindex(te_tg['canon'].values).notna().values
print(f"\ntest tg rows in archive-only WITH archive tg label: {m.sum()}")
te_egc = test[test.target_type=='egc']
m2 = te_egc['canon'].isin(arch_only) & arch_piv['egc'].reindex(te_egc['canon'].values).notna().values
print(f"test egc rows in archive-only WITH archive egc label: {m2.sum()}")

print("\n=== R1 archive test overlap with R2 test ===")
archt = pd.read_csv(f"{BASE}/archive/test.csv")
acol = [c for c in archt.columns if 'smile' in c.lower()][0]
archt['canon'] = [canon_c(s) for s in archt[acol]]
at_structs = set(archt['canon'].dropna())
for t in targets:
    te = test[test.target_type==t]
    print(f"test {t}: {te['canon'].isin(at_structs).sum()}/{len(te)} in R1-test structures")

print("\n=== duplicate policy check: test tg rows w/ same-canon in train tg ===")
tr_tg = train[train.target_type=='tg']
ov = set(te_tg['canon']) & set(tr_tg['canon'])
print("overlap structs:", len(ov))

# nc^2 vs eps direction for imputation: fit isotonic-ish linear both ways
mask = piv['eps'].notna() & piv['nc'].notna()
eps, nc = piv.loc[mask,'eps'], piv.loc[mask,'nc']
A = np.c_[nc**2, nc, np.ones(len(nc))]
coef,_,_,_ = np.linalg.lstsq(A, eps, rcond=None)
pred = A@coef
r2 = 1-((eps-pred)**2).sum()/((eps-eps.mean())**2).sum()
print(f"\neps ~ quad(nc): R2={r2:.4f} coef={coef.round(4)}")
A2 = np.c_[eps, np.sqrt(eps.clip(lower=0)), np.ones(len(eps))]
coef2,_,_,_ = np.linalg.lstsq(A2, nc, rcond=None)
pred2 = A2@coef2
r22 = 1-((nc-pred2)**2).sum()/((nc-nc.mean())**2).sum()
print(f"nc ~ (eps, sqrt(eps)): R2={r22:.4f} coef={coef2.round(4)}")

with open(f"{SCR}/canon_cache.pkl","wb") as f:
    pickle.dump(cache, f)
piv.to_csv(f"{SCR}/train_pivot.csv")
arch_piv.to_csv(f"{SCR}/arch_pivot.csv")
print("\nDONE")
