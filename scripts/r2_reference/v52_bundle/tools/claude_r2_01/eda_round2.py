"""Round 2 deep EDA: overlap, cross-property structure, correlations."""
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
arch = pd.read_csv(f"{BASE}/archive/train.csv")
print("train cols:", list(train.columns), "test cols:", list(test.columns), "archive cols:", list(arch.columns))
print("train rows:", len(train), "test rows:", len(test), "archive rows:", len(arch))

def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None

for df, name in [(train,'train'),(test,'test')]:
    df['canon'] = [canon(s) for s in df['smiles']]
    print(name, "canon failures:", df['canon'].isna().sum())
# archive columns may differ
arch_smiles_col = [c for c in arch.columns if 'smile' in c.lower()][0]
arch['canon'] = [canon(s) for s in arch[arch_smiles_col]]
print("archive canon failures:", arch['canon'].isna().sum())
print("archive head:\n", arch.head(3).to_string())

print("\n=== per-target counts ===")
print("train:\n", train['target_type'].value_counts().to_string())
print("test:\n", test['target_type'].value_counts().to_string())

print("\n=== target distributions ===")
print(train.groupby('target_type')['target'].describe().to_string())

print("\n=== within-train duplicates (same canon, same target_type) ===")
d = train.groupby(['target_type','canon']).size()
print((d>1).groupby(level=0).sum().to_string(), "\nmax dup count:", d.max())
# label spread within duplicate groups
dup = train[train.duplicated(subset=['target_type','canon'], keep=False)]
if len(dup):
    spread = dup.groupby(['target_type','canon'])['target'].agg(['count','std','mean'])
    print("dup groups label std (describe):\n", spread.groupby(level=0)['std'].describe().to_string())

print("\n=== test<->train same-target exact canonical overlap ===")
for t in sorted(test['target_type'].unique()):
    te = set(test.loc[test.target_type==t,'canon'].dropna())
    tr = set(train.loc[train.target_type==t,'canon'].dropna())
    inter = te & tr
    print(f"{t}: test {len(te)} unique, train {len(tr)} unique, overlap {len(inter)} ({100*len(inter)/max(1,len(te)):.1f}% of test)")

print("\n=== test<->train ANY-target structure overlap (cross-property availability) ===")
tr_by_struct = train.groupby('canon')['target_type'].agg(set)
for t in sorted(test['target_type'].unique()):
    te_rows = test[test.target_type==t]
    n = len(te_rows)
    has_any = te_rows['canon'].isin(tr_by_struct.index).sum()
    # has some OTHER property label
    other = 0
    for c in te_rows['canon']:
        s = tr_by_struct.get(c)
        if s is not None and len(s - {t}) > 0: other += 1
    print(f"{t}: {n} rows, {has_any} ({100*has_any/n:.1f}%) structure in train at all, {other} ({100*other/n:.1f}%) with OTHER property label")

print("\n=== archive overlap ===")
if 'target_type' in arch.columns:
    print(arch['target_type'].value_counts().to_string())
else:
    print("archive value cols:", [c for c in arch.columns if c not in (arch_smiles_col,'canon')])
arch_structs = set(arch['canon'].dropna())
for t in sorted(test['target_type'].unique()):
    te = test.loc[test.target_type==t,'canon']
    print(f"{t}: {te.isin(arch_structs).sum()}/{len(te)} test rows in archive structs")
tr_structs = set(train['canon'].dropna())
print("archive structs also in round2 train (any target):", len(arch_structs & tr_structs), "/", len(arch_structs))

print("\n=== cross-property co-label matrix (train pivot) ===")
piv = train.pivot_table(index='canon', columns='target_type', values='target', aggfunc='mean')
present = piv.notna()
targets = list(piv.columns)
co = pd.DataFrame(index=targets, columns=targets, dtype=float)
for a in targets:
    for b in targets:
        co.loc[a,b] = (present[a] & present[b]).sum()
print("co-label counts:\n", co.to_string())

print("\n=== pairwise correlations on co-labeled structures ===")
for i,a in enumerate(targets):
    for b in targets[i+1:]:
        mask = present[a] & present[b]
        n = mask.sum()
        if n >= 20:
            r = np.corrcoef(piv.loc[mask,a], piv.loc[mask,b])[0,1]
            print(f"{a} ~ {b}: n={n}, pearson r={r:.4f}, r2={r*r:.4f}")

# physics: eps vs nc^2
mask = present.get('eps', pd.Series(False,index=piv.index)) & present.get('nc', pd.Series(False,index=piv.index))
if mask.sum() >= 20:
    eps = piv.loc[mask,'eps']; nc = piv.loc[mask,'nc']
    r = np.corrcoef(eps, nc**2)[0,1]
    print(f"\neps ~ nc^2: n={mask.sum()}, r={r:.4f}, r2={r*r:.4f}")
    resid = eps - nc**2
    print("eps - nc^2 describe:\n", resid.describe().to_string())

# ei ~ eea + egc?
for combo in [('ei','eea','egc'), ('ei','eea','egb')]:
    a,b,c = combo
    if all(x in present.columns for x in combo):
        m = present[a] & present[b] & present[c]
        if m.sum() >= 20:
            X = piv.loc[m,[b,c]].values; y = piv.loc[m,a].values
            from numpy.linalg import lstsq
            A = np.c_[X, np.ones(len(X))]
            coef,_,_,_ = lstsq(A,y,rcond=None)
            pred = A@coef
            ss = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
            print(f"{a} ~ {b}+{c}: n={m.sum()}, linear R2={ss:.4f}, coef={coef.round(3)}")

print("\n=== test rows whose structure has SAME-target label in train (exact lookup potential) ===")
same_lookup = 0
for t in sorted(test['target_type'].unique()):
    te_rows = test[test.target_type==t]
    tr_t = set(train.loc[train.target_type==t,'canon'].dropna())
    hits = te_rows['canon'].isin(tr_t).sum()
    same_lookup += hits
print(f"total exact same-target lookup rows: {same_lookup}/{len(test)}")

print("\n=== SMILES char stats / wildcard counts ===")
train['nstar'] = train['smiles'].str.count(r'\*')
test['nstar'] = test['smiles'].str.count(r'\*')
print("train nstar:\n", train['nstar'].value_counts().head().to_string())
print("test nstar:\n", test['nstar'].value_counts().head().to_string())
print("train smiles len describe:\n", train['smiles'].str.len().describe().to_string())

# save canon caches for the pipeline
piv.to_parquet("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad/train_pivot.parquet")
print("\nDONE")
