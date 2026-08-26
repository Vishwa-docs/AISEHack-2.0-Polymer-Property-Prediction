"""Build and cache the structure feature matrix for all unique canonical polymers
appearing in round-2 train/test plus the official archive.

Features (all computed from official SMILES only, no external data, no pretrained weights):
  A. RDKit descriptors (full Descriptors list, sanitized)
  B. Morgan count fingerprints, radius 1/2/3, folded (hashed) counts
  C. MACCS keys
  D. Atom-pair + topological-torsion hashed counts
  E. Wildcard/backbone topology features specific to polymer SMILES (2 '*' per row)
  F. Oligomer (dimer) descriptors: connect the two '*' ends to build a periodic-ish
     repeat, then recompute a small descriptor block
  G. Character n-gram TF-IDF on canonical SMILES (SVD compressed)
"""
import os, pickle, time, sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys, rdFingerprintGenerator, Crippen, GraphDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
t0 = time.time()

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
arch = pd.read_csv(f"{BASE}/archive/train.csv")
archt = pd.read_csv(f"{BASE}/archive/test.csv")

all_smiles = pd.unique(pd.concat([
    train['smiles'], test['smiles'], arch['smiles'], archt['smiles']
], ignore_index=True))
print("unique raw smiles:", len(all_smiles))

def canon(s):
    m = Chem.MolFromSmiles(s)
    return (Chem.MolToSmiles(m) if m is not None else None), m

canon_map = {}
mols = {}
for s in all_smiles:
    c, m = canon(s)
    canon_map[s] = c
    if c is not None and c not in mols:
        mols[c] = m
print("unique canonical:", len(mols), "elapsed", round(time.time()-t0,1))

canon_list = sorted(mols.keys())
idx = {c: i for i, c in enumerate(canon_list)}

# ---------- A. RDKit descriptors ----------
desc_names = [n for n, _ in Descriptors._descList]
# drop Ipc (explodes numerically) -- keep log version manually
desc_fns = [(n, f) for n, f in Descriptors._descList if n != 'Ipc']
print("n descriptors:", len(desc_fns))

# ---------- helper: build oligomer (dimer) from a 2-star polymer SMILES ----------
def make_oligomer(smi, n=2):
    """Chain n copies of the repeat unit, capping terminal stars with H."""
    try:
        parts = []
        for k in range(n):
            s = smi
            parts.append(s)
        # use RDKit RWMol joining: replace * atoms
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(stars) != 2: return None
        combo = mol
        for k in range(n - 1):
            nxt = Chem.MolFromSmiles(smi)
            off = combo.GetNumAtoms()
            merged = Chem.RWMol(Chem.CombineMols(combo, nxt))
            cstars = [a.GetIdx() for a in merged.GetAtoms() if a.GetAtomicNum() == 0]
            # last star of first fragment, first star of second fragment
            left = [i for i in cstars if i < off]
            right = [i for i in cstars if i >= off]
            if not left or not right: return None
            a_star, b_star = left[-1], right[0]
            a_nb = [nb.GetIdx() for nb in merged.GetAtomWithIdx(a_star).GetNeighbors()]
            b_nb = [nb.GetIdx() for nb in merged.GetAtomWithIdx(b_star).GetNeighbors()]
            if not a_nb or not b_nb: return None
            merged.AddBond(a_nb[0], b_nb[0], Chem.BondType.SINGLE)
            for i in sorted([a_star, b_star], reverse=True):
                merged.RemoveAtom(i)
            combo = merged.GetMol()
            Chem.SanitizeMol(combo)
        return combo
    except Exception:
        return None

def cap_h(smi):
    """Replace * with H (i.e. remove them) to get the capped monomer."""
    try:
        m = Chem.MolFromSmiles(smi.replace('*', '[H]'))
        if m is None:
            m = Chem.MolFromSmiles(smi)
            if m is None: return None
            rw = Chem.RWMol(m)
            for i in sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
                rw.RemoveAtom(i)
            m = rw.GetMol(); Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def make_ring(smi):
    """Close the two stars onto each other -> cyclic 'periodic' surrogate."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        rw = Chem.RWMol(m)
        stars = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        if len(stars) != 2: return None
        nbs = []
        for s in stars:
            n = [x.GetIdx() for x in rw.GetAtomWithIdx(s).GetNeighbors()]
            if not n: return None
            nbs.append(n[0])
        if nbs[0] == nbs[1]:
            return None
        rw.AddBond(nbs[0], nbs[1], Chem.BondType.SINGLE)
        for i in sorted(stars, reverse=True):
            rw.RemoveAtom(i)
        m2 = rw.GetMol(); Chem.SanitizeMol(m2)
        return m2
    except Exception:
        return None

# ---------- compute ----------
from multiprocessing import Pool

SMALL_DESC = ['MolWt','MolLogP','TPSA','NumRotatableBonds','FractionCSP3','NumHAcceptors',
              'NumHDonors','RingCount','NumAromaticRings','HeavyAtomCount','BertzCT',
              'BalabanJ','Chi0v','Chi1v','Chi2v','Chi3v','Chi4v','Kappa1','Kappa2','Kappa3',
              'HallKierAlpha','LabuteASA','MolMR','NumAliphaticRings','NumSaturatedRings',
              'NHOHCount','NOCount','NumHeteroatoms','qed','MaxPartialCharge','MinPartialCharge']
small_fns = [(n, f) for n, f in Descriptors._descList if n in SMALL_DESC]

def featurize_one(c):
    m = Chem.MolFromSmiles(c)
    out = {}
    if m is None:
        return None
    # A. descriptors
    vals = []
    for n, f in desc_fns:
        try:
            v = f(m)
        except Exception:
            v = np.nan
        vals.append(v)
    out['desc'] = np.array(vals, dtype=np.float64)
    # log-Ipc
    try:
        ipc = GraphDescriptors.Ipc(m, avg=True)
    except Exception:
        ipc = np.nan
    out['ipc'] = ipc
    # E. polymer topology features
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    nstar = len(stars)
    natoms = m.GetNumAtoms()
    extra = [nstar, natoms]
    if nstar == 2:
        try:
            d = Chem.GetDistanceMatrix(m)
            backbone_len = d[stars[0], stars[1]]
        except Exception:
            backbone_len = np.nan
        extra.append(backbone_len)
        extra.append(backbone_len / max(1, natoms))
        # neighbours of stars: element, aromatic, degree
        for s in stars[:2]:
            nb = list(m.GetAtomWithIdx(s).GetNeighbors())
            if nb:
                a = nb[0]
                extra += [a.GetAtomicNum(), float(a.GetIsAromatic()), a.GetDegree(), a.GetTotalNumHs()]
            else:
                extra += [0, 0, 0, 0]
    else:
        extra += [np.nan, np.nan] + [0] * 8
    # element counts
    from collections import Counter
    cnt = Counter(a.GetSymbol() for a in m.GetAtoms())
    for el in ['C','N','O','S','F','Cl','Br','I','Si','P','B','Se','Ge','Sn','Fe','*']:
        extra.append(cnt.get(el, 0))
    tot = max(1, natoms)
    for el in ['C','N','O','S','F','Cl','Si']:
        extra.append(cnt.get(el, 0) / tot)
    # bond type counts
    bt = Counter(str(b.GetBondType()) for b in m.GetBonds())
    for t in ['SINGLE','DOUBLE','TRIPLE','AROMATIC']:
        extra.append(bt.get(t, 0))
    nb_ = max(1, m.GetNumBonds())
    for t in ['SINGLE','DOUBLE','TRIPLE','AROMATIC']:
        extra.append(bt.get(t, 0) / nb_)
    # ring info
    ri = m.GetRingInfo()
    extra.append(ri.NumRings())
    sizes = [len(r) for r in ri.AtomRings()]
    extra += [len([s for s in sizes if s == k]) for k in (3,4,5,6,7)]
    n_arom_atoms = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    extra.append(n_arom_atoms)
    extra.append(n_arom_atoms / tot)
    # conjugation
    n_conj = sum(1 for b in m.GetBonds() if b.GetIsConjugated())
    extra.append(n_conj)
    extra.append(n_conj / nb_)
    # formal charges, radicals
    extra.append(sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()))
    extra.append(sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()))
    # sp/sp2/sp3
    hyb = Counter(str(a.GetHybridization()) for a in m.GetAtoms())
    for h in ['SP','SP2','SP3','SP3D','SP3D2']:
        extra.append(hyb.get(h, 0))
        extra.append(hyb.get(h, 0) / tot)
    out['extra'] = np.array(extra, dtype=np.float64)

    # B. Morgan count fps
    fps = []
    for r, nb2 in [(1, 1024), (2, 2048), (3, 1024)]:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=nb2, countSimulation=False)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        fps.append(fp)
    out['morgan'] = np.concatenate(fps)
    # binary morgan r2 2048 for tanimoto
    gen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out['morgan_bin'] = np.packbits(gen2.GetFingerprintAsNumPy(m).astype(np.uint8))

    # C. MACCS
    out['maccs'] = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)
    # D. atom pair + torsion (hashed counts)
    apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    out['ap'] = apgen.GetCountFingerprintAsNumPy(m).astype(np.float32)
    out['tt'] = ttgen.GetCountFingerprintAsNumPy(m).astype(np.float32)
    # rdkit path fp
    rkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024, maxPath=6)
    out['rk'] = rkgen.GetCountFingerprintAsNumPy(m).astype(np.float32)

    # F. oligomer / capped / ring surrogates
    sub = []
    for maker in (cap_h, make_ring, lambda s: make_oligomer(s, 2), lambda s: make_oligomer(s, 3)):
        mm = maker(c)
        if mm is None:
            sub += [np.nan] * (len(small_fns) + 2)
        else:
            for n, f in small_fns:
                try: sub.append(f(mm))
                except Exception: sub.append(np.nan)
            sub.append(mm.GetNumAtoms())
            try:
                sub.append(Chem.GetDistanceMatrix(mm).max())
            except Exception:
                sub.append(np.nan)
    out['oligo'] = np.array(sub, dtype=np.float64)

    # scaffold
    try:
        out['scaffold'] = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        out['scaffold'] = ''
    return out


if __name__ == '__main__':
    with Pool(20) as p:
        res = p.map(featurize_one, canon_list, chunksize=32)
    print("featurized", len(res), "elapsed", round(time.time()-t0, 1))

    ok = [r is not None for r in res]
    print("failures:", ok.count(False))
    blocks = {}
    for key in ['desc','extra','morgan','maccs','ap','tt','rk','oligo']:
        dim = len(next(r[key] for r in res if r is not None))
        M = np.full((len(res), dim), np.nan, dtype=np.float32)
        for i, r in enumerate(res):
            if r is not None:
                M[i] = r[key]
        blocks[key] = M
        print(key, M.shape)
    ipc = np.array([r['ipc'] if r is not None else np.nan for r in res], dtype=np.float64)
    blocks['ipc'] = np.log1p(np.abs(ipc)).reshape(-1, 1).astype(np.float32)
    mb = np.stack([r['morgan_bin'] if r is not None else np.zeros(256, np.uint8) for r in res])
    scaffolds = [r['scaffold'] if r is not None else '' for r in res]

    desc_cols = [n for n, _ in desc_fns]
    with open(f"{SCR}/features.pkl", "wb") as f:
        pickle.dump(dict(canon_list=canon_list, idx=idx, canon_map=canon_map,
                         blocks=blocks, morgan_bin=mb, scaffolds=scaffolds,
                         desc_cols=desc_cols), f, protocol=4)
    print("saved. total elapsed", round(time.time()-t0, 1))
