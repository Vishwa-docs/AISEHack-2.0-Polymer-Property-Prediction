"""Do test rows match train/archive rows of the SAME target under alternate
canonical forms of the same infinite polymer? (ring-closure key, stereo-stripped
key, heavy-atom formula+scaffold key)"""
import pandas as pd, numpy as np, pickle
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
T = ['tg','egc','egb','ei','eea','nc','eps']
train = pd.read_csv(f"{BASE}/train.csv"); test = pd.read_csv(f"{BASE}/test.csv")
arch = pd.read_csv(f"{BASE}/archive/train.csv")

def ring_key(smi):
    """Close the two * onto each other: invariant to which bond of the backbone
    the repeat unit was cut at, so two different repeat-unit phasings of the same
    polymer collapse to one key."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        rw = Chem.RWMol(m)
        st = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        if len(st) != 2: return None
        nb = []
        for s in st:
            n = [x.GetIdx() for x in rw.GetAtomWithIdx(s).GetNeighbors()]
            if not n: return None
            nb.append(n[0])
        if nb[0] == nb[1]: return None
        rw.AddBond(nb[0], nb[1], Chem.BondType.SINGLE)
        for i in sorted(st, reverse=True): rw.RemoveAtom(i)
        mm = rw.GetMol(); Chem.SanitizeMol(mm)
        return Chem.MolToSmiles(mm)
    except Exception:
        return None

def nostereo_key(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None

cache = {}
def K(fn, s):
    k = (fn.__name__, s)
    if k not in cache: cache[k] = fn(s)
    return cache[k]

for name, fn in [('canonical', canon), ('ring-closure', ring_key), ('no-stereo', nostereo_key)]:
    for df in (train, test, arch):
        df[name] = [K(fn, s) for s in df['smiles']]
    print(f"\n===== key: {name} =====")
    lab = pd.concat([train[[name, 'target_type', 'target']],
                     arch[[name, 'target_type', 'target']]], ignore_index=True)
    lab = lab.dropna(subset=[name])
    tbl = lab.groupby([name, 'target_type'])['target'].agg(['mean', 'std', 'count'])
    tot = 0
    for t in T:
        te = test[test.target_type == t]
        try:
            sub = tbl.xs(t, level='target_type')
        except KeyError:
            print(f"  {t:4s}: no train labels"); continue
        hit = te[name].isin(sub.index)
        # how consistent are the labels inside a multi-row key group?
        multi = sub[sub['count'] > 1]
        incons = (multi['std'] > 1e-6).sum() if len(multi) else 0
        tot += int(hit.sum())
        print(f"  {t:4s}: {hit.sum():5d}/{len(te):5d} test rows matched "
              f"({100*hit.mean():5.1f}%)   groups>1={len(multi)} inconsistent={incons}")
    print(f"  TOTAL matched test rows: {tot}/4940")
