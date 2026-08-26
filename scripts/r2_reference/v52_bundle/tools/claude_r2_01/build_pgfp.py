"""Polymer Genome style atomic fingerprint, reimplemented from scratch with RDKit.

Huan et al. PRB 92 (2015) / Mannodi-Kanakkithodi et al. Chem Mater 29, 9001 (2017):
occurrence counts of fixed atomic triples with explicit coordination, written
`O1-C3-C4` = 1-fold-coordinated O bonded to 3-fold C bonded to 4-fold C.  This
371-dim block is the backbone of the Gaussian-process Polymer Genome model that
still reports R2 0.90 / 0.91 / 0.90 on Egc / Egb / Eea — the strongest classical
baseline on exactly these six DFT targets.

Also included:
  * the singles and pairs of the same alphabet;
  * a morphological block (ring-ring topological distances, side-chain fractions);
  * the same triples restricted to the BACKBONE path, since backbone chemistry and
    pendant chemistry act differently on both bandgap and Tg.
"""
import pickle, time
from collections import Counter
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"


def atom_type(a):
    if a.GetAtomicNum() == 0:
        return "X"
    return f"{a.GetSymbol()}{a.GetDegree()}"


def keys_for(c):
    m = Chem.MolFromSmiles(c)
    if m is None:
        return Counter(), Counter(), []
    at = {a.GetIdx(): atom_type(a) for a in m.GetAtoms()}
    cnt = Counter()
    for a in m.GetAtoms():
        cnt["S|" + at[a.GetIdx()]] += 1
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p = sorted([at[i], at[j]])
        cnt["P|" + "-".join(p)] += 1
    # triples i-j-k centred on j
    for a in m.GetAtoms():
        j = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        for u in range(len(nb)):
            for v in range(u + 1, len(nb)):
                i, k = nb[u], nb[v]
                ends = sorted([at[i], at[k]])
                cnt["T|" + ends[0] + "-" + at[j] + "-" + ends[1]] += 1
    # backbone-restricted triples
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    bcnt = Counter()
    path = []
    if len(stars) == 2:
        try:
            path = list(Chem.GetShortestPath(m, stars[0], stars[1]))
        except Exception:
            path = []
        pset = set(path)
        for n_ in range(1, len(path) - 1):
            i, j, k = path[n_ - 1], path[n_], path[n_ + 1]
            ends = sorted([at[i], at[k]])
            bcnt["B|" + ends[0] + "-" + at[j] + "-" + ends[1]] += 1
    return cnt, bcnt, path


def morphological(c):
    m = Chem.MolFromSmiles(c)
    if m is None:
        return [np.nan] * MORPH_N
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    n = m.GetNumAtoms()
    try:
        D = Chem.GetDistanceMatrix(m)
    except Exception:
        return [np.nan] * MORPH_N
    # shortest topological distance between distinct rings
    dd = []
    for a_ in range(len(rings)):
        for b_ in range(a_ + 1, len(rings)):
            dd.append(min(D[i, j] for i in rings[a_] for j in rings[b_]))
    ring_atoms = set().union(*rings) if rings else set()
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    side, path = set(), []
    if len(stars) == 2:
        try:
            path = list(Chem.GetShortestPath(m, stars[0], stars[1]))
        except Exception:
            path = []
        side = set(range(n)) - set(path)
    sizes = []
    if path:
        for i in set(path) - set(stars):
            for nb in m.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() in side:
                    st, comp = [nb.GetIdx()], set()
                    while st:
                        u = st.pop()
                        if u in comp or u in path:
                            continue
                        comp.add(u)
                        st.extend(x.GetIdx() for x in m.GetAtomWithIdx(u).GetNeighbors())
                    sizes.append(len(comp))
    sz = sizes or [0]
    return [len(rings),
            float(np.mean(dd)) if dd else -1.0,
            float(np.min(dd)) if dd else -1.0,
            float(np.max(dd)) if dd else -1.0,
            len(ring_atoms) / max(1, n),
            len(side) / max(1, n),
            float(np.max(sz)), float(np.mean(sz)), len(sizes),
            float(np.max(sz)) / max(1, n),
            len(path) / max(1, n),
            float(np.sum([1 for i in ring_atoms if i in set(path)])) / max(1, len(path) or 1)]

MORPH_N = 12


if __name__ == '__main__':
    t0 = time.time()
    F = pickle.load(open(f"{SCR}/features.pkl", "rb"))
    canon_list = F['canon_list']
    from multiprocessing import Pool
    with Pool(20) as p:
        res = p.map(keys_for, canon_list, chunksize=32)
        morph = p.map(morphological, canon_list, chunksize=32)
    doc = Counter()
    for cnt, bcnt, _ in res:
        for k in set(cnt) | set(bcnt):
            doc[k] += 1
    vocab = sorted(k for k, v in doc.items() if v >= 5)
    vi = {k: i for i, k in enumerate(vocab)}
    print(f"vocab {len(vocab)} keys (>=5 structures) of {len(doc)} seen")
    M = np.zeros((len(canon_list), len(vocab)), dtype=np.float32)
    for r, (cnt, bcnt, _) in enumerate(res):
        for k, v in cnt.items():
            if k in vi: M[r, vi[k]] = v
        for k, v in bcnt.items():
            if k in vi: M[r, vi[k]] = v
    MO = np.array(morph, dtype=np.float32)
    print("pgfp", M.shape, "morph", MO.shape, "elapsed", round(time.time() - t0, 1))
    with open(f"{SCR}/pgfp.pkl", "wb") as f:
        pickle.dump(dict(M=M, vocab=vocab, morph=MO), f, protocol=4)

    # quick univariate check against each target
    import pandas as pd
    BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
    tr = pd.read_csv(f"{BASE}/train.csv")
    tr['fi'] = tr['smiles'].map(F['canon_map']).map(F['idx'])
    Mn = M / np.maximum(1.0, M.sum(1, keepdims=True))     # composition-normalised
    for t in ['egc', 'egb', 'ei', 'eea', 'nc', 'eps', 'tg']:
        r = tr[tr.target_type == t]; y = r['target'].values; X = Mn[r['fi'].values]
        sd = X.std(0); ok = sd > 1e-9
        cc = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in np.where(ok)[0]])
        cc = cc[np.isfinite(cc)]
        print(f"  {t:4s}: n={len(r):5d}  best |r| among triples = {cc.max():.3f}, "
              f"#|r|>0.5 = {(cc > 0.5).sum()}")
    print("saved.")
