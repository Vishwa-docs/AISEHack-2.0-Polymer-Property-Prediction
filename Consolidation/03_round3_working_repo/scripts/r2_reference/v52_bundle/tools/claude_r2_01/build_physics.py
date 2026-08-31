"""Physics feature blocks computed from official SMILES only.

1. Periodic tight-binding (Huckel) band structure of the infinite polymer chain.
   The repeat unit graph is the unit cell; the two '*' attachment points define the
   inter-cell bond.  H(k) = H0 + H1 e^{ik} + H1^T e^{-ik} is diagonalised on a
   k-grid, giving band gap, band edges (-> Ei, Eea), band widths and effective
   masses.  This is the mechanism the Egc/Egb/Ei/Eea targets actually measure.

2. Polarisability / refraction block for nc and eps: Lorentz-Lorenz ratio built
   from Crippen molar refractivity and a van der Waals volume, plus Gasteiger
   charge based dipole/polarity descriptors.

3. Backbone / side-chain topology for Tg (Bicerano-style stiffness and free volume).
"""
import pickle, time, warnings
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors, AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"

# ---- Huckel heteroatom parameters (alpha_X = alpha + h_X * beta) ----
H_PARAM = {6:0.0, 7:0.6, 8:1.4, 16:0.7, 9:2.7, 17:1.6, 35:1.3, 53:1.1,
           14:-0.3, 15:0.4, 5:-0.45, 34:0.6, 32:-0.2, 50:-0.2, 26:-0.1}
K_ELEM = {6:1.0, 7:0.9, 8:0.75, 16:0.65, 9:0.6, 17:0.4, 35:0.3, 53:0.25,
          14:0.5, 15:0.6, 5:0.7, 34:0.55, 32:0.4, 50:0.3, 26:0.3}
VDW_R = {1:1.20, 5:1.92, 6:1.70, 7:1.55, 8:1.52, 9:1.47, 14:2.10, 15:1.80,
         16:1.80, 17:1.75, 32:2.11, 34:1.90, 35:1.85, 50:2.17, 53:1.98, 26:2.05}

def bond_beta(b, m):
    a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
    k = np.sqrt(K_ELEM.get(a1.GetAtomicNum(), 0.4) * K_ELEM.get(a2.GetAtomicNum(), 0.4))
    bt = b.GetBondType()
    if b.GetIsAromatic():      f = 1.00
    elif bt == Chem.BondType.DOUBLE:  f = 1.20
    elif bt == Chem.BondType.TRIPLE:  f = 1.40
    elif b.GetIsConjugated():  f = 0.90
    else:
        sp3 = sum(1 for a in (a1, a2) if a.GetHybridization() == Chem.HybridizationType.SP3)
        f = 0.30 if sp3 else 0.55
    return k * f

def band_features(m):
    """Periodic tight-binding band structure of the 1-D chain."""
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() != 0]
    if len(stars) != 2 or len(heavy) < 2:
        return None
    pos = {ai: i for i, ai in enumerate(heavy)}
    n = len(heavy)
    # link atoms: the heavy neighbours of the two stars
    link = []
    for s in stars:
        nb = [x.GetIdx() for x in m.GetAtomWithIdx(s).GetNeighbors() if x.GetAtomicNum() != 0]
        if not nb: return None
        link.append(nb[0])
    # inter-cell beta: use the mean of the two star-bond betas
    betas_star = []
    for s in stars:
        b = m.GetBondBetweenAtoms(s, [x.GetIdx() for x in m.GetAtomWithIdx(s).GetNeighbors()][0])
        a = m.GetAtomWithIdx([x.GetIdx() for x in m.GetAtomWithIdx(s).GetNeighbors()][0])
        k = K_ELEM.get(a.GetAtomicNum(), 0.4)
        bt = b.GetBondType()
        if b.GetIsAromatic(): f = 1.0
        elif bt == Chem.BondType.DOUBLE: f = 1.2
        elif bt == Chem.BondType.TRIPLE: f = 1.4
        elif b.GetIsConjugated(): f = 0.9
        else: f = 0.30 if a.GetHybridization() == Chem.HybridizationType.SP3 else 0.55
        betas_star.append(k * f)
    b_inter = float(np.sqrt(betas_star[0] * betas_star[1]))

    H0 = np.zeros((n, n)); H1 = np.zeros((n, n))
    for ai in heavy:
        a = m.GetAtomWithIdx(ai)
        H0[pos[ai], pos[ai]] = -H_PARAM.get(a.GetAtomicNum(), 0.0)
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            v = -bond_beta(b, m)
            H0[pos[i], pos[j]] = v; H0[pos[j], pos[i]] = v
    # inter-cell hop: link[1] of cell 0 -> link[0] of cell 1
    H1[pos[link[1]], pos[link[0]]] = -b_inter

    nk = 17
    ks = np.linspace(0, np.pi, nk)
    bands = np.empty((nk, n))
    for t_, k in enumerate(ks):
        Hk = H0 + H1 * np.exp(1j * k) + H1.conj().T * np.exp(-1j * k)
        bands[t_] = np.linalg.eigvalsh(Hk)
    nocc = max(1, n // 2)
    vb = bands[:, nocc - 1]; cb = bands[:, min(n - 1, nocc)]
    vbm = vb.max(); cbm = cb.min()
    gap = cbm - vbm
    out = [gap, vbm, cbm, vb.max() - vb.min(), cb.max() - cb.min(),
           bands[:, 0].min(), bands[:, -1].max(),
           bands[0, nocc - 1] - bands[-1, nocc - 1],
           float(np.argmax(vb)) / (nk - 1), float(np.argmin(cb)) / (nk - 1),
           n, nocc]
    # gap at gamma and at zone boundary
    out += [bands[0, min(n-1,nocc)] - bands[0, nocc-1], bands[-1, min(n-1,nocc)] - bands[-1, nocc-1]]
    # effective masses ~ curvature at band edge
    def curv(band, i):
        i = int(np.clip(i, 1, nk - 2))
        return band[i-1] - 2*band[i] + band[i+1]
    out += [curv(vb, np.argmax(vb)), curv(cb, np.argmin(cb))]
    # density of states near edges
    out += [float((np.abs(bands - vbm) < 0.15).mean()), float((np.abs(bands - cbm) < 0.15).mean())]
    # molecular (non-periodic) Huckel of the capped unit for contrast
    ev = np.linalg.eigvalsh(H0)
    out += [ev[nocc] - ev[nocc-1] if n > nocc else np.nan, ev[nocc-1], ev[min(n-1,nocc)],
            ev.min(), ev.max(), b_inter]
    return out

BAND_N = 24

# ---- pi-only periodic Huckel: the physically correct model for the DFT bandgaps ----
PI_H = {6:0.0, 7:0.5, 8:1.0, 16:0.7, 15:0.4, 34:0.6, 5:-0.45, 14:-0.3}
PI_NE = {6:1, 7:1, 8:1, 16:1, 15:1, 34:1, 5:0, 14:1}   # pi electrons contributed

def pi_band_features(m):
    """Periodic Huckel restricted to the conjugated pi sub-system with correct
    pi-electron counting.  Saturated repeat units have an empty pi system, which
    is itself the signal that the gap is large."""
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return [np.nan] * PIB_N
    pi_atoms = []
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            continue
        if a.GetIsAromatic() or a.GetHybridization() in (Chem.HybridizationType.SP2,
                                                          Chem.HybridizationType.SP):
            pi_atoms.append(a.GetIdx())
        elif a.GetAtomicNum() in (7, 8, 16) and any(b.GetIsConjugated() for b in a.GetBonds()):
            pi_atoms.append(a.GetIdx())
    n = len(pi_atoms)
    if n < 2:
        return [0.0, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, float(n)]
    pos = {ai: i for i, ai in enumerate(pi_atoms)}
    H0 = np.zeros((n, n)); H1 = np.zeros((n, n))
    nelec = 0
    for ai in pi_atoms:
        a = m.GetAtomWithIdx(ai)
        z = a.GetAtomicNum()
        h = PI_H.get(z, 0.5)
        ne = PI_NE.get(z, 1)
        # lone-pair donors (pyrrole-type N, ether O, thiophene S) contribute 2
        if z in (7, 8, 16) and not any(b.GetBondType() in (Chem.BondType.DOUBLE,
                                                            Chem.BondType.TRIPLE)
                                       for b in a.GetBonds()):
            ne = 2; h = h + 1.0
        nelec += ne
        H0[pos[ai], pos[ai]] = -h
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            k = np.sqrt(K_ELEM.get(m.GetAtomWithIdx(i).GetAtomicNum(), 0.5) *
                        K_ELEM.get(m.GetAtomWithIdx(j).GetAtomicNum(), 0.5))
            f = 1.0 if (b.GetIsAromatic() or b.GetIsConjugated()) else 0.6
            H0[pos[i], pos[j]] = H0[pos[j], pos[i]] = -k * f
    # inter-cell hop only if both star-neighbours are in the pi system
    link = []
    for s in stars:
        nb = [x.GetIdx() for x in m.GetAtomWithIdx(s).GetNeighbors() if x.GetAtomicNum() != 0]
        link.append(nb[0] if nb else None)
    conj_link = all(l is not None and l in pos for l in link)
    if conj_link:
        H1[pos[link[1]], pos[link[0]]] = -0.9
    nocc = max(1, min(n - 1, int(round(nelec / 2))))
    nk = 13
    bands = np.empty((nk, n))
    for t_, k in enumerate(np.linspace(0, np.pi, nk)):
        Hk = H0 + H1 * np.exp(1j * k) + H1.conj().T * np.exp(-1j * k)
        bands[t_] = np.linalg.eigvalsh(Hk)
    vb = bands[:, nocc - 1]; cb = bands[:, nocc]
    gap = cb.min() - vb.max()
    return [gap, vb.max(), cb.min(), vb.max() - vb.min(), cb.max() - cb.min(),
            float(conj_link), float(nelec), float(nelec) / max(1, m.GetNumHeavyAtoms()),
            float(n)]

PIB_N = 9


def pi_features(m):
    """Conjugation topology."""
    n = m.GetNumAtoms()
    conj_adj = {i: [] for i in range(n)}
    for b in m.GetBonds():
        if b.GetIsConjugated() or b.GetIsAromatic():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            conj_adj[i].append(j); conj_adj[j].append(i)
    seen = set(); comps = []
    for s in range(n):
        if s in seen or not conj_adj[s]: continue
        stack = [s]; comp = set()
        while stack:
            u = stack.pop()
            if u in comp: continue
            comp.add(u); seen.add(u)
            stack.extend(conj_adj[u])
        comps.append(comp)
    sizes = sorted((len(c) for c in comps), reverse=True) or [0]
    # longest conjugated path (BFS diameter within largest component)
    diam = 0
    if comps:
        big = max(comps, key=len)
        for s in list(big)[:40]:
            dist = {s: 0}; q = [s]
            while q:
                u = q.pop(0)
                for v in conj_adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1; q.append(v)
            diam = max(diam, max(dist.values()))
    return [len(comps), sizes[0], sum(sizes), diam,
            sizes[0] / max(1, n), sum(sizes) / max(1, n)]

PI_N = 6

def polar_features(m):
    """Refraction / polarisability / dipole block for nc and eps."""
    try:
        mr = Crippen.MolMR(m)
    except Exception:
        mr = np.nan
    mw = Descriptors.MolWt(m)
    # van der Waals volume from atomic radii (Bondi), with H's
    mh = Chem.AddHs(m)
    vol = 0.0
    for a in mh.GetAtoms():
        r = VDW_R.get(a.GetAtomicNum(), 1.8)
        vol += 4.0 / 3.0 * np.pi * r ** 3
    # crude overlap correction: subtract per-bond spherical cap volume
    vol *= 0.62
    ll = mr / max(1e-6, vol)          # Lorentz-Lorenz ratio  (n^2-1)/(n^2+2)
    ll = float(np.clip(ll, 0.0, 0.95))
    n_est = np.sqrt((1 + 2 * ll) / max(1e-6, 1 - ll))
    try:
        AllChem.ComputeGasteigerCharges(m)
        q = np.array([float(a.GetProp('_GasteigerCharge')) for a in m.GetAtoms()])
        q[~np.isfinite(q)] = 0.0
    except Exception:
        q = np.zeros(m.GetNumAtoms())
    try:
        d = Chem.GetDistanceMatrix(m)
        dip_topo = float(np.abs(np.outer(q, q) * d).sum() ** 0.5)
    except Exception:
        dip_topo = np.nan
    tpsa = rdMolDescriptors.CalcTPSA(m)
    smarts = {
        'carbonyl': '[CX3]=[OX1]', 'ester': '[CX3](=O)[OX2H0]', 'amide': '[CX3](=O)[NX3]',
        'nitrile': '[NX1]#[CX2]', 'sulfone': '[$([SX4](=O)(=O))]', 'ether': '[OD2]([#6])[#6]',
        'hydroxyl': '[OX2H]', 'amine': '[NX3;H2,H1;!$(NC=O)]', 'nitro': '[N+](=O)[O-]',
        'halogen': '[F,Cl,Br,I]', 'fluorine': '[F]', 'aromatic_n': '[n]', 'thio': '[#16]',
        'siloxane': '[Si]-[O]', 'imide': 'O=C[NX3]C=O', 'urethane': '[NX3]C(=O)[OX2]',
        'sulfide': '[#16X2]([#6])[#6]', 'phosphate': '[PX4](=O)',
    }
    counts = []
    for pat in smarts.values():
        try:
            p = Chem.MolFromSmarts(pat)
            counts.append(len(m.GetSubstructMatches(p)) if p is not None else 0)
        except Exception:
            counts.append(0)
    nheavy = max(1, m.GetNumHeavyAtoms())
    out = [mr, mr / max(1e-6, mw), mr / nheavy, vol, vol / max(1e-6, mw), ll, n_est,
           n_est ** 2, mw / max(1e-6, vol), float(np.abs(q).sum()), float((q ** 2).sum()),
           float(np.abs(q).max()) if len(q) else 0.0, dip_topo, dip_topo / nheavy,
           tpsa, tpsa / max(1e-6, vol), tpsa / nheavy]
    out += counts
    out += [c / nheavy for c in counts]
    return out

POLAR_N = 17 + 18 * 2

def backbone_features(m):
    """Backbone stiffness / side-chain block for Tg."""
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    n = m.GetNumAtoms()
    if len(stars) != 2:
        return [np.nan] * BB_N
    # shortest path between the two attachment atoms = backbone of the repeat unit
    try:
        path = Chem.GetShortestPath(m, stars[0], stars[1])
    except Exception:
        return [np.nan] * BB_N
    if not path:
        return [np.nan] * BB_N
    bb = set(path) - set(stars)
    side = set(range(n)) - set(path)
    rot = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    rot_bonds = m.GetSubstructMatches(rot) if rot is not None else ()
    bb_rot = sum(1 for a, b in rot_bonds if a in bb and b in bb)
    bb_ring = sum(1 for i in bb if m.GetAtomWithIdx(i).IsInRing())
    bb_arom = sum(1 for i in bb if m.GetAtomWithIdx(i).GetIsAromatic())
    bb_sp3 = sum(1 for i in bb if m.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP3)
    bb_het = sum(1 for i in bb if m.GetAtomWithIdx(i).GetAtomicNum() not in (6, 0))
    L = max(1, len(bb))
    # side chains hanging off the backbone
    side_sizes = []
    for i in bb:
        for nb in m.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() in side:
                stack = [nb.GetIdx()]; comp = set()
                while stack:
                    u = stack.pop()
                    if u in comp or u in bb: continue
                    comp.add(u)
                    stack.extend(x.GetIdx() for x in m.GetAtomWithIdx(u).GetNeighbors())
                side_sizes.append(len(comp))
    ss = side_sizes or [0]
    # rotational-barrier proxy: rotatable bonds per backbone atom
    mw = Descriptors.MolWt(m)
    hbd = rdMolDescriptors.CalcNumHBD(m); hba = rdMolDescriptors.CalcNumHBA(m)
    return [L, bb_rot, bb_rot / L, bb_ring, bb_ring / L, bb_arom, bb_arom / L,
            bb_sp3 / L, bb_het, bb_het / L, len(side), len(side) / max(1, n),
            len(side_sizes), float(np.mean(ss)), float(np.max(ss)), float(np.sum(ss)),
            mw / L, hbd / L, hba / L, (hbd + hba) / max(1, n),
            float(np.sum(ss)) / L, len([s for s in ss if s >= 4])]

BB_N = 22

def featurize(c):
    m = Chem.MolFromSmiles(c)
    if m is None:
        return [np.nan] * (BAND_N + PIB_N + PI_N + POLAR_N + BB_N)
    try:
        bf = band_features(m)
    except Exception:
        bf = None
    if bf is None or len(bf) != BAND_N:
        bf = [np.nan] * BAND_N
    try: pbf = pi_band_features(m)
    except Exception: pbf = [np.nan] * PIB_N
    if len(pbf) != PIB_N: pbf = [np.nan] * PIB_N
    bf = list(bf) + list(pbf)
    try: pf = pi_features(m)
    except Exception: pf = [np.nan] * PI_N
    try: qf = polar_features(m)
    except Exception: qf = [np.nan] * POLAR_N
    try: bb = backbone_features(m)
    except Exception: bb = [np.nan] * BB_N
    if len(qf) != POLAR_N: qf = [np.nan] * POLAR_N
    if len(bb) != BB_N: bb = [np.nan] * BB_N
    return list(bf) + list(pf) + list(qf) + list(bb)


if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    F = pickle.load(open(f"{SCR}/features.pkl", "rb"))
    canon_list = F['canon_list']
    print("structures:", len(canon_list))
    with Pool(20) as p:
        res = p.map(featurize, canon_list, chunksize=32)
    M = np.array(res, dtype=np.float64)
    print("physics block:", M.shape, "nan frac:", np.isnan(M).mean().round(4),
          "elapsed", round(time.time() - t0, 1))
    names = ([f'band_{i}' for i in range(BAND_N)] + [f'piband_{i}' for i in range(PIB_N)]
             + [f'pi_{i}' for i in range(PI_N)]
             + [f'pol_{i}' for i in range(POLAR_N)] + [f'bb_{i}' for i in range(BB_N)])
    with open(f"{SCR}/physics.pkl", "wb") as f:
        pickle.dump(dict(M=M.astype(np.float32), names=names), f, protocol=4)
    # sanity: correlation of the Huckel gap with egc
    import pandas as pd
    BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
    tr = pd.read_csv(f"{BASE}/train.csv")
    tr['canon'] = tr['smiles'].map(F['canon_map']); tr['fi'] = tr['canon'].map(F['idx'])
    probe = [(0, 'tb_gap'), (1, 'tb_VBM'), (2, 'tb_CBM'),
             (BAND_N + 0, 'pi_gap'), (BAND_N + 1, 'pi_VBM'), (BAND_N + 2, 'pi_CBM'),
             (BAND_N + 6, 'n_pi_elec'),
             (BAND_N + PIB_N + 3, 'conj_diameter'),
             (BAND_N + PIB_N + PI_N + 6, 'LL_n_est'),
             (BAND_N + PIB_N + PI_N + 5, 'LL_ratio')]
    for t in ['egc', 'egb', 'ei', 'eea', 'nc', 'eps', 'tg']:
        r = tr[tr.target_type == t]; y = r['target'].values
        out = []
        for j, nm in probe:
            v = M[r['fi'].values, j]
            ok = np.isfinite(v) & (np.nanstd(v) > 0)
            if ok.sum() > 20 and np.std(v[ok]) > 0:
                out.append(f"{nm}={np.corrcoef(v[ok], y[ok])[0,1]:+.3f}")
        print(f"  {t:4s}: " + "  ".join(out))
    print("saved.")
