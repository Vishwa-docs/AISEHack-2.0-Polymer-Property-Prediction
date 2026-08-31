"""Is the cross-property stack circular for rows whose partner label is missing?

Suspicion: for a row of target t whose partner p is NOT observed, the stack uses
the partner's PREDICTION cur[s,p].  But the partner's model was allowed to use
t's observed label for s as an input feature.  So cur[s,p] encodes t's own external_label
and feeding it back is circular.  At test time this cannot happen, because for a
test row of target t the label t is unknown to everything.

Test: split each weak target's pool by whether its physics partner is observed and
score the two subsets separately.  If the partner-MISSING subset scores as well as
or better than the partner-OBSERVED subset, the missing-partner path is leaking,
because it has strictly less genuine information.
"""
import pickle, numpy as np, pandas as pd

SCR = "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
BASE = "/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/ppp-round-2"
T = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
ti = {t: j for j, t in enumerate(T)}
PARTNER = {'egb': 'egc', 'ei': 'eea', 'eea': 'ei', 'nc': 'eps', 'eps': 'nc'}

F = pickle.load(open(f"{SCR}/features.pkl", "rb"))
idx, cm = F['idx'], F['canon_map']; NS = len(F['canon_list'])
tr = pd.read_csv(f"{BASE}/train.csv"); ar = pd.read_csv(f"{BASE}/archive/train.csv")
def piv(d):
    p = d.pivot_table(index='canon', columns='target_type', values='target', aggfunc='mean')
    for c in T:
        if c not in p.columns: p[c] = np.nan
    return p[T]
for d in (tr, ar): d['canon'] = d['smiles'].map(cm)
L = np.full((NS, len(T)), np.nan)
for j, t in enumerate(T):
    for src in (piv(ar), piv(tr)):
        v = src[t].dropna(); L[[idx[c] for c in v.index], j] = v.values
LOBS = np.isfinite(L)
pool = {t: np.where(LOBS[:, j])[0] for j, t in enumerate(T)}

def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

for name in ['P1', 'PFINAL', 'PFINAL_routed']:
    try:
        M = np.load(f"{SCR}/out_weak/{name}.npy")
    except FileNotFoundError:
        print(f"\n### {name}: missing"); continue
    print(f"\n### {name}")
    print(f"{'target':7s} {'partner':8s} {'ALL':>9s} {'obs':>9s} {'n_obs':>6s} "
          f"{'MISSING':>9s} {'n_miss':>6s}")
    for t, p in PARTNER.items():
        j = ti[t]; rows = pool[t]; y = L[rows, j]; pr = M[rows, j]
        obs = LOBS[rows, ti[p]]
        so = r2(y[obs], pr[obs]) if obs.sum() > 5 else float('nan')
        sm = r2(y[~obs], pr[~obs]) if (~obs).sum() > 5 else float('nan')
        flag = "  <-- missing >= observed: circular" if sm >= so - 0.01 else ""
        print(f"{t:7s} {p:8s} {r2(y,pr):9.4f} {so:9.4f} {obs.sum():6d} "
              f"{sm:9.4f} {(~obs).sum():6d}{flag}")
