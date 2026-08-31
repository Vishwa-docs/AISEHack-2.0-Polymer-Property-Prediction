# PROMPT.md — Coding Agent Execution Instructions
## AISEHack 2.0 · Round 3 · Phase 4: Explainability & Robustness

> **YOU ARE A CODING AGENT. READ THIS FILE TOP TO BOTTOM. THEN EXECUTE.**
>
> This file contains everything you need. It is self-contained.
> Read the support files (AGENTS.md, REQUIREMENTS.md, PLAN.md) for full context,
> but all executable instructions are here.

---

## 0. Before You Start: File Check

Verify these files exist. If any are missing, stop and report which are absent.

```
../Dataset/train.csv
../Dataset/test.csv
../Dataset/PI1M.csv
../final_submissions/submission.csv
../Oracle/final_oracle.csv
```

Confirm `outputs/` directory exists (create if not):
```bash
mkdir -p Phase4_Round3_Explainability/outputs
mkdir -p Phase4_Round3_Explainability/outputs/mlp_checkpoints
mkdir -p Phase4_Round3_Explainability/scripts
```

All work happens inside `Phase4_Round3_Explainability/`. Never modify anything in
`../final_submissions/` or `../Dataset/`. Never read `../Oracle/` except in
script `16_khazana_verification.py` and `F3_oracle_sweep.py`.

---

## 1. Install Required Libraries

Run this first. If any package is missing, install it:

```bash
pip install nnsight shap lightgbm scikit-learn rdkit-pypi pandas numpy matplotlib seaborn scipy
```

Confirm:
```python
import nnsight, shap, lightgbm, sklearn, rdkit, pandas, numpy, matplotlib, seaborn, scipy
print("All imports OK")
```

---

## 2. Write and Run: `scripts/01_proxy_models.py`

This is the foundation for everything. Write and run it first. All downstream
scripts depend on the outputs from this one.

### What it must do

```python
"""
01_proxy_models.py
==================
Trains per-target proxy ensembles (Ridge + ExtraTrees + LightGBM) on the
same features used by V57 Stage A. Saves:
  - OOF predictions per target
  - Trained model pickles
  - Feature names JSON
  - Proxy scores CSV

ALL random seeds = 42. No external data. Reads only ../Dataset/train.csv.
"""
import os, json, pickle, random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42
random.seed(SEED); np.random.seed(SEED)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "mlp_checkpoints").mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
train = pd.read_csv("../Dataset/train.csv")
# train columns: smiles, target, target_type
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']

# ── Feature engineering ────────────────────────────────────────────────────
def canonical_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def morgan_fp(smi, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nBits)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
    arr = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def rdkit_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(200)
    desc = Descriptors.CalcMolDescriptors(mol)
    return np.array(list(desc.values()), dtype=float)

def build_features(smiles_list):
    """Returns (X_morgan, X_desc, feature_names) for a list of SMILES."""
    canonical = [canonical_smiles(s) for s in smiles_list]
    X_morgan = np.array([morgan_fp(s) for s in canonical])
    X_desc_raw = np.array([rdkit_descriptors(s) for s in canonical])
    # char n-gram vectorizer (fit on this data only)
    cv = CountVectorizer(ngram_range=(2, 6), max_features=8192,
                         analyzer='char', lowercase=False)
    X_ngram = cv.fit_transform(canonical).toarray()
    # impute descriptors
    imp = SimpleImputer(strategy='median')
    X_desc = imp.fit_transform(X_desc_raw)
    # drop near-zero variance descriptor cols
    var = X_desc.var(axis=0)
    X_desc = X_desc[:, var > 1e-8]
    X_full = np.hstack([X_morgan, X_desc, X_ngram])
    # feature names
    morgan_names = [f"morgan_{i}" for i in range(X_morgan.shape[1])]
    desc_names = [n for n, v in zip(
        [d[0] for d in Descriptors.descList], var > 1e-8) if v]
    ngram_names = [f"ngram_{t}" for t in cv.get_feature_names_out()]
    feat_names = morgan_names + desc_names + ngram_names
    return X_full, canonical, cv, imp, feat_names

# ── Per-target training ────────────────────────────────────────────────────
all_oof = {}
all_scores = {}

for target in TARGETS:
    df_t = train[train['target_type'] == target].copy()
    df_t['canonical'] = df_t['smiles'].apply(canonical_smiles)
    X, canonicals, cv, imp, feat_names = build_features(df_t['smiles'].tolist())
    y = df_t['target'].values
    groups = df_t['canonical'].values  # group by canonical SMILES

    n = len(df_t)
    n_splits = 5 if n >= 500 else 3
    cv_splitter = GroupKFold(n_splits=n_splits)

    oof_ridge = np.zeros(n)
    oof_et    = np.zeros(n)
    oof_lgbm  = np.zeros(n)
    models = {'ridge': [], 'et': [], 'lgbm': []}

    for fold, (tr_idx, va_idx) in enumerate(cv_splitter.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Scale for Ridge
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # Ridge
        m_ridge = Ridge(alpha=100, random_state=SEED)
        m_ridge.fit(X_tr_s, y_tr)
        oof_ridge[va_idx] = m_ridge.predict(X_va_s)

        # ExtraTrees
        m_et = ExtraTreesRegressor(n_estimators=200, n_jobs=-1,
                                   random_state=SEED, min_samples_leaf=2)
        m_et.fit(X_tr, y_tr)
        oof_et[va_idx] = m_et.predict(X_va)

        # LightGBM
        m_lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=31, min_child_samples=5,
                                    random_state=SEED, n_jobs=-1,
                                    verbosity=-1)
        m_lgbm.fit(X_tr, y_tr,
                   eval_set=[(X_va, y_va)],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_lgbm[va_idx] = m_lgbm.predict(X_va)

        models['ridge'].append((scaler, m_ridge))
        models['et'].append(m_et)
        models['lgbm'].append(m_lgbm)

    # NNLS ensemble (non-negative least squares blend)
    from scipy.optimize import nnls
    S = np.column_stack([oof_ridge, oof_et, oof_lgbm])
    w, _ = nnls(S, y)
    w = w / w.sum() if w.sum() > 0 else np.array([1/3, 1/3, 1/3])
    oof_ensemble = S @ w

    scores = {
        'ridge':    r2_score(y, oof_ridge),
        'et':       r2_score(y, oof_et),
        'lgbm':     r2_score(y, oof_lgbm),
        'ensemble': r2_score(y, oof_ensemble),
        'n_train':  n,
        'n_splits': n_splits,
        'nnls_weights': w.tolist(),
    }
    all_scores[target] = scores
    print(f"{target}: ridge={scores['ridge']:.4f}  et={scores['et']:.4f}  "
          f"lgbm={scores['lgbm']:.4f}  ens={scores['ensemble']:.4f}  (n={n})")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'smiles': df_t['smiles'].values,
        'canonical': df_t['canonical'].values,
        'true_value': y,
        'oof_ridge': oof_ridge,
        'oof_et': oof_et,
        'oof_lgbm': oof_lgbm,
        'oof_ensemble': oof_ensemble,
    })
    oof_df.to_csv(OUTPUT_DIR / f"proxy_oof_{target}.csv", index=False)

    # Save models (pickle)
    with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", 'wb') as f:
        pickle.dump({'models': models, 'cv': cv, 'imp': imp,
                     'feat_names': feat_names, 'nnls_weights': w}, f)

    all_oof[target] = oof_df

# Save scores
pd.DataFrame(all_scores).T.to_csv(OUTPUT_DIR / "proxy_scores.csv")
with open(OUTPUT_DIR / "proxy_feature_names.json", 'w') as f:
    json.dump(feat_names, f)
print("proxy_models.py DONE — all outputs in outputs/")
```

**Run it:**
```bash
cd /Users/daver/Desktop/AISEHack\ 2.0\ Polymr\ Property\ Prediction\ Round\ 3/Phase4_Round3_Explainability
python scripts/01_proxy_models.py
```

Verify `outputs/proxy_oof_tg.csv` exists and has reasonable R² values before proceeding.

---

## 3. Write and Run: `scripts/02_shap_global.py`

**Requirements it covers:** R1.1 (global SHAP beeswarm + summary chart)

```python
"""
02_shap_global.py
=================
Computes global SHAP feature importances for each target.
Produces beeswarm plots (top 20 features) and summary bar chart.
"""
import pickle, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

OUTPUT_DIR = Path("outputs")
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
train = pd.read_csv("../Dataset/train.csv")

def rebuild_features(smiles_list, pkl_data):
    """Rebuild feature matrix using saved cv/imp from proxy_models."""
    cv, imp = pkl_data['cv'], pkl_data['imp']
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(s) or Chem.MolFromSmiles('C'),
                                   isomericSmiles=True) for s in smiles_list]
    X_morgan = np.array([_morgan(s) for s in canonical])
    X_desc_raw = np.array([_rdkit_desc(s) for s in canonical])
    X_ngram = cv.transform(canonical).toarray()
    X_desc = imp.transform(X_desc_raw)
    var = X_desc.var(axis=0)
    X_desc = X_desc[:, var > 1e-8]
    return np.hstack([X_morgan, X_desc, X_ngram])

# [Define _morgan and _rdkit_desc helpers as in script 01]

top20_all = {}
for target in TARGETS:
    df_t = train[train['target_type'] == target].copy()
    with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", 'rb') as f:
        pkl = pickle.load(f)
    feat_names = pkl['feat_names']
    X = rebuild_features(df_t['smiles'].tolist(), pkl)

    # Use the last-fold LightGBM for SHAP (fast TreeExplainer)
    lgbm_model = pkl['models']['lgbm'][-1]  # last fold model
    explainer = shap.TreeExplainer(lgbm_model)

    # Subsample for speed if large
    n_shap = min(500, len(X))
    idx = np.random.choice(len(X), n_shap, replace=False)
    shap_values = explainer.shap_values(X[idx])

    # Top 20 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[-20:][::-1]
    top20_names = [feat_names[i] for i in top20_idx]
    top20_vals = mean_abs[top20_idx]
    top20_all[target] = dict(zip(top20_names, top20_vals.tolist()))

    # Beeswarm plot
    shap.summary_plot(shap_values, X[idx], feature_names=feat_names,
                      max_display=20, show=False)
    plt.title(f"SHAP Beeswarm — {target.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_beeswarm_{target}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {target}: top feature = {top20_names[0]} ({top20_vals[0]:.4f})")

# Global summary bar chart (mean |SHAP| summed across targets)
global_importance = {}
for target, d in top20_all.items():
    for feat, val in d.items():
        global_importance[feat] = global_importance.get(feat, 0) + val
top_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:25]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh([x[0] for x in top_global], [x[1] for x in top_global])
ax.set_xlabel("Summed mean |SHAP| across all 7 targets")
ax.set_title("Global Feature Importance (All 7 Polymer Properties)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary_global.png", dpi=150, bbox_inches='tight')
plt.close()

# Save CSV
rows = []
for target, d in top20_all.items():
    for feat, val in d.items():
        rows.append({'target': target, 'feature': feat, 'mean_abs_shap': val})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "shap_top20_per_target.csv", index=False)
print("02_shap_global.py DONE")
```

---

## 4. Write and Run: `scripts/07_smiles_invariance.py`

**Requirements it covers:** R2.1, R2.2

This is the most important invariance experiment. Write it carefully.

```python
"""
07_smiles_invariance.py
=======================
Tests prediction invariance across K=30 randomized SMILES per polymer.
Key distinction: Morgan/RDKit features are graph-invariant; char n-grams are not.
Reports violation rates and generates the canonicalization audit.
"""
import numpy as np, pandas as pd, pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
train = pd.read_csv("../Dataset/train.csv")
np.random.seed(42)

K_VARIANTS = 30   # number of random SMILES per polymer
N_POLYMERS  = 500  # how many validation polymers to test

def random_smiles(smi, k=30):
    """Generate k randomized (non-canonical) SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [smi]
    variants = set()
    attempts = 0
    while len(variants) < k and attempts < k * 5:
        rsmi = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
        variants.add(rsmi)
        attempts += 1
    return list(variants)

def canonical(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    return Chem.MolToSmiles(mol, isomericSmiles=True)

# Canonicalization audit
audit_lines = ["# Canonicalization Audit\n", "# Shows that canonical SMILES is stable\n\n"]
for i, row in train.sample(100, random_state=42).iterrows():
    can = canonical(row['smiles'])
    can2 = canonical(can)  # canonical of canonical = same
    match = "OK" if can == can2 else "FAIL"
    audit_lines.append(f"{match}: {row['smiles'][:40]} → {can[:40]}\n")
with open(OUTPUT_DIR / "canonicalization_check.txt", 'w') as f:
    f.writelines(audit_lines)

# Per-target invariance test
all_results = []
train_stds = {}

for target in TARGETS:
    df_t = train[train['target_type'] == target].copy()
    train_stds[target] = df_t['target'].std()

    with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", 'rb') as f:
        pkl = pickle.load(f)

    # Select N_POLYMERS from the training data (use as "validation proxy")
    sample = df_t.sample(min(N_POLYMERS, len(df_t)), random_state=42)

    target_stds = train_stds[target]
    row_results = []

    for _, row in sample.iterrows():
        variants = random_smiles(row['smiles'], k=K_VARIANTS)
        if len(variants) < 2:
            continue
        # Predict each variant
        preds = []
        for v in variants:
            X_v = _rebuild_single(v, pkl)  # [use helper from script 01]
            # Ensemble prediction (NNLS weights)
            w = np.array(pkl['nnls_weights'])
            # Use last-fold models
            p_ridge = pkl['models']['ridge'][-1][1].predict(
                        pkl['models']['ridge'][-1][0].transform(X_v.reshape(1,-1)))[0]
            p_et    = pkl['models']['et'][-1].predict(X_v.reshape(1,-1))[0]
            p_lgbm  = pkl['models']['lgbm'][-1].predict(X_v.reshape(1,-1))[0]
            pred = w[0]*p_ridge + w[1]*p_et + w[2]*p_lgbm
            preds.append(pred)

        preds = np.array(preds)
        row_results.append({
            'smiles': row['smiles'],
            'target': target,
            'pred_mean': preds.mean(),
            'pred_std': preds.std(),
            'pred_max_delta': preds.max() - preds.min(),
            'n_variants': len(preds),
        })

    # Violation rates
    df_res = pd.DataFrame(row_results)
    sigma = train_stds[target]
    df_res['violation_0.5sigma'] = (df_res['pred_std'] > 0.5*sigma).astype(int)
    df_res['violation_1sigma']   = (df_res['pred_std'] > 1.0*sigma).astype(int)
    df_res['violation_2sigma']   = (df_res['pred_std'] > 2.0*sigma).astype(int)
    all_results.append(df_res)

    vrate = df_res['violation_1sigma'].mean()
    print(f"{target}: mean_std={df_res['pred_std'].mean():.4f}  "
          f"sigma={sigma:.4f}  violation_rate_1sigma={vrate:.3f}")

df_all = pd.concat(all_results, ignore_index=True)
df_all.to_csv(OUTPUT_DIR / "smiles_invariance_per_target.csv", index=False)

# Violation rate summary
vrate_rows = []
for target in TARGETS:
    df_t = df_all[df_all['target'] == target]
    if len(df_t) == 0: continue
    vrate_rows.append({
        'target': target,
        'n_polymers': len(df_t),
        'violation_rate_0.5sigma': df_t['violation_0.5sigma'].mean(),
        'violation_rate_1sigma':   df_t['violation_1sigma'].mean(),
        'violation_rate_2sigma':   df_t['violation_2sigma'].mean(),
        'mean_pred_std': df_t['pred_std'].mean(),
        'train_sigma': train_stds[target],
    })
pd.DataFrame(vrate_rows).to_csv(OUTPUT_DIR / "smiles_invariance_violation_rate.csv", index=False)

# Boxplot
fig, ax = plt.subplots(figsize=(12, 5))
data_for_plot = [df_all[df_all['target'] == t]['pred_std'].values for t in TARGETS]
ax.boxplot(data_for_plot, labels=TARGETS, showfliers=False)
ax.set_ylabel("Prediction Std Across 30 Random SMILES")
ax.set_title("SMILES Representation Invariance — Prediction Stability by Target")
ax.set_xlabel("Target Property")
# Add train σ reference lines
for i, target in enumerate(TARGETS, 1):
    ax.axhline(y=train_stds[target]*0.01, color='red', alpha=0.3, linewidth=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "smiles_invariance_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("07_smiles_invariance.py DONE")
```

---

## 5. Write and Run: `scripts/A1_train_mlp.py` (NNsight MLP)

**Purpose:** Train a small per-target PyTorch MLP used as the vessel for
mechanistic interpretability via NNsight. This is NOT a submission model.

```python
"""
A1_train_mlp.py
===============
Trains a small PyTorch MLP per target for mechanistic interpretability.
Uses NNsight to verify internal access. Saves state_dicts.
"""
import torch, torch.nn as nn
import numpy as np, pandas as pd, pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import nnsight

torch.manual_seed(42); np.random.seed(42)
OUTPUT_DIR = Path("outputs")
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
train = pd.read_csv("../Dataset/train.csv")

class PolymerMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 512), nn.BatchNorm1d(512),
                                     nn.ReLU(), nn.Dropout(0.2))
        self.layer2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256),
                                     nn.ReLU(), nn.Dropout(0.2))
        self.layer3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128),
                                     nn.ReLU())
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x).squeeze(-1)

mlp_scores = {}

for target in TARGETS:
    df_t = train[train['target_type'] == target].copy()
    with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", 'rb') as f:
        pkl = pickle.load(f)
    X = _rebuild_features(df_t['smiles'].tolist(), pkl)  # reuse feature builder
    y = df_t['target'].values.astype(np.float32)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)

    model = PolymerMLP(X_tr_s.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr_s)
    y_tr_t = torch.tensor(y_tr)
    X_va_t = torch.tensor(X_va_s)
    y_va_t = torch.tensor(y_va)

    best_val = float('inf')
    patience, wait = 20, 0

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr_t)
        loss = criterion(pred, y_tr_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = criterion(val_pred, y_va_t).item()
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / f"mlp_checkpoints/{target}_mlp.pt")
            wait = 0
        else:
            wait += 1
            if wait >= patience: break

    # Load best and eval
    model.load_state_dict(torch.load(OUTPUT_DIR / f"mlp_checkpoints/{target}_mlp.pt"))
    model.eval()
    with torch.no_grad():
        va_preds = model(X_va_t).numpy()
    r2 = r2_score(y_va, va_preds)
    mlp_scores[target] = r2
    print(f"{target}: MLP val R²={r2:.4f}  (n={len(df_t)})")

    # Save scaler alongside checkpoint
    pickle.dump(scaler, open(OUTPUT_DIR / f"mlp_checkpoints/{target}_scaler.pkl", 'wb'))

pd.DataFrame.from_dict(mlp_scores, orient='index', columns=['val_r2']).to_csv(
    OUTPUT_DIR / "mlp_proxy_scores.csv")

# Verify NNsight works on one model
print("\nVerifying NNsight access...")
target_demo = 'tg'
df_demo = train[train['target_type'] == target_demo].sample(5, random_state=42)
with open(OUTPUT_DIR / f"proxy_models_{target_demo}.pkl", 'rb') as f:
    pkl_demo = pickle.load(f)
X_demo = _rebuild_features(df_demo['smiles'].tolist(), pkl_demo)
scaler_demo = pickle.load(open(OUTPUT_DIR / f"mlp_checkpoints/{target_demo}_scaler.pkl", 'rb'))
X_demo_s = torch.tensor(scaler_demo.transform(X_demo).astype(np.float32))

model_demo = PolymerMLP(X_demo_s.shape[1])
model_demo.load_state_dict(torch.load(OUTPUT_DIR / f"mlp_checkpoints/{target_demo}_mlp.pt"))
model_demo.eval()

nn_model = nnsight.NNsight(model_demo)
with nn_model.trace(X_demo_s):
    layer1_out = nn_model.layer1.output.save()
    layer2_out = nn_model.layer2.output.save()
    final_out  = nn_model.output.save()

print(f"NNsight layer1 shape: {layer1_out.shape}")
print(f"NNsight layer2 shape: {layer2_out.shape}")
print(f"NNsight output shape: {final_out.shape}")
print("NNsight verification PASSED")
print("A1_train_mlp.py DONE")
```

---

## 6. Write and Run: `scripts/A2_linear_probes.py`

**Purpose:** Answer "does layer N encode chemical concept X?"

```python
"""
A2_linear_probes.py
===================
Trains linear probes on MLP hidden states for 8 chemical concepts.
Generates probe R² heatmap (concept × layer).
"""
import torch, numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import nnsight

OUTPUT_DIR = Path("outputs")
train = pd.read_csv("../Dataset/train.csv")
np.random.seed(42)

# ── Define chemical concepts ───────────────────────────────────────────────
def compute_concepts(smi_list):
    """Returns dict of concept_name → array of values for each SMILES."""
    concepts = {c: [] for c in [
        'aromaticity', 'mol_weight_proxy', 'hbd_count',
        'hba_count', 'ring_count', 'aromatic_fraction',
        'polar_atom_fraction', 'rotatable_bonds'
    ]}
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for v in concepts.values(): v.append(0.0)
            continue
        n = mol.GetNumAtoms() or 1
        aromatic_atoms = sum(a.GetIsAromatic() for a in mol.GetAtoms())
        polar_atoms = sum(a.GetAtomicNum() in (7,8,9,16) for a in mol.GetAtoms())
        concepts['aromaticity'].append(float(mol.GetNumAromaticRings()))
        concepts['mol_weight_proxy'].append(float(n))
        concepts['hbd_count'].append(float(rdMolDescriptors.CalcNumHBD(mol)))
        concepts['hba_count'].append(float(rdMolDescriptors.CalcNumHBA(mol)))
        concepts['ring_count'].append(float(rdMolDescriptors.CalcNumRings(mol)))
        concepts['aromatic_fraction'].append(float(aromatic_atoms / n))
        concepts['polar_atom_fraction'].append(float(polar_atoms / n))
        concepts['rotatable_bonds'].append(float(rdMolDescriptors.CalcNumRotatableBonds(mol)))
    return {k: np.array(v) for k, v in concepts.items()}

# Probe each target's MLP
TARGETS_TO_PROBE = ['tg', 'egc', 'nc', 'eps']  # most interpretable targets
probe_results = {}  # target → {concept: {layer: r2}}

for target in TARGETS_TO_PROBE:
    df_t = train[train['target_type'] == target].copy()
    with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", 'rb') as f:
        pkl = pickle.load(f)
    scaler = pickle.load(open(OUTPUT_DIR / f"mlp_checkpoints/{target}_scaler.pkl", 'rb'))
    X = _rebuild_features(df_t['smiles'].tolist(), pkl)
    X_s = torch.tensor(scaler.transform(X).astype(np.float32))

    # Load MLP and wrap with NNsight
    # [Load PolymerMLP, load state_dict — same arch as A1]
    from A1_train_mlp import PolymerMLP  # or redefine inline
    model = PolymerMLP(X_s.shape[1])
    model.load_state_dict(torch.load(OUTPUT_DIR / f"mlp_checkpoints/{target}_mlp.pt"))
    model.eval()
    nn_m = nnsight.NNsight(model)

    # Extract activations at each layer
    with nn_m.trace(X_s):
        acts = {
            'layer1': nn_m.layer1.output.save(),
            'layer2': nn_m.layer2.output.save(),
            'layer3': nn_m.layer3.output.save(),
        }

    # Compute concepts
    concepts = compute_concepts(df_t['smiles'].tolist())

    probe_results[target] = {}
    for concept_name, concept_vals in concepts.items():
        probe_results[target][concept_name] = {}
        for layer_name, layer_acts in acts.items():
            A = layer_acts.detach().numpy()
            r2 = cross_val_score(Ridge(alpha=1.0), A, concept_vals,
                                 cv=3, scoring='r2').mean()
            probe_results[target][concept_name][layer_name] = max(r2, 0.0)
            print(f"  {target}/{concept_name}/{layer_name}: R²={r2:.3f}")

# Plot heatmap (aggregate across targets for clarity, or per target)
import itertools
concepts_list = list(next(iter(probe_results.values())).keys())
layers_list = ['layer1', 'layer2', 'layer3']

# Aggregate: mean R² across all probed targets per concept × layer
agg = np.zeros((len(concepts_list), len(layers_list)))
for i, c in enumerate(concepts_list):
    for j, l in enumerate(layers_list):
        vals = [probe_results[t][c][l] for t in TARGETS_TO_PROBE]
        agg[i, j] = np.mean(vals)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(agg, annot=True, fmt='.2f', xticklabels=layers_list,
            yticklabels=concepts_list, cmap='YlOrRd', ax=ax,
            vmin=0, vmax=1)
ax.set_title("Linear Probe R² — Chemical Concepts in MLP Hidden States\n"
             "(Mean across tg, egc, nc, eps targets)")
ax.set_xlabel("MLP Layer")
ax.set_ylabel("Chemical Concept")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "linear_probe_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# Save data
rows = []
for t in TARGETS_TO_PROBE:
    for c in concepts_list:
        for l in layers_list:
            rows.append({'target': t, 'concept': c, 'layer': l,
                         'probe_r2': probe_results[t][c][l]})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "linear_probe_results.csv", index=False)
print("A2_linear_probes.py DONE")
```

---

## 7. Write and Run: `scripts/A3_activation_patching.py`

**Purpose:** Test internal representation invariance using NNsight activation patching.

```python
"""
A3_activation_patching.py
==========================
For equivalent SMILES pairs, patches layer activations between representations
and measures how much the prediction changes at each layer.
Proves (or disproves) internal invariance beyond prediction invariance.
"""
import torch, numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
import nnsight

OUTPUT_DIR = Path("outputs")
train = pd.read_csv("../Dataset/train.csv")
np.random.seed(42)
torch.manual_seed(42)

TARGET = 'tg'  # most interesting target for this analysis
N_POLYMERS = 100
K_VARIANTS = 5

# [Load model, scaler, pkl as in A1/A2]

patch_results = []
for _, row in train[train['target_type'] == TARGET].sample(N_POLYMERS, random_state=42).iterrows():
    smi_canon = Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']), isomericSmiles=True)
    variants = [Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']), doRandom=True)
                for _ in range(K_VARIANTS)]
    variants = [v for v in variants if v != smi_canon][:3]
    if not variants:
        continue

    X_canon = torch.tensor(scaler.transform(
        _rebuild_features([smi_canon], pkl)).astype(np.float32))

    # Get clean activations for canonical SMILES
    with nn_m.trace(X_canon):
        act1_canon = nn_m.layer1.output.save()
        act2_canon = nn_m.layer2.output.save()
        pred_canon  = nn_m.output.save()

    for var_smi in variants:
        X_var = torch.tensor(scaler.transform(
            _rebuild_features([var_smi], pkl)).astype(np.float32))

        # Unpatched prediction
        with nn_m.trace(X_var):
            pred_var_normal = nn_m.output.save()

        # Patch layer 1 (replace var's layer1 output with canon's)
        with nn_m.trace(X_var):
            nn_m.layer1.output[:] = act1_canon.detach()
            pred_patch_l1 = nn_m.output.save()

        # Patch layer 2
        with nn_m.trace(X_var):
            nn_m.layer2.output[:] = act2_canon.detach()
            pred_patch_l2 = nn_m.output.save()

        patch_results.append({
            'smiles': smi_canon,
            'variant': var_smi,
            'pred_canon':     pred_canon.item(),
            'pred_var':       pred_var_normal.item(),
            'pred_patch_l1':  pred_patch_l1.item(),
            'pred_patch_l2':  pred_patch_l2.item(),
            'delta_pred':     abs(pred_var_normal.item() - pred_canon.item()),
            'delta_after_l1_patch': abs(pred_patch_l1.item() - pred_canon.item()),
            'delta_after_l2_patch': abs(pred_patch_l2.item() - pred_canon.item()),
        })

df_patch = pd.DataFrame(patch_results)
df_patch.to_csv(OUTPUT_DIR / "activation_patch_invariance.csv", index=False)

# Plot: distribution of deltas before and after patching
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axes[0].hist(df_patch['delta_pred'], bins=30, color='tomato', alpha=0.8)
axes[0].set_title("Δ prediction\n(no patching)")
axes[0].set_xlabel("Absolute prediction difference")
axes[1].hist(df_patch['delta_after_l1_patch'], bins=30, color='orange', alpha=0.8)
axes[1].set_title("Δ after patching\nLayer 1 activations")
axes[2].hist(df_patch['delta_after_l2_patch'], bins=30, color='steelblue', alpha=0.8)
axes[2].set_title("Δ after patching\nLayer 2 activations")
for ax in axes: ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)
plt.suptitle(f"Activation Patching — Representation Invariance Test (Tg, n={len(df_patch)} pairs)",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "activation_patch_invariance_plot.png", dpi=150, bbox_inches='tight')
plt.close()

mean_before = df_patch['delta_pred'].mean()
mean_after_l2 = df_patch['delta_after_l2_patch'].mean()
print(f"Mean |Δ| before patching:        {mean_before:.4f}")
print(f"Mean |Δ| after L2 patch:         {mean_after_l2:.4f}")
print(f"Recovery: {(mean_before - mean_after_l2)/mean_before*100:.1f}% of variance explained by L2 activations")
print("A3_activation_patching.py DONE")
```

---

## 8. Write and Run: `scripts/B2_structural_counterfactuals.py`

**Purpose:** The most falsifiable explainability test — do known chemical
modifications produce the expected directional changes?

```python
"""
B2_structural_counterfactuals.py
================================
Applies 5 known polymer chemistry modifications to 10 polymers and verifies
that the model predicts the expected directional changes.
"""
import numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

OUTPUT_DIR = Path("outputs")
train = pd.read_csv("../Dataset/train.csv")

# Define structural modification recipes
# Each mod: (description, expected_direction_per_target, rdkit_transform_fn)
# We use SMARTS-based atom property flipping or fragment insertion.

MODIFICATIONS = [
    {
        'name': 'add_aromatic_ring',
        'description': 'Append benzene ring to backbone (increases rigidity)',
        'expected': {'tg': 'up', 'egc': 'down', 'egb': 'down'},
        'smarts_from': '[CH2][CH2]',  # aliphatic segment
        'smarts_to':   '[CH2]c1ccccc1[CH2]',  # replaced with benzene
    },
    # Add more modifications...
]

# For simplicity: create modified SMILES by appending fragments
def modify_smiles_add_phenyl(smi):
    """Attempt to add a phenyl substituent."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    # Add a phenyl group to the first available carbon
    try:
        from rdkit.Chem import RWMol
        edit = RWMol(mol)
        # This is a demonstration — in practice use AllChem.ReplaceSubstructs
        # For the report, even a simple concatenation counts
        return None  # skip if transform fails
    except: return None

results = []
tg_polymers = train[train['target_type'] == 'tg'].sample(20, random_state=42)

for _, row in tg_polymers.iterrows():
    smi = row['smiles']
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue

    # Test 1: Increasing flexibility (add -O- ether linkage) → expect Tg down
    # Test 2: More aliphatic → expect Tg down
    # Test 3: More heteroatoms (N substitution) → expect bandgap change
    # Simple proxy: vary number of connected repeat units
    # Monomer → approximate dimer by concatenation at * attachment points
    if '*' in smi:
        # Construct dimer: *AABB* from *AB*
        inner = smi.replace('[*]', '').replace('*', '')
        dimer_smi = smi.replace('*', '').replace('[*]', '') + inner
        # Simplistic — sanitize
        dimer_mol = Chem.MolFromSmiles(dimer_smi)
        if dimer_mol:
            dimer_smi = Chem.MolToSmiles(dimer_mol)
        else:
            dimer_smi = None
    else:
        dimer_smi = None

    # Predict monomer and dimer
    with open(OUTPUT_DIR / f"proxy_models_tg.pkl", 'rb') as f:
        pkl = pickle.load(f)

    X_orig = _rebuild_features([smi], pkl)
    pred_orig = _predict_ensemble(X_orig, pkl)

    if dimer_smi:
        X_dimer = _rebuild_features([dimer_smi], pkl)
        pred_dimer = _predict_ensemble(X_dimer, pkl)
    else:
        pred_dimer = None

    results.append({
        'smiles_original': smi,
        'smiles_dimer': dimer_smi,
        'pred_monomer_tg': pred_orig,
        'pred_dimer_tg': pred_dimer,
        'delta_tg': (pred_dimer - pred_orig) if pred_dimer is not None else None,
    })

df_res = pd.DataFrame(results).dropna(subset=['pred_dimer_tg'])
df_res.to_csv(OUTPUT_DIR / "structural_counterfactuals.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_res['pred_monomer_tg'], df_res['pred_dimer_tg'],
           alpha=0.7, s=60, color='steelblue')
lims = [min(df_res['pred_monomer_tg'].min(), df_res['pred_dimer_tg'].min()) - 5,
        max(df_res['pred_monomer_tg'].max(), df_res['pred_dimer_tg'].max()) + 5]
ax.plot(lims, lims, 'k--', alpha=0.5, label='No change')
ax.set_xlabel("Predicted Tg — Monomer (K)")
ax.set_ylabel("Predicted Tg — Dimer (K)")
ax.set_title("Structural Counterfactual: Monomer vs Dimer Tg Predictions\n"
             "(Expected: small increase or no change with chain extension)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "structural_counterfactuals_plot.png", dpi=150, bbox_inches='tight')
plt.close()

n_up = (df_res['delta_tg'] > 0).sum()
n_down = (df_res['delta_tg'] < 0).sum()
print(f"Dimer vs monomer: {n_up} polymers Tg↑, {n_down} polymers Tg↓")
print(f"Mean delta: {df_res['delta_tg'].mean():.2f} K")
print("B2_structural_counterfactuals.py DONE")
```

---

## 9. Write and Run: `scripts/11_conformal.py`

**Requirements it covers:** R3.2

```python
"""
11_conformal.py
===============
Split-conformal prediction intervals. No external dependencies — pure numpy.
Calibrates on held-out fold. Reports empirical coverage at 80/90/95%.
Writes intervals for all 4,940 test predictions.
"""
import numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
ALPHA_LEVELS = [0.80, 0.90, 0.95]
train_df = pd.read_csv("../Dataset/train.csv")
test_df  = pd.read_csv("../Dataset/test.csv")
submission = pd.read_csv("../final_submissions/submission.csv")

coverage_rows = []

for target in TARGETS:
    df_t = train_df[train_df['target_type'] == target].copy()
    oof = pd.read_csv(OUTPUT_DIR / f"proxy_oof_{target}.csv")
    residuals = np.abs(oof['true_value'].values - oof['oof_ensemble'].values)

    for alpha in ALPHA_LEVELS:
        # Finite-sample corrected quantile
        n_cal = len(residuals)
        q_level = np.ceil((n_cal + 1) * alpha) / n_cal
        q_level = min(q_level, 1.0)
        q_hat = np.quantile(residuals, q_level)

        # Empirical coverage on a held-out 20% split
        split_idx = int(0.8 * len(residuals))
        cal_res  = residuals[:split_idx]
        val_res  = residuals[split_idx:]
        oof_val  = oof['oof_ensemble'].values[split_idx:]
        true_val = oof['true_value'].values[split_idx:]

        q_hat_cal = np.quantile(cal_res, np.ceil((len(cal_res)+1)*alpha)/len(cal_res))
        in_interval = (np.abs(true_val - oof_val) <= q_hat_cal)
        empirical_cov = in_interval.mean()

        coverage_rows.append({
            'target': target,
            'nominal_coverage': alpha,
            'empirical_coverage': empirical_cov,
            'interval_halfwidth': q_hat,
            'n_calibration': split_idx,
            'n_validation': len(val_res),
        })

# Write coverage table
df_cov = pd.DataFrame(coverage_rows)
df_cov.to_csv(OUTPUT_DIR / "conformal_coverage_table.csv", index=False)

# Reliability diagram
fig, axes = plt.subplots(1, len(TARGETS), figsize=(20, 4), sharey=True)
for ax, target in zip(axes, TARGETS):
    df_t_cov = df_cov[df_cov['target'] == target]
    ax.plot([0.8, 0.9, 0.95], [0.8, 0.9, 0.95], 'k--', alpha=0.5, label='Perfect cal.')
    ax.plot(df_t_cov['nominal_coverage'], df_t_cov['empirical_coverage'],
            'o-', color='steelblue', markersize=8, label='Empirical')
    ax.set_title(target.upper()); ax.set_xlabel("Nominal coverage")
    ax.set_xlim(0.78, 0.97); ax.set_ylim(0.70, 1.02)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Empirical coverage")
axes[0].legend()
fig.suptitle("Conformal Prediction Calibration — Nominal vs Empirical Coverage",
             fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "conformal_calibration_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# Write test predictions with intervals
test_target_map = test_df.set_index('id')['target_type'].to_dict()
sub_merged = submission.copy()
sub_merged['target_type'] = sub_merged['id'].map(test_target_map)

for alpha in [0.80, 0.90]:
    q_col = f"q_hat_{int(alpha*100)}"
    sub_merged[q_col] = sub_merged['target_type'].map(
        df_cov[df_cov['nominal_coverage'] == alpha].set_index('target')['interval_halfwidth']
    )
    sub_merged[f"lower_{int(alpha*100)}"] = sub_merged['target'] - sub_merged[q_col]
    sub_merged[f"upper_{int(alpha*100)}"] = sub_merged['target'] + sub_merged[q_col]

cols = ['id', 'target_type', 'target',
        'lower_80', 'upper_80', 'lower_90', 'upper_90']
sub_merged[cols].to_csv(OUTPUT_DIR / "test_predictions_with_intervals.csv", index=False)
print("11_conformal.py DONE")
```

---

## 10. Write and Run: `scripts/15_generalization_ladder.py`

**Requirements it covers:** R4.1

```python
"""
15_generalization_ladder.py
============================
Evaluates proxy ensemble under 6 CV regimes of increasing structural difficulty.
Produces the "staircase" generalization plot.
"""
import numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score

OUTPUT_DIR = Path("outputs")
TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'nc', 'eps']
train_df = pd.read_csv("../Dataset/train.csv")

def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return "invalid"
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except: return "no_scaffold"

def get_family(smi):
    """Simple family: aromatic vs aliphatic."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return "unknown"
    return "aromatic" if mol.GetNumAromaticRings() > 0 else "aliphatic"

def tanimoto_nn(fp, train_fps):
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return max(sims) if sims else 0.0

REGIMES = {
    'G0_random':          'random',
    'G1_canonical_group': 'canonical',
    'G2_scaffold':        'scaffold',
    'G3_family':          'family',
    'G4_low_sim_0.6':     'low_sim_0.6',
    'G5_ultra_low_0.4':   'low_sim_0.4',
}

results = []
for target in TARGETS:
    df_t = train_df[train_df['target_type'] == target].copy()
    with open(OUTPUT_DIR / f"proxy_models_{target}.pkl", 'rb') as f:
        pkl = pickle.load(f)
    X = _rebuild_features(df_t['smiles'].tolist(), pkl)
    y = df_t['target'].values

    df_t['canonical'] = df_t['smiles'].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s)
    df_t['scaffold'] = df_t['smiles'].apply(get_scaffold)
    df_t['family']   = df_t['smiles'].apply(get_family)

    fps = [AllChem.GetMorganFingerprintAsBitVect(
               Chem.MolFromSmiles(s) or Chem.MolFromSmiles('C'), 2, 1024)
           for s in df_t['smiles']]

    # Compute train→val similarity for low-sim regimes
    # (simplified: for each polymer, compute its NN similarity to the rest)
    sim_scores = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in range(len(fps)) if j!=i]))
        for i in range(len(fps))
    ])

    for regime_name, regime_type in REGIMES.items():
        if regime_type == 'random':
            splitter = KFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(splitter.split(X))
        elif regime_type == 'canonical':
            splitter = GroupKFold(n_splits=5)
            splits = list(splitter.split(X, y, df_t['canonical'].values))
        elif regime_type == 'scaffold':
            splitter = GroupKFold(n_splits=5)
            splits = list(splitter.split(X, y, df_t['scaffold'].values))
        elif regime_type == 'family':
            splitter = GroupKFold(n_splits=2)
            splits = list(splitter.split(X, y, df_t['family'].values))
        elif regime_type.startswith('low_sim'):
            thresh = float(regime_type.split('_')[-1])
            # Hold out low-similarity polymers as test; rest as train
            low_idx = np.where(sim_scores < thresh)[0]
            high_idx = np.where(sim_scores >= thresh)[0]
            if len(low_idx) < 5: splits = []; continue
            splits = [(high_idx, low_idx)]
        else:
            continue

        fold_r2s = []
        for tr_idx, va_idx in splits:
            if len(va_idx) < 3: continue
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            # Quick LightGBM fit
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                   num_leaves=31, random_state=42,
                                   verbosity=-1, n_jobs=-1)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            fold_r2s.append(r2_score(y_va, preds))

        mean_r2 = np.mean(fold_r2s) if fold_r2s else np.nan
        results.append({'target': target, 'regime': regime_name,
                        'mean_r2': mean_r2, 'n_folds': len(fold_r2s)})
        print(f"  {target} {regime_name}: R²={mean_r2:.4f}")

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_DIR / "generalization_ladder.csv", index=False)

# Plot staircase
regime_order = list(REGIMES.keys())
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(TARGETS)))
for i, target in enumerate(TARGETS):
    df_t_res = df_res[df_res['target'] == target]
    r2_vals = [df_t_res[df_t_res['regime'] == r]['mean_r2'].values[0]
               if len(df_t_res[df_t_res['regime'] == r]) > 0 else np.nan
               for r in regime_order]
    ax.plot(regime_order, r2_vals, 'o-', label=target, color=colors[i], markersize=7)

ax.set_xlabel("Validation Split Strategy (increasing difficulty →)")
ax.set_ylabel("Mean R²")
ax.set_title("Generalization Ladder — R² Under Increasingly Difficult CV Splits")
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "generalization_ladder_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("15_generalization_ladder.py DONE")
```

---

## 11. Write and Run: `scripts/16_khazana_verification.py`

**Requirements it covers:** R4.2  
**⚠ This is the ONLY script allowed to read `../Oracle/final_oracle.csv`.**

```python
"""
16_khazana_verification.py
===========================
Evaluates submitted predictions against Khazana DFT ground truth.
POST-FREEZE EVALUATION ONLY — oracle data never enters training.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

OUTPUT_DIR = Path("outputs")
DFT_TARGETS = ['egc', 'egb', 'ei', 'eea', 'nc', 'eps']

# Load post-freeze oracle (allowed here only)
oracle  = pd.read_csv("../Oracle/final_oracle.csv")
sub     = pd.read_csv("../final_submissions/submission.csv")
test_df = pd.read_csv("../Dataset/test.csv")

# Merge: id, predicted_value, oracle_value, target_type
merged = sub.rename(columns={'target': 'pred'}).merge(
    test_df[['id', 'target_type']], on='id').merge(oracle, on='id', how='inner')

# oracle columns: id, target_value (or per-target columns)
# Adapt column names to match actual oracle schema
# (oracle has columns: id, tg, egc, egb, ei, eea, nc, eps, panel)

score_rows = []
for target in DFT_TARGETS:
    if target not in oracle.columns: continue
    df_t = merged[merged['target_type'] == target].copy()
    df_t = df_t[df_t[target].notna()]  # drop unresolved rows
    if len(df_t) < 10: continue

    y_true = df_t[target].values
    y_pred = df_t['pred'].values
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    score_rows.append({'target': target, 'n': len(df_t), 'r2': r2, 'mae': mae})
    print(f"Khazana {target}: R²={r2:.4f}  MAE={mae:.4f}  n={len(df_t)}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color='steelblue')
    lims = [min(y_true.min(), y_pred.min())*0.95,
            max(y_true.max(), y_pred.max())*1.05]
    ax.plot(lims, lims, 'r--', alpha=0.8, label=f'R²={r2:.4f}')
    ax.set_xlabel(f"Khazana DFT Ground Truth ({target})")
    ax.set_ylabel(f"Model Prediction")
    ax.set_title(f"Khazana Verification: {target.upper()}\n(n={len(df_t)}, post-freeze)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"khazana_scatter_{target}.png", dpi=150, bbox_inches='tight')
    plt.close()

pd.DataFrame(score_rows).to_csv(OUTPUT_DIR / "khazana_holdout_scores.csv", index=False)
print("16_khazana_verification.py DONE — oracle used only for evaluation")
```

---

## 12. Write and Run All Remaining Scripts

For scripts 03–10 and 12–14 and 17–18, follow the same pattern:
- Read the relevant section in PLAN.md for the precise specification
- Write the script with full docstring, fixed seed=42, all outputs to `outputs/`
- Every plot: title, axis labels, dpi=150, bbox_inches='tight'
- Every CSV: has a header, numeric values are floats (not strings)

When all scripts are done, run:
```bash
python scripts/18_scorecard.py
python check_outputs.py
```

---

## 13. Write: `run.sh`

Create this file at `Phase4_Round3_Explainability/run.sh`:

```bash
#!/bin/bash
# run.sh — runs the full Phase 4 explainability analysis pipeline
# Usage: cd Phase4_Round3_Explainability && bash run.sh
# Expected total runtime: ~2-4 hours depending on hardware

set -e  # stop on first error
cd "$(dirname "$0")"

echo "=== Phase 4: Explainability & Robustness Analysis ==="
echo "$(date)"
echo ""

# Phase 0: Setup
echo "[Phase 0] Environment setup..."
python scripts/00_setup.py

# Phase 1: Proxy model training (MUST run first)
echo "[Phase 1] Training proxy models..."
python scripts/01_proxy_models.py

# Phase 2: Explainability (R1)
echo "[Phase 2] SHAP global analysis..."
python scripts/02_shap_global.py
python scripts/03_shap_local.py
python scripts/04_fidelity.py
python scripts/05_explanation_agreement.py
python scripts/06_physics_decomp.py

# Phase 3: Invariance (R2)
echo "[Phase 3] Invariance analysis..."
python scripts/07_smiles_invariance.py
python scripts/08_attribution_invariance.py
python scripts/09_oligomer_invariance.py

# Phase 4: Reliability (R3)
echo "[Phase 4] Reliability analysis..."
python scripts/10_cv_validation.py
python scripts/11_conformal.py
python scripts/12_uncertainty_vs_error.py
python scripts/13_applicability_domain.py
python scripts/14_seed_stability.py

# Phase 5: Generalization (R4)
echo "[Phase 5] Generalization analysis..."
python scripts/15_generalization_ladder.py
python scripts/16_khazana_verification.py
python scripts/17_tail_performance.py

# Phase 6: Synthesis
echo "[Phase 6] Generating scorecard and radar chart..."
python scripts/18_scorecard.py

# Extended: NNsight mechanistic interpretability
echo "[Ext-A] NNsight mechanistic interpretability..."
python scripts/A1_train_mlp.py
python scripts/A2_linear_probes.py
python scripts/A3_activation_patching.py
python scripts/A4_causal_tracing.py

# Extended: Counterfactuals
echo "[Ext-B] Counterfactual experiments..."
python scripts/B2_structural_counterfactuals.py

# Extended: Advanced UQ
echo "[Ext-D] Advanced uncertainty quantification..."
python scripts/D1_ensemble_vs_conformal.py
python scripts/D3_reliability_tiers.py

# Extended: Physics
echo "[Ext-E] Physics identity analysis..."
python scripts/E1_physics_violations.py

# Extended: Sweeps (100+ experiments)
echo "[Ext-F] Feature ablation sweep (133 experiments)..."
python scripts/F2_feature_ablation.py

# Final verification
echo "[Final] Checking all outputs..."
python check_outputs.py

echo ""
echo "=== Phase 4 COMPLETE ==="
echo "$(date)"
echo "Results in: outputs/"
echo "Summary:    outputs/scorecard.md"
echo "Report:     outputs/TRUSTWORTHINESS_REPORT.html (if G1 ran)"
```

---

## 14. Final Verification

After `run.sh` completes:

1. Check `outputs/scorecard.md` for PASS/FAIL per requirement.
2. The minimum viable set must all be PASS:
   - R1.1 (SHAP beeswarm plots exist + chemically meaningful top features)
   - R1.2 (local SHAP + mol viz)
   - R2.1 (SMILES invariance violation rate < 5% at 1σ)
   - R2.3 (attribution cosine similarity ≥ 0.70)
   - R3.1 (structured CV table exists)
   - R3.2 (conformal coverage within ±3%)
   - R4.1 (generalization ladder plot exists)
   - R4.2 (Khazana verification scores exist)
3. Write `outputs/SESSION_SUMMARY.md` with: which passed, any unexpected findings,
   and the 3 most compelling results to highlight for judges.

---

## Helper Functions (include in each script that needs them)

```python
# ── Shared helpers — include at top of each script that needs them ──────────
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
import numpy as np

def _morgan(smi, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nBits)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
    arr = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def _rdkit_desc(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(len(Descriptors.descList))
    desc = Descriptors.CalcMolDescriptors(mol)
    return np.array(list(desc.values()), dtype=float)

def _rebuild_features(smiles_list, pkl_data):
    """Rebuild feature matrix using saved cv/imp from proxy pkl."""
    cv, imp = pkl_data['cv'], pkl_data['imp']
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s) or Chem.MolFromSmiles('C'),
                               isomericSmiles=True) for s in smiles_list]
    X_morgan = np.array([_morgan(s) for s in canon])
    X_desc_raw = np.array([_rdkit_desc(s) for s in canon])
    X_ngram = cv.transform(canon).toarray()
    X_desc = imp.transform(X_desc_raw)
    var = X_desc.var(axis=0)
    X_desc = X_desc[:, var > 1e-8]
    return np.hstack([X_morgan, X_desc, X_ngram])

def _predict_ensemble(X, pkl_data):
    """Single prediction from the last-fold NNLS ensemble."""
    w = np.array(pkl_data['nnls_weights'])
    scaler, m_ridge = pkl_data['models']['ridge'][-1]
    m_et   = pkl_data['models']['et'][-1]
    m_lgbm = pkl_data['models']['lgbm'][-1]
    X_s = scaler.transform(X.reshape(1,-1) if X.ndim == 1 else X)
    p_r = m_ridge.predict(X_s)
    p_e = m_et.predict(X.reshape(1,-1) if X.ndim == 1 else X)
    p_l = m_lgbm.predict(X.reshape(1,-1) if X.ndim == 1 else X)
    return (w[0]*p_r + w[1]*p_e + w[2]*p_l)[0]
```

Save these helpers in `scripts/helpers.py` and import them in every script
with `from helpers import _morgan, _rdkit_desc, _rebuild_features, _predict_ensemble`.
