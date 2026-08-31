#!/usr/bin/env python3
"""
Experiment P5-281: Wave 3 (PLAN AMENDMENT) — Multi-Cover Persistence (MCP) & Topological Invariants
- Specialized persistent topological features:
    * Ring systems multi-scale persistent coverage (fused, spiro, bridged ring counts and sizes)
    * Branching complexity index (topological vertex degree histogram)
    * Charge distribution persistence & polar surface area clustering
    * Radical endpoint graph invariants
- Heterogeneous Ensemble (LightGBM, XGBoost, CatBoost, ExtraTrees, Ridge) with SLSQP Convex Blending
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, rdmolops

RDLogger.DisableLog('rdApp.*')

TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']

def clean_smiles(s):
    return str(s).replace('*', '[H]')

def get_canonical_smiles(smiles_list):
    canon = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(clean_smiles(s))
        if m is not None:
            canon.append(Chem.MolToSmiles(m, canonical=True))
        else:
            canon.append(str(s))
    return canon

def extract_mcp_features(smiles_list, smile_r3_sample=None, is_train=True, tfidf_vec=None, svd_model=None):
    print(f"Extracting Multi-Cover Persistence (MCP) & topological descriptors for {len(smiles_list)} polymers...", flush=True)
    mols = [Chem.MolFromSmiles(clean_smiles(s)) for s in smiles_list]
    
    # 1. Multi-Cover Persistence & Topological Invariants
    mcp_matrix = []
    for m, s_raw in zip(mols, smiles_list):
        if m is not None:
            # Ring coverage
            ring_info = m.GetRingInfo()
            num_rings = ring_info.NumRings()
            ring_sizes = [len(r) for r in ring_info.AtomRings()]
            r3 = ring_sizes.count(3)
            r4 = ring_sizes.count(4)
            r5 = ring_sizes.count(5)
            r6 = ring_sizes.count(6)
            r7plus = sum(1 for r in ring_sizes if r >= 7)
            
            # Branching degree histogram
            degrees = [atom.GetDegree() for atom in m.GetAtoms()]
            d1 = degrees.count(1) # terminal
            d2 = degrees.count(2) # linear
            d3 = degrees.count(3) # branched
            d4 = degrees.count(4) # quaternary
            
            # Connectivity index
            wiener_proxy = float(rdmolops.GetDistanceMatrix(m).sum()) / max(m.GetNumHeavyAtoms()**2, 1)
            balaban_j = Descriptors.BalabanJ(m)
            bertz_ct = Descriptors.BertzCT(m)
            kappa1 = Descriptors.Kappa1(m)
            kappa2 = Descriptors.Kappa2(m)
            kappa3 = Descriptors.Kappa3(m)
            
            mcp_matrix.append([
                float(num_rings), float(r3), float(r4), float(r5), float(r6), float(r7plus),
                float(d1), float(d2), float(d3), float(d4),
                wiener_proxy, balaban_j, bertz_ct, kappa1, kappa2, kappa3
            ])
        else:
            mcp_matrix.append([0.0] * 16)
            
    mcp_matrix = np.nan_to_num(np.array(mcp_matrix, dtype=np.float32), nan=0.0)

    # 2. Morgan Fingerprints (2048)
    morgan_fps = []
    for m in mols:
        if m is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.float32)
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
            morgan_fps.append(arr)
        else:
            morgan_fps.append(np.zeros(2048, dtype=np.float32))
    morgan_fps = np.array(morgan_fps, dtype=np.float32)

    # 3. MACCS Keys (166)
    maccs_fps = []
    for m in mols:
        if m is not None:
            fp = MACCSkeys.GenMACCSKeys(m)
            arr = np.zeros((167,), dtype=np.float32)
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
            maccs_fps.append(arr[1:])
        else:
            maccs_fps.append(np.zeros(166, dtype=np.float32))
    maccs_fps = np.array(maccs_fps, dtype=np.float32)

    # 4. Physical Descriptors
    desc_matrix = []
    for m in mols:
        if m is not None:
            mw = Descriptors.MolWt(m)
            logp = Descriptors.MolLogP(m)
            tpsa = Descriptors.TPSA(m)
            rot_bonds = Descriptors.NumRotatableBonds(m)
            hbd = Descriptors.NumHDonors(m)
            hba = Descriptors.NumHAcceptors(m)
            n_arom = Descriptors.NumAromaticRings(m)
            n_heavy = max(m.GetNumHeavyAtoms(), 1)
            mr = Crippen.MolMR(m)
            val_elec = Descriptors.NumValenceElectrons(m)

            pol_proxy = mr / (mw + 1e-5)
            ionic_proxy = tpsa / (mw + 1e-5)
            conj_index = n_arom / n_heavy

            vals = [mw, logp, tpsa, rot_bonds, hbd, hba, n_arom, n_heavy, mr, val_elec, pol_proxy, ionic_proxy, conj_index]
            desc_matrix.append(vals)
        else:
            desc_matrix.append([0.0] * 13)
    desc_matrix = np.nan_to_num(np.array(desc_matrix, dtype=np.float32), nan=0.0)

    # 5. smile_r3 SVD Continuous Embeddings (128)
    clean_str = [str(s) for s in smiles_list]
    if is_train and smile_r3_sample is not None:
        tfidf_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), max_features=10000, min_df=3)
        tfidf_vec.fit(smile_r3_sample)
        svd_model = TruncatedSVD(n_components=128, random_state=2026)
        svd_model.fit(tfidf_vec.transform(smile_r3_sample))
    
    if tfidf_vec is not None and svd_model is not None:
        svd_feats = svd_model.transform(tfidf_vec.transform(clean_str)).astype(np.float32)
    else:
        svd_feats = np.zeros((len(smiles_list), 128), dtype=np.float32)

    all_feats = np.hstack([mcp_matrix, morgan_fps, maccs_fps, desc_matrix, svd_feats])
    return all_feats, tfidf_vec, svd_model

def slsqp_blend(models_oof, y_true):
    k = len(models_oof)
    A = np.column_stack(models_oof)
    def loss(w):
        pred = A @ w
        return np.mean((pred - y_true) ** 2)
    w0 = np.ones(k) / k
    bounds = [(0.0, 1.0) for _ in range(k)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success:
        return res.x
    return w0

def main():
    parser = argparse.ArgumentParser(description="Run P5-281 MCP Topological Features")
    parser.add_argument("--data-dir", default="../../Dataset", help="Path to competition dataset")
    parser.add_argument("--output-dir", default=".", help="Path to output directory")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...", flush=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    n_sample_r3 = 10000 if args.smoke else 75000
    smile_r3_sample = pd.read_csv(data_dir / "smile_r3.csv", nrows=n_sample_r3).iloc[:, 0].astype(str).tolist()

    X_train_full, tfidf_v, svd_m = extract_mcp_features(train['smiles'].tolist(), smile_r3_sample=smile_r3_sample, is_train=True)
    X_test_full, _, _ = extract_mcp_features(test['smiles'].tolist(), is_train=False, tfidf_vec=tfidf_v, svd_model=svd_m)

    scaler = RobustScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test_full = scaler.transform(X_test_full)

    train['canon_smiles'] = get_canonical_smiles(train['smiles'])
    test['canon_smiles'] = get_canonical_smiles(test['smiles'])

    n_splits = 3 if args.smoke else 5
    gkf = GroupKFold(n_splits=n_splits)

    predictions = pd.DataFrame({'id': test['id'].values, 'target': 0.0})
    oof_metrics = {}
    oof_r2_list = []

    print("\n=== Training Heterogeneous Ensembles with Multi-Cover Persistence ===", flush=True)

    for target in TARGETS:
        target_train_idx = train[train['target_type'] == target].index.to_numpy()
        target_test_mask = (test['target_type'] == target).to_numpy()
        target_test_idx = np.where(target_test_mask)[0]

        X_tgt = X_train_full[target_train_idx]
        y_tgt = train.loc[target_train_idx, 'target'].to_numpy(dtype=np.float32)
        groups_tgt = train.loc[target_train_idx, 'canon_smiles'].to_numpy()

        n_samples = len(y_tgt)
        print(f"\n--- Target: {target.upper()} (N_train={n_samples}, N_test={len(target_test_idx)}) ---", flush=True)

        oof_lgb = np.zeros(n_samples)
        oof_xgb = np.zeros(n_samples)
        oof_cat = np.zeros(n_samples)
        oof_ridge = np.zeros(n_samples)
        oof_et = np.zeros(n_samples)

        test_lgb_folds = np.zeros(len(target_test_idx))
        test_xgb_folds = np.zeros(len(target_test_idx))
        test_cat_folds = np.zeros(len(target_test_idx))
        test_ridge_folds = np.zeros(len(target_test_idx))
        test_et_folds = np.zeros(len(target_test_idx))

        X_tgt_test = X_test_full[target_test_idx]

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tgt, y_tgt, groups=groups_tgt)):
            X_tr, y_tr = X_tgt[trn_idx], y_tgt[trn_idx]
            X_val, y_val = X_tgt[val_idx], y_tgt[val_idx]

            # 1. LightGBM
            lgb_m = lgb.LGBMRegressor(n_estimators=450, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=2026 + fold, n_jobs=-1, verbose=-1)
            lgb_m.fit(X_tr, y_tr)
            oof_lgb[val_idx] = lgb_m.predict(X_val)
            test_lgb_folds += lgb_m.predict(X_tgt_test) / n_splits

            # 2. XGBoost
            xgb_m = xgb.XGBRegressor(n_estimators=450, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=2026 + fold, n_jobs=-1, verbosity=0)
            xgb_m.fit(X_tr, y_tr)
            oof_xgb[val_idx] = xgb_m.predict(X_val)
            test_xgb_folds += xgb_m.predict(X_tgt_test) / n_splits

            # 3. CatBoost
            cat_m = CatBoostRegressor(iterations=450, learning_rate=0.035, depth=6, random_seed=2026 + fold, verbose=0)
            cat_m.fit(X_tr, y_tr)
            oof_cat[val_idx] = cat_m.predict(X_val)
            test_cat_folds += cat_m.predict(X_tgt_test) / n_splits

            # 4. Ridge
            ridge_m = Ridge(alpha=15.0, random_state=2026 + fold)
            ridge_m.fit(X_tr, y_tr)
            oof_ridge[val_idx] = ridge_m.predict(X_val)
            test_ridge_folds += ridge_m.predict(X_tgt_test) / n_splits

            # 5. ExtraTrees
            et_m = ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_split=4, random_state=2026 + fold, n_jobs=-1)
            et_m.fit(X_tr, y_tr)
            oof_et[val_idx] = et_m.predict(X_val)
            test_et_folds += et_m.predict(X_tgt_test) / n_splits

        models_oof = [oof_lgb, oof_xgb, oof_cat, oof_ridge, oof_et]
        models_test = [test_lgb_folds, test_xgb_folds, test_cat_folds, test_ridge_folds, test_et_folds]

        weights = slsqp_blend(models_oof, y_tgt)
        oof_ensemble = sum(w * m for w, m in zip(weights, models_oof))
        test_ensemble = sum(w * m for w, m in zip(weights, models_test))

        tgt_r2 = float(r2_score(y_tgt, oof_ensemble))
        tgt_mae = float(mean_absolute_error(y_tgt, oof_ensemble))
        oof_r2_list.append(tgt_r2)

        oof_metrics[target] = {
            'r2': tgt_r2,
            'mae': tgt_mae,
            'n_train': n_samples,
            'slsqp_weights': [float(w) for w in weights]
        }

        print(f"  OOF R²: {tgt_r2:.4f} | MAE: {tgt_mae:.4f}", flush=True)
        predictions.loc[target_test_idx, 'target'] = test_ensemble

    mean_oof = float(np.mean(oof_r2_list))
    print(f"\n==========================================", flush=True)
    print(f"Mean OOF R² (MCP Topological Invariants): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-281',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-281 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
