#!/usr/bin/env python3
"""
Experiment P5-045: Phase E — Tg Specialist Push (Group Contribution & Chain Stiffness Descriptors)
- Engineered polymer physical descriptors specifically tailored for glass transition (Tg):
    * Rotatable bond density (backbone flexibility)
    * Aromatic / ring stiffness ratios
    * Cohesive energy density proxies (TPSA/MW, H-bond donor/acceptor interaction density)
    * Labute ASA, MolMR, FractionCSP3
- smile_r3 SVD SSL continuous grammar embeddings (128 dims)
- 5-Model NNLS Stacking with Grouped Cross-Validation
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
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
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, rdMolDescriptors

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

def extract_tg_specialist_features(smiles_list, smile_r3_sample=None, is_train=True, tfidf_vec=None, svd_model=None):
    print(f"Extracting Tg specialist physical descriptors for {len(smiles_list)} polymers...", flush=True)
    mols = [Chem.MolFromSmiles(clean_smiles(s)) for s in smiles_list]
    
    # 1. Morgan Fingerprints (2048)
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

    # 2. MACCS Keys (166)
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

    # 3. Domain-Specific Tg Chain Stiffness & Intermolecular Physics Descriptors
    tg_physics = []
    for m in mols:
        if m is not None:
            mw = Descriptors.MolWt(m)
            num_heavy = max(m.GetNumHeavyAtoms(), 1)
            num_rot = Descriptors.NumRotatableBonds(m)
            tpsa = Descriptors.TPSA(m)
            logp = Descriptors.MolLogP(m)
            hbd = Descriptors.NumHDonors(m)
            hba = Descriptors.NumHAcceptors(m)
            n_aromatic = Descriptors.NumAromaticRings(m)
            n_rings = Descriptors.RingCount(m)
            f_csp3 = Descriptors.FractionCSP3(m)
            labute_asa = Descriptors.LabuteASA(m)
            mol_mr = Crippen.MolMR(m)
            
            # Derived Tg features
            flexibility_ratio = num_rot / num_heavy
            aromatic_density = n_aromatic / num_heavy
            ring_density = n_rings / num_heavy
            cohesive_energy_proxy = tpsa / (mw + 1e-5)
            hbond_density = (hbd + hba) / (mw + 1e-5)
            polar_volume_ratio = tpsa / (labute_asa + 1e-5)

            vals = [
                mw, logp, tpsa, num_rot, hbd, hba, n_aromatic, n_rings, f_csp3,
                labute_asa, mol_mr, flexibility_ratio, aromatic_density,
                ring_density, cohesive_energy_proxy, hbond_density, polar_volume_ratio
            ]
            tg_physics.append(vals)
        else:
            tg_physics.append([0.0] * 17)
    tg_physics = np.nan_to_num(np.array(tg_physics, dtype=np.float32), nan=0.0)

    # 4. smile_r3 SVD Continuous Embeddings (128)
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

    all_feats = np.hstack([morgan_fps, maccs_fps, tg_physics, svd_feats])
    return all_feats, tfidf_vec, svd_model

def train_nnls_blend(oof_preds_list, y_true):
    A = np.column_stack(oof_preds_list)
    weights, _ = nnls(A, y_true)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(len(oof_preds_list)) / len(oof_preds_list)
    return weights

def main():
    parser = argparse.ArgumentParser(description="Run P5-045 Tg Specialist Push")
    parser.add_argument("--data-dir", default="../../Dataset", help="Path to competition dataset")
    parser.add_argument("--output-dir", default=".", help="Path to output directory")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train, test, and smile_r3 datasets...", flush=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    n_sample_r3 = 10000 if args.smoke else 75000
    smile_r3_sample = pd.read_csv(data_dir / "smile_r3.csv", nrows=n_sample_r3).iloc[:, 0].astype(str).tolist()

    # Feature extraction
    X_train_full, tfidf_v, svd_m = extract_tg_specialist_features(train['smiles'].tolist(), smile_r3_sample=smile_r3_sample, is_train=True)
    X_test_full, _, _ = extract_tg_specialist_features(test['smiles'].tolist(), is_train=False, tfidf_vec=tfidf_v, svd_model=svd_m)

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

    print("\n=== Training Tg-Specialist Models Across 7 Targets ===", flush=True)

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

        # For Tg, use deeper trees and tuned shrinkage
        n_trees = 550 if target == 'tg' else (100 if args.smoke else 400)
        lr = 0.025 if target == 'tg' else 0.03

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tgt, y_tgt, groups=groups_tgt)):
            X_tr, y_tr = X_tgt[trn_idx], y_tgt[trn_idx]
            X_val, y_val = X_tgt[val_idx], y_tgt[val_idx]

            # 1. LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=n_trees,
                learning_rate=lr,
                num_leaves=40 if target == 'tg' else 31,
                subsample=0.85,
                colsample_bytree=0.8,
                random_state=2026 + fold,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_tr, y_tr)
            oof_lgb[val_idx] = lgb_model.predict(X_val)
            test_lgb_folds += lgb_model.predict(X_tgt_test) / n_splits

            # 2. XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=n_trees,
                learning_rate=lr,
                max_depth=7 if target == 'tg' else 6,
                subsample=0.85,
                colsample_bytree=0.8,
                random_state=2026 + fold,
                n_jobs=-1,
                verbosity=0
            )
            xgb_model.fit(X_tr, y_tr)
            oof_xgb[val_idx] = xgb_model.predict(X_val)
            test_xgb_folds += xgb_model.predict(X_tgt_test) / n_splits

            # 3. CatBoost
            cat_model = CatBoostRegressor(
                iterations=n_trees,
                learning_rate=0.035,
                depth=7 if target == 'tg' else 6,
                random_seed=2026 + fold,
                verbose=0
            )
            cat_model.fit(X_tr, y_tr)
            oof_cat[val_idx] = cat_model.predict(X_val)
            test_cat_folds += cat_model.predict(X_tgt_test) / n_splits

            # 4. Ridge
            ridge_model = Ridge(alpha=15.0, random_state=2026 + fold)
            ridge_model.fit(X_tr, y_tr)
            oof_ridge[val_idx] = ridge_model.predict(X_val)
            test_ridge_folds += ridge_model.predict(X_tgt_test) / n_splits

            # 5. ExtraTrees
            et_model = ExtraTreesRegressor(
                n_estimators=75 if args.smoke else 250,
                max_depth=18 if target == 'tg' else 15,
                min_samples_split=4,
                random_state=2026 + fold,
                n_jobs=-1
            )
            et_model.fit(X_tr, y_tr)
            oof_et[val_idx] = et_model.predict(X_val)
            test_et_folds += et_model.predict(X_tgt_test) / n_splits

        models_oof = [oof_lgb, oof_xgb, oof_cat, oof_ridge, oof_et]
        models_test = [test_lgb_folds, test_xgb_folds, test_cat_folds, test_ridge_folds, test_et_folds]
        weights = train_nnls_blend(models_oof, y_tgt)

        oof_ensemble = sum(w * m for w, m in zip(weights, models_oof))
        test_ensemble = sum(w * m for w, m in zip(weights, models_test))

        tgt_r2 = float(r2_score(y_tgt, oof_ensemble))
        tgt_mae = float(mean_absolute_error(y_tgt, oof_ensemble))
        oof_r2_list.append(tgt_r2)

        oof_metrics[target] = {
            'r2': tgt_r2,
            'mae': tgt_mae,
            'n_train': n_samples,
            'nnls_weights': {
                'lgb': float(weights[0]),
                'xgb': float(weights[1]),
                'cat': float(weights[2]),
                'ridge': float(weights[3]),
                'et': float(weights[4])
            }
        }

        print(f"  OOF R²: {tgt_r2:.4f} | MAE: {tgt_mae:.4f}", flush=True)
        predictions.loc[target_test_idx, 'target'] = test_ensemble

    mean_oof = float(np.mean(oof_r2_list))
    print(f"\n==========================================", flush=True)
    print(f"Mean OOF R² (with Tg Specialist Descriptors): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-045',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-045 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
