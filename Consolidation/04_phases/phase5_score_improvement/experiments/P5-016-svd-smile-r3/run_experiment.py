#!/usr/bin/env python3
"""
Experiment P5-016: smile_r3 Self-Supervised Continuous Embeddings (TF-IDF + TruncatedSVD 128)
- Learns sub-monomer grammar from 100k smile_r3.csv molecules from scratch (no external weights)
- Fits TruncatedSVD(128) on character n-grams (2-6 chars)
- Combines continuous latent coordinates with Morgan + MACCS + physical descriptors
- 5-Model NNLS Stacking with GroupKFold CV
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
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

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

def extract_base_features(smiles_list):
    print(f"Extracting base chemical features for {len(smiles_list)} polymers...", flush=True)
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

    # 3. Selected RDKit 2D Descriptors
    desc_names = [
        'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors',
        'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'NumHeteroatoms',
        'FractionCSP3', 'HeavyAtomCount', 'NumValenceElectrons'
    ]
    desc_matrix = []
    for m in mols:
        if m is not None:
            vals = []
            for d in desc_names:
                fn = getattr(Descriptors, d, None)
                if fn is not None:
                    try:
                        vals.append(float(fn(m)))
                    except:
                        vals.append(0.0)
                else:
                    vals.append(0.0)
            desc_matrix.append(vals)
        else:
            desc_matrix.append([0.0] * len(desc_names))
    desc_matrix = np.nan_to_num(np.array(desc_matrix, dtype=np.float32), nan=0.0)

    return np.hstack([morgan_fps, maccs_fps, desc_matrix])

def train_nnls_blend(oof_preds_list, y_true):
    A = np.column_stack(oof_preds_list)
    weights, _ = nnls(A, y_true)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(len(oof_preds_list)) / len(oof_preds_list)
    return weights

def main():
    parser = argparse.ArgumentParser(description="Run P5-016 smile_r3 SVD SSL")
    parser.add_argument("--data-dir", default="../../Dataset", help="Path to competition dataset")
    parser.add_argument("--output-dir", default=".", help="Path to output directory")
    parser.add_argument("--sample-size", type=int, default=100000, help="smile_r3 sample size")
    parser.add_argument("--n-components", type=int, default=128, help="SVD components")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train, test, and smile_r3 datasets...", flush=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    
    sample_size = 10000 if args.smoke else args.sample_size
    smile_r3_sample = pd.read_csv(data_dir / "smile_r3.csv", nrows=sample_size)
    smile_col = smile_r3_sample.columns[0]
    unlabeled_smiles = smile_r3_sample[smile_col].astype(str).tolist()

    print(f"Fitting from-scratch TF-IDF on {len(unlabeled_smiles):,} smile_r3 molecules...", flush=True)
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), max_features=10000, min_df=3)
    tfidf.fit(unlabeled_smiles)

    print(f"Fitting TruncatedSVD({args.n_components}) on smile_r3 representations...", flush=True)
    unlabeled_tfidf = tfidf.transform(unlabeled_smiles)
    svd = TruncatedSVD(n_components=args.n_components, random_state=2026)
    svd.fit(unlabeled_tfidf)
    print(f"Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}", flush=True)

    print("Transforming train and test polymers into SVD latent coordinates...", flush=True)
    train_svd = svd.transform(tfidf.transform(train['smiles'].astype(str))).astype(np.float32)
    test_svd = svd.transform(tfidf.transform(test['smiles'].astype(str))).astype(np.float32)

    # Base features
    train_base = extract_base_features(train['smiles'].tolist())
    test_base = extract_base_features(test['smiles'].tolist())

    X_train_full = np.hstack([train_base, train_svd])
    X_test_full = np.hstack([test_base, test_svd])

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

    print("\n=== Training Stacking Models with SSL Continuous Embeddings ===", flush=True)

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
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100 if args.smoke else 450,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
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
                n_estimators=100 if args.smoke else 450,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
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
                iterations=100 if args.smoke else 450,
                learning_rate=0.04,
                depth=6,
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
                n_estimators=50 if args.smoke else 200,
                max_depth=15,
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
    print(f"Mean OOF R² (with smile_r3 SVD SSL): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-016',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'sample_size': sample_size,
        'n_components': args.n_components,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-016 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
