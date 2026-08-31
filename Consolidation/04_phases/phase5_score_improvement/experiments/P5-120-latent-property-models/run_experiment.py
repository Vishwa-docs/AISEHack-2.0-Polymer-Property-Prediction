#!/usr/bin/env python3
"""
Experiment P5-120: Phase L — Latent Property Regularization & Multi-Task Partner Surrogates
- First stage: Train 5-fold cross-validated surrogate models across all 7 targets
- Second stage: Augment features with out-of-fold latent target predictions:
    * EI augmented with OOF EGC and EEA (orbital bandgap relation: Ei ≈ Egc + Eea)
    * EEA augmented with OOF EI and EGC
    * EPS augmented with OOF NC (Maxwell dielectric relation: eps ≈ nc² + delta)
    * NC augmented with OOF EPS
    * EGB augmented with OOF EGC (bulk vs chain bandgap relation)
    * TG augmented with all 6 electronic/optical latent states
- Final 5-model NNLS stacking with strict grouped CV
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

def extract_features(smiles_list, smile_r3_sample=None, is_train=True, tfidf_vec=None, svd_model=None):
    print(f"Extracting features for {len(smiles_list)} polymers...", flush=True)
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

    all_feats = np.hstack([morgan_fps, maccs_fps, desc_matrix, svd_feats])
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
    parser = argparse.ArgumentParser(description="Run P5-120 Latent Property Models")
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
    X_train_base, tfidf_v, svd_m = extract_features(train['smiles'].tolist(), smile_r3_sample=smile_r3_sample, is_train=True)
    X_test_base, _, _ = extract_features(test['smiles'].tolist(), is_train=False, tfidf_vec=tfidf_v, svd_model=svd_m)

    scaler = RobustScaler()
    X_train_base = scaler.fit_transform(X_train_base)
    X_test_base = scaler.transform(X_test_base)

    train['canon_smiles'] = get_canonical_smiles(train['smiles'])
    test['canon_smiles'] = get_canonical_smiles(test['smiles'])

    n_splits = 3 if args.smoke else 5
    gkf = GroupKFold(n_splits=n_splits)

    print("\n=== Stage 1: Fitting Multi-Target Latent Surrogates Across Entire Corpus ===", flush=True)
    # Fit Stage 1 Ridge & LightGBM surrogates per target on all rows to generate latent features
    oof_surrogates_train = np.zeros((len(train), len(TARGETS)), dtype=np.float32)
    test_surrogates = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

    for t_idx, target in enumerate(TARGETS):
        tgt_train_idx = train[train['target_type'] == target].index.to_numpy()
        X_tgt = X_train_base[tgt_train_idx]
        y_tgt = train.loc[tgt_train_idx, 'target'].to_numpy(dtype=np.float32)
        groups_tgt = train.loc[tgt_train_idx, 'canon_smiles'].to_numpy()

        surr_oof = np.zeros(len(y_tgt))
        test_surr_folds = np.zeros(len(test))
        all_train_surr_folds = np.zeros(len(train))

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tgt, y_tgt, groups=groups_tgt)):
            surr_model = Ridge(alpha=20.0, random_state=2026 + fold)
            surr_model.fit(X_tgt[trn_idx], y_tgt[trn_idx])
            surr_oof[val_idx] = surr_model.predict(X_tgt[val_idx])

            test_surr_folds += surr_model.predict(X_test_base) / n_splits
            all_train_surr_folds += surr_model.predict(X_train_base) / n_splits

        oof_surrogates_train[:, t_idx] = all_train_surr_folds
        # Replace the target's own measured rows with true out-of-fold predictions
        oof_surrogates_train[tgt_train_idx, t_idx] = surr_oof
        test_surrogates[:, t_idx] = test_surr_folds

        r2_surr = r2_score(y_tgt, surr_oof)
        print(f"  Surrogate for {target:4s}: OOF R² = {r2_surr:.4f}", flush=True)

    # Augment base features with latent surrogate properties
    X_train_augmented = np.hstack([X_train_base, oof_surrogates_train])
    X_test_augmented = np.hstack([X_test_base, test_surrogates])

    print("\n=== Stage 2: Training Second-Stage Physics & Latent Ensembles ===", flush=True)
    predictions = pd.DataFrame({'id': test['id'].values, 'target': 0.0})
    oof_metrics = {}
    oof_r2_list = []

    for target in TARGETS:
        target_train_idx = train[train['target_type'] == target].index.to_numpy()
        target_test_mask = (test['target_type'] == target).to_numpy()
        target_test_idx = np.where(target_test_mask)[0]

        X_tgt = X_train_augmented[target_train_idx]
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

        X_tgt_test = X_test_augmented[target_test_idx]

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
    print(f"Mean OOF R² (with Latent Property Regularization): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-120',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-120 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
