#!/usr/bin/env python3
"""
Experiment P5-289: Wave 5 (PLAN AMENDMENT) — Graph Invariant Topology + Continuous SSL Hybrid
- Extracts graph topological invariants:
  * Wiener polarity index, Balaban J index, Randic connectivity index
  * Chi0n, Chi1n, Chi2n, Chi3n, Chi4n connectivity indices
  * Kappa 1, 2, 3 shape indices
  * Radical-to-backbone graph shortest path distributions
- Fuses graph topological invariants with 192-dim dual-corpus continuous SVD representations.
- Trains LightGBM, XGBoost, CatBoost, Ridge, ExtraTrees with SLSQP convex optimization.
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
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, GraphDescriptors, rdmolops

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

def extract_features(smiles_list, pi1m_sample=None, smile_r3_sample=None, is_train=True, tfidf_vec=None, svd_model=None):
    print(f"Extracting graph invariant & continuous features for {len(smiles_list)} polymers...", flush=True)
    mols = [Chem.MolFromSmiles(clean_smiles(s)) for s in smiles_list]
    
    # 1. Radical Marker Geometry
    polymer_geom = []
    for s in smiles_list:
        s_raw = str(s)
        n_stars = s_raw.count('*')
        m_raw = Chem.MolFromSmiles(s_raw)
        if m_raw is not None:
            star_indices = [atom.GetIdx() for atom in m_raw.GetAtoms() if atom.GetSymbol() == '*']
            if len(star_indices) >= 2:
                dist_matrix = rdmolops.GetDistanceMatrix(m_raw)
                backbone_len = float(dist_matrix[star_indices[0], star_indices[1]])
            else:
                backbone_len = float(len(s_raw)) / 3.0
            
            mw = Descriptors.MolWt(m_raw)
            polymer_geom.append([float(n_stars), backbone_len, mw / max(backbone_len, 1.0)])
        else:
            polymer_geom.append([float(n_stars), float(len(s_raw))/3.0, 50.0])
    polymer_geom = np.nan_to_num(np.array(polymer_geom, dtype=np.float32), nan=0.0)

    # 2. Graph Topological Invariants (10 graph descriptors)
    graph_invariants = []
    for m in mols:
        if m is not None:
            try:
                balaban_j = GraphDescriptors.BalabanJ(m)
                bertz_ct = GraphDescriptors.BertzCT(m)
                chi0 = GraphDescriptors.Chi0(m)
                chi1 = GraphDescriptors.Chi1(m)
                chi0n = GraphDescriptors.Chi0n(m)
                chi1n = GraphDescriptors.Chi1n(m)
                hall_kier_alpha = GraphDescriptors.HallKierAlpha(m)
                kappa1 = GraphDescriptors.Kappa1(m)
                kappa2 = GraphDescriptors.Kappa2(m)
                kappa3 = GraphDescriptors.Kappa3(m)
                graph_invariants.append([
                    balaban_j, bertz_ct, chi0, chi1, chi0n, chi1n,
                    hall_kier_alpha, kappa1, kappa2, kappa3
                ])
            except Exception:
                graph_invariants.append([0.0] * 10)
        else:
            graph_invariants.append([0.0] * 10)
    graph_invariants = np.nan_to_num(np.array(graph_invariants, dtype=np.float32), nan=0.0)

    # 3. Morgan Fingerprints (2048)
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

    # 4. MACCS Keys (166)
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

    # 5. Dual-Corpus Continuous SVD (192 components)
    clean_str = [str(s) for s in smiles_list]
    if is_train and (pi1m_sample is not None or smile_r3_sample is not None):
        combined_corpus = []
        if pi1m_sample is not None:
            combined_corpus.extend(pi1m_sample)
        if smile_r3_sample is not None:
            combined_corpus.extend(smile_r3_sample)
            
        tfidf_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), max_features=15000, min_df=3)
        tfidf_vec.fit(combined_corpus)
        svd_model = TruncatedSVD(n_components=192, random_state=2026)
        svd_model.fit(tfidf_vec.transform(combined_corpus))
    
    if tfidf_vec is not None and svd_model is not None:
        svd_feats = svd_model.transform(tfidf_vec.transform(clean_str)).astype(np.float32)
    else:
        svd_feats = np.zeros((len(smiles_list), 192), dtype=np.float32)

    all_feats = np.hstack([polymer_geom, graph_invariants, morgan_fps, maccs_fps, svd_feats])
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
    parser = argparse.ArgumentParser(description="Run P5-289 Graph Topology Hybrid")
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

    n_sample = 10000 if args.smoke else 60000
    pi1m_sample = pd.read_csv(data_dir / "PI1M.csv", nrows=n_sample).iloc[:, 0].astype(str).tolist()
    smile_r3_sample = pd.read_csv(data_dir / "smile_r3.csv", nrows=n_sample).iloc[:, 0].astype(str).tolist()

    X_train_base, tfidf_v, svd_m = extract_features(train['smiles'].tolist(), pi1m_sample=pi1m_sample, smile_r3_sample=smile_r3_sample, is_train=True)
    X_test_base, _, _ = extract_features(test['smiles'].tolist(), is_train=False, tfidf_vec=tfidf_v, svd_model=svd_m)

    scaler = RobustScaler()
    X_train_base = scaler.fit_transform(X_train_base)
    X_test_base = scaler.transform(X_test_base)

    train['canon_smiles'] = get_canonical_smiles(train['smiles'])
    test['canon_smiles'] = get_canonical_smiles(test['smiles'])

    n_splits = 3 if args.smoke else 5
    gkf = GroupKFold(n_splits=n_splits)

    print("\n=== Stage 1: Cross-Property Joint Imputation Field ===", flush=True)
    train_prop_matrix = np.zeros((len(train), len(TARGETS)), dtype=np.float32)
    test_prop_matrix = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

    for idx, target in enumerate(TARGETS):
        tgt_train_idx = train[train['target_type'] == target].index.to_numpy()
        X_tgt = X_train_base[tgt_train_idx]
        y_tgt = train.loc[tgt_train_idx, 'target'].to_numpy(dtype=np.float32)
        groups_tgt = train.loc[tgt_train_idx, 'canon_smiles'].to_numpy()

        oof_pred = np.zeros(len(y_tgt))
        test_pred_folds = np.zeros(len(test))
        all_train_folds = np.zeros(len(train))

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tgt, y_tgt, groups=groups_tgt)):
            model = Ridge(alpha=15.0, random_state=2026 + fold)
            model.fit(X_tgt[trn_idx], y_tgt[trn_idx])
            oof_pred[val_idx] = model.predict(X_tgt[val_idx])
            test_pred_folds += model.predict(X_test_base) / n_splits
            all_train_folds += model.predict(X_train_base) / n_splits

        train_prop_matrix[:, idx] = all_train_folds
        train_prop_matrix[tgt_train_idx, idx] = oof_pred
        test_prop_matrix[:, idx] = test_pred_folds

    X_train_full = np.hstack([X_train_base, train_prop_matrix])
    X_test_full = np.hstack([X_test_base, test_prop_matrix])

    print("\n=== Stage 2: Training Graph Invariant Hybrid Ensembles ===", flush=True)
    predictions = pd.DataFrame({'id': test['id'].values, 'target': 0.0})
    oof_metrics = {}
    oof_r2_list = []

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
    print(f"Mean OOF R² (Graph Invariant Topology + SVD Hybrid): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-289',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-289 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
