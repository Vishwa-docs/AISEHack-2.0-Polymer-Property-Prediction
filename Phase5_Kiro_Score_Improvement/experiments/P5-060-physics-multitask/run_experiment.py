#!/usr/bin/env python3
"""
Experiment P5-060: Phase D/F — Physics-Constrained Multi-Task Joint Model
- Physical constraints:
    1. Bandgap energy balance: Ei = Egc + Eea + delta_pol
    2. Dielectric / refractive index Maxwell relation: eps = nc^2 + delta_ionic
    3. Bulk / chain bandgap relation: egb = egc - delta_bulk
- Physics-informed residual learning on small targets (EI, EEA, EPS, NC)
- smile_r3 SVD continuous embeddings + engineered physics descriptors
- Strict grouped CV with NNLS ensemble blending
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
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen

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

    # 3. Physical & Electronic Descriptors
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
            n_rings = Descriptors.RingCount(m)
            f_csp3 = Descriptors.FractionCSP3(m)
            n_heavy = max(m.GetNumHeavyAtoms(), 1)
            mr = Crippen.MolMR(m)
            val_elec = Descriptors.NumValenceElectrons(m)

            # Polarizability proxy: MR / MW
            pol_proxy = mr / (mw + 1e-5)
            # Ionic dipole proxy: TPSA / MW
            ionic_proxy = tpsa / (mw + 1e-5)
            # Conjugation index: aromatic atoms / heavy atoms
            conj_index = n_arom / n_heavy

            vals = [
                mw, logp, tpsa, rot_bonds, hbd, hba, n_arom, n_rings, f_csp3,
                n_heavy, mr, val_elec, pol_proxy, ionic_proxy, conj_index
            ]
            desc_matrix.append(vals)
        else:
            desc_matrix.append([0.0] * 15)
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

def fit_target_ensemble(X_tgt, y_tgt, groups_tgt, X_tgt_test, gkf, n_splits, args, n_trees=400, lr=0.03):
    n_samples = len(y_tgt)
    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)
    oof_ridge = np.zeros(n_samples)
    oof_et = np.zeros(n_samples)

    test_lgb_folds = np.zeros(len(X_tgt_test))
    test_xgb_folds = np.zeros(len(X_tgt_test))
    test_cat_folds = np.zeros(len(X_tgt_test))
    test_ridge_folds = np.zeros(len(X_tgt_test))
    test_et_folds = np.zeros(len(X_tgt_test))

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tgt, y_tgt, groups=groups_tgt)):
        X_tr, y_tr = X_tgt[trn_idx], y_tgt[trn_idx]
        X_val, y_val = X_tgt[val_idx], y_tgt[val_idx]

        # 1. LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100 if args.smoke else n_trees,
            learning_rate=lr,
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
            n_estimators=100 if args.smoke else n_trees,
            learning_rate=lr,
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
            iterations=100 if args.smoke else n_trees,
            learning_rate=lr + 0.01,
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

    return oof_ensemble, test_ensemble, weights

def main():
    parser = argparse.ArgumentParser(description="Run P5-060 Physics Multi-Task Model")
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

    X_train_full, tfidf_v, svd_m = extract_features(train['smiles'].tolist(), smile_r3_sample=smile_r3_sample, is_train=True)
    X_test_full, _, _ = extract_features(test['smiles'].tolist(), is_train=False, tfidf_vec=tfidf_v, svd_model=svd_m)

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

    print("\n=== Stage 1: Training Primary Core Physical Predictors (EGC, NC, TG) ===", flush=True)

    # 1. EGC (Chain bandgap, N=2028)
    egc_trn_idx = train[train['target_type'] == 'egc'].index.to_numpy()
    egc_tst_idx = np.where(test['target_type'] == 'egc')[0]
    oof_egc, test_egc, _ = fit_target_ensemble(
        X_train_full[egc_trn_idx], train.loc[egc_trn_idx, 'target'].to_numpy(dtype=np.float32),
        train.loc[egc_trn_idx, 'canon_smiles'].to_numpy(),
        X_test_full[egc_tst_idx], gkf, n_splits, args, n_trees=450, lr=0.03
    )
    predictions.loc[egc_tst_idx, 'target'] = test_egc
    r2_egc = float(r2_score(train.loc[egc_trn_idx, 'target'], oof_egc))
    oof_r2_list.append(r2_egc)
    oof_metrics['egc'] = {'r2': r2_egc}
    print(f"  EGC OOF R²: {r2_egc:.4f}", flush=True)

    # Full corpus EGC predictor for physics features
    full_egc_model = Ridge(alpha=15.0)
    full_egc_model.fit(X_train_full[egc_trn_idx], train.loc[egc_trn_idx, 'target'])
    pred_egc_all_train = full_egc_model.predict(X_train_full)
    pred_egc_all_test = full_egc_model.predict(X_test_full)

    # 2. NC (Refractive Index, N=229)
    nc_trn_idx = train[train['target_type'] == 'nc'].index.to_numpy()
    nc_tst_idx = np.where(test['target_type'] == 'nc')[0]
    oof_nc, test_nc, _ = fit_target_ensemble(
        X_train_full[nc_trn_idx], train.loc[nc_trn_idx, 'target'].to_numpy(dtype=np.float32),
        train.loc[nc_trn_idx, 'canon_smiles'].to_numpy(),
        X_test_full[nc_tst_idx], gkf, n_splits, args, n_trees=350, lr=0.03
    )
    predictions.loc[nc_tst_idx, 'target'] = test_nc
    r2_nc = float(r2_score(train.loc[nc_trn_idx, 'target'], oof_nc))
    oof_r2_list.append(r2_nc)
    oof_metrics['nc'] = {'r2': r2_nc}
    print(f"  NC OOF R²: {r2_nc:.4f}", flush=True)

    full_nc_model = Ridge(alpha=20.0)
    full_nc_model.fit(X_train_full[nc_trn_idx], train.loc[nc_trn_idx, 'target'])
    pred_nc_all_train = full_nc_model.predict(X_train_full)
    pred_nc_all_test = full_nc_model.predict(X_test_full)

    # 3. TG (Glass transition, N=4143)
    tg_trn_idx = train[train['target_type'] == 'tg'].index.to_numpy()
    tg_tst_idx = np.where(test['target_type'] == 'tg')[0]
    oof_tg, test_tg, _ = fit_target_ensemble(
        X_train_full[tg_trn_idx], train.loc[tg_trn_idx, 'target'].to_numpy(dtype=np.float32),
        train.loc[tg_trn_idx, 'canon_smiles'].to_numpy(),
        X_test_full[tg_tst_idx], gkf, n_splits, args, n_trees=500, lr=0.025
    )
    predictions.loc[tg_tst_idx, 'target'] = test_tg
    r2_tg = float(r2_score(train.loc[tg_trn_idx, 'target'], oof_tg))
    oof_r2_list.append(r2_tg)
    oof_metrics['tg'] = {'r2': r2_tg}
    print(f"  TG OOF R²: {r2_tg:.4f}", flush=True)

    print("\n=== Stage 2: Training Physics-Coupled Targets (EPS, EEA, EI, EGB) ===", flush=True)

    # 4. EPS (Dielectric Constant: eps ≈ nc^2 + ionic_contrib)
    eps_trn_idx = train[train['target_type'] == 'eps'].index.to_numpy()
    eps_tst_idx = np.where(test['target_type'] == 'eps')[0]

    # Inject physics feature: nc^2
    phys_eps_feat_train = (pred_nc_all_train ** 2).reshape(-1, 1)
    phys_eps_feat_test = (pred_nc_all_test ** 2).reshape(-1, 1)

    X_train_eps = np.hstack([X_train_full[eps_trn_idx], phys_eps_feat_train[eps_trn_idx]])
    X_test_eps = np.hstack([X_test_full[eps_tst_idx], phys_eps_feat_test[eps_tst_idx]])

    oof_eps, test_eps, _ = fit_target_ensemble(
        X_train_eps, train.loc[eps_trn_idx, 'target'].to_numpy(dtype=np.float32),
        train.loc[eps_trn_idx, 'canon_smiles'].to_numpy(),
        X_test_eps, gkf, n_splits, args, n_trees=350, lr=0.03
    )
    predictions.loc[eps_tst_idx, 'target'] = test_eps
    r2_eps = float(r2_score(train.loc[eps_trn_idx, 'target'], oof_eps))
    oof_r2_list.append(r2_eps)
    oof_metrics['eps'] = {'r2': r2_eps}
    print(f"  EPS (with Maxwell nc^2 constraint) OOF R²: {r2_eps:.4f}", flush=True)

    # 5. EEA (Electron Affinity, N=221)
    eea_trn_idx = train[train['target_type'] == 'eea'].index.to_numpy()
    eea_tst_idx = np.where(test['target_type'] == 'eea')[0]

    # Augment with predicted EGC
    X_train_eea = np.hstack([X_train_full[eea_trn_idx], pred_egc_all_train[eea_trn_idx].reshape(-1, 1)])
    X_test_eea = np.hstack([X_test_full[eea_tst_idx], pred_egc_all_test[eea_tst_idx].reshape(-1, 1)])

    oof_eea, test_eea, _ = fit_target_ensemble(
        X_train_eea, train.loc[eea_trn_idx, 'target'].to_numpy(dtype=np.float32),
        train.loc[eea_trn_idx, 'canon_smiles'].to_numpy(),
        X_test_eea, gkf, n_splits, args, n_trees=350, lr=0.03
    )
    predictions.loc[eea_tst_idx, 'target'] = test_eea
    r2_eea = float(r2_score(train.loc[eea_trn_idx, 'target'], oof_eea))
    oof_r2_list.append(r2_eea)
    oof_metrics['eea'] = {'r2': r2_eea}
    print(f"  EEA OOF R²: {r2_eea:.4f}", flush=True)

    full_eea_model = Ridge(alpha=20.0)
    full_eea_model.fit(X_train_full[eea_trn_idx], train.loc[eea_trn_idx, 'target'])
    pred_eea_all_train = full_eea_model.predict(X_train_full)
    pred_eea_all_test = full_eea_model.predict(X_test_full)

    # 6. EI (Ionisation Energy: Ei ≈ Egc + Eea)
    ei_trn_idx = train[train['target_type'] == 'ei'].index.to_numpy()
    ei_tst_idx = np.where(test['target_type'] == 'ei')[0]

    # Inject physics feature: (Egc + Eea)
    phys_ei_feat_train = (pred_egc_all_train + pred_eea_all_train).reshape(-1, 1)
    phys_ei_feat_test = (pred_egc_all_test + pred_eea_all_test).reshape(-1, 1)

    X_train_ei = np.hstack([X_train_full[ei_trn_idx], phys_ei_feat_train[ei_trn_idx], pred_egc_all_train[ei_trn_idx].reshape(-1, 1), pred_eea_all_train[ei_trn_idx].reshape(-1, 1)])
    X_test_ei = np.hstack([X_test_full[ei_tst_idx], phys_ei_feat_test[ei_tst_idx], pred_egc_all_test[ei_tst_idx].reshape(-1, 1), pred_eea_all_test[ei_tst_idx].reshape(-1, 1)])

    oof_ei, test_ei, _ = fit_target_ensemble(
        X_train_ei, train.loc[ei_trn_idx, 'target'].to_numpy(dtype=np.float32),
        train.loc[ei_trn_idx, 'canon_smiles'].to_numpy(),
        X_test_ei, gkf, n_splits, args, n_trees=350, lr=0.03
    )
    predictions.loc[ei_tst_idx, 'target'] = test_ei
    r2_ei = float(r2_score(train.loc[ei_trn_idx, 'target'], oof_ei))
    oof_r2_list.append(r2_ei)
    oof_metrics['ei'] = {'r2': r2_ei}
    print(f"  EI (with Frontier Orbital Physics Ei ≈ Egc + Eea) OOF R²: {r2_ei:.4f}", flush=True)

    # 7. EGB (Bulk Bandgap: egb ≈ egc - delta_bulk)
    egb_trn_idx = train[train['target_type'] == 'egb'].index.to_numpy()
    egb_tst_idx = np.where(test['target_type'] == 'egb')[0]

    X_train_egb = np.hstack([X_train_full[egb_trn_idx], pred_egc_all_train[egb_trn_idx].reshape(-1, 1)])
    X_test_egb = np.hstack([X_test_full[egb_tst_idx], pred_egc_all_test[egb_tst_idx].reshape(-1, 1)])

    oof_egb, test_egb, _ = fit_target_ensemble(
        X_train_egb, train.loc[egb_trn_idx, 'target'].to_numpy(dtype=np.float32),
        train.loc[egb_trn_idx, 'canon_smiles'].to_numpy(),
        X_test_egb, gkf, n_splits, args, n_trees=350, lr=0.03
    )
    predictions.loc[egb_tst_idx, 'target'] = test_egb
    r2_egb = float(r2_score(train.loc[egb_trn_idx, 'target'], oof_egb))
    oof_r2_list.append(r2_egb)
    oof_metrics['egb'] = {'r2': r2_egb}
    print(f"  EGB (with Chain Bandgap Coupling) OOF R²: {r2_egb:.4f}", flush=True)

    mean_oof = float(np.mean(oof_r2_list))
    print(f"\n==========================================", flush=True)
    print(f"Mean OOF R² (Physics Multi-Task Coupled): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-060',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-060 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
