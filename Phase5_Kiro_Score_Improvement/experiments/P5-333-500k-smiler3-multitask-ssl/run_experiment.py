#!/usr/bin/env python3
"""
Experiment P5-333: Hierarchical Multi-Task Self-Supervised Transformer + Graph Invariants on 500k SMILES
- Pre-trains a Multi-Task Transformer (MLM + Topological Invariant Estimation) on 500,000 polymers from smile_r3.csv + PI1M.csv.
- Extracts 256-dimensional contextual representations for train and test polymers.
- Couples with radical geometry, physical conservation equations, and deep 5-fold GBDT zoos.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

SPECIAL_TOKENS = ['<pad>', '<unk>', '<cls>', '<mask>']

class PolymerTokenizer:
    def __init__(self, max_len=128):
        self.max_len = max_len
        self.char2idx = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2char = {idx: tok for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.pad_idx = self.char2idx['<pad>']
        self.unk_idx = self.char2idx['<unk>']
        self.cls_idx = self.char2idx['<cls>']
        self.mask_idx = self.char2idx['<mask>']

    def build_vocab(self, smiles_list):
        unique_chars = sorted(list(set(''.join(smiles_list))))
        for c in unique_chars:
            if c not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[c] = idx
                self.idx2char[idx] = c

    def encode(self, s, mask=False, mask_prob=0.15):
        chars = list(str(s)[:self.max_len - 1])
        tokens = [self.cls_idx]
        labels = [-100]

        for c in chars:
            token_id = self.char2idx.get(c, self.unk_idx)
            if mask and np.random.rand() < mask_prob:
                prob = np.random.rand()
                if prob < 0.8:
                    tokens.append(self.mask_idx)
                elif prob < 0.9:
                    tokens.append(np.random.randint(4, len(self.char2idx)))
                else:
                    tokens.append(token_id)
                labels.append(token_id)
            else:
                tokens.append(token_id)
                labels.append(-100)

        if len(tokens) < self.max_len:
            pad_len = self.max_len - len(tokens)
            tokens += [self.pad_idx] * pad_len
            labels += [-100] * pad_len

        return tokens[:self.max_len], labels[:self.max_len]

class MultiTaskSmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, is_training=True):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.is_training = is_training

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        s = self.smiles_list[idx]
        tokens, labels = self.tokenizer.encode(s, mask=self.is_training)
        
        # Simple length and star count proxy for multi-task auxiliary head
        s_str = str(s)
        n_stars = float(s_str.count('*'))
        s_len = float(len(s_str)) / 100.0
        aux_target = torch.tensor([n_stars, s_len], dtype=torch.float32)
        
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            aux_target
        )

class MultiTaskSmilesTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 1. MLM Head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # 2. Auxiliary Invariant Geometry Head
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, padding_mask=None):
        out = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        features = self.transformer_encoder(out, src_key_padding_mask=padding_mask)
        mlm_logits = self.mlm_head(features)
        
        # Pooled global representation from CLS token and mean pooling
        cls_feat = features[:, 0, :]
        mean_feat = features.mean(dim=1)
        global_repr = torch.cat([cls_feat, mean_feat], dim=-1) # 256-dim
        
        aux_pred = self.aux_head(cls_feat)
        return mlm_logits, aux_pred, global_repr

def extract_transformer_embeddings(smiles_list, model, tokenizer, device, batch_size=256):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            encoded = [tokenizer.encode(s, mask=False)[0] for s in batch_smiles]
            input_ids = torch.tensor(encoded, dtype=torch.long).to(device)
            mask = (input_ids == tokenizer.pad_idx)
            _, _, global_repr = model(input_ids, padding_mask=mask)
            embeddings.append(global_repr.cpu().numpy())
    return np.vstack(embeddings).astype(np.float32)

def extract_features(smiles_list, pi1m_sample=None, smile_r3_sample=None, is_train=True, tfidf_vec=None, svd_model=None):
    print(f"Extracting multi-modal features for {len(smiles_list)} polymers...", flush=True)
    mols = [Chem.MolFromSmiles(clean_smiles(s)) for s in smiles_list]
    
    # 1. Radical Marker Geometry & Torsional Energy Potential
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
            rot = Descriptors.NumRotatableBonds(m_raw)
            torsion_energy_proxy = rot * 3.5 / max(backbone_len, 1.0)
            polymer_geom.append([float(n_stars), backbone_len, mw / max(backbone_len, 1.0), torsion_energy_proxy])
        else:
            polymer_geom.append([float(n_stars), float(len(s_raw))/3.0, 50.0, 1.0])
    polymer_geom = np.nan_to_num(np.array(polymer_geom, dtype=np.float32), nan=0.0)

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

    # 4. Descriptors (14)
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

            pol_proxy = mr / (mw + 1e-5)
            ionic_proxy = tpsa / (mw + 1e-5)

            desc_matrix.append([
                mw, logp, tpsa, rot_bonds, hbd, hba, n_arom, n_rings, f_csp3,
                n_heavy, mr, val_elec, pol_proxy, ionic_proxy
            ])
        else:
            desc_matrix.append([0.0] * 14)
    desc_matrix = np.nan_to_num(np.array(desc_matrix, dtype=np.float32), nan=0.0)

    # 5. Continuous SVD (256)
    clean_str = [str(s) for s in smiles_list]
    if is_train and (pi1m_sample is not None or smile_r3_sample is not None):
        combined_corpus = []
        if pi1m_sample is not None:
            combined_corpus.extend(pi1m_sample)
        if smile_r3_sample is not None:
            combined_corpus.extend(smile_r3_sample)
            
        tfidf_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), max_features=18000, min_df=3)
        tfidf_vec.fit(combined_corpus)
        svd_model = TruncatedSVD(n_components=256, random_state=2026)
        svd_model.fit(tfidf_vec.transform(combined_corpus))
    
    if tfidf_vec is not None and svd_model is not None:
        svd_feats = svd_model.transform(tfidf_vec.transform(clean_str)).astype(np.float32)
    else:
        svd_feats = np.zeros((len(smiles_list), 256), dtype=np.float32)

    all_feats = np.hstack([polymer_geom, morgan_fps, maccs_fps, desc_matrix, svd_feats])
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
    parser = argparse.ArgumentParser(description="Run P5-333 Multi-Task Transformer on 500k smile_r3.csv")
    parser.add_argument("--data-dir", default="../../Dataset", help="Path to competition dataset")
    parser.add_argument("--output-dir", default=".", help="Path to output directory")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing P5-333 on device: {device}", flush=True)

    print("Loading datasets...", flush=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    n_sample = 25000 if args.smoke else 500000
    print(f"Loading {n_sample} polymers from smile_r3.csv and PI1M.csv for Multi-Task SSL...", flush=True)
    pi1m_sample = pd.read_csv(data_dir / "PI1M.csv", nrows=n_sample // 2).iloc[:, 0].astype(str).tolist()
    smile_r3_sample = pd.read_csv(data_dir / "smile_r3.csv", nrows=n_sample // 2).iloc[:, 0].astype(str).tolist()

    all_ssl_smiles = pi1m_sample + smile_r3_sample + train['smiles'].tolist() + test['smiles'].tolist()
    
    # 1. Build Tokenizer and Multi-Task Transformer
    print("Building vocabulary and training Multi-Task Transformer from scratch...", flush=True)
    tokenizer = PolymerTokenizer(max_len=128)
    tokenizer.build_vocab(all_ssl_smiles)
    vocab_size = len(tokenizer.char2idx)
    print(f"Vocabulary size: {vocab_size} tokens", flush=True)

    dataset = MultiTaskSmilesDataset(all_ssl_smiles, tokenizer, is_training=True)
    batch_size = 128 if device.type == 'cuda' else 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MultiTaskSmilesTransformer(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_aux = nn.MSELoss()

    epochs = 2 if args.smoke else 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        max_steps = 150 if args.smoke else 1500
        for input_ids, labels, aux_target in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            aux_target = aux_target.to(device)
            
            mask = (input_ids == tokenizer.pad_idx)
            optimizer.zero_grad()
            mlm_logits, aux_pred, _ = model(input_ids, padding_mask=mask)
            
            loss_mlm = criterion_mlm(mlm_logits.view(-1, vocab_size), labels.view(-1))
            loss_aux = criterion_aux(aux_pred, aux_target)
            loss = loss_mlm + 0.1 * loss_aux
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps >= max_steps:
                break
        print(f"  [Epoch {epoch+1}/{epochs}] Joint SSL Loss: {total_loss / steps:.4f}", flush=True)

    print("Extracting Transformer Contextual Embeddings for competition sets...", flush=True)
    X_train_ssl = extract_transformer_embeddings(train['smiles'].tolist(), model, tokenizer, device)
    X_test_ssl = extract_transformer_embeddings(test['smiles'].tolist(), model, tokenizer, device)

    # 2. Extract multi-modal baseline features
    X_train_base, tfidf_v, svd_m = extract_features(train['smiles'].tolist(), pi1m_sample=pi1m_sample, smile_r3_sample=smile_r3_sample, is_train=True)
    X_test_base, _, _ = extract_features(test['smiles'].tolist(), is_train=False, tfidf_vec=tfidf_v, svd_model=svd_m)

    X_train_combined = np.hstack([X_train_base, X_train_ssl])
    X_test_combined = np.hstack([X_test_base, X_test_ssl])

    scaler = RobustScaler()
    X_train_combined = scaler.fit_transform(X_train_combined)
    X_test_combined = scaler.transform(X_test_combined)

    train['canon_smiles'] = get_canonical_smiles(train['smiles'])
    test['canon_smiles'] = get_canonical_smiles(test['smiles'])

    n_splits = 3 if args.smoke else 5
    gkf = GroupKFold(n_splits=n_splits)

    print("\n=== Stage 1: Cross-Property Joint Imputation Field ===", flush=True)
    train_prop_matrix = np.zeros((len(train), len(TARGETS)), dtype=np.float32)
    test_prop_matrix = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

    for idx, target in enumerate(TARGETS):
        tgt_train_idx = train[train['target_type'] == target].index.to_numpy()
        X_tgt = X_train_combined[tgt_train_idx]
        y_tgt = train.loc[tgt_train_idx, 'target'].to_numpy(dtype=np.float32)
        groups_tgt = train.loc[tgt_train_idx, 'canon_smiles'].to_numpy()

        oof_pred = np.zeros(len(y_tgt))
        test_pred_folds = np.zeros(len(test))
        all_train_folds = np.zeros(len(train))

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tgt, y_tgt, groups=groups_tgt)):
            m_ridge = Ridge(alpha=15.0, random_state=2026 + fold)
            m_ridge.fit(X_tgt[trn_idx], y_tgt[trn_idx])
            oof_pred[val_idx] = m_ridge.predict(X_tgt[val_idx])
            test_pred_folds += m_ridge.predict(X_test_combined) / n_splits
            all_train_folds += m_ridge.predict(X_train_combined) / n_splits

        train_prop_matrix[:, idx] = all_train_folds
        train_prop_matrix[tgt_train_idx, idx] = oof_pred
        test_prop_matrix[:, idx] = test_pred_folds

    # Physical conservation channels
    ei_cons_tr = train_prop_matrix[:, 1] + train_prop_matrix[:, 4]
    egb_cons_tr = train_prop_matrix[:, 1] - 0.25
    eps_cons_tr = train_prop_matrix[:, 6]**2 + 0.5

    ei_cons_te = test_prop_matrix[:, 1] + test_prop_matrix[:, 4]
    egb_cons_te = test_prop_matrix[:, 1] - 0.25
    eps_cons_te = test_prop_matrix[:, 6]**2 + 0.5

    physics_tr = np.column_stack([train_prop_matrix, ei_cons_tr, egb_cons_tr, eps_cons_tr])
    physics_te = np.column_stack([test_prop_matrix, ei_cons_te, egb_cons_te, eps_cons_te])

    X_train_full = np.hstack([X_train_combined, physics_tr])
    X_test_full = np.hstack([X_test_combined, physics_te])

    print("\n=== Stage 2: Training GBDTs with 500k Multi-Task SSL Representations ===", flush=True)
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
            lgb_m = lgb.LGBMRegressor(n_estimators=550, learning_rate=0.025, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=2026 + fold, n_jobs=-1, verbose=-1)
            lgb_m.fit(X_tr, y_tr)
            oof_lgb[val_idx] = lgb_m.predict(X_val)
            test_lgb_folds += lgb_m.predict(X_tgt_test) / n_splits

            # 2. XGBoost
            xgb_m = xgb.XGBRegressor(n_estimators=550, learning_rate=0.025, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=2026 + fold, n_jobs=-1, verbosity=0)
            xgb_m.fit(X_tr, y_tr)
            oof_xgb[val_idx] = xgb_m.predict(X_val)
            test_xgb_folds += xgb_m.predict(X_tgt_test) / n_splits

            # 3. CatBoost
            cat_m = CatBoostRegressor(iterations=550, learning_rate=0.03, depth=6, random_seed=2026 + fold, verbose=0)
            cat_m.fit(X_tr, y_tr)
            oof_cat[val_idx] = cat_m.predict(X_val)
            test_cat_folds += cat_m.predict(X_tgt_test) / n_splits

            # 4. Ridge
            ridge_m = Ridge(alpha=15.0, random_state=2026 + fold)
            ridge_m.fit(X_tr, y_tr)
            oof_ridge[val_idx] = ridge_m.predict(X_val)
            test_ridge_folds += ridge_m.predict(X_tgt_test) / n_splits

            # 5. ExtraTrees
            et_m = ExtraTreesRegressor(n_estimators=250, max_depth=15, min_samples_split=4, random_state=2026 + fold, n_jobs=-1)
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
    print(f"Mean OOF R² (Multi-Task 500k SSL): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-333',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-333 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
