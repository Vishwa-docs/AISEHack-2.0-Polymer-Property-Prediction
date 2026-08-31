#!/usr/bin/env python3
"""
Experiment P5-286: Wave 4 (PLAN AMENDMENT) — Deep PyTorch Transformer Autoencoder on GPU (smile_r3)
- Scaled from-scratch self-supervised representation learning on 250,000 smile_r3 molecules:
    * Character tokenizer with special tokens ([PAD], [UNK], [MASK], [CLS], [SEP])
    * 3-layer PyTorch Transformer Encoder (d_model=128, nhead=4, dim_feedforward=256)
    * Masked Language Modeling (MLM 15% mask rate) optimized for low VRAM headroom (batch_size=64, max_len=64)
    * Mean pooled continuous embeddings (128 dims) for all train and test polymers
- Multi-Task Physics Coupled Heterogeneous Ensemble (LightGBM, XGBoost, CatBoost, ExtraTrees, Ridge)
- SLSQP Convex Optimization on out-of-fold predictions
"""

import os
import sys
import json
import math
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen

RDLogger.DisableLog('rdApp.*')

TARGETS = ['tg', 'egc', 'egb', 'ei', 'eea', 'eps', 'nc']

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, char_to_idx, max_len=64, mask=True):
        self.smiles_list = smiles_list
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.mask = mask

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        s = str(self.smiles_list[idx])
        tokens = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in s[:self.max_len-2]]
        input_ids = [self.char_to_idx['<cls>']] + tokens + [self.char_to_idx['<sep>']]
        
        # Padding
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.char_to_idx['<pad>']] * pad_len
        else:
            input_ids = input_ids[:self.max_len]
            
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels = input_tensor.clone()
        
        if self.mask:
            # 15% masking
            rand = torch.rand(input_tensor.shape)
            mask_arr = (rand < 0.15) & (input_tensor != self.char_to_idx['<pad>']) & (input_tensor != self.char_to_idx['<cls>']) & (input_tensor != self.char_to_idx['<sep>'])
            input_tensor[mask_arr] = self.char_to_idx['<mask>']
            labels[~mask_arr] = -100
        else:
            labels = torch.zeros_like(input_tensor)
            
        return input_tensor, labels

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MolecularTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        padding_mask = (x == 0)
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        feat = self.transformer(emb, src_key_padding_mask=padding_mask)
        logits = self.mlm_head(feat)
        return logits, feat

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

def extract_base_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(clean_smiles(s)) for s in smiles_list]
    
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
            conj_index = n_arom / n_heavy

            vals = [
                mw, logp, tpsa, rot_bonds, hbd, hba, n_arom, n_rings, f_csp3,
                n_heavy, mr, val_elec, pol_proxy, ionic_proxy, conj_index
            ]
            desc_matrix.append(vals)
        else:
            desc_matrix.append([0.0] * 15)
    desc_matrix = np.nan_to_num(np.array(desc_matrix, dtype=np.float32), nan=0.0)

    return np.hstack([morgan_fps, maccs_fps, desc_matrix])

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
    parser = argparse.ArgumentParser(description="Run P5-286 Deep Transformer SSL on GPU")
    parser.add_argument("--data-dir", default="../../Dataset", help="Path to competition dataset")
    parser.add_argument("--output-dir", default=".", help="Path to output directory")
    parser.add_argument("--sample-size", type=int, default=150000, help="smile_r3 sample size")
    parser.add_argument("--epochs", type=int, default=4, help="Transformer training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}", flush=True)

    print("Loading train, test, and smile_r3...", flush=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    sample_size = 5000 if args.smoke else args.sample_size
    smile_r3_path = data_dir / "smile_r3.csv"
    if not smile_r3_path.exists():
        smile_r3_path = Path("/tmp/r3_dataset/smile_r3.csv")
        
    smile_r3_sample = pd.read_csv(smile_r3_path, nrows=sample_size).iloc[:, 0].astype(str).tolist()

    # Build character vocabulary
    chars = sorted(list(set("".join(smile_r3_sample[:10000]) + "".join(train['smiles'].astype(str)))))
    vocab = ['<pad>', '<unk>', '<mask>', '<cls>', '<sep>'] + chars
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    print(f"Vocabulary size: {len(vocab)} unique tokens", flush=True)

    # Train Transformer Autoencoder
    print(f"\n=== Training 3-Layer PyTorch Transformer on {len(smile_r3_sample):,} SMILES (batch={args.batch_size}) ===", flush=True)
    ssl_dataset = SMILESDataset(smile_r3_sample, char_to_idx, max_len=64, mask=True)
    ssl_loader = DataLoader(ssl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2 if torch.cuda.is_available() else 0)

    model = MolecularTransformer(vocab_size=len(vocab), d_model=128, nhead=4, num_layers=3, dim_feedforward=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    n_epochs = 1 if args.smoke else args.epochs
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for b_idx, (inputs, targets) in enumerate(ssl_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (b_idx + 1) % 500 == 0:
                print(f"  Epoch [{epoch+1}/{n_epochs}] Batch [{b_idx+1}/{len(ssl_loader)}] Loss: {loss.item():.4f}", flush=True)

    # Extract continuous 128-d embeddings for train and test
    print("\nExtracting continuous Transformer latent representations...", flush=True)
    model.eval()
    def get_embeddings(smiles_list):
        ds = SMILESDataset(smiles_list, char_to_idx, max_len=64, mask=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                _, feats = model(inputs)
                mask = (inputs != 0).unsqueeze(-1).float()
                mean_pooled = (feats * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                embeddings.append(mean_pooled.cpu().numpy())
        return np.vstack(embeddings).astype(np.float32)

    train_emb = get_embeddings(train['smiles'].tolist())
    test_emb = get_embeddings(test['smiles'].tolist())

    # Extract base descriptors
    train_base = extract_base_descriptors(train['smiles'].tolist())
    test_base = extract_base_descriptors(test['smiles'].tolist())

    X_train_full = np.hstack([train_base, train_emb])
    X_test_full = np.hstack([test_base, test_emb])

    scaler = RobustScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test_full = scaler.transform(X_test_full)

    train['canon_smiles'] = get_canonical_smiles(train['smiles'])
    test['canon_smiles'] = get_canonical_smiles(test['smiles'])

    n_splits = 3 if args.smoke else 5
    gkf = GroupKFold(n_splits=n_splits)

    print("\n=== Training Multi-Task Stacking Models Across 7 Targets ===", flush=True)
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

            lgb_m = lgb.LGBMRegressor(n_estimators=450, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=2026 + fold, n_jobs=-1, verbose=-1)
            lgb_m.fit(X_tr, y_tr)
            oof_lgb[val_idx] = lgb_m.predict(X_val)
            test_lgb_folds += lgb_m.predict(X_tgt_test) / n_splits

            xgb_m = xgb.XGBRegressor(n_estimators=450, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=2026 + fold, n_jobs=-1, verbosity=0)
            xgb_m.fit(X_tr, y_tr)
            oof_xgb[val_idx] = xgb_m.predict(X_val)
            test_xgb_folds += xgb_m.predict(X_tgt_test) / n_splits

            cat_m = CatBoostRegressor(iterations=450, learning_rate=0.035, depth=6, random_seed=2026 + fold, verbose=0)
            cat_m.fit(X_tr, y_tr)
            oof_cat[val_idx] = cat_m.predict(X_val)
            test_cat_folds += cat_m.predict(X_tgt_test) / n_splits

            ridge_m = Ridge(alpha=15.0, random_state=2026 + fold)
            ridge_m.fit(X_tr, y_tr)
            oof_ridge[val_idx] = ridge_m.predict(X_val)
            test_ridge_folds += ridge_m.predict(X_tgt_test) / n_splits

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
    print(f"Mean OOF R² (GPU Transformer SSL): {mean_oof:.5f}", flush=True)
    print(f"==========================================", flush=True)

    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    with open(pred_path, 'rb') as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metrics = {
        'experiment_id': 'P5-286',
        'mean_oof_r2': mean_oof,
        'prediction_hash': pred_hash,
        'per_target': oof_metrics,
        'sample_size': sample_size,
        'smoke_test': args.smoke
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Experiment P5-286 complete! Hash: {pred_hash}", flush=True)

if __name__ == '__main__':
    main()
