"""C169: from-scratch PI1M SMILES masked-token encoder for Ei.

This is a materially new representation family relative to the earlier
hashed-character PI1M probes.  A small Transformer encoder is initialized from
random weights and pretrained only on a deterministic 100k subset of the
official unlabeled PI1M.csv plus unlabeled official train/test structures.
The frozen [CLS] embeddings are then evaluated as Ei residual features under
clean five-fold OOF.  No external rows, pretrained assets, test external_labels, or
local_eval values are read.  The script stops before a full-data fit unless the
predeclared clean gate passes.
"""
from pathlib import Path
import importlib.util
import json
import pickle
import re
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
SEED = 20260804
MAX_LEN = 128
CORPUS_ROWS = 100_000
EPOCHS = 3
BATCH = 512
HIDDEN = 96
LAYERS = 2
HEADS = 4
started = time.time()


def tokenize(text):
    # Chemistry-aware SMILES tokens, including bracket atoms and two-character
    # elements. This is not an external tokenizer or pretrained vocabulary.
    pattern = re.compile(r"Cl|Br|Si|Na|Li|Al|Ca|Fe|Cu|Zn|Mg|%\d{2}|\[[^\]]+\]|[A-Z][a-z]?|[bcnops]|\*|[0-9]|[()=#[\].+\-/:\\@]")
    return pattern.findall(str(text))


def normalize_tokens(text, vocab, max_len=MAX_LEN):
    toks = tokenize(text)
    ids = [vocab.get("[CLS]", 1)] + [vocab.get(tok, vocab["[UNK]"]) for tok in toks[: max_len - 2]] + [vocab["[SEP]"]]
    ids += [vocab["[PAD]"]] * (max_len - len(ids))
    return np.asarray(ids[:max_len], dtype=np.int64)


class SmilesDataset(Dataset):
    def __init__(self, rows, vocab):
        self.ids = np.stack([normalize_tokens(x, vocab) for x in rows])
        self.pad = vocab["[PAD]"]
        self.mask = vocab["[MASK]"]
        self.vocab_size = len(vocab)
        self.cls = vocab["[CLS]"]
        self.sep = vocab["[SEP]"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        original = torch.from_numpy(self.ids[index].copy())
        masked = original.clone()
        labels = torch.full_like(original, -100)
        eligible = (original != self.pad) & (original != self.cls) & (original != self.sep)
        choose = (torch.rand(original.shape) < 0.15) & eligible
        labels[choose] = original[choose]
        # BERT-style replacement: mask most selected tokens and retain a small
        # random/original fraction without adding any learned artifact.
        mask_choose = choose & (torch.rand(original.shape) < 0.80)
        random_choose = choose & ~mask_choose & (torch.rand(original.shape) < 0.50)
        masked[mask_choose] = self.mask
        if torch.any(random_choose):
            masked[random_choose] = torch.randint(5, self.vocab_size, (int(random_choose.sum()),), dtype=torch.long)
        return masked, labels, (original != self.pad)


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token = nn.Embedding(vocab_size, HIDDEN, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, HIDDEN)
        layer = nn.TransformerEncoderLayer(d_model=HIDDEN, nhead=HEADS, dim_feedforward=HIDDEN * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.norm = nn.LayerNorm(HIDDEN)
        self.lm = nn.Linear(HIDDEN, vocab_size)

    def forward(self, ids, pad_mask=None):
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        h = self.token(ids) + self.position(pos)
        h = self.encoder(h, src_key_padding_mask=~pad_mask if pad_mask is not None else None)
        return self.norm(h)


def r2(y, p):
    return float(r2_score(np.asarray(y, float), np.asarray(p, float)))


def structure_features():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    P = pickle.loads((SCR / "physics.pkl").read_bytes())
    G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    spec = importlib.util.spec_from_file_location("c162_feature_builder", ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    return F, mod.features(F, P, G).astype(np.float64)


def build_labels(F):
    cmap, idx = F["canon_map"], F["idx"]
    ns = len(F["canon_list"])
    labels = np.full((ns, 7), np.nan, dtype=np.float64)
    for name in ("archive/train.csv", "train.csv"):
        frame = pd.read_csv(BASE / name); frame["canon"] = frame["smiles"].map(cmap)
        for j, target in enumerate(["tg", "egc", "egb", "ei", "eea", "nc", "eps"]):
            vals = frame.loc[frame["target_type"].eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items(): labels[idx[canon], j] = float(value)
    return labels


def train_encoder(corpus, vocab):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmilesEncoder(len(vocab)).to(device)
    data = SmilesDataset(corpus, vocab)
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.train(); history = []
    for epoch in range(EPOCHS):
        total = 0.0; count = 0
        for ids, target, valid in loader:
            ids, target, valid = ids.to(device, non_blocking=True), target.to(device, non_blocking=True), valid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            h = model(ids, valid)
            loss = loss_fn(model.lm(h).reshape(-1, len(vocab)), target.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(ids); count += len(ids)
        history.append(float(total / max(1, count)))
        print(json.dumps({"phase": "mlm", "epoch": epoch + 1, "epochs": EPOCHS, "loss": history[-1], "device": str(device)}), flush=True)
    return model, device, history


def embed(model, device, rows, vocab):
    ids = np.stack([normalize_tokens(x, vocab) for x in rows])
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), BATCH):
            batch = torch.from_numpy(ids[start:start + BATCH]).to(device)
            valid = batch != vocab["[PAD]"]
            h = model(batch, valid)
            out.append(h[:, 0].detach().cpu().numpy())
    return np.vstack(out).astype(np.float64)


def main():
    F, X = structure_features()
    labels = build_labels(F)
    corpus_frame = pd.read_csv(BASE / "PI1M.csv", usecols=["SMILES"])
    corpus = corpus_frame["SMILES"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    if len(corpus) > CORPUS_ROWS:
        corpus = corpus[rng.choice(len(corpus), size=CORPUS_ROWS, replace=False)]
    official = np.asarray(F["canon_list"], dtype=object)
    corpus = np.concatenate([corpus, official])
    counts = {}
    for text in corpus:
        for tok in set(tokenize(text)):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for tok in sorted(counts):
        if tok not in vocab: vocab[tok] = len(vocab)
    print(json.dumps({"phase": "setup", "corpus_rows": int(len(corpus)), "vocab_size": len(vocab), "max_len": MAX_LEN}), flush=True)
    model, device, history = train_encoder(corpus, vocab)
    embeddings = embed(model, device, official, vocab)
    ei_rows = np.where(np.isfinite(labels[:, 3]))[0]
    y = labels[ei_rows, 3]
    parent_all = np.load(SCR / "out_clean_corrected/PFINAL.npy").astype(np.float64)
    parent = parent_all[ei_rows, 3].copy()
    both = np.isfinite(labels[ei_rows, 1]) & np.isfinite(labels[ei_rows, 4])
    parent[both] = 0.5 * parent[both] + 0.5 * (labels[ei_rows[both], 1] + labels[ei_rows[both], 4])
    # Frozen heads: embedding-only and structure+embedding residual Ridge,
    # plus a small fixed parent blend for each. Selection is clean OOF only.
    reps = {
        "embedding": embeddings[ei_rows],
        "structure_embedding": np.hstack([X[ei_rows], embeddings[ei_rows]]),
    }
    candidates = {}
    fold_reports = {}
    splits = KFold(5, shuffle=True, random_state=SEED + 169).split(ei_rows)
    for name, rep in reps.items():
        for weight in (0.25, 0.50):
            key = f"{name}_residual_weight_{weight:.2f}"
            pred = parent.copy(); rows = []
            for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + 169).split(rep), 1):
                fit = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
                fit.fit(rep[tr], y[tr] - parent[tr])
                raw = parent[va] + weight * fit.predict(rep[va])
                pred[va] = raw
                rows.append({"fold": fold, "parent_r2": r2(y[va], parent[va]), "candidate_r2": r2(y[va], raw), "delta_r2": r2(y[va], raw) - r2(y[va], parent[va])})
            candidates[key] = pred
            fold_reports[key] = rows
    metrics = {}
    for key, pred in candidates.items():
        metrics[key] = {"parent_r2": r2(y, parent), "candidate_r2": r2(y, pred), "delta_r2": r2(y, pred) - r2(y, parent), "positive_folds": int(sum(x["delta_r2"] > 0 for x in fold_reports[key])), "folds": fold_reports[key]}
    best = max(metrics, key=lambda key: metrics[key]["candidate_r2"])
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C169-pi1m-smiles-transformer-ei",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "mechanism": "random-initialized SMILES-token Transformer MLM on official PI1M plus unlabeled official structures; frozen [CLS] Ei residual Ridge heads",
        "corpus": {"pi1m_rows_used": int(min(len(corpus_frame), CORPUS_ROWS)), "official_structure_rows": int(len(official)), "total_rows": int(len(corpus)), "max_len": MAX_LEN, "vocab_size": int(len(vocab)), "epochs": EPOCHS, "hidden": HIDDEN, "layers": LAYERS, "heads": HEADS, "device": str(device), "loss_history": history},
        "rows": {"ei": int(len(ei_rows)), "both_eea_egc": int(np.sum(both))},
        "metrics": metrics,
        "best_arm_by_clean_oof": best,
        "gate": {"ei_gain_at_least_0.005": bool(metrics[best]["delta_r2"] >= 0.005), "positive_folds_at_least_4": bool(metrics[best]["positive_folds"] >= 4), "passed_screen": bool(metrics[best]["delta_r2"] >= 0.005 and metrics[best]["positive_folds"] >= 4)},
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "R2-C169-pi1m-smiles-transformer-ei-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["gate"]["passed_screen"]:
        print("C169 STOP: PI1M Transformer Ei screen failed; no full-data fit, local_eval read, or submission artifact.", flush=True)


if __name__ == "__main__":
    main()
