"""Torch models trained from scratch on official data only.

- SkMLP: sklearn-compatible MLP regressor wrapper (fit/predict).
- MLPMulti: multi-task MLP with target-masked loss + optional physics penalties.
- TinyMLM: small transformer encoder + masked-token pretraining over either
  character-level or chemically-aware atom-level tokens.
- PolymerGNN: hand-rolled message-passing GNN (GCN/GAT/MPNN styles).

No pretrained weights, no external vocabularies. Fixed seeds.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn as nn

SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp_block(dims, dropout):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SkMLP:
    """sklearn-style wrapper around MLPMulti for single-target regression."""

    def __init__(self, params: dict, seed: int = SEED):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        hidden = self.params.get("hidden", (256, 128))
        dims = [X.shape[1], *hidden]
        self.net = MLPMulti(dims=dims, n_out=1, dropout=self.params.get("dropout", 0.2))
        self.net.to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.params.get("lr", 1e-3), weight_decay=1e-5)
        ds = torch.tensor(X)
        ys = torch.tensor(y).unsqueeze(1)
        n = len(ds)
        bs = min(self.params.get("batch_size", 128), n)
        epochs = self.params.get("epochs", 120)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(ds[idx].to(DEVICE))[:, 0], ys[idx].to(DEVICE)[:, 0])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 4096):
                out.append(self.net(torch.tensor(X[s:s + 4096]).to(DEVICE))[:, 0].cpu().numpy())
        return np.concatenate(out)


class MLPMulti(nn.Module):
    """Shared encoder + per-target heads; availability-masked loss + optional physics penalty."""

    def __init__(self, dims: list, n_out: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dims[0], dims[-1]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dims[-1], dims[-1]), nn.GELU(),
        )
        self.heads = nn.ModuleList([_mlp_block([dims[-1], 64, 1], dropout) for _ in range(n_out)])

    def forward(self, x):
        h = self.encoder(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


def fit_multitask_mlp(X, Y, mask, *, hidden=256, depth=3, dropout=0.2, lr=1e-3,
                      epochs=200, batch_size=256, seed=SEED, device=None,
                      physics_penalty=None, physics_lambda=0.05):
    """Y: (n, n_targets) with NaN for missing; mask: (n, n_targets) bool.
    physics_penalty(preds, feats) -> scalar tensor or None."""
    device = device or DEVICE
    torch.manual_seed(seed)
    X_t = torch.tensor(np.asarray(X, dtype=np.float32))
    dims = [X.shape[1]] + [hidden] * max(depth - 1, 1) + [hidden]
    net = MLPMulti(dims=dims, n_out=Y.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(X_t)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        net.train()
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = X_t[idx].to(device)
            yb = torch.tensor(np.nan_to_num(Y[idx], nan=0.0), dtype=torch.float32).to(device)
            mb = torch.tensor(mask[idx], dtype=torch.float32).to(device)
            opt.zero_grad()
            preds = net(xb)
            mb_sum = mb.sum().clamp(min=1)
            loss = ((preds - yb) ** 2 * mb).sum() / mb_sum
            if physics_penalty is not None:
                loss = loss + physics_lambda * physics_penalty(preds, xb)
            loss.backward()
            opt.step()
    net.eval()
    return net


def predict_multitask(net, X, batch=4096, device=None):
    device = device or DEVICE
    X_t = torch.tensor(np.asarray(X, dtype=np.float32))
    outs = []
    with torch.no_grad():
        for s in range(0, len(X_t), batch):
            outs.append(net(X_t[s:s + batch].to(device)).cpu().numpy())
    return np.concatenate(outs)


# ---------------------------------------------------------------------------
# Graph neural network (hand-rolled message passing, no PyG dependency)
# ---------------------------------------------------------------------------

class MPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, style="gcn", heads=4, dropout=0.2):
        super().__init__()
        self.style = style
        self.heads = heads
        self.lin = nn.Linear(in_dim, out_dim)
        if style in ("gat", "mpnn"):
            self.attn_a = nn.Linear(out_dim, heads, bias=False)
            self.attn_b = nn.Linear(out_dim, heads, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index):
        src, dst = edge_index[0], edge_index[1]
        m = self.lin(h)
        if self.style in ("gat", "mpnn"):
            e = (self.attn_a(m[src]) + self.attn_b(m[dst])).leaky_relu(0.2)
            out_dim = m.shape[1] // self.heads
            m = m.view(-1, self.heads, out_dim)
            alpha = torch.softmax(e, dim=0).unsqueeze(-1)
            agg = torch.zeros_like(m).index_add_(0, dst, alpha * m[src])
            agg = agg.reshape(-1, self.heads * out_dim)
        else:
            agg = torch.zeros_like(m).index_add_(0, dst, m[src])
            deg = torch.zeros(m.shape[0], device=m.device).index_add_(
                0, dst, torch.ones(len(dst), device=m.device)
            )
            agg = agg / deg.clamp(min=1).unsqueeze(1)
        h = self.norm(h + self.dropout(agg))
        return torch.nn.functional.gelu(h)


class PolymerGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, style="gcn", n_out=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [MPLayer(hidden, hidden, style=style, dropout=dropout) for _ in range(layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, n_out))

    def forward(self, x, edge_index):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h, self.head(h)


class GNNRegressor:
    """sklearn-style wrapper: fit(graphs, y); graphs = list of (feat_matrix, edge_index)."""

    def __init__(self, params: dict, seed: int = SEED):
        self.params = params
        self.seed = seed

    @staticmethod
    def _batch(graphs, device):
        feats, edges = [], []
        off = 0
        for x, ei in graphs:
            feats.append(torch.tensor(np.asarray(x), dtype=torch.float32))
            edges.append(torch.tensor(np.asarray(ei), dtype=torch.long) + off)
            off += x.shape[0]
        X = torch.cat(feats).to(device)
        E = torch.cat(edges, dim=1).to(device)
        counts = np.array([g[0].shape[0] for g in graphs])
        return X, E, counts

    def fit(self, graphs, y):
        device = DEVICE
        torch.manual_seed(self.seed)
        in_dim = graphs[0][0].shape[1]
        self.net = PolymerGNN(
            in_dim,
            hidden=self.params.get("hidden", 128),
            layers=self.params.get("layers", 3),
            style=self.params.get("style", "gcn"),
            dropout=self.params.get("dropout", 0.2),
        ).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.params.get("lr", 1e-3), weight_decay=1e-5)
        X, E, counts = self._batch(graphs, device)
        owner = torch.tensor(np.repeat(np.arange(len(graphs)), counts), dtype=torch.long, device=device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
        n_graphs = len(graphs)
        for _ in range(self.params.get("epochs", 60)):
            self.net.train()
            opt.zero_grad()
            _, preds = self.net(X, E)
            deg = torch.zeros(n_graphs, device=device).index_add_(
                0, owner, torch.ones(X.shape[0], device=device)
            )
            sums = torch.zeros(n_graphs, device=device).index_add_(0, owner, preds[:, 0])
            loss = nn.functional.mse_loss(sums / deg.clamp(min=1), y_t)
            loss.backward()
            opt.step()
        return self

    def predict(self, graphs):
        device = DEVICE
        self.net.eval()
        X, E, counts = self._batch(graphs, device)
        owner = torch.tensor(np.repeat(np.arange(len(graphs)), counts), dtype=torch.long, device=device)
        with torch.no_grad():
            _, preds = self.net(X, E)
            deg = torch.zeros(len(graphs), device=device).index_add_(
                0, owner, torch.ones(X.shape[0], device=device)
            )
            sums = torch.zeros(len(graphs), device=device).index_add_(0, owner, preds[:, 0])
            return (sums / deg.clamp(min=1)).cpu().numpy()


# ---------------------------------------------------------------------------
# Masked-LM pretraining (char-level or atom-level tokens) -- from scratch
# ---------------------------------------------------------------------------

ATOM_TOKENS = list("CNOSPFIClBr[*]()=#@+-0123456789\\/%.")

# Chemically-aware atom tokenizer (ask sec 3.2): multi-char atoms (Cl, Br, Si...),
# bracketed groups ([*] endpoints, [nH], [O-]...), bonds, branches and ring-closure
# digits are each one indivisible token.  Char-level tokenization fragments
# functional groups (S, O, O instead of a sulfone motif) and is a known failure
# mode for sequence models -- this tokenizer fixes that.
ATOM_TOKEN_RE = re.compile(
    r"\[[^\]]+\]|Br|Cl|Si|Se|Na|Mg|Ca|Fe|Zn|Cu|Ni|Co|Sn|Pb|"
    r"Al|Li|Be|Ti|V|Cr|Mn|B|C|N|O|F|P|S|K|[A-Z][a-z]?|"
    r"[0-9]{1,2}|[*]|[=#@+\(\)\/\-\.%]"
)


class CharTokenizer:
    def __init__(self, vocab=None, max_len=256):
        self.vocab = vocab or ATOM_TOKENS
        self.stoi = {c: i + 3 for i, c in enumerate(self.vocab)}  # 0=pad 1=mask 2=unk
        self.max_len = max_len

    def encode(self, s):
        ids = [self.stoi.get(c, 2) for c in str(s)[: self.max_len]]
        if not ids:
            ids = [0]
        return ids


class AtomTokenizer:
    """Regex-based atom-level tokenizer (multi-char atoms and [*] are single tokens)."""

    def __init__(self, vocab=None, max_len=256):
        if vocab is None:
            # Fixed deterministic vocabulary covering organic chemistry tokens.
            vocab = ["[*]", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B",
                     "Se", "=", "#", "(", ")", "\\", "/", "@", "+", "-", "%",
                     "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]
        self.vocab = list(vocab)
        self.stoi = {t: i + 3 for i, t in enumerate(self.vocab)}  # 0=pad 1=mask 2=unk
        self.max_len = max_len

    def encode(self, s):
        tokens = ATOM_TOKEN_RE.findall(str(s))[: self.max_len]
        ids = [self.stoi.get(t, 2) for t in tokens]
        if not ids:
            ids = [0]
        return ids


class TinyMLM(nn.Module):
    def __init__(self, vocab_size, dim=64, layers=2, heads=4, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, dropout=0.1, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        h = self.tok(ids) + self.pos(ids[:, : ids.shape[1]])
        pad_mask = ids == 0
        return self.encoder(h, src_key_padding_mask=pad_mask)


def pretrain_mlm(smiles_list, *, dim=64, layers=2, heads=4, epochs=2, batch=256,
                 mask_rate=0.15, lr=3e-4, seed=SEED, device=None, max_len=256,
                 log_every=200, atom=False):
    """Pretrain a tiny MLM from scratch on the given SMILES. Returns (model, tok).

    atom=True uses the chemically-aware AtomTokenizer (multi-char atoms / [*] as
    single tokens) instead of character-level tokens.
    """
    device = device or DEVICE
    torch.manual_seed(seed)
    tok = AtomTokenizer(max_len=max_len) if atom else CharTokenizer(max_len=max_len)
    vocab_size = len(tok.vocab) + 3
    model = TinyMLM(vocab_size, dim=dim, layers=layers, heads=heads, max_len=max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    texts = [str(s) for s in smiles_list]
    order = rng.permutation(len(texts))
    step = 0
    model.train()
    for _ in range(epochs):
        for s in range(0, len(order), batch):
            chunk = [texts[i] for i in order[s:s + batch]]
            enc = [tok.encode(t) for t in chunk]
            maxlen = min(max_len, max(len(e) for e in enc))
            ids = np.zeros((len(enc), maxlen), dtype=np.int64)
            for i, e in enumerate(enc):
                ids[i, : len(e)] = e[:maxlen]
            labels = ids.copy()
            mask = (rng.random(ids.shape) < mask_rate) & (ids != 0)
            ids[mask] = 1
            ids_t = torch.tensor(ids, device=device)
            labels_t = torch.tensor(labels, device=device)
            mask_t = torch.tensor(mask, device=device)
            opt.zero_grad()
            h = model(ids_t)
            logits = model.lm_head(h[mask_t])
            loss = nn.functional.cross_entropy(logits, labels_t[mask_t])
            loss.backward()
            opt.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"    mlm step {step} loss {loss.item():.4f}", flush=True)
    model.eval()
    return model, tok


def mlm_embed(model, tok, texts, *, device=None, batch=256):
    """Mean-pooled token embeddings for a list of texts."""
    device = device or DEVICE
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = [tok.encode(str(t)) for t in texts[s:s + batch]]
            maxlen = min(tok.max_len, max(len(e) for e in enc))
            ids = np.zeros((len(enc), maxlen), dtype=np.int64)
            for i, e in enumerate(enc):
                ids[i, : len(e)] = e[:maxlen]
            h = model(torch.tensor(ids, device=device))
            mask = torch.tensor(ids != 0, device=device).unsqueeze(-1)
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(emb.cpu().numpy())
    return np.concatenate(out).astype(np.float32)

