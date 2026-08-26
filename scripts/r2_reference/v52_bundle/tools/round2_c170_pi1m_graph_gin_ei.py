"""C170: from-scratch PI1M graph message-passing encoder for Ei.

This is the final distinct Ei representation probe in the current branch. It
parses official SMILES into atom/bond graphs without RDKit, pretrains a small
GIN-style encoder from random weights by masked atom prediction on 100k
official PI1M molecules plus unlabeled official structures, and evaluates
frozen graph embeddings as Ei residual features. No labels are used during
pretraining and no test-external_label/local_eval file is read.
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
from torch.utils.data import Dataset, DataLoader
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
ROWS = 100_000
EPOCHS = 3
BATCH = 256
HIDDEN = 96
LAYERS = 3
started = time.time()
TOKEN_RE = re.compile(r"Cl|Br|Si|Na|Li|Al|Ca|Fe|Cu|Zn|Mg|%\d{2}|\[[^\]]+\]|[A-Z][a-z]?|[bcnops]|\*|[0-9]|[()=#[\].+\-/:\\@]")
ATOM_RE = re.compile(r"^(\[[^\]]+\]|[A-Z][a-z]?|[bcnops]|\*)$")


def toks(text):
    return TOKEN_RE.findall(str(text))


def parse_graph(text, vocab):
    tokens = toks(text)
    nodes, edges = [], []
    branch, rings = [], {}
    prev, bond = -1, 1
    bond_map = {"-": 1, "=": 2, "#": 3, ":": 4, "/": 1, "\\": 1}
    for token in tokens:
        if ATOM_RE.match(token):
            here = len(nodes); nodes.append(vocab.get(token, vocab["[UNK]"]))
            if prev >= 0:
                edges.extend([(prev, here, bond), (here, prev, bond)])
            prev, bond = here, 1
        elif token == "(":
            branch.append(prev)
        elif token == ")":
            if branch: prev = branch.pop()
        elif token in bond_map:
            bond = bond_map[token]
        elif token.isdigit() or token.startswith("%"):
            if prev < 0: continue
            if token in rings:
                other, stored = rings.pop(token)
                b = bond if bond != 1 else stored
                edges.extend([(other, prev, b), (prev, other, b)])
            else:
                rings[token] = (prev, bond)
            bond = 1
    if not nodes:
        nodes = [vocab["[UNK]"]]
    return np.asarray(nodes, np.int64), np.asarray(edges, np.int64).reshape(-1, 3) if edges else np.empty((0, 3), np.int64)


class Graphs(Dataset):
    def __init__(self, rows, vocab):
        self.graphs = [parse_graph(row, vocab) for row in rows]
        self.vocab = vocab

    def __len__(self): return len(self.graphs)
    def __getitem__(self, i): return self.graphs[i]


def collate(batch):
    all_nodes, src, dst, bond, graph_ids = [], [], [], [], []
    offset = 0
    for g, (nodes, edges) in enumerate(batch):
        all_nodes.append(torch.from_numpy(nodes))
        graph_ids.append(torch.full((len(nodes),), g, dtype=torch.long))
        if len(edges):
            src.append(torch.from_numpy(edges[:, 0] + offset))
            dst.append(torch.from_numpy(edges[:, 1] + offset))
            bond.append(torch.from_numpy(edges[:, 2]))
        offset += len(nodes)
    return torch.cat(all_nodes), (torch.cat(src) if src else torch.empty(0, dtype=torch.long)), (torch.cat(dst) if dst else torch.empty(0, dtype=torch.long)), (torch.cat(bond) if bond else torch.empty(0, dtype=torch.long)), torch.cat(graph_ids), len(batch)


class GIN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, HIDDEN)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 2), nn.GELU(), nn.Linear(HIDDEN * 2, HIDDEN), nn.LayerNorm(HIDDEN))
            for _ in range(LAYERS)
        ])
        self.bond = nn.Embedding(5, HIDDEN)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(self, node_ids, src, dst, bond, graph_ids, n_graphs):
        h = self.embedding(node_ids)
        for layer in self.layers:
            agg = torch.zeros_like(h)
            if len(src):
                msg = h[src] + self.bond((bond - 1).clamp(0, 4))
                agg.index_add_(0, dst, msg)
            h = h + layer(agg)
        pooled = torch.zeros((n_graphs, HIDDEN), device=h.device)
        pooled.index_add_(0, graph_ids, h)
        counts = torch.bincount(graph_ids, minlength=n_graphs).clamp_min(1).to(h.device).unsqueeze(1)
        return h, pooled / counts


def train_gin(graphs, vocab):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GIN(max(vocab.values()) + 1).to(device)
    loader = DataLoader(graphs, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    history = []
    for epoch in range(EPOCHS):
        model.train(); total = 0.; count = 0
        for node_ids, src, dst, bond, graph_ids, n_graphs in loader:
            node_ids, src, dst, bond, graph_ids = [x.to(device, non_blocking=True) for x in (node_ids, src, dst, bond, graph_ids)]
            target = node_ids.clone()
            choose = torch.rand(len(node_ids), device=device) < 0.15
            masked = node_ids.clone(); masked[choose] = 4
            opt.zero_grad(set_to_none=True)
            h, _ = model(masked, src, dst, bond, graph_ids, n_graphs)
            loss = loss_fn(model.head(h), torch.where(choose, target, torch.full_like(target, -100)))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * n_graphs; count += n_graphs
        history.append(float(total / max(1, count)))
        print(json.dumps({"phase": "graph_mlm", "epoch": epoch + 1, "epochs": EPOCHS, "loss": history[-1], "device": str(device)}), flush=True)
    return model, device, history


def embed(model, device, graphs, batch_size=512):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    values = []; model.eval()
    with torch.no_grad():
        for node_ids, src, dst, bond, graph_ids, n_graphs in loader:
            node_ids, src, dst, bond, graph_ids = [x.to(device) for x in (node_ids, src, dst, bond, graph_ids)]
            _, pooled = model(node_ids, src, dst, bond, graph_ids, n_graphs)
            values.append(pooled.cpu().numpy())
    return np.vstack(values).astype(np.float64)


def load_structure_features():
    F = pickle.loads((SCR / "features.pkl").read_bytes()); P = pickle.loads((SCR / "physics.pkl").read_bytes()); G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    spec = importlib.util.spec_from_file_location("c162_feature_builder", ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    return F, mod.features(F, P, G).astype(np.float64)


def load_labels(F):
    cmap, idx = F["canon_map"], F["idx"]; ns = len(F["canon_list"]); labels = np.full((ns, 7), np.nan)
    for name in ("archive/train.csv", "train.csv"):
        frame = pd.read_csv(BASE / name); frame["canon"] = frame["smiles"].map(cmap)
        for j, target in enumerate(["tg", "egc", "egb", "ei", "eea", "nc", "eps"]):
            vals = frame.loc[frame["target_type"].eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items(): labels[idx[canon], j] = value
    return labels.astype(np.float64)


def r2(y, p): return float(r2_score(np.asarray(y, float), np.asarray(p, float)))


def main():
    F, X = load_structure_features(); labels = load_labels(F)
    pi = pd.read_csv(BASE / "PI1M.csv", usecols=["SMILES"])["SMILES"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    if len(pi) > ROWS: pi = pi[rng.choice(len(pi), ROWS, replace=False)]
    official = np.asarray(F["canon_list"], dtype=object); corpus = np.concatenate([pi, official])
    count = {}
    for text in corpus:
        for token in set(toks(text)): count[token] = count.get(token, 0) + 1
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 4}
    next_id = 5
    for token in sorted(count):
        if token not in vocab: vocab[token] = next_id; next_id += 1
    graphs = Graphs(corpus, vocab)
    print(json.dumps({"phase": "setup", "corpus_rows": int(len(corpus)), "vocab_size": int(len(vocab)), "official_rows": int(len(official))}), flush=True)
    vocab_size = max(vocab.values()) + 1
    model, device, history = train_gin(graphs, vocab)
    official_graphs = Graphs(official, vocab); emb = embed(model, device, official_graphs)
    ei_rows = np.where(np.isfinite(labels[:, 3]))[0]; y = labels[ei_rows, 3]
    parent_all = np.load(SCR / "out_clean_corrected/PFINAL.npy").astype(np.float64); parent = parent_all[ei_rows, 3].copy()
    both = np.isfinite(labels[ei_rows, 1]) & np.isfinite(labels[ei_rows, 4]); parent[both] = .5 * parent[both] + .5 * (labels[ei_rows[both], 1] + labels[ei_rows[both], 4])
    reps = {"graph": emb[ei_rows], "structure_graph": np.hstack([X[ei_rows], emb[ei_rows]])}; metrics = {}
    for name, rep in reps.items():
        for weight in (.25, .50):
            key = f"{name}_residual_weight_{weight:.2f}"; pred = parent.copy(); rows = []
            for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + 170).split(rep), 1):
                fit = make_pipeline(StandardScaler(), Ridge(alpha=30.0)).fit(rep[tr], y[tr] - parent[tr]); raw = parent[va] + weight * fit.predict(rep[va]); pred[va] = raw
                rows.append({"fold": fold, "parent_r2": r2(y[va], parent[va]), "candidate_r2": r2(y[va], raw), "delta_r2": r2(y[va], raw) - r2(y[va], parent[va])})
            metrics[key] = {"parent_r2": r2(y, parent), "candidate_r2": r2(y, pred), "delta_r2": r2(y, pred) - r2(y, parent), "positive_folds": int(sum(x["delta_r2"] > 0 for x in rows)), "folds": rows}
    best = max(metrics, key=lambda key: metrics[key]["candidate_r2"])
    report = {"schema_version": "ppp.round2.clean-oof.v1", "experiment": "R2-C170-pi1m-graph-gin-ei", "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "kaggle_compute": False, "same_row_ei_label_as_feature": False, "mechanism": "random-initialized GIN-style graph encoder with masked atom prediction on official PI1M plus unlabeled official structures; frozen graph embeddings and residual Ridge", "corpus": {"pi1m_rows_used": int(min(len(pd.read_csv(BASE / "PI1M.csv", usecols=["SMILES"])), ROWS)), "official_structure_rows": int(len(official)), "total_rows": int(len(corpus)), "epochs": EPOCHS, "hidden": HIDDEN, "layers": LAYERS, "device": str(device), "vocab_size": int(vocab_size), "loss_history": history}, "rows": {"ei": int(len(ei_rows)), "both_eea_egc": int(np.sum(both))}, "metrics": metrics, "best_arm_by_clean_oof": best, "gate": {"ei_gain_at_least_0.005": bool(metrics[best]["delta_r2"] >= .005), "positive_folds_at_least_4": bool(metrics[best]["positive_folds"] >= 4), "passed_screen": bool(metrics[best]["delta_r2"] >= .005 and metrics[best]["positive_folds"] >= 4)}, "elapsed_seconds": time.time() - started}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "R2-C170-pi1m-graph-gin-ei-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2), flush=True)
    if not report["gate"]["passed_screen"]: print("C170 STOP: PI1M graph GIN Ei screen failed; no full-data fit, local_eval read, or submission artifact.", flush=True)


if __name__ == "__main__": main()
