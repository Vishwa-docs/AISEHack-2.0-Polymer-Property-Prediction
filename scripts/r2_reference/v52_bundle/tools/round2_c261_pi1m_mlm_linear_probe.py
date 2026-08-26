"""C261: an all-target PI1M masked-language representation screen.

This is deliberately a small, decisive gate for the second-wave representation
program.  It trains a Transformer from random initialization on a deterministic
100,000-row subset of the supplied PI1M.csv, freezes its pooled embeddings, and
compares fold-local Ridge probes against an identically initialized, untrained
control.  No target is read during pretraining, and no local_eval/test-external_label file
is opened.  The run is diagnostic clean evidence only; it does not create a
submission or perform a full-data fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ppp-round-2"
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
SEED = 20260805
PI1M_ROWS = 100_000
MAX_LEN = 64
HIDDEN = 128
LAYERS = 2
HEADS = 4
BATCH = 256
EPOCHS = 1
MLM_PROB = 0.15
TOKEN_PATTERN = re.compile(
    r"Cl|Br|Si|Na|Li|Al|Ca|Fe|Cu|Zn|Mg|%\d{2}|\[[^\]]+\]|"
    r"[A-Z][a-z]?|[bcnops]|\*|[0-9]|[()=#[\].+\-/:\\@]"
)
SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))


def tokenize(smiles: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(smiles))


def group_key(smiles: str) -> str:
    """Cheap endpoint-orientation key used only for this pre-screen.

    The strict production loop uses the stronger canonical/repeat grouping.  A
    raw-string reverse key is intentionally disclosed here because C261 is a
    representation gate, not a deployable candidate.
    """
    text = "".join(str(smiles).split())
    return min(text, text[::-1])


def sha_rank(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def build_vocab(texts: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            counts[token] = counts.get(token, 0) + 1
    vocab = {token: i for i, token in enumerate(SPECIAL)}
    for token in sorted(counts):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int]) -> np.ndarray:
    body = [vocab.get(token, vocab["[UNK]"]) for token in tokenize(text)]
    body = body[: MAX_LEN - 2]
    ids = [vocab["[BOS]"]] + body + [vocab["[EOS]"]]
    ids.extend([vocab["[PAD]"]] * (MAX_LEN - len(ids)))
    return np.asarray(ids[:MAX_LEN], dtype=np.int64)


class SmilesDataset(Dataset):
    def __init__(self, texts: np.ndarray, vocab: dict[str, int]) -> None:
        self.ids = np.stack([encode(text, vocab) for text in texts])
        self.pad = vocab["[PAD]"]
        self.mask = vocab["[MASK]"]
        self.vocab_size = len(vocab)
        self.bos = vocab["[BOS]"]
        self.eos = vocab["[EOS]"]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        original = torch.from_numpy(self.ids[index].copy())
        masked = original.clone()
        labels = torch.full_like(original, -100)
        eligible = (original != self.pad) & (original != self.bos) & (original != self.eos)
        choose = (torch.rand(original.shape) < MLM_PROB) & eligible
        if not bool(choose.any()) and bool(eligible.any()):
            eligible_indices = torch.where(eligible)[0]
            choose[eligible_indices[0]] = True
        labels[choose] = original[choose]
        mask_choose = choose & (torch.rand(original.shape) < 0.80)
        random_choose = choose & ~mask_choose & (torch.rand(original.shape) < 0.50)
        masked[mask_choose] = self.mask
        if bool(random_choose.any()):
            masked[random_choose] = torch.randint(5, self.vocab_size, (int(random_choose.sum()),))
        valid = original != self.pad
        return masked, labels, valid


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, HIDDEN, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=HEADS,
            dim_feedforward=HIDDEN * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.norm = nn.LayerNorm(HIDDEN)
        self.lm = nn.Linear(HIDDEN, vocab_size)

    def forward(self, ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        hidden = self.token(ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~valid)
        return self.norm(hidden)


def write_progress(run_dir: Path, event: dict) -> None:
    with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def train_mlm(corpus: np.ndarray, vocab: dict[str, int], run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED)
    model = SmilesEncoder(len(vocab)).to(device)
    dataset = SmilesDataset(corpus, vocab)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    history: list[float] = []
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        batches = 0
        for ids, labels, valid in loader:
            ids = ids.to(device)
            labels = labels.to(device)
            valid = valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model(ids, valid)
            logits = model.lm(hidden)
            loss = loss_fn(logits.reshape(-1, len(vocab)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            if batches % 100 == 0:
                write_progress(run_dir, {"phase": "mlm", "epoch": epoch + 1, "batch": batches, "loss": total_loss / batches})
        mean_loss = total_loss / max(1, batches)
        history.append(mean_loss)
        event = {"phase": "mlm_epoch", "epoch": epoch + 1, "epochs": EPOCHS, "loss": mean_loss, "device": str(device), "batches": batches}
        write_progress(run_dir, event)
        print(json.dumps(event), flush=True)
    return model, device, history


def embed(model: SmilesEncoder, device: torch.device, texts: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    ids = np.stack([encode(text, vocab) for text in texts])
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), BATCH):
            batch = torch.from_numpy(ids[start : start + BATCH]).to(device)
            valid = batch != vocab["[PAD]"]
            hidden = model(batch, valid)
            # BOS is the pooled sequence representation.
            output.append(hidden[:, 0].cpu().numpy())
    return np.vstack(output).astype(np.float64)


def random_control_embeddings(model: SmilesEncoder, device: torch.device, texts: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    # The model has just completed MLM training, so recreate the exact same
    # architecture with the same seed to obtain a matched random-init control.
    seed_everything(SEED)
    control = SmilesEncoder(len(vocab)).to(device)
    return embed(control, device, texts, vocab)


def probe_target(
    target_frame: pd.DataFrame,
    embedding_by_smiles: dict[str, np.ndarray],
    control_by_smiles: dict[str, np.ndarray],
) -> dict:
    frame = target_frame.dropna(subset=["target"]).copy()
    texts = frame["smiles"].astype(str).to_numpy()
    y = frame["target"].astype(float).to_numpy()
    groups = np.asarray([group_key(text) for text in texts], dtype=object)
    candidate_x = np.vstack([embedding_by_smiles[text] for text in texts])
    control_x = np.vstack([control_by_smiles[text] for text in texts])
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    candidate_oof = np.full(len(frame), np.nan)
    control_oof = np.full(len(frame), np.nan)
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=n_splits).split(candidate_x, y, groups), 1):
        candidate_fit = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        control_fit = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        candidate_fit.fit(candidate_x[train_idx], y[train_idx])
        control_fit.fit(control_x[train_idx], y[train_idx])
        candidate_oof[valid_idx] = candidate_fit.predict(candidate_x[valid_idx])
        control_oof[valid_idx] = control_fit.predict(control_x[valid_idx])
        folds.append(
            {
                "fold": fold,
                "rows": int(len(valid_idx)),
                "candidate_r2": float(r2_score(y[valid_idx], candidate_oof[valid_idx])),
                "control_r2": float(r2_score(y[valid_idx], control_oof[valid_idx])),
            }
        )
    candidate_r2 = float(r2_score(y, candidate_oof))
    control_r2 = float(r2_score(y, control_oof))
    return {
        "rows": int(len(frame)),
        "groups": int(n_groups),
        "n_splits": int(n_splits),
        "candidate_r2": candidate_r2,
        "control_r2": control_r2,
        "delta_r2": candidate_r2 - control_r2,
        "positive_folds": int(sum(f["candidate_r2"] > f["control_r2"] for f in folds)),
        "folds": folds,
        "oof": {"candidate": candidate_oof, "control": control_oof},
        "y": y,
        "texts": texts,
        "groups_vector": groups,
    }


def write_manifest(run_dir: Path) -> None:
    entries = []
    for path in sorted(run_dir.glob("*")):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(run_dir)}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    started = time.time()
    write_progress(run_dir, {"phase": "started", "experiment": "R2-C261", "seed": SEED, "timestamp": time.time()})

    pi1m = pd.read_csv(DATA / "PI1M.csv", usecols=["SMILES"])
    pi1m_texts = pi1m["SMILES"].astype(str).dropna().drop_duplicates().to_numpy()
    selected = sorted(pi1m_texts, key=sha_rank)[:PI1M_ROWS]
    corpus = np.asarray(selected, dtype=object)
    vocab = build_vocab(corpus)
    write_progress(run_dir, {"phase": "corpus_ready", "pi1m_source_rows": int(len(pi1m_texts)), "corpus_rows": int(len(corpus)), "vocab_size": int(len(vocab)), "max_len": MAX_LEN})
    print(json.dumps({"phase": "corpus_ready", "rows": len(corpus), "vocab": len(vocab)}), flush=True)

    model, device, loss_history = train_mlm(corpus, vocab, run_dir)

    train = pd.read_csv(DATA / "train.csv", usecols=["smiles", "target_type", "target"])
    test = pd.read_csv(DATA / "test.csv", usecols=["smiles"])
    official_texts = pd.concat([train[["smiles"]], test[["smiles"]]], ignore_index=True)["smiles"].astype(str).drop_duplicates().to_numpy()
    # Fit no target-dependent object before these embeddings are materialized.
    trained_embeddings = embed(model, device, official_texts, vocab)
    control_embeddings = random_control_embeddings(model, device, official_texts, vocab)
    embedding_by_smiles = {text: trained_embeddings[i] for i, text in enumerate(official_texts)}
    control_by_smiles = {text: control_embeddings[i] for i, text in enumerate(official_texts)}
    np.savez_compressed(run_dir / "official_embeddings.npz", trained=trained_embeddings, control=control_embeddings)
    (run_dir / "official_embedding_keys.json").write_text(json.dumps(official_texts.tolist()) + "\n", encoding="utf-8")

    target_metrics: dict[str, dict] = {}
    oof_rows: list[dict] = []
    for target in TARGETS:
        target_frame = train.loc[train["target_type"].eq(target), ["smiles", "target"]]
        result = probe_target(target_frame, embedding_by_smiles, control_by_smiles)
        target_metrics[target] = {key: value for key, value in result.items() if key not in {"oof", "y", "texts", "groups_vector"}}
        for i, (text, y_value, group) in enumerate(zip(result["texts"], result["y"], result["groups_vector"])):
            oof_rows.append(
                {
                    "target_type": target,
                    "smiles": text,
                    "group": group,
                    "target": float(y_value),
                    "candidate": float(result["oof"]["candidate"][i]),
                    "control": float(result["oof"]["control"][i]),
                }
            )
        print(json.dumps({"phase": "probe", "target": target, **target_metrics[target]}), flush=True)
        write_progress(run_dir, {"phase": "probe", "target": target, **target_metrics[target]})

    candidate_mean = float(np.mean([target_metrics[t]["candidate_r2"] for t in TARGETS]))
    control_mean = float(np.mean([target_metrics[t]["control_r2"] for t in TARGETS]))
    positive_targets = int(sum(target_metrics[t]["candidate_r2"] > target_metrics[t]["control_r2"] for t in TARGETS))
    weak_positive = any(
        target_metrics[t]["delta_r2"] >= 0.010 for t in ("eps", "nc", "ei")
    )
    passed = bool(candidate_mean > control_mean and (positive_targets >= 4 or weak_positive))
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment_id": "R2-C261-20260805-pi1m-mlm-linear-probe-v1",
        "status": "passed_screen" if passed else "rejected_screen",
        "official_only_fitting": True,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "pretrained_weights": False,
        "external_targets": False,
        "architecture": {"hidden": HIDDEN, "layers": LAYERS, "heads": HEADS, "max_len": MAX_LEN, "epochs": EPOCHS, "mlm_probability": MLM_PROB, "batch": BATCH, "seed": SEED, "device": str(device), "loss_history": loss_history},
        "corpus": {"source": "ppp-round-2/PI1M.csv", "selected_rows": int(len(corpus)), "selection": "lowest SHA256 rank of raw SMILES", "vocab_size": int(len(vocab))},
        "rows": {target: int((train["target_type"] == target).sum()) for target in TARGETS},
        "targets": target_metrics,
        "candidate_mean_r2": candidate_mean,
        "control_mean_r2": control_mean,
        "candidate_minus_control_mean": candidate_mean - control_mean,
        "positive_targets": positive_targets,
        "gate": {"mean_beats_control": bool(candidate_mean > control_mean), "positive_targets_at_least_4": bool(positive_targets >= 4), "weak_target_gain_at_least_010": bool(weak_positive), "passed_screen": passed},
        "elapsed_seconds": time.time() - started,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(oof_rows).to_csv(run_dir / "oof_probe.csv", index=False)
    decision = "passed_screen" if passed else "rejected_screen"
    (run_dir / "decision.md").write_text(
        f"# C261 decision\n\n"
        f"Status: **{decision}**.\n\n"
        f"The frozen PI1M Transformer probe mean is `{candidate_mean:.9f}` and the matched random-init control mean is `{control_mean:.9f}`. "
        f"This is clean official-only OOF evidence; it does not read an local_eval, create a submission, or authorize a larger representation run.\n",
        encoding="utf-8",
    )
    write_manifest(run_dir)
    write_progress(run_dir, {"phase": "completed", "status": decision, "candidate_mean_r2": candidate_mean, "control_mean_r2": control_mean, "elapsed_seconds": time.time() - started})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
