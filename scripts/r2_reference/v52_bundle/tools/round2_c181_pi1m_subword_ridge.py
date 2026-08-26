#!/usr/bin/env python3
"""C181: low-cost PI1M-from-scratch SMILES subword screen.

The tokenizer is learned from a deterministic, hash-ranked subset of the
official unlabeled PI1M.csv plus unlabeled official train/test SMILES.  It is
not a pretrained tokenizer and it never reads external_label labels.  The resulting
subword count matrix is used by target-local Ridge regressors and compared
with the exact C050 parent under grouped OOF evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = tuple(reference.TARGETS)
SEED = 2026
PI1M_LIMIT = 25000
MERGES = 96
MAX_FEATURES = 8192
ALPHA = 30.0
BLEND = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_smiles_column(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    candidates = [name for name in frame.columns if str(name).lower() in {"smiles", "smile", "canonical_smiles", "p-smiles", "psmiles"}]
    if not candidates:
        candidates = [name for name in frame.columns if "smile" in str(name).lower()]
    if not candidates:
        raise RuntimeError(f"no SMILES column in {path}")
    return frame[candidates[0]].dropna().astype(str).tolist()


def hash_ranked(values: list[str], limit: int) -> list[str]:
    unique = sorted(set(values), key=lambda value: hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest())
    return unique[:limit]


def train_merges(corpus: list[str], merges: int) -> list[tuple[str, str]]:
    sequences = [list(value) + ["<E>"] for value in corpus]
    learned: list[tuple[str, str]] = []
    for _ in range(merges):
        counts: Counter[tuple[str, str]] = Counter()
        for sequence in sequences:
            counts.update(zip(sequence, sequence[1:]))
        if not counts:
            break
        pair, count = counts.most_common(1)[0]
        if count < 2:
            break
        learned.append(pair)
        merged = "".join(pair)
        for index, sequence in enumerate(sequences):
            updated: list[str] = []
            cursor = 0
            while cursor < len(sequence):
                if cursor + 1 < len(sequence) and (sequence[cursor], sequence[cursor + 1]) == pair:
                    updated.append(merged)
                    cursor += 2
                else:
                    updated.append(sequence[cursor])
                    cursor += 1
            sequences[index] = updated
    return learned


def apply_merges(smiles: list[str], merges: list[tuple[str, str]]) -> list[str]:
    output: list[str] = []
    for value in smiles:
        sequence = list(value) + ["<E>"]
        for pair in merges:
            merged = "".join(pair)
            updated: list[str] = []
            cursor = 0
            while cursor < len(sequence):
                if cursor + 1 < len(sequence) and (sequence[cursor], sequence[cursor + 1]) == pair:
                    updated.append(merged)
                    cursor += 2
                else:
                    updated.append(sequence[cursor])
                    cursor += 1
            sequence = updated
        output.append(" ".join(sequence))
    return output


def build_matrix(smiles: list[str], pi1m: list[str]) -> tuple[sparse.csr_matrix, dict[str, Any], list[tuple[str, str]]]:
    corpus = hash_ranked(pi1m, PI1M_LIMIT)
    merges = train_merges(corpus, MERGES)
    tokenized_corpus = apply_merges(corpus, merges)
    tokenized_values = apply_merges(smiles, merges)
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\S+",
        lowercase=False,
        min_df=2,
        max_features=MAX_FEATURES,
        binary=False,
    )
    vectorizer.fit(tokenized_corpus + tokenized_values)
    counts = vectorizer.transform(tokenized_values).astype(np.float64)
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    matrix = tfidf.fit_transform(sparse.vstack([vectorizer.transform(tokenized_corpus), counts], format="csr"))[-len(smiles):]
    report = {
        "source": "official PI1M.csv plus unlabeled official train/test SMILES",
        "pi1m_rows": int(len(pi1m)),
        "pi1m_hash_ranked_rows": int(len(corpus)),
        "merge_count": int(len(merges)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "matrix_shape": [int(value) for value in matrix.shape],
        "nnz": int(matrix.nnz),
        "tfidf": True,
    }
    return matrix.tocsr(), report, merges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    pi1m = read_smiles_column(data_dir / "PI1M.csv")
    matrix, feature_report, merges = build_matrix(parent["keys"], pi1m)
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        y = np.asarray(info["y"], dtype=float)
        parent_oof = np.asarray(info["parent"], dtype=float)
        indices = np.asarray(info["indices"], dtype=np.int64)
        groups = np.asarray(info["groups"], dtype=object)
        folds = carrier.grouped_folds(groups)
        subword_oof = np.full(len(y), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(carrier.N_FOLDS):
            validation = np.flatnonzero(folds == fold)
            training = np.flatnonzero(folds != fold)
            model = Ridge(alpha=ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
            model.fit(matrix[indices[training]], y[training])
            subword_oof[validation] = model.predict(matrix[indices[validation]])
            fold_rows.append({
                "fold": fold,
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(y[validation], parent_oof[validation])),
                "subword_r2": float(r2_score(y[validation], subword_oof[validation])),
            })
        candidate = parent_oof + BLEND * (subword_oof - parent_oof)
        delta = float(r2_score(y, candidate) - r2_score(y, parent_oof))
        positive = int(sum(r["delta_r2"] > 0 for r in [{"delta_r2": row["subword_r2"] - row["parent_r2"]} for row in fold_rows]))
        lower = carrier.bootstrap_lower(y, parent_oof, candidate, groups)
        full_model = Ridge(alpha=ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
        full_model.fit(matrix[indices], y)
        test_indices = np.asarray([
            parent["key_to_index"][value]
            for value in parent["test"].loc[parent["test"]["target_type"] == target, "canonical"]
        ], dtype=np.int64)
        direct_test = full_model.predict(matrix[test_indices])
        selected = bool(delta >= 0.005 and positive >= 4 and lower > 0.0)
        target_reports[target] = {
            "parent_r2": float(r2_score(y, parent_oof)),
            "subword_r2": float(r2_score(y, subword_oof)),
            "candidate_r2": float(r2_score(y, candidate)),
            "delta_r2": delta,
            "positive_folds": positive,
            "group_bootstrap_lower": lower,
            "pass": selected,
            "folds": fold_rows,
        }
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": y,
            "parent": parent_oof,
            "subword": subword_oof,
            "candidate": candidate,
            "assembled": np.where(selected, candidate, parent_oof),
            "fold": folds,
        }))
        test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_parts.append(pd.DataFrame({"id": test_rows["id"].astype(int), "target_type": target, "subword": direct_test}))

    banked = [target for target in TARGETS if target_reports[target]["pass"]]
    oof = pd.concat(oof_parts, ignore_index=True)
    parent_mean = float(np.mean([target_reports[t]["parent_r2"] for t in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    test_parts_frame = pd.concat(test_parts, ignore_index=True)
    predictions = parent_test.merge(test_parts_frame, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["target"] + BLEND * (predictions["subword"] - predictions["target"]), predictions["target"])
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C181 complete output contract failed")
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and len(banked) > 0)
    report = {
        "schema_version": "ppp.round2.c181.pi1m-subword-ridge.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"] | {"PI1M.csv": {"sha256": sha256_file(data_dir / "PI1M.csv"), "rows": len(pi1m)}},
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
            "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py"),
        },
    }
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    test_parts_frame.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c181.pi1m-subword-ridge.v1", "seed": SEED, "pi1m_limit": PI1M_LIMIT, "merges": MERGES, "max_features": MAX_FEATURES, "ridge_alpha": ALPHA, "blend": BLEND, "local_eval_read": False, "tokenizer": "learned from scratch on hash-ranked official PI1M SMILES"})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; gain `{assembled_mean - parent_mean:+.12f}`. Official PI1M was used only for from-scratch unlabeled subword learning; no local_eval read.\n", encoding="utf-8")
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": assembled_mean, "mean_gain": assembled_mean - parent_mean, "decision": report["decision"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
