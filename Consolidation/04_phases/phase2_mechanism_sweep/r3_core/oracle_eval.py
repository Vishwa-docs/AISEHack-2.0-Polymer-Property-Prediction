"""POST-FREEZE oracle evaluation helper — SEPARATE namespace.

This module is intentionally NOT imported by any experiment or clean path.  It is
invoked only AFTER a candidate CSV is frozen and hashed, by the agent, to score
against the local verification panel.  It must never appear in any notebook or
submitted artifact.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score_panel(candidate: pd.DataFrame, truth: pd.DataFrame) -> dict[str, object]:
    merged = truth[["id", "target_type", "target"]].merge(
        candidate.rename(columns={"target": "prediction"}), on="id", how="left", validate="one_to_one"
    )
    if merged["prediction"].isna().any():
        raise RuntimeError("candidate is missing oracle IDs")
    per_target: dict[str, object] = {}
    scores: list[float] = []
    for target in TARGETS:
        rows = merged[(merged["target_type"] == target) & merged["target"].notna()]
        if len(rows) < 2:
            raise RuntimeError(f"insufficient covered rows for {target}")
        y = rows["target"].to_numpy(float)
        p = rows["prediction"].to_numpy(float)
        r2 = float(r2_score(y, p))
        scores.append(r2)
        per_target[target] = {
            "covered_rows": int(len(rows)),
            "r2": r2,
            "mae": float(mean_absolute_error(y, p)),
        }
    return {
        "covered_rows": int(merged["target"].notna().sum()),
        "per_target": per_target,
        "unweighted_mean_r2": float(np.mean(scores)),
    }


def score_candidate(
    candidate_path: str,
    verified_path: str,
    proxy_path: str,
    output_path: str,
) -> dict[str, object]:
    candidate = pd.read_csv(candidate_path)
    if list(candidate.columns) != ["id", "target"] or len(candidate) != 4940:
        raise RuntimeError("candidate schema invalid")
    verified = pd.read_csv(verified_path)
    proxy = pd.read_csv(proxy_path)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "ORACLE_ASSISTED_RESEARCH_ONLY",
        "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "verified_oracle": {"score": score_panel(candidate, verified)},
        "proxy_diagnostic": {"score": score_panel(candidate, proxy)},
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
