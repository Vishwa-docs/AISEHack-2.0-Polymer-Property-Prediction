# logs/

Append-only records. Never rewrite existing lines; never recycle experiment ids.

- `experiments.jsonl` — one JSON line per experiment allocation/result (schema
  follows the Round 2 convention: experiment_id, state, timestamps, metrics,
  hashes, decision).
- `LEADERBOARD_LOG.md` — human-readable leaderboard history. Only the user
  reports public scores; agents never submit.
- `oracle_scores.jsonl` — post-freeze oracle/proxy scoring records
  (`oracle-observed`, verification lane only).

## Known Round 2 anchors (for reference)

- R2 C001 public score: 0.859 (user-reported, 2026-08-03).
- R2 best clean no-archive composite: verified 0.9041496 / proxy 0.9030465
  (Sandman V57, local oracle lane).
- R2 user-submitted no-archive public score: 0.891.
- Round 3 leaderboard: a submission at 0.92 exists — we must beat it.
- Round 3 targets: public ≥ 0.93; local verified-oracle ≥ 0.935 (see PLAN.md).
