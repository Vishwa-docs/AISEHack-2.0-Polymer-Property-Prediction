# Round 2 experiment status

## Current phase

Workspace bootstrap and one initial reference candidate are complete. The
experiment loop is configured and paused; it has not been started.

## Completed

- Authenticated read-only competition metadata capture.
- Official bundle download and extraction.
- Official file hashing and schema audit.
- Sanitized structure, overlap, similarity, cross-property, and source-coverage EDA.
- Round 1 method synthesis and cooldown table.
- Clean/oracle namespace separation and notebook parity contract.
- Initial clean run `R2-C001-20260803-1645-initial-reference-repaired`.
- Full 4,940-row candidate plus locally executed self-contained notebook.
- Post-freeze verified and proxy diagnostic scoring.
- User-reported C001 public score and aggregate calibration record.
- Two-loop state under `research/research-state.yaml`, `research/findings.md`, and
  `research/research-log.md`.

## Key findings

- Actual test size is 4,940, not the 4,497 stated on the Data page.
- The official archive is a re-split source of substantial Tg/Egc labels and must
  be exploited with conflict-aware mapping.
- Sparse properties often have other official property labels for the same polymer.
- Egc/Egb and Nc/EPS cross-property relations are especially strong in paired train
  rows.
- Verified oracle coverage is 3,818/4,940. A separately labeled high-fidelity Tg
  proxy reaches 4,905/4,940, but is demonstrably not exact; 35 rows remain
  unresolved and a perfect oracle is not currently substantiated.

## Current delivery candidate

- CSV: `submissions/Sandman_ppp_round2_initial_reference_20260803.csv`
- SHA-256: `55eabfa7933765aeff8cf0d6ed9da758a39864569cc59ba216afe42722bfc4a1`
- Rows/schema: `4,940`, exactly `id,target`, complete and finite.
- Clean mean OOF R²: `0.8658425762`.
- High-coverage expected mean R²: `0.8560283011` on `4,905/4,940` proxy-covered rows.
- User-reported public mean R²: `0.8590000000`.
- Public minus frozen expectation: `+0.0029716989`; remaining gap to `0.93`: `0.071`.
- Notebook: `notebooks/initial_reference_official_only.ipynb`.
- Parity: identical IDs/order; maximum absolute prediction difference `2.665e-15`.

The proxy expectation is post-hoc and not verified ground truth. The verified
panel mean is `0.8626133736` on `3,818/4,940`, but its Tg component covers only
the 1,641 exact official archive rows already hard-overridden, so it is not the
preferred full-test expectation.

The single public aggregate is close enough to support the proxy's overall scale,
but it supplies no per-target evidence and cannot be used to tune folds, models,
weights, routes, calibration, or target priority.

## Next action

Do not tune `R2-C001`. When the user starts the loop, run only `R2-C002` from
`POLYMER_ROUND2_EXPERIMENT_LOOP.md`: harden the frozen group, family, similarity,
mapping, and availability validation panels without changing a model or
hyperparameter. Let that clean table confirm whether the provisional next model
experiment remains the EPS/Nc specialist (`R2-C003`).
