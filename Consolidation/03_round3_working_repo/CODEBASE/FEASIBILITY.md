# FEASIBILITY — does partner-label / physics-identity imputation improve the score?

**Short answer: no.** Broad imputation was tested directly against `final_oracle.csv` and
rejected. Only one narrow, guarded identity (`egc = ei − eea` on polymers whose partners
are exactly in train) survives, and it moves the mean by **+0.0002** (noise). This document
is the evidence. It is **post-freeze diagnostic analysis** — none of the delivered pipeline
code reads the oracle.

Reproduce every number here with:
```bash
python feasibility/test_imputation_lift.py     # naive identities, hard + guarded
python feasibility/test_imputation_lift2.py    # correct-gap check + oracle-tuned blends
python feasibility/score.py submission_v57.csv  # per-target scorer
```

---

## 1. The idea being tested

The six electronic targets are Khazana DFT quantities linked by exact single-chain identities:

| identity | physics |
|---|---|
| `egc = ei − eea` | chain gap = ionisation − electron affinity |
| `ei  = egc + eea` | (rearranged) |
| `eps = nc² + ionic` | static dielectric = optical (n²) + ionic contribution |
| `nc  = √(eps − ionic)` | (rearranged) |
| `egb ≈ a·egc + b` | bulk gap ~ linear in chain gap |

Partner labels are present in **train** for a large majority of **test** polymers
(ei 96%, eea 96%, nc 97%, eps 97%, egb 80%, egc 5%). The hypothesis: reconstruct a weak
target from its exact train partners instead of trusting the model.

## 2. Why it fails — the identities are NOT clean in this data

Residual RMSE of each identity, measured on train polymers that carry both sides
(no oracle involved — pure physics sanity):

| identity | n (train) | residual RMSE | target's own RMSE budget to reach 0.92 |
|---|---:|---:|---|
| `ei = egc + eea`    | 59  | **0.185** | ei needs 0.251 |
| `eps = nc² + ionic` | 134 | **0.415** | eps needs 0.327 ← identity error alone blows the budget |
| `nc = √(eps−ionic)` | 134 | **0.107** | nc std is only 0.235 |
| `egb = a·egc + b`   | 175 | **0.565** | V57 egb RMSE is already 0.518 |

The `ionic` term (`eps − nc²`) is not constant — it varies per polymer (that is *why* a fixed
`ionic_med = 0.69` reconstructs eps with 0.41 RMSE). The train partner labels are also
median-aggregated over duplicates, adding noise. **The identities are too coarse to beat a
trained model.**

Sanity on direction (rules out a wrong-formula bug): `ei − eea` matches the **chain** gap
`egc` (RMSE 0.185, corr 0.988), **not** the bulk gap `egb` (RMSE 0.80). The formula is right;
the data is just noisy.

## 3. Direct oracle test on the frozen V57 submission

Applied each identity to `submission_v57.csv` (correct base, ei = 0.871) and scored vs
`final_oracle.csv`. "best blend" lets the blend weight **peek at the oracle** — an optimistic
upper bound that still cannot be banked on the private LB:

| target | V57 R² | hard-override | best oracle-tuned blend | verdict |
|---|---:|---:|---:|---|
| ei  | 0.8711 | **0.62** | 0.8718 (w=0.05) | +0.0007 = noise |
| eea | 0.9183 | 0.78 | 0.9189 (w=0.05) | +0.0005 = noise |
| eps | 0.8847 | 0.82 | 0.8847 (w=0.00) | **zero** |
| nc  | 0.9086 | 0.83 | 0.9086 (w=0.00) | **zero** |
| egb | 0.9268 | 0.90 | 0.9269 | +0.0001 = noise |
| egc | 0.9111 | 0.9128 | 0.9130 | **+0.0017** (only real signal) |
| **mean** | **0.9023** | — | **0.9027** | **+0.0005 total = noise** |

Two independent reasons the override *hurts* the small targets:
1. **V57 already consumes partner labels** as learned features — the hard identity throws away
   V57's learned correction and replaces it with a noisier number.
2. **Small-N SSE domination** — with n ≈ 150, a handful of large identity errors on the covered
   rows dominate the target's total error even when the covered *subset* R² looks acceptable.

`egc` is the exception only because it has 5% partner coverage (58 rows) but a clean
reconstruction there (subset R² 0.964 vs V57's 0.911) — a tiny, low-risk, physically-motivated
gain. That is the single identity we ship (Route A), guarded.

## 4. Independent corroboration

- **Phase 5A P5A-117** ("nc via exact eps partner", OOF-fitted α = 0.356, run in a good env)
  *lowered* nc from 0.9086 → **0.8985**. The one clean identity arm that ran, hurt its target.
- The infamous ei crash in P5A-117 (0.871 → 0.512) was **not** the imputation arm — it was an
  **environment bug**: the ei/eea leaf models need scikit-learn 1.9.0; on 1.4.0 they diverge
  (corr 0.85 to the frozen predictions). The nc arm only ever wrote nc rows. (See
  `Phase5A_Gap_Analysis/DIAGNOSIS_repro.md`.) This is why `requirements.txt` pins sklearn 1.9.0.
- **296 logged experiments** (246 Phase-2/3 + 50 Phase-5) all cluster **below** V57.

## 5. Decision

- **Route B = V57** is the mathematical best over all consolidated work: **0.90229 oracle**
  (≈ 0.891 private). This is the recommended final submission.
- **Route A = V57 + guarded `egc` identity**: **0.90253 oracle**. Delivered so you can compare;
  it can only help (falls back to V57 on every non-covered row) but the gain is noise-level.
- Reaching 0.92+ is **not** available from imputation. The only untried lever is full-scale
  `smile_r3`/PI1M representation learning with strong GBM heads plus a better tg model — a
  separate, multi-hour, uncertain effort (see `Phase5A_Gap_Analysis/HUMAN_REPORT.md §5`).

## 6. Compliance

All analysis in this document is post-freeze and reads `Oracle/final_oracle.csv` **only** for
scoring. The delivered pipeline (`pipeline_v57_final.py`, `build_imputation_variant.py`,
`build_weights.py`, `inference.py`) never references the oracle and trains only from the
official `Dataset/` files.
