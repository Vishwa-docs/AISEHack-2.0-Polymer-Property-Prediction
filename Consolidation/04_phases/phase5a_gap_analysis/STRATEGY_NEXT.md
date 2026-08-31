# STRATEGY_NEXT — what to do now (answers to: normalize Tg? weak targets? next techniques)

All numbers from Phase5A_Gap_Analysis/output (scripts 09, 10 + earlier tables). Oracle used post-freeze only.

---

## 1. Your "1% of entries = 26% of Tg errors" claim: VERIFIED (and it is the key to Tg)

| target | top 1% rows share of SSE | top 5% | top 10% | R2 now | R2 if worst-1% errors were HALVED |
|--------|------------------------:|-------:|--------:|-------:|-----------------------------------:|
| tg | **26.4%** (27 rows) | 55.5% | 69.9% | 0.8953 | **+0.0207 -> 0.9160** |
| egc | 20.0% | 47.8% | 64.2% | 0.9111 | +0.0134 |
| nc | 22.4% | 49.2% | 63.0% | 0.9086 | +0.0154 |
| eps | 19.1% | 48.0% | 62.7% | 0.8847 | +0.0165 |
| ei | 12.3% | 37.4% | 58.8% | 0.8711 | +0.0119 |
| egb | 14.7% | 44.3% | 59.6% | 0.9268 | +0.0080 |
| eea | 7.9% | 39.2% | 56.2% | 0.9183 | +0.0048 |

Implication: Tg's score is set by a fat tail of ~27-140 hard rows (novel/unusual polymers).
The winning Tg move is NOT global tuning — it is generalizing on hard/novel structures
(features from smile_r3 at scale, scaffold-diverse CV, residual correction).
If the worst-1% Tg errors were halved: tg +0.021 = **+0.0030 on the mean** (the single biggest
reliable lever in the whole competition).

## 2. "Should we normalize Tg?": NO — provably a no-op for the score

R2(y,p) = R2(0.01*y, 0.01*p) = R2(0.01*y+100, 0.01*p+100) = 0.895346 (identical, script 09).
Per-target R2 is affine-invariant: scaling Tg (or any target) changes NOTHING on the leaderboard.

Normalization matters ONLY if you train joint/multi-target models (one loss over targets):
tg = 99.99% of all TSS, so any unnormalized joint MSE loss IS a Tg loss. If you ever train a
shared model, standardize each target first (per-target z-score) — never because "values look
big", always because of the loss scale. Also: prediction clipping to train bounds gains
~0.0000-0.0021 everywhere (quantile clipping slightly HURTS egc/ei/nc/eps) — not a lever.

## 3. Weak targets — how to raise them (the routes, with numbers)

The small targets are NOT "easy points": their R2 SE is 0.020-0.024 (script 04), so oracle deltas
under ~0.04 are indistinguishable from noise, and +0.01 R2 on any of them is still only
+0.00143 mean. The legitimate routes, all clean (train-only):

| target | route | evidence from train |
|--------|-------|---------------------|
| eps | predict ionic = eps - nc^2 (residual model), not eps | 134 paired polymers; nc^2 removes 61.8% of eps variance; ionic std 0.41 vs eps std 1.07 -> model the small residual |
| egb | use egc as covariate | egb = 1.103*egc - 0.694, R2 = 0.86 on 82 pairs, resid std 0.68 eV |
| ei | partner reconstruction ei ~ egc + eea | DFT-level identity; only 10 same-polymer triples in train to validate -> use as soft constraint/guard, not hard equality |
| nc | nc^2 consistency with eps (joint eps-nc fit) | shared 134 polymers |
| eea | joint ei+eea model (chi = (ei+eea)/2 reparam) | 123 shared polymers |

Partner availability (test rows whose SMILES has another property's label in train):
ei 96%, eea 96%, nc 97%, eps 97%, egb 80%, egc 5%, tg 0.2%. So for the DFT cluster, build
per-polymer partner features (predict all 6 DFT props per polymer, feed cross-property)
and guard every overlay with shrinkage toward the incumbent (transfer-guard style).

## 4. What to do next — prioritized queue (with kill gates)

### A. Tg thrust (biggest reliable lever; SE 0.007)
1. A1 — Hard-row attack: identify the ~5% hardest Tg rows (low train similarity), build
   smile_r3/PI1M-scale features (word2vec/SVD/Morgan vocab at 5.97M — UNTRIED at scale; all
   prior probes were <=250k rows) + scaffold/grouped 5-fold CV. Gate: tg >= +0.010.
2. A2 — OOF residual kriging: fit Tanimoto kNN on TRAIN OOF residuals only, add weighted
   neighbor residual to test preds. Gate: tg >= +0.005 or kill (read-across variants hurt before).
3. A3 — Huber/robust loss + fold-local NNLS blend on the Tg stack. Gate: tg >= +0.005.

### B. Small targets via physics (each worth up to +0.004 mean if +0.03 R2 lands)
4. B1 — eps ionic residual model (61.8% variance removed) with transfer guard. Gate: eps >= +0.010 fold-consistent.
5. B2 — ei partner reconstruction (egc+eea covariate) with guard. Gate: ei >= +0.010 fold-consistent.
6. B3 — egb via egc covariate (+ nc-eps consistency). Gate: egb >= +0.005.

### C. egc (2028 train rows — 2nd most data; SE 0.006 — reliable measurement)
7. C1 — GBM push on egc with best features + fold-local blend. Gate: egc >= +0.010.

Expected result if all gates hit: tg +0.02, eps +0.02, ei +0.02, egc +0.01, nc +0.01, eea +0.01,
egb +0.005 -> mean +0.0135 -> 0.916 oracle (~0.905 private). Aggressive case: tg +0.035, ei +0.04,
eps +0.035, rest +0.02 -> ~0.928-0.932 oracle (~0.917-0.921 private). The 0.935 stretch needs the
aggressive case PLUS breaking the tg ~0.92 ceiling.

## 5. "Nothing arbitrary" principles (contest-safe)
- Every transform (clipping, scaling, kriging, blending) fitted/derived from TRAIN data only;
  no test-value inspection, no oracle in any training/feature/blend path (post-freeze scoring only).
- Residual correction must use OOF residuals from grouped folds (never in-sample).
- Fixed seeds, standalone single-run notebook, grouped folds (457 shared train/test SMILES).
- Small-target promotions require fold-level evidence, not just oracle deltas (SE ~0.02-0.05).

## 6. Files
- output/09_error_concentration.csv/.txt — concentration + clipping + invariance numbers
- output/10_physics_identities.txt — identity fits on train
- scripts/09_error_concentration.py, scripts/10_physics_identities.py — rerunnable
