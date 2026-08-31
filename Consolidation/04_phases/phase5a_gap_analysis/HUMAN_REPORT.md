# HUMAN_REPORT - Phase5A Gap Analysis: where the score is, and the numbers to 0.935

Metric is **settled**: the official Kaggle page states "mean coefficient of determination R2
across the seven targets" = **unweighted mean of the 7 per-target R2**. Verified empirically:
the same frozen V57 file scores **0.9023** as per-target mean (-> private LB 0.891 via the
documented -0.011 calibration) but **0.9370** if rows were pooled into one array - a pooled
metric would have given V57 a LB ~ 0.937, beating the 0.92 competitor. Pooling is ruled out.
**Consequence: every target is worth exactly 1/7 of the score. +0.01 R2 on ANY target =
+0.00143 on the mean. Tg's 56% row share buys it nothing extra.**

---

## 1. Dataset variance (train.csv) - your numbers all check out

| target | train n | test n | mean | std | TSS (=n*var) | % of total TSS |
|--------|--------:|-------:|-----:|----:|-------------:|---------------:|
| tg | 4,143 | 2,763 | 143.46 | 109.08 | 49,286,633 | **99.99%** |
| egc | 2,028 | 1,352 | 4.529 | 1.568 | 4,985 | 0.010% |
| egb | 337 | 224 | 4.276 | 1.979 | 1,315 | 0.003% |
| ei | 222 | 148 | 6.346 | 1.047 | 242 | 0.0005% |
| eea | 221 | 147 | 2.278 | 1.107 | 270 | 0.0005% |
| nc | 229 | 153 | 1.934 | 0.235 | 12.6 | 0.00003% |
| eps | 229 | 153 | 4.577 | 1.094 | 273 | 0.0006% |

- Your pasted numbers verified: tg std 109.1 OK, ei std 1.04 OK, nc std 0.23 OK,
  Tg = 55.93% of test rows OK, "~7,000 SSE" for the 6 small targets OK (actual 7,098),
  "~84M pooled TSS" OK (actual 84.7M on train).
- The one trap: **tg = 99.99% of all within-target TSS**. Any pooled/MSE-based multi-target
  training (one loss over all rows, unnormalized) is silently a Tg model - the small targets
  get ~0 weight in the loss. Train per-target or normalize targets before joint training.

## 2. Where we are (V57, final_oracle panel, 4,909/4,940 rows)

| target | n | **R2** | SE(R2) | RMSE | MAE | SSE | rank |
|--------|--:|-------:|-------:|-----:|----:|----:|-----:|
| tg | 2,732 | **0.8953** | 0.007 | 35.33 | 22.97 | 3,409,962 | 5th |
| egc | 1,352 | **0.9111** | 0.006 | 0.464 | 0.317 | 290.8 | 4th |
| egb | 224 | **0.9268** | 0.012 | 0.518 | 0.375 | 60.1 | 1st |
| ei | 148 | **0.8711** | 0.022 | 0.319 | 0.224 | 15.1 | 7th |
| eea | 147 | **0.9183** | 0.014 | 0.303 | 0.226 | 13.5 | 2nd |
| nc | 153 | **0.9086** | 0.020 | 0.074 | 0.051 | 0.85 | 5th |
| eps | 153 | **0.8847** | 0.024 | 0.393 | 0.273 | 23.6 | 6th |
| **mean** | 4,909 | **0.9023** | | | | | -> private ~ 0.891 |

Weakest -> strongest: **ei (0.871) < eps (0.885) < tg (0.895) < nc (0.909) < egc (0.911) < eea (0.918) < egb (0.927)**.

## 3. The gap to 0.935

Need **+0.0327 mean = +0.229 total R2** spread over 7 targets (0.935 - 0.9023).
Equal spread = +0.033 on every target. One exact profile that lands on 0.9350 (est. private ~ 0.924):

| target | R2 now | R2 needed | dR2 | RMSE now | RMSE needed | dRMSE |
|--------|-------:|----------:|----:|---------:|------------:|------:|
| tg | 0.8953 | 0.935 | +0.040 | 35.33 | 27.84 | **-7.5 C** |
| egc | 0.9111 | 0.935 | +0.024 | 0.464 | 0.397 | -0.067 eV |
| egb | 0.9268 | 0.955 | +0.028 | 0.518 | 0.406 | -0.112 eV |
| ei | 0.8711 | 0.920 | +0.049 | 0.319 | 0.252 | -0.068 eV |
| eea | 0.9183 | 0.945 | +0.027 | 0.303 | 0.249 | -0.054 eV |
| nc | 0.9086 | 0.935 | +0.026 | 0.074 | 0.063 | -0.012 |
| eps | 0.8847 | 0.920 | +0.035 | 0.393 | 0.327 | -0.066 |

Reality ladder (mean -> est. private = mean - 0.011):

| scenario | tg | egc | egb | ei | eea | nc | eps | mean | est. private |
|----------|----|----|----|----|----|----|----|-----:|-------------:|
| V57 now | .895 | .911 | .927 | .871 | .918 | .909 | .885 | 0.9023 | 0.891 |
| S2 realistic | .920 | .922 | .940 | .905 | .928 | .920 | .910 | 0.9207 | 0.910 |
| S7 aggressive | .925 | .925 | .945 | .915 | .935 | .925 | .920 | 0.9271 | 0.916 |
| S8 ceiling | .930 | .930 | .950 | .920 | .940 | .930 | .925 | 0.9321 | 0.921 |
| 0.935 profile | .935 | .935 | .955 | .920 | .945 | .935 | .920 | 0.9350 | 0.924 |

**Bottom line:** beating the 0.92 competitor needs oracle >= 0.931 -> S8-level scores:
**tg >= 0.930, ei >= 0.920, eps >= 0.920 simultaneously** (plus ~+0.02 on egc/nc/eea/egb).
Tg alone cannot do it: even a *perfect* tg model (R2=1.0) only reaches mean 0.9172.
0.935 is the stretch goal; the realistic win condition is S7-S8 (private 0.916-0.921).

## 4. Focus ranking (where the points are)

| # | target | R2 now | headroom to ceiling* | R2 SE (reliability) | +0.01 R2 costs (RMSE) |
|---|--------|-------:|---------------------:|--------------------:|----------------------:|
| 1 | **tg** | 0.8953 | +0.025 (ceiling ~0.92, experimental noise) | 0.007 -> gains lock in | -1.73 C (35.33->33.60) |
| 2 | **ei** | 0.8711 | +0.06 (ceiling 0.92-0.94, DFT but starved) | 0.022 -> noisy | -0.012 eV |
| 3 | **eps** | 0.8847 | +0.05 (physics route available) | 0.024 -> noisy | -0.018 |
| 4 | egc | 0.9111 | +0.04 | 0.006 | -0.028 eV |
| 5 | nc | 0.9086 | +0.04 | 0.020 | -0.004 |
| 6 | eea | 0.9183 | +0.03 | 0.014 | -0.018 eV |
| 7 | egb | 0.9268 | +0.03 | 0.012 | -0.037 eV |

* ceilings are assumptions (tg: published PolyInfo ceiling ~ 0.92; DFT targets deterministic, data-limited).

- **Variance/fragility:** small-target R2 is noisy - SE(ei)=0.022, SE(eps)=0.024, SE(nc)=0.020,
  so oracle deltas < ~0.04 on these are NOT bankable; validate gains on folds. One bad row costs
  ei 1.6% / eps 1.1% / nc 1.1% of that target's TSS (fixing the single worst row: ei 0.871->0.884,
  eps 0.885->0.896, nc 0.909->0.919, tg only 0.895->0.898). Top-5% worst rows carry 37-55% of each
  target's SSE (tg 55%) -> outlier clipping / robust loss pays.
- **Structure that decides strategy:** cross-property partner labels exist in train for test rows:
  **ei 96%, eea 96%, nc 97%, eps 97%, egb 80%, egc 5%, tg 0.2%**. So ei=eea+egc, eps=nc^2+ionic
  only work for the DFT cluster; **tg is isolated** (only 8 of 2,763 tg test rows have their SMILES
  in train at all; exact retrieval ~ 0% everywhere -> similarity/features only). Tg label noise is
  real (3 conflicting train labels, e.g. 98.28 vs 105.00) -> hard ceiling ~0.92.

## 5. What to do (priority order, expected gains)

1. **tg -> 0.920** (+0.025 R2 = +0.0036 mean): best-effort Tg model + OOF residual correction on
   similar polymers + generalization-focused validation. Most reliable lever (SE 0.007).
2. **ei -> 0.900** (+0.029 = +0.0041): partner reconstruction (egc+eea identity) with guard.
3. **eps -> 0.910** (+0.025 = +0.0036): ionic decomposition eps = nc^2 + ionic.
4. **egc +0.014, eea +0.012, nc +0.011, egb +0.013** (each +0.0016-0.0020 mean).
   -> total ~ **0.9207 oracle (~0.910 private)**. S7/S8 need +0.02-0.04 more on tg/ei/eps.

Don't repeat: GPR/KRR (tried, hurt), small-scale SSL/pseudo-label (hurt), read-across routers
(hurt), pooled multi-task LGBM (neutral) - the untried lever is smile_r3/PI1M representation
learning **at full scale with strong GBM heads** (all prior probes were 50k-250k rows).

## 6. Oracle compliance (one line)

Everything above is post-freeze oracle diagnostics - the final submission notebook must never
read Oracle/ (per AGENTS.md); all final techniques must fit from scratch on official data only.

---
*Full tables: output/01_eda_summary.csv . 01_claims.json . 03_per_target.csv . 03_variants.csv .
03_tg_categories.csv . 02_leverage.csv . 02_scenarios.csv . 04_headroom.csv .
05_partner_availability.csv . 05_overlap_train/test.csv . 07_residual_structure.csv .
08_target_profiles.txt . figures fig_01..fig_04.*
