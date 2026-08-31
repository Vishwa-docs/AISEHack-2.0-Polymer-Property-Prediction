# STATUS_NEXT — where we are, and the real path to 0.935 (2026-09-01)

## 1. What the logs say (post-freeze scores vs Oracle/final_oracle.csv)

| exp | mean | tg | egc | egb | ei | eea | nc | eps | verdict |
|-----|-----:|----:|----:|----:|----:|----:|----:|----:|---------|
| V57 (final_submissions/submission.csv) | **0.9023** | .895 | .911 | .927 | .871 | .918 | .909 | .885 | incumbent |
| P5A-000 baseline-core | 0.8265 | .881 | .880 | .918 | .712 | .829 | .764 | .801 | floor |
| P5A-001 tg-robust | 0.8249 | .875 | .880 | .917 | .721 | .830 | .764 | .787 | -0.002 vs floor |
| P5A-002 tg-kriging | 0.8265 | .881 | .880 | .918 | .712 | .829 | .764 | .801 | alpha=0, no-op |
| P5A-003 smile_r3-SVD | 0.8032 | .876 | .876 | .899 | .633 | .799 | .754 | .786 | -0.023 (SVD on ALL targets hurt the small ones) |
| P5A-004..008 | — | not run | | | | | | | |

## 2. Verdict: NOT on track via the simple core — and that was expected

The P5A-000..008 core is a minimal from-scratch pipeline (Morgan + descriptors + char-grams +
partner features). It is **0.076 below V57** because V57 carries ~40 banked components (339-node
compound chain, char-residual arm, spread arm, EHT co-test, ionic/identity overlays, exact
overrides). Per-target gap to V57: ei -0.159, nc -0.145, eea -0.089, eps -0.084, egc -0.031,
tg -0.014, egb -0.009. The core was never a candidate — it was the testbed, and it confirmed:
- stage-2 cross-property partner features work (egb 0.918 is the core's best; egc stage-2 OOF 0.93 in smoke),
- kriging on a WEAK model's OOF has no value (alpha = 0.000) — only worth testing on V57's strong OOF,
- bolting the smile_r3 SVD block onto ALL targets hurts the small targets (ei collapsed to 0.633):
  representation features must be per-target (tg only) and kill-gated.

What IS on track: metric understanding, target priority (ei < eps < tg), the physics levers
(eps=nc^2+ionic removes 61.8% of variance; egb=1.10*egc-0.69, R2=0.86; partner availability
ei/eea/nc/eps 96-97%), and the fat-tail insight (top-1% of tg rows = 26.4% of tg SSE).

## 3. The real path: arms on top of the V57 standalone (P5A-100 series)

V57's final assembly (final_submissions/v57_reproduction_standalone.py lines ~9390-9403) is:
    final = base_target (C1572 spine)
    tg/egc/egb/nc/eps: final += 0.20 * char_delta   (Ridge on char n-grams of C282 OOF residual)
    ei/eea:            final = clip(median + 1.05*(base-median), lo, hi)   (spread arm)
That loop is THE injection point: add one new per-target arm per experiment, each with its
alpha fitted on the C282 OOF residuals (train-only, clean lane). The fragile 339-node chain is
never touched — arms are post-hoc, exactly like the existing char/spread arms.

Proposed series (each = copy of the standalone + ONE arm, compile-checked, post-freeze scored):
| exp | arm | mechanism | expected | gate |
|-----|-----|-----------|----------|------|
| P5A-100-v57-baseline | none (untouched copy) | verify the Mac's fresh run reproduces ~0.9023 (DEFECT-1/3 caveat: fresh runs differ slightly from the frozen CSV) | ~0.902-0.904 | yardstick |
| P5A-101-v57-tg-kriging | tg Tanimoto kNN residual arm | OOF residuals kriged to test via Morgan Tanimoto, alpha on OOF (closed form) | tg +0.005-0.010 | tg >= +0.005 or kill |
| P5A-102-v57-oof-calib | per-target linear calibration | slope/intercept + blend alpha per target on C282 OOF (fixes over-dispersion: egb 1.03, eps 1.05) | +0.002-0.005 mean | mean >= +0.002 |
| P5A-103-v57-ei-identity | ei arm | candidate = egc+eea partner predictions (Ridge on char n-grams, same machinery as the char arm), alpha on OOF | ei +0.01-0.02 | ei >= +0.010 |
| P5A-104-v57-eps-ionic | eps arm | candidate = nc^2 + ionic_med via partner nc predictions, alpha on OOF | eps +0.010-0.015 | eps >= +0.010 |
| P5A-105-v57-egb-egc | egb arm | candidate = a*egc + b (fit on OOF), alpha on OOF | egb +0.005 | egb >= +0.005 |
| P5A-106-v57-all-arms | combined | winners of 101-105 spliced | 0.910-0.920 mean | mean > 0.9023 + 0.005, no target < -0.003 |

## 4. Honest math on 0.935

- 0.9023 -> 0.935 needs +0.0327 mean = +0.229 total R2 (S8-level: tg ~0.93, ei ~0.92, eps ~0.92
  simultaneously, plus +0.02 on egc/nc/eea/egb). Even PERFECT tg only gives 0.9172.
- Realistic arm series: 0.91-0.92 oracle (~0.90-0.91 private). That already beats nothing on its
  own — it is the required base.
- The remaining jump (tg 0.93, ei 0.92) needs the smile_r3/PI1M representation bet done RIGHT:
  tg-only injection (NOT all targets — P5A-003's lesson), per-target kill gates, and validation
  on grouped folds. It is untried at scale; it is also the only genuinely new signal available.
- Verdict: 0.935 is a stretch goal, not a promise. The near-term win condition is oracle >= 0.931
  (private > 0.92, beats the competitor).

## 5. Next actions (all inside Phase5A_Gap_Analysis/)
1. Finish/collect the current queue (P5A-004..008 are informational only now).
2. Build P5A-100-v57-baseline: copy of final_submissions/v57_reproduction_standalone.py +
   thin wrapper; run on the Mac (or GPU laptop) — this is the yardstick for every arm.
3. Build arms 101-106 one at a time; each is a single patched copy; score post-freeze;
   promote only arms that pass their gate; splice winners into P5A-106.
4. Only after arms: re-attempt the tg representation bet (tg-only, kill-gated).
