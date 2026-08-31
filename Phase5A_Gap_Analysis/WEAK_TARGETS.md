# WEAK TARGETS — where we are lacking and how to get there (oracle-analyzed, frozen V57)

## 1. The numbers (frozen V57 vs final_oracle, 4,909 rows)

| target | R2 | MAE | RMSE | slope | value bins (n, R2, MAE) | similarity bins (n, R2, MAE) |
|--------|-----|-----|------|-------|--------------------------|-------------------------------|
| ei  | 0.8711 | 0.224 | 0.319 | 1.00 | lo (50,.456,.239) mid (49,**-.214**,.163) hi (49,.552,.269) | lo .890 / mid .859 / hi .852 |
| eea | 0.9183 | 0.226 | 0.303 | 1.00 | lo .425 / mid **-.042** / hi .618 | lo .896 / mid .914 / hi .940 |
| nc  | 0.9086 | 0.051 | 0.074 | 1.00 | lo .570 / mid **-.935** / hi .815 | lo .910 / mid .878 / hi .945 |
| eps | 0.8847 | 0.273 | 0.393 | **1.047** | lo .507 / mid **-1.80** / hi .766 | lo .819 / mid .876 / hi .920 |
| tg  | 0.8953 | 22.97 | 35.33 | 1.00 | lo .514 / mid -.106 / hi .373 | flat .892-.898 |

## 2. What this tells us (the diagnosis)

1. **The mid-value band is the failure mode everywhere** (ei mid R2=-0.21, nc mid -0.94, eps mid -1.80):
   models compress mid-range predictions toward the mean -> zero discrimination there.
2. **eps is over-dispersed and heteroscedastic** (slope 1.047; MAE grows 0.16 -> 0.36 with value).
   Same, milder, for nc (MAE 0.039 -> 0.063). => value-dependent corrections, NOT global tweaks.
3. **Weak-target errors are NOT mainly OOD-driven** (similarity bins are flat-to-mild, e.g. ei .890/.859/.852,
   eps .819/.876/.920). The exact-partner lookup route is therefore available for ~96-97% of rows.
4. tg: errors are a fat tail of individual hard rows + mid-band compression; similarity-flat.

## 3. The right tools for N=148-229 (trees alone saturate — this is what works)

| lever | why | coverage | arms |
|-------|-----|----------|------|
| **Exact partner labels from train + physics identity** | for nc/eps/ei rows, the same polymer's other property is IN TRAIN (96-97%) -> near-deterministic constraint (nc=sqrt(eps-ionic), eps=nc^2+ionic, ei=egc+eea) | ei 96%, eea 96%, nc 97%, eps 97% | 103 (ei), 104 (eps), 117 (nc exact), 120 (both) |
| **Kernel/similarity models** | Tanimoto KRR beats trees at N<300 (literature + Round-2 evidence) | all weak | 122 |
| **Stacking with partner preds + base** | learns the value-dependent correction directly | all weak | 121 |
| **Value-aware calibration** | fixes mid-band compression + eps slope 1.047 | all | 102 (linear), 125 (base+base^2) |
| **MAE-tuned alphas** | user goal: cut MAE, not just R2 | all weak + tg | 118, 112, 116 |
| **SMILES augmentation** | more char-ngram diversity at fixed N (train-only) | weak + tg | 113, 123 |

## 4. What each weak target needs (RMSE budgets)

| target | now | 0.92 needs | 0.93 needs |
|--------|-----|------------|------------|
| ei  | 0.319 RMSE / 0.224 MAE | 0.251 / 0.176 | 0.235 / 0.165 |
| eea | 0.303 / 0.226 | 0.300 / 0.224 | 0.280 / 0.209 |
| nc  | 0.074 / 0.051 | 0.070 / 0.048 | 0.065 / 0.045 |
| eps | 0.393 / 0.273 | 0.327 / 0.227 | 0.306 / 0.213 |

(eea needs the LEAST - it is 0.003 RMSE from 0.92. nc needs -0.004. ei and eps need -0.07 RMSE each.)

## 5. Priority run order (ALL on the GPU laptop - Mac env breaks ei/eea, see DIAGNOSIS_repro.md)

1. P5A-100  re-baseline gate (one run, ~0.9023 expected on laptop env)
2. P5A-117  nc via exact eps partner (97% coverage, near-deterministic)  [nc 0.909 -> 0.92+]
3. P5A-120  nc-eps consistency both directions                          [nc+eps]
4. P5A-103  ei via egc+eea identity (exact partners overlaid)           [ei 0.871 -> 0.89-0.91]
5. P5A-104  eps via nc^2+ionic (exact partners overlaid)                [eps 0.885 -> 0.90-0.91]
6. P5A-121  weak stacker (value-dependent correction)                   [all weak]
7. P5A-122  Tanimoto KRR per weak target                                [all weak]
8. P5A-125  value calibration (base+base^2)                             [all targets]
9. P5A-102  linear calibration                                          [all targets]
10. P5A-118  MAE-tuned alphas for weak targets                          [MAE down]
11. P5A-114  ei+eps identity + augmentation package                     [ei+eps]
12. P5A-107  tg smile_r3 arm (tg-only; all-target SVD already hurt - P5A-003)  [tg]
13. P5A-112/116/119/123  tg MAE/GBM/aug arms                            [tg MAE 22.97 down]
14. P5A-124  ALL arms final candidate                                   [everything]

## 6. Expected outcome
- eea to ~0.93 (cheapest), nc to ~0.92 (exact partners), eps to ~0.91-0.92 (ionic + calib),
  ei to ~0.90-0.91 (identity + stack + kernel) => weak-target mean +0.03-0.045 over the 4 targets
  = +0.017-0.026 on the overall mean. tg arms add +0.005-0.015. Total: 0.9023 -> 0.925-0.943 possible
  IF the arms bank; realistic midpoint ~0.92-0.93 oracle (private ~0.91-0.92).
- smil_r3 verdict pending P5A-107 (tg-only design); all-target SVD is dead (P5A-003 evidence).
