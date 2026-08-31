# Phase5A_Gap_Analysis — metric + variance + gap analysis + experiment scaffold

## HOW TO RUN (two runners, both inside this folder)
    bash Phase5A_Gap_Analysis/run.sh            # P5A-000..008: simple-core testbed (floor ~0.83)
    bash Phase5A_Gap_Analysis/run_final.sh      # P5A-100..115: V57-standalone + arm experiments (THE real path)
    bash Phase5A_Gap_Analysis/run_final.sh P5A-115            # one experiment
    FORCE=1 bash .../run_final.sh               # re-run scored ones
    P5A_PYTHON=/path/python bash .../run_final.sh             # interpreter override (default python3)
    SSH_PASS=<pass> bash .../run_final.sh --gpu P5A-115       # run on GPU laptop (needs /tmp/r3_dataset there)

Each run: execute -> freeze (sha256) -> score vs Oracle/final_oracle.csv -> oracle_scores.json ->
logs/phase5a_final_summary.tsv -> printed table. Incumbent V57 = 0.9023, target 0.9350,
est private = mean - 0.011.

## FINAL series (P5A-100..115): V57 standalone + one arm each
All arms are injected at the final assembly (before the validity check), alpha fitted on the
C282 OOF residuals (train-only). The 339-node chain is never touched.
| exp | arm | lever |
|-----|-----|-------|
| P5A-100 | none | pristine V57 yardstick (~0.9023) |
| P5A-101 | tg kriging | Tanimoto kNN residual correction for tg (fat tail: top-1% = 26% of SSE) |
| P5A-102 | oof calib | per-target linear calibration of the base |
| P5A-103 | ei identity | ei = egc + eea via partner predictions + exact train partners |
| P5A-104 | eps ionic | eps = nc^2 + ionic (nc^2 removes 61.8% of eps variance) |
| P5A-105 | egb-egc | egb = a*egc + b overlay |
| P5A-106 | core arms | 101+102+103+104+105 + char alpha tune |
| P5A-107 | tg smile_r3 | tg residual model on smile_r3-fitted char-SVD(48), 400k sample |
| P5A-108 | char tune | char arm per-target alpha on OOF (replaces fixed 0.20) |
| P5A-109 | median shrink | per-target shrink toward train median |
| P5A-110 | spread tune | ei/eea spread scale fitted on OOF (replaces 1.05) |
| P5A-111 | tg char huber | tg char arm with Huber loss (fat-tail robust) |
| P5A-112 | tg MAE | MAE-optimal tg residual model (absolute_error HGB), MAE-tuned alpha |
| P5A-113 | weak augment | ei/eea/nc/eps residual models, x8 random-SMILES augmentation (train-only) |
| P5A-114 | weak impute | 103 + 104 + 113 package (identity imputation + augmentation) |
| P5A-115 | final combo | all arms combined (best-possible candidate) |
| P5A-116 | tg MAE package | MAE-tuned HGB + Huber GBM + Huber char + kriging (cuts Tg MAE) |
| P5A-117 | nc-eps exact | nc = sqrt(eps - ionic) via exact train eps partners (~97% availability) |
| P5A-118 | weak MAE | ei/eea/nc/eps MAE-tuned residual models |
| P5A-119 | tg GBM | tg Huber-GBM residual model on char n-grams |
| P5A-120 | nc-eps consistency | eps=nc^2+ionic AND nc=sqrt(eps-ionic), exact partners |
| P5A-121 | weak stacker | Ridge on [partner preds, base] per weak target (imputation cascade) |
| P5A-122 | weak kernel | Tanimoto KRR residual models for weak targets |
| P5A-123 | tg aug char | tg char residual model with x8 random-SMILES augmentation |
| P5A-124 | final combo v2 | ALL 17 arms combined (best-possible candidate v2) |

Wall time per run: ~1.5-4 h on Mac (PI1M SVD + ~40 models + arms); ~0.5-1.5 h on the GPU laptop.

## Analysis outputs (why these experiments)
- Metric verified: unweighted mean of 7 per-target R2 (pooled variant 0.9370 contradicts LB 0.891).
- V57 state: 0.9023 (tg .895, egc .911, egb .927, ei .871, eea .918, nc .909, eps .885).
- Gap to 0.935 = +0.033 mean; needs near-simultaneous SOTA on tg/ei/eps; honest ladder 0.91-0.93.
- Full tables: output/01..10 + STATUS_NEXT.md + HUMAN_REPORT.md + STRATEGY_NEXT.md.
