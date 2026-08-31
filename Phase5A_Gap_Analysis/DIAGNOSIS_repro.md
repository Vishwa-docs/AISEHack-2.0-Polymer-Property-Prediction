# DIAGNOSIS — why results look worse (2026-09-01)

## The smoking gun: the Mac environment breaks ei/eea in the V57 reproduction

P5A-100 is the UNTOUCHED V57 standalone. It scored 0.8454 on the Mac, not 0.9023.
Per-target diff between the frozen submission.csv (laptop env, 0.9023) and the fresh
P5A-100 run (Mac env):

| target | mean|frozen-fresh| | corr | frozen R2 | fresh R2 |
|--------|-------------------:|-----:|----------:|---------:|
| tg  | 0.0038 | 1.0000 | 0.895 | 0.895 | (reproduces) |
| egc | 0.0001 | 1.0000 | 0.911 | 0.911 | (reproduces) |
| egb | 0.0007 | 1.0000 | 0.927 | 0.927 | (reproduces) |
| nc  | 0.0000 | 1.0000 | 0.909 | 0.908 | (reproduces) |
| eps | 0.0003 | 1.0000 | 0.885 | 0.884 | (reproduces) |
| **ei** | **0.3485** | **0.8502** | **0.871** | **0.512** | (BROKEN) |
| **eea** | **0.1737** | **0.9738** | **0.918** | **0.880** | (diverged) |

Cause: the standalone's ei/eea path uses version-sensitive models -
MLPRegressor (x2), rdEHT (x3), GaussianProcessRegressor (x1), RDKit Descriptors3D (x23).
Mac: sklearn 1.4.0 / rdkit 2026.03.5. Laptop (where V57 was validated): sklearn 1.9.0 /
rdkit 2026.3.4. Different versions -> different ei/eea leaf models -> the whole mean drops.

## Consequences for the results so far
- P5A-100 (yardstick) 0.8454 and P5A-101 (kriging ~alpha=0) 0.8455 are NOT meaningful for ei/eea.
- P5A-102 (calib) 0.8326 fitted calibration on the broken ei OOF -> amplified damage. Ignore.
- tg/egc/egb/nc/eps paths reproduce EXACTLY (corr 1.0): tg-focused arms are still meaningful.
- Core series (P5A-000..008) is a separate weak floor (~0.83) - informational only.

## The fix
1. Run the V57 series on the GPU laptop (its exact validated env):
   SSH_PASS='kumaresh@123' bash run_final.sh P5A-100    # yardstick first
   Expect ~0.9023 if env matches; then continue the queue on GPU.
2. Do NOT judge ei/eea arms (P5A-103/104/105/121/122/118/120) from Mac runs.
3. tg arms (P5A-101 kriging alpha~0, 108 char tune, 111 huber, 112 MAE, 116, 119, 123)
   are valid to evaluate on any machine (tg path reproduces).
4. Frozen submission.csv (0.9023, est private 0.891) remains the incumbent submission pair.
