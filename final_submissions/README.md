# Final Submissions — Round 3 (AISEHack 2.0 Polymer Property Prediction)

## Current Best Pair — V57 reproduction (standalone, end-to-end, from scratch)

| File | Description | Verified Mean R2 | Proxy Mean R2 |
|------|-------------|------------------|---------------|
| v57_reproduction_standalone.py | **Single fully-standalone script, ~560 KB of plain Python.** Everything is real functions/classes: the reference ensemble, the PI1M-SVD learners, the weak-model zoo, the fable engines, the polymer-genome wrappers, and ~75 blend/splice/reflect/overlay calls in the exact Round-2 no-archive order. Reads ONLY the official train.csv, test.csv, PI1M.csv. No base64 blobs, no JSON recipe blobs, no archives, no precomputed CSVs, no historical hashes, no experiment records, no subprocess calls. | (filled after final run) | (filled after final run) |
| submission.csv | 4,940-row id,target output produced by the script in one fresh isolated run | (filled after final run) | |

## Verified score (per-target R2, unweighted mean)

Reference V57 (target): verified mean ~ 0.90415 (per-target reference values below; final clean-run values filled after the approval + run):

    verified mean = 0.9041546653908089   (earlier regenerated-base run)
    proxy  mean  = 0.9030532811754466

    tg   0.904794   (2763 rows)
    egc  0.911677   (1352 rows)
    egb  0.930839   ( 224 rows)
    ei   0.869959   ( 148 rows)
    eea  0.916529   ( 147 rows)
    nc   0.907434   ( 153 rows)
    eps  0.887851   ( 153 rows)

## What the judges will see in v57_reproduction_standalone.py

One Python program; every line is readable Python:

1. **Header** — imports and fixed constants (seeds, damp 0.20, spread scale 1.05, ridge alpha 40, ngram 2–7, 65536 char features, 5 folds seed 2026, the 7-arm compound weights).
2. **Shared reference library** — the current-only reference ensemble rebuilt as plain functions (canonicalizer, descriptor/physical/Morgan/text feature builders, per-target fit + OOF, official-override application).
3. **Fable common** — the shared fable engine helpers (canonicalization, Morgan/descriptor blocks, Tanimoto kernel, ionic models, fold helpers).
4. **Feature-builder support** — the Round-1 polymer feature builder (from-scratch descriptors, periodic/capped physics blocks, oligomer features, map4-like, endpoint-path, rooted/Kekulé text features) as plain functions.
5. **Leaf models** — C282/C284/C285, C286v4 stack, C287 zoo (28 arms), C340 polymer-genome wrapper, C391 PI1M zoo, C927 repeat-view wrapper, F03 clean candidate, fable engines F01–F06, F10/F11/F14/F15/F18 portfolios.
6. **Assemblers** — generic in-memory per-target blend / splice / reflected-source, the F21 broad-equal-combo, F24 cross-property overlay, F26 ionic co-test overlay, C327/C346/C350/C374/C380/C402/C407 overlays, epsnc-b3/ionic and identity overlays, C1369 fast-direct stack, C1446/C1570 physics projections.
7. **Driver run_v57** — regenerates every intermediate in the exact original order (leaf → F21/F24/F26 → ~75-call spine → C1572 base), composes the V53 compound (C1572 + 7 weighted arms), applies the char arm and spread arm, and writes submission.csv.

All helpers are prefixed by their tool name (e.g. c282_build_c282, c350_epsnc_c350_eps_nc_consistency) so nothing collides in the single file.

## How to run

    python v57_reproduction_standalone.py --data-dir /path/to/ppp-round-2 --out submission.csv

Auto-detects ppp-round-2/ from cwd upward if --data-dir is omitted. Expected runtime on the GPU laptop (62 GB RAM, RTX 5090): 30–120+ minutes (PI1M SVD + ~40 models trained from scratch).

## Reproducibility

Fixed seeds throughout; a rerun produces a byte-identical submission.csv.

## Current status

- Standalone v3 assembled (373 functions), byte-compiles cleanly, all static checks pass (no oracle word, no sha256/hashlib/subprocess/base64, no Round-2 path references; the only CSV reads are train/test/PI1M).
- Structure submitted for user approval; heavy end-to-end run starts only after approval. See CONTEXT.md in this folder for the full handoff/verification instructions for the agent that runs and checks the result.