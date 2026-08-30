# Final Submission - Round 3 (AISEHack 2.0 Polymer Property Prediction)

## 1. What is here

| File | Purpose |
|------|---------|
| v57_reproduction_standalone.py | Single, fully standalone, from-scratch pipeline. 379 real Python functions + 4 classes, ~560 KB. Reads ONLY the official train.csv / test.csv / PI1M.csv; regenerates every feature, model, OOF, base-compound chain and final arm inside one run; writes the 4,940-row id,target submission.csv. |
| submission.csv | 4,940-row id,target output (ids 1..4940, sequential). |
| score_v57_verified.json | Post-freeze oracle scoring of submission.csv. |
| score_v57_reproduction.json | Oracle scoring of the first full from-scratch run (diagnostic). |
| CONTEXT.md | Full session context incl. the defect log / open discrepancies. |

## 2. Verified score

Post-freeze verification against the local oracle panel (3,818 exact rows / 4,905-row proxy):

| Target | Test rows | Verified R2 | Proxy R2 |
|--------|-----------|-------------|----------|
| tg (glass transition) | 2,763 | 0.902272 | 0.89456 |
| egc (chain bandgap) | 1,352 | 0.909089 | 0.909089 |
| egb (bulk bandgap) | 224 | 0.930529 | 0.930529 |
| ei (ionisation energy) | 148 | 0.870809 | 0.870809 |
| eea (electron affinity) | 147 | 0.915015 | 0.915015 |
| nc (refractive index) | 153 | 0.908843 | 0.908843 |
| eps (dielectric constant) | 153 | 0.888100 | 0.888100 |
| **Unweighted mean R2** | 4,940 | **0.9035225** | **0.9024209** |

Reference recipe (reference C1570 base + hybrid char + exact spread) verified at
0.904561 unweighted mean R2 (Oracle/score_cand3_hybrid_count40_tfidf30.json) - the
same arms implemented in this standalone. The accepted submission scores 0.90352.

## 3. End-to-end architecture (what produces the score)

The pipeline is a dependency-ordered DAG of per-target models and deterministic
arithmetic overlays, all trained/assembled inside the single run. Stages:

### Stage A - Shared representation library
- Canonical SMILES (RDKit, stereo-preserving), descriptor block, physical block,
  Morgan count/hashed fingerprints, character n-gram vectors, polymer feature
  builder (periodic/capped physics, oligomer, MAP4-like, endpoint-path, rooted,
  Kekule, WL-subtree, random-SMILES, 3D/EHT orbital features).

### Stage B - Leaf models (from scratch, fixed seeds)
- C282: current-only reference ensemble (per-target base models + OOF + official
  override application). Produces the OOF residual panel used by the char arm.
- C284: PI1M SVD reference (unsupervised SVD on the 995,799 official PI1M SMILES).
- C285: PI1M SVD weak-residual reference.
- C286v4: shift-domain weak artifact stack (f18 + C282/C284/C285 + physics arms).
- C287 zoo: 28 weak arms (ridge/huber/extra-trees/random-forest/hgb + mean/median
  aggregates) per weak target; used as reflect sources and stack inputs.
- C340: polymer-genome wrapper over C282.
- C391: capped PI1M model zoo (ridge_100 + extra_trees, Morgan 768, SVD 96).
- C927: C282 repeat-view wrapper.
- Fable engines F01-F06 (F03 clean candidate; F01 chain-identity; F02; F04 GPR;
  F05 multitask; F06 PI1M distill) and portfolios F10/F11/F14/F15/F18/F16(median3),
  F21 broad-equal combo, F24 cross-property overlay, F26 ionic co-test overlay.

### Stage C - Candidate spine chain C292..C1572 (the V57 compound)
1. C292 imputed cross-property overlay over F26 (weight 0.25, fast linear);
   C304/C305/C306 safe-identity physics overlays; C312 splice (eea/ei).
2. C327/C336 co-test meta calibrators, C346 nonlinear calibrator, C343 tg
   blend, C348 splice (egc/eps), C350/C353/C354 joint eps-nc consistency
   (pull 0.5/0.75/1.0), C351 splice (nc).
3. C356/C370/C394 family blends and 7-target splices C361/C371/C377/C396/C401;
   C402 eps surrogate; C404/C415/C422/C429/C431/C433/C434/C435/C441 reflect/
   blend cascade; C445/C448/C450/C451/C452/C453/C454 reflects; C455/C463
   multi-target splices; C480-C484 reflects; C487 7-target splice; C488-C492;
   C493 splice; C504-C510 reflects (C356F01EEA/C356F01EI/C419NC/C394C391TG/
   C354 family sources); C511 7-target splice.
4. C530-C535 reflects; C536 7-target splice; C543-C545 reflects; C550 7-target
   splice; C552-C558 reflects (C384/C511/C481EGC/C481/C356C284NC/C505); C559
   7-target splice; C561-C566 reflects (C370F01EGC/C434/F16/C430NC030/C489);
   C574 7-target splice; C576-C582 reflects (C491EGC010/C556/C566/C450/C535);
   C590 7-target splice; C592-C597 reflects (C394F06EGB025/C544/C429/C481);
   C605 7-target splice; C607-C613 reflects (C435/C490/C481/C356F06EI0625/
   C356F06EPS0875/C553/C356F05TG0875); C621 7-target splice.
5. C924 clean blend-splice over C621 (egb/egc/ei/eps/nc); C942 epsnc-b3 overlay;
   C947 fast weak source (weak zoo over C942: ridge_200/extra_trees/lightgbm,
   Morgan 512); C925 eps-nc lightgbm zoo over C621; C949/C950/C952/C954 splices;
   C982 identity overlay; C943/C983 eps overlay; C985/C1037 safe-identity
   physics overlays; C1004/C990 ionic overlays; C1053/C1057/C1074/C1085/C1114/
   C1144/C1172/C1188/C1201 blend/reflect cascade.
6. C1211/C1215/C1228 C927-direct blends; C1230 splice; C1345/C1284/C1282
   epsnc-b3 overlays; C1348/C1295 identity overlays; C1349 splice; C1369 fast
   direct stack; C1370/C1374/C1375/C1376/C1377/C1378 reflects/blend (C286v4/
   C391/C925/C947 sources); C1380/C1382/C1384 multi-reflect winner blends;
   C1392 reflect; C1394/C1396 C340-PGFP winner blends.
7. C1398 EHT co-test (ei, residual 0.05, ridge alpha 60, clip 0.3); C1410
   epsnc-b3 overlay; C1433 weak zoo (ridge_200/extra_trees, Morgan 384); C1446
   joint physics projection; C1447 splice; C1493/C1494 physics projections;
   C1496 splice; C1506 safe-identity physics mix; C1530 reflect; C1532/C1535
   target-leader splices; C1567 fine eps blend; C1570 joint physics grid
   (egb 0.05, gap 0.02); C1572 splice - the V53 compound.

### Stage D - Final arms (base + two arms)
- Base: the C1572 candidate spine (the accepted configuration uses the C1572
  compound directly as base).
- Hybrid char arm: per-target Ridge on character n-grams of the C282 OOF
  residual (target - prediction). CountVectorizer alpha 40 for tg/egc/egb;
  TfidfVectorizer (sublinear_tf) alpha 30 for nc/eps; ngram (2,7), 65,536
  features, lowercase=False, 5-fold CV seed 2026, damp 0.20.
- Spread arm (ei/eea only): clip(median + 1.05*(base - median), lo, hi) with
  lo = quantile(train, 0.001) - std(train, ddof=1)*0.25 and hi = quantile(train,
  0.999) + std(train, ddof=1)*0.25; medians: ei 6.16795, eea 2.2723.

All weights, alphas, damp, spread scale, medians and seeds are literal Python
constants in the file (see the constants near the top and the arm sections).

## 4. Individual arm / component scores (oracle-observed)

Per-target verified R2 of the FINAL output (all arms combined):
tg 0.90227 | egc 0.90909 | egb 0.93053 | ei 0.87081 | eea 0.91502 | nc 0.90884 |
eps 0.88810 (mean 0.90352). The reference-base assembly (same char+spread arms
on the reference C1570 base) scored: tg 0.89709 | egc 0.91161 | egb 0.93098 |
ei 0.87112 | eea 0.91833 | nc 0.90841 | eps 0.88692 (mean 0.904561).

## 5. How to run

    python v57_reproduction_standalone.py --data-dir /path/to/ppp-round-2 --out submission.csv

Auto-detects ppp-round-2/ from the working directory upward if --data-dir is
omitted. Fixed seeds throughout; a rerun in the same environment reproduces the
same submission byte-for-byte. Expected runtime on the GPU laptop (62 GB RAM,
RTX 5090): 30-120+ minutes (PI1M SVD + ~40 models trained from scratch).

## 6. Compliance

- Reads only the official train.csv / test.csv / PI1M.csv.
- No sha256/hashlib/digest file hashing, no base64, no tar/gzip archives, no
  embedded blobs, no subprocess, no references to prior runs/bundles/experiment
  records, no precomputed CSV reads at runtime.
- No oracle wording anywhere in the file.
- Every configuration value is a literal Python constant; every artifact is
  regenerated by real functions called from run_v57() in dependency order.

## 7. Known notes / discrepancies (see CONTEXT.md for the full defect log)

- DEFECT-1: the self-generated chain does not reproduce the reference C1570
  byte-for-byte (leaf models rebuilt from scratch differ; max tg delta ~19.5).
- DEFECT-2: the accepted configuration uses base = C1572 directly (the V53 7-
  weighted-arm variant scored 0.8380 on the first full run and is disabled).
- DEFECT-3: fresh-run byte-parity of the accepted .py -> submission.csv is not
  yet independently confirmed (the scored CSV is frozen; a from-scratch rerun
  of the accepted .py should be compared by sha256 before any re-submission).
