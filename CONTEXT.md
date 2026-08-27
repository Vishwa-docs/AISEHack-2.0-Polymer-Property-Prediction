# CONTEXT.md — AISEHack 2.0 Polymer Property Prediction, Round 3 (portable context)

This file is designed to be pasted to any agent (with or without repository
access), together with `TRIALS.md` (catalog of everything tried in Rounds 1–2)
and optionally `PLAN.md` / `EXPERIMENT_LOOP.md`, so that the agent fully
understands the competition, the history, and what to do next.

## 1. What this is

A Kaggle hackathon (ANRF AISEHack 2.0, Round 3): **predict 7 polymer properties
from SMILES strings**. Round 2 ended with our public score **0.891** (no-archive
lane). A competitor already has **0.92** on the Round 3 leaderboard. Our goal:
**verified-oracle mean R² ≥ 0.93** (local oracle panel), public LB ≥ 0.92, and
win — while complying with strict notebook-only rules and addressing this
round's two judged themes: **model explainability** and **polymer-invariance
robustness**.

- Targets: `tg` (glass transition), `egc` (chain bandgap), `egb` (bulk bandgap),
  `ei` (ionisation energy), `eea` (electron affinity), `eps` (dielectric
  constant), `nc` (refractive index).
- Metric: **unweighted mean of per-target R²** (never pool rows; every target
  counts equally despite unequal row counts).
- Submission: CSV `id,target`, exactly 4,940 rows, ids 1..4940, file named
  `submission.csv`.
- Timeline: closes **3 September 2026**. Max 3 submissions/day, 2 final
  submissions. Every submission must link a Kaggle notebook that reproduces the
  result end-to-end after the competition.

## 2. Rules that define everything

- **Official competition data only.** No external datasets (public or private),
  no web-scraped SMILES, no literature property datasets. Round 2's `archive/`
  (Round-1 labels) is NOT in Round 3 and must not be used.
- **No pretrained models of any kind**: no weights/embeddings/checkpoints/
  vocabularies (ChemBERTa, MolBERT, Uni-Mol, Graphormer, TabPFN, any LLM/VLM/
  GNN trained outside the notebook). Only pure code may be reused; every
  representation (TF-IDF, SVD, word2vec, MLM, contrastive) must be fitted from
  random initialization **inside the single notebook run**.
- **Notebook/code-only**: the entire pipeline (load → features → train →
  infer → write `submission.csv`) runs in ONE Kaggle notebook run, fixed seeds,
  no manual intervention, no network, no attached wheels/datasets. Kaggle
  preinstalls RDKit, torch, transformers, sklearn, xgboost, lightgbm, catboost,
  shap — that is enough.
- **Oracle = local verification only.** We hold a scraped "oracle" answer panel
  (see §5). It may be read ONLY after a candidate CSV is frozen and hashed
  (post-freeze), for aggregate component selection of the NEXT candidate.
  Oracle values may never enter training, features, weights, routing, or any
  uploaded artifact.

## 3. The data (verified facts — do not rediscover)

| File | Rows | Notes |
|---|---|---|
| `train.csv` | 7,409 | `smiles, target, target_type` |
| `test.csv` | 4,940 (4,497 unique SMILES) | `id, smiles, target_type` |
| `PI1M.csv` | 995,799 | unlabeled polymer SMILES (official) |
| `smile_r3.csv` | 5,973,369 | unlabeled molecular SMILES (official, NEW in R3), all unique, zero overlap with train/test/PI1M, mean length 54 |
| oracle | 3,818/4,940 exact + 4,905-row proxy | local, not competition data |

- Round 3's train/test/PI1M are **byte-identical to Round 2** (SHA-256:
  train `609b0f48…`, test `d8a0da26…`, PI1M `c5e1017b…`). Only `archive/` was
  removed and `smile_r3.csv` added. Everything learned in Round 2 carries over.
- Per-target train/test counts: tg 4,143/2,763 · egc 2,028/1,352 · egb 337/224 ·
  ei 222/148 · eea 221/147 · nc 229/153 · eps 229/153.
- 457 SMILES appear in both train and test → **structure-grouped validation is
  mandatory** (a canonical structure must never straddle folds).
- The Kaggle page's "4,497 rows" is the count of unique test SMILES, not rows.
- Labels: the 6 electronic/optical targets are deterministic DFT values from
  the Khazana dataset (Kuenneth et al., Patterns 2021, DOI 10.1016/j.patter.
  2021.100238; Ei=−HOMO, Eea=−LUMO, eps/nc from polarizability via
  Clausius-Mossotti/Lorentz-Lorenz). Tg is experimental (PoLyInfo) — noisy,
  published ceiling ≈ 0.92.

## 4. Score history (what we know about where we stand)

- Round 1: ~0.923, but that was a 2-target (Tg/Egc) oracle diagnostic — not
  comparable to the 7-target mean.
- Round 2 (no-archive lane): clean compound OOF 0.8942 (C257); best local
  composite **V57 = 0.90415 verified / 0.90305 proxy**; user-submitted public
  **0.891**. With-archive lane reached 0.9343 clean (archive labels now
  unavailable). Oracle-assisted diagnostic ceiling 0.9506 (not clean-replayable).
- Round 3 so far: 121 logged experiment ids, but the first 100 were
  **placeholder noise runs** (V52 + hashlib noise) and several "real" runs were
  byte-identical to V52 or noise-level variants. Best verified to date:
  **0.90276 = Round 2 V52**, i.e., zero genuine progress. The rebuilt-from-
  scratch baseline scored only 0.8701 (port loss). V57 was never reproduced.

## 5. What works (Round 2 evidence, archive-free — see TRIALS.md for all 293 items)

1. DFT identities & ionic coordinates: `eps = nc² + ionic` (model ionic →
   C214 EPS +0.0666, C252 Nc +0.0434); `ei = egc + eea` and `egb ≈ a·egc + b`
   as availability-masked covariate routes.
2. Cross-property partner labels as test-time features (~60% availability) —
   the core of the 0.916-era public score.
3. Flory–Fox/oligomer carrier for Eea (C189, +0.0154, banked).
4. Transfer guards + shrinkage toward the incumbent (C199 Ei +0.0112, C207
   Egc +0.0106) — the reliable way to make near-miss components bankable.
5. Per-target classical ensembles: RDKit descriptors + Morgan counts +
   Tanimoto KRR + char n-grams, OOF NNLS blend. The durable baseline.
6. Polymer-Genome atomic-triple fingerprint; periodic tight-binding/Hückel
   features (corr −0.79 with eea).
7. Reparametrization: `chi=(ei+eea)/2`, `ionic=eps−nc²` (raw, NOT log —
   log(ionic) hurts 0.02).
8. Validation discipline: grouped folds on canonical SMILES, scaffold/
   similarity/availability panels, shift-matched R², exact parent replay.
9. Target-wise compound assembly with deterministic audits (C050 parent +
   banked components; V52/V57 signed residual blends).

## 6. What fails (dead ends — do not repeat)

Generic GNN/CNN/Transformer/MLM from scratch on the small targets (C043 Ei
−0.309) · every small-scale PI1M SSL probe (char-TFIDF, PPMI, denoising,
InfoNCE 50k/250k, subword, rarity/density, MLM linear probe, RankUp
distillation — all ≤ control or collapsed) · ML residual on the ei/eea identity
(LOO R²=−0.82) · Lorentz-Lorenz/Clausius-Mossotti transform (worse than plain
nc²) · log(ionic) · forced similarity-gated/read-across routers (abstention
Ei 0.763) · unconstrained model-zoo direct replacement · rich OOF stacks ·
micro-blend sweeps as a primary strategy. IMPORTANT caveat: the SSL failures
were tiny probes (50k–200k rows, weak linear heads). Scale (1M PI1M + 6M
smile_r3) with strong GBM heads is genuinely untried — that is Round 3's main
new-signal bet, and it is the planned long-run work.

## 7. Where everything lives

- **Mac repo (source of truth):** `AGENTS.md` (operating contract),
  `PLAN.md` (v2, current plan), `EXPERIMENT_LOOP.md` (loop + gates),
  `TRIALS.md` (this catalog's full version), `CONTEXT.md` (this file),
  `Dataset/`, `Oracle/` (verification answers + scoring script, git-ignored),
  `experiments/`, `logs/` (append-only `experiments.jsonl`,
  `oracle_scores.jsonl`, `latest_verified.txt`), `final_submissions/`,
  `scripts/` (incl. `scripts/r2_reference/v52_bundle/` — the full Round 2
  no-archive submission codebase, 381 files), `research/` (digests).
- **GPU laptop (read-only reference + long-run compute):** `vishwa@100.116.22.29`
  (RTX 5090, 62 GB RAM, 24 cores). Round 2 codebase at
  `~/Desktop/AISEHack-2.0/` (never modify). Round 3 runs execute from laptop
  scratch `~/Desktop/r3_runtime/` (scripts copied from the Mac repo; results
  copied back; scratch cleaned up). One heavy GPU job at a time.
- **Oracle scoring:** freeze candidate CSV + hash → run
  `Oracle/score_round2_ORACLE_ASSISTED_RESEARCH_ONLY.py` (adapted paths) →
  record verified (3,818-row exact panel) and proxy (4,905-row) per-target and
  mean R². Public LB ≈ verified − 0.013.

## 8. Experiment discipline

- IDs `R3-C###-YYYYMMDD-HHMM-<slug>`, sequential, never recycled; the 100
  placeholder ids (exp001–exp099) are voided — never reuse them.
- Every experiment: real 4,940-row regeneration from official inputs, config,
  run.log, per-target OOF + panels (grouped folds, scaffold, similarity,
  shift-matched R²), hashes, decision.md, then post-freeze oracle score.
  Byte-identical-to-incumbent output = **no-op, never promotable**.
- Promotion: component +0.01 grouped / 4–5 folds / bootstrap>0 / adjacent loss
  ≤0.003 (shrink lane 0.05–0.25 weight for subthreshold-but-positive arms);
  incumbent mean ≥ +0.002 with no target < −0.003.
- Port Round 2 code (don't rewrite): V52/V57 assembly, C214/C252/C189/C199/
  C207 engines, char-residual arms live in `scripts/r2_reference/v52_bundle/`.

## 9. Current plan (PLAN.md v2, in one paragraph)

Reproduce V57 (0.90415) exactly → run selection repair over the clean arm bank
(shift-matched R², co-test, NNLS) to close the clean-vs-oracle gap → deepen the
weak targets (eps/nc via ionic + polarizability; ei via partner reconstruction
with repaired guard; tg via bagged GBM + group contributions) → run the SSL
long-run ladder at real scale (MLM transformer on 1M PI1M + up to 6M smile_r3,
Morgan-SVD-6M, word2vec-6M, char-TFIDF-6M, pseudo-labeling; equal-budget
controls; strong GBM heads) → TTA with a real sequence head + train-time
randomized-SMILES augmentation → per-target SHAP explainability → assemble the
final compound by fold-local NNLS → package ONE clean end-to-end notebook +
submission.csv pair. Success = `logs/latest_verified.txt` ≥ 0.93 before the
3 September deadline.

## 10. Quick-start for a fresh agent

1. Read this file + TRIALS.md (+ PLAN.md if available).
2. State the current best verified score (0.90276 unless logs say otherwise)
   and the next planned experiment id.
3. Never propose an experiment without checking TRIALS.md and the Round 2 logs
   (on the GPU laptop, read-only) for the same idea.
4. Verify data hashes; confirm no oracle references in any clean code path.
5. Run experiments with real bodies only; score post-freeze; append logs.
6. The user submits to Kaggle; agents never submit.
