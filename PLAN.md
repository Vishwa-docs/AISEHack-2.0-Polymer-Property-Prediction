# PLAN.md — Round 3 Experiment Plan (Polymer Property Prediction)

Status: v1, 2026-08-26. Owner: user + coding agent. Read together with
`AGENTS.md` and `EXPERIMENT_LOOP.md`. Deadlines: competition closes
**3 September 2026** (8 days from this plan).

## 1. Goal

Win Round 3. Concrete targets, all on the **no-archive lane** (Round 3 removed
`archive/`):

| Metric | Current state | Target |
|---|---|---|
| Kaggle public leaderboard | 0.92 (best competitor submission) | **≥ 0.93** (top the board) |
| Local verified-oracle mean R² (3,818/4,940 rows) | 0.9042 (our R2 best, V52/V57) | **≥ 0.935** |
| Local proxy mean R² (4,905/4,940 rows) | 0.9030 (R2) | ≥ 0.928 |
| Our R2 user-submitted public score | 0.891 | beaten by ≥ +0.03 |

Calibration (R2 evidence): public LB ran ≈ 0.013 below the local verified-oracle
score on the no-archive lane. A 0.935 verified score ≈ public 0.92+; a 0.94
verified score ≈ public 0.925+. **We must push the verified oracle to ≥ 0.94 to
win with margin.** Summed gap from R2's 0.9042: ≈ +0.25 R² points across seven
targets.

Also judged this round (non-score, but part of winning): **explainability** and
**polymer invariance**. Deliverables: notebook sections + `FINAL_REPORT.md`.

## 2. Rules snapshot (read Competition_Details/ for full text)

- Official competition data ONLY. No external datasets, no pretrained
  weights/embeddings/checkpoints, no artifacts made outside the notebook.
  `archive/` is gone — do not use Round 1 labels. `PI1M.csv` and `smile_r3.csv`
  are official and allowed for **from-scratch** representation learning.
- Notebook/code-only: the ENTIRE pipeline runs in ONE Kaggle notebook run, fixed
  seeds, writes `submission.csv` (4,940 rows, `id,target`), reproducible after
  the competition. Notebook shared with hosts; pinned version = submitted version.
- 3 submissions/day, 2 final submissions. The USER submits; agents never submit.
- Kaggle runtime: no internet at scoring; CPU/GPU as configured; RDKit, torch,
  sklearn, xgboost, lightgbm, catboost, shap are preinstalled. **Attach nothing**
  to the submission notebook (no wheels/datasets/checkpoints) — use the
  preinstalled stack only.

## 3. Data facts (verified 2026-08-26 — do not redo)

- `Dataset/train.csv`, `Dataset/test.csv`, `Dataset/PI1M.csv` are **byte-identical
  to Round 2** (SHA-256 in `EXPERIMENT_LOOP.md`). No train/test EDA needed.
- `Dataset/smile_r3.csv` is NEW: 5,973,369 unique molecular SMILES, zero overlap
  with train/test/PI1M, mean length 54. This is the only new data asset.
- `PI1M.csv` is official Round 3 data (user confirmed the organizers provided
  everything in `Dataset/`), even though the Dataset Description page omits it.
  Sanity-check its presence in the Kaggle input dir at notebook time.
- Test set: 4,940 rows / 4,497 unique SMILES; 457 test SMILES also appear in
  train → grouped validation is mandatory.
- Oracle: unchanged and valid (see `Oracle/NOTES_R3.md`): 3,818 exact values
  (all six DFT targets complete: egc 1,352 · egb 224 · ei 148 · eea 147 · eps 153 ·
  nc 153; Tg exact for 1,641 rows only), plus a 4,905-row approximate proxy.
  Verification-only, post-freeze, never in submitted artifacts.

## 4. Where Round 2 left us (baseline anatomy)

R2 no-archive per-target verified-oracle bests (from
`research/r2-experiment-history-digest.md`):

| Target | verified R² | Target |
|---|---:|---|
| tg | 0.9018 | 0.930 |
| egc | 0.9089 | 0.940 |
| egb | 0.9295 | 0.945 |
| ei | 0.8700 | 0.915 |
| eea | 0.9138 | 0.940 |
| nc | 0.9083 | 0.930 |
| eps | 0.8870 | 0.920 |
| mean | 0.9028 | **0.9314** |

Key R2 evidence:
- The clean best was assembled as: C050 7-target parent + banked per-target
  components (Egc C207 transfer-guard, Ei C199, Eea C189 Flory–Fox, Nc C252
  ionic projection, EPS C214 ionic full-amplitude), plus the V52/V57 signed
  per-target residual blend. All code is now local: `scripts/r2_reference/`
  (notebooks + fable engines + the full 381-file `v52_bundle/`).
- The **oracle-assisted noarchive diagnostic ceiling was 0.9506** (C1565). The
  clean lane only reached 0.9042. **≈ +0.047 of signal exists in the component
  bank that clean selection failed to capture** — closing this selection gap is
  the single highest-expected-value work item (Phase 2).
- Weak targets: ei (never exceeded 0.87 without oracle-assisted sources), eps,
  nc, and tg (0.90 without archive labels).
- Cooled families (do not redo): generic GNN/CNN/Transformer/MLM from scratch,
  PI1M PPMI/density/denoising/contrastive as previously designed, broad
  read-across, rich OOF stacking, micro-blend sweeps, abstention gates,
  Mordred/trimer/3D/AutoGluon sweeps, hard physics equalities.

External benchmark (research digests `research/web-research-kaggle-strategy-20260826.md`
and `research/web-research-polymer-methods-20260826.md`): NeurIPS Open Polymer
Prediction 2025 winners got their edge from randomized-SMILES augmentation
(10×) + 50× TTA median + GroupKFold + OOF stacking + per-target models + Tg
distribution-shift correction. Everything except their pretrained models is
rules-compliant and directly reusable.

### 4.1 Literature calibration (what is actually achievable)

- The labels come from the **Khazana dataset** (Kuenneth et al., *Patterns* 2021,
  DOI 10.1016/j.patter.2021.100238). The six electronic/optical targets
  (egc/egb/ei/eea/eps/nc) are **deterministic DFT values** (≈8-repeat-unit
  oligomers, hybrid functional; Ei = −HOMO, Eea = −LUMO via Koopmans; eps/nc
  from electronic polarizability via Clausius-Mossotti/Lorentz-Lorenz) → no
  label noise, high achievable R². **Tg is experimental** (PoLyInfo) → noisy;
  published Tg R² tops out ≈ 0.80–0.92. R2's no-archive Tg (≈0.90 oracle) is
  already near that ceiling — expect at most +0.02–0.03 from modeling, and rely
  on the R3-C005 shift diagnosis rather than chasing 0.95+.
- Khazana's own headline result: **multi-task learning beats single-task**,
  especially for sparse targets — the strongest literature justification for a
  shared-backbone multi-task experiment (R2's few multitask attempts were
  low-rank/linear or a failed concat-selector NN; a proper shared-encoder MLP /
  multi-head GBM with physics heads was NOT explored).
- Published 5-fold-CV benchmark (PolyCL paper) on the same property family:
  pretrained-from-scratch transformers ≈ 0.83–0.85 mean, driven by Ei/EPS/Nc
  gains (+0.03–0.09 over fingerprint-only). Our R2 oracle-verified weak-target
  scores (eps 0.887, nc 0.908, ei 0.870) are already ABOVE those CV numbers —
  the competition test set is friendlier than a random split (457 train/test
  shared structures), so treat literature numbers as relative guidance, not
  ceilings.

## 5. Phases

### Phase 0 — Bootstrap (Day 1, ~2 h)

- Freeze seeds, target order, fold scheme (GroupKFold on canonical no-stereo
  SMILES), canonicalizer, evaluation code. Port `fable_common.py` shift-matched
  R² metric as the decision metric.
- Verify all hashes; verify oracle isolation; create `logs/experiments.jsonl`.
- Adapt `scripts/r2_reference/v52_bundle/scripts/sandman_runner.py`:
  - `locate_bundle()` must find the Round 3 Kaggle input dir (likely
    `/kaggle/input/ppp-round-3` or `/kaggle/input/aisehack-2-0` — add both, and
    the local `Dataset/` fallback for Mac runs);
  - remove anything that touches `archive/` (the no-archive lane already does
    not read it — verify by grep).

### Phase 1 — Reproduce the R2 baseline (Day 1, experiment R3-C000)

Gate: run the adapted V52 engine locally (Mac/GPU laptop scratch) from
`Dataset/` and reproduce **verified 0.9042 / proxy 0.9030** within 0.0005, with
a clean source scan and a 4,940-row output. This is the frozen incumbent for
every later comparison. If reproduction fails, fix before anything else.
Deliverable: `experiments/R3-C000-.../` + a **plain, readable** single-script
copy of the essential pipeline (C050 parent + banked components + V52 blend
weights) in `scripts/r3_baseline.py` — this will become the submission skeleton
(no base64 bundles, no subprocess tarballs).

### Phase 2 — Close the clean-vs-oracle selection gap (Days 1–3, R3-C001…C006)

Highest expected value: recover part of the 0.904 → 0.951 diagnostic gap
**without the oracle**. Experiments (one at a time, each with a pass gate):

1. **Shift-matched selection** (R3-C001): select per-target components by the
   shift-matched R² (OOF reweighted to the test nearest-neighbour similarity
   histogram) instead of plain OOF. Gate: verified mean > 0.9042.
2. **Co-test / consistency selection** (R3-C002): select among the existing
   component bank by agreement of 2+ independent models per target and
   availability-masked scoring. Gate: verified mean > best-so-far.
3. **Conservative target-wise promotion with per-target transfer guards**
   (R3-C003): re-run R2's C199/C207-style transfer guards on every weak-target
   component with the frozen fold map; promote only guard-passing components.
4. **Ensemble of selection criteria** (R3-C004): NNLS over the candidate
   selection signals, fitted on OOF only.
5. **Tg shift diagnosis** (R3-C005): the 2025 winner found a real Tg
   train-vs-test distribution shift. Quantify it on the proxy panel and test a
   shift correction (e.g., additive bias fitted on OOF residuals; never on
   oracle).
6. **Deduplication audit** (R3-C006): Tanimoto >0.99 train/test near-duplicates;
   re-score after dropping near-duplicate train rows; keep only if CV improves.

Gate for all: verified-mean improvement ≥ +0.003 over incumbent, no target
worse than −0.003, panels pass. Bank the winners.

### Phase 3 — Weak-target physics deepening (Days 2–5, R3-C010…C030)

The R2 physics wins are the strongest transferable signals. Deepen them with
rules-compliant features (all computable with RDKit; no external data):

- **EPS/Nc (eps 0.887 → 0.920+, nc 0.908 → 0.930+)**: ionic-coordinate family
  (raw `ionic = eps − nc²`; R2 showed `log(ionic)` HURTS ~0.02 and the
  Lorentz–Lorenz/Clausius–Mossotti transform underperforms plain `nc²` — do not
  repeat either) + molar refractivity/polarizability descriptors (Crippen MR,
  TPSA, atomic polarizabilities), dipole-moment proxies, 3D conformer
  polarizability estimates, oligomer 1/n extrapolation (Flory–Fox style),
  co-test consistency with the paired target, and soft (never hard) eps ≥ nc²
  constraints.
- **Ei (0.870 → 0.915+)**: identity coordinates Ei = Eea + Egc and chi/gap
  coordinates with strict availability masks (R2 C171 reached 0.905 OOF but
  failed scaffold bootstrap — repair the transfer guard, don't repeat blindly);
  EHT/quantum-chemistry descriptors (Gasteiger/Hückel spectra already exist in
  the feature bank); conjugation/donor–acceptor SMARTS counters; 3D
  HOMO/LUMO-proxy features.
- **Tg (0.902 → 0.930+)**: group-contribution / Bicerano-style rigidity and
  free-volume features (the feature bank already has bicerano + mobility
  blocks), backbone/pendant decomposition, multi-seed bagged GBM/Ridge stacks,
  char n-gram Ridge carriers, TTA (Phase 5), shift correction (Phase 2, R3-C005).
- **Egc/Egb/Eea (marginal, +0.01…0.03 each)**: identity routes
  Egb = 1.1178·Egc − 0.9221 (R2 C160), Flory–Fox carriers already banked for
  Eea; add TTA and SSL features.
- **Group-contribution feature blocks (new for R3, cheap, explainable)**:
  van Krevelen/Hoftyzer additive molar-refraction and molar-polarization
  contributions (SMARTS group counts × tabulated R_M/P_M values), Bicerano Tg
  contributions (Tg = Σ Nᵢ·Yᵢ / M + structural corrections), Gladstone-Dale
  refraction, Fox/Flory-Fox corrections, and Mordred autocorrelation families
  (ATS/AATS/GATS/Moran/Geary, APol/BPol, EState) — the descriptor-space analogs
  of conjugation length and polarizability. These double as SHAP-ready
  interpretable features (Phase 6).
- **Proper multi-task experiment (R3-C031, bounded)**: shared-encoder MLP with
  7 heads (or 6 DFT heads + Tg) + soft physics constraints (Egc ≈ Ei − Eea,
  eps ≥ nc²), target-balanced sampling, evaluated per-target with the standard
  gates. Khazana's paper makes this the single best-motivated neural design;
  R2 never ran this exact configuration. Kill if no weak-target gain.

Gate: component gate +0.01 grouped, 4/5 folds, bootstrap > 0, adjacent loss
≤ 0.003 (EXPERIMENT_LOOP.md). No more than 3 attempts per target family before
moving on.

### Phase 4 — From-scratch SSL on the new 6M SMILES (Days 3–6, R3-C040…C060)

Round 3's auxiliary dataset is 6× the R2 PI1M corpus and is *molecular* (short,
valid SMILES). R2's PI1M probes all failed — new probes must differ in
representation AND scale AND head. Both `smile_r3.csv` and `PI1M.csv` are
official (user confirmed); the ladder below prioritizes `smile_r3.csv` because
it is 6× larger and new. Bounded ladder, cheapest first, hash-ranked
subsamples, every probe vs an equal-budget official-only control:

1. **Morgan/substructure-count SVD** (100k → 1M → 6M): count ECFP4/ECFP6
   (1,024–4,096 bits) matrix → TruncatedSVD to 128–512 dense features → append
   to tabular models. Deterministic, CPU, minutes-to-1 h. (R2's PPMI/SVD
   variant failed at 50k; the count-SVD at 6M scale + GBM heads is a different
   design.)
2. **Char n-gram TF-IDF** on 1M–6M SMILES (n=2–6, cap features ~50k, then SVD):
   matches the silver-medal recipe, extremely cheap.
3. **word2vec/fastText-style token embeddings** on 6M SMILES (regex/char
   tokens, dim 128–256, ~10–30 min) → mean-pooled molecule vectors.
4. **Tiny char/atom-level BERT MLM from scratch** (only if 1–3 fail to clear a
   gate): 2–4 layers × 128–256 hidden, vocab = SMILES token set, seq ≤ 128,
   1–2M sequences, ~30–90 min on the laptop's RTX 5090, ~1–2 h on Kaggle GPU.
   Frozen-embedding linear/GBM probe vs equal-budget control. (R2 C261's probe
   was 4–6 layers/256–384 on 100k PI1M rows — different design: shallower,
   more data, molecular corpus, better heads.)
5. **Pairwise-comparison pretraining** (winner's trick, cheap): predict which
   of two polymers has the higher property using unlabeled pairs; use the
   learned representation as features.
6. **Pseudo-labeling / self-training** (only after a probe passes): predict
   the sparse targets (ei/eea/eps/nc/egb) on unlabeled PI1M/smile_r3 polymers,
   keep confident rows, retrain the supervised heads. Bounded, fold-local,
   no oracle anywhere.

Gate (per probe): improves ≥ 4/7 targets or robustly improves one of
eps/nc/ei/tg (+0.01 grouped, panels pass) vs the control; otherwise cool that
probe and move down the ladder. Also record: tokens, dims, epochs, time,
memory, and whether the representation regenerates in one notebook run.
**Any SSL component that passes must still fit inside the final notebook's
runtime budget** — prefer frozen features over fine-tuning.

### Phase 5 — Invariance & TTA (Days 4–6, R3-C070…C079) — also a judged theme

1. **TTA**: for every test polymer, generate N randomized valid SMILES
   (N=20–50; doRandom=True) + repeat-unit recuttings (1/2/3-mer), predict, take
   the **median** per row. Measure the oracle gain vs canonical-only.
2. **Train-time augmentation**: k randomized SMILES per training row (k=2–5;
   10× if runtime allows). Report CV change.
3. **Consistency regularization** for any NN component: λ·mean((f(x)−f(x′))²)
   over paired views. Report invariance metric: std of predictions across
   20 random SMILES per polymer (target-wise), before/after.
4. **Invariance audit of the whole pipeline** (deliverable for the notebook and
   FINAL_REPORT.md): canonical vs randomized vs recut predictions, spread
   quantiles, worst-case polymers.
5. Note: tree/linear models are canonicalization-invariant by construction;
   document this as part of the explainability story.

### Phase 6 — Explainability build-out (Days 5–7) — judged theme

Per-target explainability for the final ensemble (cheap, deterministic, seeded):
- SHAP TreeExplainer on the GBM arms (sample ≤ 1,000 rows): global beeswarm +
  bar plots, per-target feature tables with directionality.
- Permutation importance for the Ridge/linear arms; coefficient magnitudes.
- 2–3 local explanations (waterfall/force) per target, including a high- and a
  low-prediction polymer.
- "Chemical sense" narrative: why aromaticity/conjugation drive bandgaps, why
  polarizability drives eps/nc, why rigidity/free volume drive Tg, why the
  ionic coordinate eps − nc² matters. Ground it in the Khazana paper's own
  SHAP findings (ring fraction ↑ → Tg/Nc/EPS ↑, Egc/Egb/Ei ↓; conjugated
  rings ↑ → Nc/EPS ↑) — the digest is in
  `research/web-research-polymer-methods-20260826.md`.
- Honest limitations note (data sparsity per target, Tg oracle gaps).

Store everything under `analysis/explainability/` and embed the reproducible
version in the final notebook.

### Phase 7 — Assembly, promotion, packaging (continuous from Day 1, hard gate Day 6)

- After each banked component: rebuild the 7-target compound, run all panels,
  post-freeze oracle scoring, update the incumbent (EXPERIMENT_LOOP promotion
  gates).
- The final compound = C050-style parent + banked per-target components +
  Phase-2 selection improvements + TTA; assembled by fold-local NNLS on OOF.
- **Submission packaging** (AGENTS.md §10): a SINGLE plain notebook (or .py)
  containing the complete inline pipeline — data load, features, folds, all
  per-target models, TTA, consistency/explainability outputs, `submission.csv`
  write — no base64 bundles, no subprocess, no local file reads, no oracle
  references. Verify: local parity (identical CSV), source scan, runtime
  estimate ≤ 9 h with ≥ 30% headroom (Kaggle CPU or GPU; prefer CPU-only for
  reliability — the R2 pipeline is CPU-friendly).
- Keep exactly the 2 best pairs in `final_submissions/`, delete superseded,
  update its README.

## 6. Timeline (8 days)

| Day | Date | Deliverables |
|---|---|---|
| 1 | 26 Aug | Phase 0 + C000 baseline reproduction (0.9042 verified) |
| 2 | 27 Aug | Phase 2 selection experiments; bank ≥ 1 improvement |
| 3 | 28 Aug | Phase 3 weak targets (ei/eps first); Phase 4 probe 1 |
| 4 | 29 Aug | Phase 3 tg/nc; Phase 4 probes 2–3; Phase 5 TTA integration |
| 5 | 30 Aug | Phase 4 probe 4 (MLM); Phase 5 augmentation; first full compound ≥ 0.92 verified |
| 6 | 31 Aug | Compound ≥ 0.93 verified; FIRST submission-ready notebook (public ≥ 0.92 attempt) |
| 7 | 1 Sep | Second candidate; explainability/invariance sections; submit 2 finals if user approves |
| 8 | 2–3 Sep | Final notebook parity/repro checks, FINAL_REPORT.md, final submissions |

Buffer: 3 submissions/day remain for last-minute corrections, but never chase
the public LB with unvalidated swaps (2025 lesson: public split ≈ 8% of test).

## 7. Decision rules

- Trust GroupKFold + shift-matched R² + transfer panels. The public LB is one
  noisy aggregate; the local oracle is the best pre-submission signal but is
  read only post-freeze.
- Bank a component only through the fixed gates. A target regresses ≥ 0.003 →
  reject the component even if the mean improves.
- No oracle value may influence pre-freeze choices (features, weights, rows,
  routing). Post-freeze oracle may choose aggregate components for the NEXT
  candidate (label `oracle-observed`).
- Every experiment writes its full record (AGENTS.md §8). Subagent research
  proposals must be verified by the main runner.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Can't reproduce R2 0.9042 baseline | Fix reproduction first (C000 gate); the full v52_bundle is local |
| Same-OOF gains don't transfer (R2's main failure mode) | Shift-matched metric, transfer guards, panels, no bank without gates |
| 6M-SMILES SSL doesn't transfer (R2 PI1M history) | Bounded ladder, equal-budget controls, kill gates, frozen-feature heads |
| Notebook too slow for one Kaggle run | CPU-only design, cached feature stages, runtime budget with 30% headroom, fallback paths |
| Tg oracle weak (1,122 rows unverified) | Track verified vs proxy panels separately; rely on proxy only as diagnostic |
| Public/private split shift | Grouped CV + shift diagnostics (Phase 2, R3-C005); no last-day LB chasing |
| Rules audit | Single clean notebook, no artifacts, no oracle references, seeds everywhere, pinned version = submitted version |

## 9. What NOT to do

- No retries of cooled R2 families without a genuinely new mechanism +
  pre-registered kill gate (list in EXPERIMENT_LOOP.md).
- No base64-embedded code bundles in the final submission (R2's was judged
  "tainted" by its own author). A clean, readable single file only.
- No oracle values in any artifact that gets uploaded. No external labeled
  data (the Tg datasets in `Oracle/sources/` are verification-only).
- No more than 1 heavy GPU job on the laptop; no Kaggle submissions by agents.
