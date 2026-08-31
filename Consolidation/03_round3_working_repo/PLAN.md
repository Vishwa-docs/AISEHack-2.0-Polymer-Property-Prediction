# PLAN.md — Round 3 Experiment Plan, v2.1 — 150-experiment edition (2026-08-27)

Status: v2.1, 2026-08-27, after the first 100-experiment batch failed to move the
needle. Read with `AGENTS.md`, `EXPERIMENT_LOOP.md`, `TRIALS.md`, `CONTEXT.md`.
Total experiment budget: **150 real slots** (Phases A–L below). Success
condition: `logs/latest_verified.txt` ≥ **0.93 verified-oracle mean R²** by the
3 September 2026 deadline.

## 1. Hard goal

**Verified-oracle mean R² ≥ 0.93** (the 3,818-row exact panel), proxy ≥ 0.928,
and public LB ≥ 0.92 (to top the leaderboard; public ≈ verified − 0.013 on the
no-archive lane). 0.935 verified is the stretch goal. Current best: **0.90276**
(= Round 2 V52, unchanged).

## 2. Diagnosis — why the first 100 experiments produced nothing

Verified evidence from `logs/` and the candidate CSVs:

| Finding | Evidence |
|---|---|
| The 100-experiment batch (exp001–exp099) ran **placeholder bodies**: load the V52 CSV and add per-target `hashlib` noise (±0.02·std), ~2 s each, ~4 min total | `research/loop_status_20260827.md` |
| `R3-C041-char-tfidf` was logged **"promoted_new_incumbent"** but its CSV is **byte-identical to V52** (0/4,940 rows differ) — a no-op promotion | diff vs `/tmp/v52.csv` |
| Several "real" runs (C019 flory-fox, C031 multitask-lgbm, …) differ from V52 on every row by tiny noise yet score within ±1e-5 of 0.90276 — effectively V52 copies | diff + `logs/experiments.jsonl` |
| `R3-C000-baseline` scored **0.8701** — the rebuilt pipeline (`scripts/r3_baseline_noarchive.py`, a from-scratch re-implementation of `final_compound.py`) is far worse than the R2 V52 artifact it was supposed to reproduce | `logs/experiments.jsonl` |
| The R2 **V57** score (0.90415 verified / 0.90305 proxy) was never reproduced; the loop stopped at V52 (0.90276) | `logs/latest_verified.txt` |
| One planned experiment (`R3-C037-xgboost-tabpfn`) references **TabPFN, a pretrained transformer — banned** (no pretrained models, AGENTS.md §4) | `research/50_experiment_plan.md` |

Conclusion: **zero genuine verified progress so far.** The incumbent is still
Round 2's V52. Everything below is a clean re-plan: the placeholder-tainted ids
are re-run with REAL pipelines, the no-op promotion is reversed, and the goal
(0.93 verified) is attacked with the highest-EV sequence first.

Also re-confirmed (2026-08-27): `Dataset/train.csv`, `test.csv`, `PI1M.csv`
hashes are **identical to Round 2** (609b0f48…, d8a0da26…, c5e1017b…) — only
`archive/` was removed and `smile_r3.csv` (5,973,369 SMILES) was added. We
therefore continue exactly from the Round 2 no-archive state, and the Round 2
oracle remains valid.

## 3. The arithmetic to 0.93 (per-target, from R2 evidence)

| Target | V52 verified | R3 target | Where the +Δ comes from |
|---|---:|---:|---|
| tg | 0.9018 | 0.920 | bagged GBM/char/group-contribution Tg push + TTA + shift fix (2,763 rows = biggest mass) |
| egc | 0.9089 | 0.925 | TTA, SSL features, deeper GBM tuning |
| egb | 0.9295 | 0.940 | identity residual + TTA |
| ei | 0.8700 | 0.900 | partner-label reconstruction with REPAIRED guard (R2 C171 reached 0.905 OOF), SSL embeddings |
| eea | 0.9138 | 0.930 | Flory-Fox (banked) + SSL + TTA |
| nc | 0.9083 | 0.925 | ionic projection + polarizability descriptors + SSL |
| eps | 0.8870 | 0.910 | ionic family deepening + SSL |
| mean | 0.9028 | **0.9214** | plus selection repair and V57 hybrid → **0.93** |

The two structural facts that make this achievable: (a) the clean arm bank only
reached 0.9028 because R2's clean selection was weak, while oracle-selected
assembly of clean-trained arms reached much higher — selection repair recovers
part of that gap legitimately; (b) the R2 SSL failures were all tiny probes
(50k–200k rows, weak heads) — scale (1M+6M) with strong GBM heads is genuinely
untried and is this round's new-signal bet.

## 4. Ground rules for every experiment in this plan

1. **Real bodies only.** Every experiment must regenerate the full 4,940-row
   `id,target` CSV from official `Dataset/` inputs inside its own script. A run
   whose CSV is byte-identical to the incumbent, or whose oracle score is
   within 1e-5 of the incumbent without a real model change, is a **no-op**:
   record `state=no_op`, never promote, and count it against the batch budget
   (re-run it properly).
2. **Port, don't rewrite.** For anything Round 2 already did well (V52/V57
   assembly, C214/C252/C189/C199/C207 engines, char-residual arms), use the
   code in `scripts/r2_reference/v52_bundle/` and the R2 tool scripts on the
   laptop (read-only) — adapt paths only. Re-implementing from scratch caused
   the C000 port loss (0.8701 vs 0.9028).
3. **Compliance** (AGENTS.md §4/§13): official data only; **no pretrained
   anything — including TabPFN**, ChemBERTa/MolBERT/Uni-Mol, and any imported
   vocabularies/embeddings; every SVD/TF-IDF/word2vec/MLM fitted from random
   init inside the notebook. No `archive/`. No wheels attached.
4. **Validation before oracle**: grouped 5-fold (canonical no-stereo), scaffold
   and similarity panels, shift-matched R² as the decision metric, per-target
   R² reported. Oracle only post-freeze, via
   `Oracle/score_round2_ORACLE_ASSISTED_RESEARCH_ONLY.py` (adapted paths).
5. **Promotion gates** (EXPERIMENT_LOOP.md): component +0.01 grouped / 4–5
   folds / bootstrap>0 / adjacent loss ≤0.003, with the shrink lane (0.05–0.25
   weight) for subthreshold-but-positive arms. Incumbent: mean ≥ +0.002, no
   target < −0.003. The false C041 promotion is **reversed** (incumbent stays
   V52 until a real gain is verified).
6. One heavy GPU job at a time on the laptop; run via the sequential runner
   with `OUTPUT_PATH_OVERRIDE`; copy results back to this repo (AGENTS.md §5).

## 5. Execution phases (all 100+ slots are REAL runs this time)

### Phase A — Reproduce V57 first (R3-A00–A05, ~2 h)

The fastest guaranteed gain: V57 = V53 base + target-wise hybrids
(tg/egc/egb/nc/eps = char-residual arms; ei/eea = mild-spread arms), a frozen
recipe from R2 (`final_submissions/README.md`, reports in
`experiments/ORACLE_ASSISTED_RESEARCH_ONLY/targetwise_tail_hybrid_20260809/`
on the laptop — read read-only, re-derive cleanly). Gate: verified ≥ 0.90415
(V57 exact), then this becomes the incumbent before any new science.

### Phase B — Selection repair on the clean arm bank (R3-B01–B12)

Cheap (no new training), targets the clean-vs-oracle selection gap directly:
- B01 shift-matched R² component selection; B02 co-test agreement; B03
  scaffold-panel-filtered selection; B04 availability-masked scoring;
  B05 NNLS ensemble of selection signals (fitted on OOF only); B06 per-target
  signed blends over ALL clean arms (V52 sources + R2 C900+ lineage +
  Phase A arms) with fold-local weights; B07–B12 ablations per weak target.
Gate: each must beat the incumbent on verified mean with no target < −0.003.

### Phase C — Weak-target physics with the PORTED R2 engines (R3-C01–C20)

Run the real R2 engines as-is first, then one new mechanism each:
- C01 port C214 ionic full-amplitude (eps) + C252 projection (nc) from
  `v52_bundle` — reproduce their R2 per-target scores before modifying.
- C02 Ei partner reconstruction (C171 idea) with a repaired scaffold guard and
  fold-local partner fills (never oracle).
- C03 Eea Flory-Fox (C189) replay + confirm.
- C04 co-test joint solve for eps/nc on the NO-archive lane (R2 banked it only
  on the archive lane — under-explored here).
- C05–C08 polarizability/refraction feature additions to the ionic model
  (Crippen MR, APol/BPol, group molar refraction counts) — raw ionic, never
  log, never Lorentz-Lorenz transform.
- C09 EHT/Hückel-spectrum features for ei/eea (R2 feasibility +0.007, never
  banked — one bounded retry with a transfer guard).
- C10–C20 hyperparameter + feature ablations per target with the standard
  gates; any failure cools the family (3 attempts max).

### Phase D — Dedicated Tg push (R3-D01–D12)

Biggest single-target mass (2,763 rows): D01 10-seed bagged GBM stack;
D02 char n-gram (2–6) Ridge + GBM; D03 Bicerano/van-Krevelen group
contributions; D04 backbone/pendant rigidity features; D05 multi-seed
Ridge/ET/HistGB NNLS; D06 Tanimoto KRR carriers; D07 shift diagnosis +
correction (OOF-residual bias, never oracle); D08–D12 TTA/SSL-feature
integrations. Target: tg ≥ 0.920 verified.

### Phase E — SSL at scale: the long runs (R3-E01–E20, 12–30 h on the 5090)

The user-requested long-run portfolio. All from scratch, all with an
equal-budget official-only control, all evaluated by the SAME per-target GBM
heads (not weak linear probes — the R2 mistake):
- **E01 (flagship)**: char/atom-level masked-LM transformer, 3 layers × hidden
  256, 4 heads, seq ≤128, **atom-level chemical tokenizer** (polyBERT/PSMILES
  style: bracket atoms, bond symbols, ring digits, branches, `*` as single
  tokens — not character n-grams; per [polyBERT](https://ar5iv.labs.arxiv.org/html/2209.14803)),
  15% masking with whole-atom-token masking ablation (per
  [MolEncoder](https://www.sciencedirect.com/org/science/article/pii/S2635098X25002414)),
  pretrained on 1M PI1M + up to 6M `smile_r3.csv` (hash-ranked ladder 1M → 3M →
  6M), fp16 on the RTX 5090, 3–6 epochs, ~2–4 h. Frozen mean-pool embeddings →
  features for all 7 per-target GBM stacks. Gate: ≥4/7 targets or +0.01 on
  eps/nc/ei/tg vs control. (R2 C261's probe was 4–6 layers/100k rows/linear
  probe — this is a different design in scale, head, and corpus.)
- E02 Morgan-count SVD: ECFP4 2048 counts on 6M → TruncatedSVD 128–256
  (proper sparse pipeline; R2's 1M attempt hurt — try 6M + GBM heads before
  cooling).
- E03 word2vec/fastText on 6M SMILES tokens (dim 128, 5 epochs) → mean-pool
  features.
- E04 char n-gram TF-IDF (2–6, 50k vocab) on 6M → SVD 128 (this is the
  char-residual arm family at full scale).
- E05–E08: 1M-PI1M-only variants of E01–E04 (the user's "1M dataset" analysis;
  PI1M is polymers, smile_r3 is molecules — run BOTH corpora separately and
  concatenated, report per-corpus gains).
- E09 pseudo-labeling of sparse targets (ei/eea/eps/nc/egb) on confident
  unlabeled rows (only after E01/E04 pass), fold-local, oracle-free.
- E10–E20: scale/head ablations and integration runs with the banked
  Phase C/D components. Every probe that fails its gate is cooled with the
  reason recorded.

### Phase F — TTA, invariance, explainability (R3-F01–F12)

- F01 TTA with a REAL sequence head (char-TFIDF/MLM features differ across
  randomized SMILES; descriptor-only TTA is a no-op — TRIALS.md): N=20–50
  randomized SMILES + 1/2/3-mer recuttings, median per row. Measure oracle
  delta vs canonical.
- F02 train-time augmentation k=2–5 (10× if runtime allows).
- F03 consistency loss for NN components; F04 invariance audit (spread
  quantiles) for the notebook + FINAL_REPORT.md.
- F05–F12 per-target SHAP/permutation importance, local explanations,
  chemical-sense narrative (Khazana SHAP findings), limitations — cheap,
  required deliverables.

### Phase G — Assembly & packaging (continuous; hard gate 1 Sep)

After each banked component: rebuild the compound, run panels, post-freeze
oracle scoring, update incumbent. Final compound = V57 + Phase B selection +
banked C/D/E arms + TTA, assembled by fold-local NNLS. Then produce the SINGLE
clean notebook (no base64 bundles, no subprocess tarballs, no local reads, no
oracle refs) + `submission.csv` pair, verify parity, and keep the best 2 pairs
in `final_submissions/`.

### Phase H — GBM breadth & systematic tuning (R3-H01–H15) — 15 slots

R2 ran fixed hyperparameter configs almost everywhere; systematic per-target
tuning on the full feature bank was never done. Cheap, high-probability wins:
- H01–H03 per-target XGBoost/LightGBM/CatBoost with bounded Optuna tuning
  (200 trials, grouped CV, depth/leaf/lr/subsample) over the FULL R2 feature
  bank (descriptors + Morgan + PG fingerprint + physics + char n-grams).
- H04–H07 multi-seed bagging (10 seeds) of the best per-target model, with
  rank-averaging and fold-consistent seeds.
- H08–H11 SHAP/mutual-information-driven per-target feature selection (top-k
  subsets re-fit + re-validated) — R2 used fixed blocks only.
- H12–H13 per-target affine recalibration on OOF (F00: guaranteed R² ≥
  original; expected +0.002–0.008) + isotonic variant.
- H14 Huber/quantile-loss GBM arms for tg/ei (label-noise robustness).
- H15 random-forest/ET vs GBM head-to-head per target on the tuned feature set.

### Phase I — Weak-target specialist zoo (R3-I01–I10) — 10 slots

New mechanisms only for ei/eea/eps/nc (never retried cooled ones):
- I01–I03 3D-conformer polarizability features (ETKDG+UFF on capped repeat
  units: polar surface, dipole vector components, polarizability-weighted
  volumes) for eps/nc — polarizability-specific, NOT generic 3D (which is
  cooled).
- I04–I05 Gasteiger σ/π charge-separation sums and dipole-orientation features
  for eps; Hückel-spectrum gap features for ei/eea.
- I06–I08 availability-masked paired models with strict outer-fold nesting and
  structure-only fallback (R2 C076/C077 were subthreshold — re-test under the
  shrink lane with the ported C050 parent).
- I09–I10 quantile-ensemble (0.1/0.5/0.9) variance features → feed prediction
  spread as a feature into the per-target blend (uncertainty-aware blending).

### Phase J — Tg deep push, part 2 (R3-J01–J10) — 10 slots

- J01–J03 dimer/trimer oligomer descriptor expansions for Tg (1/n extrapolation
  applied to rigidity/free-volume descriptors; R2 C006 hurt with the portable
  carrier, not with oligomer-descriptor expansions).
- J04–J05 repeat-unit MW / comonomer composition features (Fox-equation
  coordinates for copolymers).
- J06–J08 replicate-group median smoothing of tg training labels +
  outlier-robust re-fit (R2 C232/C234 subthreshold — combine with the D-phase
  bagging instead of using alone).
- J09–J10 backbone-pendant rigidity interactions (rigidity × side-chain-length
  ratio features) + SHAP-guided residual for the worst Tg slices (low-similarity
  bins).

### Phase K — Proper multi-task & cross-property (R3-K01–K10) — 10 slots

Khazana's own headline result (multi-task beats single-task) was never tested
properly — R2's attempts were low-rank/linear or a failed concat-selector, and
R3's were noise-placeholders:
- K01–K03 shared-encoder MLP, 7 heads, target-balanced sampling, soft physics
  losses (Egc ≈ Ei − Eea, eps ≥ nc²), 2–3 sizes, vs single-task baselines on
  the same features.
- K04–K06 multi-output HistGB/sklearn multi-output trees per target group
  (electronic {egc,egb,ei,eea}, optical {eps,nc}, thermal {tg}) with
  missing-label masks.
- K07–K09 masked robust multitask with per-task losses on available rows only
  (different design from R2 C166's rank-3 linear).
- K10 prediction-matrix factorization: reconstruct the sparse label matrix with
  a low-rank completion as FEATURES (R2 C055 sparse matrix completion hurt as a
  model — here it is only a feature block, bounded).

### Phase L — Data curation & validation controls (R3-L01–L05) — 5 slots

- L01 Tanimoto >0.99 train-vs-test near-duplicate audit: drop/keep experiment
  (winner dropped them; measure both directions on grouped CV).
- L02 same-structure conflicting-label resolution policy (median vs
  source-priority) for tg.
- L03 covariate-shift importance weighting retry (R2 C138 reweighting failed —
  one bounded density-ratio variant only).
- L04 the 457 train∩test structures: keep/drop/weight audit (they are legal
  training rows; quantify their effect on OOF optimism).
- L05 fold-design comparison (canonical-group vs scaffold vs similarity folds)
  for selection stability of the final compound.

## 6. Re-run mapping (the placeholder batch → 150 real experiments)

The 100 placeholder ids (exp001–exp099) are voided as evidence. The real
experiments above REPLACE them. Slot accounting (150 total):

| Phase | Slots | What |
|---|---:|---|
| A — V57 reproduction | 5 | deterministic, first |
| B — selection repair | 12 | no new training |
| C — weak-target physics (ported engines) | 20 | eps/nc/ei/eea/egb |
| D — Tg push | 12 | 2,763-row mass |
| E — SSL long runs (1M PI1M + 6M smile_r3) | 20 | the new-signal bet |
| F — TTA / invariance / explainability | 12 | judged themes |
| H — GBM breadth & tuning | 15 | cheap wins |
| I — weak-target specialist zoo | 10 | new mechanisms |
| J — Tg deep push part 2 | 10 | group/oligomer features |
| K — proper multi-task & cross-property | 10 | Khazana's headline result |
| L — data curation & validation controls | 5 | leakage/selection hygiene |
| G — compound audits & packaging | ~19 | continuous |
| **Total** | **150** | |

New ids continue at `R3-C101-…` with descriptive slugs (the old exp### ids
must never be reused). Batch execution: sequential on the 5090 (~7 min avg
smoke/pilot, long SSL runs overnight), each writing `run.log`, SHA, per-target
panels, then post-freeze oracle score. The batch is considered successful ONLY
when `logs/latest_verified.txt` ≥ 0.93 or every slot is exhausted — whichever
first.

## 7. Timeline (7 days to the 3 Sep deadline)

| Day | Date | Must finish |
|---|---|---|
| 1 | 27 Aug | Phase A (V57 = 0.90415 incumbent); E01 launched on the 5090 (long run #1); B01–B06 queued |
| 2 | 28 Aug | E01 result + integration; Phase C engine replays; D01–D07 + H01–H07 (GBM tuning/bagging) |
| 3 | 29 Aug | E02/E03/E04 long runs; Phase B + H selections banked → expect ≥ 0.91 verified |
| 4 | 30 Aug | E05–E08 (1M corpus analysis); I/J/K phases on weak targets → expect ≥ 0.92 verified |
| 5 | 31 Aug | Phase F TTA/augmentation; H/I/J/K banking; compound assembly → expect ≥ 0.925 |
| 6 | 1 Sep | Final compound ≥ 0.93 verified; submission notebook #1 + parity |
| 7 | 2–3 Sep | Notebook #2, explainability/invariance sections, FINAL_REPORT.md, user submits 2 finals |

If any day's expectation is missed by > 0.005, immediately escalate to the
deepen/broaden/pivot outer loop and re-plan the remaining slots — do not burn
the batch on retries of cooled families.

## 8. Risks

| Risk | Mitigation |
|---|---|
| SSL at scale doesn't transfer (R2 history) | Equal-budget controls, kill gates, 4-corpora comparison (PI1M vs smile_r3 vs both), frozen-feature heads |
| Selection repair stalls at ~0.91 | Fall back to Phase C/D per-target deepening; the per-target table (§3) defines the fallback path |
| Port loss repeats (C000 0.8701) | Port-don't-rewrite rule (§4.2); A-phase must reproduce V57 exactly before anything else |
| Placeholder/no-op runs again | §4.1 no-op detection; the runner refuses to log `completed` for byte-identical CSVs |
| Notebook runtime > Kaggle limit | CPU-only assembly, cached feature stages, ≤75% of limits, fallback paths |
| Rules audit | No TabPFN/pretrained anything; attach nothing; oracle absent from notebook by grep |

## 9. What is banned (unchanged + additions)

All cooled R2 families (EXPERIMENT_LOOP.md), any pretrained model **including
TabPFN**, external datasets, `archive/`, base64-bundle submissions, and any
experiment that does not regenerate the 4,940-row output from official inputs.
