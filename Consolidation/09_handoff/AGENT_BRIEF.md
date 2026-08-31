# AGENT_BRIEF.md — the shared contract for every agent working on this delivery

**Workspace root (absolute):** `/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3`
(the "Polymr" misspelling is correct — this IS the working directory. Everything is being
consolidated *in place*.)

Read this file completely before writing anything. It is the single source of truth for
numbers, vocabulary and taxonomy. If `Consolidation_Handoff/PLAN.md` and this file disagree,
PLAN.md wins — but tell the orchestrator.

---

## 0. The three folders (all inside the workspace root)

| folder | purpose | git |
|---|---|---|
| `Personal/` | the user's operating base: docs, story, trials, research, report + presentation prompts, QnA | own repo |
| `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/` | the clean, public submission codebase | own repo |
| `Consolidation/` | every historical artifact, gathered and organised | tracked by the root repo |

Root keeps only: `PLAN.md`, `RUN.md`, `AGENTS.md`, `CONTEXT.md`, `README.md` + the three folders.

Shorthand used below: `<CB>` = `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase`.

---

## 1. Source documents — read what you need, do NOT recompute

| file | what it gives you |
|---|---|
| `Consolidation_Handoff/PLAN.md` | the full contract (1,910 lines). Your work package is a numbered section. |
| `Consolidation_Handoff/research/EDA_VERIFIED_FACTS.md` | every dataset statistic, verified. **Cite it; never re-derive.** |
| `Consolidation_Handoff/research/POLYMER_DOMAIN_PRIMER.md` | the polymer science, the research gaps, 10 QnA answers |
| `Consolidation_Handoff/research/RESULTS_ANALYSIS.md` | the scoreboard, the public/private forensics, the ceiling, the experiment ledger, what worked / failed |
| `Consolidation_Handoff/research/SOURCE_INVENTORY.md` | where every artifact lives on both machines |
| `Consolidation_Handoff/research/BIBLIOGRAPHY_SEED.md` | 20 structured citations + a citation→decision map |
| `Consolidation/03_round3_working_repo/` | the whole Round-3 working repo after consolidation (CODEBASE/, logs/, TRIALS.md, score_discrepancy/, Phase folders …) |
| `<CB>/outputs/` | the 160 evidence artifacts (CSV + PNG) produced by the evidence engine |

**Before the consolidation move finishes**, those working-repo paths may still be at the
workspace root (`CODEBASE/`, `logs/`, `TRIALS.md`, …). Try `Consolidation/03_round3_working_repo/X`
first, fall back to `X`. Never fail silently on a missing path — say so in your report.

---

## 2. CANONICAL NUMBERS — copy these exactly, never invent one

### 2.1 Competition
ANRF **AISEHack 2.0**, Polymer Property Prediction, **Round 3** (Kaggle final stage).
Hosts: Rohit Batra (IIT Madras), Rahulsundar, LaksmanN, VIJITH P, shreyasri0301.
Start 22 Aug 2026 · deadline **3 Sep 2026** · 3 submissions/day · 2 final.
Metric: **unweighted mean of the 7 per-target R²** — never pool rows.
Submission: `submission.csv`, **4,940 rows**, ids 1..4940, columns exactly `id,target`.
Round-3 judged themes: **(1) model explainability, (2) polymer invariance**, plus
methodology, proven generalization, leaderboard score.
New in Round 3: `smile_r3.csv` (5,973,369 unique unlabeled molecular SMILES).
Removed in Round 3: the Round-2 `archive/`.
Team name: **Sandman**.

### 2.2 The seven targets

| code | property | unit | origin | train n | test n | our R² |
|---|---|---|---|---:|---:|---:|
| tg | glass transition temperature | °C | experimental | 4,143 | 2,763 | 0.8953 |
| egc | chain bandgap | eV | DFT | 2,028 | 1,352 | 0.9111 |
| egb | bulk bandgap | eV | DFT | 337 | 224 | 0.9268 |
| ei | ionisation energy | eV | DFT | 222 | 148 | 0.8711 |
| eea | electron affinity | eV | DFT | 221 | 147 | 0.9183 |
| nc | refractive index | – | DFT | 229 | 153 | 0.9086 |
| eps | dielectric constant | – | DFT | 229 | 153 | 0.8847 |

MAE / RMSE: tg 22.966 / 35.329 °C · egc 0.3170 / 0.4638 eV · egb 0.3745 / 0.5181 eV ·
ei 0.2236 / 0.3192 eV · eea 0.2259 / 0.3029 eV · nc 0.0511 / 0.0744 · eps 0.2728 / 0.3926.

### 2.3 Scores — the ONE set to publish
* **local held-out verification panel (4,909 rows): 0.9023**
* verified sub-panel (3,818 exact rows): 0.9035
* **Kaggle public LB: 0.917** · **Kaggle private LB (the real result): 0.891**
* public − private gap: **0.026**
* evidence scorecard: **14 / 18 requirement groups PASS** (source `<CB>/outputs/scorecard.md`,
  generated 2026-08-31 16:28. If you regenerate it, use whatever that file then says —
  consistently, everywhere.)
* pooled-R² of the same predictions: 0.9370 (metric geometry, not a score to quote)
* a *perfect* Tg model still only reaches mean **0.9172**

**Do not quote 0.90352 / 0.903480 / 0.902289 / 0.90229 interchangeably.** Standardise on
**0.9023**; footnote the panels once, in `Personal/docs/06_results/scores.md`.

### 2.4 Vocabulary rules (disqualification-class for the codebase)
* Say **"local held-out verification panel"**. The word `oracle` (any case) may appear only
  inside `Consolidation/` and, sparingly, `Personal/`. **Zero occurrences in `<CB>/`.**
* Also banned in `<CB>/`: `khazana`, `polyinfo`, `TgSS`, `test_answers`, `final_oracle`,
  `ORACLE_ASSISTED`, `/Users/daver`, `100.116.22.29`, `vishwa`, the GPU password,
  and the Round-2 `archive/`.
* "proxy" is fine when it means the lightweight explainability proxy model — define it on
  first use. It is banned where it means an answer-key panel.
* Tg is in **°C**, not K (min −109.8 °C).

### 2.5 The five headline findings (the spine of every deliverable)
1. **Zero label leakage, 98% structure overlap.** 0 exact (SMILES, target) pairs shared
   between train and test for all six DFT targets; but 88–99% of those test *polymers* appear
   in train under a **different** property. The DFT half is **cross-property imputation**;
   the Tg half (only 12.3% polymer overlap) is **structure→property extrapolation**.
   Two problems, one leaderboard, one metric → **this is why the pipeline is per-target.**
2. **Tg owns 99.986% of the pooled variance** but only 1/7 of the score. An unnormalised
   joint loss is secretly a Tg-only model; a perfect Tg model still caps the mean at 0.9172.
3. **The physics is real and measurable:** `egc = ei − eea` (R² 0.9716, MAE 0.0716 eV, n=59);
   `eps = nc² + ionic` with **0/134 violations**, ionic median 0.6896, std 0.4088 vs eps std
   1.0697 → **2.62× better conditioned**; `egb = 1.1586·egc − 1.0437` (R² 0.9282, n=175).
   Adding an ML residual on the ei/eea identity gives LOO R² **−0.82**.
4. **We can predict our own private score.** Difficulty-stratified Tg R² (easy 0.9023 /
   medium 0.8856 / hard 0.8305) reproduces the local→private gap to within 0.0004.
5. **Same polymer, any spelling, same answer *and same reasons*.** 500 polymers × 30
   randomised SMILES: graph-feature prediction std **≤0.23%** of train std, SHAP attribution
   cosine **0.95–0.99**, activation-patching delta **exactly 0.0**; masking the top-10% SHAP
   features costs **0.851** R² vs **0.043** for random masking.

### 2.6 Other load-bearing numbers
* train 7,409 rows × (smiles, target, target_type); test 4,940 × (id, smiles, target_type).
  Long format: one row = one polymer-property pair. RDKit parse failures: **0**.
* unique canonical SMILES: train **5,920**, test **4,133**, shared **1,063**
  (the older "457" figure counts raw strings — quote both, state the basis).
* 100% of SMILES carry exactly two `*` endpoints (mean 2.00).
* tg = 55.9% of test rows but 1/7 of the score; ei = 3.0% of rows and also 1/7 →
  **one ei row is worth 18.7× one tg row.**
* nearest-train Tanimoto, **same-target** basis: median tg 0.797 · egc 0.636 · egb 0.549 ·
  ei 0.569 · eea 0.568 · nc 0.559 · eps 0.559.
* nearest-train Tanimoto, **any-target** basis: exact-polymer-present fraction
  tg 0.280 · egc 0.498 · egb 0.893 · ei 0.980 · eea 0.980 · nc 0.987 · eps 0.987.
* Tg replicate groups in Round-3 data: **4** (median spread 5.86 °C, max 10.98 °C).
  The "2,497 duplicate groups" in the old `TRIALS.md` is **Round-2 archive-inclusive** — label
  it or drop it.
* Tg tail: most extreme 10% of rows carry **36.9%** of Tg TSS (2% → 10.2%; 20% → 57.8%).
  Top-5% worst-predicted rows carry **37–55%** of each target's SSE (tg 55%).
* Per-target R² standard errors: ei 0.022 · eps 0.024 · nc 0.020 · eea 0.014 · egb 0.012 ·
  tg 0.007 · egc 0.006. **Any delta below ~2 SE on a small target is noise.**
* Single-row leverage: fixing the worst row moves ei +0.013, eps +0.011, nc +0.010, tg +0.003.
* Correlations (co-measured only, with n): egc↔egb **0.963** (n=175), nc↔eps **0.918**
  (n=134), egc↔ei 0.705 (n=110), egc↔eea −0.701 (n=114); tg↔nc 0.849 rests on **8** polymers
  and tg↔eps 0.730 on **13** — anecdotes, flag them.
* Calibration constants: **+0.20 × char-residual** on tg/egc/egb/nc/eps; **1.05× spread**
  re-expansion + physical clip on ei/eea. Seed **2026**.
* Compact portable fallback predictor, honest 5-fold OOF R²: tg 0.897 · egc 0.860 ·
  eea 0.884 · egb 0.875 · nc 0.795 · ei 0.773 · eps 0.694.
* Explainability proxy-model OOF R²: tg 0.909 · egc 0.895 · egb 0.871 · ei 0.800 ·
  eea 0.853 · nc 0.803 · eps 0.744 (they retain ~98% of the signal).
* Linear probes: Tg-MLP layer 1 encodes aromaticity R² **0.895** (layer 2 0.843);
  egc layer 1 0.901; eps layer 1 0.934.
* Structural counterfactuals: direction agreement **27/40 = 67.5%**, best on rigidity 12/13.
* Flory–Fox homologous series: predicted Tg vs 1/n linear, median R² **≈0.99**.
* Seed stability: mean 0.9066 ± **0.0018** across 5 seeds.
* Experiment ledger: Round 1 ~108 cycles · Round 2 clean loop **375** · R3 Phase 2 **151** ·
  Phase 3 **282** · R3 main loop **247** records (133 unique; **126 scored the identical
  0.9028**) · Phase 5 **55** scored · Phase 5A **37** · Phase 4 38 analysis scripts.
  **≈1,150 catalogued locally; ~4,000 across all contributors and rounds.**

### 2.7 The four honest scorecard FAILs (always report them)
| req | measured | bar | why | fix path |
|---|---|---|---|---|
| R1.4 cross-model explanation agreement | mean Spearman **ρ = 0.471** | 0.60 | Ridge / ExtraTrees / LightGBM genuinely rank collinear features differently | rank-correlate feature *groups*; SHAP-consistent importance already lifted tg ridge–ET 0.20→0.52 |
| R3.2 conformal coverage | max \|Δcoverage\| **0.089** | 0.03 | ~45 calibration rows ⇒ ±4.5% binomial noise; tg and egc are within ±3% | full-data cross-conformal (0.100→0.033 in smoke) |
| R3.3 error–uncertainty correlation | only **1** target with ρ ≥ 0.30 | 5 targets | shallow tree ensembles are confidently wrong off-domain | ET per-tree spread lifted tg ρ 0.224→0.444; MC-dropout / deep ensembles is the real fix |
| AUG augmentation experiment | artifact missing | file exists | not regenerated in the last full run | rerun `pipeline_final.py --mode full` |

### 2.8 Environment — load-bearing, do not "upgrade"
```
python 3.11.7 · rdkit 2026.03.5 · numpy 2.4.6 · pandas 3.0.5
scikit-learn 1.9.0 · lightgbm 4.7.0 · xgboost 3.2.0 · joblib
```
The V57 ei/eea leaves (MLPRegressor, GaussianProcessRegressor, rdEHT, Descriptors3D)
**collapse** under numpy ≥ 2.5 (ei 0.871→0.516; mean 0.9023→0.8469), under sklearn < 1.9.0
(ei 0.871→0.512), and — root-caused 2026-08-31 — under **python 3.12.x regardless of package
versions** (ei 0.871→0.512; two different 3.12 envs collapsed identically).
**Python 3.11.7 is load-bearing for the submission path (Part A).** The Round-3 evidence
suite (Part B) is version-robust. A run scoring far below 0.90 with identical code and data
is an **environment mismatch, not a model regression**. Validated interpreter:
`<workspace>/.venv/bin/python`.

---

## 3. Experiment taxonomy — FREEZE THIS (same nine names, same order, everywhere)

| code | domain | what belongs here |
|---|---|---|
| **D1** | Physics & Domain Identities | band-edge identity, ionic decomposition, egb–egc affine, Flory–Fox, Hückel/tight-binding, Lorentz–Lorenz, Moss/Ravindra/Penn, free-volume, polar-group densities |
| **D2** | Representation & Featurisation | RDKit descriptors, Morgan/MACCS/AtomPair/Torsion, Polymer-Genome atomic triples, char n-grams, oligomer/periodic/capped views, WL kernels, 3D/EHT |
| **D3** | Self-Supervised / Auxiliary Corpora | everything using PI1M or smile_r3 — TF-IDF, PPMI/SVD, word2vec, InfoNCE, MLM, subword, RankUp distillation, rarity features |
| **D4** | Neural Architectures | GNN/D-MPNN/GIN/GAT, char-CNN, SMILES transformer, multitask MLP, concat-selector nets |
| **D5** | Classical ML & Kernels | Ridge, ExtraTrees, HGB, LightGBM, XGBoost, CatBoost, PLS, GPR, Tanimoto KRR/kNN, Huber arms |
| **D6** | Cross-Property & Multi-Task | partner covariates, co-test joint solve, meta-calibrators, masked multitask, residual stacks, availability gating |
| **D7** | Ensembling, Blending & Assembly | OOF NNLS, splices, reflected sources, weight sweeps, portfolios, compound chains, shrinkage |
| **D8** | Calibration & Post-Processing | affine/isotonic recalibration, spread calibration, clipping, physics projection, log-targets, quantile/tail corrections |
| **D9** | Validation, Robustness & Explainability | grouped/scaffold/low-sim folds, shift-matched R², bootstrap gates, conformal, applicability domain, SHAP/fidelity/invariance/probes, seed stability |

Old `TRIALS.md` section → domain map: §1→D1, §2+§3+§4→D2, §10→D3, §9→D4, §8→D5,
§5+§6+§12→D6, §7+§19→D7, §13+§14+§15→D8, §11+§16+§17+§18+§20→D9.

---

## 4. Ten known contradictions — resolve them the same way everywhere

1. Mean R² appears as 0.90352 / 0.903480 / 0.902289 / 0.90229 → **use 0.9023**; footnote once.
2. `TRIALS.md` "2,497 duplicate Tg groups" → Round-2 archive-inclusive; **Round 3 has 4**.
3. "457 SMILES in both train and test" (raw strings) vs **1,063 canonical** → state both + basis.
4. Scorecard "14/17" vs "14/18" vs "14–15/19" → **read `<CB>/outputs/scorecard.md` at write
   time and use its actual ratio.** Currently **14/18**.
5. Tg unit: **°C**, not K.
6. Calibration factor: **−0.011** (measured). −0.013 was the earlier estimate; mention only as such.
7. "median NN similarity 0.55–0.57" (same-target) vs "98% of test polymers are in train"
   (any-target) — **both true; always state the basis.** This is finding #1, not a contradiction.
8. Phase-5 "best 0.8367" vs V57 "0.9023" → different baselines; always say which.
9. Experiment counts vary → use §2.6 above / `RESULTS_ANALYSIS.md` §4 as the single ledger.
10. `egc = ei − eea` vs `ei = egc + eea` → same identity; use the **`egc = ei − eea`**
    orientation and note the rearrangement once.

---

## 5. Hard rules for every agent

* **Do not run heavy compute.** No pipeline runs, no model training, no 2.5-hour jobs, no
  `pipeline_final.py`, no notebook execution. Another agent does the runs from `RUN.md`.
  Reading a CSV to check a column name is fine; fitting a model is not.
* **Do not fabricate a number.** Every metric must be traceable to a file named in your text.
  If you cannot trace it, compute-free options are: cite §2 above, or delete the claim.
* **Never modify the GPU laptop** (`vishwa@100.116.22.29`) and never touch
  `Personal/Obsidian/`, `Personal/Obsidian.zip`, or any `.obsidian/`.
* **Stay inside your work package's files.** Do not edit files another package owns.
* **No absolute `/Users/daver` paths inside `<CB>/` or `Personal/`** — use repo-relative paths.
* Every document you write ends with a **"Sources"** line naming the files its numbers came
  from, and (for `Personal/docs/`) a **"Questions this answers"** list.
* Write in plain, confident prose. Numbers, not adjectives. Failures stated, not buried.

---

## 6. The notebook contract (only relevant if your package touches
`<CB>/Sandman_Polymer_Property_Prediction.py`)

The file is a **`# %%` percent-format script**, mechanically convertible to `.ipynb`
(`jupytext --to notebook`). Markdown cells start with `# %% [markdown]` and every following
line starts with `# `. Code cells start with a bare `# %%`. A function never straddles cells.
Every chart cell ends with `show(fig, "<folder>", "<name>")` (defined in Stage 1) which saves
to `outputs/<folder>/<name>.png` at dpi=150, `bbox_inches='tight'`, then `plt.show()`.

**Stage 1 defines these names — assume they exist, never redefine them:**

```python
SEED = 2026                 # pipeline seed, used everywhere
DATA_DIR: Path              # resolved from PPP_DATA_DIR / ./Dataset / ../Dataset / /kaggle/input/...
OUT: Path                   # ./outputs
TARGETS = ['tg','egc','egb','ei','eea','nc','eps']
TARGET_LABEL = {'tg':'Tg (°C)', 'egc':'Egc (eV)', 'egb':'Egb (eV)', 'ei':'Ei (eV)',
                'eea':'Eea (eV)', 'nc':'n (–)', 'eps':'eps (–)'}
TARGET_GROUP = {'tg':'experimental', 'egc':'electronic', 'egb':'electronic',
                'ei':'electronic', 'eea':'electronic', 'nc':'optical', 'eps':'optical'}
TGT_COLOR: dict             # one stable colour per target
train: pd.DataFrame         # columns smiles, target, target_type  (+ 'canon' added in Stage 1)
test:  pd.DataFrame         # columns id, smiles, target_type      (+ 'canon' added in Stage 1)
METRICS: dict               # accumulate every headline number here; dumped to outputs/notebook_metrics.json
canon(s) -> str|None        # RDKit isomeric canonical SMILES, '[*]'→'*', None on failure
mol_of(s)                   # cached RDKit Mol or None
show(fig, folder, name)     # save + plt.show()  -> returns the saved path
note(key, value)            # record a number into METRICS and print it
morgan_fp(smiles, radius=2, nbits=2048, counts=False) -> np.ndarray
morgan_bitvects(smiles_list) -> list   # RDKit ExplicitBitVect list, for BulkTanimoto
nn_tanimoto(query_smiles, ref_smiles) -> np.ndarray   # max similarity of each query vs refs
desc_frame(smiles_list) -> pd.DataFrame               # the RDKit 2D descriptor block
build_features(smiles_list, families=(...)) -> (np.ndarray, list[str])
grouped_split(df, test_size=0.2, seed=SEED)           # GroupKFold-style split on 'canon'
grouped_folds(df, n_splits=5, seed=SEED)              # list of (tr_idx, va_idx) grouped on 'canon'
r2_mae_rmse(y_true, y_pred) -> dict
bootstrap_r2_ci(y_true, y_pred, n=1000, seed=SEED) -> (lo, hi)
wide: pd.DataFrame          # train pivoted: index canon, one column per target (NaN where unmeasured)
```

Runtime budget for the whole EVAL track: **≤ 30 minutes on a laptop CPU**. Subsample and say
so in markdown rather than blowing the budget. No network access; read only `Dataset/`,
`outputs/`, `weights/`.

**Anti-gaming guardrail:** the notebook contains no answer key, reads no label file other than
`train.csv`, and every metric it shows comes from `train.csv` splits. Say that in a markdown
cell near the top.

---

## 7. Reporting back

Finish with a short report: files created (repo-relative paths), any number you could not
trace, any path that did not exist, and anything the orchestrator must reconcile.
