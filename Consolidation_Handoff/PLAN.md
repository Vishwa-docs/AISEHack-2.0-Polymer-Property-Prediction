# PLAN.md — Consolidation, Codebase, Docs, Report, Presentation & Website
## AISEHack 2.0 · Polymer Property Prediction · Round 3 · Final Delivery

**Written:** 2026-08-31 by the planning agent.
**Audience:** the single execution agent that will carry this out end-to-end.
**Status of this file:** this is your contract. Read it completely before touching anything.

---

# 0. HOW TO USE THIS FILE

1. Read this file top to bottom. Do not skim §1 (boundaries) or §2 (canonical numbers).
2. Read the four companion research documents that the planning agent already produced —
   **they contain verified work you must NOT redo**:
   - `Consolidation_Handoff/research/EDA_VERIFIED_FACTS.md` — every dataset statistic,
     computed and verified from the raw CSVs (novelty, partner availability, physics
     identities, correlations, variance shares, tail concentration). **Cite it, do not
     recompute it** (except to turn it into charts).
   - `Consolidation_Handoff/research/POLYMER_DOMAIN_PRIMER.md` — the polymer science
     behind the seven targets, the research-gap analysis, and 10 QnA answers.
   - `Consolidation_Handoff/research/RESULTS_ANALYSIS.md` — the canonical scoreboard, the
     public/private gap decomposition, the mathematical ceiling, the full experiment ledger,
     what worked / what failed with numbers, and the Round-3 evidence results.
   - `Consolidation_Handoff/research/SOURCE_INVENTORY.md` — every path on both machines,
     what is in it, and where it should end up.
3. Work through the work packages in the order given in §17. They have dependencies.
4. Maintain `Consolidation_Handoff/PROGRESS.md` (create it) as an append-only log:
   one line per completed step with a timestamp. If your session dies, the next agent
   resumes from that file.
5. When you finish a work package, run its acceptance checklist (§16) before moving on.

---

# 1. SCOPE, BOUNDARIES AND HARD RULES

## 1.1 What you are building

Three sibling folders inside
`/Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3/` (note: this is
the folder with **"Polymer"** spelled correctly — it is the *destination*):

| # | Folder | Purpose | Git |
|---|---|---|---|
| 1 | `Personal/` | The user's operating base: docs, findings, story, trials, research papers, report/presentation prompts, QnA prep | own repo |
| 2 | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/` | The clean public submission codebase | own repo |
| 3 | `Consolidation/` | Everything else ever produced, gathered and organised | own repo, **containing nested repos** |

Plus a **root-level** `AGENTS.md` + `CONTEXT.md` + `README.md` at
`AISEHack 2.0 Polymer Property Prediction Round 3/` that routes any future agent to the
right sub-folder.

## 1.2 Hard boundaries — violating any of these is a failure

**B1 — Do not touch `Personal/Obsidian/`.** Those 12 notes are the user's own. Read them
for context; never edit, move, rename or reformat them. Same for `Personal/Obsidian.zip`
and the vault config `.obsidian/`.

**B2 — The source repo is read-mostly.** The live working repo
`/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/` (misspelled
"Polymr") is the source of truth for Round 3. **Copy from it; do not delete from it.**
Its git working tree was clean at HEAD `9db3154` when this plan was written — if you
change anything there, commit it with a clear message first.

**B3 — Never modify the GPU laptop.** `vishwa@100.116.22.29`. You may read, search and
`scp` **from** it. You may not create, edit or delete anything under
`~/Desktop/AISEHack-2.0/` or `~/Desktop/r3_runtime/`. If you need scratch space use
`/tmp/` and clean it up. Do **not** copy the 66.7 GB
`~/Desktop/AISE Full Codebase.zip`.

**B4 — Do not remove or re-init the git repos that already exist inside folders you move.**
They carry tags and history the user needs. If a folder you copy already has `.git/`,
keep it and register it in `Consolidation/AGENTS.md` as a nested repo (see §5.3 for how
to handle nesting so the outer repo does not swallow it).

**B5 — The oracle and all external label data never leave `Consolidation/`.**
That means: `Oracle/` (16 MB), `Oracle/sources/`,
`AISEHack-2.0/Polymer Prediction/scraped/` (Khazana + PolyInfo-derived Tg),
`Phase5_Kiro_Score_Improvement/data/final_oracle.csv`, and any file whose name contains
`oracle`, `test_answers`, `polyinfo`, `khazana`, `TgSS`.
They must be `.gitignore`d inside Consolidation too. **The submission codebase must contain
zero occurrences of the string `oracle` (any case) — this is a disqualification-class
requirement.** See §7.9 for the mandatory scan.

**B6 — Do not fabricate a number.** Every metric that appears in any document must be
traceable to a file. If you cannot trace it, either compute it (and record how) or delete
the claim. The four companion research docs give you the traceable set; §2 below is the
canonical list.

**B7 — Competition compliance is non-negotiable.** Only official data
(`train.csv`, `test.csv`, `PI1M.csv`, `smile_r3.csv`). No external datasets,
no pretrained weights/embeddings/vocabularies, everything fitted from scratch inside one
run, fixed seeds. Anything in the submission repo must obey this.

**B8 — No more than 80 experiments shown anywhere in the submission codebase.**
The full ledger (~1,150 catalogued locally, ~4,000 across all contributors) belongs in
`Personal/` and `Consolidation/`.

**B9 — Ask before doing anything destructive.** Deleting, overwriting outside the three
new folders, force-pushing, or anything the user has not authorised in this plan.

## 1.3 What "done" looks like

The user can open `Personal/AGENTS.md`, ask an agent *"prepare my 5-minute
presentation"* or *"produce the midnight report"* or *"what do I say if they ask why the
GNN failed"*, and that agent finds everything it needs without leaving `Personal/` (plus
read-only reference into the submission codebase).

---

# 2. CANONICAL FACTS — memorise these, use them everywhere

Any document that contradicts this section is wrong and must be fixed.

## 2.1 The competition

| item | value |
|---|---|
| Event | ANRF **AISEHack 2.0**, Polymer Property Prediction, **Round 3** (Kaggle, final stage) |
| Hosts | Rohit Batra (IIT Madras), Rahulsundar, LaksmanN, VIJITH P, shreyasri0301 |
| Start / deadline | 22 August 2026 / **3 September 2026** |
| Limits | 3 submissions/day, 2 final |
| Metric | **unweighted mean of the 7 per-target R²** — never pool rows |
| Submission | `submission.csv`, 4,940 rows, ids 1..4940, columns exactly `id,target` |
| Round-3 themes | **(1) model explainability, (2) polymer invariance**, plus methodology, proven generalization and leaderboard score |
| New in Round 3 | `smile_r3.csv` — 5,973,369 unique molecular SMILES, unlabeled, optional |
| Removed in Round 3 | the Round-2 `archive/` (which was our single biggest Round-2 lever) |

Team name for all artifacts: **Sandman** (from
`AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase` and the
`SANDMAN_Version_57` submission files). Confirm with the user before printing it on a
title slide.

## 2.2 The seven targets

| code | property | unit | origin | train n | test n | our R² |
|---|---|---|---|---:|---:|---:|
| `tg` | glass transition temperature | °C | **experimental** | 4,143 | 2,763 | 0.8953 |
| `egc` | chain bandgap | eV | DFT | 2,028 | 1,352 | 0.9111 |
| `egb` | bulk bandgap | eV | DFT | 337 | 224 | 0.9268 |
| `ei` | ionisation energy | eV | DFT | 222 | 148 | 0.8711 |
| `eea` | electron affinity | eV | DFT | 221 | 147 | 0.9183 |
| `nc` | refractive index | – | DFT | 229 | 153 | 0.9086 |
| `eps` | dielectric constant | – | DFT | 229 | 153 | 0.8847 |

## 2.3 Scores — the ONE set of numbers to publish

| measurement | value |
|---|---|
| **Local held-out verification panel (4,909 rows)** | **0.9023** |
| Verified sub-panel (3,818 exact rows) | 0.9035 |
| **Kaggle public LB** | **0.917** |
| **Kaggle private LB (the real result)** | **0.891** |
| public − private gap | **0.026** |
| MAE / RMSE per target | see `RESULTS_ANALYSIS.md` §1 |
| Round-3 evidence scorecard | **14 / 18 requirement groups PASS** |

**Vocabulary rule for anything public (submission repo, report, slides, website):** say
**"local held-out verification panel"**, never "oracle". The word `oracle` may appear only
inside `Consolidation/` and `Personal/` (and even in `Personal/`, prefer
"verification panel" in anything you might paste into a slide).

**Do not quote 0.90352 / 0.903480 / 0.902289 / 0.90229 interchangeably.** They are the same
model on different panels. Standardise on **0.9023** and footnote the rest once, in
`Personal/docs/`.

## 2.4 The five headline findings (the spine of every deliverable)

1. **Zero label leakage, 98% structure overlap.** For all six DFT targets there are
   **0 exact (SMILES, target) pairs** shared between train and test — but 88–99% of those
   test *polymers* appear in train under a **different** property. So the DFT half of this
   competition is a **cross-property imputation problem**, and the Tg half (only 12.3%
   polymer overlap) is a **structure→property extrapolation problem**. Two problems, one
   leaderboard, one metric. **This is why the pipeline is per-target.**
2. **Tg owns 99.986% of the pooled variance** but only 1/7 of the score. An unnormalised
   joint loss is secretly a Tg-only model. And a *perfect* Tg model still only reaches
   mean **0.9172** — Tg alone cannot win.
3. **The physics is real and measurable**: `egc = ei − eea` (R² 0.9716, n=59),
   `eps = nc² + ionic` with **0/134 violations** and 2.62× better conditioning,
   `egb = 1.1586·egc − 1.0437` (R² 0.9282, n=175). Exploiting them beat every
   learned alternative; adding an ML residual on the ei/eea identity gives LOO R² **−0.82**.
4. **We can predict our own private score.** Difficulty-stratified Tg R² (easy 0.9023 /
   medium 0.8856 / hard 0.8305) reproduces the observed local→private gap to within 0.0004.
5. **Same polymer, any spelling, same answer *and same reasons*.** 500 polymers × 30
   randomised SMILES: graph-feature prediction std **≤0.23%** of train std, SHAP attribution
   cosine **0.95–0.99**, activation-patching delta **exactly 0.0**; and masking the top-10%
   SHAP features costs **0.851** R² vs **0.043** for random masking.

## 2.5 Experiment-domain taxonomy — FREEZE THIS

Every place that groups experiments (`TRIALS.md`, `Experiment_Logs/`, report
Appendix A, presentation ablation slide, `docs/`) **must use exactly these nine domain
names, in this order, with these codes.** Inconsistent taxonomy across artifacts is the #1
way to look sloppy in front of a technical panel.

| code | domain | what belongs here |
|---|---|---|
| **D1** | **Physics & Domain Identities** | band-edge identity, ionic decomposition, egb–egc affine, Flory–Fox, Hückel/tight-binding, Lorentz–Lorenz, Moss/Ravindra/Penn, free-volume, polar-group densities |
| **D2** | **Representation & Featurisation** | RDKit descriptors, Morgan/MACCS/AtomPair/Torsion, Polymer-Genome atomic triples, char n-grams, oligomer/periodic/capped views, WL kernels, 3D/EHT |
| **D3** | **Self-Supervised / Auxiliary Corpora** | everything using PI1M or smile_r3 — TF-IDF, PPMI/SVD, word2vec, InfoNCE, MLM, subword, RankUp distillation, rarity features |
| **D4** | **Neural Architectures** | GNN/D-MPNN/GIN/GAT, char-CNN, SMILES transformer, multitask MLP, concat-selector nets |
| **D5** | **Classical ML & Kernels** | Ridge, ExtraTrees, HGB, LightGBM, XGBoost, CatBoost, PLS, GPR, Tanimoto KRR/kNN, Huber arms |
| **D6** | **Cross-Property & Multi-Task** | partner covariates, co-test joint solve, meta-calibrators, masked multitask, residual stacks, availability gating |
| **D7** | **Ensembling, Blending & Assembly** | OOF NNLS, splices, reflected sources, weight sweeps, portfolios, compound chains, shrinkage |
| **D8** | **Calibration & Post-Processing** | affine/isotonic recalibration, spread calibration, clipping, physics projection, log-targets, quantile/tail corrections |
| **D9** | **Validation, Robustness & Explainability** | grouped/scaffold/low-sim folds, shift-matched R², bootstrap gates, conformal, applicability domain, SHAP/fidelity/invariance/probes, seed stability |

Mapping note for the current `TRIALS.md`: its 20 sections collapse into these nine as
§1→D1, §2+§3+§4→D2, §10→D3, §9→D4, §8→D5, §5+§6+§12→D6, §7+§19→D7, §13+§14+§15→D8,
§11+§16+§17+§18+§20→D9.

## 2.6 Environment — load-bearing, do not "upgrade"

```
python 3.11.7 · rdkit 2026.03.5 · numpy 2.4.6 · pandas 3.0.5
scikit-learn 1.9.0 · lightgbm 4.7.0 · xgboost 3.2.0 · joblib
```

The V57 `ei`/`eea` leaves (MLPRegressor, GaussianProcessRegressor, rdEHT,
Descriptors3D) **collapse** under numpy ≥ 2.5 (ei 0.871→0.516; mean 0.9023→0.8469) and under
sklearn < 1.9.0 (ei 0.871→0.512). A run that scores far below 0.90 with identical code and
data is an **environment mismatch, not a model regression**. The validated interpreter is
`<source repo>/.venv/bin/python` — verified by this planning agent to hold exactly the
pinned versions. **Never delete that venv; never copy it either — recreate from
`requirements.txt`.**

---

# 3. PRE-FLIGHT (do this first, before creating anything)

**P1. Snapshot the source repo.**
```bash
cd "/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3"
git status --short          # was CLEAN at HEAD 9db3154 when this plan was written
git add -A && git commit -m "snapshot before Round-3 consolidation" || echo "nothing to commit"
git tag -a consolidation-start-$(date +%Y%m%d-%H%M) -m "state before consolidation"
git log --oneline | head -3
```

**P2. Snapshot the misc Mac folder.** `/Users/daver/Desktop/AISEHack-2.0` is **not** a git
repo. Do not init one there. Its safety copy is the `Consolidation/` move in §6.

**P3. Verify disk.** `df -h /Users/daver` — there were **78 GiB** free. Consolidation
needs roughly 3–4 GB if you follow the dedup rules in §14.6; it needs 40+ GB if you blindly
copy everything, so follow the rules.

**P4. Verify the environment.**
```bash
"/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/.venv/bin/python" -c \
  "import numpy,pandas,sklearn,rdkit;print(numpy.__version__,pandas.__version__,sklearn.__version__,rdkit.__version__)"
# expect: 2.4.6 3.0.5 1.9.0 2026.03.5
```

**P5. Verify GPU reachability** (needed for Work Package A3 and the research paper): use the
SSH_ASKPASS recipe in `SOURCE_INVENTORY.md` §D. Note: **the Mac has no `sshpass` and no
`timeout` command** — do not use them in scripts.

**P6. Confirm the destination.** `/Users/daver/Desktop/AISEHack 2.0 Polymer Property
Prediction Round 3/` currently contains only `Personal/` (with Obsidian + samples) and an
**empty** `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/`.
`Consolidation/` **does not exist yet** — you create it.

**P7. Note for the executing agent:** the planning agent's file sandbox could not write into
the destination folder (workspace-write scope). Yours may need an escalation. Ask the user
for the wider scope once, up front, rather than discovering it midway.

---

# 4. THE THREE TARGET TREES (build exactly these)

## 4.1 Root of the destination folder

```
AISEHack 2.0 Polymer Property Prediction Round 3/
├── AGENTS.md          ← global router: which folder answers which question
├── CONTEXT.md         ← the hackathon context, one page, portable
├── README.md          ← 20-line map for a human
├── .obsidian/         ← DO NOT TOUCH
├── Personal/                                                     (git repo #1)
├── AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/    (git repo #2)
└── Consolidation/                                                (git repo #3, nested repos inside)
```

## 4.2 Personal/

```
Personal/
├── AGENTS.md                  ← §12.1
├── CONTEXT.md                 ← §12.2
├── FINDINGS.md                ← §11.1 (mirror of the codebase FINDINGS.md, extended)
├── STORY.md                   ← §11.2 (the 5–6 min narrative)
├── TRIALS.md                  ← §11.3 (D1–D9 taxonomy, headline + extensive)
├── REMAINING_EXPERIMENTS.md   ← §11.4
├── QNA.md                     ← §11.5 (short index into docs/11_qna/)
├── docs/                      ← §10   THE BIG ONE
│   ├── 00_INDEX.md
│   ├── 01_task/
│   ├── 02_domain/
│   ├── 03_eda/
│   ├── 04_experiments/
│   ├── 05_architecture/
│   ├── 06_results/
│   ├── 07_explainability/
│   ├── 08_robustness/
│   ├── 09_generalization/
│   ├── 10_gaps_and_future/
│   ├── 11_qna/
│   └── 12_assets/             ← every chart referenced by any doc, plus a manifest
├── Research/                  ← §9   papers + articles, one .md per source + INDEX + .bib
├── Research_Paper/            ← §9.4 the Round-2 paper (draft, latex, figures, code)
├── Midnight_Report/           ← §7
│   ├── PROMPT_10PAGE.md
│   ├── PROMPT_3PAGE.md
│   ├── REPORT_STYLE_GUIDE.md
│   ├── Sample Reports/        ← MOVED here from Personal/Sample Reports/
│   └── analysis/SAMPLE_REPORT_ANALYSIS.md
├── Presentation/              ← §8
│   ├── PROMPT_PRESENTATION.md
│   ├── SLIDE_PLAN.md
│   ├── SPEAKER_NOTES.md
│   ├── DEMO_SCRIPT.md
│   ├── Sample Presentations/  ← MOVED here from Personal/Sample Presentations/
│   └── analysis/SAMPLE_DECK_ANALYSIS.md
└── Obsidian/                  ← UNTOUCHED (+ Obsidian.zip)
```

## 4.3 AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/

```
AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/
├── README.md                                  ← §5.2 (the front door)
├── ARCHITECTURE.md                            ← §5.3 end-to-end methodology
├── FINDINGS.md                                ← §5.4
├── RESULTS.md                                 ← §5.5
├── Sandman_Polymer_Property_Prediction.py     ← §6  THE NOTEBOOK SCAFFOLD
├── inference.py                               ← §5.6 (rewrite required)
├── weights/                                   ← §5.7 (+ weights/README.md)
├── submission.csv                             ← the exact file that was submitted
├── requirements.txt
├── setup.sh                                   ← one-command env bootstrap
├── LICENSE                                    ← ask the user; MIT unless told otherwise
├── .gitignore
├── src/
│   ├── pipeline_final.py
│   ├── evidence_engine.py
│   └── featurize.py
├── outputs/                                   ← curated charts (NOT all 169 — see §5.8)
│   ├── eda/  training/  explainability/  robustness/  generalization/
│   ├── architecture.png (+ source)
│   ├── CAPTIONS.md
│   └── TRUSTWORTHINESS_REPORT.html
├── Experiment_Logs/                           ← §5.9  ≤80 experiments, grouped D1–D9
│   ├── README.md
│   ├── D1_physics_identities.md … D9_validation_explainability.md
│   └── summary_table.csv
├── Optimized_Codes/                           ← empty; .gitkeep + one-line README
└── Website/                                   ← §13
```

## 4.4 Consolidation/

```
Consolidation/
├── AGENTS.md              ← §12.4 the map + nested-repo register + reference paths
├── README.md
├── MANIFEST.md            ← every moved item: source → destination → size → note
├── .gitignore             ← Oracle, scraped, big CSVs, venvs, __pycache__, zips
├── 00_competition/        ← Overview.txt, Dataset Description.txt, Competition Rules.txt,
│                            the Kaggle HTML, and the Round-1/Round-2 rule docs
├── 01_round1/             ← from AISEHack-2.0/Polymer Prediction/  (minus scraped/, minus venv/)
├── 02_round2/             ← from AISEHack-2.0/Polymer Pred Round 2/
├── 03_round3_working_repo/← the FULL "Polymr" repo, git and all (nested repo)
├── 04_phases/
│   ├── phase2_mechanism_sweep/
│   ├── phase3_clean_stack/
│   ├── phase4_explainability/
│   ├── phase5_score_improvement/
│   └── phase5a_gap_analysis/
├── 05_submissions/        ← every submission CSV ever made + a provenance table
├── 06_oracle_QUARANTINE/  ← Oracle/ + scraped/ + external Tg DBs. gitignored. README explains.
├── 07_gpu_reference/      ← NO bulk copies. Path index + connection recipe + small results
├── 08_research/           ← research notes, web-research digests, novelty ledger
└── 09_handoff/            ← this PLAN.md + the four research docs + PROGRESS.md
```

---

# 5. WORK PACKAGE B — THE SUBMISSION CODEBASE

> Do this **second** (after §14 Consolidation gives you a safe copy), because it pulls
> curated pieces out of the source repo.

## 5.1 What goes in, from where

| destination | source (in the "Polymr" repo) | notes |
|---|---|---|
| `src/pipeline_final.py` | `CODEBASE/pipeline_final.py` | 652 KB; V57 Part A byte-identical to the verified standalone (570,044 chars) |
| `src/evidence_engine.py` | `CODEBASE/evidence_engine.py` | 84 KB |
| `src/featurize.py` | `CODEBASE/featurize.py` | Morgan counts r=2/2048 + 10 stable descriptors |
| `inference.py` | `CODEBASE/inference.py` | **must be rewritten — §5.6** |
| `weights/` | `CODEBASE/weights/polymer_weights.joblib` | 2 MB |
| `submission.csv` | `final_submissions/submission.csv` | the file actually submitted; verify it byte-matches `CODEBASE/submission_v57.csv` and if not, ship `final_submissions/` and say so |
| `requirements.txt` | `CODEBASE/requirements.txt` | keep the version-pin warning comments |
| `outputs/` | curated subset of `CODEBASE/outputs/` | §5.8 |
| `ARCHITECTURE.md` | `CODEBASE/ARCHITECTURE.md` | rewrite per §5.3 |

**Explicitly do NOT copy:** `CODEBASE/feasibility/` (reads the answer key),
`CODEBASE/FEASIBILITY.md`, `build_imputation_variant.py` + `submission_imputation.csv`
(a second route worth +0.0002 = noise; two submissions invites "which one is it?"),
`CODEBASE/__pycache__/`, `CODEBASE/.claude/`, and `pipeline_v57_final.py` (redundant
with `pipeline_final.py` Part A — note the equivalence in ARCHITECTURE.md instead of
shipping 564 KB twice).

## 5.2 README.md — exact section order

Reference style: https://github.com/Vetri-78640/AISEHack-2026 . Target **250–350 lines**.
Rich but not bloated: 4–6 embedded charts, one results table, then link out.

1. **Title + one-line pitch + badges** (python 3.11 · AISEHack 2.0 Round 3 · private LB 0.891).
2. **Project Overview** — 2 paragraphs: the task, what we built, the headline result
   (public 0.917 / private 0.891 / local panel 0.9023 / 14-of-18 evidence groups PASS).
3. **Problem Statement** — the 7-target table (§2.2 with units and n), the metric formula,
   and the sentence that makes it interesting: *"tg is 56% of the rows and 1/7 of the score;
   ei is 3% of the rows and also 1/7 of the score."*
4. **Key Findings (teaser)** — 4 bullets from §2.4 + one embedded chart
   (`outputs/eda/novelty_two_regimes.png`). Link to `FINDINGS.md`.
5. **Dataset** — file table with row counts and SHA-256; the long-format explanation (one row
   = one polymer-property pair); the fact that 100% of SMILES carry exactly two `*`
   endpoints; the auxiliary corpora and **which we actually used** (PI1M yes; smile_r3 **no**,
   with the one-line reason: measured to hurt, P5A-003).
6. **Methodology** — the 5-stage pipeline in prose + the architecture diagram (§5.10).
7. **Model Architecture** — per-target table: which family solves which target and **why**
   (tied to n and to the physics); the feature families; the calibration layer with its exact
   constants (0.20 char-residual multiplier for tg/egc/egb/nc/eps; 1.05 spread gain plus a
   physical clip for ei/eea); seeds (2026).
8. **Explainability, Robustness & Generalization** — the Round-3 themes as a table
   (theme → artifact → measured number) with 3 embedded charts (tg SHAP beeswarm, invariance
   boxplot, generalization ladder).
9. **Results** — per-target R²/MAE/RMSE, LB scores, the scorecard summary **including the
   four honest failures**.
10. **Requirements** — the pinned block **with the numpy/sklearn collapse warning verbatim**.
11. **Usage / Quickstart** — §5.11 commands.
12. **Project Structure** — the tree from §4.3 with one-line annotations.
13. **Future Scope** — 5 bullets from `REMAINING_EXPERIMENTS.md`.
14. **Acknowledgements & References** — organisers + the 8–12 load-bearing citations.
15. **Compliance statement** — official data only, from scratch, fixed seeds, one run, no
    pretrained weights, no external data.

**Keep out of the README:** the full experiment ledger, the failure taxonomy in detail, the
ceiling maths. Those live in `FINDINGS.md`, `Experiment_Logs/` and `Personal/docs/`.

## 5.3 ARCHITECTURE.md vs METHODOLOGY.md — decision

**Ship ONE file: `ARCHITECTURE.md`.** Two overlapping deep-dives is exactly the bloat to
avoid. Build it from the existing `CODEBASE/ARCHITECTURE.md` (already strong) with these
edits:

- **Remove** the two-route / imputation-overlay discussion and every mention of
  `FEASIBILITY.md`, `oracle`, `final_oracle`, and "proxy" in the *scoring* sense.
- **Rename** every "oracle R²" column to "local verification panel R²".
- **Add §0 "Design rationale"** opening with the two-regimes finding (§2.4.1) — the reader
  must understand *why* per-target before reading *what* per-target.
- **Add** an explicit per-target justification table: target → n → chosen family → the
  empirical reason (e.g. *ei: n=222; MLP + GaussianProcess + Tanimoto-KRR, because trees
  saturate below ~300 rows and our own directed-MPNN scored −0.309 on this target*).
- **Keep** the calibration-layer maths verbatim (existing §4.4).
- **Rewrite** the evidence-suite section with the limitations promoted, not buried.
- **Add** a Reproducibility section: one run, seeds, env pin, the collapse warning, expected
  runtime (~2.5 h full; minutes in smoke mode).

If the user still wants a `METHODOLOGY.md`, make it a 15-line pointer file.

## 5.4 FINDINGS.md (nine findings, each with number + file + chart + mechanism)

1. **F1 — Two problems wearing one leaderboard.** Novelty and partner tables
   (EDA_VERIFIED_FACTS §5–§6). Chart: grouped bar, "test polymer in train (any target)" vs
   "(same target)" per target.
2. **F2 — The variance trap.** tg holds 99.986% of pooled TSS; a perfect tg model still only
   reaches mean 0.9172. Chart: log-scale TSS-share bar.
3. **F3 — The physics holds.** Three identities with n, R², violation counts. Charts:
   eps vs nc² scatter with the ionic gap shaded; (ei−eea) vs egc parity plot.
4. **F4 — Tg is noisy and long-tailed.** std 109 °C; only 4 replicate groups (spread ≤11 °C);
   the extreme 10% of rows carry 36.9% of TSS; the top-5% worst-predicted rows carry 55% of
   SSE. Charts: Tg histogram; Lorenz curve of squared error.
5. **F5 — Small-n R² is untrustworthy.** SE(ei)=0.022, SE(eps)=0.024; fixing one row moves ei
   by 0.013. Chart: per-target R² with bootstrap error bars. *This is the finding that most
   impresses a statistically literate panel.*
6. **F6 — Correlations are real but thin.** egc↔egb 0.963 (n=175), nc↔eps 0.918 (n=134);
   tg↔nc 0.849 on **8 polymers** — flagged unreliable. Chart: correlation heatmap annotated
   with n, greyed where n<30.
7. **F7 — The model learned physics, not a lookup.** Flory–Fox 1/n linearity, median R²≈0.99;
   counterfactual direction agreement 27/40, rigidity 12/13.
8. **F8 — Explanations are faithful and invariant.** 0.851 vs 0.043 fidelity; attribution
   cosine 0.95–0.99; activation-patch delta exactly 0.0; layer-1 aromaticity probe R² 0.895.
9. **F9 — Where we are weak, stated plainly.** The four scorecard FAILs with their causes and
   fixes.

A finding without a mechanism is trivia. Every one needs a sentence of chemistry or
statistics explaining *why*.

## 5.5 RESULTS.md

≤120 lines: per-target table; LB numbers; the **train/test-split** validation table produced
by the notebook (§6); the scorecard summary; and an index of every chart in `outputs/` with
a one-line caption. This exists so the README can stay lean.

## 5.6 inference.py — REWRITE REQUIRED (important, read carefully)

The current file resolves a query through: `v57_exact(id)` → `v57_exact(canonical
SMILES)` → `identity(egc = ei − eea)` → `train_label` → `base_model` (compact LightGBM).

**The problem.** For any official test row it returns a *cached* prediction. If a judge types
a test SMILES into the demo and we present a cache hit as "the model predicting", it looks
like a lookup table — the worst possible impression in a round judged on trust. And the
honest fallback quality is much lower than the headline (5-fold OOF: tg 0.897, egc 0.860,
eea 0.884, egb 0.875, nc 0.795, ei 0.773, eps 0.694).

**Required changes.**
1. Add `--mode {model,cached,auto}` with **`model` as the default**: the cache is bypassed
   and the compact model always runs.
2. **Always return and print the resolution source.** Never hide which path fired.
3. Return an **uncertainty estimate and an applicability-domain tier** with every prediction:
   nearest-train Tanimoto, the AD tier, and the conformal half-width for that target. The
   plumbing exists in `outputs/test_predictions_with_intervals.csv`,
   `outputs/ad_analysis_table.csv` and `outputs/reliability_tiers_*.csv`.
4. Document the fallback model's honest per-target OOF R² in `--help` **and** in the README,
   right next to the headline score.
5. Batch mode emits `id,target,source,ad_tier,nn_tanimoto,pi_low,pi_high`.

## 5.7 model.pt — format decision (raise with the user)

The V57 spine is **transductive** — test SMILES participate in the unsupervised feature
construction — so it genuinely cannot be serialised into per-row weights.

- **Do NOT fabricate a `model.pt`.** Ship `weights/polymer_weights.joblib` (exact V57 cache
  by id and by canonical key; partner lookup over 5,920 polymers; identity coefficients
  `ionic_med = 0.6896`, `egb_a = 1.1586`, `egb_b = −1.0437`; one LightGBM per target) plus a
  short `weights/README.md` stating plainly what is and is not inside, and why a single
  tensor file is the wrong container for this architecture.
- **If the user insists on `model.pt`**: export only the compact per-target LightGBM fallback
  to ONNX as `model.onnx`, labelled unambiguously *"portable fallback predictor for novel
  polymers — not the leaderboard pipeline"*. ~20 minutes of work plus a dependency. **Ask
  first.**

## 5.8 outputs/ curation — do NOT dump all 169 files

Ship **~45–55 charts**, foldered, each captioned in `outputs/CAPTIONS.md`.

| folder | include | drop |
|---|---|---|
| `eda/` | the 8–10 **new** EDA charts you produce in §6 Stage 2 | — |
| `training/` | loss/learning curves, OOF-vs-true parity per target (7), residual plots, CV bar chart, seed-stability strip | — |
| `explainability/` | `shap_beeswarm_*.png` (7), `shap_summary_global.png`, 3 of the 7 `fidelity_curve_*`, 2 `local_shap_*` + 2 `shap_force_*`, `explanation_agreement_heatmap.png`, `physics_decomp_eps_shap.png`, `linear_probe_heatmap_tg.png` | the other ~15 local_shap and ~15 shap_force files |
| `robustness/` | `smiles_invariance_boxplot.png`, `attribution_invariance_scatter.png`, `oligomer_invariance_plot.png`, `activation_patch_invariance_plot.png`, `structural_counterfactuals_plot.png`, `consistency_reg_plot.png` | per-target invariance plots |
| `generalization/` | `generalization_ladder_plot.png`, `cv_validation_barplot.png`, `ad_analysis_plot.png`, `conformal_calibration_plot.png`, `tail_performance_plot.png`, `reliability_2d_map.png`, `trustworthiness_radar.png`, 3 of 7 `error_vs_uncertainty_scatter_*` | the `khazana_scatter_*` files — **the word "khazana" names an external database and will be challenged.** Either drop them and rely on the ladder, or rename to `external_holdout_scatter_*` **only if** the panel can be justified without naming a prohibited source. Default: **drop**. |
| root | `TRUSTWORTHINESS_REPORT.html` (de-oracled), `architecture.png`, `scorecard.md` (de-oracled) | — |

Also ship the small evidence CSVs: `shap_top20_per_target.csv`, `fidelity_table.csv`,
`smiles_invariance_per_target.csv`, `attribution_invariance_per_target.csv`,
`cv_validation_table.csv`, `conformal_coverage_table.csv`, `generalization_ladder.csv`,
`ad_analysis_table.csv`, `seed_stability.csv`, `tail_performance.csv`,
`error_uncertainty_correlation.csv`, `linear_probe_results.csv`,
`structural_counterfactuals.csv`, `relation_homologous_series.csv`.

**Do not ship:** `candidate_proxy_ensemble.csv`, `proxy_sweep_*`, `proxy_oof_*.csv`,
`setup_log.txt`, `SESSION_SUMMARY.md` (agent-flavoured), anything in `outputs_and_logs/`.

## 5.9 Experiment_Logs/ — the ≤80-experiment showcase

One markdown file per domain **D1–D9** (§2.5), plus `README.md` and `summary_table.csv`.
Each domain file carries **6–10 experiments** in this table:

| id | mechanism | hypothesis | result (Δ vs incumbent) | verdict | why |

with `verdict ∈ {helped, subthreshold, neutral, hurt, cooled}` and a **2–4 sentence "why"
that gives a mechanism, not a shrug**. Sources: `Personal/TRIALS.md` (§11.3),
`logs/experiments.jsonl` (247 records), `Phase5_Kiro_Score_Improvement/logs/phase5_summary.tsv`
(54 scored), `Phase5A_Gap_Analysis/logs/*.tsv` (37).

**Mandatory inclusions:** the 9 winners and the 11 documented failures in
`RESULTS_ANALYSIS.md` §4b/§4c, plus the three failure *signatures* in §4a, plus 3–4
near-misses (the +0.009 results rejected by a +0.01 gate — they demonstrate statistical
honesty, and they are a great QnA hook).

`README.md` must state: *"A curated subset of N experiments drawn from a programme of
approximately 1,150 logged runs on this machine (and roughly 4,000 across all contributors
and rounds); selection criterion = one representative per mechanism family, plus every result
cited elsewhere in this repository."* Use the real N.

## 5.10 The architecture diagram — you must produce it (nothing like it exists today)

The user's own notes record that previous winners had *"a hand drawn architecture which we
were able to explain end to end"*. Produce `outputs/architecture.png` **and its source**.

Layout: **inputs** (train / test / PI1M) → **Stage A shared representation** (parallel boxes:
RDKit 2D descriptors · Morgan count FPs · character n-grams · Tanimoto kernel ·
Polymer-Genome atomic triples · PI1M character-SVD) → **Stage B per-target leaves** (7 lanes,
each labelled with its n and its chosen family) → **Stage C physics overlays** (three boxes
carrying their equations) → **Stage D assembly** (OOF-NNLS + splice/blend chain) →
**Stage E calibration** (0.20 char-residual on 5 targets; 1.05 spread + clip on ei/eea) →
**submission.csv**. A parallel branch from Stage B/D feeds the **Evidence engine**
(R1 explainability · R2 invariance · R3 reliability · R4 generalization) → scorecard + report.

Annotate every lane with its final R². Colour by target group (experimental / electronic /
optical-dielectric). Tooling preference: (a) Graphviz `dot` — deterministic and
version-controllable; (b) matplotlib patches; (c) draw.io / Excalidraw exported to PNG+SVG.
**Commit the source, not just the PNG.** Produce three sizes: full detail (report),
simplified (slide), and a 3-box version (title slide / README hero).

## 5.11 Quickstart commands (put these in the README verbatim, and make them work)

```bash
# 0. one-time setup (creates .venv with the pinned, load-bearing versions)
bash setup.sh

# 1. reproduce the submission end to end (~2.5 h)
.venv/bin/python src/pipeline_final.py --mode submission --data-dir /path/to/Dataset --out submission.csv

# 2. reproduce the submission AND the full evidence bundle
.venv/bin/python src/pipeline_final.py --mode full --data-dir /path/to/Dataset --out submission.csv --out-dir outputs

# 3. fast demo on tiny data (minutes) — for a live presentation
PHASE4_SMOKE=1 .venv/bin/python src/pipeline_final.py --mode full --smoke --data-dir /path/to/Dataset

# 4. predict a single polymer without retraining
.venv/bin/python inference.py --smiles "*CCc1ccccc1*" --target tg --mode model

# 5. batch predict
.venv/bin/python inference.py --infile /path/to/test.csv --out predictions.csv --mode model

# 6. run the annotated analysis notebook
jupyter lab Sandman_Polymer_Property_Prediction.ipynb
```

`setup.sh` must: find python 3.11, create `.venv`, install `requirements.txt`, then
**assert** `numpy==2.4.6` and `scikit-learn==1.9.0` and fail loudly with the collapse
warning if not. It must also print the activation command.

## 5.12 Strip before committing

- Every `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.codex/`, `.agents/`, `.mcp.json`,
  `PROMPT.md`, `PLAN.md`, `SESSION_SUMMARY.md`, `.cc-writes`.
- `__pycache__/`, `.pytest_cache/`, `.DS_Store`, `.ipynb_checkpoints/`.
- Any absolute path containing `/Users/daver`, `Desktop`, `r3_runtime`, the GPU username or host,
  `100.116.22.29`, or the GPU password.
- Any occurrence of `oracle`, `final_oracle`, `ORACLE_ASSISTED`, `khazana`,
  `polyinfo`, `TgSS`, `test_answers`, or the Round-2 `archive/`.
- The word "proxy" where it means the answer-key proxy. It is fine where it means the
  lightweight explainability proxy models — but define it on first use.


---

# 6. WORK PACKAGE C — `Sandman_Polymer_Property_Prediction.py` (the notebook scaffold)

## 6.1 What the user asked for, precisely

> *"You can keep this as a py file... set up the py file like an ipynb file itself. When you
> are done, I will scaffold the py file into an ipynb and then run everything one by one.
> Use # for comments and headers, and I will need markdown blocks as well... I will basically
> just copy it into blocks of code and markdown to make it an ipynb and then run it. The
> submission one will have the charts and stuff displayed so write it in that format.
> Lastly, ensure that it is easy for me to run, like it automatically activates the venv or
> has instructions to do it."*

So: **one `.py` file, cell-delimited, mechanically convertible to `.ipynb`, with explicit
markdown cells, that runs top to bottom and displays charts inline.**

## 6.2 Cell format — use this exactly

Use the `# %%` percent format (Jupytext / VS Code / PyCharm all understand it, and
`jupytext --to notebook` converts it losslessly — offer the user that one-liner, but the
file must also be hand-copyable):

```python
# %% [markdown]
# ## 2.3  Why Tg and Ei need different models
#
# Tg has 4,143 training rows and 12.3% test-polymer overlap ...

# %%
# ---- code cell ----
fig, ax = plt.subplots(figsize=(9, 5))
...
plt.show()
```

Rules:
- Every markdown cell starts with `# %% [markdown]` and every following line starts with `# `.
- Every code cell starts with a bare `# %%`.
- **Never** let a function definition straddle two cells.
- Every chart-producing cell ends with `plt.show()` so the notebook renders inline, **and**
  saves to `outputs/<folder>/<name>.png` with `dpi=150, bbox_inches='tight'` so the same
  run populates the repo.
- Set a consistent style once, in the setup cell: `plt.rcParams` (figsize, dpi 130, font
  sizes, `axes.grid=True`, `grid.alpha=0.3`), colormap `tab10`/`viridis`, and avoid pure
  red/green pairs (accessibility; the Phase-4 contract already required this).
- Every figure needs a title, axis labels **with units**, and a legend when multi-series.

## 6.3 Environment bootstrap — cell 1 must be bulletproof

The first code cell must:
1. Print the python executable, and the versions of numpy / pandas / sklearn / rdkit /
   lightgbm / shap / matplotlib.
2. **Assert** `numpy == 2.4.6` and `scikit-learn == 1.9.0`, and if they differ print a
   large, unmissable warning that reads: *"ENVIRONMENT MISMATCH — the ei/eea leaf models
   collapse under numpy ≥ 2.5 (ei 0.871 → 0.516). Results below are NOT comparable to the
   reported scores. Run `bash setup.sh` and restart the kernel."*
3. Resolve `DATA_DIR` from, in order: an env var `PPP_DATA_DIR`, `./Dataset`,
   `../Dataset`, `/kaggle/input/...` — and print which one it picked.
4. Create `outputs/{eda,training,explainability,robustness,generalization}`.
5. Set every seed: `random`, `numpy`, `PYTHONHASHSEED`, and pass `random_state=2026`
   (pipeline) / `42` (analysis) consistently — **pick one and document it**.

Also put the venv instructions **in a markdown cell above it**, both forms:
```bash
# option A (recommended): run the notebook with the project kernel
bash setup.sh
.venv/bin/python -m ipykernel install --user --name ppp-r3 --display-name "Polymer R3"
jupyter lab   # then choose the "Polymer R3" kernel

# option B: convert and run headless
jupytext --to notebook Sandman_Polymer_Property_Prediction.py
.venv/bin/jupyter nbconvert --to notebook --execute Sandman_Polymer_Property_Prediction.ipynb
```

## 6.4 The train/test-split protocol — READ THIS TWICE

The user's constraint: *"we cannot use oracle here, so: run the full train.csv to get the
model weights. Then, in the codebase, do a train-test split to get the metrics to show them
and charts whatever is required. Then have commented-out code to do the full train.csv
training. Our submission should be the full best model and csv file that I submitted."*

Implement it as **two clearly separated tracks**:

**Track EVAL (runs by default, produces every metric and chart in the notebook).**
- Split `train.csv` into train/holdout **with `GroupKFold`-style grouping on canonical
  SMILES** so that the same structure never straddles the split. There are 1,063 canonical
  SMILES shared between train and test and duplicated structures within train — a naive
  `train_test_split` leaks and will inflate every number you show.
- Use **80/20** for tg and egc; for the five small targets (n = 221–337) use **repeated
  5-fold grouped CV (5 repeats)** and report mean ± std instead of a single holdout — a
  single 20% holdout of ei is 44 rows and its R² has ±0.05 noise. **Say this in the markdown
  cell**; it is exactly the kind of care the panel rewards.
- Report per target: R², MAE, RMSE, and a bootstrap 95% CI on R².
- **Expect these numbers to be LOWER than the leaderboard numbers** (no test-time partner
  covariates available in a pure train split for some rows). Do not hide the difference —
  add a markdown cell explaining that the leaderboard pipeline is transductive and has
  access to partner labels at inference, which the internal split deliberately withholds.

**Track FULL (commented out, clearly banner-marked).**
- The exact call that regenerates the submitted `submission.csv` from the full
  `train.csv` (`src/pipeline_final.py --mode submission`). Commented out with a banner
  explaining that it takes ~2.5 hours and that the shipped `submission.csv` is its output.

**Anti-gaming guardrail.** The notebook must contain **no** answer-key file, no
`final_oracle`, no `Oracle/` path, no downloaded label set. Every metric shown comes from
`train.csv` splits only. State this in a markdown cell near the top — turning a constraint
into a credibility statement.

## 6.5 Notebook outline — build these cells, in this order

**Stage 0 — Front matter (markdown)**
- Title, team, competition, one-paragraph abstract, the headline result, the compliance
  statement, the reading order, and a table of contents.

**Stage 1 — Setup**
- 1.1 markdown: how to run (§6.3).
- 1.2 code: imports, version assertions, seeds, paths, plot style, `outputs/` creation.
- 1.3 code: load `train.csv` / `test.csv`, print shapes, print SHA-256 of each file.

**Stage 2 — EDA (this is where the user wants substance; ~14 cells, ~10 charts)**

Everything below is already computed and verified in
`Consolidation_Handoff/research/EDA_VERIFIED_FACTS.md` — your job is to *recompute it live
in the notebook and plot it*, not to rediscover it.

| # | cell | chart | the point |
|---|---|---|---|
| 2.1 | dataset shape, long format, parse success (0 failures), 100% of SMILES carry exactly 2 `*` | annotated text + a small bar of star-counts | "this is a polymer dataset" |
| 2.2 | per-target counts train/test | grouped bar, **with a second axis showing "share of the metric" flat at 1/7** | the 56%-of-rows / 14%-of-score asymmetry |
| 2.3 | target distributions | 7-panel histogram + KDE, with mean/median/skew annotated | tg is huge-range and near-symmetric; eps/nc are right-skewed |
| 2.4 | within-target TSS share | horizontal log-scale bar | tg = 99.986% → the multi-task variance trap |
| 2.5 | canonicalisation + overlap | Venn-style bar: 5,920 train / 4,133 test unique / 1,063 shared | grouped CV is mandatory |
| 2.6 | **novelty: two regimes** | grouped bar per target: "polymer in train (same target)" vs "(any target)" vs "exact (smiles,target) pair" | **the single most important chart in the deck** |
| 2.7 | nearest-train Tanimoto distributions | 7 overlaid KDEs or a ridgeline, same-target basis | median 0.55–0.57 for the DFT targets |
| 2.8 | partner availability matrix | heatmap: test target (rows) × available train partner (cols), counts | why cross-property works |
| 2.9 | co-measurement counts + correlation | correlation heatmap **annotated with n**, cells greyed where n < 30 | egc↔egb 0.963 is real; tg↔nc 0.849 (n=8) is not |
| 2.10 | physics identity 1 | `(ei − eea)` vs `egc` parity plot + residual histogram | R² 0.9716, MAE 0.072 eV, n=59 |
| 2.11 | physics identity 2 | `eps` vs `nc²` scatter, ionic gap shaded, plus an ionic histogram | 0/134 violations; std 0.409 vs 1.070 → 2.62× better conditioned |
| 2.12 | physics identity 3 | `egb` vs `egc` with the fitted line | a = 1.1586, b = −1.0437, R² 0.9282 |
| 2.13 | tg tail structure | histogram + Lorenz curve of squared deviation | extreme 10% of rows carry 36.9% of TSS |
| 2.14 | replicate / label noise | table: 4 tg duplicate groups, spreads; 0 for the DFT targets | **and a markdown note that the "2,497 duplicate groups" figure in older docs is Round-2 archive-inclusive, not Round 3** |

**Stage 3 — Featurisation (markdown-heavy, code light)**
- 3.1 markdown: the five feature families and *why each exists* (descriptors = bulk physics;
  Morgan = local substructure; char n-grams = cheap sequence signal but **the only
  non-invariant component**; Tanimoto kernel = smooth interpolation at small n; PI1M SVD =
  label-free representation prior).
- 3.2 code: build the feature matrix for the EVAL track (reuse `src/featurize.py` and the
  evidence-engine feature builder so the notebook and the pipeline agree).
- 3.3 code: feature-count table and a sparsity/variance summary.

**Stage 4 — Model training (EVAL track)**
- 4.1 markdown: the per-target architecture rationale table (§5.3).
- 4.2 code: train the per-target models on the grouped split, with progress printing.
- 4.3 charts: **learning curves** (R² vs training-set fraction, per target — this directly
  answers "would more data help?" and is a great slide), **LightGBM iteration curves**
  (train vs valid loss), and **OOF-vs-true parity plots** with the identity line and the
  fitted slope annotated (the slope is the compression story: mid-band compression is why
  the 1.05 spread calibration exists).
- 4.4 charts: residual plots (residual vs predicted, residual vs nearest-train similarity).
- 4.5 code+chart: **ablation** — retrain per target dropping one feature family at a time;
  bar chart of ΔR². This is the "ablation study" the report format demands and it currently
  exists only as `feature_ablation_results.csv`; make it a first-class chart.
- 4.6 code+chart: **seed stability** across 5 seeds (existing result: 0.9066 ± 0.0018).

**Stage 5 — Assembly, physics overlays and calibration**
- 5.1 markdown: the OOF-NNLS blend, then the identity overlays, then calibration.
- 5.2 code+chart: blend weights per target (stacked bar) — shows which family wins where.
- 5.3 code+chart: before/after the physics overlay, per affected target.
- 5.4 code+chart: before/after calibration; a predicted-vs-true slope table showing the
  1.05 spread gain fixing mid-band compression.

**Stage 6 — Results**
- 6.1 table: per-target R² / MAE / RMSE on the internal split, with bootstrap CIs.
- 6.2 chart: per-target R² with error bars, plus a horizontal line at the mean.
- 6.3 markdown: the honest comparison to the leaderboard numbers and *why they differ*.
- 6.4 chart: score trajectory across the project (baseline → +physics → +blend →
  +calibration → final), a waterfall. **The winning decks all had one of these.**

**Stage 7 — Explainability (R1)**
- 7.1 markdown: why proxy models are used to open a 339-node DAG (and that they retain ~98%
  of the signal: proxy OOF tg 0.909, egc 0.895, egb 0.871, ei 0.800, eea 0.853, nc 0.803,
  eps 0.744).
- 7.2 SHAP global beeswarm per target (7 charts) + a top-20 table.
- 7.3 SHAP local force plots + RDKit `SimilarityMaps` atom colouring for 2 polymers.
- 7.4 **fidelity test** chart: R² drop vs fraction masked, SHAP-top vs random (0.851 vs 0.043).
- 7.5 cross-model agreement heatmap (report ρ = 0.471 honestly, and explain *why* different
  families rank collinear features differently).
- 7.6 physics-decomposed SHAP for eps (nc² channel vs ionic channel).
- 7.7 linear probes on the MLP hidden layers (aromaticity R² 0.895 at layer 1).

**Stage 8 — Robustness / polymer invariance (R2)**
- 8.1 markdown: what invariances a polymer admits (SMILES writing order, canonicalisation,
  cut-point choice, oligomer length, stereo).
- 8.2 code+chart: 500 polymers × 30 randomised SMILES → prediction-std boxplot, per target,
  **split into "graph features only" and "full ensemble"** so the char-n-gram contribution is
  visible. Report the 1σ violation rate.
- 8.3 canonicalisation audit output.
- 8.4 attribution invariance: SHAP cosine across spellings (scatter + per-target table).
- 8.5 oligomer/chain-extension invariance.
- 8.6 **Flory–Fox homologous-series demo**: predicted Tg vs 1/n, fitted line, median R² ≈0.99.
  *This is the "can the model find a relation between different polymer structures?"
  question the organisers explicitly asked.* Give it its own markdown cell and make the chart
  beautiful.
- 8.7 structural counterfactuals: the 4 chemistry edits, direction-agreement table (27/40),
  with the misses shown, not hidden.

**Stage 9 — Reliability and generalization (R3/R4)**
- 9.1 structured CV table + bar chart (random / grouped / scaffold / low-similarity).
- 9.2 **generalization ladder** chart — the staircase.
- 9.3 conformal intervals: coverage table + calibration plot; be explicit that ~45
  calibration rows gives ±4.5% binomial noise.
- 9.4 applicability domain: error vs nearest-train similarity, monotone; AD tiers.
- 9.5 error-vs-uncertainty scatter (3 targets) with the honest ρ values.
- 9.6 tail performance: R² in the top/bottom 10% of each property.
- 9.7 trustworthiness radar.

**Stage 10 — Post-training analysis (the "interesting findings" the user asked for)**
- 10.1 error decomposition: which rows dominate SSE; the top-5% → 37–55% result.
- 10.2 error vs novelty bins; error vs molecular size; error vs aromatic fraction.
- 10.3 per-target bias/calibration check (predicted-vs-true slope and intercept).
- 10.4 outlier gallery: the 6 worst-predicted polymers per target, drawn with RDKit, with a
  sentence of chemistry for each.
- 10.5 **variance analysis for tg** — the user specifically wants this. Show: tg residual
  variance by value band, by novelty bin, and by molecular class; connect it to the
  experimental-noise argument.
- 10.6 markdown: the five headline findings (§2.4), each with its chart reference.

**Stage 11 — Inference demo**
- 11.1 load `weights/polymer_weights.joblib`, predict 3 hand-picked polymers (a flexible
  aliphatic, a rigid aromatic, a fluorinated one), print value + source + AD tier + interval.
- 11.2 markdown: pointer to `inference.py` and to `Website/`.

**Stage 12 — Reproducing the submission (commented out)**
- 12.1 markdown banner: what this does, how long it takes, what it produces.
- 12.2 the commented-out full-train call and the submission-contract check
  (4,940 unique ids, all finite, columns exactly `id,target`).

**Stage 13 — Conclusions, limitations, future work (markdown)**

## 6.6 Chart inventory (target ≥ 40 figures)

Minimum by stage: EDA ≥ 10 · training/ablation ≥ 8 · results ≥ 3 · explainability ≥ 12 ·
robustness ≥ 6 · generalization ≥ 7 · post-analysis ≥ 6. Chart ideas worth stealing from
strong NeurIPS/Kaggle write-ups (the user asked for this explicitly):

- **parity plots with marginal histograms** (joint plot) instead of bare scatter;
- **calibration/reliability curves** with a shaded binomial confidence band;
- **learning curves with a shaded ±1σ band** over repeats;
- **waterfall / cascade** for the ablation and the score trajectory;
- **ridgeline plot** for the 7 nearest-neighbour-similarity distributions;
- **Lorenz curve / cumulative-error curve** for tail concentration;
- **radar** for the trustworthiness axes;
- **heatmap annotated with n** for correlations (grey out low-n cells);
- **UMAP/t-SNE of the feature space coloured by target value**, with test points overlaid —
  a single picture that shows train/test coverage and the novelty story
  (use only train+test features; no external data);
- **per-atom SHAP molecule renders** (RDKit `SimilarityMaps.GetSimilarityMapFromWeights`).

## 6.7 Hard requirements for the file

- Runs top to bottom on a clean machine after `bash setup.sh`, with no manual edits.
- Total runtime for the EVAL track: **target ≤ 30 minutes** on the Mac. If SHAP over all
  targets is slower, subsample and say so in the markdown.
- No network access. No file read outside `Dataset/`, `outputs/`, `weights/`.
- `grep -in "oracle\|khazana\|polyinfo\|/Users/\|100\.116\|vishwa" Sandman_*.py` must return
  **nothing**.
- Every number printed in a cell that also appears in `README.md` / `FINDINGS.md` /
  `RESULTS.md` must match. Add a final cell that dumps all headline metrics to
  `outputs/notebook_metrics.json` so the docs can be cross-checked mechanically.


---

# 7. WORK PACKAGE D — `Personal/Midnight_Report/`

## 7.1 What must happen physically

1. **Move** `Personal/Sample Reports/` → `Personal/Midnight_Report/Sample Reports/`
   (the user explicitly asked for this move). Six files:
   `Final Submission - Achievers.md`, `Final Submission Report Template_ AI for Science &
   Engineering_team_triverse.md`, `Final submission Report.md`,
   `RuVision_FinalSubmission_Report.docx.md` (currently has no headings — probably an empty
   or image-only export; note that in the analysis), `Final_Submission_Report .pdf`,
   `VibeCoders_Final_Submission_Report_3pg.pdf`.
2. Write `Personal/Midnight_Report/analysis/SAMPLE_REPORT_ANALYSIS.md` — the structural
   analysis (§7.2). The two PDFs should be text-extracted (`pdftotext` or
   `python -m pypdf`) into `analysis/extracted/` so the format analysis is complete;
   if extraction fails, say so rather than guessing.
3. Write `PROMPT_10PAGE.md` and `PROMPT_3PAGE.md` (§7.4, §7.5).
4. Write `REPORT_STYLE_GUIDE.md` (§7.3).

## 7.2 The template, distilled from the samples (this IS the required format)

All the strong samples share a fixed 10-section skeleton. **Follow it exactly** — the
organisers issue a template and the judges read against it.

```
Header block:  Project Title · Team Name · GitHub Repository link · Model Weights link
 1. Executive Summary (Abstract)                — one dense paragraph, no bullets
 2. Final Problem Formulation over and above the baseline
       - Target Phenomena / Physics
       - The "Science Gap"
       - Datasets used
 3. Architecture & Technical Novelty (Finalized)
       - Detailed strategy
       - Hard constraints
       - Soft constraints
       - Hyperparameter & training evolution
 4. Quantitative Performance & Benchmarks       — baseline → final, with the delta
 5. Salient Visualizations ("The Proof")
 6. Ablation Studies (explicitly flagged "very important" in the template)
 7. Scientific Insights & Interpretability      - Discovery / Interpretability
 8. Robustness & Scalability                    - Generalization / Computational efficiency
 9. Limitations & Future Roadmap                - Known failure modes / The path forward
10. Individual Contributions & References       - Team roles / References
Appendices (do not count toward the page limit)
Closing declaration: "We, team <NAME>, have made our submissions wholly based on our own
efforts and have not taken help from third parties / members not part of the team."
```

Observed conventions worth copying:
- **Prose paragraphs, not bullet soup** in §1, §4, §7, §8, §9. The winning reports read like
  a paper, not a checklist.
- **Every claim carries a number.** ("0.1747 → 0.3245 after fixing the temporal direction.")
- **A named "critical breakthrough" narrative.** One or two moments where something specific
  was discovered and the score jumped. Ours: (a) the two-regimes novelty discovery that
  forced per-target design, (b) the ionic decomposition (+0.0666 on eps), (c) the
  difficulty-stratified measurement that explained the public/private gap.
- **Failures are stated in §6 with numbers and a mechanism.** This is where we are strongest.
- **An "Engineering Journey" appendix** listing Challenge → Issue → Solution.
- **A GenAI-disclosure note.** The Achievers report has one; the presentation analysis
  recommends it. Include an honest one.

## 7.3 `REPORT_STYLE_GUIDE.md` — write this file

Contents: the section skeleton above; the canonical numbers (§2.3) as a copy-paste block;
the D1–D9 taxonomy; the vocabulary rules (no "oracle"; "local held-out verification panel";
"evidence suite" not "Phase 4"); figure-caption format; citation format; the declaration
text; and a "banned phrases" list (*"the model learned the features"*, *"state of the art"*
without a citation, *"we achieved"* without a number).

## 7.4 `PROMPT_10PAGE.md` — specification of the prompt file

This file is executed by a future agent to *generate the whole report*. It must be
self-contained enough that the agent needs no other instructions. Structure it as:

**A. Role and objective.** "You are producing the Final Submission Report for AISEHack 2.0
Round 3, team Sandman. Output a single markdown file `Midnight_Report/REPORT_10PAGE.md`,
~10 pages excluding appendices (≈5,000–6,500 words), following the exact section skeleton in
`REPORT_STYLE_GUIDE.md`."

**B. Source manifest — the exact files to read, with what to take from each.** List them
explicitly (this is the part that makes the prompt work):

| read | take |
|---|---|
| `Personal/docs/00_INDEX.md` | orientation |
| `Personal/docs/01_task/` | problem statement, metric, constraints |
| `Personal/docs/02_domain/` | the physics of the seven targets, the science gap |
| `Personal/docs/03_eda/` | every dataset number and chart |
| `Personal/docs/04_experiments/` | the D1–D9 ledger, ablations, failures |
| `Personal/docs/05_architecture/` | the pipeline description and the diagram |
| `Personal/docs/06_results/` | scores, per-target tables, the public/private analysis |
| `Personal/docs/07_explainability/` | SHAP, fidelity, probes, counterfactuals |
| `Personal/docs/08_robustness/` | invariance, augmentation, Flory–Fox |
| `Personal/docs/09_generalization/` | ladder, CV, conformal, AD, tails |
| `Personal/docs/10_gaps_and_future/` | limitations, roadmap |
| `Personal/Research/INDEX.md` + `CITATIONS.bib` | references |
| `<codebase>/outputs/` + `Personal/docs/12_assets/` | figures (reference by relative path) |
| `Personal/TRIALS.md` | Appendix A content |
| `Phase5A_Gap_Analysis/HUMAN_REPORT.md` (via Consolidation) | Appendix B content |

**C. Content mandates — what MUST appear.**
- The exact canonical numbers from §2.3; never a number not present in the sources.
- The five headline findings (§2.4), at least three of them in §1 (the abstract).
- The architecture diagram in §3, and the ablation waterfall in §6.
- At least 5 figures with numbered captions (Figure 1..N) and in-text references.
- The **four honest limitations** (scorecard FAILs) in §9, each with its cause and its fix.
- The public→private gap analysis in §4 or §8 — it is our best methodology story.
- ≥ 15 references in §10, all from `Personal/Research/`, no invented citations.
- Both appendices (§7.6).
- The closing declaration.

**D. Constraints.** No "oracle"; no external-data claims; no path leakage; no fabricated
numbers; British/American spelling consistent; no bullet-only sections in §1/§4/§7/§8/§9;
tables in markdown; every figure path must exist.

**E. Output contract.** One file; a front-matter block with title/team/repo/weights links;
then the 10 sections; then the appendices. Finish with a self-check list the agent must
tick: numbers cross-checked against `outputs/notebook_metrics.json`, every figure path
resolved, no banned words present (`grep` command included), section skeleton complete.

**F. A worked example** for one section (write §2 in full, ~350 words) so the generating
agent has a concrete tone target.

## 7.5 `PROMPT_3PAGE.md`

Same structure, but: ~1,800–2,200 words; keep sections 1, 2, 3, 4, 6, 7, 9, 10; fold §5 into
§4 and §8 into §7; **maximum 3 figures** (architecture, ablation waterfall, generalization
ladder); the ablation table becomes 6 rows; references trimmed to the 8 load-bearing ones.
Everything else moves to the appendices. Explicitly instruct: *"if forced to cut, cut prose
from §3 before cutting any number from §4 or §6."*

## 7.6 The two appendices — freeze their definitions

**Appendix A — Experiment Log by Domain.** Organised by **D1–D9** (§2.5), same taxonomy as
`Experiment_Logs/` and the presentation. For each domain: a 6–10 row table (mechanism,
hypothesis, result, verdict) plus a 3-sentence "what we learned". End with the three failure
signatures (§4a of `RESULTS_ANALYSIS.md`): *"126 experiments, one number"* (no-op
detection), *"Phase 5 measured the wrong ladder"* (baseline mismatch), *"deep chains amplify
variance"* (V53 0.838 standalone; 19.5 °C chain drift).

**Appendix B — The Mathematical Score Ceiling.** This is the "max cap" the user wants,
and the material already exists in `Phase5A_Gap_Analysis/HUMAN_REPORT.md`. It must contain:
1. **The metric identity.** Mean-of-per-target-R² = 0.9023 vs pooled-R² = 0.9370 on the
   *same* predictions. Show the algebra: pooled R² is dominated by tg because tg carries
   99.986% of the pooled TSS, so pooling silently reweights the problem.
2. **The Tg-alone bound.** With the other six frozen, R²(tg) = 1 gives mean **0.9172**.
   Therefore Tg alone cannot reach 0.92+; state this as a theorem with its one-line proof.
3. **The per-target standard errors** (SE(ei)=0.022, SE(eps)=0.024, SE(nc)=0.020,
   SE(eea)=0.014, SE(egb)=0.012, SE(tg)=0.007, SE(egc)=0.006) and the corollary: on the small
   targets, **any measured improvement below ~0.04 is inside 2 SE** and is not bankable
   without repeated folds. Derive the SE (delta method / bootstrap over rows).
4. **Single-row leverage.** Fixing the worst row moves ei by +0.013, eps +0.011, nc +0.010,
   tg +0.003. Corollary: on a 148-row target, one hard polymer is worth more than a month of
   architecture search.
5. **The label-noise bound on Tg.** Experimental Tg reproducibility across methods and
   molecular-weight variation is of order 5–15 °C. With Tg std = 109.08 °C, a noise floor of
   σ_noise gives R²_max = 1 − (σ_noise/109.08)². At σ_noise = 15 °C ⇒ **R²_max ≈ 0.981**;
   at the *effective* noise implied by the observed per-difficulty spread, ≈**0.92**.
   Present both, and be explicit that the second is empirical, not derived.
6. **The composite ceiling.** Combine: tg ≤ ~0.92 (noise), the six DFT targets bounded by
   data scarcity (n = 148–224 test rows, SE 0.012–0.024). Conclusion: **≈0.93 ± 0.01 is the
   practical ceiling for a rules-compliant, single-run pipeline on this data**, and our 0.891
   private is ≈96% of it. Say clearly which parts are proven and which are estimated —
   an over-claimed bound is worse than no bound.
7. **What 0.935 would have required** (the exact per-target profile table).

---

# 8. WORK PACKAGE E — `Personal/Presentation/`

## 8.1 Physical moves

Move `Personal/Sample Presentations/` → `Personal/Presentation/Sample Presentations/`
(9 PDFs + `contents.md`). **`contents.md` is the analysis of what made the winning decks
win — it is the single most valuable file in that folder.** The user noted the images may not
be extractable; do not burn time trying to OCR the PDFs. Use `contents.md` as the primary
source and note in the analysis that the PDFs were not machine-read.

## 8.2 `SLIDE_PLAN.md` — the deck, slide by slide (5–6 minutes)

The user's own flow, refined against the winning-deck blueprint in `contents.md`:
*problem → research gap → experiments (fast) → EDA findings that inspired the pipeline →
architecture walk-through → explainability/robustness/generalization → live website demo →
leaderboard → future scope → conclusion with links.*

**Timing reality check: 5–6 minutes is ~9 slides at ~35 s each.** Do not plan 15.

| # | slide | ~time | content | visual |
|---|---|---|---|---|
| 1 | **Title & Salient Contributions** | 20 s | team, theme; three one-line boxes: **f1 Modelling strategy** ("per-target physics-guided ensemble over shared molecular representations"), **f2 Training** ("grouped CV, OOF-NNLS assembly, physics overlays, spread calibration, seed 2026, single run"), **f3 Result** ("private LB 0.891; 14/18 trust criteria PASS; invariance std ≤0.23%") | 3-box architecture mini-diagram |
| 2 | **The problem & the gap** | 45 s | 7 properties from one SMILES; **the gap: polymer informatics reports accuracy, not trust — invariance is asserted, not measured; explanations stop at global feature importance; nobody reports fidelity or attribution stability** | the 7-target table + one gap bullet list |
| 3 | **The EDA finding that decided the architecture** | 60 s | the two regimes: 98% of DFT test polymers are in train under a *different* property, 0 exact label pairs; tg only 12.3%. Plus the variance trap (tg = 99.986% of TSS but 1/7 of the score) | **chart 2.6** (novelty two regimes) + the TSS bar |
| 4 | **Architecture** | 60 s | the 5-stage diagram; per-target lanes annotated with n and R²; the three physics identities with their equations and measured R² | `outputs/architecture.png` (simplified) |
| 5 | **Experiments & ablations** | 45 s | the D1–D9 grid as a compact matrix, then the waterfall: baseline → +physics → +blend → +calibration → final; and a **"what failed" strip**: GNN −0.309 on ei · 9 SSL variants all ≤ control · Lorentz–Lorenz worse than n² | ablation waterfall + failure strip |
| 6 | **Explainability & invariance (the Round-3 themes)** | 60 s | fidelity 0.851 vs 0.043 · attribution cosine 0.95–0.99 · activation-patch delta 0.0 · layer-1 aromaticity probe R² 0.895 · **Flory–Fox: predicted Tg vs 1/n is linear, R²≈0.99** | SHAP beeswarm + invariance boxplot + the Flory–Fox line |
| 7 | **Generalization & knowing when to doubt** | 40 s | the ladder staircase; conformal intervals on every prediction; AD tiers; and the public/private forensics (0.9023 local → 0.891 private, predicted to within 0.0004) | ladder plot + the three-number table |
| 8 | **Live demo** | 45 s | the website: paste a SMILES, pick a target, get value + interval + AD tier + SHAP atom map | screenshot fallback **always prepared** |
| 9 | **Results, future scope, links** | 35 s | LB table; the ceiling statement ("≈0.93 practical ceiling; we are at ~96% of it"); 4 future bullets; QR codes / links to repo, report, website | LB bar + QR block |

**Backup slides (not counted, kept after the end):** the ceiling maths; the full ablation
table; the per-target justification table; the conformal coverage caveat; the 31 unmatched
structures; the correlation heatmap with n; the counterfactual misses.

## 8.3 `PROMPT_PRESENTATION.md`

Same shape as the report prompt: role · source manifest (docs/, outputs/, SLIDE_PLAN.md,
contents.md) · per-slide content mandates with the exact numbers · visual specs (which chart
file goes on which slide) · the 5–6 minute timing budget · output format. Support **two output
modes** and say which the user should pick:
- **Marp / reveal.js markdown** (`PRESENTATION.md` → PDF/HTML) — version-controllable,
  fast to regenerate, works offline. **Recommended.**
- **A PPTX built with `python-pptx`** — if the organisers mandate their template. In that
  case the prompt must instruct the agent to read the organiser template PDF in
  `Sample Presentations/` (`ANRFAISEHack_Template_FinalePresentation.pdf`) and match its
  section titles.

## 8.4 `SPEAKER_NOTES.md`

For each slide: the exact 2–4 sentences to say, the one number to land, and the transition
line. Plus a **90-second version** and a **3-minute version** for when the panel cuts you
short. Plus five "if they interrupt with X, say Y" pairs.

## 8.5 `DEMO_SCRIPT.md`

The live-demo runbook: which three polymers to paste (one flexible aliphatic → low Tg, one
rigid aromatic → high Tg, one fluorinated → high Ei), what to point at, what the fallback is
if the network/laptop fails (a recorded GIF plus static screenshots committed to the repo),
and the exact commands to start the site. **Rehearsed timing: 45 seconds.**

## 8.6 QnA preparation

`Personal/docs/11_qna/` is the substance (§10). `Presentation/SPEAKER_NOTES.md` carries
only the top 10 one-liners. The user's note is explicit: *"don't underestimate the panel...
they'll grill you on the architecture, why you chose this approach, why you didn't choose
that one."* Every "why not X" must have a **measured** answer, not an opinion.


---

# 9. WORK PACKAGE F — `Personal/Research/` and `Personal/Research_Paper/`

## 9.1 What already exists (do NOT redo)

- `Consolidation_Handoff/research/BIBLIOGRAPHY_SEED.md` — 20 structured citations already
  extracted from `Phase5_Kiro_Score_Improvement/REFERENCES.md` (561 lines), 8 live-verified
  URLs found this session, the 23-reference list already link-checked inside our own Round-2
  paper, and a **citation → design-decision mapping table**.
- `Consolidation_Handoff/research/POLYMER_DOMAIN_PRIMER.md` — the domain science and the
  research-gap analysis, ready to be split into `docs/02_domain/`.
- `score_discrepancy/NEW_EXPERIMENTS.md` (3,742 lines) — a full ML-workflow blueprint with
  inline citations; mine it, do not rewrite it.
- `research/web-research-polymer-methods-20260826.md` and
  `research/web-research-kaggle-strategy-20260826.md` in the source repo.

## 9.2 What to build

`Personal/Research/`:
```
Research/
├── INDEX.md                 ← master table: id · title · venue/year · url · verified_on ·
│                               which claim it defends · strength (primary/supporting)
├── CITATIONS.bib            ← BibTeX for the report and the paper
├── 01_polymer_informatics/  ← Polymer Genome, Kuenneth multi-task, polyBERT, TransPolymer,
│                               PolyCL, polymer-chemprop, PolyMetriX, PolyMon, RadonPy
├── 02_property_physics/     ← Tg (Bicerano, Fox–Flory, free volume), band structure,
│                               dielectric/refractive (Maxwell, DFPT, Lorentz–Lorenz,
│                               Moss/Ravindra/Penn)
├── 03_ml_methods/           ← tree-vs-deep on tabular, D-MPNN, GIN, GAT, GPR/KRR, NNLS
│                               stacking, isotonic calibration, scaffold splits
├── 04_explainability/       ← SHAP, TreeSHAP, ROAR/fidelity, linear probes, causal tracing
├── 05_uncertainty/          ← conformal, cross-conformal, applicability domain, deep
│                               ensembles, MC dropout
├── 06_invariance/           ← SMILES enumeration/augmentation, canonicalisation, equivariance
└── 07_competition_context/  ← NeurIPS 2025 Open Polymer Prediction write-ups, Kaggle
                                solution posts, the AISEHack past-edition analysis
```

**One file per source**, named `<firstauthor><year>-<slug>.md`, with fields:
`title / authors / venue / year / url / verified_on / one_paragraph_summary /
key_numbers / what_it_defends / how_we_use_it / caveats_or_disagreements`.

The `caveats_or_disagreements` field is the one that wins QnA points: e.g. *"polyBERT
reports R² 0.80 across 29 properties after pretraining on ~100M hypothetical polymers; our
corpus is 6M and our compute budget ~100× smaller, and our matched control shows the
representation loses to supervised features — we cite it as the method we tested, not as
support for our design."*

## 9.3 Searches still to run

Run the 15 queries listed in `BIBLIOGRAPHY_SEED.md` §6 and file each result. Highest
priority, in order:
1. **Grinsztajn/Oyallon/Varoquaux, "Why do tree-based models still outperform deep learning
   on typical tabular data?" (NeurIPS 2022)** — this is the single most load-bearing missing
   citation; it defends the entire architecture choice.
2. **Lundberg & Lee SHAP (NeurIPS 2017)** and **TreeSHAP (Nat. Mach. Intell. 2020)**.
3. **Hooker et al., ROAR (NeurIPS 2019)** — defends the fidelity-by-masking test.
4. **Angelopoulos & Bates, conformal prediction tutorial** and **Vovk cross-conformal (2015)**.
5. **NeurIPS 2025 Open Polymer Prediction Challenge** winning write-ups — the Round-3 themes
   clearly echo it, and the panel will know it.
6. **OECD / applicability-domain guidance** for the AD tiers.
7. **Alain & Bengio, linear classifier probes (2016)**.
8. Moss / Ravindra / Penn original gap–index papers (to cite what we *disproved*).

## 9.4 `Personal/Research_Paper/` — the Round-2 paper

Pull the whole tree from the GPU laptop (read-only copy):
```bash
# from the Mac; see SOURCE_INVENTORY.md §D for the SSH_ASKPASS recipe
scp -r vishwa@100.116.22.29:'~/Desktop/AISEHack-2.0/Polymer_Research_Paper' \
      "<dest>/Personal/Research_Paper/"
```
It contains `drafts/paper_draft_v1.md` (~10,100 words, 5 figures, 5 tables, 2 appendices),
`drafts/figures/` (5 PNGs), `latex/paper.tex` → `paper.pdf` (IEEEtran conference, 8 pages,
23 live-verified references), `experiments/code/` (the ChemBERTa control experiment), and
`RESEARCH_PAPER_PROMPT.md`. Also copy `AISEHack-2.0/Polymer Pred Round 2/Round 2
Submissions/paper.pdf` from the Mac.

Then write `Personal/Research_Paper/README.md` covering:
- the title and the one-sentence thesis (*"cross-contributor replication of negative results
  is strong evidence for a genuine performance ceiling"*);
- **the publication gate**: the draft states publication requires explicit sign-off from the
  competition hosts. Flag this prominently — do **not** put the paper on a public GitHub or
  cite it as published;
- **what is safely presentable now**: the ChemBERTa control (frozen 0.751 / fine-tuned 0.784
  vs tree baseline 0.810) is a genuinely strong "we tested pretrained foundation models and
  they lost" slide, and it was run *outside* the competition so it breaks no rule — but say
  that clearly;
- what would need updating for a Round-3 version (the explainability/invariance suite is new
  material and is arguably the stronger paper).

---

# 10. WORK PACKAGE G — `Personal/docs/` (the comprehensive knowledge base)

> The user's brief: *"THIS IS HIGHLY, HIGHLY COMPREHENSIVE... The goal is to be prepared for
> absolutely anything, be it inside our experiment scope or outside of it."*

Every doc must be **self-contained enough to paste into an agent prompt**, must cite the file
its numbers came from, and must end with a **"Questions this answers"** list.

## 10.1 `00_INDEX.md`
A routing table: question → document. Plus a "canonical numbers" block (§2.3), the D1–D9
taxonomy, and the vocabulary rules. Anyone (human or agent) starts here.

## 10.2 `01_task/`
- `competition.md` — hosts, timeline, rules, submission mechanics, the Round-3 themes
  verbatim from `Competition_Details/Overview.txt`, and what changed from Round 2
  (archive removed; smile_r3 added; explainability + invariance now judged).
- `metric.md` — the mean-of-R² definition, the per-row-weight asymmetry, the pooled-vs-mean
  identity (0.9023 vs 0.9370), and the leverage arithmetic (+0.01 on any target = +0.00143).
- `constraints.md` — official data only, no pretrained weights, single-run reproducibility,
  seeds, sharing with hosts, 3/day + 2 final.
- `judging.md` — the five judged axes (score, explainability, invariance, methodology,
  proven generalization) mapped to our evidence artifacts.

## 10.3 `02_domain/` (source: `POLYMER_DOMAIN_PRIMER.md`, split and expanded)
- `what_is_a_polymer.md` — repeat units, PSMILES, the two `*` endpoints, why the molecule
  is not the material, cut-point degeneracy, tacticity/Mw/crystallinity as missing variables.
- `target_tg.md` — the physics, the five structural drivers with example polymers and their
  real Tg values, Fox–Flory, free volume, measurement variability, the practical ceiling.
- `target_bandgaps.md` — Egc vs Egb, conjugation, the affine relation and its residual.
- `target_ei_eea.md` — band edges, the identity, Mulliken electronegativity, substituent
  effects.
- `target_nc_eps.md` — Maxwell n² relation, DFPT electronic+ionic split, polar-group
  chemistry, and **why Lorentz–Lorenz / Moss failed for us, with numbers**.
- `relationships.md` — the full correlation story with the n caveats; which relations are
  usable and which are anecdotes.
- `research_gaps.md` — the five gaps from the primer §4, each with "what the literature
  does / what is missing / what we did about it".

## 10.4 `03_eda/` (source: `EDA_VERIFIED_FACTS.md`)
- `dataset_overview.md`, `per_target_statistics.md`, `novelty_and_overlap.md`,
  `partner_availability.md`, `physics_identities.md`, `correlations.md`,
  `label_noise_and_replicates.md`, `tails_and_outliers.md`, `smiles_corpus.md`
  (PI1M and smile_r3: what they are, what we tried, what happened).
- Each carries its chart(s) in `docs/12_assets/eda/`.
- **Include the Round-2 vs Round-3 correction note** about duplicate Tg groups (4, not 2,497).

## 10.5 `04_experiments/`
- `00_overview.md` — the campaign map (Round 1 ~108 cycles · Round 2 375 clean · R3 Phase 2
  151 · Phase 3 282 · R3 main loop 247 · Phase 5 55 · Phase 5A 37 · Phase 4 38 analysis
  scripts) with where each lives.
- `D1_physics.md` … `D9_validation.md` — one per domain, the **full** ledger (this is the
  place with no 80-experiment cap), each entry: id · mechanism · hypothesis · protocol ·
  result · verdict · **why** · where the artifacts live.
- `what_worked.md` — the 9 ranked winners with mechanisms.
- `what_failed.md` — the 11+ documented failures with numbers, plus the three failure
  signatures, plus the "gate rejected real signal" analysis (~+0.06 of summed signal
  rejected by an unpassable +0.01 bootstrap gate at n=222).
- `phase5_postmortem.md` — why 55 experiments never beat V57: they ran on a lighter
  baseline (0.8257) that was 0.077 below the champion, so relative gains never transplanted.
  **This is an honest, instructive story about experimental design — write it properly.**
- `negative_results_value.md` — the argument (from our own paper) that cross-contributor
  replication of negative results is evidence of a genuine ceiling.

## 10.6 `05_architecture/`
- `pipeline_overview.md` — the 5 stages, the diagram, the data flow.
- `feature_families.md` — each family, what it captures, its cost, its invariance status.
- `per_target_design.md` — the justification table (target → n → family → evidence).
- `physics_overlays.md` — the three identities as implemented, with guards.
- `assembly_and_calibration.md` — OOF-NNLS, the splice/blend chain, the exact calibration
  constants, and *why* spread calibration exists (mid-band compression).
- `design_decisions.md` — a decision log: decision · alternatives considered · evidence ·
  who/when. **This is the QnA goldmine.**
- `known_weaknesses.md` — the 339-node depth and its variance cost; the transductive design
  and what it means for deployment; the string-feature non-invariance.

## 10.7 `06_results/`
- `scores.md` — the canonical table, all panels explained, the vocabulary rule.
- `public_private_analysis.md` — the full forensics (§2 of `RESULTS_ANALYSIS.md`).
- `ceiling_analysis.md` — Appendix B material (§7.6).
- `per_target_deep_dive.md` — one section per target: score, error profile, what limits it,
  what would move it.
- `score_history.md` — the trajectory with dates and versions.

## 10.8 `07_explainability/`, `08_robustness/`, `09_generalization/`
One doc per evidence method: **what question it answers · how it is computed · the result ·
the artifact file · how to explain it in 30 seconds · the honest caveat.** Cover: global SHAP,
local SHAP + atom maps, fidelity, cross-model agreement, physics-decomposed SHAP, linear
probes, causal tracing, activation patching, attribution patching, counterfactuals; SMILES
invariance, canonicalisation audit, attribution invariance, oligomer invariance, stereo
invariance, consistency regularisation, augmentation, the Flory–Fox relation demo; structured
CV, the ladder, conformal + cross-conformal, AD, seed stability, tail performance,
shift-aware conformal, reliability tiers.

## 10.9 `10_gaps_and_future/`
- `limitations.md` — the four scorecard FAILs, the transductive constraint, the deep-chain
  variance, the Tg noise floor, the 31 unmatchable structures.
- `future_work.md` — mirrors `REMAINING_EXPERIMENTS.md` but with technical detail.
- `if_we_had_more_data.md` — what specifically would move each target.
- `deployment.md` — what it would take to serve this model for real (the inference-ladder
  redesign, retraining cadence, AD-gated abstention).

## 10.10 `11_qna/` — write ≥ 60 Q&A pairs, grouped

Groups and minimum counts: **architecture (12)** · **why-not-X (10)** · **data & EDA (8)** ·
**physics/domain (10)** · **metrics & statistics (8)** · **explainability (8)** ·
**robustness/invariance (6)** · **generalization/uncertainty (6)** · **process & tooling (5)**
· **hostile/awkward (7)**.

Format per entry: **Q** · **30-second answer** · **the number** · **the backing file** ·
**follow-up they might ask** · **what NOT to say**.

The ten hostile ones must include, at minimum:
1. *"98% of your test polymers are in your training file — explain why that isn't leakage."*
2. *"Your public score was 0.917 and private 0.891. Did you overfit the leaderboard?"*
3. *"A 339-node ensemble isn't a model, it's a pile. How is that science?"*
4. *"You said the model is invariant, but your char n-gram features aren't. Which is it?"*
5. *"Your cross-model explanation agreement is 0.47. Doesn't that mean your SHAP story is
   arbitrary?"*
6. *"Why didn't you use the 5.97M-molecule dataset the organisers gave you?"*
7. *"How much of this did an AI write?"*
8. *"Your conformal coverage fails on four targets."*
9. *"Isn't ei = egc + eea just cheating with the answer's siblings?"*
10. *"What would you do differently with another week?"*

**Answer 6 carefully and honestly:** we tried it at scale (P5-330 large-scale MLM,
P5-332 contrastive InfoNCE, P5-333 500k multitask SSL, P5-016 SVD, P5-286 deep teacher,
P5-291 topological SSL, P5-331 multi-teacher distillation) plus 9 PI1M variants in Round 2,
and a matched supervised control beat every one; the decisive measurement was an MLM frozen
linear probe scoring **0.651 against a random-initialisation control at 0.708**. We report it
as a negative result rather than shipping a component that does not help.

## 10.11 `12_assets/`
Every chart used by any doc, in subfolders mirroring the doc sections, plus
`ASSET_MANIFEST.md` (filename · what it shows · which doc/slide uses it · which script
generated it · date). **A chart with no generating script is a liability** — if a judge asks
"how did you compute that", you need the script.

---

# 11. WORK PACKAGE H — `Personal/` top-level files

## 11.1 `FINDINGS.md`
Same nine findings as §5.4 (the codebase copy), **plus** the internal-only ones:
the public/private forensics, the difficulty-stratified Tg table, the 31 unmatchable
structures, the ceiling arithmetic, and the "126 experiments, one number" no-op discovery.
Add a short header: *"the codebase FINDINGS.md is the public subset of this file."*

## 11.2 `STORY.md`
Rewrite the existing `STORY.md` (it is currently Phase-4-only and GPU-path-flavoured) into
the **full 5–6 minute narrative** matching `Presentation/SLIDE_PLAN.md`, in four acts:

- **Act 0 — The setup (30 s).** Seven properties, one string, and a metric that makes a
  148-row target worth as much as a 2,763-row one.
- **Act 1 — The finding that decided everything (60 s).** Two regimes. Show the chart. *"We
  stopped trying to build one great model and started building the right model per target."*
- **Act 2 — The architecture and the physics (90 s).** Feature stack, per-target lanes, three
  identities with their measured R², assembly, calibration. Name the numbers.
- **Act 3 — Trust (120 s).** Explainability (fidelity 0.851 vs 0.043; layer-1 aromaticity
  0.895), invariance (std ≤0.23%, attribution cosine 0.95–0.99, patch delta 0.0), the
  Flory–Fox relation demo, the generalization staircase, conformal intervals, and the
  public/private prediction. Then the honest failures.
- **Act 4 — The demo and the close (60 s).** Website, leaderboard, the ceiling statement,
  future scope, links.

Keep the existing "three sentences we want judges to remember" device — it is good. Update
it so sentence 3 is about **knowing when to distrust the model**, which is the Round-3 theme.

## 11.3 `TRIALS.md`
Restructure the existing 452-line `TRIALS.md` into the D1–D9 taxonomy (§2.5), with:
- **Part 1 — The showcase (top of file):** exactly the experiments that appear in the
  codebase `Experiment_Logs/`, the report Appendix A and the presentation, so the three
  never diverge. Mark each with where it is shown.
- **Part 2 — The extensive list:** everything else, per domain, with reference links to the
  experiment directories and log lines (Mac paths for Mac work, GPU reference paths for GPU
  work — path only, per the user's instruction).
- **Part 3 — Round 2 / Round 1 carry-over**, clearly dated and labelled, including the
  transferability note (what does and does not carry to Round 3 without the archive).
- **Part 4 — Corrections register:** facts in the old file that are wrong for Round 3
  (starting with the 2,497 duplicate-Tg-groups figure).

## 11.4 `REMAINING_EXPERIMENTS.md`
Areas of improvement only — the user explicitly does **not** want them run. Organise by
target and by mechanism, each with: hypothesis · why it is plausible · expected magnitude ·
cost · the kill gate that would settle it. Seed content:

**Weak targets (ei 0.871, eps 0.885).** Proper GPR with a learned Tanimoto/Matérn mixture and
marginal-likelihood tuning (only crude GPR was tried); joint (chi, gap) reparametrised
multi-output GP; conformalised quantile regression as the estimator, not just the wrapper;
active-learning-style row weighting toward the mid-band where compression is worst.

**Tg (0.895, 56% of rows, 12.3% overlap).** Bicerano group-contribution features as an
explicit block; backbone/side-chain decomposition; a rigidity index from rotatable-bond
fraction × aromatic fraction; tail-robust losses (Huber/quantile) given that the extreme 10%
carries 36.9% of TSS; and randomised-SMILES augmentation *at training time* for the sequence
components only.

**Large-corpus SSL, done properly.** Atom-level tokenisation (polyBERT-style regex
`Br|Cl|\[[^\]]+\]|[A-Z][a-z]?|.`), whole-token masking, 6-layer encoder, **GBM heads instead
of linear probes**, and — critically — a **matched supervised control** and a **low-similarity
bin** as the acceptance gate. Everything tried so far failed with linear probes at small
scale; the honest statement is "not shown to work here", not "impossible".

**Architecture.** A shallow 4–6 model OOF stack to replace the 339-node chain: expected to
score *lower* publicly and possibly *higher* privately, because chain variance is the
diagnosed cause of the 0.026 gap. This is the single most interesting untested hypothesis in
the project — frame it that way.

**Uncertainty.** MC-dropout / deep ensembles to fix the ρ 0.13–0.44 error–uncertainty
correlation; full-data cross-conformal to close the R3.2 coverage gap.

**Invariance.** A cut-point-invariant featuriser (enumerate backbone cuts, average continuous
descriptors) to remove the last non-invariant component; consistency regularisation across
randomised SMILES as a training loss.

## 11.5 `QNA.md`
A one-page index into `docs/11_qna/` plus the ten hostile questions with their 30-second
answers inline, because that is what gets read five minutes before walking in.


---

# 12. WORK PACKAGE I — AGENTS.md and CONTEXT.md files

Four AGENTS.md files. Each is short, operational, and **routing-first**. None of them repeats
the others; each links.

## 12.1 `Personal/AGENTS.md` (the most important one)

Sections:
1. **What this folder is** — the user's operating base for docs, story, report, presentation
   and QnA. Not a codebase.
2. **Where everything is** — a routing table: *"asked to build a presentation → read
   `Presentation/PROMPT_PRESENTATION.md` + `SLIDE_PLAN.md` + `docs/06_results/` +
   `docs/12_assets/`"*; *"asked for the midnight report → `Midnight_Report/PROMPT_10PAGE.md`"*;
   *"asked a QnA question → `docs/11_qna/` then the specific doc"*; *"asked about a number →
   `docs/00_INDEX.md` canonical block"*; etc. Cover at least 12 request types.
3. **The canonical numbers block** (§2.3) — verbatim, so no agent has to go looking.
4. **The D1–D9 taxonomy** — verbatim.
5. **Vocabulary and safety rules** — never write "oracle" into anything the user might paste
   publicly; never invent a number; never cite a paper that is not in `Research/INDEX.md`;
   never edit `Obsidian/`.
6. **Cross-repo references** — the submission codebase is a sibling folder and is
   **read-only** from here: *"you may read
   `../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/` for code, charts and
   results; you may not write into it from a Personal/ task."* Same for
   `../Consolidation/` (read-only historical archive).
7. **Update discipline** — if a new number is produced, update `docs/00_INDEX.md` first,
   then every doc that quotes it; then re-run the consistency check (§15.3).

## 12.2 `Personal/CONTEXT.md`
A **portable**, self-contained context page (target ~2 pages) that can be pasted into any
agent with no repo access: the competition, the seven targets, the metric, the constraints,
our result, the architecture in six sentences, the five headline findings, the judged themes,
and the vocabulary rules. Model it on the existing root `CONTEXT.md` of the source repo but
strip all oracle/GPU/agent-loop content.

## 12.3 `<codebase>/` — **no AGENTS.md**
The user was explicit: the submission repo stays clean of agent files. Instead, put a short
**"For reviewers"** section at the end of `README.md` (reading order, how to verify the
score, what to run first).

## 12.4 `Consolidation/AGENTS.md`
1. **Purpose** — a historical archive, not a working tree. Nothing here is run; things here
   are *found*.
2. **The map** — §4.4 tree with one line each.
3. **Nested git repositories register** — a table of every embedded `.git` (path, origin if
   any, tags, HEAD at the time of consolidation, why it is preserved). At minimum:
   `03_round3_working_repo/` (tags v1.0.0, v1.0.1, v2.0.0, v3.0.0, v3.0.1; HEAD 9db3154).
   **Explain the mechanic:** an inner `.git` inside an outer repo is ignored by the outer
   one unless added as a submodule; we deliberately do **not** use submodules (no remotes),
   so the outer repo's `.gitignore` must exclude the inner working trees from being
   committed as loose files — see §14.5 for the exact recipe.
4. **GPU reference paths** — the full path index (from `SOURCE_INVENTORY.md` §D), the
   connection recipe, and the standing rule: **read-only, never modify**.
5. **Quarantine rules** — what is in `06_oracle_QUARANTINE/` and why it must never be
   copied outward, plus the grep command that proves a folder is clean.
6. **How to find things** — a "where did X come from?" table for the ~15 artifacts that exist
   in more than one place (`SOURCE_INVENTORY.md` §E).

## 12.5 Root `AGENTS.md` + `CONTEXT.md` + `README.md`
Root `AGENTS.md` is a **20-line router only**: three folders, one line each, and *"read the
AGENTS.md inside the folder you need"*. Root `CONTEXT.md` = a copy of
`Personal/CONTEXT.md` (single source; note in both that they are mirrors, or symlink).
Root `README.md` = a human-readable map plus the current status.

---

# 13. WORK PACKAGE J — `<codebase>/Website/` (the live demo)

## 13.1 What it must do

Paste a SMILES → pick a target (or all seven) → get: **the predicted value**, **a calibrated
interval**, **an applicability-domain tier**, **the nearest training analogue with its
similarity**, and **an explanation** (SHAP bar for the top features and an atom-highlighted
molecule image). Plus the invariance demo: a **"rewrite this SMILES"** button that generates
5 randomised spellings and shows the predictions are the same.

That last feature is the whole Round-3 theme rendered as a button. **Build it.**

## 13.2 Architecture (keep it boring and offline-capable)

- **Backend:** FastAPI (or Flask) + the rewritten `inference.py` `Predictor` class, loading
  `weights/polymer_weights.joblib` once at startup. Endpoints:
  `POST /predict {smiles, targets[]} → [{target, value, pi_low, pi_high, ad_tier,
  nn_smiles, nn_tanimoto, source}]`; `POST /explain {smiles, target} → {top_features[],
  atom_weights[], png_b64}`; `POST /randomize {smiles, n} → {variants[], predictions[],
  std, pct_of_train_std}`; `GET /health`.
- **Frontend:** a single static page (vanilla JS + a small chart lib, or Streamlit/Gradio if
  time is short). **Streamlit is the pragmatic choice if the demo is 3 days away** — say so
  and let the user pick.
- **Molecule rendering:** RDKit `Draw.MolToImage` / `SimilarityMaps` server-side → base64
  PNG. No JS chemistry dependency, no CDN.
- **Must run fully offline** on the presenting laptop. **No CDN links, no external fonts,
  no telemetry.** Conference wifi will fail.

## 13.3 Honesty requirements (non-negotiable — this is a trust-judged round)

- The UI must display the **prediction source** (model vs exact-training-label) and default
  to `--mode model` so it is never a lookup.
- The UI must display the **AD tier** and, for out-of-domain inputs, a visible
  *"outside applicability domain — treat with caution"* banner.
- A footer line stating the honest accuracy of the served model: *"served by the compact
  portable predictor (OOF R²: tg 0.897 · egc 0.860 · egb 0.875 · eea 0.884 · nc 0.795 ·
  ei 0.773 · eps 0.694). The leaderboard pipeline is the full ensemble (see README)."*
  Understating here buys enormous credibility; overstating loses the round.

## 13.4 Contents to ship

`Website/README.md` (run instructions, one command), `app.py`, `static/`,
`requirements-web.txt`, `screenshots/` (3 PNGs + a recorded GIF for the fallback), and
`sample_inputs.md` with the three demo polymers and their expected outputs.

```bash
cd Website && ../.venv/bin/python -m pip install -r requirements-web.txt
../.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8080
# or, if Streamlit:  ../.venv/bin/streamlit run app.py
```

## 13.5 Time-box

If the full FastAPI build exceeds half a day, ship the Streamlit version and put the
FastAPI one in `Optimized_Codes/` as future work. A working, honest, offline demo beats a
beautiful one that fails on stage.

---

# 14. WORK PACKAGE A — `Consolidation/`

> Do this **first** — it creates the safety copy that every other work package draws from.

## 14.1 Method

Use `rsync -a` (preserves timestamps and permissions) or `cp -a`. **Copy, do not move**,
until the user confirms; then optionally delete sources in a second, explicitly-approved pass.
Log every operation into `Consolidation/MANIFEST.md` as
`source → destination · size · date · note`.

## 14.2 What goes where

| destination | source | how |
|---|---|---|
| `00_competition/` | `<r3>/Competition_Details/*`, `AISEHack-2.0/Polymer Prediction/{Challenge Details.md, Competition Rules.md}`, `AISEHack-2.0/Polymer Pred Round 2/ROUND2_COMPETITION_DETAILS.md` | copy |
| `01_round1/` | `AISEHack-2.0/Polymer Prediction/` **minus** `scraped/`, `scraped.zip`, `venv/` | copy |
| `02_round2/` | `AISEHack-2.0/Polymer Pred Round 2/` (all of it, incl. `Round 2 Submissions/` with `paper.pdf` and the leaderboard screenshots) | copy |
| `03_round3_working_repo/` | the entire `AISEHack 2.0 Polymr Property Prediction Round 3/` repo **including `.git/`**, **excluding** `.venv/`, `Dataset/smile_r3.csv`, `Dataset/PI1M.csv`, `Oracle/`, `Phase5_Kiro_Score_Improvement/data/` | copy with excludes (§14.3) |
| `04_phases/phase4_explainability/` | `<r3>/Phase4_Round3_Explainability/` + the GPU `Phase_4_Explainability/{scripts,REQUIREMENTS.md,PLAN.md,PROMPT.md,AGENTS.md,EXPERIMENTS.md,STORY.md,README.md}` (scripts and docs only — **not** the 2.5 GB of outputs, which we already have) | copy + scp |
| `04_phases/phase5_score_improvement/` | `<r3>/Phase5_Kiro_Score_Improvement/` minus `data/` | copy |
| `04_phases/phase5a_gap_analysis/` | `<r3>/Phase5A_Gap_Analysis/` | copy |
| `04_phases/phase2_mechanism_sweep/` | `<r3>/scripts/phase2/` + a `GPU_REFERENCE.md` pointing at `r3_runtime/Phase_2/` (151 scripts) + any small result files pulled back | copy + scp small files |
| `04_phases/phase3_clean_stack/` | `<r3>/scripts/phase3/` + `GPU_REFERENCE.md` → `r3_runtime/Phase_3/` (282 scripts) + small results | copy + scp small files |
| `05_submissions/` | `<r3>/final_submissions/*`, `<r3>/CODEBASE/submission_v57.csv`, `CODEBASE/submission_imputation.csv`, `AISEHack-2.0/Polymer Pred Round 2/SANDMAN_Version_5{4,7}_*.csv`, GPU `r3_runtime/latest_submission.csv` | copy + scp, then write `PROVENANCE.md` (file · what produced it · score · where it was submitted) |
| `06_oracle_QUARANTINE/` | `<r3>/Oracle/`, `AISEHack-2.0/Polymer Prediction/scraped/` (+ `scraped.zip`), `Phase5_Kiro.../data/final_oracle.csv` | copy, then gitignore |
| `07_gpu_reference/` | **no bulk copy.** `PATH_INDEX.md` + `CONNECT.md` + `results/` (only small files: `phase5_summary.tsv`-equivalents, run logs' tails, `REQUIREMENTS.md`, README/summary MDs) | write + scp small |
| `08_research/` | `<r3>/research/`, `<r3>/score_discrepancy/`, `AISEHack-2.0/Polymer Pred Round 2/{research-log.md, findings.md, RESEARCH_NOVELTY_LEDGER.md}` | copy |
| `09_handoff/` | this `PLAN.md` + the four research docs + `PROGRESS.md` | copy |

## 14.3 The exclude list (use it every time)

```
.venv/  venv/  __pycache__/  .pytest_cache/  .ipynb_checkpoints/  .DS_Store  Thumbs.db
Dataset/smile_r3.csv        (330 MB — reference it, do not duplicate)
Dataset/PI1M.csv            (47 MB — one copy only, in 03_round3_working_repo)
Phase5_Kiro_Score_Improvement/data/     (duplicate of Dataset/, ~380 MB)
*.zip  (except small ones you deliberately keep)
```

**Keep exactly ONE copy of the official dataset** inside Consolidation, at
`Consolidation/00_competition/dataset/`, with a `README.md` giving the SHA-256 of each
file and pointing at the live copy in the working repo. Everything else references it.

## 14.4 Git setup for the three repos

```bash
# repo 1 and 2: ordinary
cd "<dest>/Personal" && git init && git add -A && git commit -m "Personal: initial consolidation"
cd "<dest>/AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase" && git init && \
  git add -A && git commit -m "Submission codebase: initial"

# repo 3: has nested repos inside — see 14.5 first
cd "<dest>/Consolidation" && git init && git add -A && git commit -m "Consolidation: initial archive"
```

Set `user.name`/`user.email` if not already global. Do **not** add remotes or push
without asking. Tag each initial commit (`v0.1.0-consolidation`).

## 14.5 Nested repositories — the exact mechanic

`03_round3_working_repo/` contains its own `.git/` with five tags. If you `git add` it
from the outer repo, git creates a **gitlink** (an unregistered submodule) — the contents are
*not* stored and the tags are *not* preserved in the outer repo. That is silent data loss.

**Do this instead:**
1. Keep the inner repo intact on disk (its `.git/` is the thing that carries the tags).
2. Add to `Consolidation/.gitignore`:
   ```
   03_round3_working_repo/
   02_round2/**/.git/
   06_oracle_QUARANTINE/
   ```
3. Record in `Consolidation/AGENTS.md` §nested-repos: the path, its HEAD, its tag list, and
   the sentence *"this directory is preserved on disk with its own git history and is
   deliberately not tracked by the outer repository."*
4. **Verify:** `cd 03_round3_working_repo && git log --oneline | head -3 && git tag` must
   still show `9db3154` and the five tags after the copy. If `cp` dropped `.git/`, redo
   with `rsync -a`.

Alternative (only if the user wants one repo to rule them all): create a bare mirror of the
inner repo at `Consolidation/07_gpu_reference/mirrors/round3_working_repo.git` via
`git clone --mirror`, which preserves every ref and tag in a single tracked directory.
**Ask before choosing this.**

## 14.6 Deduplication rules

1. One copy of the official dataset (§14.3).
2. One copy of the 169 Phase-4 outputs — keep `04_phases/phase4_explainability/outputs/`;
   the copies inside `03_round3_working_repo/CODEBASE/outputs/` come along with the repo
   copy and that is acceptable, but do not add a third.
3. One copy of each submission CSV in `05_submissions/`, with the provenance table; the
   in-repo copies stay where they are.
4. GPU material: **never bulk-copy**. Only `Phase_4_Explainability/scripts/` +
   `REQUIREMENTS.md` (they are the only GPU artifacts not already mirrored on the Mac) and
   small result files.

Target total size for `Consolidation/` (excluding the quarantine and the nested working
repo's own `.git`): **≤ 4 GB**.

## 14.7 `Consolidation/AGENTS.md` and `MANIFEST.md`
See §12.4 for AGENTS.md. `MANIFEST.md` is a plain table of every copy operation, written as
you go, ending with a summary: total items, total size, and the verification commands used.

## 14.8 The GPU reference file `07_gpu_reference/CONNECT.md`
Contains: host, the SSH_ASKPASS recipe verbatim (it works; the Mac has no `sshpass` and no
`timeout`), the hardware spec, the python path
(`~/Desktop/AISEHack-2.0/.venv-polymer/bin/python`, numpy 2.4.6 — the *correct* env; note
that `~/Desktop/r3_runtime/Phase_4_Explainability/.venv` has numpy 2.5.2 and **collapses
ei/eea**), the read-only rule, and the path index. **Do not commit the password to any repo
that could become public** — put it in the file only if the user confirms Consolidation stays
private; otherwise write `<password in 1Password / ask the user>`.

---

# 15. CONSISTENCY REGISTER (the thing that keeps you out of trouble)

## 15.1 One number, one place
Create `Personal/docs/00_INDEX.md` with the canonical block, and make **every** other
document quote it rather than restate it. When a number changes, it changes there first.

## 15.2 Known contradictions in the existing material — resolve these explicitly

| # | contradiction | resolution |
|---|---|---|
| 1 | Mean R² appears as 0.90352 / 0.903480 / 0.902289 / 0.90229 | use **0.9023** everywhere; footnote the panels once |
| 2 | `TRIALS.md` says Tg has **2,497** duplicate groups; Round-3 data has **4** | the 2,497 figure is Round-2 archive-inclusive. Label it or drop it |
| 3 | `AGENTS.md` says **457** SMILES appear in both train and test; canonicalised the number is **1,063** | state both: 457 raw string matches, 1,063 canonical |
| 4 | Scorecard says "14/17" in some docs and "14/18" in `scorecard.md` | the generated scorecard has **18** rows (AUG was added). Use **14/18** |
| 5 | Tg unit is °C in the data (min −109.8) but Phase-4 `AGENTS.md` says "K" | it is **°C**. Fix wherever it says K |
| 6 | `score_discrepancy` uses calibration −0.013; the verified figure is **−0.011** | use −0.011 (it is measured); mention −0.013 only as the earlier estimate |
| 7 | "median NN similarity 0.55–0.57 for weak targets" (TRIALS) vs "98% of test polymers are in train" | both true — the first is *same-target*, the second is *any-target*. **Always state the basis.** This is the two-regimes finding; do not let it read as a contradiction |
| 8 | Phase-5 "best 0.8367" vs V57 "0.9023" | different baselines. Always say which |
| 9 | Experiment counts vary across docs | use §4 of `RESULTS_ANALYSIS.md` as the single ledger |
| 10 | `egc = ei − eea` vs `ei = egc + eea` | the same identity; pick the **`egc = ei − eea`** orientation for the report and note the rearrangement once |

## 15.3 Mechanical checks to run before declaring done

```bash
CB="<dest>/AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase"

# 1. no forbidden words anywhere in the submission codebase
grep -rniE "oracle|final_oracle|khazana|polyinfo|tgss|test_answers|100\.116|vishwa|/Users/daver" "$CB" \
  --exclude-dir=.git || echo "CLEAN"

# 2. no agent files
find "$CB" \( -name "AGENTS.md" -o -name "CLAUDE.md" -o -name ".claude" -o -name ".codex" \
  -o -name "PROMPT.md" -o -name "PLAN.md" -o -name ".mcp.json" \) -print

# 3. submission contract
python - <<'PY'
import pandas as pd
d = pd.read_csv("submission.csv")
assert list(d.columns) == ["id","target"], d.columns
assert len(d) == 4940 and d.id.is_unique and d.id.min()==1 and d.id.max()==4940
assert d.target.notna().all() and d.target.map(float).abs().lt(1e6).all()
print("submission.csv OK")
PY

# 4. every figure referenced by a doc exists
grep -rhoE '\]\(([^)]+\.(png|svg|jpg))\)' "$CB" | sed -E 's/^\]\(//; s/\)$//' | sort -u | \
  while read f; do [ -e "$CB/$f" ] || echo "MISSING: $f"; done

# 5. numbers agree between docs and the notebook run
#    (compare outputs/notebook_metrics.json against the canonical block by hand or by script)
```

Also: **experiment count in `Experiment_Logs/` must be ≤ 80** —
`grep -c '^| ' Experiment_Logs/D*.md` and sum.

---

# 16. ACCEPTANCE CHECKLISTS

**Consolidation.** All Mac sources copied · `MANIFEST.md` complete · three git repos
initialised and committed · nested repo verified (`9db3154` + 5 tags still present) ·
quarantine gitignored and verified · GPU path index written, nothing on the GPU modified ·
total size ≤ 4 GB · `AGENTS.md` answers "where did X come from?" for the §E duplicates.

**Submission codebase.** All §15.3 checks pass · README has the 15 sections · ARCHITECTURE.md
de-oracled and has the per-target justification table · FINDINGS.md has all nine findings with
charts · the `.py` runs top to bottom in ≤30 min and produces ≥40 figures · `inference.py`
defaults to `--mode model` and reports source + AD tier + interval · `weights/README.md`
explains the container decision · `Experiment_Logs/` ≤80 entries across D1–D9 ·
`outputs/` curated to ~45–55 charts with `CAPTIONS.md` · `architecture.png` exists with
its source · `Website/` runs offline and shows source + AD tier · `Optimized_Codes/` has
`.gitkeep` + README · `setup.sh` asserts the pinned versions.

**Personal.** `docs/` has all 12 sections populated · ≥60 QnA pairs including the 10 hostile
ones · `Research/INDEX.md` has ≥30 entries with verified URLs · `Research_Paper/` copied
with its publication-gate warning · `Midnight_Report/` has both prompts + the style guide +
the moved samples + the analysis · `Presentation/` has the slide plan, prompt, speaker
notes, demo script + the moved samples + the analysis · `FINDINGS/STORY/TRIALS/
REMAINING_EXPERIMENTS/QNA` written · `AGENTS.md` routes ≥12 request types ·
`Obsidian/` untouched (verify with `git status` / mtimes).

---

# 17. EXECUTION ORDER

1. **§3 pre-flight** — commit, tag, verify env, verify disk, confirm the destination and the
   sandbox scope.
2. **§14 Consolidation** — the safety copy. Nothing else starts until this is committed.
3. **§10 docs/ skeleton + §15.1 canonical block** — create `docs/00_INDEX.md` and the folder
   structure with stub files first, so everything written afterwards can link into it.
4. **§6 the notebook** — because it *generates* the EDA charts and the metrics that the docs,
   README, FINDINGS, report and slides all cite. **Run it and capture
   `outputs/notebook_metrics.json` before writing any prose that quotes a number.**
5. **§5 the submission codebase** — assemble, curate outputs, write README / ARCHITECTURE /
   FINDINGS / RESULTS / Experiment_Logs, build `architecture.png`, rewrite `inference.py`.
6. **§10 docs/ fill** — now that every number exists.
7. **§9 Research** — run the searches, file the sources, copy the paper.
8. **§11 Personal top-level files** — FINDINGS, STORY, TRIALS, REMAINING, QNA.
9. **§7 Midnight_Report** — move the samples, write the analysis and the two prompts.
10. **§8 Presentation** — move the samples, write the analysis, slide plan, prompt, notes,
    demo script.
11. **§13 Website**.
12. **§12 AGENTS/CONTEXT** files (last, so they describe what actually exists).
13. **§15.3 + §16** — run every check, fix, commit, tag.

**Checkpoint with the user after step 2, after step 5, and after step 10.**

---

# 18. RISKS, DECISIONS AND OPEN QUESTIONS

## 18.1 Decisions to put to the user before acting

| # | question | recommendation |
|---|---|---|
| D1 | `model.pt` — real file, ONNX export of the fallback, or a documented joblib bundle? | **joblib bundle + weights/README.md** explaining why; ONNX only if asked |
| D2 | Ship both submission routes (V57 and the +0.0002 imputation variant) or one? | **one** (V57) — the second is noise and invites confusion |
| D3 | Website stack: FastAPI+static vs Streamlit | **Streamlit** if the demo is within 3 days, FastAPI otherwise |
| D4 | Keep the `khazana_scatter_*` external-verification charts? | **drop them** — the name implies a prohibited external database; the generalization ladder carries the same message safely |
| D5 | Team name on public artifacts: "Sandman"? | confirm |
| D6 | LICENSE for the public repo? | MIT unless told otherwise |
| D7 | Does `Consolidation/` stay private? (decides whether the GPU password may be written down) | assume **private**, but do not write the password unless confirmed |
| D8 | Publish the Round-2 research paper? | **no** — the draft says host sign-off is required; cite it as "in preparation" |
| D9 | Delete the Mac sources after copying into Consolidation? | **no**, not in this pass; propose it as a separate approved step |

## 18.2 Risks

| risk | mitigation |
|---|---|
| **Environment drift silently halves the score** | `setup.sh` asserts numpy 2.4.6 + sklearn 1.9.0; the notebook asserts in cell 1; the warning text is in README, ARCHITECTURE and requirements.txt |
| **A stray "oracle"/"khazana" string ships publicly** | the §15.3 grep is a release gate, run before every commit to the codebase repo |
| **The demo looks like a lookup table** | `--mode model` default, source displayed, AD tier displayed, honest footer |
| **Numbers disagree across README / report / slides** | the canonical block (§15.1) + `outputs/notebook_metrics.json` + the §15.3 cross-check |
| **Nested git repos lose their tags** | §14.5 gitlink recipe + explicit verification |
| **Copying blows up the disk** | the exclude list (§14.3); 78 GiB free, target ≤4 GB |
| **The 2.5 h pipeline is re-run under time pressure and fails** | never re-run it for the deliverable; the frozen `submission.csv` is the artifact. Only the ≤30-min EVAL track runs in the notebook |
| **A judge asks for a number we cannot trace** | every doc ends with its source file; `ASSET_MANIFEST.md` ties every chart to its script |
| **Time runs out** | priority order if you must cut: (1) notebook + README + FINDINGS + architecture diagram, (2) slide plan + speaker notes, (3) report prompts, (4) website, (5) the long tail of `docs/` |

## 18.3 What this plan deliberately does NOT do

- It does not try to improve the score. The score is frozen at private 0.891 / local 0.9023.
  `REMAINING_EXPERIMENTS.md` records the ideas; nobody runs them.
- It does not re-run the 2.5-hour pipeline.
- It does not copy the GPU laptop's bulk data (only scripts, docs and small results).
- It does not modify `Personal/Obsidian/` or anything on the GPU.
- It does not delete anything from the existing Mac folders.

---

# 19. QUICK REFERENCE CARD (pin this)

```
DESTINATION   /Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3/
SOURCE (R3)   /Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/   ("Polymr")
SOURCE (misc) /Users/daver/Desktop/AISEHack-2.0/
GPU           vishwa@100.116.22.29   READ-ONLY   ~/Desktop/{r3_runtime,AISEHack-2.0}
PYTHON        <SOURCE R3>/.venv/bin/python   (numpy 2.4.6 · sklearn 1.9.0 · rdkit 2026.03.5)

SCORES        local panel 0.9023 · public 0.917 · private 0.891 · scorecard 14/18
TARGETS       tg .8953 | egc .9111 | egb .9268 | ei .8711 | eea .9183 | nc .9086 | eps .8847
METRIC        unweighted mean of 7 per-target R² — every target = 1/7 exactly
DOMAINS       D1 physics · D2 representation · D3 SSL · D4 neural · D5 classical ·
              D6 cross-property · D7 ensembling · D8 calibration · D9 validation/XAI

NEVER         write "oracle" outside Consolidation/ · modify the GPU · touch Personal/Obsidian/ ·
              invent a number · exceed 80 experiments in the submission repo ·
              ship external label data · re-init an existing .git
```

