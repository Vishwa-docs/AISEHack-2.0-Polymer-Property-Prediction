# AGENTS.md — Consolidation/

## 1. Purpose

A **historical archive**, not a working tree. Nothing here is run; things here are **found**.
Every artifact ever produced for this competition — Round 1, Round 2, Round 3, every phase, every
submission — gathered and organised so that any question of the form *"where did X come from?"*
has an answer.

If you are trying to **do** something, you are in the wrong folder: code lives in
`../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/`, and everything user-facing lives
in `../Personal/`.

## 2. The map

| folder | contents | size | files |
|---|---|---:|---:|
| `00_competition/` | the official Round-3 brief (Overview, Dataset Description, Competition Rules, the Kaggle page) **and the single canonical copy of the dataset** in `dataset/` | 361 MB | 11 |
| `01_round1/` | the Round-1 working folder, minus its venv and its scraped data | 352 KB | 8 |
| `02_round2/` | the whole Round-2 folder including `Round 2 Submissions/`, the research log, findings and the novelty ledger | 55 MB | 38 |
| `03_round3_working_repo/` | **the Round-3 working repository as it stood at consolidation** — CODEBASE, logs, experiments, scripts, research, score_discrepancy, TRIALS, the original AGENTS/PLAN/CONTEXT | 30 MB | 808 |
| `04_phases/` | phase2_mechanism_sweep · phase3_clean_stack · phase4_explainability · phase5_score_improvement · phase5a_gap_analysis | 49 MB | 1,334 |
| `05_submissions/` | every submission CSV ever made, plus the score records and the standalone reproduction script | 1.1 MB | 10 |
| `06_oracle_QUARANTINE/` | **the held-out verification data and all external label sources.** Read `README.md` in there before touching anything | 23 MB | 108 |
| `07_gpu_reference/` | **no bulk copies** — a path index, the connection recipe and small pulled-back results | — | — |
| `08_research/` | the Round-3 research notes, the score-discrepancy analysis and the Round-2 research log / findings / novelty ledger | 924 KB | 20 |
| `09_handoff/` | the consolidation plan's companion research documents, the agent brief and the progress log | 104 KB | 8 |

Total ≈ **520 MB**, of which 361 MB is the one canonical dataset copy and 23 MB is quarantine.

## 3. Git — how this repository is arranged

The **root repository is the Round-3 working repo's own git history**, preserved. Its working
tree was relocated into `03_round3_working_repo/` during consolidation, so the history, the tags
(`v1.0.0`, `v1.0.1`, `v2.0.0`, `v3.0.0`, `v3.0.1`) and the consolidation-start tag all still
resolve from the repository root.

`git log --oneline | head` and `git tag` at the **root** show that history. There is no nested
`.git` inside `03_round3_working_repo/` — the history was not duplicated, it was kept.

`Personal/` and the submission codebase are **their own repositories** (they are published
separately and must not carry this history); they are listed in the root `.gitignore`.

## 4. Quarantine rules — `06_oracle_QUARANTINE/`

Contains the held-out verification answers and the external label databases used to build them.

* **Nothing in that folder may be copied outward**, referenced by any pipeline, or named in any
  public artifact.
* It is git-ignored.
* Verification against it happens **only after** a candidate CSV is frozen and hashed.
* Proof that a folder is clean:

```bash
grep -rInwE "oracle|khazana|polyinfo|tgss|test_answers" <folder> --exclude-dir=.git || echo CLEAN
```

  (Use `-w` word boundaries: base64 image blobs in generated HTML can contain those letter
  sequences by chance, and a naive grep will report a false positive.)

## 5. GPU reference

**Never modify the GPU laptop.** `07_gpu_reference/PATH_INDEX.md` says what is where;
`CONNECT.md` has the connection recipe. Bulk data is never copied off it — only scripts,
documents and small result files.

## 6. Where did X come from?

| artifact | it exists in | which is canonical |
|---|---|---|
| the official dataset | `00_competition/dataset/` and symlinks from the codebase and the working repo | **`00_competition/dataset/`** — one copy, everything else links |
| `submission.csv` | `05_submissions/`, `03_round3_working_repo/final_submissions/`, and the codebase root | **the codebase root** is the shipped file; `05_submissions/` is the archive with provenance |
| the evidence outputs (169 artifacts) | `04_phases/phase4_explainability/outputs/` and `03_round3_working_repo/CODEBASE/outputs/` | the phase folder is the original run; the codebase copy is the curated subset |
| the phase 2/3 scripts | `04_phases/phase{2,3}_*/` and `03_round3_working_repo/scripts/phase{2,3}/` | the working-repo copy is the original; the phase folder is the organised view |
| the research documents | `09_handoff/research/` | canonical |
| the Round-2 findings and research log | `08_research/round2_*.md` and `02_round2/` | `02_round2/` is the original |

## 7. What is deliberately **not** here

* The GPU laptop's bulk outputs (2.5 GB of Phase-4 artifacts) — the curated copies suffice.
* The 66.7 GB `AISE Full Codebase.zip`.
* Any virtual environment.
* The Round-1 `venv/` (258 MB) and its scraped data (moved to quarantine instead).
