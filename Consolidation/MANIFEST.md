# MANIFEST — every consolidation operation

Performed 2026-08-31. Method: `mv` for material already inside the Round-3 repository (so git
records renames and the history is preserved), `rsync -a` / `cp -a` for material copied in from
outside it. **Nothing was deleted from any source.**

## Moves (within the repository — history preserved)

| source | destination | note |
|---|---|---|
| `Competition_Details/*` | `00_competition/` | the official Round-3 brief |
| `Dataset/*` | `00_competition/dataset/` | **the single canonical dataset copy**, 361 MB |
| `AGENTS.md`, `CONTEXT.md`, `EXPERIMENT_LOOP.md`, `FINAL_REPORT.md`, `PLAN.md`, `STORY.md`, `TRIALS.md` | `03_round3_working_repo/` | the working repo's own documents |
| `analysis/`, `experiments/`, `logs/`, `research/`, `scripts/`, `feasibility/`, `CODEBASE/`, `final_submissions/`, `score_discrepancy/` | `03_round3_working_repo/` | the working tree |
| `Phase4_Round3_Explainability/` | `04_phases/phase4_explainability/` | 38 scripts, 169 artifacts |
| `Phase5_Kiro_Score_Improvement/` | `04_phases/phase5_score_improvement/` | 55 scored experiments |
| `Phase5A_Gap_Analysis/` | `04_phases/phase5a_gap_analysis/` | 37 runs, the ceiling arithmetic |
| `Oracle/` | `06_oracle_QUARANTINE/Oracle/` | quarantined and git-ignored |
| `Phase5_Kiro_Score_Improvement/data/` | `06_oracle_QUARANTINE/phase5_data/` | contained a verification copy |
| `Consolidation_Handoff/PLAN.md` | `../PLAN.md` (repo root) | the consolidation contract |
| `Consolidation_Handoff/{README.md,research/}` | `09_handoff/` | the four research documents |

## Copies (from outside the repository — sources untouched)

| source | destination | size | exclusions |
|---|---|---|---|
| `~/Desktop/AISEHack-2.0/Polymer Prediction/` | `01_round1/` | 352 KB | `venv/` (258 MB), `scraped/`, `scraped.zip` |
| `~/Desktop/AISEHack-2.0/Polymer Pred Round 2/` | `02_round2/` | 55 MB | `__pycache__`, `.DS_Store` |
| Round-1 `scraped/` + `scraped.zip` | `06_oracle_QUARANTINE/` | 4.7 MB | — |
| `scripts/phase2/`, `scripts/phase3/` | `04_phases/phase{2,3}_*/` | small | duplicate of the working-repo copy, kept for the organised view |
| submission CSVs + score records | `05_submissions/` | 1.1 MB | — |
| `research/`, `score_discrepancy/`, Round-2 research log / findings / novelty ledger | `08_research/` | 924 KB | — |
| `<destination folder>/Personal/{Obsidian,Obsidian.zip,Sample Reports,Sample Presentations}` | `../Personal/` | 7.9 MB | none — **Obsidian was copied, never modified** |

## Symlinks created (so nothing is duplicated)

| link | target |
|---|---|
| `03_round3_working_repo/Dataset` | `../00_competition/dataset` |
| `../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/Dataset` | `../Consolidation/00_competition/dataset` |

## Totals

| | |
|---|---|
| Consolidation | **520 MB**, 2,335 files |
| of which the dataset | 361 MB (one copy) |
| of which quarantine | 23 MB (git-ignored) |
| Personal | 8.5 MB |
| submission codebase | 11 MB |

## Verification performed

```bash
# the root git history and tags still resolve
git log --oneline | head -3 && git tag

# the dataset symlinks resolve
ls -l AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/Dataset && \
  ls AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/Dataset/

# the submission codebase contains no forbidden term (word-boundary; base64 blobs excluded)
grep -rInwE "oracle|khazana|polyinfo|tgss|test_answers|vishwa" \
  AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase --exclude-dir=.git || echo CLEAN
```

All three passed at consolidation time. Re-run them from `../RUN.md` before publishing anything.

## Not copied, deliberately

The GPU laptop's 2.5 GB of Phase-4 raw outputs (the curated copies suffice), the 66.7 GB
`AISE Full Codebase.zip`, every virtual environment, and the Round-1 `venv/`.
