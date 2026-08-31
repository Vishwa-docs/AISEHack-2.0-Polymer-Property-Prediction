# SOURCE_INVENTORY.md — every location that holds Round-1/2/3 material

Verified by direct inspection 2026-08-31. Sizes are `du -sh`.
**Nothing was moved or modified while writing this file.**

---

## A. Mac — `/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3` (2.0 GB)

This is the **live Round-3 working repo** and the primary source of truth. Git repo,
working tree **clean**, HEAD `9db3154` ("FINAL_REPORT draft with Round-3 evidence
narrative"), tags `v1.0.0 v1.0.1 v2.0.0 v3.0.0 v3.0.1`. 15 commits.
`.gitignore` excludes `Oracle/`, `.venv/`, `*.pkl/.npz/.npy/.joblib/.parquet`,
`experiments/**/predictions.csv`.

| Path | Size | What it is | Consolidation verdict |
|---|---|---|---|
| `CODEBASE/` | 17 MB | **The Round-3 deliverable.** `pipeline_final.py` (652 KB, V57 + evidence engine), `pipeline_v57_final.py` (564 KB), `evidence_engine.py` (84 KB), `featurize.py`, `inference.py`, `build_weights.py`, `build_imputation_variant.py`, `weights/polymer_weights.joblib` (2 MB), `submission_v57.csv`, `submission_imputation.csv`, `README.md`, `ARCHITECTURE.md`, `FEASIBILITY.md`, `requirements.txt`, `outputs/` (13 MB, **169 evidence artifacts**: SHAP, invariance, conformal, AD, ladder, scorecard, TRUSTWORTHINESS_REPORT.html) | **→ becomes the Submission Codebase** (clean, strip agent files) |
| `Dataset/` | 361 MB | `train.csv` (7,409), `test.csv` (4,940), `PI1M.csv` (995,799 / 47 MB), `smile_r3.csv` (5,973,369 / 330 MB), `sample_submission.csv`, `base_line_model.ipynb` | **→ Consolidation** (do NOT duplicate into the submission repo; reference or symlink) |
| `Oracle/` | 16 MB | `final_oracle.csv` (authoritative), `oracle.csv`, proxies, `tg_external_matches.csv`, ~28 candidate CSVs + score JSONs, `build_round2_oracle.py`, `score_against_oracle.py`, `NOTES_R3.md` | **→ Consolidation ONLY. Must NEVER enter the submission repo or any public artifact.** git-ignored today — keep it that way |
| `Phase4_Round3_Explainability/` | 14 MB | `AGENTS.md`, `EXPERIMENTS.md`, `outputs/` (169 files), `outputs_and_logs/` incl. 2 failed reruns | → Consolidation |
| `Phase5_Kiro_Score_Improvement/` | 17 MB | 55 experiment dirs (P5-001…P5-333), `diagnostic/` (8 EDA scripts + outputs/plots/tables/reports), `logs/phase5_summary.tsv` (54 scored rows), `logs/FINAL_PHASE5_EXPERIMENT_SUMMARY.md`, `EXECUTIVE_SUMMARY.md`, `PLAN.md` (1,462 lines / 210 experiments), `PROMPT.md`, `RESULTS.md`, `REFERENCES.md` (561 lines), `NEW_EXPERIMENTS.md`, `NEW_NEW_EXPERIMENTS.md`, `PLAN_AMENDMENT.md`, `INTEGRATION_CHECKLIST.md`, `final_experiment_submission/`, `data/` (duplicate copies of the 4 official CSVs) | → Consolidation. **Mine `REFERENCES.md` and `diagnostic/` for the docs.** Delete the duplicate `data/` copies during consolidation to save ~380 MB |
| `Phase5A_Gap_Analysis/` | 19 MB | 37 experiment dirs (P5A-000…P5A-127), `HUMAN_REPORT.md` (**the score-ceiling maths — highest-value single doc in the repo**), `DIAGNOSIS_repro.md`, `logs/phase5a_summary.tsv`, `logs/phase5a_final_summary.tsv`, `output/` (01_eda_summary.csv, 01_claims.json, 02_leverage.csv, 02_scenarios.csv, 03_per_target.csv, 03_variants.csv, 03_tg_categories.csv, 04_headroom.csv, 05_partner_availability.csv, 07_residual_structure.csv, 08_target_profiles.txt, fig_01..fig_04), `investigation_r3/` | → Consolidation. **HUMAN_REPORT.md feeds Appendix B (mathematical ceiling)** |
| `score_discrepancy/` | 204 KB | `AGENTS.md`, `README.md`, `oracle_vs_private.md`, `tg_oracle_extension.md`, `previous_runs_better.md`, `khazana_tg.md`, `priority_action_plan.md`, `NEW_EXPERIMENTS.md` (3,742 lines — a full ML-workflow blueprint), `PLAN.md` | → Consolidation. **The pub/priv-gap story lives here** |
| `final_submissions/` | 1.6 MB | `v57_reproduction_standalone.py`, `submission.csv` (**the file actually submitted**), 4 score JSONs, `README.md`, `CONTEXT.md` | `submission.csv` **→ Submission Codebase**; rest → Consolidation |
| `logs/` | 1 MB | `experiments.jsonl` (247), `oracle_scores.jsonl` (246), `latest_verified.txt` (R3-C099 = 0.90276), `README.md` | → Consolidation (source for Experiment_Logs/) |
| `scripts/` | 11 MB | `phase2/` (build_phase2_suite.py, experiment_mechanisms.py, r3_core, tests), `phase3/`, `r2_reference/` (Sandman V52/V53 notebooks, `fable_engines/`, `v52_bundle/`), `r3_baseline_noarchive.py` | → Consolidation |
| `research/` | 152 KB | `50_experiment_plan.md`, `r2-experiment-history-digest.md`, `r2-final-notebook-dissection.md`, `r2-governance-digest.md`, `web-research-kaggle-strategy-20260826.md`, `web-research-polymer-methods-20260826.md`, `loop_status_20260827.md` | → Consolidation + mine into `Personal/Research/` |
| `Competition_Details/` | 64 KB | `Overview.txt`, `Dataset Description.txt`, `Competition Rules.txt`, Kaggle HTML | → Consolidation + `Personal/CONTEXT.md` source |
| `feasibility/` | 20 KB | `score.py`, `test_imputation_lift.py`, `test_imputation_lift2.py` (**these read the oracle**) | → Consolidation only. `CODEBASE/feasibility/` is a duplicate that currently ships in the deliverable — **must be removed or the oracle dependency neutered** before submission |
| `AGENTS.md` (22 KB), `PLAN.md` (19 KB), `EXPERIMENT_LOOP.md` (16 KB), `CONTEXT.md` (20 KB), `TRIALS.md` (40 KB, 452 lines), `STORY.md` (9 KB), `FINAL_REPORT.md` (3.4 KB) | — | root context files | `TRIALS.md`/`STORY.md` → `Personal/` (extended); rest → Consolidation |
| `analysis/`, `experiments/` | 4 KB each | README stubs only — **empty** | note as empty; do not pretend they hold work |
| `.venv/` | ~1.5 GB | the **validated env** (python 3.11, pandas 3.0.5, **numpy 2.4.6**, sklearn 1.9.0, rdkit 2026.03.5) | do NOT copy; recreate from `requirements.txt` |

## B. Mac — `/Users/daver/Desktop/AISEHack-2.0` (≈320 MB, **NOT a git repo**)

| Path | Size | Contents |
|---|---|---|
| `Polymer Pred Round 2/` | 55 MB | `ROUND2_COMPETITION_DETAILS.md`, `ROUND2_PROGRESS_REPORT.md`, `POLYMER_ROUND2_EXPERIMENT_LOOP.md`, `POLYMER_ROUND2_IMPLEMENTATION_RUNBOOK.md`, `README_EXPERIMENTS.md`, `RESEARCH_NOVELTY_LEDGER.md`, `research-log.md`, `findings.md`, `SANDMAN_Version_54_9th_Aug_with_archive copy.csv`, `SANDMAN_Version_57_9th_Aug_without_archive copy.csv` (**the R2 submitted files**), `ppp-round-2/` (R2 official data incl. `archive/`), `Round 2 Submissions/` (**`paper.pdf`**, `Leaderboard Ranking.png`, 3 screenshots, `Submission Points.txt`, `final_submissions/`) |
| `Polymer Prediction/` | 263 MB | Round-1 material: `Challenge Details.md`, `Competition Rules.md`, `Gemini Suggestions.md`, `Final_Prompt.txt`, `readme_experiments.md`, `20260722_polymer_experiment_status.md`, `train_sample.csv`, `test_answers_sample.csv.xlsx`, **`scraped/`** (`MTL_Khazana.zip`, `export.csv`, `Tg_SMILES_class_pid_polyinfo_median.csv`, `TgSS_enriched_cleaned.csv`, `test_answers.csv`, `train_answers.csv`, `Full_Dataset_Pull.py`), `venv/` |
| `.claude/`, `.mcp.json`, 2 screenshots | small | agent config |

⚠ **`Polymer Prediction/scraped/` is external labelled data (Khazana + PolyInfo-derived Tg).
It is the raw material of the oracle. It must be quarantined in Consolidation with the same
"never enters a submission" treatment as `Oracle/`.**

## C. Mac — `/Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3` (the NEW destination)

Current state (2026-08-31): **no git anywhere**, and only two of the three folders exist.

```
AISEHack 2.0 Polymer Property Prediction Round 3/
├── .obsidian/                                             (Obsidian vault config — leave alone)
├── Personal/
│   ├── Obsidian/            12 notes — 0. Strategy, Judging Criteria and Expectations,
│   │                        Presentation, Analysis and Understanding, Deliverables, Codebase
│   │                        (empty), Additional, Post Readiness Score Analysis, Logs and
│   │                        Sessions, Prompts and Skills, Celestial, Travel Plans
│   │                        ★ USER-OWNED — DO NOT EDIT ANY FILE IN HERE
│   ├── Obsidian.zip
│   ├── Sample Reports/      5 files: 'Final Submission - Achievers.md',
│   │                        'Final Submission Report Template_ AI for Science &
│   │                        Engineering_team_triverse.md', 'Final submission Report.md',
│   │                        'RuVision_FinalSubmission_Report.docx.md' (empty of headings),
│   │                        'Final_Submission_Report .pdf', 'VibeCoders_Final_Submission_Report_3pg.pdf'
│   └── Sample Presentations/ 9 PDFs + contents.md (the winning-deck structural analysis)
├── AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/   ← EMPTY
└── (Consolidation/  ← DOES NOT EXIST YET)
```

**Note the user's message said the sample presentations are under `Sample Reports/`; they
are actually under `Sample Presentations/`, and `contents.md` lives there too.**

## D. GPU laptop — `vishwa@100.116.22.29` (Tailscale, password `kumaresh@123`)

RTX 5090 (24 GB), 62 GB RAM, 24 cores. **Read-only reference. Never modify
`~/Desktop/AISEHack-2.0/`.** Connection recipe: see `AGENTS.md` §5 (SSH_ASKPASS pattern;
Mac has no `sshpass` and no `timeout`).

### D1. `~/Desktop/r3_runtime/` — all Round-3 compute

| Path | Size | Contents |
|---|---|---|
| `Phase_4_Explainability/` | **2.5 GB** | `AGENTS.md`, `REQUIREMENTS.md` (**the R1.1–R4.3 requirement definitions — the grading contract**), `PLAN.md`, `PROMPT.md`, `EXPERIMENTS.md`, `STORY.md`, `README.md`, `scripts/` (**38 files**: 00_setup … 18_scorecard, A1_train_mlp, A2_linear_probes, A3_activation_patching, A4_causal_tracing, A5_attribution_patching, B1/B2_counterfactuals, C1_bigsmiles_invariance, C2_stereo_invariance, C3_consistency_reg, D1_ensemble_vs_conformal, D2_shift_aware_conformal, D3_reliability_tiers, E1_physics_violations, E2_physics_decomp_comparison, F1_proxy_sweep, F2_feature_ablation, F3_oracle_sweep, G1_html_report, **G2_demo_notebook.ipynb**), `outputs/` (169 files), `outputs_and_logs/`, `Dataset/`, `Oracle/`, `final_submissions/` |
| `Phase_3/` | 126 MB | **282 experiment scripts** `exp001_p3A01-clean-stack-v1.py` … , `r3_core/`, `run.sh`, `run_log.txt`, `outputs_and_logs/` |
| `Phase_2/` | 130 MB | **151 experiment scripts** `exp001_aA00.py` … , `experiment_spec.json`, `mechanisms.json`, `r3_core/`, `source_bundle/`, `workspace/`, `tests/`, `Polymer Prediction Challenge Round 2/` |
| `recreation/` | 453 MB | the 0.904 reproduction workspace |
| `experiments/` | 12 MB | **101 dirs** `R3-C001…R3-C099` + `R2-F02-…-ionic-engine-without_archive` |
| `final_submissions/`, `fable/`, `scripts/`, `logs/` | small | |
| `v57_reproduction_without_archive.py` | 16 KB | |
| `Prompt 1.txt` (44 K), `Prompt 2.txt` (40 K), `Prompt 3.txt` (8 K) | | the prompts that drove the phases |
| `train.csv / test.csv / PI1M.csv / smile_r3.csv` | 361 MB | official data copies |
| `latest_submission.csv`, `latest_experiment_id.txt`, `README.md`, `run_all_50.sh`, `run_all_100.sh`, `run_loop.sh`, `monitor.sh` | | |

`~/Desktop/ppp-round-2` is a **symlink** → `~/Desktop/r3_runtime`.

### D2. `~/Desktop/AISEHack-2.0/` — the Round-2 codebase (git repo; safety commit `d75bd74`, tag `round3-before-start-20260826-2230`)

Top level: `AGENTS.md`, `HUMAN_HANDOFF.md`, `Fable_Revised_Prompt.md`, `configs/`, `docs/`,
`experiments/`, `research/`, `src/`, `tests/`, `tools/` (368 scripts), `Sonnet_Findings/`,
`agent-model-switcher/`, `Polymer Prediction Challenge/`,
**`Polymer Prediction Challenge Round 2/`** (375 clean experiments in
`experiments/CLEAN_OFFICIAL_ONLY/`, `logs/EXPERIMENT_LOG.md`, `research/research-log.md`,
`research/findings.md`, `research/best_component_registry.yaml`,
`research/per_target_best_leaderboard.json`,
`POLYMER_ROUND2_FINAL_REPORT_20260804.md`), and **`Polymer_Research_Paper/`**:

```
Polymer_Research_Paper/
├── RESEARCH_PAPER_PROMPT.md
├── drafts/paper_draft_v1.md        (~10,100 words, 5 figures, 5 tables, 2 appendices)
├── drafts/figures/                 fig1_replication_grid.png · fig2_tail_concentration.png
│                                   fig3_nn_distance_correlation.png · fig4_six_strategies.png
│                                   fig5_timeline.png
├── drafts/make_figures.py, make_figure5_timeline.py
├── latex/                          paper.tex → paper.pdf (IEEEtran conference, 8 pp, 23 refs)
├── experiments/code/               02_extract_embeddings.py, 03_cv_pretrained_embeddings.py,
│                                   04_cv_tree_baseline.py, 05_finetune_chemberta.py,
│                                   06_final_oracle_score.py, chemberta_utils.py, common_paper.py
├── experiments/outputs/            (empty)
├── literature/                     (empty)
└── source_survey/                  (empty)
```

Paper title: *"When Six Independent Attempts Agree: Cross-Contributor Evidence for a
Performance Ceiling in Small-Sample, Multi-Target Polymer Property Prediction."*
**This is a genuine differentiator — a real research paper from our own work, with a
ChemBERTa control experiment showing pretrained embeddings LOSE (0.751 frozen / 0.784
fine-tuned vs 0.810 tree baseline).**

### D3. Other GPU items

`~/Desktop/AISEHack-2nd-Edition-Codebase/` — `Polymer Property Prediction/`,
`Polymer Property Round 2/`, `README.md` (the codebase up to phase 2).
`~/Desktop/AISE Full Codebase.zip` — **66.7 GB** archive. Do not touch, do not copy.

## E. What exists in more than one place (deduplication targets)

| Artifact | Copies |
|---|---|
| Official 4 CSVs | `Round 3/Dataset/`, `Round 3/Phase5_Kiro.../data/`, GPU `r3_runtime/`, GPU `Phase_4_Explainability/Dataset/` |
| Phase-4 outputs (169 files) | `Round 3/CODEBASE/outputs/`, `Round 3/Phase4_Round3_Explainability/outputs/`, GPU `Phase_4_Explainability/outputs/` |
| Oracle | `Round 3/Oracle/`, `Round 3/Phase5_Kiro.../data/final_oracle.csv`, GPU `Phase_4_Explainability/Oracle/` |
| V57 pipeline source | `CODEBASE/pipeline_v57_final.py`, `CODEBASE/pipeline_final.py` (Part A byte-identical, 570,044 chars), `final_submissions/v57_reproduction_standalone.py` |
| Submission CSV | `final_submissions/submission.csv`, `CODEBASE/submission_v57.csv`, GPU `latest_submission.csv`, `AISEHack-2.0/Polymer Pred Round 2/SANDMAN_Version_57_...csv` |

**Rule for consolidation: keep ONE canonical copy per artifact inside `Consolidation/`, and
record every other location in `Consolidation/AGENTS.md` as a reference path rather than
copying it again.**
