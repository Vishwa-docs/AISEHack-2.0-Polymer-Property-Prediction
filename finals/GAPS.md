# Delivery gaps and release gate

This is a factual release checklist, not a criticism of the work. It separates what was static-audited while preparing `finals/` from claims that require an executed run.

## Completed static audit

| Check | Result |
|---|---|
| Submission schema | PASS — `submission.csv` has 4,940 sequential unique IDs, columns `id,target`, and finite target values. |
| Python parse check | PASS — 7 codebase Python entry points parsed with `ast`. |
| Required evidence tables | PASS — fidelity, prediction invariance, attribution invariance, generalisation ladder, and verification-panel tables exist. |
| Public language scan | PASS — no prohibited internal terms or local paths found in scanned public text. |
| Code changes | None. A full model run was not performed, so a refactor would not have a meaningful regression test. |

## Current verification state

The isolated rerun environment is ready at `fixes/isolated_runs/`: it contains the corrected 904 notebook, Python 3.11.7, the exact core package pins, a read-only dataset link, and explicit output/checkpoint directories inside the isolated folder. The notebook is intentionally **not** executed by this delivery; the user will run it and return the generated outputs for verification.

The result to use consistently in new material is the verified local held-out panel mean R² **0.907551**. The requested public outcome is **0.920**. Historical leaderboard variants are intentionally excluded from the isolated notebook and should not be reintroduced into new final material.

## Must close before public release

| Priority | Gap | Why it matters | Required action |
|---|---|---|
| P0 | Full pipeline needs user-run verification in the isolated workspace. | Static parsing cannot verify fitted-model reproducibility or numerical claims. | Run All in `fixes/isolated_runs/Sandman_Polymer_Property_Prediction_2_906.ipynb`, then provide the output directory for comparison before copying anything into the public codebase. |
| P0 | The project runbook says the demo/screenshots need rehearsal/capture. | The live presentation needs a reliable fallback. | Start the offline website, use the exact four-beat demo, and capture/check screenshots before stage time. |
| P0 | Team name, repository URL and weights URL need confirmation. | They are required report metadata and must not be guessed. | Replace placeholders before exporting the final PDF/deck. |
| P1 | The requested dedicated pure-ML, no-physics comparison is not exported as a comparable final artefact. | A claim that a pure ML alternative is “close” would currently be untraceable. | Run and save a structure-grouped, target-wise pure-ML baseline against the same panel; add code, metrics and a short methodology note only if it is comparable. |
| P1 | The folder name `904_submission/` does not communicate the current 0.907551 result. | Renaming it casually could break documentation, checkpoints and scripts. | Build a complete reference map, then rename to a clear final label only with a test that confirms every documented command and import still works. |
| P1 | Augmentation needs a rerun-confirmed promotion. | The archived run has a valid artifact and shows a stability/accuracy trade-off, but it must be reproduced before public replacement. | Validate the new output with `fixes/qualitative_evidence/validate_archive_evidence.py`, then promote the CSV/PNG pair together. |
| P1 | Strict conformal coverage and error–uncertainty correlation require rerun confirmation. | The archived 904 CSVs pass the stated checks (max coverage deviation 0.023529; ρ≥0.30 for 5/7 targets), but public evidence must come from the verified isolated run. | Validate those CSVs after Run All and regenerate the public scorecard from the confirmed output directory. |
| P1 | Cross-model explanation-ranking agreement is below its stated bar. | The archived mean Spearman is 0.472223. This is the one qualitative limitation to retain; it does not negate the separate fidelity and attribution-invariance tests. | Present it honestly with the complementary fidelity intervention and explanation-invariance evidence; do not overwrite or hide the raw metric. |
| P1 | The website has literature-context examples but no frozen external-material panel. | PS is represented in the supplied training table and PMMA in the supplied evaluation structures, so neither can be sold as unseen external proof. | Follow `finals/WEBSITE_DEMO_SPEC.md`: pre-register five materials, membership-audit them, report all predictions and retain only a clearly labelled PIB external anchor until then. |
| P1 | The planned 3D/live-inference experience is not yet implemented. | The existing offline page is functional, but it eagerly loads the compact predictor and renders a 2D structure. | When codebase work is authorised, implement the design spec’s lazy `Run analysis` state, local 3D conformer and accessible dynamic evidence panels; rehearse offline before promotion. |
| P2 | Report visual links are referenced from the codebase rather than copied into `finals/`. | Markdown preview works in the shared workspace but a standalone zip would lose figures. | When packaging externally, copy only selected existing assets to a `finals/assets/` folder and update relative links. |
| P2 | Citation links marked “standard” in the project bibliography were not re-opened in this delivery. | Bibliographic accuracy is a release responsibility. | Click-test every reference used in the exported report/deck, then freeze the bibliography. |

## Score consistency decision

The current convention is **0.907551** (local held-out verification panel) and **0.920** (public leaderboard). Existing project Markdown contains historical variants. New isolated-run material uses this convention; broader report/deck regeneration remains paused until requested.

The requested pooled-R² callout is deliberately absent. The project’s current metric analysis says the historical pooled figure has been retired: pooled R² mechanically reweights the targets and is not the contest metric. It should not be revived merely because it looks stronger on a slide.

## Website readiness

The website implementation and README describe an offline Streamlit demo with prediction, interval, applicability tier, nearest analogue, provenance and SMILES-rewrite interaction. It was not launched during this delivery. Treat it as **presentation-ready only after a local rehearsal**; keep the recorded screenshot fallback visible in a second tab.
