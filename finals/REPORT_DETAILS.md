# Report production record

## Scope and format

`finals/Report.md` is the final technical report: target **no more than five rendered
main pages**, with references and appendices additional. It follows the official
check-in's four required sections and keeps only the architecture, measured results,
qualitative validation, pivotal negative results and release plan in the core.

The official template suggests 3–5 pages; this draft is deliberately designed to fit
the five-page end of that range. `finals/REPORT_10_Page.md` is a prior long-form
alternative retained for comparison, not the canonical final draft. The template at
`Personal/Midnight_Report/Finals_Report/Polymer Track Midnight Check-in Submission Template.docx.md`
is the controlling format.

## Claim policy

| Claim class | Permitted treatment |
|---|---|
| Submitted outcome | local held-out verification-panel mean R² **0.907551** and public leaderboard **0.920** |
| Compact verification model | the named `verification_panel_scores.csv` values, clearly labelled as a distinct compact model |
| Invariance/generalisation/applicability | only values traceable to codebase `outputs/evidence_tables/` |
| Explanation fidelity | state proxy/model scope and intervention; do not claim chemical causality |
| Coverage and error–uncertainty correlation | **release-gated**: archived tables conflict, so do not report as a win until the isolated rerun resolves them |
| External materials | illustrative context only until a pre-registered multi-material panel is complete |

The detailed audit is `fixes/qualitative_evidence/claim_evidence_map.md`. Do not
replace an unfavorable or uncertain value with a historical scorecard value.

## Required before DOCX/PDF export

1. Replace the team-name and GitHub-release placeholders.
2. Confirm the authorship declaration and GenAI disclosure wording with the team.
3. Add `fixes/isolated_runs/outputs/training/parity_plots.png` only after its
   active run completes and its source/split are validated.
4. Re-run `finals/scripts/build_verified_qualitative_figure.py` with `MPLBACKEND=Agg`
   if any of its input tables change; it exports PNG and PDF at 300 dpi.
5. Convert to DOCX/PDF and size the three in-flow visuals so the main manuscript is
   no more than five pages; references and appendices follow afterwards.

## Approved visual assets

| Asset | Local path | Role |
|---|---|---|
| Architecture | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/architecture.png` | Section 1 workflow |
| Verified qualitative figure | `finals/assets/verified_qualitative_evidence.png` | Section 1 qualitative proof |
| SHAP fidelity | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/explainability/fidelity_curve_tg.png` | Section 2 explainability proof |
| Parity plot | `fixes/isolated_runs/outputs/training/parity_plots.png` | pending run validation |
| Report claim map | `finals/REPORT_CLAIM_EVIDENCE.md` | audit companion for every major claim |

## Research draft boundary

`Personal/Research_Paper/paper.md` is an in-preparation, out-of-competition
Round-2 research draft. It is useful background for later work on replication and
negative results, but it is neither a published paper nor evidence for this
hackathon check-in. Do not cite its internal scores in the report.
