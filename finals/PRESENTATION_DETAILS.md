# Presentation generation details

## Objective and constraints

Create a six-minute expert presentation that proves understanding rather than reciting a pipeline. The central evidence flow is: **EDA revealed two regimes → feature roles and architecture follow the regimes → physics and validation constrain the system → explainability/invariance/generalisation are measured → the website makes the boundaries visible.**

## Non-negotiable content

1. State the two reporting outcomes exactly once in the title/results sequence: **0.907551** local held-out verification-panel mean R² and **0.920** public leaderboard.
2. Explain the unweighted seven-target metric before describing the architecture.
3. Explain the job and guard of every feature family before showing architecture; do not call an ablation a chemical-causality claim.
4. Show EDA, architecture, SHAP fidelity, SMILES/attribution invariance, and the novelty/generalisation ladder.
5. Demonstrate the website as an explicit run-inference interaction with interval, applicability tier, analogue, local explanation, 3D conformer caveat and rewrite test—not as a leaderboard lookup.
6. Include future direction and voluntarily identify the retained boundaries: raw rank agreement, process variables, two uncertainty-correlation targets and the pre-registered external-panel requirement.
6. Make the field-level contribution explicit: test explanation fidelity, encoding invariance and applicability alongside accuracy.

## Visual priority

| Slide | Asset | Single takeaway |
|---:|---|---|
| 3 | `outputs/eda/novelty_two_regimes.png` | data regime determined the model design |
| 4 | `outputs/architecture.png` | five stages have distinct roles |
| 6 | `outputs/explainability/shap_beeswarm_tg.png` | explanations are tested by intervention |
| 7 | `outputs/robustness/smiles_invariance_boxplot.png` | same graph, stable output/reason |
| 8 | `outputs/generalization/generalization_ladder_plot.png` | novelty creates a measurable boundary |
| 9 | website evidence console | qualified live inference, not a lookup |

## Speaker delivery rules

- State the answer first, then the mechanism. “0.851 versus 0.043 under masking” is stronger than “SHAP looks reasonable.”
- Say “we tested” only when pointing to a chart, table or explicit experiment artefact.
- When challenged on a weakness, agree with the premise, show the scope/cause, state the next test.
- Do not call a relation an identity unless it is physically defined and the route is guarded.
- Do not call the deployment model the same as the full competition ensemble.
- Keep citations in notes or a final reference slide; never leave a full URL in the centre of a technical slide.

## 360-second timing

| Segment | Seconds |
|---|---:|
| title + result | 18 |
| task + metric | 30 |
| EDA | 34 |
| architecture | 48 |
| experimental discipline | 32 |
| explainability | 33 |
| invariance | 33 |
| generalisation | 30 |
| live demo | 42 |
| future/close | 60 |

## Regeneration prompt

Generate a sparse 10-slide, six-minute deck from `finals/Presentation.md`. Each slide needs: title, ≤30 on-slide words (excluding labels), one existing visual asset, the exact spoken script, the claim supported, and one likely hostile question with a 20-second answer. Use no invented metrics. Performance outcomes must be stated as local held-out verification-panel mean R² 0.907551 and public leaderboard 0.920. End with limitations and a future-work slide. Include a 45-second offline website demo and a screenshot fallback; do not describe the planned 3D/live-inference interaction as implemented until the acceptance checks in `finals/WEBSITE_DEMO_SPEC.md` pass.
