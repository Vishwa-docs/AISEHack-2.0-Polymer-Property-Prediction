# Sandman finale — technical run-of-show

`index.html` is a 24-slide, 10-minute technical deck. It uses speaker-led visuals, but every
slide maps to a project experiment, architecture choice, or recorded evidence artifact.

| Slides | Time | What to establish |
|---|---:|---|
| 1–2 | 0:35 | Seven equally weighted target-wise R² tasks require a model that exposes its validation and operating boundary. |
| 3 | 0:30 | The D1–D9 experiment program selected the final system by retaining wins and rejecting attractive failures. |
| 4–6 | 1:15 | EDA: target imbalance, cross-property partner availability, and measured physical identities dictated the architecture. |
| 7–13 | 2:55 | Full architecture: shared representation, per-target lanes, physics gate, cross-fitted OOF assembly, GNN complement, diagnosis-led calibration. |
| 14–16 | 1:05 | Quantitative result, feature/learning evidence, and the key failed experiments that changed the final design. |
| 17–21 | 1:55 | Explainability fidelity and its limitation; representation invariance; structural generalization; applicability and uncertainty boundary. |
| 22 | 1:15 | Live dashboard: input → strict repeat check when eligible → estimate/interval/tier → scoped explanation. |
| 23–24 | 0:25 | Contribution, future experimental plan, code/report/dashboard links, references. |

## Interaction instructions

- Slides 5, 7, 11, 12, 17, 20–22 have progressive reveals. Pause after each reveal.
- Slide 5: use `0.935 → 0.907` only as a rejected leakage-path example, never as a result.
- Slide 10: state the key physical pivot exactly: an Egc learned residual gave LOO R² −0.82 on
  the 59 jointly measured polymers; the bare identity stayed.
- Slide 18: state the limitation: mean explanation-rank Spearman ρ = 0.472, below the 0.60 bar.
- Slide 19: claim strict repeat invariance only for the declared simple unbranched PEO grammar.
- Slide 22: demo PEO first, then accept a judge input only with its applicability tier visible.

## Source map

- Architecture details: `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/ARCHITECTURE.md`.
- Decision evidence: `.../Experiment_Logs/D1_physics_identities.md` through
  `D9_validation_explainability.md`.
- Completed-run figures: `.../SUBMISSION/evidence/`.
- Exact deliverable: `.../SUBMISSION/README.md` and `manifest.json`.
- Research citations: `REFERENCES.md`.

## Non-negotiable scope labels

1. The GNN is trained on official labelled data; it is not pretrained.
2. SHAP/removal evidence belongs to the proxy tree ensemble, not the full assembly.
3. The PEO primitive/dimer/trimer test is not universal PSMILES invariance.
4. Applicability is a structural-similarity warning, not an OOD accuracy guarantee.
5. The interval is experimental; do not advertise universal calibrated coverage.
