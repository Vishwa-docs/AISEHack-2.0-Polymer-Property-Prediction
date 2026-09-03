# Isolated-run output audit

- **Source (read only):** `/Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3/fixes/isolated_runs/outputs`
- **Files observed:** 58
- **Newest observed artifact:** 2026-09-02T15:21:52+05:30
- **Promotion ready:** NO

This is a filesystem-completeness audit. It does not prove that a notebook kernel is idle or that a result is scientifically valid.

## Standard notebook artifacts

| Status | Artifact |
|---|---|
| present | `eda/novelty_two_regimes.png` |
| present | `training/parity_plots.png` |
| present | `explainability/fidelity_curves.png` |
| present | `robustness/smiles_invariance.png` |
| present | `generalization/generalization_ladder.png` |
| present | `generalization/applicability_domain.png` |

## Release-gate evidence artifacts

| Status | Artifact |
|---|---|
| missing | `explanation_agreement.csv` |
| missing | `attribution_invariance_per_target.csv` |
| missing | `smiles_invariance_graph_violation_summary.csv` |
| missing | `fidelity_table.csv` |
| missing | `generalization_ladder.csv` |
| missing | `conformal_coverage_table.csv` |
| missing | `error_uncertainty_correlation.csv` |
| missing | `augmentation_experiment.csv` |
| missing | `scorecard.md` |

## Next action

Do not promote or quote release-gated uncertainty claims yet. Let the notebook reach its late evidence-engine and scorecard cells, then rerun this audit.
