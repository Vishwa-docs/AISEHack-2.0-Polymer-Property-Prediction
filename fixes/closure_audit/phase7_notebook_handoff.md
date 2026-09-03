# STATUS - submission notebook conversion

Workspace: Phase 7 / 904_submission delivery.

## Deliverable

Sandman_Polymer_Property_Prediction_2_906.ipynb - one standalone notebook that reads only
the four official CSVs, writes submission_final.csv, and shows EDA, train/test-split metrics,
explainability, invariance, generalization, robustness and three research frontiers, with
per-stage checkpointing.

## Stage ledger

| # | stage | state | evidence |
|---|---|---|---|
| 0 | back up sources | done | backup/pipeline_final.py.orig sha d4bfdd48 |
| 1 | link the four official CSVs | done | data/{train,test,PI1M,smile_r3}.csv symlinks |
| 2 | build python 3.11.7 env | done | numpy 2.4.6, sklearn 1.9.0, rdkit 2026.03.5, lightgbm 4.7.0, matplotlib 3.11.1, shap 0.51.0, torch 2.13.0, torch_geometric |
| 3 | split engine into Part A / Part B | done | partA 9,440 lines (run_v57), partB 1,734 lines (evidence) |
| 4 | assemble notebook .py | done | 15,071 lines, 156 cells, syntax OK |
| 5 | jupytext to .ipynb | done | 157 cells (79 code / 78 markdown), kernel ppp311 |
| 6a | analysis half | done, 0 errors | 58 charts: eda 14, training 14, explainability 9, robustness 9, generalization 12 |
| 6b | Part A submission engine | done | 4,940-row baseline submission generated on macOS (R2 = 0.90229) |
| 6c | GNN blend (the +0.0045) | done | 3 seeds x 7 targets checkpointed in checkpoints/gnn_models.pkl (CV mean 0.8762) |
| 6d | evidence engine (18 checkpointed cells) | done | 18 evidence tables + 58 figure artifacts |
| 6e | frontiers: SAE, adversarial, element-holdout | done | mechanistic SAE, adversarial SMILES search, element-holdout zero-shot |
| 7 | verify contract, then score OUTSIDE notebook | done | 4,940 rows, unique IDs 1..4940, columns id,target, PASS |
| 8 | post-freeze verification | done | Local held-out panel Mean R2 = 0.907551 (all 7 targets verified) |

## Final Verified Score Summary (Local Held-Out Panel)

- **Mean R²**: **0.907551** (beats 0.90680 and 0.90230 champion baseline)
- **Per-Target Breakdown**:
  - `tg`: 0.9039 (n=2,732)
  - `egc`: 0.9213 (n=1,352)
  - `egb`: 0.9318 (n=224)
  - `ei`: 0.8741 (n=148)
  - `eea`: 0.9253 (n=147)
  - `nc`: 0.9101 (n=153)
  - `eps`: 0.8864 (n=153)

