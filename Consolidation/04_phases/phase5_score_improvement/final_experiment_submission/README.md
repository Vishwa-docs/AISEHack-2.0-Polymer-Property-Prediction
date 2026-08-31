# AISEHack 2.0 Polymer Property Prediction — Final Submission Pipeline

## Overview
This standalone pipeline trains an end-to-end multi-target ensemble to predict all 7 polymer properties:
- **$T_g$ (Glass Transition Temperature)**: Polymer backbone torsional energy potential fields & radical marker geometry.
- **$E_{gc}$ (Chain Bandgap)**: Deep stacking feature space with multi-scale GBDT ensembles.
- **$E_{gb}$ (Bulk Bandgap)**: Optimal transport manifold alignment.
- **$E_i$ (Ionisation Energy)**: High-order topological graph diffusion dynamics.
- **$E_{ea}$ (Electron Affinity)**: Extreme quantile residual outlier correction.
- **$\epsilon_r$ (Dielectric Constant)**: Optical-dielectric physical coupling constraints.
- **$n_c$ (Refractive Index)**: Multi-stage physics auto-refinement feedback loop.

## Compliance & Standalone Execution
- Uses **ONLY** official competition datasets: `train.csv`, `test.csv`, `PI1M.csv`, `smile_r3.csv`.
- **Zero external data, weights, or checkpoints.**
- Self-contained execution in a single run.

## Usage
```bash
python generate_submission.py --data-dir ../../Dataset --output-csv submission.csv
```
