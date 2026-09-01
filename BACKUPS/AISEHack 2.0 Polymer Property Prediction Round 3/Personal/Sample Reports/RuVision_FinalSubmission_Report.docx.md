**RuVision — PM2.5 Forecasting | AISEHack Phase 2**

Final Submission | Raj Shah  ·  Saptarshi Misra  ·  Hiteshri Shastri  ·  April 5, 2026

**Final Score: 0.8636**  

**What We Built**

A deep learning model that forecasts PM2.5 air pollution across India 16 hours ahead from 10 hours of observations, with a specific focus on accurately capturing pollution spike events (episodes) — the hardest and most impactful component of the competition metric.

**Final Architecture (v7 — 0.6M Parameters)**

**Key insight:** Reducing model capacity from 5M → 0.6M parameters improved leaderboard score by \+0.014. With \~2000 training samples, the v6 model (5M params) was overfitting. Removing CBAM attention and reducing depth gave better generalisation.

| Module | Config | What it does |
| :---- | :---- | :---- |
| Input | 20ch × 10 steps | PM2.5 \+ 8 met vars \+ 7 emission vars \+ sin/cos hour \+ ep\_freq\_prior \+ cluster\_id |
| ConvLSTM | 1 layer, dim=48 | Encodes 10-hour temporal dynamics (v6 had 2 layers, dim=64) |
| UNet | 2 levels, base=24ch | Multi-scale spatial features — 2 levels sufficient vs v6's 4 levels |
| No CBAM | Removed entirely | CBAM was overfitting at this data scale; plain skip concat performs better |
| Loss weights | α=1.0, β=3.0, γ=0.15 | Explicit GlobalSMAPE \+ EpSMAPE \+ EpCorr weighting (vs v6 implicit ramp) |
| Clustering | KMeans K=16 | 16 spatial regimes for loss weighting (vs v6's K=12) |
| Output | PM2.5 \+ Δ residual | Persistence residual head; TTA: E-W flip average |

**What's Novel (Phase 2 Changes)**

* **Architecture simplification —** Simplified architecture: 1-layer ConvLSTM \+ 2-level UNet \+ no CBAM \= 0.6M params. Validated that less is more at this data scale.

* **Episode-aware loss (Exp4-D) —** Explicit episode loss: α·GlobalSMAPE \+ β·EpSMAPE \+ γ·(1−EpCorr) with α=1.0, β=3.0, γ=0.15. The β=3.0 weight on EpisodeSMAPE directly targets the hardest competition sub-metric.

* **K=16 clustering (Exp3-B) —** K=16 spatial clustering replaces K=12. Finer regional granularity (IGP, W coast, E coast, NE India, Deccan, ocean) gives better per-cluster loss weighting.

* **Rigorous experiment design —** Sequential ablation methodology: each change tested independently on the leaderboard before combining, ensuring the gain from each component is real.

**Results (Final)**

| Metric | v6 | v7 Final | Change |
| :---- | :---- | :---- | :---- |
| Kaggle Leaderboard Score | 0.8403 (rank 24\) | 0.8636 (top 5%) | \+0.0233 |
| Model Parameters | \~5M | \~0.6M | −88% |
| Training Time | \~45 min (28 epochs) | \~15 min (28 epochs) | Faster per epoch |
| Inference | \~2s / batch (TTA) | \~2s / batch (TTA) | Same |

**Experiment Ablations (All vs. v6 \= 0.8403)**

Each experiment was run independently from the v6 baseline to isolate the effect of each change. v7 combines the three best-performing variants.

| Experiment | Change | Score | vs. v6 |
| :---- | :---- | :---- | :---- |
| Exp1-A | 1-layer LSTM, 2-level UNet, CBAM at bottleneck only | 0.8519 | \+0.0116 |
| Exp1-B ✓ | 1-layer LSTM, 2-level UNet, NO CBAM | 0.8542 | \+0.0139 |
| Exp2-A | 3-stage curriculum loss (MAE → EpSMAPE → full) | 0.8399 | −0.0004 |
| Exp2-B | Fixed-denominator SMAPE (convex in ŷ) | 0.8270 | −0.0133 |
| Exp2-C | Huber(δ=1) \+ EpCorr | 0.8314 | −0.0089 |
| Exp2-D | Log-cosh SMAPE | NaN loss | — |
| Exp3-A | KMeans K=8 | 0.8416 | \+0.0013 |
| Exp3-B ✓ | KMeans K=16 | 0.8449 | \+0.0046 |
| Exp3-C | DBSCAN clustering | 0.8412 | \+0.0009 |
| Exp3-D | ep\_freq-only cluster weights | 0.8422 | \+0.0019 |
| Exp3-E | Wider weight range \[0.2, 4.2\] | 0.8392 | −0.0011 |
| Exp4-A | α=1.0, β=2.0, γ=0.30 | 0.8601 | \+0.0198 |
| Exp4-B | α=1.0, β=1.0, γ=0.60 | 0.8520 | \+0.0117 |
| Exp4-C | α=0.5, β=2.0, γ=0.50 | 0.8593 | \+0.0190 |
| Exp4-D ✓ | α=1.0, β=3.0, γ=0.15 | 0.8608 | \+0.0205 |
| Exp4-E | α=0.5, β=1.0, γ=0.50 | 0.8555 | \+0.0152 |
| Exp5 | Lighter arch: base=16, dim=32 (all v7 else) | 0.8632 | \+0.0229 |
| Exp6-A | v7 arch \+ β=4.0, γ=0.15 | 0.8628 | \+0.0225 |
| v7 combined ✓ | Exp1-B \+ Exp3-B \+ Exp4-D | 0.8636 | \+0.0233 |
| v7 Regional | 6-block regional ensemble (2×3 grid) | 0.8552 | \+0.0149 |

**Challenges & Fixes**

| Problem | Root Cause | Fix |
| :---- | :---- | :---- |
| Overfitting | v6 5M-param model too large for \~2000 training samples | Reduced to 0.6M params — 1-layer LSTM, 2-level UNet, removed CBAM |
| Episode SMAPE | v6 used implicit episode weight ramp; EpSMAPE not explicitly optimised | Explicit β=3.0·EpSMAPE term as separate loss component |
| Spatial granularity | K=12 clusters too coarse; IGP boundaries imprecise | K=16 gives finer regional loss weighting |
| Regional ensemble failed | 6-block 2×3 regional models scored 0.8552 vs v7's 0.8636 | Reverted to single global model; less data per block hurts |
| Loss variants (Exp2) all worse | Fixed-denom SMAPE / Huber / log-cosh all hurt generalization | Kept v6 pseudo-Huber smooth SMAPE as base |
| SMAPE denominator (v5→v6) | v5 used 2× larger denom than competition formula | v6+ corrects to exact 0.5·(|y|+|ŷ|)+ε |

**Appendix: What Didn't Work**

**Exp2 — Alternative loss formulations:** All four variants (curriculum, fixed-denom SMAPE, Huber, log-cosh) scored below v6 baseline. The pseudo-Huber smooth SMAPE used in v6 is already well-calibrated to this problem; replacing the denominator or shape function reduces gradient signal on normal timesteps.

**Exp3-E — Wider cluster weight range \[0.2, 4.2\]:** Increasing the spatial weight spread beyond v6's \[0.5, 3.0\] hurt performance (0.8392 vs 0.8403). Too much emphasis on high-PM2.5 clusters starves the model of gradient from lower-concentration regions needed for GlobalSMAPE.

**Exp5 — Even lighter architecture (base=16, dim=32):** Scored 0.8632, marginally below v7's 0.8636 but very close. Diminishing returns: going from 5M→0.6M params helps significantly; 0.6M→0.15M gives negligible gain.

**Exp6 — Further loss weight fine-tuning:** Pushing β beyond 3.0 (to 4.0 or 5.0) or dropping γ below 0.15 did not improve beyond v7. The Exp4-D weights (α=1.0, β=3.0, γ=0.15) appear to be near-optimal for this task.

**Regional ensemble (v7-regional):** Training 6 independent models on spatial sub-grids (140×124 → 2×3 blocks of 70×41) and stitching with cosine overlap blending scored 0.8552 — worse than the single global model (0.8636). Each block has significantly fewer training samples per model, hurting generalisation.

**Links & Artefacts**

| Artefact | Link / Location |
| :---- | :---- |
| GitHub Repository | Public repo — ANRF Open License, all notebooks, README, GENAI\_USAGE.md  |
| Kaggle Notebook (v7 final) | https://www.kaggle.com/code/rajshahcx28/training-pipeline-v7 |
| Model Checkpoint (v7\_final.pt) | Kaggle Hub — https://www.kaggle.com/datasets/rajshahcx28/pm25-v7 |
| Inference Notebook | inference.ipynb — standalone, loads checkpoint and runs TTA |
| Dataset | WRF-Chem 2016 (Apr, Jul, Oct, Dec) — provided by IIT Delhi / AISEHack |
| AI Tool Disclosure | GENAI\_USAGE.md in repo — Claude (Anthropic) used for code generation, experiment design, and documentation |

**Declaration**

We, the team RuVision — Raj Shah, Saptarshi Misra, Hiteshri Shastri — have made our submissions wholly based on our own efforts and have not taken help from third parties or members outside the team. GenAI tools (Claude by Anthropic) were used for code generation and documentation; all modelling decisions were made by the team and validated on the leaderboard.

**Digital signatures:**

Raj Shah          Saptarshi Misra          Hiteshri Shastri