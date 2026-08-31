# Round 3 Final Pipeline — Presentation Story

## What this is

A single, standalone, oracle-free file — `pipeline_final.py` — that:
1. **Reproduces the V57 submission** (verified R² 0.9035 / final-oracle 0.9023)
   from official data only, in one run, from scratch.
2. **Generates the complete Round-3 evidence bundle** for the three judged
   themes — explainability, polymer-invariance robustness, generalization —
   into `outputs/`.

The V57 engine inside `pipeline_final.py` is **byte-identical** to the verified
standalone `pipeline_v57_final.py` (570,044 chars, confirmed by comparison),
so the score is preserved by construction: the evidence suite is purely
additive and never touches the prediction path.

## The one-file command

```bash
# full run: submission + evidence (hours)
python pipeline_final.py --mode full --data-dir ../Dataset --out submission.csv --out-dir outputs

# submission only (for Kaggle)
python pipeline_final.py --mode submission --data-dir ../Dataset --out submission.csv

# evidence only, on an existing submission (fast demo)
python pipeline_final.py --mode evidence --data-dir ../Dataset --submission submission.csv --out-dir outputs

# tiny-data demo (minutes, for a live presentation)
PHASE4_SMOKE=1 python pipeline_final.py --mode full --smoke --data-dir ../Dataset
```

Afterwards open `outputs/TRUSTWORTHINESS_REPORT.html` (single judge-facing page
with every plot/table) and `outputs/scorecard.md` (PASS/FAIL per requirement).

## The three themes, with the evidence that proves them

### 1. Model Explainability (R1)
- **Global SHAP** per target (beeswarms + `shap_top20_per_target.csv`): the top
  features are chemically defensible — tg → aromatic-ring/EState descriptors,
  nc/eps → PEOE-VSA polarisability, egc → conjugation-adjacent features.
- **Fidelity test** (masking SHAP-top features vs random): dropping the top-10%
  SHAP features collapses validation R² (0.85 drop) while random masking barely
  moves it (0.04) — the explanations are *faithful*, not decorative.
- **Local SHAP + atom maps**: per-polymer force plots and RDKit SimilarityMap
  coloring show *which atoms* drive each prediction (2–3 polymers per target).
- **Physics-decomposed explanation**: eps is explained as `nc² + ionic` with
  separate SHAP for each channel (R1.5) — we explain the *components*.
- **Cross-model agreement** (R1.4, upgraded): SHAP-consistent importance across
  Ridge/ExtraTrees/LightGBM shows *different model families agree* on what
  matters (mean Spearman ρ reported honestly).
- **Linear probes** (extended): layer 1 of the Tg MLP encodes aromaticity with
  R² = 0.90 — the model internally represents the chemistry that explains its
  predictions.

### 2. Robustness to Polymer Invariances (R2)
- **SMILES prediction invariance**: 500 polymers × 30 randomized SMILES each →
  graph-feature prediction std ≤ 0.23% of train std (Tg), ~0.000% for the DFT
  targets. The full ensemble (including string-sensitive char n-grams) shows
  6–14% std — quantified honestly, with the 1σ violation rate still 0.1–1.5%
  (< 5% requirement).
- **Canonicalization audit**: all 30 spellings reduce to one canonical form.
- **Attribution invariance**: SHAP attribution vectors across equivalent SMILES
  agree at cosine 0.95–0.99 (requirement ≥ 0.70) — *the reasons are stable too*.
- **Oligomer (chain-extension) invariance**: monomer vs dimer predictions within
  |Δ| < 3σ for 96% of pairs.
- **Data augmentation (NEW)**: training the proxy with 3 randomized SMILES per
  polymer keeps OOF R² (0.780 → 0.784) while cutting prediction std across
  spellings ~10× — evidence that augmentation is a valid invariance lever with
  no accuracy penalty.
- **Homologous-series relation demo (NEW)**: for `*`-endcapped repeat units,
  predicted Tg vs chain length (1/n) fits Flory–Fox linearly (median R² ≈ 0.99)
  — the model *finds the physical relation*, not a lookup.

### 3. Proven Generalization (R4) + Reliability (R3)
- **Generalization ladder**: R² decays smoothly from random CV → canonical-group
  → scaffold → family → low-similarity → ultra-low-similarity holdout — the
  "staircase" a trustworthy model should show.
- **Structured CV** (random / group / scaffold / low-sim) per target.
- **Cross-conformal intervals** (upgraded): all 4,940 test predictions ship with
  80/90/95% intervals calibrated per fold (cross-conformal, Vovk 2015).
- **Applicability domain**: error rises monotonically as nearest-train Tanimoto
  similarity falls; every test row gets an AD tier.
- **Seed stability**: mean OOF R² = 0.9066 ± 0.0018 across 5 seeds (< 0.005).
- **Tail performance**: R² holds in the top/bottom 10% of property extremes.
- **External verification (post-freeze)**: frozen submission vs ground-truth
  panels — tg 0.897, egc 0.911, egb 0.927, ei 0.871, eea 0.918, nc 0.909,
  eps 0.885 (all six DFT targets meet the ≥ 0.85–0.88 bar).

## Honest limitations (reported, not hidden)

- R1.4 cross-model agreement ρ ≈ 0.47–0.60: different model families rank
  features differently; SHAP-consistent attribution is used and reported as-is.
- R3.2 conformal coverage on the tiny DFT targets: ±4.5% sampling noise with
  ~45 validation rows; Tg/Egc are within ±3%.
- R3.3 ensemble uncertainty–error correlation: ρ 0.13–0.44 depending on target;
  tree-spread uncertainty (ρ ≈ 0.44 on Tg) is the upgraded estimator, with an
  MLP/MC-dropout path documented for further gains.

## How to present it

1. **Live demo (5 min)**: `PHASE4_SMOKE=1 python pipeline_final.py --mode full --smoke`
   → opens the HTML report with every plot.
2. **Deep-dive (15 min)**: walk the scorecard PASS/FAIL, then the three themes
   above, pointing at specific outputs.
3. **The punchline**: "same prediction, same reasons, and we can prove it" —
   invariance std 0.23%, attribution cosine 0.95–0.99, and a model whose layer-1
   encodes aromaticity at R² = 0.90.
