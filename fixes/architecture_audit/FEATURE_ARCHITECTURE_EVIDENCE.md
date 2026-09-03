# Feature strategy and architecture — claim-to-evidence map

This is the organiser-facing technical spine. It answers, for every retained component:
**what chemical or statistical signal it represents, why it is present, how it was checked, and
where the boundary lies.** Values below are from the recorded full evidence run and must be
reconfirmed against `fixes/isolated_runs/outputs/` before public promotion.

## 1. Design premise: the metric and data regime decide the architecture

The score is the unweighted mean of seven target-wise R² values. Therefore, a pooled objective
would over-emphasise the high-variance Tg task and hide weak scarce targets. The architecture
uses target-specific lanes with canonical-structure grouped validation rather than a single
joint regressor. The six DFT-derived targets have a different-property route for many repeated
structures; Tg is chiefly a structure-to-property extrapolation problem. This is the reason for
separate leaves and guarded cross-property routes—not a generic preference for ensembling.

| Decision | Why it is scientifically justified | Executed evidence | Reference |
|---|---|---|---|
| Canonical graph grouping | Equivalent SMILES must not straddle train/validation; it also establishes the invariance test unit. | `cv_validation_table.csv`; SMILES-rewrite tables | C1, C3, C5 |
| Per-target lanes | Sample counts and physical observability differ sharply between Tg and the scarce electronic/optical targets. | per-target architecture and grouped validation table | G1, P2 |
| Classical trees/kernels as the spine | Small tabular targets require measured baselines; deep models are complementary only when their residual errors diversify. | Experiment log D4/D5; out-of-fold assembly | G1, L1, L2, C4, L5 |
| Physics as guarded routes | A relation is used only where its allowed inputs exist and only after its held-out route check. | band-edge, bulk-gap and dielectric identity tables | M2, M4–M6 |

## 2. Feature families: representation, benefit and risk

| Feature family | Chemical/statistical role | Why retain or constrain it | Evidence and test | Boundary | Reference |
|---|---|---|---|---|---|
| RDKit 2D/3D descriptors | size, polarity, ring/connectivity, surface and electronic proxies | Interpretable bulk chemistry, especially important for scarce-property leaves. | Recorded ablation: removing descriptors lowers Ei R² by 0.053. | 3D/rdEHT path is environment-sensitive. | C1, P3 |
| Morgan count fingerprints (r=2) | local functional-group and bonding motifs | Complement descriptors with local substructure counts. | Recorded ablation: removing Morgan features lowers Tg by 0.007 and Egc by 0.016; it also affects small electronic targets. | A bit alone is not a causal functional-group explanation. | C1, C2 |
| Polymer-Genome atomic triples | coordination-labelled polymer environments | Polymer-specific vocabulary beyond generic fingerprints. | Keep only where ablation/target routing supports it. | Do not imply every target benefits equally. | P3 |
| Tanimoto kernel | smooth local interpolation among known structures | Useful when labelled n is small and the applicability-domain similarity is meaningful. | Nearest-neighbour tier/error table; structured validation ladder. | Not a licence to extrapolate below the similarity domain. | C1, C2, U3 |
| PI1M label-free SVD | broad polymer-string prior | Use only because it is fit without labels and measured against supervised controls. | D3 experiment record; retained as a residual representation. | It inherits string-order sensitivity. | P5, P1, P4 |
| Character n-grams | residual string signal | Deliberately constrained rather than presented as a structural anchor. | Separate rewrite/invariance analysis. | Not invariant by construction; never average away its cost. | C5 |
| Cross-property covariates | measured companion property of the same structure | Physical routes can reduce uncertainty for some DFT targets. | Identity route checks and grouped validation. | Requested evaluation label is never read; route is unavailable for genuine new-polymer screening. | P2 |

## 3. Architecture: hypothesis → test → decision

| Stage | Hypothesis | Test | Decision and why |
|---|---|---|---|
| A. Complementary representations | Different representations make different errors. | Feature-family ablations and out-of-fold predictions. | Keep graph descriptors/fingerprints as the invariance backbone; allow a measured residual arm only when useful. |
| B. Per-target leaves | n≈200 and n≈4,000 should not share a default estimator. | Structure-grouped target-specific validation. | Use boosted/tree models where data support them; small-n kernel/probabilistic/classical leaves where their evidence earns them. |
| C. Guarded physics overlays | Known relations should reduce error more reliably than a flexible residual in low-data regimes. | Route-specific held-out checks. | Enforce/use only successful routes; reject a learned correction when it adds noise. |
| D. Out-of-fold non-negative assembly | Diverse predictions can improve only if weights do not fit validation noise. | Cross-fitted NNLS versus unconstrained alternatives. | Use non-negative out-of-fold weights; they are interpretable as contribution shares and limit unstable cancellation. |
| E. Calibration and domain layer | A scalar without its scope is not deployable. | Conformal table, applicability analysis, seed stability and novelty ladder. | Serve interval, closest analogue and tier; do not treat interval width as a universal error predictor. |

## 4. Qualitative proof package

| Criterion | Test and visual | Recorded result | Defensible claim |
|---|---|---|---|
| **Invariance** | 500-polymer randomised-SMILES protocol; graph violation table; `qualitative_scorecard.png` | graph prediction violations are zero at the recorded thresholds; attribution cosine mean 0.979737, minimum 0.946731 | Equivalent spellings preserve the graph-model prediction and its explanation under the tested rewrite protocol. |
| **Explainability** | SHAP-ranked masking versus random masking | At 5% masking, mean R² drop is 0.810487 for top-ranked features versus 0.022373 random, across 7 targets. | Explanations are intervention-tested, not merely displayed. |
| **Robustness** | conformal coverage, seed variation, error–uncertainty audit | coverage is within ±3 percentage points for 7/7 targets; Tg seed SD 0.001730; error–uncertainty correlation ≥0.30 for 5/7 targets. | The recorded protocol supports calibrated coverage and seed stability; uncertainty correlation is not universal. |
| **Generalizability** | canonical-group and scaffold structured holdouts | mean R² 0.823752 and 0.658082 respectively; all seven targets remain positive. | The method remains informative under these stricter structural splits, with applicability tiers for farther inputs. |

### Boundaries that must remain visible

1. Raw cross-model feature-rank agreement is 0.472223. It is a secondary sensitivity
   diagnostic below the stated bar; explanation fidelity and rewrite-stable attributions are
   the primary positive evidence.
2. Family/ultra-low-similarity and extreme-tail regimes are not advertised as reliable
   deployment settings. The website must show tier and interval before a scalar prediction.
3. The portable website model is not the full transductive competition ensemble. It is a
   real per-row model with separately displayed out-of-fold diagnostics.

## 5. Design goal: from prediction to candidate screening

The defensible current capability is **forward screening**: a scientist proposes a chemically
plausible repeat unit, selects a desired property, receives a prediction, interval, nearest
analogue, applicability tier and active live-model descriptors, then decides whether synthesis
or higher-fidelity simulation is justified. It does not claim to generate a synthesizable polymer
or to replace experimental measurement.

Use structural counterfactual evidence to explain direction rather than to promise universal
rules. In the recorded counterfactual suite, pendant-phenyl Tg edits agreed with the expected
direction on 12/14 cases, unsaturation decreased Egc on 17/17 cases, and fluorination increased
Ei on 6/7 cases. Ether-insertion Tg edits agreed on 9/14 cases; this is the appropriate example
of a competing-effect boundary, not a feature to overstate.

## 6. Evaluation/retraining story

1. Fix the architecture and evaluate it on a local held-out verification panel using grouped
   structures. That is where the reported mean R² is measured.
2. Freeze the decision record and evidence suite.
3. Retrain the selected pipeline on all official training labels to create the submission model.
4. The retrained submission has no additional local held-out score by itself. The public
   leaderboard is an external outcome, not a tuning target.

This sequence prevents the common error of presenting an in-sample post-retraining number as a
test score.

## 7. Required talk references

- **Feature/architecture:** G1, P3, P5, C1, C2, C4, L1, L2, L5.
- **Invariance:** C5; RDKit/C1 for graph canonicalisation.
- **Explainability:** X1–X3; especially X3 for the masking control.
- **Robustness/generalization:** U1, U3, C3.
- **Polymer-design framing:** M2 and M7, with an explicit note that real Tg depends on
  molecular weight, tacticity, processing and morphology.
