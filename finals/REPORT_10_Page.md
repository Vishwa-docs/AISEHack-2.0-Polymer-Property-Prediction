# Physics-guided, evidence-first polymer property prediction from repeat-unit SMILES

**ANRF AISEHack 2.0 · Polymer Property Prediction**  
**Team:** Sandman *(confirm official name before external export)*  
**Primary domain:** Materials science / polymer informatics  
**Status:** expanded report alternative. The official-template version is [`Report.md`](Report.md).

**Reported submission outcome:** local held-out verification-panel mean R² **0.907551**; public leaderboard **0.920**. The contest metric is the unweighted mean of the seven target-wise R² values, not a pooled-regression score.

## Executive summary

We predict seven polymer properties from repeat-unit SMILES: glass-transition temperature (Tg), chain and bulk bandgaps (Egc and Egb), ionisation energy (Ei), electron affinity (Eea), refractive index (n), and dielectric constant (ε). The system combines canonical molecular-graph features, target-specialised classical models, narrow physics-guided routes, and an explicitly scoped graph-neural second opinion. Its submitted notebook recorded a mean R² of **0.907551** on the local held-out verification panel and **0.920** on the public leaderboard.

The main contribution is not a claim that one architecture is universally best. It is a decision framework for a difficult small-data polymer task. We first establish that the seven targets are not interchangeable: Tg is primarily a structure-to-property extrapolation task, whereas most electronic and optical targets have cross-property support for the same structures. We then use grouped structural validation to decide which representation, model lane, or physical relation deserves inclusion. A physical identity is a guarded option, not an excuse to read a label from an evaluation row; it is used only when its legal inputs are available and it improves an out-of-fold route.

The report also treats qualitative evidence as a first-class deliverable. Prediction invariance is tested under valid randomised SMILES spellings of the same polymer graph. Explanation stability is tested under the same rewrites. Explanation fidelity is tested by feature removal rather than by a coloured SHAP plot alone. Generalisation is measured under canonical-group and scaffold splits, and the website exposes a nearest-neighbour applicability tier. The favorable evidence is bounded: archived interval-coverage and error–uncertainty summaries currently conflict across source tables, so they are not reported as wins until the isolated Python-3.11 run reproduces a single authoritative result. This is a release gate, not a weakness hidden from the reader.

## 1. Problem formulation and scientific motivation

### 1.1 Seven linked prediction problems, not one pooled target

The repeat-unit string has two connection points and denotes an indefinitely repeating chain, not a conventional finite molecule. This makes polymer property prediction different from molecular property prediction in two ways. First, a SMILES string omits material-level factors such as molecular-weight distribution, tacticity, morphology, processing history and measurement conditions. Second, valid string spellings are not unique: atom ordering and repeat-unit cut point can change without changing the chemical graph. A useful system must therefore disclose the boundary imposed by its representation and test the particular invariances it claims.

The seven labels also arise from different mechanisms. Tg reflects segmental mobility, rigidity, packing and free volume. Egc, Egb, Ei and Eea are electronic-structure quantities with a band-edge relationship. The optical pair n and ε admit a useful decomposition into electronic and ionic/polar contributions. The data regime is equally heterogeneous: Tg has thousands of labels while several electronic and optical targets have only roughly 150–300. A pooled raw regression loss would be dominated by Tg variance and would make small targets appear numerically unimportant despite their one-seventh contest weight.

Accordingly, the objective is

\[
\mathrm{Score}=\frac{1}{7}\sum_{t\in\{Tg,Egc,Egb,Ei,Eea,n,\epsilon\}}R_t^2.
\]

Every target receives equal metric weight. This dictates the architecture: shared molecular representation, target-specific estimation, and only guarded sharing of physics or sibling-property information.

### 1.2 EDA changed the modelling strategy

The overlap audit separates two regimes. There are no exact polymer–property label duplicates between training and evaluation for the six DFT-derived targets. However, many evaluation structures occur in training under a *different* DFT property. Tg is materially different: it has much less cross-property support and therefore acts more like direct structural extrapolation. The result is not a loophole; it is a data characteristic that must be made explicit. It motivates separate target lanes, canonical-structure grouping, and a rigorous availability guard on any cross-property route.

![Figure 1. Structure and label-overlap audit.](../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/eda/novelty_two_regimes.png)

**Figure 1.** The two-regime EDA result. It explains why a homogeneous single-model story would be misleading and why every physics route needs a legal-input guard.

The same audit establishes why random row splits are insufficient. Canonicalisation exposes structures that may otherwise appear under different strings. Grouping by canonical polymer structure prevents the model from validating on an alternative spelling of a structure it has already seen.

## 2. Feature strategy and architecture

![Figure 2. Five-stage architecture.](../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/architecture.png)

**Figure 2.** Shared representation; seven target-specific model lanes; guarded physical overlays; out-of-fold non-negative assembly; and target-aware calibration.

### 2.1 Complementary molecular representations

The representation deliberately combines signals with different failure modes rather than accumulating many correlated columns.

| Family | Scientific role | Why it is not used alone |
|---|---|---|
| RDKit 2D/3D descriptors | size, polarity, surface, ring content, charge-related and geometric proxies | a global descriptor can miss a decisive local motif |
| Morgan count fingerprints | local environments and repeated functional substructures | fingerprints do not directly provide the bulk physical descriptors useful in low-data regression |
| Polymer Genome-style atomic triples | polymer-specific atomic environments | their contribution is retained only when a target-level ablation supports it |
| Tanimoto kernel route | smooth close-neighbour interpolation | cannot safely extrapolate outside similar structure space |
| character n-grams + Ridge | limited residual sequence signal | SMILES-order-sensitive, so deliberately not the invariance backbone |
| label-free polymer-corpus SVD | structural context learned without target labels | used only where grouped validation supports it |
| compact GINE graph network | connectivity-aware, decorrelated second opinion | not assumed superior to trees in the smallest label regimes |

This design follows two practical observations. Descriptor-led tree ensembles are powerful small-tabular baselines, and a graph model contributes only when it makes different errors—not merely when it adds architectural novelty. The string channel is intentionally a restrained residual correction: the graph/canonical path is the route on which invariance claims are made.

### 2.2 Target-specific lanes

Tg and Egc have sufficient data for boosted/tree ensembles. Small band-edge targets combine carefully selected kernel, probabilistic and compact neural members. Optical targets exploit polar-group information and the relation between n and ε. The final selection is not based on a generic “deep versus classical” preference; it follows grouped out-of-fold evidence target by target.

The graph neural network is blended as a second opinion. Its purpose is diversity of error. The blend uses non-negative least squares fit on out-of-fold predictions so that a superficially good result is less likely to rely on large cancelling positive and negative weights fitted to noise.

### 2.3 Guarded physical overlays

Three physical relationships are operationally useful:

1. **Band-edge relation:** Egc is connected to Ei−Eea. On a co-measured subset this relation is strong; adding a flexible learned residual in the smallest-data setting worsened leave-one-out performance, so the bare relation is retained when it is legally available.
2. **Bulk versus chain bandgap:** Egb can be represented as an affine function of Egc plus a validated residual, reflecting bulk packing/interchain effects.
3. **Optical/ionic decomposition:** ε is written as n² plus an ionic/polar remainder. The audited co-measured subset supports a non-negative remainder and a more tightly conditioned learning problem than direct raw ε regression.

Each overlay has two safeguards: its partner input must come from an allowed training-side measurement, and it must beat or match the learned route on the equivalent held-out slice. These controls matter because a seemingly innocuous sibling-property fallback can become target leakage if it accidentally creates a path from a row’s answer back into its prediction.

## 3. Quantitative results and model-selection discipline

The submitted notebook achieved **0.907551** mean R² on the local held-out verification panel and **0.920** on the public leaderboard. The metric is target-balanced, so a single pooled R² would not be an adequate diagnostic. A compact verification model evaluated on the named verification panel records target-wise R² from **0.871** (Ei) to **0.927** (Egb), with mean **0.902509**. That compact table is reported separately from the final submission architecture; it is useful for transparent per-target inspection, not a substitute for the submitted score.

The selection rule was simple: a mechanism is retained when it improves a structure-aware validation route or provides a physically validated constraint, and removed otherwise. Important negative results are therefore part of the architecture:

- A learned residual on the low-data Egc=Ei−Eea relation overfit rather than improving the identity.
- Several self-supervised and generic molecular representation trials did not displace the compact in-domain feature set at this scale.
- A graph model is not advertised as a universal replacement for classical small-data models; its role is limited to validated error diversity.
- Global recalibration is not treated as a cure for sparse, out-of-domain failures.

This is the practical meaning of “physics-guided”: physics structures the candidate space and supplies a falsifiable check, while grouped validation decides whether a component earns a place in the final system.

## 4. Explainability: from feature attribution to discovery questions

### 4.1 What is explained

The project uses tree-based SHAP explanations for selected model members, but a local bar chart is not interpreted as causal evidence. Explanations are used to turn a newly proposed repeat unit into a testable materials question.

- For **electronic discovery** (Egc, Egb, Ei, Eea), the explanation can highlight ring/conjugation-adjacent descriptors, local electronic environments and the consistency of band-edge predictions. The scientist’s question becomes: which changes are predicted to move the band edges, and do the companion properties remain physically compatible?
- For **optical discovery** (n and ε), the interface separates the n² optical channel from the ionic/polar remainder. This is more informative than one aggregate feature-importance list: it distinguishes a predicted refractive-index contribution from a predicted polar/dielectric contribution.
- For **Tg**, descriptors associated with rigidity, aromatic surface, flexibility and local motif counts can generate hypotheses about segmental motion and free volume. They do not replace missing molecular weight, tacticity or thermal-history information.

### 4.2 Fidelity intervention

![Figure 3. Tg feature-removal fidelity curve.](../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/explainability/fidelity_curve_tg.png)

**Figure 3.** A proxy-model fidelity test: removing the top 10% SHAP-ranked features causes a markedly larger loss (0.851 R²) than removing an equally sized random feature set (0.043). It demonstrates load-bearing predictive information in the stated feature space; it does not prove a feature is a causal molecular mechanism.

This intervention is more useful than decorative attribution because it asks a falsifiable question: if these ranked features are important to the model, does removing them damage performance more than a matched random removal? The answer is yes for the stated Tg proxy test. Cross-model per-descriptor rank agreement is nevertheless below the pre-set threshold (mean Spearman ρ=0.472 versus 0.60). This is retained as a limitation: correlated descriptor families can be ranked differently by linear and tree models. The proposed improvement is to compare chemically grouped concepts and re-run the same removal test, rather than to suppress the disagreement.

## 5. Invariance to polymer representation

SMILES invariance is an engineering property to test, not a label to attach to a model. The experiment starts with a polymer graph, generates valid randomised SMILES spellings, and compares graph-derived predictions under these equivalent encodings. In the recorded table, the graph feature path has zero 1σ violations for each of the seven targets; the worst target standard deviation is **0.230%** of the relevant training-standard-deviation scale. The scope is important: it concerns valid alternative spellings of the same graph and graph-derived features. It does not claim invariance to a genuinely different chemistry, changed stereochemistry, molecular weight, tacticity or processing history.

![Figure 4. Prediction variation under randomised valid SMILES spellings.](../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/robustness/smiles_invariance_boxplot.png)

The second test asks whether the explanation is preserved as well as the scalar prediction. Mean attribution cosine similarity across targets is **0.980**, with the recorded range **0.947–0.996**. Together, the two tests support a precise statement: under the project’s valid-SMILES rewrite protocol, the graph-derived route gives nearly unchanged predictions and stable feature-space explanations.

## 6. Robustness and generalisability

### 6.1 Generalisation ladder

Random splits can exaggerate performance by placing near-identical structural variants in both folds. We therefore report a ladder from canonical-group to scaffold splitting. The recorded mean target-wise R² is **0.824** under canonical grouping and **0.660** under scaffold splitting, with all seven target values positive in both regimes. The decline is expected: unseen scaffold families are a harder question than a different row from a familiar family. Its value is not that it is high in every regime, but that it reveals the boundary rather than hiding it in a single aggregate number.

![Figure 5. Generalisation ladder.](../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/generalization/generalization_ladder_plot.png)

### 6.2 Applicability domain

The strongest currently verified robustness result is a usable boundary condition. For Tg, error increases monotonically as nearest-training Tanimoto similarity falls: MAE rises from **14.8 °C** in the ≥0.9 tier to **43.6 °C** below 0.5. That does not make the low-similarity prediction useless; it makes the correct user interface clear. A prediction should show its nearest analogue, similarity tier and uncertainty context, and it should warn rather than sound certain when the polymer is outside the representation’s support.

![Figure 6. Applicability-domain analysis.](../AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/generalization/ad_analysis_plot.png)

### 6.3 Uncertainty release gate

The current archive contains contradictory summaries for strict conformal coverage and error–uncertainty correlation. The raw tables and a later scorecard do not describe the same values. We therefore do **not** state that coverage is calibrated across all targets or that error–uncertainty correlation passes a fixed bar. A fresh isolated run under the pinned environment must regenerate the tables and be validated before those claims appear in external material. This is the right scientific response to a contradictory audit trail: preserve the recorded evidence, identify the mismatch, and release only what can be reproduced.

## 7. Offline website and demonstration boundary

The intended dashboard supports a presentation scenario: select a curated polymer or enter a repeat-unit string; view its rendered structure, target prediction, interval, applicability tier, nearest analogue and explanation; then compare valid representations of the same polymer. It is designed to illustrate the discovery workflow, not to turn a single literature example into an external benchmark.

The website must distinguish two artifacts. The competition submission contains transductive components that are not appropriate for arbitrary row-by-row serving. The portable website model is a compact per-row inference route with its own published out-of-fold values, a domain warning and a clear statement of prediction source. The user should never be led to believe that a portable live model exactly recreates every component of a frozen contest pipeline.

For external context, a materials panel should be pre-registered before predictions are viewed, include source conditions, and show all results—including any misses. The proposed literature examples are demonstrations of chemical interpretation and representation invariance, not a new benchmark or a substitute for the contest metric.

## 8. Limitations and next work

The input is a repeat unit, so it lacks important material variables. This is most consequential for Tg, where molecular weight, thermal history, tacticity, crystallinity and morphology can change experimental outcomes. The proposed scientific message is therefore not “the model determines material behaviour from structure alone.” It is: the system identifies a structure-conditioned estimate, explains its model-space drivers, exposes its domain of support, and creates hypotheses for experimental follow-up.

Other limitations are specific and actionable:

| Limitation | Current treatment | Next test |
|---|---|---|
| Cross-model descriptor-ranking agreement is low | report it rather than averaging it away | chemically grouped attribution comparison plus matched feature-removal tests |
| Low-similarity families are harder | show applicability tier and nearest analogue | acquire/curate family-diverse labels and use scaffold-aware selection |
| Strict uncertainty audit conflicts | withhold coverage-pass claim | regenerate all UQ artifacts from the isolated run |
| External literature comparison is not a benchmark | label illustrative cases honestly | pre-register a multi-material panel, include sources and misses |
| Website inference differs from transductive submission path | display model source and compact-model scope | package a reproducible model artifact and test its serving equivalence |

The next experiments should be disciplined rather than broad. Candidate Tg features related to rigidity/free volume should be tested with the same grouped split. Chemically grouped explanation agreement should be measured rather than assumed. A Tabular foundation-model or pretrained representation should be compared under an exactly matched structure-aware protocol, but only after it satisfies the same data-provenance and reproducibility constraints.

## 9. Reproducibility and delivery checklist

The deliverable consists of the submitted notebook and environment, model/prediction artefacts, architecture documentation, evidence tables, an offline demonstration interface, and this report. The release order matters:

1. Finish the user-operated isolated Python-3.11.7 notebook run without writing outside `fixes/isolated_runs/`.
2. Rebuild the qualitative evidence tables from its completed output and resolve the archive discrepancies.
3. Add a prediction-versus-truth parity plot only when its predictions, targets and split are traceable.
4. Freeze the code revision, environment requirements/lockfile, model artifact and submission CSV.
5. Publish a repository tag and add its URL to the short check-in report.

The related draft in `Personal/Research_Paper/paper.md` is a separate, in-preparation Round-2 research document on replication and negative results. It is useful background for future work but is not cited as a published source or used as empirical evidence for this check-in.

**GenAI disclosure.** Generative-AI tools assisted with code scaffolding, documentation and literature triage. Every numerical claim retained here is tied to a named project artifact; generated prose is not experimental evidence.

## References

1. Kim, C. *et al.* Polymer Genome: A data-powered polymer informatics platform. *J. Phys. Chem. C* (2018). https://doi.org/10.1021/acs.jpcc.8b02913  
2. Grinsztajn, L., Oyallon, E. & Varoquaux, G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS* (2022). https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html  
3. Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org  
4. Rogers, D. & Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* (2010). https://doi.org/10.1021/ci100050t  
5. Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. *NeurIPS* (2017). https://arxiv.org/abs/1705.07874  
6. Hooker, S. *et al.* A benchmark for interpretability methods in deep neural networks. *NeurIPS* (2019). https://arxiv.org/abs/1806.10758  
7. Angelopoulos, A. N. & Bates, S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification (2021). https://arxiv.org/abs/2107.07511  
8. OECD. *Guidance Document on the Validation of (Q)SAR Models* (2007). https://www.oecd.org/env/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models-9789264085442-en.htm  
9. Krogh, A. & Vedelsby, J. Neural network ensembles, cross validation, and active learning. *NeurIPS* (1994). https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html  
10. Fox, T. G. & Flory, P. J. Second-order transition temperatures and related properties of polystyrene. *J. Appl. Phys.* (1950). https://doi.org/10.1063/1.1699435

## Appendix A — visual manifest

| Visual | Local path | Status |
|---|---|---|
| Architecture | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/architecture.png` | ready |
| Two-regime EDA | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/eda/novelty_two_regimes.png` | ready |
| Invariance | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/robustness/smiles_invariance_boxplot.png` | ready |
| Generalisation | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/generalization/generalization_ladder_plot.png` | ready |
| Applicability domain | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/generalization/ad_analysis_plot.png` | ready |
| Verified four-panel evidence | `finals/assets/verified_qualitative_evidence.png` | ready |
| Prediction-versus-truth parity | `fixes/isolated_runs/outputs/training/parity_plots.png` | add only after active run validation |

> We, team **Sandman**, have made our submissions wholly based on our own efforts and have not taken help from third parties / members not part of the team. *(Confirm the declaration and team name before submission.)*
