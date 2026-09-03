# Analysis and understanding guide

Read this in two passes: first the architecture map, then the Q&A. The core discipline is: **lead with the evidence, state the mechanism, then name the limitation.** Do not substitute a leaderboard score for a scientific answer.

## 1. The problem in plain language

A polymer repeat unit is a compact description of the motif that is repeated along a chain. A SMILES string is one way to write that motif, but it is not a unique name: the same molecular graph can have multiple valid SMILES spellings. The task asks for seven physical properties from that repeat unit. The scientific difficulty is that the properties do not share the same amount of data or the same mechanism. Tg is tied to chain mobility and free volume; the electronic targets relate to occupied and unoccupied band edges; n and ε have optical and polarisation components.

The reported contest outcomes are a local held-out verification-panel mean R² of **0.907551** and public leaderboard **0.920**. R² answers “how much variance did the prediction explain relative to predicting the mean?” An R² of 1 is perfect; 0 means no better than the target mean; negative means worse. The contest takes the mean of seven separate R² values, so every target matters equally even when their datasets are very different sizes.

## 2. Architecture, component by component

### 2.1 Canonicalise and split by structure

RDKit turns a SMILES into a molecular graph. Canonicalisation lets us identify alternative spellings of the same graph. Splitting by canonical structure prevents a subtle optimistic validation error: a model must not validate on a rewritten version of a polymer it already saw in training.

### 2.2 Build several representations

- **Descriptors:** interpretable scalar summaries such as size, rings, heteroatom content and surface-related quantities. They help trees capture broad physicochemical trends.
- **Morgan count fingerprints:** circular neighbourhood counts around atoms. They capture reusable local chemistry and support Tanimoto similarity.
- **Polymer atomic triples:** structured local motifs motivated by Polymer Genome.
- **Character n-grams:** short string fragments. They can capture useful sequence patterns cheaply, but are not inherently SMILES-invariant; use them as a controlled residual rather than the sole representation.
- **Label-free polymer-corpus SVD:** compresses permitted unlabelled polymer strings into low-dimensional directions without looking at target labels.

### 2.3 Train per-target leaves

Do not force the same learner on every property. For data-rich targets, boosted trees and ExtraTrees can learn non-linear descriptor/fingerprint interactions. For sparse targets, kernel and probabilistic members may interpolate local structure more cautiously. A GINE message-passing model learns from atom and bond messages; it is blended because its errors can differ from those of classical models, not because “neural” is automatically better.

### 2.4 Apply guarded physical relationships

The physics stage uses relationships only when a partner quantity exists in the allowed training information. Egc relates the two band edges Ei and Eea. Egb is linked to Egc but requires an empirical residual because bulk packing/interchain effects are not identical to an isolated-chain picture. For ε, n² is the electronic portion and a non-negative ionic contribution accounts for the remainder. A guard compares routes and prevents a physics projection from worsening the chosen validated prediction.

### 2.5 Combine and calibrate without leakage

Each base model produces out-of-fold predictions for training rows it did not fit. The combiner learns non-negative weights from those predictions. This is important: fitting the stacker on in-sample base predictions would exaggerate performance. Calibration is similarly fitted on training-fold residuals; it corrects systematic compression but is not allowed to peek at evaluation labels.

## 3. What worked, what did not, and why

| Topic | Evidence-led conclusion | What to say in Q&A |
|---|---|---|
| Per-target modelling | Necessary because target sizes and physical routes differ. | “The metric weights targets equally, so pooled training creates a mismatch.” |
| Classical ensembles | Strong small-data backbone. | “We used them because grouped controls justified them, not because deep learning was excluded.” |
| GINE | Complementary component when it diversifies errors. | “It is a second opinion, not a universal replacement.” |
| Band-edge identity | Retained when partner labels are allowed and route is validated. | “Physics constrained the model; it did not read the requested evaluation label.” |
| Learned residual on identity | Rejected in the scarce-data setting. | “The extra flexibility did not earn its place.” |
| Broad self-supervision | Not assumed useful at this scale. | “A control is stronger than a fashionable architecture.” |
| Flexible calibration | Can overfit a small calibration split. | “We prefer conservative, fold-fitted correction.” |

## 4. The qualitative evidence, explained

### Explainability

SHAP assigns each input feature a contribution to a particular prediction relative to a baseline. It is useful for showing which descriptors or fragments drive a tree-based prediction, but it is not automatically causal. The stronger test is **fidelity**: remove what the explanation says is important and see whether prediction quality falls much more than under a random removal control. The original Tg proxy gives a top-10% loss of 0.851 versus 0.043 random; the recorded all-target audit gives a top-5% loss of 0.810487 versus 0.022373 random. This supports “load-bearing under this intervention,” not “the feature causes the real-world property.”

Linear probes are simple linear models trained to decode a known chemical attribute from an internal representation. If a probe predicts a chemical concept, the representation retains accessible information about it. It still does not prove the model uses that concept causally; pair it with fidelity and counterfactual tests.

### Invariance

Prediction invariance asks whether different valid SMILES strings for the same graph receive the same result. Attribution invariance asks whether the stated reasons also stay the same. Graph-derived features should be invariant by construction after parsing; character n-grams should not be assumed invariant. The project reports them separately. In the recorded graph audit, the tested violation count is zero and mean attribution cosine is 0.979737. This is evidence for the stated rewrite protocol, not a blanket claim over all chemical transformations.

### Robustness and generalisation

The generalisation ladder places test points into increasing structural novelty bands. If accuracy falls with lower nearest-neighbour similarity, that is not necessarily a bug; it is an applicability-domain result. The recorded canonical-group and scaffold checks stay positive for all targets (mean R² 0.823752 and 0.658082). The correct product behaviour is to tell the user which tier their polymer occupies, show its closest training analogue, and widen or qualify uncertainty accordingly.

Conformal prediction creates intervals with a coverage guarantee under exchangeability assumptions. The recorded audit is within ±3 percentage points of nominal coverage for all seven targets, but error–uncertainty correlation clears its stated threshold for five of seven targets. That distinction matters: coverage is a calibration check; interval width is not automatically a perfect ranker of future error. Both results require the pinned rerun refresh before final promotion.

## 5. Presentation-ready Q&A

### Data, target science and EDA

**Q1. Why separate models instead of one multitask network?**  
**Answer:** “The seven properties have very different label counts and mechanisms, while the metric weights each target equally. A pooled loss would optimise the largest/most variable target, so we share representation but choose target-specific leaves and only use validated cross-property routes.”

**Q2. Is the DFT structure overlap leakage?**  
**Answer:** “No. The target label being predicted is never available. A different training measurement for the same polymer can only trigger a guarded physical relation. We also use canonical-structure grouped validation so rewrites cannot cross folds.”

**Q3. Why is Tg different?**  
**Answer:** “Tg is experimentally controlled by chain mobility, packing and free volume, and it has less cross-property support. It is primarily a structure-to-property extrapolation task.”

**Q4. Why is pooled R² inappropriate?**  
**Answer:** “The contest metric is the mean of seven target-wise R² values. Pooling silently weights targets by row count and variance, which is not what the contest or scientific question asks.”

**Q5. Why use a repeat unit instead of a full polymer chain?**  
**Answer:** “The supplied representation is the repeat unit. It encodes local chemistry and connectivity, but it does not fully encode molecular weight, morphology or processing history; that is a real limitation, especially for Tg.”

**Q6. What did EDA change?**  
**Answer:** “It revealed two data regimes and showed why target-specific validation and guarded physics are necessary. It changed the architecture before hyperparameter tuning began.”

### Architecture and modelling

**Q7. What exactly does GINE add?**  
**Answer:** “It passes messages over atoms and bond features, so it sees connectivity differently from descriptors and fingerprints. We keep it only where grouped validation shows useful error diversity.”

**Q8. Why trees in a modern ML project?**  
**Answer:** “At small tabular sample sizes, trees are strong, stable baselines. We tested rather than assumed the model family; deep learning is an additional hypothesis, not a status symbol.”

**Q9. Why several representations?**  
**Answer:** “Descriptors capture bulk chemistry, fingerprints capture local motifs, graphs capture connectivity and character features capture residual sequence signal. The ensemble is useful only if these views make non-identical errors.”

**Q10. Why non-negative blend weights?**  
**Answer:** “They make the stacker less able to fit validation noise by cancelling large correlated predictions. We trade a little flexibility for a safer out-of-fold combination.”

**Q11. How do you avoid stacking leakage?**  
**Answer:** “The meta-model sees only out-of-fold base predictions, never a prediction from a base learner trained on that same row.”

**Q12. What is calibration?**  
**Answer:** “Ensembles can pull extremes toward the mean. Calibration corrects systematic predicted-versus-observed slope or spread, but it must be fitted inside validation folds or it becomes another source of leakage.”

### Physics and domain

**Q13. Why is Egc related to Ei and Eea?**  
**Answer:** “They describe the separation of electronic band edges. The relation is physically motivated, then checked on co-measured data; it is not merely a fitted correlation used without a guard.”

**Q14. Why not force the identity for every row?**  
**Answer:** “A physical relation needs valid inputs. We only apply it when partner information is legitimately available and the validated route supports it.”

**Q15. Why can ε involve n²?**  
**Answer:** “The refractive index reflects the high-frequency electronic response, while dielectric response can include additional ionic polarisation. The decomposition gives a physically interpretable constraint, not an exact universal law for every experimental condition.”

**Q16. What does Flory–Fox demonstrate?**  
**Answer:** “It is a qualitative polymer-physics relation connecting Tg and chain length through an inverse-size term. A homologous-series stress test checks whether predicted trends behave consistently with that expectation.”

### Explainability, invariance and uncertainty

**Q17. Why should we believe SHAP?**  
**Answer:** “We do not ask you to trust the plot. We intervene: masking top-ranked features harms performance far more than random masking. That is the fidelity evidence.”

**Q18. Does a high SHAP value prove causality?**  
**Answer:** “No. It shows contribution within the fitted model. Causality needs experimental or carefully designed counterfactual evidence. We use the correct narrower claim.”

**Q19. Why test the reasons as well as predictions?**  
**Answer:** “A stable scalar prediction with unstable explanations is not a stable scientific story. Attribution invariance tests that the stated drivers do not change under equivalent spelling.”

**Q20. Your character model is not invariant. Is that a contradiction?**  
**Answer:** “No; it is a documented trade-off. Graph features supply the invariant backbone. The character arm is small, measured separately, and never described as invariant by construction.”

**Q21. What is an applicability domain?**  
**Answer:** “The region of chemical space resembling the data on which the model was assessed. We operationalise it through nearest-neighbour similarity tiers and show the user the tier.”

**Q22. Do the intervals pass calibration?**  
**Answer:** “In the recorded audit, all seven targets are within the predeclared ±3-percentage-point coverage tolerance. We still show the actual tier and interval because coverage is protocol-specific; the pinned rerun must refresh that result before final release.”

**Q23. What does imperfect error–uncertainty correlation mean?**  
**Answer:** “Five of seven targets clear the stated correlation bar. For the other two, interval width alone should not rank future errors; the nearest-analogue/applicability signal remains useful, but uncertainty needs further work.”

### Results, reliability and future work

**Q24. What are the only performance numbers we should quote?**  
**Answer:** “0.907551 mean R² on the local held-out verification panel and 0.920 on the public leaderboard for the submitted notebook.”

**Q25. Why not claim the public score as final generalisation?**  
**Answer:** “It is a useful external outcome, but one partition can be easier or harder. We use structured validation and the local held-out panel to assess mechanisms and use the public result as an outcome, not a tuning target.”

**Q26. What is the strongest limitation?**  
**Answer:** “Raw cross-model feature-rank agreement is only 0.472223, and the repeat unit omits molecular-weight, processing and morphology variables. We therefore present fidelity and graph-invariance as the primary explanation evidence, and treat material comparisons as scoped examples rather than universal proof.”

**Q26a. Does the website prove external generalisation?**  
**Answer:** “No single card can do that. PS and PMMA are disclosed literature context, while PIB is one membership-audited external anchor. The next evidence step is a pre-registered five-material panel that reports every result, including misses.”

**Q27. What would you do with more data?**  
**Answer:** “Add experimentally meaningful metadata—molecular-weight distribution, processing, morphology and measurement conditions—then evaluate pretrained polymer representations under matched controls. More data only helps if it resolves missing physical variables.”

**Q28. What would you do with another week?**  
**Answer:** “First rerun the full evidence/release gate in the pinned environment; then improve small-target cross-conformal calibration and compare shallow, diversity-selected stacks against deeper assemblies.”

**Q29. Was AI used?**  
**Answer:** “Yes, for scaffolding, refactoring, documentation and literature triage. The defensible standard is that every reported claim traces to an executed project artefact and every design decision has an evidence record.”

**Q30. What is the field-level contribution?**  
**Answer:** “A reusable evaluation pattern: do not report only an average R². Report data regime, structured split, applicability, explanation fidelity, encoding invariance, and the failures that set the scope of deployment.”

## 6. Papers to understand first

1. Grinsztajn et al. (2022): why classical trees remain strong on ordinary tabular data.
2. Kim et al. (2018), *Polymer Genome*: descriptor-led polymer informatics.
3. Lundberg & Lee (2017) and Hooker et al. (2019): attributions versus intervention/fidelity.
4. Bjerrum (2017): randomised SMILES as a testing/augmentation device.
5. Angelopoulos & Bates (2021): what conformal intervals guarantee and what finite samples obscure.
6. Fox & Flory (1950): the Tg relation used for a qualitative polymer sanity check.

Full citations and links are in `finals/Report.md`; the project bibliography is `Personal/Research/INDEX.md`.
