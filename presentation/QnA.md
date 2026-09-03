# Judge Q&A — concise, evidence-bound answers

## Modelling strategy

### Why not one model for all seven properties?

The targets have different label counts, scales and chemical drivers. We share a representation but use target-specific learned lanes so model capacity and blending are selected per property. This is a practical multi-task compromise, not a claim that all properties share one physical mechanism.

### What is the final model?

It is a target-specific assembly over a common feature bank: RDKit/global descriptors, circular fingerprints, polymer-aware descriptors and a limited string residual; compact tabular learners are the backbone. A graph lane is blended only using out-of-fold predictions. Two physical routes are retained only where their validation supported them.

### Is the graph model pretrained?

No. The graph models are trained on the official labelled training data, target by target, with multiple seeds and structure-grouped folds. It is not a foundation model or external pretrained component.

### Why retain trees for a scientific task?

The data regime is tabular and heterogeneous, with several small targets. On such data, strong gradient-boosted/tree baselines remain difficult to beat reliably; see Grinsztajn et al. [R1]. We validated this choice with grouped out-of-fold results rather than assuming it.

### What does the “ML-only alternative” mean?

It is a compact, independently runnable tabular pipeline, useful as a reproducible baseline and deployment fallback. Its recorded grouped-CV mean R² is 0.816344, so it is not presented as the final selected submission.

## Data and validation

### What leakage did you guard against?

We canonicalize repeat-unit representations and use canonical-SMILES GroupKFold for internal selection. This prevents a shared canonical structure from landing in both the train and validation partitions of an internal fold. It is a stronger check than row-wise random KFold for duplicated representations.

### Is GroupKFold a proof of out-of-domain generalization?

No. It measures generalization to withheld canonical structure groups within the supplied data. We reserve “out-of-domain” for a separately defined external chemical domain and label applicability as a diagnostic, not a guarantee.

### What score should we interpret?

The historical final selected submission has mean R² 0.907551 on the local held-out verification panel and 0.920 on the public leaderboard. It is a mean of target-wise R² values, so a large target cannot dominate through pooled variance. Use the release manifest/score record to bind a final CSV before any new resubmission.

### Why not report a pooled R² as the headline?

Pooled R² can be dominated by the highest-variance or most populous target. The contest evaluates properties equally, so the arithmetic mean of seven target-wise R² values is the appropriate headline metric.

## Feature and physics choices

### Why these features?

Descriptors provide composition, ring and functional-group summaries; Morgan fingerprints retain local environments; polymer-aware descriptors express repeat-unit context; the limited string channel can add notation-level residual information. The components preserve different information, rather than being multiple names for the same representation. Morgan fingerprints are an established cheminformatics representation [R2]; Polymer Genome motivates polymer-focused descriptor workflows [R3].

### Why include SMILES text at all if SMILES is not chemistry?

It is not the primary chemistry representation. The channel is deliberately limited and is paired with canonical/grouped validation. Molecular descriptors, fingerprints and graphs carry the chemically meaningful representation.

### Why is Egc = Ei − Eea used directly?

The relation is a useful physical identity to test. On the small co-measured subset, the learned residual was less stable than the direct route in the appropriate validation. We therefore retained the direct relation rather than adding a correction merely because one is possible.

### Why does Egb receive an ExtraTrees residual?

Unlike the Egc correction, its residual route improved the relevant validation result. The choice is empirical and target-specific; it is not a universal law of polymer band gaps.

## Qualitative claims

### Is SHAP proving which chemical group causes a property?

No. SHAP describes feature attribution for a model. Here it is used on a proxy tree ensemble, with a rank-removal comparison as a fidelity diagnostic. It supports whether the proxy’s ranked features matter to that proxy’s predictions; it does not establish chemical causality. See TreeSHAP [R4] and the ROAR evaluation framing [R5].

### What exactly has been proven about invariance?

One bounded result: a primitive-repeat normalizer for a declared simple, unbranched PEO grammar maps tested translated, monomer, dimer and trimer forms to one primitive repeat before featurization. We do not claim that all polymer-SMILES grammars or arbitrary cut points are solved.

### Why does monomer/dimer/trimer invariance matter?

They can describe the same repeating material at different string lengths. If a representation changes only because a unit is repeated, a prediction should not change for that syntactic reason. The primitive-repeat step makes this explicit in the tested grammar.

### Is an oligomer just a repeated monomer?

In the way the demo uses the word, yes: an oligomer is a short chain made from a small number of repeat units, and dimer/trimer are the two-repeat and three-repeat cases. The code normalizes supported repeat spellings before prediction. It does not collapse separately synthesized finite oligomers with real end groups, because those can have different measured material properties.

### What exactly happens in the PEO invariance demo?

`*CCO*`, `*OCC*`, `*CCOCCO*` and `*CCOCCOCCO*` are parsed by the strict terminal-star linear grammar. The normalizer finds the smallest repeated period, rotates it to the canonical spelling, and sends `*CCO*` to the portable predictor. That is why all seven target predictions have zero range in the tested panel.

### Do randomized SMILES prove polymer invariance?

No. They test serialization robustness, not the complete polymer-repeat equivalence problem. Random SMILES augmentation is a known molecular representation technique [R6], but polymer grammar needs separate validation.

### Is the uncertainty interval calibrated at 90%?

No such coverage claim is made. The dashboard calls it an experimental uncertainty interval. Reliable calibration on scarce targets remains a limitation and an active validation task.

### What does the applicability tier mean?

It is a structural-similarity diagnostic relative to the training set. It helps the user distinguish familiar and less familiar inputs; it does not certify accuracy or novelty.

## Demo and delivery

### Is the 3D structure in the dashboard a simulated material morphology?

No. It is an illustrative molecular conformer/structure rendering to help inspect the input. It does not claim to represent amorphous morphology, packing or a bulk polymer microstructure.

### Can a judge enter an arbitrary polymer?

Yes, the interface can accept a parseable input. The result must be interpreted together with the applicability tier and the stated modelling scope. We do not present arbitrary unseen chemistry as experimentally validated.

### What is actually delivered?

The codebase, reproducible environment/run instructions, trained/artifact documentation, selected prediction CSV, notebook evidence, claim registry and interactive dashboard. The graph contribution is documented as a reproducible training stage rather than a silently embedded pretrained artifact.

### Where should reviewers start in the repository?

Start with `SUBMISSION/README.md`, then open the completed notebook, `SUBMISSION/submission.csv`, `ARCHITECTURE.md`, `CLAIM_REGISTRY.md` and the offline dashboard in `Website/`. The root README has a file map for notebook, CSV, weights, GNN artifact, evidence, run notes and demo.

## Research direction

### What would you do with more time?

Benchmark primitive-repeat normalization across several polymer grammars and measured properties; build a literature-curated external panel; and compare pretrained polymer representations under the same no-leakage validation. Each would be reported separately from the current selected submission.

### How does this advance polymer discovery?

It creates a disciplined screening loop: propose a repeat unit, make a prediction, inspect a scoped explanation and applicability diagnostic, then prioritize promising candidates for measurement. The model reduces search cost; it does not replace synthesis or experiments.

## References used in answers

- [R1] L. Grinsztajn, E. Oyallon, G. Varoquaux. *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS, 2022.
- [R2] D. Rogers, M. Hahn. *Extended-connectivity fingerprints.* J. Chem. Inf. Model., 2010. DOI: 10.1021/ci100050t.
- [R3] K. A. Mannodi-Kanakkithodi et al. *Polymer Genome: A data-powered polymer informatics platform for property predictions.* J. Phys. Chem. C, 2018. DOI: 10.1021/acs.jpcc.8b02913.
- [R4] S. M. Lundberg, S.-I. Lee. *A unified approach to interpreting model predictions.* NeurIPS, 2017; TreeSHAP implementation analysis: Nature Machine Intelligence, 2020. DOI: 10.1038/s42256-019-0138-9.
- [R5] S. Hooker et al. *A benchmark for interpretability methods in deep neural networks.* NeurIPS, 2019 / arXiv:1806.10758.
- [R6] J. H. Jensen. *A molecular transformer for property prediction using randomized SMILES.* arXiv:1703.07076, 2017.

## Architecture deep-dive questions

### Explain the pipeline in one minute without diagrams.

Canonicalize the repeat unit, create a shared descriptor/fingerprint/polymer-feature bank, train target-specific learners, add only validated physical or cross-property routes, then fit target-wise non-negative blend weights from out-of-fold predictions and apply small diagnosis-led calibration. The GNN is a complementary out-of-fold arm, not the entire model.

### What prevents cross-property leakage?

Every partner prediction is cross-fitted: a validation row receives a partner estimate from a model that did not train on that row. The rejected non-cross-fitted route looked 0.935 OOF and transferred 0.907, which is why cross-fitting is structural rather than optional.

### Why non-negative least squares instead of a flexible meta-model?

NNLS combines out-of-fold component predictions while forbidding negative cancellation weights that can fit fold noise. It is a deliberate bias–variance and auditability trade-off for scarce-target folds.

### What exactly does the GNN contribute?

It reads atom/bond connectivity while the tabular stack reads descriptors, fingerprints and polymer features. Its purpose is not to be the best individual model; it provides decorrelated connectivity errors for the OOF blend. It is trained on official labelled data, with grouped folds and three seeds per target, and is not pretrained.

### Which physical choices were rejected and which stayed?

Egc = Ei − Eea stayed directly because it reached R² 0.9716 and MAE 0.0716 eV on 59 co-measured polymers while its learned correction gave LOO R² −0.82. The Egb ExtraTrees residual stayed because it improved its validation route. The epsilon decomposition stayed because its audited ionic remainder was non-negative and better conditioned.

### Where is the current completed notebook with outputs?

Use `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/SUBMISSION/Sandman_Polymer_Property_Prediction_Final.ipynb`. It has 76 code cells with retained outputs and sits beside the generated CSV, curated evidence, pinned requirements, portable model and GNN aggregate artifact.

### What are the two headline qualitative limitations?

Cross-model explanation-rank agreement is ρ = 0.472, below the predeclared 0.60 bar. The strict PEO repeat test is a declared grammar-level construction, not a general PSMILES parser. We show both limitations rather than hide them behind a qualitative scorecard.
