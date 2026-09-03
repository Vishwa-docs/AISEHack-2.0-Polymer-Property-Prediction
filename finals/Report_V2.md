# Mid-Point Check-in: AI for Science & Engineering

**Project title:** Representation-aware, physics-guided polymer property prediction from repeat-unit SMILES  
**Team name:** Sandman (Vishwa Kumaresh)  
**Primary domain:** Materials Science and Polymer Informatics  
**Evaluation metric:** Unweighted mean of seven target-wise R² values

We predict glass-transition temperature (Tg), chain and bulk bandgaps (Egc, Egb), ionisation energy (Ei), electron affinity (Eea), refractive index (n), and dielectric constant (ε) from polymer repeat-unit SMILES. The submitted notebook records **0.907551 mean R²** on the local held-out verification panel and **0.920** on the public leaderboard.

## 1. Proposed Strategy and Technical Novelty

### A polymer-aware answer to two different problems

Our novelty is not one more model. It is a property-aware pipeline that recognises that this dataset contains two very different prediction settings. Tg usually requires learning from polymer structure to a new measured property. The smaller electronic and optical targets often have the same polymer measured for a *different* property, so they can benefit from a carefully checked cross-property signal. Treating all seven targets as one pooled problem would miss this distinction and over-focus on Tg.

The pipeline therefore combines three ideas: (1) polymer-specific and molecular representations, (2) one learned prediction lane for every property, and (3) simple physical relations used only when they improve validation. This lets us use chemistry where it helps while keeping a learned model available for every target. The approach is informed by polymer-informatics work on multi-property learning, polymer representations and large label-free polymer corpora [1–3].

### Architecture and why each part is present

![][image1]

**Figure 1. Pipeline architecture.** Each target has a direct learned prediction lane. The physical routes are additional, guarded inputs—not replacements for the learned models. Only the final held-out and public leaderboard results are shown in the figure.

| Stage | What we use | Why it matters |
| :--- | :--- | :--- |
| Shared representation | RDKit descriptors, Morgan fingerprints, Polymer Genome atomic triples, a label-free PI1M SVD prior, similarity kernels and a compact GINE graph signal | captures size, rigidity, polarity, local chemistry, polymer environments and graph structure [2–5] |
| Direct learned lanes | boosting/ExtraTrees for larger targets; kernel, Gaussian-process and MLP lanes for scarce targets | every property can be predicted directly from its structure, including Egc, Egb, Ei, Eea, n and ε |
| Guarded chemistry routes | Egc from Ei−Eea; Egb from Egc; ε from n² plus a polar remainder | adds useful information only when the companion measurement is available in the contest data protocol |
| Assembly | out-of-fold NNLS blend and small target-specific calibration | combines models that make different errors without giving any one lane unchecked control |
| Decision support | nearest analogue, explanation and applicability tier | shows users when a polymer is close to, or far from, known chemistry |

The learned lanes are essential. For example, Egc has a boosting plus PI1M-SVD lane; Egb has an ExtraTrees lane; Ei combines MLP, Gaussian-process and Tanimoto-kernel learners; Eea has a learned small-data lane; and n and ε use classical ensembles. The GINE graph encoder is an additional learned structural signal. A physical relation is only an extra route: if it is unavailable or does not help, the direct learned lane remains in use.

### Feature choices and tested chemistry

The feature set is deliberately complementary. RDKit descriptors summarise size, ring content, polarity and flexibility. Morgan fingerprints capture local functional groups. Polymer Genome atomic triples describe local environments in a polymer-specific form [2,4,5]. Removing RDKit descriptors reduced grouped-CV proxy performance for Egb from 0.856 to 0.830 and for Ei from 0.794 to 0.756. Removing Morgan counts reduced n from 0.787 to 0.767. Atomic triples improved targeted Egb from 0.917 to 0.926 and n from 0.844 to 0.852. String n-grams are used only as a small supporting feature, because different valid SMILES strings can describe the same polymer.

The physics routes are also tested, not assumed. The Egc = Ei−Eea relationship is already very accurate on the co-measured subset, so we keep the simple relation and reject a more flexible correction that performed poorly in leave-one-out testing. For Egb, a learned ExtraTrees correction improved the Egc-based route, so it was retained. For ε, we predict the polar/ionic contribution in ε = n² + ionic and then reconstruct ε. This separates optical and polar contributions in a way that is easier for the model to learn. We also tested published gap–refractive-index relations; they did not help this dataset, so they were not used [6,7].

### Qualitative validation: same polymer, stable answer

![][image2]

**Figure 2. Qualitative evidence beyond the score.** The upper panels test representation invariance and stable explanations under alternative valid SMILES spellings. The lower panels show the change in performance for harder structural splits and the rise in Tg error as a query moves away from known chemistry.

**Invariance.** A SMILES string is one way to write a polymer, not the polymer itself. We generated valid alternative spellings of the same repeat-unit graph. The graph-based path had zero 1σ prediction violations across all seven targets; its largest spread was 0.230% of the target training scale. The explanation similarity was 0.947–0.996 (mean 0.980). This supports the use of graph and canonical features as the main representation [8].

**Explainability.** TreeSHAP identifies which features matter most to a prediction [9]. We tested this with a feature-removal check: removing the top 10% SHAP-ranked features from the Tg proxy lowered R² by 0.851, versus 0.043 for a same-size random removal. The reported explanations therefore identify features the model genuinely relies on [10].

**Generalisability and robustness.** The mean score is 0.824 when related structures are kept together and 0.660 when entire scaffolds are held out; all seven targets remain positive in both tests. The decline is useful information: a new scaffold is harder than a familiar polymer family. Tg error also rises as nearest-neighbour similarity falls, so the demo shows an applicability tier and nearest analogue rather than treating every prediction equally [11,12].

## 2. Preliminary Salient Results

### EDA finding: one leaderboard contains two data regimes

![][image3]

**Figure 3. EDA: overlap of evaluation polymers with the training data.** Almost no exact polymer–property pairs are repeated. However, 88–99% of evaluation polymers for the small DFT targets occur in training with another measured property, while the Tg regime has much less such support. This finding motivates the target-specific lanes and guarded cross-property routes.

The data analysis changed the architecture before model tuning began. Tg contributes **55.9%** of rows and **99.986%** of pooled variance, but it receives only one-seventh of the score—exactly the same weight as Ei, which contributes only **3.0%** of rows. A single pooled loss would therefore learn mainly Tg, even though the evaluation values every property equally. We use grouped validation because train and test contain shared canonical structures; this prevents a polymer family from appearing in both sides of an internal split.

For discovery, the result is a useful workflow rather than only a number. A high-Tg candidate can be examined through ring content, rigidity and flexibility. An electronic candidate can be checked for agreement between direct and band-edge routes. An optical candidate can be inspected through the n² and polar/ionic parts of ε. These are practical prompts for what to synthesise or measure next.

## 3. Technical Challenges and Pivots

| Challenge | What we learned | Pivot |
| :--- | :--- | :--- |
| One representation does not suit every target | large Tg and scarce electronic/optical tasks behave differently | use target-specific learned lanes over a shared representation |
| A more flexible formula can be worse | the learned Egc residual failed leave-one-out testing | retain the simpler guarded relation |
| A reliable interval needs one verified audit trail | archived uncertainty summaries disagree | show applicability distance now; release calibrated intervals only after the isolated rerun |

Repeat-unit SMILES also omit molecular weight, tacticity, morphology, processing history and measurement conditions. We address this openly through an applicability tier and by positioning the tool for screening and experiment planning. Tg group-contribution and free-volume research provides a clear next feature direction [13,14].

## 4. Final Sprint Roadmap and Evaluation Alignment

| Evaluation expectation | What is already shown | Next release step |
| :--- | :--- | :--- |
| Feature strategy and architecture | ablations, polymer-specific representations, direct lanes and guarded routes | freeze choices with their source artifacts |
| Polymer invariance | equivalent-SMILES prediction and explanation tests | show two spellings of one polymer in the dashboard |
| Explainability | TreeSHAP plus feature-removal fidelity | group correlated descriptors into chemical concepts |
| Generalisability and robustness | grouped/scaffold tests and similarity-based applicability tier | add a pre-registered external polymer panel with literature conditions |
| Transparency | failed residual and pending uncertainty item are stated | regenerate the late uncertainty tables before release |

The remaining work is focused: complete the Python-3.11 isolated run, validate the uncertainty artifacts, and add a parity plot only after its source and split are verified. The final package will include the pipeline, environment, evidence tables and offline dashboard. Together, these support the hackathon goal: screen a new polymer, understand the structural signals behind the prediction, identify whether it is inside the model’s known chemical space, and decide what experiment should follow.

## References

1. Kuenneth, C. *et al.* Polymer informatics with multi-task learning. *Patterns* (2021). https://doi.org/10.1016/j.patter.2021.100238  
2. Kim, C. *et al.* Polymer Genome: a data-powered polymer informatics platform. *Journal of Physical Chemistry C* (2018). https://doi.org/10.1021/acs.jpcc.8b02913  
3. Ma, R. & Luo, Y. PI1M: a benchmark database for polymer informatics. *Journal of Chemical Information and Modeling* (2020). https://doi.org/10.1021/acs.jcim.0c00726  
4. Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org  
5. Rogers, D. & Hahn, M. Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling* (2010). https://doi.org/10.1021/ci100050t  
6. Moss, T. S. Relation between the refractive index and the energy gap of semiconductors. *physica status solidi (b)* (1985). https://doi.org/10.1002/pssb.2221310202  
7. Ravindra, N. M. & Srivastava, V. K. On the Penn model of the dielectric constant. *Infrared Physics* (1979). https://doi.org/10.1016/0020-0891(79)90013-1  
8. Bjerrum, E. J. SMILES enumeration as data augmentation for neural network modelling of molecules (2017). https://arxiv.org/abs/1703.07076  
9. Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. *NeurIPS* (2017). https://arxiv.org/abs/1705.07874  
10. Hooker, S. *et al.* A benchmark for interpretability methods in deep neural networks. *NeurIPS* (2019). https://arxiv.org/abs/1806.10758  
11. Bemis, G. W. & Murcko, M. A. The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry* (1996). https://doi.org/10.1021/jm9602928  
12. OECD. *Guidance Document on the Validation of (Q)SAR Models* (2007). https://www.oecd.org/env/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models-9789264085442-en.htm  
13. Bicerano, J. *Prediction of Polymer Properties.* Marcel Dekker (2002).  
14. Williams, M. L., Landel, R. F. & Ferry, J. D. The temperature dependence of relaxation mechanisms in amorphous polymers. *Journal of the American Chemical Society* (1955). https://doi.org/10.1021/ja01619a008

## Appendix A — experiment register

| Theme | Tested result | Decision |
| :--- | :--- | :--- |
| Feature families | RDKit/Morgan removal hurts selected grouped-CV proxy lanes; atomic triples help Egb and n | retain complementary representations |
| Physics routes | Egc residual fails leave-one-out; Egb correction improves; ε decomposition is better conditioned | retain only supported guarded routes |
| Invariance | graph-path 1σ violations 0; attribution cosine 0.947–0.996 | make graph/canonical features the invariance anchor |
| Explainability | top-10% SHAP mask loss 0.851 versus 0.043 random | retain intervention-tested explanations |
| Uncertainty | archived audit conflict | release-gated pending isolated rerun |

## Appendix B — reproducibility checklist

| Item | Location / condition |
| :--- | :--- |
| Submitted pipeline and architecture | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/` |
| Isolated reproduction | `fixes/isolated_runs/Sandman_Polymer_Property_Prediction_2_906.ipynb` (Python 3.11.7) |
| Claim audit | `finals/REPORT_CLAIM_EVIDENCE.md` |
| Public code link | **[Insert final GitHub release URL/tag]** |
