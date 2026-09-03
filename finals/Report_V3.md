# **Mid-Point Check-in: Team Sandman (Polymer Property Prediction)**

**Project Title:** Physics-guided, evidence based polymer property prediction from repeat-unit SMILES

**Team Name:** Sandman (Vishwa Kumaresh)

**Primary Domain:** (e.g., Fluid Dynamics, Genomics, Materials Science, Structural Engineering)

This report reports a seven-property pipeline for **glass-transition temperature (Tg)**, **chain and bulk bandgaps (Egc, Egb)**, **ionisation energy (Ei)**, **electron affinity (Eea)**, **refractive index (n)**, **and dielectric constant (ε)**. The submitted notebook records a **0.907551** mean R² on the local held-out verification panel and **0.920** on the public leaderboard.

## **1\. Proposed Strategy and Technical Novelty**

### **Novelty : A polymer-aware answer to two different problems**

We propose a property-aware pipeline that recognises that this dataset contains two very different prediction settings. Tg usually requires learning from polymer structure to a new measured property. The smaller electronic and optical targets often have the same polymer measured for a *different* property, so they can benefit from a carefully checked cross-property signal. Treating all seven targets as one pooled problem would miss this distinction and over-focus on Tg.

The pipeline therefore combines polymer-specific and molecular representations, one learned prediction lane for every property, and physical relations used only when they improve validation. This lets us use chemistry where it helps while keeping a learned model available for every target \[1–3\].

### **Methodology, Architecture and Reasoning**

![][image1]

**Figure 1\. Pipeline architecture.** Each target has a direct learned prediction lane. The physical routes are additional, guarded inputs.

| Stages (In Order) | What we use | Why it matters |
| :---- | :---- | :---- |
| Shared representation | RDKit descriptors, Morgan fingerprints, Polymer Genome atomic triples, a label-free PI1M SVD prior, similarity kernels and a compact GINE graph signal | captures size, rigidity, polarity, local chemistry, polymer environments and graph structure \[2–5\] |
| Direct learned lanes | boosting/ExtraTrees for larger targets; kernel, Gaussian-process and MLP lanes for scarce targets | every property can be predicted directly from its structure, including Egc, Egb, Ei, Eea, n and ε |
| Guarded chemistry routes | Egc from Ei−Eea; Egb from Egc; ε from n² plus a polar remainder | adds useful information only when the companion measurement is available in the contest data protocol |
| Assembly | out-of-fold NNLS blend and small target-specific calibration | combines models that make different errors without giving any one lane unchecked control |
| Decision support | nearest analogue, explanation and applicability tier | shows users when a polymer is close to, or far from, known chemistry |

 

Every property retains a direct learned lane: Egc uses boosting plus PI1M-SVD; Egb uses ExtraTrees; Ei combines MLP, Gaussian-process and Tanimoto-kernel learners; Eea has a learned small-data lane; and n and ε use classical ensembles. The GINE graph encoder adds a learned structural signal. Physical relations are extra routes; the direct lane remains available.

### **Featurization Choices**

The feature set is deliberately complementary: RDKit descriptors summarise size, ring content, polarity and flexibility; Morgan fingerprints capture local functional groups; and Polymer Genome atomic triples describe polymer-specific local environments \[2,4,5\].

Removing RDKit descriptors reduced grouped-CV proxy performance for Egb from 0.856 to 0.830 and for Ei from 0.794 to 0.756. Removing Morgan counts reduced n from 0.787 to 0.767. Atomic triples improved targeted Egb from 0.917 to 0.926 and n from 0.844 to 0.852. String n-grams are used only as a small supporting feature, because different valid SMILES strings can describe the same polymer.

The physics routes are also tested. The Egc \= Ei−Eea relationship is already very accurate on the co-measured subset, so we keep the simple relation and reject a more flexible correction that performed poorly in leave-one-out testing. For Egb, a learned ExtraTrees correction improved the Egc-based route, so it was retained. For ε, we predict the polar/ionic contribution in ε \= n² \+ ionic and then reconstruct ε. This separates optical and polar contributions in a way that is easier for the model to learn. We also tested published gap–refractive-index relations; they did not help this dataset, so they were not used \[6,7\].

### **Qualitative Validation : Invariance, Explainability, Generalizability, Robustness**

![][image2]  
**Figure 2\. Qualitative evidence.** The upper panels test representation invariance and stable explanations under alternative valid SMILES spellings. The lower panels show the change in performance for harder structural splits and the rise in Tg error as a query moves away from known chemistry.

1. **Invariance.** A SMILES string is one way to write a polymer, not the polymer itself. We **generated valid alternative spellings** of the same repeat-unit graph. The graph-based path had zero 1σ prediction violations across all seven targets; its largest spread was 0.230% of the target training scale. The explanation similarity was 0.947–0.996 (mean 0.980). This supports the use of graph and canonical features as the main representation \[8\].  
2. **Explainability.** TreeSHAP identifies which features matter most to a prediction \[9\]. We tested this with a feature-removal check: removing the top 10% SHAP-ranked features from the Tg proxy lowered R² by 0.851, versus 0.043 for a same-size random removal. The reported explanations therefore identify features the model genuinely relies on \[10\].  
3. **Generalisability and robustness.** The mean score is 0.824 when related structures are kept together and 0.660 when entire scaffolds are held out; all seven targets remain positive in both tests. The decline is useful information: a new scaffold is harder than a familiar polymer family. Tg error also rises as nearest-neighbour similarity falls, so the demo shows an applicability tier and nearest analogue rather than treating every prediction equally \[11,12\].

## **2\. Preliminary Salient Results**

### **EDA Findings that Impacted our Pipeline**

![][image3]  
**Figure 3\. EDA: label imbalance and overlap with the training data.** The left panel shows the uneven 7,409-label dataset despite equal target weights. The right panel shows that almost no exact polymer–property pairs repeat, while 88–99% of the small DFT targets have cross-property support. This motivates the target-specific lanes and guarded routes.

The data analysis changed the architecture before model tuning began. Tg contributes **55.9%** of rows and **99.986%** of pooled variance, but receives the same one-seventh score weight as Ei, which contributes **3.0%** of rows. **A single pooled loss would therefore learn mainly Tg.** Train and test share **1,063** canonical structures, so we use canonical-SMILES `GroupKFold`; the same polymer family cannot appear on both sides of an internal split.

## **3\. Technical Challenges and Pivots**

| Challenge | What we learned | Pivot |
| :---- | :---- | :---- |
| One representation does not suit every target | large Tg and scarce electronic/optical tasks behave differently | use target-specific learned lanes over a shared representation |
| A more flexible formula can be worse | the learned Egc residual failed leave-one-out testing | retain the simpler guarded relation |
| A reliable interval needs one verified audit trail | archived uncertainty summaries disagree | show applicability distance now; release calibrated intervals only after the isolated rerun |

 

Repeat-unit SMILES also omit molecular weight, tacticity, morphology, processing history and measurement conditions. We address this openly through an applicability tier and by positioning the tool for screening and experiment planning. Tg group-contribution and free-volume research provides a clear next feature direction \[13,14\].

## **4\. Final Sprint Roadmap and Evaluation Alignment**

| Evaluation expectation | What is already shown | Next release step |
| :---- | :---- | :---- |
| Feature strategy and architecture | ablations, polymer-specific representations, direct lanes and guarded routes | freeze choices with their source artifacts |
| Polymer invariance | equivalent-SMILES prediction and explanation tests | show two spellings of one polymer in the dashboard |
| Explainability | TreeSHAP plus feature-removal fidelity | group correlated descriptors into chemical concepts |
| Generalisability and robustness | grouped/scaffold tests and similarity-based applicability tier | add a pre-registered external polymer panel with literature conditions |
| Transparency | failed residual and pending uncertainty item are stated | regenerate the late uncertainty tables before release |

 

The remaining work is focused: complete the Python-3.11 isolated run, validate the uncertainty artifacts, and add a parity plot only after its source and split are verified. We are also testing a pure-ML ExtraTrees pipeline; its current grouped-CV mean R² is **0.816344**. The final package will include the pipeline, environment, evidence tables and offline dashboard to screen a new polymer, explain its structural signals, identify its known chemical space, and decide what experiment should follow.

## 

## **References**

1\.   Kuenneth, C. *et al.* Polymer informatics with multi-task learning. *Patterns* (2021). [https://doi.org/10.1016/j.patter.2021.100238](https://doi.org/10.1016/j.patter.2021.100238)

2\.   Kim, C. *et al.* Polymer Genome: a data-powered polymer informatics platform. *Journal of Physical Chemistry C*(2018). [https://doi.org/10.1021/acs.jpcc.8b02913](https://doi.org/10.1021/acs.jpcc.8b02913)

3\.   Ma, R. & Luo, Y. PI1M: a benchmark database for polymer informatics. *Journal of Chemical Information and Modeling* (2020). [https://doi.org/10.1021/acs.jcim.0c00726](https://doi.org/10.1021/acs.jcim.0c00726)

4\.   Landrum, G. RDKit: Open-source cheminformatics. [https://www.rdkit.org](https://www.rdkit.org/)

5\.   Rogers, D. & Hahn, M. Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*(2010). [https://doi.org/10.1021/ci100050t](https://doi.org/10.1021/ci100050t)

6\.   Moss, T. S. Relation between the refractive index and the energy gap of semiconductors. *physica status solidi (b)*(1985). [https://doi.org/10.1002/pssb.2221310202](https://doi.org/10.1002/pssb.2221310202)

7\.   Ravindra, N. M. & Srivastava, V. K. On the Penn model of the dielectric constant. *Infrared Physics* (1979). https://doi.org/10.1016/0020-0891(79)90013-1

8\.   Bjerrum, E. J. SMILES enumeration as data augmentation for neural network modelling of molecules (2017). [https://arxiv.org/abs/1703.07076](https://arxiv.org/abs/1703.07076)

9\.   Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. *NeurIPS* (2017). [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)

10\.                  Hooker, S. *et al.* A benchmark for interpretability methods in deep neural networks. *NeurIPS* (2019). [https://arxiv.org/abs/1806.10758](https://arxiv.org/abs/1806.10758)

11\.                  Bemis, G. W. & Murcko, M. A. The properties of known drugs. 1\. Molecular frameworks. *Journal of Medicinal Chemistry* (1996). [https://doi.org/10.1021/jm9602928](https://doi.org/10.1021/jm9602928)

12\.                  OECD. *Guidance Document on the Validation of (Q)SAR Models* (2007). [https://www.oecd.org/env/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models-9789264085442-en.htm](https://www.oecd.org/env/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models-9789264085442-en.htm)

13\.                  Bicerano, J. *Prediction of Polymer Properties.* Marcel Dekker (2002).

14\.                  Williams, M. L., Landel, R. F. & Ferry, J. D. The temperature dependence of relaxation mechanisms in amorphous polymers. *Journal of the American Chemical Society* (1955). [https://doi.org/10.1021/ja01619a008](https://doi.org/10.1021/ja01619a008)

## 

## **Appendix A — experiment register**

| Experiment family | Representative scored result | Decision |
| :---- | :---- | :---- |
| D1: physical routes | ε +0.0666; n +0.0434; Egb route 0.9205 → 0.9478 | retain the three supported routes |
| D2: representations | atomic triples: Egb 0.9167 → 0.9259; n 0.8438 → 0.8519 | retain polymer-specific triples |
| D3: label-free corpora | PI1M SVD helped; all-target corpus SVD hurt | retain the polymer-only prior |
| D5: classical ML and kernels | NNLS blend +0.02 to +0.05 over the best member | match model family to target size |
| D6: cross-property learning | partner labels helped all five small DFT targets | use availability-guarded routes |
| D8: calibration | character residual positive on five targets; isotonic calibration overfit | keep only validation-supported adjustment |

 

## 

## **Appendix B — reproducibility checklist**

| Item | Location / condition |
| :---- | :---- |
| Submitted pipeline and architecture | AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/ |
| Isolated reproduction | fixes/isolated\_runs/Sandman\_Polymer\_Property\_Prediction\_2\_906.ipynb(Python 3.11.7) |
| Claim audit | finals/REPORT\_CLAIM\_EVIDENCE.md |
| Public code link | **\[Insert final GitHub release URL/tag\]** |
