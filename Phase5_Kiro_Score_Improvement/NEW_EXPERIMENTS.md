# Polymer Property Prediction Challenge (PPP Round 3) – Overview and Data  
In this Kaggle challenge (NeurIPS Open Polymer Prediction 2025), the goal is to predict five simulated polymer properties – glass transition temperature (T<sub>g</sub>), fractional free volume (FFV), thermal conductivity (T<sub>c</sub>), mass density, and radius of gyration (R<sub>g</sub>) – from polymer structures given as SMILES strings.  The organizers generated ~11,475 unique polymer SMILES (monomer units), of which about 9,625 have at least one labeled property.  Only the training split (monomer SMILES with labels) is provided; the private test split (unseen polymers) is held out for scoring.  The official metric is a weighted mean absolute error (wMAE) across tasks (to balance scales and label imbalance).  The user mentions an additional “new dataset with 5M+ rows of SMILES” – presumably unlabeled polymer structures released by Kaggle.  **This unlabeled pool can be used for self-supervised pretraining or data augmentation**.  In summary, the data is multi-task, moderately small (∼10k), with missing labels (not every polymer has all five properties) and imbalanced scales.  

## Data Inspection and Processing  
A thorough EDA is crucial.  First, inspect target distributions and correlations: prior competitors noted that some properties are correlated (e.g. FFV and T<sub>c</sub> correlate with T<sub>g</sub> and R<sub>g</sub>).  For example, one team used predicted FFV and T<sub>c</sub> as features to predict the other targets because these abundant labels improved accuracy.  Plot histograms of each target to spot skew or outliers.  In the original challenge, a subset of polymers had extreme TC values; the winner **removed polymers with TC > 0.402** to avoid outlier bias.  Check for distribution shifts between train and test: the organizers observed a shift in the T<sub>g</sub> mean between training and private test sets.  Teams handled this by calibrating targets (see below).  Also verify SMILES validity (RDKit version issues were reported) and consistency.  

Next, perform chemical standardization: **canonicalize and kekulize** SMILES to avoid duplicate representations.  For example, top teams generated canonical SMILES (and assigned explicit bond orders) to dedupe the dataset.  Record unique polymers and remove exact duplicates.  For missing labels (some polymers lack certain properties), you can either train multi-task models that handle missing outputs or train separate models per target – try both.  If available, include simple polymer descriptors: number of atoms, atomic composition counts, or even chain-length estimates if known. 

## Feature Engineering and Augmentation  
Given small data, rich features are key.  **Fingerprint and Descriptor Features:**  All top solutions used extended-connectivity (Morgan/ECFP) fingerprints of the SMILES.  Compute Morgan bits (e.g. radius 2 or 3, 1024+ bits) for each monomer SMILES.  Also generate diverse chemical descriptors: RDKit descriptors (e.g. molecular weight, LogP, topological surface area), Mordred descriptors, and structural keys (MACCS, Atom-Pair, Topological Torsion).  One may generate hundreds of descriptors (geometry, charge, etc.) and then select with a preliminary model (see below).  Notably, the champion added **polymer-specific descriptors**: e.g. molecular volume/shape, Gasteiger charges, element composition ratios, and bond-type counts.  These capture higher-level polymer structure.  If any coarse 3D information is available (unlikely from SMILES), shape descriptors could help.  

**Augmentation / SMILES Variations:**  Data augmentation can multiply training examples.  Several teams used alternative SMILES forms: e.g. **randomized (non-canonical) SMILES** to create additional sequences.  One team explicitly concatenated multiple monomer units into a longer SMILES chain to mimic the polymer repeat (a “multi-monomer” SMILES).  You should experiment with generating BigSMILES or repeated-SMILES representations of polymers, and with stereoisomer/tautomer variants if chemically valid.  However, avoid aggressive augmentation (random permutations or stereoisomers) without domain validation, as some found it led to overfitting.  

**Feature Selection:**  With potentially thousands of raw features, apply selection.  A common pipeline is to train a simple model (like XGBoost) on all features to compute importance scores, then retain the top-N features.  Also perform collinearity filtering: drop features highly correlated with each other, or retain only descriptors with significant correlation or mutual information with the target.  For example, the 10th-place team kept descriptors only if they passed correlation or information criteria, then pruned redundant ones.  Iteratively prune feature sets and validate by CV.  

## Modeling Approaches – Proposed Experiments  
Design a matrix of experiments ranging from simple baselines to advanced models.  Key ideas, informed by top solutions and literature:

- **Baseline Models (Fingerprint + Trees):** Start with classical ML: train regression models (Random Forest, LightGBM, XGBoost, CatBoost) on the fingerprint+descriptor feature matrix.  This was the backbone of nearly every top solution.  For each target, tune hyperparameters (via grid search or Optuna).  Evaluate 5-fold CV R² (or wMAE).  Experiment with multi-output versus separate per-property models – separate models might allow per-target optimization (e.g. handle Tg shift separately).  

- **Neural Models on Descriptors:** Train feedforward neural networks on the same features (normalize inputs).  Simple dense nets can be tested, perhaps with dropout/batch-norm.  While tree models dominated, NN can capture non-linear interactions.  Use regularization (dropout, early stopping) to avoid overfitting given limited data.

- **Sequence Models (SMILES):** Treat the SMILES string as text.  Preprocess SMILES into token sequences (atom tokens or byte-level).  Train RNNs (LSTM/GRU) or Transformers from scratch (no external pretraining) on the Kaggle training set.  Optionally perform masked-language-model (MLM) pretraining on the unlabeled 5M SMILES dataset (in a BERT/RoBERTa style) – effectively building a **polymer language model**.  A Kaggle-winning idea was to use a Transformer-based polymer embedder (polyBERT) pre-trained on millions of SMILES and then fine-tuned on properties.  While you can’t use external pre-trained weights, you can pretrain on the provided 5M+ SMILES.  Evaluate the SMILES model by extracting its learned embeddings (e.g. the [CLS] token) and feeding those to a regressor, or by fine-tuning the whole model end-to-end.

- **Graph Neural Networks:** Convert each monomer SMILES into a molecular graph (atoms as nodes, bonds as edges).  Train graph conv nets (GCN, GraphSAGE) or graph attention nets (GAT/GATv2) for regression.  Indeed, the 3rd-place solution centered on a **GATv2** model with extensive feature input.  Implement a GNN in PyTorch Geometric or DGL, including node/edge features (atom type, bond type, etc.).  Try both global pooling and per-property heads (for multi-task output).  Optionally augment node features with your computed descriptors (e.g. atomic partial charges).  Evaluate GNNs with 5-fold CV.  

- **Advanced Graph Methods:** Explore recent research: e.g. **Mol-TDL** represents polymers as multi-scale simplicial complexes and uses contrastive pretraining.  While implementing simplicial networks is complex, you could mimic the idea by creating multi-scale graph features (e.g. include edges for 2-hop neighborhoods) and using contrastive learning (e.g. GraphCL on polymer graphs) before fine-tuning.  Even if not reproducing Mol-TDL exactly, consider unsupervised graph embedding (Graph Autoencoder, or graph transformer) on the unlabeled SMILES.  

- **Ensemble Techniques:** Plan experiments that ensemble different model types.  For example, you can average predictions from tree models, GNNs, and SMILES-transformers.  The champion ensemble combined a polymer Transformer (polyBERT), an AutoML tabular ensemble (AutoGluon), and a 3D molecular model (Uni-Mol).  While you may not use Uni-Mol (external), you can stack your best models.  Also try “stacked” features: one idea is to include predictions of one model as features for another.  A top-5 team fed predicted FFV and T<sub>c</sub> as inputs to the Tg/R<sub>g</sub> model.  You could replicate this by first training models for some targets, then using those predictions as extra features for the remaining targets.  

- **Target Transformations and Calibration:** Experiment with transforming targets (e.g. log or Box-Cox) if distributions are skewed.  In particular, teams saw a shift in T<sub>g</sub> distribution; they corrected it by adding a bias term or fitting a linear regressor to align train vs test.  Implement a “calibrator”: after initial predictions, fit a simple linear model on a hold-out fold to minimize error bias (especially for T<sub>g</sub>).  The 4th-place team even used **quantile regression** (predicting a high quantile) to emphasize accuracy on the high end of the Tg/R<sub>g</sub> range – this could be tried by training a model to predict the 85th percentile.

- **Hyperparameter and Feature Search:** For each model, perform thorough hyperparameter tuning (Optuna or grid).  Also tune the number of fingerprint bits, tree depths, learning rates, GNN layers, etc.  Use early stopping.  Apply k-fold CV (5-fold) with different random seeds to estimate variance.  **Feature pruning** should also be validated: e.g., start with all descriptors and use XGBoost importance to drop the least useful, verifying via CV loss.

## Validation, Ensembling, and Uncertainty  
Use rigorous cross-validation: all winning teams used 5-fold CV to estimate test performance.  Make sure folds are stratified in a chemically meaningful way if possible (e.g. by polymer family or property range). Compute metrics on each fold and check stability. Once models are trained on CV folds, **ensemble their predictions** (simple average or weighted by CV score).  For example, average your LGBM, GNN, and Transformer models’ outputs for each target.  Optionally train a meta-model (stacker) on out-of-fold predictions.  

Quantify uncertainty: you might use dropout at inference or train quantile-forest variants.  One Kagglers’ trick was to use a **quantile regression loss** (𝛼=0.85) for Tg/R<sub>g</sub> to focus on high-value errors. You could similarly predict a high quantile or compute prediction intervals from tree ensembles.  Evaluating uncertainty helps gauge confidence in predictions.  

Remember to calibrate predictions: if you observe a consistent bias (e.g. underpredicting Tg), fit a post-hoc linear correction using CV (as done for T<sub>g</sub>).  

## Implementation Roadmap  
1. **Environment Setup:** Install Python 3.8+, RDKit (for chemistry), PyTorch (with Geometric for GNNs), and ML libraries (scikit-learn, LightGBM, CatBoost, Optuna).  Use Conda to manage versions.  
2. **Data Pipeline:** Write scripts to load Kaggle CSVs.  Perform SMILES standardization (via RDKit).  Compute and save features (fingerprints, descriptors).  Incorporate any SMILES augmentation here (generate alternate SMILES strings and their features).  For reproducibility, fix random seeds.  (The third-place codebase shows a structured pipeline: prepare data → 5-fold training → prediction.)  
3. **Feature Selection:** Use a subset of data to train a preliminary XGBoost; rank features by importance.  Drop low-importance descriptors.  Also check feature correlations and remove redundant ones. Document the final feature set.  
4. **Model Training:** For each model type (tree, NN, GNN, SMILES-Transformer), implement 5-fold CV training. Log hyperparameters and CV scores. Save the fold models.  When tuning, try grid/Optuna loops on each fold.  
5. **Ensembling:** After CV, generate predictions on the validation folds (out-of-fold) and on the full train set. Ensemble models by averaging or a meta-learner. Test different ensemble weights.  
6. **Evaluation:** Write an evaluation script to compute the competition’s weighted MAE (and R² if needed) on a validation set.  Also track per-target MAEs.  
7. **Checkpoints:** Save intermediate models and results. Record observations (e.g. how each feature set or model affected CV score).  
8. **Iteration:** Based on CV results, iterate: add/remove features, try a new model variant, adjust hyperparameters.  Given the goal R²=0.935, compare your CV R² and wMAE to target.  
9. **Final Submission:** Once satisfied, train each model on the full training data (or use saved folds to predict test), ensemble them, and produce final predictions CSV for Kaggle.  

Throughout, document all steps carefully.  Follow practices such as those in the Kaggle solution: fixed seeds, pipeline scripts, and modular code.  

## Key Takeaways from References  
- **Feature-Rich, Not Overly Complex:** Top teams emphasized rich handcrafted features (fingerprints, descriptors, polymer-specific features) with careful selection rather than relying purely on black-box deep models.  
- **Data Curation Matters:** Clean and augment data thoughtfully. Canonicalize SMILES, remove duplicates/outliers, and augment with chemically valid variants.  
- **Simple Models Can Excel:** Most winners used gradient-boosted trees (XGBoost/LightGBM/CatBoost). These are strong baselines before moving to deep models.  
- **Smart Ensembling:** Combine complementary models. For example, blend a tree-ensemble, a transformer-embedding model, and a graph net. Use intermediate predictions as features for other targets if correlated.  
- **Careful Validation:** Use 5-fold CV and calibrate for any label shifts. Validate every change on held-out folds.  

By systematically applying these strategies—thorough EDA, diverse feature engineering, multiple modeling paradigms, and robust ensembling/validation—you maximize the chance of reaching the 0.935 R² target. 

**Sources:** Best practices and solution insights are drawn from the Open Polymer Challenge report, published Kaggle winners’ repositories, and recent polymer ML literature. Each recommendation above is grounded in these references.

Yes. And after going through the entire existing `NEW_EXPERIMENTS.md` and `TRIALS.md`, I agree that another layer is warranted. The previous list already covered the obvious ideas—basic EDA, ordinary SSL, generic GNNs, standard kernels, generic manifold learning, simple residuals, etc.—so I would now go after **second-order effects, data geometry, target-specific mathematics, and combinations that are materially different from what has already failed**.

One important reality check first: your current best assembly is about **0.90456**, with per-target scores around Tg 0.8971, Egc 0.9116, Egb 0.9310, Ei 0.8711, Eea 0.9183, Nc 0.9084, and Eps 0.8869.  That means you need about **+0.2206 in summed per-target R²** to reach 0.935. So this is not going to come from a single +0.003 blend.

Also, the project history explicitly says generic GNN/MPNN approaches, small-scale PI1M SSL, `ei/eea` residual learning, forced similarity routers, and several other families have already failed.  The interesting exception is that the **full-scale 6M `smile_r3` experiment remains genuinely untested**, because the earlier SSL failures were much smaller and used weak probes. 

So below are **100 additional experiments/analyses**, deliberately going beyond the existing 120.

---

# The 100-extra attack list

## 121–140 — deeper dataset and chemistry EDA

### 121. Target-label acquisition fingerprint

For each target, identify whether label availability depends on:

* SMILES length
* atom count
* aromaticity
* heteroatom count
* chemical family
* similarity to other labeled samples.

Train an `observed_target ~ X` classifier.

This is a more granular version of missingness analysis: not merely "is the target missing?", but **what selection process produced the labeled subset?**

This is particularly important because six targets have only 221–337 training observations, versus 4,143 for Tg. 

### 122. Label-support topology

For each target, construct:

```text
labeled points
      ↓
chemical clusters
      ↓
fraction of cluster labeled
```

Find clusters in which almost every point is labeled versus clusters with one or two labels.

Those highly labeled clusters are ideal candidates for local modeling.

### 123. "One labeled point per family" analysis

Measure how predictive a single labeled observation is for its chemical neighborhood.

For every training molecule:

```text
nearest same-family labeled target
nearest chemically similar labeled target
target difference
```

Estimate the empirical local smoothness of each property.

### 124. Family entropy

For each chemical cluster, calculate target entropy/variance.

A cluster with:

```text
high chemical similarity
low target variance
```

is an excellent candidate for read-across.

A cluster with high within-cluster variance tells you the representation is missing an important structural variable.

### 125. Family-conditioned R² ceiling

For every major chemical family, calculate the R² obtainable from the family mean alone, family linear model, and full model.

This answers:

> Is the remaining error mostly within families or between families?

That determines whether you should invest in **family recognition** or **fine-grained chemistry**.

### 126. Within-family residual correlation

After the best model, calculate residual correlations within chemical families.

If a model systematically overpredicts fluorinated polymers, for example, that is actionable.

### 127. Property variance decomposition

Decompose:

$$
Var(y)=Var(E[y|family])+E[Var(y|family)]
$$

Do this for every target.

A target dominated by between-family variance wants better family descriptors; one dominated by within-family variance wants local structural descriptors.

### 128. Hierarchical cluster tree

Don't use one clustering level.

Construct:

```text
global
 ├── family
 │    ├── subfamily
 │    └── subfamily
 └── family
```

Then test models at each level.

### 129. Cluster stability analysis

Repeat clustering using:

* Morgan
* continuous descriptors
* Polymer Genome features
* graph spectra
* learned embedding.

Clusters appearing consistently across representations are likely chemically meaningful.

### 130. Cluster-purity versus target-variance plot

Plot:

$$
\text{chemical cohesion} \rightarrow \text{target variance}
$$

for clusters.

This tells you whether your chemistry distance is actually useful for a given target.

### 131. Duplicate-chain-length audit

Investigate whether apparently different SMILES are really the same repeat chemistry encoded with different explicit chain contexts.

This is more specific than ordinary canonicalization.

### 132. Attachment-point equivalence classes

Group polymers by the local environments of the two polymer attachment atoms.

Then ask whether targets differ strongly between otherwise similar structures with different attachment chemistry.

This is especially promising for Tg and electronic properties.

### 133. Backbone graph isomorphism grouping

Strip side-chain information and cluster based only on backbone connectivity.

Then analyze target variance within each backbone class.

### 134. Side-chain graph isomorphism grouping

Perform the inverse:

```text
same side-chain topology
different backbone
```

This could expose systematic Tg or dielectric effects.

### 135. Backbone/side-chain ANOVA

For Tg:

$$
y = \mu + B_i + S_j + B_iS_j + \epsilon
$$

where possible.

This tells you whether backbone, side chain, or their interaction dominates.

### 136. Functional-group co-occurrence graph

Create a graph where nodes are functional groups and edges count co-occurrence.

Then derive graph statistics such as:

* degree
* clustering coefficient
* community
* motif counts.

Use them as features.

### 137. Functional-group mutual information network

Calculate mutual information between functional-group counts and every target.

Then specifically search **group-pair interactions** whose joint information exceeds either group's individual information.

### 138. Rare-chemistry indicator audit

Identify features occurring in <1%, <2%, <5% of polymers.

Test whether rare motifs disproportionately occupy the high-error validation set.

### 139. Target-space holes

For each target, find unusually large gaps between neighboring target values.

Then inspect the chemistry of the polymers surrounding those gaps.

These discontinuities may reveal physical regimes.

### 140. Property-space nearest-neighbor topology

Build kNN graphs in target space and inspect the corresponding molecular distances.

A very useful diagnostic is:

$$
d_{chem}(i,j)\quad\text{vs}\quad d_{property}(i,j)
$$

by target.

This formalizes whether the chemical representation is aligned with the property.

This kind of geometric reasoning is particularly relevant in ultra-low-data molecular prediction, where representation and coverage can dominate model complexity. ([DOI][1])

---

# 141–160 — representations that are more exotic than ordinary fingerprints

### 141. Graph Laplacian eigenvalue descriptors

Compute the first 10–30 normalized Laplacian eigenvalues of each molecular graph.

These provide global topology information absent from ordinary atom-count descriptors.

### 142. Adjacency-spectrum descriptors

Use eigenvalues of the adjacency matrix.

Try them alone and with Morgan.

### 143. Signless-Laplacian descriptors

A second graph spectrum can capture different structural information.

### 144. Spectral moments

Calculate:

$$
\mathrm{tr}(A^k),\quad k=1,\ldots,10
$$

and corresponding Laplacian moments.

### 145. Graph energy

Calculate graph energy:

$$
E(G)=\sum_i |\lambda_i|
$$

as a compact global structural descriptor.

### 146. Resistance-distance statistics

Compute effective-resistance summaries between atoms or attachment atoms.

Potentially useful for backbone connectivity and electronic transport proxies.

### 147. Wiener-index family

Calculate Wiener, Balaban, Zagreb and related graph indices.

Do not dump all descriptors into the model; test them as small specialist families.

### 148. Distance-distribution histograms

For every molecule:

```text
fraction atom pairs at graph distance 1
distance 2
...
distance 8
```

This captures topology much more explicitly than simple atom counts.

### 149. Cycle-basis descriptors

Count:

* 3/4/5/6-member cycles
* fused cycles
* spiro systems
* bridged cycles.

Especially attractive for Tg and electronic targets.

### 150. Ring-fusion topology

Separate:

```text
isolated aromatic ring
fused aromatic rings
bridged rings
condensed ring systems
```

Electronic properties may respond very differently to these.

### 151. Conjugated-subgraph size distribution

Instead of one conjugation score, calculate:

```text
largest conjugated component
mean conjugated component
95th percentile
number of components
```

### 152. Alternant/non-alternant conjugation descriptor

Classify conjugated components according to graph properties relevant to π-electronic structure.

Potentially useful for Egc/Egb/Ei/Eea.

### 153. Bond-order weighted path statistics

Calculate path lengths weighted by bond order.

### 154. Aromatic-path distance statistics

Measure distances between aromatic atoms along graph paths rather than just aromatic fraction.

### 155. Heteroatom-to-aromatic distance maps

For each heteroatom class, calculate its graph distance to the nearest aromatic center.

Examples:

```text
O→aromatic
N→aromatic
F→aromatic
carbonyl→aromatic
```

This can encode substitution chemistry much more specifically.

### 156. Donor-acceptor graph

Construct a molecular graph where donor/acceptor groups are nodes and graph distances are edges.

Features:

```text
D-D distances
A-A distances
D-A distances
```

This is more structural than simply HBD/HBA counts.

### 157. Polar-group clustering

Two polymers with identical oxygen counts can have very different chemistry if the polar groups are clustered.

Calculate:

$$
\text{polar clustering} =
\frac{\text{within-polar-group distances}}
{\text{all distances}}
$$

### 158. Heteroatom dispersion

Measure whether heteroatoms are:

* uniformly distributed
* clustered
* isolated.

### 159. Branching-depth distribution

Do not use one branching count.

Use:

```text
branch count
maximum branch depth
mean branch depth
side-chain size distribution
branching entropy
```

### 160. Descriptor ratios rather than descriptors

Systematically generate physically interpretable ratios:

$$
\frac{\text{aromatic atoms}}{\text{rotatable bonds}+1}
$$

$$
\frac{\text{polar groups}}{\text{heavy atoms}}
$$

$$
\frac{\text{ring atoms}}{\text{backbone atoms}}
$$

$$
\frac{\text{heteroatoms}}{\text{aromatic atoms}+1}
$$

Ratios often encode composition better than raw counts.

---

# 161–180 — statistically stronger small-n models

The 221–337-row targets deserve a completely different statistical toolbox. Your archive already showed that naïve GP/KRR variants were not enough, so these are **structured variants**, not "try GP again." 

### 161. Sparse additive Gaussian process

Use:

$$
K=K_{chemical}+K_{physics}+K_{topology}
$$

rather than a single kernel.

### 162. Matérn-3/2 versus Matérn-5/2 kernel sweep

Test these independently.

### 163. Spectral-mixture kernel

A spectral mixture GP can model multiple characteristic length scales.

### 164. Rational-quadratic kernel

Useful if chemistry operates over multiple distance scales.

### 165. ARD kernel

Give each descriptor family its own length scale.

This can reveal which chemistry the target actually "cares about."

### 166. Grouped ARD

Instead of one length scale per feature:

```text
topology
electronic
polarity
shape
polymer
```

each gets its own.

### 167. GP residual on **Egb only**

Your archive established that Egb is the one identity where residual learning genuinely helped. 

So specifically test:

$$
Egb=aEgc+b+GP(X)
$$

rather than a generic residual strategy.

### 168. Bayesian linear model over selected chemistry

Use strong priors/shrinkage instead of ordinary Ridge.

### 169. Bayesian polynomial regression

Only 10–30 scientifically selected interactions; regularize strongly.

### 170. Horseshoe-prior regression

Useful when only a handful of descriptors truly matter.

### 171. Elastic Net interaction model

Use nonlinear interaction candidates but enforce sparsity.

### 172. Orthogonal polynomial basis

For variables such as:

* aromaticity
* flexibility
* polarizability
* molecular volume.

Better conditioned than naïve \(x^k\).

### 173. Robust local polynomial regression

Within chemical neighborhoods, fit low-order polynomial models.

### 174. Local linear regression

Predict from nearest chemical neighbors with distance-weighted coefficients.

### 175. LOESS-on-chemical-manifold

Reduce chemistry to 3–10 stable manifold coordinates, then use local regression.

### 176. Quantile random forest median versus mean

For tiny targets, test whether the forest's conditional median is more stable than its mean.

### 177. Extremely randomized trees with target-specific leaf regularization

Optimize `min_samples_leaf` separately per target rather than globally.

### 178. Bayesian bootstrap prediction

Instead of ordinary bootstrap averages, average predictions over Dirichlet-weighted training samples.

### 179. Bootstrap-of-model-and-feature

Resample both:

* rows
* descriptor families.

Then average.

This produces *structural* rather than only seed diversity.

### 180. Jackknife model averaging

Train leave-one-cluster-out models and use their predictions as a diversity ensemble.

---

# 181–200 — semi-supervised approaches genuinely exploiting 6M

This is where I would spend serious compute.

The project archive itself says the previous PI1M SSL failures do **not** rule out a full-scale 6M experiment with stronger downstream heads. 

And independent research supports using large unlabeled molecular pools through semi-supervised representation learning—but it also warns that unlabeled-data geometry can cause degeneration if handled naïvely. ([ScienceDirect][2])

### 181. 6M character language model → tree head

Train a modest character LM on all `smile_r3`.

Then:

```text
SMILES → embedding → LightGBM
```

not a linear probe.

### 182. 6M masked-token model → Ridge + ET

Use the embedding with both linear and nonlinear heads.

### 183. 6M masked-span model

Mask contiguous SMILES spans rather than individual characters/tokens.

### 184. 6M atom-token MLM

Tokenize chemically rather than by character.

### 185. 6M bond-token MLM

Train on bond sequences specifically.

### 186. 6M graph autoencoder

Encode molecular graphs and reconstruct:

* node identity
* degree
* bond type.

Use latent vectors as features.

### 187. 6M masked-node graph pretraining

Mask atoms and predict them.

### 188. 6M masked-edge graph pretraining

Predict hidden bonds.

### 189. 6M graph contrastive learning

Positive examples:

* randomized SMILES
* atom permutation
* graph traversal.

Negative examples from unrelated structures.

### 190. 6M **physics-aware** contrastive learning

Do not merely make two views close.

Make representation distances preserve inexpensive deterministic structural properties:

```text
Δ aromaticity
Δ polarity
Δ conjugation
Δ topology
```

### 191. 6M denoising autoencoder with property-oriented reconstruction

Reconstruct:

* atom types
* bond types
* physicochemical descriptors.

### 192. 6M multi-task autoencoder

Predict both:

* reconstructed SMILES
* RDKit descriptor vector.

### 193. 6M autoencoder + target alignment

After unsupervised training, learn a linear transformation that maximizes covariance with each target.

### 194. 6M embedding density estimator

Rather than using embeddings directly, derive:

```text
local density
kNN density
distance to centroid
distance to training manifold
```

### 195. 6M nearest-neighbor retrieval

Use unlabeled molecules to identify *dense chemical regions* around each test molecule.

You don't need labels.

The neighbors themselves define a prior over structural behavior.

### 196. 6M manifold prototype features

Cluster the 6M dataset into thousands of prototypes.

For each train/test molecule record:

```text
distance to prototype 1
distance to nearest prototype
prototype assignment
prototype density
```

### 197. 6M out-of-distribution score

Measure:

$$
OOD(x)=\min_j d(x,c_j)
$$

where \(c_j\) are learned unlabeled-corpus centroids.

Use this in validation and final ensemble weighting.

### 198. 6M density-conditioned CV

Construct validation so train/validation density matches test density.

This is different from simple random/group CV.

### 199. 6M self-training with **validation-calibrated confidence**

Train a teacher on labeled data.

Pseudo-label only when:

```text
ensemble variance < threshold
+
6M density > threshold
+
representation agreement > threshold
```

Recent work specifically supports uncertainty-aware semi-supervised molecular prediction, but consensus/pseudo-labeling should be tested carefully rather than assumed to work. ([MI Research][3])

### 200. 6M consensus learning

Train several independently initialized models and force an additional student to agree on **high-confidence unlabeled molecules**.

A July 2026 paper reports promising ensemble-consensus semi-supervised molecular learning, including improved robustness/calibration; this is especially interesting because it is a mechanism rather than merely "more MLM." ([arXiv][4])

---

# 201–215 — electronic-property attack specifically

I would aggressively prioritize these four:

```text
Egc
Egb
Ei
Eea
```

because they arise from related underlying electronic structure, while the dataset is tiny.

### 201. HOMO proxy from graph spectral descriptors

Use graph spectral variables to estimate a cheap HOMO-like latent.

### 202. LUMO proxy separately

Same architecture, but target specifically electron-accepting behavior.

### 203. Donor-strength index

Construct a purely structural electron-donor score from:

* heteroatom type
* aromatic substitution
* resonance-capable bonds
* electron-rich motifs.

### 204. Acceptor-strength index

Analogously for:

* carbonyl
* nitrile
* sulfone
* fluorinated
* strongly electron-withdrawing groups.

### 205. Donor × acceptor separation

Calculate graph distance between donor and acceptor groups.

This may matter more than their counts.

### 206. Conjugation × donor interaction

Explicitly generate:

$$
D \times L_{\pi}
$$

### 207. Conjugation × acceptor interaction

$$
A \times L_{\pi}
$$

### 208. Conjugation saturation feature

Instead of assuming longer conjugation always changes the property linearly, use:

$$
1-e^{-L_\pi/\tau}
$$

with \(\tau\) learned through CV.

### 209. Heteroatom substitution position

Encode whether electron-active atoms are:

* directly attached to aromatic system
* one bond away
* remote.

### 210. Resonance-path count

Count paths connecting electron-active groups through conjugated bonds.

### 211. Aromatic fusion score

Separate:

```text
isolated aromatic
fused aromatic
hetero-fused aromatic
```

### 212. Electronic-property latent factor model

Fit:

$$
[Egc,Egb,Ei,Eea]^T = \Lambda z + \epsilon
$$

but allow nonlinear target-specific decoders.

This is deliberately **not** the old naïve low-rank multitask model.

### 213. Conditional latent electronic model

Infer:

```text
z_electronic
     ↓
Egc
Egb
Ei
Eea
```

while also retaining direct target-specific residual paths.

### 214. Identity-constrained prediction with inequality constraints

Rather than forcing:

$$
Ei=Egc+Eea
$$

during all training, enforce it as a **soft constraint only on rows with sufficient partner support**, while allowing uncertainty.

### 215. Egb conditional on Egc plus structural delta

Explicitly:

$$
Egb=aEgc+b+g(X)
$$

where \(g(X)\) uses only structural features.

This is strongly motivated by the one identity/residual mechanism that actually worked historically. 

---

# 216–225 — Eps/Nc attack

These deserve their own mini-research program because your archive already demonstrated that the decomposition

$$
\epsilon=n_c^2+\epsilon_{ionic}
$$

was by far the strongest weak-target mechanism. C214 improved Eps by about 0.0666 and C252 improved Nc by about 0.0434 in the earlier work. 

### 216. Multiple ionic estimators

Predict ionic contribution with:

```text
Ridge
ExtraTrees
LightGBM
Huber
quantile ET
```

and ensemble them.

### 217. Ionic uncertainty

Estimate uncertainty specifically on:

$$
\epsilon-n_c^2
$$

and use it as a routing variable.

### 218. Ionic heteroscedastic model

Predict:

$$
E[\epsilon_{ionic}|X],\quad Var(\epsilon_{ionic}|X)
$$

instead of just the mean.

### 219. Polarizability-density interaction

Create:

$$
\alpha / V
$$

as a proxy for optical density.

### 220. Polarizability × aromaticity

Explicit nonlinear interaction.

### 221. Polarizability × heteroatom type

Separate oxygen/nitrogen/halogen/sulfur effects.

### 222. Polarizability per repeat mass

Use:

$$
\alpha/M
$$

or the equivalent RDKit-computable normalized forms.

### 223. Nc model → Eps correction

Treat Nc as the first-stage model and let Eps learn only its residual **after** inserting \(n_c^2\).

### 224. Eps model → Nc correction

Run the reverse direction and compare transfer.

### 225. Joint covariance model

Model:

$$
[n_c,\epsilon_{ionic}]
$$

jointly, then reconstruct Eps.

This is a refinement of the known decomposition, not the failed Lorentz–Lorenz transform. Your archive specifically found Lorentz–Lorenz/Clausius–Mossotti formulations worse than the simpler decomposition, so I would not revisit those. 

---

# 226–235 — Tg-only ideas

Tg is different because it is experimental and noisy. The project records a no-archive Tg around 0.89 and notes the experimental nature of the target.  Recent polymer Tg work likewise found that topological descriptors and nonlinear manifold transformations matter, with rotational degrees of freedom and backbone characteristics being important. ([PubMed Central (PMC)][5])

### 226. Torsional entropy proxy

Estimate the number and diversity of low-cost torsional degrees of freedom.

### 227. Rotational-state diversity

For rotatable bonds, distinguish:

```text
backbone rotor
side-chain rotor
aryl-single-bond rotor
heteroatom-adjacent rotor
```

### 228. Steric rotor suppression

A rotor surrounded by bulky substituents is not equivalent to a free rotor.

Construct a local steric penalty for every rotatable bond.

### 229. Backbone rigidity score from shortest-path structure

Calculate how constrained the backbone is by rings and unsaturation.

### 230. Side-chain packing score

Combine:

* side-chain length
* branching
* aromaticity
* heteroatom density.

### 231. Symmetry score

Construct approximate molecular symmetry measures.

### 232. Shape anisotropy from low-cost conformers

Not a generic 3D sweep: use **statistics of anisotropy across several generated conformers**.

### 233. Conformational diversity score

Generate a small number of conformers and calculate:

```text
RMSD distribution
radius of gyration distribution
principal-axis distribution
```

Then use the *distribution* rather than one conformer.

### 234. Tg high-tail specialist

Train an explicit model on the upper Tg regime and combine it with the global model.

The rationale is supported by prior competition experience: teams found target-distribution correction and quantile regression useful for shifted Tg distributions. ([PubMed Central (PMC)][5])

### 235. Tg pairwise-difference model

For nearby chemical pairs:

$$
Tg_i-Tg_j=f(X_i-X_j)
$$

then reconstruct absolute Tg.

This is attractive because absolute experimental Tg contains measurement and family effects that may partly cancel in differences.

---

# 236–245 — genuinely stronger ensemble strategies

Your old work has already shown that ordinary per-target NNLS blending is strong, while rich stacks and forced routers can collapse.  So the next step is **structured ensemble selection**, not "more models."

### 236. Error-correlation constrained NNLS

Optimize weights subject to a maximum pairwise residual correlation.

### 237. Minimum-description ensemble

Prefer the smallest model subset whose OOF error is within 0.001 of the best ensemble.

This guards against accidental overfitting.

### 238. Stability-weighted ensemble

Downweight components whose CV score varies substantially across folds.

### 239. Regime-specific model weights

Use:

```text
model weights = f(
    chemical density,
    OOD score,
    uncertainty,
    family
)
```

with a *very low-capacity* linear meta-model.

### 240. Target-specific diversity penalty

For each target choose components by:

$$
Score =
R^2-\lambda\,Corr(error_i,error_j)
$$

### 241. Fold-consensus weights

Calculate ensemble weights independently on each fold.

Use only weights that remain stable.

### 242. Bootstrap weight stability

Repeat blend fitting over bootstrap OOF samples.

Discard weights that swing wildly.

### 243. Leave-one-family-out ensemble selection

Choose the ensemble using family-held-out OOF rather than ordinary OOF alone.

### 244. Test-geometry weighted ensemble

Choose model weights according to the similarity distribution of actual test structures.

### 245. Physics-aware ensemble

For Eps/Nc/Ei/Eea, score models by:

$$
R^2-\lambda\,PhysicsViolation
$$

rather than R² alone.

Recent molecular prediction literature supports using uncertainty and calibrated ensembles as an explicit component of reliable prediction rather than treating UQ as an afterthought. ([ACS Publications][6])

---

# 246–255 — final 10: unusual but worth trying

### 246. Nearest-neighbor residual covariance

Instead of independent residuals, model:

$$
Cov(e_i,e_j)
$$

as a function of chemical similarity.

### 247. Gaussian Markov random field over chemical clusters

Smooth predictions within cluster while preserving the supervised signal.

### 248. Graph-harmonic interpolation with a **small unlabeled graph**

Use only the local neighborhood around a test molecule rather than the full 6M graph.

This avoids the degeneracy problem identified for naïve Laplacian SSL with huge unlabeled sets. ([cris.technion.ac.il][7])

### 249. Prototype regression

Replace thousands of individual training samples with chemically coherent prototype centroids and learn prototype-to-target mappings.

### 250. Prototype residual model

Predict from global model, then correct relative to nearest chemical prototype.

### 251. Conformalized ensemble diagnostics

Use conformal-style residual intervals to identify unreliable regions; don't necessarily use intervals directly for predictions.

Conformalized molecular regression has shown that calibrated uncertainty can be useful for identifying unreliable/OOD predictions. ([ACS Publications][6])

### 252. Error-directed representation search

Take only the worst 10% OOF errors and search for descriptors that discriminate them from the rest.

This is deliberately **error-first feature discovery**, not global feature selection.

### 253. Adversarial feature perturbation

For neural models, perturb chemically meaningful descriptor groups during training and require prediction stability.

### 254. Representation dropout ensemble

Train multiple models where each removes an entire feature family:

```text
no topology
no polarity
no electronics
no polymer descriptors
```

Then ensemble.

This often creates more useful diversity than seeds.

### 255. Cross-representation nearest-neighbor agreement

For each test molecule compare neighbors under:

```text
Morgan
graph spectrum
physics
SMILES
6M learned embedding
```

If all agree, confidence is high.

If they disagree, use the disagreement as a learned uncertainty feature.

---

# The experiments I would actually prioritize

I would **not** run these 135 ideas sequentially. With the deadline approaching, I would create five attack tracks.

## Track A — immediate, essentially free

Do these first:

1. Target-support topology.
2. Family entropy.
3. Within-family residual correlation.
4. Backbone/side-chain grouping.
5. Functional-group co-occurrence.
6. Property-space/chemical-space geometry.
7. Laplacian spectral descriptors.
8. Conjugated-component distributions.
9. donor/acceptor distance features.
10. ratio descriptors.
11. error-atlas-driven feature discovery.
12. representation-specific distance → error curves.

The important point is that these are **analysis engines that tell you what to build next**, not just more models.

Your project files already argue strongly for an error-atlas style research loop rather than blind `model → CV → model`; the archive explicitly proposes discovering the missing mechanism from the error structure. 

---

# Track B — the biggest potential gain: Eps/Nc/Ei/Eea

My highest-priority target strategy would be:

```text
                 ┌── Eps
                 │
polarizability ──┼── Nc
                 │
                 └── ionic

conjugation ─────┬── Egc
                 ├── Egb
                 ├── Ei
                 └── Eea
```

The reason is historical evidence.

Your prior work already demonstrated that the electronic/optical targets have physically meaningful relationships:

$$
\epsilon=n_c^2+\epsilon_{ionic}
$$

$$
E_i=E_{gc}+E_{ea}
$$

and approximately

$$
E_{gb}=aE_{gc}+b.
$$

These were among the strongest archive-free mechanisms in the entire experiment history. 

But the next breakthrough probably isn't another application of those identities. It is likely to be **better representations of the residual physical degrees of freedom**.

That is why I'd prioritize:

**Eps/Nc**
→ improved ionic model
→ uncertainty of ionic model
→ polarizability-density representation
→ target-specific kernel
→ joint \([n_c,\epsilon_{ionic}]\) model
→ physics-aware ensemble.

**Ei/Eea**
→ donor/acceptor topology
→ graph spectral descriptors
→ conjugation paths
→ donor/acceptor distance
→ property-aware kernel
→ identity-constrained latent model.

---

# Track C — the 6M bet

This is the experiment family I would take most seriously because it is the biggest resource available that has not already been exhausted.

The competition rules explicitly allow `smile_r3.csv` and `PI1M.csv` for representation learning **from scratch inside the notebook**, with no pretrained weights or vocabulary. 

And your experiment archive makes an important distinction:

> the older SSL failures were small-scale experiments and do **not** establish that a 6M-scale representation with a strong nonlinear downstream head will fail. 

So my serious ladder would be:

```text
6M corpus
  │
  ├── char LM
  ├── atom-token MLM
  ├── masked-span MLM
  ├── graph masked-node
  ├── graph masked-edge
  ├── graph contrastive
  └── descriptor reconstruction
           ↓
      embeddings
           ↓
   Ridge + ET + LGBM
           ↓
      target-specific
```

Do **not** declare this family dead based on a linear probe. The literature supports semi-supervised molecular representation learning, but also shows why naïve graph-Laplacian or poorly controlled unlabeled-data methods can collapse. ([MI Research][3])

---

# Track D — the Tg rescue

For Tg, I would stop thinking primarily in terms of "more descriptors."

The external literature is unusually compatible with your existing observations: recent Tg work identifies rotational degrees of freedom and backbone characteristics as important, and reports nonlinear relationships that benefit from manifold transformations. ([PubMed Central (PMC)][5])

So the Tg experiment should become:

```text
backbone rigidity
+
torsional freedom
+
steric rotor suppression
+
side-chain packing
+
symmetry
+
conformational diversity
+
nonlinear manifold
+
high-tail specialist
```

The key new idea is to make the representation **explicitly mechanistic**, rather than adding another generic RDKit descriptor block.

---

# Track E — use the test set structurally, without leaking labels

The test set is 4,940 rows but only 4,497 unique SMILES, and **457 structures occur in both train and test**, so structure grouping is mandatory. 

I would make an entire "test geometry engine":

```text
test molecule
    │
    ├── nearest Morgan neighbors
    ├── nearest graph-spectrum neighbors
    ├── nearest physics neighbors
    ├── nearest 6M-embedding neighbors
    ├── family
    ├── cluster
    ├── OOD score
    ├── local density
    └── model disagreement
```

Then make these features available to the **ensemble-selection layer**, not directly to arbitrary models.

That is substantially safer than the forced similarity/read-across approaches that previously collapsed.

---

# A concrete attack on the +0.032

Here is how I would *plan* the required improvement rather than pretending any particular number is guaranteed.

Current approximate target profile:

| Target | Current | Strategic target |
| ------ | ------: | ---------------: |
| Tg     |   0.897 |      0.922–0.925 |
| Egc    |   0.912 |      0.920–0.925 |
| Egb    |   0.931 |           0.950+ |
| Ei     |   0.871 |      0.925–0.935 |
| Eea    |   0.918 |      0.925–0.930 |
| Nc     |   0.908 |      0.940–0.950 |
| Eps    |   0.887 |      0.940–0.950 |

One possible allocation is approximately:

$$
+0.025 +0.010 +0.019 +0.055 +0.007 +0.032 +0.061
\approx +0.209
$$

with the remaining ~0.012 coming from ensemble/selection effects.

That is **not a forecast**. It is the research budget I would use.

The point is that the route to 0.935 is almost certainly something like:

```text
Tg        moderate breakthrough
Egc       modest breakthrough
Egb       targeted breakthrough
Ei        LARGE breakthrough
Eea       modest breakthrough
Nc        LARGE breakthrough
Eps       LARGE breakthrough
```

rather than trying to make Tg go from .897 → .97.

---

# One particularly important thing I would change

I would stop asking:

> "Did this model improve CV?"

and start asking three separate questions:

### 1. Does it improve interpolation?

Ordinary grouped CV.

### 2. Does it improve the **test-like chemical regime**?

Your shift-matched panels.

### 3. Does it improve a **different validation geometry**?

Family holdout, low-similarity, cluster holdout, OOD.

Your project already has exactly this philosophy built into the experiment loop. 

A model that goes:

```text
ordinary CV    +0.015
shift CV       +0.009
family CV      +0.011
```

is dramatically more interesting than one that goes:

```text
ordinary CV    +0.020
shift CV       -0.015
family CV      -0.030
```

The latter is almost certainly another false positive.

---

# And one more thing: don't throw away the small-target "near misses"

Your archive contains a very important lesson.

Several experiments were:

```text
positive in all/most folds
positive bootstrap
```

but rejected because they missed the +0.01 promotion threshold.

The project itself concludes that this gate may have thrown away real cumulative signal and recommends **shrinkage rather than automatic rejection** for these cases. 

That is important.

For a seven-target average, ten separate:

$$
+0.003
$$

components can be worth more than one:

$$
+0.012
$$

component.

So the final research architecture should be:

```text
             strong parent
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
    +0.003      +0.004     +0.005
       │          │          │
       └──────────┼──────────┘
                  ↓
            shrinkage
                  ↓
             final target
```

rather than insisting that every individual component must clear +0.01.

---

# My top 20 from these 100

If I personally had the machine and this deadline, my next twenty would be:

1. **6M char/atom LM → nonlinear GBM embeddings**
2. **6M masked-span model**
3. **6M graph masked-node pretraining**
4. **6M graph contrastive + ET/LGBM head**
5. **6M embedding density/OOD features**
6. **Eps joint `[Nc², ionic]` model**
7. **Ionic uncertainty → ensemble weighting**
8. **Polarizability-density Eps/Nc model**
9. **Donor/acceptor graph features for Ei/Eea**
10. **Conjugated-component distributions for Egc/Egb/Ei/Eea**
11. **Egb = affine(Egc) + graph-structured residual**
12. **Property-specific graph spectral descriptors**
13. **Tg torsional/steric rotor representation**
14. **Tg conformational-diversity features**
15. **Tg family/backbone/side-chain hierarchical model**
16. **Target-specific ARD kernels**
17. **Error-directed feature discovery from worst 10%**
18. **Chemical-family residual ensemble**
19. **Test-geometry-conditioned ensemble weighting**
20. **Consensus semi-supervised learning on 6M**

There is respectable evidence behind several of these directions: hierarchical backbone/side-chain representations are an explicit focus of PolyMetriX; semi-supervised molecular models have been proposed for limited labels; missing-label imputation has shown gains in multi-task molecular prediction; and recent work supports uncertainty/ensemble-consensus mechanisms for molecular SSL. ([Nature][8])

Most importantly, **none of those citations are evidence that the experiment will improve this particular Kaggle score**. They establish plausibility and precedent. Whether a given experiment clears your 0.935 objective has to be established by the project's own grouped/shift-matched validation and then the post-freeze oracle check.

The big strategic difference is that this isn't another 100-item generic "try XGBoost / try a GNN" list. It is aimed at finding **missing information** in the representations, particularly for the four difficult regimes `Ei/Eea/Nc/Eps`, while exploiting the one genuinely new asset—**~6 million official unlabeled SMILES**—at a scale your previous experiments did not test. Your own project records support that as the most defensible place to hunt for a step-change. 

You're right to call this out. The previous answer gave you **more ideas**, but it did not solve the real problem: **why your system is stuck around 0.90 and what kind of new information could plausibly move it to 0.935**.

I think I understand the gap much better now.

## The diagnosis: you are not suffering from a model shortage

Your history is unusually informative.

You have already tried hundreds of experiments. The strongest clean/no-archive system is around 0.9035–0.9046, while the oracle-assisted diagnostics can reach ~0.95 but are not reproducible and therefore unusable. 

Your current target profile is approximately:

| Target |     R² |
| ------ | -----: |
| Tg     | ~0.897 |
| Egc    | ~0.912 |
| Egb    | ~0.931 |
| Ei     | ~0.871 |
| Eea    | ~0.918 |
| Nc     | ~0.908 |
| Eps    | ~0.887 |

The weak points are **not uniformly weak**. Egb is already strong. Tg/Egc are moderately strong. The real damage comes from **Ei, Eps, Nc**, followed by Tg/Eea. The project itself identifies Ei and Eps as particularly weak. 

And there is an even more important fact:

**the winning historical mechanisms were not generic ML.**

They were:

* physical decomposition;
* cross-target information;
* exact/near structure information;
* specially constructed representations;
* carefully selected target-specific residuals;
* robust assembly/shrinkage.

That is explicitly what the experiment archive says. 

Meanwhile, generic GNNs, small-data transformers, PI1M SSL probes, similarity routers, naïve multitask models, etc. have already been killed. 

So **I would stop trying to find a better predictor.**

I'd try to determine whether there is a **better representation of the problem itself**.

---

# The biggest thing I think we have missed

## Treat the data as a partially observed PROPERTY MATRIX

This is the most important new direction I would attack.

Your data is not really:

```text
SMILES → one target
```

It is closer to:

```text
                    Tg  Egc Egb Ei Eea Nc Eps
polymer A            ?    ?   ✓   ?  ✓   ?  ?
polymer B            ✓    ?   ?   ✓  ?   ✓  ?
polymer C            ?    ✓   ?   ?  ?   ?  ✓
...
```

because the **same chemical structures occur across target types**, and the project has already exploited approximately 60% test-time partner-label availability. 

The existing system mostly treats those as separate regressions plus some manually designed identities.

I think we should attack the underlying **matrix-completion problem** directly.

There is recent molecular-ML literature specifically addressing missing molecular-property labels by constructing molecule–task relationships and imputing missing labels, with reliable pseudo-label selection. ([MI Research][1])

And structured multitask learning has independently shown that explicit **relationships among tasks** can improve low-label molecular prediction. ([Proceedings of Machine Learning Research][2])

### What I would build

First construct:

$$
Y_{m,t}
$$

where \(m\) is canonical polymer structure and \(t\) is one of the seven targets.

Then factorize it as:

$$
Y_{m,t}\approx \mu_t + U_m^\top V_t
$$

but **with chemistry features attached to \(U_m\)**.

Something like:

$$
U_m=f(X_m)
$$

and

$$
Y_{m,t}=f_t(X_m,z_m)
$$

where \(z_m\) is a learned latent property state.

But critically:

### Do not use a naïve low-rank model.

That was already tried and rejected. 

Instead:

```text
                    ┌── Tg
                    │
chemistry → latent ─┼── Egc
                    │
                    ├── Egb
                    │
                    ├── Ei
                    │
                    ├── Eea
                    │
                    ├── Nc
                    │
                    └── Eps
```

with **target-specific nonlinear heads**.

And then impose the known structural relationships:

$$
E_i = E_{gc}+E_{ea}
$$

$$
E_{gb}\approx aE_{gc}+b
$$

$$
\epsilon=n_c^2+\epsilon_{ionic}.
$$

The point is that this becomes a **joint partially observed latent-variable problem**, rather than seven regressions.

This is different enough from the old multitask experiments that I think it deserves a serious run.

---

# But there is an even more interesting variation

## Learn the *task graph*

Don't assume the seven properties have equal relationships.

Construct the empirical task graph:

```text
            Egc
           /   \
         Egb   Ei
               |
              Eea
                 
             Nc ─── Eps

Tg mostly separate
```

Estimate edges from:

1. rows sharing the same polymer;
2. correlations where both labels exist;
3. residual correlations;
4. physical identities;
5. representation similarity.

Then allow the model to learn:

$$
\text{message passing over tasks}.
$$

For example:

```text
polymer encoder
      ↓
property latent
      ↓
task graph
 ┌────┼────┐
Egc  Eea   Nc
 │    │     │
Egb  Ei    Eps
```

This is directly inspired by structured molecular multitask learning, where task relationships are explicitly modeled rather than treating tasks as an undifferentiated set. ([Proceedings of Machine Learning Research][2])

---

# Second major gap: you are probably modeling the wrong *unit*

This is where I think the competition may be exploitable in a completely legitimate way.

Your data has:

**7,409 training rows, but only 4,497 unique test SMILES and 457 train/test overlapping structures.** 

The natural learning unit isn't necessarily the row.

It is:

> **chemical identity × target × structural family**

I would build a **polymer identity graph**.

Every unique canonical structure is a node.

Add edges for:

* exact identity;
* same backbone;
* same side-chain family;
* Morgan similarity;
* graph-spectrum similarity;
* attachment-point similarity;
* same elemental composition pattern;
* same conjugated-system topology.

Then learn property propagation over that graph.

Not generic GNN message passing.

Something much simpler:

$$
\hat y_i =
w_0 f_{global}(x_i)
+
\sum_{j\in N(i)}w_{ij}(y_j-f_{global}(x_j)).
$$

In words:

> global chemical model + locally propagated residual.

This is different from your previously failed similarity router because **you aren't deciding whether the neighbor is trustworthy using a hard threshold**. You are using neighbors to estimate a *residual field*.

That distinction matters.

---

# Third: I think the 6M data should not primarily be used as "pretraining"

This is where I think your earlier research went slightly wrong.

You have:

* ~1M PI1M
* ~6M `smile_r3`

and the existing plan says "MLM / SSL / embedding."

But perhaps **the value of the 6M isn't a better encoder**.

The value may simply be:

# a gigantic estimate of chemical-space density.

TransPolymer is a useful clue here: its authors showed that large unlabeled polymer corpora can produce embeddings whose geometry aligns with polymer-property distributions, and the official implementation pretrains on roughly 5M polymer sequences. ([GitHub][3])

But your constraints prevent using that pretrained model.

That's fine.

We can instead learn **chemical-space geometry from scratch** and use it only as geometry.

---

# 6M experiment #1: density fields

For every train/test molecule calculate:

$$
\rho(x)
$$

using the 6M corpus.

Then:

$$
d(x,\text{6M})
$$

and

$$
d(x,\text{labeled training})
$$

and importantly:

$$
\frac{\rho_{\text{6M}}(x)}
{\rho_{\text{labeled}}(x)}.
$$

Now ask:

> Does high density in the unlabeled universe predict lower validation error?

That's a much more fundamental question than:

> "Did my MLM embedding improve R²?"

If yes, density becomes a **test-regime correction mechanism**.

---

# 6M experiment #2: discover chemical prototypes

Cluster 6M molecules into, say:

```text
5k prototypes
10k prototypes
20k prototypes
```

Then represent each polymer as:

```text
distance to prototype 1
distance to prototype 2
...
prototype distribution
```

Don't feed all distances.

Instead derive:

* nearest prototype;
* distance to nearest prototype;
* entropy of prototype membership;
* local prototype density;
* number of nearby prototypes.

Then fit the target models.

This gives you a learned **chemical coordinate system without pretrained weights**.

---

# 6M experiment #3: property-conditioned manifold transfer

Here's the interesting part.

Fit the unlabeled chemical manifold first.

Then take the ~220 Ei labels.

Ask:

> Along which manifold directions does Ei vary?

Do the same separately for:

* Ei
* Eea
* Nc
* Eps
* Egb.

You may discover that the useful directions differ.

That gives:

$$
z_{chemical}
\rightarrow
z_{Ei}
$$

rather than assuming that a generic embedding is useful for all properties.

---

# Fourth major gap: the small targets are statistically absurdly underdetermined

Look at the counts:

* Egb: 337
* Ei: 222
* Eea: 221
* Nc: 229
* Eps: 229

versus 4,143 Tg samples. 

Trying to make a huge nonlinear network learn these functions independently is almost certainly the wrong regime.

The literature now explicitly studies molecular prediction in ultra-low-data regimes and notes severe task imbalance as a core problem. ([PubMed Central (PMC)][4])

Therefore:

## use the high-data targets as *structural teachers*, not merely predictors.

For example:

```text
Tg / Egc
     ↓
learn rich chemistry representation

         ↓

Egb / Ei / Eea / Nc / Eps
```

But don't simply feed predicted Tg into Ei.

Instead use **representation transfer**:

$$
X\rightarrow z
$$

trained using all available labels, then use target-specific heads.

That distinction lets the small targets borrow information about chemistry without creating the circularity problem that previously destroyed the cross-property stacks. Your archive explicitly documented a circularity failure where apparently excellent OOF performance collapsed on transfer. 

---

# Fifth gap: you may be losing massive R² through the extremes

Remember:

$$
R^2 = 1-\frac{SSE}{SST}
$$

so one or two catastrophic errors can matter enormously.

The previous research correctly noticed this, but I think we should operationalize it much more aggressively.

For every target:

### Make a "catastrophe table"

For each OOF sample:

```text
true y
prediction
error
squared error
chemical family
nearest training distance
ensemble disagreement
descriptor percentile
```

Now rank by **squared error contribution**.

Then calculate:

```text
top 1% → % of total SSE
top 5% → % of total SSE
top 10% → % of total SSE
```

This is possibly the single most useful diagnostic you can perform.

Suppose Ei tells us:

```text
5% of molecules = 42% of SSE
```

Then the real problem isn't "Ei prediction."

It is:

> **What chemistry causes those 5% to catastrophically fail?**

That immediately points toward a specialist.

---

# Sixth: I would attack Ei very differently

The archive tells us that:

$$
Ei=Egc+Eea
$$

works **when partner observations are available**, but generic ML residual learning around this identity actually hurts badly. 

So stop attempting to improve:

$$
Ei = Egc+Eea+r(X)
$$

with a generic residual learner.

Instead do:

$$
Ei =
\underbrace{Egc+Eea}_{physical coordinate}
+
\underbrace{r(z_{electronic})}_{very\ low\ dimension}.
$$

Where \(z_{electronic}\) consists of only a handful of scientifically motivated coordinates:

* conjugation length;
* donor strength;
* acceptor strength;
* donor–acceptor separation;
* aromatic fusion;
* heteroatom environment;
* backbone rigidity.

Then fit **very heavily regularized** regression.

I would test:

* Bayesian linear regression;
* spline regression;
* monotonic GAM;
* ridge polynomial;
* GP with ARD.

Not a deep network.

---

# Seventh: the same philosophy for Eps

The most important question isn't:

> Can we predict epsilon better?

It is:

> Can we predict the **ionic component** better?

Because your existing work established:

$$
\epsilon=n_c^2+\epsilon_{ionic}
$$

and the ionic representation was substantially better conditioned. 

So I would create a dedicated latent decomposition:

$$
X\rightarrow
\begin{cases}
\hat n\\
\hat{\epsilon}_{ionic}
\end{cases}
\rightarrow
\epsilon.
$$

But then do something new:

### model their covariance

Instead of two independent regressors:

$$
p(n,\epsilon_{ionic}\mid X).
$$

Even a simple multivariate Gaussian head could work.

Why?

Because the two quantities are not arbitrary independent numbers; they arise from related polarizability/electronic structure.

---

# Eighth: a very promising new route — **learn the residual field**

This is the experiment I most want run after the matrix model.

Suppose:

$$
\hat y_{global}(x)
$$

is your current best model.

For every training molecule, compute the cross-fitted residual:

$$
r_i=y_i-\hat y_{global}(x_i).
$$

Now instead of asking:

> what predicts \(y\)?

ask:

> what predicts **where the global model is wrong**?

Train:

$$
r(x)=g(
chemical\ family,
local\ density,
backbone,
sidechain,
spectral,
physics).
$$

But critically:

**do not use this as a generic residual booster.**

Use it only where the residual field is demonstrably spatially smooth.

We can test this.

Calculate:

$$
Corr(r_i,r_j)
$$

for chemically similar \(i,j\).

If residual similarity exists, you've found something extremely valuable:

> the error itself has structure.

Then local residual correction becomes justified.

---

# Ninth: exploit *signed* residual neighborhoods

Suppose:

```text
polymer A
global model error = -0.8

neighbors:
B -0.7
C -0.9
D -0.8
```

That strongly suggests the model has a systematic local bias.

Your current local methods focus on predicting \(y\).

Instead predict:

$$
E[r|x,N(x)].
$$

This is essentially **kriging of model error**.

I would try:

$$
\hat y =
\hat y_{global}
+
\alpha(x)\hat r_{local}
$$

where \(\alpha(x)\) depends smoothly on local density.

This is much more promising to me than another standalone KNN regressor.

---

# Tenth: I would resurrect GNNs — but only as *residual encoders*

This is one place where I would deliberately contradict the archive.

The archive is right that standalone small-data D-MPNNs performed badly. 

But that does **not** mean graph neural networks cannot help.

I would not train:

```text
graph → Ei
```

Instead:

```text
graph
 ↓
very small encoder
 ↓
32-dimensional residual representation
 ↓
Ridge / ET
 ↓
residual correction
```

The graph network isn't being asked to learn the entire property function from ~220 samples.

It only needs to learn:

> **what structural information is missing from the current descriptor model?**

And recent work specifically addresses graph consistency regularization under limited molecular labels, because naïve graph augmentation can destroy molecular semantics. ([ScienceDirect][5])

That's a substantially more defensible GNN experiment.

---

# Eleventh: do NOT use ordinary random SMILES augmentation

Recent graph/SSL research reinforces a point your own experiments discovered: arbitrary molecular augmentations can alter the semantics/property itself. ([ScienceDirect][5])

Instead, exploit only **representation invariants**:

```text
same graph
different traversal
different atom order
different valid SMILES
```

Then use:

$$
L=L_{property}
+\lambda L_{invariance}.
$$

This is especially attractive for the judged polymer-invariance criterion.

---

# Twelfth: the newest literature gives us an interesting multimodal direction

A 2026 molecular-property paper, MPMFMol, combines:

* graph;
* sequence;
* fingerprint;
* multitask self-supervision;

and explicitly uses fragment-aware augmentation rather than arbitrary perturbations. ([ACS Publications][6])

The obvious adaptation here is:

```text
SMILES encoder
       +
graph encoder
       +
handcrafted chemistry
       +
physics coordinates
```

but **late fusion**, not one huge concatenated vector.

That gives every target access to whichever representation carries its signal.

---

# Here is where I think we can genuinely make a breakthrough

## Stop optimizing the seven outputs separately.

Build:

# A chemistry state model

Something like:

```text
                         ┌── thermal state
                         │
SMILES ── chemistry ─────┼── electronic state
          encoder        │
                         ├── optical state
                         │
                         └── topology state
                                  │
                          target-specific
                              decoders
```

Where the latent state contains perhaps:

```text
rigidity
flexibility
polarity
conjugation
donor strength
acceptor strength
packing
polarizability
size
local density
```

But these don't need to be manually named.

They can be learned.

Then:

```text
thermal → Tg

electronic → Egc/Egb/Ei/Eea

optical → Nc/Eps
```

with cross-links.

This corresponds much more closely to the actual scientific structure of the problem than:

```text
seven independent regressors.
```

And it fits the literature: structured multitask learning and missing-label learning both point toward explicitly representing task relationships when labels are sparse. ([MI Research][1])

---

# And now the really important part: what I would NOT do

I would stop burning compute on:

**Another generic GNN.**

Already killed. 

**Another ordinary MLM.**

Multiple PI1M attempts failed. 

**Another random feature concatenation.**

You have already discovered that richer isn't automatically better.

**Another similarity threshold/router.**

Already failed badly.

**Another huge stacking architecture.**

Already suffered from transfer/circularity.

**Another 100 hyperparameter sweep.**

You don't have an optimization problem. You have an information problem.

---

# The five experiments I would run NOW

Not 100.

These five.

## Experiment 1 — Property-matrix latent model

Input:

```text
all canonical structures
×
7 targets
```

Architecture:

$$
X\rightarrow z
$$

$$
z,t\rightarrow \hat y_t
$$

with:

* missing-label masking;
* target embeddings;
* target graph;
* physics penalties;
* cross-target attention;
* target-specific heads.

**Kill:** if it cannot beat current grouped OOF by ~0.005 on small targets, stop.

---

## Experiment 2 — Global model + residual field

Build current best prediction.

Then learn:

$$
r(x)=y-\hat y.
$$

Use:

* chemical neighborhoods;
* graph topology;
* local density;
* physics descriptors.

Test whether nearby molecules have correlated residuals.

**Kill:** if residual autocorrelation is near zero.

If positive, deploy local residual correction.

This could be much more powerful than another absolute local model.

---

## Experiment 3 — 6M density/prototype geometry

Don't train another giant predictor yet.

Use `smile_r3` to create:

```text
chemical density
prototype
locality
OOD
manifold coordinates
```

Then ask:

$$
\text{validation error}
\sim
density + OOD + target.
$$

If it works, use it to:

* reweight training;
* choose model weights;
* choose local-vs-global correction;
* identify test regions requiring specialists.

---

## Experiment 4 — Ei electronic latent coordinate

Build only ~10–20 explicit electronic features.

Then compare:

```text
Ridge
Bayesian ridge
GAM
GP
ET
```

on:

$$
Ei-(Egc+Eea)
$$

but with the key change:

**fit in a low-dimensional electronic manifold, not hundreds of descriptors.**

This is where I'd aim for a genuinely large Ei gain.

---

## Experiment 5 — Eps/Nc joint latent model

Predict:

$$
(n_c,\epsilon_{ionic})
$$

jointly.

Then:

$$
\epsilon=n_c^2+\epsilon_{ionic}.
$$

Use:

* polarizability;
* polarizability/size;
* heteroatom topology;
* electronic latent;
* chemical density.

This is the target where the historical evidence says there is the most physical structure to exploit. 

---

# My revised belief about the 0.935 target

I am **less convinced now that the problem is "we need a magical new model."**

I think the problem is closer to:

```text
Current system
    ↓
captures ~90% of the useful low-order information
    ↓
remaining error is highly structured
    ↓
structure exists in:
    • missing labels
    • task relationships
    • local residuals
    • chemical geometry
    • electronic latent structure
    • optical decomposition
    • chemical-space density
```

That is why all the generic model experiments keep bouncing around ~0.90.

And there is strong external evidence for the exact kind of shift I'm recommending: polymer ML papers repeatedly find that feature-space representation, extrapolation geometry, and physically meaningful decomposition matter strongly; the 2024 Tg study specifically concluded that representation and training-data range affect extrapolation and that simpler models can be competitive with more complex nonlinear models. ([Wiley Online Library][7])

More recent Tg work combines **group-additive structure information with QSPR**, rather than relying on either alone, and reports markedly better results within its polymer family. ([ACS Publications][8]) A 2026 study likewise reports strong Tg prediction using molecular descriptors plus group information and explicitly uses SHAP to identify chemically meaningful drivers. ([ScienceDirect][9])

That is very consistent with the lesson from your own archive.

---

# One last, potentially huge idea

I would investigate this before almost anything else:

## Are the "seven targets" actually seven manifestations of a much smaller number of hidden chemical variables?

Do a **complete target covariance / partial-correlation / latent-factor analysis only on structures having multiple labels**.

Not just:

$$
Corr(y_i,y_j)
$$

but:

$$
Corr(y_i,y_j\mid X)
$$

and:

$$
Corr(r_i,r_j).
$$

If, for example, after controlling for descriptors you still find:

```text
Egc ↔ Ei       extremely strong
Egc ↔ Eea      extremely strong
Nc  ↔ Eps      extremely strong
```

then the real problem isn't seven-property prediction.

It's probably:

```text
              3–4 latent physical states
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       electronic    optical      thermal
          ↓            ↓            ↓
       4 targets     2 targets      Tg
```

And **that** is the model I would build.

The missing-label literature gives us exactly the conceptual basis for taking that direction: molecules have multiple correlated properties, missing labels are a major bottleneck, and explicitly modeling molecule–task relationships can recover useful supervision. ([MI Research][1])

---

## Bottom line

I don't think the next move is "try more techniques."

I think the gap is:

> **Your current system predicts properties. It does not yet explicitly model the hidden chemical state that generates all seven properties and the structured error left by your current model.**

So my priority order is now:

**1. molecule × property matrix / structured latent model**

**2. residual-field modeling**

**3. 6M chemical-space geometry, not merely 6M pretraining**

**4. Ei electronic latent model**

**5. joint Nc/ionic optical model**

**6. only then multimodal SSL/GNN residual models**

And I would use the existing oracle **only after freezing candidate predictions**, exactly as required by your project rules; the rules explicitly prohibit oracle values from entering training, feature construction, routing, calibration, or blend selection. 

I cannot honestly promise that these will produce 0.935—nothing in the literature supports such a guarantee—but **this is the first point where I think we're attacking the actual bottleneck rather than adding another layer of model zoo experimentation.**
