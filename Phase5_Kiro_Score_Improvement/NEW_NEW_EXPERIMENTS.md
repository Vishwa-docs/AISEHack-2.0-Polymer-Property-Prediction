Yes. The new evidence changes my recommendation materially.

I think the previous responses made one major mistake: they kept searching for **new model families**, when the strongest evidence now says the real opportunity is to **reconstruct the best cross-property + heterogeneous-ensemble pipeline much more faithfully, and then attack the exact failure mode of that pipeline**.

The competitor recipe you pasted is not just “another model.” It contains several mechanisms that line up almost perfectly with what this problem appears to reward.

Your own history also tells us exactly where we should stop being generic. The project archive records that cross-property covariates were one of the only robust cross-target wins, with the earlier implementation using partner labels on only about 60% of test rows; it also records that co-test joint solving worked, while careless cross-property stacking suffered a serious circularity failure. 

That is where I would concentrate now.

# What I think happened

The external 0.894 recipe has four ingredients:

1. **Aggressive cross-property transfer**
2. **A genuinely heterogeneous model zoo**
3. **PI1M used for pseudo-labeling**
4. **Convex OOF blending**

And your current line has some versions of all four, but not necessarily the **same statistical construction**.

The difference may be enormous.

The published polymer-informatics literature independently supports the first idea very strongly: the Ramprasad group showed that multi-task learning becomes especially useful when properties are correlated and the individual property datasets are small; they explicitly built a coalesced polymer × property matrix with missing entries and found that a selector-based multi-task architecture could outperform conventional single-task models. ([PubMed Central (PMC)][1])

And the recently published 2026 ADEPT-PolyGraphMT work is even closer to this competition: it evaluates joint polymer-property prediction, task grouping, shared graph encoders, task-specific heads, and performance as training data become scarce. ([Royal Society of Chemistry Publications][2])

So here is my revised position:

# Do not invent another 100 random ideas.

Build a **Cross-Property Transfer 2.0** program.

---

# 1. First question: are we actually reproducing the competitor's +0.036 mechanism?

This should be Experiment Zero.

The description you provided is:

> for every polymer, use the other six properties; true value where known, LGB estimate where missing, with a missingness flag.

That is **not equivalent** to simply saying "we use cross-property features."

The exact construction matters.

Suppose predicting `Ei`.

You need something like:

$$
X_{Ei}
=
[X_{chem},
Egc^{known/pred},
Egb^{known/pred},
Eea^{known/pred},
Nc^{known/pred},
Eps^{known/pred},
M_{Egc},
M_{Egb},
...
].
$$

But the critical part is how those missing partner values are generated.

## There are at least six possible versions

### A. In-sample partner fill

Wrong for OOF evaluation.

### B. Single global LGB fill

May be what you currently use.

### C. Fold-local cross-fitted fill

Much safer.

### D. Target-specific fill

Different model for predicting missing Egc versus Egb.

### E. Chained fill

For example:

```text
Egc
 ↓
Eea
 ↓
Ei
 ↓
Egb
 ↓
Nc
 ↓
Eps
```

### F. Joint iterative fill

```text
initial estimates
     ↓
predict all properties
     ↓
use predictions as features
     ↓
retrain
     ↓
repeat 2–5 times
```

I don't see enough evidence in your archive that **F has been properly explored with the exact test-time missingness pattern**.

That is my first major suspect.

---

# 2. The key new experiment: iterative cross-property imputation

I would build the entire problem as a chained imputation system.

For every target:

$$
y_t = f_t(X, y_{-t}).
$$

For missing partner labels:

$$
\hat y_j = g_j(X,y_{observed}).
$$

Then feed those predictions into the next target.

The architecture becomes:

```text
                    ┌──── Egc ────┐
                    ↓              │
SMILES → chemistry → Egb → ... → Ei
                    ↑              │
                    └──── Eea ─────┘
```

Use **cross-fitted predictions at every stage**.

Then do:

### Iteration 0

Chemistry-only.

### Iteration 1

Chemistry + partner predictions.

### Iteration 2

Chemistry + improved partner predictions.

### Iteration 3

Repeat.

Stop when validation stops improving.

This is essentially a **stacked missing-label multi-task system**.

The reason I consider it high priority is that the original Ramprasad work explicitly showed the benefit of coalescing sparse property datasets and exploiting the correlations among them. ([PubMed Central (PMC)][1])

---

# 3. More important: infer the optimal TARGET ORDER

Don't arbitrarily choose:

```text
Egc → Eea → Ei
```

Test every plausible directed ordering.

For six properties that is already a useful combinatorial search.

We want:

$$
P_1\rightarrow P_2\rightarrow P_3...
$$

where each step maximizes downstream transfer.

For example, one ordering might be:

```text
Egc
 ↓
Egb
 ↓
Eea
 ↓
Ei
 ↓
Nc
 ↓
Eps
```

while another could be:

```text
Egc
 ↓
Eea
 ↓
Ei

Nc
 ↓
Eps

Egb ← Egc
```

The graph should be **learned from the data**, subject to known physical edges.

This could be one of the largest remaining gains because you're converting correlation into a **directed information pipeline**, rather than merely concatenating six predictors.

---

# 4. The thing I now distrust most: the "~60% availability" number

Your project states approximately 60% test-time partner availability. 

Your external competitor claims **88–99%**.

That's not a small discrepancy.

That deserves immediate forensic investigation.

One of the two numbers may be measuring something different.

Possibilities:

* exact SMILES match;
* canonical SMILES match;
* target-row availability;
* any-partner availability;
* at-least-one-partner availability;
* availability conditional on the target;
* availability counting duplicate rows;
* availability after a secondary merge.

You need a table like:

| target | exact partner availability | canonical | any partner | ≥2 partners | ≥3 partners |
| ------ | -------------------------: | --------: | ----------: | ----------: | ----------: |
| Tg     |                            |           |             |             |             |
| Egc    |                            |           |             |             |             |
| Egb    |                            |           |             |             |             |
| Ei     |                            |           |             |             |             |
| Eea    |                            |           |             |             |             |
| Nc     |                            |           |             |             |             |
| Eps    |                            |           |             |             |             |

And for every test row:

```text
target
number of observed partners
which partners
number of predicted partners
chemical similarity
```

### I would actually bet that this discrepancy is important.

Because if the competitor truly had 88–99% availability and your implementation exposes only ~60%, **you may simply not be giving your model the same information.**

That is far more likely to produce a +0.03 jump than another exotic neural architecture.

---

# 5. The second huge gap: your archive does NOT say AttentiveFP/GINE were exhausted

I checked the experiment catalog carefully.

It records:

* directed MPNN;
* periodic graph;
* GIN masked-atom;
* SMILES Transformer;
* etc.

But it does **not** contain a proper AttentiveFP/GINE experiment corresponding to the recipe you posted. `GINE` only appears in unrelated context; there is no proper AttentiveFP baseline in the catalog. 

That matters.

You therefore should **not** conclude:

> "GNNs are dead."

The evidence actually says:

> "Several GNN formulations are dead."

The difference is important.

The 2026 ADEPT-PolyGraphMT paper is particularly relevant because it uses GINE/GIN/GCN shared graph encoders with target-specific heads, explores 3–7 layers and 256–768 dimensions, and finds that multi-task configurations can become particularly valuable in low-data settings. ([Royal Society of Chemistry Publications][2])

### So I would run a very specific GINE experiment.

Not a generic GNN.

```text
SMILES
  ↓
heavy-atom graph
  ↓
GINE
  ↓
shared representation
  ├── Egc
  ├── Egb
  ├── Eea
  ├── Ei
  ├── Nc
  └── Eps
```

And **feed cross-property features into the heads**.

That is much closer to the successful external recipe.

---

# 6. AttentiveFP should be tested separately

AttentiveFP is interesting because its attention mechanism can prioritize chemically relevant neighborhoods rather than treating every message equally.

Train:

```text
AttentiveFP × 3 seeds
```

with target-specific heads.

Then compare:

```text
AttentiveFP
GINE
classical ensemble
```

not only on average R², but on residual correlation.

The goal is not:

> "Does AttentiveFP beat Ridge?"

It is:

> "Does AttentiveFP make sufficiently different errors to improve the final blend?"

That's exactly how I would use it.

The 2025 Open Polymer Challenge provides useful precedent for property-specific graph architectures: one top solution used a GATv2 model plus fingerprint features, and another strong GNN solution explicitly specialized graph models by property. ([Kaggle][3])

---

# 7. The PI1M pseudo-label idea deserves to be resurrected

This is another place where the previous history may be misleading us.

Your archive says PI1M SSL attempts failed:

* char TF-IDF;
* PPMI;
* InfoNCE;
* MLM;
* subword;
* rarity;
* pseudo-label ranking/distillation;
* etc. 

But the competitor recipe isn't primarily saying:

> "PI1M embedding helps."

It says:

> **PI1M pseudo-labeled LightGBM.**

That's a different mechanism.

The pseudo-labels themselves become additional supervised samples.

That can work even if the representation isn't useful.

---

# 8. But do pseudo-labeling correctly

Do NOT:

```text
train → predict 1M → append all → retrain
```

Instead:

### Step 1

Train a teacher ensemble.

### Step 2

Predict PI1M.

### Step 3

Compute:

```text
teacher variance
distance to labeled manifold
prediction range
model agreement
```

### Step 4

Keep only the high-confidence region.

### Step 5

Weight pseudo-labels according to confidence.

For example:

$$
w_i = \frac{1}{\sigma_i^2+\epsilon}.
$$

### Step 6

Train student LightGBM.

### Step 7

Evaluate against **pseudo-label-free held-out validation**.

This is much more defensible than the earlier failed RankUp experiment.

---

# 9. And here is a new variant: pseudo-label only the residual

This is something I would strongly test.

Instead of generating:

$$
\hat y_{pseudo}
$$

generate:

$$
\hat r_{pseudo}
=
\hat y_{teacher}-\hat y_{baseline}.
$$

Then train the student on:

```text
baseline chemistry model
+
pseudo-labeled residual
```

Why?

Because the pseudo-label distribution of absolute targets can reinforce model bias.

Residual pseudo-labeling asks the large unlabeled set to learn:

> where does the teacher think the baseline should move?

That is a much lower-variance problem.

---

# 10. Another new one: **multi-teacher pseudo-labels**

Your competitor has 15 models.

Use them as teachers.

For each PI1M structure:

```text
LGB
XGB
Cat
ET
kNN
MLP
GINE
AttentiveFP
Transformer
...
```

Then calculate:

$$
\mu(x)=mean(predictions)
$$

$$
\sigma(x)=std(predictions).
$$

Only pseudo-label where:

$$
\sigma(x)<\tau.
$$

And weight by \(\sigma^{-2}\).

This turns the model zoo into a **pseudo-label confidence engine**.

---

# 11. Your current ensemble is probably missing a key idea: *ensemble family orthogonality*

The competitor specifically reported:

> within-family residual correlation 0.84–0.92, cross-family 0.55–0.76.

That is extremely informative.

Your archive says your classical Ridge/ET/Tanimoto ensemble is strong, but it also shows that model-zoo replacement collapsed. 

Those aren't contradictory.

The model zoo doesn't need to replace your parent.

It needs to produce:

$$
\text{different errors}.
$$

So create this exact matrix:

```text
             Ridge ET  LGB XGB Cat kNN GINE AFP MLP Transformer
Ridge
ET
LGB
XGB
...
```

of **OOF residual correlations**.

Then optimize the ensemble on:

$$
\min_w
\sum_i (y_i-X_iw)^2
+
\lambda\sum_{j<k}w_jw_k\rho_{jk}.
$$

This is a better formulation than simply maximizing individual component R².

---

# 12. More radical: use ensemble diversity as a *training target*

Train a model to predict:

$$
e_{best}-e_{alternative}.
$$

This tells you where an alternative model is likely to outperform the incumbent.

Then:

$$
\hat y =
w(x)\hat y_A +(1-w(x))\hat y_B.
$$

But \(w(x)\) is learned from OOF performance.

That's different from your old hard similarity router.

---

# 13. The competitor's blending method deserves an exact recreation

You currently have NNLS/compound assembly.

The external recipe says:

**convex SLSQP weights.**

These are not necessarily identical.

Compare:

### NNLS

$$
w_i\ge0
$$

but sum can vary.

### Simplex/SLSQP

$$
w_i\ge0,\quad\sum_i w_i=1.
$$

### Ridge-constrained simplex

$$
\min_w MSE+\lambda||w||^2.
$$

### Stability-constrained simplex

Require weights to remain similar across folds.

I would test all four.

The winning solution could be a surprisingly boring optimizer rather than a new model.

---

# 14. Even better: blend on **squared-error geometry**

Because R² is based on SSE, weight fitting should directly minimize:

$$
\sum_i(y_i-\sum_jw_j\hat y_{ij})^2.
$$

But do it **target-by-target and regime-by-regime**:

```text
all
near
medium-distance
far
```

Then regularize toward global weights.

---

# 15. The big hidden variable: the test set itself

There is a dangerous possibility:

Your validation strategy is still not approximating the actual test distribution.

The project explicitly documents an OOF-to-test gap such as:

* Ei ≈ −0.089;
* Eps ≈ −0.046;
* Nc ≈ −0.024. 

That's enormous.

It means a model that looks good locally can collapse on test-like chemistry.

This screams:

# adversarial validation is not just a diagnostic; it should drive model training.

Train:

$$
D(X)=P(test|X).
$$

Then use:

$$
w(x)=\frac{P(test|x)}
{1-P(test|x)}
$$

to weight OOF training.

But your older attempt at ordinary reweighting failed. 

So change the methodology:

### Instead of weighting ALL samples,

build **test-distribution-matched folds**.

For each target, select validation samples so their distribution matches test with respect to:

* Morgan similarity;
* chemical family;
* descriptor PCA;
* SMILES length;
* functional groups;
* partner availability.

This is closer to **importance sampling the validation process** than ordinary domain weighting.

---

# 16. A new experiment: nearest-neighbour *coverage matching*

For every test row compute:

$$
d_1,d_2,\ldots,d_{10}.
$$

Then find validation rows whose joint distance-distribution matches the test.

Train on the complement.

This creates a pseudo-private-validation set with similar support geometry.

I would trust this validation substantially more than ordinary random CV for model selection.

---

# 17. The MCP paper gave us something we completely overlooked

This is perhaps the most concrete **new feature family** I would add.

Zhang et al. introduced **Multi-Cover Persistence (MCP)** for polymer property prediction. Their representation uses Delaunay slices, rhomboid tiling and persistent barcodes, and they report competitive performance against ECFP, Polymer Genome, TransPolymer and geometric deep learning. In their benchmark, MCP achieved approximately:

| property |       R² |
| -------- | -------: |
| Eea      |     0.84 |
| Eps      | **0.71** |
| Nc       | **0.80** |
| Ei       | **0.78** |
| Egb      | **0.92** |
| Egc      | **0.90** |

with the exact values depending on dataset/configuration. ([PubMed Central (PMC)][4])

And the authors released the implementation on GitHub. ([GitHub][5])

[MCP GitHub repository](https://github.com/ZhangYipeng01/MCP?utm_source=chatgpt.com)

### Why I think this matters

Your incumbent already has:

* Morgan;
* Polymer Genome;
* Hückel;
* RDKit;
* polar descriptors.

But **it does not appear to have MCP**.

This is exactly the kind of representation I would want for the remaining error because it is not just "another fingerprint."

It captures **higher-order geometric/topological structure**.

And the published result is especially interesting for **Nc/Eps/Ei**, the same weak targets hurting your score. ([PubMed Central (PMC)][4])

### Experiment

Take the GitHub implementation as *source code only*.

Reimplement the features from scratch inside the notebook using competition data.

Then:

```text
C282 parent
+
MCP Ridge
+
MCP LightGBM
+
MCP ExtraTrees
```

and evaluate residual correlation.

This is now in my **top five**.

---

# 18. The other overlooked feature family: radical-marker geometry

The 4th-place Open Polymer Challenge solution explicitly improved a LightGBM model using features that encode the positional relationships of radical markers in polymer SMILES, because topology and local neighborhood affect polymer properties. ([Kaggle][6])

This is highly relevant.

Your project already has a sophisticated polymer representation, but I would still build a dedicated **endpoint/radical geometry representation**:

```text
distance * → *
atoms between *
bond types along * → *
aromatic atoms near *
heteroatoms near *
rings crossed
branch count along path
conjugation along radical path
```

Then feed these only to:

* Tg
* Egb
* Egc
* Ei
* Eea.

This is more specific than generic dimer/trimer descriptors.

---

# 19. Heavy-atom-only representation

The Open Polymer Challenge's 3rd-place solution reported an intriguing post-hoc result: simply removing hydrogen atoms from the graph improved the single-model performance to approximately their competitive public score. ([Kaggle][7])

For your problem I would test:

### Graph A

full atom graph

### Graph B

RemoveHs

### Graph C

RemoveHs + explicit H-bond descriptors

### Graph D

heavy atoms + implicit hydrogen count.

Then compare.

Why?

Because your targets are dominated by heavy-atom connectivity and electronic topology, while explicit hydrogens can increase graph complexity without proportional useful information.

---

# 20. Don't just build a GINE. Build a **dual graph**

One graph:

```text
atom graph
```

Another:

```text
backbone + side-chain graph
```

Then:

$$
z =
z_{atom}
+
z_{backbone}
+
z_{sidechain}.
$$

This is more aligned with polymer-specific models such as PolyMetriX, which explicitly supports full-polymer and backbone/side-chain featurization. ([GitHub][8])

---

# 21. The recent 2026 polymer paper gives us a very useful architecture blueprint

ADEPT-PolyGraphMT uses:

* shared GINE/GIN/GCN graph encoder;
* task-specific prediction heads;
* task grouping;
* target-specific architectures;
* optional feature-wise conditioning. ([Royal Society of Chemistry Publications][2])

So I would NOT build:

```text
one giant seven-output network
```

I'd build:

```text
                 GINE
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
    electronic optical    thermal
        │         │         │
 Egc Egb Ei Eea  Nc Eps     Tg
```

And then add **cross-property covariates only at the relevant heads**.

That is far more scientifically defensible than naïve multitask learning, and it is directly supported by current polymer ML research.

---

# 22. Physics/operator direction

The official AISEHack program explicitly frames the polymer track around **physics/operator-based deep learning**. ([Precog Research Group][9])

That matters.

I would therefore try one deliberately "competition theme aligned" model:

## Physics-conditioned neural operator

Not a full PDE solver.

Instead:

```text
SMILES
 ↓
structural encoder
 ↓
latent physical state
 ↓
operator block
 ↓
target vector
```

with the operator constrained by:

$$
Ei-Egc-Eea=0
$$

and

$$
Eps-Nc^2-\epsilon_{ionic}=0.
$$

Use Fourier features or a small spectral mixing block.

Recent operator-learning work shows that physics-informed operator regression can embed symmetry/conservation constraints directly into the learned operator, while newer materials work uses graph/operator hybrids for topology-aware response prediction. ([ScienceDirect][10])

This is **high risk**, but unlike another generic GNN, it directly matches the stated direction of the hackathon.

---

# 23. I would actually search for a hidden "property simulator"

The six DFT quantities may be downstream of fewer latent quantities.

The target identities already reveal some:

$$
Ei=E_{vac}-VBM
$$

$$
Eea=E_{vac}-CBM
$$

$$
Egc=E_{CBM}-E_{VBM}
$$

so conceptually the model could infer:

```text
vacuum reference
VBM
CBM
```

and derive the three/four observable outputs.

Similarly:

```text
electronic polarizability
+
ionic response
```

could explain optical properties.

That suggests a **latent physical coordinate model**:

```text
SMILES
  ↓
[VBM, CBM, vacuum-level]
  ↓
Egc / Ei / Eea
```

and:

```text
SMILES
 ↓
[electronic polarizability, ionic response]
 ↓
Nc / Eps
```

This is much better than trying to regress each target independently.

The reason I put this above ordinary multitask is that the latent variables have actual physical meaning.

---

# 24. Very important: test whether your "known physics" is actually exact on THIS data

Don't assume.

For each identity, calculate:

$$
r_1=Ei-Egc-Eea
$$

$$
r_2=Egb-(aEgc+b)
$$

$$
r_3=Eps-Nc^2-\epsilon_{ionic}.
$$

Then characterize:

* standard deviation;
* distribution;
* correlation with chemical family;
* correlation with target magnitude;
* outliers.

Your archive already knows the first and third identities are highly useful. 

The new question is whether the **residual itself is a function of chemistry**, and if so, *which chemistry*.

---

# 25. One thing I would NOT do anymore: generic residual learning around Ei/Eea

The existing evidence is unusually strong here.

The archive reports:

> `ei/eea` identity residual LOO R² = −0.82.

So don't waste time trying:

$$
Ei=Egc+Eea+XGB(X).
$$

The mechanism that worked was the identity **itself**.

Use the latent electronic representation instead.

---

# 26. But Egb is different

This one *did* support residual learning:

$$
Egb=aEgc+b+r(X)
$$

and the archive reports an improvement from about 0.9205 to 0.9478 with an ExtraTrees residual. 

So for Egb I would now do:

```text
Egc
 ↓
affine Egb
 ↓
MCP residual
+
radical-marker residual
+
GINE residual
+
electronic descriptors
```

That could plausibly push Egb toward very high R².

And because Egb is already ~0.93, even +0.02 here directly contributes to the total average.

---

# 27. The other huge missed opportunity: **ensemble member selection conditioned on target availability**

Imagine a test molecule where:

```text
Egc observed
Eea observed
Ei target
```

That row should use a different ensemble from:

```text
Egc missing
Eea missing
Ei target
```

Yet most final pipelines tend to use one global blend per target.

Instead define:

$$
w_{t,a}
$$

where \(a\) is the availability pattern.

For example:

```text
Ei:
000011
000101
001101
011111
...
```

There are only limited meaningful patterns.

For each pattern, train a separate convex blend using validation rows with the **same observed partner pattern**.

This is extremely close to the competitor's cross-property strategy.

I think this has much more promise than another 50-model global blend.

---

# 28. Go further: include NUMBER of partner labels

Add:

$$
N_{partners}
$$

as a feature.

Also:

$$
\sum_j M_j
$$

and target-specific indicator vectors.

Then allow:

$$
prediction =
f(X,\text{observed partner values},\text{pattern}).
$$

This gives the model information about **how much uncertainty should remain**.

---

# 29. Test the external recipe's exact 15-model portfolio

I would actually recreate it as closely as rules permit:

### Family 1

* LGB
* XGB
* Cat
* ET

### Family 2

* Tanimoto kNN

### Family 3

* PI1M pseudo-LGB

### Family 4

* multi-task MLP

### Family 5

* AttentiveFP × 2
* GINE × 2

### Family 6

* SMILES Transformer.

Then:

**do not optimize the individual models endlessly.**

Train them, generate OOF residuals, and ask:

> does this portfolio improve the incumbent when blended?

That is the fair comparison against the 0.894 recipe.

And because your project already has an established strong classical parent, I'd use:

$$
C_{parent}
+
\{15\ models\}.
$$

Not replacement.

---

# 30. The competitor's PI1M Transformer may still matter—but differently

TransPolymer is genuine evidence that masked-language pretraining on large unlabeled polymer corpora can help polymer property prediction; the original paper pretrained on roughly 5M augmented polymer sequences and then fine-tuned the model. ([Nature][11])

But your earlier PI1M MLM probe failed.

So I would change two things:

### First

Use **all 6M `smile_r3`**, not only PI1M.

### Second

Do **multi-property fine-tuning**, rather than a weak linear probe.

And make the output head:

```text
embedding
+
handcrafted features
+
partner features
```

rather than embedding-only.

That gives the sequence model a route to contribute information the tree model doesn't have.

---

# 31. A potentially enormous experiment: 6M supervised-by-teacher *property fields*

This is more novel.

Teacher:

$$
T(X)\rightarrow
(Egc,Egb,Eea,Ei,Nc,Eps,Tg)
$$

Run over the 6M corpus.

Now you have an unlabeled chemical universe with **synthetic property vectors**.

Rather than treating pseudo-labels independently, learn the **joint manifold**:

$$
X\rightarrow \hat Y_{teacher}.
$$

Train a student on:

```text
SMILES
→ teacher property vector
```

with a physics loss.

Then fine-tune on real labels.

The important piece is that the student is learning **joint property geometry**, not merely one pseudo-label.

This is essentially a teacher-generated multitask pretraining corpus.

I think this is more promising than the earlier RankUp-style distillation, because it retains **all seven outputs simultaneously**.

---

# 32. But there is an even better version

Don't distill the teacher's absolute prediction.

Distill:

$$
T(X)-B(X)
$$

where \(B\) is the classical parent.

Now the student learns the **teacher's additional nonlinear correction**.

That is likely to be much less biased.

---

# 33. Use model disagreement to select pseudo-labels

A molecule is pseudo-label-worthy when:

```text
GINE ≈ AttentiveFP ≈ LGB ≈ XGB ≈ Ridge
```

and the chemistry is inside the known support.

That's much stronger evidence than one model being confident.

---

# 34. Use 6M as a *negative* dataset too

This is novel and underexplored.

Find chemical structures that are:

```text
close to test
but far from labeled train
```

and include them in the **domain discriminator**.

This tells the model what extrapolation actually looks like.

Then train the predictor with an OOD-aware objective.

---

# 35. One thing I think may be seriously underestimated: test-set duplication structure

Your test has 4,940 rows but only 4,497 unique structures. 

I would build the final prediction at the **unique-structure level**.

Predict each canonical molecule once.

Then map predictions back to rows.

This prevents tiny numerical differences between representations of identical structures.

More importantly:

### Analyze test duplicates across target types.

The same polymer may appear multiple times because different rows represent different target properties.

That creates a **cross-property relational structure in the test itself**.

Do not treat those rows as unrelated.

---

# 36. Build a test-polygon

For each unique test polymer:

```text
vertices = available target labels
edges = physical target relations
```

Then solve the unknown values jointly.

For example:

```text
Egc known
Eea known
Ei unknown
```

gives a nearly deterministic constraint.

But if:

```text
Egc predicted
Eea known
Ei unknown
```

weaken the constraint according to Egc uncertainty.

This naturally creates **uncertainty-aware physics fusion**.

---

# 37. Probabilistic cross-property inference

For each target, estimate:

$$
p(y_t\mid X,y_{observed}).
$$

Then propagate distributions rather than point estimates.

For example:

$$
E_i \sim p(E_i\mid E_{gc},E_{ea},X).
$$

Even a simple Gaussian approximation can help.

This is the principled form of the competitor's "true if available, LGB estimate otherwise."

---

# 38. A powerful variant: mixture of exact-partner and predicted-partner modes

For each target:

```text
Mode 1:
all relevant partners observed

Mode 2:
some observed, some estimated

Mode 3:
no partners
```

Train separate models for the three regimes.

Then blend.

This is likely better than giving one model a missingness flag.

---

# 39. The competition's actual objective makes this especially attractive

The score is:

$$
\frac1{7}\sum_tR_t^2.
$$

So each target is worth exactly 1/7.

That means **small-target breakthroughs are disproportionately valuable**.

A gain from:

$$
0.87\rightarrow0.92
$$

on Ei contributes ~0.0071 to the overall score.

Do that on four targets and you've gained ~0.028.

That is why I would now aggressively sacrifice generic-model elegance for target-specific systems.

Your target counts make that mathematically justified. 

---

# 40. My new target attack map

## Ei

**Highest priority.**

Use:

1. exact partner values;
2. uncertainty-aware partner predictions;
3. physical identity;
4. donor/acceptor topology;
5. Hückel features;
6. MCP;
7. GINE residual model;
8. pseudo-labeled 6M teacher;
9. availability-specific ensemble.

Target:

**0.92–0.94.**

---

## Eps

Also highest priority.

Use:

1. ionic decomposition;
2. polar group features;
3. joint Nc/ionic model;
4. MCP;
5. GINE;
6. availability-specific partner inference;
7. teacher/pseudo-label field;
8. physical-consistency ensemble.

Target:

**0.93–0.95.**

---

## Nc

Use:

1. polarizability;
2. ionic coupling;
3. MCP;
4. radial/graph topology;
5. attention model;
6. partner availability;
7. physics-constrained ensemble.

Target:

**0.93–0.95.**

---

## Egb

Use:

1. Egc;
2. affine relationship;
3. ExtraTrees residual;
4. MCP residual;
5. GINE residual;
6. radical-marker geometry.

This one is probably the easiest target for a material improvement because the archive already proves its structural residual is learnable. 

---

## Eea

Use:

1. Ei/Egc partner information;
2. Flory–Fox / oligomer carrier;
3. Hückel;
4. donor/acceptor topology;
5. MCP;
6. AttentiveFP/GINE.

---

## Egc

Don't over-invest.

It is already reasonably strong.

Use it primarily as a **teacher / upstream property**.

---

## Tg

Use:

1. radical-marker geometry;
2. hierarchical backbone/sidechain representation;
3. conformational/rotational features;
4. test-distribution correction;
5. property-specific model zoo;
6. robust target calibration.

Again, don't spend 70% of the research budget here.

---

# 41. One completely new idea: **teacher properties as privileged information**

This is a very important distinction.

During **training only**, a model can see all target labels associated with the same training molecule.

At test time it cannot.

Instead of using those labels directly, learn a representation:

$$
z_{privileged}=f(X,Y_{available}^{train}).
$$

Then train a student:

$$
z_{student}=g(X)
$$

to reproduce that representation.

Finally predict targets from \(z_{student}\).

This is a form of **learning using privileged information (LUPI)**.

The training labels teach the model the latent structure of the polymer-property system without being directly available for new test molecules.

This could be extremely interesting here.

---

# 42. Even stronger: privileged-information distillation by target

For Ei:

```text
teacher:
SMILES + Egc + Eea + Egb + ...
```

Student:

```text
SMILES only
```

But train the student to reproduce the teacher's internal representation.

At test time, use student.

This avoids circularity at inference while exploiting cross-property structure during learning.

---

# 43. Another new one: counterfactual property training

Take the same polymer and mask different partner properties:

```text
[chemistry, Egc, Eea]
[chemistry, Egc]
[chemistry, Eea]
[chemistry]
```

Train the model to predict Ei under all four contexts.

This teaches it:

> what information is valuable, and how to degrade gracefully when information disappears.

Then the test-time missingness pattern is no longer distributionally novel.

I think this is substantially better than simply using missingness indicators.

---

# 44. Another new one: partner dropout

During training, randomly hide otherwise available target labels.

For example:

```text
50% Egc availability
50% Eea availability
20% Egb availability
...
```

Choose dropout rates to mimic the **actual test availability pattern**.

Then the model learns cross-property transfer under realistic missingness.

This is exactly the sort of thing I would expect a competition solution to benefit from.

---

# 45. One more extremely promising idea: **masked target prediction**

Don't train the network only to predict the real target.

Randomly mask one of the seven properties and ask the network to reconstruct it.

That creates:

```text
polymer
+
subset of properties
→ missing property
```

training examples from every multi-label polymer.

This is a much more direct self-supervised objective than SMILES MLM.

It is **property-space self-supervision**.

And it directly matches the competition structure.

---

# 46. Why this may be better than SMILES MLM

Your problem's scarce resource is not molecular syntax.

It's **property relationships**.

Your dataset itself already contains those relationships.

So instead of:

```text
SMILES → next token
```

learn:

```text
SMILES + 3 properties → other 4 properties.
```

That's a much more relevant pretraining task.

---

# 47. Property masking can be done purely from official data

No external information.

And the model still trains entirely from scratch.

So it appears compatible with the rules you've documented, provided the implementation doesn't leak held-out validation labels.

The rules explicitly allow from-scratch representation learning on the official auxiliary datasets but prohibit pretrained models and external information. 

---

# 48. A completely different route: multi-fidelity thinking without external data

The 2026 ADEPT paper emphasizes multi-fidelity learning because polymer properties can arise from different sources/fidelities, and reports that fidelity-aware learning helps when data are scarce. ([Royal Society of Chemistry Publications][2])

Your competition contains an analogous structure:

* experimental Tg;
* DFT electronic/optical properties.

Don't force Tg into the same objective as DFT properties.

Instead:

```text
shared chemistry
     │
 ┌───┴────┐
 Tg branch   DFT branch
```

That is a much more scientifically sensible multitask network.

Then the DFT branch can share a latent representation across six targets.

---

# 49. This suggests a better architecture than our previous one

```text
                         SMILES
                           │
                ┌──────────┴──────────┐
                │                     │
           chemistry encoder      polymer encoder
                │                     │
                └──────────┬──────────┘
                           │
                     shared latent
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
      thermal          electronic          optical
         │                 │                  │
        Tg           Egc Egb Ei Eea       Nc Eps
```

Then append:

```text
observed partner labels
partner masks
uncertainties
physics coordinates
```

only inside the relevant heads.

This is essentially the modern multi-task polymer architecture suggested by current literature, adapted to your exact missing-label setting. ([Royal Society of Chemistry Publications][2])

---

# 50. One last thing: don't ignore the published competition archaeology

The Open Polymer Challenge has extraordinarily useful post-competition evidence.

The official winning solution says the major gains came from:

* **distribution-shift post-processing of Tg**
* pseudolabeled PI1M
* target-specific models
* AutoGluon/ensemble approaches. ([Kaggle][3])

The official 4th-place solution shows that **radical topology features** can improve LightGBM. ([Kaggle][6])

The official 3rd-place solution shows property-specific GNNs can work and that preprocessing such as removing hydrogens materially changed performance. ([Kaggle][7])

The 2024 MCP paper introduces another genuinely different polymer representation and reports particularly strong results on several of your weak properties. ([PubMed Central (PMC)][4])

And the 2026 PolyGraphMT work is remarkably close to your six DFT tasks and supports shared GINE/GIN representations with task-specific heads under low-data conditions. ([Royal Society of Chemistry Publications][2])

That is a much more useful research foundation than simply trying another generic GNN.

---

# The actual battle plan

If I were taking control of the remaining research, I would do this in exactly this order.

### Wave 1 — forensic replication

**Do not run another "novel" experiment yet.**

Reconstruct the 0.894 competitor methodology as literally as possible.

Measure:

```text
partner availability
partner pattern frequencies
partner fill accuracy
15-model OOF scores
cross-family error correlation
convex blend
```

The biggest question is:

> **Why does their cross-property mechanism apparently generate +0.036 while our cross-property mechanism is much smaller?**

That is the question worth answering.

---

### Wave 2 — Cross-property 2.0

Run:

```text
iterative partner imputation
+
partner dropout
+
masked-property reconstruction
+
availability-specific models
+
availability-specific ensemble
```

This is where I would expect the next genuine leap, if one exists.

---

### Wave 3 — new representation injection

Add **MCP**.

Then:

```text
parent
+
MCP
+
AttentiveFP
+
GINE
+
radical-marker topology
```

Do not replace the parent.

Blend them.

The MCP result is sufficiently targeted to your weak properties that I consider it one of the best-backed genuinely new candidates. ([PubMed Central (PMC)][4])

---

### Wave 4 — PI1M pseudo-label field

Not generic MLM.

Do:

```text
15-model teacher
→ 1M pseudo property vectors
→ uncertainty filtering
→ confidence weighting
→ LightGBM student
```

then the **same thing on 6M `smile_r3`**.

---

### Wave 5 — property-space SSL

Train:

```text
mask target
predict target
```

and:

```text
mask partner
predict partner
```

from the official multi-property data.

This is the SSL objective I now like more than ordinary SMILES MLM.

---

### Wave 6 — physics latent model

Build:

```text
VBM / CBM / vacuum-like latent
electronic polarizability
ionic response
```

then derive the targets.

Use this primarily for:

```text
Egc
Egb
Ei
Eea
Nc
Eps
```

---

# What I think the required +0.032 might look like

I would now aim for something along these lines:

| Target | Current approx. | Attack target |
| ------ | --------------: | ------------: |
| Tg     |            .897 |     .915–.925 |
| Egc    |            .912 |     .920–.925 |
| Egb    |            .931 |     .945–.955 |
| Ei     |            .871 |     .920–.940 |
| Eea    |            .918 |     .925–.935 |
| Nc     |            .908 |     .935–.950 |
| Eps    |            .887 |     .935–.955 |

I'm **not claiming those are achievable**. They are the *research targets* I'd set.

The interesting thing is that you don't actually need a miracle on every target.

For example, something like:

```text
Tg   +0.018
Egc  +0.008
Egb  +0.015
Ei   +0.050
Eea  +0.010
Nc   +0.030
Eps  +0.050
----------------
sum  +0.181
```

already adds ~0.026 to the seven-target mean.

Then ensemble selection and a couple of smaller gains could potentially close the remaining gap.

---

# And I want to be very explicit about where I would put the bets

## Highest confidence

**Exact reconstruction of the competitor's cross-property pipeline.**

Because you have direct evidence from a public competitor claiming +0.036 and because the published polymer literature strongly supports exploiting sparse property correlations. ([PubMed Central (PMC)][1])

## Highest upside

**Property-space masked reconstruction + iterative partner imputation.**

Because this uses the structure of *your actual dataset* rather than importing assumptions from generic molecular ML.

## Highest-value new representation

**MCP.**

Because it is genuinely different from your existing fingerprints and has published results specifically on Eea/Eps/Nc/Ei/Egb/Egc. ([PubMed Central (PMC)][4])

## Highest-value new model family

**Task-grouped shared GINE / AttentiveFP.**

Not standalone small-data GNNs—the exact property-grouped architecture is what the current polymer literature supports. ([Royal Society of Chemistry Publications][2])

## Highest-value auxiliary-data bet

**Multi-teacher pseudo-property learning on PI1M + `smile_r3`.**

Not generic SSL.

---

# The most important conclusion

I no longer think the main enemy is your current 0.904 model.

The enemy is that **you have not yet turned the problem into the form in which the strongest external evidence says it wants to be solved**:

$$
\boxed{
\text{polymer}
+
\text{partially observed property vector}
+
\text{chemical representation}
+
\text{physics}
}
$$

rather than just:

$$
\text{polymer}\rightarrow\text{target}.
$$

And the competitor's recipe you supplied is the strongest clue we've had so far.

I would therefore make the **very next experiment a forensic reconstruction of their cross-property + 15-model recipe**, with an explicit comparison against your current V57 parent. Not another model. Not another generic SSL experiment.

That experiment should answer four things:

1. **Is their 88–99% partner availability genuinely present in this dataset?**
2. **Does their exact partner-imputation construction reproduce their claimed mechanism?**
3. **Which of their model families are actually additive to our parent?**
4. **Does simplex/SLSQP blending produce an improvement that our current NNLS assembly misses?**

That is the most direct path I currently see to finding the missing ~0.032.

And unlike the earlier lists, every major branch above now has either **direct evidence from your own experiment archive**, **a publicly documented competition solution**, or **published polymer-ML evidence** behind it.  

[Ramprasad multi-task polymer learning GitHub](https://github.com/Ramprasad-Group/multi-task-learning?utm_source=chatgpt.com)
[MCP polymer-property GitHub](https://github.com/ZhangYipeng01/MCP?utm_source=chatgpt.com)
[PolyMetriX GitHub](https://github.com/lamalab-org/PolyMetriX?utm_source=chatgpt.com)
[TransPolymer paper](https://www.nature.com/articles/s41524-023-01016-5?utm_source=chatgpt.com)
[Open Polymer Challenge 1st-place writeup](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/1st-place-solution?utm_source=chatgpt.com)
[Open Polymer Challenge 3rd-place writeup](https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/3rd-place-solution?utm_source=chatgpt.com)
[Open Polymer Challenge 4th-place writeup](https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/4th-place-solution-lightgbm-with-smiles-derived-fe?utm_source=chatgpt.com)

I would make **the cross-property availability/imputation audit + exact 15-model reproduction the immediate priority**. That is now the most evidence-backed place to hunt the missing score.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8085610/?utm_source=chatgpt.com "Polymer informatics with multi-task learning - PMC"
[2]: https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00206d?utm_source=chatgpt.com "ADEPT-PolyGraphMT: automated molecular simulation and multi-task multi-fidelity machine learning for polymer property generation and prediction - Digital Discovery (RSC Publishing) DOI:10.1039/D6DD00206D"
[3]: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/1st-place-solution?utm_source=chatgpt.com "1st Place Solution | Kaggle"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11424509/?utm_source=chatgpt.com "Multi-Cover Persistence (MCP)-based machine learning for polymer property prediction - PMC"
[5]: https://github.com/ZhangYipeng01/MCP "GitHub - ZhangYipeng01/MCP: MCP-based machine learning for polymer property prediction · GitHub"
[6]: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/4th-place-solution-lightgbm-with-smiles-derived-fe?utm_source=chatgpt.com "4th Place Solution — LightGBM with SMILES-Derived Features | Kaggle"
[7]: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/3rd-place-solution?utm_source=chatgpt.com "3rd Place Solution | Kaggle"
[8]: https://github.com/lamalab-org/PolyMetriX?utm_source=chatgpt.com "GitHub - lamalab-org/PolyMetriX: PolyMetriX is a comprehensive Python library that powers the entire machine learning workflow for polymer informatics. · GitHub"
[9]: https://precog.iiit.ac.in/aisehack?utm_source=chatgpt.com "AISEHack | ANRF's AI for Science & Engineering Hackathon — Polymers & Space"
[10]: https://www.sciencedirect.com/science/article/abs/pii/S004578252030685X?utm_source=chatgpt.com "A physics-informed operator regression framework for extracting data-driven continuum models - ScienceDirect"
[11]: https://www.nature.com/articles/s41524-023-01016-5 "TransPolymer: a Transformer-based language model for polymer property predictions | npj Computational Materials"

You're right. I was still answering the wrong question.

You are not asking me for another collection of “interesting ML ideas.” You are asking me to **search the solution space hard enough that we can construct a substantially different end-to-end attack, with enough independent levers that the accumulated gain can plausibly cross 0.935**.

I also want to correct one thing explicitly: I cannot honestly write that 100 experiments “have to” get you to 0.935. No paper, Kaggle notebook, or experiment can guarantee that on an unseen private leaderboard. What I *can* do is give you 100 **new, non-duplicate attack candidates**, rank the ones I think have the most asymmetric upside, and show you how I would chain them instead of testing them as isolated models.

Your current situation is unusually favorable for that kind of approach because the problem has:

* 7,409 train rows but only ~221–337 labels for five of the six DFT/optical targets;
* 5.97M new unlabeled SMILES plus 995k PI1M;
* 457 exact train/test SMILES overlaps;
* strong known relationships among targets;
* a large amount of prior experimentation that lets us eliminate many families;
* a current ceiling around 0.904;
* and a demonstrated external ecosystem of successful polymer-competition solutions using heterogeneous models, pseudo-labeling, graph models, domain-specific features, and unusual post-processing.  

The external evidence is also now more interesting than when we started. PolymerGNN explicitly separates different monomer sets and learns multitask representations; the 2024 polymer SSL paper reports sizable low-data gains for electron affinity and ionization prediction; 2026 Polymer-JEPA reports benefit from predictive SSL; MCP offers a genuinely different topological/geometric representation; and recent work on multimodal polymer prediction uses graph + fingerprint + geometry + sequence representations. ([Nature][1])

And the 2025 Open Polymer Challenge postmortems are particularly relevant: first place used an ensemble of BERT/AutoGluon/Uni-Mol plus PI1M pseudo-label pretraining and Tg distribution-shift correction; third place used GATv2+fingerprints; fourth place used radical-topology features, 3D descriptors and target-specific quantile models; fifth place explicitly built a hierarchical physics→ML pipeline. ([Kaggle][2])

So here is the much more aggressive list.

---

# 100 additional experiments

These are **beyond the ideas already present in your previous 255-ish experiment/idea inventory**. I am deliberately including combinations and algorithmic variants that I think were missing.

---

## A. DATA-GEOMETRY ATTACKS

### 1. Reverse nearest-neighbor density

For each training point, count how many other points regard it as one of their nearest neighbors.

Use:

$$
RNN_i=\#\{j:i\in NN_k(j)\}.
$$

High-RNN points are chemical prototypes; low-RNN points are isolated.

Use RNN density in weighting.

**Targets:** all, especially Ei/Eea/Nc/Eps.

---

### 2. Local reachability density

Use a LOF-style density measure in chemical space.

The important variable becomes:

$$
density(x)
$$

rather than nearest-neighbor Tanimoto alone.

Use it to continuously control local/global model weight.

---

### 3. Mahalanobis chemical distance

Morgan similarity is binary and topology-specific.

Build a standardized descriptor covariance matrix and calculate:

$$
d_M(x,\mu).
$$

Compare its relationship with OOF error.

---

### 4. Robust Mahalanobis distance

Estimate the covariance using Minimum Covariance Determinant rather than sample covariance.

Useful for weird chemistry/outliers.

---

### 5. Local covariance distance

For each chemical cluster, calculate distance using that cluster's covariance instead of the global covariance.

This can distinguish "far from the whole dataset" from "normal for its family."

---

### 6. Density-ratio residual correction

Model:

$$
e(x)=f(\rho_{train}(x),\rho_{test}(x)).
$$

Do not change the predictor yet.

First establish whether error itself changes systematically with support density.

---

### 7. Nearest-neighbor *target slope*

For each training point calculate:

$$
\frac{|y_i-y_j|}{d(X_i,X_j)}
$$

for its nearest neighbors.

This estimates local target roughness.

Then ask which test points lie in rough regions.

---

### 8. Local Lipschitz score

Estimate:

$$
L_i=\max_j\frac{|y_i-y_j|}{d_{ij}+\epsilon}.
$$

High \(L_i\) means a sharp structure-property region.

Train a specialist specifically on high-Lipschitz regions.

---

### 9. Curvature of chemical-property manifold

Using nearest-neighbor triples, estimate whether the target bends locally.

Flat neighborhoods → local linear model.

High-curvature neighborhoods → nonlinear specialist.

---

### 10. Density × curvature model selection

Create a two-dimensional regime:

```text
high density / low curvature
high density / high curvature
low density / low curvature
low density / high curvature
```

Use different ensembles in the four quadrants.

---

## B. CHEMICAL-CONSTRUCTION FEATURES

### 11. Reaction-center-like attachment environment

For the two polymer attachment atoms, extract the complete radius-3 environments.

Separate them from ordinary Morgan fingerprints.

---

### 12. Attachment asymmetry

For a polymer repeat unit, compare the two attachment environments.

Features:

$$
|\phi(left)-\phi(right)|.
$$

This could identify asymmetric repeating units.

**Targets:** Tg, Egc, Egb.

---

### 13. Attachment-distance symmetry

Measure whether the same functional group is equally distant from both polymerization endpoints.

---

### 14. Backbone shortest-path descriptor

Find the shortest graph path between the attachment points and describe every atom/bond along it.

This is much more polymer-specific than whole-molecule fingerprints.

---

### 15. Backbone path aromaticity

Calculate aromatic fraction *only along the polymerization path*.

---

### 16. Backbone heteroatom ordering

Encode sequence patterns such as:

```text
C-C-O-C-C-N
```

along the backbone.

Not whole-SMILES n-grams.

---

### 17. Backbone bond-order profile

Encode the ordered series of bond types between attachment points.

---

### 18. Backbone torsional profile

For every consecutive backbone bond, characterize local torsional flexibility.

---

### 19. Side-chain radial shells

Count chemical groups at graph distance 1, 2, 3, … from backbone atoms.

This creates a structural "radial chemistry" representation.

---

### 20. Functional-group radial distribution

For every functional group:

$$
N_g(d)
$$

for graph distance from backbone.

This may be particularly useful for Tg and Eps.

---

## C. GROUP-CONTRIBUTION / QSPR HYBRIDS

This is an area where the latest Tg literature is especially encouraging: a 2025 QSPR–GAP study found that combining group-additive information with QSPR materially improved Tg on a polymer family, rather than relying on either alone. ([ACS Publications][3])

### 21. Learned group-contribution Tg

Instead of fixed literature coefficients:

$$
Tg=\sum_g c_gN_g+b
$$

learn \(c_g\) from your train data with strong regularization.

---

### 22. Group-contribution + nonlinear residual

Then:

$$
Tg=Tg_{GAP}+f(X).
$$

But use a small residual model.

---

### 23. Target-specific group contribution for Egb

Learn contributions of:

* aromaticity;
* heteroatoms;
* conjugated motifs;
* electron-withdrawing groups.

---

### 24. Group contribution for Eea

Build a donor/acceptor motif inventory.

---

### 25. Group contribution for Ei

Use the same motif inventory but fit a separate coefficient system.

---

### 26. Pairwise group contribution

Instead of:

$$
\sum c_gN_g
$$

include only selected:

$$
c_{g,h}N_gN_h.
$$

---

### 27. Group-contribution interaction selection by bootstrap

Keep only motif interactions that recur across many resamples.

---

### 28. GAP/QSPR mixture-of-experts

One model learns additive effects.

One learns nonlinear QSPR.

A low-capacity gate chooses the blend.

---

### 29. Family-specific group contributions

Use separate coefficient sets for major chemical families, with shared shrinkage toward global coefficients.

---

### 30. Hierarchical Bayesian group contributions

Global motif coefficient:

$$
c_g
$$

plus family deviation:

$$
c_{g,f}=c_g+\delta_{g,f}.
$$

This is especially promising for Tg.

---

## D. ELECTRONIC STRUCTURE ATTACK

Your existing Hückel work was strong, but this does not mean electronic structure is exhausted. The archive says Hückel-derived features had raw correlations around −0.79 with Eea and −0.72 with Ei. 

### 31. Hückel eigenvalue distribution

Don't only use HOMO/LUMO proxies.

Use:

* top 5 eigenvalues;
* bottom 5;
* eigenvalue gaps;
* spectral variance.

---

### 32. Hückel spectral entropy

Calculate entropy of normalized electronic-state eigenvalue weights.

---

### 33. Hückel state localization proxy

Measure whether frontier states are localized or delocalized over the graph.

---

### 34. Frontier-state backbone fraction

Estimate how much of the frontier eigenvector lies on backbone atoms.

---

### 35. Frontier-state heteroatom fraction

Analogously calculate the weight on heteroatoms.

---

### 36. Frontier-state attachment fraction

How much frontier-state weight lies near the polymerization sites?

---

### 37. Hückel topology ensemble

Run multiple chemically reasonable Hückel parameterizations and use:

```text
mean eigenvalues
variance
spread
```

as features.

The variance itself could be predictive.

---

### 38. EHT–Hückel disagreement

You already have EHT experiments.

Instead of choosing one:

$$
\Delta_{EHT-Huckel}
$$

may tell you where simple electronic approximations break.

---

### 39. Electronic-model disagreement as uncertainty

Use disagreement between:

* Hückel;
* EHT;
* classical descriptors;
* learned graph model.

---

### 40. Electronic specialist on only high-conjugation polymers

Don't force one electronic function over all structures.

Build a conjugated-family specialist.

---

## E. GEOMETRY / 3D

Recent multimodal polymer work explicitly combines graph, geometry and fingerprint modalities, while the 2025 competition's 4th-place solution found 3D descriptors complementary to topology. ([Nature][4])

### 41. Conformer ensemble mean descriptors

Generate multiple conformers and average:

* radius of gyration;
* inertia;
* surface area;
* volume.

---

### 42. Conformer descriptor variance

Use variation across conformers as a flexibility signal.

---

### 43. Conformer energy statistics

Use:

* minimum energy;
* mean energy;
* energy range;
* energy entropy.

---

### 44. Shape anisotropy

Use principal moments to estimate anisotropy.

Especially interesting for Tg/Rg-like structural behavior.

---

### 45. Asphericity

Use:

$$
A=\frac{\sum (I_i-I_j)^2}{...}
$$

as a compact shape descriptor.

---

### 46. Planarity statistics

Measure fraction of atoms approximately coplanar within aromatic/conjugated regions.

---

### 47. Surface polarity distribution

Don't just use total TPSA.

Measure polar atoms' spatial clustering over a conformer.

---

### 48. 3D donor-acceptor distance

For each donor-acceptor pair, calculate minimum 3D distance.

---

### 49. 3D aromatic stacking proxy

Approximate parallel aromatic-ring distances/orientation from conformers.

Potentially relevant to packing-sensitive properties.

---

### 50. 3D attachment orientation

Measure angles between attachment vectors and local backbone plane.

This may affect polymer conformation and optical/electronic structure.

---

# F. NEW SELF-SUPERVISED LEARNING

This is where I would substantially change course.

Your old SSL attempts were mostly:

* small subsets;
* generic objectives;
* weak downstream probes.

The literature has now moved toward **predictive SSL**, particularly JEPA-style objectives. Polymer-JEPA reports improved downstream performance in scarce-data settings, and M-JEPA's 2026 molecular work found predictive SSL superior to compute-matched InfoNCE in its evaluation. ([Royal Society of Chemistry Publications][5])

### 51. Polymer-JEPA on 6M

Train a JEPA-style encoder from scratch on `smile_r3`.

No pretrained model.

---

### 52. Graph-JEPA with subgraph masking

Predict the representation of a masked connected subgraph from its context.

---

### 53. JEPA with polymer attachment-aware masks

Always include attachment environments in selected masked regions.

---

### 54. JEPA with chemically stratified masks

Mask:

* aromatic regions;
* heteroatom regions;
* backbone;
* side chain.

separately.

---

### 55. EMA teacher JEPA

Use the teacher-student EMA strategy from M-JEPA.

---

### 56. Multi-view JEPA

Same polymer represented through:

* SMILES;
* graph;
* randomized SMILES.

Predict one view's embedding from another.

---

### 57. Descriptor-prediction SSL

Train the encoder to reconstruct hundreds of RDKit descriptors.

This forces the latent space to carry physical chemistry information.

---

### 58. Multi-resolution graph SSL

Jointly predict:

* atom properties;
* local graph properties;
* whole-molecule descriptors.

This follows the node/edge/graph multilevel idea shown useful in polymer SSL. ([Royal Society of Chemistry Publications][6])

---

### 59. 6M-to-7k domain adaptation

Pretrain on 6M, then continue unsupervised training for several epochs on the 7,409 competition structures.

Recent molecular-transformer work found that **domain adaptation on a small domain-relevant dataset can be more useful than simply increasing pretraining corpus size**. ([PubMed Central (PMC)][7])

That is a major clue.

---

### 60. Target-aware domain adaptation

After 6M pretraining, continue training representations using only the chemistry distribution of structures possessing a particular target.

Different encoder adaptation for:

```text
Ei
Eea
Nc
Eps
```

---

# G. NEW MULTITASK OPTIMIZATION

The issue may not be whether multitask learning is good. It may be that **the optimization method is wrong**.

Recent ultra-low-data work specifically proposes adaptive checkpointing to mitigate negative transfer between imbalanced tasks. ([Nature][8])

### 61. Adaptive checkpointing specialization

Implement ACS-style task-specific checkpoints.

---

### 62. GradNorm

Normalize task gradient magnitudes dynamically. ([Proceedings of Machine Learning Research][9])

---

### 63. PCGrad

Project away conflicting task gradients. ([GitHub][10])

---

### 64. CAGrad

Use conflict-averse gradient descent. ([arXiv][11])

---

### 65. Task-wise early stopping

Don't early-stop the whole model.

Stop each target head at its own best point.

---

### 66. Task-wise dropout

Different dropout probability for each target head.

---

### 67. Task-specific latent bottleneck

Make separate latent dimensions:

```text
thermal latent
electronic latent
optical latent
```

plus a shared latent.

---

### 68. Task-adaptive sharing

Learn which encoder layers are shared and which are target-specific.

---

### 69. Mixture-of-experts by target

Multiple shared experts, with target-specific soft routing.

---

### 70. Pairwise multitask objectives

Instead of optimizing all seven simultaneously, train pairwise:

```text
Egc ↔ Egb
Egc ↔ Ei
Ei ↔ Eea
Nc ↔ Eps
```

then assemble.

---

# H. MISSING-LABEL / PROPERTY-MATRIX METHODS

This is one of the most literature-backed gaps now.

A dedicated molecular-property paper explicitly models the molecule–task relationship as a bipartite graph, imputes missing property labels, then retains reliable pseudo-labels using uncertainty. ([MI Research][12])

### 71. Molecule–target bipartite graph

Nodes:

```text
polymer
property
```

Edges:

```text
observed label
```

Run graph propagation.

---

### 72. Uncertainty-filtered property imputation

Impute only high-confidence missing labels.

---

### 73. Iterative property matrix completion

Alternate:

```text
matrix completion
→ chemical model
→ matrix completion
```

until convergence.

---

### 74. Nuclear-norm matrix completion + chemistry residual

Use low-rank completion only for the shared property structure, then predict residual with chemistry.

---

### 75. Nonlinear matrix completion

Use an MLP factorized latent representation rather than linear matrix factorization.

---

### 76. Property graph neural network

Properties themselves are graph nodes; learn task relationships.

---

### 77. Target covariance attention

The model learns an attention matrix over targets.

---

### 78. Masked-property pretraining

Randomly hide observed properties and predict them.

This creates huge numbers of property-space training cases from your existing multi-label rows.

---

### 79. Curriculum property masking

Begin with one masked target and gradually increase missingness.

---

### 80. Test-pattern conditioning

Train explicitly on the exact distribution of partner-availability patterns seen in test.

This is different from merely including missingness flags.

---

# I. NEW MODELING / ENSEMBLE COMBINATIONS

### 81. Stacking of *representations*, not predictions

Learn a meta-feature representation from:

```text
Morgan
MCP
GINE
JEPA
Hückel
RDKit
```

then train the final predictor.

---

### 82. Residual cascade across representations

```text id="6e74k9"
classical
   ↓
MCP residual
   ↓
GINE residual
   ↓
electronic residual
```

Each stage only sees out-of-fold residuals.

---

### 83. Orthogonal residual fitting

Before fitting the next model, project its features orthogonally to those already explained by previous models.

This deliberately searches for **new signal**, not redundant signal.

---

### 84. Adversarial ensemble selection

Choose ensemble members that maximize:

$$
R^2+\lambda\,diversity-\gamma\,instability.
$$

---

### 85. Bayesian stacking

Treat model weights as distributions rather than point estimates.

---

### 86. Fold-specific ensemble then shrink

Fit each fold's weights, then shrink them toward the global weights.

---

### 87. Cluster-specific blend with global shrinkage

For each chemical family:

$$
w_f=(1-\lambda_f)w_{global}+\lambda_fw_{family}.
$$

---

### 88. Target × availability × family ensemble

Three-level conditional weighting:

$$
w=f(target,availability,family).
$$

Use a very low-capacity model.

---

### 89. Winner-takes-softly ensemble

For each OOF point, determine which model performs best among a small candidate pool.

Train a calibrated soft probability of that event.

---

### 90. Prediction interval overlap ensemble

When two models produce overlapping uncertainty intervals, favor consensus; when they diverge, revert toward the safer parent.

---

# J. EXTREME / RESEARCH-GRADE ATTACKS

### 91. Optimal-transport train→test alignment

MROT-style work explicitly uses optimal transport to bridge chemical-domain shifts in molecular regression. ([PubMed Central (PMC)][13])

Construct train/test transport based on:

* Morgan;
* descriptors;
* learned embedding.

Then use transport mass as training weights.

---

### 92. Property-aware optimal transport

Instead of matching only chemical geometry, include the **observed target values** where available.

This is a label-aware domain adaptation variant.

---

### 93. Sinkhorn-weighted local regression

Use OT transport weights as the kernel for local prediction.

---

### 94. Train/test barycenter representation

Learn a common latent chemical representation whose marginal distribution matches both train and test.

---

### 95. Domain-adversarial representation learning

Encoder tries to predict target while being unable to distinguish:

```text
train chemistry
vs
test chemistry.
```

Use only unlabeled test structures for domain classification.

---

### 96. Target-specific domain adversary

Train separate domain-invariant encoders for:

```text
Ei
Eea
Nc
Eps
Tg
```

because their labeled subsets have different distributions.

---

### 97. Self-distilled ensemble representation

Train teacher ensemble → use predictions + disagreement → train student encoder.

Then use student + teacher predictions together.

---

### 98. Evolutionary feature search

Use a genetic algorithm to evolve expressions from:

```text
+ - * / log sqrt exp
```

over a small scientifically selected descriptor pool.

The 2025 QSPR–GAP Tg work itself used a genetic algorithm to identify dominant descriptors, so this is not merely speculative. ([ACS Publications][14])

---

### 99. Symbolic discovery separately for each *residual regime*

Instead of symbolic regression globally:

```text
high-Tg residuals
low-Tg residuals
high-conjugation residuals
ionic-heavy residuals
```

Search a simple formula independently.

---

### 100. Automated "model → error → feature" evolution loop

This is the one I would eventually build as the overall research engine:

```text
model
 ↓
OOF residual
 ↓
find chemical region where error concentrates
 ↓
discover descriptors distinguishing that region
 ↓
train specialist
 ↓
test specialist residual correlation
 ↓
promote / shrink
 ↓
repeat
```

This turns the competition into an iterative **scientific discovery problem**, rather than a static model comparison.

---

# The 15 I think deserve serious compute

After looking at the competitor evidence, your own failure log, and newer 2025–2026 literature, these are the ones I would put at the top.

### Tier 1 — highest expected value

**1. Property-mask pretraining**

`SMILES → hidden property vector reconstruction`

Why: it attacks exactly the partially observed structure of your dataset.

**2. Molecule–target bipartite graph imputation**

Why: direct literature precedent for missing molecular-property labels. ([MI Research][12])

**3. ACS-style adaptive multitask GINE**

Why: low-data, imbalanced targets + negative transfer is exactly your situation. ([Nature][8])

**4. Polymer-JEPA from scratch on 6M**

Why: this is a genuinely new SSL objective versus the old InfoNCE/MLM attempts. ([Royal Society of Chemistry Publications][5])

**5. MCP features + incumbent**

Why: completely different structural information, with published polymer-property results. ([OUP Academic][15])

**6. Exact radical-marker/attachment topology features**

Why: direct evidence from a top polymer Kaggle solution. ([Kaggle][16])

**7. AttentiveFP + GINE heterogeneous ensemble**

Why: direct precedent from the 2025 competition and still not actually exhausted in your archive. ([Kaggle][17])

**8. 6M domain adaptation after pretraining**

Why: recent molecular-transformer literature suggests domain adaptation can matter more than simply scaling the pretraining corpus. ([PubMed Central (PMC)][7])

**9. QSPR + learned group contribution Tg**

Why: directly supported by recent Tg polymer work. ([ACS Publications][3])

**10. Optimal-transport test-domain alignment**

Why: your documented OOF→test gaps are huge, and molecular OT methods specifically target domain mismatch. ([PubMed Central (PMC)][13])

---

# The 10 highest-upside target-specific experiments

## Ei

**11. JEPA/GINE representation → Ei only**

The 2024 polymer SSL paper specifically reported its largest low-data benefits around electron affinity and ionization potential. ([Royal Society of Chemistry Publications][6])

## Eea

**12. JEPA/GINE → Eea + Flory–Fox/Hückel/MCP fusion**

You already have a strong classical Eea component, so the new model only has to produce decorrelated residuals.

## Eps

**13. MCP + ionic decomposition**

Don't predict Eps directly.

Predict:

$$
ionic
$$

from MCP + graph + polar descriptors.

## Nc

**14. MCP + polarizability + 3D**

Focus on optical structure.

## Egb

**15. Egb affine(Egc) + MCP/GINE residual**

This directly exploits the one relationship where your history shows residual ML already works. 

---

# And this is the pipeline I would actually build

Not 100 independent submissions.

Something like this:

```text
                              OFFICIAL DATA
                                  │
                 ┌────────────────┼─────────────────┐
                 │                │                 │
              train           test SMILES        6M/PI1M
                 │                │                 │
                 └───────────────┼─────────────────┘
                                 │
                       CHEMICAL REPRESENTATIONS
                                 │
       ┌──────────────┬──────────┼──────────┬────────────┐
       │              │          │          │            │
     RDKit          Morgan      MCP       GINE        JEPA
       │              │          │          │            │
       └──────────────┴──────────┼──────────┴────────────┘
                                 │
                         PHYSICAL FEATURES
                                 │
             ┌───────────────────┼──────────────────┐
             │                   │                  │
          Hückel             ionic              GAP/QSPR
             │                   │                  │
             └───────────────────┼──────────────────┘
                                 │
                       PROPERTY-SPACE LEARNING
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
       masked-property      target graph      missing-label
         prediction          / MT learning       imputation
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 │
                       TARGET-SPECIFIC MODELS
                                 │
      ┌────────┬────────┬────────┬────────┬────────┬────────┐
      Tg      Egc      Egb      Ei       Eea      Nc       Eps
      │        │        │        │        │        │        │
      └────────┴────────┴────────┴────────┴────────┴────────┘
                                 │
                    PHYSICS / PARTNER RECONSTRUCTION
                                 │
              ┌──────────────────┼────────────────────┐
              │                  │                    │
          Ei identity         Egb relation        eps ionic
              │                  │                    │
              └──────────────────┼────────────────────┘
                                 │
                        DIVERSE MODEL POOL
                                 │
      Classical / MCP / GINE / AttentiveFP / JEPA / symbolic
                                 │
                      residual correlation matrix
                                 │
                      target-specific simplex blend
                                 │
                availability / family / density weighting
                                 │
                     test-distribution correction
                                 │
                         final submission
```

That is the first architecture I would genuinely call a **research pipeline** rather than a model zoo.

---

# One especially important distinction

Your prior experiments tell us that some ideas fail when used as **replacement models**.

For example:

* GNNs;
* low-rank multitask;
* SSL embeddings;
* KRR;
* similarity models.

The history shows that many of these were worse than the incumbent. 

But that doesn't imply:

> representation/model has zero value.

It may imply:

> **representation/model contains complementary information but is too weak to replace the incumbent.**

That is exactly why I would test almost every new representation as:

$$
\boxed{
\text{incumbent}
+
\text{new model's OOF residual}
}
$$

rather than:

$$
\boxed{\text{new model instead of incumbent}}.
$$

The 2025 Open Polymer Challenge evidence strongly supports heterogeneous ensembling: first place combined BERT/AutoGluon/Uni-Mol, third combined GATv2 and fingerprint information, fourth used LightGBM plus new chemistry-derived features and 3D information, and fifth explicitly built a hierarchical physics/ML ensemble. ([Kaggle][2])

---

# The biggest 5 things I think were genuinely missing

After all of this, my strongest conclusion is:

### 1. **Property-space self-supervision**

Not SMILES SSL.

Learn the relationships among the seven actual targets.

### 2. **JEPA-style predictive SSL**

Not another InfoNCE/MLM run.

The recent polymer JEPA result makes this a particularly interesting new branch. ([Royal Society of Chemistry Publications][5])

### 3. **Molecule–target bipartite modeling**

Treat missing labels as a graph problem rather than a nuisance. ([MI Research][12])

### 4. **A genuinely different polymer geometry representation**

MCP is the clearest candidate. ([OUP Academic][15])

### 5. **Adaptive multitask optimization**

ACS / GradNorm / PCGrad / CAGrad rather than another naïve shared encoder. ([Nature][8])

---

# One thing I would absolutely not do

I would **not** run these 100 one at a time and submit every slightly improved model.

Use this logic:

### Phase A — discovery

Run cheap EDA / representation diagnostics.

### Phase B — specialists

Generate candidate component models.

### Phase C — residual correlation

Discard redundant components.

### Phase D — compound

Build target-specific mixtures.

### Phase E — physics

Apply the validated identities.

### Phase F — domain correction

Apply only validated distribution-shift adjustments.

### Phase G — final two portfolios

One conservative.

One aggressive.

That is how you turn 20 modest improvements into one potentially large final improvement.

---

# Most importantly: I think the next serious objective should be 0.920 first

Your current clean/oracle history is around 0.904–0.905, while the archive explicitly says the 0.93+ values were only achieved in oracle-assisted diagnostic experiments that were not clean-replayable. 

So I would set these gates:

$$
0.904\rightarrow0.912
$$

**prove one new family.**

Then:

$$
0.912\rightarrow0.920
$$

**prove compound architecture.**

Then:

$$
0.920\rightarrow0.930
$$

**attack Ei/Nc/Eps/Eea specifically.**

Then:

$$
0.930\rightarrow0.935+
$$

**ensemble + distribution + last-mile corrections.**

That is much more realistic scientifically than pretending one of the 100 bullets will magically output 0.935.

And given the current date and your stated September 3 competition deadline, I would put **JEPA + property-masking + MCP + AttentiveFP/GINE + molecule–target imputation + exact availability-pattern modeling** ahead of another generic descriptor sweep. The strongest recent literature points in those directions, and they are genuinely less explored in your experiment history than the families you've already exhausted. ([Royal Society of Chemistry Publications][5])

**That is the new research frontier I would attack now.**

[1]: https://www.nature.com/articles/s41524-023-01034-3?utm_source=chatgpt.com "Polymer graph neural networks for multitask property learning | npj Computational Materials"
[2]: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/1st-place-solution?utm_source=chatgpt.com "1st Place Solution | Kaggle"
[3]: https://pubs.acs.org/doi/10.1021/acs.macromol.5c00178?utm_source=chatgpt.com "Polymer Informatics Method for Fast and Accurate Prediction of the Glass Transition Temperature from Chemical Structure | Macromolecules | ACS Publications"
[4]: https://www.nature.com/articles/s41524-025-01652-z?utm_source=chatgpt.com "Unified multimodal multidomain polymer representation for property prediction | npj Computational Materials"
[5]: https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00308c?utm_source=chatgpt.com "Joint embedding predictive architecture for self-supervised pretraining on polymer molecular graphs - Digital Discovery (RSC Publishing) DOI:10.1039/D5DD00308C"
[6]: https://pubs.rsc.org/en/content/articlehtml/2024/me/d4me00088a?utm_source=chatgpt.com "Self-supervised graph neural networks for polymer property prediction - Molecular Systems Design & Engineering (RSC Publishing) DOI:10.1039/D4ME00088A"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC13430829/?utm_source=chatgpt.com "Transformers for molecular property prediction: domain adaptation efficiently improves performance - PMC"
[8]: https://www.nature.com/articles/s42004-025-01592-1?utm_source=chatgpt.com "Molecular property prediction in the ultra‐low data regime | Communications Chemistry"
[9]: https://proceedings.mlr.press/v80/chen18a.html?utm_source=chatgpt.com "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
[10]: https://github.com/NoteDance/PCGrad?utm_source=chatgpt.com "GitHub - NoteDance/PCGrad: TensorFlow implementation for \"Gradient Surgery for Multi-Task Learning\" · GitHub"
[11]: https://arxiv.org/abs/2110.14048?utm_source=chatgpt.com "Conflict-Averse Gradient Descent for Multi-task Learning"
[12]: https://www.mi-research.net/en/article/doi/10.1007/s11633-023-1443-7?utm_source=chatgpt.com "Improving Multi-task GNNs for Molecular Property Prediction via Missing Label Imputation"
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10140620/?utm_source=chatgpt.com "Improving molecular representation learning with metric learning-enhanced optimal transport - PMC"
[14]: https://pubs.acs.org/doi/abs/10.1021/acs.macromol.5c00178?utm_source=chatgpt.com "Polymer Informatics Method for Fast and Accurate Prediction of the Glass Transition Temperature from Chemical Structure | Macromolecules | ACS Publications"
[15]: https://academic.oup.com/bib/article/25/6/bbae465/7774896?utm_source=chatgpt.com "Multi-Cover Persistence (MCP)-based machine learning for polymer property prediction | Briefings in Bioinformatics | Oxford Academic"
[16]: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/4th-place-solution-lightgbm-with-smiles-derived-fe?utm_source=chatgpt.com "4th Place Solution — LightGBM with SMILES-Derived Features | Kaggle"
[17]: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/3rd-place-solution?utm_source=chatgpt.com "3rd Place Solution | Kaggle"
