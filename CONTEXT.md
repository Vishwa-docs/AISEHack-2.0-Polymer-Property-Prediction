# CONTEXT.md — the whole project on one page

*Portable. Paste this into any agent with no repo access.*
*(Mirror of `Personal/CONTEXT.md` — keep them identical.)*

## The competition

**ANRF AISEHack 2.0 · Polymer Property Prediction · Round 3** (Kaggle, final stage).
Hosts: Rohit Batra (IIT Madras), Rahulsundar, LaksmanN, VIJITH P, shreyasri0301.
22 Aug – **3 Sep 2026** · 3 submissions/day · 2 final. Team **Sandman**.

Predict **seven properties** of a polymer from its repeat unit, given as a SMILES string with
two `*` connection points. Data is **long format**: one row = one (polymer, property) pair.

**Metric: the unweighted mean of the seven per-target R².** Every target is worth exactly 1/7
regardless of row count. Submission: `submission.csv`, **4,940 rows**, ids 1..4940, columns
exactly `id,target`.

**Round 3 is judged on**: leaderboard score, **model explainability**, **polymer invariance**,
methodology, and proven generalization.

## The seven targets

| code | property | unit | origin | train n | test n | our R² |
|---|---|---|---|---:|---:|---:|
| tg | glass transition temperature | °C | experimental | 4,143 | 2,763 | 0.8953 |
| egc | chain bandgap | eV | DFT | 2,028 | 1,352 | 0.9111 |
| egb | bulk bandgap | eV | DFT | 337 | 224 | 0.9268 |
| ei | ionisation energy | eV | DFT | 222 | 148 | 0.8711 |
| eea | electron affinity | eV | DFT | 221 | 147 | 0.9183 |
| nc | refractive index | – | DFT | 229 | 153 | 0.9086 |
| eps | dielectric constant | – | DFT | 229 | 153 | 0.8847 |

## Our result

**private LB 0.891** · public LB 0.917 · local held-out verification panel **0.9023** ·
evidence scorecard **14/18 groups PASS**. Practical ceiling ≈0.93 ± 0.01, so we are at ~96%.

## The constraints

Official data only (`train.csv`, `test.csv`, `PI1M.csv`, `smile_r3.csv`). **No external
datasets, no pretrained weights, embeddings or vocabularies, no transfer learning, no artifacts
created outside the run.** Everything fitted from scratch, in one run, fixed seed **2026**.

## The architecture in six sentences

A shared representation is built from RDKit 2D descriptors, Morgan count fingerprints, character
n-grams, a Tanimoto kernel, Polymer-Genome atomic triples and a label-free PI1M character SVD.
Each of the seven targets then gets its own lane, with the model family chosen by sample size
and by physics. Three measured physical identities are overlaid where their inputs exist:
`egc = ei − eea`, `eps = n² + ionic`, `egb = 1.1586·egc − 1.0437`. The lanes are assembled by
out-of-fold non-negative least squares plus signed-residual splices. A calibration layer adds
0.20 × a character-residual on five targets and re-expands ei/eea by 1.05 around the training
median. An evidence engine runs in parallel and produces the explainability, invariance,
reliability and generalization bundle.

## The five headline findings

1. **Zero label leakage, 98% structure overlap.** 0 exact (polymer, property) pairs shared with
   train for all six DFT targets, but 88–99% of those polymers *are* in train under a different
   property; Tg only 12.3%. Two problems — imputation and extrapolation — one leaderboard.
   **This is why the pipeline is per-target.**
2. **Tg owns 99.986% of the pooled variance but 1/7 of the score.** A perfect Tg model still caps
   the mean at **0.9172**.
3. **The physics is real and beats the model that corrects it.** `egc = ei − eea` holds at
   R² 0.9716 (n=59); adding an ML residual scores leave-one-out **R² −0.82**.
   `eps = n² + ionic` has **0/134 violations** and is 2.62× better conditioned (+0.0666 on eps).
4. **We predicted our own private score to within 0.0004** from difficulty-stratified Tg R²
   (easy 0.9023 / medium 0.8856 / hard 0.8305).
5. **Same polymer, any spelling, same answer *and the same reasons*.** Prediction std ≤0.23% of
   the training std, SHAP cosine 0.95–0.99, activation-patching delta exactly 0.0; masking the
   top-10% SHAP features costs 0.851 R² against 0.043 random.

## The four honest failures

cross-model explanation agreement ρ **0.471** (bar 0.60) · conformal coverage max |Δ| **0.089**
(bar 0.03, ~45 calibration rows) · error–uncertainty correlation ρ ≥ 0.30 on **1** target
(bar 5) · a missing augmentation artifact. Each has an identified cause and a fix path.

## Vocabulary rules

Say **"local held-out verification panel"**, never "oracle". Tg is in **°C**. Standardise on
**0.9023**. Always state whether a similarity/overlap figure is on a *same-property* or
*any-property* basis.

## Environment (load-bearing)

`python 3.11.7 · rdkit 2026.03.5 · numpy 2.4.6 · pandas 3.0.5 · scikit-learn 1.9.0 ·
lightgbm 4.7.0 · xgboost 3.2.0`. On **python 3.12** the ei leaf models collapse 0.871 → 0.512
*regardless of package versions*. A far-below-0.90 score with identical code and data is an
**environment mismatch, not a model regression**.
