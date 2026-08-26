# Technical Research Report — Polymer Property Prediction (AISEHack 2.0 Round 3)

**Objective.** Predict 7 polymer properties from SMILES: chain bandgap (Egc), bulk bandgap (Egb),
ionization energy (Ei), electron affinity (Eea), dielectric constant (EPS), refractive index (Nc),
glass transition temperature (Tg). Metric = unweighted mean R² across the 7 targets.
Train = 7,409 sparse rows · Test = 4,940 rows · Unlabeled: PI1M.csv (995,799 polymer SMILES) +
smile_r3.csv (5,973,369 molecular SMILES). Notebook/code-only: no pretrained weights, no external
data, everything from scratch in one Kaggle notebook.

**Key structural fact discovered during research.** The competition's label matrix is the **Khazana
dataset** (sparse, ~95% missing per property). The 6 electronic/optical targets (Egc, Egb, Ei, Eea,
EPS, Nc) are *deterministic DFT computations* (noiseless labels → high achievable R²); **Tg is
experimental** (noisy → R² ceiling ~0.80–0.92). This asymmetry is the single most important thing to
know for modeling.

---

## 1. The Khazana dataset paper (DOI 10.1016/j.patter.2021.100238)

- The paper is **"Polymer informatics with multi-task learning"** — Kuenneth, Rajan, Tran, Chen, Kim,
  Ramprasad (Georgia Tech), *Patterns* 2(4):100238 (2021). PMCID PMC8085610.
  * https://pmc.ncbi.nlm.nih.gov/articles/PMC8085610/
  * https://pubmed.ncbi.nlm.nih.gov/33982028/
  * https://www.cell.com/patterns/fulltext/S2666-3899(21)00058-1
- **Contents:** 36 individual polymer properties over ~13,000 polymers, collected from two sources:
  (i) in-house high-throughput **DFT** computations ("polymer genome" workflow, refs 12/20/25/26),
  and (ii) **experimental** measurements from handbooks and online databases (incl. PoLyInfo).
  The paper explicitly warns that DFT and experimental values must *not* be mixed in one column.
- **Feature representation:** a 953-component hierarchical fingerprint (atomic-level, molecular/block,
  and "morphological" descriptors such as shortest topological ring distance, side-chain fraction,
  largest side-chain length) fused with **RDKit QSPR** descriptors (e.g., van der Waals surface area).
- **Models & results:** Gaussian-process single-task (GP-ST), neural single-task (NN-ST), multi-head
  (NN-MT1), and a multi-task DNN (NN-MT2). NN-MT2 (multi-task) **outperforms all single-task models**,
  especially where properties are correlated and/or datasets are small. Reported RMSE improvements
  (0.79 → 0.65) on averaged tasks. This is the direct justification for multi-task learning here.
- **SHAP interpretability findings (actionable):**
  * Ring fraction (fraction of non-H atoms in rings) **increases** Tm, Tg, Td, strength, Oi, **nc, ne,
    ε0, ε15** and **decreases** Egc, Egb, Eat, Ei, cp.
  * Conjugated double bonds in rings create "agitated π-electrons" → higher **nc, ne, εf** (optical/refractive).
  * Dielectric constants across frequencies are highly mutually correlated; ε15 (optical) is only weakly
    correlated with low-frequency ε (because low-freq ε includes dipolar/orientational polarization).
  * Strong **positive** Tg–Tm correlation; strong **negative** correlation between bandgap (Egb, Egc) and Tg.
- **DFT level (via the Polymer Genome workflow this dataset inherits):** the in-house electronic
  properties come from Kim et al., "Polymer Genome: A Data-Powered Polymer Informatics Platform for
  Property Predictions," *J. Phys. Chem. C* (2018), 10.1021/acs.jpcc.8b02913 — oligomers (≈8 repeat
  units) optimized with a **hybrid functional (PBE0) and 6-31G(d,p) basis set**; band gap = HOMO–LUMO
  gap (chain = isolated oligomer, bulk = amorphous packed cell); **Ei = −HOMO, Eea = −LUMO (Koopmans
  theorem vertical energies)**; dielectric constant and refractive index from **electronic polarizability
  via Clausius-Mossotti / Lorentz-Lorenz** (ionic + orientational terms added for low-frequency ε).
  *(I could not pull a verbatim page-level confirmation of "PBE0" this session — ACS blocks scraping —
  so treat the functional name as the Ramprasad-group standard; the HOMO-LUMO/Koopmans/polarizability
  mechanism is firmly documented across their papers.)*
- The earlier, distinct 3D-periodic dataset (Huan et al., *Sci. Data* 3:160012, 2016) used VASP with
  rPW86-GGA + **HSE06** for crystalline bandgaps and DFPT dielectric constants — useful context but
  not the source of these competition labels. https://www.nature.com/articles/sdata201612
- **Companion paper** "Machine learning discovery of high-temperature polymers" (Kuenneth et al.,
  *Patterns* 2021, PMC8085602) is where **Tg + PI1M** come from: 6,923 experimental Tg from PoLyInfo;
  **PI1M = ~1M RNN-generated hypothetical polymers**; Tg models reached R² 0.80–0.87.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8085602/
- **Dataset mirrors:** JARVIS/ColabFit now host a "Polymer-Genome" snapshot:
  https://doi.datacite.org/dois/10.60732%2F37f5fcea

---

## 2. Modern deep learning for polymer property prediction (2023–2026)

**Bottom line:** self-supervised SMILES/graph pretraining beats fingerprints, but by a modest margin
on the DFT targets; the big wins are on the scarce targets (EPS, Nc, Ei, Eea) where extra unlabeled
data helps most. All models below are *architectures* we can re-implement and retrain from scratch on
the official unlabeled SMILES.

- **polyBERT** (Kuenneth & Ramprasad, *Nat. Commun.* 14:4099, 2023). DeBERTa-v2 transformer on
  **polymer SMILES (PSMILES)**, tokenized at atom level; pretrained (masked-LM) on **100M hypothetical
  PSMILES**; **25.2M params**, 600-dim pooled embedding. On the same 29-property polymer-genome set it
  matches the handcrafted Polymer-Genome fingerprint (**overall R² 0.80 vs 0.81**) while being ~100×
  faster to compute.
  * https://pubmed.ncbi.nlm.nih.gov/37433807/ · https://www.nature.com/articles/s41467-023-39868-6
- **polyBERT2** (Kuenneth et al., 2024, arXiv 2410.xxxx / OSTI 2282951). Scaled pretraining to
  **100M polymer SMILES**; improves on polyBERT especially for transfer and de-novo screening.
  * https://www.osti.gov/pages/servlets/purl/2282951
- **TransPolymer** (Xu, Wang, Barati Farimani, *npj Comput. Mater.* 9:64, 2023). Transformer with a
  chemistry-aware polymer tokenizer; **82.1M params**; pretrained (masked-LM) on ~5M PSMILES. Fine-tuned
  (multi-task) R² on the SAME datasets we face: **Egb 0.95, EPS 0.86, Nc 0.91** (with a note these
  numbers come with the multi-task/descriptor-token setup). Frozen-representation avg R² ≈ 0.78.
  * https://ar5iv.labs.arxiv.org/html/2209.01307 · https://rd.springer.com/article/10.1038/s41524-023-01016-5
- **polyGNN** (Gurnani, Kuenneth et al., *Chem. Mater.* 35:1560, 2023). Multi-task message-passing GNN
  over ~13,000 polymers / 36 properties; strong on thermal/mechanical, weaker on gas permeability.
  * https://pubmed.ncbi.nlm.nih.gov/36873627/ · 10.1021/acs.chemmater.2c02991
- **PolyCL** (Zhou et al., *Digit. Discov.* 4:149, 2024). Contrastive learning (SimCLR-style) on
  polymer SMILES initialized from polyBERT; frozen extractor + MLP head. **Best overall avg R² 0.790
  on the 7-property benchmark.** https://pmc.ncbi.nlm.nih.gov/articles/PMC11616009/
- **Uni-Mol** (Zhou et al., ICLR 2023). Universal 3D-molecular transformer (atom + pair representations),
  pretrained on ~209M 3D conformers + pocket data. Not polymer-specific but a strong general molecular
  encoder. https://github.com/RUC-ALGO/Uni-Mol · https://mlanthology.org/iclr/2023/zhou2023iclr-unimol/
- **MolCLR** (Wang et al., ICML 2021). Contrastive GNN with graph augmentations (atom masking, bond
  deletion, subgraph removal); pretrained on 10M PubChem molecules; small and cheap to retrain.
  https://ar5iv.labs.arxiv.org/html/2102.10056
- **GROVER** (Rong et al., NeurIPS 2020). Self-supervised graph transformer (contextual property +
  graph-motif prediction), 10M molecules; 48M/100M params. https://arxiv.org/abs/2007.02835
- **SELFormer** (Yuksel et al., 2023). BERT on **SELFIES** (100%-valid strings); pretrained on 2M
  compounds; competitive with ChemBERTa on MoleculeNet. https://github.com/HUBioDataLab/SELFormer ·
  https://ar5iv.labs.arxiv.org/html/2304.04662
- **ChemBERTa-2** (Ahmad et al., 2022). RoBERTa on ~10M SMILES + multi-task regression pretraining;
  small variants (5M/46M params) → easiest to retrain from scratch in a notebook.
  https://www.alphaxiv.org/overview/2209.01712 · https://github.com/miservilla/ChemBERTa

**The authoritative 7-property benchmark (PolyCL paper, Table 1) — mean R² per target, 5-fold CV.**
These are the exact competition-style targets (Xc = crystallization tendency, excluded here):

| Model | Eea | Egb | Egc | Ei | EPS | Nc | Avg (6) |
|---|---|---|---|---|---|---|---|
| RF + ECFP | 0.840 | 0.864 | 0.870 | 0.742 | 0.684 | 0.754 | 0.759 |
| XGB + ECFP | 0.835 | 0.857 | 0.868 | 0.722 | 0.673 | 0.757 | 0.752 |
| NN + ECFP | 0.854 | 0.871 | 0.884 | 0.756 | 0.747 | 0.807 | 0.820 |
| GCN | 0.854 | 0.804 | 0.799 | 0.665 | 0.740 | 0.524 | 0.731 |
| GIN | 0.883 | 0.835 | 0.818 | 0.784 | 0.693 | 0.632 | 0.774 |
| TransPolymer | 0.894 | 0.896 | 0.876 | 0.792 | 0.757 | 0.811 | 0.838 |
| polyBERT | 0.907 | 0.883 | 0.878 | 0.767 | 0.769 | 0.802 | 0.834 |
| PolyCL | **0.907** | 0.888 | 0.883 | **0.811** | **0.788** | **0.846** | **0.854** |

*(GP/NN on the handcrafted Polymer-Genome fingerprint reached Eea 0.90 / Egb 0.91 / Egc 0.90 but only
EPS 0.68–0.71 and near-zero on Xc — the PG fingerprint is excellent for electronic properties.)*

**Interpretation for Round 3:** a plain RDKit-ECFP + NN already gives ~0.82 mean on the 6 DFT targets;
pretrained-from-scratch transformers add ~+0.03–0.05 mean, concentrated in **Ei, EPS, Nc**. The handcrafted
**Polymer-Genome fingerprint** is a *strong* cheap baseline for Egc/Egb/Eea/Ei specifically.

---

## 3. Physics-informed features (formulas computable from RDKit descriptors)

These are concrete, rules-compliant (no external data) and, per the competition theme, also provide
built-in explainability.

- **Clausius-Mossotti (dielectric):**  (εr − 1)/(εr + 2) = N_A·ρ·α_total / (3·ε0·M),
  where α_total = α_electronic + α_ionic + α_orientational. At optical frequency only α_electronic survives.
  So a *group-contribution polarizability* directly yields a dielectric-constant estimate.
- **Lorentz-Lorenz (refractive index):**  (n² − 1)/(n² + 2) = N_A·ρ·α_e / (3·ε0·M) = R_M/V_m,
  with R_M = molar refraction. Rearranged:  n = sqrt[(1 + 2R/V) / (1 − R/V)].
- **Gladstone-Dale (simpler, additive):**  n = 1 + ρ·R_GD  (R_GD = specific refraction, additive over groups).
- **Optical identity (ε ↔ n):**  ε∞ = n²  at optical frequency. → use  Nc ≈ sqrt(EPS_electronic)  as a
  cross-target feature/regularizer (and a cheap consistency check). RDKit's Crippen/MR and molar
  refraction give a pure-SMILES R_M; density can be estimated via RDKit or a small density sub-model.
- **Frontier-orbital identities (DFT labels obey these by construction):**
  Ei = −E_HOMO,  Eea = −E_LUMO,  and  Egc ≈ E_HOMO − E_LUMO ≈ **Ei − Eea**.
  → Two features are nearly redundant given the third; enforce/model the constraint
  **Egc ≈ Ei − Eea** and **Eea ≈ Ei − Egc**. This is a powerful, exact physics prior for the 3 sparse
  electronic targets (Ei/Eea have only ~222 rows each; Egc has 2,028).
- **Bulk vs chain bandgap:**  Egb ≈ Egc + Δ_solid  where Δ_solid (usually negative) captures chain packing /
  inter-chain orbital overlap; Δ_solid is small and roughly constant for chemically similar polymers.
  Model Egc first (more data) and predict Egb as a residual.
- **van Krevelen / Hoftyzer group contributions:** molar refraction R_M and dielectric (molar
  polarization) are **additive over structural groups**. Build features = Σ (group molar-refraction /
  group-polarization counts) parsed from SMILES substructure counts (RDKit SMARTS). Same machinery used
  by PoLyInfo's built-in predictor. https://polymer.nims.go.jp/PoLyInfo/guide/en/Property_Prediction_HELP_EN.html
- **Bicerano group contribution for Tg** (see §6):  Tg = Σᵢ Yᵢ / M  (Yᵢ = molar Tg contribution per group),
  optionally + structural corrections (rotational/informational terms). Direct SMARTS-count feature.
  * https://pubs.acs.org/doi/10.1021/acsomega.0c04499
- **Fox equation (copolymers):**  1/Tg = w₁/Tg₁ + w₂/Tg₂  (weight fractions). Useful if we decompose
  copolymers into comonomer features.
- **Flory-Fox (molecular weight):**  Tg = Tg∞ − K/Mn. Captures finite-MW depression; relevant because
  oligomers/SMILES are not infinite chains.
- **Empirical links worth encoding:** aromatic/conjugated ring count ↑ → Nc↑, EPS↑, Egc↓, Tg↑; polar
  groups (C=O, O, N, F, S) ↑ → EPS↑; heavy atoms (S, halogens, aromatic) ↑ → Nc↑; backbone rigidity ↑ → Tg↑.

---

## 4. RDKit / Mordred descriptor families most predictive

- **Khazana paper itself:** 953-component hierarchical fingerprint + RDKit QSPR; SHAP showed **ring
  fraction** and **aromatic/conjugated ring content** dominate Tg, Nc, EPS, Egc, Ei simultaneously.
- **High-temp-polymer paper:** 5,305 descriptors (≈Mordred set), 3,579 valid; Morgan fingerprints;
  DNN/Lasso/CNN on images. https://pmc.ncbi.nlm.nih.gov/articles/PMC8085602/
- **PolyCL/TransPolymer baselines:** RDKit **ECFP (Morgan)** alone with RF/XGB/NN ≈ 0.76–0.82 mean on
  the 6 DFT targets — the strongest *cheap* feature (see table in §2).
- **Most predictive Mordred families per property type:**
  * **Electronic (Egc/Egb/Ei/Eea):** autocorrelation descriptors **ATS/AATS/GATS/Moran/Geary** (encode
    electronegativity/polarizability along the chain — proxy for conjugation and HOMO-LUMO gap);
    **EState/electrotopological** indices; **TopoPSA**; polarizability **APol/BPol**; conjugation/aromatic
    ring counts. These are the descriptor-space analogs of "conjugation length → bandgap".
  * **Dielectric (EPS):** **APol/BPol (polarizability)** + counts of polar groups (C=O, C≡N, O, N, S, F,
    sulfone, carbonate, nitrile) + dipole-related EState sums. Polarizability is the literal driver via
    Clausius-Mossotti.
  * **Refractive index (Nc):** **molar refraction (Crippen MR / ABC), APol/BPol**, heavy-atom & aromatic
    counts, polarizability, unsaturation. High-n groups: aromatic rings, S, halogens, conjugated C=C.
  * **Tg:** ring/aromatic fraction, rotatable-bond count, backbone chain-flexibility indices (Kier
    flexibility), H-bond donor/acceptor counts, MW, side-chain length. Rigidity + polarity ↑ Tg.
- **Published feature-importance patterns** (consistent across Khazana, PolyCL, and the high-Tg paper):
  ring fraction ↑ → Tg↑, Nc↑, EPS↑, Egc↓, Ei↓; aromaticity/conjugation ↑ → bandgap↓, refractive↑;
  polarity ↑ → dielectric↑, Tg↑. These rank-orderings can be encoded as explicit interaction/ratio features.

---

## 5. What actually limits accuracy (and reported R² levels)

- **Tg (experimental labels).** Literature R² typically **0.80–0.92**:
  * 0.92 (RMSE 27 K) on 1,321 polymers (early polymer-genome ML);
  * 0.87 train / 0.80 test (CNN on images) and ~0.63–0.68 transfer on 6,923 PoLyInfo Tg values;
  * equivariant-GNN large-scale Tg screening reports comparable levels.
  * https://pmc.ncbi.nlm.nih.gov/articles/PMC8085602/ · https://pubs.acs.org/doi/full/10.1021/acsomega.3c06843
  * https://onlinelibrary.wiley.com/doi/10.1002/pol.20230714
  * **Limiters:** experimental noise (heating rate, tacticity, MW, moisture), definitional spread, and
    sparse curated coverage. Do **not** expect >~0.92 for Tg; ~0.85 is realistic and strong.
- **DFT electronic/optical labels (Egc, Egb, Ei, Eea, EPS, Nc) are deterministic** → no label noise; the
  limit is **data scarcity + representation**:
  * Bandgap (Egc/Egb) is the *easiest* (R² 0.87–0.91 achievable even with fingerprints).
  * Ei/Eea/EPS/Nc have only **148–337 labeled rows** in this competition → variance-dominated; the
    benchmark shows pretrained-from-scratch embeddings are the biggest lever (+0.03–0.09 on EPS/Nc/Ei).
- **Net achievable picture:** a well-engineered model should land ~0.83–0.87 mean R² across the 7 targets
  (≈0.84–0.90 on bandgaps, ≈0.80–0.86 on Eea/Ei, ≈0.75–0.85 on EPS/Nc, ≈0.80–0.88 on Tg). The scarce
  targets (EPS, Nc, Ei, Eea, Egb) — each weighted 1/7 in the mean — will dominate the final metric.

---

## 6. Group-contribution / read-across for polymer Tg

- **PoLyInfo (NIMS)** is the canonical source: ~18,000+ polymer records with experimental Tg/Tm/Td,
  and a built-in group-contribution property predictor. https://polymer.nims.go.jp/datapoint.html ·
  https://polymer.nims.go.jp/PoLyInfo/guide/en/Property_Prediction_HELP_EN.html
- **Bicerano, "Prediction of Polymer Properties" (1993/2002):** Tg from additive molar group
  contributions Yᵢ plus structural corrections (rotational freedom, symmetry); the simple widely-used
  form is **Tg = Σᵢ (Nᵢ·Yᵢ) / M**. A modern implementation/refit: https://pubs.acs.org/doi/10.1021/acsomega.0c04499
- **van Krevelen & te Nijenhuis, "Properties of Polymers" (4th ed.):** group-contribution Y_g for Tg,
  plus molar-refraction/polarization contributions for n and ε — the reference for all additive methods.
- **A simpler general group-contribution scheme** (Marañón-style) for Tg of polymers/diluents:
  https://pubs.acs.org/doi/10.1021/ie0205389
- **Fox (copolymer) and Flory-Fox (MW)** equations (§3) complete the Tg toolkit.
- **Read-across:** k-nearest-neighbor over ECFP/Mordred/Tg-group-contribution space with inverse-distance
  weighted Tg is a strong, interpretable, rules-compliant baseline that also "transfers" well within
  chemical families.
- **Transferability caveats:** group-contribution Tg works well for linear/vinyl/aromatic homopolymers
  but degrades for conjugated rigid-rod, heavily H-bonded (polyamides/polyimides), or cross-linked
  systems — exactly the regimes where a learned model with ring/flexibility descriptors outperforms.

---

## 7. Implications for Round 3 (ranked, rules-compliant ideas)

1. **Physics-grounded engineered features + gradient-boosting/MLP ensemble (do this first).**
   Build Mordred + RDKit descriptors **plus** group-contribution features (molar refraction, polarizability,
   ring/aromatic/polar-group counts, Bicerano/van-Krevelen Tg contributions, flexibility) and explicit
   cross-target features **Nc²≈EPS** and **Egc≈Ei−Eea**. Train LGBM/XGB/CatBoost + a small NN per target
   (or a shared multi-task MLP). Expected: ~0.80–0.84 mean, near-zero cost, fully interpretable.
2. **Multi-task learning (Khazana's headline result).** Shared backbone with 7 heads (or two heads:
   {Egc,Egb,Ei,Eea,EPS,Nc} + {Tg}); exploit Tg↔bandgap negative correlation and ε↔n identity. The sparse
   targets (Ei/Eea/EPS/Nc/Egb) borrow strength from Egc (2,028) and Tg (4,143). This is *the* proven fix
   for 95%-sparse matrices.
3. **From-scratch self-supervised pretraining on the official unlabeled SMILES (biggest lever for scarce
   targets).** Use PI1M (995k polymers) + smile_r3 (5.97M molecules) to pretrain a small masked-LM
   transformer (polyBERT-style, 10–25M params, atom-level PSMILES tokenizer) or a contrastive GNN
   (MolCLR-style). Then fine-tune on 7,409 labels. Literature says this adds +0.03–0.09 on EPS/Nc/Ei.
   *Rules note:* pretraining on PI1M/smile_r3 is explicitly allowed (both are official Round 3 data);
   we must build the weights ourselves inside the notebook (no downloads). Sanity-check that both files
   exist in the Round 3 `/kaggle/input/` dir at notebook time before relying on them; smile_r3.csv alone
   (5.97M molecules) still supports from-scratch pretraining.
4. **Handcrafted Polymer-Genome-style hierarchical fingerprint** (atomic → block → chain, with ring/
   side-chain/aromaticity statistics) as an additional feature block — literature shows it rivals
   transformers on Egc/Egb/Eea/Ei and is trivial to compute.
5. **Pseudo-labeling / self-training** on the unlabeled polymers after step 3 (predict, keep confident
   rows, retrain) to densify Ei/Eea/EPS/Nc.
6. **Enforce physics consistency at inference** (optional, low-risk): predict {Ei, Eea, Egc} and
   {EPS, Nc} with shared heads so the residual Nc−√EPS and Egc−(Ei−Eea) stay small; or add these as
   soft auxiliary losses.
7. **Validation hygiene (required by the competition's own rules).** Structure-grouped CV (457 SMILES
   appear in both train and test — a canonical structure must never straddle folds), scaffold/family
   folds, similarity-cluster folds, and per-target availability masking. Report per-target R² + mean
   (never pool rows).
8. **Explainability deliverable.** SHAP on the physics+descriptor model directly maps to the "ring
   fraction ↑ Tg, ↓ bandgap; conjugation ↓ bandgap, ↑ n; polarity ↑ ε" rules — this doubles as the
   required interpretability narrative.

**Expected impact ranking:** (3) pretraining ≈ (2) multi-task > (1) physics features ≈ (4) PG fingerprint
> (5) pseudo-labeling > (6) consistency. Aim first at a fused {physics features + ECFP + Mordred +
PG-fingerprint + from-scratch embedding} model with a multi-task head; that combination is what the
literature (polyBERT/PolyCL/TransPolymer/polyGNN + Khazana) collectively says wins on exactly these targets.
