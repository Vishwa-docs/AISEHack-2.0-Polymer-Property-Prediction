# POLYMER_DOMAIN_PRIMER.md — the science behind the seven targets

Written 2026-08-31 for QnA preparation. Everything here is *domain background* the judges
(Rohit Batra's group at IIT Madras — polymer informatics specialists, i.e. the people who
built Polymer Genome/Khazana) can and will probe. Numbers tying back to our data are
cross-referenced to `EDA_VERIFIED_FACTS.md`.

---

## 1. What a polymer is, and why SMILES for a polymer is different

A polymer is a long chain built by repeating a **repeat unit** (monomer residue) n times.
In this dataset each row is a **repeat unit written in "polymer SMILES" (PSMILES) form**:
a normal SMILES string with **two `*` wildcards** marking where the unit bonds to the
previous and next unit. Verified: **100% of the 12,349 strings carry exactly 2 stars.**

Consequences that matter for modelling:

1. **The molecule is not the material.** RDKit will happily parse `*CCc1ccccc1*` as a
   molecule with two radical/dummy atoms, but the physical object is an infinite chain.
   Descriptors computed on the *capped monomer* (replace `*` with H) are systematically
   biased at the endpoints; descriptors on the *ring-closed* form (bond the two stars) or
   on an *oligomer* (2–4 repeats) are better proxies for the chain. Our pipeline builds
   all three views ("capped / ring / dimer / trimer oligomer descriptors"), which is why
   the model is robust to endpoint artefacts.
2. **Cut-point degeneracy.** The same chain can be cut at different bonds, giving several
   *different but equivalent* PSMILES. This is a **polymer-specific invariance** on top of
   the ordinary SMILES-writing invariance — and it is exactly what Round 3 means by
   "invariant to different valid representations of the same polymer structure".
3. **Chain length is unspecified.** Molecular weight, tacticity, crystallinity, branching
   and processing history all move real Tg by tens of degrees but are **absent from the
   input**. This is an irreducible noise floor for Tg (see §3) and is the honest answer to
   "why is your Tg R² not 0.99?".

## 2. The seven targets, grouped by physics

### Group A — one experimental, thermal property

**Tg — glass transition temperature (°C).**
Below Tg the amorphous chains are kinetically frozen (glassy, brittle); above it, segmental
motion unlocks (rubbery). It is *not* a thermodynamic phase transition — it is a kinetic
freezing point, which is why it depends on cooling rate and measurement method (DSC vs DMA
vs dilatometry can differ by 5–20 °C on the same material).

Structure→Tg drivers (this is the causal story to tell over SHAP plots):
- **Backbone rigidity ↑ ⇒ Tg ↑.** Aromatic rings, imide/amide linkages, fused rings and
  double bonds in the main chain raise the rotational barrier. (Polystyrene ~100 °C,
  polycarbonate ~150 °C, polyimides 300–400 °C.)
- **Backbone flexibility ⇒ Tg ↓.** Ether (–O–), methylene (–CH₂–) and siloxane (–Si–O–)
  linkages rotate cheaply. (Polyethylene ≈ −120 °C, PDMS ≈ −125 °C.)
- **Intermolecular forces ↑ ⇒ Tg ↑.** H-bonding (amide, urethane, –OH), strong dipoles
  (nitrile, sulfone), polarity — all restrict chain slip. (Nylon-6 ≈ 47 °C vs PE ≈ −120 °C.)
- **Bulky, stiff side groups ⇒ Tg ↑; long flexible side groups ⇒ Tg ↓ (internal
  plasticisation).** PMMA ≈ 105 °C vs poly(n-butyl methacrylate) ≈ 20 °C.
- **Free volume.** Simmha–Boyer / Fox–Flory: Tg tracks the temperature at which fractional
  free volume falls to ≈0.025. Bulky groups that pack badly *increase* free volume and
  *lower* Tg; efficient packing raises Tg.
- **Chain length: Flory–Fox**, `Tg(n) = Tg∞ − K/Mn`. Linear in 1/n. **We use this as an
  explainability probe** — feeding the model oligomers of increasing n and checking the
  predicted Tg is linear in 1/n (median R² ≈ 0.99 in `relation_homologous_series.csv`).
  That is a *falsifiable physics test*, not a decoration.

**Why Tg is the noisy one:** PolyInfo-derived experimental values carry measurement,
sample-preparation and molecular-weight variability. Published polymer-informatics work
typically reports Tg R² in the 0.85–0.92 band on comparable splits, which is why we quote a
**practical ceiling of ≈0.92** rather than claiming more.

### Group B — four DFT electronic-structure properties

All computed with density functional theory (Ramprasad group / Khazana), so they are
**deterministic functions of the structure** — no measurement noise, only data scarcity.

- **Egc — chain bandgap (eV):** HOMO–LUMO gap of an isolated, periodic single chain.
- **Egb — bulk bandgap (eV):** the same for the 3-D packed bulk. Bulk < chain in general
  because interchain interaction broadens the bands. Empirically on our data
  **Egb = 1.1586·Egc − 1.0437, R² = 0.9282 (n=175)** — a strong but not exact affine map;
  the residual is interchain packing, which the SMILES cannot fully express (this is why the
  `dgap = egb − egc` coordinate was weak, TRIALS.md §1).
- **Ei — ionisation energy (eV):** `Ei = E_vac − E_VBM` (energy to remove an electron).
- **Eea — electron affinity (eV):** `Eea = E_vac − E_CBM` (energy released on adding one).
- **The identity:** `Egc = Ei − Eea` (the gap is the distance between the two band edges).
  Verified on our train data: **R² 0.9716, MAE 0.072 eV, n=59.** It is not exact because Ei
  and Eea are reported for the *bulk/chain* references with slightly different corrections —
  the residual bias is +0.044 eV.
- **Useful reparametrisations:** `chi = (Ei + Eea)/2` is the **Mulliken electronegativity**
  (the *centre* of the gap) and is nearly orthogonal to the gap *width*; modelling
  (chi, gap) instead of (Ei, Eea) decorrelates the problem.

**Chemistry:** gaps are set by **π-conjugation length and heteroatom substitution**.
Extended conjugation (long polyene, fused aromatics, thiophene) delocalises electrons →
narrow gap, higher Eea. Electron-withdrawing groups (–F, –CN, –NO₂, C=O) pull both band
edges down → **higher Ei and higher Eea**. Saturated aliphatic backbones → wide gap (>6 eV),
insulating.

### Group C — two DFT optical/dielectric properties

- **Nc — refractive index** (optical, high-frequency: only electrons respond).
- **EPS — dielectric constant** ε_r (static: electrons **plus** ions/dipoles respond).
- **The exact decomposition:** static permittivity = electronic + ionic contributions, and
  the electronic part is `n²` (Maxwell's relation). Hence
  `eps = nc² + ε_ionic`, with **ε_ionic ≥ 0 by construction.**
  Verified on our train data: **0 violations in 134 co-measured pairs**, ionic median 0.6896,
  and — crucially — **std(ionic) = 0.409 vs std(eps) = 1.070**, i.e. the physics coordinate is
  **2.62× better conditioned**. Predicting ionic and reconstructing eps is therefore an
  easier regression problem than predicting eps directly. This is the single strongest
  weak-target mechanism we have (Round 2: eps 0.7835 → 0.8501, +0.0666).
- **Why ionic is chemically predictable:** it is driven by *polar group density* —
  C–F, C=O, ester, ether, –OH, nitrile, amide, N–H, sulfone, thioether, imide, siloxane,
  P=O, urethane, plus TPSA/HBD/HBA. A 26-feature polar-group block on ExtraTrees beats
  adding 512 Morgan bits (which actively *hurts* by 0.004–0.006) — a nice "domain knowledge
  beats brute force" slide.
- **Lorentz–Lorenz / Clausius–Mossotti** relate n and ε to molar polarisability and density.
  We tried them; they **underperform plain n²** (0.797 vs 0.844) because ε_ionic is additive
  in ε, *not* in the CM function. Similarly **Moss / Ravindra / Penn** gap–index relations
  (`n⁴·Eg ≈ const`) do not transfer with the literature constant (we measure ≈54.5, not 95)
  and cost nc −0.137 when imposed. **Good QnA material: we tested the textbook relations,
  measured that they fail here, and can say why.**

## 3. Why the seven targets need different architectures (the core narrative)

| | Tg | Egc | Egb / Ei / Eea / Nc / EPS |
|---|---|---|---|
| origin | experiment (PolyInfo-like) | DFT | DFT |
| train n | 4,143 | 2,028 | 221–337 |
| label noise | **real** (method, Mw, processing) | none | none |
| test polymer in train (any target) | **12.3%** | 37.2% | **88–99%** |
| test polymer in train (same target) | 0.1% exact | 0% exact | **0% exact** |
| physics identity available | none | Egc = Ei − Eea | yes (all) |
| ⇒ correct approach | structure→property learning, ensemble + residual, tail-robust | ensembles + partner covariates | **cross-property / physics reconstruction + small-n kernels** |
| ⇒ wrong approach | small-n kernels | — | deep nets, more fingerprint bits, GNNs |

Two supporting empirical laws we can cite:
- **Trees beat deep learning on small tabular/molecular data.** Our own C043 directed-MPNN
  scored Ei **−0.309** (0/5 folds). Literature crossover: weighted D-MPNN only overtakes
  random forests at roughly **859–1000 training rows**; below that, tuned tree/kernel
  ensembles win. With n=222 for Ei we are 4× below the crossover.
- **Kernels beat trees at very small n.** Tanimoto KRR / GPR interpolate smoothly where
  trees saturate into a handful of leaves (a tree with min_samples_leaf=2 on 222 rows has
  ~111 effective cells). This is why the Ei/Eea leaves in V57 are MLP + GaussianProcess +
  Tanimoto-KRR, not deeper boosting.

## 4. Where the research literature is thin (the "gap" slide)

1. **Multi-property polymer datasets are sparse and heterogeneous by construction** —
   properties are measured on overlapping but non-identical structure sets. Most published
   pipelines either drop incomplete rows or train independent single-task models; the
   *masked-loss multi-task* option (Kuenneth 2021) is known to help sparse targets, but the
   literature does not address the case we actually face — **the partner label exists for
   the test polymer at inference time**. Exploiting test-time partner availability with a
   cross-fitted, guarded reconstruction is genuinely under-explored.
2. **Explainability in polymer informatics is almost entirely global feature importance.**
   Papers show SHAP/LIME rankings and stop. Almost nobody reports a **fidelity test**
   (does masking the top-SHAP features actually destroy performance more than masking random
   ones?), **attribution invariance** (do the explanations survive rewriting the SMILES?),
   or **mechanistic probes** (linear probes / activation patching). Our Phase-4 suite does
   all three — that is the defensible novelty claim.
3. **Representation invariance is asserted, not measured.** "We canonicalise, therefore we
   are invariant" is the standard sentence. But any model containing a *string* feature
   (char n-grams, SMILES transformers) is **not** invariant, and nobody quantifies the
   residual. We measure it: graph-feature std ≤0.23% of train std, full-ensemble 6–14% std
   with 1σ violation rate 0.1–1.5%. **Reporting the honest non-zero number is the novelty.**
4. **Self-supervised pretraining on unlabeled polymer corpora is reported as a success in
   the literature** (polyBERT on ~100M hypothetical polymers, TransPolymer, PolyCL) **but
   we could not reproduce a benefit at competition scale.** 9+ independent representation-
   pretraining variants on PI1M (995k) and smile_r3 (5.97M) all lost to a matched supervised
   control. That is a publishable negative result and it is honest: our corpus/compute
   budget is 100× below polyBERT's, and MLM-probe quality (0.651) actually came in *below*
   a random-init control (0.708) — the strongest possible kill signal.
5. **Uncertainty for regression on tiny chemical datasets is unsolved.** Ensemble spread
   correlates with error only weakly here (ρ 0.13–0.44). Conformal prediction gives valid
   *marginal* coverage but on ~45 calibration rows the coverage estimate itself has ±4.5%
   sampling noise — a limitation we can state precisely instead of hiding.

## 5. Ten questions a judge will ask, with the one-line answer

1. *"Why not a GNN?"* → n=222 for Ei; measured −0.309 R²; literature crossover ~859–1000 rows.
2. *"Why is Tg your worst-understood target?"* → it is the only experimental one, and 87.7%
   of its test polymers never appear in train under any property.
3. *"Isn't 98% of your test set in the training file? Isn't that leakage?"* → the *polymer*
   is, the *label* is not: **0 exact (SMILES,target) pairs** for all six DFT targets. What we
   exploit is physics between properties, which is legitimate and is what a materials
   scientist would do.
4. *"Why per-target models instead of one multi-task model?"* → Tg carries 99.986% of the
   pooled variance; an unnormalised joint loss is a Tg model with six decorative heads.
5. *"How do you know your SHAP explanations mean anything?"* → fidelity test: masking the
   top-10% SHAP features drops R² by 0.85; random masking drops it 0.04.
6. *"How do you know the model isn't reading the string?"* → 500 polymers × 30 randomised
   SMILES; graph-feature prediction std ≤0.23% of train std, attribution cosine 0.95–0.99,
   and we report the string-feature component's residual 6–14% instead of hiding it.
7. *"Can it extrapolate?"* → generalisation ladder: random CV → group → scaffold → family →
   low-similarity → ultra-low-similarity, monotone smooth decay, no cliff.
8. *"What is the ceiling?"* → perfect Tg alone gives 0.9172; the practical Tg ceiling is
   ≈0.92 from label noise; therefore the mean is bounded near ≈0.93–0.94 without new data.
9. *"Did the 5.97M-molecule auxiliary dataset help?"* → no, and we can show 9+ controlled
   attempts and a probe that scored *below* random initialisation.
10. *"What would you do with more time?"* → see `REMAINING_EXPERIMENTS.md`: atom-level
    tokenised MLM at true scale with GBM heads, MC-dropout/deep-ensemble UQ, and a
    cut-point-invariant featuriser that removes the last non-invariant component.
