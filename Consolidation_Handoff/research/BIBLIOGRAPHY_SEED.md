# BIBLIOGRAPHY_SEED.md — citation ledger (started 2026-08-31)

**Purpose.** The judges explicitly expect architecture choices to be backed by literature
("Ensure that you have a good understanding of your proposed architecture and have some
proof like a published research to back it up" — `Personal/Obsidian/Presentation.md`).
This file is the seed. It is **not complete** — §6 lists the exact searches the next agent
must still run.

**Rule:** every entry must be tied to a *decision we actually made*. A citation that does
not defend a design choice is padding and should be cut.

---

## 1. Already collected in-repo (DO NOT redo this work)

`Phase5_Kiro_Score_Improvement/REFERENCES.md` (561 lines) already contains structured
entries with key insights + implementation notes for:

| Topic | Citation |
|---|---|
| Chemical LM for polymers | Kuenneth et al., **polyBERT**, *Nature Communications* 14:4099 (2023) — https://pubmed.ncbi.nlm.nih.gov/37433807/ |
| SMILES BERT baseline | Chithrananda, Grand, Ramsundar, **ChemBERTa**, arXiv:2010.09885 (2020) |
| SMILES-BERT | Wang et al., ACM-BCB (2019) |
| **Multi-task polymer informatics** | Kuenneth et al., *Patterns* 2(4):100238 (2021) — https://pubmed.ncbi.nlm.nih.gov/33982028/ — **this is the source of our six DFT targets** |
| Physics-informed NN | Raissi, Perdikaris, Karniadakis, *J. Comput. Phys.* 378:686 (2019) |
| D-MPNN | Yang et al., *JCIM* 59(8):3370 (2019) |
| GIN | Xu et al., ICLR (2019) |
| GATv2 | Brody, Alon, Yahav, ICLR (2021) |
| Tanimoto/graph kernels | Ralaivola et al., *Neural Networks* 18(8):1093 (2005) |
| KRR for molecules | Rupp et al., *PRL* 108:058301 (2012) |
| **Group contribution for Tg** | Bicerano, *Prediction of Polymer Properties*, 3rd ed., Marcel Dekker (2002) |
| **Polymer Genome** | Kim et al., *IJMS* 19(9):2809 (2018); also *J. Phys. Chem. C* (2018) |
| **Free volume / chain length** | Fox & Flory, *J. Appl. Phys.* 21:581 (1950) |
| Stacking | Breiman, *Machine Learning* 24(1):49 (1996) |
| Isotonic calibration | Zadrozny & Elkan, KDD (2002) |
| **SMILES enumeration / augmentation** | Bjerrum, arXiv:1703.07076 (2017) — https://arxiv.org/abs/1703.07076 |
| Scaffold splits | Bemis & Murcko, *J. Med. Chem.* 39(15):2887 (1996) |
| ECFP / Morgan | Rogers & Hahn, *JCIM* 50(5):742 (2010) |
| Mordred descriptors | Moriwaki et al., *J. Cheminform.* 10:4 (2018) |

## 2. Additional live-verified links found in this session

| Claim it defends | Source |
|---|---|
| polyBERT full text / abstract | https://pubmed.ncbi.nlm.nih.gov/37433807/ |
| TransPolymer (transformer LM for polymer properties) | https://www.nature.com/articles/s41524-023-01016-5 |
| PolyCL (contrastive polymer representation learning) | https://arxiv.org/html/2408.07556v1 |
| Multi-task polymer informatics (our target set's origin) | https://pubmed.ncbi.nlm.nih.gov/33982028/ |
| SMILES enumeration as augmentation | https://arxiv.org/abs/1703.07076 |
| Conformal prediction in cheminformatics — current applications & challenges (review, 2025) | https://www.sciencedirect.com/science/article/pii/S2667318525000030 |
| Adaptive conformal prediction for QSAR UQ | https://pubs.acs.org/doi/full/10.1021/acs.chemrestox.6c00065 |
| Physics-enforced NN on polymer-genome fingerprints (Ramprasad group, 2026) | https://ramprasad.mse.gatech.edu/wp-content/uploads/2026/01/AEM-PENN-degradation-JPC2026.pdf |

## 3. Already-cited-in-our-own-paper (Round 2 draft, on the GPU laptop)

`vishwa@100.116.22.29:~/Desktop/AISEHack-2.0/Polymer_Research_Paper/` — the LaTeX build
(`latex/paper.tex` → `paper.pdf`, 8 pages, IEEEtran conference class) has **23 live-verified
references** including the meta-science replication literature that backs our "negative
results are evidence" framing:

Klein et al. (Many Labs, *Social Psychology* 45(3), 2014) · Open Science Collaboration
(*Science* 349:aac4716, 2015) · Silberzahn et al. (*AMPPS* 1(3), 2018) · Kapoor & Narayanan
(*Patterns* 4(9):100804, 2023) · Pineau et al. (*JMLR* 22(164), 2021) · Bouthillier et al.
(MLSys 2021) · Schaeffer et al. (arXiv:2506.19882, 2025) · Karl et al. (arXiv:2406.03980,
2024) · Praski et al. (arXiv:2508.06199, 2025) · Eraqi et al. (*Commun. Chem.* 2025,
doi 10.1038/s42004-025-01592-1) · Ben Hicham et al. (arXiv:2604.16123, 2026) ·
Krstajic et al. (*J. Cheminform.* 6:10, 2014) · Agrawal (*MRS Commun.* 2026,
doi 10.1557/s43579-026-00948-5) · Gelman & Loken "garden of forking paths" ·
polymer-chemprop · PolymerGNN · PolyMetriX · PolyMon · RadonPy.

**Action for next agent:** SCP the whole `Polymer_Research_Paper/` tree to
`Personal/Research_Paper/` and lift the verified reference list verbatim — it is already
link-checked, and re-deriving it is wasted effort.

## 4. Mapping citations → our design decisions (this is what goes in the report)

| Our decision | Backed by |
|---|---|
| Per-target models, not one shared model | Kuenneth 2021 (multi-task helps *sparse* targets but our variance analysis shows unnormalised joint loss is degenerate); our own §3 EDA |
| Tree/kernel ensembles over GNNs at n≈222 | Yang 2019 (D-MPNN needs ~1k rows), Grinsztajn et al. "Why do tree-based models still outperform deep learning on tabular data" (NeurIPS 2022) — **must add this citation** |
| Tanimoto KRR / GPR for the small DFT targets | Ralaivola 2005, Rupp 2012 |
| `eps = nc² + ionic` decomposition | Maxwell relation n²=ε_optical; DFPT electronic+ionic permittivity split; verified 0/134 violations |
| `Egc = Ei − Eea` band-edge identity | standard band-structure definition; verified R²=0.9716 |
| Flory–Fox 1/n test as an explainability probe | Fox & Flory 1950 |
| Randomised-SMILES augmentation for invariance | Bjerrum 2017 |
| Conformal intervals on every test row | Vovk cross-conformal (2015); Angelopoulos & Bates "Gentle Introduction to Conformal Prediction" (2021) — **must add**; cheminformatics review S2667318525000030 |
| Applicability-domain tiers by nearest-train Tanimoto | Sahigara et al. / OECD QSAR AD guidance — **must add** |
| SHAP as the attribution method | Lundberg & Lee, NeurIPS (2017); Lundberg TreeSHAP *Nat. Mach. Intell.* (2020) — **must add** |
| Fidelity-by-masking as the faithfulness check | Hooker et al. "ROAR: A Benchmark for Interpretability Methods" NeurIPS (2019) — **must add** |
| Linear probes on hidden layers | Alain & Bengio, arXiv:1610.01644 (2016) — **must add** |
| Activation patching / causal tracing | Meng et al. ROME, NeurIPS (2022) — **must add** (used as method inspiration only) |
| Scaffold + low-similarity splits as honest generalisation tests | Bemis & Murcko 1996; Wu et al. MoleculeNet (2018) — **must add** |

## 5. Explicit "we tested the textbook and it failed" citations (strong QnA material)

| Relation | Reference | Our measured outcome |
|---|---|---|
| Lorentz–Lorenz / Clausius–Mossotti | standard optics/dielectrics texts | worse than plain nc² (0.797 vs 0.844) — ε_ionic is additive in ε, not in the CM function |
| Moss (`n⁴Eg ≈ 95`), Ravindra, Penn gap–index relations | Moss (1950), Ravindra (1979), Penn (1962) — **add full citations** | constant transfers as ≈54.5 not 95; imposing it cost nc −0.137 |
| ML residual on the Ei/Eea identity | — | LOO R² = **−0.82**; the bare affine identity is better than any learned correction |

## 6. Searches the next agent MUST still run (I did not)

Run these and add live-verified links + one-line "what it defends":

1. `Grinsztajn Oyallon Varoquaux why do tree-based models outperform deep learning tabular NeurIPS 2022`
2. `Lundberg Lee SHAP unified approach interpreting model predictions NeurIPS 2017` and `Lundberg TreeSHAP local explanations to global understanding Nature Machine Intelligence 2020`
3. `Hooker ROAR benchmark interpretability remove and retrain 2019`
4. `Angelopoulos Bates gentle introduction conformal prediction distribution-free uncertainty`
5. `Vovk cross-conformal predictors 2015`
6. `applicability domain QSAR OECD principles Tanimoto distance to model definition`
7. `Alain Bengio understanding intermediate layers using linear classifier probes`
8. `NeurIPS 2025 Open Polymer Prediction Challenge winning solution writeup` ← **high value; the Round-3 themes clearly echo this challenge**
9. `polymer chemprop weighted directed message passing periodic polymer graph Aldeghi Coley`
10. `PolyMetriX polymer benchmarking framework` and `PolyMon polymer property benchmark`
11. `Bicerano group contribution glass transition prediction accuracy review`
12. `dielectric constant polymer machine learning ionic electronic decomposition DFPT Ramprasad`
13. `refractive index polymer machine learning Lorentz-Lorenz limitations`
14. `negative transfer multi-task learning materials property prediction`
15. `uncertainty quantification deep ensembles MC dropout molecular property regression calibration`

**Storage convention:** write each into `Personal/Research/` as
`Personal/Research/<topic>/<firstauthor><year>-<slug>.md` with fields
`title / authors / venue / year / url / verified_on / what_it_defends / key_numbers /
how_we_use_it / caveats`, and maintain `Personal/Research/INDEX.md` as the master table
plus `Personal/Research/CITATIONS.bib` for the report/paper.
