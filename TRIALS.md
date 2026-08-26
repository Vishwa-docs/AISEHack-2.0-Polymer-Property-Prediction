# TRIALS.md — Polymer Property Prediction: catalog of everything tried in Rounds 1-2 (Round 3 reference)

**Purpose:** exhaustive reference of every technique/approach tried in Round 2, for a new round and for a presentation. Read-only synthesis; no files were modified.
**Task context:** unweighted mean of per-target R² across 7 properties: `tg`, `egc`, `egb`, `ei`, `eea`, `nc`, `eps` (4,940 test rows; 7,409 train rows; 6,171 archive rows; 995,799-row PI1M auxiliary). Six of seven targets are the Ramprasad-group Khazana DFT sets (Kuenneth et al., *Patterns* 2021); only `tg` is experimental.
**Score lineage:** C001 reference OOF 0.8658 (public 0.859) → C050 incumbent OOF 0.8731 → public incumbent **0.916** (C143+C144+C148 composite) → clean composite C257 0.8942 → best clean arithmetic **archive 0.9343 (C1579) / no-archive 0.9028 (C1572)** → best oracle-assisted diagnostic **0.9506 (no-archive C1565)**. Final one-run notebooks scored V46=0.9141 / V47=0.9111 (with archive), V48=0.8357 / V49=0.8336 (without).
**Notation:** `helped` = documented gain; `hurt` = regression; `neutral` = no change; `subthreshold` = positive but below the +0.01 promotion gate; `rejected_gate` = failed a transfer/robustness gate; `runtime_invalid` = no scientific evidence produced.

---

## 1. Domain Knowledge & Physics (formulas, identities, band-structure)

- **`eps = nc² + ε_ionic` (ε_ionic ≥ 0) identity** — DFPT decomposition; 0 violations in 134 official pairs; model the ionic term then reconstruct eps/nc. C162, C187/C190, C214, F02, C350. *Helped — single biggest EPS/Nc lever; C214 EPS +0.0666, C252 Nc +0.0434.*
- **`ei = egc + eea` band-edge identity** — `ei = E_vac − VBM`, `eea = E_vac − CBM`; n=92, R² 0.928/0.955. C041, C048, C049, F01, C349. *Helped on observed-partner rows; but ML residual on the identity HURTS (LOO R² = −0.82).*
- **`egb ≈ a·egc + b + residual`** — chain vs bulk gap; n=268 affine R² 0.9205, +ExtraTrees residual (α=1.0) → 0.9478. C005, C063, C072, F01. *Helped for egb specifically (the only identity where ML residual helps).*
- **`chi = (ei+eea)/2` gap-centre coordinate** — Mulliken electronegativity; corr(chi,Egc)=−0.124; Stage-1 OOF 0.8595. C146, Claude Stage-1. *Helped as reparametrization; independent of gap width.*
- **`ionic = eps − nc²` coordinate** — std 0.41 vs eps std 1.09 (better conditioned). F02, C144. *Helped; raw ExtraTrees on 26 polar features ionic R² 0.690.*
- **`dgap = egb − egc` coordinate** — Stage-1 OOF 0.5070. *Neutral/weak — interchain packing not recoverable.*
- **Lorentz–Lorenz / Clausius–Mossotti transform** — R² 0.797 vs plain `Nc²` 0.844. C046, C052, C099, C126. *Hurt/neutral — ε_ionic is additive in ε, not in the CM function; do not use.*
- **Moss / Ravindra / Penn gap–index relations** — transfer in form not constant; `n⁴·Eg`≈54.5 (not 95); use Egb not Egc. C210. *Hurt — Nc −0.137 (C210).*
- **Periodic tight-binding / Hückel band structure** — `H(k)=H0+H1·e^{ik}+H1†·e^{-ik}` on k-grid; π-only Hückel and all-heavy-atom variants. Claude §2.3. *Helped as raw features — π-Hückel CBM corr −0.791 with eea, VBM −0.718 with ei.*
- **Extended Hückel theory (EHT) orbital/charge residual** — RDKit/YAeHMOP; deterministic endpoint-capped repeat units. C258, C268, C269, C270, C283, C374, C381. *Feasibility +0.0072 over C050 (4/5 folds) but never banked; transfer repeatedly failed.*
- **Flory–Fox oligomer carrier for Eea** — C180, C189 (independently confirmed). *Helped — Eea 0.9008→0.9163, banked.*
- **Tg mobility / free-volume carrier** — C006, C015. *Hurt — portable Round-1 Tg carrier regressed.*
- **Tg backbone/pendant rigidity** — C254. *Subthreshold (+0.0016) and negative panel.*
- **Backbone/pendant polarizability partition (Nc)** — C236. *Neutral (+0.00013).*
- **Bond-polarity orientational residual (EPS)** — C238. *Hurt (−0.00045 vs selected C214 ref).*
- **Electro-polar topological autocorrelation** — C182, C220 (Ei), C240 (Nc). *Hurt — family closed.*
- **3D shape / free-volume residual (Tg)** — C061, C062. *Neutral (+0.0002).*
- **Endpoint conjugation residual** — C063 (Egb +0.0007), C065 (Eea +0.0012), C067 (fold-masked, +0.0). *Subthreshold, not banked.*
- **Charge / partial-charge residuals** — C074 (Ei nested charge +0.0124, best-ever weak-target point gain), C075, C078, C079, C085. *Positive point gains but failed support/bootstrap gates.*
- **`eps ≥ nc² + 0.02` assembly projection** — F02/F09. *Helped as physics consistency at assembly.*
- **Ionic-coordinate ensemble** — Ridge/ExtraTrees/HistGB over ionic coords. C162. *Helped — EPS +0.0157, banked.*
- **Ionic full-amplitude route** — C214 (blend amplitude 0.50→1.00). *Helped — EPS 0.7835→0.8501 (+0.0666), banked.*
- **Ionic EPS→Nc projection** — C252 (selected-EPS ionic projection to Nc). *Helped — Nc 0.8397→0.8832 (+0.0434), banked.*
- **Joint EPS/Nc consistency solve (B3)** — predict (nc², ionic) jointly for co-test pairs. F02-B3, C350, C1284, C1360. *Mixed — helped NC (0.906–0.917) but often hurt EPS on transfer.*
- **Surrogate-NC ionic deployment** — C366/C402 (current-only EPS ionic + surrogate-NC). *Hurt on transfer — overcorrects EPS (0.829 vs 0.850).*
- **Optical-dispersion gap (Nc)** — C210. *Hurt strongly (Nc −0.137).*
- **Coulomb-matrix / Lorentz–Lorenz paired route** — C126. *Neutral (zero strict paired support).*

---

## 2. SMILES / Token Representation Tricks

- **SMILES character n-gram hash features (2,7)** — C001/C282 baseline component. *Helped as part of the reference ensemble.*
- **Character n-gram Ridge (Eea)** — C025. *Rejected (component gate).*
- **Scratch SMILES char-CNN residual** — C057 (EPS/Nc +0.0224 point but wrong parent), C058 (exact-v7 −0.0019), C060 (EPS −0.0004). *Neutral/hurt once parent-corrected.*
- **Character TF-IDF on PI1M** — C010. *Hurt — regressed all 7 targets (−0.109 to −0.347); cooled the char-TFIDF family.*
- **Polymer long-repeat grammar** — C066. *Hurt (−0.0002).*
- **Endpoint / path n-gram residual** — C103. *Neutral (+0.00015).*
- **Structure-key OOF target encoding** — C032 (hierarchical). *Hurt (−0.012).*
- **Repeat-view invariance (1/2/3 repeat, reversal, recut)** — C277 (−0.0017), C278 (−0.0019), C927 (−0.0014), C340 wrapper (+0.0016 OOF but failed transfer). *Hurt/neutral — repeat invariance never transferred.*
- **Polymer views (capped/ring/dimer/trimer oligomer)** — C011, C086. *Rejected.*
- **Structure-semantics weak-target encoding** — C222. *Rejected (no banked target).*
- **PI1M subword Ridge** — C181. *Hurt badly (Ei 0.70, EPS 0.63, Nc 0.70); family closed.*
- **PI1M char-ngram SVD** — C284, C285, C391. *Neutral/hurt (SVD 96-dim reference below C282).*
- **Canonicalization / no-stereo keys** — EDA. *Neutral — ring-closure + stereo-strip yield 0 extra matches on all 5 weak targets (+2 tg only).*
- **Cut-point-invariant averaging / n-mer expansion** — Fable F03. *Proposed (not cleanly banked) — averaged continuous descriptors over all backbone cuts.*
- **SMILES enumeration** — Fable §4 (Bjerrum R² 0.56→0.66). *No-op for descriptor/fingerprint features (graph-invariant); only helps sequence models.*

---

## 3. Molecular Fingerprints & Descriptors

- **RDKit descriptors (217)** — C001/C282 baseline. *Helped (core of reference).*
- **Morgan count/bit fingerprints r=1/2/3** — baseline; radii r2/r3 in C282. *Helped (core).*
- **MACCS bits** — C102 and later model zoos. *Neutral (part of zoo; C102 minimal stack −0.0005).*
- **AtomPair / TopologicalTorsion** — C026, C071. *Hurt (C071 Nc −0.011).*
- **Physical/count features (16)** — baseline + Claude 114 physics columns. *Helped.*
- **Polymer-topology capped/ring/dimer/trimer oligomer descriptors** — Claude Stage-0. *Helped (diverse family).*
- **Polymer Genome atomic-triple fingerprint (664 keys, `O1-C3-C4` coordination-labelled)** — Claude §2.4, F03, C279, C340. *Helped — egb 0.9167→0.9259, nc 0.8438→0.8519 with nothing else changed.*
- **Polymer Genome morphological block** (ring–ring topo distance, side-chain fraction, largest side chain) — F03/C279. *Helped in literature (15 K on Tg); clean transfer weak.*
- **Electro-topological / E-state absolute** — C027. *Rejected.*
- **Graph degree spectrum** — C064. *Hurt (−0.0012).*
- **Fragment path kernel (typed length-2/3)** — C188. *Hurt — all active targets declined slightly; family closed.*
- **Periodic vs non-periodic WL kernel** — C183. *Neutral (+0.001 Ei, +0.003 EPS, +0.001 Nc) but negative bootstrap; closed.*
- **Graph grammar + HistGB** — C097. *Hurt (−0.0012).*
- **Tanimoto landmark residuals** — C113. *Hurt (−0.0018).*
- **Rich sparse fingerprint refresh** — C101. *Neutral.*
- **Compact QSPR + RBF** — C045. *Hurt badly (EPS R² −1.05, overfit).*
- **Rank-2 PCA + Bayesian Ridge (Eea)** — C030. *Hurt (−0.283).*
- **Tanimoto landmarks (256)** — C113. *Hurt (−0.0018; 256 landmarks > per-fold rows).*
- **Capped/periodic/backbone-side-chain HGB arm (892 feats)** — C011. *Hurt (regressed 6 targets).*
- **Mordred / trimer / 3D / AutoGluon sweeps** — Round-1 cooldown. *Hurt/redundant (R1).*

---

## 4. Feature Engineering (substructure, bond/angle, topology)

- **26 polar-group density features** (C–F, C=O, ester, ether, OH, nitrile, amide, N–H, sulfone, thioether, aromatic N/O/S, imide, siloxane, P=O, urethane, TPSA/HBD/HBA/FractionCSP3/rotatable/MR/logP) — Fable §3.3. *Helped — beats adding 512 Morgan bits (which HURTS −0.004); drives ε_ionic.*
- **Nested electronic residual** — C073 (Eea +0.0024). *Subthreshold.*
- **Nested charge residual** — C074 (Ei +0.0124). *Positive point gain; not banked.*
- **Charge-bank Ridge** — C075, C078, C079, C085, C155. *Positive point gains; failed support/bootstrap.*
- **Symbolic QSPR interactions** — C059. *Hurt badly (Ei −0.67).*
- **Replicate-reliability feature** — C232 (Tg +0.0094, 5/5 folds), C234 (Nc +0.005). *Subthreshold (missed +0.01, negative panel).*
- **Source-aware paired covariates** — C054. *Hurt (−0.019 mean).*
- **PI1M fragment-context PPMI/SVD embedding** — C117/C118/C119. *Hurt — lost to matched official-corpus control; family cooled.*
- **PI1M rarity/density features** — C185, C192. *Hurt — Eea/Ei/EPS/Nc all declined.*
- **PI1M substructure-context pilot** — C116. *Rejected pre-metric (non-nested risk).*

---

## 5. Per-Target Specialization (separate model per property)

- **Target-specific classical ensemble (per-target Ridge/ET/Tanimoto blend)** — C001/C282 baseline. *Helped — the fundamental winning pattern (beats any shared model).*
- **EPS/Nc paired-property specialist** — C003. *Hurt (−0.14 EPS, −0.10 Nc).*
- **Ei/Eea electronic specialist** — C004. *Hurt (Ei −0.059, Eea −0.046).*
- **Egc/Egb coupled specialist** — C005. *Hurt (Egb −0.004).*
- **Portable Round-1 Tg carrier** — C006. *Hurt (−0.013).*
- **Nc size/free-volume specialist** — C009. *Hurt (−0.11).*
- **Target-tree zoo** — C013. *Rejected.*
- **Target-routed QSPR** — C098. *Helped (+0.0017) but below gates — best early near-miss.*
- **Three-target route** — C019. *Candidate generated; not banked.*
- **Target-specific periodic graph** — C106. *Hurt (−0.00003).*
- **Round-1 target-specific screen** — C053. *Rejected.*
- **Structure-only shallow ExtraTrees (Nc)** — C083. *Neutral (+0.0012).*
- **EPS fully nested specialist** — C084. *Positive (+0.011) vs alternate parent only.*
- **Ei v7 charge specialist** — C085. *Neutral (+0.0014 best variant).*
- **Round-1 carrier factory** — C127. *Near-misses: Tg +0.0097, Egc +0.0097, Eea +0.0115, Ei +0.0092 — all failed ≥1 transfer gate.*
- **Weak-target model zoo** — C1433/C1502 (ridge+ET). *Hurt on transfer (verified ~0.84–0.89); cooled.*
- **Weak-target direct ridge (fast)** — C1369. *Hurt (0.805).*
- **C050 = C001 + Ei gap-identity (C049) + Eea gap-identity (C048)** — the "mixed-c001-gap-components" incumbent. *Helped — clean OOF 0.8658→0.8731 (+0.00696), no target loss; the durable clean incumbent.*

---

## 6. Cross-Property Learning (multi-task, residual stacking, transfer)

- **Cross-property covariates (other-label availability + values)** — C001/C282 baseline. *Helped — legitimately recovered Ei/Eea; ~60% test-time availability.*
- **Cross-target OOF residual stacks** — C031 (Eea), C033 (Egb), C092/C093/C094/C095/C096. *Mostly rejected (subthreshold or wrong parent).*
- **Low-rank cross-property calibration** — C030. *Rejected.*
- **Multitask-z** — C012. *Runtime (index bugs) — no science.*
- **Pooled multi-task LightGBM** — Claude §2.5, C087. *Neutral — standalone 0.76–0.85, NNLS gives ~0 weight; not the lever literature suggested.*
- **Masked low-rank multitask** — C091 (−0.009), C166. *Hurt.*
- **Structure-kernel multitask (EPS/Nc)** — C055. *Hurt (−0.066 entry-mask).*
- **Periodic-graph multitask** — C105. *Hurt.*
- **Concat-selector multitask network** — F05. *Hurt (proxy 0.714, rejected).*
- **Kuenneth multi-head vs concat-selector routing** — Fable §5. *Guidance: single-task for tg/egc, multitask for nc/eps/egb/ei.*
- **Identity routes (`ei=egc+eea`, `eea=ei−egc`)** — C048 (Eea gap-identity +0.0208, 4/5, bootstrap +0.0033) and C049 (Ei gap-identity +0.0279, 5/5) were **BANKED into C050**; later identity attempts C179 (+0.0111, scaffold bootstrap −0.005) and C204 (−0.108) were *not bankable.* *Helped (the only clean cross-property wins) — became the "gap-components" in C050.*
- **Identity robustness audit** — C176, C179. *Not bankable.*
- **Availability-gated residual stack** — C034, C035. *Rejected.*
- **Strict nested predicted-label residual** — C047. *Rejected.*
- **Availability paired heads (EPS/Nc)** — C076 (+0.0072), C077 (+0.0054). *Subthreshold (5/5 folds, positive bootstrap, but <0.01).*
- **Cross-property partner Ridge (Ei)** — C171 (+0.0128, nested), C172/C173/C174. *Promising point estimate; negative scaffold bootstrap; not bankable.*
- **Partner-conditioned ionic models** — C147. *OOF/transfer mismatch.*
- **Co-test partner joint solve** — F01/F02, F25/F26, C332/C333/C350. *Helped — C333 Eea→0.9473, EPS→0.8589; C332 Nc→0.9066; F25 archive 0.9171.*
- **Cross-property overlay (F23/F24)** — F23 archive 0.9223, F24 no-archive 0.8693. *Helped (archive), subthreshold (no-archive).*
- **Cross-property circularity bug (C132/C139/Claude §3.4)** — apparent OOF 0.935→actual transfer 0.907. *Hurt — must cross-fit partner fills and use Stage-1 fallback for missing partners.*

---

## 7. Ensembling & Stacking

- **Per-target OOF NNLS blend** — C001/C282 baseline. *Helped (core of reference).*
- **Dense + sparse Ridge + ExtraTrees + Tanimoto blend** — baseline family. *Helped; blend beats every single family by 0.02–0.05.*
- **Median residual stack (Tg signed-agreement)** — C244 (+0.0099). *Subthreshold (missed +0.01 by 0.00011).*
- **Signed compounds / iterative signed-source assembly** — C1434–C1483, C1581–C1596, C1551–C1565. *Diagnostic-only (oracle-selected) — reached 0.9506 no-archive / 0.9385 archive but 0 clean-replayable rows.*
- **Fine weight sweeps** — C355–C413, C1524–C1529. *Small target-level gains; bookkeeping-scale.*
- **Reflected sources (`2·base − source`)** — C415–C433, C1530–C1532. *Tiny gains (EI/EPS/NC +0.001–0.002 each).*
- **Target-leader splice** — C1534/C1535, C1577/C1579. *Helped (archive 0.9343) but oracle-observed.*
- **Fixed equal blends** — F12 (+0.003), F14 (+0.011), F15/F16 (worse), F18/F19. *Helped moderately; no sweep.*
- **Consensus microblend** — C1519/C1529 (EI/NC consensus, shrink). *Helped EI/NC, hurt EPS.*
- **Shrinkage portfolio assembly** — F09 (bagged greedy selection, shrink toward incumbent, cap K=5–8). *Proposed; converts subthreshold signal; not fully realized.*
- **Parent + weighted residual formula** — the standard component recipe. *Helped when parent exact and residual nested.*
- **Exact parent replay / parity (1e-13)** — acceptance discipline. *Helped prevent false gains (but OMP_NUM_THREADS bug caused C188-v2 invalidity).*
- **Rank-residual / extreme-residual sign classifier** — C164. *Hurt (Ei −0.0056).*
- **Motif-family residual shrinkage** — C165. *Neutral (Ei −0.0007, Nc +0.0043).*

---

## 8. Classical ML Models Tried

- **Ridge (sparse + dense)** — baseline + many residuals. *Helped (workhorse).*
- **ExtraTrees** — baseline; best for ionic term (min_samples_leaf=2). *Helped.*
- **HistGradientBoosting** — C086, C097, C129, C397. *Mixed; C097 graph-grammar hurt.*
- **LightGBM** — C289, C926, C925, zoo arms, deep LGBM for Egc (R1). *Helped for egb/ei/nc in Stage-1 blend; LGBM-only bank stalled.*
- **XGBoost** — C151. *Hurt (Ei declined to 0.887).*
- **CatBoost** — C129. *Neutral (no banked target).*
- **PLS** — C029 (scaffold-balanced), C110/C111 (compact PLS residual portfolio). *Hurt (−0.0002).*
- **Gaussian Process / KRR** — C090 (Nc GP −0.042), F04 (three variants, proxy 0.72). *Hurt/neutral.*
- **Tanimoto KRR/kNN** — baseline + C020 variants. *Helped for Tg (R1) and as blend member; variants rejected.*
- **RBF KRR** — Claude Stage-1 zoo. *Neutral (weakest member on several targets).*
- **Random Forest** — zoo arms, C289 stalled. *Neutral.*
- **Huber-loss arms** — C074 (+0.0124 ei), C286. *Positive point gain; robustness concern.*
- **Monotonic counterpart calibration** — C057. *Positive (+0.015) but entry-local overfit.*
- **Affine calibration** — C039, F00 (per-target OLS on OOF). *Guaranteed R² ≥ original; expected +0.002–0.008.*
- **Fixed spline GAM** — C114. *Hurt (−0.0038).*
- **SISSO / chemistry grammar** — C186. *Neutral (Eea +0.0027 negative bootstrap).*
- **Physical spline Ridge (Nc)** — C037/C038. *Rejected.*
- **Scaffold-abstaining gap identity** — C048/C049. *Component pass (ei/eea) only.*

---

## 9. Neural Architectures

- **Directed message-passing GNN** — C043-v2. *Hurt badly (Ei −0.309, 0/5 folds) — matches literature (negative R² at n<141).*
- **Graph-tree specialist** — C021. *Rejected.*
- **Periodic graph encoder** — C105 (shared, −0.0002), C106 (target-specific), C128 (fragment encoder). *Hurt/neutral; negative panels.*
- **Directed edge-conditioned MPN** — C108/C109. *Runtime-invalid (graph scope too large).*
- **Scratch char-CNN** — C057/C058/C060. *Neutral (see §2).*
- **GIN masked-atom encoder** — C170. *Hurt (Ei 0.888 vs parent 0.892).*
- **Scratch SMILES Transformer MLM** — C169. *Hurt (Ei 0.890 vs 0.892).*
- **Concat-selector multitask net** — F05. *Hurt (proxy 0.714).*
- **Literature crossover:** wD-MPNN overtakes RF only at 859–1000 rows; plain D-MPNN never beats RF at these sizes. *Guidance: do not build standalone GNNs for the 5 small targets.*

---

## 10. Self-Supervised Pretraining on PI1M / Auxiliary

- **Character TF-IDF** — C010. *Hurt.*
- **Fragment-context PPMI/SVD** — C117/C118/C119. *Hurt — lost to matched official-corpus control.*
- **Denoising functional-group bottleneck** — C131. *Neutral (EPS +0.0007).*
- **InfoNCE contrastive 50k / 250k** — C157 (+0.00013), C158 (+0.000009). *Hurt — scaling does not rescue a failing arm.*
- **Token-transformer MLM + probe** — C169. *Hurt.*
- **995k subword Ridge** — C181. *Hurt badly.*
- **Rarity/density features** — C185/C192. *Hurt.*
- **MLM frozen linear probe** — C261 (probe 0.651 vs random-init control 0.708, 0/7 targets). *Hurt — the decisive PI1M representation kill.*
- **Char-ngram SVD (96-dim)** — C284/C285/C391. *Neutral/hurt (below C282).*
- **RankUp pseudo-label ranking distillation** — F06 (proxy 0.813/0.753), C1445 (student beat teacher +0.011 but 3/5 folds, transfer collapsed). *Hurt — the "never-tried" mechanism also failed.*
- **Multi-view co-training** — F07. *Not run (F06 failed).*
- **PI1M full-model bank (LGBM/HGB)** — C984, C289. *Runtime-stalled.*
- **Meta-conclusion:** all 9+ representation-pretraining variants failed; PI1M unlabeled data gave no reliable transfer.

---

## 11. Validation & CV Strategy

- **5-fold grouped CV** (10-fold × 2 seeds for n<1200) — baseline. *Helped (standard).*
- **Canonical-structure grouping** — prevents same-structure leakage. *Helped.*
- **Scaffold / family holdout** — C002. *Helped catch transfer failure.*
- **Butina/Tanimoto-cluster folds** — C002. *Helped.*
- **Low-similarity bin panels** — nearest-train-Tanimoto slices. *Helped identify non-transferring gains.*
- **Availability simulation panels** — which partner labels exist at test time. *Helped (led to cross-property design).*
- **Group-bootstrap lower bound** — promotion gate. *Found to be unpassable at n=222 (±0.02–0.04 width); rejected real signal (C171, C242, C244).*
- **Shift-matched R²** — reweight OOF residuals to match test NN-similarity histogram. *Helped as decision metric (Fable); better than grouped OOF for selection.*
- **Exact parent replay / parity (1e-12/1e-13)** — acceptance. *Helped; thread-setting bug cost C188-v2.*
- **Fold-local / nested CV** — all partner fills and blend weights fitted inside folds. *Helped (fixes circularity).*
- **OOF-to-test gap calibration** — per-target gaps (ei −0.089, eps −0.046, nc −0.024). *Helped — OOF was optimistic on every weak target.*
- **Replicate reliability** — C159 (preflight stopped: zero replicates on 5 small targets), C232/C234. *No-op for weak targets.*
- **Promotion gate (+0.01, 4/5 folds, positive bootstrap, non-negative panels)** — *Hurt — filtered out signal; Fable recommends "shrink, don't reject."*
- **Deterministic sanitizer (float32 overflow)** — C000 repair: |x|>1e12 → median-impute. *Helped (fixed the initial reference crash).*

---

## 12. Similarity-Based Routes (read-across, NN, Tanimoto gating)

- **Tanimoto KRR/kNN local model** — baseline. *Helped for Tg/Eea.*
- **Similarity-gated route** — C017. *Rejected (gate).*
- **Direction-consistent route** — C018. *Rejected.*
- **Tanimoto variants** — C020 v1–v3. *Rejected.*
- **EPS graph similarity route (threshold <0.70)** — C022. *Component pass (fixed threshold).*
- **Nested EPS graph route** — C024. *Rejected (inner-fold threshold selection).*
- **Tanimoto landmark residuals** — C113. *Hurt.*
- **Read-across pi-spectrum residual (Ei)** — C130 (+0.000965, negative bootstrap/panel). *Hurt.*
- **Logistic abstention / read-across gate** — C168. *Hurt strongly (Ei 0.763).*
- **Near-miss bridge** — C120 (runtime), C121 (+0.0015). *Subthreshold.*
- **Near-miss stability ensemble (Nc)** — C242 (+0.0099, 5/5 folds, missed +0.01 by 0.00005). *Subthreshold.*
- **Tg/Egc light read-across overlay** — C324. *Rejected (changed 0 rows).*
- **Test-density weighted ridge/Tanimoto** — C1566/C1566B. *Hurt (best 0.889).*
- **Literature:** regressing a specialist's absolute error for gating fails (R²=−0.005); binary "which model wins" gate captures only ~40% of oracle gain. *Guidance.*
- **Median NN similarity:** weak targets sit at 0.55–0.57; only 7–13% of test rows have a ≥0.7 train analogue → explains OOF-to-test collapse. *Diagnosis.*

---

## 13. Calibration & Post-Processing

- **Log-target models** — C028 (EPS log-target ET), C144 (log(EPS−Nc²)). *Mixed — C144 was a useful carrier, but log(ionic) specifically HURTS (−0.02 vs raw).*
- **Clipping / range enforcement** — C005 repair. *Neutral.*
- **Ordinal residual (high-tail)** — C216 (EPS), C218 (Nc +0.005). *Subthreshold.*
- **Abstention / scaffold-abstaining routes** — C048/C049, C168. *Component pass only; logistic abstention strongly rejected.*
- **Per-target affine recalibration (OLS on OOF)** — C039, F00. *Guaranteed non-negative R² effect; expected +0.002–0.008.*
- **Monotonic counterpart calibration** — C057. *Positive (+0.015) but entry-local.*
- **Fold-masked conjugation calibration** — C067. *Neutral (+0.0).*
- **Robust-rank loss** — C212. *Hurt (−0.0047).*
- **Current-domain residual** — C256. *Hurt (Egb −0.107).*
- **Domain-classifier train-support subset** — C153. *Hurt (Ei 0.805).*
- **Co-test meta-calibrator** — C327 (Eea +0.0031 clean, transfer +0.0061), C332/C333. *Helped (archive NC/EEA/EPS banked).*
- **`eps ≥ nc² + 0.02` projection at assembly** — F02/F09. *Helped.*

---

## 14. Weak/Target Engineering (transforms, shrinkage)

- **Log transform of target** — C028, C144. *Mixed (see §13).*
- **Target encoding (structure-key OOF)** — C032. *Rejected.*
- **Shrinkage toward incumbent** — C140, C196, F09, C363/C375/C1524–C1529. *Helped (rescued transfer; e.g. C375 EI +0.821).*
- **Yeo–Johnson transform** — C129. *Neutral (no banked target).*
- **Target standardization / z-scoring** — pooled multitask. *Neutral.*
- **Reparametrization to chi/ionic/dgap** — Claude Stage-1. *Helped (better-conditioned than raw targets).*
- **Robust / Huber residual loss** — C074 (+0.0124), C286. *Positive point gain; revisit under strict protocol.*
- **Rank-loss** — C212. *Hurt.*
- **Flory–Fox shrinkage confirmation (Ei)** — C196 (+0.0102 but negative panel). *Subthreshold.*

---

## 15. Data Curation

- **Deduplication by canonical structure** — baseline. *Helped.*
- **Archive exact-lookup override** — C001 (2,445/4,940 rows; 1,645 tg + 804 egc). *Helped massively — Tg/Egc test scores 0.959/0.963; worth ~0.016 mean R².*
- **Conflict-aware mapping** — 6 conflicting tg groups (max 24 K spread); abstain from override on conflicts. *Helped (Fable addendum).*
- **Source-priority label aggregation** — C224. *Rejected (no banked target).*
- **Median grouping for Tg replicates** — Tg has 2,497 duplicate groups; median spread 0.0, max 24 K. *Neutral.*
- **Replicate reliability feature** — C232/C234. *Subthreshold.*
- **Source-aware paired covariates** — C054. *Hurt.*
- **Test-density weighted sources** — C1566. *Hurt.*
- **Archive re-split exploitation** — archive is a re-split source of Tg/Egc labels. *Helped (see override).*
- **Ring-closure / stereo-strip key merge** — Claude §4.4. *Neutral (0 extra weak-target matches).*

---

## 16. Explainability / Interpretability

- **Train/test covariate-shift audit** — C138 (domain classifier AUC 1.0 for all weak targets; reweighting rejected, no OOF benefit). *Neutral — confirmed distribution shift but reweighting didn't help.*
- **Train-only residual diagnosis** — C008 (highest-ratio slice Tg NN-sim<0.30, 25 rows, ratio 2.57 but high fold variance). *Helped narrow search; slice rejected as unstable.*
- **Feature–target correlation tables** — Claude §2.3 (π-Hückel CBM −0.791 eea, LL n_est +0.802 nc). *Helped prioritize physics features.*
- **Deterministic compound audits** — C193–C257 (track which components enter the compound). *Helped prevent contamination of the mean.*
- **Partner-observed vs partner-missing subset scoring** — Claude §3.4. *Helped detect the circularity bug (collapse of advantage = leak signature).*
- **Per-target published-ceiling comparison** — Fable §2. *Helped know when to stop (incumbent already beat published SOTA on eea/nc/eps).*

---

## 17. Robustness / Invariance Attempts

- **Repeat-view invariance** — C277/C278/C927. *Hurt (see §2).*
- **Cut-point invariance (ring-close + enumerate backbone cuts, average descriptors)** — Fable F03. *Proposed; not cleanly banked.*
- **SMILES augmentation invariance** — Fable §4. *No-op for descriptors.*
- **Identity robustness audit** — C176/C179. *Not bankable.*
- **Near-miss stability ensemble** — C242 (+0.0099). *Subthreshold.*
- **Signed-agreement / agreement-consistency gates** — C244, C250. *Subthreshold.*
- **Replicate-reliability robustness** — C232/C234. *Subthreshold.*

---

## 18. Hyperparameter Tuning Methodology

- **Ridge alpha grid** — dense 10, sparse 30 (C282 config). *Helped (fixed, not swept).*
- **Estimator counts** — ExtraTrees 160, min_leaf 2–3. *Fixed per config.*
- **min_samples_leaf for ionic ET** — 5 vs 2 (2 better, ionic R² 0.690 vs 0.609). *Helped (specific tuning mattered).*
- **GP noise floor ≥0.01–0.05 (standardized)** — F04. *Guidance (prevents interpolation).*
- **Tanimoto kernel numerics (PSD form, +1e-6, clamp, +1e-3 diagonal)** — F04. *Guidance.*
- **Blend weight sweeps** — C355–C413, C1524–C1529. *Bookkeeping-scale gains.*
- **Fold count / repeat count** — R=5 × K=10 averaged OOF (F00). *Guidance (less overfit than single-pass).*
- **`OMP_NUM_THREADS` overrides** — caused C188-v2 parity failure. *Hurt — never set thread env vars.*
- **Deterministic seeds** — fixed seed 2026. *Helped (reproducibility).*
- **Cooldown discipline (no retune after family fails)** — loop rule. *Helped prevent wasted compute.*

---

## 19. Compound Pipelines & Trees (sequential routing)

- **Sequential model trees / target routing** — C013, C098, C099. *Mixed — C098 +0.0017 near-miss.*
- **Target-routed QSPR full** — C098 (+0.0017). *Subthreshold.*
- **Lorentz–Lorenz routed full** — C099 (+0.0004). *Subthreshold.*
- **Round-1 anchored nonlinear** — C100. *Hurt (−0.0008).*
- **Nested QSPR endpoint stack** — C103/C104 (+0.00015). *Neutral.*
- **Residual portfolio (PLS/Ridge)** — C110/C111. *Hurt.*
- **Physical/electronic boosted absolute** — C129. *Neutral (EPS/Nc/Ei/Eea +0.004–0.007, all failed a gate).*
- **Compound component audit** — C193–C257. *Helped (assembled best clean composite 0.8942).*
- **Target-wise compound loop** — C291–C422 (each candidate contributes only the target it improves, then rescore). *Helped — drove archive 0.9028→0.9343.*
- **Source stacker (fold-local)** — C1536–C1539. *Hurt (train OOF not predictive of test).*
- **Co-test meta-calibrator** — C327/C332/C333. *Helped (banked EEA/NC/EPS on archive).*
- **Fixed joint-physics projection** — C1446/C1448 (adjust co-test groups via train relations). *Mixed — helped Egc/Egb/EEA, hurt others.*
- **Self-regenerating multitask+physics ridge** — C1574–C1576. *Hurt (OOF 0.8598 but transfer 0.836).*
- **Cross-branch current-only arm splice** — C1577/C1579. *Helped (archive 0.9343).*
- **Weak-consensus microblend** — C1519/C1529. *Helped EI/NC.*

---

## 20. Tooling / Infrastructure

- **Clean / oracle namespace separation** — `experiments/CLEAN_OFFICIAL_ONLY/` vs `ORACLE_ASSISTED_RESEARCH_ONLY/`. *Helped — core integrity mechanism.*
- **Watchdog queue + heartbeats** — `research/watchdog-queue.json`, `_watchdog/`. *Operational evidence only.*
- **Post-freeze oracle scoring lane** — only after candidate frozen/hashed. *Helped (unbiased evaluation).*
- **Exact parent replay / parity tools** — replay C050 at 1e-13. *Helped.*
- **Per-experiment `report.json`/`decision.md`/`config.json`/`command.txt`** — immutable run record. *Helped.*
- **Research state YAML + findings.md + research-log.md** — persistent memory. *Helped.*
- **Novelty ledger (URL + content hash)** — prevents rediscovery. *Helped.*
- **Five-role sidecars** — historian, adversary, planner, property-researcher, notebook-auditor (+ general-explorer). *Helped catch leaks/circularity.*
- **Notebook parity builder** — build + execute + hash-compare at 1e-12. *Helped (delivery requirement).*
- **Deterministic sanitizer** — float32 overflow fix (C000→C001). *Helped.*
- **Hash-chained artifacts** — all inputs/outputs hashed. *Helped.*
- **Target-splice / overlay / reflection builders** — reusable assembly tools. *Helped (mechanical, oracle-free).*
- **Two-branch split (with_archive / without_archive)** — *Helped clarify the archive's value.*
- **Anti-oracle replay audit** — C1444/C1489 (0 clean-replayable rows in signed sources). *Helped expose oracle-selection dependency.*

---

## WHAT WORKED BEST (ranked by evidence of gain)

1. **Archive exact-lookup override for Tg/Egc** — +0.016 mean R²; Tg 0.959 / Egc 0.963 (≈60% of those test rows exact). C001/C050. *(Biggest single lever, rules-legal.)*
2. **`eps = nc² + ionic` identity + raw ExtraTrees on 26 polar features** — EPS +0.0666 (C214), Nc +0.0434 (C252). *(The strongest weak-target mechanism.)*
3. **Cross-property partner labels as test-time covariates** (~60% availability) — recovered Ei/Eea/Egb; the core of the 0.916 public score. C001/C282, C143/C144.
4. **Co-test joint solve / meta-calibration** — C332/C333/F25/F26 (EEA→0.947, NC→0.907, EPS→0.859). *(Only mechanism that reaches the "missing partner is a co-test row" case.)*
5. **`ei = egc + eea` bare affine identity (NO ML residual)** — ei R² 0.928, eea 0.955 on observed-partner rows; banked into C050 as the "gap-components" (C048 Eea +0.0208, C049 Ei +0.0279). F01/C048/C049.
6. **`egb = a·egc + b + ExtraTrees residual (α=1.0)`** — 0.9205→0.9478. *(The one identity where ML residual helps.)*
7. **Per-target classical ensemble (Ridge + ExtraTrees + Tanimoto KRR, OOF NNLS blend)** — the durable baseline; blend beats every single family by 0.02–0.05. C001/C282/Claude Stage-1.
8. **Polymer Genome atomic-triple fingerprint (664 keys)** — egb 0.9167→0.9259, nc 0.8438→0.8519. Claude §2.4/C279/C340.
9. **Flory–Fox / oligomer Eea carrier** — Eea 0.9008→0.9163, banked. C180/C189.
10. **Periodic tight-binding / π-Hückel band-structure features** — raw corr −0.791 (eea) / −0.718 (ei); part of Stage-1 +0.0064. Claude §2.3.
11. **`chi = (ei+eea)/2` and `ionic = eps−nc²` reparametrization** — better-conditioned targets (chi OOF 0.8595, ionic 0.7236). Claude Stage-1.
12. **Fixed transfer guards / shrinkage toward incumbent** — rescued bankable Ei (+0.0112, C199) and Egc (+0.0106, C207); shrinkage rescued no-archive EI (C375).
13. **Target-wise compound loop + deterministic audits** — converted isolated component gains into archive 0.894→0.934 mean (C193–C422).
14. **Fixed equal-blend ensembling** — F12 (+0.003), F14 (+0.011) — cheap, transfer-safe gains.
15. **Shift-matched R² + strict parent replay discipline** — the honest decision metric and parity checks that prevented false convergence.

## DEAD ENDS (measured failures — do not repeat)

- **Generic GNN / directed message passing** — C043 Ei −0.309; literature: negative R² at n<141. Do not build standalone GNNs for small targets.
- **Every PI1M representation-pretraining variant** (char-TFIDF, PPMI, denoising, InfoNCE 50k/250k, subword, rarity/density, MLM probe, RankUp distillation) — all ≤ control or transfer-collapsed. C010/C119/C131/C157/C158/C169/C181/C185/C261/F06/C1445.
- **ML residual on the ei/eea identity** — LOO R² = −0.82; adding it always hurts. Fable §3.2.
- **Lorentz–Lorenz / Clausius–Mossotti transform** — worse than plain Nc² (0.797 vs 0.844). C046/C052/C099/C126.
- **Moss/Ravindra/Penn gap–index relations** — C210 Nc −0.137.
- **Log-transform of the ionic term** — costs ~0.02 R² vs raw. Fable §3.3.
- **Adding fingerprints to the 26-feature ionic model** — costs 0.004–0.006. Fable §3.3.
- **SMILES enumeration for descriptor/fingerprint features** — graph-invariant no-op. Fable §4.
- **Forced similarity-gated / read-across routers** — C017–C019/C022/C024/C168 (logistic abstention Ei 0.763); absolute-error gating is ill-posed (R²=−0.005).
- **Cross-property stacking without cross-fitted partner fills** — C132/C139/Claude §3.4 circularity: apparent 0.935→actual 0.907.
- **Unconstrained weak-target model-zoo direct replacement** — C388/C1433/C1502 collapse to 0.84–0.89 on transfer.
- **Tg/Nc size & free-volume / mobility specialists** — C006/C009/C015 regressed.
- **Concat-selector multitask network** — F05 proxy 0.714.
- **Generic Mordred/trimer/3D/AutoGluon sweeps** — Round-1 cooldown.
- **Rich OOF stacks / forced residual routers / per-row overlays** — Round-1 severe test collapse (CV 0.941 → test 0.897).
- **Trying to raise the mean by only tuning Tg/Egc** — they are capped ~0.98 by their 40% uncovered model rows; the gap is in ei/eea/nc/eps/egb.
- **Replicate denoising on the five small targets** — zero replicates exist (C159).

---

## Trajectory Summary (for the presentation)

Final clean OOF per-target R² (C257 compound): Tg 0.9089 (C050 fallback), Egc 0.9221 (banked C207), Egb 0.9221 (C050 fallback), Ei 0.8567 (banked C199), Eea 0.9163 (banked C189), Nc 0.8832 (banked C252), EPS 0.8501 (banked C214) → mean **0.8942**. Only 5 targets were ever banked; Tg and Egb resisted every attempt and stayed on C050.

- Round 1 (~0.923) was a two-target (Tg/Egc) oracle diagnostic; Round 2 is a 7-target unweighted mean, so scores are not comparable.
- The clean Codex loop (C001→C131) plateaued at OOF 0.8731 because its +0.01/bootstrap promotion gate was statistically unpassable at n=222 and rejected ~+0.06 of real summed signal (C035/C077/C082/C171/C179/C242/C244 each 5/5 positive folds, positive bootstrap, rejected for being <+0.01).
- The independent Claude/Fable/Antigravity lines identified the winning structure: exploit the Ramprasad DFT generation identities (`eps=nc²+ionic`, `ei=egc+eea`, `egb=a·egc+b`) and the ~60% test-time partner-label availability, with co-test joint solving.
- The best fully-clean deliverable (one-run notebook) was ~0.914 verified / 0.908 proxy (with archive); the 0.916 public score came from the earlier C143+C144+C148 composite. The 0.93/0.95 targets were **not** reached by any rules-clean, one-run notebook; only oracle-assisted diagnostics crossed 0.93 (no-archive 0.9506, archive 0.9385) and were correctly withheld as non-replayable (0 clean-replayable rows).

---

## Round 3 Transferability Note (Round 3 has NO archive)

**Context change that re-ranks everything below:** Round 3 does not ship `archive/train.csv`, so the *single biggest* Round 2 lever — the archive exact-lookup override that gave Tg 0.959 / Egc 0.963 (~+0.016 mean R²) — **does not transfer**. Round 2's no-archive lane is the honest Round 3 baseline: public **0.891**, best local verified ~0.9042, weak targets `eps`/`nc`/`ei` (and `tg` without archive). Round 3 additionally requires **explainability** and **polymer-invariance robustness**, and adds a 5.97M-row unlabeled `smile_r3.csv` (plus PI1M) for representation learning.

**What carries over (all archive-free Round 2 wins):**
1. DFT identity + ionic-coordinate modeling — `eps=nc²+ionic`, `ei=egc+eea`, `egb=a·egc+b`. The strongest weak-target mechanism and fully archive-free. C214/C252/C162/C187/C190/F02.
2. Flory–Fox / oligomer carrier for Eea — C180/C189 (+0.0154).
3. Transfer-guard / shrinkage pattern (fixed C050-style fallback on predeclared negative scaffolds/similarity slices) — the single most reliable robustification; turned near-misses into banked components (C196→C199, C180→C207).
4. Per-target classical ensemble (Ridge + ExtraTrees + Tanimoto KRR, OOF NNLS) — the durable no-archive baseline (C282 no-archive reference).
5. Polymer Genome atomic-triple fingerprint — Claude §2.4/C279/C340 (ebg 0.9167→0.9259, nc 0.8438→0.8519).
6. Periodic tight-binding / π-Hückel band-structure features — Claude §2.3 (corr −0.79 with eea).
7. `chi=(ei+eea)/2` and `ionic=eps−nc²` reparametrization — better-conditioned targets.
8. Cross-property partner labels as test-time covariates — the ei/eea/nc/eps/egb partner availability is a *train/test* property, not an archive property; it still holds in Round 3.
9. Shift-matched CV + strict parent replay + grouped/scaffold/low-similarity/availability panels — the honest evaluation discipline.

**What does NOT transfer / still dead in Round 3:** the archive exact-lookup override (Tg/Egc must now be modeled — Round 2 no-archive Tg was only ~0.889); every PI1M representation-pretraining variant (all 9+ failed; `smile_r3.csv` is a NEW unlabeled resource worth one bounded probe with a matched supervised control, but do not assume it works); generic GNN/directed-MP (C043 Ei −0.309); Lorentz–Lorenz/Clausius–Mossotti; `log(ionic)`; ML residual on the `ei/eea` identity (LOO R²=−0.82); forced similarity-gated routers.
