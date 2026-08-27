# 50-Experiment Plan to Reach 0.935 Verified (R3 Loop Continuing)

**Goal:** 0.935 verified oracle mean R² (≈ public 0.92-0.93). Current best 0.90276 (V52 no-archive). Gap +0.03224 across 7 targets → need +0.0046 per target average, but weak targets need +0.02-0.05.

**Constraints (AGENTS.md §4 — no external data, no pretrained models — violations = disqualification):**
- Official competition data **only**: `Dataset/train.csv` (7,409), `Dataset/test.csv` (4,940), `Dataset/PI1M.csv` (995,799), `Dataset/smile_r3.csv` (5,973,369) — **no other datasets** (no Kaggle/other competition data, no web-scraped SMILES, no literature Tg/property datasets).
- **No pretrained models/weights/embeddings/checkpoints/vocabularies** — includes HuggingFace ChemBERTa/MolBERT/Uni-Mol/Graphormer, any LLM/VLM/GNN pretrained on molecules/polymers, no transfer learning from outside. `archive/` from R2 is **banned** in R3.
- Every representation (TF-IDF, SVD, word2vec, MLM, contrastive, GNN, Transformer) **must be fitted from random initialization inside the single notebook run** on official data only — no artifacts created outside notebook, no wheels/datasets/checkpoints attached. Kaggle image preinstalls RDKit, torch, transformers, sklearn, xgboost, lightgbm, catboost, shap — use only those.
- Single end-to-end notebook (`id,target` 4,940 rows), fixed seeds, structure-grouped validation (457 SMILES overlap), post-freeze oracle scoring only via `Oracle/score_round2_ORACLE_ASSISTED_RESEARCH_ONLY.py`.

**Cooled to avoid (TRIALS.md:405):** generic GNN/CNN/Transformer/MLM from scratch (C043 -0.309), PI1M PPMI/contrastive/density (C010/C119/C157), `log(ionic)` hurts -0.02, Lorentz-Lorenz hurts -0.04, `ei=egc+eea` ML residual hurts -0.82, broad read-across.

## Ladder (ordered by EV)

### Phase 2: Selection Repair (C001-C005) — close clean vs oracle gap (0.904→0.951 diagnostic)
- **R3-C001** shift-matched R² selection (OOF reweighted to test NN-sim histogram, `fable_common.py:231` bins [0,0.3,0.4,0.5,0.6,0.7,0.85,1]) — expected +0.003 mean vs plain OOF. Gate: verified >0.9027.
- **R3-C002** co-test consistency (agreement of 2 independent models per target) — gate >best.
- **R3-C003** conservative transfer guards (C199/C207 style, 4/5 folds + bootstrap >0) — gate >best.
- **R3-C004** NNLS ensemble of selection signals (shift-matched + scaffold + availability) — gate >best.
- **R3-C005** Tg shift diagnosis + correction (NeurIPS winner std*0.5644 grid, but our grid found 0 optimal; test additive bias via OOF residuals, never oracle) — gate >best.

### Phase 3: Weak-Target Physics (C010-C020) — strongest transferable signals
- **R3-C010** EPS/NC ionic ensemble deepening (32 polar features + 3-model ionic ensemble, 0.6/0.4 blend, `eps=nc²+ionic`, raw not log) — *running* (polar_block, 680-dim). Expected eps +0.01, nc +0.01. **Web:** Clausius-Mossotti ` (eps-1)/(eps+2) = Nα/3ε0` and Lorentz-Lorenz `(n²-1)/(n²+2)=4πNα/3` — but TRIALS shows raw `ionic` beats CM/LL (0.797 vs 0.844), so keep raw.
- **R3-C011** EPS/NC with Mordred autocorrelation (ATS/AATS/GATS/Moran/Geary, APol/BPol, EState) — descriptor-space analogs of conjugation length.
- **R3-C012** Ei via `Ei=Eea+Egc` soft identity + chi/gap coordinates (C171 0.905 OOF but failed scaffold — repair guard)
- **R3-C013** EHT orbital features (Gasteiger/Hückel spectra corr -0.791 with eea per `TRIALS.md:10`)
- **R3-C014** Conjugation/donor-acceptor SMARTS counters (C=O, C#C, c1ccccc1 etc.)
- **R3-C015** 3D HOMO/LUMO proxies (ETKDG conformer, MMFF, 3D polarizability)
- **R3-C016** Tg group-contribution Bicerano/Van Krevelen (Tg = ΣYi/M + corrections, SMARTS×R_M/P_M)
- **R3-C017** Backbone/pendant decomposition (largest side chain, ring-ring distance)
- **R3-C018** Egb `a·egc+b + ExtraTrees residual` (C160 0.920→0.947)
- **R3-C019** Flory-Fox oligomer carrier for Eea (C189 banked +0.015)
- **R3-C020** Chi/ionic/dgap reparametrization (chi OOF 0.8595, ionic 0.7236)

### Phase 4: SSL Ladder on 5.97M smile_r3 + 1M PI1M (C040-C050) — from-scratch, hash-ranked, equal-budget controls
- **R3-C040** Morgan count (r2,1024) → SVD 128 on 1M smile_r3 (cheap, CPU) — *done 0.861 rejected* (hurt due to noise, need smaller 64 or different head)
- **R3-C041** Char n-gram TF-IDF (2-6, 50k cap → SVD 128) on 1M smile_r3 — silver-medal recipe, ~10 min
- **R3-C042** word2vec/fastText token embeddings (regex/char tokens, dim 128-256, 10-30 min) → mean pool
- **R3-C043** Tiny char BERT MLM from scratch (2-4 layers ×128-256, vocab SMILES tokens, seq 128, 1-2M seq, 30-90 min on 5090) — frozen linear/GBM probe vs control
- **R3-C044** Pairwise comparison pretraining (predict higher property of pair, cheap winner trick)
- **R3-C045** Contrastive multi-view (InfoNCE on 2 random SMILES per polymer, 50k/250k scaling test)
- **R3-C046** Denoising functional-group bottleneck (C131 neutral)
- **R3-C047** VAE/MLM on 6M full (expensive, only if 1M probes pass kill gate)
- **R3-C048** Morgan SVD 1M → 6M scaling (100k→1M→6M ladder)
- **R3-C049** Pseudo-labeling self-training (predict sparse targets on PI1M/smile_r3, keep confident, fold-local retrain) — only after probe passes
- Gate per probe: ≥4/7 targets or +0.01 on eps/nc/ei/tg with panels vs control, else cool.

### Phase 5: Invariance & TTA (C070-C079) — judged theme, free 0.002-0.01
- **R3-C070** TTA: randomized SMILES (N=20-50, doRandom, 1/2/3-mer recut), median per row — *queued* (needs sequence model head, else no-op for descriptors per `TRIALS.md:52`)
- **R3-C071** Train-time augmentation k=2-5 random SMILES per train row (10× if runtime)
- **R3-C072** Consistency regularization λ·mean((f(x)-f(x'))²) for NN
- **R3-C073** Invariance audit: canonical vs random vs recut spread quantiles, worst-case polymers
- **R3-C074** Cut-point invariance (ring-close + enumerate backbone cuts, average descriptors, `F03` Fable)
- **R3-C075** Repeat-view invariance 1/2/3 (C277 -0.0017 but with new head)
- **R3-C076** SMILES enumeration for char model (10×, median)
- **R3-C077** Multi-view TTA ensemble (canonical + 2 random + oligomer)
- **R3-C078** Test-time dropout MC + TTA (dropout 0.1, 10 variants)
- **R3-C079** Invariance robustness report for FINAL_REPORT.md

### Phase 6: Multitask & Ensemble (C030-C039) — Khazana paper single best-motivated neural
- **R3-C030** Shared-encoder MLP 7 heads (6 DFT + Tg) + soft physics losses `Egc≈Ei-Eea`, `eps≥nc²`, target-balanced sampling — kill if no weak gain
- **R3-C031** Multi-task LightGBM (masked, low-rank `C091` failed but new)
- **R3-C032** CatBoost multitask with ordered boosting
- **R3-C033** GNN from scratch (directed MP, but with physics heads + periodic recurrence, kill gate strict per `TRIALS.md:42`)
- **R3-C034** Transformer SMILES (tiny, 4 layers, but with random SMILES augmentation)
- **R3-C035** Tabular + GNN + LM multi-view ensemble (NeurIPS 9th place 0.082 MAE) — per-property uniform 1/n average
- **R3-C036** Stacking meta-learner (OOF NNLS, but fold-local to avoid `C132` circularity)
- **R3-C037** XGBoost + TabPFN (per `kunjanbansal/Polymer-Prediction` high-score solution)
- **R3-C038** LightGBM + ExtraTrees + Ridge per-target zoo with 10-fold CV
- **R3-C039** Final compound assembly: C050 parent + banked per-target + TTA via fold-local NNLS

### Phase 7: Explainability (C080-C089) — judged theme, no score impact
- **R3-C080** SHAP TreeExplainer GBM (≤1000 rows) global beeswarm + bar
- **R3-C081** Permutation importance Ridge
- **R3-C082** Local waterfall 2-3 per target
- **R3-C083** Chemical narrative (aromaticity → bandgap, polarizability → eps/nc, rigidity → Tg, ionic eps-nc²)
- **R3-C084** Invariance robustness SHAP
- **R3-C085** Limitations note (sparsity n≈222, Tg oracle gaps 1,122)
- **R3-C086-089** Notebook sections + `FINAL_REPORT.md` (before 3 Sep)

## Execution Order (next 50 sequential, one GPU at a time, 30% headroom)
1. R3-C040 done, C010 running, seed sweep running → next C100 (queued), C070, C041, C011, C012...
2. After each, post-freeze oracle score, promotion gate `EXPERIMENT_LOOP.md:159` (≥0.01 grouped, 4/5 folds, bootstrap>0, adjacent loss ≤0.003), else shrink lane 0.05-0.25 weight.
3. Full incumbent gate ≥0.002 mean, no target <-0.003.
4. After every 4 experiments, outer loop `deepen/broaden/pivot` in `research/research-state.yaml`.

## Resource Budgets
smoke ≤15 min, pilot ≤60 min, confirm ≤4h, final ≤8h. All SSL probes hash-ranked 100k→1M→6M, every probe vs equal-budget control.

## References (webfetch)
- NeurIPS OPP 1st place: ModernBERT + AutoGluon + Uni-Mol2, 10× train + 30× median TTA, Tg shift std*0.5644 (`dev.to/nk_maker`)
- Tg GAP+QSPR (arXiv 2411.06461), Group contributions (Weyland 1970, Van Krevelen)
- Dielectric: npj Comp Mater 2020 (Khazana DFPT, 134 pairs, R² 0.844 plain Nc² vs 0.797 CM)
- Clausius-Mossotti / Lorentz-Lorenz (wiki, arXiv 2506.01993 refined ELF)

Loop continues until verified ≥0.935 — poller `mac_poller_daemon.py` + GPU queue `~/Desktop/r3_runtime/run_loop.sh` keep one heavy job at a time.

