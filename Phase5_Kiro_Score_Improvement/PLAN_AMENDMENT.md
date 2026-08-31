# PLAN AMENDMENT: Cross-Property Pipeline 2.0

**Date:** 2026-08-30  
**Source:** `NEW_NEW_EXPERIMENTS.md`  
**Purpose:** This amendment overrides the generic model search in the original `PLAN.md` with a highly targeted, evidence-backed strategy. The primary focus shifts to **Cross-Property Imputation**, **Heterogeneous Ensembles**, and **Property-Space SSL**.

---

## Strategic Pivot

The previous plan focused heavily on searching for novel model architectures. Evidence from the 0.894 external competitor and recent (2025/2026) polymer informatics literature indicates the actual path to >0.93 relies on:
1. **Aggressive cross-property transfer** (using the data as a partially observed property matrix).
2. **A genuinely heterogeneous model zoo** (producing decorrelated residuals).
3. **Multi-teacher pseudo-labeling** on the 6M `smile_r3` dataset.
4. **Target-specific convex OOF blending**.

---

## New Execution Waves (Experiments 271-300+)

These waves take precedence over generic Phase A-J experiments.

### Wave 1: Forensic Replication (Experiments 271-275)
**Goal:** Reconstruct the competitor's 0.894 mechanism exactly before innovating.
- **Exp 271:** Audit partner availability (Test exact vs canonical vs any-partner). Discrepancy check: 60% vs 88-99%.
- **Exp 272:** 15-Model Portfolio Baseline (LGB, XGB, Cat, ET, Tanimoto kNN, multi-task MLP, AttentiveFPx2, GINEx2, SMILES Transformer).
- **Exp 273:** Cross-property fold-local iterative imputation (Train with exact missingness pattern).
- **Exp 274:** Residual correlation matrix across all 15 models.
- **Exp 275:** Convex SLSQP blending (Target-specific) vs current NNLS.

### Wave 2: Cross-Property 2.0 (Experiments 276-280)
**Goal:** Exploit property correlations optimally.
- **Exp 276:** Iterative joint fill (Chemistry → Initial properties → Refined properties).
- **Exp 277:** Target order optimization (e.g., Egc → Egb → Eea → Ei → Nc → Eps).
- **Exp 278:** Partner dropout training (Simulate test-time missingness dynamically).
- **Exp 279:** Availability-specific ensemble routing (Different blend weights based on the observed partner pattern).

### Wave 3: New Representation Injection (Experiments 281-285)
**Goal:** Add specific, non-redundant structural representations (Do not replace the parent, blend the OOF residuals).
- **Exp 281:** MCP (Multi-Cover Persistence) features + incumbent (Strong literature support for Eea/Eps/Nc/Ei).
- **Exp 282:** Radical-marker geometry / Endpoint topology features (Especially for Tg, Egb, Egc).
- **Exp 283:** Heavy-atom-only graph vs Full graph.
- **Exp 284:** Dual Graph (Atom graph + Backbone/Side-chain graph).
- **Exp 285:** Task-grouped shared GINE/AttentiveFP with cross-property covariates at the heads.

### Wave 4: PI1M / smile_r3 Pseudo-Label Field (Experiments 286-290)
**Goal:** Use the 6M dataset for synthetic property vectors, not generic MLM.
- **Exp 286:** 15-model teacher generates full 7-target property vectors for 6M `smile_r3`.
- **Exp 287:** Uncertainty filtering (Keep only low teacher variance rows).
- **Exp 288:** Residual pseudo-labeling (Teach the student the teacher's nonlinear correction over the baseline).
- **Exp 289:** Train LightGBM/MLP student on filtered multi-teacher pseudo-labels.

### Wave 5: Property-Space SSL (Experiments 291-295)
**Goal:** Direct self-supervised learning on property relationships.
- **Exp 291:** Masked Target Prediction (Mask 1 property, reconstruct from SMILES + 6 properties).
- **Exp 292:** Counterfactual property training (Train model to predict target under varying partner availability).
- **Exp 293:** Privileged-information distillation (LUPI) - Student predicts teacher's internal representation.

### Wave 6: Physics Latent Model (Experiments 296-300)
**Goal:** Align with the AISEHack physics/operator theme.
- **Exp 296:** VBM/CBM/Vacuum-level latent coordinate inference → Ei, Eea, Egc.
- **Exp 297:** Electronic polarizability + ionic response latent → Nc, Eps.
- **Exp 298:** Physics-conditioned neural operator (Constraints: Ei-Egc-Eea=0).

---

## Target-Specific Attack Map

- **Ei (Priority 1):** Target 0.92-0.94. Use exact partner values, physical identity, donor/acceptor topology, MCP, GINE residual, 6M teacher.
- **Eps (Priority 2):** Target 0.93-0.95. Use ionic decomposition, MCP, joint Nc/ionic model, teacher field.
- **Nc (Priority 3):** Target 0.93-0.95. Use polarizability, MCP, radial/graph topology.
- **Egb:** Target 0.945-0.955. Use Egc affine relation + ExtraTrees/MCP/GINE residual + radical-marker.
- **Eea:** Target 0.925-0.935. Use Ei/Egc partner info, MCP, Flory-Fox, AttentiveFP.
- **Egc & Tg:** Do not over-invest. Use Egc as an upstream property. Use Tg with radical-markers and rotational features.

---

## New Milestone Gates

Do not attempt to jump from 0.904 to 0.935 in one model. Use the following progression:
1. **0.904 → 0.912:** Prove one new family (e.g., MCP + parent blend).
2. **0.912 → 0.920:** Prove compound architecture (15-model cross-property imputation).
3. **0.920 → 0.930:** Attack Ei / Nc / Eps / Eea specifically (Target-specific physical latent models).
4. **0.930 → 0.935+:** Final ensemble, SLSQP blending, distribution/domain corrections.

*Execution Note: The agent should execute the Diagnostic suite first, followed immediately by Wave 1 (Forensic Replication), as it forms the baseline for all Cross-Property 2.0 experiments.*
