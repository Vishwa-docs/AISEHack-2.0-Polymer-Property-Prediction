# Final presentation: six-minute evidence story

**Core message:** Polymer property prediction becomes more credible when we match the model to the data regime, encode only validated physics, and measure trust rather than assert it.

**Reported outcome:** local held-out verification-panel mean R² **0.907551**; public leaderboard **0.920**.  
**Time budget:** 6:00, then Q&A.  
**Visual rule:** one claim per slide, one chart or diagram per slide, no dense tables.

## Before presenting

- Open the offline website once before the session; have the supplied screenshots open as a fallback.
- Have the leaderboard-ranking screenshot ready. Do not include it in the technical deck; show it after the technical results.
- Export only the slides marked “show”. Keep the full speaker text in presenter notes.
- Do not claim that the compact live website exactly reproduces the full transductive competition ensemble. Say that it is an honest offline demonstration with its own served-model diagnostics.

## Slide-by-slide plan and spoken story

### 1. Title — “From a repeat unit to seven properties you can argue with” (0:00–0:18)

**Show:** Title, a clean polymer-repeat-unit graphic or codebase architecture thumbnail, team name, and the two outcome badges: `0.907551 local held-out panel` and `0.920 public leaderboard`.

**Say:** “We predict seven polymer properties from only a repeat-unit SMILES. Our submitted notebook achieved 0.907551 on our local held-out verification panel and 0.920 on the public leaderboard. But the point of this work is not just a score: it is a model that tells us why it predicted, whether an equivalent SMILES changes the answer, and when the input is too far from its experience.”

**Proof to point at:** architecture image; ranking screenshot can appear as a small postscript only.

### 2. Problem — “One leaderboard, seven different scientific regimes” (0:18–0:48)

**Show:** Seven target names grouped as `Tg`, `electronic structure`, `optical/dielectric`; one-line metric: `mean of seven per-target R²`.

**Say:** “A single pooled model is the wrong abstraction. Tg is experimental and needs structure-to-property extrapolation. The electronic and optical targets are scarce and often have a different measured property for the same polymer. Yet the metric weights every target equally. So we framed this as seven linked problems, not one table.”

**Proof to point at:** simple metric equation; no detailed values.

### 3. EDA insight — “The data chose the architecture” (0:48–1:22)

**Show:** `outputs/eda/novelty_two_regimes.png`; annotate “no exact target-label duplicates” and “different property can be present”.

**Say:** “Before fitting models, we measured overlap at the polymer–property level, not merely by raw SMILES. The DFT targets have no exact label duplicates in evaluation, but many structures occur under a different property. Tg has much less of that support. This is the key EDA finding: one part is cross-property inference under strict label separation; the other is extrapolation. It is why the final system has target-specific lanes and grouped structural validation.”

**Anticipated question:** “Is that leakage?” Answer: “No evaluation label is available to the predictor; partner values come only from training labels and are guarded.”

### 4. Architecture — “Five stages, each with one job” (1:22–2:10)

**Show:** `outputs/architecture.png`, with five numbered overlays and a five-word feature key: `bulk descriptors · local motifs · polymer triples · graph · controlled residual`.

**Say:** “Stage one creates complementary representations with different jobs: descriptors express bulk chemistry, fingerprints preserve local motifs, atomic triples make the encoding polymer-aware, a graph model provides a connectivity view, and character features are a deliberately controlled residual. Stage two chooses a model family per target rather than pretending 200 and 4,000 examples are the same problem. Stage three adds only guarded physical routes: band edges for the chain gap, a calibrated bulk-gap relation, and the dielectric decomposition. Stage four combines independent predictions with weights fitted out of fold. Stage five corrects conservative ensemble compression. The GINE graph network is a decorrelated second opinion, not a claim that deep learning dominates small polymer data.”

**Show, not say:** Use arrows coloured by “structure”, “physics”, and “validation”; keep all hyperparameters out of the slide.

### 5. Why this is science rather than an ensemble pile (2:10–2:42)

**Show:** a three-column “Hypothesis → test → decision” card.

| Hypothesis | Test | Decision |
|---|---|---|
| Physics should help | held-out route comparison | retain only guarded identities |
| A neural arm may diversify errors | grouped out-of-fold blend test | use as a complement |
| Pretraining must earn its place | matched control | keep negative results visible |

**Say:** “Every component had to earn its place by a controlled validation question. A physical identity stays only when it beats or protects a learned route. A neural component stays because it makes different errors, not because it is fashionable. And approaches that lost to their controls remain in the experiment record. That is our safeguard against overfitting an attractive story.”

### 6. Explainability — “An explanation must survive intervention” (2:42–3:15)

**Show:** `outputs/explainability/shap_beeswarm_tg.png` beside one large callout: `top-5% mask: 0.810; random: 0.022`.

**Say:** “SHAP plots are not evidence by themselves. We performed a removal test: mask the features SHAP calls most important and compare it with masking equally many random features. Across the recorded full-run scorecard, the top-5% intervention loses 0.810 R² while the random control loses 0.022. So this is not merely a visually plausible explanation; those features are operationally load-bearing in the tested proxy setting. The display is released only after the pinned rerun refreshes the scorecard.”

**Cite in small type:** Lundberg & Lee, 2017; Hooker et al., 2019.

### 7. Invariance and robustness — “Same polymer, different spelling, same answer and reason” (3:15–3:48)

**Show:** `outputs/robustness/smiles_invariance_boxplot.png` and a two-line card: `zero tested graph violations`; `mean attribution cosine 0.980`.

**Say:** “SMILES is a string encoding of a graph, so equivalent strings are a real failure mode. We randomised valid SMILES spellings for the same polymer. The recorded graph-representation test has zero tested violations and the corresponding explanations have mean cosine 0.980. We report the character arm separately because string features are not invariant by construction; accuracy is never an excuse to hide that trade-off.”

### 8. Generalisation and honesty — “A prediction needs a boundary” (3:48–4:18)

**Show:** `outputs/generalization/generalization_ladder_plot.png`, simple T1–T4 tier icons, and one compact support line: `canonical-group 0.824 · scaffold 0.658 · coverage 7/7`.

**Say:** “An average score can hide novelty failures. We therefore stratify performance by structural similarity and carry that into the interface as an applicability tier, nearest training analogue, and calibrated interval. The recorded run remains positive under the canonical-group and scaffold tests, and its coverage check passes all seven targets. Error–uncertainty correlation clears the stated bar on five of seven targets, so tier and interval are evidence context—not a blanket warranty. The system does not say ‘trust me’; it says ‘this is how far this polymer is from what I know.’”

### 9. Live demo — “Prediction with provenance, not a black box” (4:18–5:00)

**Show:** The evidence-console build **after** the acceptance checks in `finals/WEBSITE_DEMO_SPEC.md` pass; until then, retain the current website and its screenshot fallback rather than simulating the planned interaction.

1. Select a preloadable polymer card; let the labelled 3D repeat-unit conformer rotate, then pause it. Point at `Structure ready · model not loaded`.
2. Press **Run analysis** for PIB, the disclosed external anchor; show value, interval, tier, nearest analogue and the separately revealed reported-material value.
3. Click “rewrite this SMILES”; point at multiple strings, canonical collapse and the stable graph-based trace.
4. Enter a deliberately unfamiliar string; point at the out-of-domain warning rather than attempting to impress with its scalar prediction.

**Say:** “The website is fully offline. The 3D view is an illustrative conformer, not a morphology simulation. The prediction only appears after a visible live inference step, and it exposes source, applicability tier, nearest analogue and interval. PS and PMMA are labelled literature context; PIB is the one current external anchor. This compact serving path is deliberately distinct from the full competition ensemble; it never silently presents cached results as a live model.”

**Fallback:** use `Website/screenshots/` without apologising; say “Here is the recorded offline fallback.”

### 10. Results, future direction and close (5:00–6:00)

**Show:** two outcome cards, a 3-row “what changed / what remains” table, QR links to codebase and report.

| What we established | What remains |
|---|---|
| 0.907551 local held-out panel; 0.920 public leaderboard | refresh recorded qualitative evidence in the pinned environment |
| intervention-tested explanation and SMILES invariance | strengthen cross-model explanation agreement |
| applicability-aware offline prediction | establish a pre-registered external-material panel |

**Say:** “Our contribution is a pattern for small-data polymer informatics: match the estimator to the sample regime, use physics as a guarded constraint rather than a slogan, and make trust measurable. The next work is clear: better Tg physics features, a pre-registered external-material panel, and carefully controlled pretrained representations. We have shown the score, the mechanism, the stress tests, and the limitations. Thank you.”

## Appendix slides for Q&A (do not present unless asked)

### A. “Why not a single multitask deep model?”

Use the sample-size imbalance and the tabular-data reference: Grinsztajn et al., NeurIPS 2022. State that the GINE is tested as a complementary component, and that a deep model does not get a free pass at the smallest targets.

### B. “Is cross-property information leakage?”

Use a train/evaluation flow diagram. State: the requested target label is never read; only an allowed training measurement of a different property can trigger a guarded physical relation. The architecture is evaluated with structure-grouped splits.

### C. “What failed?”

List the retained boundaries in a neutral layout: raw cross-model explanation agreement, two error–uncertainty targets below threshold, missing process/morphology variables, and the unfinished external-material panel. For each: `scope → next measurement`. This is a credibility slide, not a defensive slide.

### D. “What validates explainability?”

Show the SHAP masking fidelity chart and distinguish: feature importance is a hypothesis; the masking experiment is the intervention test.

## Visual asset manifest

| Purpose | Existing asset | Notes |
|---|---|---|
| Architecture | `outputs/architecture.png` | use on slide 4 |
| EDA / two regimes | `outputs/eda/novelty_two_regimes.png` | use on slide 3 |
| Explainability | `outputs/explainability/shap_beeswarm_tg.png` | use on slide 6 |
| Prediction invariance | `outputs/robustness/smiles_invariance_boxplot.png` | use on slide 7 |
| Generalisation | `outputs/generalization/generalization_ladder_plot.png` | use on slide 8 |
| Qualitative scorecard | `fixes/qualitative_evidence/figures/qualitative_scorecard.png` | refresh before final export |
| Website interaction | `finals/WEBSITE_DEMO_SPEC.md` | build/rehearse before stage time |

## Research citations to place in notes or final deck

- Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022): small/medium tabular-model baseline rationale.
- Kim et al. (2018), *Polymer Genome*: polymer descriptors.
- Lundberg & Lee (2017) and Lundberg et al. (2020): SHAP/TreeSHAP.
- Hooker et al. (2019): intervention-based importance validation.
- Bjerrum (2017): randomised-SMILES augmentation and invariance testing.
- Angelopoulos & Bates (2021): conformal uncertainty.
- Fox & Flory (1950): polymer-chain-length/Tg relation.
- Koike & Kumaki (2022) and Keszler et al. (2000): disclosed PS/PMMA context and PIB external-anchor context.

Use full references from `finals/Report.md` in the final deck’s references slide.
