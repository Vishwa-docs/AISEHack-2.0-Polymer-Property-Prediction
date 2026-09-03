# Website demo specification — evidence console

**Status:** design and evidence contract only. No website code is changed by this document.

## Decision

The website should behave like a small scientific instrument, not a leaderboard dashboard:

1. choose a documented polymer;
2. inspect its structure before any model is loaded;
3. explicitly press **Run analysis**;
4. see a prediction together with applicability, uncertainty, provenance and explanation;
5. test an equivalent representation and, where available, compare with a disclosed reference.

This structure follows the UI/UX Pro Max review priorities: accessible feedback before visual
effects, intentional motion, clear provenance, responsive reflow, and charts whose meaning does
not depend on colour alone. The visual direction is **scientific instrument / evidence console**:
quiet neutral surfaces, deep ink text, teal for verified evidence, amber for a boundary and red
only for an out-of-domain warning. It is deliberately not a generic neon “AI” dashboard.

## What the stage demo must prove

| Judge requirement | Visible interaction | Honest scope statement |
|---|---|---|
| Predict a new repeat unit | paste or select a SMILES, then press **Run analysis** | forward screening, not synthesis certification |
| Invariance | rewrite the same graph and re-predict | valid randomised SMILES encodings only |
| Explainability | inspect active structural signals, then open the fidelity evidence | live descriptor view is not relabelled as TreeSHAP |
| Robustness | show interval, nearest analogue, applicability tier and source | interval is informative only in its calibrated regime |
| Generalisability | open the disclosed external-anchor comparison | one anchor is not an external benchmark |
| Transparency | show an unfamiliar input that triggers the boundary treatment | never silently replace a live inference with a lookup |

## Interaction design

### 1. Pre-inference structure stage

The page opens with three preloadable cards and one free-text input. Selecting a card must render
the structure immediately but **must not instantiate the predictor**. A persistent state marker
reads `Structure ready · model not loaded` until the user presses the primary button.

| Card | UI label | Evidence status | Why it belongs on stage |
|---|---|---|---|
| Polystyrene | `Literature context` | Its Tg label is present in the supplied training table. | Familiar rigid-aromatic chemistry; use only as a sanity illustration. |
| PMMA | `Literature context` | Its structure is present in the supplied evaluation table. | Familiar ester-bearing chemistry; use only as a sample-level literature comparison. |
| Polyisobutylene (PIB) | `External anchor` | Its canonical repeat unit is absent from both supplied tables. | A simple, genuinely external repeat unit with a disclosed Tg reference. |
| New candidate | `Unlabelled design screen` | No reference label asserted. | Demonstrates forward screening and the applicability boundary. |

The primary control is `Run analysis`. It provides a visible loading state—`Loading compact
offline predictor → featurising graph → estimating target`—and only then fills the results area.
This makes the live computation legible without pretending that a 3D spin is model inference.

### 2. 3D structure, used honestly

Use a **locally bundled client-side 3D canvas** for the interaction, not Manim. RDKit should make
an ETKDG conformer of a capped, short oligomer solely for display; a label must say
`illustrative repeat-unit conformer — not a bulk morphology simulation`. The canvas may rotate
slowly while idle, but must have Pause, Reset and reduced-motion behaviour. Atom legend and
bond order must remain readable without relying on the palette.

Manim is appropriate for a pre-rendered 6–8 second deck transition or an architecture explainer,
but not for a result that changes after a click. The dynamic chart and 3D view must be a local
browser component so that the displayed state is the current inference state and the offline demo
remains usable without conference Wi-Fi.

### 3. Results stage

After analysis, reveal information in this order rather than dumping seven unrelated cards:

1. **Result:** property, value and units.
2. **Confidence context:** 90% interval, applicability tier and nearest-analogue similarity.
3. **Why this result:** compact-model descriptor signal chart; a separate link/card states that
   full-pipeline TreeSHAP and its masking-fidelity evidence are different artefacts.
4. **Representation check:** generate randomised SMILES, list their canonical collapse and animate
   the prediction trace between variants.
5. **Reference context:** only for an eligible card, reveal the reported material value, source,
   test condition and a large `illustrative comparison, not benchmark` label.
6. **Design screen:** compare the candidate with a desired target and retain the interval/tier in
   the decision instead of displaying an unqualified “best polymer”.

All result transitions should be subtle opacity/position changes under 250 ms. Respect
`prefers-reduced-motion`; never use animation to hide a discontinuous prediction or an error.

## Reference examples and anti-cherry-picking protocol

The prior PS/PMMA pair is useful for chemistry context, but it is **not** an external validation
set. A canonical-structure audit of the supplied tables found PS in training and PMMA in the
evaluation structures. The UI must retain their literature sources but must not badge either as
“unseen.”

PIB is the one presently eligible external anchor: its repeat unit is
`*CC(C)C*`, it is absent from both supplied tables, and M8 reports a Tg of −73 °C for high-
molecular-weight PIB. The current compact predictor gave −67.559 °C in the documented read-only
audit; that proximity is an illustrative single-material observation, not a performance claim.

Before the site makes *any* external-accuracy statement, create a fixed five-material external
panel with these gates:

1. choose materials and cite their source **before** looking at model predictions;
2. record repeat unit, tacticity/molecular-weight/DSC condition and target units;
3. canonicalise and prove absence from both supplied competition tables;
4. run the compact model once with a frozen version/hash;
5. publish every chosen material, including misses, with absolute error and applicability tier.

The external panel belongs in a separate evidence card, never in the contest-score headline. This
is the credible way to answer “does it work beyond the table?” without selecting only favourable
examples.

## Required assets and implementation boundary

| Need | Proposed implementation when codebase work is authorised | Non-negotiable check |
|---|---|---|
| 3D conformer | local browser component + RDKit-generated coordinates | no CDN or remote asset fetch |
| dynamic charts | local interactive chart component fed by the current inference state | tooltips, label and values accessible without colour |
| live trigger | session-state lazy predictor load behind `Run analysis` | no implicit cached evaluation result |
| preload cases | versioned local JSON with source/status/membership fields | no “external” badge without audit evidence |
| evidence links | local static figures for fidelity/generalisation | live compact importance never called SHAP |
| fallback | one static screenshot for each four-stage demonstration | rehearse offline before stage time |

## Acceptance test before code promotion

- Browser starts offline; selecting a preload card does not load model weights.
- Pressing `Run analysis` changes the visible state and shows only the compact predictor result.
- The structure canvas has a pause/reset control and reduced-motion path.
- A randomised-SMILES comparison proves canonical graph equality before interpreting stability.
- Each value carries unit, interval, tier, nearest analogue and source; a T4 input has a visible
  warning rather than a celebratory result.
- PS and PMMA say `Literature context`; PIB says `External anchor`; neither wording is hidden in
  a tooltip.
- The external panel is not called validation until all five pre-registered cases pass the audit.

## Sources

- M7: Koike & Kumaki (2022), PS/PMMA sample-level Tg context.
- M8: Keszler et al. (2000), PIB Tg context.
- C1/C5: RDKit and randomised-SMILES representation protocol.
- X2/X3/U1/U3: TreeSHAP/fidelity, interval and applicability-domain framing.

