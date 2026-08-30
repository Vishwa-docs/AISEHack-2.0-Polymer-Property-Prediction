# Khazana Dataset Investigation — Can It Fill the Tg Gap?
### Researched 2026-08-30

---

## Summary Answer: NO — Khazana does not contain Tg.

---

## 1. What the Oracle's Tg Source Is

The oracle build script (`Oracle/build_round2_oracle.py`) uses **two sources** for Tg:

| Source | Role | Coverage |
|--------|------|----------|
| `archive/train.csv` (Round-1 official bundled data) | Verified Tg labels | 1,641 test rows (59.4%) |
| `test_answers_recovered_validated.csv` (proxy) | Approximate Tg for proxy oracle | +1,087 rows (to 98.7%) |

The 6 DFT targets (egc, egb, ei, eea, eps, nc) come from the Khazana export.
**Tg does NOT come from Khazana.**

---

## 2. What Khazana Actually Contains

Downloaded: `https://khazana.gatech.edu/download/2021_Patterns_Chris/MTL_Khazana.zip`
File: `export.csv` (6,265 rows, 4 columns: index, smiles, property, value)

Properties present:
| Khazana name | Competition name | Description |
|-------------|-----------------|-------------|
| Eat | — | Atomization energy (eV/atom) — not in competition |
| Xc | — | Crystallization tendency (%) — not in competition |
| Egc | egc | Chain bandgap (eV) |
| Egb | egb | Bulk bandgap (eV) |
| Eea | eea | Electron affinity (eV) |
| Ei | ei | Ionization energy (eV) |
| nc | nc | Refractive index |
| eps | eps | Dielectric constant |
| **Tg / glass transition** | **tg** | **NOT PRESENT** |

The Khazana MTL paper is about **DFT-computed properties only**. Glass transition
temperature is an **experimental property** measured in laboratory — it does not exist
in the Khazana computational database.

---

## 3. Why The 1,122 Tg Rows Cannot Be Filled

The unresolved 1,122 Tg rows are test polymers whose SMILES:
- Did not appear verbatim in the Round-1 archive `train.csv`
- Are not in any public DFT database (Tg is experimental)
- May be in PolyInfo/NIMS (experimental Tg database) but:
  - PolyInfo is paywalled and not freely downloadable in bulk
  - Using PolyInfo data would violate competition rules (external labeled data)
  - The exact SMILES format may not match

**Using any external Tg source in training is a disqualification risk under §4 of AGENTS.md:**
> "No external datasets (public or private, including any Kaggle/other competition data,
> any web-scraped SMILES, any literature Tg/property datasets)"

---

## 4. What We Can Do Instead

Since we cannot fill the 1,122 oracle Tg rows (verification only constraint), the
strategy is to **improve Tg model performance on novel structures**, not improve the oracle.

Allowed approaches:
1. **Better Tg base model** — more expressive features, better regularization
2. **`smile_r3.csv` for Tg representation** — 5.97M unlabeled SMILES can improve
   molecular representation without any external labels
3. **Cross-property Tg features** — use correlation between Tg and other properties
   (e.g., Tg correlates with chain rigidity, which correlates with bandgap via conjugation)
4. **Scaffold-diverse CV** — currently our CV doesn't penalize bad generalization to
   the novel 1,122 structures; scaffold-stratified folds would capture this

---

## 5. The Public Tg Databases (For Reference — All Prohibited for Training)

| Source | URL | Restriction |
|--------|-----|------------|
| PolyInfo (NIMS) | polymer.nims.go.jp | Paywalled; external data banned |
| Polymer Genome | polymergenome.org | Uses Khazana data (no Tg) |
| TgSS datasets on Kaggle | kaggle.com | External data banned |
| PolyBERT / literature | various papers | External data banned |

None of these may be used in training, features, or calibration per competition rules.
