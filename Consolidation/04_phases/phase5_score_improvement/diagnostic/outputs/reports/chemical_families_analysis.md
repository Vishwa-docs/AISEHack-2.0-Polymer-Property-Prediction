# Polymer Chemical Family Classification Report (eda_chemical_families.py)

**Execution Date:** 2026-08-30 22:11:21  

## 1. Family Distribution (Train vs Test)

| Chemical Family | Train Count | Train % | Test Count | Test % | Shift / Hazard Ratio |
|---|---|---|---|---|---|
| **Polyester** | 2,083 | 28.1% | 1,432 | 29.0% | Balanced |
| **Polyamide/Imide** | 2,687 | 36.3% | 1,673 | 33.9% | Balanced |
| **Polyether** | 3,855 | 52.0% | 2,549 | 51.6% | Balanced |
| **Polyurethane/Urea** | 382 | 5.2% | 244 | 4.9% | Balanced |
| **Halogenated (F, Cl, Br)** | 1,076 | 14.5% | 751 | 15.2% | Balanced |
| **Aromatic/Conjugated** | 5,388 | 72.7% | 3,591 | 72.7% | Balanced |
| **Sulfur-containing** | 1,698 | 22.9% | 1,175 | 23.8% | Balanced |
| **Silicon-containing** | 211 | 2.8% | 159 | 3.2% | Balanced |
| **Pure Hydrocarbon/Polyolefin** | 7,123 | 96.1% | 4,730 | 95.7% | Balanced |

## 2. Key Observations

1. **Aromatic / Conjugated Backbone Dominance:** Both train (72.7%) and test (72.7%) are heavily aromatic polymers (polyphenylene, polyimide, polyetherketone cores).
2. **Heteroatom Substructures:** Polyesters and polyamides represent key functional classes with hydrogen-bonding and polar contributions critical to Tg and dielectric constant ($\epsilon$).
