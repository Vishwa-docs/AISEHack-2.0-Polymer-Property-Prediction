# 995k Polymer SMILES Characterization Report (eda_pi1m.py)

**Execution Date:** 2026-08-30 22:11:12  
**Total Rows in PI1M.csv:** 995,799  
**Sample Analyzed:** 50,000 rows  
**Molecules with Polymer Attachment Points (*):** 100.00%

## 1. Monomer Structural Properties

- **Mean SMILES Length:** 46.81 ± 22.53 chars (Range: 3 - 166)
- **Mean Cleaned Monomer Weight:** 366.70 Da

## 2. Complementarity with smile_r3

- `PI1M.csv` contains specialized repeat units with stoichiometric polymerization attachment points (`*`), capturing backbone connectivity.
- `smile_r3.csv` (5.97M) provides dense molecular space for chemical fragment representation.
- Combined self-supervised token learning leverages both general organic chemistry and polymer repeating unit geometry.
