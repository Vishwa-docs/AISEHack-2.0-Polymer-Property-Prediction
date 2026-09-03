# Primitive-repeat invariance audit

This isolated audit distinguishes three properties that are often conflated:

- **serialization stability:** randomized SMILES of one molecular graph;
- **translation/cut-point invariance:** moving the repeat window;
- **repetition invariance:** monomer, dimer and trimer spellings of one repeat.

`primitive_repeat.py` implements only the linear grammar exercised by `panel.json`.
`run_panel.py` first validates normalisation and then evaluates the compact portable model on
the one normalised representation. Its outputs remain under `results/`.

Run with the Python 3.11.7 environment after it is created:

```bash
.venv/bin/python run_panel.py
```

The resulting table is an exact demonstration of representation invariance for the declared
panel. It is not evidence that arbitrary branched or copolymer PSMILES can be safely reduced.
