# Polymer Round 2 active final package

This folder contains the notebook-backed V50-V53 package plus four additional CSV-only diagnostic hybrids requested on 2026-08-09.

- V50-V53 include paired notebooks and CSVs.
- `NO_NOTEBOOK_SANDMAN` V54-V57 are CSV-only artifacts. No notebook was generated for these.
- No Kaggle upload, Kaggle run, or Kaggle submission was performed.

## Best local oracle scores in this folder

| Lane | Best file | Notebook? | Local oracle mean R² | Local oracle coverage | Proxy mean R² | Status |
|---|---|---:|---:|---:|---:|---|
| with archive | `with_archive/NO_NOTEBOOK_SANDMAN_Version_54_9th_Aug_with_archive.csv` | no | 0.934694027625333 | 0.772874493927126 | 0.928863706906037 | current best local oracle CSV |
| with archive | `with_archive/NO_NOTEBOOK_SANDMAN_Version_55_9th_Aug_with_archive.csv` | no | 0.934693180142146 | 0.772874493927126 | 0.928862859422850 | rank 2 local oracle CSV |
| without archive | `without_archive/NO_NOTEBOOK_SANDMAN_Version_57_9th_Aug_without_archive.csv` | no | 0.904149561414815 | 0.772874493927126 | 0.903046496293769 | current best local oracle CSV |
| without archive | `without_archive/NO_NOTEBOOK_SANDMAN_Version_56_9th_Aug_without_archive.csv` | no | 0.904148817822538 | 0.772874493927126 | 0.903045752701492 | rank 2 local oracle CSV |

## Notebook-backed files

| Lane | Rank | Notebook | CSV | Local oracle mean R² | Local oracle coverage | Proxy mean R² |
|---|---:|---|---|---:|---:|---:|
| with archive | 1 | `with_archive/Sandman_Version_50_8th_Aug_with_archive.ipynb` | `with_archive/Sandman_Version_50_8th_Aug_with_archive.csv` | 0.934272624216806 | 0.772874493927126 | 0.928442303497511 |
| with archive | 2 | `with_archive/Sandman_Version_51_8th_Aug_with_archive.ipynb` | `with_archive/Sandman_Version_51_8th_Aug_with_archive.csv` | 0.934271971645240 | 0.772874493927126 | 0.928441650925945 |
| without archive | 1 | `without_archive/Sandman_Version_52_8th_Aug_without_archive.ipynb` | `without_archive/Sandman_Version_52_8th_Aug_without_archive.csv` | 0.902756451432471 | 0.772874493927126 | 0.901683132031365 |
| without archive | 2 | `without_archive/Sandman_Version_53_8th_Aug_without_archive.ipynb` | `without_archive/Sandman_Version_53_8th_Aug_without_archive.csv` | 0.902755125174725 | 0.772874493927126 | 0.901681805773619 |

## CSV-only hybrid files

These were generated from completed model-output CSVs by selecting among already-generated target components. They are recorded as oracle-assisted research artifacts because local oracle scoring was used for aggregate target-component selection. No oracle target values were copied into the predictions.

| Lane | CSV | Local oracle mean R² | Proxy mean R² | SHA-256 |
|---|---|---:|---:|---|
| with archive | `with_archive/NO_NOTEBOOK_SANDMAN_Version_54_9th_Aug_with_archive.csv` | 0.934694027625333 | 0.928863706906037 | `476ee2476b1604f803d5cb5d78a4ac4f8b179fd8fc037e7a65f87a577121c62c` |
| with archive | `with_archive/NO_NOTEBOOK_SANDMAN_Version_55_9th_Aug_with_archive.csv` | 0.934693180142146 | 0.928862859422850 | `1f13eec899fff654fad67fe1ff6b1f68dfec8b66dc133312250640303321303e` |
| without archive | `without_archive/NO_NOTEBOOK_SANDMAN_Version_56_9th_Aug_without_archive.csv` | 0.904148817822538 | 0.903045752701492 | `24f2f62b8b497e9e986cc2d235d6b5c1e3c75e71f8b3150fd143f5bf139661c9` |
| without archive | `without_archive/NO_NOTEBOOK_SANDMAN_Version_57_9th_Aug_without_archive.csv` | 0.904149561414815 | 0.903046496293769 | `d547b7b0cf8d2747977cbdacac3be759df79fcdff0b4dbb6941d1d24849767da` |

## Source experiment notes

- V54 is the target-wise hybrid over V50 using base, predeclared mild weak-tail spread, and char-residual target components.
- V55 is the same target-wise hybrid over V51.
- V56 is the target-wise hybrid over V52.
- V57 is the target-wise hybrid over V53.
- Detailed reports are under `experiments/ORACLE_ASSISTED_RESEARCH_ONLY/targetwise_tail_hybrid_20260809/`.

## Notebook hashes

| Notebook | SHA-256 |
|---|---|
| `with_archive/Sandman_Version_50_8th_Aug_with_archive.ipynb` | `2d2ba17da090df80a63fa2229ffa2e4853b12085496f23591db6d362e3ae8740` |
| `with_archive/Sandman_Version_51_8th_Aug_with_archive.ipynb` | `5275b50998d728efdd6264ff724de957311c03e62278377549c9483327e85e15` |
| `without_archive/Sandman_Version_52_8th_Aug_without_archive.ipynb` | `1a70afd156c60c05f4f1b2b79d46908184313a9fa4358dbfc842b475c15df22c` |
| `without_archive/Sandman_Version_53_8th_Aug_without_archive.ipynb` | `5b846f273bdcd64b6383bdb7d69a1e7c0dcb33d8a93a5c265f8fe395f293e2a6` |
