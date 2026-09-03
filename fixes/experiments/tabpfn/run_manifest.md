# TabPFN Experiment Run Manifest

## Environment
- **Python**: 3.11.7
- **Virtual Environment**: `/Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3/fixes/experiments/tabpfn/.venv`
- **Base Interpreter**: `/Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3/isolated_runs/.venv/bin/python`

## Package Versions
- tabpfn: 8.5.0
- numpy: 2.4.6
- pandas: 3.0.5
- scikit-learn: 1.9.0
- rdkit: 2026.03.5
- torch: 2.14.0
- certifi: 2026.7.22
- python-dotenv: 1.2.3

## Cache Directories (all local to experiment folder)
- HF_HUB_CACHE: `$PWD/hf_cache/hub`
- HF_ASSETS_CACHE: `$PWD/hf_cache/assets`
- TORCH_HOME: `$PWD/torch_cache`
- PIP_CACHE_DIR: `$PWD/pip_cache`
- TMPDIR: `$PWD/tmp`
- Model Cache: `$PWD/model_cache`
- PYTHONDONTWRITEBYTECODE: 1

## Commands Executed

### Smoke Test
```bash
cd /Users/daver/Desktop/AISEHack\ 2.0\ Polymer\ Property\ Prediction\ Round\ 3/fixes/experiments/tabpfn
HF_HUB_CACHE="$PWD/hf_cache/hub" \
HF_ASSETS_CACHE="$PWD/hf_cache/assets" \
TORCH_HOME="$PWD/torch_cache" \
PIP_CACHE_DIR="$PWD/pip_cache" \
TMPDIR="$PWD/tmp" \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python run_tabpfn.py \
  --data-dir ../../isolated_runs/data \
  --output-dir outputs/smoke \
  --model-cache-dir model_cache \
  --smoke
```
- **Exit Status**: 0 (success)
- **Log File**: `logs/smoke.log`
- **Timestamp**: 2026-09-02 (approx)

### Full Evaluation
```bash
cd /Users/daver/Desktop/AISEHack\ 2.0\ Polymer\ Property\ Prediction\ Round\ 3/fixes/experiments/tabpfn
HF_HUB_CACHE="$PWD/hf_cache/hub" \
HF_ASSETS_CACHE="$PWD/hf_cache/assets" \
TORCH_HOME="$PWD/torch_cache" \
PIP_CACHE_DIR="$PWD/pip_cache" \
TMPDIR="$PWD/tmp" \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python run_tabpfn.py \
  --data-dir ../../isolated_runs/data \
  --output-dir outputs/full \
  --model-cache-dir model_cache
```
- **Exit Status**: 0 (success)
- **Log File**: `logs/full.log`
- **Timestamp**: 2026-09-02 (approx)

## Output Paths
- Smoke Results: `outputs/smoke/tabpfn_grouped_cv.csv`
- Full Results: `outputs/full/tabpfn_grouped_cv.csv`
- Smoke Log: `logs/smoke.log`
- Full Log: `logs/full.log`
- Model Cache: `model_cache/`
- HF Cache: `hf_cache/`
- Torch Cache: `torch_cache/`
- Pip Cache: `pip_cache/`
- Temp: `tmp/`

## Results Summary

### Smoke Test (250 samples per target max)
| Target | N | Folds | Grouped-CV R² |
|--------|---|-------|---------------|
| egb    | 250 | 3 | 0.8669 |
| ei     | 222 | 3 | 0.8008 |
| eea    | 221 | 3 | 0.8701 |
| nc     | 229 | 3 | 0.8263 |
| eps    | 229 | 3 | 0.7831 |
| **mean** | 1151 | — | **0.8295** |

### Full Evaluation (all available samples)
| Target | N | Folds | Grouped-CV R² |
|--------|---|-------|---------------|
| egb    | 337 | 3 | 0.8861 |
| ei     | 222 | 3 | 0.8008 |
| eea    | 221 | 3 | 0.8701 |
| nc     | 229 | 3 | 0.8263 |
| eps    | 229 | 3 | 0.7831 |
| **mean** | 1238 | — | **0.8333** |

## Consistency Check
- Smoke and full results are **internally consistent** for targets with full data in smoke (ei, eea, nc, eps all match exactly).
- egb differs (0.8669 vs 0.8861) because smoke used only 250/337 samples.
- Mean R² improves from 0.8295 (smoke) to 0.8333 (full) as expected with more data.

## Compliance Statements
1. **This is a pretrained, research-only TabPFN result.** TabPFN uses pretrained weights downloaded from Prior Labs' gated Hugging Face repository and requires a valid Prior Labs local-inference license.
2. **This result is not directly comparable with the submission's held-out 0.907551 score** and is **not promoted to the contest pipeline**. The TabPFN experiment is isolated in `fixes/experiments/tabpfn/` and follows the same grouped-CV protocol for fair internal comparison only.
3. All artifacts (caches, models, outputs, logs) remain within `fixes/experiments/tabpfn/`. No files were written to the submission codebase, `isolated_runs/`, `pure_ml/`, `Personal/`, or any global cache.
4. The `.env` file containing the TABPFN_TOKEN was never read, printed, committed, or transmitted.