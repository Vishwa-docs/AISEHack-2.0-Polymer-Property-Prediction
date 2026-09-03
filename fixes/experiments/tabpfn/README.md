# TabPFN experiment scaffold

TabPFN is evaluated here as a **controlled tabular regressor**, not as an ensemble-discovery oracle. It will receive the same fixed RDKit feature matrix, target-wise grouped folds, and target-wise mean-R² accounting as the classical baseline. The experiment is intentionally restricted to the scarce targets first (`egb`, `ei`, `eea`, `nc`, `eps`), where tabular priors may have the most value.

**Compliance boundary:** TabPFN downloads and uses pretrained weights. Its result is research-only and cannot be substituted into the competition pipeline or relabelled as a plain-ML result under the codebase’s no-pretrained-weights rule.

## One-time access approval

TabPFN's current weights are in a gated Hugging Face repository, and TabPFN 8.5
also requires a Prior Labs local-inference license. The project does not contain
credentials and does not attempt to bypass either gate. To authorize the experiment:

1. Sign in to Hugging Face and accept the terms at <https://huggingface.co/Prior-Labs/tabpfn_3>.
2. From this directory, authenticate the isolated environment with a read token:

   ```bash
   ../../isolated_runs/.venv/bin/hf auth login
   ```

3. Open <https://ux.priorlabs.ai>, log in, accept the license on the **Licenses**
   tab, then create/copy an API key from <https://ux.priorlabs.ai/account>.
4. In this folder, create a private token file from the provided template and put
   the API key after `TABPFN_TOKEN=`. Do not paste the key into chat.

   ```bash
   cp .env.example .env
   open -e .env
   ```

   `.env` is ignored by Git and is read only by this experiment. Tell me when it
   is saved, and I will rerun the already-scaffolded experiment.

The runner uses the standard `certifi` CA bundle for macOS framework Python; it
does not disable TLS validation.

## Run after approval

```bash
../../isolated_runs/.venv/bin/python -m pip install tabpfn
../../isolated_runs/.venv/bin/python run_tabpfn.py \
  --data-dir ../../isolated_runs/data \
  --output-dir outputs \
  --model-cache-dir model_cache --smoke
```

Do not copy a result into the competition codebase unless it improves the same grouped-CV protocol and can be reproduced from this folder without external labels. If the model helps, replace the TabPFN dependency with a documented plain estimator only after confirming that the replacement preserves the measured improvement.
