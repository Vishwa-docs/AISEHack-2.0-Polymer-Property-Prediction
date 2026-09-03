# TabPFN experiment handoff

TabPFN uses pretrained weights and is research-only. It requires both accepted access
terms and a valid Prior Labs `TABPFN_TOKEN`; `.env` is private, Git-ignored, and must
never be read, displayed, committed, or shared. A non-empty token is not proof that
the license service accepts it.

Run only `run_tabpfn.py` with workspace-local `--output-dir` and `--model-cache-dir`.
Use the 3.11 isolated environment. If licensing fails, report the error category and
ask the user to complete the external licensing step—do not bypass authentication or
TLS checks.
