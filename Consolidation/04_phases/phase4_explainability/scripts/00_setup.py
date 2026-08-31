"""
00_setup.py
===========
Phase 0: verify environment, data presence, and per-target train sizes.
Writes outputs/setup_log.txt. Also runs the compliance scan over sibling
scripts (no ground-truth strings outside the 2 marked scripts).
"""
import importlib.util
import sys
from pathlib import Path
import pandas as pd

from helpers import OUTPUT_DIR, TARGETS, load_train, load_test, data_file

REQUIRED_LIBS = ["numpy", "pandas", "sklearn", "lightgbm", "rdkit",
                 "shap", "matplotlib", "seaborn", "scipy"]


def main():
    lines = []
    def log(msg):
        print(msg)
        lines.append(str(msg))

    log("=== Phase 4 setup ===")
    for lib in REQUIRED_LIBS:
        spec = importlib.util.find_spec(lib)
        if spec is None:
            log(f"  LIB MISSING: {lib}")
            sys.exit(1)
    log("  all required libraries importable")

    train = load_train()
    test = load_test()
    log(f"  train.csv: {len(train)} rows, cols={list(train.columns)}")
    log(f"  test.csv:  {len(test)} rows, cols={list(test.columns)}")
    counts = train["target_type"].value_counts()
    for t in TARGETS:
        log(f"    {t}: train n={int(counts.get(t, 0))}, test n={int((test['target_type'] == t).sum())}")

    for name in ["train.csv", "test.csv", "PI1M.csv", "smile_r3.csv"]:
        p = data_file(name)
        log(f"  data {name}: {'OK ' + str(p) if p.exists() else 'MISSING'}")

    # compliance scan: only 16_khazana_verification.py and F3_oracle_sweep.py
    # may reference ground-truth answer files
    banned = ["oracle", "ORACLE", "Oracle", "sources/", "final_oracle", "oracle_proxy"]
    scripts = sorted(Path(__file__).resolve().parent.glob("*.py"))
    violations = []
    for sp in scripts:
        if sp.name in ("16_khazana_verification.py", "F3_oracle_sweep.py",
                          "helpers.py", "00_setup.py", "G1_html_report.py"):  # scanner itself + report bundler (lists artifact filenames only)
            continue
        text = sp.read_text()
        hits = [b for b in banned if b in text]
        if hits:
            violations.append((sp.name, hits))
    if violations:
        log("  COMPLIANCE WARN (fix before finalizing):")
        for name, hits in violations:
            log(f"    {name}: {hits}")
    else:
        log("  compliance scan: no forbidden strings outside marked scripts")

    with open(OUTPUT_DIR / "setup_log.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    log("00_setup.py DONE -> outputs/setup_log.txt")


if __name__ == "__main__":
    main()
