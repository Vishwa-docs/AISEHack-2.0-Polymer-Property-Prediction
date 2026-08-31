#!/usr/bin/env python3
"""C335 no-archive relaxed weak-target co-test meta-calibrator.

Diagnostic child of C327.  It starts from the current C327 no-archive compound
and retries only EI/EPS/NC with a relaxed OOF stability gate.  EEA is excluded
to avoid double-applying the C327 EEA residual update.
"""

from __future__ import annotations

import build_round2_c327_noarchive_cotest_meta_calibrator as c327


c327.ACTIVE_TARGETS = ("ei", "eps", "nc")
c327.DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C327-NOARCHIVE-COTEST-META-CALIBRATOR-20260808.csv"
)
c327.CONFIGS.update(
    {
        "ei": c327.TargetConfig(0.001, 3, -0.020, 0.45, 0.55),
        "eps": c327.TargetConfig(0.001, 3, -0.020, 0.70, 0.55),
        "nc": c327.TargetConfig(0.001, 3, -0.020, 0.080, 0.55),
    }
)


if __name__ == "__main__":
    c327.main()
