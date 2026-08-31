#!/usr/bin/env python3
"""C331 no-archive broad co-test residual meta-calibrator.

Small wrapper around the C327 calibrator.  The changed factor is the active
target set/config only: try Tg/Egc/Egb in addition to the weak electronic and
dielectric targets, with conservative clean-OOF gates.  LocalEval/proxy scoring is
still separate and post-freeze only.
"""

from __future__ import annotations

import build_round2_c327_noarchive_cotest_meta_calibrator as c327


c327.ACTIVE_TARGETS = ("tg", "egc", "egb", "ei", "eea", "eps", "nc")
c327.DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C327-NOARCHIVE-COTEST-META-CALIBRATOR-20260808.csv"
)
c327.CONFIGS.update(
    {
        "tg": c327.TargetConfig(0.002, 4, -0.003, 35.0, 0.50),
        "egc": c327.TargetConfig(0.002, 4, -0.003, 0.45, 0.60),
        "egb": c327.TargetConfig(0.002, 4, -0.003, 0.45, 0.60),
        # Re-test EEA over the current C327 base, but keep the same clean gate.
        "eea": c327.TargetConfig(0.003, 4, -0.003, 0.45, 0.75),
        # Keep weak-target settings identical to C327 for comparable gating.
        "ei": c327.TargetConfig(0.003, 4, -0.003, 0.45, 0.75),
        "eps": c327.TargetConfig(0.003, 4, -0.003, 0.70, 0.65),
        "nc": c327.TargetConfig(0.003, 4, -0.003, 0.080, 0.65),
    }
)


if __name__ == "__main__":
    c327.main()
