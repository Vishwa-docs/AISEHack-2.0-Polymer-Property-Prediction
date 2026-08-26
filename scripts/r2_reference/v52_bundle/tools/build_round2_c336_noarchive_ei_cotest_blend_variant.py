#!/usr/bin/env python3
"""C336/C337 no-archive EI-only co-test blend variant.

Diagnostic wrapper around C327.  It starts from C327, changes only EI, and uses
a caller-provided residual blend weight.  No local_eval/external_label file is read by the
builder.
"""

from __future__ import annotations

import argparse
import sys

import build_round2_c327_noarchive_cotest_meta_calibrator as c327


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ei-blend-weight", type=float, default=0.55)
    args, rest = parser.parse_known_args()
    c327.ACTIVE_TARGETS = ("ei",)
    c327.DEFAULT_BASE = (
        "experiments/final_submission_runs/without_archive/"
        "R2-C327-NOARCHIVE-COTEST-META-CALIBRATOR-20260808.csv"
    )
    c327.CONFIGS.update(
        {
            "ei": c327.TargetConfig(0.001, 3, -0.020, 0.45, float(args.ei_blend_weight)),
        }
    )
    sys.argv = [sys.argv[0], *rest]
    c327.main()


if __name__ == "__main__":
    main()
