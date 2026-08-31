"""C273: bounded clean rerun of the corrected C272 EHT-response screen.

This child changes only the output path so the interrupted C272 evidence is
preserved byte-for-byte. It is an exact scientific reproduction, not a new
model configuration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/round2_c271_eht_response_weak_target_screen.py"
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C273-20260805-eht-response-weak-target-screen-rerun-v1"
spec = importlib.util.spec_from_file_location("c273_response_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load C273 response screen")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.RUN = RUN


if __name__ == "__main__":
    module.main()
