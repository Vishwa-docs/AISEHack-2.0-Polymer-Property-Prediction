"""C272: fresh corrected child for the C271 EHT response screen."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/round2_c271_eht_response_weak_target_screen.py"
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C272-20260805-eht-response-weak-target-screen-corrected-v1"
spec = importlib.util.spec_from_file_location("c272_response_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load corrected response screen")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.RUN = RUN


if __name__ == "__main__":
    module.main()
