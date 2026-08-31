#!/usr/bin/env python3
"""Build the Phase_2 experiment suite: 150 real experiment files.

Output layout (per user requirement):
    Phase_2/
      experiments/            one .py per experiment (150 files)
      outputs_and_logs/
        output/               values only: metrics.json, predictions.csv, oof.csv
        logs/                 run logs per experiment
      run.sh                  sequential runner with visible status
      tests/                  pytest suite that verifies every experiment is real
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MECH_PATH = ROOT / "mechanisms.json"
SPEC_PATH = ROOT / "experiment_spec.json"

HEAD_LINES = [
    "#!/usr/bin/env python3",
    '"""{doc}"""',
    "from __future__ import annotations",
    "",
    "import argparse",
    "import json",
    "import sys",
    "import time",
    "from pathlib import Path",
    "",
    "import numpy as np",
    "from rdkit import Chem",
    "",
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
    "from r3_core import data as d",
    "from r3_core import engine as eng",
    "from r3_core import features as f",
    "from r3_core import metrics as m",
    "from r3_core import models as mo",
    "from r3_core import physics as ph",
    "from r3_core import panels as pn",
    "from rdkit.Chem import Crippen, rdMolDescriptors",
    "",
    'EXP_ID = "{exp_id}"',
    'EXP_NAME = "{slug}"',
    "TARGETS = d.TARGETS",
    "SEED = {seed}",
    "",
    "",
]

BODY_LINES = [
    "def run_experiment(output_dir: Path, smoke: bool = False, data_dir: Path | None = None) -> dict[str, object]:",
    "    output_dir = Path(output_dir)",
    "    output_dir.mkdir(parents=True, exist_ok=True)",
    '    print(f"[{EXP_ID}] starting - {title}")',
    "{body}",
    "    metrics = eng.run_protocol(",
    "        name=EXP_NAME, exp_id=EXP_ID, output_dir=output_dir,",
    "        feature_fn=feature_fn, model_fn=model_fn,",
    "        n_splits={splits}, seed=SEED, targets=TARGETS, data_dir=data_dir, smoke=smoke,",
    "    )",
    '    print("mean OOF R2 =", metrics.get("mean_r2"))',
    "    return metrics",
    "",
    "",
]

FOOTER_LINES = [
    "def main() -> None:",
    "    parser = argparse.ArgumentParser()",
    '    parser.add_argument("--output", default="outputs_and_logs/output/" + EXP_NAME, help="output directory")',
    '    parser.add_argument("--smoke", action="store_true", help="fast smoke mode")',
    '    parser.add_argument("--data-dir", default=None, help="official Dataset dir")',
    "    args = parser.parse_args()",
    "    metrics = run_experiment(Path(args.output), smoke=args.smoke,",
    "                             data_dir=Path(args.data_dir) if args.data_dir else None)",
    '    print(json.dumps({"exp_id": EXP_ID, "mean_r2": metrics.get("mean_r2"),',
    '                      "per_target": metrics.get("per_target", {})}, indent=2))',
    "",
    "",
    'if __name__ == "__main__":',
    "    main()",
]


def build() -> None:
    mechs = json.loads(MECH_PATH.read_text(encoding="utf-8"))
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    exp_dir = ROOT / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    num = 0
    for phase in spec["phases"]:
        code = phase["phase"]
        title = phase["title"]
        for slot in phase["ids"]:
            num += 1
            slug = f"{code.lower()}{slot}"
            exp_id = f"R3-{code}{slot}-20260827-{slug}"
            mech = mechs[slot]
            body = "\n".join(mech["body"])
            splits = int(mech.get("splits", 5))
            doc = f"R3-{code}{slot} [{slug}] - Phase {code}: {title}. Experiment {num}/150. Real grouped-CV pipeline; reads ONLY official Dataset/ inputs."
            lines = []
            for line in HEAD_LINES:
                lines.append(line)
            for line in BODY_LINES:
                lines.append(line)
            lines.extend(FOOTER_LINES)
            text = "\n".join(lines) + "\n"
            text = text.replace("{doc}", doc)
            text = text.replace("{exp_id}", exp_id)
            text = text.replace("{slug}", slug)
            text = text.replace("{seed}", str(2026 + num))
            text = text.replace("{title}", title)
            text = text.replace("{splits}", str(splits))
            text = text.replace("{body}", body)
            path = exp_dir / f"exp{num:03d}_{slug}.py"
            path.write_text(text, encoding="utf-8")
    print(f"wrote {num} experiment files")


if __name__ == "__main__":
    build()
