#!/usr/bin/env python3
"""Validate the fixed strict-invariance demo panel without training any model."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from primitive_repeat import normalize_linear_repeat  # noqa: E402
from inference import Predictor, TARGETS  # noqa: E402


def main() -> None:
    panel = json.loads((HERE / "panel.json").read_text())
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pred = Predictor(str(ROOT / "weights" / "polymer_weights.joblib"))
    records: list[dict] = []
    for variant in panel["variants"]:
        norm = normalize_linear_repeat(variant["smiles"])
        if norm.status != "strict" or norm.normalized != panel["expected_primitive"]:
            raise AssertionError(f"normalization failed for {variant['label']}: {norm}")
        for target in TARGETS:
            result = pred.predict(norm.normalized, target, mode="model")
            records.append({"family": panel["family"], "variant": variant["label"],
                            "kind": variant["kind"], "input_psmiles": variant["smiles"],
                            "normalized_psmiles": norm.normalized,
                            "repeat_count": norm.repeat_count, "target": target,
                            "prediction": result["value"], "unit": result["unit"],
                            "applicability": result["ad_tier"]})
    tab = pd.DataFrame(records)
    spread = tab.groupby("target", as_index=False).prediction.agg(["min", "max", "std"]).reset_index()
    spread["range"] = spread["max"] - spread["min"]
    if not np.allclose(spread["range"], 0.0, atol=1e-12, rtol=0.0):
        raise AssertionError("normalised predictions disagree across equivalent panel forms")
    tab.to_csv(out / "invariance_panel_predictions.csv", index=False)
    spread.to_csv(out / "invariance_panel_summary.csv", index=False)

    order = [v["label"] for v in panel["variants"]]
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#66a61e"]
    lines = []
    for i, label in enumerate(order):
        row = tab[tab.variant == label].iloc[0]
        y = 86 + i * 56
        lines.append(
            f'<circle cx="38" cy="{y - 6}" r="10" fill="{colors[i]}"/>'
            f'<text x="64" y="{y}" font-size="18" font-family="Arial, sans-serif">'
            f'{label}: {row.input_psmiles}  →  {row.normalized_psmiles}</text>')
    svg = ("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1150\" height=\"340\" "
           "viewBox=\"0 0 1150 340\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
           "<text x=\"32\" y=\"38\" font-size=\"22\" font-family=\"Arial, sans-serif\" "
           "font-weight=\"700\">Strict representation check: one primitive repeat</text>"
           "<text x=\"32\" y=\"62\" font-size=\"15\" font-family=\"Arial, sans-serif\" fill=\"#555\">"
           "Translated and repeated PEO forms normalize before model inference</text>"
           + "".join(lines) + "</svg>")
    (out / "invariance_normalization_panel.svg").write_text(svg)

    digest = hashlib.sha256((out / "invariance_panel_predictions.csv").read_bytes()).hexdigest()
    manifest = {"status": "PASS", "panel": panel, "targets": TARGETS,
                "prediction_csv_sha256": digest, "max_prediction_range": float(spread["range"].max()),
                "scope": "validated unbranched terminal-star linear PSMILES grammar"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS strict panel; max target prediction range:", manifest["max_prediction_range"])
    print("results:", out)


if __name__ == "__main__":
    main()
