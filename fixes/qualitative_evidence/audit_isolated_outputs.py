"""Audit an isolated notebook output directory without changing it.

The audit intentionally checks for artifact *completeness*, not model quality. It
is safe to run while a notebook is active because it reads the output directory
and writes its own report elsewhere.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_PLOT_ARTIFACTS = (
    "eda/novelty_two_regimes.png",
    "training/parity_plots.png",
    "explainability/fidelity_curves.png",
    "robustness/smiles_invariance.png",
    "generalization/generalization_ladder.png",
    "generalization/applicability_domain.png",
)

# These are emitted by the late evidence-engine cells. Their absence means the
# full qualitative release gate has not been reached, even if earlier plots exist.
REQUIRED_RELEASE_ARTIFACTS = (
    "explanation_agreement.csv",
    "attribution_invariance_per_target.csv",
    "smiles_invariance_graph_violation_summary.csv",
    "fidelity_table.csv",
    "generalization_ladder.csv",
    "conformal_coverage_table.csv",
    "error_uncertainty_correlation.csv",
    "augmentation_experiment.csv",
    "scorecard.md",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.outputs.resolve()
    present_plots = [path for path in REQUIRED_PLOT_ARTIFACTS if (source / path).is_file()]
    missing_plots = [path for path in REQUIRED_PLOT_ARTIFACTS if path not in present_plots]
    present_release = [name for name in REQUIRED_RELEASE_ARTIFACTS if (source / name).is_file()]
    missing_release = [name for name in REQUIRED_RELEASE_ARTIFACTS if name not in present_release]
    files = [path for path in source.rglob("*") if path.is_file()]
    latest = max((path.stat().st_mtime for path in files), default=None)
    promotion_ready = not missing_plots and not missing_release

    stamp = (
        datetime.fromtimestamp(latest, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        if latest is not None
        else "no files found"
    )
    lines = [
        "# Isolated-run output audit",
        "",
        f"- **Source (read only):** `{source}`",
        f"- **Files observed:** {len(files)}",
        f"- **Newest observed artifact:** {stamp}",
        f"- **Promotion ready:** {'YES' if promotion_ready else 'NO'}",
        "",
        "This is a filesystem-completeness audit. It does not prove that a notebook kernel is idle or that a result is scientifically valid.",
        "",
        "## Standard notebook artifacts",
        "",
        "| Status | Artifact |",
        "|---|---|",
        *[f"| present | `{path}` |" for path in present_plots],
        *[f"| missing | `{path}` |" for path in missing_plots],
        "",
        "## Release-gate evidence artifacts",
        "",
        "| Status | Artifact |",
        "|---|---|",
        *[f"| present | `{path}` |" for path in present_release],
        *[f"| missing | `{path}` |" for path in missing_release],
        "",
    ]
    if promotion_ready:
        lines.extend([
            "## Next action",
            "",
            "Run `validate_archive_evidence.py` and `summarize_qualitative.py` against this directory; then compare their results with the claim-to-evidence map before promotion.",
        ])
    else:
        lines.extend([
            "## Next action",
            "",
            "Do not promote or quote release-gated uncertainty claims yet. Let the notebook reach its late evidence-engine and scorecard cells, then rerun this audit.",
        ])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    json_path = args.report.with_suffix(".json")
    json_path.write_text(json.dumps({
        "source": str(source),
        "file_count": len(files),
        "latest_observed_artifact": stamp,
        "present_standard_artifacts": present_plots,
        "missing_standard_artifacts": missing_plots,
        "present_release_artifacts": present_release,
        "missing_release_artifacts": missing_release,
        "promotion_ready": promotion_ready,
    }, indent=2) + "\n")
    print(f"Wrote {args.report}")
    print(f"promotion_ready={promotion_ready}")


if __name__ == "__main__":
    main()
