#!/usr/bin/env python3
"""C231: deterministic compound audit v17 with C230 Egb priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c229_clean_component_compound_audit_v16 as c229


SCHEMA = "ppp.round2.c231.clean-component-compound-audit.v17"
C230_ID = "R2-C230-20260805-0624-egb-c180-transfer-guard-v1"
CHANGED_FACTOR = (
    "Insert C230 as first Egb priority under the normal component gate while preserving C229's C228 Tg, "
    "C226 Nc, and existing guarded component priorities."
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def resolve_run_dir() -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir")
    args, _ = parser.parse_known_args()
    if args.run_dir is None:
        return None
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    return run_dir.resolve() if run_dir.is_absolute() else (root / run_dir).resolve()


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    helper = c229.c227.c225.c223.c221.c219.c217.base.c200
    manifest = [
        f"{helper.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in source_hashes.items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def post_patch(run_dir: Path) -> None:
    helper = c229.c227.c225.c223.c221.c219.c217.base.c200
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = metrics.get("schema_version")
    metrics["schema_version"] = SCHEMA
    metrics["changed_factor"] = CHANGED_FACTOR
    metrics["component_priority"] = c229.c227.c225.c223.c221.c219.c217.base.COMPONENT_PRIORITY
    metrics["selection_rule"] = (
        "first completed clean-passing target component in frozen priority order; C230 first for Egb, "
        "C228 first for Tg, and C226 first for Nc under normal component gates; no local_eval/public feedback; "
        "no same-OOF max search"
    )
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["base_c213_runner"] = helper.sha256_file(Path(c229.c227.c225.c223.c221.c219.c217.base.__file__).resolve())
    source_hashes["c217_wrapper_runner"] = helper.sha256_file(Path(c229.c227.c225.c223.c221.c219.c217.__file__).resolve())
    source_hashes["c219_wrapper_runner"] = helper.sha256_file(Path(c229.c227.c225.c223.c221.c219.__file__).resolve())
    source_hashes["c221_wrapper_runner"] = helper.sha256_file(Path(c229.c227.c225.c223.c221.__file__).resolve())
    source_hashes["c223_wrapper_runner"] = helper.sha256_file(Path(c229.c227.c225.c223.__file__).resolve())
    source_hashes["c225_wrapper_runner"] = helper.sha256_file(Path(c229.c227.c225.__file__).resolve())
    source_hashes["c227_wrapper_runner"] = helper.sha256_file(Path(c229.c227.__file__).resolve())
    source_hashes["c229_wrapper_runner"] = helper.sha256_file(Path(c229.__file__).resolve())
    source_hashes["wrapper_runner"] = helper.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "component_priority": c229.c227.c225.c223.c221.c219.c217.base.COMPONENT_PRIORITY,
            "changed_factor": CHANGED_FACTOR,
            "selection_rule": metrics["selection_rule"],
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: `{metrics.get('decision')}`. {CHANGED_FACTOR}\n\n"
        f"Mean parent `{float(metrics.get('mean_parent_r2', float('nan'))):.12f}`; "
        f"compound `{float(metrics.get('mean_candidate_r2', float('nan'))):.12f}`; "
        f"gain `{float(metrics.get('mean_gain', float('nan'))):+.12f}`. "
        f"Gap to 0.95: `{float(metrics.get('gap_to_0_95', float('nan'))):+.12f}`. "
        "Audit-only; no local_eval read; no Kaggle action; no final-notebook action.\n",
        encoding="utf-8",
    )
    rewrite_manifest(run_dir, source_hashes)


def main() -> None:
    run_dir = resolve_run_dir()
    base = c229.c227.c225.c223.c221.c219.c217.base
    base.COMPONENT_PRIORITY["egb"] = [
        C230_ID,
        "R2-C201-20260805-0305-safe-egb-cross-property-stage2-v1",
    ]
    c229.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
