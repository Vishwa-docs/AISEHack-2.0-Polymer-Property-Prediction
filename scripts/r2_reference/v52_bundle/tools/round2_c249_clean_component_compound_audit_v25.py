#!/usr/bin/env python3
"""C249: deterministic compound audit v25 with C248 Egb priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c245_clean_component_compound_audit_v24 as c245


SCHEMA = "ppp.round2.c249.clean-component-compound-audit.v25"
C248_ID = "R2-C248-20260805-0839-egb-low-gap-abstaining-coupled-route-v1"
CHANGED_FACTOR = (
    "Insert C248 Egb low-gap abstaining coupled route as first Egb priority under the normal component gate while "
    "preserving C245's C244 Tg priority, older Tg fallbacks, C242/C240/C236/C234/C226 Nc fallbacks, "
    "and existing EPS/Ei/Eea/Egc priorities."
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


def helper() -> Any:
    return c245.helper()


def base_module() -> Any:
    return c245.base_module()


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    h = helper()
    manifest = [
        f"{h.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in source_hashes.items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def post_patch(run_dir: Path) -> None:
    h = helper()
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["base_schema_version"] = metrics.get("schema_version")
    metrics["schema_version"] = SCHEMA
    metrics["changed_factor"] = CHANGED_FACTOR
    metrics["component_priority"] = base_module().COMPONENT_PRIORITY
    metrics["selection_rule"] = (
        "first completed clean-passing target component in frozen priority order; C248 first for Egb, "
        "C244 first for Tg; no local_eval/public feedback; no same-OOF max search"
    )
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["c245_wrapper_runner"] = h.sha256_file(Path(c245.__file__).resolve())
    source_hashes["wrapper_runner"] = h.sha256_file(Path(__file__).resolve())
    source_hashes["runner"] = source_hashes["wrapper_runner"]
    metrics["source_hashes"] = source_hashes
    write_json(metrics_path, metrics)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "component_priority": base_module().COMPONENT_PRIORITY,
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
    base = base_module()
    base.COMPONENT_PRIORITY["tg"] = [
        c245.C244_ID,
        c245.c243.c241.c239.c237.c235.c233.C232_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.C228_ID,
        "R2-C208-20260805-0352-tg-robust-group-measurement-v1",
    ]
    base.COMPONENT_PRIORITY["egb"] = [
        C248_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.C230_ID,
        "R2-C201-20260805-0305-safe-egb-cross-property-stage2-v1",
    ]
    base.COMPONENT_PRIORITY["nc"] = [
        c245.c243.C242_ID,
        c245.c243.c241.C240_ID,
        c245.c243.c241.c239.c237.C236_ID,
        c245.c243.c241.c239.c237.c235.C234_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.C226_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C224_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C222_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C218_ID,
        "R2-C212-20260805-0422-nc-robust-rank-loss-v1",
        "R2-C210-20260805-0415-nc-optical-dispersion-gap-v1",
        "R2-C202-20260805-0315-nc-support-uncertainty-refractivity-v1",
        "R2-C197-20260805-0237-nc-c195-consensus-gated-v1",
        "R2-C195-20260805-0215-nc-nearmiss-residual-diversity-v1",
        "R2-C191-20260805-0027-nested-predicted-eps-to-nc-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    base.COMPONENT_PRIORITY["eps"] = [
        c245.c243.c241.c239.C238_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C224_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C222_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C216_ID,
        "R2-C214-20260805-0440-eps-ionic-full-amplitude-v1",
        "R2-C190-20260805-0023-ionic-eps-reproduction-v3",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    base.COMPONENT_PRIORITY["ei"] = [
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C224_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C222_ID,
        c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.c225.C220_ID,
        "R2-C199-20260805-0254-ei-c196-transfer-guard-v1",
        "R2-C196-20260805-0225-ei-ffox-shrinkage-confirmation-v1",
        "R2-C194-20260805-0152-safe-ei-cross-property-stage2-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    base.c203.metric_passes = c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.guarded_metric_passes
    base.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
