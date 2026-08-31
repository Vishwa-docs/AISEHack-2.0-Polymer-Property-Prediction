#!/usr/bin/env python3
"""C255: deterministic compound audit v27 with C254 Tg priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import round2_c253_clean_component_compound_audit_v26 as c253


SCHEMA = "ppp.round2.c255.clean-component-compound-audit.v27"
C254_ID = "R2-C254-20260805-0909-tg-backbone-pendant-rigidity-v1"
CHANGED_FACTOR = (
    "Insert C254 Tg backbone/pendant rigidity support-gated residual as first Tg priority under the normal "
    "component gate while preserving C253's C252 Nc priority, C249's C248 Egb priority, older fallbacks, "
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
    return c253.helper()


def base_module() -> Any:
    return c253.base_module()


def rewrite_manifest(run_dir: Path, source_hashes: dict[str, str]) -> None:
    h = helper()
    manifest = [
        f"{h.sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in sorted(source_hashes.items()):
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
        "first completed clean-passing target component in frozen priority order; C254 first for Tg, "
        "C252 first for Nc, C248 first for Egb; no local_eval/public feedback; no same-OOF max search"
    )
    source_hashes = dict(metrics.get("source_hashes", {}))
    source_hashes["c253_wrapper_runner"] = h.sha256_file(Path(c253.__file__).resolve())
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
        C254_ID,
        "R2-C244-20260805-0821-tg-median-residual-stack-v1",
        "R2-C232-20260805-0650-tg-replicate-reliability-feature-v1",
        "R2-C228-20260805-0616-tg-c208-transfer-guard-v1",
        "R2-C208-20260805-0352-tg-robust-group-measurement-v1",
    ]
    base.COMPONENT_PRIORITY["egc"] = [
        "R2-C207-20260805-0344-egc-c180-transfer-guard-v1",
    ]
    base.COMPONENT_PRIORITY["egb"] = [
        "R2-C248-20260805-0839-egb-low-gap-abstaining-coupled-route-v1",
        "R2-C230-20260805-0624-egb-c180-transfer-guard-v1",
        "R2-C201-20260805-0305-safe-egb-cross-property-stage2-v1",
    ]
    base.COMPONENT_PRIORITY["ei"] = [
        "R2-C224-20260805-0553-source-priority-label-aggregation-v1",
        "R2-C222-20260805-0540-structure-semantics-weaktarget-v1",
        "R2-C220-20260805-0510-ei-electro-polar-autocorr-v1",
        "R2-C199-20260805-0254-ei-c196-transfer-guard-v1",
        "R2-C196-20260805-0225-ei-ffox-shrinkage-confirmation-v1",
        "R2-C194-20260805-0152-safe-ei-cross-property-stage2-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    base.COMPONENT_PRIORITY["eea"] = [
        "R2-C204-20260805-0323-safe-eea-gap-identity-stage2-v1",
        "R2-C189-20260804-ffox-eea-confirmation-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    base.COMPONENT_PRIORITY["nc"] = [
        "R2-C252-20260805-0856-nc-eps-ionic-projection-v1",
        "R2-C242-20260805-0754-nc-nearmiss-stability-ensemble-v1",
        "R2-C240-20260805-0733-nc-electro-polar-autocorr-v1",
        "R2-C236-20260805-0710-nc-backbone-pendant-polarizability-v1",
        "R2-C234-20260805-0700-nc-replicate-reliability-feature-v1",
        "R2-C226-20260805-0607-nc-c180-transfer-guard-v1",
        "R2-C224-20260805-0553-source-priority-label-aggregation-v1",
        "R2-C222-20260805-0540-structure-semantics-weaktarget-v1",
        "R2-C218-20260805-0500-nc-high-tail-ordinal-residual-v1",
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
        "R2-C238-20260805-0721-eps-bond-polarity-orientational-residual-v1",
        "R2-C224-20260805-0553-source-priority-label-aggregation-v1",
        "R2-C222-20260805-0540-structure-semantics-weaktarget-v1",
        "R2-C216-20260805-0450-eps-high-tail-ordinal-residual-v1",
        "R2-C214-20260805-0440-eps-ionic-full-amplitude-v1",
        "R2-C190-20260805-0023-ionic-eps-reproduction-v3",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ]
    base.c203.metric_passes = c253.c249.c245.c243.c241.c239.c237.c235.c233.c231.c229.c227.guarded_metric_passes
    base.main()
    if run_dir is not None:
        post_patch(run_dir)


if __name__ == "__main__":
    main()
