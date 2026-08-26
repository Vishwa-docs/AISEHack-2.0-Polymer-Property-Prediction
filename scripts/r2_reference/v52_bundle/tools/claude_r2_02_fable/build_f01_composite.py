"""Build the strongest current F01 composite for one input branch."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fable_common as fc
import F01_ei_eea_egb_chain_engine as f01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_candidate_path(path: Path, branch: str, role: str) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if branch == "without_archive":
        if "/with_archive/" in low or "/archive/" in low:
            raise RuntimeError(f"Refusing archive/cross-branch {role} path: {path}")
        if role != "output" and "/without_archive/" not in low:
            raise RuntimeError(f"{role} must be a without_archive path: {path}")
    elif branch == "with_archive":
        if "/without_archive/" in low:
            raise RuntimeError(f"Refusing opposite-branch {role} path: {path}")
        if role != "output" and "/with_archive/" not in low and "r2-best-known-clean-composite" not in low:
            raise RuntimeError(f"{role} must be a with_archive path: {path}")
    else:
        raise RuntimeError(f"Invalid branch: {branch}")


def full_target_prediction(data, target: str, partner_predictions: dict[str, dict[str, float]]) -> pd.Series:
    train = data.train[data.train["target_type"].eq(target)].reset_index(drop=True)
    test = data.test[data.test["target_type"].eq(target)].reset_index(drop=True)
    cans = train["can"].tolist()
    test_cans = test["can"].tolist()
    y = train["target"].to_numpy(float)
    xf = f01.morgan_count_block(cans)
    xd = f01.descriptor_block(cans)
    xf_test = f01.morgan_count_block(test_cans)
    xd_test = f01.descriptor_block(test_cans)

    partners = f01.PARTNERS[target]
    L = np.full((len(cans), len(partners)), np.nan)
    Lt = np.full((len(test_cans), len(partners)), np.nan)
    for j, prop in enumerate(partners):
        if prop not in data.wide.columns:
            continue
        L[:, j] = [data.wide.loc[c, prop] if c in data.wide.index and pd.notna(data.wide.loc[c, prop]) else np.nan for c in cans]
        Lt[:, j] = [data.wide.loc[c, prop] if c in data.wide.index and pd.notna(data.wide.loc[c, prop]) else np.nan for c in test_cans]

    P = np.array([[partner_predictions[p][c] for p in partners] for c in cans])
    Pt = np.array([[partner_predictions[p][c] for p in partners] for c in test_cans])
    fill = np.where(np.isnan(L), P, L)
    fill_t = np.where(np.isnan(Lt), Pt, Lt)
    isp = np.isnan(L).astype(float)
    isp_t = np.isnan(Lt).astype(float)

    comps, signs = f01.IDENTITY[target]
    idx = [partners.index(c) for c in comps]
    base = sum(s * fill[:, i] for s, i in zip(signs, idx))
    base_t = sum(s * fill_t[:, i] for s, i in zip(signs, idx))
    hub = HuberRegressor().fit(base.reshape(-1, 1), y)
    pred = hub.predict(base_t.reshape(-1, 1))
    if f01.IDENTITY_RESID[target] > 0:
        model = ExtraTreesRegressor(500, min_samples_leaf=3, random_state=fc.SEED, n_jobs=-1)
        model.fit(np.hstack([xd, fill, isp]), y - hub.predict(base.reshape(-1, 1)))
        pred += model.predict(np.hstack([xd_test, fill_t, isp_t]))
    return pd.Series(pred, index=test["id"].astype(int))


def main() -> None:
    include_archive = os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1"
    branch = "with_archive" if include_archive else "without_archive"
    data = fc.load_data(include_archive=include_archive)
    base_override = os.environ.get("FABLE_BASE_CSV")
    if base_override:
        base_path = Path(base_override).expanduser().resolve()
    else:
        if include_archive:
            base_path = Path(fc.ROUND2_DIR) / "submissions/R2-BEST-KNOWN-CLEAN-COMPOSITE-20260805.csv"
        else:
            base_path = Path(fc.ROUND2_DIR) / "experiments/CLEAN_OFFICIAL_ONLY/R2-F03-CLEAN-20260805-1924/candidate.csv"
    guard_candidate_path(base_path, branch, "base")
    base = pd.read_csv(base_path)
    if list(base.columns) != ["id", "target"] or len(base) != len(data.test):
        raise RuntimeError("base candidate schema mismatch")
    out = base.set_index("id")["target"].copy()
    need = list(dict.fromkeys(data.train["can"].tolist() + data.test["can"].tolist()))
    partner_predictions = {}
    needed_props = sorted({p for target in ("ei", "eea", "egb") for p in f01.PARTNERS[target]})
    for prop in needed_props:
        rows = data.all_labels[data.all_labels["target_type"].eq(prop)].groupby("can")["target"].mean()
        pcans = rows.index.tolist()
        xf = f01.morgan_count_block(pcans)
        xd = f01.descriptor_block(pcans)
        xn = f01.morgan_count_block(need)
        xdn = f01.descriptor_block(need)
        partner_model = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=fc.SEED, n_jobs=-1
        )
        partner_model.fit(np.hstack([xd, xf]), rows.to_numpy(float))
        pred = partner_model.predict(np.hstack([xdn, xn]))
        partner_predictions[prop] = dict(zip(need, pred))
        for can, value in rows.items():
            partner_predictions[prop][can] = float(value)
    for target in ("ei", "eea", "egb"):
        pred = full_target_prediction(data, target, partner_predictions)
        ids = data.test.loc[data.test["target_type"].eq(target), "id"].astype(int)
        out.loc[ids] = pred.loc[ids]
    result = pd.DataFrame({"id": data.test["id"].astype(int), "target": out.loc[data.test["id"].astype(int)].to_numpy(float)})
    if len(result) != 4940 or result["id"].duplicated().any() or not np.isfinite(result["target"]).all():
        raise RuntimeError("composite validation failed")
    output_override = os.environ.get("FABLE_OUTPUT_CSV")
    if output_override:
        path = Path(output_override).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing F01 output: {path}")
        guard_candidate_path(path, branch, "output")
    else:
        root = Path(fc.ROUND2_DIR) / "final_submission" / branch
        root.mkdir(parents=True, exist_ok=True)
        version = "V2" if include_archive else "V3"
        path = root / f"R2-BEST-COMPOUND-{branch}-{version}.csv"
        if path.exists():
            raise RuntimeError(
                "Refusing to overwrite the current branch final. Set FABLE_OUTPUT_CSV "
                "to a new versioned path under experiments/final_submission_runs/."
            )
    result.to_csv(path, index=False)
    manifest_override = os.environ.get("FABLE_OUTPUT_MANIFEST")
    if manifest_override:
        manifest_path = Path(manifest_override).expanduser().resolve()
    else:
        manifest_path = path.with_suffix(".manifest.json")
    guard_candidate_path(manifest_path, branch, "output")
    if manifest_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing F01 manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "ppp.round2.f01-composite.v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": "CLEAN_OFFICIAL_ONLY",
        "branch": branch,
        "include_archive": include_archive,
        "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
        "targets_replaced": ["ei", "eea", "egb"],
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "output": {"path": str(path), "sha256": sha256_file(path), "rows": len(result), "bytes": path.stat().st_size},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print({"path": str(path), "rows": len(result), "branch": branch, "manifest": str(manifest_path)})


if __name__ == "__main__":
    main()
