"""Build a branch-safe F02-B3 joint eps/nc consistency candidate.

This extends the existing F02-B2 full-test candidate by reconciling co-test
eps/nc rows so ``eps = nc^2 + ionic`` for a shared predicted ionic response.
It reads only official inputs and an explicit base candidate artifact.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fable_common as fc
import F01_ei_eea_egb_chain_engine as f01
import F02_eps_nc_ionic_engine as f02


def reconcile_nc(eps0: float, nc0: float, ionic: float, eps_scale: float, nc_scale: float) -> float:
    ionic = max(float(ionic), f02.MIN_IONIC)
    implied = np.sqrt(max(float(eps0) - ionic, 1.0e-9))
    center = 0.5 * (float(nc0) + implied)
    lo = max(0.2, min(float(nc0), implied, center) - 1.0)
    hi = max(float(nc0), implied, center) + 1.0
    grid = np.linspace(lo, hi, 801)
    eps_grid = grid * grid + ionic
    loss = ((eps_grid - float(eps0)) / max(eps_scale, 1.0e-6)) ** 2 + (
        (grid - float(nc0)) / max(nc_scale, 1.0e-6)
    ) ** 2
    return float(grid[int(np.argmin(loss))])


def main() -> None:
    archive = os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1"
    branch = "with_archive" if archive else "without_archive"
    data = fc.load_data(include_archive=archive)
    root = Path(fc.ROUND2_DIR)
    base_override = os.environ.get("FABLE_BASE_CSV")
    if base_override:
        base_path = Path(base_override).expanduser().resolve()
    else:
        base_path = root / (
            "final_submission/with_archive/R2-BEST-COMPOUND-with_archive-V2.csv"
            if archive
            else "final_submission/without_archive/R2-BEST-COMPOUND-without_archive-V3.csv"
        )
    output_override = os.environ.get("FABLE_OUTPUT_CSV")
    if not output_override:
        raise RuntimeError("Set FABLE_OUTPUT_CSV to a new versioned output path")
    path = Path(output_override).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing F02-B3 output: {path}")

    out = pd.read_csv(base_path).set_index("id")["target"].copy()
    all_need = list(dict.fromkeys(data.train["can"].tolist() + data.test["can"].tolist()))
    partner_pred: dict[str, dict[str, float]] = {}
    for prop in ("eps", "nc"):
        rows = data.all_labels[data.all_labels["target_type"].eq(prop)].groupby("can")["target"].mean()
        cans = rows.index.tolist()
        model = ExtraTreesRegressor(300, min_samples_leaf=2, random_state=fc.SEED, n_jobs=-1)
        model.fit(
            np.hstack([f01.descriptor_block(cans), f01.morgan_count_block(cans)]),
            rows.to_numpy(float),
        )
        pred = model.predict(np.hstack([f01.descriptor_block(all_need), f01.morgan_count_block(all_need)]))
        partner_pred[prop] = dict(zip(all_need, pred))
        partner_pred[prop].update({c: float(v) for c, v in rows.items()})

    pairs = data.wide[data.wide["eps"].notna() & data.wide["nc"].notna()]
    pair_cans = pairs.index.tolist()
    ionic_model = f02.fit_ionic_model(pair_cans, (pairs["eps"] - pairs["nc"] ** 2).to_numpy(float))
    ionic_test = dict(zip(all_need, np.maximum(f02.MIN_IONIC, f02.predict_ionic(ionic_model, all_need))))
    target_frames: dict[str, pd.DataFrame] = {}
    raw_pred: dict[str, pd.Series] = {}

    for target, partner in (("eps", "nc"), ("nc", "eps")):
        tr = data.train[data.train["target_type"].eq(target)].reset_index(drop=True)
        te = data.test[data.test["target_type"].eq(target)].reset_index(drop=True)
        cans, tcans = tr["can"].tolist(), te["can"].tolist()
        xf, xt = f01.morgan_count_block(cans), f01.morgan_count_block(tcans)
        xd, xdt = f01.descriptor_block(cans), f01.descriptor_block(tcans)
        b0 = f01.fit_predict_structure_blend(xf, xd, tr["target"].to_numpy(float), xt, xdt)
        observed = np.array(
            [c in data.wide.index and pd.notna(data.wide.loc[c, partner]) for c in tcans],
            dtype=bool,
        )
        pval = np.array(
            [
                data.wide.loc[c, partner]
                if c in data.wide.index and pd.notna(data.wide.loc[c, partner])
                else partner_pred[partner][c]
                for c in tcans
            ],
            dtype=float,
        )
        ion = np.array([ionic_test[c] for c in tcans], dtype=float)
        if target == "eps":
            phys = pval**2 + ion
        else:
            phys = np.sqrt(np.clip(pval - ion, 1.0, None))
        pred = np.where(observed, phys, 0.5 * phys + 0.5 * b0)
        target_frames[target] = te.assign(can=tcans)
        raw_pred[target] = pd.Series(pred, index=te["id"].astype(int))

    eps_by_can = dict(zip(target_frames["eps"]["can"], target_frames["eps"]["id"].astype(int)))
    nc_by_can = dict(zip(target_frames["nc"]["can"], target_frames["nc"]["id"].astype(int)))
    co_cans = sorted(set(eps_by_can) & set(nc_by_can))
    eps_scale = float(data.train[data.train["target_type"].eq("eps")]["target"].std())
    nc_scale = float(data.train[data.train["target_type"].eq("nc")]["target"].std())
    reconciled = 0
    for can in co_cans:
        eps_id = eps_by_can[can]
        nc_id = nc_by_can[can]
        eps0 = float(raw_pred["eps"].loc[eps_id])
        nc0 = float(raw_pred["nc"].loc[nc_id])
        nc_new = reconcile_nc(eps0, nc0, ionic_test[can], eps_scale, nc_scale)
        eps_new = nc_new * nc_new + max(float(ionic_test[can]), f02.MIN_IONIC)
        raw_pred["eps"].loc[eps_id] = eps_new
        raw_pred["nc"].loc[nc_id] = nc_new
        reconciled += 1

    for target in ("eps", "nc"):
        te = target_frames[target]
        out.loc[te["id"].astype(int)] = raw_pred[target].loc[te["id"].astype(int)].to_numpy(float)

    result = pd.DataFrame(
        {"id": data.test["id"].astype(int), "target": out.loc[data.test["id"].astype(int)].to_numpy(float)}
    )
    if len(result) != 4940 or result["id"].duplicated().any() or not np.isfinite(result["target"]).all():
        raise RuntimeError("invalid F02-B3 candidate")
    result.to_csv(path, index=False)
    print({"path": str(path), "rows": len(result), "branch": branch, "reconciled_co_test_structures": reconciled})


if __name__ == "__main__":
    main()
