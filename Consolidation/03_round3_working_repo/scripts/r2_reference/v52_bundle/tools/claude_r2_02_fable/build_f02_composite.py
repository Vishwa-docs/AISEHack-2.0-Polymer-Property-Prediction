"""Build an local_eval-testable F02 ionic-coordinate composite."""
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

def main():
    archive = os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1"
    branch = "with_archive" if archive else "without_archive"
    data = fc.load_data(include_archive=archive)
    root = Path(fc.ROUND2_DIR)
    base_path = root / ("submissions/R2-BEST-KNOWN-CLEAN-COMPOSITE-20260805.csv" if archive else "experiments/CLEAN_OFFICIAL_ONLY/R2-F03-CLEAN-20260805-1924/candidate.csv")
    out = pd.read_csv(base_path).set_index("id")["target"].copy()
    all_need = list(dict.fromkeys(data.train["can"].tolist() + data.test["can"].tolist()))
    partner_pred = {}
    for prop in ("eps", "nc"):
        rows = data.all_labels[data.all_labels["target_type"].eq(prop)].groupby("can")["target"].mean()
        cans = rows.index.tolist()
        model = ExtraTreesRegressor(300, min_samples_leaf=2, random_state=fc.SEED, n_jobs=-1)
        model.fit(np.hstack([f01.descriptor_block(cans), f01.morgan_count_block(cans)]), rows.to_numpy(float))
        pred = model.predict(np.hstack([f01.descriptor_block(all_need), f01.morgan_count_block(all_need)]))
        partner_pred[prop] = dict(zip(all_need, pred))
        partner_pred[prop].update({c: float(v) for c, v in rows.items()})
    pairs = data.wide[data.wide["eps"].notna() & data.wide["nc"].notna()]
    pair_cans = pairs.index.tolist()
    ionic_model = f02.fit_ionic_model(pair_cans, (pairs["eps"] - pairs["nc"] ** 2).to_numpy(float))
    ionic_test = dict(zip(all_need, f02.predict_ionic(ionic_model, all_need)))
    for target, partner in (("eps", "nc"), ("nc", "eps")):
        tr = data.train[data.train["target_type"].eq(target)].reset_index(drop=True)
        te = data.test[data.test["target_type"].eq(target)].reset_index(drop=True)
        cans, tcans = tr["can"].tolist(), te["can"].tolist()
        xf, xt = f01.morgan_count_block(cans), f01.morgan_count_block(tcans)
        xd, xdt = f01.descriptor_block(cans), f01.descriptor_block(tcans)
        b0 = f01.fit_predict_structure_blend(xf, xd, tr["target"].to_numpy(float), xt, xdt)
        observed = np.array([c in data.wide.index and pd.notna(data.wide.loc[c, partner]) for c in tcans])
        pval = np.array([data.wide.loc[c, partner] if c in data.wide.index and pd.notna(data.wide.loc[c, partner]) else partner_pred[partner][c] for c in tcans], float)
        ion = np.array([ionic_test[c] for c in tcans], float)
        if target == "eps": phys = pval ** 2 + ion
        else: phys = np.sqrt(np.clip(pval - ion, 1.0, None))
        pred = np.where(observed, phys, 0.5 * phys + 0.5 * b0)
        out.loc[te["id"].astype(int)] = pred
    result = pd.DataFrame({"id": data.test["id"].astype(int), "target": out.loc[data.test["id"].astype(int)].to_numpy(float)})
    if len(result) != 4940 or not np.isfinite(result["target"]).all(): raise RuntimeError("invalid F02 candidate")
    output_override = os.environ.get("FABLE_OUTPUT_CSV")
    if output_override:
        path = Path(output_override).expanduser().resolve()
    else:
        path = root / "final_submission" / branch / f"R2-F02-COMPOUND-{branch}-candidate.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing F02 output: {path}")
    result.to_csv(path, index=False)
    print({"path": str(path), "rows": len(result)})
if __name__ == "__main__": main()
