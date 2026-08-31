"""F04 exploratory Gaussian-process candidate, trained from official inputs."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fable_common as fc
import F01_ei_eea_egb_chain_engine as f01

def main():
    archive = os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1"
    branch = "with_archive" if archive else "without_archive"
    data = fc.load_data(include_archive=archive)
    root = Path(fc.ROUND2_DIR)
    base = pd.read_csv(root / ("submissions/R2-BEST-KNOWN-CLEAN-COMPOSITE-20260805.csv" if archive else "experiments/CLEAN_OFFICIAL_ONLY/R2-F03-CLEAN-20260805-1924/candidate.csv"))
    out = base.set_index("id")["target"].copy()
    for target in fc.TARGETS:
        tr = data.train[data.train["target_type"].eq(target)].reset_index(drop=True)
        te = data.test[data.test["target_type"].eq(target)].reset_index(drop=True)
        cans, tcans = tr["can"].tolist(), te["can"].tolist()
        xtr = np.hstack([f01.descriptor_block(cans), f01.morgan_count_block(cans)])
        xte = np.hstack([f01.descriptor_block(tcans), f01.morgan_count_block(tcans)])
        scaler = StandardScaler().fit(np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0))
        a = scaler.transform(np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0))
        b = scaler.transform(np.nan_to_num(xte, nan=0.0, posinf=0.0, neginf=0.0))
        ncomp = min(32, a.shape[0] - 1, a.shape[1])
        pca = PCA(n_components=ncomp, random_state=fc.SEED).fit(a)
        a, b = pca.transform(a), pca.transform(b)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.05, (1e-4, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=fc.SEED)
        gp.fit(a, tr["target"].to_numpy(float))
        pred = gp.predict(b)
        out.loc[te["id"].astype(int)] = pred
    result = pd.DataFrame({"id": data.test["id"].astype(int), "target": out.loc[data.test["id"].astype(int)].to_numpy(float)})
    path = root / "final_submission" / branch / f"R2-F04-GPR-{branch}-candidate.csv"
    result.to_csv(path, index=False)
    print({"path": str(path), "rows": len(result)})
if __name__ == "__main__": main()
