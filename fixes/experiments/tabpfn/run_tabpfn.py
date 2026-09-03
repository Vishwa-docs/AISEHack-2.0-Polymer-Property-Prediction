"""Optional grouped-CV TabPFN regression experiment for the scarce targets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import certifi
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from pathlib import Path as _Path
import sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "pure_ml"))
from train import canonical, featurize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path("model_cache"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    # A local, ignored .env lets the user supply their Prior Labs API key without
    # exposing it in source control or chat. Existing shell variables take priority.
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    # macOS framework Python may omit the system trust roots. Point urllib/requests
    # to certifi's verified CA bundle; this preserves TLS validation.
    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
    # TabPFN reads its settings during import. Set this before importing the
    # estimator so no download/lock/cache path can escape this experiment folder.
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(args.model_cache_dir.resolve())
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:
        raise SystemExit("Install the optional package first: ../../isolated_runs/.venv/bin/python -m pip install tabpfn") from exc
    train = pd.read_csv(args.data_dir / "train.csv")
    rows = []
    for target in ("egb", "ei", "eea", "nc", "eps"):
        frame = train[train.target_type.eq(target)].reset_index(drop=True)
        if args.smoke and len(frame) > 250:
            frame = frame.sample(250, random_state=2026).reset_index(drop=True)
        x = featurize(frame.smiles.tolist())
        y = frame.target.to_numpy(float)
        groups = frame.smiles.map(canonical).to_numpy()
        oof = np.zeros(len(frame))
        for tr, va in GroupKFold(n_splits=3).split(x, y, groups):
            model = TabPFNRegressor(random_state=2026)
            model.fit(x[tr], y[tr])
            oof[va] = model.predict(x[va])
        score = float(r2_score(y, oof))
        rows.append({"target": target, "n": len(frame), "folds": 3, "grouped_cv_r2": score})
        print(f"{target}: grouped-CV R2={score:.4f}")
    report = pd.DataFrame(rows)
    report.loc[len(report)] = {"target": "mean", "n": report.n.sum(), "folds": np.nan,
                               "grouped_cv_r2": report.grouped_cv_r2.mean()}
    report.to_csv(args.output_dir / "tabpfn_grouped_cv.csv", index=False)
    print(f"mean scarce-target R2={report.iloc[-1].grouped_cv_r2:.4f}")


if __name__ == "__main__":
    main()
