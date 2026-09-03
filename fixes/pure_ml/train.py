"""Train a portable, target-specific RDKit + ExtraTrees polymer baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

SEED = 2026
TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)
RDLogger.DisableLog("rdApp.warning")


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def featurize(smiles: list[str], n_bits: int = 1024) -> np.ndarray:
    rows: list[np.ndarray] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
        desc = np.asarray([fn(mol) for _, fn in Descriptors._descList], dtype=np.float64)
        desc[~np.isfinite(desc)] = 0.0
        # A few RDKit descriptors can be finite yet astronomically large for unusual
        # repeat units; keep the tabular learner inside its documented numeric range.
        np.clip(desc, -1_000_000.0, 1_000_000.0, out=desc)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        bits = np.zeros(n_bits, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, bits)
        rows.append(np.concatenate((desc, bits)))
    return np.vstack(rows)


def build_model(n_estimators: int, n_jobs: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        max_features=0.7,
        n_jobs=n_jobs,
        random_state=SEED,
    )


def evaluate_target(frame: pd.DataFrame, features: np.ndarray, smoke: bool, n_estimators: int, n_jobs: int) -> tuple[ExtraTreesRegressor, dict[str, float]]:
    y = frame["target"].to_numpy(float)
    groups = frame["smiles"].map(canonical).to_numpy()
    n_splits = 3 if smoke or len(frame) < 500 else 5
    oof = np.zeros(len(frame), dtype=float)
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, valid_idx in cv.split(features, y, groups):
        model = build_model(n_estimators, n_jobs)
        model.fit(features[train_idx], y[train_idx])
        oof[valid_idx] = model.predict(features[valid_idx])
    final_model = build_model(n_estimators, n_jobs).fit(features, y)
    return final_model, {
        "n": float(len(frame)),
        "folds": float(n_splits),
        "r2": float(r2_score(y, oof)),
        "mae": float(mean_absolute_error(y, oof)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    models: dict[str, ExtraTreesRegressor] = {}
    records: list[dict[str, float | str]] = []
    test_prediction = np.empty(len(test), dtype=float)

    for target in TARGETS:
        frame = train[train["target_type"] == target].reset_index(drop=True)
        if args.smoke and len(frame) > 300:
            frame = frame.sample(300, random_state=SEED).reset_index(drop=True)
        x_train = featurize(frame["smiles"].tolist())
        n_estimators = min(args.n_estimators, 100) if args.smoke else args.n_estimators
        model, metrics = evaluate_target(frame, x_train, args.smoke, n_estimators, args.n_jobs)
        models[target] = model
        records.append({"target": target, **metrics})
        mask = test["target_type"].eq(target).to_numpy()
        test_prediction[mask] = model.predict(featurize(test.loc[mask, "smiles"].tolist()))
        print(f"{target}: grouped-CV R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}", flush=True)

    metrics = pd.DataFrame(records)
    metrics.loc[len(metrics)] = {"target": "mean", "n": metrics["n"].sum(), "folds": np.nan,
                                 "r2": metrics["r2"].mean(), "mae": np.nan}
    metrics.to_csv(args.output_dir / "grouped_cv_metrics.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": test_prediction}).to_csv(args.output_dir / "submission_pure_ml.csv", index=False)
    joblib.dump({"models": models, "descriptor_names": DESCRIPTOR_NAMES, "fingerprint_bits": 1024,
                 "seed": SEED, "metric": "mean target-wise grouped-CV R2"}, args.output_dir / "pure_ml_models.joblib")
    print(f"mean target-wise grouped-CV R2={metrics.iloc[-1]['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
