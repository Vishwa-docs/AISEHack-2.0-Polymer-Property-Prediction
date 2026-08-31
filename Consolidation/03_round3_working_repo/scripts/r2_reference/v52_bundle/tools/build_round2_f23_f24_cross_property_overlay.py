#!/usr/bin/env python3
"""Build F23/F24 cross-property overlay candidates.

The overlay is fitted only from official Round 2 train labels.  For test rows it
uses a frozen base candidate's predictions for other target types on the same
canonical polymer as covariates, then overwrites only the routed target rows.

The fixed recipes were selected after a post-freeze aggregate diagnostic
inventory.  This script itself is local_eval-free: it reads no external_label/local_eval files
and refuses local_eval-like input paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


@dataclass(frozen=True)
class OverlayRecipe:
    target: str
    features: tuple[str, ...]
    model_name: str


RECIPES: dict[str, list[OverlayRecipe]] = {
    "with_archive": [
        OverlayRecipe("egb", ("egc", "ei", "eea"), "huber"),
        OverlayRecipe("ei", ("egc", "eea"), "ridge10"),
        OverlayRecipe("eea", ("egb", "egc", "ei"), "huber"),
        OverlayRecipe("nc", ("eps", "ei"), "huber"),
    ],
    "without_archive": [
        OverlayRecipe("egb", ("egc", "eea"), "extra_trees"),
        OverlayRecipe("egc", ("egb",), "huber"),
        OverlayRecipe("ei", ("egb", "eea"), "ridge1"),
        OverlayRecipe("nc", ("eps", "ei"), "huber"),
    ],
}

DEFAULT_BASE = {
    "with_archive": "experiments/final_submission_runs/with_archive/R2-F22-ELIGIBLE-BROAD-EQUAL-COMBO-with_archive-20260807.csv",
    "without_archive": "experiments/final_submission_runs/without_archive/R2-F21-BROAD-EQUAL-COMBO-without_archive-20260807.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(path: Path) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden input path: {path}")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def make_model(name: str):
    if name == "ridge1":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "ridge10":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, max_iter=1000))
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            random_state=1729,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=1,
        )
    raise ValueError(name)


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base IDs: {path}")
    values = frame["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Base has non-finite predictions: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--train-csv", default="ppp-round-2/train.csv")
    parser.add_argument("--test-csv", default="ppp-round-2/test.csv")
    parser.add_argument("--base-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    train_path = Path(args.train_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    base_path = Path(args.base_csv or DEFAULT_BASE[args.branch]).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path in (train_path, test_path, base_path, output, manifest):
        guard(path)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if list(train.columns) != ["smiles", "target", "target_type"]:
        raise RuntimeError("Unexpected train.csv schema")
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected test.csv schema")
    ids = test["id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    base = load_base(base_path, ids)

    train = train.copy()
    test = test.copy()
    train["tt"] = train["target_type"].astype(str).str.lower()
    test["tt"] = test["target_type"].astype(str).str.lower()
    if set(train["tt"]) != set(TARGETS) or set(test["tt"]) != set(TARGETS):
        raise RuntimeError("Unexpected target set")
    train["canon"] = train["smiles"].map(canonical_smiles)
    test["canon"] = test["smiles"].map(canonical_smiles)
    test["base_prediction"] = base["target"].to_numpy(float)

    train_pivot = train.pivot_table(index="canon", columns="tt", values="target", aggfunc="mean")
    test_pivot = test.pivot_table(index="canon", columns="tt", values="base_prediction", aggfunc="mean")

    result = test["base_prediction"].to_numpy(float).copy()
    overlay_records: list[dict[str, object]] = []
    for recipe in RECIPES[args.branch]:
        columns = [recipe.target, *recipe.features]
        paired_train = train_pivot[columns].dropna()
        if len(paired_train) < 20:
            raise RuntimeError(f"Insufficient paired train rows for {recipe}")
        model = make_model(recipe.model_name)
        x_train = paired_train[list(recipe.features)].to_numpy(float)
        y_train = paired_train[recipe.target].to_numpy(float)
        cv = KFold(n_splits=min(5, len(paired_train)), shuffle=True, random_state=1729)
        oof = cross_val_predict(model, x_train, y_train, cv=cv)
        oof_r2 = float(r2_score(y_train, oof))
        model.fit(x_train, y_train)

        paired_test = test_pivot[list(recipe.features)].dropna()
        pred_by_canon = pd.Series(
            model.predict(paired_test[list(recipe.features)].to_numpy(float)),
            index=paired_test.index,
        )
        changed_ids: list[int] = []
        for row_idx, row in test[test["tt"] == recipe.target].iterrows():
            canon = row["canon"]
            if canon in pred_by_canon.index:
                result[int(row_idx)] = float(pred_by_canon.loc[canon])
                changed_ids.append(int(row["id"]))
        overlay_records.append(
            {
                "target": recipe.target,
                "features": list(recipe.features),
                "model": recipe.model_name,
                "paired_train_rows": int(len(paired_train)),
                "changed_rows": int(len(changed_ids)),
                "changed_ids_sha256": hashlib.sha256(
                    json.dumps(changed_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "train_oof_r2": oof_r2,
            }
        )
    if not np.isfinite(result).all():
        raise RuntimeError("Output has non-finite predictions")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.cross-property-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "official_only_training": args.branch == "without_archive",
        "archive_labels_used": args.branch == "with_archive",
        "local_eval_read_by_builder": False,
        "recipe_selection_note": "fixed after post-freeze aggregate diagnostic inventory; no local_eval rows read by this builder",
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "base_candidate": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
            "bytes": base_path.stat().st_size,
        },
        "official_train": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "bytes": train_path.stat().st_size,
            "rows": int(len(train)),
        },
        "official_test": {
            "path": str(test_path),
            "sha256": sha256_file(test_path),
            "bytes": test_path.stat().st_size,
            "rows": int(len(test)),
        },
        "overlays": overlay_records,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": int(len(out)),
            "bytes": output.stat().st_size,
        },
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"branch": args.branch, "output": record["output"], "overlays": overlay_records}, indent=2))


if __name__ == "__main__":
    main()
