"""Materialize a clean, current-only F03 candidate for post-hoc diagnostics."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fable_common as fc
import F03_polymer_genome as f03


def main() -> None:
    data = fc.load_data()
    run_id = time.strftime("R2-F03-CLEAN-%Y%m%d-%H%M")
    mode = "with_archive" if os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1" else "without_archive"
    root = os.environ.get("FABLE_OUTPUT_ROOT", os.path.join(fc.ROUND2_DIR, "experiments", "CLEAN_OFFICIAL_ONLY"))
    out_dir = os.path.join(root, f"{run_id}-{mode}")
    os.makedirs(out_dir, exist_ok=False)

    train_cans = data.train["can"].tolist()
    test_cans = data.test["can"].tolist()
    all_cans = list(dict.fromkeys(train_cans + test_cans))
    x, feature_report = f03.hierarchical_features(list(dict.fromkeys(train_cans)), all_cans)
    index = {c: i for i, c in enumerate(all_cans)}
    predictions = []
    match_rows = 0

    for target in fc.TARGETS:
        sub = data.train[data.train["target_type"].eq(target)]
        cans = sub["can"].tolist()
        y = sub["target"].to_numpy(float)
        train_idx = np.asarray([index[c] for c in cans])
        test_sub = data.test[data.test["target_type"].eq(target)]
        target_test_cans = test_sub["can"].tolist()
        test_idx = np.asarray([index[c] for c in target_test_cans])
        support = np.isfinite(x[train_idx]).sum(axis=0)
        spread = np.nanstd(x[train_idx], axis=0)
        keep = (support > 0) & (spread > 1.0e-12)
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=f03.ALPHAS[0], solver="lsqr", max_iter=5000, tol=1.0e-4),
        )
        model.fit(np.clip(x[train_idx][:, keep], -1.0e6, 1.0e6), y)
        pred = model.predict(np.clip(x[test_idx][:, keep], -1.0e6, 1.0e6))

        lookup = sub.groupby("can")["target"].mean().to_dict()
        for row_id, can, value in zip(test_sub["id"], target_test_cans, pred):
            if can in lookup:
                value = lookup[can]
                match_rows += 1
            predictions.append((int(row_id), float(value), target))

    candidate = pd.DataFrame(predictions, columns=["id", "target", "target_type"])
    candidate = candidate.sort_values("id").reset_index(drop=True)
    if len(candidate) != len(data.test):
        raise RuntimeError("candidate row count mismatch")
    output = candidate[["id", "target"]]
    csv_path = os.path.join(out_dir, "candidate.csv")
    output.to_csv(csv_path, index=False)
    pd.DataFrame({"id": data.test["id"], "target_type": data.test["target_type"]}).to_csv(
        os.path.join(out_dir, "candidate_target_types.csv"), index=False
    )
    print({"csv": csv_path, "rows": len(output), "exact_match_rows": match_rows,
           "feature_report": feature_report})


if __name__ == "__main__":
    main()
