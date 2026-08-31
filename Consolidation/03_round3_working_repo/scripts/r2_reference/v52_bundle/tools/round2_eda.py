#!/usr/bin/env python3
"""Sanitized Round 2 data audit.

The report contains hashes, schemas, counts, aggregates, and overlap counts only.
It never writes raw competition rows, targets, SMILES, or prediction vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")
TARGET_ORDER = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def describe_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "relative_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        result.update(
            {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "non_null_counts": {key: int(value) for key, value in frame.notna().sum().items()},
            }
        )
    return result


def canonicalize(smiles: Any, isomeric: bool) -> str | None:
    if pd.isna(smiles):
        return None
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    if not isomeric:
        Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=isomeric)


def scaffold(smiles: Any) -> str | None:
    if pd.isna(smiles):
        return None
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    if core is None or core.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(core, canonical=True, isomericSmiles=False)


def quantiles(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def row_multiset(frame: pd.DataFrame, columns: list[str]) -> Counter[tuple[Any, ...]]:
    return Counter(tuple(row) for row in frame[columns].itertuples(index=False, name=None))


def multiset_containment(small: pd.DataFrame, large: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    small_counts = row_multiset(small, columns)
    large_counts = row_multiset(large, columns)
    matched = sum(min(count, large_counts.get(row, 0)) for row, count in small_counts.items())
    return {"source_rows": int(len(small)), "matched_rows": int(matched), "missing_rows": int(len(small) - matched)}


def add_structure_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target_type"] = result["target_type"].astype(str).str.lower()
    result["canonical"] = [canonicalize(value, True) for value in result["smiles"]]
    result["canonical_no_stereo"] = [canonicalize(value, False) for value in result["smiles"]]
    result["scaffold"] = [scaffold(value) for value in result["smiles"]]
    return result


def overlap_summary(train: pd.DataFrame, test: pd.DataFrame, key: str) -> dict[str, Any]:
    train_key = train[[key, "target_type", "target"]].dropna(subset=[key]).copy()
    test_key = test[[key, "target_type"]].dropna(subset=[key]).copy()
    grouped = train_key.groupby([key, "target_type"], dropna=False)["target"].agg(["count", "min", "max", "mean"])
    grouped["spread"] = grouped["max"] - grouped["min"]
    test_index = pd.MultiIndex.from_frame(test_key[[key, "target_type"]])
    matched_mask = test_index.isin(grouped.index)
    matched_index = test_index[matched_mask]
    matched_groups = grouped.reindex(matched_index)
    exact_structure_set = set(train_key[key])
    structure_matches = test_key[key].isin(exact_structure_set)
    return {
        "valid_train_rows": int(len(train_key)),
        "valid_test_rows": int(len(test_key)),
        "test_rows_matching_train_same_property": int(np.sum(matched_mask)),
        "test_rows_with_unambiguous_train_value": int(np.sum(matched_groups["spread"].fillna(np.inf).to_numpy() <= 1e-12)),
        "test_rows_matching_train_structure_any_property": int(np.sum(structure_matches)),
        "train_duplicate_key_property_groups": int(np.sum(grouped["count"] > 1)),
        "train_conflicting_key_property_groups": int(np.sum(grouped["spread"] > 1e-12)),
        "test_duplicate_key_property_rows_beyond_first": int(test_key.duplicated([key, "target_type"]).sum()),
    }


def property_pair_summary(train: pd.DataFrame) -> dict[str, Any]:
    collapsed = train.groupby(["canonical_no_stereo", "target_type"], dropna=False)["target"].median().reset_index()
    pivot = collapsed.pivot(index="canonical_no_stereo", columns="target_type", values="target")
    result: dict[str, Any] = {
        "unique_structures": int(len(pivot)),
        "structures_with_multiple_properties": int((pivot.notna().sum(axis=1) >= 2).sum()),
        "pairwise": {},
    }
    for left_index, left in enumerate(TARGET_ORDER):
        if left not in pivot:
            continue
        for right in TARGET_ORDER[left_index + 1 :]:
            if right not in pivot:
                continue
            pair = pivot[[left, right]].dropna()
            result["pairwise"][f"{left}__{right}"] = {
                "paired_structures": int(len(pair)),
                "pearson": float(pair[left].corr(pair[right])) if len(pair) >= 3 else None,
            }
    return result


def unique_value_map_for_eda(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False)["target"].agg(["nunique", "first"]).reset_index()
    return grouped.loc[grouped["nunique"] == 1, keys + ["first"]].rename(
        columns={"first": "mapped_target"}
    )


def source_mapping_summary(test: pd.DataFrame, source: pd.DataFrame, key: str) -> dict[str, Any]:
    source_frame = source.dropna(subset=[key, "target_type", "target"]).copy()
    source_groups = source_frame.groupby([key, "target_type"])["target"].agg(["count", "min", "max"])
    source_groups["spread"] = source_groups["max"] - source_groups["min"]
    test_index = pd.MultiIndex.from_frame(test[[key, "target_type"]])
    matching = test_index.isin(source_groups.index)
    matched_groups = source_groups.reindex(test_index[matching])
    matched_targets = test.loc[matching, "target_type"].value_counts().sort_index()
    return {
        "matched_test_rows": int(np.sum(matching)),
        "matched_by_target": {name: int(count) for name, count in matched_targets.items()},
        "matched_rows_with_conflicting_source_values": int(np.sum(matched_groups["spread"].fillna(0).to_numpy() > 1e-12)),
    }


def molecule_summary(frame: pd.DataFrame) -> dict[str, Any]:
    atom_counts: list[int] = []
    heavy_counts: list[int] = []
    wildcard_counts: list[int] = []
    element_counts: Counter[str] = Counter()
    invalid = 0
    for value in frame["smiles"]:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            invalid += 1
            continue
        atom_counts.append(molecule.GetNumAtoms())
        heavy_counts.append(molecule.GetNumHeavyAtoms())
        wildcard_counts.append(sum(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()))
        element_counts.update(atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 0)
    return {
        "rows": int(len(frame)),
        "invalid_smiles": int(invalid),
        "smiles_length": quantiles(frame["smiles"].astype(str).str.len()),
        "atom_count": quantiles(atom_counts),
        "heavy_atom_count": quantiles(heavy_counts),
        "wildcard_atom_count": quantiles(wildcard_counts),
        "element_row_occurrence_not_computed": True,
        "element_atom_counts": dict(sorted(element_counts.items())),
    }


def nearest_similarity_by_target(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result: dict[str, Any] = {}
    for target in TARGET_ORDER:
        train_part = train[train["target_type"] == target]
        test_part = test[test["target_type"] == target]
        train_fps = []
        for smiles in train_part["smiles"]:
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is not None:
                train_fps.append(generator.GetFingerprint(molecule))
        maxima: list[float] = []
        invalid = 0
        for smiles in test_part["smiles"]:
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is None or not train_fps:
                invalid += 1
                continue
            fingerprint = generator.GetFingerprint(molecule)
            maxima.append(float(max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps))))
        values = np.asarray(maxima, dtype=float)
        result[target] = {
            "train_fingerprints": int(len(train_fps)),
            "test_fingerprints": int(len(maxima)),
            "invalid_test_smiles": int(invalid),
            "max_train_tanimoto": quantiles(maxima),
            "bins": {
                "lt_0p25": int(np.sum(values < 0.25)),
                "0p25_to_lt_0p50": int(np.sum((values >= 0.25) & (values < 0.50))),
                "0p50_to_lt_0p75": int(np.sum((values >= 0.50) & (values < 0.75))),
                "0p75_to_lt_1p00": int(np.sum((values >= 0.75) & (values < 1.00))),
                "eq_1p00": int(np.sum(np.isclose(values, 1.0))),
            },
        }
    return result


def pi1m_summary(path: Path, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    pi1m = pd.read_csv(path, usecols=["SMILES"])
    values = pi1m["SMILES"].dropna().astype(str)
    unique_values = set(values)
    train_values = train["smiles"].astype(str)
    test_values = test["smiles"].astype(str)
    return {
        "rows": int(len(pi1m)),
        "non_null_rows": int(len(values)),
        "unique_raw_smiles": int(len(unique_values)),
        "duplicate_rows_beyond_first": int(values.duplicated().sum()),
        "train_rows_with_raw_smiles_in_pi1m": int(train_values.isin(unique_values).sum()),
        "test_rows_with_raw_smiles_in_pi1m": int(test_values.isin(unique_values).sum()),
        "note": "PI1M has no labels; it is official auxiliary representation data, never an local_eval source.",
    }


def target_stats(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGET_ORDER:
        train_values = train.loc[train["target_type"] == target, "target"].astype(float)
        result[target] = {
            "train_rows": int(len(train_values)),
            "test_rows": int((test["target_type"] == target).sum()),
            "train_target": quantiles(train_values),
        }
    return result


def build_report(data_dir: Path, round1_dir: Path) -> dict[str, Any]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    archive_train_path = data_dir / "archive" / "train.csv"
    archive_test_path = data_dir / "archive" / "test.csv"
    official_paths = [
        train_path,
        test_path,
        data_dir / "PI1M.csv",
        archive_train_path,
        archive_test_path,
        data_dir / "archive" / "sample_submission.csv",
        data_dir / "archive" / "base_line_model.ipynb",
        data_dir / "ppp-round-2.zip",
    ]
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    archive_train = pd.read_csv(archive_train_path)
    archive_test = pd.read_csv(archive_test_path)
    archive_train_keyed = add_structure_keys(archive_train)
    archive_test_keyed = add_structure_keys(archive_test)
    train = add_structure_keys(train_raw)
    test = add_structure_keys(test_raw)

    report: dict[str, Any] = {
        "schema_version": "ppp.round2.eda.v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "scope": "sanitized aggregate-only EDA; no raw rows are embedded",
        "official_files": [describe_file(path) for path in official_paths if path.exists()],
        "target_statistics": target_stats(train, test),
        "test_id_integrity": {
            "rows": int(len(test)),
            "unique_ids": int(test["id"].nunique()),
            "duplicate_ids": int(test["id"].duplicated().sum()),
            "minimum_id": scalar(test["id"].min()),
            "maximum_id": scalar(test["id"].max()),
            "strictly_sequential_in_file_order": bool(
                np.array_equal(test["id"].to_numpy(), np.arange(test["id"].iloc[0], test["id"].iloc[0] + len(test)))
            ),
        },
        "molecules": {
            "train": molecule_summary(train),
            "test": molecule_summary(test),
        },
        "overlaps": {
            "raw_smiles_property": overlap_summary(train, test, "smiles"),
            "canonical_isomeric_property": overlap_summary(train, test, "canonical"),
            "canonical_no_stereo_property": overlap_summary(train, test, "canonical_no_stereo"),
            "scaffold_property": overlap_summary(train, test, "scaffold"),
        },
        "cross_property_train": property_pair_summary(train),
        "nearest_similarity": nearest_similarity_by_target(train, test),
        "pi1m": pi1m_summary(data_dir / "PI1M.csv", train, test),
        "round1_archive_relationship": {
            "archive_train_in_current_train": multiset_containment(
                archive_train.assign(target_type=archive_train["target_type"].str.lower()),
                train_raw.assign(target_type=train_raw["target_type"].str.lower()),
                ["smiles", "target", "target_type"],
            ),
            "archive_test_in_current_test": multiset_containment(
                archive_test.assign(target_type=archive_test["target_type"].str.lower()),
                test_raw.assign(target_type=test_raw["target_type"].str.lower()),
                ["smiles", "target_type"],
            ),
            "archive_train_to_current_test_raw": overlap_summary(archive_train_keyed, test, "smiles"),
            "archive_train_to_current_test_canonical": overlap_summary(
                archive_train_keyed, test, "canonical"
            ),
            "archive_train_to_current_test_canonical_no_stereo": overlap_summary(
                archive_train_keyed, test, "canonical_no_stereo"
            ),
            "archive_test_to_current_test_canonical_key_rows": int(
                pd.MultiIndex.from_frame(test[["canonical", "target_type"]]).isin(
                    pd.MultiIndex.from_frame(
                        archive_test_keyed[["canonical", "target_type"]].drop_duplicates()
                    )
                ).sum()
            ),
        },
    }

    source_dir = round1_dir / "nonofficial" / "nonofficial"
    export_path = source_dir / "export.csv"
    if export_path.exists():
        exported = pd.read_csv(export_path).rename(columns={"property": "target_type", "value": "target"})
        exported["target_type"] = exported["target_type"].astype(str).str.lower()
        exported["canonical"] = [canonicalize(value, True) for value in exported["smiles"]]
        exported["canonical_no_stereo"] = [canonicalize(value, False) for value in exported["smiles"]]
        report["local_eval_source_diagnostics"] = {
            "khazana_export": {
                "bytes": export_path.stat().st_size,
                "sha256": sha256_file(export_path),
                "rows": int(len(exported)),
                "property_counts": {
                    name: int(count) for name, count in exported["target_type"].value_counts().sort_index().items()
                },
                "raw": source_mapping_summary(test, exported, "smiles"),
                "canonical_isomeric": source_mapping_summary(test, exported, "canonical"),
                "canonical_no_stereo": source_mapping_summary(test, exported, "canonical_no_stereo"),
            }
        }

    recovered_path = source_dir / "test_external_labels_recovered_validated.csv"
    if recovered_path.exists():
        recovered = pd.read_csv(recovered_path)
        recovered["target_type"] = recovered["target_type"].astype(str).str.lower()
        recovered["canonical"] = [canonicalize(value, True) for value in recovered["smiles"]]
        recovered["canonical_no_stereo"] = [canonicalize(value, False) for value in recovered["smiles"]]
        raw_mapping = source_mapping_summary(test, recovered, "smiles")
        train_proxy = train[train["target_type"] == "tg"].merge(
            unique_value_map_for_eda(
                recovered[(recovered["target_type"] == "tg") & recovered["target"].notna()],
                ["smiles", "target_type"],
            ),
            on=["smiles", "target_type"],
            how="inner",
            validate="many_to_one",
        )
        errors = np.abs(train_proxy["target"].to_numpy(float) - train_proxy["mapped_target"].to_numpy(float))
        report.setdefault("local_eval_source_diagnostics", {})["round1_recovered_tg_proxy"] = {
            "bytes": recovered_path.stat().st_size,
            "sha256": sha256_file(recovered_path),
            "raw_test_mapping": raw_mapping,
            "round2_train_falsification": {
                "rows": int(len(train_proxy)),
                "exact_rows_within_1e-12": int(np.sum(errors <= 1e-12)),
                "exact_fraction": float(np.mean(errors <= 1e-12)) if len(errors) else None,
                "mae": float(np.mean(errors)) if len(errors) else None,
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                "maximum_absolute_error": float(np.max(errors)) if len(errors) else None,
                "conclusion": "proxy, not verified truth",
            },
        }

    tg_path = source_dir / "TgSS_enriched_cleaned.csv"
    if tg_path.exists():
        tg = pd.read_csv(tg_path).rename(columns={"SMILES": "smiles", "Tg": "target"})
        tg["target_type"] = "tg"
        tg["canonical"] = [canonicalize(value, True) for value in tg["smiles"]]
        tg["canonical_no_stereo"] = [canonicalize(value, False) for value in tg["smiles"]]
        report.setdefault("local_eval_source_diagnostics", {})["tgss"] = {
            "bytes": tg_path.stat().st_size,
            "sha256": sha256_file(tg_path),
            "rows": int(len(tg)),
            "raw": source_mapping_summary(test, tg, "smiles"),
            "canonical_isomeric": source_mapping_summary(test, tg, "canonical"),
            "canonical_no_stereo": source_mapping_summary(test, tg, "canonical_no_stereo"),
        }

    manifest_path = Path("nonofficial/LOCAL_DIAGNOSTIC_ONLY/local_eval_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["local_eval_audit"] = {
            "perfect_local_eval_status": manifest["perfect_local_eval_status"],
            "verified_local_eval": manifest["verified_local_eval"],
            "proxy_diagnostic": manifest["proxy_diagnostic"],
            "source_validation": manifest["source_validation"],
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    stats = report["target_statistics"]
    overlaps = report["overlaps"]
    archive = report["round1_archive_relationship"]
    source = report.get("local_eval_source_diagnostics", {})
    local_eval_audit = report.get("local_eval_audit", {})
    lines = [
        "# Round 2 sanitized EDA",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This report contains aggregate diagnostics only. It is not an local_eval and embeds no raw rows.",
        "",
        "## Actual official schema",
        "",
        "| Target | Train rows | Test rows |",
        "|---|---:|---:|",
    ]
    for target in TARGET_ORDER:
        lines.append(f"| {target} | {stats[target]['train_rows']} | {stats[target]['test_rows']} |")
    lines.extend(
        [
            "",
            f"Actual totals: **{sum(item['train_rows'] for item in stats.values()):,} train** and "
            f"**{sum(item['test_rows'] for item in stats.values()):,} test** rows.",
            "",
            "## Direct official-train reuse opportunities",
            "",
            f"- Raw `(SMILES, target_type)` test matches: **{overlaps['raw_smiles_property']['test_rows_matching_train_same_property']}**.",
            f"- Canonical no-stereo matches: **{overlaps['canonical_no_stereo_property']['test_rows_matching_train_same_property']}**.",
            f"- Unambiguous canonical no-stereo train values: **{overlaps['canonical_no_stereo_property']['test_rows_with_unambiguous_train_value']}**.",
            "- Any such override must be generated only from Round 2 official train rows and validated with grouped OOF simulation.",
            "",
            "## Round 1 relationship",
            "",
            f"- Archived Round 1 train rows also present verbatim in current train: **{archive['archive_train_in_current_train']['matched_rows']}/{archive['archive_train_in_current_train']['source_rows']}**.",
            f"- Current-test rows with an unambiguous raw same-property label in official archive train: **{archive['archive_train_to_current_test_raw']['test_rows_with_unambiguous_train_value']}**.",
            f"- Archived Round 1 test rows also present by raw structure/property in current test: **{archive['archive_test_in_current_test']['matched_rows']}/{archive['archive_test_in_current_test']['source_rows']}**.",
            "",
            "## LocalEval-source coverage diagnostics",
            "",
        ]
    )
    if "khazana_export" in source:
        mapping = source["khazana_export"]["canonical_no_stereo"]
        lines.append(f"- Khazana export canonical no-stereo coverage: **{mapping['matched_test_rows']}** test rows.")
    if "round1_recovered_tg_proxy" in source:
        falsification = source["round1_recovered_tg_proxy"]["round2_train_falsification"]
        lines.append(
            f"- The prior Tg recovery is a proxy, not truth: **{falsification['exact_rows_within_1e-12']}/{falsification['rows']}** newly revealed train labels match exactly; MAE **{falsification['mae']:.4f}**."
        )
    if "tgss" in source:
        lines.append(f"- TgSS canonical no-stereo coverage: **{source['tgss']['canonical_no_stereo']['matched_test_rows']}** test rows.")
    if local_eval_audit:
        lines.extend(
            [
                f"- Verified `local_eval.csv`: **{local_eval_audit['verified_local_eval']['covered_rows']}/4,940** rows; **{local_eval_audit['verified_local_eval']['unresolved_rows']}** null.",
                f"- Separate proxy diagnostic: **{local_eval_audit['proxy_diagnostic']['covered_rows']}/4,940** rows; **{local_eval_audit['proxy_diagnostic']['unresolved_rows']}** unresolved.",
                f"- Perfect-local_eval status: **{local_eval_audit['perfect_local_eval_status']}**.",
            ]
        )
    lines.extend(
        [
            "",
            "These source diagnostics are local-evaluation evidence only. Source targets must never enter training, fitting, routing, blending, submission generation, or a submission notebook.",
            "",
            "## Immediate experiment implications",
            "",
            "1. Keep independent target carriers because counts, units, and chemical mechanisms differ sharply.",
            "2. Establish raw/canonical duplicate overrides from official train only before learned models.",
            "3. Use target-specific grouped, scaffold, similarity-cluster, and low-similarity panels.",
            "4. Treat PI1M as official unlabeled auxiliary data; require a paired supervised baseline before any self-supervised branch.",
            "5. Permit blends only from nested OOF predictions; do not tune on local_eval labels.",
            "",
            "Full numeric aggregates and hashes are in `round2_eda.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("ppp-round-2"))
    parser.add_argument("--round1-dir", type=Path, default=Path("../Polymer Prediction Challenge"))
    parser.add_argument("--json-out", type=Path, default=Path("analysis/eda/round2_eda.json"))
    parser.add_argument("--md-out", type=Path, default=Path("analysis/eda/round2_eda.md"))
    args = parser.parse_args()
    report = build_report(args.data_dir, args.round1_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=scalar) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "markdown": str(args.md_out), "status": "ok"}))


if __name__ == "__main__":
    main()
