#!/usr/bin/env python3
"""Validate and summarize schema-v4 row-sparsity simulation outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core import BRANCH_LABELS
from ..sat import normalize_cardinality_encoding
from .common import SCHEMA_VERSION


from ..paths import data_directory

REQUIRED_COLUMNS = {
    "schema_version",
    "design",
    "design_id",
    "operator",
    "solver_name",
    "cardinality_encoding_requested",
    "maximal_candidate_requested",
    "rng_engine",
    "J",
    "K",
    "N",
    "seed",
    "sim",
    "m_requested",
    "m_effective",
    "min_row_size_requested",
    "row_size_distribution",
    "row_size_support",
    "row_size_probs",
    "identifiable",
    "branch",
    "branch_label",
    "sat_called",
    "identity_original",
    "not_complete_original",
    "has_pure_node_original",
    "no_pure_nodes_original",
    "has_zero_row_original",
    "two_col_violation_original",
    "identity_basis",
    "two_col_violation_basis",
    "J_basis",
    "M_basis",
    "basis_time",
    "identity_check_time",
    "two_col_check_time",
    "three_col_check_time",
    "sat_time",
    "preprocess_time",
    "algorithm_time",
}

NUMERIC_COLUMNS = {
    "maximal_candidate_requested",
    "schema_version",
    "J",
    "K",
    "N",
    "seed",
    "sim",
    "m_requested",
    "m_effective",
    "min_row_size_requested",
    "identifiable",
    "branch",
    "sat_called",
    "identity_original",
    "not_complete_original",
    "has_pure_node_original",
    "no_pure_nodes_original",
    "has_zero_row_original",
    "two_col_violation_original",
    "identity_basis",
    "two_col_violation_basis",
    "J_basis",
    "M_basis",
    "basis_time",
    "identity_check_time",
    "two_col_check_time",
    "three_col_check_time",
    "sat_time",
    "preprocess_time",
    "algorithm_time",
    "sat_variables",
    "sat_clauses",
}

BINARY_COLUMNS = {
    "maximal_candidate_requested",
    "identifiable",
    "sat_called",
    "identity_original",
    "not_complete_original",
    "has_pure_node_original",
    "no_pure_nodes_original",
    "has_zero_row_original",
    "two_col_violation_original",
    "identity_basis",
    "two_col_violation_basis",
}


def _validate_one_file(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"Empty simulation file: {path}")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    data = frame.copy()
    for column in sorted(NUMERIC_COLUMNS & set(data.columns)):
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except Exception as exc:
            raise ValueError(f"{path}: column {column!r} is not numeric.") from exc
        if data[column].isna().any() or not np.isfinite(data[column]).all():
            raise ValueError(f"{path}: column {column!r} contains missing/nonfinite values.")

    if set(data["schema_version"].astype(int)) != {SCHEMA_VERSION}:
        raise ValueError(f"{path}: expected schema_version={SCHEMA_VERSION}.")
    if set(data["design"].astype(str)) != {"row_sparsity"}:
        raise ValueError(f"{path}: contains a non-row-sparsity design.")
    if set(data["operator"].astype(str)) != {"conj"}:
        raise ValueError(f"{path}: only conjunctive outputs are supported.")
    for encoding in data["cardinality_encoding_requested"].unique():
        normalize_cardinality_encoding(encoding)
    if not set(data["maximal_candidate_requested"]) <= {0, 1}:
        raise ValueError(f"{path}: maximal_candidate_requested is not binary.")

    for column in BINARY_COLUMNS:
        values = set(data[column].astype(int))
        if not values <= {0, 1}:
            raise ValueError(f"{path}: column {column!r} is not binary.")

    if not np.array_equal(
        data["not_complete_original"].astype(int).to_numpy(),
        1 - data["identity_original"].astype(int).to_numpy(),
    ):
        raise ValueError(f"{path}: completeness diagnostics are inconsistent.")
    if not np.array_equal(
        data["no_pure_nodes_original"].astype(int).to_numpy(),
        1 - data["has_pure_node_original"].astype(int).to_numpy(),
    ):
        raise ValueError(f"{path}: pure-node diagnostics are inconsistent.")

    expected_identifiable = data["branch"].isin([4, 6]).astype(int)
    if not np.array_equal(expected_identifiable, data["identifiable"].astype(int)):
        raise ValueError(f"{path}: branch and identifiable status disagree.")
    expected_sat_called = data["branch"].isin([5, 6]).astype(int)
    if not np.array_equal(expected_sat_called, data["sat_called"].astype(int)):
        raise ValueError(f"{path}: branch and sat_called disagree.")
    expected_labels = data["branch"].map(BRANCH_LABELS)
    if expected_labels.isna().any() or not np.array_equal(
        expected_labels.astype(str),
        data["branch_label"].astype(str),
    ):
        raise ValueError(f"{path}: branch labels are inconsistent.")

    # Each output is one complete seed-specific job.
    singleton_columns = [
        "design_id",
        "solver_name",
        "cardinality_encoding_requested",
        "maximal_candidate_requested",
        "operator",
        "rng_engine",
        "J",
        "K",
        "N",
        "seed",
    ]
    for column in singleton_columns:
        if data[column].nunique(dropna=False) != 1:
            raise ValueError(f"{path}: {column!r} varies within one job file.")
    expected_n = int(data["N"].iloc[0])
    if len(data) != expected_n or set(data["sim"].astype(int)) != set(range(expected_n)):
        raise ValueError(f"{path}: incomplete or duplicated simulation indices.")

    data["source_file"] = path.name
    return data


def read_row_sparsity_csvs(
    data_dir: Path,
    pattern: str = "rowsparse*_diag.csv",
) -> pd.DataFrame:
    files = sorted(path for path in data_dir.glob(pattern) if not path.name.endswith(".part"))
    if not files:
        raise FileNotFoundError(f"No completed files matched {data_dir / pattern}")
    frames = [_validate_one_file(pd.read_csv(path), path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _deduplicate(
    data: pd.DataFrame,
    *,
    drop_identical_duplicates: bool,
) -> pd.DataFrame:
    key = [
        "design_id",
        "solver_name",
        "cardinality_encoding_requested",
        "maximal_candidate_requested",
        "operator",
        "rng_engine",
        "J",
        "K",
        "seed",
        "sim",
    ]
    duplicate_mask = data.duplicated(key, keep=False)
    if not duplicate_mask.any():
        return data
    if not drop_identical_duplicates:
        examples = data.loc[duplicate_mask, key + ["source_file"]].head(8)
        raise ValueError(
            "Duplicate simulation keys were found. Use "
            "drop_identical_duplicates=True only after checking them.\n"
            + examples.to_string(index=False)
        )

    comparison_columns = [column for column in data.columns if column != "source_file"]
    for _, group in data.loc[duplicate_mask].groupby(key, dropna=False):
        normalized = group[comparison_columns].astype(str).drop_duplicates()
        if len(normalized) != 1:
            raise ValueError("Conflicting duplicate simulation rows were found.")
    return data.drop_duplicates(key, keep="first").copy()


def summarize(
    data: pd.DataFrame,
    *,
    drop_identical_duplicates: bool = False,
) -> pd.DataFrame:
    df = _deduplicate(
        data,
        drop_identical_duplicates=drop_identical_duplicates,
    )
    group_columns = [
        "design_id",
        "solver_name",
        "cardinality_encoding_requested",
        "maximal_candidate_requested",
        "operator",
        "rng_engine",
        "J",
        "K",
        "m_requested",
        "m_effective",
        "min_row_size_requested",
        "row_size_distribution",
        "row_size_support",
        "row_size_probs",
    ]

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        identifiable = group["identifiable"].astype(bool)
        not_complete = group["not_complete_original"].astype(bool)
        no_pure = group["no_pure_nodes_original"].astype(bool)
        sat_called = group["sat_called"].astype(bool)
        n_identifiable = int(identifiable.sum())

        row.update({
            "n": len(group),
            "n_seeds": int(group["seed"].nunique()),
            "id_prop": float(identifiable.mean()),
            "identity_original_prop": float(group["identity_original"].mean()),
            "not_complete_prop": float(not_complete.mean()),
            "no_pure_prop": float(no_pure.mean()),
            "id_and_not_complete_prop": float((identifiable & not_complete).mean()),
            "not_complete_given_id": (
                float((identifiable & not_complete).sum() / n_identifiable)
                if n_identifiable else np.nan
            ),
            "id_and_no_pure_prop": float((identifiable & no_pure).mean()),
            "no_pure_given_id": (
                float((identifiable & no_pure).sum() / n_identifiable)
                if n_identifiable else np.nan
            ),
            "zero_row_prop": float(group["has_zero_row_original"].mean()),
            "two_col_violation_original_prop": float(
                group["two_col_violation_original"].mean()
            ),
            "two_col_violation_basis_prop": float(
                group["two_col_violation_basis"].mean()
            ),
            "identity_basis_prop": float(group["identity_basis"].mean()),
            "sat_called_prop": float(sat_called.mean()),
            "mean_J_basis": float(group["J_basis"].mean()),
            "mean_M_basis": float(group["M_basis"].mean()),
            "mean_algorithm_time": float(group["algorithm_time"].mean()),
            "median_algorithm_time": float(group["algorithm_time"].median()),
            "mean_preprocess_time": float(group["preprocess_time"].mean()),
            "mean_sat_time_all": float(group["sat_time"].mean()),
            "mean_sat_time_given_SAT": (
                float(group.loc[sat_called, "sat_time"].mean())
                if sat_called.any() else np.nan
            ),
        })
        for branch, label in BRANCH_LABELS.items():
            row[f"branch_{branch}_{label}_prop"] = float(
                (group["branch"] == branch).mean()
            )
        rows.append(row)

    output = pd.DataFrame(rows)
    return output.sort_values(
        ["K", "J", "m_effective", "row_size_distribution", "solver_name",
         "cardinality_encoding_requested", "maximal_candidate_requested"]
    ).reset_index(drop=True)


def add_percentage_display(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    for column in output.columns:
        if column.endswith("_prop") or column.endswith("_given_id"):
            output[column] = output[column].map(
                lambda value: "" if pd.isna(value) else f"{100 * value:.1f}"
            )
    return output


def _write_csv_atomic(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete output already exists: {temporary}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_directory() / "raw" / "row_sparsity")
    parser.add_argument("--pattern", default="**/rowsparse*_diag.csv")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=data_directory() / "processed" / "row_sparsity_summary.csv",
    )
    parser.add_argument(
        "--display-csv",
        type=Path,
        default=data_directory() / "processed" / "row_sparsity_summary_display.csv",
    )
    parser.add_argument("--drop-identical-duplicates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = read_row_sparsity_csvs(args.data_dir, args.pattern)
    numeric = summarize(
        raw,
        drop_identical_duplicates=args.drop_identical_duplicates,
    )
    display = add_percentage_display(numeric)
    _write_csv_atomic(numeric, args.output_csv, overwrite=args.overwrite)
    _write_csv_atomic(display, args.display_csv, overwrite=args.overwrite)
    print(f"Validated and summarized {len(raw)} simulation rows.")


if __name__ == "__main__":
    main()
