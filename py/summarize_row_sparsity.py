#!/usr/bin/env python
"""
Summarize row-sparsity simulation CSVs.

Example:
    python summarize_row_sparsity.py \
        --data-dir ../data/raw \
        --output-csv ../data/processed/row_sparsity_summary.csv

This script expects CSVs produced by row_sparsity_expr.py.  It computes the
main reviewer-facing quantities:
  - identifiable proportion;
  - identity-submatrix/completeness proportion;
  - identifiable but not complete proportion;
  - identifiable with no pure nodes proportion;
  - conditional versions among identifiable matrices;
  - basis size, M, SAT-pass, and branch diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


BRANCH_LABELS = {
    -1: "basis_J_less_than_K",
     0: "all_zero_column",
     1: "all_one_column",
     2: "two_column_check_failed",
     3: "three_column_check_failed",
     4: "identity_submatrix_direct_id",
     5: "SAT_found_counterexample",
     6: "SAT_unsat_identifiable",
}


NUMERIC_COLS = [
    "J", "K", "N", "seed", "sim", "solver", "p", "m", "min_row_size",
    "J_basis", "M_basis", "identifiable", "branch", "sat_called",
    "identity_original", "not_complete_original",
    "has_pure_node_original", "no_pure_nodes_original",
    "two_col_violation_original", "identity_basis", "two_col_violation_basis",
    "basis_time", "trivial_check_time", "two_col_check_time",
    "three_col_check_time", "identity_check_time", "sat_time",
    "preprocess_time", "algorithm_time", "diagnostic_time",
]


def read_row_sparsity_csvs(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {data_dir / pattern}")

    frames = []
    for f in files:
        tmp = pd.read_csv(f)
        tmp["source_file"] = f.name
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True)

    # Defensive cleanup in case accidental headers were appended.
    df = df[df["J"].astype(str) != "J"].copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "design" in df.columns:
        df = df[df["design"].fillna("") == "row_sparsity"].copy()

    return df


def summarize(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    group_cols = [
        "solver", "op", "J", "K", "m", "min_row_size",
        "row_size_distribution", "row_size_support", "row_size_probs",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    if drop_duplicates:
        key = [c for c in ["solver", "op", "J", "K", "m", "min_row_size", "seed", "sim"] if c in df.columns]
        if key:
            df = df.drop_duplicates(key, keep="last").copy()

    rows: List[dict] = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))

        identifiable = g["identifiable"].astype(bool)
        not_complete = g["not_complete_original"].astype(bool)
        no_pure = g["no_pure_nodes_original"].astype(bool)
        sat_called = g["sat_called"].astype(bool)

        n = len(g)
        n_id = int(identifiable.sum())

        row.update({
            "n": n,
            "n_seeds": g["seed"].nunique() if "seed" in g.columns else np.nan,
            "id_prop": identifiable.mean(),
            "identity_original_prop": g["identity_original"].mean(),
            "not_complete_prop": not_complete.mean(),
            "no_pure_prop": no_pure.mean(),
            "id_and_not_complete_prop": (identifiable & not_complete).mean(),
            "not_complete_given_id": ((identifiable & not_complete).sum() / n_id) if n_id > 0 else np.nan,
            "id_and_no_pure_prop": (identifiable & no_pure).mean(),
            "no_pure_given_id": ((identifiable & no_pure).sum() / n_id) if n_id > 0 else np.nan,
            "two_col_violation_original_prop": g["two_col_violation_original"].mean(),
            "two_col_violation_basis_prop": g["two_col_violation_basis"].mean(),
            "identity_basis_prop": g["identity_basis"].mean(),
            "sat_called_prop": sat_called.mean(),
            "mean_J_basis": g["J_basis"].mean(),
            "mean_M_basis": g["M_basis"].mean(),
            "mean_algorithm_time": g["algorithm_time"].mean(),
            "median_algorithm_time": g["algorithm_time"].median(),
            "mean_preprocess_time": g["preprocess_time"].mean(),
            "mean_sat_time_all": g["sat_time"].mean(),
            "mean_sat_time_given_SAT": g.loc[sat_called, "sat_time"].mean() if sat_called.any() else np.nan,
        })

        for b, label in BRANCH_LABELS.items():
            row[f"branch_{b}_{label}_prop"] = (g["branch"] == b).mean()

        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["K", "J", "m", "min_row_size", "row_size_distribution", "solver"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def add_percentage_display(summary: pd.DataFrame) -> pd.DataFrame:
    percent_cols = [
        "id_prop", "identity_original_prop", "not_complete_prop", "no_pure_prop",
        "id_and_not_complete_prop", "not_complete_given_id",
        "id_and_no_pure_prop", "no_pure_given_id",
        "two_col_violation_original_prop", "two_col_violation_basis_prop",
        "identity_basis_prop", "sat_called_prop",
    ]
    out = summary.copy()
    for col in percent_cols:
        if col in out.columns:
            out[col] = (100 * out[col]).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize row-sparsity diagnostics.")
    parser.add_argument("--data-dir", default="../data/raw", help="Directory containing raw CSV files.")
    parser.add_argument(
        "--pattern",
        default="rowsparse*_diag.csv",
        help="Glob pattern for row-sparsity CSV files. Default: rowsparse*_diag.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default="../data/processed/row_sparsity_summary.csv",
        help="Path for raw numeric summary CSV.",
    )
    parser.add_argument(
        "--display-csv",
        default="../data/processed/row_sparsity_summary_display.csv",
        help="Path for display summary CSV with percentages.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate (seed, sim) rows instead of dropping duplicates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)
    display_csv = Path(args.display_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    display_csv.parent.mkdir(parents=True, exist_ok=True)

    df = read_row_sparsity_csvs(data_dir=data_dir, pattern=args.pattern)
    summary = summarize(df, drop_duplicates=not args.keep_duplicates)
    display = add_percentage_display(summary)

    summary.to_csv(output_csv, index=False)
    display.to_csv(display_csv, index=False)

    print(f"Read {len(df)} raw rows from {data_dir / args.pattern}")
    print(f"Wrote numeric summary to {output_csv}")
    print(f"Wrote display summary to {display_csv}")
