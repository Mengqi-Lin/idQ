#!/usr/bin/env python
"""
Row-sparsity/applied diagnostic setting Q-matrix identifiability simulation.

This is the simulation corresponding to the reviewer-facing design:
    ||q_j||_0 <= m, usually with m = 3 or m = 4.

Default sampling scheme:
    D_j = ||q_j||_0 ~ Uniform{1, ..., m}, independently over rows;
    conditional on D_j, sample D_j attributes uniformly without replacement.

Usage:
    python row_sparsity_expr.py <J> <K> <N> <m> <seed> <solver>

Examples:
    python row_sparsity_expr.py 50 10 100 3 1 -1
    python row_sparsity_expr.py 50 10 100 4 1 -1

To exclude pure nodes by construction, use:
    python row_sparsity_expr.py 50 10 100 4 1 -1 --min-row-size 2

By default this writes one seed-specific CSV to:
    ../data/raw/rowsparse_solver{solver}_J{J}_K{K}_m{m}_dmin{min_row_size}_seed{seed}_diag.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence

from idq_experiment_helpers import (
    parse_probability_list,
    run_design_expr,
    sample_row_sparse_q,
)


def default_output_path(
    J: int,
    K: int,
    m: int,
    min_row_size: int,
    row_size_distribution: str,
    seed: int,
    solver: int,
) -> str:
    """Default per-seed output path for row-sparsity experiments."""
    dist_tag = row_size_distribution.replace(" ", "")
    return (
        f"../data/raw/rowsparse_{dist_tag}_solver{solver}"
        f"_J{J}_K{K}_m{m}_dmin{min_row_size}_seed{seed}_diag.csv"
    )


def run_expr(
    J: int,
    K: int,
    N: int,
    m: int,
    seed: int,
    solver: int = -1,
    output_csv: Optional[str] = None,
    op: str = "conj",
    min_row_size: int = 1,
    row_size_distribution: str = "uniform",
    row_size_probs: Optional[Sequence[float]] = None,
    append: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run the row-sparsity simulation.

    Each row has support size at most m.  By default, D_j is uniform on
    {1, ..., m}; use min_row_size=2 to exclude pure nodes by construction.
    """
    if output_csv is None:
        output_csv = default_output_path(
            J=J,
            K=K,
            m=m,
            min_row_size=min_row_size,
            row_size_distribution=row_size_distribution,
            seed=seed,
            solver=solver,
        )

    m_eff = min(int(m), int(K))
    support = list(range(int(min_row_size), m_eff + 1))

    metadata = {
        "design": "row_sparsity",
        "p": "",
        "m": int(m),
        "min_row_size": int(min_row_size),
        "row_size_distribution": row_size_distribution,
        "row_size_support": support,
        "row_size_probs": "" if row_size_probs is None else row_size_probs,
    }

    def sampler(rng):
        return sample_row_sparse_q(
            J=J,
            K=K,
            m=m,
            rng=rng,
            min_row_size=min_row_size,
            row_size_distribution=row_size_distribution,
            row_size_probs=row_size_probs,
        )

    return run_design_expr(
        J=J,
        K=K,
        N=N,
        seed=seed,
        solver=solver,
        sampler=sampler,
        metadata=metadata,
        output_csv=output_csv,
        op=op,
        append=append,
        verbose=verbose,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run row-sparsity Q-matrix identifiability simulations."
    )
    parser.add_argument("J", type=int, help="Number of rows/items.")
    parser.add_argument("K", type=int, help="Number of latent attributes.")
    parser.add_argument("N", type=int, help="Number of Monte Carlo replicates.")
    parser.add_argument("m", type=int, help="Maximum row support size ||q_j||_0 <= m.")
    parser.add_argument("seed", type=int, help="Random seed.")
    parser.add_argument("solver", type=int, help="Solver code: -1, 0, 1, or 2.")
    parser.add_argument(
        "--min-row-size",
        type=int,
        default=1,
        help="Minimum row support size. Default: 1. Use 2 to exclude pure nodes by construction.",
    )
    parser.add_argument(
        "--row-size-distribution",
        default="uniform",
        choices=["uniform", "fixed", "custom"],
        help=(
            "Distribution for D_j. uniform: D_j uniform on support; "
            "fixed: D_j=m; custom: use --row-size-probs. Default: uniform."
        ),
    )
    parser.add_argument(
        "--row-size-probs",
        default=None,
        help=(
            "Comma-separated probabilities for custom row sizes over "
            "{min_row_size, ..., min(m,K)}. Example for sizes 1,2,3: 0.4,0.4,0.2."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Default is ../data/raw/rowsparse..._seed..._diag.csv.",
    )
    parser.add_argument(
        "--op",
        default="conj",
        choices=["conj", "disj"],
        help="Boolean operator used to compute Phi diagnostics. Default: conj.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output CSV instead of overwriting it. Not recommended for parallel jobs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print branch-level messages during each replicate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    row_size_probs = parse_probability_list(args.row_size_probs)

    RR = run_expr(
        J=args.J,
        K=args.K,
        N=args.N,
        m=args.m,
        seed=args.seed,
        solver=args.solver,
        output_csv=args.output_csv,
        op=args.op,
        min_row_size=args.min_row_size,
        row_size_distribution=args.row_size_distribution,
        row_size_probs=row_size_probs,
        append=args.append,
        verbose=args.verbose,
    )
    print(f"Wrote {len(RR)} row-sparsity simulation rows.")
