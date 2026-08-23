#!/usr/bin/env python
"""
Bernoulli-iid Q-matrix identifiability simulation.

Usage, matching the old positional interface:
    python idQ_expr.py <J> <K> <N> <p> <seed> <solver>

Example:
    python idQ_expr.py 50 10 100 0.7 1 -1

By default this writes one seed-specific CSV to:
    ../data/raw/solver{solver}_J{J}_K{K}_p{p}_seed{seed}_diag.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from idq_experiment_helpers import run_design_expr, sample_bernoulli_q


def default_output_path(J: int, K: int, p: float, seed: int, solver: int) -> str:
    """Default per-seed output path for Bernoulli experiments."""
    return f"../data/raw/solver{solver}_J{J}_K{K}_p{p}_seed{seed}_diag.csv"


def run_expr(
    J: int,
    K: int,
    N: int,
    p: float,
    seed: int,
    solver: int = -1,
    output_csv: Optional[str] = None,
    op: str = "conj",
    append: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Randomly sample N binary matrices with iid Bernoulli(p) entries.

    This keeps the same public function name as your current file, but delegates
    all shared identifiability and diagnostic logic to idq_experiment_helpers.py.
    """
    if output_csv is None:
        output_csv = default_output_path(J=J, K=K, p=p, seed=seed, solver=solver)

    metadata = {
        "design": "bernoulli",
        "p": float(p),
        "m": "",
        "min_row_size": "",
        "row_size_distribution": "",
        "row_size_support": "",
        "row_size_probs": "",
    }

    def sampler(rng):
        return sample_bernoulli_q(J=J, K=K, p=p, rng=rng)

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
        description="Run Bernoulli-iid Q-matrix identifiability simulations."
    )
    parser.add_argument("J", type=int, help="Number of rows/items.")
    parser.add_argument("K", type=int, help="Number of latent attributes.")
    parser.add_argument("N", type=int, help="Number of Monte Carlo replicates.")
    parser.add_argument("p", type=float, help="Bernoulli success probability.")
    parser.add_argument("seed", type=int, help="Random seed.")
    parser.add_argument("solver", type=int, help="Solver code: -1, 0, 1, or 2.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Default is ../data/raw/solver..._seed..._diag.csv.",
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
    RR = run_expr(
        J=args.J,
        K=args.K,
        N=args.N,
        p=args.p,
        seed=args.seed,
        solver=args.solver,
        output_csv=args.output_csv,
        op=args.op,
        append=args.append,
        verbose=args.verbose,
    )
    print(f"Wrote {len(RR)} Bernoulli simulation rows.")
