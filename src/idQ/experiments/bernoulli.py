#!/usr/bin/env python3
"""Bernoulli Q-matrix simulations for the conjunctive algorithm.

The original positional interface is still accepted::

    python -m idQ.experiments.bernoulli J K N p seed [solver]

Legacy integer solver codes remain valid, but ``-1`` now runs the exact
CaDiCaL formulation rather than the former restricted search.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import (
    rng_label,
    run_design_expr,
    sample_bernoulli_q,
)
from ..sat import NON_EQUIVALENCE_ENCODINGS, normalize_cardinality_encoding, normalize_solver_name


from ..paths import data_directory


def _safe_tag(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "")


def default_output_path(
    J: int,
    K: int,
    p: float,
    seed: int,
    solver: str | int,
    *,
    condition_nonzero_rows: bool,
    rng_engine: str,
    cardinality_encoding: str = "exclude_x",
    maximal_candidate: bool = False,
) -> Path:
    """Return a collision-resistant, project-relative per-seed output path."""
    zero_tag = "nonzero" if condition_nonzero_rows else "allowzero"
    return data_directory() / "raw" / "bernoulli" / (
        f"bernoulli_conj_solver-{_safe_tag(normalize_solver_name(solver))}"
        f"_enc-{normalize_cardinality_encoding(cardinality_encoding)}_max-{int(bool(maximal_candidate))}"
        f"_J{J}_K{K}_p{float(p):.12g}_{zero_tag}"
        f"_rng-{_safe_tag(rng_engine)}_seed{seed}_diag.csv"
    )


def run_expr(
    J: int,
    K: int,
    N: int,
    p: float,
    seed: int,
    solver: str | int = "glucose42",
    output_csv: str | None = None,
    *,
    condition_nonzero_rows: bool = True,
    rng_engine: str = "legacy",
    overwrite: bool = False,
    verbose: bool = False,
    cardinality_encoding: str = "exclude_x",
    maximal_candidate: bool = False,
    checkpoint_every: int = 10,
) -> list[dict[str, Any]]:
    """Run one Bernoulli simulation job."""
    if output_csv is None:
        output_csv = str(default_output_path(
            J,
            K,
            p,
            seed,
            solver,
            condition_nonzero_rows=condition_nonzero_rows,
            rng_engine=rng_engine,
            cardinality_encoding=cardinality_encoding,
            maximal_candidate=maximal_candidate,
        ))

    zero_policy = "condition_nonzero" if condition_nonzero_rows else "allow_iid"
    metadata = {
        "design": "bernoulli",
        "design_id": (
            f"bernoulli|conj|p={float(p):.12g}|zero={zero_policy}|"
            f"rng={rng_label(rng_engine)}"
        ),
        "p": float(p),
        "zero_row_policy": zero_policy,
    }

    def sampler(rng):
        return sample_bernoulli_q(
            J,
            K,
            p,
            rng,
            condition_nonzero_rows=condition_nonzero_rows,
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
        rng_engine=rng_engine,
        overwrite=overwrite,
        verbose=verbose,
        cardinality_encoding=cardinality_encoding,
        maximal_candidate=maximal_candidate,
        checkpoint_every=checkpoint_every,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bernoulli Q-matrix identifiability simulations."
    )
    parser.add_argument("J", type=int, help="Number of items/rows.")
    parser.add_argument("K", type=int, help="Number of attributes/columns.")
    parser.add_argument("N", type=int, help="Number of Monte Carlo replicates.")
    parser.add_argument("p", type=float, help="Bernoulli success probability.")
    parser.add_argument("seed", type=int, help="Random seed.")
    parser.add_argument(
        "solver",
        nargs="?",
        default="glucose42",
        help="PySAT solver name or legacy code (-1, 0, 1, 2). Default: glucose42.",
    )
    parser.add_argument("--output-csv", default=None, help="Explicit output path.")
    parser.add_argument(
        "--allow-zero-rows",
        action="store_true",
        help=(
            "Sample unrestricted iid Bernoulli entries, including zero rows; "
            "use this option for the paper's Bernoulli design."
        ),
    )
    parser.add_argument(
        "--rng-engine",
        choices=["legacy", "pcg64"],
        default="legacy",
        help="legacy preserves the original MT19937 random stream.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing completed output file.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--cardinality-encoding", choices=NON_EQUIVALENCE_ENCODINGS,
        default="exclude_x", help="Non-equivalence encoding (default: exclude_x).",
    )
    parser.add_argument(
        "--maximal-candidate", action="store_true",
        help="Enable the optional maximal-candidate restriction; off for the paper runs.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_expr(
        J=args.J,
        K=args.K,
        N=args.N,
        p=args.p,
        seed=args.seed,
        solver=args.solver,
        output_csv=args.output_csv,
        condition_nonzero_rows=not args.allow_zero_rows,
        rng_engine=args.rng_engine,
        overwrite=args.overwrite,
        verbose=args.verbose,
        cardinality_encoding=args.cardinality_encoding,
        maximal_candidate=args.maximal_candidate,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Wrote {len(rows)} Bernoulli simulation rows.")


if __name__ == "__main__":
    main()
