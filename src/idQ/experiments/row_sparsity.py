#!/usr/bin/env python3
"""Row-sparsity simulations for the conjunctive identifiability algorithm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .common import (
    actual_row_size_metadata,
    parse_probability_list,
    rng_label,
    run_design_expr,
    sample_row_sparse_q,
)
from ..sat import NON_EQUIVALENCE_ENCODINGS, normalize_cardinality_encoding, normalize_solver_name


from ..paths import data_directory


def _design_hash(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def default_output_path(
    J: int,
    K: int,
    m: int,
    min_row_size: int,
    row_size_distribution: str,
    row_size_probs: Sequence[float] | None,
    seed: int,
    solver: str | int,
    *,
    rng_engine: str,
    cardinality_encoding: str = "exclude_x",
    maximal_candidate: bool = False,
) -> Path:
    design = actual_row_size_metadata(
        K=K,
        m=m,
        min_row_size=min_row_size,
        row_size_distribution=row_size_distribution,
        row_size_probs=row_size_probs,
    )
    tag = _design_hash(design)
    return data_directory() / "raw" / "row_sparsity" / (
        f"rowsparse_conj_solver-{normalize_solver_name(solver)}"
        f"_enc-{normalize_cardinality_encoding(cardinality_encoding)}_max-{int(bool(maximal_candidate))}"
        f"_J{J}_K{K}_m{m}_dmin{min_row_size}"
        f"_dist-{row_size_distribution}_design-{tag}"
        f"_rng-{rng_engine}_seed{seed}_diag.csv"
    )


def run_expr(
    J: int,
    K: int,
    N: int,
    m: int,
    seed: int,
    solver: str | int = "glucose42",
    output_csv: str | None = None,
    *,
    min_row_size: int = 1,
    row_size_distribution: str = "uniform",
    row_size_probs: Sequence[float] | None = None,
    rng_engine: str = "legacy",
    overwrite: bool = False,
    verbose: bool = False,
    cardinality_encoding: str = "exclude_x",
    maximal_candidate: bool = False,
    checkpoint_every: int = 10,
) -> list[dict[str, Any]]:
    """Run one row-sparsity simulation job."""
    design = actual_row_size_metadata(
        K=K,
        m=m,
        min_row_size=min_row_size,
        row_size_distribution=row_size_distribution,
        row_size_probs=row_size_probs,
    )
    design_id = f"row_sparsity|conj|{_design_hash(design)}|rng={rng_label(rng_engine)}"
    metadata = {
        "design": "row_sparsity",
        "design_id": design_id,
        "p": "",
        **design,
    }

    if output_csv is None:
        output_csv = str(default_output_path(
            J,
            K,
            m,
            min_row_size,
            row_size_distribution,
            row_size_probs,
            seed,
            solver,
            rng_engine=rng_engine,
            cardinality_encoding=cardinality_encoding,
            maximal_candidate=maximal_candidate,
        ))

    def sampler(rng):
        return sample_row_sparse_q(
            J,
            K,
            m,
            rng,
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
        rng_engine=rng_engine,
        overwrite=overwrite,
        verbose=verbose,
        cardinality_encoding=cardinality_encoding,
        maximal_candidate=maximal_candidate,
        checkpoint_every=checkpoint_every,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run row-sparsity Q-matrix identifiability simulations."
    )
    parser.add_argument("J", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("m", type=int, help="Maximum row support size.")
    parser.add_argument("seed", type=int)
    parser.add_argument(
        "solver",
        nargs="?",
        default="glucose42",
        help="PySAT solver name or legacy code (-1, 0, 1, 2).",
    )
    parser.add_argument("--min-row-size", type=int, default=1)
    parser.add_argument(
        "--row-size-distribution",
        choices=["uniform", "fixed", "custom"],
        default="uniform",
    )
    parser.add_argument(
        "--row-size-probs",
        default=None,
        help="Comma-separated probabilities for distribution=custom.",
    )
    parser.add_argument("--rng-engine", choices=["legacy", "pcg64"], default="legacy")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    probabilities = parse_probability_list(args.row_size_probs)
    rows = run_expr(
        J=args.J,
        K=args.K,
        N=args.N,
        m=args.m,
        seed=args.seed,
        solver=args.solver,
        output_csv=args.output_csv,
        min_row_size=args.min_row_size,
        row_size_distribution=args.row_size_distribution,
        row_size_probs=probabilities,
        rng_engine=args.rng_engine,
        overwrite=args.overwrite,
        verbose=args.verbose,
        cardinality_encoding=args.cardinality_encoding,
        maximal_candidate=args.maximal_candidate,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Wrote {len(rows)} row-sparsity simulation rows.")


if __name__ == "__main__":
    main()
