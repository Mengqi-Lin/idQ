"""Public implementation of Algorithm 1 for conjunctive Q matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import time
from typing import Any, Sequence

import numpy as np

from .basis import BasisReduction, reconstruct_from_basis, reduce_to_basis
from .sat import (
    SATOutcome, normalize_cardinality_encoding, normalize_solver_name, solve_sat,
)
from .utils import (
    canonical_columns,
    is_valid_counterexample,
    is_valid_factorization_counterexample,
    lex_sort_columns,
    validate_binary_matrix,
)


BRANCH_LABELS = {
    -1: "empty_basis_nonidentifiable",
    2: "two_column_check_failed",
    3: "three_column_check_failed",
    4: "identity_submatrix_identifiable",
    5: "SAT_found_counterexample",
    6: "SAT_unsat_identifiable",
}


@dataclass(frozen=True)
class IdentificationResult:
    """Auditable result returned by :func:`identify`."""

    identifiable: bool
    branch: int
    branch_label: str
    counterexample: np.ndarray | None = None
    factorization: np.ndarray | None = None
    violating_columns: tuple[int, ...] | None = None
    basis: np.ndarray | None = field(default=None, repr=False)
    timings: dict[str, float] = field(default_factory=dict)
    solver_name: str | None = None
    sat_formulation: str | None = None
    sat_cardinality_encoding: str | None = None
    sat_variables: int = 0
    sat_clauses: int = 0
    solver_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> int:
        """Legacy integer status: one for identifiable, zero otherwise."""
        return int(self.identifiable)


def contains_identity_submatrix(Q: Sequence[Sequence[int]]) -> bool:
    """Return whether ``Q`` contains every K-dimensional unit row."""
    array = validate_binary_matrix(Q)
    rows = {tuple(row) for row in array}
    return {
        tuple(row) for row in np.eye(array.shape[1], dtype=int)
    }.issubset(rows)


def has_any_pure_node(Q: Sequence[Sequence[int]]) -> bool:
    """Return whether at least one row is a unit vector."""
    array = validate_binary_matrix(Q)
    return bool(np.any(array.sum(axis=1) == 1))


def first_two_column_violation(
    Q: Sequence[Sequence[int]],
) -> tuple[int, int] | None:
    """Return the first pair missing either ``(1,0)`` or ``(0,1)``."""
    array = validate_binary_matrix(Q)
    for k1, k2 in combinations(range(array.shape[1]), 2):
        patterns = {tuple(row) for row in array[:, [k1, k2]]}
        if (1, 0) not in patterns or (0, 1) not in patterns:
            return k1, k2
    return None


def first_three_column_violation(
    Q: Sequence[Sequence[int]],
) -> tuple[int, int, int] | None:
    """Return the first triple containing no row with exactly one selected 1."""
    array = validate_binary_matrix(Q)
    for triple in combinations(range(array.shape[1]), 3):
        if not np.any(array[:, triple].sum(axis=1) == 1):
            return tuple(int(k) for k in triple)
    return None


def violates_two_column_necessary(Q: Sequence[Sequence[int]]) -> bool:
    return first_two_column_violation(Q) is not None


def violates_three_column_necessary(Q: Sequence[Sequence[int]]) -> bool:
    return first_three_column_violation(Q) is not None


# Compatibility names.  They now return the violating column tuple rather than
# fabricating an invalid global Q_bar witness.
check_two_column_submatrices = first_two_column_violation
check_three_column_submatrices = first_three_column_violation


def canonicalize(Q: Sequence[Sequence[int]]) -> np.ndarray:
    """Return ``Q`` with columns in canonical non-increasing lexicographic order."""
    return lex_sort_columns(Q)[0]


def _finish(
    *,
    identifiable: bool,
    branch: int,
    basis: np.ndarray,
    timings: dict[str, float],
    start_time: float,
    counterexample: np.ndarray | None = None,
    factorization: np.ndarray | None = None,
    violating_columns: tuple[int, ...] | None = None,
    sat_outcome: SATOutcome | None = None,
) -> IdentificationResult:
    final_timings = dict(timings)
    final_timings["algorithm_time"] = time.perf_counter() - start_time
    return IdentificationResult(
        identifiable=identifiable,
        branch=branch,
        branch_label=BRANCH_LABELS[branch],
        counterexample=counterexample,
        factorization=factorization,
        violating_columns=violating_columns,
        basis=basis,
        timings=final_timings,
        solver_name=None if sat_outcome is None else sat_outcome.solver_name,
        sat_formulation=None if sat_outcome is None else sat_outcome.formulation,
        sat_cardinality_encoding=(
            None if sat_outcome is None else sat_outcome.cardinality_encoding
        ),
        sat_variables=0 if sat_outcome is None else sat_outcome.n_variables,
        sat_clauses=0 if sat_outcome is None else sat_outcome.n_clauses,
        solver_stats={} if sat_outcome is None else sat_outcome.solver_stats,
    )


def identify(
    Q: Sequence[Sequence[int]],
    solver_name: str | int | None = "glucose42",
    *,
    verbose: bool = False,
    maximal_candidate: bool = False,
    cardinality_encoding: str = "exclude_x",
) -> IdentificationResult:
    """Run the exact algorithm from the manuscript.

    Preprocessing branches certify a yes/no conclusion but do not generally
    construct a global alternative matrix.  Consequently, ``counterexample``
    is populated only when the SAT solver explicitly returns and independently
    validates one.

    ``cardinality_encoding="exclude_x"`` (default) uses one clause to exclude
    the sorted copy of Q from the lexicographically ordered candidate matrix.
    ``exclude_h`` instead excludes the corresponding permutation factor.
    The earlier ``prefix``, ``prefix_oneway``, ``seqcounter``, ``cardnetwrk``,
    ``totalizer`` and ``legacy`` encodings remain explicit comparison options.
    The solver remains Glucose 4.2 for every encoding unless overridden.
    """
    array = validate_binary_matrix(Q)
    # Validate the solver choice up front, even if preprocessing later returns.
    normalized_solver = normalize_solver_name(solver_name)
    cardinality_encoding = normalize_cardinality_encoding(cardinality_encoding)
    start_time = time.perf_counter()
    timings = {
        "basis_time": 0.0,
        "identity_check_time": 0.0,
        "two_col_check_time": 0.0,
        "three_col_check_time": 0.0,
        "sat_time": 0.0,
    }

    def log(message: str) -> None:
        if verbose:
            print(message)

    t0 = time.perf_counter()
    reduction: BasisReduction = reduce_to_basis(array)
    basis = reduction.basis
    timings["basis_time"] = time.perf_counter() - t0

    if basis.shape[0] == 0:
        log("Not identifiable: the basis submatrix is empty.")
        return _finish(
            identifiable=False,
            branch=-1,
            basis=basis,
            timings=timings,
            start_time=start_time,
        )

    # Algorithm Step 1: the identity check must come first.  In particular,
    # Q=[[1]] is complete and identifiable.
    t0 = time.perf_counter()
    complete = contains_identity_submatrix(basis)
    timings["identity_check_time"] = time.perf_counter() - t0
    if complete:
        log("Identifiable: the basis contains an identity submatrix.")
        return _finish(
            identifiable=True,
            branch=4,
            basis=basis,
            timings=timings,
            start_time=start_time,
        )

    t0 = time.perf_counter()
    pair = first_two_column_violation(basis)
    timings["two_col_check_time"] = time.perf_counter() - t0
    if pair is not None:
        log(f"Not identifiable: two-column check failed at columns {pair}.")
        return _finish(
            identifiable=False,
            branch=2,
            basis=basis,
            timings=timings,
            start_time=start_time,
            violating_columns=pair,
        )

    t0 = time.perf_counter()
    triple = first_three_column_violation(basis)
    timings["three_col_check_time"] = time.perf_counter() - t0
    if triple is not None:
        log(f"Not identifiable: three-column check failed at columns {triple}.")
        return _finish(
            identifiable=False,
            branch=3,
            basis=basis,
            timings=timings,
            start_time=start_time,
            violating_columns=triple,
        )

    # Algorithm Step 2: exact SAT verification.
    t0 = time.perf_counter()
    sat_outcome = solve_sat(
        basis, solver_name=normalized_solver, maximal_candidate=maximal_candidate,
        cardinality_encoding=cardinality_encoding,
    )
    timings["sat_time"] = time.perf_counter() - t0

    if not sat_outcome.satisfiable:
        log("Identifiable: the exact SAT instance is unsatisfiable.")
        return _finish(
            identifiable=True,
            branch=6,
            basis=basis,
            timings=timings,
            start_time=start_time,
            sat_outcome=sat_outcome,
        )

    assert sat_outcome.counterexample is not None
    counterexample = reconstruct_from_basis(
        sat_outcome.counterexample,
        reduction.original_to_basis,
    )
    assert sat_outcome.factorization is not None
    if not is_valid_factorization_counterexample(
        array,
        counterexample,
        sat_outcome.factorization,
        # Zero rows are excluded by the standing model convention, but the
        # public API still accepts them for legacy reproducibility.  Basis
        # lifting necessarily restores such rows in both Q and Q_bar.
        require_nonzero_rows=False,
    ):
        raise RuntimeError(
            "The lifted basis factorization certificate failed validation."
        )

    log("Not identifiable: the SAT solver found a validated counterexample.")
    return _finish(
        identifiable=False,
        branch=5,
        basis=basis,
        timings=timings,
        start_time=start_time,
        counterexample=counterexample,
        factorization=sat_outcome.factorization,
        sat_outcome=sat_outcome,
    )


def identifiability(Q, solver="glucose42", *, verbose: bool = False):
    """Legacy ``(status, Q_bar)`` wrapper around :func:`identify`."""
    result = identify(Q, solver_name=solver, verbose=verbose)
    return result.status, result.counterexample


__all__ = [
    "BRANCH_LABELS",
    "IdentificationResult",
    "canonical_columns",
    "canonicalize",
    "check_three_column_submatrices",
    "check_two_column_submatrices",
    "contains_identity_submatrix",
    "first_three_column_violation",
    "first_two_column_violation",
    "has_any_pure_node",
    "identifiability",
    "identify",
    "lex_sort_columns",
    "violates_three_column_necessary",
    "violates_two_column_necessary",
]
