"""Exact SAT formulations for conjunctive Q-matrix identifiability.

The canonical formulation uses the Boolean factorization

    Q = Q_bar odot H

after the two-column preprocessing check. In that branch the nonempty column
supports of Q form an antichain, so Q_bar is equivalent to Q exactly when H is
a permutation matrix. Since every column of H is then nonempty, non-equivalence
is encoded by ``sum(H) >= K + 1``.

The former response-profile formulation is retained as an independent
reference implementation for regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Hashable, Sequence
import warnings

import numpy as np

from .utils import (
    cardinality_bounds,
    column_inclusion_holds,
    equivalent_up_to_column_permutation,
    find_boolean_factor,
    has_nonempty_antichain_columns,
    is_valid_counterexample,
    is_valid_factorization_counterexample,
    lex_sort_columns,
    representative_supports,
    validate_binary_matrix,
)


LEGACY_SOLVER_CODES = {
    -1: "cadical195",  # formerly an incomplete restricted formulation
    0: "cadical195",
    1: "glucose42",
    2: "minisat22",
}

# The prefix encoding exploits nonempty H columns. The other choices use
# PySAT's general cardinality encoders and are useful for reproducible studies.
CARDINALITY_ENCODINGS = (
    "prefix", "prefix_oneway", "seqcounter", "cardnetwrk", "totalizer", "legacy",
)
# The default exclude_x and alternative exclude_h depend on Q and the complete
# lexicographic ordering of X;
# unlike the entries above, they are not standalone cardinality encoders.
SINGLE_CLAUSE_ENCODINGS = ("exclude_x", "exclude_h")
NON_EQUIVALENCE_ENCODINGS = CARDINALITY_ENCODINGS + SINGLE_CLAUSE_ENCODINGS


def normalize_cardinality_encoding(encoding: str) -> str:
    """Validate an encoding choice even when preprocessing avoids SAT."""
    if not isinstance(encoding, str) or encoding not in NON_EQUIVALENCE_ENCODINGS:
        raise ValueError(
            f"Unknown cardinality encoding {encoding!r}; choose one of "
            f"{NON_EQUIVALENCE_ENCODINGS}."
        )
    return encoding


class VariablePool:
    """Small deterministic replacement for PySAT's IDPool."""

    def __init__(self) -> None:
        self._ids: dict[Hashable, int] = {}
        self.top = 0

    def id(self, key: Hashable) -> int:
        if key not in self._ids:
            self.top += 1
            self._ids[key] = self.top
        return self._ids[key]


@dataclass(frozen=True)
class SATEncoding:
    """CNF plus the metadata needed to decode a SAT model."""

    clauses: tuple[tuple[int, ...], ...]
    n_variables: int
    decision_variables: tuple[tuple[int, ...], ...]
    factor_variables: tuple[tuple[int, ...], ...] = ()
    formulation: str = "boolean_factorization"
    cardinality_bounds: tuple[int, ...] = ()
    n_supports: int = 0
    cardinality_encoding: str | None = None


@dataclass(frozen=True)
class SATOutcome:
    """Result of one exact SAT verification."""

    satisfiable: bool
    counterexample: np.ndarray | None
    factorization: np.ndarray | None
    solver_name: str
    formulation: str
    n_variables: int
    n_clauses: int
    solver_stats: dict[str, Any]
    cardinality_encoding: str | None = None


def normalize_solver_name(solver: str | int | None) -> str:
    """Normalize solver names and retain the old integer CLI codes safely."""
    if solver is None:
        return "glucose42"
    if isinstance(solver, (int, np.integer)):
        try:
            return LEGACY_SOLVER_CODES[int(solver)]
        except KeyError as exc:
            raise ValueError(
                f"Unknown legacy solver code {solver!r}; use one of "
                f"{sorted(LEGACY_SOLVER_CODES)} or a PySAT solver name."
            ) from exc
    name = str(solver).strip().lower()
    if not name:
        raise ValueError("solver_name must not be empty.")
    if name.lstrip("-").isdigit():
        code = int(name)
        if code in LEGACY_SOLVER_CODES:
            return LEGACY_SOLVER_CODES[code]
        raise ValueError(
            f"Unknown legacy solver code {code!r}; use one of "
            f"{sorted(LEGACY_SOLVER_CODES)} or a PySAT solver name."
        )
    return name


# ---------------------------------------------------------------------------
# Shared symmetry and cardinality helpers
# ---------------------------------------------------------------------------


def _add_lex_nonincreasing(
    clauses: list[list[int]],
    pool: VariablePool,
    X: list[list[int]],
) -> None:
    """Encode weak non-increasing lexicographic order on adjacent columns."""
    J = len(X)
    K = len(X[0])
    for k in range(K - 1):
        equal_prefix = pool.id(("lex_equal", k, 0))
        clauses.append([equal_prefix])

        for j in range(J):
            left = X[j][k]
            right = X[j][k + 1]
            clauses.append([-equal_prefix, left, -right])

            if j == J - 1:
                continue

            next_equal = pool.id(("lex_equal", k, j + 1))
            clauses.append([-next_equal, equal_prefix])
            clauses.append([-next_equal, -left, right])
            clauses.append([-next_equal, left, -right])
            clauses.append([-equal_prefix, -left, -right, next_equal])
            clauses.append([-equal_prefix, left, right, next_equal])
            equal_prefix = next_equal


def _add_at_most_sequential(
    clauses: list[list[int]],
    pool: VariablePool,
    literals: Sequence[int],
    bound: int,
    row_index: Hashable,
) -> None:
    """Encode ``sum(literals) <= bound`` using a sequential counter."""
    n = len(literals)
    if bound >= n:
        return
    if bound < 0:
        clauses.append([])
        return
    if bound == 0:
        clauses.extend([[-literal] for literal in literals])
        return

    counters: dict[tuple[int, int], int] = {}
    for i in range(1, n + 1):
        for c in range(1, min(i, bound + 1) + 1):
            counters[i, c] = pool.id(("row_counter", row_index, i, c))

    for i, literal in enumerate(literals, start=1):
        clauses.append([-literal, counters[i, 1]])

        for c in range(1, min(i - 1, bound + 1) + 1):
            clauses.append([-counters[i - 1, c], counters[i, c]])

        for c in range(2, min(i, bound + 1) + 1):
            clauses.append([
                -literal,
                -counters[i - 1, c - 1],
                counters[i, c],
            ])

    clauses.append([-counters[n, bound + 1]])


# ---------------------------------------------------------------------------
# Boolean-factorization formulation (canonical)
# ---------------------------------------------------------------------------


def _add_candidate_admissibility(
    clauses: list[list[int]],
    X: list[list[int]],
    *,
    require_nonzero_columns: bool,
) -> None:
    """Encode the requested comparison-class restrictions on ``Q_bar``."""
    J = len(X)
    K = len(X[0])
    clauses.extend([list(X[j]) for j in range(J)])
    if require_nonzero_columns:
        clauses.extend([[X[j][k] for j in range(J)] for k in range(K)])


def _add_boolean_factorization(
    clauses: list[list[int]],
    pool: VariablePool,
    X: list[list[int]],
    H: list[list[int]],
    Q: np.ndarray,
) -> None:
    """Encode ``Q = X odot H`` exactly."""
    J, K = Q.shape
    for j in range(J):
        for k in range(K):
            if Q[j, k] == 0:
                clauses.extend(
                    [-X[j][ell], -H[ell][k]] for ell in range(K)
                )
                continue

            witnesses: list[int] = []
            for ell in range(K):
                witness = pool.id(("factor_witness", j, k, ell))
                witnesses.append(witness)
                clauses.append([-witness, X[j][ell]])
                clauses.append([-witness, H[ell][k]])
            clauses.append(witnesses)


def _add_nonpermutation_factor(
    clauses: list[list[int]],
    pool: VariablePool,
    H: list[list[int]],
    *,
    encoding: str = "prefix",
) -> None:
    """Require nonempty columns and ``sum(H) >= K + 1`` exactly.

    This standalone helper defaults to prefix OR gates and witnesses for a
    second one in a column. Its size is O(K**2) in variables, clauses, and literals.
    Witnesses imply the corresponding conjunction; they need not be unique.
    General PySAT encodings are selectable for controlled comparisons.
    """
    encoding = normalize_cardinality_encoding(encoding)
    if encoding in SINGLE_CLAUSE_ENCODINGS:
        raise ValueError(
            "Single-clause exclusions require Q and sorted X; use "
            "build_factorization_sat_instance instead."
        )
    K = len(H)
    if encoding == "legacy":
        # Retain the former clause order for reproducible comparisons. This
        # uses only K auxiliaries, but O(K**3) literal occurrences.
        selected_columns = []
        for k in range(K):
            column = [H[ell][k] for ell in range(K)]
            clauses.append(column)
            selected = pool.id(("h_multiple_ones", k))
            selected_columns.append(selected)
            for ell in range(K):
                clauses.append([-selected] + [column[r] for r in range(K) if r != ell])
        clauses.append(selected_columns)
        return

    for k in range(K):
        clauses.append([H[ell][k] for ell in range(K)])
    if K == 1:
        # No 1-by-1 factor can contain at least two ones.
        # Use contradictory units: some PySAT bootstrap paths inspect the
        # first literal of every clause and do not accept an empty clause.
        clauses.append([-H[0][0]])
        return

    if encoding not in ("prefix", "prefix_oneway"):
        from pysat.card import CardEnc, EncType

        cnf = CardEnc.atleast(
            lits=[literal for row in H for literal in row],
            bound=K + 1,
            top_id=pool.top,
            encoding=getattr(EncType, encoding),
        )
        clauses.extend(cnf.clauses)
        # Reserve all externally allocated identifiers for later constraints.
        pool.top = max(pool.top, cnf.nv)
        return

    witnesses: list[int] = []
    for k in range(K):
        prefix = H[0][k]
        for ell in range(1, K):
            witness = pool.id(("h_second_one", k, ell))
            witnesses.append(witness)
            clauses.append([-witness, prefix])
            clauses.append([-witness, H[ell][k]])
            if ell < K - 1:
                next_prefix = pool.id(("h_prefix", k, ell))
                clauses.append([-next_prefix, prefix, H[ell][k]])
                if encoding == "prefix":
                    clauses.append([-prefix, next_prefix])
                    clauses.append([-H[ell][k], next_prefix])
                # For prefix_oneway, a selected prefix only promises that
                # an earlier one exists; recursively expanding the single
                # implication proves that promise without forcing all u's.
                prefix = next_prefix
    clauses.append(witnesses)


def build_factorization_sat_instance(
    Q: Sequence[Sequence[int]],
    *,
    require_nonzero_columns: bool = False,
    symmetry_breaking: bool = True,
    maximal_candidate: bool = False,
    cardinality_encoding: str = "exclude_x",
) -> SATEncoding:
    """Build the compact exact CNF used after the two-column check.

    With ``maximal_candidate``, X[j, ell] is forced to one whenever row ell of
    H is contained in row j of Q. For fixed H this is the largest possible X,
    and adding these entries cannot change the Boolean product. Any existing
    nonzero rows/columns stay nonzero; the antichain condition and unchanged H
    preserve non-equivalence. Columns of this X and corresponding rows of H
    can be jointly permuted to retain weak lexicographic symmetry breaking.

    The default ``exclude_x`` replaces the cardinality construction with one
    clause excluding the sorted copy of Q from X. ``exclude_h`` instead requires an
    entry outside the unique permutation P satisfying Q = sorted(Q) odot P.
    Both retain the implied nonempty-column clauses on H and require
    ``symmetry_breaking=True``. The product uses Q in its supplied column order;
    excluding its sorted copy is equivalent to sorting Q before encoding.
    Earlier cardinality encodings remain available through explicit selection.
    """
    cardinality_encoding = normalize_cardinality_encoding(cardinality_encoding)
    if cardinality_encoding in SINGLE_CLAUSE_ENCODINGS and not symmetry_breaking:
        raise ValueError(
            f"{cardinality_encoding!r} requires symmetry_breaking=True, "
            "because its exclusion assumes all columns of X are lexicographically sorted."
        )
    Q_array = validate_binary_matrix(Q)
    if not has_nonempty_antichain_columns(Q_array):
        raise ValueError(
            "The factor-cardinality shortcut requires nonempty, pairwise "
            "incomparable columns of Q; run the two-column check first."
        )

    J, K = Q_array.shape
    pool = VariablePool()
    X = [[pool.id(("x", j, ell)) for ell in range(K)] for j in range(J)]
    H = [[pool.id(("h", ell, k)) for k in range(K)] for ell in range(K)]
    clauses: list[list[int]] = []

    _add_candidate_admissibility(
        clauses,
        X,
        require_nonzero_columns=require_nonzero_columns,
    )
    if symmetry_breaking:
        _add_lex_nonincreasing(clauses, pool, X)
    _add_boolean_factorization(clauses, pool, X, H, Q_array)
    if maximal_candidate:
        # Existing zero-product clauses give X[j,ell] -> not H[ell,k] for
        # every zero entry Q[j,k]. This adds the converse implication:
        # (all such H[ell,k] are zero) -> X[j,ell]. If there are no zero
        # entries in this Q row, the clause is simply the unit X[j,ell].
        for j in range(J):
            zeros = np.flatnonzero(Q_array[j] == 0)
            for ell in range(K):
                clauses.append([X[j][ell]] + [H[ell][k] for k in zeros])
    if cardinality_encoding in SINGLE_CLAUSE_ENCODINGS:
        # Keep these clauses and their order identical to the prefix baseline.
        clauses.extend([[H[ell][k] for ell in range(K)] for k in range(K)])
        Q_sorted, restore_order = lex_sort_columns(Q_array)
        if cardinality_encoding == "exclude_x":
            _add_not_equal(clauses, X, Q_sorted)
        elif K == 1:
            # The off-permutation disjunction is empty. Contradict the
            # nonempty-column unit without passing an empty clause to PySAT.
            clauses.append([-H[0][0]])
        else:
            # P[restore_order[k], k] = 1, since Q_sorted[:, restore_order] = Q.
            # Under the antichain condition, X equivalent to Q forces H=P.
            # Conversely, nonempty H columns with no off-P entry force H=P.
            clauses.append([
                H[ell][k]
                for ell in range(K)
                for k in range(K)
                if ell != restore_order[k]
            ])
    else:
        _add_nonpermutation_factor(clauses, pool, H, encoding=cardinality_encoding)

    return SATEncoding(
        clauses=tuple(tuple(clause) for clause in clauses),
        n_variables=pool.top,
        decision_variables=tuple(tuple(row) for row in X),
        factor_variables=tuple(tuple(row) for row in H),
        formulation="boolean_factorization",
        cardinality_encoding=cardinality_encoding,
    )


# The public builder now denotes the canonical formulation.
build_sat_instance = build_factorization_sat_instance


# ---------------------------------------------------------------------------
# Former response-profile formulation (independent regression reference)
# ---------------------------------------------------------------------------


def _add_not_equal(
    clauses: list[list[int]],
    X: list[list[int]],
    Q_sorted: np.ndarray,
) -> None:
    clauses.append([
        -X[j][k] if Q_sorted[j, k] else X[j][k]
        for j in range(Q_sorted.shape[0])
        for k in range(Q_sorted.shape[1])
    ])


def _add_row_cardinality(
    clauses: list[list[int]],
    pool: VariablePool,
    X: list[list[int]],
    bounds: Sequence[int],
) -> None:
    for j, row_literals in enumerate(X):
        clauses.append(list(row_literals))
        _add_at_most_sequential(
            clauses,
            pool,
            row_literals,
            int(bounds[j]),
            row_index=j,
        )


def _add_column_inclusion(
    clauses: list[list[int]],
    pool: VariablePool,
    X: list[list[int]],
    supports: Sequence[Sequence[int]],
) -> None:
    J = len(X)
    K = len(X[0])
    for support_index, support_tuple in enumerate(supports):
        support = set(support_tuple)
        for outside_row in range(J):
            if outside_row in support:
                continue

            witnesses: list[int] = []
            for k in range(K):
                witness = pool.id(("c2_witness", support_index, outside_row, k))
                witnesses.append(witness)
                clauses.append([-witness, X[outside_row][k]])
                clauses.extend([-witness, -X[j][k]] for j in support)
            clauses.append(witnesses)


def build_profile_sat_instance(
    Q_sorted: Sequence[Sequence[int]],
) -> SATEncoding:
    """Build the former exact response-profile CNF for cross-validation."""
    Q_array = validate_binary_matrix(Q_sorted)
    canonical, _ = lex_sort_columns(Q_array)
    if not np.array_equal(Q_array, canonical):
        raise ValueError("Q_sorted must have non-increasing lexicographic columns.")

    J, K = Q_array.shape
    bounds = cardinality_bounds(Q_array)
    supports = representative_supports(Q_array)
    pool = VariablePool()
    X = [[pool.id(("x", j, k)) for k in range(K)] for j in range(J)]
    clauses: list[list[int]] = []

    _add_lex_nonincreasing(clauses, pool, X)
    _add_not_equal(clauses, X, Q_array)
    _add_row_cardinality(clauses, pool, X, bounds)
    _add_column_inclusion(clauses, pool, X, supports)

    return SATEncoding(
        clauses=tuple(tuple(clause) for clause in clauses),
        n_variables=pool.top,
        decision_variables=tuple(tuple(row) for row in X),
        formulation="response_profile",
        cardinality_bounds=tuple(int(bound) for bound in bounds),
        n_supports=len(supports),
    )


# ---------------------------------------------------------------------------
# Independent small-instance oracles and solver front ends
# ---------------------------------------------------------------------------


def _profile_oracle_counterexample(
    Q_sorted: np.ndarray,
    bounds: Sequence[int],
    *,
    max_decision_variables: int = 20,
) -> np.ndarray | None:
    J, K = Q_sorted.shape
    if J * K > max_decision_variables:
        raise ValueError(
            "The exhaustive oracle is limited to "
            f"{max_decision_variables} decision variables; got {J * K}."
        )

    target_columns = tuple(tuple(column) for column in Q_sorted.T)
    for values in product((0, 1), repeat=J * K):
        candidate = np.asarray(values, dtype=int).reshape(J, K)
        candidate_columns = tuple(tuple(column) for column in candidate.T)
        if tuple(sorted(candidate_columns, reverse=True)) != candidate_columns:
            continue
        if candidate_columns == target_columns:
            continue
        row_sums = candidate.sum(axis=1)
        if np.any(row_sums == 0) or np.any(row_sums > np.asarray(bounds)):
            continue
        if column_inclusion_holds(Q_sorted, candidate):
            return candidate
    return None


def _factorization_oracle_counterexample(
    Q: np.ndarray,
    *,
    require_nonzero_columns: bool,
    max_decision_variables: int = 20,
) -> tuple[np.ndarray, np.ndarray] | None:
    J, K = Q.shape
    if J * K > max_decision_variables:
        raise ValueError(
            "The exhaustive oracle is limited to "
            f"{max_decision_variables} decision variables; got {J * K}."
        )

    for values in product((0, 1), repeat=J * K):
        candidate = np.asarray(values, dtype=int).reshape(J, K)
        candidate_columns = tuple(tuple(column) for column in candidate.T)
        if tuple(sorted(candidate_columns, reverse=True)) != candidate_columns:
            continue
        if np.any(candidate.sum(axis=1) == 0):
            continue
        if require_nonzero_columns and np.any(candidate.sum(axis=0) == 0):
            continue
        if equivalent_up_to_column_permutation(Q, candidate):
            continue
        factor = find_boolean_factor(Q, candidate)
        if factor is not None and int(factor.sum()) >= K + 1:
            return candidate, factor
    return None


def _run_external_solver(
    encoding: SATEncoding,
    solver_name: str,
) -> tuple[bool, set[int], dict[str, Any]]:
    try:
        from pysat.solvers import Solver
    except ImportError as exc:
        raise ImportError(
            "The SAT step requires python-sat. Install the dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    try:
        with Solver(
            name=solver_name,
            bootstrap_with=encoding.clauses,
        ) as solver:
            satisfiable = solver.solve()
            if satisfiable is None:
                raise RuntimeError("SAT solver returned UNKNOWN; no decision was made.")
            model = set(solver.get_model() or []) if satisfiable else set()
            try:
                stats = dict(solver.accum_stats() or {})
            except NotImplementedError:
                # Kissat provides decisions and models, but not PySAT statistics.
                stats = {}
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize or run PySAT solver {solver_name!r}."
        ) from exc
    return satisfiable, model, stats


def _decode_matrix(
    variables: Sequence[Sequence[int]],
    model: set[int],
) -> np.ndarray:
    return np.asarray(
        [[int(variable in model) for variable in row] for row in variables],
        dtype=int,
    )


def solve_sat(
    Q: Sequence[Sequence[int]],
    solver_name: str | int | None = "glucose42",
    *,
    oracle_max_variables: int = 20,
    require_nonzero_columns: bool = False,
    symmetry_breaking: bool = True,
    maximal_candidate: bool = False,
    cardinality_encoding: str = "exclude_x",
) -> SATOutcome:
    """Run factorization SAT with single-clause exclusion after the two-column check.

    The default ``exclude_x`` requires weak lexicographic symmetry breaking.
    Select an earlier cardinality encoding explicitly for comparisons without
    symmetry breaking. Glucose 4.2 and ``maximal_candidate=False`` remain default.
    """
    Q_array = validate_binary_matrix(Q)
    encoding = build_factorization_sat_instance(
        Q_array,
        require_nonzero_columns=require_nonzero_columns,
        symmetry_breaking=symmetry_breaking,
        maximal_candidate=maximal_candidate,
        cardinality_encoding=cardinality_encoding,
    )
    normalized_solver = normalize_solver_name(solver_name)

    if normalized_solver in {"oracle", "bruteforce"}:
        result = _factorization_oracle_counterexample(
            Q_array,
            require_nonzero_columns=require_nonzero_columns,
            max_decision_variables=oracle_max_variables,
        )
        stats: dict[str, Any] = {}
        if result is None:
            counterexample = None
            factorization = None
        else:
            counterexample, factorization = result
            if maximal_candidate:
                counterexample = (
                    (Q_array == 0).astype(int) @ factorization.T == 0
                ).astype(int)
                if symmetry_breaking:
                    order = sorted(
                        range(Q_array.shape[1]),
                        key=lambda ell: tuple(counterexample[:, ell]),
                        reverse=True,
                    )
                    counterexample = counterexample[:, order]
                    factorization = factorization[order, :]
    else:
        satisfiable, model, stats = _run_external_solver(
            encoding,
            normalized_solver,
        )
        if satisfiable:
            counterexample = _decode_matrix(encoding.decision_variables, model)
            factorization = _decode_matrix(encoding.factor_variables, model)
        else:
            counterexample = None
            factorization = None

    if counterexample is None:
        return SATOutcome(
            satisfiable=False,
            counterexample=None,
            factorization=None,
            solver_name=normalized_solver,
            formulation=encoding.formulation,
            n_variables=encoding.n_variables,
            n_clauses=len(encoding.clauses),
            solver_stats=stats,
            cardinality_encoding=encoding.cardinality_encoding,
        )

    assert factorization is not None
    if not is_valid_factorization_counterexample(
        Q_array,
        counterexample,
        factorization,
        require_nonzero_columns=require_nonzero_columns,
    ):
        raise RuntimeError(
            "The SAT model failed independent factorization-certificate validation."
        )

    return SATOutcome(
        satisfiable=True,
        counterexample=counterexample,
        factorization=factorization,
        solver_name=normalized_solver,
        formulation=encoding.formulation,
        n_variables=encoding.n_variables,
        n_clauses=len(encoding.clauses),
        solver_stats=stats,
        cardinality_encoding=encoding.cardinality_encoding,
    )


def solve_sat_profile(
    Q: Sequence[Sequence[int]],
    solver_name: str | int | None = "glucose42",
    *,
    oracle_max_variables: int = 20,
) -> SATOutcome:
    """Run the former exact response-profile solver for cross-validation."""
    Q_array = validate_binary_matrix(Q)
    Q_sorted, restore_order = lex_sort_columns(Q_array)
    encoding = build_profile_sat_instance(Q_sorted)
    normalized_solver = normalize_solver_name(solver_name)

    if normalized_solver in {"oracle", "bruteforce"}:
        sorted_counterexample = _profile_oracle_counterexample(
            Q_sorted,
            encoding.cardinality_bounds,
            max_decision_variables=oracle_max_variables,
        )
        stats: dict[str, Any] = {}
    else:
        satisfiable, model, stats = _run_external_solver(
            encoding,
            normalized_solver,
        )
        sorted_counterexample = (
            _decode_matrix(encoding.decision_variables, model)
            if satisfiable
            else None
        )

    if sorted_counterexample is None:
        return SATOutcome(
            satisfiable=False,
            counterexample=None,
            factorization=None,
            solver_name=normalized_solver,
            formulation=encoding.formulation,
            n_variables=encoding.n_variables,
            n_clauses=len(encoding.clauses),
            solver_stats=stats,
        )

    if not is_valid_counterexample(Q_sorted, sorted_counterexample):
        raise RuntimeError("The profile SAT model failed counterexample validation.")
    if np.any(sorted_counterexample.sum(axis=1) > encoding.cardinality_bounds):
        raise RuntimeError("The profile SAT model violated a row-cardinality bound.")

    counterexample = sorted_counterexample[:, restore_order]
    if not is_valid_counterexample(Q_array, counterexample):
        raise RuntimeError("Column-order restoration corrupted the counterexample.")

    return SATOutcome(
        satisfiable=True,
        counterexample=counterexample,
        factorization=None,
        solver_name=normalized_solver,
        formulation=encoding.formulation,
        n_variables=encoding.n_variables,
        n_clauses=len(encoding.clauses),
        solver_stats=stats,
    )


def solve_SAT(Q, solver_name="glucose42"):
    """Compatibility wrapper returning a counterexample matrix or ``None``."""
    return solve_sat(Q, solver_name=solver_name).counterexample


def solve_SAT_fast(Q, solver_name="glucose42"):
    """Deprecated safe alias for the exact factorization formulation."""
    warnings.warn(
        "solve_SAT_fast is deprecated; the exact SAT formulation is now used.",
        DeprecationWarning,
        stacklevel=2,
    )
    return solve_SAT(Q, solver_name=solver_name)


__all__ = [
    "CARDINALITY_ENCODINGS",
    "SINGLE_CLAUSE_ENCODINGS",
    "NON_EQUIVALENCE_ENCODINGS",
    "LEGACY_SOLVER_CODES",
    "SATEncoding",
    "SATOutcome",
    "VariablePool",
    "build_factorization_sat_instance",
    "build_profile_sat_instance",
    "build_sat_instance",
    "normalize_solver_name",
    "normalize_cardinality_encoding",
    "solve_SAT",
    "solve_SAT_fast",
    "solve_sat",
    "solve_sat_profile",
]
