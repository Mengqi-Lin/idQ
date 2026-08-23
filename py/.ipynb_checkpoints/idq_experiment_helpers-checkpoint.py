#!/usr/bin/env python
"""
Shared utilities for identifiability simulation experiments.

This module contains the common code used by both:
  1. Bernoulli-iid Q-matrix simulations; and
  2. row-sparsity/applied diagnostic setting simulations.

It intentionally writes one CSV per seed/job by default.  This avoids unsafe
concurrent appends to the same CSV when simulations are run in parallel.
"""

from __future__ import annotations

import csv
import itertools
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from Qbasis import get_basis, get_Q_from_Qbasis
from solve_SAT import solve_SAT, solve_SAT_fast
from idQ import (
    check_two_column_submatrices,
    check_three_column_submatrices,
    contains_identity_submatrix,
    lex_sort_columns,
)


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


# -----------------------------------------------------------------------------
# Basic diagnostics
# -----------------------------------------------------------------------------

def has_any_pure_node(Q: np.ndarray) -> bool:
    """
    Return True if Q contains at least one pure node.

    A pure node is a row equal to e_k for some k.  Since Q is binary, this is
    equivalent to having row sum equal to 1.
    """
    Q = np.asarray(Q)
    return bool(np.any(Q.sum(axis=1) == 1))


def violates_two_column_necessary(Q: np.ndarray) -> bool:
    """
    Return True if some pair of columns violates the two-column necessary check.

    For each pair (k1, k2), the two-column submatrix must contain both row
    patterns (1, 0) and (0, 1).  If either pattern is missing for any pair,
    the necessary condition fails.
    """
    Q = np.asarray(Q)
    _, K = Q.shape

    if K < 2:
        return False

    for k1, k2 in itertools.combinations(range(K), 2):
        c1 = Q[:, k1]
        c2 = Q[:, k2]

        has_10 = np.any((c1 == 1) & (c2 == 0))
        has_01 = np.any((c1 == 0) & (c2 == 1))

        if not (has_10 and has_01):
            return True

    return False


def count_representative_classes(Q_basis: np.ndarray, op: str = "conj") -> int:
    """
    Compute M = |R(Q_basis)| as the number of distinct columns of Phi(Q_basis).

    Parameters
    ----------
    Q_basis:
        Basis submatrix of Q.
    op:
        Boolean operator.  Use "conj" for the conjunctive/DINA operator and
        "disj" for the disjunctive/DINO operator.
    """
    Q_basis = np.asarray(Q_basis, dtype=np.int8)
    _, K = Q_basis.shape

    alphas = np.array(list(itertools.product([0, 1], repeat=K)), dtype=np.int8)

    if op == "conj":
        # Shape: (2^K, J_basis). Each row is one Phi column transposed.
        phi_cols = (alphas[:, None, :] >= Q_basis[None, :, :]).all(axis=2)
    elif op == "disj":
        phi_cols = ((alphas[:, None, :] & Q_basis[None, :, :]).any(axis=2))
    else:
        raise ValueError("op must be either 'conj' or 'disj'.")

    return int(np.unique(phi_cols.astype(np.int8), axis=0).shape[0])


def _to_csv_value(x: Any) -> Any:
    """Convert values that are inconvenient for CSV writing into stable strings."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return ";".join(str(v) for v in list(x))
    return x


# -----------------------------------------------------------------------------
# Q-matrix samplers
# -----------------------------------------------------------------------------

def sample_bernoulli_q(
    J: int,
    K: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Q with iid Bernoulli(p) entries."""
    if not (0 <= p <= 1):
        raise ValueError("p must lie in [0, 1].")
    return rng.binomial(1, p, size=(J, K)).astype(int)


def sample_row_sparse_q(
    J: int,
    K: int,
    m: int,
    rng: np.random.Generator,
    min_row_size: int = 1,
    row_size_distribution: str = "uniform",
    row_size_probs: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Sample a row-sparse Q matrix for applied diagnostic settings.

    The row support size D_j = ||q_j||_0 is sampled first.  Conditional on D_j,
    D_j attributes are sampled uniformly without replacement from {1, ..., K}.

    Defaults
    --------
    The default is D_j ~ Uniform{1, ..., m}.  This encodes the reviewer-facing
    design "||q_j||_0 <= m" while avoiding zero rows by default.

    Parameters
    ----------
    J, K:
        Dimensions of the Q matrix.
    m:
        Maximum row support size.  The actual maximum used is min(m, K).
    rng:
        NumPy random generator.
    min_row_size:
        Minimum row support size.  Use 2 if you want to exclude pure nodes by
        construction.  Use 0 only if zero rows are allowed in your experiment.
    row_size_distribution:
        One of {"uniform", "fixed", "custom"}.
        - "uniform": D_j is uniform on {min_row_size, ..., m}.
        - "fixed": D_j = m for all rows.
        - "custom": probabilities are supplied by row_size_probs over the same
          support {min_row_size, ..., m}.
    row_size_probs:
        Probabilities for the custom row-size distribution.  Length must equal
        m - min_row_size + 1 after m is truncated to K.
    """
    if J <= 0 or K <= 0:
        raise ValueError("J and K must be positive integers.")
    if m < 0:
        raise ValueError("m must be nonnegative.")

    m_eff = min(int(m), int(K))
    min_eff = int(min_row_size)

    if min_eff < 0:
        raise ValueError("min_row_size must be nonnegative.")
    if min_eff > m_eff:
        raise ValueError("min_row_size must be <= min(m, K).")

    support = np.arange(min_eff, m_eff + 1, dtype=int)

    if row_size_distribution == "uniform":
        sizes = rng.choice(support, size=J, replace=True)
    elif row_size_distribution == "fixed":
        sizes = np.full(J, m_eff, dtype=int)
    elif row_size_distribution == "custom":
        if row_size_probs is None:
            raise ValueError("row_size_probs must be supplied when distribution is 'custom'.")
        probs = np.asarray(row_size_probs, dtype=float)
        if len(probs) != len(support):
            raise ValueError(
                "row_size_probs has wrong length. Expected one probability for "
                f"each size in {support.tolist()}."
            )
        if np.any(probs < 0):
            raise ValueError("row_size_probs must be nonnegative.")
        total = probs.sum()
        if total <= 0:
            raise ValueError("row_size_probs must have positive sum.")
        probs = probs / total
        sizes = rng.choice(support, size=J, replace=True, p=probs)
    else:
        raise ValueError("row_size_distribution must be one of: uniform, fixed, custom.")

    Q = np.zeros((J, K), dtype=int)
    for j, d in enumerate(sizes):
        if d > 0:
            attrs = rng.choice(K, size=int(d), replace=False)
            Q[j, attrs] = 1

    return Q


def parse_probability_list(prob_string: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated probability string into a list of floats."""
    if prob_string is None or str(prob_string).strip() == "":
        return None
    return [float(x.strip()) for x in str(prob_string).split(",") if x.strip() != ""]


# -----------------------------------------------------------------------------
# Identifiability algorithm with diagnostics
# -----------------------------------------------------------------------------

def identifiability_expr(
    Q: np.ndarray,
    solver: int,
    op: str = "conj",
    verbose: bool = False,
) -> Tuple[int, Optional[np.ndarray], int, Dict[str, Any]]:
    """
    Run the identifiability algorithm and return diagnostics.

    Returns
    -------
    status:
        1 if identifiable, 0 otherwise.
    Q_bar:
        A non-equivalent alternative matrix when non-identifiability is found;
        otherwise None.
    branch:
        Integer code describing where the algorithm stopped.
    diag:
        Dictionary of timing and diagnostic quantities.
    """
    Q = np.asarray(Q, dtype=int).copy()

    diag: Dict[str, Any] = {
        "basis_time": 0.0,
        "trivial_check_time": 0.0,
        "two_col_check_time": 0.0,
        "three_col_check_time": 0.0,
        "identity_check_time": 0.0,
        "sat_time": 0.0,
    }

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    t_alg_start = time.perf_counter()

    # Step 1: get Q_basis.
    t = time.perf_counter()
    (
        Q_basis,
        basis_to_original,
        orig_indices_for_basis,
        Q_unique,
        unique_to_original,
        basis_to_unique,
    ) = get_basis(Q)
    diag["basis_time"] = time.perf_counter() - t

    J_basis, K = Q_basis.shape
    diag["J_basis"] = int(J_basis)

    def finish(status: int, Q_bar: Optional[np.ndarray], branch: int):
        # Algorithm time excludes the extra diagnostics below.
        diag["algorithm_time"] = time.perf_counter() - t_alg_start

        # Extra diagnostics for tables.  These should not be counted as the
        # algorithm runtime.
        t_diag = time.perf_counter()

        diag["identifiable"] = int(status)
        diag["branch"] = int(branch)
        diag["branch_label"] = BRANCH_LABELS.get(branch, "unknown")
        diag["sat_called"] = int(branch in (5, 6))

        # Original-Q diagnostics: use these for pure-node/completeness gaps.
        diag["identity_original"] = int(contains_identity_submatrix(Q))
        diag["not_complete_original"] = int(not diag["identity_original"])
        diag["has_pure_node_original"] = int(has_any_pure_node(Q))
        diag["no_pure_nodes_original"] = int(not diag["has_pure_node_original"])
        diag["two_col_violation_original"] = int(violates_two_column_necessary(Q))

        # Basis-Q diagnostics: use these for preprocessing/SAT behavior.
        diag["identity_basis"] = int(contains_identity_submatrix(Q_basis))
        diag["two_col_violation_basis"] = int(violates_two_column_necessary(Q_basis))
        diag["M_basis"] = count_representative_classes(Q_basis, op=op)

        diag["preprocess_time"] = (
            diag["basis_time"]
            + diag["trivial_check_time"]
            + diag["two_col_check_time"]
            + diag["three_col_check_time"]
            + diag["identity_check_time"]
        )

        diag["diagnostic_time"] = time.perf_counter() - t_diag

        return status, Q_bar, branch, diag

    if J_basis < K:
        Q_basis_bar = np.eye(K, dtype=int)[:J_basis]
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        log("Q_basis has J < K, thus not identifiable.")
        return finish(0, Q_bar, -1)

    # Step 3: trivial non-identifiability checks on Q_basis.
    t = time.perf_counter()
    for k in range(K):
        if np.all(Q_basis[:, k] == 0):
            diag["trivial_check_time"] = time.perf_counter() - t
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 1
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            log("Q is trivially not identifiable: all-zero column.")
            return finish(0, Q_bar, 0)

        if np.all(Q_basis[:, k] == 1):
            diag["trivial_check_time"] = time.perf_counter() - t
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 0
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            log("Q is trivially not identifiable: all-one column.")
            return finish(0, Q_bar, 1)

    diag["trivial_check_time"] = time.perf_counter() - t

    # Two-column check.
    t = time.perf_counter()
    candidate = check_two_column_submatrices(Q_basis)
    diag["two_col_check_time"] = time.perf_counter() - t

    if candidate is not None:
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        log("Q is not identifiable: two-column submatrix not identifiable.")
        return finish(0, Q_bar, 2)

    # Three-column check.
    t = time.perf_counter()
    candidate = check_three_column_submatrices(Q_basis)
    diag["three_col_check_time"] = time.perf_counter() - t

    if candidate is not None:
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        log("Q is not identifiable: three-column submatrix not identifiable.")
        return finish(0, Q_bar, 3)

    # Identity-submatrix direct check.
    t = time.perf_counter()
    has_identity_basis = contains_identity_submatrix(Q_basis)
    diag["identity_check_time"] = time.perf_counter() - t

    if has_identity_basis:
        log("Q is identifiable by identity-submatrix direct check.")
        return finish(1, None, 4)

    # SAT step.
    t = time.perf_counter()

    if solver == -1:
        solution = solve_SAT_fast(Q_basis)
        diag["sat_time"] = time.perf_counter() - t

        if solution is not None:
            Q_basis_bar = solution
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return finish(0, Q_bar, 5)
        return finish(1, None, 6)

    Q_sorted, sorted_to_original = lex_sort_columns(Q_basis)

    if solver == 0:
        solution = solve_SAT(Q_sorted, solver_name="cadical195")
    elif solver == 1:
        solution = solve_SAT(Q_sorted, solver_name="Glucose42")
    elif solver == 2:
        # Keep this matching your current convention.  Change the solver name
        # here if solver == 2 should map to a different backend.
        solution = solve_SAT(Q_sorted, solver_name="Glucose42")
    else:
        raise ValueError(f"Unknown solver code: {solver}")

    diag["sat_time"] = time.perf_counter() - t

    if solution is not None:
        Q_basis_bar = solution[:, sorted_to_original]
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return finish(0, Q_bar, 5)

    return finish(1, None, 6)


# -----------------------------------------------------------------------------
# Generic simulation runner
# -----------------------------------------------------------------------------

def run_design_expr(
    J: int,
    K: int,
    N: int,
    seed: int,
    solver: int,
    sampler: Callable[[np.random.Generator], np.ndarray],
    metadata: Optional[Mapping[str, Any]] = None,
    output_csv: Optional[str] = None,
    op: str = "conj",
    append: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generic simulation loop for any Q-matrix sampler.

    Parameters
    ----------
    sampler:
        Function taking a NumPy Generator and returning a sampled Q matrix.
    metadata:
        Dictionary of design-specific fields to store in every row, e.g.
        {"design": "bernoulli", "p": 0.7} or
        {"design": "row_sparsity", "m": 3, ...}.
    append:
        If False, overwrite the seed-specific CSV.  If True, append to it.
        For parallel jobs, prefer append=False and one output file per seed/job.
    """
    if output_csv is None:
        raise ValueError("output_csv must be provided by the design-specific driver.")

    metadata = dict(metadata or {})
    rng = np.random.default_rng(seed)
    results_all: List[Dict[str, Any]] = []

    outdir = os.path.dirname(output_csv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    mode = "a" if append else "w"

    with open(output_csv, mode=mode, newline="") as csvfile:
        writer: Optional[csv.DictWriter] = None

        for i in range(N):
            Q = sampler(rng)

            status, Q_bar, branch, diag = identifiability_expr(
                Q,
                solver=solver,
                op=op,
                verbose=verbose,
            )

            row: Dict[str, Any] = {
                "J": int(J),
                "K": int(K),
                "N": int(N),
                "seed": int(seed),
                "sim": int(i),
                "solver": int(solver),
                "op": op,
            }
            row.update({k: _to_csv_value(v) for k, v in metadata.items()})
            row.update({k: _to_csv_value(v) for k, v in diag.items()})

            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
                if not (append and file_exists):
                    writer.writeheader()

            writer.writerow(row)
            csvfile.flush()
            results_all.append(row)

    return results_all
