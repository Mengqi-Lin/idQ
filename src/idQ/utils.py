"""Mathematical utilities for conjunctive Q-matrix identifiability.

The functions in this module are deliberately solver-independent.  They are
also used by the test suite to check every SAT counterexample against the
algebraic column-inclusion condition from the manuscript.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Iterable, Sequence

import numpy as np


def validate_binary_matrix(Q: Sequence[Sequence[int]], *, name: str = "Q") -> np.ndarray:
    """Return ``Q`` as an integer array after validating its shape and entries."""
    array = np.asarray(Q)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0/1 entries.")
    return array.astype(int, copy=True)


def row_masks(Q: Sequence[Sequence[int]]) -> tuple[list[int], int]:
    """Return an integer bit mask for every row and the mask of all K attributes."""
    array = validate_binary_matrix(Q)
    _, K = array.shape
    masks = [
        sum(int(array[j, k]) << k for k in range(K))
        for j in range(array.shape[0])
    ]
    return masks, (1 << K) - 1


def representative_masks(Q: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return the closure of the row masks under bitwise OR.

    These masks are precisely the representative latent vectors needed for the
    conjunctive operator.  Sorting makes every downstream CNF deterministic.
    """
    array = validate_binary_matrix(Q)
    masks, _ = row_masks(array)
    closure = {0}
    for row_mask in masks:
        closure.update(mask | row_mask for mask in tuple(closure))
    return tuple(sorted(closure, key=lambda mask: (mask.bit_count(), mask)))


def representative_node_set(Q: Sequence[Sequence[int]]) -> set[tuple[int, ...]]:
    """Return the representative latent vectors as binary tuples."""
    array = validate_binary_matrix(Q)
    _, K = array.shape
    return {
        tuple((mask >> k) & 1 for k in range(K))
        for mask in representative_masks(array)
    }


def representative_supports(Q: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Return all distinct supports ``S_Q(alpha) = {j: alpha >= q_j}``.

    Empty and full supports are retained.  They are part of the mathematical
    response-column set, even though their C2 constraints are respectively
    implied by nonzero alternative rows and vacuous.
    """
    array = validate_binary_matrix(Q)
    masks, _ = row_masks(array)
    supports: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for alpha_mask in representative_masks(array):
        support = tuple(
            j for j, q_mask in enumerate(masks)
            if q_mask & ~alpha_mask == 0
        )
        if support not in seen:
            seen.add(support)
            supports.append(support)
    return tuple(supports)


def unique_pattern_supports(Q: Sequence[Sequence[int]]) -> list[set[int]]:
    """Backward-compatible list-of-sets form of :func:`representative_supports`."""
    return [set(support) for support in representative_supports(Q)]


def response_columns(Q: Sequence[Sequence[int]]) -> set[tuple[int, ...]]:
    """Return ``Cols(Phi(Q))`` without enumerating all ``2**K`` profiles."""
    array = validate_binary_matrix(Q)
    J, _ = array.shape
    return {
        tuple(int(j in support) for j in range(J))
        for support in representative_supports(array)
    }


def phi_matrix(Q: Sequence[Sequence[int]]) -> np.ndarray:
    """Return the full conjunctive ``Phi(Q)`` matrix (mainly for diagnostics)."""
    array = validate_binary_matrix(Q)
    _, K = array.shape
    alphas = np.asarray(list(product((0, 1), repeat=K)), dtype=int)
    return np.all(array[:, None, :] <= alphas[None, :, :], axis=2).astype(int)


# Compatibility names used by the original research scripts.
Phi_mat = phi_matrix
unique_response_columns = response_columns


def longest_chain_lengths(Q: Sequence[Sequence[int]]) -> np.ndarray:
    """Compute the manuscript's ``ell_j(Q)`` for every row.

    Starting from ``q_j``, each strict step ORs in another row.  This spans all
    strict chains in the representative-node lattice and avoids materializing
    its full edge set.
    """
    array = validate_binary_matrix(Q)
    masks, _ = row_masks(array)

    @lru_cache(maxsize=None)
    def depth(mask: int) -> int:
        next_masks = {mask | row_mask for row_mask in masks}
        next_masks.discard(mask)
        if not next_masks:
            return 0
        return 1 + max(depth(next_mask) for next_mask in next_masks)

    return np.asarray([depth(mask) for mask in masks], dtype=int)


def cardinality_bounds(Q: Sequence[Sequence[int]]) -> np.ndarray:
    """Return ``C_j(Q) = K - ell_j(Q)`` for all rows."""
    array = validate_binary_matrix(Q)
    return array.shape[1] - longest_chain_lengths(array)


# Original name retained for callers outside this package.
distances = longest_chain_lengths


def minimal_size_parent(supports: Iterable[Iterable[int]]) -> list[int | None]:
    """Choose a smallest strict superset for each support, if one exists.

    The exact SAT encoding no longer needs this optimization, but the helper is
    retained as a validated compatibility function.
    """
    support_sets = [frozenset(support) for support in supports]
    parents: list[int | None] = []
    for i, support in enumerate(support_sets):
        candidates = [
            (len(candidate), j)
            for j, candidate in enumerate(support_sets)
            if i != j and support < candidate
        ]
        parents.append(min(candidates)[1] if candidates else None)
    return parents


def canonical_columns(Q: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Canonical column multiset, invariant to attribute relabeling."""
    array = validate_binary_matrix(Q)
    return tuple(sorted((tuple(col) for col in array.T), reverse=True))


def lex_sort_columns(Q: Sequence[Sequence[int]]) -> tuple[np.ndarray, list[int]]:
    """Sort columns in non-increasing lexicographic order.

    The second return value restores a matrix in sorted-column coordinates to
    the original column order: ``Q_sorted[:, restore_order] == Q``.
    """
    array = validate_binary_matrix(Q)
    order = sorted(
        range(array.shape[1]),
        key=lambda k: tuple(array[:, k]),
        reverse=True,
    )
    sorted_array = array[:, order]
    restore_order = np.argsort(order).astype(int).tolist()
    return sorted_array, restore_order


def equivalent_up_to_column_permutation(
    Q: Sequence[Sequence[int]],
    Q_bar: Sequence[Sequence[int]],
) -> bool:
    """Return whether two equal-shaped matrices differ only by column order."""
    left = validate_binary_matrix(Q, name="Q")
    right = validate_binary_matrix(Q_bar, name="Q_bar")
    if left.shape != right.shape:
        return False
    return canonical_columns(left) == canonical_columns(right)


def has_nonempty_antichain_columns(
    Q: Sequence[Sequence[int]],
) -> bool:
    """Return whether the column supports form a nonempty antichain.

    For ``K >= 2`` this is exactly the condition certified by the manuscript's
    two-column preprocessing check: every pair of columns contains both row
    patterns ``(1, 0)`` and ``(0, 1)``.  The explicit nonemptiness check also
    handles ``K == 1``.
    """
    array = validate_binary_matrix(Q)
    columns = [array[:, k] for k in range(array.shape[1])]
    if any(not np.any(column) for column in columns):
        return False
    for k in range(len(columns)):
        for ell in range(k + 1, len(columns)):
            if np.all(columns[k] <= columns[ell]):
                return False
            if np.all(columns[ell] <= columns[k]):
                return False
    return True


def boolean_product(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return the Boolean matrix product ``left odot right``."""
    left_array = validate_binary_matrix(left, name="left")
    right_array = validate_binary_matrix(right, name="right")
    if left_array.shape[1] != right_array.shape[0]:
        raise ValueError(
            "The number of columns of left must equal the number of rows of right."
        )
    return (left_array @ right_array > 0).astype(int)


def find_boolean_factor(
    Q: Sequence[Sequence[int]],
    Q_bar: Sequence[Sequence[int]],
) -> np.ndarray | None:
    """Construct ``H`` with ``Q = Q_bar odot H``, or return ``None``.

    For each target column, the returned factor selects every column of
    ``Q_bar`` contained in that target.  If any Boolean factorization exists,
    the union of all such eligible columns is exactly the target, so this
    maximal deterministic choice is also a valid factorization.
    """
    target = validate_binary_matrix(Q, name="Q")
    candidate = validate_binary_matrix(Q_bar, name="Q_bar")
    if target.shape != candidate.shape:
        raise ValueError("Q and Q_bar must have the same shape.")

    _, K = target.shape
    factor = np.zeros((K, K), dtype=int)
    for k in range(K):
        for ell in range(K):
            if np.all(candidate[:, ell] <= target[:, k]):
                factor[ell, k] = 1
        if not np.array_equal(
            boolean_product(candidate, factor[:, [k]])[:, 0],
            target[:, k],
        ):
            return None
    return factor


def column_inclusion_holds(
    Q: Sequence[Sequence[int]],
    Q_bar: Sequence[Sequence[int]],
) -> bool:
    """Check ``Cols(Phi(Q)) <= Cols(Phi(Q_bar))`` via Proposition C2."""
    left = validate_binary_matrix(Q, name="Q")
    right = validate_binary_matrix(Q_bar, name="Q_bar")
    if left.shape != right.shape:
        raise ValueError("Q and Q_bar must have the same shape.")

    J, K = left.shape
    for support_tuple in representative_supports(left):
        support = set(support_tuple)
        if support:
            h = np.bitwise_or.reduce(right[list(support)], axis=0)
        else:
            h = np.zeros(K, dtype=int)
        for j in range(J):
            if j not in support and np.all(right[j] <= h):
                return False
    return True


# Original theorem-checking name retained for compatibility.
thm_check = column_inclusion_holds


def is_valid_counterexample(
    Q: Sequence[Sequence[int]],
    Q_bar: Sequence[Sequence[int]],
) -> bool:
    """Return whether ``Q_bar`` is a genuine algebraic non-ID witness for ``Q``."""
    left = validate_binary_matrix(Q, name="Q")
    right = validate_binary_matrix(Q_bar, name="Q_bar")
    return (
        left.shape == right.shape
        and not equivalent_up_to_column_permutation(left, right)
        and column_inclusion_holds(left, right)
    )


def is_valid_factorization_counterexample(
    Q: Sequence[Sequence[int]],
    Q_bar: Sequence[Sequence[int]],
    H: Sequence[Sequence[int]],
    *,
    require_nonzero_rows: bool = True,
    require_nonzero_columns: bool = False,
) -> bool:
    """Validate a direct certificate without enumerating latent profiles.

    The Boolean-factorization theorem makes the product identity sufficient
    for column inclusion. Non-equivalence is checked independently by sorting
    the actual columns, so no SAT auxiliary or antichain shortcut is trusted.
    """
    target = validate_binary_matrix(Q, name="Q")
    candidate = validate_binary_matrix(Q_bar, name="Q_bar")
    factor = validate_binary_matrix(H, name="H")
    if target.shape != candidate.shape:
        return False
    K = target.shape[1]
    if factor.shape != (K, K):
        return False
    if require_nonzero_rows and np.any(candidate.sum(axis=1) == 0):
        return False
    if require_nonzero_columns and np.any(candidate.sum(axis=0) == 0):
        return False
    return (
        np.array_equal(target, boolean_product(candidate, factor))
        and int(factor.sum()) >= K + 1
        and not equivalent_up_to_column_permutation(target, candidate)
    )


def item_node_set(Q: Sequence[Sequence[int]]) -> set[tuple[int, ...]]:
    """Return the distinct item-node vectors (rows) of ``Q``."""
    return {tuple(row) for row in validate_binary_matrix(Q)}
