"""Basis-submatrix reduction and reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .utils import validate_binary_matrix


@dataclass(frozen=True)
class BasisReduction:
    """All deterministic maps created while reducing a Q matrix to its basis."""

    basis: np.ndarray
    original_to_basis: tuple[tuple[int, ...], ...]
    original_indices_for_basis: tuple[tuple[int, ...], ...]
    unique_rows: np.ndarray
    original_to_unique: tuple[int, ...]
    unique_to_basis: tuple[tuple[int, ...], ...]


def _bitwise_or(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a | b for a, b in zip(left, right))


def reduce_to_basis(Q: Sequence[Sequence[int]]) -> BasisReduction:
    """Compute the manuscript's basis submatrix and reconstruction maps.

    Distinct nonzero rows are processed in increasing lexicographic order.
    For each row, take the OR of all previously selected basis rows contained
    componentwise in it.  Equality means that the row is generated and can be
    omitted; otherwise it becomes a new basis row.  This is exact because every
    row in an OR representation must be contained in the target row and thus
    precedes it unless equal to it.

    At most J basis rows are checked for each of at most J distinct rows, each
    check costing O(K).  Hence reduction costs O(J**2 * K) elementary work,
    including sorting, and never enumerates the possibly exponential OR-closure.
    Reconstruction maps may use every contained basis row, so their total size
    is at most O(J**2).
    """
    array = validate_binary_matrix(Q)
    _, K = array.shape
    rows = [tuple(int(value) for value in row) for row in array]
    zero = (0,) * K

    row_to_original: dict[tuple[int, ...], list[int]] = {}
    for j, row in enumerate(rows):
        row_to_original.setdefault(row, []).append(j)

    unique = sorted(row for row in row_to_original if row != zero)
    unique_index = {row: index for index, row in enumerate(unique)}

    # Store representations only for input rows, never for the whole closure.
    # A zero row is represented by the empty OR.
    row_representation: dict[tuple[int, ...], tuple[int, ...]] = {zero: ()}
    basis_rows: list[tuple[int, ...]] = []
    original_indices_for_basis: list[tuple[int, ...]] = []

    for row in unique:
        contained_indices: list[int] = []
        generated = zero
        for basis_index, basis_row in enumerate(basis_rows):
            if all(a <= b for a, b in zip(basis_row, row)):
                contained_indices.append(basis_index)
                generated = _bitwise_or(generated, basis_row)

        if generated == row:
            row_representation[row] = tuple(contained_indices)
            continue

        basis_index = len(basis_rows)
        basis_rows.append(row)
        original_indices_for_basis.append(tuple(row_to_original[row]))
        row_representation[row] = (basis_index,)

    basis = np.asarray(basis_rows, dtype=int).reshape((-1, K))
    unique_rows = np.asarray(unique, dtype=int).reshape((-1, K))
    original_to_basis = tuple(row_representation[row] for row in rows)
    original_to_unique = tuple(
        -1 if row == zero else unique_index[row]
        for row in rows
    )
    unique_to_basis = tuple(row_representation[row] for row in unique)

    return BasisReduction(
        basis=basis,
        original_to_basis=original_to_basis,
        original_indices_for_basis=tuple(original_indices_for_basis),
        unique_rows=unique_rows,
        original_to_unique=original_to_unique,
        unique_to_basis=unique_to_basis,
    )


def get_basis(Q: Sequence[Sequence[int]]):
    """Backward-compatible six-tuple interface used by the original scripts."""
    reduction = reduce_to_basis(Q)
    return (
        reduction.basis,
        [list(indices) for indices in reduction.original_to_basis],
        [list(indices) for indices in reduction.original_indices_for_basis],
        reduction.unique_rows,
        list(reduction.original_to_unique),
        [list(indices) for indices in reduction.unique_to_basis],
    )


def _validate_reduced_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0/1 entries.")
    return array.astype(int, copy=True)


def reconstruct_from_basis(
    Q_basis: np.ndarray,
    original_to_basis: Sequence[Sequence[int]],
) -> np.ndarray:
    """Lift a basis matrix to the original row structure using stored OR maps."""
    basis = _validate_reduced_matrix(Q_basis, name="Q_basis")
    n_basis, K = basis.shape
    reconstructed = np.zeros((len(original_to_basis), K), dtype=int)

    for row_index, subset in enumerate(original_to_basis):
        indices = tuple(int(index) for index in subset)
        if any(index < 0 or index >= n_basis for index in indices):
            raise IndexError("A basis reconstruction index is out of range.")
        if indices:
            reconstructed[row_index] = np.bitwise_or.reduce(
                basis[list(indices)],
                axis=0,
            )
    return reconstructed


def get_Q_from_Qbasis(Q_basis, basis_to_original):
    """Compatibility wrapper for :func:`reconstruct_from_basis`."""
    return reconstruct_from_basis(Q_basis, basis_to_original)


def get_Qunique_from_Qbasis(Q_basis, basis_to_unique):
    """Reconstruct the distinct nonzero rows from a basis matrix."""
    return reconstruct_from_basis(Q_basis, basis_to_unique)


def get_Q_from_Qunique(Q_unique, unique_to_original):
    """Reconstruct original row order from distinct rows and ``-1`` zero markers."""
    unique = _validate_reduced_matrix(Q_unique, name="Q_unique")
    n_unique, K = unique.shape
    reconstructed = np.zeros((len(unique_to_original), K), dtype=int)
    for row_index, unique_index in enumerate(unique_to_original):
        index = int(unique_index)
        if index == -1:
            continue
        if index < 0 or index >= n_unique:
            raise IndexError("A unique-row reconstruction index is out of range.")
        reconstructed[row_index] = unique[index]
    return reconstructed
