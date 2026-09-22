"""Independent correctness checks for polynomial basis reduction."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np



from idQ.basis import (
    get_Q_from_Qunique,
    reconstruct_from_basis,
    reduce_to_basis,
)


def exhaustive_closure_basis(Q):
    """Small-matrix oracle: explicitly enumerate every OR of selected rows."""
    K = Q.shape[1]
    zero = (0,) * K
    closure = {zero}
    basis = []
    for row in sorted({tuple(int(value) for value in row) for row in Q}):
        if row in closure:
            continue
        basis.append(row)
        closure.update(
            tuple(a | b for a, b in zip(previous, row))
            for previous in tuple(closure)
        )
    return np.asarray(basis, dtype=int).reshape((-1, K))


class PolynomialBasisTests(unittest.TestCase):
    def assert_all_maps_reconstruct(self, Q, reduction):
        np.testing.assert_array_equal(
            reconstruct_from_basis(reduction.basis, reduction.original_to_basis),
            Q,
        )
        np.testing.assert_array_equal(
            reconstruct_from_basis(reduction.basis, reduction.unique_to_basis),
            reduction.unique_rows,
        )
        np.testing.assert_array_equal(
            get_Q_from_Qunique(reduction.unique_rows, reduction.original_to_unique),
            Q,
        )
        for index, original_indices in enumerate(reduction.original_indices_for_basis):
            expected = tuple(
                j for j, row in enumerate(Q)
                if np.array_equal(row, reduction.basis[index])
            )
            self.assertEqual(original_indices, expected)
            for j in original_indices:
                self.assertEqual(reduction.original_to_basis[j], (index,))

    def test_all_682_binary_matrices_through_three_by_three(self):
        checked = 0
        for J, K in itertools.product(range(1, 4), repeat=2):
            for bits in itertools.product((0, 1), repeat=J * K):
                Q = np.asarray(bits, dtype=int).reshape(J, K)
                reduction = reduce_to_basis(Q)
                np.testing.assert_array_equal(
                    reduction.basis, exhaustive_closure_basis(Q),
                )
                self.assert_all_maps_reconstruct(Q, reduction)
                checked += 1
        self.assertEqual(checked, 682)

    def test_all_distinct_row_sets_through_three_columns(self):
        # These include many more than three rows, testing chains of omissions.
        checked = 0
        for K in range(1, 4):
            row_types = list(itertools.product((0, 1), repeat=K))
            for mask in range(1, 1 << len(row_types)):
                Q = np.asarray([
                    row for index, row in enumerate(row_types)
                    if mask & (1 << index)
                ], dtype=int)
                reduction = reduce_to_basis(Q)
                np.testing.assert_array_equal(
                    reduction.basis, exhaustive_closure_basis(Q),
                )
                self.assert_all_maps_reconstruct(Q, reduction)
                checked += 1
        self.assertEqual(checked, 273)

    def test_identity_with_one_hundred_columns_does_not_enumerate_closure(self):
        # An implementation enumerating the OR-closure would attempt 2**100
        # vectors.  The basis is simply the lexicographically sorted identity.
        Q = np.eye(100, dtype=int)
        reduction = reduce_to_basis(Q)
        np.testing.assert_array_equal(reduction.basis, Q[::-1])
        self.assert_all_maps_reconstruct(Q, reduction)

    def test_large_identity_with_generated_rows_and_duplicates(self):
        identity = np.eye(100, dtype=int)
        Q = np.vstack((
            np.ones((1, 100), dtype=int),
            identity,
            identity[::10],
            np.zeros((2, 100), dtype=int),
        ))
        reduction = reduce_to_basis(Q)
        np.testing.assert_array_equal(reduction.basis, identity[::-1])
        self.assert_all_maps_reconstruct(Q, reduction)


if __name__ == "__main__":
    unittest.main()
