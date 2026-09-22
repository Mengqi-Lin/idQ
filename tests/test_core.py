from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np



from idQ.basis import get_basis, reconstruct_from_basis, reduce_to_basis
from idQ import identify
from idQ.sat import (
    VariablePool,
    _add_lex_nonincreasing,
    _add_nonpermutation_factor,
    build_factorization_sat_instance,
    build_profile_sat_instance,
    build_sat_instance,
    solve_sat,
    solve_sat_profile,
)
from idQ.utils import (
    boolean_product,
    canonical_columns,
    equivalent_up_to_column_permutation,
    has_nonempty_antichain_columns,
    is_valid_counterexample,
    is_valid_factorization_counterexample,
    lex_sort_columns,
    response_columns,
    validate_binary_matrix,
)


def dpll(clauses, assignment=None):
    """Tiny test-only DPLL solver with unit and pure-literal propagation."""
    assignment = {} if assignment is None else dict(assignment)
    clauses = [tuple(clause) for clause in clauses]

    while True:
        reduced = []
        units = []
        for clause in clauses:
            pending = []
            satisfied = False
            for literal in clause:
                variable = abs(literal)
                if variable in assignment:
                    if assignment[variable] == (literal > 0):
                        satisfied = True
                        break
                else:
                    pending.append(literal)
            if satisfied:
                continue
            if not pending:
                return None
            if len(pending) == 1:
                units.append(pending[0])
            reduced.append(tuple(pending))
        if not reduced:
            return assignment

        changed = False
        for literal in units:
            variable = abs(literal)
            value = literal > 0
            if variable in assignment and assignment[variable] != value:
                return None
            if variable not in assignment:
                assignment[variable] = value
                changed = True
        if changed:
            clauses = reduced
            continue

        signs = {}
        for clause in reduced:
            for literal in clause:
                signs.setdefault(abs(literal), set()).add(literal > 0)
        pure = next(
            ((variable, next(iter(values))) for variable, values in signs.items()
             if len(values) == 1),
            None,
        )
        if pure is not None:
            assignment[pure[0]] = pure[1]
            clauses = reduced
            continue
        clauses = reduced
        break

    literal = min(clauses, key=len)[0]
    variable = abs(literal)
    for value in (literal > 0, literal < 0):
        solution = dpll(clauses, {**assignment, variable: value})
        if solution is not None:
            return solution
    return None


def brute_identifiable(Q):
    Q = np.asarray(Q, dtype=int)
    J, K = Q.shape
    target_columns = response_columns(Q)
    target_canonical = canonical_columns(Q)
    nonzero_rows = [
        row for row in itertools.product((0, 1), repeat=K)
        if any(row)
    ]
    for rows in itertools.product(nonzero_rows, repeat=J):
        candidate = np.asarray(rows, dtype=int)
        if canonical_columns(candidate) == target_canonical:
            continue
        if target_columns <= response_columns(candidate):
            return False, candidate
    return True, None


def antichain_matrices(J, K):
    """Yield one column-canonical matrix for every nonempty antichain."""
    column_types = list(itertools.product((0, 1), repeat=J))
    for columns in itertools.combinations(column_types, K):
        Q = np.asarray(columns, dtype=int).T
        if np.any(Q.sum(axis=1) == 0):
            continue
        if has_nonempty_antichain_columns(Q):
            yield Q


class BasisTests(unittest.TestCase):
    def test_all_zero_basis_preserves_k(self):
        Q = np.zeros((3, 4), dtype=int)
        reduction = reduce_to_basis(Q)
        self.assertEqual(reduction.basis.shape, (0, 4))
        reconstructed = reconstruct_from_basis(
            reduction.basis,
            reduction.original_to_basis,
        )
        np.testing.assert_array_equal(reconstructed, Q)

    def test_generated_and_duplicate_rows_reconstruct(self):
        Q = np.asarray([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        reduction = reduce_to_basis(Q)
        self.assertEqual(reduction.basis.shape, (2, 3))
        np.testing.assert_array_equal(
            reconstruct_from_basis(reduction.basis, reduction.original_to_basis),
            Q,
        )


class AlgorithmRegressionTests(unittest.TestCase):
    def test_k1_identity_is_identifiable(self):
        result = identify([[1]], solver_name="oracle")
        self.assertTrue(result.identifiable)
        self.assertEqual(result.branch, 4)

    def test_triangle_precheck_does_not_fabricate_witness(self):
        Q = np.asarray([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ])
        result = identify(Q, solver_name="oracle")
        self.assertFalse(result.identifiable)
        self.assertEqual(result.branch, 3)
        self.assertIsNone(result.counterexample)
        self.assertTrue(is_valid_counterexample(Q, np.eye(3, dtype=int)))

    def test_identifiable_noncomplete_no_pure_node(self):
        Q = np.asarray([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ])
        result = identify(Q, solver_name="oracle")
        self.assertTrue(result.identifiable)
        self.assertEqual(result.branch, 6)

    def test_sat_counterexample_is_valid(self):
        Q = np.asarray([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ])
        result = identify(Q, solver_name="oracle")
        self.assertFalse(result.identifiable)
        self.assertEqual(result.branch, 5)
        self.assertIsNotNone(result.counterexample)
        self.assertIsNotNone(result.factorization)
        self.assertTrue(is_valid_counterexample(Q, result.counterexample))
        self.assertTrue(
            is_valid_factorization_counterexample(
                Q,
                result.counterexample,
                result.factorization,
            )
        )

    def test_factorization_certificate_survives_basis_lifting(self):
        Q = np.asarray([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],  # duplicate basis row
            [1, 1, 1, 0],  # Boolean union of two basis rows
            [0, 0, 0, 0],  # legacy-degenerate row
        ])
        result = identify(Q, solver_name="oracle")
        self.assertFalse(result.identifiable)
        self.assertEqual(result.branch, 5)
        self.assertEqual(result.sat_formulation, "boolean_factorization")
        self.assertIsNotNone(result.counterexample)
        self.assertIsNotNone(result.factorization)
        self.assertTrue(
            is_valid_factorization_counterexample(
                Q,
                result.counterexample,
                result.factorization,
                require_nonzero_rows=False,
            )
        )

    def test_column_permutation_invariance(self):
        Q = np.asarray([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ])
        expected = identify(Q, solver_name="oracle").identifiable
        for permutation in itertools.permutations(range(Q.shape[1])):
            actual = identify(Q[:, permutation], solver_name="oracle").identifiable
            self.assertEqual(actual, expected)

    def test_input_validation(self):
        invalid = [
            [1, 0, 1],
            np.asarray([[0.5, 1.0]]),
            np.asarray([[0, -1]]),
            np.empty((0, 2), dtype=int),
        ]
        for Q in invalid:
            with self.subTest(Q=Q):
                with self.assertRaises(ValueError):
                    validate_binary_matrix(Q)

    def test_exhaustive_small_oracle(self):
        for J, K in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
            nonzero_rows = [
                row for row in itertools.product((0, 1), repeat=K)
                if any(row)
            ]
            for rows in itertools.product(nonzero_rows, repeat=J):
                Q = np.asarray(rows, dtype=int)
                expected, _ = brute_identifiable(Q)
                actual = identify(Q, solver_name="oracle").identifiable
                self.assertEqual(actual, expected, msg=f"Q={Q.tolist()}")


class EncodingTests(unittest.TestCase):
    def test_weak_lex_encoding_truth_table(self):
        for length in range(1, 6):
            pool = VariablePool()
            X = [
                [pool.id(("x", j, k)) for k in range(2)]
                for j in range(length)
            ]
            clauses = []
            _add_lex_nonincreasing(clauses, pool, X)
            for left in itertools.product((0, 1), repeat=length):
                for right in itertools.product((0, 1), repeat=length):
                    fixed = list(clauses)
                    for j in range(length):
                        fixed.append([X[j][0] if left[j] else -X[j][0]])
                        fixed.append([X[j][1] if right[j] else -X[j][1]])
                    satisfiable = dpll(fixed) is not None
                    self.assertEqual(
                        satisfiable,
                        tuple(left) >= tuple(right),
                        msg=f"left={left}, right={right}",
                    )

    def test_profile_cnf_matches_direct_oracle(self):
        for J, K in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
            nonzero_rows = [
                row for row in itertools.product((0, 1), repeat=K)
                if any(row)
            ]
            for rows in itertools.product(nonzero_rows, repeat=J):
                Q = np.asarray(rows, dtype=int)
                Q_sorted, _ = lex_sort_columns(Q)
                encoding = build_profile_sat_instance(Q_sorted)
                cnf_satisfiable = dpll(encoding.clauses) is not None
                oracle_satisfiable = solve_sat_profile(Q, "oracle").satisfiable
                self.assertEqual(
                    cnf_satisfiable,
                    oracle_satisfiable,
                    msg=f"Q={Q.tolist()}",
                )

    def test_h_nonpermutation_encoding_truth_table(self):
        for K in range(1, 5):
            pool = VariablePool()
            H = [
                [pool.id(("h", ell, k)) for k in range(K)]
                for ell in range(K)
            ]
            clauses = []
            _add_nonpermutation_factor(clauses, pool, H)

            for values in itertools.product((0, 1), repeat=K * K):
                matrix = np.asarray(values, dtype=int).reshape(K, K)
                fixed = list(clauses)
                for ell in range(K):
                    for k in range(K):
                        variable = H[ell][k]
                        fixed.append([variable if matrix[ell, k] else -variable])
                satisfiable = dpll(fixed) is not None
                expected = (
                    np.all(matrix.sum(axis=0) >= 1)
                    and int(matrix.sum()) >= K + 1
                )
                self.assertEqual(
                    satisfiable,
                    expected,
                    msg=f"K={K}, H={matrix.tolist()}",
                )

    def test_factorization_builder_requires_post_pair_branch(self):
        Q = np.asarray([[1, 1], [0, 1]])
        self.assertFalse(has_nonempty_antichain_columns(Q))
        with self.assertRaises(ValueError):
            build_sat_instance(Q)

    def test_factorization_and_profile_encodings_agree_through_k4(self):
        checked = 0
        for J in range(1, 5):
            for K in range(1, 5):
                for Q in antichain_matrices(J, K):
                    expected_identifiable, _ = brute_identifiable(Q)

                    factor_encoding = build_factorization_sat_instance(Q)
                    factor_satisfiable = dpll(factor_encoding.clauses) is not None

                    Q_sorted, _ = lex_sort_columns(Q)
                    profile_encoding = build_profile_sat_instance(Q_sorted)
                    profile_satisfiable = dpll(profile_encoding.clauses) is not None

                    self.assertEqual(
                        factor_satisfiable,
                        not expected_identifiable,
                        msg=f"factor Q={Q.tolist()}",
                    )
                    self.assertEqual(
                        profile_satisfiable,
                        not expected_identifiable,
                        msg=f"profile Q={Q.tolist()}",
                    )
                    self.assertEqual(
                        factor_satisfiable,
                        profile_satisfiable,
                        msg=f"Q={Q.tolist()}",
                    )
                    checked += 1

        self.assertEqual(checked, 119)

    def test_antichain_factor_equivalence_exhaustive(self):
        checked = 0
        for J, K in [(2, 2), (3, 2), (3, 3)]:
            nonzero_rows = [
                row for row in itertools.product((0, 1), repeat=K)
                if any(row)
            ]
            for rows in itertools.product(nonzero_rows, repeat=J):
                Q_bar = np.asarray(rows, dtype=int)
                for values in itertools.product((0, 1), repeat=K * K):
                    H = np.asarray(values, dtype=int).reshape(K, K)
                    Q = boolean_product(Q_bar, H)
                    if not has_nonempty_antichain_columns(Q):
                        continue
                    is_permutation = (
                        int(H.sum()) == K
                        and np.all(H.sum(axis=0) == 1)
                        and np.all(H.sum(axis=1) == 1)
                    )
                    self.assertEqual(
                        equivalent_up_to_column_permutation(Q_bar, Q),
                        is_permutation,
                    )
                    self.assertEqual(not is_permutation, int(H.sum()) >= K + 1)
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
