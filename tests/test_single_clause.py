"""Check single-clause non-equivalence against independent formulations."""

import itertools
import unittest

import numpy as np
from pysat.solvers import Solver

from idQ import identify
from idQ.sat import (
    SINGLE_CLAUSE_ENCODINGS,
    build_factorization_sat_instance,
    solve_sat,
    solve_sat_profile,
)
from idQ.utils import (
    boolean_product,
    has_nonempty_antichain_columns,
    is_valid_factorization_counterexample,
    lex_sort_columns,
)


class SingleClauseTests(unittest.TestCase):
    def assert_valid(self, Q, outcome, *, nonzero_columns=False):
        if outcome.satisfiable:
            self.assertTrue(is_valid_factorization_counterexample(
                Q, outcome.counterexample, outcome.factorization,
                require_nonzero_columns=nonzero_columns,
            ))
            np.testing.assert_array_equal(
                boolean_product(outcome.counterexample, outcome.factorization), Q,
            )

    def test_agreement_on_all_nonzero_distinct_row_sets_through_k3(self):
        """Profile SAT and its exhaustive oracle do not use the new shortcut."""
        checked = 0
        for K in range(1, 4):
            rows = [row for row in itertools.product((0, 1), repeat=K) if any(row)]
            for mask in range(1, 1 << len(rows)):
                Q = np.array([row for i, row in enumerate(rows) if mask & (1 << i)])
                if not has_nonempty_antichain_columns(Q):
                    continue
                expected = solve_sat_profile(Q).satisfiable
                self.assertEqual(solve_sat(Q).satisfiable, expected)
                if Q.size <= 12:
                    self.assertEqual(solve_sat_profile(Q, "oracle").satisfiable, expected)
                for encoding in SINGLE_CLAUSE_ENCODINGS:
                    result = solve_sat(Q, cardinality_encoding=encoding)
                    self.assertEqual(result.satisfiable, expected, msg=Q.tolist())
                    self.assertEqual(result.cardinality_encoding, encoding)
                    self.assert_valid(Q, result)
                checked += 1
        self.assertGreater(checked, 10)

    def test_original_column_order_and_optional_constraints(self):
        sat = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],
                        [1, 0, 1, 1], [0, 1, 1, 1]])
        unsat = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0],
                          [1, 0, 0, 1], [0, 1, 0, 1]])
        for matrix, expected in ((sat, True), (unsat, False)):
            for order in itertools.permutations(range(4)):
                Q = matrix[::-1, order]
                for encoding in SINGLE_CLAUSE_ENCODINGS:
                    for maximal, nonzero in ((False, False), (True, True)):
                        result = solve_sat(
                            Q, cardinality_encoding=encoding,
                            maximal_candidate=maximal, require_nonzero_columns=nonzero,
                        )
                        self.assertEqual(result.satisfiable, expected)
                        self.assert_valid(Q, result, nonzero_columns=nonzero)

    def test_equivalent_factorization_is_blocked_in_original_coordinates(self):
        Q = np.array([[0, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        sorted_Q, restore = lex_sort_columns(Q)
        self.assertFalse(np.array_equal(Q, sorted_Q))
        P = np.zeros((5, 5), dtype=int)
        P[restore, np.arange(5)] = 1
        np.testing.assert_array_equal(boolean_product(sorted_Q, P), Q)
        for name in SINGLE_CLAUSE_ENCODINGS:
            encoding = build_factorization_sat_instance(Q, cardinality_encoding=name)
            units = [
                variable if bit else -variable
                for variables, bits in ((encoding.decision_variables, sorted_Q),
                                        (encoding.factor_variables, P))
                for row_variables, row_bits in zip(variables, bits)
                for variable, bit in zip(row_variables, row_bits)
            ]
            with Solver(name="glucose42", bootstrap_with=encoding.clauses) as solver:
                self.assertFalse(solver.solve(assumptions=units))
            result = solve_sat(Q, cardinality_encoding=name)
            self.assertTrue(result.satisfiable)
            self.assert_valid(Q, result)

    def test_only_cardinality_block_changes_and_no_auxiliaries_added(self):
        for K in (2, 3, 10):
            Q = np.eye(K, dtype=int)[:, ::-1]
            original = build_factorization_sat_instance(Q, cardinality_encoding="prefix")
            cardinality_clauses = 5 * K * K - 7 * K + 1
            common = original.clauses[:-cardinality_clauses]
            for name in SINGLE_CLAUSE_ENCODINGS:
                alternative = build_factorization_sat_instance(Q, cardinality_encoding=name)
                self.assertEqual(alternative.clauses[:len(common)], common)
                self.assertEqual(len(alternative.clauses) - len(common), K + 1)
                self.assertEqual(
                    original.n_variables - alternative.n_variables, K * (2 * K - 3),
                )
                expected_length = Q.size if name == "exclude_x" else K * (K - 1)
                self.assertEqual(len(alternative.clauses[-1]), expected_length)

    def test_symmetry_guard_and_single_column_boundary(self):
        for name in SINGLE_CLAUSE_ENCODINGS:
            with self.assertRaisesRegex(ValueError, "requires symmetry_breaking=True"):
                build_factorization_sat_instance(
                    np.eye(3, dtype=int), cardinality_encoding=name,
                    symmetry_breaking=False,
                )
            self.assertFalse(solve_sat([[1]], cardinality_encoding=name).satisfiable)
            self.assertTrue(identify(np.eye(3, dtype=int), cardinality_encoding=name).identifiable)

    def test_default_excludes_equivalence_and_returns_original_coordinate_witness(self):
        Q = np.array([[0, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        sorted_Q, _ = lex_sort_columns(Q)
        encoding = build_factorization_sat_instance(Q)
        self.assertEqual(encoding.cardinality_encoding, "exclude_x")
        fixed_candidate = [
            variable if bit else -variable
            for variables, bits in zip(encoding.decision_variables, sorted_Q)
            for variable, bit in zip(variables, bits)
        ]
        with Solver(name="glucose42", bootstrap_with=encoding.clauses) as solver:
            self.assertFalse(solver.solve(assumptions=fixed_candidate))
        outcome = solve_sat(Q)
        self.assertEqual(outcome.cardinality_encoding, "exclude_x")
        self.assertEqual(outcome.solver_name, "glucose42")
        self.assertTrue(outcome.satisfiable)
        self.assert_valid(Q, outcome)
        result = identify(Q)
        self.assertEqual(result.sat_cardinality_encoding, "exclude_x")
        self.assertFalse(result.identifiable)
        self.assertTrue(is_valid_factorization_counterexample(
            Q, result.counterexample, result.factorization,
        ))
        self.assertFalse(solve_sat(np.eye(4, dtype=int)).satisfiable)

    def test_public_api_lifts_factorization_after_basis_reduction(self):
        Q = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],
                      [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        for name in SINGLE_CLAUSE_ENCODINGS:
            result = identify(Q, cardinality_encoding=name)
            self.assertFalse(result.identifiable)
            self.assertEqual(result.sat_cardinality_encoding, name)
            self.assertTrue(is_valid_factorization_counterexample(
                Q, result.counterexample, result.factorization,
            ))


if __name__ == "__main__":
    unittest.main()
