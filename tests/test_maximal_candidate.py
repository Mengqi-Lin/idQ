"""Check the optional maximal-candidate restriction against the exact CNF."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np



try:
    from pysat.solvers import Solver
except ImportError:
    Solver = None

from idQ.sat import build_factorization_sat_instance, solve_sat
from idQ.utils import has_nonempty_antichain_columns, is_valid_factorization_counterexample


class MaximalCandidateTests(unittest.TestCase):
    def assert_maximal(self, Q, outcome, *, nonzero_columns=False, sorted_columns=True):
        if not outcome.satisfiable:
            return
        candidate = outcome.counterexample
        factor = outcome.factorization
        self.assertTrue(is_valid_factorization_counterexample(
            Q, candidate, factor, require_nonzero_columns=nonzero_columns,
        ))
        expected = ((Q == 0).astype(int) @ factor.T == 0).astype(int)
        np.testing.assert_array_equal(candidate, expected)
        if sorted_columns:
            columns = tuple(tuple(column) for column in candidate.T)
            self.assertEqual(columns, tuple(sorted(columns, reverse=True)))

    def test_adds_one_clause_per_candidate_entry_with_correct_signs(self):
        Q = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        old = build_factorization_sat_instance(Q)
        new = build_factorization_sat_instance(Q, maximal_candidate=True)
        self.assertEqual(new.n_variables, old.n_variables)
        self.assertEqual(len(new.clauses) - len(old.clauses), Q.size)
        clauses = set(new.clauses)
        for j in range(Q.shape[0]):
            for ell in range(Q.shape[1]):
                expected = (new.decision_variables[j][ell],) + tuple(
                    new.factor_variables[ell][k]
                    for k in range(Q.shape[1]) if not Q[j, k]
                )
                self.assertIn(expected, clauses)

    @unittest.skipIf(Solver is None, "python-sat is unavailable")
    def test_all_postpair_distinct_nonzero_row_sets_through_k3(self):
        checked = 0
        for K in range(1, 4):
            row_types = [row for row in itertools.product((0, 1), repeat=K) if any(row)]
            for mask in range(1, 1 << len(row_types)):
                Q = np.asarray([
                    row for i, row in enumerate(row_types) if mask & (1 << i)
                ], dtype=int)
                if not has_nonempty_antichain_columns(Q):
                    continue
                for nonzero_columns in (False, True):
                    ordinary = solve_sat(Q, "cadical195", require_nonzero_columns=nonzero_columns)
                    maximal = solve_sat(Q, "cadical195", require_nonzero_columns=nonzero_columns,
                                        maximal_candidate=True)
                    self.assertEqual(ordinary.satisfiable, maximal.satisfiable)
                    self.assert_maximal(Q, maximal, nonzero_columns=nonzero_columns)
                checked += 1
        self.assertGreater(checked, 10)

    @unittest.skipIf(Solver is None, "python-sat is unavailable")
    def test_identity_boundaries_and_no_symmetry_breaking(self):
        for K in range(1, 6):
            for maximal in (False, True):
                outcome = solve_sat(np.eye(K, dtype=int), "cadical195", maximal_candidate=maximal)
                self.assertFalse(outcome.satisfiable)
        Q = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        for solver in ("cadical195", "kissat404"):
            for symmetry in (False, True):
                ordinary = solve_sat(Q, solver, symmetry_breaking=symmetry,
                                     cardinality_encoding="prefix")
                maximal = solve_sat(Q, solver, symmetry_breaking=symmetry,
                                    maximal_candidate=True, cardinality_encoding="prefix")
                self.assertTrue(ordinary.satisfiable)
                self.assertEqual(ordinary.satisfiable, maximal.satisfiable)
                self.assert_maximal(Q, maximal, sorted_columns=symmetry)

    def test_oracle_returns_maximal_witness_too(self):
        Q = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        maximal = solve_sat(Q, "oracle", maximal_candidate=True, require_nonzero_columns=True)
        self.assertTrue(maximal.satisfiable)
        self.assert_maximal(Q, maximal, nonzero_columns=True)

    @unittest.skipIf(Solver is None, "python-sat is unavailable")
    def test_public_identify_routes_option_and_lifts_certificate(self):
        from idQ import identify
        Q = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],
                        [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        result = identify(Q, solver_name="glucose42", maximal_candidate=True)
        self.assertFalse(result.identifiable)
        self.assertTrue(is_valid_factorization_counterexample(
            Q, result.counterexample, result.factorization,
        ))


if __name__ == "__main__":
    unittest.main()
