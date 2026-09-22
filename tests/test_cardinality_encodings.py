"""Independent truth-table and certificate checks for selectable encodings."""

import itertools
import unittest

import numpy as np
from pysat.solvers import Solver

from idQ import identify
from idQ.sat import (
    CARDINALITY_ENCODINGS, VariablePool, _add_nonpermutation_factor, solve_sat,
)
from idQ.utils import is_valid_factorization_counterexample


class CardinalityEncodingTests(unittest.TestCase):
    def test_all_fixed_h_through_k3_against_integer_count(self):
        for encoding in CARDINALITY_ENCODINGS:
            for K in range(1, 4):
                pool = VariablePool()
                H = [[pool.id(('h', ell, k)) for k in range(K)] for ell in range(K)]
                flat = [v for row in H for v in row]
                clauses = []
                _add_nonpermutation_factor(clauses, pool, H, encoding=encoding)
                with Solver(name='glucose42', bootstrap_with=clauses) as solver:
                    for bits in itertools.product((0, 1), repeat=K*K):
                        matrix = np.array(bits).reshape(K, K)
                        expected = bool(np.all(matrix.sum(axis=0) > 0) and matrix.sum() >= K+1)
                        assumptions = [v if bit else -v for v, bit in zip(flat, bits)]
                        self.assertEqual(solver.solve(assumptions=assumptions), expected,
                                         msg=f'{encoding}, K={K}, H={matrix.tolist()}')

    def test_every_encoding_agrees_and_returns_valid_certificates(self):
        sat = np.array([[1,1,0,0], [1,0,1,0], [0,1,0,1], [1,0,1,1], [0,1,1,1]])
        unsat = np.array([[1,1,0,0], [1,0,1,0], [0,1,1,0], [1,0,0,1], [0,1,0,1]])
        for encoding in CARDINALITY_ENCODINGS:
            for symmetry in (False, True):
                for maximal in (False, True):
                    for Q, expected in ((sat, True), (unsat, False)):
                        outcome = solve_sat(Q, cardinality_encoding=encoding,
                                            symmetry_breaking=symmetry,
                                            maximal_candidate=maximal)
                        self.assertEqual(outcome.satisfiable, expected)
                        self.assertEqual(outcome.cardinality_encoding, encoding)
                        if expected:
                            self.assertTrue(is_valid_factorization_counterexample(
                                Q, outcome.counterexample, outcome.factorization))

    def test_external_encoders_reserve_auxiliary_identifiers(self):
        for encoding in CARDINALITY_ENCODINGS:
            pool = VariablePool()
            H = [[pool.id(('h', ell, k)) for k in range(4)] for ell in range(4)]
            clauses = []
            _add_nonpermutation_factor(clauses, pool, H, encoding=encoding)
            used = {abs(v) for clause in clauses for v in clause}
            later = pool.id(('later_constraint', 0))
            self.assertNotIn(later, used)
            self.assertGreater(later, max(used))

    def test_prefix_literal_count_stays_quadratic(self):
        for K in (5, 10, 20, 50):
            pool = VariablePool()
            H = [[pool.id(('h', ell, k)) for k in range(K)] for ell in range(K)]
            clauses = []
            _add_nonpermutation_factor(clauses, pool, H, encoding='prefix')
            self.assertLessEqual(sum(map(len, clauses)), 16*K*K)

    def test_public_api_validates_choice_before_preprocessing(self):
        for invalid in ('typo', None, 1):
            with self.assertRaises(ValueError):
                identify(np.eye(3, dtype=int), cardinality_encoding=invalid)
        Q = np.array([[1,1,0,0], [1,0,1,0], [0,1,0,1], [1,0,1,1], [0,1,1,1]])
        result = identify(Q, cardinality_encoding='seqcounter')
        self.assertEqual(result.sat_cardinality_encoding, 'seqcounter')
        self.assertEqual(result.solver_name, 'glucose42')
        self.assertFalse(result.identifiable)


if __name__ == '__main__':
    unittest.main()
