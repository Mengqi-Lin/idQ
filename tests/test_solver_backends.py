"""Real backend checks and preservation of unknown solver outcomes."""
import unittest
from unittest.mock import patch

import numpy as np

from idQ.sat import solve_sat


class SolverBackendTests(unittest.TestCase):
    def test_available_backends_agree_on_sat_and_unsat(self):
        try:
            import pysat  # noqa: F401
        except ImportError:
            self.skipTest("python-sat is not installed")
        alternative = np.array([
            [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],
            [1, 0, 1, 1], [0, 1, 1, 1],
        ])
        for name in ("cadical195", "cadical300", "kissat404", "glucose42", "maplechrono"):
            with self.subTest(solver=name):
                # solve_sat also independently validates every SAT certificate.
                self.assertTrue(solve_sat(alternative, solver_name=name).satisfiable)
                self.assertFalse(solve_sat(np.eye(4, dtype=int), solver_name=name).satisfiable)

    def test_unknown_never_becomes_unsat(self):
        try:
            import pysat.solvers
        except ImportError:
            self.skipTest("python-sat is not installed")
        with patch("pysat.solvers.Solver") as factory:
            factory.return_value.__enter__.return_value.solve.return_value = None
            with self.assertRaises(RuntimeError) as caught:
                solve_sat(np.eye(2, dtype=int))
            self.assertIn("UNKNOWN", str(caught.exception.__cause__))

    def test_factor_certificate_never_enumerates_profiles(self):
        from idQ.utils import is_valid_factorization_counterexample
        candidate = np.array([[1, 1, 0], [0, 0, 1]])
        factor = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 1]])
        target = (candidate @ factor > 0).astype(int)
        with patch("idQ.utils.representative_supports", side_effect=AssertionError("enumeration")):
            self.assertTrue(is_valid_factorization_counterexample(target, candidate, factor))
            self.assertFalse(is_valid_factorization_counterexample(target, target, np.eye(3, dtype=int)))


if __name__ == "__main__":
    unittest.main()
