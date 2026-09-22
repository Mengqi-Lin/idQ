from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd



from idQ.experiments.bernoulli import run_expr as run_bernoulli
from idQ.experiments.common import (
    SCHEMA_VERSION,
    actual_row_size_metadata,
    identifiability_expr,
    make_rng,
    sample_bernoulli_q,
    sample_row_sparse_q,
    run_design_expr,
    source_provenance,
)
from idQ.experiments.row_sparsity import run_expr as run_row_sparse
from idQ.sat import normalize_solver_name
from idQ.experiments.summary import (
    _validate_one_file,
    read_row_sparsity_csvs,
    summarize,
)


class SamplerTests(unittest.TestCase):
    def test_legacy_rng_reproduces_original_stream(self):
        expected_rng = np.random.RandomState(17)
        expected = expected_rng.binomial(1, 0.3, size=(5, 4))
        actual = sample_bernoulli_q(
            5,
            4,
            0.3,
            make_rng(17, "legacy"),
            condition_nonzero_rows=False,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_conditioned_bernoulli_has_no_zero_rows(self):
        Q = sample_bernoulli_q(
            100,
            5,
            0.15,
            make_rng(3, "legacy"),
            condition_nonzero_rows=True,
        )
        self.assertTrue(np.all(Q.sum(axis=1) > 0))
        with self.assertRaises(ValueError):
            sample_bernoulli_q(
                2,
                3,
                0,
                make_rng(1),
                condition_nonzero_rows=True,
            )

    def test_row_sparse_metadata_records_actual_distribution(self):
        custom = actual_row_size_metadata(
            K=5,
            m=3,
            min_row_size=1,
            row_size_distribution="custom",
            row_size_probs=[2, 3, 5],
        )
        self.assertEqual(custom["row_size_support"], [1, 2, 3])
        np.testing.assert_allclose(custom["row_size_probs"], [0.2, 0.3, 0.5])

        fixed = actual_row_size_metadata(
            K=5,
            m=3,
            min_row_size=1,
            row_size_distribution="fixed",
            row_size_probs=None,
        )
        self.assertEqual(fixed["row_size_support"], [3])
        self.assertEqual(fixed["row_size_probs"], [1.0])

    def test_row_sparse_sampler_obeys_bounds(self):
        Q = sample_row_sparse_q(
            200,
            10,
            4,
            make_rng(8),
            min_row_size=2,
            row_size_distribution="uniform",
        )
        self.assertTrue(np.all(Q.sum(axis=1) >= 2))
        self.assertTrue(np.all(Q.sum(axis=1) <= 4))

    def test_legacy_solver_minus_one_is_now_exact_cadical(self):
        self.assertEqual(normalize_solver_name(-1), "cadical195")
        self.assertEqual(normalize_solver_name("-1"), "cadical195")


class ExperimentIOTests(unittest.TestCase):
    def test_actual_sat_path_defaults_to_single_clause_and_records_requested_choice(self):
        Q = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],
                      [1, 0, 1, 1], [0, 1, 1, 1]])
        status, _, branch, diagnostics = identifiability_expr(Q)
        self.assertEqual((status, branch), (0, 5))
        self.assertEqual(diagnostics["sat_cardinality_encoding"], "exclude_x")
        self.assertEqual(diagnostics["cardinality_encoding_requested"], "exclude_x")
        self.assertEqual(diagnostics["maximal_candidate_requested"], 0)
        _, _, _, prefix = identifiability_expr(Q, cardinality_encoding="prefix")
        self.assertEqual(prefix["sat_cardinality_encoding"], "prefix")
        self.assertGreater(prefix["sat_variables"], diagnostics["sat_variables"])
        _, _, _, prechecked = identifiability_expr(np.eye(4, dtype=int))
        self.assertEqual(prechecked["sat_called"], 0)
        self.assertEqual(prechecked["sat_cardinality_encoding"], "")
        self.assertEqual(prechecked["cardinality_encoding_requested"], "exclude_x")

    def test_original_necessary_flags_are_distinct_from_algorithm_exit_branch(self):
        fixtures = [
            (np.eye(3, dtype=int), 4, 1, 1),
            (np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]), 3, 1, 0),
            (np.array([[1, 1], [0, 1]]), 2, 0, 0),
            (np.zeros((2, 3), dtype=int), -1, 0, 0),
        ]
        for Q, expected_branch, pair_pass, both_pass in fixtures:
            _, _, branch, diagnostics = identifiability_expr(Q)
            self.assertEqual(branch, expected_branch)
            self.assertEqual(diagnostics["two_column_pass_original"], pair_pass)
            self.assertEqual(diagnostics["both_necessary_checks_pass_original"], both_pass)
            self.assertEqual(diagnostics["I_3c"], both_pass)
            self.assertGreaterEqual(diagnostics["diagnostics_time"], 0)

    def test_iid_zero_rows_replay_and_complete_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "iid.csv"
            run_bernoulli(
                3, 4, 2, 0.0, 17, output_csv=str(output),
                condition_nonzero_rows=False,
            )
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["zero_row_policy"] for row in rows}, {"allow_iid"})
            for row in rows:
                self.assertEqual(row["q_bitstring"], "0" * 12)
                self.assertEqual(row["q_sha256"], hashlib.sha256(b"3,4:000000000000").hexdigest())
                self.assertEqual(row["has_zero_row_original"], "1")
                self.assertEqual(row["branch"], "-1")
            sidecar = Path(str(output) + ".metadata.json")
            data = json.loads(sidecar.read_text())
            self.assertEqual(data["status"], "completed")
            self.assertTrue(data["completed"])
            self.assertEqual(data["completed_replicates"], 2)
            self.assertEqual(data["N"], 2)
            self.assertEqual(data["cardinality_encoding_requested"], "exclude_x")
            self.assertFalse(data["maximal_candidate_requested"])
            self.assertEqual(data["source_tree_sha256"], source_provenance()["source_tree_sha256"])
            self.assertIn("sat.py", data["source_hashes"])
            self.assertIn("algorithm_time", data["timing_definitions"])
            self.assertIn("pysat_version", data["environment"])
            self.assertEqual(data["environment"]["idQ_source_path"],
                             str(Path(__import__("idQ").__file__).resolve()))

    def test_both_drivers_propagate_encoding_into_real_sat_branch(self):
        Q = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],
                      [1, 0, 1, 1], [0, 1, 1, 1]])
        with tempfile.TemporaryDirectory() as directory:
            for name, runner, parameter in (("bernoulli", run_bernoulli, 0.5),
                                             ("row_sparsity", run_row_sparse, 3)):
                sampler = "sample_bernoulli_q" if name == "bernoulli" else "sample_row_sparse_q"
                with mock.patch(f"idQ.experiments.{name}.{sampler}", return_value=Q):
                    rows = runner(
                        5, 4, 1, parameter, 2,
                        output_csv=str(Path(directory) / f"{name}.csv"),
                        cardinality_encoding="exclude_h",
                    )
                self.assertEqual(rows[0]["branch"], 5)
                self.assertEqual(rows[0]["sat_cardinality_encoding"], "exclude_h")

    def test_interruption_preserves_checkpoint_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interrupted.csv"
            counter = 0

            def sampler(rng):
                nonlocal counter
                counter += 1
                if counter == 11:
                    raise RuntimeError("deliberate interrupted-run check")
                return np.eye(3, dtype=int)

            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                run_design_expr(
                    J=3, K=3, N=12, seed=1, solver="glucose42", sampler=sampler,
                    metadata={"design": "test"}, output_csv=output,
                )
            self.assertFalse(output.exists())
            with Path(str(output) + ".part").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)
            metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
            self.assertEqual(metadata["status"], "interrupted")
            self.assertFalse(metadata["completed"])
            self.assertEqual(metadata["completed_replicates"], 10)
            self.assertEqual(metadata["checkpoint_every"], 10)

    def test_driver_cli_options(self):
        from idQ.experiments.bernoulli import parse_args as bernoulli_args
        from idQ.experiments.row_sparsity import parse_args as sparse_args
        with mock.patch.object(sys, "argv", ["bernoulli", "20", "10", "100", "0.3", "7",
                                               "--allow-zero-rows"]):
            args = bernoulli_args()
            self.assertTrue(args.allow_zero_rows)
            self.assertEqual(args.cardinality_encoding, "exclude_x")
            self.assertFalse(args.maximal_candidate)
            self.assertEqual(args.checkpoint_every, 10)
        with mock.patch.object(sys, "argv", ["row_sparsity", "20", "10", "100", "3", "7",
                                               "--cardinality-encoding", "prefix"]):
            self.assertEqual(sparse_args().cardinality_encoding, "prefix")

    def test_core_edge_case_flows_through_helper(self):
        status, counterexample, branch, diagnostics = identifiability_expr(
            np.asarray([[1]]),
            solver="oracle",
        )
        self.assertEqual(status, 1)
        self.assertIsNone(counterexample)
        self.assertEqual(branch, 4)
        self.assertEqual(diagnostics["branch_label"], "identity_submatrix_identifiable")

    def test_bernoulli_output_is_atomic_and_refuses_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bernoulli.csv"
            rows = run_bernoulli(
                2,
                1,
                3,
                1.0,
                9,
                solver="oracle",
                output_csv=str(output),
            )
            self.assertEqual(len(rows), 3)
            self.assertTrue(output.exists())
            self.assertFalse(Path(str(output) + ".part").exists())
            frame = pd.read_csv(output)
            self.assertEqual(set(frame["schema_version"]), {SCHEMA_VERSION})
            self.assertIn("sat_formulation", frame.columns)
            self.assertEqual(set(frame["zero_row_policy"]), {"condition_nonzero"})

            with self.assertRaises(FileExistsError):
                run_bernoulli(
                    2,
                    1,
                    1,
                    1.0,
                    9,
                    solver="oracle",
                    output_csv=str(output),
                )

    def test_distinct_designs_are_not_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_row_sparse(
                2,
                1,
                2,
                1,
                5,
                solver="oracle",
                output_csv=str(root / "rowsparse_uniform_diag.csv"),
                row_size_distribution="uniform",
            )
            run_row_sparse(
                2,
                1,
                2,
                1,
                5,
                solver="oracle",
                output_csv=str(root / "rowsparse_fixed_diag.csv"),
                row_size_distribution="fixed",
            )
            data = read_row_sparsity_csvs(root)
            summary = summarize(data)
            self.assertEqual(len(data), 4)
            self.assertEqual(len(summary), 2)

            duplicated = pd.concat([data, data], ignore_index=True)
            with self.assertRaises(ValueError):
                summarize(duplicated)
            deduplicated = summarize(
                duplicated,
                drop_identical_duplicates=True,
            )
            self.assertEqual(len(deduplicated), 2)

    def test_summary_keeps_requested_encodings_separate_on_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for encoding in ("prefix", "exclude_x"):
                run_row_sparse(
                    2, 1, 2, 1, 5, solver="oracle",
                    output_csv=str(root / f"rowsparse_{encoding}_diag.csv"),
                    cardinality_encoding=encoding,
                )
            data = read_row_sparsity_csvs(root)
            self.assertEqual(data["q_sha256"].nunique(), 1)
            self.assertEqual(set(data["sat_called"]), {0})
            result = summarize(data)
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result["cardinality_encoding_requested"]), {"prefix", "exclude_x"})
            self.assertEqual(result["n"].tolist(), [2, 2])
            changed = data.iloc[:2].copy()
            changed.loc[changed.index[1], "cardinality_encoding_requested"] = "legacy"
            with self.assertRaisesRegex(ValueError, "varies within one job"):
                _validate_one_file(changed, root / "mixed.csv")

    def test_summary_keeps_maximal_candidate_settings_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for maximal in (False, True):
                run_row_sparse(
                    2, 1, 2, 1, 5, solver="oracle",
                    output_csv=str(root / f"rowsparse_max{int(maximal)}_diag.csv"),
                    maximal_candidate=maximal,
                )
            result = summarize(read_row_sparsity_csvs(root))
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result["maximal_candidate_requested"]), {0, 1})
            self.assertEqual(result["n"].tolist(), [2, 2])

    def test_summary_rejects_nan_binary_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rowsparse_uniform_diag.csv"
            run_row_sparse(
                2,
                1,
                2,
                1,
                6,
                solver="oracle",
                output_csv=str(path),
            )
            frame = pd.read_csv(path)
            frame.loc[0, "identifiable"] = np.nan
            with self.assertRaises(ValueError):
                _validate_one_file(frame, path)


if __name__ == "__main__":
    unittest.main()
