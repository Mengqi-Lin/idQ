"""Study completeness, provenance, denominators, and raw-indicator validation."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from idQ.core import (BRANCH_LABELS, contains_identity_submatrix,
                      first_two_column_violation, first_three_column_violation)
from idQ.experiments.paper_study import prepare
from idQ.experiments.paper_summary import analyze_study, read_study


def save_rows(path, rows):
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_row(task, sim, kind="complete"):
    J, K = task["J"], task["K"]
    if kind == "complete":
        Q = np.eye(K, dtype=int)[np.arange(J) % K]
        branch, Jb = 4, K
    elif kind == "pairs":
        pairs = list(itertools.combinations(range(K), 2))
        base = np.asarray([[int(k in pair) for k in range(K)] for pair in pairs])
        Q = base[np.arange(J) % len(base)]
        branch, Jb = 6, len(base)
    elif kind == "three_fail":
        Q = (1 - np.eye(K, dtype=int))[np.arange(J) % K]
        branch, Jb = 3, K
    else:
        Q = np.zeros((J, K), dtype=int)
        Q[:, 0] = 1
        branch, Jb = 2, 1
    complete = int(contains_identity_submatrix(Q))
    two = int(first_two_column_violation(Q) is None)
    three = int(first_three_column_violation(Q) is None)
    pure = int(np.any(Q.sum(axis=1) == 1))
    bits = "".join(str(x) for x in Q.flat)
    row = dict(task)
    row.update(schema_version=4, sim=sim, operator="conj", solver_name="glucose42",
               rng_engine="MT19937-RandomState", cardinality_encoding_requested="exclude_x",
               maximal_candidate_requested=0, design_id=task["cell_id"],
               identifiable=int(branch in (4, 6)), branch=branch, branch_label=BRANCH_LABELS[branch],
               sat_called=int(branch in (5, 6)), sat_cardinality_encoding="exclude_x" if branch == 6 else "",
               sat_formulation="boolean_factorization" if branch == 6 else "", J_basis=Jb,
               identity_original=complete, not_complete_original=1-complete,
               has_pure_node_original=pure, no_pure_nodes_original=1-pure, has_zero_row_original=0,
               two_column_pass_original=two, two_col_violation_original=1-two,
               three_col_violation_original=1-three, both_necessary_checks_pass_original=two*three,
               I_3c=two*three, q_bitstring=bits,
               q_sha256=hashlib.sha256(f"{J},{K}:{bits}".encode()).hexdigest(),
               basis_time=.01*(sim+1), identity_check_time=.01, two_col_check_time=.01,
               three_col_check_time=.01, preprocess_time=.01*(sim+1)+.03,
               sat_time=.2 if branch == 6 else 0,
               algorithm_time=.01*(sim+1)+.03+(.2 if branch == 6 else 0)+.01)
    if task["design"] == "row_sparsity":
        m = task["m_requested"]
        row.update(row_size_support=";".join(map(str, range(1, m+1))),
                   row_size_probs=";".join([str(1/m)]*m))
    return row


class PaperSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = prepare(self.root, replicates=3, per_task=3)
        self.tasks = self.manifest["tasks"]
        for index, task in enumerate(self.tasks):
            path = self.root/task["csv_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            kinds = ["complete", "pairs", "not_identifiable"] if index == 0 else ["complete"]*3
            rows = [make_row(task, i, kind) for i, kind in enumerate(kinds)]
            save_rows(path, rows)
            metadata = dict(rows[0])
            metadata.update(status="completed", completed=True, completed_replicates=3,
                            maximal_candidate_requested=False,
                            source_hashes=self.manifest["source_hashes"],
                            source_tree_sha256=self.manifest["source_tree_sha256"],
                            environment={"python_version": "test", "pysat_version": "test",
                                         "numpy_version": "test", "idQ_version": "test",
                                         "cpu_model": "fixture CPU", "hostname": "fixture node",
                                         "cpu_affinity": [0], "thread_environment": {"OMP_NUM_THREADS": "1"}},
                            timing_definitions={"algorithm_time": "all algorithm stages"})
            (self.root/task["metadata_path"]).write_text(json.dumps(metadata))

    def change_rows(self, mutator, task_index=0):
        path = self.root/self.tasks[task_index]["csv_path"]
        with path.open(newline="") as fp:
            rows = list(csv.DictReader(fp))
        mutator(rows)
        save_rows(path, rows)

    def test_exact_denominators_and_all_rep_runtime(self):
        result = analyze_study(self.root)
        self.assertEqual(result["status"], "smoke_complete")
        self.assertFalse(result["paper_final"])
        self.assertEqual(result["actual_total"], 108)
        cell = next(r for r in result["cells"] if r["design"] == "bernoulli" and (r["J"], r["K"], r["p"]) == (25, 5, .1))
        self.assertEqual(cell["n_identifiable"], 2)
        self.assertEqual(cell["conditional_denominator_identifiable"], 2)
        self.assertEqual(cell["pct_incomplete_given_identifiable"], 50)
        self.assertEqual(cell["pct_no_pure_given_identifiable"], 50)
        self.assertAlmostEqual(cell["pct_sat_called"], 100/3)
        self.assertAlmostEqual(cell["mean_preprocess_time"], (.04+.05+.06)/3)
        self.assertAlmostEqual(cell["mean_algorithm_time"], (.05+.26+.07)/3)
        tex = (self.root/"analysis_smoke"/"paper_tables_SMOKE.tex").read_text()
        for label in ("tab:bern", "tab:bern-pure-node-gap", "tab:sparse", "tab:bern-computation", "tab:bern-3c"):
            self.assertIn(r"\label{" + label + "}", tex)
        self.assertIn("NOT PAPER RESULTS", tex)

    def test_missing_and_part_fail_closed_but_partial_is_explicit(self):
        task = self.tasks[0]
        path = self.root/task["csv_path"]
        path.rename(Path(str(path)+".part"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze_study(self.root)
        result = analyze_study(self.root, allow_partial=True)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["actual_total"], 105)
        self.assertFalse(result["paper_final"])
        self.assertTrue((self.root/"analysis_partial"/"paper_tables_PARTIAL.tex").exists())
        self.assertFalse((self.root/"analysis"/"paper_tables.tex").exists())

    def test_duplicate_simulation_indices_rejected(self):
        self.change_rows(lambda rows: rows[1].update(sim=rows[0]["sim"]))
        with self.assertRaisesRegex(ValueError, "duplicate simulation"):
            read_study(self.root)

    def test_unlisted_copy_rejected_including_partial_mode(self):
        src = self.root/self.tasks[0]["csv_path"]
        (src.parent/"copied.csv").write_bytes(src.read_bytes())
        with self.assertRaisesRegex(ValueError, "Unlisted CSV"):
            read_study(self.root, allow_partial=True)

    def test_wrong_encoding_rejected_even_when_sat_not_called(self):
        self.change_rows(lambda rows: rows[0].update(cardinality_encoding_requested="prefix"))
        with self.assertRaisesRegex(ValueError, "cardinality_encoding_requested"):
            read_study(self.root)

    def test_wrong_bernoulli_conditioning_rejected(self):
        self.change_rows(lambda rows: rows[0].update(zero_row_policy="condition_nonzero"))
        with self.assertRaisesRegex(ValueError, "unconditioned iid"):
            read_study(self.root)

    def test_original_indicators_recomputed_and_conjunction_enforced(self):
        task = self.tasks[0]
        row = make_row(task, 2, "three_fail")
        self.assertEqual(row["two_column_pass_original"], 1)
        self.assertEqual(row["I_3c"], 0)
        self.change_rows(lambda rows: rows.__setitem__(2, row))
        read_study(self.root)
        self.change_rows(lambda rows: rows[2].update(I_3c="1", both_necessary_checks_pass_original="1"))
        with self.assertRaisesRegex(ValueError, "indicators do not match"):
            read_study(self.root)

    def test_corrupt_matrix_fingerprint_rejected(self):
        self.change_rows(lambda rows: rows[0].update(q_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            read_study(self.root)

    def test_provenance_mismatch_rejected(self):
        path = self.root/self.tasks[0]["metadata_path"]
        metadata = json.loads(path.read_text())
        metadata["source_tree_sha256"] = "0"*64
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "source provenance"):
            read_study(self.root)

    def test_mixed_software_rejected(self):
        path = self.root/self.tasks[1]["metadata_path"]
        metadata = json.loads(path.read_text())
        metadata["environment"]["pysat_version"] = "different version"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "mixed software"):
            read_study(self.root)

    def test_mixed_hardware_permitted_and_explicit(self):
        path = self.root/self.tasks[1]["metadata_path"]
        metadata = json.loads(path.read_text())
        metadata["environment"]["cpu_model"] = "second CPU"
        path.write_text(json.dumps(metadata))
        result = analyze_study(self.root)
        runtime = result["runtime_provenance"]
        self.assertTrue(runtime["mixed_hardware"])
        self.assertEqual(runtime["cpu_model_job_counts"], {"fixture CPU": 35, "second CPU": 1})
        self.assertEqual(runtime["execution_mode_job_counts"], {"local": 36})

    def test_mixed_timing_definitions_rejected(self):
        path = self.root/self.tasks[1]["metadata_path"]
        metadata = json.loads(path.read_text())
        metadata["timing_definitions"]["algorithm_time"] = "solver only"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "mixed timing"):
            read_study(self.root)

    def test_zero_identifiable_denominator_is_null(self):
        task = self.tasks[0]
        self.change_rows(lambda rows: rows.__setitem__(slice(None), [make_row(task, i, "not_identifiable") for i in range(3)]))
        result = analyze_study(self.root)
        cell = next(r for r in result["cells"] if r["design"] == "bernoulli" and (r["J"], r["K"], r["p"]) == (25, 5, .1))
        self.assertEqual(cell["conditional_denominator_identifiable"], 0)
        self.assertIsNone(cell["pct_incomplete_given_identifiable"])
        self.assertIsNone(cell["pct_no_pure_given_identifiable"])

    def test_manifest_missing_cell_rejected(self):
        self.manifest["tasks"].pop()
        (self.root/"manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "exactly the 30"):
            read_study(self.root)


if __name__ == "__main__":
    unittest.main()
