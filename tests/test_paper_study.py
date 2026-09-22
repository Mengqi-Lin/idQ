from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from idQ.experiments.paper_study import prepare, check_source, array_spec, task_state


class PaperStudyTests(unittest.TestCase):
    def test_full_design_and_partial_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare(directory, replicates=7, per_task=3, base_seed=19)
            tasks = manifest["tasks"]
            self.assertEqual(len(tasks), 36 * 3)
            self.assertEqual(sum(t["N"] for t in tasks), 36 * 7)
            self.assertEqual(len({t["csv_path"] for t in tasks}), len(tasks))
            self.assertEqual(len({t["seed"] for t in tasks}), len(tasks))
            for cell in {t["cell_id"] for t in tasks}:
                self.assertEqual([t["N"] for t in tasks if t["cell_id"] == cell], [3,3,1])
            bern = [t for t in tasks if t["design"] == "bernoulli"]
            self.assertEqual(len(bern), 30 * 3)
            self.assertEqual({t["zero_row_policy"] for t in bern}, {"allow_iid"})
            sparse = [t for t in tasks if t["design"] == "row_sparsity"]
            self.assertEqual({t["min_row_size_requested"] for t in sparse}, {1})
            self.assertEqual({t["row_size_distribution"] for t in sparse}, {"uniform"})
            self.assertEqual(manifest["cardinality_encoding"], "exclude_x")
            self.assertFalse(manifest["maximal_candidate"])
            with self.assertRaises(FileExistsError):
                prepare(directory)

    def test_source_guard_and_partial_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare(directory, replicates=1)
            root = Path(directory)
            task = manifest["tasks"][0]
            self.assertEqual(task_state(root, task), "missing")
            partial = root / (task["csv_path"] + ".part")
            partial.parent.mkdir(parents=True)
            partial.write_text("interrupted")
            self.assertEqual(task_state(root, task), "incomplete")
            with patch("idQ.experiments.common.source_provenance", return_value={"source_tree_sha256":"changed"}):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    check_source(manifest)

    def test_resubmission_array_ranges(self):
        self.assertEqual(array_spec(range(360)), "0-359")
        self.assertEqual(array_spec([0,1,4,8,9]), "0-1,4,8-9")
        self.assertEqual(array_spec([]), "")


if __name__ == "__main__":
    unittest.main()
