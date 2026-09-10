"""CPU file-only recovery tests; no model or optimizer execution."""
import json
from pathlib import Path
import tempfile
import unittest

from .train_single_gpu import prepare_training_trace


class TraceRecoveryTest(unittest.TestCase):
    def test_uncommitted_and_partial_tail_is_archived_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TRAIN_TRACE.jsonl"
            rows = [json.dumps({"optimizer_step": step}) + "\n" for step in range(3)]
            original = "".join(rows) + '{"optimizer_step":'
            path.write_text(original)
            prepare_training_trace(path, 2)
            self.assertEqual(path.read_text(), "".join(rows[:2]))
            archives = list(path.parent.glob("TRAIN_TRACE.interrupted_*.jsonl"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(), original)
            prepare_training_trace(path, 2)
            self.assertEqual(len(list(path.parent.glob("TRAIN_TRACE.interrupted_*.jsonl"))), 1)

    def test_broken_committed_prefix_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TRAIN_TRACE.jsonl"
            original = '{"optimizer_step":0}\n{"optimizer_step":2}\n'
            path.write_text(original)
            with self.assertRaises(ValueError):
                prepare_training_trace(path, 2)
            self.assertEqual(path.read_text(), original)

    def test_no_checkpoint_retains_prior_execution_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TRAIN_TRACE.jsonl"
            original = '{"optimizer_step":0}\n'
            path.write_text(original)
            prepare_training_trace(path, 0)
            self.assertEqual(path.read_text(), "")
            self.assertEqual(next(path.parent.glob("TRAIN_TRACE.interrupted_*.jsonl")).read_text(), original)


if __name__ == "__main__":
    unittest.main()
