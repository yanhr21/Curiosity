"""CPU metadata/process-identity tests; these tests never send signals."""
import os
import unittest

from .stop_at_checkpoint import checkpoint_ready, process_identity


class CheckpointStopTest(unittest.TestCase):
    def setUp(self):
        self.state = dict(protocol="paper_zero_wam_single_gpu_checkpoint_v2",
                          completed_optimizer_steps=700, optimizer_boundary_complete=True,
                          completed_epochs=2, steps_in_current_epoch=140, hash_checks=False)
        self.progress = dict(optimizer_steps_completed=700, checkpoint_complete=True,
                             formal_execution_complete=False)
        self.row = dict(optimizer_step=699, optimizer_applied=True, amp_scaler_skipped=False,
                        all_trainable_parameters_finite=True, gpu_parameter_master_readback_exact=True)

    def test_complete_boundary_only(self):
        self.assertTrue(checkpoint_ready(self.state, self.progress, self.row))
        for object_name, field, value in (
            ("state", "completed_optimizer_steps", 665),
            ("state", "optimizer_boundary_complete", False),
            ("progress", "optimizer_steps_completed", 665),
            ("progress", "checkpoint_complete", False),
            ("row", "optimizer_step", 698),
            ("row", "optimizer_applied", False),
            ("row", "gpu_parameter_master_readback_exact", False),
        ):
            state, progress, row = self.state.copy(), self.progress.copy(), self.row.copy()
            {"state": state, "progress": progress, "row": row}[object_name][field] = value
            self.assertFalse(checkpoint_ready(state, progress, row))

    def test_extra_update_is_rejected(self):
        with self.assertRaises(ValueError):
            checkpoint_ready(self.state, self.progress, dict(self.row, optimizer_step=700))

    def test_proc_identity_parser_matches_current_process(self):
        identity = process_identity(os.getpid())
        self.assertEqual(identity["ppid"], os.getppid())
        self.assertEqual(identity["pgid"], os.getpgrp())
        self.assertGreater(identity["start_ticks"], 0)


if __name__ == "__main__":
    unittest.main()
