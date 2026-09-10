"""CPU tensor-only optimizer regression; no model, forward, or GPU experiment."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from .config import PaperZeroWAMConfig
from .train_single_gpu import (
    EXECUTION, accumulate_cpu_gradients, deadline_checkpoint_needed,
    native_adamw_streamed_step, next_checkpoint_directory, single_gpu_checkpoint_step,
    requested_training_end,
)


class NativeOptimizerTest(unittest.TestCase):
    def test_user_stop_does_not_change_original_schedule_budget(self):
        self.assertEqual(requested_training_end("formal", 700), 700)
        self.assertEqual(requested_training_end("formal", None), 4200)
        self.assertEqual(requested_training_end("overfit", None), 32)
        self.assertEqual(PaperZeroWAMConfig().optimizer_steps, 4200)
        for mode, step in (("overfit", 700), ("formal", 699), ("formal", 701),
                           ("formal", True), ("formal", 700.0)):
            with self.assertRaises(ValueError):
                requested_training_end(mode, step)

    def test_partial_checkpoint_is_recovery_not_formal_completion(self):
        config = PaperZeroWAMConfig()
        state = dict(protocol="paper_zero_wam_single_gpu_checkpoint_v2", execution=EXECUTION,
                     model_and_schedule_config=json.loads(json.dumps(config.as_dict())),
                     architecture_parameter_count=config.expected_parameter_count,
                     hash_checks=False, optimizer_boundary_complete=True)
        for step in (1, 35, 36, 280, 4200):
            state.update(completed_optimizer_steps=step, completed_epochs=step // 280,
                         steps_in_current_epoch=step % 280)
            self.assertEqual(single_gpu_checkpoint_step(state, config), step)
        state["optimizer_boundary_complete"] = False
        with self.assertRaises(ValueError):
            single_gpu_checkpoint_step(state, config)
        self.assertTrue(deadline_checkpoint_needed(100, 1000, 60))
        self.assertFalse(deadline_checkpoint_needed(100, 2000, 60))
        self.assertTrue(deadline_checkpoint_needed(100, 2000, 1000))

    def test_checkpoint_write_never_targets_current_latest_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next_checkpoint_directory(root)
            first.mkdir()
            (root / "latest_checkpoint").symlink_to(first.name)
            self.assertEqual(next_checkpoint_directory(root).name, "checkpoint_slot1")
            (root / "latest_checkpoint").unlink()
            (root / "latest_checkpoint").symlink_to("checkpoint_slot1")
            self.assertEqual(next_checkpoint_directory(root).name, "checkpoint_slot0")

    def test_shared_storage_copy_preserves_offset_stride_and_eight_sums(self):
        accumulated = {}
        expected = {}
        for microbatch in range(8):
            source = torch.arange(64, dtype=torch.float32) + microbatch
            rows = [("a", SimpleNamespace(grad=source[3:15].reshape(3, 4))),
                    ("b", SimpleNamespace(grad=source[20:40:2])),
                    ("inactive", SimpleNamespace(grad=None))]
            for name, value in rows:
                if value.grad is not None:
                    expected[name] = expected.get(name, torch.zeros_like(value.grad)) + value.grad
            self.assertEqual(accumulate_cpu_gradients(rows, accumulated), 1)
            source.fill_(-999)
        for name in expected:
            self.assertTrue(torch.equal(accumulated[name], expected[name]))

    def test_streamed_native_steps_equal_original_adamw_bitwise(self):
        names = ("video.weight", "action_weight", "ifp_head.weight")
        original = [torch.nn.Parameter(torch.arange(7, dtype=torch.float32) / 7 + i)
                    for i in range(3)]
        serial = [torch.nn.Parameter(value.detach().clone()) for value in original]
        options = dict(lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                       weight_decay=0.01, foreach=False)
        reference = torch.optim.AdamW(original, **options)
        streamed = torch.optim.AdamW(serial, **options)
        initial_groups = streamed.param_groups
        for step in range(4):
            gradients = {}
            for i, (name, parameter) in enumerate(zip(names, original)):
                if step == 2 and i == 1:
                    parameter.grad = None
                    continue
                gradient = torch.full_like(parameter, (step + 1) * (i + 1) / 10)
                gradients[name] = gradient.clone()
                parameter.grad = gradient * 0.75
            before = [parameter.detach().clone() for parameter in serial]
            reference.step()
            updates, finite, calls = native_adamw_streamed_step(
                streamed, zip(names, serial), gradients, 0.75)
            self.assertTrue(finite)
            self.assertEqual(calls, 2 if step == 2 else 3)
            self.assertFalse(gradients)
            self.assertIs(streamed.param_groups, initial_groups)
            for i, (expected, actual) in enumerate(zip(original, serial)):
                self.assertTrue(torch.equal(expected, actual))
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(reference.state[expected][key],
                                                streamed.state[actual][key]))
                branch = ("video", "action", "ifp")[i]
                self.assertAlmostEqual(updates[branch], float((actual.detach() - before[i])
                                                            .double().norm()), places=12)
            reference.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
