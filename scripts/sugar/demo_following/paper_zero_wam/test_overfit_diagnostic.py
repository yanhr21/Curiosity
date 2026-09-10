"""CPU algebra/contracts only, not a replacement-model training experiment."""
import unittest
from types import SimpleNamespace

import torch

from .config import PaperZeroWAMConfig
from .model import PaperZeroWAM, shifted_flow_time, inference_sigmas
from .overfit_diagnostic import (
    DIAGNOSTIC_DIRECTORY, reduction_result, training_noise_seed,
    validate_diagnostic_request, verify_first_forward,
)


class OverfitDiagnosticTest(unittest.TestCase):
    def test_only_new_overfit_directory_is_admitted(self):
        config = PaperZeroWAMConfig()
        root = config.resolved(config.output_root)
        validate_diagnostic_request(True, "overfit", root / DIAGNOSTIC_DIRECTORY, config)
        for mode, path in (("formal", root / DIAGNOSTIC_DIRECTORY),
                           ("overfit", root / "overfit"), ("overfit", root / "formal")):
            with self.assertRaises(ValueError):
                validate_diagnostic_request(True, mode, path, config)

    def test_frozen_noise_matches_probe_without_changing_old_schedule(self):
        config = PaperZeroWAMConfig()
        for step in range(32):
            for slot in range(8):
                self.assertEqual(training_noise_seed(config, step, slot, True),
                                 config.noise_seed + 9_999_991)
                self.assertEqual(training_noise_seed(config, step, slot, False),
                                 config.noise_seed + step * 8 + slot)

    def test_reduction_does_not_hide_one_failed_task(self):
        names = ("video_loss", "action_loss", "ifp_loss")
        first = {"losses": dict.fromkeys(names, 1.0),
                 "task_losses": {task: dict.fromkeys(names, 1.0)
                                 for task in ("CarryBox", "KickBox")}}
        last = {"losses": dict.fromkeys(names, 0.4),
                "task_losses": {task: dict.fromkeys(names, 0.4)
                                for task in ("CarryBox", "KickBox")}}
        self.assertTrue(reduction_result(first, last)["passed"])
        last["task_losses"]["KickBox"]["action_loss"] = 0.8
        self.assertFalse(reduction_result(first, last)["passed"])
        initial = {"losses": {"matched": first["losses"]}}
        self.assertTrue(verify_first_forward(initial, first["losses"])["passed"])
        with self.assertRaises(RuntimeError):
            verify_first_forward(initial, last["losses"])

    def test_target_velocity_and_reverse_integration_sign(self):
        clean = torch.arange(20, dtype=torch.float32).reshape(1, 20) / 20
        noise = torch.flip(clean, (1,)) + 0.2
        for shift in (1.0, 5.0):
            time = shifted_flow_time(torch.tensor([0.35]), shift)
            noisy = (1 - time) * clean + time * noise
            velocity = noise - clean
            torch.testing.assert_close(noisy - time * velocity, clean)
            sigmas = inference_sigmas(25, shift, device=torch.device("cpu"))
            reconstructed = noise.clone()
            for start, stop in zip(sigmas, sigmas[1:]):
                reconstructed += (stop - start) * velocity
            torch.testing.assert_close(reconstructed, clean)

    def test_unpatchify_preserves_official_channel_spatial_order(self):
        # Call the real pure reshape method, without constructing any model.
        owner = SimpleNamespace(config=PaperZeroWAMConfig())
        source = torch.arange(1 * 2 * 3 * 4 * 4 * 48).reshape(1, 2 * 3 * 4, 4 * 48)
        result = PaperZeroWAM._unpatchify(owner, source, 2, 3, 4)
        expected = source.reshape(1, 2, 3, 4, 1, 2, 2, 48)
        expected = expected.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(1, 48, 2, 6, 8)
        self.assertTrue(torch.equal(result, expected))

    def test_real_layout_blocks_future_to_history_and_demo_to_action_pass(self):
        # Real mask code on short token grids: algebra test, not a toy network.
        owner = SimpleNamespace(config=PaperZeroWAMConfig())
        owner._video_positions = lambda *args: PaperZeroWAM._video_positions(owner, *args)
        owner._action_positions = PaperZeroWAM._action_positions
        for enabled in (True, False):
            layout = PaperZeroWAM._layout(owner, 2, 3, 2, 1, 1, 40, 40,
                                          torch.device("cpu"), enabled)
            mask = layout.attention_mask
            action_start = layout.video_target.stop
            target_action = slice(action_start + 40, action_start + 80)
            self.assertFalse(bool(mask[:action_start, target_action].any()))
            self.assertFalse(bool(mask[layout.robot_history, layout.video_target].any()))
            self.assertFalse(bool(mask[action_start:, layout.prompt].any()))
            self.assertEqual(bool(mask[layout.video_target, layout.prompt].all()), enabled)
            self.assertTrue(bool(mask[target_action, layout.video_target].all()))
            if not enabled:
                self.assertFalse(bool(mask[layout.prompt.stop:, layout.prompt].any()))


if __name__ == "__main__":
    unittest.main()
