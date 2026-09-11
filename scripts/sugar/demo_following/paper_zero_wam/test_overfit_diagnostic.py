"""CPU algebra/contracts only, not a replacement-model training experiment."""
import unittest
import json
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

import torch

from .config import PaperZeroWAMConfig
from .model import PaperZeroWAM, shifted_flow_time, inference_sigmas
from .train import overfit_execution_decision
from .overfit_diagnostic import (
    DIAGNOSTIC_DIRECTORY, reduction_result, training_noise_seed,
    validate_diagnostic_request, verify_first_forward, diagnostic_contract,
)


class OverfitDiagnosticTest(unittest.TestCase):
    def test_contract_distinguishes_detached_ifp_from_paper_representation_gradient(self):
        config = PaperZeroWAMConfig()
        for enabled in (False, True):
            contract = diagnostic_contract(replace(config, ifp_trunk_gradient=enabled), True)
            self.assertEqual(contract["objective_changes"], not enabled)
            self.assertEqual(contract["ifp_trunk_gradient"], enabled)
        self.assertNotIn("cannot succeed", diagnostic_contract(config)["scope"])

    def test_continuation_keeps_exactly_32_new_global_step_records(self):
        # Trace arithmetic fixture only, not an experiment or model result.
        config = PaperZeroWAMConfig()
        samples = [{"rank": i, "task": "CarryBox" if i < 4 else "KickBox"} for i in range(8)]
        base = dict(mode="overfit", world_size=1, packed_samples_per_rank=1,
                    global_packed_sample_batch=8, gradient_accumulation_steps=8,
                    main_pass_count_per_sample=2, samples=samples,
                    physical_execution_map=[dict(logical_schedule_slot=i, physical_rank=0,
                                                 microbatch_index=i) for i in range(8)],
                    losses=dict.fromkeys(("loss", "video_loss", "action_loss", "ifp_loss", "ifp_active_heads"), 1.0),
                    gradient_norms=dict.fromkeys(("video", "action", "ifp"), 1.0),
                    parameter_update_norms=dict.fromkeys(("video", "action", "ifp"), 1.0),
                    optimizer_applied=True, amp_scaler_skipped=False,
                    all_trainable_parameters_finite=True, learning_rate=1e-5)
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.sugar.demo_following.paper_zero_wam.train.expected_overfit_trace_samples",
                return_value=samples), patch(
                "scripts.sugar.demo_following.paper_zero_wam.train.expected_overfit_step_evidence",
                return_value={}):
            log = Path(directory) / "fixture.jsonl"
            for start in (0, 32, 64):
                rows = [dict(base, optimizer_step=step) for step in range(start, start + 32)]
                log.write_text("".join(json.dumps(row) + "\n" for row in rows))
                def decision(offset=start, count=32):
                    return overfit_execution_decision(log, Path(directory), count, config,
                        execution_world_size=1, accumulation_steps=8, start_step=offset)
                self.assertTrue(decision()["passed"])
                self.assertFalse(decision(start + 32)["passed"])
                self.assertFalse(decision(start, 31)["passed"])
                rows[16]["optimizer_step"] += 1
                log.write_text("".join(json.dumps(row) + "\n" for row in rows))
                self.assertFalse(decision()["passed"])
            with self.assertRaises(ValueError):
                decision(1)

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
