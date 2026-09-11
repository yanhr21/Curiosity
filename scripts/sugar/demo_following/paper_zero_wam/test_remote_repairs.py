"""Tensor/configuration checks for the imported fixes; no replacement model."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import torch

from .config import repaired_overfit_config
from .model import PaperZeroWAM
from .overfit_diagnostic import training_noise_seed
from .train import decay_parameter_groups, learning_rate


class RemoteRepairTests(unittest.TestCase):
    def test_velocity_readback_distinguishes_noise_from_clean_prediction(self):
        from .probe_overfit_inverse_dynamics import action_velocity_metrics
        # Exact arithmetic fixtures only: no learned or replacement model.
        generator = torch.Generator().manual_seed(12)
        clean = torch.randn(1, 40, 29, generator=generator, dtype=torch.float64)
        noise = torch.randn(1, 40, 29, generator=generator, dtype=torch.float64)
        ideal = action_velocity_metrics(noise - clean, clean, noise)
        self.assertEqual(ideal["velocity_mse"], 0)
        self.assertEqual(ideal["projection_design_rank"], 3)
        for actual, expected in zip(ideal["output_projection_on_noise_negative_clean_bias"], [1, 1, 0]):
            self.assertAlmostEqual(actual, expected, places=10)
        clean_only = action_velocity_metrics(-clean, clean, noise)
        self.assertAlmostEqual(clean_only["velocity_mse"], float(noise.square().mean()), places=12)
        for actual, expected in zip(clean_only["output_projection_on_noise_negative_clean_bias"], [0, 1, 0]):
            self.assertAlmostEqual(actual, expected, places=10)
        invalid = noise.clone(); invalid[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            action_velocity_metrics(invalid, clean, noise)

    def test_paired_noise_response_identifies_noise_insensitivity(self):
        from .probe_overfit_inverse_dynamics import paired_noise_response
        generator = torch.Generator().manual_seed(13)
        clean, a, b = [torch.randn(1, 40, 29, generator=generator, dtype=torch.float64) for _ in range(3)]
        ideal = paired_noise_response([a - clean, b - clean], [a, b])
        self.assertAlmostEqual(ideal["velocity_difference_gain_along_noise_difference"], 1)
        self.assertAlmostEqual(ideal["velocity_difference_error_to_ideal"], 0)
        insensitive = paired_noise_response([-clean, -clean], [a, b])
        self.assertEqual(insensitive["velocity_difference_gain_along_noise_difference"], 0)
        self.assertEqual(insensitive["velocity_difference_error_to_ideal"], 1)
        with self.assertRaises(ValueError):
            paired_noise_response([a, b], [a, a])

    def test_velocity_readback_preserves_bfloat16_training_target_rounding(self):
        from .probe_overfit_inverse_dynamics import action_velocity_metrics
        generator = torch.Generator().manual_seed(14)
        clean, noise = [torch.randn(1, 40, 29, generator=generator).bfloat16() for _ in range(2)]
        rounded_velocity = noise - clean
        self.assertGreater(float((rounded_velocity.double() - (noise.double() - clean.double())).square().mean()), 0)
        metrics = action_velocity_metrics(rounded_velocity, clean, noise)
        self.assertEqual(metrics["velocity_mse"], 0)

    def test_action_readback_exposes_temporally_constant_predictions(self):
        from .probe_overfit_inverse_dynamics import action_metrics, needs_inverse_probe
        # Arithmetic fixture, not a learned model or generated experiment.
        target = torch.arange(4, dtype=torch.float32)[None, :, None].expand(1, 4, 29)
        prediction = target.mean(dim=1, keepdim=True).expand_as(target)
        metrics = action_metrics(prediction, target)
        self.assertTrue(metrics["beats_normalized_zero"])
        self.assertFalse(metrics["beats_per_joint_constant_oracle"])
        self.assertEqual(metrics["temporal_delta_mse"], metrics["target_temporal_delta_energy"])
        self.assertEqual(metrics["per_joint_predicted_temporal_std"], [0.0] * 29)
        self.assertEqual(len(metrics["per_joint_mse"]), 29)
        self.assertTrue(needs_inverse_probe(metrics))
        self.assertFalse(needs_inverse_probe(action_metrics(target, target)))
        self.assertFalse(needs_inverse_probe(action_metrics(target * 0, target * 0)))
        self.assertEqual(action_metrics(target, target)["normalized_mse"], 0.0)

    def test_endpoint_configuration_survives_json_round_trip(self):
        from .probe_overfit_inverse_dynamics import endpoint_config
        original = repaired_overfit_config()
        saved = json.loads(json.dumps(original.as_dict()))
        self.assertIsInstance(saved["ifp_weights"], list)
        self.assertEqual(endpoint_config(saved), original)
        self.assertIsInstance(saved["ifp_weights"], list)
        saved["ifp_weights"][0] = 0.0
        with self.assertRaises(ValueError):
            endpoint_config(saved)

    def test_action_replay_consumes_the_same_random_draws_as_sampler(self):
        from .probe_overfit_inverse_dynamics import action_initial_noise
        batch = {"video_target_latents": torch.zeros(1, 48, 2, 4, 4),
                 "action_target": torch.zeros(1, 40, 29)}
        for inactive_noise in (False, True):
            config = replace(repaired_overfit_config(), inactive_action_token_noise=inactive_noise)
            torch.manual_seed(291701)
            torch.randn_like(batch["video_target_latents"])
            if inactive_noise:
                torch.randn_like(batch["action_target"])
            expected = torch.randn_like(batch["action_target"])
            self.assertTrue(torch.equal(action_initial_noise(batch, config, 291701), expected))

    def test_resampled_training_has_256_distinct_draw_seeds(self):
        config = repaired_overfit_config()
        seeds = {training_noise_seed(config, step, slot, False)
                 for step in range(32) for slot in range(8)}
        self.assertEqual(len(seeds), 256)

    def test_warmup_and_decay_groups_are_explicit(self):
        config = repaired_overfit_config()
        rates = [learning_rate(step, 32, config, "overfit") for step in range(32)]
        self.assertEqual(rates[0], 1.25e-6)
        self.assertEqual(rates[7:], [1e-5] * 25)
        values = [("block.weight", torch.nn.Parameter(torch.ones(3, 4))),
                  ("block.bias", torch.nn.Parameter(torch.ones(3))),
                  ("block.norm.weight", torch.nn.Parameter(torch.ones(3))),
                  ("block.modulation", torch.nn.Parameter(torch.ones(1, 6, 3)))]
        groups = decay_parameter_groups(values, config)
        names = {id(p): name for name, p in values}
        observed = {names[id(p)]: group["weight_decay"]
                    for group in groups for p in group["params"]}
        self.assertEqual(observed, {"block.weight": config.weight_decay,
                                    "block.bias": 0, "block.norm.weight": 0,
                                    "block.modulation": 0})

    def test_action_target_has_no_attention_path_to_video(self):
        # Exercise the real layout builder, not a learned stand-in. The
        # Boolean reachability closure checks arbitrary depth, including 30.
        owner = SimpleNamespace(config=replace(repaired_overfit_config(),
                                                repaired_conditioning=False))
        owner._video_positions = lambda *args: PaperZeroWAM._video_positions(owner, *args)
        owner._action_positions = PaperZeroWAM._action_positions
        for enabled in (False, True):
            layout = PaperZeroWAM._layout(owner, 2, 3, 2, 1, 1, 4, 4,
                                         torch.device("cpu"), enabled)
            mask = layout.attention_mask
            video_length = layout.video_target.stop
            reachable = torch.zeros(mask.shape[0], dtype=torch.bool)
            reachable[video_length + layout.action_target.start:
                      video_length + layout.action_target.stop] = True
            for _ in range(30):
                reachable = reachable | (mask & reachable[None, :]).any(dim=1)
            self.assertFalse(bool(reachable[:video_length].any()))


if __name__ == "__main__":
    unittest.main()
