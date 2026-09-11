"""Tensor/configuration checks for the imported fixes; no replacement model."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from .config import repaired_overfit_config
from .model import PaperZeroWAM
from .overfit_diagnostic import training_noise_seed
from .train import decay_parameter_groups, learning_rate


class RemoteRepairTests(unittest.TestCase):
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
