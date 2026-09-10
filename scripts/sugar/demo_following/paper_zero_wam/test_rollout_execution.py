"""CPU-only case/history routing tests; no model or physical simulation."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .config import PaperZeroWAMConfig
from .rollout_smallbox import CausalRolloutBatch


class RolloutRoutingTest(unittest.TestCase):
    def make_runtime(self):
        frames = np.broadcast_to(np.arange(8, dtype=np.uint8)[:, None, None, None],
                                 (8, 320, 320, 3)).copy()
        return CausalRolloutBatch(
            model=None, vae=None, normalizer=SimpleNamespace(deployed_history=lambda value: value),
            config=PaperZeroWAMConfig(), rank=0, device=torch.device("cpu"),
            prompts={slot: torch.tensor([slot]) for slot in range(8)},
            initial_rgb=frames, single_gpu=True)

    def test_eight_independent_causal_histories_and_terminal_chunk(self):
        runtime = self.make_runtime()
        expected_actions = [[] for _ in range(8)]
        frame_count = 1
        for count in (40, 10):
            actions = np.broadcast_to(np.arange(8, dtype=np.float32)[None, :, None],
                                      (count, 8, 29)).copy()
            rgb = np.broadcast_to(np.arange(8, dtype=np.uint8)[None, :, None, None, None],
                                  (count // 5, 8, 320, 320, 3)).copy()
            runtime.append_response(dict(executed_action=actions, rgb=rgb), count)
            frame_count += count // 5
            for slot in range(8):
                expected_actions[slot].append(torch.from_numpy(actions[:, slot]).to(torch.bfloat16))
                self.assertTrue(torch.equal(runtime.actions[slot], torch.cat(expected_actions[slot])))
                self.assertEqual(len(runtime.histories[slot]), frame_count)
                self.assertTrue(all(bool((frame == slot).all()) for frame in runtime.histories[slot]))
        runtime.histories[0][0].fill_(99)
        self.assertTrue(bool((runtime.histories[1][0] == 1).all()))
        self.assertEqual(len({id(value) for value in runtime.histories.values()}), 8)

    def test_missing_case_response_is_rejected(self):
        runtime = self.make_runtime()
        with self.assertRaises(ValueError):
            runtime.append_response(dict(executed_action=np.zeros((40, 7, 29))), 40)
        with self.assertRaises(ValueError):
            runtime._frames(np.zeros((7, 320, 320, 3)))


if __name__ == "__main__":
    unittest.main()
