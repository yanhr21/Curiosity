#!/usr/bin/env python3

import unittest

import numpy as np

from scripts.sugar.native_tactile.evaluate_tactile_only_slip import oracle_state


class TestSlipEvaluation(unittest.TestCase):
    def test_velocity_is_evaluation_only_and_patch_reduced(self):
        penetration = np.zeros((2, 1, 2, 2), dtype=np.float32)
        velocity = np.zeros((2, 1, 2, 2, 3), dtype=np.float32)
        penetration[:, 0, 0, 1] = 0.001
        velocity[0, 0, 0, 1, 0] = 0.006
        velocity[1, 0, 0, 1, 1] = 0.03
        state, maximum = oracle_state(
            penetration,
            velocity,
            incipient_speed_m_s=0.005,
            gross_speed_m_s=0.02,
        )
        np.testing.assert_array_equal(state[:, 0], [2, 3])
        np.testing.assert_allclose(maximum[:, 0], [0.006, 0.03])


if __name__ == "__main__":
    unittest.main()
