#!/usr/bin/env python3

import unittest

import numpy as np

from scripts.sugar.native_tactile.evaluate_online_patch_slip import oracle_state


class TestSlipEvaluation(unittest.TestCase):
    def test_velocity_is_evaluation_only_and_patch_reduced(self):
        contact = np.ones((2, 1), dtype=bool)
        maximum = np.asarray([[0.006], [0.03]], dtype=np.float32)
        state = oracle_state(
            contact,
            maximum,
            incipient_speed_m_s=0.005,
            gross_speed_m_s=0.02,
        )
        np.testing.assert_array_equal(state[:, 0], [2, 3])
        np.testing.assert_allclose(maximum[:, 0], [0.006, 0.03])


if __name__ == "__main__":
    unittest.main()
