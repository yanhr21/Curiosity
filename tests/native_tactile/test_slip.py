#!/usr/bin/env python3

import unittest

import numpy as np

from scripts.sugar.native_tactile.slip import SlipDetectorConfig, SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import (
    OpticalTactileFrame,
    TactileClock,
    UniversalTactileFrame,
)


def _frame(sequence: int, normal: np.ndarray, shear_x: np.ndarray) -> UniversalTactileFrame:
    normal = np.asarray(normal, dtype=np.float32).reshape(1, 1, 3, 3)
    shear = np.zeros((1, 1, 3, 3, 2), dtype=np.float32)
    shear[..., 0] = np.asarray(shear_x, dtype=np.float32).reshape(1, 1, 3, 3)
    return UniversalTactileFrame(
        backend="test",
        clock=TactileClock(sequence, 0.02 * sequence, 0.0 if sequence == 0 else 0.02),
        patch_names=("pad",),
        patch_size_m=np.asarray([[0.04, 0.03]], dtype=np.float32),
        penetration_m=(normal != 0.0).astype(np.float32) * 0.001,
        normal_force_n=normal,
        shear_force_xy_n=shear,
        active=normal != 0.0,
        taxel_position_w_m=None,
        taxel_orientation_w_xyzw=None,
        counterpart_fields={},
        optical=OpticalTactileFrame((False,), (None,), (None,), None),
        raw_samples=None,
    )


class TestTactileSlipDetector(unittest.TestCase):
    def test_metric_grid_uses_row_x_and_column_y(self):
        detector = TactileSlipDetector(["pad"], friction_coefficient=0.5)
        normal = np.zeros((3, 3), dtype=np.float32)
        normal[0, 2] = 10.0
        evidence = detector.update(_frame(0, normal, np.zeros_like(normal)))
        np.testing.assert_allclose(
            evidence.center_of_pressure_xy_m[0, 0],
            [-0.02, 0.015],
            atol=1.0e-7,
        )

    def test_stick_incipient_gross_and_release(self):
        config = SlipDetectorConfig(
            enter_frames=2,
            exit_frames=2,
            gross_cop_speed_m_s=0.2,
            gross_footprint_rate_s=100.0,
        )
        detector = TactileSlipDetector(["pad"], friction_coefficient=0.5, config=config)
        zero = np.zeros((3, 3), dtype=np.float32)
        center = zero.copy()
        center[1, 1] = 10.0

        evidence = detector.update(_frame(0, zero, zero))
        self.assertEqual(evidence.state[0, 0], SlipState.NO_CONTACT)
        evidence = detector.update(_frame(1, center, zero))
        self.assertEqual(evidence.state[0, 0], SlipState.STICK)

        high_shear = zero.copy()
        high_shear[1, 1] = 4.0
        detector.update(_frame(2, center, high_shear))
        evidence = detector.update(_frame(3, center, high_shear))
        self.assertEqual(evidence.state[0, 0], SlipState.STICK)
        self.assertAlmostEqual(evidence.friction_utilization[0, 0], 0.8)

        slight_right = center.copy()
        slight_right[1, 1] = 9.0
        slight_right[1, 2] = 1.0
        slight_right_shear = high_shear.copy()
        slight_right_shear[1, 1] = 3.6
        slight_right_shear[1, 2] = 0.4
        detector.update(_frame(4, slight_right, slight_right_shear))
        evidence = detector.update(_frame(5, center, high_shear))
        self.assertEqual(evidence.state[0, 0], SlipState.INCIPIENT)

        left = zero.copy()
        right = zero.copy()
        left[1, 0] = 10.0
        right[1, 2] = 10.0
        left_shear = zero.copy()
        right_shear = zero.copy()
        left_shear[1, 0] = 5.0
        right_shear[1, 2] = 5.0
        detector.update(_frame(6, left, left_shear))
        detector.update(_frame(7, right, right_shear))
        evidence = detector.update(_frame(8, left, left_shear))
        self.assertEqual(evidence.state[0, 0], SlipState.GROSS)
        self.assertGreater(evidence.center_of_pressure_speed_m_s[0, 0], 0.02)

        evidence = detector.update(_frame(9, zero, zero))
        self.assertEqual(evidence.state[0, 0], SlipState.NO_CONTACT)


if __name__ == "__main__":
    unittest.main()
