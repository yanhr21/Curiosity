#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.sugar.native_tactile.universal import IsaacLabTacSLAdapter


def _data(offset: float, *, optical: bool = False):
    taxels = 6
    penetration = np.zeros((2, taxels), dtype=np.float32)
    normal = np.zeros((2, taxels), dtype=np.float32)
    shear = np.zeros((2, taxels, 2), dtype=np.float32)
    penetration[:, 1] = 0.001 + offset
    normal[:, 1] = 2.0 + offset
    shear[:, 1] = [0.25 + offset, -0.5]
    positions = np.arange(2 * taxels * 3, dtype=np.float32).reshape(2, taxels, 3)
    quaternions = np.zeros((2, taxels, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    return SimpleNamespace(
        penetration_depth=penetration,
        tactile_normal_force=normal,
        tactile_shear_force=shear,
        tactile_points_pos_w=positions,
        tactile_points_quat_w=quaternions,
        tactile_rgb_image=np.zeros((2, 4, 5, 3), dtype=np.uint8) if optical else None,
        tactile_depth_image=np.zeros((2, 4, 5, 1), dtype=np.float32) if optical else None,
    )


class TestUniversalTacSLAdapter(unittest.TestCase):
    def test_preserves_order_sign_counterparts_and_clock(self):
        adapter = IsaacLabTacSLAdapter(
            ["left", "right"],
            grid_shape=(2, 3),
            patch_size_m=[(0.04, 0.03), (0.02, 0.01)],
        )
        frame = adapter.update(
            {
                "box": [_data(0.0, optical=True), _data(0.1)],
                "table": [_data(0.2), _data(0.3)],
            },
            timestamp_s=1.0,
        )
        self.assertEqual(frame.backend, "isaaclab_tacsl")
        self.assertEqual(frame.normal_force_n.shape, (2, 2, 2, 3))
        self.assertEqual(frame.shear_force_xy_n.shape, (2, 2, 2, 3, 2))
        self.assertEqual(tuple(frame.counterpart_fields), ("box", "table"))
        np.testing.assert_allclose(frame.normal_force_n[:, 0, 0, 1], 4.2)
        np.testing.assert_allclose(frame.normal_force_n[:, 0, 0, [0, 2]], 0.0)
        self.assertLess(float(frame.shear_force_xy_n[0, 0, 0, 1, 1]), 0.0)
        self.assertEqual(frame.clock.sequence, 0)
        self.assertEqual(frame.clock.dt_s, 0.0)
        self.assertEqual(frame.optical.available, (True, False))

        frame2 = adapter.update({"box": [_data(0.0), _data(0.1)]}, timestamp_s=1.02)
        self.assertEqual(frame2.clock.sequence, 1)
        self.assertAlmostEqual(frame2.clock.dt_s, 0.02)


if __name__ == "__main__":
    unittest.main()
