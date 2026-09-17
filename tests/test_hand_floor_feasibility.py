import unittest
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scripts.sugar.object_predictor.hand_floor_feasibility import (
    feasible_hand_step, interpolate_pose, swept_min_height)


def pose(z=0., angle=0.):
    return np.r_[0., 0., z, Rotation.from_euler('y', angle, degrees=True).as_quat()]


class HandFloorFeasibilityTests(unittest.TestCase):
    def test_wrist_clamp_is_not_mesh_safety(self):
        vertices = np.array([[0., 0., -.08], [.01, 0., -.08], [0., .01, -.08]])
        old, target = pose(.10), pose(.04)
        self.assertGreater(target[2], 0.)
        self.assertAlmostEqual(swept_min_height(vertices, old, target), -.04)
        out, receipt = feasible_hand_step([vertices], [old], [target])
        self.assertTrue(receipt['clipped'])
        self.assertGreaterEqual(swept_min_height(vertices, old, out[0]), 1e-5 - 1e-14)

    def test_safe_endpoints_can_have_penetrating_sweep(self):
        vertices = np.array([[1., 0., 0.], [1., .01, 0.], [.99, 0., 0.]])
        old, target = pose(.5, 0.), pose(.5, 180.)
        self.assertGreater(Rotation.from_quat(target[3:]).apply(vertices)[:, 2].min() + target[2], .49)
        self.assertAlmostEqual(swept_min_height(vertices, old, target), -.5)
        out, receipt = feasible_hand_step([vertices], [old], [target])
        self.assertLess(receipt['fraction'], 1. / 6.)
        self.assertGreaterEqual(swept_min_height(vertices, old, out[0]), 1e-5 - 1e-14)

    def test_safe_commands_exact_and_no_mutation(self):
        vertices = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
        old = np.array([pose(.1), pose(.2)])
        target = np.array([pose(.2, 30.), pose(.3, -60.)])
        before = target.copy()
        out, receipt = feasible_hand_step([vertices, vertices], old, target)
        np.testing.assert_array_equal(out, target)
        np.testing.assert_array_equal(target, before)
        self.assertEqual(receipt['fraction'], 1.)

    def test_common_fraction_preserves_two_hand_coupling(self):
        vertices = np.array([[0., 0., -.02], [.01, 0., -.02], [0., .01, -.02]])
        old = np.array([pose(.04), pose(.2)])
        target = np.array([pose(.01), pose(.25)])
        out, receipt = feasible_hand_step([vertices, vertices], old, target)
        alpha = receipt['fraction']
        self.assertLess(alpha, 1.)
        np.testing.assert_allclose(out[:, :3] - old[:, :3], alpha * (target[:, :3] - old[:, :3]), atol=1e-14)

    def test_reject_initial_penetration_no_teleport(self):
        vertices = np.array([[0., 0., -.08]])
        with self.assertRaisesRegex(ValueError, 'Initial full hand'):
            feasible_hand_step([vertices], [pose(.05)], [pose(.15)])

    def test_stationary_roots_match_dense_independent_slerp(self):
        rng = np.random.default_rng(41)
        vertices = rng.uniform(-.15, .15, (24, 3))
        for _ in range(12):
            old = np.r_[rng.normal(size=3), Rotation.random(random_state=rng).as_quat()]
            target = np.r_[rng.normal(size=3), Rotation.random(random_state=rng).as_quat()]
            fractions = np.linspace(0., 1., 4001)
            rotations = Slerp([0., 1.], Rotation.from_quat([old[3:], target[3:]]))(fractions)
            z = np.einsum('ti,vi->tv', rotations.as_matrix()[:, 2, :], vertices)
            z += (old[2] + fractions * (target[2] - old[2]))[:, None]
            exact = swept_min_height(vertices, old, target)
            self.assertLessEqual(exact, float(z.min()) + 2e-14)
            self.assertLess(float(z.min()) - exact, 2e-7)

    def test_world_xy_yaw_and_quaternion_sign_invariance(self):
        vertices = np.array([[.1, .03, -.04], [.04, .02, .01]])
        old, target = pose(.08, -10.), pose(.04, 55.)
        expected = swept_min_height(vertices, old, target)
        turn = Rotation.from_euler('z', 71., degrees=True)
        def transform(p):
            return np.r_[turn.apply(p[:3]) + [3., -7., 0.], -(turn * Rotation.from_quat(p[3:])).as_quat()]
        self.assertAlmostEqual(swept_min_height(vertices, transform(old), transform(target)), expected, places=13)

    def test_float32_storage_keeps_positive_reserved_clearance(self):
        vertices = np.array([[.1, 0., -.05], [-.1, 0., -.05]])
        old, target = pose(.1), pose(.025, 65.)
        out, _ = feasible_hand_step([vertices], [old], [target])
        rounded = out.astype(np.float32).astype(float)[0]
        self.assertGreater(swept_min_height(vertices, old, rounded), 0.)


if __name__ == '__main__':
    unittest.main()
