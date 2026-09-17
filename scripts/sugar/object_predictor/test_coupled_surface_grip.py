"""CPU geometry/controller tests; no physics, model, or rollout qualification."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .coupled_surface_grip import CoupledSurfaceGripController
from .surface_grip_scene import SurfaceGripController


class CoupledSurfaceGripTests(unittest.TestCase):
    def pair(self, **kwargs):
        common = dict(controller_revision='sensor_feedback_v2', target_load_n=16.,
                      force_gain=.00015, response_gain=True, **kwargs)
        pair = SurfaceGripController(**common), CoupledSurfaceGripController(**common)
        # Exact synthetic hand-coordinate basis isolates the intervention.
        for controller in pair:
            controller.support_normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
        return pair

    def observation(self, angles=(0., 0.), centers=None):
        poses = np.array([[-.2, 0., .3, 0., 0., 0., 1.],
                          [.2, 0., .3, 0., 0., 0., 1.]])
        if centers is None:
            centers = np.zeros((2, 3))
        points = []
        for side, angle in enumerate(angles):
            turn = Rotation.from_euler('z', angle, degrees=True)
            points.extend(turn.apply([[0., y, z] for y in (-.002, 0., .002)
                                     for z in (-.002, 0., .002)]) + centers[side])
        field = dict(pos=np.array(points), area=np.full(18, 1e-6),
                     pressure=np.full(18, 1e6), pad=np.repeat([0, 27], 9),
                     patch=np.repeat([0, 1], 9))
        return poses, field

    def assert_records_equal(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_opposed_planes_exactly_preserve_v2_with_unequal_loads(self):
        base, coupled = self.pair()
        poses, field = self.observation()
        outputs = []
        for controller in (base, coupled):
            controller.observe(poses, field)
            outputs.append(controller.command(12., np.array([8., 20.]), .02))
        for i in (0, 1):
            np.testing.assert_array_equal(outputs[0][i], outputs[1][i])
        self.assert_records_equal(outputs[0][2], outputs[1][2])

    def test_missing_side_and_coincident_centers_fall_back_exactly(self):
        for missing in (True, False):
            base, coupled = self.pair()
            centers = None if missing else np.array([[.2, 0., 0.], [-.2, 0., 0.]])
            poses, field = self.observation((20., -30.), centers)
            if missing:
                field['pressure'][9:] = 0.
            outputs = []
            for controller in (base, coupled):
                controller.observe(poses, field)
                outputs.append(controller.command(12., np.array([8., 20.]), .02))
            for i in (0, 1):
                np.testing.assert_array_equal(outputs[0][i], outputs[1][i])
            self.assert_records_equal(outputs[0][2], outputs[1][2])

    def test_rigid_world_transform_equivariance_before_lift(self):
        _, original = self.pair()
        _, transformed = self.pair()
        poses, field = self.observation((20., -30.), [[.02, .01, .01], [-.01, -.02, -.03]])
        turn = Rotation.from_euler('xyz', [27., -38., 62.], degrees=True)
        shift = np.array([.17, -.21, .14])
        moved = poses.copy()
        moved[:, :3] = turn.apply(poses[:, :3]) + shift
        moved[:, 3:] = (turn * Rotation.from_quat(poses[:, 3:])).as_quat()
        original.observe(poses, field)
        transformed.observe(moved, field)
        a, va, ra = original.command(12., np.array([8., 20.]), .02)
        b, vb, rb = transformed.command(12., np.array([8., 20.]), .02)
        np.testing.assert_allclose(b[:, :3], turn.apply(a[:, :3]) + shift, atol=1e-12)
        np.testing.assert_allclose((Rotation.from_quat(b[:, 3:]) *
                                   (turn * Rotation.from_quat(a[:, 3:])).inv()).magnitude(), 0., atol=1e-12)
        np.testing.assert_allclose(vb[:, :3], turn.apply(va[:, :3]), atol=1e-12)
        np.testing.assert_allclose(vb[:, 3:], turn.apply(va[:, 3:]), atol=1e-12)
        np.testing.assert_array_equal(rb['distance'], ra['distance'])

    def test_yaw_lift_and_contact_pivot_are_preserved(self):
        base, coupled = self.pair(yaw_delta_deg=35., lateral_xy=(.03, .05))
        poses, field = self.observation((3., -4.), [[.02, .01, .01], [-.01, -.02, -.03]])
        outputs = []
        for controller in (base, coupled):
            controller.lift_start = 24.
            controller.ready_seconds = 1.2
            controller.motion_elapsed = 2.
            controller.previous_lift = .5
            controller.previous_yaw = 17.5
            controller.observe(poses, field)
            outputs.append(controller.command(27., np.array([14., 18.]), .02))
        a, va, ra = outputs[0]
        b, vb, rb = outputs[1]
        self.assert_records_equal(ra, rb)
        self.assertGreater(rb['motion_elapsed_s'], 2.)
        self.assertTrue(rb['alignment_active'].all())
        axis = (poses[1, :3] + [-.01, -.02, -.03]) - (poses[0, :3] + [.02, .01, .01])
        axis /= np.linalg.norm(axis)
        difference = (np.stack((axis, -axis)) - ra['fit_normal_w']) * ra['distance'][:, None]
        turn = Rotation.from_euler('z', base.previous_yaw - 17.5, degrees=True)
        np.testing.assert_allclose(b[:, :3] - a[:, :3], turn.apply(difference), atol=1e-15)
        np.testing.assert_array_equal(b[:, 3:], a[:, 3:])
        np.testing.assert_array_equal(vb[:, 3:], va[:, 3:])
        np.testing.assert_allclose(vb[:, :3], (b[:, :3] - poses[:, :3]) / .02, atol=1e-14)

    def test_only_observed_fields_are_consumed_and_revision_is_explicit(self):
        _, a = self.pair()
        _, b = self.pair()
        poses, field = self.observation((20., -30.))
        a.observe(poses, field)
        with_truth = dict(field, object_pose_w=np.full(7, np.nan), mass_kg=-999.,
                          validation_full_hand_load_n=np.full(2, np.nan))
        b.observe(poses, with_truth)
        self.assertEqual(set(b.surface), {'pos', 'area', 'pressure', 'pad', 'patch'})
        x = a.command(12., np.array([8., 20.]), .02)
        y = b.command(12., np.array([8., 20.]), .02)
        for i in (0, 1):
            np.testing.assert_array_equal(x[i], y[i])
        self.assert_records_equal(x[2], y[2])
        with self.assertRaises(ValueError):
            CoupledSurfaceGripController(controller_revision='legacy')

    def test_uses_only_this_step_distance_increment_and_preserves_limits(self):
        base, coupled = self.pair()
        poses, field = self.observation((20., -30.))
        before = np.array([.199999, -.019999])
        outputs = []
        for controller in (base, coupled):
            controller.distance = before.copy()
            controller.observe(poses, field)
            outputs.append(controller.command(12., np.array([8., 20.]), .02))
        a, _, ra = outputs[0]
        b, _, rb = outputs[1]
        self.assert_records_equal(ra, rb)
        np.testing.assert_array_equal(rb['distance'], [.20, -.02])
        expected = (np.array([[1., 0., 0.], [-1., 0., 0.]]) - ra['fit_normal_w']) * (
            ra['distance'] - before)[:, None]
        np.testing.assert_allclose(b[:, :3] - a[:, :3], expected, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
