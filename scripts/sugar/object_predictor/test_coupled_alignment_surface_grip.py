"""Pure CPU controller geometry tests; no Newton stepping or model execution."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .coupled_alignment_surface_grip import CoupledAlignmentSurfaceGripController
from .surface_grip_scene import SurfaceGripController


class CoupledAlignmentTests(unittest.TestCase):
    def pair(self, **kwargs):
        common = dict(controller_revision='sensor_feedback_v2', target_load_n=16.,
                      force_gain=.000025, response_gain=True, **kwargs)
        pair = SurfaceGripController(**common), CoupledAlignmentSurfaceGripController(**common)
        for controller in pair:
            controller.support_normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
        return pair

    def observation(self, angles=(20., -30.)):
        poses = np.array([[-.2, 0., .3, 0., 0., 0., 1.],
                          [.2, 0., .3, 0., 0., 0., 1.]])
        centers = np.array([[.02, .01, .01], [-.01, -.02, -.03]])
        points = []
        for side, angle in enumerate(angles):
            points.extend(Rotation.from_euler('z', angle, degrees=True).apply(
                [[0., y, z] for y in (-.002, 0., .002) for z in (-.002, 0., .002)]) + centers[side])
        field = dict(pos=np.asarray(points), area=np.full(18, 1e-6),
                     pressure=np.full(18, 1e6), pad=np.repeat([0, 27], 9),
                     patch=np.repeat([0, 1], 9))
        return poses, field, centers

    def assert_original_records_equal(self, original, intervention):
        for key, value in original.items():
            np.testing.assert_array_equal(value, intervention[key], err_msg=key)

    def test_preserves_real_contact_pivot_and_original_plane_closure(self):
        base, candidate = self.pair()
        poses, field, centers = self.observation()
        outputs = []
        for controller in (base, candidate):
            controller.distance = np.array([.012, -.007])
            controller.observe(poses, field)
            outputs.append(controller.command(12., np.array([8., 20.]), .02))
        before, _, original_record = outputs[0]
        after, _, record = outputs[1]
        self.assert_original_records_equal(original_record, record)
        self.assertTrue(record['common_alignment_applied'].all())
        delta = record['distance'] - [.012, -.007]
        expected = poses[:, :3] + centers + record['fit_normal_w'] * delta[:, None]
        pivot_after = after[:, :3] + Rotation.from_quat(after[:, 3:]).apply(centers)
        np.testing.assert_allclose(pivot_after, expected, atol=1e-15)
        self.assertGreater(np.max(abs(after[:, 3:] - before[:, 3:])), 1e-5)
        angular_step = (Rotation.from_quat(after[:, 3:]) *
                        Rotation.from_quat(poses[:, 3:]).inv()).magnitude()
        self.assertTrue((angular_step <= np.deg2rad(3.) * .02 + 1e-14).all())

    def test_yaw_lift_pivot_and_velocity_remain_consistent(self):
        base, candidate = self.pair(yaw_delta_deg=35., lateral_xy=(.03, .05))
        poses, field, centers = self.observation((3., -4.))
        outputs = []
        for controller in (base, candidate):
            controller.lift_start = 24.
            controller.ready_seconds = 1.2
            controller.motion_elapsed = 2.
            controller.previous_lift = .5
            controller.previous_yaw = 17.5
            controller.observe(poses, field)
            outputs.append(controller.command(27., np.array([14., 18.]), .02))
        _, _, original_record = outputs[0]
        after, velocity, record = outputs[1]
        self.assert_original_records_equal(original_record, record)
        self.assertGreater(record['motion_elapsed_s'], 2.)
        turn = Rotation.from_euler('z', candidate.previous_yaw - 17.5, degrees=True)
        midpoint = poses[:, :3].mean(0)
        expected = turn.apply(poses[:, :3] + centers + record['fit_normal_w'] *
                              record['distance'][:, None] - midpoint) + midpoint
        expected += np.r_[candidate.lateral_xy, candidate.lift_height_m] * (candidate.previous_lift - .5)
        np.testing.assert_allclose(after[:, :3] + Rotation.from_quat(after[:, 3:]).apply(centers),
                                   expected, atol=1e-15)
        np.testing.assert_allclose(velocity[:, :3], (after[:, :3] - poses[:, :3]) / .02, atol=1e-14)
        np.testing.assert_allclose(velocity[:, 3:], (Rotation.from_quat(after[:, 3:]) *
                                   Rotation.from_quat(poses[:, 3:]).inv()).as_rotvec() / .02, atol=1e-14)

    def test_rigid_world_transform_equivariance_before_lift(self):
        _, original = self.pair()
        _, transformed = self.pair()
        poses, field, _ = self.observation()
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
        np.testing.assert_allclose(rb['common_alignment_axis_w'], turn.apply(ra['common_alignment_axis_w']), atol=1e-12)

    def test_common_target_alignment_cannot_fake_local_surface_readiness(self):
        base, candidate = self.pair()
        poses, field, _ = self.observation((20., -20.))
        outputs = []
        for controller in (base, candidate):
            controller.ready_seconds = 1.2
            controller.observe(poses, field)
            outputs.append(controller.command(24., np.array([16., 16.]), .02))
        original_record, record = outputs[0][2], outputs[1][2]
        self.assert_original_records_equal(original_record, record)
        np.testing.assert_allclose(record['common_alignment_target_error_deg'], 0., atol=1e-5)
        np.testing.assert_allclose(record['alignment_error_deg'], [20., 20.], atol=1e-12)
        self.assertEqual(record['ready_seconds'], 0.)
        self.assertEqual(record['lift_start_s'], -1.)
        self.assertFalse(record['motion_ready'])

    def test_missing_or_degenerate_geometry_and_undefined_axis_use_parent(self):
        for mode in ('unobserved', 'missing_side', 'degenerate', 'equal_normals'):
            base, candidate = self.pair()
            poses, field, _ = self.observation((0., 0.))
            if mode == 'missing_side':
                field['pressure'][9:] = 0.
            elif mode == 'degenerate':
                field['pos'][9:] = 0.
            elif mode == 'equal_normals':
                for controller in (base, candidate):
                    controller.support_normals[1] = [1., 0., 0.]
            outputs = []
            for controller in (base, candidate):
                if mode != 'unobserved':
                    controller.observe(poses, field)
                outputs.append(controller.command(12., np.array([8., 20.]), .02))
            for index in (0, 1):
                np.testing.assert_array_equal(outputs[0][index], outputs[1][index], err_msg=mode)
            self.assert_original_records_equal(outputs[0][2], outputs[1][2])
            self.assertFalse(outputs[1][2]['common_alignment_valid'])
            self.assertFalse(outputs[1][2]['common_alignment_applied'].any())

    def test_opposed_planes_preserve_original_commands(self):
        base, candidate = self.pair()
        poses, field, _ = self.observation((20., 20.))
        outputs = []
        for controller in (base, candidate):
            controller.observe(poses, field)
            outputs.append(controller.command(12., np.array([8., 20.]), .02))
        for index in (0, 1):
            np.testing.assert_allclose(outputs[0][index], outputs[1][index], atol=1e-14)
        self.assert_original_records_equal(outputs[0][2], outputs[1][2])

    def test_no_object_truth_consumed_and_revision_is_explicit(self):
        _, a = self.pair()
        _, b = self.pair()
        poses, field, _ = self.observation()
        a.observe(poses, field)
        b.observe(poses, dict(field, object_pose_w=np.full(7, np.nan), mass_kg=-999.,
                             validation_full_hand_load_n=np.full(2, np.nan)))
        self.assertEqual(set(b.surface), {'pos', 'area', 'pressure', 'pad', 'patch'})
        x = a.command(12., np.array([8., 20.]), .02)
        y = b.command(12., np.array([8., 20.]), .02)
        for index in (0, 1):
            np.testing.assert_array_equal(x[index], y[index])
        self.assert_original_records_equal(x[2], y[2])
        with self.assertRaises(ValueError):
            CoupledAlignmentSurfaceGripController(controller_revision='legacy')


if __name__ == '__main__':
    unittest.main()
