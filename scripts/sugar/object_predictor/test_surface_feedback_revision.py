"""CPU controller behavior tests; no simulator stepping or model execution.

Synthetic observed planes isolate readiness, feedback and motion-clock behavior.
They do not qualify grasp stability or claim a successful physical rollout.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .surface_grip_scene import SurfaceGripController


class SurfaceFeedbackRevisionTests(unittest.TestCase):
    def controller(self, revision='sensor_feedback_v2', **kwargs):
        return SurfaceGripController(controller_revision=revision, target_load_n=16.,
                                     force_gain=.000025, **kwargs)

    def observe(self, controller, loads=(16., 16.), tilt=0., collinear=False):
        poses = np.array([[-.2, 0., .3, 0., 0., 0., 1.],
                          [.2, 0., .3, 0., 0., 0., 1.]])
        samples = []
        for side in (0, 1):
            normal = controller.support_normals[side]
            tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
            tangent /= np.linalg.norm(tangent)
            normal = Rotation.from_rotvec(tangent * np.deg2rad(tilt)).apply(normal)
            second = np.cross(normal, tangent)
            points = np.array([x * tangent + y * second
                               for x in (-.002, 0., .002)
                               for y in ((0., 0., 0.) if collinear else (-.002, 0., .002))])
            samples.append(points)
        field = dict(pos=np.concatenate(samples), area=np.full(18, 1e-6),
                     pressure=np.repeat(np.asarray(loads) / 9e-6, 9),
                     pad=np.repeat([0, 27], 9), patch=np.repeat([0, 1], 9))
        controller.observe(poses, field)
        return poses, field

    def start_motion(self, controller):
        for i in range(60):
            self.observe(controller)
            _, _, record = controller.command(24. + i * .02, np.array([16., 16.]), .02)
        self.assertGreaterEqual(controller.lift_start, 24.)
        self.assertGreater(controller.motion_elapsed, 0.)
        return record

    def test_low_load_geometry_admission_does_not_relax_lift_readiness(self):
        revised = self.controller(response_gain=True)
        legacy = self.controller('legacy', response_gain=True)
        for controller in (revised, legacy):
            self.observe(controller, loads=(.5, .5), tilt=20.)
        _, _, new = revised.command(24., np.array([.5, .5]), .02)
        _, _, old = legacy.command(24., np.array([.5, .5]), .02)
        self.assertTrue(new['fit_valid'].all())
        self.assertFalse(old['fit_valid'].any())
        self.assertTrue(new['alignment_active'].all())
        self.assertEqual(revised.ready_seconds, 0.)
        self.assertEqual(revised.lift_start, -1.)

    def test_unresolved_or_absent_surface_cannot_enable_plane(self):
        for loads, collinear in (((.5, .5), True), ((0., 0.), False)):
            controller = self.controller(response_gain=True)
            self.observe(controller, loads, collinear=collinear)
            _, _, record = controller.command(24., np.array(loads), .02)
            self.assertFalse(record['fit_valid'].any())
            self.assertFalse(record['motion_ready'])
            self.assertEqual(controller.lift_start, -1.)

    def test_pause_is_immediate_but_force_and_alignment_remain_active(self):
        controller = self.controller(response_gain=True)
        before = self.start_motion(controller)
        poses, _ = self.observe(controller, loads=(8., 8.), tilt=15.)
        moved, _, paused = controller.command(25.2, np.array([8., 8.]), .02)
        self.assertTrue(paused['motion_paused'])
        self.assertEqual(paused['motion_elapsed_s'], before['motion_elapsed_s'])
        self.assertEqual(paused['lift_command_m'], before['lift_command_m'])
        self.assertTrue(paused['alignment_active'].all())
        self.assertTrue((paused['distance'] > before['distance']).all())
        self.assertGreater(np.linalg.norm(moved[:, 3:] - poses[:, 3:]), 0.)
        self.assertTrue(np.isfinite(paused['response_gain_m_per_ns']).all())

    def test_alignment_loss_alone_pauses_motion_at_target_load(self):
        controller = self.controller(response_gain=True)
        before = self.start_motion(controller)
        self.observe(controller, tilt=6.)
        _, _, paused = controller.command(25.2, np.array([16., 16.]), .02)
        self.assertTrue(paused['fit_valid'].all())
        self.assertTrue((paused['alignment_error_deg'] > 5.).all())
        self.assertTrue(paused['motion_paused'])
        self.assertEqual(paused['motion_elapsed_s'], before['motion_elapsed_s'])
        self.assertEqual(paused['ready_seconds'], 0.)
        self.assertTrue(paused['alignment_active'].all())

    def test_missing_contact_geometry_pauses_even_if_aggregate_load_is_high(self):
        controller = self.controller(response_gain=True)
        before = self.start_motion(controller)
        poses, field = self.observe(controller)
        field['pressure'][:] = 0.
        controller.observe(poses, field)
        _, _, paused = controller.command(25.2, np.array([16., 16.]), .02)
        self.assertFalse(paused['fit_valid'].any())
        self.assertTrue(paused['motion_paused'])
        self.assertEqual(paused['motion_elapsed_s'], before['motion_elapsed_s'])

    def test_contact_pivot_feedback_is_world_frame_equivariant_before_lift(self):
        original = self.controller(response_gain=True)
        transformed = self.controller(response_gain=True)
        poses, field = self.observe(original, loads=(.5, .5), tilt=20.)
        rotation = Rotation.from_euler('xyz', [25., -32., 61.], degrees=True)
        translation = np.array([.11, -.09, .18])
        world = poses.copy()
        world[:, :3] = rotation.apply(poses[:, :3]) + translation
        world[:, 3:] = (rotation * Rotation.from_quat(poses[:, 3:])).as_quat()
        transformed.observe(world, field)
        a, va, ra = original.command(12., np.array([.5, .5]), .02)
        b, vb, rb = transformed.command(12., np.array([.5, .5]), .02)
        np.testing.assert_allclose(b[:, :3], rotation.apply(a[:, :3]) + translation, atol=1e-12)
        expected = rotation * Rotation.from_quat(a[:, 3:])
        error = Rotation.from_quat(b[:, 3:]) * expected.inv()
        np.testing.assert_allclose(error.magnitude(), 0., atol=1e-12)
        np.testing.assert_allclose(vb[:, :3], rotation.apply(va[:, :3]), atol=1e-12)
        np.testing.assert_allclose(vb[:, 3:], rotation.apply(va[:, 3:]), atol=1e-12)
        np.testing.assert_allclose(rb['fit_normal_w'], rotation.apply(ra['fit_normal_w']), atol=1e-12)
        np.testing.assert_allclose(rb['alignment_error_deg'], ra['alignment_error_deg'], atol=1e-12)
        # The original three-degree/second rate cap is retained in both frames.
        change = Rotation.from_quat(a[:, 3:]) * Rotation.from_quat(poses[:, 3:]).inv()
        self.assertTrue((change.magnitude() <= np.deg2rad(3.) * .02 + 1e-12).all())

    def test_resume_requires_full_second_and_never_catches_up_wall_clock(self):
        controller = self.controller(response_gain=True)
        before = self.start_motion(controller)
        self.observe(controller, loads=(0., 0.))
        controller.command(25.2, np.zeros(2), .02)
        elapsed = controller.motion_elapsed
        for i in range(49):
            self.observe(controller)
            controller.command(25.22 + i * .02, np.array([16., 16.]), .02)
        self.assertEqual(controller.motion_elapsed, elapsed)
        self.observe(controller)
        _, _, resumed = controller.command(26.2, np.array([16., 16.]), .02)
        self.assertAlmostEqual(controller.motion_elapsed, elapsed + .02)
        self.assertFalse(resumed['motion_paused'])
        self.assertGreater(resumed['lift_command_m'], before['lift_command_m'])

    def test_completed_path_never_restarts_on_later_loss_of_readiness(self):
        controller = self.controller(response_gain=True)
        self.start_motion(controller)
        for i in range(250):
            self.observe(controller)
            _, _, record = controller.command(25.2 + i * .02, np.array([16., 16.]), .02)
        self.assertEqual(controller.motion_elapsed, 4.)
        complete = controller.lift_complete
        self.observe(controller, loads=(0., 0.))
        _, _, record = controller.command(30.2, np.zeros(2), .02)
        self.assertEqual(controller.lift_complete, complete)
        self.assertEqual(record['phase'], 2)
        self.assertFalse(record['motion_paused'])
        self.assertEqual(record['lift_command_m'], controller.lift_height_m)

    def test_legacy_behavior_and_explicit_variant_conflicts(self):
        default = SurfaceGripController(target_load_n=16., force_gain=.000025, response_gain=True)
        explicit = self.controller('legacy', response_gain=True)
        for i in range(80):
            loads = np.array([.5, .5]) if i < 10 else np.array([16., 16.])
            for controller in (default, explicit):
                self.observe(controller, loads)
            a = default.command(24. + i * .02, loads, .02)
            b = explicit.command(24. + i * .02, loads, .02)
            for actual, expected in zip(a[:2], b[:2]):
                np.testing.assert_array_equal(actual, expected)
            for name in a[2]:
                np.testing.assert_array_equal(a[2][name], b[2][name])
        for option in ('sustained_feedback', 'alignment_deadband', 'normalized_acquisition'):
            with self.assertRaises(ValueError):
                self.controller(response_gain=True, **{option: True})
        with self.assertRaises(ValueError):
            self.controller('legacy', response_gain=True, sustained_feedback=True)
        with self.assertRaises(ValueError):
            self.controller('unknown')

    def test_legacy_still_uses_wall_clock_and_stops_alignment_after_lift(self):
        controller = self.controller('legacy', response_gain=True)
        for i in range(60):
            self.observe(controller)
            _, _, before = controller.command(24. + i * .02, np.array([16., 16.]), .02)
        self.assertGreaterEqual(controller.lift_start, 24.)
        self.observe(controller, loads=(8., 8.), tilt=15.)
        _, _, after = controller.command(25.2, np.array([8., 8.]), .02)
        self.assertGreater(after['motion_elapsed_s'], before['motion_elapsed_s'])
        self.assertFalse(after['alignment_active'].any())


if __name__ == '__main__':
    unittest.main()
