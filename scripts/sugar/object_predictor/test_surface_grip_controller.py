"""Geometric feedback invariants; synthetic surfaces are tests, not training data."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .surface_grip_scene import SurfaceGripController


class SurfaceGripControllerTests(unittest.TestCase):
    def setup_surface(self, tilt):
        controller = SurfaceGripController(target_load_n=24.)
        poses, _, _ = controller.command(0., np.zeros(2), .02)
        centers = np.array([[.11, -.04, -.03], [.11, .04, -.03]])
        points = []
        for side in (0, 1):
            n = Rotation.from_rotvec([tilt, 0, 0]).apply(controller.support_normals[side])
            tangent = np.cross(n, [1., 0, 0]); tangent /= np.linalg.norm(tangent)
            second = np.cross(n, tangent)
            points.extend([centers[side] + x * tangent + y * second
                           for x in (-.001, 0., .001) for y in (-.001, .001)])
        field = dict(pos=np.array(points), area=np.full(12, 1e-6), pressure=np.full(12, 4e6),
                     patch=np.repeat([0, 1], 6), pad=np.repeat([26, 53], 6))
        controller.observe(poses, field)
        return controller, poses, centers

    def test_rotation_preserves_observed_contact_and_rate_limit(self):
        controller, old, centers = self.setup_surface(.3)
        new, velocity, record = controller.command(15., np.full(2, 24.), .02)
        old_rotation = Rotation.from_quat(old[:, 3:])
        new_rotation = Rotation.from_quat(new[:, 3:])
        np.testing.assert_allclose(old_rotation.apply(centers) + old[:, :3],
                                   new_rotation.apply(centers) + new[:, :3], atol=1e-12)
        self.assertTrue(record['fit_valid'].all())
        self.assertTrue(np.isfinite(velocity).all())
        angles = np.degrees((new_rotation * old_rotation.inv()).magnitude())
        self.assertTrue((angles <= 3. * .02 + 1e-10).all())
        self.assertTrue((angles > 0).all())
        self.assertEqual(record['lift_start_s'], -1.)

    def test_no_forced_lift_without_resolved_contact(self):
        controller, poses, _ = self.setup_surface(0.)
        controller.surface['pressure'][:] = 0.
        _, _, record = controller.command(40., np.full(2, 24.), .02)
        self.assertFalse(record['fit_valid'].any())
        self.assertEqual(record['lift_start_s'], -1.)
        self.assertEqual(record['lift_command_m'], 0.)

    def test_force_gain_sets_observed_normal_displacement(self):
        controller, old, _ = self.setup_surface(0.)
        controller.force_gain = .000025
        new, _, record = controller.command(15., np.full(2, 12.), .02)
        expected = record['fit_normal_w'] * (.000025 * 12. * .02)
        np.testing.assert_allclose(new[:, :3] - old[:, :3], expected, atol=1e-12)

    def test_integral_is_bounded_and_resets_without_contact(self):
        controller, old, _ = self.setup_surface(0.)
        controller.force_gain = .000025
        controller.force_integral_time_s = 2.
        controller.command(15., np.full(2, 12.), .02)
        np.testing.assert_allclose(controller.integral_velocity, .000025 / 2 * 12 * .02)
        controller.integral_velocity[:] = .002
        controller.command(15.02, np.full(2, 12.), .02)
        np.testing.assert_allclose(controller.integral_velocity, .002)
        controller.surface['pressure'][:] = 0.
        controller.command(15.04, np.zeros(2), .02)
        np.testing.assert_array_equal(controller.integral_velocity, 0.)

    def test_sustained_feedback_pauses_and_requires_stable_recovery(self):
        controller, _, _ = self.setup_surface(0.)
        controller.sustained_feedback = True
        controller.lift_start = 24.
        controller.motion_elapsed = .8
        controller.command(25., np.full(2, 12.), .02)
        self.assertEqual(controller.motion_elapsed, .8)
        for index in range(49):
            _, _, record = controller.command(25.02 + index * .02, np.full(2, 24.), .02)
        self.assertEqual(controller.motion_elapsed, .8)
        self.assertTrue(record['motion_paused'])
        _, _, record = controller.command(26., np.full(2, 24.), .02)
        self.assertAlmostEqual(controller.motion_elapsed, .82)
        self.assertFalse(record['motion_paused'])
        self.assertEqual(record['lift_complete_s'], -1.)
        self.assertEqual(record['phase'], 1)

    def test_postlift_alignment_is_opt_in(self):
        controller, old, _ = self.setup_surface(.3)
        controller.lift_start = 24.
        controller.yaw_delta_deg = 0.
        original, _, _ = controller.command(25., np.full(2, 24.), .02)
        np.testing.assert_array_equal(original[:, 3:], old[:, 3:])
        controller.sustained_feedback = True
        aligned, _, _ = controller.command(25.02, np.full(2, 24.), .02)
        angles = np.degrees((Rotation.from_quat(aligned[:, 3:]) * Rotation.from_quat(old[:, 3:]).inv()).magnitude())
        self.assertTrue((angles > 0).all())
        self.assertTrue((angles <= .06 + 1e-10).all())

    def test_paused_motion_cannot_be_declared_complete_by_wall_clock(self):
        controller, _, _ = self.setup_surface(0.)
        controller.sustained_feedback = True
        controller.lift_start = 24.
        controller.surface['pressure'][:] = 0.
        _, _, record = controller.command(48., np.zeros(2), .02)
        self.assertEqual(record['phase'], 1)
        self.assertEqual(record['lift_complete_s'], -1.)
        self.assertEqual(record['lift_command_m'], 0.)

    def test_alignment_deadband_preserves_early_acquisition(self):
        original, _, _ = self.setup_surface(.04)
        diagnostic, _, _ = self.setup_surface(.04)
        diagnostic.alignment_deadband = True
        for time in (15., 23.98):
            a, av, _ = original.command(time, np.full(2, 12.), .02)
            b, bv, _ = diagnostic.command(time, np.full(2, 12.), .02)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(av, bv)

    def test_alignment_deadband_stops_rotation_but_keeps_force_control(self):
        controller, old, _ = self.setup_surface(.04)
        controller.alignment_deadband = True
        controller.force_gain = .000025
        new, _, record = controller.command(24., np.full(2, 12.), .02)
        self.assertTrue(record['fit_valid'].all())
        self.assertTrue((record['alignment_error_deg'] < 5.).all())
        self.assertFalse(record['alignment_active'].any())
        np.testing.assert_array_equal(new[:, 3:], old[:, 3:])
        np.testing.assert_allclose(new[:, :3] - old[:, :3],
                                   record['fit_normal_w'] * (.000025 * 12. * .02), atol=1e-12)
        self.assertEqual(record['lift_start_s'], -1.)

    def test_alignment_deadband_keeps_rotation_outside_readiness_cone(self):
        original, _, _ = self.setup_surface(.3)
        diagnostic, _, _ = self.setup_surface(.3)
        diagnostic.alignment_deadband = True
        a, av, _ = original.command(24., np.full(2, 12.), .02)
        b, bv, record = diagnostic.command(24., np.full(2, 12.), .02)
        self.assertTrue(record['alignment_active'].all())
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(av, bv)

    def test_normalized_acquisition_preserves_nearband_and_postlift_commands(self):
        for time,lift_start,load in [(15.,-1.,9.),(15.,-1.,12.),(15.,-1.,18.),(25.,24.,3.)]:
            original,_,_=self.setup_surface(0.)
            diagnostic,_,_=self.setup_surface(0.)
            for controller in (original,diagnostic):
                controller.target_load_n=12.;controller.force_gain=.000025;controller.lift_start=lift_start
            diagnostic.normalized_acquisition=True
            a,av,_=original.command(time,np.full(2,load),.02)
            b,bv,record=diagnostic.command(time,np.full(2,load),.02)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(av,bv)
            np.testing.assert_array_equal(record['acquisition_gain_multiplier'],1.)

    def test_normalized_acquisition_preserves_24n_and_free_approach(self):
        for target,load in [(24.,0.),(24.,.2),(24.,12.),(24.,18.),(24.,30.),(12.,0.)]:
            original,_,_=self.setup_surface(0.)
            diagnostic,_,_=self.setup_surface(0.)
            for controller in (original,diagnostic):
                controller.target_load_n=target;controller.force_gain=.000025
            diagnostic.normalized_acquisition=True
            a,av,_=original.command(15.,np.full(2,load),.02)
            b,bv,_=diagnostic.command(15.,np.full(2,load),.02)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(av,bv)

    def test_normalized_acquisition_advances_weak_contact_without_relaxing_readiness(self):
        original,old,_=self.setup_surface(0.)
        diagnostic,_,_=self.setup_surface(0.)
        for controller in (original,diagnostic):
            controller.target_load_n=12.;controller.force_gain=.000025
        diagnostic.normalized_acquisition=True
        a,_,_=original.command(30.,np.full(2,6.),.02)
        b,_,record=diagnostic.command(30.,np.full(2,6.),.02)
        np.testing.assert_allclose(b[:,:3]-old[:,:3],2*(a[:,:3]-old[:,:3]),atol=1e-12)
        self.assertEqual(record['lift_start_s'],-1.)
        self.assertEqual(record['ready_seconds'],0.)
        np.testing.assert_array_equal(record['acquisition_gain_multiplier'],2.)


if __name__ == '__main__':
    unittest.main()
