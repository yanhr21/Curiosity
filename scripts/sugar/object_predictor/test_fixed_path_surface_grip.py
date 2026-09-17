"""CPU diagnostic tests; no physics or evidence of successful transport."""
import unittest

import numpy as np

from .fixed_path_surface_grip import FixedPathSurfaceGripController
from .surface_grip_scene import SurfaceGripController
from . import test_surface_feedback_revision as reference_fixture


class FixedPathTests(unittest.TestCase):
    def controller(self):
        return FixedPathSurfaceGripController(target_load_n=16., response_gain=True)

    def observe(self, controller, loads=(16., 16.), tilt=0.):
        return reference_fixture.SurfaceFeedbackRevisionTests().observe(controller, loads, tilt)

    def test_initial_admission_still_requires_geometry_load_angle_and_full_second(self):
        controller = self.controller()
        for frame in range(90):
            loads = (0., 0.) if frame < 30 else ((.5, .5) if frame < 60 else (16., 16.))
            tilt = 6. if frame >= 60 else 0.
            self.observe(controller, loads, tilt)
            _, _, record = controller.command(24. + .02 * frame, np.array(loads), .02)
            self.assertEqual(controller.lift_start, -1.)
            self.assertEqual(record['actualmotion_rate'], 0.)
        for frame in range(49):
            self.observe(controller)
            controller.command(25.8 + .02 * frame, np.array([16., 16.]), .02)
            self.assertEqual(controller.lift_start, -1.)
        self.observe(controller)
        controller.command(26.78, np.array([16., 16.]), .02)
        self.assertEqual(controller.lift_start, 26.78)
        self.assertEqual(controller.motion_elapsed, 0.)

    def test_uninterrupted_four_second_path_matches_legacy_lift_and_yaw(self):
        fixed = self.controller()
        legacy = SurfaceGripController(target_load_n=16., response_gain=True)
        for controller in (fixed, legacy):
            controller.lift_start = 24.
            controller.ready_seconds = 1.
            controller.yaw_delta_deg = 20.
        for frame in range(201):
            time = 24. + .02 * frame
            records = []
            for controller in (fixed, legacy):
                self.observe(controller)
                records.append(controller.command(time, np.array([16., 16.]), .02)[2])
            self.assertAlmostEqual(records[0]['motion_elapsed_s'], records[1]['motion_elapsed_s'], places=12)
            self.assertAlmostEqual(records[0]['lift_command_m'], records[1]['lift_command_m'], places=12)
            self.assertAlmostEqual(fixed.previous_yaw, legacy.previous_yaw, places=12)
        self.assertEqual(fixed.motion_elapsed, 4.)
        self.assertEqual(fixed.lift_complete, 28.)

    def test_bad_runtime_support_advances_without_rewriting_readiness_or_alignment(self):
        controller = self.controller()
        controller.lift_start = 24.
        controller.motion_elapsed = 1.
        controller.previous_lift = .5 * (1. - np.cos(np.pi / 4.))
        poses, _ = self.observe(controller, loads=(8., 8.), tilt=15.)
        moved, _, record = controller.command(26., np.array([8., 8.]), .02)
        self.assertAlmostEqual(controller.motion_elapsed, 1.02)
        self.assertEqual(record['ready_seconds'], 0.)
        self.assertFalse(record['sustained_ready'])
        self.assertTrue(record['motion_ready'])
        self.assertFalse(record['motion_paused'])
        self.assertAlmostEqual(record['actualmotion_rate'], 1.)
        self.assertTrue(record['alignment_active'].all())
        self.assertTrue((record['distance'] > 0).all())
        self.assertGreater(np.linalg.norm(moved[:, 3:] - poses[:, 3:]), 0.)
        self.observe(controller, loads=(0., 0.))
        _, _, absent = controller.command(26.02, np.zeros(2), .02)
        self.assertFalse(absent['fit_valid'].any())
        self.assertFalse(absent['sustained_ready'])
        self.assertAlmostEqual(controller.motion_elapsed, 1.04)

    def test_complete_path_never_restarts_and_records_actual_clipped_rate(self):
        controller = self.controller()
        controller.lift_start = 24.
        controller.motion_elapsed = 3.99
        self.observe(controller, loads=(0., 0.))
        _, _, last = controller.command(28., np.zeros(2), .02)
        self.assertEqual(controller.motion_elapsed, 4.)
        self.assertAlmostEqual(last['actualmotion_rate'], .5)
        self.assertFalse(last['motion_paused'])
        self.observe(controller, loads=(0., 0.))
        _, _, completed = controller.command(28.02, np.zeros(2), .02)
        self.assertEqual(completed['actualmotion_rate'], 0.)
        self.assertFalse(completed['motion_paused'])
        self.assertEqual(controller.lift_complete, 28.)
        with self.assertRaises(ValueError):
            FixedPathSurfaceGripController(controller_revision='legacy')


if __name__ == '__main__':
    unittest.main()
