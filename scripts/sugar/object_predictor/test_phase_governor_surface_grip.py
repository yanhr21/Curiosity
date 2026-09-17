"""CPU-only governor and inherited-controller regression tests."""
import hashlib
import unittest

import numpy as np

from .phase_governor_surface_grip import PhaseGovernorSurfaceGripController
from .surface_grip_scene import SurfaceGripController
from . import test_surface_feedback_revision as reference_fixture


def legacy_fingerprint(revision):
    """Bound original numeric trajectory, including every original record key."""
    controller = SurfaceGripController(controller_revision=revision, target_load_n=16.,
                                       force_gain=.000025, response_gain=True,
                                       yaw_delta_deg=18., lateral_xy=(.02, .04))
    helper = reference_fixture.SurfaceFeedbackRevisionTests()
    digest = hashlib.sha256()
    for frame in range(450):
        loads = np.array([.5, .5]) if frame < 5 else (
            np.array([8., 20.]) if 80 <= frame < 90 else np.array([16., 16.]))
        tilt = 15. if 80 <= frame < 90 else (6. if 180 <= frame < 185 else 0.)
        helper.observe(controller, loads=loads, tilt=tilt)
        poses, velocity, record = controller.command(24. + frame * .02, loads, .02)
        for name, value in [('poses', poses), ('velocity', velocity), *sorted(record.items())]:
            array = np.asarray(value)
            digest.update(name.encode()); digest.update(array.dtype.str.encode())
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


class PhaseGovernorTests(unittest.TestCase):
    def controller(self):
        return PhaseGovernorSurfaceGripController(target_load_n=12., response_gain=True)

    def rate(self, controller, loads=(12., 12.), ready=True, fit=(True, True), angles=(0., 0.)):
        return controller.motion_phase_rate(ready, np.array(fit), np.array(angles), np.array(loads), .02)

    def test_zero_outside_band_or_invalid_geometry(self):
        controller = self.controller()
        for kwargs in [dict(loads=(8.99, 12.)), dict(loads=(15.01, 12.)),
                       dict(fit=(True, False)), dict(angles=(0., 5.001)), dict(ready=False)]:
            controller.governor_rate = 1.
            self.assertEqual(self.rate(controller, **kwargs), 0.)

    def test_boundary_and_near_boundary_are_continuous_worst_hand_margin(self):
        controller = self.controller()
        for loads, expected in [((9., 12.), 0.), ((12., 15.), 0.),
                                ((9.000003, 12.), 1e-6), ((12., 14.999997), 1e-6),
                                ((10.5, 14.25), .25)]:
            controller.governor_rate = 1.
            self.assertAlmostEqual(self.rate(controller, loads), expected, places=12)

    def test_rise_is_two_per_second_and_drop_never_exceeds_raw(self):
        controller = self.controller()
        values = [self.rate(controller) for _ in range(30)]
        np.testing.assert_allclose(values, np.minimum(np.arange(1, 31) * .04, 1.), atol=1e-14)
        self.assertAlmostEqual(self.rate(controller, loads=(9.3, 12.)), .1)
        self.assertAlmostEqual(self.rate(controller), .14)
        self.assertEqual(self.rate(controller, loads=(8., 12.)), 0.)

    def test_initial_admission_retains_full_second(self):
        controller = self.controller()
        helper = reference_fixture.SurfaceFeedbackRevisionTests()
        for frame in range(49):
            helper.observe(controller, loads=(12., 12.))
            _, _, record = controller.command(24. + frame * .02, np.array([12., 12.]), .02)
            self.assertEqual(controller.lift_start, -1.)
            self.assertEqual(record['motion_rate'], 0.)
        helper.observe(controller, loads=(12., 12.))
        controller.command(24.98, np.array([12., 12.]), .02)
        self.assertEqual(controller.lift_start, 24.98)
        self.assertEqual(controller.motion_elapsed, 0.)

    def test_resume_uses_current_support_not_new_second_and_records_truth(self):
        controller = self.controller()
        helper = reference_fixture.SurfaceFeedbackRevisionTests()
        controller.lift_start = 24.
        controller.motion_elapsed = 1.
        controller.previous_lift = .5 * (1. - np.cos(np.pi / 4.))
        helper.observe(controller, loads=(8., 12.))
        _, _, stopped = controller.command(26., np.array([8., 12.]), .02)
        self.assertTrue(stopped['motion_paused'])
        self.assertEqual(controller.motion_elapsed, 1.)
        helper.observe(controller, loads=(12., 12.))
        _, _, resumed = controller.command(26.02, np.array([12., 12.]), .02)
        self.assertAlmostEqual(resumed['ready_seconds'], .02)
        self.assertFalse(resumed['sustained_ready'])
        self.assertTrue(resumed['motion_ready'])
        self.assertFalse(resumed['motion_paused'])
        self.assertAlmostEqual(resumed['motion_rate'], .04)
        self.assertAlmostEqual(controller.motion_elapsed, 1.0008)
        controller.motion_elapsed = 3.9999
        helper.observe(controller, loads=(12., 12.))
        _, _, completed = controller.command(26.04, np.array([12., 12.]), .02)
        self.assertEqual(controller.motion_elapsed, 4.)
        self.assertAlmostEqual(completed['motion_rate'], .005)
        self.assertFalse(completed['motion_paused'])

    def test_legacy_and_v2_numeric_records_bitexact(self):
        # Captured before introducing the parent motion_phase_rate hook.
        expected = {'legacy': '85e906e00de5c7dda36e36cfb7f0dc46798d7cea4a27193aaafd5fc7df3c05f5',
                    'sensor_feedback_v2': '4df26ef846837277bf9cf5755ccbd7dc4d33cc5116af40a94c10430688e3f58a'}
        for revision, fingerprint in expected.items():
            self.assertEqual(legacy_fingerprint(revision), fingerprint)


if __name__ == '__main__':
    unittest.main()
