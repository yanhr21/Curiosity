"""CPU-only tests of time, force signs, and supervision/input separation."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .overfit_data import (
    OBSERVATION_KEYS, INPUT_KEYS, input_conflicts, mass_supervision, observation_window,
    observed_physics_targets, uniform_history_indices,
)


CRITERIA = dict(clearance_m=.01, bilateral_load_n=.01,
                max_acceleration_m_s2=.5, max_angular_speed_rad_s=.2)


def trace():
    n = 32
    pose = np.zeros((n, 7)); pose[:, 2] = .2; pose[:, 6] = 1
    return dict(timestamp_s=(np.arange(n) + 1) * .02,
                object_pose_w=pose, object_com_w=pose[:, :3].copy(), object_local_center_m=np.zeros((n, 3)),
                object_mass_kg=np.full(n, .4),
                normal_load_n=np.full((n, 54), 2 / 27),
                validation_full_mesh_min_z_m=np.full(n, .15))


def encoded_forces():
    feat = np.zeros((3, 20), dtype=np.float64)
    feat[:, 9] = 1
    feat[:, 6:9] = [[1, 0, 0], [-1, 0, 0], [-1, 0, 0]]
    feat[:, 10] = np.log1p([2, 1, 1])
    feat[:, 11:14] = np.arcsinh(np.array([[0, 0, -1], [0, 0, -2], [0, 0, -1]]) / 2)
    feat[:, 14] = [.5, .25, .25]
    feat[:, 16] = [0, 1, 1]
    feat[:, 17:20] = [0, 0, -1]
    return feat


class OverfitDataTest(unittest.TestCase):
    def test_uniform_grid_is_causal_and_keeps_physical_interval(self):
        times = (np.arange(2400) + 1) * .02
        for frame in (31, 1181, 2381):
            indices = uniform_history_indices(times, frame, 32, .02)
            np.testing.assert_array_equal(indices, np.arange(frame - 31, frame + 1))
            np.testing.assert_allclose(np.diff(times[indices]), .02, atol=1e-12)
        indices = uniform_history_indices(times, 2381, 32, .1)
        np.testing.assert_array_equal(np.diff(indices), 5)
        self.assertEqual(indices[-1], 2381)

    def test_no_padded_missing_or_irregular_time_grid(self):
        times = (np.arange(2400) + 1) * .02
        for frame, interval in ((31, .1), (2381, .03)):
            with self.assertRaises(ValueError):
                uniform_history_indices(times, frame, 32, interval)
        broken = times.copy(); broken[2365] += .001
        with self.assertRaises(ValueError):
            uniform_history_indices(broken, 2381, 32, .02)

    def test_force_sign_and_conservation_are_physical(self):
        values = observed_physics_targets(encoded_forces())
        np.testing.assert_allclose(values['normal_load_n'], [4])
        np.testing.assert_allclose(values['normal_load_by_encoded_side_n'], [2, 2])
        np.testing.assert_allclose(values['contact_area_m2'], [1e-4])
        np.testing.assert_allclose(values['shear_on_hand_current_frame_n'], [0, 0, -4])
        np.testing.assert_allclose(values['normal_on_object_approx_current_frame_n'], [0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(values['object_resultant_approx_current_frame_n'], [0, 0, 4], atol=1e-7)
        np.testing.assert_allclose(values['shear_support_up_n'], [4])
        np.testing.assert_allclose(values['combined_support_up_approx_n'], [4])

    def test_force_targets_transform_with_hand_frame(self):
        feat = encoded_forces()
        rotation = Rotation.from_rotvec([.4, -.2, .7]).as_matrix()
        rotated = feat.copy()
        rotated[:, 6:9] = feat[:, 6:9] @ rotation
        rotated[:, 11:14] = np.arcsinh((2 * np.sinh(feat[:, 11:14]) @ rotation) / 2)
        rotated[:, 17:20] = feat[:, 17:20] @ rotation
        before, after = observed_physics_targets(feat), observed_physics_targets(rotated)
        for key in ('shear_on_hand_current_frame_n', 'normal_on_object_approx_current_frame_n',
                    'object_resultant_approx_current_frame_n'):
            np.testing.assert_allclose(after[key], before[key] @ rotation, atol=1e-6)
        for key in ('normal_load_n', 'contact_area_m2', 'shear_support_up_n', 'combined_support_up_approx_n'):
            np.testing.assert_allclose(after[key], before[key], atol=1e-6)

    def test_unavailability_is_not_filtered_and_not_based_on_mass_accuracy(self):
        arrays = trace(); ix = np.arange(32)
        available = mass_supervision(arrays, ix, CRITERIA)
        self.assertEqual(available['mass_available'], 1)
        self.assertEqual(available['mass_status'], 3)
        arrays['object_mass_kg'] *= 100  # Does not change any availability criterion.
        self.assertEqual(mass_supervision(arrays, ix, CRITERIA)['mass_available'], 1)
        arrays['normal_load_n'][:] = 0
        unavailable = mass_supervision(arrays, ix, CRITERIA)
        self.assertEqual(unavailable['mass_available'], 0)
        self.assertEqual(unavailable['mass_uncertain'], 1)
        self.assertEqual(unavailable['mass_status'], 0)

    def test_ground_support_contact_loss_and_motion_cannot_label_mass(self):
        arrays = trace(); ix = np.arange(32)
        arrays['validation_full_mesh_min_z_m'][0] = 0
        self.assertEqual(mass_supervision(arrays, ix, CRITERIA)['mass_status'], 1)
        arrays = trace(); arrays['normal_load_n'][12, 27:] = 0
        self.assertEqual(mass_supervision(arrays, ix, CRITERIA)['mass_status'], 1)
        arrays = trace(); arrays['object_com_w'][:, 0] = arrays['timestamp_s'] ** 2
        moving = mass_supervision(arrays, ix, CRITERIA)
        self.assertEqual(moving['mass_available'], 0)
        self.assertEqual(moving['mass_status'], 2)
        arrays = trace(); arrays['object_com_w'][16, 0] = .001
        self.assertEqual(mass_supervision(arrays, ix, CRITERIA)['mass_available'], 0)

    def test_actual_com_is_required_and_aabb_center_is_never_substituted(self):
        arrays = trace(); ix = np.arange(32)
        arrays['object_pose_w'][:, 0] = arrays['timestamp_s'] ** 2
        arrays['object_local_center_m'][:] = 100
        self.assertEqual(mass_supervision(arrays, ix, CRITERIA)['mass_available'], 1)
        del arrays['object_com_w']
        labels = mass_supervision(arrays, ix, CRITERIA)
        self.assertEqual(labels['mass_available'], 0)
        self.assertEqual(labels['mass_status'], 4)
        self.assertFalse(labels['evaluation_only']['qualified'])

    def test_angular_motion_excludes_mass_but_constant_translation_does_not(self):
        arrays = trace(); ix = np.arange(32)
        arrays['object_com_w'][:, 0] = arrays['timestamp_s'] * .2
        self.assertEqual(mass_supervision(arrays, ix, CRITERIA)['mass_available'], 1)
        arrays['object_pose_w'][:, 3:] = Rotation.from_rotvec(
            np.column_stack((np.zeros((32, 2)), arrays['timestamp_s'] * .3))).as_quat()
        self.assertEqual(mass_supervision(arrays, ix, CRITERIA)['mass_status'], 2)

    def test_validation_and_targets_are_excluded_from_observation_window(self):
        arrays = {key: np.arange(8, dtype=float) for key in OBSERVATION_KEYS}
        arrays.update(object_mass_kg=np.full(8, .4), validation_full_mesh_min_z_m=np.full(8, .2))
        first = observation_window(arrays, [1, 3, 7])
        arrays['object_mass_kg'][:] = 100
        arrays['validation_full_mesh_min_z_m'][:] = -100
        second = observation_window(arrays, [1, 3, 7])
        self.assertEqual(set(first), set(OBSERVATION_KEYS))
        for key in OBSERVATION_KEYS:
            np.testing.assert_array_equal(first[key], second[key])

    def test_conflicting_targets_are_reported_without_dropping_samples(self):
        rows = []
        for episode, mass in ((5000, .4), (5001, .8)):
            rows.append(dict(inputs={key: [np.ones((3, 3))] for key in INPUT_KEYS},
                             target=np.r_[np.zeros(12), np.log(mass)],
                             supervision=dict(mass_available=0),
                             metadata=dict(episode=episode, frame=31)))
        conflicts = input_conflicts(rows)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['row_indices'], [0, 1])
        self.assertTrue(conflicts[0]['mass_target_conflict'])
        self.assertFalse(conflicts[0]['state_target_conflict'])
        self.assertEqual(conflicts[0]['mass_available'], [0, 0])
        self.assertEqual(len(rows), 2)
        rows[1]['target'][0] = .1
        self.assertTrue(input_conflicts(rows)[0]['state_target_conflict'])


if __name__ == '__main__':
    unittest.main()
