"""CPU target/gate checks; no backbone, forward, optimizer, or physics."""
import unittest

import numpy as np

from scripts.sugar.object_predictor.overfit_data import input_conflicts
from scripts.sugar.object_predictor.overfit_supervision_conflicts import state_supervision_feasibility


def row(center=(0., 0., 0.), dimensions=(1., 1., 1.), rotation_scale=1., mass=1.):
    target = np.zeros(13, dtype=np.float64)
    target[:3] = center
    target[3] = target[7] = rotation_scale
    target[9:12] = np.log(dimensions)
    target[12] = np.log(mass)
    return dict(target=target, inputs={key: [np.zeros((1, 3), np.float32)]
                for key in ('coord', 'feat', 'grid_coord')},
                supervision=dict(mass_available=0.), metadata=dict(frame=31))


def inspect(rows):
    return state_supervision_feasibility(rows, input_conflicts(rows),
        center_limit_m=.01, size_max_relative_limit=.05)


class StateSupervisionFeasibilityTests(unittest.TestCase):
    def test_center_pair_impossibility_survives_unavailable_mass(self):
        rows = [row(), row(center=(.03, 0., 0.), mass=2.)]
        report = inspect(rows)
        self.assertFalse(report['passed'])
        witness = report['impossible_groups'][0]
        self.assertEqual(witness['center_pairwise_minimum_worst_error_m'], .015)
        self.assertTrue(witness['center_gate_provably_impossible'])
        self.assertFalse(witness['size_gate_provably_impossible'])
        # The midpoint attains the pairwise minimax bound, but misses the gate.
        errors = [np.linalg.norm(np.array([.015, 0., 0.])-r['target'][:3]) for r in rows]
        np.testing.assert_allclose(errors, [.015, .015], rtol=0, atol=1e-16)

    def test_size_uses_actual_max_axis_gate_not_mean_loss(self):
        report = inspect([row(), row(dimensions=(1.2, 1., 1.))])
        self.assertFalse(report['passed'])
        witness = report['impossible_groups'][0]
        bound = witness['size_axis_minimum_worst_relative_error'][0]
        self.assertAlmostEqual(bound, .2/2.2)
        self.assertLess(bound/3, .05)  # Mean loss would not justify this gate.
        common_dimension = 2*1.2/2.2
        self.assertAlmostEqual(common_dimension-1., bound)
        self.assertAlmostEqual(1.-common_dimension/1.2, bound)

    def test_different_targets_with_common_acceptable_prediction_are_not_rejected(self):
        rows = [row(), row(center=(.018, 0., 0.), dimensions=(1.1, 1., 1.))]
        self.assertTrue(input_conflicts(rows)[0]['state_target_conflict'])
        self.assertTrue(inspect(rows)['passed'])
        common_dimension = 2*1.1/2.1
        self.assertLess(max(abs(common_dimension-1.), abs(common_dimension/1.1-1.)), .05)

    def test_rotation_representation_and_unavailable_mass_do_not_prove_geometry_conflict(self):
        rows = [row(), row(rotation_scale=2., mass=2.)]
        report = inspect(rows)
        self.assertTrue(report['passed'])
        self.assertGreater(max(report['groups'][0]['raw_rotation6_spread']), 0.)
        # Positive scaling of each 6D basis vector has identical normalized R.
        for r in rows:
            basis = r['target'][3:9].reshape(2, 3)
            np.testing.assert_array_equal(basis/np.linalg.norm(basis, axis=1)[:, None],
                                          np.array([[1., 0., 0.], [0., 1., 0.]]))

    def test_only_exact_input_groups_are_checked_and_nonfinite_targets_rejected(self):
        rows = [row(), row(center=(1., 0., 0.))]
        rows[1]['inputs']['coord'][0][0, 0] = 1.
        self.assertTrue(inspect(rows)['passed'])
        rows[1]['inputs']['coord'][0][0, 0] = 0.
        groups = input_conflicts(rows)
        rows[1]['target'][0] = np.nan
        with self.assertRaises(ValueError):
            state_supervision_feasibility(rows, groups, center_limit_m=.01,
                                          size_max_relative_limit=.05)

    def test_real_qualification_rejects_state_conflict_without_relaxing_physics(self):
        from scripts.sugar.object_predictor.test_overfit_training import qualified_dataset_fixture
        from scripts.sugar.object_predictor.train_overfit import qualification
        dataset = qualified_dataset_fixture()
        self.assertTrue(qualification(dataset)['passed'])
        for i, r in enumerate(dataset.rows):
            r['inputs'] = {key: [np.array([[i, 0., 0.]], dtype=np.float32)]
                           for key in ('coord', 'feat', 'grid_coord')}
        dataset.rows[5]['inputs'] = dataset.rows[0]['inputs']
        dataset.rows[5]['target'][0] = .03
        dataset.conflicts = input_conflicts(dataset.rows)
        report = qualification(dataset)
        self.assertTrue(report['checks']['all_16_physical_qualifications_pass'])
        self.assertTrue(report['checks']['no_identical_input_different_available_mass'])
        self.assertFalse(report['checks']['no_provably_incompatible_identical_input_state'])
        self.assertFalse(report['passed'])
        dataset.rows[5]['target'][0] = .018
        dataset.conflicts = input_conflicts(dataset.rows)
        self.assertTrue(qualification(dataset)['passed'])
        dataset.collection_records[5000]['controller_checks']['peak'] = False
        self.assertFalse(qualification(dataset)['passed'])


if __name__ == '__main__':
    unittest.main()
