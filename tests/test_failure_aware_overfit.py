"""CPU-only scope/data/gate tests; no model or simulated trajectory execution."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.sugar.object_predictor.failure_aware_overfit import (
    PROFILE, OBSERVATION_SHAPES, validate_observation_arrays, failure_aware_acceptance,
    failure_aware_dense_qualification)
from scripts.sugar.object_predictor.overfit_data import (OBSERVATION_KEYS,
    FIXED_EPISODES, FIXED_FRAMES, input_conflicts)
from scripts.sugar.object_predictor.train_overfit import (qualification, acceptance,
    FIT_LIMITS, INTERPOLATION_LIMITS, INTERPOLATION_FRAMES)
from scripts.sugar.object_predictor.conditional_overfit_data import PROFILE as OLD_PROFILE
from scripts.sugar.object_predictor.conditional_overfit_training import conditional_acceptance
from scripts.sugar.object_predictor.test_overfit_training import perfect_arrays
from tests.test_conditional_overfit import conditional_dataset_fixture


def dataset_fixture():
    data = conditional_dataset_fixture()
    data.supervision_profile = PROFILE
    data.failure_source_validation = dict(passed=True)
    data.collection_result['qualification_passed'] = False
    record = data.collection_records[5014]
    record.update(controller_passed=False, late_mass_available=False)
    record['controller_checks']['hold_drift'] = False
    for row in data.rows:
        if row['metadata']['episode'] == 5014:
            row['supervision']['mass_available'] = 0.
    return data


def arrays_fixture(frames):
    arrays = perfect_arrays(frames)
    arrays['state_precision_eligible'] = (arrays['frame'] > 31).astype(float)
    failed = arrays['episode'] == 5014
    arrays['mass_available'][failed] = 0.
    arrays['availability_probability'][failed] = .01
    arrays['mass_relative'][failed] = 100.
    unknown = arrays['state_precision_eligible'] == 0
    arrays['center_cm'][unknown] = 100.
    return arrays


def gate(arrays, role, function=failure_aware_acceptance):
    return function(arrays, role, original_acceptance=acceptance,
        fit_limits=FIT_LIMITS, interpolation_limits=INTERPOLATION_LIMITS,
        episodes=FIXED_EPISODES)


class FailureAwareTests(unittest.TestCase):
    def test_scientific_failure_admitted_without_relabeling_old_physical_gate(self):
        data = dataset_fixture()
        result = qualification(data)
        self.assertTrue(result['learnable_data_gate_passed'])
        self.assertFalse(result['physical_success_gate_passed'])
        self.assertFalse(result['original_conditional_qualification_passed'])
        self.assertEqual(result['total_rows'], 80)
        self.assertEqual(result['per_case']['5014']['available_mass_denominator'], 0)
        self.assertIsNone(result['per_case']['5014']['mass_precision_passed'])
        self.assertFalse(data.collection_result['qualification_passed'])
        data.supervision_profile = OLD_PROFILE
        self.assertFalse(qualification(data)['passed'])

    def test_runtime_incomplete_schema_quality_and_fresh_mask_still_block(self):
        edits = (
            lambda d: d.collection_records[5014].update(complete=False),
            lambda d: d.collection_records[5014]['controller_checks'].pop('peak'),
            lambda d: d.failure_source_validation.update(passed=False),
            lambda d: d.rows[1]['supervision'].update(state_precision_eligible=0.),
            lambda d: d.rows.pop(),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                data = dataset_fixture(); edit(data)
                self.assertFalse(qualification(data)['passed'])

    def test_contact_target_conflict_still_blocks_and_no_case_id_disambiguation(self):
        data = dataset_fixture()
        data.rows[6]['inputs'] = deepcopy(data.rows[1]['inputs'])
        data.rows[6]['target'][0] = .03
        data.conflicts = input_conflicts(data.rows)
        self.assertFalse(qualification(data)['passed'])

    def test_dense_pre_model_qualification_rejects_missing_clock_and_candidate_conflict(self):
        data = dataset_fixture()
        # This unit fixture uses two declared synthetic evaluation clocks only;
        # the actual runner always supplies the unchanged 1440-clock grid.
        data.rows = [r for r in data.rows if r['metadata']['frame'] in (1181, 1281)]
        call = lambda: failure_aware_dense_qualification(data, frames=(1181, 1281), fit_limits=FIT_LIMITS)
        self.assertTrue(call()['passed'])
        data.rows[2]['inputs'] = deepcopy(data.rows[0]['inputs'])
        data.rows[2]['target'][0] = .03
        data.conflicts = input_conflicts(data.rows)
        self.assertFalse(call()['passed'])
        data.conflicts = []
        data.rows.pop()
        self.assertFalse(call()['passed'])

    def test_fit_mass_na_keeps_all80_errors_and_force_availability_gates(self):
        arrays = arrays_fixture(FIXED_FRAMES)
        result = gate(arrays, 'fit')
        self.assertTrue(result['passed'])
        self.assertFalse(gate(arrays, 'fit', conditional_acceptance)['passed'])
        failed = result['per_case']['5014']
        self.assertEqual(failed['mass_precision_status'], 'NOT_APPLICABLE')
        self.assertIsNone(failed['checks']['mass'])
        self.assertIsNone(failed['mass_precision_passed'])
        self.assertEqual(result['mass_cases_with_available_targets'], 15)
        self.assertEqual(result['all_items']['mass_relative']['maximum'], 100.)
        self.assertEqual(len(arrays['frame']), 80)
        i = int(np.flatnonzero(arrays['episode'] == 5014)[0])
        arrays['availability_probability'][i] = .9
        self.assertFalse(gate(arrays, 'fit')['passed'])
        arrays['availability_probability'][i] = .01
        arrays['force_rmse_n'][i] = .51
        self.assertFalse(gate(arrays, 'fit')['passed'])

    def test_dense_single_class_specificity_and_global_balanced_have_real_limits(self):
        arrays = arrays_fixture(INTERPOLATION_FRAMES)
        result = gate(arrays, 'same_trajectory_interpolation')
        self.assertTrue(result['passed'])
        self.assertEqual(len(arrays['frame']), 1440)
        self.assertIsNone(result['per_case']['5014']['checks']['mass_mean'])
        self.assertTrue(result['per_case']['5014']['checks']['availability_specificity'])
        self.assertTrue(result['checks']['global/availability_balanced'])
        failed = np.flatnonzero(arrays['episode'] == 5014)
        arrays['availability_probability'][failed[:5]] = .9  # 5/90 >5%, global diluted.
        self.assertFalse(gate(arrays, 'same_trajectory_interpolation')['passed'])
        arrays['availability_probability'][:] = .01
        self.assertFalse(gate(arrays, 'same_trajectory_interpolation')['checks']['global/availability_balanced'])

    def test_available_precision_numbers_unchanged_and_empty_mass_cannot_claim_success(self):
        arrays = arrays_fixture(FIXED_FRAMES)
        i = int(np.flatnonzero(arrays['mass_available'] > .5)[0])
        arrays['mass_relative'][i] = .05001
        self.assertFalse(gate(arrays, 'fit')['passed'])
        arrays['mass_relative'][i] = .05
        self.assertTrue(gate(arrays, 'fit')['passed'])
        arrays['mass_available'][:] = 0.
        arrays['availability_probability'][:] = .01
        self.assertFalse(gate(arrays, 'fit')['passed'])

    def test_whole_recording_quality_rejects_corrupt_unsampled_clock_and_field(self):
        arrays = {key: np.zeros(shape) for key,shape in OBSERVATION_SHAPES.items()}
        arrays['timestamp_s'] = .02*np.arange(1, 2401)
        field = dict(offset=np.arange(2401), area_m2=np.ones(2400),
            normal_pressure_pa=np.zeros(2400), position_hand_frame_m=np.zeros((2400, 3)),
            shear_traction_hand_frame_pa=np.zeros((2400, 3)),
            pad=np.zeros(2400, int), hand=np.zeros(2400, int))
        self.assertTrue(validate_observation_arrays(arrays, field)['passed'])
        field['pad'][1] = 53; field['hand'][1] = 1
        self.assertTrue(validate_observation_arrays(arrays, field)['passed'])
        field['hand'][1] = 0
        with self.assertRaises(ValueError): validate_observation_arrays(arrays, field)
        field['hand'][1] = 1
        arrays['normal_load_n'][1999] = np.nan  # Neither fit nor dense inference clock.
        with self.assertRaises(ValueError): validate_observation_arrays(arrays, field)
        arrays['normal_load_n'][1999] = 0
        field['offset'][-1] -= 1
        with self.assertRaises(ValueError): validate_observation_arrays(arrays, field)

    def test_renderer_preserves_new_scope_and_rejects_a_different_collection(self):
        from scripts.sugar.object_predictor.render_state_overfit import validate_state_evidence
        from tests.test_render_state_overfit import fixture
        protocol, arms = fixture(); protocol['supervision_profile'] = PROFILE
        self.assertTrue(validate_state_evidence(protocol, arms))
        protocol['collection_study'] = 'controlled_fixture16_gainfreeze_v1'
        with self.assertRaises(ValueError): validate_state_evidence(protocol, arms)


if __name__ == '__main__':
    unittest.main()
