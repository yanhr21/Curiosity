"""CPU supervision/scope tests; no model or physical trajectory is executed."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from scripts.sugar.object_predictor.conditional_overfit_data import (
    PROFILE, STUDY, check_collection_scope, state_history_evidence,
    validate_controlled_sources)
from scripts.sugar.object_predictor.conditional_overfit_training import (
    conditional_task_losses, conditional_acceptance)
from scripts.sugar.object_predictor.overfit_data import FIXED_FRAMES, FIXED_EPISODES, input_conflicts
from scripts.sugar.object_predictor.overfit_model import task_losses
from scripts.sugar.object_predictor.train_overfit import (
    FIT_LIMITS, INTERPOLATION_LIMITS, INTERPOLATION_FRAMES, acceptance, qualification)
from scripts.sugar.object_predictor.test_overfit_training import qualified_dataset_fixture, perfect_arrays


def encoded(index, contact=False):
    frames = [np.zeros((1, 20), np.float32) for _ in range(32)]
    for frame in frames:
        frame[0, 0] = index
    if contact:
        frames[0][0, 9] = 1.  # Earlier real evidence still belongs to this input.
    return dict(feat=frames, coord=[f[:, :3].copy() for f in frames],
                grid_coord=[np.zeros((1, 3), np.int32) for _ in frames])


def conditional_dataset_fixture():
    dataset = qualified_dataset_fixture()
    dataset.supervision_profile = PROFILE
    dataset.collection_result['study'] = STUDY
    dataset.controlled_source_validation = dict(passed=True)
    for record in dataset.collection_records.values():
        record['late_mass_available'] = True
    for i, row in enumerate(dataset.rows):
        row['inputs'] = encoded(i, contact=row['metadata']['frame'] != 31)
        row['supervision'].update(state_history_evidence(row['inputs']))
    return dataset


def loss_fixture():
    state = torch.zeros((2, 13), dtype=torch.float64)
    state[:, 3] = state[:, 7] = 1.
    state[:, 0] = .02
    state[:, 9] = .03
    target = state.clone()
    target[:, 0] = 0.
    target[:, 9] = 0.
    state.requires_grad_()
    output = dict(state=state, force=torch.zeros(2, 8, dtype=torch.float64),
        availability_logit=torch.zeros(2, dtype=torch.float64), contact_logit=torch.zeros(2, dtype=torch.float64))
    batch = dict(target=target, state_precision_eligible=torch.ones(2, dtype=torch.float64),
        mass_available=torch.zeros(2, dtype=torch.float64), physics=dict(
        normal_load_by_encoded_side_n=torch.zeros(2, 2, dtype=torch.float64),
        shear_on_hand_current_frame_n=torch.zeros(2, 3, dtype=torch.float64),
        normal_on_object_approx_current_frame_n=torch.zeros(2, 3, dtype=torch.float64),
        contact_present=torch.zeros(2, 1, dtype=torch.float64)))
    # Merely a geometry-function test fixture, never a replacement object/model.
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
    return output, batch, vertices


class ConditionalOverfitTests(unittest.TestCase):
    def test_evidence_uses_whole_input_history_and_no_target(self):
        inputs = encoded(0)
        original = deepcopy(inputs)
        self.assertEqual(state_history_evidence(inputs)['state_precision_eligible'], 0.)
        inputs['feat'][0][0, 9] = 1.
        evidence = state_history_evidence(inputs)
        self.assertEqual(evidence['state_precision_eligible'], 1.)
        self.assertEqual(evidence['state_contact_history_frames'], 1)
        inputs['feat'][0][0, 9] = 0.
        for key in inputs:
            for actual, expected in zip(inputs[key], original[key]):
                np.testing.assert_array_equal(actual, expected)

    def test_masked_state_targets_cannot_contribute_precision_gradient_or_overflow(self):
        output, batch, vertices = loss_fixture()
        batch['state_precision_eligible'][0] = 0.
        batch['target'][0, :3] = 1e30
        batch['target'][0, 9:12] = 1000.
        parts = conditional_task_losses(output, batch, vertices)
        loss = parts['center'] + parts['mesh'] + parts['size']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        np.testing.assert_array_equal(output['state'].grad[0].numpy(), np.zeros(13))
        self.assertGreater(float(output['state'].grad[1, :12].abs().sum()), 0.)
        self.assertEqual(set(parts), set(task_losses(*loss_fixture())))

    def test_all_eligible_losses_and_gradients_match_original_exactly(self):
        output, batch, vertices = loss_fixture()
        original = task_losses(output, batch, vertices)
        expected_gradient = torch.autograd.grad(sum(original.values()), output['state'])[0]
        changed = conditional_task_losses(output, batch, vertices)
        for key in original:
            torch.testing.assert_close(changed[key], original[key], rtol=0, atol=0)
        gradient = torch.autograd.grad(sum(changed.values()), output['state'])[0]
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)

    def test_no_contact_conflict_is_reported_but_contact_conflict_still_blocks(self):
        dataset = conditional_dataset_fixture()
        dataset.rows[5]['inputs'] = deepcopy(dataset.rows[0]['inputs'])
        dataset.rows[5]['target'][0] = .03
        dataset.conflicts = input_conflicts(dataset.rows)
        report = qualification(dataset)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['input_conflicts']), 1)
        self.assertEqual(report['state_precision_eligible_count'], 64)
        dataset.rows[6]['inputs'] = deepcopy(dataset.rows[1]['inputs'])
        dataset.rows[6]['target'][0] = .03
        dataset.conflicts = input_conflicts(dataset.rows)
        self.assertFalse(qualification(dataset)['passed'])

    def test_state_mask_tampering_late_mass_failure_and_wrong_scope_block(self):
        dataset = conditional_dataset_fixture()
        self.assertTrue(qualification(dataset)['passed'])
        dataset.rows[1]['supervision']['state_precision_eligible'] = 0.
        self.assertFalse(qualification(dataset)['passed'])
        dataset = conditional_dataset_fixture()
        dataset.collection_records[5000]['late_mass_available'] = False
        self.assertFalse(qualification(dataset)['passed'])
        dataset = conditional_dataset_fixture()
        dataset.supervision_profile = 'all_state'
        self.assertFalse(qualification(dataset)['passed'])

    def test_raw_errors_retained_and_every_contact_force_gate_still_applies(self):
        arrays = perfect_arrays(FIXED_FRAMES)
        arrays['state_precision_eligible'] = (arrays['frame'] != 31).astype(float)
        unknown = arrays['state_precision_eligible'] == 0
        arrays['center_cm'][unknown] = 100.
        arrays['mesh_nn_cm'][unknown] = 100.
        arrays['size_max_relative'][unknown] = 10.
        call = lambda: conditional_acceptance(arrays, 'fit', original_acceptance=acceptance,
            fit_limits=FIT_LIMITS, interpolation_limits=INTERPOLATION_LIMITS, episodes=FIXED_EPISODES)
        report = call()
        self.assertTrue(report['passed'])
        self.assertFalse(report['original_all_state_gate_passed'])
        self.assertEqual(report['state_candidate_rows'], 64)
        self.assertEqual(report['prior_unknown_rows'], 16)
        self.assertEqual(report['all_items']['center_cm']['maximum'], 100.)
        arrays['center_cm'][1] = 1.1
        self.assertFalse(call()['passed'])
        arrays['center_cm'][1] = 0.
        arrays['force_rmse_n'][0] = 1.
        self.assertFalse(call()['passed'])  # Unknown state never masks force.

    def test_dense_has_all_clocks_raw_metrics_and_cannot_pass_empty_eligibility(self):
        arrays = perfect_arrays(INTERPOLATION_FRAMES)
        arrays['state_precision_eligible'] = (arrays['frame'] > 1000).astype(float)
        arrays['center_cm'][arrays['state_precision_eligible'] == 0] = 100.
        call = lambda: conditional_acceptance(arrays, 'same_trajectory_interpolation',
            original_acceptance=acceptance, fit_limits=FIT_LIMITS,
            interpolation_limits=INTERPOLATION_LIMITS, episodes=FIXED_EPISODES)
        report = call()
        self.assertTrue(report['passed'])
        self.assertEqual(len(arrays['frame']), 1440)
        self.assertEqual(report['all_items']['center_cm']['maximum'], 100.)
        arrays['state_precision_eligible'][arrays['episode'] == 5000] = 0.
        self.assertFalse(call()['passed'])

    def test_default_and_conditional_profiles_reject_wrong_collection_scope(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('PROTOCOL.json', 'COLLECTION_RESULT.json'):
                (root/name).write_text(json.dumps(dict(study=STUDY)))
            check_collection_scope(root, PROFILE)
            with self.assertRaises(ValueError): check_collection_scope(root, 'all_state')
            (root/'COLLECTION_RESULT.json').write_text('{}')
            with self.assertRaises(ValueError): check_collection_scope(root, PROFILE)
            with self.assertRaises(ValueError): check_collection_scope(root, 'all_state')

    def test_controlled_source_identity_and_actual_configuration_mapping(self):
        from scripts.sugar.object_predictor.collect_controlled_fixture_corpus import protocol
        from scripts.sugar.object_predictor.run_canonical_approach_fixture_pilot import CHECKS, INTERVENTION
        with TemporaryDirectory() as temp:
            root = Path(temp); declared = protocol()
            (root/'PROTOCOL.json').write_text(json.dumps(declared))
            records = []
            for config in declared['configurations']:
                directory = root/'cases'/f"episode_{config['episode']}"
                directory.mkdir(parents=True)
                actual = {dict(mass='mass_kg', load='target_load_n').get(k, k): v for k, v in config.items()}
                actual.update(frames=2400, dt=.02, controller_intervention=INTERVENTION)
                (directory/'PROTOCOL.json').write_text(json.dumps(actual))
                checks = {k: True for k in CHECKS}
                (directory/'RESULT.json').write_text(json.dumps(dict(checks=checks, passed=True,
                    controller_intervention=INTERVENTION)))
                records.append(dict(episode=config['episode'], source=str(directory), split='train',
                    geometry_group=config['geometry_group'], controller_checks=checks, controller_passed=True,
                    frames=2400, late_mass_window=dict(frame=2381, first_frame=2350)))
            collection = dict(study=STUDY, scope=declared['scope'], records=records,
                              original_requested_approach_regression_repaired=False)
            path = root/'COLLECTION_RESULT.json'
            path.write_text(json.dumps(collection))
            self.assertTrue(validate_controlled_sources(root)['passed'])
            records[0]['source'] = records[1]['source']
            path.write_text(json.dumps(collection))
            with self.assertRaisesRegex(ValueError, 'source/identity'):
                validate_controlled_sources(root)
            records[0]['source'] = str(root/'cases'/'episode_5000')
            path.write_text(json.dumps(collection))
            actual_path = root/'cases'/'episode_5008'/'PROTOCOL.json'
            actual = json.loads(actual_path.read_text()); actual['approach_angle_deg'] = 60.
            actual_path.write_text(json.dumps(actual))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                validate_controlled_sources(root)


if __name__ == '__main__':
    unittest.main()
