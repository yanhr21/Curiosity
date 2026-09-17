"""Meaningful CPU loss/sampler/gate checks; never construct a backbone/model."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager

import numpy as np
import torch

from .overfit_data import FIXED_EPISODES, FIXED_FRAMES
from .overfit_model import masked_mass_loss, state_vertices, symmetric_mesh_distance
from .train_overfit import (INTERPOLATION_FRAMES, acceptance, availability_statistics,
                           REQUIRED_CONTROLLER_CHECKS, completion_exit_code, fixed_batches,
                           parameter_update_report, qualification, require_active_resource,
                           REQUIRED_ARTIFACTS, SOURCE_MODULES, save_json, sha256_file, verify_artifacts)
from . import retained_execution


def states(count):
    values = torch.zeros(count, 13, dtype=torch.float64)
    values[:, 3] = values[:, 7] = 1.
    return values


def perfect_arrays(frames):
    episodes = np.repeat(FIXED_EPISODES, len(frames))
    clocks = np.tile(frames, len(FIXED_EPISODES))
    available = (clocks > 1500).astype(float)
    return dict(episode=episodes, frame=clocks, center_cm=np.zeros(len(clocks)),
        mesh_nn_cm=np.zeros(len(clocks)), rotation_deg=np.zeros(len(clocks)),
        size_mean_relative=np.zeros(len(clocks)), size_max_relative=np.zeros(len(clocks)),
        mass_relative=np.zeros(len(clocks)), force_rmse_n=np.zeros(len(clocks)),
        mass_available=available, availability_probability=np.where(available, .99, .01))


def qualified_dataset_fixture():
    class Sized:
        def __len__(self): return len(self.rows)
    dataset = Sized()
    dataset.rows = [dict(metadata=dict(episode=episode, frame=frame), target=np.zeros(13),
                        supervision=dict(mass_available=float(frame in (1581, 2381))))
                    for episode in FIXED_EPISODES for frame in FIXED_FRAMES]
    dataset.conflicts = []; dataset.qualified = True
    dataset.collection_records = {episode: dict(episode=episode, complete=True, controller_passed=True,
        controller_checks={key: True for key in REQUIRED_CONTROLLER_CHECKS}) for episode in FIXED_EPISODES}
    dataset.collection_result = dict(complete=True, qualification_passed=True,
                                     records=list(dataset.collection_records.values()))
    return dataset


@contextmanager
def active_resource_fixture(state='RUNNING', *, observed_job='298134', missing_lock=False):
    """Mock only a temporary resource record/fd; never touch the real fd9/lock."""
    with TemporaryDirectory() as temporary:
        root = Path(temporary); resource = root / 'ACTIVE_RESOURCE.json'; lock = root / 'lock'
        save_json(resource, dict(state=state, job_id='298134', step_id='0', host='server31'))
        lock.touch()
        with patch.object(retained_execution, 'RESOURCE', resource), patch.object(retained_execution, 'LOCK', lock), \
             patch.dict(retained_execution.os.environ, SLURM_JOB_ID=observed_job, SLURM_STEP_ID='0'), \
             patch.object(retained_execution.socket, 'gethostname', return_value='server31'), \
             patch.object(retained_execution.os, 'fstat', side_effect=OSError('missing fd9') if missing_lock else None,
                          return_value=lock.stat()) as descriptor, \
             patch.object(retained_execution.fcntl, 'flock') as exclusive:
            yield descriptor, exclusive


class OverfitTrainingTest(unittest.TestCase):
    def test_mass_mask_removes_conflicting_target_precision_gradient(self):
        pred = states(3).requires_grad_()
        target = states(3); target[:, 12] = torch.tensor([1000., -1000., .1])
        available = torch.tensor([0., 0., 1.])
        loss = masked_mass_loss(pred, target, available)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(pred.grad[:2].abs().sum()), 0.)
        self.assertGreater(float(pred.grad[2, 12].abs()), 0.)
        target[:2, 12] = -target[:2, 12]
        self.assertEqual(float(masked_mass_loss(pred, target, available)), float(loss))
        zero = masked_mass_loss(pred, target, torch.zeros(3))
        self.assertEqual(float(zero), 0.)

    def test_nearest_geometry_accepts_mesh_symmetry_without_gt_rotation_fix(self):
        points = torch.tensor([[-1., -2., 0.], [1., -2., 0.], [-1., 2., 0.], [1., 2., 0.]], dtype=torch.float64)[None]
        rotated = points * torch.tensor([-1., -1., 1.])
        self.assertGreater(float((rotated - points).norm(dim=-1).mean()), 1.)
        self.assertEqual(float(symmetric_mesh_distance(rotated, points)), 0.)

    def test_geometric_nearest_distance_has_correct_translation_gradient(self):
        points = torch.tensor([[0., 0., 0.], [2., .3, .1], [.4, 3., .2]], dtype=torch.float64)[None]
        shift = torch.tensor([.03, .04, 0.], dtype=torch.float64, requires_grad=True)
        loss = symmetric_mesh_distance(points + shift, points).mean()
        loss.backward()
        np.testing.assert_allclose(float(loss), .05, atol=1e-12)
        np.testing.assert_allclose(shift.grad.numpy(), [.6, .8, 0.], atol=1e-12)

    def test_state_geometry_uses_meter_dimensions_and_center(self):
        state = states(1); state[0, :3] = torch.tensor([1., 2., 3.])
        state[0, 9:12] = torch.log(torch.tensor([.2, .4, .6], dtype=torch.float64))
        vertices = torch.tensor([[-.5, -.5, -.5], [.5, .5, .5]], dtype=torch.float64)
        output = state_vertices(state, vertices)[0]
        np.testing.assert_allclose(output.numpy(), [[.9, 1.8, 2.7], [1.1, 2.2, 3.3]], atol=1e-12)

    def test_every_epoch_covers_all_fixed_items_once(self):
        rows = [dict(metadata=dict(episode=episode, frame=frame)) for episode in FIXED_EPISODES for frame in FIXED_FRAMES]
        iterator = fixed_batches(SimpleNamespace(rows=rows), 123)
        for _ in range(3):
            batches = [next(iterator) for _ in range(20)]
            self.assertEqual(sorted(i for batch in batches for i in batch), list(range(80)))
            for batch in batches:
                self.assertEqual(len({rows[i]['metadata']['frame'] for i in batch}), 1)
                self.assertEqual(len({rows[i]['metadata']['episode'] for i in batch}), 4)

    def test_fit_gate_cannot_hide_bad_unavailable_state_or_mass_coverage(self):
        values = perfect_arrays(FIXED_FRAMES)
        self.assertTrue(acceptance(values, 'fit')['passed'])
        values['center_cm'][0] = 1.01  # unavailable no-contact example is still assessed.
        self.assertFalse(acceptance(values, 'fit')['passed'])
        values['center_cm'][0] = 0
        values['mass_available'][:5] = 0
        values['availability_probability'][:5] = .01
        self.assertFalse(acceptance(values, 'fit')['passed'])

    def test_interpolation_gate_reports_failure_despite_perfect_fit(self):
        values = perfect_arrays(INTERPOLATION_FRAMES)
        self.assertTrue(acceptance(values, 'same_trajectory_interpolation')['passed'])
        values['mesh_nn_cm'][:len(INTERPOLATION_FRAMES)] = 1.1
        self.assertFalse(acceptance(values, 'same_trajectory_interpolation')['passed'])

    def test_always_unavailable_does_not_pass_balanced_accuracy(self):
        report = availability_statistics(np.zeros(100), np.r_[np.zeros(90), np.ones(10)])
        self.assertEqual(report['accuracy'], .9)
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertEqual(report['false_confident_fraction'], 0.)

    def test_data_gate_requires_a_late_mass_target_for_every_configuration(self):
        dataset = qualified_dataset_fixture()
        self.assertTrue(qualification(dataset)['passed'])
        for row in dataset.rows[:5]: row['supervision']['mass_available'] = 0.
        self.assertFalse(qualification(dataset)['passed'])

    def test_physical_fail_stays_blocked_despite_every_late_mass_label(self):
        dataset = qualified_dataset_fixture()
        self.assertTrue(qualification(dataset)['passed'])
        dataset.collection_result['qualification_passed'] = False
        self.assertFalse(qualification(dataset)['passed'])
        dataset.collection_result['qualification_passed'] = True
        dataset.collection_records[5000]['controller_checks']['peak'] = False
        self.assertFalse(qualification(dataset)['passed'])
        dataset.collection_records[5000]['controller_checks']['peak'] = True
        del dataset.collection_records[5000]['controller_checks']['hold_drift']
        self.assertFalse(qualification(dataset)['passed'])

    def test_every_active_backbone_module_must_show_a_real_parameter_delta(self):
        # Synthetic parameter-delta fixture only: no backbone/forward/optimizer
        # step is constructed or executed to exercise the saved-update gate.
        module = torch.nn.Module(); module.predictor = torch.nn.Module()
        module.predictor.backbone = torch.nn.Module()
        backbone = module.predictor.backbone
        backbone.first = torch.nn.Linear(2, 2)
        backbone.second = torch.nn.Linear(2, 2)
        backbone.embedding = torch.nn.Module()
        backbone.embedding.mask_token = torch.nn.Parameter(torch.zeros(54))
        initial = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
        optimizer = SimpleNamespace(param_groups=[dict(params=list(module.parameters()))], state={})
        for name, parameter in module.named_parameters():
            if name.endswith('mask_token'): continue
            optimizer.state[parameter] = dict(step=torch.tensor(2000), exp_avg=torch.zeros_like(parameter),
                                             exp_avg_sq=torch.zeros_like(parameter))
        with torch.no_grad(): backbone.first.weight.add_(.01)
        report = parameter_update_report(module, optimizer, initial, 2000)
        self.assertTrue(report['checks']['complete_backbone_really_changed'])
        self.assertFalse(report['checks']['every_active_official_module_really_changed'])
        self.assertFalse(report['passed'])
        with torch.no_grad(): backbone.second.weight.add_(.01)
        self.assertTrue(parameter_update_report(module, optimizer, initial, 2000)['passed'])

    def test_failed_scientific_acceptance_is_nonzero_exit(self):
        self.assertEqual(completion_exit_code(dict(execution_complete=True, acceptance_passed=False)), 2)
        self.assertEqual(completion_exit_code(dict(execution_complete=True, acceptance_passed=True)), 0)

    def test_pending_resource_cannot_launch_training(self):
        with active_resource_fixture(state='PENDING') as (descriptor, exclusive):
            with self.assertRaisesRegex(RuntimeError, 'not recorded RUNNING'):
                require_active_resource()
            descriptor.assert_not_called(); exclusive.assert_not_called()

    def test_wrong_resource_cannot_launch_training(self):
        with active_resource_fixture(observed_job='297947') as (descriptor, exclusive):
            with self.assertRaisesRegex(RuntimeError, 'bound retained compute'):
                require_active_resource()
            descriptor.assert_not_called(); exclusive.assert_not_called()

    def test_missing_serial_lock_cannot_launch_training(self):
        with active_resource_fixture(missing_lock=True):
            with self.assertRaisesRegex(RuntimeError, 'exclusive GPU lock on fd9'):
                require_active_resource()
        with active_resource_fixture() as (descriptor, exclusive):
            self.assertEqual(require_active_resource()['job_id'], '298134')
            descriptor.assert_called_once_with(9)
            exclusive.assert_called_once_with(9, retained_execution.fcntl.LOCK_EX | retained_execution.fcntl.LOCK_NB)

    def test_renderer_manifest_rejects_one_changed_prediction_binding(self):
        # Tiny byte fixtures exercise provenance only, never a model or render.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source_fixture.py'; source.write_text('fixture = True\n')
            (root / 'PROTOCOL.json').write_text('{}\n')
            for name in REQUIRED_ARTIFACTS: (root / name).write_bytes(name.encode())
            manifest = dict(schema=1, complete=True,
                sources={name: dict(path=str(source), sha256=sha256_file(source)) for name in SOURCE_MODULES},
                protocol=dict(path='PROTOCOL.json', sha256=sha256_file(root / 'PROTOCOL.json')),
                artifacts={name: dict(path=name, sha256=sha256_file(root / name)) for name in REQUIRED_ARTIFACTS})
            save_json(root / 'ARTIFACTS.json', manifest)
            self.assertTrue(verify_artifacts(root)['complete'])
            (root / 'initial_fit.npz').write_bytes(b'changed prediction fixture')
            with self.assertRaisesRegex(ValueError, 'initial_fit.npz'):
                verify_artifacts(root)


if __name__ == '__main__':
    unittest.main()
