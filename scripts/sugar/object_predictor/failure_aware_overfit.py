"""Explicit learning qualification on complete successful AND failed attempts.

This changes neither the archived physical result nor the conditional objective.
No physical failure selects a row or supplies an input feature. All original
80/1440 clocks remain; no available mass target means N/A, never mass PASS.
"""
from __future__ import annotations

from copy import copy
import json
from pathlib import Path

import numpy as np

from .conditional_overfit_data import (PROFILE as CONDITIONAL_PROFILE, STUDY,
    ConditionalOverfitDataset, precision_eligible_conflict_groups, state_history_evidence)
from .conditional_overfit_training import conditional_acceptance
from .overfit_data import OBSERVATION_KEYS, FIXED_EPISODES, FIXED_FRAMES
from .overfit_supervision_conflicts import state_supervision_feasibility

PROFILE = 'failure_aware_controlled_contact_v1'
OBSERVATION_SHAPES = dict(hand_pose_w=(2400, 2, 7), hand_sites_w=(2400, 54, 3),
    hand_normals_w=(2400, 54, 3), contact_position_w=(2400, 54, 3),
    normal_load_n=(2400, 54), shear_force_w=(2400, 54, 3),
    contact_area_m2=(2400, 54), timestamp_s=(2400,), object_pose_w=(2400, 7),
    object_com_w=(2400, 3), validation_full_mesh_min_z_m=(2400,))
SCOPE = ('Failure-aware controlled known-scene TRAIN fit and same-trajectory interpolation; '
         'all sixteen attempts retained. Independent learning qualification, not physical '
         'grasp success, original blind/all-state qualification, generalization or tactile benefit.')


def validate_observation_arrays(arrays, field):
    """Read-only full recording integrity; no force/geometry quality threshold."""
    for key in (*OBSERVATION_KEYS, 'object_pose_w', 'object_com_w',
                'validation_full_mesh_min_z_m'):
        value = np.asarray(arrays[key])
        if value.shape != OBSERVATION_SHAPES[key] or not np.isfinite(value).all():
            raise ValueError('Incomplete/nonfinite recorded array: ' + key)
    if not np.allclose(arrays['timestamp_s'], .02*np.arange(1, 2401), rtol=0, atol=1e-8):
        raise ValueError('Recording is not the original absolute 2400 x .02 s clock')
    for key in ('normal_load_n', 'contact_area_m2'):
        if np.any(arrays[key] < 0):
            raise ValueError('Negative observed load/area: ' + key)
    offset = np.asarray(field['offset'])
    count = len(field['area_m2'])
    if (offset.shape != (2401,) or not np.issubdtype(offset.dtype, np.integer)
            or offset[0] != 0 or offset[-1] != count or np.any(np.diff(offset) < 0)):
        raise ValueError('Malformed complete surface offsets')
    for key in ('position_hand_frame_m', 'area_m2', 'normal_pressure_pa',
                'shear_traction_hand_frame_pa', 'pad', 'hand'):
        value = np.asarray(field[key])
        shape = (count, 3) if key in ('position_hand_frame_m', 'shear_traction_hand_frame_pa') else (count,)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Incomplete/nonfinite surface observation: ' + key)
    if (np.any(field['area_m2'] <= 0) or np.any(field['normal_pressure_pa'] < 0)
            or not np.isin(field['hand'], (0, 1)).all()
            or not np.issubdtype(field['pad'].dtype, np.integer)
            or not np.isin(field['pad'], np.arange(54)).all()
            or not np.array_equal(field['hand'], field['pad']//27)):
        raise ValueError('Invalid surface pressure/area/hand/pad identity')
    return dict(passed=True, frames=2400, surface_rows=count)


def validate_failure_sources(root):
    """Fresh recorded clocks/COM/controller schema/late labels, before any model."""
    from .run_canonical_approach_fixture_pilot import validate_saved_case
    root = Path(root)
    declared = json.loads((root/'PROTOCOL.json').read_text())
    collection = json.loads((root/'COLLECTION_RESULT.json').read_text())
    records = collection['records']
    if (collection.get('complete') is not True
            or tuple(r['episode'] for r in records) != FIXED_EPISODES):
        raise ValueError('Failure-aware data must retain all sixteen complete attempts')
    reports = {}
    for config, record in zip(declared['configurations'], records, strict=True):
        if record.get('complete') is not True:
            raise ValueError('Runtime-incomplete attempts are not training data')
        directory = root/'cases'/f"episode_{config['episode']}"
        replay = validate_saved_case(directory, config, declared)
        if (record.get('controller_passed') != replay['physical_passed']
                or record.get('controller_checks') != replay['physical_checks']
                or record.get('late_mass_available') != replay['late_mass_available']):
            raise ValueError('Recorded physical/late labels differ from fresh readback')
        if not all(type(v) is bool for v in record['controller_checks'].values()):
            raise ValueError('Physical checks must retain their actual boolean schema')
        with np.load(directory/f"episode_{config['episode']}.npz", allow_pickle=False) as z:
            arrays = {key:z[key] for key in (*OBSERVATION_KEYS, 'object_pose_w',
                'object_com_w', 'validation_full_mesh_min_z_m')}
        with np.load(directory/'contact_surface.npz', allow_pickle=False) as z:
            quality = validate_observation_arrays(arrays, z)
        reports[str(config['episode'])] = dict(**quality,
            physical_passed=replay['physical_passed'],
            late_mass_available=replay['late_mass_available'],
            actual_com_transform_max_error_m=replay['actual_com_transform_max_error_m'])
    expected_physical = all(r['physical_passed'] and r['late_mass_available'] for r in reports.values())
    if collection.get('qualification_passed') is not expected_physical:
        raise ValueError('Original physical collection qualification is inconsistent')
    return dict(passed=True, cases=reports, all_16_complete=True,
                original_physical_success_gate_passed=expected_physical)


class FailureAwareOverfitDataset(ConditionalOverfitDataset):
    def __init__(self, root, protocol, *, purpose='overfit', interpolation_frames=None):
        # The original source/config/controller contract remains mandatory.
        super().__init__(root, protocol, purpose=purpose, interpolation_frames=interpolation_frames)
        self.failure_source_validation = validate_failure_sources(root)
        self.supervision_profile = PROFILE


def failure_aware_qualification(dataset, *, original_qualification):
    proxy = copy(dataset)
    proxy.supervision_profile = CONDITIONAL_PROFILE
    original = original_qualification(proxy)
    physical_only = {'original_collection_qualification_passed',
        'all_16_physical_qualifications_pass', 'all_16_cases_have_predeclared_late_mass',
        'all_16_fixed_late_mass_labels'}
    checks = {k:v for k,v in original['checks'].items() if k not in physical_only}
    checks.update(
        full_recording_data_quality_and_fresh_labels=getattr(dataset, 'failure_source_validation', {}).get('passed') is True,
        all_16_records_complete_with_original_check_schema=all(
            r['complete'] and r['exact_declared_check_set'] for r in original['physical_cases'].values()),
        every_fixed_clock_identity_present=sorted((r['metadata']['episode'], r['metadata']['frame'])
            for r in dataset.rows) == sorted((e,f) for e in FIXED_EPISODES for f in FIXED_FRAMES),
        some_observable_mass_targets_present=original['mass_available_count'] > 0)
    per_case = {}
    for episode in FIXED_EPISODES:
        rows = [r for r in dataset.rows if r['metadata']['episode'] == episode]
        n = sum(r['supervision']['mass_available'] > .5 for r in rows)
        per_case[str(episode)] = dict(items=len(rows), available_mass_denominator=int(n),
            mass_precision_status='NOT_YET_EVALUATED' if n else 'NOT_APPLICABLE',
            mass_precision_passed=None, physical=original['physical_cases'][str(episode)])
    return dict(original, passed=all(checks.values()), learnable_data_gate_passed=all(checks.values()),
        checks=checks, supervision_profile=PROFILE, scope=SCOPE, per_case=per_case,
        physical_success_gate_passed=dataset.collection_result['qualification_passed'],
        original_conditional_qualification_passed=original['passed'],
        original_conditional_checks=original['checks'],
        failure_source_validation=dataset.failure_source_validation)


def failure_aware_dense_qualification(dataset, *, frames, fit_limits):
    """Dense candidates must also be consistent, checked before model creation."""
    feasibility = state_supervision_feasibility(dataset.rows,
        precision_eligible_conflict_groups(dataset), center_limit_m=fit_limits['center_cm']/100.,
        size_max_relative_limit=fit_limits['size_max_relative'])
    mass_conflicts = [group for group in dataset.conflicts if len({
        float(dataset.rows[i]['target'][12]) for i in group['row_indices']
        if dataset.rows[i]['supervision']['mass_available'] > .5}) > 1]
    checks = dict(all_fixed_dense_clocks_present=sorted(
        (r['metadata']['episode'], r['metadata']['frame']) for r in dataset.rows)
        == sorted((e,f) for e in FIXED_EPISODES for f in frames),
        fresh_actual_com_and_source_labels=dataset.qualified and dataset.failure_source_validation['passed'],
        all_state_masks_match_observed_history=all(all(row['supervision'][key] == value
            for key,value in state_history_evidence(row['inputs']).items()) for row in dataset.rows),
        no_provably_incompatible_identical_input_state=feasibility['passed'],
        no_identical_input_different_available_mass=not mass_conflicts)
    return dict(passed=all(checks.values()), checks=checks, total_rows=len(dataset),
        state_supervision_feasibility=feasibility, contradictory_available_mass=mass_conflicts,
        input_conflicts=dataset.conflicts, scope=SCOPE)


def failure_aware_acceptance(arrays, role, *, original_acceptance, fit_limits,
                             interpolation_limits, episodes):
    original = conditional_acceptance(arrays, role, original_acceptance=original_acceptance,
        fit_limits=fit_limits, interpolation_limits=interpolation_limits, episodes=episodes)
    per_case, all_checks = {}, {}
    for episode in episodes:
        raw = original['per_case'][str(episode)]
        if 'checks' not in raw:
            per_case[str(episode)] = raw
            all_checks[f'{episode}/present'] = False
            continue
        mask = arrays['episode'] == episode
        truth = arrays['mass_available'][mask] > .5
        predicted = arrays['availability_probability'][mask] >= .5
        checks = dict(raw['checks'])
        # Failed late hold is a retained outcome, not a required precision clock.
        checks.pop('late_2381_state_evidence', None)
        mass_keys = ('mass',) if role == 'fit' else ('mass_mean', 'mass_p95')
        for key in mass_keys:
            if not truth.any():
                checks[key] = None  # N/A is never represented as a mass PASS.
        if role == 'same_trajectory_interpolation' and not (truth.any() and (~truth).any()):
            checks.pop('availability_balanced')
            if not truth.any():
                checks['availability_specificity'] = bool((~predicted).mean() >= interpolation_limits['availability_balanced_accuracy'])
            else:
                checks['availability_sensitivity'] = bool(predicted.mean() >= interpolation_limits['availability_balanced_accuracy'])
                checks['availability_false_confident'] = None
        active = {key: bool(value) for key,value in checks.items() if value is not None}
        per_case[str(episode)] = dict(raw, passed=all(active.values()), checks=checks,
            not_applicable_checks=[key for key,value in checks.items() if value is None],
            available_mass_denominator=int(truth.sum()),
            mass_precision_status=('PASS' if all(checks[k] for k in mass_keys) else 'FAIL') if truth.any() else 'NOT_APPLICABLE',
            mass_precision_passed=bool(all(checks[k] for k in mass_keys)) if truth.any() else None)
        all_checks.update({f'{episode}/{k}':v for k,v in active.items()})
    summary = original['all_items']['availability']
    if role == 'fit':
        all_checks['global/availability_accuracy'] = summary['accuracy'] >= fit_limits['availability_accuracy']
    else:
        all_checks['global/availability_balanced'] = (summary['balanced_accuracy'] is not None
            and summary['balanced_accuracy'] >= interpolation_limits['availability_balanced_accuracy'])
    all_checks['global/availability_false_confident'] = (summary['false_confident_fraction'] is not None
        and summary['false_confident_fraction'] <= interpolation_limits['availability_false_confident_fraction'])
    all_checks['global/available_mass_evaluation_present'] = bool(np.any(arrays['mass_available'] > .5))
    return dict(original, passed=all(all_checks.values()), checks=all_checks, per_case=per_case,
        scope=SCOPE, supervision_profile=PROFILE,
        original_conditional_gate_passed=original['passed'],
        mass_cases_with_available_targets=sum(r.get('available_mass_denominator', 0) > 0 for r in per_case.values()),
        mass_cases_without_available_targets=[k for k,r in per_case.items() if r.get('available_mass_denominator') == 0])
