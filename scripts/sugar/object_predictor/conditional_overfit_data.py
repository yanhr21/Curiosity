"""Explicit controlled-fixture data scope and observation-only state eligibility.

This adapter reuses every original H32 input and target. Contact admits a
conditional regression candidate; it does not certify identifiable full state.
No unavailable row is dropped or altered, and no ground truth enters inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .overfit_data import FIXED_EPISODES, OverfitDataset, collate_overfit

PROFILE = 'conditional_contact_v1'
STUDY = 'controlled_fixture16_v1'


def check_collection_scope(root, profile):
    root = Path(root)
    declared = json.loads((root / 'PROTOCOL.json').read_text())
    collection = json.loads((root / 'COLLECTION_RESULT.json').read_text())
    if profile == 'all_state':
        if declared.get('study') is not None or collection.get('study') is not None:
            raise ValueError('Original all-state profile cannot consume a differently scoped collection')
    elif profile == PROFILE:
        if declared.get('study') != STUDY or collection.get('study') != STUDY:
            raise ValueError('Conditional profile requires the explicit controlled_fixture16_v1 collection')
    else:
        raise ValueError('Unknown state-supervision profile')
    return declared, collection


def validate_controlled_sources(root):
    """Check declared identities and actual saved sources, before model creation."""
    from .collect_controlled_fixture_corpus import protocol as collection_protocol
    from .run_canonical_approach_fixture_pilot import CHECKS, INTERVENTION

    root = Path(root).resolve()
    declared, collection = check_collection_scope(root, PROFILE)
    for key, expected in collection_protocol().items():
        if declared.get(key) != expected:
            raise ValueError('Controlled collection protocol differs: ' + key)
    records = collection['records']
    if tuple(row['episode'] for row in records) != FIXED_EPISODES:
        raise ValueError('Controlled collection requires all fixed sixteen identities in order')
    if collection.get('scope') != declared['scope'] or collection.get('original_requested_approach_regression_repaired') is not False:
        raise ValueError('Controlled collection misstates its scope or blind-approach status')
    for config, record in zip(declared['configurations'], records, strict=True):
        episode = config['episode']
        directory = root / 'cases' / f'episode_{episode}'
        if (Path(record['source']).resolve() != directory or record['split'] != 'train'
                or record['geometry_group'] != config['geometry_group']):
            raise ValueError('Controlled source/identity mapping differs: ' + str(episode))
        actual = json.loads((directory / 'PROTOCOL.json').read_text())
        result = json.loads((directory / 'RESULT.json').read_text())
        for key in ('mass', 'load', 'scale', 'seed', 'geometry_group', 'approach_angle_deg',
                    'yaw_delta_deg', 'lift_height_m', 'lateral_xy'):
            actual_key = {'mass': 'mass_kg', 'load': 'target_load_n'}.get(key, key)
            if actual[actual_key] != config[key]:
                raise ValueError(f'Actual controlled configuration differs: {episode}/{key}')
        if (actual.get('controller_intervention') != INTERVENTION or
                result.get('controller_intervention') != INTERVENTION or
                actual.get('frames') != 2400 or actual.get('dt') != .02):
            raise ValueError('Actual controlled controller/clock differs')
        if (set(record.get('controller_checks', {})) != set(CHECKS) or
                record['controller_checks'] != result.get('checks') or
                record['controller_passed'] != result.get('passed') or
                bool(result.get('passed')) != all(record['controller_checks'].values())):
            raise ValueError('Controlled physical results disagree with saved record')
        late = record.get('late_mass_window', {})
        if (record.get('frames') != 2400 or late.get('frame') != 2381 or
                late.get('first_frame') != 2350):
            raise ValueError('Controlled fixed late mass window differs')
    return dict(passed=True, study=STUDY, source_root=str(root),
        configurations_checked=len(records), original_requested_approach_regression_repaired=False,
        scope=declared['scope'])


def state_history_evidence(inputs):
    """Current model's actual H32 occupancy, independent of labels and errors."""
    frames = inputs['feat']
    if len(frames) != 32:
        raise ValueError('Conditional state evidence requires the original H32 input')
    present = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 2 or frame.shape[1] != 20 or not len(frame) or not np.isfinite(frame).all():
            raise ValueError('Invalid original encoded observation features')
        if np.any((frame[:, 9] < 0) | (frame[:, 9] > 1)):
            raise ValueError('Invalid encoded contact occupancy')
        present.append(bool(np.any(frame[:, 9] > 0)))
    return dict(state_precision_eligible=np.float32(any(present)),
                state_contact_history_frames=np.int64(sum(present)))


class ConditionalOverfitDataset(OverfitDataset):
    def __init__(self, root, protocol, *, purpose='overfit', interpolation_frames=None):
        if purpose != 'overfit':
            raise ValueError('Controlled conditional data requires actual COM; no legacy fallback')
        self.controlled_source_validation = validate_controlled_sources(root)
        super().__init__(root, protocol, purpose=purpose, interpolation_frames=interpolation_frames)
        self.supervision_profile = PROFILE
        for row in self.rows:
            row['supervision'].update(state_history_evidence(row['inputs']))
        for row in self.rows:
            if row['metadata']['frame'] == 2381:
                record = self.collection_records[row['metadata']['episode']]
                if bool(row['supervision']['mass_available']) != record.get('late_mass_available'):
                    raise ValueError('Fresh late mass labels disagree with controlled collection')


def collate_conditional(rows):
    import torch
    result = collate_overfit(rows)
    for key in ('state_precision_eligible', 'state_contact_history_frames'):
        result[key] = torch.as_tensor(np.asarray([row['supervision'][key] for row in rows]))
    return result


def precision_eligible_conflict_groups(dataset):
    """Retain all reports, but only admitted point targets can contradict a gate."""
    output = []
    for group in dataset.conflicts:
        members = [i for i in group['row_indices']
                   if dataset.rows[i]['supervision']['state_precision_eligible'] > .5]
        if len(members) >= 2:
            output.append(dict(group, row_indices=members))
    return output
