"""Independent observation/target adapter for a declared full-model overfit study.

This module does not change the archived training adapters.  The caller supplies
the fixed TRAIN selection and supervision protocol; no model error selects data.
Model inputs contain only the existing observation allowlist.  Ground-truth state
is used for targets and availability labels, never for input construction.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


OBSERVATION_KEYS = (
    'hand_pose_w', 'hand_sites_w', 'hand_normals_w', 'contact_position_w',
    'normal_load_n', 'shear_force_w', 'contact_area_m2', 'timestamp_s',
)
INPUT_KEYS = ('coord', 'grid_coord', 'feat')
MASS_STATUS_NAMES = ('no_contact', 'support_unconfirmed', 'moving', 'stable_supported',
                     'unqualified_missing_com')
FIXED_EPISODES = tuple(range(5000, 5004)) + tuple(range(5008, 5016)) + tuple(range(5020, 5024))
FIXED_FRAMES = (31, 1181, 1281, 1581, 2381)


def fixed_protocol():
    """Return the predeclared 16-case/80-row overfit protocol, not a data search."""
    return dict(episodes=list(FIXED_EPISODES), frames=list(FIXED_FRAMES), history=32,
                history_interval_s=.02, time_scale_s=150.,
                mass_criteria=dict(clearance_m=.01, bilateral_load_n=.01,
                    max_acceleration_m_s2=.5, max_angular_speed_rad_s=.2))


def uniform_history_indices(timestamps, frame, history, interval_s, *, tolerance_s=1e-8):
    """Select an exact causal physical-time grid; never pad or choose by labels."""
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError('Expected finite one-dimensional recording timestamps')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Recording timestamps must be strictly increasing')
    if history < 3 or not isinstance(history, (int, np.integer)):
        raise ValueError('History must be an integer >= 3')
    if not isinstance(frame, (int, np.integer)) or frame < 0 or frame >= len(times):
        raise ValueError('Current frame is outside the recording')
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError('Physical history interval must be positive')
    requested = times[frame] - np.arange(history - 1, -1, -1) * interval_s
    right = np.searchsorted(times, requested).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(times[left] - requested) <= abs(times[right] - requested), left, right)
    if (indices[-1] != frame or np.any(np.diff(indices) <= 0)
            or np.any(indices > frame) or np.max(abs(times[indices] - requested)) > tolerance_s):
        raise ValueError('Recording cannot supply the complete declared physical-time history')
    return indices.astype(np.int64)


def observation_window(arrays, indices):
    """An explicit allowlist prevents validation/GT arrays from entering inputs."""
    return {key: np.asarray(arrays[key])[indices].copy() for key in OBSERVATION_KEYS}


def observed_physics_targets(encoded_frame):
    """Physical-unit targets computed only from the existing 20 input columns.

    Shear is force ON the hand.  Encoder normals point outward from hand CAD or
    its observed contact-plane fit.  Thus ``normal * load - shear`` approximates
    force ON the object.  These normals are not solver interface normals, so the
    combined resultant is explicitly an observation-derived approximation.
    Pure shear, normal loads, contact area, and occupancy remain separate targets.
    """
    feat = np.asarray(encoded_frame, dtype=np.float64)
    if feat.ndim != 2 or feat.shape[1] != 20 or not len(feat) or not np.isfinite(feat).all():
        raise ValueError('Expected a finite nonempty current-frame 20-column cloud')
    if np.any(feat[:, 10] < 0) or np.any(feat[:, 14] < 0):
        raise ValueError('Encoded pressure load and area must be nonnegative')
    if np.any((feat[:, 16] < 0) | (feat[:, 16] > 1)):
        raise ValueError('Encoded hand-side weights must lie in [0, 1]')
    down = feat[0, 17:20]
    if not np.allclose(feat[:, 17:20], down, rtol=0, atol=1e-6) or not np.isclose(np.linalg.norm(down), 1, atol=1e-6):
        raise ValueError('A frame must have one known unit gravity direction')
    load = np.expm1(feat[:, 10])
    shear = 2 * np.sinh(feat[:, 11:14])
    normal_on_object = feat[:, 6:9] * load[:, None]
    object_resultant = (normal_on_object - shear).sum(0)
    # A voxel may contain both sides; use the encoder's actual side weights,
    # rather than inventing an exact raw-sensor hand assignment after pooling.
    side_weight = np.stack((1 - feat[:, 16], feat[:, 16]), axis=1)
    return {
        'normal_load_n': load.sum(keepdims=True).astype(np.float32),
        'normal_load_by_encoded_side_n': (side_weight.T @ load).astype(np.float32),
        'contact_area_m2': np.array([feat[:, 14].sum() / 1e4], np.float32),
        'shear_on_hand_current_frame_n': shear.sum(0).astype(np.float32),
        'normal_on_object_approx_current_frame_n': normal_on_object.sum(0).astype(np.float32),
        'object_resultant_approx_current_frame_n': object_resultant.astype(np.float32),
        'shear_support_up_n': np.array([shear.sum(0) @ down], np.float32),
        'combined_support_up_approx_n': np.array([-object_resultant @ down], np.float32),
        'contact_present': np.array([bool(np.any(feat[:, 9] > 0))], np.float32),
        'occupied_voxel_count': np.array([np.count_nonzero(feat[:, 9] > 0)], np.float32),
    }


def mass_supervision(arrays, indices, criteria):
    """GT-only support/stability labels, independent of model/force mass error.

    ``criteria`` must declare clearance, bilateral load, COM acceleration, and
    angular speed limits before execution.  The
    whole physical-time window must qualify.  This operational label does not
    certify unobserved contact-force coverage or arbitrary-state identifiability.
    """
    required = ('clearance_m', 'bilateral_load_n', 'max_acceleration_m_s2', 'max_angular_speed_rad_s')
    if set(criteria) != set(required):
        raise ValueError('Mass availability requires exactly the declared physical criteria')
    if not all(np.isfinite(criteria[k]) and criteria[k] > 0 for k in required):
        raise ValueError('Mass availability thresholds must be positive and finite')
    timestamps = np.asarray(arrays['timestamp_s'])[indices]
    dt = np.diff(timestamps)
    if len(indices) < 3 or np.any(dt <= 0):
        raise ValueError('Availability needs at least three strictly causal samples')
    if 'object_com_w' not in arrays:
        return dict(mass_available=np.float32(0), mass_status=np.int64(4),
                    mass_uncertain=np.float32(1), evaluation_only=dict(
                        qualified=False, reason='Missing actual object_com_w label; AABB center is not COM'))
    pose = np.asarray(arrays['object_pose_w'])[indices]
    rotations = Rotation.from_quat(pose[:, 3:])
    com = np.asarray(arrays['object_com_w'])[indices]
    if com.shape != (len(indices), 3):
        raise ValueError('object_com_w must be the actual per-frame world COM in meters')
    velocity = np.diff(com, axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]
    angular_speed = (rotations[1:] * rotations[:-1].inv()).magnitude() / dt
    loads = np.asarray(arrays['normal_load_n'])[indices].reshape(len(indices), 2, 27).sum(2)
    clearance = np.asarray(arrays['validation_full_mesh_min_z_m'])[indices]
    if not all(np.isfinite(v).all() for v in (com, velocity, acceleration, angular_speed, loads, clearance)):
        raise ValueError('Nonfinite state/sensor data in availability labels')
    if np.any(loads < 0):
        raise ValueError('Normal sensor loads cannot be negative')
    measured_contact = bool(np.any(loads > 1e-3))
    checks = {
        'airborne_entire_window': bool(np.all(clearance > criteria['clearance_m'])),
        'bilateral_entire_window': bool(np.all(loads > criteria['bilateral_load_n'])),
        'translation_acceleration_stable': bool(np.max(np.linalg.norm(acceleration, axis=1)) <= criteria['max_acceleration_m_s2']),
        'angular_speed_stable': bool(np.max(angular_speed) <= criteria['max_angular_speed_rad_s']),
    }
    supported = checks['airborne_entire_window'] and checks['bilateral_entire_window']
    stable = all(checks.values())
    status = 0 if not measured_contact else (1 if not supported else (3 if stable else 2))
    return {
        'mass_available': np.float32(stable), 'mass_status': np.int64(status),
        'mass_uncertain': np.float32(not stable),
        'evaluation_only': dict(qualified=True, checks=checks,
            min_clearance_m=float(clearance.min()), min_hand_load_n=float(loads.min()),
            max_speed_m_s=float(np.linalg.norm(velocity, axis=1).max()),
            max_acceleration_m_s2=float(np.linalg.norm(acceleration, axis=1).max()),
            max_angular_speed_rad_s=float(angular_speed.max())),
    }


def input_conflicts(rows):
    """Record bit-identical model inputs with distinct targets; never drop rows.

    SHA is only an indexing shortcut for this identifiability test.  Every group
    is also compared array-for-array.  Raw target spread is recorded without
    silently replacing targets or deciding a state loss mask.
    """
    groups = {}
    for index, row in enumerate(rows):
        digest = hashlib.sha256()
        for key in INPUT_KEYS:
            for value in row['inputs'][key]:
                value = np.ascontiguousarray(value)
                digest.update(str((key, value.dtype.str, value.shape)).encode())
                digest.update(value.tobytes())
        groups.setdefault(digest.digest(), []).append(index)
    result = []
    for members in groups.values():
        if len(members) < 2:
            continue
        first = rows[members[0]]['inputs']
        if not all(all(len(first[key]) == len(rows[i]['inputs'][key]) and
                       all(np.array_equal(a, b) for a, b in zip(first[key], rows[i]['inputs'][key]))
                       for key in INPUT_KEYS) for i in members):
            raise RuntimeError('Input grouping digest collision')
        targets = np.stack([rows[i]['target'] for i in members]).astype(np.float64)
        if np.array_equal(targets, np.broadcast_to(targets[:1], targets.shape)):
            continue
        spread = np.ptp(targets, axis=0)
        result.append(dict(row_indices=members,
            samples=[rows[i]['metadata'] for i in members],
            raw_target_spread=spread.tolist(),
            state_target_conflict=bool(np.any(spread[:12] > 0)),
            mass_target_conflict=bool(spread[12] > 0),
            mass_available=[float(rows[i]['supervision']['mass_available']) for i in members]))
    return result


class OverfitDataset:
    """Fixed TRAIN-only rows; inputs and supervision occupy separate dictionaries.

    Protocol keys are ``episodes``, ``frames``, ``history``,
    ``history_interval_s``, ``time_scale_s``, and ``mass_criteria``.  No automatic
    fallback, rejection of hard rows, or replacement of missing cases occurs.
    """
    def __init__(self, root, protocol, *, purpose='overfit', interpolation_frames=None):
        from .data import target_at
        from .hand_surface_normals import query_observation_normals
        from .surface_data import select_surface_frames, encode_surface_observations

        required = {'episodes', 'frames', 'history', 'history_interval_s', 'time_scale_s', 'mass_criteria'}
        if set(protocol) != required:
            raise ValueError('Unexpected or missing fixed overfit protocol fields')
        if (tuple(protocol['episodes']) != FIXED_EPISODES or tuple(protocol['frames']) != FIXED_FRAMES
                or protocol['history'] != 32 or protocol['history_interval_s'] != .02):
            raise ValueError('Preserve the fixed 16 TRAIN cases, five clocks, and 32 x .02 s history')
        if purpose not in ('overfit', 'input_check'):
            raise ValueError('Purpose must be overfit or explicitly unqualified input_check')
        frames = protocol['frames'] if interpolation_frames is None else list(interpolation_frames)
        if interpolation_frames is not None and (not frames or len(set(frames)) != len(frames)
                or set(frames) & set(FIXED_FRAMES)):
            raise ValueError('Interpolation clocks must be nonempty, unique, and separate from fit clocks')
        if len(set(protocol['episodes'])) != len(protocol['episodes']) or len(set(protocol['frames'])) != len(protocol['frames']):
            raise ValueError('Fixed selection must not contain duplicates')
        if not protocol['episodes'] or not protocol['frames']:
            raise ValueError('Fixed selection cannot be empty')
        root = Path(root)
        collection = json.loads((root / 'COLLECTION_RESULT.json').read_text())
        if not collection['complete']:
            raise ValueError('Use the complete recorded corpus')
        records = {r['episode']: r for r in collection['records']}
        self.collection_result = collection
        self.collection_records = {episode: records[episode] for episode in FIXED_EPISODES}
        self.protocol = json.loads(json.dumps(protocol))
        self.purpose = purpose
        self.clock_role = 'fit' if interpolation_frames is None else 'same_trajectory_interpolation'
        self.rows = []
        for episode in protocol['episodes']:
            record = records[episode]
            if record['split'] != 'train':
                raise ValueError('Overfit rows must all be original TRAIN configurations')
            source = Path(record['source'])
            with np.load(source / f'episode_{episode:04d}.npz') as z:
                arrays = {k: z[k] for k in z.files}
            if purpose == 'overfit' and 'object_com_w' not in arrays:
                raise ValueError(f'Episode {episode} is unqualified: missing actual object_com_w; use input_check only')
            with np.load(source / 'contact_surface.npz') as z:
                field = {k: z[k] for k in ('offset', 'position_hand_frame_m', 'area_m2',
                    'normal_pressure_pa', 'shear_traction_hand_frame_pa', 'pad', 'hand')}
            for frame in frames:
                indices = uniform_history_indices(arrays['timestamp_s'], frame, protocol['history'], protocol['history_interval_s'])
                obs = observation_window(arrays, indices)
                site, contact, _ = query_observation_normals(obs)
                obs['hand_site_surface_normals_w'] = site
                obs['hand_contact_surface_normals_w'] = contact
                encoded = encode_surface_observations(obs, select_surface_frames(field, indices),
                    'geometry_contact_force', time_scale_s=protocol['time_scale_s'])
                labels = mass_supervision(arrays, indices, protocol['mass_criteria'])
                diagnostics = labels.pop('evaluation_only')
                physics = observed_physics_targets(encoded['feat'][-1])
                summary = encoded['sensor_summary'][-1]
                physics['fitted_normal_face_fraction'] = np.array([
                    summary['fitted_normal_faces'] / max(summary['input_faces'], 1)], np.float32)
                self.rows.append(dict(
                    inputs={key: encoded[key] for key in INPUT_KEYS},
                    target=target_at(arrays, frame),
                    supervision={**labels, 'physics': physics},
                    evaluation_only=diagnostics,
                    metadata=dict(episode=int(episode), frame=int(frame),
                        geometry_group=int(record['geometry_group']),
                        timestamp_s=float(arrays['timestamp_s'][frame]), history_indices=indices.tolist()),
                ))
        self.conflicts = input_conflicts(self.rows)
        self.qualified = all(row['evaluation_only']['qualified'] for row in self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_overfit(rows):
    """Only ``batch['inputs']`` goes to the model; GT labels stay separate."""
    import torch
    from .surface_data import collate_surface
    if not rows:
        raise ValueError('Empty overfit batch')
    physics_names = rows[0]['supervision']['physics'].keys()
    return {
        'inputs': collate_surface([row['inputs'] for row in rows]),
        'target': torch.from_numpy(np.stack([row['target'] for row in rows])),
        'mass_available': torch.tensor([row['supervision']['mass_available'] for row in rows]),
        'mass_uncertain': torch.tensor([row['supervision']['mass_uncertain'] for row in rows]),
        'mass_status': torch.tensor([row['supervision']['mass_status'] for row in rows], dtype=torch.long),
        'physics': {name: torch.from_numpy(np.stack([row['supervision']['physics'][name] for row in rows])) for name in physics_names},
        'metadata': [row['metadata'] for row in rows],
        'evaluation_only': [row['evaluation_only'] for row in rows],
    }
