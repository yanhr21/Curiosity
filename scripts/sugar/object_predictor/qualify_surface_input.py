"""Physical-unit, causality and geometry checks on saved real tactile streams."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .data import OBSERVATION_KEYS, MODES, history_indices, target_at
from .surface_data import select_surface_frames, encode_surface_observations, collate_surface


def arrays(path):
    with np.load(path) as src:
        return {k: src[k] for k in src.files}


def difference(a, b, key):
    if len(a[key]) != len(b[key]) or any(x.shape != y.shape for x, y in zip(a[key], b[key])):
        return float('inf')
    return max(float(np.max(abs(x - y))) for x, y in zip(a[key], b[key]))


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    root = Path(args.root); out = Path(args.output); out.mkdir(exist_ok=False)
    if args.episodes:
        cases = [(str(episode), root / 'cases' / f'episode_{episode}', episode)
                 for episode in args.episodes]
    else:
        names = ['5007_dense_surface_recovery297659', '5007_surface_feedback',
                 '5007_surface_feedback_gain25', '5007_surface_feedback_pi2']
        cases = [(name, root / 'diagnostics' / name, 5007) for name in names]
    checks = {}; details = []; prepared = {m: [] for m in MODES}
    for case_index, (case, path, episode) in enumerate(cases):
        data = arrays(path / f'episode_{episode}.npz'); field = arrays(path / 'contact_surface.npz')
        for frame in (750, len(data['timestamp_s']) - 1):
            indices = history_indices(frame, 32, 'episode_uniform_recent')
            obs = {k: data[k][indices] for k in OBSERVATION_KEYS}
            surfaces = select_surface_frames(field, indices)
            encoded = {m: encode_surface_observations(obs, surfaces, m) for m in MODES}
            key = f'{case}/{frame}'
            contact = encoded['geometry_contact']; force = encoded['geometry_contact_force']
            checks[key + '/contact_force_same_geometry'] = all(difference(contact, force, k) == 0 for k in ('coord', 'grid_coord'))
            columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
            checks[key + '/only_force_attributes_differ'] = all(np.array_equal(a[:, columns], b[:, columns]) for a, b in zip(contact['feat'], force['feat']))
            checks[key + '/finite_unique_voxels'] = all(np.isfinite(f).all() and len(np.unique(g, axis=0)) == len(g)
                for row in encoded.values() for f, g in zip(row['feat'], row['grid_coord']))
            load_error = area_error = shear_error = 0.
            for summary, feat in zip(force['sensor_summary'], force['feat']):
                load_error = max(load_error, abs(np.expm1(feat[:, 10].astype(float)).sum() - summary['input_load_n']))
                area_error = max(area_error, abs((feat[:, 14].astype(float) / 1e4).sum() - summary['input_area_m2']))
                shear_error = max(shear_error, float(np.max(abs((np.sinh(feat[:, 11:14].astype(float)) * 2).sum(0) - summary['input_shear_current_hand_n']))))
            checks[key + '/encoded_physical_force_area_conserved'] = load_error <= 1e-4 and area_error <= 1e-10 and shear_error <= 1e-4
            zero = encode_surface_observations(obs, surfaces, 'geometry_contact_force', force_gain=0.)
            checks[key + '/forcezero_preserves_contact_geometry_area'] = all(
                np.array_equal(a[:, [*range(10), *range(14, 20)]], b[:, [*range(10), *range(14, 20)]]) and np.all(b[:, 10:14] == 0)
                for a, b in zip(force['feat'], zero['feat']))
            poisoned = {**obs, 'object_pose_w': np.full((32, 7), np.nan), 'object_mass_kg': np.full(32, np.nan),
                        'validation_surface_normal_hand_frame': np.full((32, 54, 3), np.nan)}
            poisoned_surfaces = [{**f, 'validation_surface_normal_hand_frame': np.full((len(f['pad']), 3), np.nan)} for f in surfaces]
            hidden = encode_surface_observations(poisoned, poisoned_surfaces, 'geometry_contact_force')
            checks[key + '/validation_targets_do_not_enter_input'] = difference(force, hidden, 'feat') == 0
            # Geometry-only must not consume any contact positions, forces or surface.
            corrupt = {**obs, 'normal_load_n': np.full_like(obs['normal_load_n'], np.nan),
                       'contact_position_w': np.full_like(obs['contact_position_w'], np.nan),
                       'shear_force_w': np.full_like(obs['shear_force_w'], np.nan)}
            blind = encode_surface_observations(corrupt, [{}] * 32, 'geometry')
            checks[key + '/geometry_blind_to_touch'] = difference(encoded['geometry'], blind, 'feat') == 0
            detail = dict(case=case, frame=frame, history_indices=indices.tolist(),
                          maximum_decoded_load_error_n=load_error, maximum_decoded_area_error_m2=area_error,
                          maximum_decoded_shear_error_n=shear_error, sensor=force['sensor_summary'][-1])
            if case_index == 0 and frame == 750:
                # Split every integration element into identical half-area elements.
                # The same measured surface/load must not change with tessellation.
                split = [{k: np.repeat(v, 2, axis=0) / (2 if k == 'area_m2' else 1)
                          if k == 'area_m2' else np.repeat(v, 2, axis=0) for k, v in f.items()} for f in surfaces]
                split_row = encode_surface_observations(obs, split, 'geometry_contact_force')
                checks[key + '/subdivision_invariance'] = difference(force, split_row, 'feat') <= 2e-6 and difference(force, split_row, 'grid_coord') == 0
                moved = {k: v.copy() for k, v in obs.items()}
                turn = Rotation.from_euler('z', .71); shift = np.array([.35, -.27, .11])
                for k in ('hand_sites_w', 'contact_position_w'):
                    moved[k] = turn.apply(obs[k].reshape(-1, 3)).reshape(obs[k].shape) + shift
                moved['hand_pose_w'] = obs['hand_pose_w'].astype(float).copy()
                moved['hand_pose_w'][..., :3] = turn.apply(obs['hand_pose_w'][..., :3].reshape(-1, 3)).reshape(32, 2, 3) + shift
                moved['hand_pose_w'][..., 3:] = (turn * Rotation.from_quat(obs['hand_pose_w'][..., 3:].reshape(-1, 4))).as_quat().reshape(32, 2, 4)
                for k in ('hand_normals_w', 'shear_force_w'):
                    moved[k] = turn.apply(obs[k].reshape(-1, 3)).reshape(obs[k].shape)
                transformed = encode_surface_observations(moved, surfaces, 'geometry_contact_force')
                checks[key + '/world_yaw_translation_invariance'] = difference(force, transformed, 'feat') <= 3e-5 and difference(force, transformed, 'grid_coord') == 0
                detail['world_transform_max_feature_difference'] = difference(force, transformed, 'feat')
            details.append(detail)
            if frame == len(data['timestamp_s']) - 1 and case_index in (0, 2):
                for mode, row in encoded.items():
                    row.update(target=target_at(data, frame), episode=case_index, frame=frame)
                    prepared[mode].append(row)
            print('QUALIFIED_WINDOW', key, all(checks.values()), flush=True)
    for mode, rows in prepared.items():
        batch = collate_surface(rows)
        np.savez_compressed(out / (mode + '.npz'), **{k: v.numpy() for k, v in batch.items()})
    report = dict(passed=all(checks.values()), checks=checks, details=details,
                  model_updates=0, new_physics_steps=0,
                  scope='Saved real TRAIN diagnostic traces only; not training or a held-out performance result.')
    (out / 'INPUT_REPORT.json').write_text(json.dumps(report, indent=2)); print(json.dumps({k: v for k, v in report.items() if k != 'details'}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, nargs='+', help='Explicit saved TRAIN cases instead of historical diagnostics')
    main(parser.parse_args())
