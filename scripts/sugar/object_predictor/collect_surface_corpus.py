"""Fixed perception corpus; retain controller failures and qualify TRAIN coverage.

This does not change the previous controller benchmark acceptance. Complete finite
failed grasps still supply pose/shape observations. Mass supervision is separately
labelled from actual airborne hold state, with partial sensor coverage reported.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def read_arrays(path):
    with np.load(path) as arrays:
        return {k: arrays[k] for k in arrays.files}


def annotate(path, config):
    episode = config['episode']
    data = read_arrays(path / f'episode_{episode:04d}.npz')
    surface = read_arrays(path / 'contact_surface.npz')
    result = json.loads((path / 'RESULT.json').read_text())
    if len(data['timestamp_s']) != 2400 or surface['offset'].shape != (2401,):
        raise ValueError('Incomplete recording')
    if not all(np.isfinite(value).all() for value in (*data.values(), *surface.values())):
        raise ValueError('Nonfinite recorded observation or label')
    if surface['offset'][-1] != len(surface['pad']) or np.any(np.diff(surface['offset']) < 0):
        raise ValueError('Invalid tactile surface frame offsets')
    for key, value in [('mass_kg', config['mass']), ('grip_target_per_hand_n', config['load']),
                       ('scale', config['scale']), ('seed', config['seed']),
                       ('approach_angle_deg', config['approach_angle_deg']),
                       ('yaw_delta_deg', config['yaw_delta_deg']), ('lift_height_m', config['lift_height_m']),
                       ('lateral_xy', config['lateral_xy'])]:
        if not np.allclose(result[key], value, rtol=1e-6, atol=1e-8):
            raise ValueError('Recording configuration differs: ' + key)
    loads = data['normal_load_n'].reshape(-1, 2, 27).sum(2)
    airborne = data['validation_full_mesh_min_z_m'] > .01
    bilateral = (loads > .01).all(1)
    hold = data['validation_controller_phase'] == 2
    mass_label = airborne & bilateral & hold
    # These arrays are labels/loss masks. The surface encoder has an explicit
    # observation allowlist and never receives this file.
    np.savez_compressed(path / 'supervision.npz', mass_label_weight=mass_label.astype(np.float32),
                        actual_airborne=airborne, measured_bilateral_contact=bilateral, controller_hold=hold)
    full = data['validation_resolved_normal_n']
    fractions = np.divide(loads, full, out=np.zeros_like(loads), where=full > 1e-8)
    return dict(episode=episode, split=config['split'], geometry_group=config['geometry_group'],
                frames=2400, controller_passed=result['passed'], controller_checks=result['checks'],
                airborne_hold_label_frames=int(mass_label.sum()),
                airborne_frames=int(airborne.sum()), surface_samples=len(surface['pad']),
                recorded_full_normal_fraction_during_mass_labels=fractions[mass_label].mean(0).tolist() if mass_label.any() else None,
                values=result['values'], source=str(path))


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    root = Path(args.output); protocol = json.loads((root / 'PROTOCOL.json').read_text())
    qualification = Path(protocol['input_qualification'])
    if not all(json.loads((qualification / name).read_text())['passed'] for name in ('INPUT_REPORT.json', 'FULL_MODEL_REPORT.json')):
        raise RuntimeError('Full official model/input qualification not passed')
    lock = (root / 'collection.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'cases').mkdir(exist_ok=True); (root / 'logs').mkdir(exist_ok=True)
    records = []
    for split in ('train', 'val', 'test'):
        if split != 'train' and not json.loads((root / 'TRAIN_QUALIFICATION.json').read_text())['passed']:
            raise RuntimeError('TRAIN coverage does not support held-out collection')
        for config in [c for c in protocol['configurations'] if c['split'] == split]:
            episode = config['episode']; dest = root / 'cases' / f'episode_{episode:04d}'
            if episode == 5007 and not dest.exists():
                source = Path(protocol['reuse_5007'])
                old = json.loads((source / 'PROTOCOL.json').read_text())
                if old['frames'] != 2400 or old['force_gain_m_per_ns'] != .000025 or old.get('force_integral_time_s', 0.) != 0.:
                    raise RuntimeError('Reused TRAIN diagnostic has a different controller')
                dest.mkdir()
                for name in ('episode_5007.npz', 'episode_5007.json', 'contact_surface.npz', 'RESULT.json', 'PROTOCOL.json'):
                    shutil.copy2(source / name, dest / name)
                (dest / 'REUSED_SOURCE.json').write_text(json.dumps(dict(source=str(source),
                    note='Already observed TRAIN configuration, same low-gain controller. Original failed target-tracking result retained.'), indent=2))
            if not (dest / 'RESULT.json').exists():
                if dest.exists():
                    raise RuntimeError('Incomplete prior case requires an explicit recovery: ' + str(dest))
                command = [sys.executable, '-m', 'scripts.sugar.object_predictor.collect_dense_grip',
                           '--protocol', str(root / 'PROTOCOL.json'), '--episode', str(episode),
                           '--surface-feedback', '--force-gain', '.000025', '--force-integral-time', '0',
                           '--frames', '2400', '--purpose', 'perception_dataset', '--output', str(dest)]
                print('COLLECT_START', episode, split, flush=True)
                with (root / 'logs' / f'episode_{episode:04d}.log').open('x') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            record = annotate(dest, config); records.append(record)
            (root / 'COLLECTION_RESULT.json').write_text(json.dumps(dict(complete=False, records=records), indent=2))
            print('COLLECT_COMPLETE', json.dumps(record), flush=True)
        if split == 'train':
            groups = {}
            for group in sorted({r['geometry_group'] for r in records}):
                parts = [r for r in records if r['geometry_group'] == group]
                groups[group] = dict(episodes=[r['episode'] for r in parts],
                    label_frames=[r['airborne_hold_label_frames'] for r in parts],
                    covered=len(parts) == 4 and all(r['airborne_hold_label_frames'] >= 100 for r in parts))
            covered = sum(g['covered'] for g in groups.values())
            report = dict(passed=covered >= 4, covered_geometry_groups=covered, groups=groups,
                          rule='At least 4/8 TRAIN groups; all four mass x grip settings each retain >=100 airborne bilateral hold labels.',
                          not_controller_acceptance=True, no_failed_cases_removed=True)
            (root / 'TRAIN_QUALIFICATION.json').write_text(json.dumps(report, indent=2)); print('TRAIN_COVERAGE', json.dumps(report), flush=True)
            if not report['passed']:
                raise SystemExit(2)
    (root / 'COLLECTION_RESULT.json').write_text(json.dumps(dict(complete=True, records=records), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); main(parser.parse_args())
