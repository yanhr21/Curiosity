"""Bounded repair qualification on all 16 declared original TRAIN cases.

No success-only collection, controller threshold change, or automatic training.
Failed physical cases remain in the result and fail the downstream gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from .overfit_data import FIXED_EPISODES, fixed_protocol, mass_supervision, uniform_history_indices


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use the retained compute step and serial GPU lock')
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    configs = protocol['configurations']
    if tuple(c['episode'] for c in configs) != FIXED_EPISODES or any(c['split'] != 'train' for c in configs):
        raise ValueError('Preserve all 16 original fixed TRAIN configurations')
    if protocol['overfit_data'] != fixed_protocol():
        raise ValueError('Declared clocks/history/availability thresholds changed')
    for name in ('cases', 'logs'):
        (root/name).mkdir(exist_ok=False)
    records = []
    for config in configs:
        episode = config['episode']
        dest = root/'cases'/f'episode_{episode}'
        record = dict(episode=episode, split='train', geometry_group=config['geometry_group'],
                      source=str(dest.resolve()), complete=False, controller_passed=False)
        command = [sys.executable, '-P', '-m', 'scripts.sugar.object_predictor.collect_dense_grip',
            '--protocol', str(root/'PROTOCOL.json'), '--episode', str(episode), '--output', str(dest),
            '--surface-feedback', '--frames', '2400', '--force-gain', '.000025',
            '--sensor-coverage', 'continuous_palmar_v1', '--response-gain',
            '--controller-revision', 'sensor_feedback_v2', '--purpose', 'diagnostic']
        print('REPAIR_COLLECT_START', episode, flush=True)
        try:
            with (root/'logs'/f'episode_{episode}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            actual = json.loads((dest/'RESULT.json').read_text())
            with np.load(dest/f'episode_{episode}.npz', allow_pickle=False) as data:
                arrays = {key: data[key] for key in ('timestamp_s', 'object_com_w', 'object_pose_w',
                    'normal_load_n', 'validation_full_mesh_min_z_m', 'validation_object_local_com_m')}
            if len(arrays['timestamp_s']) != 2400:
                raise ValueError('Incomplete physical trajectory')
            from scipy.spatial.transform import Rotation
            com = Rotation.from_quat(arrays['object_pose_w'][:, 3:]).apply(
                arrays['validation_object_local_com_m']) + arrays['object_pose_w'][:, :3]
            if not np.allclose(com, arrays['object_com_w'], rtol=0, atol=1e-12):
                raise ValueError('Saved COM does not match actual body COM transform')
            labels = []
            for frame in protocol['overfit_data']['frames']:
                indices = uniform_history_indices(arrays['timestamp_s'], frame, 32, .02)
                label = mass_supervision(arrays, indices, protocol['overfit_data']['mass_criteria'])
                labels.append(dict(frame=frame, mass_available=bool(label['mass_available']),
                    mass_status=int(label['mass_status']), diagnostics=label['evaluation_only']))
            record.update(complete=True, controller_passed=actual['passed'], controller_checks=actual['checks'],
                values=actual['values'], fit_clock_mass_labels=labels,
                fit_clock_qualified_mass_count=sum(row['mass_available'] for row in labels),
                com_transform_max_abs_error_m=float(abs(com-arrays['object_com_w']).max()), frames=2400)
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        records.append(record)
        result = dict(complete=len(records)==16 and all(r['complete'] for r in records),
            records=records, attempted_configurations=len(records), planned_configurations=16,
            controller_passes=sum(r['controller_passed'] for r in records),
            new_optimizer_updates=0, new_model_forwards=0,
            qualification_passed=len(records)==16 and all(r['complete'] and r['controller_passed']
                and r.get('fit_clock_qualified_mass_count', 0)>=1 for r in records))
        save(root/'COLLECTION_RESULT.json', result)
        print('REPAIR_COLLECT_COMPLETE', json.dumps(record), flush=True)
    save(root/'QUALIFICATION.json', result)
    return 0 if result['qualification_passed'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
