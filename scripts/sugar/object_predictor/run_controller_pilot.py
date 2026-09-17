"""Fixed failure/regression pair for one declared observation-only intervention."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .retained_execution import require_active_resource
from .controller_interventions import INTERVENTIONS

PAIRS = {'continuous_phase_governor_v1': [5012, 5000],
         'regional_bilateral_alignment_v1': [5008, 5000],
         'legacy_runtime_path_v1': [5012, 5000],
         'bilateral_common_alignment_v1': [5008, 5000],
         'freeze_postlift_alignment_v1': [5012, 5000],
         'observed_normal_bisector_alignment_v1': [5008, 5000]}


def main(root):
    resource = require_active_resource()
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    if protocol.get('controls_per_case') != 2400:
        raise ValueError('Keep the original 2400-control budget per case')
    for path, expected in protocol.get('prepared_source_sha256', {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared controller source changed: '+path)
    intervention = protocol['controller_intervention']
    if intervention not in PAIRS or [c['episode'] for c in protocol['configurations']] != PAIRS[intervention]:
        raise ValueError('Keep the declared failure and regression pair')
    baseline = Path(protocol['baseline_root'])
    source = json.loads((baseline/'PROTOCOL.json').read_text())
    for config in protocol['configurations']:
        if config != next(c for c in source['configurations'] if c['episode'] == config['episode']):
            raise ValueError('Preserve exact original case configuration')
    (root/'cases').mkdir(exist_ok=False)
    (root/'logs').mkdir(exist_ok=False)
    rows = []
    for config in protocol['configurations']:
        episode = config['episode']; dest = root/'cases'/f'episode_{episode}'
        command = [sys.executable, '-P', '-m', 'scripts.sugar.object_predictor.collect_dense_grip',
            '--protocol', str(root/'PROTOCOL.json'), '--episode', str(episode), '--output', str(dest),
            '--surface-feedback', '--frames', '2400', '--force-gain', '.000025',
            '--sensor-coverage', 'continuous_palmar_v1', '--response-gain',
            '--controller-revision', 'sensor_feedback_v2',
            '--controller-intervention', intervention, '--purpose', 'diagnostic']
        print('CONTROLLER_PILOT_START', intervention, episode, flush=True)
        with (root/'logs'/f'episode_{episode}.log').open('x') as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        old = json.loads((baseline/'cases'/f'episode_{episode}'/'RESULT.json').read_text())
        row = dict(episode=episode, exit_code=rc, baseline=old, complete=False, passed=False)
        if rc == 0 and (dest/'RESULT.json').exists():
            import numpy as np
            current = json.loads((dest/'RESULT.json').read_text())
            if current.get('controller_intervention') != intervention:
                raise ValueError('Actual intervention missing')
            with np.load(dest/f'episode_{episode}.npz', allow_pickle=False) as data:
                if len(data['timestamp_s']) != 2400:
                    raise ValueError('Incomplete physical trajectory')
                for key in INTERVENTIONS[intervention][3]:
                    if len(data['validation_controller_'+key]) != 2400:
                        raise ValueError('Missing intervention evidence: '+key)
            row.update(complete=True, passed=current['passed'], current=current)
        rows.append(row)
        result = dict(complete=len(rows) == 2 and all(r['complete'] for r in rows),
            paired_qualification_passed=len(rows) == 2 and all(r['passed'] for r in rows),
            cases=rows, planned_cases=2, controller_intervention=intervention,
            retained_resource=resource, new_model_forwards=0, new_optimizer_updates=0,
            scope='Fixed single-component physical pilot; not all16 qualification or learned policy')
        (root/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        print('CONTROLLER_PILOT_COMPLETE', intervention, episode, row['complete'], row['passed'], flush=True)
    if not result['complete']:
        return 1
    return 0 if result['paired_qualification_passed'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
