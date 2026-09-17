"""Execute a predeclared fresh fixture corpus, retaining every failed grasp."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .collect_response_corpus import verify_condition, write_json
from .collect_surface_corpus import annotate


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use an explicit retained compute step')
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    configs = protocol['configurations']
    assert len(configs) == 16 and len({c['episode'] for c in configs}) == 16
    assert all(c['split'] == 'test' for c in configs)
    for name, expected in protocol['collector_source_sha256'].items():
        assert hashlib.sha256(Path('scripts/sugar/object_predictor', name).read_bytes()).hexdigest() == expected, name
    for name, expected in protocol['frozen_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    lock = (root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'cases').mkdir(exist_ok=False)
    (root / 'logs').mkdir(exist_ok=False)
    records = []
    for config in configs:
        episode = config['episode']
        dest = root / 'cases' / f'episode_{episode}'
        command = [sys.executable, '-m', 'scripts.sugar.object_predictor.collect_dense_grip',
                   '--protocol', str(root / 'PROTOCOL.json'), '--episode', str(episode),
                   '--output', str(dest), '--surface-feedback', '--frames', '2400',
                   '--force-gain', '.000025', '--sensor-coverage', 'continuous_palmar_v1',
                   '--response-gain', '--purpose', 'diagnostic']
        print('PROSPECTIVE_COLLECT_START', episode, flush=True)
        with (root / 'logs' / f'episode_{episode}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        verify_condition(dest, protocol['collector_source_sha256'])
        record = annotate(dest, config)
        records.append(record)
        write_json(root / 'COLLECTION_RESULT.json', dict(
            complete=len(records) == len(configs), records=records,
            attempted_configurations=len(records), planned_configurations=len(configs),
            controller_passes=sum(r['controller_passed'] for r in records),
            new_controls=2400*len(records), reused_controls=0,
            new_model_forwards=0, new_optimizer_updates=0))
        print('PROSPECTIVE_COLLECT_COMPLETE', json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
