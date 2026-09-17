"""Qualify the original fixed corpus under the explicit new sensing/control condition.

Reuse only the three completed, source-identical TRAIN diagnostics. Preserve their
diagnostic metadata; qualification does not retroactively make them held-out data.
No model inference or training is launched here.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .collect_surface_corpus import annotate


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_condition(path, expected_sources):
    protocol = json.loads((path / 'PROTOCOL.json').read_text())
    expected = dict(frames=2400, sensor_coverage='continuous_palmar_v1',
                    response_gain=True, object_sdf_resolution=128,
                    force_gain_m_per_ns=.000025, force_integral_time_s=0.,
                    purpose='diagnostic', original_full_mesh_and_physics=True)
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f'Condition differs: {path}: {key}')
    if protocol['source_sha256'] != expected_sources:
        raise ValueError(f'Collection/controller sources differ: {path}')


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    original = json.loads(Path(protocol['original_protocol']).read_text())
    if protocol['configurations'] != original['configurations']:
        raise ValueError('Original configurations and splits must be retained exactly')
    configs = protocol['configurations']
    if [sum(c['split'] == s for c in configs) for s in ('train', 'val', 'test')] != [32, 8, 8]:
        raise ValueError('Expected original 32/8/8 split')
    if len({c['episode'] for c in configs}) != 48:
        raise ValueError('Duplicate episode')
    sources = protocol['collector_source_sha256']
    for name, expected in sources.items():
        if hashlib.sha256(Path('scripts/sugar/object_predictor', name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed collection source: ' + name)
    reuse = protocol['reuse_train_cases']
    if set(reuse) != {'5000', '5007', '5024'}:
        raise ValueError('Only the three declared TRAIN cases may be reused')
    for episode, source in reuse.items():
        if next(c for c in configs if c['episode'] == int(episode))['split'] != 'train':
            raise ValueError('Reused case is not TRAIN')
        verify_condition(Path(source), sources)
    lock = (root / 'collection.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'cases').mkdir(exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    records = []
    new_controls = 0
    reused_controls = 0
    for split in ('train', 'val', 'test'):
        if split != 'train' and not json.loads((root / 'TRAIN_QUALIFICATION.json').read_text())['passed']:
            raise RuntimeError('TRAIN acquisition qualification failed')
        for config in [c for c in configs if c['split'] == split]:
            episode = config['episode']
            dest = root / 'cases' / f'episode_{episode}'
            if dest.exists():
                raise RuntimeError('Existing case requires explicit recovery, not overwrite: ' + str(dest))
            if str(episode) in reuse:
                source = Path(reuse[str(episode)])
                dest.mkdir()
                for name in (f'episode_{episode}.npz', f'episode_{episode}.json',
                             'contact_surface.npz', 'RESULT.json', 'PROTOCOL.json'):
                    shutil.copy2(source / name, dest / name)
                write_json(dest / 'REUSED_SOURCE.json', dict(source=str(source),
                    scope='Completed TRAIN diagnostic, same condition and sources; no repeated physics.'))
                reused_controls += 2400
            else:
                command = [sys.executable, '-m', 'scripts.sugar.object_predictor.collect_dense_grip',
                    '--protocol', str(root / 'PROTOCOL.json'), '--episode', str(episode),
                    '--output', str(dest), '--surface-feedback', '--frames', '2400',
                    '--force-gain', '.000025', '--sensor-coverage', 'continuous_palmar_v1',
                    '--response-gain', '--purpose', 'diagnostic']
                print('COLLECT_START', episode, split, flush=True)
                with (root / 'logs' / f'episode_{episode}.log').open('x') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                new_controls += 2400
            verify_condition(dest, sources)
            record = annotate(dest, config)
            record['reused'] = str(episode) in reuse
            records.append(record)
            write_json(root / 'COLLECTION_RESULT.json', dict(complete=False, records=records,
                new_controls=new_controls, reused_controls=reused_controls,
                new_model_forwards=0, new_optimizer_updates=0))
            print('COLLECT_COMPLETE', json.dumps(record), flush=True)
        if split == 'train':
            groups = {}
            for group in range(8):
                parts = [r for r in records if r['geometry_group'] == group]
                groups[group] = dict(episodes=[r['episode'] for r in parts],
                    label_frames=[r['airborne_hold_label_frames'] for r in parts],
                    covered=len(parts) == 4 and all(r['airborne_hold_label_frames'] >= 100 for r in parts))
            covered = sum(g['covered'] for g in groups.values())
            qualification = dict(passed=covered >= 4, covered_geometry_groups=covered, groups=groups,
                rule='At least 4/8 TRAIN groups; all four mass x grip settings each retain >=100 airborne bilateral hold labels.',
                not_controller_acceptance=True, no_failed_cases_removed=True,
                condition='continuous_palmar_v1 + response_gain; separate from original anatomical27 corpus')
            write_json(root / 'TRAIN_QUALIFICATION.json', qualification)
            print('TRAIN_COVERAGE', json.dumps(qualification), flush=True)
            if not qualification['passed']:
                raise SystemExit(2)
    write_json(root / 'COLLECTION_RESULT.json', dict(complete=True, records=records,
        new_controls=new_controls, reused_controls=reused_controls,
        new_model_forwards=0, new_optimizer_updates=0))
    print('COLLECTION_COMPLETE', len(records), new_controls, reused_controls, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
