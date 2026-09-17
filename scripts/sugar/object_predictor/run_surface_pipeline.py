"""Wait for qualified collection, then execute the declared serial full-model study."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ARMS = [('centroid_force', 'centroid', False), ('surface_force', 'surface', False),
        ('surface_force_airborne_mass_guard', 'surface', True)]


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    root = Path(args.output); root.mkdir(exist_ok=False)
    lock = (root / 'pipeline.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = Path(args.data)
    record = dict(line.split('=', 1) for line in (data / 'collection.process').read_text().splitlines())
    collector = int(record['child_pid']); status = data / 'collection.status'
    while not status.exists():
        try:
            os.kill(collector, 0)
        except ProcessLookupError:
            raise RuntimeError('Collection owner disappeared without a terminal status; no restart or training')
        print('WAIT_COLLECTION', collector, flush=True); time.sleep(10)
    terminal = dict(line.split('=', 1) for line in status.read_text().splitlines())
    if terminal['exit_code'] != '0':
        raise RuntimeError('Collection ended without qualification: ' + terminal['exit_code'])
    if not json.loads((data / 'COLLECTION_RESULT.json').read_text())['complete']:
        raise RuntimeError('Declared 48-case corpus is incomplete')
    for path in (data / 'TRAIN_QUALIFICATION.json', Path(args.qualification) / 'TRAINING_PATH_REPORT.json'):
        if not json.loads(path.read_text())['passed']:
            raise RuntimeError('Required qualification failed: ' + str(path))
    (root / 'PROTOCOL.json').write_text(json.dumps(dict(data=str(data),
        training=json.loads((data / 'PROTOCOL.json').read_text())['planned_matched_training'],
        evaluation=json.loads((data / 'EVALUATION_PROTOCOL.json').read_text()),
        qualification=args.qualification, collector_terminal=terminal), indent=2))
    def run(tag, arguments):
        print('START_STAGE', tag, flush=True)
        with (root / (tag + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-m', *arguments], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('COMPLETE_STAGE', tag, flush=True)
    common = ['--data', str(data), '--checkpoint', args.checkpoint]
    for arm, representation, guard in ARMS:
        command = ['scripts.sugar.object_predictor.train', *common, '--output', str(root / arm),
                   '--mode', 'geometry_contact_force', '--steps', '2000', '--evaluate-every', '500',
                   '--history', '32', '--history-policy', 'episode_uniform_recent', '--normal-policy', 'hand_surface',
                   '--time-scale-s', '150', '--stride', '25', '--batch', '4', '--train-sampler', 'matched_geometry',
                   '--seed', '310016', '--head-lr', '0.00001', '--deterministic-pooling',
                   '--disable-stochastic-regularizers', '--perception-corpus', '--input-representation', representation]
        if guard:
            command.append('--mass-airborne-guard')
        run('train_' + arm, command)
        result = json.loads((root / arm / 'RESULT.json').read_text())
        if not result['complete'] or result['steps'] != 2000:
            raise RuntimeError('Training endpoint incomplete')
    for arm, _, _ in ARMS:
        run('batch1_' + arm, ['scripts.sugar.object_predictor.evaluate_endpoint', *common,
            '--endpoint', str(root / arm / 'model.pt'), '--output', str(root / 'batch1' / arm)])
    run('compare', ['scripts.sugar.object_predictor.report_surface_training', '--root', str(root)])
    for arm in (ARMS[0][0], ARMS[-1][0]):
        run('visual_' + arm, ['scripts.sugar.object_predictor.evaluate_endpoint', *common,
            '--endpoint', str(root / arm / 'model.pt'), '--output', str(root / 'visual' / arm),
            '--stride', '5', '--skip-force-zero'])
    run('render', ['scripts.sugar.object_predictor.render_surface_estimates', '--root', str(root)])
    (root / 'PIPELINE_RESULT.json').write_text(json.dumps(dict(complete=True, optimizer_updates=6000,
        note='Execution complete; scientific acceptance is reported separately. No extra experiment follows.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--qualification', required=True)
    main(parser.parse_args())
