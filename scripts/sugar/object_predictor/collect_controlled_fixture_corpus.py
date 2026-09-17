"""Explicit controlled-grasp TRAIN corpus, not original blind-approach repair.

The fixed approach map was chosen after the recorded four-case pilot and the
earlier freeze-alignment pair. All sixteen mass/load combinations are collected
afresh once; no failures are replaced and no model is launched automatically.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import traceback

from .overfit_data import FIXED_EPISODES, fixed_protocol
from .retained_execution import require_active_resource
from .run_canonical_approach_fixture_pilot import (
    CHECKS, INTERVENTION, ORIGINAL_PROTOCOL, collection_command, validate_saved_case)

STUDY = 'controlled_fixture16_v1'
APPROACH_BY_GROUP = {0: 0., 2: 0., 3: 90., 5: 0.}


def protocol():
    source = json.loads(ORIGINAL_PROTOCOL.read_text())
    original = deepcopy(source['configurations'])
    if tuple(c['episode'] for c in original) != FIXED_EPISODES:
        raise ValueError('Original fixed sixteen TRAIN identities changed')
    configs = [dict(c, approach_angle_deg=APPROACH_BY_GROUP[c['geometry_group']]) for c in original]
    return dict(study=STUDY, source_original_protocol=str(ORIGINAL_PROTOCOL),
        original_configurations=original, configurations=configs,
        approach_by_group={str(k): v for k, v in APPROACH_BY_GROUP.items()},
        approach_selection='Manually fixed from prior physical pilots; known-scene collection, not a learned grasp planner',
        scope='Controlled known-scene TRAIN fixture with kinematic hands and a free dynamic object; not original blind-approach qualification or generalization',
        controller_intervention=INTERVENTION, overfit_data=fixed_protocol(),
        controls_per_case=2400, physics_substeps=8, planned_controls=38400,
        original_physical_checks=list(CHECKS),
        requested_approach_evaluation=[dict(episode=a['episode'],
            original_requested_approach_deg=a['approach_angle_deg'],
            executed_approach_deg=b['approach_angle_deg'],
            requested_approach_status='UNCHANGED_EXECUTED' if a['approach_angle_deg']==b['approach_angle_deg'] else 'NOT_EXECUTED_REPLACED')
            for a, b in zip(original, configs)],
        late_mass_check=dict(frame=2381, history=32, history_interval_s=.02,
                             criteria=fixed_protocol()['mass_criteria']),
        gate='Every one of sixteen fresh attempts must pass all original twelve physical checks and fixed frame2381 mass support qualification',
        state_supervision='Original unconditional 80-state fit remains invalid where identical no-contact inputs have incompatible state labels. This collection does not authorize that training.',
        gt_usage='Existing known-mesh object initialization and evaluation labels only; no GT state fed to feedback controller or model observation allowlist',
        predictor_claim_limit='Approach and proprioception correlate with scene geometry; matched trained proprio-only control is required before claiming a tactile benefit',
        automatic_training=False)


def main(root):
    resource = require_active_resource()
    declared = json.loads((root / 'PROTOCOL.json').read_text())
    for key, value in protocol().items():
        if declared.get(key) != value:
            raise ValueError('Controlled fixture protocol changed: ' + key)
    for name in ('cases', 'logs'):
        (root / name).mkdir(exist_ok=False)
    records = []
    for config in declared['configurations']:
        episode = config['episode']
        dest = root / 'cases' / f'episode_{episode}'
        row = dict(episode=episode, split='train', geometry_group=config['geometry_group'],
                   source=str(dest.resolve()), complete=False, controller_passed=False)
        print('CONTROLLED_FIXTURE_START', episode, flush=True)
        try:
            with (root / 'logs' / f'episode_{episode}.log').open('x') as log:
                code = subprocess.run(collection_command(root, config), stdout=log,
                    stderr=subprocess.STDOUT, pass_fds=(9,)).returncode
            if code:
                raise RuntimeError('Collector runtime exit ' + str(code))
            validation = validate_saved_case(dest, config, declared)
            row.update(complete=True, controller_passed=validation['physical_passed'],
                controller_checks=validation['physical_checks'], values=validation['physical_values'],
                late_mass_available=validation['late_mass_available'],
                late_mass_window=validation['late_mass_window'],
                com_transform_max_abs_error_m=validation['actual_com_transform_max_error_m'],
                frames=2400)
        except Exception as exc:
            row.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        records.append(row)
        result = dict(study=STUDY, scope=declared['scope'], records=records,
            complete=len(records)==16 and all(r['complete'] for r in records),
            qualification_passed=len(records)==16 and all(r['complete'] and
                r['controller_passed'] and r.get('late_mass_available', False) for r in records),
            attempted_configurations=len(records), planned_configurations=16,
            original_requested_approach_regression_repaired=False,
            controller_passes=sum(r['controller_passed'] for r in records),
            retained_resource=resource, new_model_forwards=0, new_optimizer_updates=0)
        temporary = root / 'COLLECTION_RESULT.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(root / 'COLLECTION_RESULT.json')
        print('CONTROLLED_FIXTURE_COMPLETE', episode, row['complete'], row['controller_passed'], flush=True)
    (root / 'QUALIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['qualification_passed'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
