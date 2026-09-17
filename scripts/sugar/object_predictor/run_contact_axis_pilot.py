"""Fixed failed/successful pair for one observation-only closure intervention."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .retained_execution import require_active_resource


def main(root):
    resource=require_active_resource()
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    if [c['episode'] for c in protocol['configurations']]!=[5008,5000]:
        raise ValueError('Preserve the fixed failed case and successful regression case')
    baseline=Path(protocol['baseline_root'])
    source=json.loads((baseline/'PROTOCOL.json').read_text())
    for config in protocol['configurations']:
        if config!=next(c for c in source['configurations'] if c['episode']==config['episode']):
            raise ValueError('Intervention must preserve exact original case configuration')
    (root/'cases').mkdir(exist_ok=False);(root/'logs').mkdir(exist_ok=False)
    rows=[]
    for config in protocol['configurations']:
        episode=config['episode'];dest=root/'cases'/f'episode_{episode}'
        command=[sys.executable,'-P','-m','scripts.sugar.object_predictor.collect_dense_grip',
            '--protocol',str(root/'PROTOCOL.json'),'--episode',str(episode),'--output',str(dest),
            '--surface-feedback','--frames','2400','--force-gain','.000025',
            '--sensor-coverage','continuous_palmar_v1','--response-gain',
            '--controller-revision','sensor_feedback_v2',
            '--controller-intervention','observed_contact_axis_closure_v1','--purpose','diagnostic']
        print('CONTACT_AXIS_PILOT_START',episode,flush=True)
        with (root/'logs'/f'episode_{episode}.log').open('x') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        old=json.loads((baseline/'cases'/f'episode_{episode}'/'RESULT.json').read_text())
        row=dict(episode=episode,exit_code=rc,baseline=old,complete=False,passed=False)
        if rc==0 and (dest/'RESULT.json').exists():
            current=json.loads((dest/'RESULT.json').read_text())
            if current.get('controller_intervention')!='observed_contact_axis_closure_v1':
                raise ValueError('Missing actual intervention in recording')
            row.update(complete=True,passed=current['passed'],current=current)
        rows.append(row)
        result=dict(complete=len(rows)==2 and all(r['complete'] for r in rows),
            paired_qualification_passed=len(rows)==2 and all(r['passed'] for r in rows),
            cases=rows,planned_cases=2,retained_resource=resource,new_model_forwards=0,new_optimizer_updates=0,
            scope='One closure-direction intervention; not all16 qualification or learned policy; all original physical checks retained')
        (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        print('CONTACT_AXIS_PILOT_COMPLETE',episode,row['complete'],row['passed'],flush=True)
    return 0 if result['paired_qualification_passed'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().root))
