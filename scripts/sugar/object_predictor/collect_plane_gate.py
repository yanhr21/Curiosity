"""Run the two declared plane-admission diagnostics from an isolated overlay.

Existing prospective cases remain the baseline; their observations/results are
never rewritten. This is a post-hoc acquisition diagnostic, not a new held-out
test or a predictor-training dataset.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .collect_response_corpus import write_json
from .collect_surface_corpus import annotate


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use the retained compute step')
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    assert [c['episode'] for c in protocol['configurations']]==[6000,6004]
    assert protocol['plane_fit_min_load_n']==.2
    overlay=Path(protocol['overlay']).resolve()
    for name,expected in protocol['collector_source_sha256'].items():
        spec=importlib.util.find_spec('scripts.sugar.object_predictor.'+Path(name).stem)
        path=Path(spec.origin).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Unexpected actual imported collector source: '+str(path))
        if name in ('collect_dense_grip.py','surface_grip_scene.py') and not path.is_relative_to(overlay):
            raise ValueError('Namespace resolved original source instead of isolated diagnostic overlay')
    baseline_root=Path(protocol['baseline_root'])
    original=json.loads((baseline_root/'PROTOCOL.json').read_text())
    original_cases={c['episode']:c for c in original['configurations']}
    for config in protocol['configurations']:
        assert config==original_cases[config['episode']]
        assert (baseline_root/'cases'/f'episode_{config["episode"]}'/'RESULT.json').is_file()
    (root/'cases').mkdir(exist_ok=False); (root/'logs').mkdir(exist_ok=False)
    records=[]
    for config in protocol['configurations']:
        episode=config['episode']; destination=root/'cases'/f'episode_{episode}'
        command=[sys.executable,'-P','-m','scripts.sugar.object_predictor.collect_dense_grip',
                 '--protocol',str(root/'PROTOCOL.json'),'--episode',str(episode),'--output',str(destination),
                 '--surface-feedback','--frames','2400','--force-gain','.000025',
                 '--sensor-coverage','continuous_palmar_v1','--response-gain','--purpose','diagnostic',
                 '--plane-fit-min-load-n','.2']
        print('PLANE_GATE_COLLECT_START',episode,flush=True)
        with (root/'logs'/f'episode_{episode}.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        saved=json.loads((destination/'PROTOCOL.json').read_text())
        if saved['source_sha256']!=protocol['collector_source_sha256']:
            raise ValueError('Recorded collector source differs from actual qualified imports')
        for key,value in dict(plane_fit_min_load_n=.2,frames=2400,sensor_coverage='continuous_palmar_v1',
                              response_gain=True,object_sdf_resolution=128,force_gain_m_per_ns=.000025,
                              force_integral_time_s=0.,purpose='diagnostic',original_full_mesh_and_physics=True).items():
            if saved.get(key)!=value:
                raise ValueError('Unexpected diagnostic condition: '+key)
        baseline=json.loads((baseline_root/'cases'/f'episode_{episode}'/'PROTOCOL.json').read_text())
        for key in ('dt','mass_kg','target_load_n','seed','scale','geometry_group','approach_angle_deg',
                    'yaw_delta_deg','lift_height_m','lateral_xy','criteria','stability_criteria',
                    'readiness','alignment_rate_limit_deg_s','response_rule'):
            if saved[key]!=baseline[key]:
                raise ValueError('Change beyond plane admission: '+key)
        record=annotate(destination,config); records.append(record)
        write_json(root/'COLLECTION_RESULT.json',dict(complete=len(records)==2,records=records,
            new_controls=2400*len(records),baseline_reused_controls=4800,
            planned_new_controls=4800,model_forwards=0,model_parameter_updates=0,
            scope='Two post-hoc matched acquisition diagnostics, not generalization or predictor success'))
        print('PLANE_GATE_COLLECT_COMPLETE',json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
