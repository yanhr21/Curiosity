"""Prospectively split full-mesh probing corpus; no model training or selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import numpy as np

SOURCE_FILES=('probe_scene.py','support_scene.py','qualify_probe.py','collect_newton.py','geometry.py')


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(310330);cases=[]
    for split,n in [('train',24),('val',6),('test',6)]:
        # Independent stratification in every split. TRAIN includes the stated
        # parameter-range boundaries; VAL/TEST use interior independent draws.
        values=[]
        for _ in range(4):
            order=rng.permutation(n)
            values.append(order/(n-1) if split=='train' else (order+rng.random(n))/n)
        values=np.stack(values,axis=1)
        for row in values:
            i=len(cases)
            cases.append(dict(episode=3000+i,split=split,mass=float(.25+.95*row[0]),
                              scale=(.85+.30*row[1:]).tolist(),seed=310430+i))
    sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in SOURCE_FILES}
    protocol=dict(backend='Newton/SolverMuJoCo',scene='full_mesh_three_direction_probe',
                  configurations=cases,frames_per_episode=7500,max_new_controls=270000,
                  dt=.02,substeps=8,canonical_initial_orientation=True,stable_servo=True,
                  parameter_ranges=dict(mass_kg=[.25,1.2],independent_xyz_scale=[.85,1.15]),
                  whole_configuration_split=dict(train=24,val=6,test=6),
                  seed=310330,source_sha256=sources,full_original_collision_meshes=True,
                  controller_inputs='clock and previous measured hand loads only',
                  all_physical_failures_retained=True,new_optimizer_updates=0,
                  scope='Known scanned asset family, varied pose/size/mass. Ground-supported probing supplies geometry, not directly observable mass.',
                  primary_geometry_evaluation_window_s=[140.,148.],
                  all_window_and_all_configuration_results_required=True,
                  failure_cases_not_removed_from_test=True)
    path=root/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol: raise RuntimeError('Prepared corpus/source differs; do not mutate a running collection')
    else: path.write_text(json.dumps(protocol,indent=2))
    return protocol


def main(args):
    root=Path(args.output);protocol=prepare(root)
    if args.prepare_only:
        print(json.dumps(protocol),flush=True);return
    if not os.environ.get('SLURM_STEP_ID'): raise RuntimeError('Collection requires retained compute step')
    # Preserve an interrupted active case separately; completed cases are never
    # recollected. Replacement of an interrupted attempt requires explicit audit.
    def terminate(signum,frame): raise KeyboardInterrupt(f'External termination signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    from .qualify_probe import main as qualify
    from .plot_collection import main as plot
    from .audit_probe import main as geometry_audit
    from .audit_probe_controller import main as controller_audit
    records=[]
    for case in protocol['configurations']:
        case_root=root/'cases'/f"episode_{case['episode']:04d}"
        record_path=root/f"episode_{case['episode']:04d}.json"
        if record_path.exists():
            record=json.loads(record_path.read_text())
            assert record['configuration']==case and (root/f"episode_{case['episode']:04d}.npz").exists()
            assert json.loads((case_root/'CONTROLLER_REPLAY_AUDIT.json').read_text())['passed']
            records.append(record);continue
        if case_root.exists():
            raise RuntimeError(f'Existing incomplete case requires artifact/termination audit; preserve {case_root}')
        print('PROBE_DATASET_CASE_START '+json.dumps(case),flush=True)
        qualify(argparse.Namespace(output=str(case_root),gentle=True,stable_servo=True,
                                   canonical_init=True,mass=case['mass'],scale=case['scale'],seed=case['seed']))
        # Finish the declared cohort even when contact qualification is negative;
        # the failure stays in its original split and denominator.
        result=json.loads((case_root/'RESULT.json').read_text())
        assert result['frames']==protocol['frames_per_episode']
        plot(str(case_root/'episode_0000.npz'));geometry_audit(case_root);controller_audit(case_root)
        link=root/f"episode_{case['episode']:04d}.npz"
        link.symlink_to(Path('cases')/case_root.name/'episode_0000.npz')
        record=dict(episode=case['episode'],split=case['split'],configuration=case,
                    frames=result['frames'],mass_kg=case['mass'],scale=case['scale'],seed=case['seed'],
                    source_case=str(case_root),contact_qualification_passed=result['qualification_passed'],
                    phase_contacts=result['phases'],acquisition='ground_supported_multiface_probe',
                    mass_is_not_directly_observable_from_static_probe=True)
        record_path.write_text(json.dumps(record,indent=2));records.append(record)
        (root/'RESULT.partial.json').write_text(json.dumps(dict(episodes=records,complete=False,new_optimizer_updates=0),indent=2))
        print('PROBE_DATASET_CASE_COMPLETE '+json.dumps(record),flush=True)
    result=dict(episodes=records,complete=len(records)==36,
                total_actual_frames=sum(x['frames'] for x in records),
                contact_qualification_passes=sum(x['contact_qualification_passed'] for x in records),
                all_failures_retained=True,training_admitted=False,new_optimizer_updates=0)
    (root/'RESULT.json').write_text(json.dumps(result,indent=2));print('PROBE_DATASET_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--prepare-only',action='store_true');main(ap.parse_args())
