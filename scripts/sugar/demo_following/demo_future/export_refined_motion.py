"""Official Refiner-to-motion conversion of newly measured full PhysX traces.

This creates reference motion, never executed-action labels or live tactile input.
Uses the official formatter and its exact max-over-sensor-history contact rule.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    plan=json.loads((run/'PROTOCOL.json').read_text())
    pairing=json.loads((run/'READBACK.json').read_text())
    if not pairing['data_pair_feasibility_passed']:
        raise RuntimeError('Source pair did not pass')
    source=ROOT/'SUGAR/scripts/sugar_rl/process_refiner_rollout.py'
    spec=importlib.util.spec_from_file_location('official_refiner_formatter',source)
    formatter=importlib.util.module_from_spec(spec);spec.loader.exec_module(formatter)
    out=run/'refined_motion_export';out.mkdir(exist_ok=False)
    details={};arrays={};startups={}
    for source_id,arm in enumerate(('original','alternate')):
        folder=run/f'export_{arm}'
        result=json.loads((folder/'RESULT.json').read_text())
        if not result['full_budget_without_reset'] or not result['teacher_frozen']['passed']:
            raise RuntimeError('New full-history collection incomplete or unfrozen')
        trace=dict(np.load(folder/'TRACE.npz',allow_pickle=False));arrays[arm]=trace
        old=dict(np.load(run/arm/'TRACE.npz',allow_pickle=False))
        startup=dict(np.load(folder/'STARTUP.npz',allow_pickle=False));startups[arm]=startup
        fields=('robot_body_state_before_w','object_state_before_w','joint_pos_before',
                'joint_vel_before','teacher_observation','executed_action','reference_id')
        differences={k:float(np.max(np.abs(trace[k]-old[k]))) for k in fields}
        # Recollection is for the missing sensor history, not a new replicate.
        # Require exact reproduction of the already-inspected physical pair.
        if any(differences.values()):
            raise RuntimeError(f'Recollection differs from original {arm}: {differences}')
        body=trace['robot_body_state_before_w'][:,0]
        obj=trace['object_state_before_w'][:,0]
        forces=trace['contact_force_history_before_w'][:,0]
        max_force=np.linalg.norm(forces,axis=-1).max(axis=-1)
        hands=(max_force[:,0]>.1)&(max_force[:,1]>.1)
        raw=dict(joint_pos=trace['joint_pos_before'][:,0],joint_vel=trace['joint_vel_before'][:,0],
                 body_pos_w=body[...,:3],body_quat_w=body[...,3:7],
                 body_lin_vel_w=body[...,7:10],body_ang_vel_w=body[...,10:13],
                 obj_pos_w=obj[:,:3],obj_quat_w=obj[:,3:7],obj_lin_vel_w=obj[:,7:10],
                 obj_ang_vel_w=obj[:,10:13],hands_contact_label=hands)
        path=out/f'motion_{source_id}_env_0_t0-{len(obj)-1}_idx_0.npz'
        np.savez_compressed(path,**raw)
        formatter.process_data_to_rl_dataset(str(path),str(out),'hands_contact_label',
                                             stabilize_initial_frames_flag=False)
        target=out/'rl_dataset'/f'data_{source_id:03d}_000_t0'
        loaded=dict(np.load(target/'robot_50hz.npz',allow_pickle=False))
        if not all(np.array_equal(loaded[k],raw[k]) for k in loaded):
            raise RuntimeError('Official formatter altered measured robot motion')
        if not np.array_equal(np.load(target/'contact_labels_50hz.npy'),hands):
            raise RuntimeError('Official formatter contact labels differ')
        details[arm]=dict(reference_source=plan['reference_pair'][source_id],
                          output_motion=str(target.relative_to(ROOT)),frames=len(obj),
                          reproduced_prior_trace_max_errors=differences,
                          body_names=startup['body_names'].tolist(),joint_names=startup['joint_names'].tolist(),
                          bilateral_contact_frames=int(hands.sum()),contact_history_shape=list(forces.shape))
    switch=plan['switch_control_frame']
    checks={k:bool(np.array_equal(arrays['original'][k][:switch],arrays['alternate'][k][:switch]))
            for k in ('robot_body_state_before_w','object_state_before_w','joint_pos_before','executed_action','contact_force_history_before_w')}
    checks['startup_equal']=all(np.array_equal(v,startups['alternate'][k]) for k,v in startups['original'].items())
    if not all(checks.values()):raise RuntimeError('Full-history recollection pair lost its common prefix')
    result=dict(execution_completed=True,passed=True,official_formatter=str(source.relative_to(ROOT)),
                checks=checks,arms=details,stabilize_initial_frames=False,
                semantic_boundary='Measured PhysX poses become reference-motion targets through the official pipeline. They are not29-D executed actions. Reference bilateral-contact labels are not live tactile input. Both original and recollected actual-action traces are preserved.',
                next_action='Test released full Tracker on these refined motions; no Generator training before verifying executable data.')
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
