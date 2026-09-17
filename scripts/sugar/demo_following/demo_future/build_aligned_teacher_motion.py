"""Original numeric-demo bank corresponding to the measured Refiner motion bank.

Uses original source poses/velocities and the already-recorded rigid reference
transforms and causal source clock. No simulated state becomes teacher reference.
This piecewise timeline adaptation must be physically checked before BCPPO.
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    plan=json.loads((run/'PROTOCOL.json').read_text())
    spec=importlib.util.spec_from_file_location('official_motion_math',ROOT/'SUGAR/scripts/sugar_rl/process_refiner_rollout.py')
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    out=run/'aligned_original_teacher_bank';out.mkdir(exist_ok=False)
    summary={}
    for index,arm in enumerate(('original','alternate')):
        trace=dict(np.load(run/arm/'TRACE.npz',allow_pickle=False))
        transform=json.loads((run/arm/'REFERENCE_TRANSFORM.json').read_text())
        names=np.load(run/arm/'STARTUP.npz')['body_names'].tolist()
        # Exact order used by the original command configuration and saved
        # reference-body positions, as independently audited by SMP schema.
        import sys
        sys.path.insert(0,str(ROOT/'scripts/sugar/smp'))
        from sugar_g1_box_schema import TRACKED_BODY_NAMES
        body_ids=[names.index(n) for n in TRACKED_BODY_NAMES]
        robot={};obj={};contacts=[]
        cache={}
        for source in plan['reference_pair']:
            folder=ROOT/f'SUGAR/data/CarryBox/data_{source:03d}'
            cache[source]=(dict(np.load(folder/'robot_50hz.npz',allow_pickle=False)),
                           pickle.load(open(folder/'obj_motion_global_50hz.pkl','rb')),
                           np.load(folder/'contact_labels_50hz.npy',allow_pickle=False))
        for source,frame in zip(trace['reference_id'].tolist(),trace['reference_frame'].tolist()):
            r,o,c=cache[source]
            for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'):
                robot.setdefault(key,[]).append(r[key][frame].copy())
            for key in ('obj_trans','obj_rot','obj_lin_vel','obj_ang_vel'):
                obj.setdefault(key,[]).append(o[key][frame].copy())
            contacts.append(c[frame])
        robot={k:np.asarray(v) for k,v in robot.items()};obj={k:np.asarray(v) for k,v in obj.items()}
        switch=plan['switch_control_frame'];q=np.asarray(transform['rotation_wxyz'],dtype=np.float32)
        shift=np.asarray(transform['translation_xyz'],dtype=np.float32)
        def rotate(v):return math.quat_apply_np(np.broadcast_to(q,v.shape[:-1]+(4,)),v)
        robot['body_pos_w'][switch:]=rotate(robot['body_pos_w'][switch:])+shift
        robot['body_quat_w'][switch:]=math.quat_mul_np(np.broadcast_to(q,robot['body_quat_w'][switch:].shape),robot['body_quat_w'][switch:])
        for key in ('body_lin_vel_w','body_ang_vel_w'):robot[key][switch:]=rotate(robot[key][switch:])
        obj['obj_trans'][switch:]=rotate(obj['obj_trans'][switch:])+shift
        obj['obj_rot'][switch:]=math.matrix_from_quat_np(q)@obj['obj_rot'][switch:]
        for key in ('obj_lin_vel','obj_ang_vel'):obj[key][switch:]=rotate(obj[key][switch:])
        box_error=float(np.max(np.abs(obj['obj_trans']-trace['reference_object_pos_w'][:,0])))
        body_error=float(np.max(np.abs(robot['body_pos_w'][:,body_ids]-trace['reference_body_pos_w'][:,0])))
        if max(box_error,body_error)>1e-5:raise RuntimeError(f'Original teacher bank does not reproduce selected reference: {box_error}, {body_error}')
        folder=out/f'data_{index:03d}';folder.mkdir()
        np.savez(folder/'robot_50hz.npz',**robot,fps=np.array([50.]))
        with open(folder/'obj_motion_global_50hz.pkl','wb') as f:pickle.dump(dict(obj,fps=50.),f)
        np.save(folder/'contact_labels_50hz.npy',np.asarray(contacts))
        summary[arm]=dict(frames=len(contacts),box_reference_max_error=box_error,
                          body_reference_max_error=body_error,source_ids=plan['reference_pair'],
                          output=str(folder.relative_to(ROOT)))
    result=dict(execution_completed=True,passed=True,arms=summary,
                provenance='Original numeric demonstrations, their original velocities/contact annotations, recorded causal source-frame mapping and one rigid reference transform. No actual executed joints/actions become original-teacher targets.',
                compatibility_boundary='At the selection boundary the teacher now sees the selected composite timeline in its future context. This may differ from prior online reference switching during the preceding8 frames. Cropped450-frame bank also changes its endpoint. Full frozen-teacher execution must be tested; no exact whole-observation equivalence claim.',
                next_action='Frozen full Refiner400-frame floor on corresponding original teacher bank, before any Tracker training.')
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
