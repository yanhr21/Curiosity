"""Recover known reference packets and compare against actual live command records.

This reads desired motion banks, not future simulator observations. It creates
no physical transitions and does not replace the original PhysX trajectories.
"""
import ast
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'


def main():
    run=BASE/'tracker_bcppo_refined_pair_low_exploration'
    # Execute the exact pure-tensor official function without importing Kit's
    # omni.log on the login node. No physics or CUDA initialization is needed.
    source=ROOT/'external/IsaacLab/source/isaaclab/isaaclab/utils/math.py'
    tree=ast.parse(source.read_text())
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='quat_apply_inverse')
    function.decorator_list=[]
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    inverse=namespace['quat_apply_inverse']
    motions=[]
    for folder in sorted((BASE/'tracker_refined96_90_feasibility/motions').glob('data_*')):
        motion=dict(np.load(folder/'robot_50hz.npz',allow_pickle=False))
        motion['contact']=np.load(folder/'contact_labels_50hz.npy',allow_pickle=False)
        motions.append(motion)
    traces={arm:dict(np.load(run/name/'TRACE.npz',allow_pickle=False)) for arm,name in
            [('original','teacher_supported_replay'),('alternate','teacher_supported_alternate')]}
    packets={};errors={}
    for arm,trace in traces.items():
        rows=[]
        for step,(selected,frame) in enumerate(zip(trace['reference_id'],trace['reference_frame'])):
            index={96:0,90:1}[int(selected)]
            if step<158 and index:raise RuntimeError('Alternate reference leaked before selection')
            motion=motions[index];times=np.minimum(int(frame)+5*np.arange(8),len(motion['joint_pos'])-1)
            q=torch.as_tensor(motion['body_quat_w'][times,0],dtype=torch.float32)
            lin=inverse(q,torch.as_tensor(motion['body_lin_vel_w'][times,0],dtype=torch.float32))
            ang=inverse(q,torch.as_tensor(motion['body_ang_vel_w'][times,0],dtype=torch.float32))
            rows.append(np.concatenate([motion['joint_pos'][times],lin.numpy(),ang.numpy(),motion['contact'][times,None]],axis=-1).astype(np.float32))
        packets[arm]=np.asarray(rows)[:,None]
        errors[arm+'_current_command']=float(np.max(abs(packets[arm][:,:,0]-trace['reference_command'])))
        if 'planned_reference_command' in trace:
            errors[arm+'_all_live_future_packets']=float(np.max(abs(packets[arm]-trace['planned_reference_command'])))
    prefix=bool(np.array_equal(packets['original'][:158],packets['alternate'][:158]))
    passed=max(errors.values())<=1e-5 and prefix
    report=dict(execution_completed=True,passed=passed,max_absolute_errors=errors,required_max_error=1e-5,
                exact_preselection_plan_prefix=prefix,packet_shapes={k:list(v.shape) for k,v in packets.items()},
                official_function=str(source)+':quat_apply_inverse',physical_transitions_created=0,
                source_original=str(run/'teacher_supported_replay'),source_alternate=str(run/'teacher_supported_alternate'),
                provenance='Desired reference bank, recorded selected-reference identity/current frame, official pelvis-frame velocity transform; current packets checked on both real400-frame traces and all8-step packets checked against400 actual live alternate records.',
                coordinate_contract='Joint positions/contact and pelvis-frame velocities are invariant under the causal rigid reference yaw/translation. Omission of that world transform is accepted only when all actual packet readbacks meet1e-5.',
                limitation='Original full future packet was not recorded live. It is reference-derived, validated by both current streams and the entire alternate live packet stream; no claim of new rollout/recollection.')
    path=run/'KNOWN_REFERENCE_PLAN_AUDIT.json'
    if path.exists():raise RuntimeError('Do not overwrite reference audit')
    path.write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise RuntimeError('Known reference packet extraction failed actual command checks')
    np.savez(run/'KNOWN_REFERENCE_PLANS.npz',**packets)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
