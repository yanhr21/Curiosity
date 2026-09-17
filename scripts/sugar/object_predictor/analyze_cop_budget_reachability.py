"""Saved-only frozen-contact reachability bounds, not a physics rollout."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .hand_floor_feasibility import swept_min_height
from .run_allocated_floor_cop_pilot import sha,write,CLEARANCE_M


def left_only_projection(tangent,normal,budget,floor_margin):
    """Closest tangential translation to error: disk radius B and dz>=-margin."""
    z=np.array([0.,0.,1.]);u=z-normal*normal[2];scale=np.linalg.norm(u)
    if scale<1e-12:
        d=tangent*min(1.,budget/max(np.linalg.norm(tangent),1e-30))
        return d,float(np.linalg.norm(tangent-d))
    u/=scale;v=np.cross(normal,u)
    a,b=tangent@u,tangent@v;lower=-floor_margin/scale
    desired=np.array([a,b]);length=np.linalg.norm(desired)
    candidate=desired*min(1.,budget/max(length,1e-30))
    if candidate[0]<lower:
        # Initial poses have positive reserve, so lower<=0<=budget.
        candidate=np.array([lower,np.clip(b,-np.sqrt(max(0.,budget**2-lower**2)),np.sqrt(max(0.,budget**2-lower**2)))])
    d=candidate[0]*u+candidate[1]*v
    return d,float(np.linalg.norm(tangent-d))


def run(root,output):
    folder=root/'cases/episode_5014'
    with np.load(folder/'episode_5014.npz',allow_pickle=False) as z:
        keys=['timestamp_s','hand_pose_w','normal_load_n']+[k for k in z.files if k.startswith('validation_controller_')]
        a={k:z[k] for k in keys}
    v=lambda k:a['validation_controller_'+k]
    times=a['timestamp_s'];exhausted=np.flatnonzero(v('cop_travel_m')[:,1]>=.10-1e-12)
    if not len(exhausted):raise ValueError('Actual right-budget exhaustion absent')
    first=int(exhausted[0])
    from sugar_newton.hand.patches import load_hand_mesh
    meshes=[np.asarray(load_hand_mesh(side).vertices,np.float32).astype(float) for side in ('left','right')]
    rows=[];drift=[]
    # Next command's observation is the state AFTER the exhausting command.
    # Do not evaluate the preceding state with its final80um still available.
    for i in range(first+1,int(np.searchsorted(times,40.,side='right'))):
        if not v('cop_valid')[i].all():continue
        e=v('cop_tangent_error_m')[i];centers=v('cop_world_m')[i]
        delta=centers[1]-centers[0];normal_part=delta-e;s=np.linalg.norm(normal_part)
        if s<=1e-8:raise ValueError('Degenerate saved valid contact separation')
        n=normal_part/s;length=np.linalg.norm(e);allowed=s*np.tan(np.deg2rad(5.))
        if not np.isclose(np.degrees(np.arctan2(length,s)),v('cop_line_angle_deg')[i],rtol=0,atol=1e-10):
            raise ValueError('Stored contact angle is not the positive-separation geometry used here')
        needed=max(0.,length-allowed);direction=e/max(length,1e-30)
        pose=a['hand_pose_w'][i-1].astype(float)
        minimum=np.array([swept_min_height(m,p,p) for m,p in zip(meshes,pose,strict=True)])
        target=pose.copy();target[1,:3]-=needed*direction
        sweep=np.array([swept_min_height(m,p,q) for m,p,q in zip(meshes,pose,target,strict=True)])
        remaining_time=max(0.,40.-times[i-1]);ready_time=max(0.,39.-times[i-1])
        budget=np.maximum(0.,.10-(v('cop_travel_m')[i-1] if i else np.zeros(2)))
        left_limit=min(float(budget[0]),.004*remaining_time)
        left_delta,left_error=left_only_projection(e,n,left_limit,max(0.,minimum[0]-CLEARANCE_M))
        load=a['normal_load_n'][i-1].reshape(2,27).sum(1)
        rows.append(dict(frame=i,command_time_s=float(times[i]),observation_time_s=float(times[i-1]),
            angle_deg=float(v('cop_line_angle_deg')[i]),normal_separation_m=float(s),tangent_error_m=e.tolist(),
            tangent_error_norm_m=float(length),admission_tangent_allowance_m=float(allowed),
            right_only_required_translation_m=(-needed*direction).tolist(),minimum_relative_adjustment_m=float(needed),
            remaining_budget_before_command_m=budget.tolist(),remaining_time_to40_s=float(remaining_time),
            time_at4mmps_s=float(needed/.004),right_capacity_without_travel_cap_m=float(.004*remaining_time),
            reserve1s_admission_capacity_without_travel_cap_m=float(.004*ready_time),
            frozen_right_only_within_speed_and_deadline=bool(needed<=.004*remaining_time),
            frozen_right_only_within_speed_and_one_second_readiness=bool(needed<=.004*ready_time),
            frozen_right_only_fullmesh_floor_safe=bool((sweep>=CLEARANCE_M).all()),right_only_swept_min_z_m=sweep.tolist(),
            left_only_best_translation_m=left_delta.tolist(),left_only_best_remaining_tangent_m=left_error,
            left_only_with_budget_floor_speed_can_align=bool(left_error<=allowed),
            observed_previous_loads_approx_n=load.tolist(),recorded_ready_s=float(v('ready_seconds')[i])))
        if i+1<len(times) and v('cop_valid')[i+1].all():
            actual=a['hand_pose_w'][i].astype(float)
            oldr=Rotation.from_quat(pose[:,3:]).as_matrix();newr=Rotation.from_quat(actual[:,3:]).as_matrix()
            local=np.einsum('bji,bj->bi',oldr,centers-pose[:,:3])
            transported=np.einsum('bij,bj->bi',newr,local)+actual[:,:3]
            predicted=transported[1]-transported[0]
            nextdelta=v('cop_world_m')[i+1,1]-v('cop_world_m')[i+1,0]
            residual=nextdelta-predicted;projected=residual-n*(residual@n)
            drift.append(dict(command_time_s=float(times[i]),tangent_contact_relocation_residual_m=projected.tolist(),norm_m=float(np.linalg.norm(projected))))
    output.mkdir(parents=True,exist_ok=False)
    norms=np.array([d['norm_m'] for d in drift])
    result=dict(scope='Frozen observed-contact geometry, fixed normal and hand rotations; no object GT. Static necessary/sufficient statements apply only to this restricted translation problem, not true physical lifting or future contact observations.',
        first_exhausted_command_time_s=float(times[first]),rows=rows,drift=drift,
        contact_relocation_residual=dict(count=len(norms),median_m=float(np.median(norms)),p95_m=float(np.quantile(norms,.95)),maximum_m=float(norms.max())),
        source_sha256={str(p):sha(p) for p in (Path(__file__),folder/'episode_5014.npz',root/'PROTOCOL.json')},
        limits=['Right travel cap is an explicit controller constraint, not a user physical requirement.',
            'No hypothetical steps are summed into a new trajectory.',
            'Keeping current CoP fixed to hand and normal fixed does not preserve actual contact force, pressure distribution, or object motion.',
            'One second readiness capacity assumes immediate original force/fit qualification; real force/geometry gates remain mandatory.',
            'Left-only optimum excludes parent rotation/normal displacement and changed contact geometry; failure is not a global impossibility proof.'],
        model_forwards=0,physics_controls=0,optimizer_updates=0)
    write(output/'RESULT.json',result)
    x=rows[0]
    (output/'REPORT.md').write_text('# Saved contact-geometry reachability\n\n'+
        f"At command {x['command_time_s']:.2f}s (previous observation {x['observation_time_s']:.2f}s), tangent error {x['tangent_error_norm_m']*1000:.4f}mm, original5deg allowable {x['admission_tangent_allowance_m']*1000:.4f}mm; frozen-contact minimum relative correction {x['minimum_relative_adjustment_m']*1000:.4f}mm.\n\n"+
        f"Right-only correction takes {x['time_at4mmps_s']:.4f}s at original4mm/s; remaining time {x['remaining_time_to40_s']:.2f}s, capacity before39s to reserve1s readiness {x['reserve1s_admission_capacity_without_travel_cap_m']*1000:.3f}mm. Full-mesh floor safe for that hypothetical pure translation: {x['frozen_right_only_fullmesh_floor_safe']}.\n\n"+
        f"Restricted left-only correction with original budget/time and floor can meet angle: {x['left_only_with_budget_floor_speed_can_align']}. This excludes rotation, changing plane normals and contact migration.\n\n"+
        f"Actual observed tangential contact-relocation residual after transporting each previous CoP with the executed hand pose: median {np.median(norms)*1000:.4f}mm/control, p95 {np.quantile(norms,.95)*1000:.4f}mm, max {norms.max()*1000:.4f}mm. This is why single-state feasibility does not certify a successful dynamic trajectory.\n\n"+
        '\n'.join('- '+s for s in result['limits'])+'\n')
    print(json.dumps({'first':x,'contact_relocation_residual':result['contact_relocation_residual']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
