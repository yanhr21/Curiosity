"""Saved-only 5014 blocked-command geometry; no physical counterfactual claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sugar_newton.hand.patches import load_hand_mesh
from .hand_floor_feasibility import swept_min_height


def run(root,output):
    folder=root/'cases/episode_5014';file=folder/'episode_5014.npz'
    with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    v=lambda k:a['validation_controller_'+k]
    blocked=np.flatnonzero(v('floor_blocked'))
    if len(blocked)!=78 or not (v('lift_start_s')[blocked]<0).all():
        raise ValueError('Require the completed declared78 prelift blocked clocks')
    if output.exists():raise FileExistsError(output)
    meshes=[np.asarray(load_hand_mesh(side).vertices,np.float32).astype(float) for side in ('left','right')]
    names=('full_requested','parent_without_cop','left_cop_zero','right_cop_zero')
    minz=[];advected_errors=[];translation_errors=[];original_errors=[];angles=[];roundoff=[];rows=[]
    for i in blocked:
        old=a['hand_pose_w'][i-1].astype(float)
        target=v('floor_requested_target_pose_w')[i].astype(float)
        delta=v('floor_requested_cop_velocity_m_s')[i].astype(float)*.02
        if not v('cop_valid')[i].all() or not v('fit_valid')[i].all():raise ValueError('Bilateral observed planes required')
        parent=target.copy();parent[:,:3]-=delta
        left_zero=target.copy();left_zero[0,:3]-=delta[0]
        right_zero=target.copy();right_zero[1,:3]-=delta[1]
        variants=(target,parent,left_zero,right_zero)
        heights=np.array([[swept_min_height(m,p,q) for m,p,q in zip(meshes,old,proposed,strict=True)] for proposed in variants])
        normal=v('fit_normal_w')[i,0]-v('fit_normal_w')[i,1]
        normal/=np.linalg.norm(normal)
        centers=v('cop_world_m')[i].astype(float)
        def tangent_norm_angle(c):
            separation=c[1]-c[0];normal_distance=float(separation@normal)
            tangent=separation-normal*normal_distance
            return np.linalg.norm(tangent),float(np.degrees(np.arctan2(np.linalg.norm(tangent),normal_distance)))
        original,angle=tangent_norm_angle(centers)
        np.testing.assert_allclose(original,np.linalg.norm(v('cop_tangent_error_m')[i]),rtol=0,atol=1e-12)
        local=Rotation.from_quat(old[:,3:]).inv().apply(centers-old[:,:3])
        # This transports the already observed points as fixed hand-local points.
        # The actual contact field/pressure distribution after a new motion is unknown.
        transported=[Rotation.from_quat(p[:,3:]).apply(local)+p[:,:3] for p in variants]
        predicted=np.array([tangent_norm_angle(c) for c in transported])
        only_translation=[]
        for d in (delta,np.zeros_like(delta),np.stack([np.zeros(3),delta[1]]),np.stack([delta[0],np.zeros(3)])):
            only_translation.append(tangent_norm_angle(centers+d)[0])
        # Parent poses were not saved separately; subtraction of a float32
        # in-place addition reconstructs them only within coordinate roundoff.
        # Record a conservative four-ULP bound, never call this exact replay.
        scale=np.maximum(np.abs(target[:,:3]),np.abs(parent[:,:3])).astype(np.float32)
        ulp_bound=float(4*np.abs(np.spacing(scale)).max())
        rows.append(dict(frame=int(i),time_s=float(a['timestamp_s'][i]),
            minimum_z_m={name:heights[k].tolist() for k,name in enumerate(names)},
            original_tangent_error_m=float(original),original_angle_deg=angle,
            frozen_point_tangent_error_m={name:float(predicted[k,0]) for k,name in enumerate(names)},
            frozen_point_angle_deg={name:float(predicted[k,1]) for k,name in enumerate(names)},
            translation_only_tangent_error_m={name:float(only_translation[k]) for k,name in enumerate(names)},
            requested_cop_velocity_m_s=(delta/.02).tolist(),parent_reconstruction_four_ulp_m=ulp_bound))
        minz.append(heights);advected_errors.append(predicted[:,0]);angles.append(predicted[:,1])
        translation_errors.append(only_translation);original_errors.append(original);roundoff.append(ulp_bound)
    heights=np.asarray(minz);predicted=np.asarray(advected_errors);original=np.asarray(original_errors)
    bounds=np.asarray(roundoff)
    summary=dict(episode=5014,clocks=78,first_s=rows[0]['time_s'],last_s=rows[-1]['time_s'],
        safe_sweeps_at_original_10um_reserve={name:int((heights[:,k,:]>=1e-5).all(1).sum()) for k,name in enumerate(names)},
        global_min_z_m={name:heights[:,k,:].min(0).tolist() for k,name in enumerate(names)},
        parent_safe_even_after_four_ulp_bound=int((heights[:,1,:]-bounds[:,None]>=1e-5).all(1).sum()),
        left_cop_zero_safe_even_after_four_ulp_bound=int((heights[:,2,:]-bounds[:,None]>=1e-5).all(1).sum()),
        parent_reconstruction_max_four_ulp_m=float(bounds.max()),
        left_cop_zero_frozen_point_error_improves_vs_parent=int((predicted[:,2]<predicted[:,1]).sum()),
        left_cop_zero_frozen_point_error_improves_vs_observed=int((predicted[:,2]<original).sum()),
        left_cop_zero_error_change_vs_parent_m_quantiles=np.quantile(predicted[:,2]-predicted[:,1],[0,.5,1]).tolist(),
        left_cop_zero_error_change_vs_observed_m_quantiles=np.quantile(predicted[:,2]-original,[0,.5,1]).tolist(),
        left_cop_zero_translation_only_error_improves=int((np.asarray(translation_errors)[:,2]<original).sum()),
        limits=['Parent reconstructed by subtracting recorded CoP delta from float32 final pose, not exact separately recorded parent pose',
            'Frozen observed points/common normal only: real pressure CoP after a hypothetical motion is unknown',
            'Each clock is evaluated on the original recorded blocked state, not rolled forward into a new trajectory',
            'No force response, contact retention, admission or successful carry predicted',
            'No GT object state/mesh/mass used, no new controller, physics or model execution'],
        records=rows,new_physics=0,new_model_forwards=0,new_optimizer_updates=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('hand_floor_feasibility.py'),file,root/'PROTOCOL.json',root/'RESULT.json')})
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'GEOMETRY.npz',frames=blocked,minimum_z_m=heights,
        frozen_point_tangent_error_m=predicted,frozen_point_angle_deg=np.asarray(angles),
        translation_only_tangent_error_m=np.asarray(translation_errors),original_error_m=original,
        reconstruction_four_ulp_m=bounds)
    (output/'RESULT.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','source_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.root,args.output)
