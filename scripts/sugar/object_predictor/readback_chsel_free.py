"""Reconstruct official FREE voxel inputs and costs from frozen candidates.

No optimizer, candidate selection, or model forward. Evaluator truth is used
only after observation-based costs and grid coordinates have been saved.
"""
import argparse
import json
import os
from pathlib import Path

import chsel
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sugar_newton.hand.patches import load_hand_mesh
from .diagnose_hand_exclusion import sdf_for, read
from .qualify_chsel_backend import homogeneous


def main(args):
    assert os.environ.get('SLURM_STEP_ID') and torch.cuda.is_available()
    torch.set_num_threads(4); np.random.seed(20260916)
    root=Path(args.root); out=root/'free_readback'; out.mkdir(exist_ok=False)
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    progress=json.loads((root/'OBSERVATION_PROGRESS.json').read_text())
    core_root=Path(protocol['free_points_root'])
    mesh=read(Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz'))
    v=mesh['vertices'].astype(np.float64); dims=np.ptp(v,axis=0);v-=(v.max(0)+v.min(0))/2
    hands=[load_hand_mesh(side) for side in ('left','right')]
    hand_sdfs=[sdf_for(np.asarray(m.vertices),np.asarray(m.faces)) for m in hands]
    pending=[]; rows=[]
    for row in progress:
        name=f'episode_{row["episode"]}_frame_{row["frame"]}.npz'
        a=read(root/name); original=a['original_prediction']; core=read(core_root/name)['hand_core_10mm']
        surface=torch.as_tensor(a['points_current_left_hand_m'],device='cuda',dtype=torch.float32)
        free=torch.as_tensor(core,device='cuda',dtype=torch.float32)
        sdf=sdf_for(v*(np.exp(original[9:12])/dims),mesh['faces'])
        solver=chsel.CHSEL(sdf, positions=torch.cat([surface,free]),
            semantics=torch.cat([torch.zeros(len(surface),device='cuda',dtype=torch.long),torch.ones(len(free),device='cuda',dtype=torch.long)]),
            cost=chsel.VolumetricDoubleDirectCost,free_voxels_resolution=.003,
            qd_measure=chsel.PositionMeasure(3,device='cuda'),do_qd=True,qd_iterations=100)
        grid, known=solver.volumetric_cost.free_voxels.get_known_pos_and_values()
        grid=grid[known.reshape(-1)==1]
        alpha=solver.volumetric_cost.surface_threshold
        h=torch.stack([torch.as_tensor(np.linalg.inv(homogeneous(original)),device='cuda',dtype=torch.float32),
            torch.as_tensor(a['candidates_hand_to_object'][int(a['selected_index'])],device='cuda')])
        cost=solver.evaluate_homogeneous(h).detach().cpu().numpy()
        saved_cost=np.array([row['original_cost'],row['selected_cost']])
        assert np.max(abs(cost-saved_cost))<1e-5, 'Official reconstructed costs differ'
        surface_cost=solver.volumetric_cost._cost_sdf(h[:,:3,:3],h[:,:3,3],None).detach().cpu().numpy()
        free_cost=solver.volumetric_cost._cost_freespace(h[:,:3,:3],h[:,:3,3],None).detach().cpu().numpy()
        q=chsel.apply_similarity_transform(grid[None].expand(2,-1,-1),h[:,:3,:3],h[:,:3,3])
        d,_=sdf(q);d=d.detach().cpu().numpy();points=grid.cpu().numpy()
        hand=a['hand_pose_w'];rh=Rotation.from_quat(hand[:,3:]).as_matrix()
        world=points@rh[0].T+hand[0,:3]
        hand_dist=[]
        for i,hand_sdf in enumerate(hand_sdfs):
            local=(world-hand[i,:3])@rh[i]
            value,_=hand_sdf(torch.as_tensor(local,dtype=torch.float32));hand_dist.append(value.numpy())
        union_depth=-np.minimum(*hand_dist)
        record=dict(episode=row['episode'],frame=row['frame'],original_points=len(core),actual_grid_points=len(grid),
            alpha_m=float(alpha),max_cost_readback_difference=float(np.max(abs(cost-saved_cost))),
            actual_min_hand_interior_depth_mm=float(union_depth.min()*1000),
            actual_min_contact_distance_mm=float(cKDTree(surface.cpu().numpy()).query(points)[0].min()*1000),
            surface_cost=surface_cost.tolist(),free_unweighted_cost=free_cost.tolist(),
            free_weight=float(solver.volumetric_cost.scale_known_freespace),
            inside_fraction=(d<0).mean(1).tolist(),beyond_tolerance_fraction=(d < -alpha).mean(1).tolist(),
            max_penetration_cm=np.maximum(-d,0).max(1).tolist(),
            state_order=['original_utonia','selected_surface_and_free_chsel'])
        record['max_penetration_cm']=[x*100 for x in record['max_penetration_cm']]
        np.savez_compressed(out/name,actual_free_grid_left_hand_m=points,object_sdf_m=d,
            alpha_m=alpha,hand_union_depth_m=union_depth)
        rows.append(record);pending.append((name,points,alpha));print('FREE_READBACK',json.dumps(record),flush=True)
    # Independent truth overlap checks, after official cost equivalence readback.
    labels=read(root.parent/'chsel_qualification_inputs/evaluation_only.npz')
    for i,(name,points,alpha) in enumerate(pending):
        assert labels['episode'][i]==rows[i]['episode'] and labels['frame'][i]==rows[i]['frame']
        target=labels['target'][i];sdf=sdf_for(v*(np.exp(target[9:12])/dims),mesh['faces'])
        inv=np.linalg.inv(homogeneous(target));q=points@inv[:3,:3].T+inv[:3,3]
        d,_=sdf(torch.as_tensor(q,dtype=torch.float32));d=d.numpy()
        rows[i]['truth_evaluator_only']=dict(inside_fraction=float((d<0).mean()),
            beyond_tolerance_fraction=float((d < -alpha).mean()),max_penetration_cm=float(np.maximum(-d,0).max()*100))
    (out/'RESULT.json').write_text(json.dumps(dict(complete=True,cases=rows,
        reconstructed_official_costs_match=True,registration_calls=0,model_forwards=0,physics_steps=0,
        scope='Actual official voxel coordinates and default tolerance readback; no candidate reselection'),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);main(ap.parse_args())
