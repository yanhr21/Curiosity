"""Reduce native TacSL records to the shared causal object-predictor schema."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def convert(source,output,episode,legacy_total_force_xyz=False):
    root=Path(source)
    meta=json.loads((root/'summary.json').read_text())
    if meta['object_kind']!='carrybox':
        raise ValueError('This adapter currently has only the real CarryBox asset label path')
    with np.load(root/'whole_hand_trace.npz',allow_pickle=False) as src:
        # Read only actual sensor geometry/forces and label fields, not true slip.
        normal=src['normal_force'].reshape(-1,54,500)
        shear=src['signed_shear'].reshape(-1,54,500,2)
        positions=src['taxel_position_w'].reshape(-1,54,500,3)
        quats=src['taxel_quaternion_w'].reshape(-1,54,500,4)
        hand_names=src['robot_body_names'].tolist()
        ids=[hand_names.index(f'{side}_rubber_hand') for side in ('left','right')]
        hand=src['robot_body_state_w'][:,ids,:7].copy()
        hand[...,3:]=hand[...,(4,5,6,3)] # Isaac wxyz -> xyzw
        obj=src['object_state_w'][:,:7].copy()
        obj[:,3:]=obj[:,(4,5,6,3)]
        stamps=src['tactile_timestamp_s'].copy()
        joint=src['robot_joint_position'].copy()
        patch_sizes=src['tactile_patch_size_m'].reshape(54,2)
        legacy_normals=src['tactile_contact_normal_w'].reshape(-1,54,500,3) if legacy_total_force_xyz else None
    T=len(normal)
    R=Rotation.from_quat(quats.reshape(-1,4)).as_matrix().reshape(T,54,500,3,3)
    if legacy_total_force_xyz:
        # Historical records stored the TOTAL local XYZ contact force as
        # (signed_shear_X, signed_shear_Y, normal_force_Z). Reconstruct that
        # saved vector, then separate its original physical normal/friction
        # components. SDF normals are used only inside this sensor conversion;
        # they are never provided as geometric model observations.
        total_local=np.concatenate((shear,normal[...,None]),axis=-1)
        total_w=np.einsum('tpkij,tpkj->tpki',R,total_local)
        magnitude=np.einsum('tpki,tpki->tpk',total_w,legacy_normals)
        if magnitude.min() < -1e-5:
            raise ValueError('Historical force is inconsistent with saved contact-normal convention')
        normal=np.maximum(magnitude,0)
        friction_w=total_w-normal[...,None]*legacy_normals
        shear_w=friction_w.sum(2)
    else:
        if normal.min() < -1e-6:
            raise ValueError('Signed historical normal channel: explicit legacy vector conversion is required')
        shear3=np.concatenate((shear,np.zeros((*shear.shape[:-1],1),np.float32)),axis=-1)
        shear_w=np.einsum('tpkij,tpkj->tpki',R,shear3).sum(2)
    force=normal.sum(2)
    centers=positions.mean(2)
    contact=(positions*normal[...,None]).sum(2)/np.maximum(force[...,None],1e-12)
    contact=np.where((force>1e-6)[...,None],contact,centers)
    # Rotate each taxel's signed XY shear using its measured sensor pose.
    normals_w=R[:,:,:, :,2].mean(2)
    normals_w/=np.maximum(np.linalg.norm(normals_w,axis=-1,keepdims=True),1e-12)
    area=(normal>1e-6).sum(2)*(patch_sizes.prod(1)[None]/500)
    # USD geometry is a label only, never an observation channel. BBoxCache
    # includes authored child transforms and scale in the instantiated root.
    from pxr import Usd,UsdGeom
    repo=Path(__file__).resolve().parents[3]
    usd=repo/'SUGAR/descriptions/objects/small_box/obj_aligned.usd'
    stage=Usd.Stage.Open(str(usd))
    prim=stage.GetDefaultPrim()
    if not prim:
        raise ValueError('Object asset lacks a default prim')
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeLocalBound(prim).ComputeAlignedRange()
    lo=np.asarray(bbox.GetMin());hi=np.asarray(bbox.GetMax())
    scale=np.asarray(meta['object_scale'])
    dims=(hi-lo)*scale
    center=(hi+lo)*.5*scale
    if not np.isfinite(dims).all() or not np.all(dims>0):
        raise ValueError('Invalid asset bounds')
    mass=float(meta['object_mass_readback_kg'])
    arrays=dict(hand_pose_w=hand,hand_sites_w=centers,hand_normals_w=normals_w,
                contact_position_w=contact,normal_load_n=force,shear_force_w=shear_w,
                contact_area_m2=area,robot_joint_pos=joint,timestamp_s=stamps,
                object_pose_w=obj,object_mass_kg=np.full(T,mass,np.float32),
                object_dimensions_m=np.tile(dims,(T,1)).astype(np.float32),
                object_local_center_m=np.tile(center,(T,1)).astype(np.float32))
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite observation or label')
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'episode_{episode:04d}.npz'
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path,**arrays)
    summary=dict(episode=episode,motion=meta.get('motion_id',45),source=str(root),backend='IsaacLab/TacSL',
                 split='test',intended_use='saved native IsaacLab cross-backend evaluation; not a new rollout',
                 legacy_total_force_xyz=legacy_total_force_xyz,
                 frames=T,mass_kg=mass,scale=scale.tolist(),dimensions_m=dims.tolist(),
                 contact_frames=int((force.sum(1)>1e-3).sum()),max_assigned_force_n=float(force.sum(1).max()),
                 sensor_force_semantics='SDF penalty normal and friction-only local XY shear')
    path.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--episode',type=int,required=True)
    ap.add_argument('--legacy-total-force-xyz',action='store_true')
    args=ap.parse_args();convert(args.source,args.output,args.episode,args.legacy_total_force_xyz)
