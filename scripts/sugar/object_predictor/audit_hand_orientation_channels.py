"""Identify the geometry actually carried by saved normal channels; CPU only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from .collect_newton import hand_sites
from sugar_newton.hand.patches import load_hand_mesh


def main(args):
    sites,axes=hand_sites(palm_signs=(-1.,1.));surface=[];mesh_details=[]
    for side,name in enumerate(('left','right')):
        mesh=load_hand_mesh(name)
        distances,indices=cKDTree(mesh.vertices).query(sites[side*27:(side+1)*27])
        surface.append(mesh.vertex_normals[indices])
        mesh_details.append(dict(side=name,vertices=len(mesh.vertices),faces=len(mesh.faces),
                                 maximum_site_to_original_vertex_m=float(distances.max()),
                                 winding_consistent=bool(mesh.is_winding_consistent)))
    surface=np.concatenate(surface)
    angles=np.rad2deg(np.arccos(np.clip(np.sum(surface*axes,axis=1),-1,1)))
    records=[];checks={'all_sites_original_hand_vertices':all(m['maximum_site_to_original_vertex_m']<1e-8 for m in mesh_details)}
    for root in args.data:
        for path in sorted(Path(root).glob('episode_*.json')):
            meta=json.loads(path.read_text())
            with np.load(path.with_suffix('.npz')) as source:
                normals=source['hand_normals_w'];poses=source['hand_pose_w'];loads=source['normal_load_n']
            expected=np.empty_like(normals)
            for side in range(2):
                rotation=Rotation.from_quat(poses[:,side,3:]).as_matrix()
                expected[:,side*27:(side+1)*27]=np.einsum('tij,nj->tni',rotation,axes[side*27:(side+1)*27])
            delta=float(np.abs(normals-expected).max());weights=loads.sum(0)
            name=f'{Path(root).name}/{meta["episode"]}'
            checks[name+'_saved_constant_palmar_axes']=delta<=1e-6
            records.append(dict(source=name,frames=len(normals),maximum_saved_axis_difference=delta,
                                load_weighted_site_normal_axis_angle_deg=float(np.sum(weights*angles)/weights.sum()) if weights.sum()>0 else None,
                                contacted_sites=np.flatnonzero(weights>0).tolist()))
    report=dict(checks=checks,passed=all(checks.values()),meshes=mesh_details,
                site_vertex_normal_to_axis_degrees=angles.tolist(),episodes=records,
                new_optimizer_updates=0,gpu_calls=0,
                interpretation='Saved Newton normal channels are fixed local palmar axes rotated by hand pose, not local mesh surface normals and not true object contact normals. Vertex-normal angles describe selected anatomical sites, not instantaneous force directions. No model input or frozen physical collection changed.')
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',nargs='+',required=True);ap.add_argument('--output',required=True);main(ap.parse_args())
