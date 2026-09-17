"""Known full-hand CAD surface queries; no object geometry or model involved.

This is an unadmitted input-adapter candidate. Existing datasets/endpoints still
use their stored palmar-axis channels. Nearest hand surface normals are not true
object normals or measured force directions.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from sugar_newton.hand.patches import load_hand_mesh,hand_mesh_path


@lru_cache(maxsize=1)
def geometry_signature():
    paths={'query_code':Path(__file__),**{side:Path(hand_mesh_path(side)) for side in ('left','right')}}
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}


def verify_geometry_signature(expected):
    if expected!=geometry_signature():
        raise RuntimeError('Saved CAD-normal input geometry/code differs; qualify compatibility explicitly')


@lru_cache(maxsize=2)
def hand_mesh(side):
    return load_hand_mesh(side)


def nearest_normals(mesh,points):
    closest,distance,face=mesh.nearest.on_surface(points)
    bary=trimesh.triangles.points_to_barycentric(mesh.triangles[face],closest)
    # Interpolation makes adjacent-triangle edge queries share the same normal;
    # a raw face-normal tie would turn coordinate roundoff into a discontinuity.
    normal=np.einsum('ni,nij->nj',bary,mesh.vertex_normals[mesh.faces[face]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    return normal,distance


def query_observation_normals(obs):
    count=len(obs['hand_pose_w']);site_normals=np.empty((count,54,3),np.float32)
    contact_normals=np.empty_like(site_normals);details=[]
    for side,name in enumerate(('left','right')):
        mesh=hand_mesh(name);part=slice(side*27,(side+1)*27)
        pose=obs['hand_pose_w'][:,side];rotation=Rotation.from_quat(pose[:,3:]).as_matrix()
        sites=np.einsum('tni,tij->tnj',obs['hand_sites_w'][:,part]-pose[:,None,:3],rotation)
        deviation=float(np.abs(sites-sites[0]).max())
        if deviation>1e-6: raise ValueError('Expected fixed anatomical sites on rigid hand')
        local,site_dist=nearest_normals(mesh,sites[0])
        world=np.einsum('tij,nj->tni',rotation,local)
        site_normals[:,part]=world;contact_normals[:,part]=world
        active=obs['normal_load_n'][:,part]>1e-3
        t,p=np.nonzero(active)
        contact_local=np.einsum('ni,nij->nj',obs['contact_position_w'][t,side*27+p]-pose[t,:3],rotation[t])
        if len(t):
            local_normal,dist=nearest_normals(mesh,contact_local)
            contact_normals[t,side*27+p]=np.einsum('nij,nj->ni',rotation[t],local_normal)
        else:dist=np.empty(0)
        details.append(dict(side=name,full_mesh_vertices=len(mesh.vertices),full_mesh_faces=len(mesh.faces),
                            site_rigidity_max_difference_m=deviation,site_nearest_surface_max_m=float(site_dist.max()),
                            contact_observations=len(t),contact_nearest_surface_distance_m=dict(
                                mean=float(dist.mean()),p95=float(np.quantile(dist,.95)),maximum=float(dist.max())) if len(t) else None))
    return site_normals,contact_normals,details
