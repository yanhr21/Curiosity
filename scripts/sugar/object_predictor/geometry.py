"""Faithful SUGAR collision-geometry adapter; no bounding-box replacement."""
import numpy as np


def palmar_support_frame(mesh,sign):
    """Choose a supporting plane from CAD for fixture placement only.

    The collider remains the full original hand. A convex-hull support triangle
    supplies three real extreme vertices; rotating its outward normal upward
    keeps all original mesh vertices on/below the same support plane.
    """
    from scipy.spatial.transform import Rotation
    hull=mesh.convex_hull
    candidates=np.flatnonzero(hull.face_normals[:,1]*sign>.4)
    index=candidates[np.argmax(hull.area_faces[candidates])]
    normal=hull.face_normals[index]
    tangent=np.array([1.,0.,0.])-normal[0]*normal
    tangent/=np.linalg.norm(tangent)
    R=Rotation.from_matrix(np.stack([tangent,np.cross(normal,tangent),normal]))
    center=np.array(hull.triangles_center[index],copy=True)
    heights=R.apply(mesh.vertices)[:,2]
    if abs(heights.max()-R.apply(center)[2])>1e-7:
        raise ValueError('CAD support plane is not an extreme plane')
    return R,center


def load_sugar_outer_box(which='small'):
    """Apply SMALLBOX_SDF_CFG.solid_outer_shell_only exactly as SUGAR does.

    Source: SUGAR/source/sugar_rl/sugar_rl/assets/objects/tactile_objects.py,
    spawn_from_usd_with_sdf(). The released scan contains a positive outer shell
    and a negative inner shell. Retain the complete unique positive component,
    with no hull, remeshing, decimation, or fabricated replacement geometry.
    """
    import trimesh
    from sugar_newton.validation.g1_carrybox import load_box_mesh
    vertices,faces=load_box_mesh(which)
    source=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    components=source.split(only_watertight=False)
    positive=[c for c in components if c.is_watertight and c.is_winding_consistent and float(c.volume)>0]
    if len(positive)!=1:
        raise ValueError(f'SUGAR solid-outer conversion requires one positive component, got {len(positive)}')
    outer=positive[0]
    if not np.allclose(outer.bounds,source.bounds,atol=1e-8):
        raise ValueError('Selected outer surface changed object bounds')
    return np.asarray(outer.vertices,np.float32),np.asarray(outer.faces,np.int32)
