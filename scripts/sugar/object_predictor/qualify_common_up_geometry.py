"""CPU full-mesh initial placement certificate; GT used only in qualification."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from sugar_newton.hand.patches import load_hand_mesh
from .geometry import load_sugar_outer_box
from .qualified_cop_response import QualifiedCoPResponseController
from .common_up_cop_alignment import CommonUpCoPController


def initial_poses(config,controller=CommonUpCoPController):
    c=controller(target_load_n=config['load'],response_gain=True,force_gain=.000025,
        controller_revision='sensor_feedback_v2',approach_angle_deg=config['approach_angle_deg'],
        yaw_delta_deg=config['yaw_delta_deg'],lift_height_m=config['lift_height_m'],lateral_xy=config['lateral_xy'])
    poses=c.command(0.,np.zeros(2),.02)[0]
    return c,poses


def canonical_object(config):
    # Reproduce original static mesh placement algebra only, no Newton/physics.
    original,faces=load_sugar_outer_box()
    vertices=original*np.asarray(config['scale'],np.float32)
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    com=mesh.center_mass
    _,basis=np.linalg.eigh(np.cov(original.T))
    if np.linalg.det(basis)<0:basis[:,0]*=-1
    basis=basis[:,[1,2,0]]
    rng=np.random.default_rng(config['seed'])
    rotation=Rotation.from_euler('z',rng.uniform(-.25,.25))*Rotation.from_matrix(basis.T)
    com_xy=rng.uniform(-.012,.012,2)
    rotated=rotation.apply(vertices)
    origin=np.r_[com_xy-rotation.apply(com)[:2],.003-rotated[:,2].min()]
    return rotated+origin


def qualify(config):
    old,oldposes=initial_poses(config,QualifiedCoPResponseController)
    new,poses=initial_poses(config)
    object_vertices=canonical_object(config)
    rows=[]
    for i,side in enumerate(('left','right')):
        mesh=load_hand_mesh(side);rotation=Rotation.from_quat(poses[i,3:])
        previous=Rotation.from_quat(oldposes[i,3:])
        normal=rotation.apply(new.support_normals[i]);oldnormal=previous.apply(old.support_normals[i])
        center=rotation.apply(new.frames[i][1])+poses[i,:3]
        oldcenter=previous.apply(old.frames[i][1])+oldposes[i,:3]
        cad_x=rotation.apply([1.,0.,0.]);tangent=cad_x-normal*(cad_x@normal);tangent/=np.linalg.norm(tangent)
        vertices=rotation.apply(np.asarray(mesh.vertices,np.float32))+poses[i,:3]
        # Strict disjoint separating slabs certify every triangle, not vertex
        # sampling against a shape: every triangle lies within its vertex slab.
        handmax=float((vertices@normal).max());objectmin=float((object_vertices@normal).min())
        gap=objectmin-handmax;floor=float(vertices[:,2].min())
        checks=dict(normal_preserved=bool(np.allclose(normal,oldnormal,rtol=0,atol=1e-12)),
            support_center_preserved=bool(np.allclose(center,oldcenter,rtol=0,atol=1e-12)),
            support_height_original=bool(abs(center[2]-.14)<1e-12),
            projected_cad_x_points_world_up=bool(np.allclose(tangent,[0,0,1],rtol=0,atol=1e-12)),
            full_hand_above_floor=floor>0.,full_mesh_object_separation=gap>0.,
            separation_exceeds_sdf_margins_and_float32_roundoff=gap>.006001)
        rows.append(dict(side=side,checks=checks,passed=all(checks.values()),pose=poses[i].tolist(),
            original_pose=oldposes[i].tolist(),support_center_w=center.tolist(),normal_w=normal.tolist(),
            cad_x_w=cad_x.tolist(),projected_cad_x_w=tangent.tolist(),full_hand_vertex_count=len(vertices),
            full_hand_triangle_count=len(mesh.faces),min_hand_z_m=floor,object_separating_gap_m=gap))
    return dict(episode=config['episode'],passed=all(x['passed'] for x in rows),hands=rows,
        full_object_vertex_count=len(object_vertices),object_min_z_m=float(object_vertices[:,2].min()),
        object_use='Evaluation-only full mesh canonical placement; controller initializer receives no object pose/scale/mass/mesh.',
        qualification_scope='Initial static full-triangle separation only; no contact/grasp/carry success claim.',
        physics_controls=0,model_forwards=0)


def qualify_actual_scene(scene,expected_poses):
    """Read actual initial Newton buffers before stepping; labels never fed back."""
    states=scene.state_0.body_q.numpy()
    poses=states[scene.patch_frame].astype(float)
    if not np.allclose(poses,expected_poses,rtol=0,atol=2e-7):
        raise ValueError('Actual initial hands differ from qualified frame')
    obj=states[scene.box_body].astype(float)
    object_vertices=Rotation.from_quat(obj[3:]).apply(scene.box_verts)+obj[:3]
    rows=[]
    for i,side in enumerate(('left','right')):
        rot=Rotation.from_quat(poses[i,3:]);normal=rot.apply(scene.probe.support_normals[i])
        hand=load_hand_mesh(side)
        vertices=rot.apply(np.asarray(hand.vertices,np.float32))+poses[i,:3]
        floor=float(vertices[:,2].min())
        gap=float((object_vertices@normal).min()-(vertices@normal).max())
        rows.append(dict(side=side,min_full_hand_z_m=floor,full_triangle_separating_gap_m=gap))
        if floor<=.002001 or gap<=.006001:raise ValueError('Unsafe initial full mesh clearance')
    if scene.frame!=0 or scene.time!=0.:raise ValueError('Initialization must precede any physical step')
    return dict(passed=True,actual_hand_poses_w=poses.tolist(),actual_object_pose_w=obj.tolist(),
        hands=rows,frame=scene.frame,time_s=scene.time,object_GT_only_static_safety_check=True,
        static_certificate_not_grasp_success=True)
