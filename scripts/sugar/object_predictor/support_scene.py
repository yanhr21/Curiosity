"""Controlled sensing fixture with exact SUGAR hand and object collision meshes.

This is a tactile pretraining acquisition scene, not a whole-body controller.
Two prescribed real hand meshes support a free dynamic object. No pose/force is
fabricated for the object. Intended for separating sensing from CarryBox control.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import newton
from newton.geometry import HydroelasticSDF

from sugar_newton.hand.patches import load_hand_mesh,patch_footprints
from sugar_newton.tactile.reducer import PatchTactile
from sugar_newton.tactile.field import ContactField
from .geometry import load_sugar_outer_box,palmar_support_frame


class SupportScene:
    def __init__(self,scale=(1.,1.,1.),mass=.5,seed=0,box='small',object_sdf_resolution=128):
        if object_sdf_resolution not in (128,256):
            raise ValueError('Object SDF resolution must be the original128 or declared diagnostic256')
        rng=np.random.default_rng(seed)
        builder=newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        cfg=replace(builder.default_shape_cfg,ke=1e4,kd=320.,mu=.8,
                    kh=1e10,is_hydroelastic=True,density=0.,restitution=0.,
                    mu_torsional=0.,mu_rolling=0.)
        self.patch_shapes=[]
        self.hand_frames=[]
        self.q_starts=[]
        self.qd_starts=[]
        self.hand_poses=[]
        support_heights=[]
        support_centers=[]
        # Match SUGAR anatomical TacSL _surface_frame(), whose palmar approach
        # rays use left -Y / right +Y. Legacy Newton footprint signs differ.
        self.palm_signs=(-1.,1.)
        for i,(side,sign) in enumerate(zip(('left','right'),self.palm_signs,strict=True)):
            hand=load_hand_mesh(side)
            vertices=np.asarray(hand.vertices,np.float32)
            triangles=np.asarray(hand.faces,np.int32)
            R,contact_center=palmar_support_frame(hand,sign)
            desired_support=np.array([(-.07 if i==0 else .07),0.,.4])
            p=desired_support-R.apply(contact_center)
            pose=np.r_[p,R.as_quat()]
            body=builder.add_body(xform=wp.transform(pose[:3],pose[3:]),mass=1.,
                                  is_kinematic=True,label=f'{side}_rubber_hand')
            joint=next(j for j,c in enumerate(builder.joint_child) if c==body)
            self.q_starts.append(builder.joint_q_start[joint])
            self.qd_starts.append(builder.joint_qd_start[joint])
            mesh=newton.Mesh(vertices,triangles.flatten(),compute_inertia=False)
            mesh.build_sdf(max_resolution=128,narrow_band_range=(-.004,.004),margin=.002)
            shape=builder.add_shape_mesh(body=body,mesh=mesh,cfg=cfg,label=f'{side}_skin')
            self.patch_shapes.append(shape)
            self.hand_frames.append(body)
            self.hand_poses.append(pose)
            world_vertices=R.apply(vertices)+p
            top=world_vertices[:,2].max()
            support_heights.append(top)
            support_centers.append(desired_support[:2])
        vertices,triangles=load_sugar_outer_box(box)
        vertices=vertices*np.asarray(scale,np.float32)
        self.box_verts=vertices
        mesh=newton.Mesh(vertices,triangles.flatten(),compute_inertia=True)
        if mesh.mass<=0:
            raise ValueError('Object must have positive mesh volume')
        covariance=np.cov(vertices.T)
        _,basis=np.linalg.eigh(covariance)
        if np.linalg.det(basis)<0:
            basis[:,0]*=-1
        # Smallest principal extent is vertical: rest the widest face on the
        # hands. The previous ascending basis put the longest extent upright.
        basis=basis[:,[1,2,0]]
        # Align an intrinsic box face to the support plane, then independently
        # perturb yaw and translation. The controller never receives this label.
        R=Rotation.from_euler('z',rng.uniform(-.25,.25))*Rotation.from_matrix(basis.T)
        turned=R.apply(vertices)
        desired_com_xy=np.mean(support_centers,axis=0)+rng.uniform(-.012,.012,2)
        p=np.r_[desired_com_xy-R.apply(np.asarray(mesh.com))[:2],
                max(support_heights)-turned[:,2].min()+.003]
        self.box_body=builder.add_body(xform=wp.transform(p,R.as_quat()),mass=mass,
                                      com=mesh.com,inertia=wp.mat33(np.asarray(mesh.inertia).reshape(3,3)*mass/float(mesh.mass)),
                                      label='supported_object')
        mesh.build_sdf(max_resolution=object_sdf_resolution,narrow_band_range=(-.006,.006),margin=.004)
        self.box_shape=builder.add_shape_mesh(body=self.box_body,mesh=mesh,cfg=cfg,label='object')
        # Hands do not collide with each other; both still collide with the object.
        builder.add_shape_collision_filter_pair(*self.patch_shapes)
        builder.add_ground_plane(height=0.)
        self.model=builder.finalize()
        self.model.request_contact_attributes('force')
        self.pipeline=newton.CollisionPipeline(self.model,contact_matching='latest',contact_report=True,
            sdf_hydroelastic_config=HydroelasticSDF.Config(output_contact_surface=True,buffer_fraction=1.,buffer_mult_iso=2))
        self.contacts=self.pipeline.contacts()
        self.solver=newton.solvers.SolverMuJoCo(self.model,solver='newton',integrator='implicitfast',
            use_mujoco_contacts=False,cone='elliptic',iterations=100,ls_iterations=50,
            njmax=2048,nconmax=min(1000,self.contacts.rigid_contact_max))
        self.state_0,self.state_1=self.model.state(),self.model.state()
        self.control=self.model.control()
        self.patch_frame=self.model.shape_body.numpy()[self.patch_shapes].tolist()
        self.tactile=PatchTactile(self.model,self.patch_shapes,[self.box_shape])
        self.field=ContactField(self.tactile,frame_body=self.patch_frame,pad_footprint=patch_footprints(),
                                pad_offset=[0,27],palm_sign=list(self.palm_signs))
        self.n_joint_dofs=0 # Kinematic hand poses are the complete proprioception here.
        self.time=0.
        newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0)

    def step(self,dt=.02,substeps=8):
        for _ in range(substeps):
            self.time+=dt/substeps
            elapsed=max(0.,self.time-1.)
            # Smooth two-hand lift and horizontal excitation after settling.
            lift=.035*(1-np.cos(min(elapsed,3.)*np.pi/3))
            vz=.035*np.pi/3*np.sin(elapsed*np.pi/3) if elapsed<3 else 0.
            sway=.012*np.sin(elapsed*2)**2
            vx=.024*np.sin(elapsed*2)*np.cos(elapsed*2)*2
            q=self.state_0.joint_q.numpy();qd=self.state_0.joint_qd.numpy()
            for qs,qds,pose in zip(self.q_starts,self.qd_starts,self.hand_poses,strict=True):
                q[qs:qs+7]=pose
                q[qs:qs+3]+=np.array([sway,0.,lift])
                qd[qds:qds+6]=[vx,0.,vz,0.,0.,0.]
            self.state_0.joint_q.assign(q);self.state_0.joint_qd.assign(qd)
            newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0,
                           body_flag_filter=newton.BodyFlags.KINEMATIC)
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0,self.contacts)
            self.solver.step(self.state_0,self.state_1,self.control,self.contacts,dt/substeps)
            self.state_0,self.state_1=self.state_1,self.state_0
        self.solver.update_contacts(self.contacts,self.state_0)
        surface=self.pipeline.hydroelastic_sdf.get_contact_surface()
        self.tactile.update(self.state_0,self.contacts,contact_surface=surface)
        self.field.update(self.state_0,surface)
