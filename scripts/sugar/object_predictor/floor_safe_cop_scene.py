"""Original physical step with additional full-hand eight-substep receipts.

The solver/field order is unchanged. Actual after-FK and after-solver body poses
are saved separately; all full-mesh heights are evaluation only. No object GT.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import newton
from .surface_grip_scene import SurfaceGripScene
from .floor_safe_cop import FloorSafeCoPController


def full_hand_heights(vertices, poses):
    return np.array([float((v @ Rotation.from_quat(p[3:]).as_matrix()[2] + p[2]).min())
                     for v, p in zip(vertices, poses, strict=True)])


class FloorSafeCoPScene(SurfaceGripScene):
    def __init__(self, *args, **kwargs):
        if kwargs.get('controller_type') is not FloorSafeCoPController:
            raise ValueError('Explicit floor-safe controller required')
        super().__init__(*args, **kwargs)
        self.floor_initial_poses = self.state_0.body_q.numpy()[self.patch_frame].copy()
        self.floor_initial_min = full_hand_heights(self.probe.floor_vertices, self.floor_initial_poses)
        self.floor_substep_records = []

    def step(self, dt=.02, substeps=8):
        self.probe.observe(self.state_0.body_q.numpy()[self.patch_frame], self.field.to_numpy())
        field=self.field.to_numpy()
        loads=np.array([(field['area'][(field['pad']//27)==side]*field['pressure'][(field['pad']//27)==side]).sum() for side in range(2)]) if self.frame else np.zeros(2)
        oldposes=self.state_0.body_q.numpy()[self.patch_frame].copy()
        target,velocity,self.probe_record=self.probe.command((self.frame+1)*dt,loads,dt)
        for i in range(substeps):
            self.time+=dt/substeps;f=(i+1)/substeps
            q=self.state_0.joint_q.numpy();qd=self.state_0.joint_qd.numpy()
            for side,(qs,qds) in enumerate(zip(self.q_starts,self.qd_starts,strict=True)):
                R=Slerp([0.,1.],Rotation.from_quat([oldposes[side,3:],target[side,3:]]))([f])[0]
                q[qs:qs+7]=np.r_[(1-f)*oldposes[side,:3]+f*target[side,:3],R.as_quat()]
                qd[qds:qds+6]=velocity[side]
            self.state_0.joint_q.assign(q);self.state_0.joint_qd.assign(qd)
            newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0,body_flag_filter=newton.BodyFlags.KINEMATIC)
            fk_poses=self.state_0.body_q.numpy()[self.patch_frame].copy()
            self.state_0.clear_forces();self.pipeline.collide(self.state_0,self.contacts)
            self.solver.step(self.state_0,self.state_1,self.control,self.contacts,dt/substeps)
            self.state_0,self.state_1=self.state_1,self.state_0
            actual=self.state_0.body_q.numpy()[self.patch_frame].copy()
            self.floor_substep_records.append(dict(
                control_frame=self.frame,substep=i,timestamp_s=self.time,
                fk_hand_pose_w=fk_poses,actual_hand_pose_w=actual,
                fk_min_z_m=full_hand_heights(self.probe.floor_vertices,fk_poses),
                actual_min_z_m=full_hand_heights(self.probe.floor_vertices,actual)))
        self.solver.update_contacts(self.contacts,self.state_0)
        surface=self.pipeline.hydroelastic_sdf.get_contact_surface()
        self.tactile.update(self.state_0,self.contacts,contact_surface=surface)
        self.field.update(self.state_0,surface)
        self.frame+=1

    def save_floor_receipt(self, path):
        rows=self.floor_substep_records
        np.savez_compressed(path,initial_hand_pose_w=self.floor_initial_poses,
            initial_min_z_m=self.floor_initial_min,
            **{k:np.stack([r[k] for r in rows]) for k in rows[0]})
