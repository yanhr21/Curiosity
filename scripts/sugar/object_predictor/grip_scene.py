"""Exact-mesh Newton side-grip/lift acquisition adapter, not a learned policy.

Controller inputs are clock and the previously measured 27-pad load per hand.
Object pose, mass, mesh, contact normals and solver forces are validation only.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import newton
from .probe_scene import ProbeScene, ProbeController


class GripController(ProbeController):
    def __init__(self, target_load_n=12.):
        super().__init__(gentle=True)
        self.target_load_n=float(target_load_n)
        self.last_positions=None
        self.last_rotations=None
        self.distance=np.zeros(2)
        self.touched=np.zeros(2,bool)

    def command(self,time,loads,dt):
        centers=np.array([[-.25,0.,.14],[.25,0.,.14]])
        normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
        rotations=[Rotation.align_vectors(n[None],np.array([[0.,0.,1.]]))[0]*frame[0]
                   for n,frame in zip(normals,self.frames)]
        self.touched|=loads>=.2
        speed=np.where(self.touched,np.clip((self.target_load_n-loads)*.00015,-.008,.004),.012)
        if time<.5:speed[:]=0.
        self.distance=np.clip(self.distance+speed*dt,0.,.20)
        points=centers+normals*self.distance[:,None]
        u=np.clip((time-16.)/4.,0.,1.)
        lift=.18*.5*(1-np.cos(np.pi*u))
        lateral=.04*.5*(1-np.cos(np.pi*u))
        points[:,2]+=lift
        points[:,1]+=lateral
        poses=np.stack([np.r_[point-r.apply(frame[1]),r.as_quat()]
                        for point,r,frame in zip(points,rotations,self.frames)])
        velocity=np.zeros((2,6)) if self.last_positions is None else np.c_[(poses[:,:3]-self.last_positions)/dt,np.zeros((2,3))]
        self.last_positions=poses[:,:3].copy();self.last_rotations=rotations
        return poses,velocity,dict(phase=0 if time<16 else (1 if time<20 else 2),phase_time=time,
                                  touched=self.touched.copy(),distance=self.distance.copy(),
                                  target_load_n=self.target_load_n,lift_command_m=lift)


class GripScene(ProbeScene):
    def __init__(self,target_load_n=12.,**kwargs):
        super().__init__(gentle=True,canonical_init=True,**kwargs)
        self.probe=GripController(target_load_n)
        poses,_,self.probe_record=self.probe.command(0.,np.zeros(2),.02)
        q=self.state_0.joint_q.numpy()
        for start,pose in zip(self.q_starts,poses):q[start:start+7]=pose
        self.state_0.joint_q.assign(q)
        newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0)
        self.probe.last_positions=None
