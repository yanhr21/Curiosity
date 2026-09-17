"""Varied exact-physics grip motions using the qualified pad-force servo.

Only nominal workspace/motion parameters change. Object truth is never passed to
this controller. Existing GripScene/ProbeScene physics and full meshes are reused.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import newton
from .grip_scene import GripController,GripScene


class GripMotionController(GripController):
    def __init__(self,target_load_n=12.,approach_angle_deg=0.,yaw_delta_deg=0.,lift_height_m=.18,lateral_xy=(0.,.04)):
        super().__init__(target_load_n)
        self.approach_angle_deg=approach_angle_deg;self.yaw_delta_deg=yaw_delta_deg
        self.lift_height_m=lift_height_m;self.lateral_xy=np.asarray(lateral_xy)
        self.previous_world=None

    def command(self,time,loads,dt):
        # Reuse the measured-load servo, including saturation and contact latch.
        base,_,record=super().command(time,loads,dt)
        u=np.clip((time-16.)/4.,0.,1.);f=.5*(1-np.cos(np.pi*u))
        turn=Rotation.from_euler('z',self.approach_angle_deg+self.yaw_delta_deg*f,degrees=True)
        # Fixed 0.32 m half-workspace accommodates all declared scales, without
        # querying the randomized object. Geometry only initializes the scene.
        base[:,:3]+=np.array([[-.07,0,0],[.07,0,0]])
        base[:,:3]-=np.array([0.,.04*f,.18*f])
        base[:,:3]+=np.r_[self.lateral_xy*f,self.lift_height_m*f]
        poses=np.c_[turn.apply(base[:,:3]),(turn*Rotation.from_quat(base[:,3:])).as_quat()]
        if self.previous_world is None:velocity=np.zeros((2,6))
        else:
            velocity=np.c_[(poses[:,:3]-self.previous_world[:,:3])/dt,
                (Rotation.from_quat(poses[:,3:])*Rotation.from_quat(self.previous_world[:,3:]).inv()).as_rotvec()/dt]
        self.previous_world=poses.copy()
        record.update(approach_angle_deg=self.approach_angle_deg,yaw_delta_deg=self.yaw_delta_deg,
                      lift_command_m=self.lift_height_m*f)
        return poses,velocity,record


class GripMotionScene(GripScene):
    def __init__(self,target_load_n=12.,approach_angle_deg=0.,yaw_delta_deg=0.,lift_height_m=.18,lateral_xy=(0.,.04),**kwargs):
        super().__init__(target_load_n=target_load_n,**kwargs)
        self.probe=GripMotionController(target_load_n,approach_angle_deg,yaw_delta_deg,lift_height_m,lateral_xy)
        poses,_,self.probe_record=self.probe.command(0.,np.zeros(2),.02)
        q=self.state_0.joint_q.numpy()
        for start,pose in zip(self.q_starts,poses):q[start:start+7]=pose
        self.state_0.joint_q.assign(q)
        newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0)
        self.probe.previous_world=None
