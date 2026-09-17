"""Force-readiness gate for varied grip acquisition; no object-state feedback."""
import numpy as np
import newton
from .grip_motion_scene import GripMotionController,GripMotionScene


class GripReadyController(GripMotionController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ready_streak=0.
        self.lift_started_at=None

    def command(self,time,loads,dt):
        if np.all(np.abs(loads/self.target_load_n-1)<=.25):self.ready_streak+=dt
        else:self.ready_streak=0.
        if self.lift_started_at is None and 16.<=time<=28. and self.ready_streak>=1.:
            self.lift_started_at=time
        clock=min(time,16.-1e-6) if self.lift_started_at is None else 16.+time-self.lift_started_at
        poses,velocity,record=super().command(clock,loads,dt)
        record.update(actual_time_s=time,ready_streak_s=self.ready_streak,
                      lift_start_time_s=-1. if self.lift_started_at is None else self.lift_started_at)
        return poses,velocity,record


class GripReadyScene(GripMotionScene):
    def __init__(self,target_load_n=12.,approach_angle_deg=0.,yaw_delta_deg=0.,lift_height_m=.18,lateral_xy=(0.,.04),**kwargs):
        super().__init__(target_load_n=target_load_n,approach_angle_deg=approach_angle_deg,yaw_delta_deg=yaw_delta_deg,lift_height_m=lift_height_m,lateral_xy=lateral_xy,**kwargs)
        self.probe=GripReadyController(target_load_n,approach_angle_deg,yaw_delta_deg,lift_height_m,lateral_xy)
        poses,_,self.probe_record=self.probe.command(0.,np.zeros(2),.02)
        q=self.state_0.joint_q.numpy()
        for start,pose in zip(self.q_starts,poses):q[start:start+7]=pose
        self.state_0.joint_q.assign(q)
        newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0)
        self.probe.previous_world=None
