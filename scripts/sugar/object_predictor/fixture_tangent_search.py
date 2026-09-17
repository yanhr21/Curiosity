"""Explicit bounded, public-frame contact acquisition; no hidden object input."""
from dataclasses import asdict, replace
import numpy as np

from .fixture_touch_scene import FixtureDrive, FixtureTouchScene

DT=.02
RADII=(.02,.04,.06)
SEARCH_SPEED=.04
HOLD_SECONDS=4.
RETREAT_SECONDS=2.
MAX_CONTROLS=1700
SEARCH_LENGTH=float(RADII[-1]+2*np.pi*sum(RADII))


def search_xy(length):
    """Public parent-frame +X connectors and CCW circles at 20/40/60 mm."""
    s=float(np.clip(length,0.,SEARCH_LENGTH));previous=0.
    for radius in RADII:
        radial=radius-previous
        if s<=radial:return np.array([previous+s,0.])
        s-=radial
        arc=2*np.pi*radius
        if s<=arc:return radius*np.array([np.cos(s/radius),np.sin(s/radius)])
        s-=arc;previous=radius
    return np.array([RADII[-1],0.])


def drive_config():
    return replace(FixtureDrive(),force_gain_m_ns=.003,angular_kp_nm_rad=32.)


class TangentSearchApproach:
    """First observed contact stops search immediately, including axial phase.

    Snapshot is exactly first contact observation +4s, without reselection or
    resetting on load quality. No contact after the finite path is a failure.
    """
    def __init__(self,config=None):
        self.config=drive_config() if config is None else config
        if asdict(self.config)!=asdict(drive_config()):
            raise ValueError('Keep the declared angular-stiffness drive parameters')
        self.depth_m=0.;self.xy=np.zeros(2);self.search_length_m=0.
        self.touched=False;self.overload_seen=False
        self.contact_time_s=-1.;self.snapshot_time_s=-1.
        self.exhausted_time_s=-1.;self.overload_time_s=-1.;self.phase=0

    def observe(self,time_s,load):
        if not np.isfinite([time_s,load]).all() or load<0:raise ValueError('Invalid observed force')
        if load>self.config.overload_n and not self.overload_seen:
            self.overload_seen=True;self.overload_time_s=float(time_s)
        if (not self.touched and not self.overload_seen and load>=self.config.contact_latch_n
                and (self.exhausted_time_s<0 or time_s<=self.exhausted_time_s+1e-10)):
            self.touched=True;self.contact_time_s=float(time_s)
            self.snapshot_time_s=float(time_s+HOLD_SECONDS)

    def finish_time(self):
        if self.overload_seen:return self.overload_time_s+RETREAT_SECONDS
        if self.touched:return self.snapshot_time_s+RETREAT_SECONDS
        if self.exhausted_time_s>=0:return self.exhausted_time_s+RETREAT_SECONDS
        return MAX_CONTROLS*DT

    def command(self,time_s,measured_load_n,dt):
        if not np.isfinite([time_s,measured_load_n,dt]).all() or measured_load_n<0 or abs(dt-DT)>1e-12:
            raise ValueError('Require declared 50Hz observed control')
        self.observe(time_s-dt,measured_load_n)
        cfg=self.config;tol=1e-10
        retreat=(self.overload_seen or (self.touched and time_s>self.snapshot_time_s+tol)
                 or (not self.touched and self.exhausted_time_s>=0))
        if retreat:
            self.phase=3;self.depth_m=max(0.,self.depth_m-cfg.retreat_speed_m_s*dt)
        elif time_s<=cfg.settle_s+tol:
            self.phase=0
        elif self.touched:
            self.phase=2
            speed=np.clip((cfg.target_load_n-measured_load_n)*cfg.force_gain_m_ns,
                          -cfg.contact_speed_limit_m_s,cfg.contact_speed_limit_m_s)
            self.depth_m=float(np.clip(self.depth_m+speed*dt,0.,cfg.max_travel_m))
        elif self.depth_m<cfg.start_radius_m-1e-12:
            self.phase=1
            self.depth_m=min(cfg.start_radius_m,self.depth_m+cfg.approach_speed_m_s*dt)
        else:
            self.phase=1
            self.search_length_m=min(SEARCH_LENGTH,self.search_length_m+SEARCH_SPEED*dt)
            self.xy=search_xy(self.search_length_m)
            if self.search_length_m>=SEARCH_LENGTH-1e-12:self.exhausted_time_s=float(time_s)
        return self.depth_m


class TangentSearchFixtureScene(FixtureTouchScene):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.controller=TangentSearchApproach(self.config)

    def target_translation(self,depth):
        return [*self.controller.xy,depth]

    def step(self,dt=DT,substeps=8):
        if substeps!=8:raise ValueError('Keep eight physical substeps')
        state=super().step(dt,substeps)
        self.controller.observe(state['time_s'],state['measured_palmar_load_n'])
        c=self.controller
        state.update(target_xy_m=c.xy.copy(),search_length_m=c.search_length_m,
            touched=c.touched,overload_seen=c.overload_seen,
            contact_time_s=c.contact_time_s,snapshot_time_s=c.snapshot_time_s,
            exhausted_time_s=c.exhausted_time_s,search_phase=c.phase,
            finished=state['time_s']>=c.finish_time()-1e-10)
        return state
