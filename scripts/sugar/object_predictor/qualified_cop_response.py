"""Qualify scalar force secants by the actual preceding command window.

An independent candidate: the original CoP controller, force formula and limits
are unchanged. Mixed windows cannot update response and invalidate old context.
No qualified samples means original base-gain fallback, not online identification.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from .prelift_cop_alignment import PreliftCoPAlignmentController, TELEMETRY as COP_TELEMETRY
from .response_gain import ObservedForceGain

INTERVENTION='qualified_cop_response_v1'
DESCRIPTION='Original CoP controller with complete pure-normal secant windows and explicit qualified-upper context/expiry validity.'
# Roundoff bounds only, not physical angular/motion deadbands. Both command
# rotations and motion-phase arithmetic are float64; 64 eps is ~1.42e-14.
ROUND_OFF_TOL=64*np.finfo(np.float64).eps
TELEMETRY=COP_TELEMETRY+('response_command_rotation_rad','response_command_cop_travel_m',
    'response_command_phase_delta_s','response_command_target_quat_xyzw','response_command_pure','response_window_pure',
    'response_secant_accepted','response_qualified_secants_total',
    'response_rejected_mixed_secants_total','response_base_fallback')


class QualifiedObservedForceGain(ObservedForceGain):
    def __init__(self,base_gain=.000025):
        super().__init__(base_gain)
        self.incoming_pure=np.zeros(2,bool)
        self.intervals=[deque(maxlen=6),deque(maxlen=6)]
        self.window_pure=np.zeros(2,bool)
        self.accepted=np.zeros(2,bool)
        self.accepted_total=np.zeros(2,np.int64)
        self.rejected_total=np.zeros(2,np.int64)

    def update(self,side,time,load,distance,dt,allow_increase):
        if dt<=0 or not np.isfinite([time,load,distance,dt]).all():
            raise ValueError('Invalid causal force observation')
        samples=self.samples[side];slopes=self.slopes[side]
        if samples and time<=samples[-1][0]:raise ValueError('Non-increasing control clock')
        samples.append((time,load,distance))
        self.intervals[side].append(bool(self.incoming_pure[side]))
        if not self.incoming_pure[side]:slopes.clear()
        self.window_pure[side]=len(samples)==6 and all(list(self.intervals[side])[1:])
        self.accepted[side]=False
        if len(samples)==6:
            then,old_load,old_distance=samples[0]
            dx=distance-old_distance;df=load-old_load
            if load>=.2 and old_load>=.2 and dx>1e-6 and df>.005:
                if self.window_pure[side]:
                    slopes.append((time,df/dx));self.accepted[side]=True;self.accepted_total[side]+=1
                else:self.rejected_total[side]+=1
        # Explicit second repair: no stale upper after expiry or mixed context.
        # The time window, >=5 requirement, maximum formula and gain caps remain.
        while slopes and slopes[0][0]<time-2.:slopes.popleft()
        self.upper[side]=max(v for _,v in slopes) if len(slopes)>=5 else 0.
        upper=self.upper[side]
        candidate=min(.00015,.1/(dt*upper)) if upper>0 else self.base_gain
        if not allow_increase:candidate=min(candidate,self.base_gain)
        self.applied[side]=min(candidate,self.applied[side]*1.05)
        return self.applied[side],upper,len(slopes)


def command_evidence(old_poses,new_poses,record,previous_phase,dt):
    if old_poses is None:
        rotation=np.zeros(2);known=False
    else:
        old=Rotation.from_quat(old_poses[:,3:]);new=Rotation.from_quat(new_poses[:,3:])
        rotation=(new*old.inv()).magnitude();known=True
    travel=np.linalg.norm(record['cop_servo_velocity_m_s'],axis=1)*dt
    phase=float(record.get('motion_elapsed_s',previous_phase))-previous_phase
    # Either hand's mixed motion can change both sides' load: require a common
    # pure interval, rather than calling the stationary side uncontaminated.
    pure=np.full(2,bool((rotation<=ROUND_OFF_TOL).all() and (travel==0.).all()
        and abs(phase)<=ROUND_OFF_TOL and known))
    return rotation,travel,phase,pure


class QualifiedCoPResponseController(PreliftCoPAlignmentController):
    intervention_name=INTERVENTION

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.response=QualifiedObservedForceGain(self.force_gain)

    def command(self,time,loads,dt):
        old=None if self.observed_poses is None else self.observed_poses.copy()
        phase=self.motion_elapsed
        poses,velocity,record=super().command(time,loads,dt)
        rotation,travel,delta,pure=command_evidence(old,poses,record,phase,dt)
        record.update(response_command_rotation_rad=rotation,
            response_command_cop_travel_m=travel,response_command_phase_delta_s=delta,
            response_command_target_quat_xyzw=poses[:,3:].copy(),
            response_command_pure=pure,response_window_pure=self.response.window_pure.copy(),
            response_secant_accepted=self.response.accepted.copy(),
            response_qualified_secants_total=self.response.accepted_total.copy(),
            response_rejected_mixed_secants_total=self.response.rejected_total.copy(),
            response_base_fallback=self.response.upper==0.)
        # This command is executed after the current observation: consume its
        # evidence on the NEXT call, never retroactively qualify this update.
        self.response.incoming_pure=pure.copy()
        return poses,velocity,record


def register_intervention():
    from .controller_interventions import INTERVENTIONS,INTERVENTION_DEPENDENCIES
    spec=('qualified_cop_response','QualifiedCoPResponseController',DESCRIPTION,TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION]!=spec:raise ValueError('Conflicting local registration')
    INTERVENTIONS[INTERVENTION]=spec
    INTERVENTION_DEPENDENCIES[INTERVENTION]=('prelift_cop_alignment','freeze_postlift_alignment',
        'fixed_path_surface_grip','alignment_rollback','response_gain')
