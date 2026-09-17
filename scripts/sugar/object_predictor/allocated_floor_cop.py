"""Allocate only observed prelift CoP increments within known-hand floor safety.

Direct baseline is the completed all-or-none floor transaction. No parent force,
alignment, admission, deadline or travel limit is changed. No object GT is read.
"""
from copy import deepcopy
import numpy as np
from .floor_safe_cop import FloorSafeCoPController, TELEMETRY as FLOOR_TELEMETRY, CLEARANCE_M
from .freeze_postlift_alignment import FreezePostliftAlignmentController
from .prelift_cop_alignment import observed_alignment
from .qualified_cop_response import command_evidence
from .hand_floor_feasibility import swept_min_height

INTERVENTION='allocated_floor_cop_v1'
DESCRIPTION='Original qualified-CoP frames/force/path; keep safe full command, otherwise prelift choose safe per-hand keep/zero CoP combination with greatest retained original travel; otherwise full hold.'
MASKS=np.array([[True,True],[True,False],[False,True],[False,False]])
TELEMETRY=tuple(k for k in FLOOR_TELEMETRY if k!='floor_executed_fraction')+(
    'floor_parent_transaction_committed','floor_executed_target_pose_w',
    'cop_allocation_attempted','cop_allocation_changed','cop_allocation_changed_count',
    'cop_allocation_selected_mask','cop_allocation_parent_pose_w','cop_allocation_proposed_step_m',
    'cop_allocation_evaluated','cop_allocation_safe','cop_allocation_swept_min_z_m')


def parent_and_cop_proposal(candidate,time,loads,dt):
    """Original prelift formula, separating exact pre-addition pose from CoP.

The original measured-CoP admission guard runs before the original parent
force/alignment command. This is the same order as PreliftCoPAlignmentController.
No additional estimator update is performed during combination enumeration.
"""
    if dt<=0:raise ValueError('Positive control dt required')
    if candidate.observed_poses is None:
        valid,centers,tangent,angle=np.zeros(2,bool),np.zeros((2,3)),np.zeros(3),180.
    else:
        valid,centers,tangent,angle=observed_alignment(candidate.observed_poses,candidate.surface,candidate.support_normals)
    aligned=bool(valid.all() and angle<=5.)
    if candidate.lift_start<0 and not aligned:candidate.ready_seconds=-dt
    poses,velocity,record=FreezePostliftAlignmentController.command(candidate,time,loads,dt)
    steps=np.zeros((2,3))
    if (candidate.observed_poses is not None and candidate.lift_start<0
            and 1.<=time<=40. and valid.all() and not aligned):
        proposed=.25*tangent
        speed=np.linalg.norm(proposed)
        if speed>.004:proposed*=.004/speed
        for side,sign in ((0,1.),(1,-1.)):
            step=sign*proposed*dt;length=np.linalg.norm(step)
            remaining=max(0.,.10-candidate.cop_travel[side])
            if length>remaining:step*=remaining/length
            steps[side]=step
    record.update(cop_valid=valid,cop_world_m=centers,cop_tangent_error_m=tangent,
        cop_line_angle_deg=angle,cop_admission_ready=aligned)
    return poses,velocity,record,steps


def add_cop(parent,velocity,steps,keep,dt):
    """Preserve original per-side in-place rounding and velocity arithmetic."""
    poses=parent.copy();speed=velocity.copy()
    for side in (0,1):
        if keep[side]:
            poses[side,:3]+=steps[side]
            speed[side,:3]+=steps[side]/dt
    return poses,speed


class AllocatedFloorCoPController(FloorSafeCoPController):
    intervention_name=INTERVENTION

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cop_allocation_changed_count=0

    def command(self,time,loads,dt):
        candidate=deepcopy(self,{id(v):v for v in self.floor_vertices})
        parent,parent_velocity,record,steps=parent_and_cop_proposal(candidate,time,loads,dt)
        full,full_velocity=add_cop(parent,parent_velocity,steps,MASKS[0],dt)
        old=self.observed_poses;initial=full if old is None else old
        observed=np.array([swept_min_height(v,p,p) for v,p in zip(self.floor_vertices,initial,strict=True)])
        if (observed<CLEARANCE_M).any():raise ValueError('Observed full hand violates floor reserve; no teleport recovery')
        heights=np.zeros((4,2));evaluated=np.zeros(4,bool);safe=np.zeros(4,bool)
        def check(index,poses):
            heights[index]=[swept_min_height(v,p,q) for v,p,q in zip(self.floor_vertices,initial,poses,strict=True)]
            evaluated[index]=True;safe[index]=bool((heights[index]>=CLEARANCE_M).all())
        check(0,full)
        selected=0 if safe[0] else -1
        attempted=bool(not safe[0] and self.lift_start<0)
        alternatives=[(full,full_velocity)]
        if attempted:
            for index in (1,2,3):
                proposed=add_cop(parent,parent_velocity,steps,MASKS[index],dt)
                alternatives.append(proposed);check(index,proposed[0])
            scores=np.linalg.norm(steps,axis=1)@MASKS.T
            choices=np.flatnonzero(safe)
            # Fixed mask order resolves exact score ties; no GT/error selection.
            if len(choices):selected=int(choices[np.argmax(scores[choices])])
        committed=selected>=0
        keep=MASKS[selected].copy() if committed else np.zeros(2,bool)
        executed_steps=steps*keep[:,None]
        changed=bool(committed and np.any(np.linalg.norm(steps,axis=1)[~keep]>0.))
        diagnostics=dict(floor_blocked=not committed,floor_requested_swept_min_z_m=heights[0].copy(),
            floor_observed_min_z_m=observed,floor_requested_target_pose_w=full.copy(),
            floor_requested_cop_velocity_m_s=steps/dt,
            floor_requested_motion_elapsed_s=record.get('motion_elapsed_s',candidate.motion_elapsed),
            floor_requested_lift_start_s=record.get('lift_start_s',candidate.lift_start),
            floor_requested_distance_m=record['distance'].copy(),
            floor_requested_ready_seconds=record.get('ready_seconds',candidate.ready_seconds),
            floor_requested_alignment_active=record['actual_alignment_active'].copy(),
            floor_parent_transaction_committed=committed,
            cop_allocation_attempted=attempted,cop_allocation_changed=changed,
            cop_allocation_selected_mask=keep,cop_allocation_parent_pose_w=parent.copy(),
            cop_allocation_proposed_step_m=steps.copy(),cop_allocation_evaluated=evaluated,
            cop_allocation_safe=safe,cop_allocation_swept_min_z_m=heights)
        previous_phase=self.motion_elapsed
        if committed:
            poses,velocity=alternatives[selected]
            for side in (0,1):candidate.cop_travel[side]+=np.linalg.norm(executed_steps[side])
            candidate.cop_allocation_changed_count+=int(changed)
            self.__dict__.update(candidate.__dict__)
            executed_min=heights[selected].copy()
        else:
            if old is None:raise ValueError('Cannot reject unknown initial command')
            self.response=candidate.response
            self.contact_latched=candidate.contact_latched.copy()
            self.ready_seconds=0.;self.floor_blocked_count+=1
            poses,velocity=old.copy(),np.zeros((2,6));executed_min=observed
            record.update(distance=self.distance.copy(),touched=self.contact_latched.copy(),
                ready_seconds=0.,lift_start_s=self.lift_start,lift_complete_s=self.lift_complete,
                lift_command_m=self.lift_height_m*self.previous_lift,
                phase=0 if self.lift_start<0 else (1 if self.motion_elapsed<4. else 2),
                motion_elapsed_s=self.motion_elapsed,motion_ready=False,
                motion_paused=bool(self.lift_start>=0 and self.motion_elapsed<4.),
                motion_rate=0.,actualmotion_rate=0.,sustained_ready=False,
                alignment_active=np.zeros(2,bool),actual_alignment_active=np.zeros(2,bool))
        record.update(cop_servo_velocity_m_s=executed_steps/dt,cop_travel_m=self.cop_travel.copy(),
            cop_budget_exhausted=bool((self.cop_travel>=.10-1e-12).any() and not record['cop_admission_ready']))
        rotation,travel,phase,pure=command_evidence(old,poses,record,previous_phase,dt)
        record.update(response_command_rotation_rad=rotation,response_command_cop_travel_m=travel,
            response_command_phase_delta_s=phase,response_command_target_quat_xyzw=poses[:,3:].copy(),
            response_command_pure=pure,response_window_pure=self.response.window_pure.copy(),
            response_secant_accepted=self.response.accepted.copy(),
            response_qualified_secants_total=self.response.accepted_total.copy(),
            response_rejected_mixed_secants_total=self.response.rejected_total.copy(),
            response_base_fallback=self.response.upper==0.)
        self.response.incoming_pure=pure.copy()
        record.update(diagnostics,floor_blocked_count=self.floor_blocked_count,
            floor_executed_swept_min_z_m=executed_min,floor_executed_target_pose_w=poses.copy(),
            cop_allocation_changed_count=self.cop_allocation_changed_count)
        return poses,velocity,record


def register_intervention():
    from .controller_interventions import INTERVENTIONS,INTERVENTION_DEPENDENCIES
    spec=('allocated_floor_cop','AllocatedFloorCoPController',DESCRIPTION,TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION]!=spec:raise ValueError('Conflicting local registration')
    INTERVENTIONS[INTERVENTION]=spec
    INTERVENTION_DEPENDENCIES[INTERVENTION]=('floor_safe_cop','hand_floor_feasibility','qualified_cop_response',
        'prelift_cop_alignment','freeze_postlift_alignment','fixed_path_surface_grip','alignment_rollback','response_gain')
