"""Transfer unused CoP quota only after observed floor cancellation.

No original controller is modified. Total actual CoP travel remains <=200mm.
The original allocated transaction is preferred; borrowing requires its exact
narrow floor/quota witness and additional safe motion. No object GT is read.
"""
from copy import deepcopy
import numpy as np
from .allocated_floor_cop import AllocatedFloorCoPController, parent_and_cop_proposal, add_cop, MASKS, TELEMETRY as BASE_TELEMETRY
from .qualified_cop_response import command_evidence
from .hand_floor_feasibility import swept_min_height
from .floor_safe_cop import CLEARANCE_M

INTERVENTION='shared_budget_cop_v1'
TOTAL_TRAVEL_M=.2
DESCRIPTION='Prefer original allocated CoP. Only floor-cancelled donor plus quota-limited recipient may transfer unused travel; actual total<=200mm, original speed/deadline/gates unchanged.'
TELEMETRY=BASE_TELEMETRY+('shared_total_travel_m','shared_borrowed_travel_m','shared_borrow_attempted',
    'shared_borrow_applied','shared_borrow_applied_count','shared_total_limited','shared_available_before_m',
    'shared_desired_step_m','shared_original_proposed_step_m','shared_quota_limited','shared_floor_donor',
    'shared_borrow_eligible','shared_baseline_selected_mask','shared_baseline_blocked',
    'shared_baseline_evaluated','shared_baseline_safe','shared_baseline_swept_min_z_m',
    'shared_baseline_candidate_steps_m','shared_candidate_steps_m',
    'shared_borrow_candidate_evaluated','shared_borrow_candidate_safe',
    'shared_borrow_candidate_swept_min_z_m','shared_borrow_candidate_steps_m')


def budget_steps(steps,mask,travel):
    """Enforce real shared travel even after a hand has borrowed >100mm."""
    out=steps*mask[:,None];cost=float(np.linalg.norm(out,axis=1).sum())
    if float(travel.sum())>TOTAL_TRAVEL_M:raise ValueError('Prior actual shared travel exceeds200mm')
    remaining=max(0.,TOTAL_TRAVEL_M-float(travel.sum()))
    if cost>remaining:
        if remaining==0.:return np.zeros_like(out)
        factor=remaining/cost
        # Round toward zero until the actual float64 accumulator is within the
        # strict bound; no physical tolerance or budget inflation is used.
        for _ in range(64):
            factor=np.nextafter(factor,0.)
            candidate=out*factor
            if float((travel+np.linalg.norm(candidate,axis=1)).sum())<=TOTAL_TRAVEL_M:return candidate
        raise FloatingPointError('Shared travel rounding did not converge')
    if float((travel+np.linalg.norm(out,axis=1)).sum())>TOTAL_TRAVEL_M:
        # Covers accumulator rounding when cost and remaining compare equal.
        return budget_steps(out*np.nextafter(1.,0.),np.ones(2,bool),travel)
    return out


def choose_group(meshes,old,parent,velocity,desired,travel,prelift,dt):
    poses=[];speeds=[];steps=np.zeros((4,2,3));heights=np.zeros((4,2))
    evaluated=np.zeros(4,bool);safe=np.zeros(4,bool)
    for j,mask in enumerate(MASKS):
        steps[j]=budget_steps(desired,mask,travel)
        q,qd=add_cop(parent,velocity,steps[j],np.ones(2,bool),dt)
        poses.append(q);speeds.append(qd)
        if j and (safe[0] or not prelift):continue
        heights[j]=[swept_min_height(v,p,t) for v,p,t in zip(meshes,old,q,strict=True)]
        evaluated[j]=True;safe[j]=bool((heights[j]>=CLEARANCE_M).all())
    choices=np.flatnonzero(safe)
    scores=np.linalg.norm(steps,axis=2).sum(1)
    selected=int(choices[np.argmax(scores[choices])]) if len(choices) else -1
    return dict(poses=poses,velocities=speeds,steps=steps,heights=heights,evaluated=evaluated,safe=safe,selected=selected)


def desired_steps(record,time,lift_start,dt):
    raw=np.zeros((2,3))
    if lift_start<0 and 1.<=time<=40. and record['cop_valid'].all() and not record['cop_admission_ready']:
        velocity=.25*record['cop_tangent_error_m'];speed=np.linalg.norm(velocity)
        if speed>.004:velocity*=.004/speed
        raw[0]=velocity*dt;raw[1]=-velocity*dt
    return raw


def narrow_borrow_eligibility(original,raw,base,travel,prelift):
    quota=np.linalg.norm(raw,axis=1)>np.linalg.norm(original,axis=1)
    donor=np.zeros(2,bool)
    if prelift and base['selected']>=0:
        mask=MASKS[base['selected']]
        donor=(np.linalg.norm(original,axis=1)>0)&~mask&(base['heights'][0]<CLEARANCE_M)&(travel<.1)
    eligible=quota&donor[::-1] if float(travel.sum())<TOTAL_TRAVEL_M else np.zeros(2,bool)
    return quota,donor,eligible


class SharedBudgetCoPController(AllocatedFloorCoPController):
    intervention_name=INTERVENTION

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shared_borrow_applied_count=0

    def command(self,time,loads,dt):
        candidate=deepcopy(self,{id(v):v for v in self.floor_vertices})
        parent,pv,record,original=parent_and_cop_proposal(candidate,time,loads,dt)
        raw=desired_steps(record,time,candidate.lift_start,dt)
        old=self.observed_poses
        initial=add_cop(parent,pv,original,MASKS[0],dt)[0] if old is None else old
        observed=np.array([swept_min_height(v,p,p) for v,p in zip(self.floor_vertices,initial,strict=True)])
        if (observed<CLEARANCE_M).any():raise ValueError('Observed full hand violates original floor reserve')
        before=self.cop_travel.copy();prelift=self.lift_start<0
        base=choose_group(self.floor_vertices,initial,parent,pv,original,before,prelift,dt)
        quota,donor,eligible=narrow_borrow_eligibility(original,raw,base,before,prelift)
        desired=original.copy();desired[eligible]=raw[eligible]
        borrowed=None;group=base;used_desired=original
        applied=np.zeros(2,bool)
        if eligible.any():
            borrowed=choose_group(self.floor_vertices,initial,parent,pv,desired,before,prelift,dt)
            if borrowed['selected']>=0:
                proposed=borrowed['steps'][borrowed['selected']]
                base_actual=base['steps'][base['selected']]
                additional=eligible&(np.linalg.norm(proposed,axis=1)>np.linalg.norm(base_actual,axis=1))
                if additional.any() and np.linalg.norm(proposed,axis=1).sum()>=np.linalg.norm(base_actual,axis=1).sum():
                    group=borrowed;used_desired=desired;applied=additional
        selected=group['selected'];committed=selected>=0
        keep=MASKS[selected].copy() if committed else np.zeros(2,bool)
        executed=group['steps'][selected].copy() if committed else np.zeros((2,3))
        changed=bool(committed and np.any(np.linalg.norm(used_desired,axis=1)[~keep]>0.))
        previous_phase=self.motion_elapsed
        diag=dict(floor_blocked=not committed,floor_requested_swept_min_z_m=group['heights'][0].copy(),
            floor_observed_min_z_m=observed,floor_requested_target_pose_w=group['poses'][0].copy(),
            floor_requested_cop_velocity_m_s=group['steps'][0]/dt,
            floor_requested_motion_elapsed_s=record.get('motion_elapsed_s',candidate.motion_elapsed),
            floor_requested_lift_start_s=record.get('lift_start_s',candidate.lift_start),
            floor_requested_distance_m=record['distance'].copy(),floor_requested_ready_seconds=record.get('ready_seconds',candidate.ready_seconds),
            floor_requested_alignment_active=record['actual_alignment_active'].copy(),floor_parent_transaction_committed=committed,
            cop_allocation_attempted=bool(not group['safe'][0] and prelift),cop_allocation_changed=changed,
            cop_allocation_selected_mask=keep,cop_allocation_parent_pose_w=parent.copy(),
            cop_allocation_proposed_step_m=used_desired.copy(),cop_allocation_evaluated=group['evaluated'],
            cop_allocation_safe=group['safe'],cop_allocation_swept_min_z_m=group['heights'],
            shared_available_before_m=max(0.,TOTAL_TRAVEL_M-float(before.sum())),
            shared_desired_step_m=raw,shared_original_proposed_step_m=original,
            shared_quota_limited=quota,shared_floor_donor=donor,shared_borrow_eligible=eligible,
            shared_borrow_attempted=bool(eligible.any()),shared_borrow_applied=applied,
            shared_baseline_selected_mask=MASKS[base['selected']] if base['selected']>=0 else np.zeros(2,bool),
            shared_baseline_blocked=base['selected']<0,shared_baseline_evaluated=base['evaluated'],
            shared_baseline_safe=base['safe'],shared_baseline_swept_min_z_m=base['heights'],
            shared_baseline_candidate_steps_m=base['steps'],shared_candidate_steps_m=group['steps'],
            shared_borrow_candidate_evaluated=borrowed['evaluated'] if borrowed else np.zeros(4,bool),
            shared_borrow_candidate_safe=borrowed['safe'] if borrowed else np.zeros(4,bool),
            shared_borrow_candidate_swept_min_z_m=borrowed['heights'] if borrowed else np.zeros((4,2)),
            shared_borrow_candidate_steps_m=borrowed['steps'] if borrowed else np.zeros((4,2,3)),
            shared_total_limited=bool(committed and np.any(np.linalg.norm(executed,axis=1)<np.linalg.norm(used_desired*keep[:,None],axis=1))))
        if committed:
            poses,velocity=group['poses'][selected],group['velocities'][selected]
            candidate.cop_travel+=np.linalg.norm(executed,axis=1)
            if float(candidate.cop_travel.sum())>TOTAL_TRAVEL_M:raise ValueError('Actual200mm budget exceeded')
            candidate.cop_allocation_changed_count+=int(changed)
            candidate.shared_borrow_applied_count+=int(applied.any())
            self.__dict__.update(candidate.__dict__);executed_min=group['heights'][selected]
        else:
            if old is None:raise ValueError('Cannot reject unknown initial command')
            self.response=candidate.response;self.contact_latched=candidate.contact_latched.copy()
            self.ready_seconds=0.;self.floor_blocked_count+=1
            poses,velocity=old.copy(),np.zeros((2,6));executed_min=observed
            record.update(distance=self.distance.copy(),touched=self.contact_latched.copy(),ready_seconds=0.,
                lift_start_s=self.lift_start,lift_complete_s=self.lift_complete,lift_command_m=self.lift_height_m*self.previous_lift,
                phase=0 if self.lift_start<0 else (1 if self.motion_elapsed<4. else 2),motion_elapsed_s=self.motion_elapsed,
                motion_ready=False,motion_paused=bool(self.lift_start>=0 and self.motion_elapsed<4.),motion_rate=0.,actualmotion_rate=0.,
                sustained_ready=False,alignment_active=np.zeros(2,bool),actual_alignment_active=np.zeros(2,bool))
        record.update(cop_servo_velocity_m_s=executed/dt,cop_travel_m=self.cop_travel.copy(),
            cop_budget_exhausted=bool((self.cop_travel>=.10-1e-12).any() and not record['cop_admission_ready']))
        rotation,travel,phase,pure=command_evidence(old,poses,record,previous_phase,dt)
        record.update(response_command_rotation_rad=rotation,response_command_cop_travel_m=travel,
            response_command_phase_delta_s=phase,response_command_target_quat_xyzw=poses[:,3:].copy(),
            response_command_pure=pure,response_window_pure=self.response.window_pure.copy(),
            response_secant_accepted=self.response.accepted.copy(),response_qualified_secants_total=self.response.accepted_total.copy(),
            response_rejected_mixed_secants_total=self.response.rejected_total.copy(),response_base_fallback=self.response.upper==0.)
        self.response.incoming_pure=pure.copy()
        record.update(diag,floor_blocked_count=self.floor_blocked_count,floor_executed_swept_min_z_m=executed_min,
            floor_executed_target_pose_w=poses.copy(),cop_allocation_changed_count=self.cop_allocation_changed_count,
            shared_total_travel_m=float(self.cop_travel.sum()),shared_borrowed_travel_m=np.maximum(0.,self.cop_travel-.1),
            shared_borrow_applied_count=self.shared_borrow_applied_count)
        return poses,velocity,record


def register_intervention():
    from .controller_interventions import INTERVENTIONS,INTERVENTION_DEPENDENCIES
    spec=('shared_budget_cop','SharedBudgetCoPController',DESCRIPTION,TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION]!=spec:raise ValueError('Conflicting local registration')
    INTERVENTIONS[INTERVENTION]=spec
    INTERVENTION_DEPENDENCIES[INTERVENTION]=('allocated_floor_cop','floor_safe_cop','hand_floor_feasibility',
        'qualified_cop_response','prelift_cop_alignment','freeze_postlift_alignment','fixed_path_surface_grip','alignment_rollback','response_gain')
