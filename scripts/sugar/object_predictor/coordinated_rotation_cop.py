"""Coordinate actual prelift alignment with observed CoP correction.

The shared-budget parent runs exactly once. Only a harmful, swept-safe rotation
is suppressed about its original area pivot; force closure and CoP increments
remain unchanged. Predicted rigid transport is not object-ground-truth geometry.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .shared_budget_cop import SharedBudgetCoPController, TELEMETRY as PARENT_TELEMETRY
from .qualified_cop_response import command_evidence, ROUND_OFF_TOL
from .hand_floor_feasibility import swept_min_height
from .floor_safe_cop import CLEARANCE_M

INTERVENTION='coordinated_rotation_cop_v1'
DESCRIPTION='Shared-budget parent plus selective prelift contact-pivot rotation suppression when rigid observed-plane transport opposes CoP correction; original force/path/budget/gates unchanged.'
SUPPRESS_MASKS=np.array([[True,False],[False,True],[True,True]])
ADDED_TELEMETRY=('rotation_coord_trigger','rotation_coord_selected_suppressed',
    'rotation_coord_suppressed_count','rotation_coord_original_predicted_angle_deg',
    'rotation_coord_original_fixed_normal_angle_deg','rotation_coord_candidate_predicted_angle_deg',
    'rotation_coord_candidate_hand_fit_angle_deg','rotation_coord_candidate_floor_min_z_m',
    'rotation_coord_candidate_evaluated','rotation_coord_candidate_eligible',
    'rotation_coord_parent_target_pose_w','rotation_coord_parent_actual_alignment_active',
    'rotation_coord_parent_executed_swept_min_z_m','rotation_coord_pivot_hand_m')
TELEMETRY=PARENT_TELEMETRY+ADDED_TELEMETRY


def line_angle(delta,normal):
    separation=float(delta@normal)
    return float(np.degrees(np.arctan2(np.linalg.norm(delta-normal*separation),separation)))


def predict_transport(old,target,centers,normals):
    r0=Rotation.from_quat(old[:,3:]);r1=Rotation.from_quat(target[:,3:])
    local=r0.inv().apply(centers-old[:,:3])
    moved=r1.apply(local)+target[:,:3]
    transported_normals=(r1*r0.inv()).apply(normals)
    original=normals[0]-normals[1];normal=transported_normals[0]-transported_normals[1]
    if min(np.linalg.norm(original),np.linalg.norm(normal))<1e-8:
        return 180.,180.
    original/=np.linalg.norm(original);normal/=np.linalg.norm(normal)
    delta=moved[1]-moved[0]
    return line_angle(delta,normal),line_angle(delta,original)


def coordinate_rotation(old,target,record,field,support_normals,vertices,*,prelift,time):
    """Pure observed-input selection; unchanged target object when ineligible.

Candidates preserve each hand's old measured area pivot under removing its
alignment rotation. Prelift phase has no yaw/lift, explicitly required here.
Three masks have fixed ties: left, right, both. Minimize suppressed sides first,
then the predicted angle. No predicted angle is used to admit the object.
"""
    diag=dict(rotation_coord_trigger=False,rotation_coord_selected_suppressed=np.zeros(2,bool),
        rotation_coord_original_predicted_angle_deg=180.,rotation_coord_original_fixed_normal_angle_deg=180.,
        rotation_coord_candidate_predicted_angle_deg=np.full(3,180.),
        rotation_coord_candidate_hand_fit_angle_deg=np.full((3,2),180.),
        rotation_coord_candidate_floor_min_z_m=np.zeros((3,2)),
        rotation_coord_candidate_evaluated=np.zeros(3,bool),rotation_coord_candidate_eligible=np.zeros(3,bool),
        rotation_coord_parent_target_pose_w=target.copy(),
        rotation_coord_parent_actual_alignment_active=record['actual_alignment_active'].copy(),
        rotation_coord_parent_executed_swept_min_z_m=record['floor_executed_swept_min_z_m'].copy(),
        rotation_coord_pivot_hand_m=np.zeros((2,3)))
    if (old is None or not prelift or not 1.<=time<=40. or record['lift_start_s']>=0
        or record['floor_blocked'] or not record['cop_valid'].all()
        or record['cop_admission_ready'] or not record['actual_alignment_active'].any()):
        return target,diag
    if record['motion_elapsed_s']!=0. or record['phase']!=0:
        raise ValueError('Prelift coordination must not remove trajectory yaw/lift')
    r0=Rotation.from_quat(old[:,3:]);r1=Rotation.from_quat(target[:,3:])
    rotating=record['actual_alignment_active']&((r1*r0.inv()).magnitude()>ROUND_OFF_TOL)
    if not rotating.any():return target,diag
    original,fixed=predict_transport(old,target,record['cop_world_m'],record['fit_normal_w'])
    diag.update(rotation_coord_original_predicted_angle_deg=original,
                rotation_coord_original_fixed_normal_angle_deg=fixed)
    if not original>fixed:return target,diag
    diag['rotation_coord_trigger']=True
    pivots=diag['rotation_coord_pivot_hand_m']
    for h in (0,1):
        sel=(field['patch']==h)&(field['pad']>=0)&(field['pressure']>0)&(field['area']>0)
        points=field['pos'][sel].astype(float);area=field['area'][sel].astype(float)
        if len(points)<6 or not np.isfinite(points).all() or not np.isfinite(area).all():
            raise ValueError('Invalid observed area pivot on bilateral-valid command')
        pivots[h]=np.average(points,axis=0,weights=area)
    candidates=[]
    for k,mask in enumerate(SUPPRESS_MASKS):
        q=target.copy();candidates.append(q)
        if not rotating[mask].all():continue
        for h in np.flatnonzero(mask):
            q[h,:3]+=r1[h].apply(pivots[h])-r0[h].apply(pivots[h])
            q[h,3:]=old[h,3:]
        angle,_=predict_transport(old,q,record['cop_world_m'],record['fit_normal_w'])
        palm=Rotation.from_quat(q[:,3:]).apply(support_normals)
        fitangle=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',palm,record['fit_normal_w']),-1,1)))
        minimum=np.array([swept_min_height(v,p,t) for v,p,t in zip(vertices,old,q,strict=True)])
        diag['rotation_coord_candidate_evaluated'][k]=True
        diag['rotation_coord_candidate_predicted_angle_deg'][k]=angle
        diag['rotation_coord_candidate_hand_fit_angle_deg'][k]=fitangle
        diag['rotation_coord_candidate_floor_min_z_m'][k]=minimum
        diag['rotation_coord_candidate_eligible'][k]=bool(angle<original and (fitangle<=5.).all() and (minimum>=CLEARANCE_M).all())
    eligible=np.flatnonzero(diag['rotation_coord_candidate_eligible'])
    if not len(eligible):return target,diag
    selected=min(eligible,key=lambda k:(int(SUPPRESS_MASKS[k].sum()),diag['rotation_coord_candidate_predicted_angle_deg'][k],int(k)))
    diag['rotation_coord_selected_suppressed']=SUPPRESS_MASKS[selected].copy()
    return candidates[selected],diag


class CoordinatedRotationCoPController(SharedBudgetCoPController):
    intervention_name=INTERVENTION

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rotation_coord_suppressed_count=0

    def command(self,time,loads,dt):
        old=None if self.observed_poses is None else self.observed_poses.copy()
        prelift=self.lift_start<0;previous_phase=self.motion_elapsed
        target,velocity,record=super().command(time,loads,dt)
        chosen,diag=coordinate_rotation(old,target,record,self.surface,self.support_normals,
            self.floor_vertices,prelift=prelift,time=time)
        suppressed=diag['rotation_coord_selected_suppressed']
        if suppressed.any():
            self.rotation_coord_suppressed_count+=1
            target=chosen
            rotation=Rotation.from_quat(target[:,3:])*Rotation.from_quat(old[:,3:]).inv()
            velocity=np.c_[(target[:,:3]-old[:,:3])/dt,rotation.as_rotvec()/dt]
            record['actual_alignment_active']=record['actual_alignment_active']&~suppressed
            k=np.flatnonzero((SUPPRESS_MASKS==suppressed).all(1))[0]
            record['floor_executed_target_pose_w']=target.copy()
            record['floor_executed_swept_min_z_m']=diag['rotation_coord_candidate_floor_min_z_m'][k].copy()
            # The current observation was already consumed once by the parent.
            # Only evidence for the next observation must reflect this command.
            angle,travel,phase,pure=command_evidence(old,target,record,previous_phase,dt)
            record.update(response_command_rotation_rad=angle,response_command_cop_travel_m=travel,
                response_command_phase_delta_s=phase,response_command_target_quat_xyzw=target[:,3:].copy(),
                response_command_pure=pure)
            self.response.incoming_pure=pure.copy()
            if self.lift_start>=0 or self.motion_elapsed!=previous_phase or self.ready_seconds>0:
                raise ValueError('Suppressed prelift rotation advanced unobserved admission')
        record.update(diag,rotation_coord_suppressed_count=self.rotation_coord_suppressed_count)
        return target,velocity,record


def register_intervention():
    from .controller_interventions import INTERVENTIONS,INTERVENTION_DEPENDENCIES
    spec=('coordinated_rotation_cop','CoordinatedRotationCoPController',DESCRIPTION,TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION]!=spec:raise ValueError('Conflicting local registration')
    INTERVENTIONS[INTERVENTION]=spec
    INTERVENTION_DEPENDENCIES[INTERVENTION]=('shared_budget_cop','allocated_floor_cop','floor_safe_cop',
        'hand_floor_feasibility','qualified_cop_response','prelift_cop_alignment','freeze_postlift_alignment',
        'fixed_path_surface_grip','alignment_rollback','response_gain')
