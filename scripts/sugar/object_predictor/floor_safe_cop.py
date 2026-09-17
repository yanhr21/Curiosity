"""Known-hand swept-floor transaction on the qualified-CoP original orientation.

Accept the original full command/state or reject its complete motion. No partial
step is executed; observations still enter the original response exactly once.
No object GT, force target, acquisition deadline or original gate is changed.
"""
from copy import deepcopy
import numpy as np
from .qualified_cop_response import (
    QualifiedCoPResponseController, TELEMETRY as PARENT_TELEMETRY, command_evidence)
from .hand_floor_feasibility import swept_min_height

INTERVENTION = 'floor_safe_qualified_cop_v1'
DESCRIPTION = ('Qualified-CoP original initial hand orientation; complete known-hand '
    'linear-translation/SLERP floor certificate, all-or-none command transaction; '
    'rejected motion rolls back progress and readiness, real force observation enters once.')
CLEARANCE_M = 1e-5
TELEMETRY = PARENT_TELEMETRY + (
    'floor_blocked', 'floor_blocked_count', 'floor_requested_swept_min_z_m',
    'floor_executed_swept_min_z_m', 'floor_observed_min_z_m',
    'floor_executed_fraction', 'floor_requested_target_pose_w',
    'floor_requested_cop_velocity_m_s', 'floor_requested_motion_elapsed_s',
    'floor_requested_lift_start_s', 'floor_requested_distance_m',
    'floor_requested_ready_seconds', 'floor_requested_alignment_active')


class FloorSafeCoPController(QualifiedCoPResponseController):
    intervention_name = INTERVENTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sugar_newton.hand.patches import load_hand_mesh
        self.floor_vertices = tuple(np.asarray(load_hand_mesh(side).vertices, np.float32).astype(float)
                                    for side in ('left', 'right'))
        for vertices in self.floor_vertices:
            vertices.flags.writeable = False
        self.floor_blocked_count = 0

    def command(self, time, loads, dt):
        # Full meshes are immutable; do not duplicate them every 20ms.
        candidate = deepcopy(self, {id(v): v for v in self.floor_vertices})
        target, velocity, record = QualifiedCoPResponseController.command(candidate, time, loads, dt)
        old = self.observed_poses
        initial = target if old is None else old
        observed_min = np.array([swept_min_height(v, p, p)
                                 for v, p in zip(self.floor_vertices, initial, strict=True)])
        if (observed_min < CLEARANCE_M).any():
            raise ValueError('Observed full hand already violates floor reserve; no teleport recovery')
        requested_min = np.array([swept_min_height(v, p, q)
                                  for v, p, q in zip(self.floor_vertices, initial, target, strict=True)])
        blocked = bool((requested_min < CLEARANCE_M).any())
        diagnostic = dict(floor_blocked=blocked,
            floor_requested_swept_min_z_m=requested_min,
            floor_observed_min_z_m=observed_min,
            floor_requested_target_pose_w=target.copy(),
            floor_requested_cop_velocity_m_s=record['cop_servo_velocity_m_s'].copy(),
            floor_requested_motion_elapsed_s=record.get('motion_elapsed_s', self.motion_elapsed),
            floor_requested_lift_start_s=record.get('lift_start_s', self.lift_start),
            floor_requested_distance_m=record['distance'].copy(),
            floor_requested_ready_seconds=record.get('ready_seconds', self.ready_seconds),
            floor_requested_alignment_active=record['actual_alignment_active'].copy())
        if not blocked:
            self.__dict__.update(candidate.__dict__)
            executed_min = requested_min
        else:
            if old is None:
                raise ValueError('Cannot reject an unknown initial motion')
            # The estimator ran BEFORE closure updates, using prior real distance
            # and prior real incoming_pure. Keep this single causal observation.
            # No candidate action affects its slopes/upper; replace only its NEXT
            # interval evidence below. All motion-derived state remains uncommitted.
            self.response = candidate.response
            self.contact_latched = candidate.contact_latched.copy()
            self.ready_seconds = 0.
            self.floor_blocked_count += 1
            target, velocity = old.copy(), np.zeros((2, 6))
            executed_min = observed_min
            record.update(distance=self.distance.copy(), touched=self.contact_latched.copy(),
                ready_seconds=0., lift_start_s=self.lift_start, lift_complete_s=self.lift_complete,
                lift_command_m=self.lift_height_m * self.previous_lift,
                phase=0 if self.lift_start < 0. else (1 if self.motion_elapsed < 4. else 2),
                motion_elapsed_s=self.motion_elapsed, motion_ready=False,
                motion_paused=bool(self.lift_start >= 0. and self.motion_elapsed < 4.),
                motion_rate=0., actualmotion_rate=0., sustained_ready=False,
                alignment_active=np.zeros(2, bool), actual_alignment_active=np.zeros(2, bool),
                cop_servo_velocity_m_s=np.zeros((2, 3)), cop_travel_m=self.cop_travel.copy(),
                cop_budget_exhausted=bool((self.cop_travel >= .10-1e-12).any()
                    and not record['cop_admission_ready']))
            rotation, travel, phase, pure = command_evidence(old, target, record, self.motion_elapsed, dt)
            record.update(response_command_rotation_rad=rotation,
                response_command_cop_travel_m=travel, response_command_phase_delta_s=phase,
                response_command_target_quat_xyzw=target[:, 3:].copy(), response_command_pure=pure)
            self.response.incoming_pure = pure.copy()
        record.update(diagnostic, floor_blocked_count=self.floor_blocked_count,
            floor_executed_swept_min_z_m=executed_min, floor_executed_fraction=0. if blocked else 1.)
        return target, velocity, record


def register_intervention():
    from .controller_interventions import INTERVENTIONS, INTERVENTION_DEPENDENCIES
    spec = ('floor_safe_cop', 'FloorSafeCoPController', DESCRIPTION, TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION] != spec:
        raise ValueError('Conflicting local floor intervention')
    INTERVENTIONS[INTERVENTION] = spec
    INTERVENTION_DEPENDENCIES[INTERVENTION] = (
        'hand_floor_feasibility', 'qualified_cop_response', 'prelift_cop_alignment',
        'freeze_postlift_alignment', 'fixed_path_surface_grip', 'alignment_rollback', 'response_gain')
