"""Bounded observed-CoP prelift alignment candidate, not a qualified controller.

Keep the original 15/16 controller and response-gain estimator. Add only a
prelift symmetric tangential servo and an admission gate from observed pressure
centers and fitted normals. No object pose, mass, COM or geometry is consumed.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .freeze_postlift_alignment import FreezePostliftAlignmentController
from .inspect_contact_surface import fit_surface

INTERVENTION = 'prelift_cop_alignment_v1'
DESCRIPTION = ('Original freeze_postlift_alignment controller with bounded prelift '
    'pressure-center tangential registration, original response gain and force '
    'targets; require opposed contact line within5deg for original1s admission. '
    'No runtime tangential servo, GT feedback or automatic qualification claim.')
TELEMETRY = ('sustained_ready', 'motion_rate', 'actualmotion_rate',
    'postlift_alignment_frozen', 'postlift_alignment_suppressed', 'actual_alignment_active',
    'cop_valid', 'cop_world_m', 'cop_tangent_error_m', 'cop_line_angle_deg',
    'cop_admission_ready', 'cop_servo_velocity_m_s', 'cop_travel_m', 'cop_budget_exhausted')


def observed_alignment(poses, surface, support_normals):
    """Match the original plane validity checks; pressure×area CoP per hand."""
    rotations = Rotation.from_quat(poses[:, 3:])
    hand_normals = rotations.apply(support_normals)
    normals = hand_normals.copy()
    centers = np.zeros((2, 3))
    valid = np.zeros(2, bool)
    for side in (0, 1):
        selected = ((surface['patch'] == side) & (surface['pad'] >= 0)
                    & (surface['pressure'] > 0) & (surface['area'] > 0))
        points, area = surface['pos'][selected], surface['area'][selected]
        weights = surface['pressure'][selected] * area
        if not all(np.isfinite(a).all() for a in (points, area, weights)):
            raise ValueError('Nonfinite measured contact field')
        if len(points) < 6 or weights.sum() <= 0:
            continue
        _, rms, normal = fit_surface(points.astype(float), area.astype(float))
        if rms[1] < .0002 or rms[0] / max(rms[1], 1e-12) > .35:
            continue
        normal = rotations[side].apply(normal)
        if normal @ hand_normals[side] < 0:
            normal = -normal
        normals[side] = normal
        centers[side] = rotations[side].apply(np.average(points, axis=0, weights=weights)) + poses[side, :3]
        valid[side] = True
    tangent = np.zeros(3)
    angle = 180.
    if valid.all():
        normal = normals[0] - normals[1]
        length = np.linalg.norm(normal)
        if length < 1e-8:
            valid[:] = False
        else:
            normal /= length
            delta = centers[1] - centers[0]
            separation = float(delta @ normal)
            tangent = delta - normal * separation
            # Wrong-facing / coincident contact geometry is never admitted.
            if separation > 1e-6:
                angle = float(np.degrees(np.arctan2(np.linalg.norm(tangent), separation)))
    return valid, centers, tangent, angle


class PreliftCoPAlignmentController(FreezePostliftAlignmentController):
    intervention_name = INTERVENTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.response_gain or not self.revised_feedback:
            raise ValueError('Original sensor_feedback_v2 response gain required')
        self.cop_travel = np.zeros(2)

    def command(self, time, loads, dt):
        if dt <= 0:
            raise ValueError('Positive control dt required')
        prelift = self.lift_start < 0
        if self.observed_poses is None:
            valid, centers, tangent, angle = np.zeros(2, bool), np.zeros((2, 3)), np.zeros(3), 180.
        else:
            valid, centers, tangent, angle = observed_alignment(
                self.observed_poses, self.surface, self.support_normals)
        aligned = bool(valid.all() and angle <= 5.)
        if prelift and not aligned:
            # Parent adds at most dt before admission. Resetting to -dt prevents
            # this command admitting, while retaining its unmodified force loop.
            self.ready_seconds = -dt
        poses, velocity, record = super().command(time, loads, dt)
        servo = np.zeros((2, 3))
        if (self.observed_poses is not None and self.lift_start < 0
                and 1. <= time <= 40. and valid.all() and not aligned):
            candidate = .25 * tangent  # +/- .5 * k * e, k=.5/s
            speed = np.linalg.norm(candidate)
            if speed > .004:
                candidate *= .004 / speed
            for side, sign in ((0, 1.), (1, -1.)):
                step = sign * candidate * dt
                length = np.linalg.norm(step)
                remaining = max(0., .10 - self.cop_travel[side])
                if length > remaining:
                    step *= remaining / length
                self.cop_travel[side] += np.linalg.norm(step)
                poses[side, :3] += step
                velocity[side, :3] += step / dt
                servo[side] = step / dt
        record.update(cop_valid=valid, cop_world_m=centers,
            cop_tangent_error_m=tangent, cop_line_angle_deg=angle,
            cop_admission_ready=aligned, cop_servo_velocity_m_s=servo,
            cop_travel_m=self.cop_travel.copy(),
            cop_budget_exhausted=bool((self.cop_travel >= .10-1e-12).any() and not aligned))
        return poses, velocity, record


def register_intervention():
    from .controller_interventions import INTERVENTIONS, INTERVENTION_DEPENDENCIES
    spec = ('prelift_cop_alignment', 'PreliftCoPAlignmentController', DESCRIPTION, TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION] != spec:
        raise ValueError('Conflicting process-local intervention registration')
    INTERVENTIONS[INTERVENTION] = spec
    INTERVENTION_DEPENDENCIES[INTERVENTION] = (
        'freeze_postlift_alignment', 'fixed_path_surface_grip', 'alignment_rollback', 'response_gain')
