"""Observation-only normal-bisector alignment intervention on original v2.

Only the angular alignment target changes. Closure still follows each measured
local plane, and readiness still measures the hand against that actual plane.
The common axis does not establish force closure: any improvement requires real
contact migration. No existing scene installs this controller implicitly.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .surface_grip_scene import SurfaceGripController


def _limited_alignment(normal, target, dt):
    """The original v2 proportional angular correction and 3 degree/s cap."""
    cross = np.cross(normal, target)
    length = np.linalg.norm(cross)
    angle = np.arctan2(length, normal @ target)
    change = min(angle * .5, np.deg2rad(3.)) * dt
    return Rotation.from_rotvec(cross / max(length, 1e-12) * change)


class CoupledAlignmentSurfaceGripController(SurfaceGripController):
    intervention_name = 'observed_normal_bisector_alignment_v1'

    def __init__(self, controller_revision='sensor_feedback_v2', **kwargs):
        if controller_revision != 'sensor_feedback_v2':
            raise ValueError('Normal-bisector alignment requires sensor_feedback_v2')
        super().__init__(controller_revision=controller_revision, **kwargs)

    def command(self, time, loads, dt):
        old = self.observed_poses
        yaw_before = self.previous_yaw
        poses, velocity, record = super().command(time, loads, dt)
        # -1 denotes an unavailable diagnostic angle, never a measurement.
        # The original fit/angle/ready/gain/motion records remain untouched.
        record.update(common_alignment_valid=False,
                      common_alignment_applied=np.zeros(2, bool),
                      common_alignment_axis_w=np.zeros(3),
                      common_alignment_target_error_deg=np.full(2, -1.),
                      observed_normal_opposition_error_deg=-1.)
        if old is None or not np.all(record['fit_valid']):
            return poses, velocity, record

        fitted = record['fit_normal_w']
        difference = fitted[0] - fitted[1]
        length = np.linalg.norm(difference)
        if not np.isfinite(length) or length <= 1e-12:
            return poses, velocity, record
        axis = difference / length
        targets = np.stack((axis, -axis))
        rotations = Rotation.from_quat(old[:, 3:])
        normals = rotations.apply(self.support_normals)
        turn = Rotation.from_euler('z', self.previous_yaw - yaw_before, degrees=True)
        record.update(common_alignment_valid=True,
                      common_alignment_axis_w=axis.copy(),
                      common_alignment_target_error_deg=np.degrees(np.arccos(
                          np.clip(np.sum(normals * targets, axis=1), -1., 1.))),
                      observed_normal_opposition_error_deg=float(np.degrees(
                          np.arccos(np.clip(fitted[0] @ -fitted[1], -1., 1.)))))

        for side in (0, 1):
            if not record['alignment_active'][side]:
                continue
            field = self.surface
            selected = ((field['patch'] == side) & (field['pad'] >= 0)
                        & (field['pressure'] > 0) & (field['area'] > 0))
            # Exactly the measured, area-weighted pivot used by original v2.
            # The parent already checked finiteness and PCA validity.
            center = np.average(field['pos'][selected].astype(float), axis=0,
                                weights=field['area'][selected].astype(float))
            original = _limited_alignment(normals[side], fitted[side], dt) * rotations[side]
            replacement = _limited_alignment(normals[side], targets[side], dt) * rotations[side]
            # Replace only the rotation about the actual contact pivot. The
            # parent's local-plane closure and world lift/translation survive.
            # Its yaw was applied after alignment, so rotate this difference too.
            poses[side, :3] += turn.apply(original.apply(center) - replacement.apply(center))
            poses[side, 3:] = (turn * replacement).as_quat()
            record['common_alignment_applied'][side] = True

        velocity[:, :3] = (poses[:, :3] - old[:, :3]) / dt
        velocity[:, 3:] = (Rotation.from_quat(poses[:, 3:]) * rotations.inv()).as_rotvec() / dt
        return poses, velocity, record
