"""One prospective intervention on the unchanged sensor-feedback-v2 controller.

When both observed contact planes are valid, close along the line joining their
area-weighted contact centers. All v2 force gains, readiness rules, orientation
feedback, motion clocks and limits remain inherited. This is a control hypothesis,
not a physically qualified fix. No scene installs this controller implicitly.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .surface_grip_scene import SurfaceGripController


class CoupledSurfaceGripController(SurfaceGripController):
    intervention_name = 'observed_contact_axis_closure_v1'

    def __init__(self, controller_revision='sensor_feedback_v2', **kwargs):
        if controller_revision != 'sensor_feedback_v2':
            raise ValueError('Coupled contact-axis closure requires sensor_feedback_v2')
        super().__init__(controller_revision=controller_revision, **kwargs)

    def command(self, time, loads, dt):
        old = self.observed_poses
        distance_before = self.distance.copy()
        yaw_before = self.previous_yaw
        poses, velocity, record = super().command(time, loads, dt)
        if old is None or not np.all(record['fit_valid']):
            return poses, velocity, record

        # These are the same measured, positive-area samples used by v2's PCA.
        # The parent has already checked their finiteness and plane validity.
        local_centers = []
        for side in (0, 1):
            field = self.surface
            selected = ((field['patch'] == side) & (field['pad'] >= 0)
                        & (field['pressure'] > 0) & (field['area'] > 0))
            local_centers.append(np.average(field['pos'][selected].astype(float),
                                           axis=0, weights=field['area'][selected].astype(float)))
        rotations = Rotation.from_quat(old[:, 3:])
        centers = rotations.apply(np.asarray(local_centers)) + old[:, :3]
        separation = centers[1] - centers[0]
        length = np.linalg.norm(separation)
        # An undefined coincident-center axis cannot supply a replacement.
        if not np.isfinite(length) or length <= 1e-12:
            return poses, velocity, record
        axis = separation / length
        inward = np.stack((axis, -axis))
        delta = (inward - record['fit_normal_w']) * (self.distance - distance_before)[:, None]
        # The parent applied alignment about each contact pivot first, then
        # closure, then world yaw and translation. Rotate only this difference
        # through that same yaw; do not rotate around an object or alter lift.
        turn = Rotation.from_euler('z', self.previous_yaw - yaw_before, degrees=True)
        poses[:, :3] += turn.apply(delta)
        velocity[:, :3] = (poses[:, :3] - old[:, :3]) / dt
        return poses, velocity, record
