"""Isolate runtime readiness gating from the other sensor-feedback-v2 changes.

After the unchanged initial admission, advance the original four-second path
even outside the runtime load/geometry band. This is a fixed-case physical
diagnostic, not a safe-grasp policy or a qualified repair. Original physical
acceptance thresholds must still pass. Force and alignment feedback remain v2.
"""
from .surface_grip_scene import SurfaceGripController


class FixedPathSurfaceGripController(SurfaceGripController):
    intervention_name = 'legacy_runtime_path_v1'

    def __init__(self, controller_revision='sensor_feedback_v2', **kwargs):
        if controller_revision != 'sensor_feedback_v2':
            raise ValueError('Fixed runtime path diagnostic requires sensor_feedback_v2')
        super().__init__(controller_revision=controller_revision, **kwargs)

    def motion_phase_rate(self, ready, fit_valid, angle_deg, loads, dt):
        return 1.

    def command(self, time, loads, dt):
        elapsed_before = self.motion_elapsed
        poses, velocity, record = super().command(time, loads, dt)
        actual_rate = (self.motion_elapsed - elapsed_before) / dt
        record['sustained_ready'] = record.get('motion_ready', False)
        record['motion_rate'] = actual_rate
        record['actualmotion_rate'] = actual_rate
        record['motion_ready'] = actual_rate > 0.
        record['motion_paused'] = (self.lift_start >= 0 and time > self.lift_start
                                   and self.motion_elapsed < 4. and actual_rate <= 0.)
        return poses, velocity, record
