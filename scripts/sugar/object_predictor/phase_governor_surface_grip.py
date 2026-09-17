"""Prospective continuous runtime phase governor on sensor-feedback-v2.

Initial one-second admission, contact validity, five-degree alignment, the load
band, force feedback and the geometric path remain unchanged. Physical benefit
requires a new fixed-case qualification; this is not a validated repair.
"""
import numpy as np

from .surface_grip_scene import SurfaceGripController


class PhaseGovernorSurfaceGripController(SurfaceGripController):
    intervention_name = 'continuous_phase_governor_v1'
    phase_rate_rise_per_s = 2.

    def __init__(self, controller_revision='sensor_feedback_v2', **kwargs):
        if controller_revision != 'sensor_feedback_v2':
            raise ValueError('Continuous phase governor requires sensor_feedback_v2')
        super().__init__(controller_revision=controller_revision, **kwargs)
        self.governor_raw_rate = 0.
        self.governor_rate = 0.

    def motion_phase_rate(self, ready, fit_valid, angle_deg, loads, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Phase governor requires positive finite dt')
        # The instantaneous gate is the same v2 gate. Only its runtime use
        # differs: no new one-second dwell, and no full-speed binary restart.
        valid = (bool(ready) and np.all(fit_valid) and np.all(np.asarray(angle_deg) <= 5.)
                 and np.isfinite(loads).all())
        margin = np.clip(1. - np.abs(np.asarray(loads) / self.target_load_n - 1.) / .25, 0., 1.)
        self.governor_raw_rate = float(np.min(margin)) if valid else 0.
        self.governor_rate = min(self.governor_raw_rate,
                                 self.governor_rate + self.phase_rate_rise_per_s * dt)
        return self.governor_rate

    def command(self, time, loads, dt):
        elapsed_before = self.motion_elapsed
        poses, velocity, record = super().command(time, loads, dt)
        actual_rate = (self.motion_elapsed - elapsed_before) / dt
        # Preserve the original one-second readiness evidence explicitly.
        # New motion fields describe actual phase advancement, not a rewritten
        # force/geometry test. A completed path is never counted as paused.
        record['sustained_ready'] = record.get('motion_ready', False)
        record['motion_rate'] = actual_rate
        record['phase_governor_raw_rate'] = self.governor_raw_rate
        record['phase_governor_rate'] = self.governor_rate
        record['motion_ready'] = actual_rate > 0.
        record['motion_paused'] = (self.lift_start >= 0 and time > self.lift_start
                                   and self.motion_elapsed < 4. and actual_rate <= 0.)
        return poses, velocity, record
