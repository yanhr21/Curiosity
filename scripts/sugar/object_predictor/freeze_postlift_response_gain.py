"""Single-factor diagnostic: hold actually applied gain after lift admission.

The original estimator still runs for telemetry; only the gain returned to force
control is held. This does not fix its stale-upper rule or identify stiffness.
"""
import numpy as np

from .freeze_postlift_alignment import FreezePostliftAlignmentController

INTERVENTION = 'freeze_postlift_response_gain_v1'
TELEMETRY = ('sustained_ready', 'motion_rate', 'actualmotion_rate',
    'postlift_alignment_frozen', 'postlift_alignment_suppressed', 'actual_alignment_active',
    'response_applied_gain_frozen', 'response_frozen_gain_m_per_ns',
    'response_freeze_time_s', 'response_diagnostic_candidate_gain_m_per_ns')
DESCRIPTION = ('Relative to freeze_postlift_alignment_v1, hold the two actually applied '
    'response gains from the original lift-admission command on every later command. '
    'Keep live measured force error, unchanged PCA closure direction, postlift alignment '
    'freeze and fixed four-second path. The original response estimator still updates '
    'for diagnostic telemetry; its stale-upper defect is not repaired. No generic stability claim.')


class HeldAppliedGain:
    def __init__(self, original):
        self.original = original
        self.frozen = None
        self.candidate = np.full(2, original.base_gain)

    def update(self, side, time, load, distance, dt, allow_increase):
        gain, upper, samples = self.original.update(side, time, load, distance, dt, allow_increase)
        self.candidate[side] = gain
        return (gain if self.frozen is None else float(self.frozen[side])), upper, samples


class FreezePostliftResponseGainController(FreezePostliftAlignmentController):
    intervention_name = INTERVENTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.response_gain or not self.revised_feedback:
            raise ValueError('Explicit sensor_feedback_v2 with original response_gain required')
        self.response = HeldAppliedGain(self.response)
        self.response_freeze_time = -1.

    def command(self, time, loads, dt):
        was_frozen = self.response.frozen is not None
        poses, velocity, record = super().command(time, loads, dt)
        if not was_frozen and self.lift_start >= 0:
            if self.motion_elapsed != 0.:
                raise AssertionError('Capture actual admission gain before runtime path advances')
            self.response.frozen = record['response_gain_m_per_ns'].copy()
            self.response_freeze_time = float(time)
        record.update(response_applied_gain_frozen=was_frozen,
            response_frozen_gain_m_per_ns=(np.zeros(2) if self.response.frozen is None else self.response.frozen.copy()),
            response_freeze_time_s=self.response_freeze_time,
            response_diagnostic_candidate_gain_m_per_ns=self.response.candidate.copy())
        return poses, velocity, record


def register_intervention():
    """Explicit registration only in this diagnostic collector process.

    Never edit or replace an existing controller/default/estimator class.
    """
    from .controller_interventions import INTERVENTIONS, INTERVENTION_DEPENDENCIES
    spec = ('freeze_postlift_response_gain', 'FreezePostliftResponseGainController', DESCRIPTION, TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION] != spec:
        raise ValueError('Conflicting diagnostic registration')
    INTERVENTIONS[INTERVENTION] = spec
    INTERVENTION_DEPENDENCIES[INTERVENTION] = ('freeze_postlift_alignment', 'fixed_path_surface_grip',
        'alignment_rollback', 'coupled_alignment_surface_grip', 'response_gain')
