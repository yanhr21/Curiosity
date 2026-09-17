"""Single-factor diagnostic: suppress the common-bisector controller's unilateral fallback."""
import numpy as np

from .coupled_alignment_surface_grip import CoupledAlignmentSurfaceGripController
from .alignment_rollback import undo_alignment


class BilateralCommonAlignmentController(CoupledAlignmentSurfaceGripController):
    intervention_name = 'bilateral_common_alignment_v1'

    def command(self, time, loads, dt):
        old = self.observed_poses
        yaw_before = self.previous_yaw
        poses, velocity, record = super().command(time, loads, dt)
        suppressed = np.zeros(2, bool)
        if old is not None and not np.all(record['fit_valid']):
            suppressed = undo_alignment(self, old, yaw_before, poses, velocity, record,
                                         np.ones(2, bool), dt)
        record['unilateral_alignment_suppressed'] = suppressed
        record['actual_alignment_active'] = np.asarray(record.get('alignment_active', [False, False])) & ~suppressed
        return poses, velocity, record
