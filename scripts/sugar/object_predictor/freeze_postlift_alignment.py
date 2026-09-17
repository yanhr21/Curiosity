"""Single-factor diagnostic on fixed runtime path: restore old post-lift alignment freeze.

The admission step retains its original alignment. Later steps retain prescribed
yaw/lift, local-plane closure and force feedback, while undoing alignment about
the measured contact pivot. Passing the original physical criteria is unproven.
"""
import numpy as np

from .fixed_path_surface_grip import FixedPathSurfaceGripController
from .alignment_rollback import undo_alignment


class FreezePostliftAlignmentController(FixedPathSurfaceGripController):
    intervention_name = 'freeze_postlift_alignment_v1'

    def command(self, time, loads, dt):
        old = self.observed_poses
        yaw_before = self.previous_yaw
        started_before = self.lift_start >= 0.
        poses, velocity, record = super().command(time, loads, dt)
        suppressed = np.zeros(2, bool)
        if old is not None and started_before:
            suppressed = undo_alignment(self, old, yaw_before, poses, velocity, record,
                                         np.ones(2, bool), dt)
        record['postlift_alignment_frozen'] = started_before
        record['postlift_alignment_suppressed'] = suppressed
        record['actual_alignment_active'] = np.asarray(record.get('alignment_active', [False, False])) & ~suppressed
        return poses, velocity, record
