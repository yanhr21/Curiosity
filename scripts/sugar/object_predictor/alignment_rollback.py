"""Undo only requested v2 contact-pivot alignment in independent diagnostics."""
import numpy as np
from scipy.spatial.transform import Rotation

from .coupled_alignment_surface_grip import _limited_alignment


def undo_alignment(controller, old, yaw_before, poses, velocity, record, selected, dt):
    """Preserve closure, lift and yaw; return which original rotations were undone."""
    suppressed = np.zeros(2, bool)
    if old is None:
        return suppressed
    rotations = Rotation.from_quat(old[:, 3:])
    normals = rotations.apply(controller.support_normals)
    turn = Rotation.from_euler('z', controller.previous_yaw-yaw_before, degrees=True)
    for side in (0, 1):
        if not selected[side] or not record['alignment_active'][side]:
            continue
        field = controller.surface
        valid = ((field['patch'] == side) & (field['pad'] >= 0)
                 & (field['pressure'] > 0) & (field['area'] > 0))
        center = np.average(field['pos'][valid].astype(float), axis=0,
                            weights=field['area'][valid].astype(float))
        original = _limited_alignment(normals[side], record['fit_normal_w'][side],
                                      dt) * rotations[side]
        poses[side, :3] += turn.apply(original.apply(center)-rotations[side].apply(center))
        poses[side, 3:] = (turn * rotations[side]).as_quat()
        suppressed[side] = True
    if suppressed.any():
        velocity[:, :3] = (poses[:, :3]-old[:, :3]) / dt
        velocity[:, 3:] = (Rotation.from_quat(poses[:, 3:]) * rotations.inv()).as_rotvec() / dt
    return suppressed
