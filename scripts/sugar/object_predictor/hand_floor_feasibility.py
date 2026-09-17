"""CPU geometry only: certify complete rigid hand sweeps above a public floor.

No controller, object geometry, force, solver, or model is accessed. The path is
exactly the linear translation / shortest quaternion SLERP used by ProbeScene.
This helper is not wired into any collection protocol. A controller adapter
must separately reconcile its state and telemetry with the executed command.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _inputs(vertices, old_pose, target_pose):
    v = np.asarray(vertices, dtype=np.float64)
    old = np.asarray(old_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError('A nonempty full vertex array is required')
    if old.shape != (7,) or target.shape != (7,):
        raise ValueError('Poses must have xyz + xyzw shape (7,)')
    if not all(np.isfinite(x).all() for x in (v, old, target)):
        raise ValueError('Nonfinite geometry')
    r0, r1 = Rotation.from_quat(old[3:]), Rotation.from_quat(target[3:])
    return v, old, target, r0, r1


def swept_min_height(vertices, old_pose, target_pose, fraction=1.):
    """Minimum vertex/triangle height over every s in [0, fraction].

For a world rotation axis u and w=R0*v, vertex height is
    c + a*cos(theta*s) + b*sin(theta*s) + dz*s.
Evaluate endpoints and every analytic stationary point. A triangle's affine
height is bounded by its vertices at every s, so this covers complete faces.
Float64 arithmetic is not an interval-arithmetic proof; callers reserve a
positive clearance for float32 physics pose storage.
"""
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('Fraction must lie in [0,1]')
    v, old, target, r0, r1 = _inputs(vertices, old_pose, target_pose)
    # Matrix multiply also supports immutable CAD arrays on SciPy versions whose
    # Rotation.apply Cython buffer incorrectly requires writeable input.
    w = v @ r0.as_matrix().T
    rotvec = (r1 * r0.inv()).as_rotvec()
    theta = float(np.linalg.norm(rotvec))
    dz = float(target[2] - old[2])
    initial = float(w[:, 2].min() + old[2])
    if theta == 0. or fraction == 0.:
        return initial + min(0., dz * fraction)
    axis = rotvec / theta
    parallel_z = (w @ axis) * axis[2]
    a = w[:, 2] - parallel_z
    b = np.cross(axis, w)[:, 2]
    c = old[2] + parallel_z
    end_angle = theta * fraction
    result = min(initial, float((c + a * np.cos(end_angle)
                                + b * np.sin(end_angle) + dz * fraction).min()))
    amplitude = np.hypot(a, b)
    rotating = amplitude > 0.
    ratio = np.zeros_like(amplitude)
    np.divide(-dz, theta * amplitude, out=ratio, where=rotating)
    eligible = rotating & (np.abs(ratio) <= 1.)
    if not eligible.any():
        return result
    phase = np.arctan2(a[eligible], b[eligible])
    root = np.arccos(np.clip(ratio[eligible], -1., 1.))
    ae, be, ce = a[eligible], b[eligible], c[eligible]
    # theta <= pi; phase in [-pi, pi], hence these periods include all roots.
    for sign in (-1., 1.):
        for period in (-1, 0, 1):
            t = sign * root - phase + period * (2. * np.pi)
            inside = (t >= 0.) & (t <= end_angle)
            if inside.any():
                ti = t[inside]
                heights = ce[inside] + ae[inside] * np.cos(ti) + be[inside] * np.sin(ti) + dz * ti / theta
                result = min(result, float(heights.min()))
    return result


def interpolate_pose(old_pose, target_pose, fraction):
    old = np.asarray(old_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    r0, r1 = Rotation.from_quat(old[3:]), Rotation.from_quat(target[3:])
    rotation = Rotation.from_rotvec((r1 * r0.inv()).as_rotvec() * fraction) * r0
    return np.r_[old[:3] + fraction * (target[:3] - old[:3]), rotation.as_quat()]


def feasible_hand_step(vertices_by_hand, old_poses, target_poses, *, floor_z_m=0., clearance_m=1e-5):
    """Largest common feasible path prefix, with a 10um default storage margin.

Old unsafe poses are rejected, never teleported upward. Both hands share one
fraction. Full feasible commands are returned byte-identically. The margin is
a new explicit geometry constraint, not a relaxed physical acceptance gate.
No controller bookkeeping is changed here; this is preparation-only geometry.
"""
    old, target = np.asarray(old_poses), np.asarray(target_poses)
    if old.shape != (len(vertices_by_hand), 7) or target.shape != old.shape:
        raise ValueError('One full mesh and pose per hand are required')
    if not np.isfinite([floor_z_m, clearance_m]).all() or clearance_m <= 0.:
        raise ValueError('Positive finite storage margin is required')
    floor = floor_z_m + clearance_m
    def minimum(fraction):
        return np.asarray([swept_min_height(v, p, q, fraction)
                           for v, p, q in zip(vertices_by_hand, old, target, strict=True)])
    initial, requested = minimum(0.), minimum(1.)
    if (initial < floor).any():
        raise ValueError('Initial full hand is below the reserved floor clearance')
    if (requested >= floor).all():
        return target.copy(), dict(fraction=1., clipped=False,
            initial_min_z_m=initial, requested_swept_min_z_m=requested,
            executed_swept_min_z_m=requested, clearance_m=clearance_m)
    lo, hi = 0., 1.
    # The minimum over the entire prefix is monotone even when z(s) is not.
    for _ in range(48):
        mid = .5 * (lo + hi)
        if (minimum(mid) >= floor).all():
            lo = mid
        else:
            hi = mid
    result = np.stack([interpolate_pose(p, q, lo) for p, q in zip(old, target, strict=True)])
    return result, dict(fraction=lo, clipped=True,
        initial_min_z_m=initial, requested_swept_min_z_m=requested,
        executed_swept_min_z_m=minimum(lo), clearance_m=clearance_m)
