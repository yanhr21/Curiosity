"""Tactile-plane feedback acquisition diagnostic with original Newton physics.

No learned policy. Inputs: hand proprioception, measured pad loads and assigned
ideal tactile surface positions/areas. Neither object truth nor interface normals
enter the controller. The optional sustained-feedback variant is a bounded
acquisition diagnostic; the original corpus uses the default behavior.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .grip_motion_scene import GripMotionController, GripMotionScene
from .inspect_contact_surface import fit_surface

CONTROLLER_REVISIONS = ('legacy', 'sensor_feedback_v2')


class SurfaceGripController(GripMotionController):
    def __init__(self, force_gain=.00015, force_integral_time_s=0., sustained_feedback=False,
                 alignment_deadband=False, normalized_acquisition=False, response_gain=False,
                 controller_revision='legacy', **kwargs):
        if controller_revision not in CONTROLLER_REVISIONS:
            raise ValueError('Unknown surface controller revision')
        revised = controller_revision == 'sensor_feedback_v2'
        if revised and (sustained_feedback or alignment_deadband or normalized_acquisition or force_integral_time_s):
            raise ValueError('sensor_feedback_v2 declares its own feedback rules; do not combine legacy variants')
        super().__init__(**kwargs)
        self.controller_revision = controller_revision
        self.revised_feedback = revised
        self.force_gain = float(force_gain)
        self.force_integral_time_s = float(force_integral_time_s)
        self.integral_velocity = np.zeros(2)
        self.observed_poses = None
        self.surface = None
        self.contact_latched = np.zeros(2, bool)
        self.ready_seconds = 0.
        self.lift_start = -1.
        self.previous_lift = 0.
        self.previous_yaw = 0.
        self.sustained_feedback = bool(sustained_feedback)
        # The named revision explicitly combines measured-ready motion gating,
        # continued alignment and the optional existing response-gain estimator.
        # Keep legacy flags distinct in records; no legacy option is reinterpreted.
        self.feedback_motion = self.sustained_feedback or revised
        self.alignment_deadband = bool(alignment_deadband)
        self.normalized_acquisition = bool(normalized_acquisition)
        self.response_gain = bool(response_gain)
        if self.response_gain:
            if force_integral_time_s or sustained_feedback or alignment_deadband or normalized_acquisition:
                raise ValueError('Response gain diagnostic must be isolated from other controller changes')
            from .response_gain import ObservedForceGain
            self.response = ObservedForceGain(self.force_gain)
        if self.normalized_acquisition and (self.target_load_n not in (12.,24.) or self.force_integral_time_s != 0.):
            raise ValueError('Normalized acquisition diagnostic requires12/24N and no integral')
        self.motion_elapsed = 0.
        self.lift_complete = -1.
        self.support_normals = np.stack([frame[0].inv().apply([0., 0., 1.]) for frame in self.frames])

    def observe(self, poses, surface):
        self.observed_poses = np.asarray(poses).copy()
        self.surface = {k: np.asarray(surface[k]).copy() for k in ('pos', 'area', 'pressure', 'pad', 'patch')}

    def motion_phase_rate(self, ready, fit_valid, angle_deg, loads, dt):
        """Default v2 runtime admission; independent variants may override."""
        return float(self.ready_seconds >= 1.)

    def command(self, time, loads, dt):
        if self.observed_poses is None:
            return super().command(0., loads, dt)
        old = self.observed_poses
        poses = old.copy()
        rotations = Rotation.from_quat(old[:, 3:])
        normals = rotations.apply(self.support_normals)
        fit_valid = np.zeros(2, bool)
        angle_deg = np.full(2, 180.)
        fitted = normals.copy()
        centers = np.zeros((2, 3))
        areas = np.zeros(2)
        alignment_active = np.zeros(2, bool)
        acquisition_gain_multiplier = np.ones(2)
        response_gain_values = np.full(2, self.force_gain)
        response_upper = np.zeros(2)
        response_samples = np.zeros(2, dtype=np.int32)
        self.contact_latched |= loads >= .2
        for side in (0, 1):
            field = self.surface
            sel = (field['patch'] == side) & (field['pad'] >= 0) & (field['pressure'] > 0)
            points = field['pos'][sel]
            area = field['area'][sel]
            areas[side] = area.sum()
            if self.revised_feedback:
                positive = area > 0
                points = points[positive]
                area = area[positive]
                if not np.isfinite(points).all() or not np.isfinite(area).all():
                    raise ValueError('Nonfinite observed contact geometry')
            # In v2, resolved contact geometry is usable even at low normal
            # load. Pressure>0 above remains a measured-contact requirement;
            # the target-load readiness band below is deliberately unchanged.
            if len(points) >= 6 and (self.revised_feedback or loads[side] >= 2.):
                center, rms, normal = fit_surface(points.astype(float), area.astype(float))
                # Geometry quality, not simulator truth, decides whether a
                # patch can supply a plane. At least two resolved directions.
                if rms[1] >= .0002 and rms[0] / max(rms[1], 1e-12) <= .35:
                    n = rotations[side].apply(normal)
                    if n @ normals[side] < 0:
                        n = -n
                    fitted[side] = n
                    centers[side] = center
                    fit_valid[side] = True
                    angle_deg[side] = np.degrees(np.arccos(np.clip(n @ normals[side], -1, 1)))
            inward = fitted[side] if fit_valid[side] else normals[side]
            error = self.target_load_n - loads[side]
            if self.force_integral_time_s > 0:
                if fit_valid[side]:
                    # Integral acts much slower than the 20 ms proportional
                    # force loop. Cap its velocity and stop integration when
                    # the position or velocity limit blocks the same direction.
                    raw = error * self.force_gain + self.integral_velocity[side]
                    blocked = ((raw >= .004 and error > 0) or (raw <= -.008 and error < 0)
                               or (self.distance[side] >= .20 and error > 0)
                               or (self.distance[side] <= -.02 and error < 0))
                    if not blocked:
                        self.integral_velocity[side] = np.clip(self.integral_velocity[side] +
                            error * self.force_gain / self.force_integral_time_s * dt, -.002, .002)
                else:
                    self.integral_velocity[side] = 0.
            if self.normalized_acquisition and self.lift_start < 0:
                weight = np.clip((.75 - loads[side] / self.target_load_n) / .25, 0., 1.)
                acquisition_gain_multiplier[side] = 1. + weight * (24. / self.target_load_n - 1.)
            gain = self.force_gain * acquisition_gain_multiplier[side]
            if self.response_gain:
                gain, response_upper[side], response_samples[side] = self.response.update(
                    side, time, float(loads[side]), float(self.distance[side]), dt, bool(fit_valid[side]))
                response_gain_values[side] = gain
            speed = np.clip(error * gain + self.integral_velocity[side], -.008, .004) if self.contact_latched[side] else .012
            if time < .5:
                speed = 0.
            distance = float(np.clip(self.distance[side] + speed * dt, -.02, .20))
            speed = (distance - self.distance[side]) / dt
            self.distance[side] = distance
            # Diagnostic: after the original earliest lift clock, stop
            # alignment inside the same five-degree cone used by readiness.
            # Earlier acquisition, force control and the readiness test stay
            # identical. Do not substitute the unqualified SDF correction.
            within_deadband = self.alignment_deadband and time >= 24. and angle_deg[side] <= 5.
            if fit_valid[side] and (self.lift_start < 0 or self.feedback_motion) and not within_deadband:
                alignment_active[side] = True
                cross = np.cross(normals[side], fitted[side])
                angle = np.arctan2(np.linalg.norm(cross), normals[side] @ fitted[side])
                change = min(angle * .5, np.deg2rad(3.)) * dt
                correction = Rotation.from_rotvec(cross / max(np.linalg.norm(cross), 1e-12) * change)
                updated = correction * rotations[side]
                # Rotate about the measured contact center, not an object COM.
                pivot = rotations[side].apply(centers[side]) + old[side, :3]
                poses[side, :3] = pivot - updated.apply(centers[side])
                poses[side, 3:] = updated.as_quat()
            poses[side, :3] += inward * speed * dt
        ready = bool(fit_valid.all() and (angle_deg <= 5.).all() and
                     (abs(loads / self.target_load_n - 1) <= .25).all())
        self.ready_seconds = self.ready_seconds + dt if ready else 0.
        if self.lift_start < 0 and 24. <= time <= 40. and self.ready_seconds >= 1.:
            self.lift_start = time
        motion_ready = self.ready_seconds >= 1.
        if self.feedback_motion:
            # Advance the same four-second path only with sustained measured
            # support. Force servo and contact-pivot alignment continue while
            # paused; neither hidden object state nor full-hand forces are used.
            if self.lift_start >= 0 and time > self.lift_start:
                rate = self.motion_phase_rate(ready, fit_valid, angle_deg, loads, dt)
                if rate > 0.:
                    self.motion_elapsed = min(4., self.motion_elapsed + dt * rate)
                    if self.motion_elapsed >= 4. - 1e-9:
                        self.motion_elapsed = 4.
            u = self.motion_elapsed / 4.
        else:
            u = np.clip((time - self.lift_start) / 4., 0., 1.) if self.lift_start >= 0 else 0.
        if u >= 1. and self.lift_complete < 0:
            self.lift_complete = time
        lift = .5 * (1 - np.cos(np.pi * u))
        yaw = self.yaw_delta_deg * lift
        turn = Rotation.from_euler('z', yaw - self.previous_yaw, degrees=True)
        # Rotate around the midpoint of the observed hands. This is a known
        # robot frame, not the object's hidden center.
        midpoint = old[:, :3].mean(0)
        poses[:, :3] = turn.apply(poses[:, :3] - midpoint) + midpoint
        poses[:, 3:] = (turn * Rotation.from_quat(poses[:, 3:])).as_quat()
        poses[:, :3] += np.r_[self.lateral_xy, self.lift_height_m] * (lift - self.previous_lift)
        self.previous_lift = lift; self.previous_yaw = yaw
        velocity = np.c_[(poses[:, :3] - old[:, :3]) / dt,
                         (Rotation.from_quat(poses[:, 3:]) * rotations.inv()).as_rotvec() / dt]
        record = dict(phase=0 if self.lift_start < 0 else (1 if u < 1 else 2),
                      phase_time=time, distance=self.distance.copy(), touched=self.contact_latched.copy(),
                      target_load_n=self.target_load_n, lift_command_m=self.lift_height_m * lift,
                      lift_start_s=self.lift_start, ready_seconds=self.ready_seconds,
                      fit_valid=fit_valid, fit_normal_w=fitted, alignment_error_deg=angle_deg,
                      contact_area_m2=areas, integral_velocity_m_s=self.integral_velocity.copy(),
                      alignment_active=alignment_active,
                      acquisition_gain_multiplier=acquisition_gain_multiplier,
                      response_gain_m_per_ns=response_gain_values,
                      response_upper_n_m=response_upper, response_samples=response_samples,
                      motion_elapsed_s=self.motion_elapsed if self.feedback_motion else 4. * u,
                      motion_ready=motion_ready, lift_complete_s=self.lift_complete,
                      motion_paused=self.lift_start >= 0 and u < 1. and not motion_ready)
        return poses, velocity, record


class SurfaceGripScene(GripMotionScene):
    def __init__(self, target_load_n=12., approach_angle_deg=0., yaw_delta_deg=0.,
                 lift_height_m=.18, lateral_xy=(0., .04), force_gain=.00015, force_integral_time_s=0., sustained_feedback=False,
                 alignment_deadband=False, normalized_acquisition=False,
                 sensor_coverage='anatomical27', response_gain=False,
                 controller_revision='legacy', controller_type=None, **kwargs):
        from .palmar_coverage import COVERAGES, ContinuousPalmarField
        if sensor_coverage not in COVERAGES:
            raise ValueError('Unknown explicitly declared sensor coverage')
        params = dict(target_load_n=target_load_n, approach_angle_deg=approach_angle_deg,
                      yaw_delta_deg=yaw_delta_deg, lift_height_m=lift_height_m, lateral_xy=lateral_xy)
        super().__init__(**params, **kwargs)
        self.sensor_coverage = sensor_coverage
        if sensor_coverage == 'continuous_palmar_v1':
            self.field = ContinuousPalmarField(self.field, self.palm_signs)
        selected_controller = SurfaceGripController if controller_type is None else controller_type
        if not issubclass(selected_controller, SurfaceGripController):
            raise TypeError('Controller adapter must retain the surface feedback implementation')
        self.probe = selected_controller(**params, force_gain=force_gain, force_integral_time_s=force_integral_time_s,
                                          sustained_feedback=sustained_feedback, alignment_deadband=alignment_deadband,
                                          normalized_acquisition=normalized_acquisition, response_gain=response_gain,
                                          controller_revision=controller_revision)

    def step(self, dt=.02, substeps=8):
        self.probe.observe(self.state_0.body_q.numpy()[self.patch_frame], self.field.to_numpy())
        super().step(dt=dt, substeps=substeps)
