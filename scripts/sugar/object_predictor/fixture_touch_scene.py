"""Real dynamic-hand touch acquisition against a fixed full-mesh object.

This fixture isolates shape sensing in a public, known frame. It does not test
unknown pose, mass, material or carrying. Original Newton/MuJoCo D6 actuators
integrate the hand dynamics; no per-step hand pose/velocity is overwritten.
Contact fields use the existing solved-force tactile pipeline unchanged.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FixtureDrive:
    hand_mass_kg: float = .25
    start_radius_m: float = .28
    max_travel_m: float = .32
    target_load_n: float = 2.
    contact_latch_n: float = .2
    approach_speed_m_s: float = .04
    force_gain_m_ns: float = .001
    contact_speed_limit_m_s: float = .004
    overload_n: float = 20.
    retreat_speed_m_s: float = .06
    settle_s: float = .5
    sample_s: float = 10.
    duration_s: float = 12.
    linear_kp_n_m: float = 1000.
    linear_kd_ns_m: float = 32.
    angular_kp_nm_rad: float = 8.
    angular_kd_nms_rad: float = .3
    linear_effort_limit_n: float = 15.
    angular_effort_limit_nm: float = 2.


class FixtureApproach:
    """Public clock + measured palmar load -> one D6 target depth."""
    def __init__(self, config=FixtureDrive()):
        self.config = config
        self.depth_m = 0.
        self.touched = False
        self.overload_seen = False

    def command(self, time_s, measured_load_n, dt):
        if not np.isfinite([time_s, measured_load_n, dt]).all() or measured_load_n < 0 or dt <= 0:
            raise ValueError('Invalid measured load/control clock')
        cfg = self.config
        self.touched |= measured_load_n >= cfg.contact_latch_n
        self.overload_seen |= measured_load_n > cfg.overload_n
        # Repeated .02 s additions can place the nominal .50 s clock just
        # above the boundary. Do not turn that roundoff into a full early step.
        clock_tolerance_s = 1e-12
        if time_s <= cfg.settle_s + clock_tolerance_s:
            speed = 0.
        elif time_s > cfg.sample_s + clock_tolerance_s or self.overload_seen:
            speed = -cfg.retreat_speed_m_s
        elif self.touched:
            speed = float(np.clip((cfg.target_load_n-measured_load_n)*cfg.force_gain_m_ns,
                                  -cfg.contact_speed_limit_m_s, cfg.contact_speed_limit_m_s))
        else:
            speed = cfg.approach_speed_m_s
        self.depth_m = float(np.clip(self.depth_m+speed*dt, 0., cfg.max_travel_m))
        return self.depth_m


def fixture_hand_frames(outward_direction, center_m, support_rotation, support_center, radius_m):
    """D6 parent/child and actual zero-joint hand transform, using CAD only."""
    outward = np.asarray(outward_direction, dtype=float)
    if outward.shape != (3,) or not np.isfinite(outward).all() or not np.isclose(np.linalg.norm(outward), 1., atol=1e-6):
        raise ValueError('Expected a public unit approach direction')
    inward = -outward
    turn = Rotation.align_vectors(inward[None], np.array([[0., 0., 1.]]))[0]
    palm = np.asarray(center_m, dtype=float)+radius_m*outward
    hand_rotation = turn*support_rotation
    hand_position = palm-hand_rotation.apply(support_center)
    return dict(parent_pose=np.r_[palm, turn.as_quat()],
                child_pose=np.r_[support_center, support_rotation.inv().as_quat()],
                hand_pose=np.r_[hand_position, hand_rotation.as_quat()], inward=inward)


class FixtureTouchScene:
    """One actual touch attempt; construct a fresh scene for independent probes.

    `asset.physics_vertices_m/faces` are physics/label geometry, never supplied to
    FixtureApproach. The object has shape-body -1, and must never index body_q.
    """
    def __init__(self, asset, outward_direction, config=FixtureDrive()):
        import newton
        import warp as wp
        from newton.geometry import HydroelasticSDF
        from sugar_newton.hand.patches import load_hand_mesh, patch_footprints
        from sugar_newton.tactile.reducer import PatchTactile
        from sugar_newton.tactile.field import ContactField
        from .geometry import palmar_support_frame
        from .palmar_coverage import ContinuousPalmarField

        self.newton, self.wp = newton, wp
        self.asset, self.config = asset, config
        self.controller = FixtureApproach(config)
        hand = load_hand_mesh('left')
        support_rotation, support_center = palmar_support_frame(hand, -1.)
        self.frames = fixture_hand_frames(outward_direction, asset.center_m,
                                         support_rotation, support_center, config.start_radius_m)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        cfg = replace(builder.default_shape_cfg, ke=1e4, kd=320., mu=.8, kh=1e10,
                      is_hydroelastic=True, density=0., restitution=0., mu_torsional=0., mu_rolling=0.)
        hv = np.asarray(hand.vertices, np.float32)
        hf = np.asarray(hand.faces, np.int32)
        hand_mesh = newton.Mesh(hv, hf.flatten(), compute_inertia=True)
        raw_inertia = np.asarray(hand_mesh.inertia).reshape(3, 3)
        if hand_mesh.mass <= 0 or not np.isfinite(raw_inertia).all() or np.linalg.eigvalsh(raw_inertia).min() <= 0:
            raise ValueError('Full hand requires positive physical volume/inertia')
        inertia = raw_inertia*config.hand_mass_kg/float(hand_mesh.mass)
        pose = self.frames['hand_pose']
        self.hand_body = builder.add_link(xform=wp.transform(pose[:3], pose[3:]),
            mass=config.hand_mass_kg, com=hand_mesh.com, inertia=wp.mat33(inertia),
            is_kinematic=False, label='dynamic_full_left_hand')
        hand_mesh.build_sdf(max_resolution=128, narrow_band_range=(-.004, .004), margin=.002)
        self.hand_shape = builder.add_shape_mesh(self.hand_body, mesh=hand_mesh, cfg=cfg, label='left_skin')
        dof = builder.JointDofConfig
        axes = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        linear = tuple(dof(axis=a, target_ke=config.linear_kp_n_m, target_kd=config.linear_kd_ns_m,
                           effort_limit=config.linear_effort_limit_n, actuator_mode=newton.JointTargetMode.POSITION)
                       for a in axes)
        angular = tuple(dof(axis=a, target_ke=config.angular_kp_nm_rad, target_kd=config.angular_kd_nms_rad,
                            effort_limit=config.angular_effort_limit_nm, actuator_mode=newton.JointTargetMode.POSITION)
                        for a in axes)
        parent, child = self.frames['parent_pose'], self.frames['child_pose']
        self.drive_joint = builder.add_joint_d6(parent=-1, child=self.hand_body,
            linear_axes=linear, angular_axes=angular,
            parent_xform=wp.transform(parent[:3], parent[3:]),
            child_xform=wp.transform(child[:3], child[3:]), label='touch_fixture_servo')
        builder.add_articulation([self.drive_joint])
        self.target_start = builder.joint_q_start[self.drive_joint]
        object_mesh = newton.Mesh(np.asarray(asset.physics_vertices_m, np.float32),
                                 np.asarray(asset.faces, np.int32).flatten(), compute_inertia=False)
        object_mesh.build_sdf(max_resolution=128, narrow_band_range=(-.006, .006), margin=.004)
        self.object_shape = builder.add_shape_mesh(-1, mesh=object_mesh, cfg=cfg,
            xform=wp.transform(asset.center_m, asset.quaternion_xyzw), label='fixed_full_object')
        builder.add_ground_plane(height=0.)
        self.model = builder.finalize()
        if self.model.body_count != 1 or self.model.shape_body.numpy()[self.object_shape] != -1:
            raise AssertionError('Expected one dynamic hand and a world-fixed full object')
        self.model.request_contact_attributes('force')
        self.pipeline = newton.CollisionPipeline(self.model, contact_matching='latest', contact_report=True,
            sdf_hydroelastic_config=HydroelasticSDF.Config(output_contact_surface=True, buffer_fraction=1., buffer_mult_iso=2))
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(self.model, solver='newton', integrator='implicitfast',
            use_mujoco_contacts=False, cone='elliptic', iterations=100, ls_iterations=50,
            njmax=2048, nconmax=min(1000, self.contacts.rigid_contact_max))
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tactile = PatchTactile(self.model, [self.hand_shape], [self.object_shape])
        raw = ContactField(self.tactile, frame_body=[self.hand_body], pad_footprint=patch_footprints(),
                           pad_offset=[0], palm_sign=[-1.])
        self.field = ContinuousPalmarField(raw, (-1., 1.))
        self.time_s, self.measured_load_n, self.frame = 0., 0., 0
        self.physics_substeps = 0
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        actual = self.state_0.body_q.numpy()[self.hand_body]
        if not np.allclose(actual[:3], pose[:3], atol=1e-6) or abs(np.dot(actual[3:], pose[3:])) < 1-1e-6:
            raise AssertionError('D6 zero configuration differs from public/CAD frame construction')

    def target_translation(self, depth):
        """Original axial target; explicit independent fixtures may override."""
        return [0., 0., depth]

    def step(self, dt=.02, substeps=8):
        if dt <= 0 or substeps <= 0:
            raise ValueError('Positive control interval/substeps required')
        depth = self.controller.command(self.time_s+dt, self.measured_load_n, dt)
        target = self.control.joint_target_q.numpy()
        target[self.target_start:self.target_start+6] = [*self.target_translation(depth), 0., 0., 0.]
        self.control.joint_target_q.assign(target)
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt/substeps)
            self.physics_substeps += 1
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)
        surface = self.pipeline.hydroelastic_sdf.get_contact_surface()
        self.tactile.update(self.state_0, self.contacts, contact_surface=surface)
        self.field.update(self.state_0, surface)
        field = self.field.to_numpy()
        if self.field.overflow_steps:
            raise RuntimeError('Actual contact surface exceeded field capacity; refusing a truncated observation')
        for name in ('pos', 'area', 'pressure', 'traction_vec'):
            if not np.isfinite(field[name]).all():
                raise RuntimeError(f'Nonfinite raw observed contact field: {name}')
        if (field['area'] < 0).any() or (field['pressure'] < 0).any():
            raise RuntimeError('Negative measured area/normal pressure')
        if not (field['patch'] == 0).all() or not ((field['pad'] == -1) | ((field['pad'] >= 0) & (field['pad'] < 27))).all():
            raise RuntimeError('Unexpected single-hand contact identity')
        active = (field['pad'] >= 0) & (field['pressure'] > 0) & (field['area'] > 0)
        self.measured_load_n = float(np.sum(field['pressure'][active]*field['area'][active]))
        self.time_s += dt
        self.frame += 1
        pose = self.state_0.body_q.numpy()[self.hand_body].copy()
        velocity = self.state_0.body_qd.numpy()[self.hand_body].copy()
        if not np.isfinite(pose).all() or not np.isfinite(velocity).all() or not np.isfinite(self.measured_load_n):
            raise RuntimeError('Nonfinite actual dynamic hand/tactile state')
        return dict(time_s=self.time_s, hand_pose_w=pose, hand_velocity_w=velocity,
                    measured_palmar_load_n=self.measured_load_n,
                    validation_full_hand_load_n=float(self.tactile.normal_load.numpy()[0]),
                    target_depth_m=depth, touched=self.controller.touched,
                    overload_seen=self.controller.overload_seen, field=field)
