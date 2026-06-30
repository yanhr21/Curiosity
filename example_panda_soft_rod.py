# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Panda Hydro
#
# Shows how to set up a pick-and-place manipulation simulation using a
# Franka Panda arm with SDF hydroelastic contacts and inverse kinematics.
# Supports different scene configurations: pen or cube.
#
# Command: python -m newton.examples panda_hydro --scene pen --world-count 1
#
###########################################################################

import copy
from dataclasses import replace
from enum import Enum

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverVBD

# --- Soft (FEM) rod parameters (the ONLY object change vs example_robot_panda_hydro) ---
# The rigid pencil capsule (radius 0.005, length 0.14, along world X) becomes a soft FEM
# rod of the same footprint, simulated by SolverVBD coupled one-way to the MuJoCo arm.
E_ROD, NU_ROD = 2.0e7, 0.45  # Young's modulus [Pa] / Poisson ratio; higher = stiffer/less bend
ROD_DENSITY = 200.0
ROD_NX, ROD_CX = 20, 0.007  # 20 * 0.007 = 0.14 m length (along world X)
ROD_NT, ROD_CT = 2, 0.005  # 2 * 0.005 = 0.010 m thick (each transverse axis)
ROD_PRAD = 0.003
# Gripper finger positions [m]: open / closed-on-rod. The soft rod cannot push the rigid
# fingers back (one-way coupling), so closing fully (0.0) would crush it — close to a
# rod-thickness gap instead.
GRIP_OPEN = 0.04
GRIP_CLOSE = 0.016  # closed finger pos [m]: grips a ~1 cm rod; closing further jams the pads
VBD_ITERS = 50  # VBD iterations/substep; stiff FEM needs many to converge (else it acts soft)


def lame(E, nu):
    nu = max(-0.999, min(nu, 0.499))
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def cylinder_tetmesh(radius, length, n_ring=12, n_axial=24):
    """Tetrahedral mesh of a ROUND cylinder (axis along local X, centered at origin) with
    the given radius/length — i.e. the original pencil capsule's geometry, tetrahedralized
    so SolverVBD can deform it. Structured center-fan prisms split into 3 tets (no slivers);
    each tet is reordered to positive volume so the FEM rest state is valid.
    Returns (vertices: list[wp.vec3], tet_indices: flat list[int])."""
    xs = np.linspace(-length / 2.0, length / 2.0, n_axial)
    pps = n_ring + 1  # points per cross-section: 1 center + n_ring boundary points
    verts = []
    for x in xs:
        verts.append((float(x), 0.0, 0.0))
        for k in range(n_ring):
            # +0.5 phase puts a flat facet (not a vertex) at the bottom so the rod rests on
            # a small flat and does not roll on the table before being grasped.
            a = 2.0 * np.pi * (k + 0.5) / n_ring
            verts.append((float(x), radius * np.cos(a), radius * np.sin(a)))
    V = np.asarray(verts)

    def add_tet(out, i, j, k, l):
        p = V[[i, j, k, l]]
        if np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])) < 0.0:
            k, l = l, k  # flip to positive signed volume
        out.extend((i, j, k, l))

    tets = []
    for s in range(n_axial - 1):
        b0, t0 = s * pps, (s + 1) * pps
        for k in range(n_ring):
            k2 = (k + 1) % n_ring
            a, b, c = b0, b0 + 1 + k, b0 + 1 + k2  # bottom triangle (center, ring, ring+1)
            ap, bp, cp = t0, t0 + 1 + k, t0 + 1 + k2  # top triangle
            add_tet(tets, a, b, c, ap)
            add_tet(tets, b, c, ap, bp)
            add_tet(tets, c, ap, bp, cp)
    return [wp.vec3(*v) for v in verts], tets


class SceneType(Enum):
    PEN = "pen"
    CUBE = "cube"


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    """Convert a quaternion to a vec4."""
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array2d[wp.float32],
    gripper_value: float,
):
    world_idx = wp.tid()
    for j in range(7):
        joint_targets[world_idx, j] = ik_solution[0, j]
    joint_targets[world_idx, 7] = gripper_value
    joint_targets[world_idx, 8] = gripper_value


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.scene = SceneType(args.scene)
        self.test_mode = args.test
        self.show_isosurface = False  # Disabled by default for performance
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.collide_substeps = 2  # run collision detection every X simulation steps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = args.world_count
        self.viewer = viewer

        sdf_max_resolution = 64
        sdf_narrow_band_range = (-0.01, 0.01)

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            gap=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        # meshes need explicit call to build_sdf with sdf parameters, while primitive sdf are configured directly via shape config flags
        shape_cfg_meshes = replace(shape_cfg, is_hydroelastic=True)
        shape_cfg_primitives = replace(
            shape_cfg,
            is_hydroelastic=True,
            sdf_max_resolution=sdf_max_resolution,
            sdf_narrow_band_range=sdf_narrow_band_range,
        )

        builder = newton.ModelBuilder()
        # URDF mesh colliders are imported as plain meshes; keep hydroelastic disabled
        # for import-time shapes unless they provide explicit mesh.sdf payloads.
        builder.default_shape_cfg = shape_cfg

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform((-0.5, -0.5, 0.05), wp.quat_identity()),
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )

        def find_body(name):
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        # Set SDF collisions on panda hand and fingers for hydroelastic contact
        finger_body_indices = {
            find_body("fr3_leftfinger"),
            find_body("fr3_rightfinger"),
            find_body("fr3_hand"),
        }
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx in finger_body_indices and builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        # Hydroelastic mesh SDFs must be scale-baked for non-unit shape scale.
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        builder.shape_source[shape_idx] = mesh
                        builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=sdf_max_resolution,
                        narrow_band_range=sdf_narrow_band_range,
                        margin=shape_cfg.gap,
                    )
                builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
            elif body_idx not in finger_body_indices:
                non_finger_shape_indices.append(shape_idx)

        # Convert non-finger shapes to convex hulls
        builder.approximate_meshes(
            method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        )

        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[:9] = [*init_q, 0.05, 0.05]
        builder.joint_target_q[:9] = [*init_q, 1.0, 1.0]

        builder.joint_target_ke[:9] = [650.0] * 9
        builder.joint_target_kd[:9] = [100.0] * 9
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [20.0] * 2
        builder.joint_armature[:7] = [0.1] * 7
        builder.joint_armature[7:9] = [0.5] * 2

        # Add gripper pads
        left_finger_idx = find_body("fr3_leftfinger")
        right_finger_idx = find_body("fr3_rightfinger")

        pad_asset_path = newton.utils.download_asset("manipulation_objects/pad")
        pad_stage = Usd.Stage.Open(str(pad_asset_path / "model.usda"))
        pad_mesh = newton.usd.get_mesh(
            pad_stage.GetPrimAtPath("/root/Model/Model"),
            load_normals=True,
            face_varying_normal_conversion="vertex_splitting",
        )
        pad_scale = np.asarray(newton.usd.get_scale(pad_stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
        if not np.allclose(pad_scale, 1.0):
            # Hydroelastic mesh SDFs must be scale-baked for non-unit shape scale.
            pad_mesh = pad_mesh.copy(vertices=pad_mesh.vertices * pad_scale, recompute_inertia=True)
        pad_mesh.build_sdf(
            max_resolution=sdf_max_resolution,
            narrow_band_range=sdf_narrow_band_range,
            margin=shape_cfg.gap,
        )
        pad_xform = wp.transform(
            wp.vec3(0.0, 0.005, 0.045),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi),
        )
        builder.add_shape_mesh(body=left_finger_idx, mesh=pad_mesh, xform=pad_xform, cfg=shape_cfg_meshes)
        builder.add_shape_mesh(body=right_finger_idx, mesh=pad_mesh, xform=pad_xform, cfg=shape_cfg_meshes)

        # Table
        box_size = 0.05
        table_half_extents = (box_size * 2, box_size * 2, box_size)  # half-extents
        table_mesh = newton.Mesh.create_box(
            table_half_extents[0],
            table_half_extents[1],
            table_half_extents[2],
            duplicate_vertices=True,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=True,
        )
        table_mesh.build_sdf(
            max_resolution=sdf_max_resolution,
            narrow_band_range=sdf_narrow_band_range,
            margin=shape_cfg.gap,
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(wp.vec3(0.08, -0.5, box_size), wp.quat_identity()),
            cfg=shape_cfg_meshes,
        )

        # Object to manipulate
        self.put_in_cup = True

        if self.scene == SceneType.PEN:
            radius = 0.005
            length = 0.14
            self.object_pos = [0.0, -0.5, 2 * box_size + radius + 0.001]
            # SOFT FEM pencil: the SAME round capsule geometry (radius, length) as the
            # original rigid pencil, tetrahedralized so SolverVBD can deform it (Newton has
            # no soft primitive). Stepped by VBD instead of being a rigid body.
            k_mu, k_lambda = lame(E_ROD, NU_ROD)
            verts, tets = cylinder_tetmesh(radius, length)
            builder.default_particle_radius = ROD_PRAD
            self.rod_p0 = builder.particle_count
            builder.add_soft_mesh(
                # spawn at rest height (bottom particle surfaces just touching the table top)
                # so the rod neither launches from penetration nor bounces and rolls away
                pos=wp.vec3(self.object_pos[0], self.object_pos[1], 2 * box_size + radius + ROD_PRAD),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=verts,
                indices=tets,
                density=ROD_DENSITY,
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=1.0,
                particle_radius=ROD_PRAD,
            )
            self.object_body_local = None
            self.grasping_offset = [-0.03, 0.0, 0.12]  # reach slightly lower than the original 0.13 to grab the rod
            self.place_offset = 0.01

        elif self.scene == SceneType.CUBE:
            size = 0.04
            self.object_pos = [0.0, -0.5, 2 * box_size + 0.5 * size]
            object_xform = wp.transform(wp.vec3(self.object_pos), wp.quat_identity())
            self.object_body_local = builder.add_body(xform=object_xform, label="object")
            builder.add_shape_box(
                body=self.object_body_local, hx=size / 2, hy=size / 2, hz=size / 2, cfg=shape_cfg_primitives
            )
            self.grasping_offset = [0.03, 0.0, 0.14]
            self.place_offset = 0.0

        if self.put_in_cup:
            self.cup_pos = [0.13, -0.5, box_size + 0.1]

            cup_asset_path = newton.utils.download_asset("manipulation_objects/cup")
            cup_stage = Usd.Stage.Open(str(cup_asset_path / "model.usda"))
            prim = cup_stage.GetPrimAtPath("/root/Model/Model")
            cup_mesh = newton.usd.get_mesh(prim, load_normals=True, face_varying_normal_conversion="vertex_splitting")
            cup_scale = np.asarray(newton.usd.get_scale(cup_stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
            if not np.allclose(cup_scale, 1.0):
                # Hydroelastic mesh SDFs must be scale-baked for non-unit shape scale.
                cup_mesh = cup_mesh.copy(vertices=cup_mesh.vertices * cup_scale, recompute_inertia=True)
            cup_mesh.build_sdf(
                max_resolution=sdf_max_resolution,
                narrow_band_range=sdf_narrow_band_range,
                margin=shape_cfg.gap,
            )
            cup_xform = wp.transform(
                wp.vec3(self.cup_pos),
                wp.quat_identity(),
            )
            cup_body = builder.add_body(label="cup", xform=cup_xform)
            builder.add_shape_mesh(body=cup_body, mesh=cup_mesh, cfg=shape_cfg_meshes)

        # build model for IK
        self.model_single = copy.deepcopy(builder).finalize()

        # Store bodies per world before replication
        self.bodies_per_world = builder.body_count

        scene = newton.ModelBuilder()
        scene.replicate(builder, self.world_count)
        scene.add_ground_plane(cfg=shape_cfg)

        # Soft-body coupling prep: let every rigid shape (pads, table, cup, ground, arm)
        # collide with the rod's particles, and color the particle graph — both required
        # by SolverVBD and both must happen before finalize().
        for i in range(len(scene.shape_flags)):
            scene.shape_flags[i] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        scene.color()

        self.model = scene.finalize()

        # num_hydroelastic = (self.model.shape_flags.numpy() & newton.ShapeFlags.HYDROELASTIC).astype(bool).sum()
        # print(f"Number of hydroelastic shapes: {num_hydroelastic} / {self.model.shape_count}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline with SDF hydroelastic config
        # Enable output_contact_surface so the kernel code is compiled (allows runtime toggle)
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            output_contact_surface=hasattr(viewer, "renderer"),  # Compile in if viewer supports it
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Create MuJoCo solver with Newton contacts
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        # SolverVBD for the soft rod, one-way coupled to the MuJoCo arm: VBD reads the arm
        # body poses/velocities (integrate_with_external_rigid_solver) to apply contact +
        # friction to the rod particles, but never moves the rigid bodies. Params mirror
        # example_softbody_franka. Only the global soft-contact params are set, so the
        # rigid arm/table/cup contact materials (and MuJoCo dynamics) are left untouched.
        self.soft_solver = SolverVBD(
            self.model,
            iterations=VBD_ITERS,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.004,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )
        self.model.soft_contact_ke = 2.0e6
        self.model.soft_contact_kd = 2.0e-1
        self.model.soft_contact_mu = 1.0

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False  # Disable interactive picking for this example
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)
            self.viewer.set_world_offsets(wp.vec3(1.0, 1.0, 0.0))
            self.viewer.show_hydro_contact_surface = self.show_isosurface
            self.viewer.register_ui_callback(self.render_ui, position="side")

        # Initialize state for IK setup
        self.state = self.model_single.state()
        newton.eval_fk(self.model_single, self.model.joint_q, self.model.joint_qd, self.state)

        self.setup_ik()
        self.control = self.model.control()
        self.joint_target_shape = self.control.joint_target_q.reshape((self.world_count, -1)).shape
        self.joint_targets_2d = wp.zeros(self.joint_target_shape, dtype=wp.float32)
        wp.copy(self.control.joint_target_q[:9], self.model.joint_q[:9])

        # Track maximum object height for testing (only in test mode)
        self.object_max_z = [self.object_pos[2]] * self.world_count if self.test_mode else None

        self.capture()
        self.capture_ik()

    def set_joint_targets(self):
        self.time_in_waypoint += self.frame_dt

        # interpolate between waypoints
        t = self.time_in_waypoint / self.waypoints[self.current_waypoint][1]
        next_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
        target_position = self.waypoints[self.current_waypoint][0] * (1 - t) + self.waypoints[next_waypoint][0] * t

        target_angle_z = self.waypoints[self.current_waypoint][-1] * (1 - t) + self.waypoints[next_waypoint][-1] * t
        target_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
        target_rotation = wp.mul(target_rotation, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), target_angle_z))

        self.pos_obj.set_target_positions(wp.array([target_position], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(target_rotation)], dtype=wp.vec4))

        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # Broadcast single IK solution to all worlds
        t_gripper = self.waypoints[self.current_waypoint][2] * (1 - t) + self.waypoints[next_waypoint][2] * t
        # Soft rod cannot push the rigid fingers back (one-way coupling), so close to a
        # rod-thickness gap (GRIP_CLOSE) rather than 0.0 to grip without crushing. The EE
        # trajectory/timing is unchanged — only the closed finger width differs.
        gripper_value = GRIP_OPEN * (1.0 - t_gripper) + GRIP_CLOSE * t_gripper
        wp.launch(
            broadcast_ik_solution_kernel,
            dim=self.world_count,
            inputs=[self.joint_q_ik, self.joint_targets_2d, gripper_value],
        )
        wp.copy(self.control.joint_target_q, self.joint_targets_2d.flatten())

        if self.time_in_waypoint >= self.waypoints[self.current_waypoint][1]:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            self.time_in_waypoint = 0.0

    def capture_ik(self):
        self.graph_ik = None
        with wp.ScopedCapture() as capture:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        self.graph_ik = capture.graph

    def capture(self):
        # Run simulate() eagerly. SolverVBD's per-frame BVH rebuild and soft-contact
        # allocation do not CUDA-graph-capture cleanly; offline render speed is fine.
        self.graph = None

    def simulate(self):
        self.soft_solver.rebuild_bvh(self.state_0)
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            # MuJoCo integrates the arm (it ignores particles); VBD then integrates the rod,
            # reading the arm's freshly-updated body velocities for contact friction.
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.soft_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.set_joint_targets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Track maximum object height for testing (only in test mode)
        if self.test_mode:
            if self.object_body_local is None:  # soft rod: track the particles' mean z
                rod_z = float(self.state_0.particle_q.numpy()[self.rod_p0 :, 2].mean())
                for world_idx in range(self.world_count):
                    self.object_max_z[world_idx] = max(self.object_max_z[world_idx], rod_z)
            else:
                body_q = self.state_0.body_q.numpy()
                for world_idx in range(self.world_count):
                    object_body_idx = world_idx * self.bodies_per_world + self.object_body_local
                    z_pos = float(body_q[object_body_idx][2])
                    self.object_max_z[world_idx] = max(self.object_max_z[world_idx], z_pos)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        # Always call log_hydro_contact_surface - it handles show_hydro_contact_surface internally
        # and will clear the lines when disabled
        self.viewer.log_hydro_contact_surface(
            (
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                if self.collision_pipeline.hydroelastic_sdf is not None
                else None
            ),
            penetrating_only=True,
        )
        self.viewer.end_frame()

    def render_ui(self, imgui):
        changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_isosurface

    def test_final(self):
        # Verify that the object was picked up by checking the maximum height reached
        initial_z = self.object_pos[2]
        min_lift_height = 0.15  # Object should be lifted at least 15cm above initial position

        for world_idx in range(self.world_count):
            max_z = self.object_max_z[world_idx]
            max_lift = max_z - initial_z

            assert max_lift > min_lift_height, (
                f"World {world_idx}: Object was not picked up high enough. "
                f"Initial z={initial_z:.3f}, max z reached={max_z:.3f}, "
                f"max lift={max_lift:.3f} (expected > {min_lift_height})"
            )

        # In-cup placement check disabled — see newton-physics/newton#1337.
        # Hydroelastic contact ordering on GPU still occasionally lets the pen
        # slip out of the gripper during transport, producing both small drifts
        # and complete misses. Lift-height check above remains as a coarse
        # pickup verification.
        # # Verify that the object ended up in the cup
        # if self.put_in_cup:
        #     body_q = self.state_0.body_q.numpy()
        #     cup_x, cup_y, cup_z = self.cup_pos
        #     tolerance_xy = 0.05
        #     min_z = cup_z - 0.05
        #
        #     for world_idx in range(self.world_count):
        #         object_body_idx = world_idx * self.bodies_per_world + self.object_body_local
        #         x, y, z = body_q[object_body_idx][:3]
        #         assert abs(x - cup_x) < tolerance_xy and abs(y - cup_y) < tolerance_xy and z > min_z, (
        #             f"World {world_idx}: Object is not in the cup. "
        #             f"Object pos=({x:.3f}, {y:.3f}, {z:.3f}), "
        #             f"cup pos=({cup_x:.3f}, {cup_y:.3f}, {cup_z:.3f})"
        #         )

    def setup_ik(self):
        self.ee_index = 10
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        # Position objective (single IK problem)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),
        )

        # Rotation objective (single IK problem)
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([quat_to_vec4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
        )

        # Joint limit objective
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_single.joint_limit_lower,
            joint_limit_upper=self.model_single.joint_limit_upper,
        )

        # Variables the solver will update (single IK problem)
        self.joint_q_ik = wp.array(self.model_single.joint_q, shape=(1, self.model_single.joint_coord_count))

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.time_in_waypoint = 0.0
        self.current_waypoint = 0
        self.z_rest = 0.5
        grasping_pos = wp.vec3(self.object_pos) + wp.vec3(self.grasping_offset)
        resting_pos = wp.vec3([grasping_pos[0], grasping_pos[1], self.z_rest])
        grasp_pos = 1.0
        no_grasp_pos = 0.0
        rot_hand = 0.0
        self.waypoints = [
            [resting_pos, 1.0, no_grasp_pos, rot_hand],
            [grasping_pos, 1.0, no_grasp_pos, rot_hand],
            [grasping_pos, 1.0, grasp_pos, rot_hand],
            [resting_pos, 1.0, grasp_pos, rot_hand],
        ]

        if self.put_in_cup:
            loose_pos = 0.69
            wps = []
            cup_pos_higher = wp.vec3([self.cup_pos[0] + self.place_offset, self.cup_pos[1], self.z_rest])
            cup_pos_lower = wp.vec3([self.cup_pos[0] + self.place_offset, self.cup_pos[1], self.z_rest - 0.1])
            wps.extend(
                [
                    [cup_pos_higher, 2.0, grasp_pos, rot_hand],
                    [cup_pos_higher, 0.25, loose_pos, rot_hand],
                    [cup_pos_higher, 1.0, grasp_pos, rot_hand],
                    [cup_pos_lower, 1.0, grasp_pos, rot_hand],
                    [cup_pos_lower, 1.0, no_grasp_pos, rot_hand],
                ]
            )
            self.waypoints.extend(wps)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=720)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--scene",
            type=str,
            choices=[scene.value for scene in SceneType],
            default=SceneType.PEN.value,
            help="Scene type to load (pen, cube)",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
