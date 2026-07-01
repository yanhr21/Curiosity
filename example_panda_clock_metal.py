# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Panda Hydro — metal clock pick-and-place
#
# Fork of newton/examples/robot/example_robot_panda_hydro.py. Everything
# (Franka arm, hydroelastic pads, table, cup, IK waypoints, MuJoCo solver,
# camera, colors) is byte-for-byte the original scene; ONLY the manipulated
# object changes: the rigid capsule "pen" is replaced by the TRELLIS.2-
# generated clock mesh (pipeline_out/object.glb), given a metal material.
#
# The clock is loaded as a rigid body: convex-hull collision (hydroelastic
# SDF, robust inertia) + the full generated mesh as a visual overlay. It is
# assigned a metal material — friction + restitution + a density DERIVED from
# a common-sense weight and the mesh volume (density = mass / volume).
#
# Command: python example_panda_clock_metal.py --viewer null --headless
###########################################################################

import copy
from dataclasses import replace
from enum import Enum

import numpy as np
import trimesh
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton.geometry import HydroelasticSDF


class SceneType(Enum):
    CLOCK = "clock"
    PEN = "pen"
    CUBE = "cube"


# Metal material (verified reasoning: rigid materials share a rigid kh and are
# distinguished by friction / restitution / density, not by hydroelastic stiffness).
METAL_MU = 0.5  # dry metal-on-pad friction
METAL_RESTITUTION = 0.3  # metal is springy (note: MuJoCo largely ignores restitution)
METAL_KH = 1.0e12  # hard/rigid hydroelastic contact stiffness


def load_clock_geometry(glb_path: str, scale: float):
    """Load the generated GLB and return Newton-frame arrays for the clock.

    Rotates the GLB (Y-up) to Newton (Z-up) so the clock stands upright, scales
    it to physical size, and centers it. Returns the convex hull (collision) and
    the full mesh (visual) plus the hull volume [m^3] for density derivation.
    """
    scene = trimesh.load(glb_path)
    mesh = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    v = np.asarray(mesh.vertices, dtype=np.float64)

    # GLB is Y-up; Newton is Z-up. Rotate +90 deg about X: (x, y, z) -> (x, -z, y).
    rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    v = (rot @ v.T).T * float(scale)
    # center on the XY bbox midpoint and on the Z bbox midpoint
    center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    v = v - center

    faces = np.asarray(mesh.faces, dtype=np.int32)
    half_extents = 0.5 * (v.max(axis=0) - v.min(axis=0))

    hull = trimesh.Trimesh(vertices=v, faces=faces).convex_hull
    hull_v = np.asarray(hull.vertices, dtype=np.float32)
    hull_f = np.asarray(hull.faces, dtype=np.int32).flatten()
    hull_volume = float(hull.volume)  # m^3

    return {
        "viz_v": v.astype(np.float32),
        "viz_f": faces.flatten(),
        "hull_v": hull_v,
        "hull_f": hull_f,
        "hull_volume": hull_volume,
        "half_extents": half_extents,
    }


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
        # Compliant pads (real gripper pads deform and conform to the object). A softer
        # pad kh -> broader, more realistic contact patch against the rigid clock.
        pad_cfg = replace(shape_cfg_meshes, kh=args.pad_kh)
        builder.add_shape_mesh(body=left_finger_idx, mesh=pad_mesh, xform=pad_xform, cfg=pad_cfg)
        builder.add_shape_mesh(body=right_finger_idx, mesh=pad_mesh, xform=pad_xform, cfg=pad_cfg)

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

        if self.scene == SceneType.CLOCK:
            clock = load_clock_geometry(args.clock_glb, args.clock_scale)
            hx, hy, hz = (float(e) for e in clock["half_extents"])
            # metal density derived from a common-sense weight and the mesh volume
            density = args.clock_mass / clock["hull_volume"]
            print(
                f"[clock] extents (m) = {2 * hx:.3f} x {2 * hy:.3f} x {2 * hz:.3f} | "
                f"hull_vol = {clock['hull_volume'] * 1e6:.1f} cm^3 | "
                f"mass = {args.clock_mass:.3f} kg -> density = {density:.0f} kg/m^3"
            )

            # collision hull is not drawn (is_visible=False); the full mesh below is the visual
            metal_cfg = replace(
                shape_cfg_meshes,
                density=density,
                mu=METAL_MU,
                restitution=METAL_RESTITUTION,
                kh=METAL_KH,
                is_visible=False,
            )

            self.object_pos = [0.0, -0.5, 2 * box_size + hz + 0.001]
            object_xform = wp.transform(wp.vec3(self.object_pos), wp.quat_identity())
            self.object_body_local = builder.add_body(xform=object_xform, label="object")

            # collision: convex hull (watertight -> robust SDF + inertia), hydroelastic metal
            hull_mesh = newton.Mesh(clock["hull_v"], clock["hull_f"], compute_inertia=True)
            hull_mesh.build_sdf(
                max_resolution=sdf_max_resolution,
                narrow_band_range=sdf_narrow_band_range,
                margin=shape_cfg.gap,
            )
            hull_shape = builder.add_shape_mesh(body=self.object_body_local, mesh=hull_mesh, cfg=metal_cfg)
            builder.shape_flags[hull_shape] |= newton.ShapeFlags.HYDROELASTIC

            # visual overlay: the full generated clock mesh, non-colliding, no mass, metal-gray
            viz_cfg = replace(shape_cfg, has_shape_collision=False, is_hydroelastic=False, density=0.0)
            viz_mesh = newton.Mesh(clock["viz_v"], clock["viz_f"], compute_inertia=False)
            builder.add_shape_mesh(body=self.object_body_local, mesh=viz_mesh, cfg=viz_cfg, color=(0.62, 0.64, 0.67))

            # grip across the thin (Y) axis at the clock's mid-height, from above.
            # grip_dx centers the finger line over the clock (the hand link origin is
            # offset ~3 cm in x from the finger centerline, as in the cube scene).
            self.grasping_offset = [args.grip_dx, 0.0, args.grip_dz]
            self.place_offset = 0.0

        elif self.scene == SceneType.PEN:
            radius = 0.005
            length = 0.14
            self.object_pos = [0.0, -0.5, 2 * box_size + radius + 0.001]
            object_xform = wp.transform(
                wp.vec3(self.object_pos),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2),
            )
            self.object_body_local = builder.add_body(xform=object_xform, label="object")
            builder.add_shape_capsule(
                body=self.object_body_local, radius=radius, half_height=length / 2, cfg=shape_cfg_primitives
            )
            self.grasping_offset = [-0.03, 0.0, 0.13]
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

        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline with SDF hydroelastic config
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
        gripper_value = 0.06 * (1 - t_gripper)
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
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
            body_q = self.state_0.body_q.numpy()
            for world_idx in range(self.world_count):
                object_body_idx = world_idx * self.bodies_per_world + self.object_body_local
                z_pos = float(body_q[object_body_idx][2])
                self.object_max_z[world_idx] = max(self.object_max_z[world_idx], z_pos)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
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

    def object_pose(self):
        """Return current object (x, y, z) in world 0."""
        body_q = self.state_0.body_q.numpy()
        return tuple(float(c) for c in body_q[self.object_body_local][:3])

    def test_final(self):
        # Verify the clock was picked up by checking the maximum height reached.
        initial_z = self.object_pos[2]
        min_lift_height = 0.10  # clock should be lifted at least 10 cm above start

        for world_idx in range(self.world_count):
            max_z = self.object_max_z[world_idx]
            max_lift = max_z - initial_z
            assert max_lift > min_lift_height, (
                f"World {world_idx}: clock was not picked up high enough. "
                f"Initial z={initial_z:.3f}, max z reached={max_z:.3f}, max lift={max_lift:.3f}"
            )

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
            # Release from high above the cup WITHOUT lowering the gripper: carry the
            # clock over the cup at rest height, hold, then open the jaws so it drops in.
            cup_pos_higher = wp.vec3([self.cup_pos[0] + self.place_offset, self.cup_pos[1], self.z_rest])
            self.waypoints.extend(
                [
                    [cup_pos_higher, 2.0, grasp_pos, rot_hand],  # carry over the cup, held high
                    [cup_pos_higher, 1.0, grasp_pos, rot_hand],  # settle above the cup
                    [cup_pos_higher, 0.5, no_grasp_pos, rot_hand],  # release (open) from high above
                    [cup_pos_higher, 2.0, no_grasp_pos, rot_hand],  # hold; clock free-falls into the cup
                ]
            )

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
            default=SceneType.CLOCK.value,
            help="Scene type to load (clock, pen, cube)",
        )
        parser.add_argument(
            "--clock-glb",
            type=str,
            default="pipeline_out/object.glb",
            help="Path to the TRELLIS.2-generated clock GLB.",
        )
        parser.add_argument(
            "--clock-mass", type=float, default=0.5, help="Clock mass [kg] (metal); density is derived from this."
        )
        parser.add_argument(
            "--clock-scale", type=float, default=0.10, help="Uniform scale from GLB units to meters (0.10 = 2x)."
        )
        parser.add_argument("--grip-dx", type=float, default=0.03, help="EE x-offset from clock center for grasp.")
        parser.add_argument("--grip-dz", type=float, default=0.14, help="EE z-offset above clock center for grasp.")
        parser.add_argument(
            "--pad-kh",
            type=float,
            default=1e11,
            help="Gripper-pad hydroelastic stiffness. Lower = more compliant pad = broader contact patch.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
