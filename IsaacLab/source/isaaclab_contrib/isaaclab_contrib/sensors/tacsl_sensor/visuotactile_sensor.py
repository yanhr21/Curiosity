# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import Camera, TiledCamera
from isaaclab.sensors.sensor_base import SensorBase

from .visuotactile_render import GelsightRender
from .visuotactile_sensor_data import VisuoTactileSensorData

if TYPE_CHECKING:
    from .visuotactile_sensor_cfg import VisuoTactileSensorCfg

import trimesh

logger = logging.getLogger(__name__)


class VisuoTactileSensor(SensorBase):
    r"""A tactile sensor for both camera-based and force field tactile sensing.

    This sensor provides:
    1. Camera-based tactile sensing: depth images from tactile surface
    2. Force field tactile sensing: Penalty-based normal and shear forces using SDF queries

    The sensor can be configured to use either or both sensing modalities.

    **Computation Pipeline:**
        Camera-based sensing computes depth differences from a nominal (no-contact) baseline and
        processes them through the tac-sl GelSight renderer to produce realistic tactile images.

        Force field sensing queries Signed Distance Fields (SDF) to compute penetration depths,
        then applies penalty-based spring-damper models
        (:math:`F_n = k_n \cdot \text{depth}`, :math:`F_t = \min(k_t \cdot \|v_t\|, \mu \cdot F_n)`)
        to compute normal and shear forces at discrete tactile points.

    **Example Usage:**
        For a complete working example, see: ``scripts/demos/sensors/tacsl/tacsl_example.py``

    **Current Limitations:**
        - SDF collision meshes must be pre-computed and objects specified before simulation starts
        - Force field computation requires specific rigid body and mesh configurations
        - No support for dynamic addition/removal of interacting objects during runtime

    Configuration Requirements:
        The following requirements must be satisfied for proper sensor operation:

        **Camera Tactile Imaging**
            If ``enable_camera_tactile=True``, a valid ``camera_cfg`` (TiledCameraCfg) must be
            provided with appropriate camera parameters.

        **Force Field Computation**
            If ``enable_force_field=True``, the following parameters are required:

            * ``contact_object_prim_path_expr`` - Prim path expression to find the contact object prim

        **SDF Computation**
            When force field computation is enabled, penalty-based normal and shear forces are
            computed using Signed Distance Field (SDF) queries. To achieve GPU acceleration:

            * Interacting objects should have pre-computed SDF collision meshes
            * An SDFView must be defined during initialization, therefore interacting objects
              should be specified before simulation.

    """

    cfg: VisuoTactileSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: VisuoTactileSensorCfg):
        """Initializes the tactile sensor object.

        Args:
            cfg: The configuration parameters.
        """

        # Create empty variables for storing output data
        self._data: VisuoTactileSensorData = VisuoTactileSensorData()

        # Camera-based tactile sensing
        self._camera_sensor: Camera | TiledCamera | None = None
        self._nominal_tactile: dict | None = None

        # Force field tactile sensing
        self._tactile_pos_local: torch.Tensor | None = None
        self._tactile_quat_local: torch.Tensor | None = None
        self._sdf_object: Any | None = None

        # COMs for velocity correction
        self._elastomer_com_b: torch.Tensor | None = None
        self._contact_object_com_b: torch.Tensor | None = None

        # Physics views
        self._physics_sim_view = None
        self._elastomer_body_view = None
        self._elastomer_tip_view = None
        self._contact_object_body_view = None

        # Visualization
        self._tactile_visualizer: VisualizationMarkers | None = None

        # Tactile points count
        self.num_tactile_points: int = 0

        # Now call parent class constructor
        super().__init__(cfg)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        if self._camera_sensor is not None:
            self._camera_sensor.__del__()
        # unsubscribe from callbacks
        super().__del__()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Tactile sensor @ '{self.cfg.prim_path}': \n"
            f"\trender config     : {self.cfg.render_cfg.base_data_path}/{self.cfg.render_cfg.sensor_data_dir_name}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tcamera enabled    : {self.cfg.enable_camera_tactile}\n"
            f"\tforce field enabled: {self.cfg.enable_force_field}\n"
            f"\tnum instances     : {self.num_instances}\n"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._num_envs

    @property
    def data(self) -> VisuoTactileSensorData:
        # Update sensors if needed
        self._update_outdated_buffers()
        # Return the data
        return self._data

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals."""
        # reset the timestamps
        super().reset(env_ids)

        # Reset camera sensor if enabled
        if self._camera_sensor:
            self._camera_sensor.reset(env_ids)

    """
    Implementation
    """

    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        super()._initialize_impl()

        # Obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Initialize camera-based tactile sensing
        if self.cfg.enable_camera_tactile:
            self._initialize_camera_tactile()

        # Initialize force field tactile sensing
        if self.cfg.enable_force_field:
            self._initialize_force_field()

        # Initialize visualization
        if self.cfg.debug_vis:
            self._initialize_visualization()

    def get_initial_render(self) -> dict | None:
        """Get the initial tactile sensor render for baseline comparison.

        This method captures the initial state of the tactile sensor when no contact
        is occurring. This baseline is used for computing relative changes during
        tactile interactions.

        .. warning::
            It is the user's responsibility to ensure that the sensor is in a "no contact" state
            when this method is called. If the sensor is in contact with an object, the baseline
            will be incorrect, leading to erroneous tactile readings.

        Returns:
            dict | None: Dictionary containing initial render data with sensor output keys
                        and corresponding tensor values. Returns None if camera tactile
                        sensing is disabled.

        Raises:
            RuntimeError: If camera sensor is not initialized or initial render fails.
        """
        if not self.cfg.enable_camera_tactile:
            return None

        self._camera_sensor.update(dt=0.0)

        # get the initial render
        initial_render = self._camera_sensor.data.output
        if initial_render is None:
            raise RuntimeError("Initial render is None")

        # Store the initial nominal tactile data
        self._nominal_tactile = dict()
        for key, value in initial_render.items():
            self._nominal_tactile[key] = value.clone()

        return self._nominal_tactile

    def _initialize_camera_tactile(self):
        """Initialize camera-based tactile sensing."""
        if self.cfg.camera_cfg is None:
            raise ValueError("Camera configuration is None. Please provide a valid camera configuration.")
        # check image size is consistent with the render config
        if (
            self.cfg.camera_cfg.height != self.cfg.render_cfg.image_height
            or self.cfg.camera_cfg.width != self.cfg.render_cfg.image_width
        ):
            raise ValueError(
                "Camera configuration image size is not consistent with the render config. Camera size:"
                f" {self.cfg.camera_cfg.height}x{self.cfg.camera_cfg.width}, Render config:"
                f" {self.cfg.render_cfg.image_height}x{self.cfg.render_cfg.image_width}"
            )
        # check data types
        if not all(data_type in ["distance_to_image_plane", "depth"] for data_type in self.cfg.camera_cfg.data_types):
            raise ValueError(
                f"Camera configuration data types are not supported. Data types: {self.cfg.camera_cfg.data_types}"
            )
        if self.cfg.camera_cfg.update_period != self.cfg.update_period:
            logger.warning(
                f"Camera configuration update period ({self.cfg.camera_cfg.update_period}) is not equal to sensor"
                f" update period ({self.cfg.update_period}), changing camera update period to match sensor update"
                " period"
            )
            self.cfg.camera_cfg.update_period = self.cfg.update_period

        # gelsightRender
        self._tactile_rgb_render = GelsightRender(self.cfg.render_cfg, device=self.device)

        # Create camera sensor
        self._camera_sensor = TiledCamera(self.cfg.camera_cfg)

        # Initialize camera
        if not self._camera_sensor.is_initialized:
            self._camera_sensor._initialize_impl()
            self._camera_sensor._is_initialized = True

        # Initialize camera buffers
        self._data.tactile_rgb_image = torch.zeros(
            (self._num_envs, self.cfg.camera_cfg.height, self.cfg.camera_cfg.width, 3),
            dtype=torch.uint8,
            device=self._device,
        )
        self._data.tactile_depth_image = torch.zeros(
            (self._num_envs, self.cfg.camera_cfg.height, self.cfg.camera_cfg.width, 1), device=self._device
        )

        logger.info("Camera-based tactile sensing initialized.")

    def _initialize_force_field(self):
        """Initialize force field tactile sensing components.

        This method sets up all components required for force field based tactile sensing:

        1. Creates PhysX views for elastomer and contact object rigid bodies
        2. Generates tactile sensing points on the elastomer surface using mesh geometry
        3. Initializes SDF (Signed Distance Field) for collision detection
        4. Creates data buffers for storing force field measurements

        The tactile points are generated by ray-casting onto the elastomer mesh surface
        to create a grid of sensing points that will be used for force computation.

        """

        # Generate tactile points on elastomer surface
        self._generate_tactile_points(
            num_divs=list(self.cfg.tactile_array_size),
            margin=getattr(self.cfg, "tactile_margin", 0.003),
            visualize=self.cfg.trimesh_vis_tactile_points,
        )

        self._create_physx_views()

        # Initialize force field data buffers
        self._initialize_force_field_buffers()
        logger.info("Force field tactile sensing initialized.")

    def _create_physx_views(self) -> None:
        """Create PhysX views for contact object and elastomer bodies.

        This method sets up the necessary PhysX views for force field computation:
        1. Creates rigid body view for elastomer
        2. If contact object prim path expression is not None, then:
            a. Finds and validates the object prim and its collision mesh
            b. Creates SDF view for collision detection
            c. Creates rigid body view for object

        """
        elastomer_pattern = self._parent_prims[0].GetPath().pathString.replace("env_0", "env_*")
        self._elastomer_body_view = self._physics_sim_view.create_rigid_body_view([elastomer_pattern])
        # Get elastomer COM for velocity correction
        self._elastomer_com_b = self._elastomer_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]

        if self.cfg.contact_object_prim_path_expr is None:
            return

        contact_object_mesh, contact_object_rigid_body = self._find_contact_object_components()
        # Create SDF view for collision detection
        num_query_points = self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1]
        mesh_path_pattern = contact_object_mesh.GetPath().pathString.replace("env_0", "env_*")
        self._contact_object_sdf_view = self._physics_sim_view.create_sdf_shape_view(
            mesh_path_pattern, num_query_points
        )

        # Create rigid body views for contact object and elastomer
        body_path_pattern = contact_object_rigid_body.GetPath().pathString.replace("env_0", "env_*")
        self._contact_object_body_view = self._physics_sim_view.create_rigid_body_view([body_path_pattern])
        # Get contact object COM for velocity correction
        self._contact_object_com_b = self._contact_object_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]

    def _find_contact_object_components(self) -> tuple[Any, Any]:
        """Find and validate contact object SDF mesh and its parent rigid body.

        This method searches for the contact object prim using the configured filter pattern,
        then locates the first SDF collision mesh within that prim hierarchy and
        identifies its parent rigid body for physics simulation.

        Returns:
            Tuple of (contact_object_mesh, contact_object_rigid_body)
            Returns None if contact object components are not found.

        Note:
            Only SDF meshes are supported for optimal force field computation performance.
            If no SDF mesh is found, the method will log a warning and return None.
        """
        # Find the contact object prim using the configured pattern
        contact_object_prim = sim_utils.find_first_matching_prim(self.cfg.contact_object_prim_path_expr)
        if contact_object_prim is None:
            raise RuntimeError(
                f"No contact object prim found matching pattern: {self.cfg.contact_object_prim_path_expr}"
            )

        def is_sdf_mesh(prim: Usd.Prim) -> bool:
            """Check if a mesh prim is configured for SDF approximation."""
            return (
                prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"
            )

        # Find the SDF mesh within the contact object
        contact_object_mesh = sim_utils.get_first_matching_child_prim(
            contact_object_prim.GetPath(), predicate=is_sdf_mesh
        )
        if contact_object_mesh is None:
            raise RuntimeError(
                f"No SDF mesh found under contact object at path: {contact_object_prim.GetPath().pathString}"
            )

        def find_parent_rigid_body(prim: Usd.Prim) -> Usd.Prim | None:
            """Find the first parent prim with RigidBodyAPI."""
            current_prim = prim
            while current_prim and current_prim.IsValid():
                if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return current_prim
                current_prim = current_prim.GetParent()
                if current_prim.GetPath() == "/":
                    break
            return None

        # Find the rigid body parent of the SDF mesh
        contact_object_rigid_body = find_parent_rigid_body(contact_object_mesh)
        if contact_object_rigid_body is None:
            raise RuntimeError(
                f"No contact object rigid body found for mesh at path: {contact_object_mesh.GetPath().pathString}"
            )

        return contact_object_mesh, contact_object_rigid_body

    def _generate_tactile_points(self, num_divs: list, margin: float, visualize: bool):
        """Generate tactile sensing points from elastomer mesh geometry.

        This method creates a grid of tactile sensing points on the elastomer surface
        by ray-casting onto the mesh geometry. Visual meshes are used for smoother point sampling.

        Args:
            num_divs: Number of divisions [rows, cols] for the tactile grid.
            margin: Margin distance from mesh edges in meters.
            visualize: Whether to show the generated points in trimesh visualization.

        """

        # Get the elastomer prim path
        elastomer_prim_path = self._parent_prims[0].GetPath().pathString

        def is_visual_mesh(prim) -> bool:
            """Check if a mesh prim has visual properties (visual mesh, not collision mesh)."""
            return prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)

        explicit_sampling_path = self.cfg.tactile_sampling_mesh_prim_path
        if explicit_sampling_path is not None:
            elastomer_mesh_prim = self.stage.GetPrimAtPath(explicit_sampling_path)
            if (
                not elastomer_mesh_prim.IsValid()
                or not is_visual_mesh(elastomer_mesh_prim)
                or not elastomer_mesh_prim.GetPath().HasPrefix(
                    self._parent_prims[0].GetPath()
                )
            ):
                raise RuntimeError(
                    "Explicit TacSL sampling mesh must be a non-collision "
                    f"UsdGeom.Mesh below {elastomer_prim_path}: "
                    f"{explicit_sampling_path}"
                )
        else:
            elastomer_mesh_prim = sim_utils.get_first_matching_child_prim(
                elastomer_prim_path, predicate=is_visual_mesh
            )
        if elastomer_mesh_prim is None:
            # The official R15 elastomer is referenced into each cloned G1
            # palm.  Isaac Sim 5.1 keeps its mesh behind an instance proxy,
            # which the upstream child-search helper does not traverse even
            # though the referenced geometry is unchanged.  Keep the official
            # visual-mesh path as first choice and traverse instance proxies
            # only for this audited compatibility case.
            elastomer_root_prim = self.stage.GetPrimAtPath(elastomer_prim_path)
            elastomer_mesh_prim = next(
                (
                    prim
                    for prim in Usd.PrimRange(elastomer_root_prim, Usd.TraverseInstanceProxies())
                    if prim.IsA(UsdGeom.Mesh)
                ),
                None,
            )
            if elastomer_mesh_prim is None:
                raise RuntimeError(f"No mesh found under elastomer at path: {elastomer_prim_path}")
            logger.warning(
                "No directly traversable elastomer mesh found below %s; reading the instanced official "
                "R15 elastomer mesh as the TacSL sampling surface.",
                elastomer_prim_path,
            )

        logger.info(f"Generating tactile points from USD mesh: {elastomer_mesh_prim.GetPath().pathString}")

        # Extract mesh data
        usd_mesh = UsdGeom.Mesh(elastomer_mesh_prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get())
        face_indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())

        # Simple triangulation
        faces = face_indices.reshape(-1, 3)

        # Create bounds
        mesh_bounds = np.array([points.min(axis=0), points.max(axis=0)])

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=points, faces=faces)

        # Generate grid on elastomer
        elastomer_dims = np.diff(mesh_bounds, axis=0).squeeze()
        configured_axis = self.cfg.tactile_sampling_axis
        if configured_axis is None:
            slim_axis = int(np.argmin(elastomer_dims))
        elif configured_axis not in (0, 1, 2):
            raise ValueError(
                "tactile_sampling_axis must be one of 0, 1, 2, or None"
            )
        else:
            slim_axis = int(configured_axis)

        # Determine tip direction using dome geometry
        # For dome-shaped elastomers, the center of mass is shifted toward the dome (contact) side
        mesh_center_of_mass = mesh.center_mass[slim_axis]
        bounding_box_center = (mesh_bounds[0, slim_axis] + mesh_bounds[1, slim_axis]) / 2.0

        configured_tip_sign = self.cfg.tactile_tip_direction_sign
        if configured_tip_sign is None:
            tip_direction_sign = (
                1.0 if mesh_center_of_mass > bounding_box_center else -1.0
            )
        elif configured_tip_sign not in (-1, 1):
            raise ValueError(
                "tactile_tip_direction_sign must be -1, 1, or None"
            )
        else:
            tip_direction_sign = float(configured_tip_sign)

        # Determine gap between adjacent tactile points
        axis_idxs = list(range(3))
        axis_idxs.remove(int(slim_axis))  # Remove slim idx
        center = (mesh_bounds[0] + mesh_bounds[1]) / 2.0
        mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        ray_dir = np.array([0, 0, 0])
        ray_dir[slim_axis] = -tip_direction_sign  # Ray points towards elastomer (opposite of tip direction)
        expected_points = int(np.prod(num_divs))

        def sample_grid(active_margin: float):
            div_sz = (elastomer_dims[axis_idxs] - active_margin * 2.0) / (
                np.array(num_divs) + 1
            )
            if np.any(div_sz <= 0.0):
                return None
            # Released TacSL behavior preserves square spacing and also leaves
            # one extra grid interval inside the declared margin. The
            # anatomical adapter puts the first/last taxel at the margin on
            # each axis so the grid covers the physical patch footprint.
            tactile_points_dx = np.repeat(np.min(div_sz), len(axis_idxs))
            planar_grid_points = []
            idx = 0
            for axis_i in range(3):
                if axis_i == slim_axis:
                    planar_grid_points.append([tip_direction_sign])
                else:
                    if self.cfg.tactile_fill_mesh_extents:
                        axis_grid_points = np.linspace(
                            mesh_bounds[0, axis_i] + active_margin,
                            mesh_bounds[1, axis_i] - active_margin,
                            num_divs[idx],
                        )
                        planar_grid_points.append(axis_grid_points)
                    else:
                        axis_grid_points = np.linspace(
                            center[axis_i]
                            - tactile_points_dx[idx]
                            * (num_divs[idx] + 1.0)
                            / 2.0,
                            center[axis_i]
                            + tactile_points_dx[idx]
                            * (num_divs[idx] + 1.0)
                            / 2.0,
                            num_divs[idx] + 2,
                        )
                        planar_grid_points.append(axis_grid_points[1:-1])
                    idx += 1
            grid_corners = np.array(
                list(itertools.product(*planar_grid_points))
            )
            return mesh_data.intersects_id(
                grid_corners,
                np.tile([ray_dir], (grid_corners.shape[0], 1)),
                return_locations=True,
                multiple_hits=False,
            )

        if self.cfg.tactile_fill_mesh_extents:
            maximum_margin = min(
                0.003,
                0.5 * float(np.min(elastomer_dims[axis_idxs])) - 1.0e-6,
            )
            candidate_margins = np.arange(
                margin, maximum_margin + 5.0e-5, 1.0e-4
            )
        else:
            candidate_margins = np.asarray([margin])
        sampled = None
        effective_margin = float(margin)
        for candidate_margin in candidate_margins:
            candidate = sample_grid(float(candidate_margin))
            if candidate is None:
                continue
            if len(candidate[2]) == expected_points:
                sampled = candidate
                effective_margin = float(candidate_margin)
                break
        if sampled is None:
            raise RuntimeError(
                f"{elastomer_prim_path}: could not place {expected_points} "
                f"taxels inside the mesh from margin {margin} through "
                f"{float(candidate_margins[-1])}"
            )
        index_tri, index_ray, locations = sampled
        self.tactile_grid_effective_margin_m = effective_margin
        if effective_margin > float(margin) + 1.0e-12:
            logger.info(
                "%s: increased anatomical taxel margin from %.6f m to "
                "%.6f m to keep all taxels on the sampling mesh",
                elastomer_prim_path,
                margin,
                effective_margin,
            )

        if visualize:
            query_pointcloud = trimesh.PointCloud(locations, colors=(0.0, 0.0, 1.0))
            trimesh.Scene([mesh, query_pointcloud]).show()

        # Sort and store tactile points
        ray_order = index_ray.argsort()
        tactile_points = locations[ray_order]
        tactile_triangle_ids = index_tri[ray_order]
        # in the frame of the elastomer
        self._tactile_pos_local = torch.tensor(tactile_points, dtype=torch.float32, device=self._device)
        self.num_tactile_points = self._tactile_pos_local.shape[0]
        if self.num_tactile_points != expected_points:
            raise RuntimeError(
                f"{elastomer_prim_path}: number of tactile points does not "
                f"match expected: {self.num_tactile_points} !="
                f" {self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1]}"
            )

        # Preserve the released local-Z normal and local-X/Y signed-shear
        # contract, but bind that frame to the actual triangle sampled by each
        # taxel.  A single constant patch normal is only exact for a planar
        # elastomer and is measurably wrong on the released R15 dome.  The
        # outward triangle normal becomes local Z, the first grid axis is
        # projected into its tangent plane as local X, and local Y completes a
        # right-handed basis.  No force values are changed, clamped, or
        # absolute-valued by this geometry correction.
        tactile_z = np.asarray(
            mesh.face_normals[tactile_triangle_ids], dtype=np.float32
        )
        tip_axis = np.zeros(3, dtype=np.float32)
        tip_axis[slim_axis] = tip_direction_sign
        reverse = np.sum(tactile_z * tip_axis[None], axis=1) < 0.0
        tactile_z[reverse] *= -1.0
        tactile_z /= np.linalg.norm(tactile_z, axis=1, keepdims=True)

        grid_axis = np.zeros(3, dtype=np.float32)
        # The released v2.3.2 implementation uses a constant -pi rotation
        # around local Z, so its reported +X/+Y axes are the negatives of the
        # first/second mesh-grid axes. Preserve that signed-shear convention
        # while adapting only local Z to the sampled curved surface normal.
        # Using +1 here silently inverted both signed shear channels relative
        # to official TacSL by approximately 180 degrees.
        grid_axis[axis_idxs[0]] = -1.0
        tactile_x = grid_axis[None] - tactile_z * np.sum(
            tactile_z * grid_axis[None], axis=1, keepdims=True
        )
        tactile_x_norm = np.linalg.norm(tactile_x, axis=1, keepdims=True)
        if np.any(tactile_x_norm < 1.0e-6):
            raise RuntimeError(
                "TacSL sampled surface normal is parallel to its grid axis"
            )
        tactile_x /= tactile_x_norm
        tactile_y = np.cross(tactile_z, tactile_x)
        tactile_y /= np.linalg.norm(tactile_y, axis=1, keepdims=True)
        tactile_x = np.cross(tactile_y, tactile_z)
        tactile_x /= np.linalg.norm(tactile_x, axis=1, keepdims=True)
        tactile_basis = np.stack((tactile_x, tactile_y, tactile_z), axis=2)
        tactile_quat = math_utils.quat_from_matrix(
            torch.tensor(
                tactile_basis, dtype=torch.float32, device=self._device
            )
        )
        self._tactile_quat_local = tactile_quat

        logger.info(f"Generated {len(tactile_points)} tactile points from USD mesh using ray casting")

    def _initialize_force_field_buffers(self):
        """Initialize data buffers for force field sensing."""
        num_pts = self.num_tactile_points

        # Initialize force field data tensors
        self._data.tactile_points_pos_w = torch.zeros((self._num_envs, num_pts, 3), device=self._device)
        self._data.tactile_points_quat_w = torch.zeros((self._num_envs, num_pts, 4), device=self._device)
        self._data.penetration_depth = torch.zeros((self._num_envs, num_pts), device=self._device)
        self._data.tactile_normal_force = torch.zeros((self._num_envs, num_pts), device=self._device)
        self._data.tactile_shear_force = torch.zeros((self._num_envs, num_pts, 2), device=self._device)
        self._data.tactile_friction_force_magnitude = torch.zeros(
            (self._num_envs, num_pts), device=self._device
        )
        self._data.tactile_signed_distance_m = torch.zeros(
            (self._num_envs, num_pts), device=self._device
        )
        # Simulator-oracle output: this exposes the exact tangential relative
        # velocity already computed by the TacSL friction equation below. It
        # is not consumed by the force path; only the predeclared isolated
        # Plan-11 policy ablation may consume it as a separately named input.
        self._data.tactile_relative_tangential_velocity_w = torch.zeros(
            (self._num_envs, num_pts, 3), device=self._device
        )
        # Simulator-oracle full relative velocity from the same released
        # TacSL friction computation.  This is diagnostic/Plan-11 output only
        # and never feeds back into the force equation.
        self._data.tactile_relative_velocity_w = torch.zeros(
            (self._num_envs, num_pts, 3), device=self._device
        )
        # Evaluation-only oracle: exact SDF normals already used by the TacSL
        # tangential projection. This does not alter the force path.
        self._data.tactile_contact_normal_w = torch.zeros(
            (self._num_envs, num_pts, 3), device=self._device
        )
        # Pre-compute expanded tactile point tensors to avoid repeated unsqueeze/expand operations
        self._tactile_pos_expanded = self._tactile_pos_local.unsqueeze(0).expand(self._num_envs, -1, -1)
        self._tactile_quat_expanded = self._tactile_quat_local.unsqueeze(0).expand(self._num_envs, -1, -1)

    def _initialize_visualization(self):
        """Initialize visualization markers for tactile points."""
        if self.cfg.visualizer_cfg:
            self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data.

        This method updates both camera-based and force field tactile sensing data
        for the specified environments.

        Args:
            env_ids: Sequence of environment indices to update. If length equals
                    total number of environments, all environments are updated.
        """
        # Convert to proper indices for internal methods
        if len(env_ids) == self._num_envs:
            internal_env_ids = slice(None)
        else:
            internal_env_ids = env_ids

        # Update camera-based tactile data
        # Manager-based tasks query force observations while their observation
        # manager is being constructed, before application code can capture the
        # required no-load camera render.  Keep the official explicit nominal
        # contract: the camera branch starts only after get_initial_render(),
        # while the independent official force-field branch remains available.
        if self.cfg.enable_camera_tactile and self._nominal_tactile is not None:
            self._update_camera_tactile(internal_env_ids)

        # Update force field tactile data
        if self.cfg.enable_force_field:
            # The official TacSL force path obtains full PhysX body views,
            # transforms all tactile points, and performs one batched SDF
            # query before applying ``env_ids``.  Its pre-expanded taxel
            # tensors also retain the full environment dimension.  Passing a
            # partial reset mask into the final force routine therefore mixes
            # subset poses with full taxel tensors (and advanced-index writes
            # would update copies rather than the sensor buffers).  Recompute
            # the exact official force equations for the full batch whenever
            # any force row is stale.  This is an auditable scheduling fix,
            # not a tactile approximation or proxy.
            self._update_force_field(slice(None))

    def _update_camera_tactile(self, env_ids: Sequence[int] | slice):
        """Update camera-based tactile sensing data.

        This method updates the camera sensor and processes the depth information
        to compute tactile measurements. It computes the difference from the nominal
        (no-contact) state and renders it using the GelSight tactile renderer.

        Args:
            env_ids: Environment indices or slice to update. Can be a sequence of
                    integers or a slice object for batch processing.
        """
        if self._nominal_tactile is None:
            raise RuntimeError("Nominal tactile is not set. Please call get_initial_render() first.")
        # Update the nested camera with the actual elapsed outer-sensor time.
        # In manager-based environments this sensor is updated once per
        # control step even though ``_sim_physics_dt`` is one PhysX substep.
        # Advancing the camera by only the latter repeats each R15 frame for
        # ``decimation`` control samples and breaks same-step optical/force
        # correspondence.  SensorBase has not yet committed
        # ``_timestamp_last_update`` while this callback runs, so their
        # difference is the exact elapsed interval for this update.
        elapsed = self._timestamp - self._timestamp_last_update
        camera_dt = float(torch.max(elapsed).item())
        if camera_dt <= 0.0:
            camera_dt = self._sim_physics_dt
        self._camera_sensor.update(camera_dt)

        # Get camera data
        camera_data = self._camera_sensor.data

        # Check for either distance_to_image_plane or depth (they are equivalent)
        depth_key = None
        if "distance_to_image_plane" in camera_data.output:
            depth_key = "distance_to_image_plane"
        elif "depth" in camera_data.output:
            depth_key = "depth"

        if depth_key:
            self._data.tactile_depth_image[env_ids] = camera_data.output[depth_key][env_ids].clone()
            diff = self._nominal_tactile[depth_key][env_ids] - self._data.tactile_depth_image[env_ids]
            self._data.tactile_rgb_image[env_ids] = self._tactile_rgb_render.render(diff.squeeze(-1))

    #########################################################################################
    # Force field tactile sensing
    #########################################################################################

    def _update_force_field(self, env_ids: Sequence[int] | slice):
        """Update force field tactile sensing data.

        This method computes penalty-based tactile forces using Signed Distance Field (SDF)
        queries. It transforms tactile points to contact object local coordinates, queries the SDF of the
        contact object for collision detection, and computes normal and shear forces based on
        penetration depth and relative velocities.

        Args:
            env_ids: Environment indices or slice to update. Can be a sequence of
                    integers or a slice object for batch processing.

        Note:
            Requires both elastomer and contact object body views to be initialized. Returns
            early if tactile points or body views are not available.
        """
        # Step 1: Get elastomer pose and precompute pose components
        elastomer_pos_w, elastomer_quat_w = self._elastomer_body_view.get_transforms().split([3, 4], dim=-1)
        elastomer_quat_w = math_utils.convert_quat(elastomer_quat_w, to="wxyz")

        # Transform tactile points to world coordinates, used for visualization
        self._transform_tactile_points_to_world(elastomer_pos_w, elastomer_quat_w)

        # earlly return if contact object body view is not available
        # this could happen if the contact object is not specified when tactile_points are required for visualization
        if self._contact_object_body_view is None:
            return

        # Step 2: Transform tactile points to contact object local frame for SDF queries
        contact_object_pos_w, contact_object_quat_w = self._contact_object_body_view.get_transforms().split(
            [3, 4], dim=-1
        )
        contact_object_quat_w = math_utils.convert_quat(contact_object_quat_w, to="wxyz")

        world_tactile_points = self._data.tactile_points_pos_w
        points_contact_object_local, contact_object_quat_inv = self._transform_points_to_contact_object_local(
            world_tactile_points, contact_object_pos_w, contact_object_quat_w
        )

        # Step 3: Query SDF for collision detection
        sdf_values_and_gradients = self._contact_object_sdf_view.get_sdf_and_gradients(points_contact_object_local)
        sdf_values = sdf_values_and_gradients[..., -1]  # Last component is SDF value
        sdf_gradients = sdf_values_and_gradients[..., :-1]  # First 3 components are gradients

        # Step 4: Compute tactile forces from SDF data
        self._compute_tactile_forces_from_sdf(
            points_contact_object_local,
            sdf_values,
            sdf_gradients,
            contact_object_pos_w,
            contact_object_quat_w,
            elastomer_quat_w,
            env_ids,
        )

    def _transform_tactile_points_to_world(self, pos_w: torch.Tensor, quat_w: torch.Tensor):
        """Transform tactile points from local to world coordinates.

        Args:
            pos_w: Elastomer positions in world frame. Shape: (num_envs, 3)
            quat_w: Elastomer quaternions in world frame. Shape: (num_envs, 4)
        """
        num_pts = self.num_tactile_points

        quat_expanded = quat_w.unsqueeze(1).expand(-1, num_pts, -1)
        pos_expanded = pos_w.unsqueeze(1).expand(-1, num_pts, -1)

        # Apply transformation
        tactile_pos_w = math_utils.quat_apply(quat_expanded, self._tactile_pos_expanded) + pos_expanded
        tactile_quat_w = math_utils.quat_mul(quat_expanded, self._tactile_quat_expanded)

        # Store in data
        self._data.tactile_points_pos_w = tactile_pos_w
        self._data.tactile_points_quat_w = tactile_quat_w

    def _transform_points_to_contact_object_local(
        self, world_points: torch.Tensor, contact_object_pos_w: torch.Tensor, contact_object_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized version: Transform world coordinates to contact object local frame.

        Args:
            world_points: Points in world coordinates. Shape: (num_envs, num_points, 3)
            contact_object_pos_w: Contact object positions in world frame. Shape: (num_envs, 3)
            contact_object_quat_w: Contact object quaternions in world frame. Shape: (num_envs, 4)

        Returns:
            Points in contact object local coordinates and inverse quaternions
        """
        # Get inverse transformation (per environment)
        # wxyz in torch
        contact_object_quat_inv, contact_object_pos_inv = torch_utils.tf_inverse(
            contact_object_quat_w, contact_object_pos_w
        )
        num_pts = self.num_tactile_points

        contact_object_quat_expanded = contact_object_quat_inv.unsqueeze(1).expand(-1, num_pts, 4)
        contact_object_pos_expanded = contact_object_pos_inv.unsqueeze(1).expand(-1, num_pts, 3)

        # Apply transformation
        points_sdf = torch_utils.tf_apply(contact_object_quat_expanded, contact_object_pos_expanded, world_points)

        return points_sdf, contact_object_quat_inv

    def _get_tactile_points_velocities(
        self, linvel_world: torch.Tensor, angvel_world: torch.Tensor, quat_world: torch.Tensor
    ) -> torch.Tensor:
        """Optimized version: Compute tactile point velocities from precomputed velocities.

        Args:
            linvel_world: Elastomer linear velocities. Shape: (num_envs, 3)
            angvel_world: Elastomer angular velocities. Shape: (num_envs, 3)
            quat_world: Elastomer quaternions. Shape: (num_envs, 4)

        Returns:
            Tactile point velocities in world frame. Shape: (num_envs, num_points, 3)
        """
        num_pts = self.num_tactile_points

        # Pre-expand all required tensors once
        quat_expanded = quat_world.unsqueeze(1).expand(-1, num_pts, 4)
        tactile_pos_expanded = self._tactile_pos_expanded

        # Transform local positions to world frame relative vectors
        tactile_pos_world_relative = math_utils.quat_apply(quat_expanded, tactile_pos_expanded)

        # Compute velocity due to angular motion: ω × r
        angvel_expanded = angvel_world.unsqueeze(1).expand(-1, num_pts, 3)
        angular_velocity_contribution = torch.cross(angvel_expanded, tactile_pos_world_relative, dim=-1)

        # Add linear velocity contribution
        linvel_expanded = linvel_world.unsqueeze(1).expand(-1, num_pts, 3)
        tactile_velocity_world = angular_velocity_contribution + linvel_expanded

        return tactile_velocity_world

    def _compute_tactile_forces_from_sdf(
        self,
        points_contact_object_local: torch.Tensor,
        sdf_values: torch.Tensor,
        sdf_gradients: torch.Tensor,
        contact_object_pos_w: torch.Tensor,
        contact_object_quat_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
        env_ids: Sequence[int] | slice,
    ) -> None:
        """Optimized version: Compute tactile forces from SDF values using precomputed parameters.

        This method now operates directly on the pre-allocated data tensors to avoid
        unnecessary memory allocation and copying.

        Args:
            points_contact_object_local: Points in contact object local frame
            sdf_values: SDF values (negative means penetration)
            sdf_gradients: SDF gradients (surface normals)
            contact_object_pos_w: Contact object positions in world frame
            contact_object_quat_w: Contact object quaternions in world frame
            elastomer_quat_w: Elastomer quaternions
            env_ids: Environment indices being updated

        """
        depth = self._data.penetration_depth[env_ids]
        tactile_normal_force = self._data.tactile_normal_force[env_ids]
        tactile_shear_force = self._data.tactile_shear_force[env_ids]
        tactile_friction_force_magnitude = (
            self._data.tactile_friction_force_magnitude[env_ids]
        )
        tactile_relative_tangential_velocity_w = (
            self._data.tactile_relative_tangential_velocity_w[env_ids]
        )
        tactile_relative_velocity_w = self._data.tactile_relative_velocity_w[env_ids]
        tactile_contact_normal_w = self._data.tactile_contact_normal_w[env_ids]
        tactile_signed_distance_m = self._data.tactile_signed_distance_m[env_ids]

        # Clear the output tensors
        tactile_normal_force.zero_()
        tactile_shear_force.zero_()
        tactile_friction_force_magnitude.zero_()
        tactile_relative_tangential_velocity_w.zero_()
        tactile_relative_velocity_w.zero_()
        tactile_contact_normal_w.zero_()
        depth.zero_()
        tactile_signed_distance_m.copy_(sdf_values[env_ids])

        contact_offset_m = float(self.cfg.tactile_contact_offset_m)
        if not (0.0 <= contact_offset_m < float("inf")):
            raise ValueError("tactile_contact_offset_m must be nonnegative and finite")
        # Released TacSL uses zero offset. The anatomical adapter models the
        # same thin compliant contact layer that lets PhysX produce pad force
        # before the undeformed visual surface has geometrically penetrated.
        depth[:] = torch.clamp(
            contact_offset_m - sdf_values[env_ids], min=0.0
        )

        # Get collision mask for points inside the configured compliant layer.
        collision_mask = depth > 0.0

        # Use pre-allocated tensors instead of creating new ones
        num_pts = self.num_tactile_points

        if collision_mask.any() or self.cfg.visualize_sdf_closest_pts:
            # Get contact object and elastomer velocities (com velocities)
            contact_object_velocities = self._contact_object_body_view.get_velocities()
            contact_object_linvel_w_com = contact_object_velocities[env_ids, :3]
            contact_object_angvel_w = contact_object_velocities[env_ids, 3:]

            elastomer_velocities = self._elastomer_body_view.get_velocities()
            elastomer_linvel_w_com = elastomer_velocities[env_ids, :3]
            elastomer_angvel_w = elastomer_velocities[env_ids, 3:]

            # Contact object adjustment
            contact_object_com_w_offset = math_utils.quat_apply(
                contact_object_quat_w[env_ids], self._contact_object_com_b[env_ids]
            )
            contact_object_linvel_w = contact_object_linvel_w_com - torch.cross(
                contact_object_angvel_w, contact_object_com_w_offset, dim=-1
            )
            # v_origin = v_com - w x (com_world_offset) where com_world_offset = quat_apply(quat, com_b)
            elastomer_com_w_offset = math_utils.quat_apply(elastomer_quat_w[env_ids], self._elastomer_com_b[env_ids])
            elastomer_linvel_w = elastomer_linvel_w_com - torch.cross(
                elastomer_angvel_w, elastomer_com_w_offset, dim=-1
            )

            # Normalize gradients to get surface normals in local frame
            normals_local = torch.nn.functional.normalize(sdf_gradients[env_ids], dim=-1)

            # Transform normals to world frame (rotate by contact object orientation) - use precomputed quaternions
            contact_object_quat_expanded = contact_object_quat_w[env_ids].unsqueeze(1).expand(-1, num_pts, 4)

            # Apply quaternion transformation
            normals_world = math_utils.quat_apply(contact_object_quat_expanded, normals_local)
            tactile_contact_normal_w[:] = torch.where(
                collision_mask.unsqueeze(-1),
                normals_world,
                torch.zeros_like(normals_world),
            )

            # Compute normal contact force: F_n = k_n * depth
            fc_norm = self.cfg.normal_contact_stiffness * depth

            # Get tactile point velocities using precomputed velocities
            tactile_velocity_world = self._get_tactile_points_velocities(
                elastomer_linvel_w, elastomer_angvel_w, elastomer_quat_w[env_ids]
            )

            # Use precomputed contact object velocities
            closest_points_sdf = points_contact_object_local[env_ids] + depth.unsqueeze(-1) * normals_local

            if self.cfg.visualize_sdf_closest_pts:
                debug_closest_points_sdf = (
                    points_contact_object_local[env_ids] - sdf_values[env_ids].unsqueeze(-1) * normals_local
                )
                self.debug_closest_points_wolrd = math_utils.quat_apply(
                    contact_object_quat_expanded, debug_closest_points_sdf
                ) + contact_object_pos_w[env_ids].unsqueeze(1).expand(-1, num_pts, 3)

            contact_object_linvel_expanded = contact_object_linvel_w.unsqueeze(1).expand(-1, num_pts, 3)
            contact_object_angvel_expanded = contact_object_angvel_w.unsqueeze(1).expand(-1, num_pts, 3)
            closest_points_vel_world = (
                torch.linalg.cross(
                    contact_object_angvel_expanded,
                    math_utils.quat_apply(contact_object_quat_expanded, closest_points_sdf),
                )
                + contact_object_linvel_expanded
            )

            # Compute relative velocity at contact points
            relative_velocity_world = tactile_velocity_world - closest_points_vel_world
            tactile_relative_velocity_w[:] = torch.where(
                collision_mask.unsqueeze(-1),
                relative_velocity_world,
                torch.zeros_like(relative_velocity_world),
            )

            # Compute tangential velocity (perpendicular to normal)
            vt_world = relative_velocity_world - normals_world * torch.sum(
                normals_world * relative_velocity_world, dim=-1, keepdim=True
            )
            vt_norm = torch.norm(vt_world, dim=-1)
            tactile_relative_tangential_velocity_w[:] = torch.where(
                collision_mask.unsqueeze(-1), vt_world, torch.zeros_like(vt_world)
            )

            # Compute friction force: F_t = min(k_t * |v_t|, mu * F_n)
            ft_static_norm = self.cfg.tangential_stiffness * vt_norm
            ft_dynamic_norm = self.cfg.friction_coefficient * fc_norm
            ft_norm = torch.minimum(ft_static_norm, ft_dynamic_norm)

            # Apply friction force opposite to tangential velocity
            ft_world = -ft_norm.unsqueeze(-1) * vt_world / (vt_norm.unsqueeze(-1).clamp(min=1e-9))

            # Keep the two physical sources separate. Projecting
            # ``fc_world + ft_world`` and calling local X/Y "shear" makes a
            # stationary angled normal load look like sliding. Normal load is
            # the nonnegative penalty magnitude; signed XY shear contains
            # only the friction traction in the geometry-bound taxel frame.
            friction_force_tactile = math_utils.quat_apply_inverse(
                self._data.tactile_points_quat_w[env_ids],
                ft_world,
            )
            tactile_normal_force[:] = torch.where(
                collision_mask, fc_norm, torch.zeros_like(fc_norm)
            )
            tactile_shear_force[:] = torch.where(
                collision_mask.unsqueeze(-1),
                friction_force_tactile[..., :2],
                torch.zeros_like(friction_force_tactile[..., :2]),
            )
            tactile_friction_force_magnitude[:] = torch.where(
                collision_mask, ft_norm, torch.zeros_like(ft_norm)
            )

    #########################################################################################
    # Debug visualization
    #########################################################################################

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects."""
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if self._tactile_visualizer is None:
                self._tactile_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self._tactile_visualizer.set_visibility(True)
        else:
            if self._tactile_visualizer:
                self._tactile_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization of tactile sensor data.

        This method is called during each simulation step when debug visualization is enabled.
        It visualizes tactile sensing points as 3D markers in the simulation viewport to help
        with debugging and understanding sensor behavior.

        The method handles two visualization modes:

        1. **Standard mode**: Visualizes ``tactile_points_pos_w`` - the world positions of
            tactile sensing points on the sensor surface
        2. **SDF debug mode**: When ``cfg.visualize_sdf_closest_pts`` is True, visualizes
            ``debug_closest_points_wolrd`` - the closest surface points computed during
            SDF-based force calculations
        """
        # Safety check - return if not properly initialized
        if not hasattr(self, "_tactile_visualizer") or self._tactile_visualizer is None:
            return
        vis_points = None

        if self.cfg.visualize_sdf_closest_pts and hasattr(self, "debug_closest_points_wolrd"):
            vis_points = self.debug_closest_points_wolrd
        else:
            vis_points = self._data.tactile_points_pos_w

        if vis_points is None or vis_points.numel() == 0:
            return

        viz_points = vis_points.view(-1, 3)  # Shape: (num_envs * num_points, 3)

        indices = torch.zeros(viz_points.shape[0], dtype=torch.long, device=self._device)

        marker_scales = torch.ones(viz_points.shape[0], 3, device=self._device)

        # Visualize tactile points
        self._tactile_visualizer.visualize(translations=viz_points, marker_indices=indices, scales=marker_scales)
