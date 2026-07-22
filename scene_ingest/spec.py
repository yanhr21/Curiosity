# SPDX-License-Identifier: Apache-2.0
"""Dataset-agnostic scene intermediate representation (``SceneSpec``).

Adapters (``scene_ingest/adapters/*``) parse a dataset's native layout into these
dataclasses; :mod:`scene_ingest.newton_build` turns a ``SceneSpec`` into a Newton
``ModelBuilder`` scene with tactile/vision sensors. Physics-first, SI units, Z-up
(Newton convention), floor at ``z = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialSpec:
    """Contact/inertial material for a rigid or deformable body.

    Rigid materials share a hard hydroelastic stiffness and differ only by friction,
    restitution, and density (verified reasoning, ``context.md`` §5). Deformables are
    differentiated by Young's modulus ``young_e``.
    """

    mu: float = 0.7  # Coulomb friction coefficient
    restitution: float = 0.0
    density: float | None = None  # [kg/m^3]; if None, derived from mass / hull volume
    rigid_kh: float = 1.0e12  # hard hydroelastic contact stiffness
    deformable: bool = False
    young_e: float | None = None  # [Pa], only when deformable
    poisson_nu: float = 0.45


@dataclass
class ObjectSpec:
    """One placeable body in the scene."""

    object_id: str
    mesh_path: str  # PLY / GLB / OBJ, source local frame
    position: tuple[float, float, float]  # [m], room frame, floor z=0
    rotation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw=z
    bbox_dims: tuple[float, float, float] | None = None  # (w,l,h) [m]; fit mesh AABB to this
    mass: float | None = None  # [kg]
    material: MaterialSpec = field(default_factory=MaterialSpec)
    semantic_type: str = "object"
    support_parent: str = "floor"  # "floor" | "wall" | "<object_id>" — spawn/settle order
    is_static: bool = False  # fixed to world (walls, heavy furniture)
    up_axis: str = "z"  # source mesh up-axis ("z" or "y"); builder rotates to Z-up


@dataclass
class WallSpec:
    wall_id: str
    start: tuple[float, float]  # (x,y) [m]
    end: tuple[float, float]
    height: float
    thickness: float = 0.1


@dataclass
class DoorSpec:
    door_id: str
    wall_id: str
    position_on_wall: float  # fraction [0,1] along the wall
    width: float
    height: float
    opens_inward: bool = True


@dataclass
class RoomSpec:
    room_type: str
    bounds: tuple[float, float, float]  # (width,length,height) [m]
    walls: list[WallSpec] = field(default_factory=list)
    doors: list[DoorSpec] = field(default_factory=list)
    floor_material: MaterialSpec = field(default_factory=MaterialSpec)


@dataclass
class RobotSpec:
    """Robot to drop into the scene (URDF or Newton asset name)."""

    asset: str = "franka_emika_panda"
    urdf_subpath: str = "urdf/fr3_franka_hand.urdf"
    base_position: tuple[float, float, float] = (-0.5, -0.5, 0.05)
    tactile_link_suffixes: tuple[str, ...] = ("fr3_leftfinger", "fr3_rightfinger", "fr3_hand")
    pad_kh: float = 1.0e10  # compliant pad → broad contact patch for tactile


@dataclass
class SensorSpec:
    tactile: bool = True  # SensorContact + hydroelastic contact surface on pad links
    camera: bool = True  # SensorTiledCamera (RGB/depth)
    imu: bool = False  # SensorIMU (proprioception)


@dataclass
class RandomizationSpec:
    """Per-object physical-enrichment distributions (the Robot-Baby differentiator).

    Each entry is a multiplicative or additive range applied at build time so the same
    demonstrated intent must be solved under different physics.
    """

    mass_scale: tuple[float, float] = (1.0, 1.0)  # ×mass
    com_offset_m: float = 0.0  # max random |CoM| shift [m]
    mu_range: tuple[float, float] | None = None  # absolute friction override range
    restitution_range: tuple[float, float] | None = None
    kh_scale: tuple[float, float] = (1.0, 1.0)  # ×rigid_kh (compliance)
    enabled: bool = False


@dataclass
class SceneSpec:
    """A full ingested scene: room + objects + robot + sensors + randomization."""

    scene_id: str
    source_dataset: str  # "sage-10k", "rest3d", "robocasa", "genpipe", …
    root_dir: str  # dir holding meshes referenced by ObjectSpec.mesh_path (relative ok)
    room: RoomSpec
    objects: list[ObjectSpec] = field(default_factory=list)
    robot: RobotSpec | None = field(default_factory=RobotSpec)
    sensors: SensorSpec = field(default_factory=SensorSpec)
    randomization: RandomizationSpec = field(default_factory=RandomizationSpec)

    def summary(self) -> str:
        n_static = sum(o.is_static for o in self.objects)
        return (
            f"{self.scene_id} [{self.source_dataset}] {self.room.room_type} "
            f"{self.room.bounds} · {len(self.objects)} objects "
            f"({n_static} static) · {len(self.room.walls)} walls {len(self.room.doors)} doors"
        )
