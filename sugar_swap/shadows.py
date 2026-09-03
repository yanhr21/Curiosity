"""Stand-ins for the IsaacLab modules that exist only to author USD prims.

`isaaclab.sim`, `isaaclab.markers`, `isaaclab.terrains` and `isaaclab.actuators` are almost
entirely Isaac Sim authoring APIs: they describe how to spawn a prim, bind a material, or
draw a debug marker. Newton loads the URDF directly and has its own solver configuration, so
none of that machinery transfers, but SUGAR's configs still name these classes.

Each class below is a `LenientCfg`, so the fields Newton reads are declared and everything
else is recorded as ignored (see `lenient.report_ignored`). The spawner functions and the
marker class are no-ops: reaching them means something asked Isaac Sim to author geometry,
which on this backend is the scene builder's job instead.
"""

from __future__ import annotations

import types
from typing import Any

from .lenient import LenientCfg


# ---- isaaclab.sim -----------------------------------------------------------
class UsdFileCfg(LenientCfg):
    """Spawn from USD. Newton reads only the path, and only to locate a sibling URDF."""

    usd_path: str = ""
    scale: tuple[float, float, float] | None = None


class UrdfFileCfg(LenientCfg):
    """Spawn from URDF. `asset_path` is the one field the Newton scene builder needs."""

    asset_path: str = ""
    scale: tuple[float, float, float] | None = None
    joint_drive: Any = None


class UrdfConverterCfg(LenientCfg):
    """URDF import settings. Newton's URDF importer takes its own arguments."""

    class JointDriveCfg(LenientCfg):
        class PDGainsCfg(LenientCfg):
            stiffness: float | None = None
            damping: float | None = None

        gains: Any = None
        target_type: str = "position"

    joint_drive: Any = None


class MassPropertiesCfg(LenientCfg):
    """Consumed: `mass` overrides the density-derived mass of the spawned body."""

    mass: float | None = None


class RigidBodyMaterialCfg(LenientCfg):
    """Consumed: the friction coefficients, which map onto Newton's per-shape `mu`."""

    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None


class RigidBodyPropertiesCfg(LenientCfg):
    """PhysX per-body solver settings; Newton configures its solver globally instead."""


class ArticulationRootPropertiesCfg(LenientCfg):
    """PhysX articulation settings. `fix_root_link` is the one with a Newton equivalent."""

    fix_root_link: bool | None = None


class CollisionPropertiesCfg(LenientCfg):
    """PhysX collision offsets; Newton uses its own margin/gap (see the collider notes)."""


class SDFMeshPropertiesCfg(LenientCfg):
    """PhysX SDF collision. Newton has hydroelastic SDF, configured by the scene builder."""

    sdf_resolution: int | None = None


class MdlFileCfg(LenientCfg):
    """Visual material only; irrelevant to physics and to Newton's renderer."""


class DistantLightCfg(LenientCfg):
    """Lighting, ignored: Newton's viewer provides its own."""


class DomeLightCfg(LenientCfg):
    """Lighting, ignored: Newton's viewer provides its own."""


class PinholeCameraCfg(LenientCfg):
    """Camera intrinsics for tiled rendering, which this backend does not implement."""


class PhysxCfg(LenientCfg):
    """PhysX GPU buffer sizes and solver counts.

    None of these transfer: Newton sizes its own contact buffers (`nconmax`,
    `max_triangle_pairs`) from the collider configuration. SUGAR raises
    `gpu_max_rigid_patch_count` here, which is the PhysX analogue of the Newton contact
    capacity this project had to tune separately.
    """


class SimulationCfg(LenientCfg):
    """Consumed: `dt` and `gravity`, both of which Newton needs."""

    dt: float = 1.0 / 200.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    render_interval: int = 1
    physics_material: Any = None
    physx: Any = PhysxCfg()


def _no_spawn(*_args, **_kwargs):
    raise NotImplementedError(
        "sugar_swap: an Isaac Sim spawner was called. On the Newton backend, geometry is "
        "created by sugar_swap.scene from the URDF, so this path should never be reached."
    )


def _no_op(*_args, **_kwargs) -> None:
    return None


# ---- isaaclab.markers -------------------------------------------------------
class _MarkerDict(dict):
    """Marker sub-configs, created on first access.

    IsaacLab's command configs adjust presets in place -- `cfg.markers["arrow"].scale = ...`
    -- at class-definition time. Since the presets here are synthesised rather than real,
    the named entries have to appear on demand or those module-level tweaks raise `KeyError`
    during import.
    """

    def __missing__(self, key: str) -> LenientCfg:
        value = LenientCfg()
        self[key] = value
        return value


class VisualizationMarkersCfg(LenientCfg):
    """Debug-marker config; retained so SUGAR's `debug_vis` configs still construct."""

    markers: dict | None = None
    prim_path: str = ""

    def __init__(self, **kwargs):
        kwargs.setdefault("markers", _MarkerDict())
        if isinstance(kwargs["markers"], dict) and not isinstance(kwargs["markers"], _MarkerDict):
            kwargs["markers"] = _MarkerDict(kwargs["markers"])
        super().__init__(**kwargs)


class VisualizationMarkers:
    """No-op debug markers.

    SUGAR's command terms draw reference frames when `debug_vis` is on. Headless training
    never enables it, and Newton's viewer has no equivalent primitive, so the calls are
    accepted and dropped rather than made an error.
    """

    def __init__(self, cfg: Any = None):
        self.cfg = cfg

    def visualize(self, *_args, **_kwargs) -> None:
        pass

    def set_visibility(self, *_args, **_kwargs) -> None:
        pass


# ---- isaaclab.terrains ------------------------------------------------------
class TerrainImporterCfg(LenientCfg):
    """Consumed: nothing. Newton's scene builder adds a single ground plane."""

    terrain_type: str = "plane"
    prim_path: str = "/World/ground"
    env_spacing: float = 0.0


# ---- isaaclab.actuators -----------------------------------------------------
class ImplicitActuatorCfg(LenientCfg):
    """Consumed: stiffness, damping and the limits, which become Newton's joint drives.

    SUGAR's G1 uses this actuator, so these gains are the ones that actually reach the
    solver. `velocity_limit_sim` is declared because Newton can honour it, unlike the
    earlier hand-written port which dropped it.
    """

    joint_names_expr: list[str] | None = None
    stiffness: Any = None
    damping: Any = None
    effort_limit: Any = None
    effort_limit_sim: Any = None
    velocity_limit: Any = None
    velocity_limit_sim: Any = None
    armature: Any = None
    friction: Any = None


class IdealPDActuatorCfg(ImplicitActuatorCfg):
    """Explicit PD, computed outside the solver. Same fields as the implicit case."""


class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Adds an action-delay buffer; the delay itself is not modelled on this backend."""

    min_delay: int = 0
    max_delay: int = 0


class DelayedPDActuator:
    """Referenced by SUGAR only as a type; instantiating it would mean the delay matters."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "sugar_swap: DelayedPDActuator is not modelled on the Newton backend."
        )


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def build() -> dict[str, types.ModuleType]:
    """Construct the shadow modules, keyed by the `sys.modules` name they replace."""
    sim = _module(
        "isaaclab.sim",
        UsdFileCfg=UsdFileCfg,
        UrdfFileCfg=UrdfFileCfg,
        UrdfConverterCfg=UrdfConverterCfg,
        MassPropertiesCfg=MassPropertiesCfg,
        RigidBodyMaterialCfg=RigidBodyMaterialCfg,
        RigidBodyPropertiesCfg=RigidBodyPropertiesCfg,
        ArticulationRootPropertiesCfg=ArticulationRootPropertiesCfg,
        CollisionPropertiesCfg=CollisionPropertiesCfg,
        SDFMeshPropertiesCfg=SDFMeshPropertiesCfg,
        MdlFileCfg=MdlFileCfg,
        DistantLightCfg=DistantLightCfg,
        DomeLightCfg=DomeLightCfg,
        PinholeCameraCfg=PinholeCameraCfg,
        SimulationCfg=SimulationCfg,
        PhysxCfg=PhysxCfg,
        bind_physics_material=_no_op,
        define_mesh_collision_properties=_no_op,
        make_uninstanceable=_no_op,
        clone=lambda fn: fn,
        spawn_from_urdf=_no_spawn,
        spawn_from_usd=_no_spawn,
    )
    sim.__path__ = []
    from_files = _module(
        "isaaclab.sim.spawners.from_files",
        UsdFileCfg=UsdFileCfg,
        UrdfFileCfg=UrdfFileCfg,
        spawn_from_urdf=_no_spawn,
        spawn_from_usd=_no_spawn,
    )
    spawners = _module("isaaclab.sim.spawners", from_files=from_files)
    spawners.__path__ = []
    sim.spawners = spawners

    # USD prim queries, imported by the task-space action terms. There is no stage on this
    # backend, so a query can only ever return nothing.
    sim_utils_mod = _module(
        "isaaclab.sim.utils",
        find_matching_prims=lambda *_a, **_k: [],
        find_matching_prim_paths=lambda *_a, **_k: [],
        get_first_matching_child_prim=lambda *_a, **_k: None,
        bind_physics_material=_no_op,
    )
    sim.utils = sim_utils_mod

    markers = _module(
        "isaaclab.markers",
        VisualizationMarkers=VisualizationMarkers,
        VisualizationMarkersCfg=VisualizationMarkersCfg,
    )
    markers_config = _module("isaaclab.markers.config")

    def _marker_cfg(name: str) -> VisualizationMarkersCfg:
        """Synthesise any `*_MARKER_CFG` on demand.

        IsaacLab defines a couple of dozen debug-marker presets and its command terms import
        them by name. Since every marker is a no-op here, generating them on access avoids
        tracking that list, while a name that is not a marker preset still raises so real
        typos are not masked. Each access returns a fresh instance because SUGAR mutates
        copies of these configs.
        """
        if name.endswith("_MARKER_CFG") or name.endswith("_CFG"):
            return VisualizationMarkersCfg(markers={}, prim_path=f"/Visuals/{name.lower()}")
        raise AttributeError(f"module 'isaaclab.markers.config' has no attribute {name!r}")

    markers_config.__getattr__ = _marker_cfg
    markers.config = markers_config

    terrains = _module("isaaclab.terrains", TerrainImporterCfg=TerrainImporterCfg)

    def _terrain_attr(name: str):
        """Resolve terrain generator names on demand.

        Newton's scene builder adds a flat ground plane, so IsaacLab's terrain generation --
        height fields, sub-terrain patches, the importer itself -- has no counterpart. The
        config classes resolve to lenient placeholders because SUGAR's cfgs reference them,
        while `TerrainImporter` raises if constructed, since a silently missing terrain would
        change the contact geometry the robot walks on.
        """
        if name.endswith("Cfg"):
            return type(name, (LenientCfg,), {})
        if name == "TerrainImporter":
            def _init(self, *_a, **_k):
                raise NotImplementedError(
                    "sugar_swap: only a flat ground plane is provided on the Newton backend."
                )

            return type("TerrainImporter", (), {"__init__": _init})
        raise AttributeError(f"module 'isaaclab.terrains' has no attribute {name!r}")

    terrains.__getattr__ = _terrain_attr
    actuators = _module(
        "isaaclab.actuators",
        ImplicitActuatorCfg=ImplicitActuatorCfg,
        IdealPDActuatorCfg=IdealPDActuatorCfg,
        DelayedPDActuatorCfg=DelayedPDActuatorCfg,
        DelayedPDActuator=DelayedPDActuator,
    )

    return {
        "isaaclab.sim": sim,
        "isaaclab.sim.utils": sim_utils_mod,
        "isaaclab.sim.spawners": spawners,
        "isaaclab.sim.spawners.from_files": from_files,
        "isaaclab.markers": markers,
        "isaaclab.markers.config": markers_config,
        "isaaclab.terrains": terrains,
        "isaaclab.actuators": actuators,
    }
