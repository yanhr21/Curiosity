"""`Articulation` and `RigidObject` backed by Newton instead of PhysX tensor views.

These are the objects SUGAR reaches through `env.scene["robot"]` and `env.scene["object"]`.
They own no state: the Newton `Model`/`State` live on the scene, and each asset holds the
index maps that pick its rows out of the shared arrays. That keeps a single Newton model for
all environments, which is what makes the vectorised step fast.

Name resolution deliberately reuses IsaacLab's `resolve_matching_names`, so SUGAR's
`joint_names_expr` regexes select exactly the joints they select on Isaac Sim. Getting this
subtly different would silently reorder the action vector and invalidate the released
checkpoint, which is the failure mode the earlier hand-written port suffered from.

Joint ordering is Newton's, not Isaac Sim's. That is self-consistent for training from
scratch, but `joint_order_permutation` is exposed for the case that matters: replaying a
checkpoint whose action head was trained against Isaac Sim's ordering.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .data import ArticulationData, RigidObjectData
from .lenient import LenientCfg
from .physx_view import ArticulationPhysxView, PhysicsSimView, RigidBodyPhysxView


class AssetBaseCfg(LenientCfg):
    """Consumed: `prim_path` (as the scene key) and `spawn` (to find the URDF/USD)."""

    prim_path: str = ""
    spawn: Any = None
    init_state: Any = None
    collision_group: int = 0
    debug_vis: bool = False


class _InitialStateCfg(LenientCfg):
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_pos: dict[str, float] | None = None
    joint_vel: dict[str, float] | None = None


class RigidObjectCfg(AssetBaseCfg):
    """Config for a single free body, e.g. SUGAR's carried box."""

    class InitialStateCfg(_InitialStateCfg):
        pass

    init_state: Any = None


class ArticulationCfg(AssetBaseCfg):
    """Config for a jointed robot."""

    class InitialStateCfg(_InitialStateCfg):
        pass

    init_state: Any = None
    actuators: dict[str, Any] | None = None
    soft_joint_pos_limit_factor: float = 1.0


class AssetBase:
    """Shared behaviour: name resolution and the update/reset hooks the managers call."""

    def __init__(self, cfg: Any, scene: Any, name: str):
        self.cfg = cfg
        self._scene = scene
        self.name = name
        self.device = scene.device

    @property
    def num_instances(self) -> int:
        return self._scene.num_envs

    def _resolve(self, keys: Sequence[str], names: Sequence[str] | str, preserve_order: bool):
        from isaaclab.utils.string import resolve_matching_names

        if isinstance(names, str):
            names = [names]
        idx, matched = resolve_matching_names(list(names), list(keys), preserve_order)
        return idx, matched

    def update(self, dt: float) -> None:
        self.data.update(dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def write_data_to_sim(self) -> None:
        """Newton reads its control arrays in place, so targets are already committed."""
        pass


class RigidObject(AssetBase):
    """A single free rigid body."""

    def __init__(self, cfg: Any, scene: Any, name: str, body_index: int, joint_coord: int):
        super().__init__(cfg, scene, name)
        self._body_index = body_index
        # Start of this body's 7 free-joint coordinates within the per-env `joint_q`.
        self._joint_coord = joint_coord
        self.body_names = [name]
        self.data = RigidObjectData(self, self.device)
        self.root_physx_view = RigidBodyPhysxView(self, scene)
        self._physics_sim_view = PhysicsSimView(self, scene)

    @property
    def num_bodies(self) -> int:
        return 1

    def find_bodies(self, name_keys, preserve_order: bool = False):
        return self._resolve(self.body_names, name_keys, preserve_order)

    # ---- reads used by RigidObjectData ------------------------------------
    def _root_body_q(self) -> torch.Tensor:
        return self._scene.body_q()[:, self._body_index]

    def _root_body_qd(self) -> torch.Tensor:
        return self._scene.body_qd()[:, self._body_index]

    # ---- writes used by reset events --------------------------------------
    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: torch.Tensor | None = None):
        """`root_pose` is (n, 7) as position + wxyz quaternion, IsaacLab's layout."""
        env_ids = self._scene.all_envs if env_ids is None else env_ids
        q = self._scene.joint_q()
        base = self._joint_coord
        q[env_ids, base : base + 3] = root_pose[:, :3]
        q[env_ids, base + 3 : base + 7] = root_pose[:, [4, 5, 6, 3]]
        self._scene.mark_kinematics_dirty()

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: torch.Tensor | None = None):
        """`root_velocity` is (n, 6) as linear + angular, matching Newton's free-joint order."""
        env_ids = self._scene.all_envs if env_ids is None else env_ids
        qd = self._scene.joint_qd()
        base = self._scene.joint_dof_of_coord(self._joint_coord)
        qd[env_ids, base : base + 6] = root_velocity
        self._scene.mark_kinematics_dirty()

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: torch.Tensor | None = None):
        self.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)


class Articulation(AssetBase):
    """A jointed robot: root body, per-body state, and actuated joints."""

    def __init__(
        self,
        cfg: Any,
        scene: Any,
        name: str,
        body_indices: torch.Tensor,
        body_names: list[str],
        joint_names: list[str],
        joint_coords: torch.Tensor,
        joint_dofs: torch.Tensor,
        root_body_index: int,
        root_joint_coord: int,
    ):
        super().__init__(cfg, scene, name)
        self._body_indices = body_indices
        self._joint_coords = joint_coords
        self._joint_dofs = joint_dofs
        self._body_index = root_body_index
        self._joint_coord = root_joint_coord
        self.body_names = body_names
        self.joint_names = joint_names
        self.data = ArticulationData(self, self.device)
        self.root_physx_view = ArticulationPhysxView(self, scene)
        self._physics_sim_view = PhysicsSimView(self, scene)

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def find_joints(self, name_keys, joint_subset=None, preserve_order: bool = False):
        keys = self.joint_names if joint_subset is None else joint_subset
        return self._resolve(keys, name_keys, preserve_order)

    def find_bodies(self, name_keys, preserve_order: bool = False):
        return self._resolve(self.body_names, name_keys, preserve_order)

    # ---- reads used by ArticulationData ------------------------------------
    def _root_body_q(self) -> torch.Tensor:
        return self._scene.body_q()[:, self._body_index]

    def _root_body_qd(self) -> torch.Tensor:
        return self._scene.body_qd()[:, self._body_index]

    def _bodies_body_q(self) -> torch.Tensor:
        return self._scene.body_q()[:, self._body_indices]

    def _bodies_body_qd(self) -> torch.Tensor:
        return self._scene.body_qd()[:, self._body_indices]

    def _joint_pos(self) -> torch.Tensor:
        return self._scene.joint_q()[:, self._joint_coords]

    def _joint_vel(self) -> torch.Tensor:
        return self._scene.joint_qd()[:, self._joint_dofs]

    def _joint_target(self) -> torch.Tensor:
        return self._scene.joint_target()[:, self._joint_dofs]

    # ---- writes -----------------------------------------------------------
    def set_joint_position_target(self, target: torch.Tensor, joint_ids=None, env_ids=None) -> None:
        """Write PD position targets. Newton's control array is read in place by the solver."""
        dofs = self._joint_dofs if joint_ids is None else self._joint_dofs[joint_ids]
        control = self._scene.joint_target()
        if env_ids is None:
            control[:, dofs] = target
        else:
            control[env_ids[:, None], dofs] = target

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None) -> None:
        coords = self._joint_coords if joint_ids is None else self._joint_coords[joint_ids]
        dofs = self._joint_dofs if joint_ids is None else self._joint_dofs[joint_ids]
        q, qd = self._scene.joint_q(), self._scene.joint_qd()
        if env_ids is None:
            q[:, coords] = position
            qd[:, dofs] = velocity
        else:
            q[env_ids[:, None], coords] = position
            qd[env_ids[:, None], dofs] = velocity
        self._scene.mark_kinematics_dirty()
        self.data.reset_joint_acc(velocity, joint_ids, env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: torch.Tensor | None = None):
        env_ids = self._scene.all_envs if env_ids is None else env_ids
        q = self._scene.joint_q()
        base = self._joint_coord
        q[env_ids, base : base + 3] = root_pose[:, :3]
        q[env_ids, base + 3 : base + 7] = root_pose[:, [4, 5, 6, 3]]
        self._scene.mark_kinematics_dirty()

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: torch.Tensor | None = None):
        env_ids = self._scene.all_envs if env_ids is None else env_ids
        qd = self._scene.joint_qd()
        base = self._scene.joint_dof_of_coord(self._joint_coord)
        qd[env_ids, base : base + 6] = root_velocity
        self._scene.mark_kinematics_dirty()

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: torch.Tensor | None = None):
        self.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)


class RigidObjectCollection(AssetBase):
    """Referenced by `SceneEntityCfg`'s isinstance checks; SUGAR does not use collections."""


class SurfaceGripper:
    """Placeholder for the suction-gripper asset.

    IsaacLab's action-term package imports this unconditionally, so the name must resolve
    even though SUGAR drives joint positions and never grips. Constructing one is an error
    rather than a no-op, since a silently inert gripper would be a physics difference.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "sugar_swap: SurfaceGripper is not implemented on the Newton backend."
        )


def build():
    """Construct the `isaaclab.assets` shadow modules."""
    import types

    def _module(name: str, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    names = dict(
        AssetBase=AssetBase,
        AssetBaseCfg=AssetBaseCfg,
        Articulation=Articulation,
        ArticulationCfg=ArticulationCfg,
        RigidObject=RigidObject,
        RigidObjectCfg=RigidObjectCfg,
        RigidObjectCollection=RigidObjectCollection,
        ArticulationData=ArticulationData,
        RigidObjectData=RigidObjectData,
        SurfaceGripper=SurfaceGripper,
    )
    surface_gripper = _module(
        "isaaclab.assets.surface_gripper",
        SurfaceGripper=SurfaceGripper,
        SurfaceGripperCfg=type("SurfaceGripperCfg", (LenientCfg,), {}),
    )
    assets = _module("isaaclab.assets", surface_gripper=surface_gripper, **names)
    # Declaring a path makes this a package, so unanticipated submodule imports fail with a
    # clear ModuleNotFoundError instead of "isaaclab.assets is not a package".
    assets.__path__ = []
    articulation = _module("isaaclab.assets.articulation", **names)
    rigid_object = _module("isaaclab.assets.rigid_object.rigid_object", **names)
    rigid_object_cfg = _module("isaaclab.assets.rigid_object.rigid_object_cfg", **names)
    rigid_pkg = _module(
        "isaaclab.assets.rigid_object",
        rigid_object=rigid_object,
        rigid_object_cfg=rigid_object_cfg,
        **names,
    )
    return {
        "isaaclab.assets": assets,
        "isaaclab.assets.surface_gripper": surface_gripper,
        "isaaclab.assets.articulation": articulation,
        "isaaclab.assets.rigid_object": rigid_pkg,
        "isaaclab.assets.rigid_object.rigid_object": rigid_object,
        "isaaclab.assets.rigid_object.rigid_object_cfg": rigid_object_cfg,
    }
