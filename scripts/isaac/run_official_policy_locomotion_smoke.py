#!/usr/bin/env python3
"""Headless official Isaac Sim locomotion policy smoke.

This ports the installed NVIDIA robot policy example tests into a standalone
diagnostic.  It is meant to verify a known-good official locomotion entry point
before adding a carrying payload.  Passing this script is not a box-carrying
success claim.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official Isaac policy locomotion smoke.")
    parser.add_argument("--robot", choices=("h1", "go2"), default="h1")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--command", type=float, nargs=3, default=(1.0, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--asset-root", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--payload-mode", choices=("none", "fixed_base"), default="none")
    parser.add_argument("--payload-mass", type=float, default=2.0)
    parser.add_argument("--payload-size", type=float, nargs=3, default=(0.28, 0.18, 0.14), metavar=("X", "Y", "Z"))
    parser.add_argument("--payload-offset", type=float, nargs=3, default=(0.22, 0.0, 0.08), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--configure-simulation-manager",
        action="store_true",
        help="Use explicit SimulationManager backend/device setup. Disabled by default because it exits early in standalone CPU runs on this cluster.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/official_policy_locomotion_smoke"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
os.environ["ISAACSIM_ASSET_ROOT"] = str(args_cli.asset_root)
print("[PROGRESS] Parsed args and set ISAACSIM_ASSET_ROOT", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import isaacsim  # noqa: E402
print("[PROGRESS] Imported isaacsim package", flush=True)

ISAAC_SITE_ROOT = Path("/public/home/yanhongru/envs/isaac_arena_py312/lib/python3.12/site-packages/isaacsim")
ISAAC_EXT_ROOT = ISAAC_SITE_ROOT / "exts"
ISAAC_DEPRECATED_EXT_ROOT = ISAAC_SITE_ROOT / "extsDeprecated"
for _ext in (
    "isaacsim.robot.policy.examples",
    "isaacsim.core.experimental.prims",
    "isaacsim.core.experimental.utils",
):
    _ext_path = ISAAC_EXT_ROOT / _ext
    if _ext_path.exists() and str(_ext_path) not in sys.path:
        sys.path.insert(0, str(_ext_path))
    _pkg_path = _ext_path / "isaacsim"
    if _pkg_path.exists() and str(_pkg_path) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_pkg_path))
_legacy_ext_path = ISAAC_DEPRECATED_EXT_ROOT / "isaacsim.core.utils"
if _legacy_ext_path.exists() and str(_legacy_ext_path) not in sys.path:
    sys.path.insert(0, str(_legacy_ext_path))
_legacy_pkg_path = _legacy_ext_path / "isaacsim"
if _legacy_pkg_path.exists() and str(_legacy_pkg_path) not in isaacsim.__path__:
    isaacsim.__path__.append(str(_legacy_pkg_path))

import numpy as np  # noqa: E402
print("[PROGRESS] Added Isaac extension namespace paths", flush=True)
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaaclab.physics import PhysicsManager  # noqa: E402
isaacsim_core = importlib.import_module("isaacsim.core")  # noqa: E402
_legacy_core_path = _legacy_pkg_path / "core"
if _legacy_core_path.exists() and str(_legacy_core_path) not in isaacsim_core.__path__:
    isaacsim_core.__path__.append(str(_legacy_core_path))
import isaacsim.robot.policy.examples.robots.go2 as go2_policy_module  # noqa: E402
import isaacsim.robot.policy.examples.robots.h1 as h1_policy_module  # noqa: E402
from isaacsim.robot.policy.examples.robots.go2 import Go2FlatTerrainPolicy  # noqa: E402
from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402
print("[PROGRESS] Policy smoke imports complete", flush=True)


def _local_asset_root() -> str:
    root = get_assets_root_path(skip_check=True)
    if not Path(root).exists():
        raise RuntimeError(f"ISAACSIM_ASSET_ROOT does not exist: {root}")
    return root


def _require(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return path


def _make_ground(stage, path: str = "/World/ground") -> None:
    cube = UsdGeom.Cube.Define(stage, f"{path}/GroundPlane/CollisionPlane")
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.025))
    xform.SetScale(Gf.Vec3f(10.0, 10.0, 0.05))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.34, 0.34)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    material = UsdShade.Material.Define(stage, f"{path}/Looks/PhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(1.0)
    physics_material.CreateDynamicFrictionAttr().Set(1.0)
    physics_material.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def _set_xform(prim: Usd.Prim, translation: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def _world_translation(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found for world pose read: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = matrix.ExtractTranslation()
    return np.asarray([float(t[0]), float(t[1]), float(t[2])], dtype=float)


def _spawn_fixed_base_payload(stage, robot_prim_path: str) -> tuple[str, str]:
    if args_cli.robot != "go2":
        raise RuntimeError("--payload-mode fixed_base is currently implemented only for Go2.")

    base_link_path = f"{robot_prim_path}/Geometry/base"
    if not stage.GetPrimAtPath(base_link_path).IsValid():
        raise RuntimeError(f"Go2 base link not found for fixed payload: {base_link_path}")

    payload_path = "/World/CarryBox"
    payload = UsdGeom.Cube.Define(stage, payload_path)
    payload.CreateSizeAttr(1.0)
    base_pos = _world_translation(stage, base_link_path)
    offset = np.asarray(args_cli.payload_offset, dtype=float)
    size = tuple(float(v) for v in args_cli.payload_size)
    _set_xform(payload.GetPrim(), tuple((base_pos + offset).tolist()), size)
    payload.CreateDisplayColorAttr([Gf.Vec3f(0.58, 0.43, 0.24)])
    UsdPhysics.CollisionAPI.Apply(payload.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(payload.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(payload.GetPrim())
    mass_api.CreateMassAttr(float(args_cli.payload_mass))

    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Go2BasePayloadFixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base_link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(payload_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in args_cli.payload_offset]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    return payload_path, base_link_path


def _yaw_from_quat_xyzw(quat: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _tensor_to_np(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _update_app(steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
        simulation_app.update()


def _patch_simulation_manager_compat() -> None:
    if not hasattr(SimulationManager, "get_active_physics_engine"):
        SimulationManager.get_active_physics_engine = classmethod(lambda cls: "physx")
    if not hasattr(SimulationManager, "_physics_sim_view__warp"):
        SimulationManager._physics_sim_view__warp = getattr(SimulationManager, "_view_warp", None)
    if not hasattr(SimulationManager, "_physics_sim_interface"):
        SimulationManager._physics_sim_interface = getattr(SimulationManager, "_physx_sim", None)
    if not hasattr(SimulationManager, "_physics_stage_update_interface"):
        SimulationManager._physics_stage_update_interface = None


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "official_policy_locomotion_state.csv"
    summary_path = args_cli.output_dir / "official_policy_locomotion_summary.json"

    asset_root = _local_asset_root()
    _patch_simulation_manager_compat()
    go2_policy_module.get_assets_root_path = lambda *_, **__: asset_root
    h1_policy_module.get_assets_root_path = lambda *_, **__: asset_root
    print(f"[PROGRESS] Using asset root: {asset_root}", flush=True)
    if args_cli.robot == "h1":
        usd_path = _require(f"{asset_root}/Isaac/Robots/Unitree/H1/h1.usd")
        policy_path = _require(f"{asset_root}/Isaac/Samples/Policies/h1/physx_policy.pt")
        env_path = _require(f"{asset_root}/Isaac/Samples/Policies/h1/physx_env.yaml")
        policy_cls = H1FlatTerrainPolicy
        prim_path = "/World/H1"
        spawn_z = 1.05
        expected_dofs = 19
        fall_z = 0.55
    else:
        usd_path = _require(f"{asset_root}/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/go2.usda")
        policy_path = _require(f"{asset_root}/Isaac/Samples/Policies/go2/physx_policy.pt")
        env_path = _require(f"{asset_root}/Isaac/Samples/Policies/go2/physx_env.yaml")
        policy_cls = Go2FlatTerrainPolicy
        prim_path = "/World/Go2"
        spawn_z = 0.50
        expected_dofs = 12
        fall_z = 0.20

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is attached to the current Isaac Sim context.")
    _update_app()
    print("[PROGRESS] Reusing current USD stage", flush=True)
    physics_dt = 1.0 / 200.0
    device_str = str(args_cli.device)
    torch_device = "cuda" if device_str.startswith("cuda") else "cpu"
    backend = "torch" if torch_device == "cuda" else "numpy"
    if args_cli.configure_simulation_manager:
        print(f"[PROGRESS] Configuring SimulationManager: backend={backend} device={device_str}", flush=True)
        SimulationManager.set_backend(backend)
        print("[PROGRESS] SimulationManager backend set", flush=True)
        SimulationManager.set_physics_sim_device(device_str)
        print("[PROGRESS] SimulationManager physics device set", flush=True)
        SimulationManager.set_physics_dt(physics_dt)
        print("[PROGRESS] SimulationManager physics dt set", flush=True)
    else:
        print("[PROGRESS] Skipping explicit SimulationManager backend/device setup", flush=True)
        if not device_str.startswith("cuda"):
            try:
                SimulationManager.set_physics_sim_device(device_str)
                print(f"[PROGRESS] SimulationManager physics device set without backend: {device_str}", flush=True)
            except Exception as exc:
                print(f"[WARN] Could not set SimulationManager physics device without backend: {exc}", flush=True)
            PhysicsManager._device = device_str
            print(f"[PROGRESS] IsaacLab PhysicsManager device forced to {device_str}", flush=True)
    if not stage.GetPrimAtPath("/World").IsValid():
        stage.DefinePrim("/World", "Xform")
        print("[PROGRESS] /World defined", flush=True)
    if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")
        print("[PROGRESS] PhysicsScene defined", flush=True)
    print("[PROGRESS] Creating local ground", flush=True)
    _make_ground(stage)
    print("[PROGRESS] Local ground created", flush=True)

    print("[PROGRESS] Importing torch through Isaac deprecation manager", flush=True)
    torch = import_module("torch")
    print("[PROGRESS] Torch imported", flush=True)
    command = torch.tensor(list(args_cli.command), dtype=torch.float32, device=torch.device(torch_device))
    print("[PROGRESS] Command tensor created", flush=True)
    print(f"[PROGRESS] Creating policy robot: usd={usd_path}", flush=True)
    try:
        robot = policy_cls(
            prim_path=prim_path,
            usd_path=usd_path,
            position=[0.0, 0.0, spawn_z],
            policy_path=policy_path,
            env_config_path=env_path,
        )
    except BaseException as exc:
        summary_path.write_text(
            json.dumps(
                {
                    "scene_type": "official_isaac_policy_locomotion_smoke",
                    "success_claim": "failed_before_robot_policy_construction_completed",
                    "robot": args_cli.robot,
                    "asset_root": asset_root,
                    "usd_path": usd_path,
                    "policy_path": policy_path,
                    "env_config_path": env_path,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"[ERROR] Policy robot construction failed: {type(exc).__name__}: {exc}", flush=True)
        return summary_path
    print(f"[PROGRESS] Policy robot object created: {args_cli.robot}", flush=True)
    _update_app()

    payload_path = None
    payload_anchor_path = None
    if args_cli.payload_mode == "fixed_base":
        payload_path, payload_anchor_path = _spawn_fixed_base_payload(stage, prim_path)
        print(
            "[PROGRESS] Fixed-base payload created: "
            f"path={payload_path} anchor={payload_anchor_path} mass={args_cli.payload_mass}",
            flush=True,
        )
        _update_app()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("[PROGRESS] Timeline play", flush=True)
    _update_app(steps=4)
    physics_valid_fn = getattr(robot.robot, "is_physics_tensor_entity_valid", None)
    physics_initialized_fn = getattr(robot.robot, "is_physics_tensor_entity_initialized", None)
    physics_valid = bool(physics_valid_fn()) if callable(physics_valid_fn) else None
    physics_initialized = bool(physics_initialized_fn()) if callable(physics_initialized_fn) else None
    print(
        f"[PROGRESS] Pre-initialize physics view: valid={physics_valid} initialized={physics_initialized}",
        flush=True,
    )
    if physics_valid is False:
        summary_path.write_text(
            json.dumps(
                {
                    "scene_type": "official_isaac_policy_locomotion_smoke",
                    "success_claim": "failed_before_robot_policy_initialization",
                    "robot": args_cli.robot,
                    "asset_root": asset_root,
                    "usd_path": usd_path,
                    "policy_path": policy_path,
                    "env_config_path": env_path,
                    "device": device_str,
                    "configured_simulation_manager": bool(args_cli.configure_simulation_manager),
                    "physics_tensor_entity_valid": physics_valid,
                    "physics_tensor_entity_initialized": physics_initialized,
                    "error": "Articulation physics tensor entity is invalid before policy initialization.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print("[ERROR] Articulation physics tensor entity invalid before initialize", flush=True)
        timeline.stop()
        return summary_path
    robot.initialize()
    print("[PROGRESS] Robot policy initialized", flush=True)
    _update_app()

    callback_id = None

    positions, orientations = robot.robot.get_world_poses()
    start_pos = _tensor_to_np(positions)[0].astype(float)
    start_quat = _tensor_to_np(orientations)[0].astype(float)
    start_yaw = _yaw_from_quat_xyzw(start_quat)

    summary = {
        "scene_type": "official_isaac_policy_locomotion_smoke",
        "success_claim": "locomotion_entry_diagnostic_only_not_box_carrying",
        "robot": args_cli.robot,
        "asset_root": asset_root,
        "usd_path": usd_path,
        "policy_path": policy_path,
        "env_config_path": env_path,
        "payload_mode": args_cli.payload_mode,
        "payload_path": payload_path,
        "payload_anchor_path": payload_anchor_path,
        "payload_mass_kg": float(args_cli.payload_mass) if args_cli.payload_mode != "none" else 0.0,
        "payload_size_m": [float(v) for v in args_cli.payload_size],
        "payload_offset_m": [float(v) for v in args_cli.payload_offset],
        "device": device_str,
        "backend": backend,
        "configured_simulation_manager": bool(args_cli.configure_simulation_manager),
        "expected_dofs": expected_dofs,
        "num_dofs": int(robot.robot.num_dofs),
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "command": [float(v) for v in args_cli.command],
        "start_pos": start_pos.tolist(),
        "final_pos": None,
        "travel_xy_m": 0.0,
        "yaw_delta_rad": 0.0,
        "min_base_z_m": float(start_pos[2]),
        "initial_payload_pos": None,
        "final_payload_pos": None,
        "payload_travel_xy_m": 0.0,
        "payload_drop_events": 0,
        "fall_events": 0,
        "error": None,
    }
    initial_payload_pos = None
    if payload_path is not None:
        initial_payload_pos = _world_translation(stage, payload_path)
        summary["initial_payload_pos"] = initial_payload_pos.tolist()

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "x",
                    "y",
                    "z",
                    "travel_xy_m",
                    "yaw_delta_rad",
                    "payload_x",
                    "payload_y",
                    "payload_z",
                    "payload_travel_xy_m",
                    "fall",
                    "payload_drop",
                ]
            )
            for step in range(int(args_cli.steps)):
                robot.forward(physics_dt, command)
                _update_app()
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    positions, orientations = robot.robot.get_world_poses()
                    pos = _tensor_to_np(positions)[0].astype(float)
                    quat = _tensor_to_np(orientations)[0].astype(float)
                    travel = float(np.linalg.norm(pos[:2] - start_pos[:2]))
                    yaw_delta = float(abs(_yaw_from_quat_xyzw(quat) - start_yaw))
                    fall = int(pos[2] < fall_z)
                    summary["completed_steps"] = int(step + 1)
                    summary["final_pos"] = pos.tolist()
                    summary["travel_xy_m"] = travel
                    summary["yaw_delta_rad"] = yaw_delta
                    summary["min_base_z_m"] = min(float(summary["min_base_z_m"]), float(pos[2]))
                    summary["fall_events"] += fall
                    payload_pos = np.asarray([math.nan, math.nan, math.nan], dtype=float)
                    payload_travel = 0.0
                    payload_drop = 0
                    if payload_path is not None and initial_payload_pos is not None:
                        payload_pos = _world_translation(stage, payload_path)
                        payload_travel = float(np.linalg.norm(payload_pos[:2] - initial_payload_pos[:2]))
                        payload_drop = int(payload_pos[2] < 0.15)
                        summary["final_payload_pos"] = payload_pos.tolist()
                        summary["payload_travel_xy_m"] = payload_travel
                        summary["payload_drop_events"] += payload_drop
                    writer.writerow(
                        [
                            step,
                            step * physics_dt,
                            pos[0],
                            pos[1],
                            pos[2],
                            travel,
                            yaw_delta,
                            payload_pos[0],
                            payload_pos[1],
                            payload_pos[2],
                            payload_travel,
                            fall,
                            payload_drop,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} robot={args_cli.robot} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                        f"travel={travel:.3f} payload_travel={payload_travel:.3f} "
                        f"yaw_delta={yaw_delta:.3f} fall={fall} payload_drop={payload_drop}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)
    finally:
        try:
            if callback_id is not None:
                SimulationManager.deregister_callback(callback_id)
        except Exception:
            pass
        timeline.stop()

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")
    return summary_path


if __name__ == "__main__":
    try:
        run_scene()
    finally:
        simulation_app.close()
