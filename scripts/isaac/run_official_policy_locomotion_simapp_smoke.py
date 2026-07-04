#!/usr/bin/env python3
"""Pure Isaac Sim official Go2 locomotion smoke.

This avoids IsaacLab AppLauncher/PhysxManager and uses Isaac Sim's
SimulationApp plus NVIDIA's installed robot policy example wrapper. Passing
this script is still only a locomotion-entry diagnostic, not box carrying.
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


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure Isaac Sim official Go2 policy smoke.")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--command", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset-root", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/official_policy_locomotion_simapp_smoke"))
    parser.add_argument("--configure-simulation-manager", action="store_true")
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
os.environ["ISAACSIM_ASSET_ROOT"] = str(args_cli.asset_root)
OV_REGISTRY_MIRROR = os.environ.get("OV_REGISTRY_MIRROR", "/public/home/yanhongru/ov_registry_mirror")
sys.argv.extend(
    [
        f"--/persistent/isaac/asset_root/default={args_cli.asset_root}",
        "--/persistent/isaac/asset_root/timeout=1.0",
        f"--/exts/omni.kit.registry.nucleus/registries/0/url={OV_REGISTRY_MIRROR}/kit_prod_default",
        f"--/exts/omni.kit.registry.nucleus/registries/1/url={OV_REGISTRY_MIRROR}/kit_prod_sdk",
    ]
)

from isaacsim import SimulationApp  # noqa: E402

DEFAULT_EXPERIENCE = (
    "/public/home/yanhongru/envs/isaac_arena_py312/lib/python3.12/site-packages/"
    "isaacsim/apps/isaacsim.exp.base.python.kit"
)
simulation_app = SimulationApp(
    {
        "headless": True,
        "hide_ui": True,
        "active_gpu": 0 if str(args_cli.device).startswith("cuda") else None,
        "physics_gpu": 0 if str(args_cli.device).startswith("cuda") else None,
    },
    experience=os.environ.get("ISAAC_SIMAPP_EXPERIENCE", DEFAULT_EXPERIENCE),
)
print("[PROGRESS] Pure SimulationApp started", flush=True)

import isaacsim  # noqa: E402

ISAAC_SITE_ROOT = Path("/public/home/yanhongru/envs/isaac_arena_py312/lib/python3.12/site-packages/isaacsim")
ISAAC_EXT_ROOT = ISAAC_SITE_ROOT / "exts"
ISAAC_DEPRECATED_EXT_ROOT = ISAAC_SITE_ROOT / "extsDeprecated"
for _ext_root, _exts in (
    (ISAAC_EXT_ROOT, ("isaacsim.robot.policy.examples", "isaacsim.core.experimental.prims", "isaacsim.core.experimental.utils")),
    (ISAAC_DEPRECATED_EXT_ROOT, ("isaacsim.core.utils",)),
):
    for _ext in _exts:
        _ext_path = _ext_root / _ext
        if _ext_path.exists() and str(_ext_path) not in sys.path:
            sys.path.insert(0, str(_ext_path))
        _pkg_path = _ext_path / "isaacsim"
        if _pkg_path.exists() and str(_pkg_path) not in isaacsim.__path__:
            isaacsim.__path__.append(str(_pkg_path))

isaacsim_core = importlib.import_module("isaacsim.core")
_legacy_core_path = ISAAC_DEPRECATED_EXT_ROOT / "isaacsim.core.utils" / "isaacsim" / "core"
if _legacy_core_path.exists() and str(_legacy_core_path) not in isaacsim_core.__path__:
    isaacsim_core.__path__.append(str(_legacy_core_path))

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.robot.policy.examples.robots.go2 import Go2FlatTerrainPolicy  # noqa: E402
import isaacsim.robot.policy.examples.robots.go2 as go2_policy_module  # noqa: E402
import isaaclab.sim.utils.stage as lab_stage_utils  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


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


def _require(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return path


def _make_ground(stage: Usd.Stage) -> None:
    cube = UsdGeom.Cube.Define(stage, "/World/ground/GroundPlane/CollisionPlane")
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.025))
    xform.SetScale(Gf.Vec3f(10.0, 10.0, 0.05))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.34, 0.34)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    material = UsdShade.Material.Define(stage, "/World/ground/Looks/PhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(1.0)
    physics_material.CreateDynamicFrictionAttr().Set(1.0)
    physics_material.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def _tensor_to_np(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _yaw_from_quat_xyzw(quat: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def run() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "official_policy_locomotion_simapp_state.csv"
    summary_path = args_cli.output_dir / "official_policy_locomotion_simapp_summary.json"

    asset_root = str(args_cli.asset_root)
    _patch_simulation_manager_compat()
    go2_policy_module.get_assets_root_path = lambda *_, **__: asset_root
    usd_path = _require(f"{asset_root}/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/go2.usda")
    policy_path = _require(f"{asset_root}/Isaac/Samples/Policies/go2/physx_policy.pt")
    env_path = _require(f"{asset_root}/Isaac/Samples/Policies/go2/physx_env.yaml")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage attached to pure SimulationApp context.")
    if not stage.GetPrimAtPath("/World").IsValid():
        stage.DefinePrim("/World", "Xform")
    if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")
    lab_stage_utils._context.stage = stage
    _make_ground(stage)
    print("[PROGRESS] Stage, PhysicsScene, and local ground ready", flush=True)

    backend = "torch" if str(args_cli.device).startswith("cuda") else "numpy"
    if args_cli.configure_simulation_manager:
        SimulationManager.set_backend(backend)
        SimulationManager.set_physics_sim_device(str(args_cli.device))
        SimulationManager.set_physics_dt(1.0 / 200.0)
        print(f"[PROGRESS] SimulationManager configured: backend={backend} device={args_cli.device}", flush=True)
    else:
        print("[PROGRESS] Skipping explicit SimulationManager configuration", flush=True)

    print("[PROGRESS] Importing torch", flush=True)
    torch = import_module("torch")
    print("[PROGRESS] Torch imported", flush=True)
    command = torch.tensor(list(args_cli.command), dtype=torch.float32, device=torch.device(args_cli.device))
    print("[PROGRESS] Command tensor created", flush=True)
    try:
        robot = Go2FlatTerrainPolicy(
            prim_path="/World/Go2",
            usd_path=usd_path,
            position=[0.0, 0.0, 0.50],
            policy_path=policy_path,
            env_config_path=env_path,
        )
    except BaseException as exc:
        summary_path.write_text(
            json.dumps(
                {
                    "scene_type": "pure_isaacsim_official_go2_policy_locomotion_smoke",
                    "success_claim": "failed_before_robot_policy_construction_completed",
                    "device": args_cli.device,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"[ERROR] Policy robot construction failed: {type(exc).__name__}: {exc}", flush=True)
        return summary_path
    print("[PROGRESS] Go2 policy object created", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    _update_app(steps=8)
    if hasattr(SimulationManager, "_warmup_and_create_views"):
        print("[PROGRESS] Explicit SimulationManager warmup/view creation", flush=True)
        try:
            with stage_utils.use_stage(stage):
                SimulationManager._warmup_and_create_views()
        except Exception as exc:
            print(f"[WARN] Explicit warmup/view creation failed: {exc}", flush=True)
        _update_app(steps=4)
    valid_fn = getattr(robot.robot, "is_physics_tensor_entity_valid", None)
    valid = bool(valid_fn()) if callable(valid_fn) else None
    print(f"[PROGRESS] Pre-initialize physics view valid={valid}", flush=True)
    if valid is False:
        summary_path.write_text(json.dumps({"error": "physics_tensor_entity_invalid", "device": args_cli.device}, indent=2) + "\n")
        timeline.stop()
        return summary_path
    robot.initialize()
    _update_app(steps=4)
    print("[PROGRESS] Go2 initialized", flush=True)

    positions, orientations = robot.robot.get_world_poses()
    start_pos = _tensor_to_np(positions)[0].astype(float)
    start_yaw = _yaw_from_quat_xyzw(_tensor_to_np(orientations)[0].astype(float))
    summary = {
        "scene_type": "pure_isaacsim_official_go2_policy_locomotion_smoke",
        "success_claim": "locomotion_entry_diagnostic_only_not_box_carrying",
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "travel_xy_m": 0.0,
        "fall_events": 0,
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "z", "travel_xy_m", "yaw_delta_rad", "fall"])
            for step in range(int(args_cli.steps)):
                robot.forward(1.0 / 200.0, command)
                _update_app()
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    positions, orientations = robot.robot.get_world_poses()
                    pos = _tensor_to_np(positions)[0].astype(float)
                    yaw = _yaw_from_quat_xyzw(_tensor_to_np(orientations)[0].astype(float))
                    travel = float(np.linalg.norm(pos[:2] - start_pos[:2]))
                    fall = int(pos[2] < 0.20)
                    summary["completed_steps"] = step + 1
                    summary["travel_xy_m"] = travel
                    summary["fall_events"] += fall
                    writer.writerow([step, pos[0], pos[1], pos[2], travel, abs(yaw - start_yaw), fall])
                    print(f"[STATE] step={step} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) travel={travel:.3f} fall={fall}", flush=True)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)
    finally:
        timeline.stop()

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
