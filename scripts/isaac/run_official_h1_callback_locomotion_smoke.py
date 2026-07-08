#!/usr/bin/env python3
"""Official H1 callback locomotion smoke.

This follows NVIDIA's installed H1 policy test pattern: create an H1 policy
wrapper, play the timeline, initialize the articulation, and call forward()
from a POST_PHYSICS_STEP callback. Passing this is only a locomotion backend
candidate. It is not box carrying, active probing, RL, or video-conditioned
evidence.
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
    parser = argparse.ArgumentParser(description="Official H1 callback locomotion smoke.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--command", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset-root", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/official_h1_callback_locomotion_smoke"))
    parser.add_argument(
        "--official-test-kit-args",
        action="store_true",
        help="Inject key Kit settings used by the installed isaacsim.robot.policy.examples tests.",
    )
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
os.environ["ISAACSIM_ASSET_ROOT"] = str(args_cli.asset_root)
OV_REGISTRY_MIRROR = os.environ.get("OV_REGISTRY_MIRROR", "/public/home/yanhongru/ov_registry_mirror")
kit_args = [
    f"--/persistent/isaac/asset_root/default={args_cli.asset_root}",
    "--/persistent/isaac/asset_root/timeout=1.0",
    f"--/exts/omni.kit.registry.nucleus/registries/0/url={OV_REGISTRY_MIRROR}/kit_prod_default",
    f"--/exts/omni.kit.registry.nucleus/registries/1/url={OV_REGISTRY_MIRROR}/kit_prod_sdk",
]
if args_cli.official_test_kit_args:
    kit_args.extend(
        [
            "--enable",
            "omni.kit.loop-isaac",
            "--reset-user",
            "--vulkan",
            "--/app/asyncRendering=0",
            "--/app/asyncRenderingLowLatency=0",
            "--/app/fastShutdown=1",
            "--/app/file/ignoreUnsavedStage=1",
            "--/app/player/useFixedTimeStepping=false",
            "--/app/runLoops/main/manualModeEnabled=true",
            "--/app/runLoops/main/rateLimitEnabled=false",
            "--/app/settings/persistent=0",
            "--/app/useFabricSceneDelegate=true",
            "--/omni/kit/plugin/syncUsdLoads=1",
            "--/persistent/app/stage/upAxis=Z",
            "--/persistent/simulation/defaultMetersPerUnit=1.0",
            "--/persistent/simulation/minFrameRate=15",
            "--/renderer/multiGpu/autoEnable=0",
            "--/renderer/multiGpu/enabled=0",
        ]
    )

# Keep script arguments out of Kit's unknown-argument parser.
sys.argv = [sys.argv[0]]

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
        "extra_args": kit_args,
    },
    experience=os.environ.get("ISAAC_SIMAPP_EXPERIENCE", DEFAULT_EXPERIENCE),
)
print("[PROGRESS] SimulationApp started", flush=True)

import isaacsim  # noqa: E402

ISAAC_SITE_ROOT = Path("/public/home/yanhongru/envs/isaac_arena_py312/lib/python3.12/site-packages/isaacsim")
ISAAC_EXT_ROOT = ISAAC_SITE_ROOT / "exts"
ISAAC_DEPRECATED_EXT_ROOT = ISAAC_SITE_ROOT / "extsDeprecated"
for _ext_root, _exts in (
    (
        ISAAC_EXT_ROOT,
        (
            "isaacsim.robot.policy.examples",
            "isaacsim.core.experimental.prims",
            "isaacsim.core.experimental.utils",
        ),
    ),
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
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy  # noqa: E402
import isaacsim.robot.policy.examples.robots.h1 as h1_policy_module  # noqa: E402
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
    xform.SetScale(Gf.Vec3f(12.0, 12.0, 0.05))
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
    csv_path = args_cli.output_dir / "official_h1_callback_locomotion_state.csv"
    summary_path = args_cli.output_dir / "official_h1_callback_locomotion_summary.json"

    asset_root = str(args_cli.asset_root)
    _patch_simulation_manager_compat()
    h1_policy_module.get_assets_root_path = lambda *_, **__: asset_root
    _require(f"{asset_root}/Isaac/Robots/Unitree/H1/h1.usd")
    _require(f"{asset_root}/Isaac/Samples/Policies/h1/physx_policy.pt")
    _require(f"{asset_root}/Isaac/Samples/Policies/h1/physx_env.yaml")

    print("[PROGRESS] Creating fresh stage", flush=True)
    stage = stage_utils.create_new_stage(template="empty")
    _update_app(steps=2)
    if not stage.GetPrimAtPath("/World").IsValid():
        stage.DefinePrim("/World", "Xform")
    if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")
    _make_ground(stage)
    print("[PROGRESS] Stage, PhysicsScene, and local ground ready", flush=True)

    print("[PROGRESS] Importing torch and constructing H1 policy object", flush=True)
    torch = import_module("torch")
    command = torch.tensor(list(args_cli.command), dtype=torch.float32, device=torch.device(args_cli.device))
    try:
        robot = H1FlatTerrainPolicy(prim_path="/World/H1", position=[0.0, 0.0, 1.05])
    except BaseException as exc:
        summary_path.write_text(
            json.dumps(
                {
                    "scene_type": "official_h1_callback_locomotion_smoke",
                    "success_claim": "failed_before_h1_policy_construction_completed",
                    "device": str(args_cli.device),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"[ERROR] H1 policy construction failed: {type(exc).__name__}: {exc}", flush=True)
        return summary_path
    print("[PROGRESS] H1 policy object created", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    callback_state = {"ready": False, "init_attempts": 0, "forward_calls": 0, "errors": []}

    def on_physics_step(step_size: float, context: object) -> None:
        try:
            valid_fn = getattr(robot.robot, "is_physics_tensor_entity_valid", None)
            valid = bool(valid_fn()) if callable(valid_fn) else True
            if not valid:
                callback_state["ready"] = False
            if callback_state["ready"]:
                robot.forward(step_size, command)
                callback_state["forward_calls"] += 1
            else:
                callback_state["init_attempts"] += 1
                robot.initialize()
                callback_state["ready"] = True
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            if len(callback_state["errors"]) < 8:
                callback_state["errors"].append(message)
            callback_state["ready"] = False

    timeline.play()
    _update_app(steps=int(args_cli.warmup_steps))
    callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
    _update_app(steps=2)

    start_pos = None
    start_yaw = 0.0
    summary = {
        "scene_type": "official_h1_callback_locomotion_smoke",
        "success_claim": "locomotion_backend_candidate_only_not_box_carrying",
        "device": str(args_cli.device),
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "callback_init_attempts": 0,
        "callback_forward_calls": 0,
        "callback_errors": [],
        "travel_xy_m": 0.0,
        "max_travel_xy_m": 0.0,
        "min_base_z_m": None,
        "max_yaw_delta_rad": 0.0,
        "fall_events": 0,
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "z", "travel_xy_m", "yaw_delta_rad", "fall", "forward_calls"])
            for step in range(int(args_cli.steps)):
                _update_app()
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    positions, orientations = robot.robot.get_world_poses()
                    pos = _tensor_to_np(positions)[0].astype(float)
                    quat = _tensor_to_np(orientations)[0].astype(float)
                    if start_pos is None:
                        start_pos = pos.copy()
                        start_yaw = _yaw_from_quat_xyzw(quat)
                    yaw = _yaw_from_quat_xyzw(quat)
                    travel = float(np.linalg.norm(pos[:2] - start_pos[:2]))
                    yaw_delta = abs(float(yaw - start_yaw))
                    fall = int(float(pos[2]) < 0.45)
                    summary["completed_steps"] = step + 1
                    summary["callback_init_attempts"] = int(callback_state["init_attempts"])
                    summary["callback_forward_calls"] = int(callback_state["forward_calls"])
                    summary["callback_errors"] = list(callback_state["errors"])
                    summary["travel_xy_m"] = travel
                    summary["max_travel_xy_m"] = max(float(summary["max_travel_xy_m"]), travel)
                    summary["min_base_z_m"] = (
                        float(pos[2]) if summary["min_base_z_m"] is None else min(float(summary["min_base_z_m"]), float(pos[2]))
                    )
                    summary["max_yaw_delta_rad"] = max(float(summary["max_yaw_delta_rad"]), yaw_delta)
                    summary["fall_events"] += fall
                    writer.writerow([step, pos[0], pos[1], pos[2], travel, yaw_delta, fall, callback_state["forward_calls"]])
                    print(
                        f"[STATE] step={step} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                        f"travel={travel:.3f} forward_calls={callback_state['forward_calls']} fall={fall}",
                        flush=True,
                    )
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)
    finally:
        try:
            SimulationManager.deregister_callback(callback_id)
        except Exception:
            pass
        timeline.stop()

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
