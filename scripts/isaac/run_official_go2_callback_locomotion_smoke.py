#!/usr/bin/env python3
"""Official Go2 callback locomotion smoke.

This follows NVIDIA's installed Go2 policy tests more closely than the earlier
manual-loop smoke: create a fresh stage, configure SimulationManager physics
device/dt, start the timeline, initialize the policy from a
POST_PHYSICS_STEP callback, and drive commands from that callback.

Passing this is only robot locomotion evidence. It is not box carrying unless
the optional fixed-payload mode is separately verified.
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
    parser = argparse.ArgumentParser(description="Official Go2 callback locomotion smoke.")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--command", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--simulation-manager-mode",
        choices=("official_device_dt", "skip_device_dt", "backend_device_dt", "dt_only", "device_only"),
        default="official_device_dt",
    )
    parser.add_argument(
        "--official-test-kit-args",
        action="store_true",
        help="Inject the key Kit settings from NVIDIA's isaacsim.robot.policy.examples physx test config.",
    )
    parser.add_argument("--payload-mode", choices=("none", "fixed_base"), default="none")
    parser.add_argument("--payload-mass", type=float, default=2.0)
    parser.add_argument("--asset-root", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/official_go2_callback_locomotion_smoke"))
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
            "--/app/hydraEngine/waitIdle=0",
            "--/app/player/useFixedTimeStepping=false",
            "--/app/renderer/skipWhileMinimized=0",
            "--/app/renderer/sleepMsOnFocus=0",
            "--/app/renderer/sleepMsOutOfFocus=0",
            "--/app/runLoops/main/manualModeEnabled=true",
            "--/app/runLoops/main/rateLimitEnabled=false",
            "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
            "--/app/settings/persistent=0",
            "--/app/useFabricSceneDelegate=true",
            "--/app/viewport/createCameraModelRep=0",
            "--/crashreporter/skipOldDumpUpload=1",
            "--/exts/omni.usd/locking/onClose=0",
            "--/omni/kit/plugin/syncUsdLoads=1",
            "--/omni/replicator/asyncRendering=0",
            "--/omnihydra/parallelHydraSprimSync=1",
            '--/persistent/app/stage/upAxis="Z"',
            "--/persistent/app/viewport/defaults/tickRate=120",
            "--/persistent/app/viewport/displayOptions=31951",
            "--/persistent/omni/replicator/captureOnPlay=1",
            "--/persistent/omnigraph/updateToUsd=0",
            "--/persistent/physics/visualizationDisplayJoints=0",
            "--/persistent/renderer/startupMessageDisplayed=1",
            "--/persistent/simulation/defaultMetersPerUnit=1.0",
            "--/persistent/simulation/minFrameRate=15",
            "--/renderer/multiGpu/autoEnable=0",
            "--/renderer/multiGpu/enabled=0",
            "--/rtx-transient/dlssg/enabled=0",
            "--/rtx-transient/resourcemanager/enableTextureStreaming=1",
            "--/rtx/descriptorSets=360000",
            "--/rtx/hydra/enableSemanticSchema=1",
            "--/rtx/hydra/materialSyncLoads=1",
            "--/rtx/hydra/supportMultiTickRate=true",
            "--/rtx/materialDb/syncLoads=1",
            "--/rtx/newDenoiser/enabled=1",
            "--/rtx/rendering/perSensorTickTlas=true",
            "--/rtx/reservedDescriptors=900000",
        ]
    )
# SimulationApp only forwards launch_config["extra_args"] into the actual Kit
# startup args.  Keep parsed script args out of Kit's unknown-arg list.
sys.argv = [sys.argv[0]]

from isaacsim import SimulationApp  # noqa: E402

DEFAULT_EXPERIENCE = "/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit"
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
import omni.usd  # noqa: E402
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.robot.policy.examples.robots.go2 import Go2FlatTerrainPolicy  # noqa: E402
import isaacsim.robot.policy.examples.robots.go2 as go2_policy_module  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


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


def _make_fixed_payload(stage: Usd.Stage) -> None:
    box = UsdGeom.Cube.Define(stage, "/World/CarryBox")
    box.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(box.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.SetScale(Gf.Vec3f(0.28, 0.20, 0.18))
    box.CreateDisplayColorAttr([Gf.Vec3f(0.58, 0.43, 0.24)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(box.GetPrim())
    mass_api.CreateMassAttr(float(args_cli.payload_mass))
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Go2CarryBoxFixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Go2/Geometry/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/CarryBox")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.16))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _tensor_to_np(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _yaw_from_quat_xyzw(quat: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _usd_translation(stage: Usd.Stage, prim_path: str) -> list[float] | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    return [float(translation[0]), float(translation[1]), float(translation[2])]


def run() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "official_go2_callback_locomotion_state.csv"
    summary_path = args_cli.output_dir / "official_go2_callback_locomotion_summary.json"

    asset_root = str(args_cli.asset_root)
    _patch_simulation_manager_compat()
    go2_policy_module.get_assets_root_path = lambda *_, **__: asset_root
    usd_path = _require(f"{asset_root}/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/go2.usda")
    policy_path = _require(f"{asset_root}/Isaac/Samples/Policies/go2/physx_policy.pt")
    env_path = _require(f"{asset_root}/Isaac/Samples/Policies/go2/physx_env.yaml")

    print("[PROGRESS] Creating fresh stage", flush=True)
    stage = stage_utils.create_new_stage(template="empty")
    _update_app(steps=2)
    if not stage.GetPrimAtPath("/World").IsValid():
        stage.DefinePrim("/World", "Xform")
    if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")
    _make_ground(stage)
    print("[PROGRESS] Stage, PhysicsScene, and local ground ready", flush=True)

    device_str = str(args_cli.device)
    if args_cli.simulation_manager_mode == "backend_device_dt":
        backend = "torch" if device_str.startswith("cuda") else "numpy"
        print(f"[PROGRESS] Setting SimulationManager backend={backend}", flush=True)
        SimulationManager.set_backend(backend)
        print("[PROGRESS] SimulationManager backend set", flush=True)
    if args_cli.simulation_manager_mode in ("official_device_dt", "backend_device_dt", "device_only"):
        print(f"[PROGRESS] Setting SimulationManager physics device={device_str}", flush=True)
        SimulationManager.set_physics_sim_device(device_str)
        print("[PROGRESS] SimulationManager physics device set", flush=True)
    if args_cli.simulation_manager_mode in ("official_device_dt", "backend_device_dt", "dt_only"):
        print("[PROGRESS] Setting SimulationManager physics dt=0.005", flush=True)
        SimulationManager.set_physics_dt(1.0 / 200.0)
        print("[PROGRESS] SimulationManager physics dt set", flush=True)
    if args_cli.simulation_manager_mode == "skip_device_dt":
        print("[PROGRESS] Skipping SimulationManager device/dt setters", flush=True)

    torch = import_module("torch")
    command = torch.tensor(list(args_cli.command), dtype=torch.float32, device=torch.device(device_str))
    robot = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        usd_path=usd_path,
        position=[0.0, 0.0, 0.50],
        policy_path=policy_path,
        env_config_path=env_path,
    )
    if args_cli.payload_mode == "fixed_base":
        _make_fixed_payload(stage)
        print("[PROGRESS] Fixed payload authored", flush=True)
    print("[PROGRESS] Go2 policy object created", flush=True)

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
                robot.post_reset()
                callback_state["ready"] = True
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message not in callback_state["errors"]:
                callback_state["errors"].append(message)
            callback_state["ready"] = False

    timeline.play()
    _update_app(steps=max(1, int(args_cli.warmup_steps)))
    callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
    print("[PROGRESS] Physics callback registered", flush=True)
    _update_app(steps=4)

    start_pos = None
    start_yaw = None
    payload_start = _usd_translation(stage, "/World/CarryBox")
    summary = {
        "scene_type": "official_go2_callback_locomotion_smoke",
        "success_claim": "robot_locomotion_callback_diagnostic_only_not_free_object_carrying",
        "device": device_str,
        "official_test_kit_args": bool(args_cli.official_test_kit_args),
        "simulation_manager_mode": args_cli.simulation_manager_mode,
        "payload_mode": args_cli.payload_mode,
        "payload_mass_kg": float(args_cli.payload_mass) if args_cli.payload_mode != "none" else 0.0,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "travel_xy_m": 0.0,
        "payload_travel_xy_m": 0.0,
        "fall_events": 0,
        "callback_init_attempts": 0,
        "callback_forward_calls": 0,
        "callback_errors": [],
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "z", "travel_xy_m", "yaw_delta_rad", "payload_travel_xy_m", "fall"])
            for step in range(int(args_cli.steps)):
                _update_app()
                positions, orientations = robot.robot.get_world_poses()
                pos = _tensor_to_np(positions)[0].astype(float)
                quat = _tensor_to_np(orientations)[0].astype(float)
                if start_pos is None:
                    start_pos = pos.copy()
                    start_yaw = _yaw_from_quat_xyzw(quat)
                travel = float(np.linalg.norm(pos[:2] - start_pos[:2]))
                yaw_delta = float(abs(_yaw_from_quat_xyzw(quat) - float(start_yaw)))
                payload_pos = _usd_translation(stage, "/World/CarryBox")
                payload_travel = 0.0
                if payload_start is not None and payload_pos is not None:
                    payload_travel = float(np.linalg.norm(np.asarray(payload_pos[:2]) - np.asarray(payload_start[:2])))
                fall = int(pos[2] < 0.20)
                summary["completed_steps"] = step + 1
                summary["travel_xy_m"] = travel
                summary["payload_travel_xy_m"] = payload_travel
                summary["fall_events"] += fall
                summary["callback_init_attempts"] = int(callback_state["init_attempts"])
                summary["callback_forward_calls"] = int(callback_state["forward_calls"])
                summary["callback_errors"] = list(callback_state["errors"])
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    writer.writerow([step, pos[0], pos[1], pos[2], travel, yaw_delta, payload_travel, fall])
                    print(
                        "[STATE] "
                        f"step={step} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                        f"travel={travel:.3f} payload_travel={payload_travel:.3f} "
                        f"fall={fall} ready={callback_state['ready']} "
                        f"forward_calls={callback_state['forward_calls']}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] Rollout failed: {summary['error']}", flush=True)
    finally:
        try:
            SimulationManager.deregister_callback(callback_id)
        except Exception:
            pass
        timeline.stop()

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
