#!/usr/bin/env python3
"""Experimental-Articulation smoke for the local ANYmal-C asset.

This is a narrow control-path diagnostic.  It uses Isaac Sim's current
`isaacsim.core.experimental.prims.Articulation` API, not the deprecated
`SingleArticulation` wrapper and not IsaacLab's `Articulation` tensor class.

Passing this smoke only proves that an official local articulated robot asset
can expose DOFs and respond to joint targets in this cluster environment.  It
is not walking, balancing, payload carrying, grasping, or learned control.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANYmal experimental articulation control-path smoke.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--target-amplitude", type=float, default=0.22)
    parser.add_argument("--target-frequency", type=float, default=0.8)
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--asset-usd", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd"))
    parser.add_argument("--ground-usd", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/anymal_experimental_articulation_smoke"))
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _to_numpy(value):
    try:
        return value.numpy()
    except Exception:
        try:
            import warp as wp

            return wp.to_torch(value).detach().cpu().numpy()
        except Exception:
            return value


def _set_physics_variant(prim) -> None:
    variant_sets = prim.GetVariantSets()
    if "Physics" not in variant_sets.GetNames():
        return
    variant_set = variant_sets.GetVariantSet("Physics")
    for name in variant_set.GetVariantNames():
        if name.lower() == "physx":
            variant_set.SetVariantSelection(name)
            print(f"[PROGRESS] Selected Physics variant: {name}", flush=True)
            return


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import omni.kit.app  # noqa: E402

ext_manager = omni.kit.app.get_app().get_extension_manager()
isaacsim_ext_root = Path("/public/home/yanhongru/envs/isaac_arena_py312/lib/python3.12/site-packages/isaacsim/exts")
for ext_path in (
    isaacsim_ext_root / "isaacsim.core.experimental.utils",
    isaacsim_ext_root / "isaacsim.core.experimental.prims",
    isaacsim_ext_root / "isaacsim.core.simulation_manager",
):
    if str(ext_path) not in sys.path:
        sys.path.insert(0, str(ext_path))
        print(f"[PROGRESS] Added extension path: {ext_path}", flush=True)
    import isaacsim

    isaacsim_pkg_path = str(ext_path / "isaacsim")
    if isaacsim_pkg_path not in list(isaacsim.__path__):
        isaacsim.__path__.append(isaacsim_pkg_path)
    try:
        isaacsim_core = importlib.import_module("isaacsim.core")
        isaacsim_core_path = str(ext_path / "isaacsim" / "core")
        if isaacsim_core_path not in list(isaacsim_core.__path__):
            isaacsim_core.__path__.append(isaacsim_core_path)
    except ModuleNotFoundError:
        pass
for ext_name in (
    "isaacsim.core.experimental.utils",
    "isaacsim.core.experimental.prims",
    "isaacsim.core.simulation_manager",
):
    ext_manager.set_extension_enabled_immediate(ext_name, True)
    print(f"[PROGRESS] Enabled extension: {ext_name}", flush=True)

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaaclab.physics import PhysicsManager  # noqa: E402
import isaaclab.sim.utils.stage as lab_stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import UsdGeom  # noqa: E402


def _patch_simulation_manager_compat() -> None:
    if not hasattr(SimulationManager, "get_active_physics_engine"):
        SimulationManager.get_active_physics_engine = classmethod(lambda cls: "physx")
    if not hasattr(SimulationManager, "_physics_sim_view__warp"):
        SimulationManager._physics_sim_view__warp = getattr(SimulationManager, "_view_warp", None)
    if not hasattr(SimulationManager, "_physics_sim_interface"):
        SimulationManager._physics_sim_interface = getattr(SimulationManager, "_physx_sim", None)
    if not hasattr(SimulationManager, "_physics_stage_update_interface"):
        SimulationManager._physics_stage_update_interface = None


def _sync_simulation_manager_view_alias() -> None:
    if hasattr(SimulationManager, "_view_warp"):
        SimulationManager._physics_sim_view__warp = getattr(SimulationManager, "_view_warp", None)


def run() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "anymal_experimental_articulation_state.csv"
    summary_path = args_cli.output_dir / "anymal_experimental_articulation_summary.json"

    if not args_cli.asset_usd.is_file():
        raise FileNotFoundError(f"ANYmal USD not found: {args_cli.asset_usd}")

    print("[PROGRESS] Creating stage", flush=True)
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage(backend="usd")
    lab_stage_utils._context.stage = stage
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")

    if args_cli.ground_usd.is_file():
        stage_utils.add_reference_to_stage(str(args_cli.ground_usd), "/World/Ground")
    else:
        print(f"[WARN] Ground USD missing, continuing without reference: {args_cli.ground_usd}", flush=True)

    print(f"[PROGRESS] Referencing ANYmal: {args_cli.asset_usd}", flush=True)
    anymal_prim = stage_utils.add_reference_to_stage(str(args_cli.asset_usd), "/World/Anymal")
    print("[PROGRESS] ANYmal reference added", flush=True)
    _set_physics_variant(anymal_prim)
    print("[PROGRESS] Physics variant checked", flush=True)
    UsdGeom.XformCommonAPI(anymal_prim).SetTranslate((0.0, 0.0, 0.65))
    print("[PROGRESS] ANYmal initial transform authored", flush=True)

    device = str(args_cli.device)
    if device == "cuda":
        device = "cuda:0"
    if hasattr(SimulationManager, "set_physics_sim_device"):
        SimulationManager.set_physics_sim_device(device)
    else:
        print("[WARN] SimulationManager has no set_physics_sim_device; relying on AppLauncher device", flush=True)
    if hasattr(SimulationManager, "set_physics_dt"):
        SimulationManager.set_physics_dt(0.005)
    else:
        print("[WARN] SimulationManager has no set_physics_dt; relying on authored/default physics dt", flush=True)
    PhysicsManager._device = device
    print(f"[PROGRESS] SimulationManager compatibility checked: requested_device={device}", flush=True)

    _patch_simulation_manager_compat()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    if hasattr(SimulationManager, "_warmup_and_create_views"):
        print("[PROGRESS] Explicit PhysxManager warmup/view creation", flush=True)
        SimulationManager._warmup_and_create_views()
        _sync_simulation_manager_view_alias()
    app_utils.update_app(steps=max(1, int(args_cli.warmup_steps)))
    _sync_simulation_manager_view_alias()

    robot = Articulation("/World/Anymal", reset_xform_op_properties=False)
    print(f"[PROGRESS] Articulation created: num_dofs={robot.num_dofs}", flush=True)
    dof_names = list(robot.dof_names)
    print(f"[PROGRESS] DOFs: {dof_names}", flush=True)

    physics_valid = bool(robot.is_physics_tensor_entity_valid())
    physics_initialized = bool(robot.is_physics_tensor_entity_initialized())
    print(
        f"[PROGRESS] After warmup: physics_valid={physics_valid} physics_initialized={physics_initialized}",
        flush=True,
    )

    robot.switch_dof_control_mode("position")
    robot.set_dof_gains(
        np.full((1, robot.num_dofs), 70.0, dtype=np.float32),
        np.full((1, robot.num_dofs), 4.0, dtype=np.float32),
    )
    initial_positions = np.array(_to_numpy(robot.get_dof_positions()), dtype=np.float32).reshape(1, -1)
    initial_pose, _ = robot.get_world_poses()
    initial_pose_np = np.array(_to_numpy(initial_pose), dtype=np.float32).reshape(-1, 3)[0]

    max_joint_motion = 0.0
    max_base_travel = 0.0
    selected = list(range(min(4, robot.num_dofs)))
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_s", "max_joint_motion_rad", "base_x", "base_y", "base_z", "base_travel_xy_m"])
        for step in range(int(args_cli.steps)):
            t = step * 0.005
            targets = initial_positions.copy()
            for k, idx in enumerate(selected):
                phase = k * math.pi / 2.0
                targets[0, idx] = initial_positions[0, idx] + float(args_cli.target_amplitude) * math.sin(
                    2.0 * math.pi * float(args_cli.target_frequency) * t + phase
                )
            robot.set_dof_position_targets(targets)
            app_utils.update_app(steps=1)
            current = np.array(_to_numpy(robot.get_dof_positions()), dtype=np.float32).reshape(1, -1)
            positions, _ = robot.get_world_poses()
            base = np.array(_to_numpy(positions), dtype=np.float32).reshape(-1, 3)[0]
            joint_motion = float(np.max(np.abs(current - initial_positions)))
            base_travel = float(math.hypot(float(base[0] - initial_pose_np[0]), float(base[1] - initial_pose_np[1])))
            max_joint_motion = max(max_joint_motion, joint_motion)
            max_base_travel = max(max_base_travel, base_travel)
            if step % 10 == 0 or step == int(args_cli.steps) - 1:
                writer.writerow([step, t, joint_motion, float(base[0]), float(base[1]), float(base[2]), base_travel])
                print(
                    f"[STATE] step={step} joint_motion={joint_motion:.4f} "
                    f"base=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f}) travel={base_travel:.4f}",
                    flush=True,
                )

    timeline.stop()
    app_utils.update_app(steps=2)
    summary = {
        "scene_type": "anymal_experimental_articulation_control_path_smoke",
        "success_claim": "joint_control_diagnostic_only_not_walking_or_carrying",
        "asset_usd": str(args_cli.asset_usd),
        "device": device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": int(args_cli.steps),
        "num_dofs": int(robot.num_dofs),
        "dof_names": dof_names,
        "physics_tensor_valid_after_warmup": physics_valid,
        "physics_tensor_initialized_after_warmup": physics_initialized,
        "max_joint_motion_rad": float(max_joint_motion),
        "max_base_travel_xy_m": float(max_base_travel),
        "passed_joint_motion_gate": bool(physics_valid and max_joint_motion >= float(args_cli.motion_threshold)),
        "motion_threshold_rad": float(args_cli.motion_threshold),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run()
    except BaseException as exc:
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args_cli.output_dir / "anymal_experimental_articulation_failure_summary.json"
        failure_path.write_text(
            json.dumps(
                {
                    "scene_type": "anymal_experimental_articulation_control_path_smoke",
                    "success_claim": "failed_diagnostic_no_robot_control_evidence",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        traceback.print_exc()
        print(f"[ERROR] Failure summary written to: {failure_path}", flush=True)
        raise
    finally:
        simulation_app.close()
