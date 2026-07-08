#!/usr/bin/env python3
"""Core API single-contact free-box probe diagnostic.

This diagnostic isolates one moving contact element from the carrier.  A
world-fixed base drives one X-axis prismatic pusher toward a free dynamic box on
a support deck.  It is not carrying or walking; it exists to verify whether a
single contact closure can be low-impulse before reconnecting the mechanism to
the carrier rail.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-contact free-box probe diagnostic.")
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument("--close-steps", type=int, default=160)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--box-x", type=float, default=0.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.20, 0.16, 0.16), metavar=("X", "Y", "Z"))
    parser.add_argument("--initial-gap", type=float, default=0.040)
    parser.add_argument("--close-distance", type=float, default=0.080)
    parser.add_argument("--base-z", type=float, default=0.20)
    parser.add_argument("--pusher-size", type=float, nargs=3, default=(0.035, 0.22, 0.18), metavar=("X", "Y", "Z"))
    parser.add_argument("--pusher-mass", type=float, default=0.8)
    parser.add_argument("--drive-stiffness", type=float, default=3000.0)
    parser.add_argument("--drive-damping", type=float, default=800.0)
    parser.add_argument("--drive-max-force", type=float, default=12000.0)
    parser.add_argument("--static-friction", type=float, default=4.0)
    parser.add_argument("--dynamic-friction", type=float, default=3.0)
    parser.add_argument("--drop-z", type=float, default=0.08)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_single_contact_probe"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

print("[PROGRESS] Core imports complete", flush=True)


PROBE_PATH = "/World/Probe"
BASE_PATH = "/World/Probe/Base"
PUSHER_PATH = "/World/Probe/Pusher"
BOX_PATH = "/World/FreeBox"


def _patch_core_api_simulation_manager_compat() -> None:
    if not hasattr(SimulationManager, "_backend"):
        SimulationManager._backend = "numpy"
    if not hasattr(SimulationManager, "get_backend"):
        SimulationManager.get_backend = classmethod(lambda cls: getattr(cls, "_backend", "numpy"))
    if not hasattr(SimulationManager, "_get_backend_utils"):
        def _get_backend_utils(cls):
            import isaacsim.core.utils.numpy as np_utils

            return np_utils

        SimulationManager._get_backend_utils = classmethod(_get_backend_utils)
    if not hasattr(SimulationManager, "get_physics_sim_device"):
        SimulationManager.get_physics_sim_device = classmethod(lambda cls: args_cli.device)
    if not hasattr(SimulationManager, "get_physics_dt"):
        SimulationManager.get_physics_dt = classmethod(lambda cls: 0.005)


def _set_xform(prim: Usd.Prim, translation: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def _define_material(stage: Usd.Stage, path: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(args_cli.static_friction))
    physics_material.CreateDynamicFrictionAttr().Set(float(args_cli.dynamic_friction))
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _box_body(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    mass: float = 1.0,
    rigid: bool = True,
    collision: bool = True,
    material: UsdShade.Material,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        _bind_material(cube.GetPrim(), material)
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass_api.CreateMassAttr(float(mass))


def _fixed_joint_to_world(stage: Usd.Stage, joint_path: str, body1: str, world_pos: tuple[float, float, float]) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in world_pos]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _pusher_joint(stage: Usd.Stage, local_pos0: tuple[float, float, float]) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Probe/PusherJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(BASE_PATH)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(PUSHER_PATH)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set("X")
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(float(args_cli.close_distance))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(args_cli.drive_stiffness))
    drive.CreateDampingAttr().Set(float(args_cli.drive_damping))
    drive.CreateMaxForceAttr().Set(float(args_cli.drive_max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _pose_wxyz(prim: SingleRigidPrim) -> list[float]:
    pos, quat = prim.get_world_pose()
    return [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def design_scene(stage: Usd.Stage) -> None:
    UsdGeom.Xform.Define(stage, PROBE_PATH)
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(PROBE_PATH))
    material = _define_material(stage, "/World/Looks/HighFrictionMaterial")
    _box_body(stage, "/World/Ground", (4.0, 2.0, 0.05), (0.0, 0.0, -0.025), (0.31, 0.33, 0.33), rigid=False, material=material)
    deck_z = 0.025
    _box_body(stage, "/World/SupportDeck", (1.2, 0.7, 0.05), (0.15, 0.0, deck_z), (0.20, 0.25, 0.24), rigid=False, material=material)
    base_z = float(args_cli.base_z)
    _box_body(stage, BASE_PATH, (0.12, 0.12, 0.12), (-0.35, 0.0, base_z), (0.08, 0.12, 0.18), mass=100.0, material=material)
    _fixed_joint_to_world(stage, "/World/Probe/BaseWorldFixedJoint", BASE_PATH, (-0.35, 0.0, base_z))
    box_size = tuple(float(v) for v in args_cli.box_size)
    box_z = 2.0 * deck_z + 0.5 * box_size[2] + 0.001
    _box_body(
        stage,
        BOX_PATH,
        box_size,
        (float(args_cli.box_x), 0.0, box_z),
        (0.58, 0.43, 0.24),
        mass=float(args_cli.box_mass),
        material=material,
    )
    pusher_size = tuple(float(v) for v in args_cli.pusher_size)
    pusher_x = -0.5 * box_size[0] - float(args_cli.initial_gap) - 0.5 * pusher_size[0]
    pusher_z = box_z
    _box_body(stage, PUSHER_PATH, pusher_size, (pusher_x, 0.0, pusher_z), (0.12, 0.30, 0.46), mass=float(args_cli.pusher_mass), material=material)
    local_pos0 = (pusher_x + 0.35, 0.0, pusher_z - base_z)
    _pusher_joint(stage, local_pos0)


def run_scene() -> Path:
    print("[PROGRESS] run_scene entered", flush=True)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_single_contact_probe_state.csv"
    summary_path = args_cli.output_dir / "core_world_single_contact_probe_summary.json"
    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    create_new_stage()
    stage = get_current_stage()
    print("[PROGRESS] Fresh stage created", flush=True)
    design_scene(stage)
    print("[PROGRESS] USD scene designed", flush=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    print("[PROGRESS] Core World created", flush=True)
    probe = SingleArticulation(prim_path=PROBE_PATH, name="single_contact_probe")
    box = SingleRigidPrim(prim_path=BOX_PATH, name="free_box")
    pusher = SingleRigidPrim(prim_path=PUSHER_PATH, name="pusher")
    world.reset()
    print("[PROGRESS] World reset complete", flush=True)
    probe.initialize()
    box.initialize()
    pusher.initialize()
    print("[PROGRESS] Prims initialized", flush=True)
    initial_box = _pose_wxyz(box)
    initial_pusher = _pose_wxyz(pusher)
    pusher_half_x = 0.5 * float(args_cli.pusher_size[0])
    box_half_x = 0.5 * float(args_cli.box_size[0])
    initial_surface_gap = (float(initial_box[0]) - box_half_x) - (float(initial_pusher[0]) + pusher_half_x)
    initial_joint_positions = np.asarray(probe.get_joint_positions(), dtype=float)
    current_targets = np.asarray(probe.get_joint_positions(), dtype=float)
    dof_names = list(probe.dof_names)

    summary = {
        "scene_type": "core_world_single_contact_probe",
        "success_claim": "contact_closure_diagnostic_not_robot_carrying",
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "dof_names": dof_names,
        "articulated_joint_count": int(probe.num_dof),
        "box_mass_kg": float(args_cli.box_mass),
        "requested_box_x_m": float(args_cli.box_x),
        "initial_gap_m": float(args_cli.initial_gap),
        "actual_initial_box_x_m": float(initial_box[0]),
        "actual_initial_pusher_x_m": float(initial_pusher[0]),
        "actual_initial_surface_gap_m": float(initial_surface_gap),
        "close_distance_m": float(args_cli.close_distance),
        "root_pose_write_count": 0,
        "root_velocity_write_count": 0,
        "body_root_pose_write_count": 0,
        "body_root_velocity_command_count": 0,
        "box_pose_write_count": 0,
        "payload_pose_write_count": 0,
        "fall_events": 0,
        "box_drop_events": 0,
        "nonfinite_state_events": 0,
        "max_joint_motion_m": 0.0,
        "max_pusher_travel_x_m": 0.0,
        "max_box_travel_x_m": 0.0,
        "max_abs_box_travel_x_m": 0.0,
        "max_step_box_speed_mps": 0.0,
        "min_box_z_m": float(initial_box[2]),
        "final_box_travel_x_m": None,
        "final_pusher_travel_x_m": None,
        "final_surface_gap_m": None,
        "error": None,
    }

    previous_box_x = float(initial_box[0])
    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "phase", "target", "joint_motion", "pusher_x", "box_x", "box_z", "box_speed_x", "drop"])
            for step in range(int(args_cli.steps)):
                if step < int(args_cli.settle_steps):
                    phase_name = "settle"
                    target = 0.0
                else:
                    phase_name = "close"
                    close_phase = (step - int(args_cli.settle_steps)) / float(max(int(args_cli.close_steps), 1))
                    target = float(args_cli.close_distance) * _smooth01(close_phase)
                if len(current_targets) > 0:
                    current_targets[0] = target
                probe.apply_action(ArticulationAction(joint_positions=current_targets.tolist()))
                world.step(render=bool(args_cli.render))
                box_pose = _pose_wxyz(box)
                pusher_pose = _pose_wxyz(pusher)
                finite = np.all(np.isfinite(np.asarray(box_pose + pusher_pose, dtype=float)))
                if not finite:
                    summary["nonfinite_state_events"] += 1
                joint_positions = np.asarray(probe.get_joint_positions(), dtype=float)
                joint_motion = float(np.nansum(np.abs(joint_positions - initial_joint_positions)))
                box_travel_x = float(box_pose[0]) - float(initial_box[0])
                pusher_travel_x = float(pusher_pose[0]) - float(initial_pusher[0])
                box_speed_x = abs((float(box_pose[0]) - previous_box_x) / 0.005)
                previous_box_x = float(box_pose[0])
                drop = int(float(box_pose[2]) < float(args_cli.drop_z))
                summary["completed_steps"] = step + 1
                summary["box_drop_events"] += drop
                summary["max_joint_motion_m"] = max(float(summary["max_joint_motion_m"]), joint_motion)
                summary["max_pusher_travel_x_m"] = max(float(summary["max_pusher_travel_x_m"]), pusher_travel_x)
                summary["max_box_travel_x_m"] = max(float(summary["max_box_travel_x_m"]), box_travel_x)
                summary["max_abs_box_travel_x_m"] = max(float(summary["max_abs_box_travel_x_m"]), abs(box_travel_x))
                summary["max_step_box_speed_mps"] = max(float(summary["max_step_box_speed_mps"]), box_speed_x)
                summary["min_box_z_m"] = min(float(summary["min_box_z_m"]), float(box_pose[2]))
                summary["final_box_travel_x_m"] = box_travel_x
                summary["final_pusher_travel_x_m"] = pusher_travel_x
                summary["final_surface_gap_m"] = (float(box_pose[0]) - box_half_x) - (float(pusher_pose[0]) + pusher_half_x)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    writer.writerow([step, phase_name, target, joint_motion, pusher_pose[0], box_pose[0], box_pose[2], box_speed_x, drop])
                    print(
                        "[STATE] "
                        f"step={step} phase={phase_name} target={target:.4f} "
                        f"joint={joint_motion:.4f} pusher_x={pusher_travel_x:.4f} "
                        f"box_x={box_travel_x:.4f} box_z={box_pose[2]:.4f} speed_x={box_speed_x:.4f} drop={drop}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run_scene()
    except BaseException as exc:
        try:
            args_cli.output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = args_cli.output_dir / "core_world_single_contact_probe_startup_failure.json"
            failure_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True) + "\n")
            print(f"[ERROR] Startup failure written to: {failure_path}", flush=True)
        finally:
            print(f"[ERROR] Unhandled startup failure: {type(exc).__name__}: {exc}", flush=True)
    finally:
        simulation_app.close()
