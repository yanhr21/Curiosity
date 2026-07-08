#!/usr/bin/env python3
"""Core API cradle-cart free-box carrying diagnostic.

This is a contact-handling scaffold, not a robot locomotion result. A
world-anchored prismatic rail moves a physical tray/cradle. The payload box is a
free dynamic rigid body inside the cradle; no box pose or velocity writes are
used after scene construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cradle-cart free-box carry diagnostic.")
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--carry-steps", type=int, default=260)
    parser.add_argument("--target-x", type=float, default=0.08)
    parser.add_argument("--box-mass", type=float, default=0.5)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.20, 0.16, 0.16), metavar=("X", "Y", "Z"))
    parser.add_argument("--cradle-gap-x", type=float, default=0.025)
    parser.add_argument("--cradle-gap-y", type=float, default=0.040)
    parser.add_argument("--wall-thickness", type=float, default=0.030)
    parser.add_argument("--wall-height", type=float, default=0.20)
    parser.add_argument("--deck-thickness", type=float, default=0.035)
    parser.add_argument("--cart-z", type=float, default=0.13)
    parser.add_argument("--drive-stiffness", type=float, default=12000.0)
    parser.add_argument("--drive-damping", type=float, default=2500.0)
    parser.add_argument("--drive-max-force", type=float, default=80000.0)
    parser.add_argument("--static-friction", type=float, default=0.2)
    parser.add_argument("--dynamic-friction", type=float, default=0.1)
    parser.add_argument("--drop-z", type=float, default=0.08)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_cradle_cart_free_box_carry"),
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


CARRIER_PATH = "/World/Carrier"
ANCHOR_PATH = "/World/Carrier/Anchor"
CART_PATH = "/World/Carrier/Cart"
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


def _fixed_joint(stage: Usd.Stage, joint_path: str, body0: str, body1: str, local_pos0: tuple[float, float, float]) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _rail_joint(stage: Usd.Stage) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Carrier/CartRail")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(ANCHOR_PATH)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(CART_PATH)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set("X")
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(float(args_cli.target_x))
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
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def design_scene(stage: Usd.Stage) -> None:
    UsdGeom.Xform.Define(stage, CARRIER_PATH)
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(CARRIER_PATH))
    material = _define_material(stage, "/World/Looks/LowFrictionMaterial")
    _box_body(stage, "/World/Ground", (4.0, 2.0, 0.05), (0.0, 0.0, -0.025), (0.31, 0.33, 0.33), rigid=False, material=material)

    cart_z = float(args_cli.cart_z)
    _box_body(stage, ANCHOR_PATH, (0.08, 0.08, 0.08), (0.0, 0.0, cart_z), (0.08, 0.10, 0.14), mass=100.0, collision=False, material=material)
    _fixed_joint_to_world(stage, "/World/Carrier/AnchorWorldFixedJoint", ANCHOR_PATH, (0.0, 0.0, cart_z))
    _box_body(stage, CART_PATH, (0.08, 0.08, 0.08), (0.0, 0.0, cart_z), (0.10, 0.14, 0.20), mass=8.0, collision=False, material=material)
    _rail_joint(stage)

    box_size = tuple(float(v) for v in args_cli.box_size)
    gap_x = float(args_cli.cradle_gap_x)
    gap_y = float(args_cli.cradle_gap_y)
    wall_t = float(args_cli.wall_thickness)
    wall_h = float(args_cli.wall_height)
    deck_t = float(args_cli.deck_thickness)
    deck_z = 0.5 * deck_t
    box_z = deck_t + 0.5 * box_size[2] + 0.001
    wall_z = deck_t + 0.5 * wall_h
    cradle_x = box_size[0] + 2.0 * gap_x + 2.0 * wall_t
    cradle_y = box_size[1] + 2.0 * gap_y + 2.0 * wall_t

    _box_body(stage, BOX_PATH, box_size, (0.0, 0.0, cart_z + box_z), (0.58, 0.43, 0.24), mass=float(args_cli.box_mass), material=material)

    parts = {
        "Deck": ((cradle_x, cradle_y, deck_t), (0.0, 0.0, deck_z), (0.16, 0.28, 0.24)),
        "RearStop": ((wall_t, cradle_y, wall_h), (-0.5 * box_size[0] - gap_x - 0.5 * wall_t, 0.0, wall_z), (0.12, 0.30, 0.46)),
        "FrontStop": ((wall_t, cradle_y, wall_h), (0.5 * box_size[0] + gap_x + 0.5 * wall_t, 0.0, wall_z), (0.20, 0.30, 0.42)),
        "LeftRail": ((cradle_x, wall_t, wall_h), (0.0, 0.5 * box_size[1] + gap_y + 0.5 * wall_t, wall_z), (0.18, 0.34, 0.30)),
        "RightRail": ((cradle_x, wall_t, wall_h), (0.0, -0.5 * box_size[1] - gap_y - 0.5 * wall_t, wall_z), (0.18, 0.34, 0.30)),
    }
    for name, (size, local, color) in parts.items():
        path = f"/World/Carrier/{name}"
        world_pos = (local[0], local[1], cart_z + local[2])
        _box_body(stage, path, size, world_pos, color, mass=1.0, material=material)
        _fixed_joint(stage, f"{path}_joint", CART_PATH, path, local)


def run_scene() -> Path:
    print("[PROGRESS] run_scene entered", flush=True)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_cradle_cart_free_box_carry_state.csv"
    summary_path = args_cli.output_dir / "core_world_cradle_cart_free_box_carry_summary.json"
    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    create_new_stage()
    stage = get_current_stage()
    print("[PROGRESS] Fresh stage created", flush=True)
    design_scene(stage)
    print("[PROGRESS] USD scene designed", flush=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    carrier = SingleArticulation(prim_path=CARRIER_PATH, name="cradle_cart")
    cart = SingleRigidPrim(prim_path=CART_PATH, name="cart")
    box = SingleRigidPrim(prim_path=BOX_PATH, name="free_box")
    world.reset()
    print("[PROGRESS] World reset complete", flush=True)
    carrier.initialize()
    cart.initialize()
    box.initialize()
    print("[PROGRESS] Prims initialized", flush=True)

    initial_cart = _pose_wxyz(cart)
    initial_box = _pose_wxyz(box)
    initial_joint_positions = np.asarray(carrier.get_joint_positions(), dtype=float)
    current_targets = np.asarray(carrier.get_joint_positions(), dtype=float)
    previous_box_x = float(initial_box[0])
    dof_names = list(carrier.dof_names)

    summary = {
        "scene_type": "core_world_cradle_cart_free_box_carry",
        "success_claim": "contact_handling_scaffold_not_robot_locomotion",
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "dof_names": dof_names,
        "articulated_joint_count": int(carrier.num_dof),
        "target_x_m": float(args_cli.target_x),
        "box_mass_kg": float(args_cli.box_mass),
        "cradle_gap_x_m": float(args_cli.cradle_gap_x),
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
        "max_cart_travel_x_m": 0.0,
        "max_box_travel_x_m": 0.0,
        "max_box_relative_error_m": 0.0,
        "max_post_settle_cart_travel_x_m": 0.0,
        "max_post_settle_box_travel_x_m": 0.0,
        "max_post_settle_box_relative_error_m": 0.0,
        "max_step_box_speed_mps": 0.0,
        "min_box_z_m": float(initial_box[2]),
        "final_cart_target_distance_m": None,
        "final_box_target_distance_m": None,
        "final_box_relative_error_m": None,
        "post_settle_baseline_step": None,
        "post_settle_baseline_cart_x_m": None,
        "post_settle_baseline_box_x_m": None,
        "final_post_settle_cart_travel_x_m": None,
        "final_post_settle_box_travel_x_m": None,
        "final_post_settle_box_relative_error_m": None,
        "error": None,
    }
    post_settle_cart_x: float | None = None
    post_settle_box_x: float | None = None

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "phase",
                "target",
                "joint_motion",
                "cart_x",
                "box_x",
                "relative_error_x",
                "post_settle_cart_x",
                "post_settle_box_x",
                "post_settle_relative_error_x",
                "box_z",
                "box_speed_x",
                "drop",
            ])
            for step in range(int(args_cli.steps)):
                if step < int(args_cli.settle_steps):
                    phase_name = "settle"
                    target = 0.0
                else:
                    phase_name = "carry"
                    carry_phase = (step - int(args_cli.settle_steps)) / float(max(int(args_cli.carry_steps), 1))
                    target = float(args_cli.target_x) * _smooth01(carry_phase)
                if len(current_targets) > 0:
                    current_targets[0] = target
                carrier.apply_action(ArticulationAction(joint_positions=current_targets.tolist()))
                world.step(render=bool(args_cli.render))
                cart_pose = _pose_wxyz(cart)
                box_pose = _pose_wxyz(box)
                finite = np.all(np.isfinite(np.asarray(cart_pose + box_pose, dtype=float)))
                if not finite:
                    summary["nonfinite_state_events"] += 1
                joint_positions = np.asarray(carrier.get_joint_positions(), dtype=float)
                joint_motion = float(np.nansum(np.abs(joint_positions - initial_joint_positions)))
                cart_travel_x = float(cart_pose[0]) - float(initial_cart[0])
                box_travel_x = float(box_pose[0]) - float(initial_box[0])
                relative_error = abs(box_travel_x - cart_travel_x)
                if step >= int(args_cli.settle_steps) and post_settle_cart_x is None:
                    post_settle_cart_x = float(cart_pose[0])
                    post_settle_box_x = float(box_pose[0])
                    summary["post_settle_baseline_step"] = int(step)
                    summary["post_settle_baseline_cart_x_m"] = float(post_settle_cart_x)
                    summary["post_settle_baseline_box_x_m"] = float(post_settle_box_x)
                post_cart_travel_x = 0.0
                post_box_travel_x = 0.0
                post_relative_error = 0.0
                if post_settle_cart_x is not None and post_settle_box_x is not None:
                    post_cart_travel_x = float(cart_pose[0]) - post_settle_cart_x
                    post_box_travel_x = float(box_pose[0]) - post_settle_box_x
                    post_relative_error = abs(post_box_travel_x - post_cart_travel_x)
                box_speed_x = abs((float(box_pose[0]) - previous_box_x) / 0.005)
                previous_box_x = float(box_pose[0])
                drop = int(float(box_pose[2]) < float(args_cli.drop_z))
                summary["completed_steps"] = step + 1
                summary["box_drop_events"] += drop
                summary["max_joint_motion_m"] = max(float(summary["max_joint_motion_m"]), joint_motion)
                summary["max_cart_travel_x_m"] = max(float(summary["max_cart_travel_x_m"]), cart_travel_x)
                summary["max_box_travel_x_m"] = max(float(summary["max_box_travel_x_m"]), box_travel_x)
                summary["max_box_relative_error_m"] = max(float(summary["max_box_relative_error_m"]), relative_error)
                summary["max_post_settle_cart_travel_x_m"] = max(float(summary["max_post_settle_cart_travel_x_m"]), post_cart_travel_x)
                summary["max_post_settle_box_travel_x_m"] = max(float(summary["max_post_settle_box_travel_x_m"]), post_box_travel_x)
                summary["max_post_settle_box_relative_error_m"] = max(float(summary["max_post_settle_box_relative_error_m"]), post_relative_error)
                summary["max_step_box_speed_mps"] = max(float(summary["max_step_box_speed_mps"]), box_speed_x)
                summary["min_box_z_m"] = min(float(summary["min_box_z_m"]), float(box_pose[2]))
                summary["final_cart_target_distance_m"] = abs(float(args_cli.target_x) - cart_travel_x)
                summary["final_box_target_distance_m"] = abs(float(args_cli.target_x) - box_travel_x)
                summary["final_box_relative_error_m"] = relative_error
                summary["final_post_settle_cart_travel_x_m"] = post_cart_travel_x
                summary["final_post_settle_box_travel_x_m"] = post_box_travel_x
                summary["final_post_settle_box_relative_error_m"] = post_relative_error
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    writer.writerow([
                        step,
                        phase_name,
                        target,
                        joint_motion,
                        cart_travel_x,
                        box_travel_x,
                        relative_error,
                        post_cart_travel_x,
                        post_box_travel_x,
                        post_relative_error,
                        box_pose[2],
                        box_speed_x,
                        drop,
                    ])
                    print(
                        "[STATE] "
                        f"step={step} phase={phase_name} target={target:.4f} "
                        f"cart_x={cart_travel_x:.4f} box_x={box_travel_x:.4f} "
                        f"rel={relative_error:.4f} box_z={box_pose[2]:.4f} speed_x={box_speed_x:.4f} drop={drop}",
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
            failure_path = args_cli.output_dir / "core_world_cradle_cart_free_box_carry_startup_failure.json"
            failure_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True) + "\n")
            print(f"[ERROR] Startup failure written to: {failure_path}", flush=True)
        finally:
            print(f"[ERROR] Unhandled startup failure: {type(exc).__name__}: {exc}", flush=True)
    finally:
        simulation_app.close()
