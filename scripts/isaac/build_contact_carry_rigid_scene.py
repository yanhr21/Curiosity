#!/usr/bin/env python3
"""RigidObject-driven contact carry diagnostic.

This is the second low-level contact attempt.  Unlike
``build_contact_carry_scene.py``, kinematic palm poses are written through the
RigidObject simulation API instead of USD xform edits.  The dynamic box pose is
not commanded.
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
    parser = argparse.ArgumentParser(description="RigidObject kinematic palms carrying a dynamic box by contact.")
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--box-mass", type=float, default=3.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.42, 0.26, 0.28), metavar=("X", "Y", "Z"))
    parser.add_argument("--target-x", type=float, default=1.35)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/contact_carry_rigid_scene"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402
from isaaclab_physx.assets import RigidObject  # noqa: E402


BOX_START_X = 0.58
BOX_DROP_Z = 0.10


def _smooth01(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _phase(step: int) -> str:
    if step < int(0.18 * args_cli.steps):
        return "approach"
    if step < int(0.34 * args_cli.steps):
        return "squeeze"
    if step < int(0.56 * args_cli.steps):
        return "lift"
    return "carry"


def _rigid_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    pos: tuple[float, float, float],
    *,
    kinematic: bool,
    mass: float,
    friction: float,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=kinematic,
                disable_gravity=kinematic,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=6.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=0.85 * friction),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _make_scene_objects() -> dict[str, RigidObject]:
    floor_cfg = sim_utils.CuboidCfg(
        size=(4.5, 3.0, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    target_cfg = sim_utils.CuboidCfg(
        size=(0.50, 0.42, 0.02),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.40, 0.85), opacity=0.35),
    )
    target_cfg.func("/World/CarryTarget", target_cfg, translation=(args_cli.target_x, 0.0, 0.01))
    sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8)).func("/World/DomeLight", sim_utils.DomeLightCfg())

    box_z = float(args_cli.box_size[2]) * 0.5
    return {
        "torso": RigidObject(
            _rigid_cfg(
                "/World/Robot/Torso",
                (0.34, 0.20, 0.52),
                (0.16, 0.22, 0.32),
                (0.10, 0.0, 0.70),
                kinematic=True,
                mass=45.0,
                friction=1.2,
            )
        ),
        "left_palm": RigidObject(
            _rigid_cfg(
                "/World/Robot/LeftPalm",
                (0.32, 0.09, 0.34),
                (0.10, 0.18, 0.28),
                (BOX_START_X, 0.25, 0.31),
                kinematic=True,
                mass=3.0,
                friction=4.0,
            )
        ),
        "right_palm": RigidObject(
            _rigid_cfg(
                "/World/Robot/RightPalm",
                (0.32, 0.09, 0.34),
                (0.10, 0.18, 0.28),
                (BOX_START_X, -0.25, 0.31),
                kinematic=True,
                mass=3.0,
                friction=4.0,
            )
        ),
        "box": RigidObject(
            _rigid_cfg(
                "/World/CarryBox",
                tuple(float(v) for v in args_cli.box_size),
                (0.56, 0.42, 0.23),
                (BOX_START_X, 0.0, box_z),
                kinematic=False,
                mass=float(args_cli.box_mass),
                friction=2.0,
            )
        ),
    }


def _control_pose(step: int) -> dict[str, tuple[float, float, float]]:
    squeeze_start = int(0.18 * args_cli.steps)
    lift_start = int(0.34 * args_cli.steps)
    carry_start = int(0.56 * args_cli.steps)
    squeeze = _smooth01((step - squeeze_start) / max(1, lift_start - squeeze_start))
    lift = _smooth01((step - lift_start) / max(1, carry_start - lift_start))
    carry = _smooth01((step - carry_start) / max(1, args_cli.steps - carry_start))
    palm_y = 0.25 * (1.0 - squeeze) + 0.145 * squeeze
    palm_z = 0.31 * (1.0 - lift) + 0.62 * lift
    palm_x = BOX_START_X + (args_cli.target_x - BOX_START_X) * carry
    torso_x = palm_x - 0.34
    if step < squeeze_start:
        torso_x = -0.10 + 0.34 * _smooth01(step / max(1, squeeze_start))
    return {
        "torso": (torso_x, 0.0, 0.70),
        "left_palm": (palm_x, palm_y, palm_z),
        "right_palm": (palm_x, -palm_y, palm_z),
    }


def _write_pose(obj: RigidObject, pos: tuple[float, float, float], prev: tuple[float, float, float], dt: float) -> None:
    pose = torch.tensor([[pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 1.0]], device=args_cli.device)
    vel = torch.tensor(
        [[(pos[0] - prev[0]) / dt, (pos[1] - prev[1]) / dt, (pos[2] - prev[2]) / dt, 0.0, 0.0, 0.0]],
        device=args_cli.device,
    )
    obj.write_root_pose_to_sim_index(root_pose=pose)
    obj.write_root_velocity_to_sim_index(root_velocity=vel)


def _root_pose(obj: RigidObject) -> list[float]:
    pose = torch.as_tensor(obj.data.root_link_pose_w, device=args_cli.device)[0].detach().cpu().tolist()
    return [float(v) for v in pose]


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "contact_carry_rigid_scene_state.csv"
    summary_path = args_cli.output_dir / "contact_carry_rigid_scene_summary.json"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)

    with build_simulation_context(create_new_stage=True, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_setting("/physics/updateToUsd", True)
        sim.set_setting("/physics/updateVelocitiesToUsd", True)
        objects = _make_scene_objects()
        sim.set_camera_view(eye=[2.3, -1.8, 1.35], target=[0.75, 0.0, 0.45])
        sim.reset()
        for obj in objects.values():
            obj.reset()
        dt = sim.get_physics_dt()
        prev_targets = _control_pose(0)
        initial_box_pose = None
        summary = {
            "scene_type": "rigidobject_kinematic_palms_dynamic_box_contact_carry",
            "success_claim": "diagnostic_only_not_dynamic_robot_balance_or_learned_carrying",
            "box_pose_followed": False,
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "box_mass_kg": float(args_cli.box_mass),
            "max_box_travel_xy_m": 0.0,
            "max_box_lift_m": 0.0,
            "final_box_target_distance_xy_m": None,
            "box_drop_events": 0,
        }
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_s", "phase", "box_x", "box_y", "box_z", "travel_xy_m", "lift_m", "target_dist_m", "drop"])
            for step in range(args_cli.steps):
                targets = _control_pose(step)
                for key in ("torso", "left_palm", "right_palm"):
                    _write_pose(objects[key], targets[key], prev_targets[key], dt)
                sim.step(render=args_cli.render)
                for obj in objects.values():
                    obj.update(dt)
                prev_targets = targets
                if step % 10 == 0 or step == args_cli.steps - 1:
                    phase = _phase(step)
                    box_pose = _root_pose(objects["box"])
                    if initial_box_pose is None:
                        initial_box_pose = list(box_pose)
                    travel = math.hypot(box_pose[0] - initial_box_pose[0], box_pose[1] - initial_box_pose[1])
                    lift = box_pose[2] - initial_box_pose[2]
                    target_dist = math.hypot(box_pose[0] - args_cli.target_x, box_pose[1])
                    drop = int(phase in ("lift", "carry") and box_pose[2] < BOX_DROP_Z)
                    summary["completed_steps"] = int(step + 1)
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(travel))
                    summary["max_box_lift_m"] = max(float(summary["max_box_lift_m"]), float(lift))
                    summary["final_box_target_distance_xy_m"] = float(target_dist)
                    summary["box_drop_events"] += drop
                    writer.writerow([step, step * dt, phase, box_pose[0], box_pose[1], box_pose[2], travel, lift, target_dist, drop])
                    print(
                        "[STATE] "
                        f"step={step} phase={phase} box=({box_pose[0]:.3f},{box_pose[1]:.3f},{box_pose[2]:.3f}) "
                        f"travel={travel:.3f} lift={lift:.3f} target_dist={target_dist:.3f} drop={drop}",
                        flush=True,
                    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    run_scene()
    simulation_app.close()
