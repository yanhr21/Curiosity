#!/usr/bin/env python3
"""MuJoCo quadruped payload-carry diagnostic.

This is a fallback physics baseline while the IsaacLab/PhysX tensor backend is
unreliable.  It uses a dynamic quadruped with contact feet, sinusoidal leg
position targets, and an explicit stabilizing body controller.  The payload box
is welded to the torso, so this is not a grasp/contact-box success claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run MuJoCo simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo dynamic quadruped carrying welded payload diagnostic.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--payload-mass", type=float, default=4.0)
    parser.add_argument("--target-speed", type=float, default=0.45)
    parser.add_argument("--target-height", type=float, default=0.56)
    parser.add_argument("--fall-height", type=float, default=0.30)
    parser.add_argument("--max-tilt-rad", type=float, default=0.55)
    parser.add_argument("--assist-mode", choices=("body_force", "none"), default="body_force")
    parser.add_argument("--max-assist-force-x", type=float, default=120.0)
    parser.add_argument("--max-assist-force-z", type=float, default=250.0)
    parser.add_argument("--max-assist-torque", type=float, default=80.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/mujoco_quadruped_payload"),
    )
    return parser.parse_args()


MJCF_TEMPLATE = r"""
<mujoco model="quadruped_payload_carry">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <geom friction="1.2 0.1 0.02" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <joint damping="2.0" armature="0.02"/>
    <position kp="80" kv="8" ctrlrange="-1.6 1.6"/>
  </default>
  <worldbody>
    <light pos="0 -3 4" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="8 4 0.1" rgba="0.3 0.32 0.32 1"/>
    <body name="torso" pos="0 0 0.56">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.28 0.16 0.08" mass="18" rgba="0.15 0.22 0.32 1"/>
      <body name="payload_box" pos="0.34 0 0.03">
        <geom name="payload_geom" type="box" size="0.22 0.16 0.16" mass="{payload_mass}" rgba="0.56 0.42 0.23 1"/>
      </body>
      <body name="fl_thigh" pos="0.18 0.13 -0.07">
        <joint name="fl_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1.0" rgba="0.1 0.18 0.28 1"/>
        <body name="fl_shin" pos="0 0 -0.24">
          <joint name="fl_knee" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" mass="0.8" rgba="0.1 0.18 0.28 1"/>
          <body name="fl_foot" pos="0 0 -0.25"><geom type="sphere" size="0.055" mass="0.25" rgba="0.06 0.08 0.08 1"/></body>
        </body>
      </body>
      <body name="fr_thigh" pos="0.18 -0.13 -0.07">
        <joint name="fr_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1.0" rgba="0.1 0.18 0.28 1"/>
        <body name="fr_shin" pos="0 0 -0.24">
          <joint name="fr_knee" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" mass="0.8" rgba="0.1 0.18 0.28 1"/>
          <body name="fr_foot" pos="0 0 -0.25"><geom type="sphere" size="0.055" mass="0.25" rgba="0.06 0.08 0.08 1"/></body>
        </body>
      </body>
      <body name="rl_thigh" pos="-0.18 0.13 -0.07">
        <joint name="rl_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1.0" rgba="0.1 0.18 0.28 1"/>
        <body name="rl_shin" pos="0 0 -0.24">
          <joint name="rl_knee" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" mass="0.8" rgba="0.1 0.18 0.28 1"/>
          <body name="rl_foot" pos="0 0 -0.25"><geom type="sphere" size="0.055" mass="0.25" rgba="0.06 0.08 0.08 1"/></body>
        </body>
      </body>
      <body name="rr_thigh" pos="-0.18 -0.13 -0.07">
        <joint name="rr_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1.0" rgba="0.1 0.18 0.28 1"/>
        <body name="rr_shin" pos="0 0 -0.24">
          <joint name="rr_knee" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" mass="0.8" rgba="0.1 0.18 0.28 1"/>
          <body name="rr_foot" pos="0 0 -0.25"><geom type="sphere" size="0.055" mass="0.25" rgba="0.06 0.08 0.08 1"/></body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="fl_hip_pos" joint="fl_hip"/>
    <position name="fl_knee_pos" joint="fl_knee"/>
    <position name="fr_hip_pos" joint="fr_hip"/>
    <position name="fr_knee_pos" joint="fr_knee"/>
    <position name="rl_hip_pos" joint="rl_hip"/>
    <position name="rl_knee_pos" joint="rl_knee"/>
    <position name="rr_hip_pos" joint="rr_hip"/>
    <position name="rr_knee_pos" joint="rr_knee"/>
  </actuator>
</mujoco>
"""


def _quat_to_roll_pitch(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def main() -> None:
    _refuse_login_node()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(MJCF_TEMPLATE.format(payload_mass=args.payload_mass))
    data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    csv_path = args.output_dir / "mujoco_quadruped_payload_state.csv"
    summary_path = args.output_dir / "mujoco_quadruped_payload_summary.json"
    dt = float(model.opt.timestep)
    initial_x = None
    summary = {
        "scene_type": "mujoco_dynamic_quadruped_assisted_payload_carry",
        "success_claim": "diagnostic_only_assistive_controller_welded_payload_not_unknown_box_grasp",
        "payload_mass_kg": float(args.payload_mass),
        "steps_requested": int(args.steps),
        "completed_steps": 0,
        "target_speed_mps": float(args.target_speed),
        "assist_mode": str(args.assist_mode),
        "external_stabilizer_enabled": args.assist_mode == "body_force",
        "root_pose_write_count": 0,
        "root_velocity_write_count": 0,
        "external_force_write_count": 0,
        "external_torque_write_count": 0,
        "max_assist_force_x_n": float(args.max_assist_force_x),
        "max_assist_force_z_n": float(args.max_assist_force_z),
        "max_assist_torque_nm": float(args.max_assist_torque),
        "max_travel_x_m": 0.0,
        "min_torso_z_m": None,
        "max_tilt_rad": 0.0,
        "fall_events": 0,
        "payload_mode": "welded_child_body",
    }

    gait_order = [
        ("fl_hip", "fl_knee", 0.0),
        ("rr_hip", "rr_knee", 0.0),
        ("fr_hip", "fr_knee", math.pi),
        ("rl_hip", "rl_knee", math.pi),
    ]
    joint_to_act = {model.actuator(i).name.replace("_pos", ""): i for i in range(model.nu)}

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_s", "torso_x", "torso_z", "roll", "pitch", "tilt", "vx", "fall"])
        for step in range(args.steps):
            t = step * dt
            for hip_name, knee_name, phase in gait_order:
                swing = math.sin(2.0 * math.pi * 1.6 * t + phase)
                data.ctrl[joint_to_act[hip_name]] = 0.28 * swing
                data.ctrl[joint_to_act[knee_name]] = -0.78 + 0.18 * max(0.0, swing)

            torso_x = float(data.qpos[0])
            torso_z = float(data.qpos[2])
            vx = float(data.qvel[0])
            qw, qx, qy, qz = (float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]), float(data.qpos[6]))
            roll, pitch = _quat_to_roll_pitch(qw, qx, qy, qz)
            tilt = math.hypot(roll, pitch)

            data.qfrc_applied[:] = 0.0
            data.xfrc_applied[:] = 0.0
            if args.assist_mode == "body_force":
                # Explicit stabilizing controller. This makes the run a diagnostic,
                # not an autonomous locomotion-policy result.
                force_x = 240.0 * (args.target_speed - vx)
                force_z = 900.0 * (args.target_height - torso_z) - 55.0 * float(data.qvel[2])
                torque_x = -180.0 * roll - 25.0 * float(data.qvel[3])
                torque_y = -180.0 * pitch - 25.0 * float(data.qvel[4])
                data.qfrc_applied[0] = np.clip(force_x, -float(args.max_assist_force_x), float(args.max_assist_force_x))
                data.xfrc_applied[torso_id, 0] = np.clip(force_x, -float(args.max_assist_force_x), float(args.max_assist_force_x))
                data.xfrc_applied[torso_id, 2] = np.clip(force_z, -float(args.max_assist_force_z), float(args.max_assist_force_z))
                data.xfrc_applied[torso_id, 3] = np.clip(torque_x, -float(args.max_assist_torque), float(args.max_assist_torque))
                data.xfrc_applied[torso_id, 4] = np.clip(torque_y, -float(args.max_assist_torque), float(args.max_assist_torque))
                summary["external_force_write_count"] += 1
                summary["external_torque_write_count"] += 1
            mujoco.mj_step(model, data)

            if initial_x is None:
                initial_x = float(data.qpos[0])
            if step % 20 == 0 or step == args.steps - 1:
                torso_x = float(data.qpos[0])
                torso_z = float(data.qpos[2])
                vx = float(data.qvel[0])
                qw, qx, qy, qz = (float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]), float(data.qpos[6]))
                roll, pitch = _quat_to_roll_pitch(qw, qx, qy, qz)
                tilt = math.hypot(roll, pitch)
                fall = int(torso_z < args.fall_height or tilt > args.max_tilt_rad)
                travel = torso_x - initial_x
                summary["completed_steps"] = int(step + 1)
                summary["max_travel_x_m"] = max(float(summary["max_travel_x_m"]), float(travel))
                summary["min_torso_z_m"] = (
                    float(torso_z) if summary["min_torso_z_m"] is None else min(float(summary["min_torso_z_m"]), torso_z)
                )
                summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), float(tilt))
                summary["fall_events"] += fall
                writer.writerow([step, step * dt, torso_x, torso_z, roll, pitch, tilt, vx, fall])
                print(
                    "[STATE] "
                    f"step={step} x={torso_x:.3f} z={torso_z:.3f} vx={vx:.3f} tilt={tilt:.3f} fall={fall}",
                    flush=True,
                )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")


if __name__ == "__main__":
    main()
