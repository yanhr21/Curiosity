#!/usr/bin/env python3
"""MuJoCo quadruped free-box contact-carry diagnostic.

This is a step beyond the welded-payload diagnostic: the box is a separate
free rigid body and should be retained by contact with a torso-mounted tray.
The quadruped still uses an explicit stabilizing body-force controller, so a
passing run is only a diagnostic backend candidate, not autonomous locomotion
or final unknown-object carrying.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--box-mass", type=float, default=2.0)
    parser.add_argument("--target-speed", type=float, default=0.18)
    parser.add_argument("--target-height", type=float, default=0.56)
    parser.add_argument("--fall-height", type=float, default=0.30)
    parser.add_argument("--box-drop-height", type=float, default=0.58)
    parser.add_argument("--max-tilt-rad", type=float, default=0.55)
    parser.add_argument("--actuator-kp", type=float, default=80.0)
    parser.add_argument("--actuator-kv", type=float, default=8.0)
    parser.add_argument("--assist-mode", choices=("body_force", "none"), default="body_force")
    parser.add_argument("--max-assist-force-x", type=float, default=115.0)
    parser.add_argument("--max-assist-force-z", type=float, default=340.0)
    parser.add_argument("--max-assist-torque", type=float, default=240.0)
    parser.add_argument("--stop-after-box-travel", type=float, default=None)
    parser.add_argument("--hold-target-speed", type=float, default=0.0)
    parser.add_argument("--retention-force-mode", choices=("none", "relative_spring"), default="none")
    parser.add_argument("--retention-kp-x", type=float, default=0.0)
    parser.add_argument("--retention-kd-x", type=float, default=0.0)
    parser.add_argument("--retention-kp-y", type=float, default=0.0)
    parser.add_argument("--retention-kd-y", type=float, default=0.0)
    parser.add_argument("--retention-kp-z", type=float, default=0.0)
    parser.add_argument("--retention-kd-z", type=float, default=0.0)
    parser.add_argument("--retention-max-force-x", type=float, default=0.0)
    parser.add_argument("--retention-max-force-y", type=float, default=0.0)
    parser.add_argument("--retention-max-force-z", type=float, default=0.0)
    parser.add_argument("--leg-drive-mode", choices=("sinusoid", "foot_ik"), default="sinusoid")
    parser.add_argument("--gait-frequency-hz", type=float, default=1.6)
    parser.add_argument("--stance-duty", type=float, default=0.68)
    parser.add_argument("--stride-length", type=float, default=0.16)
    parser.add_argument("--stance-foot-z-down", type=float, default=0.43)
    parser.add_argument("--swing-foot-z-down", type=float, default=0.32)
    parser.add_argument("--foot-roll-z-gain", type=float, default=0.0)
    parser.add_argument("--hip-roll-base", type=float, default=0.0)
    parser.add_argument("--hip-roll-feedback-gain", type=float, default=0.0)
    parser.add_argument("--hold-stance-foot-z-down", type=float, default=None)
    parser.add_argument("--hold-hip-roll-base", type=float, default=None)
    parser.add_argument("--hold-hip-roll-feedback-gain", type=float, default=None)
    parser.add_argument("--hold-foot-roll-z-gain", type=float, default=None)
    parser.add_argument("--hold-front-foot-x", type=float, default=None)
    parser.add_argument("--hold-rear-foot-x", type=float, default=None)
    parser.add_argument("--hold-pitch-foot-x-gain", type=float, default=0.0)
    parser.add_argument("--hold-capture-point-foot-placement", action="store_true")
    parser.add_argument("--hold-capture-time-constant", type=float, default=0.18)
    parser.add_argument("--hold-capture-x-gain", type=float, default=0.0)
    parser.add_argument("--hold-capture-x-limit", type=float, default=0.06)
    parser.add_argument("--hold-capture-y-hip-gain", type=float, default=0.0)
    parser.add_argument("--hold-capture-y-foot-z-gain", type=float, default=0.0)
    parser.add_argument("--hold-capture-y-limit", type=float, default=0.08)
    parser.add_argument("--closed-loop-foot-placement", action="store_true")
    parser.add_argument("--stride-velocity-gain", type=float, default=0.0)
    parser.add_argument("--stride-position-gain", type=float, default=0.0)
    parser.add_argument("--stride-clip", type=float, default=0.20)
    parser.add_argument(
        "--support-controller-mode",
        choices=(
            "none",
            "stance_force",
            "centroidal_stance_force",
            "lqr_stance_force",
            "lqr_additive_stance_force",
        ),
        default="none",
    )
    parser.add_argument("--support-force-scale", type=float, default=1.0)
    parser.add_argument("--support-fx-scale", type=float, default=None)
    parser.add_argument("--hold-support-fx-scale", type=float, default=None)
    parser.add_argument("--hold-support-kp-vx-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-max-fx-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-kd-z-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-kd-roll-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-kd-pitch-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-max-foot-fz-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-max-joint-torque-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-height-offset", type=float, default=0.0)
    parser.add_argument("--support-com-x-gain", type=float, default=0.0)
    parser.add_argument("--support-com-y-gain", type=float, default=0.0)
    parser.add_argument("--support-com-vx-gain", type=float, default=0.0)
    parser.add_argument("--support-com-vy-gain", type=float, default=0.0)
    parser.add_argument("--support-com-target-x-offset", type=float, default=0.0)
    parser.add_argument("--support-com-target-y-offset", type=float, default=0.0)
    parser.add_argument("--support-com-max-fz-shift", type=float, default=0.0)
    parser.add_argument("--support-com-pre-latch-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-com-scale", type=float, default=1.0)
    parser.add_argument("--support-fy-roll-gain", type=float, default=0.0)
    parser.add_argument("--support-fy-roll-rate-gain", type=float, default=0.0)
    parser.add_argument("--support-fy-com-y-gain", type=float, default=0.0)
    parser.add_argument("--support-fy-world-y-gain", type=float, default=0.0)
    parser.add_argument("--support-fy-world-vy-gain", type=float, default=0.0)
    parser.add_argument("--support-fy-world-y-source", choices=("torso", "box", "robot_com"), default="torso")
    parser.add_argument("--support-max-total-fy", type=float, default=0.0)
    parser.add_argument("--support-fy-pre-latch-scale", type=float, default=1.0)
    parser.add_argument("--hold-support-fy-scale", type=float, default=1.0)
    parser.add_argument("--support-kp-z", type=float, default=2600.0)
    parser.add_argument("--support-kd-z", type=float, default=180.0)
    parser.add_argument("--support-kp-roll", type=float, default=260.0)
    parser.add_argument("--support-kd-roll", type=float, default=38.0)
    parser.add_argument("--support-kp-pitch", type=float, default=220.0)
    parser.add_argument("--support-kd-pitch", type=float, default=32.0)
    parser.add_argument("--support-kp-vx", type=float, default=520.0)
    parser.add_argument("--support-max-total-fx", type=float, default=260.0)
    parser.add_argument("--support-min-foot-fz", type=float, default=10.0)
    parser.add_argument("--support-max-foot-fz", type=float, default=260.0)
    parser.add_argument("--support-max-joint-torque", type=float, default=220.0)
    parser.add_argument("--support-lqr-horizon-steps", type=int, default=80)
    parser.add_argument("--support-lqr-q-pos", type=float, default=80.0)
    parser.add_argument("--support-lqr-q-vel", type=float, default=8.0)
    parser.add_argument("--support-lqr-r", type=float, default=1.0)
    parser.add_argument("--support-lqr-max-fx", type=float, default=120.0)
    parser.add_argument("--support-lqr-max-fy", type=float, default=120.0)
    parser.add_argument("--support-lqr-post-latch-only", action="store_true")
    parser.add_argument("--tray-half-length", type=float, default=0.38)
    parser.add_argument("--tray-half-width", type=float, default=0.24)
    parser.add_argument("--wall-height", type=float, default=0.16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/mujoco_quadruped_freebox"),
    )
    return parser.parse_args()


MJCF_TEMPLATE = r"""
<mujoco model="quadruped_freebox_contact_carry">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <geom friction="1.6 0.15 0.02" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <joint damping="2.0" armature="0.02"/>
    <position kp="{actuator_kp}" kv="{actuator_kv}" ctrlrange="-1.6 1.6"/>
  </default>
  <worldbody>
    <light pos="0 -3 4" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="8 4 0.1" rgba="0.3 0.32 0.32 1"/>
    <body name="payload_box" pos="0.34 0 0.83">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="box" size="0.16 0.14 0.12" mass="{box_mass}" rgba="0.56 0.42 0.23 1"/>
    </body>
    <body name="torso" pos="0 0 0.56">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.28 0.16 0.08" mass="18" rgba="0.15 0.22 0.32 1"/>
      <geom name="tray_deck" type="box" pos="0.34 0 0.11" size="{tray_half_length} {tray_half_width} 0.025" mass="1.2" rgba="0.28 0.36 0.40 1"/>
      <geom name="tray_left_wall" type="box" pos="0.34 {wall_y} {wall_z}" size="{tray_half_length} 0.025 {wall_half_height}" mass="0.4" rgba="0.32 0.42 0.46 1"/>
      <geom name="tray_right_wall" type="box" pos="0.34 -{wall_y} {wall_z}" size="{tray_half_length} 0.025 {wall_half_height}" mass="0.4" rgba="0.32 0.42 0.46 1"/>
      <geom name="tray_front_stop" type="box" pos="{front_x} 0 {wall_z}" size="0.025 {tray_half_width} {wall_half_height}" mass="0.35" rgba="0.32 0.42 0.46 1"/>
      <geom name="tray_rear_stop" type="box" pos="{rear_x} 0 {wall_z}" size="0.025 {tray_half_width} {wall_half_height}" mass="0.35" rgba="0.32 0.42 0.46 1"/>
      <body name="fl_thigh" pos="0.18 0.13 -0.07">
        <joint name="fl_hip_roll" type="hinge" axis="1 0 0" range="-0.65 0.65"/>
        <joint name="fl_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1.0" rgba="0.1 0.18 0.28 1"/>
        <body name="fl_shin" pos="0 0 -0.24">
          <joint name="fl_knee" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" mass="0.8" rgba="0.1 0.18 0.28 1"/>
          <body name="fl_foot" pos="0 0 -0.25"><geom type="sphere" size="0.055" mass="0.25" rgba="0.06 0.08 0.08 1"/></body>
        </body>
      </body>
      <body name="fr_thigh" pos="0.18 -0.13 -0.07">
        <joint name="fr_hip_roll" type="hinge" axis="1 0 0" range="-0.65 0.65"/>
        <joint name="fr_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1.0" rgba="0.1 0.18 0.28 1"/>
        <body name="fr_shin" pos="0 0 -0.24">
          <joint name="fr_knee" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" mass="0.8" rgba="0.1 0.18 0.28 1"/>
          <body name="fr_foot" pos="0 0 -0.25"><geom type="sphere" size="0.055" mass="0.25" rgba="0.06 0.08 0.08 1"/></body>
        </body>
      </body>
      <body name="rl_thigh" pos="-0.18 0.13 -0.07">
        <joint name="rl_hip_roll" type="hinge" axis="1 0 0" range="-0.65 0.65"/>
        <joint name="rl_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1.0" rgba="0.1 0.18 0.28 1"/>
        <body name="rl_shin" pos="0 0 -0.24">
          <joint name="rl_knee" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" mass="0.8" rgba="0.1 0.18 0.28 1"/>
          <body name="rl_foot" pos="0 0 -0.25"><geom type="sphere" size="0.055" mass="0.25" rgba="0.06 0.08 0.08 1"/></body>
        </body>
      </body>
      <body name="rr_thigh" pos="-0.18 -0.13 -0.07">
        <joint name="rr_hip_roll" type="hinge" axis="1 0 0" range="-0.65 0.65"/>
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
    <position name="fl_hip_roll_pos" joint="fl_hip_roll"/>
    <position name="fl_hip_pos" joint="fl_hip"/>
    <position name="fl_knee_pos" joint="fl_knee"/>
    <position name="fr_hip_roll_pos" joint="fr_hip_roll"/>
    <position name="fr_hip_pos" joint="fr_hip"/>
    <position name="fr_knee_pos" joint="fr_knee"/>
    <position name="rl_hip_roll_pos" joint="rl_hip_roll"/>
    <position name="rl_hip_pos" joint="rl_hip"/>
    <position name="rl_knee_pos" joint="rl_knee"/>
    <position name="rr_hip_roll_pos" joint="rr_hip_roll"/>
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


def _leg_ik(x: float, z_down: float) -> tuple[float, float]:
    """Planar two-link IK for the MuJoCo leg, where x is forward and z_down is positive downward."""
    l1 = 0.24
    l2 = 0.25
    dist2 = x * x + z_down * z_down
    cos_knee = (dist2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_knee = max(-0.98, min(0.98, cos_knee))
    knee = -math.acos(cos_knee)
    u = -x
    v = z_down
    hip = math.atan2(u, v) - math.atan2(l2 * math.sin(knee), l1 + l2 * math.cos(knee))
    hip = max(-1.15, min(1.15, hip))
    knee = max(-1.55, min(0.15, knee))
    return hip, knee


def _finite_horizon_double_integrator_lqr_gain(
    dt: float,
    horizon_steps: int,
    q_pos: float,
    q_vel: float,
    r: float,
) -> tuple[float, float]:
    """Return K for u = -K [pos_error, vel_error] on a 1D double integrator."""
    import numpy as np

    a = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    b = np.array([[0.5 * dt * dt], [dt]], dtype=float)
    q = np.diag([max(0.0, q_pos), max(0.0, q_vel)])
    r_mat = np.array([[max(1e-6, r)]], dtype=float)
    p = q.copy()
    for _ in range(max(1, int(horizon_steps))):
        bt_p = b.T @ p
        gain = np.linalg.solve(r_mat + bt_p @ b, bt_p @ a)
        p = q + a.T @ p @ (a - b @ gain)
    gain = np.linalg.solve(r_mat + b.T @ p @ b, b.T @ p @ a)
    return float(gain[0, 0]), float(gain[0, 1])


def _xml(args: argparse.Namespace) -> str:
    wall_y = args.tray_half_width + 0.025
    wall_half_height = args.wall_height * 0.5
    wall_z = 0.11 + 0.025 + wall_half_height
    return MJCF_TEMPLATE.format(
        box_mass=args.box_mass,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        tray_half_length=args.tray_half_length,
        tray_half_width=args.tray_half_width,
        wall_y=wall_y,
        wall_z=wall_z,
        wall_half_height=wall_half_height,
        front_x=0.34 + args.tray_half_length + 0.025,
        rear_x=0.34 - args.tray_half_length - 0.025,
    )


def main() -> None:
    _refuse_login_node()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(_xml(args))
    data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_box")
    mujoco.mj_forward(model, data)
    csv_path = args.output_dir / "mujoco_quadruped_freebox_state.csv"
    summary_path = args.output_dir / "mujoco_quadruped_freebox_summary.json"
    dt = float(model.opt.timestep)
    initial_torso_x = float(data.xpos[torso_id, 0])
    initial_box_x = float(data.xpos[box_id, 0])
    initial_box_torso_offset_x = float(data.xpos[box_id, 0] - data.xpos[torso_id, 0])
    initial_box_torso_offset_y = float(data.xpos[box_id, 1] - data.xpos[torso_id, 1])
    initial_box_torso_offset_z = float(data.xpos[box_id, 2] - data.xpos[torso_id, 2])
    target_stop_latched = False
    target_stop_step = None
    summary = {
        "scene_type": "mujoco_dynamic_quadruped_assisted_freebox_contact_carry",
        "success_claim": "diagnostic_only_assistive_controller_free_box_contact_tray_not_final_unknown_box_carrying",
        "payload_mode": "free_body_contact_tray",
        "box_mass_kg": float(args.box_mass),
        "steps_requested": int(args.steps),
        "completed_steps": 0,
        "target_speed_mps": float(args.target_speed),
        "stop_after_box_travel_m": args.stop_after_box_travel,
        "hold_target_speed_mps": float(args.hold_target_speed),
        "target_stop_latched": False,
        "target_stop_step": None,
        "target_stop_hold_steps": 0,
        "assist_mode": str(args.assist_mode),
        "actuator_kp": float(args.actuator_kp),
        "actuator_kv": float(args.actuator_kv),
        "external_stabilizer_enabled": args.assist_mode == "body_force",
        "root_pose_write_count": 0,
        "root_velocity_write_count": 0,
        "box_pose_write_count": 0,
        "box_velocity_write_count": 0,
        "external_force_write_count": 0,
        "external_torque_write_count": 0,
        "box_retention_force_mode": str(args.retention_force_mode),
        "box_retention_force_write_count": 0,
        "box_retention_equal_opposite_force": args.retention_force_mode != "none",
        "box_retention_kp_x": float(args.retention_kp_x),
        "box_retention_kd_x": float(args.retention_kd_x),
        "box_retention_kp_y": float(args.retention_kp_y),
        "box_retention_kd_y": float(args.retention_kd_y),
        "box_retention_kp_z": float(args.retention_kp_z),
        "box_retention_kd_z": float(args.retention_kd_z),
        "box_retention_max_force_x_n": float(args.retention_max_force_x),
        "box_retention_max_force_y_n": float(args.retention_max_force_y),
        "box_retention_max_force_z_n": float(args.retention_max_force_z),
        "leg_drive_mode": str(args.leg_drive_mode),
        "gait_frequency_hz": float(args.gait_frequency_hz),
        "stance_duty": float(args.stance_duty),
        "stride_length_m": float(args.stride_length),
        "stance_foot_z_down_m": float(args.stance_foot_z_down),
        "swing_foot_z_down_m": float(args.swing_foot_z_down),
        "foot_roll_z_gain": float(args.foot_roll_z_gain),
        "hip_roll_base": float(args.hip_roll_base),
        "hip_roll_feedback_gain": float(args.hip_roll_feedback_gain),
        "hold_stance_foot_z_down_m": args.hold_stance_foot_z_down,
        "hold_hip_roll_base": args.hold_hip_roll_base,
        "hold_hip_roll_feedback_gain": args.hold_hip_roll_feedback_gain,
        "hold_foot_roll_z_gain": args.hold_foot_roll_z_gain,
        "hold_front_foot_x_m": args.hold_front_foot_x,
        "hold_rear_foot_x_m": args.hold_rear_foot_x,
        "hold_pitch_foot_x_gain": float(args.hold_pitch_foot_x_gain),
        "hold_capture_point_foot_placement": bool(args.hold_capture_point_foot_placement),
        "hold_capture_time_constant_s": float(args.hold_capture_time_constant),
        "hold_capture_x_gain": float(args.hold_capture_x_gain),
        "hold_capture_x_limit_m": float(args.hold_capture_x_limit),
        "hold_capture_y_hip_gain": float(args.hold_capture_y_hip_gain),
        "hold_capture_y_foot_z_gain": float(args.hold_capture_y_foot_z_gain),
        "hold_capture_y_limit_m": float(args.hold_capture_y_limit),
        "hold_capture_active_steps": 0,
        "max_abs_hold_capture_x_adjust_m": 0.0,
        "max_abs_hold_capture_y_signal_m": 0.0,
        "closed_loop_foot_placement": bool(args.closed_loop_foot_placement),
        "stride_velocity_gain": float(args.stride_velocity_gain),
        "stride_position_gain": float(args.stride_position_gain),
        "stride_clip_m": float(args.stride_clip),
        "support_controller_mode": str(args.support_controller_mode),
        "support_joint_torque_write_count": 0,
        "support_force_scale": float(args.support_force_scale),
        "support_fx_scale": args.support_fx_scale,
        "hold_support_fx_scale": args.hold_support_fx_scale,
        "hold_support_kp_vx_scale": float(args.hold_support_kp_vx_scale),
        "hold_support_max_fx_scale": float(args.hold_support_max_fx_scale),
        "hold_support_kd_z_scale": float(args.hold_support_kd_z_scale),
        "hold_support_kd_roll_scale": float(args.hold_support_kd_roll_scale),
        "hold_support_kd_pitch_scale": float(args.hold_support_kd_pitch_scale),
        "hold_support_max_foot_fz_scale": float(args.hold_support_max_foot_fz_scale),
        "hold_support_max_joint_torque_scale": float(args.hold_support_max_joint_torque_scale),
        "hold_support_height_offset_m": float(args.hold_support_height_offset),
        "support_com_x_gain": float(args.support_com_x_gain),
        "support_com_y_gain": float(args.support_com_y_gain),
        "support_com_vx_gain": float(args.support_com_vx_gain),
        "support_com_vy_gain": float(args.support_com_vy_gain),
        "support_com_target_x_offset_m": float(args.support_com_target_x_offset),
        "support_com_target_y_offset_m": float(args.support_com_target_y_offset),
        "support_com_max_fz_shift_n": float(args.support_com_max_fz_shift),
        "support_com_pre_latch_scale": float(args.support_com_pre_latch_scale),
        "hold_support_com_scale": float(args.hold_support_com_scale),
        "support_fy_roll_gain": float(args.support_fy_roll_gain),
        "support_fy_roll_rate_gain": float(args.support_fy_roll_rate_gain),
        "support_fy_com_y_gain": float(args.support_fy_com_y_gain),
        "support_fy_world_y_gain": float(args.support_fy_world_y_gain),
        "support_fy_world_vy_gain": float(args.support_fy_world_vy_gain),
        "support_fy_world_y_source": str(args.support_fy_world_y_source),
        "support_max_total_fy_n": float(args.support_max_total_fy),
        "support_fy_pre_latch_scale": float(args.support_fy_pre_latch_scale),
        "hold_support_fy_scale": float(args.hold_support_fy_scale),
        "max_abs_support_fy_n": 0.0,
        "max_abs_support_world_y_error_m": 0.0,
        "max_abs_box_y_m": 0.0,
        "max_abs_robot_com_support_x_error_m": 0.0,
        "max_abs_robot_com_support_y_error_m": 0.0,
        "max_abs_support_com_fz_shift_n": 0.0,
        "support_kp_z": float(args.support_kp_z),
        "support_kd_z": float(args.support_kd_z),
        "support_kp_roll": float(args.support_kp_roll),
        "support_kd_roll": float(args.support_kd_roll),
        "support_kp_pitch": float(args.support_kp_pitch),
        "support_kd_pitch": float(args.support_kd_pitch),
        "support_kp_vx": float(args.support_kp_vx),
        "support_max_total_fx_n": float(args.support_max_total_fx),
        "support_min_foot_fz_n": float(args.support_min_foot_fz),
        "support_max_foot_fz_n": float(args.support_max_foot_fz),
        "support_max_joint_torque_nm": float(args.support_max_joint_torque),
        "support_lqr_horizon_steps": int(args.support_lqr_horizon_steps),
        "support_lqr_q_pos": float(args.support_lqr_q_pos),
        "support_lqr_q_vel": float(args.support_lqr_q_vel),
        "support_lqr_r": float(args.support_lqr_r),
        "support_lqr_max_fx_n": float(args.support_lqr_max_fx),
        "support_lqr_max_fy_n": float(args.support_lqr_max_fy),
        "support_lqr_post_latch_only": bool(args.support_lqr_post_latch_only),
        "support_lqr_active_steps": 0,
        "support_lqr_k_pos": 0.0,
        "support_lqr_k_vel": 0.0,
        "max_abs_support_lqr_fx_n": 0.0,
        "max_abs_support_lqr_fy_n": 0.0,
        "max_assist_force_x_n": float(args.max_assist_force_x),
        "max_assist_force_z_n": float(args.max_assist_force_z),
        "max_assist_torque_nm": float(args.max_assist_torque),
        "tray_half_length_m": float(args.tray_half_length),
        "tray_half_width_m": float(args.tray_half_width),
        "wall_height_m": float(args.wall_height),
        "max_torso_travel_x_m": 0.0,
        "max_box_travel_x_m": 0.0,
        "final_torso_travel_x_m": 0.0,
        "final_box_travel_x_m": 0.0,
        "min_torso_z_m": None,
        "min_box_z_m": None,
        "max_tilt_rad": 0.0,
        "fall_events": 0,
        "box_drop_events": 0,
        "max_box_torso_relative_offset_error_m": 0.0,
        "final_box_torso_relative_offset_error_m": 0.0,
    }

    gait_order = [
        ("fl_hip_roll", "fl_hip", "fl_knee", 0.0, 1.0, 1.0),
        ("rr_hip_roll", "rr_hip", "rr_knee", 0.0, -1.0, -1.0),
        ("fr_hip_roll", "fr_hip", "fr_knee", 0.5, -1.0, 1.0),
        ("rl_hip_roll", "rl_hip", "rl_knee", 0.5, 1.0, -1.0),
    ]
    joint_to_act = {model.actuator(i).name.replace("_pos", ""): i for i in range(model.nu)}
    joint_dof_ids = {
        name: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in joint_to_act
    }
    foot_body_ids = {
        "fl": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fl_foot"),
        "fr": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr_foot"),
        "rl": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rl_foot"),
        "rr": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rr_foot"),
    }
    actuated_dof_ids = sorted(joint_dof_ids.values())
    robot_body_ids = [
        i
        for i in range(model.nbody)
        if i != box_id and float(model.body(i).mass[0]) > 0.0
    ]
    robot_mass_kg = sum(float(model.body(i).mass[0]) for i in robot_body_ids)
    support_lqr_k_pos, support_lqr_k_vel = _finite_horizon_double_integrator_lqr_gain(
        dt=dt,
        horizon_steps=int(args.support_lqr_horizon_steps),
        q_pos=float(args.support_lqr_q_pos),
        q_vel=float(args.support_lqr_q_vel),
        r=float(args.support_lqr_r),
    )
    summary["support_lqr_k_pos"] = float(support_lqr_k_pos)
    summary["support_lqr_k_vel"] = float(support_lqr_k_vel)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time_s",
                "torso_x",
                "torso_z",
                "box_x",
                "box_y",
                "box_z",
                "roll",
                "pitch",
                "tilt",
                "vx",
                "target_speed_cmd",
                "box_rel_error",
                "fall",
                "box_drop",
            ]
        )
        for step in range(args.steps):
            t = step * dt
            box_travel_for_control = float(data.xpos[box_id, 0] - initial_box_x)
            target_speed_for_gait = float(args.target_speed)
            if args.stop_after_box_travel is not None and target_stop_latched:
                target_speed_for_gait = float(args.hold_target_speed)
            stance_z_for_gait = float(args.stance_foot_z_down)
            hip_roll_base_for_gait = float(args.hip_roll_base)
            hip_roll_feedback_gain_for_gait = float(args.hip_roll_feedback_gain)
            foot_roll_z_gain_for_gait = float(args.foot_roll_z_gain)
            if target_stop_latched:
                if args.hold_stance_foot_z_down is not None:
                    stance_z_for_gait = float(args.hold_stance_foot_z_down)
                if args.hold_hip_roll_base is not None:
                    hip_roll_base_for_gait = float(args.hold_hip_roll_base)
                if args.hold_hip_roll_feedback_gain is not None:
                    hip_roll_feedback_gain_for_gait = float(args.hold_hip_roll_feedback_gain)
                if args.hold_foot_roll_z_gain is not None:
                    foot_roll_z_gain_for_gait = float(args.hold_foot_roll_z_gain)
            qw_pre, qx_pre, qy_pre, qz_pre = (
                float(data.qpos[10]),
                float(data.qpos[11]),
                float(data.qpos[12]),
                float(data.qpos[13]),
            )
            roll_for_gait, pitch_for_gait = _quat_to_roll_pitch(qw_pre, qx_pre, qy_pre, qz_pre)
            torso_vx_for_gait = float(data.qvel[6])
            hold_capture_active = bool(target_stop_latched and args.hold_capture_point_foot_placement)
            hold_capture_x_adjust = 0.0
            hold_capture_y_signal = 0.0
            if hold_capture_active:
                capture_tau = max(0.0, float(args.hold_capture_time_constant))
                hold_capture_x_raw = capture_tau * float(data.qvel[6])
                hold_capture_x_adjust = float(args.hold_capture_x_gain) * hold_capture_x_raw
                hold_capture_x_limit = max(0.0, float(args.hold_capture_x_limit))
                hold_capture_x_adjust = max(
                    -hold_capture_x_limit,
                    min(hold_capture_x_limit, hold_capture_x_adjust),
                )
                hold_capture_y_raw = float(data.xpos[torso_id, 1]) + capture_tau * float(data.qvel[7])
                hold_capture_y_limit = max(0.0, float(args.hold_capture_y_limit))
                hold_capture_y_signal = max(
                    -hold_capture_y_limit,
                    min(hold_capture_y_limit, hold_capture_y_raw),
                )
                summary["hold_capture_active_steps"] = int(summary["hold_capture_active_steps"]) + 1
                summary["max_abs_hold_capture_x_adjust_m"] = max(
                    float(summary["max_abs_hold_capture_x_adjust_m"]),
                    abs(float(hold_capture_x_adjust)),
                )
                summary["max_abs_hold_capture_y_signal_m"] = max(
                    float(summary["max_abs_hold_capture_y_signal_m"]),
                    abs(float(hold_capture_y_signal)),
                )
            stance_feet: list[tuple[int, float, float]] = []

            for roll_name, hip_name, knee_name, phase, side_sign, fore_sign in gait_order:
                if args.leg_drive_mode == "sinusoid":
                    data.ctrl[joint_to_act[roll_name]] = 0.0
                    swing = math.sin(2.0 * math.pi * float(args.gait_frequency_hz) * t + 2.0 * math.pi * phase)
                    data.ctrl[joint_to_act[hip_name]] = 0.28 * swing
                    data.ctrl[joint_to_act[knee_name]] = -0.78 + 0.18 * max(0.0, swing)
                else:
                    hip_roll_target = side_sign * (
                        hip_roll_base_for_gait + hip_roll_feedback_gain_for_gait * roll_for_gait
                    )
                    if hold_capture_active:
                        hip_roll_target += (
                            -side_sign
                            * float(args.hold_capture_y_hip_gain)
                            * float(hold_capture_y_signal)
                        )
                    hip_roll_target = max(-0.60, min(0.60, hip_roll_target))
                    data.ctrl[joint_to_act[roll_name]] = hip_roll_target
                    cycle = (float(args.gait_frequency_hz) * t + phase) % 1.0
                    duty = max(0.50, min(0.90, float(args.stance_duty)))
                    if abs(float(args.target_speed)) > 1e-6:
                        speed_scale = max(0.0, min(1.0, target_speed_for_gait / float(args.target_speed)))
                    else:
                        speed_scale = 0.0
                    stride = float(args.stride_length) * speed_scale
                    if args.closed_loop_foot_placement:
                        speed_error = target_speed_for_gait - torso_vx_for_gait
                        position_error = 0.0
                        if args.stop_after_box_travel is not None:
                            position_error = float(args.stop_after_box_travel) - box_travel_for_control
                        stride -= float(args.stride_velocity_gain) * speed_error
                        stride -= float(args.stride_position_gain) * position_error
                        stride = max(-float(args.stride_clip), min(float(args.stride_clip), stride))
                    hold_static_support = (
                        target_stop_latched
                        and abs(target_speed_for_gait) < 1e-6
                        and args.hold_front_foot_x is not None
                        and args.hold_rear_foot_x is not None
                    )
                    if hold_static_support:
                        foot_x = float(args.hold_front_foot_x) if fore_sign > 0.0 else float(args.hold_rear_foot_x)
                        foot_x += fore_sign * float(args.hold_pitch_foot_x_gain) * pitch_for_gait
                        if hold_capture_active:
                            foot_x += fore_sign * hold_capture_x_adjust
                        foot_x = max(-0.22, min(0.22, foot_x))
                        foot_z = stance_z_for_gait
                        is_stance = True
                    elif abs(stride) < 1e-6:
                        foot_x = 0.0
                        foot_z = stance_z_for_gait
                        is_stance = True
                    elif cycle < duty:
                        progress = cycle / duty
                        foot_x = 0.5 * stride - progress * stride
                        foot_z = stance_z_for_gait
                        is_stance = True
                    else:
                        progress = (cycle - duty) / max(1e-6, 1.0 - duty)
                        foot_x = -0.5 * stride + progress * stride
                        lift = math.sin(math.pi * progress)
                        foot_z = (1.0 - lift) * stance_z_for_gait + lift * float(args.swing_foot_z_down)
                        is_stance = False
                    foot_z += side_sign * foot_roll_z_gain_for_gait * roll_for_gait
                    if hold_capture_active:
                        foot_z += (
                            -side_sign
                            * float(args.hold_capture_y_foot_z_gain)
                            * float(hold_capture_y_signal)
                        )
                    foot_z = max(0.28, min(0.47, foot_z))
                    hip_target, knee_target = _leg_ik(foot_x, foot_z)
                    data.ctrl[joint_to_act[hip_name]] = hip_target
                    data.ctrl[joint_to_act[knee_name]] = knee_target
                    if is_stance:
                        prefix = roll_name.split("_", 1)[0]
                        stance_feet.append((foot_body_ids[prefix], side_sign, fore_sign))

            if args.stop_after_box_travel is not None and not target_stop_latched:
                if box_travel_for_control >= float(args.stop_after_box_travel):
                    target_stop_latched = True
                    target_stop_step = step
            target_speed_cmd = float(args.hold_target_speed) if target_stop_latched else float(args.target_speed)
            if target_stop_latched:
                summary["target_stop_latched"] = True
                summary["target_stop_step"] = int(target_stop_step)
                summary["target_stop_hold_steps"] = int(step - int(target_stop_step) + 1)

            torso_x = float(data.qpos[7])
            torso_z = float(data.qpos[9])
            vx = float(data.qvel[6])
            qw, qx, qy, qz = (float(data.qpos[10]), float(data.qpos[11]), float(data.qpos[12]), float(data.qpos[13]))
            roll, pitch = _quat_to_roll_pitch(qw, qx, qy, qz)

            data.qfrc_applied[:] = 0.0
            data.xfrc_applied[:] = 0.0
            if args.support_controller_mode in (
                "stance_force",
                "centroidal_stance_force",
                "lqr_stance_force",
                "lqr_additive_stance_force",
            ) and stance_feet:
                target_height_cmd = float(args.target_height)
                support_kd_z = float(args.support_kd_z)
                support_kd_roll = float(args.support_kd_roll)
                support_kd_pitch = float(args.support_kd_pitch)
                support_kp_vx = float(args.support_kp_vx)
                support_max_total_fx = float(args.support_max_total_fx)
                support_max_foot_fz = float(args.support_max_foot_fz)
                support_max_joint_torque = float(args.support_max_joint_torque)
                support_com_scale = float(args.support_com_pre_latch_scale)
                support_fy_scale = float(args.support_fy_pre_latch_scale)
                support_fx_scale = (
                    float(args.support_fx_scale)
                    if args.support_fx_scale is not None
                    else float(args.support_force_scale)
                )
                if target_stop_latched:
                    target_height_cmd += float(args.hold_support_height_offset)
                    support_kd_z *= float(args.hold_support_kd_z_scale)
                    support_kd_roll *= float(args.hold_support_kd_roll_scale)
                    support_kd_pitch *= float(args.hold_support_kd_pitch_scale)
                    support_kp_vx *= float(args.hold_support_kp_vx_scale)
                    support_max_total_fx *= float(args.hold_support_max_fx_scale)
                    support_max_foot_fz *= float(args.hold_support_max_foot_fz_scale)
                    support_max_joint_torque *= float(args.hold_support_max_joint_torque_scale)
                    support_com_scale = float(args.hold_support_com_scale)
                    support_fy_scale = float(args.hold_support_fy_scale)
                    if args.hold_support_fx_scale is not None:
                        support_fx_scale = float(args.hold_support_fx_scale)
                total_fz = (
                    robot_mass_kg * 9.81
                    + float(args.support_kp_z) * (target_height_cmd - torso_z)
                    - support_kd_z * float(data.qvel[8])
                )
                total_fz = max(
                    float(args.support_min_foot_fz) * len(stance_feet),
                    min(support_max_foot_fz * len(stance_feet), total_fz),
                )
                total_fx = support_kp_vx * (target_speed_cmd - vx)
                total_fx = max(-support_max_total_fx, min(support_max_total_fx, total_fx))
                roll_rate = float(data.qvel[9])
                pitch_rate = float(data.qvel[10])
                roll_term = -float(args.support_kp_roll) * roll - support_kd_roll * roll_rate
                pitch_term = -float(args.support_kp_pitch) * pitch - support_kd_pitch * pitch_rate
                robot_com = np.zeros(3, dtype=float)
                if robot_mass_kg > 1e-6:
                    for body_id in robot_body_ids:
                        robot_com += float(model.body(body_id).mass[0]) * data.xipos[body_id]
                    robot_com /= robot_mass_kg
                stance_center_x = sum(float(data.xpos[foot_body_id, 0]) for foot_body_id, _, _ in stance_feet) / len(stance_feet)
                stance_center_y = sum(float(data.xpos[foot_body_id, 1]) for foot_body_id, _, _ in stance_feet) / len(stance_feet)
                com_x_error = float(robot_com[0] - (stance_center_x + float(args.support_com_target_x_offset)))
                com_y_error = float(robot_com[1] - (stance_center_y + float(args.support_com_target_y_offset)))
                support_lqr_fx = 0.0
                support_lqr_fy = 0.0
                support_lqr_active = (
                    args.support_controller_mode in ("lqr_stance_force", "lqr_additive_stance_force")
                    and (target_stop_latched or not bool(args.support_lqr_post_latch_only))
                )
                if support_lqr_active and robot_mass_kg > 1e-6:
                    lqr_vx_error = float(data.qvel[6] - target_speed_cmd)
                    lqr_vy_error = float(data.qvel[7])
                    lqr_ax = -support_lqr_k_pos * com_x_error - support_lqr_k_vel * lqr_vx_error
                    lqr_ay = -support_lqr_k_pos * com_y_error - support_lqr_k_vel * lqr_vy_error
                    support_lqr_fx = float(robot_mass_kg * lqr_ax)
                    support_lqr_fy = float(robot_mass_kg * lqr_ay)
                    support_lqr_fx = max(
                        -float(args.support_lqr_max_fx),
                        min(float(args.support_lqr_max_fx), support_lqr_fx),
                    )
                    support_lqr_fy = max(
                        -float(args.support_lqr_max_fy),
                        min(float(args.support_lqr_max_fy), support_lqr_fy),
                    )
                    summary["support_lqr_active_steps"] = int(summary["support_lqr_active_steps"]) + 1
                    summary["max_abs_support_lqr_fx_n"] = max(
                        float(summary["max_abs_support_lqr_fx_n"]),
                        abs(support_lqr_fx),
                    )
                    summary["max_abs_support_lqr_fy_n"] = max(
                        float(summary["max_abs_support_lqr_fy_n"]),
                        abs(support_lqr_fy),
                    )
                com_shift_x = support_com_scale * (
                    float(args.support_com_x_gain) * com_x_error
                    + float(args.support_com_vx_gain) * float(data.qvel[6])
                )
                com_shift_y = support_com_scale * (
                    float(args.support_com_y_gain) * com_y_error
                    + float(args.support_com_vy_gain) * float(data.qvel[7])
                )
                if float(args.support_com_max_fz_shift) > 0.0:
                    limit = float(args.support_com_max_fz_shift)
                    com_shift_x = max(-limit, min(limit, com_shift_x))
                    com_shift_y = max(-limit, min(limit, com_shift_y))
                summary["max_abs_robot_com_support_x_error_m"] = max(
                    float(summary["max_abs_robot_com_support_x_error_m"]), abs(com_x_error)
                )
                summary["max_abs_robot_com_support_y_error_m"] = max(
                    float(summary["max_abs_robot_com_support_y_error_m"]), abs(com_y_error)
                )
                summary["max_abs_support_com_fz_shift_n"] = max(
                    float(summary["max_abs_support_com_fz_shift_n"]),
                    abs(com_shift_x),
                    abs(com_shift_y),
                )
                if args.support_fy_world_y_source == "box":
                    world_y_error = float(data.xpos[box_id, 1])
                    world_vy = float(data.qvel[1])
                elif args.support_fy_world_y_source == "robot_com":
                    world_y_error = float(robot_com[1])
                    world_vy = float(data.qvel[7])
                else:
                    world_y_error = float(data.xpos[torso_id, 1])
                    world_vy = float(data.qvel[7])
                summary["max_abs_support_world_y_error_m"] = max(
                    float(summary["max_abs_support_world_y_error_m"]), abs(world_y_error)
                )
                total_fy = support_fy_scale * (
                    -float(args.support_fy_roll_gain) * roll
                    -float(args.support_fy_roll_rate_gain) * roll_rate
                    -float(args.support_fy_com_y_gain) * com_y_error
                    -float(args.support_fy_world_y_gain) * world_y_error
                    -float(args.support_fy_world_vy_gain) * world_vy
                )
                if float(args.support_max_total_fy) > 0.0:
                    limit_fy = float(args.support_max_total_fy)
                    total_fy = max(-limit_fy, min(limit_fy, total_fy))
                total_fy_with_lqr = total_fy + support_lqr_fy
                summary["max_abs_support_fy_n"] = max(
                    float(summary["max_abs_support_fy_n"]), abs(total_fy_with_lqr)
                )
                jacp = np.zeros((3, model.nv))
                foot_forces: list[np.ndarray] = []
                use_centroidal_allocation = (
                    args.support_controller_mode == "centroidal_stance_force"
                    or (args.support_controller_mode == "lqr_stance_force" and support_lqr_active)
                )
                if use_centroidal_allocation:
                    wrench_matrix = np.zeros((6, 3 * len(stance_feet)), dtype=float)
                    wrench_center = robot_com if robot_mass_kg > 1e-6 else data.xpos[torso_id]
                    for foot_index, (foot_body_id, _, _) in enumerate(stance_feet):
                        col = 3 * foot_index
                        foot_pos = np.asarray(data.xpos[foot_body_id], dtype=float)
                        lever = foot_pos - wrench_center
                        wrench_matrix[0:3, col:col + 3] = np.eye(3)
                        wrench_matrix[3:6, col:col + 3] = np.array(
                            [
                                [0.0, -lever[2], lever[1]],
                                [lever[2], 0.0, -lever[0]],
                                [-lever[1], lever[0], 0.0],
                            ],
                            dtype=float,
                        )
                    desired_wrench = np.array(
                        [
                            support_fx_scale * total_fx + support_lqr_fx,
                            total_fy_with_lqr,
                            float(args.support_force_scale) * total_fz,
                            roll_term,
                            pitch_term,
                            0.0,
                        ],
                        dtype=float,
                    )
                    solution, *_ = np.linalg.lstsq(wrench_matrix, desired_wrench, rcond=None)
                    max_lateral_force = max(
                        0.0,
                        float(args.support_max_total_fy),
                        float(args.support_lqr_max_fy) if support_lqr_active else 0.0,
                    )
                    for foot_index in range(len(stance_feet)):
                        force = solution[3 * foot_index:3 * foot_index + 3].astype(float)
                        force[0] = float(np.clip(force[0], -support_max_total_fx, support_max_total_fx))
                        force[1] = float(np.clip(force[1], -max_lateral_force, max_lateral_force))
                        force[2] = float(np.clip(force[2], -abs(float(args.support_force_scale)) * support_max_foot_fz, abs(float(args.support_force_scale)) * support_max_foot_fz))
                        foot_forces.append(force)
                else:
                    for _, side_sign, fore_sign in stance_feet:
                        foot_fz = total_fz / len(stance_feet)
                        foot_fz += side_sign * roll_term / max(1.0, len(stance_feet))
                        foot_fz += fore_sign * pitch_term / max(1.0, len(stance_feet))
                        foot_fz += fore_sign * com_shift_x / max(1.0, len(stance_feet))
                        foot_fz += side_sign * com_shift_y / max(1.0, len(stance_feet))
                        foot_fz = max(float(args.support_min_foot_fz), min(support_max_foot_fz, foot_fz))
                        foot_forces.append(
                            np.array(
                                [
                                    (support_fx_scale * total_fx + support_lqr_fx) / len(stance_feet),
                                    total_fy_with_lqr / len(stance_feet),
                                    float(args.support_force_scale) * foot_fz,
                                ],
                                dtype=float,
                            )
                        )
                for (foot_body_id, _, _), foot_force in zip(stance_feet, foot_forces):
                    mujoco.mj_jacBodyCom(model, data, jacp, None, foot_body_id)
                    generalized = jacp.T @ foot_force
                    for dof_id in actuated_dof_ids:
                        data.qfrc_applied[dof_id] += float(
                            np.clip(
                                generalized[dof_id],
                                -support_max_joint_torque,
                                support_max_joint_torque,
                            )
                        )
                summary["support_joint_torque_write_count"] += 1
            if args.assist_mode == "body_force":
                force_x = 240.0 * (target_speed_cmd - vx)
                force_z = 900.0 * (args.target_height - torso_z) - 55.0 * float(data.qvel[8])
                torque_x = -180.0 * roll - 25.0 * float(data.qvel[9])
                torque_y = -180.0 * pitch - 25.0 * float(data.qvel[10])
                data.qfrc_applied[6] = np.clip(force_x, -float(args.max_assist_force_x), float(args.max_assist_force_x))
                data.xfrc_applied[torso_id, 0] = np.clip(force_x, -float(args.max_assist_force_x), float(args.max_assist_force_x))
                data.xfrc_applied[torso_id, 2] = np.clip(force_z, -float(args.max_assist_force_z), float(args.max_assist_force_z))
                data.xfrc_applied[torso_id, 3] = np.clip(torque_x, -float(args.max_assist_torque), float(args.max_assist_torque))
                data.xfrc_applied[torso_id, 4] = np.clip(torque_y, -float(args.max_assist_torque), float(args.max_assist_torque))
                summary["external_force_write_count"] += 1
                summary["external_torque_write_count"] += 1
            if args.retention_force_mode == "relative_spring":
                current_offset_x = float(data.xpos[box_id, 0] - data.xpos[torso_id, 0])
                current_offset_y = float(data.xpos[box_id, 1] - data.xpos[torso_id, 1])
                current_offset_z = float(data.xpos[box_id, 2] - data.xpos[torso_id, 2])
                current_rel_vx = float(data.cvel[box_id, 3] - data.cvel[torso_id, 3])
                current_rel_vy = float(data.cvel[box_id, 4] - data.cvel[torso_id, 4])
                current_rel_vz = float(data.cvel[box_id, 5] - data.cvel[torso_id, 5])
                retention_fx = args.retention_kp_x * (initial_box_torso_offset_x - current_offset_x) - args.retention_kd_x * current_rel_vx
                retention_fy = args.retention_kp_y * (initial_box_torso_offset_y - current_offset_y) - args.retention_kd_y * current_rel_vy
                retention_fz = args.retention_kp_z * (initial_box_torso_offset_z - current_offset_z) - args.retention_kd_z * current_rel_vz
                retention_fx = float(np.clip(retention_fx, -float(args.retention_max_force_x), float(args.retention_max_force_x)))
                retention_fy = float(np.clip(retention_fy, -float(args.retention_max_force_y), float(args.retention_max_force_y)))
                retention_fz = float(np.clip(retention_fz, -float(args.retention_max_force_z), float(args.retention_max_force_z)))
                data.xfrc_applied[box_id, 0] += retention_fx
                data.xfrc_applied[box_id, 1] += retention_fy
                data.xfrc_applied[box_id, 2] += retention_fz
                data.xfrc_applied[torso_id, 0] -= retention_fx
                data.xfrc_applied[torso_id, 1] -= retention_fy
                data.xfrc_applied[torso_id, 2] -= retention_fz
                summary["box_retention_force_write_count"] += 1
            mujoco.mj_step(model, data)

            if step % 20 == 0 or step == args.steps - 1:
                torso_x = float(data.xpos[torso_id, 0])
                torso_z = float(data.xpos[torso_id, 2])
                box_x = float(data.xpos[box_id, 0])
                box_y = float(data.xpos[box_id, 1])
                box_z = float(data.xpos[box_id, 2])
                vx = float(data.qvel[6])
                qw, qx, qy, qz = (float(data.xquat[torso_id, 0]), float(data.xquat[torso_id, 1]), float(data.xquat[torso_id, 2]), float(data.xquat[torso_id, 3]))
                roll, pitch = _quat_to_roll_pitch(qw, qx, qy, qz)
                tilt = math.hypot(roll, pitch)
                torso_travel = torso_x - initial_torso_x
                box_travel = box_x - initial_box_x
                box_rel_error = math.hypot(
                    (box_x - torso_x) - initial_box_torso_offset_x,
                    (box_z - torso_z) - initial_box_torso_offset_z,
                )
                fall = int(torso_z < args.fall_height or tilt > args.max_tilt_rad)
                box_drop = int(box_z < args.box_drop_height or box_rel_error > 0.22)
                summary["completed_steps"] = int(step + 1)
                summary["max_torso_travel_x_m"] = max(float(summary["max_torso_travel_x_m"]), float(torso_travel))
                summary["max_box_travel_x_m"] = max(float(summary["max_box_travel_x_m"]), float(box_travel))
                summary["final_torso_travel_x_m"] = float(torso_travel)
                summary["final_box_travel_x_m"] = float(box_travel)
                summary["min_torso_z_m"] = (
                    torso_z if summary["min_torso_z_m"] is None else min(float(summary["min_torso_z_m"]), torso_z)
                )
                summary["min_box_z_m"] = box_z if summary["min_box_z_m"] is None else min(float(summary["min_box_z_m"]), box_z)
                summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), float(tilt))
                summary["max_abs_box_y_m"] = max(float(summary["max_abs_box_y_m"]), abs(box_y))
                summary["fall_events"] += fall
                summary["box_drop_events"] += box_drop
                summary["max_box_torso_relative_offset_error_m"] = max(
                    float(summary["max_box_torso_relative_offset_error_m"]), float(box_rel_error)
                )
                summary["final_box_torso_relative_offset_error_m"] = float(box_rel_error)
                writer.writerow([step, step * dt, torso_x, torso_z, box_x, box_y, box_z, roll, pitch, tilt, vx, target_speed_cmd, box_rel_error, fall, box_drop])
                print(
                    "[STATE] "
                    f"step={step} torso_x={torso_x:.3f} box_x={box_x:.3f} "
                    f"box_z={box_z:.3f} tilt={tilt:.3f} rel={box_rel_error:.3f} "
                    f"v_cmd={target_speed_cmd:.3f} fall={fall} drop={box_drop}",
                    flush=True,
                )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")


if __name__ == "__main__":
    main()
