#!/usr/bin/env python3
"""Check local G1 WBC assets without launching Isaac Sim.

This loads the official Arena RobotModel and HomieV2 ONNX policies. Run it on a
compute node, not on a login node.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to load WBC assets on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local Arena G1 WBC assets.")
    parser.add_argument(
        "--wbc-asset-root",
        type=Path,
        default=Path(
            "/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy"
        ),
    )
    return parser.parse_args()


def main() -> None:
    _refuse_login_node()
    args = parse_args()
    root = args.wbc_asset_root.expanduser().resolve()
    stand_onnx = root / "models/homie_v2/stand.onnx"
    walk_onnx = root / "models/homie_v2/walk.onnx"
    robot_asset_path = root / "robot_model/g1"
    robot_urdf = robot_asset_path / "g1_29dof_with_hand.urdf"
    required_paths = (stand_onnx, walk_onnx, robot_urdf, robot_asset_path / "meshes")
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local WBC asset(s): " + ", ".join(missing))

    from isaaclab_arena_g1.g1_env.g1_supplemental_info import G1SupplementalInfo
    from isaaclab_arena_g1.g1_env.robot_model import RobotModel
    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.config.configs import HomieV2Config
    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.wbc_policy_factory import get_wbc_policy

    robot_model = RobotModel(
        str(robot_urdf),
        str(robot_asset_path),
        supplemental_info=G1SupplementalInfo(),
    )
    wbc_config = HomieV2Config()
    wbc_config.wbc_model_path = f"{stand_onnx},{walk_onnx}"
    policy = get_wbc_policy("g1", robot_model, wbc_config, num_envs=1)
    print("[OK] Local G1 WBC assets loaded.")
    print(f"[OK] Robot DOFs: {robot_model.num_dofs}")
    print(f"[OK] Policy: {policy.__class__.__name__}")


if __name__ == "__main__":
    main()
