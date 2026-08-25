#!/usr/bin/env python3
"""Render actual Isaac Sim world video for one Carry9-to-Kick recovery policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SUGAR = ROOT / "SUGAR"
sys.path.insert(0, str(SUGAR / "scripts/sugar_rl"))
os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("DISPLAY", "")
job_id = os.environ.get("SLURM_JOB_ID", "local")
os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_recovery_video_{job_id}")
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT", f"/tmp/Curiosity_recovery_video_unitree_{job_id}"
)
os.chdir(SUGAR)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--seed", type=int, default=181629)
parser.add_argument("--carry-prefix-steps", type=int, default=9)
parser.add_argument("--profile-index", type=int, default=0)
parser.add_argument("--num-profiles", type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.carry_prefix_steps <= 0:
    parser.error("carry prefix must be positive")
if args.profile_index < 0:
    parser.error("profile index must be nonnegative")
if args.num_profiles is not None and args.num_profiles <= args.profile_index:
    parser.error("num profiles must include the selected profile index")
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
import sugar_rl.tasks  # noqa: F401,E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_tracker.kick_box_carry9_recovery_v2_env_cfg import (  # noqa: E402
    RobotEnvCfg,
)
from sugar_rl.utils.online_cross_skill_recovery_wrapper import (  # noqa: E402
    OnlineCrossSkillRecoveryVecEnvWrapper,
    _load_released_tracker_actor,
)


class FfmpegRgbWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int = 20) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        self.process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def _camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/WorldCamera",
        update_period=0.02,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.6, 3.6, 2.4),
            rot=(0.3043649418, 0.2319667899, 0.5600173703, 0.7348019703),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        width=960,
        height=540,
    )


def _overlay(
    rgb: np.ndarray,
    *,
    phase: str,
    step: int,
    displacement_m: float,
    foot_contact: bool,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (960, 104), (255, 255, 255), -1)
    lines = (
        f"{args.label} | actual IsaacLab/PhysX world | profile {args.profile_index}",
        f"phase: {phase} | step: {step}",
        f"box planar displacement: {displacement_m:.3f} m | foot-box contact: {foot_contact}",
    )
    for row, line in enumerate(lines):
        cv2.putText(
            bgr,
            line,
            (18, 28 + 32 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.exists() or output.exists():
        raise FileExistsError("checkpoint must exist and output video must be new")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = RobotEnvCfg()
    cfg.scene.num_envs = args.num_profiles or (args.profile_index + 1)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.episode_length_s = max(
        cfg.episode_length_s,
        (1 + args.carry_prefix_steps + args.steps + 25)
        * cfg.sim.dt
        * cfg.decimation,
    )
    cfg.observations.policy.enable_corruption = False
    cfg.terminations.physical_invalid = None
    cfg.rewards.physical_invalid_penalty = None
    cfg.scene.world_camera = _camera_cfg()
    cfg.sim.render_interval = cfg.decimation
    env = gym.make(
        "Sugar-G129dof-KickBox-Carry9-Recovery",
        cfg=cfg,
        render_mode="rgb_array",
    )
    base = env.unwrapped
    writer = FfmpegRgbWriter(output, width=960, height=540)
    initial_object_xy: np.ndarray | None = None

    def append_prefix(phase: str, step: int) -> None:
        nonlocal initial_object_xy
        rgb = base.scene["world_camera"].data.output["rgb"][
            args.profile_index, ..., :3
        ]
        frame = rgb.detach().cpu().numpy()
        obj_xy = (
            base.scene["obj"].data.root_pos_w[args.profile_index, :2]
            .detach()
            .cpu()
            .numpy()
        )
        if initial_object_xy is None:
            initial_object_xy = obj_xy.copy()
        writer.append(
            _overlay(
                frame,
                phase=phase,
                step=step,
                displacement_m=float(np.linalg.norm(obj_xy - initial_object_xy)),
                foot_contact=False,
            )
        )

    wrapped = OnlineCrossSkillRecoveryVecEnvWrapper(
        env,
        clip_actions=100.0,
        carry_tracker_checkpoint=SUGAR / "demo_ckpts/CarryBox/tracker.pt",
        kick_tracker_checkpoint=SUGAR / "demo_ckpts/KickBox/tracker.pt",
        carry_generator_checkpoint=SUGAR / "demo_ckpts/CarryBox/generator.ckpt",
        carry_prefix_steps=args.carry_prefix_steps,
        audit_path=output.with_suffix(".prefix_audit.json"),
        prefix_frame_callback=append_prefix,
    )
    actor = _load_released_tracker_actor(checkpoint, wrapped.device)
    observations = wrapped.get_observations()
    try:
        with torch.inference_mode():
            for step in range(args.steps):
                action = actor(observations["policy"])
                observations, _, done, _ = wrapped.step(action)
                if torch.any(done):
                    raise RuntimeError("video recovery window reset unexpectedly")
                left = base.scene.sensors["left_foot_forces"].data.force_matrix_w_history
                right = base.scene.sensors["right_foot_forces"].data.force_matrix_w_history
                foot_contact = bool(
                    torch.linalg.vector_norm(
                        left[args.profile_index, -1, 0, 0], dim=-1
                    ).item()
                    > 0.1
                    or torch.linalg.vector_norm(
                        right[args.profile_index, -1, 0, 0], dim=-1
                    ).item()
                    > 0.1
                )
                rgb = base.scene["world_camera"].data.output["rgb"][
                    args.profile_index, ..., :3
                ]
                frame = rgb.detach().cpu().numpy()
                obj_xy = (
                    base.scene["obj"].data.root_pos_w[args.profile_index, :2]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if initial_object_xy is None:
                    initial_object_xy = obj_xy.copy()
                writer.append(
                    _overlay(
                        frame,
                        phase="learned Kick recovery",
                        step=step,
                        displacement_m=float(
                            np.linalg.norm(obj_xy - initial_object_xy)
                        ),
                        foot_contact=foot_contact,
                    )
                )
    finally:
        writer.close()
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
