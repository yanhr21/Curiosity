# SPDX-License-Identifier: BSD-3-Clause
"""Periodic evaluation rollouts, rendered and pushed to Weights & Biases.

Scalar rewards say a policy is improving; they do not say whether it is *carrying the
box*. This renders one deterministic rollout every N iterations and logs it as a wandb
video, so quality is visible rather than inferred.

Two deliberate choices:

* **A separate one-world environment.** The training worlds are replicated at zero spacing
  and therefore sit on top of each other, so rendering the training model shows every
  robot superimposed. A dedicated ``num_envs=1`` env is built once and reused.
* **Deterministic actions.** The rollout uses the policy's mean, not a sample, so
  successive videos differ because the policy changed rather than because the noise did.
  The clip and start frame are fixed for the same reason.

Rendering needs headless EGL on the NVIDIA driver; ``renders/render_env_egl.sh`` in the
Newton checkout sets that up. Without it pyglet falls back to software rasterisation and
each frame costs seconds instead of milliseconds -- so the recorder checks and says so
once, rather than quietly making training slow.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import warp as wp

from sugar_newton.rl.carrybox_env import CarryBoxEnv


class VideoRecorder:
    """Render a deterministic rollout of the current policy and hand back an mp4 path."""

    def __init__(self, clip: str = "data_000", start: int = 0, frames: int = 400,
                 fps: int = 30, out_dir: str = "videos", device: str = "cuda:0",
                 cam_offset=(2.1, -2.1, 0.95), mu: float = 1.0, substeps: int = 4):
        self.clip, self.start, self.frames, self.fps = clip, start, frames, fps
        self.device, self.cam_offset = device, np.asarray(cam_offset, dtype=float)
        self.mu, self.substeps = mu, substeps
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env: CarryBoxEnv | None = None
        self.viewer = None
        self._warned = False

    # ---- lazy setup: building the env and the GL context costs a few seconds, and a
    # run that never reaches the first video interval should not pay it
    def _ensure(self) -> bool:
        if self.viewer is not None:
            return True
        try:
            import pyglet

            if os.environ.get("G1_XVFB") != "1":
                pyglet.options["headless"] = True
            from newton.viewer import ViewerGL

            self.env = CarryBoxEnv(num_envs=1, clip_names=[self.clip], episode_length=10**9,
                                   mu=self.mu, substeps=self.substeps, device=self.device)
            self.viewer = ViewerGL(headless=os.environ.get("G1_XVFB") != "1")
            self.viewer.set_model(self.env.model)
            if os.environ.get("PYOPENGL_PLATFORM") != "egl" and not self._warned:
                print("[video] PYOPENGL_PLATFORM is not egl; rendering may fall back to "
                      "software and be very slow. Source renders/render_env_egl.sh.")
                self._warned = True
            return True
        except Exception as exc:                       # never take training down for a video
            if not self._warned:
                print(f"[video] disabled: {type(exc).__name__}: {exc}")
                self._warned = True
            return False

    def _aim(self, state) -> None:
        bq = state.body_q.numpy()
        pel, box = bq[0, :3], bq[self.env.box_body, :3]
        if not np.isfinite(box).all():
            box = pel
        mid = 0.5 * (pel + box)
        cam = mid + self.cam_offset
        # aim slightly BELOW the midpoint: the box sits low and was being clipped
        # off the bottom of the frame when the camera looked level at the pelvis
        d = np.array([mid[0], mid[1], mid[2] - 0.10]) - cam
        d /= max(np.linalg.norm(d), 1e-9)
        self.viewer.set_camera(
            wp.vec3(*cam.tolist()),
            math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0)))),
            math.degrees(math.atan2(float(d[1]), float(d[0]))))

    @torch.no_grad()
    def record(self, policy, iteration: int) -> tuple[str | None, dict]:
        """Roll out, render, encode. Returns (path, stats); path is None if unavailable."""
        if not self._ensure():
            return None, {}
        import imageio.v2 as imageio

        env = self.env
        env.reset()
        env.t[:] = self.start
        env.reset(torch.zeros(1, dtype=torch.long, device=env.device))

        was_training = getattr(policy, "training", False)
        policy.eval()
        box0 = env._body_q()[0, env.box_body, :3].clone()
        peak = float(box0[2])
        frames = []
        for _ in range(self.frames):
            obs = self._policy_obs(policy, env)
            action = self._act_mean(policy, obs)
            env.step(action)
            bz = float(env._body_q()[0, env.box_body, 2])
            peak = max(peak, bz)
            self._aim(env.state_0)
            self.viewer.begin_frame(len(frames) / self.fps)
            self.viewer.log_state(env.state_0)
            self.viewer.end_frame()
            frames.append(self.viewer.get_frame().numpy())
        if was_training:
            policy.train()

        path = self.out_dir / f"rollout_{iteration:06d}.mp4"
        imageio.mimwrite(str(path), frames, fps=self.fps, quality=7,
                         macro_block_size=None)
        ref = env.ref["obj_pos"][env.motion_id[0], self.start:self.start + self.frames, 2]
        stats = {
            "video/box_lift": peak - float(box0[2]),
            "video/box_lift_reference": float(ref.max()) - float(ref[0]),
            "video/frames": len(frames),
        }
        return str(path), stats

    # --- the policy interface differs between rsl_rl versions; keep this in one place
    def _policy_obs(self, policy, env):
        from sugar_newton.rl import obs_890
        from tensordict import TensorDict

        return TensorDict(
            {"policy": env.observe(),
             "critic": obs_890.build(env, teacher=False),
             "teacher": obs_890.build(env, teacher=True)},
            batch_size=[1], device=env.device)

    def _act_mean(self, policy, obs):
        for fn in ("act_inference", "act_mean"):
            f = getattr(policy, fn, None)
            if f is not None:
                return f(obs)
        return policy.actor(obs["policy"])
