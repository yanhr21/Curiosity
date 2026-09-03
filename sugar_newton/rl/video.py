# SPDX-License-Identifier: BSD-3-Clause
"""Periodic evaluation rollouts, rendered and pushed to Weights & Biases.

Scalar rewards say a policy is improving; they do not say whether it is *carrying the
box*. This renders one deterministic rollout every N iterations and logs it as a wandb
video, so quality is visible rather than inferred.

Beside the scene it draws both hands' tactile heatmaps, in the same format as
``validation/make_loop_video.py`` and from the same code
(:mod:`sugar_newton.tactile.handmap`). **The tactile field is not an observation** -- the
policy never sees it, and the HUD says so on every frame. It is here because a lift curve
does not distinguish a real grasp from a wrist wedged under the box, and the map does.

Three deliberate choices:

* **A separate one-world environment.** The training worlds are replicated at zero spacing
  and therefore sit on top of each other, so rendering the training model shows every
  robot superimposed. A dedicated ``num_envs=1`` env is built once and reused -- with the
  same collider settings as training, or the map would describe different geometry.
* **Deterministic actions.** The rollout uses the policy's mean, not a sample, so
  successive videos differ because the policy changed rather than because the noise did.
  The clip and start frame are fixed for the same reason.
* **Compositing is not timed and not on the training path.** Drawing a matplotlib panel per
  frame costs more than the physics step, so it happens once per evaluation, after the
  rollout, and never inside a throughput measurement.

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
                 cam_offset=(2.1, -2.1, 0.95), mu: float = 1.0, substeps: int = 4,
                 tactile: bool = True, canvas_tris: int = 3000,
                 box_tris: int = 2000, hand_tris: int = 0, margin: float = 0.0,
                 privileged_policy: bool = False):
        self.clip, self.start, self.frames, self.fps = clip, start, frames, fps
        self.device, self.cam_offset = device, np.asarray(cam_offset, dtype=float)
        self.mu, self.substeps = mu, substeps
        self.tactile, self.canvas_tris = tactile, canvas_tris
        self.privileged_policy = privileged_policy
        self.box_tris, self.hand_tris, self.margin = box_tris, hand_tris, margin
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env: CarryBoxEnv | None = None
        self.viewer = None
        self._warned = False
        self._warned_tactile = False
        self._hands: dict | None = None
        self._box: set | None = None

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

            # Same collider settings as training, or the map would describe different
            # geometry from the one the policy is being trained against.
            self.env = CarryBoxEnv(num_envs=1, clip_names=[self.clip], episode_length=10**9,
                                   mu=self.mu, substeps=self.substeps, device=self.device,
                                   box_tris=self.box_tris, hand_tris=self.hand_tris,
                                   margin=self.margin,
                                   # tactile panels read the contact set back, which needs
                                   # the solver's capacity to match the Newton buffer
                                   contact_readback=self.tactile,
                                   # Let the clip play out. With tracking termination on, an
                                   # early policy resets every ~25 steps and the video is the
                                   # same half second on repeat; `video/drift_step` below
                                   # keeps the number that reset was reporting.
                                   track_termination=False)
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
        # Pin clip and frame so successive evaluations are comparable; see CarryBoxEnv.reset.
        env.reset(torch.zeros(1, dtype=torch.long, device=env.device),
                  start=self.start, motion=0)

        was_training = getattr(policy, "training", False)
        policy.eval()
        box0 = env._body_q()[0, env.box_body, :3].clone()
        peak = float(box0[2])
        frames = []
        film = self._film()
        buf = None
        drift_step = -1          # first frame the policy left the reference by >0.3 m
        for _ in range(self.frames):
            obs = self._policy_obs(policy, env)
            action = self._act_mean(policy, obs)
            env.step(action)
            if drift_step < 0 and bool(env.drifted[0]):
                drift_step = len(frames)
            bz = float(env._body_q()[0, env.box_body, 2])
            peak = max(peak, bz)
            self._aim(env.state_0)
            self.viewer.begin_frame(len(frames) / self.fps)
            self.viewer.log_state(env.state_0)
            self.viewer.end_frame()
            buf = self.viewer.get_frame(buf)
            frames.append(buf.numpy().copy())
            if film is not None:
                from sugar_newton.tactile.handmap import read_frame

                # after the solve: MuJoCo replaces the contact set during it
                env.solver.update_contacts(env.contacts, env.state_0)
                film.add(*read_frame(env.contacts, self._hands, self._box, film.sides))
        if was_training:
            policy.train()

        ref = env.ref["obj_pos"][env.motion_id[0], self.start:self.start + self.frames, 2]
        stats = {
            "video/box_lift": peak - float(box0[2]),
            "video/box_lift_reference": float(ref.max()) - float(ref[0]),
            "video/frames": len(frames),
            # Where training would have cut the episode. Rising towards `frames` is the
            # clearest single sign the tracker is improving, and it is not visible in the
            # video any more now that the video plays past it.
            "video/drift_step": float(drift_step if drift_step >= 0 else len(frames)),
            "video/tracked_to_end": float(drift_step < 0),
        }

        if film is not None:
            try:
                path = self._composite(film, frames, iteration, stats)
                stats["video/load_bearing_contacts"] = float(np.mean(film.loads))
                stats["video/net_force_n"] = float(np.mean(film.nets))
                return path, stats
            except Exception as exc:
                # a failed panel must not cost the rollout that has already been paid for
                print(f"[video] tactile panel failed, writing scene only: "
                      f"{type(exc).__name__}: {exc}")

        path = self.out_dir / f"rollout_{iteration:06d}.mp4"
        imageio.mimwrite(str(path), frames, fps=self.fps, quality=7,
                         macro_block_size=None)
        return str(path), stats

    # ---- tactile panels -------------------------------------------------------------
    def _film(self):
        """A :class:`TactileFilm` for this rollout, or None if tactile panels are off."""
        if not self.tactile:
            return None
        try:
            from sugar_newton.tactile.handmap import TactileFilm, hand_shapes

            if self._hands is None:
                self._hands, self._box = hand_shapes(self.env.model)
            sides = [s for s in ("left", "right") if s in self._hands]
            if not sides:
                raise RuntimeError("no rubber-hand shapes found in the model")
            return TactileFilm(sides, canvas=self.canvas_tris)
        except Exception as exc:
            if not self._warned_tactile:
                print(f"[video] tactile panels disabled: {type(exc).__name__}: {exc}")
                self._warned_tactile = True
            return None

    def _composite(self, film, frames, iteration: int, stats: dict) -> str:
        """Scene beside both hand maps, in the format of validation/make_loop_video.py.

        Deliberately outside any timing: a matplotlib panel per frame costs more than the
        simulation step does, so this is monitoring overhead paid once per evaluation, never
        something a throughput number should include.
        """
        import matplotlib

        matplotlib.use("Agg")
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt

        atlases, maps, norm = film.develop()
        sides = film.sides

        fig = plt.figure(figsize=(16.8, 6.6))
        axl = fig.add_axes([0.004, 0.02, 0.545, 0.87])
        axl.set_axis_off()
        im = axl.imshow(frames[0])
        pcs = {}
        for i, side in enumerate(sides):
            ax = fig.add_axes([0.565 + 0.185 * i, 0.04, 0.175, 0.80])
            pcs[side] = atlases[side].draw(ax, maps[side][0], norm)
            atlases[side].label_digits(ax, fontsize=7.5)
            ax.set_title(f"{side} hand", fontsize=12)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="inferno"),
                     cax=fig.add_axes([0.945, 0.12, 0.013, 0.62]),
                     label="contact force this frame [N]")
        fig.text(0.655, 0.90, "G1 rubber-hand collider, palm view, fingertips up",
                 ha="center", fontsize=11, color="#444444")
        sup = fig.text(0.5, 0.965, "", ha="center", fontsize=13)
        hud = fig.text(0.012, 0.985, "", ha="left", va="top", fontsize=11,
                       family="monospace",
                       bbox=dict(boxstyle="round,pad=0.35", fc="#ffffffcc", ec="#999999"))
        lift, ref = stats["video/box_lift"], stats["video/box_lift_reference"]
        drift = int(stats.get("video/drift_step", len(frames)))

        path = self.out_dir / f"rollout_{iteration:06d}.mp4"
        try:
            writer = imageio.get_writer(str(path), mode="I", fps=self.fps,
                                        macro_block_size=1, quality=8)
        except Exception:                       # no ffmpeg in the container
            path = path.with_suffix(".gif")
            writer = imageio.get_writer(str(path), mode="I", duration=1.0 / self.fps, loop=0)

        for k in range(len(frames)):
            im.set_data(frames[k])
            tot = 0.0
            for side in sides:
                atlases[side].paint(pcs[side], maps[side][k], norm)
                tot += float(maps[side][k].sum())
            sup.set_text(f"iter {iteration}   |   clip {self.clip} frame {self.start + k}"
                         f"   |   both hands: {film.loads[k]} load-bearing contacts, "
                         f"sum|f| {tot:.0f} N, net {film.nets[k]:.0f} N")
            track = (f"tracking  LOST at frame {drift}" if k >= drift > -1
                     else f"tracking  OK (>0.3 m ends training episodes)")
            hud.set_text(f"iteration {iteration}\n"
                         f"lift      {lift:.3f} m  ({100 * lift / max(ref, 1e-9):.0f}% of "
                         f"{ref:.3f} m ref)\n"
                         f"{track}\n"
                         f"tactile   NOT an input to this policy\n"
                         f"{film.canvas_tris}-tri tactile canvas")
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        writer.close()
        plt.close(fig)
        return str(path)

    # --- the policy interface differs between rsl_rl versions; keep this in one place
    def _policy_obs(self, policy, env):
        from sugar_newton.rl import obs_890
        from tensordict import TensorDict

        if self.privileged_policy:
            # Must mirror CarryBoxVecEnv._obs for the same stage: a refiner's actor was
            # trained with the 890-D vector under the "policy" key, so handing it the
            # 510-D one here would silently evaluate a different network's input.
            priv = obs_890.build(env, teacher=False)
            return TensorDict({"policy": priv, "critic": priv},
                              batch_size=[1], device=env.device)

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
