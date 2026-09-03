# SPDX-License-Identifier: BSD-3-Clause
"""Does BCPPO's teacher actually work in Newton? Roll it out and compare with the tracker.

BCPPO distils ``refiner_model10000.pt``, an 890-D privileged network, into the 510-D policy.
The checkpoint we have *validated* in Newton is a different one -- SUGAR's official
``demo_ckpts/CarryBox/tracker.pt``, 510-D, the one that lifts the box to 0.63 m. The refiner
had never been run here, so "the distill loss is falling" told us the student matches the
teacher and nothing about whether the teacher is worth matching. Distilling a teacher whose
targets are wrong for these dynamics looks exactly like healthy training: loss down, task
performance flat.

    python -m sugar_newton.rl.check_teacher

Both policies drive the SAME environment from the same pinned initial state, so the numbers
are directly comparable, and both are run at their action MEAN (no sampling) because that is
what distillation regresses onto. Reported per policy:

  drift_step   first frame past the 0.3 m tracking bound -- where training would cut the
               episode. This is the number to compare against the student's ~45.
  box_lift     peak box height above its start, against the reference's 0.628 m.
  |a| , sat    action magnitude and the fraction of actions beyond +-1, since a teacher whose
               targets saturate in this actuator model is a teacher that cannot be followed.

If the refiner drifts as early as the student already does, more iterations will not fix it
and the gap is the teacher or the sim, not the optimisation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import warp as wp

HERE = Path(__file__).resolve().parent
SUGAR = HERE.parents[1] / "SUGAR"
TRACKER_NPZ = HERE.parent / "validation" / "tracker_actor.npz"
REFINER_PT = (HERE.parents[1] / "experiments/sugar_reproduction/outputs/final"
              / "official_sugar/baseline/ckpts/refiner_model10000.pt")


def load_refiner(path, device):
    """The 890-D privileged actor, as a torch MLP with ELU (rsl_rl's ActorCritic layout)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.")})
    layers, prev = [], None
    for i in idx:
        w, b = sd[f"actor.{i}.weight"], sd[f"actor.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data, lin.bias.data = w.clone(), b.clone()
        if prev is not None:
            layers.append(torch.nn.ELU())
        layers.append(lin)
        prev = i
    net = torch.nn.Sequential(*layers).to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, int(sd["actor.0.weight"].shape[1])


def load_tracker(path, device):
    """SUGAR's validated 510-D tracker, stored as plain arrays by make_policy_assets."""
    z = np.load(path)
    idx = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("actor_")})
    layers = []
    for n, i in enumerate(idx):
        w, b = z[f"actor_{i}_weight"], z[f"actor_{i}_bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = torch.as_tensor(w).float()
        lin.bias.data = torch.as_tensor(b).float()
        if n:
            layers.append(torch.nn.ELU())
        layers.append(lin)
    net = torch.nn.Sequential(*layers).to(device).eval()
    return net, int(z["actor_0_weight"].shape[1])


@torch.no_grad()
def rollout(env, net, group: str, frames: int) -> dict:
    """Drive ``env`` from its pinned start with ``net`` reading observation ``group``."""
    from sugar_newton.rl import obs_890

    env.reset(torch.zeros(1, dtype=torch.long, device=env.device), start=0, motion=0)
    box0 = float(env._body_q()[0, env.box_body, 2])
    peak, drift, mags, sat = box0, -1, [], []

    for k in range(frames):
        if group == "policy":
            obs = env.observe()
        else:
            obs = obs_890.build(env, teacher=(group == "teacher"))
        act = net(obs)
        mags.append(float(act.abs().mean()))
        sat.append(float((act.abs() > 1.0).float().mean()))
        env.step(act)
        if drift < 0 and bool(env.drifted[0]):
            drift = k
        peak = max(peak, float(env._body_q()[0, env.box_body, 2]))

    ref = env.ref["obj_pos"][env.motion_id[0], 0:frames, 2]
    return {"drift_step": drift if drift >= 0 else frames,
            "tracked_to_end": drift < 0,
            "box_lift": peak - box0,
            "box_lift_reference": float(ref.max()) - float(ref[0]),
            "action_mag": float(np.mean(mags)),
            "action_saturated": float(np.mean(sat))}


@torch.no_grad()
def agreement(env, refiner, tracker, frames: int) -> list[tuple[float, float, float]]:
    """Both nets on the SAME states, driven by the known-good tracker.

    Driving with the *tracker* is what makes this readable: both nets then see states from a
    trajectory that at least partially works, so the refiner is never blamed for a mess it
    was fed itself.

    On what this can and cannot conclude, because it is easy to overread. The two networks
    are teacher and student of one distillation, but BCPPO's stage 3 decays the distillation
    weight to zero and optimises the student with PPO, so the student is deliberately no
    longer a copy of its teacher. A raw difference here is therefore NOT an error signal,
    and there is no published teacher/student agreement figure to compare against. (The
    documented 0.088 belongs to a different check: the tracker reproducing Isaac's own
    recorded tracker actions, which validates the 510-D observation pipeline only.)

    What is still informative is *magnitude*. Both nets emit joint position targets in the
    same normalised action space, so |a_refiner| persistently far above |a_tracker| -- and
    especially at frame 0, where the robot sits exactly on the reference and nothing has
    integrated -- says the teacher is being driven outside the range it was trained to emit.
    Whether that comes from a malformed observation or from saturated actuation needs the
    Isaac dumps to settle; this function only tells you it is happening.
    """
    from sugar_newton.rl import obs_890

    env.reset(torch.zeros(1, dtype=torch.long, device=env.device), start=0, motion=0)
    rows = []
    for _ in range(frames):
        a_t = tracker(env.observe())
        a_r = refiner(obs_890.build(env, teacher=True))
        rows.append((float((a_r - a_t).pow(2).mean().sqrt()),
                     float(a_t.abs().mean()), float(a_r.abs().mean())))
        env.step(a_t)
    return rows


def side_by_side(made: list[tuple[str, str]], out_dir: str) -> str:
    """Stitch the recorded clips horizontally, teacher left, tracker right.

    The two runs can stop at different lengths, so the shorter one holds its last frame
    rather than the pair being truncated to it -- cutting the video at the first policy to
    end would hide precisely what the other one goes on to do.
    """
    import imageio.v2 as imageio

    reels = [imageio.mimread(p, memtest=False) for _, p in made]
    n = max(len(r) for r in reels)
    h = max(f.shape[0] for r in reels for f in r)

    def pad(f):
        if f.shape[0] == h:
            return f[..., :3]
        top = (h - f.shape[0]) // 2
        out = np.zeros((h, f.shape[1], 3), dtype=f.dtype)
        out[top:top + f.shape[0]] = f[..., :3]
        return out

    dst = str(Path(out_dir) / "side_by_side.mp4")
    with imageio.get_writer(dst, fps=30, macro_block_size=1) as w:
        for k in range(n):
            w.append_data(np.hstack([pad(r[min(k, len(r) - 1)]) for r in reels]))
    return dst


class TeacherAdapter(torch.nn.Module):
    """Present the refiner through the interface ``VideoRecorder`` calls.

    The recorder hands a TensorDict of all three observation groups to ``act_inference``, so
    the teacher only needs to pick its own group out of it.
    """

    def __init__(self, net: torch.nn.Module, group: str = "teacher"):
        super().__init__()
        self.net, self.group = net, group

    def act_inference(self, obs):
        return self.net(obs[self.group])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--box-tris", type=int, default=2000)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--refiner", default=str(REFINER_PT))
    p.add_argument("--tracker", default=str(TRACKER_NPZ))
    p.add_argument("--compare", action="store_true",
                   help="teacher/student action agreement on tracker-driven states")
    p.add_argument("--video", default="",
                   help="directory for a video of the TEACHER driving (with tactile panels)")
    args = p.parse_args()

    wp.init()
    from sugar_newton.rl.carrybox_env import CarryBoxEnv

    # episode_length huge so only the tracking bound can end it, and drift is observable
    # without the env resetting underneath the rollout.
    env = CarryBoxEnv(num_envs=1, clip_names=["data_000"], episode_length=10**9,
                      mu=args.mu, substeps=args.substeps, device=args.device,
                      box_tris=args.box_tris, margin=args.margin,
                      track_termination=False)

    runs = {}
    refiner, nin = load_refiner(args.refiner, args.device)
    print(f"[teacher] refiner {nin}-D -> 29")
    runs["refiner (BCPPO teacher, 890-D)"] = rollout(env, refiner, "teacher", args.frames)

    try:
        tracker, nin = load_tracker(args.tracker, args.device)
        print(f"[tracker] SUGAR tracker.pt {nin}-D -> 29")
        runs["tracker.pt (validated, 510-D)"] = rollout(env, tracker, "policy", args.frames)
    except FileNotFoundError:
        print(f"[tracker] {args.tracker} missing; regenerate with make_policy_assets")

    if args.compare and "tracker.pt (validated, 510-D)" in runs:
        rows = agreement(env, refiner, tracker, min(args.frames, 60))
        print(f"\n{'frame':>6} {'|a_r - a_t| rms':>16} {'|a_t|':>8} {'|a_r|':>8}")
        for k in (0, 1, 2, 5, 10, 20, 40):
            if k < len(rows):
                r = rows[k]
                print(f"{k:>6} {r[0]:>16.3f} {r[1]:>8.2f} {r[2]:>8.2f}")
        print("  Read this as a magnitude comparison, NOT as an error against ground truth:\n"
              "  the distilled student is released from its teacher by stage-3 PPO, so the two\n"
              "  are NOT required to agree. Only |a_r| >> |a_t| at frame 0 is diagnostic.")

    if args.video:
        from sugar_newton.rl.video import VideoRecorder

        # One recorder, so both policies drive the SAME env instance from the same pinned
        # start. Rebuilding it per policy would reintroduce the variable the comparison is
        # trying to remove.
        rec = VideoRecorder(clip="data_000", frames=args.frames, out_dir=args.video,
                            mu=args.mu, substeps=args.substeps, device=args.device,
                            tactile=True, box_tris=args.box_tris, margin=args.margin)
        made = []
        for tag, (net, group) in {
            "teacher_refiner_890": (refiner, "teacher"),
            "tracker_510": (tracker, "policy"),
        }.items():
            path, st = rec.record(TeacherAdapter(net, group), 0)
            if path is None:
                print(f"[video] {tag}: renderer unavailable")
                continue
            dst = str(Path(args.video) / f"{tag}.mp4")
            os.replace(path, dst)
            made.append((tag, dst))
            print(f"[video] {tag}: lift {st.get('video/box_lift', 0):.3f} m of "
                  f"{st.get('video/box_lift_reference', 0):.3f} m, "
                  f"drift frame {int(st.get('video/drift_step', -1))} -> {dst}")

        if len(made) == 2:
            print(f"[video] side by side -> {side_by_side(made, args.video)}")

    print(f"\n{'policy':32} {'drift':>7} {'lift m':>8} {'ref m':>7} {'|a|':>6} {'sat':>6}")
    for name, r in runs.items():
        print(f"{name:32} {r['drift_step']:>7} {r['box_lift']:>8.3f} "
              f"{r['box_lift_reference']:>7.3f} {r['action_mag']:>6.2f} "
              f"{100 * r['action_saturated']:>5.0f}%")
    print(f"\ndrift is the frame the 0.3 m tracking bound is first exceeded (of {args.frames});"
          f"\nthe student currently plateaus near 45.")


if __name__ == "__main__":
    main()
