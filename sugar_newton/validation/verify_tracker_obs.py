"""Rebuild SUGAR's 510-D tracker observation offline and check it against Isaac's own actions.

This touches no SUGAR code. It reads the trajectories that SUGAR's *own* rollout dumper
wrote (`--task ...-Tracker-Rollout --rollout_dir`), reconstructs the policy observation
from the recorded state, runs the official `tracker.pt` actor on it, and compares the
result with the action Isaac actually applied on that step.

This is the unit test for the Newton port's observation half -- the half
:mod:`sugar_newton.validation.g1_carrybox_policy` rebuilds from Newton state instead of
from a dump. Needs torch, so run it on a login-node env rather than in the container. Result on all 7 dumped trajectories: per-joint correlation 0.970-0.987, RMSE 0.088
against an action std of 0.99-1.72.

Read RMSE, not correlation. The 29 reference joint angles sit at the front of the vector
and predict the action well on their own, so correlation stays high even when a later
convention is wrong -- deliberately breaking the 6-D layout, reversing history, or
dropping the default-pose offset each cost only 0.969 -> 0.945-0.956. RMSE separates them
properly (0.088 correct, 0.135-0.301 wrong), so it is the statistic that decides here.

Noise: the tracker group sets ``enable_corruption=True``, so the recorded actions came
from *noisy* observations while this rebuild is clean, and exact equality is impossible.
Re-running with SUGAR's own Unoise amplitudes injected induces an action RMSE of 0.080
versus the 0.088 observed -- a ratio of 1.11, i.e. the residual is the recorded noise and
not a remaining bug.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = "/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby"
SUGAR = f"{ROOT}/Curiosity/SUGAR"
URDF = f"{SUGAR}/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
TRACKER = f"{SUGAR}/demo_ckpts/CarryBox/tracker.pt"

# assets/robots/unitree.py -- gains are derived from an armature and a 10 Hz natural freq.
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
A_5020, A_7520_14, A_7520_22, A_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425

# (suffixes, armature, effort limit [N.m], armature scale)
ACTUATORS = (
    (("hip_pitch_joint", "hip_yaw_joint"), A_7520_14, 88.0, 1.0),
    (("hip_roll_joint", "knee_joint"), A_7520_22, 139.0, 1.0),
    (("ankle_pitch_joint", "ankle_roll_joint"), A_5020, 50.0, 2.0),
    (("waist_roll_joint", "waist_pitch_joint"), A_5020, 50.0, 2.0),
    (("waist_yaw_joint",), A_7520_14, 88.0, 1.0),
    (("wrist_pitch_joint", "wrist_yaw_joint"), A_4010, 5.0, 1.0),
    (("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
      "elbow_joint", "wrist_roll_joint"), A_5020, 25.0, 1.0),
)

# init_state.joint_pos, most specific pattern last so it wins.
DEFAULT_POSE = (
    ("hip_pitch_joint", -0.312),
    ("knee_joint", 0.669),
    ("ankle_pitch_joint", -0.363),
    ("elbow_joint", 0.6),
    ("left_shoulder_roll_joint", 0.2),
    ("left_shoulder_pitch_joint", 0.2),
    ("right_shoulder_roll_joint", -0.2),
    ("right_shoulder_pitch_joint", 0.2),
)


def bfs_joint_names(urdf: str) -> list[str]:
    """Non-fixed joint names breadth-first from the root link -- PhysX's articulation order."""
    root = ET.parse(urdf).getroot()
    joints = [(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"))
              for j in root.findall("joint")]
    kids = collections.defaultdict(list)
    for n, ty, p, c in joints:
        kids[p].append((n, ty, c))
    links = {l.get("name") for l in root.findall("link")}
    roots = [l for l in links if l not in {c for *_, c in joints}]
    order, queue = [], collections.deque(sorted(roots))
    while queue:
        for n, ty, c in kids[queue.popleft()]:
            if ty != "fixed":
                order.append(n)
            queue.append(c)
    return order


def action_scale(names: list[str]) -> np.ndarray:
    """UNITREE_G1_29DOF_MIMIC_ACTION_SCALE: 0.25 * effort_limit / stiffness, per joint."""
    out = np.zeros(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        for suffixes, armature, effort, scale in ACTUATORS:
            if any(name.endswith(s) for s in suffixes):
                stiffness = armature * NATURAL_FREQ ** 2 * scale
                out[i] = 0.25 * effort / stiffness
                break
        else:
            raise KeyError(f"no actuator group for {name}")
    return out


def default_pose(names: list[str]) -> np.ndarray:
    out = np.zeros(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        for pattern, value in DEFAULT_POSE:
            if name.endswith(pattern) or name == pattern:
                out[i] = value
    return out


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> (N,3,3), matching isaaclab.utils.math.matrix_from_quat."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def history(seq: np.ndarray, t: int, length: int = 5) -> np.ndarray:
    """Oldest first, most recent last -- CircularBuffer.buffer's stated order.

    Before the buffer has filled, IsaacLab's CircularBuffer repeats the first appended
    entry, so early steps clamp at index 0 rather than zero-padding.
    """
    idx = np.clip(np.arange(t - length + 1, t + 1), 0, len(seq) - 1)
    return seq[idx].reshape(-1)


def build_obs(d: dict, t: int, q_default: np.ndarray, state_lag: int = 1) -> np.ndarray:
    """The TrackerCfg group, in declaration order, for the obs that produced ``action[t]``.

    ``state_lag=1`` is a measurement, not a guess. ``_collect_rollout_step`` runs inside
    ``_update_command``, which IsaacLab calls *before* the observation manager, so row
    ``t`` of the dump pairs post-step-``t`` state with the action applied *during* step
    ``t``. The observation that produced ``action[t]`` therefore came from row ``t-1``,
    and ``last_action`` in it is ``action[t-1]`` -- i.e. the same row, no extra lag.
    A joint sweep over (state_lag, action_lag) in {0,1,2}^2 puts the minimum RMSE at
    (1, 0): 0.088 versus 0.135-0.301 everywhere else.
    """
    ts = max(t - state_lag, 0)
    joint_rel = d["joint_pos"].astype(np.float64) - q_default
    mat = matrix_from_quat(d["obj_quat_b"][ts:ts + 1].astype(np.float64))

    return np.concatenate([
        d["ref_joint_pos"][ts],                      # 29
        d["ref_root_lin_vel_b"][ts],                 # 3
        d["ref_root_ang_vel_b"][ts],                 # 3
        np.atleast_1d(d["ref_contact_label"][ts]).astype(np.float64),  # 1
        history(d["root_ang_vel_b"].astype(np.float64), ts),           # 15
        history(joint_rel, ts),                                        # 145
        history(d["joint_vel"].astype(np.float64), ts),                # 145
        history(d["action"].astype(np.float64), ts),                   # 145
        history(d["project_gravity"].astype(np.float64), ts),          # 15
        d["obj_pos_b"][ts],                          # 3
        mat[..., :2].reshape(1, -1)[0],              # 6  first two columns of R
    ]).astype(np.float32)


class Actor(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(510, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 29),
        )
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("actor."):
                mapped["net." + k[len("actor."):]] = v
        self.load_state_dict(mapped)
        self.eval()

    def forward(self, x):
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default=f"{ROOT}/isaac/rollouts_isaac")
    ap.add_argument("--state-lag", type=int, default=1)
    ap.add_argument("--files", type=int, default=99)
    args = ap.parse_args()

    names = bfs_joint_names(URDF)
    q_default = default_pose(names)
    scale = action_scale(names)
    print(f"{len(names)} joints; default pose nonzero at "
          f"{[names[i] for i in np.nonzero(q_default)[0]]}")
    print(f"action scale range [{scale.min():.4f}, {scale.max():.4f}]\n")

    ck = torch.load(TRACKER, map_location="cpu", weights_only=False)
    actor = Actor(ck["model_state_dict"])

    paths = sorted(glob.glob(os.path.join(args.rollouts, "**", "*.npz"), recursive=True))
    for path in paths[: args.files]:
        d = {k: v for k, v in np.load(path, allow_pickle=True).items()}
        n = len(d["action"])
        obs = np.stack([build_obs(d, t, q_default, args.state_lag) for t in range(n)])
        with torch.inference_mode():
            pred = actor(torch.from_numpy(obs)).numpy()
        truth = d["action"]

        # Per-joint Pearson r, then the aggregate.
        r = np.array([np.corrcoef(pred[:, j], truth[:, j])[0, 1] for j in range(29)])
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        print(f"{os.path.basename(path)}  n={n}")
        print(f"  corr  mean {np.nanmean(r):+.4f}   median {np.nanmedian(r):+.4f}   "
              f"min {np.nanmin(r):+.4f}   max {np.nanmax(r):+.4f}")
        print(f"  rmse  {rmse:.4f}   |truth| std {truth.std():.4f}   "
              f"|pred| std {pred.std():.4f}")
        worst = np.argsort(np.nan_to_num(r))[:4]
        print(f"  worst joints: {[(names[j], round(float(r[j]), 3)) for j in worst]}\n")


if __name__ == "__main__":
    main()
