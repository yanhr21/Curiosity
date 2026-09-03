"""Compare Newton's imported body ordering against the reference clip's ordering."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

import newton
from sugar_newton.rl.carrybox_env import URDF, bfs_body_names

b = newton.ModelBuilder()
b.add_urdf(
    str(URDF),
    floating=True,
    collapse_fixed_joints=False,
    enable_self_collisions=True,
    joint_ordering="bfs",
    ignore_inertial_definitions=False,
)
newton_labels = [l.split("/")[-1] for l in b.body_label]
clip_order = bfs_body_names(URDF)

d = np.load(REPO / "SUGAR/data/CarryBox/data_000/robot_50hz.npz", allow_pickle=True)
print("clip body_pos_w shape:", d["body_pos_w"].shape)
print("newton bodies:", len(newton_labels))
print("bfs_body_names:", len(clip_order))
print("equal:", newton_labels == clip_order)
if newton_labels != clip_order:
    for i, (a, c) in enumerate(zip(newton_labels, clip_order)):
        flag = "  " if a == c else "<-"
        print(f"{i:3d} {a:32s} {c:32s} {flag}")
    print("only in newton:", [n for n in newton_labels if n not in clip_order])
    print("only in clip  :", [n for n in clip_order if n not in newton_labels])
