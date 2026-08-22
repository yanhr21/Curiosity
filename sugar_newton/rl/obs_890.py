# SPDX-License-Identifier: BSD-3-Clause
"""The 890-D privileged / teacher observation group, and the layout that proves it.

SUGAR's tracker has three observation groups, not one::

    policy   510-D   TrackerCfg      -- what the actor sees (validation/verify_tracker_obs.py)
    critic   890-D   PrivilegedCfg   -- future reference frames + full body state
    teacher  890-D   TeacherCfg      -- same terms, from the *teacher* motion

Both 890-D groups are needed before BCPPO can exist: the frozen refiner is an
890 -> 512 -> 256 -> 128 -> 29 MLP, so there is nowhere to plug a teacher in without this.

``LAYOUT`` below is the term list with its dimensions, and ``OBS_DIM_890`` is their sum.
The assert at import time is the check: if any term's size is wrong the total stops being
890 and the module refuses to load, rather than silently producing a vector the teacher
cannot consume.

Future frames are ``t + 0 .. t + 7`` clamped to the clip length -- offsets start at 0, so
the "future" window *includes* the current frame (``commands.py:get_future_index``).
"""

from __future__ import annotations

import torch

from sugar_newton.rl import rewards as R

FUTURE_FRAMES = 8          # MotionCommandCfg.future_frames
N_DOF = 29
N_BODIES = 14              # MotionCommandCfg.body_names
ROT6 = 6                   # matrix_from_quat(...)[..., :2], two columns

# (term, dimension) in PrivilegedCfg declaration order. Order matters: this is the
# concatenation the pretrained critic and teacher were trained on.
LAYOUT: tuple[tuple[str, int], ...] = (
    ("joint_pos_vel_future", FUTURE_FRAMES * N_DOF * 2),        # 464
    ("motion_anchor_pos_b_future", FUTURE_FRAMES * 3),          # 24
    ("motion_anchor_ori_b_future", FUTURE_FRAMES * ROT6),       # 48
    ("ref_obj_pos_b_future", FUTURE_FRAMES * 3),                # 24
    ("ref_obj_ori_b_future", FUTURE_FRAMES * ROT6),             # 48
    ("ref_obj_lin_vel_b_future", FUTURE_FRAMES * 3),            # 24
    ("ref_obj_ang_vel_b_future", FUTURE_FRAMES * 3),            # 24
    ("body_pos", N_BODIES * 3),                                 # 42
    ("body_ori", N_BODIES * ROT6),                              # 84
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos", N_DOF),
    ("joint_vel", N_DOF),
    ("actions", N_DOF),
    ("obj_pos_b", 3),
    ("obj_ori_b", ROT6),
    ("obj_lin_vel_b", 3),
    ("obj_ang_vel_b", 3),
)
OBS_DIM_890 = sum(d for _, d in LAYOUT)
assert OBS_DIM_890 == 890, f"privileged layout sums to {OBS_DIM_890}, not 890"


def _rot6(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened -- observations.obj_ori_b."""
    return R.mat_from_quat(q)[..., :2].reshape(*q.shape[:-1], ROT6)


def build(env, teacher: bool = False) -> torch.Tensor:
    """Assemble the 890-D vector for every world.

    Args:
        env: a :class:`~sugar_newton.rl.carrybox_env.CarryBoxEnv`.
        teacher: read the teacher motion instead of the student's. SUGAR keeps a separate
            ``teacher_motion`` (``--teacher_motion_folder``) so the teacher can be shown a
            *refined* reference while the student tracks the raw one. Until the env carries
            a second motion set this falls back to the same clips, which makes the teacher
            group equal to the critic group -- correct only when both folders are the same,
            which is how ``play.py`` was run here.
    """
    n = env.num_envs
    dev = env.device
    body_q = env._body_q()
    body_qd = env._body_qd()

    # --- future reference window, t + 0..7, clamped per clip ---
    length = env.ref["length"][env.motion_id]
    offs = torch.arange(FUTURE_FRAMES, device=dev)
    t_fut = (env.t.unsqueeze(-1) + offs.unsqueeze(0)).clamp(min=0)
    t_fut = torch.minimum(t_fut, (length - 1).unsqueeze(-1))
    mid = env.motion_id.unsqueeze(-1).expand(-1, FUTURE_FRAMES)

    jp_f = env.ref["joint_pos"][mid, t_fut]                       # (n, 8, 29)
    jv_f = env.ref["joint_vel"][mid, t_fut]
    joint_pos_vel_future = torch.cat([jp_f, jv_f], dim=-1).reshape(n, -1)

    # anchor frame of the ROBOT, which is what every _b term is expressed in
    a_p, a_q = env._anchor_pose(body_q)
    a_p_e = a_p.unsqueeze(1).expand(n, FUTURE_FRAMES, 3)
    a_q_e = a_q.unsqueeze(1).expand(n, FUTURE_FRAMES, 4)

    ref_anchor_p = env.ref["body_pos_w"][mid, t_fut][:, :, env.ref_body_idx[env.anchor_local]]
    ref_anchor_q = env.ref["body_quat_w"][mid, t_fut][:, :, env.ref_body_idx[env.anchor_local]]
    ref_anchor_q = R.normalize(ref_anchor_q[..., [1, 2, 3, 0]])
    anchor_pos_b_future = R.quat_apply_inv(a_q_e, ref_anchor_p - a_p_e).reshape(n, -1)
    anchor_ori_b_future = _rot6(R.quat_mul(R.quat_conj(a_q_e), ref_anchor_q)).reshape(n, -1)

    ref_o_p = env.ref["obj_pos"][mid, t_fut]
    ref_o_q = R.normalize(env.ref["obj_quat"][mid, t_fut])
    obj_pos_b_future = R.quat_apply_inv(a_q_e, ref_o_p - a_p_e).reshape(n, -1)
    obj_ori_b_future = _rot6(R.quat_mul(R.quat_conj(a_q_e), ref_o_q)).reshape(n, -1)
    obj_lin_b_future = R.quat_apply_inv(a_q_e, env.ref["obj_lin_vel"][mid, t_fut]).reshape(n, -1)
    obj_ang_b_future = R.quat_apply_inv(a_q_e, env.ref["obj_ang_vel"][mid, t_fut]).reshape(n, -1)

    # --- current robot state, bodies in the anchor frame ---
    rob_p = body_q[:, env.body_idx, :3]
    rob_q = R.normalize(body_q[:, env.body_idx, 3:7])
    a_p_b = a_p.unsqueeze(1).expand_as(rob_p)
    a_q_b = a_q.unsqueeze(1).expand_as(rob_q)
    body_pos = R.quat_apply_inv(a_q_b, rob_p - a_p_b).reshape(n, -1)
    body_ori = _rot6(R.quat_mul(R.quat_conj(a_q_b), rob_q)).reshape(n, -1)

    root_q = R.normalize(env.q[:, env.root_q0 + 3:env.root_q0 + 7])
    base_lin_vel = R.quat_apply_inv(root_q, env.qd[:, env.root_qd0:env.root_qd0 + 3])
    base_ang_vel = R.quat_apply_inv(root_q, env.qd[:, env.root_qd0 + 3:env.root_qd0 + 6])

    # --- object, current ---
    o_p = body_q[:, env.box_body, :3]
    o_q = R.normalize(body_q[:, env.box_body, 3:7])
    obj_pos_b = R.quat_apply_inv(a_q, o_p - a_p)
    obj_ori_b = _rot6(R.quat_mul(R.quat_conj(a_q), o_q))
    obj_lin_vel_b = R.quat_apply_inv(a_q, body_qd[:, env.box_body, :3])
    obj_ang_vel_b = R.quat_apply_inv(a_q, body_qd[:, env.box_body, 3:6])

    out = torch.cat([
        joint_pos_vel_future, anchor_pos_b_future, anchor_ori_b_future,
        obj_pos_b_future, obj_ori_b_future, obj_lin_b_future, obj_ang_b_future,
        body_pos, body_ori, base_lin_vel, base_ang_vel,
        env.q[:, env.act_coords] - env.q_default, env.qd[:, env.act_dofs], env.last_action,
        obj_pos_b, obj_ori_b, obj_lin_vel_b, obj_ang_vel_b,
    ], dim=-1)
    if out.shape[-1] != OBS_DIM_890:
        raise RuntimeError(f"privileged obs is {out.shape[-1]}-D, expected {OBS_DIM_890}")
    return out


def describe() -> str:
    lines, off = [], 0
    for name, dim in LAYOUT:
        lines.append(f"  [{off:3d}:{off + dim:3d}) {name:28s} {dim:4d}")
        off += dim
    return "\n".join(lines) + f"\n  total {off}"
