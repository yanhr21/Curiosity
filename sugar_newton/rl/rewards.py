# SPDX-License-Identifier: BSD-3-Clause
"""SUGAR's CarryBox reward, transcribed for the Newton environment.

Weights and stds are taken from ``train_tracker/base_tracker_env_cfg.py``'s
``BaseRewardsCfg`` and the term bodies from ``locomanip/mdp/rewards.py``; none of them are
tuned here. Every tracking term has the same shape, ``exp(-error / std**2)``, so the
reward stays bounded and a term that is hopeless contributes ~0 rather than a large
negative that drowns the rest.

Two terms in SUGAR's config are ``MISSING`` at this level and filled in per-task
(``undesired_contacts`` and ``hoi_contact``). They need per-body contact forces, which the
Newton env does not surface yet, and are omitted -- so this reward is SUGAR's minus those
two. That is a deliberate, recorded gap, not an oversight: see ``WEIGHTS`` below.

Quaternions are xyzw throughout, matching Newton. The reference clips store wxyz and are
reordered at the call site, not here.
"""

from __future__ import annotations

import torch

# BaseRewardsCfg, verbatim. Terms marked None are SUGAR's MISSING entries.
WEIGHTS = {
    # regularisation
    "joint_acc": -2.5e-7,
    "joint_torque": -1.0e-5,
    "action_rate": -1.0e-1,
    "joint_limit": -10.0,
    # tracking
    "motion_joint_pos": 0.125,
    "motion_global_anchor_pos": 0.25,
    "motion_global_anchor_ori": 0.25,
    "motion_body_pos": 0.25,
    "motion_body_ori": 0.25,
    "motion_body_lin_vel": 0.25,
    "motion_body_ang_vel": 0.25,
    "motion_obj_pos": 0.5,
    "motion_obj_ori": 0.5,
    "motion_obj_lin_vel": 0.5,
    "motion_obj_ang_vel": 0.5,
    # interaction
    "obj2body_pos": 0.25,
    "obj2body_ori": 0.25,
}
STD = {
    "motion_joint_pos": 0.6,
    "motion_global_anchor_pos": 0.3,
    "motion_global_anchor_ori": 0.4,
    "motion_body_pos": 0.3,
    "motion_body_ori": 0.4,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 3.14,
    "motion_obj_pos": 0.3,
    "motion_obj_ori": 0.4,
    "motion_obj_lin_vel": 1.0,
    "motion_obj_ang_vel": 3.14,
    "obj2body_pos": 0.3,
    "obj2body_ori": 0.4,
}
# feet_slide (-0.1) and feet_air_time (+5.0) need contact sensors on the ankle rolls;
# undesired_contacts and hoi_contact are MISSING in the base config and per-task. All four
# are omitted until the env surfaces per-body contact forces.
OMITTED = ("feet_slide", "feet_air_time", "undesired_contacts", "hoi_contact")


# --- quaternion helpers, xyzw, batched ------------------------------------------
def normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u, w = q[..., :3], q[..., 3:]
    return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


def quat_apply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conj(q), v)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two rotations, IsaacLab's quat_error_magnitude."""
    d = quat_mul(a, quat_conj(b))
    # clamp both ends: a diverged body can hand us a non-unit quaternion, and asin of
    # anything outside [-1, 1] is NaN, which then spreads through the whole reward
    return 2.0 * torch.asin(d[..., :3].norm(dim=-1).clamp(0.0, 1.0))


def mat_from_quat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def _exp(error: torch.Tensor, key: str) -> torch.Tensor:
    return torch.exp(-error / STD[key] ** 2)


def compute(env) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total reward and the per-term breakdown, both detached."""
    body_q = env._body_q()
    body_qd = env._body_qd()
    terms: dict[str, torch.Tensor] = {}

    # --- reference, in this env's world frame ---
    ref_bp = env._ref("body_pos_w")[:, env.ref_body_idx]
    ref_bq = normalize(env._ref("body_quat_w")[:, env.ref_body_idx][..., [1, 2, 3, 0]])
    ref_blv = env._ref("body_lin_vel_w")[:, env.ref_body_idx]
    ref_bav = env._ref("body_ang_vel_w")[:, env.ref_body_idx]
    rob_bp = body_q[:, env.body_idx, :3]
    rob_bq = normalize(body_q[:, env.body_idx, 3:7])
    rob_blv = body_qd[:, env.body_idx, :3]
    rob_bav = body_qd[:, env.body_idx, 3:6]

    a = env.anchor_local
    terms["motion_global_anchor_pos"] = _exp(
        (ref_bp[:, a] - rob_bp[:, a]).square().sum(-1), "motion_global_anchor_pos")
    terms["motion_global_anchor_ori"] = _exp(
        quat_angle(ref_bq[:, a], rob_bq[:, a]) ** 2, "motion_global_anchor_ori")

    # bodies relative to the anchor -- SUGAR compares body_pos_relative_w, i.e. the
    # reference re-anchored onto the robot's own anchor, so a global offset is not
    # penalised twice (the anchor term above already does that)
    rel_ref = ref_bp - ref_bp[:, a:a + 1] + rob_bp[:, a:a + 1]
    terms["motion_body_pos"] = _exp(
        (rel_ref - rob_bp).square().sum(-1).mean(-1), "motion_body_pos")
    terms["motion_body_ori"] = _exp(
        (quat_angle(ref_bq, rob_bq) ** 2).mean(-1), "motion_body_ori")
    terms["motion_body_lin_vel"] = _exp(
        (ref_blv - rob_blv).square().sum(-1).mean(-1), "motion_body_lin_vel")
    terms["motion_body_ang_vel"] = _exp(
        (ref_bav - rob_bav).square().sum(-1).mean(-1), "motion_body_ang_vel")

    terms["motion_joint_pos"] = _exp(
        (env._ref("joint_pos") - env.q[:, env.act_coords]).square().sum(-1),
        "motion_joint_pos")

    # --- object ---
    obj_p = body_q[:, env.box_body, :3]
    obj_q = normalize(body_q[:, env.box_body, 3:7])
    obj_lv = body_qd[:, env.box_body, :3]
    obj_av = body_qd[:, env.box_body, 3:6]
    ref_op = env._ref("obj_pos")
    ref_oq = normalize(env._ref("obj_quat"))
    terms["motion_obj_pos"] = _exp((ref_op - obj_p).square().sum(-1), "motion_obj_pos")
    terms["motion_obj_ori"] = _exp(quat_angle(ref_oq, obj_q) ** 2, "motion_obj_ori")
    terms["motion_obj_lin_vel"] = _exp(
        (env._ref("obj_lin_vel") - obj_lv).square().sum(-1), "motion_obj_lin_vel")
    terms["motion_obj_ang_vel"] = _exp(
        (env._ref("obj_ang_vel") - obj_av).square().sum(-1), "motion_obj_ang_vel")

    # --- interaction: where the object sits in each body's frame ---
    o2b = quat_apply_inv(rob_bq, obj_p[:, None, :] - rob_bp)
    ref_o2b = quat_apply_inv(ref_bq, ref_op[:, None, :] - ref_bp)
    terms["obj2body_pos"] = _exp(
        (o2b - ref_o2b).square().sum(-1).mean(-1), "obj2body_pos")
    o2b_q = quat_mul(quat_conj(rob_bq), obj_q[:, None, :].expand_as(rob_bq))
    ref_o2b_q = quat_mul(quat_conj(ref_bq), ref_oq[:, None, :].expand_as(ref_bq))
    terms["obj2body_ori"] = _exp(
        (quat_angle(o2b_q, ref_o2b_q) ** 2).mean(-1), "obj2body_ori")

    # --- regularisation ---
    qd = env.qd[:, env.act_dofs]
    terms["joint_acc"] = ((qd - env.prev_qd) / env.dt).square().sum(-1)
    # PD torque actually commanded, clipped at the effort limit the same way the solver
    # does -- SUGAR's joint_torques_l2 penalises applied torque, not position error
    tau = env.k * (env.target[:, env.act_dofs] - env.q[:, env.act_coords]) - env.d * qd
    terms["joint_torque"] = tau.clamp(-env.effort, env.effort).square().sum(-1)
    terms["action_rate"] = (env.last_action - env.prev_action).square().sum(-1)
    lo, hi = env.joint_limit_lo, env.joint_limit_hi
    q = env.q[:, env.act_coords]
    terms["joint_limit"] = ((lo - q).clamp_min(0.0) + (q - hi).clamp_min(0.0)).sum(-1)

    total = sum(WEIGHTS[k] * v for k, v in terms.items())
    return total.detach(), {k: v.detach() for k, v in terms.items()}
