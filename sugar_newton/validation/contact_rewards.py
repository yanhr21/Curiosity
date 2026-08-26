# SPDX-License-Identifier: BSD-3-Clause
"""H200 runtime audit for SUGAR's four contact-dependent Tracker rewards.

This is deliberately an environment-level gate, not a fabricated tensor unit test.  It
builds the real Newton G1/CarryBox world, advances resolved MuJoCo-Warp contacts, and
checks the official three-frame reductions and reward signs on live forces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl import rewards
from sugar_newton.rl.carrybox_env import CarryBoxEnv


CONTACT_TERMS = ("feet_slide", "feet_air_time", "undesired_contacts", "hoi_contact")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-root", default="SUGAR/data/CarryBox")
    ap.add_argument("--clip", default="data_000")
    ap.add_argument("--start-frame", type=int, default=168)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        print("ERROR: run the contact-reward audit on a Slurm CUDA compute node")
        return 2

    root = Path(args.motion_root)
    env = CarryBoxEnv(
        num_envs=1,
        clip_names=[args.clip],
        motion_root=root,
        teacher_motion_root=root,
        episode_length=max(args.steps + 4, 32),
        device=args.device,
        auto_reset=False,
    )
    if env.contacts.force is None:
        print("FAIL: contacts.force was not allocated")
        return 1

    env.reset(motion_ids=0, start_frames=args.start_frame)
    labels = env.body_labels
    excluded = {labels[i] for i in env.contact_history.undesired_idx.cpu().tolist()}
    forbidden = {
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_rubber_hand", "right_rubber_hand", "box",
    }
    failures: list[str] = []
    if excluded & forbidden:
        failures.append(f"undesired-contact set includes excluded bodies {sorted(excluded & forbidden)}")

    force_seen = False
    box_hand_seen = False
    label_values: set[bool] = set()
    rows = []
    for _ in range(args.steps):
        # A deterministic reference-position command keeps this audit near the source
        # motion while still executing the real Newton solve.  It is a validator input,
        # not a replacement teacher or a training policy.
        action = (env._ref("joint_pos") - env.q_default) / env.a_scale
        _, reward, done, extras = env.step(action)
        terms = extras["reward_terms"]
        missing = set(CONTACT_TERMS) - terms.keys()
        if missing:
            failures.append(f"missing contact terms {sorted(missing)}")
            break

        vals = {name: float(terms[name][0]) for name in CONTACT_TERMS}
        if not all(torch.isfinite(terms[name]).all() for name in CONTACT_TERMS):
            failures.append(f"non-finite contact reward at frame {int(env.t[0])}")
        if vals["feet_slide"] < 0.0:
            failures.append("feet_slide body is negative before applying its negative weight")
        if vals["feet_air_time"] > 0.0:
            failures.append("feet_air_time body is positive; official function must be <= 0")
        if vals["undesired_contacts"] < 0.0:
            failures.append("undesired_contacts body is negative before applying its negative weight")
        if vals["hoi_contact"] not in (0.0, 1.0):
            failures.append(f"hoi_contact is not binary: {vals['hoi_contact']}")

        weighted = {name: rewards.WEIGHTS[name] * vals[name] for name in CONTACT_TERMS}
        if weighted["feet_slide"] > 1.0e-8:
            failures.append("weighted feet_slide has the wrong sign")
        if weighted["feet_air_time"] > 1.0e-8:
            failures.append("weighted feet_air_time has the wrong sign")
        if weighted["undesired_contacts"] > 1.0e-8:
            failures.append("weighted undesired_contacts has the wrong sign")
        if weighted["hoi_contact"] < -1.0e-8:
            failures.append("weighted hoi_contact has the wrong sign")

        net_peak = float(env.contact_history.net.norm(dim=-1).max())
        hand_box = env.contact_history.box[:, :, env.contact_history.hand_idx].norm(dim=-1)
        hand_peak = float(hand_box.max())
        force_seen |= net_peak > 1.0e-4
        box_hand_seen |= hand_peak > 0.1
        label = bool((env._ref("contact") > 0.5)[0])
        label_values.add(label)
        bilateral = bool((hand_box.amax(dim=1)[0] > 0.1).all())
        expected_hoi = float(bilateral == label)
        if vals["hoi_contact"] != expected_hoi:
            failures.append(
                f"frame {int(env.t[0])}: hoi={vals['hoi_contact']} but independent "
                f"bilateral/label check gives {expected_hoi}"
            )
        rows.append((int(env.t[0]), label, net_peak, hand_peak, float(reward[0]), vals, bool(done[0])))
        if bool(done[0]):
            break

    if not force_seen:
        failures.append("no nonzero resolved body contact force was observed")
    if label_values != {False, True}:
        failures.append(f"audit window did not cross both contact labels: {sorted(label_values)}")

    # The short live window need not contain a foot landing after 50 episode steps.
    # Exercise that one branch deterministically on the already-built real env so the
    # historically easy-to-misread +5 weight on a non-positive function is explicit.
    env.contact_history.last_air_time.fill_(0.2)
    env.contact_history.first_foot_contact.fill_(True)
    env.start.copy_(env.t - 51)
    _, controlled_terms = rewards.compute(env)
    controlled_air = float(controlled_terms["feet_air_time"][0])
    expected_air = 2.0 * (0.2 - 0.5)
    if abs(controlled_air - expected_air) > 1.0e-6:
        failures.append(
            f"controlled feet_air_time {controlled_air:.6f} != expected {expected_air:.6f}"
        )
    if rewards.WEIGHTS["feet_air_time"] * controlled_air >= 0.0:
        failures.append("controlled short-air event is not a negative weighted penalty")

    print("frame label net_peak hand_box reward slide air undesired hoi done")
    for frame, label, net_peak, hand_peak, reward_value, vals, done in rows:
        print(
            f"{frame:5d} {int(label):5d} {net_peak:8.3f} {hand_peak:8.3f} "
            f"{reward_value:8.4f} {vals['feet_slide']:7.4f} "
            f"{vals['feet_air_time']:7.4f} {vals['undesired_contacts']:9.1f} "
            f"{vals['hoi_contact']:3.0f} {int(done):4d}"
        )
    print(f"resolved force seen       : {force_seen}")
    print(f"box-filtered hand seen   : {box_hand_seen}")
    print(f"contact labels crossed   : {sorted(label_values)}")
    print(f"undesired body count     : {len(excluded)} / {env.nbody}")
    print(f"controlled air-time term : {controlled_air:.4f} (weighted "
          f"{rewards.WEIGHTS['feet_air_time'] * controlled_air:.4f})")

    if failures:
        print(f"FAILED ({len(failures)})")
        for failure in dict.fromkeys(failures):
            print(f"  - {failure}")
        return 1
    print("PASSED — live contact histories and all four official reward signs are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
