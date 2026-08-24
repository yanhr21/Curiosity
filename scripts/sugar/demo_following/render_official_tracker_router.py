#!/usr/bin/env python3
"""Render reference-versus-actual videos for the official Tracker router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from render_frozen_trace_behavior import (
    ROOT,
    first_episode,
    load_npz,
    load_reference,
    render_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carry-correct-dir", type=Path, required=True)
    parser.add_argument("--carry-unrelated-dir", type=Path, required=True)
    parser.add_argument("--kick-correct-dir", type=Path, required=True)
    parser.add_argument("--kick-unrelated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-env", type=int, default=0)
    parser.add_argument(
        "--joint-generator-route",
        action="store_true",
        help="Render the complete official Generator+Tracker routing experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    directories = {
        "carry_correct": args.carry_correct_dir.expanduser().resolve(),
        "carry_unrelated": args.carry_unrelated_dir.expanduser().resolve(),
        "kick_correct": args.kick_correct_dir.expanduser().resolve(),
        "kick_unrelated": args.kick_unrelated_dir.expanduser().resolve(),
    }
    if any(experiments not in path.parents for path in (*directories.values(), output)):
        raise ValueError("evaluation and video paths must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    if args.source_env != 0:
        raise ValueError("the fixed video profile is environment 0")
    specifications = (
        (
            "carry_correct",
            ROOT / "SUGAR/data/CarryBox/data_045",
            "INPUT DEMO: CARRYBOX MOTION 45",
            "01_carry_domain_carry45_condition.mp4",
            (
                "ACTUAL: FULL CARRY GENERATOR + TRACKER"
                if args.joint_generator_route
                else "ACTUAL: ROUTED CARRY EXPERT (VALID)"
            ),
        ),
        (
            "carry_unrelated",
            ROOT / "SUGAR/data/KickBox/data_021",
            "INPUT DEMO: KICKBOX MOTION 21",
            "02_carry_domain_kick21_condition.mp4",
            (
                "ACTUAL: FULL KICK GENERATOR + TRACKER"
                if args.joint_generator_route
                else "ACTUAL: ROUTED KICK EXPERT (ACTION LIMIT FAIL)"
            ),
        ),
        (
            "kick_correct",
            ROOT / "SUGAR/data/CarryBox/data_045",
            "INPUT DEMO: CARRYBOX MOTION 45",
            "03_kick_domain_carry45_condition.mp4",
            (
                "ACTUAL: FULL CARRY PAIR (BIGBOX TRANSFER REJECTED)"
                if args.joint_generator_route
                else "ACTUAL: ROUTED CARRY EXPERT (GENERATOR STILL KICKS)"
            ),
        ),
        (
            "kick_unrelated",
            ROOT / "SUGAR/data/KickBox/data_021",
            "INPUT DEMO: KICKBOX MOTION 21",
            "04_kick_domain_kick21_condition.mp4",
            (
                "ACTUAL: FULL KICK GENERATOR + TRACKER"
                if args.joint_generator_route
                else "ACTUAL: ROUTED KICK EXPERT (VALID)"
            ),
        ),
    )
    loaded: dict[str, tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    for name, _, _, _, _ in specifications:
        directory = directories[name]
        result = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        if (
            result.get("protocol") != "sugar_shared_absolute_tracker_frozen_physics_v1"
            or result.get("student_action_fraction") != 1.0
            or result.get("dagger_collection") is not False
            or bool(result.get("routed_generator_with_expert", False))
            != args.joint_generator_route
        ):
            raise RuntimeError(f"unadmitted final evaluation: {name}")
        loaded[name] = (result, load_npz(directory / "TRACE.npz"))
    checkpoints = {str(result["shared_checkpoint"]) for result, _ in loaded.values()}
    if len(checkpoints) != 1:
        raise RuntimeError("the four videos do not share one checkpoint")
    matched_initial_state: dict[str, bool] = {}
    for domain, first, second in (
        ("carry", "carry_correct", "carry_unrelated"),
        ("kick", "kick_correct", "kick_unrelated"),
    ):
        left = loaded[first][1]
        right = loaded[second][1]
        paired_keys = (
            "initial_robot_root_state_w",
            "initial_robot_joint_pos",
            "initial_robot_joint_vel",
            "initial_object_root_state_w",
            "prefix_action",
            "post_prefix_robot_root_state_w",
            "post_prefix_robot_joint_pos",
            "post_prefix_robot_joint_vel",
            "post_prefix_object_root_state_w",
        )
        if "initial_object_mass_kg" in left and "initial_object_mass_kg" in right:
            paired_keys = (*paired_keys, "initial_object_mass_kg")
        matched_initial_state[domain] = all(
            np.array_equal(left[key], right[key])
            for key in paired_keys
        )
    if not all(matched_initial_state.values()):
        raise RuntimeError("within-domain condition swap does not share exact initial state")

    output.mkdir(parents=True, exist_ok=False)
    videos: list[dict[str, object]] = []
    for name, reference_path, demo_label, filename, actual_label in specifications:
        result, trace = loaded[name]
        record = render_pair(
            load_reference(reference_path),
            first_episode(trace, args.source_env),
            trace["robot_body_names"],
            demo_label,
            output / filename,
            actual_label,
        )
        record.update(
            {
                "evaluation": name,
                "domain": result["domain"],
                "selected_demo_option": result["selected_demo_option"],
                "result_passed": result["passed"],
            }
        )
        videos.append(record)
    checks = {
        "one_shared_checkpoint": len(checkpoints) == 1,
        "carry_initial_state_exact_match": matched_initial_state["carry"],
        "kick_initial_state_exact_match": matched_initial_state["kick"],
        "matched_carry_behavior_passed": bool(loaded["carry_correct"][0]["passed"]),
        "matched_kick_behavior_passed": bool(loaded["kick_unrelated"][0]["passed"]),
        "all_four_videos_written": len(videos) == 4,
        "all_videos_h264_yuv420p": all(
            bool(record["decode"]["passed"]) for record in videos
        ),
        "all_reference_and_actual_sequences_fully_displayed": all(
            bool(record["reference_fully_displayed"])
            and bool(record["actual_fully_displayed"])
            for record in videos
        ),
    }
    if args.joint_generator_route:
        checks.update(
            {
                "matched_carry_and_kick_routes_pass": bool(
                    loaded["carry_correct"][0]["passed"]
                    and loaded["kick_unrelated"][0]["passed"]
                ),
                "carry_domain_full_kick_route_passes": bool(
                    loaded["carry_unrelated"][0]["passed"]
                    and loaded["carry_unrelated"][0]["selected_skill_success_count"]
                    >= 10
                ),
                "kick_domain_full_carry_route_rejection_is_recorded": bool(
                    not loaded["kick_correct"][0]["passed"]
                    and not loaded["kick_correct"][0]["checks"][
                        "selected_skill_behavioral_gate"
                    ]
                ),
                "all_four_route_complete_generator_tracker_pairs": all(
                    bool(result["routed_generator_with_expert"])
                    for result, _ in loaded.values()
                ),
                "all_four_traces_are_finite_and_reset_free": all(
                    bool(result["checks"]["all_trace_values_finite"])
                    and bool(result["checks"]["no_environment_reset"])
                    for result, _ in loaded.values()
                ),
            }
        )
    else:
        checks.update(
            {
                "carry_kick_route_action_explosion_rejected": bool(
                    loaded["carry_unrelated"][0]["checks"][
                        "raw_student_actions_within_released_tracker_envelope"
                    ]
                    is False
                ),
                "kick_carry_route_remains_inside_action_envelope": bool(
                    loaded["kick_correct"][0]["checks"][
                        "raw_student_actions_within_released_tracker_envelope"
                    ]
                ),
            }
        )
    proof = {
        "protocol": (
            "sugar_official_generator_tracker_router_exact_trace_video_v1"
            if args.joint_generator_route
            else "sugar_official_tracker_router_exact_trace_video_v1"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "shared_checkpoint": next(iter(checkpoints)),
        "source_env": args.source_env,
        "rendering_semantics": (
            "Exact frozen PhysX robot body centers and object pose; no physics replay. "
            "The wireframe box dimensions are illustrative. Carry/Kick reference is "
            "shown at left and the exact evaluated trace at right."
        ),
        "videos": videos,
    }
    (output / "RENDER_PROOF.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(proof, indent=2, sort_keys=True))
    if not proof["passed"]:
        raise RuntimeError("shared absolute Tracker video proof failed")


if __name__ == "__main__":
    main()
