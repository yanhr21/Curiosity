#!/usr/bin/env python3
"""Independently audit the matched SUGAR CarryBox success/failure pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pair-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode_video(path: Path) -> dict[str, int | bool | str]:
    capture = cv2.VideoCapture(str(path))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    return {
        "path": str(path.resolve()),
        "declared_frames": declared,
        "decoded_frames": decoded,
        "fully_decodes": declared > 0 and decoded == declared,
        "width": width,
        "height": height,
        "fourcc": fourcc,
    }


def main() -> None:
    pair_root = args.pair_root.resolve()
    success_root = pair_root / "successful_grasp"
    failure_root = pair_root / "failed_grasp"
    success_summary = load_json(success_root / "summary.json")
    failure_summary = load_json(failure_root / "summary.json")
    success_audit = load_json(success_root / "native_audit.json")
    failure_audit = load_json(failure_root / "native_audit.json")
    success_render = load_json(
        success_root / "successful_carrybox_whole_hand_tactile.render.json"
    )
    failure_render = load_json(
        failure_root / "failed_carrybox_whole_hand_tactile.render.json"
    )
    success_video_path = success_root / "successful_carrybox_whole_hand_tactile.mp4"
    failure_video_path = failure_root / "failed_carrybox_whole_hand_tactile.mp4"
    success_video = decode_video(success_video_path)
    failure_video = decode_video(failure_video_path)

    release_step = int(failure_summary["release_step"])
    bitwise_prefix_keys = (
        "normal_force",
        "signed_shear",
        "penetration",
        "taxel_position_w",
        "taxel_quaternion_w",
        "tactile_contact_normal_w",
        "tactile_relative_tangential_velocity_w",
        "optical_rgb",
        "optical_depth",
        "active_taxels",
        "bilateral_contact",
        "object_state_w",
        "object_velocity_w",
        "patch_box_force_w",
        "robot_box_force_w",
        "robot_joint_position",
        "applied_action",
        "motion_frame_before_action",
    )
    with np.load(
        success_root / "whole_hand_trace.npz", allow_pickle=False
    ) as success, np.load(
        failure_root / "whole_hand_trace.npz", allow_pickle=False
    ) as failure:
        prefix_equal = {
            key: bool(np.array_equal(success[key][:release_step], failure[key][:release_step]))
            for key in bitwise_prefix_keys
        }
        friction_prefix_difference = {}
        for key in (
            "patch_box_friction_force_w",
            "robot_box_friction_force_w",
        ):
            success_value = success[key][:release_step].astype(np.float64)
            failure_value = failure[key][:release_step].astype(np.float64)
            difference = np.abs(success_value - failure_value)
            friction_prefix_difference[key] = {
                "bitwise_equal": bool(np.array_equal(success_value, failure_value)),
                "different_components": int(np.count_nonzero(difference)),
                "total_components": int(difference.size),
                "maximum_absolute_difference_n": float(difference.max()),
                "maximum_absolute_value_n": float(
                    max(np.abs(success_value).max(), np.abs(failure_value).max())
                ),
            }
        failure_actions = failure["applied_action"]
        failure_active = failure["active_taxels"].sum(axis=(1, 2))
        failure_bilateral = failure["bilateral_contact"]
        failure_box = failure["object_state_w"]
        failure_velocity = failure["object_velocity_w"]
        failure_release = {
            "release_step": release_step,
            "bilateral_immediately_before_release": bool(
                failure_bilateral[release_step - 1]
            ),
            "bilateral_at_release": bool(failure_bilateral[release_step]),
            "active_taxels_immediately_before_release": int(
                failure_active[release_step - 1]
            ),
            "active_taxels_at_release": int(failure_active[release_step]),
            "active_taxels_one_step_after_release": int(
                failure_active[release_step + 1]
            ),
            "post_release_actions_exact_zero": bool(
                np.count_nonzero(failure_actions[release_step:]) == 0
            ),
            "box_z_at_release_m": float(failure_box[release_step, 2]),
            "box_z_final_m": float(failure_box[-1, 2]),
            "box_drop_before_termination_m": float(
                failure_box[release_step, 2] - failure_box[-1, 2]
            ),
            "final_box_vertical_velocity_m_s": float(failure_velocity[-1, 2]),
        }

    checks = {
        "both_native_audits_pass": bool(
            success_audit["structural_passed"]
            and failure_audit["structural_passed"]
        ),
        "all_pre_release_core_channels_bitwise_equal": all(prefix_equal.values()),
        "pre_release_physx_friction_within_2e-6_n": all(
            item["maximum_absolute_difference_n"] <= 2.0e-6
            for item in friction_prefix_difference.values()
        ),
        "failure_was_bilateral_before_release": failure_release[
            "bilateral_immediately_before_release"
        ],
        "failure_loses_bilateral_contact_at_release": not failure_release[
            "bilateral_at_release"
        ],
        "failure_has_zero_contact_one_step_later": (
            failure_release["active_taxels_one_step_after_release"] == 0
        ),
        "failure_post_release_actions_exact_zero": failure_release[
            "post_release_actions_exact_zero"
        ],
        "failure_box_physically_drops": (
            failure_release["box_drop_before_termination_m"] > 0.20
            and failure_release["final_box_vertical_velocity_m_s"] < -1.0
        ),
        "videos_fully_decode": bool(
            success_video["fully_decodes"] and failure_video["fully_decodes"]
        ),
        "videos_are_h264": bool(
            success_video["fourcc"].lower() in {"avc1", "h264"}
            and failure_video["fourcc"].lower() in {"avc1", "h264"}
        ),
        "videos_are_2560x1440": bool(
            (success_video["width"], success_video["height"]) == (2560, 1440)
            and (failure_video["width"], failure_video["height"]) == (2560, 1440)
        ),
        "render_scales_identical": bool(
            success_render["normal_scale_max_n_per_taxel"]
            == failure_render["normal_scale_max_n_per_taxel"]
            and success_render["shear_scale_max_n_per_taxel"]
            == failure_render["shear_scale_max_n_per_taxel"]
        ),
        "render_layouts_identical": bool(
            success_render["layout"] == failure_render["layout"]
        ),
    }
    payload = {
        "schema": "sugar_whole_hand_carrybox_pair_audit_v1",
        "pair_root": str(pair_root),
        "checks": checks,
        "passed": all(checks.values()),
        "pre_release_bitwise_identity": prefix_equal,
        "pre_release_physx_friction_float32_aggregation_difference": (
            friction_prefix_difference
        ),
        "failure_release": failure_release,
        "success_behavior": {
            "maximum_lift_m": success_audit["maximum_lift_m"],
            "lifted_bilateral_frames": success_audit["lifted_bilateral_frames"],
            "longest_lifted_bilateral_run": success_audit[
                "longest_lifted_bilateral_run"
            ],
        },
        "success_video": success_video,
        "failure_video": failure_video,
        "artifact_sha256": {
            str(path.relative_to(pair_root)): sha256(path)
            for path in (
                success_video_path,
                failure_video_path,
                success_root / "whole_hand_trace.npz",
                failure_root / "whole_hand_trace.npz",
                success_root / "native_audit.json",
                failure_root / "native_audit.json",
                success_root
                / "successful_carrybox_whole_hand_tactile.render.json",
                failure_root / "failed_carrybox_whole_hand_tactile.render.json",
            )
        },
        "claim_boundary": (
            "This proves matched collection, physical success/release behavior, "
            "core-channel bitwise synchronization, bounded float32 PhysX "
            "friction-aggregation variation, and decodable visualization. It "
            "does not override the separately reported TacSL force-balance "
            "failure or establish full-palm load coverage."
        ),
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
