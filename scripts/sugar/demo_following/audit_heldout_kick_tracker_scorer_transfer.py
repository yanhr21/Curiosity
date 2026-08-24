#!/usr/bin/env python3
"""Test frozen Carry45/Kick21 scoring on held-out official Kick inference.

The nine source motions ending in 9 are the motion-disjoint predictor test
split. Their exact 121-D online observations were generated in IsaacLab/PhysX
by the released official SUGAR KickBox Generator/Tracker pair. The primary
result uses the deployed fixed 650-step clock. A source-duration-normalized
phase is reported only as an evaluation diagnostic and never enters a deployed
policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SUGAR/source/sugar_rl"))

from sugar_rl.utils.demo_event_reward_runtime import (  # noqa: E402
    FrozenDemoEventReward,
    FrozenDemoEventRewardCfg,
)


TEST_MOTION_IDS = (9, 19, 29, 39, 49, 59, 69, 79, 89)
DEMOS = ("correct", "unrelated")
PHASE_VARIANTS = ("deployed_fixed_650", "source_duration_diagnostic")
HISTORY_STEPS = 10
DEPLOYED_HORIZON_STEPS = 650
TRACE_STEPS = 700
POLICY_DIM = 121
CONTACT_THRESHOLD_N = 0.1
CONTROL_DT_S = 0.02
OFFICIAL_TRACKER = (ROOT / "SUGAR/demo_ckpts/KickBox/tracker.pt").resolve()
OFFICIAL_GENERATOR = (ROOT / "SUGAR/demo_ckpts/KickBox/generator.ckpt").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT
        / "experiments/demo_following/contact_event_reward_redesign_v1/"
        "deployable_goal_core_corpus_v1",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT
        / "experiments/demo_following/contact_event_reward_redesign_v1/"
        "phase_aware_dense_feedback_scale_audit_v1/RUNTIME_CONFIG.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_runtime_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "sugar_dense_demo_event_feedback_runtime_v1":
        raise RuntimeError("unexpected phase-event runtime protocol")
    if payload.get("future_actual_events_enter_runtime") is not False:
        raise RuntimeError("future event labels may not enter deployed scoring")
    return payload


def scorer_cfg(payload: dict[str, Any], selected_option: str) -> FrozenDemoEventRewardCfg:
    selected = payload["selected_demo_options"][selected_option]
    return FrozenDemoEventRewardCfg(
        dataset_root=str(payload["dataset_root"]),
        predictor_dir=str(payload["predictor_dir"]),
        selected_task=str(selected["selected_task"]),
        selected_motion_id=int(selected["selected_motion_id"]),
        compatibility_baseline=float(payload["compatibility_baseline"]),
        eta=float(payload["eta"]),
        uncertainty_beta=float(payload["uncertainty_beta"]),
        reward_clip=float(payload["reward_clip"]),
        per_target_risk_clip=float(payload["per_target_risk_clip"]),
        target_weights=tuple(float(value) for value in payload["target_weights"]),
    )


def validate_shard_result(trace_path: Path) -> dict[str, Any]:
    result_path = trace_path.with_name("RESULT.json")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checks = payload.get("checks", {})
    if (
        payload.get("protocol")
        != "sugar_official_tracker_actual_contact_event_canary_v1"
        or payload.get("passed") is not True
        or payload.get("task_family") != "KickBox"
        or checks.get("physical_force_vectors_recorded") is not True
        or checks.get("contact_is_exact_threshold_of_force") is not True
        or Path(payload.get("tracker_checkpoint", "")).resolve()
        != OFFICIAL_TRACKER
        or Path(payload.get("generator_checkpoint", "")).resolve()
        != OFFICIAL_GENERATOR
    ):
        raise RuntimeError(f"unverified official Kick shard: {result_path}")
    return payload


def load_heldout_kick(corpus_root: Path) -> dict[str, Any]:
    records: dict[int, dict[str, np.ndarray | int]] = {}
    shard_payloads: dict[str, dict[str, Any]] = {}
    for trace_path in sorted(corpus_root.glob("kickbox_shard*/TRACE.npz")):
        shard_payloads[trace_path.parent.name] = validate_shard_result(trace_path)
        with np.load(trace_path, allow_pickle=False) as archive:
            roles = tuple(str(value) for value in archive["contact_role_names"])
            if roles != ("left_hand", "right_hand", "left_foot", "right_foot"):
                raise RuntimeError(f"Kick contact-role order drift: {roles}")
            if not np.allclose(
                np.asarray(archive["control_dt_s"], dtype=np.float32),
                np.asarray([CONTROL_DT_S], dtype=np.float32),
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError("Kick control clock drift")
            source_ids = np.asarray(
                archive["source_motion_id_by_local_motion"], dtype=np.int64
            )
            for motion_id in TEST_MOTION_IDS:
                hits = np.flatnonzero(source_ids == motion_id)
                if not hits.size:
                    continue
                if hits.size != 1 or motion_id in records:
                    raise RuntimeError(f"held-out Kick motion {motion_id} is duplicated")
                index = int(hits[0])
                records[motion_id] = {
                    "core": np.asarray(
                        archive["goal_policy_core_observation"][:, index],
                        dtype=np.float32,
                    ),
                    "motion_frame": np.asarray(
                        archive["motion_frame"][:, index], dtype=np.int64
                    ),
                    "reference_steps": int(
                        archive["source_reference_steps_by_local_motion"][index]
                    ),
                    "contact": np.asarray(
                        archive["contact"][:, index], dtype=bool
                    ),
                    "contact_force_w": np.asarray(
                        archive["contact_force_w"][:, index], dtype=np.float32
                    ),
                    "reset_before_frame": np.asarray(
                        archive["reset_before_frame"][:, index], dtype=bool
                    ),
                    "object_root_state_w": np.asarray(
                        archive["object_root_state_w"][:, index], dtype=np.float32
                    ),
                    "lift_height_m": np.asarray(
                        archive["lift_height_m"][:, index], dtype=np.float32
                    ),
                }
    if tuple(sorted(records)) != TEST_MOTION_IDS:
        raise RuntimeError(f"held-out Kick coverage drift: {sorted(records)}")

    core = np.stack([records[motion_id]["core"] for motion_id in TEST_MOTION_IDS], axis=1)
    frames = np.stack(
        [records[motion_id]["motion_frame"] for motion_id in TEST_MOTION_IDS],
        axis=1,
    )
    reference_steps = np.asarray(
        [records[motion_id]["reference_steps"] for motion_id in TEST_MOTION_IDS],
        dtype=np.int64,
    )
    contact = np.stack(
        [records[motion_id]["contact"] for motion_id in TEST_MOTION_IDS], axis=1
    )
    contact_force = np.stack(
        [records[motion_id]["contact_force_w"] for motion_id in TEST_MOTION_IDS],
        axis=1,
    )
    reset_before = np.stack(
        [records[motion_id]["reset_before_frame"] for motion_id in TEST_MOTION_IDS],
        axis=1,
    )
    object_state = np.stack(
        [records[motion_id]["object_root_state_w"] for motion_id in TEST_MOTION_IDS],
        axis=1,
    )
    lift = np.stack(
        [records[motion_id]["lift_height_m"] for motion_id in TEST_MOTION_IDS],
        axis=1,
    )
    if core.shape != (TRACE_STEPS, len(TEST_MOTION_IDS), POLICY_DIM):
        raise RuntimeError(f"held-out Kick 121-D geometry drift: {core.shape}")
    if contact.shape != (TRACE_STEPS, len(TEST_MOTION_IDS), 4):
        raise RuntimeError(f"held-out Kick contact geometry drift: {contact.shape}")
    if contact_force.shape != (TRACE_STEPS, len(TEST_MOTION_IDS), 4, 3):
        raise RuntimeError(
            f"held-out Kick physical-force geometry drift: {contact_force.shape}"
        )
    if not np.array_equal(
        contact,
        np.linalg.norm(contact_force, axis=-1) > CONTACT_THRESHOLD_N,
    ):
        raise RuntimeError("held-out Kick contact is not the exact physical-force threshold")
    if np.any(reset_before):
        raise RuntimeError("held-out Kick scorer gate may not cross an environment reset")
    if not np.isfinite(core).all() or not np.isfinite(object_state).all():
        raise RuntimeError("held-out Kick corpus contains non-finite values")
    return {
        "core": core,
        "motion_frame": frames,
        "reference_steps": reference_steps,
        "contact": contact,
        "contact_force_w": contact_force,
        "reset_before_frame": reset_before,
        "object_root_state_w": object_state,
        "lift_height_m": lift,
        "provenance": {
            "collector_protocol": "sugar_official_tracker_actual_contact_event_canary_v1",
            "shards": sorted(shard_payloads),
            "official_tracker_checkpoint": str(OFFICIAL_TRACKER),
            "official_generator_checkpoint": str(OFFICIAL_GENERATOR),
            "contact_threshold_n": CONTACT_THRESHOLD_N,
            "control_dt_s": CONTROL_DT_S,
            "physical_force_threshold_exact": True,
            "reset_count": 0,
        },
    }


@torch.no_grad()
def score_demo(
    data: dict[str, np.ndarray],
    payload: dict[str, Any],
    selected_option: str,
    device: torch.device,
) -> dict[str, np.ndarray | dict[str, Any]]:
    profile_count = len(TEST_MOTION_IDS)
    variant_count = len(PHASE_VARIANTS)
    scorer = FrozenDemoEventReward(
        num_envs=profile_count * variant_count,
        device=device,
        cfg=scorer_cfg(payload, selected_option),
    )
    core = torch.from_numpy(data["core"]).to(device)
    initial = core[0].repeat(variant_count, 1)
    scorer.begin(initial)
    reference_denominator = torch.from_numpy(
        data["reference_steps"] - HISTORY_STEPS
    ).to(device=device, dtype=torch.float32)
    if torch.any(reference_denominator <= HISTORY_STEPS):
        raise RuntimeError("held-out Kick reference duration is too short")

    risk = []
    ready = []
    phase_records = []
    for step in range(1, TRACE_STEPS):
        frame = torch.from_numpy(data["motion_frame"][step]).to(
            device=device, dtype=torch.float32
        )
        deployed_phase = torch.clamp(
            frame / float(DEPLOYED_HORIZON_STEPS), 0.0, 1.0
        )
        diagnostic_phase = torch.clamp(
            frame / reference_denominator, 0.0, 1.0
        )
        phase = torch.cat((deployed_phase, diagnostic_phase), dim=0)
        signal = scorer.process_step(
            core[step].repeat(variant_count, 1),
            phase,
            torch.zeros(profile_count * variant_count, dtype=torch.bool, device=device),
            torch.zeros(profile_count * variant_count, dtype=torch.bool, device=device),
        )
        risk.append(signal.next_risk.cpu().numpy())
        ready.append(signal.next_ready.cpu().numpy())
        phase_records.append(signal.selected_demo_phase.cpu().numpy())
    return {
        "risk": np.stack(risk).reshape(
            TRACE_STEPS - 1, variant_count, profile_count
        ).transpose(1, 0, 2),
        "ready": np.stack(ready).reshape(
            TRACE_STEPS - 1, variant_count, profile_count
        ).transpose(1, 0, 2),
        "phase": np.stack(phase_records).reshape(
            TRACE_STEPS - 1, variant_count, profile_count
        ).transpose(1, 0, 2),
        "audit": scorer.audit(),
    }


def summarize_semantics(
    data: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray | dict[str, Any]]],
) -> dict[str, Any]:
    carry = np.asarray(scores["correct"]["risk"])
    kick = np.asarray(scores["unrelated"]["risk"])
    carry_ready = np.asarray(scores["correct"]["ready"]).astype(bool)
    kick_ready = np.asarray(scores["unrelated"]["ready"]).astype(bool)
    if not np.array_equal(carry_ready, kick_ready):
        raise RuntimeError("selected demo changed causal prefix readiness")
    margin = kick - carry
    frames = data["motion_frame"][1:]
    denominator = data["reference_steps"] - HISTORY_STEPS
    valid_reference = frames <= denominator[None, :]
    output = {}
    for variant_index, variant in enumerate(PHASE_VARIANTS):
        valid = carry_ready[variant_index] & valid_reference
        profile_means = []
        profile_preference = []
        for profile in range(len(TEST_MOTION_IDS)):
            selected = valid[:, profile]
            if not selected.any():
                raise RuntimeError("held-out Kick profile has no ready valid frames")
            mean_margin = float(np.mean(margin[variant_index, :, profile][selected]))
            profile_means.append(mean_margin)
            profile_preference.append(mean_margin < 0.0)
        values = margin[variant_index][valid]
        output[variant] = {
            "mean_kick_minus_carry_risk": float(np.mean(values)),
            "kick_preferred_frame_fraction": float(np.mean(values < 0.0)),
            "kick_preferred_profile_count": int(np.count_nonzero(profile_preference)),
            "profile_count": len(TEST_MOTION_IDS),
            "profile_mean_margins": {
                str(motion_id): profile_means[index]
                for index, motion_id in enumerate(TEST_MOTION_IDS)
            },
        }
    return output


def behavior_summary(data: dict[str, np.ndarray]) -> dict[str, Any]:
    contact = data["contact"]
    object_xy = data["object_root_state_w"][..., :2]
    displacement = np.linalg.norm(object_xy - object_xy[0:1], axis=-1)
    foot_contact = np.any(contact[..., 2:4], axis=-1)
    hand_contact = np.any(contact[..., 0:2], axis=-1)
    return {
        "motion_ids": list(TEST_MOTION_IDS),
        "reference_steps": data["reference_steps"].tolist(),
        "maximum_planar_object_displacement_m": np.max(displacement, axis=0).tolist(),
        "foot_contact_fraction": np.mean(foot_contact, axis=0).tolist(),
        "hand_contact_fraction": np.mean(hand_contact, axis=0).tolist(),
        "maximum_lift_height_m": np.max(data["lift_height_m"], axis=0).tolist(),
        "profiles_with_foot_contact": int(np.count_nonzero(np.any(foot_contact, axis=0))),
        "profiles_moving_object_at_least_1cm": int(
            np.count_nonzero(np.max(displacement, axis=0) >= 0.01)
        ),
    }


def main() -> None:
    args = parse_args()
    corpus_root = args.corpus_root.expanduser().resolve()
    runtime_config = args.runtime_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")

    payload = load_runtime_payload(runtime_config)
    data = load_heldout_kick(corpus_root)
    scores = {
        demo: score_demo(data, payload, demo, device) for demo in DEMOS
    }
    semantics = summarize_semantics(data, scores)
    behavior = behavior_summary(data)
    deployed = semantics["deployed_fixed_650"]
    checks = {
        "exact_motion_disjoint_test_split_used": tuple(TEST_MOTION_IDS)
        == (9, 19, 29, 39, 49, 59, 69, 79, 89),
        "all_predictors_frozen": all(
            score["audit"]["model_training"] is False
            and score["audit"]["trainable_parameters"] == 0
            for score in scores.values()
        ),
        "released_official_generator_tracker_pair_used": (
            data["provenance"]["official_tracker_checkpoint"]
            == str(OFFICIAL_TRACKER)
            and data["provenance"]["official_generator_checkpoint"]
            == str(OFFICIAL_GENERATOR)
        ),
        "physical_contact_force_is_exact_and_reset_free": (
            data["provenance"]["physical_force_threshold_exact"] is True
            and data["provenance"]["reset_count"] == 0
        ),
        "heldout_rollouts_show_kick_interaction": (
            behavior["profiles_with_foot_contact"] >= 5
            and behavior["profiles_moving_object_at_least_1cm"] >= 5
        ),
        "deployed_clock_prefers_kick_on_mean": (
            deployed["mean_kick_minus_carry_risk"] < 0.0
        ),
        "deployed_clock_prefers_kick_on_frame_majority": (
            deployed["kick_preferred_frame_fraction"] > 0.5
        ),
        "deployed_clock_prefers_kick_on_profile_majority": (
            deployed["kick_preferred_profile_count"] >= 5
        ),
        "no_policy_training_or_environment_execution": True,
    }
    result = {
        "protocol": "sugar_phase_event_heldout_kick_tracker_transfer_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "claim_scope": (
            "Motion-disjoint official Generator/Tracker Kick-domain scorer transfer "
            "under the deployed fixed clock. This is an independent policy/data gate, "
            "but it does not establish Refiner-plus-residual Kick transfer or policy "
            "following."
        ),
        "device": str(device),
        "corpus_root": str(corpus_root),
        "runtime_config": str(runtime_config),
        "selected_demos": {
            demo: {
                "task": score["audit"]["selected_task"],
                "motion_id": score["audit"]["selected_motion_id"],
                "demo_row": score["audit"]["selected_demo_row"],
            }
            for demo, score in scores.items()
        },
        "behavior": behavior,
        "rollout_provenance": data["provenance"],
        "semantics": semantics,
        "source_duration_phase_is_evaluation_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays = {
            f"{demo}_{name}": np.asarray(scores[demo][name])
            for demo in DEMOS
            for name in ("risk", "ready", "phase")
        }
        np.savez_compressed(staging / "SCORES.npz", **arrays)
        (staging / "RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed"]:
        raise RuntimeError(
            "held-out Kick scorer transfer failed: "
            f"{[name for name, value in checks.items() if not value]}"
        )


if __name__ == "__main__":
    main()
