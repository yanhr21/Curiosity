#!/usr/bin/env python3
"""Freeze the complete motion-disjoint physical case grid before model outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import write_json_atomic
from .config import PaperZeroWAMConfig


CONDITIONS = ("matched", "reversed", "same_task_alternate", "wrong_task")
ENDPOINT_CONDITIONS = ("matched_endpoint", "wrong_task_endpoint")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt_record(
    target: dict[str, Any], condition: str
) -> dict[str, Any]:
    if condition in ("matched", "reversed"):
        task = target["task"]
        source_motion_id = int(target["source_motion_id"])
    else:
        spec = target["fixed_counterfactual_prompts"][condition]
        task = spec["task"]
        source_motion_id = int(spec["source_motion_id"])
    return {
        "condition": condition,
        "split": str(target["split"]),
        "task": task,
        "source_motion_id": source_motion_id,
        "latent_key": (
            "reversed_prompt_latents" if condition == "reversed" else "prompt_latents"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PaperZeroWAMConfig()
    config.validate()
    test_rows = sorted(
        (
            row
            for row in read_jsonl(config.resolved(config.manifest))
            if row["split"] == "test"
        ),
        key=lambda row: (row["task"], int(row["source_motion_id"])),
    )
    if len(test_rows) != 19:
        raise ValueError("expected exactly 19 test motions")
    cases = []
    for source_index, target in enumerate(test_rows):
        cases.append(
            {
                "source_index": source_index,
                "split": "test",
                "target_task": target["task"],
                "source_motion_id": int(target["source_motion_id"]),
                "prompts": [prompt_record(target, condition) for condition in CONDITIONS],
                "profiles": [
                    {
                        "profile_id": profile_id,
                        "profile_batch": profile_id // 2,
                        "profile_slot": profile_id % 2,
                        "simulator_seed": (
                            config.noise_seed
                            + 120_000
                            + source_index * 101
                            + profile_id // 2
                        ),
                        "inference_seed_base": (
                            config.noise_seed
                            + 130_000
                            + source_index * 10_003
                            + profile_id * 101
                        ),
                    }
                    for profile_id in range(10)
                ],
            }
        )
    result = {
        "protocol": "paper_zero_wam_motion_disjoint_cases_v1",
        "frozen_before_model_outcomes": True,
        "conditions": list(CONDITIONS),
        "endpoint_conditions": list(ENDPOINT_CONDITIONS),
        "cases": cases,
        "totals": {
            "test_sources": 19,
            "profiles_per_source": 10,
            "conditions_per_profile": 4,
            "endpoint_routes_per_profile": 2,
            "adapted_rollouts": 760,
            "released_endpoint_rollouts": 380,
            "rollouts": 1_140,
            "executed_actions": 1_140 * 650,
            "requested_executed_action_rows": 1_140 * 650,
            "visualizations": 190,
        },
        "clock": {
            "control_hz": 50,
            "rgb_hz": 10,
            "video_latents_per_chunk": config.inference_chunk_size,
            "actions_per_chunk": config.inference_chunk_size * 20,
            "rollout_actions": 650,
        },
        "inference": {
            "chunk_size": config.inference_chunk_size,
            "video_guidance_scale": config.video_guidance_scale,
            "action_guidance_scale": config.action_guidance_scale,
            "flow_integrator": config.flow_integrator,
            "video_steps": config.video_inference_steps,
            "action_steps": config.action_inference_steps,
            "video_snr_shift": config.video_snr_shift,
            "action_snr_shift": config.action_snr_shift,
        },
        "hash_checks": False,
    }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
