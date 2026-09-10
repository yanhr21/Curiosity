#!/usr/bin/env python3
"""Audit production attention layouts over every formal schedule geometry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .config import PaperZeroWAMConfig
from .model import PaperZeroWAM, ifp_head_execution_plan


class LayoutProbe:
    """Bind the production layout helpers without constructing model weights."""

    _video_positions = PaperZeroWAM._video_positions
    _action_positions = staticmethod(PaperZeroWAM._action_positions)

    def __init__(self, config: PaperZeroWAMConfig):
        self.config = config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_geometry(
    probe: LayoutProbe,
    *,
    latent_start: int,
    chunk_size: int,
    prompt_enabled: bool,
) -> dict[str, Any]:
    prompt_frames = 16
    history_frames = latent_start + 1
    action_history = latent_start * 20
    action_target = chunk_size * 20
    grid_height = grid_width = 10
    tokens_per_frame = grid_height * grid_width
    layout = PaperZeroWAM._layout(
        probe,
        prompt_frames,
        history_frames,
        chunk_size,
        grid_height,
        grid_width,
        action_history,
        action_target,
        torch.device("cpu"),
        prompt_enabled,
    )
    mask = layout.attention_mask
    prompt = layout.prompt
    history = layout.robot_history
    target = layout.video_target
    video_length = layout.video_positions.shape[0]
    action_history_total = slice(video_length, video_length + action_history)
    action_target_total = slice(
        video_length + action_history,
        video_length + action_history + action_target,
    )
    inverse_dynamics_layout = PaperZeroWAM._layout(
        probe,
        prompt_frames,
        history_frames,
        chunk_size,
        grid_height,
        grid_width,
        action_history,
        action_target,
        torch.device("cpu"),
        False,
    )

    checks: dict[str, bool] = {
        "square_nonempty_mask": mask.ndim == 2
        and mask.shape[0] == mask.shape[1]
        and bool(mask.any(dim=1).all()),
        "video_target_reads_complete_robot_history": bool(mask[target, history].all()),
        "video_target_reads_complete_action_history": action_history == 0
        or bool(mask[target, action_history_total].all()),
        "video_target_reads_own_noisy_chunk": bool(mask[target, target].all()),
        "video_target_cannot_read_action_target": not bool(
            mask[target, action_target_total].any()
        ),
        "video_target_prompt_membership_exact": bool(mask[target, prompt].all())
        if prompt_enabled
        else not bool(mask[target, prompt].any()),
        "action_target_reads_robot_history": bool(mask[action_target_total, history].all()),
        "action_target_reads_future_video": bool(mask[action_target_total, target].all()),
        "action_target_reads_action_history": action_history == 0
        or bool(mask[action_target_total, action_history_total].all()),
        "action_target_reads_own_noisy_chunk": bool(
            mask[action_target_total, action_target_total].all()
        ),
        "action_target_cannot_directly_read_prompt": not bool(
            mask[action_target_total, prompt].any()
        ),
        "inverse_dynamics_pass_isolates_prompt_from_all_robot_and_action_queries": not bool(
            inverse_dynamics_layout.attention_mask[prompt.stop :, prompt].any()
        ),
        "prompt_cannot_read_robot_or_actions": not bool(mask[prompt, prompt.stop :].any()),
        "prompt_mask_membership_exact": bool(mask[prompt, prompt].all())
        if prompt_enabled
        else bool(
            torch.equal(
                mask[prompt, prompt],
                torch.eye(prompt.stop - prompt.start, dtype=torch.bool),
            )
        ),
        "human_height_rope_offset_exact": int(
            layout.video_positions[prompt.start : prompt.stop, 1].min().item()
        )
        == 32,
        "robot_height_rope_starts_zero": int(
            layout.video_positions[history.start : target.stop, 1].min().item()
        )
        == 0,
        "action_rope_uses_released_fractional_time_grid": bool(
            torch.allclose(
                layout.action_positions[: min(20, layout.action_positions.shape[0]), 0],
                torch.arange(
                    1,
                    min(20, layout.action_positions.shape[0]) + 1,
                    dtype=torch.float32,
                )
                / 21.0,
            )
        )
        and (
            layout.action_positions.shape[0] <= 20
            or abs(
                float(layout.action_positions[20, 0].item())
                - (1.0 + 1.0 / 21.0)
            )
            < 1.0e-6
        ),
        "action_rope_uses_negative_spatial_sentinel": bool(
            (layout.action_positions[:, 1:] == -1).all()
        ),
    }

    for frame in range(history_frames):
        query = slice(
            history.start + frame * tokens_per_frame,
            history.start + (frame + 1) * tokens_per_frame,
        )
        visible_history_stop = history.start + (frame + 1) * tokens_per_frame
        action_limit = min(action_history, frame * 20)
        checks[f"history_frame_{frame}_causal_video"] = bool(
            mask[query, history.start:visible_history_stop].all()
        ) and not bool(mask[query, visible_history_stop:target.stop].any())
        checks[f"history_frame_{frame}_causal_action"] = (
            action_limit == 0
            or bool(
                mask[
                    query,
                    action_history_total.start : action_history_total.start + action_limit,
                ].all()
            )
        ) and not bool(
            mask[
                query,
                action_history_total.start + action_limit : action_target_total.stop,
            ].any()
        )
        checks[f"history_frame_{frame}_prompt_membership"] = (
            bool(mask[query, prompt].all())
            if prompt_enabled
            else not bool(mask[query, prompt].any())
        )

    for action_index in range(action_history):
        query = video_length + action_index
        visible_frames = min(history_frames, action_index // 20 + 1)
        visible_video_stop = history.start + visible_frames * tokens_per_frame
        checks[f"action_history_{action_index}_causal"] = (
            bool(mask[query, history.start:visible_video_stop].all())
            and not bool(mask[query, visible_video_stop:target.stop].any())
            and bool(mask[query, action_history_total.start : query + 1].all())
            and not bool(mask[query, query + 1 : action_target_total.stop].any())
            and not bool(mask[query, prompt].any())
        )

    actual_tokens = mask.shape[0]
    expected_tokens = (
        (prompt_frames + history_frames + chunk_size) * tokens_per_frame
        + action_history
        + action_target
    )
    checks["actual_sequence_tokens_exact"] = actual_tokens == expected_tokens
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "latent_start": latent_start,
        "chunk_size": chunk_size,
        "prompt_enabled": prompt_enabled,
        "sequence_tokens": actual_tokens,
        "check_count": len(checks),
        "failed_checks": failed,
        "passed": not failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PaperZeroWAMConfig()
    config.validate()
    rows = read_jsonl(args.schedule.resolve())
    geometries = sorted(
        {
            (int(row["latent_start"]), int(row["chunk_size"]), bool(row["prompt_enabled"]))
            for row in rows
        }
    )
    probe = LayoutProbe(config)
    records = [
        audit_geometry(
            probe,
            latent_start=latent_start,
            chunk_size=chunk_size,
            prompt_enabled=prompt_enabled,
        )
        for latent_start, chunk_size, prompt_enabled in geometries
    ]
    scheduled_main_tokens = sum(
        2 * int(row["main_sequence_tokens_per_pass"]) for row in rows
    )
    independently_recomputed_main_tokens = sum(
        2
        * (
            (16 + int(row["latent_stop"]) + 1) * 100
            + int(row["latent_stop"]) * 20
        )
        for row in rows
    )
    steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        steps[int(row["global_step"])].append(row)
    mixed_ifp_validity_steps = sum(
        any(
            len({bool(row["ifp_valid"][head]) for row in step_rows}) > 1
            for head in range(config.ifp_heads)
        )
        for step_rows in steps.values()
    )
    ifp_schedule_exact = all(
        [int(value) for value in row["ifp_latent_starts"]]
        == [
            int(row["latent_start"])
            + int(row["chunk_size"]) * (1 + head * config.ifp_stride)
            for head in range(config.ifp_heads)
        ]
        and [bool(value) for value in row["ifp_valid"]]
        == [
            int(start) + int(row["chunk_size"])
            <= config.latent_transitions_per_trajectory
            for start in row["ifp_latent_starts"]
        ]
        and len(row["ifp_executed_sequence_tokens"]) == config.ifp_heads
        and all(int(value) > 0 for value in row["ifp_executed_sequence_tokens"])
        and all(
            int(supervised)
            == (int(executed) if bool(valid) else 0)
            for supervised, executed, valid in zip(
                row["ifp_sequence_tokens"],
                row["ifp_executed_sequence_tokens"],
                row["ifp_valid"],
                strict=True,
            )
        )
        for row in rows
    )
    validity_patterns = {
        tuple(bool(value) for value in row["ifp_valid"]) for row in rows
    }
    ifp_execution_plan_exact = all(
        ifp_head_execution_plan(list(pattern), config.ifp_heads)
        == tuple((head, pattern[head]) for head in range(config.ifp_heads))
        for pattern in validity_patterns
    )
    scheduled_ifp_executed_tokens = sum(
        sum(int(value) for value in row["ifp_executed_sequence_tokens"])
        for row in rows
    )
    independently_recomputed_ifp_executed_tokens = sum(
        config.ifp_heads
        * (
            (
                int(row["latent_start"])
                + 1
                + 2 * int(row["chunk_size"])
            )
            * 100
            + int(row["latent_start"]) * 20
        )
        for row in rows
    )
    result = {
        "protocol": "paper_zero_wam_attention_contract_audit_v3",
        "passed": bool(records)
        and all(record["passed"] for record in records)
        and len(rows) == 33_600
        and scheduled_main_tokens == independently_recomputed_main_tokens
        and len(steps) == config.optimizer_steps
        and mixed_ifp_validity_steps > 0
        and ifp_schedule_exact
        and ifp_execution_plan_exact
        and scheduled_ifp_executed_tokens
        == independently_recomputed_ifp_executed_tokens,
        "formal_schedule_rows": len(rows),
        "unique_production_geometries": len(records),
        "geometry_records": records,
        "scheduled_two_pass_main_tokens": scheduled_main_tokens,
        "independently_recomputed_two_pass_main_tokens": independently_recomputed_main_tokens,
        "mixed_ifp_validity_steps_requiring_rank_symmetric_head_execution": mixed_ifp_validity_steps,
        "ifp_schedule_exact": ifp_schedule_exact,
        "ifp_validity_patterns": [list(pattern) for pattern in sorted(validity_patterns)],
        "production_ifp_execution_plan_exact": ifp_execution_plan_exact,
        "scheduled_ifp_executed_tokens": scheduled_ifp_executed_tokens,
        "independently_recomputed_ifp_executed_tokens": independently_recomputed_ifp_executed_tokens,
        "all_four_ifp_heads_execute_on_every_rank": ifp_execution_plan_exact,
        "invalid_ifp_targets_have_zero_supervision_weight": True,
        "uses_production_layout_function": True,
        "model_constructed": False,
        "optimizer_updates": 0,
        "hash_checks": False,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "formal_schedule_rows": len(rows),
                "unique_production_geometries": len(records),
                "scheduled_two_pass_main_tokens": scheduled_main_tokens,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit("production attention contract audit failed")


if __name__ == "__main__":
    main()
