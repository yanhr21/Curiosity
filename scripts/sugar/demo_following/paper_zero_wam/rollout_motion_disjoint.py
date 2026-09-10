#!/usr/bin/env python3
"""Run four-condition live PhysX rollouts for one held-out SUGAR motion."""

from __future__ import annotations

import atexit
import argparse
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .data import ActionNormalizer, load_validated_latent_payload, read_jsonl
from .model import IFPHead, PaperMoTLayer, PaperZeroWAM
from .physical_metrics import RESTORE_PAYLOAD_KEYS, physical_summary
from .train import validated_checkpoint_step
from .train_single_gpu import single_gpu_checkpoint_step
from .rollout_smallbox import (
    ACTION_CHUNK,
    CausalRolloutBatch,
    RGB_STRIDE,
    ROLLOUT_STEPS,
    broadcast_executed_actions,
    broadcast_rgb,
    encode_history,
    gather_tensor,
    load_frame,
    run_endpoint_baseline,
    setup,
    start_simulator,
    stop_simulator,
)


CONDITIONS = ("matched", "reversed", "same_task_alternate", "wrong_task")
ENDPOINT_CONDITIONS = ("matched_endpoint", "wrong_task_endpoint")
RANK_CONDITIONS = tuple(CONDITIONS[rank % 4] for rank in range(8))


def cache_path(root: Path, row: dict[str, Any]) -> Path:
    return root / row["split"] / row["task"] / f'{int(row["source_motion_id"]):03d}.pt'


def prompt_spec(
    target: dict[str, Any],
    condition: str,
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[str, Any], str, list[str]]:
    if condition in ("matched", "reversed"):
        row = target
    else:
        spec = target["fixed_counterfactual_prompts"][condition]
        row = rows_by_key[(target["split"], spec["task"], int(spec["source_motion_id"]))]
    key = "reversed_prompt_latents" if condition == "reversed" else "prompt_latents"
    paths = list(row["prompt"]["frame_paths"])
    if condition == "reversed":
        paths.reverse()
    return row, key, paths


def render_condition_grid(
    profile_id: int,
    condition_frames: list[list[np.ndarray]],
    prompt_paths: dict[str, list[str]],
    output: Path,
    ffmpeg: Path,
) -> int:
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x704", "-r", "10", "-i", "-", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "18", str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    frame_count = min(len(frames) for frames in condition_frames)
    for frame_index in range(frame_count):
        prompt_index = int(round(frame_index * 63 / max(1, frame_count - 1)))
        canvas = Image.new("RGB", (1280, 704), "black")
        draw = ImageDraw.Draw(canvas)
        for condition_index, condition in enumerate(CONDITIONS):
            demo = load_frame(prompt_paths[condition][prompt_index])
            canvas.paste(Image.fromarray(demo), (condition_index * 320, 32))
            canvas.paste(
                Image.fromarray(condition_frames[condition_index][frame_index]),
                (condition_index * 320, 384),
            )
            draw.text((condition_index * 320 + 8, 8), f"{condition} demo", fill="white")
            draw.text(
                (condition_index * 320 + 8, 360),
                f"{condition} PhysX profile {profile_id}",
                fill="white",
            )
        process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"motion-grid ffmpeg failed: {stderr[-2000:]}")
    return frame_count


def run_profile_batch(
    *,
    args: argparse.Namespace,
    config: PaperZeroWAMConfig,
    model: FSDP,
    vae: Any,
    normalizer: ActionNormalizer,
    target: dict[str, Any],
    prompt: dict[int, torch.Tensor],
    prompt_paths: dict[str, list[str]],
    prompt_specs: list[dict[str, Any]],
    rank: int,
    device: torch.device,
    profile_batch: int,
    source_dir: Path,
    full_parameter_count: int,
    checkpoint_step: int,
) -> dict[str, Any] | None:
    batch_dir = source_dir / f"profile_batch{profile_batch:02d}"
    result: dict[str, Any] | None = None
    terminal_state_value = 0
    if rank == 0:
        batch_dir.mkdir(parents=True, exist_ok=True)
        result_path = batch_dir / "BATCH_RESULT.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                terminal_state_value = (
                    1
                    if result.get("protocol")
                    == "paper_zero_wam_motion_disjoint_physical_batch_v1"
                    and int(result.get("source_index", -1)) == args.source_index
                    and result.get("target_task") == target["task"]
                    and int(result.get("source_motion_id", -1))
                    == int(target["source_motion_id"])
                    and int(result.get("profile_batch", -1)) == profile_batch
                    and result.get("passed_execution_contract") is True
                    and int(result.get("checkpoint_step", -1)) == 4200
                    and int(result.get("architecture_parameter_count", -1))
                    == config.expected_parameter_count
                    and int(result.get("adapted_rollout_count", -1)) == 8
                    and int(result.get("released_endpoint_rollout_count", -1)) == 4
                    and int(result.get("rollout_count", -1)) == 12
                    and result.get("video_frame_counts") == [131] * 2
                    and result.get("prompt_specs") == prompt_specs
                    and result.get("hash_checks") is False
                    and set(
                        (result.get("released_endpoint_tracker_history_reinitialized") or {})
                    ) == {"matched_endpoint", "wrong_task_endpoint"}
                    and all(
                        (result.get("released_endpoint_tracker_history_reinitialized") or {}).values()
                    )
                    else 2
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                terminal_state_value = 2
    terminal_result = torch.tensor(
        terminal_state_value,
        device=device,
    )
    dist.broadcast(terminal_result, src=0)
    if int(terminal_result.item()) == 2:
        raise ValueError(
            "existing motion-disjoint terminal result does not match this batch"
        )
    if int(terminal_result.item()) == 1:
        dist.barrier()
        return result
    dist.barrier()
    process = None
    connection = None
    ready = None
    if rank == 0:
        address = Path(
            f"/tmp/paper_zero_wam_motion_{os.environ.get('SLURM_JOB_ID', 'local')}_"
            f"{int(target['source_motion_id'])}_{profile_batch}.sock"
        )
        process, connection, ready = start_simulator(
            address,
            config.noise_seed
            + 120_000
            + args.source_index * 101
            + profile_batch,
            batch_dir,
            copies_per_profile=4,
        )
        atexit.register(stop_simulator, process, connection)

    runtime = CausalRolloutBatch(
        model=model, vae=vae, normalizer=normalizer, config=config, rank=rank, device=device,
        prompts=prompt, initial_rgb=None if ready is None else np.asarray(ready["initial_rgb"]),
        single_gpu=args.single_gpu)
    all_rgb: list[np.ndarray] | None = [] if rank == 0 else None
    if rank == 0:
        assert ready is not None and all_rgb is not None
        all_rgb.append(np.asarray(ready["initial_rgb"]).copy())
    trace_chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "robot_root_state_w", "robot_joint_pos", "robot_joint_vel",
            "object_root_state_w", "contact", "requested_action",
            "executed_action", "done",
        )
    }
    future_differences: list[list[list[float]]] = []
    action_differences: list[list[list[float]]] = []
    executed_steps = 0
    chunk_index = 0
    while executed_steps < ROLLOUT_STEPS:
        gathered_video, gathered_actions = runtime.predict([
            config.noise_seed + 130_000 + args.source_index * 10_003
            + (profile_batch * 2 + slot // 4) * 101 + chunk_index
            for slot in range(8)
        ])
        execute_count = min(ACTION_CHUNK, ROLLOUT_STEPS - executed_steps)
        response = None
        if rank == 0:
            assert connection is not None and gathered_video is not None and gathered_actions is not None
            future_differences.append(
                [
                    [
                        float((gathered_video[start] - gathered_video[start + offset]).abs().max())
                        for offset in range(1, 4)
                    ]
                    for start in (0, 4)
                ]
            )
            action_differences.append(
                [
                    [
                        float((gathered_actions[start] - gathered_actions[start + offset]).abs().max())
                        for offset in range(1, 4)
                    ]
                    for start in (0, 4)
                ]
            )
            connection.send(
                {
                    "command": "step_chunk",
                    "actions": gathered_actions[:, :execute_count].float().cpu().numpy(),
                }
            )
            response = connection.recv()
            if response.get("status") == "error":
                raise RuntimeError(response["error"])
            for name in trace_chunks:
                trace_chunks[name].append(np.asarray(response[name]))
            assert all_rgb is not None
            all_rgb.extend(frame.copy() for frame in np.asarray(response["rgb"]))
        runtime.append_response(response, execute_count)
        executed_steps += execute_count
        chunk_index += 1
        if rank == 0:
            emit_json_best_effort(
                {
                    "source_index": args.source_index,
                    "profile_batch": profile_batch,
                    "executed_steps": executed_steps,
                }
            )

    result = None
    if rank == 0:
        assert connection is not None and process is not None and ready is not None and all_rgb is not None
        connection.send({"command": "close"})
        closed = connection.recv()
        if closed.get("status") != "closed":
            raise RuntimeError(f"simulator close response invalid: {closed}")
        connection.close()
        if process.wait(timeout=120) != 0:
            raise RuntimeError("motion-disjoint simulator exited nonzero")
        atexit.unregister(stop_simulator)
        trace = {name: np.concatenate(values, axis=0) for name, values in trace_chunks.items()}
        if any(value.shape[0] != ROLLOUT_STEPS for value in trace.values()):
            raise RuntimeError("motion-disjoint trace length is not 650")
        initial_state = {
            f"initial_{name}": np.asarray(value)
            for name, value in ready["initial_state"].items()
        }
        restore_state = {
            f"restore_{name}": np.asarray(value)
            for name, value in ready["restore_payload"].items()
        }
        adapted_trace_path = batch_dir / "TRACE.npz"
        np.savez_compressed(adapted_trace_path, **initial_state, **restore_state, **trace)
        simulator_seed = (
            config.noise_seed
            + 120_000
            + args.source_index * 101
            + profile_batch
        )
        wrong_task = "KickBox" if target["task"] == "CarryBox" else "CarryBox"
        endpoint_runs = {
            "matched_endpoint": run_endpoint_baseline(
                address.with_name(f"{address.stem}_matched_endpoint.sock"),
                simulator_seed,
                batch_dir,
                f'{target["task"]}_endpoint',
                copies_per_profile=4,
                restore_payload=ready["restore_payload"],
            ),
            "wrong_task_endpoint": run_endpoint_baseline(
                address.with_name(f"{address.stem}_wrong_endpoint.sock"),
                simulator_seed,
                batch_dir,
                f"{wrong_task}_endpoint",
                copies_per_profile=4,
                restore_payload=ready["restore_payload"],
            ),
        }
        endpoint_archive: dict[str, np.ndarray] = {}
        adapted_profile_sources = np.asarray([0, 4], dtype=np.int64)
        endpoint_initial_state_exact: dict[str, bool] = {}
        endpoint_initial_rgb_exact: dict[str, bool] = {}
        endpoint_profile_readback_exact: dict[str, bool] = {}
        endpoint_observation_history_exact: dict[str, bool] = {}
        for condition, endpoint_run in endpoint_runs.items():
            for name, value in endpoint_run["trace"].items():
                endpoint_archive[f"{condition}_{name}"] = value
            for name, value in endpoint_run["initial_state"].items():
                endpoint_archive[f"{condition}_initial_{name}"] = value
            for name, value in endpoint_run["restore_payload"].items():
                endpoint_archive[f"{condition}_restore_{name}"] = value
            endpoint_initial_state_exact[condition] = all(
                np.array_equal(
                    np.asarray(ready["initial_state"][name])[adapted_profile_sources],
                    value,
                )
                for name, value in endpoint_run["initial_state"].items()
            )
            endpoint_initial_rgb_exact[condition] = np.array_equal(
                np.asarray(ready["initial_rgb"])[adapted_profile_sources],
                endpoint_run["initial_rgb"],
            )
            endpoint_profile_readback_exact[condition] = (
                endpoint_run.get("restore_exact") is True
                and all(
                    np.array_equal(
                        np.asarray(ready["restore_payload"][name]), value
                    )
                    for name, value in endpoint_run["restore_payload"].items()
                )
            )
            endpoint_observation_history_exact[condition] = all(
                endpoint_run["tracker_observation_history_reinitialized"].values()
            )
        endpoint_trace_path = batch_dir / "ENDPOINT_TRACES.npz"
        np.savez_compressed(endpoint_trace_path, **endpoint_archive)
        rollouts: list[dict[str, Any]] = []
        videos: list[str] = []
        video_frame_counts: list[int] = []
        prompt_spec_by_condition = {
            str(spec["condition"]): spec for spec in prompt_specs
        }
        if set(prompt_spec_by_condition) != set(CONDITIONS):
            raise RuntimeError("physical prompt-spec condition topology changed")
        for env_index in range(8):
            condition = CONDITIONS[env_index % 4]
            prompt_spec = prompt_spec_by_condition[condition]
            rollouts.append(
                {
                    "profile_id": profile_batch * 2 + env_index // 4,
                    "condition": condition,
                    "route": "adapted",
                    "target_task": target["task"],
                    "prompt_split": prompt_spec["split"],
                    "prompt_task": prompt_spec["task"],
                    "prompt_source_motion_id": int(
                        prompt_spec["source_motion_id"]
                    ),
                    "prompt_latent_key": prompt_spec["latent_key"],
                    "prompt_reversed": prompt_spec["latent_key"]
                    == "reversed_prompt_latents",
                    "source_motion_id": int(target["source_motion_id"]),
                    "done_count": int(trace["done"][:, env_index].sum()),
                    "physical": physical_summary(
                        trace["robot_root_state_w"][:, env_index],
                        trace["object_root_state_w"][:, env_index],
                        trace["contact"][:, env_index],
                    ),
                }
            )
        endpoint_rollouts: list[dict[str, Any]] = []
        for condition, endpoint_run in endpoint_runs.items():
            prompt_task = target["task"] if condition == "matched_endpoint" else wrong_task
            endpoint_trace = endpoint_run["trace"]
            for profile_offset, environment_index in enumerate((0, 4)):
                endpoint_rollouts.append(
                    {
                        "profile_id": profile_batch * 2 + profile_offset,
                        "environment_index": environment_index,
                        "condition": condition,
                        "route": "released_endpoint",
                        "target_task": target["task"],
                        "prompt_task": prompt_task,
                        "source_motion_id": int(target["source_motion_id"]),
                        "done_count": int(
                            endpoint_trace["done"][:, profile_offset].sum()
                        ),
                        "physical": physical_summary(
                            endpoint_trace["robot_root_state_w"][:, profile_offset],
                            endpoint_trace["object_root_state_w"][:, profile_offset],
                            endpoint_trace["contact"][:, profile_offset],
                        ),
                    }
                )
        for group_index, start in enumerate((0, 4)):
            profile_id = profile_batch * 2 + group_index
            path = batch_dir / f"profile{profile_id:02d}_four_conditions.mp4"
            video_frame_counts.append(
                render_condition_grid(
                    profile_id,
                    [
                        [frame[start + offset] for frame in all_rgb]
                        for offset in range(4)
                    ],
                    prompt_paths,
                    path,
                    args.ffmpeg,
                )
            )
            videos.append(str(path))
        dependency_checks = []
        for group_index in range(2):
            for intervention_index in range(3):
                first_future = next(
                    (
                        chunk
                        for chunk, value in enumerate(future_differences)
                        if value[group_index][intervention_index] > 0.0
                    ),
                    math.inf,
                )
                first_action = next(
                    (
                        chunk
                        for chunk, value in enumerate(action_differences)
                        if value[group_index][intervention_index] > 1.0e-6
                    ),
                    math.inf,
                )
                dependency_checks.append(first_future <= first_action < math.inf)
        profile_result = ready["profile_result"]
        checks = {
            "full_650_step_trace": all(value.shape[0] == ROLLOUT_STEPS for value in trace.values()),
            "zero_resets": not bool(trace["done"].any()),
            "four_condition_initial_state_rgb_physics_exact": (
                profile_result.get("paired_profiles_passed") is True
                and profile_result.get("copies_per_profile") == 4
                and profile_result.get("profile_count") == 2
            ),
            "all_values_finite": all(
                np.isfinite(value).all()
                for name, value in {**initial_state, **trace}.items()
                if name not in ("done", "contact")
            ),
            "every_requested_action_equals_physx_executed_action": np.array_equal(
                trace["requested_action"], trace["executed_action"]
            ),
            "four_released_endpoint_traces_present": len(endpoint_rollouts) == 4
            and all(
                value.shape[0] == ROLLOUT_STEPS and value.shape[1] == 2
                for endpoint_run in endpoint_runs.values()
                for value in endpoint_run["trace"].values()
            ),
            "released_endpoint_traces_finite_reset_free": all(
                not bool(endpoint_run["trace"]["done"].any())
                and all(
                    np.isfinite(value).all()
                    for name, value in endpoint_run["trace"].items()
                    if name not in ("done", "contact")
                )
                for endpoint_run in endpoint_runs.values()
            ),
            "released_endpoint_requested_actions_equal_physx_readback": all(
                np.array_equal(
                    endpoint_run["trace"]["requested_action"],
                    endpoint_run["trace"]["executed_action"],
                )
                for endpoint_run in endpoint_runs.values()
            ),
            "adapted_endpoint_initial_state_rgb_physics_exact": (
                all(endpoint_initial_state_exact.values())
                and all(endpoint_initial_rgb_exact.values())
                and all(endpoint_profile_readback_exact.values())
                and all(endpoint_observation_history_exact.values())
            ),
            "all_six_prompt_interventions_change_future_before_action": all(dependency_checks),
            "two_nonempty_h264_grids": len(videos) == 2
            and video_frame_counts == [131] * 2
            and all(Path(path).stat().st_size > 0 for path in videos),
        }
        result = {
            "protocol": "paper_zero_wam_motion_disjoint_physical_batch_v1",
            "execution_world_size": 1 if args.single_gpu else 8,
            "serial_inference_cases_per_worker": 8 if args.single_gpu else 1,
            "passed_execution_contract": all(checks.values()),
            "source_index": args.source_index,
            "split": "test",
            "target_task": target["task"],
            "source_motion_id": int(target["source_motion_id"]),
            "profile_batch": profile_batch,
            "profile_ids": [profile_batch * 2, profile_batch * 2 + 1],
            "checkpoint_step": checkpoint_step,
            "architecture_parameter_count": full_parameter_count,
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
            "checks": checks,
            "adapted_rollout_count": 8,
            "released_endpoint_rollout_count": 4,
            "rollout_count": 12,
            "prompt_specs": prompt_specs,
            "profile_readback": profile_result,
            "released_endpoint_routes": {
                condition: endpoint_run["route"]
                for condition, endpoint_run in endpoint_runs.items()
            },
            "released_endpoint_files": {
                condition: endpoint_run["endpoint_files"]
                for condition, endpoint_run in endpoint_runs.items()
            },
            "adapted_endpoint_initial_state_exact": endpoint_initial_state_exact,
            "adapted_endpoint_initial_rgb_exact": endpoint_initial_rgb_exact,
            "adapted_endpoint_profile_readback_exact": endpoint_profile_readback_exact,
            "released_endpoint_tracker_history_reinitialized": endpoint_observation_history_exact,
            "future_max_abs_vs_matched_by_chunk": future_differences,
            "action_max_abs_vs_matched_by_chunk": action_differences,
            "maximum_requested_executed_action_error": max(
                float(
                    np.max(
                        np.abs(
                            trace["requested_action"].astype(np.float64)
                            - trace["executed_action"].astype(np.float64)
                        )
                    )
                ),
                *(
                    float(
                        np.max(
                            np.abs(
                                endpoint_run["trace"]["requested_action"].astype(np.float64)
                                - endpoint_run["trace"]["executed_action"].astype(np.float64)
                            )
                        )
                    )
                    for endpoint_run in endpoint_runs.values()
                ),
            ),
            "rollouts": [*rollouts, *endpoint_rollouts],
            "adapted_trace": str(adapted_trace_path),
            "released_endpoint_trace": str(endpoint_trace_path),
            "videos": videos,
            "video_frame_counts": video_frame_counts,
            "hash_checks": False,
            "claim_boundary": "One motion-disjoint four-prompt physical batch; aggregate decision required.",
        }
    dist.barrier()
    if rank == 0:
        assert result is not None
        # The per-batch terminal becomes reusable only after every FSDP rank
        # completed this batch.  A failed publication leaves the source-level
        # guardian free to repeat only this missing batch.
        write_json_atomic(batch_dir / "BATCH_RESULT.json", result)
    dist.barrier()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=int, choices=range(19), required=True)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(
            "/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/"
            "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
        ),
    )
    args = parser.parse_args()
    config = PaperZeroWAMConfig()
    config.validate()
    rank, _world_size, device = setup(args.single_gpu)
    rows = read_jsonl(config.resolved(config.manifest))
    test_rows = sorted(
        (row for row in rows if row["split"] == "test"),
        key=lambda row: (row["task"], int(row["source_motion_id"])),
    )
    target = test_rows[args.source_index]
    case_manifest = json.loads(args.case_manifest.resolve().read_text(encoding="utf-8"))
    expected_totals = {
        "test_sources": 19,
        "profiles_per_source": 10,
        "conditions_per_profile": 4,
        "endpoint_routes_per_profile": 2,
        "adapted_rollouts": 760,
        "released_endpoint_rollouts": 380,
        "rollouts": 1_140,
        "executed_actions": 741_000,
        "requested_executed_action_rows": 741_000,
        "visualizations": 190,
    }
    expected_clock = {
        "control_hz": 50,
        "rgb_hz": 10,
        "video_latents_per_chunk": config.inference_chunk_size,
        "actions_per_chunk": config.inference_chunk_size * 20,
        "rollout_actions": 650,
    }
    expected_inference = {
        "chunk_size": config.inference_chunk_size,
        "video_guidance_scale": config.video_guidance_scale,
        "action_guidance_scale": config.action_guidance_scale,
        "flow_integrator": config.flow_integrator,
        "video_steps": config.video_inference_steps,
        "action_steps": config.action_inference_steps,
        "video_snr_shift": config.video_snr_shift,
        "action_snr_shift": config.action_snr_shift,
    }
    if (
        case_manifest.get("protocol") != "paper_zero_wam_motion_disjoint_cases_v1"
        or case_manifest.get("frozen_before_model_outcomes") is not True
        or case_manifest.get("conditions") != list(CONDITIONS)
        or case_manifest.get("endpoint_conditions") != list(ENDPOINT_CONDITIONS)
        or case_manifest.get("totals") != expected_totals
        or case_manifest.get("clock") != expected_clock
        or case_manifest.get("inference") != expected_inference
        or case_manifest.get("hash_checks") is not False
        or len(case_manifest.get("cases", [])) != 19
    ):
        raise ValueError("motion-disjoint case manifest is not admitted")
    frozen_case = case_manifest["cases"][args.source_index]
    if (
        int(frozen_case["source_index"]) != args.source_index
        or frozen_case["target_task"] != target["task"]
        or int(frozen_case["source_motion_id"]) != int(target["source_motion_id"])
        or [row["condition"] for row in frozen_case["prompts"]] != list(CONDITIONS)
        or [int(row["profile_id"]) for row in frozen_case["profiles"]] != list(range(10))
    ):
        raise ValueError("runtime source selection differs from the frozen case manifest")
    rows_by_key = {
        (row["split"], row["task"], int(row["source_motion_id"])): row for row in rows
    }
    resolved_prompts = []
    for condition in CONDITIONS:
        resolved_row, resolved_key, _ = prompt_spec(target, condition, rows_by_key)
        resolved_prompts.append(
            {
                "condition": condition,
                "split": str(target["split"]),
                "task": resolved_row["task"],
                "source_motion_id": int(resolved_row["source_motion_id"]),
                "latent_key": resolved_key,
            }
        )
    expected_profiles = [
        {
            "profile_id": profile_id,
            "profile_batch": profile_id // 2,
            "profile_slot": profile_id % 2,
            "simulator_seed": (
                config.noise_seed + 120_000 + args.source_index * 101 + profile_id // 2
            ),
            "inference_seed_base": (
                config.noise_seed
                + 130_000
                + args.source_index * 10_003
                + profile_id * 101
            ),
        }
        for profile_id in range(10)
    ]
    if frozen_case["prompts"] != resolved_prompts or frozen_case["profiles"] != expected_profiles:
        raise ValueError("runtime prompts or seeds differ from the frozen case manifest")
    source_dir = (
        args.output_root.resolve()
        / f"source{args.source_index:02d}_{target['task']}_{int(target['source_motion_id']):03d}"
    )
    existing: dict[str, Any] | None = None
    terminal_state_value = 0
    source_result: dict[str, Any] | None = None
    if rank == 0:
        source_dir.mkdir(parents=True, exist_ok=True)
        result_path = source_dir / "SOURCE_RESULT.json"
        if result_path.is_file():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                terminal_state_value = (
                    1
                    if existing.get("protocol")
                    == "paper_zero_wam_motion_disjoint_physical_source_v1"
                    and int(existing.get("source_index", -1)) == args.source_index
                    and existing.get("target_task") == target["task"]
                    and int(existing.get("source_motion_id", -1))
                    == int(target["source_motion_id"])
                    and int(existing.get("adapted_rollout_count", -1)) == 40
                    and int(existing.get("released_endpoint_rollout_count", -1)) == 20
                    and int(existing.get("rollout_count", -1)) == 60
                    and existing.get("prompt_specs") == resolved_prompts
                    and existing.get("visualization_frame_counts") == [131] * 10
                    and existing.get("hash_checks") is False
                    else 2
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                terminal_state_value = 2
    terminal_source = torch.tensor(
        terminal_state_value,
        device=device,
    )
    dist.broadcast(terminal_source, src=0)
    if int(terminal_source.item()) == 2:
        raise ValueError("existing motion-disjoint terminal source result does not match")
    if int(terminal_source.item()) == 1:
        if rank == 0:
            assert existing is not None
            emit_json_best_effort(
                {
                    "source_index": args.source_index,
                    "terminal_source_result_reused": True,
                    "passed": existing.get("passed"),
                }
            )
        dist.barrier()
        dist.destroy_process_group()
        return
    dist.barrier()

    model = PaperZeroWAM.from_wan_pretrained(config, dtype=torch.float32)
    full_parameter_count = model.parameter_count
    if full_parameter_count != config.expected_parameter_count:
        raise RuntimeError("motion-disjoint model parameter contract changed")
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({PaperMoTLayer, IFPHead}),
        sharding_strategy=ShardingStrategy.NO_SHARD if args.single_gpu else ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=None, reduce_dtype=torch.bfloat16, buffer_dtype=None
        ),
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    checkpoint_state = json.loads(
        (args.checkpoint / "STATE.json").read_text(encoding="utf-8")
    )
    checkpoint_step = (single_gpu_checkpoint_step(checkpoint_state, config) if args.single_gpu
                       else validated_checkpoint_step(checkpoint_state, "formal", config))
    checkpoint = torch.load(
        args.checkpoint / f"model_rank{rank:02d}.pt", map_location="cpu", weights_only=False
    )
    if (
        checkpoint_step != config.optimizer_steps
        or int(checkpoint.get("step", -1)) != checkpoint_step
    ):
        raise ValueError("motion-disjoint rollout requires the complete step-4200 checkpoint")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT if args.single_gpu
                             else StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()

    source_root = config.resolved(config.wan_source)
    if "wan.modules" not in sys.modules:
        package = types.ModuleType("wan")
        package.__path__ = [str(source_root / "wan")]
        package.__package__ = "wan"
        sys.modules["wan"] = package
        modules_package = types.ModuleType("wan.modules")
        modules_package.__path__ = [str(source_root / "wan" / "modules")]
        modules_package.__package__ = "wan.modules"
        sys.modules["wan.modules"] = modules_package
    from wan.modules.vae2_2 import Wan2_2_VAE

    vae = Wan2_2_VAE(
        vae_pth=str(config.resolved(config.wan_checkpoint) / "Wan2.2_VAE.pth"),
        dtype=torch.bfloat16,
        device=str(device),
    )
    cache_root = config.resolved(config.latent_cache)
    prompt_cache, payload_cache, prompt = {}, {}, {}
    for slot in (range(8) if args.single_gpu else [rank]):
        condition = RANK_CONDITIONS[slot]
        if condition not in prompt_cache:
            prompt_row, prompt_key, _ = prompt_spec(target, condition, rows_by_key)
            path = cache_path(cache_root, prompt_row)
            if path not in payload_cache:
                payload_cache[path] = load_validated_latent_payload(
                    path, split=str(prompt_row["split"]), task=str(prompt_row["task"]),
                    source_motion_id=int(prompt_row["source_motion_id"]))
            prompt_cache[condition] = payload_cache[path][prompt_key].to(device=device, dtype=torch.bfloat16)
        prompt[slot] = prompt_cache[condition]
    prompt_paths = {
        value: prompt_spec(target, value, rows_by_key)[2] for value in CONDITIONS
    }
    normalizer = ActionNormalizer(cache_root / "ACTION_QUANTILES.json")
    batch_results: list[dict[str, Any]] = []
    for profile_batch in range(5):
        result = run_profile_batch(
            args=args,
            config=config,
            model=model,
            vae=vae,
            normalizer=normalizer,
            target=target,
            prompt=prompt,
            prompt_paths=prompt_paths,
            prompt_specs=resolved_prompts,
            rank=rank,
            device=device,
            profile_batch=profile_batch,
            source_dir=source_dir,
            full_parameter_count=full_parameter_count,
            checkpoint_step=checkpoint_step,
        )
        if rank == 0:
            assert result is not None
            batch_results.append(result)
    if rank == 0:
        rollouts = [row for result in batch_results for row in result["rollouts"]]
        adapted_rollouts = [row for row in rollouts if row["route"] == "adapted"]
        endpoint_rollouts = [
            row for row in rollouts if row["route"] == "released_endpoint"
        ]
        profile_vectors = [
            np.asarray(vector, dtype=np.float32)
            for result in batch_results
            for vector in result["profile_readback"]["profile_vectors"]
        ]
        profiles_valid_and_distinct = (
            len(profile_vectors) == 10
            and len({value.shape for value in profile_vectors}) == 1
            and all(np.isfinite(value).all() for value in profile_vectors)
            and all(
                not np.array_equal(profile_vectors[left], profile_vectors[right])
                for left in range(10)
                for right in range(left + 1, 10)
            )
        )
        by_condition = {
            condition: [
                row for row in adapted_rollouts if row["condition"] == condition
            ]
            for condition in CONDITIONS
        }
        endpoint_by_condition = {
            condition: [
                row for row in endpoint_rollouts if row["condition"] == condition
            ]
            for condition in ("matched_endpoint", "wrong_task_endpoint")
        }
        target_success_key = (
            "safe_carry_success" if target["task"] == "CarryBox" else "safe_kick_success"
        )
        wrong_success_key = (
            "safe_kick_success" if target["task"] == "CarryBox" else "safe_carry_success"
        )
        matched_successes = sum(
            bool(row["physical"][target_success_key]) for row in by_condition["matched"]
        )
        wrong_successes = sum(
            bool(row["physical"][wrong_success_key]) for row in by_condition["wrong_task"]
        )
        matched_falls = sum(
            bool(row["physical"]["physical_fall"])
            for row in by_condition["matched"]
        )
        wrong_falls = sum(
            bool(row["physical"]["physical_fall"])
            for row in by_condition["wrong_task"]
        )
        matched_endpoint_falls = sum(
            bool(row["physical"]["physical_fall"])
            for row in endpoint_by_condition["matched_endpoint"]
        )
        wrong_endpoint_falls = sum(
            bool(row["physical"]["physical_fall"])
            for row in endpoint_by_condition["wrong_task_endpoint"]
        )
        checks = {
            "five_batches_passed": len(batch_results) == 5
            and all(result["passed_execution_contract"] for result in batch_results),
            "complete_ten_profile_four_condition_grid": len(adapted_rollouts) == 40
            and {
                (int(row["profile_id"]), str(row["condition"]))
                for row in adapted_rollouts
            }
            == {(profile, condition) for profile in range(10) for condition in CONDITIONS},
            "complete_ten_profile_two_endpoint_grid": len(endpoint_rollouts) == 20
            and {
                (int(row["profile_id"]), str(row["condition"]))
                for row in endpoint_rollouts
            }
            == {
                (profile, condition)
                for profile in range(10)
                for condition in ("matched_endpoint", "wrong_task_endpoint")
            },
            "all_ten_startup_profiles_exactly_distinct": profiles_valid_and_distinct,
            "matched_prompt_task_success_at_least_8_of_10": matched_successes >= 8,
            "wrong_prompt_task_switch_success_at_least_8_of_10": wrong_successes >= 8,
            "zero_falls_reversed_and_same_task_alternate": not any(
                bool(row["physical"]["physical_fall"])
                for condition in ("reversed", "same_task_alternate")
                for row in by_condition[condition]
            ),
            "matched_falls_do_not_exceed_released_endpoint": (
                matched_falls <= matched_endpoint_falls
            ),
            "wrong_task_falls_do_not_exceed_released_endpoint": (
                wrong_falls <= wrong_endpoint_falls
            ),
            "ten_nonempty_visualizations": sum(len(result["videos"]) for result in batch_results) == 10,
            "ten_complete_131_frame_visualizations": [
                int(value)
                for result in batch_results
                for value in result.get("video_frame_counts", [])
            ]
            == [131] * 10,
        }
        source_result = {
            "protocol": "paper_zero_wam_motion_disjoint_physical_source_v1",
            "execution_world_size": _world_size,
            "serial_inference_cases_per_worker": 8 if args.single_gpu else 1,
            "passed": all(checks.values()),
            "source_index": args.source_index,
            "split": "test",
            "target_task": target["task"],
            "source_motion_id": int(target["source_motion_id"]),
            "profile_count": 10,
            "adapted_rollout_count": 40,
            "released_endpoint_rollout_count": 20,
            "rollout_count": 60,
            "prompt_specs": resolved_prompts,
            "matched_prompt_task_successes": matched_successes,
            "wrong_prompt_task_switch_successes": wrong_successes,
            "falls": {
                "matched": matched_falls,
                "matched_endpoint": matched_endpoint_falls,
                "wrong_task": wrong_falls,
                "wrong_task_endpoint": wrong_endpoint_falls,
                "reversed": sum(
                    bool(row["physical"]["physical_fall"])
                    for row in by_condition["reversed"]
                ),
                "same_task_alternate": sum(
                    bool(row["physical"]["physical_fall"])
                    for row in by_condition["same_task_alternate"]
                ),
            },
            "checks": checks,
            "videos": [path for result in batch_results for path in result["videos"]],
            "visualization_frame_counts": [
                int(value)
                for result in batch_results
                for value in result.get("video_frame_counts", [])
            ],
            "hash_checks": False,
            "claim_boundary": "One held-out source; all-source aggregate and held-out flow gate required.",
        }
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        assert source_result is not None
        # SOURCE_RESULT is the guardian terminal and is therefore the final
        # operation after all five batch collectives have completed.
        write_json_atomic(source_dir / "SOURCE_RESULT.json", source_result)
        emit_json_best_effort(source_result)


if __name__ == "__main__":
    main()
