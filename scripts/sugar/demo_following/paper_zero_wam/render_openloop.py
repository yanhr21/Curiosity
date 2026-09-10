#!/usr/bin/env python3
"""Render held-out prompt/predicted-future/ground-truth evidence after training."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
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
from .data import (
    ActionNormalizer,
    load_executed_actions,
    load_validated_latent_payload,
    read_jsonl,
)
from .model import IFPHead, PaperMoTLayer, PaperZeroWAM
from .results import refresh_results_document_best_effort
from .train import validated_checkpoint_step
from .evaluate_heldout import evaluation_scope


CONDITIONS = ("matched", "reversed", "same_task_alternate", "wrong_task")


def setup(single_gpu: bool = False) -> tuple[int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != (1 if single_gpu else 8):
        raise ValueError("render world size does not match selected execution backend")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return rank, torch.device("cuda", local_rank)


def load_frame(path_value: str) -> np.ndarray:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def payload_path(root: Path, row: dict[str, Any]) -> Path:
    return root / str(row["split"]) / str(row["task"]) / f'{int(row["source_motion_id"]):03d}.pt'


def selected_case(
    rank: int, rows: list[dict[str, Any]], cache_root: Path
) -> tuple[dict[str, Any], dict[str, Any], str, torch.Tensor, list[str], bool]:
    task = "CarryBox" if rank < 4 else "KickBox"
    condition = CONDITIONS[rank % 4]
    row = next(
        value
        for value in rows
        if value["split"] == "test"
        and value["task"] == task
        and int(value["source_motion_id"]) == 9
    )
    prompt_row = row
    prompt_key = "prompt_latents"
    prompt_paths = list(row["prompt"]["frame_paths"])
    prompt_reversed = False
    if condition == "reversed":
        prompt_key = "reversed_prompt_latents"
        prompt_paths.reverse()
        prompt_reversed = True
    elif condition == "same_task_alternate":
        prompt_row = next(
            value
            for value in rows
            if value["split"] == "test"
            and value["task"] == task
            and int(value["source_motion_id"]) == 19
        )
        prompt_paths = list(prompt_row["prompt"]["frame_paths"])
    elif condition == "wrong_task":
        wrong_task = "KickBox" if task == "CarryBox" else "CarryBox"
        prompt_row = next(
            value
            for value in rows
            if value["split"] == "test"
            and value["task"] == wrong_task
            and int(value["source_motion_id"]) == 9
        )
        prompt_paths = list(prompt_row["prompt"]["frame_paths"])
    prompt_payload = load_validated_latent_payload(
        payload_path(cache_root, prompt_row),
        split=str(prompt_row["split"]),
        task=str(prompt_row["task"]),
        source_motion_id=int(prompt_row["source_motion_id"]),
    )
    return (
        row,
        prompt_row,
        condition,
        prompt_payload[prompt_key],
        prompt_paths,
        prompt_reversed,
    )


def encode_video(
    frames_dir: Path, output: Path, ffmpeg: Path, fps: int = 10
) -> None:
    command = [
        str(ffmpeg),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%03d.png"),
        "-frames:v",
        "8",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_case(model, config, args, device, output_dir, rank):
    """Render one logical case, shared by serial and distributed backends."""
    rows = read_jsonl(config.resolved(config.manifest))
    cache_root = config.resolved(config.latent_cache)
    (
        row,
        prompt_row,
        condition,
        prompt_latents,
        prompt_paths,
        prompt_reversed,
    ) = selected_case(rank, rows, cache_root)
    target_payload = load_validated_latent_payload(
        payload_path(cache_root, row),
        split=str(row["split"]),
        task=str(row["task"]),
        source_motion_id=int(row["source_motion_id"]),
    )
    robot = target_payload["robot_latents"].to(device=device, dtype=torch.bfloat16)
    anchor = int(args.anchor)
    chunk_size = config.inference_chunk_size
    env_index = int(row["action_target"]["environment_index"])
    actions = torch.from_numpy(
        load_executed_actions(row["action_target"]["trace_path"], env_index)
    )
    normalizer = ActionNormalizer(cache_root / "ACTION_QUANTILES.json")
    normalized_actions = normalizer(actions).to(device=device, dtype=torch.bfloat16)
    batch = {
        "inference": True,
        "prompt_latents": prompt_latents.to(device=device, dtype=torch.bfloat16).unsqueeze(0),
        "robot_history_latents": robot[:, : anchor + 1].unsqueeze(0),
        "video_target_latents": torch.zeros_like(robot[:, anchor + 1 : anchor + 3]).unsqueeze(0),
        "action_history": normalized_actions[: anchor * 20].unsqueeze(0),
        "action_target": torch.zeros_like(normalized_actions[:40]).unsqueeze(0),
        "video_inference_steps": config.video_inference_steps,
        "action_inference_steps": config.action_inference_steps,
        "video_guidance_scale": config.video_guidance_scale,
        "action_guidance_scale": config.action_guidance_scale,
    }
    # All four prompt interventions for one target task use exactly the same
    # initial flow noise, so visual/action differences are prompt-caused.
    task_noise_group = 0 if rank < 4 else 1
    inference_seed = config.noise_seed + 50_000 + task_noise_group
    torch.manual_seed(inference_seed)
    torch.cuda.manual_seed_all(inference_seed)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(batch)
    predicted_video = prediction["predicted_video_latents"].detach()
    predicted_actions = prediction["predicted_normalized_actions"].detach().float().cpu()[0]
    target_actions = normalized_actions[anchor * 20 : (anchor + 2) * 20].float().cpu()
    normalized_action_mse = float(torch.mean((predicted_actions - target_actions) ** 2).item())
    target_video = robot[:, anchor + 1 : anchor + 3].float().cpu()
    latent_video_mse = float(
        torch.mean((predicted_video[0].float().cpu() - target_video) ** 2).item()
    )

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
    decode_latents = torch.cat(
        [robot[:, : anchor + 1], predicted_video[0]], dim=1
    )
    with torch.inference_mode():
        decoded = vae.decode([decode_latents])[0]
    predicted_rgb = (
        ((decoded[:, -8:].permute(1, 2, 3, 0).float().cpu().numpy() + 1.0) * 127.5)
        .clip(0, 255)
        .astype(np.uint8)
    )
    actual_paths = row["robot_target"]["frame_paths"][
        anchor * 4 + 1 : anchor * 4 + 9
    ]
    actual_rgb = [load_frame(path) for path in actual_paths]
    prompt_indices = np.linspace(0, len(prompt_paths) - 1, 8).round().astype(int)
    prompt_rgb = [load_frame(prompt_paths[index]) for index in prompt_indices]

    render_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not render_job_id.isdigit():
        raise RuntimeError("predictive rendering requires a numeric Slurm job identity")
    # A bounded missing-terminal retry receives a new Slurm job ID and hence a
    # fresh frame directory.  This avoids both deleting evidence and consuming
    # numbered PNGs left by a failed prior process.
    frames_dir = output_dir / (
        f"rank{rank:02d}_{row['task']}_{condition}_job{render_job_id}_frames"
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(8):
        canvas = Image.new("RGB", (960, 352), "black")
        canvas.paste(Image.fromarray(prompt_rgb[frame_index]), (0, 32))
        canvas.paste(Image.fromarray(predicted_rgb[frame_index]), (320, 32))
        canvas.paste(Image.fromarray(actual_rgb[frame_index]), (640, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"prompt: {condition}", fill="white")
        draw.text((328, 8), "generated robot future", fill="white")
        draw.text((648, 8), "held-out ground truth", fill="white")
        canvas.save(frames_dir / f"{frame_index:03d}.png")
    video_path = output_dir / f"{row['task']}_source09_{condition}.mp4"
    encode_video(frames_dir, video_path, args.ffmpeg)
    frame_paths = sorted(frames_dir.glob("*.png"))
    local_result = {
        "rank": rank,
        "render_job_id": render_job_id,
        "target_split": row["split"],
        "task": row["task"],
        "source_motion_id": int(row["source_motion_id"]),
        "condition": condition,
        "prompt_split": prompt_row["split"],
        "prompt_task": prompt_row["task"],
        "prompt_source_motion_id": int(prompt_row["source_motion_id"]),
        "prompt_reversed": prompt_reversed,
        "anchor_latent_transition": anchor,
        "predicted_video_shape": list(predicted_video.shape),
        "predicted_action_shape": list(predicted_actions.shape),
        "inference_seed": inference_seed,
        "latent_video_mse": latent_video_mse,
        "normalized_action_mse": normalized_action_mse,
        "render_frame_count": len(frame_paths),
        "render_frame_names": [path.name for path in frame_paths],
        "video": str(video_path),
    }
    return local_result


def render_summary(complete_cases, config, *, full_parameter_count, checkpoint_step,
                   execution_world_size, checkpoint_binding):
    directional_margins: dict[str, dict[str, dict[str, float]]] = {}
    directional_checks: dict[str, dict[str, dict[str, bool]]] = {}
    for task in ("CarryBox", "KickBox"):
        by_condition = {
            str(record["condition"]): record
            for record in complete_cases
            if record["task"] == task
        }
        matched = by_condition["matched"]
        directional_margins[task] = {}
        directional_checks[task] = {}
        for condition in ("reversed", "same_task_alternate", "wrong_task"):
            directional_margins[task][condition] = {}
            directional_checks[task][condition] = {}
            for metric in ("latent_video_mse", "normalized_action_mse"):
                margin = float(by_condition[condition][metric]) - float(matched[metric])
                directional_margins[task][condition][metric] = margin
                directional_checks[task][condition][metric] = margin > 0.0
    expected_grid = {
        (task, condition)
        for task in ("CarryBox", "KickBox")
        for condition in CONDITIONS
    }
    actual_grid = {
        (str(record["task"]), str(record["condition"]))
        for record in complete_cases
    }
    video_paths = [Path(str(record["video"])) for record in complete_cases]
    expected_prompt_identity = {
        (task, condition): {
            "prompt_task": (
                ("KickBox" if task == "CarryBox" else "CarryBox")
                if condition == "wrong_task"
                else task
            ),
            "prompt_source_motion_id": (
                19 if condition == "same_task_alternate" else 9
            ),
            "prompt_reversed": condition == "reversed",
        }
        for task in ("CarryBox", "KickBox")
        for condition in CONDITIONS
    }
    execution_checks = {
        "exact_two_task_four_condition_grid": (
            len(complete_cases) == 8 and actual_grid == expected_grid
        ),
        "shared_noise_within_each_target_task": all(
            len(
                {
                    int(record["inference_seed"])
                    for record in complete_cases
                    if record["task"] == task
                }
            )
            == 1
            for task in ("CarryBox", "KickBox")
        ),
        "single_numeric_render_job_identity": (
            len({record["render_job_id"] for record in complete_cases}) == 1
            and all(str(record["render_job_id"]).isdigit() for record in complete_cases)
        ),
        "prediction_shapes_exact": all(
            record["predicted_video_shape"] == [1, 48, 2, 20, 20]
            and record["predicted_action_shape"] == [40, 29]
            for record in complete_cases
        ),
        "exact_eight_numbered_render_frames": all(
            int(record["render_frame_count"]) == 8
            and record["render_frame_names"]
            == [f"{index:03d}.png" for index in range(8)]
            for record in complete_cases
        ),
        "frozen_test_source09_anchor08": all(
            record["target_split"] == "test"
            and int(record["source_motion_id"]) == 9
            and int(record["anchor_latent_transition"]) == 8
            for record in complete_cases
        ),
        "exact_prompt_identity_mapping": all(
            record["prompt_split"] == "test"
            and {
                key: record[key]
                for key in (
                    "prompt_task",
                    "prompt_source_motion_id",
                    "prompt_reversed",
                )
            }
            == expected_prompt_identity[(record["task"], record["condition"])]
            for record in complete_cases
        ),
        "eight_unique_nonempty_h264_videos": (
            len(video_paths) == len(set(video_paths)) == 8
            and all(path.is_file() and path.stat().st_size > 0 for path in video_paths)
        ),
    }
    result = {
        "protocol": "paper_zero_wam_heldout_openloop_render_v1",
        "execution_completed": all(execution_checks.values()),
        "execution_checks": execution_checks,
        "architecture_parameter_count": full_parameter_count,
        "checkpoint_step": checkpoint_step,
        "execution_world_size": execution_world_size,
        "checkpoint_binding": checkpoint_binding,
        "inference": {
            "chunk_size": config.inference_chunk_size,
            "video_guidance_scale": config.video_guidance_scale,
            "action_guidance_scale": config.action_guidance_scale,
            "flow_integrator": config.flow_integrator,
            "video_steps": config.video_inference_steps,
            "action_steps": config.action_inference_steps,
            "video_snr_shift": config.video_snr_shift,
            "action_snr_shift": config.action_snr_shift,
            "paper_reported": [
                "chunk_size",
                "video_guidance_scale",
                "action_guidance_scale",
            ],
            "released_causal_va_values": [
                "flow_integrator",
                "video_steps",
                "action_steps",
                "video_snr_shift",
                "action_snr_shift",
            ],
        },
        "cases": complete_cases,
        "directional_margins_vs_matched": directional_margins,
        "directional_checks": directional_checks,
        "all_single_anchor_directional_checks_passed": all(
            passed
            for task_checks in directional_checks.values()
            for condition_checks in task_checks.values()
            for passed in condition_checks.values()
        ),
        "hash_checks": False,
        "claim_boundary": "Predictive held-out visualization; not a closed-loop physical success claim.",
    }
    if checkpoint_step == 700:
        result.update(
            protocol="paper_zero_wam_step700_interim_openloop_render_v1",
            evaluation_scope="user_requested_step700_interim",
            formal_execution_complete=False,
            physical_continuation_allowed=False,
            automatic_training_continuation=False,
            claim_boundary="Step-700 predictive visualization; not closed-loop physical rollout or formal completion.",
        )
    return result


def verify_decodable_render_cases(cases, ffmpeg):
    for record in cases:
        # Decode at most nine frames: a truncated or extra-frame clip cannot
        # satisfy the exact eight-frame RGB byte count. This is CPU-only.
        decoded = subprocess.run([
            str(ffmpeg), "-v", "error", "-xerror", "-i", str(record["video"]),
            "-map", "0:v:0", "-frames:v", "9", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(decoded.stdout) != 8 * 960 * 352 * 3:
            raise ValueError("render clip does not decode to exactly eight complete RGB frames")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--interim-step", type=int, choices=[700])
    parser.add_argument("--anchor", type=int, default=8)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(
            "/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/"
            "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
        ),
    )
    args = parser.parse_args()
    expected_step, _ = evaluation_scope(args.single_gpu, args.interim_step)
    if args.interim_step is not None and args.output_dir.name != "heldout_openloop_videos_step700":
        raise ValueError("interim rendering must use the isolated heldout_openloop_videos_step700 directory")
    if args.anchor != 8:
        raise ValueError("the frozen predictive-render anchor is latent transition 8")
    config = PaperZeroWAMConfig()
    rank, device = setup(args.single_gpu)
    output_dir = args.output_dir.resolve()
    result: dict[str, Any] | None = None
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    checkpoint_state = json.loads((args.checkpoint / "STATE.json").read_text(encoding="utf-8"))
    checkpoint_binding = dict(directory=str(args.checkpoint.resolve()), state=checkpoint_state)
    if args.single_gpu:
        from .train_single_gpu import single_gpu_checkpoint_step
        if single_gpu_checkpoint_step(checkpoint_state, config) != expected_step:
            raise ValueError(f"predictive rendering requires exact checkpoint step {expected_step}")
    if args.single_gpu and (output_dir / "RENDER_RESULT.json").exists():
        existing = json.loads((output_dir / "RENDER_RESULT.json").read_text())
        recomputed = render_summary(existing["cases"], config,
            full_parameter_count=config.expected_parameter_count, checkpoint_step=expected_step,
            execution_world_size=1, checkpoint_binding=checkpoint_binding)
        if not recomputed["execution_completed"] or existing != recomputed:
            raise ValueError("render terminal differs from frozen cases/checkpoint evidence")
        verify_decodable_render_cases(existing["cases"], args.ffmpeg)
        dist.destroy_process_group()
        emit_json_best_effort({"render_terminal_reused": True, "model_inference_calls_added": 0})
        return

    model = PaperZeroWAM.from_wan_pretrained(config, dtype=torch.float32)
    full_parameter_count = model.parameter_count
    if full_parameter_count != config.expected_parameter_count:
        raise RuntimeError("render model parameter contract changed")
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({PaperMoTLayer, IFPHead}),
        sharding_strategy=(ShardingStrategy.NO_SHARD if args.single_gpu else ShardingStrategy.FULL_SHARD),
        mixed_precision=MixedPrecision(
            param_dtype=None, reduce_dtype=torch.bfloat16, buffer_dtype=None
        ),
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    if args.single_gpu:
        from .train_single_gpu import single_gpu_checkpoint_step
        checkpoint_step = single_gpu_checkpoint_step(checkpoint_state, config)
    else:
        checkpoint_step = validated_checkpoint_step(checkpoint_state, "formal", config)
    model_payload = torch.load(
        args.checkpoint / f"model_rank{rank:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint_step != expected_step
        or int(model_payload.get("step", -1)) != checkpoint_step
    ):
        raise ValueError(f"predictive rendering requires exact checkpoint step {expected_step}")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT if args.single_gpu else StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(model_payload["model"], strict=True)
    del model_payload
    dist.barrier()

    if args.single_gpu:
        gathered = [render_case(model, config, args, device, output_dir, slot) for slot in range(8)]
    else:
        local_result = render_case(model, config, args, device, output_dir, rank)
        gathered: list[dict[str, Any] | None] | None = [None] * 8 if rank == 0 else None
        dist.gather_object(local_result, gathered, dst=0)
    if rank == 0:
        assert gathered is not None and all(record is not None for record in gathered)
        complete_cases = [record for record in gathered if record is not None]
        result = render_summary(complete_cases, config, full_parameter_count=full_parameter_count,
                                checkpoint_step=checkpoint_step, execution_world_size=dist.get_world_size(),
                                checkpoint_binding=checkpoint_binding)
    execution_ok = torch.tensor(
        int(rank == 0 and result is not None and result["execution_completed"]),
        device=device,
    )
    dist.broadcast(execution_ok, src=0)
    if not bool(execution_ok.item()):
        raise RuntimeError("predictive render execution contract failed")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        assert result is not None
        # Videos and every distributed rank are complete before the terminal
        # suppresses the bounded missing-result recovery path.
        write_json_atomic(output_dir / "RENDER_RESULT.json", result)
        refresh_results_document_best_effort()


if __name__ == "__main__":
    main()
