#!/usr/bin/env python3
"""Run paired prompt-conditioned paper Zero-WAM rollouts in live PhysX."""

from __future__ import annotations

import atexit
import argparse
from datetime import timedelta
import json
import math
import os
import subprocess
import sys
import time
import types
from multiprocessing.connection import Client
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


AUTHKEY = b"paper-zero-wam-smallbox-v1"
CONDITIONS = tuple("CarryBox" if rank % 2 == 0 else "KickBox" for rank in range(8))
SMALLBOX_PROMPT_SOURCES = {"CarryBox": 45, "KickBox": 21}
SMALLBOX_PROMPT_SPECS = [
    {
        "split": "train",
        "task": task,
        "source_motion_id": source_motion_id,
        "latent_key": "prompt_latents",
    }
    for task, source_motion_id in SMALLBOX_PROMPT_SOURCES.items()
]
ROLLOUT_STEPS = 650
ACTION_CHUNK = 40
RGB_STRIDE = 5


def setup(single_gpu: bool = False) -> tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != (1 if single_gpu else 8):
        raise ValueError("physical execution rank count differs from the selected backend")
    if not os.environ.get("SLURM_STEP_ID"):
        raise ValueError("physical evaluation requires a real retained compute step")
    torch.cuda.set_device(local_rank)
    # Rank zero runs the exact released Generator+Tracker endpoint baselines
    # sequentially while the other seven FSDP ranks wait at a batch boundary.
    # Camera-enabled 650-step Isaac routes legitimately exceed PyTorch's
    # default process-group timeout; the Slurm stage wall time and terminal
    # trace contract remain the actual fail-closed bounds.
    dist.init_process_group("nccl", timeout=timedelta(hours=12))
    return rank, world_size, torch.device("cuda", local_rank)


def start_simulator(
    address: Path,
    seed: int,
    output_dir: Path,
    copies_per_profile: int = 2,
    route: str = "model",
) -> tuple[subprocess.Popen[bytes], Any, dict[str, Any]]:
    python = Path("/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python")
    server_module = "scripts.sugar.demo_following.paper_zero_wam.smallbox_sim_server"
    environment = dict(os.environ)
    # The server is Python 3.11 and must use its own Isaac/Torch packages; the
    # controller's Python-3.10 site-packages must never leak across this boundary.
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    route_label = route.lower()
    log_path = output_dir / f"SIMULATOR_{route_label}.log"
    error_path = output_dir / f"SIMULATOR_{route_label}.err"
    last_error = "simulator did not start"
    for attempt in range(1, 6):
        if address.exists():
            address.unlink()
        log_stream = log_path.open("ab")
        error_stream = error_path.open("ab")
        command = [
            str(python),
            "-u",
            "-m",
            server_module,
            "--address",
            str(address),
            "--seed",
            str(seed),
            "--num-envs",
            "8",
            "--copies-per-profile",
            str(copies_per_profile),
            "--route",
            route,
            "--headless",
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_stream,
            stderr=error_stream,
        )
        connection = None
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                last_error = f"simulator attempt {attempt} exited {process.returncode}"
                break
            if address.exists():
                try:
                    connection = Client(str(address), family="AF_UNIX", authkey=AUTHKEY)
                    break
                except (ConnectionRefusedError, FileNotFoundError, EOFError):
                    pass
            time.sleep(1.0)
        log_stream.close()
        error_stream.close()
        if connection is not None:
            ready = connection.recv()
            if ready.get("status") != "ready":
                process.terminate()
                raise RuntimeError(f"simulator returned invalid readiness: {ready}")
            return process, connection, ready
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        time.sleep(2.0)
    raise RuntimeError(last_error)


def stop_simulator(process: subprocess.Popen[bytes], connection: Any) -> None:
    """Best-effort cleanup for failures and scheduler termination."""

    try:
        connection.close()
    except Exception:
        pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def run_endpoint_baseline(
    address: Path,
    seed: int,
    output_dir: Path,
    route: str,
    copies_per_profile: int = 2,
    restore_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run unique released-endpoint profiles and retain their full traces."""

    process, connection, ready = start_simulator(
        address,
        seed,
        output_dir,
        copies_per_profile=copies_per_profile,
        route=route,
    )
    try:
        if ready.get("route") != route:
            raise RuntimeError(f"endpoint server route mismatch: {ready.get('route')}")
        if restore_payload is None:
            raise ValueError("released endpoint requires the adapted restore payload")
        if set(restore_payload) != set(RESTORE_PAYLOAD_KEYS):
            raise ValueError("adapted restore payload topology changed")
        connection.send(
            {"command": "restore_profiles", "restore_payload": restore_payload}
        )
        restored = connection.recv()
        if restored.get("status") != "restored" or restored.get("restore_exact") is not True:
            raise RuntimeError(f"endpoint restore response invalid: {restored}")
        ready = {**ready, **restored}
        chunks: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "robot_root_state_w",
                "robot_joint_pos",
                "robot_joint_vel",
                "object_root_state_w",
                "contact",
                "requested_action",
                "executed_action",
                "done",
            )
        }
        executed_steps = 0
        while executed_steps < ROLLOUT_STEPS:
            count = min(ACTION_CHUNK, ROLLOUT_STEPS - executed_steps)
            connection.send({"command": "step_endpoint_chunk", "count": count})
            response = connection.recv()
            if response.get("status") == "error":
                raise RuntimeError(response["error"])
            for name in chunks:
                chunks[name].append(np.asarray(response[name]))
            executed_steps += count
        connection.send({"command": "close"})
        closed = connection.recv()
        if closed.get("status") != "closed":
            raise RuntimeError(f"endpoint simulator close response invalid: {closed}")
        connection.close()
        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise RuntimeError(f"endpoint simulator exited with status {return_code}")
    except Exception:
        stop_simulator(process, connection)
        raise

    # The simulator creates two exact copies of each of four random profiles.
    # Retain one source row per profile rather than counting paired copies as
    # extra trials.
    selected = np.arange(0, 8, copies_per_profile, dtype=np.int64)
    profile_count = int(selected.shape[0])
    trace = {
        name: np.concatenate(values, axis=0)[:, selected]
        for name, values in chunks.items()
    }
    initial_state = {
        name: np.asarray(value)[selected]
        for name, value in ready["initial_state"].items()
    }
    initial_rgb = np.asarray(ready["initial_rgb"])[selected]
    if any(
        value.shape[0] != ROLLOUT_STEPS or value.shape[1] != profile_count
        for value in trace.values()
    ):
        raise RuntimeError(
            "released endpoint trace did not contain the expected 650-step routes"
        )
    return {
        "route": route,
        "endpoint_files": ready["endpoint_files"],
        "trace": trace,
        "initial_state": initial_state,
        "initial_rgb": initial_rgb,
        "profile_result": ready["profile_result"],
        "restore_exact": ready["restore_exact"],
        "tracker_observation_history_reinitialized": ready[
            "tracker_observation_history_reinitialized"
        ],
        "restore_payload": ready["restore_payload"],
    }


def broadcast_rgb(
    rank: int, device: torch.device, frames: np.ndarray | None
) -> torch.Tensor:
    batch = torch.empty((8, 320, 320, 3), dtype=torch.uint8, device=device)
    if rank == 0:
        if frames is None or frames.shape != (8, 320, 320, 3):
            raise ValueError(f"broadcast RGB geometry mismatch: {None if frames is None else frames.shape}")
        batch.copy_(torch.from_numpy(frames).to(device))
    dist.broadcast(batch, src=0)
    return batch[rank].clone()


def gather_tensor(rank: int, value: torch.Tensor) -> torch.Tensor | None:
    gathered = [torch.empty_like(value) for _ in range(8)] if rank == 0 else None
    dist.gather(value.contiguous(), gather_list=gathered, dst=0)
    return torch.stack(gathered) if gathered is not None else None


def broadcast_executed_actions(
    rank: int, device: torch.device, value: np.ndarray | None, count: int
) -> torch.Tensor:
    batch = torch.empty((8, count, 29), dtype=torch.float32, device=device)
    if rank == 0:
        if value is None or value.shape != (count, 8, 29):
            raise ValueError("executed-action response geometry mismatch")
        batch.copy_(torch.from_numpy(value).permute(1, 0, 2).to(device))
    dist.broadcast(batch, src=0)
    return batch[rank].clone()


def encode_history(vae: Any, frames: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    video = torch.stack(frames).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    with torch.inference_mode():
        latents = vae.encode([video.to(device)])[0].to(torch.bfloat16)
    expected = 1 + (len(frames) - 1) // 4
    if tuple(latents.shape) != (48, expected, 20, 20):
        raise RuntimeError(
            f"online causal VAE geometry mismatch: frames={len(frames)} latents={tuple(latents.shape)}"
        )
    return latents


class CausalRolloutBatch:
    """Eight physical cases; either eight workers or one serial GPU worker.

    Only deployment history and cached prompts enter the unchanged full model.
    The simulator advances all eight cases together after every case has an
    action chunk. Serial inference therefore cannot advance another case's clock.
    """

    def __init__(self, *, model, vae, normalizer, config, rank, device,
                 prompts, initial_rgb, single_gpu):
        self.model, self.vae, self.normalizer = model, vae, normalizer
        self.config, self.rank, self.device = config, rank, device
        self.single_gpu = single_gpu
        self.slots = list(range(8)) if single_gpu else [rank]
        if set(prompts) != set(self.slots):
            raise ValueError("cached prompt slots do not cover exactly this worker's cases")
        self.prompts = prompts
        frames = self._frames(initial_rgb)
        self.histories = {slot: [frames[slot]] for slot in self.slots}
        self.actions = {slot: torch.empty((0, 29), dtype=torch.bfloat16, device=device)
                        for slot in self.slots}

    def _frames(self, frames):
        if self.single_gpu:
            if frames is None or frames.shape != (8, 320, 320, 3):
                raise ValueError("serial RGB response must preserve all eight cases")
            values = torch.from_numpy(frames).to(self.device)
            return {slot: values[slot].clone() for slot in self.slots}
        return {self.rank: broadcast_rgb(self.rank, self.device, frames)}

    def predict(self, inference_seeds):
        if len(inference_seeds) != 8:
            raise ValueError("all eight predeclared inference seeds are required")
        videos, actions = [], []
        for slot in self.slots:
            robot_history = encode_history(self.vae, self.histories[slot], self.device)
            batch = {
                "inference": True,
                "prompt_latents": self.prompts[slot].unsqueeze(0),
                "robot_history_latents": robot_history.unsqueeze(0),
                "video_target_latents": torch.zeros(
                    (1, 48, 2, 20, 20), dtype=torch.bfloat16, device=self.device),
                "action_history": self.actions[slot].unsqueeze(0),
                "action_target": torch.zeros(
                    (1, ACTION_CHUNK, 29), dtype=torch.bfloat16, device=self.device),
                "video_inference_steps": self.config.video_inference_steps,
                "action_inference_steps": self.config.action_inference_steps,
                "video_guidance_scale": self.config.video_guidance_scale,
                "action_guidance_scale": self.config.action_guidance_scale,
            }
            torch.manual_seed(inference_seeds[slot])
            torch.cuda.manual_seed_all(inference_seeds[slot])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = self.model(batch)
            videos.append(prediction["predicted_video_latents"][0].detach())
            actions.append(self.normalizer.invert(
                prediction["predicted_normalized_actions"][0].detach().float()))
        if self.single_gpu:
            return torch.stack(videos), torch.stack(actions)
        return gather_tensor(self.rank, videos[0]), gather_tensor(self.rank, actions[0])

    def append_response(self, response, execute_count):
        if self.single_gpu:
            values = np.asarray(response["executed_action"])
            if values.shape != (execute_count, 8, 29):
                raise ValueError("serial executed-action response must preserve all eight cases")
            values = torch.from_numpy(values).permute(1, 0, 2).to(self.device)
            executed = {slot: values[slot] for slot in self.slots}
        else:
            executed = {self.rank: broadcast_executed_actions(
                self.rank, self.device,
                None if response is None else np.asarray(response["executed_action"]),
                execute_count)}
        for slot in self.slots:
            self.actions[slot] = torch.cat([
                self.actions[slot],
                self.normalizer.deployed_history(executed[slot]).to(torch.bfloat16),
            ], dim=0)
        for frame_index in range(execute_count // RGB_STRIDE):
            frames = self._frames(None if response is None else np.asarray(response["rgb"])[frame_index])
            for slot in self.slots:
                self.histories[slot].append(frames[slot])


def load_frame(path_value: str) -> np.ndarray:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def encode_pair_video(
    profile_id: int,
    carry_frames: list[np.ndarray],
    kick_frames: list[np.ndarray],
    carry_prompt_paths: list[str],
    kick_prompt_paths: list[str],
    output: Path,
    ffmpeg: Path,
) -> int:
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x352",
            "-r",
            "10",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    frame_count = min(len(carry_frames), len(kick_frames))
    for index in range(frame_count):
        prompt_index = int(round(index * 63 / max(1, frame_count - 1)))
        panels = (
            load_frame(carry_prompt_paths[prompt_index]),
            carry_frames[index],
            load_frame(kick_prompt_paths[prompt_index]),
            kick_frames[index],
        )
        canvas = Image.new("RGB", (1280, 352), "black")
        for panel_index, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (panel_index * 320, 32))
        draw = ImageDraw.Draw(canvas)
        labels = (
            "Carry selected demo",
            f"Carry-prompt PhysX profile {profile_id}",
            "Kick selected demo",
            f"Kick-prompt PhysX profile {profile_id}",
        )
        for panel_index, label in enumerate(labels):
            draw.text((panel_index * 320 + 8, 8), label, fill="white")
        process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for profile {profile_id}: {stderr[-2000:]}")
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile-batch", type=int, choices=range(5), required=True)
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
    rank, world_size, device = setup(args.single_gpu)
    batch_dir = args.output_root.resolve() / f"profile_batch{args.profile_batch:02d}"
    existing: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    terminal_state_value = 0
    if rank == 0:
        batch_dir.mkdir(parents=True, exist_ok=True)
        result_path = batch_dir / "BATCH_RESULT.json"
        if result_path.is_file():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                terminal_state_value = (
                    1
                    if existing.get("protocol")
                    == "paper_zero_wam_smallbox_physical_batch_v1"
                    and int(existing.get("profile_batch", -1)) == args.profile_batch
                    and existing.get("passed_execution_contract") is True
                    and int(existing.get("checkpoint_step", -1)) == 4200
                    and int(existing.get("architecture_parameter_count", -1))
                    == config.expected_parameter_count
                    and int(existing.get("adapted_rollout_count", -1)) == 8
                    and int(existing.get("released_endpoint_rollout_count", -1)) == 8
                    and int(existing.get("rollout_count", -1)) == 16
                    and existing.get("video_frame_counts") == [131] * 4
                    and existing.get("prompt_specs") == SMALLBOX_PROMPT_SPECS
                    and existing.get("hash_checks") is False
                    and set(
                        (existing.get("released_endpoint_tracker_history_reinitialized") or {})
                    ) == {"CarryBox", "KickBox"}
                    and all(
                        (existing.get("released_endpoint_tracker_history_reinitialized") or {}).values()
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
        raise ValueError("existing SMALLBOX terminal result does not match this batch")
    if int(terminal_result.item()) == 1:
        if rank == 0:
            assert existing is not None
            emit_json_best_effort(
                {
                    "profile_batch": args.profile_batch,
                    "terminal_result_reused": True,
                    "passed_execution_contract": existing.get(
                        "passed_execution_contract"
                    ),
                }
            )
        dist.barrier()
        dist.destroy_process_group()
        return
    dist.barrier()

    model = PaperZeroWAM.from_wan_pretrained(config, dtype=torch.float32)
    full_parameter_count = model.parameter_count
    if full_parameter_count != config.expected_parameter_count:
        raise RuntimeError("SMALLBOX model parameter contract changed")
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
        args.checkpoint / f"model_rank{rank:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint_step != config.optimizer_steps
        or int(checkpoint.get("step", -1)) != checkpoint_step
    ):
        raise ValueError("SMALLBOX requires the complete step-4200 checkpoint")
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
    rows = read_jsonl(config.resolved(config.manifest))
    prompt_rows = {
        "CarryBox": next(
            row for row in rows
            if row["split"] == "train" and row["task"] == "CarryBox"
            and int(row["source_motion_id"])
            == SMALLBOX_PROMPT_SOURCES["CarryBox"]
        ),
        "KickBox": next(
            row for row in rows
            if row["split"] == "train" and row["task"] == "KickBox"
            and int(row["source_motion_id"])
            == SMALLBOX_PROMPT_SOURCES["KickBox"]
        ),
    }
    cache_root = config.resolved(config.latent_cache)
    prompt_cache = {}
    prompts = {}
    for slot in (range(8) if args.single_gpu else [rank]):
        condition = CONDITIONS[slot]
        if condition not in prompt_cache:
            payload = load_validated_latent_payload(
                cache_root / "train" / condition
                / f'{int(prompt_rows[condition]["source_motion_id"]):03d}.pt',
                split="train", task=condition,
                source_motion_id=int(prompt_rows[condition]["source_motion_id"]),
            )
            prompt_cache[condition] = payload["prompt_latents"].to(device=device, dtype=torch.bfloat16)
        prompts[slot] = prompt_cache[condition]
    normalizer = ActionNormalizer(cache_root / "ACTION_QUANTILES.json")

    process = None
    connection = None
    cleanup_registered = False
    ready = None
    if rank == 0:
        address = Path(
            f"/tmp/paper_zero_wam_{os.environ.get('SLURM_JOB_ID', 'local')}_"
            f"{args.profile_batch}.sock"
        )
        process, connection, ready = start_simulator(
            address,
            config.noise_seed + 70_000 + args.profile_batch,
            batch_dir,
        )
        atexit.register(stop_simulator, process, connection)
        cleanup_registered = True
    runtime = CausalRolloutBatch(
        model=model, vae=vae, normalizer=normalizer, config=config, rank=rank, device=device,
        prompts=prompts, initial_rgb=None if ready is None else np.asarray(ready["initial_rgb"]),
        single_gpu=args.single_gpu)
    all_rgb: list[np.ndarray] | None = [] if rank == 0 else None
    if rank == 0:
        assert ready is not None and all_rgb is not None
        all_rgb.append(np.asarray(ready["initial_rgb"]).copy())
    trace_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "robot_root_state_w",
            "robot_joint_pos",
            "robot_joint_vel",
            "object_root_state_w",
            "contact",
            "requested_action",
            "executed_action",
            "done",
        )
    }
    future_pair_differences: list[list[float]] = []
    action_pair_differences: list[list[float]] = []

    executed_steps = 0
    chunk_index = 0
    while executed_steps < ROLLOUT_STEPS:
        gathered_video, gathered_actions = runtime.predict([
            config.noise_seed + 80_000 + (args.profile_batch * 4 + slot // 2) * 101 + chunk_index
            for slot in range(8)
        ])
        execute_count = min(ACTION_CHUNK, ROLLOUT_STEPS - executed_steps)
        response = None
        if rank == 0:
            assert connection is not None and gathered_video is not None and gathered_actions is not None
            future_pair_differences.append(
                [
                    float((gathered_video[even] - gathered_video[even + 1]).abs().max())
                    for even in range(0, 8, 2)
                ]
            )
            action_pair_differences.append(
                [
                    float((gathered_actions[even] - gathered_actions[even + 1]).abs().max())
                    for even in range(0, 8, 2)
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
            for frame_batch in np.asarray(response["rgb"]):
                all_rgb.append(frame_batch.copy())

        runtime.append_response(response, execute_count)
        executed_steps += execute_count
        chunk_index += 1
        if rank == 0:
            emit_json_best_effort(
                {
                    "profile_batch": args.profile_batch,
                    "executed_steps": executed_steps,
                    "rollout_steps": ROLLOUT_STEPS,
                }
            )

    if rank == 0:
        assert connection is not None and process is not None and all_rgb is not None
        connection.send({"command": "close"})
        closed = connection.recv()
        if closed.get("status") != "closed":
            raise RuntimeError(f"simulator close response invalid: {closed}")
        connection.close()
        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise RuntimeError(f"simulator exited with status {return_code}")
        if cleanup_registered:
            atexit.unregister(stop_simulator)

        trace = {name: np.concatenate(values, axis=0) for name, values in trace_chunks.items()}
        if any(value.shape[0] != ROLLOUT_STEPS for value in trace.values()):
            raise RuntimeError("physical trace did not contain exactly 650 transitions")
        initial_state = {
            f"initial_{name}": np.asarray(value)
            for name, value in ready["initial_state"].items()
        }
        restore_state = {
            f"restore_{name}": np.asarray(value)
            for name, value in ready["restore_payload"].items()
        }
        if any(value.shape[0] != 8 for value in initial_state.values()) or not all(
            np.isfinite(value).all() for value in initial_state.values()
        ):
            raise RuntimeError("initial physical state readback is incomplete or non-finite")
        trace_path = batch_dir / "TRACE.npz"
        np.savez_compressed(trace_path, **initial_state, **restore_state, **trace)

        endpoint_seed = config.noise_seed + 70_000 + args.profile_batch
        endpoint_runs = {
            task: run_endpoint_baseline(
                address.with_name(f"{address.stem}_{task.lower()}_endpoint.sock"),
                endpoint_seed,
                batch_dir,
                f"{task}_endpoint",
                restore_payload=ready["restore_payload"],
            )
            for task in ("CarryBox", "KickBox")
        }
        endpoint_archive: dict[str, np.ndarray] = {}
        adapted_profile_sources = np.asarray([0, 2, 4, 6], dtype=np.int64)
        endpoint_initial_state_exact: dict[str, bool] = {}
        endpoint_initial_rgb_exact: dict[str, bool] = {}
        endpoint_profile_readback_exact: dict[str, bool] = {}
        endpoint_observation_history_exact: dict[str, bool] = {}
        for task, endpoint_run in endpoint_runs.items():
            prefix = task.lower()
            for name, value in endpoint_run["trace"].items():
                endpoint_archive[f"{prefix}_{name}"] = value
            for name, value in endpoint_run["initial_state"].items():
                endpoint_archive[f"{prefix}_initial_{name}"] = value
            for name, value in endpoint_run["restore_payload"].items():
                endpoint_archive[f"{prefix}_restore_{name}"] = value
            endpoint_initial_state_exact[task] = all(
                np.array_equal(
                    np.asarray(ready["initial_state"][name])[adapted_profile_sources],
                    value,
                )
                for name, value in endpoint_run["initial_state"].items()
            )
            endpoint_initial_rgb_exact[task] = np.array_equal(
                np.asarray(ready["initial_rgb"])[adapted_profile_sources],
                endpoint_run["initial_rgb"],
            )
            endpoint_profile_readback_exact[task] = (
                endpoint_run.get("restore_exact") is True
                and all(
                    np.array_equal(
                        np.asarray(ready["restore_payload"][name]), value
                    )
                    for name, value in endpoint_run["restore_payload"].items()
                )
            )
            endpoint_observation_history_exact[task] = all(
                endpoint_run["tracker_observation_history_reinitialized"].values()
            )
        endpoint_trace_path = batch_dir / "ENDPOINT_TRACES.npz"
        np.savez_compressed(endpoint_trace_path, **endpoint_archive)
        rollout_records: list[dict[str, Any]] = []
        video_paths: list[str] = []
        video_frame_counts: list[int] = []
        for env_index in range(8):
            profile_id = args.profile_batch * 4 + env_index // 2
            summary = physical_summary(
                trace["robot_root_state_w"][:, env_index],
                trace["object_root_state_w"][:, env_index],
                trace["contact"][:, env_index],
            )
            rollout_records.append(
                {
                    "environment_index": env_index,
                    "profile_id": profile_id,
                    "route": "adapted",
                    "prompt_split": "train",
                    "prompt_task": CONDITIONS[env_index],
                    "prompt_source_motion_id": SMALLBOX_PROMPT_SOURCES[
                        CONDITIONS[env_index]
                    ],
                    "source_motion_id": SMALLBOX_PROMPT_SOURCES[CONDITIONS[env_index]],
                    "prompt_latent_key": "prompt_latents",
                    "prompt_reversed": False,
                    "done_count": int(trace["done"][:, env_index].sum()),
                    "physical": summary,
                }
            )
        endpoint_rollout_records: list[dict[str, Any]] = []
        for task, endpoint_run in endpoint_runs.items():
            endpoint_trace = endpoint_run["trace"]
            for profile_offset in range(4):
                profile_id = args.profile_batch * 4 + profile_offset
                endpoint_rollout_records.append(
                    {
                        "environment_index": int(adapted_profile_sources[profile_offset]),
                        "profile_id": profile_id,
                        "route": "released_endpoint",
                        "prompt_split": "train",
                        "prompt_task": task,
                        "prompt_source_motion_id": SMALLBOX_PROMPT_SOURCES[task],
                        "source_motion_id": SMALLBOX_PROMPT_SOURCES[task],
                        "prompt_latent_key": "prompt_latents",
                        "prompt_reversed": False,
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
        for pair_index in range(4):
            profile_id = args.profile_batch * 4 + pair_index
            path = batch_dir / f"profile{profile_id:02d}_carry_vs_kick.mp4"
            carry_frames = [frame_batch[pair_index * 2] for frame_batch in all_rgb]
            kick_frames = [frame_batch[pair_index * 2 + 1] for frame_batch in all_rgb]
            video_frame_counts.append(
                encode_pair_video(
                    profile_id,
                    carry_frames,
                    kick_frames,
                    prompt_rows["CarryBox"]["prompt"]["frame_paths"],
                    prompt_rows["KickBox"]["prompt"]["frame_paths"],
                    path,
                    args.ffmpeg,
                )
            )
            video_paths.append(str(path))
        profile_result = ready["profile_result"] if ready is not None else {}
        checks = {
            "full_650_step_trace": all(value.shape[0] == ROLLOUT_STEPS for value in trace.values()),
            "zero_resets": not bool(trace["done"].any()),
            "paired_initial_physics_and_state": profile_result.get("paired_profiles_passed") is True,
            "all_values_finite": all(
                np.isfinite(value).all()
                for name, value in trace.items()
                if name not in ("done", "contact")
            ),
            "every_requested_action_equals_physx_executed_action": np.array_equal(
                trace["requested_action"], trace["executed_action"]
            ),
            "eight_released_endpoint_traces_present": len(endpoint_rollout_records) == 8
            and all(
                value.shape[0] == ROLLOUT_STEPS and value.shape[1] == 4
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
            "adapted_endpoint_initial_state_exact": all(
                endpoint_initial_state_exact.values()
            ),
            "adapted_endpoint_initial_rgb_exact": all(
                endpoint_initial_rgb_exact.values()
            ),
            "adapted_endpoint_profile_readback_exact": all(
                endpoint_profile_readback_exact.values()
            ),
            "released_endpoint_tracker_history_reinitialized": all(
                endpoint_observation_history_exact.values()
            ),
            "prompt_changes_predicted_future_every_profile": all(
                any(row[pair] > 0.0 for row in future_pair_differences)
                for pair in range(4)
            ),
            "future_precedes_action_change_every_profile": all(
                next(
                    (
                        chunk
                        for chunk, row in enumerate(future_pair_differences)
                        if row[pair] > 0.0
                    ),
                    math.inf,
                )
                <= next(
                    (
                        chunk
                        for chunk, row in enumerate(action_pair_differences)
                        if row[pair] > 1.0e-6
                    ),
                    math.inf,
                )
                < math.inf
                for pair in range(4)
            ),
            "four_h264_rollout_videos": len(video_paths) == 4
            and video_frame_counts == [131] * 4
            and all(Path(path).stat().st_size > 0 for path in video_paths),
        }
        result = {
            "protocol": "paper_zero_wam_smallbox_physical_batch_v1",
            "passed_execution_contract": all(checks.values()),
            "profile_batch": args.profile_batch,
            "profile_ids": list(range(args.profile_batch * 4, args.profile_batch * 4 + 4)),
            "checkpoint_step": checkpoint_step,
            "architecture_parameter_count": full_parameter_count,
            "global_model_batch": 8,
            "execution_world_size": world_size,
            "serial_inference_cases_per_worker": 8 if args.single_gpu else 1,
            "video_chunk_latents": config.inference_chunk_size,
            "action_chunk_steps": ACTION_CHUNK,
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
            "rollout_steps": ROLLOUT_STEPS,
            "adapted_rollout_count": 8,
            "released_endpoint_rollout_count": 8,
            "rollout_count": 16,
            "prompt_specs": SMALLBOX_PROMPT_SPECS,
            "profile_readback": profile_result,
            "released_endpoint_routes": {
                task: endpoint_run["route"] for task, endpoint_run in endpoint_runs.items()
            },
            "released_endpoint_files": {
                task: endpoint_run["endpoint_files"]
                for task, endpoint_run in endpoint_runs.items()
            },
            "adapted_endpoint_initial_state_exact": endpoint_initial_state_exact,
            "adapted_endpoint_initial_rgb_exact": endpoint_initial_rgb_exact,
            "adapted_endpoint_profile_readback_exact": endpoint_profile_readback_exact,
            "released_endpoint_tracker_history_reinitialized": endpoint_observation_history_exact,
            "checks": checks,
            "future_pair_max_abs_by_chunk": future_pair_differences,
            "action_pair_max_abs_by_chunk": action_pair_differences,
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
            "rollouts": [*rollout_records, *endpoint_rollout_records],
            "adapted_trace": str(trace_path),
            "released_endpoint_trace": str(endpoint_trace_path),
            "videos": video_paths,
            "video_frame_counts": video_frame_counts,
            "hash_checks": False,
            "claim_boundary": (
                "Paired live SMALLBOX task-switch evidence for Carry45/Kick21 prompts; "
                "not yet motion-disjoint physical identity following."
            ),
        }
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        assert result is not None
        write_json_atomic(batch_dir / "BATCH_RESULT.json", result)
        emit_json_best_effort(
            {
                "passed_execution_contract": result["passed_execution_contract"],
                "checks": result["checks"],
            }
        )


if __name__ == "__main__":
    main()
