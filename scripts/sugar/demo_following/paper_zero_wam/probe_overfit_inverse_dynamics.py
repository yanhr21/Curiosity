"""Endpoint-only diagnosis using the unchanged model and official frozen VAE.

Replay saved generated-future actions exactly before substituting the real
future. The real-future arm is an explicitly privileged diagnostic, never a
deployable success result. No optimizer or training update is constructed.
The controls-only mode exports all eight frames of each frozen-VAE control
without loading the world model or regenerating its predictions.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PaperZeroWAMConfig
from .data import ScheduledSamples, read_jsonl
from .model import PaperZeroWAM, inference_sigmas, import_wan_model


def endpoint_config(saved):
    """Restore tuple-valued configuration fields after JSON serialization."""
    restored = dict(saved)
    for field in ("patch_size", "ifp_weights", "ifp_fusion_layers"):
        if field in restored:
            restored[field] = tuple(restored[field])
    config = PaperZeroWAMConfig(**restored)
    config.validate()
    return config


def action_initial_noise(batch, config, seed):
    """Replay exactly the sampler's random draws preceding action integration."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.randn_like(batch["video_target_latents"])
    if config.inactive_action_token_noise:
        torch.randn_like(batch["action_target"])
    return torch.randn_like(batch["action_target"])


@torch.no_grad()
def integrate_action(model, batch, future, initial, config):
    action = initial.clone()
    video_time = future.new_zeros((1,))
    sigmas = inference_sigmas(config.action_inference_steps,
                             config.action_snr_shift, device=action.device)
    for index in range(config.action_inference_steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, velocity = model._predict_main_velocities(
                batch, future, action, video_time, sigmas[index:index + 1], False)
        delta = sigmas[index + 1] - sigmas[index]
        action = (action + velocity.to(action.dtype) * delta.to(action.dtype)).to(action.dtype)
    return action


def action_metrics(prediction, target):
    p, t = prediction.float().cpu(), target.float().cpu()
    mse = float((p - t).square().mean())
    baseline = float(t.square().mean())
    scalar_constant_mse = float((t - t.mean()).square().mean())
    joint_constant_mse = float((t - t.mean(dim=1, keepdim=True)).square().mean())
    return {"normalized_mse": mse, "normalized_zero_baseline_mse": baseline,
            "beats_normalized_zero": mse < baseline,
            "scalar_constant_oracle_mse": scalar_constant_mse,
            "predicted_std": float(p.std()), "target_std": float(t.std()),
            "per_joint_constant_oracle_mse": joint_constant_mse,
            "beats_per_joint_constant_oracle": mse < joint_constant_mse,
            "per_joint_mse": (p - t).square().mean(dim=(0, 1)).tolist(),
            "per_joint_target_temporal_std": t.std(dim=1, correction=0).mean(dim=0).tolist(),
            "per_joint_predicted_temporal_std": p.std(dim=1, correction=0).mean(dim=0).tolist(),
            "temporal_delta_mse": float((p.diff(dim=1) - t.diff(dim=1)).square().mean()),
            "target_temporal_delta_energy": float(t.diff(dim=1).square().mean())}


def needs_inverse_probe(metrics):
    """Catch missing action dynamics even when the global zero baseline passes."""
    mse = metrics["normalized_mse"]
    temporal = metrics["temporal_delta_mse"]
    return ((mse > 0 and any(mse >= metrics[key] for key in (
                "normalized_zero_baseline_mse", "scalar_constant_oracle_mse",
                "per_joint_constant_oracle_mse")))
            or (temporal > 0 and temporal >= metrics["target_temporal_delta_energy"]))


def readback_actions(root, output, renders):
    cases = []
    for case in sorted(renders["cases"], key=lambda c: c["slot"]):
        slot, meta = case["slot"], case["meta"]
        name = f"slot{slot:02d}_{meta['task']}_source{meta['source_motion_id']:03d}_anchor{meta['latent_start']:02d}"
        saved = torch.load(root / "training_videos" / f"{name}_predictions.pt",
                           map_location="cpu", weights_only=False)
        metrics = action_metrics(saved["predicted_normalized_actions"],
                                 saved["target_normalized_actions"])
        cases.append({"slot": slot, "case": name, **metrics})
    write_json_atomic(output, {"execution_completed": True, "optimizer_updates": 0,
        "world_model_inference_calls": 0, "cases": cases,
        "inverse_dynamics_probe_needed": any(needs_inverse_probe(c) for c in cases),
        "scope": "actual saved TRAIN sampled actions; oracle means are diagnostics, not deployable baselines; no overall success claim"})


@torch.no_grad()
def render_frozen_controls(root, output, config, device, renders):
    """Decode exactly the saved TRAIN targets, retaining every control frame."""
    import numpy as np
    from PIL import Image, ImageDraw
    from .render_openloop import load_frame

    import_wan_model(config.resolved(config.wan_source))
    from wan.modules.vae2_2 import Wan2_2_VAE

    directory = output.parent / (output.stem + "_frames")
    directory.mkdir(exist_ok=False)
    vae = Wan2_2_VAE(vae_pth=str(config.resolved(config.wan_checkpoint) / "Wan2.2_VAE.pth"),
                    dtype=torch.bfloat16, device=str(device))
    schedule = config.resolved(config.output_root) / "schedule/TRAIN_SCHEDULE.jsonl"
    manifest = {(row["task"], int(row["source_motion_id"])): row
                for row in read_jsonl(config.resolved(config.manifest)) if row["split"] == "train"}
    cases = []
    for slot in range(8):
        dataset = ScheduledSamples(schedule, config.resolved(config.latent_cache),
                                   slot, config, "overfit", 32)
        batch = dataset.sample(0, device)
        meta = batch["meta"]
        name = f"slot{slot:02d}_{meta['task']}_source{meta['source_motion_id']:03d}_anchor{meta['latent_start']:02d}"
        saved = torch.load(root / "training_videos" / f"{name}_predictions.pt",
                           map_location="cpu", weights_only=False)
        if not torch.equal(batch["video_target_latents"].float().cpu(),
                           saved["target_video_latents"].float()):
            raise RuntimeError("saved and currently loaded control targets differ")
        latents = torch.cat([batch["robot_history_latents"][0],
                             batch["video_target_latents"][0]], dim=1)
        decoded = vae.decode([latents])[0]
        rgb = (((decoded[:, -8:].permute(1, 2, 3, 0).float().cpu().numpy() + 1)
                * 127.5).clip(0, 255).astype(np.uint8))
        row = manifest[(meta["task"], meta["source_motion_id"])]
        start = meta["latent_start"] * 4 + 1
        actual = np.stack([load_frame(path) for path in
                           row["robot_target"]["frame_paths"][start:start + 8]])
        if rgb.shape != actual.shape or rgb.shape[0] != 8:
            raise RuntimeError("control decode did not produce the exact eight target frames")
        frame_paths = []
        for index in range(8):
            canvas = Image.new("RGB", (640, 352), "black")
            canvas.paste(Image.fromarray(rgb[index]), (0, 32))
            canvas.paste(Image.fromarray(actual[index]), (320, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"Frozen VAE GT latent | frame {index}", fill="white")
            draw.text((328, 8), "Actual TRAIN ground truth", fill="white")
            path = directory / f"{name}_{index:03d}.png"
            canvas.save(path)
            frame_paths.append(str(path))
        mse = float(np.mean((rgb.astype(np.float32) / 255 - actual.astype(np.float32) / 255) ** 2))
        original = next(c for c in renders["cases"] if c["slot"] == slot)
        cases.append({"slot": slot, "case": name, "control_frame_count": 8,
                      "frames": frame_paths, "rgb_mse_0_1": mse,
                      "original_control_rgb_mse_0_1": original["frozen_vae_reconstruction_rgb_mse_0_1"],
                      "saved_target_latents_exact": True})
        emit_json_best_effort({"frozen_control_case_completed": slot + 1,
                              "rgb_mse_0_1": mse})
        del batch, saved, latents, decoded, dataset
    write_json_atomic(output, {"execution_completed": True, "optimizer_updates": 0,
        "world_model_inference_calls": 0, "cases": cases, "control_frames": 64,
        "scope": "all eight frozen-VAE frames for all eight actual TRAIN targets; visual inspection still required",
        "hash_checks": False})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--vae-controls-only", action="store_true",
                        help="export every frozen-VAE control frame without world-model inference")
    modes.add_argument("--action-readback-only", action="store_true",
                       help="CPU-only metrics on actual saved TRAIN action predictions")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite an endpoint diagnosis")
    root = args.endpoint.resolve()
    result = json.loads((root / "OVERFIT_RESULT.json").read_text())
    renders = json.loads((root / "training_videos/RENDER_RESULT.json").read_text())
    if (not result["execution_completed"] or not renders["execution_completed"]
            or sorted(c["slot"] for c in renders["cases"]) != list(range(8))
            or any(c["target_split"] != "train" for c in renders["cases"])):
        raise RuntimeError("requires a complete actual eight-case TRAIN endpoint")
    if args.action_readback_only:
        readback_actions(root, args.output, renders)
        return
    if not os.environ.get("SLURM_STEP_ID") or not torch.cuda.is_available():
        raise RuntimeError("run on the retained GPU compute step")
    saved = json.loads((root / "CONFIG.json").read_text())["model_and_schedule_config"]
    config = endpoint_config(saved)
    device = torch.device("cuda", 0)
    if args.vae_controls_only:
        render_frozen_controls(root, args.output, config, device, renders)
        return
    model = PaperZeroWAM.from_wan_pretrained(config, dtype=torch.float32)
    payload = torch.load(root / "diagnostic_model_step32.pt", map_location="cpu",
                         weights_only=False)
    if payload["step"] != result["optimizer_steps"]:
        raise RuntimeError("checkpoint and endpoint step differ")
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model = model.to(device).eval()
    schedule = config.resolved(config.output_root) / "schedule/TRAIN_SCHEDULE.jsonl"
    datasets = [ScheduledSamples(schedule, config.resolved(config.latent_cache),
                                 slot, config, "overfit", 32) for slot in range(8)]
    for dataset in datasets[1:]:
        for field in ("_latent_cpu_cache", "_action_cpu_cache", "_action_shard_cpu_cache"):
            setattr(dataset, field, getattr(datasets[0], field))
    cases = []
    for slot, dataset in enumerate(datasets):
        batch = dataset.sample(0, device)
        meta = batch["meta"]
        name = f"slot{slot:02d}_{meta['task']}_source{meta['source_motion_id']:03d}_anchor{meta['latent_start']:02d}"
        tensors = torch.load(root / "training_videos" / f"{name}_predictions.pt",
                             map_location="cpu", weights_only=False)
        target = batch["action_target"]
        if not torch.equal(target.float().cpu(), tensors["target_normalized_actions"].float()):
            raise RuntimeError("saved and currently loaded action targets differ")
        if not torch.equal(batch["video_target_latents"].float().cpu(),
                           tensors["target_video_latents"].float()):
            raise RuntimeError("saved and currently loaded video targets differ")
        deployed = {key: batch[key] for key in
                    ("prompt_latents", "robot_history_latents", "action_history")}
        deployed["video_target_latents"] = torch.zeros_like(batch["video_target_latents"])
        deployed["action_target"] = torch.zeros_like(target)
        initial = action_initial_noise(deployed, config, config.noise_seed + 70_000 + slot)
        generated_future = tensors["predicted_video_latents"].to(
            device=device, dtype=batch["video_target_latents"].dtype)
        replay = integrate_action(model, deployed, generated_future, initial, config)
        exact = torch.equal(replay.float().cpu(), tensors["predicted_normalized_actions"].float())
        if not exact:
            raise RuntimeError(f"saved generated-future action replay differs for {name}; diagnosis invalid")
        oracle = integrate_action(model, deployed, batch["video_target_latents"], initial, config)
        row = {"case": name, "saved_action_replay_exact": exact,
               "generated_future": action_metrics(replay, target),
               "ground_truth_future_oracle": action_metrics(oracle, target),
               "oracle_is_privileged_not_deployable": True}
        cases.append(row)
        emit_json_best_effort(row)
    write_json_atomic(args.output, {"execution_completed": True, "optimizer_updates": 0,
        "checkpoint_step": result["optimizer_steps"], "architecture_parameter_count": model.parameter_count,
        "cases": cases, "generated_future_cases_beating_zero": sum(
            c["generated_future"]["beats_normalized_zero"] for c in cases),
        "oracle_future_cases_beating_zero": sum(
            c["ground_truth_future_oracle"]["beats_normalized_zero"] for c in cases),
        "scope": "privileged inverse-dynamics isolation, not a deployment or physical success",
        "hash_checks": False})


if __name__ == "__main__":
    main()
