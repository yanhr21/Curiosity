"""Endpoint-only inverse-dynamics diagnosis using the unchanged full model.

Replay saved generated-future actions exactly before substituting the real
future. The real-future arm is an explicitly privileged diagnostic, never a
deployable success result. No optimizer or training update is constructed.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PaperZeroWAMConfig
from .data import ScheduledSamples
from .model import PaperZeroWAM, inference_sigmas


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
    return {"normalized_mse": mse, "normalized_zero_baseline_mse": baseline,
            "beats_normalized_zero": mse < baseline,
            "predicted_std": float(p.std()), "target_std": float(t.std()),
            "temporal_delta_mse": float((p.diff(dim=1) - t.diff(dim=1)).square().mean()),
            "target_temporal_delta_energy": float(t.diff(dim=1).square().mean())}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_STEP_ID") or not torch.cuda.is_available():
        raise RuntimeError("run on the retained GPU compute step")
    if args.output.exists():
        raise RuntimeError("refusing to overwrite an endpoint diagnosis")
    root = args.endpoint.resolve()
    saved = json.loads((root / "CONFIG.json").read_text())["model_and_schedule_config"]
    config = PaperZeroWAMConfig(**saved)
    config.validate()
    result = json.loads((root / "OVERFIT_RESULT.json").read_text())
    renders = json.loads((root / "training_videos/RENDER_RESULT.json").read_text())
    if not result["execution_completed"] or len(renders["cases"]) != 8:
        raise RuntimeError("requires a complete actual eight-case TRAIN endpoint")
    device = torch.device("cuda", 0)
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
