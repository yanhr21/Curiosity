"""Bounded optimization diagnostic on the existing full paper reconstruction.

No replacement model or changed loss. Fixed-noise fitting, unseen-noise fitting,
prompt dependence and free generation are reported separately, never conflated.
"""
from __future__ import annotations

import math
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .artifacts import emit_json_best_effort, write_json_atomic
from .data import read_jsonl


DIAGNOSTIC_DIRECTORY = "overfit_debug_fixed_noise_20260909"
REPAIRED_DIRECTORY = "overfit_repaired_fixed_noise_20260910"
RESAMPLED_DIRECTORY = "overfit_resampled_noise_20260911"
LOSS_NAMES = ("video_loss", "action_loss", "ifp_loss")


def resolve_ffmpeg() -> Path:
    """Locate ffmpeg without pinning one absolute site-packages path.

    The original literal pointed into a specific conda env that does not exist
    on every machine, so rendering failed after training had already finished.
    """

    override = os.environ.get("PZW_FFMPEG")
    if override:
        return Path(override)
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    raise RuntimeError(
        "no ffmpeg available; pip install imageio-ffmpeg or set PZW_FFMPEG"
    )


def restored_conditioning_gradient_decision(cross_sums, text_sum, step,
                                            zero_init_action_head=True):
    """Require every restored cross-attention module to carry real gradient.

    With a zero-initialized action output projection, the final action block
    provably receives exactly zero gradient at update zero (its output is
    multiplied by a zero weight), so that one module is whitelisted at step 0.

    Once the head is scaled-init that whitelist is wrong in both directions:
    the module now *does* get gradient, and demanding it be exactly zero makes
    a healthy run fail.  The substantive property -- all 64 modules active, and
    the inherited text projection receiving gradient -- is unchanged.
    """

    final_action = {name for name in cross_sums
                    if ".mot_layers.29." in f".{name}." and name.endswith(".action_block")}
    allowed_zero = final_action if (step == 0 and zero_init_action_head) else set()
    expected_names = len(cross_sums) == 64 and len(final_action) == 1
    finite = all(math.isfinite(value) and value >= 0 for value in cross_sums.values())
    active = all(value > 0 for name, value in cross_sums.items() if name not in allowed_zero)
    expected_zero = all(cross_sums[name] == 0 for name in allowed_zero)
    passed = expected_names and finite and active and expected_zero and math.isfinite(text_sum) and text_sum > 0
    return {"passed": passed, "optimizer_updates_already_applied": step,
            "zero_init_action_head": zero_init_action_head,
            "cross_attention_module_squared_norms": dict(cross_sums),
            "text_embedding_gradient_squared_norm": text_sum,
            "expected_zero_modules": sorted(allowed_zero),
            "reason": ("zero action-output projection blocks the final action block at "
                       "update zero; require all 64 active thereafter"
                       if zero_init_action_head else
                       "scaled-init action head: require all 64 modules active at every step"),
            "hash_checks": False}


def validate_preupdate_recovery(root, config):
    """Only reuse completed initial probes from an interrupted zero-update run."""
    if not config.repaired_conditioning:
        raise ValueError("pre-update recovery is limited to the corrected repeat")
    previous = json.loads((root / "CONFIG.json").read_text())["model_and_schedule_config"]
    if previous != json.loads(json.dumps(config.as_dict())):
        raise ValueError("pre-update recovery configuration differs")
    if (root / "TRAIN_TRACE.jsonl").read_text().strip():
        raise ValueError("cannot restart any already-applied optimizer update")
    for name in ("INITIAL_PROMPT_GATE.json", "UNSEEN_NOISE_INITIAL.json"):
        if not (root / name).is_file():
            raise ValueError("pre-update recovery requires completed initial probes")
    for name in ("TRAIN_EVAL_FORWARD_EQUIVALENCE.json", "RESTORED_MODULE_GRADIENTS.json",
                 "RESTORED_MODULE_GRADIENTS_INITIAL.json", "OVERFIT_RESULT.json", "FITTING_RESULT.json",
                 "diagnostic_model_step32.pt"):
        if (root / name).exists():
            raise ValueError("run reached or passed the first optimizer boundary; cold restart forbidden")


def validate_diagnostic_request(enabled, mode, root, config, resampled=False):
    if not enabled:
        return
    if resampled:
        directory = RESAMPLED_DIRECTORY
    else:
        directory = REPAIRED_DIRECTORY if config.repaired_conditioning else DIAGNOSTIC_DIRECTORY
    expected = config.resolved(config.output_root) / directory
    if mode != "overfit" or root.resolve() != expected.resolve():
        raise ValueError("overfit diagnostic requires its isolated output directory")


def training_noise_seed(config, step, slot, fixed_noise):
    """Per-(step, slot) noise seed, or one frozen seed for the fixed-noise mode.

    ``fixed_noise=True`` repeats each slot's noise/time draws every update.
    This isolates optimizer behavior from resampling but does not test fitting
    across the flow trajectory used by generation. It does not mathematically
    rule out improvement on other draws either. Use resampled training plus
    independent probes and actual sampling to test generative overfit.
    """

    return config.noise_seed + 9_999_991 if fixed_noise else config.noise_seed + step * 8 + slot


def diagnostic_contract(config, resampled=False):
    return {
        "protocol": ("paper_zero_wam_resampled_noise_overfit_v1" if resampled
                     else "paper_zero_wam_user_fixed_noise_overfit_v1"),
        "optimizer_steps": 32, "fixed_training_cases": 8,
        "initialization": "original_Wan_video_and_copied_video_action_blocks_not_step700",
        "architecture_parameter_count": config.expected_parameter_count,
        "model_or_objective_changes": config.repaired_conditioning or not config.ifp_trunk_gradient,
        "objective_changes": not config.ifp_trunk_gradient,
        "ifp_trunk_gradient": config.ifp_trunk_gradient,
        "objective_variant": ("paper_IFP_representation_gradient" if config.ifp_trunk_gradient
                              else "local_IFP_feature_and_history_detach"),
        "restored_official_text_conditioning": config.repaired_conditioning,
        "repaired_prompt_coverage": config.repaired_conditioning,
        "training_noise_seed": training_noise_seed(config, 0, 0, True),
        "unseen_noise_seed": config.noise_seed + 10_000_007,
        "fixed_noise_and_flow_time_every_update": not resampled,
        "noise_and_flow_time_resampled_every_step_and_slot": resampled,
        "scope": ("generative overfit: can the model fit these eight trajectories "
                  "across the whole flow trajectory" if resampled else
                  "optimizer diagnostic at frozen noise/time; this alone "
                  "does not establish fitting across the generative flow trajectory"),
        "endpoint_ratio_limit": 0.5,
        "per_task_reduction_required": True,
        "prompt_dependence_is_separate_from_fitting": True,
        "unseen_noise_is_separate_from_fixed_noise_fitting": True,
        "render_target_split": "train",
        "render_cases": "all_eight_exact_training_source_phase_pairs",
        "render_video_guidance_scale": 1.0,
        "render_guidance_reason": "conditional_only_overfit_has_no_unconditional_training",
        "render_video_steps": config.video_inference_steps,
        "render_action_steps": config.action_inference_steps,
        "automatic_formal_training": False, "automatic_physics": False,
        "hash_checks": False,
    }


@torch.no_grad()
def matched_loss_probe(model, datasets, device, seed, repeats=8):
    """Average each case over ``repeats`` independent (noise, t) draws.

    The original probe seeded once per call, so all eight cases shared a single
    Gaussian sample AND a single flow time.  Under a resampled-noise run the
    per-sample loss varies strongly with t, which made the resulting ratio a
    high-variance point estimate: the same checkpoint scored video 1.2382 at
    t=0.767 and 0.0487 at t=0.963 -- a 25x spread from sampling alone, with one
    "FAIL" and one "PASS".

    Averaging over a fixed, reproducible ladder of draws makes the ratio a
    property of the model rather than of one lucky t.  The draw seeds are
    derived from ``seed`` so the probe stays exactly reproducible.
    """

    previous_training = model.training
    model.eval()
    cases = []
    try:
        for slot, dataset in enumerate(datasets):
            batch = dataset.conditioned_sample(0, device, "matched")
            totals = {key: 0.0 for key in LOSS_NAMES}
            draws = []
            for repeat in range(repeats):
                draw_seed = seed + 1_000_003 * repeat
                torch.manual_seed(draw_seed)
                torch.cuda.manual_seed_all(draw_seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch)
                row = {key: float(loss[key]) for key in LOSS_NAMES}
                draws.append({"draw_seed": draw_seed, **row})
                for key in LOSS_NAMES:
                    totals[key] += row[key]
                del loss
            cases.append({"slot": slot, "meta": batch["meta"],
                          "losses": {key: totals[key] / repeats for key in LOSS_NAMES},
                          "draws": draws})
            del batch
    finally:
        model.train(previous_training)
    return {"seed": seed, "draws_per_case": repeats, "cases": cases,
            "losses": {key: sum(row["losses"][key] for row in cases) / len(cases)
                       for key in LOSS_NAMES},
            "task_losses": {task: {key: sum(row["losses"][key] for row in cases
                                           if row["meta"]["task"] == task) / 4
                                   for key in LOSS_NAMES}
                            for task in ("CarryBox", "KickBox")}}


def reduction_result(initial, final):
    # prompt_gate and matched_loss_probe have intentionally different schemas.
    def values(record):
        if "matched" in record["losses"]:
            return (record["losses"]["matched"],
                    {task: value["matched"] for task, value in record["task_losses"].items()})
        return record["losses"], record["task_losses"]
    first, first_tasks = values(initial)
    last, last_tasks = values(final)
    ratios = {key: last[key] / first[key] for key in LOSS_NAMES}
    task_ratios = {task: {key: last_tasks[task][key] / first_tasks[task][key]
                         for key in LOSS_NAMES} for task in first_tasks}
    passed = all(math.isfinite(value) and value <= 0.5
                 for values_ in [ratios, *task_ratios.values()] for value in values_.values())
    return {"passed": passed, "ratios": ratios, "task_ratios": task_ratios,
            "initial": first, "final": last, "limit": 0.5}


def verify_first_forward(initial, aggregate, draws_per_condition=1):
    """Training and eval forwards must agree before the first update.

    This only holds when the gate used exactly one noise draw and that draw
    matches the fixed training noise.  Once the gate averages over several
    (noise, t) draws -- which it must, to give a low-variance ratio -- the two
    numbers are no longer comparable, so the equivalence is reported as
    inapplicable rather than silently "passing" on a mismatched comparison.
    """

    if draws_per_condition != 1:
        return {"passed": None, "applicable": False,
                "reason": ("gate averages over %d (noise, t) draws while training "
                           "step 0 uses one; the two forwards are not comparable"
                           % draws_per_condition),
                "gate_draws_per_condition": draws_per_condition}
    expected = initial["losses"]["matched"]
    errors = {key: abs(aggregate[key] - expected[key]) for key in LOSS_NAMES}
    if any(error > 1e-5 * max(1.0, abs(expected[key])) for key, error in errors.items()):
        raise RuntimeError(f"fixed-noise training/eval forward mismatch before update: {errors}")
    return {"passed": True, "applicable": True, "absolute_errors": errors,
            "description": "same full-width weights/data/noise before first optimizer update"}


@torch.no_grad()
def render_training_cases(model, datasets, config, device, output_dir):
    """Real 25/50-step generation on TRAIN cases, with no future target input."""
    from .render_openloop import encode_video, load_frame, verify_decodable_render_cases
    from wan.modules.vae2_2 import Wan2_2_VAE

    output_dir.mkdir(parents=True, exist_ok=False)
    ffmpeg = resolve_ffmpeg()
    rows = {(row["task"], int(row["source_motion_id"])): row
            for row in read_jsonl(config.resolved(config.manifest)) if row["split"] == "train"}
    vae = Wan2_2_VAE(vae_pth=os.environ.get("PZW_VAE_PATH") or
                    str(config.resolved(config.wan_checkpoint) / "Wan2.2_VAE.pth"),
                    dtype=torch.bfloat16, device=str(device))
    model.eval()
    cases = []
    for slot, dataset in enumerate(datasets):
        batch = dataset.sample(0, device)
        meta = batch["meta"]
        row = rows[(meta["task"], meta["source_motion_id"])]
        target_video = batch["video_target_latents"].detach().float().cpu()
        target_action = batch["action_target"].detach().float().cpu()
        # Explicitly remove all IFP/future supervision from the deployed input.
        deployed = {key: batch[key] for key in
                    ("prompt_latents", "robot_history_latents", "action_history")}
        deployed.update(inference=True,
                        video_target_latents=torch.zeros_like(batch["video_target_latents"]),
                        action_target=torch.zeros_like(batch["action_target"]),
                        video_inference_steps=config.video_inference_steps,
                        action_inference_steps=config.action_inference_steps,
                        video_guidance_scale=1.0, action_guidance_scale=1.0)
        seed = config.noise_seed + 70_000 + slot
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(deployed)
        predicted_video = prediction["predicted_video_latents"]
        predicted_action = prediction["predicted_normalized_actions"].float().cpu()
        if not all(bool(torch.isfinite(value).all()) for value in prediction.values()):
            raise FloatingPointError("nonfinite training-case generation")
        decode_latents = torch.cat([batch["robot_history_latents"][0], predicted_video[0]], dim=1)
        decoded = vae.decode([decode_latents])[0]
        rgb = (((decoded[:, -8:].permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5)
               .clip(0, 255).astype(np.uint8))
        # A separate frozen-VAE reconstruction control distinguishes a bad
        # generated latent from corruption already present in preprocessing.
        clean_latents = torch.cat([batch["robot_history_latents"][0],
                                   batch["video_target_latents"][0]], dim=1)
        clean_decoded = vae.decode([clean_latents])[0]
        reconstructed_rgb = (((clean_decoded[:, -8:].permute(1, 2, 3, 0).float().cpu().numpy() + 1)
                              * 127.5).clip(0, 255).astype(np.uint8))
        start = meta["latent_start"] * 4 + 1
        actual_paths = row["robot_target"]["frame_paths"][start:start + 8]
        prompt_paths = row["prompt"]["frame_paths"]
        if config.repaired_conditioning:
            from .prompt_coverage import prompt_frame_indices
            prompt_paths = [prompt_paths[index] for index in prompt_frame_indices()]
        prompt_indices = np.linspace(0, len(prompt_paths) - 1, 8).round().astype(int)
        name = f"slot{slot:02d}_{meta['task']}_source{meta['source_motion_id']:03d}_anchor{meta['latent_start']:02d}"
        frames = output_dir / f"{name}_frames"
        frames.mkdir(exist_ok=False)
        actual_rgb = np.stack([load_frame(path) for path in actual_paths])
        for index in range(8):
            canvas = Image.new("RGB", (960, 352), "black")
            canvas.paste(Image.fromarray(load_frame(prompt_paths[prompt_indices[index]])), (0, 32))
            canvas.paste(Image.fromarray(rgb[index]), (320, 32))
            canvas.paste(Image.fromarray(actual_rgb[index]), (640, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), "TRAIN demo (whole-video montage)", fill="white")
            draw.text((328, 8), "OVERFIT generated future CFG=1", fill="white")
            draw.text((648, 8), "TRAIN ground truth", fill="white")
            canvas.save(frames / f"{index:03d}.png")
        for index in (0, 4, 7):
            control = Image.new("RGB", (640, 352), "black")
            control.paste(Image.fromarray(reconstructed_rgb[index]), (0, 32))
            control.paste(Image.fromarray(actual_rgb[index]), (320, 32))
            draw = ImageDraw.Draw(control)
            draw.text((8, 8), "Frozen VAE reconstruction (GT latent)", fill="white")
            draw.text((328, 8), "TRAIN ground truth RGB", fill="white")
            control.save(output_dir / f"{name}_vae_control_{index:03d}.png")
        video = output_dir / f"{name}.mp4"
        encode_video(frames, video, ffmpeg)
        torch.save({"predicted_normalized_actions": predicted_action,
                    "target_normalized_actions": target_action,
                    "predicted_video_latents": predicted_video.detach().cpu(),
                    "target_video_latents": target_video, "meta": meta},
                   output_dir / f"{name}_predictions.pt")
        # A generated action is only meaningful if it beats the trivial
        # predictors.  The original render reported normalized_action_mse with
        # no reference, so an output that was WORSE than predicting all zeros
        # still looked like a number.  Record both baselines.
        zeros_mse = float(target_action.square().mean())
        mean_mse = float((target_action - target_action.mean()).square().mean())
        action_mse = float((predicted_action - target_action).square().mean())
        record = {"slot": slot, "meta": meta, "target_split": "train",
                  "action_mse_all_zeros_baseline": zeros_mse,
                  "action_mse_constant_mean_baseline": mean_mse,
                  "action_beats_all_zeros_baseline": action_mse < zeros_mse,
                  "predicted_action_std": float(predicted_action.std()),
                  "target_action_std": float(target_action.std()),
                  "video": str(video), "render_frame_count": 8,
                  "render_frame_names": [f"{index:03d}.png" for index in range(8)],
                  "inference_seed": seed, "video_guidance_scale": 1.0,
                  "normalized_action_mse": float((predicted_action - target_action).square().mean()),
                  "latent_video_mse": float((predicted_video.float().cpu() - target_video).square().mean()),
                  "generated_rgb_mse_0_1": float(np.mean((rgb.astype(np.float32) / 255
                                                         - actual_rgb.astype(np.float32) / 255) ** 2)),
                  "frozen_vae_reconstruction_rgb_mse_0_1": float(np.mean((
                      reconstructed_rgb.astype(np.float32) / 255 - actual_rgb.astype(np.float32) / 255) ** 2)),
                  "future_target_in_deployed_inputs": False, "physical_rollout": False}
        cases.append(record)
        write_json_atomic(output_dir / f"{name}_RESULT.json", record)
        emit_json_best_effort({"overfit_render_completed": slot + 1, "case": record})
        del prediction, predicted_video, predicted_action, decoded, decode_latents, deployed, batch
        del clean_decoded, clean_latents
    verify_decodable_render_cases(cases, ffmpeg)
    beats = sum(1 for c in cases if c["action_beats_all_zeros_baseline"])
    result = {"execution_completed": True, "target_split": "train", "cases": cases,
              "cases_where_action_beats_all_zeros": beats,
              "all_cases_beat_all_zeros": beats == len(cases),
              "all_eight_exact_training_cases": True, "physical_rollout": False,
              "video_guidance_scale": 1.0, "hash_checks": False}
    write_json_atomic(output_dir / "RENDER_RESULT.json", result)
    return result
