# Phase 06 T-Rex Checkpoint Current Sanity

## Scope

This is a reference-only sanity check for currently staged official T-Rex
checkpoint assets. It does not train, does not create data, does not create a
Newton-to-T-Rex bridge, and does not change the short-term Newton-native infant
route.

## Files

- Launcher: `experiments/configs/launch_trex_checkpoint_sanity_tmux.sh`
- Compute runner: `experiments/configs/run_trex_checkpoint_sanity_in_alloc.sh`
- Integrity checker: `experiments/configs/trex_checkpoint_integrity_sanity.py`
- Model-load checker: `experiments/configs/trex_midtrain_model_load_sanity.py`
- Log: `logs/trex/trex_checkpoint_current_sanity_20260627_2055.log`
- Integrity output:
  `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_integrity.json`
- Model-load output:
  `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_midtrain_model_load.json`

## Command

Executed inside the existing Curiosity tmux-held Slurm allocation:

```bash
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
  RUN_TAG=trex_checkpoint_current_sanity_20260627_2055 \
  bash experiments/configs/launch_trex_checkpoint_sanity_tmux.sh
```

The compute runner now fails if `SLURM_JOB_ID` is not set.

## Result

Status: pass.

- Checkpoint integrity sanity: pass.
- Official midtrain model-load sanity: pass.
- Official T-Rex `scripts/test.py` model-load path used.
- Embedded tactile VQ-VAE components were present in the model-load sanity.
- Midtrain state dict contains `tactile_vqvae.*`, `tactile_code_embedder`,
  `deform_encoder`, `deform_proj`, and TacF6 VQ-VAE stat buffers.
- Pretrain stage-1 checkpoint and Qwen3-VL-2B-Instruct safetensors passed the
  current integrity checks.
- Official model-load elapsed time: `35.837836027145386` seconds on `server56`.
- No training was run.
- No placeholder model was created.
- No generated T-Rex schema fields were created.

## Interpretation

This proves only that the currently staged official T-Rex checkpoint assets can
pass integrity checks and load through the official model-load path in the
current environment.

It does not prove that Newton Panda lift-hold data satisfies T-Rex's strict
bimanual state/action/camera/F6/deformation contract. Phase 06 strict bridge
promotion remains blocked until real synchronized 62D state/action/action_abs,
accepted cameras, calibrated nonzero `[10,6]` F6, and ten dense tactile
deformation streams exist without padding or renaming.
