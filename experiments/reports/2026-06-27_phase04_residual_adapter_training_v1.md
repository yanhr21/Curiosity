# Phase 04 Residual Adapter Real Training V1

## Scope

This report records the first real training run of the Newton-native residual
controller-parameter adapter. It is not an official T-Rex method, not a T-Rex
schema result, and not a policy-improvement claim. It only establishes that a
checkpoint was trained from the compute-verified residual-label split under
the one-GPU one-hour rule.

## Command

```bash
ALLOW_REAL_TRAINING=1 \
RUN_TAG=residual_adapter_trainer_v1_train_20260627_0548 \
RUN_MODE=train \
WINDOW_NAME=residual_adapter_train_1h \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
bash experiments/configs/launch_residual_adapter_trainer_tmux.sh
```

The run reused the existing Curiosity tmux-held Slurm allocation `154142` on
`server56`. No new allocation or one-shot `sbatch` path was used.

## Environment

- Newton sanity venv: `envs/newton/.venv`.
- Trainer venv: `envs/residual_adapter/.venv`.
- Trainer dependencies were prepared locally under `envs/`, not on the compute
  node.
- Torch version: `2.6.0+cu124`.
- Device: `cuda:0`, NVIDIA H200.

## Inputs

- Trainer config:
  `experiments/configs/residual_adapter_trainer_v1.json`.
- Preflight manifest:
  `data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
- Train split:
  `data/processed/residual_adapter_training_preflight_v1_20260627/residual_adapter_train_records.csv`.
- Validation split:
  `data/processed/residual_adapter_training_preflight_v1_20260627/residual_adapter_validation_records.csv`.
- Train records: `1440`.
- Validation records: `360`.
- Train cells: `half_low`, `empty_low`, `half_medium`, `full_high`.
- Validation cell: `empty_medium`.
- Held-out cells excluded from labels and training: `full_low`, `empty_high`.

## Outputs

- Log:
  `logs/newton/residual_adapter_trainer_v1_train_20260627_0548.log`.
- Summary:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548_summary.json`.
- Fresh official Newton sanity:
  `experiments/outputs/residual_adapter_trainer_v1_train_20260627_0548_fresh_newton_sensor_contact_sanity.json`.
- GPU utilization CSV:
  `logs/newton/residual_adapter_trainer_v1_train_20260627_0548_gpu_utilization.csv`.
- GPU utilization JSON:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548_gpu_utilization.json`.
- Checkpoint:
  `checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt`.

## Result

- status: pass;
- run mode: train;
- fresh official Newton sanity: pass;
- elapsed seconds: `3600.0302035808563`;
- optimizer steps: `32685`;
- effective train batch size: `2048`;
- train batch repeat: `512`;
- checkpoint written: true;
- real training result: true;
- generated T-Rex fields: `[]`;
- schema promotion: `blocked`;
- failures: `[]`.

Validation metrics:

- loss: `6.241170922294259e-05`;
- active BCE: `1.1115849929410615e-06`;
- continuous MSE: `6.130012479843572e-05`;
- feedback-active accuracy: `1.0`.

GPU utilization:

- sample count: `120`;
- mean utilization: `99.08333333333333%`;
- min utilization: `0.0%`;
- max utilization: `100.0%`;
- max memory used: `30233.0` MiB;
- samples below 30%: `1`;
- monitor status: pass.

## Interpretation

The real-training gate is complete: the adapter trained for at least one GPU
hour, wrote a checkpoint, passed fresh official Newton sanity, and satisfied
the GPU utilization rule.

This still does not prove learned closed-loop adaptation. The checkpoint has
not yet been inserted into the Newton controller path, has not been evaluated
on held-out `full_low` or `empty_high`, and has no trained-policy visual gate.

## Next Step

Wire the checkpoint into the Newton residual-controller evaluation path and
run visual plus metric gates. Start with a non-held-out validation rollout,
then evaluate held-out `full_low` and `empty_high`. Only after that comparison
against no-adaptation and scripted-feedback baselines can the project make a
learned-adaptation claim.
