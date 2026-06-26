# Phase 04 Residual Adapter Training Readiness V1

## Scope

This report audits the first Newton-native residual controller-parameter
adapter after real training. It is not an official T-Rex method, does not use
T-Rex schema fields, and does not claim policy improvement before held-out
controller evaluation.

## Current Result

Status: real one-GPU one-hour residual-adapter training completed; held-out
adapter evaluation has not started.

The previous active blocker was `real_training_not_started`. That blocker is
resolved by:

- run tag: `residual_adapter_trainer_v1_train_20260627_0548`;
- command:
  `ALLOW_REAL_TRAINING=1 RUN_TAG=residual_adapter_trainer_v1_train_20260627_0548 RUN_MODE=train WINDOW_NAME=residual_adapter_train_1h JOB_ID=154142 TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 bash experiments/configs/launch_residual_adapter_trainer_tmux.sh`;
- Slurm allocation: `154142`, reused existing tmux-held Curiosity allocation;
- node/device: NVIDIA H200, `cuda:0`;
- elapsed seconds: `3600.0302035808563`;
- optimizer steps: `32685`;
- checkpoint:
  `checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt`;
- summary:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548_summary.json`;
- GPU utilization:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548_gpu_utilization.json`;
- report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_training_v1.md`.

## Training Evidence

- Fresh official Newton sanity: pass.
- Train records: `1440`.
- Validation records: `360`.
- Train cells: `half_low`, `empty_low`, `half_medium`, `full_high`.
- Validation cell: `empty_medium`.
- Held-out cells excluded from source labels and training: `full_low`,
  `empty_high`.
- Validation loss: `6.241170922294259e-05`.
- Validation active accuracy: `1.0`.
- Validation active BCE: `1.1115849929410615e-06`.
- Validation continuous MSE: `6.130012479843572e-05`.
- Generated T-Rex fields: `[]`.
- Schema promotion: `blocked`.
- Failures: `[]`.

GPU utilization monitor:

- sample count: `120`;
- mean utilization: `99.08333333333333%`;
- max utilization: `100.0%`;
- max memory used: `30233.0` MiB;
- samples below 30%: `1`;
- status: pass.

## Resolved Blockers

- No real training run: resolved.
- No trained adapter checkpoint: resolved.
- No formal source runner: resolved earlier by
  `residual_label_source_runner_v1_20260627_0455`.
- No training preflight: resolved earlier by
  `residual_adapter_training_preflight_v1_20260627_0523`.
- No trainer smoke: resolved earlier by
  `residual_adapter_trainer_v1_smoke_20260627_0539`.

## Remaining Blocking Gaps

- The checkpoint has not been wired into the Newton controller evaluation path.
- No trained-adapter rollout has been visually inspected in the browser/frame
  output.
- No held-out adapter evaluation exists on `full_low` or `empty_high`.
- No policy-improvement or learned-adaptation claim is valid yet.

## Next Step

Proceed to checkpoint evaluation, not more gating on source mismatch:

1. Wire
   `checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt`
   into the Newton residual-controller evaluation path.
2. Run a non-held-out validation rollout first, with fresh official Newton
   sanity, camera export, visual/browser inspection, and lift-hold metrics.
3. Then evaluate held-out `full_low` and `empty_high` against no-adaptation and
   scripted-feedback baselines.

Do not claim T-Rex compatibility, tactile F6, curiosity policy update, or
policy improvement until those controller-evaluation gates pass.
