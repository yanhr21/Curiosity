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

- No held-out adapter evaluation exists on `full_low` or `empty_high`.
- No policy-improvement or learned-adaptation claim is valid yet.

## Ordinary Evaluation Follow-Up

The checkpoint has now been wired into the Newton controller evaluation path
and passed ordinary validation on `empty_medium`.

- Run tag: `residual_adapter_eval_v1_empty_medium_validation_20260627_0605`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_eval_empty_medium_validation_v1.md`.
- Fresh official Newton sanity: pass.
- Visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_learned_residual`.
- Metrics status: pass.
- Lift height: `0.16149283945560455` m.
- Hold duration: `2.566664218902588` s.
- Max slip: `0.0036941702785655258` m.
- Max object acceleration: `0.6439671191529558` m/s^2.
- Generated T-Rex fields: `[]`.
- Schema promotion: `blocked`.

## Next Step

Proceed to held-out checkpoint evaluation, not more gating on source mismatch:

1. Evaluate held-out `full_low` with fresh official Newton sanity, camera
   export, visual/browser inspection, and lift-hold metrics.
2. Evaluate held-out `empty_high` with the same gates.
3. Compare against no-adaptation and scripted-feedback baselines.

Do not claim T-Rex compatibility, tactile F6, curiosity policy update, or
policy improvement until those controller-evaluation gates pass.
