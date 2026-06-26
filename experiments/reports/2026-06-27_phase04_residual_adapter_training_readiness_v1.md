# Phase 04 Residual Adapter Training Readiness V1

## Scope

This report audits the first Newton-native residual controller-parameter
adapter after real training. It is not an official T-Rex method, does not use
T-Rex schema fields, and does not claim policy improvement before held-out
controller evaluation.

## Current Result

Status: real one-GPU one-hour residual-adapter training completed, ordinary
validation passed, and the two reserved held-out cup cells passed visual plus
metric evaluation.

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

## Held-Out Evaluation Follow-Up

Held-out evaluation now passes on both reserved cup cells.

- Report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_heldout_eval_v1.md`.
- `full_low` run:
  `residual_adapter_eval_v1_full_low_heldout_20260627_0613`.
- `full_low` metrics: lift `0.1548849195241928` m, hold
  `2.499997615814209` s, max slip `0.0034206882392378247` m,
  contact-loss frames `0`, max acceleration `1.5345948979069628` m/s^2.
- `empty_high` run:
  `residual_adapter_eval_v1_empty_high_heldout_20260627_0620`.
- `empty_high` metrics: lift `0.1613951474428177` m, hold
  `2.566664218902588` s, max slip `0.003700697622575275` m,
  contact-loss frames `0`, max acceleration `0.4686260874870734` m/s^2.

Baseline comparison on the same two cells: no-adaptation and scripted-feedback
baselines were visually valid but failed the full metrics schema only on
`object_accel_above_threshold` around `8.308` m/s^2. The learned residual
adapter passes those two held-out cells under the current schema.

## Remaining Blocking Gaps

- Broad object-family generalization is not proven.
- Tactile F6 and T-Rex compatibility remain unavailable.
- More seeds/cells are needed before broad learned-adaptation claims.

## Next Step

Proceed to broader evaluation, not more gating on source mismatch:

1. Repeat held-out cells with additional seeds or perturbations.
2. Add more ordinary/held-out cup cells or new object families.
3. Keep comparing against no-adaptation and scripted-feedback baselines.

Do not claim T-Rex compatibility, tactile F6, curiosity policy update, or
policy improvement until those controller-evaluation gates pass.
