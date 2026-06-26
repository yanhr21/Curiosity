# Phase 04 Residual Adapter Empty-Medium Validation V1

## Scope

This report records the first closed-loop Newton evaluation of the trained
Newton-native residual controller adapter. It is an ordinary validation cell
(`empty_medium`), not held-out generalization and not a T-Rex result.

## Command

```bash
RUN_TAG=residual_adapter_eval_v1_empty_medium_validation_20260627_0605 \
WINDOW_NAME=residual_adapter_eval_empty_medium \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
bash experiments/configs/launch_residual_adapter_evaluation_tmux.sh
```

Metrics command:

```bash
RUN_TAG=residual_adapter_eval_v1_empty_medium_validation_20260627_0605 \
WINDOW_NAME=residual_adapter_eval_metrics_rerun \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
BASELINE_NAME=learned_residual_adapter_eval \
MASS_LABEL=empty \
FRICTION_LABEL=medium \
POSE_SEED=validation_empty_medium \
MANUAL_VISUAL_INSPECTION=pass_nonblank_success_learned_residual \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

The run reused tmux-held allocation `154142`. No new allocation or one-shot
`sbatch` path was used.

## Inputs

- Checkpoint:
  `checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt`.
- Controller mode: `lift_hold_learned_residual`.
- Cell: `empty_medium`, ordinary validation.
- Object mass: `0.08` kg.
- Object friction: `0.8`.
- Pre-record warmup steps: `15`.
- Residual adapter active threshold: `0.5`.

## Outputs

- Log:
  `logs/newton/residual_adapter_eval_v1_empty_medium_validation_20260627_0605.log`.
- Run status:
  `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605_run_status.json`.
- Fresh official Newton sanity:
  `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605_fresh_newton_sensor_contact_sanity.json`.
- Summary:
  `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605_summary.json`.
- Visual validation:
  `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605_visual_validation.json`.
- Manual visual inspection:
  `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605_manual_visual_inspection.json`.
- Metrics:
  `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605_metrics.json`.
- NPZ:
  `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605.npz`.
- Frame browser:
  `experiments/visuals/residual_adapter_eval_v1_empty_medium_validation_20260627_0605/frame_browser.html`.
- Contact sheet:
  `experiments/visuals/residual_adapter_eval_v1_empty_medium_validation_20260627_0605/contact_sheet.png`.

## Result

- Fresh official Newton sanity: pass.
- Camera export: pass.
- Visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_learned_residual`.
- Metrics status: pass.
- Controller type: `newton_native_residual_controller_adapter_evaluation`.
- Model evaluation: true.
- Generated T-Rex fields: `[]`.
- Schema promotion: `blocked`.
- Final learned residual trigger count: `240`.

Metrics:

- lift height: `0.16149283945560455` m;
- hold duration: `2.566664218902588` s;
- max slip: `0.0036941702785655258` m;
- drop height loss: `0`;
- contact-loss frames: `0`;
- max contact proxy: `61`;
- max object acceleration: `0.6439671191529558` m/s^2;
- object not dropped: true;
- status: success.

Manual visual inspection of the frame browser/contact sheet and frames
`0000`, `0180`, and `0359` showed nonblank camera panels, visible cup/gripper
interaction, lift in the middle frames, and the cup still held in the final
frame.

## Interpretation

The checkpoint is now wired into the Newton closed-loop controller path and has
passed one ordinary validation rollout with visual and metric gates. This
resolves the ordinary controller-integration gate.

This still does not prove held-out generalization or policy improvement over
baselines. Held-out `full_low` and `empty_high` evaluation must run next, with
the same fresh sanity, visual/browser, manual inspection, and metrics gates.
