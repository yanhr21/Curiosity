# Phase 04 Residual Label Source Sensitive Feedback Half/Low Diagnostic

## Scope

This report records one ordinary-cell diagnostic to turn the residual-adapter
training blocker into an executable next step. It is not training, not a
learned adapter, and not an adaptation-success claim.

## Command

Executed inside the existing Curiosity tmux-held Slurm allocation:

```bash
RUN_TAG=residual_label_source_sensitive_feedback_half_low_20260627_030145 \
WINDOW_NAME=residual_label_source_half_low \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
SCENE=cube \
TRACKED_OBJECT=existing_cup_asset \
CONTROLLER_MODE=lift_hold_feedback \
PHYSICS_VARIANT_LABEL=residual_label_half_low_sensitive_contact_threshold \
OBJECT_MASS_KG=0.20 \
OBJECT_FRICTION_MU=0.35 \
FEEDBACK_MIN_CONTACT_COUNT=64 \
FEEDBACK_ACCEL_THRESHOLD=6.5 \
FEEDBACK_HEIGHT_DROP_THRESHOLD=0.015 \
NUM_STEPS=360 \
SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Metrics were extracted inside the same allocation:

```bash
RUN_TAG=residual_label_source_sensitive_feedback_half_low_20260627_030145 \
WINDOW_NAME=residual_label_metrics_half_low \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
BASELINE_NAME=residual_label_source_sensitive_feedback \
MASS_LABEL=half \
FRICTION_LABEL=low \
POSE_SEED=sensitive_contact_threshold \
MANUAL_VISUAL_INSPECTION=pass_nonblank_but_task_failure \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Evidence

- Fresh official Newton sanity:
  `experiments/outputs/residual_label_source_sensitive_feedback_half_low_20260627_030145_fresh_newton_sensor_contact_sanity.json`.
- Summary:
  `experiments/outputs/residual_label_source_sensitive_feedback_half_low_20260627_030145_summary.json`.
- Visual validation:
  `experiments/outputs/residual_label_source_sensitive_feedback_half_low_20260627_030145_visual_validation.json`.
- Manual visual inspection:
  `experiments/outputs/residual_label_source_sensitive_feedback_half_low_20260627_030145_manual_visual_inspection.json`.
- Metrics:
  `experiments/outputs/residual_label_source_sensitive_feedback_half_low_20260627_030145_metrics.json`.
- Contact sheet:
  `experiments/visuals/residual_label_source_sensitive_feedback_half_low_20260627_030145/contact_sheet.png`.
- Frame browser:
  `experiments/visuals/residual_label_source_sensitive_feedback_half_low_20260627_030145/frame_browser.html`.

## Result

Status: diagnostic succeeded, not promoted to training source.

- Official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass for nonblank diagnostic evidence, but task
  failure is consistent with metrics.
- Feedback reason: `low_contact_count`.
- Final feedback trigger count: 241.
- Feedback-active frames: 241.
- Feedback lift velocity scale range: `0.35..1.0`.
- Feedback hold height offset range: `-0.03..0.0` m.
- Feedback stabilization extension range: `0.0..2.0` s.
- Object dropped: false.
- Contact-loss frames: 0.
- Max contact proxy: 61.
- Lift height: 0.12735126912593842 m.
- Hold duration: 0.9833323955535889 s.
- Max slip: 0.000388524929963087 m.
- Max object acceleration: 8.308707788010144 m/s^2.
- Metrics status: fail.
- Failure reasons: `hold_duration_below_threshold`,
  `object_accel_above_threshold`.

## Interpretation

This run proves the official Newton rollout path can produce nonzero residual
controller-parameter labels under `candidate.controller.*`. It also shows that
`FEEDBACK_MIN_CONTACT_COUNT=64` is too aggressive for a promoted training-label
source because it destroys the hold-duration gate.

The next action is a bounded ordinary-cell threshold sweep, not training. The
sweep should search for thresholds that keep `feedback_trigger_count > 0` while
preserving lift, hold, drop, contact-loss, visual, and sanity gates. Held-out
`full_low` and `empty_high` remain untouched for training-label collection.
