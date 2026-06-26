# Phase 04 Residual Label Source Accel-Sensitive Half/Low Diagnostic

## Scope

This report records the second ordinary-cell residual-label diagnostic. It
tests whether lowering the online object-acceleration trigger can create
nonzero residual controller labels without destroying lift-hold behavior. It is
not training and not a learned-adapter claim.

## Command

Executed inside the existing Curiosity tmux-held Slurm allocation:

```bash
RUN_TAG=residual_label_source_accel_sensitive_half_low_20260627_030748 \
WINDOW_NAME=residual_label_accel_half_low \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
SCENE=cube \
TRACKED_OBJECT=existing_cup_asset \
CONTROLLER_MODE=lift_hold_feedback \
PHYSICS_VARIANT_LABEL=residual_label_half_low_accel_threshold_5p5 \
OBJECT_MASS_KG=0.20 \
OBJECT_FRICTION_MU=0.35 \
FEEDBACK_MIN_CONTACT_COUNT=20 \
FEEDBACK_ACCEL_THRESHOLD=5.5 \
FEEDBACK_HEIGHT_DROP_THRESHOLD=0.015 \
NUM_STEPS=360 \
SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Metrics were extracted inside the same allocation:

```bash
RUN_TAG=residual_label_source_accel_sensitive_half_low_20260627_030748 \
WINDOW_NAME=residual_label_metrics_accel_half_low \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
BASELINE_NAME=residual_label_source_accel_sensitive_feedback \
MASS_LABEL=half \
FRICTION_LABEL=low \
POSE_SEED=accel_threshold_5p5 \
MANUAL_VISUAL_INSPECTION=pass_nonblank_success_no_feedback \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Evidence

- Fresh official Newton sanity:
  `experiments/outputs/residual_label_source_accel_sensitive_half_low_20260627_030748_fresh_newton_sensor_contact_sanity.json`.
- Summary:
  `experiments/outputs/residual_label_source_accel_sensitive_half_low_20260627_030748_summary.json`.
- Visual validation:
  `experiments/outputs/residual_label_source_accel_sensitive_half_low_20260627_030748_visual_validation.json`.
- Manual visual inspection:
  `experiments/outputs/residual_label_source_accel_sensitive_half_low_20260627_030748_manual_visual_inspection.json`.
- Metrics:
  `experiments/outputs/residual_label_source_accel_sensitive_half_low_20260627_030748_metrics.json`.
- Contact sheet:
  `experiments/visuals/residual_label_source_accel_sensitive_half_low_20260627_030748/contact_sheet.png`.
- Frame browser:
  `experiments/visuals/residual_label_source_accel_sensitive_half_low_20260627_030748/frame_browser.html`.

## Result

Status: stable diagnostic, not a residual-label source.

- Official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_no_feedback`.
- Feedback reason: none.
- Final feedback trigger count: 0.
- Feedback-active frames: 0.
- Object dropped: false.
- Contact-loss frames: 0.
- Max contact proxy: 62.
- Lift height: 0.15682174265384674 m.
- Hold duration: 2.799997329711914 s.
- Max slip: 0.003185424534726402 m.
- Max object acceleration: 8.308707937632189 m/s^2.
- Metrics status: fail.
- Failure reason: `object_accel_above_threshold`.

## Interpretation

This run keeps the normal lift/hold/drop/contact behavior, but it does not
produce residual labels because `feedback_trigger_count=0`. The acceleration
trigger route is likely blocked by the current online trigger timing or
waypoint condition rather than by the absence of acceleration evidence.

The next sweep should not train. It should test sparse/windowed post-contact
triggering or a capped trigger frequency so the controller can emit a small
number of residual corrections without destroying the hold gate.
