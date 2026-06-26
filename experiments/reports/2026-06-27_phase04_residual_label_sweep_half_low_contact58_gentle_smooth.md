# Phase 04 Residual Label Sweep Half/Low Contact58 Gentle Smooth Diagnostic

## Scope

This report records the follow-up diagnostic after the contact58 gentle run.
The goal was to reduce the repeated strict `object_accel_above_threshold`
failure by increasing the initial lift duration scale while preserving nonzero
residual labels. It is not training and not a learned-adapter claim.

## Command

Executed inside the existing Curiosity tmux-held Slurm allocation:

```bash
RUN_TAG=residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
SCENE=cube \
TRACKED_OBJECT=existing_cup_asset \
CONTROLLER_MODE=lift_hold_feedback \
PHYSICS_VARIANT_LABEL=residual_label_half_low_contact58_gentle_smooth \
OBJECT_MASS_KG=0.20 \
OBJECT_FRICTION_MU=0.35 \
FEEDBACK_MIN_CONTACT_COUNT=58 \
FEEDBACK_ACCEL_THRESHOLD=6.5 \
FEEDBACK_HEIGHT_DROP_THRESHOLD=0.015 \
FEEDBACK_INITIAL_LIFT_DURATION_SCALE=1.8 \
FEEDBACK_LIFT_DURATION_SCALE_MAX=1.05 \
FEEDBACK_HOLD_HEIGHT_STEP=0.0005 \
FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=0.005 \
FEEDBACK_STABILIZATION_STEP=0.05 \
FEEDBACK_STABILIZATION_MAX=0.3 \
NUM_STEPS=360 \
SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Metrics were extracted inside the same allocation:

```bash
RUN_TAG=residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355 \
WINDOW_NAME=residual_label_metrics_c58_gsmooth \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
BASELINE_NAME=residual_label_sweep_contact58_gentle_smooth \
MASS_LABEL=half \
FRICTION_LABEL=low \
POSE_SEED=contact58_gentle_smooth \
MANUAL_VISUAL_INSPECTION=pass_nonblank_success_with_feedback \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Evidence

- Fresh official Newton sanity:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355_fresh_newton_sensor_contact_sanity.json`.
- Summary:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355_summary.json`.
- Visual validation:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355_visual_validation.json`.
- Manual visual inspection:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355_manual_visual_inspection.json`.
- Metrics:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355_metrics.json`.
- Contact sheet:
  `experiments/visuals/residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355/contact_sheet.png`.
- Frame browser:
  `experiments/visuals/residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355/frame_browser.html`.

## Result

Status: diagnostic only, not promoted.

- Official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_with_feedback`.
- Feedback reason: `low_contact_count`.
- Final feedback trigger count: 241.
- Feedback-active frames: 241.
- Object dropped: false.
- Contact-loss frames: 0.
- Max contact proxy: 61.
- Lift height: 0.1519654542207718 m.
- Hold duration: 2.3499977588653564 s.
- Max slip: 0.0026201330580321184 m.
- Max object acceleration: 8.308972018193668 m/s^2.
- Metrics status: fail.
- Failure reason: `object_accel_above_threshold`.

## Interpretation

Increasing `FEEDBACK_INITIAL_LIFT_DURATION_SCALE` from `1.35` to `1.8`
preserved nonzero residual labels and kept lift, hold, drop, contact-loss, and
visual gates valid. It did not reduce the strict object-acceleration failure;
the measured maximum acceleration remained around `8.309` m/s^2.

This suggests the acceleration spike is not solved by simply stretching the
initial lift waypoint duration. The next step should not be more blind
threshold sweeping. It needs an approved choice between:

- revising the strict acceleration gate if `8.309` is a known Newton/contact
  artifact and not a meaningful safety failure;
- changing the official-scripted waypoint/contact transition more directly,
  then rerunning the full sanity/visual/metrics gate;
- accepting the gentle run as a residual-label source under a documented
  "strict acceleration exception" policy.

Until that decision is made, learned residual-adapter training remains blocked.
