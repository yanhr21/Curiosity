# Phase 04 Residual Label Sweep Half/Low Contact58 Gentle Diagnostic

## Scope

This report records the strongest current ordinary-cell residual-label
diagnostic. It uses the same contact-threshold route as the prior contact58
sweep, but caps the feedback correction magnitudes to preserve the lift-hold
behavior. It is not training and not a learned-adapter claim.

## Command

Executed inside the existing Curiosity tmux-held Slurm allocation:

```bash
RUN_TAG=residual_label_sweep_half_low_contact58_gentle_20260627_0345 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
SCENE=cube \
TRACKED_OBJECT=existing_cup_asset \
CONTROLLER_MODE=lift_hold_feedback \
PHYSICS_VARIANT_LABEL=residual_label_half_low_contact58_gentle \
OBJECT_MASS_KG=0.20 \
OBJECT_FRICTION_MU=0.35 \
FEEDBACK_MIN_CONTACT_COUNT=58 \
FEEDBACK_ACCEL_THRESHOLD=6.5 \
FEEDBACK_HEIGHT_DROP_THRESHOLD=0.015 \
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
RUN_TAG=residual_label_sweep_half_low_contact58_gentle_20260627_0345 \
WINDOW_NAME=residual_label_metrics_c58_gentle \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
BASELINE_NAME=residual_label_sweep_contact58_gentle \
MASS_LABEL=half \
FRICTION_LABEL=low \
POSE_SEED=contact58_gentle \
MANUAL_VISUAL_INSPECTION=pass_nonblank_success_with_feedback \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Evidence

- Fresh official Newton sanity:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_20260627_0345_fresh_newton_sensor_contact_sanity.json`.
- Summary:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_20260627_0345_summary.json`.
- Visual validation:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_20260627_0345_visual_validation.json`.
- Manual visual inspection:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_20260627_0345_manual_visual_inspection.json`.
- Metrics:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_20260627_0345_metrics.json`.
- Contact sheet:
  `experiments/visuals/residual_label_sweep_half_low_contact58_gentle_20260627_0345/contact_sheet.png`.
- Frame browser:
  `experiments/visuals/residual_label_sweep_half_low_contact58_gentle_20260627_0345/frame_browser.html`.

## Result

Status: best current residual-label candidate, not fully promoted.

- Official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_with_feedback`.
- Feedback reason: `low_contact_count`.
- Final feedback trigger count: 241.
- Feedback-active frames: 241.
- Object dropped: false.
- Contact-loss frames: 0.
- Max contact proxy: 61.
- Lift height: 0.1518997997045517 m.
- Hold duration: 2.7166640758514404 s.
- Max slip: 0.002728142855700976 m.
- Max object acceleration: 8.308707788010144 m/s^2.
- Metrics status: fail.
- Failure reason: `object_accel_above_threshold`.

## Interpretation

This is the first diagnostic that combines nonzero residual controller labels
with preserved lift, hold, drop, contact-loss, automated visual validation, and
manual visual inspection. The only remaining strict failure is object
acceleration above the current 8.0 m/s^2 threshold.

It should become the main candidate for the next ordinary-cell diagnostic, but
learned residual-adapter training should still wait. The next concrete action
is to reduce object acceleration while preserving nonzero feedback, for example
with gentler lift/hold timing or capped trigger frequency, then rerun the same
sanity/visual/manual/metrics gates.
