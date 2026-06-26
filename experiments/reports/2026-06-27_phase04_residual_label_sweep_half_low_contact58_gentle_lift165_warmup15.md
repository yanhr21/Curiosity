# Phase 04 Residual Label Source Candidate: Contact58 Gentle Lift165 Warmup15

## Scope

This report records the first ordinary-cell residual-label source candidate
that passes strict metrics while producing nonzero scripted feedback labels.
It is not learned-adapter training and not a learned policy result.

## Commands

Export and visual gate ran inside the existing tmux-held allocation:

```bash
RUN_TAG=residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006 \
WINDOW_NAME=residual_sweep_c58_lift165_warm15 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
SCENE=cube \
TRACKED_OBJECT=existing_cup_asset \
CONTROLLER_MODE=lift_hold_feedback \
PHYSICS_VARIANT_LABEL=residual_label_half_low_contact58_gentle_lift165_warmup15 \
OBJECT_MASS_KG=0.20 \
OBJECT_FRICTION_MU=0.35 \
FEEDBACK_MIN_CONTACT_COUNT=58 \
FEEDBACK_ACCEL_THRESHOLD=6.5 \
FEEDBACK_HEIGHT_DROP_THRESHOLD=0.015 \
FEEDBACK_INITIAL_LIFT_DURATION_SCALE=1.65 \
FEEDBACK_LIFT_DURATION_SCALE_MAX=1.05 \
FEEDBACK_HOLD_HEIGHT_STEP=0.0005 \
FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=0.005 \
FEEDBACK_STABILIZATION_STEP=0.05 \
FEEDBACK_STABILIZATION_MAX=0.3 \
PRE_RECORD_WARMUP_STEPS=15 \
NUM_STEPS=360 \
SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Metrics and acceleration-peak analysis:

```bash
RUN_TAG=residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006 \
WINDOW_NAME=residual_label_metrics_c58_warm15 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
BASELINE_NAME=residual_label_sweep_contact58_gentle_lift165_warmup15 \
MASS_LABEL=half \
FRICTION_LABEL=low \
POSE_SEED=contact58_gentle_lift165_warmup15 \
MANUAL_VISUAL_INSPECTION=pass_nonblank_success_with_feedback \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh

RUN_TAG=residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006 \
WINDOW_NAME=accel_peak_c58_warm15 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
TOP_K=12 \
bash experiments/configs/launch_lift_hold_accel_peak_analysis_tmux.sh
```

## Evidence

- Fresh official Newton sanity:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006_fresh_newton_sensor_contact_sanity.json`.
- Summary:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006_summary.json`.
- Visual validation:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006_visual_validation.json`.
- Manual visual inspection:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006_manual_visual_inspection.json`.
- Metrics:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006_metrics.json`.
- Acceleration peak analysis:
  `experiments/outputs/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006_accel_peak_analysis.json`.
- Contact sheet:
  `experiments/visuals/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006/contact_sheet.png`.
- Frame browser:
  `experiments/visuals/residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006/frame_browser.html`.

## Result

Status: promoted source candidate for the next residual-label runner.

- Official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_with_feedback`.
- Metrics status: pass.
- Pre-record warmup steps: 15.
- Final feedback trigger count: 241.
- Feedback-active frames: 241.
- Object dropped: false.
- Contact-loss frames: 0.
- Max contact proxy: 62.
- Lift height: 0.15815936028957367 m.
- Hold duration: 2.5333309173583984 s.
- Max slip: 0.0030417809728431086 m.
- Max object acceleration: 0.5063306543767194 m/s^2.
- Success per contact-proxy integral: 0.003121588742547919.

## Interpretation

The previous strict acceleration failure came from a recorded initial settling
artifact: the non-warmup peak analysis put the top event at step 2, phase 0,
before feedback was active. Adding 15 pre-record warmup steps keeps the
official Newton rollout path, preserves nonzero residual labels, and removes
that artifact from the recorded metric window.

This is the first ordinary `half_low` residual-label source candidate that has
nonzero feedback and passes strict metrics. The next step is to build a formal
source manifest and residual-label runner around it, then collect more ordinary
cells. It still does not claim learned adaptation, T-Rex tactile F6, schema
promotion, or model training.
