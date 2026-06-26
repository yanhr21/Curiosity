# Phase 04 Scripted Feedback Empty-Low Variant

## Scope

This is the first scripted feedback ordinary mass/friction grid cell for Phase
04 closed-loop adaptation.

This is not a learned policy and not a curiosity result. It uses the official
Newton Panda hydro scripted infant prior plus scripted controller feedback.

## Run

- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615`
- Mass label: `empty`
- Friction label: `low`
- Requested object mass: `0.08` kg
- Requested friction mu: `0.35`
- Observed body mass: `0.07999999821186066` kg
- Observed shape material mu: `0.3499999940395355`
- Held-out cell: false

## Command

Launched through the scripted feedback baseline launcher into the existing
tmux-held Curiosity allocation `154023`.

```bash
RUN_TAG=lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
PHYSICS_VARIANT_LABEL=feedback_empty_low_mass0p08_mu0p35 \
OBJECT_MASS_KG=0.08 \
OBJECT_FRICTION_MU=0.35 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Metrics were extracted after manual visual inspection:

```bash
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615 \
WINDOW_NAME=lift_hold_feedback_metrics_fix \
BASELINE_NAME=scripted_feedback_adaptation_grasp_lift \
MASS_LABEL=empty \
FRICTION_LABEL=low \
POSE_SEED=feedback_empty_low \
MANUAL_VISUAL_INSPECTION=pass \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Gate Evidence

- Fresh official Newton sanity:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615_fresh_newton_sensor_contact_sanity.json`
- Visual validation:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615_manual_visual_inspection.json`
- Metrics:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615_metrics.json`
- Rollout:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615.npz`

Direct visual paths:

- Contact sheet:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615/contact_sheet.png`
- Frame browser:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615/frame_browser.html`
- Key inspected frames:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615/frame_0225.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615/frame_0315.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615/frame_0359.png`

## Result

Visual gate: pass.

Strict metrics: fail, only because `object_accel_above_threshold`.

Metric values:

- lift height: `0.1602502018213272` m;
- hold duration: `2.8333306312561035` s;
- max slip: `0.003562770042336314` m;
- drop height loss: `0.0` m;
- contact loss frames: `0`;
- max contact proxy: `61.0`;
- max object acceleration: `8.308707937632189` m/s^2.

Scripted feedback summary:

- feedback trigger count: `0`;
- feedback reason labels: only `none`.

## Interpretation

The feedback grid path is runnable and visually valid on a real mass/friction
variant. Real Newton mass/friction values were applied before finalize.

This cell still does not prove adaptation improvement because no feedback event
was triggered. Continue running the remaining ordinary grid cells before making
adaptation claims.
