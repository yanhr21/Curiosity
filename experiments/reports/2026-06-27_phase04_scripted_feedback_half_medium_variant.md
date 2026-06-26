# Phase 04 Scripted Feedback Half-Medium Variant

## Scope

This is the fourth scripted feedback ordinary mass/friction grid cell for Phase
04 closed-loop adaptation.

This is not a learned policy and not a curiosity result. It uses the official
Newton Panda hydro scripted infant prior plus scripted controller feedback.

## Run

- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725`
- Mass label: `half`
- Friction label: `medium`
- Requested object mass: `0.20` kg
- Requested friction mu: `0.80`
- Observed body mass: `0.20000000298023224` kg
- Observed shape material mu: `0.800000011920929`
- Held-out cell: false

## Command

The run used the existing tmux-held Curiosity allocation `154023`.

```bash
RUN_TAG=lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
PHYSICS_VARIANT_LABEL=feedback_half_medium_mass0p20_mu0p80 \
OBJECT_MASS_KG=0.20 \
OBJECT_FRICTION_MU=0.80 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Metrics were extracted after manual visual inspection:

```bash
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725 \
WINDOW_NAME=lift_hold_feedback_metrics \
BASELINE_NAME=scripted_feedback_adaptation_grasp_lift \
MASS_LABEL=half \
FRICTION_LABEL=medium \
POSE_SEED=feedback_half_medium \
MANUAL_VISUAL_INSPECTION=pass \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Gate Evidence

- Fresh official Newton sanity:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725_fresh_newton_sensor_contact_sanity.json`
- Visual validation:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725_manual_visual_inspection.json`
- Metrics:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725_metrics.json`
- Rollout:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725.npz`

Direct visual paths:

- Contact sheet:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725/contact_sheet.png`
- Frame browser:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725/frame_browser.html`
- Key inspected frames:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725/frame_0225.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725/frame_0315.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725/frame_0359.png`

## Result

Visual gate: pass.

Strict metrics: fail, only because `object_accel_above_threshold`.

Metric values:

- lift height: `0.15682105720043182` m;
- hold duration: `2.799997329711914` s;
- max slip: `0.003182727513861391` m;
- drop height loss: `0.0` m;
- contact loss frames: `0`;
- max contact proxy: `62.0`;
- max object acceleration: `8.308865546822908` m/s^2.

Scripted feedback summary:

- feedback trigger count: `0`;
- feedback reason labels: only `none`.

## Interpretation

The feedback grid path is runnable and visually valid on the half/medium
mass/friction variant. Real Newton mass/friction values were applied before
finalize.

This cell still does not prove adaptation improvement because no feedback event
was triggered. Continue running the remaining ordinary grid cells before making
adaptation claims.
