# Phase 04 Scripted Feedback Nominal Cup Gate

## Scope

This run validates the first scripted feedback controller path for Phase 04
closed-loop adaptation. It starts from the official Newton Panda hydro scripted
infant prior and adds a contact/object-motion feedback rule around controller
parameters.

This is not a learned policy and not a curiosity result.

## Files

- Config: `experiments/configs/lift_hold_scripted_feedback_baseline_v1.json`
- Launcher: `experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh`
- Shared launcher: `experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh`
- Shared runner: `experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh`
- Exporter: `experiments/configs/newton_panda_hydro_tiled_camera_export.py`
- Run tag: `lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545`
- Log: `logs/newton/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545.log`
- Metrics log: `logs/newton/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_metrics.log`

## Command

The run used the existing tmux-held Curiosity allocation `154023`.

```bash
RUN_TAG=lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

## Gate Evidence

- Fresh official Newton sanity:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_fresh_newton_sensor_contact_sanity.json`
- Visual validation:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_manual_visual_inspection.json`
- Metrics:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_metrics.json`
- Rollout:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545.npz`

Direct visual paths:

- Contact sheet:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545/contact_sheet.png`
- Frame browser:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545/frame_browser.html`
- Key inspected frames:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545/frame_0180.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545/frame_0270.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545/frame_0359.png`

## Result

Visual gate: pass.

Strict metrics: fail, only because `object_accel_above_threshold`.

Metric values:

- lift height: `0.16002337634563446` m;
- hold duration: `2.8333306312561035` s;
- max slip: `0.0035689558514817674` m;
- drop height loss: `0.0` m;
- contact loss frames: `0`;
- max contact proxy: `62.0`;
- max object acceleration: `8.308707937632189` m/s^2.

Scripted feedback summary:

- controller mode: `lift_hold_feedback`;
- learned policy: false;
- curiosity reward: none;
- feedback trigger count: `0`;
- feedback reason labels: only `none`.

## Interpretation

The nominal scripted feedback path is runnable and visually valid, and the
controller fields are present in the rollout NPZ:

- `candidate.controller.feedback_active`;
- `candidate.controller.feedback_reason_id`;
- `candidate.controller.feedback_lift_velocity_scale`;
- `candidate.controller.feedback_hold_height_offset_m`;
- `candidate.controller.feedback_stabilization_extension_s`;
- `candidate.controller.feedback_trigger_count`;
- `candidate.controller.feedback_observed_object_vz_m_s`;
- `candidate.controller.feedback_observed_object_accel_m_s2`.

This run does not prove adaptation improvement because the nominal cup did not
trigger feedback. The next valid Phase 04 step is to run the scripted feedback
baseline across the same mass/friction grid as Phase 02 and compare failures
without hiding the object-acceleration safety failure.
