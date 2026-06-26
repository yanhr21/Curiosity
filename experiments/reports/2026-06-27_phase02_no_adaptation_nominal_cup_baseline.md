# Phase 02 No-Adaptation Nominal Cup Baseline

Run tag:

```text
lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915
```

## Purpose

Run the Phase 02 short-term infant-prior baseline on the nominal official cup
asset using the same no-adaptation Newton Panda hydro lift-hold path as the
nominal cube baseline.

This is not a learned policy, not curiosity, not T-Rex schema promotion, and
not a pretrained checkpoint result.

## Command

Launched from the login node as a lightweight tmux/srun submission into the
existing Curiosity allocation:

```text
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915 \
WINDOW_NAME=phase02_noadapt_cup_lifthold \
SCENE=cube \
TRACKED_OBJECT=existing_cup_asset \
CONTROLLER_MODE=lift_hold \
FINAL_HOLD_DURATION=2.5 \
NUM_STEPS=420 \
SAMPLE_STEPS=0,60,120,180,240,300,360,419 \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

Metrics command:

```text
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915 \
WINDOW_NAME=phase02_cup_lifthold_metrics \
MASS_LABEL=nominal \
FRICTION_LABEL=nominal \
POSE_SEED=nominal \
MANUAL_VISUAL_INSPECTION=pass \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

The actual Newton simulation/rendering and metrics extraction ran inside
Slurm allocation `154023` on `server56`.

## Configuration

```text
scene=cube
tracked_object=existing_cup_asset
object_adapter=retarget_existing_official_cup_asset_as_object
controller_mode=lift_hold
controller=official_newton_panda_hydro_scripted_no_adaptation
curiosity_reward=none
learned_policy=false
pretrained_checkpoint=null
```

The cup is the official Newton `manipulation_objects/cup` asset already loaded
by the Panda hydro cube scene. The adapter retargets object bookkeeping and IK
waypoints to that official cup body; it does not create a toy object or any
T-Rex fields.

## Evidence

```text
logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915.log
logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_metrics.log
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_fresh_newton_sensor_contact_sanity.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_summary.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_visual_validation.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_manual_visual_inspection.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_downstream_gate_cleared.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_metrics.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915_metrics.csv
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915/contact_sheet.png
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915/frame_browser.html
```

## Result

```text
fresh_official_newton_sensor_contact_sanity=pass
sensor_tiled_camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
controller_mode=lift_hold
status=fail
lift_height_m=0.16000424325466156
hold_duration_s=4.099996089935303
max_slip_m=0.0034891533600654033
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308498000056417
failure_reasons=[object_accel_above_threshold]
```

The nominal cup clears lift height, hold duration, slip, drop, contact-loss,
and contact-proxy gates. It fails the full stable baseline metric because the
maximum object acceleration is above the schema threshold `8.0m/s^2`.

## Limitations

This is a valid no-adaptation nominal-cup failure case. It should be used as a
target for later scripted feedback adaptation and curiosity-driven residual
control. It does not claim mass/fill generalization, curiosity, tactile
dominance, learned policy behavior, or T-Rex compatibility.

## Non-Claims

- no learned policy;
- no curiosity result;
- no mass/fill generalization result;
- no tactile dominance result;
- no T-Rex schema fields;
- no pretrained checkpoint result.
