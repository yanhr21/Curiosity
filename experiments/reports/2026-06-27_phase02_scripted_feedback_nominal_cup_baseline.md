# Phase 02 Scripted Feedback Nominal Cup Baseline

## Scope

This report records the first scripted feedback adaptation baseline around the
official Newton Panda hydro scripted infant prior.

This is not a learned policy, not curiosity training, not a pretrained
checkpoint result, and not T-Rex schema promotion.

## Run

```text
run_tag=lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545
slurm_job_id=154023
node=server56
tmux_session=curiosity_next_source_alloc_20260626_232937
scene=cube
tracked_object=existing_cup_asset
controller_mode=lift_hold_feedback
physics_variant_label=nominal_feedback
```

The first attempt,
`lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1530`, failed
before simulation because the runner had a shell quoting defect. That defect
was fixed and syntax-checked before the successful `1545` run. The `1530`
attempt is not a physical result.

## Controller

```text
controller_adapter=official_panda_hydro_waypoints_lift_hold_scripted_feedback
learned_policy=false
curiosity_reward=none
pretrained_checkpoint=null
feedback_min_contact_count=20
feedback_accel_threshold_m_s2=6.5
feedback_initial_lift_duration_scale=1.35
feedback_lift_duration_scale_max=2.25
feedback_stabilization_step_s=0.25
feedback_stabilization_max_s=2.0
```

The controller logs feedback evidence under `candidate.controller.*`. The
nominal cup did not trigger the feedback rule:

```text
final_trigger_count=0
```

This is acceptable for the nominal gate: the scripted feedback baseline should
not perturb stable nominal behavior without a detected mismatch.

## Gates

```text
fresh_official_newton_sensor_contact_sanity=pass
sensor_tiled_camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Manual visual inspection checked the contact sheet and frame browser. The
sampled frames are nonblank, show the Panda approaching, grasping, lifting, and
holding the official cup asset, and show no obvious visual drop through frame
359.

## Metrics

```text
lift_height_m=0.16002337634563446
hold_duration_s=2.8333306312561035
max_slip_m=0.0035689558514817674
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308707937632189
failure_reasons=[object_accel_above_threshold]
```

The nominal scripted feedback baseline passes lift, hold, slip, drop,
contact-loss, and contact-proxy gates. It fails the unchanged full schema only
because object acceleration exceeds the strict threshold `8.0`. Do not lower
the threshold to make this pass.

## Evidence

```text
summary=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_summary.json
sanity=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_fresh_newton_sensor_contact_sanity.json
visual_validation=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_visual_validation.json
manual_visual_inspection=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_manual_visual_inspection.json
metrics_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_metrics.json
metrics_csv=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545_metrics.csv
frame_browser=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545/frame_browser.html
contact_sheet=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545/contact_sheet.png
```

## Next Step

Run the scripted feedback baseline over ordinary mass/friction cells while
preserving `full_low` and `empty_high` as held-out generalization evidence.
