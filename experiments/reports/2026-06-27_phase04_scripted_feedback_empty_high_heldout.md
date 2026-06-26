# Phase 04 Scripted Feedback Empty/High Held-Out Evaluation

Date: 2026-06-27

This report records the `empty_high` held-out scripted feedback evaluation cell
for the official Newton Panda hydro scripted infant prior. It is not a training
cell, not a learned policy, not a curiosity result, and not a pretrained
checkpoint claim.

## Run

```text
run_tag=lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955
slurm_job_id=154023
tmux_session=curiosity_next_source_alloc_20260626_232937
host=server56
tracked_object=existing_cup_asset
scene=cube
controller_mode=lift_hold_feedback
cell=empty_high
held_out_generalization_cell=true
object_mass_kg=0.08
object_friction_mu=1.20
physics_variant_label=feedback_empty_high_heldout_mass0p08_mu1p20
```

## Artifacts

```text
fresh_official_newton_sensor_contact_sanity=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955_fresh_newton_sensor_contact_sanity.json
summary_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955_summary.json
visual_validation=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955_visual_validation.json
manual_visual_inspection=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955_manual_visual_inspection.json
npz=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955.npz
metrics_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955_metrics.json
metrics_csv=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955_metrics.csv
contact_sheet=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955/contact_sheet.png
frame_browser=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955/frame_browser.html
```

## Sanity And Gates

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
failure_reasons=[object_accel_above_threshold]
```

## Physics Provenance

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.07999999821186066
original_shape_material_mu=1.0
observed_shape_material_mu=1.2000000476837158
```

## Metrics

```text
lift_height_m=0.16016103327274323
hold_duration_s=2.8333306312561035
max_slip_m=0.0035689078921667837
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308498000056417
feedback_trigger_count=0
```

## Interpretation

The `empty_high` held-out scripted feedback cell is runnable, visually valid,
and keeps the cup grasped, lifted, and held. It passes lift, hold, slip, drop,
contact-loss, and contact-proxy gates. The unchanged shared metrics schema
still marks the row as `fail` only because object acceleration exceeds the
strict `8.0 m/s^2` threshold.

This completes the scripted feedback held-out evaluation. The feedback rule
did not trigger on either held-out cell, so this is held-out evaluation
evidence, not a learned adaptation success and not evidence that scripted
feedback improved generalization.
