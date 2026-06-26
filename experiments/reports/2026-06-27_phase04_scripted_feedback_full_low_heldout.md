# Phase 04 Scripted Feedback Full/Low Held-Out Evaluation

Date: 2026-06-27

This report records the `full_low` held-out scripted feedback evaluation cell
for the official Newton Panda hydro scripted infant prior. It is not a training
cell, not a learned policy, not a curiosity result, and not a pretrained
checkpoint claim.

## Run

```text
run_tag=lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845
slurm_job_id=154023
tmux_session=curiosity_next_source_alloc_20260626_232937
host=server56
tracked_object=existing_cup_asset
scene=cube
controller_mode=lift_hold_feedback
cell=full_low
held_out_generalization_cell=true
object_mass_kg=0.35
object_friction_mu=0.35
physics_variant_label=feedback_full_low_heldout_mass0p35_mu0p35
```

## Artifacts

```text
fresh_official_newton_sensor_contact_sanity=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845_fresh_newton_sensor_contact_sanity.json
summary_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845_summary.json
visual_validation=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845_visual_validation.json
manual_visual_inspection=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845_manual_visual_inspection.json
npz=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845.npz
metrics_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845_metrics.json
metrics_csv=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845_metrics.csv
contact_sheet=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845/contact_sheet.png
frame_browser=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845/frame_browser.html
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
observed_body_mass_kg=0.3499999940395355
original_shape_material_mu=1.0
observed_shape_material_mu=0.3499999940395355
```

## Metrics

```text
lift_height_m=0.15313686430454254
hold_duration_s=2.7833306789398193
max_slip_m=0.0034078387381632435
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308707788010144
feedback_trigger_count=0
```

## Interpretation

The `full_low` held-out scripted feedback cell is runnable, visually valid,
and keeps the cup grasped, lifted, and held. It passes lift, hold, slip, drop,
contact-loss, and contact-proxy gates. The unchanged shared metrics schema
still marks the row as `fail` only because object acceleration exceeds the
strict `8.0 m/s^2` threshold.

The feedback rule did not trigger, so this is held-out evaluation evidence, not
a learned adaptation success and not evidence that scripted feedback improved
the held-out cell.
