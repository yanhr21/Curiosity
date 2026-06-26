# Phase 02 Scripted Feedback Full/High Cup Variant

Date: 2026-06-27

This report records the seventh and final ordinary scripted feedback
mass/friction cell for the official Newton Panda hydro scripted infant prior.
It is not a learned policy, not a curiosity result, and not a pretrained
checkpoint claim.

## Run

```text
run_tag=lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820
slurm_job_id=154023
tmux_session=curiosity_next_source_alloc_20260626_232937
host=server56
tracked_object=existing_cup_asset
scene=cube
controller_mode=lift_hold_feedback
cell=full_high
held_out_generalization_cell=false
object_mass_kg=0.35
object_friction_mu=1.20
physics_variant_label=feedback_full_high_mass0p35_mu1p20
```

## Artifacts

```text
fresh_official_newton_sensor_contact_sanity=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_fresh_newton_sensor_contact_sanity.json
summary_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_summary.json
visual_validation=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_visual_validation.json
manual_visual_inspection=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_manual_visual_inspection.json
npz=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820.npz
metrics_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_metrics.json
metrics_csv=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_metrics.csv
contact_sheet=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820/contact_sheet.png
frame_browser=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820/frame_browser.html
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
observed_shape_material_mu=1.2000000476837158
```

The real Newton pre-finalize physics adapter applied the requested full-mass,
high-friction cell before final model construction and scene replication.

## Metrics

```text
lift_height_m=0.1542350798845291
hold_duration_s=2.7833306789398193
max_slip_m=0.0032414356600358944
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=63.0
max_object_accel_m_s2=8.308127374067027
feedback_trigger_count=0
```

## Interpretation

The `full_high` ordinary scripted feedback cell is runnable, visually valid,
and keeps the cup grasped, lifted, and held. It passes lift, hold, slip, drop,
contact-loss, and contact-proxy gates. The unchanged shared metrics schema
still marks the row as `fail` only because object acceleration exceeds the
strict `8.0 m/s^2` threshold.

This completes ordinary scripted feedback grid evaluation while preserving
`full_low` and `empty_high` as held-out generalization cells. The feedback rule
did not trigger on this cell, so this is an honest scripted feedback evaluation
result, not a learned adaptation success and not evidence that feedback
improved this cell.
