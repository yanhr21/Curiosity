# Phase 02 Scripted Feedback Empty/Medium Cup Variant

## Scope

This report records a scripted feedback adaptation baseline for the ordinary
`empty_medium` mass/friction cell. It uses the official Newton Panda hydro
scripted infant prior plus deterministic feedback controller logic.

This is not a learned policy, not curiosity training, not a pretrained
checkpoint result, and not T-Rex schema promotion. This is not a held-out
generalization cell.

## Run

```text
run_tag=lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635
slurm_job_id=154023
node=server56
tmux_session=curiosity_next_source_alloc_20260626_232937
scene=cube
tracked_object=existing_cup_asset
controller_mode=lift_hold_feedback
physics_variant_label=feedback_empty_medium_mass0p08_mu0p80
object_mass_kg=0.08
object_friction_mu=0.80
held_out_generalization_cell=false
```

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
holding the cup, and show no obvious visual drop through frame 359.

## Physics Provenance

```text
adapter=pre_finalize_builder_body_mass_inertia_and_shape_friction
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.07999999821186066
original_shape_material_mu=1.0
observed_shape_material_mu=0.800000011920929
```

The real Newton builder-stage mass/friction adapter was applied before scene
replication and final model finalization.

## Feedback

```text
feedback_enabled=true
feedback_trigger_count=0
learned_policy=false
curiosity_reward=none
```

The feedback rule did not trigger on this cell. This means the current
scripted thresholds did not detect a mismatch requiring intervention, not that
the run is a learned adaptation success.

## Metrics

```text
lift_height_m=0.160252645611763
hold_duration_s=2.8333306312561035
max_slip_m=0.0035626504907293466
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308392358032673
failure_reasons=[object_accel_above_threshold]
```

The run passes lift, hold, slip, drop, contact-loss, and contact-proxy gates.
It fails the unchanged full schema only because object acceleration exceeds the
strict threshold `8.0`. Do not lower the threshold to make this pass.

## Evidence

```text
summary=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_summary.json
sanity=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_fresh_newton_sensor_contact_sanity.json
visual_validation=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_visual_validation.json
manual_visual_inspection=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_manual_visual_inspection.json
metrics_json=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_metrics.json
metrics_csv=experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_metrics.csv
frame_browser=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635/frame_browser.html
contact_sheet=experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635/contact_sheet.png
```

## Next Step

Continue scripted feedback evaluation on the remaining ordinary cells:
`half_low`, `half_medium`, `half_high`, `full_medium`, and `full_high`.
Preserve `full_low` and `empty_high` as held-out generalization evidence.
