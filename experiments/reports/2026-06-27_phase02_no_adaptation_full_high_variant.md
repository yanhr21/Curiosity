# Phase 02 No-Adaptation Full/High Cup Variant

## Scope

This is a no-adaptation scripted baseline result for the high-friction cup
variant grid. It is not curiosity, not learned adaptation, not a pretrained
checkpoint result, and not T-Rex schema promotion.

## Variant

```text
cell=full_high
tracked_object=existing_cup_asset
object_mass_kg=0.35
object_friction_mu=1.20
adapter=pre_finalize_builder_body_mass_inertia_and_shape_friction
controller_mode=lift_hold
learned_policy=false
curiosity_reward=none
```

Run tag:

```text
lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430
```

Command:

```text
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430 \
WINDOW_NAME=phase02_mass_variant_full_high \
SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold \
FINAL_HOLD_DURATION=2.5 \
PHYSICS_VARIANT_LABEL=full_high_mass0p35_mu1p20 \
OBJECT_MASS_KG=0.35 OBJECT_FRICTION_MU=1.20 \
BODY_MASS_SCALE=1.0 SHAPE_FRICTION_SCALE=1.0 \
NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

Metrics command:

```text
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
WINDOW_NAME=phase02_full_high_metrics \
MASS_LABEL=full \
FRICTION_LABEL=high \
POSE_SEED=nominal \
MANUAL_VISUAL_INSPECTION=pass \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Evidence

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Files:

```text
log=logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430.log
metrics_log=logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430_metrics.log
summary=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430_summary.json
metrics_json=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430_metrics.json
contact_sheet=experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430/contact_sheet.png
```

Observed physics provenance:

```text
observed_body_mass_kg=0.3499999940395355
observed_shape_material_mu=1.2000000476837158
generated_trex_fields=[]
schema_promotion=blocked
```

Full metrics:

```text
status=fail
lift_height_m=0.15366242825984955
hold_duration_s=3.0666637420654297
max_slip_m=0.003366944785191424
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=63.0
max_object_accel_m_s2=8.308498000056417
failure_reasons=[object_accel_above_threshold]
```

The run passes lift, hold duration, slip, drop, contact-loss, and contact-proxy
gates. It fails the unchanged full metrics schema because object acceleration
exceeds the `8.0 m/s^2` threshold.
