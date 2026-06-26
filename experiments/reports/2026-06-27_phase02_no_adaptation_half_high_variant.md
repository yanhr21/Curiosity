# Phase 02 No-Adaptation Half/High Cup Variant

## Scope

This is a no-adaptation scripted baseline result for the high-friction cup
variant grid. It is not curiosity, not learned adaptation, not a pretrained
checkpoint result, and not T-Rex schema promotion.

## Variant

```text
cell=half_high
tracked_object=existing_cup_asset
object_mass_kg=0.20
object_friction_mu=1.20
adapter=pre_finalize_builder_body_mass_inertia_and_shape_friction
controller_mode=lift_hold
learned_policy=false
curiosity_reward=none
```

Run tag:

```text
lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415
```

Command:

```text
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415 \
WINDOW_NAME=phase02_mass_variant_half_high \
SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold \
FINAL_HOLD_DURATION=2.5 \
PHYSICS_VARIANT_LABEL=half_high_mass0p20_mu1p20 \
OBJECT_MASS_KG=0.20 OBJECT_FRICTION_MU=1.20 \
BODY_MASS_SCALE=1.0 SHAPE_FRICTION_SCALE=1.0 \
NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

Metrics command:

```text
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
WINDOW_NAME=phase02_half_high_metrics \
MASS_LABEL=half \
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
log=logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415.log
metrics_log=logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415_metrics.log
summary=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415_summary.json
metrics_json=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415_metrics.json
contact_sheet=experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415/contact_sheet.png
```

Observed physics provenance:

```text
observed_body_mass_kg=0.20000000298023224
observed_shape_material_mu=1.2000000476837158
generated_trex_fields=[]
schema_promotion=blocked
```

Full metrics:

```text
status=fail
lift_height_m=0.15865357220172882
hold_duration_s=3.099997043609619
max_slip_m=0.003593952800185949
object_not_dropped=true
drop_height_loss_m=0.0000010132789611816406
contact_loss_frames=0
max_contact_proxy=61.0
max_object_accel_m_s2=8.308498000056417
failure_reasons=[object_accel_above_threshold]
```

The run passes lift, hold duration, slip, drop, contact-loss, and contact-proxy
gates. It fails the unchanged full metrics schema because object acceleration
exceeds the `8.0 m/s^2` threshold.
