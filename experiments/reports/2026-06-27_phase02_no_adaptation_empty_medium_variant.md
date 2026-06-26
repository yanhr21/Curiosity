# Phase 02 No-Adaptation Empty/Medium Cup Variant

## Scope

This is a no-adaptation scripted baseline result for the first real
mass/friction cup variant. It is not curiosity, not learned adaptation, not a
pretrained checkpoint result, and not T-Rex schema promotion.

## Variant

```text
cell=empty_medium
tracked_object=existing_cup_asset
object_mass_kg=0.08
object_friction_mu=0.80
adapter=pre_finalize_builder_body_mass_inertia_and_shape_friction
controller_mode=lift_hold
learned_policy=false
curiosity_reward=none
```

Run tag:

```text
lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140
```

Command:

```text
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140 \
WINDOW_NAME=phase02_mass_variant_empty_medium \
SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold \
FINAL_HOLD_DURATION=2.5 \
PHYSICS_VARIANT_LABEL=empty_medium_mass0p08_mu0p80 \
OBJECT_MASS_KG=0.08 OBJECT_FRICTION_MU=0.80 \
BODY_MASS_SCALE=1.0 SHAPE_FRICTION_SCALE=1.0 \
NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
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
log=logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140.log
summary=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140_summary.json
run_status=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140_run_status.json
visual_validation=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140_visual_validation.json
manual_visual_inspection=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140_manual_visual_inspection.json
metrics_json=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140_metrics.json
metrics_csv=experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140_metrics.csv
contact_sheet=experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140/contact_sheet.png
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
requested_object_mass_kg=0.08
observed_body_mass_kg=0.07999999821186066
original_shape_material_mu=1.0
requested_object_friction_mu=0.80
observed_shape_material_mu=0.800000011920929
generated_trex_fields=[]
schema_promotion=blocked
```

Full metrics:

```text
status=fail
lift_height_m=0.16022799909114838
hold_duration_s=3.099997043609619
max_slip_m=0.003480872071019147
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.307760545609415
failure_reasons=[object_accel_above_threshold]
```

The run passes lift, hold duration, slip, drop, contact-loss, and contact-proxy
gates. It fails the unchanged full metrics schema because object acceleration
exceeds the `8.0 m/s^2` threshold. Do not lower the threshold to convert this
into success.
