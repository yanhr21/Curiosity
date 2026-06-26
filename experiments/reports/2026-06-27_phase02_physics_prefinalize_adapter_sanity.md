# Phase 02 Pre-Finalize Physics Adapter Sanity

## Scope

This report records a Newton-native physics adapter diagnostic. It is not a
learned-policy result, not curiosity, not T-Rex schema promotion, and not a
formal mass/fill baseline.

## Run

```text
run_tag=physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125
allocation=154023
tmux=curiosity_next_source_alloc_20260626_232937
scene=cube
tracked_object=existing_cup_asset
controller_mode=lift_hold
num_steps=180
sample_steps=0,45,90,135,179
physics_variant_label=prefinalize_builder_mass15_friction06
body_mass_scale=1.5
shape_friction_scale=0.6
```

## Implementation

The runtime finalized-model mutation path is stopped. The replacement adapter
hooks only during official Newton Panda hydro `Example` construction and
changes the builder before `scene.replicate(...)` and before final model
`finalize()`.

Changed fields:

```text
builder.body_mass[target_body]
builder.body_inv_mass[target_body]
builder.body_inertia[target_body]
builder.body_inv_inertia[target_body]
builder.shape_material_mu[target_shape]
```

The code lives in:

```text
experiments/configs/newton_panda_hydro_tiled_camera_export.py
```

The launch path now passes:

```text
PHYSICS_VARIANT_LABEL
BODY_MASS_SCALE
SHAPE_FRICTION_SCALE
OBJECT_MASS_KG
OBJECT_FRICTION_MU
```

## Command

```text
RUN_TAG=physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
WINDOW_NAME=phase02_prefinalize_adapter_sanity_v2 \
SCENE=cube \
TRACKED_OBJECT=existing_cup_asset \
CONTROLLER_MODE=lift_hold \
FINAL_HOLD_DURATION=2.0 \
PHYSICS_VARIANT_LABEL=prefinalize_builder_mass15_friction06 \
BODY_MASS_SCALE=1.5 \
SHAPE_FRICTION_SCALE=0.6 \
NUM_STEPS=180 \
SAMPLE_STEPS=0,45,90,135,179 \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

## Result

```text
fresh_official_newton_sensor_contact_sanity=pass
sensor_tiled_camera_export=pass
visual_validation=pass
manual_visual_inspection=pass_for_physics_adapter_diagnostic
run_status=pass_downstream_blocked
```

The adapter applies mass/inertia/friction before the official Panda hydro scene
replicates the builder and before the final Newton model is finalized. The
observed final model values confirm that this is not label-only provenance:

```text
adapter=pre_finalize_builder_body_mass_inertia_and_shape_friction
builder_stage=before_scene_replicate_and_before_final_model_finalize
body_label=cup
body_index_local=15
observed_body_index_final_model=15
original_body_mass_kg=0.10100987856276333
updated_body_mass_kg=0.151514817844145
observed_body_mass_kg=0.15151481330394745
original_shape_material_mu=1.0
updated_shape_material_mu=0.6
observed_shape_material_mu=0.6000000238418579
generated_trex_fields=[]
schema_promotion=blocked
```

The short diagnostic lifted the object but did not satisfy the formal hold
duration threshold because it ran only 180 steps:

```text
max_lift=0.13472603261470795
longest_hold_s=0.1
failure_reasons=[hold_duration_below_min]
```

This is acceptable for adapter sanity. It must not be reported as a formal
mass/fill baseline success.

## Evidence

```text
logs/newton/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125.log
experiments/outputs/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125_fresh_newton_sensor_contact_sanity.json
experiments/outputs/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125_summary.json
experiments/outputs/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125_visual_validation.json
experiments/outputs/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125_manual_visual_inspection.json
experiments/outputs/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125_downstream_gate_cleared.json
experiments/outputs/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125.npz
experiments/visuals/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125/contact_sheet.png
experiments/visuals/physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125/frame_browser.html
```

## Decision

The runtime post-finalize mutation route remains rejected. The replacement
pre-finalize builder adapter is cleared for launching formal mass/friction
variant baselines under `candidate.physics.*` provenance.
