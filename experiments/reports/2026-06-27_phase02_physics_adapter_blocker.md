# Phase 02 Physics Adapter Blocker

## Scope

This is a blocker report for the real mass/friction variant adapter. It is not
a learned-policy result, not curiosity, not T-Rex schema promotion, and not a
mass/fill baseline result.

## What Worked

Diagnostic run:

```text
physics_variant_adapter_sanity_cup_mass15_friction06_20260627_0945
```

The export summary showed real Newton model values changed:

```text
body_mass_scale=1.5
shape_friction_scale=0.6
original_body_mass=0.10100987553596497
updated_body_mass=0.15151481330394745
original_shape_material_mu=1.0
updated_shape_material_mu=0.6000000238418579
generated_trex_fields=[]
schema_promotion=blocked
```

This confirms the adapter was not merely changing labels.

## What Failed

The `0945` run sampled only four frames and failed visual validation:

```text
VisualPreviewValidationError: only 4 frames, expected at least 5
```

Follow-up five-frame runs repeatedly failed with Warp CUDA illegal memory
access during SensorTiledCamera/export cleanup:

```text
lift_hold_physics_adapter_sanity_existing_cup_mass2p5_friction0p5_20260627_0935
physics_variant_adapter_sanity_cup_mass15_friction06_20260627_0955
physics_variant_adapter_sanity_cup_mass15_friction06_20260627_1015
```

Observed error class:

```text
Warp CUDA error 700: an illegal memory access was encountered
```

## Decision

Stop the runtime model-array mutation path. The next implementation must apply
mass/inertia/friction changes before model finalization in the official Panda
hydro builder path, or use a documented Newton model-update API compatible
with the collision pipeline and SensorTiledCamera.

Do not run or report mass/fill variants until the replacement adapter passes:

```text
fresh official Newton sanity
SensorTiledCamera export
visual validation
manual visual inspection
metrics extraction
```

## Follow-Up Resolution

The runtime mutation blocker is superseded by the pre-finalize builder adapter
sanity run:

```text
physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125
```

This replacement path passed fresh official Newton sanity, SensorTiledCamera
export, automated visual validation, and manual visual inspection. It observed
the requested values in the final Newton model:

```text
observed_body_mass_kg=0.15151481330394745
observed_shape_material_mu=0.6000000238418579
generated_trex_fields=[]
schema_promotion=blocked
```

The new evidence is recorded in:

```text
experiments/reports/2026-06-27_phase02_physics_prefinalize_adapter_sanity.md
```
