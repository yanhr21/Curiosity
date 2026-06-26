# Phase 02: Baselines

## Goal

Establish serious baselines before evaluating curiosity.

## Baselines

1. No-adaptation scripted grasp-and-lift.
2. Scripted feedback adaptation.
3. Behavior cloning, diffusion-policy-style, ACT-style, or another documented
   manipulation baseline if demonstrations are available.
4. Newton-native contact-aware diagnostic baseline.

## Basic Infant Prior

Current decision: use the official Newton Panda hydro scripted grasp/lift path
as the short-term "infant" prior. Do not wait for a pretrained grasping
checkpoint before building the first baselines. This prior provides approach,
gripper close, lift, and hold behavior so later learning can focus on residual
adaptation rather than discovering grasping from scratch.

User-approved decision as of 2026-06-27: this is the active short-term stable
method. The immediate plan is to continue from the Newton scripted infant
prior into scripted feedback adaptation and then residual controller-parameter
learning. Pretrained checkpoints remain audit candidates only; they do not
block the current baseline/adaptation path and must not be claimed unless a
compatible official policy passes the same sanity, visual, and metric gates.

Short-term stable method:

1. Keep official Newton Panda hydro scripted grasp/lift as the non-learned
   infant prior.
2. Use it to generate synchronized Newton episodes with camera, robot state,
   object motion, contact/slip proxies, controller phase, and success/failure
   labels.
3. Fix physics variation honestly before reporting mass/fill results: object
   mass/inertia/friction must be changed in the real Newton model and recorded
   in provenance.
4. Learn only residual controller parameters first, such as grip target, lift
   velocity, hold height, regrasp threshold, and stabilization timing.
5. Add curiosity only after forward-model diagnostics can predict
   object/contact/tactile consequences better than trivial baselines.

Reason: the current audit did not find a directly usable Newton-native Panda
grasp/lift checkpoint. Newton policy examples are mainly locomotion-oriented,
Isaac Sim exposes a Franka open-drawer policy rather than a cup/cube grasp
prior, Isaac Lab Mimic provides data-generation and training routes, and OpenPI
DROID/Franka checkpoints require a substantial observation/action adapter
before they can honestly be treated as Newton-compatible.

Checkpoint-based priors are allowed only after a source audit records:

- official code repository and commit;
- checkpoint URL/path and license;
- robot embodiment and gripper/action semantics;
- camera and proprioception requirements;
- whether the checkpoint can control Franka/Panda or can be adapted only
  through a documented controller interface;
- a smoke test command and direct visual evidence.

OpenPI/pi0, diffusion-policy-style checkpoints, ACT-style checkpoints, or
T-Rex-style checkpoints are candidates only if the observation/action contract
is compatible or the adapter is scientifically explicit. Do not treat a
generic robot checkpoint as Newton grasp success until it runs through the
same visual and metric gates as the scripted baseline.

## Training Data Contract

Baseline datasets should be synchronized Newton episodes containing robot
state, controller command, object pose/velocity, contact proxy, camera RGB-D,
task phase, success/failure labels, and later tactile/contact evidence under
source-preserving namespaces.

Held-out cells such as full low-friction and empty high-friction cups must stay
out of training and be used for generalization tests.

## No-Adaptation Baseline V1

The first configured baseline is:

```text
experiments/configs/lift_hold_no_adaptation_baseline_v1.json
```

It uses:

```text
scene=cube
tracked_object=official_object
controller_mode=lift_hold
controller=official_newton_panda_hydro_scripted_no_adaptation
curiosity_reward=none
learned_policy=false
pretrained_checkpoint=null
```

Launch path:

```text
JOB_ID=<held_curiosity_allocation> \
TMUX_SESSION=<curiosity_tmux_session> \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

The launcher reuses the existing Panda hydro camera export runner, which
reruns fresh official Newton `sensor_contact` sanity inside the allocation,
exports the namespaced rollout and camera evidence, and keeps downstream use
blocked until manual visual inspection passes. The Phase 02 baseline now uses
`controller_mode=lift_hold`: it preserves the official approach/grasp/lift
waypoints but disables the release/place segment and holds the lifted pose.

The exporter now records controller provenance:

```text
candidate.controller.phase_index
candidate.controller.commanded_gripper_target
candidate.controller.commanded_lift_target
```

Shared metrics schema:

```text
experiments/configs/lift_hold_metrics_schema_v1.json
```

This is still not a baseline result until the compute run, visual validation,
manual inspection, and metrics report exist.

## Nominal Official Cube Baseline Result

The first no-adaptation baseline evidence run has completed for the official
Newton cube object:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210
report=experiments/reports/2026-06-27_phase02_no_adaptation_nominal_baseline.md
```

Result:

```text
fresh_official_newton_sensor_contact_sanity=pass
sensor_tiled_camera_export=pass
visual_validation=pass
manual_visual_inspection=pass_with_metric_limitations
metrics_extractor=pass_as_tool
baseline_status=fail
lift_height_m=0.22359015047550201
hold_duration_s=1.3166654109954834
max_slip_m=0.09295262564260072
failure_reasons=[hold_duration_below_threshold, slip_above_threshold]
```

This clears the nominal official-cube no-adaptation evidence gate and provides
a valid failure baseline. It does not clear nominal cup success, mass/fill
variants, curiosity, or learned policy claims.

The metrics extractor is implemented as:

```text
experiments/configs/extract_lift_hold_metrics.py
experiments/configs/launch_lift_hold_metrics_tmux.sh
experiments/configs/run_lift_hold_metrics_in_alloc.sh
```

## Nominal Official Cube Lift-Hold Success Result

The corrected lift-hold no-adaptation run has completed for the official
Newton cube object:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_lifthold_v2_20260627_0255
```

Result:

```text
fresh_official_newton_sensor_contact_sanity=pass
sensor_tiled_camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass
baseline_status=success
controller_mode=lift_hold
lift_height_m=0.22698602825403214
hold_duration_s=4.316662549972534
max_slip_m=0.007660537484248558
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=98
max_object_accel_m_s2=4.997248082381072
```

This supplies the required nominal official-cube no-adaptation success case.
It does not claim cup success, mass/fill generalization, curiosity, adaptation,
learned policy behavior, tactile dominance, or T-Rex schema promotion.

## Nominal Official Cup Lift-Hold Result

The no-adaptation baseline has completed for the official Newton cup asset
retargeted from the Panda hydro cube scene:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915
tracked_object=existing_cup_asset
object_adapter=retarget_existing_official_cup_asset_as_object
```

Result:

```text
fresh_official_newton_sensor_contact_sanity=pass
sensor_tiled_camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
controller_mode=lift_hold
lift_height_m=0.16000424325466156
hold_duration_s=4.099996089935303
max_slip_m=0.0034891533600654033
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308498000056417
failure_reasons=[object_accel_above_threshold]
```

This is a valid nominal-cup no-adaptation failure case under the same metrics
schema. It passes lift, hold, slip, drop, and contact-loss gates but fails the
unstable object acceleration threshold. Do not lower the threshold to convert
this into success.

## Mass/Fill Variant Readiness

The task specification already defines empty/half/full and low/medium/high
cells, but the current exporter does not yet expose a verified physics
parameter adapter for mass, inertia, or contact friction. Therefore mass/fill
variants must not be run by changing only report labels such as `MASS_LABEL`
or `FRICTION_LABEL`.

Requirement: mass/fill variants must use a real Newton parameter adapter that
changes the tracked object's mass/inertia and contact friction, records the
requested and observed parameters in the rollout summary, and passes the same
official sanity/export/visual gate before any formal mass/fill variant is
reported.

Runtime mutation attempt status on 2026-06-27:

```text
physics_variant_adapter_sanity_cup_mass15_friction06_20260627_0945
```

This diagnostic changed real Newton model values and recorded provenance:

```text
body_mass_scale=1.5
shape_friction_scale=0.6
original_body_mass=0.10100987553596497
updated_body_mass=0.15151481330394745
original_shape_material_mu=1.0
updated_shape_material_mu=0.6000000238418579
```

However, this path is not cleared. The first diagnostic sampled only four
frames and failed the visual validator's five-frame minimum. Follow-up
five-frame diagnostics repeatedly failed with Warp CUDA illegal memory access
during SensorTiledCamera/export cleanup:

```text
lift_hold_physics_adapter_sanity_existing_cup_mass2p5_friction0p5_20260627_0935
physics_variant_adapter_sanity_cup_mass15_friction06_20260627_0955
physics_variant_adapter_sanity_cup_mass15_friction06_20260627_1015
```

Stop this runtime model-array mutation route. The next implementation should
apply mass/inertia/friction changes before model finalization in the official
Panda hydro builder path, or use a documented Newton model-update API that is
compatible with the collision pipeline and SensorTiledCamera. Do not run
mass/fill variants until that adapter passes the fresh official sanity/export
visual gate.

Pre-finalize builder adapter status on 2026-06-27:

```text
physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125
```

This replacement adapter applies the change before `scene.replicate(builder,
world_count)` and before the final Newton model is finalized. It passed fresh
official Newton sanity, SensorTiledCamera export, automated visual validation,
and manual visual inspection. Final model observations prove the requested
physics reached the simulation model:

```text
body_mass_scale=1.5
shape_friction_scale=0.6
original_body_mass_kg=0.10100987856276333
updated_body_mass_kg=0.151514817844145
observed_body_mass_kg=0.15151481330394745
original_shape_material_mu=1.0
updated_shape_material_mu=0.6
observed_shape_material_mu=0.6000000238418579
generated_trex_fields=[]
schema_promotion=blocked
```

This clears the adapter readiness gate for launching formal mass/friction
variant baselines under `candidate.physics.*` provenance. It does not itself
count as a formal mass/fill baseline because the diagnostic ran only 180 steps
and failed the formal hold-duration threshold.

## Empty/Medium Cup Variant Result

The first real mass/friction variant baseline has completed:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140
cell=empty_medium
object_mass_kg=0.08
object_friction_mu=0.80
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.07999999821186066
original_shape_material_mu=1.0
observed_shape_material_mu=0.800000011920929
```

Full metrics:

```text
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

This is a valid no-adaptation failure baseline for the empty/medium cup cell.
It does not complete the full mass/friction grid.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_empty_medium_variant.md
```

## Half/Medium Cup Variant Result

The second real mass/friction variant baseline has completed:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155
cell=half_medium
object_mass_kg=0.20
object_friction_mu=0.80
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.20000000298023224
original_shape_material_mu=1.0
observed_shape_material_mu=0.800000011920929
```

Full metrics:

```text
lift_height_m=0.15677814185619354
hold_duration_s=3.0833303928375244
max_slip_m=0.003278913129597797
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308443857335977
failure_reasons=[object_accel_above_threshold]
```

This is a valid no-adaptation failure baseline for the half/medium cup cell.
It extends the mass axis but does not complete the full mass/friction grid.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_half_medium_variant.md
```

## Full/Medium Cup Variant Result

The third real mass/friction variant baseline has completed:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_full_medium_prefinalize_20260627_1205
cell=full_medium
object_mass_kg=0.35
object_friction_mu=0.80
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.3499999940395355
original_shape_material_mu=1.0
observed_shape_material_mu=0.800000011920929
```

Full metrics:

```text
lift_height_m=0.15296600759029388
hold_duration_s=3.049997091293335
max_slip_m=0.0031441027020740002
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=61.0
max_object_accel_m_s2=8.308498000056417
failure_reasons=[object_accel_above_threshold]
```

This completes the medium-friction mass axis for no-adaptation baselines:
empty, half, and full all pass lift/hold/slip/drop/contact gates and all fail
the unchanged full schema only on object acceleration. The remaining grid cells
are low/high friction cells and held-out generalization cells.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_full_medium_variant.md
```

## Empty/Low Cup Variant Result

The first low-friction real mass/friction variant baseline has completed:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_low_prefinalize_20260627_1320
cell=empty_low
object_mass_kg=0.08
object_friction_mu=0.35
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.07999999821186066
original_shape_material_mu=1.0
observed_shape_material_mu=0.3499999940395355
```

Full metrics:

```text
lift_height_m=0.1602293699979782
hold_duration_s=3.099997043609619
max_slip_m=0.00348078277085327
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=61.0
max_object_accel_m_s2=8.308707937632189
failure_reasons=[object_accel_above_threshold]
```

This is a valid no-adaptation failure baseline for the empty/low cup cell.
It starts the low-friction axis while preserving the held-out status of
`full_low` and `empty_high`. It passes lift, hold, slip, drop, and contact-loss
gates, and fails the unchanged full schema only on object acceleration.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_empty_low_variant.md
```

## Half/Low Cup Variant Result

The second low-friction real mass/friction variant baseline has completed:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335
cell=half_low
object_mass_kg=0.20
object_friction_mu=0.35
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.20000000298023224
original_shape_material_mu=1.0
observed_shape_material_mu=0.3499999940395355
```

Full metrics:

```text
lift_height_m=0.15679897367954254
hold_duration_s=3.0833303928375244
max_slip_m=0.003276861798514688
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308498000056417
failure_reasons=[object_accel_above_threshold]
```

This is a valid no-adaptation failure baseline for the half/low cup cell.
It continues the low-friction axis while preserving the held-out status of
`full_low`. It passes lift, hold, slip, drop, and contact-loss gates, and fails
the unchanged full schema only on object acceleration.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_half_low_variant.md
```

## Full/Low Held-Out Cup Variant Result

The full/low held-out generalization cell has completed as a no-adaptation
evaluation:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_full_low_prefinalize_20260627_1350
cell=full_low
held_out_generalization_cell=true
object_mass_kg=0.35
object_friction_mu=0.35
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.3499999940395355
original_shape_material_mu=1.0
observed_shape_material_mu=0.3499999940395355
```

Full metrics:

```text
lift_height_m=0.15308110415935516
hold_duration_s=3.0666637420654297
max_slip_m=0.0034788257913540574
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308390712127508
failure_reasons=[object_accel_above_threshold]
```

This is a valid held-out no-adaptation evaluation result. It passes lift, hold,
slip, drop, and contact-loss gates, and fails the unchanged full schema only on
object acceleration. It must remain labeled as held-out evidence and not as a
training/grid-completion cell for later learned adaptation.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_full_low_variant.md
```

## Half/High Cup Variant Result

The first ordinary high-friction real mass/friction variant baseline has
completed:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415
cell=half_high
object_mass_kg=0.20
object_friction_mu=1.20
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.20000000298023224
original_shape_material_mu=1.0
observed_shape_material_mu=1.2000000476837158
```

Full metrics:

```text
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

This is a valid no-adaptation failure baseline for the half/high cup cell.
It starts the ordinary high-friction axis while preserving the held-out status
of `empty_high`. It passes lift, hold, slip, drop, and contact-loss gates, and
fails the unchanged full schema only on object acceleration.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_half_high_variant.md
```

## Full/High Cup Variant Result

The second ordinary high-friction real mass/friction variant baseline has
completed:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430
cell=full_high
object_mass_kg=0.35
object_friction_mu=1.20
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.3499999940395355
original_shape_material_mu=1.0
observed_shape_material_mu=1.2000000476837158
```

Full metrics:

```text
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

This is a valid no-adaptation failure baseline for the full/high cup cell.
It completes the ordinary mass/friction grid while preserving `full_low` and
`empty_high` as held-out generalization cells. It passes lift, hold, slip,
drop, and contact-loss gates, and fails the unchanged full schema only on
object acceleration.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_full_high_variant.md
```

## Empty/High Held-Out Cup Variant Result

The empty/high held-out generalization cell has completed as a no-adaptation
evaluation:

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_high_prefinalize_20260627_1445
cell=empty_high
held_out_generalization_cell=true
object_mass_kg=0.08
object_friction_mu=1.20
```

Gate results:

```text
fresh_official_newton_sensor_contact_sanity=pass
camera_export=pass
visual_validation=pass
manual_visual_inspection=pass
metrics_extractor=pass_as_tool
baseline_status=fail
```

Observed physics provenance:

```text
original_body_mass_kg=0.10100987856276333
observed_body_mass_kg=0.07999999821186066
original_shape_material_mu=1.0
observed_shape_material_mu=1.2000000476837158
```

Full metrics:

```text
lift_height_m=0.16015110909938812
hold_duration_s=3.099997043609619
max_slip_m=0.003486471675153751
object_not_dropped=true
drop_height_loss_m=0.0
contact_loss_frames=0
max_contact_proxy=62.0
max_object_accel_m_s2=8.308498000056417
failure_reasons=[object_accel_above_threshold]
```

This completes the 3x3 no-adaptation physics-variant evaluation grid. All cells
pass fresh official Newton sanity, camera export, visual validation, manual
inspection, and full metrics extraction. All cells pass lift/hold/slip/drop and
contact gates and fail the unchanged full schema only on object acceleration.
`full_low` and `empty_high` remain labeled as held-out evidence for later
learned adaptation comparisons.

Report:

```text
experiments/reports/2026-06-27_phase02_no_adaptation_empty_high_variant.md
```

## Rules

- Do not introduce toy T-Rex, toy VQ-VAE, toy Transformer, or toy world model.
- Any small diagnostic model must be explicitly labeled as Newton-native
  diagnostic.
- Every baseline must report identical metrics.

## Completion Criteria

- Baseline commands exist.
- Metrics table format exists.
- At least one visual success and one failure case are saved.
- A checkpoint audit exists if any pretrained policy is used.
