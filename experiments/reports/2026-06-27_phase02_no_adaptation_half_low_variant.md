# Phase 02 No-Adaptation Half/Low Cup Variant

## Scope

This report records a real Newton mass/friction variant baseline for the
low-friction axis. It uses the official Newton Panda hydro scripted grasp/lift
controller with no learned policy, no feedback adaptation, and no curiosity
reward.

This is not a T-Rex schema promotion, not a pretrained-checkpoint claim, and
not a learned-policy result.

## Run

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335
cell=half_low
tracked_object=existing_cup_asset
controller_mode=lift_hold
physics_variant_label=half_low_mass0p20_mu0p35
object_mass_kg=0.20
object_friction_mu=0.35
adapter=pre_finalize_builder_body_mass_inertia_and_shape_friction
slurm_job_id=154023
tmux_session=curiosity_next_source_alloc_20260626_232937
```

Launch command:

```bash
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335 \
WINDOW_NAME=phase02_mass_variant_half_low \
SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold \
FINAL_HOLD_DURATION=2.5 \
PHYSICS_VARIANT_LABEL=half_low_mass0p20_mu0p35 \
OBJECT_MASS_KG=0.20 OBJECT_FRICTION_MU=0.35 \
BODY_MASS_SCALE=1.0 SHAPE_FRICTION_SCALE=1.0 \
NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

Metrics command:

```bash
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335 \
WINDOW_NAME=phase02_half_low_metrics MASS_LABEL=half FRICTION_LABEL=low \
POSE_SEED=nominal MANUAL_VISUAL_INSPECTION=pass \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
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

## Physics Provenance

The adapter changed real Newton builder-stage model fields before scene
replication and before final model finalization.

```text
original_body_mass_kg=0.10100987856276333
requested_object_mass_kg=0.20
observed_body_mass_kg=0.20000000298023224
original_shape_material_mu=1.0
requested_object_friction_mu=0.35
observed_shape_material_mu=0.3499999940395355
generated_trex_fields=[]
schema_promotion=blocked
learned_policy=false
```

## Metrics

```text
status=fail
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

The run passes lift, hold, slip, drop, and contact-loss gates. It fails only
the unchanged object-acceleration threshold. Do not lower the threshold to
convert this into success.

## Interpretation

This is a valid no-adaptation failure baseline for the half/low cup cell. It
continues the low-friction axis while preserving the held-out status of
`full_low`. The repeated acceleration-only failure across medium and low
friction strengthens the first residual-adaptation target: reduce impulsive
object motion during lift/hold without replacing the official scripted grasp
prior.

## Artifacts

```text
logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335.log
logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335_metrics.log
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335_fresh_newton_sensor_contact_sanity.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335_summary.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335_visual_validation.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335_manual_visual_inspection.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335_metrics.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335_metrics.csv
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335.npz
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335/contact_sheet.png
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335/frame_browser.html
```
