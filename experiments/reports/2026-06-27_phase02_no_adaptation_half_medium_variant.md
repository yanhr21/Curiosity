# Phase 02 No-Adaptation Half/Medium Cup Variant

## Status

This report records a real Newton mass/friction variant baseline for the
short-term infant-prior route. It uses the official Newton Panda hydro scripted
grasp/lift controller with no learned policy, no curiosity reward, and no
feedback adaptation.

This is not a T-Rex schema promotion, not a pretrained-checkpoint claim, and
not a learned-policy result.

## Run

```text
run_tag=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155
cell=half_medium
tracked_object=existing_cup_asset
controller_mode=lift_hold
physics_variant_label=half_medium_mass0p20_mu0p80
object_mass_kg=0.20
object_friction_mu=0.80
adapter=pre_finalize_builder_body_mass_inertia_and_shape_friction
slurm_job_id=154023
tmux_session=curiosity_next_source_alloc_20260626_232937
```

Launch command:

```bash
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155 \
WINDOW_NAME=phase02_mass_variant_half_medium \
SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold \
FINAL_HOLD_DURATION=2.5 \
PHYSICS_VARIANT_LABEL=half_medium_mass0p20_mu0p80 \
OBJECT_MASS_KG=0.20 OBJECT_FRICTION_MU=0.80 \
BODY_MASS_SCALE=1.0 SHAPE_FRICTION_SCALE=1.0 \
NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

Metrics command:

```bash
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155 \
WINDOW_NAME=phase02_half_medium_metrics MASS_LABEL=half FRICTION_LABEL=medium \
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
requested_object_friction_mu=0.80
observed_shape_material_mu=0.800000011920929
generated_trex_fields=[]
schema_promotion=blocked
learned_policy=false
```

## Metrics

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

The run passes lift, hold, slip, drop, and contact-loss gates. It fails only
the strict object-acceleration threshold. Do not lower the threshold to convert
this into success.

## Interpretation

This is a valid no-adaptation failure baseline for the half/medium cup cell.
Together with the empty/medium result, it supports the selected short-term
route: keep the official scripted Newton grasp/lift prior, then learn residual
controller adaptation that reduces impulsive object motion during lift/hold.

## Artifacts

```text
logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155.log
logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155_metrics.log
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155_fresh_newton_sensor_contact_sanity.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155_summary.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155_visual_validation.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155_manual_visual_inspection.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155_metrics.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155_metrics.csv
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155.npz
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155/contact_sheet.png
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155/frame_browser.html
```
