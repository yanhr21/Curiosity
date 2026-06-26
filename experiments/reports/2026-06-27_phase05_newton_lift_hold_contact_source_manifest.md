# Phase 05 Newton Lift-Hold Contact Source Manifest

Date: 2026-06-27

This report records a namespace-preserving conversion from real Newton
lift-hold rollouts into a contact/tactile source manifest. It is not a T-Rex
dataset, not a learned policy, not a calibrated F6 tactile source, and not a
dense tactile deformation source.

## Run

```text
slurm_job_id=154023
tmux_session=curiosity_next_source_alloc_20260626_232937
host=server56
config=experiments/configs/newton_lift_hold_contact_source_manifest_v1.json
builder=experiments/configs/build_newton_lift_hold_contact_source_manifest.py
launcher=experiments/configs/launch_newton_lift_hold_contact_source_manifest_tmux.sh
runner=experiments/configs/run_newton_lift_hold_contact_source_manifest_in_alloc.sh
log=logs/newton/newton_lift_hold_contact_source_manifest_v1_20260627.log
```

The launcher reused the existing tmux-held allocation and did not submit a new
GPU allocation.

## Outputs

```text
manifest=data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json
records_csv=data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/contact_source_records.csv
```

## Source Scope

```text
source_config=experiments/configs/lift_hold_scripted_feedback_baseline_v1.json
source_run_count=10
record_count=3600
splits=nominal, ordinary, held_out
ordinary_cells=empty_low, empty_medium, half_low, half_medium, half_high, full_medium, full_high
held_out_cells=full_low, empty_high
```

Every source run had a fresh Newton sanity record, automated visual validation,
manual visual inspection, metrics JSON, source NPZ, contact sheet, and frame
browser.

## Converted Fields

```text
newton.panda.step
newton.panda.sim_time
newton.contact.rigid_contact_count
newton.object.body_q.z
candidate.controller.phase_index
candidate.controller.feedback_trigger_count
candidate.controller.commanded_gripper_target
candidate.controller.commanded_lift_target
```

Source NPZ fields also include real Newton camera arrays, object pose arrays,
Panda joint arrays, and controller feedback arrays. These remain source fields
and are not renamed into T-Rex keys.

## Result

```text
status=pass
contact_count_min=29
contact_count_max=63
total_feedback_trigger_count=0
generated_trex_fields=[]
schema_promotion=blocked
failures=[]
```

## Interpretation

This conversion directly addresses the dataset mismatch issue by preserving
what the current workspace actually has: real Newton contact proxy, object
motion, camera, and controller evidence. It does not gate on exact T-Rex schema
compatibility and does not fabricate missing T-Rex fields.

The output is valid for Newton-native contact-aware curiosity replay
diagnostics and for auditing residual controller inputs. It is not valid for
T-Rex training, exact T-Rex schema compatibility, calibrated tactile F6 claims,
or dense tactile deformation claims.
