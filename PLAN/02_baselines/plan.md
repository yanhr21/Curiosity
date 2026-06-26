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

Next requirement: implement and sanity-check a real Newton parameter adapter
that changes the tracked object's mass/inertia and contact friction, records
the requested and observed parameters in the rollout summary, and passes the
same official sanity/export/visual gate before any mass/fill variant is
reported.

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
