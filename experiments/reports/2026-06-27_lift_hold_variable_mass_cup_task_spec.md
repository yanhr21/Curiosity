# Lift-And-Hold Variable-Mass Cup Task Spec

Date: 2026-06-27

Phase: `01_newton_task_definition`

## Purpose

Define the first Newton-native adaptation benchmark before running new
compute-heavy experiments:

```text
lift-and-hold under object-property uncertainty
```

This is a task definition and validation artifact only. It is not training,
not policy success, not learned curiosity, not T-Rex schema promotion, and not
calibrated tactile F6.

## Official Source Basis

Selected first scene entry point:

```text
external/newton/newton/examples/robot/example_robot_panda_hydro.py
```

Reason:

- official Newton example;
- Franka/Panda manipulation path;
- hydroelastic contact;
- scripted grasp/lift trajectory;
- object pose, joint state, joint targets, and contact path;
- existing cup placement object.

Cup asset exists locally:

```text
external/newton-assets-cache/newton-assets_manipulation_objects_cup_f7f64ec3_8e8df07d/manipulation_objects/cup/model.usda
```

The first compute visual gate should validate the official Panda hydro chain
before replacing the grasped object with the cup asset.

## Config

Task config:

```text
experiments/configs/lift_hold_variable_mass_cup_task_v1.json
```

Validator:

```text
experiments/configs/validate_lift_hold_variable_mass_cup_task_v1.py
```

Human-readable spec:

```text
docs/lift_hold_variable_mass_cup_task_spec.md
```

## Validation Command

```bash
python3 experiments/configs/validate_lift_hold_variable_mass_cup_task_v1.py
```

Validation result:

- Status: `pass`
- Failures: `[]`
- Fill/mass levels: `3`
- Friction levels: `3`
- Required signals: `13`
- Curiosity diagnostic terms: `6`
- `generated_trex_fields=[]`
- `schema_promotion=blocked`
- `no_model_or_training=true`
- Visual gate status: `pending_compute_run`

Validation output:

```text
experiments/outputs/lift_hold_variable_mass_cup_task_v1_validation.json
```

## Task Summary

Episode phases:

- approach;
- pre-grasp contact;
- close gripper;
- lift;
- hold;
- optional regrasp or stabilization.

Object grid:

- mass/fill: empty, half, full;
- friction: low, medium, high;
- pose randomization: XY radius `0.025m`, yaw `[-20, 20]deg`, Z jitter `0.003m`.

Success metrics:

- lift height at least `0.12m`;
- hold duration at least `2.0s`;
- slip no more than `0.025m`;
- object not dropped.

Failure metrics:

- drop height loss above `0.05m`;
- contact loss for `12` frames;
- excessive-force proxy threshold `140`;
- unstable object acceleration above `8.0m/s^2`;
- timeout at `8.0s`.

## Next Compute Gate

The first visual gate remains pending. It must:

- run on a compute node;
- reuse a tmux-held allocation if available;
- activate only `envs/newton/.venv`;
- rerun fresh official Newton sanity first;
- export browser/contact-sheet visuals;
- block downstream use until manual visual inspection passes.

Direct image paths required after the run:

```text
experiments/visuals/<run_tag>/contact_sheet.png
experiments/visuals/<run_tag>/frame_0000.png
experiments/visuals/<run_tag>/frame_0120.png
experiments/visuals/<run_tag>/frame_0239.png
```
