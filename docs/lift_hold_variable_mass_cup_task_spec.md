# Lift-And-Hold Variable-Mass Cup Task Spec

Phase: `01_newton_task_definition`

## Goal

Define the first Newton-native adaptation benchmark:

```text
lift-and-hold under object-property uncertainty
```

The robot starts from a basic grasp/lift prior and must adapt when the object
does not respond as expected because mass, fill level, friction, or pose has
changed.

## Official Source Basis

The first scene entry point is the official Newton Panda hydroelastic
manipulation example:

```text
external/newton/newton/examples/robot/example_robot_panda_hydro.py
```

The official example already provides:

- Franka/Panda gripper manipulation;
- hydroelastic contact;
- scripted grasp/lift waypoints;
- object pose and joint targets;
- `pen` and `cube` grasped-object scenes;
- a cup placement object.

The local Newton asset cache also contains a cup asset:

```text
external/newton-assets-cache/newton-assets_manipulation_objects_cup_f7f64ec3_8e8df07d/manipulation_objects/cup/model.usda
```

The first visual gate should validate the official Panda hydro path before
replacing the grasped object with a cup-like asset. This keeps the experiment
grounded in an official sanity-checked path instead of inventing a new scene
and debugging everything at once.

## Task

Each episode has these phases:

1. approach;
2. pre-grasp contact;
3. close gripper;
4. lift;
5. hold;
6. optional regrasp or stabilization.

The first controller is scripted. Adaptation initially changes controller
parameters rather than training a policy:

- gripper closure target;
- lift velocity scale;
- hold height target;
- regrasp trigger threshold;
- stabilization duration.

## Object Grid

Fill/mass:

- `empty`: expected total mass `0.08 kg`;
- `half`: expected total mass `0.20 kg`;
- `full`: expected total mass `0.35 kg`.

Friction:

- `low`: static/dynamic `0.35/0.30`;
- `medium`: static/dynamic `0.80/0.65`;
- `high`: static/dynamic `1.20/0.95`.

Pose randomization:

- XY radius: `0.025 m`;
- yaw: `[-20, 20] deg`;
- Z jitter: `0.003 m`.

Held-out cells:

- `full_low_friction`;
- `empty_high_friction`.

## Observations

Required provenance namespaces:

- `newton.panda.*`;
- `newton.object.*`;
- `newton.contact.*`;
- `newton.camera.*`;
- `candidate.controller.*`.

Required signals:

- joint state and joint targets;
- end-effector pose;
- object pose and velocity or finite-difference velocity;
- contact count or contact proxy;
- optional contact impulse/force proxy if available;
- RGB-D proxy camera frames;
- controller phase and command parameters.

Taccel marker evidence is optional and must remain under `taccel.marker.*`.

## Metrics

Success:

- lift height at least `0.12 m`;
- hold duration at least `2.0 s`;
- slip no more than `0.025 m`;
- object not dropped.

Failure:

- drop height loss above `0.05 m`;
- contact loss for `12` frames;
- contact proxy above `140` as an excessive-force proxy;
- unstable object acceleration above `8.0 m/s^2`;
- timeout at `8.0 s`.

Adaptation:

- expected-vs-observed lift mismatch;
- expected-vs-observed contact mismatch;
- expected-vs-observed object acceleration mismatch;
- frames until success after mismatch;
- success per contact-proxy integral.

## Visual Gate

The first visual gate is pending compute execution. It must:

- run on a compute node;
- reuse a tmux-held allocation when available;
- activate only the local Newton venv;
- rerun fresh official Newton sanity first;
- export a browser and contact sheet;
- block downstream use until manual visual inspection passes.

Expected direct visual paths:

```text
experiments/visuals/<run_tag>/contact_sheet.png
experiments/visuals/<run_tag>/frame_0000.png
experiments/visuals/<run_tag>/frame_0120.png
experiments/visuals/<run_tag>/frame_0239.png
```

## Non-Claims

This task spec is not:

- learned curiosity;
- policy success;
- calibrated tactile F6;
- T-Rex schema promotion;
- T-Rex/VQ-VAE progress;
- touch-dominant manipulation success.

It is a Phase 01 benchmark definition and visual-gate plan.
