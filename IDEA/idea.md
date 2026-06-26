# Newton-Native Curiosity Adaptation

## Core Idea

This project studies embodied curiosity for contact-rich manipulation. The
agent should not learn grasping from raw pixels alone, and it should not be
forced to imitate T-Rex's full data format before the simulator can honestly
produce the same physical signals.

The current direction is:

1. Use Newton as the primary simulator for closed-loop manipulation.
2. Use reliable Newton signals first: object pose, end-effector pose, robot
   state, action targets, contact counts, contact proxies, and camera views.
3. Add Taccel tactile-marker or deformation evidence only when it is real,
   nonzero, visually inspected, and kept under explicit provenance namespaces.
4. Treat T-Rex as a strong reference policy and future bridge, not as the
   immediate gate for all progress.
5. Evaluate whether curiosity and contact prediction help a grasping system
   adapt to object properties such as mass, friction, fill level, compliance,
   slip, and unexpected force response.

The useful research question is:

Can a robot with a basic grasping prior improve closed-loop manipulation by
actively testing physical hypotheses about objects, then adapting grip force,
lift speed, regrasp timing, and stabilization based on prediction errors over
object motion, contact, and tactile/contact evidence?

## Infant Analogy

The intended analogy is not a newborn learning all motor control from scratch.
The intended agent is closer to an infant that already has a basic grasping
ability and then learns how the world pushes back.

For example:

- The agent expects a cup to be heavy because it appears full.
- It lifts the cup and observes that the acceleration, contact response, and
  force requirement are lower than expected.
- That mismatch becomes useful prediction error, not just noise.
- The agent adjusts grip force, lift speed, and stabilization.
- Over repeated encounters, it learns that visually similar cups can have
  different mass, friction, fill level, and deformation response.

The same pattern should apply to slippery objects, deformable pouches, handles,
thin cards, fragile objects, and objects whose apparent geometry does not fully
predict contact behavior.

This is the "养" stage: start from a basic manipulation prior, then use
closed-loop curiosity and physical feedback to become better adapted to
different objects.

## Why T-Rex Is No Longer The Immediate Gate

T-Rex remains important, but it is not the current main bottleneck to solve
first.

The official T-Rex tactile-reactive data was collected on a real bimanual
Dexmate Vega-1 robot with two Sharpa Wave hands. Its training data expects
real synchronized fields such as bimanual state/action, head and wrist cameras,
10 fingertip force/torque streams, and 10 tactile deformation streams.

Newton can simulate robots, including bimanual setups, but current workspace
evidence does not yet provide a faithful T-Rex-equivalent episode source:

- current Newton Panda sources are single-arm and not T-Rex bimanual Sharpa;
- current bimanual Allegro evidence gives a different state and tactile-sensor
  contract;
- current Taccel marker evidence is real and nonzero, but it is marker-flow
  evidence, not calibrated T-Rex `[10,6]` F6;
- current dense tactile render attempts did not pass the nonuniform deformation
  visual gate;
- current source packages still lack the synchronized 62D state/action,
  accepted cameras, calibrated F6, and 10 dense tactile deformation streams
  required for strict T-Rex promotion.

Therefore, generating shape-compatible simulated data is not enough. If the
robot embodiment, action semantics, tactile calibration, and camera semantics
are wrong, a T-Rex loader may run while the experiment remains scientifically
invalid.

T-Rex should now be used as:

- a reference architecture for tactile-reactive policies;
- a checkpoint/reference baseline where official data is available;
- a future bridge once a faithful or explicitly accepted equivalent simulator
  source exists;
- not the required format for every Newton curiosity experiment.

## Current Mainline

The mainline is Newton-native closed-loop curiosity adaptation:

1. Build manipulation tasks with object property variation:
   mass, friction, fill-level proxy, compliance, shape, handle geometry, and
   fragility/safety tags.
2. Provide a basic grasping prior:
   scripted controller, behavior cloning, diffusion-policy-style policy, or
   another serious manipulation baseline.
3. Add intrinsic objectives over physical prediction:
   object motion, lift success, contact change, slip proxy, force/contact
   proxy, tactile-marker response, and safety cost.
4. Compare against baselines:
   no adaptation, scripted adaptation, contact-aware prediction, curiosity
   reward, and tactile/marker-aware variants.
5. Only later attempt T-Rex bridge work if the Newton-native mechanism proves
   useful and the required data contract can be satisfied without padding or
   renaming.

## Signals

Allowed current source namespaces:

```text
newton.state.*
newton.action_target.*
newton.object.*
newton.contact.*
newton.camera.*
taccel.marker.*
taccel.ftac.*
candidate.*
```

Forbidden promotions without strict evidence:

```text
observation.state
action
action_abs
observation.images.*
observation.tactile_f6
observation.tactile_deform.*
```

The project may use Newton and Taccel source data directly under their own
names. It must not rename partial evidence into official T-Rex fields.

## First Target Task Family

The first task family should be lift-and-hold adaptation:

- objects: cups, boxes, cylinders, pouches, cards, and simple handled objects;
- variation: mass, friction, fill-level proxy, compliance, and initial pose;
- initial skill: basic grasp and lift;
- challenge: detect that the expected physical response was wrong;
- adaptation: adjust grip force, lift speed, regrasp timing, or stabilization;
- metrics: lift success, slip/drop rate, excessive-force rate, object motion
  prediction error, contact prediction error, adaptation speed, and safety cost.

This directly matches the cup example: the agent may expect a water-filled cup
to be heavy, observe that it is lighter, and adapt force without crushing or
dropping it.

## Baseline Policy

The baseline should be serious and explicit:

- scripted or impedance grasp controller as a control baseline;
- behavior cloning or diffusion-policy-style baseline from generated
  demonstrations;
- contact-aware ICM or learning-progress curiosity as the intrinsic baseline;
- T-Rex compatibility diagnostics only as reference, not as the main baseline
  unless a faithful bridge exists.

Do not write a toy VQ-VAE, toy Transformer, toy world model, or toy T-Rex clone.
Any small diagnostic model must be labeled as a diagnostic and not represented
as faithful T-Rex progress.

## T-Rex Bridge Criteria

Revisit strict T-Rex promotion only when a source can provide all of the
following from the same synchronized episode:

```text
observation.state [62]
action [16,62]
action_abs [62]
observation.images.head
observation.images.wrist_right
observation.images.wrist_left
observation.tactile_f6 [10,6]
observation.tactile_deform.l0..l4
observation.tactile_deform.r0..r4
```

Every field must be real, synchronized, visually inspected where applicable,
and generated from a documented embodiment/controller contract. Padding,
shape-only projection, unrelated stream composition, or marker-to-F6 renaming
is not acceptable.

## Historical Archive

Pre-pivot experiments, reports, logs, and compatibility attempts are archived
under:

```text
legacy/2026-06-26_pre_pivot_archive/
```

They remain useful evidence, especially for understanding why strict T-Rex
promotion is blocked. They are no longer the primary planning surface.

## Post-Pivot Guarded Objective Gate

The first post-pivot Newton-native curiosity objective is now a guarded
contact/camera/object-change objective, not raw RGB curiosity and not a learned
world model. The relevant design artifact is archived under:

```text
legacy/2026-06-26_pre_pivot_archive/experiments/outputs/curiosity_v4_newton_objective_spec_20260626/
```

The objective suppresses simulator-settling artifacts by assigning
`actionable_score=0` to transitions with `sample_step_a < 10`. This moves the
selected segment away from the raw high-scoring `cube_dense_1->2` transient and
selects `guarded_cube_dense_173->174`.

A compute-node target-window visual gate was run around steps `169..178` in
the existing tmux-held allocation `154023`. It reran fresh official Newton
sensor-contact sanity, exported 10 SensorTiledCamera frames, passed visual
validation, and passed manual inspection of the contact sheet plus frames
`0173`, `0174`, and `0178`.

Post-pivot evidence final location after automatic archive:

```text
legacy/2026-06-26_pre_pivot_archive/experiments/outputs/newton_panda_hydro_camera_cube_objective_v4_guarded_169_178_20260626_0005_manual_visual_inspection.json
legacy/2026-06-26_pre_pivot_archive/experiments/outputs/newton_panda_hydro_camera_cube_objective_v4_guarded_169_178_20260626_0005_downstream_gate_cleared.json
legacy/2026-06-26_pre_pivot_archive/experiments/reports/2026-06-27_objective_v4_target_window_visual_gate.md
```

This is source prioritization and visual gating only. It does not claim learned
ICM, world-model training, policy success, calibrated F6, T-Rex schema
promotion, or physical Newton/Taccel synchronization.

## Phase 01 Concrete Task: Variable-Mass Cup Lift-And-Hold

The current concrete Newton task is a variable-mass cup lift-and-hold
adaptation benchmark. This is the first executable task family for the
Newton-native curiosity line.

The task is defined in:

```text
docs/lift_hold_variable_mass_cup_task_spec.md
experiments/configs/lift_hold_variable_mass_cup_task_v1.json
experiments/configs/validate_lift_hold_variable_mass_cup_task_v1.py
experiments/reports/2026-06-27_lift_hold_variable_mass_cup_task_spec.md
TODO/01_newton_task_definition/todo.md
```

The task uses the official Newton Panda hydro example as the first scene
entry point:

```text
external/newton/newton/examples/robot/example_robot_panda_hydro.py
```

The local official cup asset selected for the next gate is:

```text
external/newton-assets-cache/newton-assets_manipulation_objects_cup_f7f64ec3_8e8df07d/manipulation_objects/cup/model.usda
```

The experiment grid varies fill/mass proxy, friction, and initial pose. The
first mass levels are empty, half, and full cup proxies. The first friction
levels are low, medium, and high. Held-out combinations include a full
low-friction cup and an empty high-friction cup. This is meant to test whether
the policy can adapt to wrong expectations about object response, not merely
memorize a single cup.

Allowed observations remain Newton-native and explicitly namespaced:

```text
newton.panda.*
newton.object.*
newton.contact.*
newton.camera.*
candidate.controller.*
```

No T-Rex fields are promoted in this phase. The validator requires
`generated_trex_fields=[]`, `schema_promotion=blocked`, and
`no_model_or_training=true`.

## Phase 01 First Visual Gate Result

The first Phase 01 official visual gate has passed. It reused the existing
tmux-held allocation `154023`, reran fresh official Newton `sensor_contact`
sanity on the compute node, exported 9 SensorTiledCamera frames, and passed
manual visual inspection.

Run tag:

```text
lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021
```

Key evidence paths:

```text
logs/newton/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021.log
logs/newton/phase01_lift_hold_task_validation_20260627_0021.log
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_summary.json
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_visual_validation.json
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_manual_visual_inspection.json
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_downstream_gate_cleared.json
experiments/visuals/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021/frame_browser.html
experiments/visuals/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021/contact_sheet.png
```

The inspected frames showed nonblank head, right-wrist, and left-wrist camera
panels with the Panda robot, table, official grasped cube, and cup placement
context visible. This clears only the official Panda hydro scene, camera
export, validation, and reporting path. It does not claim cup grasp success,
learned curiosity, policy adaptation, T-Rex compatibility, or tactile F6.

## Immediate Next Step

Do not wait for strict T-Rex schema compatibility before moving. The next
completed step adapted the grasped object path from the official Panda hydro
example to the local official cup asset, reran fresh official Newton sanity,
exported camera frames, manually inspected the result, and recorded the cup
asset gate result.

Cup asset run tag:

```text
lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105
```

The cup-asset gate passed for retarget and visual evidence:

```text
tracked_object=existing_cup_asset
adapter=retarget_existing_official_cup_asset_as_object
generated_trex_fields=[]
schema_promotion=blocked
no_model_or_training=true
```

Manual inspection confirmed that the cup is visible and tracked. The numeric
summary reports `max_lift=0.15901388227939606`, but the final inspected frame
shows the cup tilted/fallen. Therefore this is not a stable cup grasp success
yet.

If cup geometry, collision, mass, or grasp initialization needs adjustment,
record it as a Newton cup-asset adaptation issue and keep iterating inside the
Phase 01 task definition. Do not substitute a toy model or pretend the result
is T-Rex-style data.

The next concrete step was to distinguish a real cup-grasp issue from the
official controller's short release cycle. A second gate,
`lift_hold_variable_mass_cup_v1_existing_cup_hold_gate_20260627_0145`, added
`final_hold_duration=999.0`. It passed official sanity and visual validation,
and manual inspection showed the cup still elevated at frame `0239`.

Numeric result:

```text
final_object_z=0.30836987495422363
max_lift=0.15986861288547516
```

This clears the extended-hold visual gate, but it still does not claim the full
two-second hold metric because the 240-frame diagnostic window covers only
about one second after reaching the high hold pose. The next step is a longer
metric gate, for example 360 frames or more, with explicit success/failure
metric extraction.

That longer metric gate has now passed:

```text
run_tag=lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210
num_steps=420
success_all_worlds=true
longest_hold_s=4.1
max_lift=0.16001205146312714
drop_from_max=0.0
failure_reasons=[]
```

Manual inspection of frames `0240`, `0360`, and `0419` confirmed that the cup
remains elevated through the sampled hold window. This clears the Phase 01
scripted cup lift-and-hold metric gate, but it still does not claim learned
curiosity, policy adaptation, T-Rex compatibility, tactile F6, or training.
