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

Current reference-checkpoint status as of 2026-06-27: the staged official
T-Rex midtrain assets passed checkpoint integrity and official model-load
sanity in the Curiosity allocation. Evidence is recorded in
`experiments/reports/2026-06-27_phase06_trex_checkpoint_current_sanity.md`.
This keeps T-Rex available as a reference checkpoint, but it does not solve the
Newton-to-T-Rex data-contract gap and does not replace the Newton scripted
infant prior.

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

## Training Strategy

The first learned system should not learn grasping from scratch. It should use
a reliable basic grasp-and-lift prior, then learn residual adaptation around
that prior. The intended "infant" has primitive manipulation ability already:
it can approach, close the gripper, lift, and hold, but it still needs to learn
how different objects push back.

Current decision: the short-term infant prior is the official Newton Panda
hydro scripted grasp/lift path, not a pretrained checkpoint. Web and local
source checks did not identify a directly usable Newton-native Panda
grasp/lift checkpoint. OpenPI DROID/Franka, Isaac Lab Mimic, and Isaac Sim
Franka policy assets remain future audit candidates, but they are not the
active short-term route.

User-approved route as of 2026-06-27: use this short-term stable method now.
The project should not wait for a Newton-native pretrained grasp checkpoint
before building the first infant baseline, feedback-adaptation baseline, and
residual-learning path. Checkpoint audits can continue later, but they are
secondary to the Newton scripted-prior route until a compatible official policy
is proven through the same visual and metric gates.

Short-term stable route: keep the official Newton Panda hydro scripted
grasp/lift controller as the non-learned infant prior, make the baseline
physics honest first, then learn residual adaptation around that prior. The
first learning target is not an end-to-end grasp policy. It is a small
controller-parameter or residual policy that changes grasp/lift/hold behavior
based on object motion, contact, slip, and later tactile evidence. This route
is selected because it already gives reliable primitive manipulation behavior
and avoids pretending that an embodiment-mismatched checkpoint is a
Newton-native grasping model.

The immediate implementation must prefer a pre-finalization Newton
mass/inertia/friction adapter in the official Panda hydro builder path, or a
documented Newton model-update API that passes the camera/export/visual gate.
Do not continue the runtime model-array mutation path that repeatedly produced
Warp CUDA illegal memory access during SensorTiledCamera/export cleanup.
As of 2026-06-27, the pre-finalization builder adapter is the active short-term
route. It has passed fresh official Newton sanity, SensorTiledCamera export,
automated visual validation, and manual visual inspection, and it has already
produced real cup variants for `empty_medium`, `half_medium`, and
`full_medium`. All three medium-friction variants lift and hold the cup with
low slip and no drop, but all fail the strict baseline only on the
object-acceleration threshold. That failure is useful: it identifies the first
residual-adaptation target as gentler stabilization of the lift/hold trajectory
rather than end-to-end grasp discovery.
The low-friction axis has also completed its non-held-out empty and half cells
(`empty_low`, `half_low`); both show the same pattern: real low friction is
applied in Newton, lift/hold/slip/drop/contact gates pass, and the strict
failure remains object acceleration. `full_low` remains a held-out
generalization cell.
The ordinary high-friction axis has completed with `half_high` and
`full_high`; both apply real Newton friction, pass lift/hold/slip/drop/contact
gates, and fail only on the strict object-acceleration threshold. The held-out
`full_low` and `empty_high` cells have also been evaluated as no-adaptation
evidence and must remain labeled as held-out generalization evidence for later
learned-adaptation comparisons.
The scripted feedback adaptation baseline has been configured as
`CONTROLLER_MODE=lift_hold_feedback` around the same official Newton scripted
prior. It uses real object-motion and contact-proxy feedback to adjust lift
duration, hold target, and stabilization timing. It is not a learned policy and
its nominal cup gate has passed fresh official Newton sanity, camera export,
visual validation, and manual visual inspection. Shared metrics still mark the
nominal feedback run as fail only on the strict object-acceleration threshold;
lift, hold, slip, drop, and contact gates pass. The nominal run did not trigger
feedback, which is acceptable because the rule should not perturb stable
nominal behavior without a detected mismatch.
The first ordinary feedback grid cell, `empty_low`, has also completed with
real Newton mass/friction applied. It passes lift/hold/slip/drop/contact gates
and fails only on the strict object-acceleration threshold. Feedback did not
trigger, so this is an honest scripted-feedback baseline result rather than a
claim that adaptation improved behavior.
The second ordinary feedback grid cell, `empty_medium`, shows the same pattern:
real mass/friction applied, visual and contact gates pass, strict metrics fail
only on object acceleration, and feedback does not trigger.
The third ordinary feedback grid cell, `half_low`, also completes with real
Newton mass/friction provenance. It passes visual, lift, hold, slip, drop, and
contact gates, fails only on the strict object-acceleration threshold, and does
not trigger feedback. This keeps the scripted-feedback result honest: it is a
runnable controller-parameter feedback baseline, not a learned adaptation or
curiosity result yet.
The fourth ordinary feedback grid cell, `half_medium`, completes with the same
validated pattern: real Newton half-mass and medium-friction settings are
applied, visual/lift/hold/slip/drop/contact gates pass, strict metrics fail
only on object acceleration, and feedback does not trigger.
The fifth ordinary feedback grid cell, `half_high`, also completes. Real
Newton half-mass and high-friction settings are applied, visual and task gates
pass, strict metrics fail only on object acceleration, and feedback remains
inactive.
The sixth ordinary feedback grid cell, `full_medium`, completes with real
Newton full-mass and medium-friction settings. It passes visual, lift, hold,
slip, drop, and contact gates, fails only on object acceleration, and does not
trigger feedback.
The seventh and final ordinary feedback grid cell, `full_high`, completes the
ordinary scripted-feedback mass/friction grid. It applies real Newton full-mass
and high-friction settings, passes visual/lift/hold/slip/drop/contact gates,
fails only on object acceleration, and does not trigger feedback. `full_low`
and `empty_high` remain held-out cells for later learned-adaptation comparison.
The first held-out scripted-feedback evaluation cell, `full_low`, has now been
run as held-out evidence rather than ordinary/training evidence. It applies
real Newton full-mass and low-friction settings, passes visual/lift/hold/slip/
drop/contact gates, fails only on object acceleration, and does not trigger
feedback.
The second held-out scripted-feedback evaluation cell, `empty_high`, also
completes. Both held-out cells now pass visual/lift/hold/slip/drop/contact
gates with real Newton physics provenance, both fail only on the strict
object-acceleration threshold, and neither triggers feedback. This completes
the scripted-feedback evaluation grid, but it still does not justify an
adaptation-improvement claim.

As of 2026-06-27, the short-term stable method is explicitly selected for the
next work: do not wait for an unverified Newton-native checkpoint, and do not
train a placeholder policy. Continue from the official Newton Panda hydro
scripted infant prior, collect residual controller-parameter labels only from
ordinary cells, and promote a label source only if it has nonzero feedback
while passing official Newton sanity, automated/manual visual inspection,
lift, hold, drop, and contact gates. The first two nonzero residual diagnostics
(`residual_label_source_sensitive_feedback_half_low_20260627_030145` and
`residual_label_sweep_half_low_contact58_20260627_0310`) prove that nonzero
`candidate.controller.*` labels can be generated, but both failed the formal
hold-duration gate and therefore remain diagnostic-only. The next immediate
step is a less disruptive ordinary-cell threshold sweep, not learned-adapter
training.

The less disruptive contact58 gentle sweeps then showed that nonzero feedback
labels can preserve lift, hold, drop, contact-loss, automated visual
validation, and manual visual inspection. The repeated strict acceleration
failure was traced by peak analysis to an initial recorded settling artifact:
the non-warmup top peak occurred at step 2, phase 0, before feedback was
active. Adding `PRE_RECORD_WARMUP_STEPS=15` removes that artifact from the
recorded metric window while preserving the official Newton rollout path and
nonzero feedback labels. The first promoted source candidate is
`residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006`,
with metrics status pass, `feedback_trigger_count=241`, lift height
`0.15815936028957367` m, hold duration `2.5333309173583984` s, and
`max_object_accel_m_s2=0.5063306543767194`. The next step is no longer blind
threshold sweeping; it is to build the formal residual-label source runner
around `experiments/configs/residual_label_source_manifest_v1.json`, collect
additional ordinary cells, and keep held-out `full_low` and `empty_high` for
evaluation.

The formal source runner now exists and has passed on five ordinary cells:
`half_low`, `empty_low`, `half_medium`, `full_high`, and `empty_medium`. The
final runner `residual_label_source_runner_v1_20260627_0455` reran fresh official Newton
sanity inside tmux-held allocation `154142`, produced
`data/processed/residual_label_source_runner_v1_20260627/manifest.json`, and
validated `1800` records from `5` source runs with `failures=[]`,
`generated_trex_fields=[]`, `schema_promotion=blocked`, and
`training_started=false`. This clears the source-runner blocker but still does
not create a learned adapter. Source availability should no longer be treated
as the active gate; the main next step is a reviewed learned residual-adapter
runner that consumes these sources while preserving held-out split enforcement.

The residual-adapter training preflight now also exists and passed on compute
run `residual_adapter_training_preflight_v1_20260627_0523`. It reran fresh
official Newton sanity inside allocation `154142`, consumed the five-source
runner output, and wrote
`data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
The split is train=`1440` records from `half_low`, `empty_low`, `half_medium`,
and `full_high`, validation=`360` records from `empty_medium`, with held-out
`full_low` and `empty_high` still excluded. The preflight manifest has
`failures=[]`, `generated_trex_fields=[]`, `schema_promotion=blocked`,
`training_started=false`, and `no_model_created=true`. The active next gate is
the actual residual-adapter trainer implementation and review, not source
collection or split construction.

The training path is:

1. Start with the official Newton Panda hydro scripted grasp/lift path as the
   non-learned infant prior.
2. Generate Newton rollouts across mass, friction, fill-level proxy, and pose
   randomization.
3. Train only a small residual adapter or controller-parameter policy at first:
   gripper closure target, lift velocity scale, hold height target, regrasp
   trigger threshold, and stabilization duration.
4. Train object/contact/tactile forward models on the same rollouts before
   using curiosity for policy adaptation.
5. Add intrinsic reward only after the forward-model diagnostics show useful
   prediction of object motion, contact, slip, and tactile/contact response.
6. Evaluate on held-out mass/friction cells before claiming adaptation.

The data unit is a synchronized episode, not an isolated image. Required
episode fields include robot joint state, end-effector pose, object pose and
velocity, contact count or contact proxy, camera RGB-D, controller phase,
controller command parameters, success/failure labels, and later real tactile
evidence under `taccel.marker.*` or other explicit source namespaces.

OpenPI, pi0/pi0-FAST, diffusion policy, ACT-style policies, and T-Rex-style
tactile architectures may be considered only as serious baselines or reference
architectures. They must enter through documented source code, documented
checkpoints, and explicit observation/action adapters. They are deferred until
the scripted Newton infant prior has produced stable baseline rollouts and a
checkpoint audit is worth the extra integration cost. The immediate Newton
mainline must not depend on pretending that a mismatched checkpoint is a solved
grasping policy.

## Curiosity Mechanism

Curiosity should be driven by physical learning progress, not raw pixel
novelty. The agent should be rewarded for actions that improve predictions of
task-relevant physical consequences:

```text
intrinsic_reward =
  learning_progress(object/contact/tactile prediction)
+ controllable_disagreement
+ bounded_useful_change
- safety_penalty
- no_op_penalty
- excessive_force_penalty
```

The prediction targets are:

- object pose delta and velocity;
- lift response under expected mass;
- contact count or contact-proxy change;
- slip or contact-loss risk;
- tactile-marker flow, active marker count, or deformation response when real
  tactile evidence is available;
- success/failure risk under the current controller parameters.

Raw prediction error alone is not sufficient because it can reward chaotic
collisions, dropped objects, or visual noise. Learning progress and bounded
useful change should be preferred: the agent should seek interactions that
make its model better while staying inside force, drop, and stability limits.

Required ablations:

- no curiosity;
- random intrinsic reward;
- object-motion-only curiosity;
- contact-only curiosity;
- tactile-only curiosity;
- vision+tactile curiosity;
- shuffled tactile;
- delayed tactile.

Current diagnostic status as of 2026-06-27: the Phase 03 contact-aware
curiosity replay evaluator has passed on the full 3x3 no-adaptation
mass/friction grid, including held-out `full_low` and `empty_high`. The output
is `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`
with `status=pass` and `rollout_count=9`. This result is useful reward-shape
evidence only. It uses diagnostic replay predictors, does not train a learned
world model, does not update a policy, and uses `newton.contact_proxy_only`
rather than real tactile-marker evidence.

Curiosity is considered useful only if it improves held-out mass/friction
adaptation without hiding drop, slip, or excessive-force failures.

## Vision-Tactile Fusion And Masking

Touch must be a first-class online signal, not a late feature concatenated only
for reporting. The planned model structure is:

```text
vision_encoder(rgb, depth) -> z_v
tactile_encoder(contact_proxy, marker_flow, deform) -> z_t
proprio_encoder(joint, ee, gripper, phase) -> z_p
action_encoder(controller_params) -> z_a

fusion(z_v, z_t, z_p, z_a, masks) -> z
policy_head(z) -> residual controller params
forward_model(z, action) -> next object/contact/tactile prediction
```

The tactile stream enters both the policy and the curiosity forward model. This
ensures touch can change actions online and can also drive exploration through
prediction learning progress.

Current source status as of 2026-06-27: the first tactile/contact source is a
Newton contact-proxy manifest, not real tactile F6 or dense deformation. The
manifest at
`data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`
has `status=pass`, `source_run_count=10`, `record_count=3600`,
`generated_trex_fields=[]`, and `schema_promotion=blocked`. This can support
Newton-native contact-aware diagnostics and future residual-adapter input
audits, but it must not be described as T-Rex tactile evidence.

Current training-preparation status as of 2026-06-27: the residual adapter and
forward-model target contract is defined in
`docs/residual_adapter_forward_model_contract_v1.md` and
`experiments/configs/residual_adapter_forward_model_contract_v1.json`. It adds
controller-parameter residual outputs, object/contact prediction targets,
Newton contact-proxy tactile/contact input, modality masks, post-contact
pure-touch windows, held-out cells, and required ablations. Its status is
`target_contract_ready_training_not_started`; it is not a trained adapter and
not a learned world model.

Current residual-adapter training readiness as of 2026-06-27: training is
blocked, not started. The readiness audit is recorded in
`experiments/reports/2026-06-27_phase04_residual_adapter_training_readiness_v1.md`.
The original blocker was that all scripted-feedback evaluations had
`feedback_trigger_count=0`. The first ordinary-cell sensitive-feedback
diagnostic,
`residual_label_source_sensitive_feedback_half_low_20260627_030145`, now proves
the official Newton path can emit nonzero residual controller corrections
(`feedback_trigger_count=241`). It is not promoted to training data because the
threshold is too aggressive and fails hold-duration/object-acceleration
metrics. A follow-up acceleration-sensitive diagnostic preserves lift/hold/
drop/contact behavior but produces `feedback_trigger_count=0`, so scalar
threshold tuning alone did not solve it. The best current candidate is
`residual_label_sweep_half_low_contact58_gentle_20260627_0345`: it produces
nonzero residual labels and preserves lift/hold/drop/contact/visual/manual
gates, but strict metrics still fail on object acceleration. The next step is
to reduce object acceleration around that candidate on ordinary cells, not a
toy policy or no-op adapter.

Current residual-correction collection plan as of 2026-06-27: the first
diagnostic collection route is defined in
`experiments/configs/residual_correction_collection_plan_v1.json` and
`experiments/reports/2026-06-27_phase04_residual_correction_collection_plan_v1.md`.
It proposes an acceleration-sensitive ordinary-cell diagnostic to produce
nonzero feedback residual fields while preserving held-out `full_low` and
`empty_high`. This is not training and not an adaptation-improvement claim.

Current nonzero residual diagnostic status as of 2026-06-27: run
`residual_label_source_sensitive_feedback_half_low_20260627_030145` produced
nonzero residual feedback fields (`feedback_trigger_count=241`) on ordinary
`half_low`, but it is rejected as a training label source because the formal
metrics failed the hold-duration gate. The diagnostic proves residual fields
can be generated; it does not provide usable training labels yet.

Vision and touch must be balanced through training-time modality masking,
cross-modal prediction, and explicit ablations. The policy should see:

- both vision and touch;
- vision masked, touch visible;
- touch masked, vision visible;
- partial vision mask;
- partial tactile mask.

After contact, the curriculum should include pure tactile windows. This matches
the guitar-playing analogy: early approach still needs vision, but once contact
is established the agent should be able to stabilize, detect slip, and adjust
without continuously looking.

Initial masking policy:

```text
p(mask_vision | post_contact) = 0.3 -> 0.6 curriculum
p(mask_tactile | post_contact) = 0.1 -> 0.2
p(both_visible) remains nonzero
```

Pure tactile success is not enough. The required evidence is that multimodal
vision+touch outperforms vision-only and touch-only, while shuffled or delayed
tactile degrades performance. That is the test that touch is real, online, and
causally useful.

Current ablation-reporting status as of 2026-06-27: the Phase 05
contact-proxy ablation report is recorded in
`experiments/reports/2026-06-27_phase05_contact_proxy_ablation_report_v1.md`.
It summarizes existing Phase 03 replay diagnostics for object-motion-only,
contact-proxy-only, object+contact, shuffled-contact, and delayed-contact
ablations across 9 mass/friction rollouts. This is not yet proof of a trained
policy using touch; it is the current auditable baseline for later residual
adapter evaluation.

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
