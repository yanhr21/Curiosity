# Global Agent Rules

This repository is now for video-guided, embodiment-aware active
loco-manipulation for unknown-load carrying. Old dense-tactile Curiosity
materials were archived outside the repository at:

```text
/public/home/yanhongru/Curiosity_archive_20260702_pre_video_guided_carrying/
```

Do not treat archived Curiosity results as current success evidence. They may
only be used as historical caution about overclaiming, weak held-out transfer,
and proxy-field promotion.

## Highest Priority Cluster Safety Rules

These rules override all other project instructions.

### Login Node Hard Limit

- Never run Python experiments, data processing, validation builders, model
  loading, rendering, simulation, training, evaluation, visualization
  generation, dataset conversion, NumPy/PyTorch-heavy scripts, or any other
  compute-heavy project task on a login or management node such as
  `mgmtserver02`.
- Login nodes are only for lightweight operations: editing files, `git`
  commands, `git clone`, `git push`, small text inspection with tools such as
  `sed`/`rg`, lightweight file listing, and job/allocation submission.
- Keep login-node CPU below 300% and memory within lightweight interactive
  limits. If a command can plausibly exceed those limits, do not run it on the
  login node.
- If a project Python command is needed and it is not a trivial import-free
  syntax check, submit or run it inside a compute allocation instead.

### Compute Node Requirements

- All simulation, rendering, dataset conversion, training, evaluation, model
  loading, and visualization generation must run on compute nodes.
- GPU resources must be obtained and kept through `tmux` plus persistent
  `srun`/`salloc` allocation workflow. Do not use one-shot submission paths
  such as `sbatch` or single-use wrappers for experiments unless the user
  explicitly approves.
- Do not use `sspath` or other one-shot resource paths for this project.
- Compute nodes should only activate prebuilt local shared-filesystem
  environments. Do not perform normal dependency installation, venv creation,
  package builds, or dependency resolution on compute nodes.
- Short runs must be labeled as diagnostics or smoke tests, not as real
  training or real experiment results.

### Resource Exclusion Zone

- Do not touch, inspect, stop, reuse, attach to, or modify any `reflex`,
  `ICLR2027/Reflex`, OpenPI, Cosmos, or other non-Curiosity tmux sessions,
  allocations, processes, logs, scripts, or resources.
- If non-project sessions appear in process listings, ignore them except to
  avoid interference.

## Active Research Direction

- Active idea: `IDEA/idea.md`.
- Main survey: `docs/2026-07-02_research_overview.md`.
- Working title:
  `Video-guided, embodiment-aware active loco-manipulation for unknown-load carrying`.
- Core claim: video can provide task semantics, progress, object-motion, and
  contact-affordance priors, but the robot must actively probe unknown object
  dynamics and choose a stable, low-cost posture for its own body.
- Current negative conclusion: as of 2026-07-02, no known system fully solves
  cross-morphology humanoid box carrying with unknown weight/shape, active
  self-selected posture, long-duration carrying, and non-retargeting
  video-conditioned RL.
- Current runnable Isaac scaffold: `scripts/isaac/build_adaptive_probe_carry_scene.py`
  with launcher `scripts/isaac/run_adaptive_probe_carry_scene.sh`.
  It builds approach, active-probe, posture-adjust, lift, and carry phases in
  Isaac; estimates load from probing proxies; chooses posture from morphology
  and load; and logs belief, support-margin proxy, effort proxy, drops, and
  target distance.
- The validated 2026-07-04 adaptive Isaac smokes are diagnostic only:
  `20260704_adaptive_probe_carry_scene_smoke2_clean` selected
  `low_front_carry` for an 8 kg box and completed 300/300 steps with drop 0;
  `20260704_adaptive_probe_carry_scene_smoke3_chest` selected
  `chest_supported_slow` for a short-arm robot carrying an 11 kg larger box
  and completed 260/260 steps with drop 0.
- Do not report the adaptive scaffold as dynamic humanoid locomotion, learned
  balance, learned grasping, true contact carrying, or video-conditioned RL.
  It is the current Isaac task-construction baseline for making progress while
  the official G1/ANYmal articulation tensor path is broken.
- New direct dynamic Isaac route under test:
  `scripts/isaac/build_usd_dynamic_quadruped_carry_scene.py` with launcher
  `scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`. This route avoids
  IsaacLab Articulation/RigidObject tensors and instead uses USD/PhysX rigid
  bodies, revolute joints, fixed joints, and drive target attributes. Its
  first target is dynamic robot walking with a physical box payload fixed to
  the torso.
- 2026-07-04 USD dynamic quadruped results are negative so far:
  `smoke1` with articulation root on GPU completed but travel stayed 0 and
  PhysX emitted `setDriveTarget()` direct-GPU errors;
  `smoke2_noartroot` on GPU and `smoke3_cpu` on CPU completed with no falls
  or drops but still had torso/box travel 0;
  `smoke4_core_cpu` failed before rollout because Isaac Sim
  `SingleArticulation` expected `SimulationManager._get_backend_utils`, which
  is absent under the current IsaacLab `PhysxManager` context. Do not rerun
  these unchanged.
- 2026-07-04 dynamic rigid-body control probes are also negative so far:
  runtime USD `RigidBodyAPI.velocity` writes produced 0 travel on CPU/GPU;
  `omni.physx.get_physx_simulation_interface().apply_force_at_pos` produced 0
  travel on CPU/direct-step smokes and direct-GPU `addForce()`/`addTorque()`
  errors on GPU; bare `CuboidCfg.func` cubes showed no observed gravity drop;
  `RigidObjectCfg` root-state reads failed with
  `Failed to get rigid body transforms from backend`; Isaac Sim core
  `DynamicCuboid` wrappers stalled before a completed reset. Do not rerun these
  unchanged or tune parameters on them.

## 2026-07-04 Active Execution Objective

The active implementation objective is a real Isaac physics simulation of a
robot carrying a box. Do not redefine this as a visualization, static scene, or
box-only smoke test.

Required final behavior:

- Create or integrate a robot in Isaac that can walk and maintain balance in
  simulation.
- The robot must complete a box-carrying task in simulation.
- While carrying the box in any claimed carrying posture, the robot must keep
  balance and continue walking.

Current acceptable intermediate milestones:

- Isaac scene construction smoke tests.
- Box-only rigid-body gravity/collision validation.
- Robot asset loading and standing diagnostics.
- Locomotion policy or controller integration diagnostics.
- Contact and carry-object attachment/contact diagnostics.

These intermediate milestones are not completion. The goal is incomplete until
there is Isaac physics evidence of a robot walking while carrying a box and
remaining balanced.

Do not wait on external models or official demos if they block scene
construction. Use relevant official work and code as references or baselines,
but continue building the direct Isaac task.

### Current Direct Isaac Execution Path

- Main scene script: `scripts/isaac/build_minimal_carry_scene.py`.
- Compute launcher: `scripts/isaac/run_minimal_carry_scene.sh`.
- Stand/walk/payload sequence:
  `scripts/isaac/run_g1_wbc_smoke_sequence.sh`.
- Diagnostic proxy scene:
  `scripts/isaac/build_proxy_carry_scene.py`.
- Direct carrying-task scene diagnostic:
  `scripts/isaac/build_direct_carry_task_scene.py` and
  `scripts/isaac/run_direct_carry_task_scene.sh`.
- Low-level contact-carry diagnostics:
  `scripts/isaac/build_contact_carry_scene.py`,
  `scripts/isaac/run_contact_carry_scene.sh`,
  `scripts/isaac/build_contact_carry_rigid_scene.py`, and
  `scripts/isaac/run_contact_carry_rigid_scene.sh`.
- MuJoCo fallback dynamic payload diagnostic:
  `scripts/mujoco/run_quadruped_payload_carry.py` and
  `scripts/mujoco/run_quadruped_payload_carry.sh`.
- Non-tensor USD/PhysX dynamic quadruped carry diagnostic:
  `scripts/isaac/build_usd_dynamic_quadruped_carry_scene.py` and
  `scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`.
- Local WBC asset checker:
  `scripts/isaac/check_g1_wbc_local_assets.py`.
- Smoke summary checker:
  `scripts/isaac/check_carry_smoke_summary.py`.
- Current local WBC asset root:
  `/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy/`.

Historical first GPU validation command:

```bash
RUN_PAYLOAD=0 DEVICE=cuda:0 bash scripts/isaac/run_g1_wbc_smoke_sequence.sh
```

It has already been tried on job `164814` and failed before stepping with
`Failed to get DOF positions from backend`. Do not repeat the same G1 smoke
without a concrete tensor-backend fix or a different official Arena entry path.

If a future fix makes stand and walk pass, run payload diagnostics with:

```bash
RUN_PAYLOAD=1 DEVICE=cuda:0 bash scripts/isaac/run_g1_wbc_smoke_sequence.sh
```

CPU-only and direct GPU G1 articulation smokes are not valid paths yet on this
cluster: repeated compute-node diagnostics failed before stepping with
`Failed to get DOF positions from backend`, including after `InteractiveScene`,
`SKIP_EXPLICIT_STATE_RESET=1`, and `DISABLE_USD_PHYSICS_UPDATES=1`.
CPU may still be used for box-only rigid-body smoke, but not for robot
locomotion or carrying claims.

The proxy scene is allowed only as an Isaac scene/output skeleton diagnostic.
Its `kinematic_proxy_carrier_pose-follow_payload` output must not be reported as
humanoid walking, balancing, grasping, or carrying success.

The direct carrying-task scene is the current fastest runnable Isaac task
construction path. Its validated smoke
`20260704_direct_carry_task_scene_smoke4` reached the carry phase and target
with a kinematic humanoid proxy and massed box, but it is still diagnostic-only:
it must not be reported as learned balance, contact-rich grasping, autonomous
posture selection, or true robot carrying success.

The first low-level contact-carry smoke
`20260704_contact_carry_smoke1` is a negative result: the dynamic box did not
move or lift when palms were moved by USD xform edits. The RigidObject-driven
contact scene also failed on GPU and CPU with
`Failed to set rigid body transforms in backend`. The next contact route should
use a pure Omni/PhysX non-tensor kinematic-target API or repair the current
IsaacLab/PhysX tensor invalidation. Do not rerun the same RigidObject contact
smoke without a concrete backend change.

The MuJoCo quadruped payload script is allowed as a fallback dynamic physics
baseline while IsaacLab tensor paths are broken. It uses an assistive stabilizer
and a welded payload, so it must not be reported as unknown-box grasping,
active probing, or final Isaac robot-carrying success.

The USD dynamic quadruped script is the preferred next dynamic Isaac attempt
because it stays in Isaac while avoiding the broken tensor path. Its fixed-box
payload can count only as a dynamic fixed-payload carrying diagnostic unless it
produces verified walking, balance, and carry metrics in a compute-node run.
Even if it passes, it still does not solve unknown free-object grasping or
video-conditioned active carry.

After the 2026-07-04 USD dynamic and rigid-body negative smokes, the next Isaac
dynamic-control step must change the runtime control mechanism, not just gait
parameters. Valid next directions are: repair `SingleArticulation`/
`SimulationManager` compatibility in the IsaacLab context, find a known-good
official Isaac Sim dynamic-body example in the installed distribution and port
it faithfully, use the correct non-deprecated
`isaacsim.core.experimental.prims.Articulation` API, or use an official Arena
task entry point whose body/joint state changes are verified before adding the
box. Do not keep changing hip/knee amplitudes while travel is exactly zero.

Current direct control-path repair script:
`scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py` with launcher
`scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`. This route
avoids IsaacLab `SimulationContext` and tensor APIs entirely, creates a custom
USD articulated quadruped plus fixed physical payload in Isaac Sim core
`World`, and drives it through `SingleArticulation.apply_action()`. Passing
criterion for this diagnostic is initialized expected DOFs plus nonzero
measured joint motion in a compute-node run. Do not report it as walking,
balancing, carrying, unknown-object handling, or learned control unless later
runs also prove those properties. Current 2026-07-04 result is negative:
direct `SimulationApp` needed the local IsaacLab headless experience to start,
then the custom articulation path stopped around `SingleArticulation`
registration/wrapper and produced no summary; the AppLauncher variant reached
`Creating SingleArticulation wrapper` and also produced no summary. Do not
rerun unchanged.

2026-07-04 allocation `curiosity_core_world_0704`, job `165036`, node
`server10`, also tested the missed USD/PhysX combination
`ARTICULATION_ROOT=1 CONTROL_MODE=usd_drive_attr DEVICE=cpu` under stamp
`20260704_usd_dynamic_quad_payload_smoke5_cpu_artroot`. It completed 300/300
steps with falls 0 and drops 0, but torso and box travel stayed 0.0. This is a
negative result, not carrying evidence. Next dynamic-control work must avoid
repeating custom USD drive-target-only actuation and custom core
`SingleArticulation` wrapping unchanged.

2026-07-04 official-asset experimental ANYmal articulation probe:
`scripts/isaac/run_anymal_experimental_articulation_smoke.py` with launcher
`scripts/isaac/run_anymal_experimental_articulation_smoke.sh` loaded the local
ANYmal-C USD and exposed 12 DOFs, but smoke9 failed because the physics tensor
entity stayed invalid after warmup:
`Instance's physics tensor entity is not valid`. This is a negative
joint-control diagnostic, not walking or carrying evidence. Do not keep waiting
on this route before building the direct Isaac task.

2026-07-04 direct adaptive Isaac sweep:
`scripts/isaac/run_adaptive_probe_carry_sweep.sh` ran 5 scaffold cases in
compute allocation `165112` on `server10` under stamp
`20260704_adaptive_direct_sweep1`. Aggregate output:
`experiments/outputs/adaptive_probe_carry_scene_sweeps/adaptive_probe_sweep_20260704_adaptive_direct_sweep1/adaptive_probe_sweep_summary.json`.
Result: 5/5 cases completed, 0 drop cases, 5/5 reached the target threshold
of 0.08 m, minimum support-margin proxy over cases was 0.0769 m, and strategy
counts were `front_carry: 1`, `low_front_carry: 1`,
`chest_supported_slow: 3`. This validates parameterized scene execution and
morphology/load-dependent posture-selection plumbing only. It is still a
kinematic proxy with box pose following, not dynamic robot walking, balance,
contact grasping, or learned carrying.

2026-07-04 velocity-controlled dynamic rigid-body Isaac probe:
`scripts/isaac/build_velocity_controlled_dynamic_carry_scene.py` with launcher
`scripts/isaac/run_velocity_controlled_dynamic_carry_scene.sh` tests a dynamic
torso rigid body with a fixed-joint dynamic payload while avoiding IsaacLab
Articulation/RigidObject tensors. Completed negative smokes:
`20260704_velocity_dynamic_carry_smoke1` on GPU and
`20260704_velocity_dynamic_carry_cpu_smoke1` on CPU both completed, but torso
and box travel stayed 0.0. GPU additionally logged
`PxRigidDynamic::setLinearVelocity(): it is illegal to call this method if
PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled`. CPU produced no travel either,
showing that runtime writes to USD `RigidBodyAPI.velocity` are not an effective
control path. `CONTROL_MODE=physx_force` was also tested and is negative so
far: CPU/direct-step smokes showed 0 travel, and GPU force mode hit direct-GPU
`addForce()`/`addTorque()` restrictions.

`SKIP_EXPLICIT_STATE_RESET=1` is diagnostic-only. It may be used to isolate
Articulation/RigidObject reset-write failures, but any result with that switch
must not be reported as carrying success.

## Non-Retargeting Rule

- Do not turn the project into human-to-robot joint retargeting, motion
  shadowing, teleoperation replay, or end-effector trajectory cloning.
- Human, robot, or simulation video may be used as a weak reference for task
  phase, progress, object displacement, contact-location priors, and success
  or failure cues.
- Video must not be treated as a command to copy human joint angles, body
  posture, arm trajectories, footstep timing, or grasp geometry.
- Retargeting, teleoperation, and behavior cloning methods may be used only as
  baselines or data sources when explicitly labeled as such.

## Active-Probing Requirement

- RGB or RGB-D video alone cannot determine object mass, center of mass,
  friction, internal fill, stiffness, or required carrying force.
- A valid policy must include active probing behaviors such as micro-lift,
  push-pull, grip-force ramping, stance adjustment, footstep repositioning,
  hold-height adjustment, arm/torso contact redistribution, and gait-speed
  modulation.
- A valid world or belief model must represent uncertainty over object
  dynamics and update that belief from probing feedback.
- Do not claim video-conditioned success if probing is absent or if unknown
  load properties are secretly provided as privileged inputs.

## Embodiment-Aware Carrying Requirement

- The policy must adapt to robot morphology and limits: height, mass, limb
  lengths, joint ranges, torque limits, hand/forearm/chest contact geometry,
  foot support polygon, balance controller, and actuator thermal or effort
  limits.
- The same reference video should not force the same posture across different
  robot bodies. A successful method should choose different feasible carrying
  strategies when morphology or load changes.
- Required strategy space includes at least: front carry, low carry,
  chest/torso-supported carry, asymmetric carry, regrasp, stance widening,
  squat depth adjustment, and walking-speed reduction.

## Evidence And Metrics

- Required evidence for any real claim:
  synchronized scene video, object pose, estimated load belief, contact state,
  robot joint states, torque or effort cost, CoM/ZMP or balance margin,
  footsteps, slip/drop/contact-loss events, and safety events.
- Required metrics:
  carry distance, carry duration, drop rate, slip, contact loss, fall rate,
  recovery after perturbation, object acceleration, energy or torque cost,
  peak joint torque, balance margin, probing attempts, and posture diversity
  across robot bodies and load distributions.
- Harder held-out tests must vary object weight, center of mass, shape,
  size, friction, handle availability, robot morphology, and reference-video
  embodiment.

## Success Claim Gate

A success claim requires all of the following:

- It beats the strongest declared baseline on harder held-out tasks.
- It has no safety regression in falls, drops, excessive torque, object
  acceleration, or collision/contact-force limits.
- It shows that video conditioning improves over no-video RL or scripted
  probing without collapsing into retargeting.
- It shows that active probing improves over video-only or privileged-static
  inference.
- It shows morphology-dependent posture selection, not one fixed pose copied
  across robots.
- It includes ablations for no-video, wrong-video, mismatched embodiment
  video, retargeting baseline, behavior-cloning baseline, no-probing,
  oracle-load, and corrupted or delayed force/contact feedback.

Anything weaker is a diagnostic, engineering milestone, or negative result.

## Official Code And Serious Method Rule

- Use official repositories, released checkpoints, and faithful configs when
  claiming comparison to a serious method.
- Do not hand-roll toy VQ-VAE, toy Transformer, toy world model, toy humanoid
  controller, or simplified video-conditioned policy and present it as
  serious-method progress.
- If official weights, code, assets, or environments are unavailable or
  incompatible, document that as a blocker or comparison gap.
- Simplified code is allowed only when clearly labeled as a diagnostic or
  interface smoke test.

## Experiment Reporting Rules

- Every experiment action must be recorded in the relevant plan, TODO, or
  report with command, config, environment, output path, and status.
- A counted real-training attempt must be at least one hour inside a
  Curiosity-owned tmux-held Slurm allocation, with GPU-utilization evidence,
  exact command/log, config, checkpoint or failure record, and held-out
  evaluation.
- If the same blocker or debugging loop repeats more than 3 times without
  resolution, stop, list the issue clearly for the user, and wait for approval
  or next instructions.
- Newly generated rollout or visualization videos must be MP4 files. Do not
  generate AVI as active evidence.

## Git And Commit Rules

- Do not commit unless the user explicitly asks for a commit.
- The worktree may already be dirty. Do not revert user or unrelated changes.
- Never run destructive commands such as `git reset --hard` or
  `git checkout --` unless the user explicitly requests that operation.

## Workspace Layout

- Source code belongs under `src/`.
- Official external repositories belong under `external/`.
- Documentation belongs under `docs/`.
- The active research idea belongs under `IDEA/idea.md`.
- Active plans belong under `PLAN/`.
- Active task tracking belongs under `TODO/`.
- Old local material belongs outside the repo archive unless the user
  explicitly asks to restore it.
- Logs belong under `logs/`.
- Experiment outputs belong under `experiments/outputs/`.
- Visual outputs belong under `experiments/visuals/`.
- Experiment configs belong under `experiments/configs/`.
- Experiment reports belong under `experiments/reports/`.
- Large datasets belong under `data/`.
- Checkpoints belong under `checkpoints/`.
