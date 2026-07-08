# Plan 02: Base Simulated Box Loco-Manipulation Environment

## Purpose

Build the base simulated box-carrying environment directly in Isaac before
adding unknown-load probing or video priors.

The official Isaac Lab-Arena G1 loco-manipulation task remains a useful
reference and possible baseline, but it is no longer the first blocking target.
The first target is a controlled scene that we own and can extend:

```text
floor + carry box + target marker + configurable mass/shape/pose
+ later robot embodiment + probing/carry objectives
```

Current local entry points:

- `scripts/isaac/build_minimal_carry_scene.py`
- `scripts/isaac/run_minimal_carry_scene.sh`
- `scripts/isaac/build_core_world_simapp_staged_free_box_carry.py`
- `scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`
- `scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py`
- `scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`
- `scripts/isaac/run_official_policy_locomotion_smoke.py`
- `scripts/isaac/run_official_policy_locomotion_smoke.sh`

Useful Arena reference files, not blockers:

- `external/IsaacLab-Arena/isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py`
- `external/IsaacLab-Arena/isaaclab_arena_gr00t/lerobot/config/g1_locomanip_config.yaml`
- `external/IsaacLab-Arena/isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml`
- `external/IsaacLab-Arena/isaaclab_arena_g1/`

## 2026-07-04 Active Isaac Route

Do not wait for extra external models before building the scene. The active
route is now:

1. Build and verify direct Isaac task scaffolds that we fully control:
   physical box, target, support-state metrics, fall/drop checks, and staged
   approach/probe/lift/carry phases.
2. Move from fixed-payload diagnostics to a staged free-box scene: the box
   starts as a free dynamic rigid body, then a logged attach placeholder is
   triggered after a staged lift/hold event. This is not contact grasping, but
   it establishes the task interface.
3. Treat the current fixed-joint attach route as a negative result until it no
   longer produces disjoint-joint warnings, snapping, high post-attach error,
   or drop events. The current passing route is
   `ATTACHMENT_MODE=kinematic-pose-lock`, which is a task scaffold only.
4. Replace the staged attach placeholder with real contact, hands, or a
   constraint formulation that preserves measured relative pose without
   snapping after the scene can run reliably.
5. Replace velocity-commanded carrier motion with verified controller-backed
   legged or humanoid locomotion after the object/task path is stable.
6. Add active probing actions and unknown mass/shape/COM randomization after
   the free-box and carry metrics are reliable.

2026-07-05 update: the direct Isaac core-World dynamic quadruped path has a
passing articulated-scaffold diagnostic,
`20260705_core_world_dynamic_quad_diag39b_proxy_preplaced`. It uses a custom
USD articulated quadruped with measurable joint motion, a staged free dynamic
box, body-aware target hold, and palm/chest/shelf/front-stop contact proxies.
It passed the diagnostic gate with fall/drop events 0, nonzero torso and box
travel, and final target distance below 0.08 m. This becomes the immediate
base-scene continuation path. It is still not final robot carrying because
base pose assist and staged proxy placement remain in the loop.

2026-07-05 no-root update: do not keep waiting for external models, but also
do not keep tuning root shortcuts. The direct Isaac path now has strict
root-write checker gates and a labeled `SUPPORT_DRIVE` diagnostic. `diag47b`
(`20260705_core_world_dynamic_quad_diag47b_support_drive_dynamic_pads_no_root`)
used staged free box, contact proxies, support pads, and zero root pose /
velocity / angular-velocity writes. It failed with 70 fall events, 53 drop
events, no target hold, final target distance 4.11404 m, max tilt 3.19234 rad,
and late non-finite PhysX state. `diag48`
(`20260705_core_world_dynamic_quad_diag48_stand_fixed_payload_no_root`) reduced
the problem to fixed payload, zero target speed, zero gait amplitude, no
support drive, and zero root writes; it still failed with 20 fall events and
max tilt 2.81150 rad. The next route is therefore not video, model download,
or staged-carry tuning; it is a no-root stand/balance controller diagnostic
inside Isaac.

2026-07-05 follow-up stand diagnostics `diag49`-`diag53` kept root pose,
linear velocity, and angular velocity writes at zero while varying neutral
posture, stance width, foot size, payload mass, friction, and hip/knee PD.
All failed. Wider feet and lower payload helped, and high friction/high PD
reduced fall count, but high gains produced non-finite PhysX state while
moderate gains regressed. This means the current custom two-DOF vertical-leg
carrier is not a viable base for the final carrying objective without a real
stand/balance controller or a redesigned/official controller-backed robot.
Do not spend the next iteration on staged free-box carrying until no-root
fixed-payload stand passes.

Official Go2/H1/ANYmal policy examples are now optional baselines or future
controller replacements. They are not prerequisites for constructing the box
carrying scene. If an official route fails, record the failure and continue
with the direct Isaac task path instead of waiting on model downloads.

2026-07-05 direct-scene correction: the immediate route is direct Isaac
scene/control construction, not external model waiting and not MuJoCo fallback.
Rerun `20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating`
passed the staged-free-box scaffold gate with nonpenetrating carry geometry,
dynamic contact proxies, attach step 340, target hold 97, fall/drop events 0,
and no disjoint warning. This restores the working Isaac task scaffold as the
current base. A heavier `chest_supported_creep` rerun
`20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete`
was safe but failed the strict target/contact gate, so the next direct Isaac
work should tune chest-supported contact closure and target hold while keeping
the low-front scaffold as the passing baseline. This still does not satisfy
final robot-carrying success because the carrier is a velocity-commanded
support-proxy body rather than an articulated foot-contact robot controller.

2026-07-05 follow-up: dynamic proxy mass/thickness knobs were added so
chest-supported contact closure can be tuned without hard-coded proxy bodies.
The first stronger-proxy attempt, `diag57`, and a short health smoke, `diag57a`,
both stalled during SimulationApp startup on `server10` before scene creation.
Do not infer anything about the contact tuning from those runs; rerun in a
fresh allocation that first passes a short Isaac startup smoke.

2026-07-05 server46 result: `diag61` verified Isaac startup and staged scene
construction in a non-`server10` allocation. `diag62` then passed the strict
staged-free-box scaffold gate for `chest_supported_creep` using stronger
dynamic contact proxies, restoring posture-diversity coverage alongside
low-front `diag54`. The new `--require-no-root-shortcut` gate correctly fails
`diag62` because the carrier is still a velocity-commanded support-proxy body.
Next implementation must target an articulated foot-contact carrier that
removes root velocity commands and box pose writes rather than further tuning
the current scaffold.

2026-07-05 correction after user direction: stop waiting on external model
downloads unless they directly unblock the Isaac scene. The active execution
path is now:

1. Keep staged-free-box as the task scaffold and metric harness only.
2. Keep `diag54`/`diag62` as evidence that the box, target, contact-proxy
   closure, posture labels, and target-hold metrics can run in Isaac.
3. Do not tune this scaffold further as if it were a robot. Its summaries now
   explicitly expose root velocity/pose writes and cannot pass
   `--require-no-root-shortcut`.
4. Implement or swap in a real articulated carrier in Isaac and first pass a
   fixed-payload no-root stand diagnostic.
5. After no-root stand passes, add slow no-root locomotion with fixed payload.
6. Only after no-root locomotion is stable, reconnect the staged free-box
   contact-proxy task and then replace staged attach/proxies with real contact
   or a physically defensible constraint.

## What Counts

This phase counts only as reproduction/base environment setup. It does not
claim unknown-load carrying, active probing, or video-guided learning.

Required evidence:

- exact command;
- environment version;
- commit;
- output directory;
- log;
- one MP4 rollout or documented reason video is unavailable;
- scene/object description;
- whether the box mass/shape/load was known or randomized.

## Exit Criteria

- A base Isaac box scene runs in compute allocation with verified rigid-body
  gravity/collision.
- A robot embodiment can be inserted without breaking the box scene.
- It records object pose, robot state, contact/drop/fall status if available.
- A short report states which parts are inherited and which parts are missing
  for the target research problem.
