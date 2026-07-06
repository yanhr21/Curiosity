# Plan 03: No-Root Articulated Carrier In Isaac

## 2026-07-05 G1 Setup And PD Stand Update

## 2026-07-06 Direct Isaac G1 Baseline Suite

User correction: stop blocking on external models, checkpoints, policy-server
rollouts, or large downloads when they are not directly useful. The immediate
execution path is direct Isaac scene construction and explicit gates.

Added `scripts/isaac/run_core_world_g1_direct_carry_baseline_suite.sh`. It
directly calls `build_core_world_g1_box_scene.py` instead of relying on older
wrapper batches. The suite stages are:

1. `stage0_nobox_stand`: G1 stand, no carry box.
2. `stage1_fixed_payload_stand`: G1 stand with collision-enabled fixed torso
   payload.
3. `stage2_freebox_cradle_stand`: free dynamic box on the small torso cradle.
4. `stage3_freebox_short_carry`: 420-step staged-march short target-directed
   free-box carry gate.
5. `stage4_freebox_long_hold_validation`: 700-step long-hold validation gate.

Default `STRICT=0` means every stage writes a summary and checker report even
if one stage fails. Use `STRICT=1` only when the desired behavior is fail-fast.
The suite output root is:

```text
experiments/outputs/core_world_g1_direct_carry_baseline_suite/${SUITE_STAMP}/
```

Lightweight checks passed on the login node:

```text
bash -n scripts/isaac/run_core_world_g1_direct_carry_baseline_suite.sh
python3 -m py_compile scripts/isaac/check_core_world_g1_box_scene_summary.py scripts/isaac/build_core_world_g1_box_scene.py
```

No simulation was run on the login node. At the time of this update, Slurm
showed only the exclusion-zone `reflex_p03_rev20_0706` job running, so no new
Curiosity compute run was started.

`curiosity_g1_isaaclab_pose_retry2_0705`, Slurm job `166918`, completed.
The IsaacLab config root quaternion `(0, 0, 0.7071, 0.7071)` failed when used
directly as Core API wxyz: `diag15_retry2` and `diag16_retry2` had
`fall_events=260` and first-step roll near `1.57 rad`. Identity orientation
with no pelvis xform, IsaacLab gains, and setup joint-state write
(`diag17_retry2`) was better but still failed with `fall_events=139`,
`max_tilt_rad=1.10835`, and `min_robot_z_m=0.32789`.

Added pitch/roll rate terms to the simple balance feedback controller and
submitted `curiosity_g1_setup_pd_stand_0705`, job-name `g1_setup_pd`, no-box
stamps `diag18`-`diag20`. Do not run payload/free-box until no-box stand
passes.

Result: retry3 `166922` completed after compute-side syntax checks. The best
run was `diag18_retry3`: no box, 43 joints, setup joint-state write, arena
gains, identity root, `completed_steps=320`, `fall_events=7`,
`max_tilt_rad=0.93557`, `min_robot_z_m=0.54179`. It failed only near the end
by slow forward pitch. PD feedback variants were worse: `diag19_retry3`
`fall_events=71`, and `diag20_retry3` `fall_events=195`.

Submitted static no-box posture/height sweep
`curiosity_g1_static_posture_sweep_0705`, job-name `g1_post_sweep`, stamps
`diag21`-`diag24`.

Result: `diag22` passed the no-box G1 stand gate for 360 steps with
`fall_events=0`, `max_tilt_rad=0.00882`, and `min_robot_z_m=0.78429`.
The successful posture is `stand_hip_pitch=-0.12`, `stand_knee=0.30`,
`stand_ankle_pitch=-0.15`, root z `0.78`, arena gains, setup joint-state
write enabled. `diag21`, `diag23`, and `diag24` failed.

Next gate submitted: fixed-torso ballast stand
`curiosity_g1_fixed_payload_stand_0705`, job-name `g1_payload_stand`, stamps
`diag25`-`diag27`, testing 0.5/1/2 kg with box collision disabled for the
first ballast isolation test.

Result: fixed-payload stand passed for all three ballast masses with collision
disabled. `diag25` 0.5 kg, `diag26` 1 kg, and `diag27` 2 kg all completed
360/360 with fall/drop 0. Max tilt rose with mass but stayed small:
`0.01514`, `0.01990`, and `0.03265` rad.

Next: open-loop march was corrected to use the successful stand target as its
base posture. Submitted `curiosity_g1_openloop_march_0705`, job-name
`g1_march_smoke`, stamps `diag28` no-box and `diag29` 1 kg fixed payload.

Result: both open-loop march smokes passed stability, but travel was tiny.
`diag28` no-box reached max robot travel `0.00509 m`; `diag29` 1 kg fixed
payload reached max robot travel `0.01354 m` and max box travel `0.01463 m`,
with fall/drop 0. This is not yet carrying.

Submitted `curiosity_g1_march_creep_sweep_0705`, job-name `g1_march_creep`,
stamps `diag30`-`diag32`, to test larger open-loop march amplitudes for
short no-root fixed-payload travel.

## Purpose

Build the robot side of the carrying task directly in Isaac. External models,
datasets, and video priors are not blockers for this phase.

The existing staged-free-box scene is kept only as the object/task harness:
free dynamic box, staged approach/probe/lift/carry phases, target metrics,
fall/drop metrics, support metrics, and posture labels. It is not a robot
success because the carrier still uses body root velocity commands.

## Active Gate

A run is not a robot-carrying result unless all of these are true:

- `articulated_carrier_enabled=true`
- `articulated_joint_count > 0`
- `foot_contact_drive_enabled=true`
- `body_root_velocity_command_count == 0`
- `body_root_pose_write_count == 0`
- `box_pose_write_count == 0` after initialization/setup
- fall/drop events are 0
- the run was executed inside a Curiosity-owned tmux-held Slurm allocation

The existing staged-free-box checker flag `--require-no-root-shortcut` is the
minimum gate for this.

## Execution Path

### Stage 0: Keep The Box Harness

Use `diag54` and `diag62` only as evidence that the Isaac object scene and
contact-proxy scaffold can run. Do not tune this path further as if it were a
robot controller.

Current useful files:

- `scripts/isaac/build_core_world_simapp_staged_free_box_carry.py`
- `scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`
- `scripts/isaac/check_staged_free_box_summary.py`

### Stage 1: No-Root Fixed-Payload Stand

First build or swap in an articulated carrier that can stand with a fixed
payload without any root pose, root velocity, or root angular-velocity writes.

Minimum diagnostic:

```text
fixed payload, target speed 0, 240+ steps,
root writes 0, fall/drop 0, no non-finite state,
positive joint count, foot-contact drive enabled
```

This stage may use a simple controller-backed quadruped/humanoid if it is
available locally, or a redesigned custom articulation if it can satisfy the
gate. It must not use torso/root velocity commands as locomotion.

### Stage 2: No-Root Fixed-Payload Creep

After stand passes, add slow commanded forward travel while keeping the fixed
payload attached physically. Required evidence:

- nonzero torso/body travel
- nonzero payload travel
- root writes remain 0
- fall/drop 0
- support/contact state logged
- target distance decreases or target hold is reached

### Stage 3: Reconnect The Free Box

Only after fixed-payload no-root creep passes, reconnect the staged free-box
scene. The box should begin as a dynamic free object. Any attach/contact
placeholder must be explicitly logged.

Required first free-box run:

- approach/probe phase logged
- attach/contact event logged
- root writes 0
- box pose writes 0 after initialization
- fall/drop 0
- final target distance, relative error, and contact/proxy gaps reported

### Stage 4: Replace Scaffolds

Replace staged attach and contact proxies with a physically defensible contact
or constraint formulation. Only after this should active probing and video
conditioning be connected.

## Current Blocker

The blocker is not missing videos or model downloads. The blocker is that the
current working box scene is moved by a velocity-commanded dynamic body. The
next implementation must produce a no-root articulated carrier diagnostic.

## 2026-07-05 First Implementation Path

The first no-root carrier candidate is now a prismatic-leg articulated stand
diagnostic:

- `scripts/isaac/build_core_world_prismatic_carrier_stand.py`
- `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`
- `scripts/isaac/check_prismatic_carrier_stand_summary.py`

It uses a free torso, four driven vertical prismatic leg joints, four physical
feet in ground contact, and a fixed physical payload. This intentionally
targets Stage 1 only: no-root fixed-payload stand. It should not be described
as walking or free-box carrying until later stages pass.

2026-07-05 update: Stage 1 no-root fixed-payload stand passed, and Stage 2
short no-root fixed-payload creep passed for two fixed-payload configurations:
a front 6 kg payload and a close 10 kg payload. The first Stage 3 attempt,
passive `top_contact_free_box`, failed because the free box did not drop but
also did not move far enough with the carrier. The next Stage 3 implementation
should add a physical tray, side rails, clamp, or a defensible contact
constraint rather than relying on passive top friction alone.

2026-07-05 tray implementation: `build_core_world_prismatic_carrier_stand.py`
now has `payload_mode=tray_contact_free_box`. The tray deck, side rails, and
front/rear stops are physical rigid bodies fixed to the carrier torso, while
the box remains a free dynamic rigid body. This is the next Stage 3 candidate:
it is still a scaffold, but it removes welded payload attachment for the box.

2026-07-05 tray/free-box evidence update:

## 2026-07-07 G1 Wrapper Stop Rule And Closed-Loop Diagnostics

The narrow G1 low-carry pass remains useful as a source replay, but broad
posture/load validation failed. The following branches are now negative and
must not be recycled as if they are promising final solutions:

- terminal chest-pad geometry/timing rescue
- non-pad final freeze / stronger balance rescue
- late rescue / slow policy-to-stand blend
- 0.5 kg target-window scalar arrest / small reverse brake

Current queued diagnostics move one step beyond scalar schedules:

1. `CASE_SET=box_progress_controller`, Slurm job `169004`, uses a measured
   box target-directed progress controller and box lateral controller to
   override the Agile policy velocity command.
2. `CASE_SET=box_progress_retention`, Slurm job `169006`, adds a retention
   posture feedback layer driven by box-robot relative error and box tilt.

If these fail on the 0.5 kg cases, the next plan is to stop wrapper-level
correction around the existing Agile policy and switch to a materially
different locomotion/retention formulation:

- a controller-backed load-aware policy trained or fine-tuned in the current
  Isaac scene,
- a whole-body controller with explicit box pose/tilt objectives and support
  constraints,
- or a simpler no-root articulated carrier/contact controller that can pass
  fixed-payload and free-box gates before being replaced by a humanoid policy.

The evidence required for any future success claim remains unchanged:
root pose writes, root velocity writes, and box pose writes must be zero during
rollout; fall/drop events must be zero; target-window hold must pass; and the
result must survive held-out posture/load checks, not only the narrow 0.25 kg
low-carry replay.

- `diag1` and `diag2` showed that the tray/free-box scene can run without
  root/body/box/payload pose writes and without immediate drops, but the first
  checker gate was insufficient: the torso moved toward the target while the
  payload slid in the opposite X direction. The checker now has payload-target
  and payload relative-offset gates to prevent this false pass.
- `diag3` and `diag4` showed that rear-loading the box against the tray stop
  can destabilize the open-loop carrier badly, causing falls and drops.
- `diag5_stand_wide_center_server36` passed a no-root free-box tray stand gate
  with a wider stance, centered 2 kg payload, no horizontal legs, fall/drop 0,
  max tilt `0.00588 rad`, max torso drift `0.00444 m`, min payload z
  `0.70280 m`, and max payload relative-offset error `0.07579 m` during
  settling.
- `diag6_wide_center_micro_creep_server36` failed when horizontal x-slide legs
  were re-enabled: fall events 564 and drop events 583. The failure happens
  before any meaningful free-box carrying claim and points to the horizontal
  x-slide locomotion scaffold, not the static tray/free-box scene.

Current blocker: the prismatic x-slide creep mechanism is useful as a
fixed-payload diagnostic, but it is not a stable locomotion base for free-box
carrying. The next implementation should stop tuning this x-slide gait and
replace it with one of:

- a controller-backed IsaacLab robot that already has stable locomotion,
- a quasi-static foot-placement controller with explicit support-polygon and
  velocity/acceleration limits,
- or a constrained cart/table diagnostic that is clearly labeled as
  non-locomotion and used only to develop free-box contact handling.

2026-07-05 directive update: do not wait on external models, downloaded
checkpoints, G1/WBC, or MuJoCo fallback when they are not immediately useful.
The G1/WBC assets can be loaded locally, but the minimal Isaac G1 stand smoke
stalled before scene setup on `server63`. The MuJoCo no-assist fallback also
failed balance/travel gates. Therefore the active route is direct Isaac scene
construction first.

Immediate direct-Isaac scaffold evidence:

- `20260705_quasistatic_direct_continue_diag1_server63` reran the pure
  `SimulationApp` quasi-static fixed-payload carry scaffold in Slurm job
  `165744` on `server63`.
- It completed 420/420, selected `low_front_creep`, moved body and physical
  fixed payload `0.16585 m`, ended `0.01415 m` from the target, had
  payload-relative error `0.0 m`, min support margin `0.13252 m`, and fall/drop
  0.
- It is not final robot carrying because it still uses 420 body root velocity
  commands and has no articulated carrier or foot-contact drive.

Next implementation priority: keep the direct Isaac task interface and replace
the velocity-commanded torso with either a cleaner quasi-static foot-contact
controller or a controller-backed Isaac robot. External models may be used only
after they unblock this replacement; they are no longer on the critical path.

2026-07-05 follow-up: a no-root `stance_translate` variant was added to the
prismatic carrier, and a dedicated launcher now forces tray/free-box arguments.
The first valid tray/free-box run, `diag3_server63`, failed with large side
drift, falls, and drops despite zero root/body/box/payload pose or velocity
shortcuts. This rules out the simple synchronous x-slide support mechanism as a
stable carrying base. The next no-root implementation should change the support
mechanics/controller, not merely tune x-slide gains.

2026-07-05 low-CG cage update: the open tray was replaced with a physical cage
top-lid option while keeping the box as a free dynamic body. Heavy/high cage
geometry kept the box but destabilized the carrier. A low-CG cage variant then
passed stand and 1.5 cm no-root carrying diagnostics with zero root/body/box/
payload shortcuts. The same mechanism failed to extend to 3 cm. This upgrades
the available scaffold from static tray stand to short no-root free-box cage
translation, but it is still not a walking robot. Next step: replace
synchronous stance translation with repeated foot repositioning or a real
controller-backed locomotion policy.

2026-07-05 direct-Isaac correction: repeated-foot variants on the low-CG cage
did not break the 1.5 cm saturation. Both `creep` and newly added
`sync_inchworm` completed safely with fall/drop 0 and no root/body/box/payload
shortcuts, but failed the 3 cm travel/target gate. The prismatic cage remains
valuable as a contact/free-box stability scaffold; it should no longer be the
main locomotion path.

2026-07-05 no-root cradle baseline correction: the current strongest
direct-Isaac scaffold is the `cradle_free_box` prismatic carrier, not external
model loading. `retry9` reproduced a strict-pass 8 kg free-box sync-inchworm
baseline with zero root/body/box/payload shortcuts and final post-settle
payload target distance `0.00076 m` for target `-0.23 m`. This is still a
custom prismatic scaffold, not humanoid walking.

Walking-like support-switching is now the active improvement path. `retry10`
showed safe `quasistatic_step_cycle` and `prelift_quasistatic_step_cycle`
runs on the same 8 kg free-box cradle, but they under-shot the `-0.23 m`
target. `retry11` shortened the target to `-0.17 m` but exposed a metric
issue: stdout travel included settle drift, while the real post-settle
payload travel was only `0.132/0.137 m`, so it is not a pass.

Implementation update for the next gate: `--gait-drive-target-x` separates
diagnostic gait drive distance from the reported task target. `retry12`
evaluates `TARGET_X=-0.17` while driving the step-cycle gait internally with
`GAIT_DRIVE_TARGET_X=-0.23`. If it passes, the claim is only:
short-distance walking-like no-root/free-box prismatic scaffold carrying in
Isaac. It is not active probing, video-conditioned RL, humanoid carrying, or
final autonomous posture selection.

`retry12` passed this short-distance gate for both
`quasistatic_step_cycle` and `prelift_quasistatic_step_cycle`: fall/drop 0,
all shortcut writes 0, 8 articulated joints, max tilt `0.09174 rad`, min
payload z `0.71612 m`, and final post-settle payload target distance
`0.00543/0.00350 m` for the `-0.17 m` target. The next plan step should not
be more model waiting. It should either add audit-quality Isaac scene video,
add active-probing hooks to this scaffold, or transfer the same no-root
free-box carry task to a more robot-like/humanoid locomotion backend.

The active path is now direct Isaac G1/robot scene construction, not waiting on
external models. Local G1 WBC assets load (`43` DOFs,
`G1DecoupledWholeBodyPolicy`), but the current minimal IsaacLab G1 WBC scene
fails before rollout because the IsaacLab tensor simulation view is invalidated
when reading DOF positions. A direct Core API G1+box scene was added and runs,
but the USD reference path currently initializes the robot root near the ground
instead of a valid standing pelvis pose, even after binding to the discovered
`pelvis` articulation root. Next implementation priority:

- fix G1/USD root initialization in the direct Core API scene, or
- fix the IsaacLab `InteractiveScene` / tensor-view lifecycle in
  `build_minimal_carry_scene.py`, then run stand -> walk -> fixed-payload
  balance diagnostics without GR00T or other external policy servers.

2026-07-05 pivot after user correction: do not keep waiting on G1/WBC,
official locomotion controllers, model downloads, or external policy servers
when they do not immediately unblock the task. The direct Isaac scene path is
now the active path. The next concrete diagnostic is a multi-posture anchored
scene sweep that uses the same randomized hidden box seed across several carry
postures, runs active probing before carry, and ranks the resulting telemetry.
This is not a full robot success gate; it is the scene/task skeleton for
"try several postures and choose the least bad one" without relying on
retargeting or downloaded models.

New files:

- `scripts/isaac/run_direct_isaac_anchor_posture_sweep.sh`
- `scripts/isaac/summarize_anchor_posture_sweep.py`

Required interpretation:

- Passing this sweep only means the direct Isaac scene can execute posture
  alternatives and report a chosen candidate.
- It does not prove free-walking humanoid carrying, unknown-load belief
  calibration, or video-conditioned RL.

2026-07-05 current correction: do not wait on Arena/G1 persistent walking
because the repeated failure is before rollout: `Failed to get DOF velocities
from backend`, `completed_steps=0`. The active executable path is direct Isaac.

The immediate direct-Isaac diagnostic is now:

- `scripts/isaac/run_probe_then_adaptive_carry_strict_support_diag.sh`
- `scripts/isaac/summarize_probe_then_adaptive_carry.py`

It runs a short randomized-load probe, selects one of `front_mid`,
`low_front`, or `chest_high` with a hand-coded probe-risk rule, and then runs
the selected 64 cm carry on the current strict
`alternating_anchor_feet + cradle_free_box` backend. This is a practical
execution bridge toward active posture choice, but it remains a scaffold:
not RL, not video conditioned, and not full humanoid free walking.

Result:
`20260705_probe_then_adaptive_carry_strict_support_seed7055` passed as a
diagnostic. The randomized hidden box was `8.24950 kg` with nonzero COM offset.

2026-07-05 directional support-placement update:

The direct task-runner scaffold now supports a logical
`alternating_placement_feet` backend. Internally this still uses the anchored
support-foot physical launcher, but it mirrors swing/stance foot placement by
target direction and records explicit placement fields:
`support_foot_placement_mode`, `support_foot_placement_controller_enabled`,
and `support_foot_directional_placement`.

Validation status:

- Positive target single posture:
  `20260705_task_runner_directional_placement_seed7081_retry3` completed
  `3660/3660` with fall/drop `0`, target `0.64 m`, final post-settle box
  travel `0.65735 m`, final target distance `0.01735 m`, active-probe belief
  available, no hidden-ground-truth probe use, and directional placement true.
- Negative target single posture:
  `20260705_task_runner_directional_negative_seed7082_retry` completed
  `2200/2200` with fall/drop `0`, target `-0.32 m`, final post-settle box
  travel `-0.35174 m`, max target-directed post-settle travel `0.37768 m`,
  final target distance `0.03174 m`, and directional placement true. This
  required fixing reward/travel metrics to use target-directed progress rather
  than assuming positive X.
- Shared-hidden-box multi-posture sweep:
  Slurm job `166850`, stamp
  `20260705_task_runner_directional_postures_seed7083_server02`, completed on
  `server02` with exit `0:0` after `00:02:21`. The shared hidden box was
  `5.91337 kg`, size `[0.33273, 0.26142, 0.22331] m`, COM offset
  `[0.01569, -0.01343, 0.01675] m`. `front_mid`, `low_front`, and
  `chest_high` all completed `3660/3660`, fall/drop `0`, active probe belief
  available, no hidden-ground-truth probe use, root shortcut free, and PhysX
  contact-report support gates passed.

Current interpretation:

This is the strongest executable direct-Isaac task skeleton so far: hidden
randomized box, active probing, multiple posture candidates, target-directed
progress, explicit support/contact gates, and directional foot-placement
bookkeeping. It remains a scaffold. It must not be presented as learned
locomotion, video-conditioned RL, real humanoid walking, or complete
unknown-object carrying.

Next implementation priority:

Do not wait on model downloads or external controllers. Keep the same
task-runner contract and replace the current anchored support scaffold with a
more physically meaningful Isaac scene component:

- first choice: a support-foot/contact controller that moves through actual
  repeated support placement rather than anchored rail/cradle motion;
- second choice: a controller-backed robot only if it can enter rollout
  immediately in this Isaac environment;
- contact-development side path: improve the free dynamic box contact/cradle
  while preserving the active-probe and posture-ranking interface.

2026-07-05 immediate execution correction after user feedback:

Do not block on external models, WBC assets, official controller wrappers, or
downloaded checkpoints when they do not enter rollout quickly. The current
practical execution path is:

1. keep the direct Isaac task-runner contract stable;
2. keep randomized hidden boxes and active probing enabled;
3. make each scaffold improvement explicit and auditable;
4. replace scaffold parts one at a time, starting with support/contact
   mechanics rather than video/model integration.

The first small follow-through was to expose feedback step-controller
parameters through `run_direct_carry_task_runner_episode.sh` and run a
directional placement episode with explicit feedback gains:
`20260705_task_runner_directional_feedback_seed7084`. It completed on
`server02` in Slurm job `166853` with runner status `pass`, hidden box mass
`7.68171 kg`, `3660/3660` completed steps, fall/drop `0`, post-settle box
travel `0.62166 m`, final target distance `0.01834 m`, active probe belief
available without hidden ground truth, directional placement true, feedback
controller enabled, and `feedback_step_applied_steps=3570`.

Interpretation: this is useful direct-Isaac scene progress and shows the
feedback-support interface can be controlled from the task runner. It is not
a final success claim. Formal checker jobs for this specific run were
canceled pending due Slurm priority, so this run is runner-pass plus manual
JSON audit, not checker-validated evidence.

2026-07-05 follow-up, checker-validated feedback posture sweep:

The feedback/direct-placement path now has an in-allocation multi-posture
validation instead of a separate pending checker. `run_task_runner_active_probe_postures.sh`
passes feedback/gait parameters through to each episode and chooses the
expected checker controller mode automatically for
`SUPPORT_MODE=alternating_placement_feet`. The summarizer records feedback
fields and requires feedback steps for pass.

Slurm job `166859`, stamp
`20260705_task_runner_directional_feedback_postures_seed7085`, completed on
`server36` with exit `0:0` after `00:01:44`. The shared hidden box was
`10.22545 kg`, size `[0.35601, 0.24706, 0.22236] m`, COM offset
`[-0.03576, 0.00871, 0.01431] m`. All three carry postures passed strict
checker in the same allocation:

- `front_mid`: post-settle travel `0.61728 m`, target distance `0.02272 m`.
- `low_front`: post-settle travel `0.64413 m`, target distance `0.00413 m`.
- `chest_high`: post-settle travel `0.61305 m`, target distance `0.02695 m`.

All had `3660/3660` steps, fall/drop `0`, root shortcut free, active probe
belief without hidden ground-truth use, directional placement true, feedback
controller enabled with `3570` applied steps, and contact-report gates passed.

This is now the cleanest direct-Isaac scaffold gate for "same hidden box,
multiple postures, probe before carry, maintain balance metrics while
carrying." It still does not satisfy the final goal because the backend is a
scaffolded support-foot/cradle carrier, not a complete robot with learned or
controller-backed natural walking.

Next replacement target:

Preserve this exact task contract and checker, then replace one scaffold part
at a time. The highest-value replacement is support/contact mechanics:

- remove or reduce the anchored/cradle simplification;
- make support progression depend on physically meaningful repeated foot
  placement/contact rather than rail/cradle support;
- keep the same hidden-box, active-probe, multi-posture, feedback, contact,
  fall/drop, and target-distance gates for every replacement attempt.

2026-07-05 slip-audit correction:

The checker-validated feedback sweep still permits a major non-walking
artifact: support feet can remain near the ground while sliding quickly. This
is not acceptable as evidence of a robot that walks while carrying.

New audit support:

- `check_direct_carry_task_summary.py` can enforce both
  `--max-near-ground-foot-speed` and `--max-near-ground-foot-slip`.
- `run_check_direct_carry_task_runner_episode.sh` exposes these as
  `MAX_NEAR_GROUND_FOOT_SPEED` and `MAX_NEAR_GROUND_FOOT_SLIP`.
- `summarize_task_runner_active_probe_postures.py` records per-foot and max
  near-ground speed/slip.
- `direct_carry_task_shell_backend.py` now allows parent-env overrides for
  stance duration, step length, support-foot drive gains/limits, and friction
  so slip-reduction sweeps can be run without editing Python.

Negative evidence:

`20260705_task_runner_directional_slow_slip_audit_seed7086` used
`STANCE_STEPS=160` and a strict `MAX_NEAR_GROUND_FOOT_SPEED=0.8` audit. It
completed the carry safely but failed the new walking-realism gate:

- completed `7000` steps;
- fall/drop `0`;
- final post-settle box travel `0.64886 m`;
- final target distance `0.00886 m`;
- max near-ground foot speed `1.05842 m/s`;
- max near-ground foot slip `0.69295 m`.

This means merely slowing the stance phase does not make the current
support-foot scaffold physically acceptable. The next implementation should
change the support mechanics so planted feet stay planted, or explicitly
model a valid contact/constraint mechanism, then rerun the same
hidden-box/probe/posture/feedback/contact/slip gates.

2026-07-05 direct task-runner update: the active route is now the explicit
task-runner interface rather than external model waiting. The task runner
separates reset, action, policy observation, hidden evaluation context, and
backend summary export. A three-posture active-probe sweep completed under one
shared hidden box seed:

- stamp `20260705_task_runner_active_probe_postures_seed7080`
- Slurm job `166822`, `server02`, exit `0:0`, elapsed `00:01:56`
- shared hidden box mass `10.72455 kg`, size
  `[0.36519, 0.22971, 0.23912] m`, COM offset
  `[0.03732, 0.00523, -0.00058] m`
- postures `front_mid`, `low_front`, `chest_high` all completed `3660/3660`
  steps with fall/drop `0`
- probe belief was available and did not use hidden ground truth for all three
  postures
- final post-settle box travel / target-distance:
  `front_mid` `0.64775 / 0.00775 m`, `low_front` `0.67183 / 0.03183 m`,
  `chest_high` `0.65476 / 0.01476 m`

Interpretation: this validates scene/task bookkeeping for "probe, compare
postures, carry, and report metrics" in Isaac. It is still an anchored
scaffold backend. The next code step is not model download; it is to harden
the direct task/controller interface so the same randomized box, probing,
posture-action, observation, reward, done, and summary schema can be reused
when the locomotion backend is replaced.

2026-07-05 backend-contract hardening: the direct task runner now requires an
auditable backend capability declaration. Current capability fields explicitly
distinguish the present `anchored_support_scaffold` from a future trainable or
walking controller:

- current backend id: `physical_alternating_anchor_feet_cradle_v1`
- `free_dynamic_box=true`
- `active_probe_supported=true`
- `trainable_policy_backend=false`
- `real_robot_morphology=false`
- `support_switching_supported=false`
- `video_conditioning_supported=false`
- `scaffold_backend=true`

The validated export
`experiments/outputs/rl_interface/20260705_contract_caps_export_retry2/direct_carry_task_episode_table.jsonl`
contains three rows from the active-probe posture sweep with these fields. This
is the interface that the next real Isaac controller must satisfy.

2026-07-05 support-placement step: added a direction-aware alternating foot
placement mode to the current direct Isaac backend. This is still a scaffold,
but it removes one hard-coded gait assumption: swing/stance X targets can now
mirror with the requested travel direction. The new backend path is:

- `SUPPORT_MODE=alternating_placement_feet`
- backend id `physical_alternating_placement_feet_cradle_v1`
- backend family `directional_foot_placement_scaffold`
- summary fields:
  `support_foot_placement_mode=alternating_directional_x`,
  `support_foot_placement_controller_enabled=true`,
  `support_foot_directional_placement=true`

Validated diagnostic:
`20260705_task_runner_directional_placement_seed7081_retry3`, Slurm job
`166833`, completed on `server02` with exit `0:0`. Strict checker retry
`166840` passed with active probe, no hidden-ground-truth belief, no root/body
or box shortcuts, support-foot contact report evidence, and directional
foot-placement gates enabled. Metrics: hidden randomized box mass
`7.23482 kg`, final post-settle box travel `0.65735 m`, final post-settle
target distance `0.01735 m`, fall/drop `0`.

Interpretation: the task runner now has two swappable scaffold backends:
historical fixed-X alternating support feet and the new direction-aware
placement variant. The next milestone is to test morphology/posture variation
and then replace the scaffold with a controller that does not rely on the
anchored support structure.

2026-07-05 negative-direction update: the new direction-aware placement mode
now has a negative-target diagnostic. The initial negative run exposed that
reward and travel-loss metrics were still positive-X biased. The metric stack
now records absolute and target-directed post-settle travel, and the contract
reward uses target-directed progress. Validation:

- stamp `20260705_task_runner_directional_negative_seed7082_retry`
- Slurm job `166845`, `server02`, exit `0:0`
- target `-0.32 m`
- hidden randomized box mass `8.20882 kg`
- final post-settle box travel `-0.35174 m`
- max absolute post-settle box travel `0.37768 m`
- max target-directed post-settle box travel `0.37768 m`
- final post-settle target distance `0.03174 m`
- directional post-settle travel loss `0.02594 m`
- fall/drop `0`
- strict checker job `166846`, `server36`, exit `0:0`, status `pass`

This is useful because it prevents the future policy interface from silently
favoring only positive-X carrying. It is still scaffold locomotion; the next
controller replacement must preserve the same target-directed metric semantics.
The vertical micro-lift probe produced risk `0.607367`, so the hand-coded
selector chose `low_front` with slower gait parameters. The selected 64 cm
carry completed 3580/3580, moved the box `0.67171 m`, ended `0.00361 m` from
the target, had fall/drop 0, no root/box shortcuts, no fixed-world support,
and no strict support-continuity failures. The next plan step is to replace
the hand-coded selector and scaffold support-foot carrier with a trainable or
controller-backed robot backend; do not overclaim this diagnostic as RL or
full robot carrying.

Next gate:
run the same strict 64 cm free-box carry for `front_mid`, `low_front`, and
`chest_high` using one shared randomized hidden box seed. This addresses the
explicit "any carry posture remains balanced while walking" requirement more
directly than a single selected adaptive posture. The new scripts are
`scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh` and
`scripts/isaac/summarize_randomized_all_posture_carry.py`; this remains a
scaffold diagnostic until the carrier backend is replaced by a complete robot
walking controller.
downloaded models, or official policy paths when they are not immediately
unblocking. The direct Isaac scene path now has three factual results:

- Core API G1+box root initialization is fixed enough to place G1 at a valid
  pelvis height in setup, but open-loop stand joint targets are not a balance
  controller. `20260705_core_world_g1_box_scene_diag4_setuppose_server46`
  starts at robot z about `0.799 m` with fall 0, then falls by step 30 and
  drops the box later.
- Official Go2 policy paths are not currently usable in this standalone
  cluster launch path. `official_device_dt` and `dt_only` exit around
  `SimulationManager` setup; `skip_device_dt` enters the loop but registers
  no physics callbacks and keeps `forward_calls=0`; manual-loop Go2 fails with
  an invalid articulation physics tensor entity before policy initialization.
- The low-CG prismatic cage is a stable direct-Isaac free-box/contact scaffold,
  not a walking robot. A stronger x-slide diagnostic
  `20260705_prismatic_stance_translate_strongx_diag1_server46` still saturated
  around `0.01557 m` torso travel and `0.01834 m` payload travel while staying
  safe. This confirms that more x-slide gain tuning is not the path to real
  carrying locomotion.

Immediate next implementation priority: keep the Isaac scene and metrics, but
replace the carrier with a controller-backed robot path that actually
initializes, or write a direct Core API support-foot controller with explicit
quasi-static foot placement instead of synchronized x-slide actuation. Treat
the current cage as a contact/load harness only.

2026-07-05 anchored-support update: added a new direct Core API diagnostic
carrier:

- `scripts/isaac/build_core_world_anchored_footstep_carrier.py`
- `scripts/isaac/run_core_world_anchored_footstep_carrier.sh`

This path replaces direct torso velocity commands with a prismatic joint drive
from a support frame to the torso. It is still not a complete walking robot:
the current passing configuration uses one world-fixed support frame at torso
height, not a multi-step foot-replanting controller. However, it is a real
improvement over the velocity-commanded torso scaffold because the body and
attached payload move through an articulation joint target while root/body/
payload/box pose and velocity shortcut counters remain zero.

First passing fixed-payload result:
`20260705_anchor_footstep_fixed_diag8b_holdtarget_server46` completed 180/180
in Slurm job `165866` on `server46`, with one articulated joint, fall/drop 0,
root/body/box/payload pose and velocity shortcuts 0, max torso travel
`0.03781 m`, max payload travel `0.03781 m`, final target distance
`0.00219 m`, min payload z `0.55720 m`, and max payload relative-offset error
near zero. The checker passed the fixed-payload gate.

First free-box cage result:
`20260705_anchor_footstep_cagedfree_diag1_server46` is negative. It had
fall/drop 0 and no root/box shortcuts, but the free box shot through/along the
cage contact, with final payload target distance `0.79402 m` and max payload
relative-offset error `2.01642 m`. The next free-box work should repair cage
geometry, initial clearances, and contact impulses before claiming free-box
carrying.

Centered free-box cage follow-up:
`20260705_anchor_footstep_cagedfree_diag2_centerbox_server46` reduced the
contact impulse problem by centering the box/cage and using a lighter
`0.5 kg` box. It completed 180/180 with fall/drop 0 and no root/body/box/
payload shortcuts, and the payload moved `0.02232 m` without being expelled.
It still failed carrying because torso travel was only `0.00365 m`, final
torso target distance was `0.03635 m`, final payload target distance was
`0.02186 m`, and max payload relative-offset error remained `0.13000 m`.
Conclusion: centered cage improves contact stability but does not yet transfer
the support-frame drive into a coherent free-box carry.

2026-07-05 anchored-support follow-up:

- Added parameterized cage geometry to
  `build_core_world_anchored_footstep_carrier.py`: cage clearances, wall
  thickness, and cage part masses are now explicit launcher parameters instead
  of hard-coded dimensions.
- `20260705_anchor_cage_compact_diag1_server46` tested a compact centered cage
  with `0.015 m` XY clearance and `0.018 m` Z clearance. It was worse:
  fall/drop stayed 0, but the free box was violently expelled, reaching
  payload travel `9.08616 m` and max payload relative-offset error
  `20.67944 m`. Conclusion: simply tightening the cage amplifies contact
  impulses and is not the free-box solution.
- `20260705_anchor_fixed_8cm_diag3_upper08_server46` passed an 8 cm
  fixed-payload support-frame gate. It completed 260/260 with one articulated
  prismatic joint, fixed 4 kg payload, fall/drop 0, all root/body/box/payload
  shortcut counters 0, max torso/payload travel `0.08003 m`, final target and
  payload target distances `0.000032 m`, min payload z `0.55720 m`, and max
  payload relative-offset error near zero. This shows the support-frame joint
  drive can carry a fixed payload farther than the previous 4 cm diagnostic,
  but it is still a single support-frame motion rather than walking.

Next direct-Isaac priority: convert the fixed-payload support-frame success
into actual support switching, or replace the free-box cage with a low-impulse
tray/grasp/contact constraint. Do not keep tightening cage walls as the main
free-box strategy.

2026-07-05 direct-Isaac correction after the latest user pushback: external
models, WBC assets, and downloaded policies are not on the critical path for
the next step. The active work is direct Isaac scene construction.

Anchored-support was extended with multi-rail/telescoping joints and a
closed-stop latch. This produced safe fixed-payload longer-travel diagnostics
but not precise target stopping:

- `20260705_anchor_telescoping_fixed_24cm_diag1_server36` completed safely
  with fall/drop 0 and no root/body/box/payload shortcuts, but overshot the
  `0.24 m` target: max travel `0.29479 m`, final target distance
  `0.05479 m`.
- `20260705_anchor_telescoping_fixed_16cm_diag1_server36`,
  `diag2_damped_server36`, and
  `diag3_closedstop_server36` stayed safe with fall/drop 0 and no shortcuts,
  but settled at about `0.18291 m` for a `0.16 m` target. The closed-stop run
  latched at step 270; final target distance remained `0.02291 m`.

This confirms the support-frame joint chain can carry fixed payloads farther,
but it is still a support-frame diagnostic, not walking.

Anchored-support was also extended with `payload_mode=staged_grasp_constraint`
to test a low-impulse alternative to tightening the cage. The implementation
now supports a staged free box, optional preparation shelf, and runtime-authored
grasp fixed joint:

- `20260705_anchor_staged_grasp_diag1_server36` used delayed attach at step 20.
  It completed 220/220 with fall/drop 0 and no root/body/box/payload pose or
  velocity shortcuts, but PhysX reported a disjoint `StagedGraspJoint`, the
  payload snapped relative to the torso, max payload relative-offset error was
  `0.23856 m`, and final payload target distance was `0.07996 m`.
- `20260705_anchor_staged_grasp_diag2_runtime_step0_server36` used runtime
  attach at step 0. It also completed safely with fall/drop 0 and no shortcuts,
  but still produced a disjoint-joint warning, max payload relative-offset
  error `0.13740 m`, final target distance `0.01867 m`, and final payload
  target distance `0.12045 m`.

Conclusion: fixed-joint staged attach is not the next main line in this Core
API scene unless the joint frame/snap issue is solved. The next direct-Isaac
free-box path should use a soft/compliant grasp or driven clamp/contact
formulation with low initial impulse, not a tighter cage and not a runtime
fixed-joint snap.

2026-07-05 direct free-box contact/grasp follow-up:

- Added `payload_mode=open_tray_free_box` to test an open support tray with low
  stops, keeping the box as a free dynamic body. `diag1_slow_server36`
  completed 260/260 with fall/drop 0 and no root/body/box/payload shortcuts,
  and without disjoint-joint warnings, but it failed carrying: max torso travel
  `0.01870 m`, final target distance `0.02842 m`, final payload target
  distance `0.06563 m`, and max payload relative-offset error `0.18925 m`.
  `diag2_highstop_slow_server36` made the geometry tighter/taller and was much
  worse: the payload was accelerated to `12.49785 m` scale in the X-cradle
  variant and to `4.43549 m` scale in the high-stop open tray run. Conclusion:
  open tray is cleaner than cage but does not carry; taller/tighter stops
  reintroduce contact impulse failure.
- Added `payload_mode=side_clamp_free_box` with two prismatic side pads. This
  produced a valid 3-DOF articulation and no root/body/box/payload shortcuts,
  but the clamp joints did not meaningfully close: `diag1_slowclose_server36`,
  `diag2_strongclamp_stand_server36`, and
  `diag3_rotaxis_strongstand_server36` all had max clamp motion about
  `5.3e-05 m` for a requested `0.07 m` clamp travel. The box also dropped
  below the active drop threshold in the stand tests. Conclusion: this side
  clamp joint/frame formulation is not yet an effective gripper.
- Added `payload_mode=x_cradle_free_box` to use an X-axis rear pusher because
  the X rail axis is known to move. The rear pusher joint moved
  `0.05205 m`, proving the commanded X cradle DOF is active, but the stand run
  `20260705_anchor_x_cradle_diag1_stand_server36` fired the free box to
  `12.49785 m` during settle. Conclusion: X-axis pusher contact is active but
  currently has unacceptable initial impulse.

Next direct-Isaac free-box priority: separate contact closure from payload
support more carefully. A useful next diagnostic should start with a supported
box and first validate a single moving contact element against a fixed dummy
object or very small mass before combining clamp closure with the carrier rail.
Do not continue tuning tighter tray/cage walls, and do not treat current
side-clamp or X-cradle results as carrying.

2026-07-05 direct Isaac contact-module update after the user correction to stop
waiting on outside models:

- Added isolated single-contact probe files:
  `scripts/isaac/build_core_world_single_contact_probe.py` and
  `scripts/isaac/run_core_world_single_contact_probe.sh`.
- `20260705_single_contact_probe_diag1_slow_server10` was safe but did not
  reach contact: max pusher motion `0.04704 m` for a `0.050 m` gap.
- `20260705_single_contact_probe_diag3_boxx_contact_server10` reached contact
  after adding `BOX_X` / actual surface-gap metrics, but with high friction
  the box barely moved.
- `20260705_single_contact_probe_diag4_lowfric_strong_server10` is the useful
  contact diagnostic: completed 360/360, max joint/pusher travel `0.06979 m`,
  free-box travel `0.04469 m`, max box speed `0.07889 m/s`, min box z
  `0.12999997 m`, fall/drop 0, no root/body/box/payload shortcuts.

The existing anchored `x_cradle_free_box` did not inherit this cleanly.
`20260705_anchor_xcradle_lowfric_stand_diag2_server10` failed: target `0`,
the payload was pushed forward during settle and dropped below the active
threshold, while the cradle joint barely moved (`1.40e-05 m`). This confirms
the anchored X-cradle geometry should not remain the main free-box route.

To preserve the useful contact behavior without the broken anchored cradle,
added a cleaner non-locomotion contact scaffold:

- `scripts/isaac/build_core_world_cradle_cart_free_box_carry.py`
- `scripts/isaac/run_core_world_cradle_cart_free_box_carry.sh`

This scaffold moves a physical tray/cradle through a prismatic `CartRail` while
the box remains a free dynamic rigid body. It is a constrained cart/table
diagnostic, not robot locomotion. The first run failed because box/deck/wall z
positions omitted `cart_z`; after fixing that and adding post-settle metrics,
`20260705_cradle_cart_freebox_diag3_postsettle_8cm_server10` passed:

- completed 420/420 with one articulated `CartRail` joint,
- target `0.08 m`, max cart travel `0.07883 m`,
- post-settle cart travel `0.07881197 m`,
- post-settle free-box travel `0.07881202 m`,
- final post-settle relative error `4.79e-08 m`,
- min box z `0.27615 m`, drop 0, nonfinite 0,
- root/body/box/payload pose and velocity shortcuts all 0.

Interpretation: we now have a clean free-box contact-carry module inside Isaac.
It is not a robot or walking result. The next real robot step is to replace the
cart rail with a robot/support-switching body while keeping the same post-settle
free-box contact metrics.

2026-07-05 anchored cradle integration:

The validated cradle was integrated into
`build_core_world_anchored_footstep_carrier.py` as
`payload_mode=cradle_free_box`. Early runs showed why the previous anchored
free-box attempts were unstable: fixed-joint local positions attached to the
scaled torso were effectively scaled. Before correction, the measured initial
front stop surface gap was `-0.37491 m`, so the stop penetrated the box and
expelled it. The fix uses torso-scale-corrected joint local positions for the
new cradle parts and disables only the internal anchor/torso collision for this
payload mode.

Evidence after the fix:

- `20260705_anchor_cradle_freebox_diag6b_scaledjoint_geom_server23` confirmed
  geometry only: rear/front surface gaps about `0.025 m`, box drift
  `1.19e-07 m`, min payload z `0.7289998 m`, drop 0.
- `20260705_anchor_cradle_freebox_diag7_scaledjoint_8cm_server23` passed an
  anchored free-box 8 cm carry: one `StanceRail`, max torso and free-box travel
  `0.078173 m`, final target distances about `0.00187 m`, final post-settle
  relative error `2.18e-07 m`, fall/drop 0, no root/body/box/payload shortcuts.
- `20260705_anchor_cradle_freebox_diag9_fixed_16cm_2rail_server23` passed an
  anchored free-box 16 cm carry with two rail joints: max torso and free-box
  travel `0.158377 m`, final target distances about `0.00188 m`, final
  post-settle relative error `5.77e-08 m`, fall/drop 0, no root/body/box/
  payload shortcuts.

This is now the strongest free dynamic box carrying scaffold in Isaac. It is
still fixed-support anchored carrying, not walking.

Support switching is the next blocker. The first attempt
`20260705_anchor_cradle_freebox_diag8_supportswitch_16cm_server23` failed
because the code tried to assign a world pose to non-root articulation link
`/World/Robot/StanceAnchor`; PhysX rejected this, the carrier fell, and the box
dropped. The next walking-like implementation must restructure the stance
support so the replanted support is an articulation root, external constraint
target, or separate controller object rather than a teleported non-root link.

2026-07-05 support-switch redesign diagnostics:

The non-root pose write path was stopped and three alternatives were tested on
Slurm job `166028` in tmux session `curiosity_anchor_root_gpu_0705` on
`server53`.

- `20260705_anchor_cradle_freebox_diag10_anchorroot_supportswitch_16cm_server53`
  applied `ArticulationRootAPI` to `/World/Robot/StanceAnchor` and wrote the
  support root instead of a non-root link. This removed the original PhysX
  non-root transform warning, but failed because the support was still a free
  dynamic body: fall events `520`, box drops `455`, max tilt about `3.14 rad`.
- `20260705_anchor_cradle_freebox_diag11_kinanchor_supportswitch_16cm_server53`
  attempted to make that support root kinematic. PhysX rejected the model:
  `ArticulationRootAPI definition on a kinematic rigid body is not allowed`,
  and no articulation was created.
- `20260705_anchor_cradle_freebox_diag12_worldjoint_replant_16cm_server53`
  used an external world fixed-joint target and retargeted the joint at cycle
  boundaries. It was stable with fall/drop 0, but final travel was only about
  `0.0401 m` because the cycle phase reset and the retarget did not create an
  effective accumulated support displacement.
- `20260705_anchor_cradle_freebox_diag13_worldjoint_phasefix_16cm_server53`
  fixed the final-cycle phase reset and stayed stable, but still only reached
  about `0.0799 m`, showing runtime fixed-joint `localPos0` retargeting is not
  an effective support-replant mechanism in this scaffold.
- `20260705_anchor_cradle_freebox_diag14_cumulative_16cm_server53` then added
  an explicitly labeled `cumulative_cycle_target=true` diagnostic. It passed
  560/560 steps, two cycles, target `0.16 m`, max torso/free-box travel
  `0.158374 m`, final target distances about `0.00193 m`, final post-settle
  payload/torso relative error `3.13e-08 m`, fall/drop 0, and no root/body/box/
  payload pose or velocity shortcuts.

Interpretation: diag14 is useful because it proves the current Isaac contact
and articulated rail scaffold can carry a free dynamic box over a multi-cycle
schedule without pose-writing the payload or torso. It is not a true support
switch or walking result, because the working transport comes from cumulative
rail target displacement rather than an effective replanted stance support.
The next implementation should build a separate runtime-effective support
target/root mechanism instead of relying on non-root link pose writes,
kinematic articulation roots, or fixed-joint `localPos0` retargeting.

The cumulative-cycle scaffold was then extended to longer transport and heavier
payloads, still as diagnostics only:

- `20260705_anchor_cradle_freebox_diag15_cumulative_32cm_server53` passed
  980/980 steps with a `0.5 kg` free box, four cycles, target `0.32 m`, max
  torso/free-box travel about `0.31925 m`, final target distances about
  `0.00195 m`, fall/drop 0.
- `20260705_anchor_cradle_freebox_diag16_cumulative_32cm_4kg_server53` passed
  the same 32 cm setup with a `4.0 kg` free box, max travel about `0.31949 m`,
  final target distances about `0.00186 m`, fall/drop 0.
- `20260705_anchor_cradle_freebox_diag17_cumulative_32cm_8kg_server53` passed
  with an `8.0 kg` free box, max travel about `0.31969 m`, final target
  distances about `0.00194 m`, final post-settle payload/torso relative error
  `2.08e-07 m`, fall/drop 0, and no root/body/box/payload pose or velocity
  shortcuts.

Interpretation update: this is now a useful free-dynamic-box, load-bearing
Isaac scaffold with clean accounting, but it is still not a robot walking
controller. The immediate next research-engineering step is a runtime-effective
support-target mechanism or real robot base controller so that distance comes
from foot/support transitions rather than a cumulative rail displacement.

2026-07-05 prismatic-leg cradle/free-box update:

The validated cradle free-box geometry was ported from the anchored carrier
into `build_core_world_prismatic_carrier_stand.py` as
`payload_mode=cradle_free_box`. The port keeps the torso-scale-corrected fixed
joint positions for cradle parts and adds cradle gap plus post-settle active
travel metrics. This matters because the prismatic carrier has physical feet
and no torso/root pose or velocity writes, so it is a more relevant stepping
scaffold than the cumulative rail carrier.

Evidence from Slurm job `166052` in tmux
`curiosity_prismatic_cradle_gpu_0705` on `server53`:

- `20260705_prismatic_cradle_stand_diag1b_8kg_server53` stood for 500 steps
  with an `8 kg` free dynamic box. Rear/front cradle gaps were about
  `0.0224/0.0269 m`, fall/drop were 0, and root/body/box/payload pose or
  velocity writes were 0. The carrier settled backward about `5.6 cm`, and the
  payload relative offset changed by about `6.2 cm`, so this is a stand/contact
  diagnostic, not carrying.
- `20260705_prismatic_cradle_sync_inchworm_diag1_4cm_8kg_server53` and
  `diag2_neg4cm_8kg_server53` showed the first horizontal-leg `sync_inchworm`
  gait stayed safe with an `8 kg` free box, fall/drop 0, but absolute target
  metrics were contaminated by settle drift.
- `20260705_prismatic_cradle_sync_inchworm_diag3b_postsettle_neg4cm_8kg_server53`
  added post-settle metrics. It produced final active torso/payload travel of
  about `-0.01997 m`, only half the `-0.04 m` target, with fall/drop 0.
- `20260705_prismatic_cradle_sync_inchworm_diag4_postsettle_neg8cm_8kg_server53`
  increased stride and produced final post-settle active torso/payload travel
  about `-0.04627 m`, peak active travel about `0.06195 m`, max tilt
  `0.09624 rad`, min payload z `0.7281 m`, fall/drop 0, and no root/body/box/
  payload pose or velocity shortcuts.

Interpretation: this is the strongest current physical-foot, free-box carrying
evidence. It is still a simplified prismatic-legged scaffold, not a complete
humanoid/quadruped walking policy, and it only moved a few centimeters after
settle. The next step is to turn the prismatic stepping controller into a
repeatable distance controller with post-settle target gates, then replace the
prismatic abstraction with a more robot-like leg/controller if it remains
stable under longer distances and heavier held-out boxes.

2026-07-05 extended prismatic cradle/free-box update:

- `20260705_prismatic_cradle_sync_inchworm_diag5_postsettle_neg14cm_8kg_server53`
  safely extended the same 8 kg free-box run to final active post-settle travel
  about `-0.08707 m` and peak active travel about `0.10298 m`.
- `20260705_prismatic_cradle_sync_inchworm_diag6_postsettle_neg22cm_8kg_server53`
  safely extended to final active post-settle travel about `-0.14711 m` and
  peak active travel about `0.16318 m`.
- `20260705_prismatic_cradle_sync_inchworm_diag7_postsettle_neg30cm_8kg_server53`
  completed 2350/2350 with an 8 kg free dynamic box, fall/drop 0, nonfinite 0,
  and zero root/body/box/payload pose or velocity shortcuts. It reached final
  post-settle torso/payload travel about `-0.20588/-0.20587 m` and peak active
  post-settle travel about `0.22180/0.22179 m`, but remained about `0.09412 m`
  short of the `-0.30 m` post-settle target.

Interpretation: this is now the strongest direct Isaac physical-foot free-box
scaffold, and it should be pushed as a scene/controller engineering path
instead of waiting for external models. It is still not full walking: the
carrier is a simplified prismatic-foot mechanism, the controller is scheduled,
and the target is not fully reached. The next immediate test is
`diag8_postsettle_neg40cm_8kg_server53`, which measures whether distance can
extend past the `diag7` plateau.

`diag8_postsettle_neg40cm_8kg_server53` was a useful negative result, not a
distance improvement. Increasing target distance, step length, slide limit, and
drive forces together caused a true dynamics failure: fall events `3126`,
box-drop events `2826`, min torso/payload z near `-1071/-1074 m`, and max
payload relative-offset error `117.19 m`, while all shortcut counters remained
0. The correct next test is not more force; it is a conservative target
extension with the stable `diag7` parameter family, now launched as
`diag9_postsettle_neg34cm_8kg_server53`.

`diag9_postsettle_neg34cm_8kg_server53` also failed. The safer `diag7`
parameter family still became unstable when extended to six sync-inchworm
cycles: fall events `2637`, box-drop events `2182`, min torso z
`-828.23 m`, max tilt `3.09742 rad`, and max payload relative-offset error
`829.85 m`. The payload's final X target metric is invalid because the carrier
had fallen. This points to a cycle-transition/support-stability limit rather
than missing models.

Implementation response: add `--sync-cycle-pause-fraction` to the prismatic
carrier so each sync-inchworm cycle can end with all feet back in support for
a stabilization pause before the next cycle. Lightweight syntax checks passed.
The next run, `diag10_pause_neg34cm_8kg_server53`, tests whether this controller
change can make the sixth cycle stable.

Pause-test status: `diag10_pause_neg34cm_8kg_server53` is not a valid pause
test because its summary recorded `sync_cycle_pause_fraction=0.0`; it failed
like the no-pause six-cycle runs. A compute-node runner inspection showed the
builder had the new argument but the runner path used by the tmux session had
not picked up the new launcher argument. `diag11` was interrupted after this
was discovered. A direct-Python `diag12` explicitly passed
`--sync-cycle-pause-fraction 0.20`, but it destabilized during early
stand/settle and was interrupted instead of counted as a completed result.

Planning implication: do not keep spending runs on the current prismatic
sync-inchworm distance extension. The scaffold is useful up to the five-cycle
`diag7` regime, but six-cycle transport is unreliable. The next direct Isaac
implementation should change support mechanics: explicit quasi-static
foot-placement with support-polygon constraints, an effective support-root
replant mechanism, or a controller-backed robot that initializes reliably.

2026-07-05 feedback support-clock implementation:

`build_core_world_prismatic_carrier_stand.py` now includes
`motion_mode=feedback_sync_inchworm`. It is not a final locomotion controller,
but it changes the direction from open-loop cycle sweeping to a balance-gated
support clock: the next gait step is released only if the previous physics step
had no fall/drop, tilt below a threshold, and payload relative-offset error
below a threshold. It logs hold/release counts and the last block reason so a
failed run says whether the gait was blocked by tilt, fall, drop, or payload
slip.

The launcher now exposes:

- `FEEDBACK_TILT_HOLD_THRESHOLD`
- `FEEDBACK_PAYLOAD_ERROR_HOLD_THRESHOLD`

Lightweight checks passed with `py_compile` and `bash -n`. The next compute
run should test the new mode first on the 8 kg cradle/free-box task at the
known stable `-0.30 m` target, then only if safe try the six-cycle `-0.34 m`
case that failed in `diag9`.

2026-07-05 prismatic gait regression and route correction:

The feedback run exposed a more important issue: the current code path no
longer reproduces the historical `diag7` cumulative travel. A same-parameter
ordinary `sync_inchworm` replay,
`20260705_prismatic_cradle_sync_replay_diag13_neg30cm_stableparams_8kg_server53`,
was safe with fall/drop 0 but only reached about `0.0595 m` peak active
post-settle travel and returned near zero final post-settle travel. This is not
the old `diag7` behavior.

Diagnostic fields were added to
`build_core_world_prismatic_carrier_stand.py` and the checker:

- `max_commanded_leg_lift_m`
- `max_abs_commanded_x_slide_target_m`
- `max_actual_leg_lift_m`
- `max_abs_actual_x_slide_m`

Short command/joint diagnostics on `server53` showed that the controller does
command swing lift (`~0.05 m`) and horizontal slide (`~0.06 m`), while the
sampled physical foot world height stays essentially at ground contact during
the swing samples. The horizontal slide partially tracks, but the vertical
swing/contact response is too weak or too constrained to treat this as a
reliable stepping base.

Planning implication: keep the prismatic cradle/free-box scene as a useful
free-dynamic-box contact and load-bearing diagnostic, but stop presenting the
current prismatic sync-inchworm gait as progress toward real walking. The next
direct Isaac work should build a cleaner carry-task scene/controller interface
that can accept a real locomotion controller later, while preserving the
validated box randomization, cradle/contact metrics, active probing hooks, and
root/body/box shortcut counters.

2026-07-05 direct carry-task interface update:

`scripts/isaac/build_direct_carry_task_scene.py` is now explicitly treated as
a task-scene/controller-interface diagnostic. It supports explicit box
randomization through seed, mass range, and size jitter; writes a
`controller_contract` describing the replaceable controller inputs and outputs;
and records `robot_proxy_pose_write_count` plus
`box_kinematic_pose_write_count` so the run cannot be mistaken for no-root
robot carrying. A dedicated checker,
`scripts/isaac/check_direct_carry_task_summary.py`, enforces the non-success
claim and the proxy-write disclosure.

Compute smoke
`20260705_direct_carry_task_interface_rand_smoke1_server53` ran 180 steps in
the Curiosity tmux/Slurm session on `server53` with explicit randomization
(`BOX_SEED=7051`, mass range `4-10 kg`, size jitter `0.12`). The sampled box
was `7.2301 kg` and about `0.593 x 0.379 x 0.382 m`. It completed 180/180,
box-drop events 0, max box travel `0.67485 m`, final target distance
`0.03485 m`, robot proxy pose writes `2340`, and box kinematic pose writes
`180`. This is useful as a clean task interface and regression harness, not as
robot locomotion or physical grasping evidence.

2026-07-05 physical backend normalization update:

The direct carry-task interface now has a first swappable physical backend
wrapper:

- `scripts/isaac/run_direct_carry_task_physical_backend.sh`
- `scripts/isaac/normalize_direct_carry_backend_summary.py`

The wrapper calls the anchored/cradle free-box backend and normalizes its
summary into the direct-task schema. The normalized summary records
`controller_mode=physical_anchored_cradle`, the backend summary/log paths,
root/body/box/payload shortcut counters, support-root retarget counters, and a
controller contract stating that anchored world support is not free walking.
`scripts/isaac/check_direct_carry_task_summary.py` now checks both kinematic
proxy and physical backend summaries.

Compute evidence from Curiosity-owned tmux/Slurm session
`curiosity_direct_backend_0705`, job `166173` on `server10`:

- `20260705_direct_physical_backend_anchor_cradle_smoke1_server10` was
  interrupted after revealing a wrapper parameter bug: `RAIL_UPPER=0.04`
  limited four positive rail joints to `0.16 m`, so the 32 cm target could
  never be reached.
- The wrapper default was corrected to `RAIL_LOWER=-0.04`,
  `RAIL_UPPER=0.10`.
- `20260705_direct_physical_backend_anchor_cradle_smoke2_railupper10_server10`
  completed 980/980 with `PAYLOAD_MASS=8.0`, `TARGET_X=0.32`,
  `controller_mode=physical_anchored_cradle`, fall/drop 0, root shortcut free,
  max box travel `0.319915 m`, max post-settle box travel `0.319915 m`, final
  box target distance `0.001939 m`, final post-settle box/torso relative error
  `6.46e-08 m`, support-root pose writes 0, and
  `anchor_world_joint_retarget_count=4`.

Interpretation: this is real physical backend progress beyond the kinematic
task proxy, because the box is a free dynamic object carried by a physical
cradle/rail backend with no payload pose writes. It is still not the final
requested robot result: the support is anchored/replanted through a world-joint
mechanism and is not an unconstrained walking and balancing robot.

2026-07-05 fixed-anchor backend ablation:

The physical backend wrapper now supports `SUPPORT_MODE=fixed_anchor` as a
separate controller mode, `physical_fixed_anchor_cradle`. In this mode the
anchored cradle backend uses one fixed world support and four rail joints to
reach the 32 cm target in a single stance phase. It does not use
`--replant-anchor-world-joint` or `--cumulative-cycle-target`.

`scripts/isaac/check_direct_carry_task_summary.py` now has gates for
`--max-anchor-world-joint-retarget-count` and
`--max-support-root-pose-write-count`.

Compute evidence from Curiosity-owned tmux/Slurm session
`curiosity_fixed_anchor_backend_0705`, job `166184` on `server10`:

- `20260705_direct_physical_backend_fixed_anchor_32cm_8kg_server10` completed
  980/980 with `controller_mode=physical_fixed_anchor_cradle`,
  `backend_support_mode=fixed_anchor`, an 8 kg free dynamic box, fall/drop 0,
  root shortcut free, max box travel `0.322541 m`, max post-settle box travel
  `0.322541 m`, final box target distance `0.001794 m`, final post-settle
  box/torso relative error `6.46e-08 m`, `anchor_world_joint_retarget_count=0`,
  and `support_root_pose_write_count=0`.

Interpretation: the free-box cradle/contact/load-bearing part does not require
world-joint replanting to carry 8 kg over 32 cm. This is a useful ablation, but
it is still a fixed world-support rail backend rather than a walking robot.
The next real milestone remains replacing the fixed support with an actual
support-switching or foot-placement controller.

2026-07-05 Isaac-first route correction:

Do not wait for external models, checkpoints, or datasets before making the
simulated carrying task real. The immediate path is to build the Isaac scene
and controller stack directly. Video-conditioned models and cross-embodiment
methods remain background research only; they are not allowed to block the
core carrying substrate.

Current reliable substrate:

- free dynamic 8 kg box in Isaac
- cradle/contact backend with no box pose writes
- normalized direct-task summary and checker
- fixed-anchor ablation over 32 cm with fall/drop 0
- explicit counters for root shortcuts, support-root writes, and anchor
  retargeting

Current limitation:

- the carrier is still attached to a fixed world support/rail, so this is not
  walking, balance, or full robot carrying.

Next concrete implementation milestone:

- create the next Isaac backend by replacing the fixed world-support rail with
  an explicit support-switching / foot-placement carrier.
- keep `anchor_world_joint_retarget_count=0`,
  `support_root_pose_write_count=0`, and no payload/box pose writes.
- log support contact IDs, support phase, body target, actual support motion,
  box travel, contact loss, drop/fall events, and energy/effort proxy.
- treat any fixed support, kinematic body advance, or hidden payload teleport
  as a diagnostic only.

Interrupted posture diagnostic:

`20260705_direct_physical_backend_fixed_anchor_lowfront_32cm_8kg_server36`
was interrupted around step 410 after entering Isaac. It had fall/drop 0 in
the observed state log and was near the 32 cm target, but it produced no
summary. It is not success evidence and should only be used as a note that
low-front posture did not immediately explode before interruption.

2026-07-05 direct no-root prismatic backend:

Implemented a direct-task wrapper for the existing no-root prismatic legged
carrier:

- `scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh`

The wrapper runs `build_core_world_prismatic_carrier_stand.py` through the
existing compute-only runner and normalizes the result into the direct carry
task schema. Default configuration is deliberately conservative:

- backend support mode: `no_root_prismatic_legged`
- controller mode: `no_root_prismatic_legged_cradle`
- payload mode: `cradle_free_box`
- default mass: `8 kg`
- motion mode: `feedback_sync_inchworm`
- no robot/body/root pose or velocity writes
- no box or payload pose writes

The normalizer and checker were extended so the no-root backend can be judged
with the same gates as the fixed-anchor backend, while also reporting
motion-mode and leg diagnostics such as commanded lift, actual lift, x-slide,
and foot height.

This is not a success claim. The goal of the first compute run is to expose
whether the no-root physical-foot backend can safely move even a short distance
with the free dynamic box. If it fails, keep the fixed-anchor backend only as a
load-bearing/contact reference and debug the no-root legged mechanics directly.

First compute result:

`20260705_direct_no_root_prismatic_cradle_feedback_10cm_8kg_server46` completed
1200/1200 on `server46` with the no-root prismatic legged cradle backend. It
passed structural gates: fall/drop 0, root shortcut free, no robot proxy
writes, no box kinematic writes, no anchor-world-joint retargets, and no
support-root pose writes.

It failed the carrying objective. With a `+0.10 m` target, max positive box
travel was only `0.02458 m`, max post-settle box travel was `0.05242 m`, and
final box target distance was `0.14996 m`, meaning the system settled roughly
`5 cm` opposite the requested direction. This is a useful negative because it
separates two facts:

- the no-root/free-foot/free-box backend is now integrated into the direct
  evidence pipeline;
- the current prismatic gait/contact mechanics do not generate forward
  carrying transport.

Next implementation focus:

- inspect foot contact and commanded-vs-actual horizontal leg phases;
- add support-phase/contact-state logging rather than only aggregate leg
  lift/slide metrics;
- try a simpler quasi-static stance-transfer controller before more
  sync-inchworm tuning;
- keep the fixed-anchor backend only as a load/contact reference, not as the
  main route.

Per-leg diagnostic update:

The no-root prismatic backend now records enough per-leg information to debug
the failed 10 cm run instead of guessing from aggregate values:

- near-ground step count per leg;
- min/max foot z per leg;
- max commanded lift and x-slide per leg;
- max actual lift and x-slide per leg;
- CSV-level `near_ground_foot_count` and `commanded_swing_foot_count`.

Next compute run should repeat the same 10 cm diagnostic with these fields,
then decide whether the immediate fix is contact thresholding, stance/swing
timing, x-slide force, leg target height, or a simpler quasi-static stance
transfer controller.

Quasi-static stance-transfer result:

Implemented `motion_mode=quasistatic_stance_transfer` for the no-root
prismatic backend. The first sign was wrong: with `TARGET_X=+0.10`, the box
moved in the negative direction and final target distance was `0.25298 m`.
The `TARGET_X=-0.10` diagnostic confirmed this was a horizontal leg-command
sign issue. After fixing the mode to use the opposite sign, the corrected
`TARGET_X=+0.10` run completed 1200/1200 with:

- fall/drop 0;
- no root, body, box, payload, anchor-retarget, or support-root writes;
- max box travel `0.05647 m`;
- max post-settle box travel `0.09855 m`;
- final box target distance `0.04855 m`;
- max tilt `0.11319 rad`;
- near-ground foot counts: `fl=1170`, `fr=1170`, `rl=1160`, `rr=1160`.

Interpretation: direct Isaac progress is now concrete. The no-root physical
feet/free-box backend can move the carried box forward without root or box
pose shortcuts, but it only reaches about half of the requested 10 cm final
target and remains a quasi-static prismatic-foot scaffold. The next controller
step is not external models; it is improving stance transfer so the body keeps
more of the commanded displacement and can repeat it over multiple short
steps.

Compensated stance-transfer result:

The target metric was split into absolute displacement and post-settle active
transport. This matters because the no-root carrier settles backward before
active motion starts. Added optional settle-drift compensation for
`quasistatic_stance_transfer`: the effective command becomes
`target_x - settle_drift`.

Run
`20260705_direct_no_root_prismatic_quasistatic_compensated_10cm_8kg_server02`
passed the strongest no-root gate so far:

- 8 kg free dynamic box;
- no fixed world support, no anchor retarget, no support-root write;
- no root/body/box/payload pose or velocity shortcuts;
- fall/drop 0;
- `max_box_travel_x_m=0.10413`;
- `final_box_target_distance_x_m=0.00413`;
- `final_post_settle_box_travel_x_m=0.14621`.

This is a real Isaac no-root carrying milestone, but it is still quasi-static
stance transfer with sliding support feet. It does not satisfy walking.

Step-cycle negative result:

Implemented `quasistatic_step_cycle` to add a one-leg-at-a-time foot reset
after stance drive. This is the first attempt to move from pure stance sliding
toward support switching. It is not successful yet.

- Fast reset retry completed safely but final active transport was
  `-0.00725 m`.
- Slow/high reset reached transient post-settle transport `0.25199 m`, but
  final active transport collapsed to `0.01530 m`.

Interpretation: the current reset phase erases the stance-drive displacement.
This is not just insufficient foot lift. The next implementation should make
support switching conditional: maintain stance-lock on the other feet, reset
one foot only if body/box displacement and tilt stay inside limits, and abort
or hold if reset begins to pull the body backward.

Gated step-cycle negative result:

After user correction, external models are no longer treated as a blocker for
this phase. The direct path is to build the Isaac carrying scene and test the
controller mechanics first.

Implemented `gated_quasistatic_step_cycle` and travel-loss metrics:

- `post_settle_payload_travel_loss_after_peak_m`;
- gated hold/release/recovery counters;
- gated last block reason;
- gated peak post-settle travel.

Run
`20260705_direct_no_root_prismatic_gated_step_10cm_8kg_mgmtserver02`
completed safely on `server10` with fall/drop 0 and no root/body/box/support
shortcuts, but it did not solve support switching:

- max post-settle box travel `0.13806 m`;
- final post-settle box travel `0.04578 m`;
- final box target distance `0.09630 m`;
- travel loss after peak `0.09228 m`;
- last block reason `post_settle_payload_travel_loss`.

Interpretation: detecting backward slip after it happens is too late. The next
support-switching controller must prevent reset-induced pullback up front,
probably by solving stance lock and foot placement before allowing a reset
phase. The gated controller is useful as a diagnostic, not as a successful
walking/carrying method.

Posture sweep milestone:

Added `scripts/isaac/run_no_root_prismatic_posture_sweep.sh` to run the
strongest current Isaac no-root/free-box backend across three carry postures.
The sweep uses `quasistatic_stance_transfer`, not support switching, and is
therefore a scene/control milestone rather than walking.

All three 8 kg free-box diagnostics passed strict direct-task gates on
`server10` in Slurm job `166237`:

- `front_mid`: final box target distance `0.00413 m`, max box travel
  `0.10413 m`, max tilt `0.11319 rad`;
- `low_front`: final box target distance `0.01300 m`, max box travel
  `0.08700 m`, max tilt `0.13019 rad`;
- `chest_high`: final box target distance `0.01290 m`, max box travel
  `0.08710 m`, max tilt `0.10504 rad`.

All had fall/drop 0, root shortcut free, box kinematic writes 0, anchor
retargets 0, and support-root writes 0. This is the current best Isaac scene
baseline for "free box carried in multiple postures by a no-root articulated
carrier." It remains explicitly non-success for the full research goal because
it is quasi-static stance transfer with sliding support feet, with no active
probing, no learned policy, no true support-switching gait, and no morphology
adaptation.

Prelift and guarded support-switching diagnostics:

The next attempt was to avoid dragging a resetting foot through the ground by
splitting reset into lift, horizontal return while lifted, and lower phases.
This exposed real swing commands, but it did not solve carrying.

- `20260705_direct_no_root_prismatic_prelift_step_10cm_8kg_server36`
  completed 1800 steps but was unsafe: `fall_events=1172`, max tilt
  `1.05202 rad`, and final target distance `0.42764 m`. The prelift removed
  some reset pullback but reduced support too aggressively.
- `20260705_direct_no_root_prismatic_guarded_prelift_10cm_8kg_server10`
  added tilt/payload/travel-loss gating. It stayed safe with fall/drop 0 and
  no shortcuts, but only reached max box travel `0.07371 m`; final target
  distance was `0.09373 m`.
- `20260705_direct_no_root_prismatic_guarded_prelift_stride12_10cm_8kg_server10`
  used a larger stride. It stayed safe and preserved post-settle peak travel
  almost exactly (`loss=4.77e-7 m`), but absolute box travel was still only
  `0.05585 m`; the checker failed the raw 10 cm target gate.
- `20260705_direct_no_root_prismatic_guarded_prelift_comp_10cm_8kg_server10`
  enabled settle-drift compensation. It stayed safe and produced real lift
  commands (`max_commanded_leg_lift_m=0.10`) with no root/box/support
  shortcuts, but still failed the direct gate: max box travel `0.05720 m`,
  final box target distance `0.06995 m`, final post-settle target distance
  `0.02786 m`, and post-settle travel loss `0.02715 m`.

Interpretation: the current prismatic-foot scaffold can create a clean
no-root/free-box Isaac scene and can carry 8 kg in several quasi-static
postures, but its support switching remains the bottleneck. The prelift
variant shows that simply lifting one foot before reset is not enough; the
controller needs a stance-lock/foot-placement mechanism that preserves body
and box displacement while the swing foot is repositioned.

Longer guarded-prelift diagnostic:

`20260705_direct_no_root_prismatic_guarded_prelift_20cm_8kg_mgmtserver02`
ran on `server46`, Slurm job `166271`, with `TARGET_X=0.20`,
`STEP_LENGTH=0.14`, `X_SLIDE_LIMIT=0.25`, `STEP_HEIGHT=0.10`,
`GAIT_PERIOD_STEPS=720`, settle-drift compensation, and 8 kg free box. It
completed 2400/2400 with fall/drop 0, root shortcut free, anchor retargets 0,
and support-root writes 0. It reached max box travel `0.15029 m` and max
post-settle box travel `0.19238 m`, but failed the 20 cm target gate: final
box target distance was `0.12198 m`, final post-settle target distance
`0.07990 m`, and post-settle travel loss after peak `0.07227 m`. The
guarded controller entered recovery and then stayed blocked by historical
peak loss for most of the rollout.

Code update after that result: added
`--gated-step-loss-rebaseline-steps`, exposed it through the launchers, and
recorded `gated_step_loss_rebaseline_count` in summaries. This is diagnostic
plumbing only: it allows a run to accept the current stable post-settle travel
as a new baseline after prolonged loss recovery, so the controller can test
later support-switch phases instead of deadlocking on one transient peak. It
must not be interpreted as stable walking.

Loss-rebaseline diagnostic result:

`20260705_direct_no_root_prismatic_guarded_prelift_rebaseline_20cm_8kg` ran
on `server10`, Slurm job `166281`, with the same 20 cm guarded-prelift setup
plus `GATED_STEP_LOSS_REBASELINE_STEPS=120`. It completed 2600/2600 with
fall/drop 0, root shortcut free, anchor retargets 0, support-root writes 0,
and all four legs receiving lift commands. The diagnostic did what it was
meant to test: it allowed continuation after reset-induced loss, with
`gated_step_loss_rebaseline_count=3` and final post-settle travel loss only
`0.00158 m`.

It still failed the 20 cm carry gate. Max box travel remained `0.15029 m`,
final box target distance was `0.10593 m`, and final post-settle target
distance was `0.06384 m`. Rebaselining prevents deadlock, but it accepts lost
transport instead of solving it. The prismatic support-switching scaffold is
therefore useful for Isaac scene diagnostics and logging, but the next serious
implementation should change the support/foot-placement mechanics rather than
continue tuning this gait.

Stance-overdrive diagnostic:

Code update: added `--prelift-stance-overdrive` and launcher/normalizer
plumbing. During a prelift reset, not-yet-reset stance legs can multiply their
x-slide stance target to counter the swing foot's return reaction. This is a
diagnostic mechanism only, intended to test whether reset-induced body/box
pullback is mostly a missing stance-compensation problem.

Active run: `curiosity_guarded_overdrive_20cm_0705`, Slurm job `166289`, was
submitted with `TARGET_X=0.20`, `STEP_LENGTH=0.14`, `X_SLIDE_LIMIT=0.25`,
`STEP_HEIGHT=0.10`, `GAIT_PERIOD_STEPS=720`,
`PRELIFT_STANCE_OVERDRIVE=1.45`, no loss rebaseline, and an 8 kg free box.
The expected interpretation is binary: if it preserves displacement better
than the prior guarded-prelift run, stance compensation is worth developing;
if it fails or destabilizes, the current prismatic support-switching gait
should be replaced rather than tuned further.

First overdrive result:

`20260705_direct_no_root_prismatic_guarded_prelift_overdrive145_20cm_8kg`
ran on `server10`, Slurm job `166289`. It was unsafe: `fall_events=1976`,
max tilt `0.91439 rad`, and final box target distance `0.37494 m`. It did
produce much larger transient box travel (`max_box_travel_x_m=0.59394`), but
that was not controlled carrying; it was an overdriven/falling failure. A
smaller diagnostic, `curiosity_guarded_overdrive115_20cm_0705` with
`PRELIFT_STANCE_OVERDRIVE=1.15` and tighter tilt hold `0.11`, was submitted as
Slurm job `166292` to check whether there is any safe stance-compensation
window before abandoning this line.

Second overdrive result:

`20260705_direct_no_root_prismatic_guarded_prelift_overdrive115_20cm_8kg`
ran on `server10`, Slurm job `166292`. It stayed safe with fall/drop 0 and no
shortcuts, but still failed: max box travel `0.17314 m`, final box target
distance `0.12104 m`, max post-settle box travel `0.21522 m`, and travel loss
after peak `0.09418 m`. Smaller overdrive increased transient travel but made
reset-loss worse. Conclusion: simple stance overdrive is not the missing
support-switching mechanism.

Low-reaction swing-foot diagnostic:

Code update: added `--swing-x-force-scale`, launcher env
`SWING_X_FORCE_SCALE`, and summary fields
`swing_x_force_scaled_steps` / `per_leg_swing_x_force_scaled_steps`. During
steps where a leg has commanded lift, its x-slide drive max force can be
reduced while stance legs keep full x drive. This tests whether swing-foot
return reaction, not stance-drive shortage, is the dominant source of
reset-induced pullback.

Active run: `curiosity_guarded_swingforce_20cm_0705`, Slurm job `166300`, was
submitted with `SWING_X_FORCE_SCALE=0.08`, no stance overdrive, no loss
rebaseline, 20 cm target, and 8 kg free box.

Low-reaction swing-foot result:

`20260705_direct_no_root_prismatic_guarded_prelift_swingforce008_20cm_8kg`
ran on `server10`, Slurm job `166300`. It completed safely with fall/drop 0
and no root/box/support shortcuts, and the summary verified force scaling was
applied for 118 swing-leg steps. The outcome was essentially unchanged from
the non-scaled guarded-prelift run: max box travel `0.15029 m`, max
post-settle box travel `0.19238 m`, final box target distance `0.12198 m`,
final post-settle target distance `0.07989 m`, and travel loss after peak
`0.07227 m`. This indicates that either runtime x-drive force scaling is not
the active limiting factor in this scaffold, or the swing-return reaction is
not the dominant cause of transport loss.

Route correction:

At this point the prismatic support-switching gait has three clear negatives:
plain prelift destabilizes, stance overdrive either destabilizes or worsens
reset loss, and swing x-drive force scaling does not change the result. The
next implementation should stop tuning this gait and instead add a
contact-anchoring / stance-foot-latch diagnostic: explicitly hold stance feet
fixed to the ground while a swing foot is repositioned, count all latch
retargets as non-final scaffolding, and test whether idealized stance locking
can preserve carried-box displacement. If even idealized stance locking fails,
the carrier morphology/controller must be replaced. If it works, the latch can
define the support constraint that a real friction/contact controller must
approximate.

Stance-foot latch diagnostic implementation:

Code update: `build_core_world_prismatic_carrier_stand.py` now supports
`--enable-stance-foot-latch` and `--stance-foot-latch-lift-threshold`. For each
foot, the scene authors a disabled world fixed joint. During rollout, a stance
foot is latched to its current world pose when commanded lift is below the
threshold; the latch is disabled when that foot enters swing. The summary
records `stance_foot_latch_*` counters and per-leg enable/disable/retarget
counts. This is explicitly scaffold evidence, not final walking.

Active run: `curiosity_stance_latch_20cm_0705`, Slurm job `166310`, was
submitted with 20 cm target, 8 kg free box,
`guarded_prelift_quasistatic_step_cycle`, settle-drift compensation, and
`ENABLE_STANCE_FOOT_LATCH=1`. The checker still requires zero root shortcut,
zero box kinematic writes, zero anchor retargets, and zero support-root pose
writes; latch retargets are reported separately as non-final stance-foot
scaffold events.

Stance-foot latch result:

The first launch, Slurm job `166310`, exited during startup after reading a
transient malformed version of the builder; login-node `py_compile` then
passed. Retry
`20260705_direct_no_root_prismatic_stance_latch_retry_20cm_8kg` ran on
`server10`, Slurm job `166313`. It completed 2600/2600 with fall/drop 0, root
shortcut free, anchor retargets 0, support-root writes 0, and latch counters
visible: 27 latch enables, 23 disables, and 27 latch retargets. However it
failed the carry gate and worsened transport: max box travel `0.06334 m`,
final box target distance `0.16388 m`, max post-settle box travel
`0.10900 m`, final post-settle target distance `0.11822 m`, and travel loss
after peak `0.02722 m`.

The log also repeatedly reported PhysX disjoint fixed-joint warnings for the
stance latch joints. Interpretation: runtime world fixed-joint latching is not
a clean support constraint in this scaffold; it overconstrains or snaps the
feet and suppresses transport. The next diagnostic should replace runtime
enable/disable fixed joints with a cleaner stance-anchor design: explicit
support anchors that are part of the articulation/control model from startup,
with measured support switching and retarget counters, rather than
mid-simulation creation/enabling of world fixed joints.

Cleaner support-anchor baseline:

The existing anchored-footstep carrier already authors its stance anchor and
world fixed joint from startup and can replant support by retargeting the
world joint's `localPos0`, avoiding the runtime foot-latch enable/disable
path that produced disjoint warnings. This remains scaffold evidence because
anchor retargeting is not real walking, but it is the cleaner diagnostic for
the support constraint the real controller should approximate.

Active run: `curiosity_support_anchor_replant_32cm_0705`, Slurm job `166321`,
was submitted through `run_direct_carry_task_physical_backend.sh` with
`SUPPORT_MODE=replant_world_joint`, `TARGET_X=0.32`, `PAYLOAD_MASS=8.0`,
four rail joints, and `cradle_free_box`. The checker permits anchor retargets
but still requires no root shortcut, no box kinematic writes, no support-root
pose writes, no falls/drops, and explicit non-success labeling.

Cleaner support-anchor baseline result:

`20260705_direct_physical_backend_replant_anchor_32cm_8kg` ran on `server10`,
Slurm job `166321`, and passed the scaffold gate: 980/980 steps,
`SUPPORT_MODE=replant_world_joint`, 8 kg free cradle box, fall/drop 0, root
shortcut free, box kinematic writes 0, support-root writes 0, max box travel
`0.31992 m`, final box target distance `0.00194 m`, final post-settle box
travel `0.31806 m`, and final box/torso relative error `6.46e-08 m`. It used
`anchor_world_joint_retarget_count=4`; therefore it is cleaner support-anchor
scaffold evidence, not final robot walking. No disjoint/fatal errors were
found in the backend log.

Next diagnostic: run the same replant support-anchor scaffold across
`front_mid`, `low_front`, and `chest_high` carry postures. This does not prove
the final goal, but it tests whether the current free-box cradle and support
anchor can remain stable under posture variation before replacing retargeted
support with a real locomotion controller.

Replant support-anchor posture sweep:

The first posture-sweep launch, Slurm job `166328`, had a shell quoting error:
`$posture` expanded before reaching the compute shell, so it only produced a
default `front_mid` diagnostic under an incomplete stamp. The correctly
escaped retry, `curiosity_support_anchor_postures_retry_0705`, Slurm job
`166331` on `server10`, ran all three postures with `TARGET_X=0.32`,
`PAYLOAD_MASS=8.0`, `SUPPORT_MODE=replant_world_joint`, four rail joints, and
free cradle box.

All three passed the direct scaffold checker with fall/drop 0, root shortcut
free, box kinematic writes 0, support-root writes 0, and
`anchor_world_joint_retarget_count=4`:

- `front_mid`: max box travel `0.31992 m`, final target distance `0.00194 m`,
  max box relative offset error `0.000264 m`.
- `low_front`: max box travel `0.31991 m`, final target distance `0.00194 m`,
  max box relative offset error `0.000264 m`.
- `chest_high`: max box travel `0.31992 m`, final target distance
  `0.00194 m`, max box relative offset error `0.000264 m`.

No disjoint/fatal errors were found in the three backend logs. This is the
strongest current multi-posture free-box carrying scaffold: the box is dynamic
and the run has no root/box/support-root pose shortcut, but support is still
retargeted through an anchored world joint. It therefore defines a stable
task/contact target for the next real-controller replacement, not final
walking.

Support-anchor summary and longer-distance diagnostic:

Code update: `normalize_direct_carry_backend_summary.py` now propagates
support-anchor fields from the backend summary into the direct summary:
`rail_joint_count`, `rail_capacity_m`, `rail_joint_indices`, `cycle_count`,
`stride_m`, `foot_pose_write_count`, and `stance_anchor_pose_write_count`.
These are needed to audit scaffold support switching rather than hiding it
behind a pass/fail gate.

Active run: `curiosity_support_anchor_long64_0705`, Slurm job `166338`, was
submitted with `SUPPORT_MODE=replant_world_joint`, `TARGET_X=0.64`,
`PAYLOAD_MASS=8.0`, `STEP_LENGTH=0.08`, four rail joints, and 1500 steps. The
purpose is to test whether the cleaner support-anchor scaffold remains stable
over a longer carry distance with more support retarget cycles. This is still
not a walking claim.

Long-distance support-anchor result:

`20260705_direct_physical_backend_replant_anchor_64cm_8kg` completed 1500/1500
with fall/drop 0, root shortcut free, box kinematic writes 0, support-root
writes 0, and `anchor_world_joint_retarget_count=8`. It failed the 64 cm
distance gate because the four-rail setup had only `0.4 m` rail capacity:
max box travel was `0.40009 m`, final target distance was `0.239997 m`, and
final post-settle box/torso relative error was `5.03e-09 m`. This is a rail
capacity boundary, not a contact or balance failure.

`20260705_direct_physical_backend_replant_anchor_64cm_8kg_8rail` then repeated
the same target with eight rails and `0.8 m` rail capacity. It passed the
direct scaffold checker: 1500/1500 steps, 8 kg free cradle box, fall/drop 0,
root shortcut free, support-root writes 0, max box travel `0.64583 m`, final
target distance `0.00191 m`, final post-settle box travel `0.63809 m`, final
post-settle box/torso relative error `8.66e-08 m`, and max box relative-offset
error `0.000264 m`. The backend log had no disjoint/fatal errors.

Interpretation: the Isaac scene/contact/load scaffold can now carry a free
dynamic 8 kg box for 64 cm under a support-anchor rail model and multiple
postures at 32 cm. This is useful because it gives the next controller a
concrete target behavior and metrics. It is still not the desired robot result
because the working path uses anchored world support and eight anchor retargets
rather than a real support-switching locomotion controller.

Stricter fixed-anchor ablation:

`20260705_direct_physical_backend_fixed_anchor_64cm_8kg_8rail` reran the
64 cm / 8 kg / eight-rail setup with `SUPPORT_MODE=fixed_anchor` and checker
gates requiring `anchor_world_joint_retarget_count=0` and support-root writes
0. It passed: 1500/1500 steps, `physical_fixed_anchor_cradle`, fall/drop 0,
root shortcut free, no anchor retargets, no support-root writes, no foot pose
writes, max box travel `0.70080 m`, final box target distance `0.00111 m`,
final post-settle box travel `0.63889 m`, final post-settle box/torso relative
error `9.22e-08 m`, and max relative-offset error `0.000264 m`. The backend
log had no disjoint/fatal errors.

Interpretation update: the 64 cm direct Isaac scaffold does not require
support replanting. The remaining non-final mechanism is fixed world support
plus long rail travel. The next implementation should stop adding anchor
retarget variants and instead replace the fixed support with a real
support-switching/foot-placement controller while preserving this direct-task
summary schema and checker gates.

Fixed-anchor posture sweep:

The same fixed-anchor 64 cm / 8 kg / eight-rail scaffold was then checked on
the remaining two posture labels. The tmux checker command had a quoting bug
and tried to read `.` as the summary path, so both summaries were checked
manually on the login node after simulation completed.

- `20260705_direct_physical_backend_fixed_anchor_lowfront_64cm_8kg_8rail`
  passed: 1500/1500, fall/drop 0, root shortcut free, anchor retargets 0,
  support-root writes 0, max box travel `0.70080 m`, final target distance
  `0.00111 m`, final post-settle box travel `0.63889 m`, final post-settle
  box/torso relative error `6.05e-08 m`, and max relative-offset error
  `0.000264 m`.
- `20260705_direct_physical_backend_fixed_anchor_chesthigh_64cm_8kg_8rail`
  passed: 1500/1500, fall/drop 0, root shortcut free, anchor retargets 0,
  support-root writes 0, max box travel `0.70080 m`, final target distance
  `0.00111 m`, final post-settle box travel `0.63889 m`, final post-settle
  box/torso relative error `9.48e-08 m`, and max relative-offset error
  `0.000264 m`.

Together with the prior `front_mid` fixed-anchor pass, the current direct
Isaac scaffold now has a clean three-posture, 64 cm, 8 kg free-box target
under strict accounting: no anchor retargets, no support-root writes, no root
shortcut, no box kinematic writes, and no falls/drops. This should be treated
as the target contact/load behavior for the next controller, not as robot
locomotion.

Dynamic support-foot replacement:

The next backend variant replaces fixed world support with physical support
feet. `SUPPORT_MODE=dynamic_anchor_feet` creates dynamic support feet fixed to
the stance anchor and in ground contact, disables support reposition writes,
and leaves the anchor unfixed from the world. Checker gates require
`--forbid-fixed-world-support`, support-foot mode `fixed_to_anchor`,
support-foot joint count at least 4, anchor retargets 0, support-root writes 0,
foot pose writes 0, and stance-anchor pose writes 0.

First results:

- `20260705_direct_physical_backend_dynamic_anchor_feet_16cm_8kg_frontmid`
  passed at 16 cm / 8 kg: 700/700, fall/drop 0, no fixed world support, no
  support/root/foot pose writes, support-foot joints 4, max box travel
  `0.15831 m`, final target distance `0.00195 m`, and final post-settle
  box/torso relative error `2.99e-08 m`.
- `20260705_direct_physical_backend_dynamic_anchor_feet_64cm_8kg_frontmid`
  passed at 64 cm / 8 kg: 1500/1500, fall/drop 0, no fixed world support, no
  support/root/foot pose writes, support-foot joints 4, max box travel
  `0.66915 m`, final target distance `0.000734 m`, and final post-settle
  box/torso relative error `2.42e-08 m`.

Interpretation: the scaffold no longer needs a world fixed joint for the
`front_mid` 64 cm task. The remaining non-final mechanism is that the support
feet are a rigid support frame fixed to the anchor rather than alternating
robot feet. Before calling this a locomotion result, the next diagnostics must
log anchor/support-foot drift and then replace the rigid support frame with a
true foot-placement or support-switching controller.

Audited dynamic support-foot posture sweep:

After adding anchor/support-foot drift metrics, the dynamic support-foot
backend passed all three carry postures at 64 cm / 8 kg without fixed world
support:

- `front_mid`
  `20260705_direct_physical_backend_dynamic_anchor_feet_audit64_8kg_frontmid`:
  max box travel `0.66915 m`, final target distance `0.000734 m`, final
  post-settle box/torso relative error `2.42e-08 m`, fall/drop 0, anchor
  retargets 0, support-root/foot/stance-anchor pose writes 0, max anchor X
  drift `4.47e-07 m`, max support-foot X drift `4.17e-07 m`.
- `low_front`
  `20260705_direct_physical_backend_dynamic_anchor_feet_lowfront_audit64_8kg`:
  max box travel `0.66915 m`, final target distance `0.000733 m`, final
  post-settle box travel `0.63927 m`, fall/drop 0, anchor retargets 0,
  support-root/foot/stance-anchor pose writes 0, max anchor X drift
  `4.52e-07 m`, max support-foot X drift `4.47e-07 m`.
- `chest_high`
  `20260705_direct_physical_backend_dynamic_anchor_feet_chesthigh_audit64_8kg`:
  max box travel `0.66915 m`, final target distance `0.000734 m`, final
  post-settle box/torso relative error `9.65e-08 m`, fall/drop 0, anchor
  retargets 0, support-root/foot/stance-anchor pose writes 0, max anchor X
  drift `4.32e-07 m`, max support-foot X drift `4.77e-07 m`.

This is now the strongest Isaac scaffold: dynamic free 8 kg box, three
postures, 64 cm transport, no fixed world support, no anchor retarget, no
support pose writes, and measured support-foot drift near numerical noise. It
is still not walking: the support mechanism is a rigid four-foot frame fixed
to one stance anchor. The next controller must break this rigid frame into
support-switching feet while keeping the same direct-task summary and checker
gates.

Foot-driven and alternating-support update:

- `legged_anchor_feet` passed a 16 cm / 8 kg / `front_mid` diagnostic with no
  fixed world support and no root/support/foot pose writes. Motion came from
  X prismatic support-foot drives against ground contact, not from rail
  pulling or anchor retargeting. It remains non-final because all feet are
  driven together.
- The next implementation is `alternating_anchor_feet`, which replaces the
  one-axis all-feet drive with per-foot X/Z prismatic support legs. The first
  goal is not a success claim; it is to verify that the Isaac scene can log
  diagonal stance pairs, swing-foot lift, foot X/Z motion, no root shortcuts,
  and free-box carrying metrics in one summary. A failure is useful if it
  cleanly identifies whether the blocker is foot lift, foot slip, insufficient
  anchor travel, balance, or box retention.

First alternating-support milestone:

`20260705_direct_physical_backend_alternating_anchor_feet_5cycle_holdfix_8cm_8kg_frontmid`
passes a short 8 cm / 8 kg direct-Isaac diagnostic with alternating X/Z
support feet, actual swing-foot lift, no fixed world support, and no
root/support/foot pose-write shortcuts. The important numbers are: 8 support
foot joints, actual support-foot lift `0.06320 m`, max box travel `0.09812 m`,
final box target distance `0.01572 m`, final post-settle box travel
`0.06551 m`, fall/drop 0, anchor retargets 0, support-root writes 0, foot pose
writes 0, and stance-anchor pose writes 0.

This should now replace the rigid support-foot frame as the active locomotion
scaffold. It is still not a full walking robot or unknown-load solution; the
next stage must extend distance and add explicit contact/slip/support-polygon
metrics before connecting active probing or video guidance.

16 cm extension:

`20260705_direct_physical_backend_alternating_anchor_feet_10cycle_holdfix_16cm_8kg_frontmid`
extends the alternating X/Z support-foot scaffold to 16 cm with an 8 kg free
box. It passes the direct checker after manual normalization from the backend
summary: max box travel `0.18576 m`, final target distance `0.00436 m`,
final post-settle box travel `0.15686 m`, actual support-foot lift
`0.06320 m`, fall/drop 0, no fixed world support, no root shortcut, no anchor
retargets, no support-root writes, no foot pose writes, and no stance-anchor
pose writes. New support metrics show many near-ground steps per foot and a
positive support-polygon proxy margin, but `min_near_ground_foot_count=0`
during transition frames, so contact metrics remain diagnostic.

Next: confirm the wrapper array fix with a clean normalized run, then push the
same mechanism to 32 cm and add stricter contact/stance-phase accounting.

64 cm alternating support result:

`20260705_direct_physical_backend_alternating_anchor_feet_40cycle_holdfix_64cm_8kg_frontmid`
passes 64 cm / 8 kg with the alternating X/Z support-foot scaffold:
`0.64785 m` max box travel, `0.01181 m` final target distance, `0.62941 m`
final post-settle box travel, actual support-foot lift `0.06475 m`, fall/drop
0, no fixed world support, no root shortcut, no anchor retargets, no
support-root writes, no foot pose writes, and no stance-anchor pose writes.
This is now the strongest direct-Isaac scaffold and should be the baseline for
multi-posture testing.

64 cm multi-posture alternating support sweep:

`20260705_direct_physical_backend_alternating_anchor_feet_40cycle_holdfix_64cm_8kg_low_front`
and
`20260705_direct_physical_backend_alternating_anchor_feet_40cycle_holdfix_64cm_8kg_chest_high`
both passed the same 64 cm / 8 kg diagnostic gate. `low_front` reached
`0.70662 m` max box travel, `0.03367 m` final target distance,
`0.67882 m` final post-settle box travel, and `0.06348 m` actual support-foot
lift. `chest_high` reached `0.70446 m` max box travel, `0.02583 m` final
target distance, `0.66510 m` final post-settle box travel, and `0.06353 m`
actual support-foot lift. Both had fall/drop 0, root shortcut free, no fixed
world support, anchor retargets 0, support-root writes 0, foot pose writes 0,
stance-anchor pose writes 0, and clean log scans.

This confirms the alternating support-foot backend is not just a single
`front_mid` posture artifact. The next controller work should not wait for
external models. It should turn this direct Isaac physical backend into a
trainable task/controller interface with randomized load/geometry hooks,
explicit probing actions, and stricter support/contact evidence. Known
limitations remain: these are scaffold joints rather than a humanoid or
quadruped controller, `min_near_ground_foot_count` can be 0 during transition
frames, near-ground foot XY speeds are high, and there is still no unknown-load
belief update or video-conditioned reward.

Randomized-load interface update:

The direct physical backend now supports a reproducible randomized payload
interface: seed, mass range, size jitter, and payload center-of-mass offset
range. The sampled physical values are written to both the backend summary and
the normalized direct-task summary. The first smoke,
`20260705_direct_physical_backend_alternating_anchor_feet_randomized_8cm_seed7051`,
passed with sampled mass `8.15343 kg`, sampled size
`0.35775 x 0.25309 x 0.23354 m`, sampled COM offset
`[0.00902, 0.00821, -0.00216] m`, max box travel `0.09614 m`, final target
distance `0.01740 m`, fall/drop 0, and all shortcut counters 0.

This is important plumbing for unknown-load experiments, but it is not active
probing. The next valid research step is to add probing actions and belief
metrics that infer or update uncertainty from motion/contact feedback rather
than exposing sampled mass/COM as policy inputs.

Probe-measurement update:

The backend now has an optional pre-carry `active_probe_push_pull` phase. It
uses a small support-foot X oscillation and records probe torso displacement,
box displacement, relative box error, and final box lag. The carry
post-settle baseline starts after the probe phase, so the task does not count
probe motion as carrying progress. First smoke
`20260705_direct_physical_backend_alternating_anchor_feet_probe_randomized_8cm_seed7052`
passed with a randomized `9.72299 kg` box, `60` probe steps, `0.020 m` probe
amplitude, max probe relative error `0.00835 m`, final probe lag
`0.000325 m`, final target distance `0.02021 m`, final post-settle box travel
`0.11084 m`, fall/drop 0, and all shortcut counters 0.

This is still not active load identification. The next implementation should
turn these probe observations into a belief estimate or at least a calibrated
load/COM proxy, then test whether the controller chooses different carry
parameters under randomized hidden loads.

Probe-belief proxy update:

The first belief proxy is implemented and deliberately conservative. It only
uses probe telemetry: max probe relative box error, final probe lag, probe
amplitude, and probe travel. It does not read sampled mass/COM and records
`probe_belief_uses_hidden_ground_truth=false`. The first smoke,
`20260705_direct_physical_backend_alternating_anchor_feet_belief_probe_randomized_8cm_seed7053`,
produced `probe_compliance_proxy=0.25067`, `probe_lag_proxy=0.04821`,
`probe_risk_score=0.80478`, bucket
`high_observed_load_or_shift_response`, and recommended adjustment
`slow_gait_low_or_chest_supported_candidate`, while still passing the 8 cm
carry gate with fall/drop 0 and all shortcut counters 0.

Next: run controlled light/heavy probe calibration. If the heuristic does not
separate lighter and heavier loads reliably, do not use it for controller
adaptation; replace it with a better estimator based on commanded force,
support-foot motion, object lag, and possibly repeated micro-lift/push-pull
features.

Controlled calibration result:

The 6 kg vs 10 kg probe calibration passed the carry gates but failed as a
belief model. The 6 kg case produced risk `0.78250`; the 10 kg case produced
risk `0.79201`; both landed in the same
`high_observed_load_or_shift_response` bucket. This is too weak to drive
posture or gait adaptation. The next estimator needs additional physically
meaningful signals, starting with support-foot target-vs-actual tracking error
during probe and then effort/force proxies if Isaac exposes them reliably.

Support-foot tracking calibration result:

Adding target-vs-actual support-foot tracking error did not fix the estimator.
The 6 kg and 10 kg cases had nearly identical tracking proxy values
(`2.04663` vs `2.04556`) and stayed in the same high-risk bucket. This means
the current probe is mostly measuring controller tracking behavior, not hidden
payload mass. The next estimator attempt should read measured joint efforts or
joint forces from Isaac's articulation API and test whether effort is more
mass-sensitive.

Measured-effort calibration result:

Measured effort reads from Isaac are now wired through and reliable for this
diagnostic: fixed 6 kg and 10 kg runs both reported effort availability with
zero read errors. The signal is still not useful under the current horizontal
push-pull probe. The 6 kg case measured max/mean support-foot X effort
`459.73468` / `302.97068`; the 10 kg case measured `462.08502` / `303.88600`.
The effort proxy and risk score only changed from `0.004179` / `0.57452` to
`0.004201` / `0.58025`, and both stayed in the same moderate bucket while
passing the same carry gate safely.

This closes the horizontal push-pull probe as a credible load estimator for
this scaffold. Do not adapt posture or gait from it. The next Isaac step should
change the probe mechanics to a vertical micro-lift or partial-unload maneuver
that forces payload weight to appear in measured support/cradle efforts, then
retest 6 kg vs 10 kg before any adaptive controller logic is enabled.

Vertical micro-lift calibration result:

The vertical micro-lift path is implemented with `probe_mode=vertical_micro_lift`,
`probe_z_amplitude_m=0.030`, Z travel telemetry, Z tracking telemetry, and Z
measured-effort telemetry. The fixed 6 kg vs 10 kg retry3 run completed safely
on `server46`, Slurm job `166533`, with fall/drop 0 and shortcut-free summaries.
It is still a negative load-estimator result. The 6 kg case measured max/mean
support-foot Z effort `2371.66748` / `1386.22269`; the 10 kg case measured
`2380.76245` / `1398.41329`. Torso/box Z travel also stayed nearly identical:
6 kg `0.02686 m` / `0.02562 m`, 10 kg `0.02714 m` / `0.02592 m`.

The raw effort signal changes by less than 1%, so this all-feet micro-lift is
not adequate for hidden-load belief or posture adaptation. The implementation
also exposed and fixed a bookkeeping bug: inactive probe axes must not
contribute tracking/effort proxies to risk. The next credible estimator step is
not more threshold tuning; it needs a more direct load-sensitive signal, such as
cradle/box constraint force, contact normal impulse, support reaction force, or
a deliberate weight-transfer probe that changes which contacts carry the box.

Strict support-continuity pivot:

The immediate execution path should not wait for external model/data downloads.
Use the Isaac scene scaffold directly and harden the evidence gate. The next
gate is a 16 cm / 8 kg front-mid free-box alternating X/Z support-foot run with
no fixed-world support, no root/body/box pose shortcuts, and strict
drive-phase support continuity:

- all new support metrics recorded in the backend and normalized direct
  summary;
- `min_drive_near_ground_foot_count >= 2`;
- `drive_near_ground_zero_steps == 0`;
- `drive_near_ground_lt2_steps == 0`;
- `min_commanded_stance_near_ground_foot_count >= 2`;
- `commanded_stance_near_ground_lt2_steps == 0`.

If this fails, treat it as the next engineering target for the Isaac scene,
not as a reason to wait for video models. If it passes, rerun the same strict
gate at 32 cm and 64 cm, then across `front_mid`, `low_front`, and
`chest_high`.

Strict support-continuity results:

The strict front-mid gate passed at 16 cm, 32 cm, and 64 cm. These are still
diagnostics, but they are now stronger than the earlier 64 cm run because the
drive phase records no support discontinuity:

- 16 cm: 1180/1180, max box travel `0.18552 m`, final box target distance
  `0.00242 m`, fall/drop 0, no fixed-world support or shortcut writes,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
- 32 cm: 1980/1980, max box travel `0.38556 m`, final box target distance
  `0.03051 m`, fall/drop 0, no fixed-world support or shortcut writes,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
- 64 cm: 3580/3580, max box travel `0.67301 m`, final box target distance
  `0.02369 m`, fall/drop 0, no fixed-world support or shortcut writes,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.

The 64 cm wrapper produced a post-rollout shell EOF after backend summary/CSV
were written. A separate compute-node normalizer/checker job generated the
direct summary and passed the strict checker. Next: clean that wrapper path,
then run the same strict 64 cm gate for `low_front` and `chest_high` before
treating posture differences as meaningful diagnostics.

Posture extension result:

The same strict 64 cm gate now passes for all three scaffold postures:

- `front_mid`: 3580/3580, max box travel `0.67301 m`, final box target
  distance `0.02369 m`, fall/drop 0, no fixed-world support or shortcut
  writes, `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_lt2_steps=0`.
- `low_front`: 3580/3580, max box travel `0.66675 m`, final box target
  distance `0.00189 m`, fall/drop 0, no fixed-world support or shortcut
  writes, `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_lt2_steps=0`.
- `chest_high`: 3580/3580, max box travel `0.65313 m`, final box target
  distance `0.01468 m`, fall/drop 0, no fixed-world support or shortcut
  writes, `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_lt2_steps=0`.

This satisfies a stronger direct-Isaac scaffold gate for multiple carrying
postures, but it still does not satisfy the true project goal because the
carrier is not a full robot with learned or official locomotion control. The
next plan step is to recover an official robot walking/balance smoke, starting
with the existing G1/Arena path, then connect that walking backend to the box
carry scene instead of continuing to strengthen only the support-foot scaffold.

Randomized all-posture hidden-box result:

The direct Isaac path should keep moving without waiting for external video or
robot models. The same hidden randomized box was tested across all three
current carrying postures under the strict no-fixed-world/no-root-shortcut
support-continuity gate. Slurm job `166633` completed on `server02` with exit
`0:0`; aggregate summary:
`experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_strict_support_64cm_seed7061/randomized_all_posture_strict_support_summary.json`.

Shared hidden box: seed `7061`, mass `6.81119 kg`, size
`[0.32037, 0.22802, 0.23574] m`, COM offset
`[0.01463, 0.02498, 0.00268] m`.

- `front_mid`: 3580/3580, max box travel `0.64402 m`, final target distance
  `0.01039 m`, fall/drop 0, no fixed-world support or shortcut writes,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
- `low_front`: 3580/3580, max box travel `0.68203 m`, final target distance
  `0.01310 m`, fall/drop 0, no fixed-world support or shortcut writes,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
- `chest_high`: 3580/3580, max box travel `0.66133 m`, final target distance
  `0.00638 m`, fall/drop 0, no fixed-world support or shortcut writes,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.

Log scan found no traceback, fatal, disjoint, EOF, tensor failure, or checker
failure; only empty `failures: []` entries appeared. The result means the
current Isaac scene can sustain multiple carry postures on a randomized hidden
box under strict scaffold gates. It does not prove full humanoid walking,
balance control, learned policy behavior, or video-conditioned RL.

Next execution path:

Do not wait for external model downloads. Continue building the Isaac scene
into a research runner:

- expose posture, stance width, hold height, step length, and stance timing as
  controlled action parameters;
- run randomized hidden-box episodes with probe telemetry recorded before
  parameter selection;
- start with a transparent search or bandit-style diagnostic over the existing
  posture/gait parameter space, explicitly labeled as non-RL scaffold;
- only after the action/observation/metric interface is stable, replace the
  search with a real RL policy or video-reward-conditioned policy.

Probe parameter-search scaffold:

Implemented the first version of the direct Isaac research runner that does
not depend on external models. It runs a randomized hidden-box vertical
micro-lift probe, then evaluates five hand-authored posture/gait/stance
candidates on the same hidden box under the strict no-fixed-world,
no-root-shortcut support-continuity gate. The candidates expose the immediate
action interface: `carry_posture`, `stance_steps`, `step_length_m`,
`support_foot_stance_x_m`, and `support_foot_swing_x_m`.

The scoring rule is intentionally simple and auditable:
`final_box_target_distance_x_m + 0.25 * post_settle_box_travel_loss_after_peak_m`
plus large penalties for falls, drops, support discontinuity, or shortcut
support. This is a scaffold search, not RL and not a learned policy.

Submitted compute run: tmux `curiosity_probe_param_search_0705`, Slurm job
`166641`, command
`srun --partition=gpu --gres=gpu:1 --time=05:00:00 --job-name=probe_param_search bash scripts/isaac/run_probe_parameter_search_carry_diag.sh`.
Expected summary:
`experiments/outputs/probe_parameter_search_carry/20260705_probe_parameter_search_carry_seed7067/probe_parameter_search_carry_summary.json`.

Probe parameter-search result:

Slurm job `166641` completed on `server10` with exit `0:0`. Aggregate summary:
`experiments/outputs/probe_parameter_search_carry/20260705_probe_parameter_search_carry_seed7067/probe_parameter_search_carry_summary.json`.
The outer `tee` log was not created because the log directory was opened before
the script created it; the useful logs are the per-case logs under
`logs/probe_parameter_search_carry/` and backend logs under
`logs/core_world_anchored_footstep_carrier/`.

Shared hidden box: seed `7067`, mass `6.15402 kg`, size
`[0.32579, 0.25445, 0.24170] m`, COM offset
`[0.01250, 0.02327, 0.01980] m`. The vertical micro-lift probe completed
720/720, used no hidden ground truth, and produced risk `0.596106` /
`moderate_observed_load_response`.

Candidate outcomes:

- `front_mid_nominal`: pass, selected best; 3580/3580, score `0.00286`, max
  box travel `0.66324 m`, final target distance `0.00286 m`, fall/drop 0,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
- `low_front_slow`: pass; 3580/3580, score `0.01574`, max box travel
  `0.68474 m`, final target distance `0.01574 m`, fall/drop 0,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
- `chest_high_slowest`: reject; fall/drop 0 but
  `drive_near_ground_lt2_steps=10`.
- `front_mid_wide_slow`: reject; fall/drop 0 but
  `drive_near_ground_lt2_steps=16`.
- `low_front_wide_slowest`: reject; fall/drop 0 but
  `drive_near_ground_lt2_steps=36`.

This result is useful because the runner does not merely reward reaching the
target: candidates that move the box but violate strict support continuity are
rejected. It is still not a learned policy or humanoid walking result. The next
credible direct Isaac step is multi-seed evaluation to test whether the
selected candidate changes with hidden box properties and probe telemetry.

Multi-seed parameter-search diagnostic:

Implemented and submitted a 3-seed wrapper around the current direct Isaac
probe parameter-search runner. It runs seeds `7068`, `7069`, and `7070`, each
with one vertical micro-lift probe and five hand-authored posture/gait
candidates, then aggregates whether each seed has a strict passing candidate
and whether the selected best candidate/posture changes.

Submitted compute run: tmux `curiosity_probe_param_multiseed_0705`, Slurm job
`166649`, command
`srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_param_multi bash scripts/isaac/run_probe_parameter_search_multiseed_diag.sh`.
Expected summary:
`experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7068_7070/probe_parameter_search_multiseed_summary.json`.

Multi-seed parameter-search result:

Slurm job `166649` completed on `server02` with exit `0:0`. Aggregate summary:
`experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7068_7070/probe_parameter_search_multiseed_summary.json`.

All 3 hidden-box seeds completed with strict passing candidates:

- Seed `7068`: mass `6.55342 kg`, probe risk `0.608155`, best
  `low_front_slow`, final target distance `0.00660 m`, fall/drop 0, strict
  support continuity passed.
- Seed `7069`: mass `5.36291 kg`, probe risk `0.646449`, best
  `low_front_slow`, final target distance `0.00576 m`, fall/drop 0, strict
  support continuity passed.
- Seed `7070`: mass `9.04893 kg`, probe risk `0.596409`, best
  `low_front_slow`, final target distance `0.00045 m`, fall/drop 0, strict
  support continuity passed.

For every seed, `chest_high_slowest`, `front_mid_wide_slow`, and
`low_front_wide_slowest` were rejected for support-continuity failures even
when they moved the box without fall/drop. The aggregate result is therefore:
`best_candidate_id_counts={"low_front_slow": 3}`,
`best_carry_posture_counts={"low_front": 3}`,
`best_candidate_varied=false`, `best_posture_varied=false`.

Interpretation:

This is useful execution progress because the scene now supports repeatable
active-probe plus parameter-search evaluation across hidden boxes. It is also a
negative result for adaptation: the current candidate set and score collapse to
one conservative low-front option. The next step should make the test harder
and broaden the action space before claiming posture choice: larger mass/COM
variation, lower friction, candidate-specific hold heights, and score terms for
effort or support margin. It remains a scaffold, not RL, not video-conditioned
learning, and not full humanoid walking.

Core API G1 stand-gain gate:

The support-foot scaffold is not enough for the final goal, so the real G1 path
needs a prerequisite gate. The previous Core API G1 attempts loaded the USD and
43 joints but fell under open-loop stand targets. The next test applies
Arena-style stand PD drive gains directly to the G1 USD joints in the Core API
scene, then sweeps root height while attaching a 2 kg box to the torso. This is
not walking and not real carrying; it only asks whether the direct G1 backend
can stand with an attached payload without rollout root/box pose writes.

Implemented:

- `--apply-arena-stand-gains` and `--stand-gain-scale` in
  `build_core_world_g1_box_scene.py`;
- summary fields for applied stand drive gains;
- strict checker `check_core_world_g1_box_scene_summary.py`;
- compute wrapper `run_core_world_g1_stand_height_sweep.sh`;
- aggregate summarizer `summarize_core_world_g1_stand_height_sweep.py`.

If no height passes, the direct Core API G1 path remains blocked at standing
balance and should not be used as the walking/carrying backend until joint
drive/root initialization is repaired or the Arena tensor-view lifecycle is
fixed.

Submitted compute run: tmux `curiosity_g1_stand_height_0705`, Slurm job
`166658`, command
`srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_stand_sweep bash scripts/isaac/run_core_world_g1_stand_height_sweep.sh`.
Expected summary:
`experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep/core_world_g1_stand_height_sweep_summary.json`.

Core API G1 stand-gain result:

Slurm job `166658` ran on `server02` and failed the aggregate strict gate with
exit `1:0`. The failure is a physics/control failure, not a script crash: log
scan found no traceback, fatal, disjoint, backend failure, or unexpected EOF.
Aggregate summary:
`experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep/core_world_g1_stand_height_sweep_summary.json`.

All four height cases applied Arena-style stand gains to 23 G1 joints and had
no rollout root/box pose writes. All four still fell/tilted while carrying the
fixed-torso 2 kg box:

- `z_0p78`: 152 fall events, min robot z `0.40650 m`, max tilt `1.14366 rad`.
- `z_0p84`: 172 fall events, min robot z `0.39728 m`, max tilt `1.18476 rad`.
- `z_0p90`: 191 fall events, min robot z `0.40567 m`, max tilt `1.16031 rad`.
- `z_0p96`: 191 fall events, min robot z `0.39748 m`, max tilt `1.17449 rad`.

This is a negative result for the direct Core API G1 backend: open-loop stand
targets plus Arena PD gains are not sufficient for standing with attached
payload. The next isolation test is no-box standing with the same gains and
height sweep. If that fails too, the blocker is base G1 standing/root
initialization. If it passes, the blocker is payload attachment/posture.

Core API G1 no-box stand isolation:

Submitted compute run: tmux `curiosity_g1_stand_nobox_0705`, Slurm job
`166661`, command
`STAMP=20260705_core_world_g1_stand_height_sweep_nobox ATTACH_BOX_MODE=none EXPECT_ATTACH_BOX=none MAX_BOX_DROP_EVENTS=999 MIN_BOX_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_stand_nobox bash scripts/isaac/run_core_world_g1_stand_height_sweep.sh`.
Expected summary:
`experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep_nobox/core_world_g1_stand_height_sweep_summary.json`.

This test removes the attached payload but keeps the same G1 root-height sweep
and Arena-style drive gains. If it fails, the immediate blocker is base G1
standing/root/drive initialization in the Core API scene. If it passes, the
next Isaac step is payload attachment/posture isolation rather than more
external model downloading.

Core API G1 no-box stand result:

Slurm job `166661` completed on `server36` with exit `0:0`. Aggregate summary:
`experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep_nobox/core_world_g1_stand_height_sweep_summary.json`.
The log scan found no traceback, fatal, backend failure, disjoint articulation,
or unexpected EOF.

The no-box sweep passed 2/4 strict stand cases. `z_0p84` passed with 240/240
steps, fall events `0`, min robot z `0.77672 m`, max tilt `0.23243 rad`, and
max robot XY travel `0.15425 m`. `z_0p96` passed with 240/240 steps, fall
events `0`, min robot z `0.74072 m`, max tilt `0.45388 rad`, and max robot XY
travel `0.34181 m`. `z_0p78` and `z_0p90` failed by falling/tilting.

Interpretation: the direct Core API G1 scene is not completely blocked at base
standing. The previous fixed-torso 2 kg payload sweep failed because the
attachment/load/posture setup destabilizes G1. The next direct Isaac action is
a payload isolation sweep from the passing heights, starting with much lighter
attached masses and only then increasing mass or shifting attachment offsets.

Core API G1 fixed-payload isolation sweep:

Implemented `scripts/isaac/run_core_world_g1_payload_sweep.sh` and
`scripts/isaac/summarize_core_world_g1_payload_sweep.py`. The sweep tests the
no-box-passing heights `0.84` and `0.96` over payload masses
`0.25/0.50/1.00/2.00 kg` and torso fixed-joint attach x offsets
`0.12/0.18/0.24 m`. The strict gate still requires full rollout completion,
positive G1 joint count, Arena-style drive gains, fall/drop 0, no rollout root
pose or velocity writes, no rollout box pose writes, robot/box height gates,
and diagnostic-only success claims.

Submitted compute run: tmux `curiosity_g1_payload_sweep_0705`, Slurm job
`166663`, command
`STAMP=20260705_core_world_g1_payload_sweep_small HEIGHTS="0.84 0.96" MASSES="0.25 0.50 1.00 2.00" ATTACH_XS="0.12 0.18 0.24" srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_payload_sweep bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
Expected summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_sweep_small/core_world_g1_payload_sweep_summary.json`.

This is the direct Isaac path the project should use now: first find a stable
fixed-payload stand boundary, then add stepping/walking. External model
downloads are not on the critical path for this stage.

Open-loop march bridge:

Added a diagnostic-only `open_loop_march` mode to the same Core API G1 scene.
It periodically commands hip/knee/ankle/shoulder joint targets around the
stand pose without writing root pose or velocity during rollout. This is not a
real walking controller and must be expected to fail; its purpose is to expose
whether the current G1 initialization can tolerate leg motion at all before we
wire in a proper controller-backed IsaacLab route.

During the running payload sweep, the launcher was also hardened: optional
`--apply-arena-stand-gains` is now appended through a Bash array rather than a
multiline parameter expansion. One case in job `166663` logged the old
launcher issue after summary writing, so final interpretation of that sweep
must check per-case `run_status.txt` as well as strict gate status.

Submitted open-loop march probe: tmux `curiosity_g1_march_probe_0705`, Slurm
job `166667`, command
`STAMP=20260705_core_world_g1_open_loop_march_probe_small HEIGHTS="0.84 0.96" AMPLITUDES="0.05 0.10" ATTACH_BOX_MODE=none EXPECT_ATTACH_BOX=none MAX_BOX_DROP_EVENTS=999 MIN_BOX_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_march_probe bash scripts/isaac/run_core_world_g1_open_loop_march_probe.sh`.
Expected summary:
`experiments/outputs/core_world_g1_open_loop_march_probe/20260705_core_world_g1_open_loop_march_probe_small/core_world_g1_open_loop_march_probe_summary.json`.

Open-loop march result:

Slurm job `166667` ran on `server36` and failed the aggregate gate with exit
`1:0`. Summary:
`experiments/outputs/core_world_g1_open_loop_march_probe/20260705_core_world_g1_open_loop_march_probe_small/core_world_g1_open_loop_march_probe_summary.json`.
All 4 cases failed by fall/tilt/min-height. The least bad case,
`z_0p84_amp_0p05`, still had 26 fall events, min robot z `0.35212 m`, and max
tilt `1.21204 rad`. The high-root cases reached max tilt near `pi`.

Interpretation: open-loop periodic leg commands are not a walking path. The
next walking work should use either an IsaacLab locomotion controller path or
an explicit feedback balance controller before any carrying-walk claim.

Fixed-payload isolation result:

Slurm job `166663` ran on `server36` and failed the aggregate gate with exit
`1:0`. Summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_sweep_small/core_world_g1_payload_sweep_summary.json`.
No forward fixed-torso payload case passed: 0/24 across heights
`0.84/0.96 m`, payload masses `0.25/0.50/1.00/2.00 kg`, and attach x offsets
`0.12/0.18/0.24 m`.

Two cases were affected by the transient launcher edit and should be excluded
from clean physics interpretation: `z_0p84_m_0p50_x_0p18` (`run_status=127`)
and `z_0p84_m_1p00_x_0p24` (`run_status=2`). The other 22 cases completed the
runner and still failed strict gates by falls, excessive tilt, robot/box height
violations, and sometimes box drops, with no rollout root pose, root velocity,
or box pose writes.

Interpretation: the current forward fixed-torso box setup is not viable even
for light payloads. The next diagnostic should be centered ultra-light
attached ballast using a small box and near-zero torso local offset. If that
passes, the problem is front-mounted moment/attachment geometry. If that fails,
the Core API G1 stand drive cannot tolerate even tiny added fixed mass and the
walking/carrying route needs a stronger controller before any payload work.

Submitted centered ultra-light payload isolation: tmux
`curiosity_g1_payload_centered_0705`, Slurm job `166668`, command
`STAMP=20260705_core_world_g1_payload_centered_ultralight HEIGHTS="0.84 0.96" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.0" ATTACH_Z=0.0 BOX_POS_X=0.0 BOX_POS_Y=0.0 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_center bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
Expected summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight/core_world_g1_payload_sweep_summary.json`.

Centered ultra-light payload result:

Slurm job `166668` ran on `server36` and failed the aggregate gate with exit
`1:0`. Summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight/core_world_g1_payload_sweep_summary.json`.
No centered fixed-payload case passed: 0/8 across heights `0.84/0.96 m` and
payload masses `0.01/0.05/0.10/0.25 kg`. Even `z_0p84_m_0p01_x_0p0` had 39
fall events, min robot z `0.19566 m`, min box z `0.20512 m`, and max tilt
`1.44487 rad`.

Interpretation: the blocker is no longer just forward load moment. The direct
Isaac scene must now isolate collision/contact from the fixed-joint/added-body
effect. The next run uses the same centered ultra-light fixed payload but
disables box collision.

No-collision centered ultra-light payload isolation:

Submitted compute run from Curiosity-owned tmux session
`curiosity_g1_payload_nocoll_0705`, Slurm job `166672`, command
`STAMP=20260705_core_world_g1_payload_centered_ultralight_nocoll HEIGHTS="0.84 0.96" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.0" ATTACH_Z=0.0 BOX_POS_X=0.0 BOX_POS_Y=0.0 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 BOX_COLLISION_ENABLED=0 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_nocoll bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
Log:
`logs/core_world_g1_payload_sweep/g1_payload_nocoll_0705_srun.log`.
Expected summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight_nocoll/core_world_g1_payload_sweep_summary.json`.
Initial status: running on `server36`.

No-collision centered ultra-light payload result:

Slurm job `166672` completed on `server36` with exit `0:0`. Summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight_nocoll/core_world_g1_payload_sweep_summary.json`.
All 8 no-collision centered fixed-payload cases passed the strict stand gate
up to `0.25 kg`, with fall/drop `0`, zero rollout root pose/velocity writes,
zero rollout box pose writes, and no error-log matches. Best 0.84 m case
`z_0p84_m_0p01_x_0p0` had min robot z `0.76835 m`, max tilt `0.31023 rad`,
and max robot XY travel `0.22629 m`. Worst tilt case
`z_0p96_m_0p25_x_0p0` had min robot z `0.69804 m`, max tilt `0.59704 rad`,
and max robot XY travel `0.45718 m`.

Interpretation: tiny fixed added mass is not the immediate blocker. The
collision-enabled centered sweep failed because the payload collider was
interpenetrating or otherwise conflicting with robot/ground contact geometry.
The next direct Isaac step is a collision-enabled clearance sweep: keep small
box geometry and collision on, but place the initial box pose consistently
with the attach offset outside the robot body. Prefer the `0.84 m` root-height
baseline because it has lower tilt and drift.

Payload sweep launcher correction for clearance tests:

`scripts/isaac/run_core_world_g1_payload_sweep.sh` now defaults the requested
box initial pose to `(attach_x, 0, height + attach_z)` when `BOX_POS_*` is not
explicitly provided, and records `box_position_requested_m` in each
`case_config.json`. Lightweight checks passed with `bash -n` and
`python3 -m py_compile`.

Collision-enabled clearance sweep:

Submitted compute run from Curiosity-owned tmux session
`curiosity_g1_payload_clearance_0705`, Slurm job `166673`, command
`STAMP=20260705_core_world_g1_payload_clearance_collision HEIGHTS="0.84" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.18 0.24 0.30 0.36" ATTACH_Z=0.12 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 BOX_COLLISION_ENABLED=1 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_clear bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
Log:
`logs/core_world_g1_payload_sweep/g1_payload_clearance_0705_srun.log`.
Expected summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_clearance_collision/core_world_g1_payload_sweep_summary.json`.
Initial status: pending for priority.

Collision-enabled clearance sweep result:

Slurm job `166673` completed on `server36` with exit `0:0`. Summary:
`experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_clearance_collision/core_world_g1_payload_sweep_summary.json`.
All 16 collision-enabled clearance cases passed strict fixed-payload stand
gate through `0.25 kg`, attach x `0.18/0.24/0.30/0.36 m`, attach z `0.12 m`,
and 0.10 m cube geometry. Fall/drop events were `0`, rollout root
pose/velocity writes were `0`, rollout box pose writes were `0`, and error-log
scan found no traceback, exception, fatal, disjoint articulation,
failed-backend, unexpected EOF, or command-not-found matches.

Best case: `z_0p84_m_0p01_x_0p18` with min robot z `0.76792 m`, min box z
`0.86952 m`, max tilt `0.31326 rad`, max robot XY travel `0.22865 m`, and max
box XY travel `0.27010 m`. Worst tilt case:
`z_0p84_m_0p25_x_0p24` with min robot z `0.75721 m`, min box z `0.82233 m`,
max tilt `0.37794 rad`, max robot XY travel `0.27910 m`, and max box XY
travel `0.32262 m`. Lowest box case: `z_0p84_m_0p25_x_0p36` with min box z
`0.78429 m`.

Interpretation: the current direct Isaac scene now has a stable
collision-enabled fixed-payload standing baseline. This does not solve
carrying: the object is fixed-jointed to the torso, the robot is not walking,
and there is no free-object grasp/contact policy. The next required work is
controller-backed stepping/walking from the stable payload baseline. Do not
return to external model waiting or open-loop march for this step.

2026-07-05 pivot after user correction:

Do not block on additional external model/data downloads. Build the scene
directly in Isaac. The shortest serious path is now:

1. Use the existing Core API G1+box scene as the physics substrate, because it
   already has a stable collision-enabled fixed-payload standing baseline with
   zero rollout root/box pose writes.
2. Add a controller-backed no-box walking gate using the official local
   WBC-AGILE G1 velocity-height ONNX, not a hand-written open-loop gait.
3. If the no-box policy gate passes, rerun the same AGILE policy mode with the
   stable fixed-payload clearance baseline.
4. Only after fixed-payload walking passes should the work move toward free
   box contact/probing. Anything before that is a diagnostic.

Implemented `GAIT_MODE=agile_policy` in
`scripts/isaac/build_core_world_g1_box_scene.py`. It loads the official local
WBC-AGILE recurrent student ONNX and Arena G1 agile config, adapts Isaac Core
G1 joint/root observations into the published policy input layout, and maps
the official 12-leg-joint output back to G1 joint position targets. This is
glue around official weights/configs, not a toy replacement policy. The run
summary records policy inference count, max raw action norm, ONNX/config path,
and the root/box write counters.

First gate to run on compute:

`STAMP=20260705_core_world_g1_agile_policy_nobox_diag1 GAIT_MODE=agile_policy ATTACH_BOX=none STEPS=360 G1_ROOT_Z=0.84 APPLY_ARENA_STAND_GAINS=1 POLICY_START_STEP=40 POLICY_CONTROL_DECIMATION=4 AGILE_COMMAND_X=0.20 AGILE_COMMAND_Y=0.0 AGILE_COMMAND_YAW=0.0 AGILE_HEIGHT_COMMAND=0.72 srun --partition=gpu --gres=gpu:1 --time=01:00:00 --job-name=g1_agile_nobox bash scripts/isaac/run_core_world_g1_box_scene.sh`

Pass condition for this gate: completed steps match request, policy inference
count is positive, fall/drop events are zero, root pose/velocity rollout
writes remain zero, robot XY travel is nonzero, and tilt/root height stay
within the same strict stability envelope. If this fails, inspect whether the
failure is ONNX/runtime input incompatibility or physical instability before
trying fixed payload.

Submitted the first gate from Curiosity-owned tmux session
`curiosity_g1_agile_nobox_0705` as Slurm job `166681`. Log:
`logs/core_world_g1_box_scene/g1_agile_nobox_0705_srun.log`. Expected summary:
`experiments/outputs/core_world_g1_box_scene/20260705_core_world_g1_agile_policy_nobox_diag1/core_world_g1_box_scene_summary.json`.
Initial status: pending for priority.

2026-07-05 correction after AGILE loader failures:

The embedded WBC-AGILE path is not the current execution path. The ONNX route
and the torch-checkpoint route both exited before rollout while loading the
policy inside the Isaac Core process. Do not keep rerunning those loaders
unchanged, and do not wait for additional model downloads before building the
task scene.

The active path is direct Isaac scene construction with the physical backend
that already enters rollout:

- `scripts/isaac/build_core_world_anchored_footstep_carrier.py`
- `scripts/isaac/run_direct_carry_task_physical_backend.sh`
- `scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh`
- `scripts/isaac/run_probe_parameter_search_carry_diag.sh`
- `scripts/isaac/run_probe_parameter_search_multiseed_diag.sh`

Latest strict all-posture result:

Slurm job `166692` completed with exit `0:0`. Summary:
`experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_strict_support_64cm_seed7071/randomized_all_posture_strict_support_summary.json`.
The shared hidden box was `11.47446 kg`, size
`[0.36871, 0.22426, 0.21205] m`, COM offset
`[-0.03709, -0.01539, 0.02677] m`. `front_mid`, `low_front`, and
`chest_high` all passed: 3580 completed steps, fall/drop 0, strict support
continuity, no fixed-world support, and root/box/payload/foot shortcut writes
0. Final box target distances were `0.00315 m`, `0.01245 m`, and `0.01292 m`.

Interpretation:

This is the correct scaffold to keep extending because it actually runs the
unknown-load/randomized-box carrying task in Isaac and exposes posture choices
and support metrics. It is still not full success: the carrier is a scaffolded
articulated support system, not a humanoid walking policy, and the current
strict all-posture run has no learned controller or video reward.

Immediate next work:

1. Use the active-probe plus parameter-search runner as the main experiment
   skeleton, not the frozen AGILE loader.
2. Expand the random seeds and parameter candidates enough to reveal when
   posture choice changes with mass, COM, size, and probe telemetry.
3. Add a clearer posture-cost objective: final target distance, torque/drive
   effort, support discontinuity, tilt, travel loss, drop/slip, and probe cost.
4. Only after that interface is stable, replace the transparent search with
   RL or a video-reward-conditioned policy.

Active-probe parameter-search continuation result:

Slurm job `166694` ran the active-probe plus parameter-search scaffold on
hidden seeds `7071`, `7072`, and `7073`, and completed on `server02` with
exit `0:0`. Aggregate summary:
`experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7071_7073/probe_parameter_search_multiseed_summary.json`.

All 3 seeds passed the diagnostic wrapper. Best posture varied:
`front_mid` won seeds `7071` and `7073`; `low_front` won seed `7072`.
Every seed had fall/drop 0 for the selected candidate, strict support
continuity, and `drive_near_ground_lt2_steps=0`. The strict checker rejected
the same wide slow candidates in every seed because they violated support
continuity despite moving the box, which is the desired behavior for the gate.

This is the strongest current direct-Isaac execution evidence for the user's
requested direction: the scene can run unknown randomized boxes, perform a
probe, evaluate posture/gait candidates, and select different feasible
postures under strict safety/support gates. It remains a scaffold, not RL or
video-conditioned humanoid carrying.

Next direct implementation target:

- Add richer candidate parameters around posture height, support timing,
  stance width, and carry speed instead of only the current five hand-authored
  choices.
- Add effort/cost terms from available drive effort or commanded-force proxies
  to make the selection "省力" rather than only target-accurate and safe.
- Convert the runner output into a clean observation/action/reward interface
  for an RL policy: observation from probe telemetry plus robot/load state;
  action as posture and gait parameters; reward as distance, drop/fall, support
  continuity, tilt, travel loss, and effort.

Expanded action-space result:

Implemented the richer action interface in the direct Isaac parameter-search
runner and verified it on compute. The expanded candidate set varies posture,
stance steps, step length, support-foot stance/swing X, torso height, payload
local X/Z, support-foot step height, double-support fraction, stance
half-length, and stance half-width. Slurm job `166718` ran hidden seeds `7074`
and `7075` with `CANDIDATE_SET=expanded`; summary:
`experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_expanded_7074_7075/probe_parameter_search_multiseed_summary.json`.

Both seeds passed. Best posture varied: seed `7074` selected
`front_mid_nominal`, and seed `7075` selected the new `low_front_cautious`
candidate. Selected candidates had fall/drop 0, strict support continuity, and
no fixed-world support. The run also showed useful rejection behavior: high
clearance and wide slow variants were rejected by strict gates despite moving
the box.

Next target after this result:

- formalize the RL-ready interface file/document: observation keys, action
  parameters, reward terms, termination gates, and required logged evidence;
- then add a small batch runner that emits one JSONL row per candidate episode
  so the same interface can be used by transparent search, future RL, and
  future video-reward conditioning without changing the environment contract.

2026-07-05 feedback stepping route:

Following the user's correction, do not wait on external models when they do
not directly unblock the Isaac scene. The immediate execution route is now a
strict direct-Isaac feedback-step diagnostic:

1. Keep the existing anchored support-foot carrier as the runnable physics
   substrate, but expose rail motion explicitly with `max_rail_joint_motion_m`.
2. Add a feedback-step controller that adjusts support-foot X targets from
   torso target error and lowers swing height under tilt. This is still a
   scaffold controller, not a humanoid policy.
3. Gate the run with strict checker requirements: no root/body/box pose
   shortcuts, no stance/world support rewrites, randomized hidden box, support
   continuity, actual support-foot X/Z motion, active feedback steps, and low
   rail joint motion.
4. If this passes, use it as the next Isaac control interface to improve
   active probing and RL action spaces. If it fails, inspect whether the
   failure is rail dependency, feedback inactivity, support discontinuity, or
   physical instability.

Implemented:

- `build_core_world_anchored_footstep_carrier.py` now records feedback-step
  telemetry and explicit `max_rail_joint_motion_m`.
- `normalize_direct_carry_backend_summary.py` and
  `check_direct_carry_task_summary.py` pass/check the feedback and rail fields.
- `run_feedback_step_controller_carry_diag.sh` runs the first strict
  randomized-box front-mid feedback-step diagnostic.

Submitted:

`STAMP=20260705_feedback_step_controller_seed7076_frontmid srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_step_carry bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`

tmux: `curiosity_feedback_step_0705`, Slurm job `166750`, initial status
pending for priority. Output target:
`experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_controller_seed7076_frontmid/`.

Feedback-step diagnostic result:

The final successful run was retry6:

`STAMP=20260705_feedback_step_controller_seed7076_frontmid_retry6 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_step_carry bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`

tmux: `curiosity_feedback_step_retry6_0705`, Slurm job `166769`, completed on
`server02` with exit `0:0` after `00:00:37`. Check report:
`experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_controller_seed7076_frontmid_retry6/feedback_step_controller_check.json`.

Result:

- `status=pass`, completed `3580` steps.
- Hidden randomized box: `11.46294 kg`, size
  `[0.35104, 0.24014, 0.21103] m`, COM offset
  `[-0.02014, 0.03222, 0.02301] m`.
- Fall/drop events: `0 / 0`.
- Root/body/box/payload/foot/stance shortcut writes: `0`.
- Feedback controller active: `feedback_step_applied_steps=3570`,
  max X adjustment `0.008 m`, max tilt adjustment `0.005 m`.
- Final box target distance: `0.00247 m`; post-settle target distance:
  `0.00148 m`; post-settle travel loss after peak: `0.02889 m`.
- Support continuity passed under the explicit z-proxy contact threshold:
  min drive near-ground foot count `3`, drive near-ground lt2 steps `0`, min
  commanded stance near-ground foot count `2`.
- Rail motion was bounded but not zero:
  `max_rail_joint_motion_m=0.02151`, below the diagnostic threshold
  `0.025 m`.

Important limitation:

This is a better direct-Isaac execution interface, not the final research
claim. It is still a scaffolded support-foot carrier rather than a humanoid
walking controller. The support gate currently uses a foot-height proxy
(`SUPPORT_FOOT_CONTACT_Z_THRESHOLD=0.055`) rather than true contact-force
evidence. The next environment step should replace this z-proxy with actual
contact/force evidence and then carry the same gates into active probing and
RL.

2026-07-05 support-effort evidence update:

The feedback-step route now records measured support-foot joint-effort
telemetry and can gate support with effort evidence in addition to the previous
near-ground height proxy. The new evidence fields include support-foot effort
availability/read errors, per-foot max X/Z/measured effort, drive-phase
effort-supported foot count, and commanded-stance effort-supported foot count.

The first strict effort-gated check used the existing backend rollout
`20260705_feedback_step_effort_gate_seed7076_retry2`, because the Isaac rollout
completed but the shell launcher hit a post-summary EOF parse error. The
summary was normalized and checked in a separate compute allocation:

`STAMP=20260705_feedback_step_effort_gate_seed7076_retry2 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=fb_effort_chk ...`

tmux: `curiosity_feedback_step_effort_check_0705`, Slurm job `166786`,
completed on `server02` with exit `0:0`. Check report:
`experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_effort_gate_seed7076_retry2/feedback_step_effort_check.json`.

Result:

- `status=pass`, completed `3580` steps.
- Hidden randomized box: `11.46294 kg`, size
  `[0.35104, 0.24014, 0.21103] m`, COM offset
  `[-0.02014, 0.03222, 0.02301] m`.
- Fall/drop events: `0 / 0`.
- Root/box/foot/stance shortcut writes: `0`.
- Final post-settle box target distance: `0.00148 m`.
- Support-foot measured effort available, read errors `0`.
- Per-foot max measured support effort:
  `fl=3264.14`, `fr=3691.83`, `rl=4055.08`, `rr=4245.39`.
- Min drive effort-supported foot count: `4`; drive effort-supported lt2
  steps: `0`.
- Min commanded-stance effort-supported foot count: `2`;
  commanded-stance effort-supported lt2 steps: `0`.

Interpretation:

This is the strongest current direct-Isaac support evidence for the scaffold
because it no longer relies only on foot height. It is still a proxy:
`get_measured_joint_efforts()` proves actuator/load response in support-foot
DOFs, not calibrated ground reaction force or contact-state sensing. The next
direct Isaac environment step should add a true contact sensor or contact-force
gate while preserving the same carry-task summary schema.

2026-07-05 contact-report evidence update:

The support evidence was upgraded again from height + joint-effort proxies to
actual PhysX contact-state reports for the support feet. The current Core API
scene now applies `PhysxContactReportAPI` to `/World/Ground` and all four
support feet, subscribes to physics contact report events, and records
per-foot support contact state into the same normalized carry-task summary.

The first attempt, Slurm job `166793`, completed the Isaac rollout but failed
post-summary with the recurring shell EOF issue; its backend summary showed
`support_foot_contact_report_requested=false`, so the contact report flag had
not reached the core launcher. The feedback runner was then changed to pass
`--enable-support-foot-contact-report` directly through the wrapper instead of
relying only on environment propagation.

The successful retry:

`STAMP=20260705_feedback_step_contact_report_seed7076_retry2 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_contact2 bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`

tmux: `curiosity_feedback_step_contact_retry2_0705`, Slurm job `166797`,
completed on `server02` with exit `0:0` after `00:00:38`. Check report:
`experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_contact_report_seed7076_retry2/feedback_step_controller_check.json`.

Result:

- `status=pass`, completed `3580` steps.
- Hidden randomized box: `11.46294 kg`, size
  `[0.35104, 0.24014, 0.21103] m`, COM offset
  `[-0.02014, 0.03222, 0.02301] m`.
- Fall/drop events: `0 / 0`.
- Root/box/foot/stance shortcut writes: `0`.
- Final post-settle box target distance: `0.00148 m`.
- Contact report requested and available; enabled paths are `/World/Ground`
  and all four support feet.
- Contact report event count: `42`; contact report error count: `0`.
- Per-foot contact-report steps:
  `fl=3332`, `fr=3308`, `rl=3451`, `rr=3407`.
- Min contact-report foot count: `2`; contact-report lt2 steps: `0`.
- Min drive contact-report foot count: `2`; drive contact-report lt2 steps:
  `0`.
- Min commanded-stance contact-report foot count: `2`;
  commanded-stance contact-report lt2 steps: `0`.
- Support-foot effort evidence also passed: effort available, read errors `0`,
  min drive effort-supported foot count `4`.

Interpretation:

This is now stronger than the previous z-height and joint-effort support gates:
it verifies actual PhysX support-foot/ground contact-state events during the
direct Isaac scaffold rollout. It still does not provide calibrated ground
reaction forces or prove full humanoid walking. The next direct implementation
target should preserve this contact-report gate while moving from the current
scaffolded support-foot carrier toward a cleaner swappable locomotion/control
backend and active-probing task interface.

2026-07-05 all-posture contact-report gate:

The same PhysX contact-report support gate was carried from the single
feedback-step `front_mid` diagnostic into the randomized hidden-box
all-posture gate. Slurm job `166800` ran from tmux
`curiosity_all_posture_contact_0705` on `server02` and completed with exit
`0:0` after `00:01:50`.

Summary:
`experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_contact_report_64cm_seed7077/randomized_all_posture_strict_support_summary.json`.

Result:

- Overall status: `pass`.
- Shared hidden box: mass `4.86216 kg`, size
  `[0.35968, 0.24056, 0.24150] m`, COM offset
  `[-0.02053, 0.00841, 0.02503] m`.
- Postures passed: `front_mid`, `low_front`, `chest_high`.
- Each posture completed `3580` steps with fall/drop `0`.
- PhysX support-foot contact reports were available in every posture, with
  contact-report error count `0`.
- Each posture kept min drive contact-report foot count at least `2`, drive
  contact-report lt2 steps `0`, min commanded-stance contact-report foot count
  at least `2`, and commanded-stance contact-report lt2 steps `0`.

Interpretation:

This confirms the direct Isaac scaffold can run a posture-conditioned,
randomized hidden-box carry task with actual support-foot/ground contact-state
evidence across several carry postures. It should now be treated as the task
interface scaffold to build on. The next implementation step should be an
Isaac task/controller interface with explicit observation, action, reset,
reward, randomization, and active-probing hooks. External video/model code is
not on the critical path until this Isaac task can train or evaluate a
controller without relying on scaffolded support-foot motion.

2026-07-05 direct task-contract implementation:

The next code step is now explicit: treat the current direct Isaac scaffold as
a task interface, not as a solved robot. Added a
`direct_isaac_carry_task_episode_v1` contract and exporter:

- `scripts/isaac/direct_carry_task_contract.py`
- `scripts/isaac/export_direct_carry_task_episode_table.py`
- `scripts/isaac/run_export_direct_carry_task_episode_table.sh`
- `experiments/configs/direct_isaac_carry_task_contract_v1.json`

This contract separates policy observation from hidden evaluation context.
Unknown load properties such as `box_mass_kg` and `box_com_offset_m` are kept
out of policy observation and stored only under `hidden_eval_context`. This is
the intended bridge from hand-audited Isaac diagnostics to a trainable task:
observation, action, reward terms, safety gates, termination, and limitations
are now represented in one stable JSONL row format.

Submitted a compute-node export for the all-posture contact-report summary:
tmux `curiosity_export_direct_task_contract_0705`, Slurm job `166804`, stamp
`20260705_direct_carry_task_contract_all_posture_contact_7077`. This is a
short validation of the interface, not simulation or training.

Export result:

- Job `166804` completed on `server02` with exit `0:0` and produced `3` JSONL
  rows, one for each posture.
- The initial export exposed a useful interface bug: all-posture condensed
  rows did not include the full backend action/support fields.
- The exporter now follows each posture `summary_path` and loads the complete
  backend summary before writing a row.
- Retry job `166806` completed on `server02` with exit `0:0` and rewrote
  `experiments/outputs/rl_interface/20260705_direct_carry_task_contract_all_posture_contact_7077/direct_carry_task_episode_table.jsonl`.
- The corrected rows include `controller_mode`, `support_foot_mode`, PhysX
  contact gates, support effort metrics, reward terms, and hidden box
  properties only under `hidden_eval_context`.

Task runner skeleton:

Added `scripts/isaac/direct_carry_task_runner.py` with the intended backend
adapter shape: `reset`, `observe`, `apply_action`, `run_episode`,
`compute_reward`, `is_terminated`, and `export_episode_row`. This is not a
simulator and not RL. It is the direct interface that the existing scaffold,
then a real walking controller or trainable policy, should implement.

2026-07-05 executable task-runner backend:

The task interface is now connected to an executable backend instead of only
being a schema. Added:

- `scripts/isaac/direct_carry_task_shell_backend.py`
- `scripts/isaac/run_direct_carry_task_runner_episode.py`
- `scripts/isaac/run_direct_carry_task_runner_episode.sh`

The shell backend maps `DirectCarryReset` and `DirectCarryAction` into the
existing direct Isaac physical-backend launcher, runs one episode, then emits
the same contract row. This still uses the current support-foot scaffold and
must not be cited as final walking-robot success. It is the executable adapter
that lets the scaffold, and later a real walking backend, share one task
interface.

Submitted a full 3580-step validation episode:
tmux `curiosity_task_runner_episode_retry_0705`, Slurm job `166810`, stamp
`20260705_task_runner_frontmid_seed7078`, posture `front_mid`, target
`0.64 m`, randomized box seed `7078`. The first submission attempt failed
before Slurm because the outer log directory did not exist; after creating
`logs/direct_carry_task_runner`, job `166810` was submitted and initially
pending for priority.

Task-runner episode result:

- Slurm job `166810` completed on `server02` with exit `0:0` after
  `00:01:04`.
- Summary:
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_physical_backend_summary.json`.
- Episode row:
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_runner_episode.jsonl`.
- Hidden box: mass `4.33753 kg`, size
  `[0.34429, 0.21785, 0.21029] m`, COM offset
  `[0.03906, -0.03296, -0.00888] m`.
- Completed `3580` steps with fall/drop `0`, final post-settle box travel
  `0.65758 m`, final post-settle target distance `0.01758 m`, PhysX contact
  reports available, contact-report error count `0`, min drive contact-report
  foot count `2`, and commanded-stance contact-report lt2 steps `0`.

The first exported row incorrectly had `gates.passed=false` because the
backend summary has no explicit `status=pass` field. The contract now derives
`gates.passed` from strict no-fall/no-drop/no-root-shortcut/support-contact
fields when `status` is absent. A dedicated checker/export wrapper was added:
`scripts/isaac/run_check_direct_carry_task_runner_episode.sh`. Slurm job
`166817` was submitted to run the strict checker and re-export the corrected
row.

Checker/export result:

- Slurm job `166817` completed on `server02` with exit `0:0` after
  `00:00:01`.
- Strict checker report:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_frontmid_seed7078_check/direct_carry_task_runner_check.json`.
- Result status: `pass`.
- Corrected episode table:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_frontmid_seed7078_check/direct_carry_task_runner_episode_table.jsonl`.
- Corrected row has `gates.passed=true`.

Important next gap:

The validated task-runner episode still has
`probe_belief_source=no_active_probe`. That is not acceptable for the full
unknown-load objective. The next environment step is to expose active-probing
parameters in `DirectCarryAction`, pass them through the executable backend,
and validate at least one episode where probing is actually requested and
reported before carrying.

2026-07-05 active-probing task-runner update:

Added `probe_steps` to `DirectCarryAction`, passed it through
`direct_carry_task_shell_backend.py` as `PROBE_STEPS`, exposed it in
`run_direct_carry_task_runner_episode.py/.sh`, and added probe action fields
to the contract/export schema. Lightweight checks passed:
`python3 -m py_compile scripts/isaac/direct_carry_task_runner.py scripts/isaac/direct_carry_task_shell_backend.py scripts/isaac/direct_carry_task_contract.py scripts/isaac/run_direct_carry_task_runner_episode.py scripts/isaac/export_direct_carry_task_episode_table.py`
and
`bash -n scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.

Submitted an active-probing validation episode:
tmux `curiosity_task_runner_probe_0705`, Slurm job `166819`, stamp
`20260705_task_runner_probe_frontmid_seed7079`, `PROBE_STEPS=80`,
`PROBE_AMPLITUDE_X=0.012`, `PROBE_AMPLITUDE_Z=0.0`, target `0.64 m`, total
steps `3660`. This remains a scaffold diagnostic; the expected evidence is
that probing is actually requested/reported before carrying.

Active-probing validation result:

- Slurm job `166819` completed on `server02` with exit `0:0` after
  `00:00:45`.
- Hidden randomized box: mass `11.13313 kg`, size
  `[0.31808, 0.25514, 0.23031] m`, COM offset
  `[-0.01361, -0.02603, 0.01952] m`.
- The episode requested `80` horizontal push-pull probe steps with
  `PROBE_AMPLITUDE_X=0.012`.
- Probe evidence: `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`,
  `probe_belief_source=heuristic_from_probe_telemetry_not_calibrated_mass_estimator`,
  max probe box travel `0.03064 m`, max probe relative error `0.00869 m`,
  max probe support-foot X effort `525.43`, max probe support-foot Z effort
  `2053.72`.
- Carry evidence after probing: completed `3660` steps, fall/drop `0`, final
  post-settle box travel `0.66478 m`, final post-settle target distance
  `0.02478 m`, contact-report available, contact-report error count `0`.
- Runner report status: `pass`.

Probe-specific checker gate:

Added checker flags `--min-probe-steps`, `--require-probe-belief`,
`--forbid-probe-hidden-ground-truth`, and `--min-probe-box-travel-x`. The
wrapper now enables these when `REQUIRE_PROBE_BELIEF=1`. Slurm job `166821`
ran the probe-specific gate with `MIN_PROBE_STEPS=80` and
`MIN_PROBE_BOX_TRAVEL_X=0.01`; it completed on `server02` with exit `0:0`
after `00:00:01`. Report:
`experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_probe_frontmid_seed7079_probegate/direct_carry_task_runner_check.json`;
status `pass`. Corrected row:
`experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_probe_frontmid_seed7079_probegate/direct_carry_task_runner_episode_table.jsonl`;
`gates.passed=true`.

Interpretation:

This is the first executable task-runner episode in this branch that performs
active probing before carrying and records a non-hidden-ground-truth load
belief. It is still a scaffolded support-foot carrier, not a full walking
robot. The next direct step is to run this active-probing gate across multiple
postures, then preserve the same contract while replacing the scaffold backend
with a less artificial walking controller.

2026-07-05 multi-posture active-probe task-runner gate:

Added:

- `scripts/isaac/run_task_runner_active_probe_postures.sh`
- `scripts/isaac/summarize_task_runner_active_probe_postures.py`

The new sweep runs `front_mid`, `low_front`, and `chest_high` with one hidden
box seed. Each posture must request active probing, produce a probe belief
without hidden ground truth, pass support-foot contact/effort gates, and carry
the box to the 64 cm target under the existing direct task-runner contract.

Lightweight checks passed:
`python3 -m py_compile scripts/isaac/summarize_task_runner_active_probe_postures.py scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/direct_carry_task_contract.py`
and
`bash -n scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.

Submitted compute validation:
tmux `curiosity_active_probe_postures_0705`, Slurm job `166822`, stamp
`20260705_task_runner_active_probe_postures_seed7080`, box seed `7080`,
`PROBE_STEPS=80`, `PROBE_AMPLITUDE_X=0.012`, target `0.64 m`, total steps
`3660`. Initial status: pending for priority.

2026-07-05 stance-foot world-lock diagnostic:

To keep moving in Isaac instead of waiting on external models, I tested a
direct support-mechanics replacement inside the existing task-runner contract.
The implementation added runtime-enabled fixed joints from commanded stance
feet to world, but kept it explicitly labeled as a diagnostic fixed-world
support mode. The checker defaults still forbid this mode; explicit audits
must set `REQUIRE_STANCE_FOOT_WORLD_LOCK=1`.

Valid run:

- Slurm job `166875`, stamp
  `20260705_task_runner_stance_world_lock_slip_seed7088_server36`, fixed node
  `server36`.
- Command parameters: `BOX_SEED=7088`, `POSTURES=front_mid`,
  `TARGET_X=0.64`, `STEPS=3660`,
  `SUPPORT_MODE=alternating_placement_feet`,
  `STANCE_FOOT_WORLD_LOCK=1`, `REQUIRE_STANCE_FOOT_WORLD_LOCK=1`,
  `PROBE_STEPS=80`, `PROBE_AMPLITUDE_X=0.012`,
  `FEEDBACK_STEP_X_GAIN=0.03`, `FEEDBACK_STEP_X_LIMIT=0.012`,
  `FEEDBACK_STEP_TILT_GAIN=0.05`, `FEEDBACK_STEP_TILT_LIMIT=0.006`,
  `MAX_NEAR_GROUND_FOOT_SPEED=0.8`, and
  `MAX_NEAR_GROUND_FOOT_SLIP=0.2`.
- Summary:
  `experiments/outputs/direct_carry_task_runner_active_probe_postures/20260705_task_runner_stance_world_lock_slip_seed7088_server36/active_probe_posture_summary.json`.

Result and interpretation:

The rollout completed the carry but failed the stricter support audit. It had
fall/drop `0`, final post-settle box travel `0.64560 m`, final target distance
`0.00560 m`, active probe belief without hidden ground truth, and
`stance_foot_world_lock_enabled=true` with `4` lock joints, `81` lock-switch
events, and `324` lock pose updates. However, the strict checker failed:
actual support-foot lift was `0.01943 m` below the `0.03 m` gate, max
near-ground foot speed was `0.91486 m/s` above the `0.8` gate, and max
near-ground foot slip was `0.73106 m` above the `0.2` gate. PhysX also emitted
repeated warnings that the stance world-lock joints had disjointed body
transforms and would likely snap bodies together.

Conclusion: simple runtime world-locking is not the right support mechanism.
It proves that the task-runner/checker can expose hidden fixed-world support
and reject it, but it does not solve planted-foot sliding. The next
implementation should separate stance and swing targets so stance feet are
not driven against their locks, and should treat each legitimate lift/replant
transition as the only time a slip reference can reset.

2026-07-05 freeze-locked stance target follow-up:

Implemented the first follow-up to the failed world-lock diagnostic:
`--freeze-locked-stance-foot-targets`. When this mode is used with
`--stance-foot-world-lock`, the controller records the measured X/Z joint
positions at the moment a foot becomes locked and keeps those targets frozen
while that foot remains in stance. The intent is to test whether removing the
drive-versus-lock conflict reduces snap warnings and near-ground slip. This is
still a diagnostic fixed-world support mode, not a final walking controller.

Wiring added:

- `FREEZE_LOCKED_STANCE_FOOT_TARGETS` through the core, physical backend,
  task-runner shell backend, direct episode runner, posture sweep, checker,
  normalizer, and summarizer.
- Checker gate `REQUIRE_FREEZE_LOCKED_STANCE_FOOT_TARGETS=1`.
- Debug flag `DEBUG_CORE_CMD=1` in the core launcher to print exact argv if
  argparse fails before rollout.

Validation status:

- Lightweight checks passed with `python3 -m py_compile` and `bash -n`.
- Slurm job `166884` failed before Isaac rollout with no backend summary.
- Slurm job `166885` failed before rollout with argparse ambiguity `--tray-`.
- Debug validation job `166888`, stamp
  `20260705_task_runner_freeze_locked_stance_seed7089_debug_s10`, ran on
  `server10`, reached Isaac rollout, normalized summary, and strict checker.

Result:

The freeze-locked target mode did exactly what the diagnostic was meant to
test, and the result is negative for the final task. It reduced support-foot
sliding sharply: max near-ground slip was `0.00320 m`, and max near-ground foot
speed was `0.27180 m/s`. The freeze fields were present:
`freeze_locked_stance_foot_targets_enabled=true`,
`freeze_locked_stance_foot_target_count=8`,
`stance_foot_world_lock_enabled=true`, `4` lock joints, and `81` lock-switch
events.

However, the robot did not carry the box to the target. It had
`fall_events=1853`, final post-settle box travel `-0.14893 m`, final target
distance `0.78893 m`, and max target-directed post-settle travel only
`0.00327 m`. PhysX still emitted repeated disjoint world-lock joint snap
warnings.

Interpretation:

The old scaffold's forward motion depends on moving or driving near-ground
stance feet. Freezing those targets removes the visible foot-slip failure but
also removes useful propulsion and destabilizes the carrier. The next support
mechanic cannot be another stance-foot drag variant. It must create propulsion
by moving the body or anchor relative to truly planted contacts, with slip
references reset only on verified swing lift/replant transitions.

2026-07-05 planted-stance rail-propulsion diagnostic:

Purpose:

Test the direct idea raised after the freeze-locked failure: keep locked stance
feet planted with low slip, but let the torso rail continue moving during
stance so propulsion comes from body/anchor motion relative to planted contacts
instead of from dragging stance feet.

Implementation:

- Added `--planted-stance-rail-propulsion` in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`.
- Threaded `PLANTED_STANCE_RAIL_PROPULSION` through the core launcher,
  physical backend, task-runner shell backend, direct episode runner, posture
  sweep, normalizer, checker, and summarizer.
- Added checker gate `REQUIRE_PLANTED_STANCE_RAIL_PROPULSION=1`, which
  requires `planted_stance_rail_propulsion_enabled=true` and
  `planted_stance_rail_propulsion_steps > 0`.
- Lightweight checks passed with `python3 -m py_compile` and `bash -n`.

Invalid first attempt:

- Slurm job `166894`, stamp
  `20260705_task_runner_planted_rail_propulsion_seed7090`, ran on `server44`.
- It is not valid physical evidence because the trigger had been placed in the
  wrong code branch. The checker reported
  `planted_stance_rail_propulsion_enabled=true` but
  `planted_stance_rail_propulsion_steps=0`.

Valid compute diagnostic:

- Slurm job `166895`, stamp
  `20260705_task_runner_planted_rail_propulsion_seed7091_fixedgate`, ran on
  `server53`.
- Parameters: `BOX_SEED=7091`, `POSTURES=front_mid`, `TARGET_X=0.64`,
  `STEPS=3660`, `SUPPORT_MODE=alternating_placement_feet`,
  `STANCE_FOOT_WORLD_LOCK=1`, `FREEZE_LOCKED_STANCE_FOOT_TARGETS=1`,
  `PLANTED_STANCE_RAIL_PROPULSION=1`,
  `REQUIRE_STANCE_FOOT_WORLD_LOCK=1`,
  `REQUIRE_FREEZE_LOCKED_STANCE_FOOT_TARGETS=1`,
  `REQUIRE_PLANTED_STANCE_RAIL_PROPULSION=1`,
  `MAX_NEAR_GROUND_FOOT_SPEED=0.8`, and
  `MAX_NEAR_GROUND_FOOT_SLIP=0.2`.
- Output summary:
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_planted_rail_propulsion_seed7091_fixedgate_front_mid/direct_carry_task_physical_backend_summary.json`.
- Checker:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_planted_rail_propulsion_seed7091_fixedgate_front_mid_probecheck/direct_carry_task_runner_check.json`.

Result:

The mode triggered correctly: `planted_stance_rail_propulsion_steps=3570`.
Near-ground support-foot slip remained low:
`max_near_ground_foot_slip_m=0.00320` and
`max_near_ground_foot_speed_mps=0.27191`. But the carrying task failed
decisively. The run had `fall_events=1902`, actual support-foot lift only
`0.00428 m`, max target-directed post-settle box travel only `0.00360 m`,
final post-settle box travel `-0.13414 m`, and final target distance
`0.77414 m`. PhysX still reported repeated disjoint stance-world-lock joint
snapping warnings.

Interpretation:

This closes the current world-lock branch as a useful direction. It can make
the slip audit look good, but the resulting support system does not provide
valid forward carrying mechanics and still relies on invalid fixed-world
constraints. Do not keep adding more variants on world-lock plus frozen
stance. The next implementation path should be a contact-consistent support
model: either a true planted-foot/contact controller in Isaac or a
controller-backed robot whose stance contacts can pass the same slip,
contact, fall/drop, and target-travel gates without fixed-world snapping.

2026-07-05 no-world-lock commanded-stance freeze diagnostic:

Purpose:

Remove the fixed-world lock entirely and test whether the existing
support-foot scaffold can move through contact/friction alone. The diagnostic
freezes commanded stance-foot X/Z targets at measured joint positions, keeps
planted-stance rail propulsion enabled, and leaves the checker default
`--forbid-fixed-world-support` active.

Implementation:

- Added `--freeze-commanded-stance-foot-targets` in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`.
- Threaded `FREEZE_COMMANDED_STANCE_FOOT_TARGETS` through the core launcher,
  physical backend, task-runner shell backend, direct episode runner, posture
  sweep, normalizer, checker, and summarizer.
- Added checker gate `REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`.
- Lightweight checks passed with `python3 -m py_compile` and `bash -n`.

Compute diagnostic:

- Slurm job `166899`, stamp
  `20260705_task_runner_no_worldlock_contact_propulsion_seed7092`, ran on
  `server44`.
- Parameters: `BOX_SEED=7092`, `POSTURES=front_mid`, `TARGET_X=0.64`,
  `STEPS=3660`, `SUPPORT_MODE=alternating_placement_feet`,
  `STANCE_FOOT_WORLD_LOCK=0`, `FREEZE_LOCKED_STANCE_FOOT_TARGETS=0`,
  `FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`,
  `PLANTED_STANCE_RAIL_PROPULSION=1`,
  `REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`,
  `REQUIRE_PLANTED_STANCE_RAIL_PROPULSION=1`,
  `MAX_NEAR_GROUND_FOOT_SPEED=0.8`, and
  `MAX_NEAR_GROUND_FOOT_SLIP=0.2`.
- Output summary:
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_no_worldlock_contact_propulsion_seed7092_front_mid/direct_carry_task_physical_backend_summary.json`.
- Checker:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_no_worldlock_contact_propulsion_seed7092_front_mid_probecheck/direct_carry_task_runner_check.json`.

Result:

The diagnostic triggered correctly without fixed-world support:
`stance_foot_world_lock_enabled=false`,
`stance_foot_world_lock_joint_count=0`,
`freeze_commanded_stance_foot_target_count=8`,
`freeze_commanded_stance_foot_target_switch_count=81`, and
`planted_stance_rail_propulsion_steps=3570`. It also kept the slip audit low:
max near-ground slip `0.04693 m`, max near-ground foot speed `0.30856 m/s`,
and max actual support-foot lift `0.11852 m`.

The task still failed. The strict checker reported broken contact support:
`min_drive_contact_report_foot_count=0`,
`drive_contact_report_lt2_steps=76`,
`min_commanded_stance_contact_report_foot_count=0`,
`commanded_stance_contact_report_lt2_steps=1580`, and
`commanded_stance_near_ground_lt2_steps=813`. The rollout had
`fall_events=444`, final post-settle box travel `-0.21011 m`, final target
distance `0.85011 m`, and max target-directed post-settle travel only
`0.00355 m`.

Interpretation:

This is a cleaner negative result than the world-lock runs: it removes
fixed-world support and still cannot produce valid forward carrying. The
current prismatic support-foot scaffold can either drag stance feet to move
the box, or freeze stance feet and lose contact/propulsion. It should no
longer be the main implementation target. The next Isaac step should use a
controller-backed robot or a substantially different contact model that can
generate forward impulse while preserving stance contact and passing the same
checker gates.
## 2026-07-05 Direct G1 Core API Route

User correction: do not wait on external model downloads or controller
checkpoints when they are not immediately useful. The active implementation
route is direct Isaac scene construction around
`scripts/isaac/build_core_world_g1_box_scene.py`.

G1 now loads through Core API as a `43` joint articulation and initializes near
standing height. The blocker is stand/balance control, not missing data.

Negative stand findings so far:

- nominal stand `diag5`/`diag6`: slow forward pitch, `fall_events=5`,
  `max_tilt_rad=0.90630`, `min_robot_z_m=0.55813`;
- balance feedback pitch sign `-1`: worse, `fall_events=51`,
  `max_tilt_rad=2.23711`;
- balance feedback pitch sign `+1`: better but still fail,
  `fall_events=33`, `max_tilt_rad=1.31126`;
- mid/deep crouch: failed backward with `fall_events=153` and `247`.

Implementation correction:

- added `--disable-carry-box-spawn` / `SPAWN_CARRY_BOX=0` for pure G1 stand
  diagnostics;
- added `carry_box_spawned` to summaries and checker support via
  `--expect-carry-box-spawned true|false`;
- fixed setup root orientation so `--g1-root-orientation-wxyz` is honored.

Active compute batch:

- tmux `curiosity_g1_stand_nobox_tune_0705`, Slurm job `166915`;
- command uses `srun --partition=gpu --gres=gpu:1 --time=02:00:00
  --job-name=g1_stand_nb_tune`;
- stamps: `20260705_core_world_g1_nobox_gain2_diag11`,
  `20260705_core_world_g1_nobox_gain3_diag12`,
  `20260705_core_world_g1_nobox_mildcrouch_diag13`,
  `20260705_core_world_g1_nobox_feedback_low_diag14`.

Next gate: pass no-box G1 stand without rollout root pose or velocity writes.
Only then test `ATTACH_BOX=fixed_torso` with a small payload, and only after
fixed-payload balance succeeds reconnect the physical free box.

Follow-up result:

Slurm job `166915` ran the no-box batch on `server53`. All four stand
diagnostics failed with `carry_box_spawned=false` and `box_drop_events=0`.
This cleanly rules out the previously falling free box as the G1 stand failure
source. Metrics: `diag11` gain scale 2 had `fall_events=94`,
`max_tilt_rad=3.11633`; `diag12` gain scale 3 had `fall_events=96`,
`max_tilt_rad=3.03615`; `diag13` mild crouch had `fall_events=210`,
`max_tilt_rad=1.46066`; `diag14` low pitch feedback had `fall_events=91`,
`max_tilt_rad=3.13663`.

The next batch aligns the Core API scene more closely with IsaacLab
`G1_29DOF_CFG`: root pos `z=0.75`, rot `(0, 0, 0.7071, 0.7071)`,
IsaacLab-style drive gains, and optional disabling of design-time pelvis
xform. Submitted tmux `curiosity_g1_isaaclab_pose_tune_0705`, Slurm job
`166916`, stamps `diag15`-`diag17`.

Correction: Slurm job `166916` failed before Isaac due to a launcher
intermediate-state shell parse error and produced no summaries. It is invalid
evidence. The Core API scene now writes setup-only joint positions and
velocities before rollout, recording `joint_state_write_count_setup`; this is
intended to match IsaacLab initial state semantics without adding rollout
shortcuts. Retry submitted as tmux `curiosity_g1_isaaclab_pose_retry2_0705`,
Slurm job `166918`, stamps `diag15_retry2`-`diag17_retry2`.

## 2026-07-05 Direct Isaac G1 Progress After User Correction

Do not wait on external model downloads for this phase. The active route is
to build the carrying scene directly in Isaac and keep every result labeled by
what it actually proves.

Current G1 Core API evidence:

- `diag22` established the first stable no-box G1 stand gate: 360/360 steps,
  fall 0, max tilt `0.00882 rad`, using hip pitch `-0.12`, knee `0.30`, ankle
  pitch `-0.15`, root z `0.78`, arena drive gains, and setup joint-state
  write.
- `diag25`/`diag26`/`diag27` established fixed-torso ballast standing for
  0.5/1/2 kg with collision disabled, fall/drop 0.
- `diag28`/`diag29` established stable open-loop marching with no box and
  with 1 kg fixed payload, but travel was only centimeter scale.
- `diag30`/`diag31`/`diag32` kept stability for larger march amplitudes and
  1/2 kg fixed payloads, but travel still saturated near `0.019`-`0.028 m`.
- `diag33` added collision-enabled 1 kg fixed-torso front payload standing:
  360/360 steps, fall/drop 0, `max_tilt_rad=0.02775`,
  `max_robot_travel_xy_m=0.01859`, `max_box_travel_xy_m=0.02141`,
  rollout root/velocity/box pose writes all 0.
- `diag34` added collision-enabled 1 kg fixed-torso front payload marching:
  600/600 steps, fall/drop 0, `max_tilt_rad=0.02533`,
  `max_robot_travel_xy_m=0.02472`, `max_box_travel_xy_m=0.02236`,
  rollout root/velocity/box pose writes all 0.

Interpretation:

The G1 scene is no longer blocked at asset loading or basic balance. It has a
stable diagnostic posture and can tolerate a small front payload. The main
blocker is now free dynamic box contact: the box must stop being welded to the
torso while still remaining physically supported and measured for drop,
relative slip, and robot stability.

Implementation update:

`build_core_world_g1_box_scene.py` now has a `front_tray` torso-cradle
scaffold. The cradle is made of physical collision bodies fixed to the G1
torso: deck, side rails, front stop, and rear stop. The carry box remains a
free dynamic rigid body. The summary/checker now records `torso_cradle`,
`cradle_piece_count`, `require_box_no_drop`, and
`max_box_robot_relative_offset_error_m`.

Active compute batch:

- tmux `curiosity_g1_free_cradle_0705`, Slurm job-name `g1_free_cradle`;
- stamps `20260705_core_world_g1_free_cradle_stand_1kg_diag35` and
  `20260705_core_world_g1_free_cradle_march_1kg_diag36`;
- purpose: first free dynamic box contact-scaffold check for stand and small
  marching;
- this is not a carrying success claim even if it passes, because there is no
  real walking controller and the tray is a scaffold.

Result:

`diag35` and `diag36` are negative. Both used `attach_box=none`,
`torso_cradle=front_tray`, `cradle_piece_count=5`, `require_box_no_drop=true`,
and rollout root/velocity/box pose writes 0. The stand run failed with
`fall_events=351`, `box_drop_events=331`, `max_tilt_rad=3.14090`,
`min_robot_z_m=0.13680`, and
`max_box_robot_relative_offset_error_m=0.59789`. The marching run failed with
`fall_events=591`, `box_drop_events=571`, `max_tilt_rad=3.13946`,
`min_robot_z_m=0.16728`, and
`max_box_robot_relative_offset_error_m=0.59809`.

Interpretation:

The first free-box torso-cradle is too aggressive or geometrically wrong. It
destabilizes the robot almost immediately, so it is not a valid free-box
contact solution. The next action is isolation, not model waiting:

- `diag37`: same torso cradle, no carry box spawned;
- `diag38`: same free box pose, no torso cradle.

If `diag37` fails, reduce cradle mass/size, move it closer to the torso,
raise it away from the legs, or switch to a non-contact visual/kinematic
cradle diagnostic before reintroducing the free box. If `diag37` passes but
`diag38` fails, the free-box initial position is colliding with G1 and must be
offset. If both pass separately, the failure is combined contact impulse and
the next gate should ramp contact by lowering the box onto the tray or using a
soft/compliant stop rather than instant interpenetrating contact.

Isolation result:

`diag37` failed even with no carry box spawned: cradle-only had
`fall_events=171`, `max_tilt_rad=3.12544`, and `min_robot_z_m=0.19309`.
`diag38` kept G1 stable with no cradle: `fall_events=0`,
`max_tilt_rad=0.02695`, and `min_robot_z_m=0.78403`, while the unsupported
free box naturally dropped. The failure is therefore the torso-cradle
geometry/collision itself, not the free box alone.

Implementation update:

The G1 Core API scene now exposes `--disable-cradle-collision` and
`--cradle-mass-scale`, with launcher envs `CRADLE_COLLISION_ENABLED=0` and
`CRADLE_MASS_SCALE`. The next tuning batch is:

- `diag39`: same cradle-only geometry with cradle collision disabled, to
  isolate inertial/fixed-joint effects from collision effects;
- `diag40`: smaller, lighter, more forward and higher collision-enabled
  cradle-only geometry.

Retry2 result:

The first tuning batch `166942` is invalid because the compute node read a
stale/intermediate launcher and failed `bash -n` before Isaac. Retry2 printed
the launcher lines on the compute node and then ran normally.

`diag39_retry2` passed with the original cradle geometry when cradle collision
was disabled: 180/180 steps, fall 0, `max_tilt_rad=0.04391`, and
`min_robot_z_m=0.78342`. `diag40_retry2` passed with collision enabled after
shrinking, raising, front-shifting, and lightening the cradle: 180/180 steps,
fall 0, `max_tilt_rad=0.01649`, `min_robot_z_m=0.78422`, and max robot drift
`0.01040 m`.

Interpretation:

The G1 can tolerate a torso-fixed cradle if the collision geometry is not
intersecting or over-constraining the robot. The current usable contact
baseline is the `diag40_retry2` geometry: deck size `0.24 x 0.26 x 0.025 m`,
deck local position `(0.44, 0.0, 0.10)`, rail height `0.07 m`, stop height
`0.08 m`, rail thickness `0.018 m`, and cradle mass scale `0.15`.

Next gate:

Put a very small free dynamic box on this stable small cradle and first pass a
standing no-drop gate before trying marching. Active stamps: `diag41` with box
z `1.00 m` and `diag42` with box z `0.95 m`, both 0.25 kg.

Standing free-box result:

`diag41` and `diag42` both passed the small free-box-on-cradle stand gate. Both
used `attach_box=none`, `torso_cradle=front_tray`, cradle collision enabled,
0.25 kg free dynamic box, 43 G1 joints, and rollout root/velocity/box pose
writes 0. `diag42` is the cleaner baseline: 240/240 steps, fall/drop 0,
`max_tilt_rad=0.02406`, `min_robot_z_m=0.78411`,
`min_box_z_m=0.95931`, `max_box_robot_relative_offset_error_m=0.03337`, and
final relative-offset error `0.01054`.

Next gate:

Use the exact `diag42` geometry for a small open-loop marching disturbance
before attempting heavier boxes or longer movement. Active stamp:
`20260705_core_world_g1_small_cradle_freebox_march_diag43`.

Marching free-box result:

`diag43` passed the first small free-box-on-cradle marching gate: 420/420
steps, `attach_box=none`, free dynamic box spawned, torso cradle collision
enabled, 43 G1 joints, rollout root/velocity/box pose writes 0, fall/drop 0,
`max_tilt_rad=0.02290`, `min_robot_z_m=0.78355`, `min_box_z_m=0.95979`,
`max_box_robot_relative_offset_error_m=0.03481`, and final relative-offset
error `0.02609`.

This is real Isaac contact progress, but still not carrying. The robot is
only open-loop marching with centimeter-scale motion:
`max_robot_travel_xy_m=0.01923` and `max_box_travel_xy_m=0.03443`. The next
valid gates should increase one difficulty axis at a time: longer duration,
heavier free box, larger free box, or larger gait amplitude. Do not combine
those changes before the individual gates pass.

One-axis next-gate result:

`curiosity_g1_small_cradle_nextgates_0705`, Slurm job `166948`, ran three
independent follow-ups from `diag43`. All passed the checker with
`attach_box=none`, free dynamic box, `torso_cradle=front_tray`, cradle
collision enabled, 43 G1 joints, rollout root/velocity/box pose writes 0, and
fall/drop 0:

- `diag44`: 1200-step duration, 0.25 kg box, gait amplitude 0.05,
  `max_tilt_rad=0.02290`, `min_box_z_m=0.95979`,
  `max_box_robot_relative_offset_error_m=0.03481`.
- `diag45`: 420 steps, 0.5 kg box, gait amplitude 0.05,
  `max_tilt_rad=0.03175`, `min_box_z_m=0.95530`,
  `max_box_robot_relative_offset_error_m=0.03766`.
- `diag46`: 420 steps, 0.25 kg box, gait amplitude 0.08,
  `max_tilt_rad=0.02582`, `min_box_z_m=0.95846`,
  `max_box_robot_relative_offset_error_m=0.05755`.

Interpretation:

For the small-box scaffold, contact stability is no longer the immediate
blocker. The remaining blocker is meaningful locomotion: even passing runs
still show only centimeter-scale robot/box motion. The next gates should
increase gait amplitude to probe whether the open-loop G1 controller can
produce real forward travel, while keeping the same no-shortcut summary gates.

Gait-amplitude probe:

`curiosity_g1_amp_probe_0705`, Slurm job `166949`, tested gait amplitude
`0.12` and `0.16` with the same 0.25 kg free dynamic box. Both passed
fall/drop/no-shortcut gates. `diag47` reached
`max_robot_travel_xy_m=0.03872`, `max_box_travel_xy_m=0.06618`,
`max_tilt_rad=0.02881`, and max box-robot relative-offset error `0.10253`.
`diag48` reached `max_robot_travel_xy_m=0.05303`,
`max_box_travel_xy_m=0.10716`, `max_tilt_rad=0.04088`, and max relative-offset
error `0.15246`.

However, CSV inspection shows final forward displacement remains small; max
travel includes oscillatory lateral motion. The scene summary and checker now
include final and target-directed travel fields, and the next reruns
`diag49`/`diag50` measure whether the policy is actually moving the robot and
free box toward the target rather than only swinging in place.

Target-directed rerun result:

`diag49` and `diag50` are negative but important. The new metrics show that
large-amplitude open-loop gait can produce real target-directed motion, but it
does not yet preserve box retention or balance:

- `diag49`, amplitude `0.16`: `max_box_target_directed_travel_m=0.65748`,
  final box target-directed travel `0.65724`, but `fall_events=42`,
  `box_drop_events=37`, `max_tilt_rad=0.93175`, min box z `0.15241`, and max
  box-robot relative-offset error `0.55755`.
- `diag50`, amplitude `0.20`: final box target-directed travel `0.70231`, but
  `fall_events=41`, `box_drop_events=38`, `max_tilt_rad=1.29701`, min box z
  `0.10822`, and max relative-offset error `0.57376`.

Interpretation:

The blocker has shifted again. The scene no longer merely jitters in place:
the open-loop G1 can drive the robot and box forward by more than half a
meter. The failure is late-stage stability and box retention. The next
engineering gate is a stronger but still stable torso cradle: longer deck and
higher stops/rails, validated cradle-only first, then free-box gait.

Stronger-cradle result:

`diag51` validated the stronger cradle by itself: 240/240 steps, fall 0,
`max_tilt_rad=0.01893`, and rollout shortcut writes 0. `diag52` then used that
stronger cradle with a free 0.25 kg box and gait amplitude `0.16`; it kept the
box and robot stable with fall/drop 0, `min_box_z_m=0.96531`, and max
relative-offset error `0.14996`. But it did not carry: max box target-directed
travel was only `0.02283 m`.

Interpretation:

The stronger cradle solved retention by suppressing the dynamics that produced
motion. The valid next search space is between `diag40` and `diag51`: enough
rail/stop geometry to keep the box high, but not so restrictive that the
open-loop gait loses all forward target-directed motion. The checker should
gate both retention (`min_box_z`) and target-directed travel.

Middle-cradle and feedback result:

`diag53`/`diag54` tested a middle cradle between the minimal and stronger
geometries. It retained the free box and robot with fall/drop 0, but still
suppressed useful target-directed motion: `diag54` reached only
`max_box_target_directed_travel_m=0.03777`. Pitch-feedback stabilization
`diag55`/`diag56` had the same qualitative outcome: stable, but motion
collapsed to about `0.015` to `0.016 m` max box target-directed travel.

Gait-stop result:

The first valid gait-stop retry `diag57_retry2`/`diag58_retry2` stopped too
early. Both held the free box without fall/drop or rollout root/box writes,
but final target-directed travel remained near zero (`-0.03321 m` and
`0.00253 m` for the box). These are useful hold diagnostics, not carrying.

Current next gate:

Use the minimal cradle that produced real forward motion in `diag49`, but stop
later, inside the pre-fall window. In `diag49`, fall first appears at step 380
and drop at step 390; before that, step 320-370 shows substantial
target-directed motion without counted fall/drop. Pending batch
`curiosity_g1_late_stop_0705` / Slurm job `166968` runs stop steps 320, 340,
360, and 370 (`diag59`-`diag62`) and requires post-stop hold with fall/drop 0
and rollout root/box writes 0. A pass would still be a diagnostic short-distance
carrying window, not a full learned carrying policy.

Late-stop conclusion:

The late-stop idea is now a negative result, not a path to keep sweeping.
`cradle_mass_scale=0.15` was stable but only moved the free box about
`0.035 m` target-directed at best. Matching `diag49` with
`cradle_mass_scale=1.0` restored large target-directed motion, but stop steps
320/340/360/370 all failed after the stop with fall/drop. The best target
motion in that batch was `0.82349 m`, but it had 49 fall events and 36 box-drop
events. Therefore the open-loop controller enters an unrecoverable forward
pitch state before the counted fall/drop threshold. A pure stop schedule is not
a valid recovery mechanism.

Next control direction:

The next implementation should intervene before the runaway lean, not after
it. Two concrete options are valid:

- Add a pitch/forward-velocity feedback controller that reduces gait amplitude,
  increases ankle/hip recovery, and/or widens the contact posture once pitch or
  pitch rate crosses a low threshold, then gate final target-directed box
  travel plus fall/drop 0.
- Replace the large-amplitude gait with a lower-energy forward gait/contact
  schedule that produces target-directed motion without depending on the heavy
  cradle tipping the robot forward.

Do not count additional timing-only gait-stop sweeps as meaningful progress
unless a new recovery controller is added.

Threshold-feedback conclusion:

A first threshold-feedback controller was implemented and tested. The direct
retry2 batch bypassed the stale shell launcher by calling the Python scene
builder directly. The thresholds were real: feedback first activated at step
266 for pitch threshold `0.20 rad` and step 296 for `0.30 rad`. However, all
four diagnostics still failed fall/drop gates. The weaker feedback variants
kept about `0.64-0.67 m` max box target-directed travel but still had 52 fall
events and 36 drop events. The stronger variants reached up to `0.783 m`
target-directed box travel but also failed with 51-55 fall events and 38 drop
events.

Interpretation:

The heavy cradle plus large-amplitude gait creates forward motion by entering a
pitch runaway regime. Once pitch reaches `0.20-0.30 rad`, simple ankle/hip pitch
feedback is not enough to recover. This is not a timing problem anymore; it is
a gait/contact design problem.

Next concrete implementation:

Build a lower-energy forward gait/contact schedule instead of relying on the
heavy cradle to tip the robot forward. The next gate should combine:

- a short acceleration window,
- amplitude reduction before pitch reaches `0.20 rad`,
- explicit recovery posture targets for hip/knee/ankle/torso,
- and the same no-shortcut gates: rollout root/box writes 0, fall/drop 0, and
  at least `0.10 m` final box target-directed travel.

Staged-gait short-distance result:

The staged gait implementation produced the first clean short-distance G1
free-box carrying diagnostics. In the refinement batch, all four variants
passed the 420-step checker with fall/drop 0 and rollout root/box writes 0.
The best safer variants were:

- `diag72`: `cradle_mass_scale=0.90`, gait amp `0.11`, ramp `100-210`, final
  box target-directed travel `0.49137 m`, max tilt `0.60004`.
- `diag73`: `cradle_mass_scale=0.95`, gait amp `0.10`, ramp `90-190`, final
  box target-directed travel `0.51315 m`, max tilt `0.62928`.

`diag74` reached more travel (`0.63546 m`) but max tilt was `0.84893`, almost
on the `0.85` fall gate, so it is too close to the boundary for a robust
claim. The next gate is no longer just finding any forward motion; it is
testing whether `diag72`/`diag73` can hold balance and retain the free box in a
longer rollout after the initial carry window.

Long-validation result:

The 700-step validation of `diag72`/`diag73` failed. Both variants stayed valid
through the original short window, but entered the same forward pitch failure
around step 450 and then dropped the box. This means the current staged gait is
not yet a durable carry controller; it is a short-distance diagnostic. The next
search should move to the conservative side of the boundary, especially
`cradle_mass_scale=0.80-0.85` and amp `0.10-0.12`, and require a 700-step pass
before treating the behavior as more than short-window evidence.

Conservative long-validation result:

The conservative 700-step batch `diag77`-`diag80` also failed. Lowering
`cradle_mass_scale` to `0.80-0.85` and gait amplitude to `0.10-0.12` only
delayed the same pitch/drop failure. `diag79` delayed failure the most, but
still had `fall_events=148`, `box_drop_events=131`, and final box
target-directed travel `0.75886 m`. This rules out simply moving to a slightly
more conservative staged gait as the long-duration solution.

Terminal-hold result:

Added terminal hold triggers by step, box target-directed travel, robot
target-directed travel, pitch, and pitch rate. Summaries record active steps,
first active step, and first trigger reason. Late triggers `diag81`-`diag84`
all failed after first activation around steps `398`-`403`. Earlier triggers
`diag85`-`diag88` also failed after first activation around steps `320`-`342`.
The terminal posture stayed active for hundreds of steps, but could not arrest
forward pitch or prevent box drop. Static terminal posture targets are not
enough.

Drive-authority result:

Added `stand_force_scale` so max-force scaling is separated from
stiffness/damping scaling. All-rollout high authority was mixed but did not
solve carrying: force scale `2` and `4` still failed; `stand_gain_scale=2` and
`stand_force_scale=2` was stable for 700 steps but final box target-directed
travel collapsed to `0.03484 m`, so it is a stable hold/motion-suppression
diagnostic, not carrying. `gain=2`, `force=4`, and early terminal pitch
trigger moved far (`0.85053 m`) but failed with `fall_events=68` and
`box_drop_events=56`.

Current next gate:

Test staged drive authority instead of static terminal posture: low authority
during the movement window, then rewrite G1 drive stiffness/damping/max-force
only when terminal hold latches. Added `terminal_drive_gain_scale` and
`terminal_drive_force_scale`, with summary fields for
`terminal_drive_gain_applied_step` and the applied terminal drive table.
Batch `curiosity_g1_terminal_drive_0705`, job-name `g1_term_drive`, tested
`diag93`-`diag96`. All four failed. The terminal drive rewrite did happen:
`terminal_drive_gain_applied_step` was `342` for pitch-trigger runs, `337` for
travel trigger, and `320` for start-step trigger. But the final behavior
remained the same pitch/drop failure: final box target-directed travel stayed
around `0.70-0.71 m`, with `fall_events=244-246` and
`box_drop_events=237-239`.

Updated interpretation:

The current open-loop G1 forward motion is not a recoverable carry gait. It is
mostly a controlled-looking entry into forward pitch runaway. After the system
starts moving, static terminal posture, early terminal posture, all-rollout
high authority, and terminal-time high authority have all failed to create a
stable post-carry hold. The next implementation should not keep sweeping this
controller family. It should either:

- switch to a real controller-backed locomotion policy if the local Isaac/G1
  assets can run without new external blockers, or
- replace the current torso-cradle/open-loop setup with a more physically
  defensible support/contact scaffold that does not depend on falling forward
  for target-directed travel.

Controller-asset check:

The local G1 AGILE policy files are Git LFS pointers, not actual model
weights. For example
`external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt`
is a 132-byte pointer to a 6.65 MB object. Do not spend more time trying to
run this as a controller unless real weights are installed.

Contact-scaffold pivot:

Added `scripts/isaac/run_core_world_cradle_cart_contact_baseline_batch.sh` to
validate the free-box cradle contact mechanics independently of robot
locomotion. This batch is explicitly not a robot-carrying result: a
world-anchored prismatic rail moves a physical cradle/cart while the box
remains free dynamic. It tests 0.30/0.60 m travel, 0.5/2.0 kg boxes, and low
friction. Pending Slurm job `167041`, job-name `cart_contact`.

Cradle-cart contact result:

Slurm job `167041` completed. The contact scaffold passed all four tested
conditions with `box_drop_events=0`, `nonfinite_state_events=0`, and no box
pose writes. The key results:

- `diag1`: 0.30 m target, 0.5 kg box, friction `0.20/0.10`; final
  post-settle cart travel `0.29919 m`, box travel `0.29919 m`, post-settle
  relative error below `1e-6 m`.
- `diag2`: 0.60 m target, 0.5 kg box; final post-settle box travel
  `0.59978 m`, post-settle relative error below `1e-6 m`.
- `diag3`: 0.60 m target, 2.0 kg box; final post-settle box travel
  `0.59973 m`, post-settle relative error below `1e-6 m`.
- `diag4`: 0.30 m target, 0.5 kg box, low friction `0.05/0.03`; still no
  drop, but peak post-settle relative error rose to `0.02026 m`.

Interpretation:

The free-box contact/cage problem is solvable in Isaac when the support body
itself has stable controlled motion. The robot-side problem is therefore not
the box contact alone; it is stable locomotion/support while carrying. The next
robot scaffold should preserve the cradle/cage contact geometry and replace
the failed G1 open-loop pitch-runaway motion with a support mechanism that can
move without losing balance.

Low-CG robot-side scaffold result:

The low-CG prismatic-foot cage batch `167051` tested whether the successful
cage contact idea can be carried by a free articulated support body instead of
a world rail. It was stable but did not produce useful travel. All three
diagnostics had fall/drop 0, max tilt `0.03797 rad`, and no body root
pose/velocity commands. But final post-settle payload travel was essentially
zero:

- `diag1_translate_3cm`: `-0.00039 m` travel for a 0.03 m target.
- `diag2_translate_6cm`: `0.00017 m` travel for a 0.06 m target.
- `diag3_sync_3cm`: `-0.00053 m` travel for a 0.03 m target.

Interpretation:

The low-CG cage gives stable support and keeps the free box retained, but the
current stance-translate/sync-inchworm commands do not create propulsion. This
is the opposite failure mode from the G1 open-loop scene: G1 moves by falling;
low-CG cage stays stable but does not move. The next support-motion gate should
keep the low-CG/cage stability and add a support-consistent propulsion
mechanism that achieves at least `0.03 m` post-settle payload travel with root
writes 0, fall/drop 0, and bounded foot slip.

Negative-direction check:

The low-CG negative-direction batch `167057` ruled out target sign mismatch.
`diag4_translate_neg3cm` and `diag5_creep_neg3cm` both had fall/drop 0, root
writes 0, and max tilt `0.03797 rad`, but still produced essentially no
post-settle payload travel (`-0.00051 m` and `0.00011 m`). Therefore the
blocker is the support propulsion mechanism itself, not the sign of
`target_x`.

Current blocker:

The clean decomposition is now:

- Free-box cradle/cage contact can work when the support body moves correctly
  (`cart_contact diag1`-`diag4`).
- The open-loop G1 scene moves, but by entering unrecoverable forward pitch;
  it cannot be rescued by terminal posture or drive-authority switching.
- The low-CG prismatic-foot scaffold is stable and retains the free box, but
  its current stance-translate/creep/sync commands do not propel the payload.

Next implementation should target support-consistent propulsion for the
low-CG cage, not more G1 open-loop terminal sweeps and not more sign variants.

Rear-anchor push implementation:

Added a new `rear_anchor_push` motion mode to
`build_core_world_prismatic_carrier_stand.py`. It keeps the rear feet on the
ground as high-friction stance contacts, lifts the front feet slightly to
reduce drag, and drives only the rear x-slide joints after the settle phase.
This avoids body root writes and fixed-world foot locks. It is still a
diagnostic propulsion scaffold, not final walking.

Also added front/rear foot friction parameters and summary fields:
`front_foot_static_friction`, `front_foot_dynamic_friction`,
`rear_foot_static_friction`, and `rear_foot_dynamic_friction`.

Pending validation:

`curiosity_lowcg_rear_anchor_0705`, Slurm job `167080`, runs three variants:
positive 3 cm target, negative 3 cm target, and stronger friction asymmetry.
The required first gate is at least `0.03 m` post-settle payload travel with
fall/drop 0, root writes 0, and no fixed-world support.

Rear-anchor push result:

Slurm job `167080` completed. The mechanism stayed stable but did not produce
post-settle travel. `diag6`-`diag8` all had fall/drop 0 and root writes 0, but
the commanded x-slide target of `0.03 m` produced only about `0.00029 m`
maximum actual x-slide. Final post-settle payload travel stayed near zero.
This means the current blocker is x-slide actuator/constraint tracking under
the low-CG loaded cage, not target sign or friction asymmetry.

Rear-anchor authority result:

Added `scripts/isaac/run_core_world_prismatic_lowcg_rear_anchor_authority_batch.sh`
to raise x-slide stiffness/max-force by 10-50x. Slurm job `167094`,
job-name `rear_auth`, completed and was negative. All three diagnostics were
stable with fall/drop 0, but post-settle payload travel remained effectively
zero: `diag9` `-0.00012 m`, `diag10` `-0.00193 m`, and `diag11`
`0.00028 m`. Actual x-slide motion stayed around `0.00029-0.00032 m` even
when commanded target reached `0.03 m`.

Interpretation:

The current rear-anchor position-drive path is not just underpowered; it is
not converting commanded x-slide position into meaningful joint/body motion in
the loaded low-CG cage. Do not build a multi-step gait on top of this
position-drive behavior.

Next check:

Added `rear_anchor_velocity_push` to send x-slide velocity targets during the
rear-anchored push window while preserving the same no-root/no-fixed-world
posture. Pending Slurm job `167107`, job-name `rear_vel`, tests whether the
velocity-target channel can move the x-slide joints at all. If it also fails,
the next step should replace the prismatic x-slide support mechanism rather
than continue tuning position/velocity gains.

Rear-anchor velocity first result:

Slurm job `167107`, job-name `rear_vel`, completed and stayed stable, but it
was not a valid pure velocity-drive test. The implementation still supplied
position targets for the same x-slide DOFs, and the x-slide drive stiffness
remained nonzero. The observed result matched the earlier position-drive
blocker: actual x-slide stayed about `0.00031 m`, while final post-settle
payload travel was between `-0.00160 m` and `0.00029 m`.

Implementation correction:

`rear_anchor_velocity_push` now sets x-slide drive stiffness to `0.0` in this
mode and applies sparse actions: vertical joints receive position targets, and
x-slide joints receive velocity targets. This follows the Isaac API guidance
that velocity control needs zero stiffness and nonzero damping. Slurm job
`167124`, job-name `rear_vel2`, was invalid because the compute node read a
transient corrupt source line before Isaac startup and failed `py_compile`.
Retry2 job `167125`, job-name `rear_vel3`, is pending with a longer
compute-side startup delay and stamps `diag18`-`diag20`.

Sparse velocity retry2 result:

Slurm job `167125`, job-name `rear_vel3`, completed and is a valid negative
result. With x-slide stiffness set to `0.0` and sparse velocity targets
enabled, `diag18`-`diag20` all had fall/drop 0, but actual x-slide still
stayed around `0.00029 m`. Final post-settle payload travel was effectively
zero: `diag18` `0.00018 m`, `diag19` `0.00000 m`, and `diag20`
`0.00018 m`. Raising x-slide velocity from `0.03` to `0.08 m/s` had no
observable effect.

Next diagnostic:

Add direct rear x-slide effort control. This is not another gain sweep; it
tests whether any command path can inject horizontal joint force into this
prismatic support mechanism. If direct efforts do not move the x-slide, the
mechanism should be replaced rather than tuned further.

Rear-anchor effort result:

Slurm job `167126`, job-name `rear_eff`, completed and is a valid negative
result. Direct rear x-slide efforts of `5000 N` and `20000 N` did not move the
x-slide beyond about `0.00027 m`. Final post-settle payload travel remained
near zero: `diag21` `-0.00092 m`, `diag22` `-0.00047 m`, and `diag23`
`-0.00009 m`, with fall/drop 0.

Decision:

Stop tuning the prismatic x-slide support. The position-drive, velocity-drive,
and direct-effort command paths all fail to create support-consistent
propulsion for the loaded low-CG cage. The next scaffold should replace the
support/propulsion mechanism. A rolling-foot or wheel-joint scaffold is
acceptable only as an actuator-driven ground-contact diagnostic; it must not
be claimed as walking humanoid carrying.

Rolling-foot scaffold update:

Added `scripts/isaac/build_core_world_rolling_foot_cage_carrier.py` and
`scripts/isaac/run_core_world_rolling_foot_cage_carrier_batch.sh`. This
scaffold uses four velocity-driven revolute wheel joints, a low-CG torso, a
fixed physical cage, and a free dynamic box. It is explicitly not walking.

First validation `167128` was invalid because `/World/Robot` was not defined
as an articulation root; no summaries were produced. After adding
`UsdPhysics.ArticulationRootAPI`, retry2/retry3 ran but were negative:
1 kg cases retained the box but had torso z around `0.114 m`,
`fall_events=792`, and only `0.00993-0.01422 m` post-settle payload travel.
The 2 kg case dropped the box. Reading `/World/Robot/Torso` directly produced
the same low z, so the failure is not only articulation-root pose reporting.

Next rolling-foot check:

Added wheel joint motion metrics and a one-case launcher
`scripts/isaac/run_core_world_rolling_foot_cage_jointmotion_diag.sh`. If wheel
joints rotate but the carrier does not translate, the issue is wheel-ground or
cage geometry. If wheel joints do not rotate, the issue is the revolute
velocity command/drive setup.

Rolling-foot command-path result:

The one-case velocity diagnostic `diag10` showed that wheel velocity commands
do not meaningfully rotate the joints: max wheel motion was only `0.00986 rad`,
final wheel motion `0.00282 rad`, and final post-settle payload travel
`0.00144 m`. The direct wheel-effort diagnostic `diag11` was also negative:
`200 Nm` effort produced max wheel motion only `0.00132 rad`, final wheel
motion `0.000045 rad`, and final post-settle payload travel `0.000047 m`.

Decision:

Stop the current rolling-foot route. Both velocity and effort command paths
fail to actuate the revolute wheel joints meaningfully. Do not sweep wheel
velocity, torque, or friction on this USD model. A future rolling/wheeled
diagnostic should start from a known-good Isaac wheeled robot asset or a
minimal wheel articulation already verified to spin under Core API control.

Arena G1 loco-manipulation baseline pivot:

Local WBC-AGILE and GR00T-VisualSim2Real standalone walk model files under
`external/` are Git LFS pointers, so they are not usable controller weights.
However, the local Arena tutorial checkpoint under
`/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000/`
contains complete 4.7 GB and 1.5 GB safetensors shards plus index/config
files. Its README identifies it as a fine-tuned NVIDIA Isaac GR00T model for
the synthetic IsaacLab Arena G1 loco-manipulation pick-and-place task.

Arena baseline stopped:

Per user correction, external-model rollout checks must not block direct Isaac
scene construction. The Arena/GR00T smoke had already started its server and
loaded the G1 loco-manipulation config/assets, but it was not needed for the
current blocker: replacing the scaffolded support/locomotion backend. Slurm
job `167139` was canceled intentionally after about 8 minutes. Treat it as an
aborted optional baseline smoke, not as carrying evidence.

Direct Isaac backend replacement:

Added `scripts/isaac/run_official_h1_callback_locomotion_smoke.py` and
`scripts/isaac/run_official_h1_callback_locomotion_smoke.sh`. This is a
no-box controller-backed locomotion probe using the installed NVIDIA
`H1FlatTerrainPolicy`, local H1 USD, and local H1 PhysX policy/env files. It
follows the official test pattern by calling `forward()` from a physics-step
callback. If it passes, the next direct Isaac step is a small fixed/light
payload H1 locomotion diagnostic, then a bridge into the existing carry-task
contract. If it fails, do not repeat it unchanged; continue with a different
controller-backed robot or a new physical support backend. In all cases, the
validated task runner, hidden-box randomization, active-probe fields,
posture/action interface, contact-report gates, and no-shortcut checker remain
the reusable task contract.

H1 result:

The H1 sample-policy probe is not immediately usable in this environment.
Retry2 with the Isaac Sim base python kit failed dependency resolution for
`isaacsim.anim.robot.schema` before app startup. Retry3 and retry4 with the
IsaacLab headless kit reached H1 policy construction but failed before rollout
with `Path.IsValidPathString(NoneType)`, even after switching from explicit
paths to the same automatic path-selection style used by NVIDIA's installed
`test_h1.py`. Do not continue H1/Go2 official sample-policy retries unchanged.

Updated next step:

Keep the direct carry task runner as the stable interface, but replace the
backend directly in Isaac. The next backend must prove, in order: no-box
locomotion/support with contact/slip metrics; small fixed/light payload; then
free dynamic box carrying through the existing hidden-box/probe/posture/action
contract. External model or policy-server rollouts are optional baselines only
and must not block this path.

Direct Isaac no-box support isolation:

Per user correction, stop waiting on external models and build/debug the
carrying scene directly in Isaac first. Added `payload-mode=none` to
`scripts/isaac/build_core_world_anchored_footstep_carrier.py`. In this mode
the carry box is not spawned; legacy payload metric fields intentionally use
the torso pose as a compatibility proxy and the summary marks
`payload_spawned=false`, `no_box_support_smoke=true`, and
`payload_metric_proxy=torso_pose_when_payload_mode_none`. This is only a
support/locomotion backend isolation diagnostic, not carrying evidence.

No-box support result `20260705_anchor_nobox_support_diag1`, Slurm job
`167156`, completed on `server28`. It was a valid negative result: no fall
and no payload/drop because no payload was spawned, but the torso did not move
meaningfully. Final post-settle torso travel was only `0.00038 m` toward a
`0.24 m` target; final target distance remained `0.24398 m`. The support-foot
joint motions were nonzero and effort support proxy reported at least four
supported feet, but contact-report active contacts were not maintained. Do
not add a box on top of this configuration.

No-box planted-rail propulsion result
`20260705_anchor_nobox_propulsion_diag2`, Slurm job `167157`, completed on
`server28`. It was also a valid negative result. Enabling planted stance rail
propulsion produced rail commands, but the torso moved in the wrong direction:
final post-settle torso travel was `-0.03338 m`, while the target was
`+0.24 m`; final target distance increased to `0.27773 m`. There were still
no falls, but max tilt rose to `0.37120 rad`, and the near-ground support gate
was weak for part of the rollout (`drive_near_ground_lt2_steps=269`). This
suggests a propulsion sign/support-geometry issue in the current xz-prismatic
support backend.

Rail sign control:

The queued negative-target sign diagnostic was canceled before running because
the existing positive-target log already showed the immediate problem: positive
rail targets moved the torso backward. Added an explicit
`--rail-target-direction-scale` / `RAIL_TARGET_DIRECTION_SCALE` diagnostic
control; the default remains `1.0` for reproducibility.

Inverted-rail no-box result:

Positive-target no-box smoke with `RAIL_TARGET_DIRECTION_SCALE=-1.0`, output
`experiments/outputs/core_world_anchored_footstep_carrier/20260705_anchor_nobox_invertrail_diag4/`,
tmux `curiosity_anchor_nobox_invertrail_0705`, Slurm job `167161`, completed.
It is a valid negative result. The rail target sign changed as intended, and
fall/drop stayed 0, but final post-settle torso travel was only `0.00038 m`
toward a `0.24 m` target; final target distance remained `0.24398 m`.
Support effort and near-ground gates looked nominal, so the lack of travel is
not due to immediate falling.

Decision:

Stop the current `xz_prismatic_to_anchor` support backend. It now has three
no-box negative results: no propulsion with rail disabled by support-foot
drive, backward/tilting motion with planted rail positive sign, and no
meaningful travel with inverted rail sign. Do not add fixed payload or free
box on top of this backend. The next direct Isaac route should use a different
backend, preferably a real robot articulation whose no-box standing and small
fixed-payload standing gates already passed, and then improve travel before
returning to free-box carrying.

G1 conservative long status:

Existing outputs for `diag77`-`diag80` show that the conservative staged G1
long-validation batch already ran. All four 700-step variants failed with
late falls and box drops. Final box target-directed travel was large
(`0.75886-0.86646 m`), but fall events were `148-232`, box drop events were
`131-215`, max tilt was about `1.13-1.14 rad`, and min box z fell to
`0.05-0.108 m`. These are not near misses; they confirm that large travel in
the current staged G1 family is coupled to delayed pitch/drop failure.

Next G1 isolation:

Before adding another box/cradle variant, run the same real G1 articulation
with no carry box spawned and no torso cradle for a 700-step staged-gait
isolation. If no-box G1 also fails, the blocker is locomotion/balance, not box
retention. If no-box G1 stays stable but travels only a few millimeters, the
current open-loop gait is not a locomotion solution and should be replaced
with a controller-backed G1 policy or a new walking controller.

G1 no-box staged-gait isolation result:

`20260705_core_world_g1_nobox_staged_iso_diag1`, Slurm job `167163`, completed
on `server53`. It is a valid isolation result: no carry box was spawned, no
torso cradle was spawned, rollout root/velocity/box pose writes were 0,
completed steps were `700/700`, fall/drop were 0, max tilt was only
`0.01314 rad`, and min robot z was `0.78289 m`. However it did not walk:
final robot target-directed travel was `-0.00067 m`, and max target-directed
robot travel was only `0.00522 m`.

Decision:

The current staged/open-loop G1 gait family is stable as a stand/march
diagnostic but is not an actual locomotion backend. The large free-box travels
seen in failed 700-step carrying runs are therefore not credible walking
progress; they are coupled to delayed pitch/drop failure. Do not keep tuning
open-loop amplitude, ramp, terminal hold, or cradle mass in this family as if
it will become the final carrying solution. The next G1 path needs a
controller-backed locomotion policy or a materially different walking
controller before returning to payload/free-box carrying.

WBC-AGILE weight repair:

Re-inspected the failed WBC-AGILE Core API G1 route. The previous ONNX and
torch-checkpoint attempts stopped at policy loading because the local
`external/WBC-AGILE/agile/data/policy/velocity_height_g1/` model files were
Git LFS pointer text files, not real model weights. `git-lfs` is not installed
on the login node, so the official files were downloaded directly from
`https://media.githubusercontent.com/media/nvidia-isaac/WBC-AGILE/main/...`
into the existing official repository paths. Verified sizes now match the LFS
pointers: recurrent student checkpoint `6.4M`, ONNX `2.0M`, TorchScript `.pt`
`2.0M`. Lightweight syntax checks passed for
`scripts/isaac/build_core_world_g1_box_scene.py`,
`scripts/isaac/check_core_world_g1_box_scene_summary.py`, and
`scripts/isaac/run_core_world_g1_box_scene.sh`.

Pending controller-backed no-box smoke:

Submitted `20260705_core_world_g1_agile_policy_nobox_diag4_realweights`,
tmux `curiosity_g1_agile_torch_realweights_0705`, Slurm job `167164`, using
`GAIT_MODE=agile_policy`, `AGILE_POLICY_BACKEND=torch_checkpoint`, no carry
box, no torso cradle, and local official WBC-AGILE real checkpoint weights.
This is a no-box locomotion smoke only. If it loads and enters rollout, judge
it by nonzero target-directed robot travel, fall 0, no rollout root/velocity
writes, policy inference count, and raw action norm.

## 2026-07-05 User-Correction Direct Isaac Pivot

The real WBC-AGILE files now load, so the previous loading blocker was
removed. However, controller-backed Core API G1 smokes are negative in this
scene. `diag5_directload` entered rollout with official weights but fell
(`fall_events=359`, max tilt `3.0337 rad`). `diag6_zero_cmd` also fell, and
the corrected stable-pose zero-command test `diag7_zero_stablepose` still
fell with `fall_events=92`, max tilt `2.50074 rad`, min robot z `0.06682 m`,
and only `0.00737 m` max target-directed travel. Therefore WBC-AGILE is not
on the critical path right now. The likely issue is observation/action or
simulation-convention mismatch, but debugging that should not block direct
Isaac scene construction.

After the user correction, the active immediate path is the Core API G1 scene
itself. `20260705_core_world_g1_box_in_front_scene_smoke_retry2` passed a
diagnostic scene baseline: G1 stands with 43 joints, stable stand posture and
arena gains; a free dynamic 2 kg box rests on the ground in front of the
robot; `attach_box=none`; `torso_cradle=none`; fall/drop are 0 for 360 steps;
rollout root pose, root velocity, and box pose writes are all 0. This proves
the base Isaac scene can exist without external models or scaffolded support.
It does not prove walking, probing, lifting, or carrying.

Next plan:

1. Keep the G1 + box-in-front scene as the base environment.
2. Add explicit task phases in the scene rather than waiting on external
   policies: target marker, approach/probe placeholder, contact/grasp attempt,
   lift, and carry.
3. Each new phase must keep the no-shortcut counters and diagnostic-only claim.
4. Do not return to the current open-loop G1 gait family, AGILE smokes, or
   prismatic support scaffold as if they are final locomotion solutions.

## 2026-07-05 Front-Probe Contact Diagnostic

Implemented the first explicit task phase on top of the G1 + free-box scene:
`probe_mode=front_bumper`. This adds a light physical probe pad fixed to the
G1 torso and records probe-specific telemetry: start step, reference poses,
active steps, box displacement, target-directed probe travel, and whether the
box moved. The checker now has explicit probe gates, so this diagnostic cannot
silently pass as a generic stand scene.

Evidence:

- Aggressive geometry
  `20260705_core_world_g1_front_probe_bumper_submit_retry4` was negative.
  It moved the free box by `0.134 m` target-directed with no root/box pose
  writes, but the contact impulse toppled G1 (`fall_events=284`, max tilt
  `2.55803 rad`).
- Gentle geometry
  `20260705_core_world_g1_front_probe_bumper_submit_retry5_gentle` passed the
  probe diagnostic gate. It completed `360/360`, fall/drop `0`, max tilt
  `0.05226 rad`, min robot z `0.78356 m`, min box z `0.16048 m`, rollout
  root pose/root velocity/box pose writes all `0`, final probe box travel
  `0.15285 m`, and final target-directed probe travel `0.15260 m`.

Interpretation:

This is real direct-Isaac progress from static scene to physical object
perturbation while keeping G1 balanced. It is still not the final task: the
probe pad is torso-fixed, not a learned arm strategy; there is no grasp, lift,
carry posture, or walking while carrying. The next phase should replace or
augment the bumper with limb/end-effector contact, then add staged grasp/lift
diagnostics before returning to locomotion.

## 2026-07-05 Staged Grasp/Lift And March Diagnostics

Implemented `grasp_mode=staged_fixed_torso` in the G1 Core API scene. After
the probe phase, the rollout can create a runtime fixed joint from the G1
torso path to the free box and optionally add a small z offset. The summary
records attach step, local offset, attach poses, post-grasp box height
change, and whether the joint attached. The checker now has explicit grasp
and lift gates.

Evidence:

- `20260705_core_world_g1_probe_grasp_lift_retry1` passed the staged
  grasp/lift gate. It attached at step `140`, completed `360/360`, fall/drop
  `0`, max tilt `0.06396 rad`, no rollout root/velocity/box pose writes, max
  post-grasp box z delta `0.06895 m`, and final post-grasp z delta
  `0.01703 m`.
- `20260705_core_world_g1_probe_grasp_march_retry1` passed a 420-step
  grasp+small-march diagnostic with open-loop march amplitude `0.05`, but
  robot target-directed travel was only `0.04027 m`.
- `20260705_core_world_g1_probe_grasp_march_retry2_amp010` passed a 520-step
  grasp+larger-march diagnostic with amplitude `0.10`, but robot
  target-directed travel was only `0.03264 m` and the grasped box oscillated
  laterally.

Important caveat:

PhysX warned that `/World/CarryBox/StagedFixedTorsoGraspJoint` had disjoint
body transforms and would likely snap bodies together. Therefore these are
only staged fixed-joint grasp/lift diagnostics. They prove that the scene can
progress from free-box probe to attached/lifted-box balancing without root or
box pose writes. They do not prove physically faithful hand grasping, learned
posture choice, or walking while carrying.

Next implementation priority:

Do not keep increasing open-loop march amplitude as the locomotion solution.
The next useful branch is either a real target-directed walking controller for
G1 after grasp, or replacing the torso-fixed joint with limb/end-effector
contact and a less impulsive grasp mechanism before locomotion is retried.

## 2026-07-05 Direct Isaac Front-Tray Contact Route

The hand-link and torso-link staged fixed-joint release diagnostics did not
hold the box after support-table removal. The active route is now a direct
Isaac physical-contact posture: a torso/front tray attached to the G1 body,
with the box free and supported by contact instead of a fixed grasp joint.

Immediate gates:

- First run: stand-only free-box-on-front-tray diagnostic. Require cradle
  pieces present, no staged grasp, no support table, fall/drop `0`, no rollout
  root/velocity/box pose writes, stable robot height, and box height staying
  above the tray instead of falling.
- Second run only if the first passes: targeted-creep with the same free box
  contact setup. Require positive robot/box target-directed travel without
  root/box pose writes.
- If the box falls or destabilizes the robot, adjust tray local position,
  deck size, rail/end-stop geometry, and box spawn height/position before any
  locomotion attempt.

This route does not wait on external models. External controllers can still
be useful later, but the current blocker is scene/contact construction.

## 2026-07-05 Direct Isaac Hand-Link Grasp Branch

User correction: do not wait on external model/code paths if they are not
immediately useful. The active path is direct Isaac construction of the G1 +
box task, adding one real scene phase at a time and keeping all claims
diagnostic until the full walking-carrying task is actually demonstrated.

Implemented a body-parametric staged grasp path:

- `grasp_mode=staged_fixed_body` selects a specific rigid body/link through
  `--grasp-body-path`.
- The previous `staged_fixed_torso` behavior still exists, but now maps to
  the selected active body path instead of hard-coded attach logic.
- The runtime fixed joint computes `localPos0` by converting the box-world
  offset into the selected body frame using the body quaternion.
- Summary now records `active_grasp_body_path`,
  `grasp_body_wrapper_initialized`, `grasp_body_wrapper_error`, and
  `grasp_body_pose_at_attach_wxyz`.
- The checker can require `--expect-grasp-body-path` and
  `--expect-active-grasp-body-path`.

Pending diagnostic:

- `20260705_core_world_g1_probe_hand_grasp_lift_retry1`, Slurm job `167195`,
  tmux `curiosity_g1_hand_grasp_retry1_0705`.
- Config: gentle front bumper probe, then
  `GRASP_MODE=staged_fixed_body`,
  `GRASP_BODY_PATH=/World/G1/right_hand_palm_link`, attach step `140`, and
  lift offset `0.03 m`.

Interpretation gate:

If this fails, record whether the failure is API/link-pose, fixed-joint snap,
balance, or box drop. Then move to explicit arm/hand pose setup or a
two-hand/chest-supported staged body-contact design. Do not fall back to
calling torso-fixed grasp a hand grasp.

Result:

`20260705_core_world_g1_probe_hand_grasp_lift_retry1` passed the mechanical
diagnostic gate: `360/360`, fall/drop `0`, no rollout root/velocity/box pose
writes, right hand wrapper initialized, max tilt `0.05226 rad`, max
post-grasp z delta `0.01450 m`, final post-grasp z delta `0.00409 m`.

But it did not solve the physical grasp issue. At attach, the right palm was
still roughly `0.967 m` from the box, with local offset
`[0.651, 0.143, -0.702]`, and PhysX still warned about disjoint fixed-joint
transforms. This means the body-parametric API works, but the hand is not
actually near the object.

Next branch:

Add an arm-pose phase before attach. The scene now supports `ARM_POSE_MODE`
and manual shoulder/elbow/wrist overrides, records arm-pose active steps, and
records the selected body-to-box world distance at attach. The next diagnostic
is `20260705_core_world_g1_armreach_hand_grasp_retry1`, which uses
`ARM_POSE_MODE=right_front_reach` before a right-palm staged attach. The
primary success criterion for this branch is reducing attach distance, not
claiming final carrying.

## 2026-07-05 Moving-Carrier Scene Baseline

User correction: do not block on external models, downloaded weights, or
controller debugging when they are not immediately useful. The current direct
Isaac path separates two things:

- scene/contact validation: free dynamic box on a G1 front tray under a moving
  carrier;
- final locomotion evidence: no rollout root-pose/root-velocity writes.

Implemented an explicitly labeled `diagnostic_root_drive=smooth_x` mode in
the Core API G1 scene. It writes only the G1 root pose during rollout and never
writes the box pose, so the box must remain on the tray through Isaac contact.
This mode is allowed only for contact/scene/metric diagnostics and must never
be described as biped walking or final carrying.

Results so far:

- `20260705_core_world_g1_front_tray_freebox_rootdrive_retry1` passed the
  moving-carrier contact diagnostic with fall/drop `0`, min box z `0.78757 m`,
  final box target-directed travel `0.18729 m`, no box pose writes, and 440
  root-pose writes. It had a root-drive handoff artifact.
- `20260705_core_world_g1_front_tray_freebox_rootdrive_retry2` fixed the
  current-pose handoff and passed the explicit diagnostic checker, but kept
  sideways drift because it inherited the loaded forward-pitch posture at
  step 120.
- `20260705_core_world_g1_front_tray_freebox_rootdrive_step0_retry3` passed and
  is the preferred baseline: root-drive starts at step 0 with the same ramp and
  speed, fall/drop are `0`, box pose writes are `0`, final robot/box
  target-directed travel is `0.22521/0.22965 m`, max tilt is only
  `0.00880 rad`, and the diagnostic-root-drive checker passes.

Next gate after a clean moving-carrier baseline:

1. keep the free-box tray/contact scene and checker gates;
2. return to no-root locomotion by either a controller-backed G1 policy that
   enters rollout cleanly or a materially different walking/support controller;
3. require `root_pose_write_count_rollout=0`,
   `root_velocity_write_count_rollout=0`, and
   `box_pose_write_count_rollout=0` for any real walking-carrying claim.

## 2026-07-05 Prismatic Cradle Scaffold Recovery

The no-root prismatic scaffold remains a bridge diagnostic only, but it is
the current fastest Isaac route for testing free-box carrying without writing
robot or box root state. Retry1 and retry2 failed because cycle count/stride
changed when the target was shortened from `-0.30 m` to `-0.22 m`. Retry3
fixed cycle count and stride but still failed because the new launcher placed
the payload and cradle about `0.40-0.42 m` behind the old successful geometry.
Retry4 reproduced the old x stop geometry but failed immediately because the
current support/height/contact condition did not match the old stable run.
Therefore old x geometry alone is not a valid recovery path.

Retry5/Retry6 showed the constructive route:

- moderate front offset with the normal support polygon is stable but barely
  moves;
- front offset with a wider support polygon can carry the free 8 kg box
  without root/body/box/payload shortcut writes;
- retry6 reached `0.22147 m` max absolute post-settle payload travel and
  `0.01513 m` final target distance with fall/drop `0`, but it still fails
  on transient tilt, minimum payload height, and relative slosh.
- retry7 showed that raising payload height to `0.20 m` removes tilt/slosh
  failures but suppresses useful travel.
- retry8 found the current strict-pass point at payload height `0.16 m`:
  8 kg free box, no root/body/box/payload shortcuts, fall/drop `0`, max tilt
  `0.09174 rad`, min payload z `0.71612 m`, max post-settle travel
  `0.24384 m`, and final post-settle target distance `0.00076 m`.

Next action:

1. keep the launcher default at stable `PAYLOAD_LOCAL_X=0.08` and test
   forward payload positions only as explicit scaffold overrides;
2. if using overhung payload positions, enlarge the support polygon or revise
   payload height/contact geometry first, then require the same no-shortcut
   gates;
3. require no root/body/box/payload shortcut writes and the existing
   fall/drop/tilt/payload-height gates;
4. use this only to validate a no-root physical-contact scaffold, then return
   to a real humanoid locomotion backend for the actual research target.

Immediate scaffold refinement:

1. keep the retry6 distance-producing setup as the baseline;
2. reduce peak tilt below `0.13 rad` by lowering stride acceleration or
   adding a smoother support cycle;
3. reduce payload relative offset/slosh by tuning cradle clearance/wall
   geometry and using the new post-settle-relative metric for analysis;
4. do not call any of this humanoid walking until a G1 or other humanoid
   controller produces the same gates with root pose/velocity writes at zero.

Current milestone:

`20260705_prismatic_cradle_sync_inchworm_neg23cm_x050_z016_support065_stride007_8kg_retry8b`
is the reproducible strict-pass no-root physical-contact carrying baseline.
Use it as the scaffold reference while moving back toward the actual final
objective: a robot-like/humanoid walking backend carrying the same free box
without root-drive, payload-pose writes, or staged pose locks.

## 2026-07-05 Walking-Like Multi-Posture Scaffold

`retry12` converted the strict-pass prismatic cradle scaffold into a
walking-like support-switching diagnostic by separating the reported task
target (`TARGET_X=-0.17`) from the internal gait-drive distance
(`GAIT_DRIVE_TARGET_X=-0.23`). It passed for both quasistatic and prelift
quasistatic step cycles, but remains a custom prismatic scaffold, not a
humanoid or learned controller.

`retry13` tested posture sensitivity under the same 8 kg free-box setup. The
valid compute checker was Slurm job `167313` (`prism_r13_ck2`). Result:

- High carry (`PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.18`) stayed safe but
  failed travel/target gates. It is under-driven in this gait schedule.
- Close mid-height carry (`PAYLOAD_LOCAL_X=0.45`, `PAYLOAD_LOCAL_Z=0.16`)
  passed the strict gate: fall/drop `0`, all root/body/box/payload writes `0`,
  max tilt `0.08797 rad`, min payload z `0.72143 m`, max payload-relative
  offset error `0.03811 m`, max post-settle payload travel `0.18958 m`, and
  final post-settle target distance `0.00786 m`.
- Low carry (`PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.14`) reached the
  target but failed the clearance gate with min payload z `0.69461 m`.

Next action is direct Isaac scene work, not waiting on external models:
extend the passing close-mid posture to a longer rollout, retest the original
mid-height posture under the same longer horizon, and test whether moving the
low posture closer to the body recovers the payload-height margin.

`retry14` completed that gate. The valid checker was Slurm job `167319`
(`prism_r14_chk`), and all three valid 2800-step posture runs passed the same
strict no-shortcut gate:

- Close mid-height (`x=0.45`, `z=0.16`) remained stable and passed.
- Original mid-height (`x=0.50`, `z=0.16`) remained stable and passed.
- Close low (`x=0.45`, `z=0.14`) passed with min payload z `0.70213 m`,
  recovering the low-carry clearance failure seen at `x=0.50`, `z=0.14`.

The remaining posture boundary from `retry13` is high carry (`z=0.18`), which
was stable but under-driven. The next direct Isaac gate is to recover high
carry by testing closer payload placement and/or stronger diagnostic gait
drive while preserving the same no root/body/box/payload write counters,
fall/drop `0`, payload-height, tilt, relative-offset, and target-distance
gates.

`retry15` tested that high-carry recovery gate and did not pass. The formal
checker was Slurm job `167324` (`prism_r15_chk`) on server63. All three
high-carry variants stayed safe and clean under the no-shortcut counters, but
failed the same two transport gates:

- High-close same drive (`x=0.45`, `z=0.18`, drive `-0.23`): max post-settle
  payload travel `0.08122 m`, final post-settle target distance `0.13420 m`.
- High-mid stronger drive (`x=0.50`, `z=0.18`, drive `-0.31`): max
  post-settle payload travel `0.10265 m`, final post-settle target distance
  `0.11544 m`.
- High-close moderate drive (`x=0.45`, `z=0.18`, drive `-0.27`): max
  post-settle payload travel `0.10426 m`, final post-settle target distance
  `0.11558 m`.

The useful result is negative: high carry has good clearance and remains
stable, but the current diagnostic foot-contact drive saturates before moving
the payload far enough. The next direct Isaac step should change the scene
mechanics or contact/propulsion schedule rather than wait for external
models: for example a stronger physical propulsion phase, a posture
transition from high hold to the already passing mid/low carry envelope, or an
explicit active-probing transition that selects the lower-cost carry height.

`retry16` changed the direct Isaac propulsion envelope for the high-carry
case. It was run as Slurm job `167329` (`prism_high_r16`) on server44.
Formal checker scripts were prepared, but checker jobs `167330` and `167332`
were canceled after remaining pending for resource priority, so the current
record is from rollout summaries inspected with `jq`, not a completed checker
job. All three runs had fall/drop `0`, shortcut writes `0`, articulated joint
count `8`, and no nonfinite events.

- Larger stride high-mid (`x=0.50`, `z=0.18`, `STEP_LENGTH=0.10`,
  `GAIT_DRIVE_TARGET_X=-0.42`) reached max post-settle payload travel
  `0.16863 m`, but final target distance remained `0.05819 m`.
- Scaling swing-leg horizontal force to `0.0` made propulsion worse: max
  post-settle payload travel `0.08621 m`, final target distance `0.16092 m`.
- High-close with support overdrive `1.6` moved strongly but overshot: max
  post-settle payload travel `0.34731 m`, final target distance `0.11302 m`.

This converts the high-carry issue from "cannot move enough" to "can move,
but cannot stop or hold at the target under raw overdrive." The next direct
Isaac gate should therefore test target-aware guarded progression/stopping,
not more raw drive.

`retry17` tested target-aware guarded progression and produced a useful
negative result, not a pass. Slurm job `167333` completed. All three variants
kept the 8 kg high-carry free box safe and shortcut-clean, but the controller
stopped at roughly `-0.073 m` post-settle payload travel, leaving about
`0.096 m` final target error. The blocker was
`post_settle_payload_travel_loss`, not target tolerance. The diagnosis is a
direction bug in the guarded loss metric: for a negative-X task target, raw
`peak_x - current_x` treats correct negative-X progress as loss.

The next direct Isaac step is `retry18`: keep the same high-carry setup and
fix only the guarded loss metric to use directional progress toward the
guarded stop target. This should answer one narrow question: can the high
posture reach and hold the target once the controller no longer rejects
negative-X progress? If it still fails, the remaining issue is genuine
stop/hold mechanics rather than the previous sign error.

`retry18` answered that question positively for this scaffold. Slurm job
`167342` completed the two rollouts, and checker job `167343` reported
`status=pass` with `failures=[]` for both. The high-mid posture (`x=0.50`,
`z=0.18`) reached final post-settle payload target distance `0.00536 m`; the
high-close overdrive posture (`x=0.45`, `z=0.18`, overdrive `1.6`) reached
`0.01532 m`. Both had fall/drop `0`, no root/body/box/payload writes,
payload height above `0.73 m`, and `target_reached` as the final guarded
block reason.

The next useful Isaac-only step is not another single-case drive tweak. It is
to use this pass as a controlled scaffold for posture/load variation:
compare mid/high/low carry postures under changed mass and box shape, record
which posture remains stable and low-cost, and only then add an explicit
selector/probing policy. This keeps the work aligned with the real research
question instead of overfitting one hand-tuned scene.

`retry19` begins that variation step. It adds runner support for
`PAYLOAD_SIZE_X/Y/Z` and compares the same directional-guard controller across
mid/high postures for a heavier 12 kg standard box and an 8 kg taller box.
This is not autonomous posture selection; it is the scaffold-level evidence
needed before implementing active probing or a posture selector.

`retry19` completed successfully. Rollout job `167346` and checker job
`167349` both ran on server02, and all four variation cases passed with
`failures=[]`: 12 kg standard box at mid/high postures, and 8 kg tall box at
mid/high postures. The most useful observation is that the current scaffold
does not merely pass one hand-tuned 8 kg box: under the same controller,
payload mass and height can change while fall/drop/shortcut counters remain
clean. The remaining gap is decision-making. The next implementation should
turn these runs into a posture-choice table and add a deliberately simple
rule-based selector baseline before any active probing or RL.

`retry20` added that first rule-based selector scaffold. It reads a manifest
of nine completed, formally checked prismatic-cradle runs from retry14,
retry18, and retry19; applies the same safety/transport gates; then selects
the lowest carry height with at least `0.01 m` payload-height margin, breaking
ties by target error, tilt, and payload relative offset. Slurm job `167351`
ran the selector on server02. The report passed with no failures and selected
`mid_front` for the `standard_8kg`, `standard_12kg`, and `tall_8kg` box
conditions. The low-close 8 kg case passed the hard gate but had only about
`0.002 m` height margin, so the selector correctly avoided it. This is still
decision-making over a prismatic scaffold, not active probing or learned
locomotion. The next useful step is held-out selector-driven execution, for
example a 10 kg standard and 10 kg tall box using the selected posture, with
comparison against a non-selected posture.

`retry21` tested that held-out selector-driven execution. It used two box
conditions not present in the selector table: a 10 kg standard box and a 10 kg
tall box. For each, the selector's `mid_front` posture was run, with
`high_front` as a control. Rollout job `167353` and checker job `167366`
completed; all four cases passed the formal checker with no falls, no drops,
no shortcut writes, and `failures=[]`. For standard 10 kg, selected
`mid_front` reached final post-settle payload target error `0.00610 m`, while
the high control had `0.01969 m`. For tall 10 kg, selected `mid_front` reached
`0.00003 m`, while the high control had `0.00616 m`. The control was not
unsafe; this is not proof that mid is always superior. It is evidence that the
rule selector can execute a held-out scaffold case and preserve the safety and
target gates. The next step toward the user's actual objective is to add an
active probing hook before selection, while still treating the prismatic body
as a scaffold rather than final robot walking.

`retry22` added that active probing hook. The probe runs after settle and
before carry, logs observed micro-lift response, tilt, and payload-relative
offset, and explicitly sets `active_probe_uses_hidden_ground_truth=false`.
Rollout job `167368` and checker job `167373` completed. Both standard 10 kg
and tall 10 kg selected `mid_front` cases passed with fall/drop `0`, no
root/body/box/payload writes, 80 observed probe steps, available probe
belief, and final post-settle payload target errors under `0.01 m`.

The limitation is important: retry22 records the probe belief but does not
use it to alter the carry controller. The next direct Isaac step is retry23:
keep the same real target, but let the observed probe risk choose an internal
gait-drive scale after probing. This is the first closed-loop bridge from
active probing to action choice inside the Isaac scaffold. It is still not
RL, not video-conditioned learning, not humanoid walking, and not final
robot carrying.

`retry23` completed that first closed-loop bridge. It adds
probe-conditioned internal gait-drive scaling while keeping the real task
target and guarded stop target unchanged. Standard 10 kg produced an adaptive
bucket `low` and kept scale `1.0`; tall 10 kg produced an adaptive bucket
`medium` and selected scale `0.98`, changing the internal gait-drive target
from `-0.42 m` to `-0.41160 m`. Rollout job `167383` and checker job
`167384` completed, and both cases passed with fall/drop `0`, no shortcut
writes, active probe belief, no hidden probe ground truth, adaptive decision
fields, and final post-settle payload target error under `0.01 m`.

The next useful Isaac-only step is stronger than another gait-scale tweak:
let the probe choose a discrete carry/contact strategy, such as carry height,
payload x-position, or cradle/contact margin. That is closer to the user's
actual goal: the robot should use probing feedback to choose a body-suitable
and object-suitable carrying posture. This still needs to remain clearly
labeled as a prismatic-scaffold diagnostic until a real locomotion backend
replaces the scaffold.

`retry24` starts that step by adding probe-conditioned carry-height selection.
It keeps retry23's gait-drive adaptation but adds a second decision: low-risk
probe results keep the nominal carry height, while medium-risk probe results
select `lower_carry_medium`, implemented as a `+0.012 m` offset to the
vertical prismatic leg target during the carry phase. This is a deliberately
small posture change so the experiment isolates whether the probe-conditioned
posture branch works before using more aggressive contact or body-shape
changes.

`retry24` completed and passed formal checking. Rollout job `167387` and
checker job `167389` ran the standard 10 kg and tall 10 kg active-probe
cases. Standard 10 kg selected `nominal_height` with posture offset `0.0`.
Tall 10 kg selected `lower_carry_medium` with posture offset `0.012 m` and
effective leg target `-0.558`. Both kept fall/drop `0`, no shortcut writes,
active probe belief, no hidden probe ground truth, and final post-settle
payload target error under `0.006 m`.

The next major step is no longer another small scaffold selector. The
remaining gap to the user's real target is the locomotion backend: the current
body is a prismatic scaffold, not a humanoid or general robot that can walk
while carrying arbitrary posture choices. The next implementation should
reuse the probe/posture decision interface but attach it to a real robot
walking backend or a more faithful no-root articulated walking carrier.

## 2026-07-06 Direct Isaac G1 Scene Pivot

After the user correction, the active path should not wait on external model
or dataset readiness. The direct Isaac route is:

1. keep the already constructed G1 Core API scene with a free dynamic box on a
   front torso tray;
2. keep `DIAGNOSTIC_ROOT_DRIVE=none` for real no-root diagnostics;
3. preserve explicit checker gates for no fall, no drop, no rollout root
   pose/velocity writes, and no box pose writes;
4. try small controller changes that address the current blocker:
   strong balance feedback holds the load but cancels locomotion, while fixed
   nonzero pitch targets cause slow forward divergence.

The next implemented diagnostic is `retry10`: pulsed balance pitch targets
during `targeted_creep`. The pulse target applies for short windows and then
returns to zero so the robot can recover before pitch divergence. Three
configs are in
`scripts/isaac/run_core_world_g1_front_tray_freebox_pulsed_creep_batch.sh`.
This is a direct Isaac control diagnostic, not a model-download path, not RL,
not video guidance, and not final humanoid carrying.

Retry10 result:

The valid rollout was Slurm job `167397` on `server63`. All three pulsed-creep
variants were safe but non-locomotive. They completed `620/620` with fall/drop
`0`, root pose writes `0`, root velocity writes `0`, and box pose writes `0`.
The max box target-directed travel values were only `0.07256 m`, `0.07112 m`,
and `0.06689 m`, all below the diagnostic `0.08 m` travel gate and similar to
the previous stationary/under-driven G1 front-tray results.

This closes the current open-loop `targeted_creep` family as the main G1
locomotion route. It is useful as a stable free-box front-tray holding scene,
but not as a walking backend. The next direct-Isaac step should change the
locomotion mechanism itself: for example a real controller-backed velocity
policy with repaired observation/action conventions, or an explicit
contact/footstep controller that produces target-directed travel without
rollout root writes. Do not continue sweeping only pulse windows, pitch
targets, or feedback gains in this same gait family.

## Direct Carry Posture Suite

Because the G1 open-loop route did not become a walking backend, the current
best executable direct-Isaac path is the strict support-foot robot scaffold.
It already has evidence for a free dynamic box, alternating foot support, no
fixed-world stance anchor, no root shortcut, no support-root/anchor/foot/
stance pose writes, and three carry postures at a 64 cm target.

The next suite packages that scattered evidence into a single repeatable
diagnostic:

- script: `scripts/isaac/run_direct_carry_posture_suite_64cm.sh`;
- summarizer: `scripts/isaac/summarize_direct_carry_posture_suite.py`;
- postures: `front_mid`, `low_front`, `chest_high`;
- target: `0.64 m`, payload mass: `8 kg`;
- core gates: fall/drop `0`, root shortcut free, no fixed-world stance anchor,
  drive-phase near-ground foot count at least `2`, commanded stance support
  continuity, box travel at least `0.52 m`, target error at most `0.08 m`,
  max tilt at most `0.14 rad`, and support-polygon margin at least `0.12 m`.

If this passes, it should be treated as the current best complete-task
simulation scaffold: a robot-like articulated support-foot carrier can move a
free box across multiple carrying postures while preserving balance/support
metrics. It is still not the final research goal because it is not a humanoid
locomotion controller, not learned, and not video-conditioned.

Result:

The suite passed in Slurm job `167398` (`carry_suite64`) from tmux
`curiosity_direct_carry_posture_suite_0706` with command
`srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_suite64 bash scripts/isaac/run_direct_carry_posture_suite_64cm.sh`.
The output report is
`experiments/outputs/direct_carry_posture_suite/20260706_direct_carry_posture_suite_64cm_8kg/direct_carry_posture_suite_summary.json`
with `status=pass`, `failures=[]`. `front_mid`, `low_front`, and
`chest_high` each completed `3580` steps with fall/drop `0`,
`root_shortcut_free=true`, no fixed-world stance anchor, support continuity,
and no target/tilt/support-margin gate failure. Max box travel was
`0.67301 m`, `0.66675 m`, and `0.65313 m`; final target distance was
`0.02369 m`, `0.00189 m`, and `0.01468 m`; max tilt was `0.12141 rad`,
`0.12326 rad`, and `0.12221 rad`; min support margin stayed around
`0.1594-0.1598 m`.

A checker-only recomposition of the existing 20260705 strict 64 cm summaries
also passed as Slurm job `167399` (`carry_suite_chk`) with output
`experiments/outputs/direct_carry_posture_suite/20260706_existing_20260705_strict64_suite/direct_carry_posture_suite_summary.json`.

Interpretation: this is now the strongest direct-Isaac complete-task scaffold
baseline in the repo. It should be used as a regression target while moving
toward a real robot locomotion backend or a wider posture-space stress suite.
It is not final humanoid carrying, not learned control, and not
video-conditioned active posture selection.

Next stage:

Broaden the direct-Isaac scaffold from three named postures to a
parameterized posture/hold-space stress suite before claiming robustness. The
next suite should include at least five carry configurations: front carry, low
carry, chest-supported carry, asymmetric/contact-shifted carry, and one harder
hold-height or hold-offset case. The gate remains strict: fall/drop `0`, no
root/box/support shortcut writes, no fixed-world stance anchor, support
continuity, target-directed box travel, bounded tilt, and support-polygon
margin. This still remains a scaffold result unless a real robot locomotion
backend replaces the support-foot carrier.

Implementation start:

The first stress-suite version keeps the physics backend unchanged and only
widens the named hold/posture defaults exposed by
`scripts/isaac/run_direct_carry_task_physical_backend.sh`. Added `front_reach`
(`payload_local_x=0.28`, `payload_local_z=0.04`, `torso_z=0.56`) and
`close_mid` (`payload_local_x=0.12`, `payload_local_z=0.05`,
`torso_z=0.55`). Added
`scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh`, which runs
`front_mid`, `low_front`, `chest_high`, `front_reach`, and `close_mid` at the
same 64 cm / 8 kg setting with the same strict gates as the 3-posture suite
and `--min-postures 5`.

Planned compute command:

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=03:00:00 --job-name=carry_stress5 bash scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh
```

This remains a scaffold stress diagnostic. A pass would mean the current
direct-Isaac support-foot robot can carry the free box across a wider hold
space, not that the final humanoid/video/RL objective is solved.

Result:

The 5-posture stress suite passed. It ran from tmux
`curiosity_direct_carry_posture_stress_0706` as Slurm job `167427`
(`carry_stress5`) on `server28` with command:

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=03:00:00 --job-name=carry_stress5 bash scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh
```

The output report is
`experiments/outputs/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/direct_carry_posture_stress_suite_summary.json`
with `status=pass`, `failures=[]`, and `case_count=5`. All five cases
completed `3580` steps with fall/drop `0`, `root_shortcut_free=true`, no
fixed-world stance anchor, drive-phase and commanded stance support continuity,
and no target/tilt/support-margin failure. Metrics:

- `front_mid`: max box travel `0.67301 m`, final target distance `0.02369 m`,
  max tilt `0.12141 rad`, min support margin `0.15951 m`;
- `low_front`: max travel `0.66675 m`, final target distance `0.00189 m`,
  max tilt `0.12326 rad`, min support margin `0.15984 m`;
- `chest_high`: max travel `0.65313 m`, final target distance `0.01468 m`,
  max tilt `0.12221 rad`, min support margin `0.15943 m`;
- `front_reach`: max travel `0.69996 m`, final target distance `0.02415 m`,
  max tilt `0.12007 rad`, min support margin `0.16035 m`;
- `close_mid`: max travel `0.69125 m`, final target distance `0.01431 m`,
  max tilt `0.12311 rad`, min support margin `0.15872 m`.

Interpretation: this is stronger posture-space scaffold evidence than the
3-posture suite. The current support-foot robot scaffold can carry a free
8 kg box 64 cm across five hold configurations while preserving balance and
support metrics. The remaining gap is still large: this is not a full humanoid
controller, not learned control, not video-conditioned active posture
selection, and not proof over arbitrary posture/load/morphology variation.

Next evidence step:

Produce MP4 visual audit artifacts for the strongest scaffold result on a
compute node. The visualization should be generated from the logged CSV and
summary files into `experiments/visuals/`, and recorded as evidence only, not
as a new control result. After that, the main technical branch should either
replace the scaffold locomotion backend with a real robot controller or add
active-probe-conditioned selection over this widened posture space.

Implementation start: added
`scripts/isaac/render_direct_carry_posture_stress_suite_videos.sh` to render
all five stress-suite cases into MP4 files under
`experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/`.
Planned compute command:

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:20:00 --job-name=carry_viz5 bash scripts/isaac/render_direct_carry_posture_stress_suite_videos.sh
```

Result:

Three render attempts exposed visualization-script compatibility issues, not
new rollout/control failures: job `167431` failed on CSV string parsing, job
`167432` failed because system `python3` lacked `cv2`/`imageio`, and job
`167433` failed because the renderer assumed y/z columns that the current
one-dimensional backend CSV does not record. The renderer now preserves
nonnumeric CSV fields, defaults missing y to `0.0`, defaults missing z to the
summary's torso/payload heights, and the batch script uses the prebuilt
`/public/home/yanhongru/envs/isaac_arena_py312/bin/python`.

Retry4 job `167434` (`carry_viz5d`) ran on `server02` and completed. It wrote
five MP4 audit videos and a manifest under:

```text
experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/
```

The manifest lists nonempty MP4 files for `front_mid`, `low_front`,
`chest_high`, `front_reach`, and `close_mid`. These are metric/trajectory
visualizations generated from logged CSV/summary files, not Isaac viewport
recordings and not new control evidence.

Next implementation stage:

Connect active probing to the widened posture space. The next diagnostic
should run at least two object/load-shape conditions, use observed probe
telemetry without hidden load ground truth, select among posture labels
including `front_reach` and `close_mid`, and then pass the same no-shortcut,
support-continuity, target, tilt, fall/drop gates. This would move from
"static posture stress passes" to "the scaffold uses probing feedback to pick
a carry posture." It still remains a scaffold result until the locomotion
backend is replaced by a real robot controller.

Implementation/result:

Added `scripts/isaac/select_direct_carry_posture_from_probe.py` and
`scripts/isaac/run_direct_carry_probe_selected_posture_suite.sh`. The selector
uses only normalized probe telemetry and explicitly rejects hidden-ground-truth
probe belief. It maps risk to posture as:

- low risk: `front_reach`;
- medium risk: `close_mid`;
- high risk: `chest_high`.

Two failed attempts were renderer/control plumbing rather than carry failures:
job `167437` proved the `vertical_probe` branch but failed before
`horizontal_probe` produced a normalized summary because `run_probe | tail -1`
masked a probe failure; job `167440` failed before suite execution on nested
shell quoting. The script now stores `LAST_PROBE_SUMMARY` directly and uses
`jq -r '.selected_carry_posture'`.

Retry3 job `167441` (`carry_probe3`) completed on `server46`. Output:

```text
experiments/outputs/direct_carry_probe_selected_posture_suite/20260706_direct_carry_probe_selected_posture_suite_retry3_64cm_8kg/
```

`vertical_micro_lift` with 60 probe steps and z amplitude `0.030` produced
probe risk `0.5987436213151278`, no hidden-ground-truth probe belief, and
selected `close_mid`. `horizontal_push_pull` with 60 probe steps and x
amplitude `0.050` produced probe risk `0.45948289037895235`, no hidden-ground
truth, and selected `front_reach`.

The selected carries passed:

- `close_mid`: `3580` steps, fall/drop `0`, max box travel `0.69125 m`, final
  target distance `0.01431 m`, max tilt `0.12311 rad`, min support margin
  `0.15872 m`;
- `front_reach`: `3580` steps, fall/drop `0`, max box travel `0.69996 m`,
  final target distance `0.02415 m`, max tilt `0.12007 rad`, min support
  margin `0.16035 m`.

Interpretation: this is the first current direct-Isaac support-foot scaffold
result where active-probe telemetry selects between widened posture labels and
the selected full carry episodes pass strict no-shortcut/support gates. It is
still two-stage and scaffold-only. It is not online in-episode geometry
selection, not a full humanoid locomotion controller, not learned control, and
not video-conditioned RL.

Next gate:

Collapse the two-stage selector toward the intended control setting. The
cleanest scaffold step is to prebuild multiple hold/contact options and
activate one after probing without root/box pose writes. The larger research
step is to move this selector interface into a real robot locomotion backend.
Do not claim final success until probing, posture selection, carrying, and
balance happen in the same intended robot/control loop.

Same-Episode Controller Step:

Added a narrower same-episode bridge before attempting online hold-geometry
switching. The support-foot backend can now compute probe belief at
`drive_start_step` and immediately choose the support controller profile for
the following carry phase inside the same rollout. The profile changes support
step height, double-support fraction, stance x, and swing x. It does not move
the robot root, box, support roots, feet, or stance anchors by pose writes, and
it does not rebuild or switch the physical hold geometry.

The new diagnostic script is:

```text
scripts/isaac/run_direct_carry_online_probe_adaptive_support_suite.sh
```

Planned cases:

- `vertical_probe`: `vertical_micro_lift`, z amplitude `0.030`, expected
  medium bucket and `compact_medium_double_support`;
- `horizontal_probe`: `horizontal_push_pull`, x amplitude `0.050`, expected
  low bucket and `nominal_reach_support`.

Planned command:

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_online_support bash scripts/isaac/run_direct_carry_online_probe_adaptive_support_suite.sh
```

This is closer to the final control-loop requirement than the two-stage
selector, but it is still scaffold-only: support-controller profile adaptation,
not online hold-geometry switching, not a full humanoid controller, not learned
control, and not video-conditioned RL.

Same-Episode Hold/Contact Step:

User correction on 2026-07-06: do not block on downloaded models, optional
policy wrappers, or external checkpoints when they do not directly enter
rollout. Continue direct Isaac scene construction.

The support-adaptation bridge has already passed: retry3
`20260706_direct_carry_online_probe_adaptive_support_retry3_64cm_8kg` ran two
single-episode 64 cm / 8 kg cases with observed probe telemetry only. The
vertical probe selected `compact_medium_double_support`; the horizontal probe
selected `nominal_reach_support`; both completed with fall/drop 0 and strict
no-shortcut/support gates. This remains support-profile adaptation, not
hold-contact adaptation.

The first online hold/contact attempts with X-cradle and side-clamp are
negative. X-cradle launched the payload; side-clamp repeatedly commanded
closure but the clamp joints did not physically track. Retry6 issued thousands
of DriveAPI target-position updates and still measured only `3.83e-05 m` clamp
motion for a `0.054 m` commanded closure. Stop using the current side-clamp
formulation as the active route.

Next direct Isaac gate:

- keep `payload_mode=cradle_free_box`;
- prebuild an optional top-lid contact body;
- enable that extra contact inside the same episode after probing only for
  non-low risk buckets;
- verify that the expected contact-collision state changes are recorded;
- require fall/drop 0, no root/box pose shortcuts, no hidden-ground-truth
  probe belief, and the existing target/support/tilt gates.

This is a scaffold contact-redistribution diagnostic. It is the correct next
step for the Isaac scene, but it still must not be claimed as full humanoid
carrying, RL, or video-conditioned control.

Retry2 result:

`20260706_direct_carry_online_probe_adaptive_hold_adaptive_cradle_retry2_64cm_8kg`
passed on `server46` as Slurm job `167479`. It is the first current
same-episode scaffold run where observed active-probe telemetry changes both
the support profile and the contact/collision configuration before the carry
phase.

- `vertical_probe`: risk `0.5932174593481317`, bucket `medium`, support
  profile `compact_medium_double_support`, hold profile
  `reinforced_contact_closure`, top-lid collision enabled, `3640` steps,
  fall/drop `0`, final box target distance `0.01677 m`.
- `horizontal_probe`: risk `0.4508505528966966`, bucket `low`, support
  profile `nominal_reach_support`, hold profile `light_contact_closure`,
  top-lid collision left disabled, `3640` steps, fall/drop `0`, final box
  target distance `0.00271 m`.

Both had root shortcut free, no fixed-world stance anchor, no hidden-ground-
truth probe use, and passed the suite checker. The next research step should
not be model downloading; it should make this same task contract more
robotic: either replace the support-foot scaffold with a real/controller-
backed locomotion backend, or make the contact switch more physically
defensible and evaluate it under harder object/posture variation.

Next hardening gate:

Before attempting another controller-backend swap, test whether the same
online probe-adaptive contact switch survives multiple carry postures. The new
script is:

```text
scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh
```

It keeps the same 64 cm / 8 kg `cradle_free_box` task and runs five cases:

- `vertical_micro_lift`: `front_mid`, `close_mid`, `chest_high`; expected
  medium bucket, `compact_medium_double_support`, `reinforced_contact_closure`,
  and adaptive top-lid collision enabled.
- `horizontal_push_pull`: `front_reach`, `low_front`; expected low bucket,
  `nominal_reach_support`, `light_contact_closure`, and adaptive top-lid
  collision disabled.

This is not the final objective, but it is aligned with the final objective:
it checks whether active probing and posture/contact adaptation remain stable
across several carry poses rather than only one easy pose.

Queue status:

The five-posture suite has not yet produced compute evidence. Job `167501`
was canceled after prolonged priority pending with a 3 hour time limit. Retry2
job `167502` used a 1 hour limit, but Slurm estimated start at
`2026-07-06T09:00:00`, so it was canceled before execution to avoid leaving an
unmonitored queued experiment. These are scheduling events, not simulation
results.

The script now also supports an optional walking-realism audit via
`MAX_NEAR_GROUND_FOOT_SPEED` and `MAX_NEAR_GROUND_FOOT_SLIP`. When resources
are available, first run the five-posture suite without slip gates to test
same-episode online contact adaptation across postures; then rerun or check
with slip gates to expose whether the current support-foot scaffold is still
sliding rather than walking. A slip failure should be treated as a valid
negative result and a reason to replace the support-foot backend, not as a
reason to weaken the final goal.

Retry3 scheduling status:

Job `167505` (`carry_hold_p5r3`) used a shorter 45 minute limit, but still
remained pending with reason `Priority` and no scheduled start time from
`squeue --start`. It was canceled before execution. This is still no compute
evidence.

Slip-audit command prepared:

```bash
SUITE_STAMP=20260706_direct_carry_online_probe_adaptive_hold_posture5_slip_audit_retry1_64cm_8kg \
srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=01:00:00 \
  --job-name=carry_hold_slip5 \
  bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_slip_audit_suite.sh
```

Interpretation rule: if the non-slip-gated suite passes but the slip audit
fails, the correct conclusion is that active-probe contact adaptation works in
the current scaffold but the support-foot backend still does not satisfy the
walking requirement.

Planted/no-slide audit:

The next stronger scaffold test is not another cradle geometry change. It is
to force the existing support-foot backend into a more walking-like contract:
stance feet keep their commanded joint targets fixed while rail propulsion
moves the body and swing feet reposition. The wrapper is:

```bash
SUITE_STAMP=20260706_direct_carry_online_probe_adaptive_hold_posture5_planted_slip_audit_retry1_64cm_8kg \
srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=01:00:00 \
  --job-name=carry_hold_plant5 \
  bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_planted_slip_audit_suite.sh
```

It requires `PLANTED_STANCE_RAIL_PROPULSION=1`,
`FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`, near-ground foot speed below
`0.80 m/s`, and near-ground slip below `0.20 m`. Passing would still be
scaffold evidence, but failing would be useful: it would show that the current
support-foot morphology/controller cannot meet the walking-realism gate and
should be replaced.

The posture-suite summarizer now carries the fields needed to audit this:
planted-stance propulsion enabled/steps, freeze-commanded stance-foot
enabled/count/switch count/active feet, per-foot near-ground speed/slip, and
aggregate maximum near-ground foot speed/slip. Future reports should cite
these fields directly rather than treating a fall/drop pass as walking.

## 2026-07-06 Direct G1 Low-Cradle Targeted-Creep Update

The current direct Isaac G1 path should proceed without waiting for external
models. Added these focused runners:

- `scripts/isaac/run_core_world_g1_stable_cradle_propulsion_tune.sh`
- `scripts/isaac/run_core_world_g1_targeted_creep_stop_tune.sh`
- `scripts/isaac/run_core_world_g1_low_cradle_creep_validation.sh`
- `scripts/isaac/run_core_world_g1_low_creep_terminal_hold_tune.sh`

Best valid diagnostic so far is `low_push032` from
`20260706_g1_targeted_creep_stop_tune1`: 560 steps, low/close cradle, free
dynamic box, `targeted_creep`, fall/drop `0`, rollout root/velocity/box writes
`0`, final box target-directed travel `0.164657 m`, max tilt `0.128766 rad`,
and final relative offset `0.071063 m`.

This is useful but not the target. The same configuration failed longer
validation: 700 steps produced `18` fall events and `9` box-drop events;
1000 steps produced `318` fall events and `274` box-drop events. Corrected
terminal-hold retry2 triggered at box-travel thresholds but still failed
700-step gates for all cases. The next implementation should add a real
deceleration and recovery phase for targeted creep, not a fixed terminal
posture and not more external model waiting.

Detailed report:

```text
experiments/reports/2026-07-06_g1_low_cradle_creep_diagnostics.md
```

## 2026-07-06 Decel And Brake Follow-Up

Added parameterized deceleration and braking to the direct G1
`targeted_creep` controller. The follow-up tested four families:

- travel-based creep decel,
- pitch-brake latch,
- positive-pitch-only brake latch,
- zero-offset travel-triggered stand hold.

The result is a sharper blocker, not success. Decel/hold/latched stop can
either keep the robot stable with too little distance or produce distance but
still fail late by forward pitch and box drop. The best stable 700-step case
was `decel014_024_brake012`: fall/drop `0`, max tilt `0.120622 rad`, final
relative offset `0.018245 m`, but only `0.086960 m` target-directed box
travel. It is a diagnostic, not carrying.

Next implementation should add a true recovery phase, not another simple
decel sweep:

```text
targeted creep -> trigger by target travel/pitch -> reverse-brake or
counter-step phase -> hold after pitch and box-relative drift recover
```

Detailed report:

```text
experiments/reports/2026-07-06_g1_creep_decel_and_brake_followup.md
```

## 2026-07-06 Reverse-Brake Result And Controller Pivot

The explicit reverse-brake and simple hold-balance follow-ups have now also
failed.

Reverse-brake output:

```text
experiments/outputs/core_world_g1_low_creep_reverse_brake_tune/20260706_g1_low_creep_reverse_brake_tune1/
```

Hold-balance output:

```text
experiments/outputs/core_world_g1_low_creep_hold_balance_tune/20260706_g1_low_creep_hold_balance_tune1/
```

Interpretation:

- Reverse-brake triggered by travel but did not recover the body/box state.
  It reached about `0.76-0.79 m` target-directed box travel while still
  producing `20-26` fall events and `4-9` drops.
- Hold-balance was either destructive under negative sign or stable but nearly
  stationary under positive sign.
- The repeated blocker is now the open-loop gait backend, not one missing
  scalar brake parameter.

Next execution path:

1. Use the local WBC-AGILE G1 policy already present under `external/`.
2. Verify AGILE inside the same direct Core scene with no box.
3. Add a fixed light torso payload.
4. Add the free low-cradle box.
5. Only after controller-backed walking works, reintroduce active probing and
   posture/contact adaptation.

Prepared runner:

```text
scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh
```

The first submitted command is:

```bash
SUITE_STAMP=20260706_g1_agile_low_cradle_suite1 DEVICE=cpu STRICT=0 \
AGILE_POLICY_BACKEND=onnx \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --job-name=g1_agile_carry \
  bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh
```

If ONNX runtime or policy I/O is unavailable, the next allowed fallback is the
official local torch-checkpoint backend. Do not replace this with a toy gait
or homemade locomotion policy.

## 2026-07-06 WBC-AGILE First Smoke Result

The first AGILE Core-scene suite ran, but did not yet establish a usable
controller-backed locomotion backend.

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_low_cradle_suite1/
```

Result:

- `agile_nobox_walk`: failed with `210` fall events, min robot z
  `0.185860 m`, max tilt `3.107505 rad`, and final target-directed robot
  travel `-0.120162 m`.
- `agile_fixed_payload_walk`: failed.
- `agile_low_cradle_freebox_walk`: failed with falls and box drops.

Interpretation:

The no-box failure is the relevant blocker. Do not interpret the payload and
free-box failures as evidence about carrying yet. The controller adapter must
first make the official WBC-AGILE policy stand/walk in the Core scene without
root/velocity rollout writes.

First adapter fix:

- pass nonzero `root_ang_vel_b` to AGILE by reading Core API root angular
  velocity and rotating it into the robot body frame;
- expose angular-velocity diagnostic fields in summary and checker output;
- use a no-box-only smoke with IsaacLab 29DoF drive gains before rerunning any
  box cases.

Prepared no-box runner:

```text
scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh
```

Pending/active command:

```bash
SUITE_STAMP=20260706_g1_agile_nobox_smoke_angvel1 DEVICE=cpu STRICT=0 \
AGILE_POLICY_BACKEND=onnx \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
  --job-name=g1_agile_nb \
  bash scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh
```

If the no-box smoke still fails, continue with policy observation/action
scaling, root/body frame conventions, actuator gains, or the official
torch-checkpoint backend. Do not run more box suites until no-box locomotion
is stable.

## 2026-07-06 AGILE Backend Usable Gate

The AGILE adapter now has a valid Core-scene locomotion setting:

- original G1 root orientation;
- ONNX WBC-AGILE backend;
- IsaacLab 29DoF drive gains;
- `root_ang_vel_b` read from Core API and rotated into body frame;
- target direction `[-1.2, 0.0]`;
- `cmd_x=0.10`.

Evidence:

```text
experiments/outputs/core_world_g1_agile_policy_nobox_smoke/20260706_g1_agile_nobox_smoke_targetnegx1/
```

`onnx_cmd010_isaaclab_gains` passed 320-step no-box locomotion with fall/drop
`0`, final robot target-directed travel `0.562249 m`, max tilt
`0.209202 rad`, and no rollout root/velocity/box writes.

Fixed-payload evidence:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_fixed_payload_nocoll_targetnegx1/
```

The collision-disabled fixed 0.25 kg inertial payload passed: fall/drop `0`,
final robot target-directed travel `0.358296 m`, final box target-directed
travel `0.371363 m`, max tilt `0.204425 rad`, and no rollout root/velocity/box
writes. The collision-enabled centered fixed payload was stable but too slow,
so centered collision geometry is a confound.

Current next gate:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_lowcradle_targetnegx1/
```

The first free dynamic low-cradle negative-X run failed without falling or
dropping: final relative offset `0.374029 m`, final robot target-directed
travel `-0.099899 m`, and final box target-directed travel `-0.305000 m`.
This was a contact-geometry/relative-drift failure, not a controller fall.

The close box/cradle retry passed:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_lowcradle_targetnegx1/
```

It completed `360` steps with fall/drop `0`, final robot target-directed
travel `0.125915 m`, final box target-directed travel `0.187173 m`, final
relative offset `0.081144 m`, max tilt `0.146167 rad`, and no rollout
root/velocity/box writes.

Interpretation:

This is the first current controller-backed direct Isaac G1 diagnostic where a
free dynamic box remains on a robot-mounted low cradle and moves targetward.
It is still short, light, and carefully positioned. The next gate is not a
claim; it is to extend this exact close-cradle setup to `700+` steps, then
vary mass/position, then add active probing and belief updates.

700-step update:

The close free-box extension failed:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_lowcradle_targetnegx1_700/
```

It had fall/drop `0`, but final robot target-directed travel was
`-0.691677 m`, final box target-directed travel was `-1.076183 m`, final
relative offset was `0.415493 m`, and max tilt was `0.632334 rad`.

The matching no-box 700-step baseline passed:

```text
experiments/outputs/core_world_g1_agile_policy_nobox_smoke/20260706_g1_agile_nobox_targetnegx1_700/
```

`cmd_x=0.10` had fall/drop `0`, final target-directed travel `0.878516 m`,
max tilt `0.209202 rad`, and no rollout root/velocity/box writes.

Current conclusion:

The AGILE locomotion backend is viable over 700 steps. The blocker is
long-horizon free-box retention on the cradle. The next implementation should
improve contact retention or add an explicit stop/hold phase after short
targetward motion before retesting 700+ steps. Do not jump to heavier objects
or unknown-load claims until this retention gate passes.

## 2026-07-06 AGILE Stop/Hold Retention Gate

Prepared implementation:

- AGILE command stop/hold is now a first-class Core-scene diagnostic path.
- Trigger options: absolute step, box target-directed travel, and robot
  target-directed travel.
- After trigger, policy inference continues but the command is multiplied by
  `--agile-command-hold-scale`, defaulting to zero.
- This is not a learned carrying policy and not a success claim; it is a
  retention-isolation test.

Immediate gate:

Run the close-cradle free dynamic box for 700 steps with target
`[-1.2, 0.0]`, `cmd_x=0.10`, box/cradle local X `-0.18`, and hold triggered
when box target-directed travel reaches `0.18 m`.

Decision:

- Pass: tune stop thresholds and then vary mass/position one axis at a time.
- Fail with hold active: redesign cradle/contact retention before adding
  active probing, unknown-load belief, or video-conditioned rewards.

First result:

The zero-command hold test failed. Hold triggered at step `117`, the final
command to AGILE was `[0, 0, 0]`, but the robot continued moving and failed
with falls/drops. Therefore the next isolation test is not another threshold
sweep; it is to reset AGILE recurrent policy state at the hold trigger and see
whether the continued motion is caused by hidden-state persistence. If that
also fails, move to an explicit stand/settle transition or cradle redesign.

Reset-state result:

Resetting AGILE policy state at the hold trigger also failed, with more falls
and drops. Therefore hidden-state persistence is not the sufficient
explanation. The next and last AGILE-stop diagnostic should bypass the policy
after hold and blend toward the configured G1 stand targets. If that fails,
the plan should move away from AGILE command gating and toward cradle/contact
retention redesign.

Stand-target result:

The stand-target hold did bypass AGILE after the hold trigger and kept the box
from dropping, but the robot itself failed to settle and accumulated falls
with negative final target-directed travel. This closes the AGILE command-gate
branch for the current setup. The next plan step is to redesign cradle/contact
retention together with a settle posture or use a stable low-speed controller,
rather than rerunning zero-command, reset-state, or stand-target hold variants
unchanged.
