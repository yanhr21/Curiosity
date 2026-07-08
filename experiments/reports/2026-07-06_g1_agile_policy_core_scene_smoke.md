# 2026-07-06 G1 AGILE Policy Core-Scene Smoke

## Purpose

Test whether the local official WBC-AGILE G1 locomotion policy can replace the
hand-written open-loop `targeted_creep` gait inside the direct Isaac Core G1
box scene.

This is a controller-adapter diagnostic only. It is not video-conditioned RL
and not a carrying success claim.

## First Three-Stage Suite

Command:

```bash
SUITE_STAMP=20260706_g1_agile_low_cradle_suite1 DEVICE=cpu STRICT=0 \
COMPUTE_SIDE_STARTUP_SLEEP=20 AGILE_POLICY_BACKEND=onnx \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --job-name=g1_agile_carry \
  bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh
```

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_low_cradle_suite1/
```

Status:

```text
agile_nobox_walk              build 0  check 1
agile_fixed_payload_walk      build 0  check 1
agile_low_cradle_freebox_walk build 0  check 1
```

Key result:

- ONNX runtime and the local WBC-AGILE model loaded successfully.
- The no-box case already failed, so the fixed-payload and free-box failures
  should not be interpreted as box-carrying evidence.
- `agile_nobox_walk`: `420` steps, `210` fall events, min robot z
  `0.185860 m`, max tilt `3.107505 rad`, final target-directed robot travel
  `-0.120162 m`.
- `agile_fixed_payload_walk`: `250` fall events, max tilt `3.111951 rad`,
  final target-directed robot travel `-0.267491 m`.
- `agile_low_cradle_freebox_walk`: `333` fall events, `233` box drops, max
  tilt `3.074614 rad`, final robot target-directed travel `-0.450007 m`.

## Adapter Issue Found

The AGILE policy was being called with zero `root_ang_vel_b`, while the
official IsaacLab-Arena WBC policy path explicitly uses `root_ang_vel_b` in
the observation. This makes the first failure an adapter/observation issue
before it is a carrying issue.

Code update after the first suite:

- `build_core_world_g1_box_scene.py` now reads the Core API root angular
  velocity with `robot.get_angular_velocity()`, rotates it into the root body
  frame, and passes it to both ONNX and torch-checkpoint AGILE adapters.
- Summary/check JSON now expose `agile_root_ang_vel_source`,
  `agile_root_ang_vel_read_failures`,
  `agile_last_root_ang_vel_read_error`, and
  `max_agile_root_ang_vel_norm`.
- Added `scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh` for
  short no-box-only diagnostics using IsaacLab 29DoF drive gains.

## Next Test

Submitted no-box-only smoke:

```bash
SUITE_STAMP=20260706_g1_agile_nobox_smoke_angvel1 DEVICE=cpu STRICT=0 \
COMPUTE_SIDE_STARTUP_SLEEP=20 AGILE_POLICY_BACKEND=onnx \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
  --job-name=g1_agile_nb \
  bash scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh
```

Required interpretation:

- If no-box still fails, continue fixing AGILE/Core observation and actuator
  adaptation before running any box-carrying cases.
- If no-box passes, then rerun fixed payload and free low-cradle box.

## No-Box Smoke After Angular-Velocity Fix

Output:

```text
experiments/outputs/core_world_g1_agile_policy_nobox_smoke/20260706_g1_agile_nobox_smoke_angvel1/
```

Status:

```text
onnx_cmd010_isaaclab_gains build 0 check 1
onnx_cmd005_isaaclab_gains build 0 check 1
```

Result:

- `onnx_cmd010_isaaclab_gains` is a useful partial pass: it had fall/drop
  `0`, min robot z `0.750114 m`, max tilt `0.209202 rad`, no root/velocity
  rollout writes, policy inference count `70`, and max root-angular-velocity
  norm `6.825381`. It failed only because final target-directed travel was
  `-0.562264 m`, meaning the robot walked stably in the opposite X direction.
- `onnx_cmd005_isaaclab_gains` was worse: `18` fall events, min robot z
  `0.220899 m`, max tilt `0.976502 rad`, final target-directed travel
  `-0.390616 m`.

Interpretation:

The angular-velocity fix and IsaacLab 29DoF drive gains materially improved
the adapter. The `0.10` command case is stable but has an X-direction sign
convention mismatch. Do not run box cases yet. First verify negative command
sign in no-box:

```bash
SUITE_STAMP=20260706_g1_agile_nobox_smoke_negcmd1 DEVICE=cpu STRICT=0 \
COMPUTE_SIDE_STARTUP_SLEEP=20 AGILE_POLICY_BACKEND=onnx \
INCLUDE_POSITIVE_COMMANDS=0 INCLUDE_NEGATIVE_COMMANDS=1 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
  --job-name=g1_agile_neg \
  bash scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh
```

## Negative Command Smoke

Output:

```text
experiments/outputs/core_world_g1_agile_policy_nobox_smoke/20260706_g1_agile_nobox_smoke_negcmd1/
```

Status:

```text
onnx_cmdneg010_isaaclab_gains build 0 check 1
onnx_cmdneg005_isaaclab_gains build 0 check 1
```

Result:

- `onnx_cmdneg010_isaaclab_gains` also walked stably in negative world X:
  fall/drop `0`, min robot z `0.740933 m`, max tilt `0.213786 rad`, final
  target-directed robot travel `-0.941146 m`.
- `onnx_cmdneg005_isaaclab_gains` failed late with `11` fall events and final
  target-directed travel `-0.735462 m`.

Interpretation:

Changing the command sign does not change the world travel direction in this
Core setup. The likely mismatch is the robot/USD forward direction relative to
the fixed target direction, not simply the sign of the velocity command. Next
test: yaw the G1 root by 180 degrees at reset and rerun positive-command
no-box smoke.

## Yaw-180 Smoke

Output:

```text
experiments/outputs/core_world_g1_agile_policy_nobox_smoke/20260706_g1_agile_nobox_smoke_yaw180_1/
```

Status:

```text
onnx_cmd010_isaaclab_gains build 0 check 1
onnx_cmd005_isaaclab_gains build 0 check 1
```

Result:

- `onnx_cmd010_isaaclab_gains`: `170` fall events, min robot z
  `0.180216 m`, max tilt `1.715628 rad`, final target-directed travel
  `-0.754693 m`.
- `onnx_cmd005_isaaclab_gains`: `179` fall events, min robot z
  `0.060109 m`, max tilt `3.122284 rad`, final target-directed travel
  `-0.580300 m`.

Interpretation:

Yawing the root by 180 degrees breaks the policy/Core setup and is not the
right fix. The stable useful no-box behavior remains the original root
orientation with `cmd=0.10`, which walks in world negative X. The next fix is
to parameterize the target direction and put the task target in negative X for
AGILE-backed tests.

## Negative-X Target No-Box Smoke

Code update:

- `build_core_world_g1_box_scene.py` now accepts `--target-xy`.
- The no-box and AGILE suite runners forward `TARGET_X/TARGET_Y`.

Output:

```text
experiments/outputs/core_world_g1_agile_policy_nobox_smoke/20260706_g1_agile_nobox_smoke_targetnegx1/
```

Status:

```text
onnx_cmd010_isaaclab_gains build 0 check 0
onnx_cmd005_isaaclab_gains build 0 check 1
```

Result:

- `onnx_cmd010_isaaclab_gains` passed the no-box gate with target
  `[-1.2, 0.0]`: fall/drop `0`, min robot z `0.750114 m`, max tilt
  `0.209202 rad`, final robot target-directed travel `0.562249 m`, max robot
  target-directed travel `0.569453 m`, no root/velocity/box rollout writes.
- `onnx_cmd005_isaaclab_gains` still failed late with `18` fall events and
  max tilt `0.976502 rad`, although target-directed travel was positive
  (`0.390582 m`).

Interpretation:

The first controller-backed no-box locomotion gate is now valid for the Core
scene: AGILE ONNX, original root orientation, IsaacLab 29DoF gains,
`cmd_x=0.10`, and target in negative X. Future AGILE-backed payload/carry
tests should use this direction unless the box/cradle geometry is explicitly
reoriented.

## Fixed Light Payload Smoke

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_fixed_payload_targetnegx1/
```

Status:

```text
agile_fixed_payload_walk build 0 check 1
```

Result:

- The fixed 0.25 kg torso payload did not make the robot fall: fall/drop `0`,
  min robot z `0.757255 m`, max tilt `0.186292 rad`, final relative offset
  `0.016612 m`.
- It failed the planned distance gate: final robot target-directed travel was
  `0.115128 m`, final box target-directed travel was `0.118508 m`, below the
  `0.30 m` diagnostic gate.

Interpretation:

This is stable loaded locomotion but too little progress. Because the payload
was fixed at the torso center with collision enabled, the next diagnostic is
to disable box collision for the fixed inertial payload. That tests whether
the slowdown is from mass/load or from box-body collision geometry.

## Fixed Light Payload Without Box Collision

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_fixed_payload_nocoll_targetnegx1/
```

Status:

```text
agile_fixed_payload_walk build 0 check 0
```

Result:

- Passed the fixed-payload diagnostic with collision disabled but mass kept:
  fall/drop `0`, min robot z `0.758894 m`, max tilt `0.204425 rad`, final
  robot target-directed travel `0.358296 m`, final box target-directed travel
  `0.371363 m`, final relative offset `0.014385 m`, no root/velocity/box
  rollout writes.

Interpretation:

The AGILE Core-scene backend can now pass no-box locomotion and a light fixed
inertial payload gate. The collision-enabled fixed payload was slowed by
geometry interaction, not just by payload mass. The next valid test is the
free dynamic low-cradle box with target and cradle geometry moved to negative
X, still diagnostic-only.

## Free Dynamic Low-Cradle Box, Negative X

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_lowcradle_targetnegx1/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- The robot did not fall and the box did not drop: fall/drop `0`, min robot z
  `0.714820 m`, min box z `0.864940 m`, max tilt `0.344751 rad`.
- It failed the carry gate: final robot target-directed travel was
  `-0.099899 m`, final box target-directed travel was `-0.305000 m`, final
  relative offset was `0.374029 m`.
- The box briefly reached max target-directed travel `0.227961 m`, but it did
  not remain coupled to the robot.

Interpretation:

This is a contact-geometry failure, not a locomotion fall. The free box is not
staying in a useful support region on the low cradle. Next diagnostic moves
the box and cradle closer to the torso in negative X to reduce the forward
lever arm and relative drift.

## Free Dynamic Close Low-Cradle Box, Negative X

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_lowcradle_targetnegx1/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 0
```

Result:

- Passed the short free-dynamic-box diagnostic: `360` steps, fall/drop `0`,
  min robot z `0.757350 m`, min box z `0.878051 m`, max tilt `0.146167 rad`,
  final robot target-directed travel `0.125915 m`, final box target-directed
  travel `0.187173 m`, final relative offset `0.081144 m`, no root/velocity
  or box rollout writes.
- Max box target-directed travel reached `0.244799 m`; max relative offset
  was `0.212882 m`.

Interpretation:

This is the first current controller-backed direct Isaac G1 diagnostic where a
free dynamic box remains on a robot-mounted low cradle and moves targetward
without root/velocity/box rollout shortcuts. It is still a short, light-box,
carefully positioned cradle diagnostic. It does not include active probing,
unknown load inference, posture selection, video conditioning, or long
duration carrying.

Next gates should increase one difficulty at a time:

1. Extend the close low-cradle free-box run to 700+ steps.
2. Vary box mass and position while keeping the negative-X AGILE target.
3. Add probing and online belief updates before claiming unknown-load
   carrying.

## Free Dynamic Close Low-Cradle Box, 700 Steps

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_lowcradle_targetnegx1_700/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- The run completed `700` steps with fall/drop `0`, but failed the long-run
  carry gate.
- Final robot target-directed travel was `-0.691677 m`, final box
  target-directed travel was `-1.076183 m`, final relative offset was
  `0.415493 m`, and max tilt reached `0.632334 rad`.
- Max box target-directed travel only reached `0.244799 m`, then the coupled
  system drifted back and away from the target.

Interpretation:

The 360-step free-box result is a real short diagnostic, but it does not
extend to 700 steps. The next question is whether this is caused by free-box
contact geometry or by the AGILE no-box backend itself drifting over longer
horizons. A 700-step no-box baseline with the same negative-X target was
submitted next.

## No-Box 700-Step Baseline

Output:

```text
experiments/outputs/core_world_g1_agile_policy_nobox_smoke/20260706_g1_agile_nobox_targetnegx1_700/
```

Status:

```text
onnx_cmd010_isaaclab_gains build 0 check 0
onnx_cmd005_isaaclab_gains build 0 check 1
```

Result:

- `cmd_x=0.10` passed the 700-step no-box baseline: fall/drop `0`, min robot
  z `0.735248 m`, max tilt `0.209202 rad`, final target-directed travel
  `0.878516 m`, max target-directed travel `0.879230 m`, no rollout
  root/velocity/box writes.
- `cmd_x=0.05` failed with `398` fall events, confirming the lower command is
  not the stable setting in this Core adapter.

Interpretation:

The AGILE no-box backend is not the cause of the 700-step free-box failure.
The blocker is long-horizon coupling between the free dynamic box and the
robot-mounted cradle/contact geometry. The next implementation should improve
box retention or add a policy-level stop/hold phase after short targetward
progress, then retest the free box at 700+ steps.

## AGILE Command Stop/Hold Gate Prepared

Code update:

- `build_core_world_g1_box_scene.py` now supports AGILE-specific command
  gating:
  `--agile-command-stop-step`,
  `--agile-command-stop-box-target-travel`,
  `--agile-command-stop-robot-target-travel`, and
  `--agile-command-hold-scale`.
- The hold gate is latched. After the first trigger, the AGILE policy still
  runs, but the velocity command is scaled by `hold_scale` instead of
  continuing the walking command. The default hold scale is `0.0`.
- Summary/check output now records whether AGILE command hold was active,
  the first active step/reason, active-step count, and the last command passed
  to the policy.
- `run_core_world_g1_agile_policy_low_cradle_suite.sh` forwards these options
  through environment variables so the close free-box 700-step run can be
  retested without changing code.

Planned diagnostic:

```bash
SUITE_STAMP=20260706_g1_agile_free_close_hold_targetnegx1_700 \
DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx \
RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 \
TARGET_X=-1.2 TARGET_Y=0.0 \
FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 \
FREE_MIN_ROBOT_TRAVEL=0.05 FREE_MIN_BOX_TRAVEL=0.05 \
FREE_MAX_TILT=0.95 FREE_MAX_FINAL_REL=0.20 \
AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 \
AGILE_COMMAND_HOLD_SCALE=0.0 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 \
  --job-name=g1_agile_hold \
  bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh
```

Interpretation rule:

- If this passes the 700-step gate, the previous failure was likely caused by
  continuing to walk after short targetward progress; next test should vary
  stop thresholds and then box mass/position one axis at a time.
- If this still drifts or fails while `agile_command_hold_active=true`, the
  contact/cradle retention must be improved directly before adding unknown
  load or video-conditioned components.

## AGILE Zero-Command Hold Result

Command was launched in tmux session
`curiosity_g1_agile_hold_0706`, Slurm job `167583`, job-name
`g1_agile_hold`, on `server02`.

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_targetnegx1_700/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Hold triggered at step `117` from `box_target_travel`.
- `agile_command_hold_active=true`, active for `583` steps.
- The last command passed to AGILE was `[0.0, 0.0, 0.0]`.
- The run still failed: `87` fall events, `70` box-drop events, min robot z
  `0.194465 m`, min box z `0.042331 m`, max tilt `1.238096 rad`, final
  relative offset `0.532693 m`.
- Final robot target-directed travel was `1.762137 m`; final box
  target-directed travel was `2.057596 m`.

Interpretation:

Zeroing the command after short progress does not make the current AGILE/Core
adapter stop and hold. It continues moving long after the hold trigger. This
could come from recurrent policy state, command/heading semantics, or the
policy not being trained as a reliable stand-still controller in this Core
setup. The next diagnostic is to reset the AGILE recurrent state at the hold
trigger while still passing a zero command.

Prepared next option:

- `--agile-command-hold-reset-policy-state`
- runner env: `AGILE_COMMAND_HOLD_RESET_POLICY_STATE=1`

## AGILE Hold With Policy-State Reset Result

Command was launched in tmux session
`curiosity_g1_agile_hold_reset_0706`, Slurm job `167588`, job-name
`g1_agile_hres`, on `server02`.

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_reset_targetnegx1_700/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Hold again triggered at step `117` from `box_target_travel`.
- Policy state reset was applied once:
  `agile_command_hold_policy_state_reset_count=1`,
  `agile_command_hold_last_policy_state_reset_error=null`.
- The final command passed to AGILE was `[0.0, 0.0, 0.0]`.
- The run failed more severely than zero-command-only hold: `306` fall events,
  `252` box-drop events, min robot z `0.110636 m`, min box z `0.040000 m`,
  max tilt `1.803860 rad`, final relative offset `0.961335 m`.
- Final robot target-directed travel was `0.651390 m`; final box
  target-directed travel was `1.522979 m`.

Interpretation:

The continued/unstable post-hold behavior is not explained by recurrent state
persistence alone. Resetting the policy state at the hold trigger does not
create a reliable stop/hold transition in this Core-scene adapter. The next
diagnostic should bypass AGILE after hold and blend toward explicit G1 stand
joint targets, or else redesign cradle/contact retention directly.

## AGILE Stand-Target Hold Prepared

Code update:

- Added `--agile-command-hold-mode {policy_command,stand_targets}`.
- Added `--agile-command-hold-stand-blend-rate`.
- In `stand_targets` mode, once hold is active the scene bypasses AGILE policy
  inference and blends commanded G1 joint targets toward the configured stand
  pose. This is a stop/settle diagnostic only.
- Runner envs: `AGILE_COMMAND_HOLD_MODE=stand_targets` and
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE`.

Planned diagnostic:

```bash
SUITE_STAMP=20260706_g1_agile_free_close_hold_stand_targetnegx1_700 \
DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx \
RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 \
TARGET_X=-1.2 TARGET_Y=0.0 \
FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 \
FREE_MIN_ROBOT_TRAVEL=0.05 FREE_MIN_BOX_TRAVEL=0.05 \
FREE_MAX_TILT=0.95 FREE_MAX_FINAL_REL=0.20 \
AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 \
AGILE_COMMAND_HOLD_SCALE=0.0 \
AGILE_COMMAND_HOLD_MODE=stand_targets \
AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.025 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 \
  --job-name=g1_agile_hstd \
  bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh
```

Submitted state:

- tmux: `curiosity_g1_agile_hold_stand_0706`
- Slurm job: `167590`
- job-name: `g1_agile_hstd`
- final status: ran and failed checker.

## AGILE Stand-Target Hold Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_stand_targetnegx1_700/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Hold triggered at step `117` from `box_target_travel`.
- `agile_command_hold_mode=stand_targets`.
- `agile_command_hold_stand_target_active_steps=583`.
- `policy_inference_count=20`, confirming that after hold the policy was
  bypassed rather than continuing AGILE inference.
- Box retention improved relative to the previous two hold tests:
  `box_drop_events=0`, min box z `0.467851 m`, final relative offset
  `0.272191 m`.
- Robot stability failed: `285` fall events, min robot z `0.322664 m`, max
  tilt `1.419840 rad`.
- Final target-directed travel went negative: robot `-0.865188 m`, box
  `-1.126525 m`, although max target-directed box travel reached
  `0.193488 m`.
- No rollout root/velocity/box pose writes were used.

Interpretation:

The explicit stand-target hold confirms that the box can remain physically
high on the cradle longer, but the robot cannot settle into a stable standing
posture after the short AGILE movement. The project should stop sweeping
AGILE command/hidden-state gates for this setup. The next productive path is
cradle/contact retention plus a stable low-speed controller or a redesigned
settle posture, not more zero-command AGILE variants.

## Hold-Only Low-Crouch Settle Prepared

Code update:

- Added hold-only settle posture overrides. These do not change the AGILE
  walking policy's default stand pose; they only change the target used after
  `stand_targets` hold is active.
- New CLI/env controls cover paired hip pitch, knee, ankle pitch, hip roll,
  ankle roll, and waist pitch.
- Summary/check output records requested and applied hold-only joint targets.

Planned diagnostic:

```bash
SUITE_STAMP=20260706_g1_agile_free_close_hold_lowcrouch_targetnegx1_700 \
DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx \
RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 \
TARGET_X=-1.2 TARGET_Y=0.0 \
FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 \
FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 \
FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.25 \
AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.16 \
AGILE_COMMAND_HOLD_SCALE=0.0 \
AGILE_COMMAND_HOLD_MODE=stand_targets \
AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.012 \
AGILE_HOLD_STAND_HIP_PITCH=-0.24 \
AGILE_HOLD_STAND_KNEE=0.58 \
AGILE_HOLD_STAND_ANKLE_PITCH=-0.34 \
AGILE_HOLD_STAND_WAIST_PITCH=-0.06 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 \
  --job-name=g1_agile_lc \
  bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh
```

Submitted state:

- tmux: `curiosity_g1_agile_lowcrouch_0706`
- Slurm job: `167591`
- job-name: `g1_agile_lc`
- final status: ran and failed checker.

## Hold-Only Low-Crouch Settle Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lowcrouch_targetnegx1_700/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Hold triggered earlier at step `101` from `box_target_travel=0.16`.
- Hold-only low-crouch targets were applied:
  hip pitch `-0.24`, knee `0.58`, ankle pitch `-0.34`, waist pitch `-0.06`.
- `stand_targets` hold was active for `599` steps and policy inference count
  was only `16`, confirming the low-crouch settle target bypassed AGILE after
  hold.
- Unlike the previous stand-target hold, target-directed travel stayed
  positive: final robot `0.752768 m`, final box `0.916188 m`.
- The run still failed badly: `378` fall events, `355` box drops, min robot z
  `0.286143 m`, min box z `0.030000 m`, max tilt `1.283639 rad`, final
  relative offset `0.545589 m`.

Interpretation:

The low-crouch target improves directionality but does not solve stability or
box retention. The next step should change the physical cradle/contact
geometry, not keep sweeping AGILE hold command, hidden-state, or settle-pose
variants unchanged.

## Top-Lid Cradle Retention Prepared

Code update:

- Added optional physical cradle top lid to `front_tray`:
  `--cradle-top-lid`, `--cradle-top-lid-local-z`,
  `--cradle-top-lid-thickness`, `--cradle-top-lid-x-scale`, and
  `--cradle-top-lid-y-scale`.
- Runner can now vary side rail height, end-stop height, rail thickness,
  mass scale, and lid geometry through environment variables.
- Summary/check output records top-lid status and geometry.

Submitted diagnostic:

- tmux: `curiosity_g1_agile_lid_lowcrouch_0706`
- Slurm job: `167592`
- job-name: `g1_agile_lid`
- stamp:
  `20260706_g1_agile_free_close_hold_lid_lowcrouch_targetnegx1_700`
- status at submission-time checks: pending, reason `Priority`, no
  `squeue --start` estimate yet.

## Static Top-Lid Low-Crouch Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_lowcrouch_targetnegx1_700/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Top lid was enabled from scene start:
  `cradle_piece_count=6`, `cradle_top_lid_enabled=true`.
- Hold triggered at step `103`; low-crouch stand-target hold was active for
  `597` steps; policy inference count was `16`.
- Compared with low-crouch without top lid, drops and falls decreased
  (`205` drops vs. `355`, `262` falls vs. `378`), and final relative offset
  improved (`0.311611 m` vs. `0.545589 m`).
- It still failed: min robot z `0.174101 m`, min box z `0.072546 m`, max
  tilt `1.542573 rad`.
- Final target-directed travel became negative: robot `-0.712567 m`, box
  `-0.815065 m`; max box target-directed travel was only `0.180485 m`.

Interpretation:

A physical top lid helps retention but degrades the overall motion and does
not solve balance. The next diagnostic should activate the lid only after the
hold trigger, so early AGILE movement is less constrained but the box gets
captured during settle.

## Hold-Phase Top-Lid Runtime-Schema Failure

First hold-phase top-lid run:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_onhold_lowcrouch_targetnegx1_700/
```

This is invalid as carrying evidence. The hold trigger fired at step `85`, the
lid collision update count was `1`, and the robot/box were still stable. But
the implementation applied `UsdPhysics.CollisionAPI` during rollout, which
invalidated the PhysX tensor view and stopped at `completed_steps=85` with:

```text
Failed to get DOF position targets from backend
```

Fix:

- Apply `CollisionAPI` to the lid at scene construction.
- Set `physics:collisionEnabled=false` initially when
  `--cradle-top-lid-enable-on-hold` is used.
- At hold trigger, only set the existing `physics:collisionEnabled` attr to
  true.

Rerun stamp:

```text
20260706_g1_agile_free_close_hold_lid_onhold_attrfix_targetnegx1_700
```

## Hold-Phase Top-Lid Attr-Fix Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_onhold_attrfix_targetnegx1_700/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- The attr-toggle fix worked: the run completed `700` steps and had no tensor
  invalidation.
- Hold triggered at step `92`; top-lid collision was enabled at step `92`;
  update count `1`, update error `null`.
- Box retention improved strongly: `box_drop_events=0`, min box z
  `0.498752 m`, final relative offset `0.268659 m`.
- Robot stability still failed: `95` fall events, min robot z `0.342325 m`,
  max tilt `1.245552 rad`.
- Final target-directed travel was negative: robot `-0.814386 m`, box
  `-1.081175 m`; max box target-directed travel only reached `0.194369 m`.

Interpretation:

Hold-phase top-lid capture solves the box-drop part of the previous
low-crouch failures, but the stand-target settle still drives the robot into
an unstable pitched posture and negative final travel. Next diagnostic should
keep the hold-phase top lid but use AGILE policy-command hold instead of
stand-target hold, to separate contact retention from the explicit settle
posture.

## Hold-Phase Top-Lid Policy-Command Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_policycmd_targetnegx1_700/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Hold mode was `policy_command`: after the trigger, AGILE inference continued
  with zero command rather than switching to explicit stand targets.
- Hold triggered at step `102`; top-lid collision was enabled at step `102`;
  lid update count `1`, update error `null`.
- Policy inference count was `165`.
- Final target-directed travel stayed positive: robot `0.962488 m`, box
  `0.835041 m`; max target-directed travel was robot `1.182925 m`, box
  `0.973236 m`.
- It still failed stability and retention: `352` fall events,
  `68` box drops, min robot z `0.191603 m`, min box z `0.085619 m`, max tilt
  `3.121745 rad`, final relative offset `0.415014 m`.
- No root, velocity, or box rollout writes were used.

Interpretation:

The two hold-phase top-lid branches expose the current blocker clearly:
`stand_targets` plus lid preserves the box but loses balance and reverses
travel, while `policy_command` plus lid preserves positive travel but still
falls and drops. The next implementation should add a stable transition or
balance-aware settle controller inside the Isaac/G1 scene, not wait for
external video/model code and not repeat unchanged hold-mode sweeps.

## Hybrid Hold + Hold-Gated Balance Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_hybrid_balance_targetnegx1_700/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Mode was `policy_then_stand`: hold triggered at step `102`, AGILE stayed in
  zero-command policy mode for `80` steps, then blended toward hold stand
  targets with rate `0.006`.
- Hold-phase top lid enabled correctly at step `102`; update count `1`, error
  `null`.
- Hold-gated balance feedback was active from step `102` for `592` steps,
  using `balance_feedback_base=command`.
- The run remained stable through roughly step `390`, then pitched forward and
  failed: `309` falls, `293` drops, min robot z `0.246960 m`, min box z
  `0.097018 m`, max tilt `1.224603 rad`.
- Target-directed travel was good despite failure: robot `1.125550 m`, box
  `1.155785 m`, max robot `1.143704 m`, max box `1.161471 m`.
- Final relative offset was `0.329119 m`, just above the `0.30 m` check bound.

Interpretation:

The hybrid transition fixed the negative-travel issue and delayed failure, but
the pitch collapse after settle shows the balance feedback direction or gain is
not yet correct for this Isaac/G1 coordinate convention. Next diagnostic should
flip the pitch feedback sign while keeping the same scene, lid, and hybrid
timing.

## Hybrid Balance Pitch-Sign Flip Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_hybrid_balance_pitchsignpos_targetnegx1_700/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Only the pitch feedback sign changed to `BALANCE_PITCH_SIGN=1.0`; scene,
  top-lid timing, hybrid delay, and settle target were kept fixed.
- Pitch improved strongly: max absolute pitch dropped from `1.224603 rad` to
  `0.181384 rad`.
- Failure mode moved from forward pitch collapse to side roll: max roll
  `1.570825 rad`, final roll `-1.499206 rad`.
- First fall occurred at step `560`; first drop occurred at step `600`.
- Total failures decreased but still failed: `148` falls, `108` drops, min
  robot z `0.167102 m`, min box z `0.081154 m`.
- Final relative offset was within the previous bound at `0.279529 m`, but
  target-directed travel collapsed because the robot drifted sideways: final
  robot target-directed travel `0.043994 m`, box `0.105884 m`.

Interpretation:

The pitch sign was wrong in the previous hybrid/balance run. With pitch fixed,
the next blocker is lateral roll drift during the hold/settle phase. The next
diagnostic should keep `BALANCE_PITCH_SIGN=1.0` but disable roll feedback
first, to test whether roll feedback itself is injecting the side fall.

## Hybrid Balance Pitch-Only Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_hybrid_balance_pitchonly_targetnegx1_700/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Pitch feedback kept `BALANCE_PITCH_SIGN=1.0`; roll feedback was disabled
  with zero roll gains and high roll activation thresholds.
- This failed earlier than the pitch-sign-flip run with roll feedback:
  first fall at step `290`, first drop at step `310`.
- Failure returned to pitch collapse: max pitch `1.345647 rad`, final pitch
  `-1.284981 rad`; max roll stayed small at `0.152900 rad`.
- Total failures were `418` falls and `396` drops; min robot z `0.281595 m`,
  min box z `0.093012 m`.
- Target-directed travel stayed positive, robot `0.924829 m` and box
  `0.937723 m`, but this is not usable carrying evidence because the robot
  had already fallen.

Interpretation:

Roll feedback is not merely harmful noise: disabling it makes the pitch failure
return much earlier. The current hard-coded roll feedback is nevertheless too
coarse and can inject side drift. The next change should make roll correction
more explicitly configurable instead of continuing blind sweeps of one fixed
formula.

## Hybrid Balance Mirrored-Roll Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_hybrid_balance_mirrorroll_targetnegx1_700/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Added configurable roll-feedback left/right multipliers, then tested mirrored
  roll correction:
  left/right ankle scales `1.0/-1.0`, left/right hip scales `-0.5/0.5`.
- First fall was step `510`; first drop was step `530`.
- Mirrored roll avoided the severe side-roll failure: max roll
  `0.403314 rad`, final roll approximately `0`.
- Failure returned to pitch collapse: max pitch `1.335714 rad`, final pitch
  `-1.304736 rad`.
- Total failures: `195` falls, `173` drops, min robot z `0.239188 m`, min box
  z `0.077527 m`, final relative offset `0.350005 m`.
- Target-directed travel stayed positive: robot `1.006663 m`, box
  `1.019586 m`.

Interpretation:

The repeated hold/balance diagnostics show a real controller gap: simple joint
target blending plus ankle/hip PD feedback can either preserve forward travel,
control pitch, or control roll, but not all three with the current fixed-foot
settle. The next change should not be another gain sweep. It should add an
explicit post-capture stabilization mechanism, such as a lateral drift brake,
step/stance repositioning, or a proper whole-body controller interface for the
hold phase.

## Hybrid Hold-Rescue Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_hybrid_rescue_mirrorroll_targetnegx1_700/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Added and tested a latched hold-rescue state machine on top of the mirrored
  roll configuration.
- Rescue triggered correctly at step `433` from `forward_pitch`, with
  threshold `-0.30 rad`; rescue stayed active for `267` steps.
- Rescue target was more upright: hip pitch `0.03`, knee `0.28`, ankle pitch
  `-0.06`, waist pitch `0.08`; rescue blend rate `0.035`.
- It still failed checker: first fall step `500`, first drop step `530`,
  `208` falls, `173` drops, min robot z `0.169106 m`, min box z
  `0.086920 m`, max tilt `1.528025 rad`.
- The failure mode changed: pitch was eventually pulled back
  (`final_pitch_rad=0.050945`), but the robot rolled over
  (`final_roll_rad=-1.481067`, max roll `1.528025`).
- Final relative offset improved to `0.186977 m`, but this is not carrying
  evidence because the robot had fallen and the box had dropped.

Interpretation:

The rescue latch works mechanically, but the static-target rescue posture is
not enough; it turns a pitch fall into a roll fall. The next step should keep a
walking/balance policy active during the post-capture phase instead of
switching fully to static hold targets.

## Post-Capture Slow-Walk Policy Hold Result

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_policycmd_slowwalk_targetnegx1_700/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 0
```

Result:

- This is the first passing 700-step free-box low-cradle AGILE diagnostic in
  this sequence.
- Hold triggered at step `102` from box target travel.
- Hold mode was `policy_command`, not static stand target. The policy stayed
  active after capture with `AGILE_COMMAND_HOLD_SCALE=0.35`, so the final hold
  command was `[0.035, 0, 0]`.
- Hold-phase top lid enabled at step `102`; update count `1`, update error
  `null`.
- Balance feedback was hold-gated and active for `595` steps with
  `balance_feedback_base=command`, `BALANCE_PITCH_SIGN=1.0`, mirrored roll
  multipliers, and reduced gains.
- Checker passed: `fall_events=0`, `box_drop_events=0`, completed `700`
  steps, no root/velocity/box rollout writes.
- Stability margins: min robot z `0.758436 m`, min box z `0.879201 m`, max
  tilt `0.254946 rad`, max pitch `0.179165 rad`, max roll `0.254946 rad`.
- Carrying progress: final robot target-directed travel `1.182280 m`, final
  box target-directed travel `1.243254 m`.
- Contact retention: final relative offset `0.107421 m`, max relative offset
  `0.146449 m`.

Interpretation:

The main blocker was not the lack of external video/model code. It was the
post-capture controller: switching to static hold targets destabilized the
body, while keeping WBC-AGILE active with a reduced target-directed command
preserved gait and balance. This is a strong smoke pass, not a final success
claim; next validation must vary mass, shape, and longer duration before
claiming robust carrying.

## Heavier 0.5kg 900-Step Slow-Walk Validation

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_policycmd_slowwalk_mass0p5_900_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Same slow-walk policy-hold controller as the passing 0.25kg run, but box
  mass increased to `0.5 kg` and duration to `900` steps.
- Hold triggered later at step `123`; top lid enabled at step `123`.
- It failed: first fall step `420`, first drop step `530`, `485` falls and
  `373` drops.
- Max tilt `2.858130 rad`; max roll `2.858130 rad`; max pitch
  `1.556304 rad`.
- Final target-directed travel became negative after the fall: robot
  `-1.421516 m`, box `-1.452723 m`.

Interpretation:

The slow-walk policy hold is a real improvement for the base 0.25kg case, but
it is not yet robust to doubled mass and longer duration. The next validation
should test lower post-capture speed or an adaptive speed schedule for heavier
payloads.

## 0.5kg Lower-Speed Hold Validation

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_policycmd_mass0p5_hold015_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Same 0.5kg box as the heavier validation, shortened to `700` steps and with
  hold scale reduced to `0.15`, final hold command `[0.015, 0, 0]`.
- It failed earlier than the `0.35` hold-scale heavier run: first fall step
  `310`, first drop step `330`.
- Failures: `399` falls, `372` drops, min robot z `0.126606 m`, min box z
  `0.074592 m`, max tilt `1.141853 rad`.
- Final target-directed travel stayed positive after failure, robot
  `0.488001 m`, box `0.563155 m`, but this is not usable carrying evidence.

Interpretation:

For 0.5kg, simply lowering post-capture speed is worse. The next controller
should be load/contact-adaptive rather than a fixed slow command: it needs to
detect instability or load response online and choose gait speed/contact
strategy accordingly.

## Adaptive 0.5kg 700-Step Validation

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_adaptive_mass0p5_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 0
```

Result:

- Added adaptive post-capture command scaling and lateral correction, then
  tested the previously failing `0.5 kg` box for `700` steps.
- Hold triggered at step `123`; top lid enabled at step `123`.
- Adaptive scale stayed active for `577` steps. Observed scale range was
  `0.253006` to `0.35`; final risk was `0.394768`.
- Lateral correction was active for `577` steps; max lateral command hit the
  configured limit `0.035`; final lateral error was `-0.250461 m`.
- Checker passed: `fall_events=0`, `box_drop_events=0`, completed `700`
  steps, no root/velocity/box rollout writes.
- Stability: min robot z `0.710033 m`, min box z `0.835384 m`, max tilt
  `0.192797 rad`, max pitch `0.192797 rad`, max roll `0.188335 rad`.
- Carrying progress: final robot target-directed travel `1.812799 m`, final
  box target-directed travel `1.788483 m`.
- Contact retention: final relative offset `0.034773 m`, max relative offset
  `0.116523 m`.

Interpretation:

The adaptive command path is a substantial improvement over fixed hold scale:
the same `0.5 kg` payload that failed with fixed `0.35` and fixed `0.15` now
passes the 700-step free-box diagnostic. This is still not final project
success; next validation should vary box size/shape and duration.

## Larger-Box Adaptive 700-Step Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_free_close_hold_lid_adaptive_mass0p5_largerbox_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 0
```

Result:

- Tested the adaptive controller on a larger `0.5 kg` box
  (`0.14 x 0.10 x 0.08 m`) for `700` steps.
- Checker passed under the configured loose safety thresholds:
  `fall_events=0`, `box_drop_events=0`, completed `700` steps, and no
  root/velocity/box rollout writes.
- Robot and box made target-directed progress: final robot travel
  `1.954226 m`, final box travel `2.011707 m`.
- Contact retention stayed acceptable by the current relative-offset metric:
  final relative offset `0.071092 m`, max relative offset `0.206568 m`.
- However, robot root attitude and path quality are not yet stable enough for
  a strong claim: max/final root tilt was `0.479985 rad`, dominated by root
  roll, and the lateral correction saturated at `0.035` with final lateral
  path error `1.585361 m`.
- Adaptive scale was active for `603` steps, observed scale range
  `0.245685` to `0.35`, final risk `0.745661`.

Interpretation:

This is useful shape/size evidence but not a robust carrying result. The
controller can keep the larger box captured and avoid falling for 700 steps,
yet it lets the robot-box system drift laterally while the robot root holds a
large roll angle. The next implementation step should strengthen or redesign
target-path correction and add a stricter pass gate for robot root
attitude/path quality, rather than relying only on loose fall/drop thresholds.

## Larger-Box Yaw-Correction Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_adaptive_largerbox_yawcorr_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Added disabled-by-default hold-phase yaw correction driven by the same
  target-path lateral error as the lateral velocity correction, then tested
  the larger `0.5 kg`, `0.14 x 0.10 x 0.08 m` box under a stricter robot-root
  attitude gate: `FREE_MAX_TILT=0.35`, `FREE_MAX_FINAL_REL=0.25`.
- The run did not fall or drop the box: `fall_events=0`, `box_drop_events=0`,
  completed `700` steps, no root/velocity/box rollout writes.
- Checker failed only on robot-root attitude: `max_tilt_rad 0.532508 > 0.35`.
- Final robot-root attitude improved compared with the loose larger-box run
  (`final_roll_rad=0.226330`), but transient tilt became worse
  (`max_tilt_rad=0.532508`, max pitch `0.379108`, max roll `0.532508`).
- Forward progress degraded: final robot target-directed travel
  `0.638052 m`, final box target-directed travel `0.617244 m`.
- Contact stayed barely inside the strict relative-offset gate:
  final/max relative offset `0.201939 m`.
- Yaw correction was active for `603` steps, max yaw command `0.088684`, final
  yaw-control lateral error `1.059896 m`; lateral velocity correction still
  saturated at `0.035`.

Interpretation:

Yaw correction is not a clean fix. It reduces final root roll somewhat but
costs forward progress and does not control peak root tilt under the stricter
gate. The next direction should be root-attitude/contact strategy and direct
box-attitude instrumentation, not more target-path yaw gain sweeps: likely a
wider/more enclosing cradle posture, adaptive hold height/torso support, or
earlier slowdown based on measured box/root tilt before the larger box run
reaches high roll.

Note: after this diagnostic, the scene/checker was updated to record true box
roll/pitch/tilt separately as `max_box_tilt_rad`,
`max_abs_box_roll_rad`, `max_abs_box_pitch_rad`, `final_box_roll_rad`, and
`final_box_pitch_rad`. Historical `max_tilt_rad`, `final_roll_rad`, and
`final_pitch_rad` are robot-root attitude fields.

## Instrumentation Update

After the yaw-correction strict diagnostic, the scene/checker was also updated
to record target-line lateral path quality:

- `max_abs_robot_target_lateral_error_m`
- `max_abs_box_target_lateral_error_m`
- `final_robot_target_lateral_error_m`
- `final_box_target_lateral_error_m`

The runner now supports optional checker gates for these fields through
case-specific environment variables such as `FREE_MAX_ROBOT_LATERAL_ERROR`,
`FREE_MAX_BOX_LATERAL_ERROR`, `FREE_MAX_FINAL_ROBOT_LATERAL_ERROR`, and
`FREE_MAX_FINAL_BOX_LATERAL_ERROR`. Future larger-box diagnostics should use
these gates so that sideways drift cannot pass merely because target-directed
distance is positive.

## Larger-Box Box-Tilt And Chest-Pad Strict Results

Summary:

```text
experiments/reports/2026-07-06_g1_largerbox_strict_summary.json
```

Results:

- `20260706_g1_agile_adaptive_largerbox_boxtilt_strict_700_targetnegx1`
  failed badly despite build success. It had `fall_events=210`, min robot z
  `0.259627 m`, robot-root max tilt `1.824652 rad`, true box max tilt
  `1.650226 rad`, final box target-directed travel `-0.112571 m`, final
  relative offset `0.388508 m`, and final robot/box target-line lateral errors
  about `-1.51 m` / `-1.52 m`. No rollout root/velocity/box pose writes were
  used. Interpretation: true-box-tilt adaptive without torso support is not a
  viable larger-box posture.
- `20260706_g1_agile_largerbox_chestpad_strict_700_targetnegx1` failed the
  strict checker but is a real improvement. It had `fall_events=0`,
  `box_drop_events=0`, no rollout root/velocity/box pose writes, final robot
  travel `1.292595 m`, final box travel `1.285295 m`, final relative offset
  `0.157688 m`, and max relative offset `0.164649 m`. The remaining failures
  were robot-root max tilt `0.480753 > 0.35`, true box max tilt
  `0.493889 > 0.45`, and final box target-line lateral error
  `0.627694 > 0.60`.

Interpretation:

The torso/chest support posture is the better current direction. It preserves
contact and transport without falls or shortcut writes, but it still needs a
more conservative post-capture command and stronger lateral correction to pass
the strict attitude/path gates.

## Tuned Chest-Pad Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_tuned_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- This run kept chest-pad support but used earlier hold, lower post-capture
  command scale (`0.05-0.25` adaptive range), earlier tilt risk thresholds,
  and stronger lateral correction.
- It was worse than the first chest-pad run: `fall_events=247`,
  `box_drop_events=226`, min robot z `0.127119 m`, min box z `0.079985 m`,
  robot-root max tilt `1.277179 rad`, true box max tilt `1.223774 rad`, final
  relative offset `0.394498 m`.
- It still used no rollout root/velocity/box pose writes. Chest pad and top
  lid both enabled at step `67`.
- Lateral path error improved relative to the first chest-pad run
  (`final_box_target_lateral_error_m=0.494578`), but this is not useful
  because the robot and box fell/dropped.

Interpretation:

Over-slowing and holding too early destabilizes the larger-box chest-pad
posture. The next tuning should return to the first chest-pad transport speed
and change one thing at a time, starting with lateral correction strength or
contact geometry rather than lowering the main forward scale.

## Chest-Pad Lateral-Only Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_lateral_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- This run returned to the first chest-pad speed/hold settings and changed
  only lateral correction strength (`gain=0.12`, `limit=0.055`).
- It was worse than the first chest-pad run: `fall_events=335`,
  `box_drop_events=34`, min robot z `0.178246 m`, min box z `0.129841 m`,
  robot-root max tilt `3.134285 rad`, true box max tilt `3.127805 rad`.
- No rollout root/velocity/box pose writes occurred.
- Stronger lateral correction saturated at `0.055` and did not improve the
  path gate: final robot/box lateral errors were `0.796742 m` / `0.863152 m`.

Interpretation:

Increasing lateral velocity authority is not the fix. It destabilizes the
walking controller. The next path-correction attempt should keep the original
chest-pad lateral velocity setting and use a smaller yaw correction or contact
geometry change.

## Chest-Pad Mild-Yaw Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_mildyaw_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- This run kept the first chest-pad speed/hold and lateral settings, and added
  a small yaw correction (`gain=0.04`, `limit=0.08`).
- It was worse than the first chest-pad run: `fall_events=94`, min robot z
  `0.323945 m`, robot-root max tilt `2.273509 rad`, true box max tilt
  `2.335066 rad`.
- It did not drop the box and used no rollout root/velocity/box pose writes.
- Final robot/box target-line lateral errors were `-0.867466 m` /
  `-0.803925 m`, worse than the first chest-pad run.

Interpretation:

Adding yaw correction to the chest-pad posture is not the next main path. It
destabilizes root/box roll and worsens path error. The better next test is
contact geometry: keep the first chest-pad controller and change the support
surface/side constraint, not the commanded path.

## Chest-Pad Geometry Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_geom_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- This run preserved the first chest-pad controller and changed only contact
  geometry: chest pad local x/z `-0.08/0.12`, pad size
  `0.08 x 0.44 x 0.28`, top-lid y scale `1.25`, side rail height `0.14`,
  end-stop height `0.15`.
- It was worse than the first chest-pad run: `fall_events=328`,
  `box_drop_events=85`, min robot z `-0.542888 m`, min box z `-0.458421 m`,
  robot-root max tilt `3.140933 rad`, true box max tilt `3.109670 rad`.
- It used no rollout root/velocity/box pose writes.
- It moved far in target direction but with severe lateral drift: final
  robot/box target-line lateral errors were about `-2.42 m`.

Interpretation:

Making the chest support larger/higher is not sufficient and can make the
robot-box system drift and roll over. The first chest-pad geometry remains the
best current larger-box variant.

## Chest-Pad Opposite-Yaw Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_oppositeyaw_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 0
```

Result:

- This run preserved the first chest-pad geometry/controller, enabled mild yaw
  correction, and flipped `AGILE_COMMAND_HOLD_YAW_SIGN=-1.0`.
- It passed the larger-box strict checker: `fall_events=0`,
  `box_drop_events=0`, completed `700` steps, no rollout
  root/velocity/box pose writes.
- Stability: min robot z `0.721562 m`, min box z `0.825034 m`, robot-root max
  tilt `0.307758 rad`, true box max tilt `0.312059 rad`.
- Transport: final robot target-directed travel `1.435312 m`, final box
  target-directed travel `1.457102 m`.
- Contact/path: max relative offset `0.205432 m`, final relative offset
  `0.075546 m`, max robot/box target-line lateral error `0.115763 m` /
  `0.186329 m`.

Interpretation:

This is the first strict larger-box G1/AGILE chest-supported carrying pass in
this sequence. It is still a diagnostic, not final project success: it needs
longer-duration validation and additional posture/shape/mass held-outs before
claiming robust multi-posture carrying.

## 900-Step Opposite-Yaw Chest-Pad Validation

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_oppositeyaw_strict_900_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- The 700-step strict pass configuration was extended to `900` steps.
- It failed after longer-duration drift/tilt accumulation: `fall_events=26`,
  min robot z `0.369473 m`, robot-root max tilt `1.149047 rad`, true box max
  tilt `1.201307 rad`.
- It did not drop the box and used no rollout root/velocity/box pose writes.
- The robot/box reached larger peak target-directed travel
  (`1.626838 m` / `1.641323 m`) but ended lower (`1.184219 m` /
  `1.121610 m`), with final robot/box lateral errors `0.632209 m` /
  `0.673950 m`.

Interpretation:

The 700-step strict pass is real but not yet long-duration robust. The failure
mode is post-target drift and accumulated tilt after substantial travel. The
next controller should add a terminal carry/hold scale after the box reaches a
target-directed distance, instead of continuing to walk indefinitely.

## 900-Step Terminal-Scale Opposite-Yaw Validation

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Added terminal hold scale after box target-directed travel `1.35 m`, with
  terminal scale `0.06`, then reran the 900-step opposite-yaw chest-pad case.
- Terminal mode triggered at step `666` and was active for `234` steps.
- It improved the 900-step failure but did not pass: `fall_events=5` instead
  of `26`, no box drops, no rollout root/velocity/box pose writes.
- Path quality improved strongly: final robot/box lateral errors
  `0.073779 m` / `0.120379 m`.
- Remaining failures: robot-root max tilt `1.156729 rad`, true box max tilt
  `1.085554 rad`, final relative offset `0.286271 m`.

Interpretation:

Terminal scaling is the right direction for long-duration stability, but it
triggered too late and still allowed too much post-target motion
(`final_box_target_directed_travel_m=2.205989`). The next validation should
trigger terminal scaling earlier and use a smaller terminal scale.

## 900-Step Early-Terminal Opposite-Yaw Validation

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_early_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Terminal trigger was moved earlier to box target-directed travel `1.15 m`
  and terminal scale was reduced to `0.03`.
- It did not pass, but it removed the 900-step fall/drop failure:
  `fall_events=0`, `box_drop_events=0`, no rollout root/velocity/box pose
  writes.
- Terminal mode triggered at step `612` and was active for `288` steps.
- Remaining failures were only attitude gates: robot-root max tilt
  `0.463448 > 0.35` and true box max tilt `0.636226 > 0.45`.
- Contact and path were acceptable: final relative offset `0.072616 m`, max
  relative offset `0.205432 m`, final robot/box lateral errors `0.383956 m` /
  `0.315494 m`.

Interpretation:

Earlier terminal scaling is the right direction. It gives a 900-step no-fall,
no-drop, no-shortcut run, but strict root/box tilt still needs reduction. The
next validation should terminal-hold earlier or nearly stop after target
distance to reduce accumulated roll.

## 900-Step Near-Stop Terminal Opposite-Yaw Validation

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_nearstop_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 0
```

Result:

- Terminal trigger was moved earlier to box target-directed travel `1.05 m`
  and terminal scale was reduced to `0.015`.
- It passed the 900-step strict checker: `fall_events=0`, `box_drop_events=0`,
  completed `900` steps, no rollout root/velocity/box pose writes.
- Stability: min robot z `0.721562 m`, min box z `0.825034 m`, robot-root max
  tilt `0.307758 rad`, true box max tilt `0.384690 rad`.
- Transport: final robot target-directed travel `1.730244 m`, final box
  target-directed travel `1.759363 m`.
- Contact/path: max relative offset `0.205432 m`, final relative offset
  `0.108737 m`, max robot/box target-line lateral error `0.258455 m` /
  `0.362250 m`.
- Terminal mode triggered at step `590` and was active for `310` steps.

Interpretation:

This is now the strongest current larger-box result: a 900-step strict
diagnostic pass for chest-supported carrying with opposite-yaw correction and
near-stop terminal hold. It is still not final project success because other
postures, masses, shapes, and active probing/video-conditioned learning remain
unverified.

## Low-Carry Larger-Box Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_oppositeyaw_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Tested `LARGERBOX_STRICT_MODE=lowcarry` with no chest pad, using the
  opposite-yaw direction that passed for chest-supported carrying.
- It failed: `fall_events=117`, `box_drop_events=104`, min robot z
  `0.170354 m`, min box z `0.096605 m`, robot-root max tilt `0.990520 rad`,
  true box max tilt `0.991162 rad`.
- It used no rollout root/velocity/box pose writes.
- It made some target-directed progress, but with severe lateral drift:
  final robot/box travel `1.018163 m` / `1.075757 m`, final robot/box lateral
  errors `1.228960 m` / `1.272310 m`.

Interpretation:

Low-carry is not solved by reusing the chest-supported yaw controller. It
needs its own contact support or terminal strategy before it can count as a
second stable carrying posture.

## Low-Carry Terminal Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Added early terminal scale to low-carry: box target-directed trigger
  `0.65 m`, terminal scale `0.015`.
- It did not pass, but it removed the fall/drop failure:
  `fall_events=0`, `box_drop_events=0`, min robot z `0.761974 m`, min box z
  `0.816164 m`, robot-root max tilt `0.196663 rad`, true box max tilt
  `0.271947 rad`.
- It used no rollout root/velocity/box pose writes.
- Remaining failures are path-only: final robot/box target-line lateral errors
  `1.114195 m` / `1.306165 m`.

Interpretation:

Low-carry terminal hold can stabilize the robot and object, but path
correction is wrong for this posture. The opposite-yaw sign that works for
chest-supported carrying is likely wrong for low-carry; next test should flip
the yaw sign while keeping terminal hold.

## Low-Carry Terminal Default-Yaw Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal_defaultyaw_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- This run kept low-carry terminal hold and flipped yaw sign back to default
  (`AGILE_COMMAND_HOLD_YAW_SIGN=1.0`).
- It did not fall or drop, and used no rollout root/velocity/box pose writes.
- It failed because it moved in the wrong target direction: final robot/box
  target-directed travel `-0.387059 m` / `-0.584589 m`; terminal hold never
  triggered.
- It also exceeded true box tilt (`0.636519 rad`) and final relative offset
  (`0.286901 m`).

Interpretation:

Default yaw sign is wrong for low-carry target progress. The better low-carry
base is the terminal run with yaw sign `-1.0`, which stabilized fall/drop but
had path lateral drift. Next low-carry work should tune lateral correction
sign/gain around that base.

## Low-Carry Terminal Lateral-Sign Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal_latsign_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 1
```

Result:

- Flipping only `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0` was a clear
  regression.
- It failed with `fall_events=220`, `box_drop_events=40`, min robot z
  `0.172896 m`, min box z `0.115441 m`, robot-root max tilt `1.749395 rad`,
  true box max tilt `2.081289 rad`.
- It moved in the wrong target direction: final robot/box target-directed
  travel `-0.577424 m` / `-0.516827 m`.
- It used no rollout root/velocity/box pose writes.

Interpretation:

The flipped lateral sign is not the low-carry fix. Continue from the stable
yaw-sign `-1.0` terminal base, but disable or redesign lateral correction.

## Low-Carry Terminal No-Lateral Strict Diagnostic

Output:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_strict_700_targetnegx1/agile_low_cradle_freebox_walk/
```

Status:

```text
agile_low_cradle_freebox_walk build 0 check 0
```

Result:

- Disabled `AGILE_COMMAND_HOLD_LATERAL_CORRECTION` while keeping the stable
  low-carry terminal base.
- It passed strict 700-step check: `fall_events=0`, `box_drop_events=0`, min
  robot z `0.757182 m`, min box z `0.825777 m`, robot-root max tilt
  `0.227144 rad`, true box max tilt `0.241890 rad`.
- It made target-directed progress: final robot/box travel `1.994070 m` /
  `2.024888 m`.
- Target-line lateral error passed the strict gates: max robot/box lateral
  error `0.430948 m` / `0.414760 m`, final robot/box lateral error
  `0.427588 m` / `0.374435 m`.
- It used no rollout root/velocity/box pose writes.

Interpretation:

This is the strongest current low-carry result. The previous low-carry
path-only failure came from the lateral correction controller, not from the
low-carry posture itself. Next validation is a 900-step no-lateral run.

## Low-Carry 900-Step Followups

Output roots:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_strict_900_targetnegx1/
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_zerostop_strict_900_targetnegx1/
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_midstop_strict_900_targetnegx1/
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_nolateral_policythenstand_strict_900_targetnegx1/
```

Results:

- No-lateral with terminal scale `0.015` preserved path control but failed
  late: `fall_events=44`, `box_drop_events=25`, final box travel
  `3.440148 m`, max robot/box lateral error `0.440032 m` / `0.414760 m`.
- Zero terminal scale failed with under-driven or abrupt stop behavior:
  `fall_events=141`, `box_drop_events=97`, final box travel `0.339509 m`.
- Intermediate terminal scale `0.008` also failed:
  `fall_events=226`, `box_drop_events=36`, final box travel `0.576417 m`.
- `policy_then_stand` with delayed stand-target blending kept path errors
  acceptable but hurt object stability: `fall_events=281`,
  `box_drop_events=258`, final box travel `1.778671 m`.
- None of these runs used rollout root/velocity/box pose writes.

Interpretation:

Low-carry has a real 700-step strict pass when lateral correction is disabled,
but 900-step low-carry is not solved. The rejected fixes are:
lateral-sign reversal, generic lower terminal speed, zero terminal speed, and
generic stand-target blending. The next engineering step should be a
posture-specific low-carry hold/recovery controller that preserves
arm/cradle/object contact while limiting forward drift.

## Low-Carry Latched Terminal Followups

Outputs:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_nolateral_latchedzerostop_strict_900_targetnegx1/
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_nolateral_latchedmicro_strict_900_targetnegx1/
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export_short_strict_900_targetnegx1/
```

Results:

- Latched zero-stop worked as a command latch (`latched=True`, step `381`,
  final x command `0.0`) but failed: `fall_events=141`,
  `box_drop_events=97`.
- Latched micro-hold with terminal scale `0.006` reduced fall/drop relative
  to zero-stop (`fall_events=90`, `box_drop_events=42`) but drifted laterally:
  final robot/box lateral errors `1.375929 m` / `1.389495 m`.
- Terminal-only lateral correction with low gain/limit was a valid configured
  run and reduced lateral drift (`-0.595748 m` / `-0.657782 m`) but destabilized
  much earlier: `fall_events=288`, `box_drop_events=269`.
- First-fall comparison: no-lateral micro-hold first fell at step `810`;
  terminal-only lateral first fell at step `620` with small lateral error near
  zero. This means lateral correction should be gated by lateral error
  magnitude, not only by terminal latch.

Next:

Test terminal-only lateral correction with lateral-error threshold gating.
