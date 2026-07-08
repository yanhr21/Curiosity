# 2026-07-06 G1 Low-Cradle Targeted-Creep Diagnostics

## Scope

Direct Isaac G1 + free dynamic box diagnostics. No model download, no
retargeting, no real data, no root/box rollout shortcut. All simulation runs
were submitted through Curiosity-owned `tmux` sessions with `srun`.

The best current path is a low/close torso cradle with `targeted_creep`.
This is still diagnostic scaffold evidence, not complete long-duration
robot carrying.

## Commands And Results

### Stable-Cradle Propulsion Tune

Command:

```bash
SUITE_STAMP=20260706_g1_stable_cradle_propulsion_tune1 DEVICE=cpu STRICT=0 COMPUTE_SIDE_STARTUP_SLEEP=20 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --job-name=g1_prop_tune \
  bash scripts/isaac/run_core_world_g1_stable_cradle_propulsion_tune.sh
```

Output:

```text
experiments/outputs/core_world_g1_stable_cradle_propulsion_tune/20260706_g1_stable_cradle_propulsion_tune1/
logs/g1_stable_cradle_propulsion_tune/20260706_g1_stable_cradle_propulsion_tune1.log
```

Key results:

- `close_targeted_creep_push018`: passed 5 cm gate. Fall/drop `0`, final box
  target-directed travel `0.053698 m`, max tilt `0.114707 rad`, final
  relative offset `0.029719 m`, rollout root/box writes `0`.
- `low_targeted_creep_push028`: passed 5 cm gate. Fall/drop `0`, final box
  target-directed travel `0.085761 m`, max tilt `0.125165 rad`, final
  relative offset `0.055600 m`, rollout root/box writes `0`.
- `close_targeted_creep_push028`: moved much farther
  (`0.363493 m`) but failed stability/retention gates with max tilt
  `0.305524 rad` and final relative offset `0.151852 m`.
- `staged_march` cases remained stable but target-directed travel stayed too
  small for useful carrying.

### Targeted-Creep Stop Tune

Command:

```bash
SUITE_STAMP=20260706_g1_targeted_creep_stop_tune1 DEVICE=cpu STRICT=0 COMPUTE_SIDE_STARTUP_SLEEP=20 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --job-name=g1_creep_stop \
  bash scripts/isaac/run_core_world_g1_targeted_creep_stop_tune.sh
```

Output:

```text
experiments/outputs/core_world_g1_targeted_creep_stop_tune/20260706_g1_targeted_creep_stop_tune1/
logs/g1_targeted_creep_stop_tune/20260706_g1_targeted_creep_stop_tune1.log
```

Key result:

- `low_push032` passed the 10 cm diagnostic gate for 560 steps. Fall/drop
  `0`, final box target-directed travel `0.164657 m`, max tilt
  `0.128766 rad`, final relative offset `0.071063 m`, max relative offset
  `0.071695 m`, min box z `0.867815 m`, rollout root/box writes `0`.
- Higher/closer push cases without low cradle or with naive stop failed by
  forward pitch and box/robot relative drift.

### Low-Cradle Longer Validation

Command:

```bash
SUITE_STAMP=20260706_g1_low_cradle_creep_validation1 DEVICE=cpu STRICT=0 COMPUTE_SIDE_STARTUP_SLEEP=20 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 \
  --job-name=g1_low_creep \
  bash scripts/isaac/run_core_world_g1_low_cradle_creep_validation.sh
```

Output:

```text
experiments/outputs/core_world_g1_low_cradle_creep_validation/20260706_g1_low_cradle_creep_validation1/
logs/g1_low_cradle_creep_validation/20260706_g1_low_cradle_creep_validation1.log
```

Result:

- `low_push032_700` failed. It moved `0.758020 m`, but had `18` fall events,
  `9` box-drop events, min robot z `0.399267 m`, min box z `0.101657 m`,
  max tilt `1.045130 rad`, and final relative offset `0.429096 m`.
- `low_push032_1000` failed harder. It had `318` fall events and `274`
  box-drop events. The same open-loop creep cannot be called long-duration
  carrying.

### Low-Creep Terminal Hold Tune

First run used `--terminal-hold-start-step 0`, which made hold active from
step 0. That run is invalid for the intended "hold after travel threshold"
test.

Corrected retry command:

```bash
SUITE_STAMP=20260706_g1_low_creep_terminal_hold_tune_retry2 DEVICE=cpu STRICT=0 COMPUTE_SIDE_STARTUP_SLEEP=20 \
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --job-name=g1_low_hold2 \
  bash scripts/isaac/run_core_world_g1_low_creep_terminal_hold_tune.sh
```

Output:

```text
experiments/outputs/core_world_g1_low_creep_terminal_hold_tune/20260706_g1_low_creep_terminal_hold_tune_retry2/
logs/g1_low_creep_terminal_hold_tune/20260706_g1_low_creep_terminal_hold_tune_retry2.log
```

Result:

- All corrected terminal-hold cases failed 700-step gates.
- Hold was triggered by `box_target_travel` at steps `504`, `528`, or `552`,
  but fixed hold postures did not arrest forward pitch.
- Best interpretation: terminal hold needs a real deceleration / counter-step
  / posture-recovery controller. A fixed symmetric posture offset is not a
  sufficient brake.

## Current Best Evidence

Best valid diagnostic:

```text
low_push032, 560 steps, targeted_creep, low/close free-box cradle
```

Metrics:

```text
fall_events=0
box_drop_events=0
final_box_target_directed_travel_m=0.164657
max_tilt_rad=0.128766
final_box_robot_relative_offset_error_m=0.071063
max_box_robot_relative_offset_error_m=0.071695
min_box_z_m=0.867815
root_pose_write_count_rollout=0
root_velocity_write_count_rollout=0
box_pose_write_count_rollout=0
```

This is a short-distance no-shortcut free-box carrying diagnostic. It is not
the final objective because the same configuration fails 700/1000-step
validation.

## Next Required Step

Do not wait for external models. Implement an explicit deceleration and
recovery phase in the direct Isaac G1 controller:

- reduce creep amplitude/push as target travel grows,
- bias torso/ankle/hip to counter forward pitch without reversing the robot,
- optionally add a counter-step or asymmetric support response,
- gate by 700+ steps, no fall/drop, no root/box rollout writes, and final
  target-directed travel above the declared threshold.
