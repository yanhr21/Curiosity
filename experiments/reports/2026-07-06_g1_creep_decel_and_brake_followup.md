# 2026-07-06 G1 Creep Decel And Brake Follow-Up

## Scope

Follow-up to `experiments/reports/2026-07-06_g1_low_cradle_creep_diagnostics.md`.
All runs used direct Isaac G1, a free dynamic box on the low/close torso
cradle, no root/velocity/box rollout shortcuts, and Curiosity-owned
`tmux` + `srun` jobs.

The goal was to turn the 560-step `low_push032` short-distance diagnostic
into a 700-step stable carry by adding deceleration or braking to
`targeted_creep`.

## Code Changes

`scripts/isaac/build_core_world_g1_box_scene.py` now supports:

- travel-based creep scaling:
  `--creep-decel-box-travel-start/end`,
  `--creep-decel-robot-travel-start/end`,
  `--creep-min-amplitude-scale`, `--creep-min-push-scale`,
  `--creep-min-bias-scale`;
- pitch-based creep braking:
  `--creep-pitch-brake-threshold`,
  `--creep-pitch-brake-rate-threshold`,
  `--creep-pitch-brake-amplitude-scale`,
  `--creep-pitch-brake-push-scale`,
  `--creep-pitch-brake-bias-scale`;
- optional brake latching:
  `--creep-pitch-brake-latch`;
- optional positive-pitch-only braking:
  `--creep-pitch-brake-positive-only`;
- summary fields for decel/brake activation steps and minimum applied scales.

## Runner Scripts

Added:

```text
scripts/isaac/run_core_world_g1_low_creep_decel_tune.sh
scripts/isaac/run_core_world_g1_low_creep_latched_brake_tune.sh
scripts/isaac/run_core_world_g1_low_creep_positive_brake_tune.sh
scripts/isaac/run_core_world_g1_low_creep_zero_hold_tune.sh
scripts/isaac/run_core_world_g1_low_creep_reverse_brake_tune.sh
scripts/isaac/run_core_world_g1_low_creep_hold_balance_tune.sh
```

## Results

### Decel Tune

Output:

```text
experiments/outputs/core_world_g1_low_creep_decel_tune/20260706_g1_low_creep_decel_tune1/
```

Result:

- Travel-based decel alone did not stop late forward pitch. Most cases still
  ended around `0.74 m` target-directed travel but failed with about
  `24-25` fall events, `6-8` drops, max tilt about `1.16 rad`, and final
  relative offset about `0.44-0.45 m`.
- `decel014_024_brake012` was stable for 700 steps with fall/drop `0`, max
  tilt `0.120622 rad`, and final relative offset `0.018245 m`, but it only
  reached `0.086960 m` final box target-directed travel. This is stable
  holding, not enough carrying distance.

### Latched Brake Tune

Output:

```text
experiments/outputs/core_world_g1_low_creep_latched_brake_tune/20260706_g1_low_creep_latched_brake_tune1/
```

Result:

- `latch012_bias0` latched at step `26` due initial negative pitch because
  the first latch implementation used `abs(pitch)`. It stayed stable but only
  moved `0.004886 m`.
- Later latch thresholds triggered at steps `583-598`, too late to arrest
  forward pitch. Those cases reached about `0.71-0.73 m` but failed with
  falls/drops and final relative offset about `0.44-0.46 m`.

### Positive-Pitch Brake Tune

Output:

```text
experiments/outputs/core_world_g1_low_creep_positive_brake_tune/20260706_g1_low_creep_positive_brake_tune1/
```

Result:

- Positive-only latch fixed the early negative-pitch trigger, but thresholds
  `0.08`, `0.10`, and `0.12 rad` still triggered at steps `558-574`.
  Stopping gait at that point did not arrest the forward fall.
- All cases failed with about `24-26` fall events and `9-11` drops.

### Zero-Offset Travel Hold

Output:

```text
experiments/outputs/core_world_g1_low_creep_zero_hold_tune/20260706_g1_low_creep_zero_hold_tune1/
```

Result:

- Returning to base stand after box target travel `0.08`, `0.10`, `0.12`, or
  `0.14 m` was not sufficient. Hold triggered at steps `465`, `504`, `528`,
  and `545`, but all cases still fell later.
- This shows the problem is dynamic braking/recovery, not just terminal
  posture offsets.

## Current Interpretation

The low/close cradle plus `targeted_creep` can move the free dynamic box, but
the current open-loop creep accumulates forward momentum. Passive decel,
zeroing gait, latched stop, or fixed stand hold are not enough once the body
and box start pitching forward.

### Reverse-Brake Tune

Output:

```text
experiments/outputs/core_world_g1_low_creep_reverse_brake_tune/20260706_g1_low_creep_reverse_brake_tune1/
```

Result:

- Reverse-brake activation triggered by box target travel at steps `465`,
  `504`, `528`, and `545`, depending on threshold.
- All cases failed. The distance-producing cases ended around
  `0.763650-0.786626 m` target-directed box travel but still had `20-26`
  fall events and `4-9` box-drop events.
- Negative stance-push did not behave like a reliable braking step in this
  open-loop formulation. It increased travel but did not recover pitch or
  box-relative drift.

### Hold-Balance Tune

Output:

```text
experiments/outputs/core_world_g1_low_creep_hold_balance_tune/20260706_g1_low_creep_hold_balance_tune1/
```

Result:

- Negative balance sign was destructive: several cases drove the robot
  backward to about `-1.7` to `-2.4 m` target-directed travel and produced
  hundreds of fall/drop events.
- Positive balance sign was stable but suppressed locomotion:
  `hold010_gain035_signpos` kept fall/drop `0` and max tilt `0.077382 rad`,
  but final box target-directed travel was only `0.003041 m`.

## Updated Interpretation

The explicit reverse-brake and simple hold-balance variants also failed. At
this point, continuing to sweep hand-written open-loop creep is low value.
The next valid implementation should replace the gait backend with a real
controller-backed locomotion policy already available locally, then re-test
the same Core scene in stages:

1. AGILE policy, no box, verifies the controller moves the G1 without root or
   velocity rollout writes.
2. AGILE policy, fixed light torso payload, verifies locomotion under small
   load.
3. AGILE policy, free low-cradle box, verifies whether a dynamic box can move
   with the robot before adding active probing or video reward.

Do not claim complete carrying from these runs.
