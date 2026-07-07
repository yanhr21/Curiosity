# G1 Runtime Chest-Pad Posture Generalization

Date: 2026-07-07

## Purpose

Test whether the current best 0.60 kg G1/AGILE low-carry runtime chest-pad
configuration generalizes beyond its single tuned low-front posture.

This is still a diagnostic scaffold. It is not learned unknown-load carrying
and not evidence that arbitrary carry postures work.

## Suites

- Generalization suite:
  `scripts/isaac/run_core_world_g1_lowcarry_runtime_chestpad_posture_generalization_suite.sh`
- Generalization summary:
  `experiments/outputs/core_world_g1_lowcarry_runtime_chestpad_posture_generalization/20260707_g1_lowcarry_runtime_chestpad_posture_generalization/posture_generalization_summary.json`
- Close-front repair suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_repair_suite.sh`
- Close-front repair summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_repair/20260707_g1_lowcarry_close_front_repair/close_front_repair_summary.json`

Both suites ran through tmux-held Slurm GPU allocations, not on the login node.

## Generalization Result

Overall result: `fail`, 1/5 cases passed.

| Case | Result | Fall/Drop | Robot/Box Travel m | Max Robot/Box Tilt rad | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `low_front_060` | pass | 0/0 | 2.051/2.032 | 0.309/0.428 | Reproduces the current narrow pass. |
| `close_front_060` | fail | 0/0 | 1.574/1.651 | 0.344/0.476 | Stable but misses target window; lateral drift near 0.92 m and box tilt slightly high. |
| `forward_reach_060` | fail | 268/189 | 0.219/0.147 | 1.099/1.052 | Forward-reach geometry destabilizes and drops the box. |
| `wide_box_060` | fail | 234/62 | 1.082/0.916 | 2.396/2.418 | Wider box destabilizes late and drops. |
| `low_front_080` | fail | 436/362 | -0.101/-0.393 | 2.122/3.126 | 0.80 kg exceeds this tuned support/carry setting. |

The direct conclusion is that the current runtime chest-pad method is not
posture-general. It is a narrow engineered support solution around the
`low_front_060` geometry.

## Close-Front Repair Result

Overall result: `fail`, 0/3 cases passed.

| Case | Result | Fall/Drop | Robot/Box Travel m | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `lateral_sign_neg` | fail | 0/0 | 1.767/1.810 | 27 | Best repair; no fall/drop, but tilt and lateral-error gates still fail. |
| `lateral_sign_neg_stronger` | fail | 0/0 | 1.649/1.717 | 1 | Stronger lateral command worsens target-window hold and exceeds command-y gate. |
| `lateral_sign_neg_stronger_tiltpad` | fail | 0/0 | 1.649/1.717 | 1 | Box-tilt chest-pad trigger did not improve the stronger lateral case. |

The close-front case is the most promising adjacent posture because it can
remain upright and retain the box. However, scalar lateral sign/gain tuning is
not enough. The next aligned step is a posture-conditioned command/support
policy or controller layer that changes gait/heading/support behavior based on
carry geometry, rather than trying to reuse the exact low-front command.

## Close-Front Command-Conditioned Result

- Suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_command_conditioned_suite.sh`
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_command_conditioned/20260707_g1_lowcarry_close_front_command_conditioned/close_front_command_conditioned_summary.json`

Overall result: `fail`, 0/3 cases passed.

| Case | Result | Fall/Drop | Robot/Box Travel m | Max Robot/Box Tilt rad | Lateral Error m | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `command_y_neg002` | fail | 0/0 | 1.585/1.414 | 0.809/1.475 | 1.007/1.160 | Too much lateral drift and tilt. |
| `command_y_neg004` | fail | 0/0 | 1.374/1.443 | 0.246/0.329 | 0.054/0.034 | Strong improvement: stable and centered, but under-travels. |
| `command_y_neg006` | fail | 0/0 | 1.672/1.695 | 0.517/0.564 | 1.453/1.600 | Reaches the window briefly but drifts badly. |

This confirms that posture-conditioned base command can materially change the
close-front behavior. The `y=-0.04` command turns the close-front case from
large lateral drift into a centered, stable, under-travel case.

## Close-Front Command-Refinement Result

- Suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_command_refine_suite.sh`
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_command_refine/20260707_g1_lowcarry_close_front_command_refine/close_front_command_refine_summary.json`

Overall result: `fail`, 0/3 cases passed.

| Case | Result | Fall/Drop | Robot/Box Travel m | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `x011_yneg004` | fail | 243/111 | 1.961/1.871 | 37 | More forward command reaches target-window region but destabilizes. |
| `x012_yneg004` | fail | 332/312 | 1.538/1.320 | 0 | Too aggressive, falls/drops. |
| `x012_yneg003` | fail | 0/0 | 1.504/1.357 | 0 | Stable but laterally wrong. |

This shows the close-front posture is sensitive to forward command. Increasing
forward speed is not a clean fix; it trades under-travel for late fall/drop.

## Close-Front Hold-Delay Result

- Suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_hold_delay_suite.sh`
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_hold_delay/20260707_g1_lowcarry_close_front_hold_delay/close_front_hold_delay_summary.json`

Overall result: `fail`, 0/3 cases passed.

| Case | Result | Fall/Drop | Robot/Box Travel m | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `steps950_final080` | fail | 285/134 | 1.056/0.779 | 0 | Early delay setting destabilizes. |
| `steps1000_final100` | fail | 0/0 | 1.830/1.822 | 67 | Close, no fall/drop, but tilt and final-hold steps fail. |
| `steps1050_final120` | fail | 0/0 | 2.026/2.103 | 76 | Closest so far: centered and reaches target, but tilt slightly high and stable-window/final-hold duration short. |

The `steps1050_final120` case is the strongest close-front result so far. It
does not pass, but it shows that a posture-conditioned command plus delayed
hold can nearly reach the same task window as the tuned low-front pass.

## Pending Final-Stabilize Follow-Up

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stabilize_suite.sh`
- Intended test:
  1200-step close-front `x=0.10,y=-0.04,final=1.20` with earlier box-tilt
  chest-pad triggers.
- Slurm job:
  `169771` / `g1_finalstab`
- Status:
  cancelled before running because it stayed pending with estimated start
  `2026-07-07T17:00:00`.

There is no final-stabilize result yet. The script is available for a later GPU
slot, but it must not be reported as evidence until a summary exists.

## Posture-Conditioned Gate Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_posture_conditioned_gate_suite.sh`
- Cases:
  `low_front_060` and `close_front_060_conditioned`
- Gate:
  unchanged strict fall/drop, no rollout root/box writes, target-window,
  final-hold, tilt, and lateral-error checks.
- Status:
  submitted as Slurm job `169793` (`g1_postgate`) through tmux
  `curiosity_g1_posture_gate_0707`; pending on GPU priority as of
  `2026-07-07 13:22 CST`.

This is the next clean experiment entrypoint. It packages the known passing
low-front command and the best current close-front conditioned hypothesis into
one reproducible gate. It is not evidence until its summary JSON exists.

## Close-Front Final-Stand Follow-Up

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stand_suite.sh`
- Hypothesis:
  the `steps1050_final120` close-front case reaches the target window at step
  `968`, spawns the runtime chest pad at step `969`, and only exceeds the
  robot/box tilt gates around step `1040+`; therefore the next repair should
  stabilize after final-hold/target-window entry rather than add more forward
  drive.
- Cases:
  late crouched final-stand blend, target-window freeze then final stand, and
  policy-then-stand blend. All keep the same close-front `x=0.10,y=-0.04`
  command, runtime chest-pad trigger, strict fall/drop, target-window,
  final-hold, final-stand, tilt, lateral-error, and no-shortcut gates.
- Slurm job:
  `169822` (`g1_cfstand`) through tmux
  `curiosity_g1_close_front_final_stand_0707`.
- Status:
  pending on GPU priority as of `2026-07-07 13:22 CST`.

This is not evidence yet. It is the next targeted test for the late-tilt
failure mode after close-front already reaches the target region.

## Next Step

Do not claim posture-general carrying from the current G1 route. The next
implementation should add an explicit posture-conditioned controller gate:

- estimate or select carry posture/geometry before walking,
- choose different command/lateral/yaw/support parameters per posture,
- keep the same strict checks: fall/drop 0, no rollout root/box writes,
  final target-window hold, tilt bounds, and final lateral error bounds.
