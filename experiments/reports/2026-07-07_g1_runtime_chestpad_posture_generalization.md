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
- Result:
  `fail`, 1/2 cases passed. Slurm job `169793` (`g1_postgate`) ran on
  `server57` and exited `FAILED 1:0`, because the aggregate gate failed.
- Summary:
  `experiments/outputs/core_world_g1_posture_conditioned_gate/20260707_g1_posture_conditioned_gate/posture_conditioned_gate_summary.json`

| Case | Result | Fall/Drop | Robot/Box Travel m | Max Robot/Box Tilt rad | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `low_front_060` | pass | 0/0 | 2.051/2.032 | 0.309/0.428 | 105 | Reproduces the current narrow low-front runtime-support pass. |
| `close_front_060_conditioned` | fail | 142/0 | 0.731/0.650 | 3.130/3.129 | 0 | The packaged close-front hypothesis collapses; this is not a small final-hold error. |

The clean gate confirms the boundary: current G1/AGILE runtime chest-pad logic
passes only the tuned low-front posture. Close-front remains unsolved.

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
- Result:
  `fail`, 0/3 cases passed. Slurm job `169822` (`g1_cfstand`) ran on
  `server44` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_final_stand/20260707_g1_lowcarry_close_front_final_stand/close_front_final_stand_summary.json`

| Case | Result | Fall/Drop | Robot/Box Travel m | Max Robot/Box Tilt rad | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `stand_late_crouch_b003` | fail | 262/27 | 1.488/1.454 | 3.102/3.111 | 0 | Late crouched stand does not arrest the posture; first fall at step 924. |
| `freeze_window_then_stand_b002` | fail | 226/0 | 0.974/0.657 | 3.137/3.138 | 0 | Freeze/stand never reaches the target window; first fall at step 924. |
| `policy_then_stand_b002` | fail | 700/463 | 0.835/0.604 | 2.494/1.843 | 0 | Policy-then-stand destabilizes much earlier; first fall at step 409. |

The final-stand hypothesis is negative. The close-front failure is not solved
by standing up after final hold; the current command/support setup can enter a
bad pitch/roll trajectory before a useful target-window dwell exists.

## Close-Front Pretarget Repair Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_pretarget_repair_suite.sh`
- Purpose:
  repair the close-front trajectory before target-window/final-hold, rather
  than trying to save it with final-stand after large tilt has already begun.
- Method:
  keep the same 0.60 kg close-front geometry, runtime target-window chest-pad
  trigger, and strict gates, but add an early box-progress controller plus
  box-lateral controller. Forward command is suppressed when robot/box tilt
  exceeds configured pretarget limits.
- Cases:
  `progress_conservative`, `progress_mid`, and `progress_mid_no_hold_lat`.
- Result:
  `fail`, 0/3 cases passed. Slurm job `169858` (`g1_cfpre`) ran on
  `server44` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_pretarget_repair/20260707_g1_lowcarry_close_front_pretarget_repair/close_front_pretarget_repair_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Longest/End Streak | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `progress_conservative` | fail | 485/247 | 802/864 | 136 | 73/0 | Useful direction: reaches target window at step 652 and holds briefly, but cannot retain through the end. |
| `progress_mid` | fail | 917/757 | 383/396 | 0 | 0/0 | Too aggressive; collapses early. |
| `progress_mid_no_hold_lat` | fail | 705/579 | 595/663 | 0 | 0/0 | Disabling hold lateral/yaw makes it worse; no target-window dwell. |

The pretarget experiment changes the failure mode. Conservative box-progress
control is the first close-front variant to exceed the target-window stable
step count, but it still fails fall/drop and end-streak gates. The next repair
should not increase drive; it should arrest or retain the robot/box once the
target window is reached.

## Close-Front Window-Arrest Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_window_arrest_suite.sh`
- Purpose:
  build directly on `progress_conservative`, which reached target-window first
  stable step `652`, then stop or soften the progress command as soon as both
  robot and box enter the target window.
- Cases:
  `stop_window_pad620`, `stop_window_pad650_soft_hold`, and
  `stop_window_pad600`.
- Gate:
  unchanged strict fall/drop, target-window, final-hold, tilt,
  lateral-error, stop-window latch, and no rollout root/box writes checks.
- Result:
  `fail`, 0/3 cases passed. Slurm job `169867` (`g1_cfwin`) ran on
  `server36` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_window_arrest/20260707_g1_lowcarry_close_front_window_arrest/close_front_window_arrest_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `stop_window_pad620` | fail | 656/617 | 494/533 | 0 | Removing the early hold/adaptive behavior prevents the previous target-window entry. |
| `stop_window_pad650_soft_hold` | fail | 656/617 | 494/533 | 0 | Same failure; soft hold does not matter because target-window latch never occurs. |
| `stop_window_pad600` | fail | 656/617 | 494/533 | 0 | Same failure; earlier pad trigger is irrelevant without window entry. |

This is a useful negative control. The `progress_conservative` target-window
entry depended on the original early hold/adaptive behavior from the lowcarry
suite. A valid retention repair should preserve that early behavior and only
alter runtime support/freeze after the target-window region is reached.

## Close-Front Window-Retention V2 Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_window_retention_v2_suite.sh`
- Purpose:
  preserve the exact early hold/adaptive behavior from
  `progress_conservative`, then test only target-window retention changes.
- Cases:
  `pad620`, `pad620_freeze`, and `pad650_freeze_zero_corr`.
- Gate:
  unchanged strict fall/drop, target-window, final-hold, tilt,
  lateral-error, and no rollout root/box writes checks.
- Result:
  `fail`, 0/3 cases passed. Slurm job `169906` (`g1_cfv2`) ran on
  `server63` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_window_retention_v2/20260707_g1_lowcarry_close_front_window_retention_v2/close_front_window_retention_v2_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Freeze | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `pad620` | fail | 591/573 | 709/727 | 57 | no | Earlier pad trigger worsens the prior `progress_conservative` result. |
| `pad620_freeze` | fail | 593/0 | 707/- | 55 | yes, step 653 | Freeze prevents drop and preserves final travel/lateral, but still falls early. |
| `pad650_freeze_zero_corr` | fail | 593/0 | 707/- | 55 | yes, step 653 | Zeroing corrections does not change the freeze failure. |

The v2 result rules out "earlier support/freeze at first window entry" as the
repair. The best close-front evidence remains `progress_conservative` with
runtime support at step `700`: it held the window longer (`136` stable steps)
and failed later (`first fall 802`). The next comparison should vary support
timing/geometry around the original pad700 behavior rather than enabling it at
step `653`.

## Close-Front Support-Timing Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_support_timing_suite.sh`
- Purpose:
  keep the full `progress_conservative` controller and compare whether the
  runtime chest support helps or hurts target-window retention.
- Cases:
  `no_runtime_pad`, `pad760`, and `pad700_small`.
- Status:
  submitted as Slurm job `169916` (`g1_cfsup`) through tmux
  `curiosity_g1_close_front_support_timing_0707`; pending on GPU priority as
  of `2026-07-07 14:19 CST`.

## Next Step

Do not claim posture-general carrying from the current G1 route. The next
implementation should add an explicit posture-conditioned controller gate:

- estimate or select carry posture/geometry before walking,
- choose different command/lateral/yaw/support parameters per posture,
- repair the pre-target close-front trajectory rather than only the final
  stand/hold phase,
- for close-front specifically, build on `progress_conservative` and fix
  target-window retention after step `652`, because stronger progress settings
  collapse earlier,
- preserve the early hold/adaptive behavior that allowed `progress_conservative`
  to reach the window; removing it caused `g1_cfwin` to fail before window
  entry,
- do not trigger chest support/freeze immediately at first target-window entry;
  v2 shortened the stable window and moved first fall earlier,
- keep the same strict checks: fall/drop 0, no rollout root/box writes,
  final target-window hold, tilt bounds, and final lateral error bounds.
