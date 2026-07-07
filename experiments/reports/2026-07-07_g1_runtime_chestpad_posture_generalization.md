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
- Result:
  `fail`, 0/3 cases passed. Original long-walltime pending job `169916` was
  cancelled before running and replaced with shorter Slurm job `169922`
  (`g1_cfsup`) through tmux
  `curiosity_g1_close_front_support_timing_short_0707`; it ran on `server63`
  and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_support_timing/20260707_g1_lowcarry_close_front_support_timing/close_front_support_timing_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Longest/End Streak | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `no_runtime_pad` | fail | 355/0 | 901/- | 130 | 78/0 | Best of this suite: no drops and latest fall, but still cannot hold the target window through the end. |
| `pad760` | fail | 396/303 | 904/997 | 133 | 81/0 | Delayed chest support does not prevent fall and later causes large travel/drop failure. |
| `pad700_small` | fail | 416/241 | 884/900 | 73 | 50/0 | Smaller support is worse than no pad and worsens target-window retention. |

This rules out the current runtime chest-pad timing/geometry family for the
close-front repair. The support-free trajectory is the least bad close-front
variant: it gets into the target window, avoids drops, and fails late. The next
test should not add more chest-pad geometry; it should try late final-hold,
target-window freeze, or a short reverse brake on the no-pad trajectory.

## Close-Front Late-Hold Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_late_hold_suite.sh`
- Purpose:
  build on `no_runtime_pad` from support-timing and test whether later final
  latch plus zero command, target-window freeze, or a short reverse brake can
  retain the close-front trajectory after it reaches the target region.
- Cases:
  `late_final_180`, `late_final_180_freeze`, and `late_final_180_brake`.
- Result:
  `fail`, 0/3 cases passed. Slurm job `169927` (`g1_cflate`) ran on
  `server63` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_late_hold/20260707_g1_lowcarry_close_front_late_hold/close_front_late_hold_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Final Latch Step | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `late_final_180` | fail | 668/548 | 632/640 | 790 | 0 | Final latch occurs after collapse; worse than `no_runtime_pad`. |
| `late_final_180_freeze` | fail | 668/548 | 632/640 | 790 | 0 | Freeze never latches because the robot falls before target-window stability. |
| `late_final_180_brake` | fail | 668/544 | 632/640 | 790 | 0 | Reverse brake is activated too late and over-travels badly. |

This rules out late final latch at `1.80 m` for the current close-front
trajectory. The earlier final latch from `no_runtime_pad` at step `540` is
necessary to reach any target-window dwell. The failure to repair is now a
late roll-collapse problem, not a missing late command stop.

## Close-Front Rescue/Balance Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_balance_suite.sh`
- Purpose:
  return to the `no_runtime_pad` early-final-latch baseline and test whether
  existing controller hooks can catch the late roll collapse around steps
  `780-910`.
- Cases:
  `rescue_crouch_abs040`, `rescue_crouch_abs055`,
  `balance_roll_avg_pos`, and `balance_roll_avg_neg`.
- Gate:
  unchanged strict fall/drop, target-window, final-hold, tilt,
  lateral-error, and no rollout root/box writes checks.
- Result:
  `fail`, 0/4 cases passed. Slurm job `169935` (`g1_cfresc`) ran on
  `server63` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_rescue_balance/20260707_g1_lowcarry_close_front_rescue_balance/close_front_rescue_balance_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Longest/End Streak | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `rescue_crouch_abs040` | fail | 219/0 | 1081/- | 81 | 52/0 | Best current close-front repair: no drops and fall delayed, but final lateral drift exits the window. |
| `rescue_crouch_abs055` | fail | 386/344 | 914/956 | 142 | 90/0 | Longer window dwell but loses the box. |
| `balance_roll_avg_pos` | fail | 634/587 | 666/680 | 16 | 16/0 | Lateral roll-target positive sign destabilizes early. |
| `balance_roll_avg_neg` | fail | 481/463 | 819/837 | 86 | 86/0 | Negative sign gets dwell but over-travels and drops. |

The useful direction is not lateral roll-target. `rescue_crouch_abs040`
changes the failure from early roll collapse to late lateral drift with no box
drop. Its box-lateral command was effectively suppressed during final hold
(`agile_command_box_lateral_max_abs_command_y` about `4.5e-05`), so the next
small test should allow lateral correction during final hold while preserving
zero forward command.

## Close-Front Rescue-Lateral Refine Entrypoint

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_lateral_refine_suite.sh`
- Purpose:
  refine `rescue_crouch_abs040` by allowing final-hold box-lateral correction
  and checking whether the rescue is too strong or triggers too early.
- Cases:
  `rescue040_lat_unscaled`, `rescue040_lat_unscaled_signneg`,
  `rescue040_milder_crouch`, and `rescue045_mid_crouch`.
- Result:
  `fail`, 0/4 cases passed. Slurm job `169944` (`g1_cflat`) ran on
  `server63` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_rescue_lateral_refine/20260707_g1_lowcarry_close_front_rescue_lateral_refine/close_front_rescue_lateral_refine_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `rescue040_lat_unscaled` | fail | 489/468 | 811/832 | 84 | Unscaled final-hold lateral correction causes runaway over-travel. |
| `rescue040_lat_unscaled_signneg` | fail | 674/590 | 626/647 | 0 | Opposite lateral sign is immediately worse. |
| `rescue040_milder_crouch` | fail | 489/468 | 811/832 | 84 | Milder crouch does not repair the unscaled-lateral failure. |
| `rescue045_mid_crouch` | fail | 489/468 | 811/832 | 84 | Mid-threshold crouch also matches the runaway failure. |

This rules out unscaled final-hold box-lateral correction. The best
close-front branch remains `rescue_crouch_abs040`, which keeps the box but
under-travels at the end. The next narrow test should keep that rescue and
sweep moderate final-latch thresholds between the current early `1.20 m` latch
and the too-late `1.80 m` latch.

## Close-Front Rescue Final-Latch Sweep

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_final_latch_sweep.sh`
- Purpose:
  retain the useful `rescue_crouch_abs040` and test whether a moderate final
  latch fixes end-window retention without causing the late-latch collapse.
- Cases:
  `final135`, `final145`, and `final155`.
- Result:
  `fail`, 0/3 cases passed. Slurm job `169964` (`g1_cffinal`) ran on
  `server39` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_rescue_final_latch/20260707_g1_lowcarry_close_front_rescue_final_latch/close_front_rescue_final_latch_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `final135` | fail | 516/458 | 784/799 | 75 | Moderate final latch is worse than `rescue_crouch_abs040`. |
| `final145` | fail | 616/597 | 684/703 | 32 | Later latch collapses earlier. |
| `final155` | fail | 668/580 | 632/640 | 0 | Approaches the too-late latch failure. |

This rules out final-latch threshold sweeps for the current close-front
branch. The next useful bridge is to keep the early `1.20 m` final latch from
`rescue_crouch_abs040`, but give final hold a tiny nonzero scale so the
existing progress/lateral controllers can oppose drift without the runaway of
unscaled lateral correction.

## Close-Front Rescue Tiny-Final-Scale

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_tiny_final_scale_suite.sh`
- Purpose:
  keep `rescue_crouch_abs040` and the early final latch, then test very small
  final-hold command scales.
- Result:
  `fail`, 0/3 cases passed. Slurm job `169995` (`g1_cftiny`) ran on
  `server39` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_rescue_tiny_final_scale/20260707_g1_lowcarry_close_front_rescue_tiny_final_scale/close_front_rescue_tiny_final_scale_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `final_scale_003` | fail | 666/606 | 634/651 | 0 | Tiny command still destabilizes before useful dwell. |
| `final_scale_006` | fail | 657/642 | 643/658 | 7 | Slightly larger scale is still early collapse. |
| `final_scale_010` | fail | 577/548 | 723/752 | 49 | Best of the nonzero-scale cases but still drops and fails dwell. |

This rules out nonzero final-hold scale for the current close-front branch.
The best branch remains `rescue_crouch_abs040` with final scale `0.0`. The next
valid test is target-window joint-target freeze on top of that branch, because
the remaining failure is drift after the first target-window dwell.

## Close-Front Rescue Freeze

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_freeze_suite.sh`
- Purpose:
  keep `rescue_crouch_abs040`, final scale `0.0`, no runtime chest support,
  and add target-window joint-target freeze.
- Result:
  `fail`, 0/3 cases passed. Slurm job `169996` (`g1_cffreeze`) ran on
  `server59` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_rescue_freeze/20260707_g1_lowcarry_close_front_rescue_freeze/close_front_rescue_freeze_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Longest/End Streak | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `freeze_strict` | fail | 518/496 | 782/804 | 106 | 68/0 | Best freeze branch: good final travel/lateral, but roll collapse and later drop. |
| `freeze_loose` | fail | 577/0 | 723/- | 71 | 71/0 | Avoids drop but falls too early and misses stable-step gate. |
| `freeze_loose_zero_corr` | fail | 577/0 | 723/- | 71 | 71/0 | Same as loose freeze; zero corrections does not help. |

The most actionable close-front branch is now `freeze_strict`: it reaches the
window for 106 steps with good final travel/lateral error, then falls from roll
collapse around steps `780-790`. The next small test should keep
`freeze_strict` and increase roll/balance feedback authority.

## Close-Front Freeze-Balance Refine

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_balance_refine_suite.sh`
- Purpose:
  retain `freeze_strict` and test whether stronger balance feedback can catch
  the roll collapse.
- Result:
  `fail`, 0/3 cases passed. Slurm job `170003` (`g1_cfbal`) ran on `server39`
  and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_freeze_balance_refine/20260707_g1_lowcarry_close_front_freeze_balance_refine/close_front_freeze_balance_refine_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Main Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `roll_gain_010` | fail | 576/527 | 724/773 | 0 | Increased roll gain prevents useful target-window dwell. |
| `roll_gain_014` | fail | 626/574 | 674/704 | 42 | Stronger roll gain is worse. |
| `roll_pitch_gain` | fail | 407/376 | 853/924 | 44 | Delays fall but runs away far past the target. |

This rules out simply increasing balance gains. The next close-front test
should preserve default balance and use a delayed low-COM stand/hold transition
after target-window freeze.

## Close-Front Freeze-Stand Transition

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`
- Purpose:
  retain `freeze_strict`, default balance, and test whether delayed low-COM
  stand targets prevent the roll collapse after target-window freeze.
- Result:
  `fail`, 0/3 cases passed. Slurm job `170016` (`g1_cfstand2`) ran on
  `server20` and exited `FAILED 1:0`.
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_freeze_stand_transition/20260707_g1_lowcarry_close_front_freeze_stand_transition/close_front_freeze_stand_transition_summary.json`

| Case | Result | Fall/Drop | First Fall/Drop Step | Target Stable Steps | Longest/End Streak | Main Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `stand_delay_80` | fail | 585/546 | 715/741 | 68 | 68/0 | Too early/aggressive stand transition over-travels to about `3.7 m` and drops. |
| `stand_delay_120` | fail | 612/520 | 688/727 | 28 | 28/0 | Worse target-window retention and early fall/drop. |
| `stand_delay_160_soft` | fail | 518/496 | 782/804 | 106 | 68/0 | Reproduces the `freeze_strict` near miss; stand transition does not fix collapse. |

This rules out delayed stand target tuning for the current close-front branch.
The next useful isolation is freeze plus rescue timing: disable or delay the
post-freeze rescue posture to test whether rescue is causing the `780`-step
collapse rather than preventing it.

## Close-Front Freeze-Rescue Override Plan

A first timing-only entrypoint,
`scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_rescue_timing_suite.sh`,
was submitted as Slurm job `170095` (`g1_cfrtime`) and then cancelled before
allocation. Static control-flow inspection showed the timing-only intervention
would not change control behavior: once `final_freeze_active` is true, frozen
policy joint targets take priority and rescue targets are not applied.

The valid replacement is:

- Script:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_rescue_override_suite.sh`
- Code hook:
  `--agile-command-hold-rescue-overrides-final-freeze`
- Purpose:
  keep the `freeze_strict` target-window branch, but explicitly allow rescue
  targets to override frozen policy targets after rescue triggers.
- Cases:
  `freeze_no_rescue`, `freeze_rescue_late055`, and
  `freeze_rescue_soft035`.

This is not evidence yet. It is the next experiment entrypoint.

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
- do not trigger chest support/freeze immediately at first target-window entry
  or keep tuning chest-pad timing/geometry for close-front; v2 shortened the
  stable window and support-timing showed no-pad was least bad,
- do not move final latch later to `1.80 m`; late-hold collapsed before it
  could latch,
- do not continue lateral roll-target for close-front; rescue crouch is the
  useful controller hook, and its next issue is final lateral retention,
- do not unscale final-hold box-lateral correction; it causes runaway
  over-travel and box drops,
- do not continue final-latch threshold sweeps; moderate thresholds worsened
  fall/drop timing,
- do not use nonzero final-hold scale for this branch; tiny scales still
  caused early drop/collapse,
- continue from `freeze_strict` if pursuing close-front: it is now the best
  target-window branch, and the next failure is roll balance rather than target
  progress,
- do not continue stronger balance-gain tuning; it shortened useful
  target-window dwell,
- do not continue delayed stand target tuning unchanged; the best delayed
  stand case reproduced the same `freeze_strict` fall/drop boundary,
- do not run rescue timing without override; freeze masks rescue targets,
- isolate post-freeze rescue with explicit rescue-over-freeze override next,
- keep the same strict checks: fall/drop 0, no rollout root/box writes,
  final target-window hold, tilt bounds, and final lateral error bounds.
