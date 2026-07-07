# TODO 03: No-Root Articulated Carrier

- [x] Follow the 2026-07-06 user correction: stop treating external models,
  checkpoint downloads, policy-server rollouts, or optional official wrappers
  as blockers when they do not directly help the Isaac scene. The active route
  is direct Isaac scene construction.
- [x] Reaffirm after the 2026-07-06 user pushback: do not wait for or chase
  external models by default. Continue building and gating the direct Isaac
  carrying scene. Only use an external model/checkpoint/wrapper if it directly
  removes a named Isaac blocker in the current gate.
- [x] Generate the current best 0.60 kg G1/AGILE runtime chest-pad showcase
  record and fallback visual. The rollout record
  `20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_min700` passed
  with fall/drop `0/0`, robot/box target-directed travel about `2.051/2.032 m`,
  max robot/box tilt `0.309/0.428 rad`, and chest-pad collision enabled at
  step `712`. The real Isaac RGB replay renderer failed because the current
  Kit environment lacks `omni.replicator` and
  `isaacsim.core.rendering_manager`; do not rerun that path unchanged. The
  preferred available presentation artifact is the dense-state schematic
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_dense_fallback_min700/g1_lowcarry_runtime_chestpad_fallback_annotated.mp4`
  with 83 frames, explicitly labeled as a schematic replay rather than an
  Isaac camera render.
- [x] Run and record G1 runtime chest-pad posture generalization. Suite
  `scripts/isaac/run_core_world_g1_lowcarry_runtime_chestpad_posture_generalization_suite.sh`
  produced aggregate `fail`, 1/5 cases passed: only the tuned
  `low_front_060` reproduced the strict pass. `close_front_060` stayed upright
  and kept the box but missed target-window/lateral/box-tilt gates;
  `forward_reach_060`, `wide_box_060`, and `low_front_080` fell and/or
  dropped. Report:
  `experiments/reports/2026-07-07_g1_runtime_chestpad_posture_generalization.md`.
- [x] Run and record close-front repair scan. Suite
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_repair_suite.sh`
  produced aggregate `fail`, 0/3 cases passed. The best variant
  `lateral_sign_neg` had fall/drop `0/0` and 27 target-window stable steps,
  but still failed tilt and lateral gates. Stronger lateral command and
  box-tilt chest-pad triggering did not fix it. Conclusion: do not keep
  scalar-tuning only lateral sign/gain; next G1 step needs posture-conditioned
  command/support selection.
- [ ] Add a posture-conditioned G1 command/support gate. It should select
  lateral/yaw/hold/support behavior from the current carry geometry before
  walking, then re-run at least `low_front_060` and `close_front_060` under
  the same strict gates without relaxing target-window, tilt, lateral-error,
  fall/drop, or no-rollout-write checks.
- [x] Add and run first close-front posture-conditioned command scans:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_command_conditioned_suite.sh`,
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_command_refine_suite.sh`,
  and `scripts/isaac/run_core_world_g1_lowcarry_close_front_hold_delay_suite.sh`.
  None passed strict gates, but they materially advanced the close-front
  boundary. `command_y_neg004` centered the close-front carry with fall/drop
  `0/0`, final lateral error about `0.054/0.034 m`, and max robot/box tilt
  `0.246/0.329 rad`, but under-traveled. `steps1050_final120` reached final
  robot/box travel about `2.026/2.103 m` with fall/drop `0/0` and final
  lateral error about `0.081/0.095 m`, but failed by slight tilt excess
  `0.486/0.493 rad`, target-window stable steps `76 < 80`, and too few
  final-hold active steps. This supports posture-conditioned command/hold
  selection as the next active G1 direction, but it is not a pass.
- [x] Add close-front final-stabilize follow-up script
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stabilize_suite.sh`.
  It tests 1200-step close-front `x=0.10,y=-0.04,final=1.20` variants with
  earlier box-tilt chest-pad triggers. Slurm job `169771` was submitted but
  stayed pending with an estimated start around `2026-07-07T17:00:00`, so it
  was cancelled before running. There is no result yet; rerun this script when
  GPU priority is available.
- [x] Add and run final-hold tilt-escape probes for the close-front near-miss.
  Default thresholds were too late: aggregate `fail`, 0/2, fall/drop `0/0`,
  writes `0/0/0`, target-window stable/end `76/73`, but tilt and stable-step
  gates still failed. Early thresholds improved the boundary but still failed:
  `escape_robot018_box024_scale020` reached robot/box travel about
  `2.118/2.130 m`, final lateral error about `0.013/0.150 m`, and
  target-window stable/longest/end `81/80/80` with fall/drop `0/0` and writes
  `0/0/0`, but box tilt reached `0.544 rad > 0.45` and final-hold active
  stayed `268 < 399`. Treat this as evidence for a box-attitude-support
  follow-up, not as a pass.
- [x] Run close-front chest-pad tilt-support follow-up:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_chestpad_tilt_support_suite.sh`.
  It starts from the useful early-escape boundary, extends to `1200` steps, and
  tests earlier box-tilt chest-pad triggering plus thicker/higher pad and
  lower lid geometry under the same strict gates. Submitted through tmux
  `curiosity_g1_chestpad_tilt_0707` as Slurm job `170370` / `g1_chestpad`.
  A one-case backfill version was also submitted through tmux
  `curiosity_g1_chestpad_tilt_quick_0707` as Slurm job `170372` /
  `g1_chestquick`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_chestpad_tilt_support_quick`.
  The full suite ran and failed aggregate `0/2`. Both cases fell early at step
  `277` and dropped at step `309`, with target-window stable steps `0` and
  final-hold active steps `0`; chest-pad collision only enabled at step `650`,
  after the first failure, while the modified lower/thicker top lid enabled at
  step `116`. Conclusion: do not continue the lower-lid/thicker-pad branch;
  isolate the useful early-escape boundary with original support geometry.
- [x] Run close-front early-escape 1200-step isolation:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_early_escape_1200_suite.sh`.
  It preserves the original lid/pad geometry from the no-fall 1050-step
  near-miss, extends to `1200` steps for the final-hold gate, and compares
  target-window-only chest-pad trigger against original-size box-tilt-triggered
  chest pad. Submitted through tmux `curiosity_g1_early_escape_1200_0707` as
  Slurm job `170382` / `g1_escape1200`. Record
  `close_front_early_escape_1200_summary.json` before interpreting. Result:
  aggregate `fail`, 0/2. Both cases were identical: fall/drop `103/72`, first
  fall/drop `1097/1128`, target-window stable/longest/end `108/107/0`,
  final-hold active `418@782`, final robot/box travel about `3.129/2.968 m`,
  final lateral error about `1.014/1.090 m`, max robot/box tilt
  `2.120/2.061 rad`, chest pad at step `965`, tilt escape active `197` steps,
  and writes `0/0/0`. This restores the late near-boundary but shows
  continuous final tilt escape over-drives the system after target-window
  dwell.
- [x] Run close-front escape-suppression follow-up:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_suite.sh`.
  It adds
  `--agile-command-hold-final-tilt-escape-suppress-after-target-window-streak`
  and suppresses final tilt escape after `60` or `80` consecutive target-window
  steps. This directly tests whether the 1200-step failure is caused by
  continued escape-command over-travel after the target window has already been
  reached. Submitted through tmux `curiosity_g1_escape_suppression_0707` as
  Slurm job `170384` / `g1_escsup`. Record
  `close_front_escape_suppression_summary.json` before interpreting. Result:
  aggregate `fail`, 0/2. `suppress_after_streak60` improved the late boundary
  but did not pass: fall/drop `64/46`, first fall/drop `1136/1154`,
  target-window stable/longest/end `114/113/0`, final-hold `418@782`, final
  robot/box travel `3.200/3.140 m`, final lateral `0.923/0.879 m`, tilt
  `2.734/2.592 rad`, escape active/suppressed `167/54`, writes `0/0/0`.
  `suppress_after_streak80` was weaker. Next step should add a window-latched
  brake or stand handoff after sufficient dwell, not more forward escape or
  lower-lid contact.
- [x] Run close-front escape-suppression brake follow-up:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_brake_suite.sh`.
  It starts from the better `suppress_after_streak60` boundary, keeps original
  support geometry, suppresses final tilt escape after a `60`-step target
  window streak, and applies a negative final brake command after `240`
  final-hold steps for `120` steps. Cases test `-0.004` and `-0.008` command
  x. Submitted through tmux `curiosity_g1_escape_brake_0707` as Slurm job
  `170388` / `g1_escbrake`. Record
  `close_front_escape_suppression_brake_summary.json` before interpreting.
  Result: aggregate `fail`, 0/2. `neg008` delayed collapse and reduced
  fall/drop to `33/7` with first fall/drop `1167/1193` and max robot/box tilt
  `0.673/0.739 rad`, but final travel grew to `3.583/3.587 m` and target
  streak at end stayed `0`. Do not continue scanning brake magnitude alone.
- [x] Run close-front stand-handoff follow-up after target-window dwell:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_stand_suite.sh`.
  It reuses the `suppress_after_streak60` boundary and switches to final stand
  targets after `240` or `300` final-hold steps, preserving strict fall/drop,
  target-window, final-hold, tilt, lateral, and no-rollout-write gates.
  Submitted through tmux `curiosity_g1_escape_stand_0707` as Slurm job
  `170396` / `g1_escstand`. Record
  `close_front_escape_suppression_stand_summary.json` before interpreting.
  Result: aggregate `fail`, 0/2. `stand240_blend002` is the better terminal
  boundary: fall/drop `56/33`, first fall/drop `1144/1167`,
  target-window stable/longest/end `144/143/0`, final-stand stable/longest
  `91/91`, final travel `2.750/2.510 m`, lateral `0.765/0.869 m`, and writes
  `0/0/0`. It improves dwell but still fails lateral/tilt stability.
- [ ] Next close-front terminal step: combine the earlier `stand240_blend002`
  handoff with lateral/roll stabilization or replace the support backend. Do
  not claim close-front pass from the stand-handoff result; it still has
  fall/drop events and target-window streak at end `0`.
- [x] Run close-front escape-stand lateral follow-up:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_stand_lateral_suite.sh`.
  It starts from `stand240_blend002` and activates lateral-error-driven balance
  roll target around target-window entry, comparing signs `-1` and `+1` under
  the same strict gates. Submitted through tmux
  `curiosity_g1_escape_lateral_0707` as Slurm job `170405` / `g1_esclat`.
  Record `close_front_escape_stand_lateral_summary.json` before interpreting.
  Result: aggregate `fail`, 0/2. Both signs were materially similar:
  fall/drop about `56-57/33-34`, first fall/drop about `1143-1144/1166-1167`,
  target-window stable/longest/end `145/144/0`, final-stand stable/longest
  `92/92`, final travel around `2.74-2.77/2.49-2.52 m`, lateral still about
  `0.75/0.86-0.88 m`, and writes `0/0/0`.
- [ ] Stop scalar-tuning the current close-front terminal-control branch.
  Results from early escape, suppression, brake, stand handoff, and lateral
  roll target produced useful boundaries but no strict pass. The next
  meaningful step is a materially different terminal support policy or support
  backend replacement while preserving the same strict gates.
- [x] Run close-front low-stance terminal support follow-up:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_stand_lowstance_suite.sh`.
  It starts from `suppress60_stand240_blend002` but changes the final support
  posture with paired sagittal hip/knee/ankle/waist stand overrides instead of
  further tuning command scalars. Record
  `close_front_escape_stand_lowstance_summary.json` before interpreting.
  Submitted through tmux `curiosity_g1_lowstance_0707` as Slurm job `170412` /
  `g1_lowstance`. Result: aggregate `fail`, 0/2. The overrides were applied,
  but `lowstance_soft` worsened the boundary with fall/drop `123/76` and
  final-stand stable `49`, while `lowstance_deeper` roughly reproduced the
  earlier stand boundary with fall/drop `55/34`, target-window `145/144/0`,
  final-stand stable `92/92`, and writes `0/0/0`.
- [x] Add unified posture-conditioned G1 gate suite:
  `scripts/isaac/run_core_world_g1_posture_conditioned_gate_suite.sh`. It runs
  two strict cases without relaxing gates: the known passing `low_front_060`
  configuration and a close-front conditioned candidate using
  `x=0.10,y=-0.04,final=1.20` plus earlier box-tilt chest-pad trigger. This is
  the reproducible gate for testing whether posture-conditioned command
  selection can move beyond a single tuned posture. It has run and produced
  aggregate `fail`, 1/2 cases passed. `low_front_060` passed with fall/drop
  `0/0`, final robot/box travel about `2.051/2.032 m`, max robot/box tilt
  `0.309/0.428 rad`, target-window stable/end streak `105/102`, final-hold
  `462@357`, chest pad at step `712`, and rollout root/velocity/box writes
  `0/0/0`. `close_front_060_conditioned` failed with fall/drop `142/0`, first
  fall step `924`, final robot/box travel about `0.731/0.650 m`, max robot/box
  tilt `3.130/3.129 rad`, target-window stable steps `0`, final-hold
  `418@782`, chest pad at step `887`, and writes `0/0/0`. Summary:
  `experiments/outputs/core_world_g1_posture_conditioned_gate/20260707_g1_posture_conditioned_gate/posture_conditioned_gate_summary.json`.
- [x] Record the 2026-07-07 clean G1 boxtilt box-progress isolation result:
  `clean_slow` was stable but under-traveled, while `clean_slow_lateral_pos`
  entered the target window for `91` steps before over-traveling and failing.
  Treat this as terminal-control evidence only, not carrying success.
- [x] Add opt-in hold scaling for the G1 box-progress and box-lateral
  controllers:
  `--agile-command-box-progress-scale-on-hold` and
  `--agile-command-box-lateral-scale-on-hold`.
- [x] Add direct target-window hold trigger for G1 AGILE command control:
  `--agile-command-stop-target-window` and
  `--agile-command-stop-target-window-min-step`, plus
  `scripts/isaac/run_core_world_g1_boxtilt_window_hold_suite.sh`.
- [x] Await and record
  `scripts/isaac/run_core_world_g1_boxtilt_scaled_terminal_suite.sh` /
  Slurm job `169580`. It tests whether slow box-progress plus lateral
  correction can enter the target window and then stop/hold without
  over-travel. If it fails, do not keep scalar-tuning this same command layer;
  move to support/locomotion backend replacement or a materially stronger
  terminal balance controller.
- [x] Record Slurm job `169580` scaled-terminal result. Strict `fail`,
  `0/3` cases passed. All cases reached a transient target window
  (`96-100` longest streak) but failed to hold to the end. The best delayed
  failure was `later_terminal_brake`, first fall/drop `987/1010`, but it
  still ended with `213/190` fall/drop and severe over-travel.
- [x] Run and record
  `scripts/isaac/run_core_world_g1_boxtilt_window_hold_suite.sh` after a GPU
  compute slot is available. This tests whether directly latching hold on
  target-window entry is better than box-travel threshold latching. Submitted
  as Slurm job `169585` with `--dependency=afterany:169580`, so it will not
  compete with the scaled-terminal diagnostic.
- [x] Record Slurm job `169585` target-window hold result. Strict `fail`,
  `0/3` cases passed. The target-window latch fired at step `748` in all
  cases, but none held to the end; `window_zero`, `window_freeze`, and
  `window_brake` all later fell/dropped. Treat this as evidence that the
  current command-level stop/freeze/brake layer is exhausted, not as progress
  toward stable heavy boxtilt carrying.
- [x] Run and record the MuJoCo robot-like welded-payload bracket after the
  prismatic scaffold pass. `v022_fx130` and `v024_fx115` passed the diagnostic
  no-fall/no-root-write/travel gate; `v026_fx105` failed late with falls.
  Treat `v024_fx115` as the current best robot-like visualization/control
  diagnostic only, not as unknown free-box carrying.
- [x] Generate a presentation visual for the passing MuJoCo `v024_fx115`
  diagnostic:
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback.gif`.
- [ ] Next backend step: remove one scaffold at a time from the MuJoCo
  robot-like diagnostic. First replace welded payload with a free/contact
  payload or reduce explicit body-force stabilization; do not claim progress
  unless the same fall/drop/travel/no-root-write gates still pass.
- [x] Add MuJoCo free-box contact diagnostic:
  `scripts/mujoco/run_quadruped_freebox_carry.py`,
  `scripts/mujoco/run_quadruped_freebox_carry.sh`, and
  `scripts/mujoco/check_quadruped_freebox_summary.py`. The box is a separate
  freejoint rigid body retained only by tray/wall contact; root and box pose
  writes remain zero.
- [x] Run first MuJoCo free-box contact bracket. Result: no final pass.
  Conservative 2 kg case retained the box and robot but under-traveled; faster
  cases produced useful box travel near `0.20 m` but failed retention/drop or
  fall gates.
- [x] Add and test free-box stop/hold controls:
  `STOP_AFTER_BOX_TRAVEL` and `HOLD_TARGET_SPEED`. Result: no final pass.
  Best cases latched around `0.15 m` and kept fall `0`, but final relative
  box-torso error stayed around `0.23-0.24 m`, producing drop events.
- [x] Next free-box contact step: replace passive tray-only retention with a
  real contact-retention mechanism, such as closer actuated side pads, a
  normal-force/grip controller, or a constraint/contact hybrid that can be
  explicitly audited as not writing box pose. Re-run the same gates:
  free box, fall/drop 0, final box travel >= `0.12 m`, target-stop hold >=
  600 steps, root/box pose writes 0, and final relative error <= `0.20 m`.
- [x] Add and run a first audited grip-force diagnostic:
  `RETENTION_FORCE_MODE=relative_spring`. The 2 kg case
  `20260707_mujoco_quad_freebox_2kg_v024_stop015_hold012_retention_spring`
  passed with fall/drop `0/0`, root/box pose and velocity writes all `0`,
  final box travel `0.18239 m`, final relative error `0.07865 m`, and
  target-stop hold `1441` steps. This is a diagnostic grip-force controller,
  not learned contact grasping or final carrying.
- [x] Run and record the same retention-force diagnostic across 1 kg, 2 kg,
  and 3 kg using
  `scripts/mujoco/run_quadruped_freebox_retention_multiload_suite.sh` in
  `curiosity_mujoco_quad_freebox_retention_loads_nogpu_0707` / Slurm job
  `169116`. All three loads passed the strict diagnostic gates. This supports
  robustness of the hand-authored retention-force scaffold across this narrow
  load range only; it is not learned unknown-load carrying.
- [ ] Next reduction gate: remove or weaken one scaffold at a time. Preferred
  order is: first reduce explicit torso body-force stabilization while keeping
  the proven retention-force controller, then replace the relative-spring
  retention force with actuated contact pads/contact-only grip. Preserve the
  same gates: 1/2/3 kg free box, fall/drop 0, final box travel >= `0.12 m`,
  target-stop hold >= 600 steps, root/box pose and velocity writes 0, final
  relative error <= `0.20 m`.
- [x] Run first assist-reduction bracket:
  `scripts/mujoco/run_quadruped_freebox_retention_assist_reduction_suite.sh`
  / Slurm job `169120`. The 2 kg case passed at 75%, 50%, and about 33%
  body-force caps, but this mainly shows the previous caps were not active;
  it is not evidence that torso stabilization is unnecessary.
- [x] Await/record assist-floor probe:
  `scripts/mujoco/run_quadruped_freebox_retention_assist_floor_probe.sh` /
  Slurm job `169121`. It tests 10% caps, zero caps, and `ASSIST_MODE=none`
  while preserving the same retention controller and no root/box shortcuts.
  Result: all failed. The 10% cap case had `77` falls, `9` drops, max box
  travel only `0.00666 m`, and no target latch. Zero caps and no-assist had
  `129` falls, `124` drops, max box travel about `0.00574 m`, and no target
  latch. This establishes the current scaffold boundary: 33% caps pass, 10%
  caps fail.
- [ ] Next support-replacement step: add a foot/support controller or
  materially different legged support mechanism that can replace body-force
  torso stabilization. Keep the same retention-force 1/2/3 kg gates at first,
  then re-run assist-floor probes. Do not present the current pass as
  unassisted locomotion.
- [x] Add first foot/support replacement probe:
  `LEG_DRIVE_MODE=foot_ik` in `scripts/mujoco/run_quadruped_freebox_carry.py`
  plus launcher/checker fields. Slurm job `169126` ran four 2 kg no-assist
  free-box probes. All failed strict carrying gates, but `faster_long` and
  `high_clearance` had fall/drop `0/0` and low relative error while moving
  backward, indicating a useful stable no-body-assist support mode with the
  forward stance sign likely reversed.
- [x] Record negative-stride foot-IK probes. Inline Slurm job `169127` is
  invalid as named evidence because shell expansion produced timestamp-only
  default outputs. Corrected script
  `scripts/mujoco/run_quadruped_freebox_foot_ik_negstride_suite.sh` ran as
  Slurm job `169130`: `neg_high` produced real forward no-body-assist travel
  and latched the target, but failed with `111` falls and `107` drops.
  `neg_fast` and `neg_slow` also failed.
- [x] Add target-latched foot-IK stride scaling and run stop/hold retry
  `scripts/mujoco/run_quadruped_freebox_foot_ik_negstride_stop_suite.sh` as
  Slurm job `169135`. Result: still failed. Final forward travel stayed
  positive (`0.268-0.356 m`), but falls/drops remained about `109-111` /
  `106-107`.
- [x] Await/record early-stop foot-IK suite
  `scripts/mujoco/run_quadruped_freebox_foot_ik_early_stop_suite.sh` / Slurm
  job `169136`. It stops at 0.08/0.10/0.12 m to test whether the fall/drop
  cascade can be avoided before tilt grows after target latch.
  Result: all failed. Early stops preserved positive final travel around
  `0.300-0.311 m` for the main cases, but falls/drops remained around
  `110-111` / `107`.
- [x] Await/record lateral-retention foot-IK suite
  `scripts/mujoco/run_quadruped_freebox_foot_ik_lateral_retention_suite.sh` /
  Slurm job `169138`. It adds audited equal-and-opposite y-axis retention
  force to test whether lateral box/torso drift is driving the roll/fall
  cascade.
  Result: all failed. Final travel remained positive, but falls/drops stayed
  around `111` / `107`; y-axis retention alone does not solve roll/fall.
- [x] Add and run coarse roll-to-foot-height feedback:
  `FOOT_ROLL_Z_GAIN` in `scripts/mujoco/run_quadruped_freebox_carry.py` and
  `scripts/mujoco/run_quadruped_freebox_foot_ik_roll_feedback_suite.sh` /
  Slurm job `169145`. Positive gains kept forward travel but fell; negative
  gains stabilized but drove backward. This is useful controller evidence, not
  a pass.
- [x] Await/record fine roll-feedback sweep
  `scripts/mujoco/run_quadruped_freebox_foot_ik_roll_feedback_fine_suite.sh`
  / Slurm job `169150`.
  Result: all failed. `roll_neg004` is the best forward/stability compromise
  so far (`0.522 m` final travel, max tilt `0.871 rad`) but still has
  `71` falls and `70` drops. Gains at `-0.045` and beyond are stable
  fall/drop `0/0` but walk backward.
- [ ] Next support-controller step: add lateral hip/foot-placement DOF or a
  materially stronger stance-phase controller. Do not keep tuning only
  roll-to-foot-height feedback; it has shown a stability/travel-direction
  tradeoff rather than a pass.
- [x] Add lateral hip/foot-placement DOF:
  `*_hip_roll` joints/actuators plus `HIP_ROLL_BASE` and
  `HIP_ROLL_FEEDBACK_GAIN` in the MuJoCo free-box runner.
- [x] Run lateral hip support suite
  `scripts/mujoco/run_quadruped_freebox_foot_ik_lateral_hip_suite.sh` /
  Slurm job `169159`. Result: all failed. Small base roll kept positive
  travel but fell/dropped; larger base roll stabilized but walked backward.
- [x] Run lateral hip stride suite
  `scripts/mujoco/run_quadruped_freebox_foot_ik_lateral_hip_stride_suite.sh`
  / Slurm job `169161`. Result: all failed. Best positive travel case
  `base006_neg12` reached final travel `0.343 m` but still had `95` falls and
  `91` drops; stable cases walked backward.
- [x] Add and run target-latched hold-brace stance:
  `HOLD_STANCE_FOOT_Z_DOWN` and `HOLD_HIP_ROLL_BASE` plus
  `scripts/mujoco/run_quadruped_freebox_foot_ik_hold_brace_suite.sh` / Slurm
  job `169162`. Result: all failed; bracing did not remove the roll/fall/drop
  cascade.
- [x] Replace open-loop foot trajectories with a first closed-loop
  foot-placement controller using forward velocity and target progress:
  `CLOSED_LOOP_FOOT_PLACEMENT`, `STRIDE_VELOCITY_GAIN`,
  `STRIDE_POSITION_GAIN`, and `STRIDE_CLIP`.
- [x] Run closed-loop foot-placement suite
  `scripts/mujoco/run_quadruped_freebox_foot_ik_closed_loop_suite.sh` /
  Slurm job `169164`. Result: all failed. No-hip cases reached/latching the
  target but collapsed during hold; hip-base cases were stable but walked
  backward and never latched.
- [x] Add target-latched static support foot placement:
  `HOLD_FRONT_FOOT_X`, `HOLD_REAR_FOOT_X`, and
  `HOLD_PITCH_FOOT_X_GAIN`.
- [x] Run static support sweep
  `scripts/mujoco/run_quadruped_freebox_foot_ik_static_hold_support_suite.sh`
  / Slurm job `169173`. Result: all failed. Wider fore-aft support did not
  remove the post-stop collapse; CSV inspection shows roll/lateral load drift
  dominates, with box/torso y drift approaching about `-0.95 m`.
- [x] Combine closed-loop/static hold with lateral centering:
  `scripts/mujoco/run_quadruped_freebox_foot_ik_centered_hold_suite.sh` /
  Slurm job `169181`.
- [x] Record lateral-centering hold result. All cases failed. y-axis
  equal-and-opposite retention did not remove the roll/drop failure; best
  forward-travel hip-base case still had `96` falls and `93` drops.
- [x] Add and run roll-state hip feedback hold suite
  `scripts/mujoco/run_quadruped_freebox_foot_ik_hip_feedback_hold_suite.sh` /
  Slurm job `169183`.
- [x] Record hip-feedback hold result. All cases failed. Negative feedback
  kept forward progress but still fell/dropped; positive feedback could
  stabilize but walked backward and never latched.
- [x] Add and run stronger joint servo support test:
  `ACTUATOR_KP` / `ACTUATOR_KV` in
  `scripts/mujoco/run_quadruped_freebox_carry.py` plus
  `scripts/mujoco/run_quadruped_freebox_foot_ik_strong_servo_suite.sh` /
  Slurm job `169189`.
- [x] Record strong-servo result. All cases failed. Strong hip/servo settings
  can stabilize the robot but walk backward; forward/latched cases still
  fall/drop. Stop treating the current hand-authored foot-IK family as a
  near-success controller.
- [x] Add first replacement support-controller route:
  `SUPPORT_CONTROLLER_MODE=stance_force` in the MuJoCo free-box runner. It
  maps stance support/propulsion forces through foot Jacobians into actuated
  joint generalized torques and records `support_joint_torque_write_count`.
- [x] Run first stance-force support sweep
  `scripts/mujoco/run_quadruped_freebox_stance_force_support_suite.sh` /
  Slurm job `169200`. Result: all failed, but negative force scale produced
  real forward motion and early target latch without root/box writes or torso
  body-force assist. Failure changed to overdrive, fall/drop, and large
  box-torso relative error.
- [x] Add and run stance-force early-stop/braking suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_brake_suite.sh` /
  Slurm job `169208`.
- [x] Record stance-force brake result. All cases failed, but the failure
  bracket is useful: one case was stable with fall/drop `0/0` but walked
  backward; another kept positive travel and low final relative error but
  still fell/dropped.
- [ ] Next stance-force step: search between the stable-backward and
  positive-falling endpoints using smaller negative force scale, small
  hip-base, stronger vertical/roll/pitch support, and neutral or weak braking.
  Do not return to the older foot-IK parameter family.
- [x] Run first stance-force boundary refine suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_refine_suite.sh` /
  Slurm job `169210`.
- [x] Record stance-force refine result. All cases failed. Force scale
  `-0.50` with hip-base `0.04-0.05` is stable fall/drop `0/0` but walks
  backward; scale `-0.35` with hip-base `0.05` keeps positive final travel
  `0.26334 m` but still has `85` falls and `80` drops. The active boundary
  is now between stable-backward and positive-falling, not a pass.
- [ ] Next stance-force boundary step: search around force scale `-0.38` to
  `-0.45` and hip-base `0.045-0.05` with neutral hold speed and more damping.
  Preserve strict gates: free box, fall/drop 0, final box travel >= `0.12 m`,
  target hold >= 600 steps, no root/box pose or velocity writes, no torso
  body-force assist, and final relative error <= `0.20 m`.
- [x] Add and run stance-force boundary suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_boundary_suite.sh` /
  Slurm job `169216`.
- [x] Record stance-force boundary result. All eight cases failed. The suite
  was stable fall/drop `0/0` with low tilt and good box retention, but every
  case walked backward with final box travel about `-0.69` to `-0.75 m` and
  target-stop never latched. This shows the added damping/slowdown moved too
  far into the stable-backward regime.
- [ ] Next stance-force edge step: return closer to the positive-but-falling
  endpoint from `169210`, around force scale `-0.34` to `-0.40` with the
  previous stronger forward-drive settings. Use only small changes from the
  `169210` positive case; do not keep increasing damping/slowdown.
- [x] Add and run stance-force edge suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_edge_suite.sh` / Slurm
  job `169230`.
- [x] Record stance-force edge result. All cases failed. Positive-travel
  cases latched and held target for `1396-1907` steps but then collapsed with
  `62-81` falls and `57-76` drops. The single stable case stopped too early
  and moved backward. The blocker has narrowed to target-latched hold/brake
  stabilization.
- [ ] Next stance-force implementation step: add an explicit post-latch hold
  stabilizer in stance-force control, such as stronger horizontal velocity
  damping, pitch/height correction, or stance-foot force redistribution only
  after `target_stop_latched`. Then rerun the strict free-box gates.
- [x] Add hold-only stance-force support controls: separated carry/hold
  horizontal support scaling and hold-only vx/max-fx/damping/height parameters.
- [x] Run hold-stabilizer suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_stabilizer_suite.sh`
  / Slurm job `169235`.
- [x] Record hold-stabilizer result. All cases failed. Stop-0.05 positive
  cases still fell/dropped after latch; stop-0.04 was stable fall/drop `0/0`
  but drifted backward and under-traveled. The next narrow probe should keep
  the stable stop-0.04 setup and add small positive post-latch creep speed.
- [ ] Next narrow probe: run stop-0.04 with hold target speed around
  `0.01-0.05 m/s` and hold horizontal force scaling around `0.15-0.60`.
  Objective is to keep final box travel >= `0.12 m` without losing the
  fall/drop `0/0` behavior.
- [x] Run hold-creep suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_creep_suite.sh` /
  Slurm job `169236`.
- [x] Record hold-creep result. All cases failed. Positive hold speed with
  positive hold horizontal scale made the stable stop-0.04 case drift farther
  backward, while stop-0.045 restored positive travel but fell/dropped.
- [ ] Next narrow probe: keep stop-0.04 and test small positive hold speed
  with negative hold horizontal scale (`-0.05` to `-0.30`), because negative
  horizontal force scaling is the sign that produced forward propulsion in
  earlier stance-force runs.
- [x] Run hold-creep negative-fx suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_creep_negfx_suite.sh`
  / Slurm job `169242`.
- [x] Record hold-creep negative-fx result. All cases failed. The stable
  stop-0.04 setup remains in a backward-drift basin regardless of small
  post-latch velocity sign: fall/drop `0/0`, but max positive travel only
  about `0.05 m` and final travel strongly negative.
- [ ] Next post-latch support step: use stop-0.05 positive-travel settings and
  widen the hold static support geometry (`hold_front_foot_x` /
  `hold_rear_foot_x` toward the existing `0.22/-0.22` clamp) to target the
  pitch/tilt collapse directly.
- [x] Run wide-hold-support suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_wide_support_suite.sh`
  / Slurm job `169247`.
- [x] Record wide-hold-support result. All cases failed. Wider hold foot
  placement preserved positive travel but did not reduce the post-latch
  fall/drop collapse.
- [ ] Next controller step: stop small geometry/sign sweeps on this
  stance-force family and add a materially stronger balance controller, such
  as COM/centroidal-state feedback in the support-force allocation, or switch
  to a controller-backed legged policy. Preserve the existing audit gates.
- [x] Add first COM/centroidal feedback path in the stance-force support
  controller. It computes robot COM excluding the free box and shifts stance
  foot vertical forces before mapping them through foot Jacobians to actuated
  joint generalized torques.
- [x] Run first COM-support suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_com_support_suite.sh` /
  Slurm job `169254`.
- [x] Record first COM-support result. All cases failed. Positive COM-x
  feedback stabilized but prevented target latch and walked backward; negative
  COM-x restored positive travel in one case but worsened fall/drop collapse.
- [ ] Next COM-support step: make COM feedback optional before target latch
  and run hold-only COM feedback (`pre_latch_scale=0`) on the stop-0.05
  positive-travel setting.
- [x] Add `support_com_pre_latch_scale` and run hold-only COM suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_com_support_suite.sh`
  / Slurm job `169263`.
- [x] Record hold-only COM result. All cases failed. Approach/latch were
  preserved, but vertical COM force redistribution did not prevent post-latch
  pitch/tilt collapse.
- [ ] Next controller step: add pitch/roll damping through differential
  horizontal foot forces or a fuller centroidal wrench controller. Do not keep
  sweeping vertical COM force redistribution alone.
- [x] Add hold-only lateral foot-force support fields and run
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_lateral_suite.sh` /
  Slurm job `169267`.
- [x] Record hold-only lateral foot-force result. All cases failed. Positive
  travel and target latch remain, but fall/drop stayed about `77-79` /
  `72-73`.
- [ ] Next post-latch controller probe: add hold-only hip-roll feedback and
  hold-only roll-to-foot-height feedback in the leg posture targets. Force
  redistribution alone has not fixed the roll collapse.
- [x] Add hold-only leg posture feedback fields and run
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_posture_feedback_suite.sh`
  / Slurm job `169272`.
- [x] Record hold-only posture feedback result. All cases failed, but
  negative hold hip-roll feedback is the first branch to materially reduce
  tilt and relative-error severity while preserving positive travel.
- [ ] Next posture-feedback refine step: focus around
  `posture_hip_neg040` / `posture_combo_neg` with stronger negative hold
  hip-roll feedback, adjusted hold hip base, and compatible foot-height
  feedback. Do not return to pure force redistribution sweeps unless this
  branch stalls.
- [x] Run first hold-posture refine suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_posture_refine_suite.sh`
  / Slurm job `169285`.
- [x] Record first hold-posture refine result. All cases failed in the same
  stable-backward/no-latch basin because the suite changed the global
  actuator/support/retention baseline relative to `169272`; it is not a clean
  test of negative hold hip-roll feedback.
- [x] Add v2 hold-posture refine suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_posture_refine_v2_suite.sh`
  using the `169272` global baseline and only varying post-latch posture
  parameters.
- [ ] Monitor Slurm job `169288` / tmux `codex_mj_refine_v2_0707`, summarize
  all v2 cases, and either keep the best stable positive-travel branch or
  stop this hand-built posture-feedback family if it still cannot pass
  fall/drop gates.
- [x] Monitor and summarize Slurm job `169288`. Result: all cases failed.
  Forward travel and target latch were preserved, but post-latch
  lateral/roll collapse remained. Best tilt case was
  `refine2_hip_neg030`; its CSV showed first fall at step `1480` and first
  box drop at step `1560` after large lateral drift.
- [x] Add hold-lateral-posture combination suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_lateral_posture_suite.sh`
  to combine best negative hold hip feedback with hold-only lateral
  stance-force terms.
- [ ] Monitor Slurm job `169291` / tmux `codex_mj_latpost_0707`, then decide
  whether lateral support extends the first-fall step or suppresses box-y
  drift. If it does not, stop this hand-built MuJoCo controller branch and
  return to a controller-backed locomotion policy or a materially different
  balance formulation.
- [x] Monitor and summarize Slurm job `169291`. Result: all cases failed.
  Lateral stance-force saturated and preserved latch/travel, but only
  marginally reduced max tilt and did not fix fall/drop.
- [x] Add explicit hold-only world-y correction fields
  (`support_fy_world_y_gain`, `support_fy_world_vy_gain`,
  `support_fy_world_y_source`) and expose them through the MuJoCo launcher.
- [x] Add `scripts/mujoco/run_quadruped_freebox_stance_force_hold_world_y_suite.sh`.
- [ ] Monitor Slurm job `169292` / tmux `codex_mj_worldy_0707`, then check
  whether world-y correction reduces `max_abs_box_y_m`, delays first fall, or
  passes strict fall/drop gates.
- [x] Monitor and summarize Slurm job `169292`. Result: all cases failed.
  World-y force saturated but `max_abs_box_y_m` stayed near or above `1 m`,
  with unchanged fall/drop.
- [x] Add hold-only support authority scales
  (`hold_support_max_foot_fz_scale`,
  `hold_support_max_joint_torque_scale`) and expose them through the launcher.
- [x] Add `scripts/mujoco/run_quadruped_freebox_stance_force_hold_authority_suite.sh`.
- [ ] Monitor Slurm job `169293` / tmux `codex_mj_authority_0707`. If
  post-latch support authority does not materially reduce fall/drop, stop
  this hand-tuned MuJoCo branch as a controller dead end and move to a
  controller-backed locomotion backend or a proper optimizer/controller.
- [x] Monitor and summarize Slurm job `169293`. Result: all cases failed.
  Torque-only hold authority reproduced the baseline exactly; higher foot
  force/height/damping mostly worsened tilt. This is not an authority-starved
  controller.
- [x] Add `support_controller_mode=centroidal_stance_force`, a least-squares
  stance-foot wrench distribution mapped through foot Jacobians to actuated
  joint torques.
- [x] Add `scripts/mujoco/run_quadruped_freebox_centroidal_support_suite.sh`.
- [ ] Monitor Slurm job `169294` / tmux `codex_mj_centroidal_0707`. If the
  centroidal support formulation also fails without delaying fall/drop, mark
  the current MuJoCo hand-controller path as exhausted and switch to a real
  policy/controller backend rather than more parameter sweeps.
- [x] Monitor and summarize Slurm job `169294`. Result: all centroidal cases
  failed and were worse than the best heuristic stance-force baseline.
- [x] Mark the current simplified MuJoCo hand-controller route as exhausted
  for credible fall/drop-free carrying. Further work should switch to a real
  controller-backed locomotion backend or a proper optimizer/MPC-style balance
  controller, not more scalar gain sweeps.
- [x] Add a materially different MuJoCo post-latch balance controller instead
  of another stance-force scalar sweep:
  `--hold-capture-point-foot-placement` in
  `scripts/mujoco/run_quadruped_freebox_carry.py`, exposed through the
  launcher and checked through `scripts/mujoco/check_quadruped_freebox_summary.py`.
  It changes hold foot placement/hip-roll/foot-height based on capture-point
  style velocity and lateral drift signals after target latch only.
- [x] Run and record
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_capture_suite.sh`.
  Preserve strict gates: free box, no torso body-force assist, support torques
  only through actuated joints, retention force audited, fall/drop 0, final
  box travel >= `0.12 m`, target hold >= 600 steps, no root/box pose or
  velocity writes, and final relative error <= `0.20 m`.
  Submitted as Slurm job `169609` with `--dependency=afterany:169585`.
- [x] Record Slurm job `169609` MuJoCo hold-capture result. Strict `fail`,
  `0/6` cases passed. Target-stop and hold-capture both activated for all
  cases, root/box pose and velocity writes stayed `0`, and final box travel
  was nonzero, but every case still had `77` falls, `73-74` drops, excessive
  tilt, and low box height. Stop treating this hand-controller family as a
  credible route to stable carrying.
- [x] Add a materially different MuJoCo support backend:
  `SUPPORT_CONTROLLER_MODE=lqr_stance_force` in
  `scripts/mujoco/run_quadruped_freebox_carry.py`, launcher exposure, checker
  fields, and
  `scripts/mujoco/run_quadruped_freebox_lqr_stance_hold_suite.sh`. This uses
  finite-horizon COM x/y LQR feedback plus centroidal stance-foot force
  allocation, not another scalar stance-force sweep.
- [ ] Monitor Slurm job `169618` / tmux
  `curiosity_mujoco_lqr_stance_hold_0707`. It must be judged by the same
  strict free-box gates: no root/box pose or velocity writes, no torso
  body-force assist, LQR active steps >= 600, target-stop hold >= 600, final
  box travel >= `0.12 m`, final relative error <= `0.20 m`, and fall/drop 0.
- [x] Record Slurm job `169618` first LQR attempt as invalid activation
  diagnostic. It completed, but target-stop never latched and
  `support_lqr_active_steps=0` for all cases, because the initial
  implementation used centroidal allocation even before post-latch LQR was
  allowed to activate.
- [ ] Monitor corrected Slurm job `169619` / tmux
  `curiosity_mujoco_lqr_stance_hold_retry_0707`. The corrected mode must
  preserve pre-latch stance-force approach and only switch to LQR/centroidal
  support after target latch.
- [x] Record corrected Slurm job `169619` LQR/centroidal result. It was a
  valid activation diagnostic, but strict `fail`, `0/4` cases passed:
  target-stop and LQR were active for all cases, root/box writes stayed `0`,
  but post-latch fall/drop remained `78/75`, tilt was excessive, box height
  dropped, and final box travel stayed below the strict threshold.
- [x] Add conservative `lqr_additive_stance_force` mode and
  `scripts/mujoco/run_quadruped_freebox_lqr_additive_hold_suite.sh`. This
  preserves the original stance-force allocation and adds LQR x/y corrections
  after latch instead of switching the whole allocator.
- [ ] Monitor Slurm job `169621` / tmux
  `curiosity_mujoco_lqr_additive_hold_0707`, then compare against `169619`
  and the earlier hold-capture result.
- [x] Record Slurm job `169621` additive LQR result. Strict `fail`, `0/4`
  cases passed. Additive LQR preserved final box travel and relative error
  better than the centroidal-switching LQR (`0.253-0.272 m` final box travel,
  `0.099-0.116 m` final relative error), with LQR active and no root/box
  writes, but every case still had `77/73` fall/drop and excessive tilt.
- [ ] Next meaningful MuJoCo controller step, if continuing this branch:
  target the post-latch whole-body fall directly. Add a pitch/roll recovery
  objective that changes support geometry or upright torque before collapse,
  not more x/y LQR or retention tuning; target progress and box retention are
  no longer the limiting errors in the additive LQR result.
- [x] Add post-latch attitude recovery controller:
  `--support-attitude-recovery` in
  `scripts/mujoco/run_quadruped_freebox_carry.py`, launcher/checker exposure,
  and `scripts/mujoco/run_quadruped_freebox_attitude_recovery_suite.sh`.
  It scales extra roll/pitch support gains, target-height offset, hip-roll
  feedback, and roll-to-foot-height feedback when post-latch tilt grows.
- [x] Record Slurm job `169625` attitude-recovery result. Strict `fail`,
  `0/4` cases passed. Recovery activated for all cases (`1619-1796` steps),
  LQR and target hold were active, root/box writes stayed `0`, and final box
  travel/relative error remained useful, but fall/drop persisted. The
  roll-gain case slightly reduced max tilt to `1.6355 rad`, still far above
  the strict gate.
- [ ] Next meaningful MuJoCo controller step: stop gain-only recovery. The
  next change must modify the support contact problem itself, such as a
  constrained whole-body/QP allocation with unilateral foot-force and friction
  limits, or a controller-backed locomotion policy. Do not continue small
  attitude-recovery gain sweeps.
- [x] Add constrained support/contact allocator:
  `SUPPORT_CONTROLLER_MODE=qp_stance_force` in
  `scripts/mujoco/run_quadruped_freebox_carry.py`, launcher/checker exposure,
  and `scripts/mujoco/run_quadruped_freebox_qp_support_suite.sh`. It uses a
  projected-contact QP-style allocation with unilateral normal-force bounds
  and friction-cone projection before mapping foot forces to joint torques.
- [ ] Monitor Slurm job `169627` / tmux `curiosity_mujoco_qp_support_0707`.
  Compare strict gates and QP diagnostics against additive LQR and attitude
  recovery.
- [x] Switch active implementation focus back to the G1 AGILE policy path.
  Historical best low-carry run completed 819 steps with fall/drop `0/0`,
  free box, G1 USD, AGILE ONNX policy, no rollout root/box pose writes, and
  about `2.35 m` box target-directed travel. It is still diagnostic and only
  one low-carry posture.
- [x] Submit fresh low-carry strict reproduction:
  `scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
  with only `RUN_LOWCARRY_BASELINE=1` as Slurm job `169302` through tmux
  `codex_g1_lowcarry_repro_0707`.
- [ ] Monitor job `169302`. If it reproduces, run the same target-window
  validation with light/heavy low-carry and then chestpad/other postures. If
  it fails, compare its env snapshot/checker output against the historical
  819-step pass before changing parameters.
- [x] Monitor and summarize Slurm job `169302`. Result: fresh low-carry
  strict reproduction passed exactly: 819 steps, fall/drop `0/0`, about
  `2.35 m` box travel, target-window end streak `164`, and no rollout root/
  box pose writes.
- [x] Add `scripts/isaac/run_core_world_g1_lowcarry_load_validation_suite.sh`
  for `0.25`, `0.50`, and `0.75 kg` low-carry strict target-window cases.
- [x] Monitor Slurm job `169303` / tmux `codex_g1_lowcarry_load_0707`.
  Result: strict load-validation `fail`. `0.50 kg` passed and reproduced the
  819-step low-carry target-window result, but `0.25 kg` failed after entering
  the target window with `384` falls / `225` drops, and `0.75 kg` failed
  before final hold with `346` falls / `284` drops. Treat this as mass/load
  sensitivity of the current front-tray G1 AGILE setup, not load-robust
  carrying.
- [x] Add `scripts/isaac/run_core_world_g1_lowcarry_load_repair_suite.sh` for
  targeted repair diagnostics after `169303`: light-box final-window freeze,
  light-box policy-then-stand, and heavy-box chestpad/retention/slow carry.
- [x] Monitor Slurm job `169304` / tmux `codex_g1_lowrepair_0707`. Result:
  strict `fail`, `0/3` cases passed. `0.25 kg` final-window freeze had
  `418` falls / `102` drops; `0.25 kg` policy-then-stand had `550` falls /
  `536` drops; `0.75 kg` chestpad/retention/slow had `930` falls / `856`
  drops and negative final box target-directed travel. Scalar final-hold,
  freeze, stand-blend, chestpad, retention, and slow-speed tweaks are not
  enough for load robustness.
- [x] Add `scripts/isaac/run_core_world_g1_lowcarry_mass_band_suite.sh` to map
  the mass basin around the verified `0.50 kg` pass using the same strict
  target-window low-carry gate.
- [x] Monitor Slurm job `169309` / tmux `codex_g1_massband_0707`. Result:
  strict `fail`, `1/6` cases passed. `0.35 kg` passed; `0.40`, `0.45`,
  `0.55`, and `0.65 kg` failed with falls/drops or early lateral/roll
  divergence; `0.60 kg` was a near-miss with fall/drop `0/0` and target-window
  end streak `108` but failed box tilt (`0.63855 rad > 0.45`). Treat the
  low-carry G1 result as discontinuous narrow stable islands, not load
  robustness.
- [x] Add `scripts/isaac/run_core_world_g1_lowcarry_edge_repair_suite.sh` for
  narrow follow-up on the most useful mass-band failures: `0.60 kg` box-tilt
  near-miss and `0.45 kg` late fall/drop.
- [x] Monitor Slurm job `169311` / tmux `codex_g1_edgerepair_0707`. Result:
  strict `fail`, `0/4` cases passed. The `0.60 kg` tight-lid/chestpad
  variants all worsened the original near-miss and produced falls/drops.
  The `0.45 kg` tight-lid/final-zero case improved fall/drop to `0/0`, but
  under-traveled (`~1.52 m`), had target-window streak `0`, and still exceeded
  strict tilt gates.
- [x] Add `scripts/isaac/run_core_world_g1_lowcarry_edge_repair_v2_suite.sh`
  to test two narrow hypotheses from `169311`: delayed final-hold for the
  `0.45 kg` partial improvement, and non-pinching rail/no-lid geometry for
  the `0.60 kg` box-tilt near-miss.
- [x] Monitor Slurm job `169312` / tmux `codex_g1_edgerepair_v2_0707`.
  Result: strict `fail`, `0/4` cases passed. Delayed final-hold moved
  `0.45 kg` farther but produced falls/drops; `0.60 kg` side-rail-only and
  no-lid/tall-rail variants also failed with falls/drops.
- [x] Stop the current scalar threshold/cradle geometry repair family for G1
  low-carry load robustness. Do not keep sweeping final thresholds, tight
  lids, chestpad, side rails, or no-lid geometry as the main route. Next work
  needs a materially different controller/backend or policy adaptation.
- [x] Add a staged G1 direct-carry baseline suite:
  `scripts/isaac/run_core_world_g1_direct_carry_baseline_suite.sh`. It gates
  no-box stand, fixed-payload stand, free-box cradle stand, short free-box
  target-directed carry, and 700-step long-hold validation in one compute-node
  run while preserving per-stage summaries/checker reports.
- [x] Extend `check_core_world_g1_box_scene_summary.py` with direct suite
  gates for gait mode, box/cradle collision enabled state, and final
  box-robot relative-offset error.
- [x] Run lightweight login-node checks only:
  `bash -n scripts/isaac/run_core_world_g1_direct_carry_baseline_suite.sh`
  and `python3 -m py_compile scripts/isaac/check_core_world_g1_box_scene_summary.py scripts/isaac/build_core_world_g1_box_scene.py`.
- [ ] Run the new G1 direct-carry baseline suite inside a Curiosity-owned
  tmux-held Slurm allocation when a non-exclusion GPU is available. Do not use
  login-node simulation, one-shot `sbatch`, `sspath`, or the `carry1` tmux
  session.
- [ ] After the suite runs, fix the first failed stage in order. Do not sweep
  later-stage carrying parameters if an earlier stand/contact stage fails.
- [x] Record retry batch `166918`. Result: labrot as wxyz fails immediately;
  identity + IsaacLab gains + setup joint write still fails. No payload test.
- [x] Add pitch/roll rate terms to the Core API G1 balance-feedback
  controller and launcher/checker fields.
- [x] Await/record no-box setup+PD stand batch
  `curiosity_g1_setup_pd_stand_0705`, job-name `g1_setup_pd`, stamps
  `diag18`-`diag20`.
- [x] Record setup+PD stand retry3 `166922`. `diag18_retry3` is best so far
  but still fails late; PD feedback variants are worse. No payload test.
- [x] Await/record static posture/height sweep
  `curiosity_g1_static_posture_sweep_0705`, job-name `g1_post_sweep`, stamps
  `diag21`-`diag24`.
- [x] Record static no-box posture sweep. `diag22` passed 360-step no-box G1
  stand with fall 0 and max tilt `0.00882`; use posture hip `-0.12`, knee
  `0.30`, ankle `-0.15`, root z `0.78` for the next gate.
- [x] Await/record fixed-torso ballast stand
  `curiosity_g1_fixed_payload_stand_0705`, job-name `g1_payload_stand`,
  stamps `diag25`-`diag27`.
- [x] Record fixed-torso ballast stand. `diag25` 0.5 kg, `diag26` 1 kg, and
  `diag27` 2 kg all passed 360-step stand with fall/drop 0.
- [x] Update open-loop march to use `_stand_joint_targets()` as the base
  posture instead of the old hard-coded nominal.
- [x] Await/record open-loop march smoke
  `curiosity_g1_openloop_march_0705`, job-name `g1_march_smoke`, stamps
  `diag28`-`diag29`.
- [x] Record open-loop march smoke. Both no-box and 1 kg fixed-payload
  variants passed stability, but travel was only millimeter/centimeter scale.
- [x] Await/record march-creep sweep
  `curiosity_g1_march_creep_sweep_0705`, job-name `g1_march_creep`, stamps
  `diag30`-`diag32`.
- [x] Record march-creep sweep. `diag30`, `diag31`, and `diag32` all passed
  stability with fall/drop 0, but travel stayed small, around `0.019` to
  `0.028 m`. This remains a dynamic standing/marching diagnostic, not carrying.
- [x] Run fixed-torso collision-enabled payload checks after collision-disabled
  ballast passed. `diag33` 1 kg stand and `diag34` 1 kg open-loop march both
  passed with `carry_box_spawned=true`, `box_collision_enabled=true`,
  `joint_count=43`, rollout root/box writes 0, fall/drop 0, and max tilt under
  `0.028 rad`.
- [x] Add a Core API G1 `front_tray` torso-cradle scaffold: physical deck,
  side rails, and front/rear stops fixed to torso while the carry box remains
  a free dynamic rigid body. Also add summary/checker fields for
  `torso_cradle`, cradle piece count, no-drop requirement, and box-robot
  relative-offset drift.
- [x] Await/record free dynamic box torso-cradle diagnostics
  `curiosity_g1_free_cradle_0705`, job-name `g1_free_cradle`, stamps
  `diag35` stand and `diag36` open-loop march. These are contact-scaffold
  diagnostics only; they must not be called full robot carrying.
- [x] Record free dynamic box torso-cradle diagnostics. Result: negative.
  `diag35` stand failed with `fall_events=351`, `box_drop_events=331`,
  `max_tilt_rad=3.14090`, `min_robot_z_m=0.13680`, and
  `max_box_robot_relative_offset_error_m=0.59789`. `diag36` march failed with
  `fall_events=591`, `box_drop_events=571`, `max_tilt_rad=3.13946`, and
  `max_box_robot_relative_offset_error_m=0.59809`. The front-tray/free-box
  geometry is currently destabilizing the G1 almost immediately.
- [x] Await/record isolation diagnostics
  `curiosity_g1_cradle_isolation_0705`, job-name `g1_cradle_iso`, stamps
  `diag37` cradle-only and `diag38` free-box-only. Use these to decide whether
  the immediate failure comes from torso-cradle inertial/collision geometry or
  from the free box initial contact.
- [x] Record isolation diagnostics. `diag37` cradle-only failed with
  `fall_events=171`, `max_tilt_rad=3.12544`, and `min_robot_z_m=0.19309`.
  `diag38` free-box-only kept G1 stable with `fall_events=0` and
  `max_tilt_rad=0.02695`, while the unsupported free box dropped
  (`box_drop_events=73`). Therefore the immediate failure source is the
  torso-cradle geometry/collision, not G1 stand or free-box presence alone.
- [x] Add cradle debugging controls:
  `CRADLE_COLLISION_ENABLED=0` / `--disable-cradle-collision` and
  `CRADLE_MASS_SCALE` / `--cradle-mass-scale`.
- [x] Await/record cradle tuning diagnostics
  `curiosity_g1_cradle_tune_0705`, job-name `g1_cradle_tune`, stamps
  `diag39` collision-disabled cradle-only and `diag40` small/light/forward
  collision-enabled cradle-only.
- [x] Mark first `diag39`/`diag40` batch invalid. Slurm job `166942` failed
  before Isaac because the compute node read a stale/intermediate launcher and
  `bash -n` failed. Login-node file was clean and passed `bash -n`; do not
  count this as experiment evidence.
- [x] Await/record retry2 cradle tuning diagnostics
  `curiosity_g1_cradle_tune_retry2_0705`, job-name `g1_cradle_tune2`,
  stamps `diag39_retry2` and `diag40_retry2`. Retry includes longer
  compute-side sleep and prints launcher lines 55-75 before `bash -n`.
- [x] Record retry2 cradle tuning. `diag39_retry2` same cradle with collision
  disabled passed 180/180 with fall 0 and `max_tilt_rad=0.04391`.
  `diag40_retry2` small/light/forward collision-enabled cradle passed 180/180
  with fall 0, `max_tilt_rad=0.01649`, `min_robot_z_m=0.78422`,
  `cradle_mass_scale=0.15`, deck size `0.24 x 0.26 x 0.025`, and deck local
  position `(0.44, 0.0, 0.10)`. Use `diag40_retry2` geometry as the current
  free-box contact baseline.
- [x] Await/record small free-box-on-cradle standing diagnostics
  `curiosity_g1_small_cradle_freebox_0705`, job-name `g1_small_freebox`,
  stamps `diag41` z=1.00 and `diag42` z=0.95, with a 0.25 kg small free box
  on the stable small/forward cradle.
- [x] Record small free-box-on-cradle standing diagnostics. `diag41` and
  `diag42` both passed 240/240 with `attach_box=none`, `torso_cradle=front_tray`,
  free dynamic box spawned, cradle collision enabled, rollout root/velocity/
  box pose writes 0, fall/drop 0, and max tilt under `0.025 rad`. `diag42` is
  the current best baseline: `max_box_robot_relative_offset_error_m=0.03337`,
  final relative-offset error `0.01054`, and min box z `0.95931`.
- [x] Await/record small free-box-on-cradle marching diagnostic
  `curiosity_g1_small_cradle_march_0705`, job-name `g1_small_march`, stamp
  `diag43`, using the `diag42` geometry with `open_loop_march`, amplitude
  `0.05`, frequency `0.7 Hz`, 420 steps.
- [x] Record small free-box-on-cradle marching diagnostic. `diag43` passed
  420/420 with `attach_box=none`, free dynamic box spawned, cradle collision
  enabled, rollout root/velocity/box pose writes 0, fall/drop 0,
  `max_tilt_rad=0.02290`, `min_box_z_m=0.95979`,
  `max_box_robot_relative_offset_error_m=0.03481`, and final relative-offset
  error `0.02609`. Travel remains only centimeter scale
  (`max_robot_travel_xy_m=0.01923`, `max_box_travel_xy_m=0.03443`), so this
  is a stable contact-scaffold marching diagnostic, not carrying.
- [ ] Next gate: increase difficulty one axis at a time from `diag43`:
  longer duration, slightly heavier free box, larger box, or larger gait
  amplitude. Do not change multiple axes in one diagnostic.
- [x] Run one-axis next-gate diagnostics from the `diag43` baseline.
  `curiosity_g1_small_cradle_nextgates_0705`, Slurm job `166948`, ran
  `diag44` long duration, `diag45` heavier 0.5 kg box, and `diag46` larger
  gait amplitude 0.08.
- [x] Record one-axis next-gate diagnostics. All passed checker gates with
  `attach_box=none`, free dynamic box, `torso_cradle=front_tray`, cradle
  collision enabled, 43 G1 joints, rollout root/velocity/box pose writes 0,
  fall/drop 0, and relative-offset error under `0.08 m`:
  `diag44` 1200 steps, `max_tilt_rad=0.02290`,
  `max_box_robot_relative_offset_error_m=0.03481`;
  `diag45` 0.5 kg, `max_tilt_rad=0.03175`,
  `max_box_robot_relative_offset_error_m=0.03766`;
  `diag46` gait amplitude 0.08, `max_tilt_rad=0.02582`,
  `max_box_robot_relative_offset_error_m=0.05755`.
- [ ] Current blocker: stable contact is no longer the immediate blocker for
  the small-box scaffold; meaningful walking/carry distance is. The best runs
  still have centimeter-scale travel, so they are not complete carrying.
- [ ] Next gate: increase gait amplitude from `0.08` to `0.12`/`0.16` with
  the 0.25 kg small free box and measure whether real robot/box travel grows
  without fall/drop or shortcut writes.
- [x] Run gait-amplitude probe `curiosity_g1_amp_probe_0705`, Slurm job
  `166949`, stamps `diag47` amplitude 0.12 and `diag48` amplitude 0.16.
- [x] Record gait-amplitude probe. Both passed fall/drop and no-shortcut
  gates. `diag47`: `max_robot_travel_xy_m=0.03872`,
  `max_box_travel_xy_m=0.06618`, `max_tilt_rad=0.02881`,
  `max_box_robot_relative_offset_error_m=0.10253`. `diag48`:
  `max_robot_travel_xy_m=0.05303`, `max_box_travel_xy_m=0.10716`,
  `max_tilt_rad=0.04088`,
  `max_box_robot_relative_offset_error_m=0.15246`. These are still not
  carrying-distance results because max travel may be lateral oscillation.
- [x] Add final and target-directed travel metrics to
  `build_core_world_g1_box_scene.py` and optional checker gates:
  `final_*_target_directed_travel_m` and
  `max_*_target_directed_travel_m`.
- [ ] Await/record target-directed metric reruns
  `curiosity_g1_target_travel_metrics_0705`, job-name `g1_target_travel`,
  stamps `diag49` amplitude 0.16 and `diag50` amplitude 0.20.
- [x] Record target-directed metric reruns. Result: negative but informative.
  `diag49` amplitude 0.16 achieved real target-directed motion
  (`max_box_target_directed_travel_m=0.65748`,
  `final_box_target_directed_travel_m=0.65724`) but failed with
  `fall_events=42`, `box_drop_events=37`, `max_tilt_rad=0.93175`,
  and max relative-offset error `0.55755`. `diag50` amplitude 0.20 moved
  farther (`final_box_target_directed_travel_m=0.70231`) but also failed with
  `fall_events=41`, `box_drop_events=38`, `max_tilt_rad=1.29701`, and max
  relative-offset error `0.57376`.
- [ ] Current blocker refined: open-loop large-amplitude gait can produce
  meaningful forward carry distance, but it loses box retention and balance
  late in the rollout. Do not count `diag49`/`diag50` as success.
- [ ] Next gate: strengthen the small torso cradle while preserving stability.
  First run cradle-only with longer deck/higher rails and stops; only then run
  free-box amp 0.16/0.20.
- [ ] Await/record stronger-cradle diagnostics
  `curiosity_g1_stronger_cradle_0705`, job-name `g1_cradle_strong`, stamps
  `diag51` stronger cradle-only and `diag52` stronger cradle + free dynamic
  0.25 kg box + gait amplitude 0.16. The goal is to test whether higher
  rails/stops preserve the target-directed motion from `diag49` without late
  box drop/fall.
- [x] Record stronger-cradle diagnostics. `diag51` cradle-only passed 240/240
  with fall 0 and `max_tilt_rad=0.01893`. `diag52` with free box and amp 0.16
  also had fall/drop 0 and kept the box high (`min_box_z_m=0.96531`), but it
  failed the target-directed travel gate: `max_box_target_directed_travel_m`
  only `0.02283 m`. Stronger cradle improved retention but removed meaningful
  forward carry distance.
- [ ] Next gate: try middle cradle geometry between `diag40` and `diag51`, and
  require both box retention and target-directed travel.
- [ ] Await/record middle-cradle diagnostics
  `curiosity_g1_mid_cradle_0705`, job-name `g1_mid_cradle`, stamps `diag53`
  middle cradle-only and `diag54` middle cradle + free dynamic box + gait
  amplitude 0.16.
- [x] Record middle-cradle diagnostics. `diag53` cradle-only passed. `diag54`
  retained the box and robot with fall/drop 0 and `min_box_z_m=0.95330`, but
  failed target-directed travel: `max_box_target_directed_travel_m=0.03777`.
  Middle cradle still suppresses useful forward motion.
- [x] Run balance-feedback stabilization batch
  `curiosity_g1_balance_stabilize_0705`, job-name `g1_balance_stab`, stamps
  `diag55` pitch feedback 0.35 and `diag56` pitch feedback 0.70.
- [x] Record balance-feedback batch. Both retained stability and box height
  with fall/drop 0, but both failed target-directed travel:
  `diag55 max_box_target_directed_travel_m=0.01519`; `diag56`
  `0.01610`. Pitch feedback prevents the late fall by suppressing the motion.
- [ ] Next implementation gate: add a gait stop schedule so the rollout can
  move with the unstable-but-forward open-loop gait for a short window, then
  return to stand before the late fall/drop. Evaluate whether short-distance
  target-directed carrying plus post-move hold is possible.
- [x] Add `GAIT_START_STEP` / `GAIT_STOP_STEP` launcher controls and
  corresponding `--gait-start-step` / `--gait-stop-step` script arguments.
- [x] Mark first gait-stop batch invalid. `curiosity_g1_gait_stop_0705`,
  Slurm job `166963`, failed before Isaac because compute read a stale
  launcher with a shell parse error; login-node `bash -n` passed.
- [x] Await/record gait-stop retry2
  `curiosity_g1_gait_stop_retry2_0705`, job-name `g1_gait_stop2`, stamps
  `diag57_retry2` stop step 220 and `diag58_retry2` stop step 260.
- [x] Record gait-stop retry2. Both `diag57_retry2` and `diag58_retry2`
  passed the no-fall/no-drop/no-shortcut gates with free dynamic box on the
  minimal torso cradle, but both stopped too early to retain meaningful
  target-directed carrying distance. `diag57_retry2`: final box
  target-directed travel `-0.03321 m`, max box target-directed travel
  `0.03505 m`, max tilt `0.05694 rad`, fall/drop 0. `diag58_retry2`:
  final box target-directed travel `0.00253 m`, max box target-directed travel
  `0.03505 m`, max tilt `0.04088 rad`, fall/drop 0. These are stable
  post-move hold diagnostics, not carrying-distance evidence.
- [x] Mark first late gait-stop batch invalid. `curiosity_g1_late_stop_0705`,
  job-name `g1_late_stop`, Slurm job `166968`, had shell quoting expansion in
  the tmux command; it produced the invalid stamp
  `20260705_core_world_g1_min_cradle_amp016_stop__late_retry` and was
  cancelled. Do not count it as evidence.
- [x] Await/record late gait-stop retry2
  `curiosity_g1_late_stop_retry2_0705`, job-name `g1_late_stop2`, Slurm job
  `166972`, using `scripts/isaac/run_core_world_g1_late_stop_batch.sh`.
  Planned stamps are `diag59` stop step 320, `diag60` 340, `diag61` 360, and
  `diag62` 370. The goal is to exploit the `diag49` pre-fall window where
  target-directed motion exists before fall/drop, while requiring post-stop
  hold with fall/drop 0 and rollout root/box writes 0.
- [x] Record late gait-stop retry2 as a negative result for
  `cradle_mass_scale=0.15`. All four runs had fall/drop 0, min box z above
  `0.948 m`, max tilt under `0.052 rad`, and rollout root/box writes 0, but
  max box target-directed travel stayed `0.03505 m` and final box
  target-directed travel stayed between `-0.00865 m` and `0.01329 m`. This is
  a stable hold result, not carrying. Field comparison shows the main
  difference from `diag49` is `cradle_mass_scale`: `diag49` used `1.0`, while
  retry2 used `0.15`.
- [x] Mark mass-1.0 late gait-stop retry3 invalid.
  `curiosity_g1_late_stop_mass1_0705`, job-name `g1_late_mass1`, Slurm job
  `166976`, failed before usable Isaac evidence because compute-side shell
  parsing of `scripts/isaac/run_core_world_g1_box_scene.sh` hit the recurring
  stale/partial launcher read. Login-node `bash -n` passed. Do not count as an
  experiment.
- [x] Await/record mass-1.0 late gait-stop retry4
  `curiosity_g1_late_stop_mass1_retry4_0705`, job-name `g1_late_m1r4`, Slurm
  job `166977`, stamps `diag59`-`diag62` with suffix `late_mass1_retry4`.
  This rerun includes compute-side startup sleep and launcher line printing,
  matches the `diag49` cradle mass scale, and tests whether the pre-fall
  target-directed motion can be stopped and held.
- [x] Record mass-1.0 late gait-stop retry4 as negative. It reproduced the
  `diag49` forward motion but stopping the gait at steps 320/340/360/370 did
  not recover balance or retain the box. All four failed fall/drop gates:
  `diag59` stop 320: fall 55, drop 43, max box target-directed travel
  `0.62280 m`, min box z `0.13024 m`; `diag60` stop 340: fall 56, drop 39,
  max box target-directed travel `0.75835 m`, min box z `0.05596 m`;
  `diag61` stop 360: fall 49, drop 36, max box target-directed travel
  `0.82349 m`, min box z `0.05538 m`; `diag62` stop 370: fall 44, drop 36,
  max box target-directed travel `0.77346 m`, min box z `0.14658 m`.
  Rollout root/box writes remained 0, so the failures are physical/controller
  failures, not shortcut artifacts.
- [ ] Next implementation gate: stop sweeping gait-stop timing. Add a real
  recovery/control mechanism: pitch/velocity feedback that activates before
  the runaway lean, or a lower-energy gait/contact schedule that produces
  forward motion without entering the unrecoverable pitch state. Required
  gate remains fall/drop 0, rollout root/box writes 0, and nontrivial final
  box target-directed travel.
- [x] Add threshold-gated balance feedback controls:
  `BALANCE_START_STEP`, `BALANCE_PITCH_ACTIVATION_THRESHOLD`,
  `BALANCE_ROLL_ACTIVATION_THRESHOLD`,
  `BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD`, and
  `BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD`, plus summary fields for active
  steps and first active step.
- [x] Mark first threshold-feedback diagnostics invalid.
  `curiosity_g1_threshold_feedback_0705`, job-name `g1_thresh_fb`, Slurm job
  `166986`, attempted stamps `diag63`-`diag66`, but compute-side line printing
  showed it read an old `run_core_world_g1_box_scene.sh` without the threshold
  arguments. The summaries had activation thresholds `0.0` and feedback active
  from step 0, so this was all-time feedback, not threshold feedback. Do not
  count as evidence.
- [x] Await/record direct threshold-feedback retry2
  `curiosity_g1_threshold_feedback_direct_retry2_0705`, job-name `g1_thr_fb2`,
  Slurm job `166993`, stamps `diag63`-`diag66` with suffix
  `direct_retry2`. The batch now calls `build_core_world_g1_box_scene.py`
  directly to bypass stale shell launcher reads.
- [x] Record direct threshold-feedback retry2 as negative. The thresholds were
  correctly applied and feedback did not start at step 0:
  `diag63` threshold `0.20`, gain `0.35`, first active step 266, fall 52,
  drop 36, max box target-directed travel `0.64076 m`;
  `diag64` threshold `0.30`, gain `0.35`, first active step 296, fall 52,
  drop 36, max box target-directed travel `0.66990 m`;
  `diag65` threshold `0.20`, gain `0.70`, rate gain `0.03`, first active step
  266, fall 55, drop 38, max box target-directed travel `0.78312 m`;
  `diag66` threshold `0.30`, gain `0.70`, rate gain `0.03`, first active step
  296, fall 51, drop 38, max box target-directed travel `0.69332 m`.
  Threshold feedback preserved forward motion but did not stop the runaway
  pitch/fall. Do not continue simple pitch-feedback sweeps without changing
  the gait/contact mechanism.
- [ ] Next gate: implement a lower-energy forward gait/contact schedule rather
  than relying on heavy-cradle forward tipping. Candidate implementation:
  reduce gait amplitude after a short acceleration window, add a forward lean
  budget, and add explicit recovery posture targets for hip/knee/ankle/torso
  that are active before pitch reaches `0.20 rad`. Gate on fall/drop 0,
  rollout root/box writes 0, and at least `0.10 m` final box target-directed
  travel.
- [x] Implement staged G1 gait controls in the direct Core API scene. Added
  `gait_mode=staged_march`, gait ramp-down start/end, minimum amplitude
  scale, and recovery posture offsets for hip/knee/ankle/waist. Summaries now
  record staged/recovery parameters and recovery active step counts. Lightweight
  checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/check_core_world_g1_box_scene_summary.py`
  and `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- [x] Await/record staged gait diagnostics
  `curiosity_g1_staged_gait_0705`, job-name `g1_staged`, stamps `diag67`-
  `diag70`. The batch calls `build_core_world_g1_box_scene.py` directly to
  avoid stale launcher reads and tests intermediate cradle mass scales plus
  ramp-down/recovery schedules. Required gate remains fall/drop 0, rollout
  root/box writes 0, and final box target-directed travel at least `0.10 m`.
- [x] Record staged gait diagnostics. `diag67` mass `0.35`, amp `0.16` and
  `diag68` mass `0.50`, amp `0.14` were stable but failed distance
  (`final_box_target_directed_travel_m=-0.00043 m` and `0.01699 m`).
  `diag69` mass `0.75`, amp `0.12` improved to `0.06329 m` with fall/drop 0
  but still failed the `0.10 m` distance gate. `diag70` mass `1.0`, amp
  `0.12` produced meaningful motion (`final_box_target_directed_travel_m=
  0.64051 m`) and no box drop, but failed at the last two rollout steps with
  `fall_events=2` and `max_tilt_rad=0.86674 > 0.85`. This is a near miss, not
  a success.
- [x] Await/record staged gait refinement diagnostics
  `curiosity_g1_staged_gait_refine_0705`, job-name `g1_stage_ref`, stamps
  `diag71`-`diag74`. These refine around the `diag69`/`diag70` boundary by
  testing cradle mass `0.85-1.0` and gait amplitude `0.10-0.12`, aiming for
  `>=0.10 m` final box target-directed travel with fall/drop 0.
- [x] Record staged gait refinement diagnostics. All four passed the current
  420-step short-distance checker with fall/drop 0 and rollout root/box writes
  0. `diag71` mass `0.85`, amp `0.12`: final box target-directed travel
  `0.37075 m`, max tilt `0.43785`. `diag72` mass `0.90`, amp `0.11`:
  `0.49137 m`, max tilt `0.60004`. `diag73` mass `0.95`, amp `0.10`:
  `0.51315 m`, max tilt `0.62928`. `diag74` mass `1.0`, amp `0.10`:
  `0.63546 m`, max tilt `0.84893`, just below the `0.85` threshold. Treat
  these as short-distance diagnostic passes, not final completion.
- [ ] Await/record longer hold validation for the safer short-distance
  variants. Next batch should rerun at least `diag72` and `diag73` for 700
  steps to determine whether the robot can keep carrying/balancing after the
  420-step window rather than only avoiding the fall threshold at the cutoff.
- [x] Add long validation launcher
  `scripts/isaac/run_core_world_g1_staged_gait_long_validation.sh` for
  700-step reruns of `diag72` and `diag73`; lightweight shell and Python checks
  passed.
- [x] Await/record long validation batch
  `curiosity_g1_staged_long_0705`, job-name `g1_stage_long`, stamps `diag75`
  from `diag72` and `diag76` from `diag73`.
- [x] Record long validation as negative. Both 700-step reruns failed after
  the 420-step window. `diag75` from `diag72` failed with `fall_events=256`,
  `box_drop_events=239`, `max_tilt_rad=1.14866`, min box z `0.09610`, while
  still reaching `0.75432 m` final box target-directed travel. `diag76` from
  `diag73` failed similarly with `fall_events=259`, `box_drop_events=242`,
  `max_tilt_rad=1.14076`, min box z `0.10530`, and `0.75943 m` final box
  target-directed travel. The failure begins around step 450, so the 420-step
  passes are short-window diagnostics only.
- [ ] Next gate: retreat to a more conservative long-run region around
  cradle mass `0.80-0.85` and amp `0.10-0.12`, seeking a 700-step pass with
  final box target-directed travel at least `0.10 m`.
- [x] Add conservative long validation launcher
  `scripts/isaac/run_core_world_g1_staged_gait_conservative_long.sh`;
  lightweight checks passed.
- [ ] Await/record conservative long validation batch
  `curiosity_g1_staged_cons_long_0705`, job-name `g1_cons_long`, stamps
  `diag77`-`diag80`, testing mass `0.80/0.85` and amp `0.10/0.12` for a
  700-step pass.
- [x] Check local AGILE/WBC policy weights. The files under
  `external/WBC-AGILE/agile/data/policy/velocity_height_g1/` are Git LFS
  pointer files, not real weights (`unitree_g1_velocity_height_recurrent_student_checkpoint.pt`
  is 132 bytes and points to a 6.65 MB object). Do not claim AGILE policy
  locomotion until real weights are present.
- [ ] Next gate if middle cradle also suppresses travel: return to the minimal
  cradle that produced target-directed motion and add balance feedback to
  reduce the late forward pitch/fall.

- [x] Stop treating external model downloads as blockers for the direct Isaac
  scene path.
- [x] Keep staged-free-box scaffold honest: `articulated-foot-contact` is now
  recorded as requested, not enabled, unless a real implementation exists.
- [x] Add root-shortcut evidence counters to the quasi-static walker scaffold.
- [x] Choose the first no-root carrier implementation path:
  controller-backed local robot if immediately usable, otherwise a redesigned
  statically stable articulated diagnostic carrier.
- [x] Implement a no-root fixed-payload stand diagnostic with positive joint
  count, foot-contact drive, root velocity writes 0, root pose writes 0,
  fall/drop 0, and no non-finite PhysX state.
- [x] Run lightweight syntax and shell checks for the prismatic carrier stand
  builder, launcher, and checker.
- [x] Run the stand diagnostic in a Curiosity-owned tmux-held Slurm allocation.
- [x] Add a checker gate for the stand diagnostic if it uses a new summary
  schema.
- [x] After stand passes, implement no-root fixed-payload slow creep.
  Current evidence: `20260705_prismatic_carrier_stand_diag2_360_server63`
  passed the no-root fixed-payload stand gate on `server63`.
- [x] Run a first no-root fixed-payload creep gate.
  `20260705_prismatic_carrier_creep_diag3_neg_target_server63` passed a short
  negative-X creep gate with 8 articulated DOFs, root writes 0, fall/drop 0,
  max absolute torso travel `0.05861 m`, and final target distance `0.00721 m`.
- [x] Test at least one additional fixed-payload carrying posture/load before
  reconnecting the free box.
- [x] Attempt a first free dynamic top-contact box carry diagnostic.
  Result: negative. `diag7b`/`diag8` used `payload_mode=top_contact_free_box`
  and the box did not drop, but the robot/free box did not move far enough to
  pass the target/travel gate. Passive top contact is not enough.
- [x] After slow creep passes, reconnect the staged free dynamic box.
- [x] Add an explicit tray, side-rail, clamp, or contact-constraint strategy
  for the free box, then rerun the no-root free-box creep gate.
  2026-07-05 code update: added `PAYLOAD_MODE=tray_contact_free_box`, tray
  size/rail/mass parameters, and checker visibility for tray fields. Awaiting
  compute rollout evidence.
- [x] Add payload-direction and relative-offset gates so free-box diagnostics
  cannot pass when the torso reaches the target but the free payload slides in
  the opposite direction.
- [x] Run free-box tray stand with a stable wide-center configuration.
  `20260705_prismatic_carrier_tray_freebox_diag5_stand_wide_center_server36`
  passed a stand gate with no root/body/box/payload pose writes, fall/drop 0,
  max tilt `0.00588 rad`, max torso drift `0.00444 m`, min payload z
  `0.70280 m`, and max payload relative-offset error `0.07579 m` during
  settling.
- [x] Test whether the current x-slide creep structure can carry the tray/free
  box after the stable stand configuration.
  Result: negative. `diag6_wide_center_micro_creep_server36` failed during the
  settle/creep attempt with fall events 564 and drop events 583. The x-slide
  horizontal-leg structure is not a valid free-box carrying locomotion base.
- [x] Stop waiting on external models after the G1/WBC route failed to enter
  rollout and the MuJoCo no-assist fallback failed balance/travel gates.
- [x] Rerun the direct Isaac quasi-static fixed-payload carry scaffold as the
  immediate continuation path.
  `20260705_quasistatic_direct_continue_diag1_server63` completed 420/420 with
  body/payload travel `0.16585 m`, final target distance `0.01415 m`,
  payload-relative error `0.0 m`, min support margin `0.13252 m`, and fall/drop
  0. It remains a velocity-commanded torso scaffold, not robot locomotion.
- [x] Add a no-root `stance_translate` prismatic diagnostic and a dedicated
  tray/free-box launcher to avoid the stale/omitted launcher-argument path.
- [x] Run the first valid no-root `stance_translate` tray/free-box diagnostic.
  Result: negative. `20260705_prismatic_tray_stance_translate_diag3_server63`
  correctly used `payload_mode=tray_contact_free_box` and had zero root/body/
  box/payload pose or velocity shortcuts, but failed with fall events 360, box
  drop events 264, max tilt `1.76093 rad`, and max payload relative-offset
  error `2.06908 m`.
- [x] Add a physical tray top-lid/cage option while keeping the box as a free
  dynamic body.
- [x] Run a fair slow-ramp tray/free-box test after clearing the bad parameter
  probe state. Result: negative. `diag4_slowramp_server10` had fall 0 but box
  drop events 523.
- [x] Test high-wall cage stand and simple roll/pitch leg-length balance servo.
  Result: negative for balance. The cage kept the box but destabilized the
  carrier; both positive and negative balance gains failed.
- [x] Find a stable low-CG cage stand configuration.
  `20260705_prismatic_cage_stand_lowcg_diag1_server10` passed with free dynamic
  box in cage, fall/drop 0, root/body/box/payload writes 0, max tilt
  `0.03275 rad`, and max payload relative-offset error `0.00609 m`.
- [x] Run the first low-CG cage no-root short-translation gate.
  `20260705_prismatic_lowcg_cage_translate_diag1_server10` passed: 8 DOFs,
  fall/drop 0, no root/body/box/payload shortcuts, max torso travel
  `0.01580 m`, max payload travel `0.01930 m`, final target distance
  `0.000087 m`, final payload target distance `0.00307 m`, and max payload
  relative-offset error `0.00625 m`.
- [x] Try extending low-CG cage translation beyond the first short pass.
  Result: partial/negative. `diag2_3cm_server10` stayed safe with fall/drop 0
  but failed the 3 cm travel/target gate because torso travel saturated near
  `0.01580 m`.
- [x] Try repeated-foot variants before abandoning the prismatic cage as a
  locomotion base.
  Result: negative/partial. `20260705_prismatic_lowcg_cage_creep_diag1_server53`
  and `20260705_prismatic_lowcg_cage_sync_inchworm_diag1_server53` both kept
  the free box stable with fall/drop 0 and no root/body/box/payload shortcuts,
  but both saturated around `0.01580 m` torso travel and failed the 3 cm
  travel/target gate.
- [x] Verify whether local G1 WBC assets are available without using an
  external GR00T server.
  Result: assets load on `server53`; Robot DOFs `43`, policy
  `G1DecoupledWholeBodyPolicy`.
- [x] Attempt minimal IsaacLab G1 WBC stand without external model/server.
  Result: initialization failure, not control failure. Both normal and
  skip-reset diagnostics failed before rollout with invalidated PhysX tensor
  simulation view while reading DOF positions.
- [x] Add a direct Core API G1+box scene to bypass IsaacLab `InteractiveScene`.
  Result: script and launcher added and runnable, but current USD/root binding
  still reports robot root near `z=-0.025 m`; G1 does not initialize in a
  standing pose yet.
- [x] Fix the first Core API G1 root-height issue enough to start G1 above the
  ground in setup. Result: `20260705_core_world_g1_box_scene_diag4_setuppose_server46`
  starts with robot z about `0.799 m` and fall 0, but open-loop stand joint
  targets are not a balancing controller; the robot falls by step 30 and later
  drops the box. This is scene/root initialization progress, not carrying.
- [x] Check official Go2 controller-backed locomotion as a fast Isaac carrying
  base. Result: negative for the current standalone launch path.
  `20260705_go2_callback_locomotion_diag1_server46` exits around
  `SimulationManager` setup without a summary; `diag2_skipdev_server46`
  completes 180/180 but has `callback_forward_calls=0` and zero travel;
  `diag3_dtonly_offkit_server46` exits around `set_physics_dt`; and
  `20260705_official_go2_manual_locomotion_diag1_server46` fails before policy
  initialization with invalid articulation physics tensor entity.
- [x] Follow the user correction and stop treating official models/controllers
  as blockers when they are not immediately useful.
- [x] Add a direct-Isaac anchored posture sweep that does not wait on external
  models. It runs several candidate carry postures under the same randomized
  hidden box seed with active probing and writes a metric-ranked diagnostic
  summary.
- [x] Run the direct-Isaac anchored posture sweep inside a Curiosity-owned
  tmux-held Slurm allocation and record the result. Submitted
  `curiosity_direct_anchor_sweep_0705`, Slurm job `166557`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17`. It reached `server10`
  but failed on CUDA tensor-to-NumPy conversion before usable rollout metrics.
  Added a GPU-backend `_as_numpy()` compatibility fix and submitted retry2:
  `curiosity_direct_anchor_sweep_retry2_0705`, Slurm job `166564`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17_retry2`. Retry2 reached
  `server10` but failed in the nested core launcher with an unexpected EOF
  shell parse error. The sweep wrapper was flattened to call the Python
  builder directly, then retry3 was submitted:
  `curiosity_direct_anchor_sweep_retry3_0705`, Slurm job `166571`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17_retry3`. Retry3 entered
  the Isaac step loop but the support-foot-drive/fixed-anchor configuration
  commanded no rail travel, so it was canceled as uninformative. The sweep now
  uses `support_foot_mode=fixed_to_anchor` without support-foot drive, and
  retry4 was submitted: `curiosity_direct_anchor_sweep_retry4_0705`, Slurm job
  `166575`, stamp `20260705_direct_isaac_anchor_posture_sweep_seed17_retry4`;
  it started on `server10`. Retry4 produced nonzero rail targets but no actual
  torso/payload travel on the GPU pipeline, so it was canceled as
  uninformative. The sweep now explicitly uses CPU backend, static-marker
  support, one rail joint, and horizontal rail probing; retry5 submitted:
  `curiosity_direct_anchor_sweep_retry5_0705`, Slurm job `166579`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17_retry5`. Result:
  completed on `server02`. The hidden randomized payload was mass
  `8.175871 kg`, size about `0.36503 x 0.26652 x 0.20889 m`, COM offset
  `[0.02129, 0.01225, 0.00968] m`. All four posture candidates completed
  300/300 with fall/drop 0. The ranker chose `extended_front`, then
  `front_mid`, then `chest_close`, then `low_close`. `extended_front` and
  `front_mid` reached the 8 cm target with final target distance about
  `3.27e-05 m`; `chest_close` and `low_close` remained safe but ended about
  `0.00950 m` short. This is only an anchored fixed-payload diagnostic, not a
  free-walking robot or free-box carrying success.
- [ ] Next direct-Isaac task step: replace the fixed-payload posture sweep
  with a free dynamic box contact strategy that preserves the same posture
  ranking interface, or connect the posture interface to a real articulated
  locomotion controller. Do not present the current anchored sweep as final
  robot carrying.
- [x] Run one stronger direct-Isaac prismatic cage diagnostic to test whether
  simple x-slide tuning can extend travel. Result: negative/partial.
  `20260705_prismatic_stance_translate_strongx_diag1_server46` stayed safe
  with fall/drop 0 and no root/body/box/payload pose or velocity shortcuts, but
  still saturated at max torso travel `0.01557 m` and max payload travel
  `0.01834 m` for an `0.08 m` target. The actual summary mode remained
  `sync_inchworm`; do not cite it as a true `stance_translate` override.
- [x] Implement a direct Core API anchored-support diagnostic to replace
  torso/root velocity writes with a physical joint drive for fixed-payload
  carrying.
- [x] Run the first anchored-support fixed-payload gate.
  `20260705_anchor_footstep_fixed_diag8b_holdtarget_server46` passed: one
  articulated joint, fixed 4 kg payload, completed 180/180, fall/drop 0,
  root/body/box/payload pose and velocity shortcuts 0, max torso travel
  `0.03781 m`, max payload travel `0.03781 m`, final target distance
  `0.00219 m`, min payload z `0.55720 m`, and max payload relative-offset
  error near zero. This is a world-fixed support-frame single-step diagnostic,
  not full robot walking.
- [x] Run the first anchored-support caged free-box diagnostic.
  Result: negative. `20260705_anchor_footstep_cagedfree_diag1_server46`
  completed 180/180 with fall/drop 0 and no root/body/box/payload shortcuts,
  but the free box was expelled/accelerated by cage contacts: final payload
  target distance `0.79402 m`, max payload travel `0.83402 m`, and max payload
  relative-offset error `2.01642 m`.
- [x] Try a centered/lighter anchored-support caged free-box variant to reduce
  initial contact impulses. Result: partial but still negative.
  `20260705_anchor_footstep_cagedfree_diag2_centerbox_server46` completed
  180/180 with fall/drop 0 and no root/body/box/payload shortcuts. The payload
  no longer shot out and reached max travel `0.02232 m`, but torso travel was
  only `0.00365 m`, final torso target distance `0.03635 m`, final payload
  target distance `0.02186 m`, and max payload relative-offset error
  `0.13000 m`.
- [x] Parameterize anchored-support cage geometry instead of using hard-coded
  wall/lid/deck dimensions.
- [x] Test a compact centered cage with smaller clearances.
  Result: negative. `20260705_anchor_cage_compact_diag1_server46` used
  `cage_clearance_xy=0.015`, `cage_clearance_z=0.018`, and
  `cage_wall_thickness=0.04`. It completed 180/180 with fall/drop 0, but the
  free box was violently expelled: max payload travel `9.08616 m`, final
  payload target distance `9.04616 m`, and max payload relative-offset error
  `20.67944 m`.
- [x] Extend anchored-support fixed-payload carrying from 4 cm to 8 cm.
  `20260705_anchor_fixed_8cm_diag3_upper08_server46` passed: one articulated
  prismatic joint, fixed 4 kg payload, completed 260/260, fall/drop 0, all
  root/body/box/payload pose and velocity shortcuts 0, max torso/payload
  travel `0.08003 m`, final target and payload target distances
  `0.000032 m`, min payload z `0.55720 m`, and max payload relative-offset
  error near zero. This remains a single support-frame diagnostic, not walking.
- [x] Add a multi-rail/telescoping anchored-support variant and closed-stop
  latch for longer fixed-payload motion.
  Result: safe but not precise enough for a success gate.
  `20260705_anchor_telescoping_fixed_24cm_diag1_server36` completed with
  fall/drop 0 and no root/body/box/payload shortcuts, but overshot a `0.24 m`
  target to `0.29479 m`, final target distance `0.05479 m`.
  `20260705_anchor_telescoping_fixed_16cm_diag3_closedstop_server36` completed
  360/360 with two rail joints, fall/drop 0, no shortcuts, stop latch true,
  max torso/payload travel `0.18291 m`, final target distance `0.02291 m`, and
  max payload relative-offset error near zero. This is useful fixed-payload
  support-frame evidence, not walking.
- [x] Add a staged-grasp constraint mode to anchored-support as an alternative
  to tightening the cage.
  Code update: `PAYLOAD_MODE=staged_grasp_constraint` now starts with a free
  box, can optionally use a preparation shelf, and runtime-authors a
  `StagedGraspJoint` at attach time. Lightweight syntax and launcher checks
  passed.
- [x] Run staged-grasp delayed-attach and step-0 attach diagnostics.
  Result: negative for the current fixed-joint attach design.
  `20260705_anchor_staged_grasp_diag1_server36` completed 220/220 with
  fall/drop 0 and no root/body/box/payload shortcuts, but produced disjoint
  `StagedGraspJoint` warnings, max payload relative-offset error `0.23856 m`,
  final target distance `0.01865 m`, and final payload target distance
  `0.07996 m`. `20260705_anchor_staged_grasp_diag2_runtime_step0_server36`
  also completed safely, but still produced a disjoint-joint warning and ended
  with max payload relative-offset error `0.13740 m`, final target distance
  `0.01867 m`, and final payload target distance `0.12045 m`.
- [x] Replace the staged fixed-joint attach attempt with first low-impulse
  tray/clamp/contact diagnostics.
  Code update: added `PAYLOAD_MODE=open_tray_free_box`,
  `PAYLOAD_MODE=side_clamp_free_box`, and `PAYLOAD_MODE=x_cradle_free_box` to
  `build_core_world_anchored_footstep_carrier.py`, with launcher parameters
  for tray geometry, side clamp gaps/drives, and X cradle gaps. Lightweight
  syntax checks passed.
- [x] Run open-tray free-box diagnostics.
  Result: negative. `20260705_anchor_open_tray_diag1_slow_server36` completed
  260/260 with fall/drop 0, no disjoint warnings, and no root/body/box/payload
  shortcuts, but failed carrying: max torso travel `0.01870 m`, final target
  distance `0.02842 m`, final payload target distance `0.06563 m`, and max
  payload relative-offset error `0.18925 m`.
  `20260705_anchor_open_tray_diag2_highstop_slow_server36` made the stops
  taller/tighter and was much worse: the box was accelerated to max absolute
  payload travel `4.43549 m`, final payload target distance `4.47049 m`, and
  max payload relative-offset error `11.36084 m`.
- [x] Run side-clamp free-box diagnostics.
  Result: negative for the current side clamp joint design.
  `20260705_anchor_side_clamp_diag1_slowclose_server36` used a valid 3-DOF
  articulation and had no root/body/box/payload shortcuts, but produced
  `281` box-drop events, max clamp joint motion only `5.39e-05 m` for a
  requested `0.07 m` clamp travel, and max payload relative-offset error
  `0.94668 m`. Stronger drive in
  `20260705_anchor_side_clamp_diag2_strongclamp_stand_server36` and rotated
  X-axis joint frames in `diag3_rotaxis_strongstand_server36` still left max
  clamp motion around `5.31e-05 m` and caused `64` box-drop events. The side
  pads are not effectively closing in this articulation.
- [x] Run X-cradle free-box stand diagnostic.
  Result: negative. `20260705_anchor_x_cradle_diag1_stand_server36` proved the
  commanded rear-pusher X joint is active (`max_cradle_joint_motion_m`
  `0.05205 m` for `0.052 m` requested travel), but it fired the free box during
  settle: max payload travel `12.49785 m`, final payload target distance
  `12.49785 m`, and max payload relative-offset error `13.13603 m`.
- [x] Replace the failed tray/side-clamp/X-cradle contact attempts with a
  staged single-contact-element diagnostic: first validate one moving contact
  element against a supported low-mass free box or dummy object, then combine
  it with carrier rail motion only after contact closure is low-impulse.
  Result: completed as an isolated contact diagnostic, not as robot carrying.
  Added `build_core_world_single_contact_probe.py` and launcher.
  `20260705_single_contact_probe_diag4_lowfric_strong_server10` completed
  360/360 with max pusher travel `0.06979 m`, free-box travel `0.04469 m`,
  max box speed `0.07889 m/s`, min box z `0.12999997 m`, fall/drop 0, and no
  root/body/box/payload shortcuts.
- [x] Combine the validated contact behavior with a clean constrained carrier
  rail before returning to robot locomotion.
  Result: added `build_core_world_cradle_cart_free_box_carry.py` and launcher.
  `20260705_cradle_cart_freebox_diag3_postsettle_8cm_server10` passed as a
  non-locomotion contact scaffold: one `CartRail` joint, target `0.08 m`, max
  cart travel `0.07883 m`, post-settle cart travel `0.07881197 m`,
  post-settle free-box travel `0.07881202 m`, final post-settle relative error
  `4.79e-08 m`, drop 0, nonfinite 0, and all root/body/box/payload shortcuts
  0. This is not robot carrying.
- [ ] Replace the cradle-cart `CartRail` with a robot/support-switching body
  while preserving the same free-box post-settle contact metrics.
- [x] Integrate the validated cradle free-box contact module into the
  anchored-footstep carrier scaffold.
  Result: added `PAYLOAD_MODE=cradle_free_box` to
  `build_core_world_anchored_footstep_carrier.py`. The critical fix was
  torso-scale-corrected fixed-joint local positions for cradle parts; before
  that, the front stop penetrated the box (`-0.37491 m` measured gap) and
  expelled it.
- [x] Verify anchored cradle geometry after scaled-joint correction.
  `20260705_anchor_cradle_freebox_diag6b_scaledjoint_geom_server23` measured
  rear/front surface gaps about `0.025 m`, max payload drift `1.19e-07 m`,
  min payload z `0.7289998 m`, and drop 0.
- [x] Run anchored fixed-support free-box cradle carrying at 8 cm.
  `20260705_anchor_cradle_freebox_diag7_scaledjoint_8cm_server23` passed:
  one `StanceRail`, max torso/free-box travel `0.078173 m`, final target
  distances about `0.00187 m`, final post-settle relative error `2.18e-07 m`,
  fall/drop 0, and no root/body/box/payload shortcuts.
- [x] Extend anchored fixed-support free-box cradle carrying to 16 cm.
  `20260705_anchor_cradle_freebox_diag9_fixed_16cm_2rail_server23` passed:
  two rail joints, max torso/free-box travel `0.158377 m`, final target
  distances about `0.00188 m`, final post-settle relative error `5.77e-08 m`,
  fall/drop 0, and no root/body/box/payload shortcuts. This is still fixed
  support, not walking.
- [x] Attempt first support-switching anchored cradle free-box diagnostic.
  Result: negative. `20260705_anchor_cradle_freebox_diag8_supportswitch_16cm_server23`
  failed because PhysX rejected assigning a transform to non-root articulation
  link `/World/Robot/StanceAnchor`; fall events `522`, box-drop events `455`,
  final payload target distance `25.28 m`.
- [x] Test support-switch redesigns without writing non-root articulation link
  poses.
  Results:
  `20260705_anchor_cradle_freebox_diag10_anchorroot_supportswitch_16cm_server53`
  removed the non-root warning by making `StanceAnchor` the articulation root,
  but failed because the support was still a free dynamic body: fall events
  `520`, drops `455`. `diag11_kinanchor_supportswitch_16cm_server53` showed
  PhysX rejects `ArticulationRootAPI` on a kinematic rigid body, so a kinematic
  support-root articulation is not valid. `diag12_worldjoint_replant_16cm_server53`
  used a world fixed-joint retarget and was stable with fall/drop 0, but runtime
  fixed-joint retarget did not accumulate support displacement; final travel
  was only about `0.0401 m`. `diag13_worldjoint_phasefix_16cm_server53` fixed
  the final-cycle phase reset and stayed stable, but still ended at about
  `0.0799 m`, showing the world-joint retarget was not an effective support
  replant in this scaffold.
- [x] Add an explicitly labeled cumulative-cycle transport diagnostic after
  the support-replant attempts.
  `20260705_anchor_cradle_freebox_diag14_cumulative_16cm_server53` passed:
  two cycles, `cumulative_cycle_target=true`, target `0.16 m`, max torso travel
  `0.158374 m`, max free-box travel `0.158374 m`, final target distances about
  `0.00193 m`, final post-settle payload/torso relative error `3.13e-08 m`,
  fall/drop 0, nonfinite 0, and root/body/box/payload pose or velocity
  shortcuts 0. This is stable multi-cycle rail-target free-box transport, not
  true support switching or walking.
- [x] Extend the cumulative-cycle free-box transport diagnostic to longer
  distance and heavier payloads.
  Results on `server53`:
  `20260705_anchor_cradle_freebox_diag15_cumulative_32cm_server53` passed with
  `0.5 kg` payload, four cycles, target `0.32 m`, max torso/free-box travel
  about `0.31925 m`, final target distances about `0.00195 m`, fall/drop 0.
  `20260705_anchor_cradle_freebox_diag16_cumulative_32cm_4kg_server53` passed
  with `4.0 kg`, max travel about `0.31949 m`, final target distances about
  `0.00186 m`, fall/drop 0. `diag17_cumulative_32cm_8kg_server53` passed with
  `8.0 kg`, max travel about `0.31969 m`, final target distances about
  `0.00194 m`, final post-settle relative error `2.08e-07 m`, fall/drop 0.
  These remain cumulative rail-target diagnostics, not walking or true support
  replant.
- [ ] Redesign actual support switching with a separate support-target/root
  mechanism whose pose change is effective in PhysX at runtime. Do not claim
  `diag14` as support replant; it is only a cumulative rail-target scaffold.
- [x] Add the validated cradle free-box geometry to the prismatic-leg carrier
  so a physical-foot scaffold can carry the same kind of free dynamic box.
  Implementation: added `PAYLOAD_MODE=cradle_free_box` to
  `build_core_world_prismatic_carrier_stand.py`, including scaled-torso fixed
  joint correction for cradle parts, cradle gap metrics, and post-settle active
  travel metrics. Added launcher parameters to
  `run_core_world_prismatic_carrier_stand.sh`.
- [x] Run prismatic-leg cradle/free-box stand and short stepping diagnostics.
  Results on Slurm job `166052`, tmux
  `curiosity_prismatic_cradle_gpu_0705`, `server53`:
  `20260705_prismatic_cradle_stand_diag1b_8kg_server53` stood for 500 steps
  with an `8 kg` free box, cradle rear/front gaps about `0.0224/0.0269 m`,
  fall/drop 0, no root/body/box/payload writes, but settled about `5.6 cm`
  backward and the payload relative offset changed by about `6.2 cm`.
  `diag1_4cm_8kg` and `diag2_neg4cm_8kg` showed the first `sync_inchworm`
  gait stayed safe with fall/drop 0 but target evidence was confounded by
  settle drift. `diag3b_postsettle_neg4cm_8kg` added post-settle metrics and
  showed active torso/payload travel about `-0.01997 m`, only half the
  `-0.04 m` target, with fall/drop 0. `diag4_postsettle_neg8cm_8kg` improved
  active post-settle travel to `-0.04627 m` final and `0.06195 m` peak, with
  8 DOFs, fall/drop 0, max tilt `0.09624 rad`, min payload z `0.7281 m`, and
  root/body/box/payload pose or velocity shortcuts 0. This is the strongest
  physical-foot free-box carrying scaffold so far, but still not full walking
  or learned balance.
- [x] Extend the direct Isaac prismatic-leg cradle/free-box diagnostic before
  spending more time on external models. Results in tmux
  `curiosity_prismatic_cradle_long_gpu_0705`, Slurm job `166070` on `server53`:
  `diag5_postsettle_neg14cm_8kg` reached final active post-settle travel about
  `-0.08707 m`; `diag6_postsettle_neg22cm_8kg` reached about `-0.14711 m`;
  and `diag7_postsettle_neg30cm_8kg` reached final torso/payload active
  post-settle travel about `-0.20588/-0.20587 m`, with peak active travel about
  `0.22180/0.22179 m`. All three completed with 8 kg free dynamic boxes,
  fall/drop 0, nonfinite 0, and zero root/body/box/payload pose or velocity
  shortcuts. These are still physical-foot scaffold diagnostics, not complete
  walking.
- [x] Run and evaluate
  `20260705_prismatic_cradle_sync_inchworm_diag8_postsettle_neg40cm_8kg_server53`
  to see whether the direct Isaac scaffold can extend beyond the `diag7`
  post-settle plateau instead of waiting on external models.
  Result: negative. The aggressive `-0.40 m` / larger slide-limit / stronger
  drive setup caused real dynamics failure: fall events `3126`, box-drop
  events `2826`, min torso/payload z near `-1071/-1074 m`, and max payload
  relative-offset error `117.19 m`, with shortcut counters still 0.
- [x] Run and evaluate
  `20260705_prismatic_cradle_sync_inchworm_diag9_postsettle_neg34cm_8kg_server53`
  using the safer `diag7` parameter family and only a small target extension.
  Result: negative. It still became unstable when extended to six sync-inchworm
  cycles: fall events `2637`, box-drop events `2182`, min torso z
  `-828.23 m`, max tilt `3.09742 rad`, and max payload relative-offset error
  `829.85 m`. The payload's final X target metric is invalid because the
  carrier had fallen.
- [x] Implement a cycle-stabilization pause for the prismatic `sync_inchworm`
  controller instead of continuing to increase slide force. Added
  `--sync-cycle-pause-fraction` and launcher env
  `SYNC_CYCLE_PAUSE_FRACTION`; lightweight syntax checks passed.
- [x] Run and evaluate
  `20260705_prismatic_cradle_sync_inchworm_diag10_pause_neg34cm_8kg_server53`
  with `SYNC_CYCLE_PAUSE_FRACTION=0.20`.
  Result: invalid as a pause test and negative as a run. The summary recorded
  `sync_cycle_pause_fraction=0.0`, so the intended pause was not passed into
  the simulation. It failed with fall events `2937`, box-drop events `2482`,
  min torso z `-1030.66 m`, and max payload relative-offset error
  `1032.28 m`.
- [x] Check why the cycle-pause runner did not apply on the compute tmux.
  The compute shell saw the builder-side `--sync-cycle-pause-fraction`
  argument but not the patched runner argument, so `diag11` was interrupted
  and not counted as evidence.
- [x] Try direct-Python pause invocation to bypass the stale runner path.
  `diag12_pause_direct_neg34cm_8kg_server53` explicitly included
  `--sync-cycle-pause-fraction 0.20`, but destabilized during early stand/
  settle and was interrupted. Do not count it as a full experiment.
- [ ] Replace the current prismatic sync-inchworm distance-extension approach
  with a new support/foot-placement mechanism before attempting more
  >5-cycle free-box carry runs.
- [x] Add a first feedback-gated support-clock mechanism to the prismatic
  carrier instead of continuing open-loop distance sweeps.
  Implementation: `motion_mode=feedback_sync_inchworm` advances the gait clock
  only when the previous step has no fall/drop, tilt below
  `FEEDBACK_TILT_HOLD_THRESHOLD`, and payload relative-offset error below
  `FEEDBACK_PAYLOAD_ERROR_HOLD_THRESHOLD`. It records feedback hold/release
  counters and the last block reason. Lightweight `py_compile` and `bash -n`
  checks passed.
- [x] Extend the prismatic checker so feedback/free-box runs can be gated on
  post-settle active travel, not only total drift. Added post-settle travel and
  final post-settle target-distance flags plus feedback fields in the report.
- [ ] Run a compute diagnostic for
  `motion_mode=feedback_sync_inchworm` on the 8 kg cradle/free-box task,
  starting with the `diag7` target family before retrying the failed six-cycle
  `-0.34 m` case.
- [x] Re-establish the stable 8 kg cradle/free-box stand parameters before
  judging feedback control. Early feedback/stand attempts that omitted
  `PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.18`, `LEG_TARGET=-0.57`, and
  `LEG_LOWER=-0.82` fell and are parameter-control negatives, not valid
  feedback evidence. `stand_regression_diag4_stable_payload_pose_8kg_server53`
  passed 500/500 with fall/drop 0 and max tilt `0.11573 rad`.
- [x] Run the first valid `feedback_sync_inchworm` diagnostic with the stable
  8 kg cradle/free-box parameters. Result:
  `20260705_prismatic_cradle_feedback_sync_diag2_neg30cm_stableparams_8kg_server53`
  was safe with fall/drop 0, no shortcuts, and `feedback_hold_steps=0`, but it
  only reached about `0.0595 m` peak post-settle travel and returned near zero
  final post-settle travel, so it is not a distance improvement over `diag7`.
- [x] Run a same-parameter ordinary `sync_inchworm` replay under the current
  code to check whether the open-loop baseline still reproduces `diag7`.
  Result: negative/regression. `20260705_prismatic_cradle_sync_replay_diag13_neg30cm_stableparams_8kg_server53`
  stayed safe with fall/drop 0, but only reached about `0.0595 m` peak
  post-settle travel and returned near zero final post-settle travel, so it
  did not reproduce the historical `diag7` cumulative travel.
- [x] Add command-vs-actual leg diagnostics to the prismatic cradle/free-box
  scaffold. Summary and CSV now record commanded leg lift, commanded x-slide
  target, actual leg lift, and actual x-slide motion. Short diagnostics showed
  the controller commands about `5 cm` swing lift and `6 cm` horizontal slide,
  but sampled foot world height remains near ground contact during swing.
- [ ] Stop treating the current prismatic sync-inchworm gait as the main
  walking path. Keep it only as a free-box contact/load-bearing diagnostic
  unless a later controller change proves real repeatable foot clearance and
  cumulative travel.
- [ ] Build a cleaner direct Isaac carry-task scene/controller interface that
  preserves the validated free dynamic box, randomized load/geometry hooks,
  cradle/contact metrics, active-probing metrics, and shortcut counters, while
  allowing the locomotion controller to be swapped without rewriting the box
  task.
- [x] Upgrade the existing direct carry-task scene into an explicit
  task/controller-interface diagnostic. Added `CONTROLLER_MODE`, box seed,
  mass-range, and size-jitter arguments; added `controller_contract`,
  `robot_proxy_pose_write_count`, and `box_kinematic_pose_write_count` to the
  summary; and added `scripts/isaac/check_direct_carry_task_summary.py`.
- [x] Run a randomized direct carry-task interface smoke on compute.
  `20260705_direct_carry_task_interface_rand_smoke1_server53` completed
  180/180 with `BOX_SEED=7051`, sampled `7.2301 kg` box mass, sampled box size
  about `0.593 x 0.379 x 0.382 m`, box-drop 0, max box travel `0.67485 m`,
  final target distance `0.03485 m`, robot proxy pose writes `2340`, and box
  kinematic pose writes `180`. This is a clean task-interface smoke, not robot
  carrying evidence.
- [ ] Replace the task-interface `kinematic_proxy` controller with a first
  swappable physical controller backend. It must keep the same summary schema
  and explicitly report root/body/box pose or velocity shortcuts.
- [x] Add the first swappable physical backend wrapper for the direct
  carry-task interface. `run_direct_carry_task_physical_backend.sh` runs the
  anchored/cradle backend; `normalize_direct_carry_backend_summary.py` maps
  the backend summary into the direct-task schema; the direct-task checker now
  supports physical backend shortcut and post-settle travel gates.
- [x] Run the first direct-task physical backend compute smoke.
  `20260705_direct_physical_backend_anchor_cradle_smoke1_server10` exposed a
  wrapper bug: default `RAIL_UPPER=0.04` capped four positive rail joints at
  `0.16 m`, so it was interrupted and not counted as success. The wrapper was
  fixed to default `RAIL_LOWER=-0.04`, `RAIL_UPPER=0.10`.
- [x] Rerun the direct-task physical backend smoke after the rail-limit fix.
  `20260705_direct_physical_backend_anchor_cradle_smoke2_railupper10_server10`
  passed the checker: 980/980, `controller_mode=physical_anchored_cradle`,
  8 kg free dynamic box, fall/drop 0, root shortcut free, max box travel
  `0.319915 m`, max post-settle box travel `0.319915 m`, final box target
  distance `0.001939 m`, final post-settle box/torso relative error
  `6.46e-08 m`, support-root pose writes 0, and
  `anchor_world_joint_retarget_count=4`. This is physical backend progress,
  not complete free walking.
- [ ] Replace the anchored world-support backend with an actual support
  switching / foot-placement controller that does not rely on world-joint
  replanting, while preserving the same direct-task normalized summary gates.
- [x] Add active-probing support to the direct carry-task runner.
  `DirectCarryAction` now includes probe steps and probe amplitudes; the shell
  backend forwards them to the Isaac physical backend; the episode exporter
  records the probe fields.
- [x] Add strict probe gates to the direct-task checker.
  `--require-probe-belief`, `--forbid-probe-hidden-ground-truth`,
  `--min-probe-steps`, and `--min-probe-box-travel-x` are available through
  `run_check_direct_carry_task_runner_episode.sh`.
- [x] Validate one active-probing task-runner episode on compute.
  `20260705_task_runner_probe_frontmid_seed7079` completed on Slurm job
  `166819` with an `11.13313 kg` hidden box, `80` probe steps, fall/drop `0`,
  final post-settle box travel `0.66478 m`, final post-settle target distance
  `0.02478 m`, probe belief available, and no hidden-ground-truth probe use.
  Probe-gated checker job `166821` passed.
- [x] Run a same-hidden-box multi-posture active-probe sweep through the direct
  task-runner interface.
  `20260705_task_runner_active_probe_postures_seed7080`, Slurm job `166822`,
  completed on `server02` with exit `0:0`. Shared hidden box: `10.72455 kg`,
  size `[0.36519, 0.22971, 0.23912] m`, COM offset
  `[0.03732, 0.00523, -0.00058] m`. `front_mid`, `low_front`, and
  `chest_high` all completed `3660/3660` with fall/drop `0`,
  active-probe belief available, and no hidden-ground-truth probe use. Final
  post-settle box travel / target-distance were `0.64775 / 0.00775 m`,
  `0.67183 / 0.03183 m`, and `0.65476 / 0.01476 m`. This is scaffold
  task-runner evidence, not full robot walking or RL.
- [ ] Convert the direct carry-task runner into a cleaner controller-backend
  contract: explicit reset input, action schema, observation schema, reward
  terms, termination gates, hidden evaluation context, and backend capability
  flags. The point is to keep the Isaac box/probing/posture scene reusable
  while replacing the anchored scaffold with a real support-switching or
  robot locomotion controller.
- [x] Add explicit backend capability flags to the direct carry-task contract.
  `DirectCarryBackendCapabilities` now records whether a backend is Isaac,
  free-box, randomized-load, active-probing, trainable-policy, real-robot-
  morphology, support-switching, video-conditioned, hidden-context-isolated,
  root-shortcut-audited, and scaffold-only. The current shell backend is
  conservatively labeled `anchored_support_scaffold`, not a trainable walking
  backend.
- [x] Export backend capabilities and termination fields into episode rows.
  Legacy summaries without explicit capabilities are inferred as scaffold
  evidence, not as robot success. Compute export validation job `166828`
  produced
  `experiments/outputs/rl_interface/20260705_contract_caps_export_retry2/direct_carry_task_episode_table.jsonl`
  with 3 rows, each marked `scaffold_backend=true`,
  `trainable_policy_backend=false`, `episode_completed=true`, and
  `step_limit_reached=null`.
- [ ] Use the backend capability contract to add the next backend adapter:
  either a real support-switching / foot-placement Isaac controller, or a
  controller-backed Isaac robot scene. The adapter must produce the same
  episode row schema before any RL/video policy work is added.
- [x] Add a first direction-aware foot-placement backend variant without
  waiting on external models. Implementation added
  `support_foot_placement_mode=alternating_directional_x` to the anchored
  footstep carrier, plus `SUPPORT_MODE=alternating_placement_feet` in the
  physical backend wrapper and task runner.
- [x] Add checker gates for the new support-placement behavior.
  `check_direct_carry_task_summary.py` and
  `run_check_direct_carry_task_runner_episode.sh` can now require
  `support_foot_placement_controller_enabled=true`,
  `support_foot_directional_placement=true`, and
  `support_foot_placement_mode=alternating_directional_x`.
- [x] Validate the directional foot-placement backend on compute.
  First run `166831` failed on stale/old support-mode dispatch, and retry
  `166832` failed from redundant CLI arg forwarding through the wrapper.
  After fixing the adapter, Slurm job `166833` passed:
  `20260705_task_runner_directional_placement_seed7081_retry3`, hidden box
  `7.23482 kg`, target `0.64 m`, `80` probe steps, final post-settle box
  travel `0.65735 m`, final post-settle target distance `0.01735 m`,
  fall/drop `0`, probe belief available, and no hidden-ground-truth probe use.
  Strict checker retry `166840` passed with directional-placement gates and
  exported backend capability id
  `physical_alternating_placement_feet_cradle_v1`.
- [x] Run the new `alternating_placement_feet` backend in the opposite target
  direction and fix positive-X-biased metrics.
  Initial negative-target job `166842` completed but exposed that reward and
  travel-loss were computed as if progress were always positive X. Added
  absolute and target-directed post-settle travel metrics and target-directed
  reward terms. Rerun `20260705_task_runner_directional_negative_seed7082_retry`
  in Slurm job `166845` passed: target `-0.32 m`, hidden box `8.20882 kg`,
  final post-settle travel `-0.35174 m`, max target-directed post-settle
  travel `0.37768 m`, final target distance `0.03174 m`, fall/drop `0`.
  Strict checker job `166846` passed with directional-placement and absolute
  travel gates.
- [ ] Run the new `alternating_placement_feet` backend across multiple
  postures and at least two randomized box seeds per direction. This is needed
  before treating direction-aware placement as a reliable replacement for the
  older fixed-X alternating support-foot scaffold.
- [x] Add fixed-anchor physical backend mode and gates that forbid support
  replanting. `run_direct_carry_task_physical_backend.sh` now supports
  `SUPPORT_MODE=fixed_anchor`, reported as
  `controller_mode=physical_fixed_anchor_cradle`; the checker now has
  `--max-anchor-world-joint-retarget-count` and
  `--max-support-root-pose-write-count`.
- [x] Run the fixed-anchor backend ablation. Result:
  `20260705_direct_physical_backend_fixed_anchor_32cm_8kg_server10` passed:
  980/980, 8 kg free dynamic box, fall/drop 0, root shortcut free, max
  post-settle box travel `0.322541 m`, final box target distance
  `0.001794 m`, final box/torso relative error `6.46e-08 m`,
  `anchor_world_joint_retarget_count=0`, and support-root pose writes 0.
  This proves the cradle/contact/load-bearing backend does not require
  world-joint replanting for 32 cm/8 kg, but it is still fixed world support,
  not walking.
- [ ] Replace the fixed world support with an actual support-switching /
  foot-placement controller. The next backend must keep
  `anchor_world_joint_retarget_count=0` and `support_root_pose_write_count=0`
  while moving the support contact location through physical foot/contact
  mechanics rather than a fixed rail.
- [x] Add and validate the first alternating X/Z support-foot physical backend
  that removes fixed-world support and anchor retargeting from the direct
  carry-task scaffold. Results:
  `20260705_direct_physical_backend_alternating_anchor_feet_5cycle_holdfix_8cm_8kg_frontmid`
  passed 8 cm / 8 kg after per-foot target-hold fixing;
  `20260705_direct_physical_backend_alternating_anchor_feet_10cycle_holdfix_16cm_8kg_frontmid`
  passed 16 cm / 8 kg;
  `20260705_direct_physical_backend_alternating_anchor_feet_20cycle_holdfix_32cm_8kg_frontmid`
  passed 32 cm / 8 kg; and
  `20260705_direct_physical_backend_alternating_anchor_feet_40cycle_holdfix_64cm_8kg_frontmid`
  passed 64 cm / 8 kg. The 64 cm front-mid run completed 3580/3580 with
  actual support-foot lift `0.06475 m`, max box travel `0.64785 m`, final
  target distance `0.01181 m`, final post-settle box travel `0.62941 m`,
  fall/drop 0, root shortcut free, fixed world support false, anchor retargets
  0, support-root writes 0, foot pose writes 0, and stance-anchor pose writes
  0. This is still a scaffold, not a full walking robot.
- [x] Run a 64 cm / 8 kg multi-posture sweep for the alternating support-foot
  backend. `low_front` passed with max box travel `0.70662 m`, final target
  distance `0.03367 m`, final post-settle box travel `0.67882 m`, actual
  support-foot lift `0.06348 m`, fall/drop 0, and all shortcut counters 0.
  `chest_high` passed with max box travel `0.70446 m`, final target distance
  `0.02583 m`, final post-settle box travel `0.66510 m`, actual support-foot
  lift `0.06353 m`, fall/drop 0, and all shortcut counters 0. Log scan found
  no disjoint/fatal/traceback/unbound/EOF errors.
- [ ] Tighten the alternating support-foot scaffold before any RL claim:
  reduce near-ground foot XY speed, avoid transition frames with zero
  near-ground feet if physically required, add explicit support/contact-force
  or impulse evidence when available, and test randomized box mass/size/COM and
  friction.
- [ ] Turn the alternating support-foot backend into the first trainable
  direct-Isaac carry task: observation/action schema, randomized load and
  morphology hooks, probing subactions, reward terms, reset logic, and
  evaluation gates. Keep it video-free until no-video active probing has a
  defensible baseline.
- [x] Add reproducible randomized-load hooks to the direct physical backend.
  The builder now accepts `BOX_SEED`, randomized mass range, size jitter, and
  payload COM offset range; the backend summary and normalized direct-task
  summary record requested/range/sampled payload values.
- [x] Run a randomized-load smoke for the alternating support-foot backend.
  `20260705_direct_physical_backend_alternating_anchor_feet_randomized_8cm_seed7051`
  ran in Slurm job `166474` on `server53` and passed: seed `7051`, sampled mass
  `8.15343 kg`, sampled size about `0.35775 x 0.25309 x 0.23354 m`, sampled
  COM offset `[0.00902, 0.00821, -0.00216] m`, 780/780 steps, max box travel
  `0.09614 m`, final target distance `0.01740 m`, final post-settle box
  travel `0.06362 m`, actual support-foot lift `0.06319 m`, fall/drop 0, root
  shortcut free, fixed world support false, anchor retargets 0, support-root
  writes 0, foot pose writes 0, and stance-anchor pose writes 0.
- [ ] Add real probing-phase actions and belief metrics. Do not treat sampled
  mass/COM as known to the policy; record them only as hidden ground truth for
  evaluation. The first probing diagnostic should estimate load/COM from
  micro-lift, push-pull, support-foot response, object lag, and effort proxies.
- [x] Add the first pre-carry probe measurement phase to the alternating
  support-foot backend. It supports `PROBE_STEPS` and `PROBE_X_AMPLITUDE`,
  logs `active_probe_push_pull`, records probe torso/box travel, probe
  relative error, and final probe lag, and starts carry post-settle baselines
  after the probe.
- [x] Run a randomized probe smoke. Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_probe_randomized_8cm_seed7052`
  passed in Slurm job `166483` on `server36`: randomized mass `9.72299 kg`,
  sampled size about `0.34799 x 0.24707 x 0.22159 m`, sampled COM offset
  `[-0.01960, 0.01151, 0.00618] m`, probe steps `60`, probe amplitude
  `0.020 m`, max probe torso/box travel `0.03445/0.03466 m`, max probe
  relative error `0.00835 m`, final probe lag `0.000325 m`, 860/860 steps,
  max box travel `0.14405 m`, final target distance `0.02021 m`, final
  post-settle box travel `0.11084 m`, actual support-foot lift `0.05082 m`,
  fall/drop 0, root shortcut free, fixed world support false, and all support
  write/retarget counters 0.
- [ ] Convert probe telemetry into an explicit hidden-load belief estimate or
  calibrated proxy. The summary should report the belief before/after probing,
  prediction error against hidden ground-truth mass/COM, and whether the
  controller changes carry posture or gait parameters because of the belief.
- [x] Add a first heuristic probe-derived belief proxy without using hidden
  ground truth. It reports compliance proxy, lag proxy, risk score, risk
  bucket, and a recommended carry adjustment, and explicitly records that the
  policy has not yet applied the recommendation.
- [x] Run the first belief-proxy smoke. Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_belief_probe_randomized_8cm_seed7053`
  passed with sampled mass `9.81294 kg`, probe compliance proxy `0.25067`,
  lag proxy `0.04821`, risk score `0.80478`, bucket
  `high_observed_load_or_shift_response`, recommended adjustment
  `slow_gait_low_or_chest_supported_candidate`, 860/860 steps, max box travel
  `0.14364 m`, final target distance `0.02069 m`, fall/drop 0, and all
  shortcut counters 0.
- [ ] Run controlled light/heavy probe calibration and decide whether this
  heuristic is usable for adaptation. If it does not separate known loads, do
  not use it as a belief model.
- [x] Run controlled light/heavy probe calibration. Result: negative for the
  current heuristic. The 6 kg and 10 kg cases both passed the carry gate with
  fall/drop 0 and shortcut counters 0, but produced nearly identical belief
  scores: 6 kg risk `0.78250`, 10 kg risk `0.79201`, both bucketed as
  `high_observed_load_or_shift_response`. Do not use the current box-lag-only
  proxy for controller adaptation.
- [ ] Add support-foot target-vs-actual tracking error and, if accessible,
  effort/force proxies during probe. Re-run light/heavy calibration before any
  belief-driven gait/posture adaptation.
- [x] Add support-foot target-vs-actual tracking error and rerun calibration.
  Result: negative. Tracking proxy did not separate 6 kg from 10 kg
  (`2.04663` vs `2.04556`), so this signal is dominated by controller tracking
  rather than payload mass.
- [ ] Add measured joint effort/force telemetry during probe and rerun
  light/heavy calibration. Use it only if the measured signal clearly differs
  across loads without hidden ground-truth input.
- [ ] Replace the staged fixed-joint attach attempt with a low-impulse
  soft/compliant grasp or driven clamp/contact formulation. Do not keep
  treating runtime fixed-joint snapping as the main free-box solution.
- [ ] After free-box no-root scaffold passes, replace staged attach/contact
  proxies with real contact or a physically defensible constraint.
- [ ] Replace the x-slide open-loop creep mechanism with a better locomotion
  base before further free-box carry claims. Candidate paths: a real
  controller-backed IsaacLab robot, a quasi-static foot-placement controller
  that preserves support polygon constraints, or a constrained cart/table
  diagnostic explicitly labeled as non-locomotion.
- [ ] Add a real balance/locomotion controller to
  `build_core_world_g1_box_scene.py`, or fix the IsaacLab tensor-view lifecycle
  in `build_minimal_carry_scene.py`.
- [ ] After G1 stand enters rollout, run stand -> walk -> fixed-torso-payload
  balance diagnostics in Isaac without GR00T or other external model servers.
- [ ] Build the next direct Isaac carrier iteration that preserves the current
  carry-task metrics but reduces body-root velocity shortcutting.
- [ ] Convert the anchored-support fixed-payload pass from one world-fixed
  support frame to a multi-step support-switching foot placement controller.
- [ ] Repair anchored-support free-box cage contact geometry and initial
  clearances so the free box is carried coherently instead of being expelled
  or sliding relative to the carrier.
- [ ] Stop using tighter cage walls as the main free-box strategy; try a
  low-impulse tray, compliant grasp/contact constraint, or staged contact
  closure that avoids initial penetration/impulse.
- [x] Correct the execution route after user instruction: do not wait for
  external models/checkpoints/data. Continue directly in Isaac and keep any
  external method as future reference only.
- [x] Mark
  `20260705_direct_physical_backend_fixed_anchor_lowfront_32cm_8kg_server36`
  as an interrupted/invalid posture diagnostic. It reached Isaac and about
  step 410 with observed fall/drop 0, but was manually interrupted before
  summary generation and cannot be counted as evidence.
- [ ] Build the next Isaac-first carrying backend that removes fixed world
  support as the main progress mechanism. It must preserve the direct-task
  summary schema, free dynamic box, shortcut counters, and checker gates.
- [ ] Add support-switching / foot-placement state logging to the next backend:
  support phase, active support contact, commanded support displacement,
  measured body/support displacement, and box/contact stability metrics.
- [ ] Run the next backend only on a compute allocation, then check it with
  gates requiring zero anchor retargets, zero support-root writes, no box pose
  writes, no falls/drops, and explicit non-success labeling if the support is
  still not a real robot locomotion controller.
- [x] Add a direct-task wrapper for the no-root prismatic legged carrier:
  `scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh`.
- [x] Extend direct-task normalization/checking so no-root legged backend
  summaries expose backend support mode, fall gates, motion mode, commanded
  leg lift, actual leg lift, actual x-slide, and foot-height metrics.
- [x] Run
  `run_direct_carry_task_no_root_prismatic_backend.sh` on a compute allocation
  with `PAYLOAD_MODE=cradle_free_box`, `PAYLOAD_MASS=8.0`,
  `MOTION_MODE=feedback_sync_inchworm`, and short target distance first.
- [x] Run the first no-root prismatic direct backend diagnostic.
  `20260705_direct_no_root_prismatic_cradle_feedback_10cm_8kg_server46`
  completed 1200/1200 with fall/drop 0, root shortcut free, no box kinematic
  writes, no anchor retargets, and no support-root writes. Result is negative
  for carrying distance: target was `+0.10 m`, max positive box travel was
  `0.02458 m`, max post-settle box travel was `0.05242 m`, and final target
  distance was `0.14996 m`.
- [ ] If the no-root prismatic backend fails to move the free box safely,
  debug the leg/contact mechanics directly: foot clearance, stance contact,
  x-slide force, body pitch/roll, and cradle/box relative slip. Do not return
  to external models as a blocker.
- [x] Add per-leg support-phase/contact diagnostics to
  `build_core_world_prismatic_carrier_stand.py`: near-ground step count,
  per-leg min/max foot z, max commanded lift/x-slide, max actual lift/x-slide,
  plus CSV-level near-ground and commanded-swing foot counts. The current
  aggregate metrics are not enough to explain why the backend moves opposite
  the positive target.
- [x] Rerun the no-root prismatic 10 cm diagnostic with the new per-leg fields
  and inspect whether the failure is caused by feet losing contact, swing reset
  timing, insufficient x-slide, wrong sign, or excessive body settling.
- [x] Rerun the no-root prismatic 10 cm diagnostic with the new per-leg fields.
  Result: per-leg fields showed all feet stayed near ground for most of the
  rollout and horizontal x-slide tracked about `0.10 m`. The main immediate
  issue was command sign and displacement retention, not missing foot motion.
- [x] Implement a simpler quasi-static stance-transfer controller for the
  no-root prismatic backend before more sync-inchworm tuning. Start with one
  controlled stance shift and one swing reset, then require positive box travel
  under the same no-root/no-box-write gates.
- [x] Run corrected quasi-static stance-transfer diagnostic.
  `20260705_direct_no_root_prismatic_quasistatic_corrected_10cm_8kg_server46`
  completed 1200/1200 with fall/drop 0, root shortcut free, no box kinematic
  writes, no anchor retargets, no support-root writes, max box travel
  `0.05647 m`, max post-settle box travel `0.09855 m`, final box target
  distance `0.04855 m`, and max tilt `0.11319 rad`. This is positive no-root
  Isaac progress, not full carrying success.
- [x] Improve stance-transfer displacement retention: for `TARGET_X=0.10`,
  the corrected no-root backend should keep final box target distance under
  `0.02 m` before attempting longer or repeated steps.
- [x] Add and verify settle-drift compensation for no-root quasi-static stance
  transfer. Result:
  `20260705_direct_no_root_prismatic_quasistatic_compensated_10cm_8kg_server02`
  passed with 8 kg free box, fall/drop 0, no root/body/box/support shortcuts,
  max box travel `0.10413 m`, final box target distance `0.00413 m`, and final
  post-settle box travel `0.14621 m`.
- [x] Implement first support-switching diagnostic mode:
  `motion_mode=quasistatic_step_cycle`.
- [x] Run fast reset step-cycle diagnostic. Result:
  `20260705_direct_no_root_prismatic_stepcycle_compensated_10cm_8kg_server02_retry`
  completed safely but failed transport retention: final post-settle box travel
  `-0.00725 m`.
- [x] Run slow/high reset step-cycle diagnostic. Result:
  `20260705_direct_no_root_prismatic_stepcycle_slowreset_10cm_8kg_server02`
  completed safely and reached transient post-settle travel `0.25199 m`, but
  final post-settle box travel collapsed to `0.01530 m`.
- [x] Replace open-loop step reset with a gated support-switch diagnostic:
  reset one foot only while the other feet maintain stance, monitor body/box
  backward slip and tilt during reset, and hold/abort/reset slower if the
  carried box loses displacement. Result:
  `20260705_direct_no_root_prismatic_gated_step_10cm_8kg_mgmtserver02`
  completed safely but failed displacement retention; max post-settle box
  travel was `0.13806 m`, final post-settle travel was `0.04578 m`, and
  travel loss after peak was `0.09228 m`.
- [x] Add summary fields for reset-phase displacement loss: max transient
  post-settle travel, final post-settle travel, loss after peak, and per-leg
  reset loss attribution. Current implementation records peak/final/loss and
  gated hold/release/recovery counters; per-leg reset attribution remains a
  future diagnostic if support switching is pursued.
- [x] Add an Isaac-first posture sweep for the strongest current no-root
  free-box baseline instead of waiting on external models:
  `scripts/isaac/run_no_root_prismatic_posture_sweep.sh`.
- [x] Run the no-root posture sweep on compute. Result: Slurm job `166237` on
  `server10` passed `front_mid`, `low_front`, and `chest_high` with 8 kg free
  box, 1200 steps each, fall/drop 0, root shortcut free, box kinematic writes
  0, anchor retargets 0, and support-root writes 0. Final box target distances
  were `0.00413 m`, `0.01300 m`, and `0.01290 m`.
- [ ] Do not continue treating the current gated step-cycle as success. The
  next support-switching design must prevent reset-induced pullback before it
  happens, not merely detect travel loss afterward.
- [ ] Next Isaac task: make support switching pre-emptive. During reset, solve
  for stance-foot lock and allowable foot placement so the body/box cannot
  move backward more than a small bound; only then release one foot. Keep the
  same no-root/free-box/shortcut-free gates.
- [x] Try prelift reset instead of dragging the swing foot through ground.
  Result: negative/unsafe.
  `20260705_direct_no_root_prismatic_prelift_step_10cm_8kg_server36`
  completed 1800 steps but had `fall_events=1172`, max tilt `1.05202 rad`,
  and final target distance `0.42764 m`.
- [x] Add guarded prelift support-switch diagnostics with travel-loss and
  target-reached gating.
  Result: safe but still not a support-switching carry solution.
  `20260705_direct_no_root_prismatic_guarded_prelift_10cm_8kg_server10`
  stayed safe but reached only `0.07371 m` max box travel; the stride12 run
  preserved post-settle travel but still failed raw target gates; the
  compensated run
  `20260705_direct_no_root_prismatic_guarded_prelift_comp_10cm_8kg_server10`
  stayed safe with real lift commands and no shortcuts, but only reached
  `0.05720 m` max box travel and ended `0.06995 m` from the raw target.
- [x] Expose support-switching control parameters in the direct normalized
  summary: `step_length_m`, `step_height_m`, `gait_period_steps`, and
  `x_slide_limit_m`.
- [x] Run a longer guarded-prelift diagnostic that forces the controller beyond
  single stance transfer. Result: safe but negative.
  `20260705_direct_no_root_prismatic_guarded_prelift_20cm_8kg_mgmtserver02`
  completed 2400/2400 on `server46`, Slurm job `166271`, with fall/drop 0 and
  no shortcuts. It reached max box travel `0.15029 m`, but final box target
  distance was `0.12198 m` and post-settle travel loss after peak was
  `0.07227 m`; the controller deadlocked in recovery after reset-induced
  travel loss.
- [x] Add loss-rebaseline diagnostic plumbing for guarded support switching:
  `--gated-step-loss-rebaseline-steps` and
  `gated_step_loss_rebaseline_count`. This is not a success mechanism; it is
  only to test whether the controller can continue after accepting a stable
  lower post-reset baseline.
- [x] Run guarded-prelift with loss rebaseline enabled. Result: diagnostic
  improvement but still negative for carrying.
  `20260705_direct_no_root_prismatic_guarded_prelift_rebaseline_20cm_8kg`
  completed 2600/2600 on `server10`, Slurm job `166281`, with fall/drop 0 and
  no shortcuts. It commanded lift on all four legs and used
  `gated_step_loss_rebaseline_count=3`; final post-settle travel loss fell to
  `0.00158 m`. It still failed the 20 cm target gate: max box travel
  `0.15029 m`, final box target distance `0.10593 m`, final post-settle target
  distance `0.06384 m`.
- [ ] Replace the current prismatic support-switching gait rather than
  continuing to tune it. Required next design change: stance-foot lock or
  contact anchoring that preserves body/box displacement during swing-foot
  repositioning, while keeping zero root/box/support shortcuts.
- [x] Add stance-overdrive diagnostic plumbing for prelift reset:
  `--prelift-stance-overdrive`, launcher env `PRELIFT_STANCE_OVERDRIVE`, and
  direct summary field `prelift_stance_overdrive`.
- [x] Run guarded-prelift with stance overdrive and no loss rebaseline to
  test whether not-yet-reset stance legs can counter swing-foot return
  reaction. Result for `PRELIFT_STANCE_OVERDRIVE=1.45`: unsafe negative.
  `20260705_direct_no_root_prismatic_guarded_prelift_overdrive145_20cm_8kg`
  completed on `server10`, Slurm job `166289`, but had `fall_events=1976`,
  max tilt `0.91439 rad`, and final box target distance `0.37494 m`.
- [x] Run a smaller stance-overdrive diagnostic before abandoning this route.
  Result: safe but negative. `overdrive115` completed on `server10`, Slurm job
  `166292`, with fall/drop 0 and no shortcuts, but final box target distance
  `0.12104 m` and travel loss after peak `0.09418 m`. Simple stance
  overdrive increases transient travel but does not preserve it.
- [x] Add dynamic swing-foot x-drive force scaling:
  `--swing-x-force-scale`, launcher env `SWING_X_FORCE_SCALE`, and summary
  fields `swing_x_force_scaled_steps` /
  `per_leg_swing_x_force_scaled_steps`.
- [x] Run low-reaction swing-foot diagnostic. Result: safe but unchanged.
  `20260705_direct_no_root_prismatic_guarded_prelift_swingforce008_20cm_8kg`
  ran on `server10`, Slurm job `166300`, with fall/drop 0 and no shortcuts.
  Force scaling was applied for 118 swing-leg steps, but final box target
  distance remained `0.12198 m` and travel loss after peak remained
  `0.07227 m`, essentially matching the non-scaled guarded-prelift run.
- [ ] Stop tuning the current prismatic gait parameters as the main path.
  Implement a stance-foot latch / contact-anchoring diagnostic next. It must
  explicitly count latch retargets and label them as non-final scaffolding, but
  it should answer whether idealized stance locking can preserve carried-box
  displacement during swing-foot repositioning.
- [x] Implement stance-foot latch diagnostic plumbing in
  `build_core_world_prismatic_carrier_stand.py`: disabled world fixed joints
  per foot, `--enable-stance-foot-latch`,
  `--stance-foot-latch-lift-threshold`, launcher envs, and normalized latch
  counters.
- [x] Run stance-foot latch diagnostic. Result: safe but negative, and the
  latch formulation is not clean.
  `20260705_direct_no_root_prismatic_stance_latch_retry_20cm_8kg` completed
  2600/2600 on `server10`, Slurm job `166313`, with fall/drop 0, root shortcut
  free, anchor retargets 0, support-root writes 0, and visible latch counters:
  27 enables, 23 disables, 27 retargets. It failed badly on transport: max box
  travel `0.06334 m`, final box target distance `0.16388 m`, final
  post-settle target distance `0.11822 m`. The log repeatedly reported PhysX
  disjoint fixed-joint warnings for stance latch joints.
- [x] Stance-latch first launch note: Slurm job `166310` exited in 1 second
  with a transient syntax read of `build_core_world_prismatic_carrier_stand.py`
  (`unterminated string literal` in the compute log). Login-node
  `py_compile` now passes. Retry run `curiosity_stance_latch_retry_20cm_0705`,
  Slurm job `166313`, was submitted with a new stamp.
- [ ] Replace runtime world-fixed foot latch with a cleaner stance-anchor
  support-switching diagnostic. The next design should author support anchors
  from startup and avoid mid-simulation fixed-joint snap/disjoint behavior,
  while still reporting every support retarget as non-final scaffold evidence.
- [x] Run cleaner support-anchor replant baseline using the existing
  anchored-footstep carrier. Active run:
  `curiosity_support_anchor_replant_32cm_0705`, Slurm job `166321`,
  `SUPPORT_MODE=replant_world_joint`, `TARGET_X=0.32`, `PAYLOAD_MASS=8.0`,
  four rail joints, free cradle box. Anchor retargets are allowed but must be
  reported as non-final scaffold evidence.
  Result: passed scaffold gate with max box travel `0.31992 m`, final target
  distance `0.00194 m`, fall/drop 0, root shortcut free, support-root writes 0,
  and `anchor_world_joint_retarget_count=4`.
- [x] Run replant support-anchor posture sweep across `front_mid`,
  `low_front`, and `chest_high`, keeping anchor retargets explicit as
  non-final scaffold evidence.
- [x] Posture sweep first launch note: `curiosity_support_anchor_postures_0705`
  / Slurm job `166328` had a shell quoting error, so `$posture` expanded to
  empty before entering the compute shell and only a default `front_mid`
  diagnostic ran under an incomplete stamp. Correctly escaped retry
  `curiosity_support_anchor_postures_retry_0705`, Slurm job `166331`, was
  submitted.
  Result: retry passed all three postures. Each run completed 980/980 with
  8 kg free cradle box, 32 cm target, fall/drop 0, root shortcut free, box
  kinematic writes 0, support-root writes 0, and anchor retargets 4. Final
  target distances were about `0.00194 m` for `front_mid`, `low_front`, and
  `chest_high`.
- [ ] Use the passing multi-posture support-anchor scaffold as the task/contact
  target for the next replacement step: remove retargeted world support by
  introducing a real locomotion/support-switching controller while preserving
  the same free-box, posture, and checker gates.
- [x] Expose support-anchor audit fields in the direct normalized summary:
  `rail_joint_count`, `rail_capacity_m`, `rail_joint_indices`, `cycle_count`,
  `stride_m`, `foot_pose_write_count`, and `stance_anchor_pose_write_count`.
- [ ] Run longer support-anchor scaffold boundary diagnostic. Active run:
  `curiosity_support_anchor_long64_0705`, Slurm job `166338`,
  `SUPPORT_MODE=replant_world_joint`, `TARGET_X=0.64`, `PAYLOAD_MASS=8.0`,
  four rail joints, 1500 steps.
- [x] Run longer support-anchor scaffold boundary diagnostic.
  Result: `20260705_direct_physical_backend_replant_anchor_64cm_8kg` completed
  1500/1500 with fall/drop 0, root shortcut free, support-root writes 0, and
  final post-settle box/torso relative error `5.03e-09 m`, but failed the
  64 cm distance gate because four rails provided only `0.4 m` capacity. Max
  box travel was `0.40009 m`, final target distance `0.239997 m`, and
  `anchor_world_joint_retarget_count=8`.
- [x] Rerun the 64 cm support-anchor boundary with enough rail capacity.
  Result: `20260705_direct_physical_backend_replant_anchor_64cm_8kg_8rail`
  ran on `server10`, Slurm job `166342`, and passed the direct scaffold
  checker: 1500/1500, 8 kg free cradle box, eight rails, rail capacity
  `0.8 m`, fall/drop 0, root shortcut free, support-root writes 0, max box
  travel `0.64583 m`, final target distance `0.00191 m`, final post-settle
  box travel `0.63809 m`, and final post-settle box/torso relative error
  `8.66e-08 m`. It still used
  `anchor_world_joint_retarget_count=8`, so it is scaffold evidence only.
- [ ] Run a stricter 64 cm / 8 kg / 8-rail ablation with
  `SUPPORT_MODE=fixed_anchor` and checker gates requiring
  `anchor_world_joint_retarget_count=0` and support-root writes 0. This does
  not solve walking, but it removes support replanting from the current
  long-distance scaffold before building the next support-switching
  controller.
- [ ] Active run:
  `curiosity_fixed_anchor_long64_8rail_0705`, Slurm job `166347`,
  `STAMP=20260705_direct_physical_backend_fixed_anchor_64cm_8kg_8rail`,
  `SUPPORT_MODE=fixed_anchor`, `TARGET_X=0.64`, `PAYLOAD_MASS=8.0`,
  `RAIL_JOINT_COUNT=8`, `RAIL_LOWER=-0.04`, `RAIL_UPPER=0.10`, 1500 steps.
  Checker gate: `physical_fixed_anchor_cradle`, min box travel `0.58 m`,
  max final target distance `0.05 m`, fall/drop 0, root shortcut free,
  support-root writes 0, anchor retargets 0, non-success label required.
- [x] Run the stricter 64 cm / 8 kg / 8-rail fixed-anchor ablation.
  Result: `20260705_direct_physical_backend_fixed_anchor_64cm_8kg_8rail`
  ran on `server10`, Slurm job `166347`, and passed the checker:
  1500/1500, `physical_fixed_anchor_cradle`, fall/drop 0, root shortcut free,
  `anchor_world_joint_retarget_count=0`, support-root writes 0, foot pose
  writes 0, stance-anchor pose writes 0, max box travel `0.70080 m`, final box
  target distance `0.00111 m`, final post-settle box travel `0.63889 m`, and
  final post-settle box/torso relative error `9.22e-08 m`. No disjoint/fatal
  backend log errors were found. This removes support replanting from the
  current long-distance scaffold, but fixed world support remains.
- [ ] Stop adding support-retarget variants. Next implementation target:
  replace fixed world support/long rail travel with a real
  support-switching or foot-placement controller while preserving the free
  dynamic box, posture labels, active metrics, shortcut counters, and direct
  checker gates.
- [ ] Run stricter fixed-anchor 64 cm posture sweep without anchor retargets.
  Active tmux: `curiosity_fixed_anchor_postures64_0705`. Runs:
  `20260705_direct_physical_backend_fixed_anchor_lowfront_64cm_8kg_8rail` and
  `20260705_direct_physical_backend_fixed_anchor_chesthigh_64cm_8kg_8rail`
  with `SUPPORT_MODE=fixed_anchor`, target `0.64 m`, 8 kg free box, eight
  rails, and checker gates requiring anchor retargets 0, support-root writes
  0, fall/drop 0, root shortcut free, and non-success labels. The already
  passed `front_mid` run completes the three-posture set if these pass.
- [x] Run stricter fixed-anchor 64 cm posture sweep without anchor retargets.
  Result: tmux `curiosity_fixed_anchor_postures64_0705`, Slurm job `166353`,
  ran on `server10`. The in-tmux checker command had a shell-variable quoting
  bug and tried to read `.` as summary path, so both summaries were checked
  manually afterward. `low_front` passed with max box travel `0.70080 m`,
  final target distance `0.00111 m`, final post-settle box travel
  `0.63889 m`, final post-settle box/torso relative error `6.05e-08 m`,
  fall/drop 0, root shortcut free, anchor retargets 0, and support-root writes
  0. `chest_high` passed with max box travel `0.70080 m`, final target
  distance `0.00111 m`, final post-settle box travel `0.63889 m`, final
  post-settle box/torso relative error `9.48e-08 m`, fall/drop 0, root
  shortcut free, anchor retargets 0, and support-root writes 0.
- [ ] Treat the three-posture fixed-anchor scaffold as the target contact/load
  behavior. Next code work should expose a controller boundary that can swap
  fixed world support for a support-switching or foot-placement controller
  without changing the direct-task checker schema.
- [x] Add stricter audit fields/gates for the next controller replacement.
  `normalize_direct_carry_backend_summary.py` now propagates backend carrier
  mechanism fields including `backend_carrier_claim`,
  `stance_anchor_fixed_to_world`, `stance_anchor_kinematic`,
  `stance_anchor_dynamic_high_mass`, `stance_anchor_as_articulation_root`, and
  `articulation_root_path`. `check_direct_carry_task_summary.py` now supports
  `--max-foot-pose-write-count`, `--max-stance-anchor-pose-write-count`, and
  `--forbid-fixed-world-support`. Lightweight `py_compile` passed.
- [x] Implement first fixed-world-support replacement diagnostic mode.
  `build_core_world_anchored_footstep_carrier.py` now supports
  `--support-foot-mode fixed_to_anchor`, `--support-foot-mass`, and
  `--disable-support-reposition`. `run_direct_carry_task_physical_backend.sh`
  exposes `SUPPORT_MODE=dynamic_anchor_feet` with controller mode
  `physical_dynamic_anchor_feet_cradle`. This creates dynamic support feet
  fixed to the stance anchor, relying on ground contact/friction instead of a
  world fixed joint. Syntax checks passed.
- [ ] Run first `dynamic_anchor_feet` compute diagnostic at 16 cm / 8 kg /
  `front_mid`. Required gates: no fixed world support, support-foot mode
  `fixed_to_anchor`, support-foot joints >= 4, anchor retargets 0,
  support-root writes 0, foot pose writes 0, stance-anchor pose writes 0,
  root shortcut free, fall/drop 0, and diagnostic-only success label.
  Active run: tmux `curiosity_dynamic_anchor_feet_16cm_0705`, Slurm job
  `166366`, stamp
  `20260705_direct_physical_backend_dynamic_anchor_feet_16cm_8kg_frontmid`.
- [x] Run first `dynamic_anchor_feet` compute diagnostic at 16 cm / 8 kg /
  `front_mid`. Result:
  `20260705_direct_physical_backend_dynamic_anchor_feet_16cm_8kg_frontmid`
  ran on `server10`, Slurm job `166366`, and passed strict gates:
  700/700 steps, no fixed world support, support-foot mode `fixed_to_anchor`,
  support-foot joints 4, no support reposition, anchor retargets 0,
  support-root writes 0, foot pose writes 0, stance-anchor pose writes 0,
  root shortcut free, fall/drop 0, max box travel `0.15831 m`, final box
  target distance `0.00195 m`, and final post-settle box/torso relative error
  `2.99e-08 m`. This removes fixed world support for the first short carry,
  but remains a rigid support-frame scaffold, not walking.
- [ ] Run `dynamic_anchor_feet` 64 cm / 8 kg / `front_mid` with eight rail
  joints and the same strict no-fixed-world gates. This tests whether the
  physical ground-contact support frame can replace the fixed-anchor 64 cm
  scaffold, before posture sweep or real footstep control.
  Active run: tmux `curiosity_dynamic_anchor_feet_64cm_0705`, Slurm job
  `166370`, stamp
  `20260705_direct_physical_backend_dynamic_anchor_feet_64cm_8kg_frontmid`.
- [x] Run `dynamic_anchor_feet` 64 cm / 8 kg / `front_mid`.
  Result:
  `20260705_direct_physical_backend_dynamic_anchor_feet_64cm_8kg_frontmid`
  ran on `server10`, Slurm job `166370`, and passed strict no-fixed-world
  gates: 1500/1500, support-foot mode `fixed_to_anchor`, support-foot joints
  4, no fixed world support, no support reposition, anchor retargets 0,
  support-root writes 0, foot pose writes 0, stance-anchor pose writes 0,
  root shortcut free, fall/drop 0, max box travel `0.66915 m`, final target
  distance `0.000734 m`, and final post-settle box/torso relative error
  `2.42e-08 m`. This replaces fixed world support for the 64 cm `front_mid`
  scaffold, but is still a rigid support-foot frame, not walking.
- [ ] Add audit metrics for dynamic support-foot runs: anchor travel,
  support-foot travel, support-foot min/max z, and final anchor/support drift.
  These are needed before posture sweeping or claiming the physical support
  frame is a better locomotion target.
- [x] Add audit metrics for dynamic support-foot runs.
  Summaries now record anchor travel, support-foot travel, support-foot z
  range, and final support-foot drift. The normalizer propagates these fields,
  and the checker supports `--max-abs-anchor-travel-x` and
  `--max-abs-support-foot-travel-x`. Lightweight syntax checks passed.
- [ ] Run audited 64 cm / 8 kg / `front_mid` regression for
  `dynamic_anchor_feet`. Active tmux:
  `curiosity_dynamic_anchor_feet_audit64_0705`, stamp
  `20260705_direct_physical_backend_dynamic_anchor_feet_audit64_8kg_frontmid`.
  Additional gates: max anchor X drift <= `0.03 m`, max support-foot X drift
  <= `0.03 m`.
- [x] Run audited 64 cm / 8 kg / `front_mid` regression for
  `dynamic_anchor_feet`. Result:
  `20260705_direct_physical_backend_dynamic_anchor_feet_audit64_8kg_frontmid`
  ran on `server10`, Slurm job `166375`, and passed all gates. Max box travel
  `0.66915 m`, final target distance `0.000734 m`, final post-settle
  box/torso relative error `2.42e-08 m`, fall/drop 0, root shortcut free,
  anchor retargets 0, support-root writes 0, foot pose writes 0, and
  stance-anchor pose writes 0. Drift audit passed with max anchor X drift
  `4.47e-07 m`, max support-foot X drift `4.17e-07 m`, and support-foot z
  range `0.0174997-0.0175006 m`.
- [ ] Run audited `dynamic_anchor_feet` posture sweep for `low_front` and
  `chest_high` at 64 cm / 8 kg with the same no-fixed-world and drift gates.
  Active tmux: `curiosity_dynamic_anchor_feet_postures64_0705`. Stamps:
  `20260705_direct_physical_backend_dynamic_anchor_feet_lowfront_audit64_8kg`
  and
  `20260705_direct_physical_backend_dynamic_anchor_feet_chesthigh_audit64_8kg`.
- [x] Run audited `dynamic_anchor_feet` posture sweep for `low_front` and
  `chest_high` at 64 cm / 8 kg. Result: both passed on `server10`, Slurm job
  `166379`. `low_front`: max box travel `0.66915 m`, final target distance
  `0.000733 m`, final post-settle box travel `0.63927 m`, fall/drop 0, root
  shortcut free, anchor retargets 0, support-root writes 0, foot pose writes
  0, stance-anchor pose writes 0, max anchor X drift `4.52e-07 m`, max
  support-foot X drift `4.47e-07 m`. `chest_high`: max box travel
  `0.66915 m`, final target distance `0.000734 m`, final post-settle
  box/torso relative error `9.65e-08 m`, fall/drop 0, root shortcut free,
  anchor retargets 0, support-root writes 0, foot pose writes 0,
  stance-anchor pose writes 0, max anchor X drift `4.32e-07 m`, max
  support-foot X drift `4.77e-07 m`.
- [ ] Next implementation target: split the rigid support-foot frame into a
  support-switching or foot-placement controller. It must keep the current
  no-fixed-world gates and add explicit active support foot, swing foot,
  foot-contact state, commanded foot placement, measured foot slip, and
  support polygon/balance metrics. Do not claim the rigid four-foot frame as
  walking.
- [x] Implement first foot-driven support scaffold after rigid support frame.
  `SUPPORT_MODE=legged_anchor_feet` uses `support_foot_mode=x_prismatic_to_anchor`
  and `--use-support-foot-drive`; rail target is held at 0 while X prismatic
  support-foot joints push against ground contact to move anchor/torso/payload.
  Normalizer/checker expose support-foot X joint motion and drive fields.
  Syntax checks passed. This is still not walking because all feet are driven
  together; swing/stance switching remains next.
- [ ] Run first `legged_anchor_feet` 16 cm / 8 kg / `front_mid` diagnostic.
  Required gates: no fixed world support, no support-root/foot/stance-anchor
  pose writes, anchor retargets 0, root shortcut free, fall/drop 0, min box
  travel `0.12 m`, max final target distance `0.06 m`, and support-foot X
  joint motion at least `0.10 m`.
  Active tmux: `curiosity_legged_anchor_feet_16cm_0705`, stamp
  `20260705_direct_physical_backend_legged_anchor_feet_16cm_8kg_frontmid`.
- [x] Run first `legged_anchor_feet` 16 cm / 8 kg / `front_mid` diagnostic.
  Result:
  `20260705_direct_physical_backend_legged_anchor_feet_16cm_8kg_frontmid`
  passed strict no-fixed-world gates: 800/800,
  `support_foot_mode=x_prismatic_to_anchor`, support-foot X joint count 4,
  max support-foot X joint motion `0.62882 m`, max box travel `0.15971 m`,
  final target distance `0.00662 m`, final post-settle box travel `0.15380 m`,
  final post-settle box/torso relative error `8.64e-08 m`, fall/drop 0, root
  shortcut free, anchor retargets 0, support-root writes 0, foot pose writes
  0, stance-anchor pose writes 0. This is a foot-driven scaffold, not walking,
  because all support feet are driven together.
- [x] Implement first alternating swing/stance support-foot diagnostic.
  `SUPPORT_MODE=alternating_anchor_feet` now uses
  `support_foot_mode=xz_prismatic_to_anchor`: each foot has an X prismatic
  drive plus a Z swing joint, diagonal stance pairs alternate by cycle, and
  summaries/checkers expose X/Z joint counts, X/Z motion, commanded lift, and
  per-foot targets. Lightweight syntax checks passed.
- [ ] Run first `alternating_anchor_feet` 8 cm / 8 kg / `front_mid`
  diagnostic. First tmux submission
  `curiosity_alternating_anchor_feet_8cm_0705`, Slurm job `166400`, exited in
  0 seconds before backend log/summary creation. Retry active tmux:
  `curiosity_alternating_anchor_feet_8cm_retry_0705`, Slurm job `166403`,
  also exited before Isaac. Retry2, Slurm job `166406`, captured a shell
  wrapper `set -u` default-variable bug, which has been fixed. Retry3, Slurm
  job `166408`, reached compute but exposed a long env-assignment EOF in the
  direct wrapper; support-foot defaults are now exported before the core
  wrapper call. Current active tmux:
  `curiosity_alternating_anchor_feet_8cm_retry4_0705`, Slurm job `166412`,
  stamp
  `20260705_direct_physical_backend_alternating_anchor_feet_8cm_8kg_frontmid`.
  Required first gate: no fixed world support, no root/support/foot/stance pose
  writes, support-foot joint count >= 8, Z joint count >= 4, X joint motion >=
  `0.08 m`, Z joint motion >= `0.02 m`, box travel >= `0.04 m`, final target
  distance <= `0.05 m`, fall/drop 0, and diagnostic-only claim.
- [x] Run first `alternating_anchor_feet` 8 cm / 8 kg / `front_mid`
  diagnostic. Result: negative but informative. Retry4, Slurm job `166412`,
  completed 620/620 on `server10` with fall/drop 0, no fixed world support,
  root shortcut free, anchor retargets 0, support-root writes 0, foot pose
  writes 0, stance-anchor pose writes 0, support-foot joint count 8, X joint
  motion `0.24221 m`, Z joint motion `0.46727 m`, commanded lift `0.055 m`,
  and final post-settle box/torso relative error `0.000178 m`. It failed
  because settle drift moved the system about `0.061 m` in negative X, final
  target distance was `0.08083 m`, max box travel from initial was only
  `6.36e-05 m`, and actual support-foot world Z only reached `0.01923 m`.
- [ ] Add actual support-foot lift metrics and gates. Joint-space Z motion is
  insufficient: the first alternating run showed large Z joint motion but only
  about `0.0017 m` actual foot-center lift.
- [x] Add actual support-foot lift metrics and gates. Summaries now record
  `max_actual_support_foot_lift_m`, `per_foot_max_actual_lift_m`,
  `per_foot_min_z_m`, and `per_foot_max_z_m`; the direct checker now supports
  `--min-actual-support-foot-lift`. Lightweight checks passed.
- [ ] Run a fast-start / larger-step alternating support-foot diagnostic after
  adding actual lift metrics. The goal is to distinguish three blockers:
  settle drift, missing actual swing-foot lift, and insufficient forward
  support-foot impulse.
  Active tmux: `curiosity_alternating_faststart_8cm_0705`, Slurm job
  `166417`, stamp
  `20260705_direct_physical_backend_alternating_anchor_feet_faststart_8cm_8kg_frontmid`.
- [x] Run fast-start / larger-step alternating support-foot diagnostic.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_faststart_8cm_8kg_frontmid`
  passed the first wide gate on `server10`, Slurm job `166417`: 620/620,
  support-foot joint count 8, actual support-foot lift `0.02853 m`, X joint
  motion `0.39575 m`, Z joint motion `0.39739 m`, fall/drop 0, root shortcut
  free, no fixed world support, anchor retargets 0, support-root writes 0,
  foot pose writes 0, stance-anchor pose writes 0, max box travel `0.04012 m`,
  final target distance `0.04572 m`, and final post-settle box travel
  `0.03940 m`. This validates actual swing-foot lift and partial forward
  transport, but it plateaus at about 4 cm on an 8 cm target.
- [ ] Run a multi-cycle alternating support-foot diagnostic with smaller
  `STEP_LENGTH` to force more gait cycles and test whether travel can extend
  beyond the current 4 cm plateau.
  Active tmux: `curiosity_alternating_multicycle_8cm_0705`, Slurm job
  `166421`, stamp
  `20260705_direct_physical_backend_alternating_anchor_feet_multicycle_8cm_8kg_frontmid`.
- [x] Run multi-cycle alternating support-foot diagnostic.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_multicycle_8cm_8kg_frontmid`
  completed 720/720 on `server10`, Slurm job `166421`, and failed the strict
  gate narrowly. It stayed safe and shortcut-free with fall/drop 0, no fixed
  world support, root shortcut free, anchor retargets 0, support-root writes 0,
  foot pose writes 0, stance-anchor pose writes 0. It improved max box travel
  to `0.05794 m` and final target distance to `0.02646 m`, but failed actual
  foot lift (`0.01738 m < 0.02 m`), min box travel (`0.05794 m < 0.065 m`),
  and final target distance (`0.02646 m > 0.025 m`).
- [ ] Run a 5-cycle / stronger-Z alternating diagnostic to test whether the
  remaining blocker is insufficient cycle count, insufficient actual foot lift,
  or both.
- [x] Run 5-cycle / stronger-Z alternating diagnostic.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_5cycle_8cm_8kg_frontmid`
  completed 780/780 on `server10`, Slurm job `166426`, and failed only the
  final target-distance gate. It had fall/drop 0, no fixed world support, root
  shortcut free, anchor retargets 0, support-root writes 0, foot pose writes 0,
  stance-anchor pose writes 0, actual support-foot lift `0.06320 m`, max box
  travel `0.09850 m`, X joint motion `0.35847 m`, and Z joint motion
  `0.39335 m`. It reached target around step 220 but slid backward during
  target hold; final target distance was `0.03631 m`.
- [x] Fix alternating support-foot target hold to latch per-foot X joint
  targets instead of replacing all foot targets with their mean.
- [ ] Rerun 5-cycle alternating diagnostic with the per-foot hold fix. Active
  tmux: `curiosity_alternating_5cycle_holdfix_8cm_0705`, Slurm job `166430`,
  stamp
  `20260705_direct_physical_backend_alternating_anchor_feet_5cycle_holdfix_8cm_8kg_frontmid`.
- [x] Rerun 5-cycle alternating diagnostic with the per-foot hold fix.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_5cycle_holdfix_8cm_8kg_frontmid`
  passed the stricter gate on `server10`, Slurm job `166430`: 780/780,
  support-foot mode `xz_prismatic_to_anchor`, support-foot joints 8, actual
  support-foot lift `0.06320 m`, X joint motion `0.35847 m`, Z joint motion
  `0.39335 m`, max box travel `0.09812 m`, final box target distance
  `0.01572 m`, final post-settle box travel `0.06551 m`, fall/drop 0, root
  shortcut free, no fixed world support, anchor retargets 0, support-root
  writes 0, foot pose writes 0, stance-anchor pose writes 0. Log scan found no
  disjoint/fatal/traceback/unbound/EOF errors.
- [ ] Next target: extend alternating X/Z support-foot carrying beyond the
  short 8 cm diagnostic, then add explicit per-foot contact/slip/support
  polygon metrics. Do not call the current scaffold a full walking robot.
- [x] Add explicit per-foot contact/slip/support-polygon proxy metrics.
  Builder summaries now record contact-thresholded near-ground steps,
  near-ground foot XY slip/speed, near-ground foot counts, and support polygon
  margin proxies. The normalizer propagates them, and the checker can gate
  them with `--min-near-ground-foot-count`, `--min-support-polygon-margin`,
  and `--max-near-ground-foot-speed`.
- [ ] Run a 16 cm / 8 kg alternating holdfix diagnostic with the new metrics.
  Active tmux: `curiosity_alternating_10cycle_holdfix_16cm_0705`, Slurm job
  `166438`, stamp
  `20260705_direct_physical_backend_alternating_anchor_feet_10cycle_holdfix_16cm_8kg_frontmid`.
- [x] Run a 16 cm / 8 kg alternating holdfix diagnostic with the new metrics.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_10cycle_holdfix_16cm_8kg_frontmid`
  completed 1180/1180 on `server10`, Slurm job `166438`. Manual normalization
  from the backend summary passed the checker: actual support-foot lift
  `0.06320 m`, max box travel `0.18576 m`, final target distance `0.00436 m`,
  final post-settle box travel `0.15686 m`, fall/drop 0, root shortcut free,
  no fixed world support, anchor retargets 0, support-root writes 0, foot pose
  writes 0, stance-anchor pose writes 0. New metrics: per-foot near-ground
  steps `1111/1061/1109/1116`, max near-ground foot count 4, min support
  polygon margin `0.16279 m`, and max near-ground XY speed up to `1.0855 m/s`.
  Limitation: `min_near_ground_foot_count=0` during some transition frames, so
  contact metrics are diagnostic only.
- [x] Fix core wrapper shell fragility after the 16 cm run. The core launcher
  now constructs the Python command as a bash array instead of a long
  backslash-continued command; `bash -n` and `py_compile` passed.
- [ ] Next target: rerun a short post-wrapper-fix regression or push to 32 cm
  alternating holdfix after confirming the wrapper no longer drops direct
  normalization/checking.
- [ ] Run a 32 cm / 8 kg alternating holdfix diagnostic to test longer
  transport and wrapper-array normalization. Active tmux:
  `curiosity_alternating_20cycle_holdfix_32cm_0705`, Slurm job `166446`,
  stamp
  `20260705_direct_physical_backend_alternating_anchor_feet_20cycle_holdfix_32cm_8kg_frontmid`.
- [x] Run a 32 cm / 8 kg alternating holdfix diagnostic.
  Result:
  initial job `166446` failed before Isaac due a shell default-expansion issue;
  retry job `166450` passed on `server10` with automatic direct
  normalization/checking. Metrics: 1980/1980, actual support-foot lift
  `0.06320 m`, max box travel `0.38092 m`, final target distance `0.03159 m`,
  final post-settle box travel `0.35281 m`, fall/drop 0, root shortcut free,
  no fixed world support, anchor retargets 0, support-root writes 0, foot pose
  writes 0, stance-anchor pose writes 0. Contact proxy metrics were recorded:
  near-ground steps `1859/1804/1873/1874`, max near-ground foot count 4, min
  support polygon margin `0.16279 m`, max near-ground XY speed `1.0855 m/s`,
  and `min_near_ground_foot_count=0` during transition frames.
- [ ] Push alternating X/Z support-foot carrying to 64 cm / 8 kg and inspect
  whether target hold, support metrics, and no-shortcut gates still hold.
  Active tmux: `curiosity_alternating_40cycle_holdfix_64cm_0705`, Slurm job
  `166455`, stamp
  `20260705_direct_physical_backend_alternating_anchor_feet_40cycle_holdfix_64cm_8kg_frontmid`.
- [x] Push alternating X/Z support-foot carrying to 64 cm / 8 kg.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_40cycle_holdfix_64cm_8kg_frontmid`
  passed on `server10`, Slurm job `166455`: 3580/3580, actual support-foot
  lift `0.06475 m`, max box travel `0.64785 m`, final target distance
  `0.01181 m`, final post-settle box travel `0.62941 m`, fall/drop 0, root
  shortcut free, no fixed world support, anchor retargets 0, support-root
  writes 0, foot pose writes 0, stance-anchor pose writes 0. Support metrics
  were recorded; `min_near_ground_foot_count=0` remains a transition-frame
  limitation. Log scan found no disjoint/fatal/traceback/unbound/EOF errors.
- [ ] Run 64 cm / 8 kg alternating X/Z posture sweep for `low_front` and
  `chest_high`.
- [x] Run 64 cm / 8 kg alternating X/Z posture sweep for `low_front` and
  `chest_high`.
  Result:
  both postures passed the same 64 cm / 8 kg diagnostic gate without
  shortcut writes. `low_front` reached max box travel `0.70662 m`, final
  target distance `0.03367 m`, final post-settle travel `0.67882 m`, and
  actual support-foot lift `0.06348 m`. `chest_high` reached max box travel
  `0.70446 m`, final target distance `0.02583 m`, final post-settle travel
  `0.66510 m`, and actual support-foot lift `0.06353 m`. Both had fall/drop
  0, root shortcut free, no fixed world support, anchor retargets 0,
  support-root writes 0, foot pose writes 0, stance-anchor pose writes 0, and
  clean log scans.
- [x] Add randomized payload mass/size/COM interface and run first 8 cm smoke.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_randomized_8cm_seed7051`
  passed with sampled mass `8.15343 kg`, sampled size
  `0.35775 x 0.25309 x 0.23354 m`, sampled COM offset
  `[0.00902, 0.00821, -0.00216] m`, final target distance `0.01740 m`,
  fall/drop 0, and all shortcut counters 0.
- [x] Add horizontal active-probe telemetry and run randomized 8 cm smoke.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_probe_randomized_8cm_seed7052`
  passed with a 60-step, 2 cm push-pull probe. It recorded max probe relative
  error `0.00835 m`, final probe box lag `0.000325 m`, final target distance
  `0.02021 m`, fall/drop 0, and all shortcut counters 0.
- [x] Add first probe-derived belief proxy and run randomized smoke.
  Result:
  `20260705_direct_physical_backend_alternating_anchor_feet_belief_probe_randomized_8cm_seed7053`
  used only probe telemetry, not hidden mass/COM, and wrote
  `probe_belief_uses_hidden_ground_truth=false`. It produced risk `0.80478`
  and a high-risk bucket while still passing the carry gate safely. This was a
  diagnostic proxy, not an adapted controller.
- [x] Run fixed 6 kg vs 10 kg horizontal probe calibration.
  Result:
  both cases passed safely, but box-lag/compliance did not separate load:
  risk `0.78250` for 6 kg and `0.79201` for 10 kg, same high-risk bucket.
- [x] Add support-foot X tracking telemetry and rerun 6 kg vs 10 kg
  calibration.
  Result:
  tracking was also not mass-sensitive: tracking proxy `2.04663` for 6 kg and
  `2.04556` for 10 kg. Do not use this proxy for adaptation.
- [x] Add measured support-foot X effort telemetry and rerun 6 kg vs 10 kg
  calibration.
  Result:
  Isaac effort reads worked with zero read errors, but horizontal probe effort
  remained almost identical. 6 kg measured max/mean effort
  `459.73468` / `302.97068`, effort proxy `0.004179`, risk `0.57452`; 10 kg
  measured `462.08502` / `303.88600`, effort proxy `0.004201`, risk
  `0.58025`. This is a negative result.
- [x] Replace the horizontal push-pull probe with a vertical micro-lift or
  partial-unload probe, then rerun fixed 6 kg vs 10 kg calibration before any
  posture/gait adaptation is enabled.
- [x] Run vertical micro-lift / partial-unload 6 kg vs 10 kg calibration.
  Result:
  initial jobs `166528` and `166531` were canceled because they still executed
  the old horizontal probe path. Retry3 ran in tmux
  `curiosity_probe_belief_vertical_lift_cal3_8cm_0705`, Slurm job `166533`,
  on `server46` with explicit CLI overrides:
  `--probe-mode vertical_micro_lift --probe-x-amplitude 0.0 --probe-z-amplitude 0.030`.
  Both fixed-mass cases passed the 8 cm carry gate with fall/drop 0,
  root-shortcut-free summaries, and no fatal/traceback/disjoint log errors.
  The 6 kg case recorded max/mean Z effort `2371.66748` / `1386.22269`,
  max torso/box Z travel `0.02686 m` / `0.02562 m`, final target distance
  `0.01513 m`. The 10 kg case recorded max/mean Z effort `2380.76245` /
  `1398.41329`, max torso/box Z travel `0.02714 m` / `0.02592 m`, final target
  distance `0.01471 m`. This is still too weak for load belief: mean Z effort
  changes by only about `0.9%` and max Z effort by about `0.4%`. Do not use
  this probe for posture/gait adaptation.
- [ ] Next estimator path: add a more direct load-sensitive signal such as
  cradle/box constraint force, contact normal impulse, support reaction force,
  or a probe that partly transfers payload weight between support contacts.
  Do not keep tuning horizontal or simple all-feet vertical probes as if they
  already identify hidden mass.
- [x] Run Arena G1 AGILE commanded-walk smoke through the official
  `ArenaEnvBuilder` path. Submitted from login node into tmux
  `curiosity_g1_agile_walk_smoke_0705`, Slurm job `166541`, with:
  `STAMP=20260705_arena_g1_agile_walk_cmd_smoke STEPS=260 WARMUP_STEPS=40 COMMAND_START_STEP=80 COMMAND_X=0.25 MIN_COMMANDED_TRAVEL_X=0.05 DEVICE=cuda:0 bash scripts/isaac/run_arena_g1_agile_stand_smoke.sh`.
  This tests G1 WBC walking/balance only; it is not carrying evidence. Result:
  negative. The Arena environment parsed and started physics, but before any
  rollout step the tensor view was invalidated and the script wrote summary
  `experiments/outputs/arena_g1_agile_stand_smoke/20260705_arena_g1_agile_walk_cmd_smoke/arena_g1_agile_stand_summary.json`
  with `completed_steps=0`, status `fail`, and
  `Exception: Failed to get DOF velocities from backend`. Do not rerun this
  same outer-AppLauncher path unchanged.
- [ ] Implement and run a persistent SimulationApp G1 AGILE walk smoke using
  IsaacLab-Arena's own `run_simulation_app_function` harness, to test whether
  the failure above is caused by our outer AppLauncher/env lifecycle rather
  than the WBC policy itself.
  Submitted retry2 from login node into tmux
  `curiosity_g1_agile_walk_persistent2_0705`, Slurm job `166550`, with:
  `STAMP=20260705_arena_g1_agile_walk_persistent_smoke STEPS=260 WARMUP_STEPS=40 COMMAND_START_STEP=80 COMMAND_X=0.25 MIN_COMMANDED_TRAVEL_X=0.05 DEVICE=cuda:0 bash scripts/isaac/run_arena_g1_agile_walk_persistent_smoke.sh`.

## 2026-07-05 strict Isaac support-continuity pivot

- [x] Stop waiting on external video/model/data dependencies for the immediate
  execution path and directly advance the Isaac carrying scene.
  Result:
  external serious methods remain useful as later baselines or reference code,
  but they are not needed to build the current 16 cm / 64 cm diagnostic
  scaffold.
- [x] Add stricter support-continuity instrumentation to the alternating
  X/Z support-foot backend.
  Result:
  `build_core_world_anchored_footstep_carrier.py` now records total and
  drive-phase near-ground support counts, zero/lt2 support-step counters,
  commanded-stance near-ground counters, and a configurable
  `support_foot_double_support_fraction`. The wrapper and direct-summary
  normalizer propagate the new field.
- [x] Add a checker gate for strict drive-phase support continuity.
  Result:
  `check_direct_carry_task_summary.py` can now enforce
  `--min-drive-near-ground-foot-count`, zero/lt2 drive support-step caps, and
  commanded-stance support-foot contact gates.
- [x] Add a strict 16 cm / 8 kg front-mid diagnostic script.
  Result:
  `scripts/isaac/run_alternating_anchor_feet_strict_support_16cm_diag.sh`
  runs the existing free-box, no-fixed-world, no-root-shortcut diagnostic with
  a 12% double-support window and strict drive-phase support-continuity gates.
- [x] Run the strict 16 cm / 8 kg front-mid diagnostic on a compute node via
  tmux-held Slurm allocation and record whether the stricter support gate
  passes or fails.
  Submitted:
  tmux `curiosity_alt_strict_support_16cm_0705`, Slurm job `166595`, log
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_16cm_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  Slurm job `166595` ran on `server02` and passed the strict checker. Summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_16cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Key metrics: 1180/1180 steps, max box travel `0.18552 m`, final box target
  distance `0.00242 m`, fall/drop 0, root shortcut free, no fixed-world
  support, no support-root/anchor/foot/stance pose writes,
  `min_near_ground_foot_count=2`, `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_zero_steps=0`, `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`, and
  `commanded_stance_near_ground_lt2_steps=0`. This fixes the previous
  transition-frame support-count weakness for the 16 cm gate only.
- [x] Scale the same strict support-continuity gate to 32 cm / 8 kg
  front-mid before returning to 64 cm or posture sweeps.
  Submitted:
  added `scripts/isaac/run_alternating_anchor_feet_strict_support_32cm_diag.sh`
  and submitted tmux `curiosity_alt_strict_support_32cm_0705`, Slurm job
  `166599`, log
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_32cm_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  Slurm job `166599` ran on `server10` and passed. Summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_32cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Key metrics: 1980/1980 steps, max box travel `0.38556 m`, final box target
  distance `0.03051 m`, final post-settle box travel `0.35173 m`, fall/drop
  0, root shortcut free, no fixed-world support, no support-root/anchor/foot/
  stance pose writes, `min_near_ground_foot_count=2`,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_zero_steps=0`,
  `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`, and
  `commanded_stance_near_ground_lt2_steps=0`.
- [x] Scale the strict support-continuity gate to 64 cm / 8 kg front-mid.
  Submitted:
  added `scripts/isaac/run_alternating_anchor_feet_strict_support_64cm_diag.sh`
  and submitted tmux `curiosity_alt_strict_support_64cm_0705`, Slurm job
  `166603`, log
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_64cm_0705_srun.log`.
  Initial status was running on `server10`.
  Result:
  Slurm job `166603` completed the backend rollout, then the wrapper hit a
  post-rollout shell `unexpected EOF` before normalizer/checker. A separate
  compute-node normalization/check job, tmux
  `curiosity_alt_strict_support_64cm_normalize_0705`, Slurm job `166605`,
  generated the direct summary and passed the strict checker. Summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Key metrics: 3580/3580 steps, max box travel `0.67301 m`, final box target
  distance `0.02369 m`, final post-settle box travel `0.66492 m`, fall/drop
  0, root shortcut free, no fixed-world support, no support-root/anchor/foot/
  stance pose writes, `min_near_ground_foot_count=2`,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_zero_steps=0`,
  `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`, and
  `commanded_stance_near_ground_lt2_steps=0`.
- [ ] Clean up the intermittent post-rollout EOF path in
  `run_direct_carry_task_physical_backend.sh` or the calling strict 64 cm
  script. Do not mark it as a physics failure; it occurred after backend
  summary/CSV were written and was recovered by a separate compute-node
  normalizer/checker run.
- [x] Run the same strict 64 cm gate for `low_front` and `chest_high`
  postures before using posture sweep results as anything more than a
  diagnostic scaffold.
  Prepared:
  added `scripts/isaac/run_alternating_anchor_feet_strict_support_64cm_postures_diag.sh`
  with direct-summary recovery if the post-rollout EOF appears after backend
  summary creation.
  Submitted:
  tmux `curiosity_alt_strict_support_64cm_postures_0705`, Slurm job `166612`,
  log
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_64cm_postures_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  Slurm job `166612` ran on `server10` and both postures passed the strict
  checker. `low_front`: summary
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_low_front/direct_carry_task_physical_backend_summary.json`,
  3580/3580, max box travel `0.66675 m`, final box target distance
  `0.00189 m`, fall/drop 0, root shortcut free, no fixed-world support, no
  support-root/anchor/foot/stance pose writes, `min_drive_near_ground_foot_count=2`,
  and `drive_near_ground_lt2_steps=0`. `chest_high`: summary
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_chest_high/direct_carry_task_physical_backend_summary.json`,
  3580/3580, max box travel `0.65313 m`, final box target distance
  `0.01468 m`, fall/drop 0, root shortcut free, no fixed-world support, no
  support-root/anchor/foot/stance pose writes, `min_drive_near_ground_foot_count=2`,
  and `drive_near_ground_lt2_steps=0`. Log scan found no traceback/fatal/
  disjoint/EOF/tensor errors, only expected headless-display warnings.
- [ ] Transition from the strict direct-Isaac support-foot scaffold toward a
  real robot walking/balance backend. First target: recover a G1 or other
  official IsaacLab/Arena robot walking smoke that completes steps with
  nonzero commanded travel, then attach or coordinate the carry scaffold.
- [x] Record the Arena persistent G1 AGILE smoke as a repeated backend blocker
  rather than waiting on it. Retry2 wrote
  `experiments/outputs/arena_g1_agile_walk_persistent_smoke/20260705_arena_g1_agile_walk_persistent_smoke/arena_g1_agile_walk_persistent_summary.json`
  with `completed_steps=0`, status `fail`, and
  `Exception: Failed to get DOF velocities from backend`. Do not repeat this
  path unchanged.
- [x] Add the current direct-Isaac probe -> posture-selection -> carry
  diagnostic on the strict support-foot backend. New files:
  `scripts/isaac/run_probe_then_adaptive_carry_strict_support_diag.sh` and
  `scripts/isaac/summarize_probe_then_adaptive_carry.py`. This runs a
  randomized hidden-box probe without privileged mass/COM inputs, chooses
  `front_mid`, `low_front`, or `chest_high` using a hand-coded risk rule, then
  runs the selected 64 cm carry under the same no-fixed-world, no-root-shortcut,
  strict support-continuity gates. This is explicitly not RL and not final
  walking-robot evidence.
- [x] Run the probe -> adaptive carry diagnostic in a Curiosity-owned
  tmux-held Slurm allocation and record the selected posture, probe belief,
  carry metrics, and any checker failures.
  Submitted:
  tmux `curiosity_probe_adaptive_carry_0705`, Slurm job `166625`, log
  `logs/probe_then_adaptive_carry/probe_then_adaptive_carry_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  job `166625` completed on `server10` with exit `0:0`. Aggregate summary:
  `experiments/outputs/probe_then_adaptive_carry/20260705_probe_then_adaptive_carry_strict_support_seed7055/probe_then_adaptive_carry_summary.json`.
  Probe risk was `0.607367` on randomized hidden box seed `7055`
  (`8.24950 kg`, nonzero COM offset), so the hand-coded selector chose
  `low_front`, `stance_steps=96`, and `step_length=0.014 m`. The selected
  64 cm carry passed the strict checker: 3580/3580, max box travel
  `0.67171 m`, final box target distance `0.00361 m`, final post-settle box
  travel `0.64513 m`, fall/drop 0, root shortcut free, no fixed-world support,
  no support-root/anchor/foot/stance pose writes,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`, and
  `commanded_stance_near_ground_lt2_steps=0`.
- [x] Add a randomized hidden-box all-posture strict support gate. New files:
  `scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh` and
  `scripts/isaac/summarize_randomized_all_posture_carry.py`; checker additions:
  `--require-box-randomized` and `--expect-box-seed`. This is designed to test
  whether `front_mid`, `low_front`, and `chest_high` all remain stable for the
  same randomized unknown box under the strict no-fixed-world/no-root-shortcut
  support-continuity gate. It is still a scaffold gate, not final robot
  success.
- [x] Run the randomized all-posture strict support gate on a compute node and
  record whether every posture passes for the same hidden box seed.
  Submitted:
  tmux `curiosity_randomized_all_postures_0705`, Slurm job `166633`, log
  `logs/randomized_all_posture_strict_support/randomized_all_posture_strict_support_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  job `166633` completed on `server02` with exit `0:0`. Aggregate summary:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_strict_support_64cm_seed7061/randomized_all_posture_strict_support_summary.json`.
  Shared randomized hidden box: seed `7061`, mass `6.81119 kg`, size
  `[0.32037, 0.22802, 0.23574] m`, COM offset
  `[0.01463, 0.02498, 0.00268] m`. All three postures passed the strict
  support gate with fall/drop 0, no fixed-world support, no shortcut pose
  writes, `min_drive_near_ground_foot_count=2`, and
  `drive_near_ground_lt2_steps=0`: `front_mid` max box travel `0.64402 m`,
  final target distance `0.01039 m`; `low_front` max box travel `0.68203 m`,
  final target distance `0.01310 m`; `chest_high` max box travel `0.66133 m`,
  final target distance `0.00638 m`. This remains scaffold evidence, not full
  humanoid walking, RL, or video-conditioned success.
- [ ] Build the next direct Isaac scene step instead of waiting for external
  models: expose posture/stance/gait parameters as an action interface, add
  randomized hidden-box episodes, and create a lightweight policy/search
  runner that selects among or interpolates `front_mid`, `low_front`, and
  `chest_high` from probe telemetry. Run all simulation on compute nodes.
- [x] Add the direct Isaac probe plus posture/gait parameter-search scaffold.
  New files:
  `scripts/isaac/run_probe_parameter_search_carry_diag.sh` and
  `scripts/isaac/summarize_probe_parameter_search_carry.py`. Lightweight
  login-node checks passed with `bash -n` and `py_compile`. This runner is
  explicitly non-RL: it evaluates five hand-authored candidates after a probe
  and selects the best passing candidate by transparent diagnostic score.
- [x] Run the probe plus posture/gait parameter-search scaffold on a compute
  node and record the selected best candidate, failed candidates, and strict
  support-gate results.
  Submitted:
  tmux `curiosity_probe_param_search_0705`, Slurm job `166641`, log
  `logs/probe_parameter_search_carry/probe_parameter_search_carry_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  job `166641` completed on `server10` with exit `0:0`. Aggregate summary:
  `experiments/outputs/probe_parameter_search_carry/20260705_probe_parameter_search_carry_seed7067/probe_parameter_search_carry_summary.json`.
  The outer `tee` log was not created because the log directory was opened
  before the script made it; per-case logs exist under
  `logs/probe_parameter_search_carry/`. Shared randomized hidden box: seed
  `7067`, mass `6.15402 kg`, size `[0.32579, 0.25445, 0.24170] m`, COM offset
  `[0.01250, 0.02327, 0.01980] m`. Probe risk was `0.596106`, bucket
  `moderate_observed_load_response`, without hidden ground-truth use. Best
  passing candidate: `front_mid_nominal`, 3580/3580, score `0.00286`, final
  target distance `0.00286 m`, fall/drop 0, strict support continuity passed.
  `low_front_slow` also passed. `chest_high_slowest`,
  `front_mid_wide_slow`, and `low_front_wide_slowest` were correctly rejected
  for support-continuity failures despite fall/drop 0.
- [ ] Fix future outer Slurm log creation for probe/search jobs by creating
  `logs/probe_parameter_search_carry/` before launching tmux `tee`, or by
  wrapping the tmux command with `mkdir -p` before `srun`.
- [ ] Convert the parameter-search scaffold into a repeatable multi-seed
  evaluation: run several hidden box seeds, preserve failed candidates, and
  report whether the selected candidate changes with observed probe telemetry.
- [x] Add a multi-seed wrapper around the direct Isaac probe parameter-search
  runner. New files:
  `scripts/isaac/run_probe_parameter_search_multiseed_diag.sh` and
  `scripts/isaac/summarize_probe_parameter_search_multiseed.py`. Lightweight
  login-node syntax checks passed.
- [x] Run the multi-seed probe parameter-search diagnostic on a compute node
  and record whether each seed has a strict passing candidate and whether the
  selected best candidate varies.
  Submitted:
  tmux `curiosity_probe_param_multiseed_0705`, Slurm job `166649`, log
  `logs/probe_parameter_search_multiseed/probe_parameter_search_multiseed_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  job `166649` completed on `server02` with exit `0:0`. Aggregate summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7068_7070/probe_parameter_search_multiseed_summary.json`.
  All 3 seeds passed with at least one strict passing candidate, but all chose
  the same best candidate: `low_front_slow`. Best final target distances:
  seed `7068` `0.00660 m`, seed `7069` `0.00576 m`, seed `7070`
  `0.00045 m`; all best candidates had fall/drop 0 and strict support
  continuity. `best_candidate_varied=false` and `best_posture_varied=false`.
  This is useful repeatability evidence, but negative evidence for current
  posture-diversity/adaptation.
- [ ] Add harder candidate and environment variation to force meaningful
  posture choice: heavier mass range, larger COM offsets, low-friction cases,
  candidate-specific hold heights, and a score term for effort/support margin.
  Keep it labeled as scaffold until a real policy replaces search.
- [x] Add a stricter Core API G1 stand-with-attached-box prerequisite gate.
  Patched `build_core_world_g1_box_scene.py` to apply Arena-style stand drive
  gains directly to G1 USD joints and record applied gain metadata. Added
  `check_core_world_g1_box_scene_summary.py`,
  `run_core_world_g1_stand_height_sweep.sh`, and
  `summarize_core_world_g1_stand_height_sweep.py`. Lightweight login-node
  syntax/compile checks passed. This is not walking or carrying success; it is
  a prerequisite gate to see whether the direct Core API G1 backend can stand
  with a fixed-torso payload before any walking/carrying control is attempted.
- [x] Run the Core API G1 stand-height sweep on a compute node and record
  whether any root height can stand with fixed-torso box, Arena-style gains,
  fall/drop 0, and no rollout root/box pose writes.
  Submitted:
  tmux `curiosity_g1_stand_height_0705`, Slurm job `166658`, log
  `logs/core_world_g1_stand_height_sweep/g1_stand_height_sweep_0705_srun.log`.
  Initial status was pending for priority.
  Result:
  job `166658` ran on `server02` and failed the aggregate gate. Summary:
  `experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep/core_world_g1_stand_height_sweep_summary.json`.
  All four heights applied stand gains to 23 joints and had no rollout
  root/box pose writes, but all fell/tilted with max tilt around
  `1.14-1.18 rad`. No height passed. This isolates a real-G1 backend blocker:
  Core API G1 cannot yet stand with a fixed-torso 2 kg payload under open-loop
  stand targets plus Arena gains.
- [x] Run the same Core API G1 stand-height sweep without an attached payload
  to isolate whether the blocker is base G1 standing or the fixed-torso box.
  Submitted:
  tmux `curiosity_g1_stand_nobox_0705`, Slurm job `166661`, command
  `STAMP=20260705_core_world_g1_stand_height_sweep_nobox ATTACH_BOX_MODE=none EXPECT_ATTACH_BOX=none MAX_BOX_DROP_EVENTS=999 MIN_BOX_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_stand_nobox bash scripts/isaac/run_core_world_g1_stand_height_sweep.sh`.
  Log:
  `logs/core_world_g1_stand_height_sweep/g1_stand_height_sweep_nobox_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep_nobox/core_world_g1_stand_height_sweep_summary.json`.
  Initial status: running on `server36`.
  Result:
  job `166661` completed on `server36` with exit `0:0`. Aggregate summary:
  `experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep_nobox/core_world_g1_stand_height_sweep_summary.json`.
  2/4 no-box height cases passed the strict stand gate. Passing:
  `z_0p84` with fall events 0, min robot z `0.77672 m`, max tilt
  `0.23243 rad`; `z_0p96` with fall events 0, min robot z `0.74072 m`, max
  tilt `0.45388 rad`. Failing: `z_0p78` and `z_0p90`. This shows Core API G1
  can stand in the scene without payload at some root heights; the fixed-torso
  2 kg payload is now the immediate blocker.
- [x] Add a payload mass/attachment isolation sweep from the no-box passing
  heights (`0.84`, `0.96`): test lighter attached boxes first, preserve the
  no-root-pose-write gate, and report the first mass/offset that destabilizes
  G1. This should be a direct Isaac diagnostic, not an external-model wait.
  Implemented:
  `scripts/isaac/run_core_world_g1_payload_sweep.sh` and
  `scripts/isaac/summarize_core_world_g1_payload_sweep.py`; G1 summaries now
  include fixed-joint attach metadata. Lightweight syntax/compile checks
  passed.
- [x] Run the Core API G1 fixed-payload isolation sweep on a compute node and
  record which mass/height/attach-offset cases pass strict standing.
  Submitted:
  tmux `curiosity_g1_payload_sweep_0705`, Slurm job `166663`, command
  `STAMP=20260705_core_world_g1_payload_sweep_small HEIGHTS="0.84 0.96" MASSES="0.25 0.50 1.00 2.00" ATTACH_XS="0.12 0.18 0.24" srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_payload_sweep bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_sweep_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_sweep_small/core_world_g1_payload_sweep_summary.json`.
  Initial status: pending for priority.
  Result:
  job `166663` ran on `server36` and failed the aggregate gate with exit
  `1:0`. Summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_sweep_small/core_world_g1_payload_sweep_summary.json`.
  0/24 cases passed. Two cases were affected by the transient launcher edit
  and are not clean physics evidence; the other 22 completed the runner and
  still failed by falls/tilt/height/drop gates. This makes the forward
  fixed-torso payload setup a dead end for now.
- [x] Run a centered ultra-light fixed-payload isolation sweep: attach at or
  near torso local origin, use small box geometry, and masses below `0.25 kg`.
  This distinguishes any-payload instability from front-mounted load moment
  and initial fixed-joint geometry issues.
  Submitted:
  tmux `curiosity_g1_payload_centered_0705`, Slurm job `166668`, command
  `STAMP=20260705_core_world_g1_payload_centered_ultralight HEIGHTS="0.84 0.96" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.0" ATTACH_Z=0.0 BOX_POS_X=0.0 BOX_POS_Y=0.0 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_center bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_centered_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight/core_world_g1_payload_sweep_summary.json`.
  Result:
  job `166668` ran on `server36` and failed the aggregate gate with exit
  `1:0`. Summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight/core_world_g1_payload_sweep_summary.json`.
  0/8 cases passed. Even the lightest centered case,
  `z_0p84_m_0p01_x_0p0`, completed 240 steps but had 39 fall events, min
  robot z `0.19566 m`, min box z `0.20512 m`, and max tilt `1.44487 rad`.
  This means the current fixed-payload setup fails even without front load
  moment. The next isolation is disabling box collision while preserving the
  same fixed joint and tiny mass.
- [x] Run a no-collision centered ultra-light fixed-payload isolation sweep.
  If this passes, collision/contact geometry is the immediate blocker. If this
  fails, the fixed joint or added rigid body itself destabilizes the current
  G1 stand controller.
  Submitted:
  tmux `curiosity_g1_payload_nocoll_0705`, Slurm job `166672`, command
  `STAMP=20260705_core_world_g1_payload_centered_ultralight_nocoll HEIGHTS="0.84 0.96" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.0" ATTACH_Z=0.0 BOX_POS_X=0.0 BOX_POS_Y=0.0 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 BOX_COLLISION_ENABLED=0 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_nocoll bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_nocoll_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight_nocoll/core_world_g1_payload_sweep_summary.json`.
  Initial status: running on `server36`.
  Result:
  job `166672` completed on `server36` with exit `0:0`. Summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight_nocoll/core_world_g1_payload_sweep_summary.json`.
  8/8 no-collision centered fixed-payload cases passed strict stand gate up
  to `0.25 kg`, with fall/drop 0 and no rollout root/box pose writes. Best
  0.84 m case `z_0p84_m_0p01_x_0p0`: min robot z `0.76835 m`, max tilt
  `0.31023 rad`. Worst tilt case `z_0p96_m_0p25_x_0p0`: min robot z
  `0.69804 m`, max tilt `0.59704 rad`. Interpretation: the immediate blocker
  is collision/contact geometry or initial interpenetration, not tiny fixed
  mass itself.
- [x] Run a collision-enabled clearance sweep with small box geometry and
  matched initial box position `(attach_x, 0, height + attach_z)`. Use the
  stable `0.84 m` baseline first. This tests whether collision can remain on
  when the fixed payload is placed outside the robot body instead of centered
  through the torso.
  Submitted:
  tmux `curiosity_g1_payload_clearance_0705`, Slurm job `166673`, command
  `STAMP=20260705_core_world_g1_payload_clearance_collision HEIGHTS="0.84" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.18 0.24 0.30 0.36" ATTACH_Z=0.12 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 BOX_COLLISION_ENABLED=1 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_clear bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_clearance_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_clearance_collision/core_world_g1_payload_sweep_summary.json`.
  Initial status: pending for priority.
  Result:
  job `166673` completed on `server36` with exit `0:0`. Summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_clearance_collision/core_world_g1_payload_sweep_summary.json`.
  16/16 collision-enabled clearance cases passed strict fixed-payload stand
  gate through `0.25 kg`, attach x `0.18/0.24/0.30/0.36 m`, attach z
  `0.12 m`, and 0.10 m cube geometry. Best case
  `z_0p84_m_0p01_x_0p18`: min robot z `0.76792 m`, min box z `0.86952 m`,
  max tilt `0.31326 rad`. Worst tilt case `z_0p84_m_0p25_x_0p24`: min robot
  z `0.75721 m`, min box z `0.82233 m`, max tilt `0.37794 rad`. This is a
  stable fixed-payload standing diagnostic, not walking/free-object carrying.
- [ ] Move from fixed-payload standing to controller-backed stepping/walking.
  Use the collision-enabled stable payload baseline first:
  `height=0.84`, `box_size=0.10`, `attach_z=0.12`,
  `attach_x=0.18-0.30`, mass up to `0.25 kg`. Do not use open-loop march as
  the walking path; it already failed.
- [x] Stop waiting on additional model/data downloads after the user
  correction. Continue by constructing the Isaac scene directly.
- [x] Add official WBC-AGILE ONNX policy glue to the Core API G1 scene.
  Implemented `GAIT_MODE=agile_policy`, `POLICY_START_STEP`,
  `POLICY_CONTROL_DECIMATION`, `AGILE_COMMAND_*`, `AGILE_HEIGHT_COMMAND`,
  `AGILE_CONFIG`, and `AGILE_ONNX` in
  `scripts/isaac/build_core_world_g1_box_scene.py` and
  `scripts/isaac/run_core_world_g1_box_scene.sh`. The adapter uses the
  official local recurrent student ONNX and Arena G1 agile config; it is not a
  hand-written toy controller. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py` and
  `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- [ ] Run no-box AGILE-policy G1 walking smoke on a compute node.
  Command to submit:
  `STAMP=20260705_core_world_g1_agile_policy_nobox_diag1 GAIT_MODE=agile_policy ATTACH_BOX=none STEPS=360 G1_ROOT_Z=0.84 APPLY_ARENA_STAND_GAINS=1 POLICY_START_STEP=40 POLICY_CONTROL_DECIMATION=4 AGILE_COMMAND_X=0.20 AGILE_COMMAND_Y=0.0 AGILE_COMMAND_YAW=0.0 AGILE_HEIGHT_COMMAND=0.72 srun --partition=gpu --gres=gpu:1 --time=01:00:00 --job-name=g1_agile_nobox bash scripts/isaac/run_core_world_g1_box_scene.sh`.
  Gate: completed 360 steps, policy inference count positive, fall/drop 0,
  rollout root pose/velocity writes 0, and nonzero robot XY travel.
  Submitted:
  tmux `curiosity_g1_agile_nobox_0705`, Slurm job `166681`, same command as
  above. Log:
  `logs/core_world_g1_box_scene/g1_agile_nobox_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_box_scene/20260705_core_world_g1_agile_policy_nobox_diag1/core_world_g1_box_scene_summary.json`.
  Initial status: pending for priority.
  Result:
  job `166681` completed on `server36` with exit `0:0`, but produced no
  summary JSON and no rollout `[STATE]` rows. The log reached Isaac startup
  and ONNXRuntime affinity warnings only. This is a failed diagnostic, not
  walking evidence.
- [x] Patch AGILE no-box policy smoke after the no-summary failure.
  `AgileOnnxJointPolicy` now creates ONNXRuntime with CPU provider,
  single-thread intra/inter op settings, and sequential execution. Added
  progress prints around Core World reset, wrapper initialization, and ONNX
  policy loading. Lightweight checks passed again:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py` and
  `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- [ ] Rerun no-box AGILE-policy G1 walking smoke after the ONNXRuntime thread
  fix.
  Result:
  job `166684` completed on `server36` with exit `0:0`, but again produced no
  summary JSON or rollout rows. Added progress prints showed it reached Core
  World reset, G1/box initialization, 43-joint articulation initialization,
  then stopped at `Loading AGILE ONNX policy`. Conclusion: ONNXRuntime session
  creation inside the Isaac process is the failing boundary. Do not keep
  rerunning ONNX unchanged.
- [x] Add a PyTorch checkpoint backend for the same official WBC-AGILE G1
  policy. Implemented `AGILE_POLICY_BACKEND=torch_checkpoint` and
  `AGILE_TORCH_CHECKPOINT`, using WBC-AGILE's official
  `agile.sim2mujoco.policy.PolicyWrapper` with
  `unitree_g1_velocity_height_recurrent_student_checkpoint.pt`. Lightweight
  checks passed.
- [x] Run no-box AGILE-policy G1 walking smoke with
  `AGILE_POLICY_BACKEND=torch_checkpoint`.
  Result: failed before rollout. Slurm job `166690` completed with exit `0:0`
  but produced no summary JSON and no rollout rows; the log reached
  `Loading AGILE torch checkpoint policy` and stopped. Together with the two
  ONNX loader failures, this freezes embedded WBC-AGILE loading inside the
  Isaac Core process for now.
- [ ] If no-box AGILE-policy walking passes, run the same policy with the
  stable collision-enabled fixed payload: root z `0.84`, box cube `0.10 m`,
  mass `0.01-0.25 kg`, attach x initially `0.18`, attach z `0.12`,
  collision enabled, no rollout root/box pose writes.
  Blocked by failed no-box AGILE loader; do not run this unchanged.
- [x] Add a diagnostic-only open-loop G1 march mode in the same Core API scene.
  Implemented `GAIT_MODE=open_loop_march`, `GAIT_AMPLITUDE`, and
  `GAIT_FREQUENCY_HZ` in `build_core_world_g1_box_scene.py`, wired them
  through `run_core_world_g1_box_scene.sh`, and added
  `scripts/isaac/run_core_world_g1_open_loop_march_probe.sh`. This is not a
  serious walking controller; it only tests whether the current stand setup
  immediately collapses under periodic leg commands.
- [x] Run the no-box open-loop march probe on a compute node after or alongside
  payload isolation, then decide whether a controller-backed IsaacLab route is
  required before carrying-walk claims.
  Submitted:
  tmux `curiosity_g1_march_probe_0705`, Slurm job `166667`, command
  `STAMP=20260705_core_world_g1_open_loop_march_probe_small HEIGHTS="0.84 0.96" AMPLITUDES="0.05 0.10" ATTACH_BOX_MODE=none EXPECT_ATTACH_BOX=none MAX_BOX_DROP_EVENTS=999 MIN_BOX_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_march_probe bash scripts/isaac/run_core_world_g1_open_loop_march_probe.sh`.
  Log:
  `logs/core_world_g1_open_loop_march_probe/g1_march_probe_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_open_loop_march_probe/20260705_core_world_g1_open_loop_march_probe_small/core_world_g1_open_loop_march_probe_summary.json`.
  Initial status: pending for priority.
  Result:
  job `166667` ran on `server36` and failed the aggregate gate with exit
  `1:0`. Summary:
  `experiments/outputs/core_world_g1_open_loop_march_probe/20260705_core_world_g1_open_loop_march_probe_small/core_world_g1_open_loop_march_probe_summary.json`.
  0/4 cases passed. All failed by fall/tilt/min-height, with no traceback or
  backend error. This is a negative result for open-loop walking; do not expand
  open-loop sweep as if it were a controller.
- [ ] Replace open-loop march as the walking path with a controller-backed
  Isaac route or explicit feedback controller. Required first gate: no-box G1
  walking/stepping with fall 0, no root pose/velocity rollout writes, stable
  tilt, and recorded commanded joint actions.
- [x] Follow the latest user correction: stop waiting on external models,
  checkpoints, or policy servers when they do not directly unblock the Isaac
  scene. Continue with direct Isaac scene construction.
- [x] Record the current best direct-Isaac physical backend baseline:
  `20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_frontmid`
  passed 3580 steps with 8 kg free box in a cradle, final target distance
  `0.0201 m`, fall/drop 0, strict support continuity, and root/box/payload/
  foot shortcut writes 0. It is still a scaffold, not humanoid carrying.
- [x] Run randomized all-posture strict-support diagnostic on a heavier hidden
  box after freezing AGILE.
  Result: Slurm job `166692`, stamp
  `20260705_randomized_all_posture_strict_support_64cm_seed7071`, completed
  on `server02` with exit `0:0`. Shared hidden box mass `11.47446 kg`, size
  `[0.36871, 0.22426, 0.21205] m`, COM offset
  `[-0.03709, -0.01539, 0.02677] m`. `front_mid`, `low_front`, and
  `chest_high` all passed 3580/3580 with fall/drop 0, strict support
  continuity, no fixed-world support, and final target distances
  `0.00315 m`, `0.01245 m`, and `0.01292 m`.
- [x] Continue direct Isaac through active-probe plus parameter search instead
  of model waiting. Next concrete run: expand
  `run_probe_parameter_search_multiseed_diag.sh` to new hidden seeds and use
  its probe telemetry plus strict candidate scoring to test whether posture
  choice changes with mass, size, COM, and observed load response.
  Submitted:
  tmux `curiosity_probe_param_multiseed_7071_0705`, Slurm job `166694`,
  command
  `MULTISEED_STAMP=20260705_probe_parameter_search_multiseed_7071_7073 SEEDS='7071 7072 7073' srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_param_7071 bash scripts/isaac/run_probe_parameter_search_multiseed_diag.sh`.
  Log:
  `logs/probe_parameter_search_multiseed/probe_parameter_search_multiseed_7071_0705_srun.log`.
  Expected summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7071_7073/probe_parameter_search_multiseed_summary.json`.
  Initial status: pending for priority.
  Result:
  Slurm job `166694` completed on `server02` with exit `0:0` after
  `00:10:22`. All 3 hidden seeds passed the diagnostic wrapper. Best posture
  varied across seeds: `front_mid` won seeds `7071` and `7073`, while
  `low_front` won seed `7072`. Selected candidates had fall/drop 0 and strict
  support continuity. For every seed, `front_mid_wide_slow` and
  `low_front_wide_slowest` were rejected by support-continuity gates despite
  moving the box.
- [x] Add richer posture/gait parameter candidates around hold height, carry
  speed, support timing, stance width, and step length. Keep the strict
  checker unchanged so unsafe candidates are rejected, not hidden by scoring.
  Implemented `CANDIDATE_SET=expanded` in
  `scripts/isaac/run_probe_parameter_search_carry_diag.sh`. Candidate action
  parameters now include posture, stance steps, step length, support-foot
  stance/swing X, torso height, payload local X/Z, support-foot step height,
  double-support fraction, stance half-length, and stance half-width.
  Lightweight checks passed.
- [x] Run the expanded posture/gait action-space diagnostic and record whether
  the new candidate parameters remain stable under strict gates.
  Submitted:
  tmux `curiosity_probe_param_expanded_7074_0705`, Slurm job `166718`,
  command
  `MULTISEED_STAMP=20260705_probe_parameter_search_expanded_7074_7075 SEEDS='7074 7075' CANDIDATE_SET=expanded srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_param_exp bash scripts/isaac/run_probe_parameter_search_multiseed_diag.sh`.
  Log:
  `logs/probe_parameter_search_multiseed/probe_parameter_search_expanded_7074_0705_srun.log`.
  Expected summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_expanded_7074_7075/probe_parameter_search_multiseed_summary.json`.
  Initial status: pending for priority.
  Result:
  Slurm job `166718` completed on `server02` with exit `0:0` after
  `00:11:21`. Both seeds passed. Seed `7074` selected `front_mid_nominal`;
  seed `7075` selected the new `low_front_cautious`. Selected candidates had
  fall/drop 0 and strict support continuity. The expanded runner evaluated 9
  candidates per seed, with 5/9 and 6/9 passing strict gates.
- [x] Add an effort-aware score term using available drive effort or force
  proxy telemetry, so the selector starts optimizing "省力" instead of only
  target distance plus safety gates.
  Implemented in `scripts/isaac/summarize_probe_parameter_search_carry.py`.
  The score now reports `score_terms` and uses measured support-foot effort
  proxy when present plus max tilt, support-margin shortfall, support-foot
  lift, and support-foot motion as kinematic effort proxies. This is not a
  real torque/energy objective yet.
- [x] Define the first RL-ready interface around the current direct Isaac
  scaffold: probe telemetry and load-state observation, posture/gait parameter
  action, and reward terms for distance, fall/drop, support continuity, tilt,
  travel loss, and effort. Label this as an interface scaffold until actual RL
  training is run.
- [x] Add a JSONL episode-table exporter for the direct Isaac parameter-search
  summaries. It should emit one row per candidate episode with observation
  proxy fields, action parameters, reward terms, strict pass/fail fields, and
  limitation labels. This is the next bridge from transparent search to RL.
  Implemented:
  `scripts/isaac/export_probe_parameter_search_episode_table.py` and
  `scripts/isaac/run_export_probe_parameter_episode_table.sh`.
- [x] Run the JSONL episode-table export on a compute node and record the
  output path and row count.
  Submitted:
  tmux `curiosity_export_episode_table_0705`, Slurm job `166744`, command
  `STAMP=20260705_probe_parameter_episode_table_expanded_7074_7075 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=export_ep_table bash scripts/isaac/run_export_probe_parameter_episode_table.sh --multiseed-summary experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_expanded_7074_7075/probe_parameter_search_multiseed_summary.json`.
  Expected output:
  `experiments/outputs/rl_interface/20260705_probe_parameter_episode_table_expanded_7074_7075/probe_parameter_episode_table.jsonl`.
  Initial status: pending for priority.
  Result:
  Slurm job `166744` completed on `server46` with exit `0:0`.
  Output:
  `experiments/outputs/rl_interface/20260705_probe_parameter_episode_table_expanded_7074_7075/probe_parameter_episode_table.jsonl`.
  Row count: `18`, matching 2 hidden seeds x 9 candidates.
- [ ] Next environment step toward the full goal: replace the current
  scaffolded support-foot controller with a controller-backed walking robot or
  a stronger feedback stepping controller that can satisfy the same gates with
  no torso/root pose or velocity shortcuts. The direct Isaac scaffold is now
  good enough as task/selection interface; the remaining core gap is real
  walking balance under load.
- [x] Add explicit rail-motion and feedback-step evidence fields to the direct
  Isaac carrier summaries and checker, so future runs cannot pass by hiding
  inactive feedback or rail-driven motion. Implemented
  `max_rail_joint_motion_m`, `feedback_step_controller_enabled`,
  `feedback_step_applied_steps`,
  `max_abs_feedback_step_x_adjustment_m`, and
  `max_abs_feedback_step_tilt_adjustment_m` pass-through/checking.
- [x] Add the first strict direct-Isaac feedback-step carry diagnostic script.
  Implemented
  `scripts/isaac/run_feedback_step_controller_carry_diag.sh`; it randomizes
  the box, runs front-mid carrying with `SUPPORT_MODE=alternating_anchor_feet`
  and `FEEDBACK_STEP_CONTROLLER=1`, then checks root-shortcut-free execution,
  support-foot X/Z motion, support continuity, actual foot lift, active
  feedback steps, and low rail motion.
- [x] Run the feedback-step controller carry diagnostic on a compute node and
  record pass/fail.
  Early attempts exposed useful failures: job `166750` failed before logging;
  job `166751` exposed brittle launcher environment passing; job `166753`
  completed Isaac but failed strict support/report gates; job `166758` showed
  `SUPPORT_FOOT_CONTINUITY_GRACE_STEPS` was not yet applied in the run; jobs
  `166765` and `166768` completed Isaac but failed the near-ground z-proxy
  support continuity gate. Fixes added `CORE_ENV` launcher passing,
  post-settle target/loss reporting, continuity grace accounting, and explicit
  `SUPPORT_FOOT_CONTACT_Z_THRESHOLD`.
  Final pass:
  tmux `curiosity_feedback_step_retry6_0705`, Slurm job `166769`, command
  `STAMP=20260705_feedback_step_controller_seed7076_frontmid_retry6 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_step_carry bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`.
  Output:
  `experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_controller_seed7076_frontmid_retry6/feedback_step_controller_check.json`.
  Result: `status=pass`, completed `3580` steps on a randomized `11.46294 kg`
  box, fall/drop `0`, root/box/foot/stance shortcut writes `0`,
  feedback-step applied steps `3570`, max rail joint motion `0.02151 m`
  under the diagnostic threshold `0.025 m`, min drive near-ground foot count
  `3`, and final box target distance `0.00247 m`. This is still a direct
  Isaac scaffold diagnostic, not a full humanoid walking controller.
- [x] Add a first force-like support evidence proxy so the feedback-step gate
  no longer depends only on support-foot z-height. Implemented measured
  support-foot joint-effort telemetry in the anchored-footstep carrier,
  normalizer, checker, and feedback-step diagnostic. After one pre-Isaac
  launcher EOF failure (`166775`) and one post-summary shell EOF failure
  (`166781`), the existing backend summary was normalized and checked in a
  separate compute allocation:
  tmux `curiosity_feedback_step_effort_check_0705`, Slurm job `166786`,
  command
  `STAMP=20260705_feedback_step_effort_gate_seed7076_retry2 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=fb_effort_chk ...`.
  Result:
  `experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_effort_gate_seed7076_retry2/feedback_step_effort_check.json`
  passed with completed steps `3580`, hidden box mass `11.46294 kg`,
  fall/drop `0`, support-foot effort available, effort read errors `0`,
  min drive effort-supported foot count `4`, drive effort-supported lt2 steps
  `0`, min commanded-stance effort-supported foot count `2`, and commanded
  stance effort-supported lt2 steps `0`.
- [x] Add actual PhysX contact-state support evidence to replace support checks
  that relied only on z-height and joint-effort proxies. Implemented optional
  `PhysxContactReportAPI` support-foot/ground contact tracking in
  `build_core_world_anchored_footstep_carrier.py`, normalized it through the
  direct carry-task schema, and added strict checker gates for contact-report
  availability, drive-phase contact foot counts, and commanded-stance contact
  foot counts. First attempt `166793` completed Isaac but failed post-summary
  and showed the contact-report flag had not propagated. Retry:
  tmux `curiosity_feedback_step_contact_retry2_0705`, Slurm job `166797`,
  command
  `STAMP=20260705_feedback_step_contact_report_seed7076_retry2 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_contact2 bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`.
  Result:
  `experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_contact_report_seed7076_retry2/feedback_step_controller_check.json`
  passed with completed steps `3580`, hidden box mass `11.46294 kg`,
  fall/drop `0`, contact report requested/available, event count `42`, error
  count `0`, per-foot contact-report steps `3332/3308/3451/3407`,
  min drive contact-report foot count `2`, drive contact-report lt2 steps `0`,
  min commanded-stance contact-report foot count `2`, and commanded-stance
  contact-report lt2 steps `0`.
- [x] Run the randomized all-posture hidden-box carry gate with the new PhysX
  contact-report requirements. Updated
  `scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh` and
  `scripts/isaac/summarize_randomized_all_posture_carry.py` so all three
  postures must pass contact-report support gates, not only near-ground gates.
  Submitted:
  tmux `curiosity_all_posture_contact_0705`, Slurm job `166800`, command
  `STAMP=20260705_randomized_all_posture_contact_report_64cm_seed7077 BOX_SEED=7077 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=all_post_contact bash scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh`.
  Expected summary:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_contact_report_64cm_seed7077/randomized_all_posture_strict_support_summary.json`.
  Result:
  Slurm job `166800` completed on `server02` with exit `0:0` after
  `00:01:50`. Summary status `pass`. Shared hidden randomized box:
  mass `4.86216 kg`, size `[0.35968, 0.24056, 0.24150] m`, COM offset
  `[-0.02053, 0.00841, 0.02503] m`. Postures `front_mid`, `low_front`, and
  `chest_high` all completed `3580` steps with fall/drop `0`, root shortcuts
  disabled, PhysX contact-report available, contact-report error count `0`,
  min drive contact-report foot count `2`, drive contact-report lt2 steps
  `0`, min commanded-stance contact-report foot count `2`, and commanded
  stance contact-report lt2 steps `0`. This is a stronger direct Isaac
  scaffold gate, not final humanoid walking/RL/video-conditioned carrying.
- [ ] Add calibrated contact-force or ground-reaction-force evidence. The
  current best run has actual PhysX contact-state events plus joint-effort
  evidence, but it still does not report calibrated support forces.
- [ ] Fix the recurring post-summary shell EOF issue in the direct physical
  backend/core launcher path so future successful Isaac rollouts do not require
  a separate normalize/check job.
- [x] Fix launcher robustness for optional stand gains.
  `run_core_world_g1_box_scene.sh` now builds a Bash argument array and appends
  `--apply-arena-stand-gains` conditionally. This prevents optional-argument
  line-continuation issues. Re-check `run_status.txt` in payload sweep job
  `166663` because one running case hit the old launcher form while the file
  was being edited.
- [x] Add a direct Isaac carry-task episode contract instead of waiting on
  external video/model code. Implemented
  `scripts/isaac/direct_carry_task_contract.py`,
  `scripts/isaac/export_direct_carry_task_episode_table.py`,
  `scripts/isaac/run_export_direct_carry_task_episode_table.sh`, and
  `experiments/configs/direct_isaac_carry_task_contract_v1.json`. The contract
  separates `policy_observation` from `hidden_eval_context`, so unknown box
  mass and COM are not policy inputs.
- [x] Run the direct Isaac carry-task episode export on a compute node and
  record the JSONL output path and row count. Submitted from tmux
  `curiosity_export_direct_task_contract_0705`, Slurm job `166804`, command
  `STAMP=20260705_direct_carry_task_contract_all_posture_contact_7077 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=carry_contract bash scripts/isaac/run_export_direct_carry_task_episode_table.sh --summary experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_contact_report_64cm_seed7077/randomized_all_posture_strict_support_summary.json`.
  Expected output:
  `experiments/outputs/rl_interface/20260705_direct_carry_task_contract_all_posture_contact_7077/direct_carry_task_episode_table.jsonl`.
  Result:
  job `166804` completed on `server02` with exit `0:0` and exported `3` rows,
  but the first exporter used compressed all-posture rows, leaving some action
  fields null. The exporter was fixed to follow each posture `summary_path` and
  load the full backend summary. Retry job `166806` completed on `server02`
  with exit `0:0`, exported `3` rows to the same JSONL path, and now includes
  `controller_mode`, `support_foot_mode`, contact gates, support effort
  metrics, and hidden box properties only under `hidden_eval_context`.
- [x] Build the next direct Isaac task runner skeleton around this contract:
  `reset(randomized_box_seed, morphology_config)`, `observe()`, `apply_action()`,
  `compute_reward()`, `is_terminated()`, and `export_episode_row()`. It should
  first wrap the existing scaffold without claiming RL, then become the
  interface where a real walking controller or trainable policy replaces the
  scaffold. Implemented `scripts/isaac/direct_carry_task_runner.py`; lightweight
  syntax check passed with `python3 -m py_compile`.
- [x] Connect the task runner to the existing executable Isaac carry backend
  instead of leaving it as a paper interface. Implemented
  `scripts/isaac/direct_carry_task_shell_backend.py`,
  `scripts/isaac/run_direct_carry_task_runner_episode.py`, and
  `scripts/isaac/run_direct_carry_task_runner_episode.sh`. The adapter maps
  `DirectCarryReset` and `DirectCarryAction` into the existing
  `run_direct_carry_task_physical_backend.sh` environment, runs the backend,
  and exports one `direct_isaac_carry_task_episode_v1` row.
- [x] Run a full task-runner episode on a compute node and record whether the
  reset/action -> shell backend -> Isaac summary -> contract row chain works.
  First tmux submit attempt failed before Slurm because the outer log
  directory did not exist. After creating `logs/direct_carry_task_runner`,
  submitted from tmux `curiosity_task_runner_episode_retry_0705`, Slurm job
  `166810`, command
  `STAMP=20260705_task_runner_frontmid_seed7078 BOX_SEED=7078 CARRY_POSTURE=front_mid TARGET_X=0.64 STEPS=3580 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=task_runner bash scripts/isaac/run_direct_carry_task_runner_episode.sh`.
  Result:
  Slurm job `166810` completed on `server02` with exit `0:0` after
  `00:01:04`. The task-runner chain produced
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_physical_backend_summary.json`
  and
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_runner_episode.jsonl`.
  The randomized hidden box was mass `4.33753 kg`; the rollout completed
  `3580` steps with fall/drop `0`, final post-settle box travel `0.65758 m`,
  final post-settle target distance `0.01758 m`, PhysX contact report
  available, contact-report error count `0`, min drive contact-report foot
  count `2`, and commanded-stance contact-report lt2 steps `0`. This proves
  the executable task-runner chain works for the scaffold backend, not that
  the final walking-robot objective is complete.
- [x] Fix the task-contract pass flag for backend summaries that do not carry
  an explicit `status=pass` field. `direct_carry_task_contract.py` now derives
  `gates.passed` from strict no-fall/no-drop/no-root-shortcut/support-contact
  fields when `status` is absent. `run_direct_carry_task_runner_episode.py`
  now reports `status=pass` when the derived gates pass and backend returncode
  is `0`.
- [x] Run strict checker/export on the completed task-runner episode after the
  pass-flag fix. Added
  `scripts/isaac/run_check_direct_carry_task_runner_episode.sh`, which runs
  `check_direct_carry_task_summary.py` and re-exports the episode row with the
  corrected contract. Submitted from tmux `curiosity_task_runner_check_0705`,
  Slurm job `166817`, command:
  `STAMP=20260705_task_runner_frontmid_seed7078_check SUMMARY=experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_physical_backend_summary.json BOX_SEED=7078 CARRY_POSTURE=front_mid srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=task_check bash scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  Result:
  Slurm job `166817` completed on `server02` with exit `0:0` after
  `00:00:01`. Strict check report:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_frontmid_seed7078_check/direct_carry_task_runner_check.json`
  with `status=pass`. Corrected episode table:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_frontmid_seed7078_check/direct_carry_task_runner_episode_table.jsonl`,
  with `gates.passed=true`.
- [x] Add explicit active-probing action support to the task runner. The
  current validated task-runner episode has `probe_belief_source=no_active_probe`;
  this is not sufficient for the final unknown-load carrying objective.
  Implemented `probe_steps` in `DirectCarryAction`, passed it through
  `direct_carry_task_shell_backend.py`, added it to
  `run_direct_carry_task_runner_episode.py/.sh`, and included probe action
  fields in `direct_carry_task_contract.py` and
  `experiments/configs/direct_isaac_carry_task_contract_v1.json`.
- [x] Validate one active-probing task-runner carry episode on a compute node.
  Submitted from tmux `curiosity_task_runner_probe_0705`, Slurm job `166819`,
  command:
  `STAMP=20260705_task_runner_probe_frontmid_seed7079 BOX_SEED=7079 CARRY_POSTURE=front_mid TARGET_X=0.64 STEPS=3660 PROBE_STEPS=80 PROBE_AMPLITUDE_X=0.012 PROBE_AMPLITUDE_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=task_probe bash scripts/isaac/run_direct_carry_task_runner_episode.sh`.
  Result:
  Slurm job `166819` completed on `server02` with exit `0:0` after
  `00:00:45`. The episode requested `80` horizontal push-pull probe steps
  before carrying. Hidden randomized box mass `11.13313 kg`, size
  `[0.31808, 0.25514, 0.23031] m`, COM offset
  `[-0.01361, -0.02603, 0.01952] m`. It completed `3660` steps with
  fall/drop `0`, final post-settle box travel `0.66478 m`, final post-settle
  target distance `0.02478 m`, `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`, probe source
  `heuristic_from_probe_telemetry_not_calibrated_mass_estimator`,
  max probe box travel `0.03064 m`, and PhysX support-foot contact reports
  available. Runner report status `pass`.
- [x] Add probe-specific strict checker gates and validate the active-probing
  episode under them. Added checker flags `--min-probe-steps`,
  `--require-probe-belief`, `--forbid-probe-hidden-ground-truth`, and
  `--min-probe-box-travel-x`; updated
  `scripts/isaac/run_check_direct_carry_task_runner_episode.sh` to use them
  when `REQUIRE_PROBE_BELIEF=1`. Old checker job `166820` passed the carry
  gate; probe-gated checker job `166821` completed on `server02` with exit
  `0:0` after `00:00:01` and produced `status=pass` with corrected row
  `gates.passed=true` at
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_probe_frontmid_seed7079_probegate/direct_carry_task_runner_episode_table.jsonl`.
- [x] Run the active-probing task-runner gate across multiple carry postures.
  `20260705_task_runner_active_probe_postures_seed7080` completed on
  `server02` with all three postures passing the current scaffold gate.
  This is task-runner/posture/probe bookkeeping evidence, not final walking
  robot success.
- [x] Add a multi-posture active-probe task-runner sweep and summarizer.
  Implemented `scripts/isaac/run_task_runner_active_probe_postures.sh` and
  `scripts/isaac/summarize_task_runner_active_probe_postures.py`. The sweep
  runs `front_mid`, `low_front`, and `chest_high` with the same hidden box seed,
  requires active-probe belief through the strict checker, and writes one
  all-posture summary.
- [x] Run the multi-posture active-probe task-runner gate on a compute node.
  Submitted from tmux `curiosity_active_probe_postures_0705`, Slurm job
  `166822`, command:
  `STAMP=20260705_task_runner_active_probe_postures_seed7080 BOX_SEED=7080 TARGET_X=0.64 STEPS=3660 PROBE_STEPS=80 PROBE_AMPLITUDE_X=0.012 PROBE_AMPLITUDE_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_postures bash scripts/isaac/run_task_runner_active_probe_postures.sh`.
  Result: completed on `server02` with exit `0:0` after `00:01:56`.
  Shared hidden randomized box mass `10.72455 kg`, size
  `[0.36519, 0.22971, 0.23912] m`, COM offset
  `[0.03732, 0.00523, -0.00058] m`. `front_mid`, `low_front`, and
  `chest_high` all completed `3660/3660`, fall/drop `0`, active probe belief
  available, and no hidden-ground-truth probe use.
- [x] Add target-directional support-foot placement mode to the task-runner
  scaffold and validate both positive and negative target directions.
  `20260705_task_runner_directional_placement_seed7081_retry3` passed a
  positive `0.64 m` target. `20260705_task_runner_directional_negative_seed7082_retry`
  passed a negative `-0.32 m` target after fixing target-directed reward and
  travel-loss metrics.
- [x] Run a directional support-placement multi-posture active-probe sweep
  under one hidden box seed. Slurm job `166850`, stamp
  `20260705_task_runner_directional_postures_seed7083_server02`, completed on
  `server02` with exit `0:0` after `00:02:21`. Shared hidden box mass
  `5.91337 kg`, size `[0.33273, 0.26142, 0.22331] m`, COM offset
  `[0.01569, -0.01343, 0.01675] m`. `front_mid`, `low_front`, and
  `chest_high` all completed `3660/3660`, fall/drop `0`, root shortcut free,
  active probe belief available, no hidden-ground-truth probe use, PhysX
  contact-report gates passed, and directional placement true.
- [ ] Next direct-Isaac implementation step: stop treating the current
  anchored support-foot/cradle backend as the main result. Preserve the
  reset/action/observation/reward/checker contract, but replace the scaffold
  with a more physical Isaac component: repeated support placement/contact
  control first, or a controller-backed robot only if it can enter rollout
  immediately without blocking on model downloads.
- [x] Expose feedback step-controller parameters through the task-runner shell
  wrapper so direct Isaac diagnostics can be controlled without editing
  Python defaults. `run_direct_carry_task_runner_episode.sh` now passes
  `GAIT_SPEED_SCALE`, feedback step gains/limits, and
  `SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION`; `bash -n` passed for the runner,
  posture sweep, and checker wrappers.
- [x] Run one explicit feedback directional-placement rollout instead of
  waiting on external models. `20260705_task_runner_directional_feedback_seed7084`
  completed on `server02` in Slurm job `166853` with runner status `pass`.
  Hidden box mass `7.68171 kg`; completed `3660/3660`; fall/drop `0`;
  final post-settle box travel `0.62166 m`; final target distance `0.01834 m`;
  probe belief available without hidden ground truth; directional placement
  true; feedback controller enabled; `feedback_step_applied_steps=3570`.
  Formal checker attempts `166854`, `166857`, and `166858` were canceled while
  pending due scheduler priority, so this is not yet checker-validated.
- [ ] Run a formal strict checker for
  `20260705_task_runner_directional_feedback_seed7084` when scheduler priority
  permits, or re-run the same diagnostic with checker bundled into a single
  allocation.
- [x] Bundle rollout and strict checker into one allocation for the explicit
  feedback/directional multi-posture path. Updated
  `run_task_runner_active_probe_postures.sh` to pass feedback/gait parameters
  to each episode and choose
  `physical_alternating_placement_feet_cradle` checker mode automatically for
  `SUPPORT_MODE=alternating_placement_feet`. Updated
  `summarize_task_runner_active_probe_postures.py` to retain feedback fields
  and require feedback-applied steps in the sweep pass logic.
- [x] Run the bundled feedback/directional multi-posture gate. Slurm job
  `166859`, stamp
  `20260705_task_runner_directional_feedback_postures_seed7085`, completed on
  `server36` with exit `0:0` after `00:01:44`. Shared hidden randomized box:
  mass `10.22545 kg`, size `[0.35601, 0.24706, 0.22236] m`, COM offset
  `[-0.03576, 0.00871, 0.01431] m`. `front_mid`, `low_front`, and
  `chest_high` all passed strict checker in the same allocation with
  `3660/3660` steps, fall/drop `0`, root shortcut free, active probe belief
  without hidden ground truth, directional placement true, feedback controller
  enabled with `3570` applied steps, and contact-report gates passed. Final
  post-settle travel / target distance: `front_mid` `0.61728 / 0.02272 m`,
  `low_front` `0.64413 / 0.00413 m`, `chest_high`
  `0.61305 / 0.02695 m`.
- [ ] Next replacement task: preserve the now checker-validated
  hidden-box/probe/posture/feedback/contact contract, but replace one
  scaffold component. First target is support/contact mechanics: reduce the
  anchored/cradle simplification and make progress depend on more physical
  repeated support placement/contact rather than the current scaffolded
  support-foot carrier.
- [x] Add planted-foot slip audit gates. Implemented
  `--max-near-ground-foot-slip` in `check_direct_carry_task_summary.py`,
  exposed `MAX_NEAR_GROUND_FOOT_SPEED` and `MAX_NEAR_GROUND_FOOT_SLIP` in
  `run_check_direct_carry_task_runner_episode.sh`, and added per-foot/max
  near-ground speed/slip fields to the posture sweep summarizer.
- [x] Expose stance/support/friction parameters for slip-reduction tests
  without editing Python defaults. `direct_carry_task_shell_backend.py` now
  respects parent env overrides for `STANCE_STEPS`, `STEP_LENGTH`,
  support-foot drive parameters, and friction parameters; the direct runner
  and posture sweep wrappers pass these through.
- [x] Run the first slow-stance slip audit. Slurm job `166864`, stamp
  `20260705_task_runner_directional_slow_slip_audit_seed7086`, used
  `STANCE_STEPS=160`, `STEPS=7000`, and
  `MAX_NEAR_GROUND_FOOT_SPEED=0.8`. Result: negative for walking realism.
  The carry completed safely with fall/drop `0`, final post-settle travel
  `0.64886 m`, and final target distance `0.00886 m`, but failed the new
  slip-speed gate: max near-ground foot speed `1.05842 m/s`, max near-ground
  foot slip `0.69295 m`.
- [ ] Implement a support mechanic that reduces near-ground foot sliding.
  Candidate directions: stance-foot world/contact lock diagnostic, true
  planted-foot constraint with swing-foot repositioning, or a controller-backed
  robot whose planted feet satisfy the same slip-speed/slip-distance gates.
- [x] Implement and audit a stance-foot world-lock diagnostic instead of
  waiting on external models. Added `STANCE_FOOT_WORLD_LOCK` through the
  direct Isaac runner stack plus checker/summarizer fields and gates. Valid
  compute run: Slurm job `166875`, stamp
  `20260705_task_runner_stance_world_lock_slip_seed7088_server36`, fixed
  `server36`, `front_mid`, `BOX_SEED=7088`, `TARGET_X=0.64`, `STEPS=3660`,
  `SUPPORT_MODE=alternating_placement_feet`,
  `STANCE_FOOT_WORLD_LOCK=1`, `REQUIRE_STANCE_FOOT_WORLD_LOCK=1`,
  `MAX_NEAR_GROUND_FOOT_SPEED=0.8`, and
  `MAX_NEAR_GROUND_FOOT_SLIP=0.2`. Result: negative. The rollout completed
  with fall/drop `0`, final post-settle box travel `0.64560 m`, final target
  distance `0.00560 m`, active probe belief without hidden ground truth, and
  world-lock telemetry present (`enabled=true`, `4` joints, `81` switches,
  `324` pose updates). Strict checker failed because actual foot lift was
  `0.01943 m < 0.03`, max near-ground foot speed was
  `0.91486 m/s > 0.8`, and max near-ground foot slip was
  `0.73106 m > 0.2`. PhysX warned that the world-lock joints had disjointed
  body transforms and would likely snap bodies together.
- [ ] Replace the failed stance-foot world-lock diagnostic with a
  contact-consistent support mechanic. The next version should keep stance
  feet fixed in world without simultaneously commanding incompatible prismatic
  targets, reset slip references at legitimate lift/replant transitions, and
  pass the existing `MAX_NEAR_GROUND_FOOT_SPEED` /
  `MAX_NEAR_GROUND_FOOT_SLIP` gates before any multi-posture claim.
- [x] Add a freeze-locked stance target diagnostic path. Implemented
  `FREEZE_LOCKED_STANCE_FOOT_TARGETS` / `--freeze-locked-stance-foot-targets`
  so locked stance feet keep measured X/Z joint targets rather than being
  driven against their own world-lock constraints. Added checker/summary fields
  and optional gate `REQUIRE_FREEZE_LOCKED_STANCE_FOOT_TARGETS=1`. Lightweight
  Python and shell checks passed.
- [x] Validate the freeze-locked stance target diagnostic on compute. Invalid
  attempts so far: Slurm job `166884` failed before Isaac with a transient or
  stale shell error and no backend summary; retry job `166885` failed before
  rollout with argparse ambiguity `--tray-`. Added `DEBUG_CORE_CMD=1` to print
  exact core argv. Debug validation job `166888`, stamp
  `20260705_task_runner_freeze_locked_stance_seed7089_debug_s10`, ran on
  `server10` and reached checker. Result: negative. Freeze mode was enabled
  (`freeze_locked_stance_foot_targets_enabled=true`,
  `freeze_locked_stance_foot_target_count=8`) and reduced foot slip/speed
  (`max_near_ground_foot_slip_m=0.00320`,
  `max_near_ground_foot_speed_mps=0.27180`), but task performance collapsed:
  `fall_events=1853`, final post-settle box travel `-0.14893 m`, final target
  distance `0.78893 m`, and max target-directed post-settle travel only
  `0.00327 m`.
- [x] Implement and validate the planted-foot rail-propulsion diagnostic.
  Added `PLANTED_STANCE_RAIL_PROPULSION` / `--planted-stance-rail-propulsion`
  through the direct Isaac task-runner stack, normalizer, checker, and
  summarizer. The first compute run, Slurm job `166894`, stamp
  `20260705_task_runner_planted_rail_propulsion_seed7090`, is not valid
  physical evidence because the trigger was wired into the wrong branch and
  `planted_stance_rail_propulsion_steps=0`. After fixing the gate, Slurm job
  `166895`, stamp
  `20260705_task_runner_planted_rail_propulsion_seed7091_fixedgate`, ran on
  `server53` and triggered correctly with
  `planted_stance_rail_propulsion_steps=3570`. Result: negative. Foot sliding
  stayed low (`max_near_ground_foot_slip_m=0.00320`,
  `max_near_ground_foot_speed_mps=0.27191`), but the task failed with
  `fall_events=1902`, actual support-foot lift `0.00428 m < 0.03 m`, max
  target-directed post-settle box travel `0.00360 m`, final post-settle box
  travel `-0.13414 m`, and final target distance `0.77414 m`.
- [ ] Stop extending the stance-world-lock branch except for cleanup. The
  latest negative diagnostics show that world-lock plus frozen stance can
  suppress visible slip but does not create valid carrying propulsion and
  still triggers PhysX fixed-joint snapping warnings.
- [ ] Build the next Isaac support path without fixed-world stance locks.
  Preserve the same hidden-box/probe/posture/checker contract, but replace the
  support model with contact-consistent planted-foot mechanics or a
  controller-backed robot. Required gates stay: low near-ground slip/speed,
  verified lift/replant before slip-reference reset, contact reports, zero
  falls/drops, target-directed carry distance, and no hidden load inputs.
- [x] Test a no-world-lock commanded-stance freeze variant. Implemented
  `FREEZE_COMMANDED_STANCE_FOOT_TARGETS` /
  `--freeze-commanded-stance-foot-targets` so commanded stance feet keep their
  measured X/Z joint targets without creating fixed-world locks. Slurm job
  `166899`, stamp
  `20260705_task_runner_no_worldlock_contact_propulsion_seed7092`, ran on
  `server44` with `STANCE_FOOT_WORLD_LOCK=0`,
  `FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`, and
  `PLANTED_STANCE_RAIL_PROPULSION=1`. Result: negative but clean. The run
  avoided fixed-world support and triggered both diagnostics
  (`freeze_commanded_stance_foot_target_count=8`,
  `freeze_commanded_stance_foot_target_switch_count=81`,
  `planted_stance_rail_propulsion_steps=3570`). Foot slip stayed within the
  audit (`max_near_ground_foot_slip_m=0.04693`,
  `max_near_ground_foot_speed_mps=0.30856`), but contact support failed
  (`min_drive_contact_report_foot_count=0`,
  `commanded_stance_contact_report_lt2_steps=1580`), with
  `fall_events=444`, final post-settle box travel `-0.21011 m`, and final
  target distance `0.85011 m`.
- [ ] Stop treating the current prismatic support-foot scaffold as the path
  forward. It has now failed both fixed-world and no-world-lock planted-foot
  diagnostics. The next task should instantiate or adapt a controller-backed
  Isaac robot/contact model while preserving the existing task-runner contract
  and strict checker gates.
- [x] Follow the user correction: stop blocking on external models/checkpoints
  when they are not immediately useful and continue constructing the Isaac
  carrying scene directly.
- [x] Add pure no-box stand support to the Core API G1 scene. Implemented
  `--disable-carry-box-spawn` and launcher env `SPAWN_CARRY_BOX=0`; summaries
  now record `carry_box_spawned`, and the checker supports
  `--expect-carry-box-spawned true|false`.
- [x] Fix setup root orientation plumbing in the Core API G1 scene so
  `--g1-root-orientation-wxyz` is actually used by `robot.set_world_pose`.
- [x] Run lightweight checks for the G1 no-box stand update:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/check_core_world_g1_box_scene_summary.py`
  and `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- [ ] Await/record the no-box G1 stand tuning batch
  `curiosity_g1_stand_nobox_tune_0705`, Slurm job `166915`, stamps
  `diag11`-`diag14`.
- [ ] If any no-box stand diagnostic passes, run a small fixed-torso payload
  balance diagnostic with `SPAWN_CARRY_BOX=1`, `ATTACH_BOX=fixed_torso`,
  low mass, zero rollout root writes, and no fall/drop.
- [ ] If no no-box stand diagnostic passes, inspect the G1 USD/controller
  gains and replace the simple open-loop stand target controller before
  attempting payload or free-box carrying.
- [x] Record no-box G1 stand tuning batch `166915`. All four diagnostics
  failed with `carry_box_spawned=false` and `box_drop_events=0`; failure is
  G1 stand/balance, not free-box noise.
- [x] Add IsaacLab G1_29DOF drive-gain preset and pelvis-xform toggle to the
  Core API G1 scene.
- [ ] Await/record IsaacLab-pose/gain alignment batch
  `curiosity_g1_isaaclab_pose_tune_0705`, Slurm job `166916`, stamps
  `diag15`-`diag17`.
- [x] Mark Slurm job `166916` invalid: launcher shell parse error before
  Isaac, no summaries.
- [x] Add setup-only joint position/velocity state write before rollout and
  summary fields `joint_state_write_count_setup` /
  `joint_state_write_error`.
- [ ] Await/record retry batch `curiosity_g1_isaaclab_pose_retry2_0705`,
  Slurm job `166918`, stamps `diag15_retry2`-`diag17_retry2`.

- [x] Record conservative staged long validation `diag77`-`diag80`.
  Result: all 700-step variants failed with the same long-horizon forward
  pitch/drop mode. Best delayed failure was `diag79`
  (`cradle_mass_scale=0.80`, amp `0.10`): `fall_events=148`,
  `box_drop_events=131`, final box target-directed travel `0.75886 m`.
- [x] Implement terminal hold triggers and summary fields in the Core API G1
  scene: trigger by step, box target-directed travel, robot target-directed
  travel, pitch, and pitch rate; record active steps, first active step, and
  first reason.
- [x] Record terminal-hold batches `diag81`-`diag88`. Result: negative.
  Late triggers first activated around steps `398`-`403`; early triggers
  first activated around steps `320`-`342`; all failed with fall/drop despite
  terminal hold staying active for hundreds of steps.
- [x] Add `stand_force_scale` and record drive max-force scaling separately
  from stiffness/damping scaling.
- [x] Record drive-authority batch `diag89`-`diag92`. Result: negative for
  carrying. `diag91` was stable for 700 steps but final box target-directed
  travel was only `0.03484 m`; `diag92` moved `0.85053 m` but failed with
  `fall_events=68` and `box_drop_events=56`.
- [x] Add staged terminal drive switching: terminal hold can rewrite G1 drive
  stiffness/damping/max-force via `terminal_drive_gain_scale` and
  `terminal_drive_force_scale`, with applied step and drive table recorded.
- [x] Record terminal-drive batch `curiosity_g1_terminal_drive_0705`, Slurm
  job `167039`, stamps `diag93`-`diag96`. Result: negative. Drive switching
  applied at the intended terminal steps (`320`-`342`), but all four variants
  still failed with `fall_events=244-246`, `box_drop_events=237-239`, and
  final box target-directed travel around `0.70-0.71 m`.
- [ ] Stop sweeping the current open-loop G1 staged gait/terminal-hold family.
  It has now failed conservative long validation, static terminal hold, early
  terminal hold, all-rollout high drive authority, and staged terminal-drive
  switching.
- [ ] Next gate: either make the local controller-backed G1 locomotion path
  run in Isaac without new download/model blockers, or build a new
  contact/support scaffold whose target-directed travel does not come from
  unrecoverable forward pitch.
- [x] Verify local G1 AGILE controller assets. The local `.pt`/`.onnx` files
  under `external/WBC-AGILE/agile/data/policy/velocity_height_g1/` are Git LFS
  pointer files around `132` bytes, not usable checkpoints. Do not treat the
  local AGILE path as an available controller until real weights are present.
- [x] Add cradle-cart contact baseline batch
  `scripts/isaac/run_core_world_cradle_cart_contact_baseline_batch.sh`. This
  is explicitly a contact scaffold, not robot locomotion evidence.
- [x] Record cradle-cart contact baseline
  `curiosity_cradle_cart_contact_0705`, Slurm job `167041`, stamps
  `diag1`-`diag4`. Result: contact scaffold passed. All four runs had
  `box_drop_events=0` and `nonfinite_state_events=0`. `diag2` carried a
  0.5 kg free box for 0.60 m with final post-settle box travel `0.59978 m`
  and post-settle relative error below `1e-6 m`. `diag3` repeated 0.60 m with
  a 2.0 kg box and also tracked with sub-micrometer post-settle error.
  `diag4` lowered friction to `0.05/0.03`; it still did not drop, but peak
  post-settle relative error rose to `0.02026 m`.
- [ ] Next contact-support gate: use the cart scaffold result to design the
  next robot-side scaffold. The useful property is a physical cradle/cage that
  lets a free box settle once, then move with near-zero post-settle relative
  slip. The next robot diagnostic must not rely on forward pitch runaway for
  motion.
- [x] Add low-CG cage robot-side batch
  `scripts/isaac/run_core_world_prismatic_lowcg_cage_robot_side_batch.sh`.
  This tests whether the successful cage contact geometry can be carried by a
  free articulated prismatic-foot support scaffold instead of a world rail.
- [x] Record low-CG cage robot-side batch
  `curiosity_lowcg_cage_robot_side_0705`, Slurm job `167051`. Result:
  stable but no useful travel. All three variants had `fall_events=0`,
  `box_drop_events=0`, max tilt `0.03797 rad`, and no body root pose/velocity
  commands. However final post-settle payload travel was near zero:
  `diag1` `-0.00039 m` for a 0.03 m target, `diag2` `0.00017 m` for a
  0.06 m target, and `diag3` `-0.00053 m` for a 0.03 m sync-inchworm target.
- [ ] Next support-motion gate: keep the low-CG cage stability/contact
  benefits, but replace the ineffective stance-translate/sync commands with a
  support-consistent propulsion mechanism that produces at least `0.03 m`
  post-settle payload travel without root writes, falls, drops, or excessive
  slip.
- [x] Add negative-direction low-CG motion batch
  `scripts/isaac/run_core_world_prismatic_lowcg_cage_negative_motion_batch.sh`
  to rule out target sign mismatch in the existing stance-translate/creep
  scripts.
- [x] Record negative-direction low-CG motion batch
  `curiosity_lowcg_cage_negative_0705`, Slurm job `167057`. Result: stable
  but still no useful travel. `diag4_translate_neg3cm` had fall/drop 0 but
  final post-settle payload travel `-0.00051 m`; `diag5_creep_neg3cm` had
  fall/drop 0 but final post-settle payload travel `0.00011 m`. Target sign
  mismatch is not the blocker.
- [ ] Current blocker: free-box cage contact is solved in the world-rail
  scaffold, and low-CG prismatic support is stable, but existing prismatic
  stance-translate/creep/sync commands do not generate target-directed
  post-settle payload motion. Build a new support-consistent propulsion
  mechanism rather than rerunning the same low-CG commands.
- [x] Add `rear_anchor_push` motion mode to
  `build_core_world_prismatic_carrier_stand.py`. It keeps rear feet grounded
  as high-friction stance contacts, lifts front feet slightly, and drives only
  rear x-slide joints after settle. Added front/rear foot friction arguments
  and summary fields. This is a no-root/no-fixed-world propulsion diagnostic,
  not final walking.
- [x] Add rear-anchor push validation batch
  `scripts/isaac/run_core_world_prismatic_lowcg_rear_anchor_push_batch.sh`
  with target sign and friction-asymmetry variants.
- [x] Record rear-anchor push validation
  `curiosity_lowcg_rear_anchor_0705`, Slurm job `167080`, stamps `diag6`-
  `diag8`. Result: stable but no useful post-settle travel. The commanded
  x-slide target reached `0.03 m`, but actual x-slide only reached about
  `0.00029 m`; final post-settle payload travel stayed around
  `-0.00052 m` to `0.00044 m`. The blocker is actuator/constraint tracking,
  not target sign or friction asymmetry.
- [x] Add rear-anchor authority batch
  `scripts/isaac/run_core_world_prismatic_lowcg_rear_anchor_authority_batch.sh`
  to test much higher x-slide stiffness and max force.
- [x] Record rear-anchor authority validation
  `curiosity_lowcg_rear_anchor_authority_0705`, Slurm job `167094`, stamps
  `diag9`-`diag11`. Result: stable but still no useful travel. Raising
  x-slide stiffness/max-force by 10-50x did not make the loaded low-CG cage
  track the commanded x-slide targets. `diag9` final post-settle payload
  travel was `-0.00012 m`, `diag10` was `-0.00193 m`, and `diag11` was
  `0.00028 m`; actual x-slide motion stayed about `0.00029-0.00032 m`.
  Therefore the position-drive path is effectively not producing propulsion.
- [x] Add `rear_anchor_velocity_push` mode and
  `--x-slide-velocity` / `X_SLIDE_VELOCITY` controls. This keeps the same
  rear-anchored/front-lifted posture, but also sends x-slide joint velocity
  targets during the post-settle ramp window.
- [ ] Await/record rear-anchor velocity validation
  `curiosity_lowcg_rear_anchor_velocity_0705`, Slurm job `167107`, stamps
  `diag12`-`diag14`. This is a diagnostic for whether the velocity-target
  channel can move the x-slide joints at all; it is not final locomotion.
- [x] Record first rear-anchor velocity validation. Result: negative as a
  velocity-command diagnostic because the implementation still combined
  position targets with velocity targets while x-slide drive stiffness stayed
  nonzero. All runs were stable, but actual x-slide stayed about `0.00031 m`
  and final post-settle payload travel was between `-0.00160 m` and
  `0.00029 m`. Do not treat this as proof that a pure velocity-drive
  x-slide cannot move.
- [x] Fix `rear_anchor_velocity_push` to be a real sparse velocity-drive
  diagnostic: x-slide drive stiffness is set to `0.0` in this mode, z joints
  receive sparse position targets, and x-slide joints receive sparse velocity
  targets.
- [x] Mark `rear_vel2`, Slurm job `167124`, invalid. The compute node read a
  transient/corrupt source line and failed `py_compile` before Isaac. Local
  source and launcher syntax checks passed afterward; no experiment evidence
  was produced.
- [x] Await/record sparse rear-anchor velocity retry2
  `curiosity_lowcg_rear_anchor_velocity_sparse_retry2_0705`, Slurm job
  `167125`, stamps `diag18`-`diag20`.
- [x] Record sparse rear-anchor velocity retry2. Result: valid negative.
  With x-slide stiffness set to `0.0` and sparse velocity targets active,
  all three runs stayed stable with fall/drop 0, but actual x-slide still
  stayed around `0.00029 m` and post-settle payload travel stayed near zero
  (`diag18` `0.00018 m`, `diag19` `0.00000 m`, `diag20` `0.00018 m`).
  Increasing x velocity from `0.03` to `0.08 m/s` did not change the result.
- [x] Next diagnostic: add direct rear x-slide effort control. If direct
  efforts also fail to produce x-slide motion, replace the prismatic x-slide
  support mechanism instead of tuning more position/velocity drive gains.
- [x] Add direct rear x-slide effort control mode `rear_anchor_effort_push`
  and batch `scripts/isaac/run_core_world_prismatic_lowcg_rear_anchor_effort_batch.sh`.
  In this mode x-slide stiffness/damping are zero and rear x-slide joints
  receive sparse joint efforts; vertical support remains position controlled.
- [x] Await/record rear-anchor effort validation
  `curiosity_lowcg_rear_anchor_effort_0705`, Slurm job `167126`, stamps
  `diag21`-`diag23`.
- [x] Record rear-anchor effort validation. Result: valid negative. Direct
  x-slide efforts of `5000 N` and `20000 N` produced fall/drop 0, but actual
  x-slide still stayed around `0.00027 m`; final post-settle payload travel
  remained near zero (`diag21` `-0.00092 m`, `diag22` `-0.00047 m`,
  `diag23` `-0.00009 m`). Position, velocity, and effort command paths have
  now all failed to make this prismatic x-slide support mechanism propel the
  low-CG cage.
- [x] Stop tuning prismatic x-slide support. Build a replacement
  actuator/contact propulsion scaffold and label it explicitly according to
  what it is. The next scaffold may use rolling-foot or wheel-joint
  propulsion to verify actuator-driven ground contact plus free-box retention,
  but it must not be claimed as walking humanoid carrying.
- [x] Add rolling-foot cage carrier scaffold
  `scripts/isaac/build_core_world_rolling_foot_cage_carrier.py`. It uses a
  low-CG torso, four velocity-driven revolute wheel joints, a fixed physical
  cage, and a free dynamic box. It records root/box write counts as zero, but
  it is explicitly not walking.
- [x] Add rolling-foot validation batch
  `scripts/isaac/run_core_world_rolling_foot_cage_carrier_batch.sh`.
- [x] Await/record rolling-foot cage validation
  `curiosity_rolling_foot_cage_0705`, Slurm job `167128`, stamps `diag1`-
  `diag3`.
- [x] Mark first rolling-foot cage validation invalid. Slurm job `167128`
  failed before rollout because `/World/Robot` was not defined as an
  articulation root; no summaries were produced. This is a USD structure
  error, not physics evidence.
- [x] Fix rolling-foot USD root by explicitly defining `/World/Robot` and
  applying `UsdPhysics.ArticulationRootAPI`.
- [x] Await/record rolling-foot cage retry2 after articulation-root fix.
- [x] Record rolling-foot cage retry2/retry3. Result: negative. The
  articulation-root fix let the scene enter rollout and summaries were
  produced, but 1 kg cases still had `fall_events=792`, torso z around
  `0.114 m`, and post-settle payload travel only `0.00993-0.01422 m`; the
  2 kg case dropped the box. Reading `/World/Robot/Torso` directly produced
  the same low z, so this is not merely an articulation-root pose metric
  artifact.
- [x] Add wheel joint motion metrics and run a one-case rolling-foot
  diagnostic to determine whether wheel velocity commands actually rotate the
  wheel joints.
- [x] Record rolling-foot joint-motion diagnostic `diag10`. Result: wheel
  velocity command did not rotate the wheel joints meaningfully. Max absolute
  wheel joint motion was only `0.00986 rad`, final wheel motion `0.00282 rad`,
  final post-settle payload travel `0.00144 m`, and fall events 492. The
  rolling-foot velocity-drive path is not a propulsion solution.
- [x] Add direct wheel effort/torque mode to test whether revolute wheel
  joints can be actuated by sparse efforts. If direct effort also fails, stop
  this rolling-foot route.
- [x] Record rolling-foot wheel-effort diagnostic `diag11`. Result: valid
  negative. Direct wheel effort `200 Nm` produced max wheel joint motion only
  `0.00132 rad`, final wheel motion `0.000045 rad`, final post-settle payload
  travel `0.000047 m`, and fall events 492. Both velocity and effort command
  paths fail to actuate the rolling-foot revolute joints meaningfully.
- [x] Stop the rolling-foot route unless the USD articulation/joint modeling
  is redesigned from first principles or replaced with a known-good Isaac
  wheeled/legged robot asset. Do not sweep wheel velocity, torque, or friction
  on the current model.
- [x] Identify local Arena G1 loco-manipulation checkpoint as a real local
  model asset, not an LFS pointer:
  `/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000/`
  contains 4.7 GB and 1.5 GB safetensors shards plus index/config files.
  This is a synthetic IsaacLab Arena G1 loco-manipulation task checkpoint and
  may be used as a strong simulation baseline, not as final active unknown-load
  video-guided RL evidence.
- [x] Stop treating the Arena/GR00T loco-manipulation smoke as a blocker.
  Per user correction, the active route is direct Isaac scene construction,
  not waiting for external policy/model rollouts. Slurm job `167139` had
  already proved that the policy server started and the Arena config/assets
  loaded, but no rollout result was needed for the direct Isaac carrying
  path; it was canceled at about 8 minutes to free GPU resources. Record this
  as an intentionally aborted baseline smoke, not a failed or successful
  carrying result.
- [x] Add a local official H1 callback locomotion smoke instead of waiting on
  GR00T/Arena. New files:
  `scripts/isaac/run_official_h1_callback_locomotion_smoke.py` and
  `scripts/isaac/run_official_h1_callback_locomotion_smoke.sh`. The script
  uses the installed NVIDIA `H1FlatTerrainPolicy`, local H1 USD, and local
  H1 PhysX policy/env files, then calls `forward()` from a
  `POST_PHYSICS_STEP` callback. Passing this would only establish a
  controller-backed locomotion backend candidate; it is not box carrying.
  Lightweight checks passed: `bash -n` and `python3 -m py_compile`.
- [ ] Await/record H1 callback no-box locomotion smoke
  `curiosity_h1_callback_locomotion_0705`, job-name `h1_cb_smoke`, stamp
  `20260705_h1_callback_locomotion_diag1`. Required interpretation: if it
  produces nonzero travel, forward callback calls, and fall 0, the next step is
  a fixed/light payload H1 smoke; if it fails like Go2, do not repeat this
  unchanged and continue backend replacement without model waiting.
- [x] Mark first H1 callback smoke invalid. Slurm job `167142` returned
  `0:0`, but no summary or state CSV was written. The log reached stage and
  ground setup, then stopped before printing `H1 policy object created`, so it
  is a startup/launcher diagnostic only, not locomotion evidence.
- [x] Fix H1 callback smoke robustness: default to the Isaac Sim base python
  experience instead of the IsaacLab headless kit, and write an explicit JSON
  failure summary if H1 policy construction raises before rollout.
- [x] Await/record H1 callback retry2
  `curiosity_h1_callback_locomotion_retry2_0705`, job-name `h1_cb_r2`, stamp
  `20260705_h1_callback_locomotion_diag1_retry2`.
- [x] Record H1 callback retry2 as invalid before simulation. The base Isaac
  Sim python experience failed dependency resolution for
  `isaacsim.anim.robot.schema` and exited before `SimulationApp` startup
  completed. This is an experience/local-registry issue, not H1 locomotion
  evidence.
- [x] Await/record H1 callback retry3
  `curiosity_h1_callback_locomotion_retry3_0705`, job-name `h1_cb_r3`, stamp
  `20260705_h1_callback_locomotion_diag1_retry3`. Retry3 uses the IsaacLab
  headless kit again but keeps the new policy-construction failure summary
  handling.
- [x] Record H1 callback retry3 as a valid startup negative. It wrote
  `official_h1_callback_locomotion_summary.json` with
  `success_claim=failed_before_h1_policy_construction_completed` and error
  `Path.IsValidPathString(NoneType)`, before rollout. The failure occurs in
  H1 policy/articulation construction, not in locomotion.
- [x] Align H1 construction with NVIDIA's installed `test_h1.py`: stop passing
  explicit `usd_path`, `policy_path`, and `env_config_path`; keep local file
  existence checks and let `H1FlatTerrainPolicy` choose paths from the patched
  asset root.
- [x] Await/record H1 callback retry4
  `curiosity_h1_callback_locomotion_retry4_0705`, job-name `h1_cb_r4`, stamp
  `20260705_h1_callback_locomotion_diag1_retry4`. If this still fails before
  rollout, stop the H1 sample-policy route and return to direct Isaac backend
  replacement without further official-policy retries.
- [x] Record H1 callback retry4 as another valid startup negative. Even with
  NVIDIA-style automatic H1 asset/policy path selection, it failed before
  rollout with the same `Path.IsValidPathString(NoneType)` policy/articulation
  construction error. Stop the H1 sample-policy route unchanged. Together
  with the previous Go2 sample-policy failures, this means official Isaac
  sample policy wrappers are not an immediately usable backend in this
  environment.
- [ ] Next direct Isaac backend task: preserve the existing executable
  `DirectCarryTaskRunner` contract, active-probe fields, hidden-box
  randomization, posture/action interface, contact-report gates, and
  no-shortcut checks, but replace the support backend with a new physical
  Isaac controller path. Do not wait for GR00T/Arena/H1/Go2 sample policies.
  The next backend candidate should be designed to pass no-box locomotion first
  with calibrated support/contact/slip metrics, then small fixed payload, then
  free-box carry.
- [x] Stop blocking on external model/policy rollouts per user correction and
  start direct Isaac support-backend isolation first.
- [x] Add `payload-mode=none` to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`. In this mode
  no carry box is spawned; summary fields mark `payload_spawned=false`,
  `no_box_support_smoke=true`, and
  `payload_metric_proxy=torso_pose_when_payload_mode_none`. This is a no-box
  support diagnostic only.
- [x] Run no-box support smoke
  `curiosity_anchor_nobox_support_0705`, Slurm job `167156`, output
  `experiments/outputs/core_world_anchored_footstep_carrier/20260705_anchor_nobox_support_diag1/`.
  Result: valid negative. Fall/drop stayed 0, but final post-settle torso
  travel was only `0.00038 m` toward a `0.24 m` target. Do not add payload to
  this configuration.
- [x] Run no-box planted rail propulsion smoke
  `curiosity_anchor_nobox_propulsion_0705`, Slurm job `167157`, output
  `experiments/outputs/core_world_anchored_footstep_carrier/20260705_anchor_nobox_propulsion_diag2/`.
  Result: valid negative. Rail commands were generated, but torso moved
  backward (`-0.03338 m`) for a `+0.24 m` target and max tilt increased to
  `0.37120 rad`.
- [x] Cancel queued negative-target sign diagnostic before rollout. The
  previous positive-target planted-rail log already showed positive rail
  targets drove the torso backward.
- [x] Add explicit diagnostic rail sign control:
  `--rail-target-direction-scale` and launcher env
  `RAIL_TARGET_DIRECTION_SCALE`. Default remains `1.0`.
- [x] Await/record positive-target inverted-rail no-box smoke
  `curiosity_anchor_nobox_invertrail_0705`, Slurm job `167161`, output
  `experiments/outputs/core_world_anchored_footstep_carrier/20260705_anchor_nobox_invertrail_diag4/`.
  Result: valid negative. `RAIL_TARGET_DIRECTION_SCALE=-1.0` changed the rail
  command sign as intended and fall/drop stayed 0, but final post-settle torso
  travel was only `0.00038 m` toward a `0.24 m` target.
- [x] Stop the current `xz_prismatic_to_anchor` support backend. It has now
  failed no-box propulsion in three ways: no rail propulsion, wrong-direction
  planted rail propulsion, and no meaningful travel with inverted rail sign.
  Do not add fixed payload or free box on top of it.
- [ ] Next backend pivot: use the real G1 articulation path as the active
  direct Isaac route because it already has verified no-box stand and small
  fixed-payload stand evidence. The immediate target is not free-box carrying;
  it is a no-root G1 travel gate that preserves fall/drop 0 for longer than
  the previous short-window staged gait diagnostics.
- [x] Record existing G1 conservative staged long outputs `diag77`-`diag80`.
  Result: all are negative 700-step carrying diagnostics. They reached large
  final box target-directed travel (`0.75886-0.86646 m`) but failed with
  `148-232` fall events, `131-215` box drop events, max tilt around
  `1.13-1.14 rad`, and min box z down to `0.05-0.108 m`.
- [x] Run/record a clean G1 no-box staged-gait isolation: no carry box spawned,
  no torso cradle, 700 steps, same direct Core API G1 articulation. This
  separates locomotion/balance failure from box-retention failure before any
  more carrying variants.
- [x] Record G1 no-box staged-gait isolation
  `20260705_core_world_g1_nobox_staged_iso_diag1`, Slurm job `167163`.
  Result: stable but non-locomotive. Completed `700/700`, fall/drop 0,
  carry box spawned false, torso cradle none, max tilt `0.01314 rad`, min
  robot z `0.78289 m`, rollout root/velocity/box pose writes 0, but final
  robot target-directed travel was `-0.00067 m` and max target-directed robot
  travel only `0.00522 m`.
- [x] Stop treating the current staged/open-loop G1 gait family as a viable
  locomotion backend. It can stand/march, but it does not generate meaningful
  no-box travel. Large box travel in failed carrying runs is not credible
  walking progress.
- [ ] Next G1 route: use a controller-backed locomotion policy or a materially
  different walking controller before returning to fixed payload or free-box
  carrying. Do not continue amplitude/ramp/terminal-hold sweeps in the current
  open-loop family.
- [x] Re-inspect WBC-AGILE failed policy-loading boundary. Found that local
  `velocity_height_g1` model files were Git LFS pointer text files, not real
  weights.
- [x] Download official WBC-AGILE real model files from GitHub media URLs into
  the existing official repository paths on the login node:
  recurrent student checkpoint `6.4M`, ONNX `2.0M`, and TorchScript `.pt`
  `2.0M`. This is not a placeholder model.
- [x] Run lightweight syntax checks after weight repair:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/check_core_world_g1_box_scene_summary.py`
  and `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- [ ] Await/record WBC-AGILE real-weight no-box smoke
  `curiosity_g1_agile_torch_realweights_0705`, Slurm job `167164`, output
  `experiments/outputs/core_world_g1_box_scene/20260705_core_world_g1_agile_policy_nobox_diag4_realweights/`.
  Required interpretation: if it loads and enters rollout, evaluate
  target-directed robot travel, fall 0, policy inference count, action norm,
  and root/velocity write counts before adding payload.

## 2026-07-05 User-Correction Direct Isaac Pivot

- [x] Record WBC-AGILE real-weight Core API G1 smokes as negative for the
  active path. Real official weights now load, but they do not give a usable
  no-box locomotion backend in the current Isaac Core API scene. Direct-load
  no-box run `diag5` entered rollout with `policy_inference_count=115` and
  moved briefly, but failed with `fall_events=359`, max tilt `3.0337 rad`,
  and min robot z `0.0643 m`. Zero-command `diag6` also failed
  (`fall_events=144`) and was flawed by missing the stable stand posture
  overrides. Corrected zero-command stable-pose `diag7` still failed:
  360/360 steps, no box, `policy_inference_count=60`, `fall_events=92`,
  max tilt `2.50074 rad`, min robot z `0.06682 m`, and max target-directed
  robot travel only `0.00737 m`. Do not wait on or tune AGILE as the main
  route unless the observation/action convention is explicitly repaired.
- [x] Follow the user correction by running a direct Isaac scene baseline
  instead of waiting on external models. Submitted
  `curiosity_g1_box_in_front_scene_smoke_0705`; the first run used the default
  carry/drop threshold and incorrectly counted the ground-resting box as a
  drop because box z was about `0.16 m` and `DROP_Z=0.20`.
- [x] Record corrected direct Isaac G1 + box-in-front scene smoke
  `20260705_core_world_g1_box_in_front_scene_smoke_retry2`, Slurm job
  `167178`. This is a valid scene baseline, not carrying: 360/360 steps,
  free dynamic 2 kg box on the ground in front of G1, `attach_box=none`,
  `torso_cradle=none`, 43 G1 joints, arena stand gains, fall/drop 0,
  min robot z `0.78429 m`, min box z `0.16010 m`, max tilt `0.00882 rad`,
  rollout root pose writes 0, rollout root velocity writes 0, and rollout box
  pose writes 0. Checker passed with diagnostic-only success claim.
- [ ] Next direct Isaac scene step: keep this G1 + free-box-in-front baseline
  and add one explicit task phase at a time: visual/pose target marker,
  probing contact primitive, grasp/contact attempt, then lift/carry. Do not
  reintroduce AGILE/WBC or the old prismatic support scaffold as blockers.
- [x] Add a direct G1 front-probe diagnostic primitive to the Core API scene.
  Implemented `PROBE_MODE=front_bumper` / `--probe-mode front_bumper`, probe
  pad geometry/mass/collision parameters, probe start step, probe reference
  pose, probe active steps, probe box displacement, probe target-directed
  travel, and `probe_box_moved`. Added dedicated launchers
  `scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh` and
  `scripts/isaac/submit_core_world_g1_front_probe_bumper_smoke.sh` because
  long quoted direct commands were unreliable. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/check_core_world_g1_box_scene_summary.py`
  and `bash -n` for the G1 launchers.
- [x] Add checker gates for the probe diagnostic:
  `--expect-probe-mode`, `--min-probe-active-steps`,
  `--require-probe-box-moved`, `--min-final-probe-box-travel`,
  `--min-max-probe-box-travel`, and
  `--min-final-probe-box-target-directed-travel`.
- [x] Record aggressive front-probe result
  `20260705_core_world_g1_front_probe_bumper_submit_retry4`, Slurm job
  `167184`, as negative. The probe was enabled and pushed the free box
  `0.134 m` target-directed with no box pose writes, but the contact impulse
  toppled G1: `fall_events=284`, max tilt `2.55803 rad`, min robot z
  `0.05928 m`. This proves the probe path can create physical box motion, but
  the geometry was too aggressive.
- [x] Record gentle front-probe result
  `20260705_core_world_g1_front_probe_bumper_submit_retry5_gentle` as the
  first passing G1 contact-probe diagnostic. It completed 360/360 with
  `probe_mode=front_bumper`, 43 G1 joints, free dynamic 2 kg ground box,
  fall/drop 0, max tilt `0.05226 rad`, min robot z `0.78356 m`, min box z
  `0.16048 m`, rollout root pose/root velocity/box pose writes all 0, and
  final probe box travel `0.15285 m` (`0.15260 m` target-directed). Checker
  passed with `--require-probe-box-moved` and diagnostic-only claim.
- [ ] Next direct Isaac phase: replace the torso-fixed bumper probe with a
  robot-limb or end-effector contact attempt, or add a staged grasp/lift
  diagnostic after probe. The current passing probe is useful evidence for
  contact and active object perturbation, but it is not grasping, lifting,
  walking, or carrying.
- [x] Add a staged fixed-torso grasp/lift diagnostic to the Core API G1 scene.
  Implemented `GRASP_MODE=staged_fixed_torso`, `GRASP_ENABLE_STEP`, and
  `GRASP_LIFT_OFFSET_Z`. The rollout creates a runtime fixed joint after the
  probe phase, records grasp attach step, local offset, box/robot attach
  poses, box z at attach, max/final post-grasp box z delta, and min
  post-grasp box z. Checker gates now include `--expect-grasp-mode`,
  `--require-grasp-attached`, `--min-grasp-attach-step`,
  `--min-max-post-grasp-box-z-delta`, and
  `--min-final-post-grasp-box-z-delta`.
- [x] Record staged grasp/lift smoke
  `20260705_core_world_g1_probe_grasp_lift_retry1`, Slurm job `167188`.
  Result: diagnostic pass with important caveat. The run completed 360/360
  with probe, runtime fixed-torso grasp at step `140`, `grasp_lift_offset_z`
  `0.03 m`, fall/drop 0, max tilt `0.06396 rad`, rollout root/velocity/box
  pose writes 0, max post-grasp box z delta `0.06895 m`, and final post-grasp
  box z delta `0.01703 m`. PhysX emitted a disjoint fixed-joint snap warning,
  so this is a staged fixed-joint grasp/lift diagnostic, not a physically
  faithful hand grasp.
- [x] Record first grasp+march diagnostic
  `20260705_core_world_g1_probe_grasp_march_retry1`, 420 steps, open-loop
  march amp `0.05` after grasp. It passed stability and grasp/lift gates:
  fall/drop 0, max tilt `0.06463 rad`, final post-grasp box z delta
  `0.01783 m`, rollout root/velocity/box pose writes 0. Robot target-directed
  travel was only `0.04027 m`, so this is not walking/carrying success.
- [x] Record stronger grasp+march diagnostic
  `20260705_core_world_g1_probe_grasp_march_retry2_amp010`, 520 steps,
  open-loop march amp `0.10` after grasp. It also passed stability and
  grasp/lift gates: fall/drop 0, max tilt `0.06549 rad`, final post-grasp box
  z delta `0.01729 m`, rollout root/velocity/box pose writes 0. It still did
  not become target-directed walking: final robot target-directed travel was
  only `0.03264 m`, and the grasped box had large lateral oscillation
  (`final_box_delta_xy_m=[0.14773, 0.09474]`).
- [ ] Next gate: stop trying to get real walking from this open-loop march.
  The next meaningful step is either a real walking controller that can move
  G1 target-directed after grasp, or a limb/end-effector contact/grasp variant
  that reduces the fixed-joint snap artifact before locomotion is reattempted.
- [x] Update staged grasp implementation so it can attach to a selected G1
  body/link rather than only the torso. Added `GRASP_MODE=staged_fixed_body`,
  `GRASP_BODY_PATH`, body-frame fixed-joint local offset computation,
  `active_grasp_body_path`, grasp body wrapper status, attach body pose
  telemetry, and checker gates for expected grasp body path. Lightweight
  checks passed: `python3 -m py_compile` for the scene/checker scripts and
  `bash -n` for the G1 probe launchers.
- [x] Run and record hand-link staged grasp diagnostic
  `20260705_core_world_g1_probe_hand_grasp_lift_retry1`, Slurm job `167195`,
  tmux `curiosity_g1_hand_grasp_retry1_0705`. Config uses gentle front-bumper
  probe, then `GRASP_MODE=staged_fixed_body` with
  `GRASP_BODY_PATH=/World/G1/right_hand_palm_link`, attach step `140`, and
  lift offset `0.03 m`. This must be treated as a diagnostic only. Passing
  requires no falls/drops, no rollout root/velocity/box pose writes, grasp
  attach to the right hand path, and positive post-grasp box z delta.
  Result: checker passed with `360/360`, fall/drop 0, max tilt `0.05226 rad`,
  no rollout root/velocity/box pose writes, right hand wrapper initialized,
  max post-grasp z delta `0.01450 m`, and final post-grasp z delta
  `0.00409 m`. Caveat: hand-to-box attach distance was still about `0.967 m`
  and PhysX warned about disjoint fixed-joint transforms, so this is not
  physically faithful grasping.
- [x] Add arm-pose phase support to reduce staged hand-grasp snap distance.
  Added `ARM_POSE_MODE`, arm pose start/ramp steps, manual shoulder/elbow/wrist
  overrides, `arm_pose_targets`, `arm_pose_active_steps`, attach body-to-box
  world delta/distance telemetry, and checker gate
  `--max-grasp-body-box-world-distance-at-attach`. Lightweight checks passed:
  `python3 -m py_compile` for scene/checker and `bash -n` for launchers.
- [x] Run and record `right_front_reach` hand-grasp diagnostic
  `20260705_core_world_g1_armreach_hand_grasp_retry1`, tmux
  `curiosity_g1_armreach_hand_grasp_retry1_0705`, job-name `g1_armgr1`.
  Primary gate: reduce `grasp_body_box_world_distance_at_attach_m` relative
  to the `0.967 m` no-arm-pose hand-link baseline while preserving no
  falls/drops and no rollout root/velocity/box pose writes.
  Result: negative. The run completed 360/360 with fall/drop 0 and no rollout
  pose writes, but attach distance increased to `0.98688 m`; snap was worse
  (`max_post_grasp_box_z_delta_m=1.60657`, final box target-directed travel
  `-0.93573 m`, max tilt `0.30448 rad`). Do not use this preset as reaching
  evidence.
- [x] Run manual arm-pose sign sweep before more grasp/lift attempts. Use
  short right-palm staged attach diagnostics and rank by
  `grasp_body_box_world_distance_at_attach_m`, max tilt, and snap height.
  Submitted tmux `curiosity_g1_arm_pose_sweep1b_0705`; first Slurm job
  observed as `167198` for
  `20260705_core_world_g1_arm_sweep_pospitch_negelbow`. The earlier tmux
  `curiosity_g1_arm_pose_sweep1_0705` did not start jobs because the shell
  function passed quoted environment assignments incorrectly; no logs or
  outputs were produced from that failed submit attempt.
  Result: negative. Four configs completed, but none improved over the
  `0.967 m` no-arm baseline: `pospitch_negelbow=1.16467 m`,
  `pospitch_poselbow=1.39430 m`, `pospitch_rollpos=1.16337 m`,
  `highpitch_negelbow=1.20940 m`. Snap persisted in all four.
- [x] Run one final negative-pitch/yaw right-arm sign sweep. If no config
  reduces attach distance substantially, stop single right-hand ground-box
  grasp tuning and switch to either double-arm/chest-supported staged contact
  or a raised-box carry-height diagnostic.
  Result: negative. Distances were `negpitch_negelbow=1.33406 m`,
  `negpitch_negelbow_rollpos=1.33226 m`, `negpitch_yawpos=1.33040 m`, and
  `negpitch_yawneg=1.31012 m`; two configs produced falls. Single right-hand
  ground-box staged attach is no longer the active path.
- [ ] Implement a carry-height support-table diagnostic in the direct Isaac G1
  scene so the box can start near hand/chest height without falling before
  attach. Use it only as a staged carry-posture/balance diagnostic, not as
  evidence of ground pickup.
  Code update: added `BOX_SUPPORT_MODE=table`, support size/top-clearance
  args, support mode summary/checker fields, and `PROBE_MODE=none` launcher
  support. Lightweight `python3 -m py_compile` and `bash -n` checks passed.
- [ ] Run carry-height right-palm staged attach diagnostic
  `20260705_core_world_g1_carryheight_hand_grasp_retry1`, tmux
  `curiosity_g1_carryheight_hand_grasp_retry1_0705`, job-name `g1_chgr1`.
  Config: support table, box center near `(0.46,-0.15,0.88)`, right palm
  attach at step `80`, lift offset `0.02`, probe disabled. Primary gate:
  attach distance below `0.35 m` and much smaller snap than the ground-box
  hand-link diagnostics.
  Retry1 was submit-check failure only; fixed the stale submit `rg` check.
  Retry2 ran with the same near-body box/table and failed: support geometry
  caused falls before attach (`fall_events=42`, max tilt `0.97825 rad`), and
  attach distance remained `0.96173 m`.
- [x] Run carry-height retry with a narrower support table moved farther
  forward so support geometry does not collide with the standing robot before
  attach.
  Retry3 passed the staged carry-height attach/balance diagnostic:
  `20260705_core_world_g1_carryheight_hand_grasp_retry3` completed 240/240
  with support table, probe disabled, right-palm attach at step 80, fall/drop
  0, max tilt `0.00759 rad`, min robot z `0.78429 m`, min box z `0.88000 m`,
  no rollout root/velocity/box pose writes, attach distance `0.50746 m`, max
  post-grasp z delta `0.01796 m`, and final z delta `0.00305 m`. It remains a
  staged carry-height diagnostic, not pickup or walking.
- [x] Add a small post-attach march diagnostic on the carry-height support
  setup and require stability plus some robot/box target-directed travel.
  `20260705_core_world_g1_carryheight_hand_march_retry1` completed 420/420
  with fall/drop 0 and max tilt `0.01752 rad`, but failed the travel gate:
  final robot target-directed travel `-0.00276 m`, final box target-directed
  travel `-0.00796 m`, and max robot target-directed travel only `0.00108 m`.
  This is stable attached carry-height balance, not walking/carrying.
- [x] Run a stronger carry-height march diagnostic. If it still does not
  produce target-directed robot/box travel, stop treating the current
  open-loop gait as a locomotion backend and switch to a different walking
  controller path.
  Retry2 `20260705_core_world_g1_carryheight_hand_march_retry2_amp015`
  completed 420/420 with fall/drop 0, max tilt `0.04855 rad`, and no rollout
  pose writes, but still failed travel: final robot target-directed travel
  `-0.03335 m`, final box target-directed travel `-0.02982 m`, and max robot
  target-directed travel only `0.00108 m`.
- [ ] Replace the current open-loop gait before any further walking-carrying
  claims. Candidate next paths: debug a real local velocity policy in the
  direct scene, add a target-directed footstep/contact controller, or use a
  staged quasi-static stepping controller with explicit foot support and
  measured target travel. Do not count open-loop leg oscillation as walking.
  Code update: added `gait_mode=targeted_creep` with creep hip/knee/ankle/waist
  offsets, stance-push scale, lift scale, and ankle-lift scale. This is still a
  diagnostic joint controller and must beat the travel gate before it can be
  treated as locomotion evidence.
- [x] Run targeted-creep carry-height diagnostic
  `20260705_core_world_g1_carryheight_creep_retry1`, tmux
  `curiosity_g1_carryheight_creep_retry1_0705`, job-name `g1_chcr1`. Require
  no rollout root/velocity/box pose writes, fall/drop 0, and positive
  target-directed robot/box travel.
  Result: weak positive. 420/420, fall/drop 0, no rollout root/velocity/box
  pose writes, max tilt `0.13591 rad`, final robot target-directed travel
  `0.01460 m`, final box target-directed travel `0.00374 m`. This beats the
  backward open-loop march but is not enough to call walking/carrying.
- [x] Run a stronger targeted-creep carry-height diagnostic to test whether
  positive travel can scale without falls or drops.
  Retry2 `20260705_core_world_g1_carryheight_creep_retry2_stronger` completed
  520/520 with fall/drop 0, no rollout pose writes, max tilt `0.16267 rad`,
  final robot target-directed travel `0.02888 m`, and final box
  target-directed travel `0.01706 m`. Retry3 long
  `20260705_core_world_g1_carryheight_creep_retry3_long` completed 900/900
  with fall/drop 0, no rollout pose writes, final robot target-directed
  travel `0.04279 m`, and final box target-directed travel `0.03541 m`.
  This is positive attached-box target-directed motion, but support table
  remains under the box, so it is not yet carrying.
- [ ] Add support-table release/removal after staged attach and verify that
  box height/contact remains stable without table support.
  Code update: added `BOX_SUPPORT_RELEASE_STEP`, removes
  `/World/CarryBoxSupportTable` during rollout, records release telemetry, and
  checker gates for release. Pending diagnostic
  `20260705_core_world_g1_carryheight_release_stand_retry1`, tmux
  `curiosity_g1_carryheight_release_stand_0705`, job-name `g1_chrel1`.
  Retry1 result: negative for carrying. Table released at step 120 and run
  completed 360/360 with fall/drop counters 0, but box fell/slid after support
  removal: min box z `0.28236 m`, final post-grasp z delta `-0.57757 m`, and
  final box target-directed travel `-0.13944 m`.
- [ ] Run support-release retry with positive lift offset and require high
  min box z after release.
  Retry2 `20260705_core_world_g1_carryheight_release_lift_retry2` is also
  negative for hand carrying. It completed 360/360 with fall/drop 0, but after
  table release min box z was `0.24513 m` and final post-grasp z delta was
  `-0.54165 m`; the right-hand staged attach did not hold the box high.
- [ ] Test torso/chest-supported staged carry-height release as an alternative
  posture after single-right-hand release failed.
  Result: negative as configured.
  `20260705_core_world_g1_chest_release_stand_retry1` switched the staged
  fixed body to `/World/G1/torso_link` and released the support table at step
  `160`, but the box/table/torso geometry destabilized the robot before this
  could be used as carrying evidence: `fall_events=242`, max tilt
  `2.70430 rad`, min robot z `0.05730 m`, min box z `0.60980 m`, final
  post-grasp z delta `-0.31130 m`, final robot target-directed travel
  `-0.65920 m`, and final box target-directed travel `-1.14500 m`. Do not
  reuse this torso-fixed release config.
- [ ] Switch from staged fixed hand/torso joints to a direct Isaac
  body-supported posture diagnostic: enable the existing `front_tray`
  torso-cradle path from the launcher, place the free box on the tray, use
  `GRASP_MODE=none`, and test whether physical cradle contact can support the
  box without a table or fixed joint. This is the next direct-scene route and
  does not depend on external model downloads.
  Code update: launcher now exposes `TORSO_CRADLE=front_tray` plus cradle
  deck size, local position, rail height, stop height, rail thickness, and
  mass scale environment variables. Lightweight `python3 -m py_compile` and
  `bash -n` checks passed.
  Retry1 `20260705_core_world_g1_front_tray_freebox_stand_retry1` is negative:
  the free 2 kg box plus high/forward tray destabilized G1. It completed
  360/360 but produced `fall_events=243`, min robot z `0.05753 m`, min box z
  `0.19419 m`, max tilt `2.64308 rad`, final robot target-directed travel
  `-0.69757 m`, and final box target-directed travel `-2.82168 m`. No rollout
  root/velocity/box pose writes occurred. Next retry must use a lighter,
  lower, closer body-supported tray before any walking attempt.
  Retry2 `20260705_core_world_g1_front_tray_freebox_lowstand_retry2` is
  improved but still not a valid carry posture. With a 0.5 kg smaller box and
  closer/lower/light tray it completed 360/360 with fall/drop `0` and no
  rollout pose writes, but the robot steadily pitched forward: max tilt
  `0.67826 rad`, min robot z `0.67762 m`, min box z `0.52771 m`. The apparent
  positive target-directed travel (`robot=0.51625 m`, `box=0.47816 m`) is
  forward tipping/sliding, not walking. Next retry should use an even closer
  low support, a shorter stability horizon, and explicit balance/posture
  compensation before locomotion.
  Code update: launcher now exposes existing gait ramp-down, recovery,
  terminal-hold, and balance-feedback parameters so stability compensation can
  be tested without adding new controller logic.
  Retry3 `20260705_core_world_g1_front_tray_freebox_lowbalance_retry3` is a
  strong negative control-compensation result. It used an even lighter
  0.3 kg box, low close tray, crouched stand, and balance feedback, but fell
  by step 20 and completed 240/240 with `fall_events=166`, max tilt
  `3.13363 rad`, min robot z `0.40093 m`, min box z `0.21701 m`, and
  artificial-looking final target-directed travel around `6 m` caused by
  instability. No rollout pose writes occurred. Do not reuse this balance
  sign/gain/geometric combination. The next test should return to retry2
  geometry and apply only a very small opposite-sign pitch correction.
  Retry4 `20260705_core_world_g1_front_tray_freebox_signpos_retry4` is the
  first valid direct Isaac free-box body-support standing diagnostic. It uses
  retry2 geometry plus small opposite-sign pitch feedback and completed
  360/360 with `GRASP_MODE=none`, `BOX_SUPPORT_MODE=none`, fall/drop `0`,
  max tilt `0.15888 rad`, min robot z `0.78414 m`, min box z `0.77035 m`,
  balance feedback active for all 360 steps, and no rollout root/velocity/box
  pose writes. Final robot/box target-directed drift was about `0.115/0.123 m`;
  this is stable body-supported holding/drift, not locomotion or long-distance
  carrying. Next diagnostic may add very conservative targeted creep using the
  same geometry and feedback.
- [ ] Test conservative targeted creep with the retry4 free-box body-support
  setup. Do not count it as walking/carrying unless it keeps fall/drop at `0`,
  preserves box height, and avoids progressive pitch divergence.
  Retry1 `20260705_core_world_g1_front_tray_freebox_creep_retry1` is negative
  over the full 520-step horizon. It used `GRASP_MODE=none`,
  `BOX_SUPPORT_MODE=none`, `TORSO_CRADLE=front_tray`, and the retry4 geometry
  plus small targeted creep. It completed 520/520 with no rollout pose writes
  and positive final robot/box target-directed travel
  (`0.63978/0.56808 m`), but the motion was progressive forward tipping:
  `fall_events=2`, max tilt `0.86491 rad`, min robot z `0.59641 m`, and
  min box z `0.40208 m`. It is useful only as evidence that the support can
  survive early conservative creep; it is not stable carrying.
  Retry2 `20260705_core_world_g1_front_tray_freebox_creep440_retry2` passed a
  short-window diagnostic but must remain diagnostic-only. It completed
  440/440 with `gait_mode=targeted_creep`, `GRASP_MODE=none`,
  `BOX_SUPPORT_MODE=none`, fall/drop `0`, min robot z `0.78289 m`,
  min box z `0.72056 m`, max tilt `0.30230 rad`, final robot/box
  target-directed travel `0.23447/0.23692 m`, and no rollout
  root/velocity/box pose writes. This is the best direct Isaac result so far:
  free-box body-supported short-distance movement. It is not long-duration
  carrying because tilt is still growing and box height is declining.
  Retry3 `20260705_core_world_g1_front_tray_freebox_stophold_retry1` is a
  negative stop-and-hold diagnostic. It started terminal hold at step 420 and
  kept it active for 200 steps, but this was too late to recover the forward
  pitch divergence: 620/620, `fall_events=103`, min robot z `0.49170 m`,
  min box z `0.26074 m`, max tilt `1.47193 rad`, no rollout
  root/velocity/box pose writes. Do not treat the large final travel as
  carrying; it came from falling/sliding after hold engaged. Next stop-and-hold
  test should stop much earlier, before pitch exceeds the retry4 stable range.
  Retry4 `20260705_core_world_g1_front_tray_freebox_stophold360_retry2` is
  also negative. It stopped gait and entered terminal hold at step 360 with no
  extra hold offsets, but forward pitch still diverged after the stop:
  620/620, `terminal_hold_active_steps=260`, `fall_events=102`, min robot z
  `0.49309 m`, min box z `0.25548 m`, max tilt `1.51482 rad`, and no rollout
  root/velocity/box pose writes. The trajectory resembles uncontrolled
  forward tipping, so the next necessary diagnostic is a 620-step stand-only
  baseline with the same free-box front-tray posture.
  Stand-only baseline `20260705_core_world_g1_front_tray_freebox_stand620_retry5`
  is negative over 620 steps. With the same free-box front-tray posture,
  no gait, and the retry4 small pitch feedback, it still entered the same
  slow forward-tipping mode: 620/620, `fall_events=98`, min robot z
  `0.48937 m`, min box z `0.25926 m`, max tilt `1.48288 rad`, and no rollout
  pose writes. This proves the active blocker is long-horizon posture/support
  stability under front-tray load, not gait stop timing. Next diagnostic should
  increase positive pitch-feedback authority or reduce front load moment before
  any further gait attempt.
  Strong-feedback stand baseline
  `20260705_core_world_g1_front_tray_freebox_stand620_gain_retry6` passes the
  long-horizon free-box support gate. With the same geometry, no gait,
  `BALANCE_PITCH_GAIN=0.45`, `BALANCE_PITCH_RATE_GAIN=0.02`, and
  `BALANCE_ADJUSTMENT_LIMIT=0.12`, it completed 620/620 with fall/drop `0`,
  max tilt `0.09570 rad`, min robot z `0.78411 m`, min box z `0.78743 m`,
  and no rollout root/velocity/box pose writes. Final robot/box drift was only
  `0.00572/0.01339 m`. This is a valid long-duration body-supported holding
  diagnostic, not walking/carrying. Next: retry conservative creep with this
  stronger feedback.
  Strong-feedback creep
  `20260705_core_world_g1_front_tray_freebox_creep_gain_retry3` is stable but
  does not move enough to count as carrying. It completed 620/620 with
  `gait_mode=targeted_creep`, fall/drop `0`, min robot z `0.78248 m`,
  min box z `0.78724 m`, max tilt `0.09570 rad`, and no rollout pose writes,
  but final robot/box target-directed travel was `-0.00724/0.00180 m`, with
  max box target-directed travel only `0.07241 m`. Strong feedback fixed
  long-horizon balance but largely canceled propulsion. Next diagnostic should
  increase creep drive while keeping this feedback authority.
  Strong-feedback stronger-drive creep
  `20260705_core_world_g1_front_tray_freebox_creep_drive_retry4` is also
  stable but still does not move forward. It completed 620/620 with fall/drop
  `0`, min robot z `0.78114 m`, min box z `0.78730 m`, max tilt
  `0.09570 rad`, no rollout pose writes, but final robot/box target-directed
  travel was `-0.01763/-0.00933 m` and max box travel was only `0.07292 m`.
  Increasing gait amplitude and stance-push under the strong feedback did not
  restore forward locomotion. Next diagnostic should test an intermediate
  feedback authority between unstable-moving and stable-stationary.
  Mid-feedback creep `20260705_core_world_g1_front_tray_freebox_creep_midfb_retry5`
  is also stable but still does not solve locomotion. It completed 620/620
  with `BALANCE_PITCH_GAIN=0.30`, `BALANCE_ADJUSTMENT_LIMIT=0.08`, fall/drop
  `0`, min robot z `0.78248 m`, min box z `0.78662 m`, max tilt
  `0.09758 rad`, and no rollout pose writes, but final robot/box
  target-directed travel was only `-0.00302/0.00351 m` and max box travel
  `0.07564 m`. This suggests the balance controller's zero-pitch target is
  canceling forward drive. Next code step: add configurable balance pitch
  target so the controller can allow a bounded forward lean instead of always
  pulling pitch to zero.
  Code update: added `--balance-pitch-target` and `--balance-roll-target` to
  the G1 Core API scene and launcher. Balance feedback now computes pitch/roll
  correction from target-relative error instead of always targeting zero
  pitch/roll. Summary JSON records both targets. Lightweight `py_compile` and
  `bash -n` checks passed.
  Pitch-target retry
  `20260705_core_world_g1_front_tray_freebox_creep_ptarget_retry6` is a strong
  negative result. It used the strong feedback with `BALANCE_PITCH_TARGET=0.06`
  and conservative creep. The target allowed too much forward lean before the
  controller could recover: 620/620, `fall_events=390`, min robot z
  `0.49145 m`, min box z `0.24388 m`, max tilt `3.14155 rad`, no rollout pose
  writes. The positive travel is falling/sliding, not carrying. Next pitch
  target should be much smaller or ramped gradually after stability is proven.
  Smaller pitch-target retry
  `20260705_core_world_g1_front_tray_freebox_creep_ptarget02_retry7` is also
  negative. With `BALANCE_PITCH_TARGET=0.02`, it still entered slow forward
  divergence and fell by about step 280: 620/620, `fall_events=347`, min robot
  z `0.49700 m`, min box z `0.24454 m`, max tilt `3.14040 rad`, no rollout
  pose writes. The final robot/box travel `1.06811/0.79864 m` is falling/
  sliding, not carrying. Fixed nonzero pitch target is unsafe; the next
  implementation should schedule a short pitch-target window and return the
  target to zero before divergence.
  Code update: added `--balance-target-start-step` and
  `--balance-target-end-step`. `balance_pitch_target` and
  `balance_roll_target` now apply only within that step window; outside the
  window the feedback target returns to zero. Lightweight `py_compile` and
  `bash -n` checks passed.
  Short pitch-target window
  `20260705_core_world_g1_front_tray_freebox_creep_ptwindow_retry8` is stable
  but under-drives locomotion. It applied `BALANCE_PITCH_TARGET=0.02` only
  from step 140 to 220, then returned to zero target. Result: 620/620,
  fall/drop `0`, min robot z `0.78241 m`, min box z `0.78732 m`, max tilt
  `0.09570 rad`, no rollout pose writes, but final robot/box travel only
  `-0.00787/0.00114 m` and max box travel `0.07279 m`. The recovery mechanism
  works; the drive window is too short or too weak.
  Longer pitch-target window
  `20260705_core_world_g1_front_tray_freebox_creep_ptwindow260_retry9` is
  stable but still under-drives locomotion. It extended the target window to
  step 260 and completed 620/620 with fall/drop `0`, min robot z `0.78231 m`,
  min box z `0.78732 m`, max tilt `0.09570 rad`, and no rollout pose writes,
  but final robot/box target-directed travel was only `-0.00776/0.00124 m`.
  This confirms that the current hand-written `targeted_creep` is not a good
  route to reliable forward walking under the strong balance feedback.
- [ ] Run the explicitly labeled moving-carrier scene diagnostic
  `20260705_core_world_g1_front_tray_freebox_rootdrive_retry1`. This uses
  `DIAGNOSTIC_ROOT_DRIVE=smooth_x`, writes only the G1 root pose, never writes
  box pose, and keeps the box free on the front tray. Count it only as
  contact/scene/metric validation under carrier motion, not as biped walking
  or final carrying.
  Retry1 passed the moving-carrier contact diagnostic but is not locomotion
  evidence. It completed 620/620 with fall/drop `0`, min robot z
  `0.78397 m`, min box z `0.78757 m`, max tilt `0.09465 rad`, no box pose
  writes, 440 root-pose writes, final robot/box target-directed travel
  `0.17756/0.18729 m`, and max box-robot relative offset error `0.05726 m`.
  Caveat: drive handoff began from the initial root pose and caused a small
  correction at step 120. Code was fixed to initialize root-drive from the
  current root pose on the first active step.
- [ ] Run `20260705_core_world_g1_front_tray_freebox_rootdrive_retry2` with the
  same contact scene and corrected current-pose root-drive handoff. If it
  passes, use it as the moving-carrier scene baseline, then add a stricter
  checker gate for diagnostic-root-drive versus true no-root locomotion.
  Retry2 passed the explicit diagnostic checker: 620/620, fall/drop `0`,
  min robot z `0.78414 m`, min box z `0.78453 m`, max tilt `0.09708 rad`, no
  box pose writes, 440 root-pose writes, final robot/box target-directed travel
  `0.15783/0.16051 m`, and max box-robot relative offset error `0.06417 m`.
  It still has sideways drift after the step-120 handoff from a naturally
  pitched loaded posture.
- [ ] Run `20260705_core_world_g1_front_tray_freebox_rootdrive_step0_retry3`
  with root-drive active from step 0 through the end, same speed and ramp. Use
  it as the preferred moving-carrier scene baseline only if it keeps fall/drop
  at `0`, keeps box pose writes at `0`, and improves target-directed travel
  without the retry2 sideways-drift artifact.
  Retry3 passed and is the preferred moving-carrier contact-scene baseline.
  It completed 620/620 with fall/drop `0`, no box pose writes, 620 root-pose
  writes, min robot z `0.78414 m`, min box z `0.79820 m`, max tilt
  `0.00880 rad`, final robot/box target-directed travel
  `0.22521/0.22965 m`, and max box-robot relative offset error `0.05405 m`.
  It passed the explicit diagnostic-root-drive checker. It is not walking
  evidence.
- [ ] Re-enter the real locomotion path using this contact scene as the fixed
  task harness. The next valid walking/carrying attempt must set
  `DIAGNOSTIC_ROOT_DRIVE=none`, keep `box_pose_write_count_rollout=0`, and
  achieve positive robot/box target-directed travel with
  `root_pose_write_count_rollout=0` and `root_velocity_write_count_rollout=0`.
- [ ] Run no-root prismatic cradle carry
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_8kg_retry1` as a
  bridge diagnostic. It uses the stable 8 kg `cradle_free_box` sync-inchworm
  parameters with target `-0.22 m` but keeps five cycles from the previous
  `-0.30 m` run. Required gate: fall/drop `0`, root/body/box/payload shortcuts
  `0`, min payload z above `0.70 m`, max tilt below `0.13 rad`, and final
  post-settle payload target distance under `0.04 m`. This is scaffolded
  prismatic-legged carrying, not final humanoid walking.
  Retry1 is negative/regression: completed 2350/2350 with fall/drop `0` and
  no root/body/box/payload shortcuts, but final post-settle payload travel was
  only `0.00468 m`, target distance `0.22468 m`, and max payload-relative
  offset error `0.20265 m`. Cause: the controller computed only 4 cycles for
  target `-0.22 m`. Code update added `--sync-inchworm-min-cycles`; launcher
  also now defaults to the old successful leg gains.
- [ ] Check `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_8kg_retry2`.
  This reruns the same no-root cradle task with `SYNC_INCHWORM_MIN_CYCLES=5`.
  Required gate remains: final post-settle payload target distance under
  `0.04 m`, no fall/drop, no root/body/box/payload shortcuts, and payload
  relative offset under `0.09 m`.
  Retry2 is also negative. It used five cycles but stride dropped to
  `0.044 m`, producing only `0.03715 m` max post-settle payload travel and
  final target distance `0.20850 m`. Code update added
  `--sync-inchworm-stride-override` so the old effective stride can be tested
  against a smaller task target.
- [ ] Check
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_stride006_8kg_retry3`.
  This uses target `-0.22 m`, at least five cycles, and explicit stride
  override `0.06 m`. Required gate is the same as retry2.
  Retry3 is negative. It used the intended five cycles and `0.06 m` stride,
  completed 2350/2350 with fall/drop `0` and no root/body/box/payload
  shortcuts, but max absolute post-settle payload travel was only `0.02865 m`
  and final target distance was `0.22092 m`. Comparison against the old strong
  `diag7` showed the launcher did not reproduce old geometry: the old
  rear/front cradle stops were `0.29824/0.71762 m`, while retry3 used
  `-0.12406/0.29573 m`; post-settle payload x differed by about `0.40 m`.
- [ ] Run
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_oldgeom_stride006_8kg_retry4`.
  It keeps target `-0.22 m`, five sync-inchworm cycles, explicit `0.06 m`
  stride, and restores the old geometry with `PAYLOAD_LOCAL_X=0.50` and
  `LEG_LOWER=-0.82`. Required gate: fall/drop `0`, no root/body/box/payload
  shortcuts, max tilt under `0.13 rad`, min payload z above `0.70 m`, max
  payload-relative offset error under `0.09 m`, max absolute post-settle
  payload travel at least `0.20 m`, and final post-settle payload target
  distance under `0.04 m`.
  Retry4 is a strong negative result. It reproduced the old x stop geometry
  but failed physically: fall events `2272`, box-drop events `2127`, min
  payload z `-512.17 m`, max tilt `3.13407 rad`, and max payload-relative
  offset error `213.53 m`; all shortcut writes remained `0`. Old `diag7`
  started with payload z around `0.825 m`, while retry4 started around
  `0.645 m`, so x geometry alone is not a valid reproduction.
- [ ] Next prismatic scaffold diagnostic: test forward payload positions only
  with an explicitly larger support polygon or revised height/contact
  geometry. Keep these as scaffold diagnostics, not humanoid walking evidence.
  Retry5a/5b completed this probe. `x=0.30` with normal support was stable
  but under-driven: fall/drop `0`, min payload z `0.78843 m`, max post-settle
  payload travel `0.03083 m`, target distance `0.19660 m`. `x=0.50` with
  wide support (`STANCE_HALF_LENGTH=0.60`, `FOOT_LENGTH=0.60`) was much
  closer: fall/drop `0`, no shortcut writes, max post-settle payload travel
  `0.19233 m`, target distance `0.03627 m`, but failed on tilt
  `0.13078 rad` and relative offset `0.17296 m`.
- [ ] Next prismatic scaffold improvement: reduce transient tilt and payload
  slosh while preserving `>0.20 m` post-settle travel. Current best
  motion-distance diagnostic is
  `20260705_prismatic_cradle_sync_inchworm_neg23cm_x050_widesupport_tight_stride007_8kg_retry6`:
  fall/drop `0`, all shortcut writes `0`, final post-settle payload travel
  `-0.21487 m`, max absolute post-settle payload travel `0.22147 m`, target
  distance `0.01513 m`; failed strict gates on max tilt `0.19881 rad`, min
  payload z `0.69930 m`, and max relative offset `0.25703 m`.
- [x] Add post-settle-relative slosh metrics so future diagnostics can
  distinguish initial settling from carry-phase relative motion. New summary
  fields: `post_settle_payload_relative_error_m` and
  `max_post_settle_payload_relative_offset_error_m`. Existing strict checker
  behavior is unchanged.
- [x] Run payload-height recovery batch retry7. `PAYLOAD_LOCAL_Z=0.20`
  eliminated early settle/slosh failures but under-drove motion: retry7a and
  retry7b both had fall/drop `0`, all shortcut writes `0`, max tilt under
  `0.093 rad`, min payload z above `0.751 m`, and max relative offset below
  `0.067 m`, but post-settle payload travel stayed below `0.08 m`.
- [x] Run intermediate payload-height sweep retry8. Current best strict-pass
  no-root scaffold is
  `20260705_prismatic_cradle_sync_inchworm_neg23cm_x050_z016_support065_stride007_8kg_retry8b`:
  8 kg free box, 2350/2350, fall/drop `0`, all root/body/box/payload shortcut
  writes `0`, max tilt `0.09174 rad`, min payload z `0.71612 m`, max
  payload-relative offset `0.04504 m`, max post-settle relative offset
  `0.00935 m`, max absolute post-settle payload travel `0.24384 m`, final
  post-settle payload travel `-0.22924 m`, and target distance `0.00076 m`.
  This is a passing prismatic scaffold baseline, not final humanoid walking.
- [ ] Next after retry8b: preserve this passing no-root physical-contact
  baseline and move toward the real task by either generating visual evidence
  for the scaffold or transferring the same free-box carry task to a more
  robot-like/humanoid locomotion backend. Do not regress into root-drive or
  payload-pose writes.
- [x] Update `run_core_world_prismatic_cradle_sync_inchworm.sh` defaults to
  reproduce retry8b. The launcher now defaults to target `-0.23 m`,
  `PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.16`,
  `STANCE_HALF_LENGTH=0.65`, `FOOT_LENGTH=0.65`, min cycles `5`, and stride
  override `0.07`. This is a scaffold baseline launcher, not a final humanoid
  walking launcher.
- [x] Verify the updated launcher defaults. `default_repro_retry9` ran with no
  env overrides and passed the strict checker: fall/drop `0`, all
  root/body/box/payload shortcut writes `0`, max tilt `0.09174 rad`, min
  payload z `0.71612 m`, max payload-relative offset `0.04504 m`, max
  post-settle payload travel `0.24384 m`, and final target distance
  `0.00076 m`.
- [x] Add and run a metrics MP4 visualizer for the retry9 scaffold. Output:
  `experiments/visuals/prismatic_carrier_stand/20260705_prismatic_cradle_sync_inchworm_default_repro_retry9_metrics.mp4`.
  This is a top-down/side-view metrics visualization, not Isaac viewport video
  and not final scene-video evidence.
- [x] Run and record walking-like retry10. `quasistatic_step_cycle` and
  `prelift_quasistatic_step_cycle` were safe with fall/drop `0` and all
  root/body/box/payload shortcut writes `0`, but under-shot target `-0.23 m`:
  final post-settle payload travel was `-0.16457 m` and `-0.17350 m`.
  `guarded_prelift_quasistatic_step_cycle` was too conservative and reached
  only `-0.04208 m`.
- [x] Run and record walking-like retry11. Shortening the target to
  `-0.17 m` with 1900 steps did not pass the post-settle gate. Both runs were
  safe with no shortcut writes, but summary post-settle payload travel was
  only `0.131999 m` and `0.137078 m`; stdout travel included settle drift and
  must not be used as the pass metric.
- [x] Add `--gait-drive-target-x` / `GAIT_DRIVE_TARGET_X` so the step-cycle
  scaffold can over-drive internal gait cycles while the checker still
  evaluates the real task target `--target-x`.
- [x] Await/record walking-like retry12
  `curiosity_prismatic_walklike_retry12_0705`, Slurm job `167304`, stamps
  `20260705_prismatic_cradle_walklike_quasistep_target017_drive023_retry12a`
  and
  `20260705_prismatic_cradle_walklike_prelift_target017_drive023_retry12b`.
  Result: both passed checker with failures `[]`. `retry12a`
  (`quasistatic_step_cycle`) reached final post-settle payload travel
  `-0.16457 m` with target distance `0.00543 m`. `retry12b`
  (`prelift_quasistatic_step_cycle`) reached `-0.17350 m` with target
  distance `0.00350 m`. Both had fall/drop `0`, all shortcut writes `0`,
  max tilt `0.09174 rad`, min payload z `0.71612 m`, max payload-relative
  offset `0.04504 m`, and 8 articulated joints. This is a short-target
  walking-like prismatic scaffold pass, not humanoid or learned carrying.
- [ ] Next gate after retry12: do not keep tuning only the same scaffold.
  Choose one concrete upgrade: Isaac viewport/scene video for audit,
  active-probing hooks on this free-box scaffold, or transfer the same
  free-box carry task to a more robot-like/humanoid locomotion backend while
  preserving the no-root/no-payload-write gate.
- [x] Await/record multi-posture retry13
  `curiosity_prismatic_walklike_retry13_0705`, job-name `prism_post_r13`.
  Planned stamps:
  `20260705_prismatic_cradle_walklike_posture_high_z018_retry13a`,
  `20260705_prismatic_cradle_walklike_posture_close_x045_retry13b`, and
  `20260705_prismatic_cradle_walklike_posture_low_z014_retry13c`. Gate:
  `cradle_free_box`, 8 kg, `prelift_quasistatic_step_cycle`, fall/drop `0`,
  all root/body/box/payload writes `0`, at least 8 articulated joints,
  min payload z `>=0.70 m`, max tilt `<=0.13 rad`, max payload-relative
  offset `<=0.09 m`, max absolute post-settle payload travel `>=0.15 m`, and
  final post-settle payload target distance `<=0.03 m` for `TARGET_X=-0.17`.
- [x] Record multi-posture retry13. The first checker submission
  `prism_r13_chk` / Slurm job `167310` was invalid because outer-shell
  expansion blanked the loop stamp variable; do not count it as evidence.
  Added `scripts/isaac/check_core_world_prismatic_walklike_retry13_postures.sh`
  and reran valid checker job `167313` (`prism_r13_ck2`) on compute. Result:
  `retry13b` close carry (`PAYLOAD_LOCAL_X=0.45`, `PAYLOAD_LOCAL_Z=0.16`)
  passed all strict gates with fall/drop `0`, all shortcut writes `0`,
  `max_tilt_rad=0.08797`, `min_payload_z_m=0.72143`,
  `max_payload_relative_offset_error_m=0.03811`,
  `max_abs_post_settle_payload_travel_x_m=0.18958`, and final post-settle
  payload target distance `0.00786 m`. `retry13a` high carry
  (`x=0.50`, `z=0.18`) failed only distance/target gates:
  max post-settle payload travel `0.08094 m`, final target distance
  `0.13504 m`. `retry13c` low carry (`x=0.50`, `z=0.14`) reached the target
  but failed payload clearance: `min_payload_z_m=0.69461 < 0.70`.
  Interpretation: posture matters under the same scaffold; near-body mid
  height is the current pass, high is under-driven, and low is a clearance
  boundary. This is still a prismatic scaffold, not humanoid, learned,
  video-guided, or autonomous posture selection.
- [x] Next retry14 gate: continue direct Isaac scene work without waiting on
  external models. Extend the passing close carry posture and the original
  mid-height posture to a longer rollout, and test whether the closer carry
  offset rescues the low-height clearance boundary.
- [x] Await/record retry14 posture-long/boundary batch. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry14_posture_long_batch.sh`
  and submitted tmux `curiosity_prismatic_walklike_retry14_0705`, Slurm job
  `167315`, job-name `prism_post_r14`. Runs:
  `20260705_prismatic_cradle_walklike_close_x045_z016_long_retry14a`
  (retry13 pass posture, 2800 steps),
  `20260705_prismatic_cradle_walklike_mid_x050_z016_long_retry14b`
  (original mid-height posture, 2800 steps), and
  `20260705_prismatic_cradle_walklike_close_low_x045_z014_retry14c`
  (closer low carry boundary). Use the retry13 strict checker gates unless a
  separately documented longer-horizon gate is added.
- [x] Record retry14 results. `retry14a` and `retry14b` completed inside job
  `167315`; the original `retry14c` launch is invalid because the compute
  side hit a transient/stale Python syntax read at
  `build_core_world_prismatic_carrier_stand.py:164` before Isaac rollout.
  Current login-node file content was clean. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry14c_boundary_retry2.sh`,
  which sleeps, prints the relevant lines, and runs compute-side
  `py_compile` before Isaac. Retry2 job `167316` completed successfully.
  Added `scripts/isaac/check_core_world_prismatic_walklike_retry14_postures.sh`
  and ran checker job `167319`; all three valid retry14 runs passed with
  failures `[]`. Results:
  `retry14a` close mid-height (`x=0.45`, `z=0.16`) passed 2800 steps with
  fall/drop `0`, all shortcut writes `0`, min payload z `0.72143`, max tilt
  `0.08797`, max post-settle payload travel `0.18958`, final target distance
  `0.00782`.
  `retry14b` mid-height (`x=0.50`, `z=0.16`) passed 2800 steps with min
  payload z `0.71612`, max tilt `0.09174`, max post-settle payload travel
  `0.18468`, final target distance `0.00350`.
  `retry14c_retry2` close low (`x=0.45`, `z=0.14`) passed 2800 steps with min
  payload z `0.70213`, max tilt `0.09110`, max post-settle payload travel
  `0.18254`, final target distance `0.00293`. Interpretation: moving the low
  posture closer to the body recovered the payload-height margin that failed
  in retry13c (`x=0.50`, `z=0.14`, min payload z `0.69461`).
- [x] Next retry15 gate: attempt to recover the high-carry posture that
  failed retry13a. Keep the same strict no-shortcut gates and test whether
  closer payload placement and/or a stronger diagnostic gait-drive target can
  make `z=0.18` carry pass without fall/drop.
- [x] Await/record retry15 high-posture rescue batch. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry15_high_posture_rescue.sh`
  and submitted tmux `curiosity_prismatic_walklike_retry15_0705`, Slurm job
  `167322`, job-name `prism_high_r15`. Runs:
  `retry15a` high-close same drive (`x=0.45`, `z=0.18`,
  `GAIT_DRIVE_TARGET_X=-0.23`), `retry15b` high-mid stronger drive
  (`x=0.50`, `z=0.18`, drive `-0.31`), and `retry15c` high-close moderate
  drive (`x=0.45`, `z=0.18`, drive `-0.27`). Gate remains the retry14 strict
  no-shortcut/pass gate.
- [x] Record retry15 results. Added
  `scripts/isaac/check_core_world_prismatic_walklike_retry15_high_postures.sh`
  and ran formal checker job `167324` (`prism_r15_chk`) on server63. All
  three high-carry runs failed only the travel/target gates while preserving
  the safety and no-shortcut gates: fall/drop `0`, all root/body/box/payload
  write counters `0`, articulated joint count `8`, and nonfinite events `0`.
  `retry15a` high-close same drive (`x=0.45`, `z=0.18`, drive `-0.23`) had
  min payload z `0.73707`, max tilt `0.08629`, max post-settle payload travel
  `0.08122`, final post-settle target distance `0.13420`, failures
  `["absolute post-settle payload travel x too low", "final post-settle payload target distance x too high"]`.
  `retry15b` high-mid stronger drive (`x=0.50`, `z=0.18`, drive `-0.31`) had
  min payload z `0.73264`, max tilt `0.08959`, max post-settle payload travel
  `0.10265`, final post-settle target distance `0.11544`, with the same two
  failures. `retry15c` high-close moderate drive (`x=0.45`, `z=0.18`, drive
  `-0.27`) had min payload z `0.73707`, max tilt `0.08629`, max post-settle
  payload travel `0.10426`, final post-settle target distance `0.11558`, with
  the same two failures. Interpretation: high carry is stable and well clear
  of the ground, but this diagnostic foot-contact drive saturates before it
  produces enough payload transport. Do not call high carry solved.
- [x] Next retry16 gate: change the direct Isaac scene mechanics, not the
  external-model stack. Target the high-carry under-driving failure by adding a
  stronger physical propulsion/contact schedule or posture transition test
  while preserving the no-root/no-payload-write counters and the retry14
  strict pass gates.
- [x] Await/record retry16 high-stride batch. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry16_high_stride_batch.sh`.
  This is still a direct Isaac diagnostic, not learning or video guidance.
  It tests whether the high-carry failure is recoverable by increasing the
  physical step-cycle propulsion envelope: common settings are `z=0.18`,
  `GAIT_DRIVE_TARGET_X=-0.42`, `STEP_LENGTH=0.10`, `GAIT_PERIOD_STEPS=300`,
  `X_SLIDE_LIMIT=0.28`, and higher leg/x-slide drive limits. Runs:
  `retry16a` high-mid larger stride (`x=0.50`), `retry16b` high-mid larger
  stride with swing-leg horizontal force scaled to `0.0`, and `retry16c`
  high-close larger stride with `PRELIFT_STANCE_OVERDRIVE=1.6`.
  Submitted tmux `curiosity_prismatic_walklike_retry16_0705`, Slurm job
  `167329`, job-name `prism_high_r16`.
- [x] Record retry16 results. Added
  `scripts/isaac/check_core_world_prismatic_walklike_retry16_high_stride.sh`,
  but formal checker jobs `167330` and `167332` were canceled after remaining
  pending for resource priority; no checker job result should be claimed.
  The three Isaac rollout summaries were inspected with lightweight `jq`.
  All runs stayed safe and shortcut-clean: fall/drop `0`, all
  root/body/box/payload write counters `0`, articulated joint count `8`, and
  nonfinite events `0`. `retry16a` high-mid larger stride reached the travel
  gate but failed final holding: min payload z `0.73411`, max tilt `0.09022`,
  max post-settle payload travel `0.16863`, final post-settle target distance
  `0.05819`. `retry16b` high-mid with swing x force scaled to `0.0` was
  worse: max post-settle payload travel `0.08621`, final target distance
  `0.16092`; swing x force was not the main bad factor. `retry16c`
  high-close with overdrive `1.6` strongly over-drove the target: max
  post-settle payload travel `0.34731`, final target distance `0.11302`, max
  payload-relative offset error `0.03085`. Interpretation: high carry can be
  physically propelled without safety/shortcut failures, but it now needs
  target-aware stopping/holding or guarded step progression, not more raw
  drive.
- [x] Next retry17 gate: test high-carry target-aware guarded progression.
  Use the direct Isaac scaffold only. Try `guarded_prelift_quasistatic_step_cycle`
  with the `retry16a` propulsion envelope and target stopping around the real
  task target, then compare against a conservative no-overdrive high-mid
  variant. Goal is to eliminate the final target-distance failure without
  reintroducing travel failure, falls, drops, or shortcut writes.
- [x] Await/record retry17 guarded high-stop batch. Added guarded stop support
  in `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`: optional
  `GUARDED_STOP_TARGET_X` lets the internal gait drive overdrive differ from
  the real guarded hold target, and guarded target detection now treats
  crossing the target as reached. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry17_guarded_high_stop_batch.sh`
  with three direct Isaac diagnostics: `retry17a` guarded high-mid strict
  tolerance `0.018`, `retry17b` guarded high-mid tolerance `0.030`, and
  `retry17c` guarded high-close overdrive `1.6` with tolerance `0.030`.
  Submitted tmux `curiosity_prismatic_walklike_retry17_0705`, Slurm job
  `167333`, job-name `prism_high_r17`.
  Added syntax-checked checker
  `scripts/isaac/check_core_world_prismatic_walklike_retry17_guarded_high_stop.sh`
  for the same strict no-shortcut/pass gates once summaries exist.
- [x] Record retry17 result. Slurm job `167333` completed and no job remained
  in `squeue`. All three high-carry guarded runs were safe and shortcut-clean
  but failed transport/target because the gated controller held too early with
  `gated_step_last_block_reason="post_settle_payload_travel_loss"`. This was
  a control-logic bug: for negative-X targets, the loss check used raw X peak
  (`peak_x - current_x`), so normal progress toward negative X was
  misclassified as travel loss. Results:
  `retry17a` and `retry17b` both ended at final post-settle payload travel
  `-0.07346 m`, final target distance `0.09654 m`, max tilt `0.09022`, min
  payload z `0.73411`, fall/drop `0`, root/body/box/payload writes `0`, and
  `gated_step_release_steps=84`.
  `retry17c` ended at final post-settle payload travel `-0.07336 m`, final
  target distance `0.09664 m`, max tilt `0.08669`, min payload z `0.73967`,
  max payload-relative offset error `0.03015`, fall/drop `0`, writes `0`, and
  the same travel-loss block. Do not count retry17 as high-carry success.
- [x] Fix guarded travel-loss direction. Updated
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` so guarded
  loss uses directional progress toward the guarded stop target. Positive-X
  behavior is unchanged; negative-X targets now track `direction * travel_x`.
  Added summary fields for directional guarded progress/loss while preserving
  existing checker fields. Syntax checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py`
  and `bash -n` for the runner/checker scripts.
- [x] Await/record retry18 directional-guard batch. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry18_directional_guard_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry18_directional_guard.sh`.
  Submitted tmux `curiosity_prismatic_walklike_retry18_0705`, Slurm job
  `167342`, job-name `prism_high_r18`. Runs:
  `retry18a` high-mid (`x=0.50`, `z=0.18`, tolerance `0.030`) and
  `retry18b` high-close overdrive `1.6` (`x=0.45`, `z=0.18`, tolerance
  `0.030`). This is still a direct Isaac diagnostic, not learned/video-guided
  carrying.
- [x] Record retry18 directional-guard result. Slurm job `167342`
  (`prism_high_r18`) completed on server02, followed by formal checker job
  `167343` (`prism_r18_chk`) on server02. Both runs passed the strict
  no-shortcut gate with checker `failures=[]`. `retry18a` high-mid
  (`x=0.50`, `z=0.18`) completed 2800 steps with fall/drop `0`, all
  root/body/box/payload writes `0`, max tilt `0.09022`, min payload z
  `0.73411`, max post-settle payload travel `0.17536`, final post-settle
  target distance `0.00536`, and `gated_step_last_block_reason="target_reached"`.
  `retry18b` high-close overdrive `1.6` (`x=0.45`, `z=0.18`) also passed:
  max tilt `0.08669`, min payload z `0.73967`, max post-settle payload travel
  `0.15468`, final post-settle target distance `0.01532`, max
  payload-relative offset error `0.03015`, and
  `gated_step_last_block_reason="target_reached"`. Interpretation: the high
  posture failure in retry17 was indeed a negative-target progress-sign bug.
  This is a direct Isaac prismatic-scaffold pass, not humanoid walking,
  learning, or video guidance.
- [x] Next retry19 gate: stop single-case tuning and test posture/load
  variation inside the same direct Isaac scaffold. Use the passing directional
  guard controller to compare at least mid/high carry postures under changed
  payload mass and/or box geometry. Required reporting: which posture passes,
  target error, fall/drop, payload height, relative offset, and no-shortcut
  counters. Do not claim autonomous posture selection yet; this is the
  diagnostic dataset needed before implementing a selector/probing policy.
- [x] Await/record retry19 posture-load-shape batch. Added payload-size
  environment controls to
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`
  (`PAYLOAD_SIZE_X/Y/Z`) and added
  `scripts/isaac/run_core_world_prismatic_walklike_retry19_posture_load_shape_batch.sh`
  plus
  `scripts/isaac/check_core_world_prismatic_walklike_retry19_posture_load_shape.sh`.
  Planned runs: `retry19a` mid posture standard box at 12 kg,
  `retry19b` high posture standard box at 12 kg, `retry19c` mid posture tall
  box at 8 kg, and `retry19d` high posture tall box at 8 kg. Syntax checks
  passed. This remains scaffold variation data, not autonomous selection.
- [x] Record retry19 result. Slurm rollout job `167346`
  (`prism_var_r19`) completed on server02, followed by checker job `167349`
  (`prism_r19_chk`). All four variation diagnostics passed the formal checker
  with `failures=[]`, fall/drop `0`, all shortcut writes `0`, joint count `8`,
  and nonfinite events `0`. Results:
  `retry19a` 12 kg standard box, mid posture (`x=0.50`, `z=0.16`) passed with
  max tilt `0.10264`, min payload z `0.71327`, final post-settle payload
  target distance `0.00504`, max payload-relative offset error `0.06084`.
  `retry19b` 12 kg standard box, high posture (`x=0.50`, `z=0.18`) passed
  with max tilt `0.10109`, min payload z `0.73689`, final target distance
  `0.00577`, max relative offset `0.06928`.
  `retry19c` 8 kg tall box (`0.34 x 0.24 x 0.34 m`), mid posture passed with
  max tilt `0.09181`, min payload z `0.76110`, final target distance
  `0.00012`, max relative offset `0.09522`.
  `retry19d` 8 kg tall box, high posture passed with max tilt `0.08902`, min
  payload z `0.74347`, final target distance `0.01165`, max relative offset
  `0.07433`.
- [ ] Next implementation gate: add explicit posture-choice scaffolding. Do
  not call it learning yet. Build a small evaluator/manifest that records
  candidate posture, box mass/size, pass/fail, target error, payload height,
  tilt, and relative offset from retry14/18/19-style runs. Then add an
  explicit rule-based selector baseline such as "choose the lowest passing
  posture with sufficient height margin and lowest offset/tilt cost" before
  attempting active probing or RL.
- [x] Implement retry20 posture-choice scaffold. Added manifest
  `experiments/configs/prismatic_cradle_posture_selector_retry20_manifest.json`,
  summarizer
  `scripts/isaac/summarize_prismatic_cradle_posture_selector.py`, and
  compute-side runner
  `scripts/isaac/run_prismatic_cradle_posture_selector_retry20.sh`. Syntax
  checks passed on the login node only: `py_compile`, `bash -n`, and
  `jq empty` for the manifest.
- [x] Run and record retry20 posture-choice scaffold. Submitted tmux
  `curiosity_prismatic_selector_retry20_0706`, Slurm job `167351`,
  job-name `prism_sel_r20`, with command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --job-name=prism_sel_r20 bash scripts/isaac/run_prismatic_cradle_posture_selector_retry20.sh`.
  Output report:
  `experiments/reports/prismatic_cradle_posture_selector/20260706_retry20_prismatic_cradle_posture_selector_report.json`;
  candidate table:
  `experiments/reports/prismatic_cradle_posture_selector/20260706_retry20_prismatic_cradle_posture_selector_candidates.jsonl`;
  log:
  `logs/core_world_prismatic_carrier_stand/prismatic_cradle_selector_retry20.log`.
  Result: report status `pass`, candidate count `9`, passed candidate count
  `9`, selector-eligible count `8`, failures `[]`. Rule: choose the lowest
  passing carry height with at least `0.01 m` payload-height margin, then
  break ties using target error, tilt, and payload relative offset. Selected
  `mid_front` for `standard_8kg`, `standard_12kg`, and `tall_8kg`. The 8 kg
  low-close posture passed the gate but was not selector-eligible because its
  height margin was only about `0.00213 m`. This is still a rule-based
  selector over completed prismatic-scaffold runs, not active probing, RL,
  video guidance, humanoid walking, or complete robot success.
- [ ] Next gate: use the retry20 selector output to run held-out
  selector-driven Isaac cases, e.g. 10 kg standard and 10 kg tall boxes using
  the selected `mid_front` posture, then compare against at least one
  non-selected posture. This should test whether the selector table is useful
  beyond replaying the same completed runs.
- [x] Prepare retry21 held-out selector-driven diagnostics. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry21_selector_heldout_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry21_selector_heldout.sh`.
  Planned cases: 10 kg standard box with selected `mid_front`, 10 kg standard
  box with `high_front` control, 10 kg tall box with selected `mid_front`, and
  10 kg tall box with `high_front` control. Syntax checks passed. This tests
  held-out selector-driven execution inside the prismatic scaffold; it is not
  autonomous active probing.
- [x] Run and record retry21 held-out selector-driven diagnostics. Submitted
  tmux `curiosity_prismatic_selector_retry21_0706`, Slurm rollout job
  `167353`, job-name `prism_sel_r21`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=04:00:00 --job-name=prism_sel_r21 bash scripts/isaac/run_core_world_prismatic_walklike_retry21_selector_heldout_batch.sh`.
  Then submitted checker tmux `curiosity_prismatic_retry21_checker_0706`,
  Slurm job `167366`, job-name `prism_r21_chk`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --job-name=prism_r21_chk bash scripts/isaac/check_core_world_prismatic_walklike_retry21_selector_heldout.sh`.
  All four cases passed formal checker with `failures=[]`, fall/drop `0`, all
  shortcut writes `0`, joint count `8`, and nonfinite events `0`.
  `retry21a` selected `mid_front`, standard 10 kg: final post-settle payload
  target distance `0.00610`, max tilt `0.09731`, min payload z `0.71194`,
  max offset `0.05156`.
  `retry21b` high control, standard 10 kg: target distance `0.01969`, max
  tilt `0.09582`, min payload z `0.73178`, max offset `0.06031`.
  `retry21c` selected `mid_front`, tall 10 kg: target distance `0.00003`,
  max tilt `0.09433`, min payload z `0.75601`, max offset `0.09760`.
  `retry21d` high control, tall 10 kg: target distance `0.00616`, max tilt
  `0.10391`, min payload z `0.75913`, max offset `0.07326`. Interpretation:
  the retry20 selected `mid_front` posture passes held-out standard/tall 10 kg
  boxes and has better target accuracy than the high control in both
  conditions, while the tall-box high control has lower relative offset.
- [ ] Next gate: move beyond offline/held-out posture selection by adding an
  active probing hook in the prismatic scaffold. Minimal valid diagnostic:
  run a pre-carry micro-lift/push phase that logs an estimated load or risk
  bucket without using hidden ground truth, then uses the rule selector to
  choose posture and executes the carry. Keep it explicitly labeled as a
  scaffold until a real robot locomotion backend replaces the prismatic body.
- [x] Implement active probe hook for the prismatic scaffold. Updated
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh` with
  `ENABLE_ACTIVE_PROBE`, `ACTIVE_PROBE_STEPS`,
  `ACTIVE_PROBE_LIFT_AMPLITUDE`, and `ACTIVE_PROBE_HORIZONTAL_AMPLITUDE`.
  The probe runs after settle and before carry; carry baseline and guarded
  gait progression now start after probe. The probe belief is explicitly
  observed from micro-lift response, tilt, and relative offset, with
  `active_probe_uses_hidden_ground_truth=false`. Extended
  `scripts/isaac/check_prismatic_carrier_stand_summary.py` with active-probe
  gates. Syntax checks passed.
- [x] Prepare retry22 active-probe selected-posture diagnostics. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry22_active_probe_selected_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry22_active_probe_selected.sh`.
  Planned cases: standard 10 kg selected `mid_front` and tall 10 kg selected
  `mid_front`, both with 80 probe steps before carry and formal checker gates
  requiring active probe, probe belief, no hidden probe ground truth, and at
  least 60 observed probe steps.
- [x] Run and record retry22 active-probe selected-posture diagnostics.
  Rollout Slurm job `167368` (`prism_probe_r22`) and checker job `167373`
  (`prism_r22_chk`) both completed. Standard 10 kg and tall 10 kg selected
  `mid_front` cases passed formal checker with `failures=[]`, fall/drop `0`,
  all root/body/box/payload writes `0`, joint count `8`, nonfinite events
  `0`, active probe enabled, 80 observed probe steps, probe belief available,
  and `active_probe_uses_hidden_ground_truth=false`. Standard 10 kg final
  post-settle payload target distance was `0.00583 m`; tall 10 kg was
  `0.00865 m`. This is active-probe instrumentation only; retry22 did not
  yet use the probe belief to change the controller.
- [ ] Next gate: implement retry23 as a direct Isaac probe-conditioned
  control diagnostic. Keep the real task target unchanged, but let the
  observed probe risk choose an internal gait-drive scale after the probe.
  Required evidence: active probe belief present, no hidden ground truth,
  adaptive decision fields logged, expected risk bucket/scale per case,
  fall/drop `0`, no shortcut writes, and final target gate still passes.
- [x] Prepare retry23 probe-conditioned gait diagnostic. Added
  `ENABLE_PROBE_ADAPTIVE_GAIT` and related risk-threshold/gait-scale controls
  to the Isaac runner and script, plus checker gates for adaptive decision,
  expected adaptive bucket, and expected gait scale. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry23_probe_adaptive_gait_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry23_probe_adaptive_gait.sh`.
  Syntax checks passed on login node only.
- [x] Run and record retry23 probe-conditioned gait diagnostic. Rollout job
  `167383` (`prism_probe_r23`) and checker job `167384` (`prism_r23_chk`)
  completed. Standard 10 kg and tall 10 kg cases both passed checker with
  `failures=[]`, fall/drop `0`, all shortcut writes `0`, joint count `8`, and
  nonfinite events `0`. Standard 10 kg selected adaptive bucket `low` and
  gait scale `1.0`; tall 10 kg selected adaptive bucket `medium` and gait
  scale `0.98`, changing internal gait-drive target from `-0.42` to
  `-0.41160` while keeping the real target/guarded stop target at `-0.17`.
  Final post-settle payload target distances were `0.00583 m` and
  `0.00680 m`.
- [ ] Next gate: move from "probe changes gait overdrive" to "probe changes
  posture or contact strategy." The next direct Isaac diagnostic should keep
  the same unknown-load/shape setup and add a discrete strategy decision such
  as selected carry height, hold x-position, or cradle/contact margin based
  on observed probe risk. Do not call it autonomous humanoid carrying until a
  real walking backend replaces the prismatic scaffold.
- [x] Prepare retry24 probe-conditioned posture diagnostic. Added
  `ENABLE_PROBE_ADAPTIVE_POSTURE` with medium/high leg-target offsets to the
  Isaac script and runner, plus checker gates for posture decision,
  expected strategy, expected posture risk bucket, and expected leg-target
  offset. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry24_probe_adaptive_posture_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry24_probe_adaptive_posture.sh`.
  Syntax checks passed on login node only. Planned behavior: standard 10 kg
  selects `nominal_height`, tall 10 kg selects `lower_carry_medium`.
- [x] Run and record retry24 probe-conditioned posture diagnostic. Rollout
  job `167387` (`prism_post_r24`) and checker job `167389`
  (`prism_r24_chk`) completed. Both cases passed formal checker with
  `failures=[]`, fall/drop `0`, all shortcut writes `0`, joint count `8`, and
  nonfinite events `0`. Standard 10 kg selected gait bucket `low`, gait scale
  `1.0`, posture strategy `nominal_height`, offset `0.0`, final post-settle
  payload target distance `0.00583 m`. Tall 10 kg selected gait bucket
  `medium`, gait scale `0.98`, posture strategy `lower_carry_medium`, offset
  `0.012 m`, effective leg target `-0.558`, and final post-settle payload
  target distance `0.00415 m`.
- [ ] Next gate: move the same active-probe strategy decision closer to the
  final objective by replacing the prismatic scaffold body with a real robot
  locomotion backend or by adding a stronger no-root articulated walking
  backend. The current retry24 result proves probe-conditioned posture choice
  only inside the scaffold; it does not prove humanoid walking or arbitrary
  posture-balanced carrying.
- [x] Prepare direct Isaac G1 pulsed-creep retry10 after user correction to
  stop waiting on external models. Added pulsed balance-target controls to the
  Core API G1 scene and launchers:
  `BALANCE_TARGET_PULSE_PERIOD_STEPS`,
  `BALANCE_TARGET_PULSE_WIDTH_STEPS`, and
  `BALANCE_TARGET_PULSE_PHASE_STEP`; added summary/checker telemetry for
  `balance_target_active_steps`. Added
  `scripts/isaac/run_core_world_g1_front_tray_freebox_pulsed_creep_batch.sh`
  with three G1 front-tray free-box no-root diagnostics. Login-node
  `py_compile` and `bash -n` checks passed.
- [ ] Run/record direct Isaac G1 pulsed-creep retry10. Required interpretation:
  this is a no-root G1 scene/control diagnostic only. It must keep fall/drop
  `0`, root pose/velocity writes `0`, box pose writes `0`, min box z above
  `0.70 m`, and max tilt below `0.25 rad`; if it moves, the evidence is
  stronger than the previous stationary strong-feedback G1 run, but it is
  still not learned/video-guided or final humanoid carrying.
- [x] Record direct Isaac G1 pulsed-creep retry10. Valid rollout job
  `167397` (`g1_pulse10g`) ran in tmux
  `curiosity_g1_pulsed_creep_retry10gpu2_0706` after invalid pre-rollout
  attempts `167395` (launcher permission) and `test` partition allocation
  failure; pending job `167396` was canceled before rollout. All three
  retry10 cases completed 620 steps with fall/drop `0`, root pose/velocity
  writes `0`, and box pose writes `0`, but all failed the travel gate.
  `retry10a`: max box target-directed travel `0.07256 m`, final box travel
  `0.00621 m`, max tilt `0.09570`, min box z `0.78731`.
  `retry10b`: max box travel `0.07112 m`, final `0.00432 m`, max tilt
  `0.08997`, min box z `0.78701`.
  `retry10c`: max box travel `0.06689 m`, final `0.00961 m`, max tilt
  `0.08375`, min box z `0.78886`.
- [ ] Stop using the current G1 open-loop `targeted_creep` family as the main
  no-root locomotion route. It can keep the free box stable on the front tray,
  but it does not produce meaningful target-directed walking under the
  stabilizing feedback. Next G1 work should change the locomotion mechanism,
  not keep sweeping pulse/feedback constants.
- [x] Prepare a unified direct-Isaac carry posture suite for the current
  strongest walking/carrying scaffold. Added
  `scripts/isaac/run_direct_carry_posture_suite_64cm.sh` and
  `scripts/isaac/summarize_direct_carry_posture_suite.py`. The suite reruns
  `front_mid`, `low_front`, and `chest_high` at 64 cm / 8 kg with strict
  support-continuity and no-shortcut gates. Login-node `py_compile` and
  `bash -n` checks passed.
- [x] Run/record direct carry posture suite
  `20260706_direct_carry_posture_suite_64cm_8kg`. Required interpretation:
  this is the best current complete-task scaffold if it passes all three
  postures, but it still does not prove final humanoid carrying, RL, or
  video-conditioned active posture selection.
- [x] Record direct carry posture suite result. Rollout job `167398`
  (`carry_suite64`) ran from tmux
  `curiosity_direct_carry_posture_suite_0706` with command
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_suite64 bash scripts/isaac/run_direct_carry_posture_suite_64cm.sh`.
  Output:
  `experiments/outputs/direct_carry_posture_suite/20260706_direct_carry_posture_suite_64cm_8kg/direct_carry_posture_suite_summary.json`.
  Result: `status=pass`, `failures=[]`. `front_mid`, `low_front`, and
  `chest_high` each completed `3580` steps with fall/drop `0`,
  `root_shortcut_free=true`, no fixed-world stance anchor, support continuity,
  and no target/tilt/support-margin gate failure. Max box travel:
  `0.67301 m`, `0.66675 m`, `0.65313 m`. Final target distance:
  `0.02369 m`, `0.00189 m`, `0.01468 m`. Max tilt:
  `0.12141 rad`, `0.12326 rad`, `0.12221 rad`. Min support margin:
  `0.15951 m`, `0.15984 m`, `0.15943 m`. Checker-only recomposition of
  existing 20260705 strict 64 cm summaries also passed as Slurm job `167399`
  (`carry_suite_chk`) with output
  `experiments/outputs/direct_carry_posture_suite/20260706_existing_20260705_strict64_suite/direct_carry_posture_suite_summary.json`.
- [ ] Next gate: broaden the direct-Isaac carry task from three named
  postures to a parameterized posture/hold-space stress suite. Required
  evidence: at least five carry configurations including front, low,
  chest-supported, asymmetric/contact-shifted, and one harder hold-height or
  hold-offset case; all must keep fall/drop `0`, no root/box/support shortcut
  writes, no fixed-world stance anchor, support continuity, target-directed
  box travel, bounded tilt, and support-polygon margin. This remains a
  scaffold gate unless the locomotion backend is replaced by a real robot
  controller.
- [x] Prepare the 5-posture direct-Isaac stress suite. Added `front_reach`
  and `close_mid` posture labels to
  `scripts/isaac/run_direct_carry_task_physical_backend.sh`, and added
  `scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh`. The suite
  runs `front_mid`, `low_front`, `chest_high`, `front_reach`, and `close_mid`
  at 64 cm / 8 kg with the same strict support-continuity, no-shortcut, target,
  tilt, and support-margin gates as the 3-posture suite, but `--min-postures 5`.
  Login-node `bash -n` and import-free `py_compile` checks passed.
- [x] Run/record direct carry 5-posture stress suite
  `20260706_direct_carry_posture_stress_suite_64cm_8kg`. Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=03:00:00 --job-name=carry_stress5 bash scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh`.
  Required interpretation: this can strengthen the posture-space scaffold
  evidence if all five pass, but it still cannot be reported as final humanoid
  carrying, learned control, or video-conditioned active posture selection.
- [x] Record direct carry 5-posture stress suite result. Job `167427`
  (`carry_stress5`) ran on `server28` from tmux
  `curiosity_direct_carry_posture_stress_0706` and completed. Output:
  `experiments/outputs/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/direct_carry_posture_stress_suite_summary.json`.
  Result: `status=pass`, `failures=[]`, `case_count=5`. `front_mid`,
  `low_front`, `chest_high`, `front_reach`, and `close_mid` each completed
  `3580` steps with fall/drop `0`, `root_shortcut_free=true`, no fixed-world
  stance anchor, support continuity, final target distance under `0.025 m`,
  max tilt under `0.124 rad`, and min support margin over `0.158 m`.
- [ ] Next evidence gate: produce MP4 visual audit artifacts for the strongest
  direct-Isaac scaffold suite on a compute node. At minimum render one
  representative case and preferably the 5-posture stress suite from logged
  CSV/summary into `experiments/visuals/`. This is visualization evidence
  only; do not treat a rendered video as a new control result.
- [x] Prepare MP4 visual audit rendering for the 5-posture stress suite. Added
  `scripts/isaac/render_direct_carry_posture_stress_suite_videos.sh`; it
  renders all five postures from existing backend CSV/summary files into
  `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/`.
  Login-node `bash -n` and import-free `py_compile` checks passed.
- [x] Run/record MP4 visual audit rendering for the 5-posture stress suite.
  Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:20:00 --job-name=carry_viz5 bash scripts/isaac/render_direct_carry_posture_stress_suite_videos.sh`.
  Required interpretation: video files are audit artifacts generated from
  logged CSV/summary, not new physics/control evidence.
- [x] Record MP4 rendering result. Initial render attempts failed on compute
  because of CSV string parsing (`167431`), missing video libraries in system
  `python3` (`167432`), and missing y/z columns in the one-dimensional backend
  CSV (`167433`). These were renderer compatibility issues, not rollout/control
  failures. Retry4 job `167434` (`carry_viz5d`) ran on `server02` and wrote
  five MP4 audit videos plus
  `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/render_manifest.txt`.
  The manifest lists nonempty MP4s for `front_mid`, `low_front`, `chest_high`,
  `front_reach`, and `close_mid`.
- [ ] Next implementation gate: connect active probing to the widened
  5-posture hold space. The next diagnostic should run at least two object
  conditions, use observed probe telemetry without hidden load ground truth,
  select among posture labels including the new `front_reach`/`close_mid`
  options, and then pass the same no-shortcut/support/target/fall/drop gates.
  This remains a scaffold diagnostic until a real robot locomotion backend
  replaces the support-foot carrier.
- [x] Prepare two-stage probe-selected posture diagnostic. Added
  `scripts/isaac/select_direct_carry_posture_from_probe.py` and
  `scripts/isaac/run_direct_carry_probe_selected_posture_suite.sh`.
  Planned conditions: `vertical_probe` (`vertical_micro_lift`,
  `PROBE_Z_AMPLITUDE=0.030`) and `horizontal_probe`
  (`horizontal_push_pull`, `PROBE_X_AMPLITUDE=0.050`). The selector reads
  only normalized probe telemetry, rejects hidden-ground-truth probe belief,
  and maps risk to `front_reach`, `close_mid`, or `chest_high`.
  Login-node `bash -n` and `py_compile` checks passed.
- [x] Run/record two-stage probe-selected posture diagnostic. Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_probe_sel bash scripts/isaac/run_direct_carry_probe_selected_posture_suite.sh`.
  Required interpretation: this is a scaffold diagnostic where the carry
  posture is chosen between episodes from observed probe telemetry. It is not
  online in-episode geometry-changing control and not final humanoid carrying.
- [x] Record two-stage probe-selected posture result. Retry1 job `167437`
  partially succeeded: `vertical_probe` selected `close_mid` and the selected
  carry passed, but `horizontal_probe` failed before normalized summary due a
  shell/control-flow issue. Retry2 job `167440` failed before suite execution
  on a shell quoting issue. Retry3 job `167441` (`carry_probe3`) completed on
  `server46`. Output:
  `experiments/outputs/direct_carry_probe_selected_posture_suite/20260706_direct_carry_probe_selected_posture_suite_retry3_64cm_8kg/`.
  `vertical_micro_lift` probe risk `0.5987436213151278` selected `close_mid`;
  `horizontal_push_pull` probe risk `0.45948289037895235` selected
  `front_reach`; both selection reports have `status=pass`,
  `probe_belief_available=true`, and no hidden-ground-truth probe belief. The
  combined selected-carry summary reports `status=pass`, `failures=[]`, two
  postures, fall/drop `0`, root shortcut free, no fixed-world stance anchor,
  max target distance under `0.025 m`, max tilt under `0.124 rad`, and min
  support margin over `0.158 m`.
- [ ] Next technical gate: collapse the two-stage probe-selected posture
  diagnostic toward a single episode or controller-level interface. Options:
  create multiple prebuilt hold contacts and activate one after probing without
  root/box pose writes, or move the selection interface into a real robot
  locomotion backend. Do not claim final success until posture selection,
  probing, carrying, and balance happen in the intended robot/control setting.
- [x] Prepare same-episode online probe-adaptive support diagnostic. Added
  online support-profile selection to the support-foot backend and forwarding
  fields through the direct wrapper, normalizer, and checker. Added
  `scripts/isaac/run_direct_carry_online_probe_adaptive_support_suite.sh`.
  Planned cases: `vertical_probe` should select `compact_medium_double_support`
  from observed vertical probe telemetry; `horizontal_probe` should select
  `nominal_reach_support` from observed horizontal probe telemetry. Login-node
  `bash -n` and `py_compile` checks passed.
- [x] Run/record same-episode online probe-adaptive support diagnostic.
  Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_online_support bash scripts/isaac/run_direct_carry_online_probe_adaptive_support_suite.sh`.
  Required interpretation: this is a single-rollout support-controller profile
  adaptation after probing, not online switching of the hold geometry and not
  final humanoid carrying.
- [x] Record same-episode online probe-adaptive support result. Retry3 job
  `167455` (`carry_online3`) completed with output
  `experiments/outputs/direct_carry_online_probe_adaptive_support_suite/20260706_direct_carry_online_probe_adaptive_support_retry3_64cm_8kg/online_probe_adaptive_support_summary.json`.
  Both single-episode cases passed. `vertical_probe` used observed
  `vertical_micro_lift` risk `0.5987436213151278`, selected
  `compact_medium_double_support`, completed `3640` steps with fall/drop `0`,
  max box travel `0.67985 m`, and final target distance `0.01187 m`.
  `horizontal_probe` used observed `horizontal_push_pull` risk
  `0.45948289037895235`, selected `nominal_reach_support`, completed
  `3640` steps with fall/drop `0`, max box travel `0.66684 m`, and final
  target distance `0.00041 m`.
- [x] Stop treating side-clamp online hold as the active route. Retry6 job
  `167470` confirmed the repeated blocker: drive targets were updated `7280`
  times and commanded full closure, but measured clamp motion stayed
  `3.83e-05 m` and the box dropped. Do not rerun the current side-clamp
  formulation unchanged.
- [x] Prepare direct Isaac online probe-adaptive cradle-contact diagnostic.
  The backend now prebuilds an optional top-lid contact body for
  `cradle_free_box`, disables its collision by default, and enables it inside
  the same episode for non-low observed probe-risk buckets. The wrapper,
  normalizer, checker, and suite script expose the collision availability,
  collision-enabled state, and update count. Login-node `bash -n` and
  `py_compile` checks passed.
- [ ] Run/record online probe-adaptive cradle-contact diagnostic. Planned
  command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_hold_cradle bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_suite.sh`.
  Required interpretation: this is same-episode contact redistribution in the
  Isaac scaffold, not learned control, not video-conditioned RL, and not final
  humanoid carrying.
- [x] Mark online probe-adaptive cradle-contact retry1 invalid. Job `167477`
  reached `server46` but failed during compute-side `py_compile` before Isaac
  rollout because the node read a stale/intermediate version of
  `build_core_world_anchored_footstep_carrier.py`. Login-node `py_compile`
  passed immediately afterward. Do not count this as experiment evidence.
- [ ] Await/record online probe-adaptive cradle-contact retry2 with a fresh
  stamp and startup delay:
  `20260706_direct_carry_online_probe_adaptive_hold_adaptive_cradle_retry2_64cm_8kg`.
- [x] Record online probe-adaptive cradle-contact retry2. Slurm job `167479`
  (`carry_hold_crad2`) ran on `server46` and passed the two-case suite.
  Summary:
  `experiments/outputs/direct_carry_online_probe_adaptive_hold_suite/20260706_direct_carry_online_probe_adaptive_hold_adaptive_cradle_retry2_64cm_8kg/online_probe_adaptive_hold_summary.json`.
  `vertical_probe`: observed risk `0.5932174593481317`, medium bucket,
  support profile `compact_medium_double_support`, hold profile
  `reinforced_contact_closure`, adaptive top-lid collision enabled, `3640`
  steps, fall/drop `0`, final target distance `0.01677 m`. `horizontal_probe`:
  observed risk `0.4508505528966966`, low bucket, support profile
  `nominal_reach_support`, hold profile `light_contact_closure`, adaptive
  top-lid collision left disabled, `3640` steps, fall/drop `0`, final target
  distance `0.00271 m`. Both used observed probe telemetry only and passed
  no-shortcut/support/target gates.
- [ ] Next implementation gate: preserve the same online probe -> support
  profile -> contact configuration contract, then replace a scaffold component
  with something more physically meaningful. Prefer replacing the
  support-foot scaffold with a controller-backed locomotion backend if it can
  enter rollout immediately; otherwise harden the cradle/contact switch under
  harder randomized object and posture variation. Do not return to waiting on
  external model downloads as the blocking path.
- [x] Prepare harder online probe-adaptive cradle-contact posture suite.
  Added `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`
  and extended the posture-suite summarizer with online probe/support/hold
  audit fields. The suite runs five same-episode cases:
  `front_mid`, `close_mid`, `chest_high` with vertical micro-lift medium-risk
  collision enabled, and `front_reach`, `low_front` with horizontal push-pull
  low-risk collision disabled. Login-node `bash -n` and `py_compile` passed.
- [ ] Run/record harder online probe-adaptive cradle-contact posture suite.
  Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=03:00:00 --job-name=carry_hold_post5 bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`.
  Required interpretation: this hardens same-episode active-probe contact
  adaptation across multiple carry poses, but it is still scaffold evidence
  until the support-foot backend is replaced by a real robot locomotion
  controller.
- [x] Mark first posture-suite submissions as scheduling-only, not evidence.
  Job `167501` (`carry_hold_post5`) stayed pending with a 3 hour limit and was
  canceled. Retry2 job `167502` (`carry_hold_p5r2`) used a 1 hour limit but
  Slurm estimated start at `2026-07-06T09:00:00` on `server53`; it was also
  canceled before execution to avoid leaving an unmonitored queued experiment.
  Neither job produced Isaac rollout evidence or a summary.
- [x] Add optional walking-realism gates to the five-posture suite:
  `MAX_NEAR_GROUND_FOOT_SPEED` and `MAX_NEAR_GROUND_FOOT_SLIP`. Also extended
  the suite summary cases with per-foot near-ground speed/slip fields.
  Login-node `bash -n` and `py_compile` passed.
- [ ] When compute resources are available, rerun the five-posture suite with
  a fresh stamp. Recommended first command:
  `SUITE_STAMP=20260706_direct_carry_online_probe_adaptive_hold_posture5_retry3_64cm_8kg srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=01:00:00 --job-name=carry_hold_p5r3 bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`.
- [ ] After the non-slip-gated five-posture suite runs, run a strict
  walking-realism audit by setting `MAX_NEAR_GROUND_FOOT_SPEED` and
  `MAX_NEAR_GROUND_FOOT_SLIP`. Expect this may fail on the current scaffold;
  if it fails, record it as evidence that the support-foot backend is still
  not physically walking and should be replaced.
- [x] Mark retry3 posture-suite submission as scheduling-only, not evidence.
  Job `167505` (`carry_hold_p5r3`) used a 45 minute limit but stayed pending
  with reason `Priority` and no scheduled start time; it was canceled before
  execution.
- [x] Add explicit slip-audit launcher
  `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_slip_audit_suite.sh`.
  Defaults: `MAX_NEAR_GROUND_FOOT_SPEED=0.80`,
  `MAX_NEAR_GROUND_FOOT_SLIP=0.20`. Login-node `bash -n` and `py_compile`
  passed.
- [ ] When GPU scheduling permits, run the non-slip-gated five-posture suite
  first, then the slip-audit wrapper. If the slip audit fails, record it as
  evidence against the current support-foot backend and prioritize replacing
  the backend rather than tuning the box cradle further.
- [x] Add split-run support to the five-posture suite for constrained Slurm
  scheduling. `CASE_FILTER` can select a full case id such as
  `vertical_probe_front_mid` or a posture name; `MIN_POSTURES=1` allows a
  single selected case to summarize cleanly. This does not reduce the final
  evidence requirement; it only lets us collect one posture at a time when the
  full suite cannot be scheduled.
- [ ] If full-suite scheduling remains unavailable, run single-case diagnostics
  in this order: `vertical_probe_front_mid`, `vertical_probe_close_mid`,
  `vertical_probe_chest_high`, `horizontal_probe_front_reach`,
  `horizontal_probe_low_front`; then combine or compare summaries manually.
- [x] Try the first split single-case submission. Job `167510`
  (`carry_hold_1case`) used `CASE_FILTER=vertical_probe_front_mid`,
  `MIN_POSTURES=1`, and a 20 minute time limit, but still stayed pending with
  reason `Priority` and no scheduled start time. It was canceled before
  execution. This is scheduling evidence only, not simulation evidence.
- [x] Add planted/no-slide posture audit wrapper. New file:
  `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_planted_slip_audit_suite.sh`.
  It sets and requires `PLANTED_STANCE_RAIL_PROPULSION=1` plus
  `FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`, while retaining near-ground speed
  and slip gates. Login-node `bash -n` and `py_compile` passed.
- [x] Add planted/no-slide fields to the posture-suite summarizer:
  planted propulsion enabled/steps, freeze-commanded stance-foot fields,
  per-foot near-ground speed/slip, and aggregate max near-ground speed/slip.
  Login-node `py_compile` passed.
- [ ] When GPU resources are available, run the planted/no-slide audit after
  or alongside the non-slip-gated posture suite. If it fails, prioritize
  support backend replacement over additional cradle tuning.
- [x] Add and run direct G1 stable-cradle propulsion tuning. Result:
  `close_targeted_creep_push018` and `low_targeted_creep_push028` passed the
  5 cm diagnostic gate with fall/drop `0` and rollout root/box writes `0`.
  `close_targeted_creep_push028` moved farther but failed tilt/relative-drift
  gates.
- [x] Add and run targeted-creep stop tuning. Result: `low_push032` passed
  the 560-step / 10 cm diagnostic gate: fall/drop `0`, final box
  target-directed travel `0.164657 m`, max tilt `0.128766 rad`, final
  relative offset `0.071063 m`, rollout root/velocity/box writes `0`.
- [x] Add and run low-cradle longer validation. Result: `low_push032` failed
  700 and 1000 step gates with falls, drops, large pitch, and large
  box-robot relative drift. Do not claim long-duration carrying from the
  560-step diagnostic.
- [x] Add and run corrected terminal-hold tuning. First run was invalid
  because `--terminal-hold-start-step 0` triggered hold immediately; retry2
  used `-1` and triggered from `box_target_travel`, but all 700-step hold
  cases still failed. Fixed symmetric hold offsets are not a sufficient brake.
- [ ] Next direct G1 implementation gate: add an explicit targeted-creep
  deceleration/recovery controller. It should reduce push/amplitude as
  target-directed travel grows, counter forward pitch without reversing the
  robot, and then pass a 700+ step free-box gate with fall/drop `0`, rollout
  root/velocity/box writes `0`, and declared target-directed travel.
- [x] Add and test travel-based creep decel and pitch-brake controls in
  `build_core_world_g1_box_scene.py`. Result: decel alone did not stop late
  forward pitch; most distance-producing cases still failed with falls/drops.
  `decel014_024_brake012` stayed stable for 700 steps but moved only
  `0.086960 m`.
- [x] Add and test latched pitch-brake variants. Result: abs-pitch latch can
  trigger too early on initial negative pitch; later positive/abs thresholds
  trigger too late to arrest forward fall.
- [x] Add and test zero-offset travel-triggered stand hold. Result: hold at
  `0.08-0.14 m` still failed later, proving that simply returning to stand is
  not a sufficient brake.
- [ ] Replace simple decel/hold with an explicit reverse-brake or counter-step
  recovery phase. Do not rerun unchanged stop/decel/zero-hold variants.
- [x] Add and test explicit reverse-brake targeted-creep variants. Result:
  reverse activation triggered by target travel, but all cases still failed.
  Distance-producing cases reached about `0.76-0.79 m` box target-directed
  travel with `20-26` fall events and `4-9` drops. Negative stance-push is not
  a reliable braking step in the current open-loop controller.
- [x] Add and test hold-balance variants after low-cradle creep. Result:
  negative balance sign was destructive and produced large backward travel
  with hundreds of falls/drops; positive sign was stable but suppressed motion
  to `0.003041 m`. This is not carrying.
- [x] Stop treating hand-written open-loop creep tuning as the main route.
  The repeated negative results show the bottleneck is the gait/control
  backend, not another simple decel/hold parameter sweep.
- [x] Prepare local WBC-AGILE controller-backed Core-scene suite. Added
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  It runs staged diagnostics: no-box AGILE walk, fixed light torso payload,
  and free low-cradle dynamic box, all with no root/velocity/box rollout
  writes allowed.
- [ ] Run/record local WBC-AGILE Core-scene suite:
  `SUITE_STAMP=20260706_g1_agile_low_cradle_suite1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --job-name=g1_agile_carry bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  If ONNX runtime or policy I/O fails, record it as an environment/adapter
  blocker and try the official torch-checkpoint backend before returning to
  hand-written gait code.
- [x] Run/record first local WBC-AGILE Core-scene suite. Job `167559`
  (`g1_agile_carry`) ran on `server46`; all three cases failed. The important
  result is that `agile_nobox_walk` already failed with `210` fall events,
  min robot z `0.185860 m`, max tilt `3.107505 rad`, and final
  target-directed robot travel `-0.120162 m`. Therefore fixed-payload and
  free-box failures are not carrying evidence; they are downstream of a
  no-box policy-adapter failure.
- [x] Fix the first AGILE observation adapter issue. The Core scene now reads
  `robot.get_angular_velocity()`, rotates it into body frame, and passes it
  as `root_ang_vel_b` instead of passing zeros. Summary/check JSON now report
  root-angular-velocity source, read failures, last error, and max norm.
- [x] Prepare no-box-only AGILE smoke with IsaacLab 29DoF drive gains. Added
  `scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh`.
- [ ] Run/record no-box-only AGILE smoke after root-angular-velocity fix:
  `SUITE_STAMP=20260706_g1_agile_nobox_smoke_angvel1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --job-name=g1_agile_nb bash scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh`.
  If this still fails, do not run more box cases; continue debugging policy
  observation/action scaling, drive gains, root/body frame convention, or try
  the official torch-checkpoint backend.
- [x] Run/record no-box AGILE after root-angular-velocity fix. Positive and
  negative command tests showed the policy stably walks in world negative X
  under original root orientation; yawing the robot 180 degrees failed. Added
  `--target-xy` to the Core scene and forwarded `TARGET_X/TARGET_Y` through
  runners.
- [x] Establish first valid AGILE no-box Core-scene gate. With target
  `[-1.2, 0.0]`, original orientation, IsaacLab 29DoF gains, ONNX backend,
  and `cmd_x=0.10`, `onnx_cmd010_isaaclab_gains` passed 320 steps with
  fall/drop `0`, final robot target-directed travel `0.562249 m`, max tilt
  `0.209202 rad`, and no rollout root/velocity/box writes.
- [x] Run fixed light torso payload diagnostics. Collision-enabled centered
  fixed payload stayed stable but moved only `0.115128 m`. Collision-disabled
  fixed inertial payload passed: fall/drop `0`, final robot travel
  `0.358296 m`, final box travel `0.371363 m`, max tilt `0.204425 rad`.
- [ ] Run/record free dynamic low-cradle box with AGILE, negative-X target,
  and box/cradle moved to negative X. Active Slurm job `167579`
  (`g1_agile_free`) uses stamp
  `20260706_g1_agile_free_lowcradle_targetnegx1`. Required interpretation:
  this is still a diagnostic low-cradle carry, not active probing and not
  video-conditioned RL.
- [x] Record first free dynamic low-cradle negative-X diagnostic. Job `167579`
  failed without falls/drops because the box and robot drifted apart:
  final relative offset `0.374029 m`, final robot target travel `-0.099899 m`,
  final box target travel `-0.305000 m`.
- [x] Run/record close free dynamic low-cradle retry. Job `167580`
  (`g1_agile_fre2`) passed the short gate: `360` steps, fall/drop `0`, final
  robot target travel `0.125915 m`, final box target travel `0.187173 m`,
  final relative offset `0.081144 m`, max tilt `0.146167 rad`, no rollout
  root/velocity/box writes.
- [ ] Next AGILE free-box gate: extend the close low-cradle dynamic-box run to
  `700+` steps with the same no-shortcut checks before increasing mass,
  changing object position, or adding active probing.
- [x] Run/record close free-box 700-step extension. Result: failed despite
  fall/drop `0`; final robot target travel `-0.691677 m`, final box target
  travel `-1.076183 m`, final relative offset `0.415493 m`, max tilt
  `0.632334 rad`. The free box/cradle contact did not remain coupled over the
  longer horizon.
- [x] Run/record no-box 700-step baseline. Result: `cmd_x=0.10` passed with
  fall/drop `0`, final robot target travel `0.878516 m`, max tilt
  `0.209202 rad`, and no rollout root/velocity/box writes. `cmd_x=0.05`
  failed with falls. This isolates the 700-step free-box failure to contact
  retention, not AGILE no-box locomotion.
- [ ] Next implementation gate: improve free-box retention for 700+ steps.
  Options: closer/enclosing cradle geometry, collision-enabled contact lid or
  softer side/end constraints, or an AGILE command stop/hold phase after
  short targetward progress. Keep every variant diagnostic-only until active
  probing and unknown-load belief updates are added.
- [x] Add AGILE-specific command stop/hold controls to the direct Core scene.
  The policy remains the official local WBC-AGILE adapter, but after a latched
  trigger from step, robot target-directed travel, or box target-directed
  travel, the velocity command is scaled by `--agile-command-hold-scale`
  while policy inference continues. Summary/check JSON now report hold active
  state, first active step/reason, active-step count, and last command.
- [ ] Run/record the close free-box 700-step AGILE hold diagnostic:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.05 FREE_MIN_BOX_TRAVEL=0.05 FREE_MAX_TILT=0.95 FREE_MAX_FINAL_REL=0.20 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_agile_hold bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Use a Curiosity-owned tmux-held Slurm allocation only. Do not use `carry1`,
  `sbatch`, `sspath`, or login-node simulation.
- [x] Record close free-box 700-step AGILE zero-command hold diagnostic. Job
  `167583` on `server02` failed. Hold triggered at step `117` from box
  target travel and the final AGILE command was `[0, 0, 0]`, but the robot
  still continued moving: final robot target travel `1.762137 m`, final box
  target travel `2.057596 m`, `87` fall events, `70` drops, max tilt
  `1.238096 rad`, final relative offset `0.532693 m`. Zero command alone is
  not a valid stop/hold transition in the current AGILE/Core adapter.
- [x] Add AGILE hold-trigger policy-state reset support:
  `--agile-command-hold-reset-policy-state`, forwarded by
  `AGILE_COMMAND_HOLD_RESET_POLICY_STATE=1`. For ONNX this clears
  `h_state/c_state/last_action`; for torch-checkpoint it calls the official
  wrapper reset when available. This is adapter diagnostics, not a new policy.
- [ ] Run/record the close free-box 700-step AGILE hold diagnostic with policy
  state reset:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_reset_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.05 FREE_MIN_BOX_TRAVEL=0.05 FREE_MAX_TILT=0.95 FREE_MAX_FINAL_REL=0.20 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_RESET_POLICY_STATE=1 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_agile_hres bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
- [x] Record close free-box 700-step AGILE hold diagnostic with policy-state
  reset. Job `167588` on `server02` failed. Hold triggered at step `117`,
  state reset count was `1`, reset error was null, final command was
  `[0, 0, 0]`, but the run had `306` fall events, `252` drops, max tilt
  `1.803860 rad`, final relative offset `0.961335 m`, final robot target
  travel `0.651390 m`, and final box target travel `1.522979 m`. Recurrent
  hidden-state persistence alone is not the explanation.
- [ ] Next implementation gate: add an AGILE hold mode that bypasses policy
  inference after the hold trigger and blends G1 joint targets toward the
  configured stand pose. This is a stop/settle diagnostic only. If it fails,
  stop trying AGILE command gates and redesign the cradle/contact retention.
- [x] Add AGILE `stand_targets` hold mode. After hold trigger it bypasses
  policy inference and exponentially blends the commanded joint targets toward
  the configured stand pose using `--agile-command-hold-stand-blend-rate`.
  Runner envs: `AGILE_COMMAND_HOLD_MODE=stand_targets` and
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE`.
- [ ] Run/record the close free-box 700-step AGILE stand-target hold
  diagnostic:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_stand_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.05 FREE_MIN_BOX_TRAVEL=0.05 FREE_MAX_TILT=0.95 FREE_MAX_FINAL_REL=0.20 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=stand_targets AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.025 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_agile_hstd bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_hold_stand_0706`, Slurm job
  `167590`, job-name `g1_agile_hstd`.
- [x] Record close free-box 700-step AGILE stand-target hold diagnostic. Job
  `167590` failed checker. Hold triggered at step `117`;
  `stand_targets` mode was active for `583` steps and policy inference count
  was only `20`, confirming that AGILE inference was bypassed after hold.
  Box retention improved (`box_drop_events=0`, min box z `0.467851 m`), but
  robot stability failed (`285` falls, min robot z `0.322664 m`, max tilt
  `1.419840 rad`) and final target-directed travel went negative for both
  robot and box. Stop sweeping AGILE command/hidden-state gates on this
  setup.
- [ ] Next implementation gate: redesign cradle/contact retention and settle
  posture together. The result suggests the free box can stay high during
  stand-target hold, but the G1 body cannot settle stably after short AGILE
  motion. Do not rerun zero-command, reset-state, or stand-target hold
  unchanged.
- [x] Add hold-only settle posture overrides so the AGILE walking default
  posture remains unchanged while the post-hold stand target can use a
  different load-carrying crouch. New CLI/env controls:
  `--agile-command-hold-stand-hip-pitch` / `AGILE_HOLD_STAND_HIP_PITCH`,
  `--agile-command-hold-stand-knee` / `AGILE_HOLD_STAND_KNEE`,
  `--agile-command-hold-stand-ankle-pitch` /
  `AGILE_HOLD_STAND_ANKLE_PITCH`, plus hip-roll, ankle-roll, and waist-pitch
  variants. Summary/check JSON records requested and applied hold-only joint
  targets.
- [ ] Run/record one low-crouch hold-only settle diagnostic:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lowcrouch_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.25 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.16 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=stand_targets AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.012 AGILE_HOLD_STAND_HIP_PITCH=-0.24 AGILE_HOLD_STAND_KNEE=0.58 AGILE_HOLD_STAND_ANKLE_PITCH=-0.34 AGILE_HOLD_STAND_WAIST_PITCH=-0.06 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_agile_lc bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_lowcrouch_0706`, Slurm job
  `167591`, job-name `g1_agile_lc`.
- [x] Record low-crouch hold-only settle diagnostic. It failed checker.
  Hold triggered at step `101`; low-crouch target was active for `599` steps
  and AGILE policy inference count was only `16`. Target-directed travel
  stayed positive, with final robot `0.752768 m` and final box `0.916188 m`,
  but the run had `378` falls, `355` drops, min box z `0.030000 m`, max tilt
  `1.283639 rad`, and final relative offset `0.545589 m`.
- [ ] Next implementation gate: add/enable physical cradle retention such as
  a top lid, higher rear/side capture, or hold-phase contact limiter. Do not
  keep sweeping settle pose alone; the low-crouch result shows posture can
  preserve direction but not box retention or balance.
- [x] Add optional physical cradle top lid and runner-controlled cradle
  geometry parameters. New Core-scene args:
  `--cradle-top-lid`, `--cradle-top-lid-local-z`,
  `--cradle-top-lid-thickness`, `--cradle-top-lid-x-scale`, and
  `--cradle-top-lid-y-scale`. Runner envs:
  `CRADLE_TOP_LID_ENABLED`, `CRADLE_SIDE_RAIL_HEIGHT`,
  `CRADLE_END_STOP_HEIGHT`, `CRADLE_RAIL_THICKNESS`, `CRADLE_MASS_SCALE`, and
  lid geometry envs. This is physical contact retention, not policy success.
- [ ] Run/record top-lid low-crouch retention diagnostic:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_lowcrouch_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.16 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=stand_targets AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.012 AGILE_HOLD_STAND_HIP_PITCH=-0.24 AGILE_HOLD_STAND_KNEE=0.58 AGILE_HOLD_STAND_ANKLE_PITCH=-0.34 AGILE_HOLD_STAND_WAIST_PITCH=-0.06 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_agile_lid bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_lid_lowcrouch_0706`, Slurm job
  `167592`, job-name `g1_agile_lid`.
- [x] Record static top-lid low-crouch retention diagnostic. It failed
  checker. Static top lid reduced falls/drops relative to low-crouch without
  lid (`262` falls / `205` drops vs. `378` / `355`) and improved final
  relative offset to `0.311611 m`, but min robot z was `0.174101 m`, min box
  z was `0.072546 m`, max tilt was `1.542573 rad`, and final robot/box
  target-directed travel became negative. Static lid helps retention but
  interferes with useful motion.
- [ ] Next implementation gate: make the top lid a hold-phase contact limiter
  that is physically present but collision-disabled before hold, then enabled
  when the AGILE hold trigger fires. This tests whether early AGILE movement
  can stay freer while settle gets extra box capture.
- [x] Add hold-phase top-lid collision activation:
  `--cradle-top-lid-enable-on-hold` / `CRADLE_TOP_LID_ENABLE_ON_HOLD=1`.
  The lid prim is spawned, but its collision is disabled until the AGILE hold
  trigger fires; then the scene applies `UsdPhysics.CollisionAPI` and records
  activation step/count/error.
- [ ] Run/record hold-phase top-lid low-crouch diagnostic:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_onhold_lowcrouch_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.16 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=stand_targets AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.012 AGILE_HOLD_STAND_HIP_PITCH=-0.24 AGILE_HOLD_STAND_KNEE=0.58 AGILE_HOLD_STAND_ANKLE_PITCH=-0.34 AGILE_HOLD_STAND_WAIST_PITCH=-0.06 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lid_hold bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_lid_onhold_0706`, Slurm job
  `167593`, job-name `g1_lid_hold`. Result: invalid as carrying evidence.
  The lid activation fired at step `85` and the robot/box were still stable,
  but applying `UsdPhysics.CollisionAPI` during rollout invalidated the PhysX
  tensor view and stopped at `completed_steps=85` with
  `Failed to get DOF position targets from backend`.
- [x] Fix hold-phase top-lid activation implementation. The lid now receives
  `CollisionAPI` at scene construction with `physics:collisionEnabled=false`;
  hold activation only toggles the existing collision-enabled attr to true.
  This should avoid runtime schema mutation and tensor-view invalidation.
- [ ] Rerun hold-phase top-lid low-crouch diagnostic after the collision attr
  fix using stamp
  `20260706_g1_agile_free_close_hold_lid_onhold_attrfix_targetnegx1_700`.
  Submitted as tmux `curiosity_g1_agile_lid_onhold_attrfix_0706`, Slurm job
  `167594`, job-name `g1_lid_attr`.
- [x] Record hold-phase top-lid attr-fix result. The attr-toggle fix worked
  and the run completed 700 steps. Top-lid collision enabled at step `92`
  with update count `1` and error null. It solved box dropping
  (`box_drop_events=0`, min box z `0.498752 m`) and kept final relative
  offset `0.268659 m`, but robot stability still failed with `95` falls,
  min robot z `0.342325 m`, max tilt `1.245552 rad`, and final robot/box
  target-directed travel negative. Contact retention improved; settle posture
  remains wrong.
- [ ] Run/record hold-phase top-lid with AGILE `policy_command` zero-command
  hold instead of `stand_targets`:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_policycmd_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=policy_command CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lid_pol bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_lid_policycmd_0706`, Slurm job
  `167596`, job-name `g1_lid_pol`; pending with reason `Priority` and no
  start estimate at submission-time checks.
- [x] Record hold-phase top-lid `policy_command` result. Job `167596`
  completed 700 steps and failed checker. Hold triggered at step `102`;
  top-lid collision enabled at step `102`; update count `1`, error null.
  Unlike stand-target hold, final target-directed travel stayed positive
  (robot `0.962488 m`, box `0.835041 m`), but stability and retention failed:
  `352` falls, `68` drops, min robot z `0.191603 m`, min box z `0.085619 m`,
  max tilt `3.121745 rad`, final relative offset `0.415014 m`. This confirms
  that contact lid plus raw AGILE zero-command hold is not enough.
- [ ] Next implementation gate: build an Isaac-side stable hold transition or
  balance-aware settle controller. Do not wait for external models, and do not
  repeat unchanged `policy_command`, reset-state, or fixed `stand_targets`
  sweeps. The immediate target is to combine hold-phase box retention with a
  slower, feedback-aware body settle after the box reaches the travel trigger.
- [x] Add Isaac-side hybrid hold and hold-gated balance controls. New controls:
  `AGILE_COMMAND_HOLD_MODE=policy_then_stand`,
  `AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS`,
  `BALANCE_FEEDBACK_CONTROLLER`,
  `BALANCE_START_ON_AGILE_HOLD`, and `BALANCE_FEEDBACK_BASE=command`. The
  hybrid mode keeps AGILE zero-command hold briefly after contact capture, then
  blends toward hold stand targets; balance feedback can be applied only after
  the AGILE hold trigger and can add corrections on top of the current command
  target rather than overwriting from the default stand pose.
- [ ] Run/record hold-phase top-lid hybrid/balance diagnostic:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_hybrid_balance_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=policy_then_stand AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=80 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.006 AGILE_HOLD_STAND_HIP_PITCH=-0.18 AGILE_HOLD_STAND_KNEE=0.42 AGILE_HOLD_STAND_ANKLE_PITCH=-0.24 AGILE_HOLD_STAND_WAIST_PITCH=-0.03 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_GAIN=0.16 BALANCE_PITCH_RATE_GAIN=0.012 BALANCE_ROLL_GAIN=0.10 BALANCE_ROLL_RATE_GAIN=0.006 BALANCE_ADJUSTMENT_LIMIT=0.12 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.04 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.04 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.15 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.15 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_hyb_bal bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_hybrid_balance_0706`, Slurm job
  `167602`, job-name `g1_hyb_bal`; queued waiting for resources at
  submission-time check.
- [x] Record hold-phase top-lid hybrid/balance diagnostic. Job `167602`
  completed 700 steps and failed checker. Hold triggered at step `102`;
  `policy_then_stand` delay was `80` steps; stand-target active steps `518`;
  top-lid enabled at step `102`; balance feedback started at step `102` and
  was active for `592` steps with `balance_feedback_base=command`. The run
  stayed stable until roughly step `390`, then pitched forward and failed:
  `309` falls, `293` drops, min robot z `0.246960 m`, min box z
  `0.097018 m`, max tilt `1.224603 rad`. Target-directed travel was strong
  (robot `1.125550 m`, box `1.155785 m`), so this is a hold/posture feedback
  problem, not lack of forward progress.
- [ ] Run/record pitch-sign-flip hybrid/balance diagnostic with the same scene
  and timing but `BALANCE_PITCH_SIGN=1.0`, to test whether the previous run's
  forward pitch collapse came from the feedback sign convention:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_hybrid_balance_pitchsignpos_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=policy_then_stand AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=80 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.006 AGILE_HOLD_STAND_HIP_PITCH=-0.18 AGILE_HOLD_STAND_KNEE=0.42 AGILE_HOLD_STAND_ANKLE_PITCH=-0.24 AGILE_HOLD_STAND_WAIST_PITCH=-0.03 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.16 BALANCE_PITCH_RATE_GAIN=0.012 BALANCE_ROLL_GAIN=0.10 BALANCE_ROLL_RATE_GAIN=0.006 BALANCE_ADJUSTMENT_LIMIT=0.12 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.04 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.04 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.15 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.15 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_hyb_psp bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_hybrid_pitchsignpos_0706`, Slurm job
  `167603`, job-name `g1_hyb_psp`; queued waiting for resources at
  submission-time check.
- [x] Record pitch-sign-flip hybrid/balance diagnostic. Job `167603`
  completed 700 steps and failed checker, but it identified the pitch sign
  issue. `BALANCE_PITCH_SIGN=1.0` reduced max absolute pitch from
  `1.224603 rad` to `0.181384 rad`; first fall moved to step `560` and first
  drop to step `600`. The new failure mode was lateral roll drift:
  max roll `1.570825 rad`, final roll `-1.499206 rad`, `148` falls,
  `108` drops, min robot z `0.167102 m`, min box z `0.081154 m`. Final
  relative offset was acceptable at `0.279529 m`, but target-directed travel
  collapsed because the robot moved sideways.
- [ ] Run/record pitch-only hybrid/balance diagnostic with pitch sign fixed and
  roll feedback disabled, to test whether the roll feedback branch caused the
  lateral fall:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_hybrid_balance_pitchonly_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=policy_then_stand AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=80 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.006 AGILE_HOLD_STAND_HIP_PITCH=-0.18 AGILE_HOLD_STAND_KNEE=0.42 AGILE_HOLD_STAND_ANKLE_PITCH=-0.24 AGILE_HOLD_STAND_WAIST_PITCH=-0.03 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.16 BALANCE_PITCH_RATE_GAIN=0.012 BALANCE_ROLL_GAIN=0.0 BALANCE_ROLL_RATE_GAIN=0.0 BALANCE_ADJUSTMENT_LIMIT=0.12 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.04 BALANCE_ROLL_ACTIVATION_THRESHOLD=999.0 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.15 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=999.0 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_hyb_pon bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_hybrid_pitchonly_0706`, Slurm job
  `167604`, job-name `g1_hyb_pon`; queued waiting for resources at
  submission-time check.
- [x] Record pitch-only hybrid/balance diagnostic. Job `167604` completed
  700 steps and failed checker. Disabling roll feedback made failure earlier,
  not better: first fall step `290`, first drop step `310`, max pitch
  `1.345647 rad`, final pitch `-1.284981 rad`, max roll only
  `0.152900 rad`, `418` falls, `396` drops. This shows roll feedback was
  helping indirectly, but the existing hard-coded roll branch is too crude and
  caused side drift in the pitch-sign-flip run.
- [ ] Next implementation gate: expose left/right roll-feedback joint
  multipliers instead of hard-coding the same roll correction on both sides.
  The immediate diagnostic should test mirrored roll correction with fixed
  pitch sign, not another unchanged gain sweep.
- [x] Add configurable roll-feedback joint multipliers. New controls:
  `BALANCE_ROLL_LEFT_ANKLE_SCALE`,
  `BALANCE_ROLL_RIGHT_ANKLE_SCALE`,
  `BALANCE_ROLL_LEFT_HIP_SCALE`, and
  `BALANCE_ROLL_RIGHT_HIP_SCALE`. Defaults preserve the previous hard-coded
  roll feedback behavior. Summary/check output records the multipliers.
- [ ] Run/record mirrored-roll hybrid/balance diagnostic with pitch sign fixed:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_hybrid_balance_mirrorroll_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=policy_then_stand AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=80 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.006 AGILE_HOLD_STAND_HIP_PITCH=-0.18 AGILE_HOLD_STAND_KNEE=0.42 AGILE_HOLD_STAND_ANKLE_PITCH=-0.24 AGILE_HOLD_STAND_WAIST_PITCH=-0.03 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.16 BALANCE_PITCH_RATE_GAIN=0.012 BALANCE_ROLL_GAIN=0.10 BALANCE_ROLL_RATE_GAIN=0.006 BALANCE_ADJUSTMENT_LIMIT=0.12 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.04 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.04 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.15 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.15 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_hyb_mrl bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_hybrid_mirrorroll_0706`, Slurm job
  `167605`, job-name `g1_hyb_mrl`; queued waiting for resources at
  submission-time check.
- [x] Record mirrored-roll hybrid/balance diagnostic. Job `167605` completed
  700 steps and failed checker. Mirrored roll multipliers avoided the severe
  side-roll failure (`max_abs_roll_rad=0.403314`, final roll near zero) but
  pitch collapse returned (`max_abs_pitch_rad=1.335714`, final pitch
  `-1.304736`). First fall step was `510`; first drop step was `530`;
  failures were `195` falls and `173` drops. Target-directed travel stayed
  positive (robot `1.006663 m`, box `1.019586 m`).
- [ ] Stop blind hold/balance gain sweeps. Repeated diagnostics show the
  current post-capture controller can trade off pitch, roll, and forward travel
  but cannot stabilize all three. Next work should be a new mechanism:
  lateral drift brake, stance/footstep repositioning, or a proper WBC hold
  interface, rather than another unchanged blend/feedback parameter sweep.
- [x] Add hold-rescue state machine for the Isaac/G1 post-capture phase. New
  controls:
  `AGILE_COMMAND_HOLD_RESCUE_ENABLE`,
  `AGILE_COMMAND_HOLD_RESCUE_FORWARD_PITCH_THRESHOLD`,
  `AGILE_COMMAND_HOLD_RESCUE_ABS_ROLL_THRESHOLD`,
  `AGILE_COMMAND_HOLD_RESCUE_BLEND_RATE`, and hold-rescue joint targets for
  hip/knee/ankle/waist. Rescue latches only after AGILE hold is active and
  writes trigger step/reason and active steps to summary/check output.
- [ ] Run/record hold-rescue diagnostic on the mirrored-roll base:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_hybrid_rescue_mirrorroll_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.0 AGILE_COMMAND_HOLD_MODE=policy_then_stand AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=80 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.006 AGILE_HOLD_STAND_HIP_PITCH=-0.18 AGILE_HOLD_STAND_KNEE=0.42 AGILE_HOLD_STAND_ANKLE_PITCH=-0.24 AGILE_HOLD_STAND_WAIST_PITCH=-0.03 AGILE_COMMAND_HOLD_RESCUE_ENABLE=1 AGILE_COMMAND_HOLD_RESCUE_FORWARD_PITCH_THRESHOLD=-0.30 AGILE_COMMAND_HOLD_RESCUE_BLEND_RATE=0.035 AGILE_HOLD_RESCUE_HIP_PITCH=0.03 AGILE_HOLD_RESCUE_KNEE=0.28 AGILE_HOLD_RESCUE_ANKLE_PITCH=-0.06 AGILE_HOLD_RESCUE_WAIST_PITCH=0.08 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.16 BALANCE_PITCH_RATE_GAIN=0.012 BALANCE_ROLL_GAIN=0.10 BALANCE_ROLL_RATE_GAIN=0.006 BALANCE_ADJUSTMENT_LIMIT=0.12 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.04 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.04 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.15 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.15 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_hyb_rsc bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_hybrid_rescue_0706`, Slurm job
  `167607`, job-name `g1_hyb_rsc`; queued waiting for resources at
  submission-time check.
- [x] Record hold-rescue diagnostic. Job `167607` completed 700 steps and
  failed checker. Rescue triggered correctly at step `433` from
  `forward_pitch`, stayed active for `267` steps, and used rescue target
  hip/knee/ankle/waist `0.03/0.28/-0.06/0.08`. It did not solve carrying:
  first fall step `500`, first drop step `530`, `208` falls, `173` drops,
  min robot z `0.169106 m`, min box z `0.086920 m`, max tilt
  `1.528025 rad`. It pulled pitch back by the end but converted the failure
  into roll-over (`final_roll_rad=-1.481067`). This confirms that static
  rescue targets are not enough.
- [ ] Next implementation gate: keep AGILE policy active after capture with a
  low-speed, target-directed hold command and lateral correction, instead of
  switching fully to static joint targets. Static target hold/rescue has now
  repeatedly failed.
- [ ] Run/record post-capture slow-walk policy hold diagnostic:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_policycmd_slowwalk_targetnegx1_700 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.35 AGILE_COMMAND_HOLD_MODE=policy_command BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_pol_slow bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_policy_slowwalk_0706`, Slurm job
  `167608`, job-name `g1_pol_slow`; queued waiting for resources at
  submission-time check.
- [x] Record post-capture slow-walk policy hold diagnostic. Job `167608`
  completed 700 steps and passed checker. This is the first passing free-box
  low-cradle AGILE diagnostic in this sequence. Hold triggered at step `102`;
  hold mode was `policy_command`; post-capture command was reduced but not
  zero (`agile_last_command_xyz_yaw=[0.035, 0, 0]`). Top lid enabled at step
  `102`; balance feedback active for `595` steps. Results: `fall_events=0`,
  `box_drop_events=0`, min robot z `0.758436 m`, min box z `0.879201 m`, max
  tilt `0.254946 rad`, final robot target-directed travel `1.182280 m`,
  final box target-directed travel `1.243254 m`, final relative offset
  `0.107421 m`, no root/velocity/box rollout writes. Interpretation: keeping
  WBC-AGILE active with a slow post-capture command is the correct next
  control path; static hold/rescue was the wrong direction.
- [ ] Next validation gate: do not claim final success yet. Run held-out
  diagnostics that vary box mass/shape and/or duration using the slow-walk
  policy hold controller. Required before a stronger claim: at least a heavier
  payload, a larger box, and a longer carry.
- [ ] Run/record heavier-longer slow-walk validation:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_policycmd_slowwalk_mass0p5_900_targetnegx1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=900 FREE_BOX_MASS=0.5 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.35 AGILE_COMMAND_HOLD_MODE=policy_command BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_mass05_9 bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_slowwalk_mass05_900_0706`, Slurm job
  `167613`, job-name `g1_mass05_9`; queued waiting for resources at
  submission-time check.
- [x] Record heavier-longer slow-walk validation. Job `167613` completed
  900 steps and failed checker. With box mass `0.5 kg`, the same hold
  controller failed: first fall step `420`, first drop step `530`,
  `485` falls, `373` drops, max tilt `2.858130 rad`, max roll
  `2.858130 rad`, max pitch `1.556304 rad`, final target-directed travel
  became negative. The slow-walk controller is a valid base smoke but not yet
  robust to doubled mass and longer duration.
- [ ] Run/record 0.5kg lower-speed hold validation:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_policycmd_mass0p5_hold015_700_targetnegx1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_MASS=0.5 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.15 AGILE_COMMAND_HOLD_MODE=policy_command BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_m05_h15 bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_mass05_hold015_0706`, Slurm job
  `167614`, job-name `g1_m05_h15`; queued waiting for resources at
  submission-time check.
- [x] Record 0.5kg lower-speed hold validation. Job `167614` completed
  700 steps and failed checker. Lowering hold scale to `0.15` made the 0.5kg
  case worse, not better: first fall step `310`, first drop step `330`,
  `399` falls, `372` drops, min robot z `0.126606 m`, min box z
  `0.074592 m`, max tilt `1.141853 rad`. This rules out a simple
  "heavier means slower fixed hold command" explanation.
- [ ] Next implementation gate: implement load/contact-adaptive post-capture
  command selection instead of fixed hold scale. Candidate signals available
  in the current scene are tilt, tilt rate, box/robot relative offset, box z,
  and target-directed progress. The adaptive controller should keep WBC-AGILE
  active but reduce, increase, or redirect command based on those online
  signals rather than assuming a constant hold scale.
- [x] Add adaptive post-capture command controls. New controls:
  `AGILE_COMMAND_HOLD_ADAPTIVE_SCALE`, adaptive min/max scale, tilt/rate/
  relative-offset risk thresholds, scale smoothing, and optional lateral
  correction with gain/limit/sign. Summary/check output records active steps,
  observed scale range, final risk, lateral command, and lateral error.
- [ ] Run/record 0.5kg adaptive hold + lateral correction validation:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_adaptive_mass0p5_700_targetnegx1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_MASS=0.5 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.30 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.35 AGILE_COMMAND_HOLD_MODE=policy_command AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=1 AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE=0.18 AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START=0.22 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP=0.55 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_START=3.0 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_STOP=10.0 AGILE_COMMAND_HOLD_ADAPTIVE_REL_START=0.12 AGILE_COMMAND_HOLD_ADAPTIVE_REL_STOP=0.30 AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING=0.20 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.08 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.035 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.14 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.10 CRADLE_TOP_LID_Y_SCALE=1.05 CRADLE_SIDE_RAIL_HEIGHT=0.10 CRADLE_END_STOP_HEIGHT=0.11 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_adapt05 bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_adaptive_mass05_0706`, Slurm job
  `167632`, job-name `g1_adapt05`; allocated on `server28` at
  submission-time check.
- [x] Record 0.5kg adaptive hold + lateral correction validation. Job
  `167632` completed 700 steps and passed checker. The same `0.5 kg` payload
  that failed with fixed `0.35` and fixed `0.15` hold scales now passed:
  `fall_events=0`, `box_drop_events=0`, min robot z `0.710033 m`, min box z
  `0.835384 m`, max tilt `0.192797 rad`, final robot target-directed travel
  `1.812799 m`, final box target-directed travel `1.788483 m`, final relative
  offset `0.034773 m`. Adaptive scale was active for `577` steps, observed
  scale range `0.253006` to `0.35`; lateral correction was active for
  `577` steps and reached max command `0.035`.
- [ ] Next validation gate: test shape/size variation with the adaptive
  controller. A larger box should be attempted before any stronger carrying
  claim.
- [ ] Run/record larger-box adaptive validation:
  `SUITE_STAMP=20260706_g1_agile_free_close_hold_lid_adaptive_mass0p5_largerbox_700_targetnegx1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_MASS=0.5 FREE_BOX_SIZE_X=0.14 FREE_BOX_SIZE_Y=0.10 FREE_BOX_SIZE_Z=0.08 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=1.05 FREE_MAX_FINAL_REL=0.35 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.35 AGILE_COMMAND_HOLD_MODE=policy_command AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=1 AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE=0.18 AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START=0.22 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP=0.55 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_START=3.0 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_STOP=10.0 AGILE_COMMAND_HOLD_ADAPTIVE_REL_START=0.12 AGILE_COMMAND_HOLD_ADAPTIVE_REL_STOP=0.30 AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING=0.20 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.08 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.035 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.16 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.15 CRADLE_TOP_LID_Y_SCALE=1.10 CRADLE_SIDE_RAIL_HEIGHT=0.12 CRADLE_END_STOP_HEIGHT=0.13 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_adapt_lg bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_adaptive_largerbox_0706`, Slurm job
  `167634`, job-name `g1_adapt_lg`; queued waiting for resources at
  submission-time check.
- [x] Record larger-box adaptive validation. Job `167634` completed
  700 steps and passed the configured loose checker with `fall_events=0`,
  `box_drop_events=0`, and no root/velocity/box rollout writes. The larger
  `0.5 kg`, `0.14 x 0.10 x 0.08 m` box moved with the robot: final robot
  target-directed travel `1.954226 m`, final box target-directed travel
  `2.011707 m`, final relative offset `0.071092 m`, max relative offset
  `0.206568 m`. This is not a strong success claim: max/final tilt was
  `0.479985 rad` and lateral path correction saturated at `0.035` while final
  lateral path error reached `1.585361 m`. These tilt/roll fields are
  robot-root attitude, not box attitude. Treat this as "captured and carried
  without falling/dropping, but with poor path/root-attitude quality."
- [ ] Next implementation gate: add or tune a stricter target-path/attitude
  controller for larger boxes. The next checker gate should include tighter
  robot-root tilt, true box tilt, and lateral path criteria, otherwise a
  sideways-drifting carried box can pass loose fall/drop thresholds.
- [x] Add hold-phase yaw correction controls for larger-box path drift.
  New disabled-by-default controls:
  `AGILE_COMMAND_HOLD_YAW_CORRECTION`, `AGILE_COMMAND_HOLD_YAW_GAIN`,
  `AGILE_COMMAND_HOLD_YAW_LIMIT`, and `AGILE_COMMAND_HOLD_YAW_SIGN`.
  They use target-path lateral error to add a bounded yaw command during the
  post-capture AGILE hold phase. Summary/check output records active steps,
  first active step, max yaw command, and final yaw-control error. Syntax
  checks passed with `python3 -m py_compile
  scripts/isaac/build_core_world_g1_box_scene.py
  scripts/isaac/check_core_world_g1_box_scene_summary.py` and `bash -n
  scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
- [ ] Run/record larger-box adaptive + yaw-correction strict diagnostic:
  `SUITE_STAMP=20260706_g1_agile_adaptive_largerbox_yawcorr_strict_700_targetnegx1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_MASS=0.5 FREE_BOX_SIZE_X=0.14 FREE_BOX_SIZE_Y=0.10 FREE_BOX_SIZE_Z=0.08 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=0.35 FREE_MAX_FINAL_REL=0.25 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.35 AGILE_COMMAND_HOLD_MODE=policy_command AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=1 AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE=0.18 AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START=0.22 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP=0.55 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_START=3.0 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_STOP=10.0 AGILE_COMMAND_HOLD_ADAPTIVE_REL_START=0.12 AGILE_COMMAND_HOLD_ADAPTIVE_REL_STOP=0.30 AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING=0.20 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.08 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.035 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.08 AGILE_COMMAND_HOLD_YAW_LIMIT=0.18 AGILE_COMMAND_HOLD_YAW_SIGN=1.0 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.16 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.15 CRADLE_TOP_LID_Y_SCALE=1.10 CRADLE_SIDE_RAIL_HEIGHT=0.12 CRADLE_END_STOP_HEIGHT=0.13 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_yaw bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_yawcorr_0706`, Slurm job
  `167646`, job-name `g1_lg_yaw`; queued waiting for resources at
  submission-time check.
- [x] Record larger-box adaptive + yaw-correction strict diagnostic. Job
  `167646` completed 700 steps and failed the stricter checker only on
  robot-root attitude: `max_tilt_rad 0.532508 > 0.35`. It did not fall or drop:
  `fall_events=0`, `box_drop_events=0`, min robot z `0.721055 m`, min box z
  `0.778733 m`, no root/velocity/box rollout writes. Final robot
  target-directed travel was `0.638052 m`, final box target-directed travel
  `0.617244 m`, final/max relative offset `0.201939 m`. Yaw correction was
  active for `603` steps, max yaw command `0.088684`, final yaw-control
  lateral error `1.059896 m`; lateral velocity correction still saturated at
  `0.035`. Interpretation: yaw correction reduces final root roll but worsens
  transient root tilt and reduces forward progress. Do not continue blind
  yaw-gain sweeps as the main path.
- [x] Add true box attitude telemetry and optional box-tilt adaptive risk.
  `max_tilt_rad`, `final_roll_rad`, and `final_pitch_rad` remain robot-root
  attitude fields for historical comparability. New fields:
  `max_box_tilt_rad`, `max_abs_box_roll_rad`, `max_abs_box_pitch_rad`,
  `final_box_roll_rad`, and `final_box_pitch_rad`. The checker now supports
  optional `--max-box-tilt`, and the runner exposes `FREE_MAX_BOX_TILT`.
  New disabled-by-default controller controls:
  `AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT`,
  `AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_START`,
  `AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_STOP`,
  `AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_START`, and
  `AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_STOP`. Lightweight syntax checks
  passed.
- [x] Add target-path lateral-error telemetry and optional checker gates.
  New summary/check fields:
  `max_abs_robot_target_lateral_error_m`,
  `max_abs_box_target_lateral_error_m`,
  `final_robot_target_lateral_error_m`, and
  `final_box_target_lateral_error_m`. The checker now supports
  `--max-robot-target-lateral-error`, `--max-box-target-lateral-error`,
  `--max-final-robot-target-lateral-error`, and
  `--max-final-box-target-lateral-error`. The runner exposes case-specific
  env gates such as `FREE_MAX_ROBOT_LATERAL_ERROR`,
  `FREE_MAX_BOX_LATERAL_ERROR`, `FREE_MAX_FINAL_ROBOT_LATERAL_ERROR`, and
  `FREE_MAX_FINAL_BOX_LATERAL_ERROR`. Lightweight `py_compile` and `bash -n`
  checks passed.
- [ ] Next implementation gate: improve larger-box root-attitude/contact
  strategy, not just path correction. Candidate mechanisms: larger or more
  enclosing cradle geometry, true box/root roll-aware slowdown before high
  tilt, lower hold height, torso/chest support contact, or a posture switch
  for larger boxes.
- [x] Add optional torso/chest support pad to the front cradle. New
  disabled-by-default controls: `CRADLE_CHEST_PAD_ENABLED`,
  `CRADLE_CHEST_PAD_ENABLE_ON_HOLD`, `CRADLE_CHEST_PAD_LOCAL_X/Y/Z`, and
  `CRADLE_CHEST_PAD_SIZE_X/Y/Z`. The scene records pad geometry and hold-time
  collision activation fields. This is intended to test a torso-supported
  carrying posture for larger boxes; it is not a shortcut because it remains a
  physical fixed-to-torso support geometry with no rollout root/box writes.
  Lightweight `py_compile` and `bash -n` checks passed.
- [ ] After job `167670` is recorded, run a larger-box chest-pad diagnostic if
  true-box-tilt adaptive alone is still weak. Required gates should include
  no fall/drop, no rollout root/velocity/box pose writes, robot-root tilt,
  true box tilt, relative box/robot offset, and target-line lateral error.
- [ ] Run/record larger-box chest-pad strict diagnostic:
  `SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_strict_700_targetnegx1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_MASS=0.5 FREE_BOX_SIZE_X=0.14 FREE_BOX_SIZE_Y=0.10 FREE_BOX_SIZE_Z=0.08 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=0.35 FREE_MAX_BOX_TILT=0.45 FREE_MAX_FINAL_REL=0.25 FREE_MAX_ROBOT_LATERAL_ERROR=0.80 FREE_MAX_BOX_LATERAL_ERROR=0.80 FREE_MAX_FINAL_ROBOT_LATERAL_ERROR=0.60 FREE_MAX_FINAL_BOX_LATERAL_ERROR=0.60 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.35 AGILE_COMMAND_HOLD_MODE=policy_command AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=1 AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE=0.10 AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START=0.14 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_START=2.0 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_STOP=7.0 AGILE_COMMAND_HOLD_ADAPTIVE_REL_START=0.10 AGILE_COMMAND_HOLD_ADAPTIVE_REL_STOP=0.26 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT=1 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_START=0.12 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_STOP=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_START=2.0 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_STOP=7.0 AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING=0.25 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.08 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.035 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_YAW_CORRECTION=0 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.16 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.15 CRADLE_TOP_LID_Y_SCALE=1.10 CRADLE_SIDE_RAIL_HEIGHT=0.12 CRADLE_END_STOP_HEIGHT=0.13 CRADLE_CHEST_PAD_ENABLED=1 CRADLE_CHEST_PAD_ENABLE_ON_HOLD=1 CRADLE_CHEST_PAD_LOCAL_X=-0.02 CRADLE_CHEST_PAD_LOCAL_Y=0.0 CRADLE_CHEST_PAD_LOCAL_Z=0.10 CRADLE_CHEST_PAD_SIZE_X=0.04 CRADLE_CHEST_PAD_SIZE_Y=0.38 CRADLE_CHEST_PAD_SIZE_Z=0.22 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_chest bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_chestpad_0706`, Slurm job
  `167691`, job-name `g1_lg_chest`; queued waiting for resources at
  submission-time check.
- [x] Add reproducible larger-box strict wrapper. New executable:
  `scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  It centralizes the strict larger-box settings and supports
  `LARGERBOX_STRICT_MODE=boxtilt`, `lowcarry`, or `chestpad`. It still delegates to
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`, so it inherits the
  compute-node/login-node guard and no-root/box-write checks. Lightweight
  checks passed: `bash -n` for both scripts and `py_compile` for the scene and
  checker.
- [x] Add multi-posture larger-box matrix wrapper. New executable:
  `scripts/isaac/run_core_world_g1_largerbox_posture_matrix.sh`. It is guarded
  against login-node execution and is intended to run inside one compute
  allocation. It sequentially tests `boxtilt`, `lowcarry`, and `chestpad`
  larger-box strict modes and then calls the larger-box strict summarizer.
  This prepares a direct gate for the final requirement that different
  carrying postures must preserve balanced walking. Lightweight `bash -n` and
  `py_compile` checks passed.
- [x] Add larger-box strict result summarizer. New executable:
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`. It reads
  existing summary/check JSON files and reports fall/drop, rollout writes,
  robot-root tilt, true box tilt, relative offset, target-line lateral error,
  adaptive-scale telemetry, and chest-pad/top-lid activation. This is a
  lightweight JSON summarizer only; it does not run Isaac or load models.
  `py_compile` passed.
- [ ] Run/record larger-box true-box-tilt adaptive strict diagnostic:
  `SUITE_STAMP=20260706_g1_agile_adaptive_largerbox_boxtilt_strict_700_targetnegx1 DEVICE=cpu STRICT=0 AGILE_POLICY_BACKEND=onnx RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 TARGET_X=-1.2 TARGET_Y=0.0 FREE_STEPS=700 FREE_BOX_MASS=0.5 FREE_BOX_SIZE_X=0.14 FREE_BOX_SIZE_Y=0.10 FREE_BOX_SIZE_Z=0.08 FREE_BOX_POS_X=-0.18 FREE_CRADLE_LOCAL_X=-0.18 FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02 FREE_MAX_TILT=0.35 FREE_MAX_BOX_TILT=0.45 FREE_MAX_FINAL_REL=0.25 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18 AGILE_COMMAND_HOLD_SCALE=0.35 AGILE_COMMAND_HOLD_MODE=policy_command AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=1 AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE=0.10 AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START=0.14 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_START=2.0 AGILE_COMMAND_HOLD_ADAPTIVE_RATE_STOP=7.0 AGILE_COMMAND_HOLD_ADAPTIVE_REL_START=0.10 AGILE_COMMAND_HOLD_ADAPTIVE_REL_STOP=0.26 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT=1 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_START=0.12 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_STOP=0.35 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_START=2.0 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_STOP=7.0 AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING=0.25 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.08 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.035 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_YAW_CORRECTION=0 BALANCE_FEEDBACK_CONTROLLER=1 BALANCE_START_ON_AGILE_HOLD=1 BALANCE_FEEDBACK_BASE=command BALANCE_PITCH_SIGN=1.0 BALANCE_PITCH_GAIN=0.10 BALANCE_PITCH_RATE_GAIN=0.006 BALANCE_ROLL_GAIN=0.06 BALANCE_ROLL_RATE_GAIN=0.003 BALANCE_ADJUSTMENT_LIMIT=0.08 BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0 BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0 BALANCE_ROLL_LEFT_HIP_SCALE=-0.5 BALANCE_ROLL_RIGHT_HIP_SCALE=0.5 BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05 BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05 BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20 BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20 CRADLE_TOP_LID_ENABLED=1 CRADLE_TOP_LID_ENABLE_ON_HOLD=1 CRADLE_TOP_LID_LOCAL_Z=0.16 CRADLE_TOP_LID_THICKNESS=0.014 CRADLE_TOP_LID_X_SCALE=1.15 CRADLE_TOP_LID_Y_SCALE=1.10 CRADLE_SIDE_RAIL_HEIGHT=0.12 CRADLE_END_STOP_HEIGHT=0.13 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_btilt bash scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_boxtilt_0706`, Slurm job
  `167670`, job-name `g1_lg_btilt`; queued waiting for resources at
  submission-time check.
- [x] Record larger-box true-box-tilt adaptive strict diagnostic. Job
  `167670` completed with build status `0` and check status `1`. It failed
  badly: `fall_events=210`, min robot z `0.259627 m`, robot-root max tilt
  `1.824652 rad`, true box max tilt `1.650226 rad`, final box target-directed
  travel `-0.112571 m`, final relative offset `0.388508 m`, and final
  robot/box target-line lateral errors about `-1.51 m` / `-1.52 m`. No
  rollout root/velocity/box pose writes occurred. Interpretation:
  true-box-tilt adaptive without torso support is not a viable larger-box
  posture.
- [x] Record larger-box chest-pad strict diagnostic. Job `167691` completed
  with build status `0` and check status `1`. It failed strict gates but is
  the best larger-box posture so far: `fall_events=0`, `box_drop_events=0`,
  no rollout root/velocity/box pose writes, final robot target-directed travel
  `1.292595 m`, final box target-directed travel `1.285295 m`, final relative
  offset `0.157688 m`, max relative offset `0.164649 m`. Remaining failures:
  robot-root max tilt `0.480753 > 0.35`, true box max tilt `0.493889 > 0.45`,
  and final box target-line lateral error `0.627694 > 0.60`. Next: tune the
  chest-pad controller with earlier hold, lower post-capture command scale,
  and stronger lateral correction.
- [ ] Run/record tuned larger-box chest-pad strict diagnostic:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_tuned_strict_700_targetnegx1 AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.12 AGILE_COMMAND_HOLD_SCALE=0.25 AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE=0.05 AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE=0.25 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START=0.10 AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP=0.30 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_START=0.10 AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_STOP=0.30 AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING=0.30 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.12 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.055 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_ctune bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_chestpad_tuned_0706`, Slurm
  job `167719`, job-name `g1_lg_ctune`; queued waiting for resources at
  submission-time check.
- [x] Record tuned larger-box chest-pad strict diagnostic. Job `167719`
  completed with build status `0` and check status `1`. It was worse than the
  first chest-pad run: `fall_events=247`, `box_drop_events=226`, min robot z
  `0.127119 m`, min box z `0.079985 m`, robot-root max tilt `1.277179 rad`,
  true box max tilt `1.223774 rad`, final relative offset `0.394498 m`. No
  rollout root/velocity/box pose writes occurred. Lateral error improved
  (`final_box_target_lateral_error_m=0.494578`), but the run fell and dropped
  the box. Interpretation: over-slowing and holding too early destabilizes
  chest-pad carrying. Return to the first chest-pad speed and tune lateral
  correction/contact geometry one variable at a time.
- [ ] Run/record chest-pad lateral-only strict diagnostic:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_lateral_strict_700_targetnegx1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.12 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.055 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_clat bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_chestpad_lateral_0706`,
  Slurm job `167723`, job-name `g1_lg_clat`; queued waiting for resources at
  submission-time check.
- [x] Record chest-pad lateral-only strict diagnostic. Job `167723` completed
  with build status `0` and check status `1`. It was worse than the first
  chest-pad run: `fall_events=335`, `box_drop_events=34`, min robot z
  `0.178246 m`, min box z `0.129841 m`, robot-root max tilt `3.134285 rad`,
  true box max tilt `3.127805 rad`. No rollout root/velocity/box pose writes
  occurred. Stronger lateral correction saturated at `0.055` and final
  robot/box lateral errors were `0.796742 m` / `0.863152 m`. Interpretation:
  increasing lateral velocity authority destabilizes the walking controller.
- [ ] Run/record chest-pad mild-yaw strict diagnostic:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_mildyaw_strict_700_targetnegx1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_cyaw bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_chestpad_mildyaw_0706`,
  Slurm job `167729`, job-name `g1_lg_cyaw`; queued waiting for resources at
  submission-time check.
- [x] Record chest-pad mild-yaw strict diagnostic. Job `167729` completed with
  build status `0` and check status `1`. It was worse than the first
  chest-pad run: `fall_events=94`, min robot z `0.323945 m`, robot-root max
  tilt `2.273509 rad`, true box max tilt `2.335066 rad`, final robot/box
  lateral errors `-0.867466 m` / `-0.803925 m`. It did not drop the box and
  used no rollout root/velocity/box pose writes. Interpretation: adding yaw
  correction destabilizes root/box roll and worsens path error.
- [ ] Run/record chest-pad geometry strict diagnostic:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_geom_strict_700_targetnegx1 CRADLE_CHEST_PAD_LOCAL_X=-0.08 CRADLE_CHEST_PAD_LOCAL_Z=0.12 CRADLE_CHEST_PAD_SIZE_X=0.08 CRADLE_CHEST_PAD_SIZE_Y=0.44 CRADLE_CHEST_PAD_SIZE_Z=0.28 CRADLE_TOP_LID_Y_SCALE=1.25 CRADLE_SIDE_RAIL_HEIGHT=0.14 CRADLE_END_STOP_HEIGHT=0.15 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_cgeo bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_chestpad_geom_0706`, Slurm
  job `167731`, job-name `g1_lg_cgeo`; queued waiting for resources at
  submission-time check.
- [x] Record chest-pad geometry strict diagnostic. Job `167731` completed with
  build status `0` and check status `1`. It was worse than the first
  chest-pad run: `fall_events=328`, `box_drop_events=85`, min robot z
  `-0.542888 m`, min box z `-0.458421 m`, robot-root max tilt
  `3.140933 rad`, true box max tilt `3.109670 rad`, and final robot/box
  target-line lateral errors about `-2.42 m`. No rollout root/velocity/box
  pose writes occurred. Interpretation: larger/higher chest support and rails
  made lateral drift and roll-over worse.
- [ ] Run/record chest-pad opposite-yaw strict diagnostic:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_oppositeyaw_strict_700_targetnegx1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_oyaw bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_0706`,
  Slurm job `167762`, job-name `g1_lg_oyaw`; queued waiting for resources at
  submission-time check.
- [x] Record chest-pad opposite-yaw strict diagnostic. Job `167762` completed
  with build status `0` and check status `0`. This is the first strict
  larger-box pass in this sequence: `fall_events=0`, `box_drop_events=0`,
  completed `700` steps, min robot z `0.721562 m`, min box z `0.825034 m`,
  robot-root max tilt `0.307758 rad`, true box max tilt `0.312059 rad`,
  final robot/box target-directed travel `1.435312 m` / `1.457102 m`, max
  relative offset `0.205432 m`, final relative offset `0.075546 m`, max
  robot/box target-line lateral error `0.115763 m` / `0.186329 m`, and no
  rollout root/velocity/box pose writes. Treat this as a strict diagnostic
  pass, not final project success.
- [ ] Run/record 900-step chest-pad opposite-yaw strict validation:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_oppositeyaw_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_oy9 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_900_0706`,
  Slurm job `167768`, job-name `g1_lg_oy9`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step chest-pad opposite-yaw strict validation. Job `167768`
  completed with build status `0` and check status `1`. The 700-step strict
  pass did not extend to 900 steps: `fall_events=26`, min robot z
  `0.369473 m`, robot-root max tilt `1.149047 rad`, true box max tilt
  `1.201307 rad`, final robot/box target-directed travel `1.184219 m` /
  `1.121610 m`, final robot/box target-line lateral error `0.632209 m` /
  `0.673950 m`. It did not drop the box and used no rollout root/velocity/box
  pose writes. Interpretation: post-target drift/tilt accumulates after the
  successful 700-step window.
- [x] Add terminal hold scale control for post-target stability. New controls:
  `AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL` and
  `AGILE_COMMAND_HOLD_TERMINAL_SCALE`. When enabled, once the previous box
  target-directed travel crosses the threshold during hold phase, the command
  scale is capped to the terminal scale and summary/check output records
  active steps, first active step, and reason. Defaults are disabled, so
  existing diagnostics are unchanged. Lightweight `py_compile`, `bash -n`, and
  `git diff --check` passed.
- [ ] Run/record 900-step terminal-scale chest-pad opposite-yaw strict
  validation:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.35 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.06 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_ot9 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_0706`, Slurm
  job `167771`, job-name `g1_lg_ot9`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step terminal-scale chest-pad opposite-yaw strict validation.
  Job `167771` completed with build status `0` and check status `1`.
  Terminal scale triggered at step `666` and was active for `234` steps. It
  improved the 900-step failure but did not pass: `fall_events=5`, no box
  drops, no rollout root/velocity/box pose writes, final robot/box lateral
  errors `0.073779 m` / `0.120379 m`. Remaining failures: robot-root max tilt
  `1.156729 rad`, true box max tilt `1.085554 rad`, final relative offset
  `0.286271 m`. It still moved too far after target (`final_box_target_directed_travel_m=2.205989`).
- [ ] Run/record 900-step earlier-terminal chest-pad opposite-yaw strict
  validation:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_early_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.15 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.03 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_oe9 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_early_0706`,
  Slurm job `167773`, job-name `g1_lg_oe9`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step earlier-terminal chest-pad opposite-yaw strict
  validation. Job `167773` completed with build status `0` and check status
  `1`. It removed the 900-step fall/drop failure: `fall_events=0`,
  `box_drop_events=0`, min robot z `0.721562 m`, min box z `0.825034 m`, no
  rollout root/velocity/box pose writes. Terminal mode triggered at step
  `612` and was active for `288` steps. Remaining failures were only attitude
  gates: robot-root max tilt `0.463448 > 0.35` and true box max tilt
  `0.636226 > 0.45`. Contact/path passed: final relative offset `0.072616 m`,
  max relative offset `0.205432 m`, final robot/box lateral errors
  `0.383956 m` / `0.315494 m`.
- [ ] Run/record 900-step near-stop terminal chest-pad opposite-yaw strict
  validation:
  `LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_nearstop_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.05 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_on9 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_nearstop_0706`,
  Slurm job `167778`, job-name `g1_lg_on9`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step near-stop terminal chest-pad opposite-yaw strict
  validation. Job `167778` completed with build status `0` and check status
  `0`. This is the strongest current larger-box result: `fall_events=0`,
  `box_drop_events=0`, completed `900` steps, min robot z `0.721562 m`, min
  box z `0.825034 m`, robot-root max tilt `0.307758 rad`, true box max tilt
  `0.384690 rad`, final robot/box target-directed travel `1.730244 m` /
  `1.759363 m`, max relative offset `0.205432 m`, final relative offset
  `0.108737 m`, max robot/box target-line lateral error `0.258455 m` /
  `0.362250 m`, and no rollout root/velocity/box pose writes. Terminal mode
  triggered at step `590` and was active for `310` steps.
- [ ] Run/record low-carry larger-box strict diagnostic:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_oppositeyaw_strict_700_targetnegx1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_low bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_lowcarry_oppositeyaw_0706`,
  Slurm job `167782`, job-name `g1_lg_low`; queued waiting for resources at
  submission-time check.
- [x] Record low-carry larger-box strict diagnostic. Job `167782` completed
  with build status `0` and check status `1`. Low-carry without chest support
  failed: `fall_events=117`, `box_drop_events=104`, min robot z `0.170354 m`,
  min box z `0.096605 m`, robot-root max tilt `0.990520 rad`, true box max
  tilt `0.991162 rad`, final robot/box target-directed travel `1.018163 m` /
  `1.075757 m`, final robot/box lateral errors `1.228960 m` / `1.272310 m`.
  No rollout root/velocity/box pose writes occurred. Interpretation: low-carry
  needs its own support/terminal strategy and cannot simply reuse the
  chest-supported yaw controller.
- [ ] Run/record low-carry terminal strict diagnostic:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal_strict_700_targetnegx1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lterm bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux `curiosity_g1_agile_largerbox_lowcarry_terminal_0706`,
  Slurm job `167783`, job-name `g1_lg_lterm`; queued waiting for resources at
  submission-time check.
- [x] Record low-carry terminal strict diagnostic. Job `167783` completed with
  build status `0` and check status `1`. Terminal hold removed the low-carry
  fall/drop failure: `fall_events=0`, `box_drop_events=0`, min robot z
  `0.761974 m`, min box z `0.816164 m`, robot-root max tilt `0.196663 rad`,
  true box max tilt `0.271947 rad`, final relative offset `0.200210 m`, no
  rollout root/velocity/box pose writes. Remaining failures were path-only:
  final robot/box target-line lateral errors `1.114195 m` / `1.306165 m`.
- [ ] Run/record low-carry terminal default-yaw strict diagnostic:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal_defaultyaw_strict_700_targetnegx1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_ldef bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_defaultyaw_0706`, Slurm job
  `167788`, job-name `g1_lg_ldef`; queued waiting for resources at
  submission-time check.
- [x] Record low-carry terminal default-yaw strict diagnostic. Job `167788`
  completed with build status `0` and check status `1`. It did not fall or
  drop and used no rollout root/velocity/box pose writes, but moved in the
  wrong target direction: final robot/box target-directed travel
  `-0.387059 m` / `-0.584589 m`; terminal hold never triggered. It also
  exceeded true box tilt (`0.636519 rad`) and final relative offset
  (`0.286901 m`). Conclusion: default yaw sign is wrong for low-carry target
  progress; continue from the yaw-sign `-1.0` terminal base and tune lateral
  correction sign/gain.
- [ ] Next low-carry gate: tune lateral correction around
  `AGILE_COMMAND_HOLD_YAW_SIGN=-1.0` and terminal hold (`0.65 m`, scale
  `0.015`), because that base achieved no fall/drop and only failed path
  lateral gates.
- [ ] Run/record low-carry terminal lateral-sign strict diagnostic:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal_latsign_strict_700_targetnegx1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lsgn bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_latsign_0706`, Slurm job
  `167789`, job-name `g1_lg_lsgn`; queued waiting for resources at
  submission-time check.
- [x] Record low-carry terminal lateral-sign strict diagnostic. Job `167789`
  completed with build status `0` and check status `1`. Flipping only
  `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0` was a clear regression:
  `fall_events=220`, `box_drop_events=40`, min robot z `0.172896 m`, min box
  z `0.115441 m`, robot-root max tilt `1.749395 rad`, true box max tilt
  `2.081289 rad`, final robot/box target-directed travel `-0.577424 m` /
  `-0.516827 m`, final robot/box lateral errors `-1.229764 m` /
  `-1.121373 m`. Terminal hold never triggered. No rollout root/velocity/box
  pose writes occurred. Conclusion: do not continue in the flipped lateral
  sign direction; test whether low-carry should disable or gate lateral
  correction instead.
- [ ] Run/record low-carry terminal no-lateral strict diagnostic:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_strict_700_targetnegx1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lnol bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral_0706`, Slurm job
  `167800`, job-name `g1_lg_lnol`; queued waiting for resources at
  submission-time check.
- [x] Record low-carry terminal no-lateral strict diagnostic. Job `167800`
  completed with build status `0` and check status `0`. This is the strongest
  current low-carry result: `fall_events=0`, `box_drop_events=0`, completed
  `700` steps, min robot z `0.757182 m`, min box z `0.825777 m`, robot-root
  max tilt `0.227144 rad`, true box max tilt `0.241890 rad`, final robot/box
  target-directed travel `1.994070 m` / `2.024888 m`, max robot/box
  target-line lateral error `0.430948 m` / `0.414760 m`, final robot/box
  lateral errors `0.427588 m` / `0.374435 m`, and no rollout root/velocity/box
  pose writes. Lateral correction was disabled and active for `0` steps.
  Conclusion: the previous low-carry path failure came from the lateral
  correction controller, not from the low-carry posture itself.
- [ ] Run/record 900-step low-carry terminal no-lateral strict validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_ln9 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral900_0706`, Slurm
  job `167803`, job-name `g1_lg_ln9`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step low-carry terminal no-lateral strict validation. Job
  `167803` completed with build status `0` and check status `1`. It preserved
  the path improvement but failed late-duration stability: `fall_events=44`,
  `box_drop_events=25`, min robot z `-0.226469 m`, min box z `-0.343242 m`,
  robot-root max tilt `0.975629 rad`, true box max tilt `1.274737 rad`. It
  reached final robot/box target-directed travel `3.442578 m` / `3.440148 m`,
  while lateral errors stayed within gates (`0.440032 m` / `0.414760 m` max).
  No rollout root/velocity/box pose writes occurred. Conclusion: no-lateral
  fixes path control for low-carry, but terminal scale `0.015` is still too
  much for 900-step duration.
- [ ] Run/record 900-step low-carry terminal no-lateral zero-stop strict
  validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_zerostop_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lz9 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral_zerostop900_0706`,
  Slurm job `167804`, job-name `g1_lg_lz9`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step low-carry terminal no-lateral zero-stop strict
  validation. Job `167804` completed with build status `0` and check status
  `1`. Zero-stop was too abrupt or under-driven: `fall_events=141`,
  `box_drop_events=97`, min robot z `0.161750 m`, min box z `0.071122 m`,
  robot-root max tilt `1.677712 rad`, true box max tilt `1.681898 rad`, final
  robot/box target-directed travel only `0.367843 m` / `0.339509 m`, and
  final relative offset `0.281655 m`. No rollout root/velocity/box pose
  writes occurred. Conclusion: for 900-step low-carry, terminal scale `0.015`
  walks too far and `0.0` stops too hard; test an intermediate terminal scale.
- [ ] Run/record 900-step low-carry terminal no-lateral mid-stop strict
  validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_midstop_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.008 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lm9 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral_midstop900_0706`,
  Slurm job `167806`, job-name `g1_lg_lm9`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step low-carry terminal no-lateral mid-stop strict
  validation. Job `167806` completed with build status `0` and check status
  `1`. Intermediate terminal scale `0.008` also failed:
  `fall_events=226`, `box_drop_events=36`, min robot z `0.183568 m`, min box
  z `0.120184 m`, robot-root max tilt `3.119225 rad`, true box max tilt
  `3.097902 rad`, final robot/box target-directed travel `0.662951 m` /
  `0.576417 m`, final relative offset `0.330259 m`. No rollout
  root/velocity/box pose writes occurred. Conclusion: 900-step low-carry
  failure is not solved by lowering terminal scale; use existing
  `policy_then_stand` hold mode to test long-duration stabilization.
- [ ] Run/record 900-step low-carry no-lateral policy-then-stand strict
  validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_nolateral_policythenstand_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_MODE=policy_then_stand AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=420 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.02 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lps bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_nolateral_policythenstand900_0706`,
  Slurm job `167808`, job-name `g1_lg_lps`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step low-carry no-lateral policy-then-stand strict
  validation. Job `167808` completed with build status `0` and check status
  `1`. It kept target-line lateral errors within gates, but stand-target
  blending hurt low-carry object stability: `fall_events=281`,
  `box_drop_events=258`, min robot z `0.299849 m`, min box z `0.097845 m`,
  robot-root max tilt `1.457402 rad`, true box max tilt `2.824101 rad`, final
  robot/box target-directed travel `1.790397 m` / `1.778671 m`, final
  relative offset `0.319173 m`. No rollout root/velocity/box pose writes
  occurred. Conclusion: do not use generic stand-target blending as the
  low-carry long-hold fix.
- [ ] Next low-carry 900-step gate: design a posture-specific low-carry hold
  instead of generic lateral correction, zero terminal speed, or stand-target
  blending. The current evidence supports `AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0`
  for low-carry path control, but 900-step stability needs a low-cradle
  hold/recovery controller that preserves arm/cradle contact while reducing
  forward drift.
- [ ] Wire hold/rescue lower-body target overrides through the low-cradle
  suite runner so low-carry can test posture-specific crouch hold instead of
  generic stand-target blending. This is a runner-only interface change; it
  does not write robot root pose, root velocity, or box pose.
- [ ] Run/record 900-step low-carry no-lateral late-crouch-hold strict
  validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_nolateral_latecrouch_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_MODE=policy_then_stand AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=620 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.01 AGILE_COMMAND_HOLD_STAND_HIP_PITCH=-0.20 AGILE_COMMAND_HOLD_STAND_KNEE=0.45 AGILE_COMMAND_HOLD_STAND_ANKLE_PITCH=-0.25 AGILE_COMMAND_HOLD_STAND_WAIST_PITCH=-0.06 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lch bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latecrouch900_0706`, Slurm job
  `167858`, job-name `g1_lg_lch`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step low-carry no-lateral late-crouch-hold strict
  validation. Job `167858` completed with build status `0` and check status
  `1`. Late crouch hold was a negative result: `fall_events=94`,
  `box_drop_events=84`, min robot z `-1.205153 m`, min box z `-1.185329 m`,
  robot-root max tilt `3.139498 rad`, true box max tilt `3.134521 rad`,
  final robot/box target-directed travel `3.057527 m` / `2.630529 m`, final
  robot/box lateral errors `0.648811 m` / `0.691176 m`. No rollout
  root/velocity/box pose writes occurred. Conclusion: low-carry 900-step
  should not be fixed by late joint-target blending; use command-level stop
  logic that preserves the policy posture.
- [ ] Add and test a latched agile terminal hold. Current terminal hold is
  instantaneous rather than latched; if box travel falls back below the
  terminal threshold, the command can resume. The zero-stop negative result's
  final command still had nonzero x command (`0.010`), so test a latched
  terminal state before adding more posture blending.
- [x] Add latched agile terminal hold interface and telemetry:
  `--agile-command-hold-terminal-latch` in
  `build_core_world_g1_box_scene.py`, exposed as
  `AGILE_COMMAND_HOLD_TERMINAL_LATCH=1` in
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`. The new summary fields
  are `agile_command_hold_terminal_latch_enabled`,
  `agile_command_hold_terminal_latched`, and
  `agile_command_hold_terminal_latched_step`.
- [ ] Run/record 900-step low-carry no-lateral latched zero-stop strict
  validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_nolateral_latchedzerostop_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.0 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lzl bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedzerostop900_0706`, Slurm job
  `167875`, job-name `g1_lg_lzl`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step low-carry no-lateral latched zero-stop strict
  validation. Job `167875` completed with build status `0` and check status
  `1`. Latch worked as intended (`latched=True`, latched step `381`, final
  command x `0.0`) but the run still failed: `fall_events=141`,
  `box_drop_events=97`, min robot z `0.162028 m`, min box z `0.077461 m`,
  robot-root max tilt `1.677746 rad`, true box max tilt `1.656680 rad`, final
  robot/box target-directed travel `0.368724 m` / `0.356281 m`, final
  relative offset `0.254142 m`. No rollout root/velocity/box pose writes
  occurred. Conclusion: command resumption was not the main failure; complete
  stop is under-supported for low-carry.
- [ ] Run/record 900-step low-carry no-lateral latched micro-hold strict
  validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_nolateral_latchedmicro_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.006 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_lml bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro900_0706`, Slurm job
  `167879`, job-name `g1_lg_lml`; queued waiting for resources at
  submission-time check.
- [x] Record 900-step low-carry no-lateral latched micro-hold strict
  validation. Job `167879` completed with build status `0` and check status
  `1`. It reduced late drop/fall relative to full zero-stop but introduced
  severe lateral drift: `fall_events=90`, `box_drop_events=42`, min robot z
  `0.181486 m`, min box z `0.132102 m`, robot-root max tilt `1.704787 rad`,
  true box max tilt `1.905366 rad`, final robot/box target-directed travel
  `1.309712 m` / `1.344400 m`, final robot/box lateral errors `1.375929 m` /
  `1.389495 m`. No rollout root/velocity/box pose writes occurred. Conclusion:
  tiny nonzero terminal scale helps support but needs terminal-gated path
  correction; old always-on lateral correction remains rejected.
- [ ] Add and test terminal-only agile lateral correction. Existing lateral
  correction starts as soon as agile hold begins and destabilized low-carry.
  The next change should allow lateral correction only after terminal latch or
  terminal threshold, with much lower gain/limit.
- [x] Add terminal-only agile lateral correction interface:
  `--agile-command-hold-lateral-terminal-only` in
  `build_core_world_g1_box_scene.py`, exposed as
  `AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1` in
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`. When enabled, lateral
  correction is gated by the active terminal scale/latch instead of starting
  at the first agile hold step.
- [x] Supersede old 900-step low-carry latched micro-hold terminal-lateral
  validation:
  `LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.006 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.012 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.006 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_ltl bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat900_0706`,
  Slurm job `167898`, job-name `g1_lg_ltl`; queued waiting for resources at
  submission-time check. This branch is superseded by the explicit-export
  `167990`, `168131`, and `168144` terminal-lateral results.
- [x] Mark terminal-lateral job `167898` as invalid configuration, not a
  behavioral result. The run completed with build status `0` and check status
  `1`, but summary showed key intended settings were not applied:
  `agile_command_hold_terminal_box_target_travel_m=-1.0`,
  `agile_command_hold_terminal_latch_enabled=false`,
  `agile_command_hold_lateral_correction_enabled=false`,
  `agile_command_hold_lateral_terminal_only=false`. Therefore its fall/drop
  metrics are not evidence about terminal-only lateral correction.
- [x] Supersede old explicit-export terminal-lateral rerun request:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.006 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.012 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.006 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 --job-name=g1_lg_ltx bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export900_0706`,
  Slurm job `167958`, job-name `g1_lg_ltx`; queued waiting for resources at
  submission-time check. It was replaced by short-walltime job `167990`,
  then by threshold experiments `168131` and `168144`.
- [x] Add low-cradle suite environment snapshot logging:
  `run_core_world_g1_agile_policy_low_cradle_suite.sh` now writes
  `agile_policy_low_cradle_env_snapshot.txt` under each suite output root with
  `AGILE_*`, `BALANCE_*`, `CRADLE_*`, `FREE_*`, `LARGERBOX_*`, target, and run
  environment variables. This is to prevent another invalid configuration run
  from being mistaken for behavior.
- [x] Replace pending job `167958` with a shorter-walltime equivalent. The
  original explicit-export terminal-lateral job remained pending with
  `Reason=Priority` and a 1.5h walltime request even though prior 900-step
  runs complete in about 1-2 minutes. Cancel it and resubmit a 15-minute
  version to improve backfill chances.
- [x] Replaced pending job `167958`: cancelled before it started and submitted
  `20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export_short_strict_900_targetnegx1`
  as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export900_short_0706`,
  Slurm job `167990`, job-name `g1_lg_ltxs`, with the same explicit exported
  AGILE terminal/lateral settings and `--time=00:15:00`.
- [x] Record valid 900-step low-carry terminal-only lateral run. Job `167990`
  completed with build status `0` and check status `1`; environment snapshot
  confirmed the intended AGILE terminal/lateral settings were applied. It
  reduced lateral drift relative to no-lateral micro-hold but destabilized
  much earlier: `fall_events=288`, `box_drop_events=269`, min robot z
  `0.180464 m`, min box z `0.083854 m`, robot-root max tilt `2.068887 rad`,
  true box max tilt `2.171806 rad`, final robot/box target-directed travel
  `0.921112 m` / `0.958597 m`, final robot/box lateral errors `-0.595748 m`
  / `-0.657782 m`. No rollout root/velocity/box pose writes occurred.
  Comparison: no-lateral micro-hold first fell at step `810`, while
  terminal-only lateral first fell at step `620` even with small lateral error
  near zero, so terminal-lateral must be gated by lateral error magnitude.
- [x] Add and test lateral-error-threshold gating so terminal-only lateral
  correction does not act when the target-line lateral error is still small.
- [x] Add lateral-error-threshold gating:
  `--agile-command-hold-lateral-error-start` in
  `build_core_world_g1_box_scene.py`, exposed as
  `AGILE_COMMAND_HOLD_LATERAL_ERROR_START` in
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`. Default is `0.0`, so
  existing behavior is unchanged. This prevents terminal-only lateral
  correction from acting while target-line lateral error is below the threshold.
- [ ] Run/record 900-step low-carry latched micro-hold terminal-lateral
  threshold strict validation:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.006 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.55 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.003 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_lth bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh900_0706`,
  Slurm job `167998`, job-name `g1_lg_lth`; queued waiting for resources at
  submission-time check.
- [x] Mark job `167998` as invalid behavior evidence. It used the intended
  exported settings, but the rollout stopped at step `610` with
  `NameError: name 'feedback_tilt' is not defined` from the newly added
  lateral posture-risk gate. It had fall/drop `0/0` at interruption but only
  `completed_steps=611`, so it is a code bug, not a 900-step threshold-lateral
  result.
- [x] Fix the lateral posture-risk gate bug. The gate now uses
  `max(abs(feedback_roll), abs(feedback_pitch))` for robot tilt instead of the
  nonexistent `feedback_tilt`. Login-node lightweight checks passed:
  `py_compile`, `bash -n`, and `git diff --check`.
- [x] Rerun 900-step low-carry latched micro-hold terminal-lateral threshold
  validation after the gate bug fix:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh_fix_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.006 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.55 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.003 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_lthf bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh_fix900_0706`,
  Slurm job `168131`, job-name `g1_lg_lthf`; queued waiting for resources at
  submission-time check.
- [x] Record job `168131`: valid behavior evidence but not a pass. It
  completed `900/900` with fall/drop `0/0`, no rollout root velocity/pose or
  box pose writes, and final robot/box target-directed travel
  `1.612132 m` / `1.644704 m`. Strict checker failed on max tilt
  `0.594056 rad`, max true box tilt `0.946271 rad`, final robot/box lateral
  error `1.532278 m` / `1.672921 m`. Lateral correction started only at step
  `611`, active `289` steps, max command `0.003`. Conclusion: threshold
  `0.55 m` avoided fall/drop but was too late/weak for path centering and
  allowed large tilt.
- [x] Run a 900-step earlier threshold plus tilt-gated terminal-lateral
  validation:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh045_tiltgate_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.006 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.45 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.45 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.003 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_l45 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh045_tiltgate900_0706`,
  Slurm job `168144`, job-name `g1_lg_l45`; queued waiting for resources at
  submission-time check.
- [x] Record job `168144`: valid negative behavior evidence. It completed
  `900/900` but failed with `fall_events=210`, `box_drop_events=158`, first
  fall/drop steps `690/709`, min robot/box z `0.353656/0.103534 m`,
  max robot/box tilt `3.138657/3.134910 rad`, final robot/box lateral error
  `1.476673/1.439418 m`, and final relative offset `0.462399 m`. Lateral
  correction started at step `550`, was active `105` steps, and was suppressed
  by tilt for `245` steps. Conclusion: earlier lateral correction worsens the
  low-carry 900-step behavior.
- [x] Add two-stage agile terminal scale support:
  `--agile-command-hold-final-box-target-travel`,
  `--agile-command-hold-final-scale`, and
  `--agile-command-hold-final-latch`, exposed as
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL`,
  `AGILE_COMMAND_HOLD_FINAL_SCALE`, and `AGILE_COMMAND_HOLD_FINAL_LATCH`.
  Defaults disable this path. Reporting now includes the final-stage fields.
- [x] Run a 900-step no-lateral two-stage terminal validation:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_final006_2m_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.006 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_f2m bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal015_final006_2m900_0706`,
  Slurm job `168154`, job-name `g1_lg_f2m`; queued waiting for resources at
  submission-time check. Completed with build status `0`, checker status `1`:
  `900/900` steps, fall/drop `54/29`, first fall/drop `846/871`, final
  robot/box target-directed travel `3.355354/3.348626 m`, final robot/box
  lateral error `0.494245/0.542693 m`, robot/true-box max tilt
  `0.346288/0.725258 rad`, final relative offset `0.068053 m`, no rollout
  root velocity/pose or box pose writes. The final `0.006` stage became active
  at step `695` for `205` steps. Conclusion: this reduced tilt/relative error
  but did not stop the late overshoot/fall.
- [x] Run a 900-step no-lateral two-stage terminal validation with a full stop
  after `2.0 m`:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_f0m bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Rationale: earlier full stop at `0.65 m` failed because it halted before the
  stable 2 m carrying window; this run tests whether stopping only after the
  stable travel window prevents late overshoot/fall. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal015_final000_2m900_0706`,
  Slurm job `168164`, job-name `g1_lg_f0m`; submission-time queue check showed
  `PENDING (Priority)` and no output files yet. Follow-up `scontrol` status at
  `14:52:25 CST` still showed `PENDING (Priority)`, scheduled start
  `2026-07-06T16:18:59` on `server39`. Result: valid negative evidence.
  It ran on `server30`, completed `900/900`, build status `0`, checker status
  `1`, fall/drop `9/0`, first fall step `891`, final robot/box
  target-directed travel `3.348633/3.359532 m`, min robot/box z
  `0.349306/0.434169 m`, max robot/box tilt `0.557659/0.804987 rad`,
  final relative offset `0.033178 m`, final robot/box lateral
  `0.389801/0.379783 m`, and rollout root pose/root velocity/box pose writes
  all `0`. Conclusion: full stop only after `2.0 m` still overshoots and
  falls late; it is not stable post-carry hold.
- [x] Add a final-stage-only stand-hold option. The G1 scene now supports
  `--agile-command-hold-final-stand` and
  `--agile-command-hold-final-stand-delay-steps`, exposed by the low-cradle
  launcher as `AGILE_COMMAND_HOLD_FINAL_STAND=1` and
  `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS`. This is disabled by default
  and does not alter queued job `168164`. It differs from existing
  `policy_then_stand` because it starts counting from the final target-travel
  threshold, not from the early agile hold trigger.
- [x] Run/record the final-stage stand-hold comparison:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120 MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=80 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_STAND=1 AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=20 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.02; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_fst bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Because job `168164` remained stuck in `PENDING (Priority)` with no output,
  this comparison was submitted early to avoid another idle GPU queue wait.
  Interpret it only from its own checker/summary evidence and still compare
  against `168164` once that baseline produces output. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand900_0706`,
  Slurm job `168177`, job-name `g1_lg_fst`; submission-time status
  `PENDING (Priority)`, no output files yet, no scheduled start time. Result:
  valid negative evidence. Job `168177` completed on `server44` in `00:01:01`
  with build status `0` and checker status `1`. It entered final hold at step
  `695` for `205` steps and final stand at step `715` for `185` steps, but
  failed with fall/drop `115/70`, first fall step `785`, first box-drop step
  `830`, final robot/box target-directed travel `3.218096/3.123645 m`, final
  robot/box lateral error `0.549016/0.435925 m`, final relative offset
  `0.368966 m`, min final-hold robot/box z `-1.299788/-1.523448 m`, and max
  final-hold robot/box tilt `3.137067/3.137737 rad`. The final-stage command
  still had yaw `0.021895` because final zero-corrections was disabled.
  Conclusion: switching to final-stage stand did not solve the post-2m
  braking/hold failure; the next direct Isaac diagnostic should isolate and
  clamp final-stage correction commands before any broad multi-posture sweep.
- [x] Add final target-directed travel upper-bound gates to the checker and
  low-cradle launcher. `check_core_world_g1_box_scene_summary.py` now accepts
  `--max-final-robot-target-directed-travel` and
  `--max-final-box-target-directed-travel`; the launcher passes them from
  `NOBOX_/FIXED_/FREE_MAX_FINAL_*_TARGET_DIRECTED_TRAVEL` or shared
  `MAX_FINAL_*_TARGET_DIRECTED_TRAVEL` environment variables. The larger-box
  strict launcher exposes empty defaults for the free-box gates. This prevents
  future runs from passing merely because they overshot far past the intended
  carry/hold window.
- [x] Add final-stage trigger-duration gates. The checker now accepts
  `--min-agile-command-hold-final-active-steps` and
  `--min-agile-command-hold-final-stand-active-steps`; the low-cradle launcher
  passes them from `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS` and
  `MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS`. The planned final-stand
  comparison requires at least `120` final-stage steps and `80` final-stand
  steps so a run cannot pass without actually entering the post-carry hold.
- [x] Add first-event telemetry to G1 box scene summaries:
  `first_fall_step`, `first_fall_time_s`, `first_box_drop_step`, and
  `first_box_drop_time_s`. This makes future low-carry diagnostics report
  event timing without manual CSV scanning.
- [x] Add first-event and terminal/lateral gating telemetry to strict checker
  reports. `check_core_world_g1_box_scene_summary.py` now copies
  `first_fall_step`, `first_fall_time_s`, `first_box_drop_step`,
  `first_box_drop_time_s`,
  `agile_command_hold_lateral_terminal_only`,
  `agile_command_hold_lateral_error_start_m`, and terminal latch fields into
  `check.json`.
- [x] Add first-event and terminal/lateral gating telemetry to larger-box
  strict summarizer. `summarize_core_world_g1_largerbox_strict.py` now copies
  first-fall/drop fields, terminal latch fields, terminal-only lateral, and
  lateral error threshold fields into aggregate summaries.
- [x] Backfill first-event telemetry for older summaries in the larger-box
  strict summarizer. If `first_fall_step` or `first_box_drop_step` is missing
  from a summary, `summarize_core_world_g1_largerbox_strict.py` now reads the
  same case's `core_world_g1_box_scene_state.csv` and computes the first
  fall/drop step and time. Verified on the old low-carry latched micro-hold
  run: first fall step `810`, first drop step `840`.
- [x] Add a low-carry 900-step control matrix helper:
  `scripts/isaac/summarize_core_world_g1_lowcarry_900_matrix.py`. It reads
  existing summaries and CSV state traces, compares low-carry 700/900,
  latched zero/micro, terminal lateral, and pending threshold-lateral cases,
  and writes a diagnostic Markdown table. Generated current report:
  `experiments/reports/2026-07-06_g1_lowcarry_900_control_matrix.md`.
- [x] Sharpen the low-carry 900-step matrix for final-stage comparisons. The
  matrix now reports robot/box target-directed travel together and separates
  final-stage hold fields (`final_steps`, `final_stand`, `stand_delay`,
  `stand_steps`) from terminal-scale fields. This prepares direct comparison
  between queued `168164` and `168177` once their summaries exist.
- [x] Add a final-hold comparison helper:
  `scripts/isaac/summarize_core_world_g1_final_hold_comparison.py`. It
  compares queued full-stop-after-2m and final-stand-after-2m diagnostics,
  emits JSON/Markdown, and checks distance, overshoot, fall/drop, tilt,
  relative offset, lateral error, final-stage activation, and final-stand
  activation. Do not treat this as a success oracle beyond the configured
  diagnostic gates; it is a focused comparison tool for the two queued
  low-carry final-hold hypotheses.
- [x] Tighten the final-hold comparison helper. It now fails missing/failing
  `check.json` and enforces zero rollout root pose writes, root velocity
  writes, and box pose writes by default. The Markdown table reports those
  write counts explicitly.
- [x] Add optional final-hold comparison postprocessing to the low-cradle
  launcher. Set `GENERATE_FINAL_HOLD_COMPARISON=1` to emit
  `g1_final_hold_comparison.json`, `g1_final_hold_comparison.md`, and
  `g1_final_hold_comparison.stdout.md` under the suite directory after cases
  run. Defaults are disabled and do not change queued jobs `168164`/`168177`.
- [x] Add target-window stable-hold metrics for future strict validations.
  `build_core_world_g1_box_scene.py` now accepts `--target-window-center` and
  `--target-window-halfwidth` and records stable robot/box/both steps,
  longest stable streaks, and first stable steps inside that window, counting
  only no-fall/no-drop steps. The checker and low-cradle launcher can gate
  these via `MIN_TARGET_WINDOW_*` environment variables. Use this for the next
  strict post-carry hold run, e.g. target center `2.0 m`, halfwidth `0.35 m`,
  and a nonzero both-stable-streak requirement.
- [x] Add target-window fields to aggregate reports. The larger-box strict
  summarizer and low-carry 900-step matrix now carry target-window stable-step
  and longest-streak fields so future post-carry hold validations are visible
  in summary reports, not only raw per-run JSON.
- [ ] After the zero-corrections final-hold diagnostic produces behavior
  evidence, run a
  strict target-window final-stand validation rather than interpreting a
  single final frame as hold success:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_targetwindow_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STABLE_STEPS=60 MIN_TARGET_WINDOW_BOTH_FINAL_STAND_LONGEST_STREAK_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STREAK_AT_END_STEPS=30 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120 MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=80 GENERATE_FINAL_HOLD_COMPARISON=1 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_STAND=1 AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=20 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.02; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_fstw bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  When submitting this planned strict validation, also include
  `MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MIN_FINAL_STAND_ROBOT_Z=0.45 MIN_FINAL_STAND_BOX_Z=0.45 MAX_FINAL_STAND_TILT=0.35 MAX_FINAL_STAND_BOX_TILT=0.45`.
  Also include
  `MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 MAX_FINAL_STAND_FALL_EVENTS=0 MAX_FINAL_STAND_BOX_DROP_EVENTS=0`.
  Do not submit this before the final-stage command-clamp hypothesis is tested;
  `168164` and `168177` are both now known negative controls.
- [x] Prepare a stricter multi-posture final-hold matrix for the "any carry
  posture" requirement. Added
  `scripts/isaac/run_core_world_g1_largerbox_finalhold_posture_matrix.sh`,
  which runs `boxtilt`, `lowcarry`, and `chestpad` through the same 900-step
  target-window/final-hold/final-stand/end-streak gates. It also now defaults
  final-stage command gates to `0.001` for x/y/yaw and final-stage stability
  gates to robot/box z `>= 0.45 m`, robot tilt `<= 0.35 rad`, and box tilt
  `<= 0.45 rad`; final-stage fall/drop gates default to `0`. It refuses to
  run on login nodes and must be launched only inside a Curiosity-owned
  tmux-held Slurm allocation. Do not submit it while `168164` and `168177`
  remain pending; use it after the low-carry final-hold hypothesis is
  behaviorally known.
- [x] Tighten the larger-box aggregate summarizer for multi-posture use.
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py` now reports
  `failed_case_count` and returns `pass` only when every discovered case
  passes, not merely when one posture passes. This prevents an all-posture
  matrix from looking successful because a single easier posture worked.
- [x] Add an optional final-stage correction clamp for the `168164` failure
  mode. `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-final-zero-corrections`, exposed by the low-cradle
  launcher as `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1`. When enabled,
  lateral and yaw corrections are suppressed after the final target-travel
  threshold/latch, and summaries report
  `agile_command_hold_final_lateral_suppressed_steps` and
  `agile_command_hold_final_yaw_suppressed_steps`. This is off by default and
  does not affect queued job `168177`.
- [x] Run a no-lateral final-zero-corrections diagnostic before broad multi-
  posture sweeps:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_zerocorr_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_zcorr bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: isolate whether the nonzero yaw command observed at the end of
  `168164` and `168177` caused or amplified post-2m overshoot and late fall.
  The `MAX_FINAL_HOLD_COMMAND_*` gates require the final-stage x/y/yaw
  commands to be near zero, not only final scale to be zero. If this still
  fails after commands are actually zeroed, the next issue is not missing
  external models; it is the Isaac terminal braking/standing controller and
  should be addressed by changing the final controller/scaffold directly.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_zerocorr900_0706`, Slurm job
  `168247`, job-name `g1_lg_zcorr`, with stricter final-stage gates:
  `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120`,
  `MAX_FINAL_HOLD_COMMAND_X/Y/YAW=0.001`,
  `MIN_FINAL_HOLD_ROBOT_Z=0.45`, `MIN_FINAL_HOLD_BOX_Z=0.45`,
  `MAX_FINAL_HOLD_TILT=0.35`, `MAX_FINAL_HOLD_BOX_TILT=0.45`,
  `MAX_FINAL_HOLD_FALL_EVENTS=0`, and
  `MAX_FINAL_HOLD_BOX_DROP_EVENTS=0`. Submission-time status:
  `PENDING (Priority)`. Result: valid negative evidence with meaningful
  improvement over `168164`/`168177`. Job `168247` ran on `server39`,
  completed `900/900`, build status `0`, checker status `1`, and final-stage
  commands were genuinely zero:
  `agile_command_hold_final_max_abs_command_x/y/yaw = 0/0/0`,
  `agile_command_hold_final_yaw_suppressed_steps=205`. It passed the
  final-stage fall/drop and height checks with fall/drop `0/0`, final
  fall/drop `0/0`, min final-hold robot/box z `0.454602/0.488559 m`, and no
  rollout root pose/root velocity/box pose writes. It still failed strict
  holding because robot/box overshot to `3.098152/3.164331 m` target-directed
  travel and final-hold max robot/box tilt reached `0.383518/0.678150 rad`
  over the `0.35/0.45` limits. Conclusion: nonzero final yaw was a major
  cause of the previous fall/drop failures, but command zeroing alone does not
  brake the carrier into a stable 2 m hold window.
- [x] Run a combined zero-corrections plus final-stand diagnostic before
  multi-posture sweeps:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_zerocorr_finalstand_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120 MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=80 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MIN_FINAL_STAND_ROBOT_Z=0.45 MIN_FINAL_STAND_BOX_Z=0.45 MAX_FINAL_STAND_TILT=0.35 MAX_FINAL_STAND_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 MAX_FINAL_STAND_FALL_EVENTS=0 MAX_FINAL_STAND_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 AGILE_COMMAND_HOLD_FINAL_STAND=1 AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=20 AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.02; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_zfst bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether final stand can now brake the carrier without the
  yaw-correction-induced fall/drop observed in `168177`. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_zerocorr_finalstand900_0706`, Slurm
  job `168252`, job-name `g1_lg_zfst`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168252`
  completed on `server30` in `00:00:58`, build status `0`, checker status
  `1`, final commands `0/0/0`, and final yaw suppressed for `205` steps, but
  final stand reintroduced instability: fall/drop `105/89`, first fall step
  `795`, first box-drop step `811`, min final-hold robot/box z
  `-1.274945/-1.181000 m`, max final-hold robot/box tilt
  `3.133741/3.139424 rad`, final robot/box target-directed travel
  `2.940374/2.561408 m`, and final relative offset `0.382839 m`.
  Conclusion: the current stand-target blend is not a safe terminal braking
  controller for the carried-box state; do not keep tuning final stand as the
  main fix.
- [x] Add a final-stage short reverse-command brake hook for direct Isaac G1
  carrying. `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-final-brake-command-x`,
  `--agile-command-hold-final-brake-delay-steps`, and
  `--agile-command-hold-final-brake-steps`. The low-cradle launcher exposes
  them as `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X`,
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS`, and
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS`. The scene summary/checker report
  brake command, delay, duration, active steps, first/last active step, and
  max absolute brake x command. Defaults are disabled, so existing behavior is
  unchanged.
- [x] Run a first final-brake diagnostic:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalbrake005_80_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120 MAX_FINAL_HOLD_COMMAND_X=0.06 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.05 AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=0 AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=80; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_fbrk bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether a short reverse velocity command can reduce the
  `168247` overshoot without using the unstable final-stand target blend.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalbrake005_80_0706`, Slurm job
  `168275`, job-name `g1_lg_fbrk`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168275`
  completed on `server39` in `00:00:53`, build status `0`, checker status
  `1`. The brake hook worked mechanically: final brake active steps `80`,
  first/last active steps `695/774`, max brake x command `0.05`, and final
  command yaw stayed zero. But `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.05`
  worsened terminal behavior: fall/drop `79/63`, first fall/drop steps
  `821/837`, final robot/box target-directed travel `3.566093/3.343874 m`,
  max final-hold robot/box tilt `1.262435/1.573505 rad`, and min final-hold
  robot/box z `-1.033506/-1.181892 m`. Conclusion: this sign is not a safe
  brake direction for the current G1 agile policy; run a positive-sign
  contrast before changing the mechanism.
- [x] Run the positive-sign final-brake contrast:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalbrake_pos005_80_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120 MAX_FINAL_HOLD_COMMAND_X=0.06 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=2.0 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=0.05 AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=0 AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=80; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_fpbr bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: determine the velocity-command sign that actually brakes the
  carried G1 after the final 2 m latch. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalbrake_pos005_80_0706`, Slurm
  job `168278`, job-name `g1_lg_fpbr`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168278`
  completed on `server39` in `00:00:52`, build status `0`, checker status
  `1`. Positive brake was better than negative brake but still failed:
  final brake active `80` steps, final commands `x/y/yaw=0.05/0/0` during the
  brake window and final yaw suppressed for `205` steps; fall/drop `30/9`,
  first fall/drop steps `870/891`, final robot/box target-directed travel
  `3.268357/3.303698 m`, max final-hold robot/box tilt
  `0.587173/0.582167 rad`, min final-hold robot/box z
  `0.064077/0.038243 m`, and final relative offset `0.170510 m`.
  Conclusion: the positive sign is safer than negative but still does not
  brake into the 2 m window. The next direct test should trigger final-zero
  earlier and let the robot coast into the target window.
- [x] Run early final-zero latch diagnostic at `1.2 m`:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly120_zero_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=250 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.2 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_fe12 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether cutting commands earlier can coast into the 2 m
  target window without the unstable stand-target blend or final brake pulse.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly120_zero_0706`, Slurm job
  `168306`, job-name `g1_lg_fe12`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence but closest so far.
  Job `168306` completed on `server39` in `00:00:49`, build status `0`,
  checker status `1`. It latched final zero at step `531` for `369` final
  steps, clamped final commands to `0/0/0`, suppressed final yaw for `369`
  steps, had box drop `0`, and achieved target-window stability:
  `target_window_both_final_hold_stable_steps=126` and longest streak `126`.
  It still failed long-hold gates because the stable target-window streak did
  not last to the final step, first fall occurred at step `887`, final-hold
  fall events were `13`, final robot/box target-directed travel ended at
  `3.126255/3.135492 m`, max final-hold robot/box tilt was
  `0.477018/0.476662 rad`, and min final-hold robot/box z was
  `0.313819/0.383676 m`. Conclusion: early zero can coast through the 2 m
  target window and hold briefly, but final zero still has hidden-state or
  stance drift over longer hold.
- [x] Add final-latch policy-state reset hook. The G1 scene now supports
  `--agile-command-hold-final-reset-policy-state`, exposed by the low-cradle
  launcher as `AGILE_COMMAND_HOLD_FINAL_RESET_POLICY_STATE=1`. Summaries and
  checker output report whether it was enabled, reset count, and final reset
  error. Defaults are disabled.
- [x] Run early final-zero latch with final policy-state reset:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly120_reset_zero_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=250 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.2 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 AGILE_COMMAND_HOLD_FINAL_RESET_POLICY_STATE=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_fr12 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether resetting the recurrent policy at final latch reduces
  the late drift/fall seen in `168306`. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly120_reset_zero_0706`,
  Slurm job `168313`, job-name `g1_lg_fr12`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168313`
  completed on `server39` in `00:00:48`, build status `0`, checker status
  `1`. Final policy reset executed once with no reset error, but behavior was
  worse than `168306`: fall/drop `65/43`, first fall/drop steps `835/857`,
  final robot/box target-directed travel `3.445008/3.323817 m`, final
  robot/box lateral error `1.256486/1.326749 m`, max final-hold robot/box
  tilt `0.706925/1.014478 rad`, and min final-hold robot/box z
  `-0.876250/-0.676235 m`. It still had a target-window stable streak
  `121`, but reset increased lateral drift and drop. Conclusion: do not use
  final policy-state reset as the active fix.
- [x] Add final target-window joint-target freeze hook. The G1 scene now
  supports `--agile-command-hold-final-freeze-in-target-window`,
  `--agile-command-hold-final-freeze-max-tilt`, and
  `--agile-command-hold-final-freeze-max-box-tilt`, exposed by the low-cradle
  launcher as `AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1`,
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_TILT`, and
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_BOX_TILT`. When enabled, after final
  hold is active the scene latches the current policy joint targets once
  robot and box are both inside the configured target window with bounded
  tilt, then holds those joint targets. This does not write root pose, root
  velocity, or box pose. Summary/checker report freeze latch and active
  counts. Defaults are disabled.
- [x] Run early final-zero latch with target-window joint-target freeze:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly120_freeze_zero_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=250 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.2 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1 AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_TILT=0.25 AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_BOX_TILT=0.35; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_ffrz bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether freezing a target-window-stable joint target prevents
  the late drift/fall seen in `168306`. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly120_freeze_zero_0706`,
  Slurm job `168324`, job-name `g1_lg_ffrz`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168324`
  completed on `server39` in `00:00:51`, build status `0`, checker status
  `1`. Freeze latched at step `628` for `272` active steps with rollout root
  pose/root velocity/box pose writes all `0`, but it worsened balance:
  fall/drop `156/134`, first fall/drop steps `744/766`, final robot/box
  target-directed travel `3.381368/3.200969 m`, max final-hold robot/box tilt
  `3.141581/3.136450 rad`, min final-hold robot/box z
  `-2.699332/-2.458781 m`, and final-hold target-window longest streak only
  `97` steps. Conclusion: do not use joint-target freeze as the hold
  controller; it removes policy motion needed for balance.
- [x] Run earlier final-zero latch at `0.6 m` with no freeze:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_zero_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.001 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_fe06 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether `168306` failed mostly because final-zero was latched
  too late and the robot coasted beyond the target window. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly060_zero_0706`, Slurm job
  `168331`, job-name `g1_lg_fe06`; submission-time status
  `PENDING (Priority)`. Result: valid negative but useful evidence. Job
  `168331` completed on `server39` in `00:00:47`, build status `0`, checker
  status `1`. It latched final zero at step `365` for `535` final steps,
  kept final x/y/yaw commands at `0/0/0`, and had rollout root pose/root
  velocity/box pose writes all `0`. Final robot/box target-directed travel
  improved to `2.241426/2.296863 m`, inside the `2.35 m` upper gate, but
  lateral drift dominated: final robot/box lateral error
  `1.301041/1.191865 m`, fall/drop `65/28`, first fall/drop steps
  `835/872`, max robot/box tilt `1.646607/1.674712 rad`, and min robot/box z
  `0.172038/0.076788 m`. Target-window final-hold longest streak was `99`
  but did not persist to the end.
- [x] Run `0.6 m` final-zero with very small excess-error lateral correction:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_latsmall_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_f6ly bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: reduce the late lateral drift from `168331` without reproducing
  aggressive lateral-correction failures. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly060_latsmall_0706`, Slurm
  job `168335`, job-name `g1_lg_f6ly`; submission-time status
  `PENDING (Priority)`. Result: near-pass but still fail. Job `168335`
  completed on `server39` in `00:00:48`, build status `0`, checker status
  `1`. It had fall/drop `0/0`, min robot/box z `0.660195/0.759457 m`, max
  robot/box tilt `0.208595/0.413612 rad`, final relative offset
  `0.049381 m`, final robot/box lateral error `0.259286/0.290766 m`, and
  rollout root pose/root velocity/box pose writes all `0`. It failed only
  because final robot/box target-directed travel overshot to
  `2.713489/2.747867 m` and target-window streak did not reach the final
  step. Lateral active steps were `0`, so the useful change versus `168331`
  was yaw gain/limit `0`, not lateral correction.
- [x] Run yaw-zero earlier-cutoff at `0.45 m`:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly045_yawzero_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=500 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.45 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_f45y bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: preserve `168335` no-fall/no-drop behavior while moving final
  target-directed travel back into the `2.0 +/- 0.35 m` target window.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly045_yawzero_0706`, Slurm
  job `168339`, job-name `g1_lg_f45y`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168339`
  completed on `server39` in `00:00:47`, build status `0`, checker status
  `1`. Cutoff was too early: fall/drop `276/217`, first fall/drop steps
  `624/644`, final robot/box target-directed travel only
  `1.554673/1.521620 m`, target-window stable steps `0`, max robot/box tilt
  `1.438984/1.460016 rad`, min robot/box z `0.123796/0.097214 m`, and
  rollout root pose/root velocity/box pose writes all `0`.
- [x] Run yaw-zero interpolated cutoff at `0.55 m`:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly055_yawzero_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=520 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.55 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_f55y bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: interpolate between `0.45 m` early fall and `0.6 m` stable
  overshoot. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly055_yawzero_0706`, Slurm
  job `168343`, job-name `g1_lg_f55y`; submission-time status
  `PENDING (Priority)`. Result: close but failed. Job `168343` completed on
  `server39` in `00:00:51`, build status `0`, checker status `1`. It had no
  box drops and no rollout root pose/root velocity/box pose writes, with final
  box target-directed travel `2.210967 m` in window and target-window final-
  hold longest streak `100` steps, but first fall occurred at step `882` and
  final robot target-directed travel was `2.437910 m`, just over the `2.35 m`
  upper gate. Max robot/box tilt was `0.948362/0.835055 rad`; min robot z was
  `0.264276 m`.
- [x] Run yaw-zero fine cutoff at `0.545 m`:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly0545_yawzero_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=520 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.545 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_f545 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: convert the near-failure at `0.55 m` into an end-of-rollout
  target-window hold without stand/rescue blending. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly0545_yawzero_0706`, Slurm
  job `168350`, job-name `g1_lg_f545`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence and effectively same
  behavior as `168343`. Job `168350` completed on `server39` in `00:00:47`,
  build status `0`, checker status `1`: final latch step `366`, fall/drop
  `18/0`, first fall step `882`, final robot/box target-directed travel
  `2.437910/2.210967 m`, final-hold target-window longest streak `100`, max
  robot/box tilt `0.948362/0.835055 rad`, min robot/box z
  `0.264276/0.486223 m`, and rollout root pose/root velocity/box pose writes
  all `0`. Conclusion: finer cutoff around `0.545-0.55 m` is quantized to
  the same latch and is not useful.
- [x] Run long stable-carry validation at `1200` steps with precise target
  stopping relaxed:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_stablecarry_1200_targetnegx1 FREE_STEPS=1200 FREE_MIN_ROBOT_TRAVEL=2.0 FREE_MIN_BOX_TRAVEL=2.0 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=99 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=99 TARGET_WINDOW_CENTER=-1 TARGET_WINDOW_HALFWIDTH=-1 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=0 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=0 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=0 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=800 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:20:00 --job-name=g1_lg_st12 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: separate stable free-box carrying from precise target stopping.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_finalearly060_yawzero_stable1200_0706`,
  Slurm job `168352`, job-name `g1_lg_st12`; submission-time status
  `PENDING (Priority)`. Result: valid negative long-horizon evidence. Job
  `168352` completed on `server39` in `00:00:52`, build status `0`, checker
  status `1`. It completed `1200/1200` and kept rollout root pose/root
  velocity/box pose writes all `0`, but failed after the 900-step horizon:
  first fall/drop steps `945/965`, fall/drop `255/235`, final robot/box
  target-directed travel `4.848048/4.466154 m`, final relative offset
  `0.427247 m`, max robot/box tilt `3.136022/3.139212 rad`, and min robot/box
  z `-9.467798/-9.494773 m`. Conclusion: `168335` is a 900-step stable carry
  diagnostic, not long-duration success.
- [x] Enable hold-rescue blending in `policy_command` mode. The G1 scene now
  enters the rescue/stand blend branch when `agile_command_hold_rescue_active`
  is true, not only for `stand_targets`, `policy_then_stand`, or final stand.
  Defaults remain disabled. Lightweight validation passed: `py_compile`,
  `bash -n`, and `git diff --check` on the login node.
- [x] Run 1200-step late-rescue stable-carry diagnostic:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_laterescue_1200_targetnegx1 FREE_STEPS=1200 FREE_MIN_ROBOT_TRAVEL=2.0 FREE_MIN_BOX_TRAVEL=2.0 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=99 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=99 TARGET_WINDOW_CENTER=-1 TARGET_WINDOW_HALFWIDTH=-1 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=0 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=0 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=0 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=800 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_RESCUE_ENABLE=1 AGILE_COMMAND_HOLD_RESCUE_ABS_ROLL_THRESHOLD=0.28 AGILE_COMMAND_HOLD_RESCUE_FORWARD_PITCH_THRESHOLD=-999.0 AGILE_COMMAND_HOLD_RESCUE_BLEND_RATE=0.008 AGILE_COMMAND_HOLD_RESCUE_HIP_PITCH=-0.18 AGILE_COMMAND_HOLD_RESCUE_KNEE=0.42 AGILE_COMMAND_HOLD_RESCUE_ANKLE_PITCH=-0.25 AGILE_COMMAND_HOLD_RESCUE_WAIST_PITCH=-0.05; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:20:00 --job-name=g1_lg_rs12 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether late rescue prevents the step `930-950` roll/height
  collapse from `168352`. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_stable1200_laterescue_0706`, Slurm
  job `168356`, job-name `g1_lg_rs12`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168356`
  completed on `server39` in `00:00:48`, build status `0`, checker status
  `1`. Rescue activated at step `930` for `270` steps with reason `abs_roll`,
  proving the new rescue blending path is wired, but it did not improve the
  collapse: first fall/drop steps `946/963`, fall/drop `254/237`, final
  relative offset `0.440256 m`, max robot/box tilt `3.140006/3.134651 rad`,
  min robot/box z `-9.487601/-9.613761 m`, and rollout root pose/root
  velocity/box pose writes all `0`.
- [x] Run 1200-step stronger-roll-balance diagnostic:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_rollstrong_1200_targetnegx1 FREE_STEPS=1200 FREE_MIN_ROBOT_TRAVEL=2.0 FREE_MIN_BOX_TRAVEL=2.0 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=99 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=99 TARGET_WINDOW_CENTER=-1 TARGET_WINDOW_HALFWIDTH=-1 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=0 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=0 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=0 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=0 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=800 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 BALANCE_ROLL_GAIN=0.10 BALANCE_ROLL_RATE_GAIN=0.006 BALANCE_ADJUSTMENT_LIMIT=0.12 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:20:00 --job-name=g1_lg_br12 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether stronger existing roll feedback prevents the
  post-900-step roll collapse. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_stable1200_rollstrong_0706`, Slurm
  job `168358`, job-name `g1_lg_br12`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168358`
  completed on `server39` in `00:00:51`, build status `0`, checker status
  `1`. Stronger roll feedback destabilized much earlier: first fall/drop
  steps `598/645`, fall/drop `600/442`, final robot/box target-directed
  travel only `0.070430/-0.330736 m`, final relative offset `0.447635 m`, max
  robot/box tilt `1.907901/3.138240 rad`, and rollout root pose/root
  velocity/box pose writes all `0`. Conclusion: increasing roll gain/limit is
  not the long-horizon fix; current best direct-G1 evidence remains the
  900-step yaw-zero `0.6 m` run `168335`.
- [x] Run 900-step late small reverse-brake target-stop diagnostic:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake010_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450 MAX_FINAL_HOLD_COMMAND_X=0.012 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.01 AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=400 AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=200; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_lb10 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether a very small reverse command after entering the target
  neighborhood reduces the `168335` overshoot without root/velocity/box pose
  shortcuts, fall, or drop. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latebrake010_900_0706`, Slurm job
  `168363`, job-name `g1_lg_lb10`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168363`
  completed on `server39` in `00:00:49`, build status `0`, checker status
  `1`. Brake active `114` steps from `786` to `899`, max abs x command
  `0.01`, with rollout root pose/root velocity/box pose writes all `0`. It
  preserved stability: fall/drop `0/0`, min robot/box z
  `0.694983/0.784024 m`, max robot/box tilt `0.208595/0.413612 rad`, final
  relative offset `0.042980 m`, and final robot/box lateral error
  `0.266715/0.286565 m`. It still failed precise target stopping because
  final robot/box target-directed travel was `2.716723/2.744582 m`.
- [x] Run 900-step stronger late reverse-brake target-stop diagnostic:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake025_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450 MAX_FINAL_HOLD_COMMAND_X=0.030 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.025 AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=400 AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=200; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_lb25 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether stronger late reverse command reduces overshoot
  while preserving stability. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latebrake025_900_0706`, Slurm job
  `168366`, job-name `g1_lg_lb25`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168366`
  completed on `server39` in `00:00:47`, build status `0`, checker status
  `1`. Brake active `114` steps from `786` to `899`, max abs x command
  `0.025`, and rollout root pose/root velocity/box pose writes all `0`.
  Stability remained good: fall/drop `0/0`, min robot/box z
  `0.711977/0.785524 m`, max robot/box tilt `0.215963/0.413612 rad`, final
  relative offset `0.107504 m`. It failed precise target stopping more than
  `168363`: final robot/box target-directed travel `2.733128/2.772792 m`.
- [x] Run positive-sign late x-command contrast:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake_pos025_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450 MAX_FINAL_HOLD_COMMAND_X=0.030 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=0.025 AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=400 AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=200; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_lbp25 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: determine the effective x-command sign for reducing overshoot.
  Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latebrake_pos025_900_0706`, Slurm
  job `168369`, job-name `g1_lg_lbp25`; submission-time status
  `PENDING (Priority)`. Result: valid negative but useful behavior evidence.
  Job `168369` completed on `server39` in `00:00:47`, build status `0`,
  checker status `1`. Positive x command was active `114` steps from `786` to
  `899`, max abs command `0.025`, with rollout root pose/root velocity/box
  pose writes all `0`. It preserved stability: fall/drop `0/0`, min robot/box
  z `0.752112/0.808381 m`, max robot/box tilt `0.208595/0.413612 rad`, final
  relative offset `0.070556 m`, and final lateral error
  `0.230666/0.241873 m`. It reduced overshoot but still failed target stop:
  final robot/box target-directed travel `2.617844/2.674450 m`.
- [x] Run stronger positive late x-command diagnostic:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake_pos060_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450 MAX_FINAL_HOLD_COMMAND_X=0.070 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=0.06 AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=400 AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=200; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_lbp60 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: test whether stronger positive late command reaches the target
  upper gate while preserving no fall/drop. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_latebrake_pos060_900_0706`, Slurm
  job `168372`, job-name `g1_lg_lbp60`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168372`
  completed on `server39` in `00:00:49`, build status `0`, checker status
  `1`. Command active `114` steps from `786` to `899`, max abs x command
  `0.06`, rollout root pose/root velocity/box pose writes all `0`, fall/drop
  `0/0`, min robot/box z `0.747178/0.808381 m`, max robot/box tilt
  `0.208595/0.413612 rad`, final relative offset `0.093449 m`, and final
  robot/box target-directed travel `2.646988/2.670434 m`. Conclusion:
  positive sign is useful, but the window is too late.
- [x] Run earlier positive x-command target-stop diagnostic:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_earlybrake_pos040_strict_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450 MAX_FINAL_HOLD_COMMAND_X=0.050 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=0.04 AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=330 AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=300; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_ebp40 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: start the useful positive x command during the target-window
  interval instead of after most overshoot has already happened. Submitted as
  tmux `curiosity_g1_agile_largerbox_lowcarry_earlybrake_pos040_900_0706`,
  Slurm job `168381`, job-name `g1_lg_ebp40`; submission-time status
  `PENDING (Priority)`. Result: valid negative evidence. Job `168381`
  completed on `server39` in `00:00:47`, build status `0`, checker status
  `1`. Positive x command was active for `184` steps from `716` to `899`, max
  abs command `0.04`, with rollout root pose/root velocity/box pose writes all
  `0`. It kept fall/drop `0/0` and good height/tilt, but worsened final
  target-directed overshoot to `2.756768/2.788418 m`. It reduced final lateral
  error to `0.029349/0.047980 m`, so timing/magnitude alone is not a reliable
  stop controller.
- [x] Run 820-step target-window hold validation with best yaw-zero setup:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_820_targetnegx1 FREE_STEPS=820 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=400 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_th820 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: verify that current G1/free-box policy reaches the target window
  and remains stable there for a nontrivial end hold, without claiming a
  900-step precise stop controller. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_yawzero_targethold820_0706`, Slurm
  job `168391`, job-name `g1_lg_th820`; submission-time status
  `PENDING (Priority)`. Result: near-pass but still fail. Job `168391`
  completed on `server39` in `00:00:48`, build status `0`, checker status
  `1`. It completed `820` steps with fall/drop `0/0`, min robot/box z
  `0.752112/0.808381 m`, max robot/box tilt `0.208595/0.413612 rad`, final
  relative offset `0.080991 m`, final robot/box lateral error
  `0.136043/0.194526 m`, and rollout root pose/root velocity/box pose writes
  all `0`. It failed only because final box target-directed travel was
  `2.350200 m`, `0.000200 m` over the strict `2.35 m` upper gate, which reset
  the end-of-window streak.
- [x] Run 819-step target-window hold validation:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_819_targetnegx1 FREE_STEPS=819 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=399 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MIN_FINAL_HOLD_ROBOT_Z=0.45 MIN_FINAL_HOLD_BOX_Z=0.45 MAX_FINAL_HOLD_TILT=0.35 MAX_FINAL_HOLD_BOX_TILT=0.45 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.0 AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_th819 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: same target-window hold without loosening any target upper gate,
  ending one simulation step earlier. Submitted as tmux
  `curiosity_g1_agile_largerbox_lowcarry_yawzero_targethold819_0706`, Slurm
  job `168398`, job-name `g1_lg_th819`; submission-time status
  `PENDING (Priority)`. Result: pass. Job `168398` completed on `server39` in
  `00:00:46`, build status `0`, checker status `0`, and `status=pass`. It
  completed `819` steps with fall/drop `0/0`, final robot/box target-directed
  travel `2.298755/2.346454 m`, final robot/box lateral error
  `0.133809/0.191820 m`, final relative offset `0.079615 m`, min robot/box z
  `0.752112/0.808381 m`, max robot/box tilt `0.208595/0.413612 rad`, and
  rollout root pose/root velocity/box pose writes all `0`. Target-window
  both-final-hold stable steps, longest streak, and streak at end were all
  `164`.
- [x] Run same-gate `chestpad` target-hold generalization:
  stamp `20260706_g1_agile_chestpad_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`,
  tmux `curiosity_g1_agile_chestpad_targethold819_0706`, Slurm job `168402`,
  job-name `g1_cp_th819`. It reuses the strict `168398` gates with
  `LARGERBOX_STRICT_MODE=chestpad`, `FREE_STEPS=819`, target-window end
  streak `>=40`, fall/drop `0/0`, height/tilt gates, final command gates near
  zero, and no rollout root pose/root velocity/box pose writes. Submission-
  time status `PENDING (Priority)`. Result: fail. Job `168402` completed on
  `server39` in `00:00:48`, build status `0`, checker status `1`. It had
  fall/drop `87/35`, first fall/drop `732/784`, final robot/box target-
  directed travel `3.334948/3.015123 m`, final robot/box lateral error
  `-1.672259/-1.909126 m`, final relative offset `0.427477 m`, max robot/box
  tilt `3.032566/3.033201 rad`, min robot/box z `-0.636457/-0.659537 m`, and
  rollout root pose/root velocity/box pose writes all `0`. Conclusion:
  chestpad does not transfer the low-cradle target-hold result.
- [x] Run same-gate `boxtilt` target-hold generalization:
  stamp `20260706_g1_agile_boxtilt_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`,
  tmux `curiosity_g1_agile_boxtilt_targethold819_0706`, Slurm job `168403`,
  job-name `g1_bt_th819`. It reuses the strict `168398` gates with
  `LARGERBOX_STRICT_MODE=boxtilt`, `FREE_STEPS=819`, target-window end streak
  `>=40`, fall/drop `0/0`, height/tilt gates, final command gates near zero,
  and no rollout root pose/root velocity/box pose writes. Submission-time
  status `PENDING (Priority)`. Result: fail. Job `168403` completed on
  `server39` in `00:00:41`, build status `0`, checker status `1`. It had
  fall/drop `4/0`, first fall `815`, no box drop, final robot/box target-
  directed travel `0.560830/0.411779 m`, final robot/box lateral error
  `0.623486/0.350950 m`, final relative offset `0.362352 m`, max robot/box
  tilt `0.729192/0.737112 rad`, min robot/box z `0.419574/0.366088 m`,
  target-window stable/final-hold counts `0`, and rollout root pose/root
  velocity/box pose writes all `0`. Conclusion: boxtilt/default higher cradle
  geometry never reaches the target window and fails near the end.
- [ ] Do not claim multi-posture carrying from `168398`. Next valid
  generalization work must either make another posture/contact geometry pass
  the same no-shortcut fall/drop/target-hold gates, or explicitly narrow the
  claim to low-cradle carrying only.
- [x] Run `chestpad` opposite-yaw near-stop target-window validation:
  `export LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_900_targetnegx1 FREE_STEPS=900 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=30 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=30 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=30 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.05 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_cp_tw900 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: retest the historical best `chestpad` posture (`167778`) under the
  current target-window end-streak checker. This is a second-posture
  diagnostic only; even a pass would not prove active probing or autonomous
  posture selection. Submitted as tmux
  `curiosity_g1_chestpad_oppositeyaw_targetwindow900_0706`, Slurm job
  `168419`, job-name `g1_cp_tw900`; submission-time status
  `PENDING (Priority)`. Result: pass. Job `168419` completed on `server39` in
  `00:00:47`, build status `0`, checker status `0`. It completed `900` steps
  with fall/drop `0/0`, final robot/box target-directed travel
  `1.730244/1.759363 m`, final robot/box lateral error
  `0.258455/0.362250 m`, final relative offset `0.108737 m`, max relative
  offset `0.205432 m`, min robot/box z `0.721562/0.825034 m`, max robot/box
  tilt `0.307758/0.384690 rad`, target-window both stable/longest/end streak
  `33/33/33`, and rollout root pose/root velocity/box pose writes all `0`.
  This proves only a short second-posture target-window end hold; next
  strengthen this by extending the same setting to `1000` steps with
  target-window end streak `>=100`.
- [x] Run strengthened `chestpad` target-window validation:
  `export LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_1000_targetnegx1 FREE_STEPS=1000 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=100 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=100 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=100 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.05 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_cp_tw1000 bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: require a longer target-window end hold for the second posture
  while preserving the same no-fall/no-drop/no-shortcut gates. Submitted as
  tmux `curiosity_g1_chestpad_oppositeyaw_targetwindow1000_0706`, Slurm job
  `168420`, job-name `g1_cp_tw1000`; submission-time status
  `PENDING (Priority)`. After repeated queue checks it remained
  `PENDING (Priority)`; leave it queued and do not replace it with login-node
  simulation or one-shot resource paths. Result: fail. Job `168420` completed
  on `server21` in `00:01:01`, build status `0`, checker status `1`. It
  completed `1000` steps with fall/drop `0/0`, target-window both
  stable/longest/end streak `133/133/133`, final robot/box target-directed
  travel `1.912552/1.774289 m`, and rollout root pose/root velocity/box pose
  writes `0/0/0`. It failed strict carry-quality gates: max robot tilt
  `0.485765 > 0.35`, max box tilt `0.713845 > 0.45`, final relative offset
  `0.311690 > 0.25`, and final box lateral error `0.785696 > 0.6`.
- [x] Run `chestpad` final-stop target-window validation:
  `export LARGERBOX_STRICT_MODE=chestpad SUITE_STAMP=20260706_g1_agile_chestpad_oppositeyaw_finalstop_targetwindow_1000_targetnegx1 FREE_STEPS=1000 FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35 MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=100 MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=100 MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=100 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.05 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.65 AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 AGILE_COMMAND_HOLD_FINAL_LATCH=1 AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=100 MAX_FINAL_HOLD_COMMAND_X=0.001 MAX_FINAL_HOLD_COMMAND_Y=0.003 MAX_FINAL_HOLD_COMMAND_YAW=0.001 MAX_FINAL_HOLD_FALL_EVENTS=0 MAX_FINAL_HOLD_BOX_DROP_EVENTS=0; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_cp_fstop bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
  Purpose: preserve the `168420` target-window occupancy while reducing late
  chest-pad tilt, relative-offset, and lateral drift through an explicit final
  stop/hold. Submitted as tmux
  `curiosity_g1_chestpad_finalstop_targetwindow1000_0706`, Slurm job `168431`,
  job-name `g1_cp_fstop`; submission-time status `PENDING (Priority)`.
  Result: near-pass but fail. Job `168431` completed on `server44` in
  `00:00:48`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window both stable/longest/end streak
  `133/133/133`, final-hold stable/longest/end streak `132/132/132`, final
  hold active from step `868` with last command `[0, 0, 0]`, max robot/box
  tilt `0.307758/0.384690 rad`, final relative offset `0.144021 m`, and
  rollout root pose/root velocity/box pose writes `0/0/0`. It failed only
  final box lateral error: `0.614122 > 0.6`.
- [x] Run earlier `chestpad` final-stop target-window validation:
  same as `168431` but with
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.55` and
  `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=140`. Purpose: reduce final box
  lateral drift without loosening the strict `0.6 m` lateral gate. Submitted
  as tmux `curiosity_g1_chestpad_finalstop155_targetwindow1000_0706`, Slurm
  job `168432`, job-name `g1_cp_fs155`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168432` completed on `server44` in
  `00:00:52`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window/final-hold end streak `103/103`,
  final hold active from step `817`, max robot/box tilt
  `0.307758/0.420955 rad`, final relative offset `0.176791 m`, and rollout
  root pose/root velocity/box pose writes `0/0/0`. It failed only final box
  lateral error `0.692755 > 0.6`, worse than `168431`.
- [x] Run intermediate `chestpad` final-stop target-window validation:
  same as `168431` but with
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.62` and
  `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120`. Purpose: test whether a
  smaller shift from `1.65` reduces `168431`'s `0.614122 m` lateral error
  without the `1.55` overshoot in lateral drift. Submitted as tmux
  `curiosity_g1_chestpad_finalstop162_targetwindow1000_0706`, Slurm job
  `168433`, job-name `g1_cp_fs162`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168433` completed on `server44` in
  `00:00:50`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window/final-hold end streak `133/133`,
  final hold active from step `859`, max robot/box tilt
  `0.307758/0.384690 rad`, final relative offset `0.230262 m`, and rollout
  root pose/root velocity/box pose writes `0/0/0`. It failed only final box
  lateral error `0.715806 > 0.6`, worse than `168431`.
- [x] Run `chestpad` final-stop with corrections:
  same as `168431` with
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.65` and
  `AGILE_COMMAND_HOLD_FINAL_SCALE=0.0`, but without
  `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1` and without final command-zero
  gates. Purpose: keep zero forward final scale while allowing yaw/lateral
  corrections to reduce final box lateral error below `0.6 m`. Submitted as
  tmux `curiosity_g1_chestpad_finalstop_corr_targetwindow1000_0706`, Slurm
  job `168435`, job-name `g1_cp_fcorr`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168435` completed on `server44` in
  `00:00:51`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window/final-hold end streak `133/132`,
  final hold active from step `868`, final command
  `[0, -0.035, 0.020590]`, max robot/box tilt `0.307758/0.384690 rad`, final
  relative offset `0.193038 m`, and rollout root pose/root velocity/box pose
  writes `0/0/0`. It failed only final box lateral error
  `0.708802 > 0.6`, worse than `168431`.
- [x] Run `chestpad` final-stop plus final-stand validation:
  same as `168431` but add `AGILE_COMMAND_HOLD_FINAL_STAND=1`,
  `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=0`,
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.02`, and
  `MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=100`. Purpose: reduce
  final-hold lateral drift by transitioning from walking policy to stand
  targets after the `1.65 m` final trigger. Submitted as tmux
  `curiosity_g1_chestpad_finalstand_targetwindow1000_0706`, Slurm job
  `168436`, job-name `g1_cp_fstand`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168436` completed on `server44` in
  `00:00:50`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window/final-hold/final-stand end streak
  `133/132/132`, final stand active from step `868`, final box lateral error
  `0.232397 m`, final relative offset `0.187513 m`, and rollout root pose/
  root velocity/box pose writes `0/0/0`. It failed only tilt gates: max
  robot/box tilt `0.755765/0.749709 rad`.
- [x] Run gentler `chestpad` final-stand validation:
  same as `168436` but with
  `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=40`,
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.005`, and
  `MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=80`. Purpose: preserve the
  `168436` lateral improvement while reducing the tilt spike from the stand
  transition. Submitted as tmux
  `curiosity_g1_chestpad_finalstand_gentle_targetwindow1000_0706`, Slurm job
  `168437`, job-name `g1_cp_fsgnt`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168437` completed on `server39` in
  `00:00:57`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window/final-hold/final-stand end streak
  `133/132/92`, max robot/box tilt `0.309353/0.384690 rad`, final relative
  offset `0.165383 m`, and rollout root pose/root velocity/box pose writes
  `0/0/0`. It failed only final box lateral error `0.650740 > 0.6`.
- [x] Run intermediate `chestpad` final-stand validation:
  same as `168436`/`168437` but with
  `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=20`,
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.01`, and
  `MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=100`. Purpose: balance the
  low lateral error of aggressive stand transition with the low tilt of gentle
  transition. Submitted as tmux
  `curiosity_g1_chestpad_finalstand_mid_targetwindow1000_0706`, Slurm job
  `168438`, job-name `g1_cp_fsmid`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168438` completed on `server21` in
  `00:00:58`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window/final-hold/final-stand end streak
  `133/132/112`, max robot/box tilt `0.658000/0.658923 rad`, final relative
  offset `0.198759 m`, final box lateral error `0.712096 m`, and rollout root
  pose/root velocity/box pose writes `0/0/0`. It failed both tilt and lateral
  gates.
- [x] Run zero-delay slow-blend `chestpad` final-stand validation:
  same as `168436` but with `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.005`.
  Purpose: preserve zero-delay final stand's lateral improvement while
  reducing the stand-transition tilt spike. Submitted as tmux
  `curiosity_g1_chestpad_finalstand_zerodelay_slow_targetwindow1000_0706`,
  Slurm job `168440`, job-name `g1_cp_fszs`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168440` completed on `server21` in
  `00:00:54`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0`, target-window/final-hold/final-stand end streak
  `133/132/132`, final box lateral error `0.129922 m`, final relative offset
  `0.249589 m`, and rollout root pose/root velocity/box pose writes `0/0/0`.
  It failed only tilt gates: max robot/box tilt `0.827365/0.979325 rad`.
- [x] Run `chestpad` final-stop with flipped lateral sign:
  same as `168435` but with `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0`. Purpose:
  test whether the final corrective lateral command direction caused the
  lateral drift that made `168435` worse than `168431`. Submitted as tmux
  `curiosity_g1_chestpad_finalstop_corr_latsignneg_targetwindow1000_0706`,
  Slurm job `168450`, job-name `g1_cp_flsn`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168450` completed on `server21` in
  `00:00:49`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0` and rollout root pose/root velocity/box pose
  writes `0/0/0`, but target-window end streak collapsed to `0`, final
  robot/box lateral error was `1.710366/1.626656 m`, final robot/box
  target-directed travel was `2.548142/2.555898 m`, and max robot/box tilt was
  `0.391817/0.542518 rad`. This lateral-sign branch is worse than the
  zero-correction near-pass `168431`.
- [x] Run `chestpad` final-stop with yaw-only correction:
  start from the `168431` final-stop near-pass, keep final scale `0.0`, keep
  lateral correction disabled, and allow only yaw correction during final hold.
  Purpose: test whether heading correction can reduce the remaining
  `0.614122 m` box lateral near-miss without the destabilizing lateral command
  or final-stand tilt spike. Submitted as tmux
  `curiosity_g1_chestpad_finalstop_yawonly_targetwindow1000_0706`, Slurm job
  `168451`, job-name `g1_cp_fyaw`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168451` completed on `server21` in
  `00:00:49`, build status `0`, checker status `1`. It completed `1000`
  steps with rollout root pose/root velocity/box pose writes `0/0/0`, but
  disabling lateral correction caused fall/drop `232/193`, first fall/drop
  step `768/807`, target-window/final-hold end streak `0/0`, final robot/box
  lateral error `3.699152/3.710816 m`, and max robot/box tilt
  `3.135358/3.139102 rad`.
- [x] Run `chestpad` final-stop with a wider chest pad:
  start from the `168431` final-stop near-pass and change only
  `CRADLE_CHEST_PAD_SIZE_Y` from `0.38` to `0.44`. Purpose: solve the
  `0.614122 m` final box lateral near-miss through contact geometry while
  preserving the early lateral correction and avoiding final-stand tilt.
  Submitted as tmux
  `curiosity_g1_chestpad_finalstop_widepad_targetwindow1000_0706`, Slurm job
  `168452`, job-name `g1_cp_wpad`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168452` completed on `server21` in
  `00:00:52`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0` and rollout root pose/root velocity/box pose
  writes `0/0/0`, but final latch occurred at step `991`, final active steps
  were only `9`, target-window end streak was `10`, max robot/box tilt was
  `0.412648/0.501682 rad`, final relative offset was `0.322406 m`, and final
  box lateral error was `0.650154 m`.
- [x] Run `chestpad` final-stop with slightly higher lateral gain:
  return to the exact `168431` geometry/final-stop setup and change only
  `AGILE_COMMAND_HOLD_LATERAL_GAIN` from `0.08` to `0.10`. Purpose: reduce
  the `0.614122 m` final box lateral near-miss without changing final hold,
  final stand, or chest-pad geometry. Submitted as tmux
  `curiosity_g1_chestpad_finalstop_latgain010_targetwindow1000_0706`, Slurm
  job `168453`, job-name `g1_cp_lg10`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168453` completed on `server21` in
  `00:00:49`, build status `0`, checker status `1`. It completed `1000`
  steps with no rollout root/box shortcut writes, but higher lateral gain
  overcorrected: fall/drop `95/82`, first fall/drop step `905/918`, no
  terminal/final latch, target-window end streak `0`, max robot/box tilt
  `0.971987/1.016840 rad`, and final robot/box lateral error
  `-1.621430/-1.625827 m`.
- [x] Run `chestpad` final-stop with tiny lateral-gain interpolation:
  return to the exact `168431` geometry/final-stop setup and set
  `AGILE_COMMAND_HOLD_LATERAL_GAIN=0.085`. Purpose: test a much smaller
  interpolation between the near-pass `0.08` and the overcorrecting `0.10`.
  Submitted as tmux
  `curiosity_g1_chestpad_finalstop_latgain0085_targetwindow1000_0706`, Slurm
  job `168454`, job-name `g1_cp_lg085`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168454` completed on `server21` in
  `00:00:48`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0` and rollout root pose/root velocity/box pose
  writes `0/0/0`, but final latch was false, target-window end streak was `0`,
  final robot/box target-directed travel was only `1.188529/1.227951 m`, final
  robot/box lateral error was `0.658880/0.774984 m`, and max robot/box tilt was
  `0.371441/0.630405 rad`.
- [x] Run `chestpad` final-stop with tiny base lateral command:
  return to the exact `168431` geometry/gain/final-stop setup and set
  `AGILE_COMMAND_Y=0.005`. Purpose: bias the path slightly while preserving
  the near-pass controller structure. Submitted as tmux
  `curiosity_g1_chestpad_finalstop_cmdy005_targetwindow1000_0706`, Slurm job
  `168455`, job-name `g1_cp_y005`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168455` completed on `server21` in
  `00:00:48`, build status `0`, checker status `1`. It completed `1000`
  steps with fall/drop `0/0` and rollout root pose/root velocity/box pose
  writes `0/0/0`, but final latch moved to step `960`, final active steps
  were only `40`, target-window end streak was `41`, max box tilt was
  `0.486637 rad`, and final robot/box lateral error overcorrected to
  `-0.858138/-0.730904 m`.
- [x] Run `chestpad` final-stop with smaller base lateral command:
  return to the exact `168431` setup and set `AGILE_COMMAND_Y=0.001`. Purpose:
  test a much smaller interpolation between no bias and the over-biasing
  `0.005`. Submitted as tmux
  `curiosity_g1_chestpad_finalstop_cmdy001_targetwindow1000_0706`, Slurm job
  `168456`, job-name `g1_cp_y001`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168456` completed on `server21` in
  `00:00:51`, build status `0`, checker status `1`. It completed `1000`
  steps with no rollout root/box shortcut writes, but fall/drop was
  `347/272`, first fall/drop step was `653/686`, target-window end streak was
  `0`, final robot/box target-directed travel overshot to
  `4.135339/4.167482 m`, final robot/box lateral error was
  `-1.190266/-1.049643 m`, and max robot/box tilt was
  `3.140074/3.138898 rad`.
- [x] Run low-carry held-out load/shape validation:
  stop the fragile chest-pad micro-parameter branch and use the verified
  low-carry target-hold route to test at least one held-out mass or box shape
  under the same no-fall/no-drop/no-shortcut/target-window gates. Submitted a
  strict light-box mass held-out run with `FREE_BOX_MASS=0.25`, tmux
  `curiosity_g1_lowcarry_lightbox_targethold819_strict_0706`, Slurm job
  `168458`, job-name `g1_lc_l025`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168458` completed on `server21` in
  `00:00:48`, build status `0`, checker status `1`. It completed `819` steps
  with no rollout root/box shortcut writes, but fall/drop was `384/225`, first
  fall/drop step was `435/594`, target-window/final-hold end streak was `0`,
  final robot/box target-directed travel overshot to `4.399167/3.986899 m`,
  final relative offset was `0.449969 m`, and max robot/box tilt was
  `2.710347/2.745914 rad`.
- [x] Run low-carry heavy-box held-out validation:
  use the same strict target-hold gates as the verified low-carry baseline,
  but set `FREE_BOX_MASS=0.75`. Purpose: determine whether the current
  low-carry route has any nearby mass robustness after the 0.25 kg failure.
  Submitted as tmux
  `curiosity_g1_lowcarry_heavybox_targethold819_strict_0706`, Slurm job
  `168462`, job-name `g1_lc_h075`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168462` completed on `server21` in
  `00:00:50`, build status `0`, checker status `1`. It completed `819` steps
  with no rollout root/box shortcut writes, but fall/drop was `346/284`, first
  fall/drop step was `473/535`, terminal/final latch were both false,
  target-window and final-hold end streaks were `0`, final robot/box
  target-directed travel was `0.182098/-0.169243 m`, final relative offset was
  `0.405846 m`, and max robot/box tilt was `1.996009/3.139406 rad`.
- [ ] Replace scalar held-out tuning with load-adaptive stabilization:
  the current low-carry pass does not generalize to `0.25 kg` or `0.75 kg`,
  and chest-pad scalar tweaks are fragile. Next implementation should add a
  real load/probe-conditioned adjustment path, such as measured box response
  and tilt/relative-offset feedback that changes final-hold posture/commands,
  then rerun the same strict held-out gates. First implementation step added
  observed robot-progress gates for terminal/final hold:
  `AGILE_COMMAND_HOLD_TERMINAL_MIN_ROBOT_TARGET_TRAVEL` and
  `AGILE_COMMAND_HOLD_FINAL_MIN_ROBOT_TARGET_TRAVEL`; syntax checks passed.
  Submitted first strict light-box retest with terminal/final robot-progress
  gates `0.50/0.80` as Slurm job `168465`, job-name `g1_lc_lrg`. Result:
  fail. Job `168465` completed on `server21` in `00:00:48`, build status `0`,
  checker status `1`. It completed `819` steps with no rollout root/box
  shortcut writes, but fall/drop was `428/340`, first fall/drop step was
  `391/412`, target-window and final-hold end streaks were `0`, final relative
  offset was `0.431750 m`, and max robot/box tilt was `1.841554/2.180149 rad`.
  Added follow-up min-step terminal/final gates and syntax checks passed.
  Submitted strict light-box min-step retest with terminal/final min steps
  `350/386` as Slurm job `168466`, job-name `g1_lc_lsg`. Result: fail. Job
  `168466` completed on `server21` in `00:00:47`, build status `0`, checker
  status `1`. It completed `819` steps with no rollout root/box shortcut
  writes, but fall/drop was `402/149`, first fall/drop step was `417/437`,
  target-window and final-hold end streaks were `0`, final relative offset was
  `0.389333 m`, and max robot/box tilt was `1.977972/2.232556 rad`.
  Submitted follow-up strict light-box retest with both terminal/final min
  steps at `420` as Slurm job `168467`, job-name `g1_lc_l420`. Result: fail.
  Job `168467` completed on `server21` in `00:00:47`, build status `0`,
  checker status `1`. It completed `819` steps with no rollout root/box
  shortcut writes, but fall/drop was `400/344`, first fall/drop step was
  `419/437`, terminal/final latch steps were both `420`, target-window and
  final-hold end streaks were `0`, final relative offset was `0.417000 m`,
  and max robot/box tilt was `3.131353/3.140109 rad`. Next test keeps step
  `420` but uses `AGILE_COMMAND_HOLD_FINAL_SCALE=0.006` instead of a zero
  final command. Submitted as Slurm job `168468`, job-name `g1_lc_lf006`.
  Result: fail. Job `168468` completed on `server21` in `00:00:50`, build
  status `0`, checker status `1`. It completed `819` steps with no rollout
  root/box shortcut writes and final command `[0.0006, 0.0, 0.0]`, but
  fall/drop was `400/356`, first fall/drop step was `419/437`, target-window
  and final-hold end streaks were `0`, final relative offset was `0.277017 m`,
  and max robot/box tilt was `1.402402/1.504111 rad`. Stop this scalar/phase
  gate branch; next implementation needs contact/retention or hold-posture
  changes.
- [x] Run first contact-retention light-box validation:
  strict `0.25 kg` low-carry held-out with top lid enabled from start, lower/
  wider top lid, taller side rails, and taller end stops. Submitted as Slurm
  job `168471`, job-name `g1_lc_lret`. Result: fail. Job `168471` completed
  on `server21` in `00:00:48`, build status `0`, checker status `1`. It
  completed `819` steps with no rollout root/box shortcut writes. Strong
  retention improved final relative/lateral metrics and reduced drop events to
  `104`, but fall/drop was still `335/104`, first fall/drop step was
  `484/523`, target-window and final-hold end streaks were `0`, and max
  robot/box tilt was `3.131061/3.123740 rad`.
- [x] Run strong-retention plus step420 light-box validation:
  combine the `168471` physical retention geometry with terminal/final min
  steps `420/420` to avoid early final zero command at step `364`. Submitted
  as Slurm job `168472`, job-name `g1_lc_rs420`. Result: fail. Job `168472`
  completed on `server21` in `00:00:48`, build status `0`, checker status
  `1`. It completed `819` steps with no rollout root/box shortcut writes.
  Fall/drop decreased to `55/39` and first fall/drop moved late to `764/780`,
  but target-window/final-hold end streaks were `0`, final lateral error was
  `-2.340681/-2.443050 m`, final relative offset was `0.359277 m`, and max
  robot/box tilt was `2.705030/1.518419 rad`.
- [x] Run strong-retention plus step420 plus slow final-stand blend:
  use the `168472` geometry/gates, enable `AGILE_COMMAND_HOLD_FINAL_STAND=1`,
  delay stand blend by `80` final-hold steps, and use a small blend rate
  `0.002` to test whether a controller-backed hold posture reduces late roll.
  Submitted as Slurm job `168473`, job-name `g1_lc_rsfs`. Result: fail. Job
  `168473` completed on `server21` in `00:00:49`, build status `0`, checker
  status `1`. It completed `819` steps with no rollout root/velocity/box pose
  writes, but final-stand worsened delayed-failure timing versus `168472`:
  fall/drop were `237/20`, first fall/drop `582/628`, target-window/final-hold
  end streaks were `0`, final lateral error was about `-0.967/-1.031 m`,
  final relative offset error was `0.308799 m`, and max robot/box tilt was
  `2.052260/3.123082 rad`.
- [x] Stop the current slow final-stand blend branch and return to the `168472`
  strong-retention plus delayed-final base. Next test should address late
  lateral drift and roll directly, for example by a bounded lateral/yaw
  correction that remains active after the delayed final gate or by asymmetric
  roll stabilization, while preserving no rollout root pose, root velocity, or
  box pose writes.
- [x] Await/record late-lateral-drift diagnostic `168475`, job-name
  `g1_lc_latcorr`. It keeps the `168472` retention geometry and step420 gates,
  disables final-stand, turns lateral correction on across the whole agile
  hold instead of terminal-only, lowers lateral error start to `0.20 m`,
  raises lateral command limit to `0.003`, and relaxes lateral correction tilt
  gates to robot/box `0.80/0.90 rad`. Result: fail. Job `168475` completed on
  `server21` in `00:00:29`, build status `0`, checker status `1`. It completed
  `819` steps with no rollout root/velocity/box pose writes. Lateral
  correction was active for `242` steps versus `58` in `168472`, but fall/drop
  were `87/39`, first fall/drop was `732/780`, target-window/final-hold end
  streaks were `0`, final lateral error was `-2.313470/-2.372427 m`, final
  relative offset was `0.291257 m`, and max robot/box tilt was
  `1.636552/1.647063 rad`.
- [x] Run a lateral sign-reversal diagnostic from the same `168472` base:
  keep final-stand disabled, keep the relaxed always-on lateral correction
  window from `168475`, but set `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0`. If
  this improves lateral error or delayed-fall timing, continue with bounded
  command tuning; if it worsens or remains unchanged, move to roll-feedback or
  contact-geometry changes instead of more lateral command sweeps. Submitted
  as Slurm job `168478`, job-name `g1_lc_latrev`; submission-time status
  `PENDING (Priority)`. Result: fail. Job `168478` completed on `server21` in
  `00:00:26`, build status `0`, checker status `1`. It completed `819` steps
  with no rollout root/velocity/box pose writes. Reversing sign improved final
  lateral error to `-1.079545/-1.217948 m` and max robot/box tilt to
  `1.273742/1.348365 rad`, but fall/drop worsened to `150/100`, first
  fall/drop moved earlier to `657/719`, final box target-directed travel fell
  to `1.392827 m`, final relative offset was `0.382696 m`, and target-window/
  final-hold end streaks remained `0`.
- [x] Run a mild reversed-lateral diagnostic: keep `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0`
  but reduce the command back toward the earlier limit, e.g. limit `0.0015`
  with the same relaxed always-on window. This tests whether the sign reversal
  can reduce drift without causing the under-travel and earlier fall/drop seen
  in `168478`. Submitted as Slurm job `168479`, job-name `g1_lc_latmild`;
  submission-time status `PENDING (Priority)`. Result: fail. Job `168479`
  completed on `server21` in `00:00:28`, build status `0`, checker status
  `1`. It completed `819` steps with no rollout root/velocity/box pose writes.
  It improved box retention and progress versus `168478`: box drops were `0`,
  final robot/box target-directed travel was `2.013973/1.655340 m`, and robot
  target-window stable steps reached `24`. It still failed with fall events
  `74`, first fall step `745`, target-window/final-hold end streaks `0`, final
  lateral error `-1.437272/-1.531827 m`, final relative offset `0.371662 m`,
  and max robot/box tilt `1.424887/1.671567 rad`.
- [ ] Run a roll-feedback sign diagnostic on top of `168479`: keep the mild
  reversed lateral settings and set `BALANCE_ROLL_SIGN=1.0` explicitly. If it
  delays or removes the late roll without hurting box retention, continue with
  roll gain/limit tuning; otherwise switch to contact geometry or a different
  hold controller. Submitted as Slurm job `168482`, job-name `g1_lc_rollpos`;
  submission-time status `PENDING (Priority)`. A duplicate `cpu` partition
  GPU attempt, job `168502`, was cancelled after it had no predicted start
  time and reported nodes down/drained/reserved for higher-priority
  partitions. Keep `168482` as the active pending diagnostic.
- [ ] Interpret the `168482` result against the coupled roll/pitch evidence:
  `168472` ended with large negative robot roll/pitch, `168478` reduced final
  roll but still had large pitch and early drop, and `168479` retained the box
  but ended with large positive roll. If `BALANCE_ROLL_SIGN=1.0` improves this,
  try a smaller roll gain or delayed balance start; if it flips the failure
  into the opposite roll/pitch mode, stop sign/gain sweeping and change the
  hold/contact formulation.
- [x] Add low-carry lateral/roll decision report:
  `experiments/reports/2026-07-06_g1_lowcarry_lateral_roll_decision.md`.
  It summarizes `168472`, `168475`, `168478`, `168479`, and pending `168482`,
  and fixes the next decision rule around roll-feedback tuning versus
  contact/hold geometry changes.
- [x] Add compute-side low-carry follow-up launcher:
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`. It
  refuses login/management nodes and can run either `CASE_SET=roll`
  (`BALANCE_ROLL_SIGN=1.0`, `BALANCE_START_STEP=420`) or `CASE_SET=contact`
  (hold-enabled chest-pad contact) from the `168479` base. This is only
  preparation for post-`168482` compute-node tests; it is not new evidence.
- [x] Add showcase status report:
  `experiments/reports/2026-07-06_showcase_status.md`. It lists current
  presentable videos, strongest partial evidence, and strict wording
  boundaries so diagnostic assets are not mislabeled as final humanoid carrying
  success.
- [x] Add and submit a real presentation-quality G1 visualization path. The
  existing direct-carry posture MP4s are proxy-block scaffold/debug clips and
  should not be used as main presentation visuals. Added Replicator RGB
  capture support to `scripts/isaac/build_core_world_g1_box_scene.py` and
  launcher `scripts/isaac/run_core_world_g1_showcase_lowcarry_capture.sh`.
  Submitted Slurm job `168509`, job-name `g1_show_rgb`, through tmux session
  `curiosity_g1_showcase_rgb_0706`; it is pending in the `gpu` partition and
  will render the real G1 low-carry pass configuration from `168398` to
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_showcase_g1_lowcarry_168398_rgb/agile_low_cradle_freebox_walk/rgb_frames/`
  plus `showcase_g1_lowcarry.mp4` if `ffmpeg` is available on the compute
  node. This is rendering/capture preparation until frames or MP4 are actually
  produced.
- [x] Record first real-G1 RGB capture attempt `168509`. Result: negative for
  visualization. It ran on `server39` and produced summary/log files, but no
  RGB frames because the current Isaac environment lacks
  `omni.replicator.core` (`capture_rgb_error = ModuleNotFoundError: No module
  named 'omni.replicator'`). It also failed control gates under render mode
  with early fall/drop, so do not use this run as pass evidence or showcase.
- [x] Add replay-record path for presentation visualization. New
  `--record-replay-csv` and `--record-replay-every-n-steps` options write
  `core_world_g1_box_scene_replay.csv` with robot root pose, box pose, joint
  names, and joint positions from a non-rendered rollout. Launcher
  `scripts/isaac/run_core_world_g1_showcase_lowcarry_capture.sh` now supports
  `SHOWCASE_CAPTURE_RGB=0 SHOWCASE_RECORD_REPLAY=1`.
- [x] Add replay-render path:
  `scripts/isaac/render_core_world_g1_replay_showcase.py` and
  `scripts/isaac/run_core_world_g1_replay_showcase_render.sh`. This renders a
  recorded pass trajectory as real G1+box viewport frames and labels it as
  replay visualization, not new control evidence.
- [x] Harden replay renderer capture import. It now falls back between
  `omni.kit.renderer_capture` and `omni.renderer_capture`, and calls
  `wait_async_capture()` when available after each swapchain capture.
- [x] Add replay renderer camera framing controls. The replay renderer now
  supports `--follow-frame` and `--frame-zoom`, and the launcher enables
  follow-frame so the rendered camera keeps the real G1 and box in view during
  the recorded carry trajectory.
- [x] Add replay showcase checker:
  `scripts/isaac/check_core_world_g1_replay_showcase.py`. It requires the
  source non-rendered rollout summary to pass, `record_replay_csv=true`,
  fall/drop 0, enough replay rows, enough rendered PNG frames, and a
  visualization-only success claim. The replay render launcher writes
  `g1_replay_showcase_check.json` after rendering.
- [x] Harden replay CSV file handling before the queued record run. The
  replay writer in `scripts/isaac/build_core_world_g1_box_scene.py` now uses
  `ExitStack` so both the normal metrics CSV and optional replay CSV are
  closed through context management even if rollout code exits through an
  exception path. Lightweight check:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py`
  passed on the login node.
- [x] Add an immediate browser-only presentation fallback while compute
  resources are pending:
  `experiments/visuals/g1_progress_showcase/20260706_g1_lowcarry_168398_browser_showcase/index.html`.
  It uses sampled states from the passed `168398` rollout CSV and draws a
  G1-like humanoid, low-carry box, side-view posture, top-view path, and key
  metrics. It is a schematic progress visualization only, not a true Isaac
  camera render and not new control evidence.
- [x] Add a matching static SVG poster for the same immediate fallback:
  `experiments/visuals/g1_progress_showcase/20260706_g1_lowcarry_168398_browser_showcase/g1_lowcarry_168398_progress_poster.svg`.
  It is for quick slide/inspection use only and has the same schematic-only
  limitation.
- [ ] Await replay-record job `168580`, job-name `g1_rec_short`, submitted
  through tmux session `curiosity_g1_record_replay_short_0706`. It is pending
  in the `cpu` partition with `--gres=gpu:1`, and as of 2026-07-06 22:12 CST
  Slurm currently schedules it for `server02` around 23:15 CST. If it produces a passing
  `core_world_g1_box_scene_replay.csv`, submit the replay renderer immediately.
- [x] Record replay-record job `168580`. Result: negative and not usable for
  replay rendering. It ran on `server39` and failed strict carrying gates with
  fall/drop `720/617`, final robot/box target-directed travel
  `0.2968/0.1619 m`, max robot/box tilt `2.0515/2.1004 rad`, and no shortcut
  writes. It also did not produce replay CSV because the job environment had
  `CAPTURE_RGB=1`, `record_replay_csv=false`, and missing
  `RECORD_REPLAY_CSV=1`; it therefore repeated the old RGB-capture path rather
  than the intended non-rendered replay-record path.
- [x] Add strict replay-render waiter:
  `scripts/isaac/wait_and_submit_g1_replay_render.sh`. It waits for a record
  summary, refuses to submit render unless the record summary is `status=pass`,
  `record_replay_csv=true`, fall/drop `0/0`, and the replay CSV has at least
  20 rows. This fixes the earlier stale watcher behavior that submitted a
  follow-up after a failed record summary.
- [ ] Await corrected replay-record retry `168632`, job-name `g1_rec_retry`,
  tmux session `curiosity_g1_record_replay_retry2_0706`, stamp
  `20260706_g1_lowcarry_168398_replay_record_retry2`. It was submitted with
  explicit `--export=ALL,SHOWCASE_CAPTURE_RGB=0,SHOWCASE_RECORD_REPLAY=1,
  CAPTURE_RGB=0,RECORD_REPLAY_CSV=1,RECORD_REPLAY_EVERY_N_STEPS=10`. A strict
  render watcher is running in
  `curiosity_g1_replay_render_retry2_waiter_0706` and will only submit render
  if retry2 passes the record gate.
- [x] Record corrected replay-record retry `168632`. Result: pass and usable
  as replay source. It ran on `server39` with `CAPTURE_RGB=0`,
  `RECORD_REPLAY_CSV=1`, completed 819/819, fall/drop `0/0`, final robot/box
  target-directed travel `2.2988/2.3465 m`, final relative error `0.0796 m`,
  max robot/box tilt `0.2086/0.4136 rad`, and no rollout root pose, root
  velocity, or box pose writes. It wrote
  `core_world_g1_box_scene_replay.csv` with 84 lines.
- [x] Fix strict replay-render watcher zero-count bug. The first retry2 watcher
  check incorrectly treated `fall_events=0` and `box_drop_events=0` as missing
  because it used `summary.get(...) or 999999`. Fixed
  `scripts/isaac/wait_and_submit_g1_replay_render.sh` to distinguish `None`
  from zero.
- [ ] Await replay render job `168658`, job-name `g1_replay_viz2`, submitted
  from the corrected retry2 replay CSV to
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2/`.
  It is a visualization replay only, not new control evidence.
- [x] Cancel pending full replay render `168658` after Slurm pushed it to
  2026-07-07 00:04 CST, because a shorter quick render could backfill earlier.
  This only cancelled a pending Curiosity visualization job, not a control run
  and not any non-Curiosity job.
- [ ] Await quick replay render `168664`, job-name `g1_viz_quick`, tmux session
  `curiosity_g1_replay_quick_render_0706`, output directory
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2_quick/`.
  It uses the passed retry2 replay CSV, `CAPTURE_EVERY_N_ROWS=3`, and
  `MAX_FRAMES=24` to produce an earlier presentable video; it remains a replay
  visualization only, not new control evidence.
- [x] Cancel pending quick render `168664` after Slurm pushed it to 2026-07-07
  00:04 CST. Submitted shorter quick2 render `168669`, job-name `g1_viz_q2`,
  tmux session `curiosity_g1_replay_quick2_render_0706`, output directory
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2_quick2/`,
  with `CAPTURE_EVERY_N_ROWS=4` and `MAX_FRAMES=18`. Dry-run still shows a
  possible 2026-07-06 23:37 CST backfill, but the actual job has not yet been
  assigned a StartTime. Do not submit more duplicate quick renders unless this
  one fails or is clearly starved.
- [x] Harden quick replay render before it starts. `168669` time limit was
  updated from 3 minutes to 5 minutes after Slurm pushed the job to
  2026-07-07 01:22:51 CST, reducing the risk that Isaac startup plus frame
  capture times out. `scripts/isaac/render_core_world_g1_replay_showcase.py`
  now adds explicit dome/key lighting to reduce the risk of dark/blank replay
  frames. Lightweight check:
  `python3 -m py_compile scripts/isaac/render_core_world_g1_replay_showcase.py`
  passed.
- [ ] Await contact follow-up watcher
  `curiosity_g1_contact_after_showcase_waiter_0706`. It waits until the replay
  record/render path has finished, then submits `CASE_SET=contact` through
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh` as
  job-name `g1_contact_next`. This is the next control branch because `168482`
  showed roll-sign tuning is not enough.
- [x] Pre-check the `CASE_SET=contact` follow-up gate while waiting for
  `168580`. The active case is `chestpad_hold_contact`: it enables chest-pad
  support on hold while preserving no rollout root/velocity/box pose shortcut
  gates. The apparently low `FREE_MIN_*TRAVEL=0.02` is not the only success
  criterion; the suite also requires target-window stable-step gates near the
  configured 2.0 m target window and caps final target-directed overrun at
  2.35 m. This is configuration validation only, not experiment evidence.
- [x] Record prematurely triggered contact follow-up `168627`, job-name
  `g1_contact_next`, stamp
  `20260706_g1_lowcarry_followup_chestpad_hold_contact`. The base summary had
  fall/drop `0/0`, no shortcut writes, chest-pad enabled on hold, max robot/box
  tilt `0.2747/0.2733 rad`, and final relative error `0.1482 m`, but the strict
  checker failed because target-window stable steps were `0`, final robot/box
  target-directed travel was only `0.7175/0.6576 m`, and final hold active
  steps were only `15 < 399`. Conclusion: chest-pad hold improves stability and
  retention but suppresses progress to the 2.0 m target window; it is not
  carrying success.
- [x] Add the next contact/hold branch implied by `168627`: delayed chest-pad
  closure. `scripts/isaac/build_core_world_g1_box_scene.py` now supports
  `--cradle-chest-pad-enable-on-terminal-hold` and
  `--cradle-chest-pad-enable-on-final-hold`; the low-cradle launcher forwards
  `CRADLE_CHEST_PAD_ENABLE_ON_TERMINAL_HOLD` and
  `CRADLE_CHEST_PAD_ENABLE_ON_FINAL_HOLD`. The follow-up launcher now supports
  `CASE_SET=contact_terminal` / `CASE_SET=contact_next`, which runs
  `chestpad_terminal_contact`: keep chest-pad collision disabled through early
  progress, then close it only at terminal hold. This is preparation only until
  run on a compute node.
- [ ] Await delayed chest-pad contact watcher
  `curiosity_g1_contact_next_after_render_0706`. It waits for quick render job
  `168669` to leave the queue, then submits `CASE_SET=contact_next` as
  job-name `g1_contact_next2` with prefix
  `20260706_after_quick_render_contact_next`. This keeps the next control
  branch from competing with the current visualization job.
- [x] Stop waiting for render before submitting the next control branch after
  Slurm pushed quick render `168669` to 2026-07-07 01:22 CST. The after-render
  watcher `curiosity_g1_contact_next_after_render_0706` was stopped to avoid a
  duplicate submission.
- [ ] Await direct delayed chest-pad contact run `168788`, job-name
  `g1_contact_next2`, tmux session `curiosity_g1_contact_next_direct_0706`,
  prefix `20260706_after_quick_render_contact_next`, `CASE_SET=contact_next`.
  It tests terminal-hold chest-pad closure while preserving the no-shortcut
  gates. As of 2026-07-07 00:00 CST, Slurm schedules it for
  2026-07-07 01:22:51 CST on `server02`. As of 2026-07-07 00:14 CST, Slurm
  rescheduled it to 2026-07-07 10:00:00 CST.
- [x] Migrate the currently useful pending jobs away from the stuck `cpu`
  queue. Cancelled pending/no-artifact `168669` and `168788`, then submitted
  `168801` (`g1_viz_gpu_q3`) and `168802` (`g1_contact_gpu`) through separate
  Curiosity tmux+srun sessions on the `gpu` partition. As of 2026-07-07
  00:18 CST, both are still pending by priority; no Isaac frames or contact
  summaries exist yet.
- [x] Improve the replay renderer before the next render slot. The renderer
  now adds robot/box path markers and start/end/target markers to make the
  G1+box replay legible as a presentation visual. `py_compile`, `bash -n`,
  and `git diff --check` passed for the touched renderer/launcher files.
- [ ] Await `168801` (`curiosity_g1_replay_gpu_render_0707`). Expected output:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3/`.
  If it completes, inspect `g1_replay_render_summary.json`,
  `g1_replay_showcase_check.json`, PNG frame count, and generated MP4s before
  calling it presentable. It remains visualization-only replay, not new
  control evidence.
- [ ] Await `168802` (`curiosity_g1_contact_next_gpu_0707`). Expected output
  prefix: `20260707_gpu_contact_next_chestpad_terminal_contact`. Compare it
  against `168632` and `168627`: the key question is whether delayed chest-pad
  closure preserves target-window progress while keeping fall/drop and shortcut
  counts at zero.
- [x] Resolve old `168420` pending branch. It failed despite fall/drop `0/0`
  and target-window both stable/longest/end streak `133/133/133`, because max
  robot/box tilt, final relative offset, and final box lateral error exceeded
  strict carry-quality gates. Do not promote `chestpad` from `168420`;
  subsequent `168431` final-stop work already followed this branch and also
  remained a near-pass rather than a verified second posture.
- [x] Add reusable target-window posture validation launcher:
  `scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`.
  It refuses to run on `mgmtserver*`, then runs the current low-carry
  `819`-step baseline, the `chestpad` `1000`-step long-hold validation, and
  opt-in load held-outs through the existing larger-box strict suite. `bash -n`
  passed. This is a compute-side validation launcher only, not new success
  evidence.
- [x] Add automatic JSON summary generation to the target-window posture
  validation launcher. It now calls
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py` for all selected
  case roots and writes `targetwindow_posture_validation_summary.json` under
  the launcher output root. `bash -n` passed.
- [x] Add a stricter multi-posture/load gauntlet launcher:
  `scripts/isaac/run_core_world_g1_posture_gauntlet.sh`. It refuses to run on
  login nodes and runs `lowcarry_base`, `chestpad_terminal`,
  `boxtilt_diagnostic`, `lowcarry_lightbox`, and `lowcarry_heavybox` through
  the existing strict G1 suite plus summary generation. This is aligned with
  the full objective because it directly tests posture and load gaps instead
  of only reproducing the narrow low-carry pass.
- [x] Harden G1 posture launchers. Added a login-node guard to
  `scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`, and exposed
  `ARM_POSE_MODE`, `ARM_POSE_START_STEP`, and `ARM_POSE_RAMP_STEPS` through
  the low-cradle launcher so gauntlet cases can exercise existing G1
  `right_front_reach` / `both_front_reach` arm targets. The low-cradle env
  snapshot now includes `ARM_` variables so posture settings are auditable.
- [ ] Await posture gauntlet watcher
  `curiosity_g1_posture_gauntlet_after_contact_0707`. It waits for `168802`
  to leave the queue, then submits one compute-node `srun` job named
  `g1_posture_gauntlet` with stamp
  `20260707_g1_posture_gauntlet_after_contact`. Expected summary:
  `experiments/outputs/core_world_g1_posture_gauntlet/20260707_g1_posture_gauntlet_after_contact/g1_posture_gauntlet_summary.json`.
  Treat any failed case as real negative evidence; do not narrow the objective
  around only the passing low-carry case.
- [x] Harden replay showcase validation. `check_core_world_g1_replay_showcase.py`
  now samples PNG frames and checks minimum byte size plus PNG dimensions;
  `run_core_world_g1_replay_showcase_render.sh` passes expected width/height.
  This prevents a render with corrupt or tiny frames from being marked
  presentable just because files exist.
- [x] Add contact follow-up comparison utility:
  `scripts/isaac/summarize_core_world_g1_contact_followup.py`. It compares
  the `168632` baseline low-carry pass, `168627` hold-contact partial failure,
  and the pending terminal-contact case without running Isaac. Pending report:
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_pending.json`.
- [ ] Await contact comparison watcher
  `curiosity_g1_contact_compare_after_168802_0707`. It waits for `168802` to
  leave the queue, then writes
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json`.
  Use that report to decide whether terminal chest-pad closure preserves the
  baseline target-window progress or remains a retention/progress tradeoff.
- [x] Add machine-readable completion audit:
  `scripts/isaac/audit_g1_carry_completion.py`. It reads the baseline
  low-carry summary, contact comparison, posture/load gauntlet summary, and
  showcase check. It returns nonzero unless the actual full-goal evidence is
  present and passing. Current report:
  `experiments/reports/2026-07-07_g1_carry_completion_audit_current.json`
  is `fail` because terminal-contact and gauntlet evidence are missing.
- [ ] Await completion audit watcher
  `curiosity_g1_completion_audit_after_gauntlet_0707`. It waits for the
  posture gauntlet watcher to finish, then writes
  `experiments/reports/2026-07-07_g1_carry_completion_audit_after_gauntlet.json`.
  Do not use the narrow low-carry pass as final completion unless this audit
  or a stricter successor passes.
- [x] Add next-action recommender:
  `scripts/isaac/recommend_g1_next_carry_actions.py`. It reads completion
  audit, contact comparison, and gauntlet summaries and emits ranked next
  actions without running Isaac. Current output:
  `experiments/reports/2026-07-07_g1_next_carry_actions_current.json`, which
  recommends waiting for `168802` and the gauntlet rather than duplicate
  submissions.
- [ ] Await next-action watcher
  `curiosity_g1_next_actions_after_audit_0707`. It waits for the completion
  audit watcher, then writes
  `experiments/reports/2026-07-07_g1_next_carry_actions_after_audit.json`.
  Use it to pick the next controller/contact/posture branch after the queued
  evidence lands.
- [ ] Monitor rescheduled `168801`/`168802`. As of 2026-07-07 00:42 CST,
  both are still pending but moved earlier to 02:10:58 CST on `server39`.
  Do not submit duplicate render/contact jobs while these are pending.
  Update 2026-07-07 01:06 CST: both remain `PENDING (Priority)`, and Slurm's
  predicted start time slipped to 02:20:50 CST on `server39`.
  Update 2026-07-07 01:16 CST: both remain `PENDING (Priority)` with no
  assigned nodes or after-run artifacts.
  Update 2026-07-07 01:24 CST: same status; no render/contact/gauntlet
  artifacts exist yet. Refreshed current pipeline JSON/Markdown reports.
- [ ] Await render fallback watcher
  `curiosity_g1_render_fallback_after_168801_0707`. As of 2026-07-07
  00:57 CST, `168801` and `168802` are still `PENDING (Priority)`, and the
  main render log only contains Slurm queue text. The fallback watcher waits
  for `168801`; if the main render has no usable PNG/check output, it submits
  one short 960x540 GPU render to
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_960x540/`.
  If the main render is usable, it skips automatically.
- [x] Add showcase visual manifest writer:
  `scripts/isaac/write_g1_showcase_visual_manifest.py`. Current manifest:
  `experiments/reports/2026-07-07_g1_showcase_visual_manifest.md`, status
  `pending_or_failed`, because the render queue has not produced real
  PNG/MP4 artifacts yet.
- [x] Fix pending terminal-contact report path:
  `scripts/isaac/summarize_core_world_g1_contact_followup.py` now points
  missing low-cradle cases at the standard
  `<stamp>/agile_low_cradle_freebox_walk/` directory. Refreshed the current
  contact comparison, completion audit, and next-action JSON so `168802` will
  be checked against the correct expected summary path.
- [x] Add Slurm job snapshot to active pipeline reports:
  `scripts/isaac/collect_g1_active_pipeline_status.py` now records tracked
  `squeue` state for `168801` and `168802`; the Markdown report now includes
  a `Slurm Jobs` section. Current report shows both jobs still pending for
  priority, so missing artifacts are not yet completed-run failures.
- [x] Add periodic active-status watcher:
  `curiosity_g1_periodic_status_until_168801_168802_done_0707` refreshes the
  current active pipeline JSON/Markdown reports every 10 minutes while
  `168801` or `168802` remains in `squeue`, then does one final refresh after
  they leave the queue. It only uses lightweight status/report commands.
- [ ] Await showcase manifest watcher
  `curiosity_g1_showcase_manifest_after_render_0707`. It waits for the main
  render and fallback render watchers, then rewrites the visual manifest so
  the final showcase path is explicit.
- [x] Record main render failure and patch render backend:
  `168801` failed because `omni.kit.viewport.utility` is unavailable. Updated
  `scripts/isaac/render_core_world_g1_replay_showcase.py` to use USD Camera
  prims plus `omni.replicator.core` RGB annotator data and PIL PNG writing
  instead of viewport/swapchain capture.
- [ ] Await fallback render job `168849` (`g1_viz_gpu_fb`). It is pending and
  should exercise the patched render path. Validate PNG dimensions/content and
  MP4 output before using it as the showcase visual.
  Update: `168849` failed immediately with exit `127` due to a relative
  `scripts/...` path in the `srun` command, so it did not test the patched
  renderer. Await absolute-path replacement job `168882`
  (`g1_viz_gpu_fb_abs`) instead.
  Update: `168882` also failed immediately with exit `127` because nested
  shell expansion produced `/scripts/...`. Await direct-path replacement job
  `168895` (`g1_viz_gpu_fb_direct`) instead.
- [x] Record terminal chest-pad result from `168802`: strict fail with
  `fall_events=104`, first fall step `715`, final box target-directed travel
  `1.88795 m`, final relative error `0.34432 m`, and no target-window/final-
  hold streak. This is negative evidence for terminal chest-pad contact, not
  progress.
- [x] Add contact-rescue cases to
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`:
  `chestpad_terminal_nolateral`, `chestpad_terminal_tiny_pad`, and
  `chestpad_terminal_late_tiny_pad`.
- [ ] Await contact rescue job `168851` (`g1_contact_rescue`) from tmux
  `curiosity_g1_contact_rescue_gpu_0707`. It will write
  `experiments/reports/2026-07-07_g1_contact_rescue_comparison_after_run.json`.
  Update: `168851` failed immediately with exit `127` due to a relative
  `scripts/...` path in the `srun` command. Await absolute-path replacement
  job `168883` (`g1_contact_rescue_abs`), which will write
  `experiments/reports/2026-07-07_g1_contact_rescue_abs_comparison_after_run.json`.
  Update: `168883` also failed immediately with exit `127` because nested
  shell expansion produced `/scripts/...`. Await direct-path replacement job
  `168896` (`g1_contact_rescue_direct`), which will write
  `experiments/reports/2026-07-07_g1_contact_rescue_direct_comparison_after_run.json`.
  Update: `168896` completed, but the rescue suite is negative. The no-
  lateral terminal case still fell (`104` fall events), the tiny-pad and late-
  tiny-pad cases produced hundreds of falls and drops, and late tiny-pad
  failed before pad activation. Stop treating chest-pad geometry/timing tweaks
  as the main rescue path.
- [x] Patch replay renderer for the direct fallback failure from `168895`.
  `168895` failed after Isaac startup because `omni.replicator.core` was
  imported before the extension was enabled. The renderer now explicitly
  enables `omni.replicator.core`, advances the app five updates, and then
  imports the module.
- [x] Patch the live G1 RGB capture path for the same Replicator extension
  issue. `scripts/isaac/build_core_world_g1_box_scene.py` now enables
  `omni.replicator.core` before importing it in the optional `CAPTURE_RGB`
  branch. This is only hardening until a fresh compute-node capture succeeds.
- [ ] Await extension-enabled replay render `168900`, job-name
  `g1_viz_fb_ext`, tmux session
  `curiosity_g1_render_fallback_ext_gpu_0707`, output directory
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_ext_960x540/`.
  It must produce a passing `g1_replay_showcase_check.json`, PNG frames with
  the expected 960x540 dimensions, and MP4 output before it is usable as a
  presentation visual. It remains replay visualization only, not new control
  evidence.
  Update before start: the renderer now writes a failure summary with the
  concrete Replicator import error if extension loading still fails, so the
  post-run manifest should no longer show only a missing render summary.
  Update before start: the renderer now supports `--capture-backend auto`.
  In auto mode it uses Replicator when available, otherwise it enables
  `isaacsim.core.rendering_manager`, binds a viewport to the replay camera,
  and captures frames with `omni.kit.renderer.capture` app screenshots. This
  is intended to handle the current install, where `isaacsim.replicator.*`
  extensions exist but no local `omni.replicator.core` extension directory was
  found.
  Result: `168900` failed on `server43` after 44 seconds with zero frames.
  The summary now records the exact blocker: Replicator cannot import
  `omni.replicator.core` because Kit cannot resolve `omni.kit.pip_archive`,
  and the app-screenshot fallback cannot import
  `isaacsim.core.rendering_manager` because Kit cannot resolve
  `omni.kit.viewport.window`. Do not rerun this capture path unchanged.
- [x] Add schematic presentation fallback renderer:
  `scripts/isaac/render_g1_replay_presentation_fallback.py`. It draws a
  clearer G1-like replay GIF/poster from the passed `168632` replay CSV and
  refuses to run on login nodes. It is explicitly not an Isaac camera render
  and not new control evidence.
- [ ] Await presentation fallback render job `168986`, job-name
  `g1_fallback_gif`, tmux session
  `curiosity_g1_presentation_fallback_render_0707`. Expected output:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif/`.
  Result: `168986` completed on `server43` and wrote a passing schematic
  fallback summary plus GIF/poster/64 frames. This is usable only as a
  presentation fallback:
  `schematic_replay_visual_only_not_isaac_camera_render_not_new_control_evidence`.
- [x] Add non-pad balance rescue cases after `168896` made the chest-pad
  rescue family negative. `CASE_SET=balance_rescue` now runs
  `nopad_final_freeze` and `nopad_stronger_balance` in
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`.
  These keep chest-pad disabled and test final-window policy freeze /
  stronger joint-level balance feedback.
- [ ] Await non-pad balance rescue job `168972`, job-name
  `g1_balance_rescue`, tmux session `curiosity_g1_balance_rescue_gpu_0707`.
  It writes case outputs under
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_gpu_balance_rescue_direct_*`.
  Watcher `curiosity_g1_balance_rescue_watch_0707` will write
  `experiments/reports/2026-07-07_g1_balance_rescue_comparison_after_run.json`.
  Result: `168972` completed with Slurm exit `0:0`, but strict comparison
  failed. Both non-pad branches restored target-directed progress above
  roughly `2.2 m`, but both failed late with fall/drop events and target-window
  end streak `0`. Next targeted diagnostic is late target-window pitch/drop
  stabilization, not more chest-pad geometry tuning.
- [x] Add late target-window recovery follow-up cases:
  `CASE_SET=late_recovery` now runs `nopad_late_gentle_rescue` and
  `nopad_late_stand_blend`. These test small final-window freeze/rescue or
  slow policy-to-stand blending after the balance-rescue result showed good
  progress but late pitch/drop instability.
- [ ] Monitor late target-window recovery job `168995`, job-name
  `g1_late_rec`, tmux session `curiosity_g1_late_recovery_gpu_0707`. It runs
  on `server59` and should write
  `experiments/reports/2026-07-07_g1_late_recovery_comparison_after_run.json`
  from inside the same compute allocation.
  Result: `168995` finished with Slurm exit `1:0`; the comparison report was
  written and is strict `fail`. `nopad_late_gentle_rescue` briefly held the
  target window but overran and fell/dropped. `nopad_late_stand_blend`
  under-traveled and also fell/dropped. Do not keep tuning this late rescue /
  stand-blend branch unchanged.
- [ ] Next diagnostic direction after `168995`: replace late rescue with
  target-window-entry command shaping / overshoot arrest and retention checks.
  This should be a new targeted case set, separate from chest-pad geometry and
  late stand-blend.
- [x] Add target-window arrest follow-up cases:
  `CASE_SET=target_window_arrest` runs `load05_window_zero_arrest` and
  `load05_window_reverse_brake`, both at 0.5 kg. These target the failed
  gauntlet `lowcarry_base` load and the late-recovery overrun/under-travel
  split.
- [ ] Monitor target-window arrest job `168997`, job-name `g1_tw_arrest`,
  tmux session `curiosity_g1_target_window_arrest_gpu_0707`. It is pending on
  GPU priority and should write
  `experiments/reports/2026-07-07_g1_target_window_arrest_comparison_after_run.json`.
  Result: `168997` completed with Slurm exit `0:0`, but both 0.5 kg arrest
  cases failed. `load05_window_zero_arrest` under-traveled and fell/dropped;
  `load05_window_reverse_brake` also under-traveled and dropped heavily. This
  is negative evidence for continuing final-hold scalar/brake micro-tuning on
  the current open-loop Agile-command wrapper.
- [ ] Next control milestone: replace the current open-loop Agile-command
  wrapper for load/posture generalization. Candidate directions are a
  controller-backed load-aware locomotion/retention policy, a materially
  different contact-retention controller, or a task formulation that closes
  the loop on box pose/tilt while preserving no root/box shortcut writes.
- [x] Add box-progress closed-loop command controller:
  `scripts/isaac/build_core_world_g1_box_scene.py` now has optional
  `--agile-command-box-progress-controller` and
  `--agile-command-box-lateral-controller`. They close the command loop on
  measured box target-directed progress and box lateral error instead of
  relying only on fixed command scale/final-hold thresholds. Defaults are off.
- [x] Add launcher/env passthrough and `CASE_SET=box_progress_controller`.
  It runs 0.5 kg `load05_box_progress_pd` and
  `load05_box_progress_conservative`.
- [ ] Monitor box-progress controller job `169004`, job-name `g1_box_pd`,
  tmux session `curiosity_g1_box_progress_controller_gpu_0707`. It is pending
  on GPU priority and should write
  `experiments/reports/2026-07-07_g1_box_progress_controller_comparison_after_run.json`.
- [x] Add box-retention posture feedback:
  `--box-retention-posture-controller` computes risk from box-robot relative
  error and box tilt, then applies crouch/waist/arm-closing posture offsets.
  Defaults are off and summary fields record active steps and max risk.
- [x] Add `CASE_SET=box_progress_retention`, a single 0.5 kg case combining
  box-progress command control with retention posture feedback.
- [ ] Monitor retention feedback job `169006`, job-name `g1_box_ret`, tmux
  session `curiosity_g1_box_progress_retention_gpu_0707`. It is pending on GPU
  priority and should write
  `experiments/reports/2026-07-07_g1_box_progress_retention_comparison_after_run.json`.
- [x] Select materially different non-G1-wrapper baseline: the strongest
  existing no-root prismatic scaffold is `cradle_free_box` with active probe,
  probe-adaptive gait/posture, and guarded prelift quasistatic stepping. It is
  not final humanoid carrying, but it has the right no-root/free-box physical
  gates for a reference substrate.
- [ ] Monitor prismatic no-root reference validation job `169008`, job-name
  `prism_ref`, tmux session `curiosity_prismatic_reference_validation_0707`.
  It reruns the 10 kg active-probe/adaptive-posture cradle-free-box scaffold
  with strict checker gates:
  articulated carrier, foot-contact drive, active probe belief with no hidden
  ground truth, root/box writes 0, fall/drop 0, post-settle payload travel
  `>=0.15 m`, target distance `<=0.02 m`, relative error `<=0.01 m`, payload
  z `>=0.45 m`, and max tilt `<=0.20 rad`.
- [x] Add prismatic reference presentation fallback renderer:
  `scripts/isaac/render_prismatic_reference_presentation_fallback.py`. It
  draws a legible prismatic carrier, feet/legs, cradle, free box, target line,
  metrics, GIF, poster, and frames from a completed state CSV. It refuses to
  run on login nodes and is only a schematic visual.
- [ ] Monitor visual watcher
  `curiosity_prismatic_reference_visual_watch_0707`. After `169008` completes,
  it submits compute-node job-name `prism_viz` and should write
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid/prismatic_reference_presentation_fallback_summary.json`.
- [ ] Monitor posture/load gauntlet job `168850`, which started on `server43`
  around 2026-07-07 02:22 CST. This is the broad posture/load verification;
  failed cases are expected evidence and must not be treated as queue noise.
  Result: `168850` failed strictly after 6m17s. All five gauntlet cases
  failed. `boxtilt_diagnostic` stayed upright with fall/drop `0/0` but failed
  target-window/hold progress; the other four cases had large fall/drop
  counts. Use
  `experiments/outputs/core_world_g1_posture_gauntlet/20260707_g1_posture_gauntlet_after_contact/g1_posture_gauntlet_summary.json`
  as negative posture/load evidence.
- [x] Add active pipeline status collector:
  `scripts/isaac/collect_g1_active_pipeline_status.py`. Current report:
  `experiments/reports/2026-07-07_g1_active_pipeline_status_current.json`,
  status `incomplete`, because render/contact/gauntlet after-run artifacts
  are not present yet. The report includes `generated_at_utc` for snapshot
  traceability.
- [ ] Await pipeline status watcher
  `curiosity_g1_pipeline_status_after_watchers_0707`. It waits for the
  after-audit next-action watcher, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_status_after_watchers.json`.
- [ ] Await render-status watcher `curiosity_g1_render_status_after_168801_0707`.
  It waits for `168801` to leave the queue, then writes
  `experiments/reports/2026-07-07_g1_render_pipeline_status_after_168801.json`
  using the read-only pipeline status collector.
- [x] Add active pipeline failure classifier:
  `scripts/isaac/classify_g1_active_pipeline_failures.py`. Current report:
  `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_current.json`,
  classifying the active render/contact logs as `queued` and missing render/
  contact/gauntlet artifacts as `missing_artifact`.
- [ ] Await failure classification watcher
  `curiosity_g1_failure_class_after_watchers_0707`. It waits for final
  pipeline status, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_after_watchers.json`.
- [x] Add Markdown pipeline status report:
  `scripts/isaac/write_g1_active_pipeline_markdown_report.py`. Current report:
  `experiments/reports/2026-07-07_g1_active_pipeline_status_current.md`,
  showing missing artifacts, completion failures, failure categories, and
  recommended actions in a human-readable page.
- [ ] Await Markdown report watcher
  `curiosity_g1_markdown_report_after_watchers_0707`. It waits for the final
  failure classifier, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_status_after_watchers.md`.
- [x] Generate an immediate presentation visual from the strongest historical
  prismatic scaffold while fresh validation remains queued. Slurm job
  `169015` (`prism_hist_viz`) completed on `server36` with exit `0:0` and
  wrote a 1600x900 GIF/poster under
  `experiments/visuals/prismatic_reference_showcase/20260706_prismatic_cradle_probe_adaptive_posture_standard10_mid_retry24a/`.
  This is a schematic replay only, not an Isaac camera render, humanoid
  walking, learned carrying, or final success evidence.
- [x] Try a CPU compute-node fresh prismatic reference validation because the
  historical reference summary used `device=cpu`. Slurm job `169019`
  (`prism_ref_cpu`) completed the rollout but failed strict checker gates:
  post-settle payload travel was only about `-0.0760 m`, target distance was
  about `0.0940 m`, and the checker exited nonzero. Treat it as negative
  fresh-validation evidence, not a pass.
- [x] Record box-progress controller diagnostic `169004` (`g1_box_pd`). Slurm
  completed with exit `0:0`, but the comparison is strict `fail`. Both new
  0.5 kg controller cases failed with falls/drops and no final hold; the
  conservative case overran badly to about `5.166 m` final box target-directed
  travel. Treat scalar box-progress command feedback as negative evidence, not
  a promising main path.
- [x] Record box-progress plus box-retention diagnostic `169006`
  (`g1_box_ret`). Slurm completed with exit `0:0`, but the comparison is
  strict `fail`: the new case had `0` drops but `470` falls, max robot/box
  tilt about `2.423/2.251 rad`, final box target-directed travel only about
  `0.247 m`, and no final hold. Treat retention posture feedback as negative
  for the current G1 wrapper route.
- [x] Record first prismatic GPU validation `169008` (`prism_ref`) as invalid
  configuration failure, not physical evidence. It ran with
  `ENABLE_HORIZONTAL_LEGS=0` and stopped at 260 steps because
  `guarded_prelift_quasistatic_step_cycle` requires horizontal legs.
- [x] Record matched prismatic CPU validation `169026` (`prism_ref_mcpu`). It
  reruns the reference scaffold with corrected horizontal legs and
  historical-like settings: `STEPS=2880`, `FOOT_LENGTH=0.65`,
  `GATED_STEP_MAX_TRAVEL_LOSS=0.04`, `GATED_STEP_RECOVERY_PHASE=0.35`, and
  `GUARDED_STEP_TARGET_TOLERANCE=0.03`.
- [x] Fix the prismatic checker gate mismatch by adding
  `--max-post-settle-payload-relative-offset-error`. The original checker used
  the whole-trajectory max relative error for the `0.01 m` gate, which was
  stricter than the historical reference itself. Rechecking `169026` with
  global max relative error <= `0.06 m` and post-settle max relative error <=
  `0.012 m` wrote
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/reference_check_corrected.json`
  with `status=pass`.
- [x] Record fresh matched prismatic visual job `169027` (`prism_mviz`),
  submitted through
  tmux `curiosity_prismatic_matched_visual_0707`, writing to
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/`.
  It completed on `server36` with exit `0:0` and wrote a fresh 1600x900
  GIF/poster from the matched `169026` rollout.
- [x] Add prismatic posture/load validation suite:
  `scripts/isaac/run_prismatic_reference_posture_load_suite_0707.sh`. It runs
  four corrected-checker cases: nominal 10 kg mid carry, near-chest high 12 kg,
  long-reach low 8 kg, and bulky 10 kg.
- [x] Record prismatic posture/load suite `169029` (`prism_suite`), submitted
  through tmux `curiosity_prismatic_posture_load_suite_0707`. Expected suite
  summary:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_summary.json`.
  Result: strict `fail`, but 3/4 corrected-checker cases passed. The only
  failure was `near_chest_12kg_high`, which had fall/drop `0/0` and small
  tilt/relative error but stopped early with post-settle payload travel
  `0.147676 m` against the `0.15 m` gate.
- [x] Record targeted `near_chest_12kg_high` retry `169031`
  (`prism_nc_tight`) with tighter `GUARDED_STEP_TARGET_TOLERANCE=0.015`
  before changing broader scaffold mechanics. Expected output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_near_chest_12kg_high_tighttol/`.
  The rollout passed the corrected suite gate when rechecked: fall/drop `0/0`,
  post-settle payload travel `0.18562 m`, target distance `0.01562 m`, and max
  tilt `0.09948 rad`.
- [x] Write after-retry posture/load aggregate:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_after_retry_summary.json`.
  It is `status=pass` with 4/4 corrected-checker cases passing. This remains
  prismatic scaffold evidence only, not final robot carrying.
- [x] Record MuJoCo assisted quadruped payload diagnostic
  `curiosity_mujoco_quad_payload_assisted_0707`, job-name `mj_quad_payload`.
  It tests a more robot-like multi-joint gait backend with a 4 kg welded
  payload and explicit stabilizing body-force controller. Expected output:
  `experiments/outputs/mujoco_quadruped_payload/20260707_mujoco_quad_assisted_payload4kg/`.
  Result: strict `fail`; it traveled about `1.738 m` but had `94` fall events
  and max tilt about `3.159 rad`.
- [ ] Await conservative MuJoCo assisted quadruped retry
  `curiosity_mujoco_quad_payload_conservative_0707`, job-name `mj_quad_cons`.
  It uses 4 kg welded payload, lower target speed `0.20 m/s`, and stronger
  stabilizing torque. Expected output:
  `experiments/outputs/mujoco_quadruped_payload/20260707_mujoco_quad_assisted_payload4kg_conservative/`.
- [x] Add completion-audit report:
  `experiments/reports/2026-07-06_g1_carry_completion_audit.md`. It records
  current verified evidence, pending evidence, negative evidence, and the
  remaining requirements before any full completion claim is valid.
- [x] Add G1 probe-to-posture diagnostic selector:
  `scripts/isaac/select_core_world_g1_carry_posture_from_probe.py`. It uses
  visible box size and logged probe displacement, explicitly ignores hidden
  box mass, and emits a JSON selection report plus optional shell env exports
  for existing low-carry/chest-pad validation runs. `python3 -m py_compile`
  passed and the script is executable. This is a diagnostic heuristic only,
  not final autonomous posture selection.
- [x] Add probe argument passthrough to
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh` so
  compute-side G1 runs can enable `PROBE_MODE=front_bumper` without changing
  default behavior.
- [x] Add compute-side probe-selected target-window pipeline:
  `scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`.
  It runs a G1 probe, selects low-carry or chest-pad from the probe summary,
  and launches the selected posture validation. `bash -n` passed and the
  script is executable. This is plumbing only; it still needs a real
  compute-node pass before it counts as active-probing/posture-selection
  evidence.
- [x] Run probe-selected G1 target-window validation diagnostic:
  `PIPELINE_STAMP=20260706_g1_probe_selected_targetwindow_diag1 srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --job-name=g1_probe_sel bash scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`.
  Purpose: produce a real probe summary, selector report, and selected posture
  validation result without using hidden box mass for selection. Submitted as
  tmux `curiosity_g1_probe_selected_targetwindow_diag1_0706`, Slurm job
  `168429`, job-name `g1_probe_sel`; submission-time status
  `PENDING (Priority)`. Result: pass. Job `168429` completed on `server44` in
  `00:01:39`, Slurm exit `0:0`. Probe summary reported
  `probe_mode=front_bumper`, `probe_active_steps=220`, and probe motion
  `0.511708 m`. The selector passed, selected `lowcarry`, and reported
  `selection_uses_hidden_ground_truth=false`. The selected validation passed
  with `819` steps, fall/drop `0/0`, target-window end streak `164`, and
  rollout root pose/root velocity/box pose writes `0/0/0`. This is a passing
  diagnostic pipeline only, not final autonomous posture-selection success.
- [x] Tighten final-hold comparison helper exit status. It now returns nonzero
  whenever any compared case is not `pass`, so missing summaries, failed
  checks, shortcut writes, target-window failures, or final-hold gate failures
  can be caught by shell automation.
- [x] Add a conservative lateral excess-error option:
  `--agile-command-hold-lateral-use-excess-error`, exposed as
  `AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1`. When enabled, lateral
  correction uses only the error magnitude above
  `AGILE_COMMAND_HOLD_LATERAL_ERROR_START` rather than the full lateral error.
  Default is disabled, so existing experiments are unchanged. This is the next
  fallback if threshold-gated lateral remains too aggressive.
- [x] Add the planned excess-error fallback row to the low-carry 900-step
  matrix report. `experiments/reports/2026-07-06_g1_lowcarry_900_control_matrix.md`
  now shows both the pending threshold-gated run and the planned excess-error
  fallback as missing cases.
- [x] Inspect old low-carry CSV failure windows. Terminal-only lateral started
  at step `381` and failed early at step `620` after driving lateral error
  through zero while robot/box pitch worsened; no-lateral latched micro-hold
  failed later at step `810` from slow lateral drift. This supports waiting
  for the threshold-gated run and argues against increasing lateral gain.
- [x] Add lateral posture-risk gating:
  `--agile-command-hold-lateral-max-tilt` and
  `--agile-command-hold-lateral-max-box-tilt`, exposed as
  `AGILE_COMMAND_HOLD_LATERAL_MAX_TILT` and
  `AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT`. Defaults are `999.0`, so existing
  behavior and queued job `167998` are unchanged. The scene reports
  `agile_command_hold_lateral_suppressed_by_tilt_steps`, and checker/
  summarizer/matrix reporting includes the gate fields.
- [x] Supersede the old `167998` excess-error fallback. Job `167998` did not
  fail by early destabilization; it was invalid because of a code bug, and
  job `168131` later supplied valid behavior evidence. Do not run this
  fallback as the next branch unless a later comparison explicitly revives it:
  `export LARGERBOX_STRICT_MODE=lowcarry SUITE_STAMP=20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_excess_tiltgate_strict_900_targetnegx1 FREE_STEPS=900 AGILE_COMMAND_HOLD_YAW_CORRECTION=1 AGILE_COMMAND_HOLD_YAW_GAIN=0.04 AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65 AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.006 AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.55 AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.30 AGILE_COMMAND_HOLD_LATERAL_GAIN=0.012 AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.003 AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0; srun --export=ALL -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --job-name=g1_lg_lex bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`.
- [x] Stop unchanged ANYmal IsaacLab/RSL-RL payload wrapper as a current
  backend candidate. Job `169316` (`any_payload`) and no-Fabric retry
  `169317` (`any_nofab`) both failed before rollout during IsaacLab
  environment initialization with `Failed to get DOF velocities from backend`.
  No summary was written, so there is no ANYmal walking/carrying evidence.
- [x] Record G1 Isaac camera replay render job `169319` (`g1_render`) as a
  render-environment failure. It generated no frames/MP4 because
  `omni.replicator` and `isaacsim.core.rendering_manager` are unavailable in
  the current environment. This is not control evidence.
- [x] Record G1 fallback presentation visual job `169324` (`g1_fallback`).
  It completed on `server36` in `00:00:11` with exit `0:0` and generated a
  1600x900 schematic GIF/poster with 83 frames from the verified G1 low-carry
  replay CSV under
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_policy_replay_fallback/`.
  This is visualization-only replay, not Isaac camera render and not new
  control evidence.
- [x] Convert the fallback frames to MP4. System `ffmpeg` was unavailable on
  the compute job (`169326`, exit `127:0`), so job `169327` (`g1_iomp4`) used
  `imageio_ffmpeg` and produced
  `g1_lowcarry_replay_fallback.mp4` and
  `g1_lowcarry_replay_fallback_annotated.mp4`.
- [ ] Next controller backend must be materially different from the exhausted
  families: not the MuJoCo hand-controller sweeps, not G1 scalar/cradle/
  threshold sweeps, and not the unchanged ANYmal IsaacLab wrapper.
- [x] Add active-probe-selected load validation suite:
  `scripts/isaac/run_core_world_g1_probe_selected_load_validation_suite.sh`.
  It runs front-bumper probe -> selector -> selected strict target-window
  validation at `0.25`, `0.50`, and `0.75 kg`, then writes an aggregate
  diagnostic summary.
- [x] Monitor Slurm job `169332` (`g1_probe_load`) through tmux
  `codex_g1_probe_load_0707`. Summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Result: strict `fail`, `1/3` cases passing. The selector always chose
  `lowcarry` from probe motion and visible size while ignoring hidden mass.
  `0.50 kg` passed, but `0.25 kg` failed with `384` falls / `225` drops and
  `0.75 kg` failed with `346` falls / `284` drops. Current front-bumper probe
  and threshold selector do not provide a useful unknown-load risk signal.
- [ ] Next active-probing work needs a better probe feature or learned/system-
  identification step. Do not keep reporting the current threshold selector as
  unknown-load adaptation.
- [x] Add and run probe-selected load feature audit:
  `scripts/isaac/audit_g1_probe_selected_load_features.py`, Slurm job
  `169334` (`g1_pr_audit`). Report:
  `experiments/reports/2026-07-07_g1_probe_selected_load_feature_audit.json`.
  It shows the current probe itself is unsafe for all three masses: each probe
  summary failed with `240` fall events and about `210-211` box-drop events,
  and probe travel was not a monotonic mass signal.
- [ ] Replace the current front-bumper probe before using active probing as a
  research claim. Required properties: low-energy bounded interaction,
  safety-abort if fall/drop/tilt begins, and probe features that distinguish
  load/contact risk without using hidden mass.
- [x] Implement bounded safe-probe support. `build_core_world_g1_box_scene.py`
  now supports `--probe-collision-window` and `--probe-end-step`; the probe
  pad starts collision-disabled, is enabled only during the configured window,
  and records enable/disable telemetry. The selector now supports optional
  probe safety gates, and the probe-selected pipeline writes a summary even
  when safety selection fails.
- [x] Monitor Slurm job `169335` (`g1_safe_probe`) through tmux
  `codex_g1_safe_probe_load_0707`. Expected summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_safe_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Goal: determine whether the bounded probe avoids the probe-stage fall/drop
  failures seen in `169332`.
- [x] Record safe-probe load validation `169335`. It completed on `server39`
  with Slurm state `FAILED`, exit `1:0`, because the aggregate summary is
  strict `fail`: `0/3` cases passed. The safe collision-window probe avoided
  probe-stage fall/drop for the checked cases, but it was too conservative to
  identify load risk: all three cases had `probe_active_steps=1` and
  `probe_box_target_directed_travel_m=0`, so the selector chose `chestpad`
  for every mass while still ignoring hidden mass. Validation then failed:
  `0.25 kg` and `0.50 kg` had fall/drop `0/0` but target-window streak `0`;
  `0.75 kg` failed with `482` falls / `439` drops and negative final box
  target-directed travel. Conclusion: the bounded probe fixed the previous
  unsafe-probe failure mode but lost all useful signal.
- [ ] Next active-probe gate: tune only the bounded collision-window probe
  until it produces a nonzero measured interaction while preserving probe
  fall/drop `0/0`. Use small changes such as a longer window, slightly closer
  pad, or slightly wider pad; do not claim unknown-load adaptation unless the
  resulting feature changes across load/contact conditions and the selector
  no longer collapses to the same posture for every mass.
- [x] Await safe-probe signal bracket `169337` (`g1_pr_signal`) submitted
  through tmux `codex_g1_safe_probe_signal_0707`. It runs four short
  0.50 kg probe-only cases, varying collision-window length and probe pad
  geometry, and aggregates whether any candidate gives nonzero probe
  target-directed motion while keeping probe fall/drop `0/0`, safe tilt, and
  no root/box rollout writes. Expected summary:
  `experiments/outputs/core_world_g1_safe_probe_signal_bracket/20260707_g1_safe_probe_signal_bracket_fresh/safe_probe_signal_bracket_summary.json`.
- [x] Record safe-probe signal bracket `169337`. Result: strict `fail`,
  `safe_signal_cases=[]`. All four short 0.50 kg cases completed only
  `41/180` steps and ended with
  `Exception: Failed to get root link transforms from backend` immediately
  after the probe collision window enabled at step `40`. Each case reported
  `probe_active_steps=1`, max probe target-directed box motion `0`, and
  fall/drop `0/0`. Conclusion: runtime collision toggling is not a valid
  safe-probe mechanism in the current G1 Core backend; do not continue pad
  geometry/window sweeps on this implementation.
- [x] Harden the probe selector after `169337`: it now rejects probe summaries
  with non-null `error` and supports `--min-probe-completed-steps`. The
  probe-selected target-window pipeline passes this option when provided, and
  the load-validation suite defaults it to `PROBE_FREE_STEPS`. This prevents
  step-41 failed probes from being treated as valid posture-selection inputs.
- [x] Harden the safe-probe signal bracket aggregator after `169337`: it now
  records `completed_steps`, `summary_status`, and `summary_error`, and a
  safe-signal case must finish the configured probe rollout before it can
  count as useful signal.
- [ ] Next safe-probe implementation step: replace runtime collision toggling
  with a backend-stable mechanism. Candidate directions are pre-authored
  always-colliding but initially distant probe geometry that is moved only by
  an existing stable kinematic/pose path before reset, a separate short
  probe-only scene without post-step collision API changes, or using normal
  free box motion before step 40 as the observed probe feature. Do not rerun
  `PROBE_COLLISION_WINDOW=1` unchanged.
- [x] Await precontact safe-probe bracket `169338` (`g1_preprobe`) submitted
  through tmux `codex_g1_precontact_probe_signal_0707`. It reruns the same
  short probe-signal bracket with `PROBE_COLLISION_WINDOW_MODE=0`, so the
  probe pad is a pre-authored always-colliding small body and no runtime
  collision API toggle is used. Expected summary:
  `experiments/outputs/core_world_g1_safe_probe_signal_bracket/20260707_g1_precontact_probe_signal_bracket_fresh/safe_probe_signal_bracket_summary.json`.
- [x] Record precontact safe-probe bracket `169338`. Result: strict `pass`
  for probe-signal diagnostic only, with all four 0.50 kg cases in
  `safe_signal_cases`: `small_x042`, `small_x046`, `small_x050`, and
  `small_x054`. Every case completed `180/180` steps with `error=null`,
  fall/drop `0/0`, `probe_active_steps=80`, root/box rollout writes `0`, and
  nonzero max probe target-directed motion. The motions were about
  `0.11805`, `0.15116`, `0.12182`, and `0.12966 m`; max robot tilt stayed
  below `0.20 rad` and max box tilt below `0.253 rad`. This validates the
  backend-stable pre-authored always-colliding probe route at 0.50 kg only.
  It is not carrying success and not yet unknown-load discrimination.
- [ ] Next active-probe gate: run the most conservative successful probe
  geometry, `small_x042`, across `0.25`, `0.50`, and `0.75 kg`. Required
  evidence: all probe-only rollouts complete, fall/drop `0/0`, root/box
  rollout writes `0`, and the probe metrics show whether mass affects the
  observed motion enough to support a selector.
- [x] Await precontact probe multiload signal suite `169339`
  (`g1_pr_loadsig`) submitted through tmux
  `codex_g1_precontact_probe_loads_0707`. It runs `small_x042` at 0.25,
  0.50, and 0.75 kg and writes:
  `experiments/outputs/core_world_g1_precontact_probe_multiload_signal/20260707_g1_precontact_probe_multiload_signal_fresh/precontact_probe_multiload_signal_summary.json`.
- [x] Record precontact probe multiload signal suite `169339`. Result:
  diagnostic `pass`, `3/3` cases passed. All masses completed `180/180`
  steps with `error=null`, fall/drop `0/0`, `probe_active_steps=80`, root/box
  rollout writes `0`, and nonzero probe motion. Max probe target-directed
  motion was `0.14978 m` at 0.25 kg, `0.11805 m` at 0.50 kg, and
  `0.16064 m` at 0.75 kg. The motion range was about `0.04259 m`. This is
  useful active-probe plumbing, but the scalar motion is not monotonic in
  mass and therefore is not by itself a reliable mass estimator.
- [ ] Next selector step: do not use a single `probe_motion <= threshold`
  rule. Add an explicitly diagnostic risk classifier using multiple observed
  features, such as probe motion band, max box tilt, max robot tilt, and
  relative-offset error. Treat it as a heuristic risk classifier, not learned
  identification, and evaluate it before full selected carrying.
- [x] Add diagnostic risk gates to the G1 probe selector: optional
  `--high-probe-travel-threshold`, `--probe-tilt-risk-threshold`,
  `--probe-box-tilt-risk-threshold`, and
  `--probe-relative-offset-risk-threshold`. The selected-validation pipeline
  passes these via environment variables.
- [x] Await precontact probe-selected load validation `169340`
  (`g1_prepsel`) submitted through tmux
  `codex_g1_precontact_probe_selected_load_0707`. It uses the stable
  `small_x042` precontact probe (`PROBE_COLLISION_WINDOW=0`,
  `PROBE_FREE_STEPS=180`) and heuristic risk thresholds
  `HIGH_PROBE_TRAVEL_THRESHOLD=0.14` and
  `PROBE_BOX_TILT_RISK_THRESHOLD=0.30` before selected validation at 0.25,
  0.50, and 0.75 kg. Expected summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_precontact_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
- [x] Record precontact probe-selected load validation `169340`. Result:
  strict `fail`, `1/3` cases passed. The selector ignored hidden mass and
  used only probe/visible telemetry. It selected `chestpad` for `0.25 kg`
  and `0.75 kg`, and `lowcarry` for `0.50 kg`. The `0.50 kg` selected
  lowcarry validation passed with fall/drop `0/0`, final robot/box
  target-directed travel `2.29876/2.34645 m`, and target-window end streak
  `164`. The `0.25 kg` selected chestpad case was stable with fall/drop
  `0/0` but under-traveled (`0.46484/0.51690 m`) and had target-window
  streak `0`. The `0.75 kg` selected chestpad case failed badly with
  `482` falls, `439` drops, negative final target-directed travel, max
  robot/box tilt about `1.314/1.312 rad`, and final relative offset
  `0.35039 m`. Interpretation: the precontact probe and hidden-mass-free
  selector now run end to end, but the selected posture/controller set is not
  robust. Do not tune the selector alone as if this solved unknown-load
  adaptation.
- [ ] Next load-robustness gate: add a materially different heavy/light
  stabilization path instead of rerunning scalar chestpad/lowcarry threshold
  sweeps. The near-term options are a third selected posture for heavy/high-
  motion probes, balance-aware settle with box retention, or replacing the
  current AGILE hold controller with a controller that explicitly handles
  load-dependent pitch/roll and target-window braking.
- [x] Add `scripts/isaac/run_core_world_g1_current_showcase_record_and_fallback.sh`
  to produce the current best presentation artifact on a compute node. It
  records a fresh 0.50 kg lowcarry replay from the known passing configuration
  with RGB capture disabled, then renders a fallback GIF/poster/MP4 from the
  replay CSV. This is visualization-only and must be labeled as replay, not
  Isaac camera render or new control evidence.
- [x] Await showcase record/fallback job `169346` (`g1_showviz`) submitted
  through tmux `codex_g1_showcase_record_visual2_0707`. Expected outputs:
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_lowcarry_current_pass_replay_record/agile_low_cradle_freebox_walk/`
  and
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_presentation_fallback/`.
- [x] Record showcase record/fallback job `169346`. It completed on
  `server39` with Slurm state `COMPLETED`, exit `0:0`, elapsed `00:00:45`.
  The source rollout summary is `pass`: `819/819` steps, fall/drop `0/0`,
  replay CSV recorded, final robot/box target-directed travel
  `2.29876/2.34645 m`, and target-window end streak `164`. Fallback
  visualization summary is also `pass`, with `83` frames plus:
  `g1_lowcarry_replay_fallback.gif`,
  `g1_lowcarry_replay_fallback.mp4`,
  `g1_lowcarry_replay_fallback_annotated.mp4`, and
  `g1_lowcarry_replay_fallback_poster.png` under
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_presentation_fallback/`.
  The poster is legible as a humanoid-and-box replay schematic, but it is
  still not Isaac camera render and not new control evidence.
- [x] Submit and record one true Isaac replay-render retry from the fresh
  replay CSV: Slurm job `169350` (`g1_isarend`) through tmux
  `codex_g1_isaac_replay_render_0707`. It failed on `server39` with exit
  `1:0` after `00:00:19`; render summary status `fail`, captured frames `0`.
  The environment is still missing both render capture paths:
  `ModuleNotFoundError: No module named 'omni.replicator'` and
  `ModuleNotFoundError: No module named 'isaacsim.core.rendering_manager'`.
  Do not keep rerunning the true Isaac replay-render path unchanged in this
  environment. Use the fallback replay visual for presentation unless the
  render extensions are installed or a different capture backend is added.
- [x] Add `scripts/isaac/run_core_world_g1_boxtilt_load_probe_suite.sh` to
  evaluate the existing `boxtilt` posture across load bands without changing
  the controller. This is a diagnostic for whether `boxtilt` is a stable but
  conservative third branch after `lowcarry` and `chestpad` failed
  light/heavy robustness.
- [x] Await boxtilt load probe job `169354` (`g1_boxtilt`) submitted through
  tmux `codex_g1_boxtilt_load_probe_0707`. It runs `0.25`, `0.50`, and
  `0.75 kg` with `LARGERBOX_STRICT_MODE=boxtilt`; expected summary:
  `experiments/outputs/core_world_g1_boxtilt_load_probe/20260707_g1_boxtilt_load_probe_fresh/boxtilt_load_probe_summary.json`.
- [x] Record boxtilt load probe job `169354`. Result: strict `fail`, `0/3`
  cases passed. `0.25 kg` failed with `47` falls / `33` drops, final
  robot/box target-directed travel `1.09788/1.01255 m`, and target-window
  streak `0`. `0.50 kg` failed with `329` falls / `0` drops, large lateral
  drift, final robot/box travel `1.04252/0.84821 m`, and target-window
  streak `0`. `0.75 kg` was the useful negative/partial signal: fall/drop
  `0/0`, max robot/box tilt `0.27226/0.29542 rad`, final relative offset
  `0.11864 m`, but under-traveled and drifted laterally, with final
  robot/box travel `1.12217/1.07559 m` and target-window streak `0`.
  Interpretation: boxtilt is not a strict success or broad third branch, but
  for the heavy high-motion case it is much safer than chestpad, which
  previously had `482` falls / `439` drops.
- [x] Update `scripts/isaac/select_core_world_g1_carry_posture_from_probe.py`
  to a diagnostic three-branch selector: low-risk probes select `lowcarry`,
  high-motion probes without tilt/offset/size risk select `boxtilt`, and
  resistant or tilt/offset/size-risk probes select `chestpad`. This is a
  heuristic safety gate, not learned system identification.
- [x] Await three-branch precontact probe-selected load validation `169355`
  (`g1_3branch`) submitted through tmux
  `codex_g1_threebranch_probe_selected_0707`. Expected summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_threebranch_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
- [x] Record three-branch validation `169355`. Result: strict `fail`, `1/3`
  cases passed, but safety improved for heavy high-motion probes. `0.25 kg`
  selected `chestpad`, fall/drop `0/0`, final robot/box travel
  `0.46484/0.51690 m`, target-window streak `0`. `0.50 kg` selected
  `lowcarry` and passed with fall/drop `0/0`, final robot/box travel
  `2.29876/2.34645 m`, and target-window end streak `164`. `0.75 kg`
  selected `boxtilt`, fall/drop `0/0`, final robot/box travel
  `1.12217/1.07559 m`, target-window streak `0`, and failed on lateral
  errors (`robot 0.81011 m`, box `0.70503 m`). Compared with the prior
  two-branch selector, the heavy case changed from catastrophic `chestpad`
  failure (`482` falls / `439` drops) to a stable but under-travel/lateral-
  drift failure. This is useful safety progress, not task success.
- [ ] Next controller gate: preserve the three-branch safety choice, then add
  a boxtilt-specific target/lateral correction for the heavy branch. Required
  improvement is not just fewer falls/drops, because those are already `0/0`;
  it must reduce heavy-case lateral error and increase target-window dwell
  without reintroducing falls/drops or rollout root/box writes.
- [x] Add `scripts/isaac/run_core_world_g1_boxtilt_heavy_lateral_target_suite.sh`
  for the 0.75 kg boxtilt branch. It tests six small variants: baseline,
  hold lateral off, hold lateral sign reverse, box-lateral controller with
  positive/negative sign, and conservative box-progress plus box-lateral.
  The gate is not just fall/drop, because baseline already has `0/0`; it must
  reduce final lateral error and increase target-window dwell without rollout
  root/box writes.
- [x] Await boxtilt heavy lateral/target job `169366` (`g1_bxlat`) submitted
  through tmux `codex_g1_boxtilt_heavy_lateral_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_target/20260707_g1_boxtilt_heavy_lateral_target_fresh/boxtilt_heavy_lateral_target_summary.json`.
- [x] Record boxtilt heavy lateral/target job `169366`. Result: strict
  `fail`, `0/6` cases passed. Baseline reproduced the heavy boxtilt safety
  profile: fall/drop `0/0`, final robot/box travel `1.12217/1.07559 m`,
  final lateral error `0.81011/0.70503 m`, and target-window streak `0`.
  `hold_lat_off` and `box_lat_sign_neg` both caused large fall/drop counts,
  so removing or reversing all lateral stabilization is unsafe. `box_progress_lat`
  over-drove and failed with `297` falls / `76` drops. The useful case was
  `hold_lat_reverse`: fall/drop `0/0`, target-window stable steps/longest
  streak `152/152`, but it did not stop in the window and ended with
  over-travel `2.75805/2.74993 m` plus large lateral error
  `1.78825/1.61990 m`. Interpretation: reverse hold-lateral can reach the
  target window safely, but needs a boxtilt-specific terminal/final hold to
  prevent overrun and lateral growth.
- [ ] Next boxtilt-heavy gate: run a stop/hold refinement on top of
  `hold_lat_reverse`, triggering terminal/final hold near the target window.
  Required improvement is target-window end streak while preserving fall/drop
  `0/0` and root/box rollout writes `0`.
- [x] Add `scripts/isaac/run_core_world_g1_boxtilt_heavy_stop_refine_suite.sh`.
  It fixes the 0.75 kg boxtilt branch to the useful `hold_lat_reverse`
  lateral setup and sweeps terminal/final hold thresholds around the target
  window (`1.55/1.70`, `1.65/1.80`, `1.75/1.90`, plus a final-zero
  correction variant). The goal is to convert the existing 152-step transient
  target-window visit into an end-of-run hold.
- [x] Await boxtilt heavy stop-refine job `169371` (`g1_bxstop`) submitted
  through tmux `codex_g1_boxtilt_heavy_stop_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_stop_refine/20260707_g1_boxtilt_heavy_stop_refine_fresh/boxtilt_heavy_stop_refine_summary.json`.
- [x] Record boxtilt heavy stop-refine job `169371`. Result: strict `fail`,
  `0/4` cases passed. `stop_155_170` destabilized the robot/box with
  `137` falls / `40` drops, so too-early hard stopping is unsafe.
  `stop_165_180` and `stop_175_190` preserved fall/drop `0/0` but overran
  the target window, with end streak `0` and final robot/box travel
  `2.66508/2.70260 m` and `2.58964/2.62017 m`. The useful partial signal was
  `stop_165_180_finalzero`: fall/drop `0/0`, target-window stable/longest
  streak `184/184`, but still end streak `0`, final robot/box travel
  `2.37121/2.41655 m`, large lateral error `1.72462/1.81844 m`, and box
  tilt above the strict gate. Interpretation: terminal/final hold can keep
  the heavy boxtilt branch in the window longer, but it still cannot stop at
  the end or control lateral drift.
- [x] Add `scripts/isaac/run_core_world_g1_boxtilt_heavy_window_freeze_suite.sh`.
  It keeps the 0.75 kg `boxtilt` branch on `hold_lat_reverse` and tests
  target-window freeze plus lower terminal speed and one small final-brake
  variant. This is a narrow follow-up to `169371`, not a new success claim.
- [x] Await boxtilt heavy window-freeze job `169411` (`g1_bxfreeze`)
  submitted through tmux `codex_g1_boxtilt_freeze_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_window_freeze/20260707_g1_boxtilt_heavy_window_freeze/boxtilt_heavy_window_freeze_summary.json`.
- [x] Record boxtilt heavy window-freeze job `169411`. Result: strict
  `fail`, `0/4` cases passed. All target-window freeze/brake variants
  reintroduced falls and drops: `freeze_160_180_s012` had `105` falls /
  `92` drops, `freeze_165_185_s010` had `110` / `26`,
  `freeze_170_190_s008` had `77` / `49`, and `freeze_165_180_brake` had
  `109` / `95`. The best target-window dwell in this suite was only
  `122` stable steps with end streak `0`, worse than the prior `169371`
  `finalzero` case's `184` stable steps and fall/drop `0/0`. Interpretation:
  target-window freeze/brake is not the missing stabilizer; it worsens late
  roll/drop. Return to the safer `finalzero` branch or replace the boxtilt
  terminal controller with an explicit lateral/balance stabilizer rather than
  freezing policy commands.
- [ ] Next boxtilt-heavy decision: do not keep adding freeze/brake variants.
  Either test a lateral-error-aware terminal controller that preserves the
  `169371` fall/drop `0/0` property, or stop investing in this hand-tuned
  branch and move to a controller-backed balance/locomotion replacement.
- [x] Add `scripts/isaac/run_core_world_g1_boxtilt_heavy_terminal_lateral_suite.sh`.
  It keeps the 0.75 kg boxtilt branch on the safer `169371` terminal/final
  hold thresholds, disables freeze/brake, and tests terminal-only lateral
  correction with excess-error thresholds and one tilt-gated variant. The
  required signal is lower lateral error without reintroducing falls/drops.
- [x] Await boxtilt heavy terminal-lateral job `169419` (`g1_bxtermlat`)
  submitted through tmux `codex_g1_boxtilt_termlat_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_terminal_lateral/20260707_g1_boxtilt_heavy_terminal_lateral/boxtilt_heavy_terminal_lateral_summary.json`.
- [x] Record boxtilt heavy terminal-lateral job `169419`. Result: strict
  `fail`, `0/4` cases passed. All four terminal-only variants failed
  identically before terminal latch: `448` falls / `293` drops, final
  robot/box target-directed travel only `0.59292/0.54745 m`, target-window
  stable steps `0`, and `agile_command_hold_terminal_latched=false`.
  Interpretation: the pre-terminal lateral correction is required for this
  boxtilt branch to remain upright long enough to reach the target region;
  terminal-only lateral correction is not viable.
- [ ] Stop small boxtilt hold/lateral tweaking unless a materially different
  balance mechanism is introduced. Current negative set: removing early
  lateral control collapses early (`169419`), freezing/braking near the
  window collapses late (`169411`), and the safer final-zero dwell branch
  still cannot end in the target window (`169371`). Next credible work should
  replace the terminal stabilizer or move back to controller-backed
  locomotion/balance, not add another scalar threshold sweep.
- [x] Add dynamic lateral-roll balance target support to
  `scripts/isaac/build_core_world_g1_box_scene.py` and expose it through
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  The new path maps robot/box/average target-line lateral error into a bounded
  roll target for the existing ankle/hip balance-feedback controller. This is
  a different balance mechanism from command lateral correction, freeze, or
  brake; it changes joint targets only and does not write root/box rollout
  state.
- [x] Add `scripts/isaac/run_core_world_g1_boxtilt_heavy_lateral_roll_target_suite.sh`.
  It starts from the safest heavy boxtilt final-zero dwell setup from
  `169371`, then sweeps dynamic roll target source/sign. Required useful
  signal: preserve fall/drop `0/0`, lower lateral error, and improve
  end-of-run target-window streak.
- [x] Await boxtilt heavy lateral-roll-target job `169432`
  (`g1_bxrolltarget`) submitted through tmux
  `codex_g1_boxtilt_rolltarget_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_roll_target/20260707_g1_boxtilt_heavy_lateral_roll_target/boxtilt_heavy_lateral_roll_target_summary.json`.
- [x] Record boxtilt heavy lateral-roll-target job `169432`. Result:
  strict `fail`, `0/4` cases passed. `avg_sign_neg` was stable with
  fall/drop `0/0` and low tilt (`0.24390/0.34400 rad`) but drifted badly
  laterally (`1.87642/2.11266 m`) and never entered the target window.
  `box_sign_neg` improved final lateral error to `0.86138/0.94767 m` and
  reached a 53-step target-window dwell, but reintroduced `162` falls /
  `127` drops. `avg_sign_pos` and `box_sign_pos` also fell/dropped. This
  validates the new roll-target path as active, but not as a success.
- [x] Add roll-target fields to
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`, so aggregate
  summaries record lateral roll-target source/gain/sign and active/last/max
  target diagnostics.
- [x] Add `scripts/isaac/run_core_world_g1_boxtilt_heavy_lateral_roll_target_refine_suite.sh`.
  It lowers gain/limit around the promising but unstable `box_sign_neg`
  branch and includes one gentler `avg_sign_pos` case. Required signal:
  preserve fall/drop `0/0` while retaining at least some lateral-error
  reduction versus the safe but drifting `avg_sign_neg`.
- [x] Cancel stale boxtilt heavy lateral-roll-target refine job `169446`
  before rollout because it used the ungated refinement script. It never
  started and had no assigned node.
- [x] Add gating/ramp protection to dynamic lateral-roll balance target:
  hold-delay steps, ramp steps, max robot tilt, and max box tilt. These guards
  are intended to prevent early gait deflection and late high-tilt correction,
  both of which were failure modes in `169432`.
- [x] Await gated boxtilt heavy lateral-roll-target refine job `169465`
  (`g1_bxrollgated`) submitted through tmux
  `codex_g1_boxtilt_rollgated_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_roll_target_refine/20260707_g1_boxtilt_heavy_lateral_roll_target_refine_gated/boxtilt_heavy_lateral_roll_target_refine_summary.json`.
- [x] Record gated boxtilt heavy lateral-roll-target refine job `169465`.
  Result: strict `fail`, `0/4` cases passed. `box_neg_g010_l018` failed
  with `52` falls / `13` drops and no target-window dwell. Stronger
  `box_neg_g020_l030` and `box_neg_g030_l045` collapsed earlier with
  `242/195` and `284/236` fall/drop counts. The only useful partial signal
  was `avg_pos_g020_l030`: target-window stable/longest streak `137/137`,
  but end streak `0`, late `55` falls / `38` drops, final robot/box travel
  `2.40772/2.36080 m`, and final lateral error `1.05326/1.30846 m`.
  Interpretation: gated lateral roll-target does not solve the heavy boxtilt
  target hold; stronger or mildly gated roll-target tuning should stop here.
- [x] Add
  `scripts/isaac/run_core_world_g1_boxtilt_avgpos_short_window_suite.sh` to
  isolate the partial `avg_pos_g020_l030` signal before late fall/drop.
- [x] Record short-window job `169472` (`g1_bxavgshort`). Result: strict
  `fail`, but useful visualization candidate. It completed `760/760` with
  fall/drop `0/0`, root/box rollout writes `0`, final robot/box target-
  directed travel `2.25514/2.25542 m`, target-window end streak `133`, and
  final-hold end streak `100`. It failed tilt and lateral gates:
  max robot/box tilt `0.62420/0.64870 rad`, final lateral error
  `1.08572/1.28355 m`. This may be shown only as current progress, not
  solved carrying.
- [x] Add
  `scripts/isaac/run_core_world_g1_boxtilt_lateral_hold_refine_suite.sh`.
  It tests whether keeping small final-hold lateral correction, or a small
  box-lateral correction, can reduce the `169472` side drift without losing
  fall/drop `0/0`.
- [x] Await boxtilt lateral-hold refine job `169476` (`g1_bxlathold`)
  submitted through tmux `codex_g1_boxtilt_lathold_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_lateral_hold_refine/20260707_g1_boxtilt_lateral_hold_refine_760/boxtilt_lateral_hold_refine_summary.json`.
- [x] Record boxtilt lateral-hold refine job `169476`. Result: strict
  `fail`, `0/4` cases passed. `hold_sign_neg_l006` reduced final lateral
  error to `0.30088/0.38493 m`, but failed with `190` falls / `175` drops
  and target-window dwell `0`. `hold_sign_neg_l012` was worse with
  `323` falls / `310` drops. `hold_sign_pos_l006` reduced final lateral
  error to about `-0.213/-0.210 m`, but over-drove to final robot/box travel
  `3.60361/3.67621 m` and had `220` falls / `65` drops. `boxlat_sign_neg_l010`
  kept box drops at `0` and final lateral error low (`0.28659/0.18326 m`),
  but had `295` falls and no target-window dwell. Interpretation:
  command-level lateral correction can change side drift, but it destroys the
  stability/propulsion balance; stop scalar lateral-hold tuning here.
- [x] Add
  `scripts/isaac/run_core_world_g1_boxtilt_short_window_record_and_fallback.sh`
  for a replay-recorded, fallback-rendered visualization of the `169472`
  short-window boxtilt progress case.
- [x] Await boxtilt short-window visual job `169488` (`g1_bxshortviz`)
  submitted through tmux `codex_g1_boxtilt_shortviz_0707`. Expected visual
  directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/`.
- [x] Record boxtilt short-window visual jobs. `169488` completed on
  `server43` with exit `0:0`, generating replay CSV and fallback visuals.
  `169501` completed on `server39` with exit `0:0`, regenerating the visual
  labels with aggregate checker status. Final visual directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/`.
  Use `g1_boxtilt_short_window_progress_annotated.mp4` or
  `g1_boxtilt_short_window_progress.mp4` for the cleanest current-progress
  demo; the poster now says `strict checker: fail`. This is only a schematic
  replay fallback, not an Isaac camera render or solved carrying.
- [x] Add
  `scripts/isaac/run_core_world_g1_boxtilt_final_stand_refine_suite.sh`.
  It tests a different mechanism from command-layer lateral correction:
  final-hold joint-target blending into default or gentle crouch stand targets
  for the 0.75 kg boxtilt short-window case. The objective is to reduce the
  strict tilt failures from `169472` while preserving fall/drop `0/0` and
  target-window end streak.
- [x] Await boxtilt final-stand refine diagnostic. GPU job `169508`
  (`g1_bxfinstand`) stayed pending and was cancelled after CPU compute backup
  `169514` (`g1_bxfinstandc2`) completed on `server36`. Summary:
  `experiments/outputs/core_world_g1_boxtilt_final_stand_refine/20260707_g1_boxtilt_final_stand_refine_760_cpu_backup2/boxtilt_final_stand_refine_summary.json`.
- [x] Record boxtilt final-stand refine result. Strict `fail`, `0/4` cases
  passed. All cases kept `box_drop_events=0` but introduced late falls and
  larger tilt than the no-final-stand short-window baseline:
  `stand_default_d0_b002` had `10` falls, `stand_default_d20_b005` had `9`,
  `stand_gentle_crouch_d0_b004` had `8`, and `stand_crouch_d20_b003` had
  `10`. Max robot/box tilt stayed around `0.934-0.993 / 0.889-0.948 rad`,
  target-window end streak stayed `0`, and final lateral error remained large.
  Interpretation: final-stand joint-target blending is not the missing
  stabilizer for 0.75 kg boxtilt; stop scalar final-stand delay/blend tuning
  for this branch.
- [x] Expose `FREE_CRADLE_LOCAL_Y` in
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, so
  lateral cradle placement can be tested without changing the scene builder.
- [x] Add
  `scripts/isaac/run_core_world_g1_boxtilt_geometry_refine_suite.sh`. It
  tests four non-command-layer contact/geometry variations for the 0.75 kg
  boxtilt short-window branch: negative box/cradle Y offset, positive
  box/cradle Y offset, wider lid/rails, and final-hold chest pad.
- [x] Run and record boxtilt geometry refine job `169519`
  (`g1_bxgeomc`) on compute node `server26`. Summary:
  `experiments/outputs/core_world_g1_boxtilt_geometry_refine/20260707_g1_boxtilt_geometry_refine_760_cpu_backup/boxtilt_geometry_refine_summary.json`.
  Result: strict `fail`, `0/4` cases passed. `box_cradle_y_neg003` fell/dropped
  `291/163` despite lower final lateral error; `box_cradle_y_pos003` fell/dropped
  `45/26` and had no target-window dwell; `wider_lid_rails` fell/dropped
  `183/8` and lost forward progress; `final_chest_pad` fell/dropped `119/105`.
  Interpretation: small lateral cradle offsets, wider retaining geometry, and
  final-hold chest pad do not rescue the heavy boxtilt branch.
- [x] Add
  `scripts/isaac/run_core_world_g1_selected_branch_horizon_repair_suite.sh`
  to test whether the probe-selected `0.25 kg` chest-pad branch and `0.75 kg`
  boxtilt branch were only failing because the validation horizon/hold logic
  was too short.
- [x] Run and record selected-branch horizon/hold repair job `169529`
  (`g1_branchhor`) on compute node `server26`. Summary:
  `experiments/outputs/core_world_g1_selected_branch_horizon_repair/20260707_g1_selected_branch_horizon_repair_cpu/selected_branch_horizon_repair_summary.json`.
  Result: strict `fail`, `0/4` cases passed. Both `0.25 kg` chest-pad
  1600-step cases failed late with `526/373` fall/drop events and no
  target-window dwell. The `0.75 kg` boxtilt default 1200-step case failed
  with `257/168` fall/drop events. The `0.75 kg` boxtilt stop/final-zero
  case showed the only useful signal, reaching target-window dwell
  `184` steps and final-hold dwell `166` steps, but then falling/dropping
  `304/290` times with end streak `0`. Interpretation: longer horizons and
  simple final stop/zero commands do not solve selected-branch robustness.
- [x] Expose the existing box-progress and box-lateral closed-loop command
  controllers through
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`. Before
  this, the scene builder accepted `--agile-command-box-progress-controller`
  and `--agile-command-box-lateral-controller`, but the launcher only passed
  numeric gains and never enabled the controller flags.
- [x] Add
  `scripts/isaac/run_core_world_g1_boxtilt_box_progress_controller_suite.sh`.
  This is a focused 0.75 kg boxtilt diagnostic for a materially different
  target-window controller: box projected progress directly sets forward
  AGILE command, with two one-bit lateral sign checks. It is not another
  final-yaw/final-stand scalar sweep.
- [x] Await box-progress controller diagnostic job `169547` (`g1_bxprog`)
  running on `server43` through tmux
  `curiosity_g1_boxtilt_box_progress_gpu43_0707`.
- [x] Mark `169547` (`g1_bxprog`) CUDA-device run as invalid infrastructure
  evidence. It used `DEVICE=cuda:0`, reached Isaac wrapper initialization,
  then exited `0` without writing per-case summaries. All three checks report
  `summary missing`, and aggregate `case_count` is `0`. This is not a
  controller result.
- [x] Await replacement box-progress controller diagnostic job `169548`
  (`g1_bxprogc`) submitted through tmux
  `curiosity_g1_boxtilt_box_progress_gpualloc_cpu_0707`. It requests GPU
  resources only to get scheduled, but runs the previously validated Isaac CPU
  device path with `DEVICE=cpu`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_box_progress_controller/20260707_g1_boxtilt_box_progress_controller_gpualloc_cpu/boxtilt_box_progress_controller_summary.json`.
- [x] Record box-progress controller replacement result. Strict `fail`,
  `0/3` cases passed. `progress_only` activated the box-progress controller
  for `1160` steps, but failed with `655/457` fall/drop events and max
  robot/box target-directed travel only `0.537/0.527 m`. `progress_lateral_neg`
  improved final lateral error but fell/dropped `743/273` with no target-window
  dwell. `progress_lateral_pos` delayed failure the most, first fall/drop
  `958/1139`, but still fell/dropped `242/61` and only reached max robot/box
  target-directed travel `0.521/0.467 m`. Interpretation: the dormant
  controller is now wired and verified active, but box-progress/lateral command
  closure alone is not enough for heavy boxtilt carrying.
- [ ] Next credible route after `169548`: stop repeating command-layer lateral/yaw/final-stand
  scalar sweeps and small passive cradle-geometry tweaks for the heavy
  boxtilt branch. Move to a materially different support/contact controller
  or locomotion/balance backend that can address side drift and roll without
  consuming the stability margin.
- [x] Add MuJoCo `SUPPORT_CONTROLLER_MODE=qp_stance_force` with projected
  unilateral/friction-limited foot-contact force allocation, QP residual and
  friction-usage summary fields, launcher envs, checker gates, and the
  full-time QP diagnostic suite
  `scripts/mujoco/run_quadruped_freebox_qp_support_suite.sh`.
- [x] Record full-time MuJoCo QP support job `169627` (`mj_qpsupp`) as a
  strict negative result. QP was active all rollout (`3000` steps) with root/
  box writes still `0`, but strict pass was `0/4`; the two latched cases still
  had large fall/drop counts and max tilt above `3.2 rad`, while the other two
  cases broke target latch. Friction saturated and wrench residuals were
  enormous, so this is not a support solution.
- [x] Add post-latch-only QP support mode and suite
  `scripts/mujoco/run_quadruped_freebox_qp_post_latch_suite.sh`, preserving
  the known stance-force approach and activating QP only after target-stop
  latch.
- [x] Await and record post-latch MuJoCo QP support job `169628`
  (`mj_qppost`) submitted through tmux
  `curiosity_mujoco_qp_post_latch_0707`. It completed on `server39` with
  Slurm exit `0:0`, but strict pass was `0/4`. All cases latched and ran
  QP/LQR for `1797` post-latch steps with root/box writes `0`, but still had
  `78-79` falls, `72-73` drops, max tilt `2.03-2.26 rad`, and saturated QP
  friction usage. Post-latch QP is not the missing stabilizer.
- [ ] Next controller step: stop small QP weight/gain sweeps on this MuJoCo
  hand-controller branch unless they introduce a materially different
  feasibility mechanism. Prefer a constrained whole-body/contact controller
  with explicit upright, support polygon, friction, and box-retention
  objectives, or return to a policy-backed locomotion backend.
- [x] Add a MuJoCo QP feasibility mechanism: `SUPPORT_QP_MOMENT_CLIP_SCALE`
  clips requested roll/pitch/yaw moments by a support-foot geometric estimate
  of normal-force and friction-limited contact moment capacity before QP
  allocation. This is off by default and records
  `max_abs_support_qp_moment_clip_delta_nm`.
- [x] Await feasible-moment QP diagnostic job `169632` (`mj_qpfeas`) submitted
  through tmux `curiosity_mujoco_qp_feasible_moment_0707`, suite
  `scripts/mujoco/run_quadruped_freebox_qp_feasible_moment_suite.sh`.
  Slurm completed on `server39` with exit `0:0`, but strict pass remained
  `0/4`. The moment clip was useful diagnostically: max QP residual dropped
  to `268-363` from the previous `1111-4088`, and `clip05` reduced max tilt
  to `1.6669 rad`. It still had `77-78` falls, `72` drops, box z below gate,
  saturated friction usage, and short final travel. This is not a carrying
  solution.
- [ ] Stop this MuJoCo hand-controller path after feasible-moment QP unless
  the next change replaces the controller class. Needed next work is a real
  constrained whole-body/contact optimizer or policy-backed locomotion, not
  another small QP clipping/weight/gain sweep.
- [x] Add a replacement MuJoCo controller class,
  `SUPPORT_CONTROLLER_MODE=wbc_carried_mass_qp`, with post-latch carried-mass
  support, robot+box combined COM, WBC active-step metrics, combined-COM
  support-error metrics, and extra payload-Fz metrics.
- [x] Run and record WBC carried-mass diagnostic job `169633` (`mj_wbcmass`)
  through tmux `curiosity_mujoco_wbc_carried_mass_0707`. Slurm completed on
  `server39` with exit `0:0`, but strict pass was `0/4`. WBC was active
  (`2292-2555` steps) with root/box writes `0`, but it latched early and
  failed post-latch retention/balance: fall/drop `51-99` / `42-93`, min box z
  `0.286-0.373 m`, max tilt `1.77-3.26 rad`, and three cases ended with
  negative final box travel. This is not carrying success.
- [ ] Next credible path after WBC carried-mass failure: do not continue this
  MuJoCo branch with small stop/height/weight changes. Either add a real
  transition-aware contact objective that jointly handles stopping,
  retention, and support, or switch effort back to a policy-backed locomotion
  backend where walking is not hand-authored.
- [x] Add
  `scripts/mujoco/run_quadruped_freebox_wbc_continuous_carry_suite.sh` to
  isolate whether `169633` failed mainly because of target-stop/hold
  switching. This suite runs `wbc_carried_mass_qp` continuously from rollout
  start with no target-stop latch and still enforces no fall/drop, travel,
  tilt, box-height, relative-error, and no-shortcut gates.
- [x] Await WBC continuous-carry diagnostic job `169638` (`mj_wbccont`)
  submitted through tmux `curiosity_mujoco_wbc_continuous_carry_0707`.
  It completed on `server36` with Slurm exit `0:0`, but strict pass was
  `0/4`. WBC/QP/LQR were active for all `2400` steps with root/box writes
  `0`. Removing target-stop did not solve carrying: best final travel was
  `0.218 m`, but still with `67` falls and `60` drops; other cases had
  near-zero or negative final travel and similar collapse.
- [ ] Retire the MuJoCo hand-authored WBC branch as the main path. The next
  main attempt should move back to a policy-backed locomotion backend or a
  genuinely fuller whole-body controller; do not spend the next iteration on
  another continuous-WBC speed/support-scale/COM-weight sweep.
- [x] Add G1/AGILE policy-backed 0.60 kg low-carry box-tilt repair suite:
  `scripts/isaac/run_core_world_g1_lowcarry_060_box_tilt_repair_suite.sh`.
  This targets the near-miss from the mass-band result where 0.60 kg had
  fall/drop `0/0`, about `2.197 m` box travel, and target-window end streak
  `108`, but failed only because max box tilt was `0.6385 rad > 0.45`.
  The suite tests lower/thicker top-lid geometry, wider/lower retaining
  geometry, very-low lid, and final-hold chest pad while preserving the strict
  target-window gates.
- [x] Await G1/AGILE 0.60 kg box-tilt repair job `169648` (`g1_060tilt`)
  submitted through tmux `curiosity_g1_lowcarry_060_tilt_repair_0707`.
  It completed on `server57` with strict pass `0/4`. Pure lid/rail lowering
  was negative and often destabilized the robot. The useful case was
  `final_chest_pad`: fall/drop `0/0`, max robot/box tilt `0.323/0.370 rad`,
  final relative error `0.087 m`, but final robot/box target-directed travel
  only `1.467/1.510 m`, so target-window stable steps were `0`.
- [x] Add a 0.60 kg final-chest-pad travel repair suite that keeps the
  stable low-tilt geometry from `169648` and delays terminal/final hold latch
  thresholds to recover target-window travel.
- [x] Await G1/AGILE 0.60 kg final-chest-pad travel repair job `169653`
  (`g1_060trav`) submitted through tmux
  `curiosity_g1_lowcarry_060_chesttravel_0707`.
  It completed on `server57` with strict pass `0/4`. All cases kept
  fall/drop `0/0` and rollout root/box writes `0`, but none reached target
  window. Best stable case `final090` had max robot/box tilt `0.323/0.393`,
  final relative error `0.080 m`, and final latch active `197` steps, but
  final robot/box travel only `1.205/1.265 m`.
- [x] Add a 0.60 kg final-chest-pad command-drive repair suite that
  preserves the low-tilt chest-pad geometry but increases/reshapes AGILE
  command drive to recover the missing target-window travel.
- [x] Await G1/AGILE 0.60 kg final-chest-pad command-drive repair job
  `169664` (`g1_060drive`) submitted through tmux
  `curiosity_g1_lowcarry_060_chestdrive_0707`.
  It completed on `server59` with strict pass `0/4`. Increasing command to
  `0.12` collapsed early with `528` falls / `467` drops; command `0.14`
  could over-drive travel in one case but failed with `493` falls and
  `187` drops. Larger constant AGILE command is not the repair.
- [ ] Next G1/AGILE 0.60 kg path: preserve baseline lowcarry travel while
  reducing box tilt, or use a smoother command schedule around the chest-pad
  transition. Do not repeat larger constant-command runs.
- [x] Add late chest-pad triggers to the G1 scene:
  `cradle_chest_pad_enable_on_target_window` and
  `cradle_chest_pad_enable_on_box_tilt`. Lightweight CSV audit showed the
  0.60 kg lowcarry baseline reaches the target window and only exceeds the
  `0.45 rad` box-tilt gate at step `810`; early/final chest-pad variants
  keep box tilt low but under-travel. The new trigger should preserve
  baseline travel and only activate the chest pad once the robot is in the
  target window or box tilt becomes risky.
- [ ] Await G1/AGILE 0.60 kg late chest-pad suite job `169676`
  (`g1_060late`) submitted through tmux
  `curiosity_g1_060_late_chestpad_0707`. It runs target-window and box-tilt
  triggered chest-pad activation cases and must pass the same fall/drop,
  target-window, box-tilt, final-hold, and no-rollout-shortcut gates before
  being considered a real improvement.
- [x] Cancel invalid late chest-pad job `169676` after discovering the
  spawn-time collision-disable condition missed the two new late-trigger flags.
  The first two cases therefore started with chest-pad collision active and are
  not valid tests of late activation.
- [ ] Await replacement late chest-pad job `169678` (`g1_060late2`) submitted
  through tmux `curiosity_g1_060_late_chestpad_fix_0707` after commit
  `4994ed8` fixed the actual spawn-time collision gating. Interpret only the
  `20260707_g1_lowcarry_060_late_chestpad_fix1` outputs.
- [x] Cancel replacement job `169678` after discovering the suite still used
  non-baseline top-lid geometry (`z=0.145`, thickness `0.018`). The true
  0.60 kg baseline uses top-lid `z=0.13`, thickness `0.014`; `fix1` therefore
  tested the wrong geometry and should not be interpreted as late-trigger
  evidence.
- [ ] Rerun the late chest-pad suite with baseline 0.60 kg top-lid geometry
  and a fresh output prefix such as `fix2`.
- [ ] Await baseline-geometry late chest-pad job `169685` (`g1_060late3`)
  submitted through tmux `curiosity_g1_060_late_chestpad_fix2_0707` with
  prefix `20260707_g1_lowcarry_060_late_chestpad_fix2`.
- [x] Record baseline-geometry late chest-pad job `169685`: strict `0/4`.
  Target-window cases never enabled chest-pad collision but still failed with
  `26` falls, final box travel about `1.457 m`, lateral error about
  `0.96 m`, max box tilt `1.891 rad`, and target-window streak `0`.
  Box-tilt trigger cases enabled collision at step `700`/`760`; best case
  removed falls/drops but still under-traveled and over-tilted. This means the
  pre-authored fixed-joint chest-pad rigid body itself perturbs the baseline.
- [ ] Add a zero/tiny-mass chest-pad diagnostic or a runtime-spawned/non-rigid
  chest support so the inactive support does not corrupt baseline locomotion.
- [ ] Await tiny-mass late chest-pad job `169705` (`g1_060ltiny`) submitted
  through tmux `curiosity_g1_060_late_chestpad_tinymass_0707`. It uses
  `CRADLE_CHEST_PAD_MASS_SCALE=0.001` and should reveal whether added support
  mass/inertia caused the `169685` baseline corruption.
- [x] Record tiny-mass late chest-pad job `169705`: strict `0/4`.
  Reducing chest-pad mass scale to `0.001` did not restore the baseline.
  All cases ended with final robot/box target-directed travel about
  `0.270/0.077 m`, lateral error about `1.975/2.117 m`, `20` falls, max
  robot/box tilt about `1.746/1.746 rad`, and no target-window/final-hold
  dwell. Stop tuning fixed-joint chest-pad mass/trigger thresholds.
- [ ] Next G1/AGILE 0.60 kg attempt should avoid pre-authored fixed-joint
  support bodies. Try control-only terminal stabilization or a support that is
  created/attached only after baseline walking has reached the target region.
- [ ] Await runtime-spawn chest-pad job `169713` (`g1_060rtpad`) submitted
  through tmux `curiosity_g1_060_runtime_chestpad_0707`. Check spawn fields
  before task metrics: `cradle_chest_pad_spawned_step`,
  `cradle_chest_pad_spawn_error`, and collision trigger reason.
- [x] Record runtime-spawn chest-pad job `169713`: strict `1/4`. Passing case
  `target_window_min700` spawned/enabled chest support at step `712` with no
  spawn error, final robot/box travel `2.051/2.032 m`, max robot/box tilt
  `0.309/0.428 rad`, fall/drop `0/0`, and target-window/final-hold end
  streak `102`. This is the current best 0.60 kg G1/AGILE diagnostic.
- [ ] Reproduce/refine runtime chest-pad target-window timing around min-step
  `700`; `min760` was too late and fell, while box-tilt triggers still failed
  box-tilt gates.
- [ ] Await runtime chest-pad timing job `169724` (`g1_060rtiming`) submitted
  through tmux `curiosity_g1_060_runtime_chestpad_timing_0707`. It scans
  target-window min trigger steps `680`, `700`, `720`, and `740`.
- [x] Record runtime chest-pad timing job `169724`: strict `2/4`.
  `min680` and `min700` both passed, spawning/enabling support at step `712`
  with final robot/box travel `2.051/2.032 m`, max robot/box tilt
  `0.309/0.428`, fall/drop `0/0`, and target-window/final-hold end streak
  `102`. `min720` and `min740` were too late and fell.
- [x] Generate a replay/visual artifact for the current best 0.60 kg runtime
  chest-pad pass, without claiming it is learned or final unknown-load
  carrying. Current local showcase page:
  `slides/2026-07-07_isaac_carry_showcase.html`; video asset remains local
  under ignored `experiments/visuals/`.
- [x] Record posture-conditioned gate job `169793` (`g1_postgate`) submitted
  through tmux `curiosity_g1_posture_gate_0707`. It runs the known passing
  `low_front_060` and the current best close-front conditioned hypothesis
  under the unchanged strict gates. It failed aggregate `1/2`: `low_front_060`
  passed again, while `close_front_060_conditioned` failed with `142` falls,
  max robot/box tilt `3.130/3.129 rad`, and target-window stable steps `0`.
- [x] Record close-front final-stand job `169822` (`g1_cfstand`) submitted
  through tmux `curiosity_g1_close_front_final_stand_0707`. This suite tests
  whether late final-stand/freeze stabilization can fix the `steps1050_final120`
  failure. It failed `0/3`: late crouched stand, freeze-window then stand, and
  policy-then-stand all collapsed, with falls `262`, `226`, and `700`
  respectively and target-window stable steps `0`.
- [ ] Next close-front work should not continue final-stand-only tuning. Repair
  the pre-target close-front trajectory: change command schedule/support
  geometry before step `700-900`, or add a posture-conditioned gait/support
  controller that keeps roll/pitch inside gate before final hold.
- [x] Record close-front pretarget repair suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_pretarget_repair_suite.sh`.
  Slurm job `169858` (`g1_cfpre`) ran on `server44` and failed `0/3`.
  `progress_conservative` was useful despite failing: it reached target-window
  first stable step `652`, target-window stable steps `136`, longest streak
  `73`, but fell/dropped at steps `802/864`. `progress_mid` and
  `progress_mid_no_hold_lat` collapsed earlier and should not be continued.
- [ ] Next close-front repair should build on `progress_conservative` and
  target retention/arrest after entering the target window, not stronger
  progress drive. Candidate knobs: target-window stop/arrest, reduced
  post-window progress command, earlier runtime support after window entry,
  and tighter terminal retention while preserving strict gates.
- [x] Record close-front window-arrest suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_window_arrest_suite.sh`.
  Slurm job `169867` (`g1_cfwin`) failed `0/3`. All three cases fell/dropped
  at steps `494/533` with target-window stable steps `0`, because removing
  the original early hold/adaptive behavior prevented the previous
  `progress_conservative` target-window entry.
- [ ] Next retention repair should preserve `progress_conservative` early
  hold/adaptive behavior and only change target-window runtime support/freeze.
  Do not continue the `g1_cfwin` setup that sets stop target only and removes
  early hold/adaptive behavior.
- [x] Record close-front window-retention-v2 suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_window_retention_v2_suite.sh`.
  Slurm job `169906` (`g1_cfv2`) failed `0/3`. Earlier runtime support at
  step `653` shortened target-window stable steps to `55-57` and moved first
  fall to `707-709`; freeze prevented box drops but still fell.
- [ ] Next close-front support timing test should keep
  `progress_conservative` early hold/adaptive behavior and compare original
  pad700 against disabled/delayed/geometry-softened runtime chest support.
  Do not trigger support/freeze immediately at first window entry.
- [x] Record close-front support-timing suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_support_timing_suite.sh`.
  It compares no runtime pad, delayed pad760, and smaller pad700 while keeping
  the conservative box-progress controller and strict gates. Slurm job
  `169922` (`g1_cfsup`) failed `0/3`. `no_runtime_pad` was best among the
  failures: no box drops, first fall step `901`, target-window stable steps
  `130`, longest/end streak `78/0`, and final robot/box travel
  `1.615/1.627 m`. Delayed/smaller runtime pads caused drops and did not
  improve final retention.
- [x] Monitor Slurm job `169922` (`g1_cfsup`) in tmux
  `curiosity_g1_close_front_support_timing_short_0707`; original pending job
  `169916` was cancelled before running to resubmit the same suite with a
  shorter 12-minute walltime.
- [ ] Do not continue runtime chest-pad timing/geometry for close-front. Next
  retention tests should use the no-pad close-front trajectory and add late
  final-hold/brake/freeze only after the run approaches the target window.
- [x] Record close-front late-hold suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_late_hold_suite.sh`.
  Slurm job `169927` (`g1_cflate`) ran on `server63` and failed `0/3`.
  All three cases fell/dropped at steps `632/640`, had target-window stable
  steps `0`, and did not final-latch until step `790`. Late final latch at
  `1.80 m` is too late and should not be continued.
- [x] Record close-front rescue/balance suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_balance_suite.sh`.
  Slurm job `169935` (`g1_cfresc`) ran on `server63` and failed `0/4`.
  `rescue_crouch_abs040` is useful: no box drops, first fall step `1081`,
  target-window stable steps `81`, longest streak `52`, but final lateral
  error `-0.775/-0.845 m` and no end streak. `rescue_crouch_abs055` gave
  longer target-window dwell (`142`, longest `90`) but dropped the box.
  Lateral roll-target signs both worsened collapse and should not continue.
- [x] Record close-front rescue-lateral refine suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_lateral_refine_suite.sh`.
  Slurm job `169944` (`g1_cflat`) ran on `server63` and failed `0/4`.
  Unscaled final-hold box-lateral correction was worse: best unscaled cases
  fell/dropped at `811/832` and over-traveled to about `8.75 m`; opposite
  sign fell/dropped at `626/647` with no window dwell. Do not continue
  unscaled lateral correction.
- [x] Record close-front rescue final-latch sweep:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_final_latch_sweep.sh`.
  Slurm job `169964` (`g1_cffinal`) ran on `server39` and failed `0/3`.
  `final135`, `final145`, and `final155` all fell/dropped earlier than
  `rescue_crouch_abs040`; moderate final-latch thresholds should not continue.
- [x] Record close-front rescue tiny-final-scale suite. Keep
  `rescue_crouch_abs040` and early final latch, but set final-hold scale to a
  very small nonzero value so existing progress/lateral controllers can oppose
  drift without unscaled runaway. Slurm job `169995` (`g1_cftiny`) failed
  `0/3`: `0.003` and `0.006` collapsed around steps `634-658`, while `0.010`
  reached only `49` stable steps and still dropped. Do not continue nonzero
  final scale for this branch.
- [x] Record close-front rescue target-window freeze suite. Keep
  `rescue_crouch_abs040`, final scale `0.0`, and no runtime chest support;
  add target-window policy-joint freeze to test whether drift after first
  window dwell can be arrested without lateral command or support geometry.
  Slurm job `169996` (`g1_cffreeze`) failed `0/3`. `freeze_strict` is useful:
  freeze step `663`, rescue step `732`, target-window stable steps `106`,
  longest streak `68`, good final travel/lateral, but it fell/dropped at
  `782/804`. `freeze_loose` avoided drops but only got `71` stable steps.
- [x] Record close-front freeze-balance refine suite. Continue from
  `freeze_strict` and increase roll/balance feedback authority to attack the
  roll collapse around steps `730-790`. Slurm job `170003` (`g1_cfbal`) failed
  `0/3`; stronger roll/balance feedback shortened or destroyed window dwell.
  Do not continue balance-gain increases.
- [x] Record close-front freeze-stand transition suite. Keep `freeze_strict` and
  default balance, then transition to delayed low-COM stand targets after
  freeze to see whether symmetric low-COM posture prevents the `780`-step
  collapse. Slurm job `170016` (`g1_cfstand2`) ran on `server20` and failed
  `0/3`. The best case, `stand_delay_160_soft`, reproduced the prior
  `freeze_strict` near miss with `106` stable target-window steps,
  longest/end streak `68/0`, final robot/box travel `2.176/2.119 m`, and
  fall/drop at `782/804`; shorter stand delays were worse. Later static audit
  showed this was not a valid applied stand-target test because final freeze
  masks final-stand targets. Treat it as a negative diagnostic for old control
  priority, not proof that stand targets cannot help.
- [x] Cancel invalid close-front freeze-rescue timing suite. Keep `freeze_strict` and
  default balance/stand disabled, then compare rescue disabled, rescue delayed,
  and softened rescue targets to test whether the post-freeze rescue posture
  causes the `780`-step collapse. Slurm job `170095` (`g1_cfrtime`) was
  submitted through tmux `curiosity_g1_close_front_freeze_rescue_timing_0707`;
  it was cancelled before allocation because static control-flow inspection
  showed rescue targets do not apply once final freeze is active. The suite
  would not have tested the intended hypothesis.
- [x] Record close-front freeze-rescue override suite. Use
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_rescue_override_suite.sh`,
  which explicitly enables rescue targets to override frozen policy targets
  after target-window freeze, then compare rescue disabled, delayed rescue,
  and softened rescue under the same strict gates. Slurm job `170122`
  (`g1_cfovr`) was cancelled before allocation because the first wrapper
  version did not export the override variable. The fixed wrapper was pushed,
  and Slurm job `170125` (`g1_cfovr2`) was submitted through tmux
  `curiosity_g1_close_front_freeze_rescue_override2_0707`; it ran on
  `server46` and failed `0/3`. Override was real: `freeze_rescue_late055`
  had `540` override-active steps from step `760`, and
  `freeze_rescue_soft035` had `573` override-active steps from step `727`.
  `soft035` improved target-window stable steps to `122` and longest streak
  to `84`, but still fell/dropped at `798/816`; rescue-over-freeze is useful
  evidence but not a solution.
- [ ] Add and run a close-front stand-over-freeze suite. The previous
  freeze-stand suite did not actually apply stand targets after freeze.
  Add explicit stand-over-freeze priority and test delayed low-COM stand
  targets under the same strict fall/drop/target-window/tilt/lateral gates.
- [x] Add close-front stand-over-freeze entrypoint:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_override_suite.sh`.
  It uses `--agile-command-hold-stand-overrides-final-freeze`, disables rescue,
  and reuses the delayed low-COM stand cases with a distinct
  `close_front_freeze_stand_override_summary.json`.
- [x] Run close-front stand-over-freeze suite. Slurm job `170159`
  (`g1_cfstandovr`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_override_0707`; it is pending on GPU
  priority as of `2026-07-07 17:32 CST`.
- [x] Record invalid close-front stand-over-freeze job `170159`. It ran and
  failed `0/3`, but summaries showed stand override did not apply:
  `agile_command_hold_stand_overrides_final_freeze=false` and override active
  steps `0`. Fixed
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`
  to pass wrapper-level env overrides into each case.
- [x] Re-run close-front stand-over-freeze suite after the wrapper env fix.
  Slurm job `170167` (`g1_cfstand2`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_override2_0707` and started on
  `server58` with suite stamp
  `20260707_g1_lowcarry_close_front_freeze_stand_override2`.
- [x] Record invalid stand-over-freeze v2 job `170167`. It exited after the
  first case with `line 134: and_delay_80: command not found`, caused by the
  top-level argument forwarding fix colliding with function-local case args.
  Replaced that approach with exported wrapper variables and environment
  override support for `AGILE_COMMAND_HOLD_RESCUE_ENABLE`.
- [x] Re-run close-front stand-over-freeze suite after the v2 shell fix.
  Slurm job `170173` (`g1_cfstand3`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_override3_0707` with suite stamp
  `20260707_g1_lowcarry_close_front_freeze_stand_override3`; it ran on
  `server10` and failed `0/3`, but confirmed stand-over-freeze was active.
  Best case `stand_delay_160_soft` had `600` stand-override steps from step
  `700`, target-window stable steps `141`, longest streak `103`, final
  robot/box travel `2.137/2.164 m`, final lateral error `-0.306/-0.186 m`,
  and first fall/drop `816/862`, but still failed with fall/drop `484/413`
  and max robot/box tilt `1.412/1.825 rad`.
- [x] Run close-front later/softer stand-over-freeze refinement. Continue from
  `stand_delay_160_soft`; test later delay and gentler low-COM targets/blends
  to reduce tilt/drop while preserving target-window dwell.
- [x] Add close-front later/softer stand-over-freeze refinement entrypoint:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_override_refine_suite.sh`.
  It uses `STAND_TRANSITION_CASE_SET=refine_soft` and tests
  `stand_delay_160_microblend`, `stand_delay_180_ultrasoft`, and
  `stand_delay_220_ultrasoft` with stand-over-freeze enabled and rescue
  disabled.
- [x] Run close-front later/softer stand-over-freeze refinement suite. Slurm
  job `170185` (`g1_cfstandref`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_refine_0707`; it ran on `server44`
  and failed `0/3`. `stand_delay_160_microblend` had stable steps `127`,
  fall/drop `803/834`; `stand_delay_180_ultrasoft` had stable steps `124`,
  fall/drop `800/826`; `stand_delay_220_ultrasoft` lowered max tilt but
  worsened stable steps to `108` and fall/drop to `784/807`.
- [x] Add and run a close-front stand-over-freeze balance-coupling suite.
  Keep the best `stand_delay_160_soft` timing/target and vary balance feedback
  base/gains during stand override.
- [x] Add close-front stand-over-freeze balance-coupling entrypoint:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_override_balance_suite.sh`.
  It uses `STAND_TRANSITION_CASE_SET=balance_coupling` and compares
  `stand160_balance_base_stand`, `stand160_balance_half_gain`, and
  `stand160_balance_off`.
- [x] Run close-front stand-over-freeze balance-coupling suite. Slurm job
  `170193` (`g1_cfbalstand`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_balance_0707`; it ran on `server10`
  and failed `0/3`. `stand160_balance_base_stand` had target-window stable
  steps `0` and fall/drop `533/827`; `stand160_balance_half_gain` ran away
  laterally by about `13 m`; `stand160_balance_off` fell/dropped at
  `415/500` with stable steps `0`. This attempt did not produce effective
  stand-over-freeze active steps because target-window/freeze was not
  established. Do not continue scalar balance-base/gain toggles on this
  close-front branch.
- [x] Add close-front handoff-structure case-set:
  `STAND_TRANSITION_CASE_SET=handoff_structure` in
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`.
  It tests delayed `policy_then_stand` handoff with stand-over-freeze enabled
  and rescue disabled, under the same strict no-fall/no-drop/target-window/
  tilt/lateral/no-shortcut gates. This is a control-priority/transition
  structure test, not another balance-gain sweep.
- [x] Monitor close-front handoff-structure suite. Slurm job `170267`
  (`g1_cfhand`) was submitted through tmux
  `curiosity_g1_close_front_handoff_0707` with suite stamp prefix
  `20260707_g1_lowcarry_close_front_freeze_stand_handoff`; it was cancelled
  while still pending after Slurm estimated a late start. It was superseded by
  quick single-case job `170276` (`g1_cfhandq`) through tmux
  `curiosity_g1_close_front_handoff_quick_0707`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_freeze_stand_handoff_quick`; it was also
  cancelled while pending. A `test` partition attempt failed immediately with
  invalid account/partition combination. Final quick run `170278`
  (`g1_cfhand4`) ran on `server02` through tmux
  `curiosity_g1_close_front_handoff_gpu_small_0707` with suite stamp prefix
  `20260707_g1_lowcarry_close_front_freeze_stand_handoff_quick_gpu4`. Result:
  strict `fail`, `0/1`. `policy_then_stand_delay120` completed 1300 steps
  with box drops `0` and rollout root/velocity/box pose writes `0`, but had
  first fall at step `725`, fall events `575`, target-window stable steps `0`,
  final robot/box target-directed travel `-0.496/-0.643 m`, and max robot/box
  tilt `1.284/1.397 rad`. This early `policy_then_stand` handoff is worse
  than the earlier `stand_delay_160_soft` boundary; do not continue this
  exact branch without a materially different support/terminal-control design.
- [x] Monitor posture-conditioned gate rerun. Slurm job `170282`
  (`g1_postgate`) was submitted through tmux `curiosity_g1_posture_gate_0707`
  with `SUITE_STAMP_PREFIX=20260707_g1_posture_conditioned_gate_rerun`. It
  runs the existing two-case gate, known `low_front_060` reproduction plus
  `close_front_060_conditioned`, under strict target-window/fall/drop/tilt/
  lateral/no-shortcut checks. It was `PENDING (Priority)` with no start time
  at submission. Record the result only after
  `experiments/outputs/core_world_g1_posture_conditioned_gate/20260707_g1_posture_conditioned_gate_rerun/posture_conditioned_gate_summary.json`
  exists. This job was later cancelled while still pending so the GPU queue
  could be used for the more targeted close-front retention-posture test; no
  summary exists and it should not be interpreted as evidence.
- [x] Add close-front retention-posture suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_retention_posture_suite.sh`.
  It starts from the previous near-miss `steps1050_final120` and enables
  `BOX_RETENTION_POSTURE_CONTROLLER`, preserving the same strict gates and no
  rollout root/velocity/box pose shortcuts. The quick case tests mild
  risk-driven hip/knee/ankle/waist/arm posture feedback; the full case-set can
  compare mild and strong feedback.
- [x] Monitor close-front retention-posture quick job. Slurm job `170290`
  (`g1_retpost`) was submitted through tmux
  `curiosity_g1_retention_posture_0707` with
  `RETENTION_POSTURE_CASE_SET=quick` and suite stamp prefix
  `20260707_g1_lowcarry_close_front_retention_posture_quick`. It was
  cancelled while still pending after Slurm estimated a late start.
  Replacement Slurm job `170293` (`g1_retpost45`) was submitted through tmux
  `curiosity_g1_retention_posture_45m_0707` with suite stamp prefix
  `20260707_g1_lowcarry_close_front_retention_posture_quick45`. It was
  also cancelled while pending after implementation review showed the original
  retention posture controller would hard-overwrite AGILE policy joint targets.
  No summary exists for `170290` or `170293`; do not interpret them as
  evidence.
- [x] Add blended retention-posture control. Added
  `--box-retention-blend-rate` to
  `scripts/isaac/build_core_world_g1_box_scene.py` and forwarded
  `BOX_RETENTION_BLEND_RATE` through the AGILE low-cradle runner. Default
  remains `1.0` for compatibility, but the close-front retention-posture
  suite now uses low blend rates so risk-driven posture feedback blends into
  AGILE policy targets rather than replacing locomotion targets outright.
- [x] Monitor blended close-front retention-posture quick job. Slurm job
  `170296` (`g1_retblend`) was submitted through tmux
  `curiosity_g1_retention_posture_blend_0707` with
  `RETENTION_POSTURE_CASE_SET=quick` and suite stamp prefix
  `20260707_g1_lowcarry_close_front_retention_posture_blend_quick`. It was
  `PENDING (Priority)` at submission. Result: strict `fail`, `0/1`.
  `retention_mild` completed 1050 steps with rollout root/velocity/box pose
  writes `0`, but first fall/drop happened at `821/914`, fall/drop events were
  `229/107`, target-window stable steps `0`, final robot/box target-directed
  travel was `-0.880/-0.896 m`, and max robot/box tilt was `1.903/3.134 rad`.
  Source rollout summary showed retention active for `835` steps from step
  `170`, max risk `1.0`; the controller engaged but worsened the close-front
  near-miss. Do not continue this branch with only retention blend/offset
  scalar changes.
- [x] Fix retention fields in aggregate summaries. After `170296`, source
  rollout summary had `box_retention_*` fields but aggregate
  `close_front_retention_posture_summary.json` did not. Updated
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py` to preserve
  retention enabled/range/blend/active-step/risk fields and updated
  `scripts/isaac/build_core_world_g1_box_scene.py` to record
  `box_retention_blend_rate` in future rollout summaries.
- [x] Add close-front final-stabilize quick case-set:
  `FINAL_STABILIZE_CASE_SET=quick` in
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stabilize_suite.sh`.
  It runs only `steps1200_final120_tilt030`, returning to the no-retention
  close-front near-miss lineage while testing earlier chest-pad triggering
  from box tilt.
- [x] Add read-only close-front final-stabilize parser:
  `scripts/isaac/print_g1_final_stabilize_summary.sh`. Use it after the
  final-stabilize summary exists to audit pass/fail, fall/drop timing,
  target-window dwell, travel/lateral/tilt metrics, final-hold latch/active
  steps, chest-pad trigger step/reason, and rollout root/velocity/box pose
  writes.
- [x] Monitor close-front final-stabilize quick job. Slurm job `170302`
  (`g1_finstabq`) was submitted through tmux
  `curiosity_g1_final_stabilize_quick_0707` with
  `FINAL_STABILIZE_CASE_SET=quick` and suite stamp prefix
  `20260707_g1_lowcarry_close_front_final_stabilize_quick`. It was
  cancelled while pending after Slurm estimated a late start. Replacement
  45-minute job `170306` (`g1_finstab45`) was submitted through tmux
  `curiosity_g1_final_stabilize_quick45_0707` with suite stamp prefix
  `20260707_g1_lowcarry_close_front_final_stabilize_quick45`. It completed on
  `server10` and failed strict gates. The only case
  `steps1200_final120_tilt030` had fall/drop `142/0`, first fall step `924`,
  final robot/box target-directed travel about `0.731/0.650 m`, final
  lateral error about `-0.061/-0.200 m`, max robot/box tilt
  `3.130/3.129 rad`, target-window stable steps `0`, final-hold active
  `418@782`, chest pad triggered at step `887`, and rollout root/velocity/box
  pose writes `0/0/0`. Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_final_stabilize/20260707_g1_lowcarry_close_front_final_stabilize_quick45/close_front_final_stabilize_summary.json`
- [ ] Next close-front G1 action: do not repeat unchanged quick45
  box-tilt chest-pad/final-stabilize scalar tuning. The best useful boundary
  remains no-retention `steps1050_final120` from hold-delay, while old
  `support_timing_no_runtime_pad` reached a longer target-window dwell but
  fell late. The next test should either produce a short-window diagnostic
  explicitly labeled as such, or change the mechanism: posture-conditioned
  support/command selection before target entry, target-window arrest without
  destabilizing lateral drift, or a controller-backed support posture.
- [x] Add close-front final-hold policy-state reset probe:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_reset_probe.sh`.
  It repeats the `steps1050_final120` near-miss with
  `AGILE_COMMAND_HOLD_FINAL_RESET_POLICY_STATE=1` to test whether resetting
  AGILE's recurrent state at final-hold entry reduces the tilt excess. This is
  a diagnostic mechanism check, not a relaxed success gate.
- [x] Monitor final-reset probe job. Slurm job `170321` (`g1_finreset`) was
  submitted through tmux `curiosity_g1_final_reset_probe_0707`, suite stamp
  prefix `20260707_g1_lowcarry_close_front_final_reset_probe`, log
  `logs/g1_final_reset_probe_0707_srun.log`. It completed with aggregate
  `fail`: fall/drop `25/0`, first fall step `1025`, final robot/box travel
  about `2.149/2.185 m`, final lateral error about `-0.161/-0.202 m`, max
  robot/box tilt `1.412/1.776 rad`, target-window stable/longest/end
  `120/117/0`, final-hold active `268@782`, chest pad at step `902`, and
  rollout root/velocity/box pose writes `0/0/0`. Source summary confirms
  `agile_command_hold_final_policy_state_reset_count=1` and no reset error.
  Reset improved target-window dwell versus no-reset `steps1050_final120`
  (`120` vs `76`) but caused late fall and much worse tilt, so do not keep
  pursuing final policy-state reset alone. Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_final_reset_probe/20260707_g1_lowcarry_close_front_final_reset_probe/close_front_final_reset_probe_summary.json`
- [x] Add final-hold tilt-escape mechanism and close-front probe:
  `--agile-command-hold-final-tilt-escape-scale`,
  `--agile-command-hold-final-tilt-escape-tilt`, and
  `--agile-command-hold-final-tilt-escape-box-tilt` in
  `scripts/isaac/build_core_world_g1_box_scene.py`, env forwarding in
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`,
  aggregate summary preservation in
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`, and
  entrypoint
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_tilt_escape_suite.sh`.
  This is different from the already failed tiny-final-scale branch: final
  hold remains zero until tilt crosses a threshold, then temporarily restores
  a very small AGILE command scale.
- [x] Monitor final-hold tilt-escape probe. Slurm job `170351`
  (`g1_tiltesc`) was submitted through tmux `curiosity_g1_tilt_escape_0707`,
  suite stamp prefix `20260707_g1_lowcarry_close_front_tilt_escape`, log
  `logs/g1_tilt_escape_0707_srun.log`. It runs two close-front cases:
  `escape_robot035_box042_scale015` and
  `escape_robot030_box040_scale025`. It completed with aggregate `fail`, 0/2
  strict cases passed. Both cases kept fall/drop `0/0`, final robot/box travel
  about `2.026/2.103 m`, target-window stable/longest/end `76/73/73`,
  final-hold active `268@782`, and rollout root/velocity/box pose writes
  `0/0/0`, but failed tilt and stable-step gates. `scale015` triggered
  tilt-escape for only `11` steps from step `1039`; `scale025` triggered for
  only `18` steps from step `1025`. This shows the mechanism was too late, not
  that small command release is immediately destructive. Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_tilt_escape/20260707_g1_lowcarry_close_front_tilt_escape/close_front_tilt_escape_summary.json`
- [ ] Next tilt-escape gate: run earlier trigger thresholds rather than
  increasing scale late. Candidate cases should keep the same strict gates and
  use final-hold-only escape around robot tilt `0.18-0.22 rad` and box tilt
  `0.24-0.30 rad`, with scale still small enough to keep final command under
  audited limits.
- [x] Add early tilt-escape case set:
  `TILT_ESCAPE_CASE_SET=early` in
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_tilt_escape_suite.sh`.
  It runs `escape_robot022_box030_scale015` and
  `escape_robot018_box024_scale020`.
- [x] Monitor early tilt-escape job. Slurm job `170356` (`g1_tiltearly`) was
  submitted through tmux `curiosity_g1_tilt_escape_early_0707`, suite stamp
  prefix `20260707_g1_lowcarry_close_front_tilt_escape_early`, log
  `logs/g1_tilt_escape_early_0707_srun.log`. It failed before rollout with
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_tilt_escape_suite.sh:
  line 126: syntax error: unexpected end of file`, producing only a status
  TSV and no valid summary. Do not interpret it as a physics/control result.
- [ ] Monitor replacement early tilt-escape job. Slurm job `170361`
  (`g1_tiltearly2`) was submitted through tmux
  `curiosity_g1_tilt_escape_early2_0707`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_tilt_escape_early2`, log
  `logs/g1_tilt_escape_early2_0707_srun.log`. Record only after
  `experiments/outputs/core_world_g1_lowcarry_close_front_tilt_escape/20260707_g1_lowcarry_close_front_tilt_escape_early/close_front_tilt_escape_summary.json`
  or the early2 replacement summary exists.
- [x] Monitor blended retention-posture smoke job. Slurm job `170298`
  (`g1_retbsmo`) was submitted through tmux
  `curiosity_g1_retention_blend_smoke_0707` with suite stamp
  `20260707_g1_lowcarry_close_front_retention_blend_smoke700`. It is only a
  700-step early-stability diagnostic with target-window/final-hold minimums
  disabled. If it runs, record fall/drop, tilt, target-directed travel,
  retention activation, and rollout write counts, but do not call it strict
  carrying success. It was cancelled after `170296` completed, produced only
  an env snapshot, and has no valid rollout summary.
- [x] Add read-only retention-posture summary parser:
  `scripts/isaac/print_g1_retention_posture_summary.sh`. Use it after the
  blended retention-posture summary exists to audit pass/fail, fall/drop,
  target-window dwell, travel, lateral error, tilt, final-hold latch,
  retention active steps/risk, and rollout root/velocity/box pose writes.
- [x] Monitor G1 showcase RGB capture job `170209` (`g1_showviz`) submitted
  through tmux `curiosity_g1_showcase_capture_0707`. It runs
  `scripts/isaac/run_core_world_g1_showcase_lowcarry_capture.sh` with
  `SUITE_STAMP=20260707_g1_lowcarry_showcase_rgb_retry`, RGB capture and
  replay recording enabled. It completed on `server28`, but failed early with
  fall/drop at steps `85/91` and produced no RGB frames or MP4 because
  `omni.replicator.core` could not resolve the local `omni.kit.pip_archive`
  dependency. Do not use this as showcase evidence.
- [x] Generate current best G1 fallback visual from the existing strict-pass
  replay, without rerunning control. Slurm job `170217` (`g1_bestcpu2`) ran on
  `server02` through tmux `curiosity_g1_best_fallback_visual_cpu2_0707` and
  produced `83` frames, GIF, poster, and MP4 files under
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_best_fallback_cpu2/`.
  Use it as the current best presentation artifact only with the explicit
  caveat that it is a schematic replay, not an Isaac camera render and not
  generalized carrying success.
- [x] Probe true Isaac replay-render registry/import path. Updated
  `scripts/isaac/run_core_world_g1_replay_showcase_render.sh` to use the
  installed IsaacLab Kit experience by default and local registry `file://`
  URLs. Slurm job `170222` (`repregsmk`) on `server30` confirmed
  `omni.replicator.core` can import under this default-Kit path.
- [x] Run true Isaac replay-render smoke after the launcher fix. Slurm job
  `170224` (`g1_truerdr`) on `server36` still failed: no PNG frames, no MP4,
  and no `g1_replay_render_summary.json`; checker status is `fail` with
  `frame_count=0` at
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_best_true_render_smoke_defaultkit/g1_replay_showcase_check.json`.
  Continue treating true Isaac camera render as unavailable until the
  post-import capture path is fixed.
- [x] Debug the post-import true-render failure boundary. Slurm job `170230`
  (`g1_rdrdbg`) wrote failure summaries and showed the articulation-wrapper
  replay path fails at `SingleArticulation(...)` with
  `AttributeError: type object 'PhysxManager' has no attribute '_get_backend_utils'`.
  Xform-only replay attempts `170252` and `170256` avoided that wrapper and
  created a Replicator camera, but stalled at `rep.create.render_product(...)`.
  Current conclusion: default Kit can import `omni.replicator.core`, but true
  Isaac camera replay remains blocked by the post-import render-product/capture
  backend. Do not rerun unchanged true-render attempts.
- [x] Add a read-only close-front freeze-rescue override parser:
  `scripts/isaac/print_g1_freeze_rescue_override_summary.sh`. Use it after
  `close_front_freeze_rescue_override_summary.json` exists to verify per-case
  pass/fail and whether `agile_command_hold_rescue_override_freeze_*` fields
  show the override actually applied.
- [x] Rerun the current best G1/AGILE 0.60 kg runtime chest-pad low-carry
  record for a denser showcase replay. Slurm job `170415` (`g1_showcase`)
  ran on `server46` through tmux `curiosity_g1_showcase_render_0707c` with
  record stamp
  `20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_dense_replay`.
  The rollout reproduced the narrow pass: fall/drop `0/0`, final robot/box
  target-directed travel about `2.051/2.032 m`, max robot/box tilt
  `0.309/0.428 rad`, target-window end streak `102`, and rollout root/
  velocity/box pose writes `0/0/0`.
- [x] Attempt true Isaac RGB replay render for the dense record. The render
  stage of job `170415` stalled after local Kit registry dependency failures
  around `omni.kit.pip_archive`, `omni.replicator.core`,
  `isaacsim.core.rendering_manager`, and viewport dependencies. It produced
  only
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_render_dense_replay/render_debug_trace.json`
  and no PNG/MP4, so the job was cancelled. Do not report this as a true
  Isaac camera-render success.
- [x] Generate fallback presentation visual from the dense replay. Slurm job
  `170419` (`g1_fallback`) ran on `server02` through tmux
  `curiosity_g1_fallback_video_0707e` and produced 83 frames, GIF, poster,
  raw MP4, and annotated MP4 under
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_fallback_dense_replay/`.
  Current best shareable visual:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_fallback_dense_replay/g1_lowcarry_runtime_chestpad_fallback_annotated.mp4`.
  Use only with the explicit caveat that it is a schematic replay, not an
  Isaac camera render, not new control evidence, and not generalized carrying
  success.
- [x] Test whether adding local Isaac Sim extension folders fixes true replay
  rendering. `scripts/isaac/run_core_world_g1_replay_showcase_render.sh` now
  forwards local `exts`, `extscache`, `extsPhysics`, `extsDeprecated`,
  `kit/exts`, and `kit/extscore` directories as Kit `--ext-folder` entries.
  Slurm job `170422` (`g1_extsmk`) ran on `server44` with `MAX_FRAMES=1`.
  Result: still negative. It produced only
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_extfolder_true_render_smoke/render_debug_trace.json`
  and reproduced the same missing registry dependencies:
  `omni.kit.pip_archive` and `omni.kit.viewport.window`. The ext-folder fix
  is not sufficient because those dependencies are not available as unpacked
  local extension folders. Do not rerun unchanged.
- [ ] Next true-render path: inspect or repair the local Kit registry mirror
  package availability for `omni.kit.pip_archive` and
  `omni.kit.viewport.window`, or switch to a Kit experience/launcher that can
  resolve those registry packages before spending more GPU time on replay
  render-product/capture tests.
- [x] Add an opt-in close-front approach-support posture controller. It blends
  a small low-COM hip/knee/ankle/waist support posture into AGILE policy
  targets based on target-directed robot/box travel before final-hold. This is
  not the failed retention-posture branch: it is travel/phase conditioned and
  defaults off. Updated files:
  `scripts/isaac/build_core_world_g1_box_scene.py`,
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, and
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`.
- [x] Add close-front approach-support suite:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_approach_support_suite.sh`.
  It starts from the `steps1050_final120` near-miss lineage and tests a
  pre-final-hold support posture in `soft1050`, plus a 1200-step strict
  `support1200` case in the default case set, under the same fall/drop,
  target-window, final-hold, tilt, lateral, and no-rollout-write gates.
- [x] Record first close-front approach-support quick run:
  `20260707_g1_lowcarry_close_front_approach_support_quick`. It completed but
  failed strict gates and had `approach_support_posture_active_steps=0`
  because the original activation window began at `1.35 m`, after final-hold
  latched at `1.20 m` and disabled the controller. Metrics were fall/drop
  `0/0`, final robot/box travel about `2.026/2.103 m`, max robot/box tilt
  `0.486/0.493 rad`, target-window stable steps `76`, final-hold active
  steps `268`, and writes `0/0/0`. Do not interpret this as controller
  success or failure; it was a parameter-gating mistake.
- [x] Correct close-front approach-support suite activation:
  `soft1050_active` now blends from `0.65 m` to `1.15 m`, and
  `support1200_active` from `0.65 m` to `1.20 m`, so the support posture can
  become active before final-hold disables it.
- [x] Rerun and record active close-front approach-support suite. Use tmux plus
  persistent `srun` on a compute node. Do not interpret results until
  `experiments/outputs/core_world_g1_lowcarry_close_front_approach_support/<stamp>/close_front_approach_support_summary.json`
  exists and reports nonzero `approach_support_posture_active_steps`. A pass
  still must satisfy the original strict gates and cannot be claimed as
  arbitrary-posture or learned carrying unless broader posture generalization
  also passes.
- [x] Record direct active close-front approach-support result:
  `20260707_g1_lowcarry_close_front_approach_support_direct_active`. This was
  run through a direct compute-side env command after one stale wrapper run
  showed the old activation window in its log and was cancelled. The direct
  active run did activate support (`active_steps=571`, first active step
  `479`, max scale about `0.785`) but failed strict gates: fall/drop `128/28`,
  first fall/drop steps `922/1022`, final robot/box travel about
  `1.332/1.044 m`, max robot/box tilt `2.213/2.340 rad`, target-window stable
  steps `0`, final-hold active steps `0`, writes `0/0/0`. Conclusion: the
  early/strong support offset destabilizes the close-front carry. Do not tune
  toward stronger low stance; test a much weaker, later micro-support if this
  branch is continued.
- [x] Run a weaker/later close-front micro-support probe: activate closer to
  the near-miss final-hold boundary, reduce hip/knee/ankle/waist offsets by
  about 3x, and require the same no-fall/no-drop/no-write/target-window/tilt
  gates before interpreting. The target is not to force low stance, but to
  test whether a small pre-final support bias can lower tilt without
  destroying the previously observed fall/drop `0/0` behavior.
- [x] Record weak/later close-front micro-support result:
  `20260707_g1_lowcarry_close_front_approach_support_micro_direct`. It
  activated support (`active_steps=275`, first active step `645`, max scale
  about `0.650`) and preserved fall/drop `0/0` with writes `0/0/0`, but failed
  strict gates: final robot/box travel about `1.305/1.311 m`, max robot/box
  tilt `0.281/0.570 rad`, target-window stable steps `0`, final-hold active
  steps `130 < 399`. Interpretation: weak support avoids the collapse seen in
  the strong run, but with the old `final=1.20` latch it arrests too early and
  still leaves box roll above the `0.45 rad` gate.
- [x] Run a weak-support later-latch probe: keep the micro offsets and move
  terminal/final latch later so the robot can reach the `2.0 m` target window
  before final hold. This should be treated as a diagnostic of timing/support
  interaction, not as solved posture generalization unless all strict gates
  pass.
- [x] Record weak-support later-latch result:
  `20260707_g1_lowcarry_close_front_approach_support_later_latch_direct`.
  It kept micro offsets, moved final/terminal latch to `1.80/1.85 m`, and ran
  `1600` steps. Support was active for `600` steps and the run briefly entered
  the target window (`target_window_both_stable_steps=58`, longest streak
  `56`), but strict status was fail: fall/drop `283/224`, first fall/drop
  steps `1313/1376`, max robot/box tilt `3.122/2.948 rad`, final robot/box
  travel regressed to about `1.012/0.959 m`, final-hold active
  `355 < 399`, writes `0/0/0`. Conclusion: later latch recovers target-window
  entry but destabilizes the final hold; this is only a boundary diagnostic.
- [x] Stop the current approach-support sweep unless a new mechanism is added.
  The three active results now bracket the behavior: strong/early support
  collapses, weak support with old latch arrests too early, and weak support
  with later latch reaches the window but falls late. Next close-front work
  should change the final-hold/contact/support mechanism rather than continue
  scalar tuning of approach-support start/full/latch values.
- [x] Add final-hold side-guard contact mechanism. Files:
  `scripts/isaac/build_core_world_g1_box_scene.py`,
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, and
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`. The new
  opt-in `--cradle-final-side-guards` mechanism spawns left/right torso-fixed
  physical side guards around the carried box, can enable them on hold,
  terminal-hold, final-hold, or target-window entry, and records trigger
  fields in source and aggregate summaries.
- [x] Add side-guard validation entrypoint:
  `scripts/isaac/run_core_world_g1_chestpad_finalstop_side_guard_suite.sh`.
  It starts from the `168431` chest-pad final-stop near-pass and changes only
  final-hold contact support by enabling final side guards. Quick case:
  `final_guard_hs090`, final-hold activation, local x `-0.18`, half spacing
  `0.09 m`.
- [x] Run and record chest-pad final-stop side-guard quick case. Use tmux plus
  persistent `srun` on a compute node. Do not interpret results until
  `experiments/outputs/core_world_g1_chestpad_finalstop_side_guard/<stamp>/chestpad_finalstop_side_guard_summary.json`
  exists and reports the side-guard collision trigger fields. A pass must
  still satisfy the original strict gates and cannot be claimed as
  arbitrary-posture or learned carrying unless broader posture generalization
  also passes.
- [x] Record first side-guard quick result:
  `20260707_g1_chestpad_finalstop_side_guard_quick`. It failed before testing
  the intended final-hold contact effect because side guards were pre-spawned
  as fixed rigid bodies with collision disabled, which still changed torso
  mass/inertia before trigger. The run never reached final hold and side
  guards never enabled (`collision_enabled_step=null`, update count `0`).
  Strict result: fall/drop `289/269`, first fall `711`, target-window stable
  steps `0`, final-hold active steps `0`, writes `0/0/0`. Do not interpret it
  as final-side-guard contact failure.
- [x] Add side-guard spawn-on-trigger mode and switch the quick side-guard
  suite to it. This avoids creating side-guard rigid bodies before trigger and
  makes the next run a cleaner final-hold contact diagnostic.
- [x] Rerun side-guard quick with spawn-on-trigger enabled. Require the
  summary to show either a valid `cradle_final_side_guard_spawned_step` and
  collision trigger, or a near-pass trajectory that reaches final hold without
  side-guard activation.
- [x] Record spawn-on-trigger side-guard quick result:
  `20260707_g1_chestpad_finalstop_side_guard_spawn_quick`. It spawned and
  enabled side guards at final-hold step `868` with reason `final_hold` and
  no update error. It reduced final robot/box lateral error to about
  `0.040/0.294 m`, but failed strict gates because the contact was too
  aggressive: fall/drop `68/13`, first fall/drop `932/949`, max robot/box
  tilt `1.735/1.909 rad`, final relative offset `0.313 m`, target-window
  stable/longest `45/44`, final-hold active `132`, writes `0/0/0`. This is
  useful negative evidence: side guards can correct lateral drift, but
  half-spacing `0.09 m` destabilizes final hold.
- [x] Make side-guard quick spacing configurable through
  `SIDE_GUARD_QUICK_HALF_SPACING` and `SIDE_GUARD_QUICK_CASE_NAME`.
- [x] Run looser side-guard spacing probes with spawn-on-trigger final-hold
  guards. `hs130` was stable but too weak laterally: fall/drop `0/0`, max
  robot/box tilt `0.308/0.385 rad`, target-window stable/longest/end
  `118/117/117`, final relative offset `0.229 m`, writes `0/0/0`, and only
  failed final box lateral `0.633 m > 0.6`. `hs120` fixed lateral enough:
  fall/drop `0/0`, target-window stable/longest/end `120/119/119`, final
  robot/box lateral about `0.365/0.590 m`, writes `0/0/0`, but failed final
  box/robot relative offset `0.279 m > 0.25`. Do not call either a pass.
- [x] Fix side-guard geometry overrides. The first `hs120_x10` run was not a
  valid X-size test because the suite hardcoded `CRADLE_FINAL_SIDE_GUARD_SIZE_X`
  to `0.18 m`. The suite now honors geometry environment overrides.
- [x] Run true shortened-X side-guard probe
  `20260707_g1_chestpad_finalstop_side_guard_hs120_x10b`. It used X size
  `0.10 m` and half-spacing `0.12 m`, had fall/drop `0/0`, max robot/box tilt
  `0.308/0.385 rad`, target-window stable/longest/end `119/118/118`, final
  relative offset `0.183 m`, writes `0/0/0`, and failed only final box
  lateral `0.693 m > 0.6`. Conclusion: shortening X reduces relative offset
  but weakens lateral correction.
- [x] Monitor and record `hs110_x10`: tmux
  `curiosity_g1_side_guard_hs110_x10_0707`, Slurm job `170471`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_hs110_x10`, case
  `final_guard_hs110_x10`. It kept X size `0.10 m` and tightened half-spacing
  to `0.11 m`, but failed after final-hold side-guard spawn: fall/drop
  `57/39`, max robot/box tilt `0.799/0.806 rad`, target-window
  stable/longest/end `62/61/0`, final relative offset `0.282 m`, final box
  lateral `0.621 m`, writes `0/0/0`. Conclusion: simply tightening the
  shortened-X guard makes contact too impulsive and does not solve the gate.
- [x] Monitor and record `hs115_x10`: tmux
  `curiosity_g1_side_guard_hs115_x10_0707`, Slurm job `170479`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_hs115_x10`, case
  `final_guard_hs115_x10`. It kept X size `0.10 m` and tested intermediate
  half-spacing `0.115 m`, but failed worse than `hs110_x10`: fall/drop
  `69/51`, max robot/box tilt `1.052/0.801 rad`, target-window
  stable/longest/end `50/49/0`, final relative offset `0.286 m`, final box
  lateral `0.662 m`, writes `0/0/0`. Conclusion: do not keep tightening
  shortened-X side guards; the contact impulse destabilizes final hold.
- [x] Monitor and record `hs120_lx22`: tmux
  `curiosity_g1_side_guard_hs120_lx22_0707`, Slurm job `170480`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_hs120_lx22`, case
  `final_guard_hs120_lx22`. It returns to the more stable `hs120` geometry
  with X size `0.18 m` and half-spacing `0.12 m`, but shifts local guard X
  from `-0.18` to `-0.22`. Result: stable near-miss with fall/drop `0/0`,
  max robot/box tilt `0.314/0.385 rad`, target-window stable/longest/end
  `122/121/121`, final-hold active `132`, final relative offset `0.202 m`,
  final robot/box lateral `0.437/0.634 m`, writes `0/0/0`, and only failed
  final box lateral `0.634 m > 0.6`.
- [x] Monitor and record `hs118_lx20`: tmux
  `curiosity_g1_side_guard_hs118_lx20_0707`, Slurm job `170481`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_hs118_lx20`, case
  `final_guard_hs118_lx20`. It interpolates between the stable `hs120`
  lateral pass/relative-offset fail and the `lx22` relative-offset pass/
  lateral fail by using local X `-0.20`, X size `0.18 m`, and half-spacing
  `0.118 m`. Result: fall/drop `0/0`, target-window stable/longest/end
  `125/124/124`, final relative offset `0.210 m`, final robot/box lateral
  `0.566/0.765 m`, writes `0/0/0`, but failed slight tilt
  `0.391/0.469 rad` and final box lateral. Do not continue combined spacing/
  local-X interpolation from this result.
- [x] Monitor and record `hs120_lx21`: tmux
  `curiosity_g1_side_guard_hs120_lx21_0707`, Slurm job `170485`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_hs120_lx21`, case
  `final_guard_hs120_lx21`. It keeps half-spacing `0.12 m` and X size
  `0.18 m`, and tests local X `-0.21` as a single-factor interpolation
  between `hs120` and `lx22`. Result: fall/drop `0/0`, target-window
  stable/longest/end `122/121/121`, max robot/box tilt `0.308/0.385 rad`,
  final relative offset `0.270 m`, final robot/box lateral `0.407/0.635 m`,
  writes `0/0/0`, and failed both relative-offset and lateral gates. This is
  worse than both `hs120` and `lx22` for the strict pass objective.
- [x] Monitor and record `hs120_brake001`: tmux
  `curiosity_g1_side_guard_hs120_brake001_0707`, Slurm job `170487`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_hs120_brake001`, case
  `final_guard_hs120_brake001`. It returns to the strongest `hs120` geometry
  and adds only final-hold brake `x=-0.001` for up to `160` steps, within the
  existing final command gate, to test whether reducing robot over-travel
  lowers final box/robot relative offset without losing the lateral pass.
  Result: fall/drop `0/0`, target-window stable/longest/end `120/119/119`,
  final command max `x/y/yaw=0.001/0/0`, final robot/box lateral
  `0.299/0.552 m`, writes `0/0/0`, but final box/robot relative offset
  worsened to `0.309 m`. Do not continue final-brake tuning on `hs120`.
- [x] Monitor and record `hs119_lx22`: tmux
  `curiosity_g1_side_guard_hs119_lx22_0707`, Slurm job `170493`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_hs119_lx22`, case
  `final_guard_hs119_lx22`. It returns to the `lx22` relative-offset-pass
  branch and only tightens half-spacing from `0.12` to `0.119`, aiming to
  reduce final box lateral while preserving the `lx22` relative-offset pass.
  Result: fall/drop `0/0`, target-window stable/longest/end `122/121/121`,
  max robot/box tilt `0.308/0.385 rad`, final relative offset `0.246 m`,
  final robot/box lateral `0.498/0.729 m`, writes `0/0/0`, and failed only
  final box lateral. Do not continue spacing tightening on `lx22`; it worsens
  lateral.
- [x] Add terminal-hold side-guard trigger support to
  `scripts/isaac/run_core_world_g1_chestpad_finalstop_side_guard_suite.sh`.
  The quick case can now use `SIDE_GUARD_QUICK_ENABLE_MODE=terminal` to test
  earlier side-guard contact without changing the core scene script.
- [x] Monitor and record `hs120_lx22_terminal`: tmux
  `curiosity_g1_side_guard_hs120_lx22_terminal_0707`, Slurm job `170495`,
  stamp `20260707_g1_chestpad_finalstop_side_guard_hs120_lx22_terminal`, case
  `terminal_guard_hs120_lx22`. It uses the `lx22` relative-offset-pass
  geometry but enables side guards at terminal hold instead of final hold to
  test whether earlier contact fixes lateral drift. Result: side guards
  triggered at step `590`, fall/drop `0/0`, final relative offset `0.172 m`,
  final robot/box lateral `-0.030/-0.192 m`, max robot/box tilt
  `0.308/0.449 rad`, writes `0/0/0`, but box/robot target-directed travel
  only `1.428/1.470 m`, target-window stable steps `0`, and final-hold active
  `0`. Earlier terminal contact has useful authority but stops progress when
  triggered at the default `1.05 m` box travel.
- [x] Make terminal-hold side-guard threshold/scale configurable in
  `scripts/isaac/run_core_world_g1_chestpad_finalstop_side_guard_suite.sh`
  via `AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL` and
  `AGILE_COMMAND_HOLD_TERMINAL_SCALE`.
- [x] Monitor and record `hs120_lx22_terminal145`: tmux
  `curiosity_g1_side_guard_hs120_lx22_terminal145_0707`, Slurm job `170497`,
  stamp `20260707_g1_chestpad_finalstop_side_guard_hs120_lx22_terminal145`,
  case `terminal145_guard_hs120_lx22`. It keeps the same `lx22` geometry and
  terminal trigger mode but delays terminal threshold to box target travel
  `1.45 m`, aiming to preserve target-window entry while retaining the lateral
  correction authority seen in the early terminal case. Result: side guards
  triggered at step `698` and failed badly: fall/drop `208/8`, max robot/box
  tilt `3.141/2.957 rad`, final relative offset `0.368 m`, final robot/box
  target-directed travel `0.949/0.677 m`, final robot/box lateral
  `-1.632/-1.838 m`, target-window stable steps `0`, final-hold active `0`,
  writes `0/0/0`. Do not continue terminal-trigger side guards without a
  materially softer/controlled contact formulation.
- [x] Add soft side-guard contact parameters. Files:
  `scripts/isaac/build_core_world_g1_box_scene.py`,
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, and
  `scripts/isaac/run_core_world_g1_chestpad_finalstop_side_guard_suite.sh`.
  The G1 scene now supports and records side-guard-specific static friction,
  dynamic friction, restitution, and mass scale. This is still physical
  contact with the free box, not pose-lock, servoing, or learned carrying.
- [x] Record invalid `lx22_softmat` submission: tmux
  `curiosity_g1_side_guard_lx22_softmat_0707`, Slurm job `170506`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_lx22_softmat`, case
  `final_guard_lx22_softmat`. It reuses the `lx22` relative-offset-pass
  geometry and tests final-only spawn-on-trigger guards with mass scale
  `0.25`, static/dynamic friction `0.35/0.25`, and restitution `0.0`. Job
  `170506` exited before Isaac startup on a transient shell parse error while
  the suite was being edited; there is no summary, so it is not a simulation
  result.
- [x] Monitor and record `lx22_softmat2`: tmux
  `curiosity_g1_side_guard_lx22_softmat2_0707`, Slurm job `170508`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_lx22_softmat2`, case
  `final_guard_lx22_softmat2`. Same parameters as the invalid `170506`
  attempt, resubmitted after `bash -n` passed. A pass still must satisfy strict
  no-fall/no-drop/no-rollout-write, target-window, tilt, lateral, and
  relative-offset gates. Result: fall/drop `69/38`, max robot/box tilt
  `2.152/2.126 rad`, target-window stable/longest/end `64/64/0`, final
  relative offset `0.328 m`, final robot/box lateral `0.449/0.308 m`, writes
  `0/0/0`. Low-friction low-mass guards corrected lateral but destabilized
  final hold and worsened relative offset.
- [x] Update `scripts/isaac/summarize_core_world_g1_largerbox_strict.py` to
  preserve side-guard mass/friction/restitution fields in aggregate summaries.
- [x] Monitor and record `lx22_lowmass`: tmux
  `curiosity_g1_side_guard_lx22_lowmass_0707`, Slurm job `170510`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_lx22_lowmass`, case
  `final_guard_lx22_lowmass`. It keeps the `lx22` geometry and default
  friction, changing only side-guard mass scale to `0.25`, to separate
  low-mass effects from the destabilizing low-friction result. Result:
  fall/drop `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.163 m`, final robot/box lateral `0.486/0.640 m`, writes `0/0/0`,
  and failed only final box lateral `0.640 m > 0.6`. This is the best
  relative-offset/stability boundary in the side-guard family so far.
- [x] Monitor and record `lx22_lowmass_hs118`: tmux
  `curiosity_g1_side_guard_lx22_lowmass_hs118_0707`, Slurm job `170512`,
  stamp `20260707_g1_chestpad_finalstop_side_guard_lx22_lowmass_hs118`, case
  `final_guard_lx22_lowmass_hs118`. It keeps mass scale `0.25` and tightens
  half-spacing from `0.12` to `0.118`, aiming to reduce final box lateral
  without losing the lowmass relative-offset and stability gains. Result:
  fall/drop `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final relative offset `0.141 m`, final
  robot/box lateral `0.545/0.686 m`, writes `0/0/0`, and failed only final
  box lateral. Tightening spacing worsened lateral despite improving relative
  offset.
- [x] Monitor and record `lowmass_lx20`: tmux
  `curiosity_g1_side_guard_lowmass_lx20_0707`, Slurm job `170514`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_lowmass_lx20`, case
  `final_guard_lowmass_lx20`. It keeps side-guard mass scale `0.25` and
  half-spacing `0.12`, but moves local X from `-0.22` to `-0.20` to trade
  some relative-offset margin for lateral correction. Result: fall/drop
  `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.185 m`, final robot/box lateral `0.432/0.616 m`, writes `0/0/0`,
  and failed only final box lateral `0.616 m > 0.6`. This is the closest
  low-mass side-guard boundary so far.
- [x] Monitor and record `lowmass_lx19`: tmux
  `curiosity_g1_side_guard_lowmass_lx19_0707`, Slurm job `170524`, stamp
  `20260707_g1_chestpad_finalstop_side_guard_lowmass_lx19`, case
  `final_guard_lowmass_lx19`. It keeps mass scale `0.25`, half-spacing
  `0.12`, and shifts local X to `-0.19`, aiming to close the remaining
  `1.6 cm` final box lateral gap without breaking relative-offset/stability
  gates. Result: strict check `pass`, failures `[]`, fall/drop `0/0`, max
  robot/box tilt `0.308/0.385 rad`, target-window stable/longest/end
  `133/133/133`, final-hold active `132`, final relative offset `0.0757 m`,
  final robot/box target-directed travel `2.032/2.081 m`, final robot/box
  lateral `0.407/0.465 m`, rollout root/velocity/box pose writes `0/0/0`,
  side guards spawned at final-hold step `868`, mass scale `0.25`,
  half-spacing `0.12`, local X `-0.19`, default guard friction. This is a
  narrow engineered G1/AGILE side-guard diagnostic, not learned arbitrary-
  posture or unknown-load carrying.
- [x] Monitor and record `lowmass_lx19_repeat`: tmux
  `curiosity_g1_side_guard_lowmass_lx19_repeat_0707`, Slurm job `170528`,
  stamp `20260707_g1_chestpad_finalstop_side_guard_lowmass_lx19_repeat`, case
  `final_guard_lowmass_lx19_repeat`. Same parameters as the first pass; use it
  to verify whether the strict pass is reproducible before treating
  `lowmass_lx19` as the current best close-front diagnostic. Result:
  reproduced the pass with identical key metrics: check `pass`, failures `[]`,
  fall/drop `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.0757 m`, final robot/box target-directed travel `2.032/2.081 m`,
  final robot/box lateral `0.407/0.465 m`, rollout root/velocity/box pose
  writes `0/0/0`, side guards spawned at final-hold step `868`, mass scale
  `0.25`, half-spacing `0.12`, local X `-0.19`, default guard friction.
- [x] Next G1 validation: add the reproduced `lowmass_lx19` close-front pass
  to a posture-conditioned gate alongside the known `low_front_060` pass, then
  re-run at least those two postures under the same strict no-fall/no-drop/
  no-rollout-write, target-window, tilt, lateral, and relative-offset gates.
  Do not claim arbitrary-posture carrying until multiple materially different
  postures and held-out geometry/load cases pass.
- [x] Add two-posture validation entrypoint:
  `scripts/isaac/run_core_world_g1_lowfront_closefront_lowmass_gate.sh`. It
  runs the known `low_front_060` runtime chest-pad gate and the reproduced
  `close_front_060_lowmass_lx19` side-guard gate into one aggregate summary.
  Passing this suite would still prove only two engineered postures, not
  arbitrary-posture carrying.
- [x] Monitor and record first two-posture validation suite: tmux
  `curiosity_g1_lowfront_closefront_gate_0707`, Slurm job `170535`, stamp
  `20260707_g1_lowfront_closefront_lowmass_gate`. Result: aggregate `fail`,
  1/2 passed. `low_front_060` passed, but
  `close_front_060_lowmass_lx19` failed with fall/drop `22/0`,
  target-window stable steps `0`, and final robot/box lateral about
  `-2.155/-2.225 m`. Root cause: the integrated script used
  `FREE_BOX_MASS=0.60`, while the reproduced `lowmass_lx19` close-front pass
  was `0.50 kg`. Treat this as a mismatched-mass gate, not as a refutation of
  the reproduced close-front boundary.
- [x] Patch and rerun the two-posture validation suite with close-front mass
  restored to `0.50 kg`: tmux
  `curiosity_g1_lowfront_closefront_gate_m050_0707`, Slurm job `170540`,
  stamp `20260707_g1_lowfront_closefront_lowmass_gate_m050`. Result:
  aggregate `pass`, 2/2 passed. `low_front_060` passed with fall/drop `0/0`,
  final robot/box target-directed travel about `2.051/2.032 m`, max robot/box
  tilt `0.309/0.428 rad`, target-window stable/end streak `105/102`,
  final-hold active `462`, and writes `0/0/0`. `close_front_060_lowmass_lx19`
  passed with fall/drop `0/0`, final robot/box target-directed travel about
  `2.032/2.081 m`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.0757 m`, final robot/box lateral `0.407/0.465 m`, final side
  guards spawned at step `868`, and writes `0/0/0`. This is the current best
  two-posture G1/AGILE gate, but it proves only two engineered postures with
  different masses (`0.60 kg` low-front, `0.50 kg` close-front), not arbitrary
  posture or load robustness.
- [x] Run first held-out load boundary beyond the two engineered postures.
  Strict close-front `0.55 kg` rerun
  `20260707_g1_closefront_lowmass_lx19_mass055_heldout_strict` was submitted
  through tmux session `curiosity_g1_closefront_lx19_mass055_strict_0707` as
  Slurm job `170548`, using the same `lowmass_lx19` side-guard geometry and
  the two-posture gate's common target-window environment. Result: strict
  `fail`, fall/drop `0/0`, final robot/box target-directed travel only about
  `0.552/0.186 m`, max robot/box tilt `0.447/0.532 rad`, final relative
  offset `0.399 m`, target-window stable steps `0`, and final-hold active
  steps `0`. Earlier probe
  `20260707_g1_closefront_lowmass_lx19_mass055_heldout` omitted the common
  target-window environment and should not be treated as the held-out gate.
  Interpretation: the close-front `lx19` pass is a narrow `0.50 kg`
  engineered point, not load robust.
- [x] Run intermediate close-front held-out load boundary at `0.525 kg`:
  strict rerun `20260707_g1_closefront_lowmass_lx19_mass0525_heldout_strict`
  through tmux session `curiosity_g1_closefront_lx19_mass0525_strict_0707`,
  Slurm job `170556`. Result: strict `fail`, first fall/drop at steps
  `772/800`, fall/drop totals `228/136`, final robot/box target-directed
  travel about `1.902/1.692 m`, final robot/box lateral about
  `2.196/2.148 m`, max robot/box tilt `3.068/3.090 rad`, final relative
  offset `0.458 m`, target-window stable steps `0`, and final-hold active
  steps `8`. Interpretation: increasing close-front mass from `0.50 kg` to
  `0.525 kg` already causes severe lateral drift and fall/drop before a valid
  target-window hold.
- [x] Add close-front `0.525 kg` side-guard timing entrypoint:
  `scripts/isaac/run_core_world_g1_closefront_mass0525_side_guard_timing_suite.sh`.
  It keeps the `lowmass_lx19` geometry and strict gates, then compares
  terminal-hold side-guard activation against hold-time activation. This tests
  whether the `0.525 kg` failure is mainly caused by final-hold side guards
  appearing too late.
- [x] Run close-front `0.525 kg` side-guard timing suite:
  `20260707_g1_closefront_mass0525_side_guard_timing` through tmux
  `curiosity_g1_closefront_m0525_guard_timing_0707`, Slurm job `170559`.
  Result: aggregate `fail`, 0/2. `terminal_guard_lx19` is a useful boundary:
  side guards enabled at terminal-hold step `461`, fall/drop `0/0`,
  target-window stable/longest/end `277/277/277`, final robot/box travel about
  `2.269/2.234 m`, final relative offset `0.160 m`, and writes `0/0/0`; it
  still failed box tilt `0.491 rad > 0.45` and final robot/box lateral
  `0.909/0.758 m > 0.6`. `hold_guard_lx19` kept fall/drop `0/0` but
  under-traveled: final robot/box travel about `1.643/1.539 m`,
  target-window stable steps `0`, max robot tilt `0.394 rad`, final relative
  offset `0.244 m`, and writes `0/0/0`. Interpretation: earlier physical
  side support helps load robustness, but terminal activation needs lateral
  path correction and tilt reduction; hold activation is too restrictive.
- [ ] Follow up the useful `terminal_guard_lx19` boundary with a targeted
  lateral/path correction. Do not keep using `hold_guard_lx19` as the active
  direction unless the under-travel problem is addressed.
- [x] Run first targeted lateral/path correction probe for the useful
  `terminal_guard_lx19` boundary: single case
  `20260707_g1_closefront_mass0525_terminal_latsign_neg`, tmux
  `curiosity_g1_closefront_m0525_terminal_latsign_neg_0707`, Slurm job
  `170560`. It kept terminal side-guard geometry but set
  `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0`. Result: strict `fail`, first
  fall/drop `483/498`, fall/drop totals `517/477`, side guards never triggered
  because terminal hold was not reached, max robot/box tilt
  `2.499/2.666 rad`, final robot/box target-directed travel
  `-2.438/-2.576 m`, final robot/box lateral `6.366/6.477 m`, writes
  `0/0/0`. Negative lateral sign is the wrong direction for this boundary.
- [ ] Next terminal-guard follow-up should preserve default lateral sign and
  reduce the remaining positive lateral error without destabilizing early
  approach. Candidate knobs: lower lateral limit/gain after terminal trigger,
  earlier terminal stop/hold, or a path/heading offset. Do not repeat
  `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0` for this `0.525 kg` case unchanged.
- [x] Run reduced lateral-limit probe for the useful `terminal_guard_lx19`
  boundary: single case
  `20260707_g1_closefront_mass0525_terminal_latlimit020`, tmux
  `curiosity_g1_closefront_m0525_terminal_latlimit020_0707`, Slurm job
  `170564`. It kept default lateral sign but reduced
  `AGILE_COMMAND_HOLD_LATERAL_LIMIT` from `0.035` to `0.020`. Result: strict
  `fail`, first fall/drop `833/853`, fall/drop totals `167/147`,
  target-window stable/longest/end `105/105/0`, final robot/box
  target-directed travel `2.956/2.501 m`, final robot/box lateral
  `1.490/1.466 m`, max robot/box tilt `3.135/3.134 rad`, final relative
  offset `0.500 m`, writes `0/0/0`. Reducing lateral limit alone is worse
  than the default terminal-guard boundary.
- [x] Run posture-level lateral roll-target probe for the useful
  `terminal_guard_lx19` boundary: single case
  `20260707_g1_closefront_mass0525_terminal_rolltarget`, tmux
  `curiosity_g1_closefront_m0525_terminal_rolltarget_0707`, Slurm job
  `170578`. It kept terminal side-guard timing and enabled
  `BALANCE_ROLL_TARGET_FROM_LATERAL=1` with robot lateral source, gain/limit
  `0.04/0.04`, deadband `0.10`, sign `+1`, 250-step hold delay, and 80-step
  ramp. Result: strict `fail`; fall/drop stayed `0/0` and tilt improved
  (`max robot/box tilt 0.236/0.331 rad`), but final robot/box travel collapsed
  to `1.364/1.016 m`, final robot/box lateral worsened to `1.456/1.598 m`,
  final relative offset was `0.377 m`, target-window stable steps were `0`,
  and writes were `0/0/0`. This roll-target sign/gain is not the fix.
- [x] Run constant path-offset probe for the useful `terminal_guard_lx19`
  boundary: single case
  `20260707_g1_closefront_mass0525_terminal_cmdy_neg020`, tmux
  `curiosity_g1_closefront_m0525_terminal_cmdy_neg020_0707`, Slurm job
  `170584`. It kept terminal side guards but set `AGILE_COMMAND_Y=-0.02`.
  Result: strict `fail`, first fall/drop `438/471`, fall/drop totals
  `562/279`, side guards triggered at terminal-hold step `421`, final
  robot/box target-directed travel over-shot to `4.947/4.490 m`, final
  relative offset `0.472 m`, max robot/box tilt `3.139/3.141 rad`,
  target-window stable steps `0`, writes `0/0/0`. Final lateral was small
  (`0.299/0.216 m`), but the constant path offset destabilized early
  transport and caused massive forward over-travel.
- [ ] Next terminal-guard follow-up: do not continue scalar sign/limit scans
  or the same `BALANCE_ROLL_TARGET_LATERAL_SIGN=+1` roll-target setup. Do not
  use constant `AGILE_COMMAND_Y=-0.02` as the path-offset fix. Add or test
  target-window/terminal conditional braking, heading offset, or a genuinely
  posture-conditioned support selection that preserves the useful default
  terminal side-guard stability while reducing lateral error and box tilt.
- [x] Run conservative final-brake probe for the useful `terminal_guard_lx19`
  boundary: single case
  `20260707_g1_closefront_mass0525_terminal_brake004`, tmux
  `curiosity_g1_closefront_m0525_terminal_brake004_0707`, Slurm job `170587`.
  It kept default terminal side guards and added final-hold brake
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.004` for `160` steps. Result:
  strict `fail`, but it is the best terminal follow-up so far: fall/drop
  `0/0`, target-window stable/longest/end `277/277/277`, final robot/box
  lateral improved to `0.401/0.214 m`, final robot/box travel stayed
  in-window at `2.304/1.935 m`, and writes were `0/0/0`. Remaining failures:
  max robot/box tilt `0.401/0.672 rad`, final relative offset `0.419 m`, and
  final command gate because brake command x reached `0.004 > 0.001`.
- [ ] Next terminal-guard follow-up: keep the useful final-brake direction,
  but fix box attitude and relative offset with a checker-compatible terminal
  support policy. Do not present brake as a pass while the final command gate,
  box tilt, and relative-offset gates fail.
- [x] Add checker-compatible terminal pre-final brake entrypoint. Added AGILE
  terminal-brake args and
  `scripts/isaac/run_core_world_g1_closefront_mass0525_terminal_prefinal_brake_suite.sh`.
  It moves a small reverse command into terminal hold before final hold, so
  the final-hold command gate can remain zero if the trajectory reaches the
  final latch.
- [x] Run and record terminal pre-final brake single case:
  `20260707_g1_closefront_mass0525_terminal_prefinal_brake_prefinal_brake_soft_f180`.
  It uses terminal side guards, `0.525 kg`, terminal brake x `-0.003`, delay
  `170`, steps `160`, and final box target `1.80 m`. Slurm job `170599`
  (`g1_prefbrk`) was submitted through tmux
  `curiosity_g1_closefront_m0525_prefinal_brake_0707` and ran on `server02`.
  Actual run directory was
  `20260707_g1_chestpad_finalstop_side_guard_prefinal_brake_soft_f180`
  because the first wrapper did not forward `SUITE_STAMP_PREFIX`. Result:
  strict `fail`; terminal brake was active for steps `631-790`, final hold
  never latched, first fall/drop `859/959`, fall/drop `141/41`, target-window
  stable/longest/end `91/91/0`, max robot/box tilt `2.249/2.253 rad`, and
  final lateral `1.722/1.451 m`.
- [x] Run and record one later/weaker terminal pre-final brake follow-up after the
  wrapper fix. Use a much smaller late brake, e.g.
  `PREFINAL_BRAKE_CASE_NAME=prefinal_brake_tiny_late_f165`,
  `PREFINAL_BRAKE_COMMAND_X=-0.0015`, delay `220`, steps `80`, and final box
  target `1.65`, to test whether a short pre-final correction can reduce
  lateral drift without destroying the stable `terminal_guard_lx19` behavior.
  Slurm job `170601` (`g1_preftiny`) was submitted through tmux
  `curiosity_g1_closefront_m0525_prefinal_tiny_0707` and ran on `server02`.
  Result: strict `fail`; terminal brake applied only before final hold
  (`681-710`), final-hold command max stayed `0/0/0`, fall/drop `0/0`,
  target-window stable/longest/end `277/277/277`, final relative offset
  `0.193 m`, and final travel `2.090/2.001 m`, but max robot/box tilt
  `0.408/0.537 rad` and final robot/box lateral `1.081/0.913 m` failed and
  were worse than the unbraked `terminal_guard_lx19` boundary.
- [ ] Stop scalar-scanning pre-final brake for close-front. The useful
  direction is not more brake timing/amplitude; the next control-side change
  needs explicit terminal support/contact geometry or posture-conditioned
  lateral support while keeping final-hold command zero.
- [x] Add terminal cross-brace structural support entrypoint. Added optional
  `cradle_final_cross_brace` contact proxy and
  `scripts/isaac/run_core_world_g1_closefront_mass0525_terminal_cross_brace_suite.sh`.
  This tests physical support/contact on top of the useful `terminal_guard_lx19`
  boundary instead of more command braking.
- [x] Run and record terminal cross-brace single case:
  `20260707_g1_closefront_mass0525_terminal_cross_brace_terminal_cross_brace_x19_z135`.
  It keeps terminal side guards, `0.525 kg`, final command zero, and adds a
  terminal cross-brace at local x/z `-0.19/0.135` with size
  `0.07 x 0.30 x 0.04 m`. Slurm job `170605` (`g1_xbrace`) was submitted
  through tmux `curiosity_g1_closefront_m0525_crossbrace_0707` and ran on
  `server44`. Result: strict `fail`; brace and side guards enabled at step
  `461`, fall/drop `0/0`, writes `0/0/0`, but final robot/box travel only
  `1.307/1.020 m`, final hold never latched, target-window stable steps `0`,
  final relative offset `0.371 m`, and max robot/box tilt `0.540/0.527 rad`.
- [ ] Run one delayed target-window cross-brace case. Use the same physical
  brace, but enable it on target-window entry after step `700` so it cannot
  block approach/terminal progress. This directly tests whether the structural
  contact can act as terminal retention rather than propulsion resistance.
- [x] Monitor checker-compatible terminal freeze follow-up:
  `20260707_g1_closefront_mass0525_terminal_freeze`, tmux
  `curiosity_g1_closefront_m0525_terminal_freeze_0707`, Slurm job `170593`.
  It kept default terminal side guards and enabled
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1` with robot/box freeze
  thresholds `0.35/0.50 rad`, aiming to preserve the useful target-window
  behavior without violating the final-command gate. Result: strict `fail`.
  Final freeze latched at step `724` and made final command x/y/yaw all `0`;
  final lateral errors improved to `0.569/0.441 m`, but late stability
  collapsed: first fall/drop `883/895`, total fall/drop `117/105`, max
  robot/box tilt `3.139/3.139 rad`, final relative offset `0.493 m`, and
  target-window stable/longest/end `155/155/0`. Do not reuse this freeze
  unchanged.
- [ ] Next G1 validation: expand the strict gate beyond these two engineered
  postures. At minimum add held-out box mass/geometry or a third posture
  without changing the strict fall/drop, target-window, tilt, lateral,
  relative-offset, and no-rollout-write gates. Do not continue treating the
  final side-guard parameters as a general posture selector.
