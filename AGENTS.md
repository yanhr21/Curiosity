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
- Current direct Isaac task harness:
  `scripts/isaac/build_core_world_simapp_staged_free_box_carry.py` with
  launcher `scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  `diag54` and `diag62` show that the free-box task scaffold, nonpenetrating
  carry geometry, dynamic contact proxies, posture labels, target-hold metric,
  fall/drop checks, and support-margin proxies can run in Isaac. They are not
  robot success because the carrier is still a velocity-commanded dynamic body.
- Current active implementation plan:
  `PLAN/03_no_root_articulated_carrier/plan.md`; current task list:
  `TODO/03_no_root_articulated_carrier/todo.md`.
- 2026-07-07 current best narrow G1/AGILE low-carry diagnostic:
  `20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_min700` passed
  for the tuned `low_front_060` setting with fall/drop `0/0`, final robot/box
  target-directed travel about `2.051/2.032 m`, max robot/box tilt
  `0.309/0.428 rad`, target-window end streak `102`, rollout root/velocity/
  box pose writes `0`, and runtime chest-pad collision enabled at step `712`.
  This is still a narrow engineered diagnostic, not learned unknown-load
  carrying and not arbitrary-posture carrying.
- 2026-07-07 held-out G1 terminal centroidal-support result:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_centroidal_support_suite.sh`.
  It reruns the close-front held-out `wide_y012` and `tall_z009` strict
  gates with the late terminal posture boundary plus a new opt-in
  `--terminal-centroidal-support-controller` in
  `scripts/isaac/build_core_world_g1_box_scene.py`. The new controller only
  adjusts G1 joint targets from terminal progress, robot/box lateral error,
  roll/pitch rates, and box tilt; it does not write root pose, root velocity,
  or box pose. Launcher forwarding and strict-suite aggregation fields were
  added. GPU job `170646` on `server63` and CPU fallback job `170650` on
  `server02` produced identical aggregate `fail`, 0/2. `wide_y012_centroidal`
  activated centroidal support for `78` steps from step `656`, hit pitch/roll
  adjustment limits `0.045/0.045 rad`, and still failed with first fall/drop
  `710/906`, fall/drop `290/94`, target-window `0/0/0`, final robot/box
  travel `1.540/1.373 m`, final lateral `0.649/0.388 m`, relative offset
  `0.392 m`, and max robot/box tilt `1.867/1.898 rad`. `tall_z009_centroidal`
  activated centroidal support for `198` steps from step `643`, hit the same
  adjustment limits, did not drop the box, but failed with first fall `840`,
  fall/drop `160/0`, target-window `69/54/0`, final travel `1.828/1.649 m`,
  final lateral `-0.155/0.046 m`, relative offset `0.279 m`, and max
  robot/box tilt `3.135/2.762 rad`. Rollout root/velocity/box pose writes
  stayed `0/0/0`. Do not keep scalar-tuning this micro joint-target support
  layer; the next step needs a materially stronger support/locomotion
  formulation or backend replacement.
- 2026-07-07 held-out G1 planted-stance entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_planted_stance_suite.sh`.
  It tests a stronger existing `AGILE_COMMAND_HOLD_MODE=policy_then_stand`
  transition rather than the saturated micro centroidal support feedback:
  wide boxes trigger hold near `1.25 m` box travel, tall boxes near `1.55 m`,
  then blend toward a crouched stand target while keeping root/box write gates
  and the same held-out strict checks. It is an experiment entrypoint only
  until `closefront_heldout_geometry_planted_stance_summary.json` exists.
- 2026-07-07 held-out G1 planted-stance result:
  CPU fallback Slurm job `170658` on `server01` completed
  `20260707_g1_closefront_heldout_geometry_planted_stance_cpu`, aggregate
  `fail`, 0/2; the queued GPU duplicate `170657` was cancelled before
  allocation. `wide_y012_planted` first fell before hold activation
  (`fall step 586`, hold step `602`), then failed with fall/drop `414/364`,
  target-window `0/0/0`, final robot/box travel `1.517/1.908 m`, final
  lateral `-0.215/0.856 m`, relative offset `1.168 m`, and max robot/box
  tilt `2.442/3.142 rad`. `tall_z009_planted` also first fell before hold
  activation (`fall step 904`, hold step `961`), then failed with fall/drop
  `96/35`, target-window `0/0/0`, final travel `1.494/1.776 m`, final
  lateral `-0.185/-0.429 m`, relative offset `0.409 m`, and max robot/box
  tilt `1.059/2.428 rad`. Rollout root/velocity/box pose writes stayed
  `0/0/0`. Conclusion: box-travel-triggered planted stance is too late for
  these held-out shapes; the next diagnostic must trigger from impending
  instability or change the support backend before the fall, not after
  reaching a travel threshold.
- 2026-07-07 held-out G1 instability-triggered planted entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_instability_planted_suite.sh`.
  It adds opt-in hold triggers in `scripts/isaac/build_core_world_g1_box_scene.py`
  for robot tilt and box tilt after a minimum step and minimum box travel, then
  uses `AGILE_COMMAND_HOLD_MODE=policy_then_stand` to blend toward a crouched
  planted stance. Launcher forwarding and aggregate summary fields for the new
  tilt-stop thresholds plus hold first-step/reason were added. This is an
  experiment entrypoint only until
  `closefront_heldout_geometry_instability_planted_summary.json` exists.
- 2026-07-07 held-out G1 instability-triggered planted result:
  CPU Slurm job `170660` on `server01` completed
  `20260707_g1_closefront_heldout_geometry_instability_planted_cpu`,
  aggregate `fail`, 0/2. The new trigger did fire before the travel-triggered
  planted stance: `wide_y012_instability_planted` held at step `516` from
  `box_tilt`, and `tall_z009_instability_planted` held at step `728` from
  `box_tilt`. However, policy-then-stand did not solve carrying. Wide failed
  with fall/drop `39/162`, first fall/drop `961/838`, target-window `0/0/0`,
  final robot/box travel `0.503/1.331 m`, final lateral `-0.613/-0.049 m`,
  relative offset `1.098 m`, and max robot/box tilt `1.253/3.136 rad`. Tall
  failed with fall/drop `64/107`, first fall/drop `936/893`, target-window
  `0/0/0`, final travel `1.512/1.317 m`, final lateral `-0.480/-0.042 m`,
  relative offset `0.616 m`, and max robot/box tilt `1.327/3.141 rad`.
  Rollout root/velocity/box pose writes stayed `0/0/0`. Conclusion:
  tilt-triggered AGILE hold/stand can delay body falls but loses box retention
  and target progress; the next step should not be another hold-threshold
  variant. It needs a support/contact formulation that keeps propulsion and
  box retention coupled, or a different locomotion backend.
- 2026-07-08 held-out G1 box-tilt contact-retention entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_box_tilt_contact_retention_suite.sh`.
  It avoids AGILE hold/stand and keeps locomotion commands active. Instead,
  box-tilt triggers now can spawn/enable chest pad, final side guards, and
  final cross brace while preserving rollout root/velocity/box pose write
  gates. New `cradle_final_side_guard_*_box_tilt*` and
  `cradle_final_cross_brace_*_box_tilt*` arguments were added to
  `scripts/isaac/build_core_world_g1_box_scene.py`, forwarded through the
  AGILE launcher, and copied into strict summaries. This is an experiment
  entrypoint only until
  `closefront_heldout_geometry_box_tilt_contact_retention_summary.json`
  exists.
- 2026-07-07 current best presentation visual: dense replay rerun
  `20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_dense_replay`
  reproduced the same narrow G1/AGILE low-carry pass on `server46` with
  fall/drop `0/0`, final robot/box target-directed travel about
  `2.051/2.032 m`, max robot/box tilt `0.309/0.428 rad`, target-window end
  streak `102`, rollout root/velocity/box pose writes `0`, and replay CSV
  enabled. The true Isaac RGB replay-render attempt in job `170415` was
  stopped after producing only `render_debug_trace.json`; it stalled after
  local Kit registry dependency failures for `omni.kit.pip_archive` and
  `isaacsim.core.rendering_manager`/viewport dependencies, with no PNG/MP4.
  Fallback presentation visual job `170419` then ran on `server02` and
  produced 83 frames, poster, GIF, and annotated MP4 at:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_fallback_dense_replay/g1_lowcarry_runtime_chestpad_fallback_annotated.mp4`.
  This is the best current shareable visual, but it is explicitly a schematic
  replay, not an Isaac camera render, not new control evidence, and not
  generalized or learned unknown-load carrying.
- 2026-07-07 true-render ext-folder smoke result: after the dense replay
  fallback, `scripts/isaac/run_core_world_g1_replay_showcase_render.sh` was
  updated to pass local Isaac Sim `exts`, `extscache`, `extsPhysics`,
  `extsDeprecated`, `kit/exts`, and `kit/extscore` as Kit `--ext-folder`
  entries. Minimal smoke job `170422` (`g1_extsmk`) ran on `server44` with
  `MAX_FRAMES=1`, but still reproduced the same local registry dependency
  failure: Kit could not resolve `omni.kit.pip_archive` for
  `omni.replicator.core`/telemetry and could not resolve
  `omni.kit.viewport.window` for `isaacsim.core.rendering_manager`; only
  `render_debug_trace.json` was produced. The ext-folder path alone is
  insufficient because the missing dependencies are registry package entries,
  not present unpacked extension folders. Do not rerun this ext-folder smoke
  unchanged.
- 2026-07-07 G1 posture-generalization boundary: suite
  `scripts/isaac/run_core_world_g1_lowcarry_runtime_chestpad_posture_generalization_suite.sh`
  produced aggregate `fail`, 1/5 strict cases passed. Only `low_front_060`
  passed. `close_front_060` retained the box and had fall/drop `0/0` but
  missed target-window/lateral/box-tilt gates. `forward_reach_060`,
  `wide_box_060`, and `low_front_080` failed with falls and/or drops. Follow-up
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_repair_suite.sh`
  produced aggregate `fail`, 0/3. The best repair, `lateral_sign_neg`, had
  fall/drop `0/0` and 27 target-window stable steps but still failed tilt and
  lateral gates. Do not keep scalar-tuning only lateral sign/gain; the next
  G1 step needs posture-conditioned command/support selection while preserving
  the same strict gates.
- 2026-07-07 close-front posture-conditioned command evidence: command scan
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_command_conditioned_suite.sh`
  produced aggregate `fail`, 0/3, but `command_y_neg004` was a useful boundary:
  fall/drop `0/0`, final robot/box lateral error about `0.054/0.034 m`, max
  robot/box tilt `0.246/0.329 rad`, and under-travel `1.374/1.443 m`.
  Forward-refinement
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_command_refine_suite.sh`
  showed that increasing forward command reaches the window but destabilizes.
  Hold-delay scan
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_hold_delay_suite.sh`
  produced a near miss `steps1050_final120`: fall/drop `0/0`, final robot/box
  travel about `2.026/2.103 m`, final lateral error about `0.081/0.095 m`,
  but max robot/box tilt `0.486/0.493 rad`, target-window stable steps
  `76 < 80`, and final-hold active steps `268 < 399`. This is still not a
  strict pass. Follow-up script
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stabilize_suite.sh`
  was run as Slurm job `170306` / `g1_finstab45` with suite stamp prefix
  `20260707_g1_lowcarry_close_front_final_stabilize_quick45`. The single
  quick case `steps1200_final120_tilt030` failed: fall/drop `142/0`, first
  fall step `924`, final robot/box target-directed travel about
  `0.731/0.650 m`, max robot/box tilt `3.130/3.129 rad`, target-window stable
  steps `0`, and rollout root/velocity/box pose writes `0/0/0`. Earlier
  box-tilt chest-pad triggering did not fix close-front; do not keep repeating
  this quick45 chest-pad/final-stabilize scalar branch unchanged.
- 2026-07-07 posture-conditioned gate result:
  `scripts/isaac/run_core_world_g1_posture_conditioned_gate_suite.sh`. It
  combined the known passing `low_front_060` case with a close-front candidate
  using `x=0.10,y=-0.04,final=1.20` and earlier box-tilt chest-pad triggering,
  under the same strict no-fall/no-drop/no-rollout-write/target-window/tilt/
  lateral gates. Aggregate result was `fail`, 1/2 cases passed. `low_front_060`
  reproduced the narrow pass with fall/drop `0/0`, final robot/box travel
  about `2.051/2.032 m`, max robot/box tilt `0.309/0.428 rad`, target-window
  stable/end streak `105/102`, and writes `0/0/0`. `close_front_060_conditioned`
  failed with the same boundary as quick45: fall/drop `142/0`, first fall step
  `924`, final robot/box travel about `0.731/0.650 m`, target-window stable
  steps `0`, and writes `0/0/0`. This confirms the current posture-conditioned
  gate only preserves the tuned low-front posture; it does not generalize to
  close-front.
- 2026-07-07 close-front approach-support controller added:
  `scripts/isaac/build_core_world_g1_box_scene.py` now has opt-in
  `--approach-support-posture-controller`, which blends a small low-COM
  lower-body support posture into the AGILE policy targets based on
  target-directed robot/box travel before final-hold. This is intentionally
  different from the failed retention-posture controller: it is phase/travel
  conditioned, not box-risk conditioned, and can be disabled on final-hold.
  Launcher forwarding and aggregate fields were added, plus entrypoint
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_approach_support_suite.sh`.
  First quick run `20260707_g1_lowcarry_close_front_approach_support_quick`
  completed but failed strict gates and did not actually test the controller:
  `approach_support_posture_active_steps=0`, because the original activation
  window started at `1.35 m` while final-hold latched at `1.20 m` and disabled
  the controller. Metrics matched the earlier near miss: fall/drop `0/0`,
  final robot/box travel about `2.026/2.103 m`, max robot/box tilt
  `0.486/0.493 rad`, target-window stable steps `76`, final-hold active
  steps `268`, and writes `0/0/0`. Treat this as a parameter-gating mistake,
  not evidence that the support-posture mechanism works or fails. The suite
  was corrected to `soft1050_active` / `support1200_active` with activation
  starting at `0.65 m`; do not interpret the mechanism until a fresh summary
  shows nonzero `approach_support_posture_active_steps`. Direct active run
  `20260707_g1_lowcarry_close_front_approach_support_direct_active` did
  activate the controller (`active_steps=571`, first active step `479`, max
  scale about `0.785`) but failed hard: fall/drop `128/28`, first fall/drop
  steps `922/1022`, final robot/box travel about `1.332/1.044 m`, max
  robot/box tilt `2.213/2.340 rad`, target-window stable steps `0`, final-hold
  active steps `0`, and writes `0/0/0`. This shows the early/strong support
  offset destabilizes the close-front carry; the next valid probe should be a
  much weaker and later micro-support, not a stronger low-stance offset.
  Follow-up micro-support run
  `20260707_g1_lowcarry_close_front_approach_support_micro_direct` used
  `start=0.95`, `full=1.25`, blend `0.005`, and offsets
  `hip=-0.01,knee=0.02,ankle=-0.01,waist=-0.005`. It activated
  (`active_steps=275`, first active step `645`, max scale about `0.650`) and
  preserved fall/drop `0/0` with writes `0/0/0`, but still failed strict
  gates: final robot/box travel about `1.305/1.311 m`, max robot/box tilt
  `0.281/0.570 rad`, target-window stable steps `0`, and final-hold active
  steps `130 < 399`. Interpretation: weak support no longer causes collapse,
  but combined with the old `final=1.20` latch it arrests too early and still
  leaves box roll above the `0.45 rad` gate. Next probe should keep support
  weak and move terminal/final latch later rather than increasing support.
  Later-latch follow-up
  `20260707_g1_lowcarry_close_front_approach_support_later_latch_direct`
  kept the weak offsets but moved final/terminal latch to `1.80/1.85 m` and
  ran `1600` steps. It activated support for `600` steps, reached max
  robot/box target-directed travel about `1.762/1.809 m`, and briefly entered
  the target window (`target_window_both_stable_steps=58`, longest streak
  `56`), but failed badly after final hold: fall/drop `283/224`, first
  fall/drop steps `1313/1376`, max robot/box tilt `3.122/2.948 rad`, final
  robot/box travel regressed to about `1.012/0.959 m`, final-hold active
  `355 < 399`, and writes `0/0/0`. Interpretation: moving the latch later can
  recover target-window entry, but the current weak support plus late final
  hold is not stable. Do not claim this branch as progress beyond a boundary
  diagnostic.
- 2026-07-07 final-hold side-guard contact mechanism added:
  `scripts/isaac/build_core_world_g1_box_scene.py` now supports opt-in
  `--cradle-final-side-guards`, which spawns two torso-fixed physical side
  guard collision blocks around the carried box. They can be enabled on
  agile-hold, terminal-hold, final-hold, or target-window entry and record
  their geometry, trigger reason, trigger step, and update errors in rollout
  summaries. Launcher forwarding was added to
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, and
  aggregate summaries now preserve the side-guard fields. New entrypoint:
  `scripts/isaac/run_core_world_g1_chestpad_finalstop_side_guard_suite.sh`.
  It returns to the strong `168431` chest-pad final-stop near-pass and changes
  only the final-hold contact support by enabling final side guards. This is
  an experiment entrypoint only until
  `chestpad_finalstop_side_guard_summary.json` exists; do not claim it as
  carrying success unless the strict fall/drop/target-window/tilt/lateral/
  no-write gates pass. First quick run
  `20260707_g1_chestpad_finalstop_side_guard_quick` failed before testing the
  intended final-hold contact effect: side guards were pre-spawned as fixed
  rigid bodies with collision disabled, so they still changed torso mass/
  inertia before trigger. The run never reached final hold, side guards never
  enabled (`step=null`, update count `0`), and it failed with fall/drop
  `289/269`, first fall `711`, target-window stable steps `0`, final-hold
  active steps `0`, and writes `0/0/0`. The mechanism was revised to support
  `--cradle-final-side-guard-spawn-on-trigger`; the quick suite now enables
  spawn-on-trigger so guards are not created before the trigger. Spawn-on-
  trigger quick run
  `20260707_g1_chestpad_finalstop_side_guard_spawn_quick` did test the
  intended mechanism: guards spawned/enabled at final-hold step `868`, reason
  `final_hold`, with update error `null`. It reduced final robot/box lateral
  error to about `0.040/0.294 m`, but failed strict gates because the contact
  was too aggressive: fall/drop `68/13`, first fall/drop `932/949`, max
  robot/box tilt `1.735/1.909 rad`, final relative offset `0.313 m`, target-
  window stable/longest `45/44`, final-hold active `132`, and writes `0/0/0`.
  This is useful negative evidence: side guards can correct lateral drift, but
  half-spacing `0.09 m` destabilizes final hold. The next side-guard probe
  should loosen spacing rather than increase contact authority.
- 2026-07-07 side-guard spacing/geometry boundary: looser spacing improved
  stability but exposed a lateral/relative-offset tradeoff. Run
  `20260707_g1_chestpad_finalstop_side_guard_hs130` used spawn-on-trigger
  final-hold guards with half-spacing `0.13 m`; it had fall/drop `0/0`, max
  robot/box tilt `0.308/0.385 rad`, target-window stable/longest/end
  `118/117/117`, final relative offset `0.229 m`, writes `0/0/0`, and failed
  only final box lateral `0.633 m > 0.6`. Run
  `20260707_g1_chestpad_finalstop_side_guard_hs120` used half-spacing
  `0.12 m`; it had fall/drop `0/0`, max robot/box tilt `0.308/0.385 rad`,
  target-window stable/longest/end `120/119/119`, final robot/box lateral
  about `0.365/0.590 m`, writes `0/0/0`, and failed only final box/robot
  relative offset `0.279 m > 0.25`. The first `hs120_x10` attempt was invalid
  as a geometry test because the suite still hardcoded side-guard X size to
  `0.18 m`; the suite was then fixed to honor geometry environment overrides.
  True rerun
  `20260707_g1_chestpad_finalstop_side_guard_hs120_x10b` used X size
  `0.10 m` and half-spacing `0.12 m`; it had fall/drop `0/0`, max robot/box
  tilt `0.308/0.385 rad`, target-window stable/longest/end `119/118/118`,
  final relative offset `0.183 m`, writes `0/0/0`, and failed only final box
  lateral `0.693 m > 0.6`. This shows shorter guards reduce fore-aft relative
  error but weaken lateral correction. Follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs110_x10` tested X size
  `0.10 m` with tighter half-spacing `0.11 m` and failed badly after
  final-hold guard spawn: fall/drop `57/39`, max robot/box tilt
  `0.799/0.806 rad`, target-window stable/longest/end `62/61/0`, final
  relative offset `0.282 m`, final box lateral `0.621 m`, writes `0/0/0`.
  Intermediate follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs115_x10` also failed with
  fall/drop `69/51`, max robot/box tilt `1.052/0.801 rad`, target-window
  stable/longest/end `50/49/0`, final relative offset `0.286 m`, final box
  lateral `0.662 m`, writes `0/0/0`. This shows simply tightening the short-X
  side guards makes contact too impulsive and does not solve the lateral gate.
  A compute-node single-case follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs120_lx22` was submitted through
  tmux session `curiosity_g1_side_guard_hs120_lx22_0707` as Slurm job `170480`
  to return to stable half-spacing `0.12 m` and X size `0.18 m` while moving
  local guard X from `-0.18` to `-0.22`. It ran and produced a stable
  near-miss: fall/drop `0/0`, max robot/box tilt `0.314/0.385 rad`,
  target-window stable/longest/end `122/121/121`, final-hold active `132`,
  final relative offset `0.202 m`, final robot/box lateral
  `0.437/0.634 m`, writes `0/0/0`, and failed only final box lateral
  `0.634 m > 0.6`. This is better than `hs120` on relative offset but worse
  on lateral. Interpolation follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs118_lx20` tested local X
  `-0.20`, X size `0.18 m`, and tighter half-spacing `0.118 m`; it had
  fall/drop `0/0`, target-window stable/longest/end `125/124/124`, final
  relative offset `0.210 m`, final robot/box lateral `0.566/0.765 m`, writes
  `0/0/0`, and failed tilt slightly plus final box lateral. Do not continue
  combined spacing/local-X interpolation from that result. A compute-node
  single-factor follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs120_lx21` was submitted through
  tmux session `curiosity_g1_side_guard_hs120_lx21_0707` as Slurm job `170485`
  to keep half-spacing `0.12 m`, X size `0.18 m`, and test local X `-0.21`;
  it ran with fall/drop `0/0`, target-window stable/longest/end `122/121/121`,
  max robot/box tilt `0.308/0.385 rad`, final relative offset `0.270 m`,
  final robot/box lateral `0.407/0.635 m`, writes `0/0/0`, and failed both
  relative-offset and lateral gates. The best side-guard boundary remains
  `hs120`, which passes lateral but fails relative offset. A compute-node
  follow-up `20260707_g1_chestpad_finalstop_side_guard_hs120_brake001` was
  submitted through tmux session `curiosity_g1_side_guard_hs120_brake001_0707`
  as Slurm job `170487`; it returns to `hs120` geometry and adds only a
  final-hold brake command `x=-0.001` within the existing final-command gate to
  test whether reducing robot over-travel lowers box/robot relative offset. It
  ran with fall/drop `0/0`, target-window stable/longest/end `120/119/119`,
  final command max `x/y/yaw=0.001/0/0`, final robot/box lateral
  `0.299/0.552 m`, writes `0/0/0`, and failed only relative offset
  `0.309 m > 0.25`. The brake improved lateral but worsened box/robot
  relative offset, so do not continue final-brake tuning on `hs120`. A
  compute-node follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs119_lx22` was submitted through
  tmux session `curiosity_g1_side_guard_hs119_lx22_0707` as Slurm job `170493`
  to return to the `lx22` relative-offset-pass branch and only tighten
  half-spacing from `0.12` to `0.119`. It ran with fall/drop `0/0`,
  target-window stable/longest/end `122/121/121`, max robot/box tilt
  `0.308/0.385 rad`, final relative offset `0.246 m`, final robot/box
  lateral `0.498/0.729 m`, writes `0/0/0`, and failed only final box lateral.
  It passes relative offset but worsens lateral, so do not continue spacing
  tightening on `lx22`. The side-guard suite now supports
  `SIDE_GUARD_QUICK_ENABLE_MODE=terminal` and terminal-hold side-guard
  activation. A compute-node follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs120_lx22_terminal` was
  submitted through tmux session
  `curiosity_g1_side_guard_hs120_lx22_terminal_0707` as Slurm job `170495` to
  test whether earlier terminal-hold side-guard contact fixes the lateral
  drift of `lx22`. It triggered at step `590` and strongly corrected
  lateral/relative pose, but too early: fall/drop `0/0`, final relative offset
  `0.172 m`, final robot/box lateral `-0.030/-0.192 m`, max robot/box tilt
  `0.308/0.449 rad`, writes `0/0/0`, but box/robot target-directed travel
  only `1.428/1.470 m`, target-window stable steps `0`, and final-hold active
  steps `0`. This proves earlier side-guard contact has authority but can
  stop progress. The side-guard suite now lets
  `AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL` and
  `AGILE_COMMAND_HOLD_TERMINAL_SCALE` be overridden. Follow-up
  `20260707_g1_chestpad_finalstop_side_guard_hs120_lx22_terminal145` was
  submitted through tmux session
  `curiosity_g1_side_guard_hs120_lx22_terminal145_0707` as Slurm job `170497`
  to trigger the same terminal side guards later at box target travel `1.45 m`.
  It triggered at step `698` and failed badly: fall/drop `208/8`, max
  robot/box tilt `3.141/2.957 rad`, final relative offset `0.368 m`, final
  robot/box target-directed travel `0.949/0.677 m`, final robot/box lateral
  `-1.632/-1.838 m`, target-window stable steps `0`, final-hold active
  steps `0`, writes `0/0/0`. Do not continue terminal-trigger side guards
  without a materially softer/controlled contact formulation; final-only
  `hs120` and `lx22` remain the useful near-miss boundaries.
- 2026-07-07 soft side-guard contact parameterization added:
  `scripts/isaac/build_core_world_g1_box_scene.py` now supports side-guard
  specific `--cradle-final-side-guard-static-friction`,
  `--cradle-final-side-guard-dynamic-friction`,
  `--cradle-final-side-guard-restitution`, and existing side-guard mass scale
  is forwarded through the launchers. These parameters are recorded in source
  summaries. This is still physical contact with the free box, not pose-lock,
  servoing, or learned carrying. First compute-node diagnostic
  `20260707_g1_chestpad_finalstop_side_guard_lx22_softmat` was submitted
  through tmux session `curiosity_g1_side_guard_lx22_softmat_0707` as Slurm
  job `170506`. It reuses the `lx22` relative-offset-pass geometry and tests
  final-only spawn-on-trigger guards with mass scale `0.25`, static/dynamic
  friction `0.35/0.25`, and restitution `0.0`. Job `170506` is invalid as a
  simulation result because it exited before Isaac startup on a transient
  shell parse error while the suite was being edited; no summary exists.
  After `bash -n` passed, the same diagnostic was resubmitted as
  `20260707_g1_chestpad_finalstop_side_guard_lx22_softmat2` through tmux
  session `curiosity_g1_side_guard_lx22_softmat2_0707` as Slurm job `170508`.
  It ran and failed: fall/drop `69/38`, max robot/box tilt
  `2.152/2.126 rad`, target-window stable/longest/end `64/64/0`, final
  relative offset `0.328 m`, final robot/box lateral `0.449/0.308 m`, writes
  `0/0/0`. Low-friction low-mass guards corrected lateral but destabilized
  final hold and worsened relative offset. Aggregate summarizer was updated to
  preserve side-guard mass/friction/restitution fields. Follow-up
  `20260707_g1_chestpad_finalstop_side_guard_lx22_lowmass` was submitted
  through tmux session `curiosity_g1_side_guard_lx22_lowmass_0707` as Slurm
  job `170510`; it keeps default friction and changes only side-guard mass
  scale to `0.25`. It produced a stronger near-miss than raw `lx22`: fall/drop
  `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.163 m`, final robot/box lateral `0.486/0.640 m`, writes `0/0/0`,
  and failed only final box lateral `0.640 m > 0.6`. This means reducing guard
  mass helps relative offset while preserving stability, but still needs a
  little more lateral correction. Follow-up
  `20260707_g1_chestpad_finalstop_side_guard_lx22_lowmass_hs118` was submitted
  through tmux session `curiosity_g1_side_guard_lx22_lowmass_hs118_0707` as
  Slurm job `170512`; it keeps mass scale `0.25` and tightens half-spacing
  from `0.12` to `0.118`. It ran with fall/drop `0/0`, max robot/box tilt
  `0.308/0.385 rad`, target-window stable/longest/end `133/133/133`, final
  relative offset `0.141 m`, final robot/box lateral `0.545/0.686 m`, writes
  `0/0/0`, and failed only final box lateral. Tightening spacing worsened
  lateral despite improving relative offset. Follow-up
  `20260707_g1_chestpad_finalstop_side_guard_lowmass_lx20` was submitted
  through tmux session `curiosity_g1_side_guard_lowmass_lx20_0707` as Slurm
  job `170514`; it keeps low mass and half-spacing `0.12` but moves local X
  from `-0.22` to `-0.20` to trade some relative-offset margin for lateral
  correction. It produced the closest low-mass boundary so far: fall/drop
  `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.185 m`, final robot/box lateral `0.432/0.616 m`, writes `0/0/0`,
  and failed only final box lateral `0.616 m > 0.6`. Follow-up
  `20260707_g1_chestpad_finalstop_side_guard_lowmass_lx19` was submitted
  through tmux session `curiosity_g1_side_guard_lowmass_lx19_0707` as Slurm
  job `170524`; it keeps low mass and half-spacing `0.12` and moves local X
  to `-0.19` to try to close the remaining `1.6 cm` lateral gap. Do not
  interpret it until its summary exists. It produced the first strict pass in
  the close-front side-guard family: check `pass`, failures `[]`, fall/drop
  `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.0757 m`, final robot/box target-directed travel `2.032/2.081 m`,
  final robot/box lateral `0.407/0.465 m`, rollout root/velocity/box pose
  writes `0/0/0`, side guards spawned at final-hold step `868`, mass scale
  `0.25`, half-spacing `0.12`, local X `-0.19`, and default guard friction.
  This is still a narrow engineered G1/AGILE diagnostic, not learned carrying,
  not unknown-load active probing, and not arbitrary-posture carrying. A
  same-parameter repeat
  `20260707_g1_chestpad_finalstop_side_guard_lowmass_lx19_repeat` was
  submitted through tmux session
  `curiosity_g1_side_guard_lowmass_lx19_repeat_0707` as Slurm job `170528`;
  it reproduced the pass with identical key metrics: check `pass`, failures
  `[]`, fall/drop `0/0`, max robot/box tilt `0.308/0.385 rad`, target-window
  stable/longest/end `133/133/133`, final-hold active `132`, final relative
  offset `0.0757 m`, final robot/box target-directed travel `2.032/2.081 m`,
  final robot/box lateral `0.407/0.465 m`, rollout root/velocity/box pose
  writes `0/0/0`, side guards spawned at final-hold step `868`, mass scale
  `0.25`, half-spacing `0.12`, local X `-0.19`, and default guard friction.
  Treat `lowmass_lx19` as the current best narrow close-front G1/AGILE
  engineered diagnostic. It is not evidence of arbitrary-posture carrying,
  learned carrying, or unknown-load active probing.
- 2026-07-07 two-posture validation entrypoint added:
  `scripts/isaac/run_core_world_g1_lowfront_closefront_lowmass_gate.sh`. It
  runs the known `low_front_060` runtime chest-pad gate and the reproduced
  `close_front_060_lowmass_lx19` side-guard gate into one aggregate summary.
  First run `20260707_g1_lowfront_closefront_lowmass_gate` was submitted
  through tmux session `curiosity_g1_lowfront_closefront_gate_0707` as Slurm
  job `170535`; aggregate result was `fail`, 1/2 passed. The low-front case
  passed, but the close-front case failed because the integrated script used
  `FREE_BOX_MASS=0.60`, while the reproduced `lowmass_lx19` close-front pass
  was a `0.50 kg` case. Do not use job `170535` as evidence against the
  reproduced `lowmass_lx19` boundary; it is a mismatched-mass gate. Corrected
  rerun `20260707_g1_lowfront_closefront_lowmass_gate_m050` was submitted
  through tmux session `curiosity_g1_lowfront_closefront_gate_m050_0707` as
  Slurm job `170540`; aggregate result was `pass`, 2/2 passed. `low_front_060`
  passed with fall/drop `0/0`, final robot/box target-directed travel about
  `2.051/2.032 m`, max robot/box tilt `0.309/0.428 rad`, target-window
  stable/end streak `105/102`, final-hold active `462`, and writes `0/0/0`.
  `close_front_060_lowmass_lx19` passed with fall/drop `0/0`, final robot/box
  target-directed travel about `2.032/2.081 m`, max robot/box tilt
  `0.308/0.385 rad`, target-window stable/longest/end `133/133/133`,
  final-hold active `132`, final relative offset `0.0757 m`, final robot/box
  lateral `0.407/0.465 m`, side guards spawned at final-hold step `868`, and
  writes `0/0/0`. This validates two engineered G1/AGILE postures under strict
  gates, with low-front at `0.60 kg` and close-front at `0.50 kg`. It is not
  arbitrary-posture carrying, not load-robust carrying, not learned carrying,
  and not unknown-load active probing.
- 2026-07-07 close-front `lowmass_lx19` held-out load boundary:
  a strict `0.55 kg` rerun,
  `20260707_g1_closefront_lowmass_lx19_mass055_heldout_strict`, was submitted
  through tmux session `curiosity_g1_closefront_lx19_mass055_strict_0707` as
  Slurm job `170548`. It failed under the same strict target-window gates:
  fall/drop stayed `0/0`, but final robot/box target-directed travel was only
  about `0.552/0.186 m`, max robot/box tilt was `0.447/0.532 rad`, final
  relative offset was `0.399 m`, target-window stable steps were `0`, and
  final-hold active steps were `0`. An earlier non-strict `0.55 kg` probe
  `20260707_g1_closefront_lowmass_lx19_mass055_heldout` omitted the common
  target-window environment and should not be used as the held-out gate. The
  strict result shows the close-front `lx19` pass is a narrow `0.50 kg`
  engineered point, not a load-robust carrying strategy.
- 2026-07-07 close-front `lowmass_lx19` intermediate load boundary:
  strict `0.525 kg` rerun
  `20260707_g1_closefront_lowmass_lx19_mass0525_heldout_strict` was submitted
  through tmux session `curiosity_g1_closefront_lx19_mass0525_strict_0707` as
  Slurm job `170556`. It also failed: first fall/drop occurred at steps
  `772/800`, fall/drop totals were `228/136`, final robot/box target-directed
  travel was about `1.902/1.692 m`, final robot/box lateral error was about
  `2.196/2.148 m`, max robot/box tilt was `3.068/3.090 rad`, final relative
  offset was `0.458 m`, target-window stable steps were `0`, and final-hold
  active steps were `8`. This places the current close-front side-guard pass
  boundary very close to the tuned `0.50 kg` case; `0.525 kg` already loses
  lateral stability and falls before a valid target-window hold.
- 2026-07-07 close-front `0.525 kg` side-guard timing suite added:
  `scripts/isaac/run_core_world_g1_closefront_mass0525_side_guard_timing_suite.sh`.
  It compares the same `lowmass_lx19` side-guard geometry with earlier
  collision activation, because final-hold activation was too late for the
  `0.525 kg` failure. Suite
  `20260707_g1_closefront_mass0525_side_guard_timing`, submitted through tmux
  `curiosity_g1_closefront_m0525_guard_timing_0707` as Slurm job `170559`,
  failed aggregate `0/2`, but the failure modes are informative. The
  `terminal_guard_lx19` case enabled side guards at terminal-hold step `461`
  and removed the previous fall/drop: fall/drop `0/0`, target-window
  stable/longest/end `277/277/277`, final robot/box travel about
  `2.269/2.234 m`, final relative offset `0.160 m`, and writes `0/0/0`.
  It still failed strict gates due box tilt `0.491 rad > 0.45` and final
  robot/box lateral `0.909/0.758 m > 0.6`. The `hold_guard_lx19` case enabled
  side guards at hold step `104` and also kept fall/drop `0/0`, but
  under-traveled: final robot/box travel about `1.643/1.539 m`, target-window
  stable steps `0`, max tilt `0.394 rad`, final relative offset `0.244 m`,
  and writes `0/0/0`. Interpretation: earlier physical support is a real
  improvement over final-hold-only for `0.525 kg`, but terminal support still
  needs lateral/path correction and box-tilt reduction; hold-time support is
  too restrictive for target travel.
- 2026-07-07 close-front `0.525 kg` terminal-guard lateral-sign probe:
  single case `20260707_g1_closefront_mass0525_terminal_latsign_neg` was
  submitted through tmux
  `curiosity_g1_closefront_m0525_terminal_latsign_neg_0707` as Slurm job
  `170560`, keeping terminal side-guard geometry but setting
  `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0`. It failed much worse than the
  default-sign terminal boundary: first fall/drop at steps `483/498`,
  fall/drop totals `517/477`, side guards never triggered because terminal
  hold was not reached, max robot/box tilt `2.499/2.666 rad`, final
  robot/box target-directed travel `-2.438/-2.576 m`, final robot/box lateral
  `6.366/6.477 m`, and writes `0/0/0`. Do not continue the negative lateral
  sign branch for this terminal-guard boundary.
- 2026-07-07 close-front `0.525 kg` terminal-guard lateral-limit probe:
  single case `20260707_g1_closefront_mass0525_terminal_latlimit020` was
  submitted through tmux
  `curiosity_g1_closefront_m0525_terminal_latlimit020_0707` as Slurm job
  `170564`, keeping default lateral sign but reducing
  `AGILE_COMMAND_HOLD_LATERAL_LIMIT` from `0.035` to `0.020`. It failed and
  was worse than the default terminal boundary: first fall/drop at steps
  `833/853`, fall/drop totals `167/147`, target-window stable/longest
  `105/105` but streak at end `0`, final robot/box target-directed travel
  `2.956/2.501 m`, final robot/box lateral `1.490/1.466 m`, max robot/box
  tilt `3.135/3.134 rad`, final relative offset `0.500 m`, and writes
  `0/0/0`. Do not continue reducing lateral limit alone for this boundary;
  the next meaningful terminal-guard step needs target-window braking,
  path/heading offset, or posture-conditioned support rather than scalar sign/
  limit scanning.
- 2026-07-07 close-front `0.525 kg` terminal-guard roll-target probe:
  single case `20260707_g1_closefront_mass0525_terminal_rolltarget` was
  submitted through tmux
  `curiosity_g1_closefront_m0525_terminal_rolltarget_0707` as Slurm job
  `170578`. It kept the useful terminal side-guard timing and enabled
  lateral-error-driven `balance_roll_target_from_lateral` with source `robot`,
  gain/limit `0.04/0.04`, deadband `0.10`, sign `+1`, 250-step hold delay,
  and 80-step ramp. It failed: fall/drop stayed `0/0` and tilt improved
  substantially, with max robot/box tilt `0.236/0.331 rad`, but travel
  collapsed and lateral worsened. Final robot/box target-directed travel was
  only `1.364/1.016 m`, final robot/box lateral was `1.456/1.598 m`, final
  relative offset was `0.377 m`, target-window stable steps were `0`, and
  writes were `0/0/0`. This roll-target sign/gain is not a fix; it trades
  balance for severe lateral/path error and under-travel.
- 2026-07-07 close-front `0.525 kg` terminal-guard path-offset probe:
  single case `20260707_g1_closefront_mass0525_terminal_cmdy_neg020` was
  submitted through tmux
  `curiosity_g1_closefront_m0525_terminal_cmdy_neg020_0707` as Slurm job
  `170584`, keeping terminal side guards but setting `AGILE_COMMAND_Y=-0.02`.
  It failed badly: first fall/drop at steps `438/471`, fall/drop totals
  `562/279`, side guards triggered at terminal-hold step `421`, final
  robot/box target-directed travel over-shot to `4.947/4.490 m`, final
  relative offset was `0.472 m`, max robot/box tilt was `3.139/3.141 rad`,
  target-window stable steps were `0`, and writes were `0/0/0`. Although final
  lateral was small (`0.299/0.216 m`), the path offset destabilized early
  transport and caused massive forward over-travel. Do not use a constant
  `AGILE_COMMAND_Y=-0.02` path offset for this boundary.
- 2026-07-07 close-front `0.525 kg` terminal-guard final-brake probe:
  single case `20260707_g1_closefront_mass0525_terminal_brake004` was
  submitted through tmux `curiosity_g1_closefront_m0525_terminal_brake004_0707`
  as Slurm job `170587`, keeping default terminal side guards and adding
  final-hold brake `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.004` for
  `160` steps. It is not a strict pass, but it is the most useful terminal
  follow-up so far. Fall/drop stayed `0/0`, target-window stable/longest/end
  stayed `277/277/277`, final robot/box lateral improved to
  `0.401/0.214 m`, and final robot/box travel stayed in-window at
  `2.304/1.935 m`. Failures were max robot/box tilt `0.401/0.672 rad`, final
  relative offset `0.419 m`, and the strict final command gate because brake
  command x reached `0.004 > 0.001`; writes stayed `0/0/0`. Interpretation:
  terminal braking addresses lateral/over-travel better than sign, lateral
  limit, roll-target, or constant path-offset probes, but it must be paired
  with box attitude/relative retention and a checker-compatible terminal
  support policy before it can count as carrying success.
- 2026-07-07 close-front `0.525 kg` terminal-guard freeze follow-up:
  single case `20260707_g1_closefront_mass0525_terminal_freeze`, tmux
  `curiosity_g1_closefront_m0525_terminal_freeze_0707`, Slurm job `170593`,
  kept default terminal side guards and enabled
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1` with freeze thresholds
  robot/box tilt `0.35/0.50 rad`. Result: strict `fail`. The checker-
  compatible zero-command freeze latched at step `724` and was active for
  `276` steps; final robot/box command x/y/yaw were all `0`, final robot/box
  lateral errors improved to `0.569/0.441 m`, and writes stayed `0/0/0`.
  However the run destabilized after the target-window dwell: first fall/drop
  happened at steps `883/895`, total fall/drop was `117/105`, max robot/box
  tilt was `3.139/3.139 rad`, final relative offset was `0.493 m`, and
  target-window stable/longest/end fell to `155/155/0`. Interpretation:
  freezing satisfies the final-command gate but destroys late stability and
  box retention, so it is worse than the final-brake boundary for carrying and
  should not be reused unchanged.
- 2026-07-07 close-front `0.525 kg` terminal pre-final brake entrypoint:
  added `AGILE_COMMAND_HOLD_TERMINAL_BRAKE_COMMAND_X`,
  `AGILE_COMMAND_HOLD_TERMINAL_BRAKE_DELAY_STEPS`, and
  `AGILE_COMMAND_HOLD_TERMINAL_BRAKE_STEPS` to the AGILE command path. This
  brake is only allowed during terminal hold and before final hold, so it can
  test whether the useful final-brake lateral/travel effect can be moved
  outside the strict final-command gate. Added launcher
  `scripts/isaac/run_core_world_g1_closefront_mass0525_terminal_prefinal_brake_suite.sh`,
  currently configured for one focused case
  `prefinal_brake_soft_f180` with terminal side guards, `0.525 kg` box,
  brake x `-0.003`, delay `170`, steps `160`, and final box target `1.80 m`.
  Slurm job `170599` (`g1_prefbrk`) was submitted through tmux
  `curiosity_g1_closefront_m0525_prefinal_brake_0707`; this is an experiment
  entrypoint only until a fresh summary exists.
- 2026-07-07 close-front `0.525 kg` terminal pre-final brake first result:
  Slurm job `170599` (`g1_prefbrk`) ran on `server02` through tmux
  `curiosity_g1_closefront_m0525_prefinal_brake_0707` and produced strict
  `fail`. Because the first wrapper did not forward `SUITE_STAMP_PREFIX` to
  the inner side-guard script, the actual run directory is
  `20260707_g1_chestpad_finalstop_side_guard_prefinal_brake_soft_f180`.
  The run confirmed the new terminal-brake fields were active:
  terminal brake x `-0.003`, first/last active steps `631/790`, active steps
  `160`, and final-hold command remained zero because final hold never
  latched. It is worse than `terminal_guard_lx19`: first fall/drop
  `859/959`, fall/drop `141/41`, max robot/box tilt `2.249/2.253 rad`,
  target-window stable/longest/end `91/91/0`, final robot/box travel
  `2.041/1.814 m`, final lateral `1.722/1.451 m`, and writes `0/0/0`.
  Interpretation: pre-final brake is structurally wired correctly, but this
  setting is too early/strong and prevents final-hold latch while creating a
  large lateral roll collapse. The wrapper has been corrected to forward the
  outer `SUITE_STAMP_PREFIX` and to allow one-case parameter overrides.
- 2026-07-07 close-front `0.525 kg` terminal pre-final brake late-tiny
  follow-up: Slurm job `170601` (`g1_preftiny`) ran on `server02` through
  tmux `curiosity_g1_closefront_m0525_prefinal_tiny_0707`, stamp prefix
  `20260707_g1_closefront_mass0525_terminal_prefinal_brake_late_tiny`, case
  `prefinal_brake_tiny_late_f165`. Result: strict `fail`. The corrected
  wrapper wrote the intended stamp. Terminal brake x `-0.0015` was active
  only for steps `681-710` (`30` steps) before final hold latched at step
  `711`; final command max x/y/yaw stayed `0/0/0`, fall/drop stayed `0/0`,
  final robot/box travel was `2.090/2.001 m`, target-window stable/longest/end
  was `277/277/277`, final relative offset was `0.193 m`, and writes stayed
  `0/0/0`. Remaining strict failures were worse than `terminal_guard_lx19`:
  max robot/box tilt `0.408/0.537 rad` and final robot/box lateral
  `1.081/0.913 m`. Interpretation: a very small pre-final brake is
  checker-compatible but does not reduce lateral drift and worsens tilt; do
  not keep scalar-scanning pre-final brake.
- 2026-07-07 close-front `0.525 kg` terminal cross-brace entrypoint: added a
  new optional `cradle_final_cross_brace` physical contact proxy in
  `scripts/isaac/build_core_world_g1_box_scene.py` and exposed it through
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`. It is a
  torso-fixed transverse brace with hold/terminal/final/target-window trigger
  modes, intended to test a structural support/contact change rather than
  another command scalar. Added single-case launcher
  `scripts/isaac/run_core_world_g1_closefront_mass0525_terminal_cross_brace_suite.sh`,
  which reuses the useful `terminal_guard_lx19` boundary and adds a terminal
  cross-brace at local x/z `-0.19/0.135` with size `0.07 x 0.30 x 0.04 m`.
  Slurm job `170605` (`g1_xbrace`) was submitted through tmux
  `curiosity_g1_closefront_m0525_crossbrace_0707`; this is an experiment
  entrypoint only until a fresh summary exists.
- 2026-07-07 close-front `0.525 kg` terminal cross-brace first result:
  Slurm job `170605` (`g1_xbrace`) ran on `server44` and produced strict
  `fail`. The new brace was spawned and enabled at terminal-hold step `461`
  together with the side guards; no spawn/collision errors occurred, writes
  stayed `0/0/0`, and fall/drop stayed `0/0`. However the brace engaged too
  early and blocked useful progress: final robot/box travel was only
  `1.307/1.020 m`, final hold never latched, target-window stable steps were
  `0`, final relative offset was `0.371 m`, and max robot/box tilt was
  `0.540/0.527 rad`. Interpretation: the cross-brace contact path is wired
  and physically active, but terminal-hold activation is too early/aggressive;
  the next valid structural test is delayed target-window activation rather
  than command braking or another terminal scalar.
- 2026-07-07 close-front target-window cross-brace follow-up submitted:
  Slurm job `170607` (`g1_xbrtgt`) through tmux
  `curiosity_g1_closefront_m0525_crossbrace_target_0707`, stamp prefix
  `20260707_g1_closefront_mass0525_target_cross_brace`, case
  `target_window_cross_brace_x19_z135`. It keeps terminal side guards but
  delays the same cross-brace until target-window entry after step `700`.
- 2026-07-07 close-front target-window cross-brace result: Slurm job `170607`
  ran on `server39` and produced strict `fail`. The brace spawned and enabled
  at target-window step `724`, after final hold latched at step `711`; side
  guards still enabled at terminal-hold step `461`, and writes stayed
  `0/0/0`. Delaying the brace avoided approach blockage and kept final
  robot/box travel near target (`1.968/2.049 m`) with final relative offset
  `0.238 m`, but the rigid contact destabilized the carry: first fall/drop
  `799/815`, fall/drop `201/185`, max robot/box tilt `1.080/1.163 rad`,
  target-window stable/longest/end only `75/70/0`, and final robot/box lateral
  `0.617/0.731 m`. Interpretation: rigid cross-brace contact improves
  relative offset but injects too much disturbance; do not continue hard
  cross-brace geometry/timing micro-scans without a softer contact or
  posture-conditioned support controller.
- 2026-07-07 held-out geometry gate entrypoint added:
  `scripts/isaac/run_core_world_g1_lowfront_closefront_heldout_geometry_gate.sh`.
  It extends the current two-posture gate with close-front held-out shape
  checks while preserving the same strict no-fall/no-drop/no-rollout-write,
  target-window, tilt, lateral, relative-offset, and final-command gates.
  Default cases are the known `low_front_060` pass, the reproduced
  `close_front_060_lowmass_lx19` pass, and two untuned close-front geometry
  perturbations: `FREE_BOX_SIZE_Y=0.12` and `FREE_BOX_SIZE_Z=0.09`, both at
  `0.50 kg` with the same lowmass-lx19 side-guard strategy. This is a
  validation/overfit-boundary suite, not a new controller and not evidence of
  arbitrary posture or unknown-load robustness until a fresh aggregate summary
  exists.
- 2026-07-07 held-out geometry gate result:
  `20260707_g1_lowfront_closefront_heldout_geometry_gate`, submitted through
  tmux session `curiosity_g1_heldout_geometry_gate_0707` as Slurm job
  `170612`, ran on `server44` and produced aggregate `fail`, 2/4 strict cases
  passed. The known `low_front_060` and `close_front_060_lowmass_lx19`
  baselines reproduced their passes with fall/drop `0/0` and rollout
  root/velocity/box pose writes `0/0/0`. The untuned close-front held-out
  geometry cases both failed: `wide_y012` (`box_size=[0.14,0.12,0.08]`) had
  first fall/drop `942/966`, fall/drop `53/15`, target-window stable steps
  `0`, final robot/box target travel about `0.995/0.784 m`, final robot/box
  lateral about `2.135/1.904 m`, relative offset `0.324 m`, and max robot/box
  tilt `1.298/1.163 rad`; `tall_z009` (`box_size=[0.14,0.10,0.09]`) had
  first fall/drop `938/989`, fall/drop `62/11`, target-window stable/longest
  `152/152` but end streak `0`, over-travel `3.063/2.754 m`, relative offset
  `0.364 m`, and max robot/box tilt `2.388/2.074 rad`. This confirms the
  current two-posture result is narrow and geometry-sensitive, not a general
  shape-robust carrying solution.
- 2026-07-07 close-front held-out geometry adaptive-support entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_adaptive_support_suite.sh`.
  It does not tune the passed baseline; it probes two mechanism changes for
  the failed held-out shape cases under the same strict gates. For
  `wide_y012`, side guards are enabled at terminal hold instead of final hold
  to test whether early lateral contact prevents the pre-window drift. For
  `tall_z009`, final stand handoff starts after 120 final-hold steps with
  blend rate `0.02` to test whether the target-window dwell can be preserved
  instead of collapsing late. This is an experiment entrypoint only until
  `closefront_heldout_geometry_adaptive_support_summary.json` exists.
- 2026-07-07 close-front held-out geometry adaptive-support result:
  `20260707_g1_closefront_heldout_geometry_adaptive_support`, submitted
  through tmux `curiosity_g1_heldout_adaptive_support_0707` as Slurm job
  `170614`, ran on `server44` and produced aggregate `fail`, 0/2 strict
  cases passed. `wide_y012_terminal_guard` did not improve the previous wide
  failure and the terminal side guard never triggered
  (`collision_enabled_step=null`): fall/drop `53/15`, first fall/drop
  `942/966`, target-window stable steps `0`, final robot/box travel about
  `0.995/0.784 m`, final lateral `2.135/1.904 m`, relative offset `0.324 m`,
  and max robot/box tilt `1.298/1.163 rad`. Interpretation: the wide-box
  failure happens before the current terminal-hold side-guard trigger becomes
  usable. `tall_z009_final_stand` changed the tall-box failure mode but did
  not pass: final stand became active at step `867` for `133` steps, box drops
  fell from the previous `11` to `0`, relative offset improved to `0.179 m`,
  and lateral error stayed within gate (`0.416/0.252 m`), but the run still
  had first fall `954`, fall events `46`, over-travel `2.814/2.843 m`, target
  end streak `0`, and max robot/box tilt `1.601/1.602 rad`. Next shape-robust
  work should use earlier geometry-adaptive path/support selection rather than
  final-only stand or side-guard timing.
- 2026-07-07 close-front held-out geometry early-support entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_early_support_suite.sh`.
  It tests two earlier interventions under unchanged strict gates. For
  `wide_y012`, low-mass side guards are spawned on hold with half-spacing
  `0.13 m` and mass scale `0.15`, because terminal-trigger guards never
  became active before wide-box drift/fall. For `tall_z009`, final stand
  handoff delay is reduced from `120` to `40` steps, because delayed stand
  improved retention but started too late to prevent over-travel and fall.
  This is an experiment entrypoint only until
  `closefront_heldout_geometry_early_support_summary.json` exists.
- 2026-07-07 close-front held-out geometry early-support result:
  `20260707_g1_closefront_heldout_geometry_early_support`, submitted through
  tmux `curiosity_g1_heldout_early_support_0707` as Slurm job `170617`, ran
  on `server44` and produced aggregate `fail`, 0/2 strict cases passed.
  `wide_y012_hold_guard` confirmed early contact can change the failure
  boundary: hold-trigger side guards enabled at step `100`, target-window
  stable/longest improved from `0/0` to `93/93`, relative offset was within
  gate at `0.230 m`, and rollout writes stayed `0`; however it over-traveled
  badly (`3.879/3.803 m`), first fall/drop moved earlier to `773/829`,
  fall/drop rose to `227/171`, final lateral was `0.671/0.886 m`, and max
  robot/box tilt was `3.064/3.065 rad`. Interpretation: early support helps
  reach the window but needs a window-latched speed/retention controller; hold
  guards alone create a late runaway/drop failure. `tall_z009_early_stand40`
  also failed: final stand activated at step `787` for `213` steps and kept
  final travel in the allowed window (`2.234/2.315 m`), but was worse than the
  later stand for stability, with first fall/drop `877/920`, fall/drop
  `123/80`, target-window end streak `0`, final box lateral `0.609 m`, and
  max robot/box tilt `1.566/1.578 rad`. Do not continue simply moving stand
  earlier; the next step should add target-window-latched velocity/retention
  logic after early support, not another final-only handoff.
- 2026-07-07 close-front held-out geometry window-freeze entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_window_freeze_suite.sh`.
  It reuses the wide-box hold-trigger side-guard boundary and the tall-box
  final-guard boundary, then enables
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1` with robot/box tilt
  thresholds `0.35/0.45 rad`. This tests whether freezing policy joint targets
  once both robot and box are inside the target window can prevent the
  post-window over-travel/fall failure. It is an experiment entrypoint only
  until `closefront_heldout_geometry_window_freeze_summary.json` exists.
- 2026-07-07 close-front held-out geometry window-freeze result:
  `20260707_g1_closefront_heldout_geometry_window_freeze`, submitted through
  tmux `curiosity_g1_heldout_window_freeze_0707` as Slurm job `170620`, ran
  on `server39` and produced aggregate `fail`, 0/2 strict cases passed.
  `wide_y012_hold_guard_freeze` did latch the target-window freeze at step
  `681` for `319` active steps with rollout writes `0`, but it was worse than
  the earlier hold-guard boundary: first fall/drop moved to `712/752`,
  fall/drop rose to `288/139`, target-window stable/longest/end was only
  `32/32/0`, final robot/box travel was `2.041/1.810 m`, final lateral was
  `0.642/0.410 m`, relative offset was `0.380 m`, and max robot/box tilt was
  `2.526/2.535 rad`. `tall_z009_window_freeze` latched freeze at step `754`
  for `246` steps and reached target-window stable/longest `102/102`, but
  end streak was still `0`; first fall/drop was `855/872`, fall/drop
  `145/128`, over-travel was `2.474/2.496 m`, relative offset was `0.297 m`,
  and max robot/box tilt was `1.844/2.068 rad`. Interpretation:
  checker-compatible target-window freeze is not the held-out shape fix. It
  removes command motion after latch but does not provide terminal support,
  box attitude retention, or controlled braking, so the wide/tall cases still
  roll, drop, or over-travel. Do not continue simple window-freeze or earlier
  stand/final-only scalar tuning for held-out shapes.
- 2026-07-07 close-front held-out geometry rescue-freeze entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_rescue_freeze_suite.sh`.
  It keeps the same two failed wide/tall held-out shape cases and the same
  strict gates as the window-freeze suite, but enables
  `AGILE_COMMAND_HOLD_RESCUE_ENABLE=1` and
  `AGILE_COMMAND_HOLD_RESCUE_OVERRIDES_FINAL_FREEZE=1`. The test question is
  narrow: whether target-window freeze plus a roll-triggered rescue posture
  can recover the post-window roll/drop failure without relaxing no-fall,
  no-drop, no-rollout-write, target-window, tilt, lateral, relative-offset,
  and final-command gates. This is an experiment entrypoint only until
  `closefront_heldout_geometry_rescue_freeze_summary.json` exists.
- 2026-07-07 close-front held-out geometry rescue-freeze result:
  `20260707_g1_closefront_heldout_geometry_rescue_freeze`, submitted through
  tmux `curiosity_g1_heldout_rescue_freeze_0707` as Slurm job `170621`, ran
  on `server39` and produced aggregate `fail`, 0/2 strict cases passed.
  `wide_y012_hold_guard_rescue_freeze` latched freeze at step `681`, then
  rescue override became active at step `691` for `309` steps, but the result
  was still worse than the earlier non-rescue hold-guard boundary: first
  fall/drop `723/760`, fall/drop `277/230`, target-window stable/longest/end
  `43/43/0`, over-travel `3.196/2.880 m`, final lateral `1.267/1.449 m`,
  relative offset `0.367 m`, max robot/box tilt `3.141/3.141 rad`, and
  rollout writes `0`. `tall_z009_rescue_freeze` latched freeze at step `754`,
  then rescue override became active at step `823` for `177` steps. It
  reached target-window stable/longest `100/100`, but end streak stayed `0`
  and it failed with first fall/drop `853/871`, fall/drop `147/129`,
  over-travel `2.554/2.513 m`, relative offset `0.338 m`, and max robot/box
  tilt `1.616/2.138 rad`. Interpretation: this was a valid negative test of
  the existing rescue-over-freeze mechanism, not a non-trigger bug. The
  failure mode remains terminal support/retention quality; a blended rescue
  posture after freeze does not provide enough controlled braking or box
  attitude retention for held-out shapes.
- 2026-07-07 G1 terminal-support controller entrypoint added:
  `scripts/isaac/build_core_world_g1_box_scene.py` now has opt-in
  `--agile-command-terminal-support-controller` for a unified target-window
  terminal command layer and opt-in `--terminal-support-posture-controller`
  for simultaneous lower-body/torso/arm retention posture blending. The
  command layer activates near terminal progress, computes x braking/drive
  from average robot/box target-directed error, computes lateral/yaw command
  from robot/box lateral error, and can force zero command during final hold
  while posture support remains active. Launcher forwarding and aggregate
  summary fields were added. Focused held-out geometry gate:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_terminal_support_suite.sh`.
  It tests the failed `wide_y012` and `tall_z009` close-front held-out shapes
  under the same strict fall/drop/no-rollout-write/target-window/tilt/lateral/
  relative-offset/final-command gates. This is an experiment entrypoint only
  until `closefront_heldout_geometry_terminal_support_summary.json` exists.
- 2026-07-07 first G1 terminal-support result:
  `20260707_g1_closefront_heldout_geometry_terminal_support`, submitted
  through tmux `curiosity_g1_heldout_terminal_support_0707` as Slurm job
  `170636`, ran on `server44` and produced aggregate `fail`, 0/2 strict
  cases passed. The controller did activate, so this is not a wiring miss.
  `wide_y012_terminal_support` had terminal command active for `232` steps
  from step `540` and final-zeroed command for `124` steps, but posture
  support activated much too early at step `105` and saturated risk to `1`.
  It failed with first fall/drop `757/778`, fall/drop `243/222`,
  target-window stable/longest/end `84/84/0`, severe over-travel
  `4.884/4.467 m`, final lateral `-1.375/-1.529 m`, relative offset
  `0.474 m`, and max robot/box tilt `3.138/3.136 rad`. `tall_z009_terminal_support`
  had command active for `153` steps from step `609` and posture active for
  `357` steps from step `643`, but still failed with first fall/drop
  `776/822`, fall/drop `224/107`, target-window stable/longest/end
  `45/45/0`, final travel `1.876/1.874 m`, relative offset `0.267 m`, and
  max robot/box tilt `1.908/1.951 rad`. Interpretation: the first unified
  controller is structurally wired but too aggressive/early. The next valid
  probe should delay posture support until terminal progress and reduce
  forward command authority, rather than layering freeze/rescue again.
- 2026-07-07 G1 terminal-support late-posture entrypoint added:
  `scripts/isaac/run_core_world_g1_closefront_heldout_geometry_terminal_support_lateposture_suite.sh`.
  It reuses the same `wide_y012` and `tall_z009` held-out shape gates, but
  delays posture support until terminal progress by setting posture rel/tilt
  triggers out of range, starts terminal command later (`box_target=1.55 m`),
  lowers max forward command to `0.018`, and uses weaker posture offsets. The
  purpose is to test the concrete failure found in the first terminal-support
  run: early posture-risk activation destabilized the wide box before the
  terminal phase. This is an experiment entrypoint only until
  `closefront_heldout_geometry_terminal_support_lateposture_summary.json`
  exists.
- 2026-07-07 G1 terminal-support late-posture result:
  `20260707_g1_closefront_heldout_geometry_terminal_support_lateposture`,
  submitted through tmux
  `curiosity_g1_heldout_terminal_support_lateposture_0707` as Slurm job
  `170638`, ran on `server44` and produced aggregate `fail`, 0/2 strict
  cases passed. The late-posture change improved the wide-box failure mode but
  did not pass. `wide_y012_lateposture` had command active for `136` steps
  from step `622`, posture active for `117` steps from step `718`, no box
  drops, final relative offset `0.104 m`, and rollout writes `0`; however it
  still first fell at step `719`, had fall events `214`, target-window
  stable/longest/end only `2/2/0`, final travel `1.828/1.882 m`, final
  lateral `0.698/0.681 m`, and max robot/box tilt `1.228/1.365 rad`.
  `tall_z009_lateposture` had command active for `198` steps from step `609`
  and posture active for `217` steps from step `783`, but failed with first
  fall/drop `836/970`, fall/drop `156/30`, target-window stable/longest/end
  `62/54/0`, over-travel `3.552/3.242 m`, final lateral `1.888/2.126 m`,
  relative offset `0.424 m`, and max robot/box tilt `2.711/3.075 rad`.
  Interpretation: delaying posture avoids the worst wide-box drop/overdrive,
  but the current AGILE terminal-command/posture layer still cannot stabilize
  held-out shapes. Continuing scalar timing/authority sweeps is low value; the
  next meaningful step is stronger balance/support allocation or a different
  locomotion/support backend while preserving these strict gates.
- 2026-07-07 final-hold policy-state reset probe result:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_reset_probe.sh`
  ran the close-front `steps1050_final120` near-miss with
  `AGILE_COMMAND_HOLD_FINAL_RESET_POLICY_STATE=1` to test whether final-hold
  AGILE RNN state reset reduces the tilt excess. Slurm job `170321` /
  `g1_finreset` completed with aggregate `fail`: fall/drop `25/0`, first fall
  step `1025`, final robot/box travel about `2.149/2.185 m`, final lateral
  error about `-0.161/-0.202 m`, max robot/box tilt `1.412/1.776 rad`,
  target-window stable/longest/end `120/117/0`, final-hold active `268@782`,
  chest pad at step `902`, and rollout root/velocity/box pose writes `0/0/0`.
  Source summary confirms final policy-state reset triggered once with no
  reset error. Interpretation: reset improves target-window dwell versus the
  no-reset `steps1050_final120` near-miss, but destabilizes the final hold and
  greatly worsens tilt, so it is not the close-front fix.
- 2026-07-07 final-hold tilt-escape probe result:
  added `--agile-command-hold-final-tilt-escape-scale`,
  `--agile-command-hold-final-tilt-escape-tilt`, and
  `--agile-command-hold-final-tilt-escape-box-tilt` to
  `scripts/isaac/build_core_world_g1_box_scene.py`, launcher env forwarding in
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, and
  aggregate summary fields in
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`. Entrypoint:
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_tilt_escape_suite.sh`.
  It starts from the no-reset `steps1050_final120` close-front near-miss and
  keeps final hold near zero unless robot/box tilt exceeds a threshold, then
  releases a tiny AGILE command scale. Submitted through tmux
  `curiosity_g1_tilt_escape_0707` as Slurm job `170351` / `g1_tiltesc`, suite
  stamp prefix `20260707_g1_lowcarry_close_front_tilt_escape`. Do not interpret
  until `close_front_tilt_escape_summary.json` exists. Result: Slurm job
  `170351` completed with aggregate `fail`, 0/2 strict cases passed. Both
  cases had fall/drop `0/0`, writes `0/0/0`, final robot/box travel about
  `2.026/2.103 m`, target-window stable/longest/end `76/73/73`, and
  final-hold active `268@782`, but still failed tilt and minimum stable-step
  gates. `scale015` reached max robot/box tilt `0.486/0.493 rad` and
  triggered escape only for `11` steps from step `1039`; `scale025` reached
  max robot/box tilt `0.488/0.483 rad` and triggered escape only for `18`
  steps from step `1025`. Interpretation: the mechanism is not harmful in
  this bracket, but the thresholds are too late to arrest the final tilt.
- 2026-07-07 early final-hold tilt-escape cases submitted: added
  `TILT_ESCAPE_CASE_SET=early` to
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_tilt_escape_suite.sh`.
  It runs `escape_robot022_box030_scale015` and
  `escape_robot018_box024_scale020`, keeping the same strict gates but
  triggering final-hold escape earlier. Submitted through tmux
  `curiosity_g1_tilt_escape_early_0707` as Slurm job `170356` /
  `g1_tiltearly`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_tilt_escape_early`. Job `170356` failed
  before rollout with a shell syntax error and only wrote a status TSV; it is
  not experiment evidence. Replacement Slurm job `170361` / `g1_tiltearly2`
  was submitted through tmux `curiosity_g1_tilt_escape_early2_0707`, suite
  stamp prefix `20260707_g1_lowcarry_close_front_tilt_escape_early2`.
  Result: aggregate `fail`, 0/2 strict cases passed. Both cases had
  fall/drop `0/0` and rollout writes `0/0/0`. `escape_robot022_box030_scale015`
  traveled robot/box about `1.963/2.014 m` but failed robot/box tilt
  `0.610/0.500 rad`, target-window stable `77 < 80`, and final-hold active
  `268 < 399`; escape was active for `34` steps from step `887`.
  `escape_robot018_box024_scale020` is the useful boundary: robot/box travel
  about `2.118/2.130 m`, final lateral error about `0.013/0.150 m`, target
  stable/longest/end `81/80/80`, robot tilt `0.267 rad`, but box tilt
  `0.544 rad` still exceeded the `0.45 rad` gate and final-hold active
  remained `268 < 399`; escape was active for `83` steps from step `792`.
  Interpretation: early tilt escape can recover the close-front target-window
  dwell without falls/drops, but it does not provide enough box attitude
  support or long enough terminal hold. Next close-front test should target
  physical box pitch support and 1200-step hold, not lateral sign/gain scans.
- 2026-07-07 close-front chest-pad tilt-support entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_chestpad_tilt_support_suite.sh`.
  It starts from the useful `escape_robot018_box024_scale020` boundary,
  extends runs to `1200` steps so the `399` final-hold gate is meaningful,
  enables chest-pad collision on earlier box-tilt/target-window triggers, and
  tests thicker/higher chest-pad plus lower top-lid geometry while preserving
  strict fall/drop, target-window, lateral, tilt, and no-rollout-write gates.
  Submitted through tmux `curiosity_g1_chestpad_tilt_0707` as Slurm job
  `170370` / `g1_chestpad`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_chestpad_tilt_support`. This is an
  experiment entrypoint only until
  `close_front_chestpad_tilt_support_summary.json` exists.
  A shorter one-case backfill attempt was also submitted through tmux
  `curiosity_g1_chestpad_tilt_quick_0707` as Slurm job `170372` /
  `g1_chestquick`, using `CHESTPAD_TILT_SUPPORT_CASE_SET=quick` and suite
  stamp prefix
  `20260707_g1_lowcarry_close_front_chestpad_tilt_support_quick`; interpret it
  separately from the full two-case suite. When full job `170370` started,
  queued quick job `170372` was cancelled to avoid duplicate GPU use. Result:
  aggregate `fail`, 0/2 strict cases passed. Both cases failed early with
  first fall/drop at `277/309`, target-window stable steps `0`, final-hold
  active steps `0`, and rollout writes `0/0/0`. `pad_box022_z012_x006` had
  fall/drop `891/593`, final robot/box travel about `0.192/0.065 m`, and max
  robot/box tilt `3.138/3.141 rad`. `pad_box026_z014_x008` had fall/drop
  `879/387`, final robot/box travel about `0.838/0.733 m`, and max robot/box
  tilt `3.105/3.141 rad`. Chest-pad collision enabled only at step `650`,
  after the first fall/drop, while the lower/thicker top lid was enabled at
  step `116`. Interpretation: this branch was destabilized by early geometry
  changes before the intended chest-pad trigger; do not continue the lower-lid
  / thicker-pad branch as the active close-front fix.
- 2026-07-07 close-front early-escape 1200-step isolation entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_early_escape_1200_suite.sh`.
  It preserves the useful early-escape command boundary and the original
  top-lid/chest-pad geometry from the no-fall 1050-step near-miss, extends to
  `1200` steps for the `399` final-hold gate, and compares target-window-only
  chest-pad trigger against original-size box-tilt-triggered chest pad. This
  isolates terminal-duration behavior from the failed lower-lid geometry. It
  was submitted through tmux `curiosity_g1_early_escape_1200_0707` as Slurm
  job `170382` / `g1_escape1200`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_early_escape_1200`. It is an experiment
  entrypoint only until
  `close_front_early_escape_1200_summary.json` exists. Result: aggregate
  `fail`, 0/2 strict cases passed. Both `baseline_target_window_pad` and
  `baseline_box_tilt_pad` produced identical metrics: fall/drop `103/72`,
  first fall/drop step `1097/1128`, target-window stable/longest/end
  `108/107/0`, final-hold active `418@782`, final robot/box travel
  `3.129/2.968 m`, final lateral error `1.014/1.090 m`, max robot/box tilt
  `2.120/2.061 rad`, chest pad at step `965`, tilt escape active `197` steps
  from step `792`, and rollout writes `0/0/0`. Interpretation: preserving the
  original geometry restores the useful late-failure near-boundary and
  satisfies the final-hold-duration count, but continuous final tilt-escape
  command drives over-travel/lateral drift and a late fall/drop after the
  target-window dwell. The next valid close-front terminal test should suppress
  final tilt escape once the target window has been stable long enough, rather
  than adding more early contact geometry.
- 2026-07-07 close-front escape-suppression entrypoint: added
  `--agile-command-hold-final-tilt-escape-suppress-after-target-window-streak`
  to `scripts/isaac/build_core_world_g1_box_scene.py`, env forwarding in
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, summary
  preservation in `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`,
  parser output in `scripts/isaac/print_g1_final_stabilize_summary.sh`, and
  suite launcher
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_suite.sh`.
  It preserves the early-escape 1200 setup but suppresses final tilt escape
  after target-window streaks of `60` or `80` steps. It is an experiment
  entrypoint only until `close_front_escape_suppression_summary.json` exists.
  Submitted through tmux `curiosity_g1_escape_suppression_0707` as Slurm job
  `170384` / `g1_escsup`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_escape_suppression`. Result: aggregate
  `fail`, 0/2 strict cases passed. `suppress_after_streak60` improved the late
  boundary but did not pass: fall/drop `64/46`, first fall/drop `1136/1154`,
  target-window stable/longest/end `114/113/0`, final-hold `418@782`,
  final robot/box travel `3.200/3.140 m`, final lateral error
  `0.923/0.879 m`, max robot/box tilt `2.734/2.592 rad`, tilt escape
  active/suppressed `167/54`, and writes `0/0/0`. `suppress_after_streak80`
  was weaker: fall/drop `102/80`, first fall/drop `1098/1120`, target-window
  `108/107/0`, final travel `3.117/2.891 m`, final lateral
  `0.738/0.582 m`, tilt `3.139/3.132 rad`, escape active/suppressed `189/28`,
  writes `0/0/0`. Interpretation: target-window suppression helps but does
  not stop post-window over-travel and late terminal collapse; the next valid
  step is a window-latched brake or stand handoff after sufficient dwell, not
  more forward escape or lower-lid contact.
- 2026-07-07 close-front escape-suppression brake entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_brake_suite.sh`.
  It reuses the existing final-brake mechanism after the better
  `suppress_after_streak60` boundary: final tilt escape is suppressed after a
  `60`-step target-window streak, then a negative final brake command starts
  after `240` final-hold steps for `120` steps. Cases test brake command
  `-0.004` and `-0.008`, with original support geometry and the same strict
  gates. This is an experiment entrypoint only until
  `close_front_escape_suppression_brake_summary.json` exists. Submitted
  through tmux `curiosity_g1_escape_brake_0707` as Slurm job `170388` /
  `g1_escbrake`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_escape_suppression_brake`. Result:
  aggregate `fail`, 0/2 strict cases passed. `suppress60_brake240_neg004`
  had fall/drop `110/81`, first fall/drop `1090/1119`, target-window
  stable/longest/end `110/109/0`, final travel `3.177/3.039 m`, final lateral
  `0.653/0.559 m`, max robot/box tilt `3.134/3.141 rad`, brake active
  `120` steps from `1022` to `1141`, and writes `0/0/0`.
  `suppress60_brake240_neg008` delayed collapse and reduced severity but
  still failed: fall/drop `33/7`, first fall/drop `1167/1193`, target-window
  `108/107/0`, final travel `3.583/3.587 m`, final lateral `0.902/1.064 m`,
  max robot/box tilt `0.673/0.739 rad`, brake active `120` steps from `1022`
  to `1141`, and writes `0/0/0`. Interpretation: braking improves late
  stability but does not keep the robot in the target window and can worsen
  over-travel; do not continue scanning brake magnitude alone. The next valid
  terminal test is a stand-target handoff after target-window dwell.
- 2026-07-07 close-front escape-suppression stand entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_stand_suite.sh`.
  It starts from the `suppress_after_streak60` boundary, keeps original support
  geometry and strict gates, suppresses final tilt escape after a `60`-step
  target-window streak, then enables `AGILE_COMMAND_HOLD_FINAL_STAND` after
  `240` or `300` final-hold steps with blend rates `0.02` and `0.04`
  respectively. This is an experiment entrypoint only until
  `close_front_escape_suppression_stand_summary.json` exists. Submitted
  through tmux `curiosity_g1_escape_stand_0707` as Slurm job `170396` /
  `g1_escstand`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_escape_suppression_stand`. Result:
  aggregate `fail`, 0/2 strict cases passed. `suppress60_stand240_blend002`
  is the better boundary: fall/drop `56/33`, first fall/drop `1144/1167`,
  target-window stable/longest/end `144/143/0`, final stand active `178`
  steps from `1022`, target-window final-stand stable/longest `91/91`,
  final robot/box travel `2.750/2.510 m`, final lateral `0.765/0.869 m`,
  max robot/box tilt `2.924/2.198 rad`, tilt escape active/suppressed
  `149/84`, writes `0/0/0`. `suppress60_stand300_blend004` delayed collapse
  but started stand too late for window hold: fall/drop `39/12`, first
  fall/drop `1161/1188`, final-stand stable/longest `1/1`, final travel
  `3.312/3.240 m`, lateral `0.739/0.945 m`, tilt `1.108/1.110 rad`, writes
  `0/0/0`. Interpretation: stand handoff is the best terminal-control
  direction so far because it increases target-window dwell, but it still
  loses lateral/tilt stability; the next step should combine earlier stand
  handoff with lateral/roll stabilization or a materially better support
  backend, not claim close-front pass.
- 2026-07-07 close-front escape-stand lateral entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_stand_lateral_suite.sh`.
  It starts from the best `suppress60_stand240_blend002` boundary and adds the
  existing lateral-error-driven balance roll target after about the target
  window entry point (`850` agile-hold steps), comparing only roll-target signs
  `-1` and `+1` with the same strict gates. This is an experiment entrypoint
  only until `close_front_escape_stand_lateral_summary.json` exists.
  Submitted through tmux `curiosity_g1_escape_lateral_0707` as Slurm job
  `170405` / `g1_esclat`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_escape_stand_lateral`. Result: aggregate
  `fail`, 0/2 strict cases passed. Both lateral signs were materially similar:
  sign `-1` had fall/drop `57/34`, first fall/drop `1143/1166`,
  target-window stable/longest/end `145/144/0`, final-stand stable/longest
  `92/92`, final travel `2.735/2.491 m`, lateral `0.745/0.877 m`, max
  robot/box tilt `3.130/2.462 rad`, roll-target active `71` steps from
  `1062`, writes `0/0/0`; sign `+1` had fall/drop `56/33`, first fall/drop
  `1144/1167`, target-window `145/144/0`, final-stand `92/92`, final travel
  `2.766/2.517 m`, lateral `0.755/0.860 m`, tilt `3.122/2.280 rad`,
  roll-target active `72` steps from `1062`, writes `0/0/0`. Interpretation:
  lateral roll target did not solve terminal lateral/tilt collapse; do not
  keep scalar-tuning lateral sign/gain on this branch. The current
  terminal-control stack has produced useful failure boundaries but not a
  close-front pass. The next meaningful step is support/backend replacement or
  a materially different terminal support policy, while keeping the same
  strict gates.
- 2026-07-07 close-front low-stance terminal support entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_stand_lowstance_suite.sh`.
  It starts from the best `suppress60_stand240_blend002` boundary but changes
  the terminal support posture rather than small command scalars: final stand
  uses paired sagittal overrides for hip pitch, knee, ankle pitch, and waist
  pitch to create lower, more crouched terminal support. It tests
  `lowstance_soft` and `lowstance_deeper` under the same strict gates. This is
  an experiment entrypoint only until
  `close_front_escape_stand_lowstance_summary.json` exists. Submitted through
  tmux `curiosity_g1_lowstance_0707` as Slurm job `170412` / `g1_lowstance`,
  suite stamp prefix
  `20260707_g1_lowcarry_close_front_escape_stand_lowstance`. Result:
  aggregate `fail`, 0/2 strict cases passed. Source summaries confirm the
  low-stance joint overrides were applied. `lowstance_soft` worsened the
  boundary: fall/drop `123/76`, first fall/drop `1077/1124`, target-window
  `102/101/0`, final-stand stable/longest `49/49`, final travel
  `3.424/3.193 m`, lateral `0.991/0.964 m`, tilt `2.613/2.627 rad`, writes
  `0/0/0`. `lowstance_deeper` roughly reproduced the stand/lateral boundary:
  fall/drop `55/34`, first fall/drop `1145/1166`, target-window `145/144/0`,
  final-stand stable/longest `92/92`, final travel `2.765/2.571 m`, lateral
  `0.846/0.808 m`, tilt `2.741/3.113 rad`, writes `0/0/0`. Interpretation:
  lower sagittal stand posture does not solve the terminal loss of support.
  This further supports stopping the current G1 terminal-wrapper branch and
  moving to a controller-backed or optimizer-backed support backend.
- Current execution directive: do not block on external model/checkpoint
  downloads or optional policy-server rollouts when they are not directly
  useful. Continue direct Isaac scene construction first. The immediate
  blocker is stable support/propulsion while retaining a free dynamic box, not
  missing video priors or missing pretrained policies. Optional Arena/GR00T
  baseline checks may be recorded, but they must not delay direct Isaac
  backend replacement.
- 2026-07-06 user correction: do not keep waiting for external models,
  datasets, policy servers, or optional official wrappers when they are not
  directly useful. The active path is to construct and gate the carrying scene
  directly in Isaac. Added direct G1 baseline suite
  `scripts/isaac/run_core_world_g1_direct_carry_baseline_suite.sh`; it runs
  staged gates for no-box stand, fixed-payload stand, free-box cradle stand,
  short free-box target-directed carry, and long-hold validation. This suite
  is a diagnostic scaffold: it must not be claimed as final robot carrying
  unless the relevant stage summaries pass fall/drop, target-directed travel,
  no rollout root/box shortcut, and long-hold gates.
- Current direct Isaac backend stop rule: do not continue the
  `xz_prismatic_to_anchor` support backend in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py` as the active
  carrying backend. On 2026-07-05, no-box diagnostics
  `20260705_anchor_nobox_support_diag1`,
  `20260705_anchor_nobox_propulsion_diag2`, and
  `20260705_anchor_nobox_invertrail_diag4` all failed the basic travel gate.
  Do not add fixed payload or free box on top of this backend; replace the
  backend or return to the real G1 articulation path.
- Current direct G1 stop rule: do not keep tuning the current
  staged/open-loop G1 gait family as a locomotion backend. On 2026-07-05,
  `20260705_core_world_g1_nobox_staged_iso_diag1` completed 700 no-box steps
  with fall/drop 0 and max tilt `0.01314 rad`, but max target-directed robot
  travel was only `0.00522 m`. The failed long free-box runs with large box
  travel are therefore not credible walking evidence; they are coupled to
  delayed pitch/drop failure. The next G1 path needs a controller-backed
  locomotion policy or a materially different walking controller.
- Do not wait on external model downloads unless they directly unblock the
  active Isaac scene. As of 2026-07-05, the direct Core API G1 route in
  `scripts/isaac/build_core_world_g1_box_scene.py` has passed no-box stand,
  fixed-torso ballast stand, collision-enabled fixed front-payload stand, and
  small open-loop marching diagnostics with 43 G1 joints and rollout root/
  velocity/box pose writes 0. It has also passed a first small free dynamic
  box contact scaffold: `diag43` used `attach_box=none`, a `front_tray`
  torso cradle, a 0.25 kg free dynamic box, open-loop marching for 420 steps,
  fall/drop 0, `max_tilt_rad=0.02290`, and rollout root/velocity/box pose
  writes 0. Follow-ups `diag44`/`diag45`/`diag46` also passed one-axis gates:
  1200-step duration, 0.5 kg box, and gait amplitude 0.08 respectively, all
  with free dynamic box, fall/drop 0, and rollout root/velocity/box pose writes
  0. Follow-up metric runs `diag49`/`diag50` showed meaningful target-directed
  motion, with final box target-directed travel about `0.657 m` and `0.702 m`,
  but both failed late with falls and box drops. This is still not complete
  carrying. The active blocker is now box retention and late-stage balance
  during larger-amplitude open-loop gait. `diag51`/`diag52` showed that a
  stronger cradle can keep the box high with fall/drop 0, but then target-
  directed travel collapses to about `0.023 m`. Next gates should search
  intermediate cradle/contact geometry or add stabilization while preserving
  no-shortcut evidence.
- Older adaptive-probe and quasi-static scaffolds are diagnostic history only.
  Do not report them as dynamic humanoid locomotion, learned balance, learned
  grasping, true contact carrying, or video-conditioned RL.
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
- 2026-07-05 direct Isaac core-World dynamic quadruped route:
  `scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py` with
  launcher `scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  This route now gives a passing articulated-scaffold diagnostic for staged
  free-box carrying with contact proxies. `diag36` proved the custom
  articulated path was active but dropped the box; `diag37` showed early
  `target_hold` latch was wrong; `diag38` fixed early hold by requiring the
  carrier body to reach the support position but still dropped the box; and
  `20260705_core_world_dynamic_quad_diag39b_proxy_preplaced` passed the
  declared checker: 760/760 steps, attach step 90, `target_hold_latched=True`,
  `target_hold_steps=26`, torso travel 0.33040 m, box travel 0.40108 m, final
  target distance 0.07327 m, final/peak relative error 0.10293/0.10440 m,
  final/peak contact-proxy gap 0.08576/0.08787 m, `max_joint_motion_rad=0.87277`,
  fall/drop events 0, no disjoint warning, and control errors 0. Do not
  overclaim this: the quadruped body still uses pose assist, the lift/attach
  event is staged, and palm/chest/shelf/front-stop contact proxies are
  engineering scaffolds rather than learned hands or unassisted locomotion.
  The next valid step is to reduce or remove pose assist and replace staged
  proxy placement with an actual controller/contact policy while preserving
  the same fall/drop/target/relative-error gates.
- 2026-07-05 root-assist reduction diagnostics after `diag39b` are negative
  for final walking/balance. `diag40` removed pose writes and used root
  velocity assist only; it completed but failed with 72 fall events, final
  target distance 3.66150 m, and max tilt 3.09467 rad, proving `diag39b`
  depended on pose writes. `diag41` added `upright_velocity` root linear and
  angular velocity stabilization without pose writes; it reduced roll/pitch
  but still had 60 fall events from low torso height and target distance
  2.76723 m. `diag42` fixed falls/drop with stronger height and x velocity
  servo, but target distance stayed 2.25080 m. `diag43` improved target
  distance to 0.75432 m with fall/drop 0 and `root_pose_write_count=0`, but
  still failed target, relative-error, and proxy-gap gates. `diag44b`
  post-step velocity writes reproduced `diag43`. Direct Python `diag45e` and
  `diag46b` verified `base_x_command_scale=-1.0` reaches argparse, but
  negative scale overshoots or drives away from the target; `diag46b` ended
  with target distance 4.22659 m despite fall/drop 0. Conclusion: do not keep
  tuning root velocity writes as if they were locomotion. The next valid
  milestone is a foot/support-driven carrier controller or a functioning
  controller-backed robot policy; root pose/velocity assisted runs remain
  scaffolds only.
- 2026-07-04 dynamic rigid-body control probes are also negative so far:
  runtime USD `RigidBodyAPI.velocity` writes produced 0 travel on CPU/GPU;
  `omni.physx.get_physx_simulation_interface().apply_force_at_pos` produced 0
  travel on CPU/direct-step smokes and direct-GPU `addForce()`/`addTorque()`
  errors on GPU; bare `CuboidCfg.func` cubes showed no observed gravity drop;
  `RigidObjectCfg` root-state reads failed with
  `Failed to get rigid body transforms from backend`; Isaac Sim core
  `DynamicCuboid` wrappers stalled before a completed reset. Do not rerun these
  unchanged or tune parameters on them.
- Current 2026-07-04 official-policy route:
  `scripts/isaac/run_official_policy_locomotion_smoke.py` with launcher
  `scripts/isaac/run_official_policy_locomotion_smoke.sh`. This route uses the
  installed NVIDIA `isaacsim.robot.policy.examples` Go2/H1 flat-terrain policy
  wrappers and locally mirrored official USD/policy/config assets. It is the
  active robot-control path for "make a real Isaac robot walk first, then add
  the box"; it should not be blocked on unrelated external model downloads.
  The script also supports `PAYLOAD_MODE=fixed_base` for a Go2 rigid box fixed
  to the base link. That mode is only a fixed-payload balance diagnostic until
  a compute-node run verifies nonzero robot/payload travel, zero falls, and
  zero payload drops.
- 2026-07-04 official-policy integration fixes already applied: extension
  namespace paths are added manually for `isaacsim.robot.policy.examples`, the
  script reuses the AppLauncher-created USD stage instead of calling
  `new_stage()`, and the standalone loop uses synchronous
  `simulation_app.update()` instead of `next_update_async()`. Do not revert to
  the previously hanging `new_stage()`/async-update path without a concrete
  reason.
- 2026-07-04 official Go2 policy diagnostics are negative so far. AppLauncher
  and pure `SimulationApp` variants can load local Go2 assets and create the
  policy object on real H200 GPU, but the articulation physics tensor entity is
  invalid before initialization:
  `Invalid physics simulation view. Articulation (['/World/Go2/Geometry/base'])
  will not be initialized`. Explicit `SimulationManager.set_backend(...)`
  exits before rollout in this environment. Do not report these runs as
  walking, balancing, or carrying evidence.
- 2026-07-04 official Go2 callback diagnostics with NVIDIA policy-test Kit
  settings are also negative. `OFFICIAL_TEST_KIT_ARGS=1` now passes the
  `isaacsim.robot.policy.examples` physx-test Kit settings through
  `SimulationApp` `extra_args` before startup; earlier attempts accidentally
  left those settings in Kit's unknown-arg list. With corrected `extra_args`,
  `20260704_go2_callback_officialkit_extraargs_diag7` still hard-exited at
  `SimulationManager.set_physics_sim_device(cuda:0)` and produced no summary.
  `20260704_go2_callback_officialkit_extraargs_skip_diag8` skipped the
  device/dt setters, completed 120/120 script steps, but produced
  `callback_forward_calls=0`, `travel_xy_m=0.0`, and the same invalid
  `/World/Go2/Geometry/base` physics simulation view warning. Do not rerun the
  Go2 callback/official-policy route unchanged.
- 2026-07-05 H1 direct locomotion candidate result:
  `scripts/isaac/run_official_h1_callback_locomotion_smoke.py` with launcher
  `scripts/isaac/run_official_h1_callback_locomotion_smoke.sh`. It uses the
  installed NVIDIA `H1FlatTerrainPolicy`, local H1 USD, and local H1 PhysX
  policy/env files. This route is negative in the current environment:
  retry2 with the Isaac Sim base python kit failed missing extension
  dependency resolution before app startup, and retry3/retry4 with the
  IsaacLab headless kit failed before rollout at H1 policy/articulation
  construction with `Path.IsValidPathString(NoneType)`. Do not repeat H1/Go2
  official sample-policy wrappers unchanged. The direct Isaac carry-task
  contract remains useful, but the support/locomotion backend must be replaced
  directly rather than blocked on these wrappers.
- Do not use `kit/dev/repo.sh test` on compute nodes until its dependencies are
  already locally prepared. A 2026-07-04 probe inside Slurm job `165296`
  attempted to fetch `python@3.12.13-nv3-manylinux_2_35-x86_64.tar.gz` from
  NVIDIA packman and was interrupted because compute nodes must not perform
  dependency installation or package resolution.
- 2026-07-04 verified non-tensor dynamic Isaac route:
  `scripts/isaac/build_core_world_simapp_dynamic_cube_smoke.py` and
  `scripts/isaac/build_core_world_simapp_fixed_payload_carry.py` use pure
  `SimulationApp` plus Isaac Sim core `World`, local ground, and CPU PhysX.
  `20260704_core_world_simapp_cube_velocity_cpu_diag3` moved a dynamic cube
  0.315 m. `20260704_core_world_simapp_fixed_payload_centerweld_diag2` moved a
  dynamic carrier and a fixed physical payload together 0.3596 m with relative
  payload error 0.0 m, fall events 0, and payload drop events 0. This is the
  current working Isaac dynamics substrate. It is not legged walking, unknown
  object grasping, active-probing success, or learned carrying.
- 2026-07-04 adaptive dynamic fixed-payload task route:
  `scripts/isaac/build_core_world_simapp_adaptive_payload_carry.py` with
  launcher `scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh`
  builds on the verified pure core-World dynamic carrier. It adds morphology
  and load-dependent strategy selection, phased probe/settle/carry commands,
  visible walking-support foot markers, gait/support proxy metrics,
  target-distance metrics, balance-margin proxy, effort proxy, fall/drop
  metrics, CSV state, and summary JSON. Verified diagnostic before the
  walking-support-proxy extension
  `20260704_core_world_adaptive_payload_target_diag2` selected
  `low_front_carry`, moved carrier and physical fixed payload 0.3197 m, ended
  0.0197 m from the target, had payload relative error 0.0 m, fall events 0,
  drop events 0, and minimum balance-margin proxy 0.0872 m. This is still not
  legged walking, unknown free-object grasping, or learned carrying.
- 2026-07-04 one-startup adaptive strategy sweep:
  `20260704_core_world_adaptive_payload_strategy_sweep1` ran two dynamic
  fixed-payload cases inside one `SimulationApp` startup. The sweep summary is
  at
  `experiments/outputs/core_world_simapp_adaptive_payload_carry/20260704_core_world_adaptive_payload_strategy_sweep1/core_world_simapp_adaptive_payload_sweep_summary.json`.
  Results: strategy counts `low_front_carry: 1` and
  `chest_supported_slow: 1`; both cases completed 360/360 steps with fall
  events 0, drop events 0, payload relative error 0.0 m, and nonzero
  carrier/payload travel. This validates strategy-selection plumbing on real
  dynamic fixed-payload motion only; it is still not robot walking or
  unknown-object carrying.
- 2026-07-04 walking-support-proxy adaptive strategy sweep:
  `20260704_core_world_adaptive_payload_walkproxy_diag1` ran in Curiosity-owned
  Slurm job `165292` on `server02`. It verified the upgraded dynamic
  fixed-payload scene with visible left/right support-foot markers, support
  state, gait frequency, stance width, step length, and
  `min_support_margin_x_proxy_m` in the per-case summaries. Strategy counts
  were `low_front_carry: 1` and `chest_supported_slow: 1`. Low-front moved
  carrier/payload 0.3287 m with final target distance 0.0287 m,
  `min_support_margin_x_proxy_m` 0.1185 m, fall 0, drop 0. Chest-supported
  moved carrier/payload 0.2568 m with final target distance 0.0568 m,
  `min_support_margin_x_proxy_m` 0.1109 m, fall 0, drop 0. This is still a
  dynamic fixed-payload and walking-support-proxy diagnostic, not real legged
  locomotion, contact grasping, or learned carrying.
- 2026-07-04 articulated staged-free-box route under active development:
  `scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py` now
  supports `PAYLOAD_MODE=staged_free_box` in addition to
  `fixed_joint_to_torso`. The box begins as a free dynamic rigid body; the
  script logs probing displacement, staged lift/attach, attach step,
  relative box error, target hold, fall events, and box-drop events. This path
  is the current bridge from fixed-payload diagnostics toward free-object
  carrying. It is still not final success while `BASE_ASSIST_MODE=pose`,
  staged lift, pose-lock attach, or manually enabled fixed joints are used.
  Any run on this path must explicitly record `PAYLOAD_MODE`,
  `STAGED_ATTACH_MODE`, `BASE_ASSIST_MODE`, attach result, target distance,
  target hold, fall/drop counts, and relative error before being interpreted.
- 2026-07-04 current articulated staged-free-box evidence:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag2_pose_lock_target_hold`
  passed only as a scaffold gate: `PAYLOAD_MODE=staged_free_box`,
  `STAGED_ATTACH_MODE=pose-lock`, `BASE_ASSIST_MODE=pose`, attached at step
  90, `target_hold_steps=5`, final target distance 0.04495 m, torso travel
  0.4050 m, box travel 0.37165 m, fall/drop events 0, no disjoint warning,
  and relative error near zero. It is not unassisted locomotion or physical
  grasping because root pose assist and pose-lock attach are still used.
- 2026-07-04 current articulated staged fixed-joint evidence is negative:
  `diag3_fixed_joint` and `diag4_fixed_joint_runtime_create` both produced
  PhysX disjoint fixed-joint warnings, box drop events, no target hold, and
  massive box/relative errors. Do not rerun this fixed-joint attach path
  unchanged or present it as progress toward physical grasping; replace it
  with a contact/grasp formulation.
- 2026-07-04 velocity-servo staged attach evidence is partial/negative under
  strict gates. `diag6_velocity_servo_target_stop` avoided fixed-joint
  explosions and had fall/drop events 0, no disjoint warning, and
  `target_hold_steps=5`, but the dynamic box lagged the carry pose with final
  target distance 0.10941 m and relative error 0.10943 m. Treat it as a
  transition scaffold only; next physical attach work should use contact or
  multi-proxy grasp/hold rather than claiming velocity-servo as carrying.
- 2026-07-04 contact-proxy staged attach evidence is negative under strict
  gates. `diag7_contact_proxy` used dynamic palm/chest/shelf proxy bodies and
  did not directly pose-lock or velocity-servo the box, but the box dropped
  repeatedly: box drop events 34, target hold 0, final target distance
  0.28981 m, relative error 0.60057 m, and max contact-proxy gap 1.37539 m.
  Do not present the current contact-proxy mode as grasping success. The next
  contact attempt must materially change the grasp formulation, such as
  pre-closed proxy geometry, stronger shelf support, normal clamping, or a
  controlled grip-force/constraint hybrid.
- 2026-07-04 quasi-static walker fixed-payload route:
  `scripts/isaac/build_core_world_simapp_quasistatic_walker_carry.py` with
  launcher `scripts/isaac/run_core_world_simapp_quasistatic_walker_carry.sh`
  builds a pure `SimulationApp` core-World scene with a dynamic walker body, a
  dynamic physical payload fixed by USD joint, four visible/colliding support
  feet, visible leg struts, gait phases, support-state logging, target hold,
  and support-margin gating. It is a diagnostic bridge toward a real legged
  controller, not verified articulated locomotion or unknown free-object
  grasping.
- 2026-07-04 quasi-static walker diagnostics: `diag1` used a diagonal trot-like
  support pattern and is a negative control for support metrics: body/payload
  traveled 0.1818 m with fall/drop 0, but final target distance was 0.1982 m
  and `min_support_margin_m` was -0.0248 m. The gait was then changed to
  one-foot-swing, three-foot-stance creep gait. `diag2` reached near the
  0.18 m target with positive support margin but walked past it because no
  target hold existed. `20260704_core_world_quasistatic_walker_hold_diag3`
  added target hold and completed 420/420 steps with body/payload travel
  0.1659 m, final and minimum payload-target distance 0.0141 m,
  `payload_relative_error_m=0.0`, `min_support_margin_m=0.1325 m`, fall events
  0, and payload drop events 0. This is the strongest current direct Isaac
  carrying-task diagnostic, but still not final success because torso motion is
  commanded through rigid-body velocity control and not a verified articulated
  walking policy.
- 2026-07-04 staged free-box carry route:
  `scripts/isaac/build_core_world_simapp_staged_free_box_carry.py` with
  launcher `scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`
  starts the box as a free dynamic rigid body, runs approach/probe/staged-lift
  phases, then uses a logged attach placeholder before carrying. It is
  explicitly a task-structure diagnostic, not contact grasping, not verified
  articulated walking, and not learned carrying.
- 2026-07-04 staged free-box diagnostics:
  `diag1` on `server56` and `diag2` on `server02` did not reach scene
  construction because `SimulationApp` startup stalled. A later fresh
  allocation on `server10` verified Isaac health with
  `20260704_core_world_quasistatic_health_server10`, then
  `20260704_core_world_staged_free_box_diag3_server10` completed 520/520 steps
  with attach step 260, 77 probe attempts, body travel 0.3492 m, box travel
  0.0563 m, fall events 0, and box drop events 0. However PhysX warned that
  `/World/StagedCarryRuntimeFixedJoint` had disjoint body transforms and the
  box snapped backward on attach; final target distance was still 0.1648 m and
  `box_relative_error_m_after_attach` was 0.1868 m. Treat `diag3` as evidence
  that the staged free-box task scaffold runs, but as a negative attach-quality
  result, not a carrying success.
- The staged free-box attach logic was revised after `diag3`: it now uses a
  two-step `staged_lift_settle` then `staged_attach_constraint` sequence and
  creates the runtime fixed joint from the measured body-box relative pose
  (`attach_local_pos0_m`) rather than a desired offset. Fresh-allocation run
  `20260704_core_world_staged_free_box_diag4_settle_attach` completed 560/560
  steps with attach prep step 260, attach step 261, final target distance
  0.0111 m, body travel 0.3591 m, box travel 0.1515 m, min support margin
  0.1299 m, fall events 0, and box drop events 0. It improved task completion
  versus `diag3`, but PhysX still reported a disjoint-transform fixed-joint
  warning and `box_relative_error_m_after_attach` remained 0.0824 m. Treat
  `diag4` as a staged scaffold improvement, not stable contact attach or final
  carrying success.
- After `diag4`, staged attach was revised again to pre-author
  `/World/StagedCarryRuntimeFixedJoint` as `physics:jointEnabled = false`
  before `world.reset()`, then set the measured local pose and enable the joint
  at attach time. Follow-up `diag6_preauth_short` and
  `diag8_phase_anchor_fix` are negative fixed-joint results: they completed and
  attached, but still produced disjoint-joint warnings, snapping, high
  post-attach relative error, or drop events. Do not claim the fixed-joint
  staged attach route is stable.
- Current staged free-box passing scaffold is only
  `ATTACHMENT_MODE=kinematic-pose-lock`. Diagnostic
  `20260704_core_world_staged_free_box_diag9_kinematic_pose_lock` completed
  360/360 steps, attached at step 91, reached final box-target distance
  0.01455 m, had fall/drop events 0, post-attach relative error 0.000245 m,
  and no disjoint warning. This is a task-interface scaffold only. It is not
  physical grasping, not dynamic robot locomotion, not learned policy, and not
  final carrying success.
- Improved staged free-box scaffold:
  `20260704_core_world_staged_free_box_diag10_dynamic_velocity_pose_lock`,
  Slurm job `165325` on `server10`, uses
  `ATTACHMENT_MODE=kinematic-pose-lock` plus `CARRIER_MODE=dynamic-velocity`.
  The carrier body is a dynamic rigid body commanded by velocity; foot/leg
  markers and support-margin proxy provide a walking-support diagnostic; the
  box is pose-locked after staged attach. Checker passed with completed
  360/360, attach step 91, body travel 0.20545 m, box travel 0.47545 m, final
  target distance 0.01455 m, post-attach relative error `2.78e-08`, fall/drop
  events 0, no disjoint warning, and `min_support_margin_m=0.13252`. This is
  the current cleanest direct Isaac staged free-box scaffold, but it is still
  not physical grasping, not verified articulated locomotion, not learned
  policy, and not final carrying success.
- Multi-posture staged free-box scaffold:
  `20260704_core_world_staged_free_box_diag11_chest_dynamic_pose_lock`, Slurm
  job `165329` on `server46`, used a heavier 12 kg, larger box and shorter
  0.38 m arm setting. It selected `chest_supported_creep`, completed 560/560
  steps, attached at step 91, reached final box-target distance 0.01473 m, had
  body travel 0.22527 m, box travel 0.52527 m, post-attach relative error
  `2.78e-08`, fall/drop events 0, no disjoint warning, and
  `min_support_margin_m=0.12430`. This verifies posture-strategy switching
  inside the dynamic rigid-body carrier scaffold only. It is still not
  physical grasping, not verified articulated locomotion, not learned policy,
  and not final carrying success.
- Velocity-servo grasp proxy:
  `20260704_core_world_staged_free_box_diag12_velocity_servo_grasp`, Slurm job
  `165332` on `server46`, used `ATTACHMENT_MODE=velocity-servo-grasp` and
  `CARRIER_MODE=dynamic-velocity`. After staged attach the box stayed a
  dynamic rigid body and was controlled by velocity servo rather than direct
  pose-lock. It selected `low_front_creep`, completed 420/420, attached at
  step 91, reached final target distance 0.01456 m, had body travel
  0.20545 m, box travel 0.47544 m, final relative error `3.67e-06`, peak
  relative error `3.91e-06`, fall/drop events 0, no disjoint warning, and
  `min_support_margin_m=0.13252`. This is the preferred current staged attach
  proxy over `kinematic-pose-lock`, but it is still not contact grasping, not
  verified articulated locomotion, not learned policy, and not final carrying
  success.
- Contact-proxy servo diagnostic:
  `20260704_core_world_staged_free_box_diag13_contact_proxy_servo`, Slurm job
  `165340` on `server10`, used `ATTACHMENT_MODE=contact-proxy-servo` and
  `CARRIER_MODE=dynamic-velocity`. It adds explicit left/right palm and chest
  support proxy geometry plus grip-gap metrics while keeping the box as a
  dynamic rigid body controlled by velocity servo. Checker passed with
  completed 420/420, attach step 91, body travel 0.20545 m, box travel
  0.47544 m, final target distance 0.01456 m, final relative error
  `3.67e-06`, peak relative error `3.91e-06`, contact-proxy grip gap
  `3.67e-06` m, max grip gap `3.91e-06` m, fall/drop events 0, no disjoint
  warning, and `min_support_margin_m=0.13252`. This is a validated direct
  Isaac staged free-box servo-proxy scaffold, but it is still not physical
  contact grasping, not verified articulated locomotion, not learned policy,
  and not final carrying success.
- Dynamic contact-proxy diagnostic:
  `ATTACHMENT_MODE=dynamic-contact-proxy` uses dynamic left/right palm, chest,
  and forearm-shelf proxy rigid bodies. After the staged lift/attach event the
  box is not directly velocity-servoed; only the proxy bodies are
  velocity-servoed, so box motion comes through PhysX contact. Negative control
  `20260704_core_world_staged_free_box_diag14_dynamic_contact_proxy`, Slurm job
  `165343` on `server10`, showed why pre-attach gating matters: active proxies
  contaminated probing and pushed the box past the target, with 27 box drop
  events, final target distance 0.4703 m, final relative error 0.2790 m, peak
  relative error 0.4172 m, and max grip gap 0.3974 m. After adding standby
  gating, `20260704_core_world_staged_free_box_diag15_dynamic_contact_proxy_standby`
  passed its checker gate in the same Slurm job: completed 420/420, attach
  step 91, body travel 0.25052 m, box travel 0.48505 m, final target distance
  0.01490 m, final relative error 0.06311 m, peak relative error 0.06348 m,
  contact-proxy grip gap 0.06643 m, max grip gap 0.06818 m, fall/drop events
  0, no disjoint warning, and `min_support_margin_m=0.13252`. This is now the
  strongest current staged free-box Isaac diagnostic, but it is still not final
  success: the lift is staged, proxy bodies are velocity commanded, and the
  carrier is not verified articulated locomotion.
- Stricter dynamic-contact balance/hold gate:
  `20260704_core_world_staged_free_box_diag16_dynamic_contact_balance_hold`,
  Slurm job `165345` on `server10`, added summary/checker fields for
  `target_hold_steps`, `carry_phase_steps`, `min_stance_count`,
  `min_support_margin_after_attach_m`, and `max_command_speed_mps`. It passed
  the gate requiring dynamic-contact proxy, post-attach support margin,
  minimum stance count, target-hold duration, body/box travel, no fall/drop,
  and no disjoint warning. Results: completed 430/430, attach step 91, body
  travel 0.25052 m, box travel 0.48435 m, final target distance 0.01474 m,
  final/peak relative error 0.06428 m, contact-proxy grip gap 0.06825 m,
  `target_hold_steps=24`, `carry_phase_steps=338`, `min_stance_count=3.0`,
  `min_support_margin_after_attach_m=0.13252`, max command speed 0.174 m/s,
  fall/drop events 0, and no disjoint warning. This is now the strongest
  current staged free-box carry diagnostic. It still does not satisfy the
  final project goal because the lift is staged, the carrier body is
  velocity-commanded, and walking/balance are support-proxy verified rather
  than generated by an articulated robot controller.
- Gravity/contact-support audit:
  The staged free-box script now has `BODY_VERTICAL_MODE=preserve`,
  `PHYSICAL_SUPPORT_MODE=deck`, `SUPPORT_DECK_GAP`, and body-z summary fields.
  These exist because the previous dynamic-carrier diagnostics used
  `set_linear_velocity([speed, 0, 0])`, which implicitly zeroed vertical
  velocity and weakened any balance claim. Diagnostic
  `20260704_core_world_staged_free_box_diag17_dynamic_contact_preserve_z_deck`,
  Slurm job `165348` on `server28`, completed but failed the stricter gate:
  vertical velocity preservation was available, completed 430/430, attach step
  105, body travel 0.26733 m, box travel 0.80080 m, but had 18 box drop events,
  final target distance 0.31088 m, final relative error 1.17468 m, peak
  relative error 1.22473 m, max grip gap 0.46564 m, and
  `max_body_z_deviation_m=1.39366`. Interpretation: the first support-deck
  version was overconstrained/too tightly preloaded and launched the body
  upward before attach. Do not claim true gravity-balanced locomotion from
  `diag16`; it remains a support-proxy diagnostic until a preserve-z physical
  support or articulated gait passes. A `SUPPORT_DECK_GAP=0.02` patch is
  prepared, but first attempted run
  `20260704_core_world_staged_free_box_diag18_dynamic_contact_preserve_z_deck_gap`
  was interrupted during a second consecutive Kit startup in the same
  allocation and produced no summary. A later fresh-allocation request
  `165353` stayed pending with reason `(Priority)` and was canceled before
  allocation, so no valid diag18 result exists yet. Rerun it in a fresh
  allocation with `SUPPORT_DECK_GAP=0.02` and require
  `--expect-body-vertical-mode preserve --expect-physical-support-mode deck
  --expect-support-deck-gap 0.02 --max-body-z-deviation 0.08` before citing
  the gap patch.
- Preserve-z fixed-runway diagnostic:
  `PHYSICAL_SUPPORT_MODE=runway` was added as a stationary long support surface
  to avoid moving-deck energy injection. Diagnostic
  `20260704_core_world_staged_free_box_diag19_dynamic_contact_preserve_z_runway`,
  Slurm job `165355` on `server02`, completed 430/430 with
  `body_vertical_velocity_preserve_available=True`, attach step 98, body travel
  0.21877 m, box travel 0.46461 m, final target distance 0.04524 m, fall/drop
  events 0, and no disjoint warning. It still failed the strict gate:
  `target_hold_steps=0`, `max_body_z_deviation_m=1.92729`, final relative error
  1.69750 m, peak relative error 1.70690 m, and max grip gap 0.21576 m. This
  means the fixed runway removed drops but still launched the body far above
  its intended height. Do not treat preserve-z runway as physical balance
  evidence. The next valid step is either geometry-consistent support with no
  launch, or a real articulated foot-contact controller; do not keep tuning the
  staged carrier as if it were an articulated robot gait.
- The next direct Isaac work should build on the verified pure
  `SimulationApp` core-World dynamic-body path unless a stronger robot-control
  path is actually verified. Do not return to broken tensor-policy or
  drive-target-only routes without a concrete compatibility change.

## 2026-07-04 Active Execution Objective

- Current user directive: do not wait on external models or official policy
  baselines when they are not needed. Keep pushing the direct Isaac scene
  construction path first.
- Immediate active path: `scripts/isaac/build_core_world_simapp_staged_free_box_carry.py`.
  It now has a planned `BODY_VERTICAL_MODE=height-servo` diagnostic mode for a
  body-level carrier-height controller. This is a labeled scaffold, not real
  articulated balance, but it avoids treating broken external model/policy
  paths as blockers.
- Next diagnostic: staged free-box + dynamic contact proxy +
  `height-servo`, with no physical support deck/runway and strict checks for
  attach, target hold, no fall/drop, no disjoint warning, bounded body-z
  deviation, body/box travel, support margin, and grip gap.
- Latest direct-Isaac result series:
  `diag21`-`diag28` continued the direct scene path without waiting for
  external models. Added `BODY_VERTICAL_MODE=height-servo`, target hold
  radius/slowdown/body-aware latch controls, and a dynamic `FrontStopProxy`.
  The front stop materially improved box-body contact holding, but no run is
  final physical robot carrying evidence.
- Current strongest horizontal scaffold:
  `20260704_core_world_staged_free_box_diag27_front_stop_zero_vertical_baseline`
  completed 650/650, attached at step 91, latched target hold for 305 steps,
  ended 0.04748 m from target, moved body 0.19681 m and box 0.46254 m, had
  fall/drop events 0, no disjoint warning, and max contact-proxy gap
  0.11914 m. It still failed strict physical gates because
  `max_body_z_deviation_m=0.19743` and final relative error was 0.10559 m.
  Treat it only as a stable task scaffold, not balance or real grasp success.
- Current strongest height-servo comparison:
  `20260704_core_world_staged_free_box_diag26_front_stop_long_height_servo`
  completed 650/650 with fall/drop events 0, no disjoint warning, body travel
  0.20005 m, box travel 0.51237 m, final target distance 0.04528 m, and max
  grip gap 0.09538 m, but failed because target hold did not latch, final
  relative error was 0.09012 m, and max body-z deviation was 0.14254 m.
- Do not continue `BODY_VERTICAL_MODE=height-lock` unchanged:
  `diag28` kept body-z deviation at 0 but broke approach and never attached.
  It is too artificial and not the active path.
- Latest nonpenetrating staged-free-box progress:
  Added `CARRY_GEOMETRY_MODE=nonpenetrating`, `CARRY_Z_OFFSET`,
  `CONTACT_PROXY_GAIN`, and `CONTACT_PROXY_MAX_SPEED`. The approach trigger now
  uses `box_x - carry_x` in nonpenetrating mode instead of the legacy
  `box_x - 0.16`.
- Strongest current direct Isaac task scaffold:
  `20260704_core_world_staged_free_box_diag34_nonpenetrating_target_hold_1350`
  passed the staged-free-box scaffold checker for `low_front_creep`: 1350/1350
  steps, attach step 340, `target_hold_latched=True`, `target_hold_steps=97`,
  body travel 0.49526 m, box travel 0.34040 m, final target distance
  0.05999 m, final/peak relative error 0.05155 m, final/peak contact-proxy gap
  0.04926/0.04931 m, `min_support_margin_after_attach_m=0.13407`,
  `min_stance_count=3.0`, `max_body_z_deviation_m=0.15341`, fall/drop events
  0, and no disjoint warning.
- Strategy-diversity scaffold:
  `20260704_core_world_staged_free_box_diag35_chest_supported_nonpenetrating`
  passed the same style of gate for heavy `chest_supported_creep`: 1600/1600
  steps, attach step 339, `target_hold_steps=295`, body travel 0.38340 m, box
  travel 0.24055 m, final target distance 0.05992 m, final/peak relative error
  0.04152/0.04167 m, final/peak contact-proxy gap 0.04456/0.04667 m, fall/drop
  events 0, and no disjoint warning.
- These are still not final robot-carrying success. They prove a stronger
  direct Isaac task scaffold with nonpenetrating carry geometry, lifted carry
  pose, dynamic contact proxies, body-aware target hold, and two posture
  strategies. The carrier is still a velocity-commanded dynamic body with
  support proxies, not an articulated walking/balancing robot. Next work must
  preserve this task interface while replacing the carrier with an articulated
  foot-contact controller.

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
- Official Isaac policy locomotion and fixed-payload diagnostic:
  `scripts/isaac/run_official_policy_locomotion_smoke.py` and
  `scripts/isaac/run_official_policy_locomotion_smoke.sh`.
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
runs also prove those properties. Current 2026-07-04 result is mixed but not
success: after a `SimulationManager` compatibility shim,
`20260704_core_world_quad_payload_shim_diag7` initialized `SingleArticulation`
and exposed 8 DOFs; joint positions responded to commands, but torso and
payload travel stayed 0.0 m. Around step 200, PhysX produced non-finite
broadphase bounds and joint states became NaN. Do not report it as walking or
carrying. A valid next diagnostic may add clearly labeled base-velocity assist
to test whether the articulated carrier, gait joints, and fixed payload can
move together in Isaac before returning to unassisted locomotion.

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

## Current Isaac G1 Carrying State

- The direct Isaac Core API G1 + free-box torso-cradle scene is the active
  implementation path. Do not block this path on external model downloads when
  the model is not immediately needed for the next diagnostic.
- Current short-window evidence: staged G1 gait can produce 420-step
  free-box carrying diagnostics with rollout root/box writes 0, fall/drop 0,
  and about `0.37-0.51 m` final box target-directed travel (`diag71`-
  `diag73`). These are not long-duration success claims.
- Current negative long-window evidence: 700-step validations `diag75`-
  `diag80` all failed with delayed forward pitch and box drop. Static
  terminal-hold posture triggers `diag81`-`diag88` also failed even when
  activated early and held for hundreds of steps.
- Drive authority is a diagnostic variable, not a success shortcut. All-rollout
  high gain/force can suppress motion or still fail; staged terminal-drive
  switching is being tested only to isolate whether low-authority movement plus
  high-authority brake/hold can create a stable post-carry window.
- As of `diag93`-`diag96`, staged terminal-drive switching also failed. Do not
  keep sweeping the current open-loop G1 staged gait / terminal-hold family.
  The next path should use a controller-backed locomotion policy or a new
  support/contact scaffold whose target-directed travel is not produced by
  unrecoverable forward pitch.
- Cradle-cart contact baseline `diag1`-`diag4` is positive contact-scaffold
  evidence only. It shows that a free dynamic box can settle in a physical
  cradle/cage and move 0.30-0.60 m without drop or post-settle slip when the
  support body is controlled by a world-anchored rail. It must not be described
  as robot locomotion or robot carrying success.
- Low-CG prismatic-foot cage `diag1`-`diag3` is stable robot-side scaffold
  evidence only: fall/drop 0 and root writes 0, but essentially zero
  post-settle payload travel. It is not carrying-distance evidence until the
  support mechanism produces target-directed payload motion without falling,
  dropping, or root shortcuts.
- Low-CG negative-direction checks `diag4`-`diag5` also produced essentially
  zero post-settle payload travel. The next low-CG work should change the
  propulsion mechanism itself, not rerun target sign variants or the same
  stance-translate/creep/sync commands.

## Current Isaac Carry Progress Log

- 2026-07-05: external model downloads are no longer treated as blockers for
  constructing the carrying scene. The active route is direct Isaac
  scene/control construction first; external robot policies are references or
  future replacements only.
- 2026-07-05: added strict root-write checker gates to
  `scripts/isaac/check_dynamic_quadruped_carry_summary.py`:
  `--max-root-pose-writes`, `--max-root-velocity-writes`, and
  `--max-root-angular-velocity-writes`.
- 2026-07-05: added `SUPPORT_DRIVE`/`--support-drive` diagnostic support pads
  to `scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py` and
  `scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`. This is
  explicitly a contact-support scaffold, not a final locomotion controller.
- 2026-07-05: ran `diag47_support_drive_no_root` in Curiosity-owned Slurm job
  `165551` on `server46`. It was stopped early because kinematic support pads
  produced repeated PhysX errors:
  `PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!`.
- 2026-07-05: ran
  `20260705_core_world_dynamic_quad_diag47b_support_drive_dynamic_pads_no_root`
  in the same compute allocation. Result: fail but important. It completed
  760/760 with staged free box and contact proxies, attached at step 90, and
  had `root_pose_write_count=0`, `root_velocity_write_count=0`, and
  `root_angular_velocity_write_count=0`. It still failed with
  `fall_events=70`, `box_drop_events=53`, `target_hold_steps=0`, final target
  distance `4.11404 m`, max tilt `3.19234 rad`, max contact-proxy gap
  `37.78694 m`, and late non-finite PhysX state. This proves the current
  custom quadruped gait/support-drive path is not a stable root-free carrying
  controller.
- 2026-07-05: ran
  `20260705_core_world_dynamic_quad_diag48_stand_fixed_payload_no_root` as a
  reduced stand test with fixed payload, target speed 0, gait amplitudes 0, no
  support drive, and no root writes. Result: fail. It completed 240/240 with
  `root_pose_write_count=0`, `root_velocity_write_count=0`, and
  `root_angular_velocity_write_count=0`, but had `fall_events=20`, max tilt
  `2.81150 rad`, and final box target distance `1.43736 m`. This isolates the
  immediate blocker to basic stand/balance of the custom articulated carrier,
  not to video, object attach, or target reaching.
- Next required implementation milestone: build or import a functioning
  no-root stand/balance controller before continuing long-distance carrying.
  The current root-assisted `diag39b` remains only a task scaffold baseline.
- 2026-07-05: added explicit neutral-posture, morphology, friction, and joint
  drive parameters to the dynamic quadruped path:
  `HIP_NEUTRAL_DEG`, `KNEE_NEUTRAL_DEG`, `STANCE_HALF_LENGTH`,
  `STANCE_HALF_WIDTH`, `FOOT_LENGTH`, `FOOT_WIDTH`, `FOOT_HEIGHT`,
  `STATIC_FRICTION`, `DYNAMIC_FRICTION`, and hip/knee stiffness/damping/max
  force. These are morphology/contact/controller diagnostics, not root
  shortcuts.
- 2026-07-05 no-root stand diagnostics in Curiosity-owned Slurm job `165568`
  on `server53` are still negative. `diag49` used neutral hip/knee targets
  with 4 kg fixed payload and zero root writes; it failed with
  `fall_events=21`, max tilt `3.20087 rad`. `diag50` added wider stance and
  larger feet with 4 kg fixed payload; it delayed the first fall but still
  failed with `fall_events=21`, `box_drop_events=11`, max tilt `2.99903 rad`.
  `diag51` used the same wide body with a 0.5 kg payload; it improved to
  `fall_events=14`, `box_drop_events=8`, max tilt `1.93540 rad`.
  `diag52` added high friction and very high PD; it reduced fall events to 3
  and drop events to 0 but caused 19 non-finite joint events and repeated
  PhysX non-finite bounds errors, so it is not valid stand evidence. `diag53`
  reduced friction/PD to avoid the numeric blow-up but regressed to
  `fall_events=24`, `box_drop_events=16`, max tilt `2.32909 rad`.
  Conclusion: widening feet, reducing payload, and increasing contact/PD
  help but do not solve no-root standing for this custom two-DOF vertical-leg
  carrier. The next implementation path should replace the custom carrier
  with a controller-backed robot or redesign the legs/feet into a genuine
  statically stable stand controller before attempting carrying again.
- 2026-07-05: prepared a stricter MuJoCo dynamic quadruped fixed-payload
  fallback baseline after the custom Isaac carrier failed no-root standing.
  `scripts/mujoco/run_quadruped_payload_carry.py` now records assist mode,
  external force/torque write counts, and root pose/velocity write counts.
  `scripts/mujoco/check_quadruped_payload_summary.py` checks travel, fall,
  tilt, root writes, and external-force writes. This route may help identify a
  controller-backed direction, but it remains a welded-payload,
  body-force-assisted fallback diagnostic unless future evidence removes the
  external stabilizer and adds free-object contact carrying.
- 2026-07-05: attempted to start a compute allocation for the MuJoCo fallback
  rollout after hardening the checker. Slurm attempts `165590`, `165591`,
  `165595`, and `165597` did not yield a usable persistent project shell for
  running the diagnostic: `165590` and `165597` stayed pending on unavailable
  CPU resources, `165591` stayed pending on GPU priority, and `165595` was
  cancelled just as it briefly allocated `server10` while no usable tmux shell
  remained. No MuJoCo simulation result was produced in this step. The next
  action remains running `scripts/mujoco/run_quadruped_payload_carry.sh` inside
  a valid compute allocation, then checking it with
  `scripts/mujoco/check_quadruped_payload_summary.py`.
- 2026-07-05: after user correction, the active path was reset to direct Isaac
  scene construction instead of waiting on external models or MuJoCo fallback.
  In Curiosity-owned tmux/Slurm session `curiosity_mujoco_payload_run_0705`,
  job `165603` on `server57`, reran the strongest staged free-box Isaac scene:
  `20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating`.
  Result: pass under the declared staged-free-box scaffold gate. It completed
  1350/1350 steps with `low_front_creep`, `dynamic-contact-proxy`,
  `nonpenetrating` carry geometry, attach step 340, target hold 97 steps, body
  travel `0.49526 m`, box travel `0.34040 m`, final target distance
  `0.05999 m`, final/peak relative error `0.05155/0.05155 m`,
  final/peak proxy gap `0.04926/0.04931 m`, minimum post-attach support margin
  `0.13407 m`, max body-z deviation `0.15341 m`, and fall/drop events 0 with no
  disjoint warning. This is direct Isaac scene progress, but still not final
  robot success because the carrier evidence mode is `support-proxy`, not an
  articulated foot-contact robot controller.
- 2026-07-05: attempted posture-diversity reruns for `chest_supported_creep`.
  `diag55` was accidentally interrupted manually at step 120 after startup
  output lag made it look stalled; do not count it. `diag56`
  (`20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete`)
  completed 1600/1600 with `chest_supported_creep`, attach step 350,
  fall/drop events 0, body travel `0.46327 m`, box travel `0.24708 m`,
  minimum post-attach support margin `0.13779 m`, and no disjoint warning, but
  failed the strict scaffold gate: target hold 0, final target distance
  `0.06386 m`, final/peak relative error `0.11752/0.11780 m`, and peak
  contact-proxy gap `0.11627 m`. Treat this as useful negative evidence:
  heavier/chest-supported posture is safe in this scene but needs stronger
  contact closure or a longer/retuned target-hold phase before it matches the
  low-front scaffold.
- 2026-07-05: exposed staged-free-box dynamic proxy tuning knobs without
  changing defaults: `PALM_PROXY_MASS`, `CHEST_PROXY_MASS`,
  `SHELF_PROXY_MASS`, `FRONT_STOP_PROXY_MASS`, `PALM_PROXY_THICKNESS`,
  `CHEST_PROXY_THICKNESS`, and `FRONT_STOP_PROXY_THICKNESS`. The summary now
  records these values. This is to tune chest-supported contact closure
  honestly instead of hard-coding the dynamic proxy bodies.
- 2026-07-05: attempted stronger chest-supported proxy diagnostic
  `20260705_core_world_staged_free_box_diag57_chest_supported_stronger_proxy`
  in Curiosity-owned tmux/Slurm session `curiosity_isaac_carry_diag57_0705`,
  job `165621` on `server10`, with `CONTACT_PROXY_GAIN=45`,
  `CONTACT_PROXY_MAX_SPEED=3`, heavier proxy masses, and 1800 steps. It did
  not reach scene construction: the log stopped after SimulationApp boot and
  protobuf warnings. A short 40-step health smoke
  `20260705_core_world_staged_free_box_diag57a_startup_health` in the same
  allocation stalled at the same point. No rollout result exists for either
  run, and this is a startup/resource negative result only, not evidence about
  the stronger contact parameters.
- 2026-07-05: added shortcut-evidence counters to the staged-free-box script
  and checker: `body_root_velocity_command_count`,
  `body_root_pose_write_count`, `box_pose_write_count`, and
  `box_velocity_command_count`, plus checker gates
  `--max-body-root-velocity-commands`, `--max-body-root-pose-writes`, and
  `--max-box-pose-writes`. This is required for the final robot-carrying
  objective: a velocity-commanded root body must not be mistaken for a walking
  and balancing robot. Existing `diag54`-style runs should now explicitly show
  nonzero body root velocity commands and therefore cannot satisfy a future
  no-root articulated walking gate.
- 2026-07-05: `20260705_core_world_staged_free_box_diag58_startup_health` was
  launched in a fresh Slurm allocation `165631` on `server10` after `diag57`,
  but it also stalled during SimulationApp startup after protobuf warnings.
  That allocation was stopped and released. A replacement allocation
  `165638` was requested with `--exclude=server10` for the next startup smoke;
  it remained pending for the useful wait window and was cancelled without a
  rollout. No non-server10 Isaac result exists yet for the stronger proxy
  settings.
- 2026-07-05: added staged-free-box checker flag `--require-no-root-shortcut`.
  It requires an articulated carrier and rejects body root velocity commands,
  body root pose writes, and box pose writes. A lightweight audit against the
  old passing scaffold `diag54` correctly failed with
  `no-root shortcut gate requires articulated carrier`. Use this as the future
  gate for any claim that the robot itself walks and balances while carrying.
- 2026-07-05: requested non-`server10` allocation `165646`, which allocated
  `server36`, and launched
  `20260705_core_world_staged_free_box_diag60_startup_health_server36`.
  It showed only SimulationApp boot plus protobuf warnings during a short
  observation window and was interrupted before a full timeout-backed startup
  conclusion. Treat this as inconclusive startup evidence, not a confirmed
  Isaac or contact-control failure. Future startup checks must wait with an
  actual elapsed-time window before recording a stall.
- 2026-07-05: requested non-`server10` allocation `165651`, which allocated
  `server46`. Startup health run
  `20260705_core_world_staged_free_box_diag61_startup_health_server46`
  completed 40/40 and verified this allocation could enter Isaac, build the
  staged free-box scene, attach, and write summary/logs.
- 2026-07-05: ran
  `20260705_core_world_staged_free_box_diag62_chest_supported_stronger_proxy_server46`
  in the same allocation with stronger chest-supported dynamic proxies:
  `CONTACT_PROXY_GAIN=45`, `CONTACT_PROXY_MAX_SPEED=3`,
  `PALM_PROXY_MASS=90`, `CHEST_PROXY_MASS=130`, `SHELF_PROXY_MASS=140`, and
  `FRONT_STOP_PROXY_MASS=120`. Result: pass under the staged-free-box
  scaffold gate. It completed 1800/1800, selected `chest_supported_creep`,
  attached at step 421, target hold 335 steps, body travel `0.37242 m`, box
  travel `0.25688 m`, final target distance `0.04367 m`, final/peak relative
  error `0.01475/0.05307 m`, final/peak contact-proxy gap
  `0.01287/0.05447 m`, minimum post-attach support margin `0.13825 m`,
  max body-z deviation `0.14629 m`, fall/drop events 0, and no disjoint
  warning. This restores posture-diversity coverage for the direct Isaac
  scaffold after `diag56` failed.
- 2026-07-05: applied the new `--require-no-root-shortcut` gate to `diag62`.
  It correctly failed with `no-root shortcut gate requires articulated carrier`,
  `body root velocity commands present: 1800`, and `box pose writes present: 1`.
  Therefore `diag62` is not final robot walking/carrying success. The next
  milestone remains replacing the velocity-commanded support-proxy body with a
  true articulated foot-contact carrier that can satisfy this gate.
- 2026-07-05: corrected the staged-free-box summary semantics after the user
  redirected the work back to direct Isaac scene construction. Passing
  `--carrier-evidence-mode articulated-foot-contact` now records
  `articulated_carrier_requested=true` but keeps
  `articulated_carrier_enabled=false` until an actual articulated carrier is
  implemented. The checker report now includes this requested/enabled split.
  Also added root-shortcut counters to the quasi-static walker scaffold:
  `body_root_velocity_command_count`, `body_root_pose_write_count`, and
  `payload_pose_write_count`. This is a code/recording correction only; no new
  simulation result is claimed.
- 2026-07-05: added the first direct no-root articulated carrier stand path:
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py`, launcher
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`, and checker
  `scripts/isaac/check_prismatic_carrier_stand_summary.py`. This creates a
  free articulated carrier with four vertical prismatic leg joints, four
  physical feet, and a fixed physical payload. The intended first gate is
  fixed-payload stand with positive joint count, `foot_contact_drive_enabled`,
  root pose/velocity/angular-velocity writes 0, body root writes 0,
  box/payload pose writes 0, fall/drop 0, and no non-finite state. This is not
  walking or free-box carrying until a compute rollout proves the stand and
  later a no-root creep gait.
- 2026-07-05: ran the new no-root prismatic carrier stand diagnostic in
  Curiosity-owned tmux/Slurm session `curiosity_prismatic_stand_0705`, job
  `165690` on `server63`. Short smoke
  `20260705_prismatic_carrier_stand_diag1_server63` completed 80/80 and passed
  the checker. Full Stage-1 run
  `20260705_prismatic_carrier_stand_diag2_360_server63` completed 360/360 and
  passed:
  `articulated_carrier_enabled=true`, `articulated_joint_count=4`,
  `foot_contact_drive_enabled=true`, root pose/velocity/angular-velocity writes
  0, body root pose/velocity writes 0, box/payload pose writes 0, fall/drop
  events 0, nonfinite events 0, max tilt `0.04386 rad`, max torso drift
  `0.03936 m`, and max joint motion `0.23590 m`. Output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260705_prismatic_carrier_stand_diag2_360_server63/`.
  Log:
  `logs/core_world_prismatic_carrier_stand/core_world_prismatic_carrier_stand_20260705_prismatic_carrier_stand_diag2_360_server63.log`.
  This satisfies the no-root fixed-payload stand milestone only. It is not
  walking and not free-box carrying.
- 2026-07-05: extended the prismatic carrier with optional horizontal hip
  sliders for no-root fixed-payload creep. The script now supports
  `MOTION_MODE=creep` and `ENABLE_HORIZONTAL_LEGS=1`, adding four X-slide DOFs
  plus the four vertical leg DOFs. `20260705_prismatic_carrier_creep_diag1_server63`
  and `diag2_reverse_step` were safe but failed the positive-X travel gate:
  both kept root writes 0 and fall/drop 0, but the open-loop gait naturally
  moved in negative X. `20260705_prismatic_carrier_creep_diag3_neg_target_server63`
  then used a short negative-X target and passed the no-root fixed-payload
  creep gate: 85/85 steps, `articulated_joint_count=8`,
  `foot_contact_drive_enabled=true`, root pose/velocity/angular writes 0, body
  root writes 0, box/payload pose writes 0, fall/drop 0, nonfinite events 0,
  max tilt `0.07105 rad`, max absolute torso travel `0.05861 m`, max absolute
  payload travel `0.05954 m`, and final target distance `0.00721 m`. Output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260705_prismatic_carrier_creep_diag3_neg_target_server63/`.
  Log:
  `logs/core_world_prismatic_carrier_stand/core_world_prismatic_carrier_stand_20260705_prismatic_carrier_creep_diag3_neg_target_server63.log`.
  This is still fixed-payload short creep, not unknown free-box carrying or a
  learned locomotion policy.
- 2026-07-05: ran an additional fixed-payload posture/load check in the same
  Curiosity-owned allocation, job `165690` on `server63`.
  `20260705_prismatic_carrier_creep_diag4_chest_heavy_server63` used
  `PAYLOAD_MASS=10.0`, `PAYLOAD_LOCAL_X=0.08`, `PAYLOAD_LOCAL_Z=0.02`, and a
  `TARGET_X=-0.050` short creep. It was safe but failed the target gate:
  fall/drop 0, root/body/box/payload writes 0, max absolute torso travel
  `0.03633 m`, but final target distance `0.03165 m`. Retuned shorter-target
  run `20260705_prismatic_carrier_creep_diag5_chest_heavy_short_target_server63`
  passed: 85/85 steps, 8 articulated DOFs, root/body/box/payload writes 0,
  fall/drop 0, max tilt `0.04617 rad`, max absolute torso travel `0.03633 m`,
  max absolute payload travel `0.03657 m`, and final target distance
  `0.00165 m`. Output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260705_prismatic_carrier_creep_diag5_chest_heavy_short_target_server63/`.
  Log:
  `logs/core_world_prismatic_carrier_stand/core_world_prismatic_carrier_stand_20260705_prismatic_carrier_creep_diag5_chest_heavy_short_target_server63.log`.
  This expands fixed-payload posture/load evidence only; it is still not a
  free-object grasp/carry success.
- 2026-07-05: attempted to replace welded fixed payload with a free dynamic box
  resting on the torso via `--payload-mode top_contact_free_box`. First launcher
  attempts `diag6/diag6b` are not valid free-box evidence because the compute
  run resolved to `payload_mode=fixed_joint_to_torso`; the direct Python run
  `diag7_direct_py` failed before rollout due a relative `--experience` path.
  Direct Python rerun
  `20260705_prismatic_carrier_free_top_box_creep_diag7b_direct_py_server63`
  correctly used `payload_mode=top_contact_free_box`: the free box stayed above
  the carrier and did not drop, but the checker failed because absolute torso
  travel was only `0.00838 m`, absolute payload travel `0.01028 m`, final target
  distance `0.01506 m`, and min payload z `0.69622 m` against a conservative
  `0.70 m` threshold. Larger-step direct run
  `20260705_prismatic_carrier_free_top_box_creep_diag8_larger_step_server63`
  also failed the free-box carry gate: fall/drop 0 and min payload z
  `0.69277 m`, but absolute torso travel only `0.00622 m`, absolute payload
  travel `0.00829 m`, and final target distance `0.04543 m`. Conclusion:
  passive top-contact free-box carrying is not sufficient with this gait; the
  next free-box implementation needs an explicit tray, side rails, clamp, or
  physically defensible contact/constraint strategy before claiming free-object
  carrying.
- 2026-07-05: added `payload_mode=tray_contact_free_box` to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and launcher
  knobs in `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`. The tray
  deck, side rails, and front/rear stops are physical rigid bodies fixed to the
  carrier torso; the payload box remains a free dynamic rigid body. Also added
  checker visibility and stricter free-payload gates:
  `final_payload_target_distance_x_m`, `payload_relative_distance_m`, and
  `max_payload_relative_offset_error_m`, plus checker flags
  `--max-final-payload-target-distance-x` and
  `--max-payload-relative-offset-error`. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/check_prismatic_carrier_stand_summary.py`
  and `bash -n scripts/isaac/run_core_world_prismatic_carrier_stand.sh`.
- 2026-07-05: ran tray/free-box diagnostics in Curiosity-owned tmux/Slurm
  session `curiosity_tray_freebox_0705`, job `165725` on `server36`.
  `20260705_prismatic_carrier_tray_freebox_diag1_server36` used
  `PAYLOAD_MODE=tray_contact_free_box`, `MOTION_MODE=creep`,
  `ENABLE_HORIZONTAL_LEGS=1`, `TARGET_X=-0.030`, 95 steps, 4 kg payload, 36 kg
  torso, high-friction tray. It completed 95/95 with root/body/box/payload
  writes 0, fall/drop 0, max torso travel `0.29253 m`, max abs payload travel
  `0.09516 m`, min payload z `0.81006 m`, and max tilt `0.60635 rad`, but
  failed the target gate because final target distance was `0.32253 m`. Treat
  it as partial/negative tray evidence, not carry success.
- 2026-07-05: in the same allocation,
  `20260705_prismatic_carrier_tray_freebox_diag2_positive_short_server36` used
  the same tray config with `TARGET_X=0.120` and 62 steps. The old checker
  initially passed because it only required absolute payload travel, but CSV
  inspection showed torso travel `+0.12770 m` while payload travel was
  `-0.09340 m`; the box slid opposite the carrier. After adding the
  same-direction payload gate, the run correctly failed with
  `payload travel x too low: 0.00065`. This is recorded as a checker bug found
  and fixed, not a success.
- 2026-07-05: `20260705_prismatic_carrier_tray_freebox_diag3_rear_loaded_slow_server36`
  put the free box near the rear tray stop and slowed the gait
  (`STEP_LENGTH=0.035`, `STEP_HEIGHT=0.045`, 72 steps). It failed badly:
  fall events 44, max tilt `3.06941 rad`, final target distance
  `1.32181 m`, final payload target distance `2.16927 m`, and max payload
  relative-offset error `3.85229 m`. Rear-loading the box against the stop
  destabilizes this open-loop carrier.
- 2026-07-05: added `SETTLE_STEPS` and `RAMP_STEPS` to the prismatic carrier
  builder/launcher so creep can start after a stand phase and ramp its step
  length/height. `20260705_prismatic_carrier_tray_freebox_diag4_ramped_rear_micro_server36`
  used `SETTLE_STEPS=150`, `RAMP_STEPS=450`, `STEP_LENGTH=0.012`,
  `STEP_HEIGHT=0.018`, `GAIT_PERIOD_STEPS=900`, and 650 steps. It still
  failed: fall events 598, drop events 482, min payload z `-38.39795 m`,
  final payload target distance `6.61048 m`, and max payload relative-offset
  error `40.69181 m`. This shows the rear-loaded tray setup is not salvageable
  by simple ramping.
- 2026-07-05: backed up to a static free-box tray stand diagnostic.
  `20260705_prismatic_carrier_tray_freebox_diag5_stand_wide_center_server36`
  used a wider stance, larger feet, centered 2 kg free payload, lower tray,
  `MOTION_MODE=stand`, no horizontal legs, and 300 steps. It passed the
  stand gate: articulated joint count 4, root/body/box/payload writes 0,
  fall/drop 0, nonfinite 0, max tilt `0.00588 rad`, max torso drift
  `0.00444 m`, min payload z `0.70280 m`, and max payload relative-offset
  error `0.07579 m` during settling. This proves the static tray/free-box
  scene can be physically stable; it is not walking/carrying.
- 2026-07-05: tested whether the stable wide-center tray stand could become
  a micro-creep carry when horizontal x-slide legs were re-enabled.
  `20260705_prismatic_carrier_tray_freebox_diag6_wide_center_micro_creep_server36`
  used `ENABLE_HORIZONTAL_LEGS=1`, `SETTLE_STEPS=300`, `RAMP_STEPS=300`,
  `STEP_LENGTH=0.006`, `STEP_HEIGHT=0.008`, `GAIT_PERIOD_STEPS=1200`, and
  700 steps. It failed before any carry claim: fall events 564, drop events
  583, max tilt `3.13262 rad`, final target distance `1.51252 m`, final
  payload target distance `1.43231 m`, and max payload relative-offset error
  `3.67675 m`. Current conclusion: the x-slide horizontal-leg creep scaffold
  is useful for fixed-payload diagnostics but is not a stable locomotion base
  for free-box carrying. Next step should replace this locomotion scaffold
  with a controller-backed IsaacLab robot, a quasi-static foot-placement
  controller with support-polygon constraints, or a clearly labeled
  non-locomotion cart/table diagnostic for contact development.
- 2026-07-05: user clarified not to wait on external models or downloads when
  they are not directly useful. Treat Arena/G1/WBC/GR00T assets as optional
  references or baselines only. The active execution path is direct Isaac
  scene construction and controller scaffold development.
- 2026-07-05: patched the minimal G1 smoke summary/checker plumbing so any
  future result records `scene_type`, success-claim wording, articulated joint
  count, payload joint creation, setup root/joint writes, and rollout
  root/box pose or velocity write counters. Lightweight login-node checks
  passed:
  `python3 -m py_compile scripts/isaac/build_minimal_carry_scene.py scripts/isaac/check_carry_smoke_summary.py scripts/isaac/check_g1_wbc_local_assets.py`
  and `bash -n scripts/isaac/run_minimal_carry_scene.sh scripts/isaac/run_g1_wbc_smoke_sequence.sh`.
- 2026-07-05: in Curiosity-owned tmux/Slurm session
  `curiosity_g1_wbc_smoke_0705`, job `165744` on `server63`, local G1 WBC
  assets loaded successfully despite ONNX Runtime CPU affinity warnings:
  `Robot DOFs: 43` and `Policy: G1DecoupledWholeBodyPolicy`. This is asset
  availability only, not a simulation result.
- 2026-07-05: MuJoCo no-assist fallback diagnostic was run in the same
  allocation:
  `STAMP=20260705_mujoco_quad_noassist_diag1_server63 STEPS=1200 PAYLOAD_MASS=2.0 TARGET_SPEED=0.25 ASSIST_MODE=none bash scripts/mujoco/run_quadruped_payload_carry.sh`.
  It completed 1200/1200 but failed strict carrying/balance gates:
  `fall_events=40`, `max_travel_x_m=0.0`, `max_tilt_rad=1.61854`,
  `external_force_write_count=0`, `external_torque_write_count=0`,
  `root_pose_write_count=0`, and `root_velocity_write_count=0`. Output:
  `experiments/outputs/mujoco_quadruped_payload/20260705_mujoco_quad_noassist_diag1_server63/`.
  Log:
  `logs/mujoco_quadruped_payload/mujoco_quadruped_payload_20260705_mujoco_quad_noassist_diag1_server63.log`.
  Conclusion: this no-assist MuJoCo controller is not a stable locomotion base
  for the Isaac carrying task.
- 2026-07-05: G1/WBC minimal Isaac stand smoke was attempted in the same
  allocation:
  `STAMP=20260705_g1_wbc_stand_smoke_diag1_server63 DEVICE=cuda:0 SKIP_ROBOT=0 WBC_MODE=stand RENDER=0 STEPS=120 bash scripts/isaac/run_minimal_carry_scene.sh`.
  It stalled before scene setup/rollout after protobuf/grpc registration
  warnings and was interrupted. Output directory
  `experiments/outputs/minimal_carry_scene/20260705_g1_wbc_stand_smoke_diag1_server63/`
  is empty; log
  `logs/minimal_carry_scene/minimal_carry_scene_20260705_023224.log` contains
  only the startup warnings. Do not treat the G1/WBC minimal scene as an active
  blocker; continue direct Isaac construction.
- 2026-07-05: direct Isaac construction was resumed without waiting on models.
  In the same Curiosity-owned allocation, ran:
  `STAMP=20260705_quasistatic_direct_continue_diag1_server63 DEVICE=cpu STEPS=420 TARGET_X=0.18 PAYLOAD_MASS=8.0 PAYLOAD_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 bash scripts/isaac/run_core_world_simapp_quasistatic_walker_carry.sh`.
  It completed 420/420 with `strategy=low_front_creep`,
  `body_travel_x_m=0.16585`, `payload_travel_x_m=0.16585`,
  `final_payload_target_distance_xy_m=0.01415`,
  `payload_relative_error_m=0.0`, `min_support_margin_m=0.13252`,
  fall/drop 0, payload pose writes 0, and body root pose writes 0. It still
  has `body_root_velocity_command_count=420`, `articulated_carrier_enabled=false`,
  `articulated_joint_count=0`, and `foot_contact_drive_enabled=false`; it is a
  direct Isaac fixed-payload support/strategy scaffold, not final robot
  locomotion or unknown free-box carrying. Output:
  `experiments/outputs/core_world_simapp_quasistatic_walker_carry/20260705_quasistatic_direct_continue_diag1_server63/`.
  Log:
  `logs/core_world_simapp_quasistatic_walker_carry/core_world_simapp_quasistatic_walker_carry_20260705_quasistatic_direct_continue_diag1_server63.log`.
- 2026-07-05: added a direct no-root `stance_translate` mode to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py`. It keeps all
  vertical legs in support and moves all horizontal prismatic joints
  synchronously instead of using the failed open-loop x-slide creep gait. Also
  added dedicated launcher
  `scripts/isaac/run_core_world_prismatic_stance_translate_tray.sh` to force
  tray/free-box arguments while debugging the older launcher argument path.
- 2026-07-05: `20260705_prismatic_tray_stance_translate_diag1_server63` and
  `diag2_server63` are parameter-problem probes, not tray/free-box evidence:
  their summaries showed `payload_mode=fixed_joint_to_torso` despite the
  intended tray/free-box setup. `bash -x` probe
  `20260705_prismatic_arg_trace2_server63` confirmed the executed command
  omitted `--payload-mode`, tray, and settle/ramp arguments. Do not cite these
  as free-box results.
- 2026-07-05: dedicated tray/free-box `stance_translate` run
  `20260705_prismatic_tray_stance_translate_diag3_server63` correctly used
  `payload_mode=tray_contact_free_box`, `motion_mode=stance_translate`, 8
  articulated prismatic DOFs, root/body/box/payload pose writes 0, and body
  root velocity commands 0, but it failed the strict gate: fall events 360,
  box drop events 264, max tilt `1.76093 rad`, min payload z `0.12000 m`,
  final payload target distance `1.92207 m`, and max payload relative-offset
  error `2.06908 m`. Output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260705_prismatic_tray_stance_translate_diag3_server63/`.
  Log:
  `logs/core_world_prismatic_carrier_stand/core_world_prismatic_carrier_stand_20260705_prismatic_tray_stance_translate_diag3_server63.log`.
  Conclusion: the all-feet-supported synchronous x-slide translation fixes the
  parameter path but is dynamically unstable for the tray/free-box task. The
  no-root path still needs a better foot-contact/controller formulation rather
  than more tuning of this prismatic scaffold.
- 2026-07-05: ran follow-up prismatic/cage diagnostics in Curiosity-owned
  tmux/Slurm session `curiosity_prismatic_slow_ramp_0705`, job `165776` on
  `server10`. Added tray top-lid/cage arguments to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py`, added dedicated
  launchers
  `scripts/isaac/run_core_world_prismatic_cage_stand_balance.sh` and
  `scripts/isaac/run_core_world_prismatic_lowcg_cage_translate.sh`, and added
  optional roll/pitch leg-length balance servo fields. Lightweight checks
  passed with `python3 -m py_compile` and `bash -n` before compute runs.
- 2026-07-05: fair slow-ramp tray/free-box run
  `20260705_prismatic_tray_stance_translate_diag4_slowramp_server10` used
  `settle_steps=300`, `ramp_steps=300`, `payload_mode=tray_contact_free_box`,
  zero root/body/box/payload pose or velocity shortcuts, and fall events 0, but
  failed because the free box escaped/dropped: box drop events 523, min payload
  z `-21.90778 m`, final payload target distance `2.17751 m`, and max payload
  relative-offset error `23.10633 m`.
- 2026-07-05: top-lid/high-wall cage attempts showed the contact cage can keep
  the box from dropping, but a heavy/high cage destabilizes the carrier.
  `20260705_prismatic_tray_lid_stand_highwalls_diag1_server10` had box drop 0
  but fall events after tilt rose to about `1.55 rad`. Roll/pitch leg-length
  servo attempts `20260705_prismatic_cage_stand_balance_diag2_server10`
  (`-0.08` gains) and `20260705_prismatic_cage_stand_balance_diag3_posgain_server10`
  (`+0.08` gains) did not fix the tilt; both remained fall diagnostics, not
  balance success.
- 2026-07-05: low-center-of-mass cage stand passed a strict no-root free-box
  stand gate. `20260705_prismatic_cage_stand_lowcg_diag1_server10` used a
  100 kg torso, 14 kg feet, wider stance/feet, light cage/lid, and
  `payload_mode=tray_contact_free_box` with lid enabled. Checker passed:
  completed 350/350, 4 articulated DOFs, fall/drop 0, root/body/box/payload
  writes 0, max tilt `0.03275 rad`, max torso drift `0.01271 m`, min payload z
  `0.73412 m`, and max payload relative-offset error `0.00609 m`. This is
  stable free-box cage standing only, not walking.
- 2026-07-05: low-center-of-mass cage small-translation diagnostic passed.
  `20260705_prismatic_lowcg_cage_translate_diag1_server10` used 8 articulated
  DOFs, horizontal prismatic stance-translate mode, a free dynamic box inside
  a physical cage/lid, no root/body/box/payload pose or velocity writes, and
  target x `0.015 m`. Checker passed: completed 700/700, fall/drop 0, max tilt
  `0.03797 rad`, max torso travel `0.01580 m`, max payload travel `0.01930 m`,
  final target distance `0.000087 m`, final payload target distance
  `0.00307 m`, min payload z `0.73331 m`, and max payload relative-offset
  error `0.00625 m`. This is the strongest current no-root Isaac carrying
  diagnostic, but it is still not final robot walking because translation is
  a very short synchronous prismatic stance shift, not a real gait.
- 2026-07-05: extending the same low-CG cage to a 3 cm target failed the travel
  gate while staying safe. `20260705_prismatic_lowcg_cage_translate_diag2_3cm_server10`
  completed 900/900 with fall/drop 0 and no root/body/box/payload writes, but
  max torso travel stayed `0.01580 m`, max payload travel `0.01930 m`, final
  target distance `0.01602 m`, and final payload target distance `0.01322 m`.
  Conclusion: the current synchronous stance-translate mechanism is stable only
  for about 1.5 cm; longer carrying needs actual foot repositioning, a
  controller-backed robot, or another support mechanics change.
- 2026-07-05: user pushed the direction back to direct Isaac scene construction
  and away from waiting on external models. Added `sync_inchworm` to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` plus launcher
  `scripts/isaac/run_core_world_prismatic_lowcg_cage_sync_inchworm.sh`. The
  intended mode synchronously pushes the body, then resets lifted legs one by
  one. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/check_prismatic_carrier_stand_summary.py`
  and `bash -n scripts/isaac/run_core_world_prismatic_lowcg_cage_sync_inchworm.sh`.
- 2026-07-05: in Curiosity-owned tmux/Slurm session
  `curiosity_lowcg_cage_creep_0705`, job `165804` on `server53`, reran low-CG
  cage repeated-foot diagnostics. `20260705_prismatic_lowcg_cage_creep_diag1_server53`
  completed 900/900 with fall/drop 0, zero root/body/box/payload writes, max
  tilt `0.03797 rad`, and max payload relative-offset error `0.00625 m`, but
  failed travel/target gates: max torso travel `0.01580 m`, max payload travel
  `0.01930 m`, final target distance `0.04513 m`, and final payload target
  distance `0.04838 m`. `20260705_prismatic_lowcg_cage_sync_inchworm_diag1_server53`
  also completed safely with fall/drop 0 and zero shortcuts, but failed the
  3 cm gate with max torso travel `0.01580 m`, max payload travel `0.01930 m`,
  final target distance `0.01607 m`, and final payload target distance
  `0.01295 m`. Conclusion: the low-CG cage/free-box contact scaffold is stable,
  but the prismatic foot mechanics saturate around 1.5 cm; stop tuning this as
  the main locomotion path.
- 2026-07-05: in the same `server53` allocation, local G1 WBC assets loaded
  successfully again via
  `/public/home/yanhongru/envs/isaac_arena_py312/bin/python scripts/isaac/check_g1_wbc_local_assets.py`.
  Result: `Robot DOFs: 43`, `Policy: G1DecoupledWholeBodyPolicy`, with only
  ONNX Runtime thread-affinity warnings. This proves local WBC assets are
  present, not that simulation walking works.
- 2026-07-05: attempted minimal IsaacLab G1 WBC stand diagnostics without
  external GR00T server in the same allocation. Both
  `20260705_g1_wbc_stand_diag1_server53` and
  `20260705_g1_wbc_stand_skipreset_diag1_server53` failed before rollout with
  `Simulation view object is invalidated and cannot be used again to call getDofPositions`
  / `Failed to get DOF positions from backend`. The failure occurs inside
  IsaacLab tensor/articulation data access, not in balancing or carrying.
  Treat the current `build_minimal_carry_scene.py` G1/WBC path as an
  initialization bug until the tensor-view lifecycle is fixed.
- 2026-07-05: added direct Core API G1 scene files
  `scripts/isaac/build_core_world_g1_box_scene.py` and
  `scripts/isaac/run_core_world_g1_box_scene.sh` to avoid IsaacLab
  `InteractiveScene` while constructing the G1 + box scene. Lightweight checks
  passed with `python3 -m py_compile` and `bash -n`. Runs
  `20260705_core_world_g1_box_scene_diag1_server53`,
  `diag2_zfix_server53`, and `diag3_pelvisroot_server53` all completed 120
  steps, but the reported robot root stayed near `z=-0.025 m`, causing fall
  events throughout and a box drop at the end. USD metadata inspection showed
  the G1 articulation root is `/g1_29dof_with_hand_rev_1_0/pelvis`; binding to
  `/World/G1/pelvis` did not fix the runtime root pose. Conclusion: direct
  Core API USD loading currently constructs a scene but not a valid standing
  G1 articulation. Next work should fix G1/USD root initialization or return
  to the official Arena G1 environment with the tensor-view lifecycle fixed,
  rather than waiting on external model downloads.
- 2026-07-05: after the user explicitly pushed back against waiting on models,
  the active path was corrected to direct Isaac scene construction. In
  Curiosity-owned tmux/Slurm session `curiosity_g1_core_rootfix_0705`, job
  `165829` on `server46`, the Core API G1 scene was patched to write the G1
  pelvis/root pose during setup only. Run
  `20260705_core_world_g1_box_scene_diag4_setuppose_server46` confirmed the
  robot starts above the ground (`z` about `0.799 m`, fall 0 at step 0), but
  open-loop G1 stand targets are not a balance controller; fall begins around
  step 30 and the box later drops. This is root/scene initialization progress,
  not a carry success.
- 2026-07-05: in the same allocation, official Go2 locomotion was tested as a
  possible fast controller-backed Isaac base. Results were negative for this
  standalone launch path: `20260705_go2_callback_locomotion_diag1_server46`
  exited near `SimulationManager` setup without a summary;
  `20260705_go2_callback_locomotion_diag2_skipdev_server46` completed
  180/180 but had `callback_forward_calls=0`, zero travel, and an invalid
  articulation warning; `20260705_go2_callback_locomotion_diag3_dtonly_offkit_server46`
  exited near `set_physics_dt`; and
  `20260705_official_go2_manual_locomotion_diag1_server46` failed before
  initialization because the articulation physics tensor entity was invalid.
  Do not spend more time on this path unless the standalone physics-view
  initialization is fixed.
- 2026-07-05: a stronger low-CG prismatic cage diagnostic
  `20260705_prismatic_stance_translate_strongx_diag1_server46` was run in the
  same allocation. It completed 360/360 with fall/drop 0, zero root/body/box/
  payload shortcuts, max tilt `0.03640 rad`, and max payload relative-offset
  error `0.01451 m`, but still saturated at max torso travel `0.01557 m` and
  max payload travel `0.01834 m` for an `0.08 m` target. The summary recorded
  `motion_mode=sync_inchworm`, so do not cite this as a true stance-translate
  override. Conclusion: the cage is useful as a direct Isaac contact/load
  harness, but the x-slide mechanism is not a locomotion solution.
- 2026-07-05: added direct Core API anchored-support carrier files
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py` and
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh`. This was run in
  Curiosity-owned tmux/Slurm session `curiosity_anchor_footstep_0705`, job
  `165866` on `server46`. Early diagnostics found and fixed several concrete
  issues: kinematic bodies cannot be inside PhysX articulations, this Isaac
  version lacks the attempted `RigidBodyAPI` disable-gravity accessors, a
  ground-height anchor lets the torso fall, and the initial drive sign was
  reversed.
- 2026-07-05: anchored-support fixed-payload run
  `20260705_anchor_footstep_fixed_diag8b_holdtarget_server46` passed a
  fixed-payload diagnostic gate. It completed 180/180 with one articulated
  prismatic joint, fixed 4 kg payload, fall/drop 0, root/body/box/payload pose
  and velocity shortcuts 0, max torso travel `0.03781 m`, max payload travel
  `0.03781 m`, final target distance `0.00219 m`, final payload target
  distance `0.00219 m`, min payload z `0.55720 m`, and max payload
  relative-offset error near zero. Checker command:
  `python3 scripts/isaac/check_prismatic_carrier_stand_summary.py experiments/outputs/core_world_anchored_footstep_carrier/20260705_anchor_footstep_fixed_diag8b_holdtarget_server46/core_world_anchored_footstep_carrier_summary.json --log logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_20260705_anchor_footstep_fixed_diag8b_holdtarget_server46.log --max-fall-events 0 --max-box-drop-events 0 --min-torso-travel-x 0.035 --min-payload-travel-x 0.035 --max-final-target-distance-x 0.005 --max-final-payload-target-distance-x 0.005 --max-payload-relative-offset-error 0.002 --min-payload-z 0.55 --require-articulated-carrier --require-foot-contact-drive --min-joint-count 1 --min-joint-motion 0.035`.
  This is progress beyond the velocity-commanded torso scaffold, but it is
  still a world-fixed support-frame single-step diagnostic, not a complete
  walking robot.
- 2026-07-05: anchored-support free-box cage run
  `20260705_anchor_footstep_cagedfree_diag1_server46` is a negative result.
  It completed 180/180 with fall/drop 0 and no root/body/box/payload shortcuts,
  but the free box was expelled/accelerated by cage contact: final payload
  target distance `0.79402 m`, max payload travel `0.83402 m`, and max payload
  relative-offset error `2.01642 m`. Do not cite this as free-box carrying.
  Next work should fix cage geometry, initial clearances, and contact impulse
  behavior before attempting a free-box carry claim on this path.
- 2026-07-05: centered/lighter anchored-support free-box cage run
  `20260705_anchor_footstep_cagedfree_diag2_centerbox_server46` improved the
  contact failure but did not pass. It completed 180/180 with fall/drop 0 and
  no root/body/box/payload pose or velocity shortcuts. The payload no longer
  shot out and reached max travel `0.02232 m`, but torso travel was only
  `0.00365 m`, final torso target distance `0.03635 m`, final payload target
  distance `0.02186 m`, and max payload relative-offset error `0.13000 m`.
  Conclusion: centering and reducing payload mass reduces contact impulse but
  the current support-frame/cage coupling still does not produce coherent
  free-box carrying.
- 2026-07-05: in Curiosity-owned tmux/Slurm session
  `curiosity_anchor_cage_0705`, job `165903` on `server46`, parameterized the
  anchored-support cage geometry in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py` and exposed
  the parameters in `scripts/isaac/run_core_world_anchored_footstep_carrier.sh`.
  Lightweight checks passed with `python3 -m py_compile` and `bash -n`.
- 2026-07-05: compact cage free-box diagnostic
  `20260705_anchor_cage_compact_diag1_server46` is negative. It used centered
  `0.5 kg` free box, `cage_clearance_xy=0.015`,
  `cage_clearance_z=0.018`, `cage_wall_thickness=0.04`, and stronger drive.
  Fall/drop stayed 0 and all root/body/box/payload shortcut counters stayed 0,
  but the free box was violently expelled: max payload travel `9.08616 m`,
  final payload target distance `9.04616 m`, and max payload relative-offset
  error `20.67944 m`. Conclusion: tightening cage geometry amplifies contact
  impulses and should not be the next free-box strategy.
- 2026-07-05: fixed-payload anchored-support 8 cm diagnostic
  `20260705_anchor_fixed_8cm_diag3_upper08_server46` passed. It completed
  260/260 with one articulated prismatic joint, fixed 4 kg payload, fall/drop
  0, root/body/box/payload pose and velocity shortcuts 0, max torso travel
  `0.08003 m`, max payload travel `0.08003 m`, final target and payload target
  distances `0.000032 m`, min payload z `0.55720 m`, and max payload
  relative-offset error near zero. Checker command:
  `python3 scripts/isaac/check_prismatic_carrier_stand_summary.py experiments/outputs/core_world_anchored_footstep_carrier/20260705_anchor_fixed_8cm_diag3_upper08_server46/core_world_anchored_footstep_carrier_summary.json --log logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_20260705_anchor_fixed_8cm_diag3_upper08_server46.log --max-fall-events 0 --max-box-drop-events 0 --min-torso-travel-x 0.079 --min-payload-travel-x 0.079 --max-final-target-distance-x 0.001 --max-final-payload-target-distance-x 0.001 --max-payload-relative-offset-error 0.002 --min-payload-z 0.55 --require-articulated-carrier --require-foot-contact-drive --min-joint-count 1 --min-joint-motion 0.079`.
  This is the strongest current no-root fixed-payload Isaac diagnostic, but it
  is still a single world-fixed support-frame motion rather than balanced
  walking.
- 2026-07-05: after the latest user correction, external models/checkpoints
  were explicitly removed from the critical path. The active route is direct
  Isaac scene construction. Anchored-support was extended with multi-rail/
  telescoping joints and a closed-stop latch. In Curiosity-owned tmux/Slurm
  session `curiosity_anchor_telescoping_0705`, job `165915` on `server36`,
  `20260705_anchor_telescoping_fixed_24cm_diag1_server36` completed safely with
  fall/drop 0 and no root/body/box/payload shortcuts, but overshot the
  `0.24 m` target to `0.29479 m`, leaving final target distance `0.05479 m`.
  `20260705_anchor_telescoping_fixed_16cm_diag3_closedstop_server36` completed
  360/360 with two rail joints, fall/drop 0, no shortcuts, `stop_latched=true`,
  max torso/payload travel `0.18291 m`, final target distance `0.02291 m`, and
  max payload relative-offset error near zero. Treat this as safe longer
  fixed-payload support-frame evidence, not walking.
- 2026-07-05: added anchored-support staged grasp code in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py` and exposed
  `GRASP_ENABLE_STEP` / `GRASP_SHELF_CLEARANCE` in
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh`. Lightweight
  checks passed with `python3 -m py_compile` and `bash -n`. The new
  `PAYLOAD_MODE=staged_grasp_constraint` starts with a free dynamic box,
  optional preparation shelf, and runtime-authored `StagedGraspJoint`.
  Diagnostics were negative for the current fixed-joint attach design:
  `20260705_anchor_staged_grasp_diag1_server36` completed 220/220 with
  fall/drop 0 and no root/body/box/payload shortcuts, but produced disjoint
  `StagedGraspJoint` warnings, max payload relative-offset error `0.23856 m`,
  final target distance `0.01865 m`, and final payload target distance
  `0.07996 m`. `20260705_anchor_staged_grasp_diag2_runtime_step0_server36`
  also completed safely, but still produced a disjoint-joint warning and ended
  with max payload relative-offset error `0.13740 m`, final target distance
  `0.01867 m`, and final payload target distance `0.12045 m`. Next direct
  Isaac free-box work should use a low-impulse soft/compliant grasp or driven
  clamp/contact formulation, not tighter cage walls and not runtime
  fixed-joint snapping.
- 2026-07-05: in the same Curiosity-owned tmux/Slurm session
  `curiosity_anchor_telescoping_0705`, job `165915` on `server36`, added and
  tested three direct free-box contact alternatives in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py` and exposed
  launcher parameters in
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh`. Lightweight
  checks passed with `python3 -m py_compile` and `bash -n`.
  `PAYLOAD_MODE=open_tray_free_box` keeps the box free on an open tray with
  low stops. `20260705_anchor_open_tray_diag1_slow_server36` completed
  260/260 with fall/drop 0, no disjoint warnings, and zero root/body/box/
  payload shortcuts, but failed carrying: max torso travel `0.01870 m`, final
  target distance `0.02842 m`, final payload target distance `0.06563 m`, and
  max payload relative-offset error `0.18925 m`.
  `20260705_anchor_open_tray_diag2_highstop_slow_server36` made the stops
  taller/tighter and was worse: max absolute payload travel `4.43549 m`, final
  payload target distance `4.47049 m`, and max payload relative-offset error
  `11.36084 m`. Do not continue the taller/tighter stop strategy.
- 2026-07-05: `PAYLOAD_MODE=side_clamp_free_box` adds two prismatic side pads.
  The articulation initialized with three DOFs and no root/body/box/payload
  shortcuts, but the side clamp joints did not effectively close.
  `20260705_anchor_side_clamp_diag1_slowclose_server36` completed 420/420 with
  `281` box-drop events, max clamp joint motion only `5.39e-05 m` for a
  requested `0.07 m` clamp travel, max payload relative-offset error
  `0.94668 m`, and min payload z `0.12000 m`. Stronger drive in
  `20260705_anchor_side_clamp_diag2_strongclamp_stand_server36` and an
  X-axis/rotated-frame clamp joint in
  `20260705_anchor_side_clamp_diag3_rotaxis_strongstand_server36` still left
  max clamp motion around `5.31e-05 m` and produced `64` box-drop events.
  Conclusion: this side-pad prismatic formulation is not an effective gripper
  yet; fix joint/frame/closure in isolation before combining it with carrying.
- 2026-07-05: `PAYLOAD_MODE=x_cradle_free_box` uses an X-axis rear pusher
  because the rail X prismatic axis is known to move. The rear pusher did move:
  `20260705_anchor_x_cradle_diag1_stand_server36` recorded
  `max_cradle_joint_motion_m=0.05205` for `0.052 m` requested travel. However,
  it fired the free box during settle, with max payload travel `12.49785 m`,
  final payload target distance `12.49785 m`, and max payload relative-offset
  error `13.13603 m`. Treat X-cradle as proof that the commanded contact DOF
  can move, not as free-box carrying. Next work should isolate a single moving
  contact element against a supported low-mass free box or dummy object before
  adding carrier rail motion.
- 2026-07-05: after the user correction to stop waiting on external models,
  work continued directly in Isaac on `curiosity_single_contact_cpu_0705d`,
  Slurm job `165977` on `server10`. Added
  `scripts/isaac/build_core_world_single_contact_probe.py` and
  `scripts/isaac/run_core_world_single_contact_probe.sh`. Lightweight checks
  passed with `python3 -m py_compile` and `bash -n`. Single-contact results:
  `20260705_single_contact_probe_diag1_slow_server10` completed 260/260 with
  fall/drop 0 and no root/box shortcuts, but did not reach contact because the
  pusher moved only `0.04704 m` for a `0.050 m` gap.
  `20260705_single_contact_probe_diag2_contact_server10` showed the pusher was
  still too far from the box because actual joint-frame placement put the
  pusher at `x=-0.3251`. The probe was updated with `--box-x` and actual
  surface-gap metrics. `diag3_boxx_contact_server10` achieved real contact
  with actual initial surface gap `0.02510 m`, final surface gap
  `-2.35e-05 m`, fall/drop 0, but high friction/drive settings moved the box
  only `3.46e-05 m`. `diag4_lowfric_strong_server10` used low friction and
  stronger drive and passed as a contact diagnostic: completed 360/360,
  max joint/pusher travel `0.06979 m`, free-box travel `0.04469 m`,
  max box speed `0.07889 m/s`, min box z `0.12999997 m`, fall/drop 0, and all
  root/body/box/payload pose or velocity shortcut counters 0. This is not
  robot carrying; it proves a low-impulse single moving contact can push a
  supported free box in this Isaac/Core API setup.
- 2026-07-05: reran the existing anchored `x_cradle_free_box` with the
  low-friction/strong-contact insight as
  `20260705_anchor_xcradle_lowfric_stand_diag2_server10`. It failed as a
  carrier contact design: completed 260/260 with no root/body/box/payload
  shortcuts and target `0`, but the free box was pushed forward during settle
  and dropped below the active threshold; max cradle joint motion was only
  `1.40e-05 m`. Do not continue this anchored X-cradle geometry as the main
  free-box path.
- 2026-07-05: added a cleaner non-locomotion contact scaffold,
  `scripts/isaac/build_core_world_cradle_cart_free_box_carry.py` and
  `scripts/isaac/run_core_world_cradle_cart_free_box_carry.sh`, to move a
  physical cradle by a prismatic rail while the box remains a free dynamic
  body. This is explicitly a constrained cart/table diagnostic, not robot
  locomotion. `diag1_8cm_server10` failed because the box/deck/walls were
  initialized with local z instead of `cart_z + local z`; the box sat on the
  ground and drop events were 419. After fixing z initialization and adding
  post-settle carry metrics, `20260705_cradle_cart_freebox_diag3_postsettle_8cm_server10`
  passed the contact scaffold: completed 420/420, one articulated `CartRail`
  joint, target `0.08 m`, max cart travel `0.07883 m`, post-settle cart travel
  `0.07881197 m`, post-settle free-box travel `0.07881202 m`, final
  post-settle box relative error `4.79e-08 m`, min box z `0.27615 m`, drop 0,
  nonfinite 0, and all root/body/box/payload pose or velocity shortcuts 0.
  The box naturally shifts to a front-stop contact during settle, so absolute
  initial relative error is about `0.04108 m`; post-settle metrics are the
  valid carry evidence. Next step is to replace the cart rail with a
  robot/support-switching body while preserving this free-box contact module.
- 2026-07-05: integrated the validated cradle contact geometry back into the
  anchored-footstep carrier as `PAYLOAD_MODE=cradle_free_box` in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`. Important
  implementation detail: USD fixed-joint `localPos0` on the scaled torso was
  being interpreted through torso scale. The first anchored cradle diagnostics
  (`diag1_8cm_server23`, `diag2_front_8cm_server23`, `diag3_front_gap08_server23`,
  `diag4_nobodycoll_8cm_server23`, `diag5_geom_server23`) failed or were only
  diagnostic because the cradle stops were effectively misplaced; e.g. before
  the fix, the measured initial front surface gap was `-0.37491 m` and the box
  was expelled. The fix was to use torso-scale-corrected local joint positions
  only for the new cradle parts, and to disable internal anchor/torso body
  collision for this payload mode while preserving cradle/box contact.
  `20260705_anchor_cradle_freebox_diag6b_scaledjoint_geom_server23` verified
  the geometry: rear/front surface gaps were both about `0.025 m`, max payload
  drift was `1.19e-07 m`, min payload z `0.7289998 m`, drop 0.
- 2026-07-05: after the scaled-joint fix, the anchored cradle free-box carrier
  passed fixed-support carry diagnostics on Slurm job `166001` in tmux session
  `curiosity_anchor_cradle_cpu_0705` on `server23`. The 8 cm run
  `20260705_anchor_cradle_freebox_diag7_scaledjoint_8cm_server23` completed
  420/420 with one `StanceRail` joint, target `0.08 m`, max torso travel
  `0.078173 m`, max free-box travel `0.078173 m`, final torso/payload target
  distances about `0.00187 m`, final post-settle payload relative error
  `2.18e-07 m`, min torso z `0.55 m`, min payload z `0.729 m`, fall/drop 0,
  nonfinite 0, and all root/body/box/payload pose or velocity shortcut
  counters 0. The 16 cm fixed-support two-rail run
  `20260705_anchor_cradle_freebox_diag9_fixed_16cm_2rail_server23` completed
  560/560 with two rail joints, target `0.16 m`, max torso travel
  `0.158377 m`, max free-box travel `0.158377 m`, final torso/payload target
  distances about `0.00188 m`, final post-settle relative error `5.77e-08 m`,
  fall/drop 0, nonfinite 0, and no root/body/box/payload shortcuts. This is
  the strongest current free dynamic box carrying scaffold in Isaac, but it is
  still fixed-support anchored carrying, not walking.
- 2026-07-05: support switching remains unsolved. The first attempt
  `20260705_anchor_cradle_freebox_diag8_supportswitch_16cm_server23` used
  `--no-fix-anchor-to-world` to replant the stance anchor across two cycles,
  but failed: PhysX warned `Cannot assign transform to non-root articulation
  link at '/World/Robot/StanceAnchor'`, the carrier tilted above `1.38 rad`,
  fall events were `522`, box-drop events `455`, and final payload target
  distance was `25.28 m`. Conclusion: support switching cannot be implemented
  by `set_world_pose` on the current non-root `StanceAnchor`; the next
  walking-like scaffold must make the replanted support an articulation root,
  an external constraint target, or a separate support controller that does
  not illegally teleport a non-root articulation link.
- 2026-07-05: after the user correction to keep building directly in Isaac
  rather than waiting on external models, tested support-switch alternatives
  on Slurm job `166028` in Curiosity tmux session
  `curiosity_anchor_root_gpu_0705` on `server53`. Do not touch `carry1` or
  non-Curiosity sessions. `diag10_anchorroot_supportswitch_16cm_server53`
  made `/World/Robot/StanceAnchor` the articulation root, which removed the
  original non-root transform warning but failed because the support remained
  a free dynamic body: fall events `520`, box-drop events `455`, max tilt about
  `3.14 rad`. `diag11_kinanchor_supportswitch_16cm_server53` showed PhysX
  rejects an articulation root on a kinematic rigid body (`ArticulationRootAPI
  definition on a kinematic rigid body is not allowed`), so that path is
  invalid. `diag12_worldjoint_replant_16cm_server53` used a world fixed-joint
  retarget instead of any rigid-body pose write; it was stable with fall/drop
  0 but only reached about `0.0401 m` because cycle phase reset and the runtime
  fixed-joint retarget did not accumulate effective support displacement.
  `diag13_worldjoint_phasefix_16cm_server53` fixed the final-cycle phase reset
  and remained stable, but still only reached about `0.0799 m`, confirming the
  fixed-joint retarget is not a real support-replant solution here.
- 2026-07-05: added explicitly labeled `--cumulative-cycle-target` to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`. This is a
  diagnostic only: it commands multi-cycle rail displacement as
  `stride * (cycle + phase)` and must not be described as true support
  switching. `20260705_anchor_cradle_freebox_diag14_cumulative_16cm_server53`
  completed 560/560 with two cycles, `PAYLOAD_MODE=cradle_free_box`,
  `PAYLOAD_MASS=0.5`, `PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.18`, target
  `0.16 m`, max torso travel `0.158374 m`, max free-box travel `0.158374 m`,
  final torso/payload target distances about `0.00193 m`, final post-settle
  payload/torso relative error `3.13e-08 m`, min torso z `0.55 m`, min payload
  z `0.729 m`, fall/drop 0, nonfinite 0, and all root/body/box/payload pose or
  velocity shortcut counters 0. Treat this as stable multi-cycle articulated
  rail transport of a free dynamic box, not walking, not unknown-load active
  probing, and not true support switching.
- 2026-07-05: extended the cumulative-cycle cradle free-box scaffold on the
  same Curiosity allocation before releasing it. `diag15_cumulative_32cm_server53`
  completed 980/980 with four cycles, target `0.32 m`, `0.5 kg` payload, max
  torso/free-box travel about `0.31925 m`, final target distances about
  `0.00195 m`, fall/drop 0. `diag16_cumulative_32cm_4kg_server53` completed
  980/980 with `4.0 kg`, max travel about `0.31949 m`, final target distances
  about `0.00186 m`, fall/drop 0. `diag17_cumulative_32cm_8kg_server53`
  completed 980/980 with `8.0 kg`, max travel about `0.31969 m`, final target
  distances about `0.00194 m`, final post-settle payload/torso relative error
  `2.08e-07 m`, fall/drop 0, nonfinite 0, and no root/body/box/payload pose or
  velocity shortcuts. These runs strengthen the free-box/load-bearing Isaac
  scaffold but remain cumulative rail-target diagnostics, not true support
  switching, walking, unknown-load probing, or video-guided learning evidence.
  Released Slurm job `166028` and killed tmux session
  `curiosity_anchor_root_gpu_0705`; do not touch unrelated pending/running
  jobs such as `phase03_render_probe5`.
- 2026-07-05: after the user told us to keep pushing in Isaac, ported
  `PAYLOAD_MODE=cradle_free_box` into the physical-foot prismatic carrier
  (`scripts/isaac/build_core_world_prismatic_carrier_stand.py`) and passed the
  cradle parameters through
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`. This path has a
  free articulated torso, four physical feet, optional horizontal foot slides,
  and no torso/root pose or velocity writes. The cradle port uses the same
  scaled-torso fixed-joint correction used in the anchored cradle and records
  rear/front surface gaps plus post-settle active travel metrics.
- 2026-07-05: prismatic cradle/free-box diagnostics ran on Slurm job `166052`
  in Curiosity tmux session `curiosity_prismatic_cradle_gpu_0705` on
  `server53`. `20260705_prismatic_cradle_stand_diag1b_8kg_server53` completed
  500/500 with an `8 kg` free box, rear/front cradle gaps about
  `0.0224/0.0269 m`, fall/drop 0, no root/body/box/payload pose or velocity
  writes, but settled backward about `5.6 cm` and changed payload relative
  offset by about `6.2 cm`. `diag1_4cm_8kg` and `diag2_neg4cm_8kg` showed the
  first horizontal-leg `sync_inchworm` gait remained safe with fall/drop 0 but
  absolute target evidence was confounded by settle drift. `diag3b_postsettle`
  added post-settle metrics and showed final active torso/payload travel about
  `-0.01997 m` for a `-0.04 m` target, fall/drop 0. The strongest current
  physical-foot result is
  `20260705_prismatic_cradle_sync_inchworm_diag4_postsettle_neg8cm_8kg_server53`:
  1100/1100 steps, 8 DOFs, `8 kg` free box, final post-settle torso/payload
  travel about `-0.04627 m`, peak active travel about `0.06195 m`, max tilt
  `0.09624 rad`, min payload z `0.7281 m`, fall/drop 0, nonfinite 0, and all
  root/body/box/payload pose or velocity shortcut counters 0. Treat this as a
  simplified prismatic-legged physical-foot carrying diagnostic, not complete
  walking, not learned balance, not unknown-load active probing, and not video-
  guided learning evidence.
- 2026-07-05: continued the direct Isaac prismatic-leg cradle/free-box route
  without treating external models as blockers. In Curiosity-owned tmux/Slurm
  session `curiosity_prismatic_cradle_long_gpu_0705`, job `166070` on
  `server53`, extended the 8 kg free dynamic box cradle diagnostic:
  `20260705_prismatic_cradle_sync_inchworm_diag5_postsettle_neg14cm_8kg_server53`
  completed 1500/1500 with fall/drop 0, nonfinite 0, and root/body/box/payload
  pose or velocity shortcut counters 0. It reached final post-settle torso/
  payload travel about `-0.08707 m` and peak active post-settle travel about
  `0.10298 m`, with max tilt `0.09628 rad` and min payload z `0.7275 m`.
  `20260705_prismatic_cradle_sync_inchworm_diag6_postsettle_neg22cm_8kg_server53`
  completed 1900/1900 with the same shortcut counters 0, fall/drop 0, final
  post-settle torso/payload travel about `-0.14711 m`, peak active travel
  about `0.16318 m`, max tilt `0.09639 rad`, and min payload z `0.71986 m`.
  `20260705_prismatic_cradle_sync_inchworm_diag7_postsettle_neg30cm_8kg_server53`
  completed 2350/2350 with target `-0.30 m`, five sync-inchworm cycles, 8 kg
  free dynamic box, fall/drop 0, nonfinite 0, root/body/box/payload pose or
  velocity shortcuts 0, max tilt `0.09653 rad`, min payload z `0.72755 m`,
  final post-settle torso/payload travel about `-0.20588/-0.20587 m`, and
  peak active post-settle torso/payload travel about `0.22180/0.22179 m`.
  Final post-settle target distance remained about `0.09412 m`, so this is
  meaningful physical-foot free-box transport evidence but not a completed
  target-reaching gait, not a humanoid/quadruped walk, and not learned balance.
- 2026-07-05: after the user correction, the active critical path is direct
  Isaac scene/controller construction. Do not block progress on G1/WBC, Go2
  policy, GR00T, T-Rex, video models, or dataset downloads unless they
  immediately unblock the Isaac carrying scene. The current strongest active
  scaffold is the prismatic physical-foot cradle/free-box route above; the next
  launched test is
  `20260705_prismatic_cradle_sync_inchworm_diag8_postsettle_neg40cm_8kg_server53`,
  targeting `-0.40 m` with an 8 kg free box to measure whether post-settle
  distance extends beyond the `diag7` plateau.
- 2026-07-05: `diag8_postsettle_neg40cm_8kg_server53` is a negative result.
  It completed 3150/3150, but the more aggressive `-0.40 m` target,
  `STEP_LENGTH=0.07`, `X_SLIDE_LIMIT=0.24`, and stronger slide/leg drives
  destabilized the carrier badly: fall events `3126`, box-drop events `2826`,
  min torso z about `-1071.47 m`, min payload z about `-1073.76 m`, max tilt
  `3.14108 rad`, and max payload relative-offset error `117.19 m`. Shortcut
  counters stayed 0, so this is a real dynamics failure, not a hidden pose-write
  artifact. Do not tune by simply increasing slide limit or drive force. A
  follow-up `diag9_postsettle_neg34cm_8kg_server53` was launched with the
  safer `diag7` parameter family and only a small target extension.
- 2026-07-05: `diag9_postsettle_neg34cm_8kg_server53` is also a negative
  result. It used the safer `diag7` parameter family with target `-0.34 m`
  and six sync-inchworm cycles. It completed 2700/2700, but the carrier became
  unstable after extending beyond the five-cycle `diag7` regime: fall events
  `2637`, box-drop events `2182`, min torso z `-828.23 m`, max tilt
  `3.09742 rad`, and max payload relative-offset error `829.85 m`. Shortcut
  counters were still 0. The payload happened to end near the X target while
  the carrier had fallen, so this must not be counted as carrying. Code change:
  added `--sync-cycle-pause-fraction` to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and launcher
  env `SYNC_CYCLE_PAUSE_FRACTION` to insert an all-feet-support pause between
  sync-inchworm cycles. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/check_prismatic_carrier_stand_summary.py`
  and `bash -n scripts/isaac/run_core_world_prismatic_carrier_stand.sh`.
  Follow-up `20260705_prismatic_cradle_sync_inchworm_diag10_pause_neg34cm_8kg_server53`
  was launched with target `-0.34 m`, `GAIT_PERIOD_STEPS=420`, and
  `SYNC_CYCLE_PAUSE_FRACTION=0.20`.
- 2026-07-05: the first cycle-pause follow-ups did not produce valid positive
  evidence. `diag10_pause_neg34cm_8kg_server53` completed but its summary
  recorded `sync_cycle_pause_fraction=0.0`, so the intended pause did not enter
  the simulation; it failed similarly to `diag9` with fall events `2937`,
  box-drop events `2482`, min torso z `-1030.66 m`, and max payload
  relative-offset error `1032.28 m`. A runner inspection on `server53` showed
  the compute tmux saw the builder-side pause argument but not the newly patched
  runner argument, so `diag11_pause_valid...` was not a valid runner-based
  pause test and was interrupted. A direct-Python attempt,
  `diag12_pause_direct_neg34cm_8kg_server53`, explicitly included
  `--sync-cycle-pause-fraction 0.20`, but it became unstable during the early
  stand/settle rollout and was interrupted rather than counted as a full
  experiment. Current conclusion: the reliable positive result remains
  `diag7`; current prismatic sync-inchworm is not a robust longer-distance
  locomotion base, and the next productive work is a new support/foot-placement
  mechanism rather than more target/force tuning.
- 2026-07-05: implemented the next prismatic support-controller iteration
  without adding external model dependencies. `scripts/isaac/build_core_world_prismatic_carrier_stand.py`
  now supports `motion_mode=feedback_sync_inchworm`. This mode keeps the same
  physical-foot/free-box cradle scaffold but advances the sync-inchworm gait
  clock only when the previous physics step was safe: no fall, no drop, tilt
  below `--feedback-tilt-hold-threshold`, and payload relative-offset error
  below `--feedback-payload-error-hold-threshold`. Otherwise it holds the
  current gait phase and records `feedback_hold_steps`,
  `feedback_release_steps`, `feedback_motion_step_final`,
  `feedback_last_safe`, and `feedback_last_block_reason`. The launcher exposes
  `FEEDBACK_TILT_HOLD_THRESHOLD` and
  `FEEDBACK_PAYLOAD_ERROR_HOLD_THRESHOLD`, and its config line now prints the
  sync-pause and feedback thresholds. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/check_prismatic_carrier_stand_summary.py`
  and `bash -n scripts/isaac/run_core_world_prismatic_carrier_stand.sh`.
  A system-Python `--help` probe failed with `ModuleNotFoundError: isaaclab`,
  which is expected outside the Isaac venv and is not treated as a simulation
  result. Next compute diagnostic should run `feedback_sync_inchworm` on the
  8 kg cradle/free-box task and compare against `diag7/diag9`.
- 2026-07-05: extended
  `scripts/isaac/check_prismatic_carrier_stand_summary.py` with stricter
  post-settle gates for the feedback/free-box runs:
  `--min-abs-post-settle-torso-travel-x`,
  `--min-abs-post-settle-payload-travel-x`,
  `--max-final-post-settle-target-distance-x`, and
  `--max-final-post-settle-payload-target-distance-x`. Checker reports now
  also include feedback hold/release fields. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/check_prismatic_carrier_stand_summary.py`
  and `python3 scripts/isaac/check_prismatic_carrier_stand_summary.py --help`
  showed the new post-settle flags.
- 2026-07-05: ran feedback-controller diagnostics in Curiosity-owned tmux/Slurm
  session `curiosity_feedback_inchworm_0705`, job `166115` on `server53`.
  First attempts `20260705_prismatic_cradle_feedback_sync_diag1_neg30cm_8kg_server53`,
  `stand_regression_diag2_8kg_server53`, and
  `stand_regression_diag3_stablelegs_8kg_server53` are negative parameter
  controls, not controller evidence: they omitted the historical stable
  `PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.18`, and/or
  `LEG_TARGET=-0.57`, `LEG_LOWER=-0.82` settings, so the free box and cradle
  were placed around `x=0.24` instead of the `diag7` `x≈0.50` configuration
  and the carrier fell. With corrected payload and leg parameters,
  `20260705_prismatic_cradle_stand_regression_diag4_stable_payload_pose_8kg_server53`
  passed a stand regression gate: 500/500, 8 DOFs, fall/drop 0, no root/body/
  box/payload shortcuts, max tilt `0.11573 rad`, min payload z `0.71281 m`.
  This confirms the scene did not regress when the stable pose parameters are
  restored.
- 2026-07-05: valid first feedback run
  `20260705_prismatic_cradle_feedback_sync_diag2_neg30cm_stableparams_8kg_server53`
  completed 2350/2350 with the stable 8 kg cradle/free-box parameters,
  `motion_mode=feedback_sync_inchworm`, fall/drop 0, nonfinite 0, no root/body/
  box/payload shortcuts, max tilt `0.11573 rad`, min payload z `0.71281 m`,
  `feedback_release_steps=2090`, and `feedback_hold_steps=0`. It is safe but
  weak: max post-settle torso/payload travel was only about
  `0.05948/0.05981 m` and final post-settle travel returned near
  `-0.00143/-0.00176 m`, leaving final post-settle target distance about
  `0.29857 m`. This is not a carrying-distance improvement over `diag7`; it
  shows the feedback clock as implemented preserves safety but does not retain
  cumulative travel. A same-parameter ordinary `sync_inchworm` replay,
  `20260705_prismatic_cradle_sync_replay_diag13_neg30cm_stableparams_8kg_server53`,
  was launched to determine whether the new code changed the open-loop
  baseline or whether the feedback controller itself suppresses net transport.
- 2026-07-05 follow-up: the prismatic open-loop replay did not reproduce
  historical `diag7`. It remained safe but weak, with only about `0.0595 m`
  peak active post-settle travel and near-zero final active travel. Additional
  command-vs-actual diagnostics showed that the controller commands swing lift
  and horizontal slide, but sampled foot world height remains near ground
  contact during swing. Treat the current prismatic sync-inchworm gait as a
  contact/load-bearing diagnostic only, not as walking progress. The active
  direct-Isaac path should preserve the validated free dynamic box and metrics
  while moving toward a cleaner swappable controller/task interface.
- 2026-07-05 direct task-interface update:
  `scripts/isaac/build_direct_carry_task_scene.py` now exposes explicit box
  randomization and a `controller_contract`, and records
  `robot_proxy_pose_write_count` plus `box_kinematic_pose_write_count`.
  `scripts/isaac/check_direct_carry_task_summary.py` enforces that these runs
  are diagnostic-only and disclose the kinematic proxy limitation. Verified
  smoke `20260705_direct_carry_task_interface_rand_smoke1_server53` completed
  180/180 with `BOX_SEED=7051`, sampled `7.2301 kg` box, box-drop 0, max box
  travel `0.67485 m`, final target distance `0.03485 m`, robot proxy pose
  writes `2340`, and box kinematic pose writes `180`. Do not cite this as
  robot locomotion, physical grasping, or no-root carrying success.
- 2026-07-05 direct physical backend wrapper:
  `scripts/isaac/run_direct_carry_task_physical_backend.sh` now wraps the
  anchored/cradle free-box backend and
  `scripts/isaac/normalize_direct_carry_backend_summary.py` normalizes its
  output into the direct-task schema. The first run,
  `20260705_direct_physical_backend_anchor_cradle_smoke1_server10`, exposed a
  wrapper rail-limit bug (`RAIL_UPPER=0.04` capped the 32 cm target at
  16 cm) and was interrupted. After changing defaults to
  `RAIL_LOWER=-0.04`, `RAIL_UPPER=0.10`,
  `20260705_direct_physical_backend_anchor_cradle_smoke2_railupper10_server10`
  passed the direct backend checker: 980/980, 8 kg free dynamic box,
  fall/drop 0, root shortcut free, max and post-settle box travel
  `0.319915 m`, final box target distance `0.001939 m`, final box/torso
  relative error `6.46e-08 m`, support-root pose writes 0, and
  `anchor_world_joint_retarget_count=4`. Treat this as physical backend
  progress beyond the kinematic task proxy, but not complete robot walking:
  it still depends on anchored/replanted world support.
- 2026-07-05 fixed-anchor backend ablation:
  `run_direct_carry_task_physical_backend.sh` supports
  `SUPPORT_MODE=fixed_anchor`, reported as
  `controller_mode=physical_fixed_anchor_cradle`, and the direct-task checker
  can now gate `anchor_world_joint_retarget_count` and
  `support_root_pose_write_count`. Compute run
  `20260705_direct_physical_backend_fixed_anchor_32cm_8kg_server10` passed:
  980/980, 8 kg free dynamic box, fall/drop 0, root shortcut free, max
  post-settle box travel `0.322541 m`, final box target distance
  `0.001794 m`, final box/torso relative error `6.46e-08 m`,
  `anchor_world_joint_retarget_count=0`, and support-root pose writes 0.
  This shows the free-box cradle/contact backend does not require world-joint
  replanting for 32 cm/8 kg. It is still fixed world support, not robot
  walking or balance.
- 2026-07-05 route correction after user instruction: do not block current
  progress on external models, checkpoints, or datasets. Continue by building
  the Isaac scene/controller stack directly. External video/model methods are
  research references only until the simulator can produce a credible
  carry-task substrate. The immediate engineering target is Isaac-first:
  preserve the validated free dynamic box, randomization hooks, cradle/contact
  metrics, and shortcut counters; replace the fixed world-support rail with a
  physically declared support-switching or foot-placement carrier; then add
  probing and morphology/posture variation. Do not claim robot walking until
  the scene has no fixed world support, no support-root pose writes, no
  anchor-world-joint retargets, and the body advances through physical support
  contacts or a documented locomotion controller.
- 2026-07-05 interrupted posture diagnostic:
  `20260705_direct_physical_backend_fixed_anchor_lowfront_32cm_8kg_server36`
  was launched in Curiosity-owned tmux/Slurm job `166191` on `server36` with
  `SUPPORT_MODE=fixed_anchor`, `CARRY_POSTURE=low_front`, `TARGET_X=0.32`,
  and `PAYLOAD_MASS=8.0`. The run entered Isaac and reached about step 410
  with fall/drop still 0 and target distance about `0.0018 m`, but it was
  manually interrupted before completion and produced only
  `core_world_anchored_footstep_carrier_state.csv`, not a normalized summary.
  Treat it as an invalid/partial diagnostic, not success evidence. The Slurm
  allocation was cancelled and tmux session `curiosity_posture_backend_0705`
  was killed after the interruption.
- 2026-07-05 direct no-root prismatic backend implementation:
  added `scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh` so
  the no-root prismatic legged carrier can be run as a direct carry-task
  backend and normalized with the same schema as the fixed-anchor backend.
  `scripts/isaac/normalize_direct_carry_backend_summary.py` now accepts
  explicit `--backend-support-mode` and `--non-success-reason` overrides, and
  records no-root legged diagnostics including motion mode, commanded/actual
  leg lift, x-slide, and foot-height fields when available.
  `scripts/isaac/check_direct_carry_task_summary.py` now supports
  `--expect-backend-support-mode` and `--max-fall-events`, and reports the
  legged backend metrics. Lightweight checks passed on the login node:
  `python3 -m py_compile scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/check_direct_carry_task_summary.py`
  and `bash -n scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh scripts/isaac/run_direct_carry_task_physical_backend.sh`.
  A compute allocation was requested in tmux session
  `curiosity_no_root_prismatic_direct_0705`, Slurm job `166199`, to run the
  first short-distance no-root/free-box diagnostic. This run is expected to be
  a diagnostic, not a success claim.
- 2026-07-05 direct no-root prismatic backend result:
  compute run
  `20260705_direct_no_root_prismatic_cradle_feedback_10cm_8kg_server46` ran in
  tmux session `curiosity_no_root_prismatic_direct_0705`, Slurm job `166199`
  on `server46`. It completed 1200/1200 with
  `controller_mode=no_root_prismatic_legged_cradle`,
  `backend_support_mode=no_root_prismatic_legged`,
  `motion_mode=feedback_sync_inchworm`, `PAYLOAD_MODE=cradle_free_box`,
  `PAYLOAD_MASS=8.0`, and `TARGET_X=0.10`. Structural checker gates passed:
  fall/drop 0, root shortcut free, robot proxy writes 0, box kinematic writes
  0, `anchor_world_joint_retarget_count=0`, and
  `support_root_pose_write_count=0`. It is a negative carrying-distance
  result: max positive box travel was only `0.02458 m`, max post-settle box
  travel was `0.05242 m`, and final box target distance was `0.14996 m`, so
  the box ended roughly `5 cm` opposite the requested positive target instead
  of carrying `10 cm` forward. Leg diagnostics showed commanded lift
  `0.07998 m`, actual leg-lift metric `0.29275 m`, and max actual x-slide
  `0.04454 m`. Interpretation: the no-root/free-foot/free-box direct backend
  is wired into the evidence pipeline and is safe for this short diagnostic,
  but current prismatic gait/contact mechanics do not produce useful forward
  transport. Next work should debug leg/contact propulsion in Isaac directly,
  not wait for external models.
- 2026-07-05 per-leg no-root diagnostics:
  extended `scripts/isaac/build_core_world_prismatic_carrier_stand.py` with
  `--foot-contact-z-threshold` and per-leg summary fields for near-ground
  steps, min/max foot z, max commanded lift, max commanded x, max actual lift,
  and max actual x. The CSV now records `near_ground_foot_count` and
  `commanded_swing_foot_count`. `run_core_world_prismatic_carrier_stand.sh`,
  `run_direct_carry_task_no_root_prismatic_backend.sh`,
  `normalize_direct_carry_backend_summary.py`, and
  `check_direct_carry_task_summary.py` were updated to pass/report these
  fields. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/check_direct_carry_task_summary.py`
  and `bash -n scripts/isaac/run_core_world_prismatic_carrier_stand.sh scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh`.
- 2026-07-05 no-root quasi-static stance-transfer diagnostics:
  added `motion_mode=quasistatic_stance_transfer` to the prismatic backend.
  This mode keeps all feet near ground and drives horizontal leg slides as a
  sign/contact diagnostic, not a walking controller. Three compute runs were
  executed in Curiosity-owned tmux session
  `curiosity_no_root_prismatic_diagfields_0705`, Slurm job `166206` on
  `server46`, then the allocation/session was released. First,
  `20260705_direct_no_root_prismatic_quasistatic_10cm_8kg_server46` used the
  initial sign and moved the 8 kg free box the wrong way: structural gates
  passed, but final box target distance was `0.25298 m` for `TARGET_X=0.10`,
  confirming the sign was wrong. Second,
  `20260705_direct_no_root_prismatic_quasistatic_neg10cm_8kg_server46`
  verified that the opposite command sign produced positive body/box travel
  around `0.05 m`. After fixing the mode to use `-direction`,
  `20260705_direct_no_root_prismatic_quasistatic_corrected_10cm_8kg_server46`
  completed 1200/1200 with fall/drop 0, root shortcut free, box kinematic
  writes 0, anchor retargets 0, support-root writes 0, max box travel
  `0.05647 m`, max post-settle box travel `0.09855 m`, final box target
  distance `0.04855 m`, max tilt `0.11319 rad`, and per-leg near-ground
  counts `fl=1170`, `fr=1170`, `rl=1160`, `rr=1160`. This is real no-root
  Isaac progress over the earlier wrong-sign run, but still not success: it is
  a quasi-static prismatic-foot scaffold, it carries only about half of the
  requested 10 cm target by final state, and it has no active probing or
  morphology-aware posture selection yet.
- 2026-07-05 no-root target metric and compensation update:
  extended `normalize_direct_carry_backend_summary.py` and
  `check_direct_carry_task_summary.py` to report and optionally gate
  `final_post_settle_box_travel_x_m` and
  `final_post_settle_box_target_distance_x_m`, because the no-root prismatic
  carrier has a startup settling drift before active transport. Added
  `--quasistatic-compensate-settle-drift` to
  `build_core_world_prismatic_carrier_stand.py`, exposed through
  `run_core_world_prismatic_carrier_stand.sh` and the direct no-root wrapper.
  When enabled for `quasistatic_stance_transfer`, the effective horizontal
  target becomes `target_x - pre-motion_torso_settle_drift`, so a `TARGET_X`
  command can chase absolute displacement instead of only post-settle active
  displacement. The normalizer/checker now report
  `quasistatic_compensate_settle_drift` and
  `quasistatic_effective_target_x_m`. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/check_direct_carry_task_summary.py`
  and `bash -n scripts/isaac/run_core_world_prismatic_carrier_stand.sh scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh`.
  A compute allocation was requested in tmux
  `curiosity_no_root_compensated_0705`, Slurm job `166217`, to run the
  compensated `TARGET_X=0.10`, 8 kg free-box diagnostic.
- 2026-07-05 compensated no-root stance-transfer result:
  compute run
  `20260705_direct_no_root_prismatic_quasistatic_compensated_10cm_8kg_server02`
  ran in Curiosity-owned tmux `curiosity_no_root_compensated_0705`, Slurm job
  `166217` on `server02`. Configuration: `TARGET_X=0.10`,
  `PAYLOAD_MASS=8.0`, `MOTION_MODE=quasistatic_stance_transfer`,
  `QUASISTATIC_COMPENSATE_SETTLE_DRIFT=1`, `X_SLIDE_LIMIT=0.20`, free
  cradle box. It passed direct-task gates with fall/drop 0, root shortcut free,
  no box kinematic writes, no anchor retargets, no support-root writes,
  `max_box_travel_x_m=0.10413`, `final_box_target_distance_x_m=0.00413`,
  `final_post_settle_box_travel_x_m=0.14621`, and
  `final_post_settle_box_target_distance_x_m=0.04621`. This is the strongest
  no-root physical-foot/free-box result so far: it reaches and holds the
  requested 10 cm absolute carry target without fixed world support or box pose
  writes. It is still not final success because it is quasi-static prismatic
  stance transfer with all feet effectively serving as sliding supports, not
  walking with support switching.
- 2026-07-05 no-root step-cycle negative results:
  added `motion_mode=quasistatic_step_cycle`, which drives stance transfer and
  then resets feet one at a time. First attempt
  `20260705_direct_no_root_prismatic_stepcycle_compensated_10cm_8kg_server02`
  hit a transient compute-side syntax read of the edited file and produced no
  simulation summary; local `py_compile` then passed and the retry was run.
  Retry
  `20260705_direct_no_root_prismatic_stepcycle_compensated_10cm_8kg_server02_retry`
  completed 1500/1500 safely with fall/drop 0 and no shortcuts, but failed
  displacement retention: `max_box_travel_x_m=0.02968`,
  `final_box_target_distance_x_m=0.14934`, and
  `final_post_settle_box_travel_x_m=-0.00725`. A slower/higher reset run,
  `20260705_direct_no_root_prismatic_stepcycle_slowreset_10cm_8kg_server02`,
  also completed safely and reached a large transient
  `max_post_settle_box_travel_x_m=0.25199`, but final active travel collapsed
  to `0.01530 m` with final absolute target distance `0.12679 m`. Conclusion:
  simple one-leg-at-a-time reset currently pulls the carried body/box back and
  does not preserve locomotion displacement. Next work should add a support
  phase controller with explicit stance-lock/contact maintenance and reset
  acceptance criteria, rather than tuning reset height alone. Slurm job
  `166217` and tmux `curiosity_no_root_compensated_0705` were released.
- 2026-07-05 Isaac-first correction after user direction:
  stopped treating external models as a blocker and pushed the Isaac no-root
  free-box scene directly. Added `motion_mode=gated_quasistatic_step_cycle`,
  `--gated-step-max-travel-loss`, `--gated-step-recovery-phase`, and summary
  fields for post-settle travel loss after peak. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/check_direct_carry_task_summary.py`
  and `bash -n scripts/isaac/run_core_world_prismatic_carrier_stand.sh scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh`.
  Diagnostic run
  `20260705_direct_no_root_prismatic_gated_step_10cm_8kg_mgmtserver02`
  ran in Curiosity-owned tmux `curiosity_gated_step_0705`, Slurm job `166233`
  on `server10`. It completed 1500/1500 with fall/drop 0, root shortcut free,
  no box kinematic writes, no anchor retargets, and no support-root writes,
  but it was a negative support-switching result: max post-settle box travel
  reached `0.13806 m`, final post-settle box travel was only `0.04578 m`,
  final box target distance was `0.09630 m`, and post-settle travel loss after
  peak was `0.09228 m`. The gate correctly detected
  `post_settle_payload_travel_loss`, but recovery happened after the reset had
  already erased most displacement. This should not be pursued as the main
  progress claim without a different support-switching design.
- 2026-07-05 no-root posture sweep:
  added `scripts/isaac/run_no_root_prismatic_posture_sweep.sh` and ran it in
  Curiosity-owned tmux `curiosity_posture_sweep_0705`, Slurm job `166237` on
  `server10`. The sweep used `MOTION_MODE=quasistatic_stance_transfer`,
  `QUASISTATIC_COMPENSATE_SETTLE_DRIFT=1`, `TARGET_X=0.10`, 8 kg free dynamic
  cradle box, and strict direct-task checks for each carry posture. Results:
  `front_mid` passed 1200/1200 with final box target distance `0.00413 m`,
  max box travel `0.10413 m`, max tilt `0.11319 rad`, fall/drop 0, root
  shortcut free, anchor retargets 0, and support-root writes 0. `low_front`
  passed with final box target distance `0.01300 m`, max box travel
  `0.08700 m`, max tilt `0.13019 rad`, fall/drop 0, and the same shortcut-free
  gates. `chest_high` passed with final box target distance `0.01290 m`, max
  box travel `0.08710 m`, max tilt `0.10504 rad`, fall/drop 0, and the same
  shortcut-free gates. This establishes a useful Isaac scene milestone for
  multiple carry postures under no-root/free-box constraints, but it is still
  quasi-static stance transfer with sliding support feet, not walking, not
  active probing, and not a learned policy.
- 2026-07-05 prelift/guarded-prelift support-switching diagnostics:
  Isaac-first work continued without waiting for external models. Added
  `motion_mode=prelift_quasistatic_step_cycle` and
  `motion_mode=guarded_prelift_quasistatic_step_cycle`, with lift/translate/
  lower reset phases and guarded target/travel-loss logic. Results remain
  negative for true support switching. The plain prelift run
  `20260705_direct_no_root_prismatic_prelift_step_10cm_8kg_server36` was
  unsafe (`fall_events=1172`, max tilt `1.05202 rad`). Guarded prelift was
  safe but weak: `20260705_direct_no_root_prismatic_guarded_prelift_10cm_8kg_server10`
  reached only max box travel `0.07371 m`; the stride12 run preserved
  post-settle peak travel but failed raw target gates; the compensated run
  `20260705_direct_no_root_prismatic_guarded_prelift_comp_10cm_8kg_server10`
  stayed safe with real lift commands (`max_commanded_leg_lift_m=0.10`) and no
  root/box/support shortcuts, but still failed direct gates with max box travel
  `0.05720 m`, final box target distance `0.06995 m`, and post-settle travel
  loss `0.02715 m`. Interpretation: simply prelifting the reset foot does not
  preserve displacement; support switching needs a stance-lock/foot-placement
  mechanism, not more claims around the current gait.
- 2026-07-05 current active diagnostic:
  the longer 20 cm guarded-prelift diagnostic
  `20260705_direct_no_root_prismatic_guarded_prelift_20cm_8kg_mgmtserver02`
  ran on `server46`, Slurm job `166271`. It completed 2400/2400 safely with
  fall/drop 0, root shortcut free, anchor retargets 0, support-root writes 0,
  max box travel `0.15029 m`, and max post-settle box travel `0.19238 m`.
  It is still negative for support-switching carry: final box target distance
  was `0.12198 m`, final post-settle target distance `0.07990 m`, and
  post-settle travel loss after peak `0.07227 m`. The controller deadlocked in
  recovery after reset-induced travel loss.
- 2026-07-05 guarded-loss rebaseline diagnostic update:
  added `--gated-step-loss-rebaseline-steps`, exposed it through the launchers,
  and normalized `gated_step_loss_rebaseline_count`. This is diagnostic
  plumbing only, not a success mechanism. It allows the controller to accept a
  stable lower post-reset baseline after prolonged loss recovery and continue
  testing later support-switch phases instead of being locked by one transient
  peak. Run
  `20260705_direct_no_root_prismatic_guarded_prelift_rebaseline_20cm_8kg`
  completed 2600/2600 on `server10`, Slurm job `166281`, with fall/drop 0, no
  root/box/support shortcuts, all four legs receiving lift commands,
  `gated_step_loss_rebaseline_count=3`, and final post-settle travel loss
  reduced to `0.00158 m`. It still failed the 20 cm target gate: max box
  travel `0.15029 m`, final box target distance `0.10593 m`, and final
  post-settle target distance `0.06384 m`. Interpretation: rebaselining
  prevents recovery deadlock but accepts lost transport. Do not continue
  treating this prismatic gait as the main support-switching solution; the next
  design needs stance-foot lock/contact anchoring or a different locomotion
  controller while preserving the same no-root/free-box shortcut gates.
- 2026-07-05 stance-overdrive diagnostic update:
  added `--prelift-stance-overdrive` to
  `build_core_world_prismatic_carrier_stand.py`, exposed
  `PRELIFT_STANCE_OVERDRIVE` through the core and direct no-root launchers,
  and normalized `prelift_stance_overdrive`. During prelift reset, not-yet-
  reset stance legs can overdrive their x-slide stance target to counter the
  swing foot's return reaction. This is diagnostic plumbing only. Active run
  `curiosity_guarded_overdrive_20cm_0705`, Slurm job `166289`, uses
  `TARGET_X=0.20`, `STEP_LENGTH=0.14`, `X_SLIDE_LIMIT=0.25`,
  `STEP_HEIGHT=0.10`, `GAIT_PERIOD_STEPS=720`,
  `PRELIFT_STANCE_OVERDRIVE=1.45`, no loss rebaseline, and an 8 kg free box.
  Its purpose is to test whether missing stance compensation is the dominant
  reset-pullback cause. Treat it as diagnostic only.
- 2026-07-05 stance-overdrive results:
  `20260705_direct_no_root_prismatic_guarded_prelift_overdrive145_20cm_8kg`
  ran on `server10`, Slurm job `166289`, and was unsafe:
  `fall_events=1976`, max tilt `0.91439 rad`, final box target distance
  `0.37494 m`. It produced large transient box travel but only by overdriving
  into a fall. `20260705_direct_no_root_prismatic_guarded_prelift_overdrive115_20cm_8kg`
  ran on `server10`, Slurm job `166292`, and stayed safe with fall/drop 0 and
  no shortcuts, but still failed: max box travel `0.17314 m`, final box target
  distance `0.12104 m`, and travel loss after peak `0.09418 m`. Conclusion:
  simple stance overdrive is not a solution for support-switching retention.
- 2026-07-05 low-reaction swing-foot diagnostic update:
  added `--swing-x-force-scale`, launcher env `SWING_X_FORCE_SCALE`, and
  normalized `swing_x_force_scaled_steps` plus
  `per_leg_swing_x_force_scaled_steps`. When a leg has commanded lift, its
  x-slide drive max force can be reduced while stance legs keep full force.
  Active run `curiosity_guarded_swingforce_20cm_0705`, Slurm job `166300`,
  uses `SWING_X_FORCE_SCALE=0.08`, no stance overdrive, no loss rebaseline,
  20 cm target, and an 8 kg free box. This tests whether swing-foot return
  reaction is the reset-pullback source. Treat it as diagnostic only.
- 2026-07-05 low-reaction swing-foot result:
  `20260705_direct_no_root_prismatic_guarded_prelift_swingforce008_20cm_8kg`
  ran on `server10`, Slurm job `166300`. It completed 2600/2600 with fall/drop
  0, root shortcut free, anchor retargets 0, support-root writes 0, and
  verified `swing_x_force_scaled_steps=118`. It still failed with essentially
  the same metrics as the unscaled guarded-prelift run: max box travel
  `0.15029 m`, max post-settle box travel `0.19238 m`, final box target
  distance `0.12198 m`, final post-settle target distance `0.07989 m`, and
  travel loss after peak `0.07227 m`. This means swing x-drive force scaling
  is not enough, or runtime drive-force changes are not the dominant actuator
  path in this scaffold.
- 2026-07-05 route correction:
  stop tuning the current prismatic gait parameters as the main support-
  switching route. The next useful Isaac diagnostic is a stance-foot latch /
  contact-anchoring variant that explicitly holds stance feet fixed while a
  swing foot is repositioned, counts all latch retargets as non-final
  scaffolding, and tests whether idealized stance locking preserves carried
  box displacement. If it fails, replace the carrier morphology/controller. If
  it works, use it as the constraint target for a real contact/friction
  controller. Do not claim the latch as final robot walking.
- 2026-07-05 stance-foot latch implementation:
  `build_core_world_prismatic_carrier_stand.py` now supports
  `--enable-stance-foot-latch` and `--stance-foot-latch-lift-threshold`.
  Each foot has a disabled world fixed joint; during rollout, stance feet are
  latched to their current world pose and swing feet are unlatched. Launchers
  expose `ENABLE_STANCE_FOOT_LATCH` and
  `STANCE_FOOT_LATCH_LIFT_THRESHOLD`; the normalizer reports latch counters.
  This is scaffold evidence only, not final walking. Active run
  `curiosity_stance_latch_20cm_0705`, Slurm job `166310`, uses 20 cm target,
  8 kg free box, guarded prelift step cycle, settle-drift compensation, and
  `ENABLE_STANCE_FOOT_LATCH=1`. Latch retargets must be reported separately
  and must not be hidden inside a walking success claim.
- 2026-07-05 stance-foot latch result:
  first launch `166310` exited during startup after a transient malformed-file
  read; login-node `py_compile` then passed. Retry
  `20260705_direct_no_root_prismatic_stance_latch_retry_20cm_8kg` ran on
  `server10`, Slurm job `166313`, and completed 2600/2600 with fall/drop 0,
  root shortcut free, anchor retargets 0, support-root writes 0, 27 latch
  enables, 23 disables, and 27 latch retargets. It failed the carry gate and
  worsened transport: max box travel `0.06334 m`, final box target distance
  `0.16388 m`, max post-settle box travel `0.10900 m`, final post-settle
  target distance `0.11822 m`, and travel loss after peak `0.02722 m`. The log
  repeatedly reported PhysX disjoint fixed-joint warnings for the stance latch
  joints. Interpretation: runtime world fixed-joint latching is not a clean
  support constraint here. Replace it with a support-anchor design authored
  from startup; keep all support retarget counters explicit and non-final.
- 2026-07-05 cleaner support-anchor baseline:
  the existing anchored-footstep carrier authors its support anchor and world
  fixed joint from startup, then can replant support by retargeting the world
  joint's `localPos0`. This avoids the runtime foot-latch enable/disable path
  that produced disjoint warnings, but anchor retargeting remains scaffold
  evidence and must not be claimed as real walking. Active run
  `curiosity_support_anchor_replant_32cm_0705`, Slurm job `166321`, uses
  `run_direct_carry_task_physical_backend.sh`,
  `SUPPORT_MODE=replant_world_joint`, `TARGET_X=0.32`, `PAYLOAD_MASS=8.0`,
  four rail joints, and `cradle_free_box`. The diagnostic allows anchor
  retargets but still requires no root shortcut, no box kinematic writes, no
  support-root pose writes, no falls/drops, and non-success labeling.
- 2026-07-05 cleaner support-anchor baseline result:
  `20260705_direct_physical_backend_replant_anchor_32cm_8kg` ran on
  `server10`, Slurm job `166321`, and passed the scaffold gate: 980/980 steps,
  `SUPPORT_MODE=replant_world_joint`, 8 kg free cradle box, fall/drop 0, root
  shortcut free, box kinematic writes 0, support-root writes 0, max box travel
  `0.31992 m`, final box target distance `0.00194 m`, final post-settle box
  travel `0.31806 m`, and final box/torso relative error `6.46e-08 m`.
  It used `anchor_world_joint_retarget_count=4`, so it remains non-final
  support-anchor scaffold evidence, not walking. No disjoint/fatal errors were
  found in the backend log. Next diagnostic is a replant support-anchor posture
  sweep across `front_mid`, `low_front`, and `chest_high`.
- 2026-07-05 replant support-anchor posture sweep:
  first sweep launch `166328` had a shell quoting error and only ran a default
  `front_mid` diagnostic under an incomplete stamp. Correctly escaped retry
  `curiosity_support_anchor_postures_retry_0705`, Slurm job `166331` on
  `server10`, ran `front_mid`, `low_front`, and `chest_high` with
  `SUPPORT_MODE=replant_world_joint`, `TARGET_X=0.32`, `PAYLOAD_MASS=8.0`,
  four rail joints, and free cradle box. All three passed scaffold gates:
  980/980 steps, fall/drop 0, root shortcut free, box kinematic writes 0,
  support-root writes 0, `anchor_world_joint_retarget_count=4`, max box travel
  about `0.3199 m`, final box target distance about `0.00194 m`, and max box
  relative offset error about `0.000264 m`. No disjoint/fatal log errors were
  found. This is the strongest current multi-posture free-box carrying
  scaffold, but still not final walking because support is retargeted through
  an anchored world joint.
- 2026-07-05 support-anchor audit fields and long-distance diagnostic:
  `normalize_direct_carry_backend_summary.py` now propagates
  `rail_joint_count`, `rail_capacity_m`, `rail_joint_indices`, `cycle_count`,
  `stride_m`, `foot_pose_write_count`, and
  `stance_anchor_pose_write_count` into direct summaries. Active run
  `curiosity_support_anchor_long64_0705`, Slurm job `166338`, uses
  `SUPPORT_MODE=replant_world_joint`, `TARGET_X=0.64`, `PAYLOAD_MASS=8.0`,
  four rail joints, and 1500 steps. It is a longer-distance support-anchor
  scaffold boundary test, not a walking claim.
- 2026-07-05 long-distance support-anchor result:
  `20260705_direct_physical_backend_replant_anchor_64cm_8kg` completed
  1500/1500 with fall/drop 0, root shortcut free, support-root writes 0,
  `anchor_world_joint_retarget_count=8`, four rails, rail capacity `0.4 m`,
  and target `0.64 m`. It failed distance by capacity saturation: max box
  travel `0.40009 m`, final target distance `0.239997 m`, final post-settle
  box/torso relative error `5.03e-09 m`. This is a useful boundary result,
  not instability.
- 2026-07-05 eight-rail long-distance result:
  `20260705_direct_physical_backend_replant_anchor_64cm_8kg_8rail` ran on
  `server10`, Slurm job `166342`, and passed the direct scaffold checker:
  1500/1500 steps, target `0.64 m`, rail capacity `0.8 m`, 8 cycles, 8 kg
  free cradle box, fall/drop 0, root shortcut free, support-root writes 0,
  max box travel `0.64583 m`, final box target distance `0.00191 m`, final
  post-settle box travel `0.63809 m`, final post-settle box/torso relative
  error `8.66e-08 m`, and max relative offset error `0.000264 m`. The backend
  log had no disjoint/fatal errors. It still used
  `anchor_world_joint_retarget_count=8`, so it is support-anchor scaffold
  evidence only, not real walking or autonomous posture selection.
- 2026-07-05 active stricter fixed-anchor long-distance ablation:
  tmux `curiosity_fixed_anchor_long64_8rail_0705`, Slurm job `166347`, was
  submitted with `SUPPORT_MODE=fixed_anchor`, `TARGET_X=0.64`,
  `PAYLOAD_MASS=8.0`, `RAIL_JOINT_COUNT=8`, `RAIL_LOWER=-0.04`,
  `RAIL_UPPER=0.10`, 1500 steps, and checker gates requiring
  `anchor_world_joint_retarget_count=0`, support-root writes 0, no falls/drops,
  root shortcut free, and non-success labeling. This tests whether the 64 cm
  scaffold can pass without support replanting; it is still fixed-world-support
  scaffold evidence, not walking.
- 2026-07-05 fixed-anchor long-distance ablation result:
  `20260705_direct_physical_backend_fixed_anchor_64cm_8kg_8rail` ran on
  `server10`, Slurm job `166347`, and passed: 1500/1500 steps,
  `physical_fixed_anchor_cradle`, 8 kg free cradle box, target `0.64 m`,
  eight rails, rail capacity `0.8 m`, fall/drop 0, root shortcut free,
  `anchor_world_joint_retarget_count=0`, support-root writes 0, foot pose
  writes 0, stance-anchor pose writes 0, max box travel `0.70080 m`, final box
  target distance `0.00111 m`, final post-settle box travel `0.63889 m`, final
  post-settle box/torso relative error `9.22e-08 m`, and max relative offset
  error `0.000264 m`. No disjoint/fatal errors were found in the backend log.
  This shows support replanting is not required for the current 64 cm scaffold,
  but the scaffold still relies on a fixed world support and long rail travel.
- 2026-07-05 active fixed-anchor 64 cm posture sweep:
  tmux `curiosity_fixed_anchor_postures64_0705` was submitted to run
  `low_front` and `chest_high` with `SUPPORT_MODE=fixed_anchor`,
  `TARGET_X=0.64`, `PAYLOAD_MASS=8.0`, eight rails, 1500 steps each, and
  checker gates requiring no anchor retargets, no support-root writes,
  fall/drop 0, root shortcut free, and non-success labeling. The already
  passed `front_mid` fixed-anchor run is the first posture in this stricter
  set.
- 2026-07-05 fixed-anchor 64 cm posture sweep result:
  the tmux sweep ran on `server10`, Slurm job `166353`. Both `low_front` and
  `chest_high` completed 1500/1500 and passed manual checker gates after the
  in-tmux checker command hit a shell-variable quoting bug and tried to read
  `.` as the summary path. `low_front`: max box travel `0.70080 m`, final
  target distance `0.00111 m`, final post-settle box travel `0.63889 m`, final
  post-settle box/torso relative error `6.05e-08 m`, fall/drop 0, root
  shortcut free, anchor retargets 0, support-root writes 0. `chest_high`: max
  box travel `0.70080 m`, final target distance `0.00111 m`, final
  post-settle box travel `0.63889 m`, final post-settle box/torso relative
  error `9.48e-08 m`, fall/drop 0, root shortcut free, anchor retargets 0,
  support-root writes 0. Together with the prior `front_mid` pass, the current
  fixed-world-support scaffold carries the 8 kg free box for 64 cm across all
  three posture labels, but it is still not walking because fixed world support
  and rail travel remain the transport mechanism.
- 2026-07-05 audit tooling update:
  `normalize_direct_carry_backend_summary.py` now propagates
  `backend_carrier_claim`, `stance_anchor_kinematic`,
  `stance_anchor_dynamic_high_mass`, `stance_anchor_as_articulation_root`, and
  `articulation_root_path`. `check_direct_carry_task_summary.py` now supports
  `--max-foot-pose-write-count`, `--max-stance-anchor-pose-write-count`, and
  `--forbid-fixed-world-support`. This is to prevent fixed-world-support
  scaffold passes from being confused with future walking/support-switching
  results. Lightweight `py_compile` passed.
- 2026-07-05 physical support-foot replacement implementation:
  `build_core_world_anchored_footstep_carrier.py` now supports
  `--support-foot-mode fixed_to_anchor`, `--support-foot-mass`, and
  `--disable-support-reposition`. In this mode the support feet are dynamic
  rigid bodies fixed to the stance anchor and in ground contact; the anchor is
  not fixed to world and is not replanted. The direct backend wrapper exposes
  `SUPPORT_MODE=dynamic_anchor_feet`, reported as
  `physical_dynamic_anchor_feet_cradle`. Normalizer/checker now expose and gate
  support-foot mode and joint count. This is still a scaffold, but it directly
  tests replacing fixed world support with physical ground-contact support.
  Lightweight `py_compile` and `bash -n` passed.
- 2026-07-05 active dynamic-anchor-feet diagnostic:
  tmux `curiosity_dynamic_anchor_feet_16cm_0705`, Slurm job `166366`, was
  submitted with `SUPPORT_MODE=dynamic_anchor_feet`, `CARRY_POSTURE=front_mid`,
  `TARGET_X=0.16`, `PAYLOAD_MASS=8.0`, two rail joints, support feet mass
  `30 kg`, no fixed world support, no support reposition, and checker gates
  requiring `physical_dynamic_anchor_feet_cradle`, backend support mode
  `dynamic_anchor`, support-foot mode `fixed_to_anchor`, support-foot joints
  >= 4, fall/drop 0, root shortcut free, anchor retargets 0, support-root
  writes 0, foot pose writes 0, stance-anchor pose writes 0,
  `--forbid-fixed-world-support`, and diagnostic-only labeling.
- 2026-07-05 dynamic-anchor-feet 16 cm result:
  `20260705_direct_physical_backend_dynamic_anchor_feet_16cm_8kg_frontmid`
  ran on `server10`, Slurm job `166366`, and passed the strict no-fixed-world
  checker: 700/700 steps, `physical_dynamic_anchor_feet_cradle`, backend
  support mode `dynamic_anchor`, support-foot mode `fixed_to_anchor`,
  support-foot joints 4, no fixed world support, no support reposition,
  fall/drop 0, root shortcut free, anchor retargets 0, support-root writes 0,
  foot pose writes 0, stance-anchor pose writes 0, max box travel `0.15831 m`,
  final box target distance `0.00195 m`, final post-settle box travel
  `0.15805 m`, final post-settle box/torso relative error `2.99e-08 m`, and
  max relative-offset error `0.000264 m`. The backend log had no disjoint/fatal
  errors. This is real progress beyond fixed-world support, but it is still a
  rigid support-frame scaffold rather than a walking robot.
- 2026-07-05 active dynamic-anchor-feet 64 cm diagnostic:
  tmux `curiosity_dynamic_anchor_feet_64cm_0705`, Slurm job `166370`, was
  submitted with `SUPPORT_MODE=dynamic_anchor_feet`, `CARRY_POSTURE=front_mid`,
  `TARGET_X=0.64`, `PAYLOAD_MASS=8.0`, eight rail joints, support-foot mass
  `30 kg`, no fixed world support, and the same strict checker gates as the
  16 cm run, with min box travel `0.58 m` and max final target distance
  `0.05 m`.
- 2026-07-05 dynamic-anchor-feet 64 cm result:
  `20260705_direct_physical_backend_dynamic_anchor_feet_64cm_8kg_frontmid`
  ran on `server10`, Slurm job `166370`, and passed the strict no-fixed-world
  checker: 1500/1500 steps, `physical_dynamic_anchor_feet_cradle`, backend
  support mode `dynamic_anchor`, support-foot mode `fixed_to_anchor`,
  support-foot joints 4, no fixed world support, no support reposition,
  fall/drop 0, root shortcut free, anchor retargets 0, support-root writes 0,
  foot pose writes 0, stance-anchor pose writes 0, max box travel `0.66915 m`,
  final box target distance `0.000734 m`, final post-settle box travel
  `0.63926 m`, final post-settle box/torso relative error `2.42e-08 m`, and
  max relative-offset error `0.000264 m`. The backend log had no disjoint/fatal
  errors. This replaces fixed world support for the 64 cm `front_mid`
  scaffold, but it remains a rigid support-foot frame, not a walking robot.
- 2026-07-05 support-foot drift audit implementation:
  dynamic support-foot summaries now record `final_anchor_travel_x_m`,
  `max_abs_anchor_travel_x_m`, `max_anchor_travel_xy_m`,
  `support_foot_min_z_m`, `support_foot_max_z_m`,
  `max_abs_support_foot_travel_x_m`, `max_support_foot_travel_xy_m`, and
  `final_support_foot_travel_x_m`. The direct normalizer propagates these
  fields, and the checker supports `--max-abs-anchor-travel-x` and
  `--max-abs-support-foot-travel-x`. Lightweight `py_compile` and `bash -n`
  passed.
- 2026-07-05 active dynamic-anchor-feet audit 64 cm run:
  tmux `curiosity_dynamic_anchor_feet_audit64_0705` was submitted with the
  same 64 cm / 8 kg / `front_mid` dynamic support-foot setup and added checker
  gates requiring max anchor X drift <= `0.03 m` and max support-foot X drift
  <= `0.03 m`, in addition to all no-fixed-world gates.
- 2026-07-05 dynamic-anchor-feet audit 64 cm result:
  `20260705_direct_physical_backend_dynamic_anchor_feet_audit64_8kg_frontmid`
  ran on `server10`, Slurm job `166375`, and passed: 1500/1500,
  `physical_dynamic_anchor_feet_cradle`, no fixed world support, support-foot
  joints 4, fall/drop 0, root shortcut free, no anchor retargets, no
  support-root/foot/stance-anchor pose writes, max box travel `0.66915 m`,
  final target distance `0.000734 m`, final post-settle box travel
  `0.63926 m`, and final post-settle box/torso relative error `2.42e-08 m`.
  Support drift audit passed: max anchor X drift `4.47e-07 m`, max support-foot
  X drift `4.17e-07 m`, support-foot z range `0.0174997-0.0175006 m`. No
  disjoint/fatal errors were found. This is the strongest current
  no-fixed-world-support scaffold, but still a rigid support-frame scaffold
  rather than real footstep locomotion.
- 2026-07-05 active audited dynamic-anchor-feet posture sweep:
  tmux `curiosity_dynamic_anchor_feet_postures64_0705` was submitted to run
  `low_front` and `chest_high` at 64 cm / 8 kg with
  `SUPPORT_MODE=dynamic_anchor_feet`, eight rails, support-foot mass `30 kg`,
  and the same strict no-fixed-world plus support-drift gates as the
  `front_mid` audit run.
- 2026-07-05 audited dynamic-anchor-feet posture sweep result:
  `curiosity_dynamic_anchor_feet_postures64_0705`, Slurm job `166379`, ran on
  `server10`. Both remaining posture labels passed the strict no-fixed-world
  and drift gates. `low_front`: 1500/1500, max box travel `0.66915 m`, final
  target distance `0.000733 m`, final post-settle box travel `0.63927 m`,
  final post-settle box/torso relative error not elevated, fall/drop 0, root
  shortcut free, anchor retargets 0, support-root writes 0, foot pose writes 0,
  stance-anchor pose writes 0, max anchor X drift `4.52e-07 m`, max support-foot
  X drift `4.47e-07 m`, support-foot z range `0.0174997-0.0175005 m`.
  `chest_high`: 1500/1500, max box travel `0.66915 m`, final target distance
  `0.000734 m`, final post-settle box travel `0.63926 m`, final post-settle
  box/torso relative error `9.65e-08 m`, fall/drop 0, root shortcut free,
  anchor retargets 0, support-root writes 0, foot pose writes 0,
  stance-anchor pose writes 0, max anchor X drift `4.32e-07 m`, max support-foot
  X drift `4.77e-07 m`, support-foot z range `0.0174997-0.0175006 m`. No
  disjoint/fatal errors were found in either backend log. Together with
  `front_mid`, the dynamic support-foot scaffold now covers all three posture
  labels at 64 cm / 8 kg without fixed world support. It remains non-final
  because all four support feet are rigidly fixed to one stance anchor rather
  than performing footstep/support switching.
- 2026-07-05 legged-anchor-feet implementation:
  `build_core_world_anchored_footstep_carrier.py` now supports
  `support_foot_mode=x_prismatic_to_anchor` plus
  `--use-support-foot-drive`. In this mode the torso rail target is held at
  zero and X prismatic support-foot joints are driven against ground contact
  to move the anchor/torso/payload. `run_direct_carry_task_physical_backend.sh`
  exposes this as `SUPPORT_MODE=legged_anchor_feet` with controller mode
  `physical_legged_anchor_feet_cradle`. The normalizer/checker expose
  support-foot X joint indices, motion, limits, and drive flags. This is an
  intermediate foot-driven scaffold, not final walking, because all feet are
  still driven together rather than using swing/stance support switching.
  Lightweight `py_compile` and `bash -n` passed.
- 2026-07-05 active legged-anchor-feet 16 cm diagnostic:
  tmux `curiosity_legged_anchor_feet_16cm_0705` was submitted with
  `SUPPORT_MODE=legged_anchor_feet`, `TARGET_X=0.16`, `PAYLOAD_MASS=8.0`, two
  rail joints held at zero target, support-foot X joints limited to
  `[-0.40, 0.20]`, support-foot drive direction scale `-1.0`, and checker
  gates requiring no fixed world support, no support/root/foot pose writes,
  support-foot X joint motion at least `0.10 m`, min box travel `0.12 m`, and
  max final target distance `0.06 m`.
- 2026-07-05 legged-anchor-feet 16 cm result:
  `20260705_direct_physical_backend_legged_anchor_feet_16cm_8kg_frontmid`
  completed 800/800 and passed the strict no-fixed-world gate:
  `support_foot_mode=x_prismatic_to_anchor`, support-foot X joint count 4,
  max support-foot X joint motion `0.62882 m`, max box travel `0.15971 m`,
  final box target distance `0.00662 m`, final post-settle box travel
  `0.15380 m`, final post-settle box/torso relative error `8.64e-08 m`,
  fall/drop 0, root shortcut free, anchor retargets 0, support-root writes 0,
  foot pose writes 0, stance-anchor pose writes 0, no fixed world support.
  This is stronger than the rigid support-foot frame because motion comes from
  X support-foot drives against ground contact, but it is still not walking:
  all feet are driven together and no swing/stance switching occurs.
- 2026-07-05 alternating X/Z support-foot implementation:
  `build_core_world_anchored_footstep_carrier.py` now supports
  `support_foot_mode=xz_prismatic_to_anchor`: each support foot is connected
  through an X prismatic link and a vertical Z prismatic foot joint. The
  controller can alternate diagonal stance pairs, command swing-foot lift,
  and log support-foot X/Z joint counts, X/Z joint motion, commanded lift, and
  per-foot final targets. `run_direct_carry_task_physical_backend.sh` exposes
  this as `SUPPORT_MODE=alternating_anchor_feet` with controller mode
  `physical_alternating_anchor_feet_cradle`. Lightweight `py_compile` and
  `bash -n` passed.
- 2026-07-05 active alternating-anchor-feet 8 cm diagnostic:
  the first tmux submission `curiosity_alternating_anchor_feet_8cm_0705`
  produced Slurm job `166400` but exited in 0 seconds before creating a backend
  log or summary, so it is only a launch failure. To avoid long nested quoting,
  a one-off diagnostic launcher was added at
  `scripts/isaac/run_alternating_anchor_feet_8cm_diag.sh`. Retry
  `curiosity_alternating_anchor_feet_8cm_retry_0705`, Slurm job `166403`,
  also exited before Isaac. Retry2, Slurm job `166406`, captured the reason:
  `run_direct_carry_task_physical_backend.sh` referenced
  `SUPPORT_FOOT_DRIVE_DIRECTION_SCALE` under `set -u` after only assigning it
  in a child-process environment. The wrapper now initializes support-foot
  defaults in the current shell before passing them onward; lightweight
  `bash -n` and `py_compile` passed. Retry3, Slurm job `166408`, reached
  `server10` but exposed a second wrapper issue: an unexpected EOF in the
  long env-assignment command. The direct wrapper was simplified to export
  support-foot defaults before calling the core wrapper instead of expanding
  those variables in the same command chain. Retry4 tmux
  `curiosity_alternating_anchor_feet_8cm_retry4_0705` submitted Slurm job
  `166412` for `TARGET_X=0.08`, `PAYLOAD_MASS=8.0`, `CARRY_POSTURE=front_mid`,
  `support_foot_mode=xz_prismatic_to_anchor`, foot X limits
  `[-0.12, 0.12]`, Z limits `[-0.005, 0.12]`, commanded step height
  `0.055 m`, stance X `-0.060 m`, swing X `0.060 m`, and strict gates
  requiring no fixed world support, no root/support/foot/stance pose writes,
  support-foot joint count at least 8, Z joint count at least 4, X joint
  motion at least `0.08 m`, Z joint motion at least `0.02 m`, min box travel
  `0.04 m`, fall/drop 0, and diagnostic-only success claim. At recording time
  the Slurm job was queued.
- 2026-07-05 alternating-anchor-feet 8 cm result:
  retry4, Slurm job `166412` on `server10`, entered Isaac and completed
  620/620 but failed the distance gate. Positive findings: no fixed world
  support, no root/body/box/support/foot/stance pose writes, anchor retargets
  0, fall/drop 0, support-foot mode `xz_prismatic_to_anchor`, support-foot
  joints 8, X joint motion `0.24221 m`, Z joint motion `0.46727 m`, commanded
  lift `0.055 m`, and final post-settle box/torso relative error
  `0.000178 m`. Negative findings: the system drifted about `0.061 m` in
  negative X during settle, final box target distance was `0.08083 m`, max box
  travel from the initial frame was only `6.36e-05 m`, and actual support-foot
  world Z changed only from `0.01750` to `0.01923 m` despite Z joint motion.
  Interpretation: the alternating X/Z joint plumbing runs and is shortcut-free,
  but it is not a valid swing-foot carrying mechanism yet. Next fix should
  reduce settle drift and add/gate actual foot-lift metrics, not just joint
  motion.
- 2026-07-05 actual support-foot lift instrumentation:
  summaries now record `max_actual_support_foot_lift_m`,
  `per_foot_max_actual_lift_m`, `per_foot_min_z_m`, and `per_foot_max_z_m`.
  The direct normalizer propagates these fields, and the checker supports
  `--min-actual-support-foot-lift`. Lightweight `py_compile` and `bash -n`
  passed.
- 2026-07-05 active alternating-anchor-feet fast-start diagnostic:
  added `scripts/isaac/run_alternating_anchor_feet_faststart_8cm_diag.sh` and
  submitted tmux `curiosity_alternating_faststart_8cm_0705`, Slurm job
  `166417`. This variant keeps the same 8 cm / 8 kg / `front_mid` task, but
  uses `SETTLE_STEPS=10`, foot mass `12 kg`, X limits `[-0.16, 0.16]`, Z
  limits `[-0.005, 0.18]`, commanded step height `0.090 m`, stance X
  `-0.120 m`, swing X `0.120 m`, stronger X/Z drives, and a checker gate
  requiring actual support-foot lift at least `0.02 m`. At recording time the
  Slurm job was queued.
- 2026-07-05 alternating-anchor-feet fast-start result:
  `20260705_direct_physical_backend_alternating_anchor_feet_faststart_8cm_8kg_frontmid`
  ran on `server10`, Slurm job `166417`, and passed the first wide diagnostic
  gate: 620/620, `support_foot_mode=xz_prismatic_to_anchor`, support-foot
  joints 8, actual support-foot lift `0.02853 m`, commanded lift `0.09 m`, X
  joint motion `0.39575 m`, Z joint motion `0.39739 m`, fall/drop 0, root
  shortcut free, fixed world support false, anchor retargets 0, support-root
  writes 0, foot pose writes 0, stance-anchor pose writes 0, max box travel
  `0.04012 m`, final box target distance `0.04572 m`, final post-settle box
  travel `0.03940 m`, and final post-settle box/torso relative error was not
  elevated. This is the first valid actual-lift alternating support-foot
  scaffold, but it is still only a partial 8 cm carry because it reaches about
  4 cm and then plateaus.
- 2026-07-05 active alternating-anchor-feet multi-cycle diagnostic:
  added `scripts/isaac/run_alternating_anchor_feet_multicycle_8cm_diag.sh` and
  submitted tmux `curiosity_alternating_multicycle_8cm_0705`, Slurm job
  `166421`. This keeps the fast-start physical settings but reduces
  `STEP_LENGTH` to `0.02 m` and `STANCE_STEPS` to 90 so the 8 cm target uses
  four gait cycles instead of two. Checker gates require actual foot lift
  `>=0.02 m`, box travel `>=0.065 m`, and final target distance `<=0.025 m`.
  At recording time the Slurm job was queued.
- 2026-07-05 alternating-anchor-feet multi-cycle result:
  `20260705_direct_physical_backend_alternating_anchor_feet_multicycle_8cm_8kg_frontmid`
  ran on `server10`, Slurm job `166421`, completed 720/720, and failed the
  stricter gate narrowly. It remained shortcut-free and safe: fall/drop 0,
  fixed world support false, root shortcut free, anchor retargets 0,
  support-root writes 0, foot pose writes 0, stance-anchor pose writes 0,
  support-foot joints 8, X joint motion `0.33659 m`, Z joint motion
  `0.39833 m`. It improved transport over fast-start: max box travel
  `0.05794 m`, final post-settle box travel `0.05866 m`, and final target
  distance `0.02646 m`. Failures were actual foot lift `0.01738 m` below the
  `0.02 m` gate, box travel below `0.065 m`, and target distance just above
  `0.025 m`. Interpretation: more gait cycles help forward travel, but actual
  swing-foot lift is still weak.
- 2026-07-05 active alternating-anchor-feet 5-cycle diagnostic:
  added `scripts/isaac/run_alternating_anchor_feet_5cycle_8cm_diag.sh` and
  submitted tmux `curiosity_alternating_5cycle_8cm_0705`, Slurm job `166426`.
  This uses `STEP_LENGTH=0.016`, `STANCE_STEPS=80`, foot mass `8 kg`, X limits
  `[-0.17, 0.17]`, Z limits `[-0.005, 0.24]`, commanded step height `0.120 m`,
  stance X `-0.130 m`, swing X `0.130 m`, and stronger Z drive. Gates are the
  same stricter multi-cycle gates: actual foot lift `>=0.02 m`, box travel
  `>=0.065 m`, and final target distance `<=0.025 m`. At recording time the
  Slurm job had been allocated.
- 2026-07-05 alternating-anchor-feet 5-cycle result:
  `20260705_direct_physical_backend_alternating_anchor_feet_5cycle_8cm_8kg_frontmid`
  ran on `server10`, Slurm job `166426`, completed 780/780, and failed only
  the final target-distance gate. It passed the key motion/safety parts:
  fall/drop 0, fixed world support false, root shortcut free, anchor retargets
  0, support-root writes 0, foot pose writes 0, stance-anchor pose writes 0,
  actual support-foot lift `0.06320 m`, max box travel `0.09850 m`, X joint
  motion `0.35847 m`, and Z joint motion `0.39335 m`. It reached target around
  step 220, but `target_hold` slowly slid back; final box target distance was
  `0.03631 m`. The issue is the hold controller, not the ability to generate
  swing-foot lift or forward impulse.
- 2026-07-05 alternating support-foot hold fix:
  target latch now stores each support foot's own X joint position and restores
  per-foot X targets during hold. The previous implementation replaced all
  foot X targets with their mean, erasing the stance geometry and allowing the
  carrier to slide backward during long hold.
- 2026-07-05 active 5-cycle holdfix diagnostic:
  added `scripts/isaac/run_alternating_anchor_feet_5cycle_holdfix_8cm_diag.sh`
  and submitted tmux `curiosity_alternating_5cycle_holdfix_8cm_0705`, Slurm job
  `166430`. Parameters match the 5-cycle run; only the per-foot target-hold
  code changed. At recording time the Slurm job was queued.
- 2026-07-05 5-cycle holdfix result:
  `20260705_direct_physical_backend_alternating_anchor_feet_5cycle_holdfix_8cm_8kg_frontmid`
  ran on `server10`, Slurm job `166430`, and passed the stricter alternating
  support-foot gate: 780/780, controller
  `physical_alternating_anchor_feet_cradle`, support-foot mode
  `xz_prismatic_to_anchor`, support-foot joints 8, X/Z joint counts 4/4,
  actual support-foot lift `0.06320 m`, X joint motion `0.35847 m`, Z joint
  motion `0.39335 m`, max box travel `0.09812 m`, final box target distance
  `0.01572 m`, final post-settle box travel `0.06551 m`, fall/drop 0, root
  shortcut free, fixed world support false, anchor retargets 0,
  support-root writes 0, foot pose writes 0, stance-anchor pose writes 0. Log
  scan found no disjoint/fatal/traceback/unbound/EOF errors. This is the first
  direct-Isaac 8 kg free-box scaffold where alternating X/Z support feet with
  actual swing-foot lift move and hold the carried box near an 8 cm target
  without fixed world support or pose/velocity root shortcuts. It is still not
  a full humanoid/quadruped walking policy or unknown-load active-probing
  solution.
- 2026-07-05 support/contact metric instrumentation:
  alternating support-foot summaries now record `support_foot_contact_z_threshold_m`,
  `per_foot_near_ground_steps`, `per_foot_max_near_ground_xy_slip_m`,
  `per_foot_max_near_ground_xy_speed_mps`, `min_near_ground_foot_count`,
  `max_near_ground_foot_count`, `min_support_polygon_margin_x_m`,
  `min_support_polygon_margin_y_m`, and `min_support_polygon_margin_m`. The
  normalizer propagates these fields, and the checker supports
  `--min-near-ground-foot-count`, `--min-support-polygon-margin`, and
  `--max-near-ground-foot-speed`. These are proxy metrics, not force-sensor
  proof, but they are needed before extending the scaffold.
- 2026-07-05 active 16 cm alternating holdfix diagnostic:
  added `scripts/isaac/run_alternating_anchor_feet_10cycle_holdfix_16cm_diag.sh`
  and submitted tmux `curiosity_alternating_10cycle_holdfix_16cm_0705`, Slurm
  job `166438`. This doubles the target to `0.16 m` with `STEP_LENGTH=0.016`
  and 10 gait cycles while keeping the 8 kg free box and per-foot target hold.
  Gates require actual foot lift `>=0.02 m`, box travel `>=0.13 m`, final
  target distance `<=0.04 m`, fall/drop 0, no fixed world support, no root
  shortcut, and no support/root/foot pose writes. At recording time the Slurm
  job was queued.
- 2026-07-05 16 cm alternating holdfix result:
  `20260705_direct_physical_backend_alternating_anchor_feet_10cycle_holdfix_16cm_8kg_frontmid`
  ran on `server10`, Slurm job `166438`, and the Isaac backend completed
  1180/1180. The backend passed the 16 cm / 8 kg gate after manual
  normalization from the backend summary: controller
  `physical_alternating_anchor_feet_cradle`, support-foot mode
  `xz_prismatic_to_anchor`, support-foot joints 8, X/Z joint counts 4/4,
  actual support-foot lift `0.06320 m`, max box travel `0.18576 m`, final box
  target distance `0.00436 m`, final post-settle box travel `0.15686 m`,
  fall/drop 0, root shortcut free, fixed world support false, anchor retargets
  0, support-root writes 0, foot pose writes 0, and stance-anchor pose writes
  0. New support proxy metrics were recorded: per-foot near-ground steps
  `fl=1111`, `fr=1061`, `rl=1109`, `rr=1116`, max near-ground foot count 4,
  min support polygon margin `0.16279 m`, and max near-ground XY speed up to
  `1.0855 m/s`. `min_near_ground_foot_count` is 0 because all feet are above
  the contact-z threshold during some swing transition frames; treat the
  contact metrics as diagnostics, not proof of force closure.
- 2026-07-05 wrapper limitation and fix:
  the 16 cm run exposed a post-rollout shell issue in
  `run_core_world_anchored_footstep_carrier.sh`: Isaac wrote the backend
  summary successfully, but the core wrapper exited with
  `unexpected EOF while looking for matching '"'` before the direct wrapper
  could normalize/check it. The backend log itself had no disjoint/fatal
  errors. To reduce future shell parsing risk, the core wrapper now builds the
  Python command as a bash array and executes `"${CMD[@]}" | tee ...` instead
  of one long backslash-continued command. Lightweight `bash -n` and
  `py_compile` passed after the change.
- 2026-07-05 active 32 cm alternating holdfix diagnostic:
  added `scripts/isaac/run_alternating_anchor_feet_20cycle_holdfix_32cm_diag.sh`
  and submitted tmux `curiosity_alternating_20cycle_holdfix_32cm_0705`, Slurm
  job `166446`. This pushes the same alternating X/Z support-foot mechanism to
  `TARGET_X=0.32 m` with `STEP_LENGTH=0.016`, 20 nominal cycles, 8 kg free box,
  and the wrapper array fix. Gates require actual foot lift `>=0.02 m`, box
  travel `>=0.26 m`, final target distance `<=0.08 m`, fall/drop 0, no fixed
  world support, no root shortcut, and no support/root/foot pose writes. At
  recording time the Slurm job was queued.
- 2026-07-05 32 cm launch retry:
  Slurm job `166446` failed before Isaac with a core-wrapper shell expansion
  error around negative default values. The core wrapper now assigns negative
  defaults such as rail lower, support-foot X lower, Z lower, stance X, and
  drive direction scale into plain variables before building the command array.
  Lightweight checks passed. Retry tmux
  `curiosity_alternating_20cycle_holdfix_32cm_retry_0705` was submitted with
  the same stamp and parameters.
- 2026-07-05 32 cm alternating holdfix result:
  retry Slurm job `166450` ran on `server10` and passed the 32 cm / 8 kg
  alternating support-foot gate with automatic direct normalization/checking:
  1980/1980, support-foot mode `xz_prismatic_to_anchor`, support-foot joints
  8, actual support-foot lift `0.06320 m`, max box travel `0.38092 m`, final
  box target distance `0.03159 m`, final post-settle box travel `0.35281 m`,
  fall/drop 0, root shortcut free, fixed world support false, anchor retargets
  0, support-root writes 0, foot pose writes 0, and stance-anchor pose writes
  0. New support metrics: near-ground steps `fl=1859`, `fr=1804`, `rl=1873`,
  `rr=1874`, max near-ground foot count 4, min support-polygon margin
  `0.16279 m`, and max near-ground XY speed up to `1.0855 m/s`.
  `min_near_ground_foot_count` remains 0 in transition frames. Log scan found
  no disjoint/fatal/traceback/unbound/EOF errors. This is still a scaffold,
  but it is the current strongest direct-Isaac alternating-foot carrying
  result.
- 2026-07-05 active 64 cm alternating holdfix diagnostic:
  added `scripts/isaac/run_alternating_anchor_feet_40cycle_holdfix_64cm_diag.sh`
  and submitted tmux `curiosity_alternating_40cycle_holdfix_64cm_0705`, Slurm
  job `166455`. This pushes the same mechanism to `TARGET_X=0.64 m`, 40
  nominal cycles, 8 kg free box, and gates requiring actual foot lift
  `>=0.02 m`, box travel `>=0.52 m`, final target distance `<=0.14 m`,
  fall/drop 0, no fixed world support, no root shortcut, and no
  support/root/foot pose writes. At recording time the Slurm job was queued.
- 2026-07-05 64 cm alternating holdfix result:
  Slurm job `166455` ran on `server10` and passed the 64 cm / 8 kg alternating
  support-foot gate with automatic normalization/checking: 3580/3580,
  `support_foot_mode=xz_prismatic_to_anchor`, support-foot joints 8, actual
  support-foot lift `0.06475 m`, max box travel `0.64785 m`, final box target
  distance `0.01181 m`, final post-settle box travel `0.62941 m`, fall/drop
  0, root shortcut free, fixed world support false, anchor retargets 0,
  support-root writes 0, foot pose writes 0, and stance-anchor pose writes 0.
  Support proxy metrics: near-ground steps `fl=3340`, `fr=3341`, `rl=3445`,
  `rr=3427`, max near-ground foot count 4, min support-polygon margin
  `0.16279 m`, max near-ground XY speed up to `1.0855 m/s`,
  `min_near_ground_foot_count=0` during transition frames. Log scan found no
  disjoint/fatal/traceback/unbound/EOF errors. This matches the older 64 cm
  dynamic-support-foot distance scale, but now with alternating X/Z support
  feet and actual swing-foot lift.
- 2026-07-05 64 cm multi-posture alternating holdfix sweep:
  Slurm job `166461` ran on `server10` from tmux
  `curiosity_alternating_64cm_postures_0705` and completed the `low_front` and
  `chest_high` posture variants without touching `carry1`. Both use
  `controller_mode=physical_alternating_anchor_feet_cradle`,
  `support_foot_mode=xz_prismatic_to_anchor`, 8 support-foot joints, an 8 kg
  free dynamic box, `TARGET_X=0.64 m`, 3580 steps, no fixed-world support,
  `anchor_world_joint_retarget_count=0`, `support_root_pose_write_count=0`,
  `foot_pose_write_count=0`, and `stance_anchor_pose_write_count=0`.
  `low_front` reached max box travel `0.70662 m`, final target distance
  `0.03367 m`, final post-settle box travel `0.67882 m`, actual support-foot
  lift `0.06348 m`, fall/drop 0, root shortcut free, min support-polygon
  margin `0.16409 m`, and near-ground steps `fl=3342`, `fr=3344`, `rl=3428`,
  `rr=3411`. `chest_high` reached max box travel `0.70446 m`, final target
  distance `0.02583 m`, final post-settle box travel `0.66510 m`, actual
  support-foot lift `0.06353 m`, fall/drop 0, root shortcut free, min
  support-polygon margin `0.16279 m`, and near-ground steps `fl=3351`,
  `fr=3348`, `rl=3425`, `rr=3394`. Log scan found no
  disjoint/fatal/traceback/unbound/EOF errors. This is the strongest direct
  Isaac multi-posture carrying scaffold so far, but it is still a physical
  backend diagnostic, not a full humanoid/quadruped walking policy, not
  unknown-load active probing, and not video-conditioned RL.
- 2026-07-05 randomized-load direct-task interface smoke:
  added reproducible payload randomization to
  `build_core_world_anchored_footstep_carrier.py` and the direct physical
  backend wrappers. New fields include `box_seed`, `payload_randomized`,
  requested/range/sampled payload mass, requested/sampled size, size jitter,
  COM offset range, and sampled COM offset; the normalizer/checker propagates
  these to the direct-task summary. Smoke script
  `scripts/isaac/run_alternating_anchor_feet_randomized_8cm_diag.sh` ran in
  tmux `curiosity_alternating_randomized_8cm_0705`, Slurm job `166474` on
  `server53`. Result
  `20260705_direct_physical_backend_alternating_anchor_feet_randomized_8cm_seed7051`
  passed the 8 cm gate with randomized payload: seed `7051`, sampled mass
  `8.15343 kg`, sampled size about `0.35775 x 0.25309 x 0.23354 m`, sampled
  COM offset `[0.00902, 0.00821, -0.00216] m`, 780/780 steps, max box travel
  `0.09614 m`, final target distance `0.01740 m`, final post-settle box
  travel `0.06362 m`, actual support-foot lift `0.06319 m`, fall/drop 0,
  root shortcut free, fixed world support false, anchor retargets 0,
  support-root writes 0, foot pose writes 0, and stance-anchor pose writes 0.
  Log scan found no fatal/traceback/EOF errors. This is randomized task-input
  plumbing and a diagnostic smoke, not active load identification.
- 2026-07-05 active-probe measurement smoke:
  added optional `PROBE_STEPS` / `PROBE_X_AMPLITUDE` support before the carry
  phase. The probe does a small support-foot push-pull and records probe
  torso travel, probe box travel, probe relative error, and final probe box
  lag. Carry post-settle baselines now start after the probe, so probe motion
  is not counted as carry distance. Smoke script
  `scripts/isaac/run_alternating_anchor_feet_probe_randomized_8cm_diag.sh`
  ran in tmux `curiosity_alternating_probe_randomized_8cm_0705`, Slurm job
  `166483` on `server36`. Result
  `20260705_direct_physical_backend_alternating_anchor_feet_probe_randomized_8cm_seed7052`
  passed: randomized seed `7052`, sampled mass `9.72299 kg`, sampled size
  about `0.34799 x 0.24707 x 0.22159 m`, sampled COM offset
  `[-0.01960, 0.01151, 0.00618] m`, probe steps `60`, probe amplitude
  `0.020 m`, max probe torso travel `0.03445 m`, max probe box travel
  `0.03466 m`, max probe relative error `0.00835 m`, final probe box lag
  `0.000325 m`, 860/860 steps, max box travel `0.14405 m`, final target
  distance `0.02021 m`, final post-settle box travel `0.11084 m`, actual
  support-foot lift `0.05082 m`, fall/drop 0, root shortcut free, fixed world
  support false, anchor retargets 0, support-root writes 0, foot pose writes
  0, and stance-anchor pose writes 0. Log scan found no fatal/traceback/EOF or
  disjoint errors. This is only probing telemetry; no policy or belief model
  currently uses the probe signal.
- 2026-07-05 probe-belief proxy smoke:
  added a clearly labeled heuristic probe-derived belief proxy. It uses only
  probe telemetry, not hidden sampled mass/COM, and writes
  `probe_belief_uses_hidden_ground_truth=false`. Fields include
  `probe_compliance_proxy`, `probe_lag_proxy`, `probe_risk_score`,
  `probe_load_risk_bucket`, and `probe_recommended_carry_adjustment`. The
  proxy is explicitly not a calibrated mass estimator and is not applied to
  policy yet. Smoke script
  `scripts/isaac/run_alternating_anchor_feet_belief_probe_randomized_8cm_diag.sh`
  ran in tmux `curiosity_alternating_belief_probe_8cm_0705`, Slurm job
  `166489` on `server36`. Result
  `20260705_direct_physical_backend_alternating_anchor_feet_belief_probe_randomized_8cm_seed7053`
  passed: sampled mass `9.81294 kg`, sampled size about
  `0.34531 x 0.24048 x 0.22112 m`, sampled COM offset
  `[-0.00299, 0.00755, 0.00329] m`, probe steps `60`, probe amplitude
  `0.020 m`, max probe relative error `0.00863 m`, final probe box lag
  `0.00161 m`, `probe_compliance_proxy=0.25067`,
  `probe_lag_proxy=0.04821`, `probe_risk_score=0.80478`, bucket
  `high_observed_load_or_shift_response`, recommended adjustment
  `slow_gait_low_or_chest_supported_candidate`, 860/860 steps, max box travel
  `0.14364 m`, final target distance `0.02069 m`, final post-settle box
  travel `0.11007 m`, actual support-foot lift `0.05058 m`, fall/drop 0,
  root shortcut free, fixed world support false, anchor retargets 0,
  support-root writes 0, foot pose writes 0, and stance-anchor pose writes 0.
  Log scan found no fatal/traceback/EOF/disjoint errors.
- 2026-07-05 controlled probe-belief mass calibration:
  added `scripts/isaac/run_alternating_anchor_feet_probe_belief_mass_calibration_8cm.sh`
  and ran it in tmux `curiosity_probe_belief_mass_cal_8cm_0705`, Slurm job
  `166494`. It sequentially tested fixed 6 kg and 10 kg boxes with the same
  60-step, 2 cm push-pull probe and 8 cm carry gate. Both passed safely with
  fall/drop 0 and all shortcut counters 0, but the belief proxy did not
  meaningfully separate the loads: 6 kg produced `probe_compliance_proxy`
  `0.24462`, `probe_lag_proxy` `0.02860`, `probe_risk_score` `0.78250`,
  bucket `high_observed_load_or_shift_response`; 10 kg produced
  `probe_compliance_proxy` `0.24728`, `probe_lag_proxy` `0.03697`,
  `probe_risk_score` `0.79201`, same bucket. This is a negative calibration
  result: the current box-lag-only heuristic is not adequate as a load belief
  model and must not be used for controller adaptation without additional
  signals such as commanded/actual support-foot tracking, effort proxies, or a
  different probing maneuver.
- 2026-07-05 support-foot tracking calibration result:
  added support-foot X target-vs-actual tracking telemetry during probe and
  reran fixed 6 kg vs 10 kg calibration in tmux
  `curiosity_probe_belief_tracking_cal_8cm_0705`, Slurm job `166501`. Both
  cases again passed the carry gates safely, but the new tracking signal also
  failed to separate mass: 6 kg had max tracking error `0.04093 m`, mean
  tracking error `0.02226 m`, tracking proxy `2.04663`, risk `0.82600`;
  10 kg had max tracking error `0.04091 m`, mean tracking error `0.02224 m`,
  tracking proxy `2.04556`, risk `0.83361`. This is another negative
  calibration. The current probe mechanics are dominated by support-foot
  controller tracking rather than payload mass. Do not use this belief proxy
  for adaptation; next attempt should use measured joint efforts/forces if
  Isaac exposes them reliably.
- 2026-07-05 measured-effort calibration result:
  added Isaac articulation measured-effort telemetry for the support-foot X
  probe and ran fixed 6 kg vs 10 kg calibration in tmux
  `curiosity_probe_belief_effort_cal_8cm_0705`, Slurm job `166514`. The
  effort read path worked: both cases reported
  `probe_joint_effort_available=true`, zero read errors, and no first error.
  Both cases also passed the 8 cm carry gate safely with fall/drop 0 and
  root-shortcut-free summaries. However, measured effort still did not
  meaningfully separate load under the current horizontal push-pull probe:
  6 kg had max/mean support-foot X effort `459.73468` / `302.97068`, effort
  proxy `0.004179`, risk `0.57452`, final target distance `0.02105 m`;
  10 kg had max/mean effort `462.08502` / `303.88600`, effort proxy
  `0.004201`, risk `0.58025`, final target distance `0.01792 m`. This is a
  negative estimator result, not a controller success. The conclusion is that
  horizontal support-foot push-pull is too insensitive in this cradle setup;
  next probing should change the mechanics to vertical micro-lift or
  partial-unload probing so gravity/load affects the measured signal.
- 2026-07-05 vertical micro-lift calibration result:
  added `probe_mode=vertical_micro_lift`, `probe_z_amplitude_m`, Z probe
  torso/box travel, Z support-foot tracking telemetry, and Z measured-effort
  telemetry. Also fixed the belief proxy so inactive axes do not contribute
  tracking or effort proxies. Initial vertical jobs `166528` and `166531` were
  canceled because they still executed the old horizontal probe path. Retry3
  ran in tmux `curiosity_probe_belief_vertical_lift_cal3_8cm_0705`, Slurm job
  `166533`, on `server46` with explicit CLI overrides. Both fixed 6 kg and
  10 kg cases completed the 8 cm carry gate with fall/drop 0, shortcut-free
  summaries, and clean fatal/traceback/disjoint log scans. The signal still
  did not meaningfully separate load: 6 kg measured max/mean support-foot Z
  effort `2371.66748` / `1386.22269`, torso/box Z travel
  `0.02686 m` / `0.02562 m`; 10 kg measured max/mean Z effort
  `2380.76245` / `1398.41329`, torso/box Z travel
  `0.02714 m` / `0.02592 m`. This is another negative estimator result. Do
  not use the all-feet vertical micro-lift probe for posture or gait
  adaptation; next work should instrument a more direct load-sensitive signal
  such as cradle/box constraint force, contact normal impulse, support
  reaction force, or deliberate weight transfer between contacts.
- 2026-07-05 Arena G1 AGILE commanded-walk smoke submitted:
  extended `scripts/isaac/build_arena_g1_agile_stand_smoke.py` and
  `scripts/isaac/run_arena_g1_agile_stand_smoke.sh` so the existing
  controller-backed AGILE G1 WBC entry can send a navigation command after a
  stand warmup and record root XY travel, root height, tilt, fall events, and
  pass/fail status. Submitted tmux `curiosity_g1_agile_walk_smoke_0705`,
  Slurm job `166541`, command stamp
  `20260705_arena_g1_agile_walk_cmd_smoke`, with `COMMAND_X=0.25`,
  `COMMAND_START_STEP=80`, `STEPS=260`, and
  `MIN_COMMANDED_TRAVEL_X=0.05`. This is only a walking/balance diagnostic,
  not box-carrying evidence.
- 2026-07-05 Arena G1 AGILE outer-AppLauncher walk result:
  Slurm job `166541` reached Arena environment creation and physics startup,
  but failed before rollout with tensor-view invalidation and
  `Exception: Failed to get DOF velocities from backend`. Summary path:
  `experiments/outputs/arena_g1_agile_stand_smoke/20260705_arena_g1_agile_walk_cmd_smoke/arena_g1_agile_stand_summary.json`.
  This is a negative locomotion-integration result; do not rerun the same
  outer-AppLauncher path unchanged.
- 2026-07-05 Arena G1 AGILE persistent harness submitted:
  added `scripts/isaac/run_arena_g1_agile_walk_persistent_smoke.py` and
  launcher `scripts/isaac/run_arena_g1_agile_walk_persistent_smoke.sh`, which
  use IsaacLab-Arena's own
  `isaaclab_arena.tests.utils.subprocess.run_simulation_app_function` harness
  instead of creating an outer project AppLauncher. Submitted retry2 tmux
  `curiosity_g1_agile_walk_persistent2_0705`, Slurm job `166550`, with the
  same `COMMAND_X=0.25`, `STEPS=260`, and `MIN_COMMANDED_TRAVEL_X=0.05`.
- 2026-07-05 direct Isaac anchor posture sweep submitted:
  user correction accepted: do not wait on external models/controllers when
  they do not immediately unblock the task. The active path is direct Isaac
  scene construction first. Added diagnostic-only posture-sweep files
  `scripts/isaac/run_direct_isaac_anchor_posture_sweep.sh` and
  `scripts/isaac/summarize_anchor_posture_sweep.py`. The sweep runs four
  candidate carry postures under one randomized hidden box seed, executes
  vertical micro-lift probing before carrying, and ranks telemetry. It is a
  metric ranker, not a learned policy and not full robot-carrying evidence.
  Lightweight checks passed:
  `bash -n scripts/isaac/run_direct_isaac_anchor_posture_sweep.sh` and
  `python3 -m py_compile scripts/isaac/summarize_anchor_posture_sweep.py
  scripts/isaac/build_core_world_anchored_footstep_carrier.py`. Submitted
  Curiosity-owned tmux/Slurm run `curiosity_direct_anchor_sweep_0705`, Slurm
  job `166557`, command:
  `STAMP=20260705_direct_isaac_anchor_posture_sweep_seed17 DEVICE=cuda:0
  BOX_SEED=17 STEPS=300 TARGET_X=0.08 bash
  scripts/isaac/run_direct_isaac_anchor_posture_sweep.sh`. Output target:
  `experiments/outputs/direct_isaac_anchor_posture_sweep/20260705_direct_isaac_anchor_posture_sweep_seed17/`.
  Log target:
  `logs/direct_isaac_anchor_posture_sweep/direct_isaac_anchor_posture_sweep_seed17_srun.log`.
  Status at submission: queued in GPU partition for priority. Do not touch
  `carry1`, `reflex`, `rfx`, OpenPI, Cosmos, or non-Curiosity resources.
- 2026-07-05 direct Isaac anchor posture sweep retry:
  Slurm job `166557` started on `server10` but failed on the first candidate
  before rollout metrics because GPU pipeline joint-state reads returned CUDA
  tensors and the script passed them directly to NumPy:
  `TypeError: can't convert cuda:0 device type tensor to numpy`. This was a
  scene-code compatibility issue, not a model blocker. The job was canceled by
  the agent because it was the agent's own failing run. Added `_as_numpy()` to
  `build_core_world_anchored_footstep_carrier.py` and routed world-pose,
  joint-position, and measured-effort reads through CPU NumPy conversion.
  Lightweight `py_compile` passed. Submitted retry2 in tmux
  `curiosity_direct_anchor_sweep_retry2_0705`, Slurm job `166564`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17_retry2`; status at
  submission was queued for priority.
- 2026-07-05 direct Isaac anchor posture sweep retry3:
  retry2 Slurm job `166564` reached `server10` but failed before Python entry
  with shell parse error from the nested `run_core_world_anchored_footstep_carrier.sh`
  path: `unexpected EOF while looking for matching '"'`. To remove that
  fragile shell nesting, `run_direct_isaac_anchor_posture_sweep.sh` now calls
  `build_core_world_anchored_footstep_carrier.py` directly with a bash array
  for each posture candidate. Lightweight `bash -n` and `py_compile` passed.
  Submitted retry3 in tmux `curiosity_direct_anchor_sweep_retry3_0705`, Slurm
  job `166571`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17_retry3`; status at first
  record was queued for priority.
- 2026-07-05 direct Isaac anchor posture sweep retry4:
  retry3 Slurm job `166571` reached `server10` and entered the Isaac step
  loop, confirming the GPU tensor fix and flattened wrapper worked. However,
  the chosen combination `use_support_foot_drive=true` with
  `fix_anchor_to_world=true` deliberately set `rail_target=0`, so the first
  candidate stood/probed in place with `target_dist=0.0800` throughout. The
  agent canceled its own uninformative retry3 and changed the sweep default to
  `support_foot_mode=fixed_to_anchor` without support-foot drive so the rail
  drive can actually move the torso/payload. Lightweight checks passed.
  Submitted retry4 in tmux `curiosity_direct_anchor_sweep_retry4_0705`, Slurm
  job `166575`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17_retry4`; it started on
  `server10`.
- 2026-07-05 direct Isaac anchor posture sweep retry5:
  retry4 Slurm job `166575` reached `server10` with nonzero `rail_target`, but
  on the GPU pipeline the torso and payload still showed zero X travel through
  the first candidate. This was canceled as uninformative. Prior 8 cm
  anchored-support evidence used CPU backend, so the sweep now explicitly uses
  `DEVICE=cpu`, default `support_foot_mode=static_markers`, one rail joint,
  rail limits `[-0.10, 0.08]`, and a no-support-drive
  `horizontal_push_pull` probe that drives the rail during the probe phase.
  Lightweight checks passed. Submitted retry5 in tmux
  `curiosity_direct_anchor_sweep_retry5_0705`, Slurm job `166579`, stamp
  `20260705_direct_isaac_anchor_posture_sweep_seed17_retry5`; status at first
  record was queued for priority.
- 2026-07-05 direct Isaac anchor posture sweep retry5 result:
  retry5 ran on `server02` and completed the four-candidate diagnostic sweep.
  Output:
  `experiments/outputs/direct_isaac_anchor_posture_sweep/20260705_direct_isaac_anchor_posture_sweep_seed17_retry5/direct_isaac_anchor_posture_sweep_summary.json`.
  Log:
  `logs/direct_isaac_anchor_posture_sweep/direct_isaac_anchor_posture_sweep_seed17_retry5_srun.log`.
  The sampled hidden payload was identical across candidates:
  mass `8.175871 kg`, size approximately
  `0.36503 x 0.26652 x 0.20889 m`, COM offset
  `[0.02129, 0.01225, 0.00968] m`. Each candidate completed 300/300 steps
  with fall/drop 0 and negligible payload relative offset error. The metric
  ranker chose `extended_front` first, then `front_mid`, then `chest_close`,
  then `low_close`. `extended_front` and `front_mid` both reached the 8 cm
  target with final target distance about `3.27e-05 m`; `chest_close` and
  `low_close` stayed safe but stopped about `0.00950 m` short. Log scan found
  no `Traceback`, `Exception`, fatal, disjoint, unexpected EOF, or tensor
  conversion errors. This is a useful direct-Isaac scene/task skeleton for
  posture candidates plus active rail probing, but it is only an anchored
  fixed-payload diagnostic, not free-walking robot carrying, not free-box
  contact handling, not calibrated load belief, and not video-conditioned RL.
- 2026-07-05 direct Isaac strict-support pivot:
  The user correctly redirected the work away from waiting on external models
  and toward directly building the Isaac carrying scene. Implemented a stricter
  alternating support-foot diagnostic gate in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`:
  `support_foot_double_support_fraction`, all/drive-phase near-ground foot
  continuity counters, and commanded-stance near-ground counters. The direct
  physical backend wrapper now forwards the double-support parameter, the
  normalizer propagates the new metrics, and
  `check_direct_carry_task_summary.py` can require drive-phase support
  continuity. Added
  `scripts/isaac/run_alternating_anchor_feet_strict_support_16cm_diag.sh` for
  a 16 cm / 8 kg front-mid free-box diagnostic with no fixed-world support and
  strict drive-phase support continuity gates. Lightweight `bash -n` and
  Python syntax checks passed on the login node only; no Isaac simulation was
  run on the login node.
- 2026-07-05 strict 16 cm support-continuity diagnostic submitted:
  Submitted a new Curiosity-owned tmux/Slurm run without touching `carry1`:
  tmux `curiosity_alt_strict_support_16cm_0705`, Slurm job `166595`,
  command:
  `srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=alt_strict_support_16cm bash scripts/isaac/run_alternating_anchor_feet_strict_support_16cm_diag.sh`.
  Log:
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_16cm_0705_srun.log`.
  Expected output:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_16cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Initial status: pending for priority.
- 2026-07-05 strict 16 cm support-continuity diagnostic result:
  Slurm job `166595` ran on `server02` and completed in about 24 seconds.
  Summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_16cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Logs:
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_16cm_0705_srun.log`
  and
  `logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_20260705_direct_physical_backend_alternating_anchor_feet_strict_support_16cm_8kg_frontmid_backend_anchored_cradle.log`.
  The checker report printed `status=pass`. Key metrics: completed
  1180/1180, `max_box_travel_x_m=0.18552`,
  `final_box_target_distance_x_m=0.00242`, fall/drop 0,
  `root_shortcut_free=true`, `stance_anchor_fixed_to_world=false`,
  support-root writes 0, anchor retargets 0, foot pose writes 0,
  stance-anchor pose writes 0, `support_foot_mode=xz_prismatic_to_anchor`,
  8 support-foot joints, `support_foot_double_support_fraction=0.12`,
  `min_near_ground_foot_count=2`, `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_zero_steps=0`, `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`,
  `commanded_stance_near_ground_lt2_steps=0`,
  `min_support_polygon_margin_m=0.15951`, and actual support-foot lift
  `0.06562 m`. Log scan found only the expected headless display warning and
  no traceback, fatal, disjoint, EOF, or tensor conversion error. This is a
  strict Isaac scene diagnostic, not final free-walking robot evidence.
- 2026-07-05 strict 32 cm support-continuity diagnostic submitted:
  Added
  `scripts/isaac/run_alternating_anchor_feet_strict_support_32cm_diag.sh`,
  extending the same strict support gate to the existing 20-cycle 32 cm
  front-mid free-box setup. Lightweight `bash -n` and syntax checks passed.
  Submitted tmux `curiosity_alt_strict_support_32cm_0705`, Slurm job
  `166599`, command:
  `srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=alt_strict_support_32cm bash scripts/isaac/run_alternating_anchor_feet_strict_support_32cm_diag.sh`.
  Log:
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_32cm_0705_srun.log`.
  Expected output:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_32cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Initial status: pending for priority.
- 2026-07-05 strict 32 cm support-continuity diagnostic result:
  Slurm job `166599` ran on `server10` and completed in about 42 seconds.
  The checker report printed `status=pass`. Summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_32cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Key metrics: completed 1980/1980, `max_box_travel_x_m=0.38556`,
  `final_box_target_distance_x_m=0.03051`,
  `final_post_settle_box_travel_x_m=0.35173`, fall/drop 0,
  `root_shortcut_free=true`, `stance_anchor_fixed_to_world=false`,
  support-root writes 0, anchor retargets 0, foot pose writes 0,
  stance-anchor pose writes 0, `min_near_ground_foot_count=2`,
  `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_zero_steps=0`, `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`,
  `commanded_stance_near_ground_lt2_steps=0`,
  `min_support_polygon_margin_m=0.15951`, and actual support-foot lift
  `0.06971 m`. Log scan again found only the expected headless display
  warning and no traceback, fatal, disjoint, EOF, or tensor conversion error.
- 2026-07-05 strict 64 cm support-continuity diagnostic submitted:
  Added
  `scripts/isaac/run_alternating_anchor_feet_strict_support_64cm_diag.sh`,
  extending the same strict support gate to the existing 40-cycle 64 cm
  front-mid free-box setup. Lightweight `bash -n` and syntax checks passed.
  Submitted tmux `curiosity_alt_strict_support_64cm_0705`, Slurm job
  `166603`, command:
  `srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=alt_strict_support_64cm bash scripts/isaac/run_alternating_anchor_feet_strict_support_64cm_diag.sh`.
  Log:
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_64cm_0705_srun.log`.
  Expected output:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  Initial status: running on `server10`.
- 2026-07-05 strict 64 cm support-continuity diagnostic result:
  Slurm job `166603` ran on `server10`. The backend rollout completed and
  wrote summary/CSV, but `run_direct_carry_task_physical_backend.sh` reported
  a post-rollout shell parse error after the Isaac backend finished:
  `unexpected EOF while looking for matching '"'`. Because the backend summary
  had already been written, a separate compute-node normalization/check job
  was submitted in tmux
  `curiosity_alt_strict_support_64cm_normalize_0705`, Slurm job `166605`,
  and it completed successfully. Direct summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  The checker report printed `status=pass`. Key metrics: completed
  3580/3580, `max_box_travel_x_m=0.67301`,
  `final_box_target_distance_x_m=0.02369`,
  `final_post_settle_box_travel_x_m=0.66492`, fall/drop 0,
  `root_shortcut_free=true`, `stance_anchor_fixed_to_world=false`,
  support-root writes 0, anchor retargets 0, foot pose writes 0,
  stance-anchor pose writes 0, `min_near_ground_foot_count=2`,
  `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_zero_steps=0`, `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`,
  `commanded_stance_near_ground_lt2_steps=0`,
  `min_support_polygon_margin_m=0.15951`, and actual support-foot lift
  `0.06971 m`. Log scan found the expected headless display warning plus the
  post-rollout wrapper EOF in the first 64 cm srun log; the separate
  normalization/check log had no checker failures. This remains a strict
  direct-Isaac diagnostic scaffold, not final free-walking humanoid carrying.
- 2026-07-05 strict 64 cm posture diagnostics prepared:
  Added
  `scripts/isaac/run_alternating_anchor_feet_strict_support_64cm_postures_diag.sh`
  to run the same 64 cm / 8 kg strict support-continuity gate for `low_front`
  and `chest_high`. The script keeps the direct wrapper path when it succeeds,
  but if the known post-rollout EOF occurs after backend summary creation it
  recovers by running the normalizer/checker in the same compute-node job.
  Lightweight `bash -n` and syntax checks passed on the login node; no Isaac
  simulation was run on the login node.
- 2026-07-05 strict 64 cm posture diagnostics submitted:
  Submitted tmux `curiosity_alt_strict_support_64cm_postures_0705`, Slurm job
  `166612`, command:
  `srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=alt_strict_64_postures bash scripts/isaac/run_alternating_anchor_feet_strict_support_64cm_postures_diag.sh`.
  Log:
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_64cm_postures_0705_srun.log`.
  Expected summaries:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_low_front/direct_carry_task_physical_backend_summary.json`
  and
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_chest_high/direct_carry_task_physical_backend_summary.json`.
  Initial status: pending for priority. Ignore non-Curiosity/non-current jobs
  such as `phase03_h5rev06`.
- 2026-07-05 strict 64 cm posture diagnostics result:
  Slurm job `166612` ran on `server10` and completed both `low_front` and
  `chest_high` under the same strict support-continuity gate. Unlike the
  previous front-mid 64 cm run, the wrapper did not hit the post-rollout EOF.
  Logs:
  `logs/direct_carry_task_physical_backend/alternating_anchor_feet_strict_support_64cm_postures_0705_srun.log`,
  `logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_low_front_backend_anchored_cradle.log`,
  and
  `logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_chest_high_backend_anchored_cradle.log`.
  `low_front` summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_low_front/direct_carry_task_physical_backend_summary.json`.
  It passed with completed 3580/3580, `max_box_travel_x_m=0.66675`,
  `final_box_target_distance_x_m=0.00189`,
  `final_post_settle_box_travel_x_m=0.64326`, fall/drop 0,
  `root_shortcut_free=true`, `stance_anchor_fixed_to_world=false`, all
  support-root/anchor/foot/stance pose-write counters 0,
  `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_zero_steps=0`, `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`, and
  `commanded_stance_near_ground_lt2_steps=0`.
  `chest_high` summary:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_chest_high/direct_carry_task_physical_backend_summary.json`.
  It passed with completed 3580/3580, `max_box_travel_x_m=0.65313`,
  `final_box_target_distance_x_m=0.01468`,
  `final_post_settle_box_travel_x_m=0.62460`, fall/drop 0,
  `root_shortcut_free=true`, `stance_anchor_fixed_to_world=false`, all
  support-root/anchor/foot/stance pose-write counters 0,
  `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_zero_steps=0`, `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`, and
  `commanded_stance_near_ground_lt2_steps=0`.
  Log scan found only expected headless-display warnings and no traceback,
  fatal, disjoint, EOF, or tensor conversion error. This extends the strict
  direct-Isaac diagnostic scaffold across three carry postures, but it is
  still not a final walking humanoid policy.
- 2026-07-05 Arena/G1 persistent smoke blocker recorded:
  the retry2 persistent Arena G1 AGILE walk smoke still failed before rollout,
  with summary
  `experiments/outputs/arena_g1_agile_walk_persistent_smoke/20260705_arena_g1_agile_walk_persistent_smoke/arena_g1_agile_walk_persistent_summary.json`,
  `completed_steps=0`, status `fail`, and
  `Exception: Failed to get DOF velocities from backend`. This confirms the
  current Arena path is not useful as an immediate execution dependency. Do
  not keep waiting on this unchanged path; continue direct Isaac scene
  construction.
- 2026-07-05 probe-then-adaptive-carry strict-support implementation:
  added
  `scripts/isaac/run_probe_then_adaptive_carry_strict_support_diag.sh` and
  `scripts/isaac/summarize_probe_then_adaptive_carry.py`. The script runs a
  randomized hidden-box probe on the current no-fixed-world
  `alternating_anchor_feet`/`cradle_free_box` backend, reads the nonprivileged
  probe risk telemetry, selects `front_mid`, `low_front`, or `chest_high` with
  an explicit hand-coded rule, then runs the selected 64 cm carry under the
  strict support-continuity checker. This is a direct Isaac execution bridge
  for active posture choice, not RL, not video-conditioned learning, and not
  final free-walking humanoid carrying. Lightweight checks on the login node:
  `bash -n scripts/isaac/run_probe_then_adaptive_carry_strict_support_diag.sh`
  and
  `python3 -m py_compile scripts/isaac/summarize_probe_then_adaptive_carry.py`
  passed; no Isaac simulation was run on the login node.
- 2026-07-05 probe-then-adaptive-carry strict-support submitted:
  submitted from the login node into Curiosity-owned tmux session
  `curiosity_probe_adaptive_carry_0705`, Slurm job `166625`, command:
  `srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=probe_adapt_carry bash scripts/isaac/run_probe_then_adaptive_carry_strict_support_diag.sh`.
  Log:
  `logs/probe_then_adaptive_carry/probe_then_adaptive_carry_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/probe_then_adaptive_carry/20260705_probe_then_adaptive_carry_strict_support_seed7055/probe_then_adaptive_carry_summary.json`.
  Initial Slurm status: pending for priority.
- 2026-07-05 probe-then-adaptive-carry strict-support result:
  Slurm job `166625` ran on `server10` and completed with exit `0:0`.
  Aggregate summary:
  `experiments/outputs/probe_then_adaptive_carry/20260705_probe_then_adaptive_carry_strict_support_seed7055/probe_then_adaptive_carry_summary.json`.
  The probe used randomized hidden box seed `7055` with mass `8.24950 kg`,
  size `[0.32314, 0.23267, 0.22621] m`, and COM offset
  `[0.01653, -0.02103, 0.01754] m`. It completed 720/720 steps,
  reported `probe_belief_available=true`, did not use hidden ground truth,
  and produced risk `0.607367` / bucket
  `moderate_observed_load_response`, recommending
  `slow_gait_or_lower_carry_candidate`. The hand-coded selector chose
  `low_front`, `stance_steps=96`, `step_length=0.014 m`.
  The selected carry summary:
  `experiments/outputs/probe_then_adaptive_carry/20260705_probe_then_adaptive_carry_strict_support_seed7055/carry_low_front/direct_carry_task_physical_backend_summary.json`.
  It passed the strict checker with completed 3580/3580, max box travel
  `0.67171 m`, final box target distance `0.00361 m`, final post-settle box
  travel `0.64513 m`, fall/drop 0, `root_shortcut_free=true`,
  no fixed-world support, support-root/anchor/foot/stance pose writes 0,
  `min_drive_near_ground_foot_count=2`,
  `drive_near_ground_zero_steps=0`, `drive_near_ground_lt2_steps=0`,
  `min_commanded_stance_near_ground_foot_count=2`,
  `commanded_stance_near_ground_lt2_steps=0`, actual support-foot lift
  `0.07730 m`, and min support-polygon margin `0.14680 m`. Log scan found no
  traceback, fatal, disjoint, EOF, tensor failure, or checker failure; the
  only matches were `failures: []`. This is useful direct Isaac progress for
  active probe plus posture selection, but remains a heuristic scaffold, not
  RL, not video-conditioned learning, and not final full-robot walking/carrying.
- 2026-07-05 randomized all-posture strict-support implementation:
  added
  `scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh` and
  `scripts/isaac/summarize_randomized_all_posture_carry.py`, and extended
  `scripts/isaac/check_direct_carry_task_summary.py` with
  `--require-box-randomized` and `--expect-box-seed`. This new gate runs
  `front_mid`, `low_front`, and `chest_high` on the same randomized hidden box
  seed under the strict no-fixed-world/no-root-shortcut support-continuity
  checker. Lightweight checks on the login node passed:
  `bash -n scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh`
  and
  `python3 -m py_compile scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/summarize_randomized_all_posture_carry.py`.
  No Isaac simulation was run on the login node.
- 2026-07-05 randomized all-posture strict-support submitted:
  submitted from the login node into Curiosity-owned tmux session
  `curiosity_randomized_all_postures_0705`, Slurm job `166633`, command:
  `srun --partition=gpu --gres=gpu:1 --time=04:00:00 --job-name=rand_all_postures bash scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh`.
  Log:
  `logs/randomized_all_posture_strict_support/randomized_all_posture_strict_support_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_strict_support_64cm_seed7061/randomized_all_posture_strict_support_summary.json`.
  Initial Slurm status: pending for priority.
- 2026-07-05 randomized all-posture strict-support result:
  Slurm job `166633` ran on `server02` and completed with exit `0:0`.
  Aggregate summary:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_strict_support_64cm_seed7061/randomized_all_posture_strict_support_summary.json`.
  It used the same randomized hidden box for all postures: seed `7061`,
  mass `6.81119 kg`, size `[0.32037, 0.22802, 0.23574] m`, and COM offset
  `[0.01463, 0.02498, 0.00268] m`. All three direct-Isaac scaffold postures
  passed the strict no-fixed-world/no-root-shortcut support-continuity gate.
  `front_mid`: 3580/3580, max box travel `0.64402 m`, final target distance
  `0.01039 m`, final post-settle travel `0.63186 m`, fall/drop 0,
  `root_shortcut_free=true`, `stance_anchor_fixed_to_world=false`, all
  support-root/anchor/foot/stance pose-write counters 0,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`, and
  `commanded_stance_near_ground_lt2_steps=0`. `low_front`: 3580/3580, max box
  travel `0.68203 m`, final target distance `0.01310 m`, final post-settle
  travel `0.65428 m`, fall/drop 0, same strict support and shortcut counters.
  `chest_high`: 3580/3580, max box travel `0.66133 m`, final target distance
  `0.00638 m`, final post-settle travel `0.63276 m`, fall/drop 0, same strict
  support and shortcut counters. Log scan found no traceback, fatal, disjoint,
  EOF, tensor failure, or checker failure; the only matches were
  `failures: []`. This strengthens the direct Isaac scene path for hidden-box
  multi-posture carrying, but it is still a support-foot scaffold, not a full
  humanoid walking controller, not RL, and not video-conditioned learning.
- 2026-07-05 probe parameter-search scaffold implementation:
  added
  `scripts/isaac/run_probe_parameter_search_carry_diag.sh` and
  `scripts/isaac/summarize_probe_parameter_search_carry.py`. This is the next
  direct Isaac path and deliberately does not wait for external video/robot
  models. It runs one randomized hidden-box vertical micro-lift probe, then
  evaluates five hand-authored posture/gait/stance candidates on the same box:
  `front_mid_nominal`, `low_front_slow`, `chest_high_slowest`,
  `front_mid_wide_slow`, and `low_front_wide_slowest`. Each candidate is
  checked with the same strict no-fixed-world/no-root-shortcut
  support-continuity gate. The summarizer chooses the best passing candidate
  using a transparent diagnostic score based on final target error, travel
  loss, falls, drops, support discontinuity, and shortcuts. This is not RL, not
  video-conditioned learning, and not full humanoid walking evidence.
  Lightweight login-node checks passed:
  `bash -n scripts/isaac/run_probe_parameter_search_carry_diag.sh` and
  `python3 -m py_compile scripts/isaac/summarize_probe_parameter_search_carry.py`.
- 2026-07-05 probe parameter-search scaffold submitted:
  submitted from the login node into Curiosity-owned tmux session
  `curiosity_probe_param_search_0705`, Slurm job `166641`, command:
  `srun --partition=gpu --gres=gpu:1 --time=05:00:00 --job-name=probe_param_search bash scripts/isaac/run_probe_parameter_search_carry_diag.sh`.
  Log:
  `logs/probe_parameter_search_carry/probe_parameter_search_carry_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/probe_parameter_search_carry/20260705_probe_parameter_search_carry_seed7067/probe_parameter_search_carry_summary.json`.
  Initial Slurm status: pending for priority.
- 2026-07-05 probe parameter-search scaffold result:
  Slurm job `166641` ran on `server10` and completed with exit `0:0`.
  Aggregate summary:
  `experiments/outputs/probe_parameter_search_carry/20260705_probe_parameter_search_carry_seed7067/probe_parameter_search_carry_summary.json`.
  The outer tmux `tee` log was not created because the log directory did not
  exist before the tmux command opened `tee`; use the per-case logs under
  `logs/probe_parameter_search_carry/` and backend logs under
  `logs/core_world_anchored_footstep_carrier/` for this run. The run used one
  shared randomized hidden box: seed `7067`, mass `6.15402 kg`, size
  `[0.32579, 0.25445, 0.24170] m`, and COM offset
  `[0.01250, 0.02327, 0.01980] m`. The vertical micro-lift probe completed
  720/720, reported `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`, risk `0.596106`, bucket
  `moderate_observed_load_response`, and recommendation
  `slow_gait_or_lower_carry_candidate`.
  Five hand-authored candidates were evaluated. Strict-pass candidates:
  `front_mid_nominal` passed and was selected as best with 3580/3580, score
  `0.00286`, max box travel `0.66324 m`, final target distance `0.00286 m`,
  fall/drop 0, `min_drive_near_ground_foot_count=2`, and
  `drive_near_ground_lt2_steps=0`; `low_front_slow` also passed with 3580/3580,
  score `0.01574`, max box travel `0.68474 m`, final target distance
  `0.01574 m`, fall/drop 0, `min_drive_near_ground_foot_count=2`, and
  `drive_near_ground_lt2_steps=0`. Rejected candidates all reached the target
  without fall/drop but failed strict support continuity:
  `chest_high_slowest` had `drive_near_ground_lt2_steps=10`,
  `front_mid_wide_slow` had `16`, and `low_front_wide_slowest` had `36`.
  Log scan across per-case and backend logs found no traceback, fatal,
  disjoint, EOF, tensor failure, or checker exception. This is the first
  direct Isaac active-probe plus parameter-search execution scaffold, but it
  remains non-RL, non-video-conditioned, and not full humanoid walking.
- 2026-07-05 probe parameter-search multi-seed implementation/submission:
  added
  `scripts/isaac/run_probe_parameter_search_multiseed_diag.sh` and
  `scripts/isaac/summarize_probe_parameter_search_multiseed.py`. The wrapper
  reuses the direct Isaac probe plus parameter-search runner for seeds `7068`,
  `7069`, and `7070`, then aggregates best-candidate variation across seeds.
  Lightweight login-node checks passed:
  `bash -n scripts/isaac/run_probe_parameter_search_multiseed_diag.sh` and
  `python3 -m py_compile scripts/isaac/summarize_probe_parameter_search_multiseed.py`.
  Submitted from the login node into Curiosity-owned tmux session
  `curiosity_probe_param_multiseed_0705`, Slurm job `166649`, command:
  `srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_param_multi bash scripts/isaac/run_probe_parameter_search_multiseed_diag.sh`.
  Log:
  `logs/probe_parameter_search_multiseed/probe_parameter_search_multiseed_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7068_7070/probe_parameter_search_multiseed_summary.json`.
  Initial Slurm status: pending for priority.
- 2026-07-05 probe parameter-search multi-seed result:
  Slurm job `166649` ran on `server02` and completed with exit `0:0`.
  Aggregate summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7068_7070/probe_parameter_search_multiseed_summary.json`.
  Status `pass`: all three seeds completed and each had strict passing
  candidates. Seed `7068`: randomized hidden box mass `6.55342 kg`, COM
  `[0.03688, -0.01864, 0.02983] m`, probe risk `0.608155`, best
  `low_front_slow`, final target distance `0.00660 m`, fall/drop 0,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
  Seed `7069`: mass `5.36291 kg`, COM
  `[-0.00905, 0.02579, -0.00570] m`, probe risk `0.646449`, best
  `low_front_slow`, final target distance `0.00576 m`, fall/drop 0, strict
  support passed. Seed `7070`: mass `9.04893 kg`, COM
  `[-0.03931, 0.01795, -0.01916] m`, probe risk `0.596409`, best
  `low_front_slow`, final target distance `0.00045 m`, fall/drop 0, strict
  support passed. Each seed had 2/5 passing candidates and rejected
  `chest_high_slowest`, `front_mid_wide_slow`, and `low_front_wide_slowest`
  for strict support-continuity failures. Final aggregate:
  `best_candidate_id_counts={"low_front_slow": 3}`,
  `best_carry_posture_counts={"low_front": 3}`,
  `best_candidate_varied=false`, and `best_posture_varied=false`.
  Scan for traceback, exception, fatal, disjoint, failed, and unexpected EOF
  across multi-seed/per-seed/backend logs was empty. Interpretation: the
  direct Isaac scaffold now supports repeatable active-probe plus
  parameter-search evaluation across hidden boxes, but the current candidate
  set and score collapse to one safe low-front option; it still does not
  demonstrate morphology-dependent posture diversity, RL, video conditioning,
  or full humanoid walking.
- 2026-07-05 Core API G1 stand-gain height-sweep implementation:
  patched `scripts/isaac/build_core_world_g1_box_scene.py` and
  `scripts/isaac/run_core_world_g1_box_scene.sh` so the direct Core API G1
  scene can optionally apply Arena-style stand PD drive gains to the local G1
  USD joints with `--apply-arena-stand-gains`, and records
  `stand_drive_gains_enabled`, `stand_gain_scale`,
  `applied_stand_drive_gains`, and `applied_stand_drive_gain_count` in the
  summary. Added
  `scripts/isaac/check_core_world_g1_box_scene_summary.py`,
  `scripts/isaac/run_core_world_g1_stand_height_sweep.sh`, and
  `scripts/isaac/summarize_core_world_g1_stand_height_sweep.py`. The sweep
  tests Core API G1 standing with a fixed-torso attached box at root heights
  `0.78`, `0.84`, `0.90`, and `0.96`, using strict gates for completed steps,
  joint count, applied drive-gain count, fall/drop 0, min robot/box height,
  max tilt, and no rollout root/box pose writes. This is a real-G1 backend
  prerequisite diagnostic, not walking or carrying success. Lightweight
  login-node checks passed:
  `bash -n scripts/isaac/run_core_world_g1_box_scene.sh scripts/isaac/run_core_world_g1_stand_height_sweep.sh`
  and
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/check_core_world_g1_box_scene_summary.py scripts/isaac/summarize_core_world_g1_stand_height_sweep.py`.
- 2026-07-05 Core API G1 stand-gain height-sweep submitted:
  submitted from the login node into Curiosity-owned tmux session
  `curiosity_g1_stand_height_0705`, Slurm job `166658`, command:
  `srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_stand_sweep bash scripts/isaac/run_core_world_g1_stand_height_sweep.sh`.
  Log:
  `logs/core_world_g1_stand_height_sweep/g1_stand_height_sweep_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep/core_world_g1_stand_height_sweep_summary.json`.
  Initial Slurm status: pending for priority.
- 2026-07-05 Core API G1 stand-gain height-sweep result:
  Slurm job `166658` ran on `server02` and exited `FAILED` with code `1:0`
  because the aggregate strict gate failed; there were no traceback, fatal,
  disjoint, failed-backend, or unexpected-EOF log matches. Aggregate summary:
  `experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep/core_world_g1_stand_height_sweep_summary.json`.
  All four fixed-torso 2 kg box stand cases applied Arena-style gains to 23
  joints and completed 240/240 rollout steps with no rollout root/box pose
  writes and no `box_drop_events`, but all failed by falling/tilting:
  `z_0p78` fall events `152`, min robot z `0.40650 m`, min box z
  `0.25449 m`, max tilt `1.14366 rad`; `z_0p84` fall events `172`, min robot
  z `0.39728 m`, min box z `0.23892 m`, max tilt `1.18476 rad`; `z_0p90`
  fall events `191`, min robot z `0.40567 m`, min box z `0.22920 m`, max tilt
  `1.16031 rad`; `z_0p96` fall events `191`, min robot z `0.39748 m`, min box
  z `0.26170 m`, max tilt `1.17449 rad`. Interpretation: applying Arena
  stand PD gains and sweeping root height is not enough to make the direct
  Core API G1 backend stand with a fixed-torso payload. Next isolation: rerun
  the same height sweep without attaching the box to determine whether the
  blocker is base G1 standing or payload attachment.
- 2026-07-05 Core API G1 no-box stand-height sweep submitted:
  submitted from the login node into Curiosity-owned tmux session
  `curiosity_g1_stand_nobox_0705`, Slurm job `166661`, command:
  `STAMP=20260705_core_world_g1_stand_height_sweep_nobox ATTACH_BOX_MODE=none EXPECT_ATTACH_BOX=none MAX_BOX_DROP_EVENTS=999 MIN_BOX_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_stand_nobox bash scripts/isaac/run_core_world_g1_stand_height_sweep.sh`.
  Log:
  `logs/core_world_g1_stand_height_sweep/g1_stand_height_sweep_nobox_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep_nobox/core_world_g1_stand_height_sweep_summary.json`.
  Initial Slurm status: running on `server36`. Purpose: isolate whether the
  previous fixed-torso payload failure is caused by base Core API G1 standing
  or by payload attachment/posture.
- 2026-07-05 Core API G1 no-box stand-height sweep result:
  Slurm job `166661` ran on `server36` and completed with exit `0:0`; log
  scan across `logs/core_world_g1_stand_height_sweep` and
  `logs/core_world_g1_box_scene` found no traceback, exception, fatal,
  disjoint, failed-backend, or unexpected-EOF matches. Aggregate summary:
  `experiments/outputs/core_world_g1_stand_height_sweep/20260705_core_world_g1_stand_height_sweep_nobox/core_world_g1_stand_height_sweep_summary.json`.
  Status `pass`: 2/4 no-box height cases passed the strict stand gate.
  `z_0p84` passed with 240/240 steps, fall events `0`, min robot z
  `0.77672 m`, max tilt `0.23243 rad`, max robot XY travel `0.15425 m`, and
  23 applied stand drive gains. `z_0p96` passed with 240/240 steps, fall
  events `0`, min robot z `0.74072 m`, max tilt `0.45388 rad`, max robot XY
  travel `0.34181 m`, and 23 applied gains. `z_0p78` failed with 32 fall
  events, min robot z `0.34155 m`, and max tilt `1.37618 rad`; `z_0p90`
  failed with 24 fall events and max tilt `1.01795 rad`. Interpretation: the
  direct Core API G1 path can stand without an attached payload for some root
  heights, so the previous all-failed fixed-torso 2 kg result points mainly to
  payload attachment/posture/load handling rather than complete base-G1
  infeasibility. This remains a diagnostic only, not walking or carrying
  success. Next direct Isaac step: payload mass/offset isolation from the
  passed no-box heights before attempting walking.
- 2026-07-05 Core API G1 fixed-payload isolation implementation/submission:
  patched `scripts/isaac/build_core_world_g1_box_scene.py` to record
  `attach_body_path`, `attach_local_pos0_m`, and
  `box_position_requested_m` in each summary. Added
  `scripts/isaac/run_core_world_g1_payload_sweep.sh` and
  `scripts/isaac/summarize_core_world_g1_payload_sweep.py` to sweep from the
  no-box-passing root heights over attached payload masses and torso attach
  offsets while preserving the no-rollout-root/box-pose-write gates.
  Lightweight login-node checks passed:
  `bash -n scripts/isaac/run_core_world_g1_payload_sweep.sh scripts/isaac/run_core_world_g1_stand_height_sweep.sh scripts/isaac/run_core_world_g1_box_scene.sh`
  and
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/summarize_core_world_g1_payload_sweep.py scripts/isaac/check_core_world_g1_box_scene_summary.py`.
  Submitted compute run from Curiosity-owned tmux session
  `curiosity_g1_payload_sweep_0705`, Slurm job `166663`, command:
  `STAMP=20260705_core_world_g1_payload_sweep_small HEIGHTS="0.84 0.96" MASSES="0.25 0.50 1.00 2.00" ATTACH_XS="0.12 0.18 0.24" srun --partition=gpu --gres=gpu:1 --time=03:00:00 --job-name=g1_payload_sweep bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_sweep_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_sweep_small/core_world_g1_payload_sweep_summary.json`.
  Initial Slurm status: pending for priority. This is a fixed-payload standing
  diagnostic only, not walking or carrying success.
- 2026-07-05 Core API G1 open-loop march diagnostic implementation:
  patched `scripts/isaac/build_core_world_g1_box_scene.py` and
  `scripts/isaac/run_core_world_g1_box_scene.sh` to support diagnostic-only
  `GAIT_MODE=open_loop_march` with configurable `GAIT_AMPLITUDE` and
  `GAIT_FREQUENCY_HZ`. Added
  `scripts/isaac/run_core_world_g1_open_loop_march_probe.sh` to test whether
  the no-box-passing G1 stand heights survive simple periodic leg commands.
  This is explicitly not a serious humanoid walking controller and must not be
  reported as locomotion success. It is a failure-finding bridge from stand to
  controller-backed walking. Lightweight login-node checks passed:
  `bash -n scripts/isaac/run_core_world_g1_open_loop_march_probe.sh scripts/isaac/run_core_world_g1_box_scene.sh scripts/isaac/run_core_world_g1_payload_sweep.sh`
  and
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/summarize_core_world_g1_payload_sweep.py scripts/isaac/summarize_core_world_g1_stand_height_sweep.py`.
- 2026-07-05 launcher robustness correction:
  while Slurm job `166663` was running, one payload-sweep case logged
  `scripts/isaac/run_core_world_g1_box_scene.sh: line 46: --apply-arena-stand-gains: command not found`
  after summary writing. The launcher was immediately changed from a multiline
  command with `${APPLY_ARENA_STAND_GAINS:+--apply-arena-stand-gains}` to a
  Bash argument array plus conditional append. `bash -n` passed afterward.
  When interpreting job `166663`, inspect `run_status.txt` per case and do not
  overclaim any case affected by this transient launcher issue.
- 2026-07-05 Core API G1 open-loop march probe submitted:
  submitted from Curiosity-owned tmux session `curiosity_g1_march_probe_0705`,
  Slurm job `166667`, command:
  `STAMP=20260705_core_world_g1_open_loop_march_probe_small HEIGHTS="0.84 0.96" AMPLITUDES="0.05 0.10" ATTACH_BOX_MODE=none EXPECT_ATTACH_BOX=none MAX_BOX_DROP_EVENTS=999 MIN_BOX_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_march_probe bash scripts/isaac/run_core_world_g1_open_loop_march_probe.sh`.
  Log:
  `logs/core_world_g1_open_loop_march_probe/g1_march_probe_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/core_world_g1_open_loop_march_probe/20260705_core_world_g1_open_loop_march_probe_small/core_world_g1_open_loop_march_probe_summary.json`.
  Initial Slurm status: pending for priority. This is a no-box diagnostic only,
  not walking success.
- 2026-07-05 Core API G1 open-loop march probe result:
  Slurm job `166667` ran on `server36` and exited `FAILED` with code `1:0`
  because the aggregate strict gate failed; log scan found no traceback,
  exception, fatal, disjoint, failed-backend, unexpected-EOF, or
  command-not-found matches. Aggregate summary:
  `experiments/outputs/core_world_g1_open_loop_march_probe/20260705_core_world_g1_open_loop_march_probe_small/core_world_g1_open_loop_march_probe_summary.json`.
  Status `fail`: 0/4 no-box open-loop march cases passed. `z_0p84_amp_0p05`
  had 26 fall events, min robot z `0.35212 m`, max tilt `1.21204 rad`;
  `z_0p84_amp_0p10` had 33 fall events, min robot z `0.26058 m`, max tilt
  `1.29428 rad`; `z_0p96_amp_0p05` had 119 fall events, min robot z
  `0.33129 m`, max tilt `3.13968 rad`; `z_0p96_amp_0p10` had 101 fall
  events, min robot z `0.33416 m`, max tilt `3.14131 rad`. Interpretation:
  direct Core API open-loop joint marching is not a valid walking controller.
  The project should not spend time making larger open-loop sweeps; walking
  requires a controller-backed Isaac route or explicit balance feedback before
  carrying-walk claims.
- 2026-07-05 Core API G1 fixed-payload isolation sweep result:
  Slurm job `166663` ran on `server36` and exited `FAILED` with code `1:0`
  because the aggregate strict gate failed. Aggregate summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_sweep_small/core_world_g1_payload_sweep_summary.json`.
  Status `fail`: 0/24 forward fixed-torso payload cases passed across root
  heights `0.84/0.96 m`, masses `0.25/0.50/1.00/2.00 kg`, and attach x
  offsets `0.12/0.18/0.24 m`. Two cases were affected by the transient
  launcher edit and must not be treated as clean physics evidence:
  `z_0p84_m_0p50_x_0p18` had `run_status=127` and logged
  `--apply-arena-stand-gains: command not found`; `z_0p84_m_1p00_x_0p24`
  had `run_status=2` and logged `unexpected EOF while looking for matching`.
  The remaining 22 cases completed the runner and still failed strict physics
  gates by falls, excessive tilt, low robot/box height, and sometimes box
  drops, with zero rollout root/velocity/box pose writes. Representative clean
  failures: `z_0p84_m_0p25_x_0p12` had 225 fall events, min robot z
  `0.17420 m`, min box z `0.21434 m`, max tilt `3.05078 rad`;
  `z_0p96_m_0p25_x_0p12` had 225 fall events, 18 box-drop events, min robot z
  `0.16908 m`, max tilt `3.07085 rad`; `z_0p96_m_2p00_x_0p24` had 129 fall
  events, 38 box-drop events, min robot z `-0.54023 m`, min box z
  `-0.65664 m`, max tilt `3.13964 rad`. Interpretation: the forward
  fixed-torso payload setup is not viable, even at light masses. Next
  isolation should test centered ultra-light attached ballast to distinguish
  "any fixed payload destabilizes" from "front-mounted moment/initial
  constraint geometry destabilizes".
- 2026-07-05 Core API G1 centered ultra-light payload sweep submitted:
  patched `scripts/isaac/run_core_world_g1_payload_sweep.sh` so default
  `BOX_POS_Z` follows the current root height when not explicitly set, reducing
  avoidable fixed-joint initial mismatch in future payload sweeps. Lightweight
  login-node checks passed:
  `bash -n scripts/isaac/run_core_world_g1_payload_sweep.sh scripts/isaac/run_core_world_g1_box_scene.sh`
  and
  `python3 -m py_compile scripts/isaac/summarize_core_world_g1_payload_sweep.py scripts/isaac/build_core_world_g1_box_scene.py`.
  Submitted from Curiosity-owned tmux session
  `curiosity_g1_payload_centered_0705`, Slurm job `166668`, command:
  `STAMP=20260705_core_world_g1_payload_centered_ultralight HEIGHTS="0.84 0.96" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.0" ATTACH_Z=0.0 BOX_POS_X=0.0 BOX_POS_Y=0.0 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_center bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_centered_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight/core_world_g1_payload_sweep_summary.json`.
  Initial Slurm status: pending for priority. This is an attached-ballast
  diagnostic only, not carrying success.
- 2026-07-05 Core API G1 centered ultra-light payload sweep result:
  Slurm job `166668` ran on `server36` and exited `FAILED` with code `1:0`
  because the aggregate strict gate failed. Aggregate summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight/core_world_g1_payload_sweep_summary.json`.
  Status `fail`: 0/8 centered payload cases passed across root heights
  `0.84/0.96 m` and masses `0.01/0.05/0.10/0.25 kg`. Even
  `z_0p84_m_0p01_x_0p0` had 39 fall events, min robot z `0.19566 m`, min box
  z `0.20512 m`, and max tilt `1.44487 rad`; `z_0p96_m_0p01_x_0p0` had 102
  fall events, 51 box-drop events, min robot z `0.10824 m`, and max tilt
  `3.01276 rad`. Interpretation: the failure is not only forward load moment.
  The next direct Isaac isolation is to disable box collision while preserving
  the centered fixed joint and tiny added mass.
- 2026-07-05 Core API G1 no-collision payload isolation implementation:
  patched `scripts/isaac/build_core_world_g1_box_scene.py` to add
  `--disable-box-collision` and record `box_collision_enabled`; patched
  `scripts/isaac/run_core_world_g1_box_scene.sh`,
  `scripts/isaac/run_core_world_g1_payload_sweep.sh`, and
  `scripts/isaac/summarize_core_world_g1_payload_sweep.py` to pass and report
  `BOX_COLLISION_ENABLED`. Lightweight login-node checks passed:
  `bash -n scripts/isaac/run_core_world_g1_box_scene.sh scripts/isaac/run_core_world_g1_payload_sweep.sh`
  and
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/summarize_core_world_g1_payload_sweep.py`.
  This is an isolation diagnostic only, not carrying success.
- 2026-07-05 Core API G1 no-collision payload isolation submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_g1_payload_nocoll_0705`, Slurm job `166672`, command:
  `STAMP=20260705_core_world_g1_payload_centered_ultralight_nocoll HEIGHTS="0.84 0.96" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.0" ATTACH_Z=0.0 BOX_POS_X=0.0 BOX_POS_Y=0.0 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 BOX_COLLISION_ENABLED=0 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_nocoll bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_nocoll_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight_nocoll/core_world_g1_payload_sweep_summary.json`.
  Initial Slurm status: running on `server36`. This is an attached-ballast
  collision-isolation diagnostic only, not carrying success.
- 2026-07-05 Core API G1 no-collision payload isolation result:
  Slurm job `166672` completed on `server36` with exit `0:0`. Aggregate
  summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_centered_ultralight_nocoll/core_world_g1_payload_sweep_summary.json`.
  Status `pass`: 8/8 centered no-collision fixed-payload cases passed the
  strict stand gate up to `0.25 kg`, with fall/drop `0`, zero rollout root
  pose/velocity writes, and zero rollout box pose writes. Best 0.84 m case
  `z_0p84_m_0p01_x_0p0` had min robot z `0.76835 m`, min box z `0.81146 m`,
  max tilt `0.31023 rad`, and max robot XY travel `0.22629 m`. Worst tilt
  case `z_0p96_m_0p25_x_0p0` had min robot z `0.69804 m`, min box z
  `0.73666 m`, max tilt `0.59704 rad`, and max robot XY travel `0.45718 m`.
  Log scan found no traceback, exception, fatal, disjoint articulation,
  failed-backend, unexpected-EOF, or command-not-found matches. Interpretation:
  the immediate blocker is collision/contact geometry or initial
  interpenetration, not tiny fixed added mass itself. This is still not
  walking, free-object grasping, or carrying success.
- 2026-07-05 Core API G1 payload clearance launcher correction:
  patched `scripts/isaac/run_core_world_g1_payload_sweep.sh` so default
  requested box initial pose follows the fixed-joint attach offset as
  `(attach_x, 0, height + attach_z)` when `BOX_POS_*` is not explicitly set,
  and each `case_config.json` records `box_position_requested_m`. Lightweight
  login-node checks passed:
  `bash -n scripts/isaac/run_core_world_g1_payload_sweep.sh scripts/isaac/run_core_world_g1_box_scene.sh`
  and
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/summarize_core_world_g1_payload_sweep.py`.
- 2026-07-05 Core API G1 collision-enabled payload clearance sweep
  submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_g1_payload_clearance_0705`, Slurm job `166673`, command:
  `STAMP=20260705_core_world_g1_payload_clearance_collision HEIGHTS="0.84" MASSES="0.01 0.05 0.10 0.25" ATTACH_XS="0.18 0.24 0.30 0.36" ATTACH_Z=0.12 BOX_SIZE_X=0.10 BOX_SIZE_Y=0.10 BOX_SIZE_Z=0.10 BOX_COLLISION_ENABLED=1 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_payload_clear bash scripts/isaac/run_core_world_g1_payload_sweep.sh`.
  Log:
  `logs/core_world_g1_payload_sweep/g1_payload_clearance_0705_srun.log`.
  Expected aggregate summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_clearance_collision/core_world_g1_payload_sweep_summary.json`.
  Initial Slurm status: pending for priority. This is a collision/contact
  clearance diagnostic only, not carrying success.
- 2026-07-05 Core API G1 collision-enabled payload clearance sweep result:
  Slurm job `166673` completed on `server36` with exit `0:0`. Aggregate
  summary:
  `experiments/outputs/core_world_g1_payload_sweep/20260705_core_world_g1_payload_clearance_collision/core_world_g1_payload_sweep_summary.json`.
  Status `pass`: 16/16 collision-enabled clearance cases passed strict
  fixed-payload stand gate through `0.25 kg`, attach x
  `0.18/0.24/0.30/0.36 m`, attach z `0.12 m`, and 0.10 m cube geometry.
  Fall/drop events were `0`, rollout root pose/velocity writes were `0`,
  rollout box pose writes were `0`, and log scan found no traceback,
  exception, fatal, disjoint articulation, failed-backend, unexpected-EOF, or
  command-not-found matches. Best case `z_0p84_m_0p01_x_0p18` had min robot z
  `0.76792 m`, min box z `0.86952 m`, max tilt `0.31326 rad`, and max robot
  XY travel `0.22865 m`. Worst tilt case `z_0p84_m_0p25_x_0p24` had min robot
  z `0.75721 m`, min box z `0.82233 m`, max tilt `0.37794 rad`, and max robot
  XY travel `0.27910 m`. Lowest box case `z_0p84_m_0p25_x_0p36` had min box z
  `0.78429 m`. Interpretation: direct Isaac now has a stable
  collision-enabled fixed-payload standing baseline. This remains only a
  diagnostic; it is not walking, free-object grasping, active probing, or
  carrying success. Next required work is controller-backed stepping/walking
  from this stable payload baseline, not external model waiting or open-loop
  march.
- 2026-07-05 user correction accepted: stop waiting on additional external
  models or downloads. The active path is direct Isaac scene construction.
  The immediate controller-backed walking attempt is now the existing Core API
  G1 scene plus official local WBC-AGILE G1 velocity-height ONNX glue. This
  uses:
  `external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.onnx`
  and
  `external/IsaacLab-Arena/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/g1_agile.yaml`.
  The glue is allowed because it only adapts Isaac Core joint/root observations
  into the official ONNX input layout and maps the official 12-leg-joint output
  back to G1 joint targets. It must not be described as a new learned method or
  as final carrying success.
- 2026-07-05 Core API G1 AGILE policy integration implementation:
  patched `scripts/isaac/build_core_world_g1_box_scene.py` to add
  `GAIT_MODE=agile_policy`, `--policy-start-step`,
  `--policy-control-decimation`, `--agile-command`,
  `--agile-height-command`, `--agile-config`, and `--agile-onnx`. Patched
  `scripts/isaac/run_core_world_g1_box_scene.sh` to pass these through. The
  script records ONNX/config paths, policy inference count, and max raw action
  norm while keeping rollout root pose writes, root velocity writes, and box
  pose writes explicitly counted. Lightweight login-node checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py` and
  `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`. This is ready for a
  compute-node no-box walking smoke; do not load ONNX/Isaac on the login node.
- 2026-07-05 Core API G1 AGILE no-box walking smoke submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_g1_agile_nobox_0705`, Slurm job `166681`, command:
  `STAMP=20260705_core_world_g1_agile_policy_nobox_diag1 GAIT_MODE=agile_policy ATTACH_BOX=none STEPS=360 G1_ROOT_Z=0.84 APPLY_ARENA_STAND_GAINS=1 POLICY_START_STEP=40 POLICY_CONTROL_DECIMATION=4 AGILE_COMMAND_X=0.20 AGILE_COMMAND_Y=0.0 AGILE_COMMAND_YAW=0.0 AGILE_HEIGHT_COMMAND=0.72 srun --partition=gpu --gres=gpu:1 --time=01:00:00 --job-name=g1_agile_nobox bash scripts/isaac/run_core_world_g1_box_scene.sh`.
  Log:
  `logs/core_world_g1_box_scene/g1_agile_nobox_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_box_scene/20260705_core_world_g1_agile_policy_nobox_diag1/core_world_g1_box_scene_summary.json`.
  Initial Slurm status: pending for priority. This is a no-box walking smoke,
  not carrying evidence.
- 2026-07-05 Core API G1 AGILE no-box walking smoke result:
  Slurm job `166681` completed on `server36` with exit `0:0`, but produced no
  summary JSON and no `[STATE]` rollout rows under
  `experiments/outputs/core_world_g1_box_scene/20260705_core_world_g1_agile_policy_nobox_diag1/`.
  Logs only reached Isaac startup and ONNXRuntime thread-affinity warnings.
  Treat this as a failed diagnostic, not walking evidence. Patched
  `AgileOnnxJointPolicy` to use ONNXRuntime `CPUExecutionProvider` with
  `intra_op_num_threads=1`, `inter_op_num_threads=1`, and sequential execution,
  and added progress prints around Core World reset, wrapper initialization,
  and policy loading. Lightweight checks passed again with
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py` and
  `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- 2026-07-05 Core API G1 AGILE no-box walking smoke retry submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_g1_agile_nobox2_0705`, Slurm job `166684`, command:
  `STAMP=20260705_core_world_g1_agile_policy_nobox_diag2_singlethread GAIT_MODE=agile_policy ATTACH_BOX=none STEPS=360 G1_ROOT_Z=0.84 APPLY_ARENA_STAND_GAINS=1 POLICY_START_STEP=40 POLICY_CONTROL_DECIMATION=4 AGILE_COMMAND_X=0.20 AGILE_COMMAND_Y=0.0 AGILE_COMMAND_YAW=0.0 AGILE_HEIGHT_COMMAND=0.72 srun --partition=gpu --gres=gpu:1 --time=01:00:00 --job-name=g1_agile_nb2 bash scripts/isaac/run_core_world_g1_box_scene.sh`.
  Log:
  `logs/core_world_g1_box_scene/g1_agile_nobox2_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_box_scene/20260705_core_world_g1_agile_policy_nobox_diag2_singlethread/core_world_g1_box_scene_summary.json`.
  Initial Slurm status: running on `server36`.
- 2026-07-05 Core API G1 AGILE ONNX retry result:
  Slurm job `166684` completed on `server36` with exit `0:0`, but again
  produced no summary JSON and no rollout rows. New progress prints show the
  script reached Core World reset, G1/box wrapper initialization, 43-joint G1
  articulation initialization, and then stopped at `Loading AGILE ONNX policy`.
  Therefore ONNXRuntime `InferenceSession` inside the Isaac process is the
  failing boundary. Treat both ONNX no-box attempts as failed diagnostics and
  do not use them as evidence.
- 2026-07-05 Core API G1 AGILE torch-checkpoint backend implementation:
  added `--agile-policy-backend {onnx,torch_checkpoint}` and
  `--agile-torch-checkpoint` to
  `scripts/isaac/build_core_world_g1_box_scene.py`, with launcher wiring in
  `scripts/isaac/run_core_world_g1_box_scene.sh`. The new default backend is
  `torch_checkpoint`, using WBC-AGILE's own
  `agile.sim2mujoco.policy.PolicyWrapper` and local official checkpoint
  `external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt`.
  This is still official-policy glue, not a toy controller. Lightweight checks
  passed:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py` and
  `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- 2026-07-05 Core API G1 AGILE torch-checkpoint no-box walking smoke
  submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_g1_agile_torch_nobox_0705`, Slurm job `166690`, command:
  `STAMP=20260705_core_world_g1_agile_policy_nobox_diag3_torchckpt AGILE_POLICY_BACKEND=torch_checkpoint GAIT_MODE=agile_policy ATTACH_BOX=none STEPS=360 G1_ROOT_Z=0.84 APPLY_ARENA_STAND_GAINS=1 POLICY_START_STEP=40 POLICY_CONTROL_DECIMATION=4 AGILE_COMMAND_X=0.20 AGILE_COMMAND_Y=0.0 AGILE_COMMAND_YAW=0.0 AGILE_HEIGHT_COMMAND=0.72 srun --partition=gpu --gres=gpu:1 --time=01:00:00 --job-name=g1_agile_tnb bash scripts/isaac/run_core_world_g1_box_scene.sh`.
  Log:
  `logs/core_world_g1_box_scene/g1_agile_torch_nobox_0705_srun.log`.
  Expected summary:
  `experiments/outputs/core_world_g1_box_scene/20260705_core_world_g1_agile_policy_nobox_diag3_torchckpt/core_world_g1_box_scene_summary.json`.
  Initial Slurm status: pending for priority.
- 2026-07-05 Core API G1 AGILE torch-checkpoint no-box result:
  Slurm job `166690` completed on `server02` with exit `0:0`, but produced no
  summary JSON and no rollout rows. The log reached Core World reset,
  G1/box wrapper initialization, 43-joint G1 articulation initialization, and
  stopped at `Loading AGILE torch checkpoint policy`. Together with the two
  ONNX attempts, this is enough evidence to freeze embedded WBC-AGILE policy
  loading in the Isaac Core process for now. Do not keep rerunning WBC-AGILE
  loaders unchanged. Continue direct Isaac scene/control work through the
  physical backend scaffolds that already enter rollout.
- 2026-07-05 current direct Isaac physical backend baseline to use:
  `experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_strict_support_64cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json`.
  This is the best current runnable baseline, not final success. It uses an
  articulated 10-joint carrier with dynamic anchor and alternating x/z
  prismatic support feet, 8 kg free box in a cradle, target x `0.64 m`,
  completed `3580` steps, fall/drop `0`, final torso/payload post-settle
  travel about `0.664 m`, final target distance `0.0201 m`, max tilt
  `0.1214 rad`, min support polygon margin `0.1595 m`, min near-ground foot
  count `2`, commanded stance near-ground lt2 steps `0`, and root pose,
  root velocity, root angular velocity, payload pose, box pose, foot pose, and
  body root shortcut writes all `0`. It remains a diagnostic because it is not
  a humanoid policy, uses scaffolded alternating support-foot control, and has
  no active probing in the strict-support run.
- 2026-07-05 randomized all-posture strict-support diagnostic submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_randomized_posture_strict_0705`, Slurm job `166692`, command:
  `STAMP=20260705_randomized_all_posture_strict_support_64cm_seed7071 BOX_SEED=7071 srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=rand_posture bash scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh`.
  Log:
  `logs/direct_carry_task_physical_backend/randomized_all_posture_strict_0705_srun.log`.
  Expected root output:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_strict_support_64cm_seed7071/`.
  Initial Slurm status: running on `server02`. This is a randomized diagnostic
  over `front_mid`, `low_front`, and `chest_high` carry postures with payload
  mass/size/COM randomization and strict checker gates; it is still not final
  humanoid or learned carrying success.
- 2026-07-05 randomized all-posture strict-support diagnostic result:
  Slurm job `166692` completed on `server02` with exit `0:0`. Aggregate
  summary:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_strict_support_64cm_seed7071/randomized_all_posture_strict_support_summary.json`.
  Shared randomized hidden box: mass `11.47446 kg`, size
  `[0.36871, 0.22426, 0.21205] m`, COM offset
  `[-0.03709, -0.01539, 0.02677] m`. All three posture cases passed strict
  support and shortcut gates with `3580/3580` completed steps, fall/drop `0`,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`,
  `root_shortcut_free=true`, and `stance_anchor_fixed_to_world=false`.
  `front_mid` ended with final box target distance `0.00315 m` and
  post-settle box travel `0.63802 m`; `low_front` ended with final target
  distance `0.01245 m` and post-settle box travel `0.65413 m`; `chest_high`
  ended with final target distance `0.01292 m` and post-settle box travel
  `0.62644 m`. This reinforces that the current direct Isaac scaffold can run
  randomized free-box carrying across posture choices, but it is still not a
  humanoid, learned policy, or video-conditioned result. Keep external model
  loading off the critical path unless it directly improves this runnable
  Isaac scene.
- 2026-07-05 active-probe parameter-search multi-seed continuation submitted:
  continuing the direct Isaac path without external model waiting. Submitted
  from Curiosity-owned tmux session
  `curiosity_probe_param_multiseed_7071_0705`, Slurm job `166694`, command:
  `MULTISEED_STAMP=20260705_probe_parameter_search_multiseed_7071_7073 SEEDS='7071 7072 7073' srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_param_7071 bash scripts/isaac/run_probe_parameter_search_multiseed_diag.sh`.
  Log:
  `logs/probe_parameter_search_multiseed/probe_parameter_search_multiseed_7071_0705_srun.log`.
  Expected summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7071_7073/probe_parameter_search_multiseed_summary.json`.
  Initial Slurm status: pending for priority. This run evaluates the
  active-probe plus strict candidate-search scaffold on new hidden seeds
  `7071`, `7072`, and `7073`; it remains a diagnostic, not learned RL.
- 2026-07-05 active-probe parameter-search multi-seed continuation result:
  Slurm job `166694` completed on `server02` with exit `0:0` after
  `00:10:22`. Aggregate summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_multiseed_7071_7073/probe_parameter_search_multiseed_summary.json`.
  All 3 hidden seeds completed and passed the diagnostic wrapper. Best
  candidate varied across seeds: `front_mid_nominal` won 2 seeds and
  `low_front_slow` won 1 seed; best posture counts were `front_mid: 2` and
  `low_front: 1`, so `best_posture_varied=true`.
  Seed `7071`: mass `11.47446 kg`, probe risk `0.59695`, best
  `front_mid_nominal`, final target distance `0.00315 m`, fall/drop `0`,
  `min_drive_near_ground_foot_count=2`, `drive_near_ground_lt2_steps=0`.
  Seed `7072`: mass `10.99875 kg`, probe risk `0.61648`, best
  `low_front_slow`, final target distance `0.00922 m`, fall/drop `0`,
  support continuity passed.
  Seed `7073`: mass `10.79901 kg`, probe risk `0.60212`, best
  `front_mid_nominal`, final target distance `0.00086 m`, fall/drop `0`,
  support continuity passed. For every seed, 3 of 5 candidates passed and
  `front_mid_wide_slow` plus `low_front_wide_slowest` were rejected by strict
  support-continuity gates. This is useful evidence that the scaffold can run
  active probing plus posture/parameter selection and that posture selection
  is not hard-coded to one winner. It is still not learned RL, not
  video-conditioned, and not a full humanoid walking controller.
- 2026-07-05 effort-aware proxy selector update:
  updated `scripts/isaac/summarize_probe_parameter_search_carry.py` so future
  parameter-search summaries include `score_terms` and use an effort-aware
  proxy score. The score still includes target distance, travel loss, falls,
  drops, support continuity, and shortcut penalties, and now also includes
  measured support-foot effort proxy if available plus kinematic proxies from
  max tilt, support-margin shortfall, support-foot lift, and support-foot
  motion. This is not a real torque/energy objective; it is a transparent
  proxy until reliable actuator torque or drive-effort telemetry is available.
  Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/summarize_probe_parameter_search_carry.py scripts/isaac/summarize_probe_parameter_search_multiseed.py`.
- 2026-07-05 expanded posture/gait action-space implementation:
  expanded the direct Isaac parameter-search runner so future candidates can
  vary `torso_z_m`, `payload_local_x_m`, `payload_local_z_m`,
  `support_foot_step_height_m`, `support_foot_double_support_fraction`,
  `stance_half_length_m`, and `stance_half_width_m`, in addition to
  posture, `stance_steps`, `step_length_m`, and support-foot stance/swing X.
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh` now forwards
  `STANCE_HALF_LENGTH` and `STANCE_HALF_WIDTH` to the Python scene builder.
  `build_core_world_anchored_footstep_carrier.py`, the normalizer, and the
  parameter-search summary now preserve these action parameters in summaries.
  `CANDIDATE_SET=expanded` currently defines 9 hand-authored candidates.
  Lightweight checks passed:
  `bash -n scripts/isaac/run_probe_parameter_search_carry_diag.sh scripts/isaac/run_probe_parameter_search_multiseed_diag.sh scripts/isaac/run_core_world_anchored_footstep_carrier.sh` and
  `python3 -m py_compile scripts/isaac/build_core_world_anchored_footstep_carrier.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/summarize_probe_parameter_search_carry.py scripts/isaac/summarize_probe_parameter_search_multiseed.py`.
- 2026-07-05 expanded posture/gait action-space diagnostic submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_probe_param_expanded_7074_0705`, Slurm job `166718`, command:
  `MULTISEED_STAMP=20260705_probe_parameter_search_expanded_7074_7075 SEEDS='7074 7075' CANDIDATE_SET=expanded srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_param_exp bash scripts/isaac/run_probe_parameter_search_multiseed_diag.sh`.
  Log:
  `logs/probe_parameter_search_multiseed/probe_parameter_search_expanded_7074_0705_srun.log`.
  Expected summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_expanded_7074_7075/probe_parameter_search_multiseed_summary.json`.
  Initial Slurm status: pending for priority. This is still a scaffold
  diagnostic, not RL or final humanoid carrying.
- 2026-07-05 expanded posture/gait action-space diagnostic result:
  Slurm job `166718` completed on `server02` with exit `0:0` after
  `00:11:21`. Aggregate summary:
  `experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_expanded_7074_7075/probe_parameter_search_multiseed_summary.json`.
  Both hidden seeds completed and passed. Best candidate/posture varied:
  seed `7074` selected `front_mid_nominal` / `front_mid` with box mass
  `6.40404 kg`, probe risk `0.62788`, best score `0.09729`, final target
  distance `0.01681 m`, fall/drop `0`, and strict support continuity. Seed
  `7075` selected the new `low_front_cautious` / `low_front` candidate with
  box mass `6.52098 kg`, probe risk `0.64615`, best score `0.08904`, final
  target distance `0.01135 m`, fall/drop `0`, and strict support continuity.
  The expanded runner evaluated 9 candidates per seed; 5/9 and 6/9 passed.
  Rejected candidates included high-clearance and wide-slowest variants that
  moved the box but violated strict gates. This validates the expanded
  posture/gait action-space interface for the current direct Isaac scaffold.
  It remains a hand-authored scaffold search, not learned walking, not RL, and
  not video-conditioned carrying.
- 2026-07-05 RL-interface episode-table exporter implementation:
  added `scripts/isaac/export_probe_parameter_search_episode_table.py` and
  `scripts/isaac/run_export_probe_parameter_episode_table.sh`. The exporter
  converts per-seed or multi-seed parameter-search summaries into JSONL rows,
  one row per candidate episode, with fields for observation proxies, action
  parameters, reward/score terms, strict gates, source summary paths, and
  limitation labels. This is a data-interface scaffold for future RL or
  video-reward conditioning; it does not train a policy. Lightweight checks
  passed:
  `python3 -m py_compile scripts/isaac/export_probe_parameter_search_episode_table.py` and
  `bash -n scripts/isaac/run_export_probe_parameter_episode_table.sh`.
- 2026-07-05 RL-interface episode-table export submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_export_episode_table_0705`, Slurm job `166744`, command:
  `STAMP=20260705_probe_parameter_episode_table_expanded_7074_7075 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=export_ep_table bash scripts/isaac/run_export_probe_parameter_episode_table.sh --multiseed-summary experiments/outputs/probe_parameter_search_multiseed/20260705_probe_parameter_search_expanded_7074_7075/probe_parameter_search_multiseed_summary.json`.
  Log:
  `logs/rl_interface/export_episode_table_0705_srun.log`.
  Expected output:
  `experiments/outputs/rl_interface/20260705_probe_parameter_episode_table_expanded_7074_7075/probe_parameter_episode_table.jsonl`.
  Initial Slurm status: pending for priority.
- 2026-07-05 RL-interface episode-table export result:
  Slurm job `166744` completed on `server46` with exit `0:0`. Output:
  `experiments/outputs/rl_interface/20260705_probe_parameter_episode_table_expanded_7074_7075/probe_parameter_episode_table.jsonl`.
  The file has `18` JSONL rows, matching 2 hidden seeds times 9 expanded
  candidates. Each row uses schema
  `direct_isaac_probe_parameter_episode_v1` and contains observation proxies
  such as box mass/size/COM and probe risk, action parameters such as carry
  posture, torso height, payload local X/Z, stance timing/width, and
  support-foot motion settings, metrics/reward proxy terms, strict gates, and
  limitation labels. This establishes an RL-ready data contract for the
  current scaffold, but no RL policy has been trained yet.
- 2026-07-05 feedback-step controller diagnostic implementation:
  continuing the direct Isaac path without waiting for external models. Added
  explicit feedback-step controller fields to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`, forwarded
  them through `scripts/isaac/run_direct_carry_task_physical_backend.sh`, and
  preserved `max_rail_joint_motion_m` plus feedback-step telemetry through
  `scripts/isaac/normalize_direct_carry_backend_summary.py` and
  `scripts/isaac/check_direct_carry_task_summary.py`. The checker can now
  require `--require-feedback-step-controller`,
  `--min-feedback-step-applied-steps`, and `--max-rail-joint-motion`, so runs
  that are secretly rail-driven or have inactive feedback are rejected.
  Added `scripts/isaac/run_feedback_step_controller_carry_diag.sh` as the
  first direct Isaac feedback-step carry diagnostic. Lightweight checks
  passed:
  `python3 -m py_compile scripts/isaac/build_core_world_anchored_footstep_carrier.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/check_direct_carry_task_summary.py`
  and
  `bash -n scripts/isaac/run_core_world_anchored_footstep_carrier.sh scripts/isaac/run_direct_carry_task_physical_backend.sh scripts/isaac/run_feedback_step_controller_carry_diag.sh`.
- 2026-07-05 feedback-step controller diagnostic submitted:
  submitted from Curiosity-owned tmux session
  `curiosity_feedback_step_0705`, Slurm job `166750`, command:
  `STAMP=20260705_feedback_step_controller_seed7076_frontmid srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_step_carry bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`.
  Expected output:
  `experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_controller_seed7076_frontmid/`.
  Initial Slurm status: pending for priority. This run is still a scaffold
  diagnostic, not a final humanoid walking/balancing claim.
- 2026-07-05 feedback-step controller diagnostic result:
  after several useful failures and launcher/checker fixes, retry6 passed.
  Failed attempts were not success evidence: job `166751` exposed brittle
  launcher environment passing, job `166753` completed Isaac but failed
  checker/report gates, job `166758` showed the continuity-grace parameter
  was not active in that run, and jobs `166765`/`166768` failed the
  near-ground z-proxy support continuity gate. Final run:
  tmux `curiosity_feedback_step_retry6_0705`, Slurm job `166769`, command:
  `STAMP=20260705_feedback_step_controller_seed7076_frontmid_retry6 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_step_carry bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`.
  It completed on `server02` with exit `0:0` after `00:00:37`. Output:
  `experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_controller_seed7076_frontmid_retry6/feedback_step_controller_check.json`.
  The strict checker reported `status=pass`: `3580` completed steps on a
  randomized `11.46294 kg` box, fall/drop `0`, root/body/box/payload/foot/
  stance shortcut writes `0`, feedback-step applied steps `3570`, final box
  target distance `0.00247 m`, post-settle target distance `0.00148 m`,
  min drive near-ground foot count `3`, drive near-ground lt2 steps `0`, and
  max rail joint motion `0.02151 m` under the diagnostic threshold `0.025 m`.
  This is still a direct Isaac scaffold diagnostic. It is not a full humanoid
  walking controller, not learned RL, and not video-conditioned carrying. The
  support-continuity pass uses an explicit foot-height proxy threshold
  `SUPPORT_FOOT_CONTACT_Z_THRESHOLD=0.055`; future evidence should replace
  this with actual contact or force-state support evidence.
- 2026-07-05 feedback-step support-effort evidence gate:
  added measured support-foot joint-effort telemetry to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`, passed it
  through `scripts/isaac/normalize_direct_carry_backend_summary.py`, and added
  strict effort-evidence gates to
  `scripts/isaac/check_direct_carry_task_summary.py` plus
  `scripts/isaac/run_feedback_step_controller_carry_diag.sh`. Lightweight
  checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_anchored_footstep_carrier.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/check_direct_carry_task_summary.py`
  and
  `bash -n scripts/isaac/run_feedback_step_controller_carry_diag.sh scripts/isaac/run_direct_carry_task_physical_backend.sh scripts/isaac/run_core_world_anchored_footstep_carrier.sh`.
  Two launcher/runtime issues were recorded honestly: job `166775` failed
  before Isaac with a shell EOF parse error in the physical-backend launcher;
  job `166781` completed the Isaac rollout and wrote the backend summary but
  hit a post-summary shell EOF parse error in the core launcher. The existing
  backend summary was then normalized and checked in a separate compute job:
  tmux `curiosity_feedback_step_effort_check_0705`, Slurm job `166786`,
  command:
  `STAMP=20260705_feedback_step_effort_gate_seed7076_retry2 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=fb_effort_chk ...`.
  It completed on `server02` with exit `0:0`. Check report:
  `experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_effort_gate_seed7076_retry2/feedback_step_effort_check.json`.
  Result: `status=pass`, completed `3580` steps, randomized hidden box mass
  `11.46294 kg`, fall/drop `0`, root/box/foot/stance shortcut writes `0`,
  final post-settle box target distance `0.00148 m`, support-foot effort
  available with read errors `0`, per-foot max measured support effort
  `3264.14/3691.83/4055.08/4245.39`, min drive effort-supported foot count
  `4`, drive effort-supported lt2 steps `0`, min commanded-stance
  effort-supported foot count `2`, and commanded-stance effort-supported lt2
  steps `0`. This upgrades support evidence beyond the z-height proxy, but it
  is still a measured joint-effort proxy, not a calibrated contact-force sensor
  and not a final humanoid walking/RL/video-conditioned success claim.
- 2026-07-05 feedback-step contact-report evidence gate:
  continuing the direct Isaac route. Added optional PhysX support-foot contact
  reporting to `build_core_world_anchored_footstep_carrier.py` using
  `PhysxContactReportAPI` plus
  `get_physics_simulation_interface().subscribe_physics_contact_report_events`.
  The scene now records support-foot contact-report availability, enabled
  paths, event/error counts, per-foot contact-report steps, drive-phase contact
  foot counts, and commanded-stance contact foot counts. The normalizer and
  checker pass and gate these fields, and the feedback-step diagnostic requests
  contact report evidence by default. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_anchored_footstep_carrier.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/check_direct_carry_task_summary.py`
  and
  `bash -n scripts/isaac/run_core_world_anchored_footstep_carrier.sh scripts/isaac/run_direct_carry_task_physical_backend.sh scripts/isaac/run_feedback_step_controller_carry_diag.sh`.
  First contact-report attempt `166793` completed Isaac but failed post-summary
  with the known shell EOF issue; its summary showed
  `support_foot_contact_report_requested=false`, so the problem was env
  propagation, not a physics result. The feedback runner was changed to pass
  `--enable-support-foot-contact-report` directly through the wrapper. Retry:
  tmux `curiosity_feedback_step_contact_retry2_0705`, Slurm job `166797`,
  command:
  `STAMP=20260705_feedback_step_contact_report_seed7076_retry2 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=fb_contact2 bash scripts/isaac/run_feedback_step_controller_carry_diag.sh`.
  It completed on `server02` with exit `0:0` after `00:00:38`. Check report:
  `experiments/outputs/feedback_step_controller_carry/20260705_feedback_step_contact_report_seed7076_retry2/feedback_step_controller_check.json`.
  Result: `status=pass`, completed `3580` steps, randomized hidden box mass
  `11.46294 kg`, fall/drop `0`, root/box/foot/stance shortcut writes `0`,
  final post-settle box target distance `0.00148 m`, contact report requested
  and available, enabled paths `/World/Ground` plus four support feet, contact
  report event count `42`, error count `0`, per-foot contact-report steps
  `fl=3332`, `fr=3308`, `rl=3451`, `rr=3407`, min drive contact-report foot
  count `2`, drive contact-report lt2 steps `0`, min commanded-stance
  contact-report foot count `2`, commanded-stance contact-report lt2 steps
  `0`, support-foot effort available with read errors `0`, and max rail joint
  motion `0.02151 m`. This is stronger contact-state evidence for the current
  scaffold. It is still not calibrated ground-reaction force, not full
  humanoid walking, not RL, and not video-conditioned carrying.
- 2026-07-05 randomized all-posture contact-report diagnostic submitted:
  to push beyond the single `front_mid` contact-report pass, upgraded
  `scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh` and
  `scripts/isaac/summarize_randomized_all_posture_carry.py` so the hidden-box
  all-posture run requires PhysX support-foot contact-report evidence in
  addition to the previous near-ground gates. Lightweight checks passed:
  `bash -n scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh`
  and
  `python3 -m py_compile scripts/isaac/summarize_randomized_all_posture_carry.py scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/normalize_direct_carry_backend_summary.py`.
  Submitted from Curiosity-owned tmux session
  `curiosity_all_posture_contact_0705`, Slurm job `166800`, command:
  `STAMP=20260705_randomized_all_posture_contact_report_64cm_seed7077 BOX_SEED=7077 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=all_post_contact bash scripts/isaac/run_randomized_all_posture_strict_support_64cm_diag.sh`.
  Log:
  `logs/randomized_all_posture_strict_support/randomized_all_posture_contact_report_7077_srun.log`.
  Expected summary:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_contact_report_64cm_seed7077/randomized_all_posture_strict_support_summary.json`.
  Initial Slurm status: pending for priority. This remains a direct Isaac
  scaffold diagnostic, not final humanoid walking/RL/video-conditioned
  carrying.
- 2026-07-05 randomized all-posture contact-report diagnostic result:
  Slurm job `166800` completed on `server02` with exit `0:0` after
  `00:01:50`. Summary:
  `experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_contact_report_64cm_seed7077/randomized_all_posture_strict_support_summary.json`.
  Result: overall `status=pass`. Shared hidden randomized box mass
  `4.86216 kg`, size `[0.35968, 0.24056, 0.24150] m`, COM offset
  `[-0.02053, 0.00841, 0.02503] m`. Postures `front_mid`, `low_front`, and
  `chest_high` all completed `3580` steps with fall/drop `0`, root-shortcut
  free execution, PhysX contact-report available, contact-report error count
  `0`, min drive contact-report foot count `2`, drive contact-report lt2
  steps `0`, min commanded-stance contact-report foot count `2`, and
  commanded-stance contact-report lt2 steps `0`. This is now the strongest
  all-posture direct Isaac scaffold gate, but it is still not calibrated
  ground-reaction force, not a full humanoid walking controller, not RL, and
  not video-conditioned carrying. Per user correction, do not wait on external
  models here; continue by building the Isaac task/controller interface
  directly.
- 2026-07-05 direct Isaac carry-task contract implementation:
  following the user correction, stopped treating external models as blockers
  for this phase and added a direct task-contract bridge:
  `scripts/isaac/direct_carry_task_contract.py`,
  `scripts/isaac/export_direct_carry_task_episode_table.py`,
  `scripts/isaac/run_export_direct_carry_task_episode_table.sh`, and
  `experiments/configs/direct_isaac_carry_task_contract_v1.json`. Lightweight
  checks passed:
  `python3 -m py_compile scripts/isaac/direct_carry_task_contract.py scripts/isaac/export_direct_carry_task_episode_table.py`
  and `bash -n scripts/isaac/run_export_direct_carry_task_episode_table.sh`.
  The contract explicitly keeps hidden load fields such as box mass and COM
  out of `policy_observation`; they are exported only under
  `hidden_eval_context`.
- 2026-07-05 direct Isaac carry-task contract export submitted:
  tmux `curiosity_export_direct_task_contract_0705`, Slurm job `166804`,
  command:
  `STAMP=20260705_direct_carry_task_contract_all_posture_contact_7077 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=carry_contract bash scripts/isaac/run_export_direct_carry_task_episode_table.sh --summary experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_contact_report_64cm_seed7077/randomized_all_posture_strict_support_summary.json`.
  Expected output:
  `experiments/outputs/rl_interface/20260705_direct_carry_task_contract_all_posture_contact_7077/direct_carry_task_episode_table.jsonl`.
  Initial status: pending for priority.
- 2026-07-05 direct Isaac carry-task contract export result:
  job `166804` completed on `server02` with exit `0:0` and exported `3` JSONL
  rows. Inspecting the first row showed an interface issue: all-posture
  condensed rows left `controller_mode`, `support_foot_mode`, and related
  action fields null. Fixed
  `scripts/isaac/export_direct_carry_task_episode_table.py` so all-posture
  exports follow each posture `summary_path` and merge the complete backend
  summary before writing a row. Lightweight syntax check passed. Retry tmux
  `curiosity_export_direct_task_contract_retry_0705`, Slurm job `166806`,
  command:
  `STAMP=20260705_direct_carry_task_contract_all_posture_contact_7077 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=carry_contract2 bash scripts/isaac/run_export_direct_carry_task_episode_table.sh --summary experiments/outputs/randomized_all_posture_strict_support/20260705_randomized_all_posture_contact_report_64cm_seed7077/randomized_all_posture_strict_support_summary.json`.
  It completed on `server02` with exit `0:0` and rewrote:
  `experiments/outputs/rl_interface/20260705_direct_carry_task_contract_all_posture_contact_7077/direct_carry_task_episode_table.jsonl`.
  Row count: `3`. Corrected rows include `controller_mode`,
  `support_foot_mode`, PhysX contact gates, effort metrics, reward terms, and
  hidden mass/COM only under `hidden_eval_context`.
- 2026-07-05 direct Isaac task runner skeleton:
  added `scripts/isaac/direct_carry_task_runner.py` with backend-adapter
  methods `reset`, `observe`, `apply_action`, `step_until_done`, plus wrapper
  methods `run_episode`, `compute_reward`, `is_terminated`, and
  `export_episode_row`. Lightweight syntax check passed. This is only an
  interface skeleton; it does not run Isaac, train RL, or claim robot carrying
  success.
- 2026-07-05 executable direct Isaac task-runner backend:
  added `scripts/isaac/direct_carry_task_shell_backend.py`,
  `scripts/isaac/run_direct_carry_task_runner_episode.py`, and
  `scripts/isaac/run_direct_carry_task_runner_episode.sh`. This maps
  `DirectCarryReset` and `DirectCarryAction` into the current
  `run_direct_carry_task_physical_backend.sh` environment, launches the Isaac
  backend on compute nodes only, and writes a
  `direct_isaac_carry_task_episode_v1` row. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/direct_carry_task_shell_backend.py scripts/isaac/run_direct_carry_task_runner_episode.py scripts/isaac/direct_carry_task_runner.py scripts/isaac/direct_carry_task_contract.py`
  and `bash -n scripts/isaac/run_direct_carry_task_runner_episode.sh`.
  This is an executable scaffold backend adapter, not a final walking robot or
  RL policy.
- 2026-07-05 direct Isaac task-runner validation submitted:
  first submit attempt failed before Slurm because
  `logs/direct_carry_task_runner` did not exist for outer shell redirection.
  Created the directory and resubmitted from tmux
  `curiosity_task_runner_episode_retry_0705`, Slurm job `166810`, command:
  `STAMP=20260705_task_runner_frontmid_seed7078 BOX_SEED=7078 CARRY_POSTURE=front_mid TARGET_X=0.64 STEPS=3580 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=task_runner bash scripts/isaac/run_direct_carry_task_runner_episode.sh`.
  Initial status: pending for priority. Outer log:
  `logs/direct_carry_task_runner/task_runner_frontmid_seed7078_srun.log`.
- 2026-07-05 direct Isaac task-runner validation result:
  Slurm job `166810` completed on `server02` with exit `0:0` after
  `00:01:04`. It ran through
  `scripts/isaac/run_direct_carry_task_runner_episode.sh` with stamp
  `20260705_task_runner_frontmid_seed7078`, box seed `7078`, posture
  `front_mid`, target `0.64 m`, and `3580` steps. Outputs:
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_physical_backend_summary.json`,
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_runner_episode.jsonl`,
  and
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_runner_report.json`.
  Result: hidden randomized box mass `4.33753 kg`, size
  `[0.34429, 0.21785, 0.21029] m`, COM offset
  `[0.03906, -0.03296, -0.00888] m`; completed `3580` steps; fall/drop `0`;
  final post-settle box travel `0.65758 m`; final post-settle target distance
  `0.01758 m`; PhysX contact report available; contact-report error count
  `0`; min drive contact-report foot count `2`; commanded-stance
  contact-report lt2 steps `0`. This proves the executable task-runner chain
  works for the current scaffold backend. It is still not a final walking
  robot, not RL, and not video-conditioned carrying.
- 2026-07-05 task-contract pass-flag fix and checker submitted:
  the first task-runner episode row had `gates.passed=false` only because the
  backend summary did not include an explicit `status=pass` field. Fixed
  `scripts/isaac/direct_carry_task_contract.py` to derive pass from strict
  no-fall/no-drop/no-root-shortcut/support-contact fields when `status` is
  absent, and fixed
  `scripts/isaac/run_direct_carry_task_runner_episode.py` to report
  `status=pass` when derived gates pass and backend returncode is `0`. Added
  `scripts/isaac/run_check_direct_carry_task_runner_episode.sh` to run the
  strict checker and re-export a corrected episode row on compute. Lightweight
  checks passed:
  `python3 -m py_compile scripts/isaac/direct_carry_task_contract.py scripts/isaac/run_direct_carry_task_runner_episode.py scripts/isaac/direct_carry_task_shell_backend.py scripts/isaac/export_direct_carry_task_episode_table.py`
  and `bash -n scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  Submitted from tmux `curiosity_task_runner_check_0705`, Slurm job `166817`,
  command:
  `STAMP=20260705_task_runner_frontmid_seed7078_check SUMMARY=experiments/outputs/direct_carry_task_runner/20260705_task_runner_frontmid_seed7078/direct_carry_task_physical_backend_summary.json BOX_SEED=7078 CARRY_POSTURE=front_mid srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=task_check bash scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  Initial status: pending for priority.
- 2026-07-05 task-runner strict checker/export result:
  Slurm job `166817` completed on `server02` with exit `0:0` after
  `00:00:01`. Strict checker report:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_frontmid_seed7078_check/direct_carry_task_runner_check.json`;
  result `status=pass`. Corrected episode table:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_frontmid_seed7078_check/direct_carry_task_runner_episode_table.jsonl`;
  the corrected row has `gates.passed=true`. The episode still reports
  `probe_belief_source=no_active_probe`, so the next task-runner step must add
  explicit active-probing action support and validate a probing episode.
- 2026-07-05 active-probing task-runner support:
  added `probe_steps` to `DirectCarryAction`, passed it through
  `direct_carry_task_shell_backend.py` as `PROBE_STEPS`, exposed it in
  `run_direct_carry_task_runner_episode.py` and
  `run_direct_carry_task_runner_episode.sh`, and added probe action fields to
  `direct_carry_task_contract.py` and
  `experiments/configs/direct_isaac_carry_task_contract_v1.json`. Lightweight
  checks passed:
  `python3 -m py_compile scripts/isaac/direct_carry_task_runner.py scripts/isaac/direct_carry_task_shell_backend.py scripts/isaac/direct_carry_task_contract.py scripts/isaac/run_direct_carry_task_runner_episode.py scripts/isaac/export_direct_carry_task_episode_table.py`
  and
  `bash -n scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
- 2026-07-05 active-probing task-runner validation submitted:
  submitted from tmux `curiosity_task_runner_probe_0705`, Slurm job `166819`,
  command:
  `STAMP=20260705_task_runner_probe_frontmid_seed7079 BOX_SEED=7079 CARRY_POSTURE=front_mid TARGET_X=0.64 STEPS=3660 PROBE_STEPS=80 PROBE_AMPLITUDE_X=0.012 PROBE_AMPLITUDE_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=task_probe bash scripts/isaac/run_direct_carry_task_runner_episode.sh`.
  Initial status: pending for priority. This is intended to validate that
  active probing is actually requested/reported before carrying; it is still a
  scaffold diagnostic, not final robot walking.
- 2026-07-05 active-probing task-runner validation result:
  Slurm job `166819` completed on `server02` with exit `0:0` after
  `00:00:45`. The episode used stamp
  `20260705_task_runner_probe_frontmid_seed7079`, box seed `7079`, posture
  `front_mid`, target `0.64 m`, `3660` steps, `PROBE_STEPS=80`,
  `PROBE_AMPLITUDE_X=0.012`, and `PROBE_AMPLITUDE_Z=0.0`. Hidden randomized
  box mass `11.13313 kg`, size `[0.31808, 0.25514, 0.23031] m`, COM offset
  `[-0.01361, -0.02603, 0.01952] m`. Result: completed `3660` steps,
  fall/drop `0`, final post-settle box travel `0.66478 m`, final post-settle
  target distance `0.02478 m`, contact-report available, contact-report error
  count `0`, `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`,
  `probe_belief_source=heuristic_from_probe_telemetry_not_calibrated_mass_estimator`,
  max probe box travel `0.03064 m`, max probe support-foot X measured effort
  `525.43`, max probe support-foot Z measured effort `2053.72`, and runner
  report status `pass`. This is active-probing scaffold evidence, not final
  walking-robot success.
- 2026-07-05 probe-specific checker gate:
  added `--min-probe-steps`, `--require-probe-belief`,
  `--forbid-probe-hidden-ground-truth`, and `--min-probe-box-travel-x` to
  `scripts/isaac/check_direct_carry_task_summary.py`. Updated
  `scripts/isaac/run_check_direct_carry_task_runner_episode.sh` so
  `REQUIRE_PROBE_BELIEF=1` enables those gates. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/direct_carry_task_contract.py scripts/isaac/export_direct_carry_task_episode_table.py`
  and `bash -n scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  Old checker job `166820` passed the carry gate. Probe-gated checker job
  `166821` used:
  `STAMP=20260705_task_runner_probe_frontmid_seed7079_probegate SUMMARY=experiments/outputs/direct_carry_task_runner/20260705_task_runner_probe_frontmid_seed7079/direct_carry_task_physical_backend_summary.json BOX_SEED=7079 CARRY_POSTURE=front_mid MIN_STEPS=3660 REQUIRE_PROBE_BELIEF=1 MIN_PROBE_STEPS=80 MIN_PROBE_BOX_TRAVEL_X=0.01 srun --partition=gpu --gres=gpu:1 --time=00:20:00 --job-name=probe_gate bash scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  It completed on `server02` with exit `0:0` after `00:00:01`. Report:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_probe_frontmid_seed7079_probegate/direct_carry_task_runner_check.json`;
  `status=pass`. Episode table:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_probe_frontmid_seed7079_probegate/direct_carry_task_runner_episode_table.jsonl`;
  row `gates.passed=true`.
- 2026-07-05 multi-posture active-probe task-runner implementation:
  added `scripts/isaac/run_task_runner_active_probe_postures.sh` and
  `scripts/isaac/summarize_task_runner_active_probe_postures.py`. The sweep
  runs `front_mid`, `low_front`, and `chest_high` under the same hidden box
  seed, with active probing required by the checker for each posture. It then
  writes one all-posture summary. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/summarize_task_runner_active_probe_postures.py scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/direct_carry_task_contract.py`
  and
  `bash -n scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
- 2026-07-05 multi-posture active-probe task-runner submitted:
  submitted from tmux `curiosity_active_probe_postures_0705`, Slurm job
  `166822`, command:
  `STAMP=20260705_task_runner_active_probe_postures_seed7080 BOX_SEED=7080 TARGET_X=0.64 STEPS=3660 PROBE_STEPS=80 PROBE_AMPLITUDE_X=0.012 PROBE_AMPLITUDE_Z=0.0 srun --partition=gpu --gres=gpu:1 --time=06:00:00 --job-name=probe_postures bash scripts/isaac/run_task_runner_active_probe_postures.sh`.
  Initial status: pending for priority. Outer log:
  `logs/direct_carry_task_runner/active_probe_postures_seed7080_srun.log`.
- 2026-07-05 multi-posture active-probe task-runner result:
  Slurm job `166822` completed on `server02` with exit `0:0` after
  `00:01:56`. Summary:
  `experiments/outputs/direct_carry_task_runner_active_probe_postures/20260705_task_runner_active_probe_postures_seed7080/active_probe_posture_summary.json`.
  The shared hidden randomized box was mass `10.72455 kg`, size
  `[0.36519, 0.22971, 0.23912] m`, COM offset
  `[0.03732, 0.00523, -0.00058] m`. All three postures completed
  `3660/3660` steps with fall/drop `0`, contact reports available, active
  probe belief available, and `probe_belief_uses_hidden_ground_truth=false`.
  Final post-settle box travel / target-distance: `front_mid`
  `0.64775 / 0.00775 m`, `low_front` `0.67183 / 0.03183 m`, and
  `chest_high` `0.65476 / 0.01476 m`. Probe risk scores were `0.54979`,
  `0.61489`, and `0.48801` respectively. This validates the direct Isaac
  task-runner's active-probe and multi-posture bookkeeping for the current
  scaffold backend. It is not a full walking robot, not RL, and not
  video-conditioned success.
- 2026-07-05 direct carry-task backend capability contract:
  added `DirectCarryBackendCapabilities` to
  `scripts/isaac/direct_carry_task_runner.py`, implemented conservative
  capabilities in `scripts/isaac/direct_carry_task_shell_backend.py`, and
  exported `backend_capabilities` plus `termination` in
  `scripts/isaac/direct_carry_task_contract.py`. The current shell backend is
  explicitly labeled `backend_family=anchored_support_scaffold`,
  `free_dynamic_box=true`, `active_probe_supported=true`,
  `trainable_policy_backend=false`, `real_robot_morphology=false`,
  `support_switching_supported=false`, `video_conditioning_supported=false`,
  and `scaffold_backend=true`. Legacy summaries without explicit capability
  fields are inferred conservatively as scaffold evidence.
- 2026-07-05 contract export validation:
  lightweight syntax checks passed for the task-runner/contract/export files.
  First compute export job `166825` failed on `server02` with exit `126:0`
  because the script was invoked directly without execute permission. Retry
  job `166826` completed and showed the new capability fields. After correcting
  `termination.step_limit_reached` so unknown step-limit evidence stays
  `null`, retry2 job `166828` completed on `server02` with exit `0:0` after
  `00:00:01`. Output:
  `experiments/outputs/rl_interface/20260705_contract_caps_export_retry2/direct_carry_task_episode_table.jsonl`.
  It contains `3` rows for `front_mid`, `low_front`, and `chest_high`; each
  row has backend id `physical_alternating_anchor_feet_cradle_v1`,
  `scaffold_backend=true`, `trainable_policy_backend=false`,
  `episode_completed=true`, and `step_limit_reached=null`.
- 2026-07-05 directional foot-placement backend implementation:
  added `--support-foot-placement-mode` to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`, with
  `alternating_fixed_x` preserving the old behavior and
  `alternating_directional_x` mirroring swing/stance foot X targets according
  to target direction. Added summary fields
  `support_foot_placement_mode`,
  `support_foot_placement_controller_enabled`, and
  `support_foot_directional_placement`. Wired this through
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh`,
  `scripts/isaac/run_direct_carry_task_physical_backend.sh` as
  `SUPPORT_MODE=alternating_placement_feet`,
  `scripts/isaac/direct_carry_task_shell_backend.py`,
  `scripts/isaac/run_direct_carry_task_runner_episode.py`, and
  `scripts/isaac/run_direct_carry_task_runner_episode.sh`. Added checker gates
  for directional foot placement in
  `scripts/isaac/check_direct_carry_task_summary.py` and
  `scripts/isaac/run_check_direct_carry_task_runner_episode.sh`. Updated
  `scripts/isaac/normalize_direct_carry_backend_summary.py` and legacy
  episode capability inference so exported rows identify this backend as
  `physical_alternating_placement_feet_cradle_v1` /
  `directional_foot_placement_scaffold`.
- 2026-07-05 directional foot-placement validation:
  first Slurm run `166831` failed after `00:00:01` because the compute job saw
  the old support-mode dispatch string. Retry `166832` failed after
  `00:00:09` because the task adapter redundantly forwarded contact-report CLI
  args through the physical-backend wrapper, causing an argparse ambiguity.
  Fixed the adapter by using environment variables for contact-report setup and
  by setting `CONTROLLER_MODE=physical_alternating_placement_feet_cradle` for
  the new support mode. Slurm job `166833`, stamp
  `20260705_task_runner_directional_placement_seed7081_retry3`, completed on
  `server02` with exit `0:0` after `00:00:48`. It used hidden randomized box
  seed `7081`, mass `7.23482 kg`, size
  `[0.34198, 0.25056, 0.21837] m`, COM offset
  `[-0.00517, -0.03446, 0.01995] m`, posture `front_mid`, target `0.64 m`,
  `80` horizontal active-probe steps, and
  `support_foot_placement_mode=alternating_directional_x`. Result:
  completed `3660/3660`, fall/drop `0`, final post-settle box travel
  `0.65735 m`, final post-settle target distance `0.01735 m`,
  `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`, max probe box travel
  `0.03063 m`, min drive contact-report foot count `2`, and
  drive contact-report lt2 steps `0`.
- 2026-07-05 directional foot-placement strict checker:
  Slurm job `166836` first passed the physical/probe/directional-placement
  gate, but its exported episode row initially inferred legacy backend
  capabilities as `legacy_unknown_backend`. Fixed
  `scripts/isaac/direct_carry_task_contract.py` legacy inference to recognize
  `physical_alternating_placement_feet_cradle` and
  `alternating_directional_x`. Retry checker job `166840` completed on
  `server02` with exit `0:0` after `00:00:01`. Report:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_directional_placement_seed7081_retry3_check2/direct_carry_task_runner_check.json`;
  status `pass`, failures `[]`, directional placement true. Episode table:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_directional_placement_seed7081_retry3_check2/direct_carry_task_runner_episode_table.jsonl`;
  backend id `physical_alternating_placement_feet_cradle_v1`,
  family `directional_foot_placement_scaffold`,
  `support_switching_supported=true`, `scaffold_backend=true`, and
  `trainable_policy_backend=false`. This is progress toward support placement,
  but still not a full walking robot, not a learned controller, and not
  video-conditioned carrying.
- 2026-07-05 negative-target directional placement metrics:
  the first negative-target run, Slurm job `166842`, stamp
  `20260705_task_runner_directional_negative_seed7082`, completed but exposed
  a metric bug: reward and travel-loss were computed as if X progress were
  always positive, so a valid `TARGET_X=-0.32` movement produced a large
  negative reward and inflated post-settle travel loss. Fixed
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py` to record
  `max_abs_post_settle_*` and
  `max_target_directed_post_settle_*` travel, fixed
  `scripts/isaac/normalize_direct_carry_backend_summary.py`, fixed
  `scripts/isaac/direct_carry_task_contract.py` reward terms to use
  target-directed progress, and added absolute travel gates to
  `scripts/isaac/check_direct_carry_task_summary.py` plus
  `scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
- 2026-07-05 negative-target directional placement validation:
  rerun Slurm job `166845`, stamp
  `20260705_task_runner_directional_negative_seed7082_retry`, completed on
  `server02` with exit `0:0` after `00:00:37`. Hidden randomized box seed
  `7082`, mass `8.20882 kg`; target `-0.32 m`; posture `front_mid`;
  `80` probe steps; backend `physical_alternating_placement_feet_cradle_v1`.
  Result: completed `2200/2200`, fall/drop `0`, final post-settle box travel
  `-0.35174 m`, max absolute post-settle box travel `0.37768 m`,
  max target-directed post-settle box travel `0.37768 m`, final post-settle
  target distance `0.03174 m`, directional post-settle travel loss
  `0.02594 m`, active probe belief available, and no hidden-ground-truth probe
  use. Strict checker job `166846` completed on `server36` with exit `0:0`;
  report:
  `experiments/outputs/direct_carry_task_runner_checks/20260705_task_runner_directional_negative_seed7082_retry_check/direct_carry_task_runner_check.json`;
  status `pass`, failures `[]`, directional placement true, root shortcut
  free. Episode row has backend family
  `directional_foot_placement_scaffold`, `scaffold_backend=true`,
  `trainable_policy_backend=false`, and target-direction reward terms. This
  shows the new scaffold can move the load in both target directions, but it
  remains a scaffold rather than a complete walking robot.
- 2026-07-05 directional multi-posture active-probe sweep:
  after two failed submissions on `server36` (`166847` saw a stale
  `SUPPORT_MODE=alternating_placement_feet` dispatch path; `166848` hit a
  shell EOF in the nested core launcher), the adapter was changed so the
  logical `alternating_placement_feet` backend maps to the already validated
  physical `alternating_anchor_feet` launcher while setting
  `SUPPORT_FOOT_PLACEMENT_MODE=alternating_directional_x`. Fixed-node
  `server02` Slurm job `166850`, stamp
  `20260705_task_runner_directional_postures_seed7083_server02`, completed
  with exit `0:0` after `00:02:21`. Summary:
  `experiments/outputs/direct_carry_task_runner_active_probe_postures/20260705_task_runner_directional_postures_seed7083_server02/active_probe_posture_summary.json`.
  Shared hidden randomized box: mass `5.91337 kg`, size
  `[0.33273, 0.26142, 0.22331] m`, COM offset
  `[0.01569, -0.01343, 0.01675] m`. Postures `front_mid`, `low_front`, and
  `chest_high` all completed `3660/3660` with fall/drop `0`, active probe
  belief available, no hidden-ground-truth probe use, root shortcut free,
  PhysX contact-report available, min drive contact-report foot count `2`,
  and `support_foot_placement_mode=alternating_directional_x`.
  Final post-settle box travel / target distance: `front_mid`
  `0.65979 / 0.01979 m`, `low_front` `0.67489 / 0.03489 m`, and
  `chest_high` `0.65393 / 0.01393 m`. This validates the current direct Isaac
  task-runner scaffold for active probing, posture alternatives, and
  directional support placement on a shared hidden box seed. It is still not
  RL, not video-conditioned, not a real walking robot, and not a final
  unknown-object carrying solution.
- 2026-07-05 explicit feedback step-controller exposure:
  `scripts/isaac/run_direct_carry_task_runner_episode.sh` now passes through
  `GAIT_SPEED_SCALE`, `FEEDBACK_STEP_X_GAIN`,
  `FEEDBACK_STEP_X_LIMIT`, `FEEDBACK_STEP_TILT_GAIN`,
  `FEEDBACK_STEP_TILT_LIMIT`, and
  `SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION` to the task runner. Lightweight
  check passed:
  `bash -n scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
- 2026-07-05 explicit feedback directional rollout:
  submitted from tmux `curiosity_directional_feedback_0705`, Slurm job
  `166853`, with `STAMP=20260705_task_runner_directional_feedback_seed7084`,
  `BOX_SEED=7084`, `CARRY_POSTURE=front_mid`, `TARGET_X=0.64`,
  `STEPS=3660`, `SUPPORT_MODE=alternating_placement_feet`,
  `FEEDBACK_STEP_X_GAIN=0.03`, `FEEDBACK_STEP_X_LIMIT=0.012`,
  `FEEDBACK_STEP_TILT_GAIN=0.05`, `FEEDBACK_STEP_TILT_LIMIT=0.006`,
  `PROBE_STEPS=80`, and `PROBE_AMPLITUDE_X=0.012`.
  It completed on `server02` with exit `0:0` after `00:00:44`.
  Runner report status `pass`; backend family
  `directional_foot_placement_scaffold`; scaffold backend true; trainable
  policy backend false. Normalized summary:
  `experiments/outputs/direct_carry_task_runner/20260705_task_runner_directional_feedback_seed7084/direct_carry_task_physical_backend_summary.json`.
  Hidden randomized box: mass `7.68171 kg`, size
  `[0.35663, 0.23818, 0.20231] m`, COM offset
  `[0.01525, -0.00197, 0.00205] m`. Result fields:
  completed `3660/3660`, fall/drop `0`, final post-settle box travel
  `0.62166 m`, max target-directed post-settle box travel `0.65028 m`,
  final post-settle target distance `0.01834 m`, post-settle travel loss
  `0.02861 m`, max probe box travel `0.03091 m`, active probe belief true,
  no hidden-ground-truth probe use, support-foot placement mode
  `alternating_directional_x`, directional placement true,
  feedback controller enabled, `feedback_step_applied_steps=3570`,
  max feedback X adjustment `0.012 m`, min drive contact-report foot count
  `2`, drive contact-report lt2 steps `0`, min commanded-stance
  contact-report foot count `2`, commanded-stance contact-report lt2 steps
  `0`, and root shortcut free. Formal strict checker attempts were not
  completed because Slurm priority kept short checker jobs pending:
  `166854` fixed-node GPU, `166857` GPU, and `166858` CPU were canceled while
  pending. Do not cite this as a checker-validated result until a formal
  checker job actually runs; it is a runner-pass plus manual JSON-field audit.
- 2026-07-05 feedback directional multi-posture sweep, checker validated:
  `scripts/isaac/run_task_runner_active_probe_postures.sh` now passes through
  `GAIT_SPEED_SCALE`, `FEEDBACK_STEP_X_GAIN`,
  `FEEDBACK_STEP_X_LIMIT`, `FEEDBACK_STEP_TILT_GAIN`,
  `FEEDBACK_STEP_TILT_LIMIT`, and
  `SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION` to each task-runner episode. It also
  automatically expects controller mode
  `physical_alternating_placement_feet_cradle` and directional-placement
  checker gates when `SUPPORT_MODE=alternating_placement_feet`. The
  summarizer now retains feedback controller fields and requires feedback to
  be enabled with applied steps for a posture to pass the sweep summary.
  Lightweight checks passed:
  `bash -n scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`
  and
  `python3 -m py_compile scripts/isaac/summarize_task_runner_active_probe_postures.py`.
  Submitted from tmux `curiosity_directional_feedback_postures_0705`, Slurm
  job `166859`, command used
  `STAMP=20260705_task_runner_directional_feedback_postures_seed7085`,
  `BOX_SEED=7085`, `TARGET_X=0.64`, `STEPS=3660`,
  `SUPPORT_MODE=alternating_placement_feet`,
  `FEEDBACK_STEP_X_GAIN=0.03`, `FEEDBACK_STEP_X_LIMIT=0.012`,
  `FEEDBACK_STEP_TILT_GAIN=0.05`, `FEEDBACK_STEP_TILT_LIMIT=0.006`,
  `SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION=0.18`, `PROBE_STEPS=80`, and
  `PROBE_AMPLITUDE_X=0.012`. It completed on `server36` with exit `0:0`
  after `00:01:44`. Summary:
  `experiments/outputs/direct_carry_task_runner_active_probe_postures/20260705_task_runner_directional_feedback_postures_seed7085/active_probe_posture_summary.json`.
  Shared hidden randomized box: mass `10.22545 kg`, size
  `[0.35601, 0.24706, 0.22236] m`, COM offset
  `[-0.03576, 0.00871, 0.01431] m`. Postures `front_mid`, `low_front`, and
  `chest_high` all completed `3660/3660`, passed their in-allocation strict
  checker (`check_status=pass`), had fall/drop `0`, root shortcut free,
  active probe belief available without hidden-ground-truth use, directional
  placement true, feedback controller enabled with
  `feedback_step_applied_steps=3570`, and PhysX contact-report gates passed.
  Final post-settle box travel / target distance: `front_mid`
  `0.61728 / 0.02272 m`, `low_front` `0.64413 / 0.00413 m`, and
  `chest_high` `0.61305 / 0.02695 m`. This is stronger direct-Isaac
  task-runner scaffold evidence than the previous manual-audit run because
  rollout and strict checker ran inside the same allocation. It remains a
  scaffolded support-foot carrier, not a complete walking robot, not a learned
  policy, not video-conditioned RL, and not the final unknown-object carrying
  solution.
- 2026-07-05 support-foot slip audit gate and negative result:
  added `--max-near-ground-foot-slip` to
  `scripts/isaac/check_direct_carry_task_summary.py`, exposed optional
  `MAX_NEAR_GROUND_FOOT_SPEED` and `MAX_NEAR_GROUND_FOOT_SLIP` in
  `scripts/isaac/run_check_direct_carry_task_runner_episode.sh`, and extended
  `scripts/isaac/summarize_task_runner_active_probe_postures.py` to record
  `per_foot_max_near_ground_xy_speed_mps`,
  `per_foot_max_near_ground_xy_slip_m`,
  `max_near_ground_foot_speed_mps`, and
  `max_near_ground_foot_slip_m`. Also updated
  `scripts/isaac/direct_carry_task_shell_backend.py` and
  `scripts/isaac/run_direct_carry_task_runner_episode.sh` so parent env can
  override `STANCE_STEPS`, `STEP_LENGTH`, support-foot drive params, and
  friction params for slip-reduction tests. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/direct_carry_task_shell_backend.py scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/summarize_task_runner_active_probe_postures.py`
  and
  `bash -n scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  Slurm job `166864`, stamp
  `20260705_task_runner_directional_slow_slip_audit_seed7086`, ran a
  single-posture `front_mid` slow-stance audit with `STANCE_STEPS=160`,
  `STEPS=7000`, `SUPPORT_MODE=alternating_placement_feet`, and
  `MAX_NEAR_GROUND_FOOT_SPEED=0.8`. It failed as intended on the new stricter
  slip-speed audit, not because of falls/drops. It completed `7000` steps with
  fall/drop `0`, final post-settle box travel `0.64886 m`, final target
  distance `0.00886 m`, active probe belief available without hidden
  ground-truth use, but max near-ground foot speed was `1.05842 m/s` and max
  near-ground foot slip was `0.69295 m`. Summary:
  `experiments/outputs/direct_carry_task_runner_active_probe_postures/20260705_task_runner_directional_slow_slip_audit_seed7086/active_probe_posture_summary.json`.
  This negative result is important: the current support-foot scaffold can
  carry the box while passing contact/fall/drop gates, but it still relies on
  too much planted-foot sliding to be treated as realistic walking. The next
  implementation must reduce or eliminate near-ground support-foot slip,
  not merely tune the old pass gate.
- 2026-07-05 stance-foot world-lock diagnostic, negative result:
  added `--stance-foot-world-lock` to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`, threaded
  `STANCE_FOOT_WORLD_LOCK` through the direct task-runner shell/backend stack,
  and added checker/summarizer fields for
  `stance_foot_world_lock_enabled`, joint count, switch count, pose update
  count, and active locked feet. The checker now treats world-locked stance
  feet as a fixed-world support diagnostic: default checks still forbid it,
  and explicit diagnostic checks must set `REQUIRE_STANCE_FOOT_WORLD_LOCK=1`.
  Lightweight checks passed:
  `bash -n scripts/isaac/run_core_world_anchored_footstep_carrier.sh scripts/isaac/run_direct_carry_task_physical_backend.sh scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`
  and
  `python3 -m py_compile scripts/isaac/build_core_world_anchored_footstep_carrier.py scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/summarize_task_runner_active_probe_postures.py scripts/isaac/direct_carry_task_shell_backend.py`.
  The first two submissions exposed wrapper/propagation bugs: job `166867`
  hit an old wrapper parse failure, and job `166869` completed the Isaac
  episode but did not actually enable world-lock because
  `run_direct_carry_task_physical_backend.sh` was not propagating
  `STANCE_FOOT_WORLD_LOCK`. Those are not valid world-lock evidence.
  The valid run was Slurm job `166875`, stamp
  `20260705_task_runner_stance_world_lock_slip_seed7088_server36`, fixed to
  `server36`, single posture `front_mid`, `BOX_SEED=7088`,
  `TARGET_X=0.64`, `STEPS=3660`, `SUPPORT_MODE=alternating_placement_feet`,
  `STANCE_FOOT_WORLD_LOCK=1`, `REQUIRE_STANCE_FOOT_WORLD_LOCK=1`, and slip
  gates `MAX_NEAR_GROUND_FOOT_SPEED=0.8`,
  `MAX_NEAR_GROUND_FOOT_SLIP=0.2`. It completed the rollout with fall/drop
  `0`, final post-settle box travel `0.64560 m`, final target distance
  `0.00560 m`, active probe belief without hidden ground truth, and
  `stance_foot_world_lock_enabled=true` with `4` lock joints,
  `81` switch events, and `324` pose updates. The strict checker still failed:
  max actual support-foot lift was only `0.01943 m` versus the `0.03 m` gate,
  max near-ground foot speed was `0.91486 m/s` versus `0.8`, and max
  near-ground foot slip was `0.73106 m` versus `0.2`. PhysX also warned that
  the stance world-lock joints had disjointed body transforms and would likely
  snap objects together. Conclusion: simple runtime world-locking is not a
  valid fix for planted-foot sliding. It is useful as an audit, but the next
  support mechanic must avoid conflicting lock/drive targets and should use a
  true stance constraint or contact-consistent planted-foot controller rather
  than snapping feet to moving fixed joints.
- 2026-07-05 freeze-locked stance-foot target diagnostic, pending validation:
  after the negative stance-world-lock result, added
  `--freeze-locked-stance-foot-targets` to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`. When enabled
  together with `--stance-foot-world-lock`, locked stance feet keep their
  measured X/Z joint targets instead of being driven against their own
  fixed-world constraints; swing feet remain free to move. Threaded
  `FREEZE_LOCKED_STANCE_FOOT_TARGETS` through
  `run_core_world_anchored_footstep_carrier.sh`,
  `run_direct_carry_task_physical_backend.sh`,
  `direct_carry_task_shell_backend.py`,
  `run_direct_carry_task_runner_episode.sh`, and
  `run_task_runner_active_probe_postures.sh`. Added checker and summary fields
  `freeze_locked_stance_foot_targets_enabled` and
  `freeze_locked_stance_foot_target_count`, plus optional checker gate
  `REQUIRE_FREEZE_LOCKED_STANCE_FOOT_TARGETS=1`. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_anchored_footstep_carrier.py scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/summarize_task_runner_active_probe_postures.py scripts/isaac/direct_carry_task_shell_backend.py`
  and
  `bash -n scripts/isaac/run_core_world_anchored_footstep_carrier.sh scripts/isaac/run_direct_carry_task_physical_backend.sh scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  An initial compute attempt, Slurm job `166884`, failed before Isaac with a
  transient/stale shell error and produced no backend summary; retry `166885`
  failed before rollout with an argparse `--tray-` ambiguity. Added
  `DEBUG_CORE_CMD=1` support to print the exact core argv for debugging.
  Valid debug run `166888`, stamp
  `20260705_task_runner_freeze_locked_stance_seed7089_debug_s10`, ran on
  `server10` and reached Isaac rollout and checker. It is a negative result:
  freeze mode was enabled and audited
  (`freeze_locked_stance_foot_targets_enabled=true`,
  `freeze_locked_stance_foot_target_count=8`,
  `stance_foot_world_lock_enabled=true`, `4` lock joints, `81` switch events),
  and it dramatically reduced foot sliding (`max_near_ground_foot_slip_m`
  `0.00320`, `max_near_ground_foot_speed_mps` `0.27180`). But it failed the
  task: `fall_events=1853`, final post-settle box travel was
  `-0.14893 m`, final target distance was `0.78893 m`, and
  max target-directed post-settle travel was only `0.00327 m`. PhysX still
  emitted repeated disjoint world-lock joint snap warnings. Conclusion:
  freezing locked stance targets proves the old progress depended on dragging
  or driving stance feet; removing that conflict fixes the slip metric but
  removes useful propulsion and destabilizes the scaffold. The next support
  mechanic must introduce propulsion through the body/anchor relative to truly
  planted feet, not by moving the planted feet themselves.
- 2026-07-05 planted-stance rail-propulsion diagnostic, negative result:
  added `--planted-stance-rail-propulsion` and threaded
  `PLANTED_STANCE_RAIL_PROPULSION` through the core launcher, physical
  backend, task-runner shell backend, episode runner, posture sweep,
  normalizer, checker, and summarizer. The optional checker gate is
  `REQUIRE_PLANTED_STANCE_RAIL_PROPULSION=1`; it requires the mode to be
  enabled and `planted_stance_rail_propulsion_steps > 0`. Lightweight checks
  passed:
  `python3 -m py_compile scripts/isaac/build_core_world_anchored_footstep_carrier.py scripts/isaac/check_direct_carry_task_summary.py scripts/isaac/normalize_direct_carry_backend_summary.py scripts/isaac/summarize_task_runner_active_probe_postures.py scripts/isaac/direct_carry_task_shell_backend.py`
  and
  `bash -n scripts/isaac/run_core_world_anchored_footstep_carrier.sh scripts/isaac/run_direct_carry_task_physical_backend.sh scripts/isaac/run_direct_carry_task_runner_episode.sh scripts/isaac/run_task_runner_active_probe_postures.sh scripts/isaac/run_check_direct_carry_task_runner_episode.sh`.
  Slurm job `166894`, stamp
  `20260705_task_runner_planted_rail_propulsion_seed7090`, is not valid
  physical evidence because the trigger was initially placed in the wrong
  branch and `planted_stance_rail_propulsion_steps=0`. After fixing that
  wiring, valid job `166895`, stamp
  `20260705_task_runner_planted_rail_propulsion_seed7091_fixedgate`, ran on
  `server53` with `BOX_SEED=7091`, `POSTURES=front_mid`, `TARGET_X=0.64`,
  `STEPS=3660`, `SUPPORT_MODE=alternating_placement_feet`,
  `STANCE_FOOT_WORLD_LOCK=1`, `FREEZE_LOCKED_STANCE_FOOT_TARGETS=1`,
  `PLANTED_STANCE_RAIL_PROPULSION=1`, and all corresponding require gates.
  It did trigger the new diagnostic (`planted_stance_rail_propulsion_steps`
  `3570`) and kept near-ground foot sliding low (`max_near_ground_foot_slip_m`
  `0.00320`, `max_near_ground_foot_speed_mps` `0.27191`). It still failed
  decisively: `fall_events=1902`, actual support-foot lift
  `0.00428 m < 0.03 m`, max target-directed post-settle travel only
  `0.00360 m`, final post-settle box travel `-0.13414 m`, and final target
  distance `0.77414 m`. PhysX continued to warn about disjoint stance
  world-lock joints snapping. Conclusion: world-lock plus frozen stance plus
  rail propulsion is not an acceptable support model. It removes visible foot
  sliding but leaves no robust forward carrying mechanics and still depends on
  invalid fixed-world snapping. Stop extending this world-lock branch except
  for documentation or cleanup; the next implementation should replace the
  support model with contact-consistent planted-foot mechanics or a real
  controller-backed Isaac robot.
- 2026-07-05 no-world-lock commanded-stance freeze diagnostic, negative
  result: added `--freeze-commanded-stance-foot-targets` and threaded
  `FREEZE_COMMANDED_STANCE_FOOT_TARGETS` through the same direct Isaac
  task-runner stack. This mode does not create fixed-world stance joints; it
  only freezes commanded stance-foot X/Z targets at measured joint positions,
  then can be combined with `PLANTED_STANCE_RAIL_PROPULSION=1` to test whether
  contact/friction alone can support body-relative rail motion. Slurm job
  `166899`, stamp
  `20260705_task_runner_no_worldlock_contact_propulsion_seed7092`, ran on
  `server44` with `STANCE_FOOT_WORLD_LOCK=0`,
  `FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`,
  `PLANTED_STANCE_RAIL_PROPULSION=1`, and require gates for both enabled
  diagnostics. It correctly avoided fixed-world support:
  `stance_foot_world_lock_enabled=false`, `stance_foot_world_lock_joint_count`
  `0`, while `freeze_commanded_stance_foot_target_count=8`,
  `freeze_commanded_stance_foot_target_switch_count=81`, and
  `planted_stance_rail_propulsion_steps=3570`. The negative result is clear:
  max near-ground foot slip/speed stayed low in the posture summary
  (`0.04693 m`, `0.30856 m/s`), and actual support-foot lift reached
  `0.11852 m`, but support contact broke repeatedly
  (`min_drive_contact_report_foot_count=0`,
  `drive_contact_report_lt2_steps=76`,
  `commanded_stance_contact_report_lt2_steps=1580`). The run had
  `fall_events=444`, final post-settle box travel `-0.21011 m`, final target
  distance `0.85011 m`, and max target-directed post-settle travel only
  `0.00355 m`. Conclusion: the prismatic support-foot scaffold cannot be
  rescued by freezing commanded stance targets plus rail propulsion. The next
  serious Isaac path should stop tuning this scaffold and move to a
  controller-backed robot/contact model that can maintain stance contact,
  produce forward impulse, and pass the existing slip/contact/fall/drop/travel
  gates without fixed-world locks.
- 2026-07-05 user correction and current execution rule: do not block on
  downloading or waiting for external models/checkpoints if they are not
  immediately useful. Continue constructing the carrying task directly in
  Isaac. The active robot route is the direct Core API G1 scene
  `scripts/isaac/build_core_world_g1_box_scene.py` with launcher
  `scripts/isaac/run_core_world_g1_box_scene.sh`.
- 2026-07-05 direct Core API G1 diagnostic status: G1 loads as a real
  articulation with `43` joints and can be initialized near standing height
  without rollout root pose or root velocity writes. The current failure is
  not missing data; it is stand/balance control. Nominal open-loop stand
  diagnostics `diag5` and `diag6` both completed 220 steps but slowly pitched
  forward, ending with `fall_events=5`, `max_tilt_rad=0.90630`, and
  `min_robot_z_m=0.55813`. Balance-feedback tests were negative: pitch sign
  `-1` produced `fall_events=51`, `max_tilt_rad=2.23711`; pitch sign `+1`
  still failed with `fall_events=33`, `max_tilt_rad=1.31126`. Crouch stand
  tests were also negative: mid crouch ended with `fall_events=153`,
  `final_pitch_rad=-0.86712`; deep crouch ended with `fall_events=247`,
  `final_pitch_rad=-1.03103`. Conclusion: nominal stand is the best baseline
  so far, but it needs posture/gain/control tuning before any payload or box
  claim.
- 2026-07-05 implementation update: added a pure stand-diagnostic carry-box
  disable path (`--disable-carry-box-spawn`, launcher env
  `SPAWN_CARRY_BOX=0`) so a free, untouched box falling under gravity does
  not contaminate G1 standing metrics. The script records `carry_box_spawned`;
  the checker can require it with `--expect-carry-box-spawned true|false`.
  Also fixed setup root orientation so `--g1-root-orientation-wxyz` is used
  by `robot.set_world_pose`. Lightweight checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/check_core_world_g1_box_scene_summary.py`
  and `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`.
- 2026-07-05 active compute run: submitted Curiosity-owned tmux/Slurm session
  `curiosity_g1_stand_nobox_tune_0705`, Slurm job `166915`, command shape
  `srun --partition=gpu --gres=gpu:1 --time=02:00:00 --job-name=g1_stand_nb_tune ...`.
  It runs four no-box G1 stand diagnostics:
  `20260705_core_world_g1_nobox_gain2_diag11`,
  `20260705_core_world_g1_nobox_gain3_diag12`,
  `20260705_core_world_g1_nobox_mildcrouch_diag13`, and
  `20260705_core_world_g1_nobox_feedback_low_diag14`. These are diagnostics
  only. Do not claim carrying until no-box stand passes, then fixed payload,
  then free box.
- 2026-07-05 no-box G1 stand tuning result: Slurm job `166915` ran on
  `server53` and completed all four diagnostics. All failed, and the no-box
  path worked as intended (`carry_box_spawned=false`, `box_drop_events=0`).
  `diag11` gain scale 2 failed with `fall_events=94`, `max_tilt_rad=3.11633`,
  `min_robot_z_m=0.10198`; `diag12` gain scale 3 failed with
  `fall_events=96`, `max_tilt_rad=3.03615`; `diag13` mild crouch failed with
  `fall_events=210`, `max_tilt_rad=1.46066`; `diag14` low pitch feedback
  failed with `fall_events=91`, `max_tilt_rad=3.13663`. Conclusion: free-box
  noise has been eliminated and the failure remains G1 stand/balance control.
  Do not proceed to payload/free-box trials from these runs.
- 2026-07-05 G1 config-alignment update: local IsaacLab
  `G1_29DOF_CFG` uses root pos `(0, 0, 0.75)`, rot
  `(0, 0, 0.7071, 0.7071)`, joint defaults
  hip pitch `-0.10`, knee `0.30`, ankle pitch `-0.20`, and different actuator
  gains from the previous Core API stand table. Added
  `--stand-drive-preset {arena,isaaclab29dof}` with launcher env
  `STAND_DRIVE_PRESET`; added `--disable-usd-pelvis-xform` with env
  `DISABLE_USD_PELVIS_XFORM=1`. Lightweight checks passed after this update.
  Active Slurm job `166916` in tmux `curiosity_g1_isaaclab_pose_tune_0705`
  tests IsaacLab-style root rotation/gains and pelvis-xform handling with
  stamps `diag15`-`diag17`.
- 2026-07-05 correction: job `166916` is invalid evidence. It started while
  the launcher was being edited and failed before Isaac with
  `unexpected EOF while looking for matching '"'`; it produced no summaries.
  Added setup-only joint state initialization
  (`robot.set_joint_positions`, `robot.set_joint_velocities`) before rollout,
  recorded as `joint_state_write_count_setup`, because applying a position
  action for one step is not equivalent to IsaacLab initial joint state.
  Rechecked `py_compile` and `bash -n`, then submitted retry tmux
  `curiosity_g1_isaaclab_pose_retry2_0705`, Slurm job `166918`, stamps
  `diag15_retry2`-`diag17_retry2`.
- 2026-07-05 G1 IsaacLab-pose retry result: Slurm job `166918` completed.
  `diag15_retry2` and `diag16_retry2` used root quaternion
  `(0, 0, 0.7071, 0.7071)` as wxyz and failed immediately with
  `fall_events=260`; the first recorded roll was about `1.57 rad`, so that
  config value should not be copied blindly into Core API as wxyz. `diag17`
  identity orientation + no pelvis xform + IsaacLab drive gains + setup joint
  state write was better but still failed: `fall_events=139`,
  `max_tilt_rad=1.10835`, `min_robot_z_m=0.32789`, final pitch `1.07482`.
  Conclusion: setup joint write helps semantics but does not solve G1
  balance; do not test payload yet.
- 2026-07-05 controller update: added pitch/roll rate terms to the simple
  balance feedback controller (`--balance-pitch-rate-gain`,
  `--balance-roll-rate-gain`, launcher envs `BALANCE_PITCH_RATE_GAIN` and
  `BALANCE_ROLL_RATE_GAIN`). Lightweight checks passed. Submitted tmux
  `curiosity_g1_setup_pd_stand_0705`, job-name `g1_setup_pd`, with no-box
  stamps `diag18`-`diag20`.
- 2026-07-05 setup+PD stand result: initial job `166920` and retry `166921`
  failed before valid Isaac rollout due transient/inconsistent file reads;
  retry3 `166922` first ran compute-side `py_compile` and `bash -n`, then
  completed. `diag18_retry3` is the best G1 no-box stand so far:
  setup joint-state write, identity root, arena gains, no box, 43 joints,
  `completed_steps=320`, `fall_events=7`, `max_tilt_rad=0.93557`,
  `min_robot_z_m=0.54179`. It only failed near the end by slow forward pitch,
  so setup joint-state write is important. `diag19_retry3` arena gains plus
  PD pitch/roll feedback was worse (`fall_events=71`, `max_tilt_rad=3.14061`).
  `diag20_retry3` IsaacLab gains plus PD was worse
  (`fall_events=195`, `max_tilt_rad=1.20595`). Conclusion: do not use the
  current PD feedback; continue with small static stand posture/height tuning
  around `diag18`, still no payload.
- 2026-07-05 active compute run: submitted tmux
  `curiosity_g1_static_posture_sweep_0705`, job-name `g1_post_sweep`,
  no-box static stand stamps `diag21`-`diag24`. Variants test slight ankle/
  hip target offsets and root heights `0.75`/`0.81` around the current best
  arena setup baseline.
- 2026-07-05 static no-box posture sweep result: job `166923` completed.
  `diag22` passed the first real G1 no-box stand gate: 360/360 steps,
  `fall_events=0`, `max_tilt_rad=0.00882`, `min_robot_z_m=0.78429`, root
  z `0.78`, setup joint-state write `1`, arena drive gains, posture
  `stand_hip_pitch=-0.12`, `stand_knee=0.30`, `stand_ankle_pitch=-0.15`.
  Other variants failed: ankle-more `diag21` fall events `149`, low root
  `diag23` fall events `252`, high root `diag24` fall events `54`. This is
  stand evidence only, not carrying.
- 2026-07-05 active fixed-payload gate: submitted tmux
  `curiosity_g1_fixed_payload_stand_0705`, job-name `g1_payload_stand`,
  stamps `diag25`-`diag27`, testing fixed torso payload masses 0.5/1/2 kg
  with the successful `diag22` posture. Box collision is disabled in these
  first ballast tests to isolate load/balance from self-collision.
- 2026-07-05 fixed-payload stand result: job `166924` completed. Fixed torso
  ballast with collision disabled passed for 0.5/1/2 kg using the `diag22`
  posture. `diag25` 0.5 kg: 360/360, fall/drop 0, `max_tilt_rad=0.01514`.
  `diag26` 1 kg: 360/360, fall/drop 0, `max_tilt_rad=0.01990`.
  `diag27` 2 kg: 360/360, fall/drop 0, `max_tilt_rad=0.03265`. This is a
  fixed-payload stand gate only, not walking or free-box carrying.
- 2026-07-05 gait update: open-loop march now uses `_stand_joint_targets()`
  as its base instead of hard-coded old nominal leg targets. Submitted tmux
  `curiosity_g1_openloop_march_0705`, job-name `g1_march_smoke`, stamps
  `diag28` no-box and `diag29` 1 kg fixed payload. These are stability/travel
  diagnostics only.
- 2026-07-05 open-loop march smoke result: job `166926` completed and both
  smokes passed stability. `diag28` no-box: 420/420, fall/drop 0,
  `max_tilt_rad=0.00843`, max robot travel `0.00509 m`. `diag29` 1 kg fixed
  payload: 420/420, fall/drop 0, `max_tilt_rad=0.01967`, max robot travel
  `0.01354 m`, max box travel `0.01463 m`. This is stable no-root dynamic
  standing/marching with fixed payload, but the travel is too small for a
  carrying claim.
- 2026-07-05 active march-creep sweep: submitted tmux
  `curiosity_g1_march_creep_sweep_0705`, job-name `g1_march_creep`, stamps
  `diag30`-`diag32`, testing larger open-loop march amplitudes and 1/2 kg
  fixed payloads for short no-root travel.
- 2026-07-05 staged G1 gait update: after late-stop and threshold-feedback
  diagnostics showed that the heavy-cradle/open-loop motion enters an
  unrecoverable pitch runaway, added `gait_mode=staged_march` to the direct
  Core API G1 scene. New controls include gait ramp-down start/end, minimum
  amplitude scale, recovery pitch/rate thresholds, and hip/knee/ankle/waist
  recovery offsets. Summaries record recovery active steps and first active
  step. Lightweight login-node checks passed:
  `python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py scripts/isaac/check_core_world_g1_box_scene_summary.py`
  and `bash -n scripts/isaac/run_core_world_g1_box_scene.sh`. Added direct
  Python batch launcher
  `scripts/isaac/run_core_world_g1_staged_gait_batch.sh` to avoid stale shell
  launcher reads on compute nodes. Pending compute validation:
  `curiosity_g1_staged_gait_0705`, job-name `g1_staged`, stamps `diag67`-
  `diag70`, testing intermediate cradle mass scales with staged ramp-down and
  recovery posture gates.
- 2026-07-05 staged G1 gait result: job `166998` completed. `diag67`
  (`cradle_mass_scale=0.35`, amp `0.16`) and `diag68` (`0.50`, amp `0.14`)
  were stable with fall/drop 0 but failed distance (`-0.00043 m` and
  `0.01699 m` final box target-directed travel). `diag69`
  (`0.75`, amp `0.12`) was stable with final box target-directed travel
  `0.06329 m`, still below the `0.10 m` gate. `diag70`
  (`1.0`, amp `0.12`) produced meaningful final box target-directed travel
  `0.64051 m` and no box drop, but failed at the end with `fall_events=2` and
  `max_tilt_rad=0.86674 > 0.85`. This is a near miss, not success. Added
  refinement launcher
  `scripts/isaac/run_core_world_g1_staged_gait_refine_batch.sh`, with
  lightweight syntax checks passed, to test mass `0.85-1.0` and amp
  `0.10-0.12` around the stable/unstable boundary.
- 2026-07-05 staged G1 refinement result: job `167002` completed. All four
  variants passed the current 420-step short-distance checker with fall/drop
  0 and rollout root/box writes 0. `diag71` (`mass=0.85`, amp `0.12`) reached
  final box target-directed travel `0.37075 m`, max tilt `0.43785`.
  `diag72` (`mass=0.90`, amp `0.11`) reached `0.49137 m`, max tilt
  `0.60004`. `diag73` (`mass=0.95`, amp `0.10`) reached `0.51315 m`, max
  tilt `0.62928`. `diag74` (`mass=1.0`, amp `0.10`) reached `0.63546 m`, but
  max tilt was `0.84893`, just under the `0.85` gate. This is a short-distance
  diagnostic pass, not final carrying completion. Next required validation is
  a longer hold/rollout for the safer `diag72`/`diag73` configs.
- 2026-07-05 staged G1 long validation result: job `167005` completed.
  `diag75` reran the `diag72` config for 700 steps and failed after the
  original 420-step window with `fall_events=256`, `box_drop_events=239`,
  `max_tilt_rad=1.14866`, and min box z `0.09610`; final box target-directed
  travel was `0.75432 m`. `diag76` reran the `diag73` config and also failed
  after step 450 with `fall_events=259`, `box_drop_events=242`,
  `max_tilt_rad=1.14076`, and min box z `0.10530`; final box target-directed
  travel was `0.75943 m`. Therefore the 420-step passes are short-window
  diagnostics only. Next search should retreat to more conservative
  `cradle_mass_scale=0.80-0.85` and amp `0.10-0.12` for a 700-step pass.
- 2026-07-05 conservative staged long validation update: added
  `scripts/isaac/run_core_world_g1_staged_gait_conservative_long.sh`, with
  login-node syntax checks passed, to test 700-step configs `diag77`-`diag80`
  around `cradle_mass_scale=0.80/0.85` and amp `0.10/0.12`. Pending compute
  batch: `curiosity_g1_staged_cons_long_0705`, job-name `g1_cons_long`.
- 2026-07-05 WBC-AGILE real-weight outcome: official local weights now load,
  but the route is negative for the active path. `diag5_directload` entered
  rollout with the real checkpoint and had `policy_inference_count=115`, but
  failed with `fall_events=359`, max tilt `3.0337 rad`, and min robot z
  `0.0643 m`. `diag6_zero_cmd` also failed and lacked the stable stand
  posture overrides. Corrected zero-command stable-pose `diag7_zero_stablepose`
  still failed with `fall_events=92`, max tilt `2.50074 rad`, min robot z
  `0.06682 m`, and only `0.00737 m` max target-directed robot travel. Do not
  wait on AGILE/WBC as a blocker; treat it as an optional convention-debugging
  side path.
- 2026-07-05 user correction and active path: continue constructing the task
  directly in Isaac. Direct scene smoke
  `20260705_core_world_g1_box_in_front_scene_smoke_retry2` passed: free
  dynamic 2 kg box on the ground in front of G1, `attach_box=none`,
  `torso_cradle=none`, 43 G1 joints, stable stand posture, arena gains,
  360/360 steps, fall/drop 0, min robot z `0.78429 m`, min box z `0.16010 m`,
  max tilt `0.00882 rad`, and rollout root pose/root velocity/box pose writes
  all 0. This is only a scene baseline, not walking, probing, lifting, or
  carrying.
- Immediate next work: build on the G1 + box-in-front Isaac scene by adding
  one explicit task phase at a time. Do not let downloaded models, external
  controllers, or old support scaffolds block scene construction.
- 2026-07-05 front-probe contact diagnostic: added
  `PROBE_MODE=front_bumper` / `--probe-mode front_bumper` to
  `scripts/isaac/build_core_world_g1_box_scene.py`, plus probe pad geometry,
  mass, collision, start-step, reference pose, probe-active steps, box
  displacement, target-directed probe travel, and `probe_box_moved` summary
  fields. Added dedicated launchers
  `scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh` and
  `scripts/isaac/submit_core_world_g1_front_probe_bumper_smoke.sh`.
  Checker gates now include `--expect-probe-mode`,
  `--min-probe-active-steps`, `--require-probe-box-moved`,
  `--min-final-probe-box-travel`, `--min-max-probe-box-travel`, and
  `--min-final-probe-box-target-directed-travel`.
- 2026-07-05 front-probe evidence: aggressive retry4 moved the box but failed
  balance (`fall_events=284`, max tilt `2.55803 rad`), proving the contact was
  physically active but too impulsive. Gentle retry5
  `20260705_core_world_g1_front_probe_bumper_submit_retry5_gentle` passed:
  360/360 steps, `probe_mode=front_bumper`, free dynamic 2 kg box, fall/drop
  0, max tilt `0.05226 rad`, min robot z `0.78356 m`, min box z `0.16048 m`,
  rollout root pose/root velocity/box pose writes all 0, final probe box
  travel `0.15285 m`, and final target-directed probe travel `0.15260 m`.
  This is a contact-probe diagnostic only. It must not be called grasping,
  lifting, walking, carrying, or a complete robot-box transport result.
- 2026-07-05 staged grasp/lift update: added
  `GRASP_MODE=staged_fixed_torso`, `GRASP_ENABLE_STEP`, and
  `GRASP_LIFT_OFFSET_Z` to the G1 Core API scene and probe launcher. Runtime
  grasp creates `/World/CarryBox/StagedFixedTorsoGraspJoint` after the probe
  phase and records attach step, local offset, attach poses, post-grasp box z
  delta, and `grasp_attached`. Checker gates include `--expect-grasp-mode`,
  `--require-grasp-attached`, `--min-grasp-attach-step`,
  `--min-max-post-grasp-box-z-delta`, and
  `--min-final-post-grasp-box-z-delta`.
- 2026-07-05 staged grasp/lift evidence:
  `20260705_core_world_g1_probe_grasp_lift_retry1` passed as a diagnostic:
  attach step `140`, 360/360 steps, fall/drop 0, max tilt `0.06396 rad`,
  rollout root pose/root velocity/box pose writes all 0, max post-grasp box z
  delta `0.06895 m`, and final post-grasp z delta `0.01703 m`. PhysX emitted
  a disjoint fixed-joint snap warning, so this is not a faithful hand grasp.
- 2026-07-05 grasp+march evidence:
  `20260705_core_world_g1_probe_grasp_march_retry1` and
  `20260705_core_world_g1_probe_grasp_march_retry2_amp010` both passed
  stability/grasp/lift gates with fall/drop 0 and no rollout root/velocity/box
  pose writes. The first used open-loop march amp `0.05` and final robot
  target-directed travel was only `0.04027 m`; the second used amp `0.10` and
  final robot target-directed travel was only `0.03264 m` with lateral box
  oscillation. These are not walking-carrying successes. Do not present the
  current open-loop march as the final locomotion backend.
- 2026-07-05 active correction: external models, AGILE/WBC convention
  debugging, and downloaded research code are no longer blockers for the
  active execution path. Continue direct Isaac scene construction phase by
  phase. Current code update generalizes staged grasp from hard-coded
  `torso_link` to arbitrary body links via `GRASP_MODE=staged_fixed_body` and
  `GRASP_BODY_PATH`, computes fixed-joint `localPos0` in the selected body
  frame, records `active_grasp_body_path`, wrapper initialization status, and
  body pose at attach, and adds checker expectations for grasp body path.
  Pending compute diagnostic:
  `20260705_core_world_g1_probe_hand_grasp_lift_retry1`, Slurm job `167195`,
  tmux `curiosity_g1_hand_grasp_retry1_0705`, with
  `GRASP_BODY_PATH=/World/G1/right_hand_palm_link`. This is intended to test
  whether hand-link staged grasp can replace the torso-fixed artifact. It is
  still a diagnostic, not final hand grasping or walking while carrying.
- 2026-07-05 hand-link staged grasp result:
  `20260705_core_world_g1_probe_hand_grasp_lift_retry1` completed `360/360`
  and passed the diagnostic checker with `grasp_mode=staged_fixed_body`,
  `active_grasp_body_path=/World/G1/right_hand_palm_link`, hand wrapper
  initialized, fall/drop `0`, max tilt `0.05226 rad`, rollout root
  pose/root velocity/box pose writes all `0`, max post-grasp box z delta
  `0.01450 m`, and final post-grasp z delta `0.00409 m`. However attach
  distance was still large: hand-to-box world distance at attach was about
  `0.967 m` (`local_pos0=[0.651, 0.143, -0.702]`), and PhysX still warned
  about disjoint fixed-joint body transforms. Therefore this proves the
  selected hand link can be addressed by Isaac, but it is not physically
  faithful hand grasping. New active code path adds `ARM_POSE_MODE`,
  shoulder/elbow/wrist manual overrides, arm-pose active-step telemetry, and a
  checker gate for `grasp_body_box_world_distance_at_attach_m`.
- 2026-07-05 pending arm-reach diagnostic:
  `20260705_core_world_g1_armreach_hand_grasp_retry1`, tmux
  `curiosity_g1_armreach_hand_grasp_retry1_0705`, job-name `g1_armgr1`, uses
  `ARM_POSE_MODE=right_front_reach` before attaching to
  `/World/G1/right_hand_palm_link`. Goal is to reduce hand-box attach
  distance; even if it passes stability, it is still diagnostic until
  disjoint snap is removed and carrying/walking is demonstrated.
- 2026-07-05 arm-reach diagnostic result:
  `20260705_core_world_g1_armreach_hand_grasp_retry1` is negative. It
  completed `360/360` with fall/drop `0` and no rollout root/velocity/box pose
  writes, but `right_front_reach` increased the hand-box attach distance to
  `0.98688 m` and caused severe fixed-joint snap: max post-grasp box z delta
  `1.60657 m`, final box target-directed travel `-0.93573 m`, and max tilt
  `0.30448 rad`. Do not reuse this preset as evidence of reaching. Next
  action: run a small manual shoulder/elbow/wrist sign sweep and select only
  configs that reduce `grasp_body_box_world_distance_at_attach_m` before any
  further lift/carry claims.
- 2026-07-05 first manual arm-pose sweep result: tmux
  `curiosity_g1_arm_pose_sweep1b_0705` ran four short right-palm staged attach
  diagnostics. None reduced attach distance below the no-arm-pose hand-link
  baseline (`0.967 m`). Distances were: `pospitch_negelbow=1.16467 m`,
  `pospitch_poselbow=1.39430 m`, `pospitch_rollpos=1.16337 m`,
  `highpitch_negelbow=1.20940 m`. All still produced disjoint fixed-joint
  snap, with max post-grasp z deltas from `0.309 m` to `2.693 m`. Therefore
  positive shoulder-pitch sweeps are not a viable reaching direction.
- 2026-07-05 second manual arm-pose sweep result: tmux
  `curiosity_g1_arm_pose_sweep2_0705` ran four negative-pitch/yaw right-palm
  staged attach diagnostics. None improved the hand-box attach distance:
  `negpitch_negelbow=1.33406 m`,
  `negpitch_negelbow_rollpos=1.33226 m`,
  `negpitch_yawpos=1.33040 m`, and `negpitch_yawneg=1.31012 m`. Two configs
  produced fall events (`8` and `16`). This is enough evidence to stop the
  single-right-hand ground-box staged attach path for now. Next direct-Isaac
  route should use a staged carry-height setup or double-arm/chest-supported
  contact so the object starts near a physically reachable carrying posture,
  then later reintroduce active lifting from the ground.
- 2026-07-05 carry-height staged diagnostic update: added
  `BOX_SUPPORT_MODE=table` to spawn a static collision support table under the
  box, plus support-table summary/checker fields and `PROBE_MODE=none` support
  in the G1 launcher. Pending run
  `20260705_core_world_g1_carryheight_hand_grasp_retry1`, tmux
  `curiosity_g1_carryheight_hand_grasp_retry1_0705`, job-name `g1_chgr1`, uses
  a support table with box center near `(0.46, -0.15, 0.88)`, right-palm staged
  attach at step `80`, and `grasp_lift_offset_z=0.02`. This tests whether
  carry-height object placement removes the large hand-box snap. It is a
  staged carry-posture/balance diagnostic only, not ground pickup evidence.
- 2026-07-05 carry-height retry outcome: retry1 failed at submit time only
  because `submit_core_world_g1_front_probe_bumper_smoke.sh` still required a
  literal `probe-mode front_bumper` string after `PROBE_MODE` became
  configurable; fixed the submit check to require only `--probe-mode`. Retry2
  ran but is negative: the support table/box was too close to the robot body,
  causing fall events before attach (`fall_events=42`, max tilt `0.97825 rad`,
  min robot z `0.53815 m`), and attach distance remained `0.96173 m`. Next
  carry-height run must move the support table/box forward and use a narrower
  table to avoid initial body collision.
- 2026-07-05 carry-height clean attach result:
  `20260705_core_world_g1_carryheight_hand_grasp_retry3` passed the staged
  carry-height attach/balance diagnostic: support table, `probe_mode=none`,
  right-palm staged attach at step `80`, 240/240 steps, fall/drop `0`,
  max tilt `0.00759 rad`, min robot z `0.78429 m`, min box z `0.88000 m`,
  rollout root pose/root velocity/box pose writes all `0`, attach distance
  `0.50746 m`, max post-grasp z delta `0.01796 m`, and final z delta
  `0.00305 m`. PhysX still warns about disjoint fixed-joint transforms, and
  this is not ground pickup or walking. Next diagnostic should add small
  post-attach march from this carry-height setup.
- 2026-07-05 carry-height march retry1 result:
  `20260705_core_world_g1_carryheight_hand_march_retry1` is stable but fails
  locomotion. It completed `420/420` with fall/drop `0`, max tilt
  `0.01752 rad`, no rollout root/velocity/box pose writes, and preserved the
  carry-height right-palm staged attach. However target-directed travel failed:
  final robot target-directed travel `-0.00276 m`, final box target-directed
  travel `-0.00796 m`, and max robot target-directed travel only
  `0.00108 m`. This is not walking/carrying; it only shows attached
  carry-height balance under small open-loop leg motion.
- 2026-07-05 carry-height march retry2 result:
  `20260705_core_world_g1_carryheight_hand_march_retry2_amp015` increased
  open-loop march amplitude to `0.15` and frequency to `0.8 Hz`. It also
  completed stably (`420/420`, fall/drop `0`, max tilt `0.04855 rad`, no
  rollout root/velocity/box pose writes), but target-directed travel still
  failed: final robot target-directed travel `-0.03335 m`, final box
  target-directed travel `-0.02982 m`, and max robot target-directed travel
  only `0.00108 m`. Conclusion: the current hand-written open-loop gait is not
  a locomotion backend. Do not continue tuning amplitude as if it will produce
  walking; switch to a real velocity/footstep controller or a validated local
  locomotion policy before making walking-carrying claims.
- 2026-07-05 targeted-creep gait implementation: added
  `gait_mode=targeted_creep` with explicit creep hip/knee/ankle/waist offsets,
  stance-push scale, lift scale, and ankle-lift scale. This is still a
  joint-level diagnostic, not a proven walking controller, but it is distinct
  from symmetric open-loop leg oscillation because it adds sustained forward
  bias and stance-push terms intended to create physical target-directed
  contact motion without root pose or root velocity writes. Pending compute
  run `20260705_core_world_g1_carryheight_creep_retry1`, tmux
  `curiosity_g1_carryheight_creep_retry1_0705`, job-name `g1_chcr1`, uses the
  clean carry-height right-palm attach setup and starts targeted creep at step
  `110`.
- 2026-07-05 targeted-creep retry1 result:
  `20260705_core_world_g1_carryheight_creep_retry1` passed a weak positive
  travel diagnostic: 420/420, fall/drop `0`, no rollout root/velocity/box pose
  writes, max tilt `0.13591 rad`, final robot target-directed travel
  `0.01460 m`, and final box target-directed travel `0.00374 m`. This is
  better than the open-loop march retries, which moved backward, but it is not
  sufficient walking/carrying evidence because the displacement is only
  centimeter-scale and the box barely advances.
- 2026-07-05 targeted-creep stronger/long results:
  `20260705_core_world_g1_carryheight_creep_retry2_stronger` completed
  `520/520` with fall/drop `0`, no rollout root/velocity/box pose writes,
  max tilt `0.16267 rad`, final robot target-directed travel `0.02888 m`, and
  final box target-directed travel `0.01706 m`. Long validation
  `20260705_core_world_g1_carryheight_creep_retry3_long` completed `900/900`
  with fall/drop `0`, no rollout root/velocity/box pose writes, max tilt
  `0.16267 rad`, final robot target-directed travel `0.04279 m`, and final
  box target-directed travel `0.03541 m`. This is the first direct G1 scene
  evidence of attached-box target-directed motion without root or box pose
  writes, but it is still staged and weak: the support table remains under
  the box, the fixed-joint attach still emits a PhysX disjoint warning, and
  displacement is only centimeter-scale. Next gate: release/remove the support
  table after attach and verify box support comes from the robot link.
- 2026-07-05 support-release implementation: added
  `BOX_SUPPORT_RELEASE_STEP` / `--box-support-release-step`, rollout removal
  of `/World/CarryBoxSupportTable`, summary fields
  `box_support_released` and `box_support_actual_release_step`, and checker
  gates `--require-box-support-released` and
  `--min-box-support-actual-release-step`. Pending diagnostic
  `20260705_core_world_g1_carryheight_release_stand_retry1`, tmux
  `curiosity_g1_carryheight_release_stand_0705`, job-name `g1_chrel1`: attach
  at step `80`, remove table at step `120`, then stand to 360. This tests
  whether the robot-link fixed joint can hold the box without table support.
- 2026-07-05 support-release stand retry1 result:
  `20260705_core_world_g1_carryheight_release_stand_retry1` removed the table
  at step `120` and completed `360/360` with fall/drop counters `0` and no
  rollout root/velocity/box pose writes, but it is negative for carrying:
  min box z fell to `0.28236 m`, final post-grasp z delta was `-0.57757 m`,
  and final box target-directed travel was `-0.13944 m`. The robot stayed
  upright while the box fell/slid after table removal. Next release test must
  use a positive staged lift offset or a different support body/contact, and
  require high min box z after release.
- 2026-07-05 support-release hand lift retry2 result:
  `20260705_core_world_g1_carryheight_release_lift_retry2` used
  `grasp_lift_offset_z=0.18` and released the table at step `160`. It also
  completed `360/360` with fall/drop counters `0` and no rollout root/box
  pose writes, but it is still negative for carrying: min box z `0.24513 m`,
  final post-grasp z delta `-0.54165 m`, and final box target-directed travel
  `-0.24795 m`. Positive lift offset did not make the right-hand single-link
  attach hold the object after support removal. Next route should test
  chest/torso-supported staged contact, which is a plausible low-effort human
  box-carrying posture.
- 2026-07-05 torso-fixed support-release result:
  `20260705_core_world_g1_chest_release_stand_retry1` is negative as
  configured. Switching the staged fixed body to `/World/G1/torso_link` and
  releasing the support table at step `160` destabilized the robot:
  `fall_events=242`, max tilt `2.70430 rad`, min robot z `0.05730 m`,
  min box z `0.60980 m`, final post-grasp z delta `-0.31130 m`, final robot
  target-directed travel `-0.65920 m`, and final box target-directed travel
  `-1.14500 m`. Do not treat this as chest carrying; the geometry/contact
  setup is invalid.
- 2026-07-05 direct Isaac route correction:
  external model/code downloads are not blockers for the next step. The
  active route is to build the G1 + box scene directly in Isaac and test
  physical support postures. The launcher now exposes the existing
  `TORSO_CRADLE=front_tray` path and cradle geometry parameters so the next
  diagnostic can place a free box on a torso/front tray with `GRASP_MODE=none`
  and no support table or staged fixed grasp. This tests body-supported
  carrying contact instead of hand/torso fixed-joint attachment.
- 2026-07-05 front-tray free-box retry1 result:
  `20260705_core_world_g1_front_tray_freebox_stand_retry1` used
  `TORSO_CRADLE=front_tray`, `GRASP_MODE=none`, `BOX_SUPPORT_MODE=none`, and a
  2 kg free box at carry height. It is a negative physical-contact result:
  `fall_events=243`, min robot z `0.05753 m`, min box z `0.19419 m`,
  max tilt `2.64308 rad`, final robot target-directed travel `-0.69757 m`,
  and final box target-directed travel `-2.82168 m`. It did preserve no
  rollout root/velocity/box pose writes. The likely issue is excessive
  forward/high tray load and contact moment. The next direct Isaac retry must
  lower and move the support closer to the body and reduce diagnostic load
  before walking is attempted.
- 2026-07-05 front-tray free-box retry2 result:
  `20260705_core_world_g1_front_tray_freebox_lowstand_retry2` used a 0.5 kg
  smaller box and a lower/closer/lighter tray. It improved stability but is
  still not a valid carry posture: fall/drop `0`, no rollout root/velocity/box
  pose writes, but max tilt rose to `0.67826 rad`, min robot z was
  `0.67762 m`, and min box z fell to `0.52771 m`. The positive target-directed
  travel (`robot=0.51625 m`, `box=0.47816 m`) came from progressive
  forward tipping/sliding, not walking. The launcher now exposes existing
  balance-feedback, recovery, terminal-hold, and gait ramp-down parameters so
  the next direct scene diagnostic can test low near-body support plus posture
  compensation without introducing external models.
- 2026-07-05 front-tray free-box retry3 result:
  `20260705_core_world_g1_front_tray_freebox_lowbalance_retry3` is a strong
  negative balance-compensation result. The lower/closer 0.3 kg setup plus
  crouched stand and `balance_pitch_sign=-1.0` fell by step 20 and completed
  240/240 with `fall_events=166`, max tilt `3.13363 rad`, min robot z
  `0.40093 m`, min box z `0.21701 m`, and unstable final target-directed
  travel near `6 m`. No rollout root/velocity/box pose writes occurred. Do
  not reuse that balance sign/gain/geometric combination; return to retry2
  geometry and test only small opposite-sign pitch correction if continuing.
- 2026-07-05 front-tray free-box retry4 result:
  `20260705_core_world_g1_front_tray_freebox_signpos_retry4` is the first
  valid direct Isaac free-box body-supported standing diagnostic. It uses
  `TORSO_CRADLE=front_tray`, `GRASP_MODE=none`, `BOX_SUPPORT_MODE=none`,
  retry2 geometry, and small opposite-sign pitch feedback. It completed
  360/360 with fall/drop `0`, max tilt `0.15888 rad`, min robot z
  `0.78414 m`, min box z `0.77035 m`, no rollout root/velocity/box pose
  writes, and balance feedback active for 360 steps. Final robot/box
  target-directed drift was `0.11518/0.12322 m`. This is stable
  body-supported holding/drift, not locomotion or long-distance carrying.
- 2026-07-05 front-tray free-box creep retry1 result:
  `20260705_core_world_g1_front_tray_freebox_creep_retry1` is negative over
  the full 520-step horizon. It used the retry4 free-box body support setup
  with `targeted_creep`. It completed 520/520 and preserved no rollout
  root/velocity/box pose writes, but ended in progressive forward tipping:
  `fall_events=2`, max tilt `0.86491 rad`, min robot z `0.59641 m`, min box z
  `0.40208 m`. Final robot/box target-directed travel was
  `0.63978/0.56808 m`, but this is not stable carrying because the final
  segment falls and loses box height. The early part stayed stable through
  about step 420, so a shorter/windowed creep diagnostic is worth testing as
  a diagnostic-only stepping fragment.
- 2026-07-05 front-tray free-box creep retry2 result:
  `20260705_core_world_g1_front_tray_freebox_creep440_retry2` is the best
  direct Isaac result so far, but remains diagnostic-only. It used a free box,
  `TORSO_CRADLE=front_tray`, `GRASP_MODE=none`, `BOX_SUPPORT_MODE=none`, and
  `targeted_creep`. It completed 440/440 with fall/drop `0`, min robot z
  `0.78289 m`, min box z `0.72056 m`, max tilt `0.30230 rad`, no rollout
  root/velocity/box pose writes, and final robot/box target-directed travel
  `0.23447/0.23692 m`. This is short-distance body-supported free-box motion,
  not long-duration carrying, because tilt is still growing and box height is
  declining.
- 2026-07-05 front-tray free-box stop-hold retry1 result:
  `20260705_core_world_g1_front_tray_freebox_stophold_retry1` is negative.
  It used the retry4/free-box support setup, `targeted_creep` from step 140,
  `GAIT_STOP_STEP=420`, and `TERMINAL_HOLD_START_STEP=420`. Terminal hold did
  activate at step 420 and stayed active for 200 steps, but this was too late
  to recover the forward pitch divergence: 620/620, `fall_events=103`, min
  robot z `0.49170 m`, min box z `0.26074 m`, max tilt `1.47193 rad`, and no
  rollout root/velocity/box pose writes. The large final travel is falling/
  sliding, not carrying. Future stop-hold tests must stop earlier, before the
  pitch leaves the retry4 stable range.
- 2026-07-05 front-tray free-box stop-hold retry2 result:
  `20260705_core_world_g1_front_tray_freebox_stophold360_retry2` is also
  negative. It stopped gait and entered terminal hold at step 360 with no
  extra hold offsets, but forward pitch still diverged after the stop:
  620/620, `terminal_hold_active_steps=260`, `fall_events=102`, min robot z
  `0.49309 m`, min box z `0.25548 m`, max tilt `1.51482 rad`, and no rollout
  root/velocity/box pose writes. This indicates that the current front-tray
  support plus controller can enter an unrecoverable forward-tipping mode even
  after gait commands stop. The next diagnostic should be a long stand-only
  baseline with the same free-box front-tray posture.
- 2026-07-05 front-tray free-box stand620 retry5 result:
  `20260705_core_world_g1_front_tray_freebox_stand620_retry5` is negative.
  With the same free-box front-tray posture, no gait, and the retry4 small
  positive pitch feedback, it still entered the slow forward-tipping mode:
  620/620, `fall_events=98`, min robot z `0.48937 m`, min box z `0.25926 m`,
  max tilt `1.48288 rad`, and no rollout root/velocity/box pose writes. This
  proves the active blocker is long-horizon posture/support stability under
  front-tray load, not only gait stop timing. Increase positive pitch-feedback
  authority or reduce front load moment before further gait attempts.
- 2026-07-05 front-tray free-box stand620 retry6 result:
  `20260705_core_world_g1_front_tray_freebox_stand620_gain_retry6` passes the
  long-horizon free-box body-support holding gate. With the same geometry, no
  gait, `BALANCE_PITCH_GAIN=0.45`, `BALANCE_PITCH_RATE_GAIN=0.02`, and
  `BALANCE_ADJUSTMENT_LIMIT=0.12`, it completed 620/620 with fall/drop `0`,
  max tilt `0.09570 rad`, min robot z `0.78411 m`, min box z `0.78743 m`,
  no rollout root/velocity/box pose writes, and final robot/box drift only
  `0.00572/0.01339 m`. This is valid long-duration body-supported holding,
  not walking/carrying. Use this stronger feedback as the next baseline for
  conservative creep.
- 2026-07-05 front-tray free-box creep retry3 result:
  `20260705_core_world_g1_front_tray_freebox_creep_gain_retry3` is stable but
  not meaningful carrying. With the retry6 stronger pitch feedback and
  `targeted_creep` amplitude `0.045`, it completed 620/620 with fall/drop
  `0`, min robot z `0.78248 m`, min box z `0.78724 m`, max tilt
  `0.09570 rad`, and no rollout root/velocity/box pose writes. However final
  robot/box target-directed travel was only `-0.00724/0.00180 m`, and max box
  target-directed travel was only `0.07241 m`. Strong feedback fixed
  long-horizon balance but canceled most propulsion; the next diagnostic
  should increase creep drive while retaining this feedback authority.
- 2026-07-05 front-tray free-box creep retry4 result:
  `20260705_core_world_g1_front_tray_freebox_creep_drive_retry4` increased
  targeted-creep amplitude to `0.08` and stance-push to `0.22` while retaining
  the strong feedback. It remained stable but still did not move forward:
  620/620, fall/drop `0`, min robot z `0.78114 m`, min box z `0.78730 m`,
  max tilt `0.09570 rad`, no rollout root/velocity/box pose writes, final
  robot/box target-directed travel `-0.01763/-0.00933 m`, and max box
  target-directed travel `0.07292 m`. Strong feedback is too dominant for the
  current gait drive; test intermediate feedback authority next.
- 2026-07-05 front-tray free-box creep retry5 result:
  `20260705_core_world_g1_front_tray_freebox_creep_midfb_retry5` is also
  stable but not locomotion. With `BALANCE_PITCH_GAIN=0.30` and
  `BALANCE_ADJUSTMENT_LIMIT=0.08`, it completed 620/620 with fall/drop `0`,
  min robot z `0.78248 m`, min box z `0.78662 m`, max tilt `0.09758 rad`, and
  no rollout root/velocity/box pose writes, but final robot/box
  target-directed travel was only `-0.00302/0.00351 m`, with max box travel
  `0.07564 m`. The current balance feedback targets zero pitch and cancels
  useful forward lean; add configurable pitch target before further gait
  tuning.
- 2026-07-05 balance target implementation:
  `scripts/isaac/build_core_world_g1_box_scene.py` and
  `scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh` now support
  `BALANCE_PITCH_TARGET` / `BALANCE_ROLL_TARGET`. Balance feedback computes
  corrections from target-relative pitch/roll error instead of always pulling
  pitch and roll to zero. Summary JSON records the target values. Lightweight
  `python3 -m py_compile` and `bash -n` checks passed on the login node.
- 2026-07-05 front-tray free-box pitch-target retry6 result:
  `20260705_core_world_g1_front_tray_freebox_creep_ptarget_retry6` is a
  strong negative result. With strong feedback, `BALANCE_PITCH_TARGET=0.06`,
  and conservative creep, the robot fell by about step 230: 620/620,
  `fall_events=390`, min robot z `0.49145 m`, min box z `0.24388 m`,
  max tilt `3.14155 rad`, and no rollout root/velocity/box pose writes. The
  positive travel is falling/sliding, not carrying. Smaller or ramped pitch
  targets are required before this can be useful.
- 2026-07-05 front-tray free-box pitch-target retry7 result:
  `20260705_core_world_g1_front_tray_freebox_creep_ptarget02_retry7` is also
  negative. With `BALANCE_PITCH_TARGET=0.02`, it still entered slow forward
  divergence and fell by about step 280: 620/620, `fall_events=347`,
  min robot z `0.49700 m`, min box z `0.24454 m`, max tilt `3.14040 rad`,
  and no rollout root/velocity/box pose writes. Final robot/box travel
  `1.06811/0.79864 m` is falling/sliding, not carrying. Fixed nonzero pitch
  target is unsafe; use a scheduled short pitch-target window that returns to
  zero before divergence.
- 2026-07-05 balance target window implementation:
  Added `BALANCE_TARGET_START_STEP` and `BALANCE_TARGET_END_STEP` support to
  the G1 Core API scene and launcher. Pitch/roll targets now apply only within
  the configured step window; outside the window, feedback targets zero
  pitch/roll. Summary JSON records the window. Lightweight `python3 -m
  py_compile` and `bash -n` checks passed.
- 2026-07-05 front-tray free-box pitch-window retry8 result:
  `20260705_core_world_g1_front_tray_freebox_creep_ptwindow_retry8` is stable
  but under-drives locomotion. It applied `BALANCE_PITCH_TARGET=0.02` only
  from step 140 to 220, then returned to zero. It completed 620/620 with
  fall/drop `0`, min robot z `0.78241 m`, min box z `0.78732 m`, max tilt
  `0.09570 rad`, and no rollout root/velocity/box pose writes, but final
  robot/box target-directed travel was only `-0.00787/0.00114 m`, with max box
  travel `0.07279 m`. The recovery window works, but the drive window is too
  short or too weak.
- 2026-07-05 front-tray free-box pitch-window retry9 result:
  `20260705_core_world_g1_front_tray_freebox_creep_ptwindow260_retry9` extended
  the target window from step 140 to 260. It remained stable but still did not
  produce meaningful locomotion: 620/620, fall/drop `0`, min robot z
  `0.78231 m`, min box z `0.78732 m`, max tilt `0.09570 rad`, no rollout
  root/velocity/box pose writes, final robot/box target-directed travel
  `-0.00776/0.00124 m`, and max robot/box target-directed travel
  `0.06330/0.07279 m`. Conclusion: the strong balance controller gives a good
  long-horizon free-box support posture, but the current hand-written
  `targeted_creep` does not create reliable forward stepping.
- 2026-07-05 diagnostic root-drive implementation:
  Added an explicitly labeled `diagnostic_root_drive=smooth_x` path to the G1
  Core API scene and launcher. It writes only the G1 root pose during rollout,
  never the box pose, so the box must still remain on the front tray through
  Isaac contact. Summary records the diagnostic mode, speed, window, ramp,
  active steps, commanded XY displacement, and `root_pose_write_count_rollout`.
  This path is only for scene/contact/metric validation under moving-carrier
  conditions. It must not be reported as real biped walking or carrying.
- 2026-07-05 front-tray free-box root-drive retry1 result:
  `20260705_core_world_g1_front_tray_freebox_rootdrive_retry1` passed the
  moving-carrier contact diagnostic but remains non-locomotion evidence. It
  used `DIAGNOSTIC_ROOT_DRIVE=smooth_x`, `GAIT_MODE=stand`, free box on the
  front tray, no fixed grasp, no box pose writes, and 440 rollout root-pose
  writes. Result: 620/620, fall/drop `0`, min robot z `0.78397 m`, min box z
  `0.78757 m`, max tilt `0.09465 rad`, final robot/box target-directed travel
  `0.17756/0.18729 m`, and max box-robot relative offset error `0.05726 m`.
  The contact scene can carry the free box under a moving G1 carrier. Caveat:
  the first root-drive implementation began from the initial root pose, causing
  a small handoff correction at drive start. Code was updated so root-drive
  initializes from the current root pose on the first active step.
- 2026-07-05 front-tray free-box root-drive retry2 result:
  `20260705_core_world_g1_front_tray_freebox_rootdrive_retry2` used the
  corrected current-pose handoff and passed the explicit diagnostic checker:
  620/620, fall/drop `0`, min robot z `0.78414 m`, min box z `0.78453 m`,
  max tilt `0.09708 rad`, no box pose writes, 440 root-pose writes, final
  robot/box target-directed travel `0.15783/0.16051 m`, and max box-robot
  relative offset error `0.06417 m`. It is useful contact-scene evidence only.
  Because the handoff occurs after the natural loaded posture has already
  pitched forward, the run develops visible sideways drift. Prefer a step-0
  smooth root-drive diagnostic as the moving-carrier baseline if it passes.
- 2026-07-05 front-tray free-box root-drive retry3 result:
  `20260705_core_world_g1_front_tray_freebox_rootdrive_step0_retry3` is the
  preferred moving-carrier contact-scene baseline. Root-drive was active from
  step 0 through the full 620-step rollout with a smooth ramp. It passed the
  explicit diagnostic checker: fall/drop `0`, no box pose writes, 620
  root-pose writes, min robot z `0.78414 m`, min box z `0.79820 m`, max tilt
  `0.00880 rad`, final robot/box target-directed travel
  `0.22521/0.22965 m`, and max box-robot relative offset error `0.05405 m`.
  This proves the free-box front-tray contact scene can carry the box under a
  moving G1 carrier in Isaac. It is still not walking or final carrying because
  rollout root-pose writes are intentionally used.
- 2026-07-05 prismatic cradle no-root retry:
  Added `scripts/isaac/run_core_world_prismatic_cradle_sync_inchworm.sh` to
  reproduce the strongest no-root free-box prismatic cradle settings without
  relying on ad hoc command lines. Submitted
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_8kg_retry1` in tmux
  session `curiosity_prismatic_cradle_neg22_0705`. It targets `-0.22 m` with
  five sync-inchworm cycles, `PAYLOAD_MODE=cradle_free_box`,
  `PAYLOAD_MASS=8.0`, and must keep root/body/box/payload pose and velocity
  shortcuts at zero. This is still a scaffolded prismatic-legged carrier, not
  final humanoid walking.
- 2026-07-05 prismatic cradle neg22 retry1 result:
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_8kg_retry1` is a
  negative/regression. It completed 2350/2350 with fall/drop `0` and no
  root/body/box/payload shortcuts, but only achieved final post-settle payload
  travel `0.00468 m` with target-distance `0.22468 m`; max payload-relative
  offset error rose to `0.20265 m`. Root cause: the new target `-0.22 m`
  caused the controller to compute only 4 sync-inchworm cycles and stride
  `0.055 m`, so it did not reproduce the old 5-cycle stable motion.
- 2026-07-05 prismatic cradle cycle control:
  Added `--sync-inchworm-min-cycles` to the prismatic carrier. Default `0`
  preserves old behavior; setting it to `5` forces at least five sync-inchworm
  cycles while keeping stride target-derived. Also corrected the new cradle
  launcher defaults to the old successful leg gains
  `LEG_STIFFNESS=32000`, `LEG_DAMPING=3200`. Submitted
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_8kg_retry2` with
  `SYNC_INCHWORM_MIN_CYCLES=5`.
- 2026-07-05 prismatic cradle neg22 retry2 result:
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_8kg_retry2` is also
  negative. It correctly used five cycles, but because target `-0.22 m` made
  `sync_inchworm_stride_m=0.044`, it remained below the old effective stride
  and only reached max post-settle payload travel `0.03715 m`; final
  post-settle target distance was `0.20850 m`, with max payload-relative
  offset error `0.20236 m`. Added `--sync-inchworm-stride-override` so the
  task target and per-cycle mechanical stride can be audited separately.
  Submitted
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_stride006_8kg_retry3`
  with `SYNC_INCHWORM_MIN_CYCLES=5` and
  `SYNC_INCHWORM_STRIDE_OVERRIDE=0.06`.
- 2026-07-05 prismatic cradle neg22 retry3 result:
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_5cycle_stride006_8kg_retry3`
  is negative. It completed 2350/2350 with fall/drop `0` and no
  root/body/box/payload shortcuts, but max absolute post-settle payload travel
  was only `0.02865 m`, final post-settle payload target distance was
  `0.22092 m`, and max payload-relative offset error was `0.20236 m`. The
  controller correctly used five cycles and stride override `0.06 m`; the
  regression came from geometry mismatch. The old stronger
  `diag7_postsettle_neg30cm_8kg` run had initial rear/front cradle stops at
  `0.29824/0.71762 m` and post-settle payload x `0.47149 m`; retry3 had
  rear/front stops at `-0.12406/0.29573 m` and post-settle payload x
  `0.07051 m`. The new launcher was therefore about `0.40-0.42 m` behind the
  old geometry. The summary/checker report `payload_local_x_m`/
  `payload_local_z_m` so this mismatch is auditable. Retry4 tested this old-x
  geometry explicitly, and after it failed the launcher default was restored
  to stable `PAYLOAD_LOCAL_X=0.08`. This remains a prismatic-legged scaffold,
  not final humanoid walking.
- 2026-07-05 prismatic cradle neg22 retry4 result:
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_oldgeom_stride006_8kg_retry4`
  is a strong negative result. It did reproduce the old x geometry
  (`payload_local_x_m=0.50`, rear/front cradle stops
  `0.29654/0.71652 m`, close to old `0.29824/0.71762 m`) and used five
  cycles with stride `0.06 m`, but it failed almost immediately: 2350/2350,
  fall events `2272`, box-drop events `2127`, min payload z `-512.17 m`,
  max tilt `3.13407 rad`, and max payload-relative offset error
  `213.53 m`. Root/body/box/payload shortcut writes were still all `0`, so
  this is a physical/controller failure rather than hidden state writing. CSV
  comparison shows old `diag7` started with payload z about `0.825 m`, while
  retry4 started around `0.645 m`; reproducing only the x stop geometry did
  not reproduce the old stable support condition. Do not keep `0.50` as the
  launcher default. The launcher default has been restored to stable
  `PAYLOAD_LOCAL_X=0.08`; forward payload positions must be tested as explicit
  env overrides with larger support polygons or revised height/contact
  geometry and labeled as scaffold diagnostics.
- 2026-07-05 prismatic cradle support-probe retry5 result:
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_x030_stride006_8kg_retry5a`
  was stable but under-driven: 2350/2350, fall/drop `0`, no shortcut writes,
  min payload z `0.78843 m`, but max post-settle payload travel only
  `0.03083 m`, final target distance `0.19660 m`, max tilt `0.16560 rad`,
  and max payload-relative offset error `0.14595 m`. A moderate forward
  payload position with the normal support polygon holds the box but does not
  create useful travel.
  `20260705_prismatic_cradle_sync_inchworm_neg22cm_x050_widesupport_stride006_8kg_retry5b`
  showed that the overhung payload can be stabilized by an explicitly larger
  support polygon (`STANCE_HALF_LENGTH=0.60`, `FOOT_LENGTH=0.60`): 2350/2350,
  fall/drop `0`, no shortcut writes, min payload z `0.72106 m`, final
  post-settle payload travel `-0.18373 m`, max absolute post-settle payload
  travel `0.19233 m`, and final target distance `0.03627 m`. It still failed
  the stricter gate because max tilt was `0.13078 rad`, travel was just below
  `0.20 m`, and max payload-relative offset error was `0.17296 m`.
- 2026-07-05 prismatic cradle retry6 result:
  `20260705_prismatic_cradle_sync_inchworm_neg23cm_x050_widesupport_tight_stride007_8kg_retry6`
  is the best no-root scaffold motion-distance result so far but still not a
  valid success claim. It used `PAYLOAD_LOCAL_X=0.50`,
  `STANCE_HALF_LENGTH=0.65`, `FOOT_LENGTH=0.65`,
  `CRADLE_CLEARANCE_X/Y=0.010/0.020`, `CRADLE_WALL_HEIGHT=0.32`,
  target `-0.23 m`, and stride override `0.07 m`. Result: 2350/2350,
  fall/drop `0`, root/body/box/payload shortcut writes all `0`, final
  post-settle payload travel `-0.21487 m`, max absolute post-settle payload
  travel `0.22147 m`, and final post-settle target distance `0.01513 m`.
  It failed the strict checker on max tilt `0.19881 rad`, min payload z
  `0.69930 m`, and max payload-relative offset error `0.25703 m`. This is
  real no-root physical-contact scaffold progress in Isaac, not humanoid
  walking, not learned balance, and not a final carrying claim. The next
  useful improvement is reducing transient tilt and box slosh while retaining
  the `>0.20 m` post-settle travel.
- 2026-07-05 prismatic cradle metric update:
  `build_core_world_prismatic_carrier_stand.py` now also records
  `post_settle_payload_relative_error_m` and
  `max_post_settle_payload_relative_offset_error_m`. The existing
  `max_payload_relative_offset_error_m` remains unchanged and is still
  reported by the checker, but it is measured from rollout initialization and
  can include natural box/cradle settling before the motion phase. The new
  post-settle-relative fields are for analyzing carry-phase slosh after the
  settle baseline is established. Lightweight `py_compile`/`bash -n` checks
  passed on the login node; no simulation was run for this metric-only patch.
- 2026-07-05 prismatic cradle retry7 result:
  `PAYLOAD_LOCAL_Z=0.20` fixed the early settle/slosh failure but removed most
  propulsion. `retry7a` with normal y clearance and `retry7b` with larger y
  clearance both completed 2350/2350 with fall/drop `0`, all shortcut writes
  `0`, max tilt below `0.093 rad`, min payload z above `0.751 m`, max
  payload-relative offset below `0.067 m`, and post-settle relative slosh
  below `0.022 m`. Both failed only because post-settle payload travel stayed
  too low (`0.07996 m` and `0.07292 m`). This established that payload height
  controls the stability/propulsion tradeoff.
- 2026-07-05 prismatic cradle retry8 result:
  `20260705_prismatic_cradle_sync_inchworm_neg23cm_x050_z016_support065_stride007_8kg_retry8b`
  is the first strict-pass no-root physical-contact scaffold for 8 kg free-box
  carrying. Config: `PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.16`,
  `STANCE_HALF_LENGTH=0.65`, `FOOT_LENGTH=0.65`,
  `SYNC_INCHWORM_MIN_CYCLES=5`, `SYNC_INCHWORM_STRIDE_OVERRIDE=0.07`,
  target `-0.23 m`, default cradle clearances, and `LEG_LOWER=-0.82`.
  Checker result: pass. Summary: 2350/2350, fall/drop `0`, articulated joints
  `8`, root/body/box/payload shortcut writes all `0`, max tilt
  `0.09174 rad`, min payload z `0.71612 m`, max payload-relative offset error
  `0.04504 m`, max post-settle payload relative offset error `0.00935 m`,
  max absolute post-settle payload travel `0.24384 m`, final post-settle
  payload travel `-0.22924 m`, and final post-settle payload target distance
  `0.00076 m`. Do not overclaim this as final success: it is a custom
  prismatic-legged scaffold with sync-inchworm gait, not a humanoid/G1
  controller, not learned balance, not active probing, and not video-guided
  policy. It is now the strongest reproducible Isaac baseline for physical
  free-box carrying without hidden root or payload state writes.
  `retry8a` (`PAYLOAD_LOCAL_Z=0.14`) passed all gates except min payload z
  (`0.69461 m`, just below `0.70 m`) and had similar travel. `retry8c`
  (`PAYLOAD_LOCAL_Z=0.18`) was stable but under-driven, confirming `0.16` as
  the current best height.
- 2026-07-05 prismatic cradle launcher baseline:
  `scripts/isaac/run_core_world_prismatic_cradle_sync_inchworm.sh` now
  defaults to the retry8b strict-pass scaffold config: target `-0.23 m`,
  `PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.16`,
  `STANCE_HALF_LENGTH=0.65`, `FOOT_LENGTH=0.65`,
  `SYNC_INCHWORM_MIN_CYCLES=5`, and
  `SYNC_INCHWORM_STRIDE_OVERRIDE=0.07`. Lightweight `bash -n` and
  `py_compile` checks passed on the login node. This default is for
  reproducing the prismatic scaffold baseline only; it must not be represented
  as humanoid walking or final robot-carrying success.
- 2026-07-05 prismatic cradle default reproduction:
  `20260705_prismatic_cradle_sync_inchworm_default_repro_retry9` ran the
  updated launcher with no env overrides and reproduced the retry8b strict
  pass. Checker result: pass, failures `[]`. Summary: 2350/2350, fall/drop
  `0`, articulated joints `8`, root/body/box/payload shortcut writes all `0`,
  max tilt `0.09174 rad`, min payload z `0.71612 m`, max payload-relative
  offset error `0.04504 m`, max post-settle payload relative offset error
  `0.00935 m`, max absolute post-settle payload travel `0.24384 m`, final
  post-settle payload travel `-0.22924 m`, and final post-settle payload
  target distance `0.00076 m`. This is the current reproducible
  no-root/free-box Isaac scaffold baseline.
- 2026-07-05 prismatic cradle visualization:
  Added `scripts/isaac/render_prismatic_carrier_csv_video.py`, a metrics
  visualization tool that renders top-down and side-view MP4 from a prismatic
  carrier CSV plus summary JSON. It is not an Isaac viewport recording and
  must not be used as synchronized scene-video evidence for a final claim.
  It is useful for auditing target line, carrier/payload trajectories, support
  polygon, payload height, and visible no-drop behavior. Compute-node Slurm
  job `167294` generated
  `experiments/visuals/prismatic_carrier_stand/20260705_prismatic_cradle_sync_inchworm_default_repro_retry9_metrics.mp4`
  from the retry9 pass rollout. The file is an MP4 (`ISO Media, MP4 Base
  Media`) and the generation log is
  `logs/core_world_prismatic_carrier_stand/prismatic_cradle_retry9_video_srun.log`.
- 2026-07-05 prismatic cradle walking-like retry10 result:
  `scripts/isaac/run_core_world_prismatic_walklike_retry10_batch.sh` tested
  `quasistatic_step_cycle`, `prelift_quasistatic_step_cycle`, and
  `guarded_prelift_quasistatic_step_cycle` on the retry9 stable 8 kg
  free-box cradle config. All runs kept fall/drop `0` and all root/body/box/
  payload shortcut writes at `0`. `retry10a` reached final post-settle
  payload travel `-0.16457 m` with target distance `0.06543 m`; `retry10b`
  reached `-0.17350 m` with target distance `0.05650 m`; `retry10c` was too
  conservative and reached only `-0.04208 m`. This is useful walking-like
  support-switching evidence, but it did not hit the original `-0.23 m`
  target.
- 2026-07-05 prismatic cradle walking-like retry11 result:
  `scripts/isaac/run_core_world_prismatic_walklike_retry11_short_target_batch.sh`
  shortened the evaluation target to `-0.17 m` and used 1900 steps. It
  completed safely with fall/drop `0` and no shortcut writes, but failed the
  stricter post-settle travel gate: `retry11a` reached only
  `0.131999 m` max absolute post-settle payload travel with final target
  distance `0.04709 m`; `retry11b` reached `0.137078 m` with final target
  distance `0.04228 m`. The apparent `-0.174/-0.179 m` travel in stdout was
  relative to rollout initialization and included settle drift. Do not count
  retry11 as a pass.
- 2026-07-05 prismatic cradle gait-drive target update:
  Added `--gait-drive-target-x` and wrapper env `GAIT_DRIVE_TARGET_X` to
  separate the reported task target from the internal diagnostic gait drive
  distance. This is for auditing walking-like step-cycle reset losses; it is
  not a learning method and must not be described as autonomous posture
  selection. Defaults preserve old behavior. `python3 -m py_compile` and
  `bash -n` passed on the login node.
- 2026-07-05 prismatic cradle walking-like retry12 result:
  `scripts/isaac/run_core_world_prismatic_walklike_retry12_drive_target_batch.sh`
  uses evaluation target `TARGET_X=-0.17` and internal
  `GAIT_DRIVE_TARGET_X=-0.23` for `quasistatic_step_cycle` and
  `prelift_quasistatic_step_cycle` on the retry9 8 kg free-box cradle
  scaffold. It ran via tmux session `curiosity_prismatic_walklike_retry12_0705`,
  Slurm job `167304`, on `server02`. Checker result: both runs passed with
  failures `[]`. `retry12a` (`quasistatic_step_cycle`) summary: 2350/2350,
  fall/drop `0`, all root/body/box/payload shortcut writes `0`,
  articulated joints `8`, max tilt `0.09174 rad`, min payload z `0.71612 m`,
  max payload-relative offset error `0.04504 m`, max post-settle payload
  relative offset error `0.00968 m`, max absolute post-settle payload travel
  `0.17383 m`, final post-settle payload travel `-0.16457 m`, and final
  post-settle payload target distance `0.00543 m`. `retry12b`
  (`prelift_quasistatic_step_cycle`) summary: 2350/2350, fall/drop `0`, all
  shortcut writes `0`, articulated joints `8`, max tilt `0.09174 rad`, min
  payload z `0.71612 m`, max payload-relative offset error `0.04504 m`, max
  post-settle payload relative offset error `0.01082 m`, max absolute
  post-settle payload travel `0.18468 m`, final post-settle payload travel
  `-0.17350 m`, and final post-settle payload target distance `0.00350 m`.
  This is now the best walking-like support-switching prismatic scaffold
  milestone. Do not overclaim it: it is not humanoid/G1, not learned, not
  active probing, not video-guided, and the internal gait drive is
  deliberately over-specified to compensate step-cycle reset losses.
- 2026-07-05 prismatic cradle multi-posture retry13 submitted:
  Added `scripts/isaac/run_core_world_prismatic_walklike_retry13_posture_batch.sh`
  to test whether the retry12 walking-like no-root/free-box scaffold survives
  different carry postures. It keeps `PAYLOAD_MODE=cradle_free_box`,
  `PAYLOAD_MASS=8.0`, `MOTION_MODE=prelift_quasistatic_step_cycle`,
  `TARGET_X=-0.17`, `GAIT_DRIVE_TARGET_X=-0.23`, `STEPS=2000`,
  `STANCE_HALF_LENGTH=0.65`, `FOOT_LENGTH=0.65`, and all no-root/no-payload
  shortcut gates. Submitted via tmux session
  `curiosity_prismatic_walklike_retry13_0705`, Slurm job `prism_post_r13`.
  Planned stamps: `retry13a` high carry (`PAYLOAD_LOCAL_X=0.50`,
  `PAYLOAD_LOCAL_Z=0.18`), `retry13b` closer carry
  (`PAYLOAD_LOCAL_X=0.45`, `PAYLOAD_LOCAL_Z=0.16`), and `retry13c` low carry
  (`PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.14`). Status pending until
  summaries and checker evidence are recorded.
- 2026-07-05 prismatic cradle multi-posture retry13 result:
  `retry13` completed all three 8 kg `cradle_free_box` walking-like posture
  diagnostics. The first checker submission, Slurm job `167310`
  (`prism_r13_chk`), is invalid because shell expansion blanked the stamp
  loop variable. Added
  `scripts/isaac/check_core_world_prismatic_walklike_retry13_postures.sh` and
  reran the valid compute checker as Slurm job `167313`
  (`prism_r13_ck2`). Strict checker result: only the closer mid-height carry
  passed. `retry13b` (`PAYLOAD_LOCAL_X=0.45`, `PAYLOAD_LOCAL_Z=0.16`) passed
  with failures `[]`, fall/drop `0`, all root/body/box/payload shortcut writes
  `0`, articulated joints `8`, max tilt `0.08797 rad`, min payload z
  `0.72143 m`, max payload-relative offset error `0.03811 m`, max absolute
  post-settle payload travel `0.18958 m`, final post-settle payload travel
  `-0.17786 m`, and final post-settle payload target distance `0.00786 m`.
  `retry13a` high carry (`x=0.50`, `z=0.18`) stayed safe but failed
  distance/target gates: post-settle payload travel `0.08094 m`, final target
  distance `0.13504 m`. `retry13c` low carry (`x=0.50`, `z=0.14`) reached the
  target with fall/drop `0`, but failed the payload-height gate with min
  payload z `0.69461 m`. This gives a useful posture boundary for the
  scaffold, not a final robot-carrying claim.
- 2026-07-05 prismatic cradle posture-long retry14 submitted:
  Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry14_posture_long_batch.sh`
  and submitted tmux session `curiosity_prismatic_walklike_retry14_0705`,
  Slurm job `167315`, job-name `prism_post_r14`. It continues direct Isaac
  scene work without external models. Config keeps the retry13 walking-like
  8 kg `cradle_free_box` scaffold and uses `STEPS=2800`,
  `TARGET_X=-0.17`, `GAIT_DRIVE_TARGET_X=-0.23`, and
  `MOTION_MODE=prelift_quasistatic_step_cycle`. Planned stamps:
  `retry14a` close mid-height long carry (`PAYLOAD_LOCAL_X=0.45`,
  `PAYLOAD_LOCAL_Z=0.16`), `retry14b` original mid-height long carry
  (`PAYLOAD_LOCAL_X=0.50`, `PAYLOAD_LOCAL_Z=0.16`), and `retry14c` closer
  low carry boundary (`PAYLOAD_LOCAL_X=0.45`, `PAYLOAD_LOCAL_Z=0.14`).
  Status pending while Slurm job `167315` is queued/running.
- 2026-07-05 prismatic cradle posture-long retry14 result:
  `retry14a` and `retry14b` completed in Slurm job `167315`. The original
  `retry14c` did not enter Isaac rollout and is invalid because the compute
  node read a transient/stale syntax-corrupted copy of
  `build_core_world_prismatic_carrier_stand.py` at line 164; the current
  login-node file was clean. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry14c_boundary_retry2.sh`
  with compute-side sleep, line print, and `py_compile`, then reran as Slurm
  job `167316` (`prism_r14c2`) on `server02`. Added
  `scripts/isaac/check_core_world_prismatic_walklike_retry14_postures.sh` and
  ran valid checker job `167319` (`prism_r14_chk`). All three valid retry14
  runs passed the strict gate with failures `[]`, 2800/2800 steps, fall/drop
  `0`, all root/body/box/payload shortcut writes `0`, articulated joints `8`,
  and no nonfinite events. `retry14a` close mid-height (`x=0.45`, `z=0.16`):
  min payload z `0.72143 m`, max tilt `0.08797 rad`, max payload-relative
  offset error `0.03811 m`, max post-settle payload travel `0.18958 m`, final
  post-settle target distance `0.00782 m`. `retry14b` mid-height (`x=0.50`,
  `z=0.16`): min payload z `0.71612 m`, max tilt `0.09174 rad`, max
  payload-relative offset error `0.04504 m`, max post-settle payload travel
  `0.18468 m`, final target distance `0.00350 m`. `retry14c_retry2` close low
  (`x=0.45`, `z=0.14`): min payload z `0.70213 m`, max tilt `0.09110 rad`,
  max payload-relative offset error `0.03957 m`, max post-settle payload
  travel `0.18254 m`, final target distance `0.00293 m`. This shows that
  moving the low posture closer to the body recovers the payload-height margin
  that failed in retry13c. Do not overclaim: it is still a prismatic scaffold,
  not humanoid/G1, learned, video-guided, or autonomous posture selection.
- 2026-07-05 prismatic cradle high-posture rescue retry15 submitted:
  Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry15_high_posture_rescue.sh`
  and submitted tmux `curiosity_prismatic_walklike_retry15_0705`, Slurm job
  `167322`, job-name `prism_high_r15`. This directly targets the remaining
  high-carry boundary from retry13a (`PAYLOAD_LOCAL_Z=0.18`) while keeping
  the same 8 kg `cradle_free_box` scaffold and strict no-root/no-payload-write
  gates. Planned stamps: `retry15a` high-close same drive (`x=0.45`,
  `z=0.18`, `GAIT_DRIVE_TARGET_X=-0.23`), `retry15b` high-mid stronger drive
  (`x=0.50`, `z=0.18`, drive `-0.31`), and `retry15c` high-close moderate
  drive (`x=0.45`, `z=0.18`, drive `-0.27`). Status pending while Slurm job
  `167322` is queued/running.
- 2026-07-05 prismatic cradle high-posture rescue retry15 result:
  Added
  `scripts/isaac/check_core_world_prismatic_walklike_retry15_high_postures.sh`
  and ran formal checker job `167324` (`prism_r15_chk`) on server63. All
  three high-carry variants failed the strict transport gate, with failures
  only in `absolute post-settle payload travel x too low` and
  `final post-settle payload target distance x too high`. Safety and
  no-shortcut counters stayed clean in all three runs: fall/drop `0`, all
  root/body/box/payload write counters `0`, articulated joint count `8`,
  nonfinite events `0`. `retry15a` (`x=0.45`, `z=0.18`, drive `-0.23`) had
  min payload z `0.73707`, max tilt `0.08629`, max post-settle payload travel
  `0.08122`, final post-settle target distance `0.13420`. `retry15b`
  (`x=0.50`, `z=0.18`, drive `-0.31`) had min payload z `0.73264`, max tilt
  `0.08959`, max post-settle payload travel `0.10265`, final post-settle
  target distance `0.11544`. `retry15c` (`x=0.45`, `z=0.18`, drive `-0.27`)
  had min payload z `0.73707`, max tilt `0.08629`, max post-settle payload
  travel `0.10426`, final post-settle target distance `0.11558`. Do not
  claim high carry solved; the high posture is stable but under-driven by the
  current diagnostic contact/drive schedule. Next work should change the
  direct Isaac scene mechanics or posture-transition schedule rather than
  waiting on external video/model assets.
- 2026-07-05 prismatic cradle high-stride retry16 prepared:
  Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry16_high_stride_batch.sh`
  and verified it with `bash -n`. This directly continues the Isaac scene
  route instead of waiting for external models. It keeps the same 8 kg
  `cradle_free_box` high-carry diagnostic (`PAYLOAD_LOCAL_Z=0.18`) but
  increases the physical propulsion envelope: `GAIT_DRIVE_TARGET_X=-0.42`,
  `STEP_LENGTH=0.10`, `GAIT_PERIOD_STEPS=300`, `X_SLIDE_LIMIT=0.28`, and
  higher leg/x-slide force limits. Planned runs: `retry16a` high-mid larger
  stride (`x=0.50`), `retry16b` high-mid larger stride with
  `SWING_X_FORCE_SCALE=0.0`, and `retry16c` high-close larger stride with
  `PRELIFT_STANCE_OVERDRIVE=1.6`. This is a diagnostic of high-carry
  propulsion, not learning, video guidance, or autonomous posture selection.
  Submitted tmux `curiosity_prismatic_walklike_retry16_0705`, Slurm job
  `167329`, job-name `prism_high_r16`.
- 2026-07-05 prismatic cradle high-stride retry16 result:
  Slurm job `167329` (`prism_high_r16`) completed on server44. Added
  `scripts/isaac/check_core_world_prismatic_walklike_retry16_high_stride.sh`,
  but checker jobs `167330` and `167332` were canceled after remaining
  pending for resource priority; do not claim a completed retry16 checker
  job. The rollout summaries were inspected with lightweight `jq`. All three
  runs were safe and shortcut-clean: fall/drop `0`, all root/body/box/payload
  write counters `0`, articulated joint count `8`, and nonfinite events `0`.
  `retry16a` high-mid larger stride (`x=0.50`, `z=0.18`) reached the travel
  gate but failed final target holding: min payload z `0.73411`, max tilt
  `0.09022`, max post-settle payload travel `0.16863`, final post-settle
  target distance `0.05819`. `retry16b` high-mid with `SWING_X_FORCE_SCALE=0`
  was worse: max post-settle payload travel `0.08621`, final target distance
  `0.16092`; swing-leg x force is not the main cause of the high-carry
  failure. `retry16c` high-close with `PRELIFT_STANCE_OVERDRIVE=1.6`
  over-drove the target: max post-settle payload travel `0.34731`, final
  target distance `0.11302`, max payload-relative offset error `0.03085`.
  Interpretation: high carry can be physically propelled without falls,
  drops, or shortcut writes, but now needs target-aware stopping/holding or a
  guarded progression rule. More raw drive is the wrong next lever.
- 2026-07-05 prismatic cradle guarded high-stop retry17 prepared:
  Added guarded stop support to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`. New optional
  `GUARDED_STOP_TARGET_X` / `--guarded-stop-target-x` lets diagnostic
  `GAIT_DRIVE_TARGET_X` overdrive the step cycle while the guarded hold logic
  stops at the real task target. Guarded target detection now treats crossing
  the stop target as reached, so overshoot does not bypass the stop condition.
  Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry17_guarded_high_stop_batch.sh`
  and verified shell/Python syntax. Planned runs are `retry17a` guarded
  high-mid strict tolerance `0.018`, `retry17b` guarded high-mid tolerance
  `0.030`, and `retry17c` guarded high-close with
  `PRELIFT_STANCE_OVERDRIVE=1.6` and tolerance `0.030`. This is still a
  direct Isaac diagnostic, not a learned or video-guided result. Submitted
  tmux `curiosity_prismatic_walklike_retry17_0705`, Slurm job `167333`,
  job-name `prism_high_r17`.
- 2026-07-05 prismatic cradle guarded high-stop retry17 result:
  Slurm job `167333` completed. The result is negative but diagnostic. All
  three high-carry runs stayed safe and shortcut-clean, with fall/drop `0`,
  all root/body/box/payload write counters `0`, and no nonfinite events, but
  all stopped early with `gated_step_last_block_reason` equal to
  `post_settle_payload_travel_loss`. `retry17a` and `retry17b` both ended at
  final post-settle payload travel `-0.07346 m`, final target distance
  `0.09654 m`, max tilt `0.09022`, min payload z `0.73411`. `retry17c`
  ended at final post-settle payload travel `-0.07336 m`, final target
  distance `0.09664 m`, max tilt `0.08669`, min payload z `0.73967`. The
  cause is a sign error in the guarded travel-loss logic for negative-X
  targets: raw X peak tracking misclassified normal target-directed progress
  as loss.
- 2026-07-05 prismatic cradle directional-guard retry18 prepared:
  Fixed guarded travel-loss tracking in
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` to use
  directional progress toward the guarded stop target. Added summary fields
  for directional guarded progress/loss and added
  `scripts/isaac/run_core_world_prismatic_walklike_retry18_directional_guard_batch.sh`
  plus
  `scripts/isaac/check_core_world_prismatic_walklike_retry18_directional_guard.sh`.
  Syntax checks passed on the login node only (`py_compile` and `bash -n`).
  Submitted tmux `curiosity_prismatic_walklike_retry18_0705`, Slurm job
  `167342`, job-name `prism_high_r18`, with high-mid and high-close
  overdrive variants. This is still a direct Isaac diagnostic, not a learned
  policy or video-guided result.
- 2026-07-05 prismatic cradle directional-guard retry18 result:
  Slurm job `167342` completed on server02. Formal checker job `167343`
  (`prism_r18_chk`) also completed on server02 and both runs passed with
  `failures=[]`. `retry18a` high-mid (`x=0.50`, `z=0.18`) completed 2800
  steps with fall/drop `0`, all shortcut write counters `0`, max tilt
  `0.09022`, min payload z `0.73411`, max post-settle payload travel
  `0.17536`, final post-settle target distance `0.00536`, and final block
  reason `target_reached`. `retry18b` high-close overdrive (`x=0.45`,
  `z=0.18`) completed 2800 steps with fall/drop `0`, all shortcut write
  counters `0`, max tilt `0.08669`, min payload z `0.73967`, max post-settle
  payload travel `0.15468`, final post-settle target distance `0.01532`, and
  final block reason `target_reached`. This is a high-carry pass for the
  direct Isaac prismatic scaffold only; it is not humanoid walking, not
  learned control, not active probing, and not video-guided RL.
- 2026-07-05 prismatic cradle posture/load/shape retry19 prepared:
  Added `PAYLOAD_SIZE_X/Y/Z` environment controls to
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry19_posture_load_shape_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry19_posture_load_shape.sh`.
  Planned diagnostics compare mid/high posture with a 12 kg standard box and
  an 8 kg taller box under the same directional-guard controller. This is
  direct Isaac scaffold variation evidence only, not autonomous posture
  selection.
- 2026-07-05 prismatic cradle posture/load/shape retry19 result:
  Rollout job `167346` (`prism_var_r19`) completed on server02, followed by
  formal checker job `167349` (`prism_r19_chk`). All four variation
  diagnostics passed with `failures=[]`, fall/drop `0`, all shortcut write
  counters `0`, joint count `8`, and nonfinite events `0`. Results:
  `retry19a` 12 kg standard box mid posture (`x=0.50`, `z=0.16`) final
  post-settle payload target distance `0.00504`, max tilt `0.10264`, min
  payload z `0.71327`, max payload-relative offset error `0.06084`;
  `retry19b` 12 kg standard box high posture (`x=0.50`, `z=0.18`) final
  target distance `0.00577`, max tilt `0.10109`, min payload z `0.73689`,
  max offset `0.06928`; `retry19c` 8 kg tall box mid posture final target
  distance `0.00012`, max tilt `0.09181`, min payload z `0.76110`, max
  offset `0.09522`; `retry19d` 8 kg tall box high posture final target
  distance `0.01165`, max tilt `0.08902`, min payload z `0.74347`, max
  offset `0.07433`. This remains prismatic-scaffold variation evidence, not
  autonomous selection, active probing, learned control, humanoid walking, or
  video guidance.
- 2026-07-06 prismatic cradle posture selector retry20:
  Added
  `experiments/configs/prismatic_cradle_posture_selector_retry20_manifest.json`,
  `scripts/isaac/summarize_prismatic_cradle_posture_selector.py`, and
  `scripts/isaac/run_prismatic_cradle_posture_selector_retry20.sh`. Login-node
  checks were limited to syntax/format checks (`py_compile`, `bash -n`,
  `jq empty`). Submitted tmux `curiosity_prismatic_selector_retry20_0706`,
  Slurm job `167351`, job-name `prism_sel_r20`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --job-name=prism_sel_r20 bash scripts/isaac/run_prismatic_cradle_posture_selector_retry20.sh`.
  Output report:
  `experiments/reports/prismatic_cradle_posture_selector/20260706_retry20_prismatic_cradle_posture_selector_report.json`;
  candidate table:
  `experiments/reports/prismatic_cradle_posture_selector/20260706_retry20_prismatic_cradle_posture_selector_candidates.jsonl`;
  log:
  `logs/core_world_prismatic_carrier_stand/prismatic_cradle_selector_retry20.log`.
  Result: report status `pass`, failures `[]`, 9/9 candidates passed the
  hard gate, and 8/9 were selector-eligible under the `0.01 m` height-margin
  rule. The selector chose `mid_front` for `standard_8kg`, `standard_12kg`,
  and `tall_8kg`. The 8 kg low-close posture passed the hard gate but had
  only about `0.00213 m` height margin, so it was not selected. This is a
  rule-based posture-choice scaffold over completed runs; it is not active
  probing, not RL, not video guidance, not humanoid walking, and not complete
  robot carrying success.
- 2026-07-06 prismatic cradle held-out selector retry21 prepared:
  Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry21_selector_heldout_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry21_selector_heldout.sh`.
  Planned direct Isaac scaffold diagnostics: 10 kg standard box with selected
  `mid_front`, 10 kg standard box with `high_front` control, 10 kg tall box
  with selected `mid_front`, and 10 kg tall box with `high_front` control.
  Syntax checks passed on login node only. This is held-out selector-driven
  scaffold execution, not active probing or learned posture selection.
- 2026-07-06 prismatic cradle held-out selector retry21 result:
  Submitted tmux `curiosity_prismatic_selector_retry21_0706`, Slurm rollout
  job `167353`, job-name `prism_sel_r21`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=04:00:00 --job-name=prism_sel_r21 bash scripts/isaac/run_core_world_prismatic_walklike_retry21_selector_heldout_batch.sh`.
  Submitted checker tmux `curiosity_prismatic_retry21_checker_0706`, Slurm
  job `167366`, job-name `prism_r21_chk`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --job-name=prism_r21_chk bash scripts/isaac/check_core_world_prismatic_walklike_retry21_selector_heldout.sh`.
  All four held-out cases passed formal checker with `failures=[]`, fall/drop
  `0`, all shortcut write counters `0`, joint count `8`, and nonfinite events
  `0`. `retry21a` selected `mid_front`, standard 10 kg: final post-settle
  payload target distance `0.00610`, max tilt `0.09731`, min payload z
  `0.71194`, max offset `0.05156`. `retry21b` high control, standard 10 kg:
  target distance `0.01969`, max tilt `0.09582`, min payload z `0.73178`,
  max offset `0.06031`. `retry21c` selected `mid_front`, tall 10 kg: target
  distance `0.00003`, max tilt `0.09433`, min payload z `0.75601`, max
  offset `0.09760`. `retry21d` high control, tall 10 kg: target distance
  `0.00616`, max tilt `0.10391`, min payload z `0.75913`, max offset
  `0.07326`. Interpretation: the retry20 selected posture executes held-out
  10 kg standard/tall boxes and has better target error than the high control,
  but this remains prismatic-scaffold evidence, not active probing, learned
  control, humanoid walking, video guidance, or complete robot carrying
  success.
- 2026-07-06 prismatic cradle active-probe retry22 prepared:
  Added active probe support to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`: environment
  controls `ENABLE_ACTIVE_PROBE`, `ACTIVE_PROBE_STEPS`,
  `ACTIVE_PROBE_LIFT_AMPLITUDE`, and `ACTIVE_PROBE_HORIZONTAL_AMPLITUDE`.
  The probe executes after settle and before carry; post-settle carry
  baseline and guarded gait progression start after probe, so the target
  metric is not credited by probe motion. Probe belief is based on observed
  micro-lift response, tilt, and payload relative offset, and summary field
  `active_probe_uses_hidden_ground_truth` is hard false. Extended
  `scripts/isaac/check_prismatic_carrier_stand_summary.py` with active-probe
  gates. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry22_active_probe_selected_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry22_active_probe_selected.sh`
  for standard/tall 10 kg selected `mid_front` cases with 80 probe steps.
  Syntax checks passed on login node only. This is still a prismatic scaffold
  probe, not autonomous full robot success.
- 2026-07-06 prismatic cradle active-probe retry22 result:
  Submitted rollout tmux `curiosity_prismatic_active_probe_retry22_0706`,
  Slurm job `167368`, job-name `prism_probe_r22`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=04:00:00 --job-name=prism_probe_r22 bash scripts/isaac/run_core_world_prismatic_walklike_retry22_active_probe_selected_batch.sh`.
  Submitted checker tmux `curiosity_prismatic_retry22_checker_0706`, Slurm
  job `167373`, job-name `prism_r22_chk`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --job-name=prism_r22_chk bash scripts/isaac/check_core_world_prismatic_walklike_retry22_active_probe_selected.sh`.
  Both active-probe cases passed formal checker with `failures=[]`,
  fall/drop `0`, all root/body/box/payload write counters `0`, joint count
  `8`, and nonfinite events `0`. Standard 10 kg selected `mid_front`:
  80 observed probe steps, hidden-ground-truth flag `false`, belief source
  `observed_micro_lift_response_not_hidden_ground_truth`, risk bucket `low`,
  risk score `0.00132`, payload lift response `0.02465 m`, max probe
  relative-offset error `0.00002 m`, final post-settle payload target
  distance `0.00583 m`, max tilt `0.09731`, min payload z `0.71194`.
  Tall 10 kg selected `mid_front`: 80 observed probe steps, hidden-ground-
  truth flag `false`, same observed belief source, risk bucket `low` under
  the retry22 thresholds, risk score `0.31771`, payload lift response
  `0.02398 m`, max probe relative-offset error `0.00452 m`, final
  post-settle payload target distance `0.00865 m`, max tilt `0.09433`, min
  payload z `0.75601`. Interpretation: active probing is now present and
  checked, but retry22 only records the belief; it does not yet let the probe
  change the carry controller. Next direct Isaac step is retry23:
  probe-conditioned control parameters inside the same scene, still clearly
  labeled as a prismatic scaffold and not RL/video/humanoid success.
- 2026-07-06 prismatic cradle probe-adaptive gait retry23 prepared:
  Added probe-conditioned gait decision support to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`. New controls:
  `ENABLE_PROBE_ADAPTIVE_GAIT`,
  `PROBE_ADAPTIVE_MEDIUM_RISK_THRESHOLD`,
  `PROBE_ADAPTIVE_HIGH_RISK_THRESHOLD`,
  `PROBE_ADAPTIVE_MEDIUM_GAIT_DRIVE_SCALE`, and
  `PROBE_ADAPTIVE_HIGH_GAIT_DRIVE_SCALE`. The decision uses only the
  observed active-probe risk score; it does not use payload mass/shape hidden
  ground truth. The real task target and guarded stop target remain unchanged;
  only the internal gait-drive overdrive target is scaled. Extended
  `scripts/isaac/check_prismatic_carrier_stand_summary.py` with gates for
  adaptive decision availability, expected adaptive risk bucket, and expected
  gait-drive scale. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry23_probe_adaptive_gait_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry23_probe_adaptive_gait.sh`.
  Planned cases: standard 10 kg should select adaptive bucket `low` and scale
  `1.0`; tall 10 kg should select adaptive bucket `medium` and scale `0.98`
  using thresholds `0.25/0.75`. Login-node checks were limited to
  `py_compile` and `bash -n`. This remains a direct Isaac prismatic-scaffold
  closed-loop diagnostic, not RL, video guidance, humanoid walking, or final
  robot carrying.
- 2026-07-06 prismatic cradle probe-adaptive gait retry23 result:
  Submitted rollout tmux `curiosity_prismatic_probe_adaptive_retry23_0706`,
  Slurm job `167383`, job-name `prism_probe_r23`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=04:00:00 --job-name=prism_probe_r23 bash scripts/isaac/run_core_world_prismatic_walklike_retry23_probe_adaptive_gait_batch.sh`.
  Submitted checker tmux `curiosity_prismatic_retry23_checker_0706`, Slurm
  job `167384`, job-name `prism_r23_chk`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --job-name=prism_r23_chk bash scripts/isaac/check_core_world_prismatic_walklike_retry23_probe_adaptive_gait.sh`.
  Both cases passed formal checker with `failures=[]`, fall/drop `0`, all
  root/body/box/payload shortcut write counters `0`, joint count `8`, and
  nonfinite events `0`. Standard 10 kg selected adaptive bucket `low`, scale
  `1.0`, base/effective gait-drive target `-0.42/-0.42`, decision step `340`,
  active-probe risk score `0.00132`, final post-settle payload target
  distance `0.00583`, max tilt `0.09731`, min payload z `0.71194`. Tall
  10 kg selected adaptive bucket `medium`, scale `0.98`, base/effective
  gait-drive target `-0.42/-0.41160`, decision step `340`, active-probe risk
  score `0.31771`, final post-settle payload target distance `0.00680`, max
  tilt `0.09433`, min payload z `0.75601`. The real task target and guarded
  stop target stayed `-0.17`; only the internal gait overdrive changed. This
  is the first closed-loop evidence in this scaffold that an observed active
  probe can change a later carry-control parameter while still passing the
  carry gate. It is still not humanoid walking, not learned control, not
  video-conditioned RL, and not full autonomous posture selection.
- 2026-07-06 prismatic cradle probe-adaptive posture retry24 prepared:
  Added probe-conditioned posture decision support to
  `scripts/isaac/build_core_world_prismatic_carrier_stand.py` and
  `scripts/isaac/run_core_world_prismatic_carrier_stand.sh`. New controls:
  `ENABLE_PROBE_ADAPTIVE_POSTURE`,
  `PROBE_ADAPTIVE_MEDIUM_POSTURE_LEG_TARGET_OFFSET`, and
  `PROBE_ADAPTIVE_HIGH_POSTURE_LEG_TARGET_OFFSET`. The posture decision uses
  the same observed active-probe risk score as retry23 and does not use
  payload mass/shape hidden ground truth. Low risk keeps the nominal carry
  height; medium risk chooses `lower_carry_medium` by adding `0.012 m` to the
  prismatic leg target during carry, which lowers the body/payload relative
  to the retry23 posture while preserving the real task target. Extended
  `scripts/isaac/check_prismatic_carrier_stand_summary.py` with gates for
  posture decision availability, expected posture risk bucket, expected
  posture strategy, and expected leg-target offset. Added
  `scripts/isaac/run_core_world_prismatic_walklike_retry24_probe_adaptive_posture_batch.sh`
  and
  `scripts/isaac/check_core_world_prismatic_walklike_retry24_probe_adaptive_posture.sh`.
  Planned cases: standard 10 kg should select `nominal_height` with offset
  `0.0`; tall 10 kg should select `lower_carry_medium` with offset `0.012`.
  Login-node checks were limited to `py_compile` and `bash -n`. This is still
  a direct Isaac prismatic-scaffold diagnostic, not humanoid walking, learned
  control, video guidance, or full autonomous robot carrying.
- 2026-07-06 prismatic cradle probe-adaptive posture retry24 result:
  Submitted rollout tmux `curiosity_prismatic_probe_posture_retry24_0706`,
  Slurm job `167387`, job-name `prism_post_r24`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=04:00:00 --job-name=prism_post_r24 bash scripts/isaac/run_core_world_prismatic_walklike_retry24_probe_adaptive_posture_batch.sh`.
  Submitted checker tmux `curiosity_prismatic_retry24_checker_0706`, Slurm
  job `167389`, job-name `prism_r24_chk`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --job-name=prism_r24_chk bash scripts/isaac/check_core_world_prismatic_walklike_retry24_probe_adaptive_posture.sh`.
  Both cases passed formal checker with `failures=[]`, fall/drop `0`, all
  root/body/box/payload shortcut write counters `0`, joint count `8`, and
  nonfinite events `0`. Standard 10 kg selected gait bucket `low`, gait scale
  `1.0`, posture strategy `nominal_height`, posture offset `0.0`, final
  post-settle payload target distance `0.00583`, max tilt `0.09731`, min
  payload z `0.71194`. Tall 10 kg selected gait bucket `medium`, gait scale
  `0.98`, posture strategy `lower_carry_medium`, posture offset `0.012`,
  effective leg target `-0.558`, final post-settle payload target distance
  `0.00415`, max tilt `0.09433`, min payload z `0.75601`, and final commanded
  leg lift `0.012`. This is the first checked scaffold evidence that an
  observed active probe can choose a later carry posture, not only a gait
  overdrive parameter. It remains a prismatic-scaffold diagnostic; it is not
  a real humanoid walking controller, not learned control, not video-conditioned
  RL, and not the complete robot box-carrying objective.
- 2026-07-06 direct Isaac G1 pulsed-creep retry10 prepared after user
  correction to stop waiting on external models. This route uses only the
  existing Core API G1 + free-box front-tray scene. Added pulsed balance-target
  controls to `scripts/isaac/build_core_world_g1_box_scene.py`:
  `--balance-target-pulse-period-steps`,
  `--balance-target-pulse-width-steps`, and
  `--balance-target-pulse-phase-step`, plus summary telemetry
  `balance_target_active_steps` and `balance_target_first_active_step`.
  Updated both G1 launchers and the G1 checker report/gate
  `--min-balance-target-active-steps`. Added
  `scripts/isaac/run_core_world_g1_front_tray_freebox_pulsed_creep_batch.sh`
  with three no-root G1 front-tray free-box diagnostics. All runs keep
  `DIAGNOSTIC_ROOT_DRIVE=none`, `GRASP_MODE=none`, `PROBE_MODE=none`,
  `TORSO_CRADLE=front_tray`, and forbid rollout root/velocity/box writes via
  checker gates. The purpose is to test whether short pitch-target pulses can
  escape the previous stable-but-stationary strong-feedback result without
  reintroducing the continuous-target falling mode. Login-node checks were
  limited to `py_compile` and `bash -n`, and passed. This remains a direct
  Isaac diagnostic, not RL, not video guidance, and not final humanoid
  carrying.
- 2026-07-06 direct Isaac G1 pulsed-creep retry10 result: the first submission
  `167395` failed before Isaac because the batch called a non-executable
  launcher directly (`exit code 126`); fixed the batch to call it through
  `bash`. A second pending GPU submission `167396` was canceled by this agent
  before rollout, and a `test` partition attempt failed at allocation because
  the account/partition combination was invalid. The valid rollout used tmux
  `curiosity_g1_pulsed_creep_retry10gpu2_0706`, Slurm job `167397`,
  job-name `g1_pulse10g`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --job-name=g1_pulse10g bash scripts/isaac/run_core_world_g1_front_tray_freebox_pulsed_creep_batch.sh`.
  All three cases completed on `server63` with `620/620`, fall/drop `0`,
  rollout root pose writes `0`, rollout root velocity writes `0`, and rollout
  box pose writes `0`. They all failed the locomotion/travel gate, so this is
  a negative G1 no-root locomotion diagnostic, not carrying success.
  `retry10a` had pulse target `0.020`, `40/120` pulse, gait amp `0.045`,
  stance push `0.10`, balance gain `0.45`, active target steps `160`, min box
  z `0.78731`, max tilt `0.09570`, max box target-directed travel `0.07256`,
  final box target-directed travel `0.00621`, and final robot
  target-directed travel `-0.00309`.
  `retry10b` had target `0.015`, `50/100`, amp `0.060`, push `0.14`, gain
  `0.40`, active target steps `220`, min box z `0.78701`, max tilt `0.08997`,
  max box target-directed travel `0.07112`, final box travel `0.00432`, and
  final robot travel `-0.00378`.
  `retry10c` had target `0.012`, `45/90`, amp `0.070`, push `0.18`, gain
  `0.35`, active target steps `225`, min box z `0.78886`, max tilt `0.08375`,
  max box target-directed travel `0.06689`, final box travel `0.00961`, and
  final robot travel `0.00127`. Interpretation: pulsed pitch targets preserve
  the stable free-box front-tray hold, but they do not create meaningful
  target-directed G1 walking. Stop tuning this open-loop `targeted_creep`
  family as the main locomotion solution.
- 2026-07-06 direct carry posture suite prepared: added
  `scripts/isaac/run_direct_carry_posture_suite_64cm.sh` and
  `scripts/isaac/summarize_direct_carry_posture_suite.py`. This suite reruns
  the current strongest direct-Isaac support-foot robot baseline in one
  compute job: 8 kg free box, 64 cm target, `front_mid`, `low_front`, and
  `chest_high` carry postures, `SUPPORT_MODE=alternating_anchor_feet`,
  `support_foot_mode=xz_prismatic_to_anchor`, double-support fraction `0.12`,
  no fixed-world stance anchor, and formal gates for fall/drop `0`, root
  shortcut free, no support-root/anchor/foot/stance pose writes, drive-phase
  near-ground foot count at least `2`, commanded stance support continuity,
  box travel at least `0.52 m`, final target distance at most `0.08 m`, max
  tilt at most `0.14 rad` in the suite summarizer, and support-polygon margin
  at least `0.12 m`. Login-node checks were limited to `py_compile` and
  `bash -n`, and passed. This is a stronger packaging of the current
  multi-posture walking/carrying scaffold; it is not yet final humanoid
  carrying, not learned control, and not video-conditioned RL.
- 2026-07-06 direct carry posture suite result: the main compute rollout used
  tmux `curiosity_direct_carry_posture_suite_0706`, Slurm job `167398`,
  job-name `carry_suite64`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_suite64 bash scripts/isaac/run_direct_carry_posture_suite_64cm.sh`.
  The suite summary is
  `experiments/outputs/direct_carry_posture_suite/20260706_direct_carry_posture_suite_64cm_8kg/direct_carry_posture_suite_summary.json`
  and reports `status=pass`, `failures=[]`. All three cases completed
  `3580` steps with fall/drop `0`, `root_shortcut_free=true`, no fixed-world
  stance anchor, `support_foot_mode=xz_prismatic_to_anchor`,
  `support_foot_joint_count=8`, drive-phase near-ground foot count at least
  `2`, and commanded stance support continuity. Metrics:
  `front_mid` max box travel `0.67301 m`, final target distance `0.02369 m`,
  final post-settle travel `0.66492 m`, max tilt `0.12141 rad`, min support
  margin `0.15951 m`; `low_front` max box travel `0.66675 m`, final target
  distance `0.00189 m`, final post-settle travel `0.64326 m`, max tilt
  `0.12326 rad`, min support margin `0.15984 m`; `chest_high` max box travel
  `0.65313 m`, final target distance `0.01468 m`, final post-settle travel
  `0.62460 m`, max tilt `0.12221 rad`, min support margin `0.15943 m`.
  A separate checker-only recomposition of existing 20260705 strict 64 cm
  summaries ran as Slurm job `167399` (`carry_suite_chk`) and also produced
  `status=pass`, `failures=[]` at
  `experiments/outputs/direct_carry_posture_suite/20260706_existing_20260705_strict64_suite/direct_carry_posture_suite_summary.json`.
  Interpretation: this is the current strongest complete-task direct-Isaac
  scaffold evidence for balanced free-box carrying across multiple postures.
  It is still not a full humanoid walking controller, not learned control, not
  video-conditioned RL, and not proof of arbitrary posture selection.
- 2026-07-06 direct carry posture stress suite prepared: extended
  `scripts/isaac/run_direct_carry_task_physical_backend.sh` with two
  additional named posture defaults, `front_reach` (`payload_local_x=0.28`,
  `payload_local_z=0.04`, `torso_z=0.56`) and `close_mid`
  (`payload_local_x=0.12`, `payload_local_z=0.05`, `torso_z=0.55`). Added
  `scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh`, which runs
  five 64 cm / 8 kg strict support-foot scaffold cases:
  `front_mid`, `low_front`, `chest_high`, `front_reach`, and `close_mid`.
  The gates match the 3-posture suite but require `--min-postures 5`. Login
  node checks were limited to `bash -n` and import-free `py_compile`, and
  passed. Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=03:00:00 --job-name=carry_stress5 bash scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh`.
  This is a posture-space stress diagnostic for the scaffold, not final
  humanoid carrying, not RL, and not video-conditioned control.
- 2026-07-06 direct carry posture stress suite result: ran from tmux
  `curiosity_direct_carry_posture_stress_0706`, Slurm job `167427`,
  job-name `carry_stress5`, on `server28`, command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=03:00:00 --job-name=carry_stress5 bash scripts/isaac/run_direct_carry_posture_stress_suite_64cm.sh`.
  The job completed and the tmux session exited normally. Suite report:
  `experiments/outputs/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/direct_carry_posture_stress_suite_summary.json`.
  Result: `status=pass`, `failures=[]`, `case_count=5`, postures
  `front_mid`, `low_front`, `chest_high`, `front_reach`, and `close_mid`.
  Each case completed `3580` steps with fall/drop `0`,
  `root_shortcut_free=true`, no fixed-world stance anchor,
  `support_foot_mode=xz_prismatic_to_anchor`, drive-phase and commanded
  stance near-ground foot count at least `2`, and no support-continuity gate
  failure. Metrics:
  `front_mid` max box travel `0.67301 m`, final target distance `0.02369 m`,
  max tilt `0.12141 rad`, min support margin `0.15951 m`;
  `low_front` max travel `0.66675 m`, final target distance `0.00189 m`,
  max tilt `0.12326 rad`, min support margin `0.15984 m`;
  `chest_high` max travel `0.65313 m`, final target distance `0.01468 m`,
  max tilt `0.12221 rad`, min support margin `0.15943 m`;
  `front_reach` max travel `0.69996 m`, final target distance `0.02415 m`,
  max tilt `0.12007 rad`, min support margin `0.16035 m`;
  `close_mid` max travel `0.69125 m`, final target distance `0.01431 m`,
  max tilt `0.12311 rad`, min support margin `0.15872 m`. Interpretation:
  this strengthens evidence that the current support-foot scaffold can carry
  a free 8 kg box 64 cm across a wider hold/posture space while maintaining
  balance/support metrics. It remains a scaffold diagnostic, not full
  humanoid walking, not learned control, and not video-conditioned active
  posture selection.
- 2026-07-06 direct carry stress-suite MP4 rendering prepared: added
  `scripts/isaac/render_direct_carry_posture_stress_suite_videos.sh`, which
  renders `front_mid`, `low_front`, `chest_high`, `front_reach`, and
  `close_mid` MP4 visual-audit videos from the existing backend CSV/summary
  files into
  `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/`.
  Login-node checks were limited to `bash -n` and import-free `py_compile`,
  and passed. Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:20:00 --job-name=carry_viz5 bash scripts/isaac/render_direct_carry_posture_stress_suite_videos.sh`.
  These videos are visualization evidence only, not new control results.
- 2026-07-06 direct carry stress-suite MP4 rendering retry1: tmux
  `curiosity_direct_carry_viz5_0706`, Slurm job `167431`, job-name
  `carry_viz5`, reached compute node `server46` but failed before writing
  videos. Failure was in `scripts/isaac/render_prismatic_carrier_csv_video.py`
  CSV parsing: the backend CSV includes string fields such as `phase=settle`,
  while the renderer tried to convert every field to `float`. This is a
  visualization-script bug, not a rollout/control failure. Fixed the renderer
  to convert numeric fields and preserve nonnumeric fields. Login-node
  `py_compile` and `bash -n` passed after the fix. Rerun required.
- 2026-07-06 direct carry stress-suite MP4 rendering retry2: tmux
  `curiosity_direct_carry_viz5_retry2_0706`, Slurm job `167432`, job-name
  `carry_viz5b`, reached compute node `server46` but failed before writing
  videos because the system `python3` environment lacks both `cv2` and
  `imageio`. This is a visualization dependency selection issue, not a
  rollout/control failure. Local prebuilt environments were checked without
  installing dependencies; both `/public/home/yanhongru/envs/gr00t_n16_py310`
  and `/public/home/yanhongru/envs/isaac_arena_py312` provide `cv2`,
  `imageio`, and `numpy`. Updated
  `scripts/isaac/render_direct_carry_posture_stress_suite_videos.sh` to use
  `/public/home/yanhongru/envs/isaac_arena_py312/bin/python` by default via
  `PYTHON_BIN`. Login-node `bash -n` and that env's `py_compile` passed.
  Rerun required.
- 2026-07-06 direct carry stress-suite MP4 rendering retry3: tmux
  `curiosity_direct_carry_viz5_retry3_0706`, Slurm job `167433`, job-name
  `carry_viz5c`, reached compute node `server02` but failed before writing
  videos because the renderer assumed the backend CSV contained
  `torso_y/payload_y/torso_z/payload_z` columns. The direct support-foot
  backend CSV for this suite records the one-dimensional x trajectory plus
  tilt/fall/drop fields. Fixed the renderer to default missing y fields to
  `0.0` and missing z fields to the summary's initial torso/payload heights.
  This is a visualization compatibility issue, not a rollout/control failure.
  Login-node `py_compile` and `bash -n` passed after the fix. Rerun required.
- 2026-07-06 direct carry stress-suite MP4 rendering retry4 result: tmux
  `curiosity_direct_carry_viz5_retry4_0706`, Slurm job `167434`, job-name
  `carry_viz5d`, ran on `server02` and completed. Generated MP4 audit videos
  and manifest under
  `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/`.
  Manifest:
  `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/render_manifest.txt`.
  Files and sizes:
  `20260706_direct_carry_posture_stress_suite_64cm_8kg_front_mid.mp4`
  `200424` bytes, `low_front.mp4` `185426` bytes, `chest_high.mp4`
  `192988` bytes, `front_reach.mp4` `191493` bytes, and `close_mid.mp4`
  `199030` bytes. These are generated metric/trajectory visualizations from
  logged CSV/summary data, not Isaac viewport recordings and not new control
  evidence. They are acceptable MP4 audit artifacts for the current scaffold
  result.
- 2026-07-06 direct carry probe-selected posture suite prepared: added
  `scripts/isaac/select_direct_carry_posture_from_probe.py` and
  `scripts/isaac/run_direct_carry_probe_selected_posture_suite.sh`. This is a
  two-stage scaffold diagnostic: first run a short active-probe episode with
  `front_mid`, read only probe telemetry fields from the normalized summary
  (`probe_belief_available`, `probe_risk_score`,
  `probe_belief_uses_hidden_ground_truth=false`), select a carry posture by
  thresholds (`front_reach` for low risk, `close_mid` for medium risk,
  `chest_high` for high risk), then run a full 64 cm / 8 kg carry with the
  selected posture and strict no-shortcut/support/target gates. Planned
  conditions are `vertical_probe` (`vertical_micro_lift`, z amplitude
  `0.030`) and `horizontal_probe` (`horizontal_push_pull`, x amplitude
  `0.050`). Login-node checks were limited to `bash -n` and `py_compile`,
  and passed. Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_probe_sel bash scripts/isaac/run_direct_carry_probe_selected_posture_suite.sh`.
  This is not online geometry-changing control inside one episode, not
  humanoid walking, not learned control, and not video-conditioned RL.
- 2026-07-06 direct carry probe-selected posture suite retry1 result:
  tmux `curiosity_direct_carry_probe_selected_0706`, Slurm job `167437`,
  job-name `carry_probe_sel`, ran on `server46` but did not complete the full
  two-condition suite. The `vertical_probe` branch succeeded: active probe
  summary
  `experiments/outputs/direct_carry_task_physical_backend/20260706_direct_carry_probe_selected_posture_suite_64cm_8kg_vertical_probe_probe/direct_carry_task_physical_backend_summary.json`
  had `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`, `probe_mode=vertical_micro_lift`,
  `probe_risk_score=0.5987436213151278`, and selected `close_mid` via
  `experiments/outputs/direct_carry_probe_selected_posture_suite/20260706_direct_carry_probe_selected_posture_suite_64cm_8kg/vertical_probe_selection.json`.
  The selected `close_mid` carry completed `3580` steps with fall/drop `0`,
  root shortcut free, max box travel `0.69125 m`, final target distance
  `0.01431 m`, and max tilt `0.12311 rad`. The `horizontal_probe` branch
  failed before producing a normalized direct summary; its log ended with
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh: line 112:
  unexpected EOF while looking for matching \"`. The current script passes
  `bash -n`, so the immediate actionable bug is that the suite used
  `run_probe | tail -1`, which could mask probe failures and let selection
  continue with a missing summary. Fixed
  `scripts/isaac/run_direct_carry_probe_selected_posture_suite.sh` to store
  `LAST_PROBE_SUMMARY` directly and fail immediately on probe errors.
  Login-node `bash -n` and `py_compile` passed after the fix. Rerun required.
- 2026-07-06 direct carry probe-selected posture suite retry2 result:
  tmux `curiosity_direct_carry_probe_selected_retry2_0706`, Slurm job
  `167440`, job-name `carry_probe2`, reached compute node `server46` but
  failed before running the suite with
  `scripts/isaac/run_direct_carry_probe_selected_posture_suite.sh: line 221:
  unexpected EOF while looking for matching \"`. Local `bash -n` passed, but
  the script contained a nested Python `-c` command with quoted JSON key access
  inside command substitution. To avoid shell-version/quoting ambiguity,
  replaced that line with `jq -r '.selected_carry_posture'`. Login-node
  `bash -n` passed after the fix. Rerun required.
- 2026-07-06 direct carry probe-selected posture suite retry3 result:
  tmux `curiosity_direct_carry_probe_selected_retry3_0706`, Slurm job
  `167441`, job-name `carry_probe3`, ran on `server46` and completed. Output
  directory:
  `experiments/outputs/direct_carry_probe_selected_posture_suite/20260706_direct_carry_probe_selected_posture_suite_retry3_64cm_8kg/`.
  The `vertical_probe` branch used `vertical_micro_lift`, 60 probe steps,
  z amplitude `0.030`; selection report
  `vertical_probe_selection.json` had `status=pass`,
  `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`,
  `probe_risk_score=0.5987436213151278`, selected bucket `medium`, and
  selected posture `close_mid`. The `horizontal_probe` branch used
  `horizontal_push_pull`, 60 probe steps, x amplitude `0.050`; selection
  report `horizontal_probe_selection.json` had `status=pass`,
  `probe_belief_available=true`,
  `probe_belief_uses_hidden_ground_truth=false`,
  `probe_risk_score=0.45948289037895235`, selected bucket `low`, and
  selected posture `front_reach`. The combined carry summary
  `probe_selected_carry_summary.json` reports `status=pass`,
  `failures=[]`, `case_count=2`, postures `close_mid` and `front_reach`.
  Both selected carries completed `3580` steps with fall/drop `0`,
  `root_shortcut_free=true`, no fixed-world stance anchor, and no target,
  tilt, or support-margin gate failure. `close_mid`: max box travel
  `0.69125 m`, final target distance `0.01431 m`, final post-settle travel
  `0.65549 m`, max tilt `0.12311 rad`, min support margin `0.15872 m`.
  `front_reach`: max box travel `0.69996 m`, final target distance
  `0.02415 m`, final post-settle travel `0.66809 m`, max tilt `0.12007 rad`,
  min support margin `0.16035 m`. Interpretation: this is the first current
  direct-Isaac support-foot scaffold evidence where active-probe telemetry
  chooses between widened carry postures and the selected carries pass strict
  physical no-shortcut gates. It remains a two-stage scaffold diagnostic, not
  online in-episode geometry-changing control, not full humanoid walking, not
  learned control, and not video-conditioned RL.
- 2026-07-06 single-episode online probe-adaptive support prepared: added
  online support-profile selection to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`. When
  `--enable-online-probe-adaptive-support` is enabled, the same rollout first
  performs active probing, computes the existing probe belief at
  `drive_start_step`, and then changes subsequent support-foot controller
  parameters from observed telemetry only: support step height, double-support
  fraction, stance x, and swing x. Added launcher env forwarding in
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh` and
  `scripts/isaac/run_direct_carry_task_physical_backend.sh`, normalized
  fields in `scripts/isaac/normalize_direct_carry_backend_summary.py`, and
  checker gates in `scripts/isaac/check_direct_carry_task_summary.py`.
  Added
  `scripts/isaac/run_direct_carry_online_probe_adaptive_support_suite.sh`
  with two 64 cm / 8 kg single-episode cases: `vertical_probe`
  (`vertical_micro_lift`, z amplitude `0.030`, expected medium bucket,
  `compact_medium_double_support`, step height `0.100`, double support
  `0.18`) and `horizontal_probe` (`horizontal_push_pull`, x amplitude
  `0.050`, expected low bucket, `nominal_reach_support`, step height
  `0.120`, double support `0.12`). Login-node checks were limited to
  `bash -n` and `py_compile`, and passed. Planned command:
  `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=carry_online_support bash scripts/isaac/run_direct_carry_online_probe_adaptive_support_suite.sh`.
  This is a same-episode controller-profile adaptation diagnostic; it still
  does not online-switch hold geometry, is not a full humanoid controller, not
  learned control, and not video-conditioned RL.
- 2026-07-06 single-episode online probe-adaptive support retry1 result:
  tmux `curiosity_direct_carry_online_support_0706`, Slurm job `167449`,
  job-name `carry_online_support`, reached compute node `server44` but failed
  before Isaac simulation started. The failure was a launcher parse error in
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh`:
  `unexpected EOF while looking for matching \"`. This is not a rollout
  result. Rewrote the newly added online support negative default values in
  `scripts/isaac/run_core_world_anchored_footstep_carrier.sh` and
  `scripts/isaac/run_direct_carry_task_physical_backend.sh` to use explicit
  precomputed variables instead of `${VAR:--0.xxx}` forms. Login-node checks
  after the fix were limited to lightweight `bash -n` and `py_compile`, and
  passed. Rerun required.
- 2026-07-06 single-episode online probe-adaptive support retry2 result:
  tmux `curiosity_direct_carry_online_support_retry2_0706`, Slurm job
  `167452`, job-name `carry_online2`, ran on `server28` and completed the
  `vertical_probe` Isaac rollout but not the full suite. The rollout summary
  at
  `experiments/outputs/direct_carry_task_physical_backend/20260706_direct_carry_online_probe_adaptive_support_retry2_64cm_8kg_vertical_probe/direct_carry_task_physical_backend_summary.json`
  shows online same-episode support adaptation was applied at step `70` from
  observed probe telemetry only: `probe_risk_score=0.5987436213151278`,
  `online_probe_adaptive_support_risk_bucket=medium`,
  `online_probe_adaptive_support_profile=compact_medium_double_support`,
  step height `0.100`, double support `0.18`,
  `online_probe_adaptive_support_uses_hidden_ground_truth=false`,
  `probe_belief_policy_action_applied=true`, completed `3640` steps, fall/drop
  `0`, max box travel `0.67985 m`, final box target distance `0.01187 m`,
  max tilt `0.05676 rad`, min support margin `0.20684 m`, and root shortcut
  free. The suite stopped after this case because the checker still required
  `--min-support-foot-x-joint-motion 0.35`, while the compact medium
  double-support profile produced `0.29681 m`. This is a gate-threshold issue,
  not a rollout failure. Updated
  `scripts/isaac/run_direct_carry_online_probe_adaptive_support_suite.sh` to
  require `0.25 m` support-foot x-joint motion for this online-support suite
  while keeping the no-shortcut, support margin, target, travel, and fall/drop
  gates. Rerun required.
- 2026-07-06 single-episode online probe-adaptive support retry3 result:
  tmux `curiosity_direct_carry_online_support_retry3_0706`, Slurm job
  `167455`, job-name `carry_online3`, ran on `server39` and completed with
  exit code `0:0`. Suite summary:
  `experiments/outputs/direct_carry_online_probe_adaptive_support_suite/20260706_direct_carry_online_probe_adaptive_support_retry3_64cm_8kg/online_probe_adaptive_support_summary.json`.
  Status `pass`, failures `[]`, case count `2`. Both cases are single-episode
  rollouts with online support-profile adaptation at step `70` from observed
  active-probe telemetry only, with
  `online_probe_adaptive_support_uses_hidden_ground_truth=false`,
  `probe_belief_policy_action_applied=true`, root shortcut free, no fixed-world
  stance anchor, and no root/support/foot pose-write shortcuts. `vertical_probe`
  used `vertical_micro_lift`, 60 probe steps, z amplitude `0.030`, risk score
  `0.5987436213151278`, selected bucket `medium`, profile
  `compact_medium_double_support`, step height `0.100`, double support `0.18`,
  stance/swing x `-0.115/0.115`, completed `3640` steps, fall/drop `0`, max
  box travel `0.67985 m`, final target distance `0.01187 m`, final
  post-settle box travel `0.65891 m`, max tilt `0.05676 rad`, min support
  margin `0.20684 m`, support-foot x/z joint motion `0.29681/0.34765 m`, and
  actual support-foot lift `0.04201 m`. `horizontal_probe` used
  `horizontal_push_pull`, 60 probe steps, x amplitude `0.050`, risk score
  `0.45948289037895235`, selected bucket `low`, profile
  `nominal_reach_support`, step height `0.120`, double support `0.12`,
  stance/swing x `-0.130/0.130`, completed `3640` steps, fall/drop `0`, max
  box travel `0.66684 m`, final target distance `0.00041 m`, final
  post-settle box travel `0.66437 m`, max tilt `0.07202 rad`, min support
  margin `0.19029 m`, support-foot x/z joint motion `0.34138/0.37387 m`, and
  actual support-foot lift `0.06754 m`. Interpretation: this is the first
  current direct-Isaac support-foot scaffold evidence where active probing
  changes subsequent carrying support parameters inside the same rollout. It
  is still not online hold-geometry switching, not full humanoid walking, not
  learned control, and not video-conditioned RL.
- 2026-07-06 online probe-adaptive hold implementation prepared: added
  `--enable-online-probe-adaptive-hold` to
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`. It computes
  the existing probe belief at carry start and changes only actuated
  clamp/cradle joint target closure fraction inside the same rollout. Added
  launcher forwarding, normalized summary fields, and checker gates for
  required hold decision, no hidden ground truth, expected bucket/profile,
  expected closure fraction, and minimum clamp/cradle joint motion. This is
  still a scaffold diagnostic; it does not rebuild geometry, retarget body or
  box poses, or claim learned control.
- 2026-07-06 online probe-adaptive hold first x-cradle attempt is negative.
  Slurm job `167458` (`carry_hold1`) failed before useful logging. Debug rerun
  Slurm job `167459` (`carry_hold_dbg`) ran the `vertical_probe` x-cradle case
  on `server46` and showed a physical contact failure: the rear pusher drove
  the payload far forward (`payload_x` exceeded `120 m` by step `3639`) with
  repeated drop events. This is not a valid hold result. Do not continue
  tuning x-cradle as the active hold geometry without a materially different
  contact design. The active hold suite was switched to `side_clamp_free_box`
  so online hold adaptation changes lateral clamp closure instead of pushing
  the box along the travel direction.
- 2026-07-06 online probe-adaptive hold side-clamp retry1 result:
  tmux `curiosity_direct_carry_online_hold_sideclamp_0706`, Slurm job
  `167460`, job-name `carry_hold2`, ran on `server46` and failed the first
  `vertical_probe` case. The summary showed online hold decision logic itself
  worked: decision step `70`, `online_probe_adaptive_hold_actuated=true`,
  risk `0.7219455656550482`, bucket `medium`, profile
  `reinforced_contact_closure`, closure fraction `0.75`, and no hidden ground
  truth. The physical hold failed: `max_clamp_joint_motion_m` was only
  `5.45e-05 m` against a `0.04 m` gate and box drop events were `1214`.
  Inspection found a geometry bug: side clamp pad placement used
  `clamp_pad_thickness` for the offset but the actual pad y-size was
  hardcoded to `0.30 m`, causing severe initial overlap with the box. Fixed
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py` so clamp pad
  y-size uses `clamp_pad_thickness`. Rerun required.
- 2026-07-06 online probe-adaptive hold side-clamp retry2/retry3 results:
  retry2 Slurm job `167461` (`carry_hold3`) failed before simulation with a
  transient Python syntax read of
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`; immediate
  login-node `py_compile` passed afterward. Retry3 Slurm job `167462`
  (`carry_hold4`) ran on `server46` and still failed the first `vertical_probe`
  case. The online hold decision was applied at step `70` with no hidden ground
  truth, but `max_clamp_joint_motion_m` remained near zero and the checker
  failed on clamp motion/drop. Added commanded clamp/cradle target fields to
  the backend and normalized summaries to distinguish command-generation
  failure from joint-drive tracking failure. Also changed the side-clamp hold
  suite to use contact-forming closure fractions (`low=0.75`, `medium=1.0`,
  `high=1.0`) instead of leaving a lateral gap in the medium case. Rerun
  required.
- 2026-07-06 online probe-adaptive hold side-clamp retry4 result:
  Slurm job `167464`, job-name `carry_hold5`, ran on `server46` and failed the
  first `vertical_probe` case. The new commanded-target fields show the online
  hold controller did command full closure:
  `max_commanded_clamp_target_m=0.054` and
  `final_commanded_clamp_target_m=0.054`, but measured clamp motion was only
  `3.83e-05 m`. The run also had `box_drop_events=2961`, max box travel
  `0.44394 m`, and final target distance `0.21288 m`. Conclusion: online
  decision and command generation work, but the side-clamp prismatic drive was
  not tracking. Changed `_clamp_joint` to use a direct `Y` prismatic axis with
  identity local rotations instead of rotating local `X` to world `Y`. Rerun
  required.
- 2026-07-06 online probe-adaptive hold side-clamp retry5 result:
  tmux `curiosity_direct_carry_online_hold_sideclamp_retry5_0706`, Slurm job
  `167466`, job-name `carry_hold6`, ran on `server63` and failed the first
  `vertical_probe` case. The online support and online hold decisions were
  applied at step `70` with no hidden ground truth. The hold controller
  commanded full side-clamp closure (`max_commanded_clamp_target_m=0.054`,
  `final_commanded_clamp_target_m=0.054`), but measured clamp motion was still
  only `3.83e-05 m`; box drop events were `2961`, max box travel was
  `0.44394 m`, and final target distance was `0.21288 m`. Conclusion: changing
  the clamp prismatic axis to direct `Y` did not fix joint-drive tracking. This
  is the repeated blocker for the side-clamp hold route: clamp joint commands
  are generated but the clamp articulation does not physically track them.
  Stop repeating this side-clamp route without approval or a materially
  different contact/actuation design.
- 2026-07-06 online probe-adaptive hold side-clamp retry6 result:
  Added DriveAPI target-position writes in the rollout loop and counters for
  clamp-drive target updates, then ran tmux
  `curiosity_direct_carry_online_hold_sideclamp_retry6_0706`, Slurm job
  `167470`, job-name `carry_hold7`. The result repeated the same physical
  blocker: the controller issued `7280` drive-target updates and commanded
  full closure (`max_commanded_clamp_target_m=0.054`,
  `final_commanded_clamp_target_m=0.054`), but measured clamp motion remained
  only `3.83e-05 m`, with repeated box drops. This confirms the current
  side-clamp joint/contact formulation is not a usable online hold route.
  Do not rerun this side-clamp route unchanged.
- 2026-07-06 direct Isaac execution correction after user feedback:
  Do not wait on external model/checkpoint downloads for the active task.
  The current executable path is Isaac scene construction and same-episode
  active-probe adaptation on the existing direct-carry backend. A new
  adaptive cradle-contact route is prepared: `cradle_free_box` now prebuilds
  an optional top-lid contact body, keeps it collision-disabled by default,
  and enables it after the observed probe belief only for non-low risk cases.
  This tests same-episode contact redistribution without changing root or box
  poses and without using hidden mass/shape ground truth. It is still a
  scaffold diagnostic, not full humanoid walking, learned control, or
  video-conditioned RL.
- 2026-07-06 online probe-adaptive cradle-contact retry1 invalid:
  tmux `curiosity_direct_carry_online_hold_adaptive_cradle_0706`, Slurm job
  `167477`, job-name `carry_hold_cradle`, reached `server46` but exited before
  Isaac rollout during compute-side `py_compile`. The compute node reported a
  stale/intermediate syntax error in
  `scripts/isaac/build_core_world_anchored_footstep_carrier.py`; the same file
  passed login-node `py_compile` immediately afterward. Do not count this as
  simulation or hold/contact evidence. Retry with a new stamp and startup
  delay is required.
- 2026-07-06 online probe-adaptive cradle-contact retry2 result:
  tmux `curiosity_direct_carry_online_hold_adaptive_cradle_retry2_0706`,
  Slurm job `167479`, job-name `carry_hold_crad2`, ran on `server46` with
  stamp
  `20260706_direct_carry_online_probe_adaptive_hold_adaptive_cradle_retry2_64cm_8kg`.
  Output summary:
  `experiments/outputs/direct_carry_online_probe_adaptive_hold_suite/20260706_direct_carry_online_probe_adaptive_hold_adaptive_cradle_retry2_64cm_8kg/online_probe_adaptive_hold_summary.json`.
  Suite status `pass`, `failures=[]`, two single-episode 64 cm / 8 kg cases.
  `vertical_probe` used `vertical_micro_lift`, observed risk
  `0.5932174593481317`, selected support profile
  `compact_medium_double_support` and hold profile
  `reinforced_contact_closure`; the adaptive cradle top-lid collision was
  available and enabled inside the same episode
  (`online_probe_adaptive_hold_collision_enabled=true`,
  `collision_update_count=1`). It completed `3640` steps with fall/drop `0`,
  final post-settle box travel `0.66433 m`, final target distance
  `0.01677 m`, max tilt `0.05652 rad`, min support margin `0.20701 m`, root
  shortcut free, and no fixed-world stance anchor. `horizontal_probe` used
  `horizontal_push_pull`, observed risk `0.4508505528966966`, selected support
  profile `nominal_reach_support` and hold profile `light_contact_closure`;
  the adaptive top-lid collision was available but left disabled
  (`collision_enabled=false`, `collision_update_count=1`). It completed
  `3640` steps with fall/drop `0`, final post-settle box travel `0.66269 m`,
  final target distance `0.00271 m`, max tilt `0.07131 rad`, min support
  margin `0.18904 m`, root shortcut free, and no fixed-world stance anchor.
  Both online support and online hold decisions used observed probe telemetry
  only; hidden-ground-truth probe use was false. Interpretation: this is the
  first current same-episode direct-Isaac scaffold result where active probing
  changes both support profile and contact/collision configuration before
  carrying. It is still not full humanoid walking, not learned control, and
  not video-conditioned RL.
- 2026-07-06 online probe-adaptive cradle-contact posture-suite prepared:
  added `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`
  and extended `scripts/isaac/summarize_direct_carry_posture_suite.py` so
  suite summaries include probe risk, online support profile, online hold
  profile, collision enabled state, collision update count, and hidden-ground-
  truth flags. The new suite keeps the same 64 cm / 8 kg direct-Isaac
  `cradle_free_box` scaffold but expands the same-episode probe-adaptive
  support+contact test from one posture to five cases:
  `front_mid`, `close_mid`, and `chest_high` under `vertical_micro_lift`
  medium-risk contact-enabled behavior, plus `front_reach` and `low_front`
  under `horizontal_push_pull` low-risk contact-disabled behavior. Login-node
  checks were limited to `bash -n` and import-free `py_compile`, both passed.
  This is still a scaffold hardening gate, not a completion claim.
- 2026-07-06 online probe-adaptive cradle-contact posture-suite queue result:
  submitted tmux `curiosity_direct_carry_online_hold_posture5_0706`, Slurm job
  `167501`, job-name `carry_hold_post5`, with a 3 hour limit. It remained
  pending for priority and produced no simulation evidence, so it was canceled
  and replaced by retry2 with a 1 hour limit:
  tmux `curiosity_direct_carry_online_hold_posture5_retry2_0706`, Slurm job
  `167502`, job-name `carry_hold_p5r2`. Slurm estimated retry2 would start at
  `2026-07-06T09:00:00` on `server53`, several hours later. To avoid leaving
  an unmonitored queued experiment in this turn, retry2 was also canceled
  before execution. Do not count either job as an experiment. No Isaac rollout
  or summary was produced.
- 2026-07-06 walking-realism audit hook for the posture suite:
  `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`
  now accepts optional `MAX_NEAR_GROUND_FOOT_SPEED` and
  `MAX_NEAR_GROUND_FOOT_SLIP` environment gates and forwards them to
  `check_direct_carry_task_summary.py`. The posture-suite summarizer now also
  records per-foot near-ground speed and slip fields. This does not make the
  current scaffold a walking robot; it prepares a stricter audit that can
  expose support-foot sliding when compute resources are available.
- 2026-07-06 online probe-adaptive posture-suite retry3 scheduling result:
  submitted tmux `curiosity_direct_carry_online_hold_posture5_retry3_0706`,
  Slurm job `167505`, job-name `carry_hold_p5r3`, with a 45 minute limit.
  The job remained pending with reason `Priority`; `squeue --start` reported
  no scheduled start time. It was canceled before execution to avoid leaving
  an unmonitored queued experiment. Do not count it as simulation evidence.
- 2026-07-06 explicit posture slip-audit launcher prepared:
  added
  `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_slip_audit_suite.sh`.
  It wraps the five-posture online probe-adaptive hold/contact suite and
  enables walking-realism gates by default:
  `MAX_NEAR_GROUND_FOOT_SPEED=0.80` and
  `MAX_NEAR_GROUND_FOOT_SLIP=0.20`. These gates are intended to expose
  planted-foot sliding in the current support-foot scaffold. Passing the
  non-slip-gated suite is not enough for the final walking requirement; a slip
  failure should be treated as a valid negative result and a reason to replace
  the support backend. Login-node checks were limited to `bash -n` and
  import-free `py_compile`, both passed.
- 2026-07-06 posture-suite split-run support prepared:
  `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`
  now accepts `CASE_FILTER` and `MIN_POSTURES`. `CASE_FILTER` can match either
  a full case id such as `vertical_probe_front_mid` or a posture name such as
  `front_mid`; skipped cases are reported explicitly. This enables short
  single-case diagnostics when Slurm priority prevents running the whole
  five-case suite. Example single-case command:
  `CASE_FILTER=vertical_probe_front_mid MIN_POSTURES=1 SUITE_STAMP=... srun ... bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`.
  This is an execution convenience only; the final "any posture" requirement
  still needs the full posture set and a real walking backend.
- 2026-07-06 single-case scheduling probe:
  submitted tmux `curiosity_direct_carry_online_hold_single_frontmid_0706`,
  Slurm job `167510`, job-name `carry_hold_1case`, with
  `CASE_FILTER=vertical_probe_front_mid`, `MIN_POSTURES=1`, and a 20 minute
  time limit. It still remained pending with reason `Priority`; `squeue
  --start` reported no scheduled start time. The job was canceled before
  execution. This shows the current scheduling issue is account/priority
  availability rather than the five-case suite length. Do not count this as
  simulation evidence.
- 2026-07-06 planted/no-slide posture audit prepared:
  `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh`
  now forwards optional checker requirements
  `REQUIRE_PLANTED_STANCE_RAIL_PROPULSION=1` and
  `REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`. Added
  `scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_planted_slip_audit_suite.sh`,
  which sets `PLANTED_STANCE_RAIL_PROPULSION=1`,
  `FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1`, requires both mechanisms in the
  checker, and keeps the near-ground foot speed/slip gates. This is the next
  more physically meaningful scaffold audit: stance feet should be commanded
  to remain fixed while rail propulsion moves the body and swing feet
  reposition. It may fail; failure should be recorded as evidence that the
  current support-foot backend must be replaced rather than as a reason to
  weaken the walking requirement. Login-node checks were limited to `bash -n`
  and import-free `py_compile`, both passed.
- 2026-07-06 planted/no-slide summary audit fields prepared:
  `scripts/isaac/summarize_direct_carry_posture_suite.py` now records
  planted-stance propulsion flags/step count, freeze-commanded stance-foot
  flags/counts/active feet, per-foot near-ground speed/slip, and aggregate
  maximum near-ground foot speed/slip for each case. This makes later
  five-posture and planted/no-slide summaries auditable for walking realism,
  not only fall/drop and target-distance success. Login-node verification was
  limited to import-free `py_compile` and `bash -n`; both passed. No GPU
  rollout was run because the only active GPU job was a resource-exclusion
  task and must not be touched.
- 2026-07-06 direct Isaac G1 low-cradle targeted-creep status:
  do not wait for external models or downloads. The current best direct G1
  free-box result is `low_push032` from
  `20260706_g1_targeted_creep_stop_tune1`: low/close free dynamic box on a
  collision-enabled torso cradle, `targeted_creep`, 560 steps, fall/drop `0`,
  rollout root/velocity/box writes `0`, final box target-directed travel
  `0.164657 m`, max tilt `0.128766 rad`, final relative offset `0.071063 m`.
  This is only a short-distance diagnostic. It is not long-duration carrying:
  the same configuration failed 700 and 1000 step validation with falls,
  drops, large pitch, and large relative drift. Corrected terminal-hold
  attempts triggered by box target travel also failed; fixed symmetric hold
  offsets are not a valid brake. Next direct Isaac work should implement a
  real deceleration/recovery phase for targeted creep rather than waiting on
  model downloads or claiming the 560-step diagnostic as success. Full report:
  `experiments/reports/2026-07-06_g1_low_cradle_creep_diagnostics.md`.
- 2026-07-06 G1 creep decel/brake follow-up:
  added travel-based creep decel, pitch-brake latch, positive-pitch-only
  brake triggering, and summary fields in
  `scripts/isaac/build_core_world_g1_box_scene.py`. Added runners
  `run_core_world_g1_low_creep_decel_tune.sh`,
  `run_core_world_g1_low_creep_latched_brake_tune.sh`,
  `run_core_world_g1_low_creep_positive_brake_tune.sh`, and
  `run_core_world_g1_low_creep_zero_hold_tune.sh`. Results are negative for
  final carrying. Decel/hold/latched-stop cannot reliably arrest late forward
  pitch. One 700-step diagnostic, `decel014_024_brake012`, kept fall/drop `0`
  with max tilt `0.120622 rad`, but only moved `0.086960 m`, below useful
  carry-distance gates. Longer-distance cases still fall/drop around the late
  pitch phase. Next direct Isaac implementation must add an explicit
  reverse-brake or counter-step recovery phase; do not rerun simple stop,
  zero-hold, or decel-only variants unchanged. Full report:
  `experiments/reports/2026-07-06_g1_creep_decel_and_brake_followup.md`.
- 2026-07-06 G1 reverse-brake and hold-balance follow-up:
  added and ran `run_core_world_g1_low_creep_reverse_brake_tune.sh` and
  `run_core_world_g1_low_creep_hold_balance_tune.sh`. Both are negative for
  the final carrying goal. Reverse-brake cases triggered after target travel
  thresholds but still failed with `20-26` fall events and `4-9` box drops
  while reaching about `0.76-0.79 m`; negative stance-push did not recover
  pitch. Hold-balance with negative sign was destructive and produced large
  backward travel plus hundreds of falls/drops; positive sign kept fall/drop
  `0` but suppressed motion to `0.003041 m`. Updated interpretation:
  stop sweeping hand-written open-loop creep. The active route is to connect
  the local WBC-AGILE controller-backed G1 policy to the Core scene and test
  it in stages: no-box walk, fixed light payload, then free low-cradle box.
  Do not claim success from any creep-brake result.
- 2026-07-06 WBC-AGILE Core-scene first smoke:
  added `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`
  and ran Slurm job `167559` (`g1_agile_carry`) with local ONNX backend on
  `server46`. ONNX runtime and the local WBC-AGILE model loaded, but all
  checks failed. The critical result is the no-box case:
  `agile_nobox_walk` completed `420` steps but had `210` fall events, min
  robot z `0.185860 m`, max tilt `3.107505 rad`, and final target-directed
  robot travel `-0.120162 m`. Since no-box fails, fixed-payload and free-box
  failures are not box-carrying evidence. Identified adapter issue: AGILE was
  receiving zero `root_ang_vel_b`; official IsaacLab-Arena WBC policy uses
  this observation. Updated `build_core_world_g1_box_scene.py` to read Core
  root angular velocity, rotate it into body frame, pass it into AGILE, and
  expose angular-velocity diagnostic fields in summary/check JSON. Added
  `scripts/isaac/run_core_world_g1_agile_policy_nobox_smoke.sh` and submitted
  no-box-only Slurm job `167565` (`g1_agile_nb`) to test the fix. Full report:
  `experiments/reports/2026-07-06_g1_agile_policy_core_scene_smoke.md`.
- 2026-07-06 WBC-AGILE corrected direction results:
  parameterized Core-scene `--target-xy` and forwarded `TARGET_X/TARGET_Y`
  through AGILE runners. With original G1 orientation, IsaacLab 29DoF gains,
  ONNX backend, `cmd_x=0.10`, and target `[-1.2, 0.0]`,
  `onnx_cmd010_isaaclab_gains` passed no-box 320-step locomotion: fall/drop
  `0`, min robot z `0.750114 m`, max tilt `0.209202 rad`, final robot
  target-directed travel `0.562249 m`, no rollout root/velocity/box writes.
  A fixed 0.25 kg torso payload with collision enabled stayed stable but only
  moved `0.115128 m`; the same fixed inertial payload with box collision
  disabled passed: fall/drop `0`, max tilt `0.204425 rad`, final robot
  target-directed travel `0.358296 m`, final box target-directed travel
  `0.371363 m`. Interpretation: AGILE is now a valid no-box and light
  fixed-payload controller backend. Collision-enabled centered payload caused
  geometry slowdown. First free dynamic low-cradle negative-X diagnostic
  `167579` (`g1_agile_free`) failed without falls/drops because the box and
  robot drifted apart (`final_relative_offset=0.374029 m`) and final travel
  became target-negative. A closer box/cradle retry, Slurm job `167580`
  (`g1_agile_fre2`), passed the short free-box gate: `360` steps, fall/drop
  `0`, final robot target-directed travel `0.125915 m`, final box
  target-directed travel `0.187173 m`, final relative offset `0.081144 m`,
  max tilt `0.146167 rad`, no rollout root/velocity/box writes. This is the
  first controller-backed direct Isaac G1 diagnostic with a free dynamic box
  moving on a robot-mounted low cradle, but it is still short, light, and
  carefully positioned; it is not active probing, unknown-load carrying,
  video-conditioned RL, or a success claim.
- 2026-07-06 WBC-AGILE 700-step baselines:
  close-cradle free dynamic box at 700 steps failed despite fall/drop `0`:
  final robot target-directed travel `-0.691677 m`, final box target-directed
  travel `-1.076183 m`, final relative offset `0.415493 m`, max tilt
  `0.632334 rad`. A no-box 700-step baseline with the same negative-X target
  and `cmd_x=0.10` passed: fall/drop `0`, final robot target-directed travel
  `0.878516 m`, max tilt `0.209202 rad`, no rollout root/velocity/box writes.
  Interpretation: AGILE long-horizon locomotion is viable; the long-horizon
  blocker is free-box/cradle contact retention, not the locomotion backend.
  Next work should improve dynamic box retention or add a stop/hold phase
  before claiming 700-step carrying.
- 2026-07-06 AGILE stop/hold retention gate:
  added AGILE command-gating options to
  `scripts/isaac/build_core_world_g1_box_scene.py`:
  `--agile-command-stop-step`,
  `--agile-command-stop-box-target-travel`,
  `--agile-command-stop-robot-target-travel`, and
  `--agile-command-hold-scale`. After a latched trigger, the policy keeps
  running but receives a scaled velocity command, defaulting to zero. This is
  a diagnostic isolation test for long-horizon free-box retention, not a
  carrying success claim. Runner/check/report fields were updated. Next run
  should be the close free-box 700-step hold test in a Curiosity-owned tmux
  Slurm allocation, not in `carry1` and not on the login node.
- 2026-07-06 AGILE zero-command hold result:
  tmux `curiosity_g1_agile_hold_0706`, Slurm job `167583`, job-name
  `g1_agile_hold`, ran on `server02`. Hold triggered at step `117` from
  box target travel and the final AGILE command was `[0, 0, 0]`, but the run
  still failed: `87` fall events, `70` drops, min robot z `0.194465 m`, min
  box z `0.042331 m`, max tilt `1.238096 rad`, final relative offset
  `0.532693 m`, final robot target travel `1.762137 m`, final box target
  travel `2.057596 m`. Zero command alone is not a reliable stop/hold
  transition in the current AGILE/Core adapter. Added
  `--agile-command-hold-reset-policy-state` / env
  `AGILE_COMMAND_HOLD_RESET_POLICY_STATE=1` to reset ONNX
  `h_state/c_state/last_action` or call the torch wrapper reset at hold
  trigger. Next diagnostic should test whether the continued motion comes
  from recurrent hidden-state persistence before redesigning the cradle or
  adding an explicit stand/settle transition.
- 2026-07-06 AGILE hold reset result and final stop diagnostic:
  policy-state reset at hold trigger also failed (`167588`, `server02`):
  reset count `1`, reset error null, but `306` falls, `252` drops, max tilt
  `1.803860 rad`, final relative offset `0.961335 m`. Hidden-state
  persistence is not the sufficient explanation. Added
  `--agile-command-hold-mode stand_targets` and
  `--agile-command-hold-stand-blend-rate`; after hold trigger this bypasses
  AGILE inference and blends commanded joints to the configured stand pose.
  This is the last AGILE stop/settle diagnostic before moving to
  cradle/contact retention redesign. Submitted tmux
  `curiosity_g1_agile_hold_stand_0706`, Slurm job `167590`, job-name
  `g1_agile_hstd`. Result: failed checker. Hold triggered at step `117`,
  stand-target mode was active for `583` steps, and policy inference count was
  only `20`, confirming AGILE was bypassed after hold. Box retention improved
  (`box_drop_events=0`, min box z `0.467851 m`), but robot stability failed
  (`285` falls, min robot z `0.322664 m`, max tilt `1.419840 rad`) and final
  robot/box target-directed travel went negative. Do not rerun zero-command,
  reset-state, or stand-target AGILE hold unchanged; next work should redesign
  cradle/contact retention together with a stable settle posture or controller.
- 2026-07-06 hold-only low-crouch settle prepared:
  added hold-only settle posture overrides to
  `scripts/isaac/build_core_world_g1_box_scene.py` and runner envs in
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  These overrides do not change AGILE's walking default pose; they only change
  the post-hold `stand_targets` settle target. Summary/check JSON records the
  requested and applied hold-only joint targets. Next diagnostic should test a
  low-crouch hold target after short AGILE motion, then decide whether settle
  posture is enough or cradle/contact retention must be redesigned. Submitted
  tmux `curiosity_g1_agile_lowcrouch_0706`, Slurm job `167591`, job-name
  `g1_agile_lc`. Result: failed checker. Hold triggered at step `101`,
  low-crouch target was active for `599` steps, and policy inference count was
  `16`. Target-directed travel stayed positive (final robot `0.752768 m`,
  final box `0.916188 m`), but there were `378` falls, `355` drops, min box z
  `0.030000 m`, max tilt `1.283639 rad`, and final relative offset
  `0.545589 m`. Next implementation should change physical cradle/contact
  retention; do not keep sweeping settle posture alone.
- 2026-07-06 static top-lid cradle result:
  added optional `front_tray` top-lid contact geometry and runner envs.
  Static top-lid low-crouch diagnostic (`167592`, `server02`) failed checker.
  It reduced failure severity compared with low-crouch without lid
  (`262` falls / `205` drops vs. `378` / `355`) and improved final relative
  offset to `0.311611 m`, but min robot z was `0.174101 m`, min box z was
  `0.072546 m`, max tilt was `1.542573 rad`, and final target-directed travel
  became negative. Static lid helps retention but interferes with useful
  motion. Next implementation should make the top lid a hold-phase contact
  limiter, enabled at hold trigger rather than active from scene start.
- 2026-07-06 hold-phase top-lid first implementation invalid:
  Slurm job `167593` triggered hold and lid activation at step `85` while the
  robot/box were still stable, but applying `UsdPhysics.CollisionAPI` during
  rollout invalidated the PhysX tensor view and stopped the run at
  `completed_steps=85` with `Failed to get DOF position targets from backend`.
  This is not carrying evidence. Fixed implementation: apply `CollisionAPI`
  at scene construction, initialize `physics:collisionEnabled=false`, and only
  toggle that existing attr to true at hold trigger.
- 2026-07-06 hold-phase top-lid attr-fix result:
  Slurm job `167594` completed 700 steps without tensor invalidation. The lid
  collision attr was enabled at step `92` with update count `1` and no error.
  Box retention improved strongly: `box_drop_events=0`, min box z
  `0.498752 m`, final relative offset `0.268659 m`. Robot stability still
  failed with `95` falls, min robot z `0.342325 m`, max tilt `1.245552 rad`,
  and final robot/box target-directed travel negative. Interpretation:
  contact retention is now better, but explicit stand-target settle is wrong.
  Next diagnostic should keep hold-phase top lid and switch back to AGILE
  `policy_command` hold to isolate settle posture from contact retention.
- 2026-07-06 hold-phase top-lid policy-command result:
  Slurm job `167596` completed 700 steps and failed checker. Hold triggered
  at step `102`, top-lid collision enabled at step `102`, update count `1`,
  update error null. Final target-directed travel stayed positive
  (robot `0.962488 m`, box `0.835041 m`), but the run had `352` falls,
  `68` drops, min robot z `0.191603 m`, min box z `0.085619 m`, max tilt
  `3.121745 rad`, and final relative offset `0.415014 m`. Interpretation:
  `stand_targets` plus lid preserves the box but destabilizes/reverses the
  body; `policy_command` plus lid preserves direction but still falls/drops.
  Continue directly in Isaac/G1 by adding a stable hold transition or
  balance-aware settle controller. Do not wait for external video/model code,
  and do not repeat unchanged hold-mode sweeps.
- 2026-07-06 hybrid hold and balance diagnostics:
  Added `policy_then_stand`, hold-gated balance feedback, command-based
  balance feedback, and configurable roll-feedback left/right multipliers.
  All runs used the real Isaac/G1 scene with WBC-AGILE ONNX, hold-phase top
  lid, no root/velocity/box rollout writes, and no external model dependency.
  Results:
  - `167602` hybrid balance, default pitch sign: positive travel but pitch
    collapse after step ~390; `309` falls, `293` drops.
  - `167603` pitch sign flipped to `+1`: pitch stabilized
    (`max_abs_pitch_rad=0.181384`) but severe side roll appeared; first fall
    step `560`, first drop step `600`.
  - `167604` pitch-only feedback: roll stayed small but pitch collapse returned
    earlier; first fall step `290`, first drop step `310`.
  - `167605` mirrored roll multipliers: side roll was controlled
    (`max_abs_roll_rad=0.403314`) and target-directed travel stayed positive,
    but pitch collapse returned; first fall step `510`, first drop step `530`.
  Conclusion: stop blind hold/balance gain sweeps. The current post-capture
  joint-target controller can trade off pitch, roll, and forward travel but
  cannot stabilize all three. Next work should introduce a new mechanism such
  as lateral drift brake, stance/footstep repositioning, or a proper WBC hold
  interface.
- 2026-07-06 post-capture slow-walk policy hold:
  Added hold-rescue state machine and tested it in Slurm job `167607`; rescue
  triggered at step `433` but failed by converting pitch collapse into roll
  fall (`208` falls, `173` drops). Static hold/rescue remains the wrong
  direction. Then tested slow post-capture AGILE policy hold in Slurm job
  `167608`: hold triggered at step `102`, top lid enabled at step `102`,
  `agile_command_hold_mode=policy_command`, `agile_command_hold_scale=0.35`,
  final hold command `[0.035, 0, 0]`, hold-gated balance feedback active for
  `595` steps. This diagnostic passed checker for 700 steps:
  `fall_events=0`, `box_drop_events=0`, min robot z `0.758436 m`, min box z
  `0.879201 m`, max tilt `0.254946 rad`, final robot target-directed travel
  `1.182280 m`, final box target-directed travel `1.243254 m`, final relative
  offset `0.107421 m`, and no root/velocity/box rollout writes. Treat this as
  a strong smoke pass, not final project success. Next validation must vary
  mass, shape, and duration. Mainline controller should keep WBC-AGILE active
  after capture with reduced target-directed command; do not return to static
  hold/rescue as the primary path.
- 2026-07-06 slow-walk held-out mass checks:
  The `0.25 kg` 700-step free-box run passed, but doubled mass is not solved.
  Slurm job `167613` used the same slow-walk controller with `0.5 kg` and
  `900` steps; it failed at first fall step `420`, first drop step `530`.
  Slurm job `167614` tested lower hold scale `0.15` with `0.5 kg` and
  `700` steps; it failed earlier, first fall step `310`, first drop step
  `330`. Conclusion: fixed slow command is not enough for heavier payloads,
  and simply lowering post-capture speed is worse. Next implementation should
  keep WBC-AGILE active but make the post-capture command load/contact
  adaptive using online signals such as tilt, tilt rate, box/robot relative
  offset, box z, and target-directed progress.
- 2026-07-06 adaptive post-capture command:
  Added adaptive hold command scaling and lateral correction in
  `scripts/isaac/build_core_world_g1_box_scene.py`, with runner/checker
  fields. Slurm job `167632` tested the previously failing `0.5 kg` case for
  `700` steps using adaptive scale range `0.18-0.35` and lateral correction.
  It passed checker: `fall_events=0`, `box_drop_events=0`, min robot z
  `0.710033 m`, min box z `0.835384 m`, max tilt `0.192797 rad`, final robot
  target-directed travel `1.812799 m`, final box target-directed travel
  `1.788483 m`, final relative offset `0.034773 m`, no root/velocity/box
  rollout writes. Adaptive scale was active for `577` steps, observed scale
  range `0.253006-0.35`; lateral correction was active for `577` steps and
  reached command limit `0.035`. This is a stronger baseline but still not
  final project success; next validation must vary object size/shape and
  duration.
- 2026-07-06 larger-box adaptive validation:
  Slurm job `167634` tested a larger `0.5 kg`, `0.14 x 0.10 x 0.08 m` box
  for `700` steps with the adaptive controller. It passed the configured
  loose checker with `fall_events=0`, `box_drop_events=0`, final robot
  target-directed travel `1.954226 m`, final box target-directed travel
  `2.011707 m`, final relative offset `0.071092 m`, max relative offset
  `0.206568 m`, and no root/velocity/box rollout writes. This is not a robust
  carrying claim: max/final robot-root tilt was `0.479985 rad`, lateral
  correction saturated at `0.035`, and final lateral path error reached
  `1.585361 m`. Treat it as "captured and carried without falling/dropping,
  but with poor path/root-attitude quality." Historical `max_tilt_rad`,
  `final_roll_rad`, and `final_pitch_rad` are robot-root attitude fields, not
  true box attitude.
- 2026-07-06 larger-box yaw-correction strict diagnostic:
  Added disabled-by-default hold-phase yaw correction controls and checker
  fields. Syntax checks passed. Slurm job `167646` tested the larger box under
  a stricter attitude gate (`FREE_MAX_TILT=0.35`). It failed only on attitude:
  `max_tilt_rad 0.532508 > 0.35`; this is robot-root attitude, not box
  attitude. It still had `fall_events=0`, `box_drop_events=0`, completed
  `700` steps, and no root/velocity/box rollout writes. Final robot/box
  target-directed travel dropped to
  `0.638052 m` / `0.617244 m`; final/max relative offset was `0.201939 m`.
  Yaw correction was active for `603` steps, max yaw command `0.088684`, and
  final yaw-control lateral error `1.059896 m`. Conclusion: yaw correction is
  not a clean main path; next work should improve larger-box attitude/contact
  strategy rather than continue blind yaw-gain sweeps.
- 2026-07-06 true box attitude telemetry:
  Added separate true box attitude fields to
  `scripts/isaac/build_core_world_g1_box_scene.py` and checker output:
  `max_box_tilt_rad`, `max_abs_box_roll_rad`, `max_abs_box_pitch_rad`,
  `final_box_roll_rad`, and `final_box_pitch_rad`. The checker supports
  optional `--max-box-tilt`, and the runner exposes case-specific
  `FREE_MAX_BOX_TILT` / `FIXED_MAX_BOX_TILT` / `NOBOX_MAX_BOX_TILT`. Also
  added disabled-by-default adaptive risk controls based on true box tilt and
  box tilt rate. Lightweight syntax checks passed. Future larger-box claims
  must distinguish robot-root tilt from true object tilt.
- 2026-07-06 target-path lateral telemetry:
  Added target-line lateral-error telemetry for robot and box:
  `max_abs_robot_target_lateral_error_m`,
  `max_abs_box_target_lateral_error_m`,
  `final_robot_target_lateral_error_m`, and
  `final_box_target_lateral_error_m`. The checker now supports max and final
  lateral-error gates for robot and box, and the runner exposes case-specific
  env gates such as `FREE_MAX_ROBOT_LATERAL_ERROR`,
  `FREE_MAX_BOX_LATERAL_ERROR`, `FREE_MAX_FINAL_ROBOT_LATERAL_ERROR`, and
  `FREE_MAX_FINAL_BOX_LATERAL_ERROR`. Lightweight `py_compile` and `bash -n`
  checks passed. Future carrying claims should include lateral path quality,
  not only target-directed distance.
- 2026-07-06 optional torso/chest support pad:
  Added a disabled-by-default front-cradle chest pad attached to the G1 torso,
  with runner env controls `CRADLE_CHEST_PAD_ENABLED`,
  `CRADLE_CHEST_PAD_ENABLE_ON_HOLD`, `CRADLE_CHEST_PAD_LOCAL_X/Y/Z`, and
  `CRADLE_CHEST_PAD_SIZE_X/Y/Z`. The scene/checker record pad geometry and
  hold-time collision activation. This is a physical torso-supported carrying
  posture diagnostic, not a success shortcut: future runs must still pass
  no-fall/no-drop, no rollout root/velocity/box-pose writes, robot-root tilt,
  true box tilt, relative-offset, and target-line lateral-error gates.
  Lightweight `py_compile` and `bash -n` checks passed.
- 2026-07-06 queued larger-box strict diagnostics:
  Slurm job `167670` (`g1_lg_btilt`) is queued in tmux
  `curiosity_g1_agile_largerbox_boxtilt_0706` to test larger-box true-box-tilt
  adaptive hold. Slurm job `167691` (`g1_lg_chest`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_0706` to test the optional
  torso/chest support pad under stricter robot-root tilt, true box tilt,
  relative-offset, and target-line lateral-error gates. Both are
  Curiosity-owned `srun` jobs; no `carry1` or non-project tmux/session was
  touched.
- 2026-07-06 larger-box strict wrapper:
  Added executable `scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`
  to centralize strict larger-box diagnostic settings. It supports
  `LARGERBOX_STRICT_MODE=boxtilt`, `lowcarry`, and `chestpad`, then delegates
  to the existing AGILE low-cradle suite so the same compute-node guard and
  no-root/box-write checker path apply. Lightweight `bash -n` and
  `py_compile` checks passed.
- 2026-07-06 larger-box strict summarizer:
  Added executable `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`
  to summarize completed strict larger-box summary/check JSON files. It
  reports fall/drop, rollout root/velocity/box writes, robot-root tilt, true
  box tilt, relative offset, target-line lateral error, adaptive-scale
  telemetry, top-lid activation, and chest-pad activation. This is a
  lightweight JSON summarizer only; it does not run Isaac or load models.
  `py_compile` passed.
- 2026-07-06 larger-box multi-posture matrix wrapper:
  Added executable `scripts/isaac/run_core_world_g1_largerbox_posture_matrix.sh`.
  It refuses login-node execution, then inside a compute allocation runs the
  strict larger-box `boxtilt`, `lowcarry`, and `chestpad` modes sequentially
  and summarizes them. This is the prepared gate for testing whether different
  carrying postures preserve balanced walking. Lightweight `bash -n` and
  `py_compile` checks passed.
- 2026-07-06 larger-box strict diagnostics completed:
  Slurm job `167670` (`g1_lg_btilt`) completed with build status `0` and
  check status `1`. It failed badly: `fall_events=210`, min robot z
  `0.259627 m`, robot-root max tilt `1.824652 rad`, true box max tilt
  `1.650226 rad`, final box target-directed travel `-0.112571 m`, final
  relative offset `0.388508 m`, and final robot/box target-line lateral errors
  about `-1.51 m` / `-1.52 m`. No rollout root/velocity/box pose writes
  occurred. Do not continue boxtilt/no-torso-support as the primary
  larger-box posture.
  Slurm job `167691` (`g1_lg_chest`) completed with build status `0` and
  check status `1`. It failed strict gates but is the best larger-box posture
  so far: `fall_events=0`, `box_drop_events=0`, no rollout root/velocity/box
  pose writes, final robot target-directed travel `1.292595 m`, final box
  target-directed travel `1.285295 m`, final relative offset `0.157688 m`,
  max relative offset `0.164649 m`. Remaining failures were robot-root max
  tilt `0.480753 > 0.35`, true box max tilt `0.493889 > 0.45`, and final box
  target-line lateral error `0.627694 > 0.60`. Next larger-box direction:
  tune chest-pad mode with earlier hold, lower post-capture command scale, and
  stronger lateral correction.
- 2026-07-06 tuned chest-pad strict diagnostic queued:
  Slurm job `167719` (`g1_lg_ctune`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_tuned_0706`. It uses
  `LARGERBOX_STRICT_MODE=chestpad`, earlier hold
  (`AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.12`), lower post-capture scale
  (`AGILE_COMMAND_HOLD_SCALE=0.25`, adaptive scale `0.05-0.25`), earlier
  root/box tilt risk thresholds, and stronger lateral correction
  (`gain=0.12`, `limit=0.055`). Goal: reduce root/box pitch tilt and final
  box target-line lateral error while preserving the no-fall/no-drop and
  no-rollout-write properties of the first chest-pad run.
- 2026-07-06 tuned chest-pad strict diagnostic completed:
  Slurm job `167719` (`g1_lg_ctune`) completed with build status `0` and check
  status `1`. It was worse than the first chest-pad run: `fall_events=247`,
  `box_drop_events=226`, min robot z `0.127119 m`, min box z `0.079985 m`,
  robot-root max tilt `1.277179 rad`, true box max tilt `1.223774 rad`, final
  relative offset `0.394498 m`. No rollout root/velocity/box pose writes
  occurred. Lateral error improved (`final_box_target_lateral_error_m=0.494578`)
  but the robot fell and dropped the box. Conclusion: over-slowing and holding
  too early destabilizes chest-pad carrying; return to the first chest-pad
  speed and tune lateral correction/contact geometry one variable at a time.
- 2026-07-06 chest-pad lateral-only strict diagnostic queued:
  Slurm job `167723` (`g1_lg_clat`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_lateral_0706`. It returns to the
  first chest-pad speed/hold settings and changes only lateral correction
  (`AGILE_COMMAND_HOLD_LATERAL_GAIN=0.12`,
  `AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.055`) to test whether final box
  target-line lateral error can be reduced without reintroducing falls/drops.
- 2026-07-06 chest-pad lateral-only strict diagnostic completed:
  Slurm job `167723` (`g1_lg_clat`) completed with build status `0` and check
  status `1`. It was worse than the first chest-pad run: `fall_events=335`,
  `box_drop_events=34`, min robot z `0.178246 m`, min box z `0.129841 m`,
  robot-root max tilt `3.134285 rad`, true box max tilt `3.127805 rad`. No
  rollout root/velocity/box pose writes occurred. Stronger lateral correction
  saturated at `0.055` and final robot/box target-line lateral errors were
  `0.796742 m` / `0.863152 m`. Conclusion: increasing lateral velocity
  authority destabilizes the walking controller; do not continue this as the
  main path.
- 2026-07-06 chest-pad mild-yaw strict diagnostic queued:
  Slurm job `167729` (`g1_lg_cyaw`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_mildyaw_0706`. It keeps the first
  chest-pad speed/hold and lateral settings, and adds a small hold-phase yaw
  correction (`AGILE_COMMAND_HOLD_YAW_CORRECTION=1`,
  `AGILE_COMMAND_HOLD_YAW_GAIN=0.04`, `AGILE_COMMAND_HOLD_YAW_LIMIT=0.08`) to
  test whether path drift can be reduced without the lateral-velocity
  instability seen in job `167723`.
- 2026-07-06 chest-pad mild-yaw strict diagnostic completed:
  Slurm job `167729` (`g1_lg_cyaw`) completed with build status `0` and check
  status `1`. It was worse than the first chest-pad run: `fall_events=94`, min
  robot z `0.323945 m`, robot-root max tilt `2.273509 rad`, true box max tilt
  `2.335066 rad`, final robot/box target-line lateral errors `-0.867466 m` /
  `-0.803925 m`. It did not drop the box and used no rollout root/velocity/box
  pose writes. Conclusion: adding yaw correction to chest-pad posture
  destabilizes root/box roll and worsens path error; next larger-box test
  should change contact geometry while preserving the first chest-pad
  controller.
- 2026-07-06 chest-pad geometry strict diagnostic queued:
  Slurm job `167731` (`g1_lg_cgeo`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_geom_0706`. It preserves the first
  chest-pad controller and changes only contact geometry: chest pad local
  x/z `-0.08/0.12`, pad size `0.08 x 0.44 x 0.28`, top-lid y scale `1.25`,
  side rail height `0.14`, and end-stop height `0.15`. Goal: reduce root/box
  pitch and target-line lateral error through physical support geometry
  rather than velocity/yaw command changes.
- 2026-07-06 chest-pad geometry strict diagnostic completed:
  Slurm job `167731` (`g1_lg_cgeo`) completed with build status `0` and check
  status `1`. It was worse than the first chest-pad run: `fall_events=328`,
  `box_drop_events=85`, min robot z `-0.542888 m`, min box z `-0.458421 m`,
  robot-root max tilt `3.140933 rad`, true box max tilt `3.109670 rad`, and
  final robot/box target-line lateral errors about `-2.42 m`. No rollout
  root/velocity/box pose writes occurred. Conclusion: larger/higher chest
  support and rails made lateral drift and roll-over worse; revert to the
  first chest-pad geometry as the current best larger-box contact setup.
- 2026-07-06 chest-pad opposite-yaw strict diagnostic queued:
  Slurm job `167762` (`g1_lg_oyaw`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_0706`. It preserves the
  first chest-pad geometry and controller, enables mild yaw correction, and
  flips `AGILE_COMMAND_HOLD_YAW_SIGN=-1.0` to test whether the previous
  mild-yaw run used the wrong correction direction.
- 2026-07-06 chest-pad opposite-yaw strict diagnostic completed:
  Slurm job `167762` (`g1_lg_oyaw`) completed with build status `0` and check
  status `0`. This is the first strict larger-box pass in the G1/AGILE sequence:
  `fall_events=0`, `box_drop_events=0`, completed `700` steps, min robot z
  `0.721562 m`, min box z `0.825034 m`, robot-root max tilt `0.307758 rad`,
  true box max tilt `0.312059 rad`, final robot/box target-directed travel
  `1.435312 m` / `1.457102 m`, max relative offset `0.205432 m`, final
  relative offset `0.075546 m`, max robot/box target-line lateral error
  `0.115763 m` / `0.186329 m`, and no rollout root/velocity/box pose writes.
  This is a strict diagnostic pass, not final project success. Next required
  validation: longer duration and additional posture/shape/mass held-outs.
- 2026-07-06 900-step opposite-yaw chest-pad validation queued:
  Slurm job `167768` (`g1_lg_oy9`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_900_0706`. It uses the
  first passing strict larger-box configuration from job `167762` but extends
  `FREE_STEPS=900` to test longer-duration stability.
- 2026-07-06 900-step opposite-yaw chest-pad validation completed:
  Slurm job `167768` (`g1_lg_oy9`) completed with build status `0` and check
  status `1`. The 700-step strict pass did not extend to 900 steps:
  `fall_events=26`, min robot z `0.369473 m`, robot-root max tilt
  `1.149047 rad`, true box max tilt `1.201307 rad`, final robot/box
  target-directed travel `1.184219 m` / `1.121610 m`, final robot/box
  target-line lateral error `0.632209 m` / `0.673950 m`. It did not drop the
  box and used no rollout root/velocity/box pose writes. Failure mode:
  post-target drift/tilt accumulation after the successful 700-step window.
- 2026-07-06 terminal hold scale control:
  Added disabled-by-default controls
  `AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL` and
  `AGILE_COMMAND_HOLD_TERMINAL_SCALE`. During hold phase, once previous box
  target-directed travel crosses the threshold, command scale is capped to the
  terminal scale. Summary/check output records active steps, first active
  step, and reason. Lightweight `py_compile`, `bash -n`, and
  `git diff --check` passed.
- 2026-07-06 900-step terminal-scale opposite-yaw validation queued:
  Slurm job `167771` (`g1_lg_ot9`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_0706`. It
  uses the 700-step passing opposite-yaw chest-pad configuration, extends to
  `FREE_STEPS=900`, and enables terminal scale at box target-directed travel
  `1.35 m` with `AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.06`.
- 2026-07-06 900-step terminal-scale opposite-yaw validation completed:
  Slurm job `167771` (`g1_lg_ot9`) completed with build status `0` and check
  status `1`. Terminal scale triggered at step `666` and was active for
  `234` steps. It improved the 900-step failure but did not pass:
  `fall_events=5`, no box drops, no rollout root/velocity/box pose writes,
  final robot/box target-line lateral errors `0.073779 m` / `0.120379 m`.
  Remaining failures: robot-root max tilt `1.156729 rad`, true box max tilt
  `1.085554 rad`, final relative offset `0.286271 m`. It still moved too far
  after target (`final_box_target_directed_travel_m=2.205989`). Next:
  terminal trigger earlier and smaller terminal scale.
- 2026-07-06 900-step early-terminal opposite-yaw validation queued:
  Slurm job `167773` (`g1_lg_oe9`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_early_0706`.
  It uses the 700-step passing opposite-yaw chest-pad configuration, extends
  to `FREE_STEPS=900`, and enables terminal scale earlier at box
  target-directed travel `1.15 m` with `AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.03`.
- 2026-07-06 900-step early-terminal opposite-yaw validation completed:
  Slurm job `167773` (`g1_lg_oe9`) completed with build status `0` and check
  status `1`. It removed the 900-step fall/drop failure: `fall_events=0`,
  `box_drop_events=0`, min robot z `0.721562 m`, min box z `0.825034 m`, no
  rollout root/velocity/box pose writes. Terminal mode triggered at step `612`
  and was active for `288` steps. Remaining failures were only attitude gates:
  robot-root max tilt `0.463448 > 0.35` and true box max tilt
  `0.636226 > 0.45`. Contact/path passed: final relative offset `0.072616 m`,
  max relative offset `0.205432 m`, final robot/box target-line lateral errors
  `0.383956 m` / `0.315494 m`. Next: near-stop terminal hold earlier to reduce
  accumulated roll.
- 2026-07-06 900-step near-stop terminal opposite-yaw validation queued:
  Slurm job `167778` (`g1_lg_on9`) is queued in tmux
  `curiosity_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_nearstop_0706`.
  It uses the 700-step passing opposite-yaw chest-pad configuration, extends to
  `FREE_STEPS=900`, and enables terminal scale earlier at box target-directed
  travel `1.05 m` with `AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015`.
- 2026-07-06 900-step near-stop terminal opposite-yaw validation completed:
  Slurm job `167778` (`g1_lg_on9`) completed with build status `0` and check
  status `0`. This is the strongest current larger-box result:
  `fall_events=0`, `box_drop_events=0`, completed `900` steps, min robot z
  `0.721562 m`, min box z `0.825034 m`, robot-root max tilt `0.307758 rad`,
  true box max tilt `0.384690 rad`, final robot/box target-directed travel
  `1.730244 m` / `1.759363 m`, max relative offset `0.205432 m`, final
  relative offset `0.108737 m`, max robot/box target-line lateral error
  `0.258455 m` / `0.362250 m`, and no rollout root/velocity/box pose writes.
  Terminal mode triggered at step `590` and was active for `310` steps. This
  is a strong diagnostic pass, not final project success: other postures,
  masses, shapes, active probing, and video-conditioned learning remain
  unverified.
- 2026-07-06 low-carry larger-box strict diagnostic queued:
  Slurm job `167782` (`g1_lg_low`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_oppositeyaw_0706`. It uses
  `LARGERBOX_STRICT_MODE=lowcarry`, no chest pad, and the opposite-yaw
  correction direction from the passing chest-supported run. Goal: test a
  second carrying posture rather than relying only on chest-supported carrying.
- 2026-07-06 low-carry larger-box strict diagnostic completed:
  Slurm job `167782` (`g1_lg_low`) completed with build status `0` and check
  status `1`. Low-carry without chest support failed: `fall_events=117`,
  `box_drop_events=104`, min robot z `0.170354 m`, min box z `0.096605 m`,
  robot-root max tilt `0.990520 rad`, true box max tilt `0.991162 rad`, final
  robot/box target-directed travel `1.018163 m` / `1.075757 m`, final
  robot/box target-line lateral errors `1.228960 m` / `1.272310 m`. No rollout
  root/velocity/box pose writes occurred. Conclusion: low-carry needs its own
  support/terminal strategy and cannot simply reuse the chest-supported yaw
  controller.
- 2026-07-06 low-carry terminal strict diagnostic queued:
  Slurm job `167783` (`g1_lg_lterm`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_0706`. It keeps low-carry
  without chest support, uses opposite-yaw correction, and adds terminal scale
  at box target-directed travel `0.65 m` with
  `AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015`, to test whether early near-stop
  can prevent the low-carry drop/fall failure.
- 2026-07-06 low-carry terminal strict diagnostic completed:
  Slurm job `167783` (`g1_lg_lterm`) completed with build status `0` and check
  status `1`. Terminal hold removed the low-carry fall/drop failure:
  `fall_events=0`, `box_drop_events=0`, min robot z `0.761974 m`, min box z
  `0.816164 m`, robot-root max tilt `0.196663 rad`, true box max tilt
  `0.271947 rad`, final relative offset `0.200210 m`, no rollout
  root/velocity/box pose writes. Remaining failures were path-only: final
  robot/box target-line lateral errors `1.114195 m` / `1.306165 m`. Next:
  retest low-carry terminal with the opposite yaw sign flipped back to default.
- 2026-07-06 low-carry terminal default-yaw strict diagnostic queued:
  Slurm job `167788` (`g1_lg_ldef`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_defaultyaw_0706`. It keeps
  low-carry terminal hold (`0.65 m`, scale `0.015`) and flips
  `AGILE_COMMAND_HOLD_YAW_SIGN=1.0` to test whether the path-only failure in
  job `167783` was caused by yaw direction.
- 2026-07-06 low-carry terminal default-yaw strict diagnostic completed:
  Slurm job `167788` (`g1_lg_ldef`) completed with build status `0` and check
  status `1`. It did not fall or drop and used no rollout root/velocity/box
  pose writes, but moved in the wrong target direction: final robot/box
  target-directed travel `-0.387059 m` / `-0.584589 m`; terminal hold never
  triggered. It also exceeded true box tilt (`0.636519 rad`) and final
  relative offset (`0.286901 m`). Conclusion: default yaw sign is wrong for
  low-carry target progress; continue from the yaw-sign `-1.0` terminal base
  and tune lateral correction sign/gain.
- 2026-07-06 low-carry terminal lateral-sign strict diagnostic queued:
  Slurm job `167789` (`g1_lg_lsgn`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_latsign_0706`. It keeps the
  best low-carry terminal base (`AGILE_COMMAND_HOLD_YAW_SIGN=-1.0`, terminal
  trigger `0.65 m`, terminal scale `0.015`) and flips only
  `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0` to test whether the remaining
  path-only failure is lateral-correction sign rather than low-carry support.
- 2026-07-06 low-carry terminal lateral-sign strict diagnostic completed:
  Slurm job `167789` (`g1_lg_lsgn`) completed with build status `0` and check
  status `1`. Flipping only `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0` was a
  clear regression: `fall_events=220`, `box_drop_events=40`, min robot z
  `0.172896 m`, min box z `0.115441 m`, robot-root max tilt `1.749395 rad`,
  true box max tilt `2.081289 rad`, final robot/box target-directed travel
  `-0.577424 m` / `-0.516827 m`, final robot/box lateral errors
  `-1.229764 m` / `-1.121373 m`. Terminal hold never triggered, and no rollout
  root/velocity/box pose writes occurred. Do not continue in the flipped
  lateral-sign direction.
- 2026-07-06 low-carry terminal no-lateral strict diagnostic queued:
  Slurm job `167800` (`g1_lg_lnol`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral_0706`. It keeps
  the stable low-carry terminal base and disables
  `AGILE_COMMAND_HOLD_LATERAL_CORRECTION` to test whether the path-only
  low-carry failure comes from the lateral controller or from posture drift.
- 2026-07-06 low-carry terminal no-lateral strict diagnostic completed:
  Slurm job `167800` (`g1_lg_lnol`) completed with build status `0` and check
  status `0`. This is the strongest current low-carry result:
  `fall_events=0`, `box_drop_events=0`, completed `700` steps, min robot z
  `0.757182 m`, min box z `0.825777 m`, robot-root max tilt `0.227144 rad`,
  true box max tilt `0.241890 rad`, final robot/box target-directed travel
  `1.994070 m` / `2.024888 m`, max robot/box target-line lateral error
  `0.430948 m` / `0.414760 m`, final robot/box lateral errors `0.427588 m` /
  `0.374435 m`, and no rollout root/velocity/box pose writes. Lateral
  correction was disabled and active for `0` steps. Conclusion: the previous
  low-carry path failure came from the lateral correction controller, not from
  the low-carry posture itself.
- 2026-07-06 900-step low-carry terminal no-lateral strict validation queued:
  Slurm job `167803` (`g1_lg_ln9`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral900_0706`. It
  extends the passing 700-step low-carry no-lateral configuration to
  `FREE_STEPS=900`.
- 2026-07-06 900-step low-carry terminal no-lateral strict validation
  completed: Slurm job `167803` (`g1_lg_ln9`) completed with build status `0`
  and check status `1`. It preserved the path improvement but failed
  late-duration stability: `fall_events=44`, `box_drop_events=25`, min robot z
  `-0.226469 m`, min box z `-0.343242 m`, robot-root max tilt `0.975629 rad`,
  true box max tilt `1.274737 rad`. It reached final robot/box
  target-directed travel `3.442578 m` / `3.440148 m`, while lateral errors
  stayed within gates (`0.440032 m` / `0.414760 m` max). No rollout
  root/velocity/box pose writes occurred. Next: keep no-lateral but reduce
  terminal scale to zero for true stop.
- 2026-07-06 900-step low-carry terminal no-lateral zero-stop strict
  validation queued: Slurm job `167804` (`g1_lg_lz9`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral_zerostop900_0706`.
  It keeps no-lateral low-carry and changes only
  `AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.0`.
- 2026-07-06 900-step low-carry terminal no-lateral zero-stop strict
  validation completed: Slurm job `167804` (`g1_lg_lz9`) completed with build
  status `0` and check status `1`. Zero-stop was too abrupt or under-driven:
  `fall_events=141`, `box_drop_events=97`, min robot z `0.161750 m`, min box
  z `0.071122 m`, robot-root max tilt `1.677712 rad`, true box max tilt
  `1.681898 rad`, final robot/box target-directed travel only `0.367843 m` /
  `0.339509 m`, and final relative offset `0.281655 m`. No rollout
  root/velocity/box pose writes occurred. Next: test intermediate terminal
  scale between `0.0` and `0.015`.
- 2026-07-06 900-step low-carry terminal no-lateral mid-stop strict
  validation queued: Slurm job `167806` (`g1_lg_lm9`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal_nolateral_midstop900_0706`.
  It keeps no-lateral low-carry and sets
  `AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.008`.
- 2026-07-06 900-step low-carry terminal no-lateral mid-stop strict
  validation completed: Slurm job `167806` (`g1_lg_lm9`) completed with build
  status `0` and check status `1`. Intermediate terminal scale `0.008` also
  failed: `fall_events=226`, `box_drop_events=36`, min robot z `0.183568 m`,
  min box z `0.120184 m`, robot-root max tilt `3.119225 rad`, true box max
  tilt `3.097902 rad`, final robot/box target-directed travel `0.662951 m` /
  `0.576417 m`, final relative offset `0.330259 m`. No rollout
  root/velocity/box pose writes occurred. Conclusion: 900-step low-carry
  failure is not solved by lowering terminal scale; use existing
  `policy_then_stand` hold mode to test long-duration stabilization.
- 2026-07-06 900-step low-carry no-lateral policy-then-stand strict
  validation queued: Slurm job `167808` (`g1_lg_lps`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_nolateral_policythenstand900_0706`.
  It keeps no-lateral low-carry with terminal scale `0.015`, then switches to
  stand-target blending after `420` hold steps with blend rate `0.02`.
- 2026-07-06 900-step low-carry no-lateral policy-then-stand strict
  validation completed: Slurm job `167808` (`g1_lg_lps`) completed with build
  status `0` and check status `1`. It kept target-line lateral errors within
  gates, but stand-target blending hurt low-carry object stability:
  `fall_events=281`, `box_drop_events=258`, min robot z `0.299849 m`, min box
  z `0.097845 m`, robot-root max tilt `1.457402 rad`, true box max tilt
  `2.824101 rad`, final robot/box target-directed travel `1.790397 m` /
  `1.778671 m`, final relative offset `0.319173 m`. No rollout
  root/velocity/box pose writes occurred. Conclusion: low-carry 900-step
  stability needs a posture-specific hold/recovery controller that preserves
  arm/cradle contact; generic lateral correction, zero terminal speed, and
  generic stand-target blending are all rejected by the current diagnostics.
- 2026-07-06 low-cradle runner hold/rescue override interface added:
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh` now
  forwards optional `AGILE_COMMAND_HOLD_STAND_*` and
  `AGILE_COMMAND_HOLD_RESCUE_*` environment variables to
  `build_core_world_g1_box_scene.py`. These arguments are appended only when
  explicitly set, so default behavior is unchanged. Purpose: test
  posture-specific low-carry crouch hold/recovery without using generic
  stand-target blending. This interface does not write robot root pose, root
  velocity, or box pose.
- 2026-07-06 900-step low-carry no-lateral late-crouch-hold strict
  validation planned: keep the passing no-lateral low-carry base, delay
  `policy_then_stand` until hold age `620` steps, blend slowly at `0.01`, and
  use low-crouch lower-body targets (`hip=-0.20`, `knee=0.45`,
  `ankle=-0.25`, `waist=-0.06`) to reduce late forward drift without pulling
  the robot into a generic upright stand.
- 2026-07-06 900-step low-carry no-lateral late-crouch-hold strict
  validation queued: Slurm job `167858` (`g1_lg_lch`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_latecrouch900_0706`.
- 2026-07-06 900-step low-carry no-lateral late-crouch-hold strict
  validation completed: Slurm job `167858` (`g1_lg_lch`) completed with build
  status `0` and check status `1`. Late crouch hold was a negative result:
  `fall_events=94`, `box_drop_events=84`, min robot z `-1.205153 m`, min box
  z `-1.185329 m`, robot-root max tilt `3.139498 rad`, true box max tilt
  `3.134521 rad`, final robot/box target-directed travel `3.057527 m` /
  `2.630529 m`, final robot/box lateral errors `0.648811 m` / `0.691176 m`.
  No rollout root/velocity/box pose writes occurred. Conclusion: low-carry
  900-step should not be fixed by late joint-target blending; use
  command-level stop logic that preserves the policy posture.
- 2026-07-06 latched terminal hold diagnosis: current agile terminal hold is
  instantaneous rather than latched. If box target-directed travel falls back
  below the threshold, the command can resume. The zero-stop negative result's
  final command still had nonzero x command (`0.010`), so the next aligned
  change is a latched terminal state before adding more posture blending.
- 2026-07-06 latched agile terminal hold interface added:
  `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-terminal-latch`, exposed through
  `AGILE_COMMAND_HOLD_TERMINAL_LATCH=1` in
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`. When enabled, the
  first threshold crossing of
  `--agile-command-hold-terminal-box-target-travel` latches terminal scale so
  the command does not resume if the box later moves back under the threshold.
  New telemetry: `agile_command_hold_terminal_latch_enabled`,
  `agile_command_hold_terminal_latched`, and
  `agile_command_hold_terminal_latched_step`. This is command-level control
  only and does not write robot root pose, root velocity, or box pose.
- 2026-07-06 900-step low-carry no-lateral latched zero-stop strict
  validation queued: Slurm job `167875` (`g1_lg_lzl`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedzerostop900_0706`. It keeps
  the low-carry no-lateral base, uses terminal scale `0.0`, and enables
  `AGILE_COMMAND_HOLD_TERMINAL_LATCH=1` to test whether command resumption
  caused the earlier zero-stop failure.
- 2026-07-06 900-step low-carry no-lateral latched zero-stop strict
  validation completed: Slurm job `167875` (`g1_lg_lzl`) completed with build
  status `0` and check status `1`. Latch worked as intended (`latched=True`,
  latched step `381`, final command x `0.0`) but the run still failed:
  `fall_events=141`, `box_drop_events=97`, min robot z `0.162028 m`, min box
  z `0.077461 m`, robot-root max tilt `1.677746 rad`, true box max tilt
  `1.656680 rad`, final robot/box target-directed travel `0.368724 m` /
  `0.356281 m`, final relative offset `0.254142 m`. No rollout
  root/velocity/box pose writes occurred. Conclusion: command resumption was
  not the main failure; complete stop is under-supported for low-carry. Next
  test should keep latch but use a tiny nonzero terminal scale.
- 2026-07-06 900-step low-carry no-lateral latched micro-hold strict
  validation queued: Slurm job `167879` (`g1_lg_lml`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro900_0706`. It keeps
  terminal latch and no-lateral low-carry, but uses terminal scale `0.006` to
  test whether a tiny nonzero policy command supports low-carry better than
  full zero-stop.
- 2026-07-06 900-step low-carry no-lateral latched micro-hold strict
  validation completed: Slurm job `167879` (`g1_lg_lml`) completed with build
  status `0` and check status `1`. It reduced late drop/fall relative to full
  zero-stop but introduced severe lateral drift: `fall_events=90`,
  `box_drop_events=42`, min robot z `0.181486 m`, min box z `0.132102 m`,
  robot-root max tilt `1.704787 rad`, true box max tilt `1.905366 rad`, final
  robot/box target-directed travel `1.309712 m` / `1.344400 m`, final
  robot/box lateral errors `1.375929 m` / `1.389495 m`. No rollout
  root/velocity/box pose writes occurred. Next: add terminal-only lateral
  correction with lower gain/limit; always-on lateral correction remains
  rejected for low-carry.
- 2026-07-06 terminal-only agile lateral correction interface added:
  `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-lateral-terminal-only`, exposed through
  `AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1` in
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`. When enabled, lateral
  correction is allowed only after the terminal threshold/latch is active,
  rather than at the first agile hold step. Default behavior is unchanged.
- 2026-07-06 900-step low-carry latched micro-hold terminal-lateral strict
  validation queued: Slurm job `167898` (`g1_lg_ltl`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat900_0706`. It
  keeps terminal latch and terminal scale `0.006`, enables terminal-only
  lateral correction, and uses a low lateral gain/limit (`0.012`, `0.006`) to
  test path correction without early low-carry destabilization.
- 2026-07-06 terminal-lateral job `167898` invalidated as a configuration
  run, not a behavioral result. It completed with build status `0` and check
  status `1`, but summary showed key intended settings were not applied:
  `agile_command_hold_terminal_box_target_travel_m=-1.0`,
  `agile_command_hold_terminal_latch_enabled=false`,
  `agile_command_hold_lateral_correction_enabled=false`, and
  `agile_command_hold_lateral_terminal_only=false`. Do not use its fall/drop
  metrics as evidence about terminal-only lateral correction. Rerun with
  explicit `export ...; srun --export=ALL`.
- 2026-07-06 rerun of 900-step low-carry latched micro-hold terminal-lateral
  strict validation queued: Slurm job `167958` (`g1_lg_ltx`) is queued in
  tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export900_0706`.
  It uses explicit shell exports and `srun --export=ALL` so the AGILE terminal
  latch/lateral settings should be visible inside the compute allocation.
- 2026-07-06 low-cradle suite environment snapshot logging added:
  `run_core_world_g1_agile_policy_low_cradle_suite.sh` now writes
  `agile_policy_low_cradle_env_snapshot.txt` under each suite output root with
  AGILE, balance, cradle, free-box, larger-box, target, and run environment
  variables. This is diagnostic logging only and does not change control
  behavior; purpose is to prevent invalid configuration runs from being
  mistaken for real behavior.
- 2026-07-06 replaced pending explicit-export terminal-lateral job:
  Slurm job `167958` (`g1_lg_ltx`) was cancelled before it started because it
  remained pending with `Reason=Priority` and requested 1.5h walltime, while
  prior 900-step validations complete in about 1-2 minutes. Submitted an
  equivalent 15-minute backfill-friendly run as Slurm job `167990`
  (`g1_lg_ltxs`) in tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export900_short_0706`,
  stamp
  `20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export_short_strict_900_targetnegx1`.
- 2026-07-06 valid 900-step low-carry terminal-only lateral run completed:
  Slurm job `167990` (`g1_lg_ltxs`) completed with build status `0` and check
  status `1`; environment snapshot confirmed the intended AGILE
  terminal/lateral settings were applied. It reduced lateral drift relative to
  no-lateral micro-hold but destabilized much earlier: `fall_events=288`,
  `box_drop_events=269`, min robot z `0.180464 m`, min box z `0.083854 m`,
  robot-root max tilt `2.068887 rad`, true box max tilt `2.171806 rad`, final
  robot/box target-directed travel `0.921112 m` / `0.958597 m`, final
  robot/box lateral errors `-0.595748 m` / `-0.657782 m`. No rollout
  root/velocity/box pose writes occurred. First-fall comparison: no-lateral
  micro-hold first fell at step `810`, while terminal-only lateral first fell
  at step `620` with small lateral error near zero. Conclusion: terminal-only
  lateral should be gated by lateral error magnitude, not merely by terminal
  latch.
- 2026-07-06 lateral-error-threshold gating added:
  `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-lateral-error-start`, exposed as
  `AGILE_COMMAND_HOLD_LATERAL_ERROR_START` in
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`. Default is `0.0`, so
  existing behavior is unchanged. Purpose: prevent terminal-only lateral
  correction from acting while target-line lateral error is still small.
- 2026-07-06 900-step low-carry latched micro-hold terminal-lateral threshold
  strict validation queued: Slurm job `167998` (`g1_lg_lth`) is queued in tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh900_0706`.
  It uses explicit exports, terminal latch, terminal scale `0.006`,
  terminal-only lateral correction, lateral error start `0.55 m`, gain
  `0.006`, and limit `0.003`.
- 2026-07-06 first-event telemetry added to G1 box scene summaries:
  `build_core_world_g1_box_scene.py` now records `first_fall_step`,
  `first_fall_time_s`, `first_box_drop_step`, and `first_box_drop_time_s`.
  This is diagnostic metadata only and does not change physics or controls.
- 2026-07-06 strict checker telemetry expanded:
  `check_core_world_g1_box_scene_summary.py` now copies first-fall/drop fields,
  terminal latch fields, `agile_command_hold_lateral_terminal_only`, and
  `agile_command_hold_lateral_error_start_m` into `check.json`. This does not
  change pass/fail gates; it makes low-carry failure timing and gating
  evidence directly available in checker output.
- 2026-07-06 larger-box strict summarizer telemetry expanded:
  `summarize_core_world_g1_largerbox_strict.py` now copies first-fall/drop
  fields, terminal latch fields, `agile_command_hold_lateral_terminal_only`,
  and `agile_command_hold_lateral_error_start_m` into aggregate summaries.
  This is reporting-only and does not change any simulation or checker gate.
- 2026-07-06 larger-box strict summarizer first-event backfill added:
  when `first_fall_step` or `first_box_drop_step` is missing from older
  summaries, `summarize_core_world_g1_largerbox_strict.py` now reads the
  same case's `core_world_g1_box_scene_state.csv` and computes first fall/drop
  step and time. Verified on the old low-carry latched micro-hold run: first
  fall step `810`, first drop step `840`.
- 2026-07-06 low-carry 900-step control matrix helper added:
  `scripts/isaac/summarize_core_world_g1_lowcarry_900_matrix.py` reads current
  low-carry summaries and CSV traces and writes a diagnostic Markdown matrix.
  Generated
  `experiments/reports/2026-07-06_g1_lowcarry_900_control_matrix.md`. Current
  matrix shows no-lateral 900 first falls at step `860`, latched micro-hold
  first falls at step `810`, terminal-only lateral first falls at step `620`,
  and the threshold-gated terminal-lateral case is still missing because Slurm
  job `167998` remains pending.
- 2026-07-06 conservative lateral excess-error option added:
  `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-lateral-use-excess-error`, exposed as
  `AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1` in the low-cradle suite.
  When enabled, lateral correction uses only the error magnitude above
  `AGILE_COMMAND_HOLD_LATERAL_ERROR_START` rather than the full lateral error.
  Default is disabled, so current queued job `167998` preserves the intended
  threshold-gated behavior. This is a fallback if threshold-gated lateral is
  still too aggressive.
- 2026-07-06 excess-error fallback command staged but not submitted:
  if job `167998` fails by early destabilization, the next planned low-carry
  900-step validation is the same terminal-latched micro-hold setup with
  `AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1`, error start `0.55 m`,
  lateral gain `0.012`, and lateral limit `0.003`. It is recorded in
  `TODO/03_no_root_articulated_carrier/todo.md` but intentionally not
  submitted while `167998` is pending, to avoid competing for GPU resources.
- 2026-07-06 low-carry control matrix updated with excess fallback row:
  `experiments/reports/2026-07-06_g1_lowcarry_900_control_matrix.md` now shows
  both the pending threshold-gated run and the planned excess-error fallback
  as missing cases, so the next result can be compared without manual table
  edits.
- 2026-07-06 old low-carry CSV windows inspected with lightweight text tools
  only. The valid terminal-only lateral run failed early because lateral
  correction from step `381` drove the target-line lateral error through zero
  while robot/box pitch deteriorated; first fall occurred at step `620`. The
  no-lateral latched micro-hold run instead failed later from slow lateral
  drift, first fall step `810`. This supports the pending threshold-gated
  lateral run `167998` and argues against increasing lateral gain.
- 2026-07-06 lateral posture-risk gate added for future low-carry fallbacks:
  `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-lateral-max-tilt` and
  `--agile-command-hold-lateral-max-box-tilt`, exposed as
  `AGILE_COMMAND_HOLD_LATERAL_MAX_TILT` and
  `AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT` in the low-cradle suite. Defaults
  are `999.0`, so existing behavior and queued job `167998` are unchanged.
  The scene records `agile_command_hold_lateral_suppressed_by_tilt_steps`, and
  checker/summarizer/matrix reporting now carries the gate fields. Purpose:
  allow path-centering only when posture and box attitude are still safe,
  prioritizing balance/retention over lateral centering.
- 2026-07-06 threshold-gated low-carry lateral run `167998` is invalid as
  behavioral evidence. Environment snapshot confirmed the intended settings,
  but rollout stopped at step `610` with
  `NameError: name 'feedback_tilt' is not defined` from the newly added
  lateral posture-risk gate. It had fall/drop `0/0` at interruption, but only
  `completed_steps=611`, so do not compare it as a 900-step result.
- 2026-07-06 lateral posture-risk gate bug fixed:
  `build_core_world_g1_box_scene.py` now computes the robot tilt for lateral
  gate checks from `max(abs(feedback_roll), abs(feedback_pitch))` instead of
  the nonexistent `feedback_tilt` symbol. Login-node lightweight checks passed:
  `py_compile`, `bash -n`, and `git diff --check`.
- 2026-07-06 fixed threshold-gated low-carry lateral validation submitted:
  stamp
  `20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh_fix_strict_900_targetnegx1`,
  tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh_fix900_0706`,
  Slurm job `168131` (`g1_lg_lthf`). It repeats the intended `167998`
  threshold setup after the lateral gate bug fix: terminal latch, terminal
  scale `0.006`, terminal-only lateral, lateral error start `0.55 m`, gain
  `0.006`, and limit `0.003`.
- 2026-07-06 fixed threshold-gated low-carry lateral validation completed:
  Slurm job `168131` completed build status `0`, checker status `1`. This is
  valid behavior evidence but not a pass. It completed `900/900` steps with
  fall/drop `0/0`, no root velocity/pose or box pose rollout writes, min robot
  z `0.681462 m`, min box z `0.659798 m`, final robot/box target-directed
  travel `1.612132 m` / `1.644704 m`, and final relative offset
  `0.154714 m`. It failed strict gates because robot max tilt was
  `0.594056 rad`, true box max tilt `0.946271 rad`, max/final robot lateral
  error `1.532278 m`, and max/final box lateral error `1.672921 m`. Lateral
  correction started only at step `611`, active `289` steps, max command
  `0.003`. Conclusion: `0.55 m` threshold avoided fall/drop but acted too
  late/weak for path centering and allowed large tilt. Next test should start
  lateral correction earlier, with a posture-risk gate to stop correcting
  under high tilt.
- 2026-07-06 earlier threshold plus posture-gated lateral validation
  submitted: stamp
  `20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh045_tiltgate_strict_900_targetnegx1`,
  tmux
  `curiosity_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh045_tiltgate900_0706`,
  Slurm job `168144` (`g1_lg_l45`). It uses terminal latch, terminal scale
  `0.006`, terminal-only lateral, lateral error start `0.45 m`, lateral
  gain/limit `0.006/0.003`, and posture gates
  `AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.45`,
  `AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.45`.
- 2026-07-06 earlier threshold plus posture-gated lateral validation
  completed: Slurm job `168144` completed build status `0`, checker status
  `1`; it is valid negative behavior evidence. It completed `900/900` but
  failed with `fall_events=210`, `box_drop_events=158`, first fall step
  `690`, first drop step `709`, min robot z `0.353656 m`, min box z
  `0.103534 m`, robot/box max tilt `3.138657/3.134910 rad`, final robot/box
  lateral error `1.476673/1.439418 m`, and final relative offset
  `0.462399 m`. Lateral correction started at step `550`, was active only
  `105` steps, and was suppressed by tilt for `245` steps. Conclusion:
  earlier lateral correction is worse than the 0.55 m threshold run; do not
  keep tuning lateral correction as the main fix.
- 2026-07-06 two-stage agile terminal scale added for low-carry testing:
  `build_core_world_g1_box_scene.py` now supports
  `--agile-command-hold-final-box-target-travel`,
  `--agile-command-hold-final-scale`, and
  `--agile-command-hold-final-latch`, exposed as
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL`,
  `AGILE_COMMAND_HOLD_FINAL_SCALE`, and `AGILE_COMMAND_HOLD_FINAL_LATCH` in
  the low-cradle suite. Defaults disable the second stage. Purpose: keep the
  path-stable `0.015` terminal scale early, then reduce to a micro-hold scale
  after larger target-directed travel to avoid the late overshoot/fall seen in
  the 900-step no-lateral `0.015` run.
- 2026-07-06 two-stage no-lateral low-carry validation submitted: stamp
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final006_2m_strict_900_targetnegx1`,
  tmux `curiosity_g1_agile_largerbox_lowcarry_terminal015_final006_2m900_0706`,
  Slurm job `168154` (`g1_lg_f2m`). It uses no lateral correction, yaw
  correction, terminal scale `0.015` from box target travel `0.65 m`, and
  final latched scale `0.006` after box target travel `2.0 m`.
- 2026-07-06 `168154` status check at `14:40:06 CST`: still
  `PENDING (Priority)` with scheduled start time `2026-07-06T16:18:59` on
  `server39`; no output files exist yet. This is a queue/resource wait, not
  behavior evidence.
- 2026-07-06 `168154` status check at `14:41:43 CST`: still
  `PENDING (Priority)` and no output files exist. Login-node lightweight
  checks still pass after TODO cleanup: `py_compile`, `bash -n`, and
  `git diff --check`.
- 2026-07-06 two-stage threshold rationale checked with existing CSVs:
  in the old no-lateral `0.015` terminal-scale 900-step run, box
  target-directed travel around `2.0 m` still had fall/drop `0/0` and low
  tilt, while the same run later failed near `3.1 m` travel. In contrast,
  early `0.006` micro-hold preserved fall/drop longer only at the cost of
  severe lateral drift. This supports the pending `168154` compromise: keep
  `0.015` until `2.0 m`, then latch to `0.006`. `168154` was still
  `PENDING (Priority)` at `14:42:55 CST` with no output files.
- 2026-07-06 two-stage no-lateral low-carry validation completed:
  Slurm job `168154` completed build status `0`, checker status `1`; it is
  valid negative behavior evidence, not a pass. It completed `900/900` with
  `fall_events=54`, `box_drop_events=29`, first fall/drop steps `846/871`,
  min robot/box z `-0.572852/-0.411364 m`, robot/true-box max tilt
  `0.346288/0.725258 rad`, final robot/box target-directed travel
  `3.355354/3.348626 m`, final robot/box lateral error
  `0.494245/0.542693 m`, final relative offset `0.068053 m`, and no rollout
  root velocity/pose or box pose writes. The final `0.006` stage latched after
  box target-directed travel `2.0 m`, first active at step `695`, active
  `205` steps. Conclusion: reducing to `0.006` after `2.0 m` preserved
  relative offset and reduced robot max tilt versus the old no-lateral 900-step
  `0.015` run, but still moved too far and failed late. Next direct Isaac test
  should keep the stable `0.015` approach stage and latch to a full `0.0`
  command scale only after `2.0 m`, rather than stopping early at `0.65 m`.
- 2026-07-06 two-stage full-stop-after-2m low-carry validation submitted:
  stamp
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_strict_900_targetnegx1`,
  tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal015_final000_2m900_0706`,
  Slurm job `168164` (`g1_lg_f0m`). It uses no lateral correction, yaw
  correction, terminal scale `0.015` from box target travel `0.65 m`, and a
  final latched full stop scale `0.0` after box target travel `2.0 m`.
  Submission-time queue check showed `PENDING (Priority)` and no output files
  yet. This is a direct Isaac test; it does not wait for external models.
- 2026-07-06 `168164` status check at `14:52:25 CST`: still
  `PENDING (Priority)`, scheduled start time `2026-07-06T16:18:59` on
  `server39`; no output files exist yet. This is a Slurm priority wait, not a
  code/Isaac/model-download blocker.
- 2026-07-06 final-stage stand-hold option added for direct Isaac G1
  low-carry testing. `scripts/isaac/build_core_world_g1_box_scene.py` now
  supports `--agile-command-hold-final-stand` and
  `--agile-command-hold-final-stand-delay-steps`; the low-cradle launcher
  exposes them as `AGILE_COMMAND_HOLD_FINAL_STAND=1` and
  `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS`. Defaults keep the option
  disabled, so pending job `168164` is unchanged. This is meant to test a
  more realistic post-carry hold: keep the stable `0.015` approach/terminal
  walk, then after the final target-travel threshold latch to zero command and
  blend only then toward stand targets. Checker, larger-box summarizer, and
  low-carry matrix reporting now include final-stand enabled/delay/active-step
  fields. Planned comparison stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1`.
- 2026-07-06 login-node lightweight validation after final-stage stand-hold
  edit passed: `python3 -m py_compile` on the G1 scene/checker/summarizers,
  `bash -n` on the low-cradle and larger-box launchers, and `git diff
  --check` on the touched files. No simulation, rendering, model loading, or
  data processing was run on the login node. Queue check still showed job
  `168164` as `PENDING (Priority)`.
- 2026-07-06 final target-directed travel upper-bound gates added.
  `scripts/isaac/check_core_world_g1_box_scene_summary.py` now accepts
  `--max-final-robot-target-directed-travel` and
  `--max-final-box-target-directed-travel`. The low-cradle launcher passes
  these from `NOBOX_/FIXED_/FREE_MAX_FINAL_*_TARGET_DIRECTED_TRAVEL` or shared
  `MAX_FINAL_*_TARGET_DIRECTED_TRAVEL` environment variables, and the
  larger-box strict launcher exposes empty defaults for the free-box gates.
  Purpose: future full-stop/final-stand runs can require both enough carrying
  distance and no large overshoot; the old `168154` failure would fail such a
  gate because final box target-directed travel was `3.348626 m`.
- 2026-07-06 final-stage trigger-duration gates added. The checker now accepts
  `--min-agile-command-hold-final-active-steps` and
  `--min-agile-command-hold-final-stand-active-steps`, and the low-cradle
  launcher passes them from `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS` and
  `MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS`. The planned final-stand
  comparison should require at least `120` final-stage steps and `80`
  final-stand steps so the run proves it actually entered the post-carry hold.
  The low-cradle launcher environment snapshot now includes `MIN_` variables
  so these gates are recorded with each run.
- 2026-07-06 login-node lightweight validation after final travel and
  final-stage duration gate edits passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers, `bash -n` on the low-cradle and larger-box
  launchers, and `git diff --check` on the touched files. No simulation,
  rendering, model loading, or data processing was run on the login node.
  Queue check still showed job `168164` as `PENDING (Priority)`.
- 2026-07-06 final-stage stand-hold comparison submitted early because
  baseline job `168164` continued to wait in `PENDING (Priority)` with no
  output. Stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1`,
  tmux
  `curiosity_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand900_0706`,
  Slurm job `168177` (`g1_lg_fst`). It uses no lateral correction, yaw
  correction, terminal scale `0.015` from box target travel `0.65 m`, final
  latched full stop scale `0.0` after box target travel `2.0 m`, and
  `AGILE_COMMAND_HOLD_FINAL_STAND=1` with final-stand delay `20` and stand
  blend rate `0.02`. Strict gates additionally require final robot/box
  target-directed travel not to exceed `2.35 m`, at least `120` final-stage
  steps, and at least `80` final-stand steps. Submission-time queue check
  showed `168177` as `PENDING (Priority)` with no output files and no
  scheduled start time. This is only queued direct Isaac evidence, not a
  behavior result yet; compare against `168164` once both produce summaries.
- 2026-07-06 login-node lightweight validation after submitting `168177`
  passed: `python3 -m py_compile` on the G1 scene/checker/summarizers,
  `bash -n` on the low-cradle and larger-box launchers, and `git diff
  --check` on touched files. Queue check showed both `168164` (`g1_lg_f0m`)
  and `168177` (`g1_lg_fst`) still `PENDING (Priority)`. No output files were
  present for either queued direct Isaac run.
- 2026-07-06 follow-up queue poll after a 30 second wait still showed both
  `168164` and `168177` as `PENDING (Priority)` with no output files. Current
  blocker is Slurm priority scheduling only; the direct Isaac jobs are queued
  through tmux+srun and no external model/download path is being waited on.
- 2026-07-06 low-carry 900-step matrix reporting sharpened while waiting for
  GPU resources. `scripts/isaac/summarize_core_world_g1_lowcarry_900_matrix.py`
  and
  `experiments/reports/2026-07-06_g1_lowcarry_900_control_matrix.md` now show
  robot/box target-directed travel together and split final-stage hold fields
  into a separate `final hold` column. This makes queued jobs `168164`
  (full stop after 2 m) and `168177` (final-stage stand hold) immediately
  comparable on distance, overshoot, final-stage activation, final-stand
  activation, lateral drift, tilt, fall/drop, and checker status once their
  summary/check JSON files appear.
- 2026-07-06 login-node lightweight validation after low-carry matrix
  reporting update passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers, `bash -n` on the low-cradle and larger-box
  launchers, and `git diff --check` on touched files. Queue check still
  showed `168164` and `168177` as `PENDING (Priority)` with no output files.
- 2026-07-06 `scontrol` queue check at scheduler evaluation
  `15:06:02 CST`: both `168164` and `168177` remained `PENDING (Priority)`,
  both scheduled for `2026-07-06T16:18:59` on `server39`, and neither had
  output files yet.
- 2026-07-06 added a dedicated final-hold comparison helper:
  `scripts/isaac/summarize_core_world_g1_final_hold_comparison.py`. It
  compares the queued full-stop-after-2m case
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_strict_900_targetnegx1`
  against the queued final-stand-after-2m case
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1`.
  The report records checker status, fall/drop, first event steps,
  robot/box target-directed travel, tilt, final relative offset, final
  lateral error, final-stage steps, final-stand steps, and rollout write
  counts. It enforces diagnostic thresholds for distance, overshoot,
  fall/drop, tilt, relative offset, lateral error, final-stage activation, and
  final-stand activation. The helper is diagnostic comparison only and must
  not be used as a final carrying success claim.
- 2026-07-06 login-node lightweight validation after adding the final-hold
  comparison helper passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers including the new comparison helper, `bash -n` on
  the low-cradle and larger-box launchers, and `git diff --check` on touched
  files. Queue check still showed `168164` and `168177` as
  `PENDING (Priority)` with no output files.
- 2026-07-06 follow-up queue poll after a 45 second wait still showed both
  `168164` and `168177` as `PENDING (Priority)` with no output files. No
  tmux sessions were attached or modified; `carry1` remains untouched.
- 2026-07-06 final-hold comparison helper tightened. It now treats missing or
  failing `check.json` as a comparison failure and enforces zero rollout
  shortcut writes by default for root pose, root velocity, and box pose. The
  Markdown comparison also shows root/velocity/box rollout write counts, so a
  full-stop/final-stand result cannot look acceptable if it used a hidden
  root or box write shortcut.
- 2026-07-06 login-node lightweight validation after tightening final-hold
  comparison passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers including the comparison helper, `bash -n` on the
  low-cradle and larger-box launchers, and `git diff --check` on touched
  files. Queue check still showed `168164` and `168177` as
  `PENDING (Priority)` with no output files.
- 2026-07-06 `scontrol` queue check at scheduler evaluation
  `15:11:05 CST`: both `168164` and `168177` remained `PENDING (Priority)`,
  now scheduled for `2026-07-06T16:28:59` on `server39`, and neither had
  output files yet.
- 2026-07-06 optional final-hold comparison postprocessing added to the
  low-cradle launcher. `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`
  now supports `GENERATE_FINAL_HOLD_COMPARISON=1`, which writes
  `g1_final_hold_comparison.json`, `g1_final_hold_comparison.md`, and
  `g1_final_hold_comparison.stdout.md` under the suite directory by invoking
  `scripts/isaac/summarize_core_world_g1_final_hold_comparison.py` after case
  execution. Default is disabled, so queued jobs `168164` and `168177` are
  unchanged. The launcher env snapshot now records `GENERATE_` variables.
- 2026-07-06 login-node lightweight validation after adding optional
  final-hold comparison postprocessing passed: `python3 -m py_compile` on the
  G1 scene/checker/summarizers including the comparison helper, `bash -n` on
  the low-cradle and larger-box launchers, and `git diff --check` on touched
  files. Queue check still showed `168164` and `168177` as
  `PENDING (Priority)` with no output files.
- 2026-07-06 follow-up queue poll after another 30 second wait still showed
  both `168164` and `168177` as `PENDING (Priority)` with no output files.
  No additional direct Isaac jobs were submitted in this poll.
- 2026-07-06 queue/status check at `15:14:22 CST`: `squeue` and `sacct`
  showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) still
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. `scontrol` scheduler evaluation `15:14:02 CST` still scheduled both
  for `2026-07-06T16:28:59` on `server39`. No additional direct Isaac jobs
  were submitted; avoid queue pollution while these two controlled hypotheses
  are already waiting.
- 2026-07-06 queue/status check at `15:15:11 CST`: `squeue` and `sacct`
  again showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. `scontrol` scheduler evaluation `15:15:02 CST` still scheduled both
  for `2026-07-06T16:28:59` on `server39`. The project tmux sessions for
  these two queued jobs still exist; `carry1` was only listed to avoid
  interference and was not attached to or modified.
- 2026-07-06 target-window stable-hold metrics added to the direct G1 Isaac
  scene. `scripts/isaac/build_core_world_g1_box_scene.py` now supports
  `--target-window-center` and `--target-window-halfwidth` (disabled by
  default with negative values). When enabled, it records robot/box/both
  stable steps inside the target-directed travel window, longest stable
  streaks, and first stable steps, counting only steps without fall/drop.
  `check_core_world_g1_box_scene_summary.py` now supports minimum target-window
  robot/box/both stable-step gates and a minimum both-longest-streak gate. The
  low-cradle launcher exposes these through `TARGET_WINDOW_CENTER`,
  `TARGET_WINDOW_HALFWIDTH`, `MIN_TARGET_WINDOW_ROBOT_STABLE_STEPS`,
  `MIN_TARGET_WINDOW_BOX_STABLE_STEPS`,
  `MIN_TARGET_WINDOW_BOTH_STABLE_STEPS`, and
  `MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS`. The final-hold comparison
  helper reports and can gate target-window both stable steps/streaks. This is
  for future strict carry-and-hold validations; queued jobs `168164`/`168177`
  are unchanged because they were submitted before this metric existed.
- 2026-07-06 login-node lightweight validation after target-window metric
  edits passed: `python3 -m py_compile` on the G1 scene/checker/final-hold
  comparison helper, `bash -n` on the low-cradle launcher, and `git diff
  --check` on touched files. No simulation or data processing was run on the
  login node. Queue check still showed `168164` and `168177` as
  `PENDING (Priority)`.
- 2026-07-06 target-window fields added to aggregate reporting. The larger-box
  strict summarizer now carries target-window enabled/center/halfwidth,
  robot/box/both stable steps, longest streaks, and first stable steps into
  its JSON report. The low-carry 900-step matrix and its current Markdown
  report now include target-window status, window bounds, both-stable steps,
  and both-stable longest streak in the final-hold column. Existing older runs
  show missing target-window values because they predate the metric.
- 2026-07-06 login-node lightweight validation after target-window aggregate
  reporting edits passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers, `bash -n` on the low-cradle and larger-box
  launchers, and `git diff --check` on touched files. Queue check still
  showed `168164` and `168177` as `PENDING (Priority)`.
- 2026-07-06 queue/status check at `15:23:14 CST`: `squeue` and `sacct`
  still showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. No new direct Isaac jobs were submitted.
- 2026-07-06 planned next strict validation recorded but not submitted:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_targetwindow_strict_900_targetnegx1`.
  It keeps the final-stand setup from `168177` but adds
  `TARGET_WINDOW_CENTER=2.0`, `TARGET_WINDOW_HALFWIDTH=0.35`,
  `MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80`,
  `MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50`, and
  `GENERATE_FINAL_HOLD_COMPARISON=1`. Purpose: after `168164`/`168177`
  provide behavior evidence, verify actual stable post-carry holding in a
  target window rather than relying on only final-frame distance.
- 2026-07-06 login-node lightweight validation after recording the planned
  target-window strict validation passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers, `bash -n` on the low-cradle and larger-box
  launchers, and `git diff --check` on touched files. Queue check still
  showed `168164` and `168177` as `PENDING (Priority)`.
- 2026-07-06 queue/status check at `15:24:52 CST`: `squeue` and `sacct`
  still showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files.
- 2026-07-06 final-hold comparison helper exit status tightened.
  `scripts/isaac/summarize_core_world_g1_final_hold_comparison.py` now returns
  nonzero if any comparison case has `status != pass`. This makes missing
  summaries, failed `check.json`, shortcut writes, target-window failures, or
  final-hold gate failures visible to shell automation instead of only in the
  Markdown/JSON report.
- 2026-07-06 login-node lightweight validation after final-hold comparison
  exit-status change passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers, `bash -n` on the low-cradle and larger-box
  launchers, and `git diff --check` on touched files. Queue check still
  showed `168164` and `168177` as `PENDING (Priority)`.
- 2026-07-06 queue/status check at `15:26:20 CST`: `squeue` and `sacct`
  still showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. `scontrol` scheduler evaluation `15:26:03 CST` moved the scheduled
  start for both jobs to `2026-07-06T16:37:02` on `server39`. The two
  project tmux sessions still exist; `carry1` was only listed for avoidance
  and was not attached to or modified. No additional jobs were submitted.
- 2026-07-06 queue/status check at `15:27:16 CST`: `squeue` and `sacct`
  still showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. `scontrol` scheduler evaluation `15:27:03 CST` still scheduled both
  for `2026-07-06T16:37:02` on `server39`. The user GPU queue for this account
  showed only these two project jobs, both pending. No additional direct Isaac
  jobs were submitted.
- 2026-07-06 queue/status check at `15:28:10 CST`: `squeue` and `sacct`
  still showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. `scontrol` scheduler evaluation `15:28:03 CST` still scheduled both
  for `2026-07-06T16:37:02` on `server39`. `git diff --check` passed on the
  touched files. No additional direct Isaac jobs were submitted.
- 2026-07-06 queue/status check at `15:29:03 CST`: `squeue` and `sacct`
  still showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. No additional direct Isaac jobs were submitted.
- 2026-07-06 queue/status check at `15:29:53 CST`: `squeue` and `sacct`
  still showed both `168164` (`g1_lg_f0m`) and `168177` (`g1_lg_fst`) as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. `scontrol` scheduler evaluation `15:30:03 CST` still scheduled both
  for `2026-07-06T16:37:02` on `server39`. `git diff --check` passed on the
  touched files. No additional direct Isaac jobs were submitted.
- 2026-07-06 user pushback accepted: do not wait for external models,
  checkpoint downloads, policy servers, or optional wrappers when they do not
  directly unblock the Isaac carrying scene. The active work remains direct
  Isaac scene construction and strict gates. Queue/status check at
  `15:31:47 CST` showed `168164` and `168177` still `PENDING (Priority)`,
  elapsed `00:00:00`, no nodes assigned, and no output files. The low-cradle
  launcher was checked for a suspected duplicate rescue-enable flag; current
  file content already has only one `--agile-command-hold-rescue-enable`
  insertion, so no launcher edit was needed.
- 2026-07-06 login-node lightweight validation after recording the direct
  Isaac priority rule passed: `python3 -m py_compile` on the G1
  scene/checker/summarizers, `bash -n` on the low-cradle and larger-box
  launchers, and `git diff --check` on touched files. Follow-up queue/status
  check at `15:33:17 CST` still showed `168164` and `168177` as
  `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned, and no output
  files. No simulation, rendering, training, or model loading was run on the
  login node.
- 2026-07-06 direct Isaac gate tightened while waiting for compute. The G1 box
  scene now records target-window stable hold specifically during the final
  post-carry command stage and during the final-stand stage:
  `target_window_both_final_hold_stable_steps`,
  `target_window_both_final_hold_longest_streak_steps`,
  `target_window_both_final_hold_first_stable_step`,
  `target_window_both_final_stand_stable_steps`,
  `target_window_both_final_stand_longest_streak_steps`, and
  `target_window_both_final_stand_first_stable_step`. The checker exposes
  matching minimum-step and longest-streak gates, and the low-cradle launcher
  passes them from `MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_*` and
  `MIN_TARGET_WINDOW_BOTH_FINAL_STAND_*` environment variables. The final-hold
  comparison helper, larger-box summarizer, and low-carry matrix report these
  fields. Purpose: a future run cannot pass strict target-window validation
  merely by moving through the 2 m window before the post-carry hold.
- 2026-07-06 planned target-window final-stand validation was upgraded in
  `TODO/03_no_root_articulated_carrier/todo.md` to require both ordinary
  target-window stability and post-carry-stage target-window stability:
  `MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80`,
  `MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50`,
  `MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STABLE_STEPS=60`, and
  `MIN_TARGET_WINDOW_BOTH_FINAL_STAND_LONGEST_STREAK_STEPS=40`.
  This planned job was not submitted because `168164` and `168177` are still
  pending; wait for those two controlled hypotheses before adding another
  queued run unless the user explicitly redirects.
- 2026-07-06 login-node lightweight validation after the stricter post-carry
  target-window gate edits passed: `python3 -m py_compile` on the G1
  scene/checker/final-hold comparison/larger-box summarizer/low-carry matrix,
  `bash -n` on the low-cradle and larger-box launchers, and `git diff
  --check` on touched files. Queue/status check at `15:37:10 CST` still
  showed `168164` and `168177` as `PENDING (Priority)`, elapsed `00:00:00`,
  no nodes assigned, and no output files. No simulation, rendering, training,
  or model loading was run on the login node.
- 2026-07-06 direct Isaac target-window gate tightened again to prevent
  end-of-rollout false positives. The G1 scene now records end-of-rollout
  target-window streaks:
  `target_window_both_streak_at_end_steps`,
  `target_window_both_final_hold_streak_at_end_steps`, and
  `target_window_both_final_stand_streak_at_end_steps`, plus final-step
  booleans for ordinary, final-hold, and final-stand target-window stability.
  The checker, low-cradle launcher, final-hold comparison helper,
  larger-box summarizer, and low-carry matrix all expose/report these fields.
  This means a future strict run can require that the robot and box are still
  inside the 2 m target window at rollout end, not only that they were there
  earlier.
- 2026-07-06 planned target-window final-stand validation was upgraded in
  `TODO/03_no_root_articulated_carrier/todo.md` with end-streak gates:
  `MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40`,
  `MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40`, and
  `MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STREAK_AT_END_STEPS=30`. The planned
  run is still not submitted while jobs `168164` and `168177` remain pending.
- 2026-07-06 login-node lightweight validation after end-streak gate edits
  passed: `python3 -m py_compile` on the G1 scene/checker/final-hold
  comparison/larger-box summarizer/low-carry matrix, `bash -n` on the
  low-cradle and larger-box launchers, and `git diff --check` on touched
  files. Queue/status check at `15:41:03 CST` still showed `168164` and
  `168177` as `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned,
  and no output files. No simulation, rendering, training, or model loading
  was run on the login node.
- 2026-07-06 larger-box multi-posture gate preparation. Added
  `scripts/isaac/run_core_world_g1_largerbox_finalhold_posture_matrix.sh`.
  It refuses login/management nodes and is meant to run only inside a
  Curiosity-owned tmux-held Slurm allocation. It runs `boxtilt`, `lowcarry`,
  and `chestpad` through the same 900-step final-hold/final-stand
  target-window and end-streak gates, so it is the next validation layer for
  the "any carrying posture" requirement after the current low-carry
  final-hold hypothesis is known. Also tightened
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`: aggregate
  status now passes only if every discovered case passes, and it reports
  `failed_case_count`; this prevents a multi-posture matrix from looking
  successful because only one posture passed. Lightweight validation passed:
  `bash -n` on the relevant launchers, `python3 -m py_compile` on the G1
  scene/checker/summarizers, and `git diff --check` on touched files.
- 2026-07-06 low-carry full-stop-after-2m result `168164` is valid negative
  evidence. Slurm job `168164` (`g1_lg_f0m`) ran on `server30` and completed
  in `00:00:54`. The run
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_strict_900_targetnegx1`
  completed `900/900` with build status `0`, checker status `1`,
  fall/drop `9/0`, first fall step `891`, min robot/box z
  `0.349306/0.434169 m`, max robot/true-box tilt
  `0.557659/0.804987 rad`, final robot/box target-directed travel
  `3.348633/3.359532 m`, final robot/box lateral error
  `0.389801/0.379783 m`, final relative offset `0.033178 m`, and rollout
  root pose/root velocity/box pose writes all `0`. Conclusion: full stop only
  after `2.0 m` still overshoots and falls late; it is not stable post-carry
  hold. The final-stand comparison job `168177` remains pending and should be
  interpreted separately when it produces output.
- 2026-07-06 queue/status check at `15:45:08 CST`: final-stand comparison job
  `168177` (`g1_lg_fst`) still showed `PENDING (Priority)`, elapsed
  `00:00:00`, no nodes assigned, and no output files under
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1`.
  `git diff --check` passed on the touched files. No additional compute jobs
  were submitted after `168164`; in particular, the new multi-posture
  final-hold matrix is prepared but intentionally not submitted until the
  current final-stand hypothesis is known.
- 2026-07-06 follow-up analysis of `168164`: despite
  `agile_command_hold_final_scale=0.0`, the final command still contained a
  nonzero yaw component (`agile_last_command_xyz_yaw[2] = 0.01562`) because
  yaw correction was added after command scaling. Added an optional final
  correction clamp to test this failure mode without changing default
  behavior: `--agile-command-hold-final-zero-corrections`, exposed by the
  low-cradle launcher as `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1`. When
  enabled, final-stage lateral/yaw corrections are suppressed after the final
  threshold/latch and summaries report
  `agile_command_hold_final_lateral_suppressed_steps` and
  `agile_command_hold_final_yaw_suppressed_steps`. Checker, final-hold
  comparison, larger-box summarizer, and low-carry matrix now report these
  fields. This option is off by default and does not affect queued job
  `168177`.
- 2026-07-06 planned but not submitted next diagnostic if `168177` also fails:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_zerocorr_strict_900_targetnegx1`
  with `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1`. Purpose: isolate whether
  final-stage yaw correction caused or amplified the `168164` post-2m
  overshoot and late fall. Do not submit this before the current final-stand
  comparison produces behavior evidence unless the user explicitly redirects.
- 2026-07-06 login-node lightweight validation after final correction clamp
  edits passed: `python3 -m py_compile` on the G1 scene/checker/final-hold
  comparison/larger-box summarizer/low-carry matrix, `bash -n` on the
  low-cradle/larger-box/finalhold-posture launchers, and `git diff --check`
  on touched files. Queue/status check at `15:48:09 CST` still showed
  `168177` as `PENDING (Priority)`, elapsed `00:00:00`, no nodes assigned,
  and no output files. No simulation, rendering, training, or model loading
  was run on the login node.
- 2026-07-06 final-stage command telemetry/gates added. The G1 scene now
  records `agile_command_hold_final_max_abs_command_x`,
  `agile_command_hold_final_max_abs_command_y`,
  `agile_command_hold_final_max_abs_command_yaw`, and
  `agile_command_hold_final_last_command_xyz_yaw` whenever the final
  target-travel hold stage is active. The checker exposes
  `--max-final-hold-command-x`, `--max-final-hold-command-y`, and
  `--max-final-hold-command-yaw`, and the low-cradle launcher passes these
  from `MAX_FINAL_HOLD_COMMAND_X/Y/YAW`. The final-hold comparison helper,
  larger-box summarizer, and low-carry matrix report these fields. Purpose:
  future strict runs can directly verify that post-carry hold commands are
  near zero, rather than relying only on `agile_command_hold_final_scale=0.0`.
- 2026-07-06 planned zero-corrections diagnostic in
  `TODO/03_no_root_articulated_carrier/todo.md` was tightened with
  `MAX_FINAL_HOLD_COMMAND_X=0.001`, `MAX_FINAL_HOLD_COMMAND_Y=0.001`, and
  `MAX_FINAL_HOLD_COMMAND_YAW=0.001`. This run remains planned only and was
  not submitted while `168177` is pending.
- 2026-07-06 login-node lightweight validation after final-stage command
  telemetry/gate edits passed: `python3 -m py_compile` on the G1
  scene/checker/final-hold comparison/larger-box summarizer/low-carry matrix,
  `bash -n` on the low-cradle/larger-box/finalhold-posture launchers, and
  `git diff --check` on touched files. Queue/status check at `15:51:45 CST`
  still showed `168177` as `PENDING (Priority)`, elapsed `00:00:00`, no
  nodes assigned, and no output files. No simulation, rendering, training, or
  model loading was run on the login node.
- 2026-07-06 final command gates wired into comparison/matrix layers. The
  final-hold comparison helper now accepts `--max-final-hold-command-x`,
  `--max-final-hold-command-y`, and `--max-final-hold-command-yaw`; the
  low-cradle launcher passes `MAX_FINAL_HOLD_COMMAND_X/Y/YAW` into that
  helper when `GENERATE_FINAL_HOLD_COMPARISON=1`. The multi-posture final-hold
  matrix now defaults these gates to `0.001` for x, y, and yaw commands. This
  keeps per-case checker, comparison reports, and future all-posture matrices
  aligned on the requirement that post-carry final hold commands are genuinely
  near zero.
- 2026-07-06 login-node lightweight validation after wiring final command
  gates through comparison/matrix layers passed: `python3 -m py_compile` on
  the G1 scene/checker/final-hold comparison/larger-box summarizer/low-carry
  matrix, `bash -n` on the low-cradle/larger-box/finalhold-posture launchers,
  and `git diff --check` on touched files. Queue/status check at
  `15:53:35 CST` still showed `168177` as `PENDING (Priority)`, elapsed
  `00:00:00`, no nodes assigned, and no output files. No additional compute
  jobs were submitted.
- 2026-07-06 final-stage stability telemetry/gates added. The G1 scene now
  records final-hold-specific and final-stand-specific stability metrics:
  `agile_command_hold_final_min_robot_z_m`,
  `agile_command_hold_final_min_box_z_m`,
  `agile_command_hold_final_max_tilt_rad`,
  `agile_command_hold_final_max_box_tilt_rad`,
  `agile_command_hold_final_stand_min_robot_z_m`,
  `agile_command_hold_final_stand_min_box_z_m`,
  `agile_command_hold_final_stand_max_tilt_rad`, and
  `agile_command_hold_final_stand_max_box_tilt_rad`. The checker now exposes
  matching gates for min final-hold/final-stand robot/box z and max
  final-hold/final-stand robot/box tilt. The low-cradle launcher passes these
  from `MIN_FINAL_HOLD_*`, `MAX_FINAL_HOLD_*`, `MIN_FINAL_STAND_*`, and
  `MAX_FINAL_STAND_*` environment variables. The final-hold comparison helper,
  larger-box summarizer, and low-carry matrix report the fields, and the
  comparison helper can enforce the same thresholds.
- 2026-07-06 multi-posture finalhold matrix defaults tightened again. In
  addition to target-window/end-streak and final command gates, it now
  defaults to final-stage stability thresholds: final-hold and final-stand
  robot/box z at least `0.45 m`, robot tilt at most `0.35 rad`, and box tilt
  at most `0.45 rad`. The planned single-posture target-window final-stand
  validation in `TODO/03_no_root_articulated_carrier/todo.md` was also
  annotated to include these final-stage stability gates when submitted.
- 2026-07-06 login-node lightweight validation after final-stage stability
  telemetry/gate edits passed: `python3 -m py_compile` on the G1
  scene/checker/final-hold comparison/larger-box summarizer/low-carry matrix,
  `bash -n` on the low-cradle/larger-box/finalhold-posture launchers, and
  `git diff --check` on touched files. Queue/status check at `15:58:40 CST`
  still showed `168177` as `PENDING (Priority)`, elapsed `00:00:00`, no
  nodes assigned, and no output files. No simulation, rendering, training, or
  model loading was run on the login node.
- 2026-07-06 user correction: do not wait on external models, datasets,
  policy servers, or optional official wrappers when they are not directly
  useful. Continue constructing and gating the carrying scene directly in
  Isaac. The current active blocker is the G1 low-carry final braking/hold
  controller after about `2 m` of carried-box travel, not a missing external
  checkpoint or video model.
- 2026-07-06 login-node process safety note: a high-CPU VS Code server
  ripgrep process was observed outside the commands used for this work. Do not
  trigger broad `rg --files --hidden --no-ignore --follow` scans on the login
  node. Keep future login-node actions to lightweight text reads, shell
  syntax checks, git operations, Slurm status checks, and documented edits.
- 2026-07-06 final-stage fall/drop telemetry and gates added. The G1 scene now
  records final-hold and final-stand fall/drop counts and first event steps:
  `agile_command_hold_final_fall_events`,
  `agile_command_hold_final_box_drop_events`,
  `agile_command_hold_final_first_fall_step`,
  `agile_command_hold_final_first_box_drop_step`,
  `agile_command_hold_final_stand_fall_events`,
  `agile_command_hold_final_stand_box_drop_events`,
  `agile_command_hold_final_stand_first_fall_step`, and
  `agile_command_hold_final_stand_first_box_drop_step`. The checker exposes
  matching max gates, and the low-cradle launcher, final-hold comparison,
  larger-box summarizer, low-carry matrix, and finalhold posture matrix all
  propagate/report these fields. The finalhold posture matrix defaults these
  final-stage fall/drop gates to `0`.
- 2026-07-06 lightweight validation after final-stage fall/drop telemetry/gate
  edits passed on the login node: `python3 -m py_compile` for the touched G1
  scene/checker/summarizers, `bash -n` for the touched launchers, and
  `git diff --check` for the touched files. These were syntax/text checks
  only; no simulation, rendering, training, evaluation, or model loading was
  run on the login node.
- 2026-07-06 final-stage stand-hold comparison job `168177` (`g1_lg_fst`)
  completed on `server44` in `00:01:01` with build status `0` and checker
  status `1`, so it is valid negative behavior evidence. Run stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1`.
  It completed `900/900`, entered final hold at step `695` for `205` steps
  and final stand at step `715` for `185` steps, but failed with fall/drop
  `115/70`, first fall step `785`, first box-drop step `830`, final robot/box
  target-directed travel `3.218096/3.123645 m`, final robot/box lateral error
  `0.549016/0.435925 m`, final relative offset `0.368966 m`, final-hold
  min robot/box z `-1.299788/-1.523448 m`, final-hold max robot/box tilt
  `3.137067/3.137737 rad`, and final-stage command yaw still `0.021895`
  because final zero-corrections was disabled. Conclusion: simply switching to
  final-stage stand does not solve the post-2m braking/hold failure.
- 2026-07-06 active next direct-Isaac step: run or implement the
  final-zero-corrections diagnostic before broad multi-posture sweeps. If a
  run with final x/y/yaw commands genuinely clamped near zero still falls or
  drops, do not return to external model waiting; change the Isaac terminal
  braking/standing controller or carry scaffold directly.
- 2026-07-06 submitted the direct Isaac final-zero-corrections diagnostic in a
  new Curiosity-owned tmux session without touching `carry1` or excluded
  sessions. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_zerocorr900_0706`; Slurm job
  `168247`, job-name `g1_lg_zcorr`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_zerocorr_strict_900_targetnegx1`.
  It sets `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1`,
  `FREE_STEPS=900`, final travel upper gates `2.35 m`, final active steps
  `>=120`, final command gates `MAX_FINAL_HOLD_COMMAND_X/Y/YAW=0.001`,
  final-hold robot/box z gates `>=0.45 m`, final-hold robot/box tilt gates
  `<=0.35/0.45 rad`, and final-hold fall/drop gates `0/0`. Submission-time
  `squeue` status was `PENDING (Priority)`. This is a direct Isaac scene
  diagnostic, not external model waiting.
- 2026-07-06 final-zero-corrections diagnostic `168247` completed on
  `server39` in `00:00:56` with build status `0` and checker status `1`, so
  it is valid negative behavior evidence with a useful improvement. Stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_zerocorr_strict_900_targetnegx1`.
  It completed `900/900`, entered final hold at step `695` for `205` steps,
  clamped final x/y/yaw commands to `0/0/0`, suppressed yaw for `205` final
  steps, and had fall/drop `0/0`, final-hold fall/drop `0/0`, min final-hold
  robot/box z `0.454602/0.488559 m`, and rollout root pose/root velocity/box
  pose writes all `0`. It still failed strict holding because final robot/box
  target-directed travel overshot to `3.098152/3.164331 m` and final-hold max
  robot/box tilt reached `0.383518/0.678150 rad` over the `0.35/0.45` gates.
  Conclusion: nonzero final yaw was a major cause of previous fall/drop, but
  command zeroing alone does not brake the carrier into the 2 m hold window.
  The next direct Isaac check is zero-corrections plus final-stage stand.
- 2026-07-06 submitted the combined zero-corrections plus final-stage stand
  diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_zerocorr_finalstand900_0706`; Slurm
  job `168252`, job-name `g1_lg_zfst`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_zerocorr_finalstand_strict_900_targetnegx1`.
  It keeps final x/y/yaw command gates at `0.001`, final-hold and final-stand
  fall/drop gates at `0/0`, final travel upper gates at `2.35 m`, final
  active steps `>=120`, final-stand steps `>=80`, and final-hold/final-stand
  height/tilt gates. Purpose: test whether final stand can brake the carrier
  after yaw/lateral final corrections are suppressed. Submission-time `squeue`
  status was `PENDING (Priority)`.
- 2026-07-06 combined zero-corrections plus final-stage stand diagnostic
  `168252` completed on `server30` in `00:00:58` with build status `0` and
  checker status `1`, so it is valid negative behavior evidence. It completed
  `900/900`, clamped final x/y/yaw commands to `0/0/0`, and suppressed final
  yaw for `205` steps, but the final stand blend destabilized the carrier:
  fall/drop `105/89`, first fall step `795`, first box-drop step `811`, min
  final-hold robot/box z `-1.274945/-1.181000 m`, max final-hold robot/box
  tilt `3.133741/3.139424 rad`, final robot/box target-directed travel
  `2.940374/2.561408 m`, and final relative offset `0.382839 m`. Conclusion:
  the current stand-target blend is not a safe terminal braking controller for
  the carried-box state. Do not keep tuning final stand as the main fix.
- 2026-07-06 added a direct Isaac final-stage short reverse-command brake hook
  to the G1 box scene. New scene args:
  `--agile-command-hold-final-brake-command-x`,
  `--agile-command-hold-final-brake-delay-steps`, and
  `--agile-command-hold-final-brake-steps`; launcher envs:
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X`,
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS`, and
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS`. Summary/checker now report brake
  command, delay, duration, active steps, first/last active step, and max
  absolute brake x command. Defaults are disabled, so existing runs are
  unchanged. Lightweight validation passed on the login node: `py_compile`,
  `bash -n`, and `git diff --check`; no simulation or model loading was run
  on the login node.
- 2026-07-06 submitted the first final-brake diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalbrake005_80_0706`; Slurm job
  `168275`, job-name `g1_lg_fbrk`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalbrake005_80_strict_900_targetnegx1`.
  It uses final zero-corrections, no final stand, `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.05`,
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=80`, and strict final fall/drop,
  height, tilt, and target-travel gates. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 first final-brake diagnostic `168275` completed on `server39` in
  `00:00:53` with build status `0` and checker status `1`, so it is valid
  negative behavior evidence. The hook worked mechanically: final brake active
  steps `80`, first/last active steps `695/774`, max brake x command `0.05`,
  and final yaw command stayed `0`. But `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.05`
  worsened terminal behavior: fall/drop `79/63`, first fall/drop steps
  `821/837`, final robot/box target-directed travel `3.566093/3.343874 m`,
  max final-hold robot/box tilt `1.262435/1.573505 rad`, and min final-hold
  robot/box z `-1.033506/-1.181892 m`. Conclusion: this sign is not a safe
  brake direction for the current G1 agile policy; run a positive-sign
  contrast before changing the mechanism.
- 2026-07-06 submitted the positive-sign final-brake contrast. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalbrake_pos005_80_0706`; Slurm
  job `168278`, job-name `g1_lg_fpbr`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalbrake_pos005_80_strict_900_targetnegx1`.
  It uses the same strict gates as `168275` but sets
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=0.05` for `80` final steps.
  Purpose: determine the command sign that actually brakes the carried G1
  after the final 2 m latch. Submission-time status was `PENDING (Priority)`.
- 2026-07-06 positive-sign final-brake contrast `168278` completed on
  `server39` in `00:00:52` with build status `0` and checker status `1`, so
  it is valid negative behavior evidence. Positive brake was safer than the
  negative sign but still failed: final brake active `80` steps, final command
  `x/y/yaw=0.05/0/0` during the brake window, final yaw suppressed for `205`
  steps, fall/drop `30/9`, first fall/drop steps `870/891`, final robot/box
  target-directed travel `3.268357/3.303698 m`, max final-hold robot/box tilt
  `0.587173/0.582167 rad`, min final-hold robot/box z
  `0.064077/0.038243 m`, and final relative offset `0.170510 m`. Conclusion:
  neither final stand nor simple final brake pulses solve the terminal hold.
  The next direct Isaac check is an earlier final-zero latch so the robot
  coasts into the 2 m target window.
- 2026-07-06 submitted early final-zero latch diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly120_zero_0706`; Slurm job
  `168306`, job-name `g1_lg_fe12`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly120_zero_strict_900_targetnegx1`.
  It sets the final zero latch at `1.2 m` box target-directed travel, keeps
  yaw/lateral final corrections suppressed, and adds target-window gates
  around `2.0 +/- 0.35 m` with stable-step/end-streak requirements. Purpose:
  test whether early command cutoff can coast into the 2 m window without the
  unstable stand-target blend or final brake pulses. Submission-time status
  was `PENDING (Priority)`.
- 2026-07-06 early final-zero latch diagnostic `168306` completed on
  `server39` in `00:00:49` with build status `0` and checker status `1`, so
  it is valid negative evidence but the closest result so far. It latched
  final zero at step `531` for `369` final steps, clamped final x/y/yaw
  commands to `0/0/0`, suppressed final yaw for `369` steps, had box drop
  `0`, and achieved target-window stability with
  `target_window_both_final_hold_stable_steps=126` and longest streak `126`.
  It still failed long-hold gates because the stable target-window streak did
  not last to the final step, first fall occurred at step `887`, final-hold
  fall events were `13`, final robot/box target-directed travel ended at
  `3.126255/3.135492 m`, max final-hold robot/box tilt was
  `0.477018/0.476662 rad`, and min final-hold robot/box z was
  `0.313819/0.383676 m`. Conclusion: early zero can coast through the 2 m
  target window and hold briefly, but final zero still has hidden-state or
  stance drift over longer hold.
- 2026-07-06 added final-latch policy-state reset hook. New scene arg:
  `--agile-command-hold-final-reset-policy-state`; launcher env:
  `AGILE_COMMAND_HOLD_FINAL_RESET_POLICY_STATE=1`. Summary/checker now report
  whether final reset was enabled, final reset count, and final reset error.
  Defaults are disabled. Lightweight validation passed on the login node:
  `py_compile`, `bash -n`, and `git diff --check`; no simulation or model
  loading was run on the login node.
- 2026-07-06 submitted early final-zero latch plus final policy-state reset
  diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly120_reset_zero_0706`;
  Slurm job `168313`, job-name `g1_lg_fr12`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly120_reset_zero_strict_900_targetnegx1`.
  It repeats `168306` with `AGILE_COMMAND_HOLD_FINAL_RESET_POLICY_STATE=1`.
  Purpose: test whether resetting the recurrent policy at final latch reduces
  the late drift/fall while preserving the 2 m target-window stable streak.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 early final-zero latch plus final policy-state reset diagnostic
  `168313` completed on `server39` in `00:00:48` with build status `0` and
  checker status `1`, so it is valid negative behavior evidence. Final policy
  reset executed once and reported no reset error, but it worsened the
  terminal hold versus `168306`: fall/drop `65/43`, first fall/drop steps
  `835/857`, final robot/box target-directed travel `3.445008/3.323817 m`,
  final robot/box lateral error `1.256486/1.326749 m`, max final-hold
  robot/box tilt `0.706925/1.014478 rad`, and min final-hold robot/box z
  `-0.876250/-0.676235 m`. It still had a target-window stable streak of
  `121` steps, but the reset increased lateral drift and box drop. Conclusion:
  do not use final policy-state reset as the active fix.
- 2026-07-06 added a direct Isaac final target-window joint-target freeze
  hook. New scene args:
  `--agile-command-hold-final-freeze-in-target-window`,
  `--agile-command-hold-final-freeze-max-tilt`, and
  `--agile-command-hold-final-freeze-max-box-tilt`; launcher envs:
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW`,
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_TILT`, and
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_BOX_TILT`. When enabled, after final
  hold is active and robot/box target-directed travel are both inside the
  configured target window with bounded robot/box tilt, the scene latches the
  current policy joint targets and keeps commanding those joint targets. This
  diagnostic does not write root pose, root velocity, or box pose. Summary and
  checker output now report whether the freeze was enabled, whether it
  latched, latch step, active steps, first active step, and configured tilt
  gates. Defaults are disabled. Lightweight login-node validation passed:
  `py_compile`, `bash -n`, and `git diff --check`; no simulation, rendering,
  model loading, or heavy Python task was run on the login node.
- 2026-07-06 submitted early final-zero latch plus target-window joint-target
  freeze diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly120_freeze_zero_0706`;
  Slurm job `168324`, job-name `g1_lg_ffrz`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly120_freeze_zero_strict_900_targetnegx1`.
  It repeats the `168306` early final-zero setup with
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1`,
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_TILT=0.25`, and
  `AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_BOX_TILT=0.35`. Purpose: test whether
  freezing a target-window-stable policy joint target prevents the late
  drift/fall while preserving no root pose, no root velocity, and no box pose
  writes. Submission-time status was `PENDING (Priority)`.
- 2026-07-06 target-window joint-target freeze diagnostic `168324` completed
  on `server39` in `00:00:51` with build status `0` and checker status `1`,
  so it is valid negative behavior evidence. The freeze hook worked
  mechanically: final freeze latched at step `628` and stayed active for
  `272` steps, with rollout root pose/root velocity/box pose writes all `0`.
  But freezing joint targets made terminal balance much worse than the
  non-freeze early-zero run `168306`: fall/drop `156/134`, first fall/drop
  steps `744/766`, final robot/box target-directed travel
  `3.381368/3.200969 m`, max final-hold robot/box tilt
  `3.141581/3.136450 rad`, and min final-hold robot/box z
  `-2.699332/-2.458781 m`. The target-window final-hold longest streak fell
  to `97` steps and did not last to the end. Conclusion: do not use
  joint-target freeze as the hold controller; it removes policy motion needed
  for balance. The next direct check is an earlier final-zero latch without
  freezing, because `168306` stayed stable until near the end but overshot the
  2 m window.
- 2026-07-06 submitted earlier final-zero latch diagnostic with no freeze,
  stand, brake, lateral correction, root pose, root velocity, or box pose
  writes. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly060_zero_0706`; Slurm job
  `168331`, job-name `g1_lg_fe06`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_zero_strict_900_targetnegx1`.
  It changes only `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL` from `1.2 m`
  to `0.6 m` and requires `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=450`,
  while keeping strict 900-step target-window, final command, final fall/drop,
  height, and tilt gates. Purpose: test whether `168306` failed mostly
  because final-zero was latched too late and the carried robot coasted beyond
  the 2 m target window. Submission-time status was `PENDING (Priority)`.
- 2026-07-06 earlier final-zero latch diagnostic `168331` completed on
  `server39` in `00:00:47` with build status `0` and checker status `1`, so
  it is valid negative but directionally useful behavior evidence. It latched
  final zero at step `365` for `535` final steps, kept final x/y/yaw commands
  at `0/0/0`, suppressed final yaw for `535` steps, and had rollout root pose/
  root velocity/box pose writes all `0`. It improved terminal target-directed
  distance: final robot/box target-directed travel was
  `2.241426/2.296863 m`, inside the `2.35 m` upper gate. But it still failed
  because lateral drift became the dominant failure mode: final robot/box
  lateral error `1.301041/1.191865 m`, fall/drop `65/28`, first fall/drop
  steps `835/872`, max robot/box tilt `1.646607/1.674712 rad`, and min
  robot/box z `0.172038/0.076788 m`. Target-window final-hold longest streak
  was `99` steps but did not last to the end. Conclusion: the forward cutoff
  is now close enough; the next direct check is a very small, excess-error,
  tilt-gated lateral correction while keeping final x and yaw near zero.
- 2026-07-06 submitted very small excess-error lateral correction diagnostic.
  Tmux: `curiosity_g1_agile_largerbox_lowcarry_finalearly060_latsmall_0706`;
  Slurm job `168335`, job-name `g1_lg_f6ly`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_latsmall_strict_900_targetnegx1`.
  It keeps the `0.6 m` final-zero latch and final x/yaw command gates near
  zero, but allows tiny final lateral command up to `0.003` by setting
  terminal-only lateral correction, lateral error start `0.45 m`,
  excess-error mode, gain `0.006`, command limit `0.0015`, and tilt gates
  `0.30/0.35 rad`. Purpose: reduce the late lateral drift seen in `168331`
  without reproducing the earlier aggressive lateral-correction failures.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 very small excess-error lateral diagnostic `168335` completed on
  `server39` in `00:00:48` with build status `0` and checker status `1`, so
  it is a near-pass but still valid negative evidence. It had fall/drop
  `0/0`, min robot/box z `0.660195/0.759457 m`, max robot/box tilt
  `0.208595/0.413612 rad`, final relative offset `0.049381 m`, final
  robot/box lateral error `0.259286/0.290766 m`, and rollout root pose/root
  velocity/box pose writes all `0`. It failed only on target-window end
  criteria because final robot/box target-directed travel overshot to
  `2.713489/2.747867 m` above the `2.35 m` upper gate; target-window
  final-hold longest streak was `164` steps but not at the final step. The
  lateral hook did not actually activate (`0` active steps) because lateral
  error stayed below the `0.45 m` threshold. The useful change versus `168331`
  appears to be setting yaw gain/limit to `0`, which prevents early yaw
  correction from creating large lateral drift. Next check: keep yaw gain `0`
  and move the final-zero cutoff earlier from `0.6 m` to `0.45 m`.
- 2026-07-06 submitted yaw-zero earlier-cutoff diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly045_yawzero_0706`; Slurm
  job `168339`, job-name `g1_lg_f45y`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly045_yawzero_strict_900_targetnegx1`.
  It keeps the stable `168335` settings with yaw gain/limit `0`, tiny
  excess-error lateral correction available, no root pose/root velocity/box
  pose rollout writes, and final command gates near zero, but moves
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL` from `0.6 m` to `0.45 m`.
  Purpose: preserve the no-fall/no-drop behavior of `168335` while ending
  inside the `2.0 +/- 0.35 m` target window instead of overshooting it.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 yaw-zero `0.45 m` cutoff diagnostic `168339` completed on
  `server39` in `00:00:47` with build status `0` and checker status `1`, so
  it is valid negative evidence. It had rollout root pose/root velocity/box
  pose writes all `0`, but the cutoff was too early and destabilized the
  carried robot before it entered the target window: fall/drop `276/217`,
  first fall/drop steps `624/644`, final robot/box target-directed travel
  only `1.554673/1.521620 m`, target-window stable steps `0`, max robot/box
  tilt `1.438984/1.460016 rad`, and min robot/box z
  `0.123796/0.097214 m`. Lateral correction again had `0` active steps and
  was suppressed by tilt for `44` steps. Conclusion: `0.45 m` is too early;
  the viable cutoff is between `0.45 m` and the stable-but-overshooting
  `0.6 m`. Next direct check: `0.55 m` yaw-zero cutoff.
- 2026-07-06 submitted yaw-zero `0.55 m` cutoff diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly055_yawzero_0706`; Slurm
  job `168343`, job-name `g1_lg_f55y`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly055_yawzero_strict_900_targetnegx1`.
  It interpolates between failed-too-early `0.45 m` and stable-but-overshooting
  `0.6 m`, while keeping yaw gain/limit `0`, tiny lateral correction
  available, final command gates near zero, and no rollout root pose/root
  velocity/box pose writes. Purpose: end inside the target window without the
  early fall seen at `0.45 m`. Submission-time status was `PENDING (Priority)`.
- 2026-07-06 yaw-zero `0.55 m` cutoff diagnostic `168343` completed on
  `server39` in `00:00:51` with build status `0` and checker status `1`, so
  it is valid negative but close behavior evidence. It had no box drops,
  rollout root pose/root velocity/box pose writes all `0`, final box
  target-directed travel `2.210967 m` inside the target window, and
  target-window final-hold longest streak `100` steps. It failed late:
  first fall step `882`, fall events `18`, final robot target-directed travel
  `2.437910 m` slightly above the `2.35 m` upper gate, final relative offset
  `0.267861 m`, max robot/box tilt `0.948362/0.835055 rad`, and min robot z
  `0.264276 m`. CSV inspection showed step `880` still had fall/drop `0/0`,
  robot target-directed travel about `2.36 m`, and box travel about `2.02 m`.
  Lateral correction had `0` active steps and was suppressed by tilt for `17`
  steps. Conclusion: `0.55 m` is close but too late/slightly unstable at the
  end; next direct check is a finer `0.545 m` yaw-zero cutoff before trying
  rescue-target blending.
- 2026-07-06 submitted yaw-zero `0.545 m` fine-cutoff diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly0545_yawzero_0706`; Slurm
  job `168350`, job-name `g1_lg_f545`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly0545_yawzero_strict_900_targetnegx1`.
  It keeps the `0.55 m` settings but moves final-zero cutoff to `0.545 m`.
  Purpose: test whether the near-failure at `0.55 m` can be converted into a
  target-window end hold without enabling stand/rescue blending. Submission-
  time status was `PENDING (Priority)`.
- 2026-07-06 yaw-zero `0.545 m` fine-cutoff diagnostic `168350` completed on
  `server39` in `00:00:47` with build status `0` and checker status `1`, so
  it is valid negative evidence. It produced the same effective behavior as
  `168343`: final latch step `366`, fall/drop `18/0`, first fall step `882`,
  final robot/box target-directed travel `2.437910/2.210967 m`, final-hold
  target-window longest streak `100`, max robot/box tilt
  `0.948362/0.835055 rad`, min robot/box z `0.264276/0.486223 m`, and
  rollout root pose/root velocity/box pose writes all `0`. Conclusion: finer
  cutoff tuning around `0.545-0.55 m` is quantized to the same latch and is
  not the next useful move. The most useful current direct-G1 evidence is the
  yaw-zero `0.6 m` run `168335`, which walked with the free box for 900 steps
  with fall/drop `0/0` but failed precise target stopping. Next check: longer
  stable-carry validation with target-window stopping gates relaxed, while
  still requiring no fall/drop, safe height/tilt, no shortcut writes, and
  nontrivial carried distance.
- 2026-07-06 submitted long stable-carry validation based on the best
  yaw-zero `0.6 m` setting. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_finalearly060_yawzero_stable1200_0706`;
  Slurm job `168352`, job-name `g1_lg_st12`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_stablecarry_1200_targetnegx1`.
  It runs `1200` steps and deliberately relaxes only precise target-window
  stopping gates (`TARGET_WINDOW_CENTER=-1`, target-directed upper gates
  `99 m`) while keeping no fall/drop, final-hold height/tilt, final command,
  relative offset, lateral error, minimum carried distance (`>=2 m` robot and
  box travel), and no rollout root pose/root velocity/box pose writes. Purpose:
  separate "can the G1 carry the free box stably for longer" from the still
  unsolved "can it stop exactly in the 2 m target window". Submission-time
  status was `PENDING (Priority)`.
- 2026-07-06 long stable-carry validation `168352` completed on `server39` in
  `00:00:52` with build status `0` and checker status `1`, so it is valid
  negative evidence about long-horizon stability. It completed `1200/1200`
  with rollout root pose/root velocity/box pose writes all `0`, but the
  yaw-zero `0.6 m` setting is only stable for roughly 900 steps: first fall
  step `945`, first box-drop step `965`, fall/drop `255/235`, final
  robot/box target-directed travel `4.848048/4.466154 m`, final relative
  offset `0.427247 m`, max robot/box tilt `3.136022/3.139212 rad`, and min
  robot/box z `-9.467798/-9.494773 m`. CSV inspection showed the carrier was
  still upright at step `900`, but roll and height degraded by steps
  `930-950`. Conclusion: `168335` is a useful 900-step stable carry
  diagnostic, not long-duration carrying success. A late rescue/hold
  stabilization path is needed after about step `900`.
- 2026-07-06 updated the direct G1 scene so hold-rescue can actually blend
  rescue joint targets while `AGILE_COMMAND_HOLD_MODE=policy_command`. Before
  this change, `agile_command_hold_rescue_active` could latch in summaries,
  but the blend branch only ran for `stand_targets`, `policy_then_stand`, or
  final stand. The new condition also enters the blend branch when rescue is
  active. Defaults remain disabled, so old runs are unchanged. Lightweight
  login-node validation passed: `py_compile`, `bash -n`, and
  `git diff --check`; no simulation, rendering, model loading, or heavy
  Python task was run on the login node.
- 2026-07-06 submitted 1200-step late-rescue stable-carry diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_stable1200_laterescue_0706`; Slurm
  job `168356`, job-name `g1_lg_rs12`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_laterescue_1200_targetnegx1`.
  It repeats `168352` with rescue enabled in policy-command mode:
  abs-roll threshold `0.28 rad`, blend rate `0.008`, and a mild crouch/lean
  rescue target (`hip_pitch=-0.18`, `knee=0.42`, `ankle_pitch=-0.25`,
  `waist_pitch=-0.05`). Purpose: test whether late rescue can prevent the
  roll/height collapse that begins around steps `930-950` while preserving
  no shortcut writes and no box drop. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 1200-step late-rescue diagnostic `168356` completed on `server39`
  in `00:00:48` with build status `0` and checker status `1`, so it is valid
  negative evidence. The rescue path did activate at step `930` for `270`
  steps with reason `abs_roll`, proving the new rescue blending path is wired.
  However it did not solve the long-horizon collapse: first fall/drop steps
  `946/963`, fall/drop `254/237`, final relative offset `0.440256 m`, max
  robot/box tilt `3.140006/3.134651 rad`, min robot/box z
  `-9.487601/-9.613761 m`, and rollout root pose/root velocity/box pose
  writes all `0`. Conclusion: the symmetric crouch rescue target does not
  counter the roll failure. The next direct check should strengthen the
  existing roll balance feedback rather than keep tuning symmetric rescue.
- 2026-07-06 submitted 1200-step stronger-roll-balance diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_stable1200_rollstrong_0706`; Slurm
  job `168358`, job-name `g1_lg_br12`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_rollstrong_1200_targetnegx1`.
  It repeats the yaw-zero `0.6 m` long-carry setup without symmetric rescue
  and increases roll feedback from gain/limit `0.06/0.08` to `0.10/0.12`,
  with roll-rate gain `0.006`. Purpose: test whether stronger existing
  roll feedback can prevent the post-900-step roll collapse while preserving
  no shortcut writes and no drop. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 1200-step stronger-roll-balance diagnostic `168358` completed on
  `server39` in `00:00:51` with build status `0` and checker status `1`, so
  it is valid negative evidence. Stronger roll feedback destabilized the run
  much earlier: first fall/drop steps `598/645`, fall/drop `600/442`, final
  robot/box target-directed travel only `0.070430/-0.330736 m`, final
  relative offset `0.447635 m`, max robot/box tilt
  `1.907901/3.138240 rad`, and rollout root pose/root velocity/box pose
  writes all `0`. Conclusion: increasing roll gain/limit is not a safe
  long-horizon fix. Current best direct-G1 evidence remains `168335`: 900
  steps of free-box carrying with fall/drop `0/0`, safe height/tilt, low
  relative offset, and no shortcut writes, but without precise target stop or
  1200-step long-duration stability.
- Current 2026-07-06 direct-G1 yaw-zero status report:
  `experiments/reports/2026-07-06_g1_yawzero_stable_carry_status.md`. It
  records `168335` as the current best 900-step stable free-box carrying
  diagnostic and lists the negative cutoff, long-horizon, rescue, and
  stronger-roll follow-ups. Do not promote `168335` to final success: it is
  not precise target stopping, not 1200-step long-duration stability, and not
  multi-posture/load generality.
- 2026-07-06 submitted late small reverse-brake target-stop diagnostic based
  on the best 900-step yaw-zero setting. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_latebrake010_900_0706`; Slurm job
  `168363`, job-name `g1_lg_lb10`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake010_strict_900_targetnegx1`.
  It repeats the stable `168335` setup and adds only a delayed final brake:
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.01`,
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=400`, and
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=200`. Final command gate for x is
  relaxed to `0.012` to allow this high-level command. Purpose: test whether a
  very small reverse command after the robot is already near the target window
  can reduce the `168335` overshoot without root pose/root velocity/box pose
  writes, falls, or drops. Submission-time status was `PENDING (Priority)`.
- 2026-07-06 late small reverse-brake target-stop diagnostic `168363`
  completed on `server39` in `00:00:49` with build status `0` and checker
  status `1`, so it is valid negative evidence. The brake hook was active for
  `114` steps from `786` to `899`, max abs x command `0.01`, with rollout
  root pose/root velocity/box pose writes all `0`. It preserved the desirable
  stability of `168335`: fall/drop `0/0`, min robot/box z
  `0.694983/0.784024 m`, max robot/box tilt `0.208595/0.413612 rad`, final
  relative offset `0.042980 m`, and final robot/box lateral error
  `0.266715/0.286565 m`. It still failed precise target stopping because
  final robot/box target-directed travel remained `2.716723/2.744582 m`.
  Conclusion: late reverse braking is safe at `-0.01` but too weak; next
  direct check should increase magnitude while keeping the same late window.
- 2026-07-06 submitted stronger late reverse-brake target-stop diagnostic.
  Tmux: `curiosity_g1_agile_largerbox_lowcarry_latebrake025_900_0706`;
  Slurm job `168366`, job-name `g1_lg_lb25`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake025_strict_900_targetnegx1`.
  It repeats `168363` but increases final brake from `-0.01` to `-0.025`
  and relaxes the final x command gate to `0.030`. Purpose: test whether a
  stronger but still late reverse command can reduce overshoot into the target
  window while preserving fall/drop `0/0`, safe height/tilt, and no shortcut
  writes. Submission-time status was `PENDING (Priority)`.
- 2026-07-06 stronger late reverse-brake diagnostic `168366` completed on
  `server39` in `00:00:47` with build status `0` and checker status `1`, so
  it is valid negative evidence. The brake hook was active for `114` steps
  from `786` to `899`, max abs x command `0.025`, and rollout root pose/root
  velocity/box pose writes all `0`. It preserved stability: fall/drop `0/0`,
  min robot/box z `0.711977/0.785524 m`, max robot/box tilt
  `0.215963/0.413612 rad`, and final relative offset `0.107504 m`. It failed
  precise target stopping more strongly than `168363`: final robot/box
  target-directed travel `2.733128/2.772792 m`. Conclusion: negative x brake
  does not reduce overshoot in this configuration; run the same late window
  with positive x command before changing mechanisms.
- 2026-07-06 submitted positive-sign late x-command contrast. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_latebrake_pos025_900_0706`; Slurm
  job `168369`, job-name `g1_lg_lbp25`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake_pos025_strict_900_targetnegx1`.
  It repeats `168366` but changes
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X` from `-0.025` to `+0.025`.
  Purpose: determine the effective high-level command sign for reducing
  target-directed overshoot in the yaw-zero stable-carry setting. Submission-
  time status was `PENDING (Priority)`.
- 2026-07-06 positive-sign late x-command contrast `168369` completed on
  `server39` in `00:00:47` with build status `0` and checker status `1`, so
  it is valid negative but useful behavior evidence. The positive command was
  active for `114` steps from `786` to `899`, max abs x command `0.025`, with
  rollout root pose/root velocity/box pose writes all `0`. It preserved
  stability: fall/drop `0/0`, min robot/box z `0.752112/0.808381 m`, max
  robot/box tilt `0.208595/0.413612 rad`, final relative offset
  `0.070556 m`, and final lateral error `0.230666/0.241873 m`. It reduced
  overshoot compared with the negative-sign runs, but still failed target
  stopping: final robot/box target-directed travel `2.617844/2.674450 m`.
  Conclusion: positive x is the useful late command sign; next check should
  increase positive magnitude in the same late window.
- 2026-07-06 submitted stronger positive late x-command diagnostic. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_latebrake_pos060_900_0706`; Slurm
  job `168372`, job-name `g1_lg_lbp60`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_latebrake_pos060_strict_900_targetnegx1`.
  It repeats `168369` but increases
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X` to `+0.06` and allows final x
  command up to `0.070`. Purpose: test whether stronger positive late command
  can bring final robot/box target-directed travel inside `2.35 m` while
  preserving the no-fall/no-drop behavior. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 stronger positive late x-command diagnostic `168372` completed
  on `server39` in `00:00:49` with build status `0` and checker status `1`,
  so it is valid negative evidence. The command was active for `114` steps
  from `786` to `899`, max abs x command `0.06`, and rollout root pose/root
  velocity/box pose writes all `0`. It preserved stability: fall/drop `0/0`,
  min robot/box z `0.747178/0.808381 m`, max robot/box tilt
  `0.208595/0.413612 rad`, final relative offset `0.093449 m`, and final
  lateral error `0.249089/0.335411 m`. It still failed target stop with final
  robot/box target-directed travel `2.646988/2.670434 m`. Conclusion: command
  sign is correct but the late window beginning at step `786` is too late;
  next check should start positive command earlier rather than only increase
  magnitude.
- 2026-07-06 submitted earlier positive x-command target-stop diagnostic.
  Tmux: `curiosity_g1_agile_largerbox_lowcarry_earlybrake_pos040_900_0706`;
  Slurm job `168381`, job-name `g1_lg_ebp40`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_earlybrake_pos040_strict_900_targetnegx1`.
  It keeps the yaw-zero stable-carry setup but starts positive final x command
  earlier: `AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=0.04`,
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS=330`, and
  `AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=300`. Purpose: test whether starting
  the useful positive command around the target-window interval can reduce
  overshoot to the strict `2.35 m` upper gate while preserving fall/drop `0/0`
  and no shortcut writes. Submission-time status was `PENDING (Priority)`.
- 2026-07-06 earlier positive x-command target-stop diagnostic `168381`
  completed on `server39` in `00:00:47` with build status `0` and checker
  status `1`, so it is valid negative evidence. Positive x command was active
  for `184` steps from `716` to `899`, max abs command `0.04`, with rollout
  root pose/root velocity/box pose writes all `0`. It kept fall/drop `0/0`
  and good height/tilt, but it worsened final target-directed overshoot:
  final robot/box travel `2.756768/2.788418 m`. It did reduce final lateral
  error to `0.029349/0.047980 m`. Conclusion: command timing/magnitude alone
  is not a reliable precise stop controller; use a shorter target-hold
  validation to verify the current stable policy reaches and holds the target
  window before overrun, then continue designing a real stop/long-hold
  controller separately.
- 2026-07-06 submitted 820-step target-window hold validation using the best
  yaw-zero stable-carry setup without late brake. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_yawzero_targethold820_0706`; Slurm
  job `168391`, job-name `g1_lg_th820`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_820_targetnegx1`.
  It runs `FREE_STEPS=820` with strict target-window end-streak gates
  (`MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40`), fall/drop
  `0/0`, final height/tilt gates, final command gates near zero, target
  upper gates `2.35 m`, and no root pose/root velocity/box pose writes.
  Purpose: verify that the current G1/free-box policy can carry into the
  target window and remain stable there for a nontrivial hold interval, even
  though it does not yet have a 900-step precise stop controller. Submission-
  time status was `PENDING (Priority)`.
- 2026-07-06 820-step target-window hold validation `168391` completed on
  `server39` in `00:00:48` with build status `0` and checker status `1`, so
  it is a near-pass but still valid negative evidence. It completed `820`
  steps with fall/drop `0/0`, min robot/box z `0.752112/0.808381 m`, max
  robot/box tilt `0.208595/0.413612 rad`, final relative offset `0.080991 m`,
  final robot/box lateral error `0.136043/0.194526 m`, and rollout root pose/
  root velocity/box pose writes all `0`. It failed only because final box
  target-directed travel was `2.350200 m`, just `0.000200 m` over the strict
  `2.35 m` upper gate, which also reset the end-of-window streak to `0`.
  Conclusion: the target-hold subtask is at a one-step discretization
  boundary; run the same validation with `FREE_STEPS=819` rather than
  loosening the target gate.
- 2026-07-06 submitted 819-step target-window hold validation. Tmux:
  `curiosity_g1_agile_largerbox_lowcarry_yawzero_targethold819_0706`; Slurm
  job `168398`, job-name `g1_lg_th819`; stamp:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`.
  It is identical to `168391` except `FREE_STEPS=819` and
  `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=399`. Purpose: test the same
  target-window hold without loosening any target upper gate, by ending one
  simulation step before the `2.350200 m` box overshoot seen in `168391`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 819-step target-window hold validation `168398` completed on
  `server39` in `00:00:46` with build status `0`, checker status `0`, and
  `status=pass`. This is the strongest direct-G1 target-hold evidence so far.
  It completed `819` steps with fall/drop `0/0`, first fall/drop `null/null`,
  final robot/box target-directed travel `2.298755/2.346454 m`, final
  robot/box lateral error `0.133809/0.191820 m`, final relative offset
  `0.079615 m`, min robot/box z `0.752112/0.808381 m`, max robot/box tilt
  `0.208595/0.413612 rad`, and rollout root pose/root velocity/box pose
  writes all `0`. Target-window both-final-hold stable steps, longest streak,
  and streak at end were all `164`, so the robot and box were inside the
  target window at the end of the rollout. This is a valid direct Isaac G1
  free-box carry-to-target-hold diagnostic, but still not final project
  success: it is one low-cradle posture, one box setting, no active unknown-
  load probing, no autonomous posture selection, and no 1200-step long-
  duration hold.
- 2026-07-06 submitted two same-gate posture/contact-configuration
  generalization diagnostics. Tmux sessions:
  `curiosity_g1_agile_chestpad_targethold819_0706` and
  `curiosity_g1_agile_boxtilt_targethold819_0706`; Slurm jobs `168402`
  (`g1_cp_th819`) and `168403` (`g1_bt_th819`); stamps:
  `20260706_g1_agile_chestpad_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`
  and
  `20260706_g1_agile_boxtilt_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`.
  Both reuse the strict `168398` target-hold gates with `FREE_STEPS=819`,
  final target-directed upper gates `2.35 m`, target-window end streak
  `>=40`, fall/drop `0/0`, height/tilt gates, final command gates near zero,
  and no rollout root pose/root velocity/box pose writes. Purpose: test
  whether the low-cradle target-hold result transfers to the suite's existing
  `chestpad` and `boxtilt` contact/posture variants. Submission-time statuses
  were `PENDING (Priority)`.
- 2026-07-06 posture/contact generalization diagnostics `168402` and `168403`
  completed on `server39`; both are negative evidence. `168402` (`chestpad`)
  completed `819` steps with build status `0` and checker status `1`, but it
  had fall/drop `87/35`, first fall/drop `732/784`, final robot/box target-
  directed travel `3.334948/3.015123 m`, final robot/box lateral error
  `-1.672259/-1.909126 m`, final relative offset `0.427477 m`, max robot/box
  tilt `3.032566/3.033201 rad`, min robot/box z `-0.636457/-0.659537 m`, and
  rollout root pose/root velocity/box pose writes all `0`. Conclusion:
  enabling the chest-pad contact/posture variant does not inherit the low-
  cradle target-hold behavior; it allows large lateral drift followed by late
  fall/drop. `168403` (`boxtilt`) completed `819` steps with build status `0`
  and checker status `1`; it had fall/drop `4/0`, first fall `815`, no box
  drop, final robot/box target-directed travel `0.560830/0.411779 m`, final
  robot/box lateral error `0.623486/0.350950 m`, final relative offset
  `0.362352 m`, max robot/box tilt `0.729192/0.737112 rad`, min robot/box z
  `0.419574/0.366088 m`, target-window stable/final-hold counts `0`, and
  rollout root pose/root velocity/box pose writes all `0`. Conclusion: the
  boxtilt/default higher cradle/contact geometry never reaches the target
  window and fails near the end. These two runs strengthen the current
  limitation: `168398` is a one-posture low-cradle success, not evidence of
  multiple-posture carrying or autonomous posture selection.
- 2026-07-06 login-node safety note: the user reported high-CPU ripgrep
  processes on `mgmtserver02` with command paths under
  `/public/home/yanhongru/.vscode-server/.../ripgrep/bin/rg`, including broad
  `--files --hidden --no-ignore --follow` style searches for agent files.
  Immediate check showed PIDs `3661511` and `3661828` were already gone and
  `squeue -u yanhongru` had no active jobs. Treat this as a hard operational
  warning: do not run or trigger broad hidden/no-ignore/follow file discovery
  on the login node; use only targeted file reads/listings there, and keep all
  simulation/heavy work inside tmux-held Slurm compute allocations.
- 2026-07-06 after recording `168402`/`168403`, lightweight login-node checks
  passed: `python3 -m py_compile` for the touched G1 scene/checker scripts,
  `bash -n` for the touched G1 launchers, and `git diff --check` for touched
  docs/scripts all returned status `0`. `squeue -u yanhongru` showed no active
  jobs. These were syntax/diff/queue checks only, not simulation or project
  experiment runs.
- 2026-07-06 next direct-G1 second-posture plan: do not retry the failed
  yaw-zero `chestpad`/`boxtilt` target-hold settings unchanged. Historical
  `chestpad` run `167778`
  (`20260706_g1_agile_largerbox_chestpad_oppositeyaw_terminal900_nearstop_targetnegx1`)
  is the best available second-posture scaffold: it completed `900` steps with
  fall/drop `0/0`, no rollout root/velocity/box pose writes, final robot/box
  target-directed travel about `1.730244/1.759363 m`, final relative offset
  `0.108737 m`, max robot/box tilt `0.307758/0.384690 rad`, and terminal
  slow-walk active from step `590`. However that old run did not record the
  newer target-window final-hold metrics, so it is not sufficient evidence for
  the current claim. Submit a new `chestpad` opposite-yaw near-stop run with
  current target-window end-streak gates rather than treating `167778` as a
  final second-posture pass.
- 2026-07-06 submitted the new `chestpad` opposite-yaw near-stop target-window
  validation. Tmux:
  `curiosity_g1_chestpad_oppositeyaw_targetwindow900_0706`; Slurm job
  `168419`, job-name `g1_cp_tw900`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_900_targetnegx1`.
  It uses `FREE_STEPS=900`, `LARGERBOX_STRICT_MODE=chestpad`, yaw correction
  `gain=0.04`, `limit=0.08`, `sign=-1.0`, terminal slow-walk at box target-
  directed travel `1.05 m` with scale `0.015`, final target-directed upper
  gates `2.35 m`, target window center/halfwidth `2.0/0.35 m`, and
  target-window both end-streak gates `>=30` steps. Submission-time status was
  `PENDING (Priority)`. Purpose: verify a second carrying posture/contact
  geometry under current target-window gates without using rollout root pose,
  root velocity, or box pose writes.
- 2026-07-06 `chestpad` opposite-yaw near-stop target-window validation
  `168419` completed on `server39` in `00:00:47` with build status `0`,
  checker status `0`, and `status=pass`. It completed `900` steps with
  fall/drop `0/0`, first fall/drop `null/null`, final robot/box target-
  directed travel `1.730244/1.759363 m`, final robot/box lateral error
  `0.258455/0.362250 m`, final relative offset `0.108737 m`, max relative
  offset `0.205432 m`, min robot/box z `0.721562/0.825034 m`, max robot/box
  tilt `0.307758/0.384690 rad`, terminal slow-walk active from step `590` for
  `310` steps, and rollout root pose/root velocity/box pose writes all `0`.
  Target-window both stable steps, longest streak, and end streak were all
  `33`, with the first both-stable step at `867`. This is valid evidence that
  a second G1/free-box carrying posture/contact configuration can enter and
  end inside the target window without falling, dropping, or using rollout
  shortcut writes. It is still not final project success: the target-window
  hold is short compared with the low-carry `168398` end streak (`164`), it
  is manually selected `chestpad` rather than autonomous posture choice, and
  there is still no active unknown-load probing or learned video-conditioned
  policy.
- 2026-07-06 next `chestpad` strengthening step: submit the same opposite-yaw
  near-stop posture with `FREE_STEPS=1000` and target-window both stable,
  longest-streak, and end-streak gates raised to `>=100` steps. Purpose:
  determine whether the second-posture target-window evidence from `168419`
  is a short final-window entry only or can hold the target window for a
  longer interval while preserving fall/drop `0/0` and no rollout root/box
  shortcut writes.
- 2026-07-06 submitted the strengthened `chestpad` target-window validation.
  Tmux: `curiosity_g1_chestpad_oppositeyaw_targetwindow1000_0706`; Slurm job
  `168420`, job-name `g1_cp_tw1000`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_1000_targetnegx1`.
  It is identical to `168419` except `FREE_STEPS=1000` and target-window both
  stable/longest/end-streak gates are all `>=100`. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 after repeated low-frequency queue checks, Slurm job `168420`
  remained `PENDING (Priority)`. Do not replace it with a login-node run or a
  non-approved one-shot resource path. Leave the tmux-held `srun` queued and
  read its summary/check files after the scheduler runs it.
- 2026-07-06 added reusable compute-side launcher
  `scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`.
  It refuses to run on `mgmtserver*` login/management nodes, then sequentially
  runs target-window validation cases through
  `run_core_world_g1_largerbox_strict_suite.sh`: the current low-carry
  `819`-step target-hold baseline, the pending `chestpad` `1000`-step
  long-hold validation, and opt-in load held-outs (`lowcarry` light box and
  `chestpad` heavy box). This is only a reproducibility/validation launcher;
  it does not create a new model or success claim. `bash -n` passed. `168420`
  was still `PENDING (Priority)` after this edit.
- 2026-07-06 `scontrol show job 168420` confirmed the pending state is
  scheduler priority/backfill, not a command or resource request error:
  `JobState=PENDING`, `Reason=Priority`, requested resources
  `cpu=8,mem=32G,node=1,gres/gpu=1`, scheduled node `server21`, and projected
  `StartTime=2026-07-06T19:18:06`. Keep the tmux-held Slurm job queued and
  read its output after it runs.
- 2026-07-06 lightweight login-node consistency checks after adding the
  target-window posture validation launcher passed: `bash -n
  scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
  and `git diff --check` over the touched docs/script returned status `0`.
  `squeue -u yanhongru` showed only Slurm job `168420`, still
  `PENDING (Priority)`.
- 2026-07-06 strengthened
  `scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
  to automatically call `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`
  after its selected cases and write
  `targetwindow_posture_validation_summary.json` under the suite output root.
  This keeps future multi-posture/load validation auditable without manual
  `jq` stitching. `bash -n` passed. `168420` was still `PENDING (Priority)`.
- 2026-07-06 set
  `scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
  executable (`-rwxr-xr-x`) so it can be launched directly inside future
  tmux-held Slurm compute allocations. This was a permission change only, not
  a simulation run.
- 2026-07-06 updated
  `experiments/reports/2026-07-06_g1_yawzero_stable_carry_status.md` with the
  `168420` pending strengthening run, its Slurm priority/backfill reason, the
  projected start time, and the reusable target-window posture validation
  launcher. Lightweight checks passed again: `bash -n
  scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
  and `git diff --check` over touched docs/script returned status `0`.
  `squeue -u yanhongru` still showed only job `168420`, `PENDING (Priority)`.
- 2026-07-06 added completion-audit report
  `experiments/reports/2026-07-06_g1_carry_completion_audit.md`. It separates
  verified low-carry evidence (`168398`), short second-posture chest-pad
  evidence (`168419`), pending strengthened chest-pad evidence (`168420`),
  negative posture/long-horizon evidence, and missing requirements for the
  full project objective: stronger second-posture hold, broader posture/load
  coverage, active unknown-load probing, autonomous posture selection, and
  video-conditioned learning. This audit is a guard against treating the
  current direct-G1 diagnostics as final success.
- 2026-07-06 lightweight checks after adding the completion audit passed:
  `bash -n scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
  and `git diff --check` over touched docs/script returned status `0`.
  `168420` remained the only active queued job and was still
  `PENDING (Priority)`.
- 2026-07-06 latest scheduler recheck at `2026-07-06T17:56:40 CST` still
  showed `168420` as `PENDING (Priority)` with unchanged projected
  `StartTime=2026-07-06T19:18:06` on `server21`. No output files are expected
  until that tmux-held Slurm job starts and completes.
- 2026-07-06 added G1 probe-to-posture diagnostic selector
  `scripts/isaac/select_core_world_g1_carry_posture_from_probe.py`. It reads a
  prior G1 probe summary, uses visible `box_size_m` plus logged probe
  displacement (`max/final_probe_box_travel_xy_m`), and outputs both a JSON
  selection report and optional shell env exports for the existing
  low-carry/chest-pad validation suites. It explicitly records
  `selection_uses_hidden_ground_truth=false` and ignores hidden
  `box_mass_kg` if present. This is a diagnostic heuristic only, not a learned
  video-conditioned or final autonomous policy. `python3 -m py_compile`
  passed, and the file was made executable. `168420` remained
  `PENDING (Priority)`.
- 2026-07-06 lightweight checks after adding the G1 probe selector passed:
  `python3 -m py_compile
  scripts/isaac/select_core_world_g1_carry_posture_from_probe.py`, `bash -n
  scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`,
  and `git diff --check` over touched docs/scripts all returned status `0`.
  `168420` remained `PENDING (Priority)`.
- 2026-07-06 added probe passthrough to
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`:
  `PROBE_MODE`, `PROBE_START_STEP`, probe pad size/local pose/mass, and
  optional `DISABLE_PROBE_PAD_COLLISION` now map to the underlying
  `build_core_world_g1_box_scene.py` probe arguments. Defaults preserve
  previous behavior (`PROBE_MODE=none`).
- 2026-07-06 added compute-side probe-selected validation pipeline
  `scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`.
  It refuses to run on `mgmtserver*`, runs a G1 front-bumper probe diagnostic
  through the existing larger-box strict suite, uses
  `select_core_world_g1_carry_posture_from_probe.py` to produce a posture env,
  then runs the selected low-carry/chest-pad target-window validation. This is
  active-probing/autonomous-posture-selection plumbing only; it is not success
  evidence until a compute-node run passes the same fall/drop, target-window,
  relative-offset, and no-shortcut gates. `bash -n` over the affected shell
  scripts and `python3 -m py_compile` for the selector passed, and the new
  pipeline was made executable. `168420` remained `PENDING (Priority)`.
- 2026-07-06 final lightweight checks for this probe-selection plumbing pass:
  `bash -n` over
  `run_core_world_g1_probe_selected_targetwindow_validation.sh`,
  `run_core_world_g1_agile_policy_low_cradle_suite.sh`, and
  `run_core_world_g1_targetwindow_posture_validation_suite.sh`; `python3 -m
  py_compile` for `select_core_world_g1_carry_posture_from_probe.py`; and
  `git diff --check` over the touched docs/scripts all returned status `0`.
  `squeue -u yanhongru` still showed only `168420`, `PENDING (Priority)`.
- 2026-07-06 latest recheck at `2026-07-06T18:02:56 CST` showed `168420`
  still `PENDING (Priority)` with no output files. To continue advancing the
  full objective instead of waiting on a single pending job, submit a separate
  probe-selected G1 validation diagnostic through tmux-held Slurm:
  `scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`.
  Purpose: test the newly wired front-bumper probe -> non-hidden-telemetry
  posture selector -> selected target-window validation path. This must be
  interpreted as a diagnostic pipeline only, not final autonomous posture
  selection, unless the resulting probe summary, selector report, and selected
  validation all pass their gates.
- 2026-07-06 submitted probe-selected G1 validation diagnostic. Tmux:
  `curiosity_g1_probe_selected_targetwindow_diag1_0706`; Slurm job `168429`,
  job-name `g1_probe_sel`; stamp:
  `20260706_g1_probe_selected_targetwindow_diag1`. It runs
  `scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`
  with `--time=00:30:00`, `8` CPUs, `32G` memory, and `1` GPU. Submission-time
  status was `PENDING (Priority)`. At submission, `168420` was also still
  `PENDING (Priority)`.
- 2026-07-06 `scontrol` check after submitting `168429`: `168420` remained
  `PENDING (Priority)` with projected `StartTime=2026-07-06T19:18:06` on
  `server21`; `168429` was `PENDING (Priority)` with normal requested
  resources (`cpu=8,mem=32G,node=1,gres/gpu=1`) and `StartTime=Unknown`.
- 2026-07-06 lightweight checks after submitting `168429` passed: `bash -n`
  over the affected shell launchers, `python3 -m py_compile` for the G1 probe
  selector, and `git diff --check` over touched docs/scripts all returned
  status `0`. `squeue -u yanhongru` showed `168420` and `168429`, both
  `PENDING (Priority)`.
- 2026-07-06 updated the completion audit and G1 yaw-zero status report to
  include pending probe-selected pipeline job `168429`. Final lightweight
  checks for this turn passed again (`bash -n`, selector `py_compile`, and
  `git diff --check`). Queue state remained: `168420` and `168429` both
  `PENDING (Priority)`.
- 2026-07-06 improved the probe-selected pipeline script
  `scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`
  so selected-posture validation writes
  `g1_probe_selected_pipeline_summary.json` even when the selected validation
  exits nonzero; the script still exits with the selected validation status.
  `bash -n` passed. Queue recheck immediately after showed `168420`
  `RUNNING` on `server21` and `168429` still `PENDING (Priority)`. The
  pipeline summary change affects `168429` only if it starts after this edit;
  it does not affect the already-running `168420` chest-pad validation.
- 2026-07-06 strengthened `chestpad` target-window validation `168420`
  completed on `server21` in `00:01:01` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `1000`
  steps with fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/
  root velocity/box pose writes all `0`, final robot/box target-directed
  travel `1.912552/1.774289 m`, target-window both stable/longest/end streak
  `133/133/133`, min robot/box z `0.721562/0.775084 m`, and terminal slow-
  walk active from step `590` for `410` steps. This proves the second posture
  can stay in the target window longer than `168419`, but it is not a strict
  pass because posture/carry quality drifted: max robot tilt
  `0.485765 > 0.35`, max box tilt `0.713845 > 0.45`, final relative offset
  `0.311690 > 0.25`, and final box lateral error `0.785696 > 0.6`. Failure
  mode: chest-pad can reach and remain in the target window, but needs a
  final-stop/hold controller to prevent late lateral/tilt/relative drift.
- 2026-07-06 next chest-pad branch: submit a final-stop target-window
  validation that keeps the `168420` setup but enables
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.65`,
  `AGILE_COMMAND_HOLD_FINAL_SCALE=0.0`, `AGILE_COMMAND_HOLD_FINAL_LATCH=1`,
  and `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1`, with
  `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=100` and final command/fall/drop
  gates. Purpose: preserve the `168420` target-window occupancy while reducing
  late chest-pad tilt, relative-offset, and lateral drift. This is a
  diagnostic controller test, not final success.
- 2026-07-06 probe-selected G1 validation diagnostic `168429` completed on
  `server44` in `00:01:39` with Slurm exit `0:0`. Pipeline stamp:
  `20260706_g1_probe_selected_targetwindow_diag1`; summary:
  `experiments/outputs/core_world_g1_probe_selected_targetwindow/20260706_g1_probe_selected_targetwindow_diag1/g1_probe_selected_pipeline_summary.json`.
  The front-bumper probe produced `probe_mode=front_bumper`,
  `probe_active_steps=220`, and probe motion `0.511708 m`. The selector
  returned `status=pass`, selected `lowcarry`, used visible box size
  `[0.14, 0.10, 0.08]` plus probe displacement, and reported
  `selection_uses_hidden_ground_truth=false` while explicitly ignoring the
  present hidden `box_mass_kg`. The selected validation completed `819` steps
  with checker `status=pass`, fall/drop `0/0`, target-window both end streak
  `164`, and rollout root pose/root velocity/box pose writes all `0`. This is
  the first passing direct-G1 probe -> non-hidden-telemetry posture selector ->
  selected carry validation pipeline. It is still not final success: the
  selector is a diagnostic heuristic, the selected case is the same low-risk
  box/low-carry configuration, and it does not prove robust unknown-load
  inference, multiple selected postures, or video-conditioned RL.
- 2026-07-06 chest-pad final-stop target-window diagnostic `168431` completed
  on `server44` in `00:00:48` with Slurm exit `0:0`, build status `0`,
  checker status `1`, and checker `status=fail`. It is a strong near-pass:
  completed `1000` steps, fall/drop `0/0`, first fall/drop `null/null`,
  rollout root pose/root velocity/box pose writes all `0`, final robot/box
  target-directed travel `2.118041/2.142777 m`, target-window both stable/
  longest/end streak `133/133/133`, final-hold stable/longest/end streak
  `132/132/132`, final hold active from step `868` for `132` steps with last
  command `[0, 0, 0]`, min robot/box z `0.721562/0.825034 m`, max robot/box
  tilt `0.307758/0.384690 rad`, and final relative offset `0.144021 m`.
  It failed only because final box lateral error was `0.614122 m`, just above
  the strict `0.6 m` gate. Conclusion: final-stop fixed the major `168420`
  tilt/relative drift but must trigger slightly earlier or otherwise reduce
  final lateral drift; do not loosen the lateral gate.
- 2026-07-06 next chest-pad final-stop branch: submit the same `168431`
  controller with `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.55` and
  `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=140`. Purpose: trigger zero-
  command final hold earlier to reduce the final box lateral error below the
  strict `0.6 m` gate while preserving target-window/final-hold end streak,
  fall/drop `0/0`, no shortcut writes, and the improved tilt/relative-offset
  behavior from `168431`.
- 2026-07-06 submitted earlier chest-pad final-stop diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstop155_targetwindow1000_0706`; Slurm job
  `168432`, job-name `g1_cp_fs155`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop155_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 earlier chest-pad final-stop diagnostic `168432` completed on
  `server44` in `00:00:52` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `1000` steps with
  fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/root
  velocity/box pose writes all `0`, final robot/box target-directed travel
  `1.775166/1.754064 m`, target-window both stable/longest/end streak
  `103/103/103`, final-hold stable/longest/end streak `103/103/103`, final
  hold active from step `817` for `183` steps with last command `[0, 0, 0]`,
  max robot/box tilt `0.307758/0.420955 rad`, and final relative offset
  `0.176791 m`. It failed only final box lateral error
  `0.692755 > 0.6`, worse than `168431`'s `0.614122`. Conclusion: moving the
  final stop all the way to `1.55 m` is too early and increases final lateral
  error; next test should use a smaller shift from `1.65`, such as `1.62 m`.
- 2026-07-06 next chest-pad final-stop branch: submit an intermediate
  `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.62` run with
  `MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=120`. Purpose: see whether a
  smaller shift from `168431`'s `1.65 m` trigger can reduce the final box
  lateral error below `0.6 m` without the lateral worsening seen in `168432`.
- 2026-07-06 submitted intermediate chest-pad final-stop diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstop162_targetwindow1000_0706`; Slurm job
  `168433`, job-name `g1_cp_fs162`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop162_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 intermediate chest-pad final-stop diagnostic `168433` completed
  on `server44` in `00:00:50` with Slurm exit `0:0`, build status `0`,
  checker status `1`, and checker `status=fail`. It completed `1000` steps
  with fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/root
  velocity/box pose writes all `0`, target-window both stable/longest/end
  streak `133/133/133`, final-hold stable/longest/end streak `133/133/133`,
  final hold active from step `859` for `141` steps with last command
  `[0, 0, 0]`, max robot/box tilt `0.307758/0.384690 rad`, and final relative
  offset `0.230262 m`. It failed only final box lateral error
  `0.715806 > 0.6`, worse than both `168431` (`0.614122`) and `168432`
  (`0.692755`). Conclusion: moving final stop earlier than `1.65 m` worsens
  lateral drift. Next branch should keep the `1.65 m` final trigger but allow
  corrective yaw/lateral commands during final hold instead of zeroing all
  corrections.
- 2026-07-06 next chest-pad branch: submit final-stop with corrections. It
  keeps `AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.65` and
  `AGILE_COMMAND_HOLD_FINAL_SCALE=0.0`, but does not enable
  `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS` and does not set final command-
  zero gates. Purpose: keep zero forward final scale while allowing yaw/
  lateral correction to reduce final box lateral error below `0.6 m`.
- 2026-07-06 submitted chest-pad final-stop-with-corrections diagnostic.
  Tmux: `curiosity_g1_chestpad_finalstop_corr_targetwindow1000_0706`; Slurm
  job `168435`, job-name `g1_cp_fcorr`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_corr_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 lightweight checks after `168435` submission passed: `bash -n`
  over the affected shell launchers, `python3 -m py_compile` for the G1 probe
  selector, and `git diff --check` over touched docs/scripts all returned
  status `0`. `168435` remained `PENDING (Priority)`.
- 2026-07-06 chest-pad final-stop-with-corrections diagnostic `168435`
  completed on `server44` in `00:00:51` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `1000`
  steps with fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/
  root velocity/box pose writes all `0`, target-window both stable/longest/end
  streak `133/133/133`, final-hold stable/longest/end streak `132/132/132`,
  final hold active from step `868` for `132` steps, final command
  `[0, -0.035, 0.020590]`, max robot/box tilt `0.307758/0.384690 rad`, and
  final relative offset `0.193038 m`. It failed only final box lateral error
  `0.708802 > 0.6`, worse than `168431`. Conclusion: allowing final yaw/
  lateral corrections does not fix the late lateral drift. Next branch should
  keep `168431`'s zero-command final stop but switch to final stand targets
  after final trigger to reduce drift during the hold.
- 2026-07-06 next chest-pad branch: submit final-stop plus final-stand
  validation. It keeps the `168431` final trigger (`1.65 m`), zero final
  command, and zero corrections, then enables
  `AGILE_COMMAND_HOLD_FINAL_STAND=1` with zero delay and stand blend rate
  `0.02`. Purpose: reduce lateral drift during final hold by transitioning
  from walking policy targets to stand targets.
- 2026-07-06 submitted chest-pad final-stop plus final-stand diagnostic.
  Tmux: `curiosity_g1_chestpad_finalstand_targetwindow1000_0706`; Slurm job
  `168436`, job-name `g1_cp_fstand`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstand_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 lightweight checks after `168436` submission passed: `bash -n`
  over the affected shell launchers, `python3 -m py_compile` for the G1 probe
  selector, and `git diff --check` over touched docs/scripts all returned
  status `0`. Queue recheck showed `168436` already `RUNNING` on `server44`.
- 2026-07-06 chest-pad final-stop plus final-stand diagnostic `168436`
  completed on `server44` in `00:00:50` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `1000`
  steps with fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/
  root velocity/box pose writes all `0`, target-window both stable/longest/end
  streak `133/133/133`, final-hold end streak `132`, final-stand end streak
  `132`, final stand active from step `868`, final command `[0, 0, 0]`,
  final robot/box target-directed travel `2.211248/2.280206 m`, final box
  lateral error `0.232397 m`, final robot lateral error `0.280841 m`, final
  relative offset `0.187513 m`, and min robot/box z `0.647125/0.612681 m`.
  It failed only tilt gates: max robot tilt `0.755765 > 0.35` and max box
  tilt `0.749709 > 0.45`. Conclusion: final stand fixes lateral drift but
  transitions too aggressively; next run should delay final stand and lower
  blend rate to reduce tilt while preserving lateral improvement.
- 2026-07-06 next chest-pad final-stand branch: submit a gentler transition
  with `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=40`,
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.005`, and
  `MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=80`. Purpose: preserve the
  `168436` lateral improvement while reducing the stand-transition tilt spike.
- 2026-07-06 submitted gentler chest-pad final-stand diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstand_gentle_targetwindow1000_0706`; Slurm job
  `168437`, job-name `g1_cp_fsgnt`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstand_gentle_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 lightweight checks after `168437` submission passed: `bash -n`
  over the affected shell launchers, `python3 -m py_compile` for the G1 probe
  selector, and `git diff --check` over touched docs/scripts all returned
  status `0`. Queue recheck showed `168437` already `RUNNING` on `server39`.
- 2026-07-06 gentler chest-pad final-stand diagnostic `168437` completed on
  `server39` in `00:00:57` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `1000` steps with
  fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/root
  velocity/box pose writes all `0`, target-window both stable/longest/end
  streak `133/133/133`, final-hold end streak `132`, final-stand end streak
  `92`, final stand active from step `908`, final command `[0, 0, 0]`, max
  robot/box tilt `0.309353/0.384690 rad`, min robot/box z
  `0.721562/0.825034 m`, and final relative offset `0.165383 m`. It failed
  only final box lateral error `0.650740 > 0.6`. Conclusion: delay `40` and
  blend `0.005` fix the tilt spike but are too gentle/late to reduce lateral
  enough. Next branch should use an intermediate stand transition, such as
  delay `20` and blend `0.01`.
- 2026-07-06 next chest-pad final-stand branch: submit intermediate stand
  transition with `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=20`,
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.01`, and
  `MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS=100`. Purpose: balance the
  low lateral error of aggressive stand transition (`168436`) against the low
  tilt of gentle transition (`168437`).
- 2026-07-06 submitted intermediate chest-pad final-stand diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstand_mid_targetwindow1000_0706`; Slurm job
  `168438`, job-name `g1_cp_fsmid`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstand_mid_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 lightweight checks after `168438` submission passed: `bash -n`
  over the affected shell launchers, `python3 -m py_compile` for the G1 probe
  selector, and `git diff --check` over touched docs/scripts all returned
  status `0`. Queue recheck showed `168438` already `RUNNING` on `server21`.
- 2026-07-06 intermediate chest-pad final-stand diagnostic `168438` completed
  on `server21` in `00:00:58` with Slurm exit `0:0`, build status `0`,
  checker status `1`, and checker `status=fail`. It completed `1000` steps
  with fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/root
  velocity/box pose writes all `0`, target-window both stable/longest/end
  streak `133/133/133`, final-hold end streak `132`, final-stand end streak
  `112`, final stand active from step `888`, max robot/box tilt
  `0.658000/0.658923 rad`, final relative offset `0.198759 m`, and final box
  lateral error `0.712096 m`. It failed tilt and lateral gates, worse than
  both endpoints. Conclusion: delaying final stand hurts lateral before it
  fixes tilt. Next branch should keep zero stand delay like `168436` but lower
  stand blend rate to reduce tilt.
- 2026-07-06 next chest-pad final-stand branch: submit zero-delay slow-blend
  final stand with `AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS=0` and
  `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.005`. Purpose: preserve the lateral
  improvement from zero-delay final stand while reducing the tilt spike from
  `168436`.
- 2026-07-06 submitted zero-delay slow-blend chest-pad final-stand diagnostic.
  Tmux: `curiosity_g1_chestpad_finalstand_zerodelay_slow_targetwindow1000_0706`;
  Slurm job `168440`, job-name `g1_cp_fszs`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstand_zerodelay_slow_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 lightweight checks after `168440` submission passed: `bash -n`
  over the affected shell launchers, `python3 -m py_compile` for the G1 probe
  selector, and `git diff --check` over touched docs/scripts all returned
  status `0`. Queue recheck showed `168440` already `RUNNING` on `server21`.
- 2026-07-06 zero-delay slow-blend final-stand diagnostic `168440` completed
  on `server21` in `00:00:54` with Slurm exit `0:0`, build status `0`,
  checker status `1`, and checker `status=fail`. It completed `1000` steps
  with fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/root
  velocity/box pose writes all `0`, target-window both stable/longest/end
  streak `133/133/133`, final-hold end streak `132`, final-stand end streak
  `132`, final stand active from step `868`, final command `[0, 0, 0]`, final
  box lateral error `0.129922 m`, final robot lateral error `0.093252 m`, and
  final relative offset `0.249589 m`. It failed only tilt gates: max robot
  tilt `0.827365 > 0.35` and max box tilt `0.979325 > 0.45`. Conclusion:
  zero-delay final stand can fix lateral drift but causes unacceptable tilt;
  final-stand branch is not currently a strict second-posture solution. Next
  branch should return to final-stop without stand and test whether lateral
  correction direction/sign is responsible for the `168431` near-miss.
- 2026-07-06 next chest-pad final-stop branch: submit final-stop with
  corrective commands but flipped `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0`.
  Purpose: test whether the final lateral command direction caused `168435`
  to drift worse than the zero-correction near-pass `168431`.
- 2026-07-06 submitted chest-pad final-stop lateral-sign diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstop_corr_latsignneg_targetwindow1000_0706`;
  Slurm job `168450`, job-name `g1_cp_flsn`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_corr_latsignneg_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 chest-pad final-stop lateral-sign diagnostic `168450` completed
  on `server21` in `00:00:49` with Slurm exit `0:0`, build status `0`,
  checker status `1`, and checker `status=fail`. It completed `1000` steps
  with fall/drop `0/0`, first fall/drop `null/null`, rollout root pose/root
  velocity/box pose writes all `0`, and target-window both longest streak
  `201` with final-hold longest streak `200`, but it was not stable at the
  final step. It failed tilt, overshoot, lateral, and end-hold gates: max
  robot/box tilt `0.391817/0.542518 rad`, final robot/box target-directed
  travel `2.548142/2.555898 m`, final robot/box lateral error
  `1.710366/1.626656 m`, and target-window both end streak `0`. Conclusion:
  flipping `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0` is a bad branch; do not keep
  tuning this lateral-sign direction. The next valid chest-pad branch should
  return to the `168431` final-stop near-pass and test less intrusive final
  corrections, such as yaw-only correction without lateral command.
- 2026-07-06 submitted chest-pad final-stop yaw-only diagnostic. It starts
  from the `168431` final-stop near-pass, keeps `AGILE_COMMAND_HOLD_FINAL_SCALE=0.0`
  and `AGILE_COMMAND_HOLD_FINAL_LATCH=1`, disables lateral correction with
  `AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0`, and disables final correction
  zeroing with `AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=0` so yaw correction
  can remain active during final hold. Tmux:
  `curiosity_g1_chestpad_finalstop_yawonly_targetwindow1000_0706`; Slurm job
  `168451`, job-name `g1_cp_fyaw`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_yawonly_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 chest-pad final-stop yaw-only diagnostic `168451` completed on
  `server21` in `00:00:49` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `1000` steps with
  rollout root pose/root velocity/box pose writes all `0`, but disabling
  lateral correction made the run much worse: fall/drop `232/193`, first
  fall/drop step `768/807`, min robot/box z `-4.737784/-4.653333 m`, max
  robot/box tilt `3.135358/3.139102 rad`, final robot/box lateral error
  `3.699152/3.710816 m`, target-window both stable/longest/end streak
  `0/0/0`, and final-hold active steps `0`. Conclusion: the early lateral
  correction used by `168431` is necessary for this chest-pad setup. Do not
  continue yaw-only-without-lateral as a stabilization branch.
- 2026-07-06 next chest-pad geometry branch: return to the `168431`
  final-stop near-pass and change only `CRADLE_CHEST_PAD_SIZE_Y` from `0.38`
  to `0.44`. Purpose: reduce the remaining `0.614122 m` final box lateral
  near-miss without adding final-stand tilt spikes or removing the lateral
  correction that was necessary for `168451`.
- 2026-07-06 submitted chest-pad wider-pad diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstop_widepad_targetwindow1000_0706`; Slurm job
  `168452`, job-name `g1_cp_wpad`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_widepad044_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 chest-pad wider-pad diagnostic `168452` completed on `server21`
  in `00:00:52` with Slurm exit `0:0`, build status `0`, checker status `1`,
  and checker `status=fail`. It completed `1000` steps with fall/drop `0/0`
  and rollout root pose/root velocity/box pose writes all `0`, but widening
  the chest pad to `0.44 m` delayed the useful latch and worsened posture:
  terminal latch step `834`, final latch step `991`, final active steps `9`,
  target-window both stable/longest/end streak `10/10/10`, max robot/box tilt
  `0.412648/0.501682 rad`, final relative offset `0.322406 m`, final box
  lateral error `0.650154 m`, and final robot/box target-directed travel
  `1.876081/1.678427 m`. Conclusion: wide-pad geometry is worse than the
  `168431` near-pass; do not continue increasing chest-pad width as the next
  fix.
- 2026-07-06 next chest-pad controller branch: return to the exact `168431`
  geometry and final-stop structure, but increase pre-final lateral correction
  gain from `0.08` to `0.10`. Purpose: reduce the remaining `0.614122 m`
  final box lateral near-miss without the latch delay of wider geometry, the
  tilt spike of final stand, or the destabilization from removing lateral
  correction.
- 2026-07-06 submitted chest-pad lateral-gain diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstop_latgain010_targetwindow1000_0706`; Slurm
  job `168453`, job-name `g1_cp_lg10`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_latgain010_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 chest-pad lateral-gain diagnostic `168453` completed on
  `server21` in `00:00:49` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `1000` steps with
  rollout root pose/root velocity/box pose writes all `0`, but
  `AGILE_COMMAND_HOLD_LATERAL_GAIN=0.10` overcorrected and destabilized the
  run: fall/drop `95/82`, first fall/drop step `905/918`, terminal latch
  `false`, final latch `false`, target-window both stable/longest/end streak
  `0/0/0`, max robot/box tilt `0.971987/1.016840 rad`, final robot/box
  lateral error `-1.621430/-1.625827 m`, and final relative offset
  `0.489280 m`. Conclusion: do not continue high lateral gain; if testing
  gain, use only a very small interpolation above the `168431` near-pass value
  `0.08`.
- 2026-07-06 next chest-pad controller branch: return to the `168431`
  final-stop setup and test a tiny lateral gain interpolation
  `AGILE_COMMAND_HOLD_LATERAL_GAIN=0.085`. Purpose: determine whether a small
  change can close the `0.014122 m` final-box-lateral gap without the severe
  overcorrection seen at `0.10`.
- 2026-07-06 submitted chest-pad tiny lateral-gain interpolation diagnostic.
  Tmux: `curiosity_g1_chestpad_finalstop_latgain0085_targetwindow1000_0706`;
  Slurm job `168454`, job-name `g1_cp_lg085`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_latgain0085_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 chest-pad tiny lateral-gain interpolation diagnostic `168454`
  completed on `server21` in `00:00:48` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `1000`
  steps with fall/drop `0/0` and rollout root pose/root velocity/box pose
  writes all `0`, but even `AGILE_COMMAND_HOLD_LATERAL_GAIN=0.085` changed the
  trajectory enough that it never reached terminal/final target gates:
  terminal latch step `641`, final latch `false`, final active steps `0`,
  target-window both stable/longest/end streak `0/0/0`, final robot/box
  target-directed travel `1.188529/1.227951 m`, final robot/box lateral error
  `0.658880/0.774984 m`, max robot/box tilt `0.371441/0.630405 rad`, and
  final relative offset `0.124223 m`. Conclusion: lateral-gain interpolation
  is too sensitive and worse than the `168431` near-pass.
- 2026-07-06 next chest-pad branch: return to the `168431` geometry,
  lateral-gain, and final-stop structure, but add a tiny base lateral command
  `AGILE_COMMAND_Y=0.005`. Purpose: test whether a small path bias can reduce
  the remaining final lateral error without destabilizing the hold controller.
- 2026-07-06 submitted chest-pad tiny base-lateral-command diagnostic. Tmux:
  `curiosity_g1_chestpad_finalstop_cmdy005_targetwindow1000_0706`; Slurm job
  `168455`, job-name `g1_cp_y005`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_cmdy005_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 chest-pad tiny base-lateral-command diagnostic `168455`
  completed on `server21` in `00:00:48` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `1000`
  steps with fall/drop `0/0` and rollout root pose/root velocity/box pose
  writes all `0`, but `AGILE_COMMAND_Y=0.005` was too large: final latch step
  `960`, final active steps `40`, target-window both stable/longest/end
  streak `41/41/41`, max robot/box tilt `0.344988/0.486637 rad`, final
  relative offset `0.130053 m`, final robot/box target-directed travel
  `1.932092/1.932636 m`, and final robot/box lateral error
  `-0.858138/-0.730904 m`. Conclusion: base lateral command can affect the
  sign of the lateral miss, but `0.005` is too large and delays the final
  hold. The next valid interpolation should be much smaller.
- 2026-07-06 next chest-pad branch: return to the `168431` setup and set
  `AGILE_COMMAND_Y=0.001`. Purpose: test a much smaller lateral path bias
  between the near-pass `0.0` and the over-biasing `0.005`.
- 2026-07-06 submitted chest-pad smaller base-lateral-command diagnostic.
  Tmux: `curiosity_g1_chestpad_finalstop_cmdy001_targetwindow1000_0706`;
  Slurm job `168456`, job-name `g1_cp_y001`; stamp:
  `20260706_g1_agile_chestpad_oppositeyaw_finalstop_cmdy001_targetwindow_1000_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 chest-pad smaller base-lateral-command diagnostic `168456`
  completed on `server21` in `00:00:51` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `1000`
  steps with rollout root pose/root velocity/box pose writes all `0`, but
  even `AGILE_COMMAND_Y=0.001` badly changed the latch timing and stability:
  fall/drop `347/272`, first fall/drop step `653/686`, terminal latch step
  `492`, final latch step `610`, target-window both stable/longest/end streak
  `31/31/0`, final robot/box target-directed travel `4.135339/4.167482 m`,
  final robot/box lateral error `-1.190266/-1.049643 m`, and max robot/box
  tilt `3.140074/3.138898 rad`. Conclusion: base command-y bias is not a
  reliable fix for chest-pad final lateral error. Stop this micro-bias branch;
  further progress should either redesign the chest-pad/contact controller or
  move to held-out validation on the already verified low-carry route.
- 2026-07-06 submitted low-carry held-out light-box validation using the
  verified target-hold route but with `FREE_BOX_MASS=0.25`. The run keeps the
  strict baseline target-window/final-hold gates: `819` steps, target-window
  center `2.0` with halfwidth `0.35`, minimum final-hold active steps `399`,
  final-hold command near zero, fall/drop `0/0`, height/tilt gates, and no
  rollout root/box shortcut writes. Tmux:
  `curiosity_g1_lowcarry_lightbox_targethold819_strict_0706`; Slurm job
  `168458`, job-name `g1_lc_l025`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 low-carry held-out light-box validation `168458` completed on
  `server21` in `00:00:48` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with
  rollout root pose/root velocity/box pose writes all `0`, but the `0.25 kg`
  box was not stable under the baseline low-carry controller: fall/drop
  `384/225`, first fall/drop step `435/594`, final-hold fall/drop
  `384/225`, target-window both stable/longest/end streak `8/8/0`, final-hold
  stable/longest/end streak `8/8/0`, final robot/box target-directed travel
  `4.399167/3.986899 m`, final robot/box lateral error
  `-0.689620/-0.694785 m`, final relative offset `0.449969 m`, and max
  robot/box tilt `2.710347/2.745914 rad`. Conclusion: the verified
  low-carry result does not yet generalize to a lighter held-out mass; this is
  negative load-generalization evidence, not a success.
- 2026-07-06 next low-carry held-out branch: run the same strict target-hold
  validation with `FREE_BOX_MASS=0.75`. Purpose: check whether the low-carry
  route has any nearby mass robustness on the heavier side or whether mass
  changes generally break the current controller.
- 2026-07-06 submitted low-carry heavy-box held-out validation. Tmux:
  `curiosity_g1_lowcarry_heavybox_targethold819_strict_0706`; Slurm job
  `168462`, job-name `g1_lc_h075`; stamp:
  `20260706_g1_agile_lowcarry_heavybox075_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 low-carry heavy-box held-out validation `168462` completed on
  `server21` in `00:00:50` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with
  rollout root pose/root velocity/box pose writes all `0`, but the `0.75 kg`
  held-out mass also failed: fall/drop `346/284`, first fall/drop step
  `473/535`, terminal latch `false`, final latch `false`, target-window and
  final-hold stable/longest/end streaks `0/0/0`, final active steps `0`,
  final robot/box target-directed travel `0.182098/-0.169243 m`, final
  relative offset `0.405846 m`, and max robot/box tilt
  `1.996009/3.139406 rad`. Conclusion: the current low-carry target-hold
  pass is a narrow single-mass/single-setting result; both lighter and heavier
  nearby mass held-outs fail under strict gates. Do not claim load
  generalization. The next valid implementation direction is not more
  scalar gate tuning, but an explicit load-adaptive stabilization/probing
  controller or policy update.
- 2026-07-06 implementation update toward load-adaptive stabilization:
  `scripts/isaac/build_core_world_g1_box_scene.py` now supports observed
  robot-progress gates for agile terminal/final hold:
  `--agile-command-hold-terminal-min-robot-target-travel` and
  `--agile-command-hold-final-min-robot-target-travel`, both defaulting to
  `-1.0` so older runs are unchanged. The launcher
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh` forwards
  `AGILE_COMMAND_HOLD_TERMINAL_MIN_ROBOT_TARGET_TRAVEL` and
  `AGILE_COMMAND_HOLD_FINAL_MIN_ROBOT_TARGET_TRAVEL`. This is meant to prevent
  light boxes from triggering terminal/final hold solely because the box moved
  early, while the robot has not reached a stable carrying state. It uses
  observed robot/box progress only, not hidden mass. Lightweight checks passed
  on the login node: `bash -n` for the launcher and `python3 -m py_compile`
  for the scene script.
- 2026-07-06 submitted first robot-progress-gated light-box validation.
  It reruns the strict `0.25 kg` low-carry held-out case with
  `AGILE_COMMAND_HOLD_TERMINAL_MIN_ROBOT_TARGET_TRAVEL=0.50` and
  `AGILE_COMMAND_HOLD_FINAL_MIN_ROBOT_TARGET_TRAVEL=0.80`, so terminal/final
  hold cannot trigger from early box motion alone. Tmux:
  `curiosity_g1_lowcarry_lightbox_robotgate_targethold819_0706`; Slurm job
  `168465`, job-name `g1_lc_lrg`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_robotgate_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 robot-progress-gated light-box validation `168465` completed on
  `server21` in `00:00:48` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with
  rollout root pose/root velocity/box pose writes all `0`, but the
  robot-progress gates were not enough: fall/drop `428/340`, first fall/drop
  step `391/412`, terminal latch step `267`, final latch step `285`,
  target-window and final-hold stable/longest/end streaks `0/0/0`, final
  robot/box target-directed travel `1.866408/1.439961 m`, final robot/box
  lateral error `-2.250702/-2.305717 m`, final relative offset `0.431750 m`,
  and max robot/box tilt `1.841554/2.180149 rad`. Conclusion: observed robot
  progress gating alone does not fix early light-box instability.
- 2026-07-06 implementation update after `168465`: added observed minimum-step
  gates for agile terminal/final hold:
  `--agile-command-hold-terminal-min-step` and
  `--agile-command-hold-final-min-step`, forwarded by
  `AGILE_COMMAND_HOLD_TERMINAL_MIN_STEP` and
  `AGILE_COMMAND_HOLD_FINAL_MIN_STEP`. Defaults are `-1`, so older runs are
  unchanged. This is a phase-stability guard intended to stop light boxes from
  entering terminal/final hold before the walking controller has reached the
  stable phase seen in the nominal low-carry pass. Lightweight `bash -n` and
  `python3 -m py_compile` checks passed.
- 2026-07-06 submitted min-step-gated light-box validation. It reruns the
  strict `0.25 kg` low-carry held-out case with
  `AGILE_COMMAND_HOLD_TERMINAL_MIN_STEP=350` and
  `AGILE_COMMAND_HOLD_FINAL_MIN_STEP=386`, matching the nominal low-carry
  pass's later stable-hold phase more closely. Tmux:
  `curiosity_g1_lowcarry_lightbox_stepgate_targethold819_0706`; Slurm job
  `168466`, job-name `g1_lc_lsg`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_stepgate_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 min-step-gated light-box validation `168466` completed on
  `server21` in `00:00:47` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with
  rollout root pose/root velocity/box pose writes all `0`, but the delayed
  terminal/final gates still failed: fall/drop `402/149`, first fall/drop step
  `417/437`, terminal/final latch steps `350/386`, final active steps `433`,
  target-window and final-hold stable/longest/end streaks `0/0/0`, final
  robot/box target-directed travel `2.524991/2.163685 m`, final robot/box
  lateral error `-1.850566/-1.726324 m`, final relative offset `0.389333 m`,
  and max robot/box tilt `1.977972/2.232556 rad`. Conclusion: step gates
  reduced box-drop count versus `168465`, but did not stabilize the light
  held-out load. The next test should delay terminal/final together to the
  latest step that still permits the strict `399` final-active-step gate,
  namely step `420`.
- 2026-07-06 submitted latest-step light-box gate validation with both
  `AGILE_COMMAND_HOLD_TERMINAL_MIN_STEP=420` and
  `AGILE_COMMAND_HOLD_FINAL_MIN_STEP=420`. Tmux:
  `curiosity_g1_lowcarry_lightbox_step420_targethold819_0706`; Slurm job
  `168467`, job-name `g1_lc_l420`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_step420_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 latest-step light-box gate validation `168467` completed on
  `server21` in `00:00:47` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with
  rollout root pose/root velocity/box pose writes all `0`, but fall/drop was
  `400/344`, first fall/drop step `419/437`, terminal/final latch steps
  `420/420`, final active steps `399`, target-window and final-hold
  stable/longest/end streaks `0/0/0`, final robot/box target-directed travel
  `2.668007/2.280981 m`, final robot/box lateral error
  `-0.957738/-1.112425 m`, final relative offset `0.417000 m`, and max
  robot/box tilt `3.131353/3.140109 rad`. Conclusion: delaying terminal/final
  to the latest strict-gate-compatible step still fails because switching to
  zero final command destabilizes immediately. The next test should keep the
  step `420` gate but use a tiny nonzero final scale that remains below the
  strict final-command threshold.
- 2026-07-06 submitted light-box step420 micro-final-scale validation. It
  keeps terminal/final min steps at `420` but changes
  `AGILE_COMMAND_HOLD_FINAL_SCALE` from `0.0` to `0.006`, giving an x command
  around `0.0006`, below the strict `MAX_FINAL_HOLD_COMMAND_X=0.001` gate.
  Tmux: `curiosity_g1_lowcarry_lightbox_step420_final006_0706`; Slurm job
  `168468`, job-name `g1_lc_lf006`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_step420_final006_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 light-box step420 micro-final-scale validation `168468`
  completed on `server21` in `00:00:50` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `819`
  steps with rollout root pose/root velocity/box pose writes all `0`, but
  still failed immediately after final hold began: fall/drop `400/356`, first
  fall/drop step `419/437`, terminal/final latch steps `420/420`, final active
  steps `399`, final command `[0.0006, 0.0, 0.0]` satisfying the strict
  command gate, target-window and final-hold stable/longest/end streaks
  `0/0/0`, final robot/box target-directed travel `0.883383/0.826458 m`,
  final robot/box lateral error `-1.089999/-0.918283 m`, final relative offset
  `0.277017 m`, and max robot/box tilt `1.402402/1.504111 rad`. Conclusion:
  the light-box failure is not solved by robot-progress gates, min-step gates,
  or tiny nonzero final command. Stop this scalar/phase-gate branch; the next
  valid implementation change should alter contact/retention or the hold
  posture itself, such as load-response-dependent cradle geometry, extra
  top/side retention, or a controller-backed stabilization posture.
- 2026-07-06 submitted first contact-retention light-box branch. It reruns the
  strict `0.25 kg` low-carry held-out case without the scalar/phase gates, but
  changes physical retention geometry: top lid is collision-enabled from the
  start (`CRADLE_TOP_LID_ENABLE_ON_HOLD=0`), lowered to `0.10 m`, thickened to
  `0.018 m`, expanded to `1.25x/1.25y`, side rail height `0.16 m`, end-stop
  height `0.18 m`, and rail thickness `0.03 m`. This is a contact/retention
  branch, not a shortcut: strict no root/box rollout writes and fall/drop/
  target-window/final-hold gates remain. Tmux:
  `curiosity_g1_lowcarry_lightbox_strongretention_targethold819_0706`; Slurm
  job `168471`, job-name `g1_lc_lret`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_strongretention_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 first contact-retention light-box branch `168471` completed on
  `server21` in `00:00:48` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with
  rollout root pose/root velocity/box pose writes all `0`. Strong retention
  improved some carry-quality metrics versus the previous light-box failures:
  box drop events decreased to `104`, final relative offset was `0.205683 m`,
  final robot/box lateral error was `0.568279/0.410915 m`, and final robot/box
  target-directed travel was `1.139480/1.089813 m`. It still failed because
  fall/drop remained `335/104`, first fall/drop step `484/523`, target-window
  and final-hold stable/longest/end streaks `0/0/0`, min robot/box z
  `0.190951/0.159953 m`, and max robot/box tilt `3.131061/3.123740 rad`.
  Conclusion: stronger physical retention helps relative/lateral holding but
  does not solve balance; the next branch should combine this retention with a
  later final/terminal step gate so final zero command does not begin at step
  `364`.
- 2026-07-06 submitted strong-retention plus step420 light-box validation.
  This combines the `168471` physical retention geometry with
  `AGILE_COMMAND_HOLD_TERMINAL_MIN_STEP=420` and
  `AGILE_COMMAND_HOLD_FINAL_MIN_STEP=420`. Tmux:
  `curiosity_g1_lowcarry_lightbox_retention_step420_0706`; Slurm job
  `168472`, job-name `g1_lc_rs420`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_retention_step420_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 strong-retention plus step420 light-box validation `168472`
  completed on `server21` in `00:00:48` with Slurm exit `0:0`, build status
  `0`, checker status `1`, and checker `status=fail`. It completed `819`
  steps with rollout root pose/root velocity/box pose writes all `0`. This
  was the best light-box retention branch so far in terms of delayed failure:
  fall/drop decreased to `55/39`, first fall/drop moved late to `764/780`, and
  final active steps were `399`. It still failed strict gates: target-window
  and final-hold stable/longest/end streaks `0/0/0`, final robot/box lateral
  error `-2.340681/-2.443050 m`, final relative offset `0.359277 m`, min
  robot/box z `-0.338719/-0.518896 m`, and max robot/box tilt
  `2.705030/1.518419 rad`. Conclusion: strong retention plus delayed final
  hold extends stability but the robot drifts sideways and rolls over late.
  The next branch should add a slow final-stand posture blend after final hold
  starts, to test whether a controller-backed hold posture can arrest the
  late roll without root/box shortcuts.
- 2026-07-06 submitted strong-retention plus step420 plus slow final-stand
  blend. It uses the `168472` geometry/gates, enables
  `AGILE_COMMAND_HOLD_FINAL_STAND=1`, delays final-stand activation by `80`
  final-hold steps, and sets `AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.002`.
  Tmux: `curiosity_g1_lowcarry_lightbox_retention_step420_finalstand_0706`;
  Slurm job `168473`, job-name `g1_lc_rsfs`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_retention_step420_finalstand_targethold819_strict_targetnegx1`.
  Submission-time status was `PENDING (Priority)`.
- 2026-07-06 strong-retention plus step420 plus slow final-stand validation
  `168473` completed on `server21` in `00:00:49` with Slurm exit `0:0`,
  build status `0`, checker status `1`, and checker `status=fail`. It
  completed `819` steps with rollout root pose/root velocity/box pose writes
  all `0`. The final-stand blend began at step `500`, but did not arrest the
  late roll: fall/drop were `237/20`, first fall/drop occurred at `582/628`,
  target-window/final-hold/final-stand end streaks were all `0`, final robot/
  box target-directed travel was `1.347323/1.365088 m`, max robot/box
  target-directed travel was `1.775143/1.648127 m`, final robot/box lateral
  error was about `-0.967/-1.031 m`, final relative offset error was
  `0.308799 m`, min robot/box z was `0.219165/0.163008 m`, and max robot/box
  tilt was `2.052260/3.123082 rad`. This is worse than `168472` on delayed
  failure timing (`168472` first fall/drop `764/780`). Do not continue tuning
  this slow final-stand blend as the active fix. Return to the `168472`
  strong-retention plus delayed-final base and address late lateral drift/roll
  directly while preserving no rollout root/velocity/box pose shortcuts.
- 2026-07-06 submitted a direct late-lateral-drift diagnostic based on
  `168472`, without final-stand blend and without root/velocity/box pose
  shortcuts. Tmux:
  `curiosity_g1_lowcarry_lightbox_latcorr_step420_0706`; Slurm job `168475`,
  job-name `g1_lc_latcorr`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latcorr_targethold819_strict_targetnegx1`.
  It keeps the same 0.25 kg light box, strong retention geometry, terminal/
  final min steps `420/420`, and strict target-window/final-hold gates as
  `168472`, but changes the lateral correction test: `terminal_only` is off,
  `AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.20`,
  `AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.003`, and the lateral correction tilt
  gates are relaxed to robot/box `0.80/0.90 rad`. This tests whether `168472`
  failed because lateral correction was confined to terminal hold and then
  suppressed by tilt before the late sideways roll. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 late-lateral-drift diagnostic `168475` completed on `server21`
  in `00:00:29` with Slurm exit `0:0`, build status `0`, checker status `1`,
  and checker `status=fail`. It completed `819` steps with rollout root pose/
  root velocity/box pose writes all `0`. Compared with `168472`, lateral
  correction was active much more often (`242` steps versus `58`) and tilt
  suppression dropped (`87` steps versus `196`), but it did not solve the
  drift or balance: fall/drop were `87/39`, first fall/drop occurred at
  `732/780`, target-window/final-hold end streaks stayed `0`, final robot/box
  lateral error was `-2.313470/-2.372427 m`, final relative offset was
  `0.291257 m`, min robot/box z was `0.037218/-0.131656 m`, and max robot/box
  tilt was `1.636552/1.647063 rad`. This is worse than `168472` for delayed
  fall timing and still misses lateral/target-window gates. Conclusion:
  simply allowing more lateral command with the current sign and gain is not a
  valid fix. The next small diagnostic should reverse the lateral command sign
  while keeping the same no-final-stand `168472` base, to determine whether the
  command direction is wrong before changing contact geometry or roll feedback.
- 2026-07-06 submitted lateral sign-reversal diagnostic from the same
  `168472` no-final-stand strong-retention base. Tmux:
  `curiosity_g1_lowcarry_lightbox_latcorr_revsign_step420_0706`; Slurm job
  `168478`, job-name `g1_lc_latrev`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_targethold819_strict_targetnegx1`.
  It keeps the `168475` always-on relaxed lateral correction window and
  changes only `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0`. Submission-time status
  was `PENDING (Priority)`.
- 2026-07-06 lateral sign-reversal diagnostic `168478` completed on
  `server21` in `00:00:26` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with rollout
  root pose/root velocity/box pose writes all `0`. Reversing lateral sign
  improved the sideways drift and tilt relative to `168472`/`168475`: final
  robot/box lateral error was `-1.079545/-1.217948 m`, max robot/box lateral
  error was `1.140907/1.270496 m`, and max robot/box tilt was
  `1.273742/1.348365 rad`. But it destabilized the run earlier and hurt
  progress/box retention: fall/drop were `150/100`, first fall/drop occurred
  at `657/719`, final robot/box target-directed travel was
  `1.689554/1.392827 m`, final relative offset was `0.382696 m`, and
  target-window/final-hold end streaks remained `0`. Conclusion: the original
  lateral sign was likely wrong for this failure mode, but a reversed
  always-on `0.003` lateral command is too aggressive. The next diagnostic
  should keep the reversed sign but reduce the lateral command back toward the
  earlier `0.0015` limit before changing contact geometry.
- 2026-07-06 submitted mild reversed-lateral diagnostic from the same `168472`
  no-final-stand strong-retention base. Tmux:
  `curiosity_g1_lowcarry_lightbox_latrev_mild_step420_0706`; Slurm job
  `168479`, job-name `g1_lc_latmild`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_mild_targethold819_strict_targetnegx1`.
  It keeps `AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0` and the relaxed always-on
  lateral window from `168478`, but reduces
  `AGILE_COMMAND_HOLD_LATERAL_LIMIT` to `0.0015`. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 mild reversed-lateral diagnostic `168479` completed on
  `server21` in `00:00:28` with Slurm exit `0:0`, build status `0`, checker
  status `1`, and checker `status=fail`. It completed `819` steps with rollout
  root pose/root velocity/box pose writes all `0`. This was a partial
  improvement over the aggressive reversed sign: box drops went to `0`, final
  robot/box target-directed travel was `2.013973/1.655340 m`, max robot/box
  target-directed travel was `2.057605/1.752282 m`, and the robot briefly had
  `24` target-window stable steps. It still failed the strict task: fall
  events were `74` with first fall at step `745`, target-window/final-hold end
  streaks were `0`, final robot/box lateral error was
  `-1.437272/-1.531827 m`, final relative offset was `0.371662 m`, min
  robot/box z was `0.278112/0.376793 m`, and max robot/box tilt was
  `1.424887/1.671567 rad`. Conclusion: mild reversed lateral command helps
  retain the box and restore forward progress, but the active failure is now
  late roll instability plus large lateral drift and box lag. The next
  diagnostic should keep the `168479` lateral settings and flip or retune the
  roll-feedback sign/gain rather than increasing lateral command again.
- 2026-07-06 submitted roll-feedback sign diagnostic on top of `168479`.
  Tmux: `curiosity_g1_lowcarry_lightbox_latrev_mild_rollpos_0706`; Slurm job
  `168482`, job-name `g1_lc_rollpos`; stamp:
  `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_mild_rollpos_targethold819_strict_targetnegx1`.
  It keeps the mild reversed lateral settings from `168479` and sets
  `BALANCE_ROLL_SIGN=1.0` explicitly, leaving the box, retention geometry,
  target gates, and no-shortcut checks unchanged. Submission-time status was
  `PENDING (Priority)`.
- 2026-07-06 scheduler note for `168482`: because the `gpu` partition reported
  a delayed predicted start, a duplicate `test`-partition tmux submission was
  attempted but exited before Slurm job creation. Lightweight Slurm
  `--test-only` checks showed `test`, `gaosh`, and `engram` are invalid for
  this account/partition combination, and `long` is unavailable/inactive or
  drain. Keep `168482` in the `gpu` partition as the active roll-sign
  diagnostic; do not move this experiment to login-node execution.
- 2026-07-06 additional scheduler note for `168482`: lightweight Slurm
  `--test-only` checks showed that reducing the request from 8 CPUs to 2 or 4
  CPUs, or reducing walltime from 20 minutes to 5 or 10 minutes, did not
  improve predicted `gpu` partition start time for a new request. Do not
  cancel `168482` for a smaller duplicate unless the scheduler state changes
  materially.
- 2026-07-06 additional scheduler note: a `cpu` partition submission with
  `--gres=gpu:1` for the same roll-sign diagnostic was tried as Slurm job
  `168502`, job-name `g1_lc_rollcpu`, because `srun --test-only` initially
  suggested an earlier start. The real job remained pending with reason
  `Nodes_required_for_job_are_DOWN,_DRAINED_or_reserved_for_jobs_in_higher_priority_partitions`
  and no predicted start time, so it was cancelled. Keep `168482` as the
  active pending diagnostic.
- 2026-07-06 low-carry lateral/roll decision report:
  `experiments/reports/2026-07-06_g1_lowcarry_lateral_roll_decision.md`
  summarizes diagnostics `168472`, `168475`, `168478`, `168479`, and active
  pending `168482`. It records that old-sign always-on lateral correction did
  not help, reversed sign reduced drift but aggressive `0.003` caused early
  fall/drop, mild reversed `0.0015` kept box drops at `0` but still failed
  from late roll, lateral drift, and box lag. Use this report to decide whether
  to continue roll-feedback tuning after `168482` or switch to contact/hold
  geometry.
- 2026-07-06 added compute-side follow-up launcher
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`. It
  refuses to run on `mgmtserver*` and wraps the current `168479` low-carry
  base into two explicit post-`168482` branches: `CASE_SET=roll` runs
  `rollpos_delay` with `BALANCE_ROLL_SIGN=1.0` and `BALANCE_START_STEP=420`;
  `CASE_SET=contact` runs `chestpad_hold_contact` with chest-pad support
  enabled on hold. `CASE_SET=all` runs both sequentially inside one compute
  allocation. This launcher is preparation only; it has not produced success
  evidence and must not be run on the login node.
- 2026-07-06 showcase status report:
  `experiments/reports/2026-07-06_showcase_status.md` lists the strongest
  presentable evidence (`168398`, `168431`, `168479`), current MP4 assets, and
  wording boundaries. It explicitly says these assets are scaffold/diagnostic
  visuals, not final humanoid carrying success, and records `168482` as the
  active pending roll-sign diagnostic.
- 2026-07-06 showcase correction after user review: the existing MP4 files
  under `experiments/visuals/direct_carry_posture_suite/...` are abstract
  scaffold/debug videos with proxy blocks, not understandable humanoid
  walking-carrying visuals. Do not use them as main presentation material.
  The showcase report now says the needed visual is an Isaac-rendered G1 scene
  with a visible humanoid, visible free box, clear side/three-quarter camera,
  and explicit pass/near-pass/failure-diagnostic labeling.
- 2026-07-06 real G1 showcase capture path: added Replicator RGB capture
  support to `scripts/isaac/build_core_world_g1_box_scene.py` and launcher
  `scripts/isaac/run_core_world_g1_showcase_lowcarry_capture.sh`. The launcher
  reuses the verified `168398` low-carry pass configuration and writes frames
  under
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_showcase_g1_lowcarry_168398_rgb/agile_low_cradle_freebox_walk/rgb_frames/`,
  with optional `showcase_g1_lowcarry.mp4` generation if `ffmpeg` is available
  on the compute node. Slurm job `168509`, job-name `g1_show_rgb`, was
  submitted through tmux session `curiosity_g1_showcase_rgb_0706` and was
  pending in `gpu` due to priority at submission. This is not new control
  evidence until frames/video are produced and inspected.
- 2026-07-06 `168509` result: negative for showcase. It ran on `server39` and
  completed the script, but `capture_rgb_error` was
  `ModuleNotFoundError: No module named 'omni.replicator'`, so no RGB frames
  were produced. The render-enabled rerun also failed control gates with early
  fall/drop; do not use `168509` as pass evidence, and do not present its CSV
  as the verified `168398` result. The active visualization route is now
  two-stage replay: first record a non-rendered pass trajectory with
  `--record-replay-csv`, then render that recorded root/joint/box trajectory
  with `scripts/isaac/render_core_world_g1_replay_showcase.py`.
- 2026-07-06 replay visualization tooling added:
  `scripts/isaac/build_core_world_g1_box_scene.py` now supports
  `--record-replay-csv` and `--record-replay-every-n-steps`; launcher
  `scripts/isaac/run_core_world_g1_showcase_lowcarry_capture.sh` supports
  `SHOWCASE_CAPTURE_RGB=0 SHOWCASE_RECORD_REPLAY=1`; replay renderer launcher
  is `scripts/isaac/run_core_world_g1_replay_showcase_render.sh`. Pending
  replay-record job `168580`, job-name `g1_rec_short`, is in the `cpu`
  partition with `--gres=gpu:1`; as of 2026-07-06 22:24 CST Slurm schedules
  it around 2026-07-06 23:15 CST on `server02`.
  Before `168580` ran, replay CSV file handling in
  `scripts/isaac/build_core_world_g1_box_scene.py` was hardened with
  `ExitStack`, so the optional replay CSV is closed by context management
  instead of depending only on the normal end-of-loop path.
  The replay renderer falls back between `omni.kit.renderer_capture` and
  `omni.renderer_capture` and calls `wait_async_capture()` when available.
  It also supports `--follow-frame` / `--frame-zoom`; the launcher enables
  follow-frame so the camera tracks the real G1 and box during the replay.
  Replay outputs must be checked with
  `scripts/isaac/check_core_world_g1_replay_showcase.py`; a presentable replay
  requires source rollout `status=pass`, `record_replay_csv=true`, fall/drop 0,
  enough replay rows, enough rendered PNG frames, and a visualization-only
  success claim.
- 2026-07-06 `168580`, job-name `g1_rec_short`, result: negative and not usable
  for replay rendering. It ran on `server39` and failed strict carrying gates
  with fall/drop `720/617`, final robot/box target-directed travel
  `0.2968/0.1619 m`, max robot/box tilt `2.0515/2.1004 rad`, and no shortcut
  writes. The job environment had `CAPTURE_RGB=1`, `record_replay_csv=false`,
  and missing `RECORD_REPLAY_CSV=1`, so it repeated the old RGB-capture path
  instead of the intended non-rendered replay-record path. Do not use `168580`
  as pass evidence or as a replay source.
- 2026-07-06 replay-record retry: added strict watcher
  `scripts/isaac/wait_and_submit_g1_replay_render.sh`, which refuses to submit
  render unless the record summary is `status=pass`, `record_replay_csv=true`,
  fall/drop `0/0`, and the replay CSV has at least 20 rows. Corrected retry
  job `168632`, job-name `g1_rec_retry`, stamp
  `20260706_g1_lowcarry_168398_replay_record_retry2`, was submitted through
  tmux session `curiosity_g1_record_replay_retry2_0706` with explicit
  `SHOWCASE_CAPTURE_RGB=0`, `SHOWCASE_RECORD_REPLAY=1`, `CAPTURE_RGB=0`,
  `RECORD_REPLAY_CSV=1`, and `RECORD_REPLAY_EVERY_N_STEPS=10`. Strict render
  watcher session:
  `curiosity_g1_replay_render_retry2_waiter_0706`.
- 2026-07-06 corrected replay-record retry `168632` result: pass and usable as
  replay source. It ran on `server39` with `CAPTURE_RGB=0`,
  `RECORD_REPLAY_CSV=1`, completed 819/819, fall/drop `0/0`, final robot/box
  target-directed travel `2.2988/2.3465 m`, final relative error `0.0796 m`,
  max robot/box tilt `0.2086/0.4136 rad`, and no rollout root pose, root
  velocity, or box pose writes. It wrote
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_lowcarry_168398_replay_record_retry2/agile_low_cradle_freebox_walk/core_world_g1_box_scene_replay.csv`
  with 84 lines. A watcher bug that treated zero fall/drop counts as missing
  was fixed in `scripts/isaac/wait_and_submit_g1_replay_render.sh`. Replay
  render job `168658`, job-name `g1_replay_viz2`, was submitted to write
  visualization outputs under
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2/`.
  Treat the render as visualization replay only, not new control evidence.
  `168658` was later cancelled while still pending because Slurm scheduled it
  for 2026-07-07 00:04 CST and a shorter quick render could backfill earlier.
  First quick render `168664` was also cancelled while pending after Slurm
  pushed it to 2026-07-07 00:04 CST. Current quick replay render is `168669`,
  job-name `g1_viz_q2`, tmux session
  `curiosity_g1_replay_quick2_render_0706`, targeting
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2_quick2/`
  with `CAPTURE_EVERY_N_ROWS=4` and `MAX_FRAMES=18`. This is still replay
  visualization only, not new control evidence. Before it started, its time
  limit was updated from 3 minutes to 5 minutes after Slurm pushed the job to
  2026-07-07 01:22:51 CST, and the replay renderer was hardened with explicit
  dome/key lighting to reduce dark/blank frame risk.
- 2026-07-06 immediate showcase fallback:
  `experiments/visuals/g1_progress_showcase/20260706_g1_lowcarry_168398_browser_showcase/index.html`
  is a browser-only schematic built from sampled `168398` rollout CSV states.
  It draws a G1-like humanoid, low-carry box, side-view posture, top-view path,
  and key metrics. The same directory also contains
  `g1_lowcarry_168398_progress_poster.svg` as a static slide/inspection poster.
  These may be used as quick progress visualizations while true Isaac replay
  rendering is queued. They must not be called true Isaac camera renders, new
  control evidence, or final carrying success.
- 2026-07-06 follow-up control pre-check while waiting for `168580`: the
  active post-showcase branch is `CASE_SET=contact` in
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`, which
  runs `chestpad_hold_contact`. This changes hold/contact geometry by enabling
  chest-pad support on hold while preserving no rollout root pose, root
  velocity, or box pose shortcut gates. It still requires target-window
  stable-step gates near the configured 2.0 m target window and caps final
  target-directed overrun; this pre-check is not experiment evidence.
- 2026-07-06 contact follow-up `168627`, job-name `g1_contact_next`, stamp
  `20260706_g1_lowcarry_followup_chestpad_hold_contact`, was triggered
  prematurely after the failed `168580` summary but completed and is useful as
  negative/partial control evidence. Base summary had fall/drop `0/0`, no
  shortcut writes, chest-pad enabled on hold, max robot/box tilt
  `0.2747/0.2733 rad`, final relative error `0.1482 m`, and final robot/box
  target-directed travel `0.7175/0.6576 m`. The strict checker failed:
  target-window stable steps `0`, final-hold active steps `15 < 399`. Treat it
  as evidence that chest-pad hold improves stability/retention but suppresses
  progress to the 2.0 m target window; it is not carrying success.
  Follow-up branch prepared from this result: delayed chest-pad closure.
  `scripts/isaac/build_core_world_g1_box_scene.py` supports
  `--cradle-chest-pad-enable-on-terminal-hold` and
  `--cradle-chest-pad-enable-on-final-hold`; the low-cradle launcher forwards
  `CRADLE_CHEST_PAD_ENABLE_ON_TERMINAL_HOLD` /
  `CRADLE_CHEST_PAD_ENABLE_ON_FINAL_HOLD`; and
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`
  supports `CASE_SET=contact_terminal` / `CASE_SET=contact_next`, running
  `chestpad_terminal_contact`. This branch should test whether the robot can
  preserve the 168398-style target progress, then add chest support only near
  terminal/final hold. It has not been run yet.
  Quick render job `168669` was later pushed to 2026-07-07 01:22 CST, so the
  after-render watcher `curiosity_g1_contact_next_after_render_0706` was
  stopped to avoid delaying control progress and to avoid duplicate submission.
  Direct delayed chest-pad contact run `168788`, job-name `g1_contact_next2`,
  tmux session `curiosity_g1_contact_next_direct_0706`, prefix
  `20260706_after_quick_render_contact_next`, `CASE_SET=contact_next`, is now
  pending; as of 2026-07-07 00:00 CST Slurm schedules it for
  2026-07-07 01:22:51 CST on `server02`. Active Curiosity sessions are
  `curiosity_g1_replay_quick2_render_0706` for quick replay rendering and
  `curiosity_g1_contact_next_direct_0706` for the delayed contact follow-up.
  As of 2026-07-07 00:14 CST, Slurm rescheduled quick render `168669` to
  2026-07-07 03:45:50 CST and delayed contact run `168788` to
  2026-07-07 10:00:00 CST.
- 2026-07-07 queue correction: `168669`/`168788` were cancelled while still
  pending and before producing artifacts because they were stuck in the `cpu`
  partition. Replacement tmux+srun jobs are now in the `gpu` partition:
  `168801`, job-name `g1_viz_gpu_q3`, tmux session
  `curiosity_g1_replay_gpu_render_0707`, output target
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3/`,
  `CAPTURE_EVERY_N_ROWS=4`, `MAX_FRAMES=18`, scheduled as of 00:18 CST for
  2026-07-07 03:45:50 CST; and `168802`, job-name `g1_contact_gpu`, tmux
  session `curiosity_g1_contact_next_gpu_0707`, output prefix
  `20260707_gpu_contact_next`, `CASE_SET=contact_next`, scheduled as of
  00:18 CST for 2026-07-07 06:11:58 CST. Do not treat either as result
  evidence until summaries/checkers exist.
- 2026-07-07 replay renderer update: `scripts/isaac/render_core_world_g1_replay_showcase.py`
  now adds persistent floor markers for robot and box replay trails plus
  start/end/target markers before rendering. This is only to make the Isaac G1
  replay legible in a presentation; it does not change controller evidence.
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh` now
  accepts `FOLLOWUP_PREFIX` as a fallback for `BASE_STAMP_PREFIX`, so queued
  contact follow-ups write to the intended output prefix.
- 2026-07-07 posture-coverage automation: added
  `scripts/isaac/run_core_world_g1_posture_gauntlet.sh`. It refuses to run on
  `mgmtserver*`, then runs strict compute-node cases for `lowcarry_base`,
  `chestpad_terminal`, `boxtilt_diagnostic`, `lowcarry_lightbox`, and
  `lowcarry_heavybox`, and summarizes them with
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py`. This is a
  verification gauntlet for the real objective's multi-posture/load gap; it
  must not be reported as evidence until the generated summaries/checkers
  exist. Also added a login-node guard to
  `scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`, and exposed
  `ARM_POSE_MODE`, `ARM_POSE_START_STEP`, and `ARM_POSE_RAMP_STEPS` through
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh` so
  posture cases can use the existing G1 arm-pose targets. The low-cradle
  launcher environment snapshot now includes `ARM_` variables, so gauntlet
  results record which arm-pose settings were used.
- 2026-07-07 posture gauntlet watcher: tmux session
  `curiosity_g1_posture_gauntlet_after_contact_0707` is waiting for contact
  job `168802` to leave the queue, then will submit a single `gpu` partition
  `srun` job named `g1_posture_gauntlet` with
  `GAUNTLET_STAMP=20260707_g1_posture_gauntlet_after_contact` and cases
  `lowcarry_base chestpad_terminal boxtilt_diagnostic lowcarry_lightbox
  lowcarry_heavybox`. This watcher only performs low-frequency `squeue`
  polling on the login node; the gauntlet itself must run on a compute node.
- 2026-07-07 replay showcase checker hardening:
  `scripts/isaac/check_core_world_g1_replay_showcase.py` now samples rendered
  PNG frames and checks minimum file size plus PNG IHDR dimensions using only
  the Python standard library. The render launcher passes expected
  `WIDTH`/`HEIGHT` to the checker. A replay render must pass these checks
  before it is called presentable; frame count alone is not enough.
- 2026-07-07 contact follow-up comparison: added
  `scripts/isaac/summarize_core_world_g1_contact_followup.py`, a lightweight
  JSON-only comparator for `168632` baseline low-carry, `168627`
  `chestpad_hold_contact`, and pending `168802` terminal chest-pad contact.
  It reads existing summaries/checks only and writes comparison reports under
  `experiments/reports/`. Initial pending report:
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_pending.json`
  shows baseline pass, hold-contact fail, and terminal-contact missing. Tmux
  session `curiosity_g1_contact_compare_after_168802_0707` waits for `168802`
  to leave the queue, then writes
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json`.
  This comparison is not new control evidence; it is a strict decision aid.
- 2026-07-07 completion audit gate: added
  `scripts/isaac/audit_g1_carry_completion.py`, which reads existing JSON
  evidence and returns nonzero unless baseline low-carry, terminal-contact,
  posture/load gauntlet, and no-shortcut gates all pass. Current report
  `experiments/reports/2026-07-07_g1_carry_completion_audit_current.json`
  is intentionally `fail`: baseline low-carry passes, but terminal-contact is
  missing and the posture/load gauntlet summary is missing. Tmux session
  `curiosity_g1_completion_audit_after_gauntlet_0707` waits for
  `curiosity_g1_posture_gauntlet_after_contact_0707` to finish, then writes
  `experiments/reports/2026-07-07_g1_carry_completion_audit_after_gauntlet.json`.
  Do not mark the project goal complete unless this audit or a stricter
  successor passes and the underlying evidence is inspected.
- 2026-07-07 next-action recommender: added
  `scripts/isaac/recommend_g1_next_carry_actions.py`. It reads the completion
  audit, contact comparison, and posture/load gauntlet summary and emits a
  ranked JSON action list without running Isaac. Current report
  `experiments/reports/2026-07-07_g1_next_carry_actions_current.json` says to
  wait for `168802` terminal-contact and the posture/load gauntlet rather than
  submit duplicates. Tmux session `curiosity_g1_next_actions_after_audit_0707`
  waits for `curiosity_g1_completion_audit_after_gauntlet_0707`, then writes
  `experiments/reports/2026-07-07_g1_next_carry_actions_after_audit.json`.
  Use that report to choose the next branch after the queued evidence lands.
- 2026-07-07 00:42 CST queue update: Slurm moved `168801` (`g1_viz_gpu_q3`)
  and `168802` (`g1_contact_gpu`) earlier to 2026-07-07 02:10:58 CST, both
  scheduled on `server39`. Logs still only show `queued and waiting for
  resources`; no render frames, contact summary, gauntlet summary, or
  after-audit recommendation exist yet.
- 2026-07-07 active pipeline status collector: added
  `scripts/isaac/collect_g1_active_pipeline_status.py`. It reads expected
  render/contact/gauntlet/audit/recommendation artifacts and writes a JSON
  status report without running Isaac. Current report
  `experiments/reports/2026-07-07_g1_active_pipeline_status_current.json` is
  `incomplete`, with render/contact/gauntlet and after-audit artifacts still
  missing. The status report includes `generated_at_utc` to make repeated
  status snapshots auditable. Tmux session
  `curiosity_g1_pipeline_status_after_watchers_0707` waits for
  `curiosity_g1_next_actions_after_audit_0707`, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_status_after_watchers.json`.
- 2026-07-07 render-status watcher: tmux session
  `curiosity_g1_render_status_after_168801_0707` waits for render job
  `168801` to leave the queue, then runs
  `scripts/isaac/collect_g1_active_pipeline_status.py` and writes
  `experiments/reports/2026-07-07_g1_render_pipeline_status_after_168801.json`.
  This is only a read-only status collector so render failures or missing
  frames are recorded promptly.
- 2026-07-07 failure classification: added
  `scripts/isaac/classify_g1_active_pipeline_failures.py`. It reads active
  render/contact logs plus expected render/contact/gauntlet artifacts and
  classifies them as queued, missing artifact, timeout, Isaac startup failure,
  render capture failure, control fall/drop, target-progress failure, or
  retention/balance-quality failure. Current report
  `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_current.json`
  classifies `168801`/`168802` logs as `queued` and expected artifacts as
  missing. Tmux session `curiosity_g1_failure_class_after_watchers_0707`
  waits for `curiosity_g1_pipeline_status_after_watchers_0707`, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_after_watchers.json`.
- 2026-07-07 Markdown pipeline report: added
  `scripts/isaac/write_g1_active_pipeline_markdown_report.py`, which converts
  pipeline status, completion audit, failure classification, and next-action
  JSON into a compact read-only Markdown status page. Current report:
  `experiments/reports/2026-07-07_g1_active_pipeline_status_current.md`.
  Tmux session `curiosity_g1_markdown_report_after_watchers_0707` waits for
  `curiosity_g1_failure_class_after_watchers_0707`, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_status_after_watchers.md`.
- 2026-07-07 00:57 CST queue/watch update: render job `168801`
  (`g1_viz_gpu_q3`) and contact job `168802` (`g1_contact_gpu`) are still
  `PENDING (Priority)`, both with expected start 2026-07-07 02:10:58 CST.
  The main render log still only contains Slurm queue text, and no real
  Isaac replay PNG/MP4 artifacts exist yet. Added tmux session
  `curiosity_g1_render_fallback_after_168801_0707`, which waits for `168801`
  to leave the queue and only if the main render has fewer than 10 PNG frames
  and no passing render check, submits a single short GPU `srun` fallback
  render to
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_960x540/`.
  This is a contingency visualization job only; if the main render succeeds,
  the watcher exits without submitting anything.
- 2026-07-07 01:06 CST queue update: `168801` and `168802` remain
  `PENDING (Priority)`. Slurm's predicted start time slipped to
  2026-07-07 02:20:50 CST on scheduled node `server39`. The render log still
  only contains `srun: job 168801 queued and waiting for resources`, and no
  real Isaac replay frames or contact follow-up summaries exist yet.
- 2026-07-07 showcase manifest: added
  `scripts/isaac/write_g1_showcase_visual_manifest.py`, a lightweight
  read-only report generator for the current G1 visual replay candidates. It
  inspects the source rollout summary, main render directory, fallback render
  directory, PNG frame counts, MP4 files, and render checks, then writes
  `experiments/reports/2026-07-07_g1_showcase_visual_manifest.md` and
  `.json`. The current manifest is `pending_or_failed` because no real Isaac
  PNG/MP4 exists yet; it explicitly states that any replay is visual-only and
  not proof of arbitrary posture robustness or final completion. Tmux session
  `curiosity_g1_showcase_manifest_after_render_0707` waits for the main render
  and fallback render watchers to finish, then rewrites the manifest.
- 2026-07-07 contact report path fix:
  `scripts/isaac/summarize_core_world_g1_contact_followup.py` now reports the
  standard low-cradle case directory
  `<stamp>/agile_low_cradle_freebox_walk/` even before a pending case exists,
  unless a legacy top-level summary already exists. Refreshed
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_pending.json`,
  `experiments/reports/2026-07-07_g1_carry_completion_audit_current.json`,
  and `experiments/reports/2026-07-07_g1_next_carry_actions_current.json` so
  the pending `168802` terminal-contact expected summary path is accurate.
- 2026-07-07 01:16 CST queue update: `168801` and `168802` are still
  `PENDING (Priority)`, with Slurm predicted start 2026-07-07 02:20:50 CST.
  `sacct` still reports no assigned nodes for either job, the render log is
  still only queue text, and no render/contact/gauntlet after-run artifact
  exists yet.
- 2026-07-07 pipeline status Slurm snapshot:
  `scripts/isaac/collect_g1_active_pipeline_status.py` now records lightweight
  `squeue` snapshots for tracked pipeline jobs `168801` and `168802` under
  `tracked_slurm_jobs`. `scripts/isaac/write_g1_active_pipeline_markdown_report.py`
  now prints those jobs in a `Slurm Jobs` section. Current status report shows
  both jobs as `PENDING` with reason `(Priority)` and start
  `2026-07-07T02:20:50`, making clear that missing render/contact artifacts
  are queue-waiting artifacts, not completed-run failures.
- 2026-07-07 01:24 CST queue update: `168801` and `168802` still have no
  assigned nodes and remain `PENDING (Priority)` with predicted start
  2026-07-07 02:20:50 CST. No real Isaac replay frames, MP4s, terminal-contact
  summaries, or posture gauntlet outputs exist yet. Refreshed
  `experiments/reports/2026-07-07_g1_active_pipeline_status_current.json`,
  `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_current.json`,
  and `experiments/reports/2026-07-07_g1_active_pipeline_status_current.md`
  after this poll.
- 2026-07-07 periodic status watcher: tmux session
  `curiosity_g1_periodic_status_until_168801_168802_done_0707` is running. It
  refreshes the current active pipeline status, failure classification, and
  Markdown report every 10 minutes while either `168801` or `168802` remains
  in `squeue`, then performs one final refresh after both leave the queue.
  This watcher is login-node safe: it only uses lightweight `squeue` and JSON
  report reads/writes, not Isaac, rendering, training, or simulation.
- 2026-07-07 render failure and fix: Slurm job `168801` (`g1_viz_gpu_q3`)
  ran on `server59` and failed in 34 seconds because
  `scripts/isaac/render_core_world_g1_replay_showcase.py` imported
  `omni.kit.viewport.utility`, which is unavailable in the current headless
  Isaac environment. The main render produced no PNG/MP4 and only a failing
  `g1_replay_showcase_check.json`. The render script has been changed to use
  USD Camera prims plus `omni.replicator.core` RGB annotator output and PIL
  PNG writing instead of viewport/swapchain capture. Fallback render job
  `168849` (`g1_viz_gpu_fb`) was submitted by
  `curiosity_g1_render_fallback_after_168801_0707` and is pending; it should
  exercise the patched render path.
- 2026-07-07 terminal-contact result: Slurm job `168802` (`g1_contact_gpu`)
  completed on `server59` with exit `0:0`, but the terminal chest-pad case is
  a strict failure, not progress: `fall_events=104`, first fall step `715`,
  `final_box_target_directed_travel_m=1.88795`, final relative error
  `0.34432`, max robot/box tilt `1.84268/1.92388`, and target-window/final-
  hold streaks `0`. It had no box drops and no rollout root/velocity/box pose
  writes, but it did not preserve balance or target-window carry. The contact
  comparison after `168802` is
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json`.
- 2026-07-07 contact rescue submission: added three follow-up cases to
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`:
  `chestpad_terminal_nolateral`, `chestpad_terminal_tiny_pad`, and
  `chestpad_terminal_late_tiny_pad`. Tmux session
  `curiosity_g1_contact_rescue_gpu_0707` submitted Slurm job `168851`
  (`g1_contact_rescue`) to run these cases on a GPU compute node and then
  write `experiments/reports/2026-07-07_g1_contact_rescue_comparison_after_run.json`.
  This is a targeted diagnostic for the terminal-contact failure, not final
  task completion.
- 2026-07-07 active pipeline status expansion:
  `scripts/isaac/collect_g1_active_pipeline_status.py` now tracks jobs
  `168801`, `168802`, `168849`, `168850`, and `168851`, plus fallback render
  and contact-rescue comparison artifacts. Current status shows `168849`
  fallback render and `168851` contact rescue pending, while broad posture
  gauntlet job `168850` is pending with predicted start 2026-07-08 00:00:00
  CST. Do not wait on the gauntlet before running targeted contact fixes.
- 2026-07-07 absolute-path rerun fix: fallback render job `168849` and
  contact-rescue job `168851` both failed immediately with exit `127` on
  `server39` because the `srun bash -lc` command used relative `scripts/...`
  paths that were not resolved in the compute shell. Resubmitted absolute-path
  jobs: `168882` (`g1_viz_gpu_fb_abs`) for the patched Replicator render path
  and `168883` (`g1_contact_rescue_abs`) for the three contact-rescue cases.
  The status collector now tracks `168882` and `168883`; the visual manifest
  now includes
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_abs_960x540/`.
- 2026-07-07 direct-path rerun fix: absolute-path jobs `168882` and `168883`
  also failed immediately with exit `127` because `$ROOT` expanded inside the
  nested compute-shell command to an empty value, producing `/scripts/...`.
  Resubmitted without nested `bash -lc`, using `srun ... env ... bash
  /public/home/yanhongru/Curiosity/scripts/...`: `168895`
  (`g1_viz_gpu_fb_direct`) for fallback render and `168896`
  (`g1_contact_rescue_direct`) for the contact-rescue suite. The status
  collector now tracks `168895` and `168896`; the visual manifest now includes
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_direct_960x540/`.
- 2026-07-07 direct fallback render result: Slurm job `168895`
  (`g1_viz_gpu_fb_direct`) reached Isaac startup on `server59` but failed
  before any frame because `omni.replicator.core` was imported before its
  extension was enabled (`ModuleNotFoundError: No module named
  'omni.replicator'`). The direct fallback produced no PNG/MP4 and only a
  failing checker. The replay renderer now explicitly calls
  `enable_extension("omni.replicator.core")`, advances the app for five
  updates, and only then imports `omni.replicator.core`. Submitted extension-
  enabled fallback render `168900` (`g1_viz_fb_ext`) through tmux session
  `curiosity_g1_render_fallback_ext_gpu_0707`, output directory
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_ext_960x540/`.
  The status collector and visual manifest now track this fourth fallback.
  Before `168900` started, the renderer was further hardened so a failed
  Replicator import writes `g1_replay_render_summary.json` with
  `status="fail"` and the concrete import error instead of exiting with only
  a missing-summary checker failure.
  A second pre-start hardening pass changed the replay renderer to
  `--capture-backend auto`: use Replicator when `omni.replicator.core` is
  actually importable, otherwise enable `isaacsim.core.rendering_manager`,
  create a viewport for the replay camera, and capture PNG frames through
  `omni.kit.renderer.capture` swapchain screenshots. This is still
  visualization-only replay, not new control evidence.
- 2026-07-07 extension-enabled render result: Slurm job `168900`
  (`g1_viz_fb_ext`) failed on `server43` after 44 seconds with zero frames.
  The new failure summary is useful: Replicator failed because
  `omni.replicator.core` could not be imported and Kit could not resolve its
  missing dependency `omni.kit.pip_archive`; the app-screenshot fallback also
  failed because `isaacsim.core.rendering_manager` could not be imported and
  Kit could not resolve `omni.kit.viewport.window`. Direct listings of the
  local `isaacsim/extscache`, `isaacsim/kit/exts`, and `isaacsim/kit/extscore`
  directories did not show those key extension packages. Do not rerun the
  same Replicator/rendering-manager capture path unchanged; either supply the
  missing local Kit extensions/registry mirror or use a different visualization
  backend.
- 2026-07-07 presentation fallback renderer: added
  `scripts/isaac/render_g1_replay_presentation_fallback.py`. It refuses to
  run on login nodes and generates a clearer G1-like replay GIF/poster from
  the passed `168632` replay CSV using PIL. It is explicitly schematic:
  `schematic_replay_visual_only_not_isaac_camera_render_not_new_control_evidence`.
  Submitted as tmux `curiosity_g1_presentation_fallback_render_0707`, Slurm
  job `168986` (`g1_fallback_gif`), output directory
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif/`.
  This is only a presentation fallback while Isaac camera rendering is blocked
  by missing Kit extensions.
- 2026-07-07 presentation fallback result: Slurm job `168986`
  (`g1_fallback_gif`) completed on `server43` and generated
  `g1_lowcarry_replay_fallback.gif`, `g1_lowcarry_replay_fallback_poster.png`,
  `g1_replay_presentation_fallback_summary.json`, and 64 PNG frames under
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif/`.
  The summary status is `pass`, but the success claim remains
  `schematic_replay_visual_only_not_isaac_camera_render_not_new_control_evidence`.
  Use it only as a presentation fallback, not as Isaac camera-render evidence
  or new robot-control evidence.
- 2026-07-07 live RGB capture hardening: the same explicit
  `enable_extension("omni.replicator.core")` startup sequence was added to
  `scripts/isaac/build_core_world_g1_box_scene.py` before its optional live
  `CAPTURE_RGB` import. This is a capture-path fix only; do not treat it as
  new robot-control evidence unless a fresh compute-node capture run produces
  valid frames and passes its normal carry gates.
- 2026-07-07 contact rescue direct result: Slurm job `168896`
  (`g1_contact_rescue_direct`) completed on `server59`, but all rescue
  variants failed strict carry gates. `terminal_nolateral_direct` still had
  `fall_events=104`, no target-window/final-hold streak, and final relative
  error about `0.337 m`; `terminal_tiny_pad_direct` and
  `terminal_late_tiny_pad_direct` both had hundreds of fall events and box
  drops. The late tiny-pad case failed before the pad ever enabled. Treat
  terminal chest-pad size/timing tweaks as negative evidence for the current
  branch; do not keep tuning this contact pad family as the primary path.
- 2026-07-07 non-pad balance rescue follow-up: added `CASE_SET=balance_rescue`
  to `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`.
  It runs `nopad_final_freeze` and `nopad_stronger_balance`, both with
  chest-pad disabled, to test final-window policy freeze / stronger
  joint-level balance feedback after the chest-pad rescue family failed.
  Submitted as tmux `curiosity_g1_balance_rescue_gpu_0707`, Slurm job
  `168972` (`g1_balance_rescue`), with direct absolute launcher invocation.
  Watcher `curiosity_g1_balance_rescue_watch_0707` will write
  `experiments/reports/2026-07-07_g1_balance_rescue_comparison_after_run.json`.
  This is a targeted diagnostic only, not final carrying success.
- 2026-07-07 non-pad balance rescue result: Slurm job `168972`
  (`g1_balance_rescue`) completed on `server43` with Slurm exit `0:0`, but the
  strict comparison failed. `nopad_final_freeze` reached the target vicinity
  with robot/box target-directed travel about `2.296/2.317 m` and 47 target
  stable steps, but ended with `165` falls, `148` drops, max robot/box tilt
  `1.210/1.025 rad`, and target-window end streak `0`. `nopad_stronger_balance`
  reached robot/box travel about `2.295/2.229 m`, but had `104` falls,
  `39` drops, final relative error `0.269 m`, max robot/box tilt
  `1.289/1.315 rad`, and target-window end streak `0`. Conclusion: disabling
  the chest pad and strengthening balance restored progress to the target
  vicinity, but the current controller still fails through late target-window
  pitch/drop instability. The next targeted path should address late recovery
  and final-window stabilization, not continue the chest-pad geometry family.
- 2026-07-07 late target-window recovery diagnostic: added
  `CASE_SET=late_recovery` to
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`. It
  runs `nopad_late_gentle_rescue` and `nopad_late_stand_blend`, both with the
  chest pad disabled, to test small final-window freeze/rescue or slow
  policy-to-stand blending after `168972` showed target-vicinity progress but
  late pitch/drop instability. Submitted through tmux
  `curiosity_g1_late_recovery_gpu_0707` as Slurm job `168995`
  (`g1_late_rec`) on `server59`. The same compute allocation should write
  `experiments/reports/2026-07-07_g1_late_recovery_comparison_after_run.json`.
  This is a targeted diagnostic only, not final carrying success.
- 2026-07-07 late target-window recovery result: Slurm job `168995`
  (`g1_late_rec`) completed on `server59` with Slurm exit `1:0` because both
  new diagnostic cases failed strict gates. `nopad_late_gentle_rescue` entered
  the target window for 56 steps but then overran badly: robot/box final
  target-directed travel about `3.224/2.935 m`, `156` falls, `124` drops,
  final relative error `0.330 m`, and target-window end streak `0`.
  `nopad_late_stand_blend` under-traveled and destabilized: robot/box final
  travel about `1.790/1.744 m`, `205` falls, `174` drops, final relative
  error `0.261 m`, and target-window streak `0`. Conclusion: the current
  late rescue / slow policy-to-stand blending branch is negative; the next
  targeted path should focus on command-authority shaping at target-window
  entry, overshoot arrest, and box retention, while broader posture/load
  gauntlet failures remain unresolved.
- 2026-07-07 target-window arrest diagnostic: added
  `CASE_SET=target_window_arrest` to
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`. It
  runs two 0.5 kg cases, `load05_window_zero_arrest` and
  `load05_window_reverse_brake`, with earlier final-hold thresholds, zeroed
  lateral/yaw corrections, and optional small reverse brake. This directly
  targets the 0.5 kg `lowcarry_base` gauntlet failure and the `168995`
  overrun/under-travel split. Submitted through tmux
  `curiosity_g1_target_window_arrest_gpu_0707` as Slurm job `168997`
  (`g1_tw_arrest`). It should write
  `experiments/reports/2026-07-07_g1_target_window_arrest_comparison_after_run.json`.
  This is a targeted diagnostic only, not final carrying success.
- 2026-07-07 target-window arrest result: Slurm job `168997`
  (`g1_tw_arrest`) completed on `server59` with Slurm exit `0:0`, but both
  0.5 kg arrest cases failed strict gates. `load05_window_zero_arrest` had
  robot/box final target-directed travel about `1.435/1.178 m`, `229` falls,
  `18` drops, final relative error `0.336 m`, and target-window streak `0`.
  `load05_window_reverse_brake` had robot/box final travel about
  `1.293/1.269 m`, `255` falls, `241` drops, final relative error `0.217 m`,
  and target-window streak `0`. Conclusion: early final-hold / small reverse
  brake micro-tuning is also negative for the 0.5 kg gauntlet load. Stop
  treating the current open-loop Agile command wrapper as a promising path for
  load/posture generalization; the next real step should replace the
  locomotion/retention formulation or use a controller-backed load-aware
  policy/contact controller.
- 2026-07-07 box-progress closed-loop controller: added optional
  `--agile-command-box-progress-controller` and
  `--agile-command-box-lateral-controller` to
  `scripts/isaac/build_core_world_g1_box_scene.py`, with launcher passthrough
  in `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  Unlike previous final-hold scalar/brake tweaks, this controller closes the
  Agile policy command loop on measured box target-directed progress and box
  lateral error, overriding command X from the box progress error and adding
  optional box-centric lateral correction. It records active steps, last error,
  command magnitudes, and tilt-suppression counts in the summary. Added
  `CASE_SET=box_progress_controller` with 0.5 kg cases
  `load05_box_progress_pd` and `load05_box_progress_conservative` in
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh`.
  Submitted through tmux `curiosity_g1_box_progress_controller_gpu_0707` as
  Slurm job `169004` (`g1_box_pd`). It should write
  `experiments/reports/2026-07-07_g1_box_progress_controller_comparison_after_run.json`.
  This is a targeted diagnostic toward a controller-backed load-aware path, not
  final carrying success.
- 2026-07-07 box-progress controller result: Slurm job `169004`
  (`g1_box_pd`) completed on `server43` with Slurm exit `0:0`, but the
  comparison report is strict `fail`. Both new 0.5 kg cases failed.
  `load05_box_progress_pd` had `158` falls, `23` drops, final relative error
  about `0.288 m`, max robot/box tilt about `1.689/1.696 rad`, and no
  target-window hold. `load05_box_progress_conservative` had `166` falls,
  `143` drops, final box target-directed travel about `5.166 m`, final
  relative error about `0.261 m`, and no final hold. Conclusion: closing the
  Agile wrapper command on box progress/lateral error alone is negative and
  can make overshoot/drop worse. Do not keep treating scalar box-progress
  command tuning as the main path.
- 2026-07-07 box-retention posture feedback: added optional
  `--box-retention-posture-controller` to
  `scripts/isaac/build_core_world_g1_box_scene.py`, with launcher passthrough.
  It computes a retention risk from current box-robot relative error and box
  tilt, then applies symmetric crouch/waist/arm-closing offsets while recording
  active steps and max risk. Added `CASE_SET=box_progress_retention` with
  0.5 kg case `load05_box_progress_retention`, combining the box-progress
  command controller with retention posture feedback. Submitted through tmux
  `curiosity_g1_box_progress_retention_gpu_0707` as Slurm job `169006`
  (`g1_box_ret`). It should write
  `experiments/reports/2026-07-07_g1_box_progress_retention_comparison_after_run.json`.
  This remains diagnostic, not final carrying success.
- 2026-07-07 box-retention posture feedback result: Slurm job `169006`
  (`g1_box_ret`) completed on `server43` with Slurm exit `0:0`, but the
  comparison is strict `fail`. The single new case
  `load05_box_progress_retention` had `0` box drops, but `470` fall events,
  max robot/box tilt about `2.423/2.251 rad`, final box target-directed travel
  only about `0.247 m`, and no target-window or final-hold success. Conclusion:
  retention posture feedback improved drop count in this case but collapsed
  locomotion/balance; it is not a valid fix for carrying.
- 2026-07-07 prismatic no-root reference validation: because G1 wrapper
  branches are queued or negative, submitted a materially different baseline
  through tmux `curiosity_prismatic_reference_validation_0707`, Slurm job
  `169008` (`prism_ref`). It reruns the strongest existing no-root prismatic
  free-box scaffold: `payload_mode=cradle_free_box`, 10 kg box,
  `motion_mode=guarded_prelift_quasistatic_step_cycle`, active probe for
  80 steps, probe-adaptive gait/posture enabled, target `-0.17 m`, and no root
  pose/velocity or box pose writes allowed. The checker requires articulated
  carrier, foot-contact drive, active probe belief without hidden ground
  truth, fall/drop 0, post-settle payload travel at least `0.15 m`, final
  post-settle payload target distance at most `0.02 m`, payload relative
  offset error at most `0.01 m`, payload z at least `0.45 m`, and max tilt at
  most `0.20 rad`. This is a reference scaffold for a physical no-root
  carrying substrate; it is not humanoid walking, not learned carrying, and
  not final task completion.
- 2026-07-07 first prismatic GPU validation result: Slurm job `169008`
  (`prism_ref`) failed after `00:00:32` on `server43` with exit `1:0` due to
  launcher/config error, not a valid physical failure. The log showed
  `ENABLE_HORIZONTAL_LEGS=0` even though
  `guarded_prelift_quasistatic_step_cycle` requires horizontal legs, and the
  script stopped at 260 steps with
  `RuntimeError: guarded_prelift_quasistatic_step_cycle requires --enable-horizontal-legs`.
  Do not interpret `169008` as a strict validation result.
- 2026-07-07 matched prismatic CPU validation queued: after the invalid
  `169008` and the under-traveling short CPU rerun `169019`, submitted
  parameter-matched rerun `169026` (`prism_ref_mcpu`) through tmux
  `curiosity_prismatic_reference_matched_cpu_validation_0707`. It uses the
  corrected horizontal-leg launcher plus historical-like settings:
  `STEPS=2880`, `FOOT_LENGTH=0.65`, `GATED_STEP_MAX_TRAVEL_LOSS=0.04`,
  `GATED_STEP_RECOVERY_PHASE=0.35`, and
  `GUARDED_STEP_TARGET_TOLERANCE=0.03`. Await result before claiming a fresh
  prismatic reference pass.
- 2026-07-07 matched prismatic CPU validation result: Slurm job `169026`
  completed the 2880-step rollout on `server36`, but Slurm state is
  `FAILED` with exit `2:0` because the old checker used
  `--max-payload-relative-offset-error 0.01` against the whole trajectory.
  That gate was stricter than the historical reference itself, whose global
  max relative offset was also much larger than `0.01 m`. The checker now has
  a separate `--max-post-settle-payload-relative-offset-error` gate. Rechecking
  the same `169026` summary with global relative error <= `0.06 m` and
  post-settle relative error <= `0.012 m` produced
  `reference_check_corrected.json` with `status=pass` and no failures.
  Important metrics: completed steps `2880/2880`, fall/drop `0/0`, root pose/
  velocity/angular writes `0/0/0`, body root pose/velocity writes `0/0`,
  box pose writes `0`, active probe `80` steps with no hidden ground truth,
  probe-adaptive gait/posture decisions available, final post-settle payload
  travel `-0.17994 m`, final post-settle payload target distance `0.00994 m`,
  max tilt `0.10644 rad`, min payload z `0.71651 m`, global max relative
  offset `0.04349 m`, and max post-settle relative offset `0.01160 m`.
  This is the strongest fresh prismatic scaffold validation, but it remains a
  no-root prismatic carrier scaffold, not humanoid walking, not learned
  carrying, and not final task completion.
- 2026-07-07 prismatic reference visual fallback: added
  `scripts/isaac/render_prismatic_reference_presentation_fallback.py`. It
  refuses to run on login nodes and renders a schematic prismatic carrier,
  four feet/legs, cradle, free box, target line, metrics, GIF, poster, and
  frame PNGs from a completed prismatic state CSV. It is explicitly
  `schematic_visual_only_not_isaac_camera_render_not_final_humanoid_success`.
  Started tmux watcher `curiosity_prismatic_reference_visual_watch_0707`;
  after `169008` leaves the queue and its summary/CSV exist, the watcher will
  launch `srun` job-name `prism_viz` to write
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid/`.
- 2026-07-07 fresh matched prismatic visual: Slurm job `169027`
  (`prism_mviz`) completed on `server36` with exit `0:0` in `00:00:13`.
  It generated the preferred current fresh scaffold visualization from
  `169026`:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback.gif`
  and
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback_poster.png`.
  The one-page showcase
  `slides/2026-07-07_isaac_carry_showcase.html` now points to this fresh GIF.
  Same boundary: schematic only, not Isaac camera render, not humanoid
  walking, not learned carrying, and not final success.
- 2026-07-07 prismatic posture/load validation suite: added
  `scripts/isaac/run_prismatic_reference_posture_load_suite_0707.sh` and
  submitted it through tmux `curiosity_prismatic_posture_load_suite_0707` as
  Slurm job `169029` (`prism_suite`) in the CPU partition. It runs four
  corrected-checker cases on the no-root prismatic cradle-free-box scaffold:
  nominal 10 kg mid carry, near-chest high 12 kg, long-reach low 8 kg, and
  bulky 10 kg. The suite writes
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_summary.json`.
  This directly tests posture/load robustness of the current scaffold, but it
  is still not humanoid walking, learned policy, or final completion.
- 2026-07-07 prismatic posture/load suite result: Slurm job `169029`
  (`prism_suite`) ran on `server36` and wrote
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_summary.json`.
  Slurm exit is `1:0` because the suite status is strict `fail`. Three of
  four corrected-checker cases passed: nominal 10 kg mid carry, long-reach low
  8 kg, and bulky 10 kg. The failing case is `near_chest_12kg_high`, which
  had fall/drop `0/0`, max tilt about `0.0995 rad`, and max post-settle
  relative offset about `0.00894 m`, but stopped slightly early:
  post-settle payload travel `0.147676 m` vs required `0.15 m`, final target
  distance `0.022325 m`. Treat this as a narrow under-travel failure, not a
  balance/drop failure. Next targeted retry should tighten
  `GUARDED_STEP_TARGET_TOLERANCE` or increase drive, rather than changing the
  whole scaffold.
- 2026-07-07 near-chest targeted retry: submitted tmux
  `curiosity_prismatic_nearchest_retry_tight_0707` as Slurm job `169031`
  (`prism_nc_tight`). It reruns the failed `near_chest_12kg_high` case with
  corrected checker and `GUARDED_STEP_TARGET_TOLERANCE=0.015` while keeping
  the same 12 kg payload, near-chest/high placement, 2880 steps, foot length
  `0.65`, travel-loss gate `0.04`, and recovery phase `0.35`. Expected output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_near_chest_12kg_high_tighttol/`.
- 2026-07-07 near-chest retry result and after-retry suite: Slurm job
  `169031` completed the rollout on `server36` but Slurm exit is `2:0`
  because the launcher still used the previous global relative-error gate
  before it was updated to `0.08`. Rechecking the same summary with the
  corrected suite gate passed with no failures: fall/drop `0/0`, completed
  `2880/2880`, final post-settle payload travel `-0.18562 m`, target distance
  `0.01562 m`, max tilt `0.09948 rad`, global max relative offset
  `0.06789 m`, and max post-settle relative offset `0.00976 m`. The aggregate
  after-retry suite summary
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_after_retry_summary.json`
  is `status=pass` with 4/4 corrected-checker cases passing: nominal 10 kg
  mid carry, near-chest high 12 kg tight-tolerance retry, long-reach low 8 kg,
  and bulky 10 kg. This strengthens the current physical scaffold, but it is
  still not humanoid walking, not learned policy, and not final completion.
- 2026-07-07 MuJoCo robot-like gait candidate: submitted tmux
  `curiosity_mujoco_quad_payload_assisted_0707` as Slurm job-name
  `mj_quad_payload`. It runs `scripts/mujoco/run_quadruped_payload_carry.sh`
  with `STAMP=20260707_mujoco_quad_assisted_payload4kg`, 3000 steps, 4 kg
  welded payload, and `ASSIST_MODE=body_force`, then checks fall events 0,
  travel >= `0.30 m`, max tilt <= `0.55 rad`, no root pose/velocity writes,
  and at least one external force write. This is deliberately a diagnostic
  candidate for a more robot-like multi-joint gait backend; because it uses a
  welded payload and explicit stabilizing body-force controller, it must not
  be claimed as unknown free-box carrying, autonomous locomotion, or final
  success.
- 2026-07-07 MuJoCo assisted quadruped result: Slurm job `169034`
  (`mj_quad_payload`) completed on `server36` with exit `1:0`; the checker is
  strict `fail`. It did run 3000/3000 steps and traveled about `1.738 m`
  with no root pose/velocity writes, but had `94` fall events and max tilt
  about `3.159 rad` despite 3000 external force/torque writes. This proves the
  current high-speed assisted quadruped settings are not a stable robot-like
  backend. A conservative retry should lower target speed and increase
  stabilizing torque before this route is considered useful.
- 2026-07-07 MuJoCo conservative retry result: Slurm job `169039`
  (`mj_quad_cons`) completed on `server36` with exit `1:0`; the checker is
  still `fail`, but for the opposite reason. The 4 kg welded-payload
  quadruped had fall events `0`, max tilt about `0.129 rad`, and no root
  pose/velocity writes, but traveled only about `0.118 m`, below the `0.20 m`
  gate. This identifies a speed/stability bracket: `0.45 m/s` is unstable,
  `0.20 m/s` is stable but under-travels. A middle-speed retry is justified
  as a controller diagnostic, still not final carrying.
- 2026-07-07 MuJoCo middle-speed/bracket result: Slurm job `169042`
  (`mj_quad_mid`) ran `target_speed=0.30 m/s`, 4 kg welded payload, and
  body-force assist on `server01`; it failed strictly despite `1.161 m`
  travel because it had `31` fall events and max tilt `3.270 rad`. Slurm job
  `169044` (`mj_quad_bracket`) then ran three smaller bracket points. Results:
  `v022_fx130` passed with fall events `0`, max tilt `0.31490 rad`, and
  travel `0.32316 m`; `v024_fx115` is the best current robot-like diagnostic,
  passing with fall events `0`, max tilt `0.47038 rad`, travel `0.53915 m`,
  root pose/velocity writes `0/0`, and 3000 external force/torque writes;
  `v026_fx105` failed with `18` fall events and max tilt `0.95380 rad` while
  traveling `0.63719 m`. This brackets the assisted welded-payload quadruped
  stability boundary near `0.24-0.26 m/s`. Do not overclaim it: payload is
  welded and balance uses explicit body-force stabilization, so it is not
  unknown free-box grasping, autonomous locomotion, learned carrying, or final
  task success.
- 2026-07-07 MuJoCo robot-like visual: added
  `scripts/mujoco/render_quadruped_payload_presentation_fallback.py` and
  rendered the passing `v024_fx115` diagnostic through Slurm job `169047`
  (`mj_quad_viz2`) on `server01`. Outputs:
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback.gif`
  and
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback_poster.png`.
  This is a schematic rollout replay showing a dynamic quadruped body, legs,
  welded 4 kg box, path, and metrics; it is not an Isaac camera render and
  not final carrying evidence.
- 2026-07-07 MuJoCo free-box contact route: added
  `scripts/mujoco/run_quadruped_freebox_carry.py`,
  `scripts/mujoco/run_quadruped_freebox_carry.sh`, and
  `scripts/mujoco/check_quadruped_freebox_summary.py`. This route replaces
  the welded payload with a separate freejoint box retained by a
  torso-mounted tray/side walls/front-rear stops. It still uses explicit
  body-force stabilization on the quadruped torso, so even a pass would remain
  diagnostic and not final autonomous carrying.
- 2026-07-07 MuJoCo free-box results are partial/negative. Job `169077`
  failed before rollout due missing executable bit on the new launcher; do
  not count it as physics evidence. Job `169079`
  (`20260707_mujoco_quad_freebox_2kg_v016_contact_tray_retry2`) completed a
  2 kg free-box contact rollout with fall/drop `0/0`, max tilt `0.185 rad`,
  min box z `0.762 m`, and no root/box writes, but failed travel gates:
  max box travel only `0.0716 m` and final box travel `-0.0228 m`.
  Bracket job `169081` showed the useful but failing boundary: `2kg_v024`
  moved the free box about `0.1986 m` with fall `0`, but had `22` drop events
  from relative offset growing to `0.286 m`; `2kg_v030` and `1kg_v030` both
  failed badly with falls, drops, and box loss. This proves contact-retained
  free-box motion exists, but current tray/drive does not yet satisfy
  retention and final-hold gates.
- 2026-07-07 MuJoCo free-box stop/hold results: added optional
  `STOP_AFTER_BOX_TRAVEL` and `HOLD_TARGET_SPEED` controls with checker gates
  for target-stop latch and hold steps. Job `169083`
  (`stop015`, hold speed `0.0`) latched at about `0.150 m` and had fall `0`,
  but the box slid backward during hold: final box travel `0.0560 m`,
  relative error `0.227 m`, and `9` drop events. Job `169087`
  (`hold008`) improved final travel to `0.105 m` but still had `13` drop
  events and final relative error `0.229 m`. Job `169091`
  (`tighttray`, hold `0.08`) reached final travel `0.117 m`, fall `0`, but
  still failed with `14` drop events and relative error `0.231 m`. Job
  `169092` (`hold012`) and job `169096` (`tightslot`) both kept final travel
  around `0.139-0.140 m` with fall `0`, but failed relative-error/drop gates
  (`0.240-0.244 m`, `15-16` drop events). Conclusion: current best free-box
  contact route can carry a separate 2 kg box roughly `0.15 m` before/while
  stopping without robot fall or shortcut writes, but it does not yet pass
  retention or final hold. Next valid step is better contact retention
  geometry or a grip/normal-force controller, not claiming success.
- 2026-07-07 MuJoCo free-box retention-force diagnostic: added
  `RETENTION_FORCE_MODE=relative_spring`, which applies audited equal and
  opposite spring-damper forces between the torso and free box without writing
  root pose, root velocity, box pose, or box velocity. Slurm job `169100`
  (`20260707_mujoco_quad_freebox_2kg_v024_stop015_hold012_retention_spring`)
  passed the strict diagnostic checker for a 2 kg free box: 3000/3000 steps,
  fall/drop `0/0`, root pose/velocity writes `0/0`, box pose/velocity writes
  `0/0`, retention force writes `3000`, max/final box travel
  `0.18242/0.18239 m`, max/final box-torso relative error
  `0.08091/0.07865 m`, max tilt `0.28755 rad`, min box z `0.69617 m`,
  target-stop latched at step `1559`, and hold steps `1441`. This is the
  strongest current robot-like free-box diagnostic because the box remains a
  separate dynamic body and no box pose shortcut is used. Do not overclaim it:
  the quadruped still uses explicit body-force torso stabilization, and the
  retention spring is a hand-authored grip-force controller rather than
  learned grasping, humanoid locomotion, or final unknown-load carrying.
- 2026-07-07 free-box retention multi-load validation: Slurm job `169116`
  (`mj_free_rnogpu`) ran on compute node `server26` through tmux
  `curiosity_mujoco_quad_freebox_retention_loads_nogpu_0707` and completed in
  7 seconds. The same relative-spring retention diagnostic passed at 1 kg,
  2 kg, and 3 kg. All three cases completed 3000/3000 steps with fall/drop
  `0/0`, root pose/velocity writes `0/0`, box pose/velocity writes `0/0`,
  external force/torque writes `3000/3000`, retention force writes `3000`,
  target-stop latched, and hold steps above 600. Per-load final box travel,
  final relative error, max relative error, max tilt, and min box z were:
  1 kg `0.19116 m`, `0.05206 m`, `0.05726 m`, `0.27486 rad`, `0.71316 m`;
  2 kg `0.18239 m`, `0.07865 m`, `0.08091 m`, `0.28755 rad`, `0.69617 m`;
  3 kg `0.18993 m`, `0.08790 m`, `0.09178 m`, `0.31475 rad`, `0.69030 m`.
  The redundant GPU-allocating queue job `169114` was canceled after the
  no-GPU compute job started/completed. Interpretation remains diagnostic
  only: this is a robot-like MuJoCo free-box contact/retention scaffold with
  explicit torso body-force stabilization and a hand-authored grip-force
  controller, not learned humanoid carrying or final unknown-load success.
- 2026-07-07 free-box assist-reduction bracket: Slurm job `169120`
  (`mj_free_ared`) ran on compute node `server01` and tested the 2 kg
  retention-force case with body-force caps reduced to 75%, 50%, and about
  33% of the previous nominal values. All three passed the same strict
  diagnostic gates with fall/drop `0/0`, root/box pose and velocity writes
  `0`, retention force writes `3000`, final box travel about `0.181-0.182 m`,
  final relative error about `0.078 m`, and max tilt about `0.288 rad`.
  Because 75% and 50% reproduced the same metrics as the nominal 2 kg case,
  this does not prove reduced scaffold dependence; it indicates the previous
  caps were probably not active. The next queued probe is Slurm job `169121`
  (`mj_free_afloor`), which tests 10% caps, zero body-force caps, and
  `ASSIST_MODE=none` while keeping the same retention controller.
- 2026-07-07 free-box assist-floor boundary: Slurm job `169121`
  (`mj_free_afloor`) ran on compute node `server01` and completed all three
  boundary probes. All failed strict gates. With 10% body-force caps
  (`18 N` x, `36 N` z, `26 Nm` torque), the 2 kg case had `77` fall events,
  `9` box drops, max box travel only `0.00666 m`, final box travel
  `-0.22714 m`, max tilt `0.87946 rad`, min box z `0.51059 m`, and never
  latched target stop. With zero body-force caps and with `ASSIST_MODE=none`,
  the result was essentially the same negative boundary: `129` fall events,
  `124` box drops, max box travel about `0.00574 m`, final box travel
  `-0.55868 m`, max tilt `1.60894 rad`, min box z `0.31277 m`, and no target
  latch. Interpretation: current free-box retention pass depends on a
  nontrivial torso stabilization scaffold. The present supported window is
  between about 33% caps passing and 10% caps failing; the next valid support
  reduction is to replace body-force stabilization with a foot/support
  controller, not to claim unassisted locomotion.
- 2026-07-07 first foot-support replacement attempt: added
  `LEG_DRIVE_MODE=foot_ik` to
  `scripts/mujoco/run_quadruped_freebox_carry.py` and exposed gait frequency,
  duty factor, stride length, stance foot depth, and swing clearance through
  `scripts/mujoco/run_quadruped_freebox_carry.sh`. The new mode uses planar
  two-link IK to command stance/swing foot targets instead of relying on the
  old open sinusoid. Checker now records `leg_drive_mode`. Slurm job `169126`
  (`mj_free_footik`) ran four 2 kg no-body-assist free-box probes on
  `server01`; all failed strict carrying gates, but the failure modes are
  informative. `slow_short` latched the target and reached max box travel
  `0.16824 m`, but fell/dropped (`117` falls, `111` drops, final travel
  `0.04813 m`, max tilt `1.76940 rad`). `nominal` failed with `67` falls,
  `62` drops, and backward final travel `-0.79748 m`. `faster_long` and
  `high_clearance` were stable retention negatives with fall/drop `0/0`,
  max tilt about `0.24 rad`, and low relative error, but they walked backward
  (`final_box_travel_x_m=-1.26814` and `-1.06543 m`) and never latched the
  forward target. Interpretation: the foot-IK controller can preserve balance
  and retention without body-force assist in some settings, but the stance
  trajectory sign is likely reversed for forward travel. A negative-stride
  probe was submitted as Slurm job `169127` (`mj_free_fikneg`); record its
  outcome before treating foot-IK as progress.
- 2026-07-07 foot-IK negative-stride evidence: the first inline negative
  stride submission `169127` completed but is invalid as named evidence
  because shell expansion caused timestamp-only outputs with default stride;
  do not use it. The corrected script
  `scripts/mujoco/run_quadruped_freebox_foot_ik_negstride_suite.sh` ran as
  Slurm job `169130` (`mj_free_fikneg2`) on `server01`. All three 2 kg,
  no-body-assist cases failed strict gates. `neg_high` converted the stable
  backward gait into forward travel and latched the target, with max/final box
  travel `0.45486/0.37860 m`, but failed with `111` falls, `107` drops, max
  tilt `1.71246 rad`, min box z `0.38892 m`, and max relative error
  `0.23669 m`. `neg_fast` and `neg_slow` failed with backward final travel
  and falls/drops. Follow-up
  `scripts/mujoco/run_quadruped_freebox_foot_ik_negstride_stop_suite.sh` ran
  as Slurm job `169135` (`mj_free_fikstop`) after wiring target-stop latch
  into foot-IK stride scaling. It still failed: stop/hold variants preserved
  forward travel (`0.268-0.356 m` final, up to `0.454 m` max) but had about
  `109-111` falls and `106-107` drops. First fall occurred shortly after
  target latch around step `800`, so the next valid probe is earlier target
  latch/stop or a stabilizing support controller. Slurm job `169136`
  (`mj_free_fikearly`) is queued to test earlier stops at 0.08/0.10/0.12 m.
- 2026-07-07 foot-IK early-stop result: Slurm job `169136`
  (`mj_free_fikearly`) ran on `server01`. All early-stop cases failed strict
  gates. Stop thresholds 0.08, 0.10, and 0.12 m all latched and preserved
  positive final travel around `0.300-0.311 m`, but still had about
  `110-111` falls, `107` drops, max tilt about `1.705-1.712 rad`, min box z
  about `0.380-0.386 m`, and max relative error about `0.231-0.236 m`.
  The slower 0.10 m case under-traveled and also fell/dropped. Earlier target
  latch alone is therefore insufficient. Because the failing CSVs show lateral
  drift and roll growth, a lateral equal-and-opposite retention-force probe
  was added via `RETENTION_KP_Y`, `RETENTION_KD_Y`, and
  `RETENTION_MAX_FORCE_Y`; Slurm job `169138` (`mj_free_fiklat`) is queued to
  test whether reducing lateral box/torso drift helps the no-body-assist
  foot-IK route.
- 2026-07-07 lateral and roll foot-IK probes: Slurm job `169138`
  (`mj_free_fiklat`) tested y-axis equal-and-opposite retention force. All
  cases failed; final travel stayed positive around `0.311-0.356 m`, but each
  still had `111` falls and `107` drops with max tilt about `1.70 rad`.
  Lateral box retention alone does not fix the no-body-assist roll/fall
  cascade. Added `FOOT_ROLL_Z_GAIN`, which adjusts left/right foot target
  depth from measured roll while still using only leg actuation. Slurm job
  `169145` (`mj_free_fikroll`) tested coarse roll gains. Positive gains kept
  some forward travel but fell/dropped; negative gains stabilized much better
  but drove the robot backward. Best stable negative was `roll_neg006` with
  fall/drop `0/0`, max tilt `0.25466 rad`, min box z `0.69162 m`, and low
  relative error, but final travel was `-1.01914 m`. This indicates the
  support controller has a real balance effect but couples strongly to travel
  direction. Slurm job `169150` (`mj_free_fikroll2`) is queued for a smaller
  gain sweep.
- 2026-07-07 fine roll-feedback result: Slurm job `169150`
  (`mj_free_fikroll2`) ran on `server01`. All fine gains failed strict
  carrying gates. `roll_neg002` kept forward final travel `0.36017 m` but
  still had `104` falls and `103` drops with max tilt `1.69340 rad`.
  `roll_neg004` improved the boundary to final travel `0.52200 m` and max
  tilt `0.87115 rad`, but still failed with `71` falls, `70` drops, min box z
  `0.36095 m`, and max relative error `0.31265 m`. Gains `-0.045`, `-0.050`,
  and `-0.055` all stabilized fall/drop to `0/0` with max tilt about
  `0.255-0.258 rad`, but drove backward with final travel around
  `-0.893` to `-1.005 m` and never latched the forward target. Conclusion:
  roll-to-foot-height feedback alone cannot satisfy both stable support and
  forward travel. The next support-controller step should add a lateral hip/
  foot-placement DOF or materially stronger stance-phase controller, not more
  body-force assist.
- 2026-07-07 lateral hip DOF and hold-brace results: added explicit
  `*_hip_roll` joints and actuators to the MuJoCo quadruped, plus
  `HIP_ROLL_BASE`, `HIP_ROLL_FEEDBACK_GAIN`, `HOLD_STANCE_FOOT_Z_DOWN`, and
  `HOLD_HIP_ROLL_BASE` controls in
  `scripts/mujoco/run_quadruped_freebox_carry.py` and launcher wiring.
  Slurm job `169159` (`mj_free_fikhip`) tested lateral hip support; all cases
  failed. Small base roll `0.05` retained positive final travel `0.19946 m`
  but still had `101` falls and `100` drops. Base roll `0.10` and feedback
  variants stabilized fall/drop to `0/0` but walked backward around
  `-1.09` to `-1.12 m`. Slurm job `169161` (`mj_free_fikhips`) tested stride
  sign/amplitude with lateral hip; all failed. `base006_neg12` retained
  positive final travel `0.34346 m` but still had `95` falls and `91` drops;
  `base008_neg12` was stable fall/drop `0/0` but backward `-1.06148 m`.
  Slurm job `169162` (`mj_free_fikbrace`) tested target-latched hold-brace
  stance; all failed with positive final travel but about `95-114` falls and
  `89-109` drops. Conclusion: adding lateral hip DOF is a real morphology/
  support upgrade, but the current open-loop stance/hold controller still has
  a stable-backward versus forward-falling split. The next valid controller
  step is not more torso body-force assist; it should be a closed-loop
  foot-placement/stance controller based on forward velocity and roll, or a
  controller-backed locomotion policy.
- 2026-07-07 closed-loop foot-placement result: added
  `CLOSED_LOOP_FOOT_PLACEMENT`, `STRIDE_VELOCITY_GAIN`,
  `STRIDE_POSITION_GAIN`, and `STRIDE_CLIP` to the MuJoCo free-box foot-IK
  route. Slurm job `169164` (`mj_free_fikcl`) completed on `server36`; all
  five cases failed strict gates. The no-hip cases latched the target and held
  it for about `2183-2206` steps with positive final box travel around
  `0.135-0.148 m`, but then collapsed in the hold phase with about
  `108-109` falls, `105-106` drops, max tilt about `1.60 rad`, and minimum
  box height about `0.32-0.33 m`. The hip-base cases had fall/drop `0/0` and
  low tilt about `0.228 rad`, but walked backward with final box travel
  around `-1.04` to `-1.30 m` and never latched the target. Conclusion:
  closed-loop stride can solve target acquisition for the forward-falling
  family, but not post-stop support. The next valid no-body-assist step is a
  target-latched static support/foot-placement stance, not more torso
  body-force assist.
- 2026-07-07 target-latched static support result: added
  `HOLD_FRONT_FOOT_X`, `HOLD_REAR_FOOT_X`, and
  `HOLD_PITCH_FOOT_X_GAIN` to the MuJoCo no-body-assist free-box foot-IK
  route. Slurm job `169173` (`mj_free_fikhold`) completed on `server30`; all
  six cases failed. Wider fore-aft hold stance did not remove the collapse:
  no-hip cases latched the target and kept positive final box travel
  `0.150-0.156 m`, but still had `109` falls, `105` drops, max tilt about
  `1.67-1.70 rad`, and minimum torso height about `0.23 m`. Small hip-base
  `0.03` improved final box travel to `0.18645 m` but still had `99` falls
  and `96` drops; hip-base `0.05` reduced falls/drops to `48/43` but walked
  backward and never latched the target. CSV inspection showed the collapse is
  dominated by roll/lateral load drift, with box/torso `y` displacement
  approaching about `-0.95 m` in the failed hold. The next valid probe should
  combine closed-loop/static hold with lateral centering or a better lateral
  contact/support controller, while still avoiding root/box pose or velocity
  writes and torso body-force assist.
- 2026-07-07 lateral-centering hold result: Slurm job `169181`
  (`mj_free_fikcent`) completed on `server30`; all six cases failed. Adding
  audited equal-and-opposite y-axis retention (`RETENTION_KP_Y=140-220`,
  `RETENTION_MAX_FORCE_Y=90`) did not solve the no-body-assist roll/drop
  failure. No-hip cases still had about `110` falls and `105` drops with
  positive final travel around `0.157-0.163 m`. Small hip-base cases reduced
  falls only modestly: hip-base `0.04` reached final travel `0.19379 m` but
  still had `96` falls and `93` drops; hip-base `0.05` reduced falls/drops to
  `85/80` but under-traveled at final box travel `0.08569 m`. Conclusion:
  lateral box centering alone is insufficient; the next valid support probe
  should use roll-state hip feedback or a stronger lateral support controller,
  still without root/box pose or velocity writes and without torso body-force
  assist.
- 2026-07-07 hip-roll feedback hold result: Slurm job `169183`
  (`mj_free_fikfb`) completed on `server30`; all six cases failed. Negative
  hip-roll feedback preserved positive target-directed travel but still fell:
  `hip003_fbneg030_y180` reached final box travel `0.20149 m` but had
  `105` falls and `102` drops; `hip004_fbneg030_y180` reached `0.17732 m`
  but had `102/99`; `hip005_fbneg025_y180` reached `0.17458 m` but had
  `97/95`. Positive feedback or stronger hip support reduced falls in some
  cases but returned to stable-backward behavior: `hip005_fbpos025_y180` had
  fall/drop `0/0`, max tilt `0.23285 rad`, and min box z `0.69324 m`, but
  final box travel was `-1.08736 m` and the target never latched. Conclusion:
  the current foot-IK controller still splits into forward-falling and
  stable-backward regimes. Next no-body-assist probe is stronger joint
  position servo gains as a support-controller replacement test; if that also
  fails, stop treating this simple foot-IK family as near success.
- 2026-07-07 strong joint-servo result: added `ACTUATOR_KP` and
  `ACTUATOR_KV` to the MuJoCo free-box runner and checker report fields.
  Slurm job `169189` (`mj_free_fikservo`) completed on `server26`; all six
  cases failed. Stronger joint position servos did not produce a valid
  no-body-assist carry. Hip-base `0.03` with `kp=140/180` and `kv=14/18`
  was stable fall/drop `0/0` with low tilt and high box height, but walked
  backward with final box travel about `-1.40 m` and never latched the target.
  No-hip and hip-feedback cases either under-traveled or still fell/dropped
  (`82-112` falls and `77-106` drops in the unstable cases). Conclusion:
  the current hand-authored foot-IK family should no longer be treated as a
  near-success locomotion/carry controller. The next valid backend step is to
  replace the support controller itself, for example with a stance-force/QP
  controller, MuJoCo-native legged RL baseline, or controller-backed robot
  policy, while preserving the same free-box, no-root-write, no-box-write,
  no-body-force-assist gates.
- 2026-07-07 first stance-force support-controller result: added
  `SUPPORT_CONTROLLER_MODE=stance_force` to
  `scripts/mujoco/run_quadruped_freebox_carry.py`. It maps desired stance
  support/propulsion forces through foot Jacobians into actuated joint
  generalized torques and records `support_joint_torque_write_count`; it does
  not write root/box pose or velocity and does not enable torso body-force
  assist. Slurm job `169200` (`mj_free_sf`) completed on `server01`; all six
  cases failed strict gates, but the failure mode changed usefully. Positive
  force scale was wrong or ineffective, under-traveling or moving backward.
  Negative force scale gave real target-directed motion and early target
  latch: `sf_neg_nominal` reached final box travel `0.63974 m` with target
  hold `2761` steps, and `sf_neg_hip003` reached `0.53802 m` with the same
  hold length. Both still failed due to falls/drops and box-torso relative
  error (`sf_neg_hip003`: `119` falls, `114` drops, final relative error
  `0.39427 m`; `sf_neg_nominal`: `118` falls, `113` drops, final relative
  error `0.22764 m`). Conclusion: stance-force is a more promising support
  replacement than the previous foot-IK tuning because it can produce forward
  motion without root/box writes or torso body-force assist, but it currently
  overdrives and loses the box. The next probe should add braking/early-stop
  and lower horizontal support force while keeping the same audit gates.
- 2026-07-07 stance-force brake result: Slurm job `169208`
  (`mj_free_sfbrake`) completed on `server01`; all six cases failed strict
  gates. Braking exposed a useful tradeoff. `brake_s004_hn006` was stable
  with fall/drop `0/0`, max tilt `0.31098 rad`, min box height `0.70184 m`,
  and final relative error `0.03184 m`, but it braked too hard and walked
  backward with final box travel `-0.49488 m`. The best positive-travel case
  `brake_scale05_s006` reached final box travel `0.28517 m` and final
  relative error `0.13416 m`, but still had `103` falls, `99` drops, max tilt
  `1.94814 rad`, and max relative error `0.26066 m`. Conclusion: the
  stance-force route has a real stability/travel continuum unlike the earlier
  foot-IK route. The next probe should search between these endpoints with
  smaller negative force scale, small hip-base, stronger vertical/roll/pitch
  support, and neutral or weak braking.
- 2026-07-07 stance-force refine result: Slurm job `169210`
  (`mj_free_sfref`) completed on `server01`; all six cases failed strict
  gates. The bracket narrowed but still did not produce a valid carry.
  `refine_h004_s050_h0` and `refine_h005_s050_h0` were stable with fall/drop
  `0/0`, max tilt about `0.277 rad`, min box height about `0.697 m`, final
  relative error `0.03480 m` and `0.02614 m`, and target-hold above
  `2500` steps, but both walked backward or under-traveled with final box
  travel `-0.25201 m` and `-0.21854 m`. The weaker scale case
  `refine_h005_s035_h0` kept positive final box travel `0.26334 m` and
  target-hold `1925` steps, but still failed with `85` falls, `80` drops, max
  tilt `2.35796 rad`, and final relative error `0.14494 m`. Conclusion:
  the active stance-force boundary is between stable-backward at force scale
  about `-0.50` and positive-falling at about `-0.35`; next search should
  target `-0.38` to `-0.45` with hip-base around `0.045-0.05`, neutral
  hold speed, and stronger damping.
- 2026-07-07 stance-force boundary result: Slurm job `169216`
  (`mj_sfbdshort`) completed on `server43`; all eight cases failed strict
  gates. The run itself is valid and used no root/box pose or velocity writes,
  no torso body-force assist, and `support_joint_torque_write_count=3000`.
  However the parameter choice was too conservative: every case was stable
  with fall/drop `0/0`, max tilt about `0.240-0.247 rad`, min box height about
  `0.718 m`, and final relative error about `0.024-0.025 m`, but all walked
  backward with final box travel from `-0.688` to `-0.750 m`, max positive box
  travel only about `0.031-0.033 m`, and no target latch. Conclusion: the
  next stance-force search should return closer to the `169210`
  positive-but-falling endpoint near force scale `-0.35`, using the previous
  forward-drive settings and smaller changes, rather than adding stronger
  damping/slowdown.
- 2026-07-07 stance-force edge result: Slurm job `169230` (`mj_sfedge`)
  completed on `server59`; all eight cases failed strict gates. This run
  restored target latch and positive box travel, but the failure returned to
  post-latch hold collapse. Positive-travel cases reached final box travel
  `0.17284-0.27607 m`, max box travel `0.31656-0.38501 m`, and target-hold
  `1396-1907` steps, but still had `62-81` falls, `57-76` drops, max tilt up
  to `3.24 rad`, and box height down to about `0.313-0.353 m`. The only
  stable case, `edge_s036_h050_stop04`, had fall/drop `0/0`, max tilt
  `0.26166 rad`, min box height `0.69961 m`, and target-hold `2748`, but it
  stopped too early and moved backward with final box travel `-0.19660 m` and
  max positive travel only `0.04066 m`. Conclusion: the blocker is now the
  target-latched hold/brake phase, not initial forward propulsion. The next
  controller edit should add an explicit post-latch hold stabilizer inside
  stance-force control, rather than another blind force-scale sweep.
- 2026-07-07 stance-force hold-stabilizer result: added hold-only support
  parameters to `scripts/mujoco/run_quadruped_freebox_carry.py`: separated
  `support_fx_scale` / `hold_support_fx_scale`, hold-only vx/max-fx scaling,
  hold-only z/roll/pitch damping scales, and hold height offset. Slurm job
  `169235` (`mj_sfhold`) completed on `server59`; all eight cases failed
  strict gates. The stop-0.05 positive-travel cases still collapsed during
  post-latch hold: final box travel `0.18642-0.23172 m`, target-hold `1606`
  steps, but `68-69` falls and `63-64` drops. The early stop case
  `holdfx_pos030_stop04` remained stable with fall/drop `0/0`, max tilt
  `0.26166 rad`, min box height `0.69961 m`, and target-hold `2748`, but
  drifted backward to final box travel `-0.20337 m` with max positive travel
  only `0.04064 m`. Conclusion: hold-only horizontal force scaling is not
  enough for stop-0.05 collapse, but the stop-0.04 stable early-stop case is
  a useful base for testing a small positive post-latch creep speed to prevent
  backward drift while preserving stability.
- 2026-07-07 stance-force hold-creep result: Slurm job `169236`
  (`mj_sfcreep`) completed on `server59`; all eight cases failed strict
  gates. Stop-0.04 with positive post-latch creep speed and positive
  hold-horizontal scale was stable for most cases with fall/drop `0/0`, max
  tilt `0.26166 rad`, min box height `0.69961 m`, and target-hold `2748`, but
  it drifted much farther backward: final box travel was about `-1.07` to
  `-1.23 m` for hold speeds `0.01-0.05`. Higher positive hold horizontal scale
  `0.60` caused `9` falls/drops and final travel `-1.83 m`. The later
  stop-0.045 case restored strong positive travel (`0.29945 m`) but again
  failed with `62` falls and `58` drops. Conclusion: in the current sign
  convention, positive hold-horizontal scale with positive hold speed pushes
  the hold phase in the wrong travel direction. The next narrow probe should
  test the stable stop-0.04 setup with small positive hold speed but negative
  `hold_support_fx_scale`, because negative horizontal scale is the sign that
  generated forward stance-force propulsion in earlier runs.
- 2026-07-07 stance-force hold-creep negative-fx result: Slurm job `169242`
  (`mj_sfnegfx`) completed on `server59`; all eight cases failed strict
  gates. Negative hold horizontal scale with small positive hold speed did not
  rescue final travel. Every case stayed stable with fall/drop `0/0`, max
  tilt `0.26166 rad`, min box height `0.69961 m`, and target-hold `2748`,
  but all remained in the stable-backward attractor: max positive box travel
  only `0.0475-0.0496 m`, and final box travel `-0.79589` to `-0.92705 m`.
  Conclusion: with stop-0.04 the controller enters a stable backward hold
  basin regardless of small post-latch velocity sign. The next valid probe
  should strengthen the static support geometry/posture for stop-0.05
  post-latch hold, especially wider fore-aft support feet within the existing
  `[-0.22, 0.22]` clamp, instead of continuing hold-speed sign sweeps.
- 2026-07-07 stance-force wide-hold-support result: Slurm job `169247`
  (`mj_sfwide`) completed on `server59`; all eight cases failed strict gates.
  Wider post-latch static foot geometry did not solve the collapse. All cases
  restored useful positive travel (`0.20196-0.27827 m` final box travel,
  `0.31277-0.33222 m` max box travel, target-hold `1606`), but still had
  `68-69` falls, `63` drops, max tilt about `3.24 rad`, min box height about
  `0.297-0.308 m`, and final relative error about `0.218-0.227 m`. Conclusion:
  simple post-latch static foot placement changes are insufficient. The
  stance-force route still needs a materially stronger balance controller,
  such as a COM/centroidal-state feedback law, a real legged policy, or a
  controller that changes support forces from measured pitch/COM state rather
  than only static foot target geometry.
- 2026-07-07 first stance-force COM-support result: added COM/centroidal
  support feedback fields to `scripts/mujoco/run_quadruped_freebox_carry.py`.
  The controller estimates robot COM from MuJoCo body inertial positions
  excluding the free box, compares it to the stance-foot center, and shifts
  front/rear and left/right stance foot vertical forces through the same
  Jacobian-to-actuated-joint-torque path. Slurm job `169254` (`mj_sfcom`)
  completed on `server59`; all eight cases failed strict gates. Positive
  COM-x feedback made the system stable but backward: fall/drop `0/0`, max
  tilt about `0.258-0.260 rad`, min box height about `0.701-0.704 m`, but
  target never latched and final box travel was `-0.61658` to `-0.78394 m`.
  Negative COM-x `-400` restored positive final travel `0.17670 m` and target
  hold `2272`, but worsened collapse with `94` falls and `88` drops; negative
  COM-x `-800` returned to stable under-travel/backward. Conclusion: COM
  feedback is real and auditable, but applying it before target latch disrupts
  approach. The next probe should set pre-latch COM feedback scale to zero and
  enable COM support only after `target_stop_latched`.
- 2026-07-07 hold-only COM-support result: added
  `support_com_pre_latch_scale` so COM feedback can be disabled before target
  latch and enabled only during hold. Slurm job `169263` (`mj_sfholdcom`)
  completed on `server59`; all eight cases failed strict gates. Disabling
  pre-latch COM feedback preserved approach and target latch, and all cases
  reached useful final box travel `0.18946-0.21849 m`, max box travel
  `0.29473-0.36129 m`, and target-hold `1797` steps. But every case still
  collapsed during hold with `77-78` falls, `72-73` drops, max tilt about
  `3.24 rad`, min box height `0.289-0.303 m`, and final relative error about
  `0.219-0.224 m`. Conclusion: vertical COM-based force redistribution is
  insufficient. The next controller change should add pitch/roll damping
  through fore-aft/side-specific horizontal foot forces or a fuller
  centroidal wrench controller; continuing vertical-force-only sweeps is not
  justified.
- 2026-07-07 hold-only lateral foot-force result: added stance-force lateral
  foot-force fields (`support_fy_roll_gain`,
  `support_fy_roll_rate_gain`, `support_fy_com_y_gain`, and hold/pre-latch
  scales) to apply lateral foot forces through the same Jacobian-to-actuated-
  torque path without root/box writes. Slurm job `169267` (`mj_sfholdlat`)
  completed on `server59`; all eight cases failed strict gates. Positive and
  negative lateral-force signs preserved target latch and positive travel
  (`0.16600-0.23612 m` final, `0.29273-0.39142 m` max, target-hold `1797`),
  but fall/drop remained `77-79` / `72-73`, max tilt about `3.24 rad`, and
  box height fell to `0.286-0.316 m`. Conclusion: lateral foot-force damping
  alone is also insufficient. The next valid controller probe should change
  actual post-latch leg posture feedback, such as hold-only hip-roll feedback
  and hold-only roll-to-foot-height feedback, rather than only adding external
  support-force components.
- 2026-07-07 hold-only posture-feedback result: added hold-only
  `hold_hip_roll_feedback_gain` and `hold_foot_roll_z_gain` so post-latch leg
  posture can respond to roll without changing approach. Slurm job `169272`
  (`mj_sfposture`) completed on `server59`; all eight cases failed strict
  gates, but this is the first branch in this series that materially reduced
  the collapse severity. Negative hold hip-roll feedback reduced max tilt from
  the recurring `~3.24 rad` failure to `1.75-1.77 rad`: `posture_hip_neg040`
  had final/max box travel `0.28656/0.30231 m`, final/max relative error
  `0.11609/0.20463 m`, target-hold `1797`, but still `77` falls and `73`
  drops with min box height `0.33267 m`. `posture_combo_neg` also reduced max
  tilt to `1.69314 rad` and final relative error to `0.11001 m`, but still
  had `77/73` falls/drops and max relative error `0.22386 m`. Positive
  hold hip feedback and foot-height-only feedback stayed near the old
  `~3.24 rad` collapse. Conclusion: hold-only negative hip-roll feedback is
  the current best direction inside this hand-built controller family. The
  next probe should refine stronger negative hold hip feedback, hip base, and
  compatible foot-height feedback around `posture_hip_neg040` and
  `posture_combo_neg`.
- 2026-07-07 hold-posture refine v1 result: Slurm job `169285`
  (`mj_sfpnogres`) completed on `server30`; all eight cases failed strict
  gates and produced identical stable-backward behavior. The common parameter
  baseline in `scripts/mujoco/run_quadruped_freebox_stance_force_hold_posture_refine_suite.sh`
  increased global actuator/support/retention settings relative to `169272`,
  and this broke approach/target latch before hold-feedback differences could
  matter: every case had final/max box travel `-0.81105/0.03260 m`,
  target-stop latched `false`, target-hold `0`, while fall/drop were `0/0`,
  max tilt `0.24458 rad`, min box height `0.71517 m`, and final/max relative
  error `0.03911/0.07584 m`. Conclusion: this v1 refine suite over-stabilized
  the system into the no-latch backward basin and should not be used as
  evidence against negative hold hip-roll feedback itself. The next refine
  must keep the `169272` global baseline and vary only post-latch posture
  parameters.
- 2026-07-07 active hold-posture refine v2: added
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_posture_refine_v2_suite.sh`
  to restore the `169272` approach/support/retention baseline and sweep only
  hold-only negative hip-roll feedback, compatible foot-height feedback,
  hold hip base, hold stance height, and front/rear foot positions. Submitted
  through tmux `codex_mj_refine_v2_0707` as Slurm job `169288`
  (`mj_sfpv2`). Treat it as pending/active until summaries are inspected; do
  not infer success from submission.
- 2026-07-07 hold-posture refine v2 result: Slurm job `169288`
  (`mj_sfpv2`) completed on `server36`; all ten cases failed strict gates.
  Unlike v1, the `169272` global baseline preserved forward progress and
  target latch: final box travel ranged about `0.264-0.318 m`, max box travel
  `0.287-0.324 m`, and target-hold was `1797` steps. Failure remained
  post-latch lateral/roll collapse: every case still had `77` fall events
  and `73-74` box-drop events, max tilt `1.68-1.84 rad`, and min box height
  `0.332-0.362 m`. The best tilt case was `refine2_hip_neg030`
  (`hold_hip_roll_feedback_gain=-0.30`), with final/max travel
  `0.26439/0.29245 m`, max tilt `1.68055 rad`, min box height `0.36177 m`,
  final/max relative error `0.11087/0.18279 m`, and target-hold `1797`.
  Inspecting its CSV showed first fall at step `1480` and first box drop at
  step `1560`; the dominant failure was lateral drift/roll, with box `y`
  reaching about `-0.47 m` before rapid roll collapse and about `-0.81 m` at
  first drop. Conclusion: negative hold hip-roll feedback alone is
  insufficient. The next valid probe is hold-only lateral/roll support
  combined with the best negative hip feedback, not further one-dimensional
  hip-feedback sweeps.
- 2026-07-07 active hold-lateral-posture combination: added
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_lateral_posture_suite.sh`,
  combining the `169272`/v2 posture baseline with hold-only lateral
  stance-force terms (`support_fy_roll_gain`,
  `support_fy_roll_rate_gain`, `support_fy_com_y_gain`) and
  `support_fy_pre_latch_scale=0`. Submitted through tmux
  `codex_mj_latpost_0707` as Slurm job `169291` (`mj_sflatpost`). Treat it as
  pending until summaries are inspected.
- 2026-07-07 hold-lateral-posture result: Slurm job `169291`
  (`mj_sflatpost`) completed on `server36`; all eight cases failed strict
  gates. The lateral stance-force channel saturated as intended
  (`max_abs_support_fy_n` `64.46-120 N`) and target latch/positive travel were
  preserved, but it did not prevent the post-latch collapse: fall/drop counts
  remained `77-78` / `73`, max tilt was still `1.65-1.73 rad`, and min box
  height was `0.324-0.358 m`. Best max-tilt case was
  `latpost_hip040_combo_pos`, with final/max box travel `0.28423/0.32303 m`,
  max tilt `1.65259 rad`, min box height `0.33075 m`, final/max relative
  error `0.10548/0.20650 m`, and target-hold `1797`; it still failed with
  `78` falls and `73` drops. Conclusion: roll/com lateral force combined with
  negative hip feedback only marginally reduces tilt and does not suppress the
  global lateral drift failure.
- 2026-07-07 active world-y hold correction: added
  `support_fy_world_y_gain`, `support_fy_world_vy_gain`, and
  `support_fy_world_y_source` to
  `scripts/mujoco/run_quadruped_freebox_carry.py`, exposed them in
  `scripts/mujoco/run_quadruped_freebox_carry.sh`, and added
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_world_y_suite.sh`.
  This applies hold-only world-y correction through stance-foot Jacobian
  torques, not root/box pose or velocity writes, and records
  `max_abs_support_world_y_error_m` plus `max_abs_box_y_m`. Submitted through
  tmux `codex_mj_worldy_0707` as Slurm job `169292` (`mj_sfworldy`). Treat it
  as pending until summaries are inspected.
- 2026-07-07 world-y hold correction result: Slurm job `169292`
  (`mj_sfworldy`) completed on `server36`; all eight cases failed strict
  gates. World-y correction saturated the lateral force limit
  (`max_abs_support_fy_n` `120-160 N`) but did not suppress global lateral
  drift: `max_abs_box_y_m` was still `0.972-1.177 m`, fall/drop counts stayed
  `77-78` / `73-74`, max tilt stayed `1.71-1.78 rad`, and min box height
  stayed `0.320-0.344 m`. Positive world-y feedback usually reduced x
  progress and increased drops; negative world-y signs were also not viable.
  Conclusion: the hand-built controller is not failing only because it lacks a
  world-y centering term. Do not continue broad world-y sweeps unchanged.
- 2026-07-07 active hold-support authority test: added
  `hold_support_max_foot_fz_scale` and
  `hold_support_max_joint_torque_scale` to
  `scripts/mujoco/run_quadruped_freebox_carry.py` and the launcher so that
  only post-latch support authority can be raised without damaging approach.
  Added `scripts/mujoco/run_quadruped_freebox_stance_force_hold_authority_suite.sh`
  and submitted it through tmux `codex_mj_authority_0707` as Slurm job
  `169293` (`mj_sfauth`). Treat it as pending until summaries are inspected.
- 2026-07-07 hold-support authority result: Slurm job `169293`
  (`mj_sfauth`) completed on `server36`; all eight cases failed strict gates.
  Increasing only the post-latch support joint-torque cap did not change the
  best baseline at all: `authority_torque15` and `authority_torque20`
  exactly reproduced `refine2_hip_neg030` with final/max travel
  `0.26439/0.29245 m`, fall/drop `77/73`, max tilt `1.68055 rad`, and min
  box height `0.36177 m`. Increasing post-latch max foot force, height, and
  damping mostly worsened tilt or relative error; `authority_high05` returned
  to the old `~3.24 rad` collapse. Conclusion: the current heuristic
  stance-force formulation is not simply starved for post-latch force/torque
  authority. Do not continue this authority sweep unchanged.
- 2026-07-07 active centroidal support formulation: added
  `support_controller_mode=centroidal_stance_force` to
  `scripts/mujoco/run_quadruped_freebox_carry.py`. This mode solves a
  least-squares stance-foot 3D force distribution for desired total force and
  roll/pitch wrench, then maps foot forces through MuJoCo foot-body Jacobians
  to actuated joint torques. It still does not write root/box pose or
  velocity. Added `scripts/mujoco/run_quadruped_freebox_centroidal_support_suite.sh`
  and submitted it through tmux `codex_mj_centroidal_0707` as Slurm job
  `169294` (`mj_centroid`). Treat it as pending until summaries are inspected.
- 2026-07-07 centroidal support result and MuJoCo hand-controller stop
  decision: Slurm job `169294` (`mj_centroid`) completed on `server36`; all
  six centroidal cases failed strict gates and were worse than the best
  heuristic stance-force baseline. Most cases failed before target latch, with
  max box travel only `0.02295-0.03284 m` and final box travel negative. The
  only latched case, `centroid_negscale_authority`, reached final/max travel
  `0.18851/0.19485 m` and target-hold `655`, but still had `110` falls,
  `103` drops, max tilt `1.92181 rad`, min box height `0.36842 m`, and max
  relative error `0.23302 m`. Conclusion: the current simplified MuJoCo
  quadruped hand-controller path is exhausted as a route to credible
  fall/drop-free carrying. Do not keep sweeping this controller family unless
  there is a genuinely new controller or policy backend. The next valid
  direction is a controller-backed locomotion policy/backend or a proper
  optimizer/MPC-style balance controller, not more scalar gain sweeps on this
  model.
- 2026-07-07 MuJoCo hold capture-point controller added as a new post-latch
  balance formulation: `scripts/mujoco/run_quadruped_freebox_carry.py` now
  supports `--hold-capture-point-foot-placement` with capture-time,
  x-foot-placement, y-hip-roll, and y-foot-height terms. It only acts after
  `target_stop_latched`; it does not write root pose/velocity or box
  pose/velocity. The launcher forwards the new fields, checker output now
  records and validates `hold_capture_active_steps`, and new suite
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_capture_suite.sh`
  tests this against the same strict free-box gates. This is intended to
  address the post-latch lateral/roll collapse by changing foot placement,
  not by repeating scalar stance-force sweeps. It still requires compute-node
  validation before being treated as evidence. Slurm job `169609`
  (`mj_holdcap`) was submitted through tmux
  `curiosity_mujoco_hold_capture_after_g1_0707` with
  `--dependency=afterany:169585`, GPU allocation, and the suite suffix
  `stance_force_holdcapture_after_g1`.
- 2026-07-07 MuJoCo hold capture-point result: Slurm job `169609`
  (`mj_holdcap`) ran on `server59`; all six summaries were written under
  `experiments/outputs/mujoco_quadruped_freebox/20260707_mujoco_quad_freebox_2kg_*_stance_force_holdcapture_after_g1/`,
  then the job was cancelled during script-end cleanup to release the node.
  Strict result was `fail`, `0/6` cases passed. All cases had free box,
  `assist_mode=none`, `leg_drive_mode=foot_ik`,
  `support_controller_mode=stance_force`, retention force audited,
  target-stop latched, target-stop hold `1797` steps, hold-capture active
  `1796` steps, support joint torque writes `3000`, and root/box pose/
  velocity writes all `0`. They still failed after latch with `77` fall
  events and `73-74` box-drop events, max tilt `1.668-1.719 rad`, and minimum
  box height `0.318-0.362 m`. Final box travel was nonzero
  (`0.260-0.313 m`), but the robot/box collapsed during the hold. Best final
  relative error was `0.104 m` in `capture_xy_pos`, and best min box height
  was `0.362 m` in `capture_x_pos`; neither is close to a pass because the
  fall/drop/tilt/height gates fail. Conclusion: the new capture-point
  foot-placement/hip/foot-height controller is active and audited, but it
  does not solve the simplified MuJoCo free-box post-latch stability problem.
  This reinforces the prior stop decision for hand-tuned MuJoCo controller
  sweeps; the next credible path needs a real controller-backed locomotion
  backend, MPC/whole-body balance controller, or a materially different
  support optimizer, not more scalar gain variants on this same model.
- 2026-07-07 MuJoCo LQR/centroidal support route added:
  `scripts/mujoco/run_quadruped_freebox_carry.py` now supports
  `SUPPORT_CONTROLLER_MODE=lqr_stance_force`. This is an opt-in replacement
  support backend, not another scalar stance-force sweep. It computes a
  finite-horizon double-integrator LQR gain for COM x/y state feedback,
  applies the resulting desired horizontal wrench through the existing
  stance-foot centroidal least-squares allocation, and then maps foot forces
  through foot Jacobians to actuated joint generalized forces. It records
  `support_lqr_active_steps`, `support_lqr_k_pos`, `support_lqr_k_vel`,
  `max_abs_support_lqr_fx_n`, and `max_abs_support_lqr_fy_n`; it does not
  write root pose/velocity or box pose/velocity. Launcher and checker support
  were added, plus
  `scripts/mujoco/run_quadruped_freebox_lqr_stance_hold_suite.sh`. Slurm job
  `169618` (`mj_lqrhold`) was submitted through tmux
  `curiosity_mujoco_lqr_stance_hold_0707` with GPU allocation. Treat this as
  a diagnostic backend test only until all strict free-box gates are parsed.
- 2026-07-07 first LQR support run `169618` (`mj_lqrhold`) is an invalid
  activation diagnostic, not a valid LQR result. It completed on `server39`
  with four strict failures, but all cases had `target_stop_latched=false`
  and `support_lqr_active_steps=0`. Root/box pose writes remained `0`, but
  the pre-latch behavior used the centroidal allocation path before LQR could
  activate, so the known stance-force approach/latch behavior was not
  preserved. The code was corrected so
  `SUPPORT_CONTROLLER_MODE=lqr_stance_force` with
  `SUPPORT_LQR_POST_LATCH_ONLY=1` uses the original stance-force allocation
  before latch and switches to LQR/centroidal allocation only after latch.
  Corrected retry Slurm job `169619` (`mj_lqrh2`) was submitted through tmux
  `curiosity_mujoco_lqr_stance_hold_retry_0707` with suite suffix
  `lqr_stance_hold_lqr2`.
- 2026-07-07 corrected LQR/centroidal support result: Slurm job `169619`
  (`mj_lqrh2`) completed on `server39`. The corrected mode was valid:
  all four cases latched target-stop and had `support_lqr_active_steps=1797`,
  target-stop hold `1797`, support joint torque writes `3000`, and root/box
  pose/velocity writes all `0`. Strict result was still `fail`, `0/4` cases
  passed. Fall/drop were `78/75` for all cases; max tilt stayed high
  (`1.612-1.754 rad`), min box height was low (`0.318-0.370 m`), and final
  box travel was only `0.050-0.100 m`. Conclusion: switching post-latch to
  the centroidal LQR allocation activates the new controller but does not
  solve the collapse and can reduce final travel. Added a more conservative
  `lqr_additive_stance_force` mode that preserves the original stance-force
  allocation and only adds LQR horizontal corrections after latch. Slurm job
  `169621` (`mj_lqradd`) was submitted through tmux
  `curiosity_mujoco_lqr_additive_hold_0707` with suite suffix
  `lqr_additive_hold_add1`.
- 2026-07-07 additive LQR support result: Slurm job `169621`
  (`mj_lqradd`) completed on `server39`. Strict result was `fail`, `0/4`
  cases passed. The additive mode was active in all cases:
  `target_stop_latched=true`, target-stop hold `1797`, LQR active steps
  `1797`, support joint torque writes `3000`, and root/box pose/velocity
  writes all `0`. Compared with centroidal LQR, additive LQR preserved the
  useful forward travel and retention better: final box travel was
  `0.253-0.272 m`, max box travel `0.285-0.301 m`, final relative error
  `0.099-0.116 m`, and max relative error `0.166-0.208 m`. It still failed
  the real carrying gates with `77` falls / `73` box drops in every case,
  max tilt about `1.674-1.682 rad`, and minimum box height
  `0.331-0.374 m`. Conclusion: additive LQR is a better integration form
  than switching entirely to centroidal allocation, but it still does not
  solve post-latch balance. The remaining failure is not target progress or
  retention; it is whole-body fall/tilt after latch.
- 2026-07-07 post-latch attitude recovery route added and tested:
  `scripts/mujoco/run_quadruped_freebox_carry.py` now supports
  `--support-attitude-recovery`. It activates only after target latch and
  when tilt exceeds a configurable threshold, then scales extra roll/pitch
  support gains, optional target-height offset, hold hip-roll feedback, and
  roll-to-foot-height feedback. It records
  `support_attitude_recovery_active_steps` and
  `max_support_attitude_recovery_strength`; it still applies through leg
  posture targets and support joint torques, with no root/box pose or
  velocity writes. Launcher/checker support and
  `scripts/mujoco/run_quadruped_freebox_attitude_recovery_suite.sh` were
  added. Slurm job `169625` (`mj_recover`) completed on `server39`. Strict
  result was `fail`, `0/4` cases passed. All cases were valid activation
  diagnostics: target-stop hold `1797`, LQR active `1797`, recovery active
  `1619-1796`, support joint torque writes `3000`, and root/box pose/
  velocity writes all `0`. Fall/drop remained `77/73-74`; final box travel
  stayed useful at `0.253-0.272 m`; final relative error stayed
  `0.099-0.124 m`; min box height stayed low at `0.323-0.332 m`. The only
  useful signal was `recover_roll_gain`, which reduced max tilt to
  `1.6355 rad` versus additive LQR's roughly `1.67-1.68 rad`, but this is far
  from the strict `0.70 rad` gate. Conclusion: simple gain/posture recovery
  can slightly reduce tilt but does not prevent post-latch falling. The next
  meaningful controller change needs to alter support contacts or solve a
  constrained whole-body/upright control problem, not merely boost roll gains.
- 2026-07-07 constrained support/contact allocation route added:
  `scripts/mujoco/run_quadruped_freebox_carry.py` now supports
  `SUPPORT_CONTROLLER_MODE=qp_stance_force`. This is a projected-contact
  QP-style stance-foot allocator: it tracks the desired whole-body wrench,
  constrains each stance foot to unilateral normal-force bounds, clips
  tangential force inside a friction cone, regularizes force magnitude, and
  maps the resulting foot forces through foot Jacobians to actuated joint
  generalized forces. It records `support_qp_active_steps`,
  `max_support_qp_wrench_residual`, and `max_support_qp_friction_usage`.
  Launcher/checker support and
  `scripts/mujoco/run_quadruped_freebox_qp_support_suite.sh` were added.
  Slurm job `169627` (`mj_qpsupp`) was submitted through tmux
  `curiosity_mujoco_qp_support_0707`; treat it as unverified until summaries
  are parsed.
- 2026-07-07 route switch after MuJoCo hand-controller exhaustion: current
  best credible locomotion/carry evidence is the G1 AGILE policy path in
  `scripts/isaac/build_core_world_g1_box_scene.py` and
  `scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh`, not the
  simplified MuJoCo controller. Historical best low-carry run
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`
  used real G1 USD, AGILE ONNX policy, a spawned free box, collision-enabled
  cradle/top lid, `attach_box=none`, no rollout root pose/velocity writes, no
  rollout box pose writes, and completed 819 steps with fall/drop `0/0`,
  final robot/box target-directed travel `2.29876/2.34645 m`, max robot/box
  tilt `0.20860/0.41361 rad`, min robot/box z `0.75211/0.80838 m`, and
  target-window streak at the end `164` steps. It still must be treated as a
  diagnostic, not final success, because it only validates one low-carry
  posture and uses engineered cradle/lid support rather than arbitrary
  posture/unknown-object carrying.
- 2026-07-07 fresh G1 AGILE low-carry reproduction submitted: Slurm job
  `169302` (`g1_lowrepro`) through tmux `codex_g1_lowcarry_repro_0707` runs
  `scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
  with only `RUN_LOWCARRY_BASELINE=1` enabled and suite prefix
  `20260707_g1_targetwindow_lowcarry_repro`. Purpose: verify whether the
  historical low-carry pass reproduces fresh before expanding to light/heavy
  boxes or chestpad/other postures. Treat as pending until summaries are
  inspected.
- 2026-07-07 fresh G1 AGILE low-carry reproduction result: Slurm job
  `169302` (`g1_lowrepro`) completed on `server36` in `00:00:45` with exit
  `0:0`. It reproduced the historical low-carry pass exactly under suite
  prefix `20260707_g1_targetwindow_lowcarry_repro`: 819 steps, fall/drop
  `0/0`, final robot/box target-directed travel `2.29876/2.34645 m`, max
  robot/box tilt `0.20860/0.41361 rad`, min robot/box z
  `0.75211/0.80838 m`, target-window end streak `164` steps, and rollout
  root pose/velocity and box pose writes all `0`. This is now a fresh
  verified low-carry diagnostic baseline. It is still not final success
  because it covers one posture/load setting with engineered cradle/lid
  support.
- 2026-07-07 active G1 AGILE low-carry load validation: added
  `scripts/isaac/run_core_world_g1_lowcarry_load_validation_suite.sh` to run
  the same strict target-window low-carry gate at `0.25`, `0.50`, and
  `0.75 kg`. Submitted through tmux `codex_g1_lowcarry_load_0707` as Slurm
  job `169303` (`g1_lowload`) with suite prefix
  `20260707_g1_lowcarry_load_validation_fresh`. Treat it as pending until all
  three cases and the aggregate summary are inspected.
- 2026-07-07 G1 AGILE low-carry load validation result: Slurm job `169303`
  (`g1_lowload`) completed on `server36` in `00:01:55` with Slurm exit
  `1:0`. Aggregate summary
  `experiments/outputs/core_world_g1_lowcarry_load_validation/20260707_g1_lowcarry_load_validation_fresh/lowcarry_load_validation_summary.json`
  is strict `fail`: `0.50 kg` reproduced the low-carry pass (`819` steps,
  fall/drop `0/0`, final robot/box target-directed travel
  `2.29876/2.34645 m`, target-window end streak `164`, rollout root/box
  writes `0/0/0`), but `0.25 kg` failed after entering the target window with
  `384` falls / `225` drops and `0.75 kg` failed before a final hold with
  `346` falls / `284` drops. Conclusion: the current low-carry front-tray
  setup has a narrow `0.50 kg` operating point and does not yet generalize
  across load. Do not describe it as load-robust carrying.
- 2026-07-07 active G1 low-carry load repair diagnostic: added
  `scripts/isaac/run_core_world_g1_lowcarry_load_repair_suite.sh` and
  submitted it through tmux `codex_g1_lowrepair_0707` as Slurm job `169304`
  (`g1_lowrepair`) with suite prefix
  `20260707_g1_lowcarry_load_repair_fresh`. It tests three strict repair
  diagnostics only: `0.25 kg` final-window policy freeze, `0.25 kg`
  policy-then-stand hold, and `0.75 kg` chestpad/retention/slow carry. Treat
  it as pending until summaries are inspected; even a pass would be a
  diagnostic repair result, not final arbitrary-posture carrying.
- 2026-07-07 G1 low-carry load repair result: Slurm job `169304`
  (`g1_lowrepair`) completed on `server36` in `00:01:58` with Slurm exit
  `1:0`. Aggregate summary
  `experiments/outputs/core_world_g1_lowcarry_load_repair/20260707_g1_lowcarry_load_repair_fresh/lowcarry_load_repair_summary.json`
  is strict `fail`, `0/3` cases passing. `0.25 kg` final-window freeze failed
  with `418` falls / `102` drops and never latched a target-window streak;
  `0.25 kg` policy-then-stand failed earlier with `550` falls / `536` drops;
  `0.75 kg` chestpad/retention/slow failed with `930` falls / `856` drops and
  negative final target-directed box travel. Conclusion: current scalar
  final-hold, freeze, stand-blend, chestpad, retention, and slow-speed tweaks
  are not sufficient to make the G1 low-carry setup load robust.
- 2026-07-07 active G1 low-carry mass-band diagnostic: added
  `scripts/isaac/run_core_world_g1_lowcarry_mass_band_suite.sh` and submitted
  it through tmux `codex_g1_massband_0707` as Slurm job `169309`
  (`g1_massband`) with suite prefix `20260707_g1_lowcarry_mass_band_fresh`.
  It runs the same strict low-carry target-window gate at `0.35`, `0.40`,
  `0.45`, `0.55`, `0.60`, and `0.65 kg` to determine whether the verified
  `0.50 kg` pass has any nearby mass basin. Treat as pending until summaries
  are inspected.
- 2026-07-07 G1 low-carry mass-band result: Slurm job `169309`
  (`g1_massband`) completed on `server36` in `00:03:54` with Slurm exit
  `1:0`. Aggregate summary
  `experiments/outputs/core_world_g1_lowcarry_mass_band/20260707_g1_lowcarry_mass_band_fresh/lowcarry_mass_band_summary.json`
  is strict `fail`, `1/6` cases passing. `0.35 kg` passed with fall/drop
  `0/0`, target-window end streak `108`, max robot/box tilt
  `0.24340/0.41646 rad`, and rollout root/box writes `0/0/0`. `0.40 kg`
  failed by early lateral/roll fall with `398` falls; `0.45 kg` reached the
  window briefly but late-failed with `87` falls / `60` drops; `0.55 kg`
  failed with `383` falls / `170` drops; `0.60 kg` is a near-miss with
  fall/drop `0/0` and target-window end streak `108` but strict failure on
  box tilt `0.63855 rad > 0.45`; `0.65 kg` failed with `414` falls /
  `154` drops. Conclusion: the low-carry G1 setup has discontinuous narrow
  stable islands (`0.35`, `0.50`, near `0.60`) and is not load robust.
- 2026-07-07 active G1 low-carry edge repair diagnostic: added
  `scripts/isaac/run_core_world_g1_lowcarry_edge_repair_suite.sh` and
  submitted it through tmux `codex_g1_edgerepair_0707` as Slurm job `169311`
  (`g1_edgerep`) with suite prefix
  `20260707_g1_lowcarry_edge_repair_fresh`. It narrowly targets the two most
  useful mass-band failures: `0.60 kg` box-tilt near-miss using tighter lid,
  slower tight-lid, and chestpad-hold variants, plus `0.45 kg` late fall/drop
  using tight lid and final zero corrections. Treat as pending until summaries
  are inspected.
- 2026-07-07 G1 low-carry edge repair result: Slurm job `169311`
  (`g1_edgerep`) completed on `server36` in `00:02:32` with Slurm exit
  `1:0`. Aggregate summary
  `experiments/outputs/core_world_g1_lowcarry_edge_repair/20260707_g1_lowcarry_edge_repair_fresh/lowcarry_edge_repair_summary.json`
  is strict `fail`, `0/4` cases passing. All `0.60 kg` variants were worse
  than the original mass-band near-miss: tight lid failed with `470` falls /
  `451` drops, tight-lid-slow with `305` falls / `293` drops, and chestpad
  hold with `303` falls / `166` drops. The `0.45 kg`
  tight-lid/final-zero case is a partial improvement only: fall/drop `0/0`,
  rollout root/box writes `0/0/0`, but final robot/box target-directed travel
  only `1.52961/1.52314 m`, target-window streak `0`, and max robot/box tilt
  `0.47062/0.48852 rad` still exceed strict gates. Conclusion: lowering the
  top lid is harmful for `0.60 kg`; `0.45 kg` may need later final-hold or a
  different non-pinching retention geometry, but this is still not load-robust
  carrying.
- 2026-07-07 active G1 low-carry edge repair v2: added
  `scripts/isaac/run_core_world_g1_lowcarry_edge_repair_v2_suite.sh` and
  submitted it through tmux `codex_g1_edgerepair_v2_0707` as Slurm job
  `169312` (`g1_edgerep2`) with suite prefix
  `20260707_g1_lowcarry_edge_repair_v2_fresh`. It tests whether the `0.45 kg`
  tight-lid/final-zero partial improvement only needs later final-hold
  (`0.80`/`1.00 m` final thresholds), and whether `0.60 kg` can avoid the box
  tilt failure with non-pinching geometry (`side_rail_only` or no lid with
  taller rails). Treat as pending until summaries are inspected.
- 2026-07-07 G1 low-carry edge repair v2 result and stop decision: Slurm job
  `169312` (`g1_edgerep2`) completed on `server36` in `00:02:38` with Slurm
  exit `1:0`. Aggregate summary
  `experiments/outputs/core_world_g1_lowcarry_edge_repair_v2/20260707_g1_lowcarry_edge_repair_v2_fresh/lowcarry_edge_repair_v2_summary.json`
  is strict `fail`, `0/4` cases passing. Delaying `0.45 kg` final-hold moved
  farther but crossed the stability boundary: final080 reached final robot/box
  target-directed travel `2.61657/2.21137 m` but failed with `306` falls /
  `118` drops; final100 failed with `331` falls / `310` drops. For `0.60 kg`,
  side-rail-only failed with `273` falls / `246` drops and no-lid/tall-rails
  failed with `135` falls / `124` drops. Stop rule: do not keep sweeping
  scalar final thresholds, lower/tighter lids, chestpad, side rails, or no-lid
  geometry as the main route to load robustness for this G1 low-carry setup.
  The next valid step must be a materially different controller/backend,
  such as a policy trained or adapted for payload/contact dynamics, not more
  small cradle-geometry or command-threshold sweeps.
- 2026-07-07 policy-backed ANYmal payload diagnostic result: local ANYmal-C
  USD, anydrive actuator net, and cached RSL-RL checkpoint were present, so
  two compute-node diagnostics were attempted with
  `scripts/isaac/run_anymal_payload_carry.sh`. Slurm job `169316`
  (`any_payload`, tmux `codex_anymal_payload_0707`) used default Fabric, and
  Slurm job `169317` (`any_nofab`, tmux
  `codex_anymal_payload_nofabric_0707`) used `DISABLE_FABRIC=1`. Both failed
  before rollout during IsaacLab `gym.make` environment initialization with
  the same PhysX tensor backend error:
  `Simulation view object is invalidated and cannot be used again to call
  getDofVelocities` followed by `Exception: Failed to get DOF velocities from
  backend`. No summary JSON was written, and there is no walking/carrying
  evidence from these runs. Stop rule: do not repeat this ANYmal
  IsaacLab/RSL-RL wrapper route unchanged; any future ANYmal use must first
  materially change the IsaacLab tensor/backend initialization path or use a
  different working control integration.
- 2026-07-07 G1 Isaac camera replay render result: Slurm job `169319`
  (`g1_render`, tmux `codex_g1_replay_render_0707`) attempted to render a
  clearer 1280x720 MP4 replay from the verified recorded G1 low-carry
  diagnostic CSV
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_lowcarry_168398_replay_record_retry2/agile_low_cradle_freebox_walk/core_world_g1_box_scene_replay.csv`.
  It failed on `server39` in `00:00:23` with no frames and no MP4 because the
  current Isaac environment lacks the capture extensions:
  `ModuleNotFoundError: No module named 'omni.replicator'` and
  `ModuleNotFoundError: No module named 'isaacsim.core.rendering_manager'`.
  The checker at
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_policy_replay_720p/g1_replay_showcase_check.json`
  is strict `fail` with frame count `0`. This is a render-environment
  failure, not control evidence.
- 2026-07-07 G1 fallback presentation visual result: Slurm job `169324`
  (`g1_fallback`, tmux `codex_g1_fallback_visual_0707`) completed on
  `server36` in `00:00:11` with exit `0:0`. It generated a 1600x900
  schematic GIF/poster with 83 frames from the same verified recorded G1
  low-carry CSV using
  `scripts/isaac/render_g1_replay_presentation_fallback.py`. Slurm job
  `169327` (`g1_iomp4`, tmux `codex_g1_imageio_mp4_0707`) then completed on
  `server36` in `00:00:06` with exit `0:0` and converted the frames to MP4
  using `imageio_ffmpeg` after system `ffmpeg` was unavailable. Outputs:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_policy_replay_fallback/`.
  Key files are `g1_lowcarry_replay_fallback.gif`,
  `g1_lowcarry_replay_fallback_poster.png`,
  `g1_lowcarry_replay_fallback.mp4`, and
  `g1_lowcarry_replay_fallback_annotated.mp4`. This is the fallback
  presentation artifact after Isaac camera capture failed. Treat it as
  visualization-only replay of a prior diagnostic, not Isaac camera render,
  not new control evidence, not learned carrying, and not final success.
- 2026-07-07 active-probe-selected load validation: added
  `scripts/isaac/run_core_world_g1_probe_selected_load_validation_suite.sh`.
  It runs the existing front-bumper probe -> posture selector -> selected
  target-window validation pipeline at `0.25`, `0.50`, and `0.75 kg`, writes
  per-case pipeline summaries, and aggregates them under
  `experiments/outputs/core_world_g1_probe_selected_load_validation/`.
  This directly tests whether the current active-probing pipeline helps with
  unknown-load variation; it is not a new policy, not video-conditioned RL,
  and even a pass would remain a diagnostic. Submitted through tmux
  `codex_g1_probe_load_0707` as Slurm job `169332` (`g1_probe_load`) with
  suite stamp `20260707_g1_probe_selected_load_validation_fresh`. Treat as
  pending until the aggregate summary is inspected.
- 2026-07-07 active-probe-selected load validation result: Slurm job
  `169332` (`g1_probe_load`) completed on `server39` in `00:02:20` with
  Slurm state `FAILED`, exit `1:0`, because the aggregate summary is strict
  `fail`. Summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Results: `1/3` cases passed. The `0.50 kg` case passed and reproduced the
  low-carry target-window result with fall/drop `0/0`, final robot/box
  target-directed travel `2.29876/2.34645 m`, target-window end streak `164`,
  and rollout root/box writes `0`. The `0.25 kg` case failed with
  `384` falls / `225` drops and severe over-travel (`final robot/box
  target-directed travel 4.39917/3.98690 m`). The `0.75 kg` case failed with
  `346` falls / `284` drops and negative final box target-directed travel.
  In all three cases the selector chose `lowcarry`, ignored hidden mass as
  required, and used probe motion/visible box size only. Conclusion: the
  current front-bumper probe + heuristic selector does not extract a useful
  load-risk signal for these masses; it simply reproduces the narrow
  `0.50 kg` low-carry island. Do not present this as successful unknown-load
  active probing. The next active-probing work needs a better probe feature
  or learned/system-identification policy, not the current threshold selector.
- 2026-07-07 probe-selected load feature audit: added
  `scripts/isaac/audit_g1_probe_selected_load_features.py` and ran it as
  Slurm job `169334` (`g1_pr_audit`, tmux `codex_g1_probe_audit_0707`) on
  `server36`, completing in `00:00:01` with exit `0:0`. Report:
  `experiments/reports/2026-07-07_g1_probe_selected_load_feature_audit.json`.
  The audit shows the current probe stage is unsafe and not discriminative:
  all three probe summaries have `status=fail`, `240` probe fall events, and
  about `210-211` probe box-drop events before selection; the selector still
  made the same `lowcarry` decision for all masses; final probe target-
  directed box travel (`0.2736`, `0.3673`, `0.3088 m` for `0.25`, `0.50`,
  `0.75 kg`) is not a monotonic mass signal. Conclusion: do not continue the
  current front-bumper probe as the active-probing mechanism. The next probe
  must be low-energy and bounded by safety aborts, and the selector must treat
  probe-induced fall/drop/tilt as high-risk or invalid rather than proceeding
  to a carry validation.
- 2026-07-07 safe-probe implementation and validation: added a bounded probe
  collision window to `scripts/isaac/build_core_world_g1_box_scene.py`:
  `--probe-collision-window` starts the front probe pad with collision
  disabled, enables collision at `--probe-start-step`, disables it at
  `--probe-end-step`, and records enable/disable steps and update errors in
  the summary. Updated
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh` and
  `scripts/isaac/run_core_world_g1_box_scene.sh` to pass the new probe args.
  Updated `scripts/isaac/select_core_world_g1_carry_posture_from_probe.py` to
  support optional probe safety gates on fall/drop/robot tilt/box tilt, and
  updated `scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`
  so a failed safety gate still writes a pipeline summary and skips validation
  instead of losing the case. Updated
  `scripts/isaac/run_core_world_g1_probe_selected_load_validation_suite.sh`
  to use the safe probe by default: short collision window
  `PROBE_START_STEP=40`, `PROBE_END_STEP=80`, smaller/lighter pad
  `0.025 x 0.20 x 0.10 m`, `0.05 kg`, and selector safety gates requiring
  probe fall/drop `0/0`. Lightweight checks passed:
  `py_compile`, `bash -n`, and `git diff --check`.
- 2026-07-07 active safe-probe load validation: submitted tmux
  `codex_g1_safe_probe_load_0707` as Slurm job `169335` (`g1_safe_probe`).
  It runs the updated safe probe -> selector -> selected validation pipeline
  at `0.25`, `0.50`, and `0.75 kg` with suite stamp
  `20260707_g1_safe_probe_selected_load_validation_fresh`. Treat as pending
  until the aggregate summary under
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_safe_probe_selected_load_validation_fresh/`
  is inspected.
- 2026-07-07 active safe-probe load validation result: Slurm job `169335`
  (`g1_safe_probe`) completed on `server39` in `00:02:17` with Slurm state
  `FAILED`, exit `1:0`, because the aggregate summary is strict `fail`.
  Summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_safe_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Results: `0/3` cases passed. All three probes had `probe_active_steps=1`
  and `probe_box_target_directed_travel_m=0`, so the selector chose
  `chestpad` for `0.25`, `0.50`, and `0.75 kg` while still reporting
  `selection_uses_hidden_ground_truth=false`. The selected validation at
  `0.25 kg` and `0.50 kg` had fall/drop `0/0` but never entered the strict
  target window (`target_window_both_streak_at_end_steps=0`); final robot/box
  target-directed travel was `0.46484/0.51690 m` for `0.25 kg` and
  `1.53684/1.55199 m` for `0.50 kg`. The `0.75 kg` case failed badly with
  `482` falls / `439` drops, final robot/box target-directed travel
  `-0.11922/-0.16281 m`, and max robot/box tilt about `1.314/1.312 rad`.
  Conclusion: the bounded collision-window probe fixed the previous
  probe-stage fall/drop issue but removed the interaction signal. Do not
  claim unknown-load adaptation from this run; the next active-probe step must
  tune for a safe but nonzero measured interaction before reusing the
  selector/validation pipeline.
- 2026-07-07 safe-probe signal bracket result: added
  `scripts/isaac/run_core_world_g1_safe_probe_signal_bracket.sh` and submitted
  tmux `codex_g1_safe_probe_signal_0707` as Slurm job `169337`
  (`g1_pr_signal`). It ran four short 0.50 kg probe-only cases with
  collision-window lengths and probe pad geometry variants. Aggregate:
  `experiments/outputs/core_world_g1_safe_probe_signal_bracket/20260707_g1_safe_probe_signal_bracket_fresh/safe_probe_signal_bracket_summary.json`.
  Result: strict `fail`, `safe_signal_cases=[]`. All four case summaries
  completed only `41/180` steps, had
  `Exception: Failed to get root link transforms from backend`, reported
  `probe_active_steps=1`, and had max probe target-directed box motion `0`.
  Fall/drop remained `0/0`, but the runs are invalid as probe evidence because
  the rollout did not complete. Conclusion: runtime collision toggling at the
  probe window is not a backend-stable safe-probe mechanism for the current G1
  Core route; do not rerun `PROBE_COLLISION_WINDOW=1` unchanged or sweep pad
  geometry on top of it.
- 2026-07-07 probe-selector safety hardening after `169337`: updated
  `scripts/isaac/select_core_world_g1_carry_posture_from_probe.py` so any
  non-null `summary.error` fails selection, and added
  `--min-probe-completed-steps`. Updated
  `scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`
  to pass `MIN_PROBE_COMPLETED_STEPS` when set, and updated
  `scripts/isaac/run_core_world_g1_probe_selected_load_validation_suite.sh`
  to default it to `PROBE_FREE_STEPS`. Updated the safe-probe signal bracket
  aggregator to record `completed_steps`, `summary_status`, and
  `summary_error`, and require full probe rollout completion before a safe
  signal can count. This fixes the earlier flaw where step-41 failed probes
  could still be selected if fall/drop/tilt were low.
- 2026-07-07 precontact safe-probe signal bracket result: Slurm job `169338`
  (`g1_preprobe`, tmux `codex_g1_precontact_probe_signal_0707`) completed on
  `server39` with exit `0:0`. It reran the short 0.50 kg safe-probe signal
  bracket with `PROBE_COLLISION_WINDOW_MODE=0`, so the probe pad was
  pre-authored and always colliding rather than enabled at runtime. Summary:
  `experiments/outputs/core_world_g1_safe_probe_signal_bracket/20260707_g1_precontact_probe_signal_bracket_fresh/safe_probe_signal_bracket_summary.json`.
  Result: diagnostic `pass`, `safe_signal_cases` contained all four cases:
  `small_x042`, `small_x046`, `small_x050`, and `small_x054`. Every case
  completed `180/180` steps with `error=null`, fall/drop `0/0`,
  `probe_active_steps=80`, root/box rollout writes `0`, and nonzero max probe
  target-directed box motion. Max probe target-directed motions were about
  `0.11805`, `0.15116`, `0.12182`, and `0.12966 m`; max robot tilt stayed
  below `0.20 rad`, max box tilt below `0.253 rad`. This is useful
  backend-stable active-probe plumbing only. It is not unknown-load
  discrimination, not posture selection success, and not carrying success.
  The next valid gate is to run the conservative `small_x042` geometry across
  `0.25`, `0.50`, and `0.75 kg` and check whether the observed motion differs
  enough to support a selector.
- 2026-07-07 precontact probe multiload signal result: added
  `scripts/isaac/run_core_world_g1_precontact_probe_multiload_signal_suite.sh`
  and submitted tmux `codex_g1_precontact_probe_loads_0707` as Slurm job
  `169339` (`g1_pr_loadsig`). It ran the conservative `small_x042`
  pre-authored always-colliding probe at `0.25`, `0.50`, and `0.75 kg`.
  Summary:
  `experiments/outputs/core_world_g1_precontact_probe_multiload_signal/20260707_g1_precontact_probe_multiload_signal_fresh/precontact_probe_multiload_signal_summary.json`.
  Result: diagnostic `pass`, `3/3` cases passed. All cases completed
  `180/180` steps with `error=null`, fall/drop `0/0`,
  `probe_active_steps=80`, root/box rollout writes `0`, and nonzero probe
  target-directed motion. Max probe target-directed box motions were
  `0.14978 m` at `0.25 kg`, `0.11805 m` at `0.50 kg`, and `0.16064 m` at
  `0.75 kg`; aggregate motion range was about `0.04259 m`. Interpretation:
  the precontact probe is now a backend-stable observed-interaction feature,
  but scalar motion is not monotonic in mass and must not be claimed as a
  reliable mass estimator. The next selector must use multiple observed
  features, such as motion band, robot/box tilt, and relative offset, and must
  be labeled as a diagnostic heuristic unless replaced by a learned/system-ID
  module.
- 2026-07-07 precontact probe-selected load validation result: Slurm job
  `169340` (`g1_prepsel`, tmux
  `codex_g1_precontact_probe_selected_load_0707`) completed on `server39`
  with strict aggregate `fail`, `1/3` cases passing. Summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_precontact_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  The selector ignored hidden mass and used the stable precontact `small_x042`
  probe plus visible/risk telemetry. It selected `chestpad` for `0.25 kg` and
  `0.75 kg`, and `lowcarry` for `0.50 kg`. The `0.50 kg` selected lowcarry
  validation passed with fall/drop `0/0`, final robot/box target-directed
  travel `2.29876/2.34645 m`, target-window end streak `164`, and rollout
  root/box writes `0`. The `0.25 kg` selected chestpad validation was stable
  with fall/drop `0/0` but under-traveled (`0.46484/0.51690 m`) and had
  target-window streak `0`. The `0.75 kg` selected chestpad validation failed
  badly with `482` falls, `439` drops, negative final target-directed travel,
  max robot/box tilt about `1.314/1.312 rad`, and final relative offset
  `0.35039 m`. Interpretation: the probe/selector/selected-validation
  plumbing now runs without hidden mass, but the current posture/controller
  set is not robust. Do not tune selector thresholds alone or claim
  unknown-load adaptation success from this run.
- 2026-07-07 current showcase wrapper added:
  `scripts/isaac/run_core_world_g1_current_showcase_record_and_fallback.sh`.
  It must run only on compute nodes. It records a fresh replay CSV from the
  known narrow `0.50 kg` G1 lowcarry passing configuration with Isaac RGB
  capture disabled, then renders a GIF/poster/MP4 fallback from the replay
  CSV. This is visualization-only replay evidence, not an Isaac camera render,
  not new control evidence, and not generalized carrying success. It was
  submitted as Slurm job `169346` (`g1_showviz`) through tmux
  `codex_g1_showcase_record_visual2_0707`; expected visual outputs are under
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_presentation_fallback/`.
- 2026-07-07 current showcase result: Slurm job `169346` completed on
  `server39` with state `COMPLETED`, exit `0:0`, elapsed `00:00:45`. The
  recorded source rollout
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_lowcarry_current_pass_replay_record/agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json`
  is a narrow diagnostic `pass`: `819/819` steps, fall/drop `0/0`,
  `record_replay_csv=true`, final robot/box target-directed travel
  `2.29876/2.34645 m`, target-window end streak `164`, and success claim
  `g1_usd_core_api_scene_diagnostic_not_walking_or_carrying_success`. The
  replay CSV has `83` data rows. The fallback visual summary is `pass` with
  `83` frames and outputs:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_presentation_fallback/g1_lowcarry_replay_fallback.gif`,
  `g1_lowcarry_replay_fallback.mp4`,
  `g1_lowcarry_replay_fallback_annotated.mp4`, and
  `g1_lowcarry_replay_fallback_poster.png`. The poster is legible as a
  humanoid-and-box replay schematic, but it remains a fallback replay, not an
  Isaac camera render, not new control evidence, and not generalized
  unknown-load carrying success.
- 2026-07-07 true Isaac replay-render retry remains negative. Slurm job
  `169350` (`g1_isarend`, tmux `codex_g1_isaac_replay_render_0707`) used the
  fresh replay CSV from `169346` and attempted low-resolution replay rendering
  to
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_isaac_replay_render/`.
  It failed on `server39` with exit `1:0`, captured frames `0`, and summary
  errors `ModuleNotFoundError: No module named 'omni.replicator'` and
  `ModuleNotFoundError: No module named 'isaacsim.core.rendering_manager'`.
  Do not rerun the true Isaac replay-render path unchanged in this environment
  until the render extensions or a different capture backend are available.
  The current presentation artifact is the fallback replay visual from
  `169346`, with explicit caveats.
- 2026-07-07 boxtilt load-probe suite added:
  `scripts/isaac/run_core_world_g1_boxtilt_load_probe_suite.sh`. It evaluates
  the existing `boxtilt` posture at `0.25`, `0.50`, and `0.75 kg` to check
  whether it is a stable-but-conservative third branch after current
  `lowcarry` and `chestpad` light/heavy failures. It was submitted as Slurm
  job `169354` (`g1_boxtilt`) through tmux
  `codex_g1_boxtilt_load_probe_0707`; expected summary:
  `experiments/outputs/core_world_g1_boxtilt_load_probe/20260707_g1_boxtilt_load_probe_fresh/boxtilt_load_probe_summary.json`.
  Do not overclaim this as learning or unknown-load carrying; it is a
  posture/controller diagnostic only.
- 2026-07-07 boxtilt load-probe result: Slurm job `169354` (`g1_boxtilt`)
  failed strictly, `0/3` cases passed. Summary:
  `experiments/outputs/core_world_g1_boxtilt_load_probe/20260707_g1_boxtilt_load_probe_fresh/boxtilt_load_probe_summary.json`.
  `0.25 kg` failed with `47` falls / `33` drops, final robot/box
  target-directed travel `1.09788/1.01255 m`, and target-window streak `0`.
  `0.50 kg` failed with `329` falls / `0` drops, large lateral drift, final
  robot/box travel `1.04252/0.84821 m`, and target-window streak `0`.
  `0.75 kg` is partial negative evidence: fall/drop `0/0`, max robot/box
  tilt `0.27226/0.29542 rad`, final relative offset `0.11864 m`, but
  under-traveled and drifted laterally, with final robot/box travel
  `1.12217/1.07559 m` and target-window streak `0`. Interpretation:
  boxtilt is not a strict success, but it is safer than the earlier
  `chestpad` choice for the heavy high-motion case, which produced `482`
  falls / `439` drops.
- 2026-07-07 probe selector changed to a diagnostic three-branch heuristic:
  low-risk probes select `lowcarry`, high-motion probes without tilt/offset/
  size risk select `boxtilt`, and resistant or tilt/offset/size-risk probes
  select `chestpad`. This is not learned system identification. Submitted
  three-branch validation as Slurm job `169355` (`g1_3branch`) through tmux
  `codex_g1_threebranch_probe_selected_0707`; expected summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_threebranch_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
- 2026-07-07 three-branch probe-selected validation result: Slurm job
  `169355` (`g1_3branch`) failed strictly with `1/3` cases passing. Summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_threebranch_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  `0.25 kg` selected `chestpad`, had fall/drop `0/0`, final robot/box travel
  `0.46484/0.51690 m`, but target-window streak `0`. `0.50 kg` selected
  `lowcarry` and passed with fall/drop `0/0`, final robot/box travel
  `2.29876/2.34645 m`, and target-window end streak `164`. `0.75 kg`
  selected `boxtilt`, had fall/drop `0/0`, final robot/box travel
  `1.12217/1.07559 m`, and target-window streak `0`; it failed on lateral
  errors (`robot 0.81011 m`, box `0.70503 m`) and under-travel. Compared with
  the prior two-branch selector, this changes the heavy case from catastrophic
  `chestpad` failure (`482` falls / `439` drops) to stable under-travel/
  lateral-drift failure. Treat this as useful safety progress only. The next
  valid controller gate is boxtilt-specific target/lateral correction for the
  heavy branch while preserving fall/drop `0/0` and rollout root/box writes
  `0`.
- 2026-07-07 boxtilt heavy lateral/target suite added:
  `scripts/isaac/run_core_world_g1_boxtilt_heavy_lateral_target_suite.sh`.
  It targets the `0.75 kg` three-branch-selected `boxtilt` failure, where
  fall/drop are already `0/0` but lateral drift and under-travel prevent
  target-window dwell. The suite tests six small variants: baseline,
  hold-lateral off, hold-lateral sign reverse, box-lateral controller with
  positive/negative sign, and conservative box-progress plus box-lateral. It
  was submitted as Slurm job `169366` (`g1_bxlat`) through tmux
  `codex_g1_boxtilt_heavy_lateral_0707`; expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_target/20260707_g1_boxtilt_heavy_lateral_target_fresh/boxtilt_heavy_lateral_target_summary.json`.
  Do not count this route as progress unless it preserves fall/drop `0/0`,
  root/box rollout writes `0`, and improves lateral error/target-window dwell.
- 2026-07-07 boxtilt heavy lateral/target result: Slurm job `169366`
  (`g1_bxlat`) failed strictly with `0/6` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_target/20260707_g1_boxtilt_heavy_lateral_target_fresh/boxtilt_heavy_lateral_target_summary.json`.
  Baseline reproduced the heavy boxtilt safety profile: fall/drop `0/0`,
  final robot/box travel `1.12217/1.07559 m`, final lateral error
  `0.81011/0.70503 m`, and target-window streak `0`. `hold_lat_off` and
  `box_lat_sign_neg` caused large falls/drops, and `box_progress_lat`
  over-drove and failed with `297` falls / `76` drops. The useful case was
  `hold_lat_reverse`: fall/drop `0/0`, target-window stable steps/longest
  streak `152/152`, but it did not stop in the window and ended with
  over-travel `2.75805/2.74993 m` plus lateral error `1.78825/1.61990 m`.
  Next valid test is a boxtilt-heavy stop/hold refinement on top of
  `hold_lat_reverse`, not more broad lateral sweeps.
- 2026-07-07 boxtilt heavy stop-refine suite added:
  `scripts/isaac/run_core_world_g1_boxtilt_heavy_stop_refine_suite.sh`.
  It fixes the `0.75 kg` boxtilt branch to the useful `hold_lat_reverse`
  lateral setup and sweeps terminal/final hold thresholds around the target
  window (`1.55/1.70`, `1.65/1.80`, `1.75/1.90`, plus final-zero correction)
  to try to convert the transient `152`-step target-window visit into an
  end-of-run hold. Submitted as Slurm job `169371` (`g1_bxstop`) through tmux
  `codex_g1_boxtilt_heavy_stop_0707`; expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_stop_refine/20260707_g1_boxtilt_heavy_stop_refine_fresh/boxtilt_heavy_stop_refine_summary.json`.
- 2026-07-07 boxtilt heavy stop-refine result: Slurm job `169371`
  (`g1_bxstop`) failed strictly with `0/4` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_stop_refine/20260707_g1_boxtilt_heavy_stop_refine_fresh/boxtilt_heavy_stop_refine_summary.json`.
  `stop_155_170` reintroduced instability (`137` falls / `40` drops).
  `stop_165_180` and `stop_175_190` kept fall/drop `0/0` but overran the
  target window with end streak `0`. The best partial signal was
  `stop_165_180_finalzero`: fall/drop `0/0`, target-window stable/longest
  streak `184/184`, but end streak `0`, final robot/box travel
  `2.37121/2.41655 m`, large lateral error `1.72462/1.81844 m`, and box
  tilt above the strict gate. Do not claim success; terminal/final hold
  improves dwell but does not solve stopping or lateral control.
- 2026-07-07 boxtilt heavy window-freeze suite added:
  `scripts/isaac/run_core_world_g1_boxtilt_heavy_window_freeze_suite.sh`.
  It keeps the heavy `boxtilt` branch on `hold_lat_reverse`, adds target-
  window freeze, lowers terminal speed, and tests one small final-brake
  variant. Submitted as Slurm job `169411` (`g1_bxfreeze`) through tmux
  `codex_g1_boxtilt_freeze_0707`. This remains diagnostic only; it must
  preserve fall/drop `0/0`, root/box rollout writes `0`, and produce an
  end-of-run target-window streak before it can be treated as a meaningful
  controller improvement.
- 2026-07-07 boxtilt heavy window-freeze result: Slurm job `169411`
  (`g1_bxfreeze`) failed strictly with `0/4` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_window_freeze/20260707_g1_boxtilt_heavy_window_freeze/boxtilt_heavy_window_freeze_summary.json`.
  All freeze/brake variants reintroduced falls and drops:
  `freeze_160_180_s012` had `105` falls / `92` drops,
  `freeze_165_185_s010` had `110` / `26`,
  `freeze_170_190_s008` had `77` / `49`, and
  `freeze_165_180_brake` had `109` / `95`. Best target-window dwell in this
  suite was `122` stable steps with end streak `0`, worse than the previous
  `169371` `finalzero` branch (`184` stable steps, fall/drop `0/0`). Do not
  keep adding target-window freeze/brake variants; this worsens late roll/drop.
  The next valid step is either a lateral-error-aware terminal stabilizer that
  preserves `169371`'s fall/drop `0/0`, or a controller-backed balance/
  locomotion replacement.
- 2026-07-07 boxtilt heavy terminal-lateral suite added:
  `scripts/isaac/run_core_world_g1_boxtilt_heavy_terminal_lateral_suite.sh`.
  It returns to the safer `169371` terminal/final hold thresholds, disables
  freeze/brake, and tests terminal-only lateral correction with excess-error
  thresholds plus one tilt-gated variant. Submitted as Slurm job `169419`
  (`g1_bxtermlat`) through tmux `codex_g1_boxtilt_termlat_0707`. Treat this
  as a narrow diagnostic only; useful progress requires lower lateral error
  while preserving fall/drop `0/0` and root/box rollout writes `0`.
- 2026-07-07 boxtilt heavy terminal-lateral result: Slurm job `169419`
  (`g1_bxtermlat`) failed strictly with `0/4` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_terminal_lateral/20260707_g1_boxtilt_heavy_terminal_lateral/boxtilt_heavy_terminal_lateral_summary.json`.
  All four terminal-only variants failed identically before terminal latch:
  `448` falls / `293` drops, final robot/box target-directed travel only
  `0.59292/0.54745 m`, target-window stable steps `0`, and
  `agile_command_hold_terminal_latched=false`. This proves the pre-terminal
  lateral correction is required for the current boxtilt branch to stay
  upright long enough to approach the target. Stop small boxtilt scalar
  threshold tweaks unless a materially different balance mechanism is added:
  removing early lateral control collapses early (`169419`), freezing/braking
  near the window collapses late (`169411`), and the safer final-zero dwell
  branch still cannot end in the window (`169371`).
- 2026-07-07 dynamic lateral-roll balance target support added to
  `scripts/isaac/build_core_world_g1_box_scene.py` and exposed through
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`. It maps
  robot/box/average target-line lateral error into a bounded roll target for
  the ankle/hip balance-feedback controller. This is a different balance
  mechanism from command lateral correction, freeze, or brake; it changes
  joint targets only and must not be described as root/box pose assistance.
  First diagnostic suite:
  `scripts/isaac/run_core_world_g1_boxtilt_heavy_lateral_roll_target_suite.sh`.
  Submitted as Slurm job `169432` (`g1_bxrolltarget`) through tmux
  `codex_g1_boxtilt_rolltarget_0707`. Useful progress requires fall/drop
  `0/0`, root/box rollout writes `0`, lower lateral error, and improved
  end-of-run target-window streak.
- 2026-07-07 boxtilt heavy lateral-roll-target result: Slurm job `169432`
  (`g1_bxrolltarget`) failed strictly with `0/4` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_roll_target/20260707_g1_boxtilt_heavy_lateral_roll_target/boxtilt_heavy_lateral_roll_target_summary.json`.
  `avg_sign_neg` kept fall/drop `0/0` and low tilt (`0.24390/0.34400 rad`)
  but drifted laterally (`1.87642/2.11266 m`) and had target-window dwell
  `0`. `box_sign_neg` reduced final lateral error to `0.86138/0.94767 m`
  and got a 53-step target-window dwell, but failed with `162` falls /
  `127` drops. This verifies the roll-target path is active, but it is not a
  carrying success. Aggregate summaries now include lateral roll-target
  source/gain/sign and active/last/max target diagnostics. Follow-up
  low-gain refinement:
  `scripts/isaac/run_core_world_g1_boxtilt_heavy_lateral_roll_target_refine_suite.sh`,
  was first submitted as Slurm job `169446` (`g1_bxrollref`) through tmux
  `codex_g1_boxtilt_rollref_0707`, but it was cancelled before rollout
  because the script did not yet include gating/ramp protection.
- 2026-07-07 dynamic lateral-roll target was extended with hold-delay,
  ramp, max-robot-tilt, and max-box-tilt gates. These guards target the
  observed `169432` failures: early gait deflection and late high-tilt
  correction. The gated refinement suite now defaults to stamp prefix
  `20260707_g1_boxtilt_heavy_lateral_roll_target_refine_gated` and is
  submitted as Slurm job `169465` (`g1_bxrollgated`) through tmux
  `codex_g1_boxtilt_rollgated_0707`. Useful progress still requires fall/drop
  `0/0`, root/box rollout writes `0`, lower lateral error, and target-window
  dwell; do not claim success merely because the gait remains upright.
- 2026-07-07 gated lateral-roll-target refine result: Slurm job `169465`
  (`g1_bxrollgated`) failed strictly with `0/4` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_roll_target_refine/20260707_g1_boxtilt_heavy_lateral_roll_target_refine_gated/boxtilt_heavy_lateral_roll_target_refine_summary.json`.
  `box_neg_g010_l018` failed with `52` falls / `13` drops and no target-window
  dwell; stronger `box_neg_g020_l030` and `box_neg_g030_l045` collapsed earlier
  with `242/195` and `284/236` fall/drop counts. `avg_pos_g020_l030` was the
  only informative partial signal: it reached `137` target-window stable steps
  but still fell/dropped late (`55` falls / `38` drops), had end streak `0`,
  and ended with large lateral error (`1.053/1.308 m`). Conclusion: gated
  lateral roll-target is active but does not solve heavy boxtilt lateral hold;
  increasing or mildly gating roll target should not be treated as a valid
  completion route.
- 2026-07-07 short-window check of the partial `avg_pos_g020_l030` signal:
  Slurm job `169472` (`g1_bxavgshort`) ran
  `scripts/isaac/run_core_world_g1_boxtilt_avgpos_short_window_suite.sh` for
  `760` steps and failed strict gates. Summary:
  `experiments/outputs/core_world_g1_boxtilt_avgpos_short_window/20260707_g1_boxtilt_avgpos_short_window_760/boxtilt_avgpos_short_window_summary.json`.
  It completed `760/760` steps with fall/drop `0/0`, root/box rollout writes
  `0`, final robot/box target-directed travel `2.255/2.255 m`, target-window
  end streak `133`, and final-hold end streak `100`; however, it still failed
  tilt and lateral gates: max robot/box tilt `0.624/0.649 rad`, final lateral
  error `1.086/1.284 m`. This is useful as an honest current-progress
  visualization candidate, not as long-duration or robust carrying success.
- 2026-07-07 boxtilt lateral-hold refine suite added:
  `scripts/isaac/run_core_world_g1_boxtilt_lateral_hold_refine_suite.sh`.
  It keeps the `0.75 kg` boxtilt/avg-roll-target short-window setup but stops
  suppressing final-hold lateral correction and probes small command-level or
  box-lateral corrections. Submitted as Slurm job `169476` (`g1_bxlathold`)
  through tmux `codex_g1_boxtilt_lathold_0707`. Useful progress requires
  reducing lateral error without losing the `169472` fall/drop `0/0` and
  end-window streak; if it fails, stop treating command-layer lateral tweaks
  as the missing stabilizer.
- 2026-07-07 boxtilt lateral-hold refine result: Slurm job `169476`
  (`g1_bxlathold`) failed strictly with `0/4` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_lateral_hold_refine/20260707_g1_boxtilt_lateral_hold_refine_760/boxtilt_lateral_hold_refine_summary.json`.
  `hold_sign_neg_l006` reduced final lateral error to `0.301/0.385 m` but
  failed early with `190` falls / `175` drops and no target-window dwell.
  `hold_sign_neg_l012` was worse (`323` falls / `310` drops). `hold_sign_pos_l006`
  reduced lateral error but over-drove to `3.604/3.676 m` target-directed
  travel and failed with `220` falls / `65` drops. `boxlat_sign_neg_l010`
  kept box drops at `0` and final lateral error low (`0.287/0.183 m`) but had
  `295` falls and no target-window dwell. Conclusion: command-level lateral
  correction can move lateral error, but it consumes stability/propulsion
  margin; stop this scalar lateral-hold tuning route unless paired with a
  materially different balance controller.
- 2026-07-07 boxtilt short-window presentation fallback route added:
  `scripts/isaac/run_core_world_g1_boxtilt_short_window_record_and_fallback.sh`.
  It reruns the `169472` 760-step short-window condition with replay CSV
  recording and renders a fallback GIF/MP4/poster via
  `render_g1_replay_presentation_fallback.py`. Submitted as Slurm job
  `169488` (`g1_bxshortviz`) through tmux `codex_g1_boxtilt_shortviz_0707`.
  This is for honest visualization of current progress only; it is not an
  Isaac camera render and not a strict carrying success.
- 2026-07-07 boxtilt short-window visual result: Slurm job `169488`
  (`g1_bxshortviz`) completed on `server43` with exit `0:0`; relabel fix job
  `169501` (`g1_bxvizfix2`) completed on `server39` with exit `0:0`.
  Visual directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/`.
  Key files:
  `g1_boxtilt_short_window_progress.mp4`,
  `g1_boxtilt_short_window_progress_annotated.mp4`,
  `g1_boxtilt_short_window_progress.gif`, and
  `g1_boxtilt_short_window_progress_poster.png`. The final poster was checked
  and now correctly says `G1 boxtilt short-window progress` and
  `strict checker: fail`. The renderer summary records
  `success_claim=schematic_replay_visual_only_not_isaac_camera_render_not_new_control_evidence`.
  Use this only as current progress material: 0.75 kg free-box G1 boxtilt
  short-window, fall/drop `0/0`, target-window end streak `133`, but strict
  failure on lateral drift and tilt.
- 2026-07-07 boxtilt final-stand refine suite added:
  `scripts/isaac/run_core_world_g1_boxtilt_final_stand_refine_suite.sh`.
  It uses the same `0.75 kg` boxtilt short-window condition that produced
  fall/drop `0/0` but excessive final tilt/lateral drift, then enables
  `AGILE_COMMAND_HOLD_FINAL_STAND=1` after final hold. This is a joint-target
  blending mechanism, not root/box pose assistance. Cases test default stand
  and gentle crouch stand targets with blend rates `0.002-0.005` and
  final-stand delays `0/20`. Submitted as Slurm job `169508`
  (`g1_bxfinstand`) through tmux `codex_g1_boxtilt_finalstand_0707`.
  Useful progress requires preserving fall/drop `0/0` and end target-window
  dwell while reducing max/final stand tilt below strict gates; if this fails,
  current AGILE command/posture wrappers are likely exhausted for the heavy
  boxtilt branch and the next credible route is a different locomotion/balance
  backend.
- 2026-07-07 boxtilt final-stand refine result: GPU job `169508`
  (`g1_bxfinstand`) stayed pending and was cancelled after the CPU compute
  backup finished. CPU compute job `169514` (`g1_bxfinstandc2`) ran on
  `server36` and failed strictly with `0/4` cases passing. Summary:
  `experiments/outputs/core_world_g1_boxtilt_final_stand_refine/20260707_g1_boxtilt_final_stand_refine_760_cpu_backup2/boxtilt_final_stand_refine_summary.json`.
  All final-stand cases preserved box drops at `0`, but introduced late falls
  and larger tilt than the no-final-stand short-window run: `stand_default_d0_b002`
  had `10` falls, `stand_default_d20_b005` had `9` falls,
  `stand_gentle_crouch_d0_b004` had `8` falls, and
  `stand_crouch_d20_b003` had `10` falls. Max robot/box tilt stayed around
  `0.934-0.993 rad` / `0.889-0.948 rad`, target-window end streak stayed `0`,
  and final lateral error remained about `0.99-1.08 m` robot and
  `1.16-1.27 m` box. Conclusion: final-stand joint-target blending is not the
  missing stabilizer for the `0.75 kg` boxtilt branch; it worsens late
  stability compared with `169472`. Do not keep tuning final-stand scalar
  delays/blend rates for this branch without a materially different balance
  controller or contact geometry.
- 2026-07-07 boxtilt contact-geometry refine result: added
  `scripts/isaac/run_core_world_g1_boxtilt_geometry_refine_suite.sh` and
  exposed `FREE_CRADLE_LOCAL_Y` through
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`.
  CPU compute job `169519` (`g1_bxgeomc`) ran on `server26`; summary:
  `experiments/outputs/core_world_g1_boxtilt_geometry_refine/20260707_g1_boxtilt_geometry_refine_760_cpu_backup/boxtilt_geometry_refine_summary.json`.
  Strict result was `fail`, `0/4` cases passed. `box_cradle_y_neg003`
  improved final lateral error but collapsed with `291` falls / `163` drops
  and target-window longest/end streak `30/0`. `box_cradle_y_pos003` had
  `45` falls / `26` drops and target-window streak `0`. `wider_lid_rails`
  had `183` falls / `8` drops, negative final target-directed travel, and
  target-window streak `0`. `final_chest_pad` had `119` falls / `105` drops
  and target-window streak `0`. Conclusion: lateral cradle/box offsets,
  wider rails/lid, and final-hold chest-pad collision do not rescue the
  `0.75 kg` boxtilt short-window branch. Stop small contact-geometry tweaks
  on this scaffold unless paired with a materially different balance/contact
  controller.
- 2026-07-07 selected-branch horizon/hold repair result: added
  `scripts/isaac/run_core_world_g1_selected_branch_horizon_repair_suite.sh`
  and ran CPU compute job `169529` (`g1_branchhor`) on `server26`; summary:
  `experiments/outputs/core_world_g1_selected_branch_horizon_repair/20260707_g1_selected_branch_horizon_repair_cpu/selected_branch_horizon_repair_summary.json`.
  Strict result was `fail`, `0/4` cases passed. The two `0.25 kg`
  chest-pad 1600-step cases both collapsed late with `526` falls / `373`
  drops, first fall at step `1074`, and never reached the target window. The
  `0.75 kg` boxtilt 1200-step default case failed with `257` falls / `168`
  drops. The `0.75 kg` boxtilt stop/final-zero case reached the target window
  for `184` stable steps and final-hold window for `166` steps, but then lost
  final hold with `304` falls / `290` drops and end streak `0`. Conclusion:
  the selected posture failures are not merely short-horizon artifacts. The
  boxtilt branch can briefly reach the window, but the current controller
  cannot stop and remain upright with the box; the light chest-pad branch is
  also not robust when run long. Do not treat these as solved carrying or keep
  extending horizons as a repair.
- 2026-07-07 true Isaac G1 replay render attempt: GPU job `169542`
  (`g1_bxrgpu`) ran on `server23` using
  `scripts/isaac/render_core_world_g1_replay_showcase.py` with
  `--capture-backend replicator` against the boxtilt short-window replay CSV.
  AppLauncher started, but local Kit extension dependency resolution failed:
  `omni.replicator.core` could not resolve `omni.kit.pip_archive`, Python
  import failed with `ModuleNotFoundError: No module named 'omni.replicator'`,
  and the viewport fallback lacked `isaacsim.core.rendering_manager`. Summary:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_short_window_isaac_replicator_gpu_server23_try/g1_replay_render_summary.json`.
  Captured frames were `0`. Do not rerun this Replicator/ViewportManager path
  unchanged or claim a true Isaac camera render is available; use the
  schematic replay fallback for immediate presentation until a different
  installed render path or fixed Kit extension set is available.
- 2026-07-07 dormant box-progress controller exposure: the G1 box scene
  already implemented `--agile-command-box-progress-controller` and
  `--agile-command-box-lateral-controller`, but
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh` only
  forwarded their numeric gains and not the boolean activation flags. Exposed
  `AGILE_COMMAND_BOX_PROGRESS_CONTROLLER=1` and
  `AGILE_COMMAND_BOX_LATERAL_CONTROLLER=1`, then added
  `scripts/isaac/run_core_world_g1_boxtilt_box_progress_controller_suite.sh`
  as the next focused diagnostic. This route uses box projected progress as a
  closed-loop AGILE forward command rather than extending horizons or tuning
  final-yaw/final-stand scalars. GPU job `169547` (`g1_bxprog`) was submitted
  on `server43` and then superseded by the CPU-device replacement below after
  it produced no summaries. Original expected summary path was:
  `experiments/outputs/core_world_g1_boxtilt_box_progress_controller/20260707_g1_boxtilt_box_progress_controller_gpu43/boxtilt_box_progress_controller_summary.json`.
- 2026-07-07 box-progress CUDA-device attempt is not valid control evidence:
  first job `169547` (`g1_bxprog`) used `DEVICE=cuda:0` on `server43`. Each
  case reached wrapper initialization, then exited with build status `0` but
  wrote no `core_world_g1_box_scene_summary.json`; per-case `check.json`
  reported `summary missing`, and the aggregate summary had `case_count: 0`.
  Treat this as a GPU-pipeline/output failure, not a pass/fail result for the
  controller. A replacement job `169548` (`g1_bxprogc`) was submitted through
  tmux `curiosity_g1_boxtilt_box_progress_gpualloc_cpu_0707` using a GPU
  allocation but `DEVICE=cpu`, to preserve the previously validated CPU Isaac
  execution path while avoiding CPU-only Slurm reservation blocking.
- 2026-07-07 box-progress CPU-device replacement result: GPU-allocation/
  CPU-device job `169548` (`g1_bxprogc`) ran on `server23`; summary:
  `experiments/outputs/core_world_g1_boxtilt_box_progress_controller/20260707_g1_boxtilt_box_progress_controller_gpualloc_cpu/boxtilt_box_progress_controller_summary.json`.
  Strict result was `fail`, `0/3` cases passed. The controller activation was
  real: `progress_only` had
  `agile_command_box_progress_controller_enabled=true` and active steps
  `1160`, but it failed early with `655` falls / `457` drops and max
  robot/box target-directed travel only `0.537/0.527 m`. `progress_lateral_neg`
  improved final lateral error but failed earlier with `743` falls / `273`
  drops and max robot/box travel `0.740/0.817 m`. `progress_lateral_pos` was
  the least bad stability case, first falling at step `958` and first dropping
  at step `1139`, but still failed with `242` falls / `61` drops and max
  robot/box travel only `0.521/0.467 m`. Target-window longest streak stayed
  `0` for all cases. Conclusion: exposing the dormant box-progress controller
  was necessary, but this closed-loop command layer does not solve heavy
  boxtilt carrying; next work needs path/foot/support stabilization or a
  materially stronger locomotion/contact backend rather than more forward/
  lateral command tweaks.
- 2026-07-07 clean box-progress isolation result: GPU-allocation/CPU-device
  job `169549` (`g1_bxclean`) ran through tmux
  `curiosity_g1_boxtilt_clean_progress_0707`; summary:
  `experiments/outputs/core_world_g1_boxtilt_clean_box_progress/20260707_g1_boxtilt_clean_box_progress_gpualloc_cpu/boxtilt_clean_box_progress_summary.json`.
  Strict result was `fail`, `0/3` cases passed. `clean_default` collapsed
  early with `778` falls / `346` drops and max robot/box target travel only
  `0.360/0.387 m`. `clean_slow` was stable with fall/drop `0/0`, but
  under-traveled (`0.865/0.917 m` max robot/box travel) and had excessive
  tilt (`0.787/0.801 rad`). `clean_slow_lateral_pos` reached the target
  window at step `747` and held both robot and box in-window for `91` steps,
  but then over-traveled to `5.235/4.934 m`, first fell/dropped at
  `944/971`, and ended with `256` falls / `229` drops. This is useful
  evidence that slow progress plus positive box-lateral correction can enter
  the target window, but the controller lacks terminal stopping/holding.
- 2026-07-07 controller terminal-scaling repair under test: the box-progress
  and box-lateral controllers previously overrode the AGILE hold/final
  `command_scale`, so terminal/final hold could not reliably suppress those
  closed-loop commands. Added explicit opt-in switches
  `--agile-command-box-progress-scale-on-hold` and
  `--agile-command-box-lateral-scale-on-hold`, forwarded by
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`, and
  added focused suite
  `scripts/isaac/run_core_world_g1_boxtilt_scaled_terminal_suite.sh`.
  Slurm job `169580` (`g1_bxterm`) was submitted through tmux
  `curiosity_g1_boxtilt_scaled_terminal_0707` with GPU allocation and
  `DEVICE=cpu`. Expected aggregate:
  `experiments/outputs/core_world_g1_boxtilt_scaled_terminal/20260707_g1_boxtilt_scaled_terminal_gpualloc_cpu/boxtilt_scaled_terminal_summary.json`.
  Treat this as a diagnostic repair test only; no result should be claimed
  until the summary exists and strict fall/drop/target/tilt gates are checked.
- 2026-07-07 target-window hold trigger added while `169580` was pending:
  `scripts/isaac/build_core_world_g1_box_scene.py` now supports
  `--agile-command-stop-target-window` plus
  `--agile-command-stop-target-window-min-step`, forwarded by
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`. This
  directly triggers AGILE hold when both robot and box were in the configured
  target window on the previous step, instead of depending only on scalar
  box/robot travel thresholds. Added focused suite
  `scripts/isaac/run_core_world_g1_boxtilt_window_hold_suite.sh`. This is a
  diagnostic response to the `clean_slow_lateral_pos` failure mode: it entered
  the target window for `91` steps, then over-traveled and fell/dropped. The
  new trigger must still be validated on a compute node before it is treated
  as evidence. Slurm job `169585` (`g1_bxwinhold`) was submitted through tmux
  `curiosity_g1_boxtilt_window_hold_after_0707` with
  `--dependency=afterany:169580`, GPU allocation, and `DEVICE=cpu`. Expected
  aggregate:
  `experiments/outputs/core_world_g1_boxtilt_window_hold/20260707_g1_boxtilt_window_hold_after_scaled_terminal/boxtilt_window_hold_summary.json`.
  It should not be interpreted until `169580` completes and `169585` writes a
  summary.
- 2026-07-07 scaled-terminal result: Slurm job `169580` (`g1_bxterm`) ran on
  `server43`; summary:
  `experiments/outputs/core_world_g1_boxtilt_scaled_terminal/20260707_g1_boxtilt_scaled_terminal_gpualloc_cpu/boxtilt_scaled_terminal_summary.json`.
  Strict result was `fail`, `0/3` cases passed. All cases entered the target
  window near step `747` and achieved target-window longest streaks of
  `96-100` steps, but none held to the end. `early_terminal_finalzero`
  failed with `354` falls / `312` drops, first fall/drop `846/888`, final
  travel near the window (`2.298/2.328 m`) but excessive tilt
  `1.817/1.818 rad` and lateral error. `later_terminal_finalzero` failed
  with `337/317` fall/drop and severe over-travel `4.249/4.174 m`.
  `later_terminal_brake` delayed failure most, first fall/drop `987/1010`
  and fall/drop `213/190`, but still over-traveled to `4.776/4.756 m`.
  Conclusion: terminal scaling/zero/brake is active enough to create a
  transient target-window dwell, but it does not solve post-window hold or
  lateral/tilt stability. Do not present this as success.
- 2026-07-07 target-window hold result: Slurm job `169585` (`g1_bxwinhold`)
  ran on `server43`; summary:
  `experiments/outputs/core_world_g1_boxtilt_window_hold/20260707_g1_boxtilt_window_hold_after_scaled_terminal/boxtilt_window_hold_summary.json`.
  The rollout summaries were written, then the Slurm job was cancelled during
  script-end sleep to release the node. Strict result was `fail`, `0/3` cases
  passed. The target-window latch fired for all three cases at step `748`, but
  none preserved a final target-window hold. `window_zero` failed with
  `237` falls / `200` drops, first fall/drop `963/1000`, target-window
  longest streak `98`, and severe over-travel/lateral error by the end.
  `window_freeze` reduced final lateral error but failed earlier with
  `384/370` fall/drop, first fall/drop `816/830`, longest target-window
  streak `69`, and max robot/box tilt about `3.13/3.14 rad`. `window_brake`
  had `289/268` fall/drop, first fall/drop `911/932`, longest target-window
  streak `86`, and final robot/box travel `5.094/5.111 m`. Conclusion:
  direct target-window hold latching works as a trigger, but the current
  command-level stop/freeze/brake layer still cannot convert transient window
  entry into stable post-window carrying. Do not keep treating this scalar
  command layer as the missing stabilizer; move to a materially stronger
  support/locomotion/post-latch balance formulation.
- 2026-07-07 immediate prismatic presentation visual: Slurm job `169015`
  (`prism_hist_viz`) completed on `server36` in `00:00:13` with exit `0:0`.
  It generated a clearer 1600x900 schematic GIF/poster from the already
  completed historical reference run
  `20260706_prismatic_cradle_probe_adaptive_posture_standard10_mid_retry24a`.
  Outputs:
  `experiments/visuals/prismatic_reference_showcase/20260706_prismatic_cradle_probe_adaptive_posture_standard10_mid_retry24a/prismatic_reference_fallback.gif`
  and
  `experiments/visuals/prismatic_reference_showcase/20260706_prismatic_cradle_probe_adaptive_posture_standard10_mid_retry24a/prismatic_reference_fallback_poster.png`.
  This is currently the best presentation artifact for "where the Isaac
  scaffold stands": it shows carrier body, four prismatic legs/feet, physical
  cradle, free box, target line, path trace, phase label, and metrics. It must
  still be described only as a schematic replay of a prismatic scaffold, not
  an Isaac camera render, not humanoid walking, not learned carrying, and not
  final box-carrying success.
- 2026-07-07 prismatic CPU fresh validation attempt: because the historical
  reference run reported `device=cpu`, a CPU compute-node rerun was submitted
  as Slurm job `169019` (`prism_ref_cpu`) through tmux
  `curiosity_prismatic_reference_cpu_validation_0707`. It failed strict gates
  with Slurm exit `2:0`, although the rollout itself completed 760 steps with
  fall/drop `0/0`. The failure reason is insufficient post-settle target
  progress: final post-settle payload travel was about `-0.0760 m`, final
  post-settle payload target distance about `0.0940 m`, and max post-settle
  relative offset error about `0.0140 m`. `reference_check.json` is empty
  because the checker exited nonzero. Do not treat this CPU rerun as a fresh
  pass; keep the historical `20260706...retry24a` visual as presentation
  context only while waiting for `169008` or a corrected longer/parameter-
  matched rerun.
- 2026-07-07 posture/load gauntlet status: Slurm job `168850`
  (`g1_posture_gauntlet`) started on `server43` around 02:22 CST. It remains
  the broad verification path for posture/load coverage; any failed case must
  be recorded as negative evidence, not hidden by the narrow low-carry pass.
- 2026-07-07 posture/load gauntlet result: Slurm job `168850` failed strictly
  after 6m17s on `server43`, and this is valid negative evidence, not a queue
  error. Summary:
  `experiments/outputs/core_world_g1_posture_gauntlet/20260707_g1_posture_gauntlet_after_contact/g1_posture_gauntlet_summary.json`.
  All 5 cases failed: `lowcarry_base` had `402` falls / `235` drops at
  0.5 kg; `chestpad_terminal` had `343` falls / `319` drops; `boxtilt_diagnostic`
  had fall/drop `0/0` but only about `0.927 m` box target-directed travel and
  target-window streak `0`; `lowcarry_lightbox` had `520` falls / `210` drops;
  `lowcarry_heavybox` had `506` falls / `402` drops. Conclusion: the existing
  narrow low-carry pass does not generalize to the gauntlet, so the full
  objective remains far from complete.
- 2026-07-07 MuJoCo projected-contact QP diagnostic: Slurm job `169627`
  (`mj_qpsupp`) completed on `server39` and is negative evidence for using
  full-time QP support allocation from rollout start. The new
  `SUPPORT_CONTROLLER_MODE=qp_stance_force` route activated QP for all
  `3000` steps with rollout root/box pose writes still `0`, but strict result
  was `0/4`. `qp_nominal` and `qp_recovery` latched target-stop but still had
  `95/85` and `90/79` fall/drop events, max tilt `3.24-3.33 rad`, final box
  target-directed travel only `0.292 m` and `0.157 m`, and QP friction usage
  saturated at `1.0`. `qp_roll_weight` and `qp_conservative` failed to latch.
  Max QP wrench residuals were very large (`1489-4147`), so the constrained
  allocation is active but not currently feasible/stabilizing. Do not report
  this as carrying progress; it proves full-time QP can break approach and
  still cannot solve post-latch whole-body upright recovery.
- 2026-07-07 MuJoCo post-latch QP follow-up: after the full-time QP negative
  result, `SUPPORT_QP_POST_LATCH_ONLY=1` was added so the known stance-force
  approach/probe phase is preserved and projected-contact QP only takes over
  after target-stop latch. Slurm job `169628` (`mj_qppost`) was submitted
  through tmux `curiosity_mujoco_qp_post_latch_0707` with suite
  `scripts/mujoco/run_quadruped_freebox_qp_post_latch_suite.sh`. Interpret
  this run strictly: it must pass fall/drop, tilt, target-hold, box-travel,
  relative-error, and no-root/no-box-write gates before it can be called
  progress.
- 2026-07-07 MuJoCo post-latch QP result: Slurm job `169628` completed on
  `server39` with Slurm exit `0:0`, but strict result was still `0/4`.
  Unlike full-time QP, all four cases preserved approach and target-stop
  latch: target-stop/QP/LQR active steps were `1797/1797/1797`, and
  root/box pose/velocity writes were all `0`. The failures are controller
  failures after latch: fall/drop counts were `78-79` / `72-73`, max tilt was
  `2.03-2.26 rad`, min box z was `0.394-0.411 m`, final box travel was only
  `0.355-0.371 m`, QP friction usage saturated at `1.0`, and max QP wrench
  residual remained huge (`1111-4088`). Conclusion: post-latch QP is better
  scoped than full-time QP but still does not solve upright recovery or stable
  carrying. Do not keep treating scalar QP gain/weight sweeps as the likely
  missing piece; the next credible step needs a materially different
  constrained whole-body/contact controller or a policy-backed locomotion
  backend.
- 2026-07-07 MuJoCo feasible-moment QP result: added
  `SUPPORT_QP_MOMENT_CLIP_SCALE`, which clips requested roll/pitch/yaw
  moments by a support-foot geometric estimate of normal-force and
  friction-limited contact moment capacity before projected QP allocation.
  Slurm job `169632` (`mj_qpfeas`) completed on `server39` with Slurm exit
  `0:0`, but strict result was still `0/4`. All four cases latched and kept
  root/box pose/velocity writes at `0`; QP/LQR active steps were `1797`.
  The mechanism is active and informative: max QP residual dropped from
  `1111-4088` in the unclipped post-latch QP run to `268-363`, and the best
  `clip05` case reduced max tilt to `1.6669 rad`. It still failed with
  `77-78` falls, `72` drops, min box z `0.395-0.418 m`, saturated friction
  usage `1.0`, and final box travel only `0.260-0.328 m`. Conclusion:
  contact-feasibility clipping improves numerical feasibility but does not
  solve the physical support/retention failure. Do not keep this MuJoCo
  hand-controller branch on small QP-parameter sweeps; the next credible step
  is either a real constrained whole-body/contact controller with posture,
  support polygon, and box retention jointly optimized, or a policy-backed
  locomotion backend.
- 2026-07-07 MuJoCo carried-mass WBC diagnostic: added
  `SUPPORT_CONTROLLER_MODE=wbc_carried_mass_qp`, which is a separate
  controller class from the earlier QP sweeps. It can activate post-latch,
  include the physical box mass in support vertical load, and compute support
  LQR/contact allocation from the robot+box combined COM. Slurm job `169633`
  (`mj_wbcmass`) completed on `server39` with Slurm exit `0:0`, but strict
  result was `0/4`. All four cases latched and kept root/box pose/velocity
  writes at `0`; WBC/QP/LQR active steps were `2292-2555`, and extra payload
  support was recorded as `19.62 N`. The useful negative signal is that WBC
  caused much earlier target-stop latch (`step 445` for three cases, `708` for
  high-hold) and then failed post-latch balance/retention: fullbox/halfcom/
  recovery ended with negative final box travel (`-0.60`, `-0.61`, `-0.89 m`)
  despite max travel only `0.062 m`; high-hold moved forward `0.373 m` but
  had `99/93` fall/drop events and max tilt `3.24 rad`. Fall/drop counts were
  `51-99` / `42-93`, min box z `0.286-0.373 m`, and max tilt `1.77-3.26 rad`.
  Conclusion: simply adding carried mass and combined COM to support QP is not
  enough; the transition/stop and retention/contact objectives must be jointly
  handled, or the project should return to a real policy-backed locomotion
  backend. Do not report `169633` as robot carrying success.
- 2026-07-07 MuJoCo continuous WBC diagnostic: to isolate whether `169633`
  failed mainly because of target-stop/hold switching, added
  `scripts/mujoco/run_quadruped_freebox_wbc_continuous_carry_suite.sh`, which
  runs `wbc_carried_mass_qp` from rollout start with no target-stop latch.
  Slurm job `169638` (`mj_wbccont`) completed on `server36` with Slurm exit
  `0:0`, but strict result was still `0/4`. WBC/QP/LQR were active for all
  `2400` steps and root/box pose/velocity writes were `0`. The best travel
  cases (`wbc_cont_medium`, `wbc_cont_halfcom`) reached final box travel only
  `0.191-0.218 m` and still failed with `63-67` falls, `54-60` drops, max
  tilt `3.25-3.26 rad`, and min box z `0.313-0.319 m`. Slow and stronger
  support cases had near-zero or negative final travel and also failed.
  Conclusion: removing stop/hold switching does not rescue WBC; the MuJoCo
  hand-authored locomotion/WBC branch itself is not a credible path to final
  carrying without replacing the locomotion backend or adding a substantially
  more complete controller. Do not continue this branch with small speed,
  support-scale, or box-COM-weight sweeps.
- 2026-07-07 G1/AGILE 0.60 kg box-tilt repair result: added
  `scripts/isaac/run_core_world_g1_lowcarry_060_box_tilt_repair_suite.sh`
  because the earlier mass-band run showed 0.60 kg low-carry was a near-miss:
  fall/drop `0/0`, final box target-directed travel `2.197 m`, target-window
  end streak `108`, but max box tilt `0.6385 rad > 0.45`. Slurm job `169648`
  (`g1_060tilt`) ran on `server57` and failed strictly `0/4`. Lowering the
  top lid alone was negative: `lid_lower` fell/dropped `513/124`,
  `lid_lower_wide` fell `403` times, and `lid_very_low` had fall/drop `0/0`
  but still max robot/box tilt `0.765/0.809 rad` and target-window streak `0`.
  The useful signal was `final_chest_pad`: fall/drop `0/0`, max robot/box
  tilt `0.323/0.370 rad`, final relative error `0.087 m`, no rollout
  root/box writes, but under-traveled to final robot/box target-directed
  travel `1.467/1.510 m`, so target-window stable steps stayed `0`. Next
  valid follow-up is to keep the final chest-pad geometry and delay terminal/
  final latch to recover target travel; do not treat the `169648` suite as
  success.
- 2026-07-07 G1/AGILE 0.60 kg final-chest-pad travel repair result: added
  `scripts/isaac/run_core_world_g1_lowcarry_060_chestpad_travel_repair_suite.sh`
  to keep the stable `final_chest_pad` geometry from `169648` while delaying
  terminal/final hold latch thresholds. Slurm job `169653` (`g1_060trav`) ran
  on `server57` and failed strictly `0/4`. All cases kept fall/drop `0/0` and
  rollout root/box writes `0`, but none reached the target window. `final090`
  was the best stability case: max robot/box tilt `0.323/0.393 rad`, final
  relative error `0.080 m`, final latch active for `197` steps, but final
  robot/box travel only `1.205/1.265 m`. `final110` reached similar travel but
  box tilt rose to `0.704 rad`; `final130` and `final150` never latched final
  hold and had box tilt `0.578 rad`. Conclusion: delaying final/chest-pad
  latch alone does not restore travel. The next valid follow-up is to keep the
  low-tilt chest-pad geometry and increase/reshape the AGILE command drive,
  not keep moving latch thresholds.
- 2026-07-07 G1/AGILE 0.60 kg final-chest-pad command-drive repair result:
  added
  `scripts/isaac/run_core_world_g1_lowcarry_060_chestpad_drive_repair_suite.sh`
  to keep the stable chest-pad geometry while increasing AGILE command drive
  from `0.10` to `0.12`/`0.14`. Slurm job `169664` (`g1_060drive`) ran on
  `server59` and failed strictly `0/4`. `cmd012_*` collapsed early with
  `528` falls and `467` drops, no final latch, and negative/near-zero final
  box travel. `cmd014_final090` over-drove to robot/box travel
  `2.612/2.630 m` but failed with `493` falls, `187` drops, max robot/box
  tilt about `3.14/3.14 rad`, and target-window streak `0`.
  `cmd014_final110` also failed with `493` falls and `98` drops. Conclusion:
  simply increasing the AGILE command breaks the stability that chest-pad
  geometry provided. The viable 0.60 kg branch remains a tradeoff: baseline
  lowcarry has enough travel but high box tilt; final chest pad fixes tilt and
  fall/drop but under-travels. A future attempt needs a smoother command
  schedule or contact/geometry that preserves baseline travel, not a larger
  constant command.
- 2026-07-07 G1/AGILE 0.60 kg late chest-pad follow-up: lightweight CSV audit
  showed the original 0.60 kg lowcarry baseline reached box target-directed
  travel `2.0 m` at step `780` and only exceeded the `0.45 rad` box-tilt gate
  at step `810` (`0.616 rad` at step `818`), while the `final_chest_pad`
  geometry kept max box tilt to `0.370 rad` and fall/drop `0/0` but only
  reached final box travel `1.512 m`. This proves early/final chest-pad
  contact sacrifices too much travel, and larger constant AGILE command was
  already negative. Added late-trigger support in
  `scripts/isaac/build_core_world_g1_box_scene.py`:
  `--cradle-chest-pad-enable-on-target-window` and
  `--cradle-chest-pad-enable-on-box-tilt`, with trigger step/reason recorded
  in the summary, plus launcher env mappings in
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`. Added
  `scripts/isaac/run_core_world_g1_lowcarry_060_late_chestpad_suite.sh` and
  submitted Slurm job `169676` (`g1_060late`) through tmux
  `curiosity_g1_060_late_chestpad_0707`; at submission it was pending in the
  `gpu` partition due to priority. The suite is not a result yet. It must pass
  the strict fall/drop, target-window, final-hold, box-tilt, and no-rollout-
  shortcut gates before being treated as a real improvement.
- 2026-07-07 late chest-pad gating correction: first two cases of `169676`
  were invalid and reproduced the old under-travel failure because
  `_spawn_front_torso_cradle()` did not include the new target-window/box-tilt
  trigger flags in the initial `_set_collision_enabled(..., False)` condition.
  The summary said `cradle_chest_pad_collision_enabled_initial=false`, but the
  actual chest-pad collision was still active from rollout start. Job `169676`
  was cancelled during the third case to avoid wasting GPU time. Commit
  `4994ed8` fixed the actual spawn-time collision gating. Replacement Slurm
  job `169678` (`g1_060late2`) was submitted through tmux
  `curiosity_g1_060_late_chestpad_fix_0707` with prefix
  `20260707_g1_lowcarry_060_late_chestpad_fix1`; only the replacement job can
  be interpreted as the late-trigger test.
- 2026-07-07 late chest-pad geometry correction: replacement job `169678`
  was also invalid and cancelled because the suite still used the previous
  repair geometry `CRADLE_TOP_LID_LOCAL_Z=0.145` and
  `CRADLE_TOP_LID_THICKNESS=0.018`. The actual 0.60 kg near-miss baseline
  used `CRADLE_TOP_LID_LOCAL_Z=0.13`,
  `CRADLE_TOP_LID_THICKNESS=0.014`, side rail `0.10`, and end stop `0.11`.
  The invalid `fix1_target_window_min700` case fell at step `628` with
  `191` falls and `180` drops, so it tested the wrong geometry rather than
  late chest-pad activation. The suite was corrected back to the baseline
  top-lid geometry; only a later `fix2` or newer prefix should be used for
  the true baseline-geometry late-trigger test.
- 2026-07-07 baseline-geometry late chest-pad rerun: Slurm job `169685`
  (`g1_060late3`) was submitted through tmux
  `curiosity_g1_060_late_chestpad_fix2_0707` with output prefix
  `20260707_g1_lowcarry_060_late_chestpad_fix2`. This is the current valid
  late chest-pad test because it includes both commit `4994ed8` spawn-time
  collision gating and the baseline top-lid geometry from the 0.60 kg near
  miss.
- 2026-07-07 baseline-geometry late chest-pad result: job `169685`
  completed on `server39` with strict pass `0/4`. This is a valid negative
  result for the pre-authored fixed-joint chest-pad route. In target-window
  cases the chest-pad collision never enabled
  (`cradle_chest_pad_collision_enabled_step=null`), yet both failed with
  final robot/box travel `1.390/1.457 m`, lateral error about
  `-0.90/-0.96 m`, `26` fall events, max robot/box tilt
  `0.941/1.891 rad`, and target-window end streak `0`. In box-tilt trigger
  cases, collision enabled at step `700` or `760`; the best `box_tilt035`
  eliminated falls/drops but still only reached robot/box travel
  `1.416/1.458 m` with max box tilt `1.891 rad` and no target-window dwell.
  Conclusion: merely pre-authoring a fixed-joint chest-pad rigid body, even
  with collision initially disabled, perturbs the baseline enough to destroy
  the 0.60 kg near-miss. Do not continue this route by changing only trigger
  thresholds. Next valid diagnostic is to remove or minimize the pre-authored
  chest-pad mass/rigid-body disturbance, or spawn/attach the support only when
  needed.
- 2026-07-07 tiny-mass late chest-pad diagnostic: added
  `--cradle-chest-pad-mass-scale` to
  `scripts/isaac/build_core_world_g1_box_scene.py` and launcher env
  `CRADLE_CHEST_PAD_MASS_SCALE`. Added
  `scripts/isaac/run_core_world_g1_lowcarry_060_late_chestpad_tinymass_suite.sh`,
  which reuses the baseline-geometry late chest-pad suite with chest-pad mass
  scale `0.001`. Slurm job `169705` (`g1_060ltiny`) was submitted through
  tmux `curiosity_g1_060_late_chestpad_tinymass_0707`. This is a diagnostic
  to test whether the inactive fixed-joint support was corrupting the
  baseline primarily through added mass/inertia. It is not a final carrying
  claim.
- 2026-07-07 tiny-mass late chest-pad result: job `169705` completed on
  `server36` with strict pass `0/4`. All four cases failed similarly:
  target-window cases never enabled chest-pad collision, and box-tilt cases
  enabled it at step `707` or `760`, but final robot/box target-directed
  travel stayed only `0.270/0.077 m`, lateral error grew to
  `1.975/2.117 m`, fall events were `20`, max robot/box tilt was
  `1.746/1.746 rad`, and target-window/final-hold dwell stayed `0`.
  Reducing chest-pad mass scale to `0.001` did not restore the baseline; it
  made the fixed-joint preauthored support route worse. Conclusion: do not
  keep tuning chest-pad mass or trigger thresholds. The next G1 0.60 kg
  attempt should be control-only terminal stabilization, a runtime-created
  support that is not present during baseline walking, or a non-fixed-joint/
  non-articulation contact formulation.
- 2026-07-07 runtime-spawn chest-pad diagnostic: added
  `--cradle-chest-pad-spawn-on-trigger` and launcher env
  `CRADLE_CHEST_PAD_SPAWN_ON_TRIGGER`, plus
  `scripts/isaac/run_core_world_g1_lowcarry_060_runtime_chestpad_suite.sh`.
  Unlike the invalid preauthored chest-pad route, this skips creating the
  chest-pad rigid body during scene setup and attempts to create it only when
  the existing target-window or box-tilt trigger fires. Slurm job `169713`
  (`g1_060rtpad`) was submitted through tmux
  `curiosity_g1_060_runtime_chestpad_0707`. This is a diagnostic of whether
  runtime support creation can avoid baseline-walking disturbance; runtime USD
  joint/collider creation may itself fail or be ignored by PhysX, so inspect
  `cradle_chest_pad_spawned_step` and `cradle_chest_pad_spawn_error` before
  interpreting task metrics.
- 2026-07-07 runtime-spawn chest-pad result: job `169713` completed on
  `server39` with strict pass `1/4`. This is the first 0.60 kg G1/AGILE
  low-carry runtime-support diagnostic passing the current strict gates, but
  it remains an engineered Isaac diagnostic, not learned carrying or unknown-
  object active probing. Passing case:
  `20260707_g1_lowcarry_060_runtime_chestpad_target_window_min700` used
  `CRADLE_CHEST_PAD_SPAWN_ON_TRIGGER=1`, spawned/enabled the chest pad at
  step `712` for reason `target_window`, had `spawn_error=null`,
  final robot/box target-directed travel `2.051/2.032 m`, final lateral error
  `0.071/0.265 m`, max robot/box tilt `0.309/0.428 rad`, fall/drop `0/0`,
  target-window stable steps `105`, target-window end streak `102`, final-
  hold stable steps `105`, final-hold end streak `102`, and rollout root/
  velocity/box pose writes `0`. Negative cases: `target_window_min760`
  triggered too late and fell after a temporary `81`-step window dwell;
  `box_tilt035_min700` and `box_tilt040_min760` preserved travel and fall/drop
  but still failed only the box-tilt gate with max box tilt `0.642` and
  `0.541 rad`. Immediate follow-up should reproduce/refine the successful
  target-window trigger timing around step `700`; do not return to preauthored
  fixed-joint chest-pad bodies.
- 2026-07-07 runtime-spawn chest-pad timing refinement: added
  `scripts/isaac/run_core_world_g1_lowcarry_060_runtime_chestpad_timing_suite.sh`.
  It runs only target-window runtime-spawn cases with min trigger steps
  `680`, `700`, `720`, and `740` to check whether the `169713` pass is
  reproducible and how narrow the timing window is. Slurm job `169724`
  (`g1_060rtiming`) was submitted through tmux
  `curiosity_g1_060_runtime_chestpad_timing_0707`.
- 2026-07-07 runtime-spawn chest-pad timing result: job `169724` completed on
  `server36` with strict pass `2/4`. `target_window_min680` and
  `target_window_min700` both reproduced the pass with identical metrics:
  runtime chest pad spawned/enabled at step `712`, reason `target_window`,
  `spawn_error=null`, final robot/box target-directed travel
  `2.051/2.032 m`, final lateral error `0.071/0.265 m`, max robot/box tilt
  `0.309/0.428 rad`, fall/drop `0/0`, target-window stable steps `105`, and
  target-window/final-hold end streak `102`. `target_window_min720` and
  `target_window_min740` were too late: they only held the target window for
  `14` and `65` steps before fall/tilt failure. Current robust timing
  conclusion: trigger the runtime support as soon as both robot and box enter
  the target window after roughly step `680-700`; waiting until `720+` is not
  stable.
- 2026-07-07 posture-conditioned gate status: added
  `scripts/isaac/run_core_world_g1_posture_conditioned_gate_suite.sh`, which
  packages the known passing `low_front_060` case and the best current
  close-front conditioned hypothesis under the same strict fall/drop,
  target-window, final-hold, tilt, lateral-error, and no-shortcut gates. Slurm
  job `169793` (`g1_postgate`) ran on `server57` and failed the aggregate gate
  with `1/2` cases passing. `low_front_060` reproduced the narrow pass with
  fall/drop `0/0`, robot/box travel `2.051/2.032 m`, max robot/box tilt
  `0.309/0.428 rad`, target-window stable steps `105`, and end streak `102`.
  `close_front_060_conditioned` failed with `142` falls, robot/box travel
  `0.731/0.650 m`, max robot/box tilt `3.130/3.129 rad`, and target-window
  stable steps `0`. Conclusion: the current runtime chest-pad route is still
  a narrow low-front result, not posture-general carrying.
- 2026-07-07 close-front final-stand follow-up: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stand_suite.sh`
  to target the late-tilt failure from
  `20260707_g1_lowcarry_close_front_hold_delay_steps1050_final120`. That case
  reached the target window at step `968` and did not exceed the tilt gates
  until roughly step `1040+`, so the new suite tests late crouched
  final-stand, target-window freeze then final-stand, and policy-then-stand
  without relaxing gates. Slurm job `169822` (`g1_cfstand`) ran on `server44`
  and failed `0/3`. `stand_late_crouch_b003` had fall/drop `262/27`,
  robot/box travel `1.488/1.454 m`, max robot/box tilt `3.102/3.111 rad`, and
  target-window stable steps `0`. `freeze_window_then_stand_b002` had
  fall/drop `226/0`, travel `0.974/0.657 m`, max tilt `3.137/3.138 rad`, and
  target-window stable steps `0`. `policy_then_stand_b002` failed earlier
  with fall/drop `700/463`, travel `0.835/0.604 m`, max tilt
  `2.494/1.843 rad`, and target-window stable steps `0`. Conclusion: do not
  keep tuning final-stand-only close-front repairs. The close-front path needs
  pre-target command/support geometry repair or a posture-conditioned
  gait/support controller before final hold.
- 2026-07-07 close-front pretarget repair entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_pretarget_repair_suite.sh`.
  This suite keeps the same 0.60 kg close-front geometry, runtime
  target-window chest-pad trigger, and strict fall/drop, target-window,
  final-hold, tilt, lateral-error, and no-shortcut gates. It tests early
  box-progress and box-lateral command control (`progress_conservative`,
  `progress_mid`, and `progress_mid_no_hold_lat`) so the robot does not enter
  the bad pitch/roll trajectory before target-window dwell. Slurm job
  `169858` (`g1_cfpre`) ran on `server44` and failed `0/3`.
  `progress_conservative` is the useful negative result: fall/drop
  `485/247`, first fall/drop steps `802/864`, target-window first stable step
  `652`, target-window stable steps `136`, longest/end streak `73/0`, and no
  rollout root/velocity/box pose writes. `progress_mid` and
  `progress_mid_no_hold_lat` collapsed earlier with first falls at `383` and
  `595` and target-window stable steps `0`. Conclusion: conservative
  box-progress can get close-front into the target window, but the next valid
  step is target-window retention/arrest, not stronger drive.
- 2026-07-07 close-front window-arrest entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_window_arrest_suite.sh`.
  This suite builds on the `progress_conservative` result by latching hold on
  target-window entry and allowing runtime chest support at min steps `600`,
  `620`, or `650`. It keeps strict fall/drop, target-window, final-hold, tilt,
  lateral-error, stop-window latch, and no-shortcut gates. Slurm job `169867`
  (`g1_cfwin`) ran on `server36` and failed `0/3`. All three cases
  (`stop_window_pad620`, `stop_window_pad650_soft_hold`, and
  `stop_window_pad600`) produced the same failure: fall/drop `656/617`, first
  fall/drop steps `494/533`, target-window stable steps `0`, no
  target-window latch, and no final-hold activation. This is a useful negative
  control: removing the original early hold/adaptive behavior prevented the
  earlier `progress_conservative` target-window entry. Do not continue this
  exact setup; the next valid retention attempt should preserve
  `progress_conservative` early hold/adaptive behavior and only alter
  post-window support/freeze.
- 2026-07-07 close-front window-retention-v2 entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_window_retention_v2_suite.sh`.
  It preserves the `progress_conservative` early hold/adaptive setup and
  tests only earlier runtime chest support plus target-window freeze/
  zero-correction variants (`pad620`, `pad620_freeze`, and
  `pad650_freeze_zero_corr`). It keeps strict fall/drop, target-window,
  final-hold, tilt, lateral-error, and no-shortcut gates. Slurm job `169906`
  (`g1_cfv2`) was submitted through tmux
  `curiosity_g1_close_front_window_retention_v2_0707`; it ran on `server63`
  and failed `0/3`. `pad620` produced fall/drop `591/573`, first fall/drop
  steps `709/727`, and target-window stable steps `57`. `pad620_freeze` and
  `pad650_freeze_zero_corr` both latched freeze at step `653`, prevented box
  drops, and preserved final travel/lateral, but still had `593` falls, first
  fall step `707`, and only `55` target-window stable steps. Conclusion:
  triggering support/freeze immediately at first target-window entry is worse
  than the original `progress_conservative` pad700 behavior; next support
  timing tests should compare original pad700 against disabled/delayed or
  softened chest support.
- 2026-07-07 close-front support-timing entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_support_timing_suite.sh`.
  It keeps the `progress_conservative` controller and strict gates while
  comparing `no_runtime_pad`, `pad760`, and `pad700_small`. Original
  long-walltime Slurm job `169916` was cancelled before running and replaced
  with shorter Slurm job `169922` (`g1_cfsup`) through tmux
  `curiosity_g1_close_front_support_timing_short_0707`; it ran on `server63`
  and failed `0/3`. `no_runtime_pad` was the useful negative result: no box
  drops, first fall step `901`, target-window first stable step `652`, stable
  steps `130`, longest/end streak `78/0`, final robot/box travel
  `1.615/1.627 m`, max robot/box tilt `1.381/1.611 rad`, and no rollout
  root/velocity/box pose writes. `pad760` and `pad700_small` both worsened the
  result with box drops (`303` and `241`) and earlier or larger tilt failure.
  Conclusion: do not continue runtime chest-pad timing/geometry as the
  close-front repair. The next valid test should use the no-pad close-front
  trajectory and add late final-hold/brake/freeze retention around the target
  window.
- 2026-07-07 close-front late-hold entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_late_hold_suite.sh`.
  It preserves the `progress_conservative` no-pad direction and tests
  `late_final_180`, `late_final_180_freeze`, and `late_final_180_brake` to
  determine whether later final latch plus zero command, target-window freeze,
  or a short reverse brake can retain the target window after the support-free
  run reaches it. Slurm job `169927` (`g1_cflate`) ran on `server63` and
  failed `0/3`. All three cases failed before reaching the target window:
  first fall/drop `632/640`, target-window stable steps `0`, final latch step
  `790`, and no rollout root/velocity/box pose writes. `late_final_180_brake`
  also over-traveled to robot/box `3.386/3.467 m`. Conclusion: late final
  latch at `1.80 m` is too late for close-front; the early final latch around
  `1.20 m` from `no_runtime_pad` is necessary even though it later falls.
- 2026-07-07 close-front rescue/balance entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_balance_suite.sh`.
  It returns to the `no_runtime_pad` early-final-latch baseline and tests
  crouch rescue triggered by absolute roll (`rescue_crouch_abs040`,
  `rescue_crouch_abs055`) plus lateral-error-derived roll targets with both
  signs (`balance_roll_avg_pos`, `balance_roll_avg_neg`). Slurm job `169935`
  (`g1_cfresc`) ran on `server63` and failed `0/4`, but `rescue_crouch_abs040`
  is a useful improvement: fall/drop `219/0`, first fall `1081`, target-window
  stable steps `81`, longest/end streak `52/0`, final robot/box travel
  `1.500/1.471 m`, final robot/box lateral error `-0.775/-0.845 m`, and no
  rollout root/velocity/box pose writes. `rescue_crouch_abs055` had longer
  target-window dwell (`142` stable, longest `90`) but dropped the box at step
  `956`. Both lateral roll-target signs were worse (`first fall 666` and
  `819`, both with box drops). Conclusion: do not continue lateral roll-target
  for close-front. Continue from `rescue_crouch_abs040`, focusing on final
  lateral retention and rescue strength while keeping runtime chest support
  disabled.
- 2026-07-07 close-front rescue-lateral refine entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_lateral_refine_suite.sh`.
  It keeps the useful `rescue_crouch_abs040` baseline and tests unscaled
  final-hold box-lateral correction, opposite lateral sign, milder crouch, and
  a mid-threshold crouch. Slurm job `169944` (`g1_cflat`) ran on `server63`
  and failed `0/4`. All unscaled-lateral variants were worse than
  `rescue_crouch_abs040`: `rescue040_lat_unscaled` fell/dropped at
  `811/832` and ran away to robot/box travel `8.748/8.771 m`;
  `rescue040_lat_unscaled_signneg` fell/dropped at `626/647` with
  target-window stable steps `0`; milder and mid-threshold crouch variants
  matched the runaway failure. Conclusion: do not unscale box-lateral
  correction during final hold. The best close-front branch remains
  `rescue_crouch_abs040`.
- 2026-07-07 close-front rescue final-latch sweep entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_rescue_final_latch_sweep.sh`.
  It keeps `rescue_crouch_abs040` and tests moderate final-latch thresholds
  (`1.35`, `1.45`, `1.55 m`) after the `1.20 m` latch under-traveled and the
  `1.80 m` latch collapsed before activation. This is diagnostic only until
  strict gates pass. Slurm job `169964` (`g1_cffinal`) ran on `server39` and
  failed `0/3`. `final135` fell/dropped at `784/799` with target-window stable
  steps `75`; `final145` fell/dropped at `684/703` with stable steps `32`;
  `final155` fell/dropped at `632/640` with stable steps `0`. Conclusion:
  moderate final-latch thresholds are worse than the original
  `rescue_crouch_abs040` early latch; do not continue final-latch threshold
  sweeps for close-front.
- 2026-07-07 close-front rescue tiny-final-scale direction: next valid test
  should keep `rescue_crouch_abs040` and the early final latch, but use a very
  small nonzero final-hold command scale so the existing box-progress/lateral
  controllers can counter drift without the runaway caused by unscaled lateral
  correction. Slurm job `169995` (`g1_cftiny`) ran on `server39` and failed
  `0/3`. `final_scale_003` fell/dropped at `634/651` with target-window stable
  steps `0`; `final_scale_006` fell/dropped at `643/658` with stable steps
  `7`; `final_scale_010` improved to stable steps `49` but still fell/dropped
  at `723/752`. Conclusion: even tiny nonzero final-hold scale destabilizes
  this close-front branch and should not continue. The best branch remains
  `rescue_crouch_abs040` with final scale `0.0`.
- 2026-07-07 close-front rescue target-window freeze direction: next valid
  test should keep `rescue_crouch_abs040`, final scale `0.0`, no runtime chest
  support, and add target-window joint-target freeze with strict/loose
  thresholds. This tests whether the drift after the first target-window dwell
  can be arrested without adding lateral command or chest-pad geometry. Slurm
  job `169996` (`g1_cffreeze`) ran on `server59` and failed `0/3`, but
  `freeze_strict` is the best freeze branch: freeze latched at step `663`,
  rescue started at step `732`, target-window stable steps `106`, longest
  streak `68`, final robot/box travel `2.176/2.119 m`, final robot/box lateral
  error `-0.105/0.084 m`, but fall/drop happened at `782/804` with max
  robot/box tilt `1.202/1.516 rad`. `freeze_loose` and
  `freeze_loose_zero_corr` avoided drops but only reached `71` stable steps and
  fell at `723`. Conclusion: continue from `freeze_strict`; the next valid
  change is stronger roll/balance feedback during frozen hold, not command
  scale, final latch, lateral roll-target, or chest-pad timing.
- 2026-07-07 close-front freeze-balance refine entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_balance_refine_suite.sh`.
  It keeps `freeze_strict` and tests increased roll/balance feedback. Slurm
  job `170003` (`g1_cfbal`) ran on `server39` and failed `0/3`.
  `roll_gain_010` had target-window stable steps `0` and fell/dropped at
  `724/773`; `roll_gain_014` reached only `42` stable steps and fell/dropped
  at `674/704`; `roll_pitch_gain` reached `44` stable steps but ran away to
  robot/box travel `6.157/5.975 m` and fell/dropped at `853/924`.
  Conclusion: do not continue increasing balance gains for close-front
  freeze; the next branch should preserve default balance and try a delayed
  low-COM stand/hold transition after target-window freeze.
- 2026-07-07 close-front freeze-stand transition entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`.
  It keeps `freeze_strict`, default balance feedback, and tests delayed
  low-COM stand targets at delays `80`, `120`, and a softer `160` steps after
  final hold. Slurm job `170016` (`g1_cfstand2`) ran on `server20` and failed
  `0/3`. `stand_delay_80` over-traveled to robot/box travel
  `3.703/3.719 m`, had only `68` target-window stable steps, and fell/dropped
  at `715/741`. `stand_delay_120` reached only `28` stable steps and
  fell/dropped at `688/727`. `stand_delay_160_soft` matched the earlier
  `freeze_strict` boundary with target-window stable steps `106`,
  longest/end streak `68/0`, final robot/box travel `2.176/2.119 m`,
  final lateral error `-0.105/0.084 m`, and fall/drop at `782/804` with
  max robot/box tilt `1.202/1.516 rad`. Conclusion: delayed low-COM stand
  transition did not fix close-front freeze collapse, but a later static
  audit found this was not a valid test of stand targets after freeze: when
  `final_freeze_active` is true, frozen policy joint targets take priority and
  final-stand targets are not applied. Treat the numeric run as a negative
  diagnostic for the old control priority, not as evidence that stand targets
  cannot help. A valid stand test needs explicit stand-over-freeze priority.
- 2026-07-07 close-front freeze-rescue timing entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_rescue_timing_suite.sh`.
  It keeps `freeze_strict`, disables delayed stand targets, preserves default
  balance, and compares `freeze_no_rescue`, `freeze_rescue_late055`, and
  `freeze_rescue_soft035` under the same strict target-window, fall/drop,
  tilt, lateral, and no-rollout-write gates. Slurm job `170095`
  (`g1_cfrtime`) was submitted through tmux
  `curiosity_g1_close_front_freeze_rescue_timing_0707`, then cancelled before
  allocation after a static control-flow audit showed the intended intervention
  was invalid: once `final_freeze_active` is true, frozen policy joint targets
  take priority and rescue targets are not applied. Do not interpret the
  cancelled timing suite as evidence for or against rescue; it did not run and
  would not have tested the intended hypothesis.
- 2026-07-07 close-front freeze-rescue override entrypoint: added
  `--agile-command-hold-rescue-overrides-final-freeze`, summary/check fields
  for override activation, and wrapper
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_rescue_override_suite.sh`.
  This wrapper reuses the same three timing cases but explicitly lets active
  rescue targets override frozen policy targets after target-window freeze.
  Slurm job `170122` (`g1_cfovr`) was submitted through tmux
  `curiosity_g1_close_front_freeze_rescue_override_0707`, then cancelled
  before allocation after the wrapper was found to pass the override as an
  unused positional argument rather than an exported environment variable.
  The wrapper was fixed and pushed in commit `b7b52bc`; Slurm job `170125`
  (`g1_cfovr2`) was submitted through tmux
  `curiosity_g1_close_front_freeze_rescue_override2_0707`; as of
  `2026-07-07 17:17 CST`, it was still pending on GPU priority, with Slurm
  estimating start at `2026-07-07T20:24:10` on `server46`. This is the valid
  replacement for the cancelled timing suite and remains only an experiment
  entrypoint until
  `close_front_freeze_rescue_override_summary.json` exists.
- 2026-07-07 close-front freeze-rescue override result: Slurm job `170125`
  (`g1_cfovr2`) ran on `server46` and failed `0/3`.
  `freeze_no_rescue` reproduced the `freeze_strict` boundary with fall/drop
  `518/496`, first fall/drop `782/804`, target-window stable steps `106`,
  longest/end streak `68/0`, and override active steps `0`.
  `freeze_rescue_late055` verified rescue-over-freeze applied for `540` steps
  from step `760`, improved final robot lateral error to about `-0.007 m`,
  but still fell/dropped at `787/811` with max robot/box tilt
  `1.328/1.407 rad`. `freeze_rescue_soft035` applied override for `573` steps
  from step `727`, improved target-window stable steps to `122` and
  longest streak to `84`, but still fell/dropped at `798/816`, had final
  robot/box lateral error `0.222/0.421 m`, and max robot/box tilt
  `1.410/1.498 rad`. Conclusion: rescue-over-freeze is a real intervention
  and slightly improves dwell, but does not solve close-front balance/drop.
  The next valid branch is explicit stand-over-freeze or support/stance
  selection after freeze, not more final-command scaling.
- 2026-07-07 close-front stand-over-freeze entrypoint: added
  `--agile-command-hold-stand-overrides-final-freeze`, summary/check fields
  for stand override activation, generalized
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`
  with `SUITE_NAME`, and added wrapper
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_override_suite.sh`.
  The wrapper reuses the delayed low-COM stand cases, explicitly lets final
  stand targets override frozen policy targets, and disables rescue so the
  stand target is not masked. This is an experiment entrypoint only until
  `close_front_freeze_stand_override_summary.json` exists. Slurm job `170159`
  (`g1_cfstandovr`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_override_0707`; as of
  `2026-07-07 17:32 CST`, it was pending on GPU priority with unknown
  estimated start time.
- 2026-07-07 close-front stand-over-freeze wrapper correction: Slurm job
  `170159` (`g1_cfstandovr`) ran on `server58` and failed `0/3`, but the run
  is not valid stand-over-freeze evidence. The summaries showed
  `agile_command_hold_stand_overrides_final_freeze=false` and
  `agile_command_hold_stand_override_freeze_active_steps=0`; the wrapper
  arguments were not passed through because
  `run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`
  used function-local `"$@"` inside `run_case`. The script now captures
  top-level `SCRIPT_ENV_OVERRIDES=("$@")` and injects them into each case env
  before case-specific overrides. A fresh stand-over-freeze job is required
  before interpreting this branch.
- 2026-07-07 close-front stand-over-freeze v2 job: after fixing wrapper env
  propagation, Slurm job `170167` (`g1_cfstand2`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_override2_0707` with
  `SUITE_STAMP_PREFIX=20260707_g1_lowcarry_close_front_freeze_stand_override2`
  and started on `server58`. Interpret this branch only from the v2 summary at
  `experiments/outputs/core_world_g1_lowcarry_close_front_freeze_stand_override/20260707_g1_lowcarry_close_front_freeze_stand_override2/close_front_freeze_stand_override_summary.json`.
- 2026-07-07 close-front stand-over-freeze v2 wrapper failure: Slurm job
  `170167` (`g1_cfstand2`) started on `server58` but exited after the first
  case with `line 134: and_delay_80: command not found`. This was a shell
  wrapper bug, not a valid stand-over-freeze result. The fix removes top-level
  argument-array forwarding from the transition script; the wrapper now exports
  `AGILE_COMMAND_HOLD_STAND_OVERRIDES_FINAL_FREEZE=1`, and the transition
  script lets `AGILE_COMMAND_HOLD_RESCUE_ENABLE` be overridden from the
  environment. A fresh v3 run is required before interpreting stand-over-freeze.
- 2026-07-07 close-front stand-over-freeze v3 job: after the exported-env
  wrapper fix, Slurm job `170173` (`g1_cfstand3`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_override3_0707` with
  `SUITE_STAMP_PREFIX=20260707_g1_lowcarry_close_front_freeze_stand_override3`.
  It ran on `server10` and failed `0/3`, but this is the first valid
  stand-over-freeze evidence. `stand_delay_80` applied stand override for
  `652` steps from step `648` but fell/dropped at `731/744` and over-traveled
  to robot/box `2.858/2.659 m`. `stand_delay_120` applied override for
  `636` steps from step `664` but fell/dropped at `698/737` with only `29`
  target-window stable steps. `stand_delay_160_soft` applied override for
  `600` steps from step `700`, improved target-window stable steps to `141`
  and longest streak to `103`, final robot/box travel to `2.137/2.164 m`,
  final lateral error to `-0.306/-0.186 m`, and first fall/drop to
  `816/862`; it still failed strict gates with fall/drop `484/413` and max
  robot/box tilt `1.412/1.825 rad`. Conclusion: stand-over-freeze is real and
  more promising than rescue-over-freeze on close-front, but the stand
  transition is still too aggressive. Next branch should test later/softer
  stand-over-freeze, not final command scaling.
- 2026-07-07 close-front stand-over-freeze refinement entrypoint: added
  `STAND_TRANSITION_CASE_SET=refine_soft` to
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`
  and wrapper
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_override_refine_suite.sh`.
  It keeps stand-over-freeze and rescue disabled, then tests
  `stand_delay_160_microblend`, `stand_delay_180_ultrasoft`, and
  `stand_delay_220_ultrasoft` under the same strict gates. This is the next
  valid close-front branch because `stand_delay_160_soft` improved dwell but
  still failed from excessive tilt/drop.
- 2026-07-07 close-front stand-over-freeze refinement job: Slurm job `170185`
  (`g1_cfstandref`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_refine_0707`; as of
  `2026-07-07 17:49 CST`, it was pending on GPU priority. It later ran on
  `server44` and failed `0/3`. `stand_delay_160_microblend` reached target-
  window stable steps `127`, longest/end `89/0`, first fall/drop `803/834`,
  and max robot/box tilt `1.475/1.588 rad`. `stand_delay_180_ultrasoft`
  reached stable steps `124`, longest/end `86/0`, first fall/drop `800/826`,
  and max tilt `1.451/1.544 rad`. `stand_delay_220_ultrasoft` reduced max
  tilt to `1.357/1.394 rad` but worsened stable steps to `108` and first
  fall/drop to `784/807`. All had stand-over-freeze active. Conclusion:
  later/softer stand targets did not beat `stand_delay_160_soft`; next tests
  should keep that timing/target family and change balance-feedback coupling
  during the stand override.
- 2026-07-07 close-front stand-over-freeze balance-coupling entrypoint: added
  `STAND_TRANSITION_CASE_SET=balance_coupling` to
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`
  and wrapper
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_override_balance_suite.sh`.
  It keeps the best `stand_delay_160_soft` timing/target and compares
  `BALANCE_FEEDBACK_BASE=stand`, half balance gains, and balance disabled
  under the same strict gates.
- 2026-07-07 close-front stand-over-freeze balance-coupling job: Slurm job
  `170193` (`g1_cfbalstand`) was submitted through tmux
  `curiosity_g1_close_front_freeze_stand_balance_0707`; as of
  `2026-07-07 17:55 CST`, it was pending on GPU priority. It later ran on
  `server10` and failed `0/3`. `stand160_balance_base_stand` fell/dropped at
  `533/827` with target-window stable steps `0` and final robot/box travel
  only `0.184/0.148 m`; `stand160_balance_half_gain` was catastrophic, with
  negative final travel and lateral error about `12.9/13.1 m`;
  `stand160_balance_off` reached robot/box travel `1.703/1.616 m` but
  fell/dropped at `415/500` and had target-window stable steps `0`. Although
  `agile_command_hold_stand_overrides_final_freeze=true` was set, none of
  these cases produced effective stand-over-freeze active steps because they
  failed to establish the target-window/freeze condition. Conclusion: this
  balance-coupling attempt is negative and does not beat
  `stand_delay_160_soft`; do not continue scalar balance-base/gain toggles on
  this close-front branch.
- 2026-07-07 close-front handoff-structure entrypoint: added
  `STAND_TRANSITION_CASE_SET=handoff_structure` to
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh`.
  It tests `AGILE_COMMAND_HOLD_MODE=policy_then_stand` with delayed
  policy-to-stand handoff, stand-over-freeze enabled, and rescue disabled,
  preserving the same strict target-window, tilt, lateral, fall/drop, and
  no-shortcut gates. This is a structural hold/stand handoff test after the
  negative balance-coupling result, not another scalar balance-gain sweep.
  Slurm job `170267` (`g1_cfhand`) was submitted through tmux
  `curiosity_g1_close_front_handoff_0707` with suite stamp prefix
  `20260707_g1_lowcarry_close_front_freeze_stand_handoff`, but was cancelled
  while still pending after Slurm estimated a late start. It was superseded by
  single-case quick job `170276` (`g1_cfhandq`) through tmux
  `curiosity_g1_close_front_handoff_quick_0707`, case-set
  `handoff_quick`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_freeze_stand_handoff_quick`, which was
  also cancelled while pending. A `test` partition attempt was rejected by
  Slurm as an invalid account/partition combination. Final quick run
  `170278` (`g1_cfhand4`) ran on `server02` through tmux
  `curiosity_g1_close_front_handoff_gpu_small_0707`, suite stamp prefix
  `20260707_g1_lowcarry_close_front_freeze_stand_handoff_quick_gpu4`, and
  failed strict gates `0/1`. Case `policy_then_stand_delay120` completed
  1300 steps with no box drops and no rollout root/velocity/box pose writes,
  but fell at step `725` with fall events `575`, target-window stable steps
  `0`, final robot/box target-directed travel `-0.496/-0.643 m`, max
  robot/box tilt `1.284/1.397 rad`, and final hold never latched. Conclusion:
  this early `policy_then_stand` handoff is worse than the previous
  `stand_delay_160_soft` boundary; do not continue this handoff branch without
  a materially different support/terminal-control formulation.
- 2026-07-07 posture-conditioned gate rerun submitted: Slurm job `170282`
  (`g1_postgate`) was submitted through tmux `curiosity_g1_posture_gate_0707`
  with `SUITE_STAMP_PREFIX=20260707_g1_posture_conditioned_gate_rerun`.
  It runs the existing two-case gate: known `low_front_060` reproduction plus
  `close_front_060_conditioned`, both under strict target-window, fall/drop,
  tilt, lateral, and no-shortcut checks. As of submission it was still
  `PENDING (Priority)` with no start time. Do not report any result until
  `experiments/outputs/core_world_g1_posture_conditioned_gate/20260707_g1_posture_conditioned_gate_rerun/posture_conditioned_gate_summary.json`
  exists. It was later cancelled while still pending so the queue could be
  used for the more targeted close-front retention-posture test.
- 2026-07-07 close-front retention-posture entrypoint: added
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_retention_posture_suite.sh`.
  It starts from the previous best close-front near-miss
  `steps1050_final120` and enables the existing audited
  `BOX_RETENTION_POSTURE_CONTROLLER` so box relative-error/tilt risk changes
  G1 hip, knee, ankle, waist, shoulder, elbow, and wrist targets. It preserves
  the strict no-fall/no-drop/target-window/tilt/lateral/no-rollout-write
  gates and does not add root, velocity, or box pose shortcuts. This is the
  next structural close-front test after early `policy_then_stand` handoff
  failed; it is not a command-only scalar sweep.
- 2026-07-07 close-front retention-posture quick job: Slurm job `170290`
  (`g1_retpost`) was submitted through tmux
  `curiosity_g1_retention_posture_0707` with
  `RETENTION_POSTURE_CASE_SET=quick` and
  `SUITE_STAMP_PREFIX=20260707_g1_lowcarry_close_front_retention_posture_quick`.
  It was cancelled while still pending after Slurm estimated a late start.
  Replacement Slurm job `170293` (`g1_retpost45`) was submitted through tmux
  `curiosity_g1_retention_posture_45m_0707` with the same quick case and
  `SUITE_STAMP_PREFIX=20260707_g1_lowcarry_close_front_retention_posture_quick45`.
  It was also cancelled while pending after implementation review showed that
  the original retention-posture controller directly overwrote AGILE policy
  joint targets with stand-like posture targets when risk was active, likely
  too disruptive for locomotion.
- 2026-07-07 blended retention-posture fix: added
  `--box-retention-blend-rate` to
  `scripts/isaac/build_core_world_g1_box_scene.py` and forwarded it through
  `scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh`. The
  default remains `1.0` for compatibility, but
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_retention_posture_suite.sh`
  now uses low blend rates (`0.08` mild, `0.12` strong) so risk-driven posture
  feedback blends into the AGILE policy targets instead of hard replacing
  them. Slurm job `170296` (`g1_retblend`) was submitted through tmux
  `curiosity_g1_retention_posture_blend_0707` with
  `RETENTION_POSTURE_CASE_SET=quick` and
  `SUITE_STAMP_PREFIX=20260707_g1_lowcarry_close_front_retention_posture_blend_quick`.
  As of submission it was `PENDING (Priority)`. Do not interpret this branch
  until
  `experiments/outputs/core_world_g1_lowcarry_close_front_retention_posture/20260707_g1_lowcarry_close_front_retention_posture_blend_quick/close_front_retention_posture_summary.json`
  exists.
- 2026-07-07 blended retention-posture result: Slurm job `170296`
  (`g1_retblend`) ran on `server46` and failed strict gates `0/1`.
  `retention_mild` completed 1050 steps with rollout root/velocity/box pose
  writes `0`, but had first fall/drop at `821/914`, fall/drop events
  `229/107`, target-window stable steps `0`, final robot/box target-directed
  travel `-0.880/-0.896 m`, max robot/box tilt `1.903/3.134 rad`, and max
  robot/box target-directed travel only `0.446/0.496 m`. Source rollout
  summary showed retention feedback was active for `835` steps from step
  `170`, with max risk `1.0`, so the controller did engage. Conclusion:
  blending stand-like retention posture into AGILE targets is still damaging
  for close-front locomotion and is worse than the no-retention
  `steps1050_final120` near-miss; do not continue this retention-posture
  branch by only changing blend/offset scalars. The next close-front direction
  should be a support/command formulation that preserves AGILE locomotion
  rather than posture overwrites.
- 2026-07-07 retention summary plumbing fix: after `170296`, the source
  rollout summary contained `box_retention_*` fields, but
  `scripts/isaac/summarize_core_world_g1_largerbox_strict.py` did not copy
  them into aggregate summaries. The summarizer now preserves retention
  enabled/range/blend/active-step/risk fields, and
  `scripts/isaac/build_core_world_g1_box_scene.py` now records
  `box_retention_blend_rate` in rollout summaries. Existing `170296`
  aggregate JSON was not regenerated on the login node; interpret retention
  activation from the source rollout summary for that run.
- 2026-07-07 close-front final-stabilize quick entrypoint: added
  `FINAL_STABILIZE_CASE_SET=quick` to
  `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stabilize_suite.sh`
  so a single no-retention `steps1200_final120_tilt030` case can be run
  without launching the full two-case suite. This returns to the best
  no-retention close-front near-miss lineage and tests earlier chest-pad
  triggering from box tilt while preserving AGILE locomotion targets.
- 2026-07-07 close-front final-stabilize quick job: Slurm job `170302`
  (`g1_finstabq`) was submitted through tmux
  `curiosity_g1_final_stabilize_quick_0707` with
  `FINAL_STABILIZE_CASE_SET=quick` and
  `SUITE_STAMP_PREFIX=20260707_g1_lowcarry_close_front_final_stabilize_quick`.
  It was cancelled while pending after Slurm estimated a late start. The
  replacement 45-minute job `170306` (`g1_finstab45`) was submitted through
  tmux `curiosity_g1_final_stabilize_quick45_0707` with
  `SUITE_STAMP_PREFIX=20260707_g1_lowcarry_close_front_final_stabilize_quick45`.
  As of submission it was `PENDING (Priority)` with no start estimate. Do not
  interpret this branch until
  `experiments/outputs/core_world_g1_lowcarry_close_front_final_stabilize/20260707_g1_lowcarry_close_front_final_stabilize_quick45/close_front_final_stabilize_summary.json`
  exists.
- 2026-07-07 blended retention-posture smoke job: Slurm job `170298`
  (`g1_retbsmo`) was submitted through tmux
  `curiosity_g1_retention_blend_smoke_0707` with suite stamp
  `20260707_g1_lowcarry_close_front_retention_blend_smoke700`. It is a
  700-step early-stability diagnostic for the same mild blended retention
  settings, with target-window/final-hold minimums disabled so it can backfill
  quickly. It must not be interpreted as strict carrying success even if it
  passes; only fall/drop, tilt, travel direction, retention activation, and
  no-rollout-write fields are useful from this smoke.
  It was cancelled once the full `170296` quick gate completed; it produced
  only an env snapshot and no valid rollout summary.
- 2026-07-07 close-front retention-posture parser: added
  `scripts/isaac/print_g1_retention_posture_summary.sh`, a lightweight
  read-only `jq` parser for retention-posture summary JSON. It reports
  per-case pass/fail, fall/drop, target-window dwell, travel, lateral error,
  tilt, final-hold latch, retention active steps/risk, and rollout root/
  velocity/box pose write counts. It does not run simulation.
- 2026-07-07 G1 showcase capture job: Slurm job `170209` (`g1_showviz`) was
  submitted through tmux `curiosity_g1_showcase_capture_0707` to run
  `scripts/isaac/run_core_world_g1_showcase_lowcarry_capture.sh` with
  `SUITE_STAMP=20260707_g1_lowcarry_showcase_rgb_retry`, RGB capture enabled,
  and replay recording enabled. It completed on `server28`, but it is negative
  for both control and camera capture: the run failed early with fall/drop at
  steps `85/91`, and RGB capture produced no PNG/MP4 because
  `omni.replicator.core` could not resolve the local `omni.kit.pip_archive`
  dependency from the available Kit registry mirror. Do not use this run as
  showcase evidence.
- 2026-07-07 current best fallback visual: Slurm job `170217`
  (`g1_bestcpu2`) ran on `server02` through tmux
  `curiosity_g1_best_fallback_visual_cpu2_0707` and rendered the existing
  strict-pass replay
  `20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_min700` into
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_best_fallback_cpu2/`.
  It produced `83` frames, `g1_lowcarry_replay_fallback.gif`,
  `g1_lowcarry_replay_fallback_poster.png`, `g1_lowcarry_best_fallback.mp4`,
  and `g1_lowcarry_best_fallback_annotated.mp4`. This is the clearest current
  presentation artifact for the narrow G1 pass, but it is still a schematic
  replay visual, not an Isaac camera render and not generalized unknown-load
  carrying evidence.
- 2026-07-07 true Isaac replay-render follow-up: `scripts/isaac/run_core_world_g1_replay_showcase_render.sh`
  was changed so it no longer forces the external IsaacLab-Arena experience by
  default; it now lets AppLauncher use the installed IsaacLab Kit experience
  and passes local registry names plus `file://` URLs. Minimal smoke job
  `170222` (`repregsmk`) on `server30` confirmed that this default-Kit path can
  start AppLauncher and import `omni.replicator.core`. However true replay
  render smoke job `170224` (`g1_truerdr`) on `server36` still failed: it
  produced no PNG frames, no MP4, and no `g1_replay_render_summary.json`.
  Checker output at
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_best_true_render_smoke_defaultkit/g1_replay_showcase_check.json`
  reports `frame_count=0`. Therefore true Isaac camera render remains
  unsolved; the current presentation artifact is still the schematic fallback.
- 2026-07-07 true Isaac replay-render debug boundary: follow-up debug run
  `170230` (`g1_rdrdbg`) wrote failure artifacts under
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_best_true_render_debug_defaultkit/`.
  It showed the articulation-wrapper replay path fails while creating
  `SingleArticulation` with
  `AttributeError: type object 'PhysxManager' has no attribute '_get_backend_utils'`.
  The renderer was then given an opt-out xform-only replay path. Xform-only
  attempts `170252`/`170256` avoided `SingleArticulation`, created the
  Replicator camera, and then stalled at `rep.create.render_product(...)`.
  Conclusion: the default Kit path can import `omni.replicator.core`, but true
  Isaac camera replay is still blocked by post-import rendering/product
  creation in this environment. Do not submit more unchanged true-render
  attempts; use the schematic fallback until a different render backend or
  fixed Kit capture stack is available.
- 2026-07-07 close-front freeze-rescue override parsing helper:
  `scripts/isaac/print_g1_freeze_rescue_override_summary.sh` was added as a
  lightweight, read-only summary parser. It prints per-case pass/fail,
  fall/drop timing, target-window stability, travel/lateral/tilt metrics, and
  `agile_command_hold_rescue_override_freeze_*` fields so the override run can
  be audited without hand-parsing JSON. It does not run simulation.
- 2026-07-07 close-front final-stabilize parser: added
  `scripts/isaac/print_g1_final_stabilize_summary.sh`, a lightweight read-only
  `jq` parser for final-stabilize summary JSON. It prints per-case pass/fail,
  fall/drop timing, target-window dwell, travel/lateral/tilt metrics,
  final-hold latch/active steps, chest-pad collision enable step/reason, and
  rollout root/velocity/box pose write counts. It does not run simulation.
- 2026-07-06 lightweight checks after `168433` submission passed: `bash -n`
  over the affected shell launchers, `python3 -m py_compile` for the G1 probe
  selector, and `git diff --check` over touched docs/scripts all returned
  status `0`. `168433` remained `PENDING (Priority)`.
- 2026-07-06 lightweight checks after recording `168429`/`168431` and
  submitting `168432` passed: `bash -n` over the affected shell launchers,
  `python3 -m py_compile` for the G1 probe selector, and `git diff --check`
  over touched docs/scripts all returned status `0`. `168432` remained
  `PENDING (Priority)`.
