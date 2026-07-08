# Showcase Status

Date: 2026-07-06

This report separates presentable material from final-success claims. The
project has not yet completed the full goal: a balanced walking robot that can
carry the box in every selected posture under the strict gates.

## 2026-07-07 Update

- The only currently usable visual fallback remains the browser/SVG schematic
  under
  `experiments/visuals/g1_progress_showcase/20260706_g1_lowcarry_168398_browser_showcase/`.
  It is better than the old blocky scaffold MP4s for explaining progress, but
  it is still schematic, not an Isaac camera render.
- The replay source rollout `168632` is still the valid source for a real G1
  replay visual: fall/drop `0/0`, replay CSV present, no rollout root pose,
  root velocity, or box pose shortcut writes.
- Main Isaac replay render `168801` failed because
  `omni.kit.viewport.utility` is unavailable in the headless environment.
  Direct fallback `168895` then failed because `omni.replicator.core` was
  imported before its extension was enabled. Both produced zero frames and
  must not be used.
- The renderer now explicitly enables `omni.replicator.core` before import.
  Extension-enabled fallback render `168900` (`g1_viz_fb_ext`) is queued in
  the GPU partition for
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_ext_960x540/`.
  It becomes presentable only if `g1_replay_showcase_check.json` passes and
  frames/MP4 are present.
- Contact rescue job `168896` completed and is negative. Terminal chest-pad
  no-lateral still fell, tiny-pad variants worsened drops/falls, and late
  tiny-pad failed before activation. Do not present the chest-pad rescue
  branch as progress.

## Strongest Current Evidence

- `168398`: strongest direct-G1 low-carry target-hold evidence so far.
  - Direct G1 scene with AGILE policy, free box, low-carry setup.
  - Passed its declared 819-step target-window hold gate.
  - Output:
    `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_819_targetnegx1/agile_low_cradle_freebox_walk/`.
  - Key metrics: 819/819 completed steps, fall events 0, box-drop events 0,
    final robot/box target-directed travel about 2.299 m / 2.346 m, final
    relative box-robot offset error about 0.080 m, max robot/box tilt about
    0.209 rad / 0.414 rad.
  - Rollout root pose writes, rollout root velocity writes, and box pose writes
    were zero.
  - Limitation: one posture and one narrow mass/geometry setting. Held-out
    light/heavy variants failed, so this is not general carrying success.

- `168431`: strongest chest-pad near-pass.
  - Fall/drop stayed at zero and target-window behavior improved.
  - Limitation: final box lateral error still exceeded the strict gate. This is
    not a verified second posture.

- `168479`: best current light-box low-carry failure for diagnosis.
  - Box drops were zero and forward progress recovered.
  - Limitation: late roll instability, lateral drift, and box lag still fail
    strict gates.

## Visual Assets

Immediate presentation fallback:

- `experiments/visuals/g1_progress_showcase/20260706_g1_lowcarry_168398_browser_showcase/index.html`
  is a browser-only schematic generated from sampled `168398` rollout CSV
  states. It shows a G1-like humanoid, the low-carry box, side-view posture,
  top-view path, and key metrics. It is acceptable as an immediate progress
  visualization while the true Isaac replay render waits for compute resources.
  It is not a true Isaac camera render and is not new control evidence.
- `experiments/visuals/g1_progress_showcase/20260706_g1_lowcarry_168398_browser_showcase/g1_lowcarry_168398_progress_poster.svg`
  is the matching static poster for slides or quick inspection. It has the
  same limitation: schematic only, not a camera render.

The existing MP4 files below are not suitable as the main presentation visual.
They are abstract scaffold/debug videos, not G1 humanoid walking-carrying
renders. Use them only in an internal appendix if explaining how the task
scaffold evolved.

- `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/20260706_direct_carry_posture_stress_suite_64cm_8kg_front_reach.mp4`
- `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/20260706_direct_carry_posture_stress_suite_64cm_8kg_chest_high.mp4`
- `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/20260706_direct_carry_posture_stress_suite_64cm_8kg_close_mid.mp4`
- `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/20260706_direct_carry_posture_stress_suite_64cm_8kg_low_front.mp4`
- `experiments/visuals/direct_carry_posture_suite/20260706_direct_carry_posture_stress_suite_64cm_8kg/20260706_direct_carry_posture_stress_suite_64cm_8kg_front_mid.mp4`
- `experiments/visuals/prismatic_carrier_stand/20260705_prismatic_cradle_sync_inchworm_default_repro_retry9_metrics.mp4`

## Wording Boundary

Acceptable:

- "We have a direct Isaac/G1 carrying scaffold with one narrow low-carry
  target-hold pass."
- "We have multiple posture and contact diagnostics showing where stability
  and retention fail."
- "The current active blocker is late balance and retention under posture and
  load variation."

Not acceptable:

- "The robot can generally carry boxes."
- "All carrying postures are stable."
- "Unknown-load carrying is solved."
- "Video-guided RL has been demonstrated."

## Needed Visual

The next presentation-quality visual should be an Isaac-rendered G1 scene:

- visible G1 humanoid robot, not proxy blocks;
- visible free box and cradle/contact geometry;
- side or three-quarter camera view, not top-down-only;
- overlay or caption stating whether the clip is `pass`, `near-pass`, or
  `failure diagnostic`;
- no claim of final success unless the associated checker passes the strict
  gates.

## Active Pending Result

- `168482`, `g1_lc_rollpos`, completed on `server39`. Its result should be
  recorded separately in the low-carry lateral/roll decision note before being
  used for any next controller choice.
- `168509`, `g1_show_rgb`, ran on `server39` and is negative for showcase. It
  produced summary/log files but no RGB frames because the current Isaac
  environment lacks `omni.replicator.core`
  (`capture_rgb_error = ModuleNotFoundError: No module named 'omni.replicator'`).
  The render-enabled rerun also failed control gates with early fall/drop, so
  it must not be used as pass evidence or as the main visual.
- `168561`, `g1_show_look`, was cancelled before running after it became clear
  that the cpu-partition render path was not the right primary route.
- `168580`, `g1_rec_short`, is the active replacement path. It records the
  same low-carry configuration without rendering, with `SHOWCASE_CAPTURE_RGB=0`
  and `SHOWCASE_RECORD_REPLAY=1`, to produce
  `core_world_g1_box_scene_replay.csv`. If that rollout reproduces the pass,
  render it through
  `scripts/isaac/run_core_world_g1_replay_showcase_render.sh` as a real G1+box
  replay visualization. As of 2026-07-06 22:12 CST it is still pending in the
  `cpu` partition and Slurm currently schedules it around 23:15 CST on
  `server02`.
- `168580` later ran on `server39` and is negative. It did not run the intended
  replay-record path: env snapshot had `CAPTURE_RGB=1` and
  `record_replay_csv=false`, with no `RECORD_REPLAY_CSV=1`. The rollout failed
  with fall/drop `720/617`, so it cannot be used as pass evidence or replay
  source.
- `168632`, `g1_rec_retry`, is the corrected replay-record retry. It was
  submitted with explicit `SHOWCASE_CAPTURE_RGB=0`,
  `SHOWCASE_RECORD_REPLAY=1`, `CAPTURE_RGB=0`, `RECORD_REPLAY_CSV=1`, and
  `RECORD_REPLAY_EVERY_N_STEPS=10`. Its strict render watcher is
  `curiosity_g1_replay_render_retry2_waiter_0706`; render will only submit if
  the record summary passes and replay CSV exists.
- `168632` passed and produced a replay CSV. Metrics: 819/819 steps,
  fall/drop `0/0`, final robot/box target-directed travel `2.2988/2.3465 m`,
  final relative error `0.0796 m`, max robot/box tilt `0.2086/0.4136 rad`, no
  rollout root pose/root velocity/box pose writes, and
  `core_world_g1_box_scene_replay.csv` with 84 lines. The first strict watcher
  attempt exposed and fixed a zero-count bug in
  `scripts/isaac/wait_and_submit_g1_replay_render.sh`. Replay render job
  `168658`, `g1_replay_viz2`, is now queued for
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2/`.
- `168658` was cancelled while still pending after Slurm pushed it to
  2026-07-07 00:04 CST. A shorter quick render `168664`, `g1_viz_quick`, was
  submitted instead with `CAPTURE_EVERY_N_ROWS=3`, `MAX_FRAMES=24`, and output
  directory
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2_quick/`.
  This is still replay visualization only, not new control evidence.
- `168664` was also cancelled while pending after Slurm pushed it to the same
  2026-07-07 00:04 CST slot. Current quick render is `168669`, `g1_viz_q2`,
  with `CAPTURE_EVERY_N_ROWS=4`, `MAX_FRAMES=18`, and output directory
  `experiments/visuals/g1_replay_showcase/20260706_g1_lowcarry_168398_replay_render_retry2_quick2/`.
- `168669` was cancelled while still pending and before producing artifacts
  because it remained stuck in the `cpu` partition. Replacement render job
  `168801`, `g1_viz_gpu_q3`, tmux session
  `curiosity_g1_replay_gpu_render_0707`, is queued in the `gpu` partition.
  It targets
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3/`
  with `CAPTURE_EVERY_N_ROWS=4`, `MAX_FRAMES=18`, and the updated renderer
  that adds path/start/end/target markers. As of 2026-07-07 00:18 CST it is
  still pending and scheduled for 03:45:50 CST. No Isaac render frames exist
  yet.

## Contact Follow-Up

- `168627`, `g1_contact_next`, ran `chestpad_hold_contact` after the failed
  `168580` summary. It is useful negative/partial evidence: fall/drop stayed
  `0/0`, no shortcut writes occurred, chest-pad support enabled on hold, and
  max robot/box tilt stayed `0.2747/0.2733 rad`. But it failed strict carrying:
  target-window stable steps were `0`, final robot/box target-directed travel
  was only `0.7175/0.6576 m`, and final-hold active steps were `15 < 399`.
  Chest-pad hold improves retention but suppresses progress to the 2.0 m target.
- The delayed chest-pad follow-up is now `168802`, `g1_contact_gpu`, tmux
  session `curiosity_g1_contact_next_gpu_0707`, queued in the `gpu` partition
  with `CASE_SET=contact_next` and prefix `20260707_gpu_contact_next`.
  `scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh` now
  accepts `FOLLOWUP_PREFIX` as a fallback for `BASE_STAMP_PREFIX`, so the
  expected output case is
  `20260707_gpu_contact_next_chestpad_terminal_contact`. As of 2026-07-07
  00:18 CST it is pending and scheduled for 06:11:58 CST.

## Posture Coverage

- Added `scripts/isaac/run_core_world_g1_posture_gauntlet.sh` as the next
  strict validation path after the delayed chest-pad result. It runs
  `lowcarry_base`, `chestpad_terminal`, `boxtilt_diagnostic`,
  `lowcarry_lightbox`, and `lowcarry_heavybox`, then writes
  `g1_posture_gauntlet_summary.json`. This is intended to expose the real
  remaining gap to "any carrying posture" and load variation; failed cases are
  expected and must be recorded as negative evidence rather than hidden.
- The gauntlet is queued indirectly by tmux watcher
  `curiosity_g1_posture_gauntlet_after_contact_0707`, which waits for `168802`
  before submitting a compute-node `srun`. No gauntlet result exists yet.

## Render Quality Gate

- `scripts/isaac/check_core_world_g1_replay_showcase.py` now checks sampled
  PNG frame headers and dimensions, not only frame count. The render launcher
  passes the configured width/height to this checker. A valid showcase render
  must pass source-rollout gates, render-summary gates, frame-count gates, and
  sampled PNG size/dimension gates before being used as the main visual.

## Contact Comparison

- Added `scripts/isaac/summarize_core_world_g1_contact_followup.py` to compare
  baseline low-carry, hold-contact, and terminal-contact runs from existing
  summary/check JSON files. The current pending report is
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_pending.json`:
  baseline low-carry passes, hold-contact fails the progress/final-hold gates,
  and terminal-contact is still missing because `168802` has not run.
- Tmux watcher `curiosity_g1_contact_compare_after_168802_0707` waits for
  `168802` to leave the queue, then writes
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json`.

## Completion Gate

- Added `scripts/isaac/audit_g1_carry_completion.py` as a machine-readable
  full-goal gate. Current report:
  `experiments/reports/2026-07-07_g1_carry_completion_audit_current.json`.
  It is `fail`: baseline low-carry passes, terminal-contact is missing, and
  the posture/load gauntlet summary is missing. This is the expected result
  while `168802` and the gauntlet have not run.
- Tmux watcher `curiosity_g1_completion_audit_after_gauntlet_0707` waits for
  the posture gauntlet watcher to finish, then writes
  `experiments/reports/2026-07-07_g1_carry_completion_audit_after_gauntlet.json`.

## Next Action

- Added `scripts/isaac/recommend_g1_next_carry_actions.py`. Current report:
  `experiments/reports/2026-07-07_g1_next_carry_actions_current.json`. It
  recommends waiting for `168802` terminal-contact and the posture/load
  gauntlet rather than submitting duplicates.
- Tmux watcher `curiosity_g1_next_actions_after_audit_0707` waits for the
  completion audit watcher, then writes
  `experiments/reports/2026-07-07_g1_next_carry_actions_after_audit.json`.
- Queue update at 2026-07-07 00:42 CST: `168801` and `168802` are still
  pending but moved earlier to 02:10:58 CST on `server39`. No new render,
  contact, gauntlet, or after-audit recommendation artifact exists yet.
- Queue/watch update at 2026-07-07 00:57 CST: `168801` and `168802` remain
  `PENDING (Priority)`, and
  `logs/core_world_g1_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3_srun.log`
  still only shows `queued and waiting for resources`. Added contingency tmux
  watcher `curiosity_g1_render_fallback_after_168801_0707`; it waits for
  `168801` and only submits a short 960x540 fallback render if the main render
  has no usable PNG frames/check output.
- Queue update at 2026-07-07 01:06 CST: both jobs are still
  `PENDING (Priority)`. Slurm's predicted start time slipped to 02:20:50 CST
  on `server39`. No render frames, contact summaries, gauntlet output, or
  after-audit reports exist yet.
- Added `scripts/isaac/write_g1_showcase_visual_manifest.py` and current
  manifest `experiments/reports/2026-07-07_g1_showcase_visual_manifest.md`.
  The manifest is currently `pending_or_failed`: the source rollout passes,
  but no real Isaac replay PNG/MP4 exists yet. Tmux watcher
  `curiosity_g1_showcase_manifest_after_render_0707` waits for the main and
  fallback render paths before rewriting it.
- Fixed the pending terminal-contact report path in
  `scripts/isaac/summarize_core_world_g1_contact_followup.py`: missing
  low-cradle cases now report the standard
  `<stamp>/agile_low_cradle_freebox_walk/` directory. Refreshed the current
  contact comparison, completion audit, and next-action reports with the
  corrected `168802` expected path.
- Queue update at 2026-07-07 01:16 CST: `168801` and `168802` remain
  `PENDING (Priority)` with predicted start 02:20:50 CST, no assigned nodes,
  and no render/contact/gauntlet artifacts yet.
- The active pipeline status collector now includes a Slurm snapshot for
  `168801` and `168802`, and the Markdown status page prints it under
  `Slurm Jobs`. The current report shows both jobs pending for priority with
  start `2026-07-07T02:20:50`; missing render/contact artifacts are therefore
  still queue-waiting artifacts, not completed-run failures.
- Queue update at 2026-07-07 01:24 CST: `168801` and `168802` still have no
  assigned node and remain pending for priority. No new render frames, MP4,
  terminal-contact summary, or gauntlet output exists. Refreshed the current
  pipeline JSON/Markdown status reports after this poll.
- Added periodic status watcher
  `curiosity_g1_periodic_status_until_168801_168802_done_0707`. It refreshes
  the current active pipeline status/failure/Markdown reports every 10 minutes
  while `168801` or `168802` remains in `squeue`, then performs one final
  refresh after both leave the queue. It is status-only and does not run
  Isaac or simulation.
- Main render `168801` failed on `server59` because
  `omni.kit.viewport.utility` is missing in the current headless Isaac
  environment. No PNG/MP4 was produced. The render script now uses a USD
  Camera prim plus `omni.replicator.core` RGB annotator and PIL PNG writing;
  fallback render `168849` is pending and should test this patched path.
- Terminal-contact `168802` completed but failed strict carrying gates:
  104 fall events, first fall step 715, final box target-directed travel
  1.88795 m, final relative error 0.34432 m, and zero target-window/final-hold
  streak. It is negative evidence, not carrying success. Contact comparison:
  `experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json`.
- Submitted targeted contact rescue job `168851` with three cases:
  terminal no-lateral, terminal tiny pad, and terminal late tiny pad. It will
  write `experiments/reports/2026-07-07_g1_contact_rescue_comparison_after_run.json`.
- Path correction at 2026-07-07 02:00 CST: fallback render `168849` and
  contact rescue `168851` both failed immediately with exit `127` because the
  `srun bash -lc` command used relative `scripts/...` paths on the compute
  shell. Resubmitted absolute-path replacements: `168882`
  (`g1_viz_gpu_fb_abs`) for fallback render and `168883`
  (`g1_contact_rescue_abs`) for contact rescue. The visual manifest now tracks
  the absolute-path fallback render directory.
- Second path correction at 2026-07-07 02:06 CST: `168882` and `168883` also
  failed immediately with exit `127` because nested shell expansion produced
  `/scripts/...`. Resubmitted direct-path jobs without nested `bash -lc`:
  `168895` (`g1_viz_gpu_fb_direct`) and `168896`
  (`g1_contact_rescue_direct`). The status collector and visual manifest now
  track these direct-path outputs.

## Pipeline Status

- Added `scripts/isaac/collect_g1_active_pipeline_status.py`. Current report:
  `experiments/reports/2026-07-07_g1_active_pipeline_status_current.json`.
  It is `incomplete`: render summary/check, terminal-contact summary/check,
  after-contact comparison, gauntlet summary, after-gauntlet completion audit,
  and after-audit next-action artifacts are still missing.
- Tmux watcher `curiosity_g1_pipeline_status_after_watchers_0707` waits for
  the next-action watcher, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_status_after_watchers.json`.
- Tmux watcher `curiosity_g1_render_status_after_168801_0707` waits for the
  render job `168801`, then writes
  `experiments/reports/2026-07-07_g1_render_pipeline_status_after_168801.json`
  so showcase render success/failure is captured even before the later
  contact/gauntlet chain completes.

## Failure Classification

- Added `scripts/isaac/classify_g1_active_pipeline_failures.py`. Current
  report:
  `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_current.json`.
  It classifies the active render/contact logs as `queued` and expected
  render/contact/gauntlet artifacts as `missing_artifact`.
- Tmux watcher `curiosity_g1_failure_class_after_watchers_0707` waits for the
  final pipeline status watcher, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_after_watchers.json`.

## Markdown Report

- Added `scripts/isaac/write_g1_active_pipeline_markdown_report.py`. Current
  readable status page:
  `experiments/reports/2026-07-07_g1_active_pipeline_status_current.md`.
- Tmux watcher `curiosity_g1_markdown_report_after_watchers_0707` waits for
  the final failure-classification watcher, then writes
  `experiments/reports/2026-07-07_g1_active_pipeline_status_after_watchers.md`.
