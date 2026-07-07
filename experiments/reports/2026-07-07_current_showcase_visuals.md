# 2026-07-07 Current Showcase Visuals

This is a presentation inventory only. It is not a carrying-success claim.

## Recommended Current-Progress Visual

- File:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/g1_boxtilt_short_window_progress_annotated.mp4`
- Poster:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/g1_boxtilt_short_window_progress_poster.png`
- Source rollout:
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_boxtilt_avgpos_short_window_760/agile_low_cradle_freebox_walk/`
- Strict aggregate:
  `experiments/outputs/core_world_g1_boxtilt_avgpos_short_window/20260707_g1_boxtilt_avgpos_short_window_760/boxtilt_avgpos_short_window_summary.json`

Interpretation: current G1 AGILE-policy progress with a `0.75 kg` free box in
the boxtilt/cradle scaffold. It completed the short 760-step window with
fall/drop `0/0`, final robot/box target-directed travel about
`2.255/2.255 m`, and target-window end streak `133` steps. The strict checker
still failed because robot/box tilt was too high (`0.624/0.649 rad`) and
final lateral error was too large (`1.086/1.284 m`). Present this as progress
plus the current bottleneck, not as solved carrying.

Important limitation: this is a schematic replay fallback from recorded CSV,
not an Isaac camera render and not new control evidence.

## Narrow Passing Visual

- File:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_presentation_fallback/g1_lowcarry_replay_fallback_annotated.mp4`

Interpretation: this shows the narrow `0.50 kg` low-carry case that reproduced
a strict pass in the current scaffold. It should be described as a narrow
engineered-condition pass only, because subsequent load and posture gauntlets
failed.

Important limitation: this is also a schematic replay fallback, not an Isaac
camera render.

## True Isaac Camera Render Attempt

- Slurm attempt: job `169542` (`g1_bxrgpu`) ran on `server23` with
  `--gres=gpu:1`.
- Output directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_short_window_isaac_replicator_gpu_server23_try/`
- Log:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_short_window_isaac_replicator_gpu_server23_try/tmux_srun.log`
- Summary:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_short_window_isaac_replicator_gpu_server23_try/g1_replay_render_summary.json`

Result: failed as a render-pipeline issue. AppLauncher started, but
`omni.replicator.core` could not resolve local Kit extension dependencies and
the Python import failed with `ModuleNotFoundError: No module named
'omni.replicator'`. The viewport fallback also lacked
`isaacsim.core.rendering_manager`. Captured frames: `0`. Do not advertise a
true Isaac camera render as available until the local Kit extension set is
fixed or a different installed render path is found.
