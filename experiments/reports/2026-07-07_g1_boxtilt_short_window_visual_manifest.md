# G1 Boxtilt Short-Window Visual Manifest

This is current-progress visualization only. It is not a strict carrying
success, not an Isaac camera render, and not new control evidence.

## Source

- Record job: `169488` / `g1_bxshortviz`, completed on `server43`, exit `0:0`.
- Relabel/render fix job: `169501` / `g1_bxvizfix2`, completed on `server39`,
  exit `0:0`.
- Replay CSV:
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_boxtilt_avgpos_short_window_760_replay_record/agile_low_cradle_freebox_walk/core_world_g1_box_scene_replay.csv`
- Strict checker summary:
  `experiments/outputs/core_world_g1_boxtilt_avgpos_short_window/20260707_g1_boxtilt_avgpos_short_window_760_replay_record/boxtilt_avgpos_short_window_summary.json`

## Visuals

- Visual directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/`
- Primary annotated MP4:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/g1_boxtilt_short_window_progress_annotated.mp4`
- Clean MP4:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/g1_boxtilt_short_window_progress.mp4`
- GIF:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/g1_boxtilt_short_window_progress.gif`
- Poster:
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/g1_boxtilt_short_window_progress_poster.png`

## Interpretation

- Scenario: G1, `0.75 kg` free box, boxtilt short-window replay, 760 steps.
- Positive signals: fall/drop `0/0`, root/box rollout writes `0`, final
  robot/box target-directed travel `2.25514/2.25542 m`,
  target-window end streak `133`, final-hold end streak `100`.
- Strict failures: max robot/box tilt `0.62420/0.64870 rad`, final lateral
  error `1.08572/1.28355 m`.
- The final poster was visually checked and correctly labels the result as
  `strict checker: fail`.
