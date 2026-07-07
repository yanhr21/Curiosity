# G1 Runtime Chest-Pad Showcase Status

Date: 2026-07-07

## Current Best Rollout Record

- Source directory:
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_min700/agile_low_cradle_freebox_walk`
- Summary:
  `core_world_g1_box_scene_summary.json`
- Replay CSV:
  `core_world_g1_box_scene_replay.csv`
- Checker:
  `check.json`

Key result:

- Status: `pass`
- Checker failures: `[]`
- Completed steps: `819`
- Fall/drop events: `0 / 0`
- Final robot/box target-directed travel:
  `2.0513995562079526 m / 2.0317395329475403 m`
- Final robot/box lateral error:
  `0.07053927727645369 m / 0.26540452241897583 m`
- Max robot/box tilt:
  `0.3094230784273577 rad / 0.42844048251422273 rad`
- Target-window stable steps/end streak:
  `105 / 102`
- Runtime chest-pad trigger:
  reason `target_window`, collision enabled at step `712`
- Rollout root velocity and box pose writes:
  `0 / 0`

This is the current strongest narrow 0.60 kg G1/AGILE low-carry diagnostic.
It is not learned unknown-load carrying and not a general robustness result.
The chest pad is an engineered runtime support geometry.

## Visualization Available Now

- Fallback visual directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_fallback_min700`
- Annotated MP4:
  `g1_lowcarry_runtime_chestpad_fallback_annotated.mp4`
- Raw MP4:
  `g1_lowcarry_runtime_chestpad_fallback.mp4`
- GIF:
  `g1_lowcarry_replay_fallback.gif`
- Poster:
  `g1_lowcarry_replay_fallback_poster.png`

The fallback visual is a schematic replay from the recorded Isaac rollout. It
shows a G1-like humanoid, the free box, trajectory, and strict metrics. It is
not an Isaac camera render.

## Real Isaac RGB Render Attempt

- Render directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_render_min700`
- Result: `fail`
- Captured frames: `0`
- Failure reason:
  `omni.replicator` and `isaacsim.core.rendering_manager` are unavailable in
  the current Kit environment.

The rollout itself passed; the RGB render failed because the current Kit stack
lacks a working camera capture backend.

## Next Action

Do not rerun the same Replicator/RenderingManager path unchanged. The next
visualization step should either:

1. build a renderer that uses an available Kit viewport/capture API in this
   environment, or
2. generate a higher-quality offline schematic from the dense state CSV while
   retaining the explicit "not Isaac camera render" label.
