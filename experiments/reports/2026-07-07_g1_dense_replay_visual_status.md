# G1 Dense Replay Visual Status - 2026-07-07

## Result

The current best presentation artifact is:

```text
experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_fallback_dense_replay/g1_lowcarry_runtime_chestpad_fallback_annotated.mp4
```

Companion poster:

```text
experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_fallback_dense_replay/g1_lowcarry_replay_fallback_poster.png
```

This visual is a schematic replay from a recorded Isaac rollout. It is not an
Isaac RGB camera render and is not new control evidence.

## Control Record

Dense replay record stamp:

```text
20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_dense_replay
```

Record path:

```text
experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_dense_replay/agile_low_cradle_freebox_walk/
```

The recorded rollout reproduced the narrow low-front 0.60 kg runtime chest-pad
diagnostic:

- strict checker status: `pass`
- fall/drop: `0/0`
- final robot/box target-directed travel: about `2.051/2.032 m`
- max robot/box tilt: `0.309/0.428 rad`
- target-window end streak: `102`
- rollout root/velocity/box pose writes: `0/0/0`
- runtime chest-pad collision enabled at step `712`

This remains a narrow engineered diagnostic. It is not learned unknown-load
carrying and not arbitrary-posture carrying.

## Isaac RGB Render Blocker

Job `170415` (`g1_showcase`) ran on `server46`. The control record passed, but
the true Isaac replay-render stage produced only:

```text
experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_060_runtime_chestpad_showcase_render_dense_replay/render_debug_trace.json
```

No PNG frames or MP4 were generated. The render log stopped after local Kit
registry dependency failures involving `omni.kit.pip_archive`,
`omni.replicator.core`, `isaacsim.core.rendering_manager`, and viewport-related
dependencies. The job was cancelled after the render path stalled.

Follow-up job `170422` (`g1_extsmk`) added local Isaac Sim extension folders to
the launcher and ran a one-frame true-render smoke on `server44`. It still
produced only:

```text
experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_extfolder_true_render_smoke/render_debug_trace.json
```

The same missing registry dependencies remained:

- `omni.kit.pip_archive` for `omni.replicator.core` / telemetry
- `omni.kit.viewport.window` for `isaacsim.core.rendering_manager`

The ext-folder path alone is therefore insufficient. The next true-render
repair needs the local Kit registry mirror or launcher experience to resolve
those registry packages before more render-product/capture debugging.

## Fallback Visual

Job `170419` (`g1_fallback`) ran on `server02` through tmux and produced:

- `83` PNG frames
- GIF
- poster PNG
- raw MP4
- annotated MP4

Fallback summary status: `pass`

Fallback success claim:

```text
schematic_replay_visual_only_not_isaac_camera_render_not_new_control_evidence
```

## Interpretation

Use the annotated MP4 as the current best quick presentation visual only with
the explicit caveat that it is a schematic replay. The real Isaac RGB render
path remains blocked by Kit/extension registry and render-product/capture
issues, not by the G1 low-carry rollout itself.
