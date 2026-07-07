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

## Robot-Like MuJoCo Visual

- File:
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback.gif`
- Poster:
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback_poster.png`
- Source rollout:
  `experiments/outputs/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115/`

Interpretation: clearer robot-like visualization than the early block
scaffolds. It shows a quadruped-style body and legs carrying a welded payload
in MuJoCo with a diagnostic controller. Use it to explain the direction toward
robot embodiment and balance diagnostics.

Important limitation: this is a welded-payload diagnostic, not unknown
free-box carrying. The later free-box MuJoCo hand-controller line, including
the 2026-07-07 hold-capture suite, failed strict free-box fall/drop/tilt/height
gates, so this visual should not be presented as solved contact carrying.

## Prismatic Isaac Scaffold Visual

- File:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback.gif`
- Poster:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback_poster.png`
- One-page wrapper:
  `slides/2026-07-07_isaac_carry_showcase.html`

Interpretation: best schematic for the Isaac task scaffold: target, carrier,
box, posture/probe labels, and progress metrics. Use it only to explain task
structure and measurement gates.

Important limitation: this is a prismatic scaffold visualization, not a
humanoid, not learned locomotion, and not final carrying success.

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

## Latest Negative Diagnostics To Mention Honestly

- G1 boxtilt scaled-terminal and target-window hold diagnostics both failed:
  they can enter the target window transiently but cannot hold without later
  fall/drop or over-travel.
- MuJoCo free-box hold-capture diagnostic failed all six cases despite active
  target-stop and capture-point terms: no root/box pose or velocity shortcuts,
  but every case still had post-latch falls/drops and excessive tilt.

Presentation implication: the best honest message is "we have a narrow G1
low-carry pass, a boxtilt progress replay showing the hard failure mode, and
clear scaffold/robot-like visuals; robust unknown-load long-hold carrying is
not solved yet."
