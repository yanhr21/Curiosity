# Phase 00 Core Data Asset Generation Plan

Phase 00 is the new starting point for data and asset generation. All earlier
phase-numbered work is legacy evidence only and must not be treated as the
active data-generation sequence.

## Status

Status: Phase 00 asset generation complete; not a training result.

The earlier static catalog/SVG/browser files are only a lightweight preflight.
They are not enough to count as asset generation. A valid Phase 00 result must
be generated and verified by Newton inside a Curiosity-owned tmux-held H200
allocation. No simulation, rendering, validation builder, dataset conversion,
training, or NumPy/PyTorch-heavy asset verification may run on the login node.

## Source-Backed Basis

The implementation must follow the source-backed constraints recorded in:

- `docs/phase00_newton_asset_generation_references.md`

The key consequences are:

- use Newton/OpenUSD-compatible assets and avoid ad-hoc fake fields;
- validate visual outputs with real Newton camera/rendering paths;
- validate contact outputs with Newton contact/contact-sensor paths;
- check mass/inertia and collision/contact warnings;
- record full videos or dense-frame equivalents;
- keep all real generation/validation inside tmux-held H200 `srun`/`salloc`.
- refuse short full-generation runs; Phase 00 full asset generation now
  requires at least 1800 simulated steps per catalog cell by default.

## Purpose

Build the first serious core asset family for curiosity-driven manipulation:
a constrained tabletop arena with one target object per episode, systematic
physical variation, balanced visual/contact evidence, and frozen held-out
cells.

The scene should stay bounded rather than becoming a large open world, because
the current research question is whether curiosity over object-motion and
contact prediction improves grasp, lift, hold, slip recovery, and
stabilization. A huge scene would entangle navigation, clutter, perception
failure, sparse contact, and controller limits. The tabletop arena keeps the
failure modes diagnosable while still exposing hard physical variation.

## Active Scene Contract

- Arena: one tabletop workspace in front of the official Newton Panda hydro
  grasp/lift prior.
- Per episode: one primary target object; optional single distractor is a
  later robustness extension, not part of the first gate.
- Target behavior: approach, grasp, lift, hold, detect contact loss/slip, and
  recover or stabilize.
- Core object families:
  - official cup-like Newton asset;
  - procedural or official box primitive;
  - procedural or official cylinder primitive.
- Variation axes:
  - object family;
  - mass and inertia;
  - friction/contact response;
  - pose/yaw;
  - visual cue visibility and correctness;
  - vision/contact mask schedule.
- Frozen splits:
  - train cells for source generation;
  - validation cells for design repair and threshold checks;
  - held-out cells for later evaluation only.

Held-out cells must not be used for training, label construction, threshold
tuning, hyperparameter selection, controller repair, or manual source
selection.

## H200 Execution Contract

All real Phase 00 generation/validation must use:

- a Curiosity-specific tmux session;
- a persistent Slurm allocation on an H200 GPU node;
- `srun --jobid=<held_job>` from inside the tmux-held allocation;
- prebuilt local shared-filesystem environments only;
- logs under `logs/newton/`;
- outputs under `experiments/outputs/`;
- visual outputs under a phase-specific visual directory such as
  `experiments/visuals/phase00/`;
- reports under `experiments/reports/`.

The launcher must refuse to start if:

- no tmux session is present;
- no Slurm job ID is provided;
- the job is not visible in `squeue`;
- the allocated node/GPU evidence does not show H200;
- the local Newton venv is missing;
- required cached Newton assets are missing;
- a similar Phase 00 asset generation run is already active.

## Full Generation Horizon

The earlier `phase00_core_asset_generation_h200_20260629_175727` run used
450 steps per generated cell. That run is useful H200 pipeline evidence but is
too short to serve as the final full asset-generation basis.

The active full-generation profile is:

- minimum/default simulated steps per catalog cell: 1800;
- pre-record warmup: 60 steps;
- final scripted hold duration: 12 seconds;
- minimum accepted hold duration: 8 seconds;
- dense rollout evidence: video frame stride 3 at 20 FPS, plus sampled contact
  sheet/browser frames;
- any shorter run must be explicitly labeled diagnostic/smoke-test and cannot
  satisfy Phase 00 completion.

## Required Generation Evidence

For every catalog cell, the H200 run must produce:

- `*_fresh_newton_sensor_contact_sanity.json`;
- per-cell `*_summary.json`;
- per-cell `*_visual_validation.json`;
- per-cell `*_manual_visual_inspection.json`;
- per-cell contact/contact-proxy metrics;
- per-cell frame browser;
- per-cell contact sheet;
- per-cell full rollout video or dense-frame equivalent;
- per-cell log containing host, Slurm job ID, GPU evidence, env path, and
  exact command;
- aggregate Phase 00 report listing pass/fail/blocker status.

## Asset Quality Gates

An asset cell passes only if:

- official Newton sanity passes in the same allocation before cell rendering;
- video/frame outputs are nonblank and show the intended object;
- object family, mass, friction, pose, visual cue, and mask mode match the
  catalog;
- contact/contact-proxy signals are present when contact is expected;
- contact-only masked-vision cells actually mask vision in the exported data;
- no contact-buffer overflow or dropped-contact warning is present;
- mass/inertia warnings are absent or explicitly documented and accepted;
- lift/hold metrics can be computed;
- manual visual inspection passes.

If a cell fails, record it as an asset blocker or repair item. Do not proceed
to curiosity training on that cell.

## Curiosity Relevance

The asset family must expose physical properties that vision alone cannot
settle:

- visually similar cups with different mass/friction;
- hidden or misleading fill cues;
- cylinders that slip or roll after contact;
- boxes with offset mass that alter lift response;
- contact-only and alternating masks that force the contact/tactile stream to
  remain useful.

Curiosity should later use bounded learning progress and useful prediction
improvement over object motion/contact/tactile proxies, with safety penalties
for excessive force, drop, high acceleration, contact loss, or no-op
exploration. Phase 00 prepares the asset/data evidence for that later training;
it does not claim curiosity success.

## Completion Gate For Phase 00

Phase 00 is complete only when current evidence proves all of the following:

- the active plan/todo live directly under `PLAN/` and `TODO/`;
- Newton source references are recorded;
- the catalog declares object families, splits, masks, and no-leakage rules;
- tmux-held H200 launcher/run scripts exist;
- full-generation H200 run uses the active long-horizon profile, not the old
  450-step short profile;
- a real H200 run generated and validated every train/validation/held-out
  catalog cell;
- every required visual/contact/metric/manual-inspection artifact exists;
- aggregate report classifies each cell as pass/fail/blocker;
- no login-node simulation/rendering/training was used.

The current evidence below satisfies this Phase 00 asset-generation gate.

## Current H200 Evidence

Short-horizon run `phase00_core_asset_generation_h200_20260629_175727` is
historical pipeline evidence only. It is superseded by the long-horizon H200
evidence below.

The completed long-horizon evidence uses Slurm job `157630`, host `server29`,
GPU `NVIDIA H200`, and the Curiosity tmux-held allocation workflow. The primary
long run `phase00_core_asset_generation_h200_long_20260629_182052` generated
the first 4 catalog cells, then exited `127`; the filtered repair run
`phase00_core_asset_generation_h200_long_repair2_20260629_183216` generated the
remaining 11 cells after COM-offset glue was added.

Combined evidence:

- 15/15 catalog cells have long-horizon generated rows;
- future Phase 00 visual outputs must not be written flat under
  `experiments/visuals/`; the active path convention is
  `experiments/visuals/phase00/<run_tag>_<split>_<cell>/`;
- every generated summary reports `num_steps=1800`, rollout GIF
  `frame_count=601`, and video status `pass`;
- real MP4 video visualization was generated in a separate Curiosity-owned
  tmux-held H200 Slurm allocation, not on the login node:
  `phase00_video_mp4_export_h200_20260629_203527`;
- the MP4 export used Slurm job `157730`, host `server53`, GPU `NVIDIA H200`,
  and existing prebuilt environment `envs/trex_dataset/.venv` for OpenCV video
  encoding;
- 15/15 catalog cells have `rollout_video.mp4` with 601 encoded frames at
  20 FPS; aggregate report:
  `experiments/reports/phase00_video_mp4_export_h200_20260629_203527_phase00_video_mp4.md`;
- every generated cell has summary, metrics, visual validation, frame browser,
  contact sheet, and rollout GIF artifacts;
- the three COM-offset cells have matching requested, updated, and observed COM
  offsets on x: `0.014`, `0.012`, and `0.018` m;
- manual contact-sheet inspection passed for 15/15 cells:
  `experiments/reports/phase00_core_asset_generation_h200_long_combined_manual_visual_inspection.md`;
- aggregate/launch status:
  `experiments/reports/phase00_core_asset_generation_h200_launch_status.md`.

Known limitation: box and cylinder families currently use available official
Newton proxy objects rather than custom authored USD assets. This is recorded
as an asset-family limitation, not as a missing Phase 00 generation blocker.

Phase 00 asset generation and MP4 visualization export are complete. This is
not a curiosity-training result and must not be described as a policy
improvement or curiosity success claim.
