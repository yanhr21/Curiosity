# Workspace Experiment Curation Record — 2026-08-10

## Scope

The active workspace was reduced around one representation foundation and one
training question: reusable native IsaacLab tactile with synchronized CarryBox
visualization, followed by a matched serious-SUGAR tactile-versus-zero test.
Historical work on demo following, original ICM/Curiosity, and prior tactile
training is capped at five core experiment packages in total.

No experiment data was deleted. Superseded material was moved into the one
approved archive tree:

`/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260810_native_tactile_reset/`

## Five retained historical experiment packages

1. Demo following: matched correct/wrong official-demo V3 training, frozen
   evaluation, and its final visualization.
2. Demo following: fixed unrelated KickBox teacher through update 1216, with
   only the final frozen evaluation and visualization retained.
3. Original ICM: the concrete-semantics cross-seed experiment, including its
   declared fresh-noise control.
4. Original ICM: the matched policy-credit experiment and frozen evaluation.
5. Tactile effect on training: the corrected official Tactile Genesis Stage-2
   package, pruned to final training, evaluation, handoff, spatial-layout, and
   CarryBox presentation evidence.

The official SUGAR assets, curated baseline/checkpoint outputs, serious frozen
predictor, official TinyMDM prior, and rendering runtime are dependencies of
these five packages, not additional experiment claims.

## Material moved out of the active experiment tree

- all report-generation output and the superseded CHORD reproduction tree;
- older demo-reward runs, intermediate checkpoints/evaluations, duplicate
  renders, and top-level logs not belonging to the two retained demo packages;
- older whole-hand TacSL/Plan-10 version ladders and Tactile Genesis runtime
  copies outside the one retained final package;
- older ICM diagnostics, superseded tactile-genesis runs, mass/contact-velocity
  starts, and unrelated top-level logs;
- native-tactile Vulkan startup failures. These remain diagnostic evidence and
  are not counted as tactile experiments or scientific outcomes.

The archive preserves the original subtree names beneath `sugar_demo_reward/`,
`sugar_reproduction/`, `sugar_smp_exploration/`,
`native_tactile_startup_failures/`, and `top_level/` so provenance remains
recoverable.

## Active experiment layout after curation

- `experiments/native_tactile_representation/`: active Plan-12 work only;
- `experiments/native_tactile_training/`: active Plan-13 continuous-start
  tactile-versus-zero work only;
- `experiments/sugar_demo_reward/`: the two retained demo packages and their
  direct dependencies/evidence;
- `experiments/sugar_smp_exploration/`: the two retained ICM packages, the one
  retained Tactile Genesis package, and their direct prior;
- `experiments/sugar_reproduction/`: curated official baseline/checkpoint
  outputs plus assets and rendering support.

Plan and TODO queues 04--11 were moved under `PLAN/legacy/` and `TODO/legacy/`.
Plan/TODO 12 is the completed representation foundation; Plan/TODO 13 is the
only active training/fusion execution queue.

## Second-pass content pruning

The first pass reduced the visible top-level directories but left old work
inside `sugar_reproduction/outputs/final/` and retained intermediate training
endpoints. A second pass therefore moved another `2.7 GB` to:

`/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260810_native_tactile_reset/historical_five_package_pruning_v2/`

After the second pass:

- `sugar_reproduction/outputs/final/` contains only `official_sugar/`; old
  demo, RGB, ICM, SMP-strategy, tactile, physics, and report results are in the
  archive;
- the correct/wrong V3 demo package contains only the two final arms; zero-demo
  and update-1/update-128 checkpoints are archived;
- the unrelated KickBox package contains only the update-1216 endpoint in the
  active tree; update 1 and endpoints 128--1152 are archived;
- the serious predictor retains only `validation_best.pt`; epoch snapshots are
  archived;
- the ICM policy-credit package retains final policies, proofs, final frozen
  traces, and videos; update-1/update-128 checkpoints and redundant console
  logs are archived;
- official Tactile Genesis retains the final `model_5999.pt`, final evaluation,
  handoff, audits, and videos; intermediate model-0/1000/2000/3000/4000/5000
  snapshots are archived.

Active sizes after the historical-experiment pass were approximately `395 MB`
for demo following, `476 MB` for the retained ICM/Tactile-Genesis packages,
and `504 MB` for official SUGAR support. The separate active native-tactile
package later grew to `1.4 GB` because its canonical successful CarryBox trace
now preserves all `660` control frames, `2640` physics substeps, 54 raw taxel
fields, and both optical streams. This is one raw success record rather than a
version ladder. Superseded optical iterations were moved into the same archive
tree.

The later server56 minimal `TiledCamera` controls (one cuboid, one 64 x 64 RGB
camera, no TacSL, no SDF) exited 139 after `ERROR_DEVICE_LOST`. The second run
started from a clean GPU with no other Isaac process and 0 MiB usage. Their
outputs and exact runtime records were moved under
`native_tactile_startup_failures/`. A separate full 320-frame force-only rerun
proved that the hardened collector exits 0 and leaves no stale GPU process; it
was moved under `native_tactile_runtime_controls/`. These records preserve the
renderer-wide diagnosis and runtime fix without polluting the active
native-tactile experiment root.

## CarryBox canonicalization — 2026-08-11

The successful CarryBox result now has one canonical directory:

`experiments/native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/`

It contains one `660`-frame raw trace with the complete `5 ms` substep record,
one world video, one complete bilateral anatomical video, one complete
force/kinematics/friction video, and the left/right/optical supplements. All
five presentation videos display source frames `230:660` and independently
fully decode `430/430` frames.

The former success trace without physical substeps and the two obsolete
control-sampled force videos were moved, not copied, to:

`/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260810_native_tactile_reset/carrybox_canonicalization_20260811/`

The first failed attempt to regenerate the complete main video is retained in
that archive as a runtime failure record; the second attempt completed normally.
The six successful collector/render process records are grouped under
`experiments/native_tactile_representation/runtime/carrybox_complete_20260811/`
instead of being scattered at the runtime root.
The current route is documented in `whole_hand_carrybox_v3/REPRODUCE.md` and is
implemented by the single executable
`scripts/sugar/native_tactile/run_complete_carrybox_visualization.sh`.

On 2026-08-11 the active native-tactile script directory was reduced to the
CarryBox collector, its three renderers, the complete-bundle validator, shared
representation/audits, and the matched policy-fusion tools. Eleven superseded
detached/single-R15 and Vulkan-startup diagnostic scripts were moved to:

`/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260810_native_tactile_reset/scripts/native_tactile_r15_diagnostics_20260811/`

The active code map is now `scripts/sugar/native_tactile/README.md`; it names
one complete CarryBox visualization command rather than a version ladder.

The complete entry point was then executed again from scratch in retained job
`231928`. The run newly collected all 660 control frames/2640 physics substeps,
rendered all five successful CarryBox videos, and fully decoded each output
before exiting zero. Its lightweight result and runtime records remain under
`experiments/native_tactile_representation/runtime/`. The duplicate 707 MB
trace and rendered files were moved after comparison to
`carrybox_fresh_reproduction_20260811/successful_grasp` under the same single
archive root; the original canonical active bundle remains unchanged.

The later server56 live-training preflight ended during IsaacLab scene startup
with Vulkan `ERROR_DEVICE_LOST`, before producing a model or policy result. Its
two runtime records were moved out of active training to
`native_tactile_startup_failures/server56_device_lost_20260811/` under the same
archive root. The passing model-only warm-start audit remains active because it
is the current structural fusion gate.

The subsequent five-day allocation on server38 showed the same Vulkan device
loss for two-environment and one-environment training launches. A direct launch
of the already successful canonical CarryBox collector also exited at scene
startup, establishing that these launches did not test PPO or tactile fusion.
Their runtime files were moved to dated children of
`native_tactile_startup_failures/`; no partial output or empty experiment
directory remains in active `native_tactile_training/`.

## Native tactile training package closure — 2026-08-11

The active Plan-13 tree now has one readable route: `README.md` states the
scientific result, `REPRODUCE.md` gives the serial training/evaluation/video
commands, and `MANIFEST.json` inventories the two final checkpoints, the
camera-free numerical evidence, and the three synchronized policy videos.
The videos show live, exact-zero, and fixed anatomical-patch-permuted actor
inputs while preserving the actual physical sensor maps below the world view.

The first two policy renders used the much larger canonical-grasp color scale,
which made the lower taxel fields appear nearly white even when thousands of
taxels were active. Those two superseded H.264 files and render records were
moved to the existing single archive tree under
`pre_shared_scale_policy_videos_20260811/`. The active three-video cohort uses
one shared physical active-taxel 95th-percentile scale, states that scale in
every frame, and fully decodes from beginning to end. Raw traces, actions,
trajectories, and numerical evaluations were not changed.
