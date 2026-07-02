# Phase 00 Reference Tactile Source Audit

Date: 2026-07-01

Purpose: start the reference-video-aligned tactile rebuild from current serious
codebases instead of extending the legacy scalar contact-count pipeline.

## Reference Video

Reference asset:
`0780e5ec3fdb26b63ae63de0f49f07c4.mp4`

Inspection assets:

- `experiments/visuals/phase01/ref/0780/contact_sheet.jpg`
- `experiments/visuals/phase01/ref/0780/inspect.json`
- `experiments/reports/phase01/ref/0780_tactile_gap_report.md`

Required tactile target extracted from the video:

- synchronized visual scene and tactile maps;
- left/right pad views;
- pressure or compression heatmaps;
- `Fn` normal force and `Ft` tangential/shear force;
- shear vectors;
- contact area and penetration/compression time series;
- material-dependent rigid/soft behavior;
- steel/metal rigid contact first, with stress/contact/friction statistics.

## Sources Checked

### Newton

Official source: `https://github.com/newton-physics/newton`

Web finding:

- NVIDIA describes Newton 1.0 GA as an accelerated, production-ready foundation
  for dexterous manipulation and locomotion, built on Warp/OpenUSD and usable
  with Isaac Lab/Isaac Sim.
- The official GitHub describes Newton as GPU-accelerated, built on NVIDIA
  Warp, integrating MuJoCo Warp, OpenUSD, differentiability, and extensibility.

Local state:

- local checkout: `external/newton`
- local HEAD: `99a878cbb5479d2698051e8b9ceee696b999f759`
- fetched upstream main: `a217e55fab3d373a08fba374cc5cafc1826cf27f`
- latest observed tag: `v1.3.0`
- latest observed tag commit: `ce11136b3a28390944f7fe5a32801b31d8aa5670`
- status: clean before active edits; fetched but not checked out to avoid
  changing existing legacy evidence paths.

Relevant local code:

- `external/newton/newton/examples/robot/example_robot_panda_hydro.py`
  provides Panda hydroelastic grasp/manipulation with gripper pads.
- `external/newton/newton/examples/contacts/example_nut_bolt_hydro.py`
  provides hydroelastic mesh collision and user pressure callback.
- `external/newton/newton/tests/test_rigid_friction_ramp.py`
  provides Coulomb friction/static/kinetic validation structure.
- `external/newton/newton/_src/sensors/sensor_contact.py`
  exposes accumulated contact force and friction decomposition, including
  total force, friction force, force matrix, and contact normal usage.

Decision:

Newton remains the primary runtime/physics target. The immediate risk is that
the local checkout is older than upstream; create a separate latest worktree or
update after recording compatibility with current environments.

### Taccel

Official source: `https://github.com/Taccel-Simulator/Taccel`

Web finding:

- Taccel is presented as a high-performance simulation platform for
  vision-based tactile sensors and robots.
- Its docs state that Taccel integrates IPC and ABD to model robots, tactile
  sensors, and objects with realistic tactile signals and flexible APIs.
- Its examples include peg insertion, deformable object grasping, and Tac-Man.

Local state:

- local checkout: `external/Taccel`
- local HEAD: `cb23bc251b531ba6908a3788c2f91423cd543149`
- upstream main observed: `cb23bc251b531ba6908a3788c2f91423cd543149`
- status: local checkout matches upstream main.

Relevant local code:

- `external/Taccel/examples/peg.py`
  uses two soft sensor bodies around peg/hole manipulation.
- `external/Taccel/examples/grasp_soft_teddy.py`
  uses tactile RobotiQ-3F, VBTS integration, soft volumetric body, and GPU IPC.
- `external/Taccel/examples/fabricate_sensor.py`
  is relevant for sensor geometry generation.

Decision:

Taccel is the main tactile-simulation reference. The next gate should run its
official examples in compute, then adapt the relevant sensor/field outputs into
the Newton/Taccel dense tactile schema.

### T-Rex

Official source: `https://github.com/ZhuoyangLiu2005/T-Rex`

Web finding:

- T-Rex reports a 100-hour tactile-reactive dataset and about 50 hours
  open-sourced in LeRobot v3.0 format.
- The model uses asynchronous Mixture-of-Transformers with a Qwen3-VL-2B
  backbone, slow action denoising around 5 Hz, faster tactile refinement around
  20 Hz, and temporal tactile VQ-VAE.
- The released midtrain checkpoint embeds the tactile VQ-VAE and encodes
  tactile codes on the fly.

Local state:

- local checkout: `external/T-Rex`
- local HEAD: `db7a02992504ad9be53a7e764f7b05d81d86c767`
- upstream main observed: `43ff632259d76f08373c085c53111825060d029b`
- local dirty state:
  - modified `qwen_vla/lerobot_dataset.py`
  - deleted tracked pycache under `dataset_quickstart/.../__pycache__/`
- do not overwrite local dirty state silently.

Relevant local code:

- `external/T-Rex/tactile_vqvae/`
- `external/T-Rex/qwen_vla/modeling_vla.py`
- `external/T-Rex/qwen_vla/modeling_qwen3vl_mot.py`
- `external/T-Rex/qwen_vla/lerobot_dataset.py`
- `external/T-Rex/config/`

Decision:

T-Rex is the model/reference architecture path, not the simulator path. Do not
hand-roll a replacement. Use official architecture/checkpoint paths or record a
blocker.

### HydroShear

Official source: `https://github.com/MMintLab/hydroshear`

Web finding:

- HydroShear is a hydroelastic shear simulator for tactile sim-to-real RL and
  includes visualization/training code for manipulation tasks such as peg
  insertion, bin packing, book shelving, and drawer pulling.

Local state:

- cloned to `external/hydroshear`
- local HEAD: `a53a51cb74f0608ca53839415d7f1964a99f1db0`
- status: clean immediately after clone.

Relevant local code/assets:

- `external/hydroshear/README.md`
- `external/hydroshear/training.md`
- `external/hydroshear/demo_assets/`
- `external/hydroshear/configs/config.yaml`

Decision:

Use HydroShear as a reference for shear modeling and tactile-policy
visualization, not as the immediate main simulator unless Newton/Taccel hit a
recorded blocker.

### IsaacLab TacSL / IsaacLabTactile

Official sources:

- Isaac Lab visuo-tactile sensor docs:
  `https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/visuo_tactile_sensor.html`
- IsaacLabTactile repository:
  `https://github.com/UM-ARM-Lab/IsaacLabTactile`

Web finding:

- Isaac Lab docs describe a visuo-tactile sensor integrated with TacSL,
  producing tactile RGB images, force field distributions, and intermediate
  tactile measurements.
- IsaacLabTactile upstream has observed tag `v2.2.1`.

Local state:

- clone attempted on 2026-07-01;
- failed with `fetch-pack: unexpected disconnect while reading sideband packet`;
- no usable local checkout was created.

Decision:

Record as a comparison/reference source with a current network acquisition
blocker. It is not required for the Newton/Taccel mainline gate.

## Current Mainline Decision

Use:

1. Newton latest official code path for robot, hydroelastic contacts, contact
   sensor, and friction validation.
2. Taccel latest official code path for tactile sensor simulation and dense
   tactile fields.
3. T-Rex official checkpoint/model path as the tactile-reactive base-model
   reference after the simulator can produce dense tactile data.
4. HydroShear and IsaacLab tactile as comparison/reference paths for shear and
   force-field design.

Do not continue the legacy scalar-contact curiosity path as active work.

## Immediate Next Actions

1. Create a separate latest Newton worktree or compatibility branch so the old
   local checkout is not mutated unexpectedly.
2. Write the dense tactile schema and metal/steel first-scene spec.
3. Prepare compute-side official sanity runners for Newton and Taccel.
4. Run official sanity only inside a Curiosity-owned tmux-held Slurm
   allocation.
5. Generate the first rigid-metal reference diagnostic only after official
   sanity passes.

## Latest Recheck During Execution

Rechecked on 2026-07-01 after Phase 00 mechanics diagnostics.

### Remote HEADs

Commands were lightweight `git ls-remote` checks from the login node; no
simulation, rendering, training, or data conversion was run.

- Newton:
  - official source: `https://github.com/newton-physics/newton`
  - remote `main`: `a217e55fab3d373a08fba374cc5cafc1826cf27f`
  - remote `v1.3.0` tag: `ce11136b3a28390944f7fe5a32801b31d8aa5670`
  - release page still identifies `v1.3.0` as the latest observed release.
  - active experiment worktree: `external/newton_v1.3` at
    `ce11136b3a28390944f7fe5a32801b31d8aa5670`.
  - decision: keep active Phase 00 evidence on the stable v1.3.0 worktree for
    reproducibility; do not silently move to `main` until compatibility is
    checked in a separate run.
- Taccel:
  - official source: `https://github.com/Taccel-Simulator/Taccel`
  - remote `main`: `cb23bc251b531ba6908a3788c2f91423cd543149`
  - local checkout already matches this commit.
  - docs/paper continue to position Taccel as a high-throughput tactile
    simulator for VBTS with large parallel simulation.
- T-Rex:
  - official source: `https://github.com/ZhuoyangLiu2005/T-Rex`
  - remote `main`: `43ff632259d76f08373c085c53111825060d029b`
  - local checkout remains older and dirty. Do not overwrite or reuse it as
    latest without creating a separate clean checkout/worktree.
  - web/paper pages describe the current T-Rex claim as a tactile-reactive
    model with a 100-hour tactile-synchronized dataset, temporal tactile
    VQ-VAE, and strong tactile-reactive manipulation results. This remains a
    model/reference path, not a simulator replacement.
- HydroShear:
  - official source: `https://github.com/MMintLab/hydroshear`
  - remote `HEAD`: `a53a51cb74f0608ca53839415d7f1964a99f1db0`
  - local checkout matches this commit.
  - no release was observed; use as shear-modeling reference.

### Execution-Time Consequences

- Newton `external/newton_v1.3` is now the active stable runtime for Phase 00
  evidence because it passed official sanity, Panda hydro lift, USD visual,
  synchronized tactile/mechanics export, runtime benchmark, and steel-spec
  candidate diagnostics on the held H200 allocation.
- Taccel remains a tactile reference but current instrumented Taccel contact
  attempts produced zero force/collision/deformation evidence; do not claim
  Taccel dense tactile success until that is fixed.
- T-Rex must only be used via official architecture/checkpoint/dataset paths.
  Do not hand-roll a small replacement VQ-VAE, tactile encoder, or policy.
- HydroShear should be mined next for shear-vector representation and tactile
  policy visualization design, but it should not replace Newton/Taccel without
  a recorded blocker and user-visible decision.

## Newton Main Worktree Compatibility Update

Updated on 2026-07-01 after the 82 FPS performance gap remained open on
Newton v1.3.0.

- Added independent worktree: `external/newton_main`.
- Source: official Newton upstream `main`.
- Commit: `a217e55fab3d373a08fba374cc5cafc1826cf27f`.
- The v1.3.0 worktree `external/newton_v1.3` remains untouched for
  reproducibility.

Execution evidence:

- `p00_bench_main_20260701_035529` measured `92.6 FPS` on H200 for the
  official Panda hydro null-viewer benchmark, meeting the user's `82 FPS`
  target.
- `p00_main_f6_v1_20260701_035926` verified that the same main worktree can run
  the synchronized steel-spec grid/F6 tactile diagnostic and export source
  arrays, AVI, and contact sheet.

Consequence:

- Newton main is now the best current performance candidate for Phase 00.
- Gate 00E performance is satisfied by main benchmark evidence, but the full
  gate remains open because direct hydro `Ft`, USD/photoreal fusion, and
  reference-grade tactile density are still missing.

## Latest 2026 Tactile Reference Recheck

Updated on 2026-07-01 after Gate 00F bridge-spec enforcement.

- Report:
  `experiments/reports/phase00/ref_tactile/latest_reference_code_recheck.md`
- Tacmap:
  - paper: `https://arxiv.org/abs/2602.21625`
  - no official code link observed on arXiv/html;
  - common GitHub repository-name probes returned `Repository not found` or
    were unavailable within the short timeout;
  - status: code-unavailable comparison gap.
- ControlTac:
  - project: `https://dongyuluo.github.io/controltac/`
  - paper: `https://arxiv.org/abs/2505.20498`
  - no code/GitHub link observed on the project page;
  - common GitHub repository-name probes returned `Repository not found`;
  - status: code-unavailable comparison gap.
- FreeTacMan:
  - official source: `https://github.com/OpenDriveLab/FreeTacMan`
  - local checkout: `external/FreeTacMan`
  - local commit: `9285740a5d33385d3a9cf5ccdb185e3387b547bd`
  - status: secondary official 2026 real visuo-tactile data/pretraining
    reference. It is not a simulator and not a replacement for UniVTAC/TaCauchy
    Gate 00F.
- DiffTactile:
  - official source checked: `https://github.com/Genesis-Embodied-AI/DiffTactile`
  - remote HEAD: `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`
  - status: secondary remote-only differentiable tactile simulator reference.
