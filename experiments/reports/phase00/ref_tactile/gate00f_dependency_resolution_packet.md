# Gate 00F Dependency Resolution Packet

- Date: `2026-07-01`
- Classification: `dependency_resolution_packet_not_training_not_gate_completion`
- Current Gate 00F state: `open_official_semantic_validation_blocked`

This packet turns the remaining Gate 00F blocker into a concrete dependency
boundary. It does not run training, simulation, rendering, model loading,
dataset conversion, package installation, or package builds.

## Current Positive State

- UniVTAC base Python exists at `envs/univtac/conda/bin/python`.
- TaCauchy base Python exists at `envs/tacauchy/conda/bin/python`.
- TaCauchy asset file presence has been repaired from the approved UniVTAC
  bundled TacEx asset copy.
- Official IsaacLab source exists at `external/IsaacLab_official`.
- The Gate 00F reference bundle exists:
  `experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh`.
- The strict bundle acceptance checker exists:
  `src/newton_tactile_curiosity/gate00f_bundle_acceptance.py`.

These are prerequisites only. They are not official dependency-complete
reference environments.

## Official Dependency Requirements

UniVTAC source: `external/UniVTAC/README.md` and
`external/UniVTAC/scripts/install.sh`.

Required by the official path:

- Python 3.10 conda environment or equivalent project-local prefix.
- Isaac Sim `4.5.0`.
- Isaac Lab `v2.1.1`.
- TacEx from the modified local source.
- cuRobo.
- TacEx UIPC/libuipc build dependencies.
- Torch `2.5.1` CUDA `12.4`-compatible wheel.
- `git-lfs`, `cmake`/build tools, and `uv`/pip install steps.

TaCauchy source: `external/TaCauchy/README.md`,
`external/TaCauchy/ASSETS.md`, and
`external/TacEx/docs/source/installation/Local-Installation.md`.

Required by the official path:

- Python 3.11 environment compatible with TaCauchy.
- Isaac Sim `5.0` / Isaac Lab `2.2.1` per TaCauchy README.
- TacEx core install.
- TacEx assets, including GelSight Mini `Sensor.usd` and tactile test shapes.
- UIPC/libuipc build.
- `vcpkg` plus `CMAKE_TOOLCHAIN_FILE`.
- CMake `3.26`.
- GCC `11.4` or newer for C++20 support.
- CUDA/nvcc compatible with UIPC build; TacEx docs document CUDA `12.4`.

IsaacLab TacSL source: `external/IsaacLab_official/README.md` and
`external/IsaacLab_official/scripts/demos/sensors/tacsl_sensor.py`.

Required by the official path:

- Dependency-complete official Isaac Lab environment.
- Isaac Sim version compatible with the checked Isaac Lab source.
- Camera-enabled headless run support.
- TacSL contrib sensor dependencies and assets.

## Allowed Resolution Paths

1. Reuse an existing dependency-complete environment.

   Provide executable Python paths through `UNIVTAC_PYTHON`,
   `TACAUCHY_PYTHON`, and `ISAACLAB_TACSL_PYTHON`, or place them under
   approved `envs/` prefixes. Record Python versions, import checks,
   repository commits, and sanity command outputs. Run official sanity only
   inside a Curiosity tmux-held Slurm allocation.

2. Reuse an existing prebuilt container.

   The container must already exist on the shared filesystem before compute
   allocation use. Record image/SIF/tar path, build provenance if available,
   and exact bind/env command. Do not build the image on the login node or
   inside a compute experiment allocation.

3. Prepare the environment through a compliant external env-prep workflow.

   Heavy dependency install/build work must happen only in an approved
   non-login, non-experiment env-prep workflow. After preparation, register
   executable env/container paths in project evidence, then run the Gate 00F
   bundle and strict acceptance checker inside a Curiosity tmux-held Slurm
   allocation.

## Disallowed Paths

- Compile UIPC/libuipc on the login node.
- Install Isaac Sim, Isaac Lab, TacEx, cuRobo, Torch, or other large
  dependencies on the login node if the command can plausibly become heavy.
- Install dependencies or build packages inside a compute allocation reserved
  for experiments.
- Reuse Reflex/OpenPI/Cosmos/non-Curiosity allocations or tmux sessions.
- Run the Gate 00F bundle with `ALLOW_BLOCKER_SANITY=1` and treat it as
  accepted.
- Treat base Python envs as dependency-complete official reference envs.

## Acceptance Sequence

After dependency resolution:

1. Run the Gate 00F reference bundle inside a Curiosity tmux-held Slurm
   allocation.
2. Run `gate00f_bundle_acceptance.py` on the generated bundle summary.
3. Require `pass_gate00f_bundle_acceptance` before any Gate 00F completion
   claim.
4. Keep curiosity training disallowed until Gate 00D, Gate 00E, and Gate 00F
   have passed or faithful blockers have been explicitly accepted.
