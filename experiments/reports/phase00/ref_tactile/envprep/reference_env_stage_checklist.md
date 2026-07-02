# Reference Env Stage Checklist

Date: 2026-07-01

This checklist is planned environment preparation only. Nothing in this file
has been executed as installation, build, simulation, rendering, data
collection, training, or official sanity.

Machine-readable checklist:
`experiments/configs/phase00/ref_tactile/envprep/reference_env_stage_checklist_v1.json`

## Current Blocker

Gate 00F is still blocked because no approved `envs/univtac` or
`envs/tacauchy` Python executable exists. The current shell also does not
expose `conda`, `mamba`, `micromamba`, `module`, `cmake`, `git-lfs`, or `nvcc`.
There is a project-local env creator candidate,
`envs/taccel/miniforge/bin/conda` (`conda 26.3.2`), but the target envs do not
exist yet and heavy construction must remain controlled and logged.

This means the next heavy action is not a compute run. It is controlled local
environment construction or locating an already approved shared-filesystem
environment.

## UniVTAC Plan

Target env: `envs/univtac/conda`

Official requirements from the local docs and installer:

- Python 3.10
- Isaac Sim 4.5.0
- Isaac Lab v2.1.1
- PyTorch 2.5.1 / torchvision 0.20.1 on CUDA 12.4 path
- cuRobo commit `0a50de1ba72db304195d59d9d0b1ed269696047f`
- modified TacEx from `external/UniVTAC/third_party/TacEx`
- UIPC/libuipc through the bundled TacEx path
- `torch_scatter==2.1.2` for torch 2.5.1+cu124

Do not run `external/UniVTAC/scripts/install.sh` blindly. It creates envs,
installs Isaac, clones/builds dependencies, invokes `sudo apt`, runs tests, and
then calls data collection. Those steps must be split and logged.

Minimal official sanity after the env exists:
`bash collect_data.sh grasp_classify demo 0`.

Schema-only inputs to inspect before compute sanity include
`policy/task_settings.json`, ACT tactile/vision configs, and
`task_config/contact.yml`.

## TaCauchy Plan

Target env: `envs/tacauchy/conda`

Official requirements from the local docs:

- Python 3.11
- Isaac Sim 5.0.0
- Isaac Lab v2.2.1
- UIPC/libuipc build
- GCC 11.4+
- CMake >= 3.26
- vcpkg and `CMAKE_TOOLCHAIN_FILE`
- CUDA toolkit visible for the libuipc build
- large sensor/robot/shape assets under `source/tacex_assets/tacex_assets/data`

TaCauchy source semantics relevant to Gate 00F are Cauchy stress, normal
pressure, tangential traction, 20x20 interpolated force grid, tactile RGB, and
multi-sensor GelSight/DIGIT/9DTact support.

Do not run `scripts/setup_assets.sh`, `git lfs install`, `conda env update`,
or `pip install -e source/tacex_uipc -v` as an untracked shortcut. These touch
large assets, toolchain state, and compiled extensions.

Minimal official sanity after the env and assets exist:
`python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight`.

## Required Stages

1. Toolchain and env-access preflight.
2. Create local env prefixes under `envs/univtac/conda` and
   `envs/tacauchy/conda`.
3. Install Isaac Sim and Isaac Lab with exact version logs.
4. Install the correct TacEx source for each target.
5. Build or validate UIPC/libuipc.
6. Set up large tactile assets with provenance.
7. Run official sanity only inside a Curiosity tmux-held Slurm allocation after
   the envs exist.

Each stage must write logs under
`logs/newton/phase00/ref_tactile/envprep/`, reports under
`experiments/reports/phase00/ref_tactile/envprep/`, and status JSON under
`experiments/outputs/phase00/ref_tactile/envprep/`.
