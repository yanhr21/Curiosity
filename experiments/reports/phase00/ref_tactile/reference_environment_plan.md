# Phase 00 Official Tactile Reference Environment Plan

Date: 2026-07-01

This plan exists because Gate 00F now requires official UniVTAC and TaCauchy
sanity before curiosity training can restart. The current blocker is not source
code availability: both official repositories are cloned and commit-matched.
The blocker is missing approved prebuilt environments.

## Current Blocker Evidence

- UniVTAC probe:
  `p00_ref_univtac_sanity_v1_20260701_054900`
  - status: `blocked_missing_prebuilt_environment`
  - expected and observed commit:
    `05bcd3edb92237107efa40105292a24f1a9fd761`
  - missing executable: `UNIVTAC_PYTHON`, `envs/univtac/conda/bin/python`,
    or `envs/univtac/.venv/bin/python`

- TaCauchy probe:
  `p00_ref_tacauchy_sanity_v1_20260701_054900`
  - status: `blocked_missing_prebuilt_environment`
  - expected and observed commit:
    `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
  - missing executable: `TACAUCHY_PYTHON`, `envs/tacauchy/conda/bin/python`,
    or `envs/tacauchy/.venv/bin/python`

- Gate review:
  `p00_gate_review_v4_20260701_055100`
  - Gate 00D: `open_reference_semantics_blocked`
  - Gate 00E: `open_tactile_validation_blocked`
  - Gate 00F: `open_official_semantic_validation_blocked`

## Existing Environment Audit

The local shared filesystem currently contains Newton, Taccel, T-Rex, residual,
and curiosity environments, plus a Taccel miniforge base. A metadata-only scan
found no UniVTAC/TaCauchy-ready Isaac/TacEx/UIPC environment:

- no `envs/univtac/conda` or `envs/univtac/.venv`;
- no `envs/tacauchy/conda` or `envs/tacauchy/.venv`;
- no installed `isaac*`, `omni*`, `tacex*`, or `uipc*` package directories in
  the existing checked environment paths;
- `envs/taccel/miniforge` lists only its base environment inside Curiosity.

## Why The Official Scripts Cannot Be Run Blindly

### UniVTAC

`external/UniVTAC/scripts/install.sh` is an all-in-one installer. It can:

- create a conda environment named `UniVTAC`;
- install PyTorch and Isaac Sim 4.5;
- install Isaac Lab 2.1.1;
- clone/install cuRobo;
- install the bundled modified TacEx;
- install/build UIPC/libuipc;
- run tests and a data collection command.

This is too broad to run inside a compute allocation and too invasive to run
without a controlled local-environment plan. It also includes `sudo apt` and
home-directory toolchain mutations, which need explicit handling.

### TaCauchy

TaCauchy requires Isaac Sim/Lab plus UIPC/libuipc and separate large tactile
assets. Its docs mention:

- Isaac Sim 5.0 / Isaac Lab 2.2.1;
- Python 3.11;
- UIPC/libuipc build;
- GCC 11.4+ and CMake 3.26+;
- CUDA/toolchain settings;
- large TacEx sensor assets that are not in the Git repository.

This is also not a compute-node install task. It must be prepared as a local
shared-filesystem environment and then only activated on compute nodes.

## Controlled Preparation Order

1. Pick a single environment strategy before installing:
   - Option A: separate envs:
     `envs/univtac/conda` for Isaac Sim 4.5 / Isaac Lab 2.1.1, and
     `envs/tacauchy/conda` for Isaac Sim 5.0 / Isaac Lab 2.2.1.
   - Option B: one Isaac/TacEx shared env only if version requirements prove
     compatible. Current docs suggest version mismatch, so Option A is safer.

2. Prepare local shared-filesystem envs only outside compute allocation time:
   - no dependency installation on compute nodes;
   - no official demos or data collection on login nodes;
   - no unreviewed all-in-one script execution.

3. Split each official installer into audited stages:
   - conda/env creation;
   - PyTorch/Isaac Sim installation;
   - Isaac Lab checkout/version pin;
   - TacEx install;
   - UIPC/libuipc build;
   - asset download/copy;
   - minimal official sanity command.

4. Record every stage:
   - exact command;
   - environment path;
   - package source/mirror/proxy;
   - commit/version;
   - log path;
   - pass/fail/blocker.

5. After envs exist, rerun:
   - `p00_ref_univtac_sanity_*` with either
     `UNIVTAC_PYTHON=<env>/bin/python`,
     `envs/univtac/conda/bin/python`, or
     `envs/univtac/.venv/bin/python`;
   - `p00_ref_tacauchy_sanity_*` with either
     `TACAUCHY_PYTHON=<env>/bin/python`,
     `envs/tacauchy/conda/bin/python`, or
     `envs/tacauchy/.venv/bin/python`;
   - Gate review with both sanity summaries.

## Immediate Decision Needed Before Heavy Install

The next heavy step is environment construction. It may require large downloads,
compiler toolchains, Isaac Sim packages, TacEx assets, and UIPC builds. It must
not be started as a hidden downgrade or a compute-node install.

Recommended next action:

1. Build separate local shared-filesystem envs for UniVTAC and TaCauchy under
   `envs/univtac/` and `envs/tacauchy/`.
2. Use China-accessible mirrors/proxy settings if the default package route is
   blocked.
3. Stop and record a blocker if `sudo`, Isaac package access, Git LFS assets,
   or UIPC compilation cannot be completed cleanly.

Until this environment blocker is solved, Gate 00F remains open and curiosity
training remains disallowed.
