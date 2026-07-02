# Phase 00 Reference Env Toolchain Preflight

Date: 2026-07-01

Scope: lightweight executable/version inspection on the login node. No
environment creation, dependency installation, package resolution, simulation,
rendering, training, model loading, dataset conversion, or official sanity was
run.

## Available

- Curiosity local conda:
  `envs/taccel/miniforge/bin/conda`
- Conda version:
  `conda 26.3.2`
- System GCC:
  `/usr/bin/gcc-11`, version `11.4.0`
- System G++:
  `/usr/bin/g++-11`, version `11.4.0`
- Existing local cache directories:
  - `envs/taccel/conda_pkgs_cuda_retry`
  - `envs/taccel/cuda-toolkit`
  - `envs/taccel/cuda_conda_pkgs`
  - `envs/taccel/downloads`
  - `envs/taccel/miniforge/pkgs`
  - `envs/taccel/packman_cache`

## Missing From Login Environment

- `cmake`
- `git-lfs`
- `nvcc`
- `nvidia-smi`
- shell-level `conda` command on `PATH` by default

## Interpretation

The official UniVTAC/TaCauchy blocker is not solved. The environment build path
must still prepare or expose:

- CMake 3.26+ inside the target env or module path;
- CUDA/NVCC support for UIPC/libuipc build;
- Git LFS or an alternate asset acquisition path for TacEx tactile assets;
- package-index/proxy/mirror routes for Isaac Sim, Isaac Lab, PyTorch, cuRobo,
  TacEx, and UIPC dependencies.

This preflight supports the controlled-env plan. It is not official reference
sanity, not Gate 00F completion, and not curiosity progress.
