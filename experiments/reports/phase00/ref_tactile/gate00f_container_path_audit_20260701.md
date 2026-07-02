# Gate 00F Container Path Audit

Date: 2026-07-01

Classification: static container-path audit only. This did not build Docker
images, run containers, install dependencies, run official sanity, simulate,
render, or train.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/gate00f_container_path_audit_20260701_v1.json`

## Findings

`docker` exists at `/usr/bin/docker` in the current shell. `singularity`,
`apptainer`, `enroot`, and `podman` are not on the current PATH.

TacEx/TaCauchy include Docker build paths:

- `external/TacEx/docker/Dockerfile`
- `external/TaCauchy/docker/Dockerfile`
- `external/TacEx/docker/docker-compose.yaml`
- `external/TaCauchy/docs/source/installation/Docker-Container-Setup.md`

Those are build recipes, not prebuilt Curiosity environments. The Dockerfile
uses an `isaac-lab-base` image, apt installs development tools and `git-lfs`,
copies CUDA 12.4 devel content, downloads CMake 3.26.4, clones/builds vcpkg,
and sets `CMAKE_TOOLCHAIN_FILE`. The TaCauchy/TacEx docs require an Isaac Lab
Docker base image with Isaac Sim and may require NVIDIA drivers, Docker,
NVIDIA Container Toolkit, and NGC/API-key setup.

IsaacLabTactile includes a cluster singularity helper:

- `external/IsaacLabTactile/docker/cluster/run_singularity.sh`
- `external/IsaacLabTactile/docker/cluster/.env.cluster`

That path expects configured cluster cache directories, a cluster Isaac Lab
directory, a `CLUSTER_SIF_PATH`, and a pre-existing tarred SIF profile. The
local `.env.cluster` is placeholder-only. A project search found no directly
usable `.sif`, `.sqsh`, `*isaac*.tar`, `*tacex*.tar`, or `*tacauchy*.tar`
artifact.

A read-only `docker images` query also returned no Isaac/TacEx/TaCauchy/
UniVTAC-related image names.

## Gate Effect

Container source paths exist, but they do not currently clear Gate 00F. A
compliant path would need an already-built, approved, Curiosity-owned image or
SIF on shared storage, followed by official UniVTAC/TaCauchy sanity inside a
tmux-held Slurm allocation.

Do not build Docker images on the login node. Do not install dependencies on
compute nodes. Curiosity training remains disallowed.
