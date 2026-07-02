# Gate 00F Container Acquisition Plan

- Date: `2026-07-01`
- Classification: `container_acquisition_plan_not_training_not_gate_completion`

This is a source-backed acquisition plan only. It did not pull images, build
containers, install dependencies, run simulation, render, train, evaluate, or
load models.

## Official Findings

- NVIDIA Isaac Sim has an official container installation path for remote
  headless/cloud Docker deployment and requires Docker plus NVIDIA Container
  Toolkit.
  Source: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html
- NVIDIA NGC distributes official Isaac Sim containers with required EULA and
  privacy-related runtime environment flags.
  Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
- Isaac Lab provides Docker workflows through its repository `docker/` tooling
  and documents pre-built NGC Isaac Lab containers, including
  `nvcr.io/nvidia/isaac-lab:2.3.2`.
  Source: https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html
- Isaac Lab also documents container deployment and HPC adaptation paths,
  including Docker-to-Singularity/Apptainer workflows.
  Source: https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html
- TacEx Docker setup requires an Isaac Lab Docker image with Isaac Sim 4.5 /
  Isaac Lab 2.1.1, then builds a project image such as `isaac-lab-tacex`.
  Source: https://github.com/DH-Ng/TacEx/blob/main/docs/source/installation/Docker-Container-Setup.md

## Local Source Findings

- `external/TacEx/docker/.env.base` and
  `external/TaCauchy/docker/.env.base` both use
  `ISAACLAB_BASE_IMAGE=isaac-lab-base`.
- TacEx/TaCauchy docker-compose files name the project image/container
  `isaac-lab-tacex`.
- TacEx/TaCauchy Dockerfiles build from the Isaac Lab base image and include a
  CUDA `12.4` development stage for UIPC/libuipc-related build support.

## Candidate Paths

1. IsaacLab TacSL via official Isaac Lab pre-built container.

   Candidate image: `nvcr.io/nvidia/isaac-lab:2.3.2`.

   This is the strongest current candidate for an official IsaacLab TacSL
   runtime if source/version compatibility with `external/IsaacLab_official`
   can be verified. It still needs a cluster-approved container runtime and a
   Curiosity-accessible image path before it can be registered.

2. UniVTAC/TacEx via project container.

   TacEx documentation requires an Isaac Lab base image with Isaac Sim 4.5 /
   Isaac Lab 2.1.1, then a project image build. This is likely the faithful
   path for UniVTAC/TacEx reference sanity, but it requires a compliant
   external build or an already-existing prebuilt image. It cannot be built on
   the login node or inside an experiment compute allocation.

3. TaCauchy via project container.

   TaCauchy has a local Dockerfile route, but it still requires a project image
   build and UIPC/libuipc compilation. It may also have a version mismatch to
   resolve because TaCauchy documents Isaac Sim 5.0 / Isaac Lab 2.2.1 while the
   observed current official Isaac Lab pre-built image is `2.3.2`.

## Required Sequence

1. Obtain or prepare a prebuilt container outside login-node and experiment
   compute contexts.
2. Place or reference the container on a Curiosity-accessible shared
   filesystem.
3. Update the runtime registry with `kind=container`, exact path, allowed
   resolution path, expected modules, and provenance.
4. Validate the runtime registry.
5. Run registry-gated runtime preflight inside a Curiosity tmux-held Slurm
   allocation.
6. Run the Gate 00F reference bundle inside the same compliant allocation
   workflow.
7. Run strict bundle acceptance.

## Non-Claims

- No container was downloaded.
- No image was built.
- No runtime was registered as dependency-complete.
- Gate 00F is not cleared.
- Curiosity training remains disallowed.
