# Gate 00F Container Provenance Contract

- Date: `2026-07-01`
- Classification: `container_provenance_contract_not_runtime_not_gate_completion`
- Machine-readable contract:
  `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_contract_v1.json`
- Validator:
  `src/newton_tactile_curiosity/gate00f_container_provenance_validate.py`

## Purpose

This contract defines the minimum evidence required before any prebuilt
container or local image can be written into the Gate 00F runtime registry.
It is stricter than a Docker image name: a future packet must include local
image identity or an existing shared artifact path, source commit evidence,
expected modules, and real provenance paths.

The validator now also rejects weak local runtime claims:

- `image_id` must look like a local immutable digest or ID, either
  `sha256:<hex>` or at least 12 hexadecimal characters. A registry tag or
  image ref is not accepted as an image ID.
- `image_id` must not be identical to `image_ref`.
- `artifact_path` must exist, must be a file, and must end with one of `.sif`,
  `.sqsh`, `.tar`, `.tar.gz`, or `.img`.
- Remote `image_ref` alone remains acquisition evidence only.

## Target Contracts

- `isaaclab_tacsl`:
  source `external/IsaacLab_official` at
  `b4c321024792976150ca55fddb26fa34480d974e`, VERSION `2.3.2`, candidate
  image ref `nvcr.io/nvidia/isaac-lab:2.3.2`, expected modules `isaacsim`,
  `isaaclab`, and `isaaclab_contrib`.
- `univtac`:
  official UniVTAC source at `05bcd3edb92237107efa40105292a24f1a9fd761` plus
  TacEx project runtime source at `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`.
  Local TacEx Docker metadata builds image `isaac-lab-tacex` from
  `isaac-lab-base` and expects Isaac Lab 2.1.1 / Isaac Sim 4.5 lineage.
- `tacauchy`:
  official TaCauchy source at `c228cfe9050904cd5d71d64f6eb5104768d4cbda`.
  Local Docker metadata also builds image `isaac-lab-tacex` from
  `isaac-lab-base` and includes CUDA 12.4/CMake/vcpkg style UIPC build
  support.

## Gate Effect

Passing this provenance contract is necessary before registry registration,
but not sufficient for Gate 00F. The required order remains:

1. Validate container provenance packet.
2. Write a copied candidate runtime registry.
3. Validate the runtime registry.
4. Run runtime preflight in a Curiosity tmux-held Slurm allocation.
5. Run the Gate 00F reference bundle.
6. Run strict bundle acceptance.

No container was pulled, no image was built, no runtime was registered, and
Gate 00F remains open.
