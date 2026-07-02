# Gate 00F Container Runtime Registration Examples

- Date: `2026-07-01`
- Classification: `container_registration_examples_not_runtime_not_gate_completion`
- Machine-readable examples:
  `experiments/configs/phase00/ref_tactile/gate00f_container_runtime_registration_examples_v1.json`

These are examples only. They do not register any runtime and do not clear
Gate 00F.

## Important Rule

Remote `image_ref` alone is acquisition evidence, not a dependency-complete
registered runtime.

For a `container` target to pass registry validation, it must include:

- `container_runtime`: `docker`, `singularity`, `apptainer`, `enroot`, `sif`,
  `sqsh`, or `tar`
- `artifact_path` for a shared-filesystem image/archive, or local `image_id`
  for a Docker-style local image
- `status`: `dependency_complete_registered`
- allowed `resolution_path`
- expected modules
- provenance

## Example Meanings

- `isaaclab_tacsl_ngc_candidate_not_registered` shows why
  `nvcr.io/nvidia/isaac-lab:2.3.2` is only a candidate until a local image ID
  or shared artifact path and compatibility evidence exist.
- `tacex_project_image_registered_shape` shows the required shape for a future
  approved `isaac-lab-tacex` project image.
- `shared_sif_registered_shape` shows the required shape for a future shared
  SIF-style artifact.

No image was pulled, no image was built, no container was run, and no runtime
was promoted.
