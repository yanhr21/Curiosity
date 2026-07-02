# Gate 00F Runtime Registry Container Support Update

- Date: `2026-07-01`
- Classification: `registry_validator_update_not_training_not_gate_completion`

The runtime registry validator now supports strict container registration
metadata.

The runtime registration helper and both validators enforce the same local
runtime-reference rules, so a weak container entry is rejected before it can be
written and again during validation.

Allowed `container_runtime` values:

- `docker`
- `singularity`
- `apptainer`
- `enroot`
- `sif`
- `sqsh`
- `tar`

Required for a registered container runtime:

- `artifact_path` for a shared-filesystem image/archive, or local `image_id`
  for a Docker-style local image
- `image_id` must look like a local immutable digest or ID, either
  `sha256:<hex>` or at least 12 hexadecimal characters; it cannot be the same
  string as `image_ref`
- `artifact_path` must exist, must be a file, and must end with one of `.sif`,
  `.sqsh`, `.tar`, `.tar.gz`, or `.img`
- `status=dependency_complete_registered`
- allowed `resolution_path`
- expected modules
- provenance

Remote `image_ref` alone is not enough. It is acquisition evidence only, so an
NGC tag such as `nvcr.io/nvidia/isaac-lab:2.3.2` cannot pass registry
validation until a local image ID or shared artifact path is recorded with
provenance.

This update does not clear Gate 00F. It prevents remote image tags from being
mistaken for usable runtime evidence.
