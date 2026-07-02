# Gate 00F Container Provenance Negative Control

- Date: `2026-07-01`
- Classification: `negative_control_remote_image_ref_only_not_runtime`
- Packet:
  `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_isaaclab_ref_only_20260701_v1.json`
- Validator summary:
  `experiments/outputs/phase00/ref_tactile/container_provenance/p00_isaaclab_ref_only_20260701/container_provenance_validation_summary.json`
- Validator result:
  `fail_gate00f_container_provenance`

## Purpose

This negative control proves the provenance validator rejects a packet that
has only the remote IsaacLab candidate image ref
`nvcr.io/nvidia/isaac-lab:2.3.2` and source compatibility evidence, but no
local image ID and no existing shared container artifact.

Observed failure:

- `packet must include local image_id or existing artifact_path; image_ref
  alone is not enough`

## Gate Effect

This is blocker/guard evidence only. It does not register a runtime, does not
run runtime preflight, and does not clear Gate 00F.
