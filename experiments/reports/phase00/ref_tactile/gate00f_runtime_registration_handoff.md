# Gate 00F Runtime Registration Handoff

- Date: `2026-07-01`
- Classification: `runtime_registration_handoff_not_runtime_not_gate_completion`
- Script:
  `src/newton_tactile_curiosity/gate00f_runtime_register.py`
- Machine-readable handoff:
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_registration_handoff_v1.json`

## Purpose

This adds a controlled way to register a future dependency-complete Python
environment or prebuilt container into a copied Gate 00F runtime registry.
Registration requires real provenance paths and an existing official source
path; placeholder strings are rejected. Container registrations additionally
require a provenance summary whose status is
`pass_gate00f_container_provenance` and whose target matches the registry
target. Container entries may record the container-internal Python command as
`container_python`; the default is `python3`.

The script is metadata-only. It does not pull images, build images, start
containers, import Isaac/TacSL/TacEx modules, run simulators, or install
dependencies.

## Required Order

1. Register a real runtime into a copied candidate registry.
2. Validate that candidate registry with
   `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`.
3. Run runtime preflight inside a Curiosity tmux-held Slurm allocation. The
   preflight supports registered Python envs, docker local image IDs, and
   singularity/apptainer/sif artifact paths for module-spec checks only.
4. Run the Gate 00F reference bundle inside the compliant allocation workflow.
5. Run strict bundle acceptance.

## Gate Effect

This handoff does not clear Gate 00F. It prevents future ad hoc edits from
turning a remote image ref, placeholder path, base Python env, or non-Curiosity
resource into a claimed dependency-complete runtime.

## Negative Control

A registration attempt using dummy `image_id=sha256:negative-control-dummy`
and the failed IsaacLab remote-image-only provenance summary exited with code
`1` and did not write a candidate registry. The rejection reason was:
`container_provenance_summary status must be pass_gate00f_container_provenance,
observed fail_gate00f_container_provenance`.
