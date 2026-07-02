# Gate 00F Runtime Intake Chain Handoff

- Date: `2026-07-01`
- Classification: `runtime_intake_chain_handoff_not_runtime_not_gate_completion`
- Script:
  `src/newton_tactile_curiosity/gate00f_runtime_intake_chain.py`
- Machine-readable handoff:
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff_v1.json`

## Purpose

This script composes the metadata-only Gate 00F runtime intake sequence:

1. Validate a container provenance packet.
2. Register the runtime into a copied candidate registry.
3. Validate the copied candidate registry.

It stops on the first failure and does not run any container, simulator,
dependency installer, renderer, training job, evaluation, or Slurm job.

## Gate Effect

A passing intake chain would only mean the candidate registry is ready for
runtime preflight. It still would not clear Gate 00F. The next required steps
would be runtime preflight, Gate 00F reference bundle, and strict bundle
acceptance inside a Curiosity tmux-held Slurm allocation.

## Negative Control

The existing IsaacLab remote-image-only packet was run through the chain at:

`experiments/outputs/phase00/ref_tactile/runtime_intake/p00_isaaclab_ref_only_20260701/runtime_intake_summary.json`

Result: `fail_container_provenance`. The chain stopped before registry
registration and did not write `candidate_registry.json`.
