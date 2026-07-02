# Gate 00F Container Dispatch Login Refuse

- Date: `2026-07-01`
- Classification: `login_refuse_check_not_runtime_not_gate_completion`

After adding registry-aware container dispatch for official reference sanity,
the entry scripts were checked on the login node. Each refused before registry
validation, helper sourcing, container commands, module imports, official
sanity, simulation, rendering, or training.

## Results

- `run_tactile_reference_sanity_in_alloc.sh` with `TARGET=univtac`
  - Exit code: `2`
  - Stderr: `ERROR: must run inside a Slurm allocation.`
- `run_isaaclab_tacsl_sanity_in_alloc.sh`
  - Exit code: `2`
  - Stderr: `ERROR: must run inside a Slurm allocation.`
- `run_gate00f_reference_bundle_in_alloc.sh`
  - Exit code: `2`
  - Stderr: `ERROR: must run inside a Slurm allocation.`

## Gate Effect

Safety check only. This does not register a runtime, run a container, run
official sanity, clear Gate 00F, or allow curiosity training.
