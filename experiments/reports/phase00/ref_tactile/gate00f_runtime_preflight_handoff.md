# Gate 00F Runtime Preflight Handoff

- Date: `2026-07-01`
- Classification: `handoff_not_training_not_gate_completion`
- Script:
  `experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`

After dependency-complete Python envs are available and registered, run this
inside a Curiosity-owned tmux-held Slurm allocation before the Gate 00F
reference bundle:

```bash
RUN_TAG=p00_gate00f_runtime_preflight_<tag> \
RUNTIME_REGISTRY=/path/to/accepted_candidate_runtime_registry.json \
bash experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh
```

The preflight checks only runtime executability and module specs. Python paths
or container references must come from the accepted runtime registry:

- UniVTAC: `isaacsim`, `isaaclab`, `tacex`, `tacex_uipc`
- TaCauchy: `isaacsim`, `isaaclab`, `tacex`, `tacex_uipc`
- IsaacLab TacSL: `isaacsim`, `isaaclab`, `isaaclab_contrib`

Hard precondition: the runtime registry
`experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json`
must pass
`src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py` before the
module checks run. The registry is authoritative: Python paths are read from
the accepted registry entries, not from default shell paths. If the registry is
missing or not accepted, the preflight writes
`fail_gate00f_runtime_preflight_registry_not_accepted` and exits.

Container runtime note: the registry can record future prebuilt containers.
This preflight supports registered `docker` containers through local
`image_id`, and registered `singularity`/`apptainer`/`sif` containers through
existing `artifact_path`. It runs only module-spec checks inside the container;
it does not run simulation or official demos. Registered `enroot`, `sqsh`, or
`tar` runtimes currently fail as
`unsupported_container_runtime_for_preflight` until an explicit runner is
added.

It does not run simulation, rendering, training, evaluation, model loading, or
dependency installation. Passing this preflight does not clear Gate 00F by
itself; the Gate 00F reference bundle and strict bundle acceptance must still
pass.

Login-node refuse check:
`experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_20260701.md`
records that the script exits with code `2` and refuses to run when
`SLURM_JOB_ID` is missing.
