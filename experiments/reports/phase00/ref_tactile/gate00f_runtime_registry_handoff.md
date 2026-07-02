# Gate 00F Runtime Registry Handoff

- Date: `2026-07-01`
- Classification: `handoff_not_training_not_gate_completion`
- Registry:
  `experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json`
- Validator:
  `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`

Before running the runtime preflight or Gate 00F bundle, register the exact
official reference runtimes here and validate them:

```bash
python src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py \
  --registry experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json \
  --output-json experiments/outputs/phase00/ref_tactile/runtime_registry/<RUN_TAG>/gate00f_runtime_registry_validation_summary.json
```

Each target must be registered:

- `univtac`
- `tacauchy`
- `isaaclab_tacsl`

Each target must have:

- `kind`: `python_env` or `container`
- `status`: `dependency_complete_registered`
- `resolution_path`: one of the allowed resolution paths from the dependency
  resolution packet
- nonempty `expected_modules`
- nonempty provenance
- no path inside excluded non-Curiosity resource zones

For `python_env`, `path` must be a nonempty executable Python path.

For `container`, the registry must include:

- `container_runtime`: `docker`, `singularity`, `apptainer`, `enroot`, `sif`,
  `sqsh`, or `tar`
- `artifact_path` for a shared-filesystem image/archive, or local `image_id`
  for a Docker-style local image
- optional `image_ref` as source/provenance only

Remote `image_ref` alone, such as an NGC tag, is only acquisition evidence. It
is not a registered dependency-complete runtime.

Passing this registry validation is required before runtime preflight, but it
does not clear Gate 00F by itself. Gate 00F still requires runtime preflight,
official reference bundle execution, and strict bundle acceptance.
