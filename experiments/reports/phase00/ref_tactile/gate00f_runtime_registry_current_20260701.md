# Gate 00F Runtime Registry Current Validation

- Date: `2026-07-01`
- Classification: `runtime_registry_validation_not_training_not_gate_completion`
- Registry:
  `experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json`
- Summary:
  `experiments/outputs/phase00/ref_tactile/runtime_registry/p00_registry_current_20260701/gate00f_runtime_registry_validation_summary.json`
- Status: `fail_gate00f_runtime_registry`

This validation only checked JSON metadata and filesystem path properties. It
did not import Isaac/TacEx/TaCauchy/UniVTAC/TacSL packages, run simulation,
render, train, evaluate, load models, convert data, install dependencies, or
build packages.

## Findings

- `univtac`: Python exists at
  `/public/home/yanhongru/Curiosity/envs/univtac/conda/bin/python`, but the
  registry status is `base_python_only_not_dependency_complete`, not
  `dependency_complete_registered`.
- `tacauchy`: Python exists at
  `/public/home/yanhongru/Curiosity/envs/tacauchy/conda/bin/python`, but the
  registry status is `base_python_only_not_dependency_complete`, not
  `dependency_complete_registered`.
- `isaaclab_tacsl`: no registered runtime path exists.
- Validator now supports strict container registration metadata, but the
  current registry still has no accepted container runtime. A remote
  `image_ref` alone would not be enough; the registry requires a local
  `image_id` or existing shared `artifact_path`.
- Validator now also checks provenance paths. Current provenance paths for all
  three target entries exist; the current failure is still due to unresolved
  dependency-complete runtimes, not missing provenance files.

## Gate Effect

This does not clear Gate 00F. Runtime preflight and the Gate 00F bundle should
not be treated as ready until all three targets are registered as
`dependency_complete_registered` through an allowed resolution path.
