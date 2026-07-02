# Reference Environment Blocker Audit

Status: Gate 00F remains blocked. After the user said `全都允许继续`, local
base conda env creation succeeded for both UniVTAC and TaCauchy. Official
dependency installation and official sanity have not been run.

Target envs:
- UniVTAC: `envs/univtac/conda/bin/python`, present, `Python 3.10.20`
- TaCauchy: `envs/tacauchy/conda/bin/python`, present, `Python 3.11.15`

Current stage statuses:
- UniVTAC preflight: `dry_run_preflight_ready`
- UniVTAC create env: `approved_local_no_lock_create_env_succeeded`
- UniVTAC install/sanity: `blocked_official_dependencies_not_installed`
- TaCauchy preflight: `dry_run_preflight_ready`
- TaCauchy create env: `approved_local_no_lock_create_env_succeeded`
- TaCauchy install/sanity: `blocked_official_dependencies_not_installed`

Toolchain visible on current PATH:
- missing: `module`, `cmake`, `git-lfs`, `nvcc`, `nvidia-smi`
- project-local candidate env creator exists:
  `envs/taccel/miniforge/bin/conda` (`conda 26.3.2`)

Why this is blocked:
environment construction for UniVTAC/TaCauchy is a heavy Isaac/TacEx/UIPC
dependency workflow. It cannot be moved to compute nodes. The base Python env
prefixes are now present, but this is not enough for official UniVTAC/TaCauchy
sanity. The successful base env creation is recorded in
`experiments/reports/phase00/ref_tactile/envprep/reference_env_create_execution.md`.

Gate effect:
`check_reference_env_availability.sh` now reports
`gate_00f_ready=candidate_envs_present_pending_compute_sanity`.
`check_gate00f_readiness.sh` still reports `gate00f_ready=false` because
`univtac_official_reference_sanity` and `tacauchy_official_reference_sanity`
remain unpassed.
