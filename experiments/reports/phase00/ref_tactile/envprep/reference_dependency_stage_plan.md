# Reference Dependency Stage Plan

Date: 2026-07-01

This records dry-run command staging only. No UniVTAC/TaCauchy/Isaac/TacEx/UIPC
dependency installation was executed, and no official sanity was run.

## Base Env Status

- UniVTAC: `envs/univtac/conda/bin/python`, `Python 3.10.20`
- TaCauchy: `envs/tacauchy/conda/bin/python`, `Python 3.11.15`

## Dry-Run Stages Generated

- UniVTAC: `install_isaac`, `install_isaaclab`, `install_curobo_or_assets`,
  `install_tacex_core`, `build_uipc`, `setup_assets`, `official_sanity`
- TaCauchy: `install_isaac`, `install_isaaclab`, `install_curobo_or_assets`,
  `install_tacex_core`, `build_uipc`, `setup_assets`, `official_sanity`

Evidence roots:

- `experiments/outputs/phase00/ref_tactile/envprep/univtac/`
- `experiments/reports/phase00/ref_tactile/envprep/univtac/`
- `experiments/outputs/phase00/ref_tactile/envprep/tacauchy/`
- `experiments/reports/phase00/ref_tactile/envprep/tacauchy/`

## Gate Effect

The effective remaining Gate 00F failures are
`univtac_official_reference_sanity` and `tacauchy_official_reference_sanity`.
The dry-run commands do not count as dependency installation, official sanity,
Gate 00F completion, or curiosity readiness.
