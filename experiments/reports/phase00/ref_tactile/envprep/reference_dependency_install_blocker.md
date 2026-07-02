# Reference Dependency Install Blocker

Date: 2026-07-01

Current positive state:

- TaCauchy asset file presence is repaired from the approved UniVTAC bundled
  TacEx reuse: `Sensor.usd` is present and tactile test shape USD count is `21`.
- UniVTAC base env exists: `envs/univtac/conda/bin/python`, `Python 3.10.20`.
- TaCauchy base env exists: `envs/tacauchy/conda/bin/python`, `Python 3.11.15`.
- Dry-run official dependency and sanity command files have been generated for
  both reference repos.

Hard blocker:

Official UniVTAC/TaCauchy readiness now requires heavy Isaac/TacEx/UIPC
dependency installation or builds. The project rules forbid heavy work on the
login node and also forbid dependency installation/builds on compute nodes.
Therefore there is no compliant place to execute the heavy official dependency
install steps unless an approved non-login env-prep workflow or prebuilt
shared-filesystem environment is provided.

Remaining effective Gate 00F failures:

- `univtac_official_reference_sanity`
- `tacauchy_official_reference_sanity`

This is not Gate 00F completion and not curiosity readiness.
