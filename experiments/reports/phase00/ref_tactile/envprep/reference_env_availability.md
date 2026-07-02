# Reference Env Availability

Date: 2026-07-01

Scope: lightweight file and executable checks only. This script does not run
simulation, rendering, training, dependency installation, package import,
official demos, model loading, or dataset conversion.

## Checked Executables

- UniVTAC selected Python: `/public/home/yanhongru/Curiosity/envs/univtac/conda/bin/python`
  - status: `present`
- UniVTAC conda Python: `/public/home/yanhongru/Curiosity/envs/univtac/conda/bin/python`
  - status: `present`
- UniVTAC venv Python: `/public/home/yanhongru/Curiosity/envs/univtac/.venv/bin/python`
  - status: `missing`
- TaCauchy selected Python: `/public/home/yanhongru/Curiosity/envs/tacauchy/conda/bin/python`
  - status: `present`
- TaCauchy conda Python: `/public/home/yanhongru/Curiosity/envs/tacauchy/conda/bin/python`
  - status: `present`
- TaCauchy venv Python: `/public/home/yanhongru/Curiosity/envs/tacauchy/.venv/bin/python`
  - status: `missing`

## Toolchain On PATH

- `git-lfs`: `missing`
- `cmake`: `missing`
- `nvcc`: `missing`

## Interpretation

Gate 00F is not ready unless both official reference Python executables are
present and later pass their compute-side official sanity probes. This
availability check is only a preflight guard; it is not official reference
sanity and not curiosity progress.
