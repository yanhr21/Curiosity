# Reference Env Location Audit

Date: 2026-07-01

This is lightweight login-node file-location evidence only. It did not run
simulation, rendering, training, model loading, package import, dependency
installation, dataset conversion, or Slurm allocation.

Machine-readable audit:
`experiments/configs/phase00/ref_tactile/envprep/reference_env_location_audit_v1.json`

## Result

No approved prebuilt UniVTAC or TaCauchy environment was found.

Missing target executables:

- `envs/univtac/conda/bin/python`
- `envs/univtac/.venv/bin/python`
- `envs/tacauchy/conda/bin/python`
- `envs/tacauchy/.venv/bin/python`

Current shell also does not expose `conda`, `mamba`, `micromamba`, `module`,
`cmake`, `git-lfs`, or `nvcc`.

Project-local env creator candidate:

- `envs/taccel/miniforge/bin/conda`
  - version: `conda 26.3.2`
  - base: `/public/home/yanhongru/Curiosity/envs/taccel/miniforge`
  - status: present, but this does not mean the target UniVTAC/TaCauchy envs
    exist or that heavy construction can run as an untracked shortcut.

## Checked Scope

Project `envs/` contains Python executables for existing curiosity, Newton,
Taccel, T-Rex, and residual-adapter environments, including several failed or
broken historical Newton/Taccel attempts. None are named or approved as
UniVTAC/TaCauchy official reference environments.

Common home conda/env locations checked:

- `/public/home/yanhongru/.conda/envs`
- `/public/home/yanhongru/miniconda3/envs`
- `/public/home/yanhongru/anaconda3/envs`
- `/public/home/yanhongru/miniforge3/envs`
- `/public/home/yanhongru/mambaforge/envs`
- `/public/home/yanhongru/envs`

No UniVTAC/TaCauchy/Isaac/TacEx/UIPC target env was found there.

The only narrow home Python hits outside the project were:

- `/public/home/yanhongru/tmp-autoresearch/.venv/bin/python`
- `/public/home/yanhongru/tmp-ood-autoresearch/.venv/bin/python`

These are not accepted as tactile reference environments.

## Interpretation

Gate 00F remains blocked. The official source trees exist at
`external/UniVTAC` and `external/TaCauchy`, and a project-local Miniforge conda
executable exists, but source trees and an env creator are not enough: the
official sanity checks require approved target environments before any
compute-side sanity run can be meaningful.

Do not install dependencies on compute nodes to bypass this blocker. Do not
start curiosity training while this blocker is open.
