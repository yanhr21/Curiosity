# Gate 00F Readiness

Date: `2026-07-01`

This is a lightweight readiness check only. It does not import packages, install
dependencies, run simulation, run official demos, submit Slurm work, or claim
curiosity readiness.

## Result

- gate00f_ready: `false`
- reason: `blocked_official_sanity_or_gate_review_not_passed`
- latest Gate summary: `/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_d58_marker_v1_20260701_071843/phase00_gate_review_summary.json`
- latest Gate status: `open_not_curiosity_ready`
- latest Gate raw failed checks: `["reference_env_availability","reference_asset_availability","univtac_official_reference_sanity","tacauchy_official_reference_sanity"]`
- effective failed checks after current file-presence audit: `["univtac_official_reference_sanity","tacauchy_official_reference_sanity"]`

## Target Envs

- UniVTAC Python: `/public/home/yanhongru/Curiosity/envs/univtac/conda/bin/python` -> `present`
- TaCauchy Python: `/public/home/yanhongru/Curiosity/envs/tacauchy/conda/bin/python` -> `present`

## Toolchain

- project conda: `/public/home/yanhongru/Curiosity/envs/taccel/miniforge/bin/conda` -> `present`
- project nvcc: `/public/home/yanhongru/Curiosity/envs/taccel/cuda-toolkit/bin/nvcc` -> `present`
- `git-lfs` on PATH: `missing`
- `cmake` on PATH: `missing`
- `nvcc` on PATH: `missing`
- `nvidia-smi` on PATH: `missing`

## Assets

- TaCauchy asset root: `/public/home/yanhongru/Curiosity/external/TaCauchy/source/tacex_assets/tacex_assets/data`, size `412M`
- TaCauchy GelSight Mini Sensor.usd: `present`
- TaCauchy tactile test shape USD count: `21`
- UniVTAC bundled TacEx root: `/public/home/yanhongru/Curiosity/external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data`, size `410M`
- UniVTAC GelSight Mini Sensor.usd: `present`
- UniVTAC tactile test shape USD count: `21`

## Interpretation

Gate 00F remains closed until target UniVTAC/TaCauchy envs exist, required
TaCauchy assets exist, official reference sanity runs pass inside Curiosity
tmux-held Slurm allocation, and a fresh Gate review consumes that evidence.
