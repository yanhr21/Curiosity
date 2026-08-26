#!/usr/bin/env bash
# H200 validation suite. This file refuses to execute outside a Slurm compute step.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: sugar_newton GPU validation must run in a Slurm compute step" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
newton_py=${NEWTON_PY:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}
warp_root=${NEWTON_WARP_ROOT:-/public/home/yanhongru/envs/newton_warp_114}
export PYTHONPATH="$warp_root:$repo_root/third_party/newton:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$repo_root"
mkdir -p sugar_newton/_gpu_out

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$newton_py" -m sugar_newton.validation.friction
"$newton_py" -m sugar_newton.validation.friction --hydroelastic
"$newton_py" -m sugar_newton.validation.incline \
  2>&1 | tee sugar_newton/_gpu_out/incline_h200.log
"$newton_py" -m sugar_newton.validation.pressure
"$newton_py" -m sugar_newton.validation.hand_map \
  --out sugar_newton/_gpu_out/hand_map \
  2>&1 | tee sugar_newton/_gpu_out/hand_map_h200.log
"$newton_py" -m sugar_newton.validation.contact_rewards \
  2>&1 | tee sugar_newton/_gpu_out/contact_rewards_h200.log
