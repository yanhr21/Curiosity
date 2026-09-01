#!/bin/bash
# Channels 9-10 validator on the dev node (hydroelastic is CUDA-only).
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
export PATH="$NT/renders/_toolcache/bin:$PATH" HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg
export PYTHONPATH="$CUR:$NT"
cd "$NT"
uv run python -m sugar_newton.validation.pressure "$@"
echo "pressure_rc=$?"
