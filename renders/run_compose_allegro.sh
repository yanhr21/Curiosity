#!/bin/bash
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
export PATH="$NT/renders/_toolcache/bin:$PATH" MPLBACKEND=Agg PYTHONPATH="$CUR:$NT"
cd "$NT"
uv run python -m sugar_newton.validation.compose_allegro_video "$@"
echo "compose_rc=$?"
