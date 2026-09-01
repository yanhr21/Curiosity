#!/bin/bash
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
bash "$NT/renders/setup_container.sh" 2>&1 | tail -1
export PATH="$NT/renders/_toolcache/bin:$PATH" HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg
export PYTHONPATH="$CUR:$NT"
source "$NT/renders/render_env.sh"
cd "$NT"
uv run python -m sugar_newton.validation.render_friction --hydroelastic "$@"
echo "render_rc=$?"
kill "${XVFB_PID:-0}" 2>/dev/null
