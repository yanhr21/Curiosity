#!/bin/bash
# Run any sugar_newton module inside the dev container.
#   bash renders/run_sn.sh sugar_newton.validation.allegro_grasp_sweep --frames 200
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
MOD="$1"; shift
bash "$NT/renders/setup_container.sh" 2>&1 | tail -1
export PATH="$NT/renders/_toolcache/bin:$PATH" HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg
export PYTHONPATH="$CUR:$NT"
source "$NT/renders/render_env.sh"
cd "$NT"
uv run python -m "$MOD" "$@"
echo "sn_rc=$?"
kill "${XVFB_PID:-0}" 2>/dev/null
