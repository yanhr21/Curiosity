#!/bin/bash
# allegro_tactile with HARDWARE GL (headless EGL on the A100), not Xvfb + swrast.
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
bash "$NT/renders/setup_container.sh" 2>&1 | tail -1
source "$NT/renders/render_env_egl.sh"
export PYTHONPATH="$CUR:$NT"
cd "$NT"
uv run python -m sugar_newton.validation.allegro_tactile "$@"
echo "allegro_rc=$?"
