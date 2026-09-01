#!/bin/bash
set -u
R=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby
N=$R/Curiosity_newton
bash $N/renders/setup_container.sh >/dev/null 2>&1 || true
export PYTHONPATH=$R/Curiosity:$N:${PYTHONPATH:-}
export WARP_CACHE_PATH=/root/.cache/warp
cd $N
.venv/bin/python $R/Curiosity/sugar_newton/validation/g1_carrybox_policy.py "$@"
echo "POLICY_RC=$?"
