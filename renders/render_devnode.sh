#!/bin/bash
# Render a clip on the persistent dev node (renders/devnode.sbatch). Frames only -- ffmpeg is not
# in the container, so assemble the mp4 on the login node afterwards.
#   bash renders/devrun.sh "bash renders/render_devnode.sh g1_meshcoll --frames 120 --cam 2.5,-3.5,4.6 --pitch -40"
set -u
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
SCENE=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/datasets/mike_data/robot_baby_data/_inspect/layout_84b703fb.json
CLIP=${1:-g1_devnode}; shift || true
cd "$REPO"
# NB: render_env.sh sets REPO/OUT/SCENE of its own -- keep the clip name in a variable it does not
# touch, or the --record path silently becomes a nested directory tree.
source renders/render_env.sh
rm -rf "$REPO/renders/$CLIP.mp4.frames"
uv run python example_g1_in_sage.py --scene "$SCENE" --record "$REPO/renders/$CLIP.mp4" "$@"
