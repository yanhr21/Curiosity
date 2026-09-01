#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive_singlenode
#SBATCH --gres=gpu:1
#SBATCH -t 30
#SBATCH --job-name=nw_valtouch
#SBATCH -o /lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton/renders/valtouch_%j.log

# Validate tactile: 60 frames, FULL output (no truncation) so the per-frame peak_touch and the
# final peak robot<->furniture contact force are visible as the robot walks into furniture.
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
IMG=/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh
SCENE=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/datasets/mike_data/robot_baby_data/_inspect/layout_84b703fb.json

srun --container-mounts="$HOME":/home,/lustre:/lustre --container-image "$IMG" \
  bash -lc "cd $REPO && rm -rf renders/g1_val.mp4.frames && bash renders/setup_container.sh 2>&1 | tail -1 && \
    source renders/render_env.sh && \
    uv run python example_g1_in_sage.py --scene $SCENE --record renders/g1_val.mp4 \
      --frames 60 --mu 1.4 --command 1.0,0.0,0.0 --spawn 3.0,0.15 --cam 2.7,-3.0,1.9 --pitch -10 \
      2>&1 | grep -E 'g1_in_sage|peak_touch|contact force|Error|Traceback'"
echo "===== VAL_DONE ====="
