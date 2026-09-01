#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive_singlenode
#SBATCH --gres=gpu:1
#SBATCH -t 30
#SBATCH --job-name=nw_smokesdf
#SBATCH -o /lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton/renders/smokesdf_%j.log

# Smoke test: build 59 hydroelastic SDF objects + render 8 frames, full output, to validate the
# SDF build succeeds and measure feasibility (build + per-frame time) before a full render.
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
IMG=/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh
SCENE=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/datasets/mike_data/robot_baby_data/_inspect/layout_84b703fb.json

srun --container-mounts="$HOME":/home,/lustre:/lustre --container-image "$IMG" \
  bash -lc "cd $REPO && rm -rf renders/g1_smoke.mp4.frames && bash renders/setup_container.sh 2>&1 | tail -2 && \
    source renders/render_env.sh && echo BUILD_START_\$(date +%s) && \
    uv run python example_g1_in_sage.py --scene $SCENE --record renders/g1_smoke.mp4 \
      --frames 8 --mu 1.4 --command 1.0,0.0,0.0 --spawn 3.0,0.15 --cam 2.7,-3.0,1.9 --pitch -10 ; \
    echo RENDER_END_\$(date +%s)"
echo "smoke_frames=$(ls $REPO/renders/g1_smoke.mp4.frames/f*.png 2>/dev/null | wc -l)"
echo "===== SMOKE_SDF_DONE ====="
