#!/bin/bash
# Run gpu_run.sh in the CUDA container inside the existing allocation.
# No --overlap: it can silently expose only half the GPUs (operations.md).
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
IMG=/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh
srun --container-mounts="$HOME":/home,/lustre:/lustre --container-image "$IMG" \
     bash "$CUR/sugar_newton/gpu_run.sh" 2>&1 | tee "$CUR/sugar_newton/_gpu_run.log"
echo "LAUNCH_EXIT=$?"
