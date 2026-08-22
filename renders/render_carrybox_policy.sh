#!/bin/bash
# SUGAR's tracker in the Newton loop, rendered headless on the NVIDIA driver.
#
#   srun --jobid=<JID> --overlap --export=ALL,NVIDIA_DRIVER_CAPABILITIES=all \
#        --container-image=/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh \
#        --container-mounts=/lustre:/lustre \
#        bash -lc "bash renders/render_carrybox_policy.sh <outdir>"
#
# then assemble on the LOGIN node, which has ffmpeg:
#   ffmpeg -framerate 50 -i <outdir>/f%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 18 -r 30 out.mp4
#
# 481 frames of simulation plus rendering costs about 92 s. render_env_egl.sh is what
# keeps that on the GPU: without its ICD file the viewer silently falls back to software.
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
OUT="${1:?usage: render_carrybox_policy.sh <frame-dir> [extra args...]}"; shift || true
bash "$NT/renders/setup_container.sh" 2>&1 | tail -1
source "$NT/renders/render_env_egl.sh"
export PYTHONPATH="$CUR:$NT"
cd "$NT"
uv run python -m sugar_newton.validation.g1_carrybox_policy \
  --frames 481 --substeps 4 --mu 1.0 \
  --cam-offset 1.5 -1.5 0.45 --render "$OUT" --image-format jpg "$@"
echo "render_rc=$?  frames=$(ls "$OUT" | wc -l)"
