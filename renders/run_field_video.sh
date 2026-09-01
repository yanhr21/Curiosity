#!/bin/bash
# Compose the continuous tactile-field frames and assemble the mp4.
# Both steps are CPU-only (numpy + matplotlib + ffmpeg), so this runs on the LOGIN node --
# the dev container has no ffmpeg, and the composer needs no GPU.
#   bash renders/run_field_video.sh <run-dir> [extra compose args...]
set -eu
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
RUN="$1"; shift || true
cd "$CUR"
PYTHONPATH="$CUR" MPLBACKEND=Agg python3 -m sugar_newton.validation.compose_allegro_field \
    --run "$RUN" "$@"
ffmpeg -y -loglevel warning -framerate 30 -i "$RUN/field/g%05d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$RUN/allegro_field.mp4"
ls -l "$RUN/allegro_field.mp4"
