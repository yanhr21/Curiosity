#!/usr/bin/env bash
# Stitch metal | wood | rubber(soft) tactile composites side-by-side, each with a header
# bar labeling the material and its SIMULATION fps (rigid hydroelastic vs soft FEM/VBD).
set -euo pipefail
cd "$(dirname "$0")"

F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
OUT=${1:-tactile_compare_metal_wood_softrubber.mp4}

# sim fps measured earlier: rigid (MuJoCo + hydroelastic, CUDA graph) ~82 fps;
# soft (MuJoCo arm + SolverVBD 50 it, eager) ~5 fps.
ffmpeg -y \
  -i tactile_material_metal.mp4 \
  -i tactile_material_wood.mp4 \
  -i tactile_material_rubber_soft.mp4 \
  -filter_complex "\
[0:v]pad=iw:ih+72:0:72:black,drawtext=fontfile=${F}:text='metal   rigid   82 fps':x=(w-tw)/2:y=20:fontsize=40:fontcolor=white[a];\
[1:v]pad=iw:ih+72:0:72:black,drawtext=fontfile=${F}:text='wood   rigid   82 fps':x=(w-tw)/2:y=20:fontsize=40:fontcolor=white[b];\
[2:v]pad=iw:ih+72:0:72:black,drawtext=fontfile=${F}:text='rubber   soft FEM   5 fps':x=(w-tw)/2:y=20:fontsize=40:fontcolor=yellow[c];\
[a][b][c]hstack=inputs=3[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 20 "${OUT}"
echo "wrote ${OUT}"
