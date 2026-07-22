#!/bin/bash
# Two versions: rough (mu 1.4) and slippery (mu 0.05) — a global friction override on floor +
# feet + all furniture ("slippery everything" vs "rough everything"). Rough is rendered first
# (the hero clip: robot walks into the furniture) in case the node's time limit is tight.
# Frames only (ffmpeg assembled on the login node). Arg1 = frames.
FRAMES=${1:-150}
source /lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton/renders/render_env.sh
CAM="2.7,-3.0,1.9"; SPAWN="3.0,0.15"; PITCH=-10

echo "===== MARK_ROUGH (mu 1.4) ====="
uv run python example_g1_in_sage.py --scene "$SCENE" --record "$OUT/g1_rough.mp4" \
    --frames "$FRAMES" --mu 1.4 --command 1.0,0.0,0.0 --spawn "$SPAWN" --cam "$CAM" --pitch "$PITCH" 2>&1 | tail -8
echo "rough_frames=$(ls "$OUT"/g1_rough.mp4.frames/f*.png 2>/dev/null | wc -l)"

echo "===== MARK_SLIPPERY (mu 0.05) ====="
uv run python example_g1_in_sage.py --scene "$SCENE" --record "$OUT/g1_slippery.mp4" \
    --frames "$FRAMES" --mu 0.05 --command 1.0,0.0,0.0 --spawn "$SPAWN" --cam "$CAM" --pitch "$PITCH" 2>&1 | tail -8
echo "slippery_frames=$(ls "$OUT"/g1_slippery.mp4.frames/f*.png 2>/dev/null | wc -l)"

kill $XVFB_PID 2>/dev/null
echo "===== MARK_TWO_DONE ====="
