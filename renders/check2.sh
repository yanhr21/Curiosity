#!/bin/bash
source /lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton/renders/render_env.sh
echo "===== CHECK2 (foreground spawn + close cam) ====="
uv run python example_g1_in_sage.py --scene "$SCENE" --record "$OUT/g1_check2.mp4" \
    --frames 8 --mu 1.4 --command 1.0,0.0,0.0 --spawn 3.0,0.15 --cam 2.7,-3.0,1.9 --pitch -10
echo "frames=$(ls "$OUT"/g1_check2.mp4.frames/f*.png 2>/dev/null | wc -l)"
kill $XVFB_PID 2>/dev/null
echo "CHECK2_DONE"
