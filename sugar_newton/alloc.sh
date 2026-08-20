#!/bin/bash
# Persistent interactive GPU allocation for Plan 16 Newton work.
# Held in screen session `newton_gpu`. interactive_singlenode has 1243 nodes
# (vs 10 on `interactive`) and is what actually schedules; hard 4 h cap.
cd /lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
exec salloc -A nvr_nxp_visionconferencing -p interactive_singlenode \
     --gres=gpu:1 -N 1 -t 04:00:00 -J newton_gpu
